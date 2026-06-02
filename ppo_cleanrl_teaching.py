"""
A small, teaching-oriented PPO script inspired by CleanRL's ppo.py.

Goal:
    - Show the core PPO / Actor-Critic / GAE mechanics in one readable file.
    - Include small unit tests for each important submodule.
    - Run on a laptop CPU, including MacBook Pro M1.

Install:
    python3 -m venv .venv
    source .venv/bin/activate
    pip install torch gymnasium numpy

Run tests only:
    python ppo_cleanrl_teaching.py --mode test

Run tests + short CartPole training:
    python ppo_cleanrl_teaching.py

Run longer training:
    python ppo_cleanrl_teaching.py --mode train --total-timesteps 20000

This is intentionally simpler than CleanRL:
    - no wandb
    - no tensorboard
    - no video recording
    - no learning-rate annealing
    - no value-loss clipping
    - no target KL early stop

But it keeps the conceptual core:
    - separate Actor and Critic MLPs
    - rollout storage
    - GAE advantage calculation
    - PPO clipped policy loss
    - value loss
    - entropy bonus
    - gradient clipping
"""

from __future__ import annotations

import argparse
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

# Keep CPU execution predictable and laptop-friendly.
# This also avoids excessive thread usage in small teaching tests.
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


# -----------------------------------------------------------------------------
# 1. Configuration
# -----------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Small set of PPO hyperparameters.

    The command-line interface only exposes a few fields. The rest are kept here
    so the script is easy to read and modify.
    """

    env_id: str = "CartPole-v1"
    seed: int = 1
    total_timesteps: int = 10_000

    # Rollout shape: batch_size = num_envs * num_steps
    num_envs: int = 4
    num_steps: int = 64

    # Optimization
    learning_rate: float = 2.5e-4
    update_epochs: int = 4
    num_minibatches: int = 4

    # RL math
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.20
    ent_coef: float = 0.01
    vf_coef: float = 0.50
    max_grad_norm: float = 0.50

    # Network
    hidden_size: int = 64

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches

    @property
    def num_iterations(self) -> int:
        return max(1, self.total_timesteps // self.batch_size)


# -----------------------------------------------------------------------------
# 2. Reproducibility and small utilities
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal initialization, matching the style used in CleanRL.

    Input:
        layer: nn.Linear
        std: scale factor for orthogonal weight initialization
        bias_const: constant value for bias

    Output:
        the same layer, initialized in-place
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# -----------------------------------------------------------------------------
# 3. Actor-Critic neural network
# -----------------------------------------------------------------------------


class Agent(nn.Module):
    """A minimal Actor-Critic agent for discrete-action environments.

    Input:
        obs: float tensor of shape [batch, obs_dim]

    Outputs from get_action_and_value:
        action:   long tensor  [batch]
        logprob:  float tensor [batch]
        entropy:  float tensor [batch]
        value:    float tensor [batch]

    Conceptual mapping:
        actor(obs)  -> logits -> Categorical distribution -> pi_theta(a|s)
        critic(obs) -> scalar value                         -> V_phi(s)
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 64):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            # Small std makes initial policy close to uniform.
            layer_init(nn.Linear(hidden_size, action_dim), std=0.01),
        )

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Return V_phi(s) with shape [batch]."""
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample or evaluate an action.

        If action is None:
            sample an action from pi_theta(.|s)
        If action is given:
            evaluate logprob and entropy for that action
        """
        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        print(f"dist Categorical: {dist}")
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.get_value(obs)
        return action, logprob, entropy, value


# -----------------------------------------------------------------------------
# 4. GAE: Generalized Advantage Estimation
# -----------------------------------------------------------------------------


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and value targets.

    Shapes:
        rewards:    [T, N]
        values:     [T, N]
        dones:      [T, N], where dones[t] means episode ended after step t
        next_value: [N], value of the observation after the rollout

    Returns:
        advantages: [T, N]
        returns:    [T, N], equal to advantages + values

    Math:
        delta_t = r_{t+1} + gamma * V(s_{t+1}) * nonterminal - V(s_t)
        A_t     = delta_t + gamma * lambda * nonterminal * A_{t+1}
        R_t     = A_t + V(s_t)
    """
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(N, device=rewards.device)

    for t in reversed(range(T)):
        if t == T - 1:
            next_values = next_value
        else:
            next_values = values[t + 1]

        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values * nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


# -----------------------------------------------------------------------------
# 5. PPO loss
# -----------------------------------------------------------------------------


def compute_ppo_loss(
    new_logprob: torch.Tensor,
    old_logprob: torch.Tensor,
    advantages: torch.Tensor,
    new_value: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor,
    clip_coef: float,
    ent_coef: float,
    vf_coef: float,
) -> dict[str, torch.Tensor]:
    """Compute PPO clipped objective and critic loss.

    Inputs all have shape [batch].

    Important quantities:
        ratio = pi_new(a|s) / pi_old(a|s)
              = exp(log pi_new(a|s) - log pi_old(a|s))

        policy loss = - min(ratio * A, clipped_ratio * A)
        In code we use max of the negative versions, matching common PPO code.
    """
    logratio = new_logprob - old_logprob
    ratio = logratio.exp()

    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
    policy_loss = torch.max(pg_loss1, pg_loss2).mean()

    value_loss = 0.5 * ((new_value - returns) ** 2).mean()
    entropy_loss = entropy.mean()

    total_loss = policy_loss - ent_coef * entropy_loss + vf_coef * value_loss

    with torch.no_grad():
        approx_kl = ((ratio - 1.0) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean()

    return {
        "loss": total_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy_loss,
        "approx_kl": approx_kl,
        "clipfrac": clipfrac,
        "ratio": ratio,
    }


# -----------------------------------------------------------------------------
# 6. Environment and rollout collection
# -----------------------------------------------------------------------------


def make_envs(env_id: str, num_envs: int, seed: int):
    """Create vectorized Gymnasium environments.

    This function imports gymnasium lazily so unit tests that do not need an
    environment can run even if gymnasium is not installed.
    """
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise ImportError(
            "gymnasium is required for training. Install with: pip install gymnasium"
        ) from exc

    def make_one_env(rank: int):
        def thunk():
            env = gym.make(env_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.action_space.seed(seed + rank)
            env.observation_space.seed(seed + rank)
            return env

        return thunk

    return gym.vector.SyncVectorEnv([make_one_env(i) for i in range(num_envs)])


@dataclass
class RolloutBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    next_obs: torch.Tensor



def collect_rollout(
    envs,
    agent: Agent,
    cfg: TrainConfig,
    device: torch.device,
    next_obs: torch.Tensor,
    episode_returns: np.ndarray,
    recent_returns: deque,
) -> RolloutBatch:
    """Collect one PPO rollout.

    Storage shapes:
        obs:      [T, N, obs_dim]
        actions:  [T, N]
        logprobs: [T, N]
        rewards:  [T, N]
        dones:    [T, N], done after executing action
        values:   [T, N]
    """
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    T, N = cfg.num_steps, cfg.num_envs

    obs_buf = torch.zeros((T, N, obs_dim), device=device)
    actions_buf = torch.zeros((T, N), dtype=torch.long, device=device)
    logprobs_buf = torch.zeros((T, N), device=device)
    rewards_buf = torch.zeros((T, N), device=device)
    dones_buf = torch.zeros((T, N), device=device)
    values_buf = torch.zeros((T, N), device=device)

    for t in range(T):
        obs_buf[t] = next_obs

        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)

        actions_buf[t] = action
        logprobs_buf[t] = logprob
        values_buf[t] = value

        next_obs_np, reward_np, terminated_np, truncated_np, _ = envs.step(action.cpu().numpy())
        done_np = np.logical_or(terminated_np, truncated_np)

        episode_returns += reward_np
        for env_i, done in enumerate(done_np):
            if done:
                recent_returns.append(float(episode_returns[env_i]))
                episode_returns[env_i] = 0.0

        rewards_buf[t] = torch.as_tensor(reward_np, dtype=torch.float32, device=device)
        dones_buf[t] = torch.as_tensor(done_np, dtype=torch.float32, device=device)
        next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device).reshape(N, obs_dim)

    return RolloutBatch(
        obs=obs_buf,
        actions=actions_buf,
        logprobs=logprobs_buf,
        rewards=rewards_buf,
        dones=dones_buf,
        values=values_buf,
        next_obs=next_obs,
    )


# -----------------------------------------------------------------------------
# 7. Training loop
# -----------------------------------------------------------------------------


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    device = torch.device("cpu")

    envs = make_envs(cfg.env_id, cfg.num_envs, cfg.seed)
    assert envs.single_action_space.__class__.__name__ == "Discrete", "This teaching script supports discrete actions only."

    obs_dim = int(np.prod(envs.single_observation_space.shape))
    action_dim = int(envs.single_action_space.n)

    agent = Agent(obs_dim, action_dim, cfg.hidden_size).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    next_obs_np, _ = envs.reset(seed=cfg.seed)
    next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device).reshape(cfg.num_envs, obs_dim)

    episode_returns = np.zeros(cfg.num_envs, dtype=np.float32)
    recent_returns: deque[float] = deque(maxlen=20)

    global_step = 0
    start_time = time.time()

    print(f"Training PPO on {cfg.env_id}")
    print(f"obs_dim={obs_dim}, action_dim={action_dim}, batch_size={cfg.batch_size}, minibatch_size={cfg.minibatch_size}")

    for iteration in range(1, cfg.num_iterations + 1):
        rollout = collect_rollout(envs, agent, cfg, device, next_obs, episode_returns, recent_returns)
        next_obs = rollout.next_obs
        global_step += cfg.batch_size

        with torch.no_grad():
            next_value = agent.get_value(next_obs)
            advantages, returns = compute_gae(
                rewards=rollout.rewards,
                values=rollout.values,
                dones=rollout.dones,
                next_value=next_value,
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
            )

        # Flatten [T, N, ...] into [T*N, ...]
        b_obs = rollout.obs.reshape(cfg.batch_size, obs_dim)
        b_actions = rollout.actions.reshape(cfg.batch_size)
        b_logprobs = rollout.logprobs.reshape(cfg.batch_size)
        b_advantages = advantages.reshape(cfg.batch_size)
        b_returns = returns.reshape(cfg.batch_size)

        indices = np.arange(cfg.batch_size)
        last_losses: Optional[dict[str, torch.Tensor]] = None

        for _epoch in range(cfg.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, cfg.batch_size, cfg.minibatch_size):
                mb_idx = torch.as_tensor(indices[start : start + cfg.minibatch_size], dtype=torch.long, device=device)

                _, new_logprob, entropy, new_value = agent.get_action_and_value(b_obs[mb_idx], b_actions[mb_idx])

                mb_advantages = b_advantages[mb_idx]
                # Advantage normalization is a common PPO implementation detail.
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std(unbiased=False) + 1e-8)

                losses = compute_ppo_loss(
                    new_logprob=new_logprob,
                    old_logprob=b_logprobs[mb_idx],
                    advantages=mb_advantages,
                    new_value=new_value,
                    returns=b_returns[mb_idx],
                    entropy=entropy,
                    clip_coef=cfg.clip_coef,
                    ent_coef=cfg.ent_coef,
                    vf_coef=cfg.vf_coef,
                )

                optimizer.zero_grad()
                losses["loss"].backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()
                last_losses = losses

        if iteration == 1 or iteration % 5 == 0 or iteration == cfg.num_iterations:
            mean_return = float(np.mean(recent_returns)) if recent_returns else float("nan")
            sps = int(global_step / max(1e-8, time.time() - start_time))
            if last_losses is not None:
                print(
                    f"iter={iteration:03d} step={global_step:06d} "
                    f"mean_return_20={mean_return:7.2f} "
                    f"policy_loss={last_losses['policy_loss'].item(): .4f} "
                    f"value_loss={last_losses['value_loss'].item(): .4f} "
                    f"entropy={last_losses['entropy'].item(): .4f} "
                    f"clipfrac={last_losses['clipfrac'].item(): .2f} "
                    f"sps={sps}"
                )

    envs.close()
    print("Done. For CartPole-v1, mean_return_20 moving toward 200+ means it is learning.")


# -----------------------------------------------------------------------------
# 8. Unit tests
# -----------------------------------------------------------------------------


def assert_close(actual, expected, atol=1e-5, name="value"):
    actual_t = torch.as_tensor(actual, dtype=torch.float32)
    expected_t = torch.as_tensor(expected, dtype=torch.float32)
    if not torch.allclose(actual_t, expected_t, atol=atol, rtol=0):
        raise AssertionError(f"{name} mismatch. actual={actual_t}, expected={expected_t}")



def test_layer_init() -> None:
    layer = layer_init(nn.Linear(3, 2), std=0.01, bias_const=0.5)
    assert layer.weight.shape == (2, 3)
    assert layer.bias.shape == (2,)
    assert_close(layer.bias, torch.full((2,), 0.5), name="layer bias")



def test_agent_shapes() -> None:
    set_seed(0)
    agent = Agent(obs_dim=4, action_dim=2, hidden_size=16)
    obs = torch.randn(5, 4)

    action, logprob, entropy, value = agent.get_action_and_value(obs)
    assert action.shape == (5,)
    assert logprob.shape == (5,)
    assert entropy.shape == (5,)
    assert value.shape == (5,)
    assert action.dtype == torch.long

    fixed_action = torch.tensor([0, 1, 0, 1, 1], dtype=torch.long)
    action2, logprob2, entropy2, value2 = agent.get_action_and_value(obs, fixed_action)
    assert_close(action2, fixed_action, name="fixed action")
    assert logprob2.shape == (5,)
    assert entropy2.shape == (5,)
    assert value2.shape == (5,)



def test_gae_known_values() -> None:
    # Same example as the lecture:
    # rewards = [1, 2, 3]
    # values  = [4, 5, 2, 0]
    # gamma=0.9, lambda=0.8
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.tensor([[4.0], [5.0], [2.0]])
    dones = torch.tensor([[0.0], [0.0], [1.0]])
    next_value = torch.tensor([0.0])

    advantages, returns = compute_gae(rewards, values, dones, next_value, gamma=0.9, gae_lambda=0.8)

    expected_advantages = torch.tensor([[1.1544], [-0.48], [1.0]])
    expected_returns = expected_advantages + values

    assert_close(advantages, expected_advantages, atol=1e-4, name="advantages")
    assert_close(returns, expected_returns, atol=1e-4, name="returns")



def test_ppo_loss_clipping() -> None:
    old_logprob = torch.log(torch.tensor([0.5, 0.5]))
    new_logprob = torch.log(torch.tensor([0.8, 0.3]))
    advantages = torch.tensor([1.0, -1.0])
    new_value = torch.tensor([2.0, 0.0])
    returns = torch.tensor([1.0, 1.0])
    entropy = torch.tensor([0.5, 0.5])

    losses = compute_ppo_loss(
        new_logprob=new_logprob,
        old_logprob=old_logprob,
        advantages=advantages,
        new_value=new_value,
        returns=returns,
        entropy=entropy,
        clip_coef=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
    )

    # ratio = [1.6, 0.6]
    # positive advantage: max(-1.6, -1.2) = -1.2
    # negative advantage: max(0.6, 0.8) = 0.8
    # policy loss mean = (-1.2 + 0.8) / 2 = -0.2
    assert_close(losses["policy_loss"], -0.2, atol=1e-5, name="policy_loss")
    assert torch.isfinite(losses["loss"])
    assert losses["clipfrac"].item() == 1.0



def test_one_gradient_update() -> None:
    set_seed(0)
    agent = Agent(obs_dim=4, action_dim=2, hidden_size=16)
    optimizer = optim.Adam(agent.parameters(), lr=1e-3)

    obs = torch.randn(8, 4)
    with torch.no_grad():
        actions, old_logprob, _, _ = agent.get_action_and_value(obs)

    advantages = torch.randn(8)
    returns = torch.randn(8)

    before = [p.detach().clone() for p in agent.parameters()]
    _, new_logprob, entropy, new_value = agent.get_action_and_value(obs, actions)

    losses = compute_ppo_loss(
        new_logprob=new_logprob,
        old_logprob=old_logprob,
        advantages=advantages,
        new_value=new_value,
        returns=returns,
        entropy=entropy,
        clip_coef=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
    )

    optimizer.zero_grad()
    losses["loss"].backward()
    optimizer.step()

    after = list(agent.parameters())
    changed = any(not torch.allclose(b, a.detach()) for b, a in zip(before, after))
    assert changed, "parameters did not change after one PPO update"



def test_rollout_shapes_if_gymnasium_available() -> None:
    try:
        import gymnasium  # noqa: F401
    except ImportError:
        print("[SKIP] rollout shape test: gymnasium is not installed")
        return

    cfg = TrainConfig(total_timesteps=16, num_envs=2, num_steps=4, hidden_size=16)
    set_seed(cfg.seed)
    device = torch.device("cpu")
    envs = make_envs(cfg.env_id, cfg.num_envs, cfg.seed)

    obs_dim = int(np.prod(envs.single_observation_space.shape))
    action_dim = int(envs.single_action_space.n)
    agent = Agent(obs_dim, action_dim, cfg.hidden_size)

    next_obs_np, _ = envs.reset(seed=cfg.seed)
    next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32).reshape(cfg.num_envs, obs_dim)
    episode_returns = np.zeros(cfg.num_envs, dtype=np.float32)
    recent_returns: deque[float] = deque(maxlen=20)

    rollout = collect_rollout(envs, agent, cfg, device, next_obs, episode_returns, recent_returns)
    envs.close()

    assert rollout.obs.shape == (cfg.num_steps, cfg.num_envs, obs_dim)
    assert rollout.actions.shape == (cfg.num_steps, cfg.num_envs)
    assert rollout.logprobs.shape == (cfg.num_steps, cfg.num_envs)
    assert rollout.rewards.shape == (cfg.num_steps, cfg.num_envs)
    assert rollout.dones.shape == (cfg.num_steps, cfg.num_envs)
    assert rollout.values.shape == (cfg.num_steps, cfg.num_envs)
    assert rollout.next_obs.shape == (cfg.num_envs, obs_dim)



def run_tests() -> None:
    tests = [
        test_layer_init,
        test_agent_shapes,
        test_gae_known_values,
        test_ppo_loss_clipping,
        test_one_gradient_update,
        test_rollout_shapes_if_gymnasium_available,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print("All tests passed.")


# -----------------------------------------------------------------------------
# 9. CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teaching PPO: CleanRL-style Actor-Critic + GAE")
    parser.add_argument("--mode", choices=["test", "train", "all"], default="all")
    parser.add_argument("--env-id", type=str, default="CartPole-v1")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=10_000)
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    cfg = TrainConfig(env_id=args.env_id, seed=args.seed, total_timesteps=args.total_timesteps)

    if args.mode in ("test", "all"):
        run_tests()

    if args.mode in ("train", "all"):
        train(cfg)


if __name__ == "__main__":
    test_layer_init()
