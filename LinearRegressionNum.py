import numpy as np

np.random.seed(42)


class LinearRegression:
    # 4.1 初始化 w 和 b
    def __init__(self, input_dim, lr=0.0001):
        # W shape: (d, 1)，b 是标量
        self.W = np.random.randn(input_dim, 1)
        self.b = 0.0
        self.lr = lr

    # 4.2 根据 X, w, b 计算预测值 y_hat
    def forward(self, X):
        # X shape: (N, d)
        # W shape: (d, 1)
        # y_hat shape: (N, 1)
        y_hat = X @ self.W + self.b
        return y_hat

    # 4.3 计算 MSE Loss
    def compute_loss(self, y_hat, y):
        N = y.shape[0]
        loss = 1 / N * np.sum((y_hat - y) ** 2)
        return loss

    # 4.4 反向传播，求 dW 和 db
    def compute_gradient(self, X, y_hat, y):
        N = y_hat.shape[0]
        error = y_hat - y          # shape: (N, 1)
        dw = 2 / N * X.T @ error   # shape: (d, 1)
        db = 2 / N * np.sum(error) # 标量
        return dw, db

    # 4.5 梯度下降，更新参数
    def update_parameters(self, dw, db):
        self.W = self.W - self.lr * dw
        self.b = self.b - self.lr * db

    # 4.6 重复很多轮
    def fit(self, X, y, epochs=100):
        for epoch in range(epochs):
            # 前向传播
            y_hat = self.forward(X)

            # 计算损失
            loss = self.compute_loss(y_hat, y)

            # 反向传播
            dw, db = self.compute_gradient(X, y_hat, y)

            # 梯度更新
            self.update_parameters(dw, db)

            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}, Loss: {loss:.6f}")

    def predict(self, X):
        return self.forward(X)


# ────────────────────────────────────────────────
# 造数据：真实关系 y = 2*x1 + (-3)*x2 + 5 + 噪声
# ────────────────────────────────────────────────
N, d = 200, 8                                              # 200 个样本，8 个特征
X = np.random.randn(N, d)                                  # shape: (200, 8)
true_W = np.array([[2.0], [-3.0], [1.5], [0.8],
                   [-1.2], [3.0], [-0.5], [2.5]])          # shape: (8, 1)
true_b = 5.0
noise = 0.1 * np.random.randn(N, 1)      # 加一点噪声
y = X @ true_W + true_b + noise          # shape: (100, 1)

print("=" * 45)
print(f"真实 W: {true_W.ravel()},  真实 b: {true_b}")
print("=" * 45)

# ────────────────────────────────────────────────
# 训练
# ────────────────────────────────────────────────
model = LinearRegression(input_dim=d, lr=0.05)
model.fit(X, y, epochs=100)

# ────────────────────────────────────────────────
# 结果对比
# ────────────────────────────────────────────────
print("=" * 45)
print(f"学到的 W: {model.W.ravel()}")
print(f"学到的 b: {model.b:.4f}")
print("=" * 45)
