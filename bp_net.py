import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ====== 可修改参数区 ======
# 我已预设了一组通常能达到 96%-98% 准确率的参数
HIDDEN_UNITS = 64     # 隐藏层神经元个数 (建议尝试 32, 64, 128)
LEARNING_RATE = 0.001 # 学习率 (建议尝试 0.01, 0.001, 0.0005)
# ==========================


# =============================
# 1. 加载真实乳腺癌数据
# =============================
data = load_breast_cancer()
X = data.data           # (569, 30)
y = data.target         # 0/1

# 划分训练集 / 测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# 标准化（现实数据必须）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# 转为 Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test  = torch.tensor(X_test, dtype=torch.float32)
y_test  = torch.tensor(y_test, dtype=torch.long)

# =============================
# 2. 最简 BP 神经网络
# =============================
class BPNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 【修改点】此处现在读取顶部的 HIDDEN_UNITS 变量
        self.fc1 = nn.Linear(30, HIDDEN_UNITS)
        self.relu = nn.ReLU()
        # 【修改点】此处现在读取顶部的 HIDDEN_UNITS 变量
        self.fc2 = nn.Linear(HIDDEN_UNITS, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = BPNet()

# =============================
# 3. 损失函数 & 优化器
# =============================
criterion = nn.CrossEntropyLoss()

# 【修改点】此处现在读取顶部的 LEARNING_RATE 变量
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# =============================
# 4. 训练（BP 发生在这里）
# =============================
epochs = 200
print(f"开始训练: 隐藏层={HIDDEN_UNITS}, 学习率={LEARNING_RATE}")
print("-" * 30)

for epoch in range(1, epochs + 1):
    model.train()

    logits = model(X_train)          # 前向传播
    loss = criterion(logits, y_train)

    optimizer.zero_grad()
    loss.backward()                  # 反向传播（BP）
    optimizer.step()                 # 参数更新

    if epoch % 20 == 0:
        pred = logits.argmax(dim=1)
        acc = (pred == y_train).float().mean().item()
        print(f"epoch={epoch:3d} loss={loss.item():.4f} acc={acc:.4f}")

# =============================
# 5. 测试
# =============================
model.eval()
with torch.no_grad():
    logits = model(X_test)
    pred = logits.argmax(dim=1)
    test_acc = (pred == y_test).float().mean().item()

print("-" * 30)
print(f"最终结果 (Test Accuracy) = {test_acc:.4f}")
print("-" * 30)
