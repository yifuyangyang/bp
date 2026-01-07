
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
HIDDEN_UNITS = 32
LEARNING_RATE = 0.01

hidden_list = [8, 16, 32, 64, 128]
lr_list = [0.1, 0.01, 0.005, 0.001]

# N_RUNS = 10          # 连续测试 n 次
# EPOCHS = 200         # 训练轮数（不想动就固定）
# DEVICE = "cpu"       # 没有GPU就cpu
best_acc = -1
best_setting = None


# =============================
# 1. 加载真实乳腺癌数据
# =============================
data = load_breast_cancer()
X = data.data          # (569, 30)
y = data.target        # 0/1

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
        self.fc1 = nn.Linear(30, HIDDEN_UNITS)   # 输入层 -> 隐藏层
        self.relu = nn.ReLU()          # 激活函数
        self.fc2 = nn.Linear(HIDDEN_UNITS, 2)    # 输出层（二分类）

    def forward(self, x):
        x = self.fc1(x)    # z = W1x + b1
        x = self.relu(x)   # a = f(z)
        x = self.fc2(x)    # 输出层
        return x

model = BPNet()

# =============================
# 3. 损失函数 & 优化器
# =============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for HIDDEN_UNITS in hidden_list:
    for LEARNING_RATE in lr_list:
        # ===== 重新建模型 =====
        model = BPNet()  # BPNet 里用 HIDDEN_UNITS
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # ===== 训练 =====
        epochs = 500
        for epoch in range(epochs):
            model.train()
            logits = model(X_train)
            loss = criterion(logits, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ===== 测试 =====
        model.eval()
        with torch.no_grad():
            pred = model(X_test).argmax(dim=1)
            test_acc = (pred == y_test).float().mean().item()

        print(f"HIDDEN_UNITS={HIDDEN_UNITS}, LEARNING_RATE={LEARNING_RATE}, Test Accuracy={test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            best_setting = (HIDDEN_UNITS, LEARNING_RATE)

print("\n===== BEST =====")
print(f"HIDDEN_UNITS={best_setting[0]}, LEARNING_RATE={best_setting[1]}, Best Test Accuracy={best_acc:.4f}")
