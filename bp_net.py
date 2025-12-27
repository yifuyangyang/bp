import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ====== 可修改参数区 ======
HIDDEN_UNITS = 16      # 隐藏层神经元个数
LEARNING_RATE = 0.01   # 学习率
# ==========================

# 1. 加载并预处理数据
data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test  = torch.tensor(X_test, dtype=torch.float32)
y_test  = torch.tensor(y_test, dtype=torch.long)

# 2. 定义BP神经网络（使用可修改参数）
class BPNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(30, HIDDEN_UNITS)  # 用HIDDEN_UNITS替代固定值16
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_UNITS, 2)   # 隐藏层输出维度同步修改

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = BPNet()

# 3. 损失函数 & 优化器（使用可修改学习率）
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # 用LEARNING_RATE替代固定值0.01

# 4. 训练
epochs = 200
for epoch in range(1, epochs + 1):
    model.train()
    logits = model(X_train)
    loss = criterion(logits, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        pred = logits.argmax(dim=1)
        acc = (pred == y_train).float().mean().item()
        print(f"[训练] epoch={epoch:3d} loss={loss.item():.4f} acc={acc:.4f}")

# 5. 测试（重点输出最终测试准确率）
model.eval()
with torch.no_grad():
    logits = model(X_test)
    pred = logits.argmax(dim=1)
    test_acc = (pred == y_test).float().mean().item()

# 打印参数和最终准确率（方便截图）
print("\n===== 实验参数 & 结果 =====")
print(f"隐藏层神经元数: {HIDDEN_UNITS}")
print(f"学习率: {LEARNING_RATE}")
print(f"最终测试准确率: {test_acc:.4f}")