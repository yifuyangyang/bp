import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# ====== 可修改参数区 ======
HIDDEN_UNITS = 32      # 隐藏层神经元个数
LEARNING_RATE = 0.005  # 学习率
# ==========================

# 固定随机种子（保证实验可复现）
torch.manual_seed(0)
np.random.seed(0)

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
    def __init__(self, hidden_units):
        super().__init__()
        self.fc1 = nn.Linear(30, hidden_units)   # 输入层 -> 隐藏层
        self.relu = nn.ReLU()                    # 激活函数
        self.fc2 = nn.Linear(hidden_units, 2)    # 输出层（二分类）

    def forward(self, x):
        x = self.fc1(x)    # z = W1x + b1
        x = self.relu(x)   # a = f(z)
        x = self.fc2(x)    # 输出层
        return x

# =============================
# 3. 模型训练与评估（多次运行取平均，降低随机误差）
# =============================
def train_and_evaluate(hidden_units, lr, runs=5):
    test_accs = []
    for run in range(runs):
        # 初始化模型、损失函数、优化器
        model = BPNet(hidden_units)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # 训练
        epochs = 200
        for epoch in range(1, epochs + 1):
            model.train()
            logits = model(X_train)          
            loss = criterion(logits, y_train)

            optimizer.zero_grad()
            loss.backward()                  
            optimizer.step()                 

        # 测试
        model.eval()
        with torch.no_grad():
            logits = model(X_test)
            pred = logits.argmax(dim=1)
            test_acc = (pred == y_test).float().mean().item()
            test_accs.append(test_acc)
    
    # 返回平均准确率
    avg_acc = np.mean(test_accs)
    print(f"隐藏层神经元数: {hidden_units}, 学习率: {lr}")
    print(f"5次运行平均测试准确率: {avg_acc:.4f}")
    print(f"各次运行准确率: {[round(acc,4) for acc in test_accs]}")
    return avg_acc

# 执行实验
best_acc = 0.0
best_params = (0, 0)
# 测试核心参数组合（你也可以扩展更多组合）
param_combinations = [
    (8, 0.001), (8, 0.005), (8, 0.01),
    (16, 0.001), (16, 0.005), (16, 0.01),
    (32, 0.001), (32, 0.005), (32, 0.01),
    (64, 0.001), (64, 0.005), (64, 0.01)
]

print("===== 实验开始 =====")
for hidden, lr in param_combinations:
    current_acc = train_and_evaluate(hidden, lr)
    if current_acc > best_acc:
        best_acc = current_acc
        best_params = (hidden, lr)
    print("-"*50)

# 输出最优结果
print("\n===== 最优参数结果 =====")
print(f"最优隐藏层神经元数: {best_params[0]}")
print(f"最优学习率: {best_params[1]}")
print(f"最高平均测试准确率: {best_acc:.4f}")

# 单独运行最优参数，输出最终结果（用于截图）
print("\n===== 最优参数单次运行结果（截图用） =====")
model = BPNet(best_params[0])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=best_params[1])

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
        print(f"epoch={epoch:3d} loss={loss.item():.4f} acc={acc:.4f}")

model.eval()
with torch.no_grad():
    logits = model(X_test)
    pred = logits.argmax(dim=1)
    test_acc = (pred == y_test).float().mean().item()
print("Test Accuracy =", test_acc)