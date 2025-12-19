import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# =============================
# 1. 数据加载与预处理（固定不变）
# =============================
data = load_breast_cancer()
X = data.data          # (569, 30)
y = data.target        # 0/1

# 划分训练集 / 测试集（固定随机种子保证可复现）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# 转为 Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test  = torch.tensor(X_test, dtype=torch.float32)
y_test  = torch.tensor(y_test, dtype=torch.long)

# =============================
# 2. 定义模型（参数化）
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
# 3. 定义训练和测试函数
# =============================
def train_and_evaluate(hidden_units, learning_rate, epochs=200):
    """
    训练并评估模型
    返回：测试集准确率
    """
    # 初始化模型、损失函数、优化器
    model = BPNet(hidden_units)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练过程
    for epoch in range(epochs):
        model.train()
        logits = model(X_train)          
        loss = criterion(logits, y_train)

        optimizer.zero_grad()
        loss.backward()                  
        optimizer.step()                

    # 测试过程
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        pred = logits.argmax(dim=1)
        test_acc = (pred == y_test).float().mean().item()
    
    return test_acc

# =============================
# 4. 参数网格搜索
# =============================
# 定义要测试的参数范围
hidden_units_candidates = [8, 16, 32, 64, 128]  # 隐藏层神经元候选值
lr_candidates = [0.001, 0.005, 0.01, 0.05, 0.1]  # 学习率候选值

# 存储结果
results = []

# 遍历所有参数组合
for hu in hidden_units_candidates:
    for lr in lr_candidates:
        # 重复训练3次取平均值（减少随机性影响）
        acc_list = []
        for _ in range(3):
            test_acc = train_and_evaluate(hu, lr)
            acc_list.append(test_acc)
        
        avg_acc = np.mean(acc_list)
        std_acc = np.std(acc_list)
        
        results.append({
            'hidden_units': hu,
            'learning_rate': lr,
            'avg_accuracy': avg_acc,
            'std_accuracy': std_acc
        })
        
        print(f"隐藏层神经元: {hu:3d}, 学习率: {lr:.4f} -> 平均准确率: {avg_acc:.4f}, 标准差: {std_acc:.4f}")

# =============================
# 5. 找出最优参数组合
# =============================
# 按准确率排序
results.sort(key=lambda x: x['avg_accuracy'], reverse=True)
best_result = results[0]

print("\n" + "="*50)
print("最优参数组合：")
print(f"隐藏层神经元个数: {best_result['hidden_units']}")
print(f"学习率: {best_result['learning_rate']}")
print(f"最高平均测试准确率: {best_result['avg_accuracy']:.4f}")
print("="*50)

# =============================
# 6. 使用最优参数重新训练并展示详细结果
# =============================
print("\n使用最优参数重新训练：")
best_hu = best_result['hidden_units']
best_lr = best_result['learning_rate']

# 重新训练并打印训练过程
model = BPNet(best_hu)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=best_lr)

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

# 最终测试
model.eval()
with torch.no_grad():
    logits = model(X_test)
    pred = logits.argmax(dim=1)
    test_acc = (pred == y_test).float().mean().item()

print(f"\n最终测试准确率（最优参数）: {test_acc:.4f}")