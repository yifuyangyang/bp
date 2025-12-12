import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
# class BPNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(30, 16)   # 输入层 -> 隐藏层
#         self.relu = nn.ReLU()          # 激活函数
#         self.fc2 = nn.Linear(16, 2)    # 输出层（二分类）

#     def forward(self, x):
#         x = self.fc1(x)    # z = W1x + b1
#         x = self.relu(x)   # a = f(z)
#         x = self.fc2(x)    # 输出层
#         return x

# model = BPNet()

# # =============================
# # 3. 损失函数 & 优化器
# # =============================
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)

# # =============================
# # 4. 训练（BP 发生在这里）
# # =============================
# epochs = 200
# for epoch in range(1, epochs + 1):
#     model.train()

#     logits = model(X_train)          # 前向传播
#     loss = criterion(logits, y_train)

#     optimizer.zero_grad()
#     loss.backward()                  # 反向传播（BP）
#     optimizer.step()                 # 参数更新

#     if epoch % 20 == 0:
#         pred = logits.argmax(dim=1)
#         acc = (pred == y_train).float().mean().item()
#         print(f"epoch={epoch:3d} loss={loss.item():.4f} acc={acc:.4f}")

# # =============================
# # 5. 测试
# # =============================
# model.eval()
# with torch.no_grad():
#     logits = model(X_test)
#     pred = logits.argmax(dim=1)
#     test_acc = (pred == y_test).float().mean().item()

# print("Test Accuracy =", test_acc)



# =============================
# 2. 定义参数候选集（仅修改这两个参数的候选值）
# =============================
# 待测试的隐藏层神经元个数候选
hidden_units_list = [8, 16, 32, 64]
# 待测试的学习率候选
lr_list = [0.001, 0.005, 0.01, 0.02]

# 记录最优结果
best_test_acc = 0.0
best_hidden_units = 0
best_lr = 0.0

# =============================
# 3. 遍历参数组合训练并测试
# =============================
for HIDDEN_UNITS in hidden_units_list:
    for LEARNING_RATE in lr_list:
        print(f"\n===== 当前参数：隐藏层神经元={HIDDEN_UNITS}，学习率={LEARNING_RATE} =====")
        
        # =============================
        # 4. 最简 BP 神经网络（参数化隐藏层）
        # =============================
        class BPNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(30, HIDDEN_UNITS)   # 动态设置隐藏层神经元
                self.relu = nn.ReLU()                    # 激活函数
                self.fc2 = nn.Linear(HIDDEN_UNITS, 2)    # 输出层（二分类）

            def forward(self, x):
                x = self.fc1(x)    # z = W1x + b1
                x = self.relu(x)   # a = f(z)
                x = self.fc2(x)    # 输出层
                return x

        model = BPNet()

        # =============================
        # 5. 损失函数 & 优化器（动态设置学习率）
        # =============================
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # =============================
        # 6. 训练（BP 发生在这里）
        # =============================
        epochs = 200
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
        # 7. 测试
        # =============================
        model.eval()
        with torch.no_grad():
            logits = model(X_test)
            pred = logits.argmax(dim=1)
            test_acc = (pred == y_test).float().mean().item()

        print(f"当前参数测试准确率：Test Accuracy = {test_acc:.4f}")
        
        # 更新最优结果
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_hidden_units = HIDDEN_UNITS
            best_lr = LEARNING_RATE

# =============================
# 8. 输出最优结果
# =============================
print("\n==================== 最优结果 ====================")
print(f"最高测试准确率：{best_test_acc:.4f}")
print(f"最优参数：HIDDEN_UNITS = {best_hidden_units}，LEARNING_RATE = {best_lr}")
