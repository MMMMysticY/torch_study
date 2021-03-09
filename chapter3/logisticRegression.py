import torch
import torch.nn as nn
import numpy as np

data = np.loadtxt("german.data-numeric")
# 导入数据

n, l = data.shape
for j in range(l - 1):
    meanValue = np.mean(data[:, j])
    stdValue = np.std(data[:, j])
    data[:, j] = (data[:, j] - meanValue) / stdValue
# 数据归一化（？）

np.random.shuffle(data)
# 打乱数据

train_data = data[:900, :l - 1]
train_lab = data[:900, l - 1] - 1
test_data = data[900:, :l - 1]
test_lab = data[900:, l - 1] - 1


# 区分测试集和训练集 注意最后一列就是分类

class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.fc = nn.Linear(24, 2)  # 输入维度24 输出维度2

    def forward(self, x):
        out = self.fc(x)
        out = torch.sigmoid(out)
        return out


# 构建逻辑回归网络 线性回归+sigmoid

def test(pred, lab):
    t = pred.max(-1)[1] == lab
    return torch.mean(t.float())


# 测试结果准确率

net = LR()
criterion = nn.CrossEntropyLoss()  # 使用CrossEntropyLoss损失
optm = torch.optim.Adam(net.parameters())  # Adam优化
epochs = 1000  # 训练1000次

for i in range(epochs):
    # 指定模型为训练模式，计算梯度
    net.train()
    # 输入值都需要转化成torch的Tensor
    x = torch.from_numpy(train_data).float()
    y = torch.from_numpy(train_lab).long()
    y_hat = net(x)
    loss = criterion(y_hat, y)  # 计算损失
    optm.zero_grad()  # 前一步的损失清零
    loss.backward()  # 反向传播
    optm.step()  # 优化
    if (i + 1) % 100 == 0:  # 这里我们每100次输出相关的信息
        # 指定模型为计算模式
        net.eval()
        test_in = torch.from_numpy(test_data).float()
        test_l = torch.from_numpy(test_lab).long()
        test_out = net(test_in)
        # 使用我们的测试函数计算准确率
        accu = test(test_out, test_l)
        print("Epoch:{},Loss:{:.4f},Accuracy：{:.2f}".format(i + 1, loss.item(), accu))
