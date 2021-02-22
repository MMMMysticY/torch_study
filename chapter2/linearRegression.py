import torch
from torch.nn import Linear, Module, MSELoss
from torch.optim import SGD
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

x = np.random.rand(256)
noise = np.random.rand(256)/4
y = x*5+7+noise
df = pd.DataFrame()
df['x'] = x
df['y'] = y
sns.lmplot(x='x', y='y', data=df)


model=Linear(1,1) #直接调用线性回归模型
# 第一个参数是输入变量的维度是1 第二个参数是输出变量的参数维度也是1 y = wx + b

criterion = MSELoss()
#使用均方损失函数

optim = SGD(model.parameters(),lr=0.01)
#优化器用SGD

epochs = 3000
#训练3000次

x_train = x.reshape(-1, 1).astype('float32')
y_train = y.reshape(-1, 1).astype('float32')
#归一化到-1,1

for i in range(epochs):
    # 整理输入和输出的数据，这里输入和输出一定要是torch的Tensor类型
    inputs = torch.from_numpy(x_train)
    labels = torch.from_numpy(y_train)
    #使用模型进行预测
    outputs = model(inputs)
    #梯度置0，否则会累加
    optim.zero_grad()
    # 计算损失
    loss = criterion(outputs, labels)
    # 反向传播
    loss.backward()
    # 使用优化器默认方法优化
    optim.step()
    if (i%100==0):
        #每 100次打印一下损失函数，看看效果
        print('epoch {}, loss {:1.4f}'.format(i,loss.data.item()))
[w, b] = model.parameters()
print (w.item(),b.item())
