import torch
import torch.nn as nn
import torch.nn.functional as F

#按照nn进行网络搭建

class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,6,5) # 1 代表输入图像通道为1 6代表输出通道为6 5代表卷积大小为5*5
        self.conv2 = nn.Conv2d(6,16,5) #同上

        self.func1 = nn.Linear(16*5*5,120)
        self.func2 = nn.Linear(120,84)
        self.func3 = nn.Linear(84,10)

    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2)) #通过2*2的池化层
        x = F.max_pool2d(F.relu(self.conv2(x)),2) #同上
        x = x.view(-1,self.num_flat_features(x)) #view相当于reshape -1代表自动识别 其实就是拉直操作
        x = F.relu(self.func1(x))
        x = F.relu(self.func2(x))
        x = self.func3(x) #最后三层
        return x

    def num_flat_features(self,x):
        # print(x.size())
        size = x.size()[1:] #取size()的第二维
        num_features = 1
        for s in size:
            num_features *=s #计算所有像素个数
        return num_features
net = Net()
print(net)
#在模型中必须定义forward方法而backward函数（用来计算梯度）会被autogard自动创建

params = list(net.parameters())
print(len(params)) #有10层参数 很明显
print(params[0].size()) #可以获取各层的具体维度

print('1.定义一个网络')
print('-------------------------------------------------')


input = torch.randn(1,1,32,32) #输入的是32*32的一张通道为一的随机图像 是一个四维的 Samples * Channels * Height * Width
out = net(input)
print(out)

net.zero_grad() #清空所有梯度
out.backward(torch.randn(1,10)) #按照随机梯度进行反向传播

print('2.输入训练数据 反向传播')
print('-------------------------------------------------')

output = net(input)
target = torch.randn(10)#target代表样本准确值 有监督学习可以直接得到
target = target.view(1,-1)
criterion = nn.MSELoss() #调用MSE loss function

loss = criterion(output,target) #loss function进行计算网络输出值和目标值的差距
print(loss)

print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU
#可以获得一些细节属性

net.zero_grad()
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
#反向传播计算梯度

print('3.设置损失函数 计算损失值 并反向传播计算梯度')
print('-------------------------------------------------')

import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # 清空优化方法参数
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # 进行迭代优化

print('4. 定义优化方法 进行迭代优化')
print('-------------------------------------------------')