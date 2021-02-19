import torch
import torchvision

x = torch.ones(2,2,requires_grad=True)
#使用requires_grad属性设置为true 可以跟踪所有对该张量的操作
print(x)

y = x+2
print(y)
#此时可以从输出内容看出 tensor对象带了grad_fn属性并显示的是加法

z = y*y*3
out = z.mean()
print(z)
print(out)
#对于z而言 得到它所进行的操作是乘法 对于out而言 得到它的操作是均值操作

a = torch.randn(2,2)
a = ((a*3)/(a-1))
print(a.requires_grad)
a.requires_grad = True
b = (a * a).sum()
print(b.grad_fn)
#可以设置requires_grad属性 默认是False 在设为True之后 可以取grad_fn等属性

out.backward()
#backward()方法用于求 d(out)/dx 是一个链式求导过程
print(x.grad)
#在反向传播之后可以使用grad属性获取求导结果
#y = f(x) y关于x的求导是一个雅克比矩阵 上面的过程就是计算出了雅克比矩阵 x.grad

x1 = torch.rand(3,requires_grad=True)

y1 = x1 * 2
while y1.data.norm() < 1000:
    y1 = y1 *2
print(y1)

gradients = torch.tensor([0.1,1,0.001],dtype=torch.float)
y1.backward(gradients)
print(x1.grad)
#这个样例想说明什么呢
#说明可以在banckgrad()的时候加上一个参数 这个参数是l = g(y) dl/dy 已知dy/dx 得到了dl/dx
#不明白

#官方文档给的样例:
a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)
Q = 3*a**3 - b**2
#此时dQ/da = 9a**2  dQ/db = -2b
external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)
#需要在Q.backward()的过程中显式传递gradient参数，gradient是和Q形状相同的张量，它表示Q相对于本身的梯度
#dQ/dQ = 1 所以都使用1
print(9*a**2 == a.grad)
print(-2*b == b.grad)
#这个样例中就使用了gradient是1的向量

#不计算梯度的参数常被成为冻结参数，如果事先知道某些参数需要冻结，可以设置其属性requires_grad = False 可以减少部分参数迭代

from torch import nn, optim

model = torchvision.models.resnet18(pretrained=True)

# Freeze all the parameters in the network
for param in model.parameters():
    param.requires_grad = False
#如在本样例中使用预训练的resnet18 将所有参数冻结
model.fc = nn.Linear(512, 10)
#之后将最后一层线性层赋值新的 不冻结
#此时的训练就是只会修改该线性层的参数