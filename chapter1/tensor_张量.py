import torch
import numpy as np

# tensors和numpy中的ndarray类似 但是tensor可以进行gpu运算


# x = torch.empty(5, 3)
# print(x)

# x = torch.rand(5, 3)
# print(x)

x = torch.zeros(5, 3)
# print(x)

# 通过empty rand zeros可以通过设置维度创建张量

t = torch.tensor([5.5, 3])
# print(t)  # tensor的构造函数可以是数组

t = t.new_ones(5, 3, dtype=torch.double)
# print(t)  # new_ + * 方法来创建一个新的对象

# print(t.size())  # 通过size()方法可以查看张量的维度

result = torch.empty(5, 3)
y = torch.rand(5, 3)
torch.add(t, y, out=result)
# print(result)   #张量间可以进行算术运算 其他运算类同理

# print(y)
# print(y[:,1]) #可以使用类似于numpy中的切片方式进行切片

x = torch.rand(5,3)
y = x.view(15)
z = x.view(-1,5)
# print(x.size(),y.size(),z.size())   #使用view方法可以改变张量的维度和大小，与reshape类似

#numpy和tensor之间的互相转化非常便捷
a = torch.ones(5)
b = a.numpy()
# print(b)
a.add_(1)
# print(a)
# print(b)
c = np.ones(5)
d = torch.from_numpy(c)
np.add(c,1,out=c)
# print(c)
# print(d)    #可以看到在numpy和tensor互相转化时，共用的是一片内存区域，一方的修改会导致另一方的修改

if torch.cuda.is_available():
    device = torch.device('cuda')
    y = torch.ones_like(x,device=device)
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to('cpu',torch.double))