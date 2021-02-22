from torch.utils.data import Dataset
import torch
import pandas as pd
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision import transforms as transforms


# Dataset是一个抽象类，为了数据的方便读入，需要将使用的数据包装到该类中

class BulldozerDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    # init方法用于初始化 在该类中用于导入数据
    def __len__(self):
        return len(self.df)

    # len方法要返回数据的长度
    def __getitem__(self, item):
        return self.df.iloc[item].SalePrice
    # getitem方法则是根据索引获取数据


ds_demo = BulldozerDataset('median_benchmark.csv')
# 导入数据

# print(len(ds_demo))

# print(ds_demo[0])

dl = torch.utils.data.DataLoader(ds_demo, batch_size=10, shuffle=True, num_workers=0)
# DataLoader用于导入数据 常用参数有：batch_size(每个batch的大小)、 shuffle(是否进行shuffle操作)、 num_workers(加载数据的时候使用几个子进程)

for i, data in enumerate(dl):
    print(i,data)
    break


# torchvision.datasets 可以理解为PyTorch团队自定义的dataset，这些dataset帮我们提前处理好了很多的图片数据集，我们拿来就可以直接使用：
'''
MNIST
COCO
Captions
Detection
LSUN
ImageFolder
Imagenet-12
CIFAR
STL10
SVHN
'''

trainset = datasets.MNIST(root='./data', # 表示 MNIST 数据的加载的目录
                                      train=True,  # 表示是否加载数据库的训练集，false的时候加载测试集
                                      download=True, # 表示是否自动下载 MNIST 数据集
                                      transform=None) # 表示是否需要对数据进行预处理，none为不进行预处理

# torchvision不仅提供了常用图片数据集，还提供了训练好的模型，可以加载之后，直接使用，或者在进行迁移学习 torchvision.models模块的 子模块中包含以下模型结构。
'''
AlexNet
VGG
ResNet
SqueezeNet
DenseNet
'''

resnet18 = models.resnet18(pretrained=True)

# transforms 模块提供了一般的图像转换操作类，用作数据处理和数据增强


transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  #先四周填充0，在把图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.RandomRotation((-45,45)), #随机旋转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.229, 0.224, 0.225)), #R,G,B每层的归一化用到的均值和方差
])
