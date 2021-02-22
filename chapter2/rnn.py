import torch


'''
pytorch 中使用 nn.RNN 类来搭建基于序列的循环神经网络，它的构造函数有以下几个参数：

input_size：输入数据X的特征值的数目。

hidden_size：隐藏层的神经元数量，也就是隐藏层的特征数量。

num_layers：循环神经网络的层数，默认值是 1。

bias：默认为 True，如果为 false 则表示神经元不使用 bias 偏移参数。

batch_first：如果设置为 True，则输入数据的维度中第一个维度就是 batch 值，默认为 False。默认情况下第一个维度是序列的长度，
第二个维度才是 - - batch，第三个维度是特征数目。

dropout：如果不为空，则表示最后跟一个 dropout 层抛弃部分数据，抛弃数据的比例由该参数指定。

'''

#对于RNN来说，我们只要己住一个公式：
#见 RNN_FORMULATION


rnn = torch.nn.RNN(20,50,2)
input = torch.randn(100 , 32 , 20)
h_0 =torch.randn(2 , 32 , 50)
output,hn=rnn(input ,h_0)
print(output.size(),hn.size())

lstm = torch.nn.LSTM(10, 20,2)
input = torch.randn(5, 3, 10)
h0 =torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)
output, hn = lstm(input, (h0, c0))
print(output.size(),hn[0].size(),hn[1].size())
#LSTM和RNN参数基本相同

#有疑问的是input h_0 的第二维参数是什么意思