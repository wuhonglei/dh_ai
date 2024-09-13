
import torch
from torch import nn

#创建一个输入通道数是1，输出通道数是6，卷积核大小为5的卷积层
conv = nn.Conv2d(1, 6, 5)
print(f"conv: {conv}")

#10代表样本数量，1是通道数，8乘8是图片大小
x = torch.randn(10, 1, 8, 8)
print(f"x.shape: {x.shape}")

out = conv(x) #将x输入至卷积层conv
#输出结果为[10, 6, 4, 4]的张量
#10代表样本数，6是输出通道数，4乘4是经过卷积后，特征图的大小
print(f"out.shape: {out.shape}")
print("")

#为了明确的表示各参数，可以显示的将参数名写出并赋值
conv2 = nn.Conv2d(in_channels=3, #输入通道数为3
                  out_channels=10, #输出通道数为10
                  kernel_size=3, #卷积核大小是3乘3
                  stride=3, #步幅为3
                  padding=1) #填充为1
print(f"conv2: {conv2}")

#代表5个样本，3个输入通道，4乘4的大小
x2 = torch.randn(5, 3, 4, 4)
print(f"x2.shape: {x2.shape}")

#张量x2经过卷积层conv2后，会得到[5, 10, 2, 2]的结果
#5是样本数，10是输出通道，2乘2是输出特征图大小
#4乘4大小的图片，在外侧添加1圈填充后，大小变为6乘6
#由于步幅是3，因此最终输出特征图大小是2乘2
out2 = conv2(x2)
print(f"out2.shape: {out2.shape}")
print("")

print("conv.parameters:")
#打印conv的可训练参数
#conv中一共包括了6*1*5*5=150个w参数，6个b参数
for p in conv.parameters():
    # 打印可训练参数的形状和数量
    print(f"p.shape = {p.shape} num = {p.numel()}")

print("conv2.parameters:")
#打印conv2的可训练参数
#onv2中包括了10*3*3*3=270个w参数，10个b参数
for p in conv2.parameters():
    # 打印可训练参数的形状和数量
    print(f"p.shape = {p.shape} num = {p.numel()}")

print("")

#从torch.nn中导入functional
from torch.nn import functional
#创建一个输出通道为6，输入通道为1，5乘5的卷积核
w = torch.randn(6, 1, 5, 5)
print(f"w.shape: {w.shape}")
#创建一个10组样本，输入通道为1，8乘8大小的输入张量x
x = torch.randn(10, 1, 8, 8)
print(f"x.shape: {x.shape}")
#调用functional中conv2d，计算x与w的卷积结果
out = functional.conv2d(x, w)
#结果是一个10*6*4*4大小的张量
print(f"out.shape: {out.shape}")

print("")

#创建一个窗口大小为2乘2的最大池化层
pool = nn.MaxPool2d(2)
#创建一个10组样本，输入通道为1，8乘8大小的输入张量x
x = torch.randn(10, 1, 8, 8)
#将8*8的张量x输入至pool，会得到4乘4的结果
out = pool(x)
print(f"x.shape: {out.shape}")
out2 = functional.max_pool2d(x, 2)
print(f"x.shape: {out2.shape}")




