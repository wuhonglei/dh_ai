from torch import nn
import torch

# 定义一个具有特定权重的 ConvTranspose2d 层
# 输入通道数和输出通道数都为 1，卷积核大小为 2，步长为 2，填充为 0
ct = nn.ConvTranspose2d(in_channels = 1,
                        out_channels = 1,
                        kernel_size = 2,
                        stride = 2,
                        padding = 0)

# 手动设置卷积核的权重
# 权重的形状为 [输出通道数, 输入通道数, 高度, 宽度]
ct.weight.data = torch.tensor([[[[1., 4.],
                                 [2., 3.]]]])

# 手动设置偏置为 0
ct.bias.data.fill_(0)

# 创建一个大小为 [1, 1, 2, 2] 的输入张量
x = torch.tensor([[[[0., 1.],
                    [2., 3.]]]])

# 应用转置卷积层
output = ct(x)
print(f"output shape = {output.shape}")
print(output)

