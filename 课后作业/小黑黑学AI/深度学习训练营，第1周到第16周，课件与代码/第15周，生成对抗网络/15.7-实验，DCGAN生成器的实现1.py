import torch
import torch.nn.functional as F

# 创建一个大小为 [1, 1, 2, 2] 的输入张量
x = torch.tensor([[[[0., 1.],
                    [2., 3.]]]])

# 定义卷积核的权重
# 权重的形状为 [输出通道数, 输入通道数, 高度, 宽度]
w = torch.tensor([[[[1., 4.],
                    [2., 3.]]]])

# 定义偏置
b = torch.tensor([0.])  # 单个输出通道的偏置

# 使用 conv_transpose2d 函数应用转置卷积
output = F.conv_transpose2d(x, w, b, stride=2, padding=0)
print(f"output shape = {output.shape}")
print(output)
print("")
output = F.conv_transpose2d(x, w, b, stride=2, padding=1)
print(f"output shape = {output.shape}")
print(output)

