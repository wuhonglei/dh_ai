import torch as t # 导入torch库

# 创建一个2乘3的张量a
a = t.Tensor([[1, 2, 3], [4, 5, 6]])

# 打印a点size，得到它的形状
print(a.size())

# 创建值为1的张量
z1 = t.ones(2, 3)
print(z1)

# 创建值为0的张量
z2 = t.zeros(2, 3)
print(z2)

# 创建标准正太分布的随机数
z3 = t.randn(2, 3)
print(z3)

# 创建一个范围内的均匀数据
z4 = t.linspace(1, 10, 5)
print(z4)


