import torch as t

print(t.__version__) # 打印当前的PyTorch版本

a = t.Tensor(2, 3) # 创建一个2x3的张量'a'
print(a) # 打印'a'

# 使用一个2x3的列表来创建张量'b'
b = t.Tensor([[1, 2, 3], [4, 5, 6]])
print(b) # 打印'b'

# 将张量'b'转换为列表，并赋值给'c'
c = b.tolist()
print(c) # 打印'c'

# 创建一个填充有随机数的3x3张量'd'
d = t.rand(3, 3)

# 使用'd'来创建张量'e'
e = t.Tensor(d)

print(d) # 打印'd'
print(e) # 打印'e'

