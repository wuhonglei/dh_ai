# 导入torch库
import torch as t

# 注意是小写tensor
a = t.tensor([1, 2, 3, 4, 5])

print(a) # 打印a
print(a.type()) # 打印a的类型

# 注意是大写Tensor
b = t.Tensor([1, 2, 3, 4, 5])

print(b) # 打印b
print(b.type()) # 打印b的类型


