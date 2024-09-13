import torch

#设置函数f，计算 x^2 - 4x - 5的值
def f(x):
    return x * x - 4 * x - 5

#设置函数df，f'(x) = 2*x-4，为f(x)的导函数
def df(x):
    return 2 * x - 4

#初始化一个带有自动梯度的张量x
x = torch.tensor([0.0], requires_grad=True)

y = f(x)  # 计算函数值y
y.backward()  # 调用backward计算y关于x的梯度
#梯度值会保存在x.grad中

#打印x和x.grad中保存的值
print("第1次打印:")
#通过data属性，访问张量的底层数据
print("x的值: ", x.data)
print("x位置的梯度值: ", x.grad.data)
print("验证，x位置的梯度值: ", df(x).data)
print("")

x.grad.zero_() #调用grad.zero，将梯度清零
y = f(x)  # 计算函数值
y.backward()  # 计算梯度

print("第2次打印:") #结果是正确的
print("x的值: ", x.data)
print("x位置的梯度值: ", x.grad.data)
print("验证，x位置的梯度值: ", df(x).data)
print("")

#第3次计算y和梯度前，不使用x.grad.zero_()
y = f(x)  # 计算函数值
y.backward()  # 计算梯度

print("第3次打印:") #梯度结果是-8，出现了错误
print("x的值: ", x.data)
print("x位置的梯度值: ", x.grad.data)
print("验证，x位置的梯度值: ", df(x).data)
print("")

