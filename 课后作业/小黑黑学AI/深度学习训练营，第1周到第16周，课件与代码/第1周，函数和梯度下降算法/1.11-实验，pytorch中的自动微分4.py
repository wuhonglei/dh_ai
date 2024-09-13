import torch #导入torch库

#设置函数f，计算 x^2 - 4x - 5的值
def f(x):
    return x * x - 4 * x - 5

#初始化张量x
x = torch.tensor([0.0], requires_grad=True)
iteration_num = 1000 #迭代次数
alpha = 0.01 #迭代速率

#进入梯度下降算法的循环
for i in range(0, iteration_num):
    y = f(x)  # 计算函数值y
    y.backward()  # 调用backward，计算y关于x的梯度

    # 通过操作x.data，更新x的值，进行梯度下降算法
    x.data -= alpha * x.grad.data
    #修改x.data会直接影响x的值，这是一种很原始的书写方式，
    #这里这样写主要是为了表示原始的梯度下降算法。

    x.grad.zero_() # 清除梯度，为下一次迭代做准备

#运行程序，会计算出在x=2的位置，f(x)取得极值-9
print("极值点: x = %.3lf"%(x.item()))
print("函数值: f(x) =", f(x).item())

