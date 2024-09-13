# 以SGD为例，实现对Rosenbrock函数的优化过程
# 实现rosenbrock函数的计算
def rosenbrock(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

import torch
if __name__ == '__main__':
    # 定义自变量参数的取值，这里的1.3和6.7是随意设置的初始值
    x = torch.tensor(1.3, requires_grad = True)
    y = torch.tensor(6.7, requires_grad = True)

    # 将lr从0.001调整为0.0001，重新运行程序
    optimizer = torch.optim.SGD([x, y], lr = 0.0001)

    for epoch in range(500000): # 迭代的轮数暂时设置为500000
        optimizer.zero_grad()  # 清空梯度

        z = rosenbrock(x, y) # 计算函数值
        z.backward() #计算梯度
        optimizer.step() #更新模型参数

        if (epoch + 1) % 1000 == 0:
            #每1000次迭代，打印一次loss信息
            print(f'After {epoch + 1} iterations, '
                  f'x = {x.item():.3f}, '
                  f'y = {y.item():.3f}, '
                  f'z = {z.item():.3f}')




