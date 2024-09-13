# 以SGD为例，实现对Rosenbrock函数的优化过程
# 实现rosenbrock函数的计算
def rosenbrock(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

import torch
if __name__ == '__main__':
    # 定义自变量参数的取值，这里的1.3和6.7是随意设置的初始值
    x = torch.tensor(1.3, requires_grad = True)
    y = torch.tensor(6.7, requires_grad = True)

    # 定义SGD优化器，先将迭代速率lr暂时设置为常用的0.001
    optimizer = torch.optim.SGD([x, y], lr = 0.001)

    for epoch in range(30000): # 迭代的轮数暂时设置为30000
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




