# 实现rosenbrock函数的计算
def rosenbrock(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

import torch
if __name__ == '__main__':
    # 定义自变量参数的取值，这里的1.3和6.7是随意设置的初始值
    x = torch.tensor(1.3, requires_grad = True)
    y = torch.tensor(6.7, requires_grad = True)

    # 定义LBFGS优化器optimizer
    optimizer = torch.optim.LBFGS([x, y])

    epoch = [0]
    for i in range(20): # LBFGS的迭代循环次数比较小
        # 相比SGD和Adam优化器，LBFGS的使用有一些特殊
        # closure函数的调用次数，看做是真实迭代次数
        def closure(): # 闭包函数
            optimizer.zero_grad()  # 清空梯度
            z = rosenbrock(x, y)  # 计算函数值
            z.backward()  # 计算梯度
            epoch[0] += 1
            # 打印一些调试信息来观察
            print(f'After {epoch[0]} iterations, '
                  f'x = {x.item():.3f}, '
                  f'y = {y.item():.3f}, '
                  f'z = {z.item():.3f}')
            return z
        # 将closure传入optimizer.step进行迭代
        # 每次调用optimizer.step时，其内部可能会多次调用closure函数
        optimizer.step(closure)


