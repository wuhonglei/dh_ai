def gradient(x): #计算x位置的梯度
    return 2.0 * x - 4 #函数f(x) = x^2 - 4x - 5的梯度

#梯度下降的过程中，函数返回迭代完成后，f(x)取得最小值时，x的值
def gradient_descent():
    x = 0.0 #从位置0开始迭代
    iteration_num = 10 #迭代次数
    alpha = 0.001 #迭代速率
    for i in range(0, iteration_num):
        print ("%d iteration x = %lf gradient(x) = %lf"
               % (i, x, gradient(x)))
        x = x - alpha * gradient(x) #进行梯度下降，修改x的值
    return x #返回x

if __name__ == '__main__':
    gradient_descent()

