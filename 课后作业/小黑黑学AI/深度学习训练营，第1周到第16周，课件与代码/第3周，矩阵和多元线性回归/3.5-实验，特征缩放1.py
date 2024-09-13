# 设函数值hθ(x)的计算函数
# 函数传入参数theta、样本特征向量x和特征的个数n
def hypothesis(theta, x, n):
    h = 0.0  # 保存预测结果
    for i in range(0, n + 1):
        # 将theta-i和xi的乘积累加到h中
        h += theta[i] * x[i]
    return h  # 返回预测结果h


# J(θ)关于theta-j的偏导数计算函数
# 函数传入样本特征矩阵x和标记y，特征的参数列表theta，特征数n
# 样本数量m和待求偏导数的参数下标j
def gradient_thetaj(x, y, theta, n, m, j):
    sum = 0.0  # 保存sigma求和累加的结果
    for i in range(0, m):  # 遍历m个样本
        h = hypothesis(theta, x[i], n)  # 求出样本的预测值h
        # 计算预测值与真实值的差，再乘以第i个样本的第j个特征值x[i][j]
        # 将结果累加到sum
        sum += (h - y[i]) * x[i][j]
    return sum / m  # 返回累加结果sum除以样本的个数m


# 梯度下降的迭代函数
# 函数传入样本特征矩阵x和样本标签y，特征数n，样本数量m
# 迭代速率alpha和迭代次数iterate
def gradient_descent(x, y, n, m, alpha, iterate):
    theta = [0] * (n + 1)  # 初始化参数列表theta，长度为n+1
    for i in range(0, iterate):  # 梯度下降的迭代循环
        temp = [0] * (n + 1)
        # 使用变量j，同时对theta0到theta-n这n+1个参数进行更新
        for j in range(0, n + 1):
            # 通过临时变量列表temp，先保存一次梯度下降后的结果
            # 在迭代的过程中调用theta-j的偏导数计算函数
            temp[j] = theta[j] - alpha * gradient_thetaj(x, y, theta, n, m, j)
        # 将列表temp赋值给列表theta
        for j in range(0, n + 1):
            theta[j] = temp[j]
    return theta  # 函数返回参数列表theta


# 实现代价函数J(θ)的计算函数costJ
# 函数传入样本的特征矩阵x和标签y，参数列表theta，特征个数n和样本个数m
def costJ(x, y, theta, n, m):
    sum = 0.0  # 定义累加结果
    for i in range(0, m):  # 遍历每个样本
        h = hypothesis(theta, x[i], n)
        # 将样本的预测值与真实值差的平方累加到sum中
        sum += (h - y[i]) * (h - y[i])
    return sum / (2 * m)  # 返回sum除以2m


if __name__ == '__main__':
    m = 6  # 6个样本
    n = 2  # 每个样本2个特征
    #alpha = 0.01  # 更大迭代速率
    alpha = 0.0001  # 迭代速率
    #alpha = 0.00000001  # 非常小的迭代速率才可能正常迭代
    iterate = 1500  # 迭代次数iterate
    #样本的特征矩阵x和样本标签y
    #其中两个特征的特征值取值范围差别要设置的大一些
    x = [[1, 9679, 2],
         [1, 11039, 3],
         [1, 7025, 1],
         [1, 9996, 2],
         [1, 11815, 3],
         [1, 11508, 3]]
    y = [287, 343, 199, 298, 340, 350]

    #运行梯度下降算法对参数进行迭代
    theta = gradient_descent(x, y, n, m, alpha, iterate)
    costJ = costJ(x, y, theta, n, m)
    for i in range(0, n + 1):
        print("theta[%d] = %lf"%(i, theta[i]))
    print("costJ = %lf"%(costJ))


