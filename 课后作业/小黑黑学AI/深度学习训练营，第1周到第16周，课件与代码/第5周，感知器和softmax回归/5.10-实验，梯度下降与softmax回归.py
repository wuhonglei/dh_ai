import numpy as np
import matplotlib.pyplot as plt

# 函数传入所有类别的线性结果z，计算每个类别的概率值
def softmax(z):
    # e的z次方可能会计算的很大，为了避免指数计算溢出，
    # 我们要将数组z中的每个元素，同时减去z的最大值
    z -= np.max(z) #该操作不会影响最终结果
    exp_z = np.exp(z) #计算e的z次方
    return exp_z / np.sum(exp_z) #返回e的z次方除以指数总和



# 传入参数w、样本x，特征个数n、类别个数c
# 函数计算样本x的预测结果
def softmax_output(w, x, n, c):
    z = np.zeros(c) #保存c个类别的线性结果
    for i in range(c): #循环每个类别
        for j in range(n):
            # 计算每个类别的线性输出
            z[i] += w[i][j] * x[j]
    return softmax(z) #返回softmax(z)


# 计算E关于全部参数的偏导数
def gradient_wkj(x, y, w, n, m, c):
    # 如果有c个类别，n个特征，那么就包含了c*n个参数
    # 设置c*n的数组保存梯度
    gradient = np.zeros((c, n))
    for j in range(0, n): #使用j循环具体计算哪个特征
        # 计算该特征，m个样本的梯度变化累加值
        for i in range(m):
            # 计算第i个样本，c个类别的预测概率
            p = softmax_output(w, x[i], n, c)
            for k in range(c): # 使用k，循环每个类别
                # (pik-yik)*xij，对应公式的后半部分
                gradient[k][j] += (p[k] - y[i][k]) * x[i][j]
    return gradient / m #返回梯度平均值

# 计算交叉熵损失函数的代价
def softmax_cost(x, y, n, m, c, w):
    cost = 0
    # 最外层循环累加每个样本的代价值
    for i in range(m):
        # 对于每个样本，先计算模型的预测结果p
        p = softmax_output(w, x[i], n, c)
        for k in range(c): #循环k个类别的标记值
            if y[i][k] == 1:  # 如果标记值等于1
                # 累加该类别的损失y[i][k]*log(p[k])
                cost += y[i][k] * np.log(p[k])
    return -cost / m #返回平均损失



# 实现梯度下降迭代，迭代sofmax回归模型
# 函数传入训练数据X、标记y、迭代速率α和迭代次数iterate
def softmax_train(X, y, alpha, iterate):
    m = len(X)  # 样本个数
    n = len(X[0]) # 特征个数
    c = len(y[0]) # 类别个数
    w = np.zeros((c, n)) # 定义模型的参数w，是c*n规模的矩阵
    for i in range(iterate): #进行梯度下降的循环迭代
        # 每次循环，需要计算一次当前的梯度值
        gradient = gradient_wkj(X, y, w, n, m, c)
        for k in range(c): # 根据梯度下降的更新公式，使用循环
            for j in range(0, n):
                # 对c*n个梯度，进行更新
                w[k][j] = w[k][j] - alpha * gradient[k][j]
        if (i + 1) % 100 == 0: #每迭代100次，打印一次损失值
            print("iterate %d : cost = %.3lf" %
                  (i + 1, softmax_cost(X, y, n, m, c, w)))
    return w #返回模型的参数w


# 使用参数w，预测x的类别
def softmax_predict(x, w):
    n = len(x) #特征个数
    c = len(w) #类别个数
    x = np.append(x, 1)  # 增加偏置项
    p = softmax_output(w, x, n, c) #计算预测概率p
    return np.argmax(p) #返回最大概率的类别

# 函数传入num，代表每种类别的数据个数
def make_data(num):
    # 使用np.random.randn生成满足正太分布的数据
    # 将生成的数据加上向量[0, -2]，代表绿色数据会以(0, -2)为中心分布
    green = np.random.randn(num, 2) + np.array([0, -2])
    # 生成蓝色数据，以(-2, 2)为中心分布，标准差为1的正太分布数据
    blue = np.random.randn(num, 2) + np.array([-2, 2])
    # 生成红色数据，以(2, 2)为中心分布，标准差为1的正太分布数据
    red = np.random.randn(num, 2) + np.array([2, 2])
    return green, blue, red

# 生成用于绘制决策边界的等高线数据
# min-x1到max-x1是画板的横轴范围，min-x2到max-x2是画板的纵轴范围
# model是训练好的模型
# 函数中，我们会根据已训练的model，计算对应类别结果，
# 不同类别结果会对应不同的高度，从而基于数据点的坐标与高度数据，绘制等高线
def draw_decision_boundary(minx1, maxx1, minx2, maxx2, w):
    # 调用mesh-grid生成网格数据点
    # 每个点的距离是0.02，这样生成的点可以覆盖平面的全部范围
    xx1, xx2 = np.meshgrid(np.arange(minx1, maxx1, 0.02),
                           np.arange(minx2, maxx2, 0.02))
    # 设置x1s、x2s和z分别表示数据点的横坐标、纵坐标和类别的预测结果
    x1s = xx1.ravel()
    x2s = xx2.ravel()
    z = list()
    for x1, x2 in zip(x1s, x2s): #遍历全部样本
        h = softmax_predict([x1, x2], w)
        z.append(h)
    # 将z重新设置为和xx1相同的形式
    z = np.array(z).reshape(xx1.shape)
    return xx1, xx2, z #返回xx1、xx2和z

if __name__ == '__main__':
    # 调用make_data，每种类别生成30个数据
    green, blue, red = make_data(30)
    # 创建-4到4的平面画板
    board = plt.figure()
    axis = board.add_subplot(1, 1, 1)
    axis.set(xlim=[-4, 4],
             ylim=[-4, 4],
             title='Softmax Regression',
             xlabel='x1',
             ylabel='x2')

    # 使用plt.scatter绘制出绿色、蓝色和红色三种数据
    plt.scatter(green[:, 0], green[:, 1], color='green')
    plt.scatter(blue[:, 0], blue[:, 1], color='blue')
    plt.scatter(red[:, 0], red[:, 1], color='red')

    # 将绿色、蓝色、红色三种数据连在一起
    X = np.concatenate((green, blue, red), axis=0)
    # 为训练数据添加偏置特征
    X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
    # 构造数据对应的标记值y
    y_green = np.array([[1, 0, 0]] * 30)
    y_blue = np.array([[0, 1, 0]] * 30)
    y_red = np.array([[0, 0, 1]] * 30)
    y = np.concatenate((y_green, y_blue, y_red), axis=0)

    # 调用softmax_train训练模型
    w = softmax_train(X, y, 0.001, 5000)
    # 使用函数draw_decision_boundary，生成数据
    xx1, xx2, z = draw_decision_boundary(-4, 4, -4, 4, w)
    # 调用plt.contour绘制多分类的决策边界
    plt.contour(xx1, xx2, z, colors=['orange'])
    plt.show()













