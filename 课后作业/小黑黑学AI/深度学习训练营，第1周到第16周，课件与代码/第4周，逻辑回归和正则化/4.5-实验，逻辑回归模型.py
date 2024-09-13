from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy
import math

#sigmoid函数的计算
def sigmoid(z):
    return 1.0 / (1 + math.exp(-z))

# 逻辑回归假设函数的计算
# 函数传入参数theta、样本特征向量x和特征的个数n
def hypothesis(theta, x, n):
    h = 0.0  # 保存预测结果
    for i in range(0, n + 1):
        # 将theta-i和xi的乘积累加到h中
        h += theta[i] * x[i]
    return sigmoid(h)  # 返回sigmoid函数的计算结果



# J(θ)关于theta-j的偏导数计算函数
# 函数传入样本特征矩阵x和标记y，特征的参数列表theta，特征数n
# 样本数量m和待求偏导数的参数下标j
def gradient_thetaj(x, y, theta, n, m, j):
    sum = 0.0 # 保存sigma求和累加的结果
    for i in range(0, m): # 遍历m个样本
        h = hypothesis(theta, x[i], n) # 求出样本的预测值h
        # 计算预测值与真实值的差，再乘以第i个样本的第j个特征值x[i][j]
        # 将结果累加到sum
        sum += (h - y[i]) * x[i][j]
    return sum / m # 返回累加结果sum除以样本的个数m
    

# 梯度下降的迭代函数
# 函数传入样本特征矩阵x和样本标签y，特征数n，样本数量m
# 迭代速率alpha和迭代次数iterate
def gradient_descent(x, y, n, m, alpha, iterate):
    theta = [0] * (n + 1) # 初始化参数列表theta，长度为n+1
    for i in range(0, iterate): # 梯度下降的迭代循环
        temp = [0] * (n + 1)
        # 使用变量j，同时对theta0到theta-n这n+1个参数进行更新
        for j in range(0, n + 1):
            # 通过临时变量列表temp，先保存一次梯度下降后的结果
            # 在迭代的过程中调用theta-j的偏导数计算函数
            temp[j] = theta[j] - alpha * gradient_thetaj(x, y, theta, n, m, j)
        # 将列表temp赋值给列表theta
        for j in range(0, n + 1):
            theta[j] = temp[j]
    return theta # 函数返回参数列表theta
    

# 实现代价函数J(θ)的计算函数costJ
# 函数传入样本的特征矩阵x和标签y，参数列表theta，特征个数n和样本个数m
def costJ(x, y, theta, n, m):
    sum = 0.0  # 定义累加结果
    for i in range(0, m):  # 遍历每个样本
        h = hypothesis(theta, x[i], n)
        # 将样本的预测值与真实值差的平方累加到sum中
        sum += -y[i] * math.log(h) - (1 - y[i]) * math.log(1 - h)
    return sum / m  # 返回sum除以m



if __name__ == '__main__':
    # 使用make_blobs随机的生成正例和负例，其中n_samples代表样本数量，设置为50
    # centers代表聚类中心点的个数，可以理解为类别标签的数量，设置为2
    # random_state是随机种子，将其固定为0，这样每次运行就生成相同的数据
    # cluster_s-t-d是每个类别中样本的方差，方差越大说明样本越离散，这里设置为0.5
    X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.5)

    posx1 = list()  # 将正样本的第1、2维保存到pos-x1和pos-x2
    posx2 = list()
    negx1 = list()  # 负样本的第1、2维保存到neg-x1和neg-x2
    negx2 = list()
    for i in range(0, len(y)):  # 遍历所有的样本标签
        # 将特征添加到这四个列表中
        if y[i] == 1:
            posx1.append(X[i][0])
            posx2.append(X[i][1])
        else:
            negx1.append(X[i][0])
            negx2.append(X[i][1])

    # 创建画板对象，并设置坐标轴
    board = plt.figure()
    axis = board.add_subplot(1, 1, 1)
    # 横轴和纵轴分别对应x1和x2两个特征，长度从-1到5，画板名称为SVM
    axis.set(xlim=[-1, 6],
             ylim=[-1, 6],
             title='Logistic Regression',
             xlabel='x1',
             ylabel='x2')
    # 画出正例和负例，其中正例使用蓝色圆圈表示，负例使用红色叉子表示
    plt.scatter(posx1, posx2, color='blue', marker='o')
    plt.scatter(negx1, negx2, color='red', marker='x')

    m = len(X) # 保存样本个数
    n = 2 # 保存特征个数
    alpha = 0.001  # 迭代速率
    iterate = 20000  # 迭代次数
    # 将生成的特征向量X的添加一列1，作为偏移特征
    X = numpy.insert(X, 0, values=[1] * m, axis=1).tolist()
    y = y.tolist()

    #调用梯度下降算法，迭代出分界平面，并计算代价值
    theta = gradient_descent(X, y, n, m, alpha, iterate)
    costJ = costJ(X, y, theta, n, m)
    for i in range(0, len(theta)):
        print("theta[%d] = %lf" % (i, theta[i]))
    print("Cost J is %lf" % (costJ))

    #根据迭代出的模型参数，绘制分类的决策边界
    w1 = theta[1]
    w2 = theta[2]
    b = theta[0]
    # 使用linspace在-1到5之间构建间隔相同的100个点
    x = numpy.linspace(-1, 6, 100)
    # 将这100个点，代入到决策边界，计算纵坐标
    d = - (w1 * x + b) * 1.0 / w2
    #绘制分类的决策边界
    plt.plot(x, d)
    plt.show()

