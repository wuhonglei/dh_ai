from sklearn.datasets import make_gaussian_quantiles
import matplotlib.pyplot as plt
import math
import numpy

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

def polynomial(X):
    polyX = list()
    for i in range(0, len(X)):
        #在X中，包括了x1和x2两维特征
        x1 = X[i][0]
        x2 = X[i][1]
        #转换为二元二次多项式特征，包含6项
        polyX.append([1, x1, x2, x1 * x1, x2 * x2 , x1 * x2])
    return polyX


#函数生成平面上的等高数据点
#min-x1到max-x1是平面横轴范围
#min-x2到max-x2是纵轴范围
#根据已训练出的θ参数计算类别结果，不同类别结果对应有不同的高度
def make_contour_point(minx1, maxx1, minx2, maxx2, theta, n):
    #调用mesh-grid生成网格数据点，每个点的距离是0.02
    xx1, xx2 = numpy.meshgrid(numpy.arange(minx1, maxx1, 0.02),
                              numpy.arange(minx2, maxx2, 0.02))

    x1 = xx1.ravel()
    x2 = xx2.ravel()
    X = list()
    for i in range(0, len(x1)):
        #将所有数据点的x1和x2坐标组合在一起，添加到向量X中
        X.append([x1[i], x2[i]])
    polyX = polynomial(X) #将X转为多项式特征

    z = list() #设置列表z保存类别的预测结果
    for i in range(0, len(polyX)): #遍历全部样本
        #调用hypothesis，传入多项式特征预测类别
        h = hypothesis(theta, polyX[i], n)
        if h > 0.5: #如果结果大于0.5
            z.append(1) #类别是正例1
        else: #否则
            z.append(0) #是负例0
    #将结果重新设置为相同行和列的网格形式
    z = numpy.array(z).reshape(xx1.shape)
    return xx1, xx2, z #将网格坐标与高度一起返回

if __name__ == '__main__':
    #使用make_gaussian函数生成非线性数据
    X, y = make_gaussian_quantiles(n_samples=30, #30个样本
                                   n_features=2, #2个特征
                                   n_classes=2, #两个类别
                                   random_state=0)

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
    axis.set(xlim=[-4, 4],
             ylim=[-4, 4],
             title='Logistic Regression',
             xlabel='x1',
             ylabel='x2')
    # 画出正例和负例，其中正例使用蓝色圆圈表示，负例使用红色叉子表示
    plt.scatter(posx1, posx2, color='blue', marker='o')
    plt.scatter(negx1, negx2, color='red', marker='x')

    polyX = polynomial(X) #将线性特征转为多项式特征
    m = len(polyX)  # 保存样本个数
    n = 5  # 保存特征个数
    alpha = 0.001  # 迭代速率
    iterate = 10000  # 迭代次数
    # 调用梯度下降算法，迭代出分界平面，并计算代价值
    theta = gradient_descent(polyX, y, n, m, alpha, iterate)
    costJ = costJ(polyX, y, theta, n, m)
    #将训练得到的θ列表和代价值打印
    for i in range(0, len(theta)):
        print("theta[%d] = %lf" % (i, theta[i]))
    print("Cost J is %lf" % (costJ))
    #通过等高线图的方式，将决策边界绘制出来，得到分类结果
    x1, x2, z = make_contour_point(-4, 4, -4, 4, theta, n)
    plt.contour(x1, x2, z)
    plt.show()


