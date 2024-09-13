# J(θ)关于theta0的偏导数，传入保存样本数据的列表x和y，参数theta0和theta1的值
def gradient_theta0(x, y, theta0, theta1):
    sum = 0.0  # 保存sigma求和累加的结果
    m = len(x)  # 保存样本的个数
    for i in range(0, m):  # 遍历每个样本
        # 将样本的预测值与真实值的差累加到sum中
        sum += theta0 + theta1 * x[i] - y[i]
    return sum / m  # 返回求和结果sum除以样本的个数m

# J(θ)关于theta1的偏导数，传入保存样本数据的列表x和y，参数theta0和theta1的值
def gradient_theta1(x, y, theta0, theta1):
    sum = 0.0  # 保存sigma求和累加的结果
    m = len(x)  # 保存样本的个数
    for i in range(0, m):  # 遍历每个样本
        # 将样本的预测值与真实值的差乘以样本的特征x
        sum += (theta0 + theta1 * x[i] - y[i]) * x[i]
    return sum / m  # 返回求和结果sum除以样本的个数m

# 梯度下降迭代，传入样本数据列表x和y，模型迭代速率alpha和迭代次数n
def gradient_descent(x, y, alpha, n):
    # 初始化参数theta0和theta1
    theta0 = 0.0
    theta1 = 0.0
    for i in range(0, n):  # 梯度下降的迭代循环，循环n次
        # 同时对theta0和theta1进行更新
        # 通过两个临时变量temp0和temp1，先保存一次梯度下降后，theta0和theta1的结果
        temp0 = theta0 - alpha * gradient_theta0(x, y, theta0, theta1)
        temp1 = theta1 - alpha * gradient_theta1(x, y, theta0, theta1)
        theta0 = temp0  # 再将temp0和temp1赋值给theta0和theta1
        theta1 = temp1
    return theta0, theta1  # 函数返回theta0和theta1

# 代价函数J(θ)的计算，传入保存样本数据的列表x和y，参数theta0和theta1的值
def costJ(x, y, theta0, theta1):
    sum = 0.0  # 保存sigma求和累加的结果
    m = len(x)  # 保存样本的个数
    for i in range(0, m):  # 遍历每个样本
        # 将样本的预测值与真实值差的平方累加到sum中
        sum += (theta0 + theta1 * x[i] - y[i]) * (theta0 + theta1 * x[i] - y[i])
    return sum / (2 * m)  # 返回sum除以2m

# 样本预测函数predict，函数传入直线参数theta0、theta1和样本特征x
def predict(theta0, theta1, x):
    return theta0 + theta1 * x  # 返回直线方程hθ(x)的预测结果

#引入matplotlib.pyplot库，并取一个更简单的名字p-l-t
import matplotlib.pyplot as plt
import numpy

if __name__ == '__main__':
    # 定义8组样本，将特征面积与样本标记的价格赋值到列表x和y中
    x = [50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0]
    y = [280.0, 305.0, 350.0, 425.0, 480.0, 500.0, 560.0, 630.0]
    alpha = 0.0001  # 迭代速率
    n = 100  # 迭代次数
    # 调用梯度下降函数
    theta0, theta1 = gradient_descent(x, y, alpha, n)
    # 计算3000次迭代后的代价函数J(θ)的值
    cost = costJ(x, y, theta0, theta1)
    print("After %d iterations, the cost is %lf" % (n, cost))
    print("theta0 = %lf theta1 = %lf" % (theta0, theta1))
    # 使用预测函数predict，预测面积为112和110的房价
    print("predict(112) = %lf" % (predict(theta0, theta1, 112)))
    print("predict(110) = %lf" % (predict(theta0, theta1, 110)))

    board = plt.figure()  # 创建一个figure画板对象board
    # 从画板对象中分割出一个1行1列的区域
    # 并取出该区域保存至变量axis
    axis = board.add_subplot(1, 1, 1)
    # 设置一个x轴为0到150、y轴为0到800的坐标系
    # 该坐标系的标题是Linear Regression
    # x轴的名称是area，y轴的名称是price
    axis.set(xlim=[0, 150],
             ylim=[0, 800],
             title='Linear Regression',
             xlabel='area',
             ylabel='price')
    # 传入样本坐标列表、点的标记颜色和形状加号
    # 在画板上画出了8个使用加号表示的独立样本
    plt.scatter(x, y, color='red', marker='+')
    # 在0到150之间，构造出500个相同间距的浮点数，保存至x
    x = numpy.linspace(0, 150, 500)
    h = theta1 * x + theta0  # 直线的函数值
    plt.plot(x, h)  # 画出f1的图像
    plt.show()  # 调用show展示，就会得到一个空的画板



