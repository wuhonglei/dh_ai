import numpy
import matplotlib.pyplot as plt
from sklearn import linear_model #使用LinearRegression模型
from sklearn import metrics #计算平方误差

if __name__ == '__main__':
    # 定义8组样本，将特征面积与样本标记的价格赋值到列表x和y中
    # x为特征向量矩阵，特征向量矩阵是8乘1的，也就是8个样本，每个样本有1个特征
    x = [[50.0], [60.0], [70.0], [80.0], [90.0], [100.0], [110.0], [120.0]]
    # y为样本标签
    y = [280.0, 305.0, 350.0, 425.0, 480.0, 500.0, 560.0, 630.0]

    board = plt.figure()  # 创建一个figure画板对象board
    # 从画板对象中分割出一个1行1列的区域，并取出该区域保存至变量axis
    axis = board.add_subplot(1, 1, 1)
    # 设置一个x轴为0到150、y轴为0到800的坐标系
    # 该坐标系的标题是Linear Regression、x轴的名称是area，y轴的名称是price
    axis.set(xlim=[0, 150],
             ylim=[0, 800],
             title='Linear Regression',
             xlabel='area',
             ylabel='price')
    # 传入样本坐标列表、点的标记颜色和形状加号，在画板上画出了8个使用加号表示的独立样本
    plt.scatter(x, y, color='red', marker='+')

    #使用linear_model中的LinearRegression，生成model对象
    model = linear_model.LinearRegression()
    model.fit(x, y) #调用fit函数进行训练
    theta0 = model.intercept_ #设置theta0保存模型的截距
    theta1 = model.coef_[0] #theta1保存模型特征的参数
    h = model.predict(x) #通过predict函数预测训练时使用的8个样本，保存到h中
    cost = metrics.mean_squared_error(y, h) #计算平方误差
    # 将这些关键信息打印
    print("The LinearRegression cost is %lf" % (cost))
    print("theta0 = %lf theta1 = %lf" % (theta0, theta1))
    for i in range(0, len(x)):
        print("x[%d], h = %.2lf, y = %.2lf"%(i, h[i], y[i]))
    # 将截距为theta0，斜率为theta1的直线绘制在画板中
    # 在0到150之间，构造出500个相同间距的浮点数，保存至x
    x = numpy.linspace(0, 150, 500)
    h = theta1 * x + theta0  # 直线的函数值
    plt.plot(x, h)  # 画出f1的图像
    plt.show()  # 调用show展示，就会得到一个空的画板


