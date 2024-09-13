
from sklearn.datasets import make_gaussian_quantiles
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import linear_model
import numpy

#函数生成平面上的等高数据点
#min-x1到max-x1是平面横轴范围
#min-x2到max-x2是纵轴范围
#根据已训练出的θ参数计算类别结果，不同类别结果对应有不同的高度
def make_contour_point(minx1, maxx1, minx2, maxx2, poly, model):
    #调用mesh-grid生成网格数据点，每个点的距离是0.02
    xx1, xx2 = numpy.meshgrid(numpy.arange(minx1, maxx1, 0.02),
                              numpy.arange(minx2, maxx2, 0.02))
    x1 = xx1.ravel()
    x2 = xx2.ravel()
    X = list()
    for i in range(0, len(x1)):
        #将所有数据点的x1和x2坐标组合在一起，添加到向量X中
        X.append([x1[i], x2[i]])
    polyX = poly.fit_transform(X) #将X转为多项式特征

    z = model.predict(polyX)
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

    #将训练数据转换为2次的多项式特征
    poly = preprocessing.PolynomialFeatures(degree=6)
    polyX = poly.fit_transform(X)
    #生成一个逻辑回归模型实例，构造模型时，不使用任何正则化参数
    model = linear_model.LogisticRegression(penalty = 'l2')
    model.fit(polyX, y) #拟合样本数据
    #基于等高线的方式进行打印，得到二次多项式的决策边界
    x1, x2, z = make_contour_point(-4, 4, -4, 4, poly, model)
    plt.contour(x1, x2, z)
    plt.show()


