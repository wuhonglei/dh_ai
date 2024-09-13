from sklearn import preprocessing

if __name__ == '__main__':
    #设置包含两个特征、3个样本的矩阵X
    X = [[2, 3], [3, 4], [4, 5]]
    #调用PolynomialFeatures，生成2次和3次多项式生成器
    poly2 = preprocessing.PolynomialFeatures(degree=2)
    poly3 = preprocessing.PolynomialFeatures(degree=3)
    #将X进行转换
    X2 = poly2.fit_transform(X)
    X3 = poly3.fit_transform(X)
    #打印
    print("X =", X)
    print("X2 =", X2)
    print("X3 =", X3)

