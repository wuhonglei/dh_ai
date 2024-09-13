from sklearn import linear_model #使用LinearRegression模型
from sklearn import preprocessing #使用到特征标准化

if __name__ == '__main__':
    # 定义6组样本，每个样本包括4个特征
    # x保存样本的特征矩阵
    x = [[96.79, 2, 1, 2],
         [110.39, 3, 1, 0],
         [70.25, 1, 0, 2],
         [99.96, 2, 1, 1],
         [118.15, 3, 1, 0],
         [115.08, 3, 1, 2]]
    # y保存样本的标签
    y = [287, 343, 199, 298, 340, 350]
    # 设置两个测试数据
    test = [[112, 3, 1, 0],
            [110, 3, 1, 1]]

    #使用linear_model中的LinearRegression，生成model对象
    model = linear_model.LinearRegression()
    model.fit(x, y) #调用fit函数进行训练
    #打印训练的参数与预测结果
    print("model:")
    print(model.intercept_)
    print(model.coef_)
    print("predict:")
    print(model.predict(test))

    #使用StandardScaler生成一个标准化实例
    scaler = preprocessing.StandardScaler()
    nx = scaler.fit_transform(x) #将训练样本的特征向量标准化
    ntest = scaler.transform(test) #将测试样本标准化
    print("After StandardScaler:") #打印标准化后的结果
    print(nx)
    print(ntest)
    #将标准化后的特征向量重新训练
    model = linear_model.LinearRegression()
    model.fit(nx, y)  # 调用fit函数进行训练
    print("model:")
    print(model.intercept_)
    print(model.coef_)
    print("predict:")
    print(model.predict(ntest))

