import numpy as np

# 函数传入列表layer，保存每层神经元的个数
# 函数会返回一个字典表示的神经网络，并随机的初始化其中的权值。
def create_network(layer):
    network = dict() #保存神经网络参数的字典
    for i in range(1, len(layer)): #循环遍历layer数组
        # 对于第i层的参数wi，它对应一个随机生成的layer(i)*layer(i-1)的矩阵
        network["w" + str(i)] = np.random.random([layer[i], layer[i - 1]])
        # 参数bi，对应一个layer(i)*1的列向量
        network["b" + str(i)] = np.random.random((layer[i], 1))
    return network #返回network

#在神经网络中，使用sigmoid激活函数
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# 函数forward实现神经网络的前向传播
# 函数传入神经网络network和基于列向量表示的特征矩阵x
# 函数返回前向传播过程中得到的全部计算结果
def forward(network, x):
    z = list() #设置列表z和a，保存全部的中间计算过程
    a = list()
    z.append(x) #将x添加至z和a中，即z0=a0=x
    a.append(x)
    # 计算神经网络参数下标的最大值layer
    # 字典中的key的数量除以2的整数部分
    layer = len(network) // 2
    for i in range(1, layer + 1): # 循环向前计算
        # 通过network获取参数矩阵w和b
        w = network["w" + str(i)]
        b = network["b" + str(i)]
        # 每层的线性输出结果为矩阵w乘以上一层的输出a(i-1)加b
        res = np.dot(w, a[i-1]) + b
        z.append(res) # 将该结果添加至z
        # 计算激活值
        if i < layer: # 如果当前层i小于总层数layer
            res = sigmoid(res) # 调用sigmoid计算a
            # 这里其实就是最后一层不使用激活函数
        a.append(res) # 将结果添加至a
    return z, a # 返回z和a


# 实现一个函数，计算sigmoid函数的偏导数
def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

# 设置函数backward，函数传入神经网络network
# 前向传播的计算结果，线性输出z和激活结果a，样本的标记值y
def backward(network, z, a, y):
    layer = len(network) // 2 #计算神经网络的层数layer
    # 设置字典grades保存全部参数的偏导数
    grades = dict() # 字典的key是参数名，字典的值是该参数的偏导数
    delta = dict() #设置delta，保存每层的误差误差变量
    # 最后一层的误差即为神经网络的输出a[layer]与标记y的差
    delta[layer] = a[layer] - y
    # 根据公式，计算E关于最后一层参数的偏导数
    grades["w" + str(layer)] = np.dot(delta[layer], a[layer - 1].T)
    grades["b" + str(layer)] = np.sum(delta[layer], axis=1, keepdims=True)
    for i in range(layer - 1, 0, -1): # 进入反向传播的循环
        WT = network["w" + str(i + 1)].T
        # 在循环中，根据递推公式，计算每一层的误差变量δ
        delta[i] = np.dot(WT, delta[i + 1]) * sigmoid_gradient(z[i])
        # 根据计算出的δ，计算E关于w和E关于b的偏导数
        grades["w" + str(i)] = np.dot(delta[i], a[i - 1].T)
        grades["b" + str(i)] = np.sum(delta[i], axis=1, keepdims=True)
    return grades #返回保存偏导数的字典grades


# 实现一个计算整体代价的函数cost，用来训练中的调试
# 函数传入神经网络的预测值h与真实值y，计算平方误差
def cost(h, y):
    return np.mean(np.square(h - y))


from matplotlib import pyplot as plt

if __name__ == '__main__':
    # 生成训练数据，使用np.arrange生成一个从0到1，步长为0.01，
    # 含有100个数据点的数组，作为正弦函数的输入数据，保存到x中
    x = np.arange(0.0, 1.0, 0.01)
    # 将0到1的x，乘以2π，从单位间隔转换为弧度值，
    # 也就是将x映射到正弦函数的一个完整周期上
    y = 10 * np.sin(2 * np.pi * x) #计算正弦值，保存到y中
    # 将x和y通过reshape函数转为100乘1的数组，
    # 也就是100个(x, y)坐标，代表100个训练数据
    x = x.reshape(1, 100)
    y = y.reshape(1, 100)
    # 将(x, y)组成的数据点，画在画板上
    # 运行程序，会在画板上出现蓝色的正弦函数
    plt.scatter(x, y)


    # 使用create_network，创建一个3层的神经网络
    # 输入层和输出层中有一个神经元，隐藏层有10个神经元
    network = create_network([1, 10, 1])
    layer = len(network) // 2 # 设置layer保存神经网络的层数
    alpha = 0.001 # 保存迭代速率
    h = 0 # 保存预测结果

    for iterate in range(10000): #进行10000轮迭代
        # 调用forward函数，进行前向传播的计算
        z, a = forward(network, x)
        # 调用backward，使用反向传播算法，计算各参数的偏导数
        grades = backward(network, z, a, y)

        for i in range(1, layer + 1):
            # 使用梯度下降的方式，根据梯度的反方向，更新所有的权重w和b
            network["w" + str(i)] -= alpha * grades["w" + str(i)]
            network["b" + str(i)] -= alpha * grades["b" + str(i)]

        if iterate % 1000 == 0: #每1000次迭代
            h = a[layer]  # 计算神经网络的预测结果
            # 打印神经网络的误差cost，作为提示信息
            print("network cost ",cost(h, y))

    plt.scatter(x, h) # 将神经网络预测的结果，使用橙色绘制出来
    plt.show()










