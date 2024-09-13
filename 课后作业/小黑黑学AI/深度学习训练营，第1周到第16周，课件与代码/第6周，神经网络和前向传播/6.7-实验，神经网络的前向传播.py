import numpy

# 实现神经网络的激活函数，这里选用sigmoid函数
# 函数传入numpy类型的数组x
def sigmoid(x):
    # 根据sigmoid公式，计算结果
    return 1 / (1 + numpy.exp(-x))

# 定义函数create_network，创建神经网络
def create_network():
    network = {} #保存神经网络中的参数
    # 字典的key对应参数的名称，w1、w2、w3和b1、b2、b3
    # 字典的值是参数矩阵
    network['W1'] = numpy.array([[0.1,0.2],
                                 [0.3,0.4],
                                 [0.5,0.6]])
    network['W2'] = numpy.array([[0.1, 0.2, 0.3],
                                 [0.4, 0.5, 0.6]])
    network['W3'] = numpy.array([[0.1,0.2],
                                 [0.3,0.4]])
    network['B1'] = numpy.array([[0.1],
                                 [0.2],
                                 [0.3]])
    network['B2'] = numpy.array([[0.1],
                                 [0.2]])
    network['B3'] = numpy.array([[0.1],
                                 [0.2]])
    return network #返回network


# 在forword中，实现向前传播算法，函数传入神经网络network和特征矩阵X
# X可以保存多个样本的特征值，每个样本对应一个列向量
def forword(network, X):
    # 从network中获取参数矩阵w和b
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    B1, B2, B3 = network['B1'], network['B2'], network['B3']
    # 根据输入的X，第1层权重W1和B1，计算第2层的线性结果z2
    z2 = numpy.dot(W1, X) + B1
    # 将z2代入到激活函数sigmoid中，得到a2
    a2 = sigmoid(z2)
    # 第3层的z3和a3，按照同样的方式计算
    z3 = numpy.dot(W2, a2) + B2
    a3 = sigmoid(z3)
    # 计算神经网络的输出y
    y = numpy.dot(W3, a3) + B3
    return y

if __name__ == '__main__':
    #设置3乘2的矩阵X，保存3个样本，每个样本2个特征
    X = numpy.array([[1.0, 0.5],
                     [2.0, 3.0],
                     [5.0, 6.0]])
    network = create_network() #初始化神经网络
    # 调用forword函数，计算前向传播的结果y
    # 由于输入数据需要是列向量形式
    # 所以需要使用transpose，将X转置后再传入函数
    y = forword(network, numpy.transpose(X))
    print(y)

