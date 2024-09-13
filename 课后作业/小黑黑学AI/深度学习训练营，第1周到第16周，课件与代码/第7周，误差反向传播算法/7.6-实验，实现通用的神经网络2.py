import numpy as np

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


if __name__ == '__main__':
    # 设置神经网络network
    network = {'w1': np.array([[0.1, 0.2],
                               [0.3, 0.4],
                               [0.5, 0.6]]),
               'w2': np.array([[0.1, 0.2, 0.3],
                               [0.4, 0.5, 0.6]]),
               'w3': np.array([[0.1, 0.2],
                               [0.3, 0.4]]),
               'b1': np.array([[0.1],
                               [0.2],
                               [0.3]]),
               'b2': np.array([[0.1],
                               [0.2]]),
               'b3': np.array([[0.1],
                               [0.2]])}
    # 3个样本组成特征向量矩阵
    x = np.array([[1.0, 0.5],
                  [2.0, 3.0],
                  [5.0, 6.0]])

    # 调用forward函数进行测试
    z, a = forward(network, x.T)
    # 打印结果
    print("z:")
    for i in range(0, len(z)):
        print("z[%d] = %s" % (i, str(z[i])))
    print("a:")
    for i in range(0, len(a)):
        print("a[%d] = %s" % (i, str(a[i])))







