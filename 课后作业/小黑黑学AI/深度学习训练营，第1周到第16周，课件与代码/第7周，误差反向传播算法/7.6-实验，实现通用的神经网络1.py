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

if __name__ == '__main__':
    network = create_network([3, 4, 2]) #创建一个三层神经网络
    print("network:")
    for key in network: #打印网络中的权值矩阵
        value = network[key]
        print("%s:%s"%(key, value))

























