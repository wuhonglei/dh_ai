#设函数值hθ(x)的计算函数
#函数传入参数theta、样本特征向量x和特征的个数n
def hypothesis(theta, x, n):
    h = 0.0 #保存预测结果
    for i in range(0, n + 1):
        #将theta-i和xi的乘积累加到h中
        h += theta[i] * x[i]
    return h #返回预测结果h

#J(θ)关于theta-j的偏导数计算函数
#函数传入样本特征矩阵x和标记y，特征的参数列表theta，特征数n
#样本数量m和待求偏导数的参数下标j
def gradient_thetaj(x, y, theta, n, m, j):
    sum = 0.0 #保存sigma求和累加的结果
    for i in range(0, m): #遍历m个样本
        h = hypothesis(theta, x[i], n) #求出样本的预测值h
        #计算预测值与真实值的差，再乘以第i个样本的第j个特征值x[i][j]
        #将结果累加到sum
        sum += (h - y[i]) * x[i][j]
    return sum / m #返回累加结果sum除以样本的个数m

#梯度下降的迭代函数
#函数传入样本特征矩阵x和样本标签y，特征数n，样本数量m
#迭代速率alpha和迭代次数iterate
def gradient_descent(x, y, n, m, alpha, iterate):
    theta = [0] * (n + 1) #初始化参数列表theta，长度为n+1
    for i in range(0, iterate): #梯度下降的迭代循环
        temp = [0] * (n + 1)
        #使用变量j，同时对theta0到theta-n这n+1个参数进行更新
        for j in range(0, n + 1):
            #通过临时变量列表temp，先保存一次梯度下降后的结果
            #在迭代的过程中调用theta-j的偏导数计算函数
            temp[j] = theta[j] - alpha * gradient_thetaj(x, y, theta, n, m, j)
        #将列表temp赋值给列表theta
        for j in range(0, n + 1):
            theta[j] = temp[j]
    return theta #函数返回参数列表theta
    
#实现代价函数J(θ)的计算函数costJ
#函数传入样本的特征矩阵x和标签y，参数列表theta，特征个数n和样本个数m
def costJ(x, y, theta, n, m):
    sum = 0.0 #定义累加结果
    for i in range(0, m): #遍历每个样本
        h = hypothesis(theta, x[i], n)
        #将样本的预测值与真实值差的平方累加到sum中
        sum += (h - y[i]) * (h - y[i])
    return sum / (2 * m) #返回sum除以2m
    

if __name__ == '__main__':
    m = 6 #6个样本
    n = 4 #每个样本4个特征
    alpha = 0.0001 #迭代速率
    iterate = 1500 #迭代次数iterate
    #样本特征矩阵x
    #需要给特征矩阵补一列，即特征x0的值，每个样本的x0都为1
    x = [[1, 96.79, 2, 1, 2],
         [1, 110.39, 3, 1, 0],
         [1, 70.25, 1, 0, 2],
         [1, 99.96, 2, 1, 1],
         [1, 118.15, 3, 1, 0],
         [1, 115.08, 3, 1, 2]]
    #样本标签向量y
    y = [287, 343, 199, 298, 340, 350]


    theta = gradient_descent(x, y, n, m, alpha, iterate)
    costJ = costJ(x, y, theta, n, m)
    #打印参数列表与代价值
    print("After %d iterate, theta is:"%(iterate))
    for i in range(0, len(theta)):
        print("theta[%d] = %.3lf"%(i, theta[i]))
    print("Cost J is %lf"%(costJ))


    #打印6个训练样本的预测值与真实值进行比对
    print("Check h and y:")
    for i in range(0, m):
        h = hypothesis(theta, x[i], n)
        print("i = %d h = %.3f y = %d"%(i, h, y[i]))

    test1 = [1, 112, 3, 1, 0]
    test2 = [1, 110, 3, 1, 1]
    #打印两个测试样本的结果
    print("test1 = %.3f"%(hypothesis(theta, test1, n)))
    print("test2 = %.3f" % (hypothesis(theta, test2, n)))


