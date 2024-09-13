
import torch
from torch import nn

#函数传入输入张量X和卷积核张量K
#在函数中，计算X和K的互相关运算结果返回
def correlation2d(X, K):
    Xh, Xw = X.shape #保存输入数据的高度和宽度
    Kh, Kw = K.shape #保存卷积核的高度和宽度
    Yh = Xh - Kh + 1 #计算输出数据的高度
    Yw = Xw - Kw + 1 #计算输出数据的宽度
    #只需要将np.zeros函数改为torch.zeros函数
    Y = torch.zeros((Yh, Yw)) #保存输出结果
    # 通过两层循环，计算出数组Y的值
    for i in range(Yh):  # 第一层循环Y的高度
        for j in range(Yw):  # 第二层循环Y的宽度
            # 取出对应位置的输入数据X，保存到sub中
            sub = X[i:i + Kh, j:j + Kw]
            # 计算输出结果Y
            Y[i, j] = (sub * K).sum()
            # 打印出当前的i、j、sub和Y，进行调试
            #print("i = %d j = %d" % (i, j))
            #print(sub)
            #print(Y[i, j])
            #print("")
    return Y #返回结果Y

# 基于pytorch，定义一个卷积层的类
# 继承了nn模块中的Module类
class Conv2D(nn.Module):
    # 构造函数，函数传入卷积核的大小kernel_size
    def __init__(self, kernel_size):
        super().__init__() #运行父类的构造函数
        # 通过torch.rand随机生成kernel_size大小的张量kernel
        kernel = torch.rand(kernel_size)
        # 通过函数nn.Parameter，将该张量转为可训练的参数
        self.weight = nn.Parameter(kernel)

    # 前向传播的计算函数forward，函数传入输入张量x
    def forward(self, x):
        # 计算x和参数weight的互相关运算
        return correlation2d(x, self.weight)



if __name__ == '__main__':
    # 将输入数据img和卷积核kernel的声明方式，修改为张量的形式
    # 需要在声明时特别的指定张量的数据类型是float
    img = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 100, 100, 100, 100, 100, 100, 0, 0, 0],
                        [0, 0, 0, 100, 100, 100, 100, 100, 100, 0, 0, 0],
                        [0, 0, 0, 100, 100, 100, 100, 100, 100, 0, 0, 0],
                        [0, 0, 0, 100, 100, 100, 100, 100, 100, 0, 0, 0],
                        [0, 0, 0, 100, 100, 100, 100, 100, 100, 0, 0, 0],
                        [0, 0, 0, 100, 100, 100, 100, 100, 100, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                        dtype=torch.float)
    # 此处的kernel只是用来计算输出图像Y的
    # 后面会根据img和Y，重新训练出kernel中的参数
    kernel = torch.tensor([[-1, -1, -1],
                           [-1, 8, -1],
                           [-1, -1, -1]],
                           dtype=torch.float)
    # 计算img和kernel的互相关运算，它的结果是输出图像，保存在Y中
    Y = correlation2d(img, kernel)

    conv = Conv2D((3, 3)) # 构造一个3*3的二维卷积层
    #如果学习速率lr过大，梯度下降的过程中就会出现溢出错误
    #如果迭代次数num不足，则无法收敛到最优解
    #此处的学习速率和迭代次数是经过实验得出的合适值
    lr = 1e-7  # 设置学习率为0.0000001
    num = 10000 # 设置迭代次数为10000

    #进入了迭代卷积层参数的循环
    for i in range(num):
        Y_predict = conv(img) # 计算基于当前参数的预测值
        # 根据平方误差，计算预测值和真实值之间的损失值
        loss = (Y_predict - Y) ** 2
        conv.zero_grad() #清空上一轮迭代的梯度
        # 使用backward函数进行反向传播，计算出损失loss关于参数的梯度
        loss.sum().backward()
        # 使用梯度下降算法，更新参数weight的值
        conv.weight.data[:] -= lr * conv.weight.grad

        # 每迭代100轮，就打印一次loss的值进行观察
        if (i + 1) % 100 == 0:
            print("epoch %d loss %.3lf"%(i + 1, loss.sum()))

    # 将训练的结果进行打印
    print(conv.weight.data.reshape((3, 3)))


