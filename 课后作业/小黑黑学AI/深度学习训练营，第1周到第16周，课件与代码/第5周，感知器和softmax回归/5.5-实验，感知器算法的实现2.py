
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist

#绘制一个干净的数学坐标系
#函数传入left、right、down和up
#分别表示坐标轴的左、右、下、上四个边界的最大刻度
def draw_clear_board(left, right, down, up):
    board = plt.figure()  # 创建画板
    axis = axisartist.Subplot(board, 111) #创建坐标轴
    board.add_axes(axis)  # 添加坐标轴
    axis.set_aspect("equal")  # 设置坐标轴刻度
    axis.axis[:].set_visible(False)  # 隐藏原始坐标轴
    # 添加横向的x坐标轴
    axis.axis["x"] = axis.new_floating_axis(0, 0) # 创建悬浮坐标
    axis.axis["x"].set_axisline_style("->") #设置坐标轴的样式
    axis.axis["x"].set_axis_direction("top") #设置x轴的刻度方向
    axis.set_xlim(left, right) #设置x轴的显示范围
    # 添加纵向的y坐标轴，并设置样式
    axis.axis["y"] = axis.new_floating_axis(1, 0)
    axis.axis["y"].set_axisline_style("->")
    axis.axis["y"].set_axis_direction("right")
    axis.set_ylim(down, up)

# 函数传入样本特征samples，标签labels，感知器的参数w和b
# 在函数中，绘制样本数据和感知器模型对应的直线
def plot_samples_and_boundary(samples, labels, w, b):
    # 绘制一个x轴和y轴都是-2到2的坐标系
    draw_clear_board(-2, 2, -2, 2)
    # 遍历全部样本
    for sample, label in zip(samples, labels):
        if label == 0: #如果样本的标记是0
            color = 'r' #绘制红色的叉子
            marker = 'x'
        else: #如果样本的标记是1
            color = 'b'
            marker = 'o'
        plt.scatter(sample[0],
                    sample[1],
                    c=color,
                    marker=marker)

    x = np.linspace(-2, 2, 100) # 在-2到2之间，生成100个x坐标
    y = (-w[0] * x - b) / w[1]  # 根据w和b，计算y坐标
    plt.plot(x, y, 'g') #绘制直线
    plt.show()


def predict(x, w, b): # 函数传入x、w、b
    # 返回感知器的预测结果
    # 如果w乘x+b大于0，结果是1，否则是0
    return int(np.dot(w, x) + b > 0)

# 函数传入样本数据x、y，感知器权重w和b、迭代速率η
def update(x, y, w, b, eta):
    o = predict(x, w, b) #计算感知器预测结果o
    # 根据公式，更新w与b
    w = w + eta * (y - o) * x
    b = b + eta * (y - o)
    return w, b #返回更新后的w和b

if __name__ == '__main__':
    # 初始化w和b和迭代速率η
    w = np.array([1.0, -1.0])
    b = 0.5
    eta = 0.3

    #定义4个代表AND与运算的样本
    samples = np.array([[0.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 0.0],
                        [1.0, 1.0]])
    labels = np.array([0, 0, 0, 1])

    iteration = 0 #设置iteration表示迭代次数
    print(f"======== Iteration {iteration} =========")
    print('weights : ', w)
    print('bias : ', b)
    # 打印一次当前感知器对应的分界面和样本数据
    plot_samples_and_boundary(samples, labels, w, b)

    errors = True
    # 只要errors不为0，有识别错误的样本，那么就一直进行循环迭代
    while errors != 0:
        errors = 0 #每次迭代都要设置errors=0
        iteration += 1
        # 遍历并检查所有的样本
        for sample, label in zip(samples, labels):
            # 当遇到预测错误的样本时
            if predict(sample, w, b) != label:
                # 调用update，更新感知器的权重
                w, b = update(sample, label, w, b, eta)
                errors += 1 #errors加1，记录错误样本的个数

        print(f"======== Iteration {iteration} =========")
        print('weights : ', w)
        print('bias : ', b)
        # 每轮迭代，都要绘打印迭代情况
        plot_samples_and_boundary(samples, labels, w, b)

    # 如果循环结束，说明感知器完成迭代，然后输出此时的迭代结果w和b
    print("")
    print('Final weights:', w)
    print('Final bias:', b)




