
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

if __name__ == '__main__':
    #调用draw_clear_board函数，绘制一个-2到2的坐标系
    draw_clear_board(-2, 2, -2, 2)
    plt.show()

