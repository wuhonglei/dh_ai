import matplotlib.pyplot as plt

board = plt.figure()  # 创建画板对象
# 生成坐标轴，需要传入参数projection = '3d'
axis = board.add_subplot(1, 1, 1, projection='3d')
plt.show() # 展示图形

