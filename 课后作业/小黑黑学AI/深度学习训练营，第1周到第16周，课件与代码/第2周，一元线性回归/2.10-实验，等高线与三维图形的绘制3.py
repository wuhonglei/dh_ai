import matplotlib.pyplot as plt

if __name__ == '__main__':
    #横坐标和纵坐标分别保存在x和y
    x = [[0, 1, 2, 3],
         [0, 1, 2, 3],
         [0, 1, 2, 3],
         [0, 1, 2, 3]]
    y = [[0, 0, 0, 0],
         [1, 1, 1, 1],
         [2, 2, 2, 2],
         [3, 3, 3, 3]]
    #画出网格点
    plt.plot(x, y, marker='.',  markersize=10, linestyle='')
    plt.show()


