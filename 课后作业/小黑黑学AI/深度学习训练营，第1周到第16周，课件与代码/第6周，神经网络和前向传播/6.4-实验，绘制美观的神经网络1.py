# 在代码中导入networkx和matplotlib
import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph() #使用DiGraph创建一个有向图G

# 网络包括了5个节点
# 第1层的节点编号为1、2，第2层的是3、4、5
G.add_edge(1, 3) #从1向3
G.add_edge(1, 4) #从1向4
G.add_edge(1, 5) #从1向5
G.add_edge(2, 3) #从2向3
G.add_edge(2, 4) #从2向4
G.add_edge(2, 5) #从2向5

# 创建字典pos，字典的key是节点的名称
# 字典的value，是节点所在位置

# 1号和2号节点在一列
# 3、4、5在一列
# 因此设置1和2的x坐标，为0和3、4、5的x坐标为1

# 同一组中的节点，可以均匀的分布在同一列上，
# 所以我们将1和2的y坐标，设置为0.25与0.75，
# 3、4、5的y坐标0.2、0.5、0.8

# {节点的名称: (节点x坐标, 节点y坐标)}
pos = {1: (0, 0.25), #节点1的坐标(0, 0.25)
       2: (0, 0.75), #节点2的坐标(0, 0.75)
       3: (1, 0.2), #节点3的坐标(1, 0.2)
       4: (1, 0.5), #节点4的坐标(1, 0.5)
       5: (1, 0.8)} #节点5的坐标(1, 0.8)


nx.draw(G, #要绘制的图
        pos, #图中节点的坐标
        with_labels = True, #绘制节点的名称
        node_color='white', #节点的颜色
        edgecolors='black', #边的颜色
        linewidths=3, #节点的粗细
        width=2, #边的粗细
        node_size=1000) #节点的大小
plt.show()
