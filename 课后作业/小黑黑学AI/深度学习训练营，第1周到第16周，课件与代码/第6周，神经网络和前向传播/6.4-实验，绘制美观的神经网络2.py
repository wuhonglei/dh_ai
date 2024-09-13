import matplotlib.pyplot as plt
import networkx as nx

# 根据传入的输入层、隐藏层、输出层的神经元数量，绘制对应的神经网络
def draw_network_digraph(input_num, hidden_num, output_num):
    G = nx.DiGraph() # 创建一个图G
    # 连接输入层和隐藏层之间的边
    for i in range(input_num):
        for j in range(hidden_num):
            G.add_edge(i, input_num + j)
    # 连接隐藏层和输出层之间的边
    for i in range(hidden_num):
        for j in range(output_num):
            G.add_edge(input_num + i,
                       input_num + hidden_num + j)


    pos = dict() # 计算每个节点的坐标pos
    # 节点的坐标，(x, y)设置为:
    # (0, i - input_num / 2)
    # (1, i - hidden_num / 2)
    # (2, i - output_num / 2)
    # 根据每一层的节点数量，将节点从中间，向两边分布
    for i in range(0, input_num):
        pos[i] = (0, i - input_num / 2)
    for i in range(0, hidden_num):
        hidden = i + input_num
        pos[hidden] = (1, i - hidden_num / 2)
    for i in range(0, output_num):
        output = i + input_num + hidden_num
        pos[output] = (2, i - output_num / 2)

    # 调用nx.draw，绘制神经网络
    nx.draw(G,
            pos,
            node_color='white',
            edgecolors='black',
            linewidths=2,
            node_size=1000)

if __name__ == '__main__':
    # 多尝试几组参数，绘制不同结构的神经网络
    draw_network_digraph(3, 5, 2)
    plt.show()
    draw_network_digraph(5, 2, 6)
    plt.show()
    draw_network_digraph(1, 10, 1)
    plt.show()














