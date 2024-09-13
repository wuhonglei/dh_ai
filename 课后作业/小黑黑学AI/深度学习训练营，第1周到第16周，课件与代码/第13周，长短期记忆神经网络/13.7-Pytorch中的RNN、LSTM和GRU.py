# 首先导入torch、torch.nn和pad_sequence
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

# 设置3个序列，这些序列中的元素都是三维的数据
# seq1的长度是5
seq1 = torch.tensor([[0.658, 0.033, 0.210, 0.1],
                     [0.152, 0.155, 0.730, 0.2],
                     [0.826, 0.875, 0.834, 0.3],
                     [0.111, 0.222, 0.333, 0.4],
                     [0.456, 0.789, 0.001, 0.5]])
# seq2的长度是2
seq2 = torch.tensor([[0.576, 0.867, 0.880, 0.4],
                     [0.417, 0.988, 0.669, 0.5]])
# seq3的长度是1
seq3 = torch.tensor([[0.458, 0.735, 0.806, 0.6]])


# 使用pad_sequence将3个序列组合成一组数据data
data = pad_sequence([seq1, seq2, seq3], batch_first=True)
# 其中会对短的序列进行填充，填充后的结果如下
print("data:")
print(data)

rnn = nn.RNN(input_size = 4, # 输入数据的维度
             hidden_size = 6, # 隐藏层神经元个数
             batch_first = True)

# outs保存了每个时间步对应的处理结果
# ht是处理完整个输入序列之后的隐藏状态
outs, ht = rnn(seq2) #使用rnn直接处理某一个序列数据
print("RNN:")
print("seq2:")
print(f"outs: {outs}\n")
print(f"ht: {ht}\n")


outs, ht = rnn(data) #使用rnn一起处理三个数据的组合data
print("data:")
print(f"outs: {outs}\n")
print(f"ht: {ht}\n")


# 对于LSTM的使用方法，与RNN的使用方法是一样的
lstm = nn.LSTM(input_size = 4,
             hidden_size = 5,
             batch_first = True)
# LSTM的返回结果多了记忆细胞cell
outs, (ht, cell) = lstm(data)
print("LSTM:")
print(f"outs: {outs}\n")
print(f"ht: {ht}\n")
print(f"cell: {cell}\n")



# 对于GRU的使用方法，与RNN的使用方法是一样的
gru = nn.GRU(input_size = 4,
             hidden_size = 5,
             batch_first = True)
outs, ht = gru(data)
print("GRU:")
print(f"outs: {outs}\n")
print(f"ht: {ht}\n")


