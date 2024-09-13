from torch import nn
import torch.nn.functional as F
import torch

# 定义Attention类表示Attention层
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        # 定义两个线性层attn和v
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    # 函数传入当前时刻的解码器的输出ht和编码器的全部输出E
    def forward(self, ht, E):
        n = E.shape[0] # E中包含的输出数量，也就是输入序列的长度
        ht_expanded = ht.repeat(n, 1, 1) # 将ht复制n次
        # 这样后续才能将ht和每个ei组合，与编码器的n个输出对齐。

        ht_expanded = ht_expanded.transpose(0, 1) # 调整维度的顺序
        E = E.transpose(0, 1) # 调整维度的顺序
        # 将ht_expanded和E组合到一起，得到向量combine
        combine = torch.cat([ht_expanded, E], dim = 2)

        # 将combine输入至线性层attn，使用tanh激活
        # 再输入至线性层v，得到能量值energy
        energy = self.v(torch.tanh(self.attn(combine)))
        # 将energy输入至softmax函数，计算出注意力权重
        attn_weights = F.softmax(energy, dim = 1)
        # 返回调整维度顺序的注意力权重
        return attn_weights.transpose(1, 2)

# 打印计算过程中数据的尺寸变化
def print_forward(model, ht, E):
    print("E: ", E.shape)
    n = E.shape[0]
    ht_expanded = ht.repeat(n, 1, 1)
    print("after repeat: ", ht_expanded.shape)

    ht_expanded = ht_expanded.transpose(0, 1)
    print("after transpose:")
    print("ht_expanded: ", ht_expanded.shape)
    E = E.transpose(0, 1)
    print("E: ", E.shape)

    combine = torch.cat([ht_expanded, E], dim = 2)
    print("combine: ", combine.shape)
    energy = model.v(torch.tanh(model.attn(combine)))
    print("energy: ", energy.shape)
    attn_weights = F.softmax(energy, dim = 1)
    print("attention_weights: ", attn_weights.shape)
    attn_weights = attn_weights.transpose(1, 2)
    print("after transpose: ", attn_weights.shape)


if __name__ == "__main__":
    hidden_size = 3 # 定义隐藏层的维度
    attention = Attention(hidden_size) # 定义注意力机制层
    # 随机生成1个ht，它的维度是1*2*3，代表[序列长度1，批量大小2，特征数量3]
    ht = torch.randn(1, 2, hidden_size)
    # 定义编码器的输出E，它的维度是5*2*3，代表[序列长度5，批量大小2，特征数量3]
    E = torch.randn(5, 2, hidden_size)

    # 将ht和E，输入至attention层，计算注意力权重
    attention_weights = attention(ht, E)
    # 打印ht、E和attention_weights的尺寸
    print("ht: ", ht.shape)
    print("E: ", E.shape)
    print("attention_weights: ", attention_weights.shape)

    # 打印计算过程中数据的尺寸变化
    print("print_forward:")
    print_forward(attention, ht, E)

