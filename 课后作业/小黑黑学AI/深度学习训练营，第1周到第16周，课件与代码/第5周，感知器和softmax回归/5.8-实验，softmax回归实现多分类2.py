import torch.nn as nn

# 基于nn.Module模块，实现softmax回归模型
# 定义类SoftmaxRegression，继承nn.Module类
class SoftmaxRegression(nn.Module):
    # 函数传入参数n_features和n_classes
    # 代表了输入特征的数量和类别的个数
    def __init__(self, n_features, n_classes):
        # 调用父类 torch.nn.Module 的初始化函数
        super(SoftmaxRegression, self).__init__()
        # 定义线性层linear，规模是n_classes*n_features
        self.linear = nn.Linear(n_features, n_classes)

    # 实现softmax回归的线性层计算
    # 函数传入输入数据x，返回线性层的计算结果
    def forward(self, x):
        return self.linear(x)

if __name__ == '__main__':
    model = SoftmaxRegression(2, 3)
    print("parameters:")
    for p in model.parameters():
        print(f"p.shape = {p.shape} num = {p.numel()}")

