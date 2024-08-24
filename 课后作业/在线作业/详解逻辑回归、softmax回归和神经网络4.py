import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy
import torch
from torch import nn


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.hidden = nn.Linear(in_features=2, out_features=3)
        self.out = nn.Linear(in_features=3, out_features=1)

    def forward(self, x):
        x = self.hidden(x)
        x = torch.sigmoid(x)
        x = self.out(x)
        return x  # 不再应用激活函数


def draw_decision_boundary(minx1, maxx1, minx2, maxx2, model):
    xx1, xx2 = numpy.meshgrid(numpy.arange(minx1, maxx1, 0.02),
                              numpy.arange(minx2, maxx2, 0.02))
    x1s = xx1.ravel()
    x2s = xx2.ravel()
    z = []
    for x1, x2 in zip(x1s, x2s):
        test_point = torch.FloatTensor([[x1, x2]])
        output = model(test_point)
        predicted = torch.sigmoid(output) > 0.5
        z.append(predicted.item())
    z = numpy.array(z).reshape(xx1.shape)
    return xx1, xx2, z


if __name__ == '__main__':
    x, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=1)
    x1 = x[:, 0]
    x2 = x[:, 1]

    plt.scatter(x1[y == 1], x2[y == 1], color='blue', marker='o')
    plt.scatter(x1[y == 0], x2[y == 0], color='red', marker='x')

    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)  # 注意改为float32，与损失函数兼容

    model = Network()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()  # 使用BCEWithLogitsLoss损失函数

    for epoch in range(10000):
        optimizer.zero_grad()
        predict = model(x_tensor)
        loss = criterion(predict.squeeze(), y_tensor)  # 确保维度匹配
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(f'After {epoch} iterations, the loss is {loss.item():.3f}')

    xx1, xx2, z = draw_decision_boundary(-2, 6, -2, 6, model)
    plt.contour(xx1, xx2, z, colors=['orange'])
    plt.show()
