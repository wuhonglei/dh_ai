import torch  
import torch.nn as nn
import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size  
        self.output_size = output_size

        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        
    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.fc(x)
        """
        生成器的最后一个激活函数是tanh, 输出的值在-1到1之间, 可以使生成的
        图像数据更容易匹配到真实数据的分布
        """
        x = torch.tanh(x)
        return x


if __name__ == '__main__':
    noise_size = 100

    netG = Generator(noise_size, 256, 784)
    noise = torch.randn(3, noise_size)
    fake = netG(noise)
    image = fake.detach().view(-1, 28, 28)

    fig, axes = plt.subplots(1, fake.size(0))
    for i, ax in enumerate(axes):
        ax.imshow(image[i], cmap='gray')
        ax.axis('off')
    plt.show()
    