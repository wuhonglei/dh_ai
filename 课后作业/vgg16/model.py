""" 使用 vgg16 模型进行训练，识别 MNIST 手写数字 10 个类别 """

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt


from torchinfo import summary


# 设置 TORCH_HOME 环境变量
if os.path.exists('/mnt/model/nlp/'):
    os.environ['TORCH_HOME'] = '/mnt/model/nlp/pytorch'


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        print(vgg)
        self.feature = vgg.features.requires_grad_(False)
        self.classifier = vgg.classifier.requires_grad_(False)
        self.classifier[-1] = nn.Linear(in_features=4096,
                                        out_features=10).requires_grad_(True)

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # 将单通道转换为三通道
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],   # 使用 ImageNet 的均值和标准差
                         [0.229, 0.224, 0.225])
])

if __name__ == '__main__':
    model = VGG16()
    summary(model, input_size=(1, 3, 224, 224), col_names=(
        "input_size", "output_size", "num_params"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("./models/model_10.pth",
                                     map_location=device, weights_only=True))

    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()  # 保存输出并分离梯度计算图
        return hook

    model.feature[4].register_forward_hook(get_activation('conv1'))
    model.feature[9].register_forward_hook(get_activation('conv2'))
    model.feature[16].register_forward_hook(get_activation('conv3'))
    model.feature[23].register_forward_hook(get_activation('conv4'))
    model.feature[30].register_forward_hook(get_activation('conv5'))

    model.eval()
    img = Image.open('./demo/dog.png')
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for i, (name, output) in enumerate(activation.items()):
        single_img = output[0]
        avg_img = single_img.mean(dim=0)
        axes[i].imshow(avg_img, cmap='gray')
        axes[i].set_title(name)

    plt.show()
