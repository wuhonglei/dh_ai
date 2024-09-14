from utils import make_dirs
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

# 设置 TORCH_HOME 环境变量
if os.path.exists('/mnt/model/nlp/'):
    os.environ['TORCH_HOME'] = '/mnt/model/nlp/pytorch'


class VGGFeatures(nn.Module):
    def __init__(self):
        super(VGGFeatures, self).__init__()
        # 预训练的 VGG19 模型
        vgg_pretrained = models.vgg19(
            weights=models.VGG19_Weights.DEFAULT).features

        # 定义需要提取的层
        self.content_layers = ['21']  # conv4_2
        # conv1_1, conv2_1, etc.
        self.style_layers = ['0', '5', '10', '19', '28']

        self.model = nn.Sequential()
        self.layers = self.content_layers + self.style_layers
        self.layer_names = []  # 保存需要的层的名字(升序)

        # 组合需要的层
        for name, layer in vgg_pretrained._modules.items():
            self.model.add_module(name, layer)
            if name in self.layers:
                self.layer_names.append(name)

    def forward(self, x):
        features = {}
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.layer_names:
                features[name] = x
        return features


def calc_content_loss(generated_features, content_features, weight):
    content_loss = F.mse_loss(
        generated_features['21'], content_features['21']
    )
    return content_loss * weight


def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    tensor = tensor.view(b * c, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram


def calc_style_loss(generated_features, style_features, style_weight, style_layers_weights):
    style_loss = 0
    for layer, weight in style_layers_weights.items():
        gen_feature = generated_features[layer]
        style_feature = style_features[layer]
        _, c, h, w = gen_feature.size()

        generated_gram = gram_matrix(gen_feature)
        style_gram = gram_matrix(style_feature)
        layer_style_loss = F.mse_loss(generated_gram, style_gram)
        style_loss += layer_style_loss * weight / (c * h * w)
    return style_loss * style_weight


def load_image(image_path, max_size=512, shape=None) -> torch.Tensor:
    image = Image.open(image_path)
    size_one = min(max(image.size), max_size)
    size = (size_one, size_one)
    if shape is not None:
        size = shape

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        # 归一化，与 VGG19 预训练模型的预处理一致
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = transform(image).unsqueeze(0)
    return image


def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(
        1, 3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(
        1, 3, 1, 1).to(tensor.device)
    return tensor * std + mean


def save_img(tensor, filename):
    tensor = denormalize(tensor)
    tensor = tensor.clamp(0, 1)
    image = tensor.squeeze(0).cpu().detach()
    save_image(image, filename)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 加载内容图像和风格图像
content_image = load_image('./data/content.png').to(device)
style_image = load_image('./data/style.png',
                         shape=content_image.shape[-2:]).to(device)

# 提取特征
vgg = VGGFeatures().to(device).eval()
with torch.no_grad():
    content_features = vgg(content_image)
    style_features = vgg(style_image)

# 初始化生成图像（可以使用内容图像的副本）
generated_image = content_image.clone().requires_grad_(True)
with torch.no_grad():
    content_features = vgg(content_image)
    style_features = vgg(style_image)

# 定义优化器
optimizer = torch.optim.LBFGS([generated_image])

style_layers_weights = {
    '0': 1.0,    # conv1_1
    '5': 1.0,    # conv2_1
    '10': 1.0,   # conv3_1
    '19': 1.0,   # conv4_1
    '28': 1.0    # conv5_1
}

# 开始优化
epochs = 100
content_weight = 1
style_weight = 1

make_dirs('./output', remove=True)
epoch = [0]
while epoch[0] < epochs:
    def closure():
        optimizer.zero_grad()
        generated_features = vgg(generated_image)

        content_loss = calc_content_loss(
            generated_features, content_features, content_weight)
        style_loss = calc_style_loss(
            generated_features, style_features, style_weight, style_layers_weights)

        loss = content_loss + style_loss
        epoch[0] += 1
        loss.backward()
        if epoch[0] % 10 == 0:
            save_img(generated_image, f'./output/{epoch[0]}.jpg')
            print(f'Epoch [{epoch[0]}/{epochs}], Loss: {loss.item()}')

        return loss

    optimizer.step(closure)
