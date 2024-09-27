import torch
from torchinfo import summary
from PIL import Image
import matplotlib.pyplot as plt

from vgg19_feature import VGG19FeatureExtractor
from similarity import transform

model = VGG19FeatureExtractor()
model.eval()

activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[str(name)] = output.detach()
    return hook


"""
可视化显示每个卷积快的特征图
"""


def visualize_layer(layer_name):
    model.features[layer_name].register_forward_hook(
        get_activation(layer_name))


layer_names = [4, 9, 18, 27, 36]
for layer_name in layer_names:
    visualize_layer(layer_name)

img = Image.open('./images/turtle.png').convert('RGB')
img = transform(img)
img = img.unsqueeze(0)
output = model(img)

fig, axes = plt.subplots(1, 5, figsize=(20, 5))

for i, layer_name in enumerate(layer_names):
    output = activation[str(layer_name)].squeeze(0)
    axes[i].imshow(output.cpu().numpy().mean(axis=0), cmap='viridis')
    print(f'Layer: {layer_name}')
    print(output.shape)
    print('---------------------')

plt.show()
