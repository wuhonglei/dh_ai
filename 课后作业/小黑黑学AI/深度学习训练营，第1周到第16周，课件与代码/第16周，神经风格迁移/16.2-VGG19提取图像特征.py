# 使用卷积层，从内容图像与样式图像中，提取特征图
import torchvision.models as models
from torchvision.models import VGG19_Weights
# 从torchvision.models中，取得已经训练好的VGG-19模型
# 使用.feature获取特征处理的卷积层部分，使用.eval选择评估模式
cnn = models.vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
print(cnn) # 打印获取到的cnn模型


import torch.nn as nn
# 基于卷积层的位置，对这些层进行编号，并专门使用其中的某几个卷积层
cnt = 0 # 表示卷积层的位置
model = nn.Sequential() #设置一个空model
# 遍历cnn.children()，重新命名模型中的每个层
for layer in cnn.children():
    if isinstance(layer, nn.Conv2d): #当遇到卷积层时
        cnt += 1 # 记录conv层的数量
        name = 'conv_{}'.format(cnt)
    # 用这个数量，为其他层如Relu和Pool进行编号与命名
    elif isinstance(layer, nn.ReLU):
        name = 'relu_{}'.format(cnt)
    elif isinstance(layer, nn.MaxPool2d):
        name = 'pool_{}'.format(cnt)
    else:
        name = 'other_{}'.format(cnt)
    # 将每层的名字和该层的对象，添加到model中
    model.add_module(name, layer)
print(model) #打印模型

from PIL import Image
from torchvision import transforms
# 实现图片的读取函数
def image_loader(image_name, target_size):
    image = Image.open(image_name)
    # 重新定义图片的尺寸，保证样式图像与内容图片的大小是一样的
    image = image.resize(target_size)
    transform = transforms.ToTensor() #将图片转为张量
    image = transform(image).unsqueeze(0)
    return image

# 读取内容图像content_img和风格图像style_img
content_size = Image.open("./data/tower.jpg").size
style_img = image_loader("./data/vango.jpg",
                         target_size=content_size)
content_img = image_loader("./data/tower.jpg",
                           target_size=content_size)


import torch
# 为了更明显的观察神经网络提取信息的过程，实现一个获取特征图的函数
def get_feature_map(model, image):
    feature_map = []
    # 遍历models中的各个层
    for name, layer in model.named_children():
        image = layer(image)
        feature_map.append((name, image))
    outputs = list()
    # 计算每层的平均特征图，结果保存在output中
    for name, feature in feature_map:
        feature = feature.squeeze(0)
        gray_scale = torch.sum(feature, 0)
        gray_scale = gray_scale / feature.shape[0]
        outputs.append((name, gray_scale.data.numpy()))
    return outputs


import os
import matplotlib.pyplot as plt
if not os.path.exists('feature_maps'):
        os.makedirs('feature_maps')

# 计算style_img在卷积神经网络中的各层平均特征图
style_maps = get_feature_map(model, style_img)
style_layers = {'conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'}
for name, image in style_maps:
    if name in style_layers:
        fig = plt.imshow(image)
        plt.savefig('./feature_maps/' + "style_"+name + '.jpg')

# 计算content_img在卷积神经网络中的各层平均特征图
content_maps = get_feature_map(model, content_img)
content_layers = {'conv_4'}
for name, image in content_maps:
    if name in content_layers:
        fig = plt.imshow(image)
        plt.savefig('./feature_maps/' + "content_"+name + '.jpg')


