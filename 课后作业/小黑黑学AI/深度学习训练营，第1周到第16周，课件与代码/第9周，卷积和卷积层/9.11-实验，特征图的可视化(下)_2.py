
#导入torchvision库的models模块
from torchvision import models
import torch
import torch.nn as nn

#导入torchvision库的transforms模块
#该模块可以很方便的将图片数据转为张量
from torchvision import transforms
#导入PIL库的Image模块，用来读取图片文件
from PIL import Image

import matplotlib.pyplot as plt
import os

#函数传入模型model，函数返回这个模型中的卷积层对象
def get_model_conv_layers(model):
    convs = list() #设置convs列表，保存卷积层对象
    #保存传入的model模型中的每个子层
    layers = list(model.children())
    for layer in layers: #循环遍历layers
        #如果遍历的layer是Conv2d类型
        if type(layer) == nn.Conv2d:
            convs.append(layer) #将它保存到convs列表中
        #如果遍历的layer是Sequential类型
        elif type(layer) == nn.Sequential:
            #那么还需要继续遍历layer的下一层
            for sub in layer:
                for child in sub.children():
                    #将sub中的Conv2d卷积层找到
                    if type(child) == nn.Conv2d:
                        #同样保存到convs列表中
                        convs.append(child)
    return convs #函数返回convs列表

#函数传入卷积层列表conv_layers，输入图片张量image
#函数返回每个卷积层处理后的平均结果
def get_feature_map(conv_layers, image):
    feature_map = list() #设置feature_map保存特征图
    for layer in conv_layers:
        #遍历每个卷积层，使用卷积层对图像image进行处理
        image = layer(image)
        #将处理后的结果添加到feature_map
        feature_map.append(image)

    outputs = list() #保存最终输出图像的数组
    for i in range(len(feature_map)): #遍历每个特征图
        print("i = %d"%(i))
        print(f"before squeeze shape {feature_map[i].shape}")
        #将表示数据数量的第1个维度去掉
        feature = feature_map[i].squeeze(0)
        print(f"after squeeze shape {feature.shape}")
        #调用torch.sum，将所有输出通道的数据累加到一起
        gray_scale = torch.sum(feature, 0)
        print(f"gray_scale {gray_scale.shape}")
        print("")
        #使用gray_scale除以输出通道的数量feature.shape[0]
        #求出特征图的平均值
        gray_scale = gray_scale / feature.shape[0]
        #将该数据转为numpy数组，添加到outputs
        outputs.append(gray_scale.data.numpy())
    return outputs #返回outputs

if __name__ == '__main__':
    #加载models中的resnet预训练模型
    model = models.resnet18(pretrained=True)
    #提取model中的卷积层
    conv_layers = get_model_conv_layers(model)

    #使用Image打开图片文件，保存到img变量中
    img = Image.open('./catdog.jpg')
    #设置一个转换对象trans，用来将图片数据转为张量
    trans = transforms.Compose([transforms.ToTensor()])
    img = trans(img) #使用trans转换img
    img = img.unsqueeze(0) #使用unsqueeze对img添加一个维度

    #调用get_feature_map，得到每个卷积层的可视化输出结果
    output = get_feature_map(conv_layers, img)

    #创建一个feature_maps文件夹
    if not os.path.exists('feature_maps'):
        os.makedirs('feature_maps')
    for i in range(len(output)):
        #将输出结果以图片形式进行保存
        fig = plt.imshow(output[i])
        plt.savefig('./feature_maps/' + str(i) + '.jpg')



