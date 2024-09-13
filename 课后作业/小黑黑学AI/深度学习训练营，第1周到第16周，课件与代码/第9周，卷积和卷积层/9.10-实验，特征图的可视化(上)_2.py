
#导入torchvision库的models模块
from torchvision import models
import torch.nn as nn

#函数传入模型model，函数返回这个模型中的卷积层对象
def get_model_conv_layers(model):
    convs = list() #设置convs列表，保存卷积层对象
    #设置layers，保存传入的model模型中的每个子层
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


if __name__ == '__main__':
    #加载models中的resnet预训练模型
    model = models.resnet18(pretrained=True)
    #提取model中的卷积层
    conv_layers = get_model_conv_layers(model)
    #打印提取出的结果
    print("conv_layers num : %d"%(len(conv_layers)))
    for conv in conv_layers:
        print(conv)

