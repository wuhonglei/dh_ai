
#导入torchvision库的models模块
from torchvision import models

if __name__ == '__main__':
    #加载models中的resnet预训练模型
    model = models.resnet18(pretrained=True)
    print(model) #将它打印

