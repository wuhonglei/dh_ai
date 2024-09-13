#导入torchvision库的transforms模块
#该模块可以很方便的将图片数据转为张量
from torchvision import transforms
#导入PIL库的Image模块，用来读取图片文件
from PIL import Image

if __name__ == '__main__':

    #使用Image打开图片文件，保存到img变量中
    img = Image.open('./catdog.jpg')

    #设置一个转换对象trans，用来将图片数据转为张量
    trans = transforms.Compose([transforms.ToTensor()])
    img = trans(img) #将img转换为张量形式

    #打印img的维度
    print(f"before unsqueeze shape: {img.shape}")
    img = img.unsqueeze(0) #使用unsqueeze对img添加一个维度
    #打印img的维度
    print(f"after unsqueeze shape: {img.shape}")


