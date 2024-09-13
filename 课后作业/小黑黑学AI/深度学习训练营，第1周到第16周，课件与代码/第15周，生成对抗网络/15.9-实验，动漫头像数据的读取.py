from PIL import Image
import os
from torch.utils.data import Dataset
# 设置AnimeDataset继承Dataset，用于读取名字训练数据
class AnimeDataset(Dataset):
    # init函数用于初始化
    # init函数用于初始化，函数传入数据的路径data_dir
    def __init__(self, data_dir, transform):
        self.file_list = list()  # 保存每个训练数据的路径
        # 使用os.listdir，获取data_dir中的全部文件
        files = os.listdir(data_dir)
        for file in files:  # 遍历files
            # 将目录路径与文件名组合为文件路径
            path = os.path.join(data_dir, file)
            # 将path添加到file_list列表
            self.file_list.append(path)
        # 将数据转换对象transform保存到类中
        self.transform = transform
        self.length = len(self.file_list)  # 保存数据的个数

    def __len__(self):
        # 直接返回数据集中的样本数量
        # 重写该方法后可以使用len(dataset)语法，来获取数据集的大小
        return self.length

    # 函数getitem传入索引index
    def __getitem__(self, index):
        file_path = self.file_list[index] #获取数据的路径
        image = Image.open(file_path)
        image = self.transform(image)
        return image


from torch.utils.data import DataLoader
from torchvision import transforms

if __name__ == '__main__':
    # 定义数据转换对象transform
    # 使用transforms.Compose定义数据预处理流水线
    # 在transform添加Resize和ToTensor两个数据处理操作
    trans = transforms.Compose([transforms.Resize(64),
                                transforms.CenterCrop(64),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                                     (0.5, 0.5, 0.5))])

    # 定义CaptchaDataset对象dataset
    dataset = AnimeDataset('./anime-face', trans)
    # 定义数据加载器data_load
    # 其中参数dataset是数据集
    # batch_size=8代表每个小批量数据的大小是8
    # shuffle = True表示每个epoch，都会随机打乱数据的顺序
    dataloader = DataLoader(dataset,
                            batch_size = 64,
                            shuffle = True)

    # 编写一个循环，模拟小批量梯度下降迭代时的数据读取
    # 外层循环，代表了整个训练数据集的迭代轮数，3个epoch就是3轮循环
    # 对于每个epoch，都会遍历全部的训练数据
    for epoch in range(3):
        print("epoch = %d" % (epoch))
        # 内层循环代表了，在一个迭代轮次中，以小批量的方式
        # 使用dataloader对数据进行遍历
        # batch_idx表示当前遍历的批次
        # data和label表示这个批次的训练数据和标记
        for batch_idx, data in enumerate(dataloader):
            print(f"batch_idx = {batch_idx} data = {data.shape}")


