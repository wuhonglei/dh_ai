# 函数传入预测结果predict，标签label
from model import TextClassifier
from dataset import collate_batch
from dataset import build_vocab
from dataset import NewsDataset
from torchtext.datasets import AG_NEWS
from torch.utils.data import DataLoader
import pickle
from torch import optim
from torch import nn
import torch


def accuracy(predict, label):
    correct = 0  # 正确的标签数量
    total = 0  # 总标签数量
    for i in range(len(label)):  # 遍历所有的标签
        # 计算预测结果中最大的标签
        max_predict = predict[i].argmax(dim=0)
        if max_predict == label[i]:
            correct += 1  # 正确的数量加1
        total += 1  # 总数加1
    return correct / total  # 返回正确率


if __name__ == '__main__':
    # NLP任务的训练和测试流程，与计算机视觉任务相差不大
    train_data, _ = AG_NEWS()
    dataset = NewsDataset(train_data)
    # 多了一个词表构建和保存
    text_vocab = build_vocab(dataset)  # 词表的创建
    # 打印两个词表长度
    print("text_vocab:", len(text_vocab))
    # 词表的保存
    with open("text_vocab.pkl", "wb") as f:
        pickle.dump(text_vocab, f)

    # 回调函数，用于不同长度的文本进行填充
    def collate(batch): return collate_batch(batch, text_vocab)
    # 小批量读取数据
    dataloader = DataLoader(dataset,
                            batch_size=32,
                            shuffle=True,
                            collate_fn=collate)

    # 定义当前设备
    DEVICE = torch.device('cuda' if torch.cuda.is_available()
                          else 'cpu')
    print("DEVICE = ", DEVICE)

    # 定义模型的必要参数
    vocab_size = len(text_vocab)
    embed_dim = 128
    num_classes = 4
    padding_idx = text_vocab["<pad>"]
    # 定义模型
    model = TextClassifier(vocab_size, embed_dim,
                           num_classes, padding_idx).to(DEVICE)

    model.train()  # 将model调整为训练模式
    optimizer = optim.Adam(model.parameters())  # 定义优化器
    criterion = nn.CrossEntropyLoss()  # 定义交叉熵误差

    print("begin train:")
    n_epoch = 1  # 训练轮数设置为30
    for epoch in range(n_epoch):  # 进行30轮的迭代
        loss_sum = 0  # 用于打印调试信息

        # 内层循环代表了，在一个epoch中
        # 以随机梯度下降的方式，使用dataloader对于数据进行遍历
        # batch_idx表示当前遍历的批次
        # (text, label) 表示这个批次的训练数据和词性标记
        for batch_idx, (text, label) in enumerate(dataloader):
            text = text.to(DEVICE)
            label = label.to(DEVICE)
            optimizer.zero_grad()  # 将梯度清零

            predict = model(text)  # 使用模型model计算text的预测结果

            loss = criterion(predict, label)  # 计算损失
            loss.backward()  # 计算损失函数关于模型参数的梯度
            optimizer.step()  # 更新模型参数

            loss_sum += loss.item()  # 累加当前样本的损失
            # 每训练100个批次，打印一次调试信息
            if (batch_idx + 1) % 100 == 0:
                # 计算当前这一批次的正确率
                acc = accuracy(predict, label)
                print(f"Epoch {epoch + 1}/{n_epoch} "  # 当前的迭代轮数
                      f"| Batch {batch_idx + 1}/{len(dataloader)} "  # 当前的批次
                      f"| Loss: {loss_sum:.4f}"  # 当前这100组数据的累加损失
                      f"| acc: {acc:.4f}")  # 当前批次的正确率
                loss_sum = 0

    # 将训练好的模型保存为文件，文件名为text_classify.pth
    torch.save(model.state_dict(), 'text_classify.pth')
