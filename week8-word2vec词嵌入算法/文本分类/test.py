import pickle
from dataset import NewsDataset
from model import TextClassifier
from torchtext.datasets import AG_NEWS
import torch


def test_sample(model, text_vocab, text, label):
    # 实现对一个样本text的预测
    text_tokens = [text_vocab[token] for token in text]
    text_tensor = torch.tensor(text_tokens, dtype=torch.long).unsqueeze(0)
    predict = model(text_tensor)
    predict_label = predict.argmax(-1).squeeze(0).item()
    return predict_label == label-1


if __name__ == '__main__':
    # 读入text_vocab词汇表
    with open("text_vocab.pkl", "rb") as f:
        text_vocab = pickle.load(f)

    # 定义模型的必要参数
    vocab_size = len(text_vocab)
    embed_dim = 128
    num_classes = 4
    padding_idx = text_vocab["<pad>"]
    # 定义模型
    model = TextClassifier(vocab_size, embed_dim, num_classes, padding_idx)

    # 加载已经训练好的模型
    model.load_state_dict(torch.load('text_classify.pth'))
    model.eval()  # 将模型设置为测试模式

    _, test_data = AG_NEWS()
    dataset = NewsDataset(test_data)

    print("total test num: %d" % (len(dataset)))
    correct = 0  # 正确标签的总数
    all_num = 0  # 标签总数
    # 遍历全部的测试样本
    for i in range(len(dataset)):
        text = dataset[i][0]
        label = dataset[i][1]
        # 对于每个测试样本，调用函数test_sample测试效果
        if test_sample(model, text_vocab, text, label):
            correct += 1
        all_num += 1

    acc = correct / all_num
    # 打印总的测试效果
    print("accuracy: %d/%d = %.3lf" % (correct, all_num, acc))
