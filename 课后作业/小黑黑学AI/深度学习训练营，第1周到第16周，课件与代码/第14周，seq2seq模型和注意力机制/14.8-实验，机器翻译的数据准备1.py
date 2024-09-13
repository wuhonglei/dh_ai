from datasets import load_dataset

# 通过load_dataset，下载数据集，下载后，会得到数据集对象dataset
dataset = load_dataset("iwslt2017", "iwslt2017-en-zh")

print("dataset path:")
for name in dataset.cache_files:
    # 遍历cache_files，可以打印数据的保存目录
    print("%s %s\n"%(name, dataset.cache_files[name]))

# 计算train和test包含的数据数量
train_num = len(dataset['train'])
test_num = len(dataset['test'])
print("")
# 将它们打印出来
print("train data len: %d"%(train_num))
print("test data len: %d"%(test_num))





















