## 介绍
### 基线模型

![基线模型效果](https://p.ipic.vip/9otk4y.png)
结论:
svm 在使用 tf-idf 作为特征时，只需要使用 spacy 或 nltk 分词，效果最好, ，准确率最高可以达到 `0.8832`, 但是训练时间比较久，一轮训练的时间大概 `400s`
logistic 回归在使用 BOW 作为特征时，效果最好，准确率最高可以达到 `0.8708`, 训练比较快, 一轮训练时间大概 `18s`

### fasttext 模型

结论:
1. fasttext 模型在使用 spacy 分词时并且移除停用词时，效果最好, ，准确率最高可以达到 `0.8755`，模型大小 `40MB`
2. fasttext 模型使用预训练向量 `crawl-300d-2M.vec` 时，准确率最高可达到 `0.8833`, 模型大小 `264MB`

### fasttext 自定义模型
![fasttext 自定义模型效果](./screenshot/custom_fasttext.png)
使用 remove_spacy_stop_words 列，embedding=200 维，wordGram=2，效果最好，准确率最高可以达到 `0.87916`，模型大小 `65MB`

### textcnn 模型

https://wandb.ai/wuhonglei1017368065-shopee/shopee_title_textcnn_model/table?nw=nwuserwuhonglei1017368065
最好的效果: `0.8416`
模型大小: `24MB`
训练时长: `3min(5epochs)`


### bert 模型

https://wandb.ai/wuhonglei1017368065-shopee/shopee_title_bert_model?nw=nwuserwuhonglei1017368065
对比了 `bert-base-uncased`, `distilbert-base-uncased`, `albert-xlarge-v2` 三个模型
`bert-base-uncased` 最好效果: `0.8848`, 模型大小 `417MB`, 训练时长 6min(3epochs)
`distilbert-base-uncased` 最好效果: `0.8817`, 模型大小 `253MB`, 训练时长 3min(3epochs)
`albert-xlarge-v2` 最好效果: `0.2102`, 模型大小 `224MB`, 训练时长 24min(3epochs)
综上: 
1. `bert-base-uncased` 效果最好，但是模型大小比较大，训练时间比较久
2. `distilbert-base-uncased` 效果其次，模型大小较小，训练时间较短

### vgg 模型

1. vgg16 模型，使用 `valid.csv` 图像训练，使用 `test.csv` 图像测试，效果是 `0.210`
2. vgg19 模型，使用 `valid.csv` 图像训练，使用 `test.csv` 图像测试，效果是 `0.3538`

### resnet 模型

1. resnet101 模型，使用 `valid.csv` 图像训练，使用 `test.csv` 图像测试，效果是 `0.6434`
2. resnet152 模型，使用 `valid.csv` 图像训练，使用 `test.csv` 图像测试，效果是 `0.6283`

### vit 模型
 
使用 timm 加载的 vit 预训练模型，完整名称是 `vit_base_patch16_224.augreg2_in21k_ft_in1k`, learning_rate 设置为 `0.0001` 时，最好的测试集准确率是 `0.6812`.
注意：该预训练模型的图片均值和方差和 imagenet 数据集不一样，需要通过以下方式打印查看

```python
import timm
model = timm.create_model('vit_base_patch16_224', pretrained=True)
print(model.default_cfg)
```

```json
{'url': '', 'hf_hub_id': 'timm/vit_base_patch16_224.augreg2_in21k_ft_in1k', 'architecture': 'vit_base_patch16_224', 'tag': 'augreg2_in21k_ft_in1k', 'custom_load': False, 'input_size': (3, 224, 224), 'fixed_input_size': True, 'interpolation': 'bicubic', 'crop_pct': 0.9, 'crop_mode': 'center', 'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5), 'num_classes': 1000, 'pool_size': None, 'first_conv': 'patch_embed.proj', 'classifier': 'head'}
```

### efficientnet 模型

1. efficientnet_b5 模型，使用 `valid.csv` 图像训练，使用 `test.csv` 图像测试，效果是 `0.7194`
2. efficientnet_b7 模型，使用 `valid.csv` 图像训练，使用 `test.csv` 图像测试，效果是 `0.6904`

### llama 模型
模型名称: `Llama-3.2-1B`

1. `zero-shot` 零样本分类，使用 `test.csv` 测试，效果是 `0.2017`
   - Mac 电脑本地运行时间 `1hour:34min`
   - 4090 显卡运行时间 `40min`
   - 4090 显卡, batch_size=8, 运行时间 `46min`

2. 使用 fasttext 模型，使用联合搜索或者级联搜索，效果差不多，联合搜索最终只生成一个模型，级联搜索最终生成多个模型，准确率大概是 `0.75`, 级联训练使用 beam search k=2 时，能够提高 2% 的准确率
3. 使用 bert 模型
  - 一个模型集成 level1 和 leaf_level 预测，效果大概是 `0.724`
  - 联合搜索，效果大概是 `0.719`
4. 使用 custom fasttext 模型，使用级联搜索，在一个模型同时输出一级标签和叶子标签，准确率是 `0.7196`，此时使用的训练集是 `valid.csv` 测试集是 `test.csv`；使用 `train.csv` 作为训练集时的准确率是 `0.758`