## 面试准备
面试经验参考: https://github.com/amusi/Deep-Learning-Interview-Book/blob/master/docs/%E9%9D%A2%E8%AF%95%E7%BB%8F%E9%AA%8C.md
大模型 & 深度学习相关: https://github.com/315386775/DeepLearing-Interview-Awesome-2024

### 处理后的文件

- 面试经验合集: `python_crawler/merged_data/面试经验.md`
- boss直聘: `python_crawler/merged_data/boss直聘_0.md`

### 分析合集文件的 prompt 提示词
提示词的生成来源于字节 [火山方舟](https://console.volcengine.com/ark/region:ark+cn-beijing/autope/startup)
![](https://p.ipic.vip/tiurce.png)

#### 1. 面试经验分析提示词如下:

```txt
你的任务是解析 markdown 文件，从中提取你需要为 NLP 算法工程师面试准备的知识点。
请按照以下步骤进行解析：
1. 仔细通读整个 markdown 文件。
2. 识别文件中与 NLP 算法工程相关的关键概念、技术、理论、算法等内容。
3. 对识别出的知识点进行整理和归纳，去除重复内容。
4. 对于高频知识点，需要在知识点后增加标识
5. 对于 NLP 知识点，需要在知识点后增加标识
```

对话链接: https://monica.im/home/chat/Monica/monica?convId=conv%3Afac78752-e5ee-4d35-bcfd-7422f4b982b3
![](https://p.ipic.vip/7pya4j.png)


#### 2. boss直聘分析提示词如下:
```txt
你的任务是解析 markdown 文件中关于 NLP 算法工程师职位的职位要求、薪资水平和任职资格。请仔细阅读以下 markdown 文件内容：
在解析时，请按照以下步骤进行：
1. 仔细阅读整个 markdown 文件，识别与职位要求、薪资水平和任职资格相关的内容。
2. 提取相关信息，进行概括和整理。
3. 确保提取的信息准确、完整。

请在<解析结果>标签内输出你的解析结果，格式如下：
<职位要求>
[在此详细列出职位要求]
</职位要求>
<薪资水平>
[在此详细列出薪资水平信息]
</薪资水平>
<任职资格>
[在此详细列出任职资格]
</任职资格>
如果文件中没有相关信息，请在相应标签内注明“未提及”。
```

对话链接: https://monica.im/home/chat/Monica/monica?convId=conv%3A5accf9a6-3de8-4eff-8783-052b41ef4cf2
![](https://p.ipic.vip/hir1km.png)