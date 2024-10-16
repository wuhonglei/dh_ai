## 关键词定义
由于无法预先知道关键词，因此如果文本长度大于 2 时，我们即认为该文本节点包含关键词

## 元素节点文本数量的定义
num_text_nodes = own_text_nodes + child_text_nodes

## 一个商品节点（包含标题和价格或描述）的元素节点
商品节点需要同时满足以下条件：
1. 包含关键词（即文本长度 > 2）
2. 文本节点数量 >= 2 (即至少包含标题或其他属性)


## 元素可划分的定义（即可能包含多个商品的节点, 至少 2 个商品）
1. 包含的关键词节点数量 >= 2
2. 包含的文本节点数量 >= 2 * 2 = 4

## 如何将不可分割的元素划分为组
不可分割的元素可能划分为: 
- 商品组（包含多个商品完整信息）
- 导航组（包含多个导航信息）

问题：那么如何将离散的不可分割元素划分为组呢？
答案：可以根据元素的 full xpath 路径？
例如：
div[1]/section/div[1]
div[1]/section/div[2]
footer[1]/div[1]/span[1]
footer[1]/div[1]/span[3]

可以划分为 2 组：
div[1]/section: [p[1], div[2]]
footer[1]/div[1]: [span[1], span[3]]

划分为组后，在计算组内不可分元素的 tag sequence
1. div: [h3, div]
2. div: [h3, div]
3. div: [h3, div]
longest common sequence: [h3, div]
---
1. li: [span]
2. li: [span]
3. li: [span]
longest common sequence: [span]

将 longest common sequence 相似度阈值 > 50% 的元素视为可能的商品组

### 如何删除噪声组
噪声组：可能是导航组（即非商品组）