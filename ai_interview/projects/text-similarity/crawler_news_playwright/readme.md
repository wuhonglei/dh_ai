## 介绍

使用 playwright 爬取 [新浪财经新闻](https://news.sohu.com/)，爬取后的新闻语料作为评估不同模型性能的语料。

## 新闻类目
- 时政
- 国际
- 军事
- 警法
- 专题
- 公益
- 无人机
- 狐度
- 四象
- 知世

![新浪财经新闻类目](./screenshot/category.png)

获取类目 `js` 代码如下
```js
const categoryContainer = document.querySelector('.nav_header');
const nodes = [...categoryContainer.childNodes].slice(1, -3);
const result = nodes
  .filter((node) => node.nodeType == 1)
  .map((node) => node.querySelector("a"))
  .filter((a) => a && a.href)
  .map((a) => ({
    category: a.textContent.trim(),
    url: a.href,
  }))
  .filter(Boolean);

copy(result)
```

## 爬取规则
- 爬取新浪财经新闻的类目
- 爬取类目下前 10 条的新闻标题、链接、内容

