### 中文分词
中文简体或繁体分词时，可以使用 `jieba` 分词库，安装方法如下：
使用方法如下:
```py
import jieba
seg_list = jieba.cut("我来到北京清华大学")
# 输出分词结果 ['我', '来到', '北京', '清华大学']
```

知识点:
- jieba 分词能够将中文和英文分开，但是对于中文和数字之间的分词效果不好，需要自己进行处理

### 如何将程序挂后台运行
参考: https://matpool.com/learn/article/matpool-program-run-nohup

```bash
nohup python -u 123.py > /root/run.log 2>&1 &
```