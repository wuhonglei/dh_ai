# 函数make_train_data传入raw_text
# 将raw_text构造为上下文的训练数据
def make_train_data(raw_text):
    data = [] #保存训练数据
    window = 2 #设置上下文窗口window = 2
    # 遍历raw_text
    for i in range(window, len(raw_text) - window):
        # 构造上下文，保存在context中
        context = [raw_text[i - 2], raw_text[i - 1],
                   raw_text[i + 1], raw_text[i + 2]]
        target = raw_text[i] #目标词保存在target中
        # 将context和target一起添加到data
        data.append((context, target))
    return data #返回data


if __name__ == '__main__':
    # 设置raw_text包含a、b、c、b、a、d，6个词语
    raw_text = """a b c b a d""".split()
    # raw_text =  ['a', 'b', 'c', 'b', 'a', 'd']
    print("raw_text = ", raw_text)
    # 构造出两组上下文
    data = make_train_data(raw_text)
    # content =  ['a', 'b', 'b', 'a']
    # target =  c
    # content =  ['b', 'c', 'a', 'd']
    # target =  b
    for content, target in data:
        print("content = ", content)
        print("target = ", target)











