import os
import pandas as pd
import re
from tqdm import tqdm
import emoji
from sklearn.feature_extraction.text import TfidfVectorizer


def find_first_category(cat_str: str) -> list[str]:
    """
    寻找第一个category
    """
    match = re.search(r'\[([^\[\]]*?)\]', cat_str)
    if not match:
        return []

    category_name = []
    for cat in re.findall(r'(?:\d+):(.+?)(?=,\s+\d{2,}|$)', match.group(1)):
        category_name.append(cat)

    return category_name


def origin_csv_rename():
    df = pd.read_csv('data/fe_category_20241105-150816.csv')
    df = df.dropna(subset=['name', 'fe_display_categories']).drop_duplicates(
        subset=['name', 'item_id']).reset_index(drop=True)  # type: ignore
    data_list = []
    # 遍历所有行
    progress = tqdm(range(len(df)))
    for i in tqdm(range(len(df))):
        progress.set_description(f'Processing {i}')
        # 获取每行的item_id
        item_id = df.loc[i, 'item_id']
        # 商品名称
        name = df.loc[i, 'name']
        if not item_id or not name:
            continue

        # 获取每行的category
        category: str = df.loc[i, 'fe_display_categories']  # type: ignore
        # 获取每行的category_list
        category_list = find_first_category(category)
        if not category_list:
            continue

        # 获取每行的category_list长度
        data_list.append([item_id, name] + category_list[0:3])

    new_df = pd.DataFrame(
        columns=['item_id', 'name', 'fe_category_1', 'fe_category_2', 'fe_category_3'], data=data_list, dtype=str)

    new_df.to_csv('data/keyword_category.csv', index=False)


def keyword_len():
    df = pd.read_csv('data/keyword_category.csv')
    keyword_len = []
    for keyword in df['Keyword']:
        keyword_len.append(len(keyword.split()))

    new_df = pd.DataFrame(columns=['name_len'], data=keyword_len)
    new_df.to_csv('./data/name_len.csv', index=False)


def clean_name():
    """ 将 name 前缀的特殊字符去掉, 并新增一列 clean_name """
    df = pd.read_csv('data/keyword_category.csv')
    special_name = []
    for name in df['name']:
        if re.match(r'^[^a-zA-Z]', name):
            special_name.append(name)
    # 将特殊字符 special_name 保存到 csv 文件
    new_df = pd.DataFrame(columns=['special_name'], data=special_name)
    new_df.to_csv('./data/special_name.csv', index=False)


def remove_emoji(text, replace=' '):
    return re.sub(r'[★✦✧✩✪✫✯✰♡♥❣❤❥❦❧✿❀❁❃❇❋❉❆❄✻✼♪♫♩♬♭♮♯➜➤➔➛➙➤➥➫➳☺☻☽☾✓✔✕✖✗✘•●◦◉○◍◎◌]', replace, text)


def get_cleaned_name(name: str) -> str:
    name = name.strip()

    # 替换 emoji 表情或 颜文字
    name = emoji.replace_emoji(name, replace=' ')
    name = remove_emoji(name)

    """
    替换 【】, [], **, -><-, <>, (), （） 之间的内容
    """
    exclude_str = '|'.join([
        r'【[^】]+?】',
        r'\[[^\]]+?\]',
        r'\*[^*]+?\*',
        r'->[^<]+?<-',
        r'<[^>]+?>',
        r'\([^)]+?\)',
        r'（[^）]+?）',
        r'\{[^}]+?\}',
        r'★[^★]+?★',
        r'☆[^☆]+?☆',
    ])

    # 最多替换两次
    for _ in range(2):
        name = re.sub(f'^({exclude_str})', '', name.strip()).strip()

    return name


def clean_name_from_special_name():
    """
    移除特殊字符
    """
    df = pd.read_csv('data/keyword_category.csv')
    df['clean_name'] = df['name'].apply(get_cleaned_name)
    df.to_csv('data/keyword_category.csv', index=False)


def clean_name_from_keyword_csv():
    """
    移除特殊字符
    """
    print('Reading data...')
    df = pd.read_csv('data/keyword.csv')
    print('Cleaning data...')
    df['clean_name'] = df['clean_name'].apply(remove_emoji)
    print('Saving data...')
    df.to_csv('data/keyword_new.csv', index=False)


def extract_keywords(vec, feature_names, doc: str, top_n=3):
    tfidf_scores = vec.toarray()[0]
    # 获取非零元素的索引
    indices = tfidf_scores.nonzero()[0]
    # 词和对应的 tf-idf 值的列表
    word_scores = [(feature_names[idx], tfidf_scores[idx])
                   for idx in indices]
    # 按照 tf-idf 值排序
    sorted_word_scores = sorted(
        word_scores, key=lambda x: x[1], reverse=True)

    high_score_words = []
    for word, score in sorted_word_scores[:top_n]:
        high_score_words.append(word)

    # 按照 high word 在 doc 中出现的顺序返回
    high_score_words: list[str] = sorted(
        high_score_words, key=lambda x: doc.lower().find(x.lower()))

    return high_score_words


def extract_keyword_from_name():
    """
    从 name 中提取关键词
    """
    df = pd.read_csv('data/keyword_category.csv')
    df = df.dropna(subset=['clean_name', 'fe_category_1'])
    docs = df['clean_name'].tolist()
    vectorizer = TfidfVectorizer(lowercase=True)
    print('Fitting vectorizer...')
    tfidf_matrix = vectorizer.fit_transform(docs)
    print('Fitting vectorizer done.')
    feature_names = vectorizer.get_feature_names_out()

    data = []
    docs_tqdm = tqdm(enumerate(docs))
    for i, doc in docs_tqdm:
        docs_tqdm.set_description(f'Processing {i}/{len(docs)}')
        keywords_3 = extract_keywords(
            tfidf_matrix[i], feature_names, doc, top_n=3)  # type: ignore
        keywords_5 = extract_keywords(
            tfidf_matrix[i], feature_names, doc, top_n=5)  # type: ignore
        keywords_10 = extract_keywords(
            tfidf_matrix[i], feature_names, doc, top_n=10)  # type: ignore
        Category = df.iloc[i]['fe_category_1']

        data.append([
            doc,
            ' '.join(keywords_3),
            ' '.join(keywords_5),
            ' '.join(keywords_10),
            Category
        ])

    new_df = pd.DataFrame(
        columns=['clean_name', 'keywords_3', 'keywords_5', 'keywords_10', 'Category'], data=data)
    new_df.to_csv('data/keyword.csv', index=False)
    print('Done.')


if __name__ == '__main__':
    # origin_csv_rename()
    # keyword_len()
    # clean_name()
    # clean_name_from_special_name()
    # extract_keyword_from_name()
    clean_name_from_keyword_csv()
    pass
