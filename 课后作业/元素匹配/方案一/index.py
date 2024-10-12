import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
from lxml import etree

site = 'www.popmart.com'

# 1. 解析网页
with open(f'{site}/index.html', 'r', encoding='utf-8') as file:
    html_content = file.read()
tree = etree.HTML(html_content)

# 2. 定义特征提取函数


def extract_features(element):
    features = {}
    features['tag_name'] = element.name
    features['class_list'] = ' '.join(element.get('class', []))
    features['id'] = element.get('id', '')
    features['text_length'] = len(element.get_text(strip=True))
    features['dom_depth'] = len(list(element.parents))
    features['child_count'] = len(
        [child for child in element.children if child.name])
    features['dom_path'] = ' > '.join(
        reversed([parent.name for parent in element.parents if parent.name]))
    return features


# 3. 获取选定元素和候选元素
# 假设 selected_elements 是用户选定的元素列表
selected_elements_xpath = [
    '/html/body/div/div/div/div[2]/div[1]/div[3]/div[2]/div[1]/div/a/div[2]'
]
selected_elements = [
    tree.xpath(xpath)[0] for xpath in selected_elements_xpath
]
candidate_elements = []
for selected_xpath in selected_elements_xpath:
    xpath = selected_xpath.split('/')[-1].split('[')[0]
    candidate_elements.extend(tree.xpath(xpath))

# 4. 提取特征
selected_features = [extract_features(elem) for elem in selected_elements]
candidate_features = [extract_features(elem) for elem in candidate_elements]

# 5. 特征向量化
df_selected = pd.DataFrame(selected_features)
df_candidates = pd.DataFrame(candidate_features)
df_all = pd.concat([df_selected, df_candidates], ignore_index=True)

# 类别特征编码
categorical_features = ['tag_name', 'class_list', 'id', 'dom_path']
encoder = OneHotEncoder(handle_unknown='ignore')
encoded_categorical = encoder.fit_transform(df_all[categorical_features])

# 数值特征
numerical_features = ['text_length', 'dom_depth', 'child_count']
numerical_data = df_all[numerical_features].fillna(0).values

# 合并特征
X_all = hstack([encoded_categorical, numerical_data])
X_selected = X_all[:len(df_selected)]
X_candidates = X_all[len(df_selected):]

# 6. 计算相似度
similarity_matrix = cosine_similarity(X_candidates, X_selected)
similarity_scores = np.max(similarity_matrix, axis=1)

# 7. 筛选相似元素
threshold = 0.8
similar_indices = np.where(similarity_scores >= threshold)[0]
similar_elements = [candidate_elements[i] for i in similar_indices]

# 8. 输出结果
for elem in similar_elements:
    print(elem.get_text(strip=True))
