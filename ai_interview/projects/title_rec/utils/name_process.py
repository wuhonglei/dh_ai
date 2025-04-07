"""
标题名称预处理:
- 移除表情包: 例如 😄🚗
- 移除移除颜文字 ✦❥
- 移除标题修饰前缀 【free ship】【in stock】
"""

import re
import emoji
import os
import pandas as pd
from termcolor import colored
from nltk import FreqDist
from tqdm import tqdm
from string import punctuation

from config import root_dir, dataset_dir
from vocab import tokenize_nltk, tokenize_spacy
from common import write_file, read_lines, create_dir, is_punctuation


def get_invalid_tokens() -> set[str]:
    txt_path = os.path.join(dataset_dir, '分析结果', 'invalid_tokens.txt')
    invalid_tokens = read_lines(txt_path)
    return set(invalid_tokens)


def get_invalid_pairs() -> list[tuple[str, str]]:
    csv_path = os.path.join(dataset_dir, '分析结果', 'invalid_pair_freq.csv')
    df = pd.read_csv(csv_path)
    pairs = []
    for index, row in df.iterrows():
        pair_str = row['pair']
        start = pair_str[:len(pair_str) // 2]
        end = pair_str[len(pair_str) // 2:]
        pairs.append((start, end))
    return pairs


def compile_invalid_pairs(pairs: list[tuple[str, str]]) -> list[re.Pattern]:
    return [re.compile(f'{re.escape(pair[0])}(.+?){re.escape(pair[1])}')
            for pair in pairs]


def remove_emoji(text, replace=''):
    return emoji.replace_emoji(text, replace)


def remove_non_word(text):
    return re.sub(r'[\u3000-\u303F]', '', text)


def save_invalid_title():
    csv_path = os.path.join(root_dir, 'dataset',
                            'level1_80', 'valid.csv')
    invalid_names = []
    df = pd.read_csv(csv_path)
    total = len(df)
    for index, row in df.iterrows():
        title = row['name']
        # 如果标题开头不是字母，则判断为非法(使用正则)
        if not re.match(r'^[a-zA-Z\d]', title):
            invalid_names.append(title)

    print(f'{len(invalid_names)/total:.2%} 非法标题')
    with open(os.path.join(dataset_dir, 'invalid_names.txt'), 'w') as f:
        f.write('\n'.join(invalid_names))


def analysis_invalid_title():
    print(colored('分析非法标题', 'green'))
    txt_path = os.path.join(dataset_dir, '分析结果', 'invalid_names.txt')
    with open(txt_path, 'r') as f:
        invalid_names = f.readlines()

    data_dict = {
        'title': [],
        'pair': [],
        'prefix': [],
        'after_prefix': [],
    }
    prefix_count = 7
    # 前缀 pair 对, 例如 【Free ship】
    prefix_pairs = [
        ('[', ']'),
        ('(', ')'),
        ('【', '】'),
        ('🔥', '🔥'),
        ('*', '*'),
        ('<', '>'),
        ('{', '}'),
        ('★', '★'),
        ('❤', '❤'),
        ('✨', '✨'),
        ('⭐', '⭐'),
    ]
    # 将 prefix_pairs 转化为正则表达式
    prefix_pairs_regex = [
        re.compile(f'{re.escape(pair[0])}(.+?){re.escape(pair[1])}')
        for pair in prefix_pairs
    ]

    invalid_pairs = []
    invalid_pair_freq = FreqDist()
    invalid_token_freq = FreqDist()

    for name in invalid_names:
        name = name.strip()
        tokens = tokenize_spacy(name)
        data_dict['title'].append(name)
        data_dict['prefix'].append(' '.join(tokens[:prefix_count]))
        data_dict['after_prefix'].append(' '.join(tokens[prefix_count:]))
        # 提取 name 中前缀 pair 对
        for index, regex in enumerate(prefix_pairs_regex):
            match = regex.search(name.lower())
            if match:
                invalid_pairs.append(match.group())
                invalid_pair_freq.update([''.join(prefix_pairs[index])])
                invalid_token_freq.update(tokenize_spacy(match.group(1)))

    # write_file(os.path.join(dataset_dir, 'invalid_pairs.txt'),
    #            '\n'.join(invalid_pairs))
    # 将 invalid_pair_freq.most_common() 保存为 csv
    # df = pd.DataFrame(invalid_pair_freq.most_common(),
    #                   columns=['pair', 'freq'])
    # df.to_csv(os.path.join(dataset_dir, 'invalid_pair_freq.csv'), index=False)
    # 将 invalid_token_freq.most_common() 保存为 csv
    df = pd.DataFrame(invalid_token_freq.most_common(),
                      columns=['token', 'freq'])
    df.to_csv(os.path.join(dataset_dir, 'invalid_token_freq.csv'), index=False)
    print(colored('分析完成', 'green'))


def test_invalid_name():
    """使用统计的非法前缀对和 token 对，测试是否能正确识别非法标题"""
    txt_path = os.path.join(dataset_dir, '分析结果', 'invalid_names.txt')
    invalid_names = read_lines(txt_path)

    # 前缀 pair 对, 例如 【Free ship】
    prefix_pairs = [
        ('[', ']'),
        ('(', ')'),
        ('【', '】'),
        ('🔥', '🔥'),
        ('*', '*'),
        ('<', '>'),
        ('{', '}'),
        ('★', '★'),
        ('❤', '❤'),
        ('✨', '✨'),
        ('⭐', '⭐'),
    ]
    # 将 prefix_pairs 转化为正则表达式
    prefix_pairs_regex = [
        re.compile(f'{re.escape(pair[0])}(.+?){re.escape(pair[1])}')
        for pair in prefix_pairs
    ]
    invalid_tokens = set(read_lines(os.path.join(dataset_dir, '分析结果',
                                                 'invalid_tokens.txt')))
    result = {
        'name': [],  # 标题名称
        'prefix': [],  # 前缀
        'has_invalid_token': [],  # 是否合法
    }
    prefix_count = 7
    for name in tqdm(invalid_names):
        name = name.strip()
        tokens = tokenize_spacy(name.lower())
        result['name'].append(name)
        for index, regex in enumerate(prefix_pairs_regex):
            match = regex.search(' '.join(tokens[:prefix_count]))
            if match:
                result['prefix'].append(match.group())
                tokens_in_prefix = match.group(1).split()
                has_invalid_token = any(
                    token in invalid_tokens for token in tokens_in_prefix)
                result['has_invalid_token'].append(has_invalid_token)
                break
        else:
            result['prefix'].append('')
            result['has_invalid_token'].append(False)

    df = pd.DataFrame(result)
    df.to_csv(os.path.join(dataset_dir, '分析结果', 'invalid_name_result.csv'),
              index=False)


def is_symbol_word(word: str) -> bool:
    """判断 word 是否合法"""
    if len(word) > 3:
        return False

    if is_punctuation(word):
        return False

    if word.isdigit():
        return False

    if re.match(r'^[\w-]+$', word):
        return False

    return True


def analysis_invalid_symbol(column_name: str = 'remove_prefix_emoji'):
    """
    分析 title 中的颜文字
    """
    csv_names = ['valid.csv']
    for csv_name in tqdm(csv_names, desc='clean emoji in name'):
        csv_path = os.path.join(root_dir, 'dataset',
                                'level1_80', 'clean', csv_name)
        df = pd.read_csv(csv_path)
        invalid_word_freq = FreqDist()
        for index, row in tqdm(df.iterrows(), total=len(df), desc=f'analysis invalid symbol in {csv_name}'):
            title = row[column_name]
            tokens = tokenize_spacy(title)
            invalid_tokens = [
                token.strip() for token in tokens
                if is_symbol_word(token.strip())
            ]
            invalid_word_freq.update(invalid_tokens)

    df = pd.DataFrame(invalid_word_freq.most_common(),
                      columns=['token', 'freq'])
    df.to_csv(os.path.join(dataset_dir, '分析结果', 'invalid_symbol_freq.csv'),
              index=False)


def remove_symbol(text: str, replace: str = ''):
    pass
    symbols = read_lines(os.path.join(
        dataset_dir, '分析结果', 'invalid_symbols.txt'))
    for symbol in symbols:
        text = text.replace(symbol, replace)
    return text.strip()


def clean_prefix_in_name():
    csv_names = ['train.csv', 'valid.csv', 'test.csv']
    invalid_tokens = get_invalid_tokens()
    invalid_pairs = get_invalid_pairs()
    invalid_pairs_regex = compile_invalid_pairs(invalid_pairs)
    prefix_count = 7

    for csv_name in tqdm(csv_names, desc='clean prefix in name'):
        csv_path = os.path.join(root_dir, 'dataset',
                                'level1_80', csv_name)
        df = pd.read_csv(csv_path)
        df = df.drop(columns=['clean_name'])
        total = len(df)
        result = {
            'remove_prefix': [],
        }
        for index, row in tqdm(df.iterrows(), total=total, desc=f'clean prefix in {csv_name}'):
            title = row['name']
            tokens = tokenize_spacy(
                title, {'lower': True, 'use_stop_words': False})

            token_str = ' '.join(tokens)
            top_tokens = ' '.join(tokens[:prefix_count])
            for regex in invalid_pairs_regex:
                match = regex.search(top_tokens)
                if match:
                    pair_token_str = match.group(1).split()
                    has_invalid_token = any(
                        token in invalid_tokens for token in pair_token_str)
                    if has_invalid_token:
                        clean_title = token_str.replace(match.group(), '', 1)
                        result['remove_prefix'].append(clean_title)
                        break
            else:
                result['remove_prefix'].append(token_str)

        df['remove_prefix'] = result['remove_prefix']
        file_path = create_dir(os.path.join(root_dir, 'dataset',
                                            'level1_80', 'clean', csv_name))
        df.to_csv(file_path, index=False)


def clean_emoji_in_name():
    csv_names = ['train.csv', 'valid.csv', 'test.csv']
    for csv_name in tqdm(csv_names, desc='clean emoji in name'):
        csv_path = os.path.join(root_dir, 'dataset',
                                'level1_80', 'clean', csv_name)
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=['remove_prefix'])
        df['remove_prefix_emoji'] = df['remove_prefix'].apply(
            lambda x: remove_emoji(x, ' '))
        file_path = create_dir(os.path.join(root_dir, 'dataset',
                                            'level1_80', 'clean', csv_name))
        df.to_csv(file_path, index=False)


def clean_symbol_in_name():
    csv_names = ['train.csv', 'valid.csv', 'test.csv']
    for csv_name in tqdm(csv_names, desc='clean emoji in name'):
        csv_path = os.path.join(root_dir, 'dataset',
                                'level1_80', 'clean', csv_name)
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=['remove_prefix_emoji'])
        df['remove_prefix_emoji_symbol'] = df['remove_prefix_emoji'].apply(
            lambda x: remove_symbol(x, ' '))
        file_path = create_dir(os.path.join(root_dir, 'dataset',
                                            'level1_80', 'clean', csv_name))
        df.to_csv(file_path, index=False)


def tokenize_name():
    csv_names = ['train.csv', 'valid.csv', 'test.csv']
    for csv_name in tqdm(csv_names, desc='tokenize name'):
        csv_path = os.path.join(root_dir, 'dataset',
                                'level1_80', 'clean', csv_name)
        df = pd.read_csv(csv_path)
        df['nltk_tokenized_name'] = df['name'].apply(
            lambda x: ' '.join(tokenize_nltk(x)))
        df['spacy_tokenized_name'] = df['name'].apply(
            lambda x: ' '.join(tokenize_spacy(x)))
        file_path = create_dir(os.path.join(root_dir, 'dataset',
                                            'level1_80', 'clean', csv_name))
        df.to_csv(file_path, index=False)


if __name__ == '__main__':
    tokenize_name()
