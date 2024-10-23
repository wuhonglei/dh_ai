import re
import spacy
from .utils import get_stop_words, add_space_between_unit, contain_chinese_text, is_one_ascii, is_float, is_number
from .tw import tokenize_tw

nlp = spacy.load('en_core_web_sm')


def tokenize_sg(text: str) -> list[str]:
    stop_word_list = set(list(nlp.Defaults.stop_words) +
                         get_stop_words('./stopwords/common.txt') + get_stop_words('./stopwords/sg.txt'))

    """
    + 符号在 keyword 中表示语义或者连接关系，需要替换为空格, 例如 sharp+microwave -> sharp microwave
    + 符号在 keyword 表示品牌名称，移除后不影响语义，例如 xiaomi x10+ -> xiaomi x10
    """
    keyword = re.sub('[+]', ' ', text)

    """
    & 符号在 keyword 中表示品牌名称，需要替换为 _， 避免被粉刺, 例如 charles & keith singapore -> charles_keith singapore
    """
    keyword = re.sub(r'(?<=\w)\s*&\s*(?=\w)', '_', keyword)

    """
    xxx's 符号在 keyword 中表示 "什么什么的"，需要替换为空格, 例如 swisse men's vitality -> swisse men vitality
    经过验证，该处理在 one-hot svm 不会提升准确率，因此不需要处理
    """
    keyword = re.sub(r'(?<=\w)(\\)?\'s(?=\b)', ' ', keyword)

    """
    / 符号在数字中间，仅用于表示尺寸，因此需要移除左右两边的内容, 例如 3/4 pants mens -> pants mens
    经过验证，该处理在 one-hot svm 不会提升准确率，因此不需要处理
    """
    keyword = re.sub(r'\d+/\d+', '', keyword)

    keyword = add_space_between_unit(keyword, ['inch'])

    new_token_list = []
    doc = nlp(keyword)
    for token in doc:
        text = (token.text).strip()
        text_lemma = token.lemma_
        if not text or text in stop_word_list or token.is_punct or is_float(text) or is_number(text):
            continue
        if contain_chinese_text(text):
            # 中文文本，使用 jieba 分词
            new_token_list.extend(tokenize_tw(text))
        elif not is_one_ascii(text_lemma):
            # 过滤单个英文字符
            new_token_list.append(text_lemma)

    return new_token_list
