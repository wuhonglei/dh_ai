import re
import nltk
from nltk.corpus import stopwords
from pathlib import Path
# fmt: off
import sys
# 将当前目录加入到sys.path
sys.path.append(str(Path(__file__).resolve().parents[0]))
import utils
# fmt: on


def tokenize_sg(text: str) -> list[str]:
    stop_word_list = set(stopwords.words(
        'english') + utils.get_stop_words('./stopwords/common.txt'))

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
    keyword = re.sub(r'(?<=\w)\'s(?=\b)', ' ', keyword)

    """
    / 符号在数字中间，仅用于表示尺寸，因此需要移除左右两边的内容, 例如 3/4 pants mens -> pants mens
    经过验证，该处理在 one-hot svm 不会提升准确率，因此不需要处理
    """
    # keyword = re.sub(r'\d+/\d+', '', keyword)

    token_list = nltk.word_tokenize(keyword, language='english')
    new_token_list = []
    for token in token_list:
        strip_token = token.strip()
        if token not in stop_word_list and len(strip_token) > 1:
            new_token_list.append(strip_token)

    return new_token_list
