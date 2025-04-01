import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from typing import TypedDict

# 在模块级别加载模型，只加载一次
nltk.download('punkt')
nltk.download('stopwords')

custom_stop_words = [
    '-', '/', ',', '[', ']', ')', '(', '|', '&', '【', '*', '%',
    '🔥', '.', '!', '🇸', '🇬', ':', '】', '"', '❤', '★', '$',
    'sg', '{', '}', '<', '>'
]

nlp = spacy.blank("en")  # en_core_web_sm
nltk_stop_words = set(stopwords.words('english') + custom_stop_words)
spacy_stop_words = nlp.Defaults.stop_words.union(nltk_stop_words)


class TokenArgs(TypedDict):
    use_stop_words: bool
    lower: bool


def is_nltk_stop_word(word: str) -> bool:
    return word in nltk_stop_words


def is_spacy_stop_word(word: str) -> bool:
    return word in spacy_stop_words


def tokenize_nltk(text: str, args: TokenArgs) -> list[str]:
    if args['lower']:
        text = text.lower()
    tokens = word_tokenize(text)
    if args['use_stop_words']:
        tokens = [token for token in tokens if not is_nltk_stop_word(token)]
    return tokens


def tokenize_spacy(text: str, args: TokenArgs) -> list[str]:
    if args['lower']:
        text = text.lower()
    doc = nlp(text)
    tokens = [token.text for token in doc]
    if args['use_stop_words']:
        tokens = [token for token in tokens if not is_spacy_stop_word(token)]
    return tokens
