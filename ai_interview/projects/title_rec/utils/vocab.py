from nltk.tokenize import word_tokenize
import spacy
import nltk


# 在模块级别加载模型，只加载一次
nlp = spacy.blank("en")  # en_core_web_sm
nltk.download('punkt')


def tokenize(text: str) -> list[str]:
    return word_tokenize(text)


def tokenize_spacy(text: str) -> list[str]:
    doc = nlp(text)
    return [token.text for token in doc]
