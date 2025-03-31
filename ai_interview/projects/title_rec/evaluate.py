# NLTK
import spacy
import time
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
# nlp = spacy.load("en_core_web_sm", disable=[
#                  "tagger", "parser", "ner", "lemmatizer"])

nlp = spacy.blank("en")  # 使用空模型，仅保留分词功能

sentences = ["Dr. Smith can't wait."] * 10000
start = time.time()
for s in sentences:
    word_tokenize(s)
print(f"NLTK Time: {time.time() - start:.2f}s")

# spaCy
start = time.time()
for s in sentences:
    doc = nlp(s)
print(f"spaCy Time: {time.time() - start:.2f}s")
