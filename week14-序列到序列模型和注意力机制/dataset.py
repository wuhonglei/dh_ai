from datasets import load_dataset
import pandas as pd
import jieba
import spacy
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")


def tokenize_zh(text: str):
    word_list = []
    for token in jieba.cut(text):
        stripped = token.strip()
        if stripped:
            word_list.append(stripped)
    return word_list


def tokenize_en(text: str):
    word_list = []
    for token in nlp(text):
        stripped = token.text.strip()
        if stripped:
            word_list.append(stripped)
    return word_list


dataset = load_dataset("iwslt2017", "iwslt2017-en-zh", trust_remote_code=True)
for name, data in dataset.items():
    print(name, data)
    rows_to_add = []
    data_progress = tqdm(range(len(data)))
    for i in data_progress:
        rows_to_add.append(
            [" ".join(tokenize_en(data[i]['translation']["en"])),
             " ".join(tokenize_zh(data[i]['translation']["zh"]))]
        )

    df = pd.DataFrame(rows_to_add, columns=["en", "zh"])
    df.to_csv(f"./csv/{name}.csv", index=False, sep="\t")
