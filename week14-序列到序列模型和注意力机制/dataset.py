from datasets import load_dataset
import pandas as pd
import jieba
import spacy

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
    df = pd.DataFrame(columns=["en", "zh"])
    for i in range(len(data)):
        df.loc[len(df)] = [
            " ".join(tokenize_en(data[i]['translation']["en"])),
            " ".join(tokenize_zh(data[i]['translation']["zh"]))
        ]

    df.to_csv(f"{name}.csv", index=False, sep="\t")
