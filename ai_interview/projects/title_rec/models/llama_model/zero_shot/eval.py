from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
from config import test_csv_path, name_col, label_id_col, tree_json_path, model_path, label_names_csv_path
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader


model_name = model_path
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # 显式设置 pad_token
model = AutoModelForCausalLM.from_pretrained(model_name)

classifier = pipeline("text-generation", model=model,
                      tokenizer=tokenizer, max_new_tokens=5)


def batch_classify(texts: list[str], categories: str) -> list[str]:
    prompts = [f'Text: "{text}"\nQuestion: Which category does this text belong to? {categories}\nAnswer:'
               for text in texts]
    results: list[str] = classifier(prompts)  # type: ignore
    # 提取模型输出的类别，可以用正则或简单字符串处理
    predict_names = [result.split(
        "Answer:")[-1].strip().split('\n')[0].strip() for result in results]
    return predict_names


def get_label_map(csv_path: str) -> tuple[dict, dict, list]:
    df = pd.read_csv(csv_path)
    name_col = 'label_names'
    id_col = 'label_ids'
    label_names = df[name_col].unique().tolist()
    id_to_name = {row[id_col]: row[name_col] for _, row in df.iterrows()}
    name_to_id = {row[name_col]: row[id_col] for _, row in df.iterrows()}
    return id_to_name, name_to_id, label_names


id_to_name, name_to_id, label_names = get_label_map(label_names_csv_path)


class TextDataset(Dataset):
    def __init__(self, csv_path: str, id_to_name: dict[int, str]):
        self.df = pd.read_csv(csv_path)
        self.texts = self.df[name_col].tolist()
        self.label_ids = self.df[label_id_col].tolist()
        self.id_to_name = id_to_name

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx) -> tuple[str, str]:
        text = self.texts[idx]
        label_id = self.label_ids[idx]
        label_name = self.id_to_name[label_id]
        return text, label_name


dataset = TextDataset(csv_path=test_csv_path, id_to_name=id_to_name)
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

correct = 0
temp_total = 0
total = len(dataset)
progress_bar = tqdm(dataloader, total=len(dataloader))
for batch in progress_bar:
    batch_texts: list[str] = batch["text"]
    batch_labels: list[str] = batch["label_name"]
    predict_names = batch_classify(batch_texts, ', '.join(label_names))
    temp_total += len(batch_texts)
    for label, predict in zip(batch_labels, predict_names):
        if label.lower() == predict.lower():
            correct += 1

    # 每隔 10% 打印一次进度
    if temp_total % (total // 10) == 0:
        print(f"Accuracy: {correct / temp_total}")

print(f"Accuracy: {correct / total}")
