from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
from config import test_csv_path, label_id_col, tree_json_path, model_path, label_names_csv_path
import pandas as pd
from tqdm import tqdm


model_name = model_path
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # 显式设置 pad_token
model = AutoModelForCausalLM.from_pretrained(model_name)

classifier = pipeline("text-generation", model=model,
                      tokenizer=tokenizer, max_new_tokens=5)


def classify(text: str, labels: str) -> str:
    prompt = f'Text: "{text}"\nQuestion: Which category does this text belong to? {labels}\nAnswer:'
    result: str = classifier(prompt)[0]['generated_text']  # type: ignore
    # 提取模型输出的类别，可以用正则或简单字符串处理
    predict_name = result.split(
        "Answer:")[-1].strip().split('\n')[0].strip()
    return predict_name


def get_label_map(csv_path: str) -> tuple[dict, dict, list]:
    df = pd.read_csv(csv_path)
    name_col = 'label_names'
    id_col = 'label_ids'
    label_names = df[name_col].unique().tolist()
    id_to_name = {row[id_col]: row[name_col] for _, row in df.iterrows()}
    name_to_id = {row[name_col]: row[id_col] for _, row in df.iterrows()}
    return id_to_name, name_to_id, label_names


id_to_name, name_to_id, label_names = get_label_map(label_names_csv_path)


test_df = pd.read_csv(test_csv_path)
correct = 0
total = len(test_df)
temp_total = 0
progress_bar = tqdm(test_df.iterrows(), total=len(test_df))
for index, row in progress_bar:
    text = row['name']
    label_id = row[label_id_col]
    label_name = id_to_name[label_id]
    predict_name = classify(text, ', '.join(label_names))
    temp_total += 1
    if label_name.lower() == predict_name.lower():
        correct += 1

    # 每隔 10% 打印一次进度
    if temp_total % (total // 10) == 0:
        print(f"Accuracy: {correct / temp_total}")

print(f"Accuracy: {correct / total}")
