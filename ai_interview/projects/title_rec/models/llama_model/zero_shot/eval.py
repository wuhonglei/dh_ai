from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
from config import test_csv_path, label_id_col, tree_json_path, model_path
import pandas as pd
from tqdm import tqdm


def load_category_list(path: str):
    with open(path, 'r') as f:
        return json.load(f)['data']['list']


def get_category_by_id(id: int, category_list: list[dict]) -> dict | None:
    for category in category_list:
        if category['id'] == id:
            return category
    return None


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


category_list = load_category_list(tree_json_path)
id_to_name = {category['id']: category['name']
              for category in category_list}
name_to_id = {category['name']: category['id']
              for category in category_list}

test_df = pd.read_csv(test_csv_path)
label_ids = test_df[label_id_col].unique().tolist()
label_names = [id_to_name[id] for id in label_ids]

total = 0
correct = 0
progress_bar = tqdm(test_df.iterrows(), total=len(test_df))
for index, row in progress_bar:
    text = row['name']
    label_id = row[label_id_col]
    label_name = id_to_name[label_id]
    predict_name = classify(text, ', '.join(label_names))
    total += 1
    if label_name.lower() == predict_name.lower():
        correct += 1

print(f"Accuracy: {correct / total}")
