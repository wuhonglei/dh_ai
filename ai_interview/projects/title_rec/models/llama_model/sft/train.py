import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, PreTrainedTokenizer, PreTrainedTokenizerFast
from torch.utils.data import DataLoader

from config import train_csv_path, test_csv_path, name_col, label_id_col, label_names_csv_path, model_path
from dataset import TitleRecDataset


def get_label_map(csv_path: str) -> tuple[dict, dict, list]:
    df = pd.read_csv(csv_path)
    name_col = 'label_names'
    id_col = 'label_ids'
    categories = df[name_col].unique().tolist()
    id_to_name = {row[id_col]: row[name_col] for _, row in df.iterrows()}
    name_to_id = {row[name_col]: row[id_col] for _, row in df.iterrows()}
    return id_to_name, name_to_id, categories


def build_dataloader(csv_path: str, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, name_col: str, label_id_col: str, id_to_name: dict[int, str], categories: list[str]):
    dataset = TitleRecDataset(
        csv_path, tokenizer, name_col, label_id_col, id_to_name, categories)
    return DataLoader(dataset, batch_size=8, shuffle=True)


model_name = model_path
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

id_to_name, name_to_id, categories = get_label_map(label_names_csv_path)
train_dataloader = build_dataloader(
    train_csv_path, tokenizer, name_col, label_id_col, id_to_name, categories)
test_dataloader = build_dataloader(
    test_csv_path, tokenizer, name_col, label_id_col, id_to_name, categories)


training_args = TrainingArguments(
    per_device_train_batch_size=8,
    num_train_epochs=3,
    output_dir="./llama3-classifier"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataloader.dataset,
    eval_dataset=test_dataloader.dataset
)

trainer.train()
