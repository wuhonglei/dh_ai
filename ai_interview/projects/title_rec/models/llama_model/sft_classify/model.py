import torch
import torch.nn as nn
from transformers import AutoModel, LlamaForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
from config import model_path


class BaseModel(nn.Module):
    def __init__(self, llama_name: str, num_classes: int, dropout: float = 0.1):
        super(BaseModel, self).__init__()
        self.llama = AutoModel.from_pretrained(llama_name)
        self.fc = nn.Linear(self.llama.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        x = self.llama(input_ids, attention_mask)
        x = x.last_hidden_state[:, 0, :]
        x = self.dropout(x)
        x = self.fc(x)
        return x


def build_model(model_name: str, num_classes: int, pad_token_id):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_classes, pad_token_id=pad_token_id)
    return model


if __name__ == '__main__':
    # model = BaseModel(
    #     num_classes=10, llama_name=model_path)
    # input_data = torch.randint(0, 100, (1, 22))
    # attention_mask = torch.ones(1, 22)

    # summary(model, input_data=input_data, attention_mask=attention_mask)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=10, pad_token_id=tokenizer.pad_token_id)
    # model.config.pad_token_id = tokenizer.pad_token_id
    texts = ['This is a test', 'i love you']
    input_data = tokenizer(texts, return_tensors='pt', padding='max_length',
                           truncation=True, max_length=128)
    output = model(input_ids=input_data['input_ids'],
                   attention_mask=input_data['attention_mask'],)
    print(output)
