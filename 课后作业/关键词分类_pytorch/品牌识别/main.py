from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# 加载预训练的模型和分词器
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained(
    "dslim/bert-base-NER")

# 创建NER管道
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

# 输入关键词
keywords = ["iphone 15", "nike shoes", "macbook pro", "adidas t-shirt"]

# 识别品牌名称
for keyword in keywords:
    ner_results = nlp(keyword)
    print(f"关键词：{keyword}")
    for entity in ner_results:
        if entity['entity'] == 'B-BRAND' or entity['entity'] == 'I-BRAND':
            print(f"品牌名称：{entity['word']}")
