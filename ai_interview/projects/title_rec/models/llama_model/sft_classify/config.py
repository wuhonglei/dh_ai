train_csv_path = '../../../dataset/level1_80/clean/valid.csv'
# valid_csv_path = '../../dataset/level1_80/clean/valid.csv'
test_csv_path = '../../../dataset/level1_80/clean/test.csv'
columns = [
    'spacy_tokenized_name', 'nltk_tokenized_name',
    'remove_prefix', 'remove_prefix_emoji',
    'remove_prefix_emoji_symbol', 'remove_prefix_emoji_symbol_stop_words',
    'remove_nltk_stop_words',
    'remove_spacy_stop_words'
]
model_path = '../Llama-3.2-1B'
label_name = 'level1_global_be_category_id'
project_name = 'shopee_title_llama_sft_model'
max_length = 30
