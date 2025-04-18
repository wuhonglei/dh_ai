import os

current_dir = os.path.dirname(os.path.abspath(__file__))
train_csv_path = os.path.join(
    current_dir, '../../../../dataset/level1_80/clean/valid.csv')
# valid_csv_path = '../../../../dataset/level1_80/clean/valid.csv'
test_csv_path = os.path.join(
    current_dir, '../../../../dataset/level1_80/clean/test.csv')

vocab_dir = 'data/vocab_freq/'
columns = [
    'spacy_tokenized_name', 'nltk_tokenized_name',
    'remove_prefix', 'remove_prefix_emoji',
    'remove_prefix_emoji_symbol', 'remove_prefix_emoji_symbol_stop_words',
    'remove_nltk_stop_words',
    'remove_spacy_stop_words'
]
project_name = 'shopee_title_custom_fasttext_cascade_model'
