level1_names = [
    '100001',
    '100009',
    '100010',
    '100011',
    '100012',
    '100013',
    '100014',
    '100015',
    '100016',
    '100017',
    '100532',
    '100533',
    '100534',
    '100535',
    '100629',
    '100630',
    '100631',
    '100632',
    '100633',
    '100634',
    '100635',
    '100636',
    '100637',
    '100638',
    '100639',
    '100640',
    '100641',
    '100642',
    '100643',
    '100644',
]

columns = [
    'remove_spacy_stop_words',
    'spacy_tokenized_name', 'nltk_tokenized_name',
    'remove_prefix', 'remove_prefix_emoji',
    'remove_prefix_emoji_symbol', 'remove_prefix_emoji_symbol_stop_words',
    'remove_nltk_stop_words',
]
train_txt = 'train.txt'
test_txt = 'test.txt'
train_args = {
    'epoch': 50,
    'lr': 0.1,
    'wordNgrams': 2,
    'minCount': 2,
    'dim': 100,
    'loss': 'softmax',
}

final_test_data_path = '../joint/data/{column}/test.txt'
