import os
from utils import load_config, get_max_length, load_json
from generate import generate_captcha
from train import train
from shutdown import shutdown


def main():
    config = load_config()
    dataset_config = config['dataset']
    training_config = config['training']
    model_config = config['model']

    padding_index = dataset_config['padding_index'] if 'padding_index' in dataset_config else ''
    captcha_length = get_max_length(dataset_config['captcha_length'])
    model_path = training_config["model_path"]
    pretrained_model_path: str = training_config[
        "pretrained_model_path"] if 'pretrained_model_path' in training_config else ''
    padding = 1 if len(str(padding_index)) else 0
    class_num = len(dataset_config['characters']) + padding  # 1 表示空白字符

    if dataset_config['generate']:
        origin_captcha_length = dataset_config['captcha_length']
        characters_json = load_json(
            dataset_config['characters_json']) if 'characters_json' in dataset_config else {}
        new_characters = characters_json.get(
            'characters', dataset_config['characters'])
        new_weight = characters_json.get('freqency', None)
        generate_captcha(total=dataset_config['train_total'], captcha_length=origin_captcha_length,
                         width=dataset_config['width'], height=dataset_config['height'], characters=new_characters, dist_dir=dataset_config['train_dir'], remove=dataset_config['remove'], weight=new_weight)
        generate_captcha(total=dataset_config['test_total'], captcha_length=origin_captcha_length, width=dataset_config['width'], height=dataset_config['height'], dist_dir=dataset_config['test_dir'],
                         characters=new_characters, remove=dataset_config['remove'], weight=new_weight)

    train(train_dir=training_config['train_dir'],
          test_dir=training_config['test_dir'],
          batch_size=training_config['batch_size'],
          epochs=training_config['epochs'],
          captcha_length=captcha_length,
          class_num=class_num,
          padding_index=padding_index,
          characters=dataset_config['characters'],
          pretrained_model_path=pretrained_model_path,
          learning_rate=training_config['learning_rate'],
          width=model_config['width'],
          height=model_config['height'],
          model_path=model_path,
          log=training_config['log'],
          in_channels=model_config['in_channels'],
          hidden_size=model_config['hidden_size'],
          early_stopping=config['early_stopping'],
          )


if __name__ == '__main__':
    main()
    # shutdown(10)
