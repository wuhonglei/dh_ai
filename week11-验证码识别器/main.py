import os
from utils import load_config
from generate import generate_captcha
from train import train
from evaluate import evaluate


def main():
    config = load_config()
    dataset_config = config['dataset']
    training_config = config['training']
    testing_config = config['testing']
    model_config = config['model']

    padding_index = dataset_config['padding_index']
    captcha_length = dataset_config['captcha_length']
    model_name = os.path.basename(training_config["model_path"])
    model_path = training_config["model_path"].replace(
        model_name, f'{captcha_length}-{model_name}')
    class_num = len(dataset_config['characters']) + 1  # 1 表示空白字符

    if dataset_config['generate']:
        generate_captcha(total=dataset_config['train_total'], captcha_length=captcha_length,
                         width=dataset_config['width'], height=dataset_config['height'], characters=dataset_config['characters'], dist_dir=dataset_config['train_dir'], remove=dataset_config['remove'])
        generate_captcha(total=dataset_config['test_total'], captcha_length=captcha_length, width=dataset_config['width'], height=dataset_config['height'], dist_dir=dataset_config['test_dir'],
                         characters=dataset_config['characters'], remove=dataset_config['remove'])

    train(data_dir=training_config['train_dir'],
          test_dir=testing_config['test_dir'],
          batch_size=training_config['batch_size'],
          epochs=training_config['epochs'],
          captcha_length=captcha_length,
          class_num=class_num,
          padding_index=padding_index,
          characters=dataset_config['characters'],
          pretrained=training_config['pretrained'],
          learning_rate=training_config['learning_rate'],
          input_size=model_config['input_size'],
          model_path=model_path,
          early_stopping=config['early_stopping']
          )

    evaluate(data_dir=testing_config['test_dir'],
             model_path=model_path,
             input_size=model_config['input_size'],
             captcha_length=captcha_length,
             class_num=class_num,
             padding_index=padding_index,
             characters=dataset_config['characters']
             )


if __name__ == '__main__':
    main()
