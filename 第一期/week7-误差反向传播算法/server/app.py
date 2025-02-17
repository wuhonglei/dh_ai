from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import numpy as np

from nn_models.ann_model import AnnModel
from nn_models.cnn_model import LeNet
from nn_models.vgg_model import VGG16
from train import train
from utils import center_img, parse_request, img_transform, vgg_transform

# 加载模型
ann_model = AnnModel()
ann_model.load_state_dict(torch.load(
    './models/ann_model.pth', map_location=torch.device('cpu'), weights_only=True))
ann_model.eval()

cnn_model = LeNet()
cnn_model.load_state_dict(torch.load(
    './models/cnn_model.pth', map_location=torch.device('cpu'), weights_only=True))
cnn_model.eval()

vgg_model = VGG16()
vgg_model.load_state_dict(torch.load(
    './models/vgg16_model.pth', map_location=torch.device('cpu'), weights_only=True))
vgg_model.eval()

model_map = {
    'ann': {
        'model': ann_model,
        'transform': img_transform
    },
    'cnn': {
        'model': cnn_model,
        'transform': img_transform
    },
    'vgg16': {
        'model': vgg_model,
        'transform': vgg_transform
    }
}

app = Flask(__name__)

# 定义推理 API


@ app.route('/predict', methods=['POST'])
def predict():
    # 获取 JSON 数据
    data = request.get_json()
    # 使用 PIL.Image 读取图像
    model_name, label, img = parse_request(data)
    model = model_map[model_name]['model']
    # 将数字居中
    img = center_img(img)

    # 显示图片
    # img.show()

    img = model_map[model_name]['transform'](img)

    # 模型推理
    output = model.predict(img)

    # 处理并返回结果
    index, prob, prob_list = output
    if label is not None and index.item() != label:
        print('Prediction is wrong. Label:',
              label, 'Prediction:', index.item())
        print('Retraining...')
        train(model, img, label, model_name)
        index, prob, prob_list = model.predict(img)
        print('Retraining finished. New prediction:', index.item())

    return jsonify({'prediction': index.item(), 'probability': prob.item(), 'probabilities': prob_list.tolist()})


if __name__ == '__main__':
    app.run(debug=True)
