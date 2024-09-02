from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import numpy as np

from ann_model import AnnModel
from cnn_model import LeNet
from train import train
from utils import center_img, parse_request, img_transform

# 加载模型
ann_model = AnnModel()
ann_model.load_state_dict(torch.load(
    'ann_model.pth', map_location=torch.device('cpu')))
ann_model.eval()

cnn_model = LeNet()
cnn_model.load_state_dict(torch.load(
    'cnn_model.pth', map_location=torch.device('cpu')))
cnn_model.eval()

app = Flask(__name__)

# 定义推理 API


@app.route('/predict', methods=['POST'])
def predict():
    # 获取 JSON 数据
    data = request.get_json()
    # 使用 PIL.Image 读取图像
    model_name, label, img = parse_request(data)
    model = ann_model if model_name == 'ann_model.pth' else cnn_model
    # 将数字居中
    img = center_img(img)

    # 显示图片
    # img.show()

    img = img_transform(img)

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
