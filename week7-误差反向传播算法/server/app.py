from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import numpy as np

from model import MyModel
from train import train
from utils import center_img, parse_request, img_transform


# 加载模型
model = MyModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()

app = Flask(__name__)

# 定义推理 API


@app.route('/predict', methods=['POST'])
def predict():
    # 获取 JSON 数据
    data = request.get_json()
    # 使用 PIL.Image 读取图像
    label, img = parse_request(data)
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
        train(model, img, label)
        index, prob, prob_list = model.predict(img)
        print('Retraining finished. New prediction:', index.item())

    return jsonify({'prediction': index.item(), 'probability': prob.item(), 'probabilities': prob_list.tolist()})


if __name__ == '__main__':
    app.run(debug=True)
