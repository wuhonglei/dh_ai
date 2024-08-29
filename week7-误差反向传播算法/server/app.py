from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import numpy as np


class MyModel(nn.Module):
    """
    定义模型（与训练时的模型结构相同）
    """

    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def predict(self, x):
        with torch.no_grad():
            x = self.forward(x)
            prob = torch.softmax(x, 1)
            max_index = torch.argmax(prob, 1)
            return torch.argmax(x, 1), prob[0][max_index], prob[0]


def train(model, img, label):
    label = torch.tensor([label], dtype=torch.int64).view(1)
    """ 对于识别错误的图片，使用正确的标签进行训练 """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(15):
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch}, Loss: {loss.item()}')
    torch.save(model.state_dict(), './model.pth')
    model.eval()


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
    label = data['label'] if 'label' in data else None
    data_url = data['dataURL']
    header, encoded = data_url.split(",", 1)
    # 解码 Base64 字符串为二进制数据
    image_data = base64.b64decode(encoded)

    # 使用 PIL.Image 读取图像
    img = Image.open(io.BytesIO(image_data)).convert("L").resize((28, 28))
    img_array = np.array(img)
    print(img_array.mean())
    # 预览图像
    # img.show()

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    img = transform(img)

    # 模型推理
    with torch.no_grad():
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
