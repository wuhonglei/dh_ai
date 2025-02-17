import numpy as np
import base64
from PIL import Image
import io
import torchvision.transforms as transforms


def get_pixel_range(img, axis):
    data = np.where(img.sum(axis=axis) > 0)[0]
    return data[0], data[-1]


def center_img(img):
    """
    提取图片中的有效像素并放置到画布中央
    """
    img = np.array(img)
    old_h, old_w = img.shape
    left, right = get_pixel_range(img, axis=0)
    top, bottom = get_pixel_range(img, axis=1)
    img = img[top:bottom+1, left:right+1]
    h, w = img.shape
    pad_h_start = (old_h - h) // 2
    pad_h_end = old_h - h - pad_h_start
    pad_w_start = (old_w - w) // 2
    pad_w_end = old_w - w - pad_w_start
    new_image = np.pad(img, ((pad_h_start, pad_h_end),
                       (pad_w_start, pad_w_end)), 'constant', constant_values=0)
    return Image.fromarray(new_image)


def parse_request(data):
    label = data['label'] if 'label' in data else None
    data_url = data['dataURL']
    model_name = data['modelName']
    header, encoded = data_url.split(",", 1)
    # 解码 Base64 字符串为二进制数据
    image_data = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(image_data)).convert("L").resize((28, 28))
    return model_name, label, img


def img_transform(img):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    return transform(img)


def vgg_transform(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),  # 将单通道转换为三通道
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],   # 使用 ImageNet 的均值和标准差
                             [0.229, 0.224, 0.225])
    ])
    return transform(img)
