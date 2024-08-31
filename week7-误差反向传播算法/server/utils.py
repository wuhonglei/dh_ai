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
    new_image = np.full((old_h, old_w), 0, dtype=np.uint8)
    # print(new_image.shape)
    row_start = (old_h - h) // 2
    col_start = (old_w - w) // 2
    new_image[row_start:row_start+h, col_start:col_start+w] = img
    return Image.fromarray(new_image)


def parse_request(data):
    label = data['label'] if 'label' in data else None
    data_url = data['dataURL']
    header, encoded = data_url.split(",", 1)
    # 解码 Base64 字符串为二进制数据
    image_data = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(image_data)).convert("L").resize((28, 28))
    return label, img


def img_transform(img):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    return transform(img)
