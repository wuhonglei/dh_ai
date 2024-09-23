import os
from flask import Flask, jsonify, request

from models.similarity import compare_similarity, process_img
from models.vgg19_feature import VGG19FeatureExtractor
from models.vgg16_feature import VGG16FeatureExtractor
from models.resnet_34 import ResNet34FeatureExtractor

from dataset.image_search import search_similar_images


app = Flask(__name__, static_folder="dataset/data", static_url_path='/images')

model_dict = {
    # 'vgg16': VGG16FeatureExtractor(),
    'vgg19': VGG19FeatureExtractor(),
    'resnet34': ResNet34FeatureExtractor()
}


@app.route('/')
def hello_world():
    return 'Hello, Flask!'


@app.route('/api/compare/images', methods=['POST'])
def compare_images():
    # 获取 post 请求中 form-data 的参数
    image1 = request.files.get('image1')
    image2 = request.files.get('image2')
    modelName = request.form.get('model', 'vgg19')
    if image1 is None or image2 is None:
        return jsonify({'error': 'Both image1 and image2 are required'}), 400

    # model_name = request.form.get('model')
    img1_tensor = process_img(image1.stream)
    img2_tensor = process_img(image2.stream)
    similarity = compare_similarity(
        model_dict[modelName], img1_tensor, img2_tensor)
    return jsonify({'similarity': similarity})


@app.route('/api/search/images', methods=['POST'])
def search_images():
    # 获取 post 请求中 form-data 的参数
    image = request.files.get('image')
    modelName = request.form.get('model', 'vgg19')
    limit = request.form.get('limit', 3)
    if image is None:
        return jsonify({'error': 'Both image1 and image2 are required'}), 400

    img_tensor = process_img(image.stream)
    if img_tensor is None:
        return jsonify({'error': 'Invalid image'}), 400

    similar_images = search_similar_images(
        img_tensor=img_tensor, model=model_dict[modelName], collection_name=modelName, limit=int(limit))

    return jsonify({'images': similar_images[0] if similar_images else []})


if __name__ == '__main__':
    app.run(debug=True)
