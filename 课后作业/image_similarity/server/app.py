from flask import Flask, jsonify, request
from model.similarity import compare_similarity, process_img
from model.vgg19_feature import VGG19FeatureExtractor
from model.vgg16_feature import VGG16FeatureExtractor

app = Flask(__name__)

model_dict = {
    'vgg16': VGG16FeatureExtractor(),
    'vgg19': VGG19FeatureExtractor()
}


@app.route('/')
def hello_world():
    return 'Hello, Flask!'


@app.route('/api/compare/images', methods=['POST'])
def compare_images():
    # 获取 post 请求中 form-data 的参数
    image1 = request.files.get('image1')
    image2 = request.files.get('image2')
    if image1 is None or image2 is None:
        return jsonify({'error': 'Both image1 and image2 are required'}), 400

    # model_name = request.form.get('model')
    img1_tensor = process_img(image1.stream)
    img2_tensor = process_img(image2.stream)
    similarity = compare_similarity(
        model_dict['vgg19'], img1_tensor, img2_tensor)
    return jsonify({'similarity': similarity})


if __name__ == '__main__':
    app.run(debug=True)
