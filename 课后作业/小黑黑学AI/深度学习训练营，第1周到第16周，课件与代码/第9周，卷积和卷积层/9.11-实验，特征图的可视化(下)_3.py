
from torchvision import models
from torchvision import transforms
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import os

def get_models_conv_layers(model):
    convs = list()
    layers = list(model.children())
    for i in range(len(layers)):
        if type(layers[i]) == nn.Conv2d:
            convs.append(layers[i])
        elif type(layers[i]) == nn.Sequential:
            for j in range(len(layers[i])):
                for child in layers[i][j].children():
                    if type(child) == nn.Conv2d:
                        convs.append(child)
    return convs

def read_img_to_tensor(img_path, height, width):
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0., std=1.)
    ])
    image = Image.open(img_path)
    image = transform(image)
    image = image.unsqueeze(0)
    return image

def get_feature_map(conv_layers, image):
    feature_map = []
    for layer in conv_layers:
        image = layer(image)
        feature_map.append(image)

    outputs = list()
    for feature in feature_map:
        #print(f"feature_map.shape {feature_map.shape}")

        feature = feature.squeeze(0)
        gray_scale = torch.sum(feature, 0)

        #print(f"feature_map.shape {feature_map.shape}")
        #print(f"gray_scale {gray_scale.shape}")
        #print(feature_map.shape[0])
        #print("")

        gray_scale = gray_scale / feature.shape[0]
        outputs.append(gray_scale.data.numpy())

    return outputs


if __name__ == '__main__':

    model = models.resnet18(pretrained=True)
    print(model)

    conv_layers = get_models_conv_layers(model)
    print(f"conv_layers num : {len(conv_layers)}")

    image = read_img_to_tensor("catdog.jpg", 224, 224)
    print(f"image shape {image.shape}")

    feature_map = get_feature_map(conv_layers, image)

    if not os.path.exists('feature_maps'):
        os.makedirs('feature_maps')

    for i in range(len(feature_map)):
        fig = plt.imshow(feature_map[i])
        plt.savefig('./feature_maps/' + str(i) + '.jpg')

