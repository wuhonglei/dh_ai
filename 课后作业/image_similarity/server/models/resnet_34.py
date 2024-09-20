import torch
from torchvision import models
from torchinfo import summary


class ResNet34FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super(ResNet34FeatureExtractor, self).__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.feature_extractor = torch.nn.Sequential(
            *list(resnet.children())[:-1])

    def forward(self, x):
        return self.feature_extractor(x)


if __name__ == '__main__':
    model = ResNet34FeatureExtractor()
    model.eval()
    print(model)
    with torch.no_grad():
        summary(model, input_size=(1, 3, 224, 224), device='cpu')
    # model
