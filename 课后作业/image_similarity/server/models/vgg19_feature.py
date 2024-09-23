import torch
import torch.nn as nn
import torchvision.models as models
from torchinfo import summary

# 设置 TORCH_HOME 环境变量
if os.path.exists('/mnt/model/nlp/'):
    os.environ['TORCH_HOME'] = '/mnt/model/nlp/pytorch'


class VGG19FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG19FeatureExtractor, self).__init__()
        vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        self.features = vgg19.features
        self.avgpool = vgg19.avgpool
        self.classifier = nn.Sequential(*list(vgg19.classifier)[:-1])

    def forward(self, x):
        self.eval()
        with torch.no_grad():
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x


if __name__ == '__main__':
    model = VGG19FeatureExtractor()
    summary(model, input_size=(1, 3, 224, 224))
