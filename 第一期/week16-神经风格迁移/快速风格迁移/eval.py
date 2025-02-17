import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from transformer_net import TransformerNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformer = TransformerNet()
transformer.load_state_dict(torch.load("./models/model_epoch_10.pth"))
transformer.to(device)
transformer.eval()

content_image = Image.open("content.png")
content_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])

content = content_transform(content_image).unsqueeze(  # type: ignore
    0).to(device)

with torch.no_grad():
    output = transformer(content).cpu()


def post_process(tensor):
    image = tensor.clone().squeeze(0)
    image = image.clamp(0, 255).div(255)
    image = image.permute(1, 2, 0).numpy()
    image = np.uint8(image * 255)
    return Image.fromarray(image)


output_image = post_process(output)
output_image.save("output.jpg")
