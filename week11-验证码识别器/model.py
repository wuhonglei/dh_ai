import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self, captcha_length, class_num):
        super(CNNModel, self).__init__()
        self.captcha_length = captcha_length
        self.class_num = class_num

        """
        Conv2d: 1 * 128 * 128 -> 8 * 128 * 128 -> 8 * 64 * 64
        """
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=3,
                padding='same',
                stride=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        """
        Conv2d: 8 * 64 * 64 -> 16 * 64 * 64 -> 16 * 32 * 32
        """
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                padding='same',
                stride=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        """
        Conv2d: 16 * 32 * 32 -> 16 * 32 * 32 -> 16 * 16 * 16
        """
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                padding='same',
                stride=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        """
        Linear: 16 * 16 * 16 -> 128
        """
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 16 * 16, 128),
            nn.ReLU(),
        )

        """
        Linear: 128 -> output_size
        """
        self.fc2 = nn.Sequential(
            nn.Linear(128, self.captcha_length * self.class_num),
        )

    def forward(self, x):
        x = x.view(-1, 1, 128, 128)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(-1, 16 * 16 * 16)
        x = self.fc1(x)
        logits = self.fc2(x)
        return logits.view(-1, self.captcha_length, self.class_num)

    def predict(self, x):
        logits = self.forward(x)
        _, pred = logits.max(dim=2)
        return pred, torch.softmax(logits, dim=2)


if __name__ == '__main__':
    import torch
    import os
    from torchvision import transforms
    from PIL import Image
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as F

    model = CNNModel(captcha_length=1, class_num=10)
    model.load_state_dict(torch.load(
        './models/1-model.pth', weights_only=True, map_location='cpu'))

    def print_parameters(model):
        param_count = 0
        for name, param in model.named_parameters():
            param_count += param.numel()
            print(name, param.size())
        print(f'param_count: {param_count}')

    def print_forward(model):
        import torch
        x = torch.randn(1, 1, 128, 128)
        for name, module in model.named_children():
            if name == 'fc1':
                x = x.view(-1, 16 * 16 * 16)
            print(f'name: {name}, input: {x.size()}')
            x = module(x)
            print(f'name: {name}, output: {x.size()}')
            print()

    def display_feature_maps(model):
        # 钩子函数，用于保存每个卷积层的输出
        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()  # 保存输出并分离梯度计算图
            return hook

        def plot_original_image(img_tensor):
            plt.imshow(img_tensor.squeeze(0).permute(1, 2, 0).numpy())
            plt.axis('off')
            plt.show()

        def visualize_layer_output(activation, layer_names: list[str]):
            max_num_channels = max(
                [activation[layer_name].size(1) for layer_name in layer_names]
            )
            fig, axes = plt.subplots(
                nrows=len(layer_names), ncols=max_num_channels)

            for i, layer_name in enumerate(layer_names):
                act = activation[layer_name].squeeze(0)  # 去掉 batch 维度
                num_channels = act.size(0)
                print('num_channels', num_channels)
                for j in range(num_channels):
                    axes[i, j].imshow(act[j].cpu().numpy(), cmap='gray')
                    axes[i, j].set_title(f'{i+1}/{j+1}')
                    axes[i, j].axis('off')

            # 移除空的 subplot
            for i in range(len(layer_names)):
                for j in range(max_num_channels):
                    if not axes[i, j].has_data():
                        axes[i, j].remove()

            # plt.tight_layout()  # 自动调整布局，使子图之间不重叠
            plt.show()

        def visualize_layer_output_avg(activation, layer_names: list[str]):
            fig, axes = plt.subplots(
                nrows=1, ncols=len(layer_names))

            for i, layer_name in enumerate(layer_names):
                act = activation[layer_name].squeeze(0)  # 去掉 batch 维度
                axes[i].imshow(act.cpu().numpy().sum(axis=0), cmap='gray')
            plt.show()

        model.conv1.register_forward_hook(get_activation('conv1'))
        model.conv2.register_forward_hook(get_activation('conv2'))
        model.conv3.register_forward_hook(get_activation('conv3'))
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomAffine(degrees=0, translate=(0.8, 0)),  # 仿射变换
            transforms.ToTensor()
        ])
        for img_name in os.listdir('./data/demo'):
            img_path = os.path.join('./data/demo', img_name)
            label = img_name.split('_')[0]
            image = Image.open(img_path)
            width, height = image.size
            # input_image = transform(Image.open(img_path)).unsqueeze(0)
            # 计算平移的像素距离，向右平移时为正值
            max_dx = int(0.5 * width)
            input_image = F.affine(
                Image.open(img_path), angle=0, translate=(0, 0), scale=1.0, shear=0)
            # 显示图像
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)
            # predict, prob = model.predict(input_image)
            # print('label', label)
            # print('predict', predict[0].item(), prob[0].max().item())
            plot_original_image(input_image)
            # visualize_layer_output_avg(activation, ['conv1', 'conv2', 'conv3'])
            # visualize_layer_output(activation, ['conv1', 'conv2', 'conv3'])
            break

    # print(model)
    # print_parameters(model)
    # print_forward(model)
    display_feature_maps(model)
