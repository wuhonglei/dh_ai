# cnn-captcha-pytorch
小黑黑讲AI，AI实战项目《验证码识别》

![1](https://github.com/xhh890921/cnn-captcha-pytorch/assets/112564707/d2788396-a3a5-44ef-871d-1751a68afe2f)

本readme主要介绍了如下内容：

1)《验证码识别》项目的背景。

2)如何搭建项目的运行环境。

3)如何使用项目中的各个工具。

希望帮助大家快速且充分的了解这个AI实战项目。


1.这个项目讲什么

该实战项目的教学目标是，让同学们深入学习卷积神经网络算法和Pytorch深度学习框架，进而使用这些算法技术，解决实际的工程问题。

该项目的教学视频，一共包括10个小节：

![2](https://github.com/xhh890921/cnn-captcha-pytorch/assets/112564707/b4e6c982-9667-4b96-8eca-7f5b87c68693)

从项目的背景与环境搭建开始讲起，接着讲解数据生成、小批量数据读取、CNN网络的设计、训练和测试。GPU加速模型训练，工程、模型与数据等各方面的优化方法。

想要观看视频讲解的小伙伴，可以关注我的B站账号：小黑黑讲AI

只要同学们有一定的Python程序设计基础，都能通过这个项目，快速熟悉深度学习算法，并学会使用Pytorch框架，训练深度学习模型。


2.验证码识别的背景

目前，验证码已经成为各种网站和应用程序的标准安全措施，通过让用户输入验证码，可以阻止机器程序的恶意行为。

![3](https://github.com/xhh890921/cnn-captcha-pytorch/assets/112564707/10445dcc-1404-4429-8559-432f28263b8f)

通过验证码可以，阻止爬虫抓取数据、用户批量注册，或者是刷单购票等行为。不过在某些情境下，如网站的压力测试，我们需要自动化的识别验证码。

整体来说，我们会开发一个基于深度学习的验证码识别系统，其中重点是设计并训练卷积神经网络模型：

![4](https://github.com/xhh890921/cnn-captcha-pytorch/assets/112564707/d6fbffd2-f96a-4272-a7e6-dbecd07a975e)

来专门识别某一种特定形态的验证码。

在目前的互联网环境中，有多种完全不同的验证码形态。

主要包括包括如下几种类型，分别是字符型验证码、文字型验证码、图片型验证码等等：

![5](https://github.com/xhh890921/cnn-captcha-pytorch/assets/112564707/17efd36e-a531-4570-9806-c31b4b143009)

不同形态的验证码，需要使用不同的技术解决。

例如，最简单的是字符型验证码，一般在一个图片中，会包括4-6位的数字与字母的组合。

为了增加识别难度，网站会对字符进行扭曲和旋转，并在字符的周围设置线条或点等干扰因素。

在本项目中，就是要讨论“字符型验证码”的识别方法。

验证码识别，可能涉及图片分类、目标检测、OCR光学字符识别，等等计算机视觉的相关算法：

![6](https://github.com/xhh890921/cnn-captcha-pytorch/assets/112564707/04206ab3-39d7-4d18-99ea-a9810ffb4a30)

对于字符型验证码识别，主要会使用到图片分类技术。


3.项目代码、搭建和运行

在代码中，主要包括两个部分:

第1部分是完整的工程代码，包括5个python文件和1个配置文件。

第2部分是10节课程的随堂代码，保存在lesson文件夹中。

打开lesson文件夹，会看到代码按照每节课程的标题来组织：

![7](https://github.com/xhh890921/cnn-captcha-pytorch/assets/112564707/66b336a5-3e82-49bf-90e1-ef0962ce0e0c)

每个文件都可以单独运行，在使用时，可以参考视频对应的使用。

为了运行这个工程，需要构建工程的运行环境：

![8](https://github.com/xhh890921/cnn-captcha-pytorch/assets/112564707/81817ed6-9091-4b3c-959a-f0f4a335f30b)

这里使用conda命令行，安装pytorch的GPU版本，安装命令可以从官网中复制。

安装pytorch后，继续安装captcha，用于生成验证码图片。

完成依赖的安装后，继续运行一下这个工程。

工程的使用方法非常简单，首先打开config.json，其中包括了关键的配置参数：

![9](https://github.com/xhh890921/cnn-captcha-pytorch/assets/112564707/5fca3711-7673-47c9-bcea-bc7f78fe159f)

这里直接使用默认的配置运行。

简单来说，在默认配置中，会生成2000个训练数据和1000个测试数据。

验证码字符为1位的数字字符，迭代200轮。更细节的内容可以参考教学视频。

使用generate.py生成训练和测试数据：

![10](https://github.com/xhh890921/cnn-captcha-pytorch/assets/112564707/701eed18-ab50-4282-8271-b189b8f6027a)

生成的验证码数量和形式，都是基于config.json配置的。

运行generate.py后，会发现在当前目录中，生成了data文件夹，其中train-digit文件夹中的是训练数据，test-digit中的是测试数据。

使用train.py训练模型：

![11](https://github.com/xhh890921/cnn-captcha-pytorch/assets/112564707/97b4d401-2d77-4c39-b1fc-e456ee4b7cf5)

运行train.py后，会看到模型的迭代。完成迭代后，会在当前目录生成model文件夹。

其中captcha.1digit.2k是最终的训练模型，每10轮训练，都会保存一个check模型。

使用test.py测试模型：

![12](https://github.com/xhh890921/cnn-captcha-pytorch/assets/112564707/45d29937-dcbf-4b22-9c79-ad1f983b3454)

运行test.py后，会使用test-digit中的测试数据，测试训练生成的模型。

这里可以看到，1000个测试数据，968个数据识别正确，正确率是0.968。

更详细的讲解，请小伙伴们，关注我的B站账号“小黑黑讲AI”，其中有详细的讲解这个实战工程中的每个细节。

有任何问题，也可以关注我的公众号“小黑黑讲AI”咨询。

![公众号](https://github.com/xhh890921/cnn-captcha-pytorch/assets/112564707/c97a8cb2-b6ac-43c8-bb06-f4fe6b3ac431)
