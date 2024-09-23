### 单个图片特征提取和插入的时间

```
Total time: 243.3670575618744
Total 304 images processed
time_dict: {
  'resnet34': 3.238675355911255, # reset34 特征提取
  'vgg19': 0.582634449005127, # vgg19 特征提取
  'resnet34_insert': 115.92143750190735, # reset34 插入
  'vgg19_insert': 112.51955652236938 # vgg19 插入
}
```


### 128 批量图片特征提取和插入的时间

```
Total time: 11.670644998550415
Total 300 images processed
time_dict: {
  'resnet34': 1.6777851581573486,
  'vgg19': 0.06181168556213379, 
  'resnet34_insert': 1.1519958972930908, 
  'vgg19_insert': 1.3230855464935303
}
```