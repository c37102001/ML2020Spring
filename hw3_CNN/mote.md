# Accuracy
### 50 Train-Valid, 30 Full Train
| Model                 | Best Valid Acc| Best Loss     | Full Train Acc| Public Acc    | Private Acc   |
| --------------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Vanilla CNN-baseline  | 72.15% (35)   | 1.039 (19)    | 98.06%        |               |               |
| Vanilla CNN-data norm | 70.46% (42)   | 1.062 (21)    | 97.75%        |               |               |
| Vanilla CNN-data aug  | 71.89% (47)   | 0.953 (34)    | 86.71%        |               |               |
| Vanilla CNN-wd+lr     | 74.69% (49)   | 0.800 (31)    | 86.81%        |               |               |
| Vanilla FCN-wd+lr     | 74.57% (48)   | 0.843 (38)    | 87.19%        |               |               |
| ResNet-FCN            | 76.58% (47)   | 0.731 (47)    | 86.19%        | 81.76%        |               |
| MobileNetV2           | 71.25% (48)   | 0.879 (48)    | 76.06%        |               |               |
| VGG16                 | 68.68% (50)   | 0.977 (47)    | 82.19%        |               |               |
| VGG16-FCN             | 68.48% (50)   | 0.976 (50)    | 80.92%        |               |               |



# Reference
OpenCV、Skimage、PIL圖像處理的細節差異
https://blog.csdn.net/u013010889/article/details/54347089

transforms的二十二個方法
https://zhuanlan.zhihu.com/p/53367135
https://blog.csdn.net/qq_32425195/article/details/84998030
https://blog.csdn.net/qq_24739717/article/details/102743691

Data Augmentation for Deep Learning
https://towardsdatascience.com/data-augmentation-for-deep-learning-4fe21d1a4eb9

Must Know Tips/Tricks in Deep Neural Networks
http://210.28.132.67/weixs/project/CNNTricks/CNNTricks.html?fbclid=IwAR1zUM-eSBD1vh3DRB6ge0RzrG8cLozBOfl8bQfGErFC141gpCHbvyAtE0s