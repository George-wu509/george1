
| Model               | Parameters: LR, batch size, Epoch, weight decay                         |
| ------------------- | ----------------------------------------------------------------------- |
| VGG                 | 3x3 kernel size replace 5x5,7x7                                         |
| [[ResNet]]深度殘差網絡    | Residual block, skip connection, vanishing gradient, exploding gradient |
| [[FCN]] 全卷積網絡       | Fully Convolutional Network, Transpose layer(upsampling)                |
| [[DarkNet]]         | YOLO backbone                                                           |
| [[lightweight CNN]] | MobileNet(Depthwise Separable Convolution), EfficientNet                |
|                     |                                                                         |
|                     |                                                                         |

| Layers              |                                                                                                                                                                                                                                                                                                                                                                 |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 卷積核 conv<br>[[CNN]] | Parameters: padding, stride<br>Kernel Size, In Channels, Out Channels<br><br>[[CNN architecture]]<br>[[AI model Summary architecture]]<br><br>[[1x1 conv]]<br>[dilated conv 擴張卷積](https://zhuanlan.zhihu.com/p/585500690)<br>[transposed Conv 轉置卷積](https://zhuanlan.zhihu.com/p/28186857)<br>[[Depthwise Separable Convolution]]<br>3x3x3x16 -> 3x3x3+3x16<br> |
| Normalization       | Batch Normalization<br>Layer Normalization                                                                                                                                                                                                                                                                                                                      |
| 激活Activate          | [[Activation funs]]                                                                                                                                                                                                                                                                                                                                             |
| 池化層 pool            | max pooling                                                                                                                                                                                                                                                                                                                                                     |
| Dropout層            | [[Normalization and dropout]]                                                                                                                                                                                                                                                                                                                                   |
| 全連接層 FC             | Fully connected layer                                                                                                                                                                                                                                                                                                                                           |

| Block                     |                  |
| ------------------------- | ---------------- |
| Convolution Block         |                  |
| Inception                 |                  |
| Residual Block (ResNet)   | Skip connections |
| Bottleneck Block (ResNet) |                  |
| Attention Block           |                  |
|                           |                  |

| Others                          |                             |
| ------------------------------- | --------------------------- |
| Receptive Field(感受野)            |                             |
|                                 |                             |
| object detection anchor正負樣本比例不均 |                             |
| [[Hard Negative Mining]]        | 選擇困難負(hard negative)樣本放入訓練  |
| [[Focal Loss]]                  | 通過調整損失函數的權重，使模型更關注於難分類樣本的學習 |
|                                 |                             |

參考: [[第五章_卷积神经网络(CNN)]]


![[Pasted image 20250316215533.png]]
![[Pasted image 20250316215548.png]]






Reference:
CNN卷积核与通道讲解 - 双手插袋的文章 - 知乎
https://zhuanlan.zhihu.com/p/251068800