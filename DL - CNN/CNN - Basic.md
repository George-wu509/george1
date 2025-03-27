
| Model               | Parameters: LR, batch size, Epoch, weight decay |
| ------------------- | ----------------------------------------------- |
| VGG                 |                                                 |
| ResNet深度殘差網絡        |                                                 |
| FCN 全卷積網絡           |                                                 |
|                     |                                                 |
| [[lightweight CNN]] |                                                 |
|                     |                                                 |
|                     |                                                 |

| Layers        |                                                                                                                                                                                                                                                              |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 卷積核 conv<br>  | Parameters: padding, stride<br>Kernel Size, In Channels, Out Channels<br><br>1x1 conv<br>[dilated conv 擴張卷積](https://zhuanlan.zhihu.com/p/585500690)<br>[transposed Conv 轉置卷積](https://zhuanlan.zhihu.com/p/28186857)<br>[[Depthwise Separable Convolution]] |
| 池化層 pool      |                                                                                                                                                                                                                                                              |
| 激活Activate    | [[Activation funs]]                                                                                                                                                                                                                                          |
| 全連接層 FC       |                                                                                                                                                                                                                                                              |
| Normalization | Batch Normalization<br>Layer Normalization                                                                                                                                                                                                                   |
| Dropout層      | [[Normalization and dropout]]                                                                                                                                                                                                                                |
|               |                                                                                                                                                                                                                                                              |
| 殘差層           |                                                                                                                                                                                                                                                              |
|               |                                                                                                                                                                                                                                                              |
|               |                                                                                                                                                                                                                                                              |

| Block                     |                  |
| ------------------------- | ---------------- |
| Convolution Block         |                  |
| Inception                 |                  |
| Residual Block (ResNet)   | Skip connections |
| Bottleneck Block (ResNet) |                  |
| Attention Block           |                  |
|                           |                  |

| Others                                                         |     |
| -------------------------------------------------------------- | --- |
| Receptive Field(感受野)                                           |     |
| [Hard negative mining](https://zhuanlan.zhihu.com/p/574307045) |     |
|                                                                |     |
|                                                                |     |

參考: [[第五章_卷积神经网络(CNN)]]


![[Pasted image 20250316215533.png]]
![[Pasted image 20250316215548.png]]






Reference:
CNN卷积核与通道讲解 - 双手插袋的文章 - 知乎
https://zhuanlan.zhihu.com/p/251068800