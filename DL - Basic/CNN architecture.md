
## 1. VGG

![[vgg.png]]
VGG是由牛津大學計算機視覺組Visual Geometry Group提出(這也是VGG名稱的由來)，並在2014年的ILSVRC比赛上獲得Localization Task (定位任務) 第一名和 Classification Task (分類任務) 第二名。
- 13層卷積層(convX-XXX)，第一個數字為卷積核大小，第二個數字為卷積層通道數
- 3層全連接層(FC-XXXX)
- 5層最大池化層(maxpool)

卷積層與全連接層具有權重係數，因此也被稱為權重層，總數目為13+3=16，這即是VGG16中16的來源。
#### 特點
- 卷積核大小(kernel size)統一為3x3，
- 最大池化層(maxpool)統一為2x2
- 利用較小的卷積核(ex：3x3)來替代較大的卷積核(ex：5x5，7x7)，在感受野相同的情況下提升網路深度，一個5x5的能用兩個3x3的替代，一個7x7能用三個3x3替代
#### 優/缺點
- VGG的架構簡單統一
- 證明較深的層數能提高效能
- 參數量龐大，計算資源需求高
- 訓練時間過長，難以調整參數

## 2. ResNet

ref: [直觀理解ResNet —簡介、 觀念及實作(Python Keras)](https://medium.com/@rossleecooloh/%E7%9B%B4%E8%A7%80%E7%90%86%E8%A7%A3resnet-%E7%B0%A1%E4%BB%8B-%E8%A7%80%E5%BF%B5%E5%8F%8A%E5%AF%A6%E4%BD%9C-python-keras-8d1e2e057de2)
ref: [Understanding ResNet-50 in Depth](https://wisdomml.in/understanding-resnet-50-in-depth-architecture-skip-connections-and-advantages-over-other-networks/)


![[Pasted image 20241103111718.png]]

![[resnet50.webp]]

ResNet-50 是一個50層的深度殘差網絡,主要由Bottleneck blocks組成。它的整體架構如下:

1. 一個7x7的卷積層
2. 一個最大池化層
3. 4組殘差層(每組包含多個Bottleneck blocks)
4. 一個全局平均池化層
5. 一個全連接層

## Residual Block 解釋

殘差塊(Residual Block)是ResNet的核心組件。它的主要思想是:

1. 讓網絡學習殘差函數F(x) = H(x) - x,而不是直接學習H(x)。
2. 通過skip connection(跳躍連接)將輸入直接加到輸出上。

這樣做的好處是:

1. 緩解了深度網絡的梯度消失問題。
2. 使得網絡更容易學習恆等映射,有助於網絡的訓練。

## Skip Connections 解釋

Skip connections(跳躍連接)是將輸入直接加到某一層的輸出上的連接。在ResNet中,它將輸入x直接加到F(x)上,形成H(x) = F(x) + x。Skip connections的優點:

1. 允許梯度直接流回早期層,緩解梯度消失問題。
2. 使網絡能夠輕鬆學習恆等函數,有助於訓練非常深的網絡。

## Bottleneck Block 解釋

Bottleneck block是ResNet-50及更深層次ResNet使用的一種特殊的殘差塊。它的結構是:

1. 1x1卷積降維
2. 3x3卷積
3. 1x1卷積升維

使用Bottleneck block的原因:

1. 減少參數量和計算量,使得訓練更深的網絡成為可能。
2. 1x1卷積可以有效地改變通道數,實現降維和升維。

