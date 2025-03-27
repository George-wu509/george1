
![[Pasted image 20240912132920.png]]
Reference:

[1] FCOS官方代码详解（一）：Architecture(backbone)

https://zhuanlan.zhihu.com/p/385332512


FCOS（Fully Convolutional One-Stage Object Detection）的主要作用是**物件偵測（Object Detection）**，而不是直接用於影像分割（Image Segmentation）。

以下是FCOS的架構以及設計特點：

**FCOS 架構與設計特點**

- **Anchor-free（無錨框）：**
    - FCOS 的核心創新之一是它消除了傳統物件偵測器中對預定義錨框（anchor boxes）的依賴。
    - 相反，它直接預測影像中每個像素的邊界框和類別。
- **Fully Convolutional（全卷積）：**
    - FCOS 基於全卷積網路（FCN），這意味著它可以進行端到端的訓練和推理。
    - 這種全卷積的設計使得 FCOS 能夠高效地處理不同尺寸的影像。
- **Pixel-wise Prediction（像素級預測）：**
    - FCOS 將物件偵測視為一個像素級的預測問題。
    - 對於影像中的每個像素，它預測該像素是否在物件的邊界框內，以及到邊界框四個邊的距離。
- **Feature Pyramid Networks (FPN)（特徵金字塔網路）：**
    - 為了處理不同尺寸的物件，FCOS 使用 FPN 來提取多尺度的特徵圖。
    - 這使得 FCOS 能夠有效地檢測小物件和大物件。
- **Centerness（中心度）：**
    - FCOS 引入了 "centerness" 的概念，用於抑制遠離物件中心的低品質邊界框。
    - "centerness" 分支預測像素到物件中心的歸一化距離，並用於調整邊界框的置信度。
- **優點：**
    - 簡化了物件偵測的流程，無需複雜的錨框相關計算。
    - 減少了需要人工調整的超參數數量。
    - 增加小物件的偵測效果。

**總結來說：**

FCOS通過像素級的預測，並且去除了以往模型anchor box的設計，更有效的減少計算量，並且提供良好的物件偵測精準度。