
#### Depthwise Separable Convolution （深度可分離卷積）

  分為兩步：
1. **Depthwise Convolution（逐通道卷積）**：對每個輸入通道單獨應用一個卷積核，生成與輸入通道數相同的特徵圖。
2. **Pointwise Convolution（點卷積）**：使用 1×1 卷積將 depthwise 的輸出混合，生成指定數量的輸出通道。


我們來深入探討高效卷積（如 Depthwise Separable Convolution）如何取代標準卷積，並透過具體範例來說明。

**標準卷積的運作方式**

假設我們有一個 7x7 的輸入影像，具有 3 個通道（RGB）。我們想要使用一個 3x3 的卷積核（convolutional kernel）來提取特徵，並且輸出 16 個通道的特徵圖。

- **參數數量：** 標準卷積的參數數量為 (3x3) x 3 (輸入通道) x 16 (輸出通道) = 432 個參數。
- **運算量：** 運算量（FLOPs）相對較高，因為每個輸出像素都需要對所有輸入通道進行運算。

**Depthwise Separable Convolution 的運作方式**

Depthwise Separable Convolution 將標準卷積分解為兩個步驟：

1. **Depthwise Convolution（深度卷積）：**
    - 對每個輸入通道單獨進行卷積。
    - 在這個例子中，我們會有 3 個 3x3 的卷積核，每個卷積核只對一個輸入通道進行運算。
    - 參數數量：(3x3) x 3 = 27 個參數。
2. **Pointwise Convolution（逐點卷積）：**
    - 使用 1x1 的卷積核，將 Depthwise Convolution 的輸出進行線性組合。
    - 在這個例子中，我們會有 16 個 1x1 的卷積核，每個卷積核的輸入通道數為 3，輸出通道數為 1。
    - 參數數量：1x1x3x16=48

- **總參數數量：** 27 + 48 = 75 個參數。
- **運算量：** 運算量顯著降低，因為我們將標準卷積分解為兩個更小的運算。

**具體範例說明**

假設我們有一個 7x7x3 的輸入影像，我們想要使用 3x3 的卷積核來提取特徵，並且輸出 16 個通道的特徵圖。inchannel=3, outchannel=16

- **標準卷積：**
    - 需要 432 個參數。 = 3x3x3x16
    - 運算量較高。
- **Depthwise Separable Convolution：**
    - 需要 75 個參數。 = 3x3x3 + 3x16
    - 運算量顯著降低。

**Depthwise Separable Convolution 的優勢**

- **減少參數數量：** 顯著減少模型的大小，使其更適合部署在資源有限的裝置上。
- **降低運算量：** 加快推論速度，提高模型的效率。
- **提高效率：** 在一些情況下，Depthwise Separable Convolution 甚至可以提高模型的精度。

**Depthwise Separable Convolution 的應用**

- MobileNet 系列模型大量使用了 Depthwise Separable Convolution，使其在移動裝置上實現了高效的目標檢測和圖像分類。
- Xception 模型也使用了 Depthwise Separable Convolution，提高了圖像分類的精度。

**總結**

Depthwise Separable Convolution 是一種高效的卷積運算，可以顯著減少模型的大小和運算量，同時保持甚至提高模型的精度。它在移動裝置和嵌入式裝置上的應用非常廣泛。




Reference:
卷积神经网络之深度可分离卷积（Depthwise Separable Convolution） - Civ的文章 - 知乎
https://zhuanlan.zhihu.com/p/166736637