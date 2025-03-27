
近年来Transformer[2]被广泛的应用到计算机视觉的物体分类领域，例如iGPT，ViT等。这里要介绍的DETR[1]是第一篇将Transformer应用到目标检测方向的算法。DETR是一个经典的Encoder-Decoder结构的算法，它的骨干网络是一个卷积网络，Encoder和Decoder则是两个基于Transformer的结构。DETR的输出层则是一个MLP。

DETR使用了一个基于二部图匹配（bipartite matching）的损失函数，这个二部图是基于ground truth和预测的bounding box进行匹配的。


![[Pasted image 20250323191801.png]]
解釋:  

**Backbone**:  輸入Image 經過CNN (ResNet) 是縮小32倍的feature images. 再加上positional encoding成為Encoder的輸入

**Encoder**: DETR 的编码器基本上遵循经典的 Transformer 编码器结构，包括多头自注意力（multi-head self-attention）层和前馈网络（feed-forward networks）

**Decoder**: - DETR 的解码器也遵循经典的 Transformer 解码器结构，但进行了一些修改以适应目标检测- 解码器中的每一层都包含多头自注意力层和多头交叉注意力层。

**Prediction Heads**: 预测头通常是简单的线性层, DETR的head model,是直接輸出bounding box座標與類別。
![[Pasted image 20250323192312.png]]

DETR（DEtection TRansformer）是一种利用 Transformer 架构进行目标检测的创新模型。它与传统的 Transformer 结构有一些关键的差异，尤其是在如何适应目标检测任务方面。以下是对 DETR 的编码器、解码器和预测头，以及它们与经典 Transformer 结构比较的详细介绍：

**1. DETR 的编码器（Encoder）：**

- **输入处理：**
    - DETR 首先使用 CNN 骨干网络（例如 ResNet）从输入图像中提取特征图。
    - 然后，这些特征图被展平并与位置编码相结合，以提供空间信息。
    - 与经典的 Transformer 编码器（通常处理文本序列）不同，DETR 的编码器处理的是图像的二维特征图的展平表示。
- **编码器结构：**
    - DETR 的编码器基本上遵循经典的 Transformer 编码器结构，包括多头自注意力（multi-head self-attention）层和前馈网络（feed-forward networks）。
    - 自注意力机制允许编码器建模图像中不同区域之间的关系。
    - 位置编码对于编码器理解图像中目标的位置至关重要。
- **差异：**
    - 输入数据类型不同：经典Transformer處理序列資料，DETR處理圖像特徵圖。

**2. DETR 的解码器（Decoder）：**

- **目标查询（Object Queries）：**
    - DETR 的解码器使用一组可学习的“目标查询”（object queries）作为输入。
    - 这些目标查询是向量表示，它们被解码器用来定位和识别图像中的目标。
    - 目标查询本质上是模型用来寻找图像中目标的占位符。
- **解码器结构：**
    - DETR 的解码器也遵循经典的 Transformer 解码器结构，但进行了一些修改以适应目标检测。
    - 解码器中的每一层都包含多头自注意力层和多头交叉注意力层。
    - 交叉注意力层将目标查询与编码器的输出进行交互，以提取相关特征。
- **差异：**
    - 與經典transformer的不同，經典transformer的decoder是將上一層的輸出作為query輸入，DETR是使用object queries作為輸入。
    - DETR 的解码器设计用于并行预测一组目标，而不仅仅是一个序列。

**3. DETR 的预测头（Prediction Heads）：**

- **输出：**
    - 解码器的输出被送入一组预测头，以产生最终的目标检测预测。
    - 这些预测包括：
        - 边界框坐标。
        - 目标类别标签。
- **结构：**
    - 预测头通常是简单的线性层，它们将解码器的输出映射到所需的输出格式。
    - DETR的head model,是直接輸出bounding box座標與類別。
- **差異：**
    - 與經典transformer不同，經典transformer沒有輸出bounding box座標的功能。

**DETR 与经典 Transformer 的比较：**

- **输入表示：**
    - 经典 Transformer：通常处理序列数据，例如文本。
    - DETR：处理从 CNN 提取的图像特征图。
- **解码器输入：**
    - 经典 Transformer：使用先前解码器输出的序列。
    - DETR：使用可学习的目标查询。
- **输出：**
    - 經典Transformer:多用於處理序列資料的輸出。
    - DETR：输出边界框坐标和类别标签。
- **任务：**
    - 经典 Transformer：通常用于自然语言处理任务。
    - DETR：专门为目标检测任务设计。

总而言之，DETR 在 Transformer 架构的基础上进行了创新，以适应目标检测的独特要求。






**DETR 的設計：**

- DETR（DEtection TRansformer）的目標是直接進行目標檢測，它將目標檢測視為一個集合預測問題。
- 它使用 CNN（通常是 ResNet）作為 backbone 來提取圖像的特徵圖，然後將該特徵圖（加上位置編碼）展平並輸入到 Transformer 編碼器中。
- 使用特徵圖的目的是利用 CNN 在提取視覺特徵方面的強大能力。CNN 可以有效地捕獲圖像的空間結構和層次化特徵，這對於目標檢測至關重要。
- 32倍的Feature Map，代表CNN backbone將原先的影像尺寸縮小32倍，這對於降低Transformer的運算負擔有很大的幫助。
- 位置編碼（position embedding）的加入，是因為transformer本身對於輸入的序列位置不敏感，加入位置編碼可以使模型感知到圖像的空間位置關係。

**為何 DETR 是 Encoder-Decoder，而 ViT 是 Encoder-Only？**

- **DETR 的 Encoder-Decoder 結構：**
    - DETR 使用解碼器來生成目標檢測的最終預測。
    - 解碼器接收來自編碼器的特徵表示，並使用一組可學習的 object queries 來產生邊界框和類別預測。
    - 這種解碼器設計允許模型迭代地完善其預測，並且能夠處理目標之間的複雜關係。
    - 解碼器使用object queries進行與編碼器特徵的交互，進而產生最終的目標偵測結果。
- **ViT 的 Encoder-Only 結構：**
    - 對於圖像分類任務，只需要一個全局圖像表示，因此 ViT 只需要編碼器。
    - 編碼器產生的最終特徵表示被送入一個簡單的分類頭（通常是一個線性層），以產生類別預測。
    - 因為圖像分類的特性，並不需要使用解碼器進行複雜的序列結果輸出，因此使用編碼器即可。


Reference:
Transformer目标检测之DETR - 大师兄的文章 - 知乎
https://zhuanlan.zhihu.com/p/387102036