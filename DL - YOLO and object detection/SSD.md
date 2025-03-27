
## **一、主流的算法主要分为两个类型**

1.two-stage方法：如R-CNN系列算法，其主要思路是先通过[启发式方法](https://zhida.zhihu.com/search?content_id=163259505&content_type=Article&match_order=1&q=%E5%90%AF%E5%8F%91%E5%BC%8F%E6%96%B9%E6%B3%95&zhida_source=entity)（selective search）或者 CNN 网络（RPN)产生一系列稀疏的候选框，然后对这些候选框进行分类(classification)与回归(bounding box regression)，two-stage方法的优势是准确度高； 

2.one-stage方法：如YOLO和SSD，其主要思路是均匀地在图片多个层数的特征图上进行密集抽样，抽样时可以采用不同尺度和长宽比，然后利用CNN提取特征后直接进行分类与回归，整个过程只需要一步，所以其优势是速度快。但是均匀的[密集采样](https://zhida.zhihu.com/search?content_id=163259505&content_type=Article&match_order=1&q=%E5%AF%86%E9%9B%86%E9%87%87%E6%A0%B7&zhida_source=entity)的一个重要缺点是训练比较困难，这主要是因为正样本与负样本（背景）极其不均衡，导致模型准确度稍低。不同算法的性能如图1所示，可以看到两类方法在准确度和速度上的差异。

![[Pasted image 20250317092029.png]]

![[Pasted image 20250325154804.png]]

共有两种SSD网络：SSD 300和SSD 512，用于不同输入尺寸的图像识别。下文主要以 SSD 300为例进行分析。


![[Pasted image 20250325155023.png]]




SSD 300 中输入图像的大小是 300x300，[特征提取](https://zhida.zhihu.com/search?content_id=163259505&content_type=Article&match_order=1&q=%E7%89%B9%E5%BE%81%E6%8F%90%E5%8F%96&zhida_source=entity)部分使用了 VGG16 的[卷积层](https://zhida.zhihu.com/search?content_id=163259505&content_type=Article&match_order=1&q=%E5%8D%B7%E7%A7%AF%E5%B1%82&zhida_source=entity)，并将 VGG16的两个[全连接层](https://zhida.zhihu.com/search?content_id=163259505&content_type=Article&match_order=1&q=%E5%85%A8%E8%BF%9E%E6%8E%A5%E5%B1%82&zhida_source=entity)转换成了普通的卷积层（图中conv6和conv7），之后又接了多个卷积（conv8_1，conv8_2，conv9_1，conv9_2，conv10_1，conv10_2），最后用一个Global Average Pool来变成1x1的输出（conv11_2）。 从图中我们可以看出，SSD将[conv4_3](https://zhida.zhihu.com/search?content_id=163259505&content_type=Article&match_order=1&q=conv4_3&zhida_source=entity)、conv7、conv8_2、conv9_2、conv10_2、conv11_2都连接到了最后的检测分类层做回归。具体细节如图3所示：

![[Pasted image 20250325155053.png]]

观察上述两幅图，我们可以初步得到SSD网络预测过程的基本步骤：

1. 输入一幅图片（300x300），将其输入到预训练好的[分类网络](https://zhida.zhihu.com/search?content_id=163259505&content_type=Article&match_order=1&q=%E5%88%86%E7%B1%BB%E7%BD%91%E7%BB%9C&zhida_source=entity)（改进的传统的VGG16 网络）中来获得不同大小的[特征映射](https://zhida.zhihu.com/search?content_id=163259505&content_type=Article&match_order=1&q=%E7%89%B9%E5%BE%81%E6%98%A0%E5%B0%84&zhida_source=entity)；
2. 抽取Conv4_3、Conv7、Conv8_2、Conv9_2、Conv10_2、Conv11_2层的feature map，然后分别在这些feature map层上面的每一个点构造6个不同尺度大小的Default boxes。然后分别进行检测和分类，生成多个初步符合条件的Default boxes；
3. 将不同feature map获得的Default boxes结合起来，经过NMS（[非极大值抑制](https://zhida.zhihu.com/search?content_id=163259505&content_type=Article&match_order=1&q=%E9%9D%9E%E6%9E%81%E5%A4%A7%E5%80%BC%E6%8A%91%E5%88%B6&zhida_source=entity)）方法来抑制掉一部分重叠或者不正确的Default boxes，生成最终的Default boxes 集合（即检测结果）；

这是通过观察 SSD 网络结构得出的大概流程。




| Backbone |                                                                                                                                                                     |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| RCNN     | 使用預訓練的 CNN，如 VGG16 或 ResNet50，這些模型最初為圖像分類設計，包含多層卷積和池化層。<br>設計重點：利用分類模型的強大特徵提取能力，提取深層特徵圖（如 ResNet 的 Stage 4，2048 通道，7x7）。<br>與分類的關係：原始包含全連接層（FC）用於分類，檢測時移除 FC，保留卷積層。 |
| FCOS     | 使用現代預訓練 CNN，如 ResNet50/101 或 ResNeXt，這些模型為分類設計，但檢測時移除全連接層。<br>設計重點：提取多層特徵圖（如 ResNet 的 Stage 2-4），提供從低層細節到高層語義的層次化特徵。<br>與分類的關係：移除 GAP 和 FC 層，保留卷積層，適應空間定位需求。        |
| SSD      | 使用 VGG16 作為基礎，移除全連接層（FC），保留卷積層（如 conv1-5）。<br>設計重點：VGG16 提供深層特徵，但參數量大（約 138M），後期版本可替換為輕量化模型（如 MobileNet）。<br>與分類的關係：移除 FC 層，適應空間定位，與 FCOS 類似但結構更早。                  |

| Neck |                                                                                                                                                                                                        |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| RCNN | 通常無明確 neck 結構。在 Fast RCNN 中，區域建議（region proposals）由外部方法（如選擇性搜索）生成，backbone 的特徵圖直接用於後續處理。<br>RoI pooling 可視為 neck 的一部分，將每個區域建議的特徵映射到固定尺寸（例如 7x7），但這更常被視為 head 的組成部分。<br>設計重點：無需特徵融合，簡單直接，適合兩階段檢測的區域處理。  |
| FCOS | 使用 FPN（Feature Pyramid Network），從 backbone 的不同階段（通常 Stage 2-4）提取特徵圖，通過上採樣和側連接生成多尺度特徵金字塔。<br>設計重點：FPN 融合高層語義資訊（高層特徵）和低層空間細節（低層特徵），適應不同尺寸目標。<br>與 SSD 的差異：FPN 是標準化設計，相比 SSD 的額外卷積層更系統化。<br>與 RCNN 的差異：一樣 |
| SSD  | 添加額外卷積層（conv6-11），從 backbone 的不同層提取特徵，形成多尺度特徵塔。<br>設計重點：這些層逐步下採樣，生成不同分辨率的特徵圖（如 38x38、19x19、10x10），適應多尺寸目標。<br>與 FCOS 的差異：SSD 的 neck 較為簡單，無 FPN，依賴預定義錨框。                                                |


| Head |                                                                                                                                                                                                                                                    |
| ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| RCNN | 包括 RoI pooling 和全連接層。RoI pooling 從特徵圖中提取每個區域建議的固定尺寸特徵向量，接著通過兩分支全連接層進行：<br>分類分支：輸出類別概率（softmax）。<br>回歸分支：輸出邊界框偏移量（bounding box regression）。<br>設計重點：兩階段設計確保高精度，適合複雜場景，但計算成本高（每個區域獨立處理）。<br>與現代頭部的差異：RCNN 的 head 依賴區域建議，與單階段模型（如 SSD、FCOS）無錨框設計形成對比。 |
| FCOS | 對 FPN 每個層的特徵圖應用卷積層，預測三個部分：<br>中心性（centerness）：評估位置是否為物件中心，過濾邊緣預測。<br>類別概率：每個位置的物件類別（softmax）。<br>邊界框距離：到物件四邊的距離（4 個值），用於回歸。<br>設計重點：無錨框（anchor-free）設計，簡化超參數調整，基於像素級預測，提升模型靈活性。<br>與 RCNN/SSD 的差異：FCOS 的 head 無需預定義錨框，減少計算複雜度，適合密集場景。              |
| SSD  | 每個特徵圖細胞預測多個預定義錨框（default boxes），每個錨框輸出：<br>類別概率：softmax 輸出。<br>邊界框偏移量：回歸邊界框位置。<br>設計重點：基於錨框的多尺度預測，適合實時應用，但錨框設計需調參，複雜度高於 FCOS。<br>與 RCNN 的差異：單階段設計，無區域建議，速度快於兩階段模型。                                                                                 |

