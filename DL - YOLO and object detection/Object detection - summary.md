

从2014年为分割，之前的工作都是traditional detector，之后的都是DL-based detector。从以下三个方面介绍了相应的milestone detectors。

1. Traditional detectors:  Viola Jones,  HOG,  DPM
2. _CNN based Two-stage Detectors:  **RCNN series(RCNN, Fast RCNN, Faster RCNN)** (Anchor)
3. CNN based One-stage Detectors:  **YOLO series,  FCOS, SSD,  RetainNet, EfficientDet, DETR, MobileNet-SSD**

New: 
[RF-DETR](https://github.com/roboflow/rf-detr)：60.5 mAP + 6ms延迟，实时检测领域的新王者如何碾压YOLO？ [link](https://zhuanlan.zhihu.com/p/32205292924)


|                 |     |
| --------------- | --- |
| [[### QA-list]] |     |

==================================================

| **模型**        | **Backbone 類型**       | Stage | Anchor                 | **說明**                                        |
| ------------- | --------------------- | ----- | ---------------------- | --------------------------------------------- |
| RCNN          | VGG<br>ResNet<br>     | 2     | Yes                    | 使用 VGG/ResNet，原始含 FC 層，檢測時移除 FC，保留特徵圖。        |
| YOLO          | Darknet<br>CSPDarknet | 1     | Yes <br><br>v8以上<br>NO | 全卷積設計，無 FC 層，專為檢測優化，直接輸出多尺度特徵圖。               |
| FCOS          | VGG<br>ResNet         | 1     | NO                     | 使用 ResNet/ResNeXt，移除 FC 層，搭配 FPN，直接適配無錨框檢測。   |
| SSD           | VGG<br>ResNet         | 1     | Yes                    | 使用 VGG，移除 FC 層，附加多尺度特徵層，適配單階段檢測。              |
| RetinaNet     | ResNet                | 1     | Yes                    | 使用 ResNet，移除 FC 層，搭配 FPN 和 Focal Loss，提升檢測精度。 |
| EfficientDet  | EfficientNet          | 1     | Yes                    | 基於 EfficientNet，移除 FC 層，加入 BiFPN，高效檢測設計。      |
| MobileNet-SSD | MobileNet             | 1     | Yes                    | 基於 MobileNet，移除 FC 層，輕量化設計，適配移動設備檢測。          |
| DETR          | Transformer           | 1     | NO                     | 使用 Transformer 架構，無 FC 層，直接基於注意力機制進行檢測。       |

[[R-CNN]]:    [[Mask R-CNN]]
VGG/ResNet 加上 RoI Pooling(或RPN + RoI Align)   
加上分類分支：FC+ Softmax,  回歸分支：FC + 邊框回歸

**MaskRCNN**: 
Backbone Model:   ResNet/VGG
Neck Model:           FPN (Feature Pyramid network)+RPN (Region proposal network) + ROI Align
Head Model:           

[[YOLO]]: Backbone: DarkNet(CSPDarkNet), Neck: PAN, Head: feature image切成網格單元, 在每個單元上進行回歸預測
[[YOLOv8]] 支持instance seg, Obb, pose等. Backbone 用C2f模組 + conv(改成用3x3).   neck用PAN取代FPN, head model: Anchor-free Decoupled Head

[[FCOS]]:  保留Faster RCNN的ResNet backbone + FPN但移除FC層, head model則改為anchor-free, 而且是在每個pixel做classification跟regression. (<mark style="background: #FFF3A3A6;">就是保留RCNN(但移除FC)但把head model改成anchor-free</mark>)

[[SSD]]: 使用 VGG(移除 FC 層)，neck不是用FPN而是添加額外卷積層成多尺度特徵層，不用region proposal而是適配單階段檢測(就是保留RCNN(但移除FC), Neck也不是用FPN而是多尺度特徵層. 也把head model改成anchor-free))。

**RetainNet**: 该网络旨在解决一阶段（one-stage）目标检测器在精度上低于两阶段（two-stage）检测器的问题，尤其是当面对极度不平衡的正负样本比例时。

**EfficientDet**:  EfficientNet (同時調整模型的depth、width和resolution) 作為backbone network, 用BiFPN（雙向特徵金字塔網路）更高效的多尺度特徵融合

**MobileNet-SSD**:  MobileNet (具有Depthwise Separable Convolution) 作為backbone network, SSD 的多尺度預測層被添加到 MobileNet 的不同層次的特徵圖

[[DETR]]: DETR 是第一個應用Transformer到object detection的Model. 流程是1. 用CNN(ResNet)生成feature images, 再經過經典Encoder, Decoder, 然後接上head model輸出bounding box跟類別. 主要結構依賴於一個Encoder-Decoder Transformer架構，直接預測一組目標邊界框。特點是不需要NMS（非極大值抑制），但訓練速度較慢。

[[RT-DETR]]: (Real-Time DETR) : 對實時目標檢測進行優化。結構上融入了YOLO系列的設計理念，在編碼器部分大量使用了類似YOLO的backbone和neck，在解碼端做了針對性的優化，使其擁有更快的處理速度，同時保持了較高的檢測精度。著重於提高推理速度，適用於需要快速響應的應用場景。

[[RF-DETR]]: 此類算法的出現都是為了增加DETR系列的檢測的精準度，將RF(receptive field)感受野融入了DETR網絡結構中，加強網路對於目標多尺度的檢測能力。在原有的DETR結構中做了結構性的改進，融合了多尺度的感受野訊息。針對檢測效果加強，但相較於RT-DETR這類實時目標檢測算法在檢測速度上會有所差異。

==================================================
其他:
[[lightweight CNN]]
[[Cross-Entropy Loss, Focal Loss, Balanced Loss]]
[[Object detection - Obb]]
[[YOLO  - Training]]


==================================================
![[Pasted image 20250317092029.png]]


### **物件偵測算法參數解釋**

| **指標**         | **說明**                                                  |
| -------------- | ------------------------------------------------------- |
| **Parms**      | 模型的參數數量（通常以百萬 M 為單位）                                    |
| **FLOPs**      | 浮點運算次數（FLOPs, Floating Point Operations），計算複雜度的指標       |
| **Latency**    | 推理延遲（通常以毫秒 ms 為單位），指的是單張影像的處理時間                         |
| **Throughput** | 吞吐量，指的是每秒可處理的影像數(FPS)                                   |
| **mAP**        | Mean Average Precision，平均精確度，衡量偵測模型的整體準確度               |
| **IoU**        | Intersection over Union，表示預測框與真實框的重疊程度                  |
| **Precision**  | 精確度，計算 TP / (TP + FP)，即正確預測的比例                          |
| **Recall**     | 召回率，計算 TP / (TP + FN)，即能夠找到所有正確物件的比例                    |
| **F1-score**   | Precision 和 Recall 的調和平均，反映模型在 Precision 和 Recall 之間的平衡 |

---

### **物件偵測算法性能比較表**

| **算法**            | **Parms (M)** | **FLOPs (G)** | **Latency (ms)** | **Throughput FPS(img/s)** | **mAP (%)** | **IoU (%)** | **Precision / Recall / F1**   |
| ----------------- | ------------- | ------------- | ---------------- | ------------------------- | ----------- | ----------- | ----------------------------- |
| **Viola-Jones**   |               | ~1 (CPU)      | 30-100           | ~10-30 (CPU)              | 30-40       | 40-60       | 低 Precision / 高 Recall / 低 F1 |
| **HOG (DPM)**     |               | ~2-10 (CPU)   | 100-500          | ~1-10 (CPU)               | 50-60       | 50-70       | 中 Precision / 中 Recall / 中 F1 |
| **DPM**           |               | ~5-20 (CPU)   | 200-500          | ~1-5 (CPU)                | 55-65       | 55-75       | 中 Precision / 中 Recall / 中 F1 |
| **R-CNN**         | 60-100        | 200-500       | 1000+            | 0.5-1                     | 60-70       | 60-80       | 高 Precision / 低 Recall / 低 F1 |
| **Fast R-CNN**    | 50-80         | 100-300       | 200-500          | 2-5                       | 70-75       | 70-85       | 高 Precision / 中 Recall / 中 F1 |
| **Faster R-CNN**  | 40-60         | 50-200        | 100-200          | 5-10                      | 75-80       | 75-85       | 高 Precision / 高 Recall / 高 F1 |
| **Cascade R-CNN** | 80            | 250           | 300              | 8                         | 85          | 88          | 高 Precision / 高 Recall / 高 F1 |
| **SPPNet**        | 40-60         | 50-150        | 150-300          | 3-7                       | 70-75       | 70-85       | 高 Precision / 中 Recall / 中 F1 |
| **YOLO v1**       | 15-20         | 30-50         | 30-50            | 20-50                     | 60-65       | 60-80       | 中 Precision / 中 Recall / 中 F1 |
| **YOLO v2**       | 20-25         | 50-100        | 20-40            | 30-60                     | 70-75       | 70-85       | 高 Precision / 中 Recall / 高 F1 |
| **YOLO v3**       | 60-75         | 100-150       | 10-20            | 50-100                    | 75-80       | 75-85       | 高 Precision / 高 Recall / 高 F1 |
| **YOLO v4**       | 65-85         | 150-200       | 8-15             | 70-150                    | 80-85       | 80-90       | 高 Precision / 高 Recall / 高 F1 |
| **YOLO v5**       | 7-140         | 20-300        | 5-10             | 100-200                   | 85-90       | 80-90       | 高 Precision / 高 Recall / 高 F1 |
| **YOLO v7**       | 5-150         | 10-350        | 3-8              | 200-500                   | 85-95       | 85-95       | 高 Precision / 高 Recall / 高 F1 |
| **YOLO v8**       | 5-200         | 15-400        | 2-5              | 300-800                   | 85-96       | 85-95       | 高 Precision / 高 Recall / 高 F1 |
| **YOLOX**         | 6             | 300           | 15               | 200                       | 90          | 92          | 高 Precision / 高 Recall / 高 F1 |
| **YOLO-NAS**      | 7             | 400           | 10               | 400                       | 95          | 94          | 高 Precision / 高 Recall / 高 F1 |
| **YOLO v8n**      | 3.2           | 4             | 3                | 1000                      | 88          | 88          | 高 Precision / 高 Recall / 高 F1 |
| **SSD**           | 20-30         | 50-100        | 20-50            | 30-60                     | 75-80       | 75-85       | 高 Precision / 中 Recall / 高 F1 |
| **RetinaNet**     | 30-50         | 100-150       | 50-100           | 5-10                      | 75-85       | 75-90       | 高 Precision / 高 Recall / 高 F1 |
| **EfficientDet**  | 4             | 100           | 50               | 100                       | 82          | 85          | 高 Precision / 高 Recall / 高 F1 |
| **DETR**          | 50            | 150           | 300              | 20                        | 85          | 88          | 高 Precision / 高 Recall / 高 F1 |
| **MobileNet-SSD** | 6             | 2             | 15               | 600                       | 75          | 78          | 高 Precision / 高 Recall / 高 F1 |
| **SqueezeNet**    | 1.25          | 0.5           | 2                | 1200                      | 70          | 75          | 高 Precision / 高 Recall / 高 F1 |

### **分析與結論**

1. **早期方法 (Viola-Jones, HOG, DPM)**
    
    - 這些方法基於手工特徵（如 Haar-like, HOG），雖然計算量低，但偵測精度較差，對複雜物件（如不同角度的人臉）較難處理。
    - DPM（Deformable Part Model）通過多尺度特徵提升了準確度，但計算開銷仍然較大。
2. **R-CNN 系列（R-CNN, Fast R-CNN, Faster R-CNN, SPPNet）**
    
    - R-CNN 是第一個基於深度學習的物件偵測模型，但計算速度慢，因為它需要為每個 Region Proposal 運行 CNN。
    - Fast R-CNN 使用 RoI Pooling 優化了計算，速度提高約 10 倍。
    - Faster R-CNN 通過 RPN（Region Proposal Network）直接從 CNN 生成 proposals，使得速度更快，並且成為影像辨識領域的標準方法之一。
    - SPPNet 使用 Spatial Pyramid Pooling 改進了 Fast R-CNN，能夠處理不同尺寸的輸入影像。
3. **YOLO 系列（YOLO v1~v8）**
    
    - YOLO 是單階段（One-stage）的物件偵測方法，將物件偵測轉化為回歸問題，使得速度快但早期版本的精度較低。
    - YOLO v3+ 版本大幅度提升了 mAP，同時保持高速運行，適合即時應用場景（如自駕車）。
    - YOLO v5+ 進一步提升計算效率，適用於邊緣設備（Edge AI）。
4. **SSD vs RetinaNet**
    
    - SSD（Single Shot MultiBox Detector）與 YOLO 類似，但使用多尺度特徵圖來提高小物件偵測能力。
    - RetinaNet 使用 Focal Loss 解決了單階段檢測中前景-背景不均衡的問題，提高了檢測精度。

總結來說，若需即時應用則 YOLO v8 是目前最佳選擇，而如果需要高精度但可接受較高計算成本，Faster R-CNN 或 RetinaNet 是較好的選擇。


在物件偵測領域，模型的選擇對於性能和效率有很大的影響。以下是對您提到的幾個物件偵測模型的詳細介紹，包括它們的特點以及它們屬於 one-stage 或 two-stage：

**1. SSD (Single Shot MultiBox Detector)**

- **特點：**
    - One-stage 偵測器：直接從特徵圖預測目標的邊界框和類別。
    - 多尺度特徵圖：使用不同層次的特徵圖進行預測，以檢測不同尺寸的目標。
    - 預設框（Default Boxes）：在每個特徵圖單元位置放置一組預設框，用於加速目標定位。
    - 具有快速的檢測速度，適合即時應用。
- **Stage：** One-stage。

**2. RetinaNet**

- **特點：**
    - One-stage 偵測器。
    - 引入 Focal Loss：解決 one-stage 偵測器中前景和背景類別不平衡的問題，提高檢測精度。
    - Feature Pyramid Network (FPN)：利用多尺度特徵圖，提高對不同尺寸目標的檢測能力。
- **Stage：** One-stage。

**3. EfficientDet**

- **特點：**
    - One-stage 偵測器。
    - 基於 EfficientNet 骨幹網路：通過複合縮放（compound scaling）方法，平衡模型的精度和效率。
    - BiFPN（Bi-directional Feature Pyramid Network）：改進 FPN 結構，實現更高效的多尺度特徵融合。
- **Stage：** One-stage。

**4. DETR (DEtection TRansformer)**

- **特點：**
    - One-stage 偵測器（但結構與傳統 one-stage 不同）。
    - 基於 Transformer 架構：將目標檢測視為一個集合預測問題，利用 Transformer 的注意力機制進行目標定位和分類。
    - 無需預設框或 Non-Maximum Suppression (NMS)：直接預測一組目標，並使用集合損失函數進行訓練。
    - 相對於傳統物件偵測模型有較為不同的運作方法。
- **Stage：** One-stage (transformer based)。

**5. MobileNet-SSD**

- **特點：**
    - One-stage 偵測器。
    - 結合 MobileNet 和 SSD：使用 MobileNet 作為骨幹網路，以實現輕量級和高效的目標檢測。
    - 適用於行動裝置和嵌入式系統。
- **Stage：** One-stage。

**6. SqueezeNet**

- **特點：**
    - 輕量級 CNN 架構，主要用於圖像分類。
    - 使用 "fire modules" 減少模型的參數數量，同時保持較高的準確度。
    - 有論文將squeezenet當作backbone來做object detection, 產生輕量的物件偵測模型。
    - 相較於上面提及的模型，在物件偵測上的應用較少。
- 不算是object detection的模型。常作為backbone。

**One-Stage 與 Two-Stage 的區別：**

- **One-Stage：**
    - 直接從輸入圖像預測目標的邊界框和類別。
    - 速度快，適合即時應用。
    - 但通常精度稍遜於 two-stage 偵測器。
- **Two-Stage：**
    - 首先生成候選區域（Region Proposals）。
    - 然後對候選區域進行分類和邊界框回歸。
    - 精度高，但速度較慢。




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









==========================================

Reference:

史上最详细目标检测方法介绍，包含传统目标检测，RCNN系列，SSD系列以及YOLO系列，检测数据集，相关的比赛等等（持续更新中...） - 初识CV的文章 - 知乎
https://zhuanlan.zhihu.com/p/694572995

请问各位大佬近几年有什么好的目标检测baseline呢（我导师不让用yolo）？ - 哥廷根数学学派的回答 - 知乎
https://www.zhihu.com/question/6550680651/answer/117712831185

精读一篇目标检测综述-Object Detection in 20 Years: A Survey - yearn的文章 - 知乎
https://zhuanlan.zhihu.com/p/192362333

收藏！目标检测优质综述论文总结！ - Sophia的文章 - 知乎
https://zhuanlan.zhihu.com/p/387216781

【object detection】目标检测之SSD - 小橘子的文章 - 知乎
https://zhuanlan.zhihu.com/p/30478644

一文读懂EfficientDet - william的文章 - 知乎
https://zhuanlan.zhihu.com/p/208974735

DINO系列工作回顾（DETR, DINO, Grounding DINO, DINO-X） - Derek Z的文章 - 知乎
https://zhuanlan.zhihu.com/p/28579286730

Object Detection综述 - KHao426的文章 - 知乎
https://zhuanlan.zhihu.com/p/266296069

第一卷-目标检测大杂烩
https://www.zhihu.com/column/c_1178388040400302080

《目标检测大杂烩》-第14章-浅析RT-DETR - Kissrabbit的文章 - 知乎
https://zhuanlan.zhihu.com/p/626659049


### QA-list

| Q                                                                 | Ans |
| ----------------------------------------------------------------- | --- |
| DETR是怎么做的                                                         |     |
| 能不能讲下二分图匹配（Sinkhorn原理）                                            |     |
| 做NMS-free还有什么策略                                                   |     |
| Coding>>> **NMS過程**                                               |     |
| NMS的細節, SoftNMS跟NMS的差別                                            |     |
| YOLOv4用到哪些优化方法？                                                   |     |
| 谈谈YOLO系列的改进, 并且解释一下YOLO为什么可以这么快？                                  |     |
| 介绍下NMS，并写出NMS的伪代码，和计算IOU的函数                                       |     |
| 谈谈SSD中是如何确定正负样本的？                                                 |     |
| 相比于Anchor-based方法，Anchor-free的优势在哪？                               |     |
| faster RCNN原理介绍                                                   |     |
| RPN（Region Proposal Network）网络的作用、实现细节                            |     |
| 说一下RoI Pooling是怎么做的？有什么缺陷？有什么作用                                   |     |
| Faster R-CNN是如何解决正负样本不平衡的问题？                                      |     |
| faster-rcnn中bbox回归用的是什么公式，说一下该网络是怎么回归bbox的？                       |     |
| 简述faster rcnn的前向计算过程 简述faster rcnn训练步骤                            |     |
| Faster rcnn有什么不足的地方吗？如何改进？                                        |     |
| 简要阐述一下One-Stage、Two-Stage模型                                       |     |
| YOLOV1、YOLOV2、YOLOV3复述一遍 YOLOv1到v3的发展历程以及解决的问题                    |     |
| 简要阐述一下FPN网络具体是怎么操作的 FPN网络的结构。                                     |     |
| 简要阐述一下SSD网络, 阐述一下ssd和retinanet的区别                                 |     |
| 简要阐述一下RetinaNet                                                   |     |
| faster rcnn和yolo，ssd之间的区别和联系                                      |     |
| 分析一下SSD,YOLO,Faster rcnn等常用检测网络对小目标检测效果不好的原因                      |     |
| Coding>>> **IoU算法**                                               |     |
| 讲一下目标检测优化的方向                                                      |     |
| anchor设置的意义                                                       |     |
| 如果只能修改RPN网络的话，怎么修改可以提升网络小目标检出率                                    |     |
| 如何理解concat和add这两种常见的feature map特征融合方式                             |     |
| 阐述一下如何检测小物体                                                       |     |
| 阐述一下目标检测任务中的多尺度                                                   |     |
| 如果有很长，很小，或者很宽的目标，应该如何处理目标检测中如何解决目标尺度大小不一的情况 小目标不好检测               |     |
| 检测的框角度偏移了45度，这种情况怎么处理                                             |     |
| YOLO、SSD和Faster-RCNN的区别，他们各自的优势和不足分别是什么？                          |     |
| 介绍一下CenterNet的原理，它与传统的目标检测有什么不同点？                                 |     |
| CenterNet中heatmap（热力图）如何生成？                                       |     |
| 你最常用的几种目标检测算法是什么？为什么选择这些算法，你选择它们的场景分别是什么？                         |     |
| yolov4和v5均引入了CSP结构，介绍一下它的原理和作用                                    |     |
| EfficientDet为什么可以做到速度兼精度并存                                        |     |
| 介绍Faster R-CNN和Cascade R-CNN                                      |     |
| SSD相比于YOLO做了哪些改进？                                                 |     |
| 了解哪些开源的移动端轻量型目标检测？                                                |     |
| 目标检测单阶段和双阶段优缺点，双阶段的为什么比单阶段的效果要好？                                  |     |
| 目标检测中如何处理正负样本不平衡的问题？                                              |     |
| 和SSD比网络更加深了，虽然anchors比SSD少了许多，但是加深的网络深度明显会增加更多的计算量，那么为什么会比SSD快3倍？ |     |
| 你认为当前目标检测算法发展的趋势是什么？现阶段存在什么难点？                                    |     |
| DETR用二分图匹配实现label assignment，简述其过程                                |     |
| 如何修改Yolov5目标检测，从而实现旋转目标检测？                                        |     |
| 在目标Crowded的场景下，经常在两个真正目标中间会出现误检的原因?                               |     |
| 如果在分类任务中几个类别有重叠（类间差异小）怎么办，如何设计网络结构                                |     |
| Anchor-free的target assign怎么解决多个目标中心点位置比较靠近的问题                     |     |
| 目标检测设置很多不同的anchor，能否改善小目标及非正常尺寸目标的性能，除计算速度外还存在什么问题                |     |
| Focal loss的参数如何调，以及存在什么问题                                         |     |
| Yolov5中的objectness的作用                                             |     |
| 目标检测中旋转框IOU的计算方式                                                  |     |
| DETR的检测算法的创新点                                                     |     |
| FCOS如何解决重叠样本，以及centerness的作用                                      |     |

一位算法工程师从30+场秋招面试中总结出的超强面经——目标检测篇（含答案）
https://zhuanlan.zhihu.com/p/374017926