

从2014年为分割，之前的工作都是traditional detector，之后的都是DL-based detector。从以下三个方面介绍了相应的milestone detectors。

1. Traditional detectors:  Viola Jones,  HOG,  DPM
2. _CNN based Two-stage Detectors:  **RCNN series(RCNN, Fast RCNN, Faster RCNN),  SPPNet**
3. CNN based One-stage Detectors:  **YOLO series,  SSD,  RetainNet, EfficientDet, DETR, MobileNet-SSD**


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



