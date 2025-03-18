


|                                         |     |
| --------------------------------------- | --- |
| [[###YOLO 模型架構概述]]                      |     |
| [[###YOLO跟RCNN的backbone, neck, head比較]] |     |
| [[###YOLO 系列模型版本演進]]                    |     |
|                                         |     |
|                                         |     |
|                                         |     |
|                                         |     |
|                                         |     |
|                                         |     |

**YOLO 系列演進表**

| 模型       | Backbone     | Neck         | Head                     | 主要特點與功能                      | 其他功能           | 效能提升                    |
| :------- | :----------- | :----------- | :----------------------- | :--------------------------- | :------------- | :---------------------- |
| YOLOv1   | GoogLeNet    | 無            | 全連接層                     | 單階段檢測、快速                     | 物件偵測           | 開創性，速度快但精度較低            |
| YOLOv2   | Darknet-19   | 無            | 卷積層                      | Anchor boxes、多尺度訓練           | 物件偵測           | 精度與速度提升                 |
| YOLOv3   | Darknet-53   | FPN          | 卷積層                      | 多尺度檢測、更好的小目標檢測               | 物件偵測           | 小目標檢測能力提升               |
| YOLOv4   | CSPDarknet53 | SPP、PAN      | 卷積層                      | CSP 結構、Mosaic 資料增強           | 物件偵測           | 精度大幅提升                  |
| YOLOv5   | CSPDarknet53 | SPP、PAN      | 卷積層                      | PyTorch 實現、自動化 Anchor boxes  | 物件偵測           | 易於使用，速度與精度平衡            |
| YOLOX    | Darknet53    | FPN          | Decoupled Head           | Anchor-free、SimOTA 標籤分配      | 物件偵測           | 精度與速度的進一步提升             |
| YOLOv6   | EfficientRep | Rep-PAN      | Efficient Decoupled Head | Rep 算子、硬體友善                  | 物件偵測           | 針對工業應用優化                |
| YOLOv7   | ELAN         | SPP、PAN      | 卷積層                      | ELAN 結構、輔助訓練頭                | 物件偵測           | 在速度和準確性方面都超越了所有已知的目標檢測器 |
| YOLOv8   | C2f          | C2f、SPPF、PAN | Decoupled Head           | Anchor-free、任務模式切換(分類，分割，檢測) | 物件偵測，圖像分割，圖像分類 | 多功能、易於使用                |
| YOLO-NAS | NAS 搜尋       | PAN          | Decoupled Head           | 神經架構搜尋、最佳硬體延遲                | 物件偵測           | 針對特定硬體平台優化              |



YOLO（You Only Look Once）是一種單階段（single-stage）的目標檢測（Object Detection）模型，相較於傳統的 R-CNN 或 Fast R-CNN，它的最大特點是將物件偵測視為單一回歸問題（regression problem），直接從輸入影像推斷出邊界框（bounding boxes）與類別機率（class probabilities），而不需要額外的候選區域生成（Region Proposal）。

---

### YOLO 模型架構概述

YOLO 的核心思想是將影像劃分為 **S × S** 網格，每個網格負責偵測位於其內的物件，並輸出：

1. **Bounding Box**（邊界框）：每個網格預測 `B` 個邊界框，每個框包含：
    
    - 中心座標 (x,y)(x, y)(x,y)（相對於該網格的歸一化座標）
    - 寬度 www、高度 hhh（相對於整個影像的歸一化座標）
    - 物件置信度（object confidence）：表示該框內是否有物件，以及框的準確性
2. **Class Probabilities**（類別機率）：對於每個網格，YOLO 預測所有類別的機率。
    

YOLO 的架構通常包含以下幾個部分：

1. **Backbone**：特徵提取網路，如 Darknet、MobileNet、EfficientNet、CSPNet 等。
2. **Neck**：加強特徵表達的網路，如 FPN（Feature Pyramid Network）、PAN（Path Aggregation Network）。
3. **Head**：輸出物件的邊界框和類別，通常為卷積層。

---

## YOLO 架構詳解（以 YOLOv3 為例）

### 1. **輸入層（Input Layer）**

- 影像被調整為固定大小（例如 416×416416 \times 416416×416 或 608×608608 \times 608608×608）。
- 歸一化（Normalization）：將像素值縮放到 [0,1][0,1][0,1] 或標準化到 [−1,1][-1,1][−1,1]。
- 資料增強（Data Augmentation）：隨機縮放、翻轉、裁剪等。

---

### 2. **Backbone（特徵提取網路）**

YOLO 使用一個深度卷積神經網路（CNN）來提取影像特徵，例如：

- **YOLOv1, v2**：使用 **Darknet**（類似 VGG）
- **YOLOv3, v4**：使用 **Darknet-53**
- **YOLOv5**：使用 **CSPNet**
- **YOLOv6, v7, v8**：採用了不同的高效 CNN 結構，如 EfficientNet、MobileNet

**Darknet-53（YOLOv3）細節**

- **53 層卷積神經網路**
- 主幹架構類似 **ResNet**，包含「殘差連接（Residual Connections）」。
- 主要使用 **3×3** 和 **1×1** 卷積層組成的 **Basic Residual Blocks**。
- 採用 **Batch Normalization（BN）** 和 **Leaky ReLU**。

---

### 3. **Neck（特徵融合）**

YOLO 在 Backbone 提取的特徵基礎上，使用 **Feature Pyramid Network (FPN) 和 Path Aggregation Network (PAN)** 來提升特徵表達能力：

- **FPN（Feature Pyramid Network）**
    - 從不同層級提取多尺度特徵（例如 13×13, 26×26, 52×52）。
    - 提供高解析度和低解析度的資訊。
- **PAN（Path Aggregation Network）**
    - 提供額外的路徑來強化資訊流動，提高偵測效果。

在 YOLOv3 及之後的版本，每個網格對應三種不同的尺度：

- **小型物件：** 52×5252 \times 5252×52 特徵圖
- **中型物件：** 26×2626 \times 2626×26 特徵圖
- **大型物件：** 13×1313 \times 1313×13 特徵圖

這種多尺度偵測方式讓 YOLO 在不同大小的物件上有較好的效果。

---

### 4. **Head（輸出層）**

YOLO 的輸出層主要由 **卷積層（Convolutional layers）** 組成：

- 使用 **1×1 卷積層** 降維。
- 最終輸出尺寸為 S×S×(B×(5+C))S \times S \times (B \times (5 + C))S×S×(B×(5+C))，其中：
    - **S × S** 是網格數（例如 13×13, 26×26, 52×52）。
    - **B** 是每個網格預測的邊界框數（通常為 3）。
    - **5 + C**：
        - 5 表示邊界框資訊（x,y,w,h,confidencex, y, w, h, \text{confidence}x,y,w,h,confidence）。
        - C 表示物件類別數量（例如 COCO 資料集有 80 類，則 C=80）。

**輸出格式（每個 Bounding Box）**

(x,y,w,h,confidence,p1,p2,...,pC)(x, y, w, h, confidence, p_1, p_2, ..., p_C)(x,y,w,h,confidence,p1​,p2​,...,pC​)

其中：

- (x,y)(x, y)(x,y) 是邊界框中心的相對座標。
- (w,h)(w, h)(w,h) 是邊界框的寬高（相對於整張影像）。
- **置信度（confidence score）** 表示該框內是否有物件，計算方式： P(Object)×IOU(Pred,GT)P(\text{Object}) \times IOU(\text{Pred}, \text{GT})P(Object)×IOU(Pred,GT)
- pip_ipi​ 是物件類別的機率。

YOLO 使用 **Anchor Boxes（錨點框）**，每個網格對應三個 anchor boxes，這樣可以適應不同尺寸的物件。

---

## YOLO 演算法流程

1. **輸入影像**
    
    - 影像縮放到固定大小，如 416×416416 \times 416416×416。
    - 經過 CNN Backbone 提取特徵。
2. **多尺度特徵提取**
    
    - FPN / PAN 負責融合不同解析度的特徵。
3. **Bounding Box 預測**
    
    - 預測 (x,y,w,h)(x, y, w, h)(x,y,w,h) 和物件置信度（Object Confidence）。
4. **類別分類**
    
    - 使用 Softmax（YOLOv2）或 Sigmoid（YOLOv3 及以後）來計算類別機率。
5. **非極大值抑制（Non-Maximum Suppression, NMS）**
    
    - 過濾低置信度的邊界框。
    - 移除重疊度（IOU）過高的框。
6. **輸出最終結果**
    
    - 邊界框座標 + 物件類別 + 置信度。

---

## 版本演進比較

|YOLO 版本|Backbone|主要改進點|
|---|---|---|
|YOLOv1|Custom CNN|單一尺度輸出|
|YOLOv2|Darknet-19|Anchor Boxes、多尺度輸出|
|YOLOv3|Darknet-53|FPN 多尺度偵測、Softmax 改為 Sigmoid|
|YOLOv4|CSPDarknet53|PANet、多尺度、Mish 激活函數|
|YOLOv5|CSPNet|更小、更快、更準確|
|YOLOv6|EfficientRep|更輕量的結構|
|YOLOv7|E-ELAN|訓練效率提高|
|YOLOv8|Custom CNN|進一步改進精度和速度|

---

## 結論

YOLO 透過單一前向傳播（single forward pass）直接預測物件的類別與邊界框，避免了傳統方法的區域提議（Region Proposal）步驟，使其比 R-CNN 家族（Faster R-CNN）更快。最新的 YOLO 版本（YOLOv8）進一步提升了準確度和速度，成為最流行的物件偵測模型之一。





### YOLO跟RCNN的backbone, neck, head比較

✅ **都有 Backbone**：兩者都使用 CNN 作為特徵提取網路，例如：

- Faster R-CNN 及其變種常用 **ResNet、VGG、EfficientNet**。
- YOLO 不同版本使用 **Darknet、CSPNet、EfficientRep** 等。

✅ **都有 Neck（特徵金字塔）**

- **R-CNN 家族**（特別是 Faster R-CNN 及之後的版本）使用 **FPN（Feature Pyramid Network）** 來強化不同尺度的特徵。
- **YOLO（特別是 YOLOv4+）** 也使用 **FPN**，但進一步加入 **PAN（Path Aggregation Network）** 來強化資訊流。

---

### **主要差異**

|比較項目|YOLO|R-CNN 系列（Faster R-CNN）|
|---|---|---|
|**Head 模組**|直接預測 Bounding Boxes|透過 **RPN** 產生候選區域（Region Proposals）|
|**區域提議機制**|**沒有 RPN**，直接回歸 Bounding Boxes|**有 RPN**，先產生候選框再分類|
|**計算效率**|單階段（Single-stage），速度快|雙階段（Two-stage），速度較慢但通常更準確|
|**輸出特徵圖**|多尺度輸出，使用 Anchor Boxes|先產生 RoI（Region of Interest）|
|**適合場景**|即時應用（如自駕車、監控）|需要高準確度的應用（如醫學影像）|

---

### **細節修正**

🔹 **YOLO 的 Head 並非使用 PAN 來直接生成 Bounding Boxes**

- **PAN（Path Aggregation Network）** 主要是強化 FPN 特徵，讓不同尺度的特徵更有效地傳遞，而不是直接負責預測框。
- **Bounding Boxes 其實是由最後的 1×1 卷積層（YOLO Head）輸出**，這層會產生每個 Anchor Box 的 `(x, y, w, h, confidence, class probabilities)`。

🔹 **Faster R-CNN 的 RPN 不是最後的輸出**

- **RPN（Region Proposal Network）** 只負責產生「候選區域」。
- 產生的候選框會被送入 RoI Align/RoI Pooling，然後再進行分類與框微調（Bounding Box Regression）。

---

### **結論**

你的理解大方向是正確的：

- **相同點**：兩者都使用 CNN Backbone + FPN。
- **不同點**：YOLO 直接預測 Bounding Boxes，Faster R-CNN 則透過 RPN 產生候選區域後再分類。

但細節上，YOLO 的 PAN 主要是輔助特徵學習，而不是直接生成框，這點要特別注意。




### YOLO 系列模型版本演進

以下為您整理 YOLO 系列模型從 v1 至最新版本的演進，並包含 YOLOX、YOLO-NAS 等重要模型，以及效能比較：

**YOLO 系列演進表**

|模型|Backbone|Neck|Head|主要特點與功能|其他功能|效能提升|
|:--|:--|:--|:--|:--|:--|:--|
|YOLOv1|GoogLeNet|無|全連接層|單階段檢測、快速|物件偵測|開創性，速度快但精度較低|
|YOLOv2|Darknet-19|無|卷積層|Anchor boxes、多尺度訓練|物件偵測|精度與速度提升|
|YOLOv3|Darknet-53|FPN|卷積層|多尺度檢測、更好的小目標檢測|物件偵測|小目標檢測能力提升|
|YOLOv4|CSPDarknet53|SPP、PAN|卷積層|CSP 結構、Mosaic 資料增強|物件偵測|精度大幅提升|
|YOLOv5|CSPDarknet53|SPP、PAN|卷積層|PyTorch 實現、自動化 Anchor boxes|物件偵測|易於使用，速度與精度平衡|
|YOLOX|Darknet53|FPN|Decoupled Head|Anchor-free、SimOTA 標籤分配|物件偵測|精度與速度的進一步提升|
|YOLOv6|EfficientRep|Rep-PAN|Efficient Decoupled Head|Rep 算子、硬體友善|物件偵測|針對工業應用優化|
|YOLOv7|ELAN|SPP、PAN|卷積層|ELAN 結構、輔助訓練頭|物件偵測|在速度和準確性方面都超越了所有已知的目標檢測器|
|YOLOv8|C2f|C2f、SPPF、PAN|Decoupled Head|Anchor-free、任務模式切換(分類，分割，檢測)|物件偵測，圖像分割，圖像分類|多功能、易於使用|
|YOLO-NAS|NAS 搜尋|PAN|Decoupled Head|神經架構搜尋、最佳硬體延遲|物件偵測|針對特定硬體平台優化|

![[Pasted image 20250315215556.png]]

**YOLO 模型效能指標概覽**

| 模型         | Params (百萬) | FLOPs (十億) | Latency (ms) | Throughput (FPS) | mAP (COCO) | IoU | 主要特性               |
| :--------- | :---------- | :--------- | :----------- | :--------------- | :--------- | :-- | :----------------- |
| YOLOv1     | 24          | -          | -            | 45               | 63.4       | -   | 快速單階段檢測            |
| YOLOv2     | 50          | -          | -            | 67               | 78.6       | -   | Anchor boxes、多尺度訓練 |
| YOLOv3     | 62          | -          | -            | 33               | 57.9       | -   | 多尺度檢測、FPN          |
| YOLOv4     | 64          | -          | -            | 65               | 65.7       | -   | CSPDarknet、Mosaic  |
| YOLOv5s    | 7.2         | 16.5       | -            | -                | 37.4       | -   | PyTorch、易用性        |
| YOLOv5m    | 21.2        | 49.0       | -            | -                | 44.9       | -   | 速度/精度平衡            |
| YOLOX-s    | 9.0         | 26.8       | -            | -                | 40.5       | -   | Anchor-free、SimOTA |
| YOLOX-m    | 25.3        | 73.7       | -            | -                | 46.4       | -   | 高效、高精度             |
| YOLOv6-s   | 25.0        | -          | -            | -                | 43.5       | -   | Rep算子、硬體友善         |
| YOLOv7     | 36.9        | -          | -            | -                | 56.8       | -   | ELAN、輔助訓練頭         |
| YOLOv8s    | 11.2        | -          | -            | -                | 44.9       | -   | 多任務、易用             |
| YOLOv8m    | 25.9        | -          | -            | -                | 50.2       | -   | 多任務、高精度            |
| YOLO-NAS-s | 8.3         | -          | -            | -                | 47.1       | -   | NAS搜尋，硬體優化         |


**重要模型特點補充**

- **YOLOX：**
    - 引入 Anchor-free 的設計，簡化了訓練流程。
    - 採用 SimOTA 動態標籤分配策略，提升了訓練效率。
    - Decoupled Head 的設計，將分類和回歸任務分離，提高了精度。
- **YOLO-NAS：**
    - 使用<mark style="background: #ADCCFFA6;">神經架構搜尋（NAS）技術</mark>，自動尋找最佳的網路結構。
    - 針對特定硬體平台進行延遲優化，適合部署在邊緣裝置。
- **YOLOv8:**
    - YOLOv8是一個多功能的模型，不僅支援物件檢測，還<mark style="background: #BBFABBA6;">支援圖像分割和圖像分類任務</mark>。
    - 模型結構也進行了修改，使用C2f模塊，並改變了損失函數。

**效能比較**

由於 YOLO 系列模型眾多，且在不同資料集和硬體上的效能表現有所差異，因此難以提供絕對的效能排名。但總體而言：

- YOLOv1 速度最快，但精度較低。
- YOLOv3 在小目標檢測方面有顯著提升。
- YOLOv4 和 YOLOv5 在精度和速度之間取得了較好的平衡。
- YOLOX，YOLOv6，YOLOv7，YOLOv8在精度和速度上都持續的進步。
- YOLO-NAS針對硬體有額外的優化。

**其他功能**

- 除了基本的物件偵測外，YOLOv8 新增了圖像分割和圖像分類功能，使其應用範圍更廣。

**總結**

YOLO 系列模型不斷演進，在速度、精度和功能方面都有顯著提升。YOLOv8 的出現，更將 YOLO 的應用範圍擴展到圖像分割和圖像分類，使其成為一個更加強大的視覺任務解決方案。




Reference:
从 YOLOv1 到 YOLO-NAS 的所有 YOLO 模型：论文解析 - 波吉大哥的文章 - 知乎
https://zhuanlan.zhihu.com/p/675877274

YOLO模型详解 - 山河动人的文章 - 知乎
https://zhuanlan.zhihu.com/p/624060603

目标检测新手如何阅读YOLOX的源代码呢？ - 3D视觉开发者社区的回答 - 知乎
https://www.zhihu.com/question/482419414/answer/2427219945

YOLOX源码解析--十分详细，建议收藏！ - 来一块葱花饼的文章 - 知乎  
[https://zhuanlan.zhihu.com/p/411045300](https://www.google.com/url?q=https://zhuanlan.zhihu.com/p/411045300&sa=D&source=calendar&usd=2&usg=AOvVaw3jWIgIao-nfUSCL7ZFwavk)

爆肝总结！！YoloV3源码详解 - xc谈算法的文章 - 知乎  
[https://zhuanlan.zhihu.com/p/679672749](https://www.google.com/url?q=https://zhuanlan.zhihu.com/p/679672749&sa=D&source=calendar&usd=2&usg=AOvVaw0BClm9rT9K4ir33s1of2Ju)

YOLOe问世，实时观察一切，统一开放物体检测和分割 - 机器之心的文章 - 知乎  
[https://zhuanlan.zhihu.com/p/29912457801](https://www.google.com/url?q=https://zhuanlan.zhihu.com/p/29912457801&sa=D&source=calendar&usd=2&usg=AOvVaw1Viz8Up_XxLBdF7Wjg4Jwp)

YOLO-NAS对象检测算法再一次颠覆YOLO系列算法——已超越YOLOv8 - 人工智能研究所的文章 - 知乎
https://zhuanlan.zhihu.com/p/632626074

[CV - Object Detection]目标检测YOLO系列综述（全） - Pascal算法摆渡人的文章 - 知乎
https://zhuanlan.zhihu.com/p/558217700