
YOLOv1 速度最快，但精度較低。
YOLOv3 在小目標檢測方面有顯著提升。
YOLOv4 和 YOLOv5 在精度和速度之間取得了較好的平衡。
YOLOX，YOLOv6，YOLOv7，YOLOv8在精度和速度上都持續的進步。
YOLO-NAS針對硬體有額外的優化。
[[YOLOv8]] 新增了圖像分割和圖像分類功能

```python
import cv2
import torch
from ultralytics import YOLO

# 載入 YOLOv8 模型
model = YOLO("yolov8n.pt")  # 使用 YOLOv8n（nano 版本）

# 讀取影像
image_path = "image.jpg"
image = cv2.imread(image_path)

# 進行物件偵測
results = model(image)

# 繪製 Bounding Box
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # 取得 Bounding Box 座標
        conf = box.conf[0].item()  # 置信度
        class_id = int(box.cls[0].item())  # 類別 ID
        label = f"Class {class_id}: {conf:.2f}"
        
        # 畫框 & 加入標籤
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 顯示結果
cv2.imshow("YOLO Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

```


不同版本的 YOLO **輸入（input）** 大多都是一張圖像，但 **輸出（output）** 格式可能略有不同，特別是 bounding box 的回歸方式、類別表示方式、是否加入 mask（如 YOLOv8-Seg）等。

|                                         |     |
| --------------------------------------- | --- |
| [[###YOLO 模型架構概述]]                      |     |
| [[###YOLO跟RCNN的backbone, neck, head比較]] |     |
| [[###YOLO 系列模型版本演進]]                    |     |
| [[### YOLO compare to MobileNet]]       |     |
| [[### CSPDarknet and PAN]]              |     |
|                                         |     |
|                                         |     |
|                                         |     |
|                                         |     |

**YOLO 系列演進表**
YOLOv8

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

補充:
- **Backbone 是 DarkNet 而非 ResNet 或 VGG:** 這在 **YOLOv1、YOLOv2 和 YOLOv3** 中是正確的。Joseph Redmon 為這些早期的 YOLO 版本設計了 DarkNet-19 和 DarkNet-53 等骨幹網路。這些 DarkNet 骨幹網路的設計哲學更偏向於速度和效率。
    
    然而，值得注意的是，**後續的 YOLO 版本，例如 YOLOv4、YOLOR、YOLOX 和 YOLOv7 等，已經不再局限於使用 DarkNet 作為骨幹網路。** 它們可能會採用其他更先進的骨幹網路，例如：
    
    - **CSPDarknet:** YOLOv4 引入了 CSPDarknet，這是一種基於 Darknet 但融合了 Cross Stage Partial (CSP) 思想的骨幹網路，旨在進一步提高效率和精度。
    - **ResNet、EfficientNet 等:** 一些後續的研究和 YOLO 的變體也會嘗試使用 ResNet、EfficientNet 或其他高性能的骨幹網路來提升模型的精度。
    
    所以，將 DarkNet 視為 _所有_ YOLO 模型的唯一骨幹網路是不完全準確的，但對於早期的經典 YOLO 版本來說是正確的。
    
- **YOLO 的 neck model 可能用 FPN 或 PAN:** 這是 **正確的**。Neck 模型位於骨幹網路和 Head 模型之間，用於聚合來自不同尺度特徵圖的資訊，以提高對不同大小物體的檢測能力。FPN (Feature Pyramid Network) 和 PAN (Path Aggregation Network) 是常用的 Neck 模型，它們通過自上而下和自下而上的方式融合多尺度特徵。YOLOv3 開始引入 FPN，後續的 YOLO 版本 (例如 YOLOv4、YOLOR、YOLOX、YOLOv7) 經常使用 PAN 或其變體 (例如 SPP、CSP 等模塊的組合) 作為 Neck。
    
- **主要差別在 head model:** 這也是 **一個重要的觀點**。雖然骨幹網路負責提取基礎特徵，Neck 模型負責特徵融合，但 Head 模型直接負責最終的預測輸出，包括邊界框的位置、大小、置信度和類別。YOLO 的 Head 模型與傳統的兩階段目標檢測器 (例如 Faster R-CNN) 的 Head 模型有顯著的不同。
    
- **YOLO 的 head model 主要是由 convolutional layer 組成:** 這 **基本正確**。YOLO 的 Head 模型通常由一系列的卷積層構成，這些卷積層逐步處理 Neck 模型輸出的多尺度特徵圖。最終的幾個卷積層負責輸出預測結果。
    
- **最終輸出為 SxS(網格數)xB(預測邊界框數)x(5+c):** 這 **非常精確地描述了 YOLO Head 的輸出格式。**
    
    - **S x S:** 表示輸入圖像被劃分為 S x S 個網格單元 (cell)。每個網格單元負責檢測其中心落在該單元內的物體。
    - **B:** 表示每個網格單元預測 B 個邊界框 (bounding box)。
    - **5:** 表示每個預測邊界框包含 5 個元素：
        - tx​,ty​: 相對於網格單元左上角的邊界框中心偏移量 (通常經過 Sigmoid 函數歸一化到 0-1 之間)。
        - tw​,th​: 相對於預定義的 anchor box 的寬度和高度的對數空間偏移量。
        - Confidence score (P(object)∗IOU(prediction,ground_truth)): 表示該邊界框包含一個物體的置信度以及預測框與真實框的交並比 (IOU)。
    - **c:** 表示物件類別的數量。每個預測邊界框還會輸出一個長度為 c 的條件類別概率向量 (P(Classi​∣object))。

**需要補充的是，YOLO 的 Head 模型在不同的版本中也可能有一些細微的差異，例如：**

- **輸出層的激活函數:** 邊界框的中心偏移量和置信度通常使用 Sigmoid 函數進行歸一化。邊界框的寬度和高度偏移量通常直接輸出。類別概率通常使用 Softmax 函數進行歸一化。
- **Anchor Box 的使用:** YOLOv2 及以後的版本引入了 anchor box 的概念，預定義了一組不同形狀和大小的先驗框，模型預測的是相對於這些 anchor box 的偏移量。
- **多尺度預測:** 後續的 YOLO 版本 (例如 YOLOv3 及以後) 通常會在 Neck 模型輸出的不同尺度特徵圖上進行獨立的 Head 預測，從而實現多尺度檢測。這意味著 Head 的輸出會有多個分支，每個分支對應一個尺度。

**總結:**

你的描述抓住了 YOLO 與一般 CNN based object detection model 的關鍵區別，特別是在 Head 模型的设计理念和输出格式上。YOLO 的 Head 模型直接輸出密集的預測，而不需要像兩階段檢測器那樣先生成候選區域 (region proposals)。

需要注意的是，YOLO 的骨幹網路不再局限於 DarkNet，後續版本可能會採用其他更先進的骨幹網路以追求更高的性能。但你對 Neck 模型使用 FPN/PAN 和 Head 模型由卷積層組成並輸出 S×S×B×(5+c) 的描述是準確且重要的。



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

建议收藏！超实用的 YOLO 训练和测试技巧合集 - OpenMMLab的文章 - 知乎
https://zhuanlan.zhihu.com/p/617677829




### YOLO compare to MobileNet

詳細解釋 MobileNet 和 YOLO 的效能比較，釐清它們在速度、延遲和效率方面的特點，並介紹一些更新的、高效的物件偵測模型。

---

**一、 核心概念釐清：骨幹網路 vs. 物件偵測器**

在比較 MobileNet 和 YOLO 之前，最重要的一點是理解它們在 AI 模型架構中扮演的不同角色：

1. **MobileNet：高效的骨幹網路 (Efficient Backbone Network)**
    
    - **目的：** MobileNet 主要是一個**用於影像分類的骨幹網路**，其核心設計目標是在保持可接受精度的前提下，**極大地減少計算量 (FLOPs) 和模型參數數量**，從而適用於計算資源有限的行動裝置和嵌入式硬體。
    - **角色：** 作為**特徵提取器 (Feature Extractor)**。它可以單獨用於影像分類任務，或者**作為更複雜任務（如物件偵測、影像分割）的基礎部分（即骨幹）**。
    - **關鍵技術：** **深度可分離卷積 (Depthwise Separable Convolution)**，將標準卷積分解為深度卷積 (Depthwise Convolution) 和逐點卷積 (Pointwise Convolution)，大幅降低了計算成本和參數數量。
    - **衡量指標：** 主要關注**效率**，即在有限的計算資源（低 FLOPs、少參數）下達到較好的分類精度，並因此帶來**低延遲 (Low Latency)** 和**小模型尺寸 (Small Model Size)**，以及**低功耗 (Low Power Consumption)** 的優點。
2. **YOLO (You Only Look Once)：快速的端到端物件偵測器 (Fast End-to-End Object Detector)**
    
    - **目的：** YOLO 是一個完整的**物件偵測模型架構**，其設計目標是實現**高速、即時的物件偵測**。
    - **角色：** **端到端**完成物件偵測任務，直接從輸入影像輸出物件的邊界框 (Bounding Box) 和類別。
    - **關鍵技術：** 將物件偵測視為一個回歸問題，在一個網格系統上直接預測邊界框座標和類別機率。它**只需對影像進行一次前向傳播 (Single Pass)** 即可完成所有預測，這是其速度快的核心原因。不同版本的 YOLO 採用了不同的骨幹網路（如 Darknet, CSPDarknet，部分小型版本也可能借鑒 MobileNet 思想）和改進的檢測頭 (Detection Head)、Neck 結構（如 PANet, BiFPN）。
    - **衡量指標：** 主要關注**偵測速度 (Speed)**，通常用**每秒幀數 (Frames Per Second, FPS)** 來衡量，同時也追求良好的**偵測精度 (Accuracy)**，常用 mAP (mean Average Precision) 評估。

**因此，直接比較 MobileNet 和 YOLO 的「效能」是不恰當的，因為一個是骨幹，一個是完整的偵測器。** 更合理的比較是：

- 比較 MobileNet 作為骨幹網路的**效率** vs. 其他骨幹網路（如 ResNet）的效率。
- 比較**使用 MobileNet 作為骨幹的物件偵測器**（例如 SSD-MobileNet 或某些輕量級 YOLO 變體） vs. **使用其他骨幹的 YOLO 模型**（例如 YOLOv5s 使用 CSPDarknet）的偵測速度和精度。

---

**二、 為何 YOLO 特點是速度快，而 MobileNet 是高效率？**

1. **YOLO 的速度快 (High Speed - FPS)：**
    
    - **架構設計：** 其端到端的單階段 (Single-Stage) 設計避免了傳統兩階段偵測器（如 Faster R-CNN）需要先生成候選區域 (Region Proposals) 再進行分類和回歸的複雜流程。YOLO 直接在全圖上進行預測，大幅減少了計算步驟。
    - **全圖一次處理：** 模型一次前向傳播就能得到所有物件的預測結果。
    - **持續優化：** YOLO 系列不斷演進，後續版本（如 YOLOv4, YOLOv5, YOLOv7, YOLOv8 等）在維持或提升速度的同時，也透過改進骨幹網路、Neck 結構（特徵融合）、訓練策略等方式大幅提升了精度。
2. **MobileNet 的高效率 (High Efficiency - Low Latency, Small Size, Low Power)：**
    
    - **核心技術：** 深度可分離卷積相比標準卷積，可以在相似精度的情況下將計算量和參數數量降低數倍（例如 MobileNet v1 中約 8-9 倍）。
    - **資源消耗低：** 計算量少意味著推論時所需的處理器運算時間短（低延遲），參數少意味著模型檔案小（易於部署），同時運算量和記憶體訪問減少也帶來了低功耗。
    - **應用場景：** 這種效率使其非常適合資源受限的邊緣設備。當 MobileNet 被用作物件偵測器的骨幹時，它能顯著降低整個偵測模型的延遲和資源需求。

---

**三、 解惑：YOLO 比 MobileNet 快，MobileNet 比 YOLO 低延遲？**

這句話的表述容易引起混淆，需要更精確地理解：

- **「YOLO 比 MobileNet 快」：** 如果指的是**完整的物件偵測任務**，那麼一個典型的 YOLO 模型（如 YOLOv5s）通常比一個使用 MobileNet 作為骨幹的偵測器（如 SSD-MobileNet v2）在**偵測速度 (FPS)** 上更快，尤其是在相似精度水平下或者在能夠充分利用 YOLO 架構優勢的硬體上。YOLO 的架構就是為偵測速度而優化的。
- **「MobileNet 比 YOLO 低延遲」：** 如果指的是**單次推論的延遲時間 (ms)**，那麼 MobileNet 作為一個**骨幹網路**，其本身的計算延遲非常低。當它被整合到一個偵測器中時，可以使該偵測器的**基礎延遲**很低。一個輕量級的 YOLO 模型（如 YOLOv5n 或 YOLOX-Nano，它們內部也可能採用了類似 MobileNet 的高效設計）也可以實現非常低的延遲。
    - **更準確的說法：** MobileNet 骨幹網路本身具有**極高的計算效率**，使得**基於 MobileNet 的模型（包括偵測器）** 有潛力在邊緣設備上實現**非常低的推論延遲**。而 YOLO 架構則以其**端到端的高偵測速率 (FPS)** 著稱。
    - **結論：** 這兩者並不矛盾。YOLO 的優勢在於其整體偵測流程的速度 (FPS)。MobileNet 的優勢在於其組件（骨幹）的效率，這有助於降低使用它的模型的單次推論延遲和資源消耗。在實際應用中，一個輕量化的 YOLO 模型也可以同時達到高 FPS 和低延遲。

---

**四、 比 MobileNet 更快、更高效的新物件偵測模型**

這裡的「比 MobileNet 更快更高效」應理解為：比**基於 MobileNet 的（早期）物件偵測器（如 SSD-MobileNet v1/v2）** 在速度、效率（通常指速度和精度的平衡）上表現更優的**新一代輕量級物件偵測模型**。許多新模型本身可能也借鑒或採用了 MobileNet 系列的設計思想（如深度可分離卷積、倒置殘差結構等），或是採用了更先進的技術。

以下列舉一些近年來在速度和效率方面表現突出的物件偵測模型系列（尤其關注其適用於邊緣的版本）：

1. **YOLO 系列 (近期版本)：**
    
    - **YOLOv5 (n/s/m)：** 由 Ultralytics 開發，提供了從 Nano (n) 到 Small (s), Medium (m) 等多個版本，在速度和精度之間有很好的平衡，非常流行，易於部署。n/s 版本非常適合邊緣計算。
    - **YOLOX (Nano, Tiny, S)：** 提出了解耦頭 (Decoupled Head)、Anchor-Free 設計、SimOTA 標籤分配策略等改進，其 Nano 和 Tiny 版本同樣具有極高的效率。
    - **YOLOv7 (tiny)：** 在模型結構和訓練策略上進行了大量優化，其 Tiny 版本在保持高速的同時精度也有不錯表現。
    - **YOLOv8 (n/s/m)：** YOLOv5 的下一代模型，同樣由 Ultralytics 推出，架構上做了更新，提供了與 YOLOv5 類似的可擴展版本，n/s 版本效率很高。
    - **YOLO-NAS:** 由 Deci AI 提出，利用其專有的神經架構搜索技術 (NAS) 自動生成，旨在特定硬體上達到最佳的延遲-精度平衡，提供了不同大小的版本。
    - **RTMDet (Real-Time Models for Object Detection):** 由 MMLab 開發，也是專注於即時偵測的高效模型系列，有 Tiny/S/M/L 等版本。
2. **EfficientDet 系列 (D0-D2)：**
    
    - 由 Google Brain 團隊提出，採用 EfficientNet 作為骨幹網路，並引入了 BiFPN (Bi-directional Feature Pyramid Network) 進行高效的多尺度特徵融合。
    - 其設計理念就是追求極高的**效率**（在同等計算資源下達到更高精度）。較小的版本如 EfficientDet-D0, D1, D2 適合邊緣部署，相比早期的 SSD-MobileNet 在精度上有顯著提升，同時速度也相當不錯。
3. **NanoDet 系列：**
    
    - 專門為行動端/嵌入式設備設計的超輕量級物件偵測模型。模型尺寸極小（通常 1MB 左右），推論速度非常快，適合資源極度受限的場景。

**總結比較點：**

|特性|MobileNet (作為骨幹)|YOLO (作為偵測器架構)|SSD-MobileNet (範例)|新型輕量 YOLO/EfficientDet (範例)|
|:--|:--|:--|:--|:--|
|**主要角色**|特徵提取器 (骨幹)|端到端物件偵測器|完整物件偵測器|完整物件偵測器|
|**設計重點**|**效率** (低 FLOPs/參數/功耗)|**速度** (高 FPS)|基於 MobileNet 的效率，SSD 的偵測框架|速度與精度的平衡，高效率|
|**核心技術**|深度可分離卷積|單階段預測, 網格, 特殊 Neck/Head 設計|SSD 框架 + MobileNet 骨幹|先進骨幹/Neck/Head, 優化訓練策略|
|**典型優勢**|低延遲潛力, 小尺寸, 低功耗|高 FPS|低延遲, 小尺寸|高 FPS, 低延遲, 較高精度/效率|
|**典型劣勢**|單獨不能偵測|早期版本對小物件偵測稍弱|精度可能不如新模型|-|

---

**五、 結論**

MobileNet 是一個專注於**計算效率**的骨幹網路，使得基於它的模型能夠實現低延遲、小尺寸和低功耗，非常適合資源受限的邊緣設備。YOLO 則是一個專注於**偵測速度 (FPS)** 的端到端物件偵測架構。比較兩者時，務必釐清是在比較骨幹的效率還是完整偵測器的速度與精度。

近年來，物件偵測領域發展迅速，出現了許多如 YOLOv5/v7/v8/X/NAS、EfficientDet、NanoDet 等更快、更高效的輕量級模型。這些新模型通常結合了高效的骨幹網路設計（可能吸取了 MobileNet 的思想）、優化的特徵融合結構（如 FPN, PANet, BiFPN）以及先進的訓練策略，旨在邊緣硬體上實現更優的速度、精度和效率平衡，超越了像 SSD-MobileNet 這樣的早期高效偵測器。選擇哪種模型取決於具體的應用需求、硬體平台以及對速度、精度、延遲、功耗等指標的權衡。



### CSPDarknet and PAN
## CSPDarknet (Cross Stage Partial Darknet)

**詳細解釋:**

CSPDarknet 是一種基於 Darknet 結構進行改進的骨幹網路，由 YOLOv4 論文提出。它的核心思想是**跨階段局部連接 (Cross Stage Partial Connections, CSP)**。CSP 的目標是**在不顯著增加計算量的同時，增強網路的學習能力，並減少冗餘計算。**

**核心思想 - CSP:**

CSP 的主要思想是將輸入特徵圖分成兩個部分：

1. **主路徑 (Main Path):** 這一部分會通過一系列的卷積層和其他處理單元進行正常的處理。
2. **旁路 (Cross Stage Partial Path):** 這一部分直接將原始輸入的一部分或經過少量處理的版本連接到主路徑的後續階段。

![[Pasted image 20250501152954.png]]

然後，主路徑的輸出和旁路的輸出會在最後進行融合 (通常是通道維度的拼接)。

**在 CSPDarknet 中的應用:**

CSPDarknet 將 CSP 的思想應用於 Darknet 的殘差塊 (residual block)。一個典型的 CSPDarknet 包含多個 CSP 階段 (CSP Stage)。每個 CSP 階段通常包含以下結構：

1. **Split:** 將輸入特徵圖沿通道維度分成兩個部分。
2. **CSP Block (或類似結構):** 其中一個部分通過多個殘差塊或其他卷積層進行處理。
3. **Pass-through/Shortcut:** 另一個部分直接作為旁路。
4. **Merge:** 將處理後的主路徑和旁路在通道維度上拼接在一起。
5. **Transition Layer:** 拼接後的特徵圖通常會通過一個卷積層進行通道數的調整和混合。

**優點:**

- **增強學習能力:** CSP 連接允許梯度流在網路中更有效地傳播，使得更深層的網路更容易訓練，並能學習到更豐富的特徵。
- **減少計算冗餘:** 通過將一部分原始輸入直接連接到後面的階段，CSP 可以減少重複的梯度資訊，從而降低計算冗餘。
- **提高推理速度和降低參數量:** 相較於傳統的深層網路，CSPDarknet 在保持相似甚至更高精度的情況下，通常具有更少的參數和更快的推理速度。

**總結:**

CSPDarknet 通過引入跨階段局部連接的思想，有效地優化了 Darknet 的結構，使其在目標檢測等任務中能夠實現更高的精度和效率。它是 YOLOv4 的關鍵組件之一。

## PAN (Path Aggregation Network)

**詳細解釋:**

PAN (Path Aggregation Network) 是一種用於目標檢測的**頸部 (neck)** 網路結構，由 PANet 論文提出。它的目標是**增強特徵金字塔 (Feature Pyramid Network, FPN) 的特徵融合能力，特別是對於不同尺度物體的檢測。**

**背景 - FPN 的不足:**

FPN 通過自上而下的方式融合不同尺度的特徵圖，使得高層語義資訊能夠傳遞到低層，增強了對大物體的檢測能力。然而，FPN 的信息流是單向的 (從高層到低層)，低層的精確位置資訊難以有效地傳遞到高層。

**PAN 的核心思想 - 雙向路徑聚合:**

PAN 在 FPN 的基礎上增加了一<mark style="background: #BBFABBA6;">個**自下而上的路徑聚合網路 (bottom-up path augmentation network)**</mark>。這個自下而上的路徑與 FPN 的自上而下路徑形成了一個雙向的特徵金字塔結構。

PAN 的主要組成部分包括：

1. **FPN (Feature Pyramid Network):** 生成多尺度特徵圖。
2. **Bottom-up Path Augmentation Network:** 從低層到高層聚合特徵。在這個過程中，每個階段的特徵圖會與前一個階段經過下採樣的特徵圖進行融合 (通常是相加或拼接)。
3. **Lateral Connections:** FPN 中用於連接相同語義層級的不同尺度特徵圖的橫向連接仍然存在。
4. **Adaptive Feature Pooling:** PAN 還引入了自適應的特徵池化機制，以更好地利用每個特徵層級的信息。

**優點:**

- **增強多尺度特徵融合:** PAN 的雙向路徑聚合能夠更有效地將低層的位置資訊和高層的語義資訊融合在一起，從而提高對不同大小物體的檢測性能，特別是對於小物體的檢測。
- **提高精度:** 通過更充分地利用多尺度特徵，PAN 通常可以提升目標檢測器的整體精度。

**總結:**

PAN 是一種有效的頸部網路結構，它通過引入自下而上的路徑聚合，增強了 FPN 的多尺度特徵融合能力，從而在目標檢測任務中取得了顯著的性能提升。許多現代目標檢測器 (包括一些 YOLO 的變體) 都採用了 PAN 或其改進版本作為頸部網路。

## YOLOv8 的骨幹網路

關於 YOLOv8 的骨幹網路，**官方並沒有明確聲明它固定使用 CSPDarknet。** YOLOv8 的設計更加模塊化和靈活，它允許研究人員和開發者根據具體需求選擇不同的骨幹網路。

然而，根據 Ultralytics (YOLOv8 的開發者) 提供的資訊和模型結構分析，**YOLOv8 的骨幹網路受到了 YOLOv5 和其他現代目標檢測器的影響，並進行了改進。**

**YOLOv8 的骨幹網路的關鍵特點和可能的組成部分包括：**

- **C2f (Conv-Contextualized Local Features) 模塊:** 這是一個在 YOLOv8 中被廣泛使用的核心構建塊。C2f 模塊的設計靈感可能來自於 YOLOv5 的 C3 模塊，但進行了進一步的優化。它旨在**在保持高效性的同時，提取更豐富的上下文信息。** C2f 模塊通常包含多個卷積層和殘差連接，並以一種更有效的方式組合不同層的特徵。
    
- **CSP (Cross Stage Partial) 思想的應用:** 雖然不一定完全是傳統的 CSPDarknet 的結構，但 YOLOv8 的骨幹網路很可能也借鑒了 CSP 的思想，以提高梯度流和特徵傳播效率。
    
- **可能包含其他現代骨幹網路的組件:** 根據不同的 YOLOv8 模型變體和配置，它可能也會嘗試使用或集成其他先進骨幹網路 (例如 EfficientNetV2、RepVGG 等) 的一些優點。
    

**因此，更準確的說法是：**

- **YOLOv8 的骨幹網路的核心構建塊是 C2f 模塊。**
- **YOLOv8 的骨幹網路設計受到了 CSPDarknet 和其他現代骨幹網路的啟發，並可能在其結構中融入了 CSP 的思想。**
- **YOLOv8 並不強制使用傳統的 CSPDarknet 作為其唯一的骨幹網路。**

**總結:**

YOLOv8 的骨幹網路是一個進化後的結構，其核心是高效且能提取豐富上下文信息的 C2f 模塊。它在設計上受到了 CSPDarknet 等先前工作的影響，但並非完全等同於 CSPDarknet。YOLOv8 的靈活性允許在未來探索和集成更多先進的骨幹網路。