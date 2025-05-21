
Backbone: CSPDarkNet
Neck:  PAN
head: Decoupled Head

Parms = 3-68 (M)
FLOPs = 8 - 257(B)
Latency = 1 - 4 (ms)
Throughput: 1000-280 images/s
mAP = 37 - 53 


|                                      |     |
| ------------------------------------ | --- |
| [[###YOLOv8的主要功能]]                   |     |
| [[###YOLOv8 的不同變體]]                  |     |
| [[###YOLOv8网络的backbone, neck, head]] |     |
| [[###C2f模組]]                         |     |
| [[###YOLOv8 的 PAN 架構詳解]]             |     |
| [[### DarkNet vs CSPDarkNet]]        |     |

https://docs.ultralytics.com/zh/models/yolov8/

YOLOv8 由Ultralytics 于 2023 年 1 月 10 日发布，在准确性和速度方面具有尖端性能。在以往YOLO 版本的基础上，YOLOv8 引入了新的功能和优化，使其成为广泛应用中各种[物体检测](https://www.ultralytics.com/blog/a-guide-to-deep-dive-into-object-detection-in-2025)任务的理想选择。

![Ultralytics YOLOv8](https://github.com/ultralytics/docs/releases/download/0/yolov8-comparison-plots.avif)

### YOLOv8的主要功能

- **先进的骨干和颈部架构：** YOLOv8 采用了最先进的骨干和颈部架构，从而提高了[特征提取](https://www.ultralytics.com/glossary/feature-extraction)和[目标检测](https://www.ultralytics.com/glossary/object-detection)性能。
- **无锚分裂Ultralytics 头：** YOLOv8 采用无锚(Anchor free)分裂Ultralytics 头，与基于锚的方法相比，它有助于提高检测过程的准确性和效率。
- **优化精度与**速度之间**的权衡：** YOLOv8 专注于保持精度与速度之间的最佳平衡，适用于各种应用领域的实时目标检测任务。
- **各种预训练模型：** YOLOv8 提供一系列预训练模型，以满足各种任务和性能要求，从而更容易为您的特定用例找到合适的模型。

## 支持的任务和模式

YOLOv8 系列提供多种模型，每种模型都专门用于计算机视觉中的特定任务。这些模型旨在满足从物体检测到[实例分割](https://www.ultralytics.com/glossary/instance-segmentation)、姿态/关键点检测、定向物体检测和分类等更复杂任务的各种要求。

YOLOv8 系列的每个变体都针对各自的任务进行了优化，以确保高性能和高精确度。此外，这些模型还兼容各种操作模式，包括[推理](https://docs.ultralytics.com/zh/modes/predict/)、[验证](https://docs.ultralytics.com/zh/modes/val/)、[训练](https://docs.ultralytics.com/zh/modes/train/)和[输出](https://docs.ultralytics.com/zh/modes/export/)，便于在部署和开发的不同阶段使用。

| 模型         | 文件名                                                                                                            | 任务                                                     | 推论  | 验证  | 培训  | 出口  |
| ---------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ | --- | --- | --- | --- |
| YOLOv8     | `yolov8n.pt` `yolov8s.pt` `yolov8m.pt` `yolov8l.pt` `yolov8x.pt`                                               | [检测](https://docs.ultralytics.com/zh/tasks/detect/)    | ✅   | ✅   | ✅   | ✅   |
| YOLOv8-seg | `yolov8n-seg.pt` `yolov8s-seg.pt` `yolov8m-seg.pt` `yolov8l-seg.pt` `yolov8x-seg.pt`                           | [实例分割](https://docs.ultralytics.com/zh/tasks/segment/) | ✅   | ✅   | ✅   | ✅   |
| YOLOv8-姿势  | `yolov8n-pose.pt` `yolov8s-pose.pt` `yolov8m-pose.pt` `yolov8l-pose.pt` `yolov8x-pose.pt` `yolov8x-pose-p6.pt` | [姿势/关键点](https://docs.ultralytics.com/zh/tasks/pose/)  | ✅   | ✅   | ✅   | ✅   |
| YOLOv8-obb | `yolov8n-obb.pt` `yolov8s-obb.pt` `yolov8m-obb.pt` `yolov8l-obb.pt` `yolov8x-obb.pt`                           | [定向检测](https://docs.ultralytics.com/zh/tasks/obb/)     | ✅   | ✅   | ✅   | ✅   |
| YOLOv8-cls | `yolov8n-cls.pt` `yolov8s-cls.pt` `yolov8m-cls.pt` `yolov8l-cls.pt` `yolov8x-cls.pt`                           | [分类](https://docs.ultralytics.com/zh/tasks/classify/)  | ✅   | ✅   | ✅   | ✅   |

本表概述了YOLOv8 模型的各种变体，重点介绍了它们在特定任务中的适用性以及与推理、验证、训练和输出等各种操作模式的兼容性。它展示了YOLOv8 系列的多功能性和坚固性，使其适用于[计算机视觉](https://www.ultralytics.com/glossary/computer-vision-cv)领域的各种应用。

## 性能指标
![[Pasted image 20250324132204.png]]



|模型|尺寸  <br>（像素）|mAPval  <br>50-95|速度  <br>CPU ONNX  <br>（毫秒）|速度  <br>A100 TensorRT  <br>（毫秒）|params  <br>(M)|FLOPs  <br>(B)|
|---|---|---|---|---|---|---|
|[YOLOv8n](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt)|640|37.3|80.4|0.99|3.2|8.7|
|[YOLOv8s](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt)|640|44.9|128.4|1.20|11.2|28.6|
|[YOLOv8m](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt)|640|50.2|234.7|1.83|25.9|78.9|
|[YOLOv8l](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt)|640|52.9|375.2|2.39|43.7|165.2|
|[YOLOv8x](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt)|640|53.9|479.1|3.53|68.2|257.8|

```python
from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Run inference with the YOLOv8n model on the 'bus.jpg' image
results = model("path/to/bus.jpg")
```


### YOLOv8 的不同變體

我會詳細解釋 YOLOv8 的不同變體（YOLOv8-seg, YOLOv8-pose, YOLOv8-obb, YOLOv8-cls 與基礎 YOLOv8 的差別），並提供具體例子來說明其應用場景。接著，我會解釋 yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt 的意義，並列出它們的參數量（params）、浮點運算量（FLOPs）、檔案大小（file size）、延遲（latency）和吞吐量（throughput）。

---

### **一、YOLOv8-seg, YOLOv8-pose, YOLOv8-obb, YOLOv8-cls 與 YOLOv8 的差別**

YOLOv8 是由 Ultralytics 開發的最新一代 YOLO 模型，基礎版本（通常簡稱為 YOLOv8 或 YOLOv8-detect）專注於**物體檢測（object detection）**，也就是在圖像中識別物體並用邊界框（bounding box）標示其位置與類別。然而，YOLOv8 提供了多種變體，針對不同任務進行優化。以下是各變體的具體差別與應用舉例：

#### **1. YOLOv8-seg（分割模型）**

- **功能**：除了物體檢測外，還能進行**實例分割（instance segmentation）**，即不僅標出物體的邊界框，還能精確勾勒出每個物體的像素級輪廓。
- **差別**：相較於基礎 YOLOv8（僅提供邊界框），YOLOv8-seg 會輸出每個檢測到的物體的遮罩（mask），適用於需要精確形狀的場景。
- **應用舉例**：
    - **場景**：醫療影像分析。
    - **例子**：在一張 X 光片中，YOLOv8-seg 可以檢測出肺部區域，並生成一個遮罩來標示肺部的精確範圍，而不僅僅是框出肺部位置。
- **模型文件**：例如 yolov8n-seg.pt。

#### **2. YOLOv8-pose（姿態估計模型）**

- **功能**：專注於**姿態估計（pose estimation）**，能在檢測物體的同時識別其關鍵點（keypoints），例如人體的關節點。
- **差別**：基礎 YOLOv8 只檢測物體位置，而 YOLOv8-pose 會額外輸出關鍵點座標，通常用於分析動作或結構。
- **應用舉例**：
    - **場景**：運動分析。
    - **例子**：在一張跑步者的照片中，YOLOv8-pose 可以檢測出人體，並標記出手肘、膝蓋、肩膀等關鍵點，用於分析跑步姿勢。
- **模型文件**：例如 yolov8n-pose.pt。

#### **3. YOLOv8-obb（定向邊界框模型）**

- **功能**：支援**定向邊界框（oriented bounding box, OBB）**檢測，能處理旋轉物體，提供帶角度的邊界框。
- **差別**：基礎 YOLOv8 使用水平邊界框，而 YOLOv8-obb 使用旋轉邊界框，適合非水平排列的物體。
- **應用舉例**：
    - **場景**：衛星影像分析。
    - **例子**：在一張衛星圖中，YOLOv8-obb 可以檢測停泊的船隻，並用旋轉邊界框準確框住船身，而不是水平框可能包含多餘背景。
- **模型文件**：例如 yolov8n-obb.pt。
- 格式:  [ class_index, x1, y1, x2, y2, x3, y3, x4, y4 ]   分類索引+四個頂點座標

Reference:
- YOLO里的OBB：带旋转角度的目标检测方案 - IT男的文章 - 知乎
https://zhuanlan.zhihu.com/p/713483915

#### **4. YOLOv8-cls（分類模型）**

- **功能**：專注於**圖像分類（image classification）**，對整張圖片進行類別預測，不提供位置或分割資訊。
- **差別**：基礎 YOLOv8 是檢測模型，而 YOLOv8-cls 不檢測物體位置，只判斷圖片的整體類別。
- **應用舉例**：
    - **場景**：圖片內容分類。
    - **例子**：給定一張圖片，YOLOv8-cls 可以判斷這是「貓」還是「狗」，而不標示貓或狗在圖中的位置。
- **模型文件**：例如 yolov8n-cls.pt。

#### **5. YOLOv8（基礎檢測模型）**

- **功能**：標準的**物體檢測（object detection）**，輸出物體的邊界框和類別，適用於大多數檢測任務。
- **應用舉例**：
    - **場景**：監控系統。
    - **例子**：在一張街道圖像中，YOLOv8 可以檢測出汽車、行人，並用邊界框標示它們的位置。
- **模型文件**：例如 yolov8n.pt。

---

### **二、yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt 的意義**

這些文件名中的字母（n, s, m, l, x）代表模型的大小和複雜度，從小到大依次為：

- **n（nano）**：最小的模型，適合資源有限的設備（如移動設備）。
- **s（small）**：小型模型，平衡速度與精度。
- **m（medium）**：中型模型，適合一般應用。
- **l（large）**：大型模型，提供更高精度。
- **x（extra large）**：最大的模型，追求極致精度，適合高效能硬體。

這些變體適用於基礎 YOLOv8（檢測）以及其他任務模型（seg, pose, obb, cls），只是根據任務不同，文件名會加上後綴（如 yolov8n-seg.pt）。

---

### **三、參數比較（以 YOLOv8 檢測模型為例）**

以下數據來自 Ultralytics 官方文檔，基於 COCO 數據集的預訓練模型，圖像尺寸為 640x640。以下列出 **yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt** 的參數（params）、浮點運算量（FLOPs）、檔案大小（file size）、延遲（latency）和吞吐量（throughput）。

|**模型**|**Params (M)**|**FLOPs (B)**|**File Size (MB)**|**Latency (ms, A100 TensorRT)**|**Throughput (images/s, A100)**|
|---|---|---|---|---|---|
|yolov8n.pt|3.2|8.7|6.2|0.99|~1010|
|yolov8s.pt|11.2|28.6|21.5|1.20|~833|
|yolov8m.pt|25.9|78.9|49.7|1.83|~546|
|yolov8l.pt|43.7|165.2|83.7|2.39|~418|
|yolov8x.pt|68.2|257.8|130.5|3.53|~283|

#### **解釋**：

1. **Params (M)**：參數量，單位為百萬（million）。參數越多，模型越複雜，精度通常越高，但計算成本也增加。
2. **FLOPs (B)**：浮點運算量，單位為十億（billion）。反映模型的計算複雜度，FLOPs 越高，計算需求越大。
3. **File Size (MB)**：模型檔案大小，與參數量直接相關，影響存儲需求。
4. **Latency (ms)**：在 NVIDIA A100 GPU 上使用 TensorRT 進行推理的延遲，單位為毫秒。延遲越低，推理速度越快。
5. **Throughput (images/s)**：每秒處理的圖像數量，與延遲成反比，根據公式 throughput = 1000 / latency 估算。

#### **具體分析**：

- **yolov8n.pt**：超輕量級，適合嵌入式設備，速度最快（0.99ms），但精度最低。
- **yolov8s.pt**：在速度與精度間取得平衡，適合大多數應用。
- **yolov8m.pt**：中型模型，精度更高，適用於需要更好檢測效果的場景。
- **yolov8l.pt**：大型模型，適合高效能硬體，精度進一步提升。
- **yolov8x.pt**：最大模型，精度最佳，但延遲最高，適合伺服器級應用。

---

### **四、總結與應用建議**

- **選擇依據**：
    - 如果需要快速推理（如實時監控），選 yolov8n.pt 或 yolov8s.pt。
    - 如果需要高精度（如醫療影像），選 yolov8l.pt 或 yolov8x.pt。
    - 根據任務選擇變體：分割用 -seg，姿態用 -pose，定向檢測用 -obb，分類用 -cls。
- **硬體考量**：小型模型適合邊緣設備，大型模型需要 GPU 支援。

### YOLOv8网络的backbone, neck, head

![[Pasted image 20250325165007.png]]
YOLOv8网络主要包含3个部分【Backbone，Neck, Head】,主要作用如下：

- Backbone主干网络是模型的基础，负责从输入图像中提取特征。这些特征是后续网络层进行目标检测的基础。<mark style="background: #FFB86CA6;"> 在YOLOv8中，主干网络采用了类似于CSPDarknet的结构</mark>。
- Neck颈部网络位于主干网络和头部网络之间，它的作用是进行特征融合和增强。
- Head头部网络是目标检测模型的决策部分，负责产生最终的检测结果。

`yolov8.yaml`配置文件的作用就是根据文件内容，构建出上述整个网络。下面对`yolov8.yaml`配置文件进行详细讲解。

![[Pasted image 20250325191524.png]]

### 關鍵要點

- YOLOv8 的網路結構設計使其在物件檢測中具有低延遲和高 mAP，研究顯示這主要得益於其高效的架構。
- 它使用<mark style="background: #BBFABBA6;">無錨點檢測頭，簡化預測過程，加速非最大抑制（NMS）</mark>，似乎提升了速度和準確性。
- 骨幹網路採用 <mark style="background: #BBFABBA6;">C2f 模組</mark>和 3x3 卷積，減少計算量，證據顯示這有助於保持特徵提取能力。
- 頸部設計優化特徵融合，研究表明這提高了檢測性能。

---

### 網路結構設計的特點

YOLOv8 的網路結構設計使其在物件檢測中表現出低延遲和高 mAP。以下是其主要特點：

#### 無錨點檢測頭

YOLOv8 採用無錨點(Anchor free)檢測頭，這與之前的版本不同，之前的版本使用預定義的錨點框。無錨點方法直接預測物件中心，減少了框的預測數量，並簡化了非最大抑制（NMS）過程。研究顯示，這種設計同時提升了速度和準確性，使其適合即時應用。

#### 高效的骨幹網路

YOLOv8 的骨幹網路使用 C2f 模組替代之前的 C3 模組，並將初始的 6x6 卷積改為 3x3 卷積。這些改變減少了計算複雜度，證據顯示這在保持特徵提取能力方面效果良好，特別是在處理大規模數據時。

#### 優化的頸部設計

頸部負責特徵融合，YOLOv8 的設計允許從不同尺度拼接特徵而不強制特定通道維度。研究表明，這種方法優化了參數使用，改善了特徵整合，從而提升了檢測性能。

#### 意外的細節：訓練技巧的影響

除了結構設計，YOLOv8 還在訓練過程中關閉最後 10 個 epoch 的馬賽克增強，這被證實有助於提升準確性。這一細節可能出乎意料，但對最終性能有顯著影響。

![[Pasted image 20250325203329.png]]
### 調查報告：YOLOv8 網路結構設計的詳細分析

YOLOv8 作為 YOLO 系列的最新迭代，以其低延遲和高 mAP 在物件檢測領域表現出色。以下是對其網路結構設計特點的深入分析，解釋其為何能如此快速，並與其他物件檢測模型進行比較。

#### 背景與概述

YOLO（You Only Look Once）是一種即時物件檢測算法，自 2015 年首次提出以來，經歷了多次迭代。YOLOv8 由 Ultralytics 團隊開發，於 2023 年 1 月 10 日發布，旨在提供更高的準確性和速度。與兩階段檢測器（如 Faster R-CNN）相比，YOLOv8 屬於單階段檢測器，速度更快；與其他單階段模型（如 SSD）相比，其 mAP 表現更優。根據 [YOLOv8 - Ultralytics YOLO Docs](https://docs.ultralytics.com/models/yolov8/)，YOLOv8 在 COCO 數據集上達到 50.2% mAP（640x640 解析度），並在現代 GPU 上可實現超過 100 幀/秒的處理速度。

#### 網路結構設計的特點

YOLOv8 的網路結構可分為三個主要部分：骨幹（Backbone）、頸部（Neck）和頭部（Head）。以下是其設計特點的詳細分析：

##### 1. 無錨點檢測頭：簡化與加速

- YOLOv8 採用無錨點檢測頭，這是與 YOLOv5 等前代模型的重要區別。前代模型使用預定義的錨點框，每個網格單元預測多個邊界框，然後通過 NMS 處理重疊框。根據 [What's New in YOLOv8?](https://blog.roboflow.com/whats-new-in-yolov8/)，YOLOv8 直接預測物件中心，減少了框的預測數量，簡化了 NMS 過程。這不僅加速了後處理，還提升了準確性，因為無需依賴錨點框的聚類。
- 這種設計的優勢在於減少了參數量和計算量，特別是在處理密集場景時。研究顯示，這種方法在 COCO 和 RF100 數據集上表現優於 YOLOv5，尤其是在 mAP@.50 指標上。

##### 2. 高效的骨幹網路：C2f 模組與卷積優化

- 骨幹網路負責特徵提取，YOLOv8 基於 CSPDarknet53 架構進行改進。根據 [Detailed Explanation of YOLOv8 Architecture — Part 1]([https://medium.com/@juanpedro.bc22/detailed-explanation-of-yolov8](https://medium.com/@juanpedro.bc22/detailed-explanation-of-yolov8) architecture-part-1-6da9296b954e)，其主要改變包括：
    - 將 C3 模組替換為 C2f 模組。C2f 模組將所有 Bottleneck 輸出進行拼接，而 C3 僅使用最後一個輸出，這減少了層數和計算量。
    - 將初始的 6x6 卷積改為 3x3 卷積，減少了參數數量，因為較小的卷積核計算成本更低。
    - 瓶頸層（Bottleneck）的第一個卷積核從 1x1 改為 3x3，類似於 ResNet 的設計，這被證實有助於特徵提取。
- 這些改變使得骨幹網路更高效，特別是在資源受限的設備上。性能數據顯示，YOLOv8n 在 A100 TensorRT 上僅需 0.99 毫秒的推理時間，證明其低延遲特性。

##### 3. 優化的頸部設計：特徵融合的提升

- 頸部負責將骨幹提取的特徵進行融合，為頭部提供多尺度特徵。根據 [What's New in YOLOv8?](https://blog.roboflow.com/whats-new-in-yolov8/)，YOLOv8 的頸部在拼接特徵時不強制特定通道維度，這減少了參數數量，並提高了特徵整合的靈活性。
- 此外，頸部增加了一個新的卷積層，這有助於更好地融合不同尺度的特徵，特別是在檢測小物件時。研究顯示，這種設計提升了 mAP，特別是在處理多尺度物件的場景中。

#### 性能比較與為何快速

YOLOv8 的低延遲和高 mAP 得益於上述設計的綜合效果。以下是與其他模型的比較（基於 COCO 數據集）：

|模型|mAP val 50-95 (COCO)|Speed A100 TensorRT (ms)|params (M)|FLOPs (B)|
|---|---|---|---|---|
|YOLOv8n|37.3|0.99|3.2|8.7|
|YOLOv8s|44.9|1.20|11.2|28.6|
|YOLOv8m|50.2|1.83|25.9|78.9|
|Faster R-CNN|~40|~10|~40|~180|
|SSD|~30|~5|~24|~31|

（注：Faster R-CNN 和 SSD 的數據為估計值，基於公開文獻）

從表中可見，YOLOv8 在 mAP 和速度上均優於 Faster R-CNN 和 SSD。特別是其推理時間遠低於兩階段模型，這得益於單階段設計和無錨點頭的效率。

#### 訓練技巧的影響

雖然問題聚焦於網路結構，但值得一提的是，YOLOv8 的訓練過程也對性能有貢獻。例如，根據 [What's New in YOLOv8?](https://blog.roboflow.com/whats-new-in-yolov8/)，在最後 10 個 epoch 中關閉馬賽克增強（Mosaic Augmentation），這被證實有助於提升準確性。這種訓練策略可能出乎意料，但對最終 mAP 有顯著影響。

#### 與其他模型的比較

與其他物件檢測模型相比，YOLOv8 的優勢在於其平衡了速度和準確性。兩階段模型如 Faster R-CNN 雖然準確度高，但延遲較大（約 10 毫秒），不適合即時應用。單階段模型如 SSD 速度快（約 5 毫秒），但 mAP 較低（約 30%）。YOLOv8 的設計使其在這兩者之間找到最佳平衡點，特別是在即時視頻分析中。

#### 結論

YOLOv8 的網路結構設計，包括無錨點檢測頭、優化的骨幹網路和高效的頸部特徵融合，使其在低延遲和高 mAP 方面表現卓越。其快速的原因在於減少計算量和簡化後處理，而高準確性則得益於更好的特徵提取和融合能力。這些特點使其成為即時物件檢測的領先選擇。



### C2f模組

### YOLOv8 的 C2f 模組架構詳解

YOLOv8 的 C2f 模組（Cross-Stage Partial with Feature fusion）是其骨幹網路的核心組件之一，相較於 YOLOv5 的 C3 模組進行了改進，旨在提升特徵提取效率並減少計算量。以下將詳細解釋 C2f 模組的架構，包括其組成部分（Conv、Split、Bottleneck、Concat），並分析其設計優勢。

---

### C2f 模組的架構

C2f 模組的基本設計靈感來自 CSPDarknet 和 ResNet 的思想，通過分階段處理和特徵融合來提升性能。以下是其結構的逐步分解：

#### 1. **輸入與初始卷積（Conv）**

- **功能**：模組接收來自前一層的特徵圖作為輸入，並通過一個卷積層（通常是 3x3 卷積）進行初步處理。
- **細節**：
    - 卷積核大小：3x3（與 YOLOv8 骨幹中其他部分的設計一致）。
    - 步幅（Stride）：通常為 1，保持特徵圖空間分辨率。
    - 輸出通道數：根據模型大小（n、s、m 等）有所不同，例如 YOLOv8n 中可能是 64 或 128。
- **目的**：對輸入特徵進行初步轉換，提取低層次特徵，為後續分支處理做準備。

#### 2. **通道分割（Split）**

- **功能**：將卷積後的特徵圖按通道維度一分為二，分成兩部分。
- **細節**：
    - 假設輸入特徵圖的通道數為 C C C，則 Split 操作將其分成兩組，每組通道數為 C/2 C/2 C/2。
    - 這一步是 CSP（Cross-Stage Partial）設計的核心，旨在減少計算量並實現並行處理。
- **目的**：通過分割通道，降低每個分支的計算負擔，同時保留所有特徵信息。

#### 3. **Bottleneck 處理**

- **功能**：其中一部分特徵（通常是 C/2 C/2 C/2 通道）被送入一系列 Bottleneck 結構進行深層特徵提取。
- **Bottleneck 結構細節**：
    - 每個 Bottleneck 包含兩個卷積層：
        - 第一層：3x3 卷積（取代 YOLOv5 中常用的 1x1 卷積），通道數可能保持不變或擴展。
        - 第二層：1x1 卷積，用於壓縮通道數或調整特徵。
    - 殘差連接（Residual Connection）：輸入與輸出相加，形成類似 ResNet 的結構，緩解梯度消失問題。
    - Bottleneck 的數量：根據模型配置可調，例如 YOLOv8n 中可能是 1-3 個，YOLOv8m 中可能更多。
- **目的**：通過 Bottleneck 提取更深層次的特徵，同時控制參數量和計算量。

#### 4. **特徵拼接（Concat）**

- **功能**：將經過 Bottleneck 處理的特徵與未處理的另一半特徵（另一個 C/2 C/2 C/2 通道）進行拼接。
- **細節**：
    - Concat 操作在通道維度上進行，將兩部分特徵合併為一個完整的特徵圖，通道數恢復為 C C C 或根據設計有所調整。
    - 拼接後的特徵圖保留了原始信息（未經 Bottleneck 的部分）和深層特徵（經 Bottleneck 的部分）。
- **目的**：實現特徵融合，增強模組的表達能力。

#### 5. **最終卷積（Conv）**

- **功能**：對拼接後的特徵圖進行一次卷積處理，生成最終輸出。
- **細節**：
    - 通常使用 3x3 卷積，調整通道數以匹配下一層的需求。
    - 這一步平滑特徵並準備傳遞到下一個模組。
- **目的**：整合特徵並確保輸出與網路其他部分的兼容性。

---

### C2f 模組的結構圖示

以下是 C2f 模組的簡化流程：
```text
輸入特徵圖 (C 通道)
      |
   [Conv 3x3]
      |
   [Split] --> 特徵 A (C/2) ----> [Bottleneck x N] --> 特徵 A'
      |         特徵 B (C/2) -----------------------> 特徵 B
      |                    |
      |                [Concat]
      |                    |
      |               [Conv 3x3]
      |                    |
輸出特徵圖
```


- **特徵 A**：經過 Bottleneck 處理的部分，提取深層特徵。
- **特徵 B**：未處理的部分，保留原始信息。
- **N**：Bottleneck 的數量，根據模型大小可調。

---

### 設計優勢分析

C2f 模組的設計在計算效率、特徵提取能力和模型性能之間取得了平衡。以下是其具體優勢：

#### 1. **計算效率提升**

- **減少計算量**：通過 Split 操作將特徵分成兩部分，只對一半特徵進行 Bottleneck 處理，減少了總體計算量。相比於對所有通道都應用深層卷積，C2f 的 FLOPs（浮點運算次數）更低。
- **證據**：根據 YOLOv8 的性能數據（參見 Ultralytics 官方文檔），YOLOv8n 的 FLOPs 僅為 8.7B，遠低於 YOLOv5m 的 49.0B，這部分得益於 C2f 的高效設計。

#### 2. **特徵融合的增強**

- **多層次特徵保留**：Concat 操作將原始特徵（特徵 B）與深層特徵（特徵 A'）結合，保留了低層次細節（如邊緣、紋理）和高層次語義信息（如物件形狀）。
- **優勢**：這種融合提升了模型在多尺度檢測中的性能，特別是在 COCO 數據集上檢測小物件時，mAP 有顯著提升。

#### 3. **梯度流量的改善**

- **殘差結構**：Bottleneck 中的殘差連接有助於梯度傳播，緩解深層網路中的梯度消失問題。
- **跨階段設計**：Split 和 Concat 的 CSP 結構允許特徵在不同階段間流動，進一步優化訓練過程。
- **證據**：研究顯示，這種設計使 YOLOv8 在訓練收斂速度和穩定性上優於 YOLOv5。

#### 4. **靈活性與可擴展性**

- **參數可調**：Bottleneck 的數量和卷積層的通道數可以根據模型大小（n、s、m、l、x）靈活調整。例如，YOLOv8n 使用較少的 Bottleneck（輕量化），而 YOLOv8x 使用更多（高性能）。
- **優勢**：這種模組化設計使其適應不同應用場景，從移動設備到高性能 GPU 都能高效運行。

#### 5. **與無錨點頭的協同效應**

- C2f 模組提取的特徵圖質量更高，為 YOLOv8 的無錨點檢測頭提供了更好的輸入。無錨點設計依賴於直接預測物件中心，而 C2f 的多層次特徵融合有助於精確定位和分類。

---

### 與 C3 模組的比較

為了更好地理解 C2f 的優勢，以下是與 YOLOv5 中 C3 模組的對比：

|特性|C3 模組 (YOLOv5)|C2f 模組 (YOLOv8)|
|---|---|---|
|**結構**|Split + Bottleneck + Concat|Split + Bottleneck + Concat|
|**Bottleneck 輸出**|僅使用最後一個 Bottleneck 輸出|拼接所有 Bottleneck 輸出|
|**初始卷積**|1x1 卷積|3x3 卷積|
|**計算量**|較高|較低|
|**特徵融合**|簡單拼接|更豐富的融合|

- **關鍵差異**：C2f 拼接所有 Bottleneck 的輸出，而非僅最後一個，這增加了特徵的豐富性；同時，3x3 卷積替代 1x1 卷積，提升了特徵提取能力。
- **結果**：C2f 在保持低計算量的同時，提供更高的 mAP（例如 YOLOv8n 的 37.3% 對比 YOLOv5n 的 28.0%）。

---

### 結論

C2f 模組通過 Conv、Split、Bottleneck 和 Concat 的協同設計，實現了高效的特徵提取與融合。其優勢在於降低了計算量、增強了特徵表達能力、改善了梯度流動，並提供了靈活的擴展性。這些特性使得 YOLOv8 在低延遲和高 mAP 方面表現卓越，特別適合即時物件檢測應用。














![[Pasted image 20250325192253.png]]

![[Pasted image 20250325192309.png]]
本文结合[YOLOv5](https://zhida.zhihu.com/search?content_id=227756901&content_type=Article&match_order=1&q=YOLOv5&zhida_source=entity)网络进行讲解，通过与YOLOv5网络进行比较，进一步理解YOLOv8，尽快上手

### **（1）YOLOv8与YOLOv5比较**

**相同点：**

从整体上来看，YOLOv8和YOLOv5基本一致，都是backbone + [PANet](https://zhida.zhihu.com/search?content_id=227756901&content_type=Article&match_order=1&q=PANet&zhida_source=entity) + Head的结构，且PANet部分都是先上采样融合再下采样融合；

**不同点：**

<1> Head部分不同，YOLOv5是整体上输出的，以80类为例，因为每个像素点为3个anchor，故每个像素点的size为：3*（4 + 1 + 80 ）= 255；而YOLOv8Head部分，Cls和Box是分开预测的,并且从Anchor-Based换成了Anchor-Free。

<2> YOLOv8的**Backbone**和**Neck**中采用的**[C2f结构](https://zhida.zhihu.com/search?content_id=227756901&content_type=Article&match_order=1&q=C2f%E7%BB%93%E6%9E%84&zhida_source=entity)**，其参考了**YOLOv7**的**ELAN**的设计思想，用于替换**YOLOv5**中的**[CSP结构](https://zhida.zhihu.com/search?content_id=227756901&content_type=Article&match_order=1&q=CSP%E7%BB%93%E6%9E%84&zhida_source=entity)**，由于**C2f结构**有着更多的残差连接，所以其有着**更丰富的梯度流**。(不过这个C2f模块中存在Split操作，对特定硬件部署并不友好)

<3> **Loss** 计算方面采用了**[TaskAlignedAssigner](https://zhida.zhihu.com/search?content_id=227756901&content_type=Article&match_order=1&q=TaskAlignedAssigner&zhida_source=entity)正样本匹配策略**，并引入了**[Distribution Focal Loss](https://zhida.zhihu.com/search?content_id=227756901&content_type=Article&match_order=1&q=Distribution+Focal+Loss&zhida_source=entity)**.

<4> **训练部分**，采用了**YOLOX**的训练方式，在最后的10个Epoch关闭了Mosiac增强操作，可以有效地提升精度。





### YOLOv8 的 PAN 架構詳解

在 YOLOv8 中，PAN（Path Aggregation Network）是頸部（Neck）結構的核心部分，負責將骨幹網路提取的多尺度特徵進行融合，為檢測頭提供豐富的語義和空間信息。YOLOv8 的 PAN 結構在 YOLOv5 的基礎上進行了改進，進一步優化了特徵融合效率和性能。以下將詳細解釋其架構設計，包括組成部分、數據流和工作原理，並分析其優勢。

---

### PAN 的背景與作用

PAN 最初由 Liu 等人於 2018 年提出（參見論文 _Path Aggregation Network for Instance Segmentation_），旨在通過自底向上和自頂向下的特徵融合，解決物件檢測中多尺度問題。YOLOv8 沿用了這一思想，並結合其自身的設計目標（低延遲、高 mAP），對 PAN 進行了調整。其主要作用包括：

- **多尺度特徵融合**：將骨幹網路輸出的深層語義特徵（高層）和淺層空間特徵（低層）整合。
- **提升檢測性能**：特別是在檢測小物件和大物件時，提供更強的特徵表達能力。

在 YOLOv8 中，PAN 位於骨幹（Backbone）和頭部（Head）之間，接收骨幹的輸出（通常是 P3、P4、P5 三個尺度的特徵圖），並生成融合後的特徵供檢測頭使用。

---

### YOLOv8 PAN 的架構

YOLOv8 的 PAN 結構可以分為兩個主要路徑：**自頂向下（Top-Down）路徑**和**自底向上（Bottom-Up）路徑**，通過卷積操作和特徵拼接實現高效融合。以下是其詳細架構：

#### 1. **輸入來源**

- PAN 接收骨幹網路（基於 CSPDarknet 和 C2f 模組）的多尺度特徵圖輸出，通常包括：
    - **P3**：淺層特徵（分辨率高，通道數少，例如 80x80x256）。
    - **P4**：中層特徵（分辨率中等，例如 40x40x512）。
    - **P5**：深層特徵（分辨率低，通道數多，例如 20x20x1024）。
- 這些特徵圖由骨幹網路的 C2f 模組生成，代表不同層次的語義和空間信息。

#### 2. **自頂向下（Top-Down）路徑**

- **目標**：將深層特徵（P5）的語義信息傳遞到淺層（P4 和 P3），增強淺層特徵的語義表達能力。
- **結構**：
    1. **P5 到 P4**：
        - P5 特徵圖通過 **上採樣（Upsample）** 操作（通常是 2x 上採樣，使用最近鄰插值）將分辨率提升到與 P4 一致（例如 20x20 -> 40x40）。
        - 上採樣後的 P5 特徵與 P4 特徵進行 **Concat**（通道維度拼接）。
        - 拼接後的特徵通過一個 **卷積層（Conv）**（通常是 3x3 卷積）處理，調整通道數並平滑特徵，生成新的特徵圖 N4。
    2. **P4 到 P3**：
        - N4 特徵圖再次通過 **上採樣**（2x 上採樣，例如 40x40 -> 80x80）。
        - 上採樣後的 N4 與 P3 特徵進行 **Concat**。
        - 拼接後的特徵再次通過 **卷積層（Conv）**，生成新的特徵圖 N3。
- **細節**：
    - 上採樣通常不引入額外參數，僅調整空間分辨率。
    - 卷積層使用 3x3 卷積核，步幅為 1，可能搭配 BN（Batch Normalization）和 SiLU 激活函數。

#### 3. **自底向上（Bottom-Up）路徑**

- **目標**：將淺層特徵（N3）的空間信息傳遞回深層（N4 和 P5），增強深層特徵的定位能力。
- **結構**：
    1. **N3 到 N4**：
        - N3 特徵圖通過 **下採樣（Downsample）** 操作（通常是 3x3 卷積，步幅為 2）將分辨率降低到與 N4 一致（例如 80x80 -> 40x40）。
        - 下採樣後的 N3 與 N4 進行 **Concat**。
        - 拼接後的特徵通過 **卷積層（Conv）**，生成新的特徵圖 N4'。
    2. **N4 到 P5**：
        - N4' 特徵圖再次通過 **下採樣**（3x3 卷積，步幅為 2，例如 40x40 -> 20x20）。
        - 下採樣後的 N4' 與原始 P5 特徵進行 **Concat**。
        - 拼接後的特徵通過 **卷積層（Conv）**，生成最終的特徵圖 N5。
- **細節**：
    - 下採樣使用帶步幅的卷積而非池化層，保留更多特徵信息。
    - 每次 Concat 和卷積後，通道數會根據模型配置進行調整（例如 YOLOv8n 中可能為 256、512 等）。

#### 4. **輸出**

- PAN 最終生成三個融合特徵圖：**N3、N4'、N5**，分別對應不同尺度（例如 80x80、40x40、20x20）。
- 這些特徵圖被傳遞到檢測頭，用於預測物件類別、邊界框和置信度。

---

### PAN 的結構圖示

以下是 YOLOv8 PAN 的簡化流程：
```
骨幹輸出：
P3 (80x80x256)       P4 (40x40x512)       P5 (20x20x1024)
   |                    |                    |
   |                    |                 [Upsample]
   |                    |                    |
   |                 [Concat] <-------------+
   |                    |                    |
   |                 [Conv]                 |
   |                    |                    |
   |                 N4 (40x40x512)         |
   |                    |                    |
[Concat] <---------- [Upsample]             |
   |                    |                    |
[Conv]                 |                    |
   |                    |                    |
N3 (80x80x256)         |                    |
   |                    |                    |
[Downsample] ---------> [Concat]            |
   |                    |                    |
   |                 [Conv]                 |
   |                    |                    |
   |                 N4' (40x40x512)        |
   |                    |                    |
   +----------------> [Downsample]          |
                        |                    |
                     [Concat] <------------+
                        |
                     [Conv]
                        |
                     N5 (20x20x1024)

輸出：N3, N4', N5 -> 檢測頭
```

---

### YOLOv8 PAN 的設計特點與改進

YOLOv8 的 PAN 在 YOLOv5 的基礎上進行了以下改進：

1. **簡化結構**：
    - YOLOv5 的 PAN 使用了更多的卷積模組（例如 SPPF 和額外的 C3 模組），而 YOLOv8 將這些模組替換為更高效的 C2f 模組，並簡化了頸部結構。
    - 減少了不必要的層數，降低了計算量。
2. **靈活的通道調整**：
    - 在 Concat 後的卷積層中，YOLOv8 不強制固定的通道數，而是根據模型大小（n、s、m 等）動態調整，減少了冗餘參數。
3. **與無錨點頭協同**：
    - YOLOv8 的無錨點檢測頭需要更高質量的特徵輸入，PAN 的雙向融合確保了特徵圖同時具備語義和空間信息。

---

### 設計優勢分析

#### 1. **多尺度特徵融合**

- **自頂向下**路徑將深層語義信息傳遞到淺層，提升小物件檢測的準確性。
- **自底向上**路徑將淺層空間信息傳遞到深層，增強大物件的定位能力。
- **證據**：根據 Ultralytics 官方數據，YOLOv8 在 COCO 數據集上的 mAP@.50:.95 顯著優於 YOLOv5，尤其是在小物件檢測中。

#### 2. **計算效率**

- 使用簡單的上採樣（最近鄰插值）和下採樣（步幅卷積），避免了複雜的池化或轉置卷積，減少了計算開銷。
- 相比 Faster R-CNN 的 FPN（Feature Pyramid Network），YOLOv8 的 PAN 結構更輕量化，推理速度更快（例如 YOLOv8n 在 A100 上僅 0.99ms）。

#### 3. **適應性**

- PAN 的雙向路徑設計使其適應不同尺度的物件檢測任務，從移動設備上的輕量模型（YOLOv8n）到高性能應用（YOLOv8x）都能高效運行。

#### 4. **與骨幹和頭部的協同**

- 與 C2f 骨幹生成的特徵圖無縫銜接，提供高質量的多尺度輸入。
- 為無錨點檢測頭提供精確的特徵支持，減少了框預測的計算負擔。

---

### 與其他模型的比較

|特性|YOLOv8 PAN|YOLOv5 PAN|FPN (Faster R-CNN)|
|---|---|---|---|
|**路徑**|自頂向下 + 自底向上|自頂向下 + 自底向上|自頂向下|
|**模組**|C2f + Conv|C3 + SPPF|Conv|
|**計算量**|低|中|高|
|**小物件檢測**|優|中|優|
|**速度**|快|中|慢|

- **YOLOv8 vs. YOLOv5**：YOLOv8 的 PAN 更輕量，融合效率更高，mAP 和速度均有提升。
- **YOLOv8 vs. FPN**：FPN 僅有自頂向下路徑，計算量更大，適合高精度但不適合即時應用。

---

### 結論

YOLOv8 的 PAN 通過自頂向下和自底向上的雙向路徑，結合上採樣、下採樣和卷積操作，實現了高效的多尺度特徵融合。其設計優勢在於提升了檢測性能（特別是小物件）、降低了計算量，並與無錨點檢測頭協同工作，使 YOLOv8 在低延遲和高 mAP 方面表現出色。這一結構是 YOLOv8 能夠快速且精確檢測物件的關鍵組成部分。



### DarkNet vs CSPDarkNet

詳細解釋 Darknet、CSPDarknet、YOLOv8 的骨幹網路 (backbone model) 特點，以及 C3 模組和 C2f 模組之間的差異。

### 1. Darknet 與 CSPDarknet 的差別

**Darknet (以 Darknet-53 為代表，用於 YOLOv3)**

- **核心思想**：Darknet 是一個深度卷積神經網路，專為YOLO物件偵測設計。Darknet-53（YOLOv3 使用）包含 53 個卷積層。
- **結構特點**：
    - 主要由一系列的 1x1 卷積層和 3x3 卷積層堆疊而成。
    - **殘差連接 (Residual Connections)**：Darknet-53 大量借鑒了 ResNet 的思想，引入了殘差塊 (Residual Blocks)。一個殘差塊通常包含兩個卷積層和一個跨層連接 (shortcut connection)，將輸入直接加到輸出上。這有助於解決深度網路中的梯度消失問題，使得網路可以做得更深。
    - **無池化層 (No Pooling Layers)**：在主要的特徵提取部分，Darknet-53 通常使用步長為 2 的卷積層 (strided convolution) 來進行下採樣，而不是傳統的 Max Pooling 層，這有助於保留更多特徵信息。
    - **全卷積 (Fully Convolutional)**：使其能夠處理不同尺寸的輸入圖像。
- **目標**：作為一個強大的特徵提取器，為後續的 YOLO 檢測頭提供高質量的特徵圖。
- **優點**：
    - 通過殘差結構，網路可以做得較深，提取更豐富的特徵。
    - 在準確性和速度之間取得了較好的平衡（在當時）。
- **缺點**：
    - 相較於後來的 CSPNet 結構，計算量仍然較大。
    - 梯度信息在非常深層的網路中依然可能存在傳播不夠高效的問題。

**CSPDarknet (以 CSPDarknet53 為代表，用於 YOLOv4)**

![[Pasted image 20250515130459.png]]

![[Pasted image 20250515125454.png]]

- **核心思想**：CSPNet (Cross Stage Partial Network) 的核心思想是將特徵圖在進入一個階段 (stage) 的開始時，從通道維度分成兩部分。一部分經過該階段的密集塊 (dense block) 或殘差塊 (residual block) 處理，另一部分則幾乎不經過處理（或只經過少量處理），然後在階段結束時將這兩部分拼接 (concatenate) 起來。
- **CSPDarknet53 的結構特點**：
    - **跨階段局部連接 (Cross Stage Partial connections)**：這是其最核心的改進。在 Darknet-53 的基礎上，將每個大的殘差階段應用 CSP 結構。
    - **具體實現**：
        1. 輸入特徵圖 X0​ 分成兩部分：$X_0' $ 和 X0′′​。
        2. X0′′​ 直接連接到階段的末尾（或者只經過一個簡單的 1x1 卷積）。
        3. X0′​ 經過原來 Darknet-53 中的一系列殘差塊（或其他卷積塊）進行特徵提取，得到 Xk​。
        4. 將 Xk​ 和 X0′′​ 在通道維度上進行拼接。
        5. 通常在拼接後還會進行一次過渡層 (transition layer，如 1x1 卷積) 來融合特徵。
- **目標**：
    - **減少計算瓶頸**：只有一部分特徵圖參與了複雜的計算。
    - **增強梯度反向傳播的效率**：梯度可以通過更短的路徑（未經處理的那部分）傳播，減少梯度消失的風險。
    - **提升學習能力**：通過分離特徵流再融合，可以獲得更豐富的特徵表示，同時減少重複的梯度信息。
- **優點**：
    - **更高的計算效率**：在達到相似甚至更好性能的同時，顯著降低了計算量 (FLOPs)。
    - **更快的推論速度**。
    - **更好的準確性**：通過改善梯度流和減少冗餘計算，有助於提升模型準確性。
    - **易於實現**：可以方便地應用到多種卷積神經網路架構中。
- **缺點**：相較於最新的設計，可能還有優化空間。

**總結 Darknet vs CSPDarknet：**

CSPDarknet 是對 Darknet 的一種優化。它通過在網路的每個主要階段引入跨階段局部連接，將基礎層的特徵圖分成兩部分，一部分進行主要的卷積運算，另一部分直接（或少量處理後）與處理後的特徵圖進行拼接。這樣做的好處是**減少了計算量、增強了梯度的反向傳播、並提升了學習的準確性**，因為它避免了在不同層之間學習重複的梯度信息。

### 2. YOLO v8 backbone model (基於 C2f) 與 CSPDarknet 的差別

YOLOv8 的骨幹網路可以看作是 CSPDarknet 思想的進一步演進和優化，其核心是大量使用 **C2f (CSPNet with 2 features) 模組**。

- **繼承與發展**：YOLOv8 的骨幹網路依然遵循 CSP 的設計哲學，即通過特徵圖的分割與融合來優化計算和梯度流。
- **核心模組 C2f**：YOLOv8 的骨幹網路由一個初始的 Stem 層（通常是一個卷積層）和堆疊的多個 C2f 模組構成，最後通常接一個 SPPF (Spatial Pyramid Pooling Fast) 模組來提取多尺度特徵。
- **與 CSPDarknet53 的主要區別**：
    1. **模組設計的進化**：CSPDarknet53 使用的是基於 Darknet 殘差塊的 CSP 結構。而 YOLOv8 使用的是 C2f 模組，C2f 模組本身在內部結構上與傳統的 C3 (YOLOv5 中使用的 CSP 模組) 有所不同，更加註重梯度流的豐富性和特徵融合的效率（詳見下一節 C3 vs C2f）。
    2. **輕量化和高效性**：YOLOv8 的設計更加註重模型在不同硬體平台上的效率和可擴展性。其不同大小的模型 (n, s, m, l, x) 是通過調整網路的深度和寬度（通道數）以及 C2f 模組內部 Bottleneck 的數量來實現的。整體設計比 CSPDarknet53 可能更精簡和高效。
    3. **無頭檢測頭 (Anchor-Free Head)**：雖然這部分屬於檢測頭 (Head) 而非骨幹網路 (Backbone)，但 YOLOv8 採用了 Anchor-Free 的檢測頭，並進行了 Decoupled Head 的設計，這也影響了對骨幹網路特徵的要求。
    4. **模塊化和靈活性**：YOLOv8 的代碼實現更加模塊化，使得用戶更容易替換或修改骨幹網路的組件。

可以認為 YOLOv8 的骨幹網路是 CSPNet 思想在現代硬體和模型設計理念下的最新迭代，通過 C2f 這樣更優化的 CSP 變體，力求在速度和精度上達到新的平衡。

### 3. C3 模組 與 C2f 模組 的差別

這兩個模組都是 CSPNet 思想的具體實現，主要用於 YOLO 系列的骨幹網路和頸部 (Neck) 結構。

**C3 模組 (主要用於 YOLOv5, YOLOv7 等)**

- **名稱來源**：C3 代表 CSP Bottleneck3。它有 3 個主要的卷積層（不包括 Bottleneck 內部），並且應用了 CSP 結構。
- **結構**：
    1. 輸入特徵圖首先通過兩個 1x1 卷積層，將其分成兩路 (split)。
        - 一路（主幹路）進入 `n` 個串聯的 Bottleneck 模組（Bottleneck 模組通常是 1x1 卷積降維 -> 3x3 卷積特徵提取 -> 1x1 卷積升維，並帶有殘差連接）。
        - 另一路（旁路）直接連接（或經過一個簡單的 1x1 卷積）。
    2. 兩路特徵圖在通道維度上進行拼接 (Concatenate)。
    3. 拼接後的特徵圖再經過一個 1x1 卷積層進行特徵融合和通道調整。
- **特點**：
    - 相對簡潔的 CSP 實現。
    - `n` 個 Bottleneck 是串行處理的。
    - 旨在減少計算量並保持較好的特徵提取能力。
- **變體**：
    - `C3TR`: 帶有 Transformer 模塊的 C3。
    - `C3Ghost`: 使用 GhostNet 中的 Ghost Bottleneck 的 C3。

**C2f 模組 (用於 YOLOv8)**

- **名稱來源**：C2f 代表 "CSPNet with 2 features" 或者可以理解為更高效的 CSP 結構，通常帶有兩個分支（一個主幹，一個旁路）並強調更豐富的特徵融合。
- **結構**：
    1. 輸入特徵圖首先經過一個 1x1 卷積進行通道數調整 (split_conv0)。
    2. 然後特徵圖被分成兩部分（概念上的旁路和主路）。
        - 一部分特徵（旁路，通過 split_conv0 的一部分通道）直接向後傳遞。
        - 另一部分特徵（主路，通過 split_conv0 的另一部分通道）進入 `n` 個串聯的 Bottleneck (或者在 YOLOv8 中常稱為 `DarkBlock` 或類似的基礎塊，但結構類似於 Bottleneck：包含兩個 3x3 卷積和一個殘差連接，或者一個 1x1 和一個 3x3 卷積)。
        - **核心區別**：與 C3 不同的是，在 C2f 中，這 `n` 個 Bottleneck 的**每一個**（或者說，是這些 Bottleneck 形成的序列的**中間輸出**）的輸出，都會被**再次分割並與較早的特徵融合**。具體來說，主路經過第一個 Bottleneck 後，其輸出會被收集起來；經過第二個 Bottleneck 後，其輸出也會被收集起來，以此類推。
    3. 所有從主路中收集到的中間 Bottleneck 輸出以及最初的旁路特徵，會在通道維度上進行拼接 (Concatenate)。
    4. 拼接後的特徵圖再經過一個 1x1 卷積層進行最終的特徵融合和通道調整。
- **特點**：
    - **更豐富的梯度流路徑**：由於將 Bottleneck 序列的中間輸出也引入到最終的拼接中，C2f 提供了更短、更直接的梯度傳播路徑，有利於深層網路的訓練。
    - **更強的特徵融合能力**：融合了更多不同層次的特徵，理論上可以帶來更強的表徵能力。
    - **輕量化設計**：YOLOv8 中的 Bottleneck 設計（有時僅用一個 3x3 卷積的變體，或兩個 3x3 卷積）以及整體 C2f 的參數控制，使其在保持性能的同時也注重效率。
- **與 C3 的主要差異總結**：
    - **Bottleneck 輸出的處理**：C3 中，`n` 個 Bottleneck 的輸出是串行傳遞的，只有最後一個 Bottleneck 的輸出與旁路拼接。而在 C2f 中，`n` 個 Bottleneck 序列中，**每個 Bottleneck 的輸出（或其一部分）都會被收集起來，用於最終的拼接**。這意味著 C2f 融合了更多層次的特徵。
    - **梯度流**：C2f 的設計提供了更豐富的梯度路徑，這被認為是其性能提升的關鍵之一。
    - **參數效率和計算量**：雖然看起來更複雜，但通過精心設計 Bottleneck 內部結構和通道數，C2f 旨在實現更高的參數效率和計算效率。

**簡而言之：**

- **Darknet** 是基礎，引入了殘差連接。
- **CSPDarknet** 在 Darknet 基礎上引入了跨階段部分網路（CSP），將特徵流分成兩部分處理再融合，以減少計算量和改善梯度流。
- **YOLOv8 Backbone** 是 CSP 思想的延續和優化，主要使用 **C2f** 模組。
- **C3** 模組是 YOLOv5/v7 中的 CSP 實現，將一組串行 Bottleneck 的最終輸出與旁路融合。
- **C2f** 模組是 YOLOv8 中的 CSP 實現，它將一組串行 Bottleneck 的**每個中間輸出**都收集起來與旁路進行更充分的融合，以獲得更豐富的梯度信息和特徵表示。

這種演進體現了在物件偵測領域對模型效率、準確性和梯度流優化的持續追求。






Reference: 
一文弄懂|YOLOv8网络结构配置文件，yolov

8.yaml详细解读与说明 - AI技术分享的文章 - 知乎
https://zhuanlan.zhihu.com/p/715019537

YOLOv5、YOLOv6、YOLOv7、YOLOv8、YOLOv9、YOLOv10、YOLOv11、YOLOv12各版本的网络结构图 - 笑脸惹桃花的文章 - 知乎
https://zhuanlan.zhihu.com/p/31629637609

