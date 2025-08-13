

从2014年为分割，之前的工作都是traditional detector，之后的都是DL-based detector。从以下三个方面介绍了相应的milestone detectors。

1. Traditional detectors:  Viola Jones,  HOG,  DPM
2. _CNN based Two-stage Detectors:  **RCNN series(RCNN, Fast RCNN, Faster RCNN)** (Anchor)
3. CNN based One-stage Detectors:  **YOLO series,  FCOS, SSD,  RetainNet, EfficientDet, DETR, MobileNet-SSD**

New: 
[RF-DETR](https://github.com/roboflow/rf-detr)：60.5 mAP + 6ms延迟，实时检测领域的新王者如何碾压YOLO？ [link](https://zhuanlan.zhihu.com/p/32205292924)


|                                           |     |
| ----------------------------------------- | --- |
| [[#### 五種Object detection綜合比較: ]]         |     |
| [[#### 五種Object detection用在特別datasets: ]] |     |
| [[#### 各模型在手術影片偵測專案中的fine tune]]          |     |
| [[### QA-list]]                           |     |

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

[[SSD]]: 使用 VGG(移除 FC 層)，neck不是用FPN而是添加額外卷積層成多尺度特徵層，不用region proposal而是適配單階段檢測(==就是保留RCNN(但移除FC), Neck也不是用FPN而是多尺度特徵層. 也把head model改成anchor-free)==)。

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


#### 五種Object detection綜合比較: 

```
Question: 請幫我儘量完整比較(要包括training效能, inference效能, 正確率, training, 是否容易finetune, model大小, 是否需要後續步驟譬如NMS等等等...)並列表: 比較Faster RCNN, YOLO, DETR, DINOv2加上Faster R-CNN head, Grounding DINO. 如果有數字請儘量提供數字
```

好的，這是一個非常專業且全面的問題。針對您提出的五個代表性物件偵測模型：**Faster R-CNN**, **YOLO**, **DETR**, **DINOv2 + Faster R-CNN Head**, 以及 **Grounding DINO**，我將從多個維度進行詳細的比較。

由於模型的版本迭代非常快，這裡的比較將基於各個架構的典型代表和核心思想。例如，YOLO會以現代版本（如YOLOv8/v9）為參考，而Faster R-CNN則以其經典的ResNet骨幹為基礎。

### 核心思想速覽

在進入詳細比較前，先快速理解每個模型的「靈魂」：

1. **Faster R-CNN**: **兩階段偵測的鼻祖**。先由一個子網路（RPN）判斷「哪裡可能有物體？」，再由主網路回答「這個物體是什麼？位置在哪？」。準確但較慢。
    
2. **YOLO (You Only Look Once)**: **單階段偵測的速度之王**。將圖片劃分成網格，直接在網格上預測物體類別和邊界框。極快，但早期版本對小目標偵測較弱。
    
3. **DETR (DEtection TRansformer)**: **端到端偵測的革命者**。首次將Transformer架構引入，把偵測視為「集合預測」問題，拋棄了NMS等手工組件。訓練慢，但設計優雅。
    
4. **DINOv2 + Faster R-CNN Head**: **基礎模型（Foundation Model）的威力展現**。利用在海量無標註數據上自監督學習出的超強特徵提取器（DINOv2），再接上一個經典的偵測頭進行微調。核心在於「強大的骨幹網路」。
    
5. **Grounding DINO**: **開放詞彙（Open-Vocabulary）偵測的未來**。結合了DETR的架構和視覺-語言預訓練，使其能夠偵測**任意文字描述**的物體，而不僅限於訓練集裡的固定類別。
    

---

### 詳細比較列表

下面的表格將從多個維度進行詳細比較。其中的數字（如mAP, FPS）會因骨幹網路大小、硬體、輸入尺寸而有很大差異，這裡提供的是一個具代表性的量級參考，重點在於比較相對關係。

|特性維度|Faster R-CNN|YOLO (以YOLOv8為例)|DETR|DINOv2 + Faster R-CNN|Grounding DINO|
|---|---|---|---|---|---|
|**核心架構**|兩階段 (Two-Stage)|單階段 (One-Stage)|端到端 (End-to-End) Transformer|兩階段 (自監督骨幹+偵測頭)|端到端, 開放詞彙|
|**偵測哲學**|區域提議 + 分類回歸|直接回歸|集合預測 (Set Prediction)|區域提議 + 分類回歸|語言引導的集合預測|
|**正確率 (COCO mAP)**|中-高 (約 42% AP @ ResNet50)|中-高 (YOLOv8-L: 52.9% AP)|高 (52.1% AP @ ResNet50)|**極高** (ViT-L骨幹可達60%+ AP)|**極高** (Swin-L骨幹: 63.0% AP on COCO)|
|**Inference效能/速度**|慢 (約 5-15 FPS)|**極快** (YOLOv8-L: ~100+ FPS @ T4 GPU)|較慢 (約 20-30 FPS)|慢 (與Faster R-CNN相當)|慢 (約 10-20 FPS)|
|**Training效能/效率**|較快|**最快**|**極慢** (收斂需要非常長的時間, 約500 epochs)|骨幹已預訓練, **Finetune極快**|預訓練極慢, Finetune較快|
|**訓練方式**|監督式學習|監督式學習|監督式學習|**自監督預訓練** + 監督式微調|**視覺-語言對比學習** + 監督式微調|
|**是否容易Finetune?**|容易, 非常成熟|**非常容易**, 生態系完善|較難, 訓練週期長, 超參數敏感|**非常容易**, 只需微調偵測頭, 數據需求少|較容易, 適合特定領域的開放詞彙擴展|
|**Model大小**|中 (ResNet50約100MB+)|小-中 (YOLOv8-L: 43M params)|大 (DETR-R50: 41M params)|**極大** (DINOv2 ViT-L: 300M+ params)|**極大** (Swin-L骨幹: 200M+ params)|
|**是否需要NMS?**|**是** (核心步驟)|**是** (但新版如YOLOv10嘗試移除)|**否** (核心優勢)|**是** (因使用Faster R-CNN頭)|**否** (核心優勢)|
|**獨特能力**|經典、穩定、可解釋性強|速度與精度的極致平衡, 產業應用廣泛|簡化偵測流程, 無需手工組件|利用無標註數據的力量, 在小數據集上微調效果驚人|**開放詞彙偵測**, 可用自然語言描述來找物體|
|**最適用場景**|學術研究基準, 對速度要求不高的應用|**實時檢測**, 邊緣計算, 工業級應用|需要簡化Pipeline, 避免調參NMS的場景|標註數據有限但要求高精度的專業領域 (如醫學影像, 衛星圖)|**人機互動**, 機器人, 圖像搜尋, 從未見過的物體偵測|

---

### 各模型特性深入解析

#### 1. Faster R-CNN

- **優勢**: 作為兩階段方法的標竿，其結構清晰，易於理解和修改。在許多學術論文中仍被用作比較的基準（Baseline）。
    
- **劣勢**: 速度是其最大瓶頸。RPN和後續的RoI Pooling/Align步驟使其無法達到實時。NMS的存在也讓它不是一個完全端到端的系統。
    

#### 2. YOLO

- **優勢**: 速度！YOLO系列的核心就是快，使其成為工業界和需要實時響應的應用的首選。生態非常成熟，有大量的教程、預訓練模型和部署工具（如Ultralytics框架）。
    
- **劣勢**: 為了速度，犧牲了一部分小目標和密集目標的偵測精度（儘管新版本已大幅改善）。模型的可解釋性相較Faster R-CNN稍差。
    

#### 3. DETR

- **優勢**: 概念上的革新。它證明了物件偵測可以像機器翻譯一樣，用一個純粹的Transformer架構來解決，並且擺脫了NMS的束縛。
    
- **劣勢**: **訓練收斂極慢**是其最著名的缺點，需要比Faster R-CNN多10-20倍的訓練時間才能達到好的效果。對小目標的偵測效果也不如後來經過優化的模型。
    

#### 4. DINOv2 + Faster R-CNN Head

- **優勢**: **大力出奇跡**。DINOv2骨幹網路在海量的無標註數據上學到了極其豐富和魯棒的視覺特徵。這意味著，即使你的下游任務標註數據很少（例如只有幾百張），只需微調其後面的偵測頭，就能達到非常高的精度。**極大地降低了對標註數據量的依賴**。
    
- **劣勢**: 骨幹網路本身非常龐大，對推理硬體的要求較高。其速度和架構複雜度繼承了所選偵測頭（Faster R-CNN）的缺點。
    

#### 5. Grounding DINO

- **優勢**: **真正的「所見即可得」**。它最大的突破是打破了「封閉集合」（Closed-Set）的限制。傳統模型只能偵測訓練時見過的幾十或幾百個類別。Grounding DINO可以接收任意文字輸入，例如「一隻正在跳躍的棕色小狗」或「照片左邊那個紅色的杯子」，並把它們定位出來。這使其在交互式應用和未知場景探索中潛力無限。
    
- **劣ě勢**: 模型龐大且複雜，結合了圖像Encoder和文字Encoder，推理速度較慢，對計算資源要求很高。雖然可以零樣本（Zero-shot）偵測，但在特定領域的精度可能還需要微調來提升。
    

### 總結與建議

- **追求極致速度和產業化部署**：選擇 **YOLO** 系列。
    
- **追求簡潔的端到端架構且不介意訓練時長**：選擇 **DETR** 或其改進版（如DINO）。
    
- **標註數據非常有限，但希望達到SOTA精度**：選擇 **DINOv2 + 偵測頭** 的方案。
    
- **需要偵測從未見過的新物體，或需要與自然語言交互**：**Grounding DINO** 是目前的不二之選，代表了偵測領域的未來方向。
    
- **作為學習和研究的基準**：**Faster R-CNN** 仍然是一個非常好的起點。


#### 五種Object detection用在特別datasets: 

```
Question: 那如果我要做的project是手術video上的object detection, 已經跟常用dataset譬如COCO有很大差異. 那請比較用上述五種object detection的各種優劣, 以及需要做哪些手續譬如重新fine turn等等. 以及哪幾種最適合?
```

好的，這是一個非常實際且極具價值的專案方向。手術影片的物件偵測是一個典型的「小眾但專業」（Domain-Specific）問題，與COCO這種通用資料集有天壤之別。這意味著模型的選擇策略和所需的工作流程會非常不同。

讓我們針對您的專案，重新深入評估這五種模型。

### 核心挑戰：手術影片偵測的特殊性

1. **巨大的領域差異 (Domain Gap)**：COCO中的「刀」和手術中的「手術刀 (scalpel)」、「電燒刀 (electrocautery blade)」外觀、用途、背景完全不同。模型必須重新學習這些新概念。
    
2. **數據稀缺且標註昂貴**：您不可能輕易獲得數十萬張帶有精確標註的手術圖片。標註工作需要醫學專家，成本極高，因此您的訓練資料集可能只有幾百到幾千張。
    
3. **物體特性複雜**：
    
    - **小而精細**：手術器械通常很細長。
        
    - **強反光**：金屬器械在內視鏡光源下會產生高光和反光。
        
    - **嚴重遮擋**：器械會被器官、血液、其他器械遮擋。
        
    - **相似度高**：不同型號的鉗子、剪刀外觀非常接近。
        
    - **背景複雜**：人體內部器官、組織、血液紋理複雜且不固定。
        
4. **對精準度和可靠性要求極高**：醫療應用中，漏檢或誤檢都可能導致嚴重後果。
    

基於以上挑戰，我們來逐一分析各個模型的適用性。

---

### 各模型在手術影片偵測專案中的優劣分析

#### 1. Faster R-CNN

- **優點**:
    
    - 架構成熟穩定，作為研究的基準線（Baseline）非常好。
        
    - 其兩階段的設計，對於區分外觀相似但細節不同的手術器械可能有潛在優勢。
        
- **缺點**:
    
    - 對於反光、遮擋等複雜情況的特徵表達能力可能不如最新的模型。
        
    - 速度慢，如果您的應用需要即時（Real-time）指導，這會是個致命傷。
        
    - 對數據量有一定要求，在小數據集上可能容易過擬合。
        
- **所需步驟**:
    
    1. 準備您的手術影片標註數據集（PASCAL VOC或COCO格式）。
        
    2. 加載一個在COCO上預訓練的Faster R-CNN模型。
        
    3. **修改預測頭**：將最後的分類層的輸出類別數，改成您的手術器械/器官類別數。
        
    4. **進行Finetune**：用您的數據集對整個網路（或主要是後段的層級）進行微調。
        
- **結論**: **可靠的備案，但並非最佳選擇**。適合當作一個基準模型來比較效果。
    

#### 2. YOLO (以YOLOv8為例)

- **優點**:
    
    - **速度最快**。如果您的專案目標是術中即時輔助，YOLO是首選。
        
    - 生態系極其完善，訓練、微調、部署的流程非常簡潔，能讓您快速看到結果。
        
    - 現代YOLO版本對小目標的偵測能力已有長足進步。
        
- **缺點**:
    
    - 在極度遮擋和物體高度重疊的場景下，其精度可能仍略遜於最頂尖的Transformer或兩階段模型。
        
    - 單階段的設計，對於分辨極其相似的器械類別，可能需要更多的數據和調校。
        
- **所需步驟**:
    
    1. 準備標註數據，並撰寫一個簡單的`.yaml`設定檔來描述您的數據集。
        
    2. 從官方加載COCO預訓練的YOLOv8權重。
        
    3. 執行一行命令即可開始Finetune。整個過程高度自動化。
        
- **結論**: **即時應用的首選**。如果專案對延遲有要求，應優先考慮YOLO。
    

#### 3. DETR

- **優點**:
    
    - 其集合預測（Set Prediction）和注意力機制，**理論上非常適合**處理手術中器械嚴重遮擋和重疊的場景。
        
    - 無需NMS，可以更優雅地處理多個物體的關係。
        
- **缺點**:
    
    - **訓練極度需要大量數據且收斂緩慢**。這在您數據有限的情況下是**最致命的缺點**。直接在小數據集上Finetune很難達到理想效果。
        
    - 對小物件的偵測是其原始設計的弱點（後續有很多變體在改進這一點）。
        
- **所需步驟**:
    
    1. 準備標註數據。
        
    2. 加載COCO預訓練模型。
        
    3. 進行Finetune，並準備好投入大量的時間和計算資源進行實驗調參。
        
- **結論**: **概念上很美好，但實踐上極具挑戰**。不建議作為首選，除非您有大量數據和時間。
    

#### 4. DINOv2 + Faster R-CNN Head

- **優點**:
    
    - **極高的數據效率**。DINOv2在海量無標籤數據上學到的視覺特徵極其強大和通用，對光照、紋理、視角變化有很強的魯棒性。這意味著它能**用更少的標註數據，達到更高的精度**。這完美地契合了您「數據稀缺昂貴」的痛點。
        
    - **精度上限高**。強大的骨幹網路是達到高精度的基石。
        
- **缺點**:
    
    - 模型體積巨大，對推理硬體要求高。
        
    - 速度受限於後接的偵測頭（Faster R-CNN），不適合即時應用。
        
- **所需步驟**:
    
    1. 準備您的（小規模）標註數據集。
        
    2. 加載預訓練好的DINOv2模型作為骨幹網路，並**凍結其大部分權重**。
        
    3. 在其後接上一個Faster R-CNN的偵測頭（僅隨機初始化預測器部分）。
        
    4. **僅訓練偵測頭**。由於DINOv2大部分參數被凍結，訓練會非常快，且不易過擬合。
        
    5. （可選）在偵測頭收斂後，可以解凍DINOv2的最後幾個Block，用非常小的學習率進行端到端的微調，以進一步提升性能。
        
- **結論**: **追求最高精度的首選，尤其適合數據有限的場景**。
    

#### 5. Grounding DINO

- **優點**:
    
    - **潛力巨大，具備研究價值**。您可以嘗試用自然語言來偵測，例如輸入「夾住膽管的鉗子」或「找到左側的穿刺器 (trocar)」。
        
    - 具備零樣本（Zero-Shot）偵測能力，可能可以找到訓練集中從未明確標註過的器械。
        
- **缺點**:
    
    - **高風險**。其零樣本能力在COCO等通用場景很強，但在高度專業化的醫療領域，對「止血鉗」和「持針鉗」這類專業術語的理解和視覺對應可能很差，需要大量專業的圖文對數據進行微調。
        
    - 模型複雜、速度慢、資源消耗大。
        
- **所需步驟**:
    
    1. 這不僅需要邊界框標註，更理想的是需要**圖像-文本對**的數據進行微調。
        
    2. 工作流程更偏向於學術研究，需要深入理解視覺-語言模型的原理。
        
- **結論**: **前沿的學術研究方向，不適合需要快速落地和穩定性的工程專案**。
    

### 總結與最終建議

|模型|推薦度|適合場景|主要優勢|主要劣勢|
|---|---|---|---|---|
|**DINOv2 + Head**|★★★★★|**追求最高精度、術後分析、數據有限**|**數據效率極高、精度上限高**|速度慢、模型大|
|**YOLOv8**|★★★★☆|**追求即時反饋、術中輔助、快速部署**|**速度極快、生態成熟、訓練簡單**|處理極端遮擋和相似物體能力稍弱|
|**Faster R-CNN**|★★★☆☆|學術對比、功能驗證的基準線|穩定、經典|速度慢、性能無驚喜|
|**Grounding DINO**|★★☆☆☆|前沿學術探索、人機交互研究|開放詞彙能力、潛力巨大|風險高、流程複雜、速度極慢|
|**DETR**|★☆☆☆☆|（不推薦用於此特定專案）|理論上適合遮擋場景|訓練極慢、極度依賴大數據|

#### **我的最終建議：**

根據您的專案情況，我會推薦以下策略：

1. **首選方案（追求精度）**: **DINOv2 + 偵測頭 (例如 Faster R-CNN 或 Mask R-CNN Head)**。
    
    - **理由**：這個方案能最大化地利用現有的SOTA技術來緩解您最核心的「數據稀缺」問題。您可以用相對較少的標註成本，達到最高的偵測精度。這對於需要發表論文或要求結果高度可靠的術後分析系統來說是最佳選擇。
        
2. **次選方案（追求速度）**: **YOLOv8**。
    
    - **理由**：如果您的系統需要在手術過程中提供即時反饋（例如高亮顯示危險區域、實時追蹤器械），那麼速度就是一切。YOLO的開發和部署流程最為順暢，能讓您在最短時間內搭建一個可用的原型系統。
        

**建議的實踐路徑：**

- **分兩步走**：可以先用YOLOv8快速搭建一個基線系統，驗證整個數據處理和應用流程。與此同時，開始進行DINOv2方案的實驗，因為它的精度潛力更大。最終根據專案的實際需求（精度 vs 速度）來決定部署哪一個模型，或者將兩者結合（例如用YOLO做快速初篩，再用高精度模型做二次確認）。



#### 各模型在手術影片偵測專案中的fine tune

```
Question: 我以為利用DINOv2作為backbone foundational model仍然需要針對backbone foundational model本身進行finetune因為DINOv2 本身是用COCO等常用datasets或常見objects訓練的. 所以確認像用DINOv2只需要用少量手術標註data訓練head model. 而用Faster RCNN, YOLO, 都是使用COCO上預訓練的整個model用多一點的手術標註data進行finetune, DETR也是使用COCO上預訓練的整個model用很多的手術標註data進行finetune, 然後對grounding DINO等於要重新準備大量專業的圖文重新訓練. 所以用DINOv2作為backbone foundational model, 無論應用在哪個專業領域都不需要針對DINOv2本身fine-tune. 以上敘述是否正確
```
您的總結和提問非常精準，這表示您已經觸及到了現代AI模型應用最核心的策略問題！您的理解**大部分是正確的**，但其中有一個最關鍵的點需要釐清，以及對於「是否需要fine-tune backbone」這個問題，存在一個更細緻的答案。

首先，讓我來修正您描述中一個最關鍵的前提：

### 關鍵釐清：DINOv2的訓練方式與眾不同

您提到「DINOv2 本身是用COCO等常用datasets或常見objects訓練的」，這一點**並不正確**，而這正是DINOv2如此強大的秘密所在。

- **Faster RCNN, YOLO, DETR的預訓練**：它們是在**有標註**的COCO數據集上進行**監督式學習**的。模型被明確告知「這是一隻貓，它的框在這裡」。因此，它們學到的是COCO那80個類別的特定特徵。
    
- **DINOv2的預訓練**：它是在一個包含**1.42億張**圖片的、**完全沒有人工標註**的龐大數據集（LVD-142M）上，進行**自監督學習**的。模型沒有被告知任何物體的名稱或位置，而是透過「讓圖片的不同視角（view）的輸出特徵盡量相似」等方式，自己學會了圖像中最本質、最通用的視覺規律。
    

**這導致了一個根本性的差異**：DINOv2學到的不是「貓的特徵」，而是「構成貓的紋理、輪廓、結構」以及「這些結構如何與背景區分」等更底層、更泛化的視覺語法。這使得它的特徵能夠輕易地遷移到任何專業領域，無論是手術器械、衛星圖像還是細胞照片。

---

### 對您的總結進行確認與微調

現在，基於以上釐清，我們來逐一確認您的敘述：

**1. 「用DINOv2只需要用少量手術標註data訓練head model。」**

- **結論：完全正確。**
    
- **解釋**：這是最常用且最高效的策略。因為DINOv2 backbone已經提供了極其豐富的特徵，你只需要訓練一個輕量的Head去學會如何「解讀」這些特徵，將其對應到你的手術器械類別上。這大大降低了對標註數據的需求。
    

**2. 「用Faster RCNN, YOLO, 都是使用COCO上預訓練的整個model用多一點的手術標註data進行finetune。」**

- **結論：基本正確。**
    
- **解釋**：「多一點的數據」是相對DINOv2方案而言的，因為它們的backbone是從有監督的COCO任務開始的，領域差異更大，需要更多的數據來「扭轉」和適應新的手術場景。
    

**3. 「DETR也是使用COCO上預訓練的整個model用很多的手術標註data進行finetune。」**

- **結論：完全正確。**
    
- **解釋**：您精準地捕捉到了DETR的特性。它的架構決定了它需要更長的訓練週期和更多的數據才能良好地收斂，即使是在Finetune階段。
    

**4. 「對grounding DINO等於要重新準備大量專業的圖文重新訓練。」**

- **結論：基本正確。**
    
- **解釋**：為了在專業領域達到最佳效果，是的。你需要準備高品質的「手術圖像」-「器械/操作的文字描述」對來進行微調，這比單純標註邊界框的成本更高。
    

---

### 核心問題：DINOv2 Backbone到底需不需要Finetune？

> **「所以用DINOv2作為backbone foundational model, 無論應用在哪個專業領域都不需要針對DINOv2本身fine-tune.」**

這是一個非常好的問題，答案是：**這句話不完全正確，它描述的是一種最常用、最安全的策略，但並非唯一或最佳策略。**

實際上，我們有三種策略可以選擇，這是一個權衡（Trade-off）：

#### 策略A：完全凍結Backbone (Freeze Backbone)

- **做法**：完全不更新DINOv2的任何權重，只訓練新加上去的Head。
    
- **優點**：訓練速度極快、需要計算資源少、**絕對不會過擬合**。在標註數據**極度稀少**（例如只有幾百張）時，這是最安全、最推薦的做法。
    
- **缺點**：模型的性能上限受限於Head的學習能力，Backbone無法針對手術領域的細微特徵（如器械獨特的金屬反光）做進一步的適應。
    

#### 策略B：端到端微調 (End-to-End Finetune)

- **做法**：將DINOv2和Head的所有權重全部解凍，用手術數據從頭到尾進行訓練。
    
- **優點**：理論上可以達到最高的性能，因為整個模型都為新任務進行了優化。
    
- **缺點**：**風險極高！** 在數據量不足時，極易發生「災難性遺忘」（Catastrophic Forgetting），即強大的DINOv2為了擬合你那點可憐的數據，把自己在1.4億張圖上學到的寶貴通用知識給忘掉了，導致性能不升反降。**除非你有海量的手術標註數據，否則絕對不要這麼做。**
    

#### 策略C：部分解凍與微小學習率微調 (Gradual Unfreezing / Small LR Finetune) - **最佳實踐**

- **做法**：
    
    1. **第一階段**：採用策略A，先凍結DINOv2，只訓練Head，讓Head充分學習。
        
    2. **第二階段**：在Head收斂後，**解凍DINOv2的最後幾個（例如2-3個）Block**，並使用一個**非常小**的學習率（例如比訓練Head時小10-100倍）對整個模型進行端到端的微調。
        
- **優點**：**這是效果與風險的最佳平衡點**。它允許模型在不破壞通用知識的基礎上，微調最高層的特徵以更好地適應手術器械的獨特細節。這通常會比單純的策略A帶來幾個百分點的性能提升。
    
- **缺點**：比策略A需要更多的訓練時間和調參技巧。
    

### 總結

|模型|Finetune策略總結|數據需求（Finetune階段）|
|---|---|---|
|**DINOv2 + Head**|**首選**：凍結Backbone，訓練Head。<br>**最佳**：先訓Head，再解凍尾部用小學習率微調。|**低 (Low)**|
|**YOLO / Faster RCNN**|Finetune整個模型，或Finetune尾部Layer。|**中 (Medium)**|
|**DETR**|Finetune整個模型，需要耐心和技巧。|**高 (High)**|
|**Grounding DINO**|需要用專業領域的圖文對數據進行微調。|**高 (High) + 數據類型特殊**|

所以，您的理解已經非常接近事實了。最關鍵的修正就是：**DINOv2之所以強大，是因為它在無標註數據上的自監督學習，而非有標註的COCO。** 而對於是否微調其Backbone，**「可以不調，且效果已很好；但用對方法去調，效果會更好」**。




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