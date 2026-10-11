
|                    |     |
| ------------------ | --- |
| [[#### CVAI技術面試2]] |     |
|                    |     |
|                    |     |

#### CVAI技術面試2
```
如果在技術面試上被問到下列問題要如何完整深入回答:
YOLO 和 DETR 差異在哪裡？(以及解釋甚麼是Detection Architecture、Speed、Training、Matching)
CNN 與 Vision Transformer 如何選擇？(以及解釋甚麼是Inductive Bias、Data Requirement、Compute)
小型目標只佔影像 1%，如何提高辨識率？(以及解釋甚麼是High-resolution Crops、Multi-scale、Sampling)
Class Imbalance 如何處理？(以及解釋甚麼是Sampling、Focal Loss、Dice Loss、Threshold)
Training Accuracy 很高但 Test Accuracy 很低，怎麼辦？(以及解釋甚麼是Overfitting、Data Leakage、Distribution Shift)
如何發現 Model Learning 了錯誤特徵？(以及解釋甚麼是Error Analysis、Saliency、Counterfactual Tests)
如何比較兩個 Model 是否真的有提升？(以及解釋甚麼是Independent Test Set、Statistical Significance、Slice Analysis)
Model Confidence 很高卻經常判錯，如何改善？(以及解釋甚麼是Calibration、OOD Detection、Uncertainty)
```

# Senior / Staff Computer Vision Engineer 技術面試：8 大核心問題完整解析

這八題涵蓋了 Computer Vision Engineer 從模型設計、訓練、錯誤診斷、統計驗證到 Production Deployment 的核心能力。

在美國 Senior／Staff AI Engineer 面試中，面試官通常不只想聽到某個模型或演算法的定義，而是想確認你能不能：

1. 理解底層原理：為什麼某個架構、Loss Function 或 Training Strategy 有效？
    
2. 做出工程選擇：什麼條件選 YOLO、DETR、CNN、ViT？
    
3. 診斷真實問題：遇到 Small Objects、Overfitting、Data Leakage、Class Imbalance 如何系統性調查？
    
4. 用科學方法證明改善：如何避免因為 Test Set、隨機性或 Data Distribution 而得出錯誤結論？
    
5. 負責 Production 成效：模型不只要準確，還要考慮 Latency、Uncertainty、可靠性與 Failure Modes。
    

以下每一題都會包含面試回答、基礎概念、數學原理、實際解法，以及 Senior／Staff 層級應展現的思考。

# Q1. YOLO 和 DETR 差異在哪裡？

核心：Detection Architecture、Speed、Training、Matching

## 1.1 面試時可以這樣回答

> YOLO and DETR represent two different object detection paradigms.
> 
> YOLO is traditionally a dense prediction detector. It predicts bounding boxes and class scores across spatial feature maps, using label assignment during training and typically NMS during inference.
> 
> DETR formulates object detection as a set prediction problem. It uses object queries and transformer-based feature interaction, with one-to-one bipartite matching between predictions and ground-truth objects.
> 
> Traditional YOLO architectures are often computationally efficient and well suited for low-latency inference. The original DETR has more global reasoning capability but suffers from slower training convergence and higher computational costs.
> 
> However, modern architectures such as RT-DETR and end-to-end YOLO have narrowed these differences.
> 
> I would evaluate both models using the same dataset, input resolution, hardware, precision, and latency requirements rather than assuming either is universally better.

這個回答的優點在於：你不是只說「YOLO 快、DETR 準」，而是解釋它們的 Detection Formulation、Training Assignment、Inference Pipeline 以及實際選型標準。

## 1.2 Detection Architecture 是什麼？

Object Detection 的目標是找出影像中的物件位置與類別。

例如：

![YOLO: The AI Model Powering Real-Time Object Detection | by Wicar Akhtar | Medium](https://images.openai.com/static-rsc-4/Fp_LqGz1v18i76gYTQHjqMDn8Hz010_vhtFlXLCvQfkIgpq12tCJ6ny01H837Hmd7Sk4f3rM2AVDclpMTF8QMlewTj0OX7g5L02jr8F5fsDCDDdD0sAOcpaDwfw_nUtD7iB5DT-BKAWg3qZh2XtIIMixvICbuXgBM-uFULNgsFs?purpose=inline)

![transformer系列——detr详解_detr 解码层输入-CSDN博客](https://images.openai.com/static-rsc-4/ZU4CiFTHKf-oShOI7nO7sdqWQa37U5mSAA8kEWNvi5SKU4U358tPj4N0dJxUhu019_Eanq9q_2BA_OdEr2eonBwfMxDvQ__NOvVeikLrAhs9BkD-C1ttcir5AbyD1gn1_1uIhe-US4dOhpOBNfdPGMuBAmD697BYRaEnRQu1p6M?purpose=inline)

![GitHub - guojin-yan/RT-DETR-OpenVINO: Deploy RT-DETR model based on OpenVINO.](https://images.openai.com/static-rsc-4/axvPexHCF7Q5lwNc6MA-9rlKOVKBztduR9ExBva_nvgA7giYq_8G0NXrSqhhb1GyE_JB2BLBfIdhsI7skqTOr-65g2f-i9xmGk17jG6jEiSJWG0aaBRXOdWPsQPyFBaU8Ti1Ruj-ZGvc5Z7-npaPKLb51lYY56X4M3XX7Xbkn9s?purpose=inline)

![Object Counting using Roboflow RF-DETR](https://images.openai.com/static-rsc-4/qntsTLTJ8HwwQf2u4JgytIvK8rF3rxae7w6Co9sE6tCfVtEkp3WNBg4Tk_C06Bys-8W-_T_FsSshhwjUFlZ_3fuUleBFCHuaMLPSRDa2xAZEOI3KXUK4YqLGM55hHzdZbAocvzerkcMSBfSYyrSH_wk41JnQDaPaP-ExaxLsQbg?purpose=inline)

![DFS-DETR: Detailed-Feature-Sensitive Detector for Small Object Detection in Aerial Images Using Transformer](https://images.openai.com/static-rsc-4/XWiwXhkzPy4iByurGtqwsChveUSiox05XdWjEqtD2nvH8VgA1SLN1vCnxPlJI8-CEUiF3zEnVsbdA-aUUtIp8dbS32SHJkIhgOKmp9kvESmK9k7b0HRut-Iye5KYqfKdBPGBiDEkzENRXyi-QgzQLqA45jWIGVBZjEoN3jnt6m4?purpose=inline)

模型輸入影像 \(I\)，輸出一組物件：

\[ D=\{(b_i,c_i,s_i)\}_{i=1}^{N} \]

其中：

- \(b_i=(x_i,y_i,w_i,h_i)\)：Bounding Box
    
- \(c_i\)：物件類別
    
- \(s_i\)：Confidence Score
    
- \(N\)：偵測到的物件數量
    

真正的架構差異，在於模型如何產生、訓練與篩選這些 Bounding Boxes。

## 1.3 YOLO Architecture

以典型 CNN-based YOLO 為例：

Input Image

例如 640 × 640 × 3

Backbone

CNN Feature Extraction

Neck

FPN / PAN-like Multi-scale Fusion

P3 / stride 8

P4 / stride 16

P5 / stride 32

Detection Head

Class Scores + Bounding Boxes

Post-processing

典型 YOLO：Confidence Filtering + NMS

Backbone

負責抽取 Feature，例如：

- Edge、Texture、Corner
    
- Object Parts
    
- Semantic Features
    

Neck

整合不同解析度的 Feature Maps。例如：

|Feature|640×640 輸入時|用途|
|---|---|---|
|P3|80×80|較小物件|
|P4|40×40|中等物件|
|P5|20×20|較大物件|

P3 的空間解析度比較高，因此通常更有利於保留小物件的細節。

Detection Head

對多個空間位置輸出 Bounding Boxes 與類別分數。

不同 YOLO 版本可能採用 Anchor-based 或 Anchor-free，且 Label Assignment 方法不完全相同。

### NMS（Non-Maximum Suppression）

如果一個物件產生三個重疊 Bounding Boxes：

|Box|Confidence|IoU with Box A|
|---|---|---|
|A|0.96|1.00|
|B|0.91|0.87|
|C|0.83|0.79|

在傳統 NMS 中，通常先保留 A，再根據 IoU Threshold 移除高度重疊且屬於同類別的 B、C。

NMS 的問題是它可能：

- 在密集物件場景刪掉真正相鄰的物件。
    
- 帶來額外 Post-processing。
    
- 讓效能受到候選框數量與實作方式影響。
    

但要注意，2026 年不能再說所有 YOLO 都需要 NMS。例如 YOLO26 同時訓練 one-to-many 與 one-to-one detection heads；其 one-to-one 路徑可以不經 NMS 直接推論，雖然官方預設路徑仍使用 NMS。

![](https://www.google.com/s2/favicons?domain=https://docs.ultralytics.com&sz=32)

Ultralytics

+1

## 1.4 DETR Architecture

DETR = DEtection TRansformer。

它把 Detection 視為：

Set Prediction（集合預測）。

Input Image

Image Backbone

例如 ResNet，抽取 Image Features

Transformer Encoder

Contextual Feature Interaction

Object Queries

Transformer Decoder

Prediction Set

Class + Box + No-object

Training：Hungarian Matching + Set Loss

核心概念包括：

### A. Object Queries

假設模型有 300 個 Object Queries。

這不代表影像有 300 個物件，而是模型有 300 個可用的 Prediction Slots。

每一個 Query 嘗試找出一個物件。

例如：

- Query 1 → Watch Dial
    
- Query 2 → Watch Hand
    
- Query 3 → Crown
    
- Query 4 → No Object
    
- Query 5 → No Object
    

原始 DETR 使用學習得到的 Query Embeddings。不同後續架構可能使用 Feature-based、Anchor-based 或其他 Query Initialization。

### B. Self-Attention / Cross-Attention

Encoder 的 Self-Attention 可以建立影像不同位置之間的資訊關聯。

Decoder 利用 Queries 與 Image Features 互動，預測物件。

需要注意，DETR 並不表示所有層都是純 Transformer：原始 DETR 的 Image Backbone 本身就是 CNN。

### C. Hungarian Matching

這是兩者最重要的差異之一。

假設 Ground Truth 有三個物件：

\[ G=\{g_1,g_2,g_3\} \]

DETR 產生五個候選預測：

\[ P=\{p_1,p_2,p_3,p_4,p_5\} \]

Hungarian Algorithm 尋找最小成本的一對一配對：

\[ \hat{\sigma} = \arg\min_{\sigma} \sum_{i=1}^{M} C(g_i,p_{\sigma(i)}) \]

其中 \(C\) 是 Matching Cost。

常見成本包括：

\[ C= \lambda_{cls}C_{cls} +\lambda_{L1}C_{L1} +\lambda_{IoU}C_{IoU} \]

例如：

- Classification Cost：類別預測是否正確？
    
- L1 Box Cost：Box Coordinates 距離有多遠？
    
- GIoU Cost：Bounding Box 的幾何重疊程度如何？
    

得到一對一配對後：

\[ g_1\leftrightarrow p_3 \]

\[ g_2\leftrightarrow p_1 \]

\[ g_3\leftrightarrow p_5 \]

沒有匹配的 \(p_2,p_4\) 被訓練成 No-object。

如此一來，每個 Ground Truth 主要對應一個預測，不必依靠傳統 NMS 去移除同一物件的多個候選框。

原始 DETR 的這個核心設計來自其 Bipartite Matching 與 Set-based Loss。

![](https://www.google.com/s2/favicons?domain=https://arxiv.org&sz=32)

arXiv

+1

## 1.5 YOLO vs DETR 的完整比較

|比較|傳統 YOLO|原始 DETR|
|---|---|---|
|Detection|Dense spatial prediction|Set prediction|
|Backbone|通常 CNN|CNN + Transformer|
|Global Context|透過 receptive field / feature fusion|Explicit attention|
|Label Matching|One-to-many / task-specific assignment|One-to-one Hungarian matching|
|Object Queries|通常不使用 DETR-style queries|使用|
|NMS|通常需要|不需要|
|Training Convergence|通常較容易快速收斂|原始版本較慢|
|Small Object|可利用高解析度 feature maps|原始版本相對較弱|
|Inference|通常高效率|原始版本運算成本較高|
|Deployment|Edge 生態成熟|依架構與硬體而定|

這是傳統架構比較，不是所有現代版本的絕對規則。

### 現代 DETR：RT-DETR

RT-DETR 透過 Efficient Hybrid Encoder 與改良 Query Selection，讓 DETR 架構也能進行 Real-time Detection。

因此「DETR 一定比較慢」並不正確。RT-DETR 原始論文在指定 T4 GPU 測試條件下，報告 RT-DETR-R50 達到 53.1 AP、108 FPS；這是該論文環境的數據，不能直接推論到不同硬體與模型版本。

![](https://www.google.com/s2/favicons?domain=https://arxiv.org&sz=32)

arXiv

+1

## 1.6 Speed 要怎麼公平比較？

不要只比較論文上的 FPS。

需要確認：

\[ FPS=\frac{1}{T_{\text{per image}}} \]

但 Production 更重要的是 End-to-end Latency：

\[ T_{total} = T_{decode} +T_{preprocess} +T_{inference} +T_{postprocess} +T_{transfer} \]

例如，假設：

|階段|Model A|Model B|
|---|---|---|
|Preprocessing|3 ms|3 ms|
|Inference|12 ms|18 ms|
|Postprocessing|5 ms|1 ms|
|Data Transfer|2 ms|2 ms|
|Total|22 ms|24 ms|

即使 B 幾乎沒有 Postprocessing，也不一定比 A 快。

真正應報告的是同條件下的 p50、p95、p99 Latency、Throughput、Peak GPU Memory 和 Detection Accuracy。

## 1.7 Senior 面試官追問：你會如何選擇？

我會依照場景決定：

- 低延遲 Edge Inference：先測試小型 YOLO，包含可用的 NMS-free 版本。
    
- 密集或高度重疊的物件：考慮 DETR-family，但要驗證實際的密集物件 Recall。
    
- 高精度、Latency 較寬裕：比較 RT-DETR、DINO 類模型以及較大型的 YOLO。
    
- 高解析度小物件：優先解決影像縮放與 Detail Loss，而不是立刻更換 Detection Architecture。
    
- Production：同一影像資料、相同解析度、硬體、推論精度與評估方式下進行 Benchmark。
    

Senior-Level 重點：架構選擇是一個受 Constraints 影響的最佳化問題，而不是模型排行榜的比較。

# Q2. CNN 與 Vision Transformer 如何選擇？

核心：Inductive Bias、Data Requirement、Compute

## 2.1 面試回答

> CNNs and Vision Transformers differ primarily in their inductive biases and mechanisms for modeling spatial relationships.
> 
> CNNs encode locality and translation equivariance through convolutional kernels and weight sharing. These biases often make them data-efficient and computationally practical, especially for smaller datasets.
> 
> Vision Transformers divide images into patches and use self-attention to model relationships across tokens. They provide flexible global feature interactions, but the original ViT can have higher data and computational requirements when trained from scratch.
> 
> With large-scale pretraining and modern training strategies, Transformers can work extremely well even when the downstream labeled dataset is small.
> 
> My choice would depend on data scale, pretrained model availability, image resolution, local versus global features, memory constraints, and deployment latency.

## 2.2 CNN：Convolutional Neural Network

![What Is Deep Learning? How It Works and Why It Matters](https://images.openai.com/static-rsc-4/9iBm1SRuj3MIsanIWtVeSKMavsw4Gyuw0Av2o698RTvyAZ4M0aid6qe2eovz3Ag00TR2vAOGOIWqcQhBKRo0TBI27R8whHUzmmPQY_cj6W5NbtQR3gsFtsenzoBUOgI-rnPQZmyEz73z_-sPlait3eNQ2Bx3l53ZL8QRBqP1_N8?purpose=inline)

![Vision Transformers(ViT): Transformer for Image Recognition at Scale | by VectorWorks Academy | Medium](https://images.openai.com/static-rsc-4/kuAPFizqnP0V2mCuj_lwjinYqB9O3a77KMg4RLYX6lXevKNVze76EQeruGZrIOVfa-wKQWx0RLiZTHkh-zdcdlwO-OjpAA29GKIoqrRSylftp0jToZ_1otlHunqtoCz2smNWU5Rimpx7-cPyU1jCWxGPE8b_j2CINSwsNVjKeZI?purpose=inline)

CNN 使用 Convolution Kernel 在影像中滑動，尋找局部特徵。

例如，一個 3×3 Kernel：

\[ K= \begin{bmatrix} -1&0&1\\ -2&0&2\\ -1&0&1 \end{bmatrix} \]

可以用來強調特定方向的影像邊緣。

CNN 的基本運算是：

\[ Y(i,j)= \sum_{u,v}K(u,v)X(i+u,j+v) \]

這裡用單一通道的簡化形式說明。

在 Deep CNN 中：

- 淺層抽取 Edges、Corners。
    
- 中層抽取 Textures、Patterns。
    
- 深層抽取 Object Parts、Semantics。
    

例如辨識 Rolex 錶面：

淺層可能學到文字邊緣，中層學到 Hour Marker 或 Logo 的形狀，深層組合出錶面特徵。

## 2.3 什麼是 Inductive Bias？

Inductive Bias 是模型在學習前，就由架構預先帶入的假設。

CNN 有幾個重要的 Bias：

1. Locality（局部性）

相鄰 Pixel 通常具有較高相關性。

因此先從局部 3×3、5×5 區域抽取特徵是合理的。

2. Weight Sharing（權重共享）

同一組 Convolution Kernel 在不同影像位置重複使用。

例如學會辨識一個垂直邊緣，不需要在影像每個位置重新學習另一組參數。

3. Translation Equivariance（平移等變性）

理想的 Convolution 滿足：

\[ f(T_{\Delta}x)=T_{\Delta}f(x) \]

如果物件平移，Feature Map 的回應也相應平移。

但實際 CNN 的 Padding、Stride、Pooling 等操作可能破壞嚴格的等變性。

### 為什麼這很重要？

假設只有 500 張訓練影像。

CNN 不需要完全從資料中自行發現：

- 邊緣有意義。
    
- 附近的 Pixel 通常相關。
    
- 同一個 Pattern 可以出現在不同位置。
    

這些假設已經部分寫在架構裡。

因此在某些 Small-data Settings，CNN 比從零開始訓練 ViT 更容易達到良好泛化。

## 2.4 Vision Transformer（ViT）

原始 ViT 把影像切成 Patches。

假設：

\[ Image=224\times224 \]

\[ Patch=16\times16 \]

Patch 數量：

\[ N=\frac{224}{16}\times\frac{224}{16}=196 \]

每個 Patch Flatten 後，投影成一個 Token Embedding。

然後使用 Transformer Encoder 進行 Self-Attention。

### Self-Attention

核心公式：

\[ Attention(Q,K,V)= Softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V \]

其中：

- Q：Query
    
- K：Key
    
- V：Value
    
- \(d_k\)：Key Dimension
    

每個 Token 可以根據學習到的 Attention Weights，整合其他 Token 的資訊。

例如辨識一個錶面 Logo，模型可能同時利用錶面文字、Hour Markers 與其他位置的資訊來判斷。

CNN 也能藉由深層架構取得全域資訊；差別不是 CNN 不能看全域，而是建立長距離互動的機制不同。

## 2.5 CNN vs ViT 的 Compute 差異

CNN 某一層的粗略計算量：

\[ O(HWk^2C_{in}C_{out}) \]

其中：

- H、W：Feature Map Size
    
- k：Kernel Size
    
- \(C_{in},C_{out}\)：Channels
    

標準 Global Self-Attention 的主要計算量：

\[ O(N^2d) \]

其中：

- N：Token 數量
    
- d：Token Dimension
    

Self-Attention 的 Attention Matrix 記憶體需求也是約：

\[ O(N^2) \]

不過完整 Transformer Block 還包含 Linear Projections、MLP 等，因此不能只用 \(N^2d\) 代表整個 ViT 的總計算量。

### 高解析度為什麼是問題？

互動計算：ViT Patch 數與 Attention Matrix

影像邊長

1024 × 1024

Patch Size

8×8

16×16

32×32

Token 數量 N

# 4,096

Attention Matrix 元素 N²

# 16,777,216

假設正方形影像並在邊界 Padding 到 Patch 整數倍。N² 為單一 Attention Head 的矩陣元素數，非模型總參數、FLOPs 或實際 GPU 記憶體使用量。

這解釋了為什麼高解析度 Vision Transformer 往往需要：

- Window Attention，例如 Swin Transformer 的設計。
    
- Hierarchical Features。
    
- Token Reduction / Pooling。
    
- 更高效的 Attention Implementation。
    
- ROI Cropping 或 Multi-stage Processing。
    

但不能籠統地說 ViT 一定比 CNN 慢：實際速度取決於模型大小、Token Count、Kernel Implementation、Batch Size 和硬體。

## 2.6 Data Requirement：ViT 一定需要大量 Data 嗎？

要區分兩種情況。

From Scratch Training

原始 ViT 的 Inductive Bias 較少，可能需要大量資料、合適的 Regularization 與訓練策略。

Pretrained Fine-tuning

使用大型預訓練 ViT 時，下游可以只有數百或數千張標註影像。

早期 ViT 研究已顯示大規模 Pretraining 的價值；後續 DeiT 則證明透過適當的 Training Recipe 與 Distillation，能顯著改善 Data Efficiency。

![](https://www.google.com/s2/favicons?domain=https://arxiv.org&sz=32)

arXiv

+1

所以「只有 500 張影像就不能用 ViT」是不正確的。

真正的問題是你有沒有合適的 Pretrained Representation，以及它能否 Transfer 到你的影像領域。

## 2.7 如何選 CNN 或 ViT？

|情境|優先考慮|原因|
|---|---|---|
|小資料集、從零訓練|CNN|Local Bias 有利於學習|
|已有優質 ViT Pretraining|CNN、ViT 都測|Transfer Learning 可降低資料需求|
|Edge CPU / 小型 GPU|Efficient CNN|通常較容易部署|
|高解析度 Local Texture|CNN / Hybrid|局部細節處理高效|
|大規模 Pretraining|ViT / Hybrid|擴展性與表示學習潛力|
|需要複雜 Context|ViT / Hybrid|Flexible Token Interactions|
|Industrial Defect Detection|CNN、ViT、Anomaly Models|取決於缺陷尺度與資料分布|

## 2.8 面試官追問：如果是錶面細微瑕疵辨識呢？

我的回答會是：

「我不會先選最流行的 ViT。我會先確認 Defect 的 Physical Scale 與 Pixel Scale。如果瑕疵只有 5–10 Pixels，使用 16×16 Patch 不代表細節一定會消失，但若前處理已經把瑕疵縮小，模型無論採用哪個架構都可能失去辨識訊號。

我會先做 High-resolution ROI 與 CNN Baseline，然後比較 Pretrained ViT 或 Hybrid Architecture。評估時除了 mAP 或 F1，也會追蹤非常小的缺陷 Recall、False Reject Rate 和實際 Inference Latency。」

Senior-Level 重點：不要把 Model Architecture 的優劣與 Input Information Quality 混為一談。

# Q3. 小型目標只佔影像 1%，如何提高辨識率？

核心：High-resolution Crops、Multi-scale、Sampling

這是 Industrial Computer Vision、Medical Imaging、Semiconductor Inspection 與 Precision Inspection 常見的面試題。

## 3.1 面試回答

> I would first determine whether the main bottleneck is insufficient image information, inadequate feature resolution, or insufficient positive training examples.
> 
> For high-resolution images, resizing the entire image to a small input can destroy the details needed to identify small objects.
> 
> I would evaluate high-resolution cropping or tiled inference, multi-scale feature extraction, and object-centered sampling. I would also verify the quality of annotations and the distribution of object sizes.
> 
> I would measure improvements using small-object recall and AP, along with false positives, localization quality, and end-to-end latency.
> 
> In production, I may choose a coarse-to-fine pipeline that detects candidate regions first and then performs high-resolution analysis on those regions.

## 3.2 為什麼小型目標難辨識？

假設原始影像為：

\[ 4096\times4096 \]

物件只佔整張影像的 1% 面積，而且形狀近似正方形。

物件的寬度約為：

\[ 4096\times\sqrt{0.01}=409.6 \]

假設整張影像 Resize 到：

\[ 640\times640 \]

那麼物件的寬度變成：

\[ 640\times0.1=64\ pixels \]

整張影像縮小時，物件細節也跟著縮小

原始 4096 × 4096

目標寬約 410 pixels

縮小到 640 × 640

目標寬約 64 pixels

圖中以同樣相對尺寸表示縮放前後；真正改變的是可用的 Pixel 數量，而不是物件在圖中的相對比例。

64×64 pixels 不一定算極小物件。真正需要注意的是：辨識所依賴的局部特徵可能只有數個 Pixels。

例如某個 Logo 裡的一條細線，在原始影像只有 5 Pixels 寬，整張影像縮小後可能不到 1 Pixel。

這時即使換成更大的神經網路，也無法可靠地恢復已經遺失的資訊。

還有 Feature Map Downsampling 的問題：

若一個 64×64 Pixel 物件進入 stride 32 的 Feature Map：

\[ 64/32=2 \]

它在該 Feature Map 上的範圍大約只有 2×2 個空間位置。

因此要同時考慮 Image Resolution 與 Feature Resolution。

## 3.3 方法一：High-resolution Crops / Tiled Inference

不要一次把整張 4096×4096 影像縮成 640×640，而是把它切成多個高解析度 Tiles。

4096 × 4096 Full Image

Tile 1

Tile 2

Tile 3

Tile 4

Tile 5

Tile 6

Tile 7

Tile 8

Tile 9

Detect in Each High-resolution Tile

Map Coordinates + Merge Detections

例如使用 1024×1024 Tiles，並且設定適當的 Overlap。

好處：

- 避免縮小整張影像。
    
- 保留局部紋理與小型文字。
    
- 能利用既有的 YOLO 或 DETR 模型。
    
- 可以平行處理多個 Tiles。
    

缺點：

- Tile 數量增加，Inference Latency 也可能增加。
    
- 物件可能跨越 Tile Boundary。
    
- 相鄰 Tiles 可能重複偵測同一個物件。
    
- Tile 太小可能失去 Global Context。
    

因此通常使用 Overlapping Tiles，最後將 Detection Coordinates 轉回原圖並合併重複結果。

這類技術已有 SAHI（Slicing Aided Hyper Inference）等實際方法支持。

![](https://www.google.com/s2/favicons?domain=https://doi.org&sz=32)

DOI

+1

## 3.4 方法二：Multi-scale Feature Extraction

FPN（Feature Pyramid Network）的概念是結合不同空間尺度的特徵。

例如：

\[ P_2,\ P_3,\ P_4,\ P_5 \]

其中：

- P2：stride 4，更細的局部特徵。
    
- P3：stride 8。
    
- P4：stride 16。
    
- P5：stride 32。
    

對小型物件，保留 P2／P3 Feature Maps 可能有幫助。

但增加 High-resolution Feature Maps 會提高 GPU Memory 與計算成本。

另一種方法是 Multi-scale Training：

\[ I_{train}\in\{512,640,768,1024\} \]

使模型在不同影像尺寸下學習。

需要區分：

- Multi-scale Training：增加尺度泛化能力。
    
- Multi-scale Feature Fusion：整合多層特徵。
    
- Multi-scale Inference：同一影像用不同尺寸推論並融合結果。
    

這三者不是同一件事，也不一定都值得同時採用。

## 3.5 方法三：Sampling Strategy

Sampling 直接影響模型看過多少有用的正樣本。

假設 10,000 個隨機 Crops 中，只有 300 個包含你想辨識的瑕疵。

那大量 Training Compute 可能都花在沒有瑕疵的背景上。

可以使用：

Positive-centered Cropping

以 Ground Truth Bounding Box 為中心抽取 Crop。

Hard Negative Mining

特別選取容易與正樣本混淆的負樣本，例如正常 Logo 與偽造 Logo 之間非常相似的區域。

Scale-aware Sampling

控制小、中、大物件在 Training Batch 中出現的比例。

但不能只訓練 Positive Crops。否則模型可能學到「每個 Crop 裡一定有物件」，導致 False Positive 大增。

也不能把 Negative Crops 全部丟掉。

## 3.6 方法四：Coarse-to-fine Architecture

在精密檢測中，這常比把整張影像直接丟進大型模型更實際。

High-resolution Image

Stage 1: Coarse Detector

Find candidate ROIs

Stage 2: High-resolution ROI Extractor

Preserve original pixel details

Stage 3: Fine-grained Classifier / Segmenter

Final Prediction + Confidence

例如：

Stage 1 先找到錶面 Logo。

Stage 2 從原始影像擷取 Logo 的高解析度 ROI。

Stage 3 再判斷 Logo 是否存在字體、形狀或印刷異常。

但有一個關鍵 Failure Mode：

如果 Stage 1 漏掉了物件，Stage 2 永遠沒有機會辨識它。

若 Stage 1 Recall 為 95%，而 Stage 2 在 Stage 1 成功的條件下 Recall 為 98%，理想化的總 Recall 約為：

\[ 0.95\times0.98=93.1\% \]

所以 Stage 1 必須特別優化 High Recall，而不能只看 Precision。

## 3.7 如何評估？

除了 mAP，還要分尺寸評估。

例如：

|Metric|Baseline|New Model|
|---|---|---|
|AP small|0.42|0.61|
|AP medium|0.78|0.80|
|Small-object Recall|0.55|0.82|
|False Positives / image|0.3|0.5|
|p95 Latency|25 ms|70 ms|

以上均是假設性面試數據，不代表實際測試結果。

這個例子顯示小物件 Recall 改善，但 Latency 與 False Positives 也增加。

是否值得部署，必須考慮誤判的成本。

另外，COCO 標準的 Small AP 是依絕對 Pixel Area 定義，不等於「小於整張影像 1% 面積」；實際專案最好同時用 Pixel Size、Relative Area 與物理尺寸定義自己的 Evaluation Slices。

Senior-Level 重點：先找出資訊在 Pipeline 哪一層遺失，再決定增加解析度、改 Feature Pyramid，或調整 Sampling。

# Q4. Class Imbalance 如何處理？

核心：Sampling、Focal Loss、Dice Loss、Threshold

## 4.1 面試回答

> I would first identify the type of class imbalance: dataset-level imbalance, foreground-background imbalance, or pixel-level imbalance.
> 
> I would avoid relying on accuracy alone because a model may achieve very high accuracy simply by predicting the majority class.
> 
> Depending on the task, I would evaluate balanced sampling, class-weighted cross-entropy, focal loss, or overlap-based losses such as Dice Loss.
> 
> I would also calibrate or select the decision threshold on a validation set according to the precision-recall trade-off and business cost.
> 
> Most importantly, I would evaluate performance on the original production distribution rather than only on a balanced training dataset.

## 4.2 Class Imbalance 是什麼？

假設一個 Defect Detection Dataset：

|Class|Images|比例|
|---|---|---|
|Normal|9,900|99%|
|Defective|100|1%|

如果模型永遠預測 Normal：

\[ Accuracy=99\% \]

可是對 Defective：

\[ Recall=0 \]

也就是一個表面 Accuracy 非常高、但完全無法辨識缺陷的模型。

因此應該使用：

\[ Precision=\frac{TP}{TP+FP} \]

\[ Recall=\frac{TP}{TP+FN} \]

\[ F1=\frac{2PR}{P+R} \]

其中：

- TP：真正例。
    
- FP：假正例。
    
- FN：假負例。
    

在極度 Imbalanced 的問題中，PR-AUC / Average Precision 往往比 Accuracy 更有診斷價值。

## 4.3 方法一：Sampling

### Oversampling

增加 Minority Class 的出現頻率。

例如每個 Mini-batch：

- 50% Normal。
    
- 50% Defective。
    

優點是增加少數類別的訓練訊號。

缺點是如果不斷重複相同少數樣本，模型可能 Overfit。

### Undersampling

減少 Majority Class 的訓練樣本。

缺點是可能丟失有價值的正常樣本多樣性。

### Hard Example Mining

根據模型容易出錯的樣本選取更有資訊量的 Training Cases。

例如正常零件中，有些看起來非常像缺陷；這些 Hard Negatives 對降低 False Positives 很重要。

### 關鍵問題

Sampling 會改變 Training Class Prior，但不代表 Production Class Prior 也改變。

因此：

Training 可以重新取樣，Validation / Test 仍應保留實際部署比例。

不然你會在 Test 時看到不真實的 Precision、False Alarm Rate，甚至錯誤的 Probability Interpretation。

## 4.4 方法二：Weighted Cross Entropy

一般 Binary Cross Entropy：

\[ L_{BCE} = -[y\log p+(1-y)\log(1-p)] \]

可以加入 Class Weights：

\[ L_{WBCE} = -[w_+y\log p+w_-(1-y)\log(1-p)] \]

如果 Positive Class 非常少，可以提高 \(w_+\)。

這會讓模型更重視漏判 Positive 所產生的損失。

但是 Weight 太高可能造成大量 False Positives。

因此 Class Weight 不是越大越好。

## 4.5 方法三：Focal Loss

Focal Loss 是非常重要的面試考點。

普通 Cross Entropy 對簡單樣本仍會計算 Loss。

如果存在數萬個非常簡單的 Background Negatives，累積起來仍可能主導訓練。

Focal Loss：

\[ FL(p_t)= -\alpha_t(1-p_t)^\gamma\log(p_t) \]

其中：

- \(p_t\)：模型對正確類別的預測機率。
    
- \(\alpha_t\)：類別平衡權重。
    
- \(\gamma\)：Hard-example Focusing Parameter。
    

當 \(p_t\) 很高，代表模型已經容易辨識該樣本：

\[ (1-p_t)^\gamma\rightarrow0 \]

因此它的 Loss 會被大幅降低。

### 具體例子

假設：

\[ \gamma=2 \]

忽略 \(\alpha_t\) 時，調整係數為：

Easy example (pₜ = 0.9)

0.01

Uncertain example (pₜ = 0.5)

0.25

Hard example (pₜ = 0.1)

0.81

顯示的是 Focal Loss 中的調整係數，而不是完整 Loss 數值。

因此 Easy Examples 受到更強的 Down-weighting。

Focal Loss 原本就是為了解決 Dense Object Detection 中大量 Foreground／Background 不平衡的問題而提出。

![](https://www.google.com/s2/favicons?domain=https://openaccess.thecvf.com&sz=32)

Open Access CVF

+1

但請注意：

Focal Loss 並不能保證所有被判錯的樣本都是真正有用的 Hard Examples。Label Noise 也可能被當成 Hard Examples，因此極端錯標資料可能干擾訓練。

## 4.6 方法四：Dice Loss

Dice Loss 主要應用於 Segmentation。

假設 Ground Truth Mask 中：

- Background = 99%。
    
- Defect Pixels = 1%。
    

如果只用一般 Pixel-wise Cross Entropy，模型可能偏向 Background。

Dice Coefficient：

\[ Dice= \frac{2|P\cap G|}{|P|+|G|} \]

在 Soft Dice 中可以寫成：

\[ Dice= \frac{ 2\sum_i p_ig_i+\epsilon }{ \sum_i p_i+\sum_i g_i+\epsilon } \]

Dice Loss：

\[ L_{Dice}=1-Dice \]

它直接衡量預測與 Ground Truth 的重疊程度。

### Dice Loss vs Focal Loss

|面向|Focal Loss|Dice Loss|
|---|---|---|
|主要目標|Down-weight Easy Examples|改善重疊品質|
|常見任務|Classification / Detection|Segmentation|
|Pixel-level Imbalance|有幫助|通常很適合|
|直接優化 Mask Overlap|否|是|
|機率校正|不保證|不保證|
|常見組合|BCE + Focal|BCE + Dice|

Segmentation 可以使用：

\[ L= \lambda_1L_{BCE} +\lambda_2L_{Dice} \]

也可以視任務使用 Focal Tversky、Generalized Dice 等方法。

但 Dice 不是萬靈丹。例如很小的物件若只錯幾個 Pixels，Dice 就可能大幅改變，而且小 Batch 下的行為也要特別檢查。

## 4.7 方法五：Threshold Optimization

這是面試中很容易被忽略的一點。

假設二元模型輸出：

\[ P(Defect|x)=0.4 \]

如果使用 Threshold = 0.5：

\[ Prediction=Normal \]

但如果改成 Threshold = 0.3：

\[ Prediction=Defect \]

因此不需要重新訓練模型，也可能改變 Precision、Recall 與 F1。

互動練習：調整 Defect Detection Threshold

## 0.50

Precision

## 66.7%

Recall

## 80.0%

False Positives

## 2

使用 10 筆示範資料，其中 5 筆真正有缺陷。提高 Threshold 通常減少 False Positives，但也可能犧牲 Recall。

若產線的 False Negative Cost 很高，例如嚴重瑕疵漏檢，就可能選擇較低 Threshold。

如果 False Positive 導致昂貴的人工作業，則需要平衡 Precision。

更正式可以定義：

\[ ExpectedCost(t)= C_{FN}FN(t)+C_{FP}FP(t) \]

然後在 Validation Set 上選出最適合的 Threshold。

如果機率已正確校正，而且 FP/FN 成本固定，還能根據決策理論推導 Threshold；但在實際工業系統中，成本與資料分布往往更複雜。

Senior-Level 重點：Class Imbalance 涉及 Data Distribution、Loss Design 與 Decision Policy 三個不同層次，不能只靠改 Loss Function。

# Q5. Training Accuracy 很高但 Test Accuracy 很低，怎麼辦？

核心：Overfitting、Data Leakage、Distribution Shift

## 5.1 面試回答

> I would not immediately assume this is an overfitting problem.
> 
> I would first verify the dataset split, evaluation pipeline, label quality, and preprocessing consistency.
> 
> Then I would compare training and validation learning curves, investigate data leakage, and analyze whether the test distribution differs from training.
> 
> If the issue is overfitting, I would consider stronger augmentation, regularization, early stopping, transfer learning, or reducing model capacity.
> 
> If the issue is distribution shift, I would investigate camera conditions, lighting, object populations, and acquisition differences, then collect representative data and evaluate domain robustness.
> 
> I would make the diagnosis before choosing an intervention because different root causes require different solutions.

## 5.2 三個重要概念

### A. Overfitting

模型過度記住 Training Data，無法泛化到新資料。

典型 Learning Curves：

示意：Overfitting 的 Train / Validation Accuracy

訓練集持續進步，但驗證集開始停滯並下降

Train AccuracyValidation Accuracy

40%55%70%85%100%151015202530

假設性數據，顯示 Epoch 增加後的 Generalization Gap。

可能原因包括：

- Dataset 太小。
    
- Model Capacity 過大。
    
- Training Epochs 過多。
    
- Label Noise。
    
- Augmentation 不足。
    
- 樣本之間高度重複。
    

解法：

- 增加多樣化 Training Data。
    
- Data Augmentation。
    
- Weight Decay。
    
- Early Stopping。
    
- Dropout（視架構而定）。
    
- 降低 Model Complexity。
    
- 使用合適的 Pretraining。
    
- 修正錯誤標註。
    

但注意，增加 Dropout 不是萬用解法。

### B. Data Leakage

Data Leakage 是 Training 流程不小心使用了不應該取得的 Validation／Test 資訊。

例如：

你有 100 支手錶，每支手錶拍了 40 張影像。

總共：

\[ 100\times40=4000\ images \]

如果隨機把影像分成：

- 80% Train。
    
- 10% Validation。
    
- 10% Test。
    

同一支手錶的不同角度影像就可能同時出現在 Train 和 Test。

這是一種非常危險的 Leakage。

模型可能不是學會辨識新手錶，而是記住已出現過的手錶特徵。

正確方式是：

Group Split by Watch ID。

錯誤：Image-level Random Split

Watch A

Train

Test

Validation

Watch B

Train

Test

正確：Group-level Split

Watch A

Train：全部影像

Watch B

Test：全部影像

如果要評估新型號的泛化能力，可能還需要 Series-level Holdout；如果要評估未來新拍攝環境，則應使用 Time-based 或 Site-based Holdout。

其他 Leakage 包括：

- 在分割 Train/Test 之前先進行 Augmentation，導致原圖及其變體跨集合。
    
- 同一張原圖的 Overlapping Crops 跨集合。
    
- 使用整份資料計算需要學習的 Normalization／Feature Selection 統計量。
    
- 把 Test Set 用來調 Hyperparameters。
    
- 不同檔名的 Duplicate Images 同時存在兩個集合。
    

一個有趣的判斷：

Data Leakage 通常讓 Test Accuracy 虛高，而不一定讓它偏低。

如果 Test Accuracy 很低，不應直接把 Leakage 當成唯一原因。更合理的是完整 Audit Data Splitting 與 Evaluation。

### C. Distribution Shift

Training 與 Testing 來自不同分布。

\[ P_{train}(X,Y)\ne P_{test}(X,Y) \]

常見分類：

|類型|意思|例子|
|---|---|---|
|Covariate Shift|\(P(X)\) 改變|新光源、新相機|
|Label / Prior Shift|\(P(Y)\) 改變|Defect 比例改變|
|Concept Shift|\(P(Y\mid X)\) 改變|同樣外觀特徵的標準或標籤關係改變|

嚴格來說，Covariate Shift 通常還假設 \(P(Y\mid X)\) 大致不變。

例如工業影像：

Training 使用固定 LED Ring Light，Test 使用不同曝光、偏振或相機角度，產生不同反光。

即使模型在 Training Set 有 99% Accuracy，也不一定能適應新的影像分布。

## 5.3 如何系統性 Debug？

我會使用下面順序：

1

Verify Evaluation

確認 Label Mapping、Preprocessing、Threshold、模型權重與 Evaluation Code

2

Audit Data Split

確認 Group Leakage、Duplicates、Augmentation 及來源隔離

3

Inspect Learning Curves

判斷 Overfitting、Underfitting 或 Optimization 問題

4

Slice the Errors

依 Class、Camera、Resolution、Lighting、Site 等條件拆解

5

Test Hypotheses

一次修改一項因素並做 Ablation

6

Validate on Held-out Data

使用未參與調參的新資料重新驗證

### 實際例子

假設：

- Training Accuracy = 98%。
    
- Validation Accuracy = 95%。
    
- Test Accuracy = 67%。
    

若 Train 與 Validation 很接近，但 Test 大幅下降，我會優先懷疑：

1. Test Pipeline 不一致。
    
2. Test Labels 或 Class Mapping 有問題。
    
3. Test Distribution 與 Development Distribution 不同。
    
4. Validation Set 不具代表性。
    

反之，若：

- Training = 99%。
    
- Validation = 68%。
    
- Test = 66%。
    

則更像是早在 Validation 階段就已經出現 Generalization Problem。

這兩種情況的處理策略不同。

## 5.4 如何處理 Distribution Shift？

首先量化 Shift，而不是直接重新訓練。

例如：

- 比較 Brightness / Contrast Distribution。
    
- 比較 Blur／Sharpness Distribution。
    
- 比較 Object Size Distribution。
    
- 比較 Camera 與 Lens Configurations。
    
- 在 Pretrained Embedding Space 中檢查資料分群。
    
- 使用 Domain Classifier 分辨 Train 與 Test Images。
    

若 Domain Classifier 很容易區分兩個 Domain，這是存在分布差異的證據，但不代表已經找到造成 Accuracy 下降的真正原因。

接著做 Controlled Experiments：

例如只更換 Lighting Conditions，其他保持一致，測試模型性能差異。

最後再考慮：

- Domain-specific Augmentation。
    
- Domain Adaptation。
    
- Fine-tuning。
    
- Reweighting。
    
- 新資料蒐集。
    
- Camera / Lighting Standardization。
    

Senior-Level 重點：先用數據確認 Root Cause，而不是看見 Generalization Gap 就直接增加 Epoch、Dropout 或模型複雜度。

# Q6. 如何發現 Model Learning 了錯誤特徵？

核心：Error Analysis、Saliency、Counterfactual Tests

這一題是 Senior Computer Vision Engineer 面試中非常重要的問題，因為模型即使在 Test Set 表現很好，也不保證它學到的是正確的判斷依據。

## 6.1 面試回答

> A model can achieve high test accuracy while relying on spurious correlations instead of the actual features relevant to the task.
> 
> I would investigate this using structured error analysis, attribution methods such as Grad-CAM or Integrated Gradients, and controlled counterfactual experiments.
> 
> For example, if a defect classifier is supposed to identify scratches but its predictions change when the background or lighting changes, that suggests it may be relying on shortcuts.
> 
> I would test whether predictions remain stable when irrelevant factors change and whether they respond appropriately when task-relevant features change.
> 
> I would also validate explanations against controlled perturbation tests, because saliency maps alone are not sufficient to establish causality.

## 6.2 問題本質：Spurious Correlation

假設我們訓練一個模型辨識真假手錶。

Training Data 恰好符合：

- Authentic Watches：白色背景。
    
- Fake Watches：黑色背景。
    

模型可能學到：

\[ P(Fake\mid Background=Black) \]

而不是：

\[ P(Fake\mid WatchFeatures) \]

結果：

- Training Accuracy = 99%。
    
- Random Test Accuracy = 97%。
    

但是當 Authentic Watch 放在黑色背景時，模型可能錯誤判斷為 Fake。

這稱為：

Shortcut Learning / Spurious Correlation。

因為背景與 Label 有統計相關性，但背景不是合理的真偽判斷依據。

## 6.3 方法一：Error Analysis

我會先整理 Confusion Matrix。

例如：

|Actual \ Predicted|Authentic|Fake|
|---|---|---|
|Authentic|450|50|
|Fake|30|470|

然後不只看整體 Accuracy，而是檢查 False Positives 與 False Negatives 的共通特徵。

建立 Error Slices：

- Camera Type
    
- Image Resolution
    
- Lighting
    
- Brightness
    
- Object Orientation
    
- Object Size
    
- Background
    
- Watch Model / Series
    
- Capture Date
    

假設分析結果：

|條件|Accuracy|
|---|---|
|White background|98%|
|Black background|68%|
|Bright lighting|96%|
|Low lighting|81%|

假設性數據。

這暗示背景與光線可能是模型的問題來源，但尚不能證明因果關係。

接下來需要進行 Intervention Experiments。

## 6.4 方法二：Saliency Maps

Saliency Methods 用來了解模型的 Prediction 對哪些輸入區域比較敏感，或哪些區域對輸出有較大歸因值。

常見技術包括：

- Vanilla Gradient Saliency
    
- Grad-CAM
    
- Integrated Gradients
    
- Occlusion Sensitivity
    
- SHAP 類歸因方法
    

![説明可能AI(XAI)の必要性と活用方法](https://images.openai.com/static-rsc-4/cRvWiwPlzv9p0qZGSThpl5LIgVXmO6D5SnhlFxesjrvHfRWanHKASBkPhnt7xLxZc0KWPV0AsG5-k7ElKnje76nX4KZOojqjr9XE-f9QRkT6HM2lnUTvJWi55PVoQ-ygN49IH0f8r-1XwdgTgQwcFaUFvq_mT69LVQFcUwmUlpc?purpose=inline)

![Explaining predictions of Convolutional Neural Networks with 'sauron' package.](https://images.openai.com/static-rsc-4/KN23Mn3Kr9hmFJiH11yy8iwFjx1W9MKJQ1h59IMcJ0crgJixBjwE0rEjn-FlZis76nPtCdCE_WHEO9_yO--_owMxlQyZUiwH4Ds1cIIxw7UkwXPqhE_FHu9DBNpyME8X4ap094DA8CMgx2A_tmJ1jemhYmKbcgduDHPTZR77YOY?purpose=inline)

![Sanity Checks for Saliency Maps - データ分析関連のまとめ](https://images.openai.com/static-rsc-4/K-6MUZE7RmzQogoyrhKZ03fjd6dX_xxw4i3srUYQ2i7OIK13Tlyfy15RMQcK9y-WHmtLkanTICmhXRdNpgkjaUIa6vdmUs0eoOjJc6HXTIuxePLUH3lIPNGg3TDRP2tsiQ9k8TMbsUOrrN_GZ6oESvc_GiEQ2iQkiwW-Ld_aJ-E?purpose=inline)

![Explainable AI (XAI): Are we there yet? » Artificial Intelligence - MATLAB & Simulink](https://images.openai.com/static-rsc-4/pxZQjjf27r2fyhfIkWbK2D964ZvjA8bouHNqtWBdTGgFAyAC-T1ttAE7FKuCvXDNK34rqEjDGRF1ispuVRZ15dWRrtx26TBenznKXbF7Wh0DududBPZNgMUvczOawHyZflZvtayYOGF49yQB5tMobqQ7Zq16fOwLG0BK9ea2y3Q?purpose=inline)

![Reliable Evaluation of Attribution Maps in CNNs: A Perturbation-Based Approach | International Journal of Computer Vision | Springer Nature Link](https://images.openai.com/static-rsc-4/rasK1oqKHQ8U8O-TIDLxmr3SxJjcWuBVhEImOnQ_fNGP40z1qxlFl4zf6tHRyftoJLXEeZCmVjXAUwBMKZ83VKwGA6wFv9L--3sgrKOXg23EGvr7zdAcOlGrxZklOH2QO-EN3gdJ-PzwfvpblQiTjm4LGksrLvwqEUbB5LRtgtY?purpose=inline)

![CAM, Grad-CAM và Score-CAM trong CNN](https://images.openai.com/static-rsc-4/T_vLqwI1AijTjAr4Ffi1_adMJQfUiIGKcbOmayaHT4nJtcs185kZvxKix3kgKAAUIYrf8B34MLr-8LkSYwb06yPWU5SfqESsfaowvCxaivMfgET6AETKgrifq4us6f9BoEqJelNyPj4gRCHHBgNUwPdRZnwEh2cZMWeokv0z9lU?purpose=inline)

### A. Gradient Saliency

假設模型輸出某個類別分數 \(S_c(x)\)。

計算：

\[ Saliency(x)= \left| \frac{\partial S_c(x)}{\partial x} \right| \]

這個量衡量模型輸出對輸入 Pixels 的局部敏感程度。

如果背景區域具有很高 Saliency，我們可能懷疑模型依賴背景。

但是 Gradient 高不等於該 Pixel 在因果上決定了結果；梯度也可能受到 Saturation、Noise 與非線性影響。

### B. Grad-CAM

Grad-CAM 利用最後幾層 Convolution Feature Maps 與 Gradient 產生粗略的 Class Activation Map。

其中：

\[ \alpha_k^c= \frac{1}{Z} \sum_{i,j} \frac{\partial S_c}{\partial A_{ij}^k} \]

\(A^k\) 是第 k 個 Feature Map。

接著：

\[ L_{GradCAM}^{c}= ReLU\left( \sum_k\alpha_k^cA^k \right) \]

產生一張 Heatmap。

若模型在辨識 Logo 真偽時，Heatmap 主要集中在背景邊界而不是 Logo 上，這是一個值得進一步驗證的警訊。

### C. Integrated Gradients

Vanilla Gradient 只觀察輸入位置附近的局部變化。

Integrated Gradients 則從一個 Baseline \(x'\) 沿路徑積分到實際輸入 \(x\)：

\[ IG_i(x)= (x_i-x'_i) \int_0^1 \frac{\partial F(x'+\alpha(x-x'))} {\partial x_i} d\alpha \]

它可以降低某些局部 Gradient Saturation 帶來的問題。

但 Baseline 選擇會影響 Attribution 結果，也不保證歸因就是人類理解的因果解釋。

## 6.5 方法三：Counterfactual Tests

這比單純看 Heatmap 更接近真正的實驗驗證。

Counterfactual Test 的核心問題是：

如果只改變某個因素，其他條件保持不變，模型的 Prediction 會怎麼改變？

例如：

![Luxury watch isolated on white background. Gold and black watch.](https://images.openai.com/static-rsc-4/HrqqRSuO4TgCnr7-hcf28q30SUqIRoKtqyUH-NIex-Zqsw8zWumcG-t2k1HvgvuHQIVd2eLNeilSE0D-9rm5YtGBhK9VBzuDWa7TVxNPIEF7-hFW_98nn2He-uhZ86IjLkjsEP5RF_0_lFa4at_0cR3Qyj0KNeUOfCuj5hp7rXtFyiN-imjvnSt58ApGC5L4?purpose=inline)

Test A：白色背景

相同的手錶，原始背景

![Black Background Product Photography: The Ultimate Guide to Luxury E-commerce Visuals | Lumabox](https://images.openai.com/static-rsc-4/WnfJ_tjrLWHsAAmfietDr5z3WdATOnQS2yFvZAYn3mw2Sm5lSGmdrnzUp0-tYCuukx-1XJSnWPdPzHcC0eVcjSH0_bLHBGPok6uUK8HtPA6PrZF0BDOjtM9HMWjiWU_8YMiRGWPhYbSBXSD7FlXALYeGN_Qq8jDKguizrCWDrQY?purpose=inline)

Test B：黑色背景

示意對照；正式實驗必須使用同一張錶的配對影像

正式實驗應對同一張影像進行背景替換，儘量確保錶體 Pixels 不變。

假設模型結果：

\[ P(Fake\mid WhiteBackground)=0.08 \]

\[ P(Fake\mid BlackBackground)=0.91 \]

這種大幅變化表示模型可能過度依賴 Background。

接著還可以測：

|Intervention|想驗證什麼？|
|---|---|
|Change Background|是否依賴背景|
|Change Brightness|是否過度依賴曝光|
|Rotate Object|是否具旋轉穩健性|
|Remove Logo|是否真正使用 Logo|
|Mask Text|是否依賴文字特徵|
|Blur ROI|是否依賴 Fine Texture|
|Change Camera|是否有 Device Shortcut|

需注意，背景替換、遮罩、旋轉也可能產生不自然影像。這些介入結果必須排除影像編輯 Artifact 或 OOD Effect，不能把所有 Prediction Change 都歸因於 Shortcut。

## 6.6 更進一步：Feature Ablation

假設模型使用三種特徵：

- Dial Typography
    
- Hour Marker Geometry
    
- Background Appearance
    

可以分別遮蔽各類特徵，測量 Performance Change。

例如：

|Removed Feature|Accuracy|
|---|---|
|None|96%|
|Typography|76%|
|Hour Markers|84%|
|Background|95%|

假設性 Ablation 數據。

這代表 Typography 很可能提供較多模型使用的資訊，而 Background 的影響相對較小。

但這也不是嚴格的 Causal Attribution；遮蔽本身可能改變輸入分布。

最好搭配 Controlled Data Collection，例如同一支手錶在不同光線、背景、角度下重新拍攝。

## 6.7 Senior / Staff 的改善策略

確認模型依賴錯誤特徵後，可以採取：

- Targeted Data Collection：建立背景、光線與 Label 不相關的資料。
    
- ROI Masking / Cropping：移除不應依賴的背景資訊。
    
- Domain Randomization：變化非關鍵的影像條件。
    
- Hard Counterexamples：加入會打破 Shortcut 的訓練樣本。
    
- Feature-level Constraints：對需要使用的特徵建立結構化表示。
    
- Retraining + Independent Testing：確認改善在新資料上有效。
    

Senior-Level 重點：Explainability 不只是產生漂亮的 Heatmap，而是提出可以驗證的 Hypothesis，並透過實驗確認模型是否依賴真正有用的特徵。

# Q7. 如何比較兩個 Model 是否真的有提升？

核心：Independent Test Set、Statistical Significance、Slice Analysis

這是區分一般 ML Engineer 與 Senior／Staff Engineer 很重要的問題。

## 7.1 面試回答

> I would compare the models on the same independent test set using metrics aligned with the production objective.
> 
> I would control the evaluation conditions, including preprocessing, data versions, thresholds, hardware, and latency measurements.
> 
> I would use paired statistical comparisons, such as paired bootstrap confidence intervals or McNemar's test for paired classification errors.
> 
> I would also evaluate performance across important slices, because an overall improvement can hide regressions on critical subpopulations.
> 
> Finally, I would assess practical significance, including error costs, inference latency, memory usage, and deployment risk.

## 7.2 首先：什麼叫做 Model Improvement？

假設：

|Model|Test Accuracy|
|---|---|
|Model A|94.2%|
|Model B|95.1%|

是否可以說 B 比 A 好？

不一定。

我們還需要知道：

- Test Set 有多少樣本？
    
- 是不是同一組 Test Set？
    
- 樣本是否彼此獨立？
    
- 差異是否可能來自抽樣波動？
    
- 是否是多次調參後選出的最佳結果？
    
- 是否犧牲某些 Critical Classes？
    
- 是否造成 Production Latency 大幅增加？
    

## 7.3 Independent Test Set

正確的資料流程：

Complete Dataset

Train Set

Fit Parameters

Validation Set

Tune / Select

Test Set

Final Evaluation

Test Set 不應用於：

- 選最佳 Learning Rate。
    
- 選 Threshold。
    
- 選最佳 Architecture。
    
- 決定 Training Epoch。
    
- 根據結果反覆修改模型。
    

如果每次看到 Test Result 都調整模型，即使沒有直接用 Test Data Training，也會逐漸對 Test Set Overfit。

在實務上，可以保留一組完全獨立的 Final Holdout；對已經被大量使用的 Benchmark，也可能需要新的外部測試資料。

## 7.4 Statistical Significance

假設兩個模型在相同 1,000 個獨立樣本上的結果：

||Model B 正確|Model B 錯誤|
|---|---|---|
|Model A 正確|720|80|
|Model A 錯誤|120|80|

因此：

\[ Accuracy_A=\frac{720+80}{1000}=80\% \]

\[ Accuracy_B=\frac{720+120}{1000}=84\% \]

提升：

\[ \Delta Accuracy=4\ percentage\ points \]

但要看一個非常重要的資訊：

- A 正確、B 錯誤：80。
    
- A 錯誤、B 正確：120。
    

B 對部分樣本改善，對另一部分樣本退步。

### McNemar's Test

這是一個適合成對分類預測比較的統計測試。

在一般大樣本近似下：

\[ \chi^2= \frac{(b-c)^2}{b+c} \]

其中：

\[ b=80,\quad c=120 \]

所以：

\[ \chi^2=\frac{40^2}{200}=8 \]

未使用連續性修正的近似 p-value 約為 0.0047。

在其檢定假設成立、樣本獨立且未涉及未校正的多重比較時，可以拒絕「兩者具有相同邊際錯誤率」的虛無假設。

不過：

Statistical Significance 不等於 Practical Significance。

4 Percentage Points 的提升，還要看成本與關鍵類別的表現。

而且如果這 1,000 張其實是同 20 支手錶的大量重複影像，就不能把它們當成 1,000 個獨立 Observations 直接套用一般 McNemar Test。

## 7.5 Paired Bootstrap Confidence Interval

另一個常用方法是 Paired Bootstrap。

關鍵在於每次重新抽樣時，要保留兩個模型在同一樣本上的成對結果。

操作：

1. 從 Test Units 重新抽樣。
    
2. 計算 Model A 與 Model B 的 Metrics。
    
3. 計算 \(\Delta=Metric_B-Metric_A\)。
    
4. 重複例如 2,000–10,000 次。
    
5. 從 \(\Delta\) 的分布估計 Confidence Interval。
    

假設結果：

\[ \Delta F1=+0.018 \]

\[ 95\%\ CI=[-0.004,0.041] \]

這個區間包含零，所以尚不足以在常見雙側 5% 顯著性設定下宣稱 B 優於 A。

如果：

\[ 95\%\ CI=[0.009,0.032] \]

則在相應假設下，資料更支持 B 的改善大於零。

對於 Object Detection，可以對整個 Evaluation Set 做以 Image 或 Group 為單位的 Bootstrap，然後每次重新計算 AP，而不是把每一個 Bounding Box 當成獨立樣本。

如果一支手錶含 40 張 Images，且同支錶高度相關，應以 Watch ID 作為 Bootstrap Sampling Unit。

## 7.6 Slice Analysis

Model B 的 Overall Accuracy 比較高，不代表所有環境都改善。

例如：

模型比較：不同資料切片的表現

假設性結果，Model B 整體更好，但 Small Objects 退步

Model A

Model B

0%25%50%75%100%OverallLarge objectsSmall objectsLow lightNew camera

這是非常重要的發現。

如果 Small Object 是最關鍵的 Inspection Requirement，即使 Overall Accuracy 提升，也可能不應部署 Model B。

常見 Slice 包括：

- Small / Medium / Large Objects。
    
- Rare Classes。
    
- Capture Camera。
    
- Lighting Conditions。
    
- Blur Levels。
    
- Product Models。
    
- Geographic / Site Domain。
    
- New vs Existing Data。
    

此外，關鍵 Slices 若樣本數過少，估計值會很不穩定。應報告各 Slice 的樣本量與不確定性，避免過度解讀偶然差異。

## 7.7 Production Benchmark

完整比較不應只包含 Accuracy。

|指標|Model A|Model B|
|---|---|---|
|mAP|0.87|0.90|
|Critical Defect Recall|0.97|0.95|
|False Positive Rate|0.04|0.03|
|p95 Latency|30 ms|48 ms|
|Peak GPU Memory|2.2 GB|4.1 GB|
|Drift Robustness|Better|Worse|

示意數據。

Model B 的 mAP 改善，但是 Critical Defect Recall 下降。

如果 Critical Defect 的 False Negative Cost 很高，就不能單憑 mAP 選 B。

Senior Engineer 還會進行：

- Ablation Study。
    
- Multiple Random Seeds。
    
- Fixed Evaluation Protocol。
    
- Confidence Interval。
    
- Error Cost Analysis。
    
- Shadow Deployment。
    
- Canary Rollout。
    
- Rollback Planning。
    

Staff-Level 重點：你不只是比較 Model Scores，而是建立一套可重現、可審計、能決定模型是否安全部署的 Evaluation Framework。

# Q8. Model Confidence 很高卻經常判錯，如何改善？

核心：Calibration、OOD Detection、Uncertainty

## 8.1 面試回答

> High confidence does not necessarily imply high reliability.
> 
> I would first check whether the model is poorly calibrated, encountering out-of-distribution inputs, or making systematic errors on specific data slices.
> 
> I would evaluate calibration using reliability diagrams, expected calibration error, negative log-likelihood, and Brier score.
> 
> I would then consider temperature scaling or other calibration methods using held-out validation data.
> 
> For inputs outside the training distribution, I would evaluate OOD detection and uncertainty estimation, potentially using deep ensembles or other methods.
> 
> Finally, I would implement a selective prediction policy that routes uncertain or anomalous cases to human review, while measuring the trade-off between automated coverage and error risk.

## 8.2 Softmax Confidence 不等於真正機率

假設模型輸出：

\[ z=[5.0,1.0,0.5] \]

這些是 Logits。

Softmax：

\[ P(y=i\mid x)= \frac{e^{z_i}}{\sum_j e^{z_j}} \]

得到約：

\[ P=[0.971,0.018,0.011] \]

模型的最大 Confidence 約為 97.1%。

但這不表示它真的有 97.1% 的機率判對。

原因是 Softmax 只是把模型輸出的 Scores 正規化。即使輸入完全不屬於任何已知類別，它仍會產生一組總和為 1 的分數。

這就是為什麼 Closed-set Classifier 在看到陌生物件時，仍可能對某個已知類別非常有信心。

## 8.3 Calibration 是什麼？

理想 Calibration 是：

> 對所有 Confidence 大約 0.8 的預測，其中大約 80% 應該正確。

例如：

|Confidence Bin|Average Confidence|Actual Accuracy|
|---|---|---|
|0.5–0.6|0.55|0.52|
|0.6–0.7|0.65|0.57|
|0.7–0.8|0.75|0.61|
|0.8–0.9|0.85|0.72|
|0.9–1.0|0.95|0.79|

這代表模型過度自信。

Reliability Diagram：理想校正 vs 過度自信

分數越高不代表模型實際正確率等幅增加

Perfect CalibrationExample Model

0%25%50%75%100%50%60%70%80%90%100%

模型示意曲線。Reliability Diagram 的實際值應由測試資料的 Confidence Bins 統計計算。

### Expected Calibration Error（ECE）

\[ ECE= \sum_{m=1}^{M} \frac{|B_m|}{n} \left| acc(B_m)-conf(B_m) \right| \]

其中：

- \(B_m\)：第 m 個 Confidence Bin。
    
- \(acc(B_m)\)：該 Bin 的實際 Accuracy。
    
- \(conf(B_m)\)：平均預測 Confidence。
    

ECE 越接近 0，通常代表在該分箱方法下的校正誤差越小。

但 ECE 有限制：

- Bin 數與範圍會影響結果。
    
- 可能掩蓋特定類別或特定資料切片的誤校正。
    
- 在小樣本下可能不穩定。
    

所以也建議使用 NLL（Negative Log Likelihood）、Brier Score，以及 class-wise／slice-wise reliability analysis。

## 8.4 Temperature Scaling

這是最常見的 Calibration 方法之一。

原本的 Softmax：

\[ P(y=i\mid x)= Softmax(z_i) \]

Temperature Scaling：

\[ P_T(y=i\mid x)= Softmax\left(\frac{z_i}{T}\right) \]

其中 \(T>0\)。

- \(T=1\)：原始輸出。
    
- \(T>1\)：通常讓分布更平坦，降低過度自信。
    
- \(0<T<1\)：讓分布更尖銳。
    

互動實驗：Temperature 如何改變 Softmax？

Temperature

## 1.0

Class 1 (logit = 5)

97.1%

Class 2 (logit = 1)

1.8%

Class 3 (logit = 0.5)

1.1%

Temperature Scaling 對同一筆資料不改變 Logit 的排序，因此不會單靠這一步修正模型的分類錯誤。T 應使用獨立 Calibration Set 估計，而不是任意指定。

Temperature Scaling 通常會使用 Validation Set 最小化 Negative Log Likelihood 來選擇 T。

Guo et al. 的 Calibration 研究發現，Temperature Scaling 是一種簡單且在多種深度網路情境中有效的 Post-hoc Calibration 方法。

![](https://www.google.com/s2/favicons?domain=https://proceedings.mlr.press&sz=32)

Proceedings of Machine Learning Research

但這裡必須強調：

Calibration 改善的是 Probability Reliability，不一定改善 Accuracy。

如果模型原本分類就錯，Temperature Scaling 不會憑空讓它辨識到新的正確特徵。

## 8.5 OOD Detection

OOD = Out-of-Distribution。

例如模型只訓練過：

- Rolex Submariner
    
- Rolex GMT-Master
    
- Rolex Datejust
    

但現在輸入一支未涵蓋的特殊錶款。

Closed-set Classifier 仍可能輸出：

\[ P(Submariner)=0.98 \]

但它其實不應對陌生錶款給出如此確定的判斷。

這就是 OOD Detection 要解決的問題。

### 常見 OOD Methods

Maximum Softmax Probability

\[ Score_{MSP}(x)=\max_c P(y=c\mid x) \]

方法簡單，但對 OOD 的區分能力可能有限。

Energy-based OOD

常見 Energy Score：

\[ E(x)= -T\log\sum_c e^{z_c(x)/T} \]

可以利用模型的 Logits 產生不同於 Maximum Softmax Probability 的 OOD 指標。

Energy-based OOD Detection 的研究指出，它能在一些 Benchmark 中比單純 Softmax Confidence 更有效區分 In-distribution 與 OOD Samples。

![](https://www.google.com/s2/favicons?domain=https://papers.neurips.cc&sz=32)

NeurIPS Papers

Embedding Distance

將輸入影像映射到 Feature Embedding：

\[ f(x)\in\mathbb{R}^d \]

再比較輸入 Feature 與已知 Training Distribution 的距離，例如 Mahalanobis Distance。

OOD Metrics

- AUROC：區分 ID/OOD 的能力。
    
- AUPR：在指定正類定義下的 Precision-Recall 表現。
    
- FPR@95TPR：在保留 95% 指定正類召回率時的 False Positive Rate。
    

OOD Detection 並不能保證辨識所有未知情況，尤其是與已知樣本十分相似的 Near-OOD Inputs。

## 8.6 Uncertainty：Aleatoric vs Epistemic

這是 Senior AI Engineer 很常被追問的問題。

### Aleatoric Uncertainty

來自資料本身的隨機性、雜訊或不可辨識性。

例如：

- Motion Blur。
    
- Camera Noise。
    
- Low Lighting。
    
- Severe Occlusion。
    
- 兩種類別在影像上本來就難以區分。
    

即使增加訓練資料，有些不可約的不確定性仍然存在。

### Epistemic Uncertainty

來自模型知識不足。

例如：

- 很少看過某種錶款。
    
- 某類 Defect 的 Training Samples 太少。
    
- 模型沒有學過新的 Capture Domain。
    

增加足夠且具代表性的資料，可能降低這種不確定性。

### 如何估計？

Deep Ensembles

訓練多個模型：

\[ M_1,M_2,\ldots,M_K \]

計算平均預測：

\[ \bar p(y\mid x)= \frac{1}{K}\sum_{k=1}^{K}p_k(y\mid x) \]

觀察模型之間的 Prediction Disagreement。

MC Dropout

推論時保留 Dropout，多次 Sample Predictions，用預測分散程度近似部分模型不確定性。

Predictive Entropy

\[ H[p(y\mid x)] = -\sum_c p_c\log p_c \]

可以衡量預測分布的不確定程度，但 Entropy 本身不能可靠地區分 Aleatoric 與 Epistemic Uncertainty。

### Staff-Level 補充：Selective Prediction

對低可靠性的 Prediction，系統可以選擇不自動判定。

例如：

\[ Decision(x)= \begin{cases} AutoAccept,&\text{if criteria satisfied}\\ HumanReview,&\text{otherwise} \end{cases} \]

判定條件可以包含：

- Calibrated Probability。
    
- OOD Score。
    
- Model Disagreement。
    
- Image Quality。
    
- Required Evidence Completeness。
    

要注意不同 Score 不一定可直接比較，也不能假設一個 Softmax Confidence 就等於「系統可信度」。

可以評估 Risk-Coverage Curve：

\[ Coverage= \frac{\text{自動決策的樣本數}} {\text{總樣本數}} \]

\[ SelectiveRisk= \frac{\text{自動決策中錯誤的樣本數}} {\text{自動決策的樣本數}} \]

例如：

|自動處理比例|自動處理中的錯誤率|
|---|---|
|100%|8%|
|90%|5%|
|70%|2%|
|50%|1%|

假設性結果，顯示降低 Automation Coverage 可能換得較低錯誤率。

這對有人工審核機制的工業辨識系統非常重要。

另外，Conformal Prediction 也值得了解：在 Exchangeability 等假設成立時，可以建立具有邊際 Coverage Guarantee 的 Prediction Sets，但一般不能在任意 Distribution Shift 下保證有效。

Senior-Level 重點：Confidence、Calibration、OOD、Uncertainty、Decision Threshold 是五個相關但不相同的問題，不能全部用調整 Softmax Temperature 解決。

# 綜合實戰：將八題串成一個完整的 Computer Vision Production System

現在用一個較接近精密手錶影像檢測的例子，說明 Senior／Staff Engineer 如何真正設計整套系統。

## 情境

假設要開發一個高解析度影像辨識系統：

- 使用多台工業相機拍攝手錶零件。
    
- Macro Camera 擷取 4512×4512 Images。
    
- 某些特徵只佔影像面積約 1%。
    
- 需要檢查 Logo、Hour Markers、文字與局部幾何特徵。
    
- 缺陷或偽造樣本比正常樣本少很多。
    
- 不同手錶、相機、光照條件有 Distribution Differences。
    
- Production 需要提供可以追蹤的 Prediction、Confidence 與人工複核流程。
    

## 完整 Production Architecture

Industrial Camera Acquisition

High-resolution Images + Camera Metadata

Image Quality Gate

Focus、Exposure、Blur、Saturation、Capture Completeness

Detection and ROI Extraction

YOLO / RT-DETR + High-resolution Crops

Fine-grained Analysis

CNN / ViT / UNet / Feature Extraction

Classification

Component / Defect Predictions

Statistical Evidence

Geometry / Texture / Feature Distances

Calibration + OOD + Uncertainty

Assess Reliability and Missing Evidence

Decision Policy

Automated Result / Review / Recapture

Monitoring & Retraining

Error Slices、Drift、Human Labels、Versioned Evaluation

## 系統如何解決八個問題？

|問題|對應 System Design|
|---|---|
|YOLO vs DETR|Benchmark Detection Architecture|
|CNN vs ViT|選擇適合細節與 Compute Budget 的特徵模型|
|Small Objects|Original-resolution ROIs + Multi-scale Detection|
|Class Imbalance|Sampling + Weighted / Focal / Dice Loss|
|Overfitting|Group Split + Domain Testing|
|Wrong Features|Saliency + Counterfactual + Error Analysis|
|Model Comparison|Independent Test + Paired Statistical Tests|
|High Confidence Errors|Calibration + OOD + Human Review|

### 假設性實驗設計

先建立 Baseline，然後逐步做 Ablation。

|Experiment|Small-object Recall|False Positives / Image|p95 Latency|
|---|---|---|---|
|E0：Baseline Detector|0.58|0.20|25 ms|
|E1：Higher Resolution|0.69|0.25|42 ms|
|E2：Tiled Inference|0.81|0.35|90 ms|
|E3：Coarse-to-fine|0.84|0.24|60 ms|
|E4：E3 + Hard Negative Training|0.83|0.12|60 ms|

以上為說明如何設計實驗的假設性結果，並非實際模型測試數據。

在這個例子中，E4 的 Recall 比 E3 略低，但 False Positives 大幅下降。

接下來不能只問「E4 是不是最好」，而要考慮：

- Recall 0.83 是否滿足 Critical Defect Requirement？
    
- 60 ms 是否符合設備 Cycle Time？
    
- False Positive Reduction 是否能節省人工審核成本？
    
- 是否在特定 Camera／Series 上退步？
    
- 提升是否在獨立 Test Set 上有統計支持？
    
- Calibration 與 OOD Detection 是否正常？
    
- 如果 Image Quality 太差，能否觸發 Recapture 而不是強制判斷？
    

這是把 Machine Learning Experiment 轉換為 Production Engineering 的關鍵。

# 最後：面試官如何區分 Mid-level、Senior 與 Staff Engineer？

|能力|Mid-level|Senior|Staff|
|---|---|---|---|
|Architecture|熟悉 YOLO／CNN／ViT|能根據限制選型|定義跨專案選型標準|
|Data|能訓練與標註資料|能建立可靠的 Ground Truth 和 Splits|建立 Data Governance|
|Debugging|調 Hyperparameters|Root Cause Analysis|建立團隊通用 Debug Framework|
|Evaluation|Accuracy／F1／mAP|Statistical Testing + Slices|制定全系統品質門檻|
|Deployment|模型 Export / API|Latency、Monitoring、Rollback|ML Platform 與跨團隊可靠性|
|Ownership|完成模型任務|End-to-end Production Ownership|跨團隊技術方向與品質責任|

## 八題最值得記住的面試核心句

|Interview Question|Senior-Level Key Message|
|---|---|
|YOLO vs DETR|Compare detection formulation, matching strategy, convergence, and deployment constraints.|
|CNN vs ViT|Inductive bias, pretraining availability, compute, and information scale determine the choice.|
|Small Object|Preserve information before increasing model complexity.|
|Class Imbalance|Separate sampling, loss design, evaluation distribution, and decision policy.|
|Train/Test Gap|Diagnose leakage, overfitting, and distribution shift before choosing a fix.|
|Wrong Features|Validate feature reliance through controlled interventions, not saliency alone.|
|Model Improvement|Use paired statistical evaluation, critical slices, and production metrics.|
|High Confidence Errors|Calibration, OOD detection, uncertainty estimation, and selective prediction solve different problems.|

對 Senior／Staff 面試而言，最有價值的回答方式是把這八題視為互相連結的系統問題，而不是八個獨立演算法題目。

例如，Small-object Recall 很低，可能同時涉及 Resolution、Detection Architecture、Sampling、Class Imbalance 與 Data Quality。模型看起來改善了，也可能是 Data Leakage；即使真的提升，也不表示 Confidence 已經校正，更不代表在新相機或陌生產品上可靠。

最能展現 Senior／Staff 能力的，是清楚說出每個技術決策的假設、證據、Trade-offs、Failure Modes，以及如何驗證它在 Production 中真正有效。