
|                                            |     |
| ------------------------------------------ | --- |
| [[#### 2026 美國 Senior AI Engineer 求職完整指南]] |     |
|                                            |     |

#### 2026 美國 Senior AI Engineer 求職完整指南
```
請幫我整理2026年在美國要求職 computer vision方向的AI engineer及類似以AI跟computer vision, imaging processing為主的senior以上職位, 需求的技能是甚麼請詳細列出, 以及面試時可能會考到的是甚麼請詳細列出. 最後如果是LLM方向的相同職位又各是甚麼
```


| Positions                                            | Fields                                                   | Skills                                                                                                                                              |
| ---------------------------------------------------- | -------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Senior Computer Vision Engineer                      | [ CV/AI ]<br>Detection、Segmentation、Tracking、OCR         | Python、PyTorch、OpenCV、CNN、ViT                                                                                                                       |
| Senior Machine Learning Engineer – Computer Vision   | [ AI trainer ]<br>Training、Evaluation、Deployment、MLOps   | PyTorch、GPU、Dataset、Cloud、ML Pipeline                                                                                                               |
| Senior Imaging / Image Processing Algorithm Engineer | [ Camera ]<br>HDR、影像增強、降噪、ISP、影像融合                       | 研究Camera的Image Restoration、Demosaicing、Denoising、Image Fusion，並將 Deep Learning 整合進相機處理流程。Python、Classical CV、Camera Pipeline 是核心技能；C/C++ 是加分項。      |
| Senior Perception Engineer                           | [ Robotic ]<br>3D Vision、Robotics、SLAM、Sensor Fusion	C++ | 開發 Vision、Video、Robotic Action Models；要求 PyTorch、Python、C++、GPU Optimization 與模型驗證，3D Geometry、ROS、Real-time Systems, 並熟悉 Vision Foundation Models。 |
| Senior Applied Scientist – Computer Vision           | [ Research ]<br>新模型研究、實驗設計、模型效能突破	ML                     | ML Theory、Statistics、PyTorch、Research                                                                                                               |
| Senior Machine Vision / Inspection Engineer          | [ industrial ]<br>工業瑕疵偵測、自動化影像分析                         | 結合Camera影像演算法, Lighting、Calibration、AI、Automation、AI、Motion Control、Robotics 與實際工業設備                                                                |
| Senior Multimodal / Vision-Language Engineer         | [ VLM ]<br>結合影像與文字推理                                     | VLM、Multimodal AI	ViT、CLIP、VLM、LLM、Fine-tuning                                                                                                      |
|                                                      |                                                          |                                                                                                                                                     |

# 2026 美國 Senior AI Engineer 求職完整指南

Computer Vision · Image Processing · Machine Learning · LLM · Generative AI

以 2026 年 10 月美國科技公司的實際職缺要求為基礎，包括 Apple、Amazon、OpenAI、Anthropic，以及機器人與視覺 AI 公司。

## 一、2026 年美國 AI Engineer 的招聘方向

先講最重要的結論：2026 年 Senior AI Engineer 的競爭重點，已經不只是會不會訓練 Deep Learning Model，而是能不能設計、開發、評估、部署、優化一整套能在真實環境可靠運作的 AI System。

這對 Computer Vision 和 LLM 工程師都適用，但兩者的技術深度要求並不相同。

例如，2026 年的實際招聘資訊顯示：

- Apple – Senior Computer Vision / ML Engineer： 要求 3D Vision、Multi-view Geometry、Sensor Fusion、PyTorch，以及將研究演算法整合成 Production System 的能力。
    
    ![](https://www.google.com/s2/favicons?domain=https://jobs.apple.com&sz=32)
    
    招贤纳才 (中国)
    
- Amazon – Senior Applied Scientist（Perception）： 要求 Detection、Segmentation、Tracking、實際感測器資料處理，以及獨立領導技術開發與指導工程師的能力。
    
    ![](https://www.google.com/s2/favicons?domain=https://amazon.jobs&sz=32)
    
    amazon.jobs
    
- OpenAI – Applied AI Engineer： 強調 Agent、Retrieval、Evaluation、Reliability、Latency、Cost、Security，以及將 AI 從 Prototype 推進到 Production。
    
    ![](https://www.google.com/s2/favicons?domain=https://openai.com&sz=32)
    
    OpenAI
    

### Computer Vision 與 LLM 的核心差異

|比較項目|Computer Vision AI Engineer|LLM AI Engineer|
|---|---|---|
|核心問題|讓 AI 理解影像、影片、3D 與感測器資料|讓 AI 理解、生成、推理與執行任務|
|主要模型|CNN、ViT、YOLO、SAM、VLM|Transformer、LLM、Embedding、Reasoning Models|
|主要工作|Detection、Segmentation、OCR、Tracking、Inspection|RAG、Agent、Fine-tuning、LLM Evaluation|
|重要底層技術|OpenCV、影像處理、Camera、Geometry|Attention、Tokenization、Inference、Retrieval|
|系統整合|Camera、Lighting、GPU、Edge、Robotics|API、Vector DB、Tool Calling、Cloud|
|2026 新興方向|Vision Foundation Models、Physical AI、Multimodal|Agentic AI、Reasoning、Post-training、Agent Evaluation|
|Senior 核心能力|能打造可靠的 End-to-End Vision System|能打造可靠的 End-to-End LLM System|

有一項特別值得注意：Computer Vision 和 LLM 正在 Multimodal AI／Vision-Language Model（VLM）領域交會。

因此，熟悉相機、影像處理、傳統 CV 與 Deep Learning 的工程師，不一定要完全轉職到純 LLM；往 Vision-Language AI、Multimodal Perception 發展，也可能是一條很有競爭力的路線。

## 二、美國 Computer Vision／Image Processing 的 Senior 以上職位有哪些？

Computer Vision 領域其實分成很多不同職位。即使都叫 Senior AI Engineer，實際工作內容也可能差很多。

### 2.1 七種值得搜尋的職位

|職位名稱|工作核心|主要技能|
|---|---|---|
|Senior Computer Vision Engineer|Detection、Segmentation、Tracking、OCR|Python、PyTorch、OpenCV、CNN、ViT|
|Senior Machine Learning Engineer – Computer Vision|Training、Evaluation、Deployment、MLOps|PyTorch、GPU、Dataset、Cloud、ML Pipeline|
|Senior Imaging / Image Processing Algorithm Engineer|HDR、影像增強、降噪、ISP、影像融合|Classical CV、Signal Processing、Camera、C++|
|Senior Perception Engineer|3D Vision、Robotics、SLAM、Sensor Fusion|C++、3D Geometry、ROS、Real-time Systems|
|Senior Applied Scientist – Computer Vision|新模型研究、實驗設計、模型效能突破|ML Theory、Statistics、PyTorch、Research|
|Senior Machine Vision / Inspection Engineer|工業瑕疵偵測、自動化影像分析|Camera、Lighting、Calibration、AI、Automation|
|Senior Multimodal / Vision-Language Engineer|結合影像與文字推理、VLM、Multimodal AI|ViT、CLIP、VLM、LLM、Fine-tuning|

這七個職位並不是互斥的。同一家公司也可能使用不同名稱描述相似的工作。

### 2.2 實際美國職缺範例

以下是 2026 年官方招聘頁面上具有代表性的職缺。列出它們是為了理解招聘要求，不代表每個職缺在你閱讀時都仍開放申請。

![Periscope Lens: What is it? And When Will Apple Use One in iPhone? - MacRumors](https://images.openai.com/static-rsc-4/3yNPN22KItHJR78ZBvjQBFh1vwHn_Sc1KPfQ9vq3Q3L0tqjdu9CdsQ0rsK2p-N4_dd_XhXGsL-Fk5Wecsn-CJ9swN3YbMUrPuOUhmkaEq9HbIm7dfzwY6fzqOhpc48uOa_JyU8gya6gogRlucLK7gEAXqnTzpwH2r9KCaieeHuI?purpose=inline)

Apple — Computational Photography / Computer Vision ML Engineer

Cupertino, California · Imaging Algorithm

研究 Image Restoration、Demosaicing、Denoising、Image Fusion，並將 Deep Learning 整合進相機處理流程。Python、傳統 CV、Camera Pipeline 是核心技能；C/C++ 是加分項。

![](https://www.google.com/s2/favicons?domain=https://jobs.apple.com&sz=32)

Jobs at Apple

![Da Vinci Robotic Surgical Systems | Intuitive](https://images.openai.com/static-rsc-4/PStvamKshV2ZYqsYob4KrFGDWivAWjjcKbVQNaQ3FuKWe5BJeJeRE7fWC-nxrVPGqTiXg8Ylt6MxeFc8AOYydevO2cZIq10ibdacw1CHpE5ooJcsBqn6Wr0P5_BrHHhMJs6MFw7_OD5y4jYgt-c8MTeIDDT1WkPNr9ZQd5N6TXM?purpose=inline)

Intuitive Surgical — Senior Machine Learning Engineer

Sunnyvale, California · Medical Vision / Robotics

開發 Vision、Video、Robotic Action Models；要求 PyTorch、Python、C++、GPU Optimization 與模型驗證，並熟悉 Vision Foundation Models。

![](https://www.google.com/s2/favicons?domain=https://careers.intuitive.com&sz=32)

Intuitive Surgical Careers

![Advance Sensing | Machine Vision Company in India | Industrial Automation & Sensor Solutions Bengaluru](https://images.openai.com/static-rsc-4/DExci8-nkcBj2R-o9zT2SYZEqYXPJKJ5IgpugkOzvGZXXXv1twpNrHjrNDwztB--HCkQM6ABTc06mg8_tb3bXc9JQHZQyj-I1-o33TbNzi2CUBKCaAjDDBch4tfKdd3TqSqExu6WYVuK1dvNlo8G2UwjsmCypCztUNqjsWZ-OvM?purpose=inline)

Sciotex — Machine Vision Engineer / Scientist

Pennsylvania · Industrial Automation

結合影像演算法、AI、Camera、Motion Control、Robotics 與實際工業設備。這類職位相較純 AI Research，更重視整機的穩定性與硬體整合。

![](https://www.google.com/s2/favicons?domain=https://sciotex.com&sz=32)

Sciotex

![Industrial 3D Depth Sensing Enters the Warehouse: How the Orbbec-Basler Partnership Is Giving AMRs Human-Like Spatial Awareness | CXTMS](https://images.openai.com/static-rsc-4/lbA-tjbOmM88Ic_Q_oTr4gZlIfqVCXSz5La6a-Lcs2VkfoD_haNQTmjIFcU7M1ID_gcTM_JYEVGzeIWOe0T7S7vISbLp2as3CUmByIFzRrxWICiECZ6NteHE30BEzPKwJ7la4ADW5C9kvwdE5mX9RNmDO64Xwu9MZSEGoL4pDd0?purpose=inline)

Amazon — Senior Applied Scientist, Perception

Robotics · Multimodal Sensors

聚焦 Object Detection、Segmentation、Tracking、Scene Understanding，以及 Radar、Thermal 等感測器的 Machine Learning。

![](https://www.google.com/s2/favicons?domain=https://amazon.jobs&sz=32)

amazon.jobs

![Embedded AI Module Supplier Comparison Guide | TSV](https://images.openai.com/static-rsc-4/2P3ghV_ZW3YH98gcBjC2nQEvLIrLjTHdjKoUsZN-GNjUNE2zxK_Wp75WRyF0f0MkX5m6BoHh9Oa8XoRj5A8UNMzCA8saEtPzdgZkk6ZHMsQ9cNj5Zryhi4feiagWZHTgAVm5wQ9dRYCHgNROa_NM2laTRF-WbmB1_wBBBhi5bcM?purpose=inline)

AMD — Software Engineer, Autonomy and Vision Systems

Austin, Texas · Physical AI / Edge AI

開發 Real-time Image Processing、Embedded Vision、C/C++，並優化 CPU／GPU／FPGA 等不同運算平台的延遲與效能。

![](https://www.google.com/s2/favicons?domain=https://careers.amd.com&sz=32)

Advanced Micro Devices, Inc

## 三、Senior、Staff、Principal 到底差在哪裡？

美國公司的職級名稱並沒有全產業統一標準，Google、Meta、Apple、Amazon 和新創公司也不一定使用相同名稱。

|職級|通常期待的能力|面試最重視什麼|
|---|---|---|
|Senior Engineer|獨立負責複雜功能或完整子系統|技術深度、Coding、Debugging、System Design|
|Staff Engineer|領導跨團隊技術方向，設計多個子系統|大規模架構、技術決策、Mentoring、Influence|
|Senior Staff / Principal Engineer|制定平台架構與長期技術方向|跨組織影響力、Technical Strategy、風險管理|
|Senior / Principal Applied Scientist|設計研究方向、驗證新方法並創造可量化成果|Scientific Rigor、Statistics、Model Research|
|Engineering Manager / Director|管理團隊、人員發展及交付|Technical Leadership、Hiring、Execution|

例如 Senior Engineer 通常需要回答「你如何設計並完成這個系統？」；Staff Engineer 則進一步需要回答「為什麼公司應該採用你的架構？如何讓不同團隊採用？未來三年如何擴展？」

Senior 不代表完全不寫 Code，Staff／Principal 也不一定主要做管理。 很多公司的 Staff Engineer 仍然是 Individual Contributor（IC），並且需要很強的 Hands-on 能力。

## 四、Computer Vision Senior Engineer 要準備的詳細技術技能

以下把所需能力分成八大類。重要的是你不必在每一類都達到研究專家水準，而是應該有幾項特別深入的專長。

### 4.1 Programming 與 Software Engineering

幾乎所有職位的基礎

|技能|應該掌握的內容|重要性|
|---|---|---|
|Python|NumPy、SciPy、OpenCV、資料處理、效能分析|必備|
|PyTorch|Dataset、DataLoader、Training Loop、Autograd、Checkpoint、AMP|必備|
|C++|Memory Management、STL、Multithreading、Performance|依職位；Robotics／Real-time 特別重要|
|Git|Branch、Merge、Code Review、CI|必備|
|Linux|Shell、Process、GPU Driver、Environment、Debug|高|
|Docker|容器化訓練與部署|高|
|Testing|Unit、Integration、Regression、Hardware-in-the-loop|高|
|CUDA|GPU Programming、Kernel、Memory Transfer|特定高效能職位非常重要|

面試不一定要求你用 C++ 寫所有演算法，但是對 Real-time Vision、Robotics、Embedded 和 Performance Engineer 而言，C++ 可能是重要的招聘門檻。

### 4.2 Classical Computer Vision 與 Image Processing

Imaging / Inspection 職位尤其重要

|主題|必須理解的內容|
|---|---|
|Image Filtering|Gaussian、Median、Bilateral、Convolution、Sobel、Laplacian|
|Edge / Shape Analysis|Canny、Contours、Morphology、Connected Components|
|Image Registration|Feature Matching、Homography、RANSAC、Affine Transform|
|Image Stitching|Alignment、Warping、Exposure Compensation、Blending|
|Image Enhancement|Histogram Equalization、CLAHE、Tone Mapping、Sharpening|
|HDR|Exposure Bracketing、Image Alignment、HDR Merge、Exposure Fusion|
|Camera Calibration|Intrinsic、Extrinsic、Lens Distortion、Pixel-to-mm|
|Color Processing|White Balance、Color Correction Matrix、RGB／HSV／Lab|
|Image Quality|SNR、PSNR、SSIM、MTF、Sharpness|
|Optics / Sensors|Exposure、Gain、Depth of Field、Focus、Sensor Noise|

這些內容在純 LLM 職位幾乎不會考，但在 Imaging Algorithm、Camera Software、Industrial Inspection 非常關鍵。

### 4.3 Deep Learning for Computer Vision

|領域|需要熟悉的 Models / Concepts|
|---|---|
|Classification|ResNet、EfficientNet、Vision Transformer|
|Object Detection|YOLO、Faster R-CNN、DETR|
|Semantic Segmentation|U-Net、DeepLab、SegFormer|
|Instance Segmentation|Mask R-CNN、Mask2Former、SAM|
|OCR / Text Recognition|CRNN、CTC、Transformer OCR、Text Detection|
|Anomaly Detection|Autoencoder、PatchCore、Feature-based Anomaly|
|Representation Learning|Contrastive Learning、CLIP、DINO|
|Image Restoration|Denoising、Super-resolution、Deblurring|
|Multimodal AI|VLM、Vision Encoder、Cross-modal Fusion|

除了知道模型名稱，Senior 面試更重視四件事：為什麼選它、怎麼訓練它、怎麼衡量它，以及失敗時如何改善。

例如，面試官可能會要求你比較 U-Net、SegFormer 與 SAM，在小資料集、極小目標、GPU 有限，以及需要精確 Boundary 的不同情況下，哪個較適合。

### 4.4 Mathematics、Statistics、Machine Learning Theory

需要準備：

- Linear Algebra： Matrix、Eigenvalue、SVD、PCA、Coordinate Transform、Tensor Dimension。
    
- Calculus / Optimization： Gradient、Chain Rule、Backpropagation、SGD、Adam、Learning Rate Scheduling。
    
- Probability / Statistics： Bayes Theorem、Likelihood、Prior、Posterior、Confidence Interval、Hypothesis Testing。
    
- Loss Functions： Cross Entropy、Focal Loss、Dice Loss、IoU Loss、Contrastive Loss。
    
- Evaluation： Precision、Recall、F1、ROC-AUC、PR-AUC、mAP、IoU、Calibration。
    
- Uncertainty / Reliability： Class Imbalance、Data Drift、Out-of-distribution、False Positive、False Negative。
    

對研究型 Applied Scientist 而言，通常需要比工程部署型 MLE 更深入的統計與實驗設計能力。

### 4.5 Camera、Robotics、3D Vision

這一組需要依目標職位選擇深度。

|技能|相關工作|
|---|---|
|Camera ISP / Bayer / HDR|Camera Imaging Engineer|
|Lighting、Lens、Autofocus|Machine Vision、Industrial Inspection|
|Stereo、Depth、Point Clouds|3D Perception、Robotics|
|Epipolar Geometry、PnP|3D Vision、Camera Calibration|
|SLAM、VIO、Visual Odometry|AR／VR、Autonomous Vehicles|
|Kalman Filter、Sensor Fusion|Tracking、Robotics|
|ROS2、Motion Control|Robotics、Automation|

Apple 的 2026 年 Senior 3D Vision 職缺，明確要求 Multi-view Geometry、3D Tracking 與 VIO，並偏好具有 C++ 和 Sensor Fusion 經驗的人才。

![](https://www.google.com/s2/favicons?domain=https://jobs.apple.com&sz=32)

Jobs at Apple (UA)

但要強調：應徵 Image Processing／Machine Vision 不代表一定要精通 SLAM；應徵 Robotics Perception 才需要特別投入這部分。

### 4.6 Production AI、MLOps、Cloud

2026 年 Senior 工程師應能解釋完整流程：

Image / Video Acquisition

Camera · Sensors · Data Storage

Data Pipeline

Quality Control · Labeling · Versioning

Model Training & Evaluation

PyTorch · Experiments · Metrics

Deployment

ONNX · TensorRT · Docker · AWS · Edge

Monitoring & Continual Improvement

Drift · Latency · Failures · Retraining · Rollback

除了把模型成功部署，還應熟悉 Model Registry、版本回滾、Canary Deployment、Data Lineage、Observability，以及模型升級前後的 Regression Test。

### 4.7 GPU 與 Performance Optimization

Senior 應該會分析以下問題：

- PyTorch GPU 記憶體使用量為什麼突然增加？
    
- FP32、FP16、BF16、INT8 在效能、精度、數值穩定性方面有何差異？
    
- 如何使用 ONNX Runtime、TensorRT 或 NVIDIA Triton？
    
- 如何利用 Batching、Asynchronous Execution、Pinned Memory 提升 Throughput？
    
- CPU→GPU Transfer 與影像 Preprocessing 誰才是真正瓶頸？
    
- 如何分析 P50／P95／P99 Latency，以及運算資源的使用效率？
    

### 4.8 Senior 專屬：Ownership 與 Technical Leadership

最後一個領域容易被低估。

公司想知道你是否能在不明確的需求下，獨立：

1. 定義問題與 Success Metrics。
    
2. 判斷要採用傳統影像演算法還是 Deep Learning。
    
3. 設計 Data Collection 和 Ground Truth。
    
4. 選擇模型、實驗方法與驗證方式。
    
5. 解決 Production Bottleneck 和 Failure Modes。
    
6. 跨 Hardware、Software、Research 團隊協作。
    
7. 指導其他工程師並交付可維護的系統。
    

這也是 Senior 升到 Staff／Principal 的主要能力差異之一。

## 五、Computer Vision Senior 面試實際會考哪些？

### 5.1 2026 年仍然要準備 LeetCode 嗎？

答案是要，但不同公司和職位的比重不同。

2026 年並沒有全面取消 Coding Interview。例如：

- Amazon 的 Senior Software Engineer（SDE III）招聘指南仍明確列出 Coding、System Design、Low-level Design、High-level Design 和 Leadership 面試。其公開流程包含 60 分鐘 Technical Phone Screen，以及五場各約 55 分鐘的 Interview Loop。
    
    ![](https://www.google.com/s2/favicons?domain=https://www.amazon.jobs&sz=32)
    
    Amazon.jobs
    
- Amazon Applied Scientist 的官方流程則是 1–2 場 Technical Phone Screen，之後是四場約 55 分鐘的 Interview Loop，涵蓋 Coding、Scientific Knowledge、Tech Talk 和 Leadership。
    
    ![](https://www.google.com/s2/favicons?domain=https://amazon.jobs&sz=32)
    
    Amazon.jobs
    
- Anthropic 的技術面試仍使用 Colab、CodeSignal 等 Live Coding 工具，而不是讓 AI 自動完成程式。
    
    ![](https://www.google.com/s2/favicons?domain=https://www.anthropic.com&sz=32)
    
    Anthropic
    

依據這些招聘流程，我建議把應徵方向分成以下幾種準備策略：

|職位|LeetCode／演算法|CV / ML 知識|System Design|
|---|---|---|---|
|Senior Software Engineer – AI|高|中～高|很高|
|Senior Computer Vision Engineer|中～高|很高|高|
|Senior Imaging Algorithm Engineer|中|很高|中～高|
|Senior Applied Scientist – CV|中|極高|高，側重 ML Design|
|Senior Robotics Perception Engineer|中～高|很高|很高|
|Staff / Principal AI Engineer|中～高，視職位而定|極高|極高|

此表是根據職位工作內容整理的面試準備優先度，不是公司公開的正式配分。

### 5.2 Coding Interview：建議準備的題型

|題型|可能的面試問題|應掌握技巧|
|---|---|---|
|Array / HashMap|Two Sum、Group Anagrams、頻率統計|Hash Table、Time Complexity|
|Sliding Window|Longest Substring、影像串流時間窗口統計|Two Pointers、Window|
|Heap / Top K|找出最高信心的 K 個物件|Heap、Sorting|
|Binary Search|在排序 Threshold 中尋找最佳值|Binary Search|
|Intervals|Merge Intervals、合併偵測區域|Sorting、Greedy|
|Graph / BFS / DFS|計算 Binary Image 的 Connected Components|BFS、DFS、Graph|
|Dynamic Programming|Edit Distance、Sequence Alignment|State、Transition|
|Cache Design|設計 LRU Cache|HashMap、Doubly Linked List|
|Concurrency|Producer-consumer Camera Pipeline|Queue、Lock、Thread Safety|

此外，CV 專業 Coding Interview 也可能直接考：

- 使用 NumPy 實作 2D Convolution。
    
- 計算兩個 Bounding Boxes 的 IoU。
    
- 實作 Non-Maximum Suppression（NMS）。
    
- 實作影像 Normalize、Crop、Resize、Padding。
    
- 使用 OpenCV 找出 Contours 或 Connected Components。
    
- 實作計算 Precision、Recall、Dice、Confusion Matrix 的函式。
    
- 實作一個簡單的 PyTorch Dataset／DataLoader。
    

其中 NMS、IoU、Convolution、Dataset 是很適合練習的題目，因為能同時測試 Coding 和 Computer Vision 基礎。

### 5.3 Computer Vision 專業知識面試題庫

以下題目是依據公開職缺要求編寫的模擬面試題，不是宣稱某家公司實際使用或外洩的題目。

#### A. Classical Vision / Image Processing

|可能考題|面試官期待的回答重點|
|---|---|
|1. Gaussian Filter 與 Median Filter 有什麼差異？|線性／非線性、Gaussian Noise、Salt-and-pepper Noise|
|2. Laplacian 與 Tenengrad 如何用於 Autofocus？|Focus Measure、Gradient、Noise、搜尋策略|
|3. HDR Merge 如何處理不同曝光影像？|Exposure、Alignment、Weighting、Ghosting|
|4. 如何校正 Camera Lens Distortion？|Intrinsic Matrix、Radial／Tangential Distortion|
|5. Image Stitching 有哪幾個主要步驟？|Feature Matching、RANSAC、Homography、Blending|
|6. 如何衡量影像 Sharpness？|Laplacian Variance、MTF、頻率響應|
|7. RGB、HSV、Lab 各適合什麼分析？|色彩表示、亮度分離、Color Distance|
|8. 為什麼更高解析度不一定代表更好的辨識率？|Optical Resolution、SNR、Blur、Data Quality|

#### B. Deep Learning / ML

|可能考題|面試官期待的回答重點|
|---|---|
|9. U-Net 為什麼適合 Semantic Segmentation？|Encoder／Decoder、Skip Connections|
|10. YOLO 和 DETR 差異在哪裡？|Detection Architecture、Speed、Training、Matching|
|11. CNN 與 Vision Transformer 如何選擇？|Inductive Bias、Data Requirement、Compute|
|12. 小型目標只佔影像 1%，如何提高辨識率？|High-resolution Crops、Multi-scale、Sampling|
|13. Dataset 只有 500 張時，如何訓練 Segmentation Model？|Transfer Learning、Augmentation、Validation|
|14. Class Imbalance 如何處理？|Sampling、Focal Loss、Dice Loss、Threshold|
|15. Training Accuracy 很高但 Test Accuracy 很低，怎麼辦？|Overfitting、Data Leakage、Distribution Shift|
|16. 如何發現 Model Learning 了錯誤特徵？|Error Analysis、Saliency、Counterfactual Tests|
|17. 如何比較兩個 Model 是否真的有提升？|Independent Test Set、Statistical Significance、Slice Analysis|
|18. Model Confidence 很高卻經常判錯，如何改善？|Calibration、OOD Detection、Uncertainty|

#### C. Model Deployment / Performance

|可能考題|面試官期待的回答重點|
|---|---|
|19. PyTorch Inference 太慢，怎麼找瓶頸？|Profiling、GPU Utilization、Transfer Cost|
|20. INT8 Quantization 對模型有什麼影響？|Accuracy／Latency Tradeoff、Calibration|
|21. 要把 1 秒 Inference 降成 100 ms，怎麼做？|ROI、Model Choice、Compression、Hardware|
|22. 如何設計多相機平行擷取系統？|Queue、Concurrency、Synchronization、Backpressure|
|23. 如果 Production Camera 的顏色與 Training Camera 不同怎麼辦？|Color Calibration、Domain Shift、Retraining|
|24. 如何確保新 Model 更新不會降低 Production 品質？|Golden Dataset、Regression、Canary、Rollback|

### 5.4 Imaging / Machine Vision 特別容易深入追問的實務題

這是值得特別準備的領域，因為很多純 AI 工程師可能擅長訓練模型，卻不熟悉真實相機與工業設備。

情境一：影像明明有 4K 解析度，但小文字仍然無法讀取。

面試官可能問：如何判斷是 Camera、Lens、Lighting、Focus，還是 OCR Model 的問題？

比較完整的回答應該包括：

1. 檢查 Raw Image，不要先對增強後影像下結論。
    
2. 檢查 Exposure、Gain、Motion Blur、Depth of Field。
    
3. 檢查 Lens Resolution、Focus、Pixel Scale。
    
4. 分析影像中的字元實際佔多少 Pixels。
    
5. 檢查 Demosaic、Sharpening 是否產生 Artifact。
    
6. 使用 Ground-truth Crops 區分 Detection 與 Recognition Error。
    
7. 最後才決定要改善光學、資料，還是模型。
    

情境二：模型在實驗室有 99% Accuracy，上線後只有 92%。

優秀回答不會直接說「重新訓練模型」，而是先調查：

- Camera 或 Lighting 是否更換？
    
- 新設備是否存在 Calibration 差異？
    
- Production 資料是否涵蓋不同零件／外觀？
    
- 是否因少見 Class、低品質影像、標註錯誤而下降？
    
- Accuracy 是否掩蓋了重要的 False Negative？
    
- 如何建立監測、人工複核、資料回饋與安全回滾機制？
    

### 5.5 ML System Design：Senior 以上的核心面試

考題可能是：

> Design an end-to-end automated visual inspection system that can detect tiny defects from multiple cameras and continuously improve after deployment.

這類題目可能需要你在白板上設計：

Camera / Lighting / Motion Control

Acquisition / Quality Check / Image Processing

Detection / Segmentation / Feature Extraction

Decision / Confidence / Anomaly Handling

Storage / Review / Monitoring / Audit

Dataset Versioning / Retraining / Deployment

面試時還會針對每個環節深入追問：

- What if one camera fails?
    
- How do you handle missing images?
    
- How do you make the pipeline recoverable?
    
- How do you choose decision thresholds?
    
- How do you monitor false negatives?
    
- How do you prevent data leakage between training and testing?
    
- How do you safely deploy a new model across 100 machines?
    

這些問題往往比記住某個神經網路的所有 Layer 更能反映 Senior 工程師的系統設計能力。

## 六、如果改成 LLM／Generative AI 方向，Senior 以上職位有哪些？

LLM 職位需要先區分兩大類，因為要求非常不同：

第一類：Applied LLM / AI Engineering。 使用現有模型（例如商用 API 或 Open-weight Models），設計 RAG、Agent、企業 AI 系統，並負責部署、測試與優化。

第二類：LLM Model Research / Training / Infrastructure。 改良或訓練模型本身，包括 Pretraining、Post-training、Reinforcement Learning、Distributed Training、Inference Optimization。

### 6.1 七種 LLM 相關職位

|職位|工作內容|技術深度的重點|
|---|---|---|
|Senior Applied AI Engineer|企業 LLM Application、部署、評估|RAG、Agents、API、Production|
|Senior LLM / Generative AI Engineer|模型整合、Fine-tuning、推論應用|Transformer、PyTorch、PEFT|
|Senior AI Agent Engineer|Tool Calling、多步驟推理與執行|Agent Architecture、Evals、Safety|
|Senior LLM Research Engineer|訓練與改良模型|Deep Learning、SFT、RL、Distributed Training|
|Senior LLM Inference Engineer|高效能 LLM Serving|GPU、KV Cache、vLLM、Parallelism|
|Senior Applied Scientist – NLP / LLM|模型研究、演算法與實驗|Statistics、Research、Training|
|Senior Multimodal AI Engineer|影像、文字、語音整合|VLM、Vision Encoder、Multimodal Training|

### 6.2 實際職位要求

OpenAI — Applied AI Engineer, Enterprise

需要將 Agents、Retrieval、Evaluation、Tools、Latency、Cost、Security 整合成可維護的 Production System。特別強調技術決策與實際交付，而不是只做 Demo。

![](https://www.google.com/s2/favicons?domain=https://openai.com&sz=32)

OpenAI

OpenAI — Applied AI Engineer, Codex Core Agent

主要研究 Agent 在真實 Coding 任務中的 Task Success、Tool Usage、Context Construction、Regression Testing 及 Production Failures。

![](https://www.google.com/s2/favicons?domain=https://openai.com&sz=32)

OpenAI

OpenAI — Software Engineer, Inference – Multi Modal

著重高 Throughput、低 Latency、GPU Utilization、Tensor Parallelism、vLLM、TensorRT-LLM 與 Distributed Systems。

![](https://www.google.com/s2/favicons?domain=https://openai.com&sz=32)

OpenAI

Amazon — Senior Applied Scientist, AI Lab

需要 LLM、Diffusion、Agents、Reinforcement Learning，以及大規模模型訓練和研究能力。

![](https://www.google.com/s2/favicons?domain=https://www.amazon.jobs&sz=32)

Amazon.jobs

這也表示：應徵 Senior Applied AI Engineer，不一定需要自己從頭 Pretrain 一個 LLM；但應徵 Foundation Model Research Engineer，就不能只會呼叫 API。

## 七、LLM Senior AI Engineer 詳細技能清單

### 7.1 LLM Foundation / Transformer

建議完整理解以下內容：

|知識|需要掌握的深度|
|---|---|
|Transformer|Attention、Feed-forward、Residual、LayerNorm|
|Self-attention|Query、Key、Value、Scaled Dot-product|
|Positional Encoding|Sinusoidal、RoPE 等方法|
|Tokenization|BPE、Token Count、Vocabulary|
|Model Architecture|Encoder-only、Decoder-only、Encoder-decoder|
|Generation|Greedy、Beam Search、Temperature、Top-p|
|Embeddings|Semantic Similarity、Representation|
|Context Window|Long Context、Truncation、Memory|
|KV Cache|如何加速 Autoregressive Decoding|

若是 Research Engineer，應能進一步解釋模型的數學原理、計算複雜度與實作細節。

### 7.2 RAG：Retrieval-Augmented Generation

對 Applied AI Engineer 而言，這通常是非常重要的技能。

需要理解：

- Document Loading、Parsing、Chunking、Metadata。
    
- Embedding Models、Vector Databases、FAISS。
    
- Dense Retrieval、BM25、Hybrid Search。
    
- Reranking、Query Rewriting、Multi-hop Retrieval。
    
- Retrieval Evaluation：Recall@K、MRR、NDCG。
    
- Answer Evaluation：Correctness、Groundedness、Citation Accuracy。
    
- Document Versioning、Freshness、Access Control。
    
- 如何處理 Hallucination、資料權限與過期文件。
    

面試官可能要求你設計一個能讀取公司內部幾十萬份文件的問答系統，而且不同員工只能看到自己有權限存取的文件。

### 7.3 Fine-tuning / Post-training

|技能|需要知道什麼|
|---|---|
|SFT|Supervised Fine-tuning、Instruction Dataset|
|LoRA|Low-rank Adaptation 如何降低可訓練參數量|
|QLoRA|Quantization 與 LoRA 的配合|
|DPO|Direct Preference Optimization|
|RLHF|Reward Model、Preference、Policy Optimization|
|RL / Reasoning|Reward Design、Verifiable Rewards、Policy Improvement|
|Distillation|Teacher–Student Model、Quality／Cost Tradeoff|
|Distributed Training|DDP、FSDP、ZeRO、Checkpointing|

並非每個職位都需要實作 RLHF 或大規模 Distributed Training。Applied AI 職位通常重視知道何時應用；Model Research 則可能要求真正實作、修改或優化。

### 7.4 Agentic AI

這是 2026 年值得優先投資的 LLM 應用技能。

需要具備：

- Function Calling、Tool Calling、Structured Outputs。
    
- Agent State、Memory、Context Management。
    
- Workflow Orchestration、Retries、Timeouts。
    
- Multi-step Planning、Tool Result Verification。
    
- Human-in-the-loop、Approval Gates。
    
- MCP 等 Tool Integration Protocol。
    
- Prompt Injection Defense、Privilege Boundaries。
    
- Agent Evaluation、Task Completion、Failure Recovery。
    

真正的難點不是讓模型呼叫三個 Tools，而是讓它在 Tool 失敗、資訊不完整、任務執行到一半或權限不足時，仍然安全可靠。

### 7.5 LLM Inference / Production

|技術|Senior 應理解的問題|
|---|---|
|vLLM / TensorRT-LLM|如何提高模型 Serving Efficiency|
|Continuous Batching|如何同時處理大量 Requests|
|KV Cache|GPU Memory 與 Decode Efficiency|
|Quantization|INT8／INT4 的效能與品質交換|
|Model Parallelism|Tensor、Pipeline、Data Parallelism|
|Latency|TTFT、Time per Output Token、P95|
|Throughput|Tokens/sec、Concurrent Users|
|Cloud Deployment|Scaling、Monitoring、Rollback|
|Reliability|Rate Limits、Retries、Fallback、Circuit Breakers|
|Cost Optimization|Model Routing、Caching、Token Budget|

### 7.6 LLM Evaluation 與安全性

這是 Senior 以上很值得深入準備的方向。

公司不只要知道你的 Agent 可以回答問題，還會想知道：

- 如何證明新 Prompt 比舊 Prompt 更好？
    
- 如何建立具有代表性的 Golden Test Set？
    
- 如何避免 LLM-as-a-Judge 的偏差？
    
- 如何測試 Agent 是否正確使用 Tools？
    
- 如何區分模型錯誤與 Retrieval 錯誤？
    
- 如何測試 Prompt Injection、資料外洩與越權操作？
    
- 如何量化模型成本、品質、Latency 和 Task Success Rate？
    

OpenAI 的 Applied AI 和 Agent Engineering 職缺都把 Evaluation、Failure Analysis 與可靠部署列為重要工作。

![](https://www.google.com/s2/favicons?domain=https://openai.com&sz=32)

OpenAI

+1

## 八、LLM Senior Engineer 面試題庫

同樣地，以下是依職缺技能推導的練習題，不是特定公司的正式題庫。

### 8.1 Transformer / LLM Theory

|可能考題|應準備的回答|
|---|---|
|1. Explain Self-attention.|Q、K、V、Attention Matrix、Softmax|
|2. 為什麼 Transformer 比 RNN 更容易平行訓練？|Sequence Dependencies、Parallel Computation|
|3. Self-attention 的時間／記憶體複雜度？|標準 Dense Attention 對序列長度的二次成本|
|4. Decoder-only 與 Encoder-decoder 差異？|Architecture、Training、Generation|
|5. 為什麼 LLM 會產生 Hallucination？|Training Objective、Knowledge、Grounding、Uncertainty|
|6. KV Cache 如何加速 Inference？|避免重複計算過去 Tokens 的 K/V|
|7. Temperature 和 Top-p 影響什麼？|Sampling Distribution、Output Diversity|
|8. LoRA 為何能降低 Fine-tuning 成本？|Low-rank Update、Frozen Base Weights|

### 8.2 RAG / Applied AI

|可能考題|應準備的回答|
|---|---|
|9. Design a production RAG system.|Retrieval、Ranking、Generation、Evals|
|10. Chunk Size 如何決定？|Context、Recall、Precision、Latency|
|11. RAG 和 Fine-tuning 何時使用？|Knowledge Updates vs Behavior Adaptation|
|12. Retrieval 找不到答案怎麼辦？|No-answer Policy、Fallback、Logging|
|13. 如何避免不同使用者存取彼此的文件？|ACL-aware Retrieval、Authorization|
|14. 如何評估 RAG 的品質？|Retrieval + Answer Evaluation|
|15. RAG Latency 太高怎麼改善？|Cache、Index、Rerank、Parallelism|

### 8.3 Agent / LLM System Design

|可能考題|應準備的回答|
|---|---|
|16. 設計一個能呼叫多個 API 的 Agent。|State、Tools、Orchestration|
|17. Agent 執行到第七步失敗怎麼辦？|Checkpoints、Retry、Idempotency|
|18. Tool Calling 回傳錯誤 JSON 怎麼辦？|Schema Validation、Recovery|
|19. 如何防止 Agent 執行未授權動作？|Permission、Isolation、Approval|
|20. 如何防止 Prompt Injection？|Trust Boundaries、Tool Restrictions|
|21. 怎麼測試 Multi-step Agent？|End-to-end Evals、Failure Injection|
|22. 如何控制 Agent 的成本？|Token Budget、Model Routing、Limits|

### 8.4 Model Training / Research

|可能考題|應準備的回答|
|---|---|
|23. SFT、DPO、RLHF 差異？|Objective、Data、Training Pipeline|
|24. 如何準備高品質 Instruction Dataset？|Filtering、Deduplication、Label Quality|
|25. 模型 Fine-tuning 後能力退化怎麼辦？|Regression、Data Mix、Overfitting|
|26. 如何訓練超出單張 GPU 記憶體的模型？|FSDP、ZeRO、Checkpointing|
|27. 如何設計 Reasoning Model 的 Reward？|Correctness、Reward Hacking、Evaluation|

### 8.5 LLM Coding / Performance

|可能考題|應準備的回答|
|---|---|
|28. 實作簡易 Self-attention。|Matrix Operations、Masking|
|29. 實作 Embedding Cosine Similarity Search。|NumPy、Vectorization、Top-K|
|30. 寫一個支援 Retry 的 LLM API Client。|Timeout、Backoff、Rate Limiting|
|31. 如何減少 LLM Serving 的 GPU 記憶體？|KV Cache、Quantization、Batching|
|32. P95 Latency 突然變高如何排查？|Queue、Prefill、Decode、GPU、Traffic|
|33. 如何寫 Agent Tool Execution 的 Unit Tests？|Mock、Failure Cases、Determinism|

## 九、LLM System Design 面試範例

假設面試官問：

> Design an enterprise AI assistant that uses internal documents, calls business tools, and supports 10,000 users while enforcing document access permissions.

你應能設計出類似：

User / Authentication / Authorization

API Gateway / Agent Orchestrator

RAG Pipeline

ACL Filter · Retrieval · Rerank

Tool Execution

Schema · Approval · Sandbox

LLM / Model Routing / Inference

Verification / Response / Citations

Evaluation / Tracing / Feedback / Monitoring

面試官通常會追問：如果文件每天更新怎麼辦？如果 Vector Search 找錯資料呢？如果 Agent 呼叫的 Tool 失敗？如果使用者要求未授權操作？如果 Token Cost 增加三倍如何處理？

Senior Engineer 需要從 Architecture、Failure Recovery、Cost、Security、Latency 與 Evaluation 同時回答，而不是只畫出 LLM API 和 Vector Database。

## 十、2026 年 Coding AI 普及後，面試最大的改變是什麼？

值得分清楚兩件事：

工作時使用 AI Coding Tools 已經很正常，但面試時能否使用 AI，由公司決定。

例如 Anthropic 的公開應徵政策明確指出：準備面試可以使用 Claude，但 Take-home Assessment 和 Live Interview 原則上不可使用 AI，除非公司明確允許。

![](https://www.google.com/s2/favicons?domain=https://www.anthropic.com&sz=32)

Anthropic

因此不應因為平常都用 Codex、Claude 或 Copilot 寫 Code，就停止練習獨立 Coding。

### 傳統 Coding 與 AI 職位面試應如何分配準備？

下面是我建議的個人準備時間比例，不是招聘公司的正式面試權重。

Senior Computer Vision / Applied ML Engineer

Coding 20%

CV/ML 35%

Design 30%

Leadership 15%

Senior Applied LLM / Agent Engineer

Coding 20%

LLM 25%

Design 40%

Leadership 15%

Senior LLM Research Engineer

Coding 15%

Research/ML 50%

Design 25%

Leadership 10%

Coding

專業理論

System Design

Leadership

建議準備比例。若目標是 Amazon SDE III 或其他通用 Software Engineering 職位，Coding 應另外提高優先度。

## 十一、針對你的經驗，我建議優先考慮哪些方向？

你之前開發的 Moonlight 系統，已涉及 Camera、Image Processing、Autofocus、HDR、Segmentation、影像特徵分析、AWS Training Pipeline，以及 Hybrid Statistical–Bayesian Authentication。

這些經驗可以整理成一條完整的「Hardware-aware AI → Computer Vision → Production ML System」主軸。

依照目前了解的專案內容，我會這樣排序：

|優先度|建議投遞方向|原因|
|---|---|---|
|1|Senior Computer Vision / Imaging Algorithm Engineer|與相機、影像處理、HDR、Autofocus 非常直接相關|
|2|Senior Machine Vision / Inspection AI Engineer|具備視覺設備、Automation、缺陷分析與 AI 整合經驗|
|3|Senior ML Engineer – Computer Vision|影像資料、模型、Evaluation、Cloud Pipeline 可以轉化為優勢|
|4|Senior Multimodal / Vision-Language AI Engineer|可以在既有 CV 能力上擴充 VLM 與 Multimodal|
|5|Senior Applied AI / LLM Engineer|系統架構與 AWS 經驗可轉移，但還需補 RAG／Agent／LLM Evals|
|6|LLM Foundation Model Research Engineer|通常需要更多專門的大模型訓練與研究經驗|

這只是以已知技術經驗評估的方向適配度，不是對你目前履歷、研究成果或職級的正式評估。

### 特別值得發展的方向：CV + VLM + Multimodal Perception

例如 OpenAI 的 [Machine Learning Engineer – Multimodal Perception and Authentication](https://openai.com/careers/machine-learning-engineer-multimodal-perception-and-authentication-san-francisco/) 職缺，就涉及 Real-world Sensors、Specialized Perception Models、Multimodal Models、Robustness、Real-time Integration 與 Evaluation。

![](https://www.google.com/s2/favicons?domain=https://openai.com&sz=32)

OpenAI

它的產品領域與你的手錶辨識系統不同，但所需的工程能力有不少交集。

如果要讓你的履歷對這類職位更有競爭力，我會建議特別強化三項能力：

1. 深入掌握 PyTorch Model Development： 能獨立修改、訓練、評估 Segmentation、Detection、Anomaly Detection Models，而不只是整合現有模型。
    
2. 增加 Vision Transformer / VLM 經驗： 實際做一個 CLIP、DINO 或 VLM 相關專案，並與傳統 CV 做比較。
    
3. 準備 Production AI 的量化成果： 例如 Image Quality 提升多少、Inference Latency 降低多少、False Positive Rate 改善多少、系統可靠性如何驗證。所有數字必須來自實測。
    

尤其是 Staff Engineer 職位，你需要證明自己不只是整合系統，而是曾經主導重要的架構決策，並讓其他工程師或團隊採用。

## 十二、建議的 12 週面試準備計畫

以下以 Senior CV / ML Engineer 為主要目標，同時保留轉向 Multimodal／Applied LLM 的可能性。

|週次|準備重點|驗收目標|
|---|---|---|
|Week 1–2|Python Coding、Data Structures、LeetCode|20–30 題精選題，能清楚解釋複雜度|
|Week 3–4|Classical Vision、Image Processing、Camera、Optics|能口頭解釋 15–20 題核心知識|
|Week 5–6|PyTorch、U-Net、YOLO、ViT、Model Training|獨立完成訓練、評估、Error Analysis|
|Week 7|Model Optimization、ONNX、TensorRT、GPU|能解釋 Profiling 與部署 Tradeoffs|
|Week 8–9|ML System Design、Cloud、MLOps|完成 3 個 End-to-End Design 練習|
|Week 10|VLM、Vision Foundation Models、Multimodal|實作一個 VLM PoC 和評估方法|
|Week 11|Behavioral、Project Deep Dive|準備 6–8 個 STAR 案例|
|Week 12|Mock Interview、Resume、Coding Review|完成至少 3 次模擬面試|

如果你決定主要投 LLM 職位，則應把 Week 3–6 的傳統影像課程，改成 Transformer、RAG、Agents、Fine-tuning 與 LLM Evaluation。

## 十三、可以直接參考的官方招聘與面試資料

這些是本次查核的原始來源，可以用來對照目標職位：

|官方資料|用途|
|---|---|
|[Apple – Computational Photography / CV ML Engineer](https://jobs.apple.com/en-il/details/200634411-0836/computational-photography-computer-vision-machine-learning-engineer-camera-photos?team=SFTWR)|Imaging、Camera Pipeline、Deep Learning|
|[Intuitive – Senior ML Engineer](https://careers.intuitive.com/en/jobs/744000153756279/JOB218889/senior-machine-learning-engineer/)|CV、Robotics、VLM、Production|
|[Anthropic – Careers / How We Hire](https://www.anthropic.com/careers)|Research / Engineering 職位與面試政策|
|[Amazon – Applied Scientist Interview Prep](https://amazon.jobs/content/en/how-we-hire/applied-scientist-interview-prep)|Scientist 面試流程與能力項目|
|[Amazon – Senior SDE III Interview Prep](https://www.amazon.jobs/content/en-gb/how-we-hire/sde-iii-interview-prep)|Senior Coding、System Design、Leadership|
|[OpenAI – Applied AI Engineer](https://openai.com/careers/applied-ai-engineer-enterprise-san-francisco/)|Production LLM、RAG、Agents、Evals|
|[OpenAI – Multimodal Inference Engineer](https://openai.com/careers/software-engineer-inference-multi-modal-san-francisco/)|GPU、LLM Inference、Distributed Systems|

### 最後的求職策略

若你的目標是在美國取得 Senior 以上 AI 工程師職位，我會建議把 Computer Vision / Imaging / Production ML 當作主要競爭力，再把 VLM / Multimodal AI 當作第二專長。

這比完全重新轉向 LLM Pretraining Research，更能延續你已經累積的實際系統開發經驗。

同時，LeetCode 不需要像完全沒有工程經驗的求職者一樣毫無方向地大量刷題，但也不能放棄。最理想的準備方式是：Coding 基礎足以穩定通過技術篩選，專業深度和 System Design 則用來展現 Senior／Staff 等級的價值。