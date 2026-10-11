
|                           |     |
| ------------------------- | --- |
| [[#### ML System Design]] |     |
|                           |     |
|                           |     |

#### ML System Design
```
請深入詳細回答ML System Design：Senior 以上的核心面試請以具體例子深入完整回答考題可能是：

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
```

# ML System Design：Senior / Staff AI Engineer 核心面試完整解析

## 面試題目

> Design an end-to-end automated visual inspection system that can detect tiny defects from multiple cameras and continuously improve after deployment.
> 
> 設計一套端到端自動化視覺檢測系統，利用多台相機辨識微小瑕疵，並且在正式部署後持續提升檢測能力。

這是一道非常典型的 Senior / Staff Computer Vision Engineer、Senior Machine Learning Engineer、Applied AI Engineer，以及 ML Systems Architect 面試題。

面試官真正要考的，不只是你能不能訓練一個辨識瑕疵的 CNN、YOLO 或 U-Net，而是你能不能把 Machine Learning、Imaging、Hardware、Distributed Systems、MLOps 和 Production Reliability 整合成一個可實際量產的系統。

我會分成以下層次回答：

1. 如何在面試前 5 分鐘釐清需求並訂定系統目標。
    
2. 如何畫出完整的 End-to-End System Architecture。
    
3. 深入設計 Camera、Lighting、Motion Control、Acquisition、Image Processing。
    
4. 如何選擇 Detection、Segmentation、Anomaly Detection 模型。
    
5. 如何設計 Confidence、Decision Threshold、Missing Evidence 與人工審核。
    
6. 如何讓整個 Pipeline 具備 Fault Tolerance、Recovery 與可追溯性。
    
7. 如何建立 Training Data、Ground Truth、Evaluation 和 Continuous Learning。
    
8. 如何監控 False Negatives、Data Drift 與 Production Performance。
    
9. 如何安全更新 100 台生產設備。
    
10. 如何回答面試官的深入追問，以及哪些回答才真正達到 Staff Engineer 水準。
    

# Part 1. 先建立一個具體的系統案例

為了避免面試回答過於抽象，我們假設一家高精度製造公司，希望建立一套自動化手錶零件視覺檢測機台。

每台設備包含三台高解析度工業相機、可控制的多角度光源、XYZ 移動平台與自動對焦系統。

系統必須找出：

![Electropolishing Stainless Steel: Boost Corrosion Resistance](https://images.openai.com/static-rsc-4/iHwW5A9OBKbCEjpr6g2ljvQ3aFvMaeDZlr0okl9E0TVyzcdNoKtTS9EyVe3_vEmU9MM5yVVmB3Of8wWYb7U-lFWfT-0kaheh1k4SO9F38fmx6Msco8c02ZHZ8QQrbLV5Txos6hyV32XF7EKiawV5WfD-bBIZ9hIXaD8vIEx6QKY?purpose=inline)

Scratch（刮痕）

細長、低對比，在特定光線角度下才明顯

![金属焼付塗装・品質管理事例（その２４） 巣穴のブツ - 金属焼付塗装、(有)フジックス | 埼玉川越](https://images.openai.com/static-rsc-4/DG1gAhf81u8wzbm95dV02b6GPSQPu2mlUPwnBnR7WQ2s88FH4bgEwfN3pJ9jj4jtCb7MUydKe9HNIqFzhNdy6qZNT-OvLQtXH9ET99oaoC7TEZ_2xzzV2T9SjsJ1QoDX67e-fjZvkat8k22s9uF4V7YWAGvvbanKsX3vT2rfgLs?purpose=inline)

Pit / Dent（凹點）

面積很小，需要適當解析度與照明方向

![Flexo Printing Defects: Identification, Causes, and Optimization - Oyang Group](https://images.openai.com/static-rsc-4/E8xvR4CowGW5JrwwQz7raPwnmDOrRDquFtCX5IsKmLI7wn_GOBjVsYEcahGP0KTPi7vqZdImLGOhuMIOQ10-9lDgLIwBffmff8ZVFdS-Y8UHOe0gvQOwkTMCBIgfG8f3A3nlsrk2dhuCj4wXKQ1Bp4hjXnjYVcMUbg5VEWnNpHI?purpose=inline)

Printing Defect（印刷瑕疵）

斷字、缺墨、溢墨、筆畫粗細偏差

![Optimizing the Observation of Screw Defects (Dents, Scratches, Cracks, and Curvature)  | KEYENCE America](https://images.openai.com/static-rsc-4/dCCKGe27Hsnaxvg0XvWCyHMMz46x9KvXVC6q4oyZbnbTLsBjvO3fi5OsevbJd9VwJLU48-CdMz0PnRO4mDVpXY8_vRUei-APX1HQ5aQYHn77TrRnIeAG9dHrVbvsDmTzNtCZPT9Mv-vG8u7zor8n0jJGipkGguI2O9zfPtjdU_g?purpose=inline)

Contamination（污染）

灰塵、油污、纖維等異物，可能與真正缺陷混淆

以上是瑕疵類型的示意參考，並非本案例的實際檢測樣本。

## 1.1 面試一開始應該如何提問？

不要立即開始畫 Model Architecture。

Senior Engineer 應該先釐清 Business Requirements 與 Non-Functional Requirements。

我會問面試官：

- What is the minimum defect size we need to detect?
    
- Are false negatives more expensive than false positives?
    
- What is the expected production throughput?
    
- How many cameras and machines are involved?
    
- Does inference need to run locally or can it run in the cloud?
    
- Can a defective or uncertain part be sent for manual review?
    
- Are labeled defect images already available?
    
- What happens when hardware fails or the network is unavailable?
    

這些問題會直接影響 Camera Resolution、Model Choice、Threshold、Cost、System Availability 和 Deployment Architecture。

## 1.2 假設面試官提供以下需求

以下數字是為了設計練習而設定的假設，不是既有系統的實測表現。

|項目|假設需求|
|---|---|
|Inspection target|精密手錶錶面、金屬外殼|
|Minimum critical defect|30 μm|
|Cameras|3 台高解析度相機|
|Images|每個產品約 80 張|
|Deployment|100 台機台|
|Production volume|每台每月 500 件|
|Inspection cycle|P95 不超過 180 秒|
|Local inference|最後影像完成後 10 秒內|
|Critical defect recall|目標 ≥99.5%|
|False reject rate|目標 ≤2%|
|Operating mode|網路中斷仍可執行本機檢測|
|Human review|支援不確定案例人工審核|

此處必須先說明：99.5% Recall 是設計目標，還不是已驗證的結果。要證明達到這個標準，必須準備足夠且獨立的缺陷測試樣本。

## 1.3 先定義最重要的指標

Critical Defect Recall

# ≥99.5%

重大缺陷被成功攔截的比例

P95 Cycle Time

# ≤180s

95% 的產品於時限內完成

False Reject Rate

# ≤2%

良品被錯誤自動判定為不良品的比例

Inspection Coverage

# 100%

所有必須檢查的區域都有有效證據

注意：Coverage 100% 不等於所有瑕疵都能 100% 辨識，而是所有規定的關鍵區域都必須有符合品質要求的影像或其他經驗證的檢測證據。無法完成檢查的產品不可直接當作良品。

另外，還要區分三種容易混淆的指標：

\[ \text{Recall}=\frac{TP}{TP+FN} \]

\[ \text{Precision}=\frac{TP}{TP+FP} \]

\[ \text{False Negative Rate}=\frac{FN}{TP+FN} \]

例如有 1,000 件真正存在重大瑕疵的產品，系統成功攔截 995 件、漏掉 5 件：

- Recall = 99.5%
    
- False Negative Rate = 0.5%
    

但是實務上，「攔截」可能包含自動判退以及送往人工審核，必須明確定義成功攔截的範圍，不能把人工審核中的產品誤算成已經正確辨識瑕疵。

Staff-level 的第一個重點：把品質指標定義到可以測量、驗證、稽核，而不是只說 Model Accuracy 要很高。

# Part 2. 白板上畫出完整 System Architecture

我會採用 Edge-first Hybrid Architecture（本機優先、雲端協同架構）。

理由是：

- Camera、Lighting、Motion Control 必須與本機硬體直接協作。
    
- 即時品質決策不能依賴 Internet latency。
    
- 原始影像資料量非常大。
    
- 100 台設備需要集中管理 Model Versions。
    
- Training 與大規模歷史分析適合放在 Cloud。
    

## 2.1 完整架構圖

END-TO-END ML INSPECTION SYSTEM

Physical Hardware

Camera A / B / C

Lighting / Trigger

Motion / Autofocus

Edge Acquisition & Processing

Local Machine

Inspection Orchestrator + Durable Queue

Capture + Metadata

Image Quality Gate

Calibration + ROI

Vision Inference Engine

Detection

Segmentation

Anomaly Model

Evidence Fusion + Decision Policy

PASS

FAIL

REVIEW

Local Image Store + DB + Audit Log

Async Sync • Metadata • Selected Images • Model Updates

Cloud Data & MLOps

Object Storage / S3

Metadata / Analytics

Human Annotation / Review

Dataset Versioning

Training / Evaluation

Model Registry

Validation → Approval → Canary → Fleet Rollout → Rollback

Continuous Learning Feedback Loop

這張圖最重要的設計不是哪個 Framework，而是分清楚：

Data Plane：真正執行 Capture → Inference → Decision 的生產路徑。

Control Plane：控制 Model Release、設備設定、權限、版本及部署策略。

Learning Plane：收集資料、建立 Ground Truth、重新訓練、驗證新模型。

三條路徑應該能協同運作，但不可以互相造成不必要的阻塞。

例如 Cloud Model Training 故障，不應讓現場機台無法繼續使用上一版已驗證的模型檢測產品。

## 2.2 面試時可以這樣介紹架構

> I would design an edge-first inspection system with a clear separation between acquisition, inference, decision-making, and continuous learning.
> 
> Each machine would have a local orchestrator coordinating cameras, lighting, motion stages, and GPU inference. Images must pass a quality gate before inference, and the decision engine must explicitly handle missing or low-quality evidence.
> 
> The cloud would manage dataset versioning, training, fleet monitoring, human review, and model releases. Production machines should remain operational when disconnected from the cloud, using the last validated model package.
> 
> I would optimize the system for defect escape risk, inspection coverage, recoverability, and auditability, not just model accuracy.

這段可以作為面試最初約兩分鐘的 High-level Architecture Explanation。

接下來面試官通常會選一個環節往下追問。

# Part 3. Camera / Lighting / Motion Control

這一部分通常可以很快區分「只懂 Machine Learning」與「真正做過 Production Computer Vision」的工程師。

因為對微小瑕疵檢測而言，最大的瓶頸不一定是 Model，而可能是相機根本沒有擷取到足夠的瑕疵資訊。

## 3.1 Camera Resolution：首先用物理尺寸推算

面試官可能問：

> How do you decide camera resolution for a 30-micrometer defect?

我不會只回答「使用 20MP Camera」。

我會從 Object-space Pixel Resolution 開始計算。

假設：

- Camera resolution = 4512 × 4512 pixels
    
- Field of View = 12 mm × 12 mm
    
- Minimum defect size = 30 μm
    

則每個 Pixel 對應的實際物理尺寸：

\[ r=\frac{\text{FOV width}}{\text{Image width}} \]

\[ r=\frac{12,000\ \mu m}{4512} \approx2.66\ \mu m/pixel \]

因此，一個 30 μm 的瑕疵在影像中大約是：

\[ N=\frac{30}{2.66}\approx11.3\ pixels \]

不同 Field of View 的解析能力比較

固定 4512 pixels 寬，瑕疵尺寸 30 μm

|   |   |   |
|---|---|---|
|FOV|μm / px|瑕疵寬度|
|40 mm|8.87|3.4 px|
|24 mm|5.32|5.6 px|
|12 mm|2.66|11.3 px|
|6 mm|1.33|22.6 px|

這是幾何取樣估計；並非保證可分辨或辨識瑕疵。實際效能仍取決於光學 MTF、對比、雜訊、景深和照明。

這裡要向面試官強調三個概念。

Spatial Sampling

一個瑕疵有多少 pixels，可以決定模型有多少數位資訊可以使用。

Optical Resolution

即使一個瑕疵在數位上橫跨 10 pixels，如果鏡頭模糊、景深不足或繞射限制，細節仍可能不存在。

Signal-to-Noise Ratio（SNR）

瑕疵和正常材料的對比必須大於影像雜訊，否則模型難以穩定辨識。

因此 10 pixels 不是通用物理定律，而是我們為本案例設定的初步設計基準，仍需使用實體瑕疵樣本驗證。

### 不能忽略 Field of View 與 Coverage 的 Tradeoff

FOV 越小，單一瑕疵取得的 pixels 越多，但是需要拍攝的區域越多。

假設錶面有效檢測範圍為 40 mm × 40 mm：

- 單張 FOV 12 mm × 12 mm
    
- 相鄰拍攝位置需要 20% overlap
    
- 每個方向需要約 4 個拍攝位置
    
- 總計約 16 個 Tiles
    

這就形成了另一個系統設計問題：

Resolution ↑ → Number of Captures ↑ → Motion Time ↑ → Storage ↑ → Inference Cost ↑

Staff Engineer 必須知道這是一個多目標最佳化問題，而不是單純追求最高解析度。

## 3.2 如何設計多相機分工？

我會把三台 Camera 設計成不同職責，而不是三台全部拍一樣的影像。

|Camera|職責|典型瑕疵|
|---|---|---|
|Camera A：Overview|大範圍定位、幾何校正、ROI 決定|缺件、位置偏移、較大缺陷|
|Camera B：High-resolution Macro|重要區域的高解析度檢測|刮痕、凹點、缺墨|
|Camera C：Oblique / Detail|不同角度、不同材質的表面檢測|反光缺陷、側面缺陷|

這樣設計的好處是 不要浪費高解析度 Camera 的時間去檢查不需要高解析度的區域。

例如：

Camera A 先找到錶面上的所有字元位置和特殊區域。

Camera B 再依據已知的幾何位置進行高解析度拍攝。

Camera C 專門處理難以從正面看見的刮痕或立體結構。

這就是 Hierarchical Inspection。

不過對 Critical Defects，我不會僅依賴 Overview Model 的初步判斷才決定是否拍攝。否則 Overview 漏掉的瑕疵就永遠不會進入下一階段。

關鍵區域仍應有強制的 Coverage Plan。

## 3.3 Lighting 為什麼可能比 Model Architecture 更重要？

考慮金屬錶殼上的一道 30 μm 刮痕。

![ViTiny UM02 USB handheld microscope for PC Mac and Android Phone tablet Vividia UM02-T – Oasis Scientific](https://images.openai.com/static-rsc-4/X34Q_7ZHs8aaAV7uUPH04w1_QOjZKS4-DUilxNQTPa6K6xyJQB9FDjfTJPGpuKDWhKfvMS51Gb6ELtM84Y-YeBa5k8cH8VWUM5Uk5l_6dn_JWXK9qswPAO-pjCNv51NTuZexuh7VDkiNA82m-ahPP1KubJNvGwbnbDAWZ5--s1w?purpose=inline)

Bright-field

適合檢測印刷、顏色、紋理等，但高反光表面容易產生眩光。

![Inverses Auflichtmikroskop für die Untersuchung von Klingen - Seite 3](https://images.openai.com/static-rsc-4/GYnL76NWDRO-9yDha-yZq-ZcErxoEdHKm39lIJHlDkrNccCkf0b-PnSvSncpcUOxo6XX4lKStTzRz88mngNcfyiYArIGXtVOv4sd74BJ4DATFa9RwrL9EBB79FF7Oifbx1zPpuHXSByEKPUqTqnE7Lp8JU5kU3HV0kgfBSFRsjQ?purpose=inline)

Dark-field

讓部分表面散射、細微刮痕變得更明顯，對某些低對比缺陷特別有效。

我會考慮以下照明模式：

|Illumination|用途|
|---|---|
|Coaxial light|平面與印刷檢測|
|Ring light|一般表面觀察|
|Low-angle dark-field|微小刮痕、邊緣與凹點|
|Cross-polarized light|某些材料的反光抑制|
|Multi-directional light|不同方向缺陷的顯現|
|HDR exposure|高動態範圍場景|

假設某個刮痕在 Light Angle 0° 幾乎看不到，但在 45° 時具有很明顯的對比。

那麼增加相機解析度，可能還不如增加正確的光源方向來得有效。

2026 年實際的精密手錶機芯瑕疵檢測方案，也已使用 BRDF / Reflectance-based Computational Imaging，藉由不同照明角度呈現一般固定光源不容易顯現的微小表面特徵。

![](https://www.google.com/s2/favicons?domain=https://opto-gmbh.com&sz=32)

Opto

### Staff-level 的設計選擇

我會把 Light Profile 視為 Model Input Specification 的一部分。

例如：

```
inspection_profile:
  region: dial_text
  camera: macro
  illumination:
    mode: coaxial
    intensity: 0.65
  exposure_us: 120000
  autofocus: enabled
  calibration_version: calib_v12
  preprocess_version: pp_v5
```

一旦光源改變，就可能產生 Domain Shift。

因此我不允許現場工程師任意改變 Lighting Settings，然後期待既有 Model Accuracy 完全不受影響。

## 3.4 Motion Control、Synchronization 與 Autofocus

多相機系統不是只有三條 Camera Threads 就足夠。

Camera Trigger、Light Controller、XYZ Stage、Rotary Stage、Autofocus 之間也有時序依賴。

一個拍攝點應該經過：

Move Stage to Target Position

Wait for Motion Settling + Verify Encoder

Autofocus + Focus Quality Check

Apply Light Profile / Exposure

Trigger Camera + Receive Frame

Validate Frame + Persist Capture Record

### 面試官追問：How do you synchronize three cameras?

需要先問三台相機是否真的必須在同一瞬間曝光。

如果拍攝的是靜止物體，且各 Camera 使用不同光源，未必需要嚴格的同時曝光。Sequenced Capture 反而可以避免光源相互干擾。

如果是移動中的物體，需要同步拍攝，我會使用：

- Hardware Trigger：由同一個控制器同步觸發。
    
- PTP（Precision Time Protocol）：校準相機時鐘。
    
- Scheduled Action Command：在支援的 GigE Vision Camera 上指定曝光觸發時間。
    
- Timestamp / Frame ID / Trigger ID：驗證是否屬於同一次 Capture。
    

Basler 的 GigE Vision 多相機技術文件也說明 PTP 與 Scheduled Action Commands 可用於多相機同步，但實際硬體必須支援，且需驗證交換器與網路配置。

![](https://www.google.com/s2/favicons?domain=https://www.baslerweb.com&sz=32)

Basler Inc.

+1

### Autofocus 不是單純找最大 Laplacian Variance

假設鏡頭看到有大量金屬細紋的錶面。

Autofocus 很可能對焦在對比最高的背景紋理，而不是我們真正想檢查的印刷或刮痕所在平面。

我會設計：

- ROI-based Focus Measure
    
- Coarse-to-fine Focus Search
    
- Z Position Boundaries
    
- Focus Score Threshold
    
- Peak Sharpness Check
    
- Focus Failure Retry
    
- Independent Image Quality Verification
    

重點是把 Autofocus Success 和 Defect Detection Success 分開。

Autofocus 回傳成功，不代表影像一定可供模型使用。

# Part 4. Acquisition / Quality Check / Image Processing

## 4.1 Acquisition 應該設計成有狀態的 Workflow

一件產品開始檢測後，應產生唯一的 `inspection_id`。

假設：

```
inspection_id = INS-20261011-000042

machine_id = MACHINE-017
product_id = WATCH-005012
recipe_version = inspection_recipe_v8
```

其中 `recipe_version` 很重要。

它代表這件產品預期要拍哪些影像、用哪些相機、哪些角度、哪些光源與位置。

例如原本應有 80 張影像，系統最後只收到 78 張。

系統必須知道是哪兩張不見了，而不是只檢查整個資料夾內有幾張檔案。

我會建立 Capture Manifest。

|Capture ID|Camera|Region|Status|
|---|---|---|---|
|CAP-001|A|Full Dial|Valid|
|CAP-002|B|Dial Text 1|Valid|
|CAP-003|B|Dial Text 2|Blur — Retry|
|CAP-004|C|Bezel Side|Valid|
|CAP-005|C|Case Edge|Missing|

每張影像記錄：

```
{
  "inspection_id": "INS-20261011-000042",
  "capture_id": "CAP-003",
  "camera_id": "macro_b",
  "region_id": "dial_text_2",
  "trigger_id": "TRG-003",
  "status": "QC_FAILED",
  "quality_reason": "BLUR",
  "model_version": null
}
```

這是重要的設計決策：Acquisition 必須是可驗證、可追蹤的業務流程，而不只是將相機 Callback 裡的 Array 儲存成 PNG。

## 4.2 Image Quality Gate

面試官可能問：

> What if the model receives a blurry image?

不應期待 Deep Learning Model 自己解決所有影像品質問題。

我會在 Model Inference 之前建立 Image Quality Gate。

|品質項目|檢測方法|可能動作|
|---|---|---|
|Blur|Laplacian / Tenengrad / Learned IQA|Refocus + Recapture|
|Overexposure|Saturated Pixel Ratio|調整曝光|
|Underexposure|Histogram / Local SNR|調整光源或曝光|
|Wrong ROI|Template / Landmark Matching|重新定位|
|Motion blur|Sharpness + Stage State|等待穩定|
|Missing frame|Capture Manifest / Trigger ID|Retry|
|Color drift|Reference Target / Color Statistics|校正與告警|
|Glare / Occlusion|Saturation Mask / Visibility Model|換光源、角度或 Review|

注意：對錶面深色字元而言，整張圖片平均亮度低，並不一定代表曝光不足。因此 QC 必須具備 Region-specific Criteria。

### 設計 Quality Score

可以建立：

\[ Q_{\text{image}}=f(S,B,E,C,R) \]

其中：

- \(S\)：Sharpness
    
- \(B\)：Brightness / Exposure
    
- \(E\)：有效區域曝光品質
    
- \(C\)：Contrast / Visibility
    
- \(R\)：ROI Completeness
    

但是我不會只把所有項目加權成一個數值後就決定通過。

例如影像 Sharpness 非常高，但 ROI 完全錯誤。

高 Sharpness 不應補償錯誤的 ROI。

對必須滿足的條件，應使用 Hard Constraints：

\[ Q_{\text{pass}}= Q_{\text{focus}} \land Q_{\text{exposure}} \land Q_{\text{ROI}} \land Q_{\text{metadata}} \]

任何關鍵條件失敗，都不能直接進入 Auto-pass 決策。

## 4.3 Image Preprocessing 應該如何做？

我會把 Preprocessing 分成三層。

### 第一層：Physical / Photometric Correction

用來降低成像系統造成的非理想變化。

例如：

- Lens distortion correction
    
- Dark-frame / Flat-field correction
    
- Color calibration
    
- Fixed-pattern noise correction
    
- Validated illumination normalization
    

### 第二層：Geometric Normalization

讓同類型零件對齊，方便模型比較。

例如：

- Rotation alignment
    
- Landmark detection
    
- ROI extraction
    
- Template registration
    
- Pixel-to-mm coordinate mapping
    

### 第三層：Model-specific Transformation

例如：

- Resize / Crop / Tiling
    
- Normalization
    
- Channel conversion
    
- Tensor formatting
    

必須特別注意，對微小瑕疵而言，某些 Image Enhancement 可能改變真正的 Defect Appearance。

例如 Aggressive Denoising 可能把很細的刮痕消除，而 Sharpening 也可能製造假的邊緣。

所以我會保存 Raw Image，並將 Preprocessing Version 作為模型版本的一部分。

# Part 5. Detection / Segmentation / Feature Extraction

這是 CVAI Engineer 最熟悉的部分，但 Staff Engineer 不應一開始就直接選擇最複雜的模型。

## 5.1 Model Selection：該選 YOLO、U-Net、Transformer 還是 Anomaly Detection？

我會根據問題性質選擇不同模型。

|任務|候選方法|選擇原因|
|---|---|---|
|零件與 ROI 定位|YOLO / RT-DETR|找出位置和 Bounding Box|
|微小刮痕|U-Net / SegFormer|需要 Pixel-level Mask|
|印刷缺損|Template Difference + Segmentation|有明確正常幾何|
|幾何尺寸異常|Classical CV + Measurement|具可解釋的物理量|
|未知瑕疵|PatchCore / PaDiM|不一定有已知缺陷標籤|
|多影像整合|Evidence Fusion Model|需要結合不同 View|

我的首選會是 Hybrid CV Pipeline，而不是單一 End-to-End 巨型模型。

原因是有些問題非常適合 Deep Learning，有些問題其實更適合傳統影像處理與物理量測。

例如檢查兩個字元之間是否有 0.1 mm 的間距偏差，如果 ROI 幾何校正準確，用 Character Segmentation + Geometric Measurement 可能比完全依靠 CNN 更容易驗證與解釋。

## 5.2 對 Tiny Defects，最大的陷阱是 Downsampling

假設前面設計的 30 μm 瑕疵在原始 4512 × 4512 影像內佔 11.3 pixels。

如果直接把整張圖片 Resize 成 640 × 640：

\[ 11.3\times\frac{640}{4512}\approx1.6\ pixels \]

只有大約 1.6 pixels。

這個瑕疵的辨識資訊可能已經在 Image Resizing 階段流失。

因此如果有人回答：

> I will resize all images to 640×640 and run YOLO.

面試官很可能接著問：

> But your defect is only 30 micrometers. Can your model still detect it?

這時候你應該提出 High-resolution Tiling。

High-resolution Tiling：概念示意

示意圖：將局部區域保留原始細節送入模型，而非把整張高解析影像大幅縮小。

### Tiling 的實作方式

假設原始圖片是 4512 × 4512。

可以選：

- Patch Size = 1024 × 1024
    
- Overlap = 128 pixels
    
- Stride = 896 pixels
    

如此每個方向約需要 5 個 Tiles，總計約 25 個 Tiles。

模型處理 1024 × 1024 的局部影像，保留小瑕疵的原始相對像素尺度。

但必須處理新問題：

Boundary Defects

某一道刮痕可能剛好橫跨兩個 Patch。

所以需要 Overlap，並將結果轉換回 Global Image Coordinates，再使用 NMS、Mask Merging 或其他適當的去重方法。

Compute Cost

25 個 Tiles 比一張 Resize Image 需要更多計算，因此還需要 Batch Inference、ROI Priority 與 GPU Profiling。

Dataset Leakage

由同一張影像切出的 Tiles，絕對不能隨機分配到 Training 和 Testing。

這是後面資料集設計會詳細說明的重要問題。

## 5.3 Known Defect + Unknown Anomaly 雙路徑架構

我會設計：

```
High-resolution Image / ROI
           |
           +------------------------+
           |                        |
           v                        v
    Supervised Model          Anomaly Model
    ----------------          -------------
    Scratch                   Unknown texture
    Dent                      Unusual pattern
    Printing defect           Novel defect
    Contamination             Out-of-distribution
           |                        |
           v                        v
      Defect Masks            Anomaly Map
      Defect Scores           Anomaly Scores
           |                        |
           +-----------+------------+
                       |
                       v
                Evidence Fusion
                       |
                       v
                  Decision
```

### 為什麼需要兩種模型？

Supervised Model 擅長已知的 Defect Types。

例如訓練資料包含大量 Scratch，模型通常可以學習 Scratch 的形狀、顏色、對比與紋理。

但是如果 Production 突然出現 Training Dataset 裡面從來沒有的新型瑕疵，Supervised Model 可能完全忽略。

Anomaly Detection 可以補足這個問題。

例如使用 PatchCore，從正常產品建立 Embedding Reference Bank，判斷新影像的局部特徵是否偏離正常分布。

但要注意：

Anomaly Score 不是 Defect Probability。

一個新的正常材料紋理，也可能得到很高的 Anomaly Score。

因此我的設計是：

- Known Defects → Supervised Classification / Segmentation
    
- Unfamiliar Patterns → Anomaly Detection → Human Review
    
- Critical New Defect → 人工確認後納入 Dataset 和後續模型訓練
    

Anomaly Model 很適合擔任 Safety Net，但不能在未驗證下直接把所有高 Anomaly Score 判定為瑕疵。

## 5.4 Feature Extraction：為什麼不只輸出 Bounding Box？

以刮痕為例，除了瑕疵類型與位置，我還想知道：

- Scratch Length（μm）
    
- Scratch Width（μm）
    
- Scratch Area（mm²）
    
- Position / Region
    
- Distance to Critical Feature
    
- Contrast / Severity
    
- Detection Confidence
    

假設 Segment Model 得到 Scratch Mask。

可以利用 Camera Calibration 將 pixels 轉換成物理尺寸。

如果 Pixel Resolution = 2.66 μm/pixel，刮痕 Skeleton Length = 70 pixels：

\[ L=70\times2.66=186.2\ \mu m \]

這樣 Decision Engine 就可以用明確的產品規格：

例如在特定錶面區域，Scratch Length 大於 150 μm 判定為不符合品質規範。

這個 150 μm 是示範值，真正標準應由 Quality Engineering 或產品規格決定。

相較於模型只輸出 `scratch_probability = 0.93`，提供實際尺寸和位置通常更容易讓 Manufacturing Engineer 及 QA 團隊理解。

這就是從 Image Classification 提升到 Explainable, Measurement-aware Inspection。

## 5.5 模型效能與 Latency 最佳化

面試官可能會問：

> Your model is accurate, but inference takes 20 seconds. How would you reduce it to 5 seconds?

我的第一個動作會是 Profiling，而不是立即 Quantization。

我會先量測：

|Stage|假設耗時|
|---|---|
|Image loading / decode|2.0 s|
|Tiling / preprocessing|3.0 s|
|CPU → GPU transfer|1.5 s|
|GPU inference|10.0 s|
|Postprocessing|3.5 s|
|Total|20.0 s|

然後按照瓶頸最佳化。

如果 10 秒都花在 GPU Inference：

可考慮 TensorRT、FP16、Batching、適當的 ROI Reduction。

如果主要瓶頸是 Image Loading：

應先改進 Image Decode、Pinned Memory、資料傳輸與 Pipeline Overlap。

如果是 Postprocessing：

可以優化 Mask Merging、NMS 與重複計算。

INT8 Quantization 可作為進一步選項，但必須確認小瑕疵的 Recall 不下降。NVIDIA TensorRT 官方也建議針對 Precision Reduction 造成的 Accuracy Loss 進行比較、使用具代表性的 Calibration Data，必要時採用 Mixed Precision。

![](https://www.google.com/s2/favicons?domain=https://docs.nvidia.com&sz=32)

NVIDIA TensorRT

+1

Senior Engineer 要能做 Profiling。

Staff Engineer 還需要決定：增加一張 GPU、改變光學方案，或優化 Capture Plan，哪個方案最符合成本與品質需求。

# Part 6. Decision / Confidence / Anomaly Handling

這是非常容易被忽略、卻常常是面試官最重視的部分。

因為 Production Inspection 最終需要輸出的是業務決策，不是 Model Prediction。

## 6.1 Model Prediction 不等於 Final Decision

假設某台相機的模型輸出：

```
{
  "defect_type": "scratch",
  "score": 0.87,
  "length_um": 186.2,
  "region": "dial_outer_ring"
}
```

模型只能說它找到某個可能的瑕疵。

但是系統還需要判斷：

- 這個 Region 是否允許刮痕？
    
- 刮痕長度是否超過規格？
    
- 該瑕疵是否被另一個角度的影像確認？
    
- 影像品質是否可靠？
    
- 是否還有其他未檢查的 Critical Regions？
    
- 是否遇到未知的 Anomaly？
    

所以我會把 Model 與 Decision Policy 分開。

```
Model Inference
      |
      v
Defect Evidence
      |
      v
Geometry / Severity / Visibility
      |
      v
Evidence Fusion
      |
      v
Quality Policy
      |
      v
PASS / FAIL / REVIEW / INCOMPLETE
```

這裡我會使用四種內部狀態，而不是只使用 Pass / Fail。

|狀態|意義|業務動作|
|---|---|---|
|PASS|所有必要檢查都完成，且風險低|自動放行|
|FAIL|有足夠證據確認不符合品質要求|判退|
|REVIEW|有不確定或未知瑕疵|人工複核|
|INCOMPLETE|必要影像遺失或品質不足|重新檢測或隔離|

如果 Production System 只接受三種輸出，可以把 `INCOMPLETE` 映射到隔離與 Review 流程，但不能直接映射到 PASS。

## 6.2 何謂 Model Confidence？

面試官可能問：

> If your model reports 99% confidence, can you automatically pass the part?

不能。

第一個原因是 Softmax / Sigmoid Score 不一定是真正的機率。

第二個原因是即使 Classifier 對某個已知類別很有信心，也不代表影像沒有其他未知缺陷。

第三個原因是模型對 Blurry Image 或 Domain-shifted Input 仍可能輸出高 Confidence。

所以必須區分：

Predictive Confidence：模型對某個預測有多少信心。

Calibration：模型輸出的分數和真實事件頻率是否一致。

Image Quality：輸入的影像是否足夠可靠。

Coverage：所有重要區域是否都完成檢查。

Out-of-distribution Risk：影像是否已超出模型原本驗證的資料分布。

最終的 Auto-pass 應該符合所有必要的品質與風險條件。

## 6.3 如何設定 Decision Threshold？

我會使用 Cost-sensitive Decision Making。

假設：

- False Negative：重大瑕疵被放行，成本 $2,000
    
- False Positive：良品被錯誤判退，成本 $50
    

先考慮最簡單的二元決策：

- Reject
    
- Accept
    

如果 \(p=P(\text{Defective}\mid x)\) 是已校準的瑕疵後驗機率：

\[ \text{Expected Cost of Accept}=p C_{FN} \]

\[ \text{Expected Cost of Reject}=(1-p)C_{FP} \]

當 Reject 成本較低時，就選擇 Reject：

\[ (1-p)C_{FP}<pC_{FN} \]

整理：

\[ p>\frac{C_{FP}}{C_{FP}+C_{FN}} \]

代入：

\[ p>\frac{50}{50+2000}\approx0.0244 \]

意思是：在這個簡化成本模型下，只要瑕疵後驗機率超過約 2.44%，拒收的期望損失就比接受更低。

這顯示為什麼 False Negative 成本非常高時，Threshold 可能遠低於 0.5。

但這個例子並不代表真正的工廠應該採用 0.0244。

真實系統還包括 Manual Review、誤判成本估計、模型校準誤差、不同瑕疵嚴重度、產能限制與統計不確定性。

### 三段式 Threshold

更實用的策略是：

Defect Probability：示意 Threshold

0.0

0.1

0.9

1.0

Low Risk

0–0.1 可考慮 PASS

Uncertain

0.1–0.9 REVIEW

High Risk

0.9–1.0 可考慮 FAIL

僅示意三段式政策；0.1 與 0.9 並非經實驗驗證的最佳數值，而且 PASS 仍需額外通過品質、Coverage、Anomaly 等條件。

Threshold 的正確選擇方式是：

在獨立的 Validation Set 上做 Threshold Sweep，觀察 Defect Recall、False Reject Rate、Review Rate 和 Cycle Time 的變化，選出符合 Business Constraints 的運作點。

例如：

\[ \min_{\tau} \text{Expected Operating Cost}(\tau) \]

Subject to：

\[ \text{Critical Defect Recall}(\tau)\ge99.5\% \]

\[ \text{False Reject Rate}(\tau)\le2\% \]

\[ \text{Review Rate}(\tau)\le R_{\max} \]

這是一個受品質與作業能力限制的最佳化問題。

若不存在任何 Threshold 同時符合上述條件，就要改善成像、模型、資料或作業流程，而不是硬選一個 Threshold 宣稱合格。

## 6.4 Multi-camera Evidence Fusion

面試官可能問：

> If Camera A says 80% defective, but Camera B says 20%, what do you do?

首先要確認兩台 Camera 是否在觀察同一個 Defect、同一個區域。

如果它們拍的是不同區域，就不應該直接平均。

如果它們觀察同一個實際缺陷，我會使用：

- Camera Calibration
    
- Coordinate Mapping
    
- Defect Association
    
- Per-view Visibility
    
- Quality-aware Evidence Fusion
    

例如某個 Scratch 在正面光線下不明顯，但在 Oblique Light 下十分清楚。

Camera A 的低分數不一定應該降低 Camera B 的高分數，因為 Camera A 可能本來就無法可靠觀察這種瑕疵。

此外，多張影像可能有高度相關性。

不能直接假設三台 Camera 的 Evidence 相互獨立，然後把三個 Probability 相乘。

較好的設計是將 Per-view Features、Visibility、Quality Metrics 與 Defect Type 輸入一個經驗證的 Fusion Model，或使用明確且可解釋的 Rule-based Fusion。

對於高度安全敏感的重大瑕疵，我會優先採取保守決策：有可信的重大缺陷證據就應攔截或複核，而不是被其他相機的低分數平均掉。

## 6.5 Decision Engine 範例

下面是可以在面試白板直接解釋的簡化 Pseudocode。

```
def inspect_part(manifest, model_outputs, policy):    # Required evidence must exist    if not manifest.required_views_complete():        return "INCOMPLETE"    # Image quality must pass    if not manifest.all_required_views_quality_pass():        return "INCOMPLETE"    # Detect previously unseen patterns    if model_outputs.high_anomaly_risk():        return "REVIEW"    # Critical defect evidence    if model_outputs.confirmed_critical_defect():        return "FAIL"    # Fuse all observations    risk = model_outputs.fused_defect_risk()    # Risk-based decision    if risk >= policy.reject_threshold:        return "FAIL"    if risk >= policy.review_threshold:        return "REVIEW"    return "PASS"
```

實際實作還要加入：重拍次數、模型版本、已驗證的 Rule Sets、量測公差、Defect Severity、物件定位一致性，以及系統或硬體故障時的隔離狀態。

核心原則：Low Confidence、Missing Evidence、Hardware Failure 和 No Defect 是四種不同的狀況。

不能全部當成 PASS。

# Part 7. 如何讓 Pipeline 具備 Recoverability、Fault Tolerance 與 Backpressure？

這是從 Senior CV Engineer 進入 Senior / Staff ML Systems Engineer 最重要的分界之一。

面試官很可能會說：

> Your cameras and models work under normal conditions. But what happens when the system crashes halfway through an inspection?

這時候不能只回答「我會使用 try-except 和 retry」。

必須設計真正具備 Failure Recovery 的 Production Pipeline。

## 7.1 不要把整個流程寫成一個大型 Python Function

例如：

```
def run_inspection():    capture_all_images()    process_images()    run_models()    make_decision()    upload_results()
```

這個版本有幾個問題。

如果已經拍完 70 張照片，程式突然 Crash，重新啟動後怎麼辦？

如果已完成 Inference，但 Upload Cloud 失敗，是否要重新拍照？

如果 Decision 已經下達到 PLC，但系統在記錄結果前 Crash，是否會對同一件產品重複操作？

這些都不能靠一個 `try-except` 解決。

## 7.2 使用 Durable Workflow State Machine

我會讓每個 Inspection 都有持久化狀態。

CREATED

建立 Inspection ID 和 Capture Manifest

ACQUIRING

擷取必要影像並記錄每張狀態

QUALITY_CHECK

驗證各張影像與 Coverage

READY_FOR_INFERENCE

必要證據已符合要求

INFERENCING

執行模型並持久化 Evidence

DECIDING

套用 Policy 和 Threshold

DECIDED

決策結果已寫入本機 DB

SYNC_PENDING

等待上傳 Cloud / Review

這只是主要成功路徑。

實際 State Machine 還需要 `RETRY_PENDING`、`WAITING_FOR_OPERATOR`、`QUARANTINED`、`FAILED_TERMINAL` 等異常狀態。

另外 Cloud Synchronization 和 Human Review 並不一定是單一路徑上的下一個步驟，我會把它們做成相對獨立的子流程。

例如 `DECIDED` 可以是本機決策完成的最終狀態，而 `SYNC_PENDING` 則是另一個同步任務的狀態。

這樣 Cloud 故障就不會阻止本機完成品質檢測。

## 7.3 Retry、Checkpoint 和 Idempotency

### Checkpoint

完成某個重要階段後，持久化結果。

例如：

- Capture 完成：儲存 Raw Image 與 Capture Metadata。
    
- QC 完成：儲存 QC Result。
    
- Inference 完成：儲存 Model Predictions。
    
- Decision 完成：儲存 Final Decision。
    

如果程式 Crash，重新啟動後可根據已完成的紀錄繼續。

### Idempotency

Idempotency 是指同一項邏輯操作被重複要求時，不會造成重複的業務效果。

例如同一張圖片不應因 Queue 重送而產生兩筆獨立的 Defect Evidence。

可以使用：

```
inspection_id + capture_id + processing_version
```

作為唯一的處理識別。

同樣地，PLC 執行放行或判退動作時，必須使用 Command ID 和 Ack / Status Reconciliation，避免重試時重複執行機械動作。

### Atomic Persistence

我會採取以下順序：

1. 將 Image 寫入 Temporary File。
    
2. 寫入完整影像並 Flush。
    
3. 在同一個 Filesystem 上做 Atomic Rename。
    
4. 驗證檔案與 Checksum。
    
5. 將最終影像位置及狀態更新至 DB。
    
6. 提交後才視為完成。
    

DB 與 Filesystem 之間沒有天然的單一 Transaction，因此 Crash Recovery 必須處理 Orphan File 和尚未完成登錄的檔案。

對於 Cloud Upload，則可使用 Transactional Outbox Pattern，保證上傳任務不會因主程式結束而遺失。

## 7.4 Queue、Concurrency 與 Backpressure

假設三台相機同時拍攝大量影像。

Camera Capture 速度可能比 GPU Inference 快。

如果所有 Image 都直接加入 Memory Queue：

```
Camera A -----\
Camera B ------> Image Queue ---> GPU Worker
Camera C -----/
```

當 GPU 處理速度小於 Image Arrival Rate，Queue 會持續累積。

這稱為 Backpressure 問題。

以簡化的穩態模型：

\[ \lambda=\text{Arrival Rate} \]

\[ \mu=\text{Processing Rate} \]

若長期存在：

\[ \lambda\ge\mu \]

而 Queue 容量有限，最後必然會面臨 Queue Saturation、記憶體耗盡或必須降低流量。

### 我會如何設計？

Capture Workers

三台 Camera 非同步接收影像

Bounded Memory Queue

Queue Limit + High-water Mark

Durable Image Store + Pending Task DB

避免任務只保存在 RAM

GPU Inference Workers

Batching / Scheduling / Priority

Decision + Results

我會設計：

Bounded Queue：限制記憶體中等待處理的資料量。

Durable Queue：儲存待處理任務的狀態，確保重啟後仍能繼續。

Backpressure Signal：通知 Orchestrator 暫緩新的 Capture 或下一件產品進站。

Worker Pool：依 Camera、CPU、GPU 的資源特性安排 Concurrency。

Priority Scheduling：優先處理會阻塞當前產品最終判斷的 Critical Images。

Dead-letter / Quarantine Queue：持續失敗的任務轉入人工排查，而不是無限重試。

必須避免一個錯誤：Queue 滿了就直接 Drop Images。

對於必要的 Production Inspection Images，不能靜默丟棄。應該讓 Capture 減速、停止接受下一件產品，或將產品移往隔離流程。

### Little's Law

在穩定系統中：

\[ L=\lambda W \]

其中：

- \(L\)：系統內平均等待及處理的工作量
    
- \(\lambda\)：平均到達率
    
- \(W\)：平均停留時間
    

例如平均每秒新增 10 個 Image Processing Tasks，平均停留時間是 3 秒：

\[ L=10\times3=30 \]

也就是平均約 30 個任務正在處理或等待。

這個公式可以幫助初步估計 Queue Capacity，但仍須考慮 Burst、變異、Worst-case Latency 與多種工作負載。

# Part 8. 七個核心追問：從工程實作回答到 Staff Level

## Q1. What if one camera fails?

面試官：

> Suppose Camera B stops working in the middle of inspection. How would your system handle it?

### 不夠好的回答

> We retry the camera connection. If it still fails, we show an error.

這只處理了 Hardware Error，沒有處理 Quality Risk、Workflow State 和 Business Decision。

### Senior / Staff 回答

我會把 Camera Failure 分成幾種：

|Failure Type|偵測方式|恢復策略|
|---|---|---|
|Connection Lost|Heartbeat / SDK exception|Reconnect|
|Frame Timeout|Capture Deadline|Retry Trigger|
|Corrupted Image|Payload / Checksum / Frame Status|Recapture|
|Repeated Low Quality|QC Failure Rate|Calibration / Maintenance|
|Persistent Hardware Failure|Retry Exhausted|Disable Camera + Alert|
|Capture Data Mismatch|Trigger ID / Timestamp|Discard invalid evidence|

然後建立一個 Camera Dependency Graph。

例如：

```
Critical Region: Dial Text

Required evidence:
    Camera B / High-resolution view

Optional corroboration:
    Camera C / Oblique view
```

如果 Camera B 故障：

- 若 Camera C 具有已驗證的等效解析度、Lighting 與 Coverage，可以啟動 Alternate Inspection Recipe。
    
- 如果 Camera C 不能保證 30 μm 瑕疵的可檢性，就不能把 Camera C 當成替代品。
    
- 如果必要檢查無法完成，將 Inspection 設為 `INCOMPLETE`，重新檢測或隔離產品。
    
- 發出設備故障事件，通知 Maintenance。
    
- 將 Camera Failure 計入 Reliability Metrics。
    

### Staff-level 的加分回答

> I would not blindly treat camera redundancy as equivalent inspection coverage. A secondary camera can only substitute for a failed primary camera if we have validated its optical resolution, illumination conditions, and detection performance for the same critical defect categories.
> 
> Otherwise, the system should degrade operationally, not silently degrade quality.

重點是區分：

Service Availability 和 Quality Capability。

設備還能運轉，不代表它還有足夠能力判定良品。

## Q2. How do you handle missing images?

面試官：

> What happens if two out of eighty images are missing?

首先，Missing Images 不只是資料完整性問題，而是 Inspection Coverage 問題。

### 方法一：Required / Optional Evidence Manifest

例如：

```
Capture 001  Full Dial           Required
Capture 002  Dial Text          Required
Capture 003  Case Surface       Required
Capture 004  Decorative Detail  Optional
```

如果 `Capture 004` 缺失，且它確實不是任何必須檢測區域的唯一有效證據，系統可能仍符合放行條件。

但如果 `Capture 002` 缺失，就不能 PASS。

### 方法二：Dependency-based Recovery

假設缺失的是 Dial Text Camera Image：

1. 查詢 Capture Manifest。
    
2. 確認缺少的是哪個 Region。
    
3. 判斷是否已有經驗證的有效替代影像。
    
4. 若沒有，重新移動 Stage 至對應位置。
    
5. 重新調整光源、對焦與拍攝。
    
6. 重新執行 Image QC。
    
7. 在成功取得必要證據後才完成 Decision。
    

重點不是重新跑全部 80 張，而是盡可能針對失敗的步驟做 Recovery。

### 方法三：Missing Evidence-aware Fusion

如果某些 Features 本來就是 Optional，Fusion Model 應支援 Missingness Mask。

例如：

\[ X=[f_A,f_B,f_C] \]

\[ M=[1,0,1] \]

其中 \(M\) 代表 Camera B 的 Feature 不存在。

模型可以接收 Feature 與 Missingness Mask，並且在有缺失資料的 Training Cases 上進行訓練。

但這裡要強調：

模型能處理 Missing Values，不代表品質政策允許 Missing Critical Evidence。

這是面試最值得講的一句話。

## Q3. How do you make the pipeline recoverable?

面試官：

> The machine loses power after inference but before writing the final result. What happens after reboot?

我會依據 Durable Checkpoints 判斷最後確定提交成功的階段。

例如：

```
Inspection: INS-0042

ACQUISITION       COMPLETE
IMAGE_QC          COMPLETE
INFERENCE         COMPLETE
DECISION          NOT COMMITTED
CLOUD_SYNC        NOT STARTED
```

重開機後：

- 檢查 Raw Images 與 Checksum。
    
- 讀取已完成的 Inference Evidence。
    
- 驗證 Model / Policy Version 與結果完整性。
    
- 重新執行尚未完成的 Decision。
    
- 以 Idempotent Operation 寫入結果。
    
- 依 PLC / Controller 現有狀態確認實體動作是否已執行。
    

### 為什麼不直接用 Exactly-once Processing？

在 DB、Filesystem、Message Queue、Network 和 PLC 構成的系統裡，很難保證整條鏈路天然具有真正的 Exactly-once Execution。

我會設計成：

At-least-once Delivery + Idempotent Processing + Durable State + Reconciliation

這通常比聲稱所有操作都能 Exactly Once 更實際。

### Fault Injection Testing

上線之前，我會主動測試：

|故障場景|預期結果|
|---|---|
|Capture 中斷電|能辨識已完成與未完成的 Captures|
|GPU Worker Crash|從 Persisted Tasks 恢復|
|Local DB 暫時不可寫|停止需要持久化保證的處理|
|Disk Full|禁止新檢測並告警|
|Network Disconnected|本機繼續檢測、同步排隊|
|Cloud Upload Timeout|Retry，不重複產生 Inspection|
|PLC Ack Missing|查詢設備狀態，不盲目重送|
|Model Package Damaged|使用已驗證的上一版或停止自動判定|

Staff Engineer 在這裡要展示的是 Failure Mode Analysis。

不是只有「遇到故障要重新啟動」，而是能預先說明每種故障對系統狀態、資料一致性與產品品質的影響。

## Q4. How do you choose decision thresholds?

完整回答除了 Part 6 的 Cost-sensitive Threshold，還需要三個統計概念。

### 1. Threshold 需要按 Defect Severity 區分

例如：

|Defect Type|Risk Level|Decision Preference|
|---|---|---|
|Critical Crack|Critical|極低漏檢容忍度|
|Large Scratch|High|高 Recall|
|Tiny Cosmetic Mark|Medium|平衡 FP / FN|
|Harmless Texture|Low|避免過多誤判|

不同瑕疵類型可能需要不同 Threshold。

### 2. Precision-Recall Curve

對罕見瑕疵，單看 Accuracy 或 ROC-AUC 往往不足。

例如 100,000 件產品中只有 100 件有重大瑕疵。

模型就算全部預測為正常品：

\[ Accuracy=\frac{99900}{100000}=99.9\% \]

但重大瑕疵 Recall = 0%。

所以我會評估：

- Critical Defect Recall
    
- Precision / False Alarm Burden
    
- PR-AUC
    
- False Reject Rate
    
- Review Rate
    
- Defect Escape Rate
    

### 3. Threshold 的統計可信度

即使測試結果為 Recall 100%，也不代表真實 Recall 必然為 100%。

假設測試集中有 20 個 Critical Defects，而模型全部找到了。

20/20 = 100% 是 Point Estimate，不能證明真實漏檢率低於 0.5%。

在獨立、同分布的 Bernoulli 假設下，使用所謂 Rule of Three 的近似：

\[ \text{Upper 95\% FN Rate Bound}\approx\frac{3}{N} \]

若希望漏檢率上限約小於 0.5%，而測試中一個漏檢都沒有：

\[ N\gtrsim\frac{3}{0.005}=600 \]

也就是大約需要 600 個獨立且具代表性的 Critical Defect Cases。

而且這只針對一個合併指標。若要證明不同 Camera、Defect Type、產品系列及工廠都符合要求，可能需要遠遠更多資料。

若同一個 Manufacturing Lot 的樣本高度相關，有效樣本量還會下降。

這種回答能展示 Senior / Staff 應有的 Statistical Validation 能力。

## Q5. How do you monitor false negatives?

這是整題最困難也最有價值的追問之一。

面試官：

> If the model says an item is good and no one reviews it, how would you know whether the model missed a defect?

這裡存在一個非常核心的問題：

Production Ground Truth 通常不是即時可取得的。

模型判斷 PASS，不代表我們真的知道它沒有瑕疵。

### 我的設計：四條 Feedback Channels

Production Inspection

PASS / FAIL / REVIEW

Random PASS Audit

隨機抽樣已自動放行產品

Human Review

不確定或高風險案例

Downstream QC

後續製程或人工複檢

Customer Returns

客訴、退貨、現場失效

Human-adjudicated Ground Truth

更新 Quality Metrics、Error Analysis、Training Dataset

### Channel A：Random PASS Audit

這是最關鍵的一條。

不能只檢查 Model 判 FAIL 或 REVIEW 的產品。

因為這樣永遠無法有效觀察已被自動放行的產品是否有漏檢。

我會從 Auto-pass 產品中進行具有代表性的 Random Sampling，送往獨立的人工或更高可靠度的檢測流程。

需要特別區分：

\[ P(\text{Defect}\mid\text{PASS}) \]

這代表已放行產品中的瑕疵比例，通常稱為 Escape Rate 或 Escape Risk 的一種量測方式。

但這不是：

\[ P(\text{PASS}\mid\text{Defect}) \]

後者才接近整體 Final Decision 的 False Negative Rate。

僅抽查 PASS 可以估計放行後的瑕疵比例，但若要估計總體 Recall / FNR，還需要包含非 PASS 產品在內的真實標籤分布，或使用適當的抽樣加權與統計方法。

### Channel B：Downstream Quality Inspection

例如產品通過自動檢測後，還會經過第二道品質站。

如果第二道站發現系統漏檢的 Scratch，就可以回溯：

```
Product ID
   |
   +-- Inspection ID
   +-- Raw Image
   +-- Camera / Lighting
   +-- Model Version
   +-- Threshold
   +-- Original Prediction
   +-- Final Ground Truth
```

這是非常有價值的 Error Analysis 資料。

### Channel C：Customer Feedback

例如客戶在交貨後發現刮痕。

這些產品可能有數週或數個月的 Label Delay。

所以 Production Quality Dashboard 應該區分：

- 已有完整 Ground Truth 的樣本
    
- 等待人工確認的樣本
    
- 尚未有 Feedback 的樣本
    

不能把「尚未收到客訴」當作「確認沒有瑕疵」。

### Channel D：Confidence / Drift Monitoring

即使尚未有 Ground Truth，也可以監控：

- Prediction Score Distribution
    
- Anomaly Score Distribution
    
- Image Sharpness
    
- Exposure Statistics
    
- Camera Color Statistics
    
- Feature Embedding Drift
    
- Auto-pass / Review / Reject Ratios
    

這些可以提供早期異常訊號。

但要清楚說明：Drift Metrics 只能提示潛在風險，無法單獨證明 False Negative Rate。

## Q6. How do you prevent data leakage between training and testing?

這道問題很常出現在 Senior ML Interview，而且 Multi-camera Image Inspection 特別容易產生 Leakage。

### 典型錯誤：Random Image Split

假設系統有：

```
Watch 001:
  Camera A: 20 images
  Camera B: 30 images
  Camera C: 30 images
```

如果把這 80 張影像隨機分成 Train / Validation / Test，同一件產品可能同時出現在三個 Dataset。

更嚴重的是，同一個 Scratch 會在不同 Camera、光源、角度、ROI 和 Crops 中多次出現。

模型可能只是記住同一件產品的紋理，而不是學到能泛化到新產品的瑕疵特徵。

### 正確方法：Group-aware Splitting

我會以 `product_id` 或更嚴格的獨立樣本群組為最低分割單位。

Train / Validation / Test Split

TRAIN

Watch 001

Watch 002

Watch 003

All views + crops

VALIDATION

Watch 101

Watch 102

Different watches

TEST

Watch 201

Watch 202

Unseen watches

同一件產品的所有相機影像、不同曝光、重拍、Tiling 與 Augmentations，都應保留在同一個 Dataset Partition。

### 但只有 Product-level Split 還不一定夠

如果 Train 與 Test 都來自同一個 Manufacturing Lot，而同一批產品有特殊的材質紋理或製程缺陷，Test Performance 可能仍然過度樂觀。

我還會設計：

|Split Type|測試目標|
|---|---|
|Product-group Split|避免同一產品重複|
|Lot-based Holdout|測試新的製造批次|
|Time-based Holdout|測試未來時間的資料|
|Machine-based Holdout|測試不同檢測機台|
|Factory / Site Holdout|測試跨工廠泛化|
|Camera-hardware Holdout|測試不同硬體條件|

最終不一定要使用六個完全獨立的測試集，但應將這些 Generalization Dimensions 納入 Evaluation Plan。

### Dataset 版本隔離

我會把資料分為：

- Training：模型參數學習。
    
- Validation：模型選擇與 Hyperparameter Tuning。
    
- Calibration：Probability Calibration，必要時作為獨立資料集。
    
- Threshold Selection：選擇 Operating Threshold。
    
- Locked Test：最終一次性的品質驗證。
    
- Future / Production Holdout：測試部署後的泛化能力。
    

Calibration、Threshold Selection 和 Test 不能隨便反覆共用而忽略資料重複使用造成的 Optimistic Bias。

特別是每次新模型都根據同一個 Locked Test Set 反覆調整架構，久而久之也會間接 Overfit Test Set。

Staff Engineer 應建立清楚的 Evaluation Governance，避免團隊為了達到指標而不自覺污染測試資料。

## Q7. How do you safely deploy a new model across 100 machines?

這題可以說是整場面試最後的 Staff-level System Design 關卡。

面試官：

> You have a new model that performs 2% better offline. How do you deploy it to 100 production machines without risking production quality?

首先，我不會直接回答「使用 Docker 和 Kubernetes」。

因為 100 台具有不同 Camera Calibration、Hardware Configuration、GPU、Firmware 和 Network Conditions 的工業設備，與一組同質性的 Cloud Servers 不同。

我會設計 Versioned Model Package + Shadow Testing + Staged Rollout + Automatic Rollback。

### 7.1 不只部署 Model Weights

一個 Model Release 應包含：

```
inspection_release_v2.4.0/
    model.onnx
    model_metadata.json
    preprocessing.json
    roi_config.json
    threshold_policy.json
    class_mapping.json
    hardware_compatibility.json
    test_report.json
    manifest.json
```

其中 `manifest.json` 會記錄：

```
{
  "release_id": "inspection_v2.4.0",
  "model_version": "defect_unet_v17",
  "preprocessing_version": "pp_v5",
  "policy_version": "policy_v8",
  "minimum_camera_config": "camera_cfg_v4",
  "supported_gpu_profiles": [
    "gpu_profile_a",
    "gpu_profile_b"
  ],
  "artifact_sha256": "<verified SHA-256>",
  "approval_status": "approved"
}
```

範例中的 SHA-256 欄位需由真實 Build Pipeline 產生。

每台設備自己的 Camera Calibration、Extrinsic Parameters 和 Motion Alignment 則應獨立管理，但要與 Release Package 有明確的相容性檢查。

這可以避免一個常見事故：

新 Model 本身沒有問題，但部署時搭配了錯誤的 Preprocessing 或 Camera Calibration，導致 Production Quality 下降。

### 7.2 完整部署流程

Safe Model Deployment Pipeline

1

Model Registry

註冊 Model、資料集與實驗版本

2

Offline Validation

Golden Test、Slice Analysis、Latency、Calibration

3

Hardware-in-the-loop

真實相機、GPU、Stage、Light 的整合驗證

4

Shadow Mode

新舊模型在相同影像上比較，不影響正式決策

5

Canary Deployment

只更新少數代表性機台

6

Progressive Rollout

按設備群組與品質指標逐步擴大

7

Full Deployment

全部合格機台切換並保留舊版

8

Monitoring / Rollback

持續觀察，異常時停止擴展或回退

MLflow Model Registry 可以追蹤 Model Version、Experiment Lineage、Metadata、Tags 和 Aliases，適合管理這條流程中的模型來源。正式發佈時我會把 Alias 解析為固定的不可變版本與 Artifact Hash，避免設備在執行中意外取得不同的模型。

![](https://www.google.com/s2/favicons?domain=https://mlflow.org&sz=32)

MLflow AI Platform

### 7.3 Shadow Deployment

例如目前 Production 使用 Model V1，新模型是 Model V2。

```
Production Image
       |
       +------------------------+
       |                        |
       v                        v
    Model V1                 Model V2
   Production                Shadow
       |                        |
       v                        v
  Real Decision          Prediction Only
       |                        |
       +-----------+------------+
                   |
                   v
             Comparison Log
```

新模型可以使用真實的 Production Images，但不能控制產品放行、拒收或任何實體機械動作。

我們可以比較：

- Per-defect disagreement
    
- Confidence Distribution
    
- Review Rate Change
    
- Latency / GPU Usage
    
- Existing Ground-truth Cases
    
- 各種 Camera / Region / Defect Slice 的表現
    

要注意 Shadow Mode 可能增加 GPU 負載，所以不能為了測試 V2 而讓 V1 的生產檢測延遲超標。必要時可以進行離線 Replay，或者只抽樣執行 Shadow Inference。

此外，V1 與 V2 彼此一致並不代表都正確；最終仍需要 Independent Ground Truth。

### 7.4 Canary Rollout 到 100 台機器

以下是我會在面試中畫的漸進式部署策略。

100-machine Progressive Rollout

Ring 1

# 1

1%

Ring 2

# 5

累計 6%

Ring 3

# 20

累計 26%

Ring 4

# 74

累計 100%

每一個 Ring 完成健康檢查、品質確認及必要的觀察期後，才允許下一階段開始。Ring 1 不應隨機選一台完全沒有代表性的設備。

我的部署決策會考慮 Device Cohorts：

- Camera Hardware Version
    
- GPU Type
    
- Factory / Location
    
- Product Series
    
- Lighting Configuration
    
- Current Software Version
    

這是因為在 Machine 001 表現正常，不代表 Machine 099 一定能正常運作。

AWS IoT Jobs 支援控制 Rollout Rate、Execution Timeout、Retry 和符合條件時中止部署；AWS IoT Greengrass 也提供 Component Deployment 和失敗時的 Rollback Policy。實際使用時仍需驗證目標 Windows/Linux 版本及各元件的支援情況。

![](https://www.google.com/s2/favicons?domain=https://docs.aws.amazon.com&sz=32)

AWS IoT Core

+1

### 7.5 Automatic Rollback Conditions

我會建立兩種 Guardrails。

Engineering Guardrails

例如：

- Model Loading Failed
    
- GPU Out-of-memory
    
- Unexpected Inference Error
    
- Latency SLO Violation
    
- Camera / Model Input Incompatibility
    
- Model Output Schema Mismatch
    

Quality Guardrails

例如：

- Critical Defect Recall 明顯惡化，且有足夠標籤支持
    
- Review Rate 不合理地飆升
    
- False Reject Rate 明顯增加
    
- 新舊模型在關鍵區域大量出現不一致
    
- Camera / Model Drift 超過已設定限制
    

我不會只依賴少量線上樣本就宣稱新模型已經達到 99.5% Recall。

因為重大瑕疵可能非常罕見，部署後一天之內可能根本沒有足夠的正樣本可以驗證。

所以：

Canary 的主要目的，是在有限風險下驗證實際運作與早期品質訊號，不是取代嚴格的 Offline Statistical Evaluation。

### 7.6 Rollback 不能只改 Model Version

假設 V2 使用新的 Image Normalization 和 ROI Configuration。

如果回退時只恢復 V1 Weights，卻仍使用 V2 Preprocessing，結果可能更糟。

因此需要整套 Release Atomic Switching。

可以採用 A/B Slots：

```
ACTIVE SLOT
  Release V1
  Model V1
  Preprocess V1
  Policy V1

STANDBY SLOT
  Release V2
  Model V2
  Preprocess V2
  Policy V2
```

新套件先下載、驗證 Hash / Signature、檢查硬體相容性，並在設備安全且沒有正在執行的 Inspection 時切換。

每個 Inspection 應固定使用一個 `release_id`，避免同一件產品的部分影像使用 V1、另一部分使用 V2，卻未經設計便混合結果。

如果 V2 失敗，就將 Active Slot 恢復 V1。

即使 Cloud Disconnected，設備仍能使用本機保留的已驗證版本。

### Staff-level 標準答案

> I would treat model deployment as a controlled production release, not a file replacement.
> 
> Every release would bundle the model, preprocessing, thresholds, schemas, and compatibility metadata into an immutable, signed artifact.
> 
> I would first validate it offline and on real hardware, then run it in shadow mode, and finally deploy progressively across representative machine cohorts.
> 
> Each device would retain the last known-good release and switch only at a safe inspection boundary. Fleet-wide rollout would be controlled by engineering and quality guardrails, with automatic pause and rollback where appropriate.

# Part 9. Dataset Versioning / Retraining / Continuous Improvement

題目最後一句是：

> ...and continuously improve after deployment.

面試官希望你說明怎麼讓系統不只是上線，而是能形成長期可維護的 Learning System。

## 9.1 Continuous Learning Architecture

#chatgpt-mermaid-_r_7n6_{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;fill:rgb(13, 13, 13);}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#chatgpt-mermaid-_r_7n6_ .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-_r_7n6_ .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-_r_7n6_ .error-icon{fill:rgb(243, 243, 243);}#chatgpt-mermaid-_r_7n6_ .error-text{fill:rgb(13, 13, 13);stroke:rgb(13, 13, 13);}#chatgpt-mermaid-_r_7n6_ .edge-thickness-normal{stroke-width:1px;}#chatgpt-mermaid-_r_7n6_ .edge-thickness-thick{stroke-width:3.5px;}#chatgpt-mermaid-_r_7n6_ .edge-pattern-solid{stroke-dasharray:0;}#chatgpt-mermaid-_r_7n6_ .edge-thickness-invisible{stroke-width:0;fill:none;}#chatgpt-mermaid-_r_7n6_ .edge-pattern-dashed{stroke-dasharray:3;}#chatgpt-mermaid-_r_7n6_ .edge-pattern-dotted{stroke-dasharray:2;}#chatgpt-mermaid-_r_7n6_ .marker{fill:rgb(143, 143, 143);stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_7n6_ .marker.cross{stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_7n6_ svg{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;}#chatgpt-mermaid-_r_7n6_ p{margin:0;}#chatgpt-mermaid-_r_7n6_ .label{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_7n6_ .cluster-label text{fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_7n6_ .cluster-label span{color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_7n6_ .cluster-label span p{background-color:transparent;}#chatgpt-mermaid-_r_7n6_ .label text,#chatgpt-mermaid-_r_7n6_ span{fill:rgb(13, 13, 13);color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_7n6_ .node rect,#chatgpt-mermaid-_r_7n6_ .node circle,#chatgpt-mermaid-_r_7n6_ .node ellipse,#chatgpt-mermaid-_r_7n6_ .node polygon,#chatgpt-mermaid-_r_7n6_ .node path{fill:rgb(222, 234, 251);stroke:rgb(83, 154, 248);stroke-width:1px;}#chatgpt-mermaid-_r_7n6_ .rough-node .label text,#chatgpt-mermaid-_r_7n6_ .node .label text,#chatgpt-mermaid-_r_7n6_ .image-shape .label,#chatgpt-mermaid-_r_7n6_ .icon-shape .label{text-anchor:middle;}#chatgpt-mermaid-_r_7n6_ .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#chatgpt-mermaid-_r_7n6_ .rough-node .label,#chatgpt-mermaid-_r_7n6_ .node .label,#chatgpt-mermaid-_r_7n6_ .image-shape .label,#chatgpt-mermaid-_r_7n6_ .icon-shape .label{text-align:center;}#chatgpt-mermaid-_r_7n6_ .node.clickable{cursor:pointer;}#chatgpt-mermaid-_r_7n6_ .root .anchor path{fill:rgb(143, 143, 143)!important;stroke-width:0;stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_7n6_ .arrowheadPath{fill:rgb(143, 143, 143);}#chatgpt-mermaid-_r_7n6_ .edgePath .path{stroke:rgb(143, 143, 143);stroke-width:1px;}#chatgpt-mermaid-_r_7n6_ .flowchart-link{stroke:rgb(143, 143, 143);fill:none;}#chatgpt-mermaid-_r_7n6_ .edgeLabel{background-color:rgb(252, 252, 252);text-align:center;}#chatgpt-mermaid-_r_7n6_ .edgeLabel p{background-color:rgb(252, 252, 252);}#chatgpt-mermaid-_r_7n6_ .edgeLabel rect{opacity:0.5;background-color:rgb(252, 252, 252);fill:rgb(252, 252, 252);}#chatgpt-mermaid-_r_7n6_ .labelBkg{background-color:rgba(252, 252, 252, 0.5);}#chatgpt-mermaid-_r_7n6_ .cluster rect{fill:rgb(243, 243, 243);stroke:rgba(0, 0, 0, 0.1);stroke-width:1px;}#chatgpt-mermaid-_r_7n6_ .cluster text{fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_7n6_ .cluster span{color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_7n6_ div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:12px;background:rgb(243, 243, 243);border:1px solid rgba(0, 0, 0, 0.1);border-radius:2px;pointer-events:none;z-index:100;}#chatgpt-mermaid-_r_7n6_ .flowchartTitleText{text-anchor:middle;font-size:18px;fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_7n6_ rect.text{fill:none;stroke-width:0;}#chatgpt-mermaid-_r_7n6_ .icon-shape,#chatgpt-mermaid-_r_7n6_ .image-shape{background-color:rgb(252, 252, 252);text-align:center;}#chatgpt-mermaid-_r_7n6_ .icon-shape p,#chatgpt-mermaid-_r_7n6_ .image-shape p{background-color:rgb(252, 252, 252);padding:2px;}#chatgpt-mermaid-_r_7n6_ .icon-shape .label rect,#chatgpt-mermaid-_r_7n6_ .image-shape .label rect{opacity:0.5;background-color:rgb(252, 252, 252);fill:rgb(252, 252, 252);}#chatgpt-mermaid-_r_7n6_ .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#chatgpt-mermaid-_r_7n6_ .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#chatgpt-mermaid-_r_7n6_ .node .neo-node{stroke:rgb(83, 154, 248);}#chatgpt-mermaid-_r_7n6_ [data-look="neo"].node rect,#chatgpt-mermaid-_r_7n6_ [data-look="neo"].cluster rect,#chatgpt-mermaid-_r_7n6_ [data-look="neo"].node polygon{stroke:url(#chatgpt-mermaid-_r_7n6_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_7n6_ [data-look="neo"].swimlane.cluster rect{filter:none;}#chatgpt-mermaid-_r_7n6_ [data-look="neo"].node path{stroke:url(#chatgpt-mermaid-_r_7n6_-gradient);stroke-width:1px;}#chatgpt-mermaid-_r_7n6_ [data-look="neo"].node .outer-path{filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_7n6_ [data-look="neo"].node .neo-line path{stroke:rgb(83, 154, 248);filter:none;}#chatgpt-mermaid-_r_7n6_ [data-look="neo"].node circle{stroke:url(#chatgpt-mermaid-_r_7n6_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_7n6_ [data-look="neo"].node circle .state-start{fill:#000000;}#chatgpt-mermaid-_r_7n6_ [data-look="neo"].icon-shape .icon{fill:url(#chatgpt-mermaid-_r_7n6_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_7n6_ [data-look="neo"].icon-shape .icon-neo path{stroke:url(#chatgpt-mermaid-_r_7n6_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_7n6_ .node text{font-size:14px;font-weight:600;letter-spacing:normal;fill:rgb(65, 65, 65);}#chatgpt-mermaid-_r_7n6_ .edgeLabels text{font-size:13px;font-weight:600;letter-spacing:-0.08px;fill:rgb(65, 65, 65);}#chatgpt-mermaid-_r_7n6_ .node tspan[font-weight="normal"],#chatgpt-mermaid-_r_7n6_ .edgeLabels tspan[font-weight="normal"]{font-weight:600;}#chatgpt-mermaid-_r_7n6_ .edgeLabel .label rect{opacity:1;rx:13px;ry:13px;fill:rgb(252, 252, 252);stroke:rgb(219, 219, 219);stroke-width:1px;}#chatgpt-mermaid-_r_7n6_ .node rect,#chatgpt-mermaid-_r_7n6_ .node circle,#chatgpt-mermaid-_r_7n6_ .node ellipse,#chatgpt-mermaid-_r_7n6_ .node polygon,#chatgpt-mermaid-_r_7n6_ .node path{fill:rgb(243, 243, 243);stroke:rgba(0, 0, 0, 0.1);stroke-width:1px;}#chatgpt-mermaid-_r_7n6_ .node rect{rx:16px;ry:16px;}#chatgpt-mermaid-_r_7n6_ .node.mermaid-decision .label-container{fill:rgb(249, 249, 249);stroke:rgb(219, 219, 219);stroke-dasharray:2,2;}#chatgpt-mermaid-_r_7n6_ .edgePaths .flowchart-link{stroke:rgb(143, 143, 143);stroke-width:1px;stroke-linecap:round;stroke-linejoin:round;}#chatgpt-mermaid-_r_7n6_ .marker{fill:rgb(143, 143, 143);stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_7n6_ :root{--mermaid-font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";}Production InspectionPrediction + Evidence +MetadataFeedback CollectionHuman Labeling &AdjudicationDataset Quality ValidationVersioned Training DatasetTraining / HyperparameterSearchCalibration + EvaluationRelease Gates Pass?Error Analysis / ImproveModel RegistryShadow + Canary + RolloutFleet MonitoringNoYes

最重要的是：Continuous Learning 不等於每次有新資料就自動重新訓練並部署。

我會把 Training Automation 和 Production Release Approval 分開。

## 9.2 如何選擇 Retraining Data？

如果所有圖片都上傳 Cloud 並全部用來重新訓練，會非常昂貴，而且大部分資料可能都是大量重複的正常產品。

我會使用 Data Selection Strategy。

|資料類別|用途|
|---|---|
|Confirmed False Negatives|找出危險的漏檢模式|
|Confirmed False Positives|降低不必要的判退|
|Low-confidence Cases|加強決策邊界|
|Novel Anomaly Samples|擴充未知缺陷類型|
|Randomly Sampled PASS|維持正常品分布代表性|
|New Camera / Lighting Data|處理 Domain Shift|
|Recent Production Data|偵測時間漂移|

這裡有個非常常見的陷阱：

如果只收集 Model 判斷錯誤或高不確定的資料來訓練，Dataset 會越來越偏向困難案例，失去真實 Production Distribution 的代表性。

所以需要結合：

Active Learning + Representative Random Sampling + Hard-negative Mining

並且保存每個樣本是如何被挑選的，必要時才能做無偏或加權評估。

## 9.3 Ground Truth 怎麼建立？

對微小瑕疵，Ground Truth 可能比 Training Model 更困難。

例如同一道 Scratch：

- Engineer A 認為是 Critical Scratch。
    
- Engineer B 認為是正常表面紋理。
    
- QA 認為只要長度小於 100 μm 就可接受。
    

這時候問題不在模型，而在 Label Definition 不一致。

因此我會建立 Defect Taxonomy。

```
Defect
   |
   +-- Scratch
   |     +-- Cosmetic
   |     +-- Functional
   |
   +-- Dent / Pit
   |
   +-- Printing Defect
   |
   +-- Contamination
   |
   +-- Unknown Anomaly
```

每個 Label 還要包含：

```
{
  "defect_type": "scratch",
  "severity": "high",
  "region": "dial_outer_ring",
  "length_um": 186.2,
  "review_status": "adjudicated",
  "label_version": "taxonomy_v3",
  "reviewer_agreement": true
}
```

這個設計的價值是把 Model Prediction 與產品規範分開。

未來如果 QA 改變對 Scratch Length 的容忍度，可能只需更新品質政策，不一定要重新訓練整個 Model。

### Label Quality Control

我會要求：

1. 建立一致的 Annotation Guideline。
    
2. 對 Critical Cases 使用雙人標註。
    
3. 不一致案例進入 Expert Adjudication。
    
4. 統計 Inter-annotator Agreement。
    
5. 保存 Label History。
    
6. 對定義模糊的邊界案例建立專屬 Evaluation Slice。
    

這一點對 Staff Engineer 特別重要，因為很多 ML 系統的性能上限是由 Label Quality 決定，而不是由 Model Complexity 決定。

## 9.4 Dataset Versioning 怎麼做？

一個 Dataset Version 不只是資料夾名稱。

例如：

```
Dataset Version: defect_dataset_v12

Source:
  Production 2026-06 to 2026-09

Selection:
  Random pass sampling
  Confirmed defect review
  Hard negatives

Labels:
  taxonomy_v3

Split:
  product_group_split_v4

Preprocessing:
  preprocess_v5

Manifest:
  dataset_manifest_v12.json
```

Manifest 應該能追溯每張圖片：

- 來自哪件產品、哪台設備
    
- 使用哪個 Camera / Light Profile
    
- 使用哪個 Calibration
    
- Raw Image Checksum
    
- Ground Truth 和 Label Version
    
- Train / Validation / Test Split Assignment
    
- Data Selection Rule
    
- 是否為 Synthetic / Augmented Sample
    

Dataset 可以使用 DVC、LakeFS、Object Storage Versioning、Parquet Metadata 或其他適合的工具組合。

工具名稱不是核心，真正重要的是 Data Lineage、Reproducibility 和 Immutable Evaluation Evidence。

理想狀態下，任何一個 Production Model Version 都能找到它對應的 Dataset、Training Code Commit、Hyperparameters、Experiment Results 和 Release Approval。

# Part 10. Production Monitoring：要監控的不只是 GPU

面試官：

> Your models are now deployed. What would your production dashboard show?

我會設計四個層次的監控。

## 10.1 System Health

例如：

|Metric|監控目的|
|---|---|
|Camera Heartbeat|相機是否在線|
|Capture Success Rate|影像擷取成功率|
|Frame Drop / Timeout|擷取可靠度|
|Queue Depth|是否累積 Backlog|
|GPU Utilization|計算資源利用率|
|GPU Memory|OOM 風險|
|Disk Free Space|避免 Data Loss|
|Network Sync Lag|Cloud 同步延遲|
|Process Restart Count|穩定性|

## 10.2 Image Quality Health

例如：

- Sharpness Distribution
    
- Exposure Saturation
    
- Autofocus Failure Rate
    
- ROI Alignment Error
    
- Lighting Intensity / Reference Image Drift
    
- Camera Calibration Age
    

這些可以幫助判斷模型表現變差，究竟是因為模型老化，還是相機、光源及機械設備異常。

## 10.3 ML Quality

|Metric|重點|
|---|---|
|Critical Defect Recall|已確認重大瑕疵的攔截能力|
|False Negative Rate|已標註樣本的漏檢比例|
|False Reject Rate|良品被錯誤判退的比例|
|Review Rate|人工負擔|
|Escape Rate|放行產品中發現瑕疵的比例|
|Anomaly Rate|未知模式出現程度|
|Confidence Calibration|Score 是否具可信度|
|Drift Metrics|Production 與 Reference Distribution 的差異|

對這些品質指標，應標示 Ground Truth 的來源、樣本量、抽樣方式與 Label Delay，不能只顯示一個漂亮的百分比。

## 10.4 Business / Operational Metrics

例如：

- Inspection Throughput
    
- P50 / P95 / P99 Cycle Time
    
- Automatic PASS Rate
    
- Operator Review Time
    
- Unplanned Downtime
    
- Cost per Inspected Unit
    
- Defect Escape Cost
    
- Rework / Scrap Cost
    

Google 的 Production ML 指南也強調需要監控 Training-serving Skew，以及 Input Data、Features 與實際線上表現的差異，而不能只依靠離線模型評估。

![](https://www.google.com/s2/favicons?domain=https://developers.google.com&sz=32)

Google for Developers

+1

### Dashboard 應按 Machine / Camera / Product Slice 分析

例如同樣的 Model V2：

|Slice|Critical Recall|Review Rate|
|---|---|---|
|Camera A / Site 1|99.8%|2.1%|
|Camera B / Site 1|99.6%|3.0%|
|Camera B / Site 2|94.1%|12.5%|

以上為假設數據。

這裡全體平均指標可能掩蓋 Site 2 的嚴重問題。

所以 Staff Engineer 不能只關注 Global Aggregate Metric。

要能回答：

> Is the degradation caused by a new defect distribution, a camera calibration issue, or a software release?

我會透過 Image Quality、Calibration Version、Feature Distribution、Model Outputs、Ground Truth 和 Release History 進行跨層分析。

# Part 11. Capacity Planning 與 Cost：Staff Engineer 必須能估算

面試官可能突然問：

> How much data will 100 machines generate? Can your architecture handle it?

假設每件產品約 80 張影像，平均為 4512 × 4512 的 8-bit Mono Image。

單張約：

\[ 4512^2\times1\approx20.4\ MB \]

每件產品：

\[ 80\times20.4\ MB\approx1.63\ GB \]

每台每月 500 件，100 台：

\[ 1.63\ GB\times500\times100 \approx81.5\ TB/month \]

這還不包括 HDR 中間影像、Model Outputs、Segment Masks、Logs、Metadata 與其他檔案。

若使用 16-bit Image，原始影像的容量約再增加一倍。

原始影像資料量估算

每件產品

# 1.63 GB

每台每月

# 815 GB

100 台每月

# 81.5 TB

100 台一年

# 978 TB

未壓縮 8-bit Mono 假設值，未包含資料備份、衍生影像和冗餘儲存。

這就解釋了為什麼我不會讓每件產品的即時推論都依賴 Cloud Upload。

## 11.1 Network Bandwidth

如果一張原始圖片約 20 MB，80 張約 1.6 GB：

在單條理論上限為 1 Gbit/s 的鏈路下：

\[ 1.6\ GB\times8/(1\ Gbit/s)\approx12.8\ seconds \]

這只是理論最短傳輸時間，不含 Protocol Overhead、GigE Packet Loss、Image Decode、Disk I/O 和其他工作。

如果三台 Camera 各自有獨立的有效網路頻寬，資料可以平行傳輸；但如果最後匯聚到同一個瓶頸，整體速度仍受該瓶頸限制。

因此必須考慮：

- NIC / Switch Bandwidth
    
- Per-camera Bandwidth Limits
    
- Jumbo Frames（若全鏈路支援）
    
- Buffer Allocation
    
- Packet Resend
    
- SSD Write Throughput
    
- CPU Decode Cost
    
- GPU Transfer Cost
    

## 11.2 Data Retention Policy

我會區分：

|Data Type|建議策略|
|---|---|
|Critical Defects|保存完整 Raw Image 和 Metadata|
|Confirmed False Negatives|高優先級保存|
|Uncertain / Anomaly|保存以供人工審核|
|Normal Production|分層儲存與代表性抽樣|
|Debug / Temporary|短期 Retention|
|Audit Records|依品質與法規要求保存|

這裡的數據保留策略必須經過 QA、法遵與 Business Owner 核准。

可以採用：

Local SSD → Async Object Storage → Lifecycle Management → Long-term Archive

而不是將所有 Raw Images 永久存放在每台設備裡。

另外應加上 Encryption、Artifact Signature、Least-privilege Access、Audit Trail 和 Restore Testing。

# Part 12. 如何證明整套系統可以上線？

Staff Engineer 的責任不應只是「完成開發」，而是證明整套系統符合 Production Acceptance Criteria。

我會建立下列 Verification Strategy。

|Test Layer|測試內容|Pass Criteria|
|---|---|---|
|Unit Tests|Image Processing、Threshold、Coordinate Mapping|Deterministic Expected Outputs|
|Model Tests|Golden Dataset、Critical Recall、Slice Analysis|符合品質與統計門檻|
|Component Integration|Camera SDK、Light、Motion、GPU|完成各元件契約|
|Hardware-in-the-loop|真實相機與光源、移動平台|完整 Captures 與可靠控制|
|End-to-end Test|真實產品掃描至 Decision|正確輸出、持久化及 Audit|
|Fault Injection|Crash、Network Loss、Disk Full|正確 Recovery 或隔離|
|Performance Test|連續多件產品|P95 / Throughput 達標|
|Soak Test|長時間連續運作|無不可接受的資源累積或品質漂移|
|Fleet Deployment Test|Canary、Rollback、Offline Update|可恢復到 Known-good Release|

## 12.1 Golden Reference Samples

我會建立一組穩定的 Reference Parts，包含：

- 已確認的正常樣本
    
- 已知尺寸的微小 Scratch
    
- 已知尺寸的 Pit / Dent
    
- 已確認的 Printing Defects
    
- 邊界尺寸的 Defects
    
- 特殊反光或低對比表面
    

這組資料可以分成兩種：

Digital Golden Dataset

用於驗證演算法和 Model Release。

Physical Golden Samples

用於在真實機台上定期執行光學與硬體健康檢查。

這兩者不能互相取代。

Digital Dataset 可以證明特定數位輸入下的模型表現；Physical Samples 則能測試 Camera、Lighting、Motion、Focus 和模型整條鏈路。

但 Physical Samples 也會磨損或變化，因此需要妥善保存、定期確認其狀態與校準。

# Part 13. Senior 與 Staff Engineer 在同一道題目的差別

這個比較很適合用來準備面試時的回答深度。

|面向|Senior Engineer|Staff Engineer|
|---|---|---|
|Requirements|定義模型及系統需求|讓 Business / QA / Hardware 對風險與 SLO 達成共識|
|Camera|選解析度、設計 Capture|決定多相機、光源、成本與 Throughput 的取捨|
|Model|訓練與優化 Detection / Segmentation|決定整體 Hybrid Architecture 與 Evaluation Strategy|
|Decision|設計 Threshold|制定 Risk-based Product Quality Policy|
|Reliability|實作 Retry、Queue、Recovery|設計 Failure Boundaries 與 Operational Safety|
|Data|建立 Training / Validation / Test|建立跨機台與跨團隊的 Data Governance|
|Monitoring|監控 Latency 和 Accuracy|定義品質風險、Label Delay 與 Incident Response|
|Deployment|部署一個新 Model|安全管理 100 台異質設備與回退策略|
|Leadership|帶領小組交付|協調 Hardware、ML、Software、QA、Manufacturing|

## 13.1 面試官如果問：How would you organize the team?

我會明確定義 Component Ownership 和 Interface Contracts。

例如：

Hardware / Optics Team

負責 Camera Calibration、Lighting Design、Motion Repeatability、Physical Golden Samples。

Imaging / CV Team

負責 Image Processing、ROI、Detection、Segmentation、Anomaly Detection。

ML / Data Team

負責 Training Dataset、Ground Truth、Evaluation、Model Calibration。

Platform / Software Team

負責 Orchestrator、Queues、Persistence、Edge Deployment、Cloud Synchronization。

QA / Manufacturing Team

負責 Defect Severity Definition、Acceptance Criteria、Human Review、Escaped Defect Investigation。

但這些團隊不能只分開交付後再最後整合。

我會建立共同的 Acceptance Tests 與 Versioned Interface Contracts，例如 Image Metadata Schema、Inference Output Schema、Release Compatibility Matrix。

當 Quality Metric 下降時，不能出現 Hardware Team 說是 Model 問題，而 ML Team 說是 Lighting 問題的情況。

需要共同的 Evidence、統一的 Dashboard 和明確的 Incident Owner。

# Part 14. 面試最後 5 分鐘：如何將答案總結得像 Staff Engineer？

假設這是一場 60 分鐘的系統設計面試，可以這樣分配時間：

|時間|重點|
|---|---|
|0–5 min|Clarify Requirements / SLO / Constraints|
|5–12 min|High-level Architecture|
|12–22 min|Imaging / Acquisition / Model Design|
|22–32 min|Decision / Threshold / Missing Evidence|
|32–42 min|Reliability / Recovery / Backpressure|
|42–52 min|Data / Monitoring / Continuous Learning|
|52–60 min|Fleet Deployment / Tradeoffs / Summary|

## 最後的英文總結範例

這一段可以直接練習作為面試最後的回答：

> My main design principle is to build an inspection system where image acquisition quality, ML predictions, and final production decisions are clearly separated.
> 
> I would first work backward from the minimum defect size to determine camera resolution, optics, lighting, and inspection coverage. I would use a hybrid vision pipeline combining high-resolution defect segmentation, geometric measurements, and anomaly detection for previously unseen defects.
> 
> For decision-making, I would prioritize critical defect recall and explicitly handle uncertainty, missing images, and low-quality captures. A missing critical view must never silently result in a pass.
> 
> To make the system production-ready, I would use durable inspection states, idempotent processing, bounded queues, checkpointing, and hardware-aware recovery. Each inspection must be reproducible and auditable.
> 
> For continuous improvement, I would collect production feedback from random pass audits, human reviews, downstream quality checks, and customer-reported defects. I would enforce group-aware dataset splitting and evaluate performance across cameras, product families, manufacturing lots, and deployment sites.
> 
> Finally, I would deploy models as immutable, versioned releases, validate them on real hardware, and roll them out progressively across the fleet with monitoring and rollback.
> 
> The ultimate goal is not simply to achieve high offline model accuracy. It is to deliver a reliable inspection system that minimizes defect escapes, handles failures safely, and continuously improves without compromising production quality.

# Part 15. 一個可以真正區分 Staff Engineer 的延伸追問

假設面試官最後提出：

> We achieved 99.5% recall in the lab, but after deployment the customer reports more missed defects. What would you investigate first?

我會按照因果關係，而不是立即重新訓練。

Production Recall Degradation：Root Cause Tree

Missed Defects Increased

1. Imaging Shift

Lighting、Focus、Calibration、Camera Replacement、Motion、Lens Contamination

2. Data Shift

新材質、新工廠、新缺陷、不同製造批次

3. Software / Policy

Release、Normalization、ROI、Threshold、Missing Image Handling

4. Evaluation / Labels

Ground Truth Error、Sampling Bias、Data Leakage、Label Delay

接下來進行 Controlled Comparison：

1. 先確認受影響的 Machine、Camera、Model、Release、產品與時間範圍。
    
2. 使用漏檢的 Raw Images，以相同 Preprocessing / Policy 重跑新舊模型。
    
3. 確認瑕疵在 Raw Image 中是否真的可見。
    
4. 如果不可見，優先檢查 Lighting、Focus、Resolution、Coverage。
    
5. 如果 Raw Image 可見但 Preprocessing 後消失，檢查 Image Pipeline。
    
6. 如果 Preprocessing 後仍可見但模型漏檢，執行 Model Error Analysis。
    
7. 如果模型有偵測到但最終產品放行，檢查 Evidence Fusion、Threshold 與 Policy。
    
8. 判斷影響範圍，必要時停止 Auto-pass、進入人工檢查或回退到安全版本。
    

這個調查順序很有價值，因為每種問題的修復方法不同。

|根本原因|正確修復方向|
|---|---|
|Optical Resolution 不足|改 Camera / Lens / FOV|
|Light Angle 不適合|改 Lighting Recipe|
|Autofocus 不穩定|改 AF / Mechanical Control|
|Preprocessing 抹除瑕疵|修正 Image Pipeline|
|新 Defect Type|增加資料、Retrain|
|Threshold 錯誤|重新校準與驗證 Policy|
|Missing Image 被當 PASS|修復 Decision / Coverage Logic|
|Test Set Leakage|重建 Evaluation Pipeline|

真正成熟的 Staff Engineer 不會把所有 Production Quality Problems 都當成 Model Training Problems。

## 最後：面試官最希望看見的 10 個核心能力

|核心能力|最能加分的論點|
|---|---|
|1. Requirements Engineering|從 Defect Size、Recall、Cost、Latency 反推系統|
|2. Imaging Fundamentals|用 FOV、Pixel Size、Optics、Lighting 設計輸入品質|
|3. ML Architecture|理解 Tiny Defects、Tiling、Segmentation、Anomaly Detection|
|4. Decision Theory|知道 Prediction 不等於 Production Decision|
|5. Reliability Engineering|Durable State、Idempotency、Retry、Fault Isolation|
|6. Distributed Systems|Queue、Concurrency、Backpressure、Edge / Cloud 分離|
|7. Statistical Evaluation|Defect-level Metrics、Confidence Intervals、Group Split|
|8. MLOps|Dataset Lineage、Monitoring、Retraining、Model Registry|
|9. Fleet Deployment|Hardware Compatibility、Canary、Rollback|
|10. Technical Leadership|跨團隊制定品質契約、風險策略與 Release Governance|

我認為這道題目最值得記住的核心答案是：

> A production-grade ML inspection system is not just a model that detects defects. It is a system that guarantees inspection coverage, manages uncertainty, survives failures, provides traceable decisions, and improves safely over time.

對於 Senior / Staff Computer Vision Engineer 的面試而言，能把這句話背後的光學、演算法、統計、系統可靠度和團隊交付完整展開，會比單純介紹 YOLO、U-Net 或 Transformer 更有說服力。