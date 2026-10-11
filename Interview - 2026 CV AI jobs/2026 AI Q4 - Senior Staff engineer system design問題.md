

|                                                |     |
| ---------------------------------------------- | --- |
| [[#### Senior Staff engineer system design問題]] |     |
|                                                |     |
|                                                |     |
|                                                |     |

#### Senior Staff engineer system design問題
```
關於Senior/Staff engineer如果被問到system design問題譬如設計 Data Collection 和 Ground Truth, 選擇模型、實驗方法與驗證方式, 解決 Production Bottleneck 和 Failure Modes等請以兩個具體完整例子的系統說明(一個是computer vision例子, 一個是LLM例子). technical leadership問題譬如跨 Hardware、Software、Research 團隊協作, 指導其他工程師並交付可維護的系統.
```

# Senior / Staff AI Engineer 面試：System Design + Technical Leadership 完整實戰解析

在美國應徵 Senior AI Engineer、Staff AI Engineer、Senior Computer Vision Engineer、Staff Machine Learning Engineer 等職位時，System Design 和 Technical Leadership 是非常重要的評估項目。

這類面試與一般 Software Engineer 的 System Design（例如 Design Twitter、Design Uber）有明顯不同。

AI Engineer 除了要考慮 Scalability、Reliability、Latency、Database、API，還必須回答：

- Data Collection：如何收集具有代表性的訓練資料？
    
- Ground Truth：如何確保標註是正確的？
    
- Model Selection：為什麼選擇這個模型，而不是其他模型？
    
- Experimentation：如何證明新模型比原本的模型更好？
    
- Validation：如何證明 Offline Evaluation 可以反映 Production 表現？
    
- Production：模型出錯、資料分布改變或硬體故障時怎麼辦？
    
- Technical Leadership：如何讓不同團隊協作、解決衝突並交付可靠且可維護的產品？
    

我會以兩個完整案例說明，並且把「一般工程師的回答」與「Senior／Staff 等級的回答」區分開來。

![Taiyang Technology Co., Ltd. | Automation Integration Manufacturing | Robotic Arm Integration | Intelligent Hardware & Software Integration Expert](https://images.openai.com/static-rsc-4/DscqIh3aXfRr_c0AOAcA2QzJvMa7O5Sw7I6eLvmB360fDW5__G_Ld0ANxVNwvXnB2anWFsTqXCJgIsiNq0w2MdHVeh7zDTO8dgC12vMCfmlY0UWoXaVVUZwQG8O3tA5GC15KYfg8JTv1fLD9XaF26ZLb9JD1S2WoKFQ_Ci6YI9k?purpose=inline)

案例一：Computer Vision

多相機自動化手錶檢測與真偽辨識系統。

涵蓋 Camera、Lighting、Autofocus、Motion Control、Image Processing、Deep Learning、Anomaly Detection、Bayesian Fusion、MLOps。

![Guidance for Aerospace Technician’s Assistant on AWS](https://images.openai.com/static-rsc-4/Piqyrurh2Nq6fcI00jqfK5GwXSuu2e5wSUVFTgGMLBxmnoONLNli21qvbjbobIZQkmQ2hN2igzNw1vLbB489byS9M3qMN48bDLrt11ywuU-J2wmuzwS8Qs_ueF-KMjmVKXkgtvCTG09OoabsDg1FwZtWTcUmF0YJN57Rxape064?purpose=inline)

案例二：LLM / RAG / Agent

企業級 AI 設備故障診斷助手。

涵蓋 Knowledge Ingestion、Retrieval、LLM Reasoning、Tool Calling、Agent Evaluation、Security、Latency、Cost、Production Monitoring。

以下兩個案例的業務規模、KPI、實驗數據都是用於面試練習的假設條件與設計目標，不是實際部署結果。

## Part 1. 面試官真正想區分的是 Senior 和 Staff 的哪些能力？

|評估面向|Senior Engineer|Staff Engineer|
|---|---|---|
|Problem Definition|能把需求轉換成技術規格|能釐清真正的商業問題，挑戰不合理需求|
|Architecture|設計完整且可靠的系統|決定跨團隊架構、長期技術方向與取捨|
|Data|建立可靠 Data Pipeline|制定 Data Quality、Ground Truth、Data Governance 標準|
|Model|選擇、訓練、優化模型|定義模型策略、投資方向與選型標準|
|Experiment|設計 Offline / Online Evaluation|建立整個組織可重用的 Evaluation Framework|
|Production|解決效能與可靠性問題|在成本、可靠性、風險之間進行系統級取捨|
|Leadership|指導 Junior/Mid Engineers，完成專案|帶領多個團隊，建立共同標準，解決跨部門問題|
|Ownership|Own 一項 Feature 或 Subsystem|Own 一個產品領域或跨團隊技術方向|

一個很重要的觀念：

Senior 工程師不只是能訓練出很準確的模型，而是能對整套系統在 Production 的結果負責。Staff 工程師則進一步要確保其他團隊也能持續交付這樣的系統。

# Part 2. Computer Vision System Design

## 案例一：Design an Automated Watch Inspection and Authentication System

面試題目

> Design an end-to-end computer vision system that automatically captures images of luxury watches, analyzes their components, detects anomalies, and determines whether each component is authentic. Explain data collection, ground truth, model selection, validation, production scalability, and failure handling.

我會使用你熟悉的 Moonlight 類型系統來回答，這比一般的貓狗分類或瑕疵檢測案例更能展現跨 Hardware、Computer Vision、ML、Software 的能力。

## 2.1 Step 1 — Requirement Clarification

在面試時，不要一聽到題目就直接說「我會使用 YOLO 或 ViT」。

應該先詢問業務需求、系統限制與判斷錯誤的代價。

### 假設系統需求

|項目|假設規格|
|---|---|
|檢測產品|多個系列的 Rolex 手錶|
|Capture 硬體|1 個 Micro Camera、2 個 Macro Cameras|
|Motion / AF|Zaber Motion Stages、Keyence Laser Autofocus|
|Images per watch|約 90 張|
|Image resolution|2K–4.5K 級，依不同相機而定|
|Raw data|平均約 2 GB/watch|
|Throughput|500 watches/month|
|Output|Component-level classification + Watch-level assessment|
|Deployment|Local Edge Inference + AWS Training Pipeline|
|重要限制|不能把可疑品輕易判定為 Authentic；證據不足時允許人工審核|

依照這個假設：

\[ 500 \times 90 = 45,000\ \text{images/month} \]

\[ 500 \times 2\text{ GB} = 1,000\text{ GB/month} \]

也就是每個月約 45,000 張影像、1 TB 的原始資料。

這個規模並不一定需要很大的 GPU Cluster。真正的困難可能在於攝影品質、稀有仿冒樣本、可靠 Ground Truth 與跨系列的泛化能力。

### 定義 Success Metrics

不要只用 Accuracy。

面試用的初始 SLO / Quality Targets

完整掃描時間 P95

# ≤ 8 min

推論時間 P95

# ≤ 30 sec

關鍵影像 Capture 成功率

# ≥ 99%

系統服務可用率

# ≥ 99.5%

這些是待測試驗證的規格，不能直接當成已實現的測試成績；影像完整率也不等於辨識準確率。

對於 Authentication，還要分開定義：

- False Acceptance Rate：仿冒零件被判定為真品的比例。
    
- False Rejection Rate：真品零件被錯誤拒絕的比例。
    
- Manual Review Rate：系統無法做可靠判斷而轉人工的比例。
    
- Coverage：系統願意自動給出判斷的比例。
    
- Calibration：模型的信心分數是否反映實際正確機率。
    

這些指標互相牽制。例如降低 False Acceptance 通常會提高人工審查量。

Staff Engineer 應該進一步問：

> What is the business cost of falsely accepting a counterfeit component compared with sending an authentic component for manual review?

先確認錯誤代價，才能決定模型的 Threshold 和 Escalation Policy。

## 2.2 Step 2 — 完整 System Architecture

#chatgpt-mermaid-_r_40t_{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;fill:rgb(13, 13, 13);}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#chatgpt-mermaid-_r_40t_ .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-_r_40t_ .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-_r_40t_ .error-icon{fill:rgb(243, 243, 243);}#chatgpt-mermaid-_r_40t_ .error-text{fill:rgb(13, 13, 13);stroke:rgb(13, 13, 13);}#chatgpt-mermaid-_r_40t_ .edge-thickness-normal{stroke-width:1px;}#chatgpt-mermaid-_r_40t_ .edge-thickness-thick{stroke-width:3.5px;}#chatgpt-mermaid-_r_40t_ .edge-pattern-solid{stroke-dasharray:0;}#chatgpt-mermaid-_r_40t_ .edge-thickness-invisible{stroke-width:0;fill:none;}#chatgpt-mermaid-_r_40t_ .edge-pattern-dashed{stroke-dasharray:3;}#chatgpt-mermaid-_r_40t_ .edge-pattern-dotted{stroke-dasharray:2;}#chatgpt-mermaid-_r_40t_ .marker{fill:rgb(143, 143, 143);stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_40t_ .marker.cross{stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_40t_ svg{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;}#chatgpt-mermaid-_r_40t_ p{margin:0;}#chatgpt-mermaid-_r_40t_ .label{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_40t_ .cluster-label text{fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_40t_ .cluster-label span{color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_40t_ .cluster-label span p{background-color:transparent;}#chatgpt-mermaid-_r_40t_ .label text,#chatgpt-mermaid-_r_40t_ span{fill:rgb(13, 13, 13);color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_40t_ .node rect,#chatgpt-mermaid-_r_40t_ .node circle,#chatgpt-mermaid-_r_40t_ .node ellipse,#chatgpt-mermaid-_r_40t_ .node polygon,#chatgpt-mermaid-_r_40t_ .node path{fill:rgb(222, 234, 251);stroke:rgb(83, 154, 248);stroke-width:1px;}#chatgpt-mermaid-_r_40t_ .rough-node .label text,#chatgpt-mermaid-_r_40t_ .node .label text,#chatgpt-mermaid-_r_40t_ .image-shape .label,#chatgpt-mermaid-_r_40t_ .icon-shape .label{text-anchor:middle;}#chatgpt-mermaid-_r_40t_ .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#chatgpt-mermaid-_r_40t_ .rough-node .label,#chatgpt-mermaid-_r_40t_ .node .label,#chatgpt-mermaid-_r_40t_ .image-shape .label,#chatgpt-mermaid-_r_40t_ .icon-shape .label{text-align:center;}#chatgpt-mermaid-_r_40t_ .node.clickable{cursor:pointer;}#chatgpt-mermaid-_r_40t_ .root .anchor path{fill:rgb(143, 143, 143)!important;stroke-width:0;stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_40t_ .arrowheadPath{fill:rgb(143, 143, 143);}#chatgpt-mermaid-_r_40t_ .edgePath .path{stroke:rgb(143, 143, 143);stroke-width:1px;}#chatgpt-mermaid-_r_40t_ .flowchart-link{stroke:rgb(143, 143, 143);fill:none;}#chatgpt-mermaid-_r_40t_ .edgeLabel{background-color:rgb(252, 252, 252);text-align:center;}#chatgpt-mermaid-_r_40t_ .edgeLabel p{background-color:rgb(252, 252, 252);}#chatgpt-mermaid-_r_40t_ .edgeLabel rect{opacity:0.5;background-color:rgb(252, 252, 252);fill:rgb(252, 252, 252);}#chatgpt-mermaid-_r_40t_ .labelBkg{background-color:rgba(252, 252, 252, 0.5);}#chatgpt-mermaid-_r_40t_ .cluster rect{fill:rgb(243, 243, 243);stroke:rgba(0, 0, 0, 0.1);stroke-width:1px;}#chatgpt-mermaid-_r_40t_ .cluster text{fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_40t_ .cluster span{color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_40t_ div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:12px;background:rgb(243, 243, 243);border:1px solid rgba(0, 0, 0, 0.1);border-radius:2px;pointer-events:none;z-index:100;}#chatgpt-mermaid-_r_40t_ .flowchartTitleText{text-anchor:middle;font-size:18px;fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_40t_ rect.text{fill:none;stroke-width:0;}#chatgpt-mermaid-_r_40t_ .icon-shape,#chatgpt-mermaid-_r_40t_ .image-shape{background-color:rgb(252, 252, 252);text-align:center;}#chatgpt-mermaid-_r_40t_ .icon-shape p,#chatgpt-mermaid-_r_40t_ .image-shape p{background-color:rgb(252, 252, 252);padding:2px;}#chatgpt-mermaid-_r_40t_ .icon-shape .label rect,#chatgpt-mermaid-_r_40t_ .image-shape .label rect{opacity:0.5;background-color:rgb(252, 252, 252);fill:rgb(252, 252, 252);}#chatgpt-mermaid-_r_40t_ .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#chatgpt-mermaid-_r_40t_ .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#chatgpt-mermaid-_r_40t_ .node .neo-node{stroke:rgb(83, 154, 248);}#chatgpt-mermaid-_r_40t_ [data-look="neo"].node rect,#chatgpt-mermaid-_r_40t_ [data-look="neo"].cluster rect,#chatgpt-mermaid-_r_40t_ [data-look="neo"].node polygon{stroke:url(#chatgpt-mermaid-_r_40t_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_40t_ [data-look="neo"].swimlane.cluster rect{filter:none;}#chatgpt-mermaid-_r_40t_ [data-look="neo"].node path{stroke:url(#chatgpt-mermaid-_r_40t_-gradient);stroke-width:1px;}#chatgpt-mermaid-_r_40t_ [data-look="neo"].node .outer-path{filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_40t_ [data-look="neo"].node .neo-line path{stroke:rgb(83, 154, 248);filter:none;}#chatgpt-mermaid-_r_40t_ [data-look="neo"].node circle{stroke:url(#chatgpt-mermaid-_r_40t_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_40t_ [data-look="neo"].node circle .state-start{fill:#000000;}#chatgpt-mermaid-_r_40t_ [data-look="neo"].icon-shape .icon{fill:url(#chatgpt-mermaid-_r_40t_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_40t_ [data-look="neo"].icon-shape .icon-neo path{stroke:url(#chatgpt-mermaid-_r_40t_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_40t_ .node text{font-size:14px;font-weight:600;letter-spacing:normal;fill:rgb(65, 65, 65);}#chatgpt-mermaid-_r_40t_ .edgeLabels text{font-size:13px;font-weight:600;letter-spacing:-0.08px;fill:rgb(65, 65, 65);}#chatgpt-mermaid-_r_40t_ .node tspan[font-weight="normal"],#chatgpt-mermaid-_r_40t_ .edgeLabels tspan[font-weight="normal"]{font-weight:600;}#chatgpt-mermaid-_r_40t_ .edgeLabel .label rect{opacity:1;rx:13px;ry:13px;fill:rgb(252, 252, 252);stroke:rgb(219, 219, 219);stroke-width:1px;}#chatgpt-mermaid-_r_40t_ .node rect,#chatgpt-mermaid-_r_40t_ .node circle,#chatgpt-mermaid-_r_40t_ .node ellipse,#chatgpt-mermaid-_r_40t_ .node polygon,#chatgpt-mermaid-_r_40t_ .node path{fill:rgb(243, 243, 243);stroke:rgba(0, 0, 0, 0.1);stroke-width:1px;}#chatgpt-mermaid-_r_40t_ .node rect{rx:16px;ry:16px;}#chatgpt-mermaid-_r_40t_ .node.mermaid-decision .label-container{fill:rgb(249, 249, 249);stroke:rgb(219, 219, 219);stroke-dasharray:2,2;}#chatgpt-mermaid-_r_40t_ .edgePaths .flowchart-link{stroke:rgb(143, 143, 143);stroke-width:1px;stroke-linecap:round;stroke-linejoin:round;}#chatgpt-mermaid-_r_40t_ .marker{fill:rgb(143, 143, 143);stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_40t_ :root{--mermaid-font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";}Watch / Physical SpecimenMotion + Camera + Lighting +AutofocusImage Quality GateImage PreprocessingBounded Recapture / ReviewSegmentation + RegistrationFeature ExtractionStatistical Evidence + CVModelsHierarchical Bayesian FusionOOD + UncertaintyAssessmentDecision PolicyComponent ResultHuman Expert ReviewIncomplete AssessmentValidated LabelsLocal DB + AuditS3 Raw Image ArchiveVersioned Training DatasetAWS Training + EvaluationModel Registry + ApprovalCanary + Edge DeploymentPassFailHigh confidenceUncertainMissing evidence

圖一：Online inference 與 Offline training 分開設計。即時掃描不依賴 AWS 持續連線。

整套系統可以分成三條 Pipeline：

Online Capture Pipeline

Watch → Camera → Autofocus → Image Quality → Image Storage

Online Inference Pipeline

Image → Segmentation → Feature Extraction → Classification / Statistical Evidence → Fusion → Decision

Offline Training Pipeline

S3 Dataset → Data Validation → Training → Evaluation → Model Registry → Deployment

這樣拆分的原因非常重要：

如果雲端 Training 發生故障，Local Edge 仍然可以使用已核准的模型繼續掃描手錶。

如果新模型出現問題，也能直接 Rollback 到上一個版本。

## 2.3 Step 3 — Data Collection Design

這是 Computer Vision System Design 最容易被低估的部分。

面試官可能會問：

> How would you collect enough representative data, especially when counterfeit examples are rare?

這裡要先區分 Data Quantity 與 Data Diversity。

假設你有 100,000 張 Genuine Watch Images，不代表你就有足夠資料辨識仿冒品。

因為這些影像可能全來自相同相機、相同光線、相同錶款，甚至只是少數幾只錶重複拍攝。

### Data Collection Strategy

|分類|需要收集的資料|為什麼重要|
|---|---|---|
|Product|Series、Reference、年份、零件版本|不同版本的真品可能存在差異|
|Authenticity|Original、Forgery、Aftermarket 等|支援多種 Component Status|
|Camera|Macro1、Macro2、Micro|不同解析度、顏色和 Noise 特性|
|Imaging|Exposure、Gain、Focus、Illumination|避免只適用單一攝影條件|
|Geometry|Angle、Rotation、Position|確保姿態變化下仍有效|
|Quality|Blur、Glare、Occlusion、Incomplete Image|偵測不可用影像|
|Provenance|Watch ID、Component ID、來源、查驗記錄|防止標註錯誤|
|Hardware|Camera / Lens / Firmware / Calibration Versions|支援 Domain Shift 分析|

### 實際資料格式

假設每次掃描都有：

```
{
  "watch_id": "W000128",
  "scan_session_id": "S00421",
  "series": "Series_A1",
  "component": "Dial",
  "camera_id": "Macro1",
  "view_id": "Front_0004",
  "capture_mode": "HDR",
  "focus_status": "pass",
  "image_quality": {
    "blur_score": 0.08,
    "saturation_ratio": 0.002
  },
  "capture_timestamp": "2026-10-10T10:30:00Z",
  "hardware_config_version": "hw-v12",
  "preprocessing_version": "img-v7",
  "label_version": "label-v3"
}
```

實際系統還要保存原始圖片雜湊、影像處理參數、Exposure、Camera Calibration、Capture Attempt ID 等。

關鍵設計：不要只保存 Image 和 Label，還要保存影響影像生成的 Hardware / Imaging Metadata。

不然可能發生：

模型準確率突然下降 10%，Research Team 花兩週重新訓練，結果根本不是模型問題，而是某個 Camera 的 White Balance、Lighting 或 Focus Calibration 發生變化。

## 2.4 Step 4 — Ground Truth 如何建立？

Ground Truth 不等於「找人看圖片，標一個 Label」。

尤其在真偽鑑定中，某個專家看圖片覺得是 Original，不代表這個判斷可以直接當成已確認的事實。

應建立可信度分級：

|Level|Ground Truth 來源|使用方式|
|---|---|---|
|A|製造或維修來源可追溯，並經專家驗證的實體零件|高可信度 Reference / Gold Test Set|
|B|至少兩位獨立專家判定一致，有足夠佐證|Training / Validation|
|C|專家判斷不一致，或缺少關鍵來源證據|Adjudication Queue|
|D|來源不明、僅有未核實賣家描述|Unverified Pool，不當作確定真值|

A 和 B 並不代表完全沒有錯誤，但必須有明確追溯機制。

### Annotation Workflow

#chatgpt-mermaid-_r_41d_{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;fill:rgb(13, 13, 13);}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#chatgpt-mermaid-_r_41d_ .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-_r_41d_ .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-_r_41d_ .error-icon{fill:rgb(243, 243, 243);}#chatgpt-mermaid-_r_41d_ .error-text{fill:rgb(13, 13, 13);stroke:rgb(13, 13, 13);}#chatgpt-mermaid-_r_41d_ .edge-thickness-normal{stroke-width:1px;}#chatgpt-mermaid-_r_41d_ .edge-thickness-thick{stroke-width:3.5px;}#chatgpt-mermaid-_r_41d_ .edge-pattern-solid{stroke-dasharray:0;}#chatgpt-mermaid-_r_41d_ .edge-thickness-invisible{stroke-width:0;fill:none;}#chatgpt-mermaid-_r_41d_ .edge-pattern-dashed{stroke-dasharray:3;}#chatgpt-mermaid-_r_41d_ .edge-pattern-dotted{stroke-dasharray:2;}#chatgpt-mermaid-_r_41d_ .marker{fill:rgb(143, 143, 143);stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_41d_ .marker.cross{stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_41d_ svg{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;}#chatgpt-mermaid-_r_41d_ p{margin:0;}#chatgpt-mermaid-_r_41d_ .label{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_41d_ .cluster-label text{fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_41d_ .cluster-label span{color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_41d_ .cluster-label span p{background-color:transparent;}#chatgpt-mermaid-_r_41d_ .label text,#chatgpt-mermaid-_r_41d_ span{fill:rgb(13, 13, 13);color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_41d_ .node rect,#chatgpt-mermaid-_r_41d_ .node circle,#chatgpt-mermaid-_r_41d_ .node ellipse,#chatgpt-mermaid-_r_41d_ .node polygon,#chatgpt-mermaid-_r_41d_ .node path{fill:rgb(222, 234, 251);stroke:rgb(83, 154, 248);stroke-width:1px;}#chatgpt-mermaid-_r_41d_ .rough-node .label text,#chatgpt-mermaid-_r_41d_ .node .label text,#chatgpt-mermaid-_r_41d_ .image-shape .label,#chatgpt-mermaid-_r_41d_ .icon-shape .label{text-anchor:middle;}#chatgpt-mermaid-_r_41d_ .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#chatgpt-mermaid-_r_41d_ .rough-node .label,#chatgpt-mermaid-_r_41d_ .node .label,#chatgpt-mermaid-_r_41d_ .image-shape .label,#chatgpt-mermaid-_r_41d_ .icon-shape .label{text-align:center;}#chatgpt-mermaid-_r_41d_ .node.clickable{cursor:pointer;}#chatgpt-mermaid-_r_41d_ .root .anchor path{fill:rgb(143, 143, 143)!important;stroke-width:0;stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_41d_ .arrowheadPath{fill:rgb(143, 143, 143);}#chatgpt-mermaid-_r_41d_ .edgePath .path{stroke:rgb(143, 143, 143);stroke-width:1px;}#chatgpt-mermaid-_r_41d_ .flowchart-link{stroke:rgb(143, 143, 143);fill:none;}#chatgpt-mermaid-_r_41d_ .edgeLabel{background-color:rgb(252, 252, 252);text-align:center;}#chatgpt-mermaid-_r_41d_ .edgeLabel p{background-color:rgb(252, 252, 252);}#chatgpt-mermaid-_r_41d_ .edgeLabel rect{opacity:0.5;background-color:rgb(252, 252, 252);fill:rgb(252, 252, 252);}#chatgpt-mermaid-_r_41d_ .labelBkg{background-color:rgba(252, 252, 252, 0.5);}#chatgpt-mermaid-_r_41d_ .cluster rect{fill:rgb(243, 243, 243);stroke:rgba(0, 0, 0, 0.1);stroke-width:1px;}#chatgpt-mermaid-_r_41d_ .cluster text{fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_41d_ .cluster span{color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_41d_ div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:12px;background:rgb(243, 243, 243);border:1px solid rgba(0, 0, 0, 0.1);border-radius:2px;pointer-events:none;z-index:100;}#chatgpt-mermaid-_r_41d_ .flowchartTitleText{text-anchor:middle;font-size:18px;fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_41d_ rect.text{fill:none;stroke-width:0;}#chatgpt-mermaid-_r_41d_ .icon-shape,#chatgpt-mermaid-_r_41d_ .image-shape{background-color:rgb(252, 252, 252);text-align:center;}#chatgpt-mermaid-_r_41d_ .icon-shape p,#chatgpt-mermaid-_r_41d_ .image-shape p{background-color:rgb(252, 252, 252);padding:2px;}#chatgpt-mermaid-_r_41d_ .icon-shape .label rect,#chatgpt-mermaid-_r_41d_ .image-shape .label rect{opacity:0.5;background-color:rgb(252, 252, 252);fill:rgb(252, 252, 252);}#chatgpt-mermaid-_r_41d_ .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#chatgpt-mermaid-_r_41d_ .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#chatgpt-mermaid-_r_41d_ .node .neo-node{stroke:rgb(83, 154, 248);}#chatgpt-mermaid-_r_41d_ [data-look="neo"].node rect,#chatgpt-mermaid-_r_41d_ [data-look="neo"].cluster rect,#chatgpt-mermaid-_r_41d_ [data-look="neo"].node polygon{stroke:url(#chatgpt-mermaid-_r_41d_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_41d_ [data-look="neo"].swimlane.cluster rect{filter:none;}#chatgpt-mermaid-_r_41d_ [data-look="neo"].node path{stroke:url(#chatgpt-mermaid-_r_41d_-gradient);stroke-width:1px;}#chatgpt-mermaid-_r_41d_ [data-look="neo"].node .outer-path{filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_41d_ [data-look="neo"].node .neo-line path{stroke:rgb(83, 154, 248);filter:none;}#chatgpt-mermaid-_r_41d_ [data-look="neo"].node circle{stroke:url(#chatgpt-mermaid-_r_41d_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_41d_ [data-look="neo"].node circle .state-start{fill:#000000;}#chatgpt-mermaid-_r_41d_ [data-look="neo"].icon-shape .icon{fill:url(#chatgpt-mermaid-_r_41d_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_41d_ [data-look="neo"].icon-shape .icon-neo path{stroke:url(#chatgpt-mermaid-_r_41d_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_41d_ .node text{font-size:14px;font-weight:600;letter-spacing:normal;fill:rgb(65, 65, 65);}#chatgpt-mermaid-_r_41d_ .edgeLabels text{font-size:13px;font-weight:600;letter-spacing:-0.08px;fill:rgb(65, 65, 65);}#chatgpt-mermaid-_r_41d_ .node tspan[font-weight="normal"],#chatgpt-mermaid-_r_41d_ .edgeLabels tspan[font-weight="normal"]{font-weight:600;}#chatgpt-mermaid-_r_41d_ .edgeLabel .label rect{opacity:1;rx:13px;ry:13px;fill:rgb(252, 252, 252);stroke:rgb(219, 219, 219);stroke-width:1px;}#chatgpt-mermaid-_r_41d_ .node rect,#chatgpt-mermaid-_r_41d_ .node circle,#chatgpt-mermaid-_r_41d_ .node ellipse,#chatgpt-mermaid-_r_41d_ .node polygon,#chatgpt-mermaid-_r_41d_ .node path{fill:rgb(243, 243, 243);stroke:rgba(0, 0, 0, 0.1);stroke-width:1px;}#chatgpt-mermaid-_r_41d_ .node rect{rx:16px;ry:16px;}#chatgpt-mermaid-_r_41d_ .node.mermaid-decision .label-container{fill:rgb(249, 249, 249);stroke:rgb(219, 219, 219);stroke-dasharray:2,2;}#chatgpt-mermaid-_r_41d_ .edgePaths .flowchart-link{stroke:rgb(143, 143, 143);stroke-width:1px;stroke-linecap:round;stroke-linejoin:round;}#chatgpt-mermaid-_r_41d_ .marker{fill:rgb(143, 143, 143);stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_41d_ :root{--mermaid-font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";}Physical Watch + EvidenceExpert A: Independent ReviewExpert B: Independent ReviewAgreement?Validated Component LabelSenior Expert + AdditionalEvidenceResolvable?Uncertain / Excluded fromGold SetVersioned Ground TruthDatasetYesNoYesNo

具體例子：

有一個 Rolex Dial，其印刷、Hour Markers、文字位置幾乎完全符合 Original Reference。

Expert A 判 Original。

Expert B 判 Authentic Replacement。

這時候兩個人可能都沒有犯視覺辨識錯誤。

因為「Original Dial」和「原廠更換的 Authentic Replacement Dial」在外觀上可能非常相似，真正差別來自維修歷史或來源。

所以有時候 Ground Truth 是一個多來源證據問題，而不只是 Computer Vision 問題。

Senior Engineer 應該提出 Label Disagreement Analysis。

Staff Engineer 還應該訂立整個團隊的 Annotation Policy，例如：

- 必須使用哪個版本的標註準則？
    
- 哪些狀態需要物理鑑識、文件或來源證據？
    
- 哪些 Label 允許影像單獨支持？
    
- 如何處理沒有共識的樣本？
    
- 如何監測不同專家的 Agreement Rate？
    
- 變更 Label Definition 時是否要重新審核舊 Dataset？
    

例如用 Cohen's Kappa 衡量兩位標註者超過隨機一致程度的 Agreement，但仍需要配合 Confusion Matrix 與爭議案例審查，因為類別分布高度不平衡可能影響 Kappa 的解讀。

## 2.5 Step 5 — 選擇模型：為什麼不是直接使用一個大型 CNN？

這是面試官經常追問的地方：

> Why did you choose this model? What alternatives did you consider?

我會先建立簡單 Baseline，再逐步提高複雜度。

|Approach|優勢|缺點|適合|
|---|---|---|---|
|Template Matching / Geometry|可解釋、低成本|對變形、視角敏感|特定文字、位置、尺寸|
|Classical CV + Statistics|小資料也能使用|需要設計特徵|紋理、邊緣、幾何與量測|
|CNN / ViT Classifier|能學習複雜視覺特徵|需要代表性資料|有足夠樣本的封閉類別|
|Segmentation Model|Pixel-level localization|Annotation Cost 高|Hands、Markers、Components|
|Metric Learning|能比較 Reference 與 Query|需設計良好的正負樣本|Series / Component Variant 比對|
|Anomaly Detection|可利用大量 Genuine Data|不保證所有仿冒品都顯著異常|Rare / Unknown Counterfeits|
|Hierarchical Bayesian Fusion|結合多零件、多證據與不確定性|依賴可靠 Likelihood / Calibration|Watch-level Decision|

### 我會選擇 Hybrid Statistical–Bayesian Architecture

Raw Images / Segmented Regions

Robust Visual Features + Deep Embeddings

Calibrated Component Evidence

Hierarchical Bayesian Evidence Fusion

Anomaly / OOD / Missing Evidence Handling

Decision Policy → Accept / Flag / Review / Incomplete

### 具體例子：判定 Dial 是否為 Forgery

假設我們從 Dial Image 得到：

|Feature|Reference|Scan Result|
|---|---|---|
|Logo Width|1.240 mm|1.255 mm|
|Text Baseline Angle|0.10°|0.25°|
|Marker Spacing|3.510 mm|3.470 mm|
|Character Stroke Width|0.085 mm|0.102 mm|
|Deep Embedding Similarity|—|0.91|

以上數值純屬示例。

不要直接說「0.91 很高，因此是真品」。

正確做法是先分析這些特徵在 Genuine、Forgery、不同錶款及攝影條件下的分布。

例如先用 Genuine Reference 建立多變量分布：

\[ \mathbf{x}=[x_1,x_2,\ldots,x_d] \]

對某個 Reference Group，可以利用 Robust Covariance 估計出：

\[ D_M^2=(\mathbf{x}-\boldsymbol{\mu})^T \Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu}) \]

這是 Mahalanobis Distance，可以判斷 Scan Feature 與 Reference Distribution 的距離。

但它不是直接的「仿冒機率」，也不保證所有仿冒品都會有較大的 Distance。

接著可以使用經驗分布、Density Model 或校準後的 Classifier，建立各類別的 Evidence Likelihood：

\[ P(E_{\text{dial}}\mid C_{\text{dial}}) \]

再與 Hands、Bezel、Movement 等 Component Evidence 進行 Fusion。

Bayesian 形式為：

\[ P(C\mid E) = \frac{P(E\mid C)P(C)}{P(E)} \]

不過面試中要主動說明兩個風險。

第一：Evidence Correlation。 同一張影像算出的 20 個字體特徵不是 20 份獨立證據。如果直接把它們的 Likelihood 全部相乘，可能嚴重高估 Confidence。

第二：Prior Shift。 不同市場、資料來源的 Genuine / Forgery 比例不同。Training Dataset 的 Class Balance 不能直接當成 Production Prior。

所以我會使用分層模型，先在 Component Level 處理特徵相關性，再做 Watch-level Fusion，最後經過 Calibration 和 Policy Layer。

而且系統不應把影像無法觀察的維修歷史，假裝成可以由 AI 直接推論的資訊。

這是比直接建立一個八類 CNN 更合理的起始架構。

## 2.6 Step 6 — Experiment Design 與 Validation

這部分經常是 Senior 和普通 ML Engineer 的關鍵差距。

### 最重要：避免 Data Leakage

假設同一只 Rolex 被拍攝 90 張影像。

如果隨機把 70 張放 Training，20 張放 Testing，模型可能只是記住這只錶的特徵，而不是學會辨識未見過的手錶。

因此 Split 的單位不是 Image，而是 Physical Watch / Specimen ID。

同一只錶的所有 Scan Sessions、Augmentations、Stitched Images 必須放在同一個 Partition。

再考慮以下分組驗證：

|Test Set|測試目的|
|---|---|
|In-distribution Holdout|同一 Series、未見過的實體手錶|
|New Capture Session|同一硬體設定下的新拍攝批次|
|Hardware Holdout|不同相機、光源或校準配置|
|Temporal Holdout|較晚時間收集的新資料|
|Variant Holdout|未出現在訓練中的零件版本|
|Unknown Forgery Holdout|新型態、未參與模型調整的仿冒品|
|Missing Evidence Set|缺少部分影像仍能否合理決策|
|Image Quality Stress Set|Blur、Glare、Occlusion 等|

Google 的 Production ML 指引也特別強調 Training-Serving Skew、時間切分驗證，以及 Training / Serving Feature 處理一致性的重要性。

![](https://www.google.com/s2/favicons?domain=https://developers.google.com&sz=32)

Google for Developers

+1

### 我會設計四組 Ablation Experiments

|Experiment|目標|
|---|---|
|A: Classical Features Only|Establish interpretable baseline|
|B: Deep Learning Only|Understand learned visual representation|
|C: Hybrid Features + Deep Embeddings|Verify complementary evidence|
|D: Hybrid + Bayesian Fusion + Review Policy|Measure end-to-end business impact|

除了比較 AUC、F1、Macro F1、PR-AUC，也要比較：

- False Accept Rate at a fixed Review Rate
    
- Genuine Rejection Rate
    
- Selective Risk versus Coverage
    
- Confidence Calibration
    
- Per-Series / Per-Component Performance
    
- Latency / Memory / Operational Cost
    

還應提供 Confidence Interval，而不是只報告一個整體 Accuracy。

例如，假設 Test Set 只有 100 個獨立的 Forgery Specimens，模型全部成功辨識。

你仍不能宣稱模型的 Miss Rate 低於 0.1%。

因為用簡化的 rule of three，在 100 個獨立試驗中觀察到零次失敗，失敗率的單側約 95% 上限仍接近：

\[ \frac{3}{100}=3\% \]

要驗證極低的 False Acceptance Rate，需要更多獨立且有代表性的仿冒樣本。這也是 Staff Engineer 必須向 Product / Management 清楚說明的統計限制。

## 2.7 Step 7 — Production Bottleneck 與 Failure Modes

面試官可能問：

> Your model works in the lab, but production throughput is 40% lower than expected. How would you debug it?

我不會第一時間就優化 GPU Inference。

因為這類系統的 Bottleneck 可能出現在 Camera、Motion、Autofocus、Image Transfer、HDR Processing 或人工操作，而不是模型。

### Production Critical Path

\[ T_{\text{scan}} = T_{\text{setup}} + T_{\text{motion}} + T_{\text{focus}} + T_{\text{capture}} + T_{\text{transfer}} + T_{\text{inference}} + T_{\text{review}} \]

這個式子是假設各階段串行的簡化模型。若有 Pipeline Overlap，真正的 Critical Path 需要由 Distributed Traces / Timing Logs 計算，不能把所有耗時直接相加。

第一步會建立完整 Timing Breakdown。

例如一次掃描量測到：

假設量測：單只手錶各階段耗時

用來示範 bottleneck analysis，並非實測資料。

0 sec35 sec70 sec105 sec140 secMotionAutofocusCaptureTransferPreprocessInference

這個例子告訴我們：

Autofocus 130 秒，Inference 只有 18 秒。

即使將 GPU 推論速度提升兩倍，也只能減少 9 秒；但如果能把 Autofocus 優化到 70 秒，理論上可以節省 60 秒。

所以優先順序應該由 Impact / Engineering Effort / Risk 決定。

### Failure Mode Analysis

|Failure|如何偵測|Recovery / Mitigation|
|---|---|---|
|Autofocus Fail|Focus Score、Keyence Status、Boundary Condition|Bounded Retry、Alternate AF、人工介入|
|Motion Stage Stall|Controller Error、Position Mismatch|停止危險動作、檢查 Interlock、需時進行人工 Recovery|
|Camera Timeout|Missing Frame、Acquisition Timeout|Reset Acquisition，有限次重試|
|Lighting Drift|Reference Target、Intensity Statistics|Calibration Alert、阻擋不合格資料|
|HDR Artifact|Alignment / Ghosting / Clipping Metrics|重新拍攝或改用允許的 Capture Mode|
|GPU OOM|Memory Usage、CUDA Error|Smaller Batch、資源回收、明確 Failed State|
|Model Drift|Feature Distribution、Quality Metrics|Investigation / Revalidation / Rollback|
|Unknown Counterfeit|OOD Score、Low Confidence|Manual Review，而非強制 Genuine|
|Network / AWS Failure|Sync Failure、Connectivity Monitoring|Local Queue + Retry + Offline Inference|

### Staff-level 的核心設計：Failure Isolation

Camera Timeout 不應直接讓整個 App Crash。

雲端 Upload Failure 不應讓 Local Authentication 停止。

一個可疑 Component 不應被其他 Genuine Component 的高分完全抵銷。

需要使用 State Machine 明確區分：

`CAPTURED → QUALITY_PASSED → ANALYZED → DECIDED`

以及：

`RECAPTURE_REQUIRED / INCOMPLETE / REVIEW_REQUIRED / HARDWARE_ERROR`

每個 Step 要有 Trace ID、Watch ID、Retry Count、Error Category、Model Version。

另外，State Recovery 必須遵守硬體安全限制：對 Stage Stall 不能無限重試移動，也不能因為 Software Retry 而繞過 Safety Interlock。

### Deployment Strategy

使用：

Offline Validation → Shadow Inference → Limited Canary → Production Rollout → Monitoring → Rollback

Shadow Inference 是讓新模型對真實資料執行預測，但不影響最終認證決策。

Canary 則是選擇少數設備或掃描流程實際使用新模型，持續監測 Performance、Latency、Review Rate 和安全指標。

每個版本應有可重現的：

`dataset_version + label_version + preprocessing_version + model_version + decision_policy_version`

最後才能確實回答：

> Why did watch W000128 receive a different result after model deployment?

能回答這個問題，才算真正建立具備可維護性和可稽核性的 AI Production System。

# Part 3. LLM System Design

## 案例二：Design an Enterprise AI Troubleshooting Assistant with RAG and Agents

面試題目

> Design a production-grade LLM system that helps field engineers diagnose industrial equipment failures. It should understand technical manuals, historical incidents, device logs, and real-time telemetry, provide evidence-backed recommendations, and safely interact with diagnostic tools.

這是 2026 年 Senior / Staff LLM Engineer 很值得準備的一種類型，因為它同時涉及：

LLM、Reasoning、RAG、Agent、Tool Calling、Evaluation、Data Infrastructure、Security、Reliability、Cost Optimization 和 Technical Leadership。

我們假設一家工業設備公司有：

- 2,000 位 Field Engineers。
    
- 200,000 份 Technical Manuals、SOP、Service Bulletins。
    
- 500,000 筆歷史故障和維修 Ticket。
    
- 每天 2,000 次 AI 問答。
    
- 一套可以查詢設備即時 Logs、Telemetry 和 Configuration 的 API。
    

當工程師看到以下錯誤：

> Zaber X-RST120AK rotation stage is stalled at 87 degrees with an FS fault. What is the likely cause, and what diagnostic steps should I take?

AI 必須能查詢正確機型的 Manual、歷史故障案例和最新 Logs，而不是只靠 LLM 內部記憶生成可能的答案。

## 3.1 Step 1 — Requirement Clarification

首先，我會和面試官確認：

這是一個單純回答問題的 Chatbot，還是能夠操作實際設備的 Agent？

這兩者在 Architecture、Security、Evaluation 上差非常多。

我會把需求分為三個能力等級。

|Level|AI 能做什麼|Risk|
|---|---|---|
|L1: Knowledge Assistant|搜尋文件、回答問題、引用來源|較低|
|L2: Diagnostic Agent|查 Logs、Telemetry、Configuration，提出診斷|中等|
|L3: Action Agent|執行診斷程序或操作設備|高，需要額外授權和安全控制|

我的設計會先交付 L1 與 Read-only L2，再逐步引入經過審核的 L3。

原因是設備控制可能產生物理風險，LLM 不應因為自己判斷可能正確，就直接命令硬體移動。

### Production Requirements

以下是假設目標：

|指標|初始要求|
|---|---|
|Knowledge Q&A Latency|P95 ≤ 8 seconds|
|Multi-step Diagnostic Latency|P95 ≤ 30 seconds|
|Availability|99.9%|
|Answer Quality|使用專家評分、Groundedness、Task Success 衡量|
|Security|不允許跨使用者或跨客戶資料洩漏|
|Tool Execution|所有寫入或硬體操作必須通過獨立授權|
|Explainability|顯示可核對的文件引用和診斷依據|
|Fallback|查無證據時可以拒答或轉人工|

注意，我不會一開始就要求「LLM Accuracy 99%」。

因為自由文字的「正確」不一定是一個簡單的 Binary Label。

我們必須先定義 Ground Truth 和 Evaluation Rubric。

## 3.2 Step 2 — 設計完整 Architecture

#chatgpt-mermaid-_r_42u_{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;fill:rgb(13, 13, 13);}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#chatgpt-mermaid-_r_42u_ .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-_r_42u_ .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-_r_42u_ .error-icon{fill:rgb(243, 243, 243);}#chatgpt-mermaid-_r_42u_ .error-text{fill:rgb(13, 13, 13);stroke:rgb(13, 13, 13);}#chatgpt-mermaid-_r_42u_ .edge-thickness-normal{stroke-width:1px;}#chatgpt-mermaid-_r_42u_ .edge-thickness-thick{stroke-width:3.5px;}#chatgpt-mermaid-_r_42u_ .edge-pattern-solid{stroke-dasharray:0;}#chatgpt-mermaid-_r_42u_ .edge-thickness-invisible{stroke-width:0;fill:none;}#chatgpt-mermaid-_r_42u_ .edge-pattern-dashed{stroke-dasharray:3;}#chatgpt-mermaid-_r_42u_ .edge-pattern-dotted{stroke-dasharray:2;}#chatgpt-mermaid-_r_42u_ .marker{fill:rgb(143, 143, 143);stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_42u_ .marker.cross{stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_42u_ svg{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;}#chatgpt-mermaid-_r_42u_ p{margin:0;}#chatgpt-mermaid-_r_42u_ .label{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_42u_ .cluster-label text{fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_42u_ .cluster-label span{color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_42u_ .cluster-label span p{background-color:transparent;}#chatgpt-mermaid-_r_42u_ .label text,#chatgpt-mermaid-_r_42u_ span{fill:rgb(13, 13, 13);color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_42u_ .node rect,#chatgpt-mermaid-_r_42u_ .node circle,#chatgpt-mermaid-_r_42u_ .node ellipse,#chatgpt-mermaid-_r_42u_ .node polygon,#chatgpt-mermaid-_r_42u_ .node path{fill:rgb(222, 234, 251);stroke:rgb(83, 154, 248);stroke-width:1px;}#chatgpt-mermaid-_r_42u_ .rough-node .label text,#chatgpt-mermaid-_r_42u_ .node .label text,#chatgpt-mermaid-_r_42u_ .image-shape .label,#chatgpt-mermaid-_r_42u_ .icon-shape .label{text-anchor:middle;}#chatgpt-mermaid-_r_42u_ .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#chatgpt-mermaid-_r_42u_ .rough-node .label,#chatgpt-mermaid-_r_42u_ .node .label,#chatgpt-mermaid-_r_42u_ .image-shape .label,#chatgpt-mermaid-_r_42u_ .icon-shape .label{text-align:center;}#chatgpt-mermaid-_r_42u_ .node.clickable{cursor:pointer;}#chatgpt-mermaid-_r_42u_ .root .anchor path{fill:rgb(143, 143, 143)!important;stroke-width:0;stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_42u_ .arrowheadPath{fill:rgb(143, 143, 143);}#chatgpt-mermaid-_r_42u_ .edgePath .path{stroke:rgb(143, 143, 143);stroke-width:1px;}#chatgpt-mermaid-_r_42u_ .flowchart-link{stroke:rgb(143, 143, 143);fill:none;}#chatgpt-mermaid-_r_42u_ .edgeLabel{background-color:rgb(252, 252, 252);text-align:center;}#chatgpt-mermaid-_r_42u_ .edgeLabel p{background-color:rgb(252, 252, 252);}#chatgpt-mermaid-_r_42u_ .edgeLabel rect{opacity:0.5;background-color:rgb(252, 252, 252);fill:rgb(252, 252, 252);}#chatgpt-mermaid-_r_42u_ .labelBkg{background-color:rgba(252, 252, 252, 0.5);}#chatgpt-mermaid-_r_42u_ .cluster rect{fill:rgb(243, 243, 243);stroke:rgba(0, 0, 0, 0.1);stroke-width:1px;}#chatgpt-mermaid-_r_42u_ .cluster text{fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_42u_ .cluster span{color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_42u_ div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:12px;background:rgb(243, 243, 243);border:1px solid rgba(0, 0, 0, 0.1);border-radius:2px;pointer-events:none;z-index:100;}#chatgpt-mermaid-_r_42u_ .flowchartTitleText{text-anchor:middle;font-size:18px;fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_42u_ rect.text{fill:none;stroke-width:0;}#chatgpt-mermaid-_r_42u_ .icon-shape,#chatgpt-mermaid-_r_42u_ .image-shape{background-color:rgb(252, 252, 252);text-align:center;}#chatgpt-mermaid-_r_42u_ .icon-shape p,#chatgpt-mermaid-_r_42u_ .image-shape p{background-color:rgb(252, 252, 252);padding:2px;}#chatgpt-mermaid-_r_42u_ .icon-shape .label rect,#chatgpt-mermaid-_r_42u_ .image-shape .label rect{opacity:0.5;background-color:rgb(252, 252, 252);fill:rgb(252, 252, 252);}#chatgpt-mermaid-_r_42u_ .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#chatgpt-mermaid-_r_42u_ .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#chatgpt-mermaid-_r_42u_ .node .neo-node{stroke:rgb(83, 154, 248);}#chatgpt-mermaid-_r_42u_ [data-look="neo"].node rect,#chatgpt-mermaid-_r_42u_ [data-look="neo"].cluster rect,#chatgpt-mermaid-_r_42u_ [data-look="neo"].node polygon{stroke:url(#chatgpt-mermaid-_r_42u_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_42u_ [data-look="neo"].swimlane.cluster rect{filter:none;}#chatgpt-mermaid-_r_42u_ [data-look="neo"].node path{stroke:url(#chatgpt-mermaid-_r_42u_-gradient);stroke-width:1px;}#chatgpt-mermaid-_r_42u_ [data-look="neo"].node .outer-path{filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_42u_ [data-look="neo"].node .neo-line path{stroke:rgb(83, 154, 248);filter:none;}#chatgpt-mermaid-_r_42u_ [data-look="neo"].node circle{stroke:url(#chatgpt-mermaid-_r_42u_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_42u_ [data-look="neo"].node circle .state-start{fill:#000000;}#chatgpt-mermaid-_r_42u_ [data-look="neo"].icon-shape .icon{fill:url(#chatgpt-mermaid-_r_42u_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_42u_ [data-look="neo"].icon-shape .icon-neo path{stroke:url(#chatgpt-mermaid-_r_42u_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_42u_ .node text{font-size:14px;font-weight:600;letter-spacing:normal;fill:rgb(65, 65, 65);}#chatgpt-mermaid-_r_42u_ .edgeLabels text{font-size:13px;font-weight:600;letter-spacing:-0.08px;fill:rgb(65, 65, 65);}#chatgpt-mermaid-_r_42u_ .node tspan[font-weight="normal"],#chatgpt-mermaid-_r_42u_ .edgeLabels tspan[font-weight="normal"]{font-weight:600;}#chatgpt-mermaid-_r_42u_ .edgeLabel .label rect{opacity:1;rx:13px;ry:13px;fill:rgb(252, 252, 252);stroke:rgb(219, 219, 219);stroke-width:1px;}#chatgpt-mermaid-_r_42u_ .node rect,#chatgpt-mermaid-_r_42u_ .node circle,#chatgpt-mermaid-_r_42u_ .node ellipse,#chatgpt-mermaid-_r_42u_ .node polygon,#chatgpt-mermaid-_r_42u_ .node path{fill:rgb(243, 243, 243);stroke:rgba(0, 0, 0, 0.1);stroke-width:1px;}#chatgpt-mermaid-_r_42u_ .node rect{rx:16px;ry:16px;}#chatgpt-mermaid-_r_42u_ .node.mermaid-decision .label-container{fill:rgb(249, 249, 249);stroke:rgb(219, 219, 219);stroke-dasharray:2,2;}#chatgpt-mermaid-_r_42u_ .edgePaths .flowchart-link{stroke:rgb(143, 143, 143);stroke-width:1px;stroke-linecap:round;stroke-linejoin:round;}#chatgpt-mermaid-_r_42u_ .marker{fill:rgb(143, 143, 143);stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_42u_ :root{--mermaid-font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";}Engineer UI / APIAuthentication + AuthorizationQuery Router + Risk ClassifierKnowledge RetrievalDiagnostic Tool OrchestratorManuals / SOP / TicketsIngestion + Parsing +VersioningChunking + Metadata + ACLBM25 + Vector IndexReranker + Context BuilderLive Logs / Telemetry APILLM Reasoning + ResponseGenerationEvidence + Output ValidationAction needed?Answer with CitationsPolicy + Human ApprovalRestricted Tool ExecutorAudit + Result VerificationFeedback + Traces + EvalsNoYes

這是一個結合 RAG + Structured Reasoning Workflow + Controlled Tool Calling 的架構。

不代表所有 Query 都需要 Agent。

簡單問題可以直接 Retrieval + LLM。

只有需要多次查詢、結合即時資訊或執行工具時，才啟動更複雜的 Agent Workflow。

這可以降低 Latency、Cost，也減少不可預測的行為。

## 3.3 Step 3 — Data Collection 與 Knowledge Ingestion

LLM 系統和 CV 系統有一個很大的差別：

CV 系統主要從 Images 學習視覺模式。

Enterprise RAG 系統則需要從大量異質資料中取得正確、即時且有權限存取的資訊。

### Data Sources

|Source|Example|Challenge|
|---|---|---|
|Technical Manuals|Zaber Controller Manual|PDF Parsing、Tables、Versions|
|SOP|Autofocus Recovery Procedure|新舊版本衝突|
|Incident Tickets|Stage Stall Incident|格式混亂、可能包含錯誤結論|
|Logs|Controller Error、Camera Timeout|大量、即時、半結構化|
|Telemetry|Position、Temperature、Voltage|Time Synchronization|
|Configuration|Device Model、Firmware、Axis Mapping|Device-specific Context|
|Engineer Feedback|Recommendation Accepted / Rejected|可能存在 Selection Bias|

### Document Ingestion Pipeline

Raw Documents → Parsing → Normalization → Chunking → Metadata Enrichment → Embeddings → Indexing

例如一份 Zaber Manual 有 300 頁。

其中包含：

- Device Specifications
    
- Error Codes
    
- Configuration Instructions
    
- Troubleshooting Procedures
    
- Safety Warnings
    

如果只是每 500 Tokens 隨意切割，可能會把一個 Error Code 和對應的 Recovery Procedure 切成兩段。

因此我會採用 Structure-aware Chunking：

保留 Heading、Table、Warning、Procedure Step、Device Model 和 Document Version。

### Chunk Metadata

```
{
  "document_id": "MANUAL_1028",
  "chunk_id": "MANUAL_1028_037",
  "device_family": "Zaber",
  "device_model": "X-RST120AK",
  "section": "Fault Diagnostics",
  "document_version": "rev-4",
  "is_current": true,
  "access_group": "field_engineering",
  "content_type": "troubleshooting",
  "effective_date": "2026-04-01"
}
```

這些 Metadata 非常重要。

如果使用者查的是 X-RST120AK，系統不應單純因為語意相似，就將另一個設備型號的 Troubleshooting Procedure 當成正確答案。

另一個重要問題是 Data Freshness。

如果 Manual 有新版本，而 Vector DB 仍保存舊版本，可能導致系統推薦已失效的操作步驟。

因此 Knowledge Ingestion 必須支援 Document Versioning、更新、撤銷、重新索引和可稽核的來源。

## 3.4 Step 4 — Ground Truth 如何建立？

這是 LLM 面試非常重要的部分。

面試官可能問：

> How do you build a ground truth dataset for an open-ended LLM assistant when there is no single correct answer?

假設問題是：

> Why did my Zaber rotation stage stop moving?

Ground Truth 不應只是一段「標準回答」。

因為造成停止的原因可能有機械干涉、負載、運動參數、控制器 Fault 或其他因素。

更適合的 Ground Truth 是一組 Expected Facts + Required Evidence + Expected Behavior + Safety Constraints。

### 一個完整的 Ground Truth Case

```
{
  "case_id": "EVAL_000142",
  "question": "Why did stage_R_X stop at 87 degrees?",
  "device_model": "X-RST120AK",
  "expected_behavior": "diagnose_and_recommend",
  "required_tools": [
    "get_device_status",
    "get_recent_fault_logs"
  ],
  "expected_facts": [
    "The controller reported a stall fault",
    "The exact mechanical cause is not yet confirmed",
    "Further physical and diagnostic checks are required"
  ],
  "required_evidence": [
    "relevant_manual_section",
    "current_device_fault_log"
  ],
  "forbidden_actions": [
    "automatic_unapproved_motion",
    "unbounded_retry",
    "ignore_safety_interlock"
  ],
  "expected_outcome": "safe_diagnostic_recommendation"
}
```

Ground Truth 不是要求 LLM 每個字都和 Reference Answer 一樣。

而是要求它符合核心事實、使用正確工具、遵守安全要求，並且不虛構證據。

### Ground Truth 建立程序

首先，收集歷史 Tickets 和實際故障案例。

接著讓 Domain Experts 標註正確診斷、必要資訊、允許操作、禁止操作。

對於沒有完整確認根因的歷史案例，應標註為 Unresolved，而不是自動把 Ticket 最後一句描述當成事實。

另外需要準備幾種特殊案例：

|Eval Category|測試內容|
|---|---|
|Straightforward|Manual 可以直接回答|
|Multi-document|必須結合 Manual、SOP、Ticket|
|Live Tool|必須讀取設備即時狀態|
|Missing Information|缺少 Firmware 或 Device ID|
|Conflicting Sources|Manual 和舊 Ticket 結論不同|
|Permission Restricted|使用者沒有存取特定客戶資訊的權限|
|Adversarial|文件內含 Prompt Injection|
|Unsafe Request|要求跳過 Interlock 或直接啟動危險動作|
|Unknown Problem|現有 Knowledge Base 無法確認|

這些案例必須能形成 Regression Suite。

當模型、Prompt 或 Retrieval Strategy 改變時，重新跑同一組測試，才能知道是否發生 Regression。

## 3.5 Step 5 — Model Selection

面試官可能會問：

> Would you fine-tune an LLM, build a RAG system, or use a multi-agent architecture?

我不會立即選擇 Fine-tuning，也不會先建立複雜 Multi-agent System。

我會做以下比較。

|Approach|適合|Trade-off|
|---|---|---|
|Prompt + Base LLM|先建立初始 Baseline|缺少企業內部最新知識|
|RAG|Document-grounded QA|Retrieval Quality 會限制最終結果|
|RAG + Reranker|大型 Knowledge Base、相似文件|額外 Latency 和成本|
|Tool Calling|查即時 Logs、Telemetry|權限、API Reliability、Tool Errors|
|Fine-tuning|特定輸出格式、穩定的領域行為|資料品質、Training Cost、維護成本|
|Agent Workflow|多步驟診斷與動態規劃|較高 Latency、不確定性與測試難度|

### 我的選擇

第一版使用：

Hybrid Retrieval（BM25 + Dense Retrieval）→ Reranking → LLM → Evidence Validation

對於需要設備診斷的問題，再加入 Tool Calling。

採用 Hybrid Retrieval 的原因是：

BM25 對精確 Part Number、Error Code、Firmware ID 通常很有價值。

Dense Retrieval 則能找到語意相近、但用詞不同的 Troubleshooting Documents。

例如：

`MovementFailedException: Stalled and Stopped (FS)`

與：

`rotation stage unexpectedly stops under load`

兩者的文字不同，但可能有相關的診斷內容。

Hybrid Retrieval 可以結合兩者優勢，再用 Reranker 找出最有用的片段。

不過最終選擇仍必須根據實際 Retrieval Evaluation，而不是假設 Hybrid 一定勝出。

### 如何選 LLM？

我會建立 Model Benchmark：

|指標|Small / Fast Model|Large / Reasoning Model|
|---|---|---|
|Latency|通常較低|通常較高|
|Cost per Request|通常較低|通常較高|
|Complex Diagnosis|需驗證是否足夠|可能有優勢|
|Tool Use|需要實測|需要實測|
|Context Handling|與具體模型能力相關|與具體模型能力相關|
|Deployment|API 或 Self-host|API 或 Self-host|

例如簡單 Q&A 可以交給 Fast Model，涉及複雜多來源診斷時再升級到較強的模型。

但是 Routing 本身也要納入 Evaluation，否則會發生：

本來應該交給 Reasoning Model 的複雜問題，被錯誤分配給較弱的模型。

## 3.6 Step 6 — 完整具體 Inference Example

現在用實際設備故障問題示範整條 Pipeline。

使用者輸入：

> My X-RST120AK rotary stage is stuck at 87 degrees. I received "Stalled and Stopped (FS)". What should I do?

1. User Intent Understanding
    
    Query Router 識別這是 Equipment Troubleshooting，需要讀取相關 Fault Logs，而不只是一般文件問答。
    
2. Retrieve Relevant Knowledge
    
    搜尋相同 Device Model 的 Manual、Fault Code 文件與適用的 SOP，排除不相容或過期的程序。
    
3. Tool Calling
    
    呼叫 `get_device_status`、`get_recent_fault_logs`。
    
    取得目前位置、Fault Status、最近的 Motion Commands 與 Controller State。
    
4. Reasoning and Evidence Synthesis
    
    整合 Manual 和即時資料，區分已證實的事實與需要進一步排查的可能原因。
    
5. Policy / Safety Validation
    
    檢查建議是否涉及未授權移動、清除故障狀態或跳過安全程序。
    
6. Response Generation
    
    給出可核對的診斷建議、文件來源、未確認事項，以及何時需要現場人員介入。
    

理想的 LLM Response 不是：

「這是馬達問題，請重新啟動並再次旋轉。」

而是：

> The controller reported a stall fault while executing the rotation command. The current evidence confirms a motion failure but does not establish its mechanical cause. Check the approved diagnostic procedure, inspect for potential mechanical interference under safe conditions, and review the controller's fault and motion records. Do not issue further motion commands until the fault condition and safety state have been assessed.

並附上真正檢索到的來源。

這裡的核心是 Evidence-grounded Reasoning，而不是生成一個聽起來合理的答案。

## 3.7 Step 7 — LLM Evaluation：不是只看 Answer Accuracy

LLM 系統的 Evaluation 應分為四個層次。

### Layer A — Retrieval Evaluation

測試是否真的找到正確資料。

例如：

\[ Recall@K = \frac{\text{Retrieved Relevant Documents}} {\text{Total Relevant Documents}} \]

使用 Recall@5、Recall@10、MRR、nDCG 等指標，並以 Ground Truth 的 Relevant Document / Chunk 標註為基準。

如果 Retrieval 沒有找出關鍵 Safety Procedure，即使 LLM 本身很強，也無法保證最終回答可靠。

### Layer B — Answer Quality

|Metric|說明|
|---|---|
|Factual Correctness|核心事實是否正確|
|Groundedness|答案是否受到來源支持|
|Citation Accuracy|引用是否真正支持相關敘述|
|Completeness|是否包含所有必要診斷步驟|
|Abstention Quality|無足夠證據時是否合理拒答|
|Safety Compliance|是否遵守安全限制|

LLM-as-a-Judge 可以協助大規模評估，但不能把它視為完全可靠的 Ground Truth。

高風險案例仍應使用 Domain Experts 進行 Blind Review，並定期測試 Judge 和專家的一致性。

### Layer C — Agent / Tool Evaluation

這是 Agentic AI 和一般 RAG 系統非常不同的地方。

除了 Final Answer，還要評估整條 Execution Trace：

- 是否選擇正確工具？
    
- 是否傳入正確 Device ID？
    
- 是否執行多餘或危險的 Tool Call？
    
- 是否超過規定的 Tool Iteration Budget？
    
- 是否在必要時要求人工核准？
    
- 是否在 Tool Failure 後正確停止或降級？
    

OpenAI 的 Agent Evaluation 指引也將 Trace、Tool Calls、Guardrails 和 Handoffs 納入 Workflow-level Evaluation，而不只是評估最後輸出的文字。

![](https://www.google.com/s2/favicons?domain=https://developers.openai.com&sz=32)

OpenAI API

+1

### Layer D — Business / Production Evaluation

最終要看：

|Business Metric|要回答的問題|
|---|---|
|Time to Resolution|工程師是否更快解決故障？|
|First-time Resolution Rate|是否減少重複診斷？|
|Escalation Rate|是否減少不必要的人工升級？|
|Incorrect Recommendation Rate|是否減少錯誤建議？|
|Engineer Adoption|是否真的有人持續使用？|
|Cost per Resolved Issue|是否比人工或舊流程更具成本效益？|

注意：Engineer Adoption 高，不代表答案一定正確。

同樣地，Escalation Rate 下降，也不代表系統更安全，因為它可能只是過度自信。

因此 Business Metrics 必須和 Safety / Quality Guardrails 一起觀察。

### Offline 與 Online Experiment

我會先使用 Frozen Evaluation Dataset 進行：

Baseline RAG vs. Hybrid RAG vs. RAG + Tools

每個 Variant 使用相同的 Query、Document Snapshot、可重現的 Tool Fixture，並記錄 Prompt、Model、Retriever、Index Version。

再進行 Shadow Testing。

通過 Safety Gate 後，才安排少量工程師參與 A/B Test。

Randomization 需要考慮工程師群組和設備案例複雜度，避免單純因為某組遇到較容易的案件，就誤判新系統比較好。

## 3.8 Step 8 — Production Bottleneck and Failure Modes

假設面試官問：

> The LLM application has a P95 latency of 25 seconds, but the product requirement is 8 seconds. How would you reduce latency?

我會先把 End-to-End Latency 拆開：

\[ T_{\text{total}} = T_{\text{auth}} +T_{\text{retrieval}} +T_{\text{tools}} +T_{\text{prefill}} +T_{\text{generation}} +T_{\text{validation}} \]

這也是串行流程的簡化形式。如果使用並行 Retrieval 或 Tool Calls，必須用 Trace 找出 Critical Path。

以下是假設的量測結果。

|Stage|P95 Latency|Potential Optimization|
|---|---|---|
|Authentication / Routing|0.5 s|Reduce unnecessary network hops|
|Retrieval + Reranking|2.5 s|Index、Cache、Reranker Optimization|
|Tool Calls|7 s|Parallelize independent read-only calls|
|LLM Prefill|4 s|Reduce irrelevant Context|
|LLM Generation|9 s|Model Routing、Constrain Output Length|
|Validation / Formatting|2 s|Optimize validators|

以上 Stage P95 不應直接相加視為 End-to-End P95，因為各階段延遲分布及相依性不同。

### 具體優化方法

Optimization 1 — Eliminate unnecessary Agent steps

如果只需要查 Manual，就不必呼叫三個 Tool。

Optimization 2 — Context Reduction

不要將 30 個 Retrieval Chunks 全丟給 LLM。

透過 Reranking、Metadata Filtering 和 Context Compression，保留真正重要的內容。

Optimization 3 — Parallel Tool Calls

兩個彼此獨立的 Read-only Tool Calls 可以平行執行。

但有狀態依賴的操作，尤其是硬體操作，不應任意平行化。

Optimization 4 — Model Routing

簡單問題使用較小模型；困難問題才使用較強模型。

Optimization 5 — Cache

可以快取版本固定、權限一致的文件 Retrieval 結果。

但不能把某個客戶的即時設備資料跨使用者共用，也不能因為快取而忽略撤銷的權限。

### Cost Optimization

假設每天 2,000 次 Request，平均每次：

Input 3,000 Tokens，Output 600 Tokens。

則：

\[ \text{Daily Input}=6,000,000\text{ tokens} \]

\[ \text{Daily Output}=1,200,000\text{ tokens} \]

每月 30 天：

\[ 180M\text{ input tokens}+36M\text{ output tokens} \]

因此模型月成本可以估算為：

\[ C_{\text{LLM}} = 180P_{\text{in}}+36P_{\text{out}} \]

其中 \(P_{\text{in}}\)、\(P_{\text{out}}\) 是每百萬 Tokens 的實際計費價格。

再加上 Retrieval、Reranking、Storage、Logging、Tool Infrastructure 及可能的 Cache 成本。

Staff-level 思考不應只問如何降低 Token Cost，而要問 Cost per Successfully Resolved Issue。

如果便宜模型讓工程師多花 20 分鐘處理問題，可能反而增加整體成本。

### Failure Modes

|Failure Mode|對 Production 的影響|Mitigation|
|---|---|---|
|Hallucination|錯誤診斷|Grounding、Evidence Validation、Abstention|
|Wrong Retrieval|引用錯誤文件|Metadata Filter、Hybrid Search、Reranking|
|Stale Knowledge|使用過期 SOP|Document Version、Effective Date|
|Prompt Injection|文件誘使 Agent 執行未授權操作|Untrusted-content Isolation、Policy Enforcement|
|Unauthorized Access|洩漏跨客戶資料|Query-time ACL、Backend Authorization|
|Tool Timeout|診斷流程卡住|Timeout、Circuit Breaker、Safe Fallback|
|Infinite Agent Loop|Token Cost 和 Latency 激增|Max Steps、Token Budget、Deadline|
|Model Degradation|新模型表現變差|Regression Evals、Canary、Rollback|
|Unsafe Tool Action|硬體或作業風險|Human Approval、Backend Safety Interlock|
|Provider Outage|問答不可用|Graceful Degradation、Approved Fallback|

尤其要注意：

LLM 不是 Security Boundary。

即使 System Prompt 說「不要操作硬體」，也不能單靠 Prompt 保證安全。

真正的權限必須在 Backend Tool Executor 再次檢查。

LLM 提出的 Tool Call 應被視為一項請求，由受控服務驗證權限、參數、設備狀態與必要的人工核准後才可能執行。

這也是 Staff Engineer 很適合強調的系統設計能力。

# Part 4. Technical Leadership — Senior / Staff 面試如何回答？

前面的兩個案例偏向 Architecture。

接下來更重要的是：

> How do you lead cross-functional teams to deliver a complex AI system?

Technical Leadership 不是只表示「我負責帶三個工程師」。

面試官希望看到的是：

你如何在沒有直接管理權限的情況下，讓多個團隊對技術方向、成功標準、時程與風險達成一致，並持續交付。

## 4.1 案例一：Computer Vision 跨 Hardware、Software、Research 團隊合作

### 面試題目

> Tell me about a time when you led a cross-functional project involving hardware, software, and machine learning teams.

以下是以手錶檢測系統設計的假設情境，可以用來練習 STAR 答題方法。

### Situation

公司正在開發自動化光學檢測系統。

系統包含三個 Camera、Motion Stages、Autofocus、Image Processing 和 AI Classification。

Research Team 在實驗室得到很好的 Accuracy，但當系統部署到不同機器上時，Performance 明顯下降。

Hardware Team 認為是模型 Robustness 不足。

Research Team 認為是攝影品質不一致。

Software Team 則發現 Capture Pipeline 不穩定。

三個團隊開始互相指責。

### Task

身為 Senior / Staff Engineer，你必須讓問題可被量測，確定 Root Cause，並建立一個可以長期維護的協作流程。

### Action 1 — 建立共同的 Success Criteria

先安排 Hardware、Software、Research、QA 共同定義一份 Image Acquisition Contract。

例如：

|Ownership|Contract / Responsibility|
|---|---|
|Hardware|Camera Stability、Lighting Repeatability、Motion Accuracy|
|Imaging Software|Capture Configuration、Autofocus、Image Quality|
|ML Research|Model Quality、Robustness、Calibration|
|Backend|Data Storage、Versioning、Deployment|
|QA|Integration Tests、Acceptance Criteria|

這個 Contract 必須明確定義影像品質的合格標準。

例如透過固定 Reference Target，測量：

Focus Quality、MTF、SNR、Saturation、Geometric Registration Error、Color Calibration Error。

不要只讓 Hardware Engineer 說「我覺得這張影像很清楚」。

也不能只讓 ML Engineer 說「模型 Accuracy 不好，所以 Camera 有問題」。

需要把爭論變成可測試的 Hypothesis。

### Action 2 — 設計 Root Cause Experiment

假設有三台設備：Machine A、Machine B、Machine C。

使用相同 Reference Watch、相同 Model、相同 Software Version，進行 Controlled Experiment。

|Experiment|Variable|
|---|---|
|Test A|Same Model, Same Camera, Different Lighting|
|Test B|Same Model, Different Camera|
|Test C|Same Camera, Different Autofocus Settings|
|Test D|Same Raw Image, Different Preprocessing|
|Test E|Same Image, Different Model Versions|

這樣可以分離 Hardware Variability、Image Processing Variability、Model Variability。

如果同一張 Raw Image 在不同 Software Version 得到不同 Feature，問題可能在 Preprocessing。

如果相同 Model 在不同設備的影像上表現不同，則需要檢查 Hardware Configuration、Calibration 和 Domain Shift。

### Action 3 — 建立真正的 Cross-team Ownership

我會制定一份簡短的 RFC（Request for Comments）：

RFC: Imaging Quality and AI Inference Contract

包含：

- Hardware / Software API Contract
    
- Image Quality Acceptance Criteria
    
- Error Handling
    
- Dataset Versioning
    
- Calibration Procedure
    
- Production Monitoring
    
- Release Gate
    
- Team Ownership
    

每項 Deliverable 都有一個 DRI（Directly Responsible Individual），而不是五個人共同負責、最後沒有人真正負責。

### Action 4 — 建立 Release Gate

新 Camera Firmware 不能未經測試就進 Production。

新的 Preprocessing Pipeline 不能只因為影像肉眼看起來更漂亮就通過。

新的 Model 也不能只因為 Overall Accuracy 提高就直接部署。

每個改動需要通過相應的 Contract Tests、Regression Tests 和 Acceptance Tests。

### Result

如果這是面試中的真實經驗，應該提供可驗證的實際成果，例如：

|Indicator|改善前|改善後（假設示例）|
|---|---|---|
|Cross-machine Performance Gap|12 percentage points|3 percentage points|
|Re-capture Rate|14%|5%|
|Average Scan Time|9.5 min|7 min|
|Imaging-related Incidents / month|10|3|

這些數字只是示範怎麼呈現結果，面試時必須替換成你真正完成的成果。

### Senior vs Staff 差異

Senior Engineer 的成果可能是：

「我跨三個團隊找到 Root Cause，將 Capture Failure Rate 降低，並讓模型能穩定部署。」

Staff Engineer 應該進一步說：

「我建立統一的 Imaging Quality Contract、Release Gates、Dataset Standards 和 Ownership Model，使後續新增 Camera 或新機型時，都能重用相同的驗證流程。」

Senior 解決一次困難的問題；Staff 建立讓組織未來更容易解決同類問題的能力。

## 4.2 案例二：LLM 團隊的技術方向衝突

### 面試題目

> How would you handle disagreement between a research team pushing for a sophisticated multi-agent system and a product team that needs a reliable launch?

假設 Research Team 想用 Multi-Agent Architecture。

其中包含：

Planner Agent → Retrieval Agent → Diagnostic Agent → Verification Agent

Product Team 希望六週內推出第一版。

Infrastructure Team 則擔心成本、Latency 和大量 Tool Calls。

這時候 Staff Engineer 不應直接選邊站。

### Step A — 重新對齊 Business Objective

先明確定義第一版要解決什麼：

「Field Engineers 能否更快找到相關 SOP，並根據可信證據獲得安全的診斷建議？」

如果主要需求只是 Technical Q&A，沒有充分理由直接使用複雜 Multi-Agent。

### Step B — 設計公平的 Technical Evaluation

比較三個 Architecture：

|Design|Description|
|---|---|
|A|Simple RAG|
|B|Hybrid RAG + Read-only Tools|
|C|Multi-Agent Diagnostic System|

統一測試：

Task Success Rate、Safety Violations、P95 Latency、Cost、Debugging Complexity、Maintenance Effort。

### Step C — 做 Architecture Decision

假設 A/B/C 測出以下結果：

|Metric|A: RAG|B: RAG + Tools|C: Multi-Agent|
|---|---|---|---|
|Task Success|76%|90%|92%|
|P95 Latency|5 s|8 s|23 s|
|Relative Cost|1.0×|1.6×|4.2×|
|Debugging Complexity|Low|Medium|High|

假設實驗結果，僅供示範架構取捨。

我可能會建議第一版選 B。

因為 C 的 Task Success 僅多 2 個百分點，但 Latency 和 Cost 顯著增加。

如果 C 在特定高複雜度診斷案件能帶來明顯價值，則可以保留為後續選項。

### Step D — 讓不同團隊都有合理的 Roadmap

Research Team 可以繼續在 Offline Benchmark 改善複雜案例。

Product Team 先交付穩定的 RAG + Tools。

Infrastructure Team 建立共用 Trace、Evaluation、Tool Authorization 和 Model Registry。

這樣不是否定 Multi-Agent，而是讓架構複雜度由測試結果和 Business Value 來決定。

Staff Engineer 的領導力，常常展現在能把「技術立場的爭論」轉成「共同接受的決策標準」。

# Part 5. Mentoring Engineers 與交付可維護系統

Technical Leadership 面試通常還會深入問：

> How do you mentor junior engineers while maintaining engineering quality and delivery speed?

或：

> How do you ensure that the system remains maintainable after your team finishes the initial implementation?

我的回答會把 Mentoring 和 Architecture Ownership 結合，而不是單純說「我每週與 Junior Engineer 開會」。

## 5.1 如何分配任務？

例如 CV Pipeline 有四個工程師。

|Engineer|Ownership|Development Goal|
|---|---|---|
|Junior A|Image Quality Module|學會 Unit Test、Validation、Metrics|
|Mid B|Segmentation / Feature Pipeline|學會 Model Benchmark、Performance Optimization|
|Senior C|Data / Ground Truth Pipeline|Own Data Contracts、Dataset Versioning|
|Staff / Tech Lead|End-to-end Architecture|Cross-team Trade-offs、Release Criteria|

Junior Engineer 不應只是負責寫一個沒有上下文的小 Function。

應該讓他清楚知道：

這個 Image Quality Module 的 Output Contract 是什麼？

怎樣叫 Success？

失敗時系統應如何處理？

什麼測試必須通過才能合併？

### Mentoring 的具體做法

我會讓 Junior A 先提出自己的 Design Proposal。

例如他想用 Laplacian Variance 做 Blur Detection。

我不會直接否定或直接替他寫完。

而是請他思考：

這個方法是否對不同 Camera / Exposure 都有效？

會不會把低紋理但清晰的區域判定為 Blur？

如何建立 Focus Ground Truth？

Threshold 如何設定？

Production 有沒有 Monitoring？

然後讓他用資料驗證這些問題。

這種做法比直接告訴他演算法答案更能培養 System-level Thinking。

## 5.2 如何建立 Maintainable Architecture？

我會定義一些工程原則。

Production AI Engineering Standards

Modular Design

Capture、Feature、Inference、Policy 有清楚接口

Schema Contracts

輸入輸出、型別、版本與錯誤語意固定

Test Pyramid

Unit、Integration、Hardware-in-loop、E2E

Observability

Metrics、Traces、Logs、Data Drift

Reproducibility

Dataset、Code、Model、Configuration 版本

Safe Rollback

Canary、Health Check、Rollback Playbook

此外，Release Process 應該有明確 Code Review、CI/CD、Ownership 和 Runbook。

Google 的 SRE 方法將 Service-Level Objectives 和 Error Budgets 用來平衡開發速度與系統可靠性。這個方法也適用於 Production AI：當新版本持續破壞 Reliability Target 時，需要暫停高風險發布並優先改善穩定性。

![](https://www.google.com/s2/favicons?domain=https://sre.google&sz=32)

IT Service Management: Automate Operations

+1

### Staff Engineer 還需要考慮 Bus Factor

假設所有 Autofocus 邏輯都只有一名資深工程師理解。

即使功能完全正常，這仍然是重大 Engineering Risk。

我會安排：

Design Documentation、Pair Debugging、Runbook、Architecture Review、Ownership Rotation，以及由其他工程師操作的 Recovery Drill。

目標不是讓每個人都懂所有細節，而是讓關鍵功能具備文件、測試與替代維護能力。

# Part 6. 面試官最可能進一步追問的題目

以下是 Senior / Staff AI Engineer 很值得模擬練習的 Follow-up Questions。

|Interview Question|Senior / Staff 回答重點|
|---|---|
|What if you have only 100 labeled examples?|Transfer Learning、Active Learning、Self-supervised Features、Uncertainty、合理的測試界限|
|How do you ensure Ground Truth quality?|Independent Experts、Adjudication、Provenance、Label Versioning|
|How do you handle class imbalance?|Cost-sensitive Learning、PR Curves、Thresholds、Calibrated Review、代表性測試|
|How do you know the model will generalize?|Group Holdout、Temporal Holdout、Domain Shift、OOD|
|Why not use a larger model?|Quality Gain vs. Latency、Cost、Complexity、Maintainability|
|How do you diagnose production degradation?|End-to-end Traces、Data Drift、Feature Skew、Hardware Changes|
|How do you deploy a new model safely?|Shadow、Canary、SLO Gates、Rollback|
|How do you prevent LLM hallucinations?|Retrieval Quality、Evidence Validation、Refusal、Human Review|
|How do you evaluate an Agent?|Tool-use Correctness、Trace Grading、Task Success、Safety Tests|
|How do you prioritize technical debt?|Reliability / Delivery Impact、Risk、Maintenance Cost|
|How do you resolve disagreements?|Measurable Criteria、RFC、Experiment、Decision Record|
|How do you mentor engineers?|Delegated Ownership、Design Review、Feedback、Measurable Growth|

這些問題的共同點是：

面試官不只是想聽到「哪個技術比較好」，而是想知道你如何用證據做出工程決策。

# Part 7. 45 分鐘 System Design Interview 應如何安排時間？

不論題目是 Computer Vision 還是 LLM，我建議使用相同的框架。

建議時間分配

45-minute interview

Requirements + Success Metrics

7 min

High-level Architecture

8 min

Data + Ground Truth + Models

10 min

Experiments + Evaluation

7 min

Production + Failure Modes

8 min

Trade-offs + Leadership

5 min

真正面試時，不需要一開始就把所有細節說完。

應先畫 High-level Architecture，再讓面試官選擇他希望深入的部分。

例如：

> I can go deeper into our data labeling strategy, model selection, evaluation framework, or production reliability. Which area would you like me to focus on?

這能展現你有完整的 System-level Thinking，也尊重面試時間。

# Part 8. 如何回答才能真正聽起來像 Staff Engineer？

同一個問題，三種層級的回答會有很大的差異。

### 問題 A：How do you choose a model?

Mid-level

"I would use a Vision Transformer because it has strong image classification performance."

Senior

"I would establish baselines, compare CNN, ViT and feature-based methods using watch-level holdout datasets, and evaluate performance, latency, calibration and deployment constraints."

Staff

"Before selecting a model, I would align the team on acceptable false-acceptance risk and review cost, define the evidence and ground-truth strategy, and establish a reproducible evaluation framework. Model complexity should be justified by measurable business value and operational cost."

### 問題 B：How do you resolve production failures?

Mid-level

"I would inspect the logs and fix the bug."

Senior

"I would use traces, latency breakdowns, error categories and reproducible test cases to isolate the bottleneck, fix the root cause, and add regression tests."

Staff

"I would contain customer impact first, establish ownership across the affected teams, lead the root-cause analysis, and ensure we improve our monitoring, contracts and release process so the same failure class is less likely to recur."

### 問題 C：How do you lead engineers?

Mid-level

"I help junior engineers when they have questions."

Senior

"I delegate meaningful technical ownership, review designs, provide actionable feedback and help engineers become independent owners of their components."

Staff

"I develop technical leaders across teams, create clear decision-making and ownership mechanisms, and establish reusable engineering standards that improve delivery without requiring my involvement in every decision."

# Part 9. 最後建議：如何把這兩個案例準備成真正的面試能力？

對 Senior / Staff AI Engineer 而言，我會把準備重點分成以下幾個面向：

|能力|Computer Vision 案例|LLM 案例|
|---|---|---|
|Data Engineering|Image Capture、Metadata、Dataset Versioning|Documents、Logs、Chunking、Indexing|
|Ground Truth|Expert Labels、Provenance、Adjudication|Expert Rubrics、Expected Facts、Tool Outcomes|
|Model Selection|CNN、ViT、Statistics、Anomaly Detection|RAG、Fine-tuning、Reasoning、Agent|
|Evaluation|Group Split、OOD、Calibration|Retrieval、Answer、Trace、Human Evaluation|
|Architecture|Edge Inference、Cloud Training|Retrieval、Orchestration、Model Serving|
|Scalability|Image IO、GPU、Motion Throughput|Tokens、Concurrency、Tool Latency|
|Reliability|Hardware Failure、Missing Images、Drift|Hallucination、Tool Failure、Prompt Injection|
|Technical Leadership|Hardware / Imaging / ML Alignment|Research / Product / Infrastructure Alignment|
|Staff-level Impact|Standardized Imaging / ML Platform|Shared RAG / Agent Evaluation Platform|

## 面試準備的最高優先順序

我會建議特別精通三個主題。

第一是 Ground Truth + Evaluation Design。 這往往比熟記某種 Neural Network Architecture 更能區分 Senior 與 Staff。你需要理解 Label Noise、Data Leakage、Distribution Shift、Confidence Intervals、Calibration，以及 Offline-to-Online Gap。

第二是 Production Failure Analysis。 必須能像真正的 System Owner 一樣，從 Metrics、Logs、Tracing、Dependencies、Concurrency、Hardware、Cloud 和 Model Behavior 中，系統性找出 Root Cause，而不是只提出重新訓練模型。

第三是 Technical Leadership with Measurable Impact。 必須能說清楚自己如何制定方向、說服其他團隊、處理不同意見、提升 Junior Engineers 的能力，以及建立可在自己離開後繼續運作的系統。

對你目前的技術背景而言，多相機 + Motion + Autofocus + Computer Vision + AI + AWS 的整合案例，是非常適合拿來準備 Senior / Staff Computer Vision Engineer 面試的素材。它比單純說明 YOLO、UNet、Transformer 訓練經驗更有機會展示系統級工程能力。

不過在實際面試中，要明確區分「你親自設計並交付的部分」、「團隊共同完成的部分」和「下一階段提案」，並以能夠佐證的 Metrics 說明成果。

最後記住一個最重要的區別：

Senior Engineer 的核心價值：能把一個困難的 AI System 正確設計、實作、驗證並穩定上線。

Staff Engineer 的核心價值：能在技術與組織的不確定性中做出高品質決策，讓多個團隊持續且可靠地交付更複雜的 AI Systems。

這兩者都需要很強的技術深度；Staff 的差異不只是管理更多工程師，而是能創造更廣泛、持續且可衡量的技術影響力。