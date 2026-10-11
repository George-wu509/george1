

|                                               |     |
| --------------------------------------------- | --- |
| [[#### Multimodal AI／VLM在2025,2026以後新興方向或技術]] |     |
|                                               |     |
|                                               |     |
|                                               |     |

#### Multimodal AI／VLM在2025,2026以後新興方向或技術
```
請深入詳細解釋Multimodal AI／Vision-Language Model（VLM）在2025,2026以後新興方向或技術, 首先以解釋給intern方式講解, 再以講解給senior AI engineer的專業技術講解細節, 並以一個非常具體例子講解
```

# Multimodal AI / Vision-Language Model（VLM）：2025–2026 新技術、未來方向與完整工程實例

技術範圍：2025–2026 已發布研究，以及 2027 年以後值得關注的發展方向

2025–2026 年的 Multimodal AI，最重要的改變，是 AI 從「看懂圖片並回答問題」，逐漸發展成「理解圖片與影片、進行多步推理、使用工具、理解三維空間，甚至規劃真實世界中的動作」。

這個變化對 Computer Vision Engineer、Senior AI Engineer、Robotics Engineer，以及正在開發工業影像分析系統的工程師非常重要。

我會分成三個主要部分：

1. Part I — Intern Level： 用簡單例子理解 Multimodal AI、VLM，以及 2025–2026 年的新興技術。
    
2. Part II — Senior AI Engineer Level： 深入模型架構、Transformer、Vision Encoder、Multimodal Fusion、Training、Post-training、Reasoning、Inference、Agent、Evaluation 和系統部署。
    
3. Part III — 完整實際案例： 設計一套使用多相機、高解析度影像、VLM、Computer Vision 與 Bayesian Reasoning 的 Rolex 手錶零件識別、異常偵測及真偽分析系統。
    

# Part I — 解釋給 AI Intern

## 1. Multimodal AI 和 VLM 到底是什麼？

### 1.1 傳統 AI：一種模型通常專注一種資料

假設有三個不同的 AI 模型：

![Cartier, de 4 iconische modellen](https://images.openai.com/static-rsc-4/nKfar4Q_PRZFUuw__Bp1UacRUeODABT5RwHwTp730b9-Jmjg645QWXVUagSHreVOsxFJsT14Lr-o-5MTxnM4mj0aJlVykQnT1y6Hd9ZQsHJa2xfZ1ht-QowKW-IdzVeeJqSOffUeYH6ux4woBVFtB9gnqvDw1vY8NvH5PUP6dRc?purpose=inline)

Computer Vision

輸入：Image

輸出：物件類別、座標、Segmentation Mask、缺陷位置

![Manuals - Online Printing - Crush Printing](https://images.openai.com/static-rsc-4/pbYkM0Lkq6MjfMQr6oEvFzt0Bu9zUKF0bh7YkV287eTo5wr2krdw3hIB5C04onPLKBlk5KbC8deRcax-XbPDSOTci0OROoXd9wRqH5yok4buESSuqHHPqgZNkEH9ZE9joHhExpzSkyXMoONDnAvK_k4Xr_1wu5qkOvZo2RhbJs0?purpose=inline)

LLM

輸入：Text

輸出：文字解釋、問答、程式碼

![Industrial Robotics Vision Systems: Sub-Millimeter Precision - AI CERTs News](https://images.openai.com/static-rsc-4/7OeOHL_L7T4kFztjl3OIVU82px2IJFT1Z0LzOmajqasIofnXpkxFjQ440WBBgUbUaIV-ejFAkh3Ixm85l4f90BHH_mK307q9Tch022BCTSMiv4F-a9HSula0R7mPOWb34jl3dBhqt_ddgQqP3lbSg3ydzqtUySJhg1BBe_2ZdvQ?purpose=inline)

Robotics AI

輸入：感測器、狀態

輸出：機械手臂的控制動作

以前的系統通常要讓這些模型分別工作。

例如，傳統 Computer Vision 模型可以告訴你：

「這張照片裡有一支手錶，辨識到 12 個 hour markers。」

但假設你問：

「這支 Rolex 手錶的錶面文字、指針和刻度，與 2005 年的同系列原廠規格有什麼不同？」

這個問題除了辨識圖片，還需要：

- 理解手錶零件之間的關係。
    
- 閱讀錶面文字。
    
- 查詢不同年份的原廠規格。
    
- 比較參考影像。
    
- 分析不同證據。
    
- 最後用語言解釋異常。
    

這就是 VLM 能發揮價值的地方。

### 1.2 VLM：把電腦視覺與語言理解結合

VLM（Vision-Language Model）是可以共同處理視覺資訊與語言資訊的 AI 模型。

它可以接收：

- 一張圖片 + 一個問題。
    
- 多張圖片 + 一段技術文件。
    
- 一段影片 + 一個分析指令。
    
- 一張工程圖 + 一段產品規格。
    
- 一組相機影像 + 要求輸出的結構化資料。
    

例如：

輸入影像

![Rolex Submariner 41 Steel Yellow Gold Blue Dial Mens Watch 126613 Box Card at 1stDibs](https://images.openai.com/static-rsc-4/Wn_RQLqe1wXx1BtVHgVt-p3vwn2AJxi9-efR1UA5NIcde8SBJo7Uw0GgHvGipnCRlH5J7X17ycPNSamF16VZBRtNErJX2ILqL7DmD48f6-fJSsGPh8ONUi1DO9B-KsqqC1jausv3xvuNoUFixOCYj5Bs-l-KRXJqH7Pw8m8DaHY?purpose=inline)

[1stdibs.com](https://www.1stdibs.com/jewelry/watches/wrist-watches/rolex-submariner-41-steel-yellow-gold-blue-dial-mens-watch-126613-box-card/id-j_21893672/)

使用者指令

比較錶面 12 點鐘位置的 hour marker、ROLEX 字體與指針形狀，描述可能不一致的特徵。

VLM 預期能提供的分析

辨識零件、描述文字、定位疑似異常區域，以及建議需要比較的參考影像。是否真正異常，仍需依影像與可靠的原廠資料確認。

要注意：VLM 能描述可疑特徵，不代表它能直接可靠地證明手錶是真品或仿品。

### 1.3 Multimodal AI 比 VLM 範圍更廣

## Multimodal AI

能整合兩種或以上資料模態的 AI

VLM

Image / Video + Language

Audio-Language

Audio + Language

Vision-Language-Action

Vision + Text + Action

Unified Multimodal Model

Image + Video + Audio + Text

更精確地說，VLM 也不一定是會聊天的模型。像 CLIP、SigLIP 2 這種將影像與文字對齊、用來搜尋或分類的模型，也屬於廣義 VLM。

現在常見的會看圖、能夠用自然語言回答問題的模型，通常稱為 Multimodal Large Language Model（MLLM），也有人稱 Large Vision-Language Model（LVLM）。

## 2. VLM 是怎麼看懂圖片的？

想像 intern 拿了一張 2048 × 2048 的手錶照片。

一個常見的 VLM 並不是直接把所有 RGB pixel 當成文字讀取，而是透過以下架構：

Image

手錶照片

Vision Encoder（ViT）

把圖片轉成視覺特徵向量

Projector / Connector

把視覺特徵轉成 LLM 可使用的表示

Vision Tokens + Text Tokens

整合圖片資訊與使用者問題

LLM / Reasoning Model

理解、比較與生成回答

Output

自然語言、JSON、區域座標或工具呼叫

這是典型的 Vision Encoder + Projector + LLM 架構，並不是所有多模態模型都完全相同。

先理解五個重要元件：

|元件|Intern 需要理解的意思|
|---|---|
|Vision Encoder|AI 的視覺特徵提取器，負責理解影像內容|
|Visual Tokens|把影像資訊表示成模型可處理的向量序列|
|Projector|將 Vision Encoder 的輸出連接到語言模型|
|Transformer / Attention|協助模型整合視覺與文字之間的關係|
|LLM Decoder|根據已理解的資訊生成文字或其他輸出|

一個很重要的概念：

Vision Encoder 負責擷取視覺資訊，LLM 負責利用這些資訊進行語言理解和推理；模型最終能做多精細的判斷，取決於兩者之間保留多少影像資訊。

所以 VLM 並不一定比專門的 Computer Vision 模型更擅長量測微小距離、識別細微刮痕或 Pixel-level Segmentation。

## 3. 2025–2026 年最值得關注的 10 大技術方向

這些方向互相影響，並不是彼此獨立的模型種類。

|新興技術|Intern 理解方式|主要價值|
|---|---|---|
|1. Multimodal Reasoning|不只看見，還能逐步分析|複雜圖片推理|
|2. High-Resolution Vision|能針對非常細小的區域放大分析|OCR、精密檢查|
|3. Visual Grounding|知道答案對應圖片的哪一個位置|Bounding Box、Mask|
|4. Video Understanding|能理解事件如何隨時間變化|影片、製程監控|
|5. Agentic VLM|會呼叫其他工具完成任務|自動化工作流程|
|6. Spatial / 3D Reasoning|理解物體位置、方向、深度|3D 與機器人|
|7. VLA / Embodied AI|把理解變成實際動作|Robotics|
|8. Multimodal RAG|能查圖片、圖表及技術文件|專業知識系統|
|9. Efficient / Edge VLM|降低模型記憶體與計算成本|本機 GPU、工廠設備|
|10. Verifiable / Reliable VLM|驗證 AI 的判斷是否有證據|工業、醫療、金融|

以下用幾個具體例子解釋重要變化。

### 方向一：Multimodal Reasoning — 從辨識到推理

過去 VLM 通常回答：

「圖片裡有三個齒輪。」

新一代 VLM 可以嘗試回答：

「從影像判斷，這三個齒輪的嚙合方向是否正確？如果第一個順時針旋轉，第三個預期往哪個方向旋轉？」

這需要區分三件事：

1. Perception：辨識三個齒輪。
    
2. Reasoning：理解它們之間的機械關係。
    
3. Verification：確認結論有沒有符合物理限制。
    

其中一個重要進展是 Reasoning 與視覺工具的結合。模型在分析不清楚的區域時，可以再裁切、放大、重新觀察，而不是只依賴第一眼讀取的影像。

例如 2025 年的 VTool-R1 研究，就探索使用 Reinforcement Learning 讓 VLM 學會在推理過程中操作視覺工具。

![](https://www.google.com/s2/favicons?domain=https://arxiv.org&sz=32)

arXiv

### 方向二：High-Resolution Vision — 能分析非常小的細節

這對精密製造與手錶影像非常重要。

假設有一張 4512 × 4512 的照片，其中一個非常小的雷射雕刻文字只佔 120 × 40 pixels。

如果將整張圖縮成 448 × 448，文字可能只剩約 12 × 4 pixels。

VLM 即使再聰明，也很難恢復已經遺失的細節。

新興架構因此逐漸採用：

- Dynamic Resolution：依照圖片尺寸處理不同數量的 visual tokens。
    
- Tiling / Cropping：將大型影像分割成多個高解析度區域。
    
- Adaptive Region Selection：優先處理重要的影像位置。
    
- Visual Token Compression：壓縮不重要的影像資訊。
    
- Active Zoom-in：模型判斷還需要看哪個區域。
    

ICCV 2025 已有研究專門探討利用 Dynamic Region Proposal，在高解析度影像與 visual token 成本之間取得平衡。

![](https://www.google.com/s2/favicons?domain=https://openaccess.thecvf.com&sz=32)

The Computer Vision Foundation

### 方向三：Visual Grounding — 不只回答，還要指出證據在哪裡

傳統 VLM：

「這張影像的 12 點鐘刻度可能有異常。」

Grounded VLM 系統：

「異常位於 image coordinate (x₁, y₁, x₂, y₂)，對應 12 點鐘刻度的左側輪廓。」

![Electronic PCB Detection using Computer Vision:Revolutionizing Industries with YoloV8 | by Jaykumaran R | Medium](https://images.openai.com/static-rsc-4/C42jDWtWsQhTsioadTggxiQj2bUR99mVeJHJcjcy71fblU7V-lVYN2zwEy1E8v7mbc2DZ9o3kay7mC46YsCP37H9PGF3PeCw_BejLvrs5stKyctZuQvntHAkXjXaQS32T5R7r_tbczg8lr_ZDbRRn7UGgUsmykLaX72XqzC7exI?purpose=inline)

Object Detection：找到物件或區域

![ML Gauge Synthetic Data  | SideFX](https://images.openai.com/static-rsc-4/5Yapo9_kRYDUo24tGUcBh7Iw6Z-URy5sHYPeHiMNfpAY1xIKodp1PjCnZV42Pj2X15A1rllvdiwtlQELv8RGsdcYiWkDPvb0co7HyszCLxZhBDiZyvHyLXtHRc0bwyxLUJVyAMEi5KX4nRdVvS6JMJkwIK7hvOgewP2r82ZiloE?purpose=inline)

Segmentation：找出精確 Pixel Mask

Visual Grounding 不一定代表 VLM 自己產生精確 Mask；一個很實用的方案是讓 VLM 解讀任務，再呼叫專門的 segmentation 模型。

例如 Segment Anything Model 3 (SAM 3) 在 2025 年加入以文字概念或參考影像指定目標的能力，支援影像和影片中的 detection、segmentation、tracking。

![](https://www.google.com/s2/favicons?domain=https://ai.meta.com&sz=32)

Research - AI at Meta

### 方向四：Agentic VLM — AI 不只看，還會採取下一步

假設有人要求：

「幫我檢查這張手錶影像的 logo 是否符合原廠。」

傳統 VLM 可能直接回答。

Agentic VLM 的流程則可以是：

1. 辨識錶面品牌和系列
    
2. 使用 Crop Tool 放大 Logo
    
3. 使用 Image Retrieval 找出相同型號的參考影像
    
4. 使用 OCR Tool 擷取文字
    
5. 使用 Image Registration 對齊參考影像
    
6. 比較差異並產生帶有證據的報告
    

在這種設計裡，VLM 像一個懂視覺的任務規劃者，而 Computer Vision 工具負責執行專門的計算。

### 方向五：Video VLM — 從靜態影像走向時間理解

假設工廠中的相機錄到：

- 10:00:01，機械手臂拿起零件。
    
- 10:00:03，零件開始滑落。
    
- 10:00:04，零件掉下來。
    

傳統影像分類模型只能個別分析每個 frame。

Video VLM 則試圖理解事件的順序、因果關係，以及「零件為什麼掉落」。

但影片長時間記憶、精確時間定位和細微動作推理，仍然是具有挑戰性的研究問題。CVPR 2025 的 Video-MME 便專門評估短、中、長影片中的多模態理解能力。

![](https://www.google.com/s2/favicons?domain=https://openaccess.thecvf.com&sz=32)

The Computer Vision Foundation

### 方向六：World Model + VLA — 從理解世界到操作世界

這是 2025 年以後影響很大的研究方向。

VLM 問的是：

「桌上有什麼？」

VLA（Vision-Language-Action Model）問的是：

「桌上有一支手錶，我要怎麼控制機械手臂安全拿起來？」

World Model 則進一步嘗試預測：

「如果機械手臂從這個角度碰到手錶，接下來可能發生什麼事？」

2025 年 Meta 發布 V-JEPA 2，探索從大量影片學習動作、物理環境以及規劃所需的視覺表示；NVIDIA GR00T N1 也展示以 VLM 作為高階規劃模組、搭配動作生成模型的機器人架構。

![](https://www.google.com/s2/favicons?domain=https://ai.meta.com&sz=32)

Research - AI at Meta

+1

## 4. 2025–2026 年有哪些具代表性的模型？

這張表主要列出研究架構與產品路線，不是模型效能排名。

|模型或技術|重要特色|適合研究的問題|
|---|---|---|
|Qwen3-VL|Dynamic visual processing、Reasoning、GUI Agent、Video、Spatial|開源權重 VLM、模型微調|
|Gemini 3 / 3.1 Pro|原生多模態、文件與影片推理|複雜多模態理解|
|SigLIP 2|Image–Text Alignment、Retrieval、Dense Features|影像搜尋、視覺特徵|
|DINOv3|Self-supervised Dense Representation|Fine-grained CV、Segmentation|
|SAM 3|Promptable Segmentation、Tracking|物件分割、區域定位|
|V-JEPA 2|Video Representation、Prediction、Planning|動作理解、World Models|
|GR00T N1 / π0.5|Vision-Language-Action|Robotics、Embodied AI|

這些產品或研究的能力，分別可由 Qwen 官方技術文件、Google 模型卡、SigLIP 2 與 DINOv3 研究，以及 VLA 論文核實。

![](https://www.google.com/s2/favicons?domain=https://github.com&sz=32)

GitHub

+4

Intern 階段最重要的結論：

2025–2026 年的 Multimodal AI 不是單純把 ChatGPT 加上相機，而是把 Perception、Language、Reasoning、Tool Use、Memory、Spatial Understanding 和 Action 逐漸結合成能完成任務的系統。

但在專業場景，讓不同專用模型協同工作，往往仍比要求一個 VLM 完成全部工作更可靠。

# Part II — Senior AI Engineer Level：模型架構、演算法與訓練技術

## 5. 先理解 VLM 的四種主要架構

對 Senior AI Engineer 而言，不能把所有 VLM 都當成「ViT + LLM」。

它們可能具有不同的資訊流、訓練方法和推論成本。

### Architecture A — Dual Encoder

代表：CLIP、SigLIP、SigLIP 2。

Image

Vision Encoder

Image Embedding

Text

Text Encoder

Text Embedding

Similarity / Retrieval / Classification

兩個 Encoder 各自產生 embedding：

\[ z_I=f_\theta(I),\qquad z_T=g_\phi(T) \]

利用 cosine similarity 比較影像與文字：

\[ s(I,T)=\frac{z_I^\top z_T}{\|z_I\|\|z_T\|} \]

例如輸入一萬張手錶照片，搜尋：

`Blue Submariner dial with gold hour markers`

系統可以根據語義相似度找出最相關的影像，而不需要事先為每張圖片建立完全相同的分類標籤。

優點： Retrieval 快、容易建立 Vector DB，適合 Zero-shot Classification。

限制： 它主要學習跨模態表示，不是專門設計來逐字生成複雜分析報告。

### Architecture B — Vision Encoder + Projector + Decoder-only LLM

這是當代 Generative VLM 非常重要的架構。

典型代表包括 LLaVA 系列，以及許多後來的多模態 LLM。

#chatgpt-mermaid-_r_3hn_{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;fill:rgb(13, 13, 13);}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#chatgpt-mermaid-_r_3hn_ .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-_r_3hn_ .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-_r_3hn_ .error-icon{fill:rgb(243, 243, 243);}#chatgpt-mermaid-_r_3hn_ .error-text{fill:rgb(13, 13, 13);stroke:rgb(13, 13, 13);}#chatgpt-mermaid-_r_3hn_ .edge-thickness-normal{stroke-width:1px;}#chatgpt-mermaid-_r_3hn_ .edge-thickness-thick{stroke-width:3.5px;}#chatgpt-mermaid-_r_3hn_ .edge-pattern-solid{stroke-dasharray:0;}#chatgpt-mermaid-_r_3hn_ .edge-thickness-invisible{stroke-width:0;fill:none;}#chatgpt-mermaid-_r_3hn_ .edge-pattern-dashed{stroke-dasharray:3;}#chatgpt-mermaid-_r_3hn_ .edge-pattern-dotted{stroke-dasharray:2;}#chatgpt-mermaid-_r_3hn_ .marker{fill:rgb(143, 143, 143);stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_3hn_ .marker.cross{stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_3hn_ svg{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;}#chatgpt-mermaid-_r_3hn_ p{margin:0;}#chatgpt-mermaid-_r_3hn_ .label{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_3hn_ .cluster-label text{fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_3hn_ .cluster-label span{color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_3hn_ .cluster-label span p{background-color:transparent;}#chatgpt-mermaid-_r_3hn_ .label text,#chatgpt-mermaid-_r_3hn_ span{fill:rgb(13, 13, 13);color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_3hn_ .node rect,#chatgpt-mermaid-_r_3hn_ .node circle,#chatgpt-mermaid-_r_3hn_ .node ellipse,#chatgpt-mermaid-_r_3hn_ .node polygon,#chatgpt-mermaid-_r_3hn_ .node path{fill:rgb(222, 234, 251);stroke:rgb(83, 154, 248);stroke-width:1px;}#chatgpt-mermaid-_r_3hn_ .rough-node .label text,#chatgpt-mermaid-_r_3hn_ .node .label text,#chatgpt-mermaid-_r_3hn_ .image-shape .label,#chatgpt-mermaid-_r_3hn_ .icon-shape .label{text-anchor:middle;}#chatgpt-mermaid-_r_3hn_ .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#chatgpt-mermaid-_r_3hn_ .rough-node .label,#chatgpt-mermaid-_r_3hn_ .node .label,#chatgpt-mermaid-_r_3hn_ .image-shape .label,#chatgpt-mermaid-_r_3hn_ .icon-shape .label{text-align:center;}#chatgpt-mermaid-_r_3hn_ .node.clickable{cursor:pointer;}#chatgpt-mermaid-_r_3hn_ .root .anchor path{fill:rgb(143, 143, 143)!important;stroke-width:0;stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_3hn_ .arrowheadPath{fill:rgb(143, 143, 143);}#chatgpt-mermaid-_r_3hn_ .edgePath .path{stroke:rgb(143, 143, 143);stroke-width:1px;}#chatgpt-mermaid-_r_3hn_ .flowchart-link{stroke:rgb(143, 143, 143);fill:none;}#chatgpt-mermaid-_r_3hn_ .edgeLabel{background-color:rgb(252, 252, 252);text-align:center;}#chatgpt-mermaid-_r_3hn_ .edgeLabel p{background-color:rgb(252, 252, 252);}#chatgpt-mermaid-_r_3hn_ .edgeLabel rect{opacity:0.5;background-color:rgb(252, 252, 252);fill:rgb(252, 252, 252);}#chatgpt-mermaid-_r_3hn_ .labelBkg{background-color:rgba(252, 252, 252, 0.5);}#chatgpt-mermaid-_r_3hn_ .cluster rect{fill:rgb(243, 243, 243);stroke:rgba(0, 0, 0, 0.1);stroke-width:1px;}#chatgpt-mermaid-_r_3hn_ .cluster text{fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_3hn_ .cluster span{color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_3hn_ div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:12px;background:rgb(243, 243, 243);border:1px solid rgba(0, 0, 0, 0.1);border-radius:2px;pointer-events:none;z-index:100;}#chatgpt-mermaid-_r_3hn_ .flowchartTitleText{text-anchor:middle;font-size:18px;fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_3hn_ rect.text{fill:none;stroke-width:0;}#chatgpt-mermaid-_r_3hn_ .icon-shape,#chatgpt-mermaid-_r_3hn_ .image-shape{background-color:rgb(252, 252, 252);text-align:center;}#chatgpt-mermaid-_r_3hn_ .icon-shape p,#chatgpt-mermaid-_r_3hn_ .image-shape p{background-color:rgb(252, 252, 252);padding:2px;}#chatgpt-mermaid-_r_3hn_ .icon-shape .label rect,#chatgpt-mermaid-_r_3hn_ .image-shape .label rect{opacity:0.5;background-color:rgb(252, 252, 252);fill:rgb(252, 252, 252);}#chatgpt-mermaid-_r_3hn_ .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#chatgpt-mermaid-_r_3hn_ .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#chatgpt-mermaid-_r_3hn_ .node .neo-node{stroke:rgb(83, 154, 248);}#chatgpt-mermaid-_r_3hn_ [data-look="neo"].node rect,#chatgpt-mermaid-_r_3hn_ [data-look="neo"].cluster rect,#chatgpt-mermaid-_r_3hn_ [data-look="neo"].node polygon{stroke:url(#chatgpt-mermaid-_r_3hn_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_3hn_ [data-look="neo"].swimlane.cluster rect{filter:none;}#chatgpt-mermaid-_r_3hn_ [data-look="neo"].node path{stroke:url(#chatgpt-mermaid-_r_3hn_-gradient);stroke-width:1px;}#chatgpt-mermaid-_r_3hn_ [data-look="neo"].node .outer-path{filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_3hn_ [data-look="neo"].node .neo-line path{stroke:rgb(83, 154, 248);filter:none;}#chatgpt-mermaid-_r_3hn_ [data-look="neo"].node circle{stroke:url(#chatgpt-mermaid-_r_3hn_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_3hn_ [data-look="neo"].node circle .state-start{fill:#000000;}#chatgpt-mermaid-_r_3hn_ [data-look="neo"].icon-shape .icon{fill:url(#chatgpt-mermaid-_r_3hn_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_3hn_ [data-look="neo"].icon-shape .icon-neo path{stroke:url(#chatgpt-mermaid-_r_3hn_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_3hn_ .node text{font-size:14px;font-weight:600;letter-spacing:normal;fill:rgb(65, 65, 65);}#chatgpt-mermaid-_r_3hn_ .edgeLabels text{font-size:13px;font-weight:600;letter-spacing:-0.08px;fill:rgb(65, 65, 65);}#chatgpt-mermaid-_r_3hn_ .node tspan[font-weight="normal"],#chatgpt-mermaid-_r_3hn_ .edgeLabels tspan[font-weight="normal"]{font-weight:600;}#chatgpt-mermaid-_r_3hn_ .edgeLabel .label rect{opacity:1;rx:13px;ry:13px;fill:rgb(252, 252, 252);stroke:rgb(219, 219, 219);stroke-width:1px;}#chatgpt-mermaid-_r_3hn_ .node rect,#chatgpt-mermaid-_r_3hn_ .node circle,#chatgpt-mermaid-_r_3hn_ .node ellipse,#chatgpt-mermaid-_r_3hn_ .node polygon,#chatgpt-mermaid-_r_3hn_ .node path{fill:rgb(243, 243, 243);stroke:rgba(0, 0, 0, 0.1);stroke-width:1px;}#chatgpt-mermaid-_r_3hn_ .node rect{rx:16px;ry:16px;}#chatgpt-mermaid-_r_3hn_ .node.mermaid-decision .label-container{fill:rgb(249, 249, 249);stroke:rgb(219, 219, 219);stroke-dasharray:2,2;}#chatgpt-mermaid-_r_3hn_ .edgePaths .flowchart-link{stroke:rgb(143, 143, 143);stroke-width:1px;stroke-linecap:round;stroke-linejoin:round;}#chatgpt-mermaid-_r_3hn_ .marker{fill:rgb(143, 143, 143);stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_3hn_ :root{--mermaid-font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";}Image IVision Encoder / ViTVisual Feature TokensProjector / ResamplerUser PromptTokenizer + EmbeddingMultimodal Token SequenceTransformer Decoder LayersText / JSON / Tool Calls

數學上可寫成：

\[ H_v=f_{\text{vision}}(I) \]

\[ Z_v=P_\theta(H_v) \]

\[ E_t=\operatorname{Embed}(\operatorname{Tokenizer}(T)) \]

\[ P(Y\mid I,T)= \prod_{j=1}^{m}P(y_j\mid Z_v,E_t,y_{<j}) \]

這代表模型會依照視覺特徵、問題和先前生成的 tokens，逐個生成答案。

這類架構適合：

- Visual Question Answering。
    
- OCR + Reasoning。
    
- Multi-image Comparison。
    
- Natural Language Explanation。
    
- Image-grounded Agent。
    
- Structured JSON Generation。
    

### Architecture C — Cross-Attention / Hybrid Multimodal Fusion

另一種方法不是把所有視覺 tokens 直接塞進文字序列，而是讓語言模型透過 Cross-Attention 讀取視覺表示。

概念上：

\[ Q=W_QH_{\text{text}} \]

\[ K=W_KH_{\text{vision}},\qquad V=W_VH_{\text{vision}} \]

\[ \operatorname{CrossAttn}(Q,K,V) = \operatorname{softmax} \left(\frac{QK^\top}{\sqrt{d_k}}\right)V \]

這種方式的好處是視覺資訊可作為獨立記憶或特徵集合，透過注意力機制被查詢。

不過實務上，模型的 connector、visual resampling 與 attention 設計差異很大，不應只用一個公式推斷所有產品的內部實作。

### Architecture D — Unified / Native Multimodal Transformer

這是值得持續關注的方向。

目標不再只是讓 LLM 看圖片，而是統一處理：

- Text Tokens
    
- Image Tokens
    
- Audio Tokens
    
- Video / Temporal Tokens
    
- Spatial Tokens
    
- Action Tokens
    

但「Native Multimodal」不代表完全沒有專用 Encoder，也不代表所有 modalities 必須使用同一套 tokenizer。

以 Gemini 3.1 Pro 為例，Google 在 2026 年公布的模型卡明確將其定位為能整合文字、音訊、影像與影片的多模態推理模型。

![](https://www.google.com/s2/favicons?domain=https://deepmind.google&sz=32)

Model Card — Google DeepMind

|架構|主要輸出|最適合|
|---|---|---|
|Dual Encoder|Embedding / Similarity|大規模影像搜尋|
|Encoder + Projector + LLM|Text / JSON / Tool Calls|影像問答與推理|
|Cross-Attention Hybrid|多模態條件生成|需要特殊 Fusion 設計的模型|
|Unified Multimodal|多種模態的理解或生成|通用多模態系統|

## 6. Vision Transformer（ViT）如何將圖片變成 Visual Tokens？

這是 VLM 核心的 Computer Vision 基礎。

### 6.1 Patch Embedding

假設影像大小是：

\[ I\in\mathbb R^{2048\times2048\times3} \]

如果使用 16 × 16 pixel patches，會得到：

\[ N=\frac{2048}{16}\times\frac{2048}{16} =16,384 \]

每個 patch 展平成向量：

\[ x_i\in\mathbb R^{16\times16\times3} =\mathbb R^{768} \]

透過 Linear Projection：

\[ e_i=x_iW_E+b_E \]

這些 patch embeddings 再加上位置編碼，送進 Vision Transformer。

2048 × 2048 Image

示意計算

Image Patches

## 16,384

Patch Tokens

Transformer

Feature Learning

這裡的 16,384 是原始 patch 數量，不一定是最後進入 LLM 的 visual token 數量。實際系統可能進一步使用 pooling、patch merging、resampler 或 token pruning。

### 6.2 Self-Attention

ViT 的核心仍然是 Transformer：

\[ Q=XW_Q,\quad K=XW_K,\quad V=XW_V \]

\[ \operatorname{Attention}(Q,K,V) = \operatorname{softmax} \left(\frac{QK^\top}{\sqrt{d_k}}\right)V \]

Attention 讓不同影像區域的特徵互相交互。

例如辨識 12 點鐘 hour marker 時，模型可能同時利用錶盤、其他刻度和指針的相對關係，而不只是局部紋理。

### 6.3 高解析度會帶來計算瓶頸

對全域 self-attention 而言，注意力矩陣的大小和 token 數量平方有關：

\[ O(N^2) \]

當使用 16 × 16 patches 時：

|Image|Patch 數量|原始全域 Attention 的相對成本|
|---|---|---|
|512 × 512|1,024|1×|
|1024 × 1024|4,096|16×|
|2048 × 2048|16,384|256×|
|4512 × 4512|79,524|約 6,031×|

這只是相對的注意力矩陣規模，不是整個模型實際的 latency 或 GPU memory 比例。

它解釋了為什麼工業用的超高解析度影像，不適合無條件直接丟入一般 VLM。

## 7. 2025–2026 的 High-Resolution VLM 技術

對 Senior Computer Vision / AI Engineer，這部分非常值得深入研究。

### 7.1 Dynamic Resolution

傳統方法常把全部影像縮放到固定尺寸。

Dynamic Resolution 則根據原始大小與細節量，分配不同數量的 visual tokens。

對 2048 × 2048 的影像，可以同時保留：

- 一張 global thumbnail。
    
- 幾個重要區域的高解析度 crops。
    
- 每個 crop 的原始座標。
    
- 各 crop 與全圖的對應關係。
    

### 7.2 Region-of-Interest Token Allocation

假設同一張影像中：

- 80% 是無關背景。
    
- 15% 是一般錶殼。
    
- 5% 包含微小雕刻。
    

若平均分配視覺 token，模型可能浪費大量計算在背景上。

更有效率的策略是：

\[ B=\sum_{i=1}^{n}b_i \]

其中 \(B\) 是總 token budget，\(b_i\) 是分配給第 \(i\) 個區域的 budget。

概念性最佳化目標：

\[ \max_{\{b_i\}} \sum_i U_i(b_i) \quad \text{s.t.}\quad \sum_i b_i\le B \]

\(U_i\) 可代表某個區域的視覺資訊價值或對最終任務的預期貢獻。

### 7.3 Thinking with Images

傳統視覺推理：

`Image → Reasoning → Answer`

新的 Agentic Visual Reasoning：

`Image → Reasoning → Crop → Re-encode → Reasoning → Verify → Answer`

例如模型在看到模糊字體時，不直接猜測，而是要求 Crop Tool 放大原始影像中的文字區域。

這裡有兩個重要取捨：

- 增加 zoom 次數通常會提高推論成本與 latency。
    
- 不一定每次放大都能改善判斷，特別是原始影像已經失焦時。
    

2026 年 ICML 的 Region-to-Image Distillation 探索把推論時反覆放大的部分能力轉移到訓練階段，以減少反覆呼叫工具的成本。

![](https://www.google.com/s2/favicons?domain=https://proceedings.mlr.press&sz=32)

Proceedings of Machine Learning Research

## 8. 進階 VLM：Multi-level Features、Spatial Tokens 與 Video Encoding

### 8.1 為什麼只使用 ViT 最後一層特徵可能不夠？

Vision Transformer 的不同層可能保留不同層級的資訊。

|特徵層級|常見資訊|
|---|---|
|Early Layers|邊緣、顏色、局部紋理|
|Middle Layers|局部結構、形狀、零件關係|
|Late Layers|高階語義、類別和整體物件表示|

對於手錶真偽檢測，兩種資訊都很重要：

- 高階語義：「這是 Submariner 的錶面。」
    
- 細粒度資訊：「字母 R 的筆畫與邊界是否符合參考特徵？」
    

2025 年 Qwen3-VL 技術報告提出 DeepStack 整合多層 ViT 特徵，以及 Interleaved-MRoPE 改善時空表示。

![](https://www.google.com/s2/favicons?domain=https://arxiv.org&sz=32)

arXiv

這反映一個重要趨勢：VLM 不應只依賴最後一層壓縮後的視覺語義。

### 8.2 Spatial Reasoning：2D 走向 3D

一般圖片只包含 pixel 座標：

\[ p=(u,v) \]

實際物體卻存在於三維空間：

\[ P=(X,Y,Z) \]

若使用相機內參 \(K\)、外參 \(R,t\)，並已知深度，可透過投影關係連接：

\[ \lambda \begin{bmatrix}u\\v\\1\end{bmatrix} = K[R\mid t] \begin{bmatrix}X\\Y\\Z\\1\end{bmatrix} \]

2025–2026 的 Spatial VLM 研究，開始更明確地把座標、姿態、旋轉、深度、軌跡等資訊表示成可學習的 token 或結構。

例如模型可能需要回答：

「手錶錶面的法向量與相機光軸夾角大約是多少？」

但需強調：

一般 VLM 對幾何關係的理解，不能直接取代 Camera Calibration、Stereo Geometry、Depth Measurement 或精密量測演算法。

當精度要求達到毫米甚至微米等級時，仍需要實際的幾何校正和量測。

### 8.3 Video Tokens 與時間位置編碼

Video VLM 需要同時表示：

\[ (t,x,y) \]

其中：

- \(t\)：Frame timestamp。
    
- \(x,y\)：Frame 中的空間位置。
    

最簡單的影片表示方式是：

\[ V=\{I_{t_1},I_{t_2},...,I_{t_n}\} \]

但直接把全部 frames 變成 visual tokens，成本可能極高。

實務技術包括 Temporal Sampling、Keyframe Selection、Temporal Pooling、Video Token Compression、Long-term Memory，以及時間與空間的位置編碼。

比較有代表性的設計是 Qwen3-VL 的時空位置表示與 Timestamp Alignment；它將時間資訊納入多模態模型的建模和定位過程。

![](https://www.google.com/s2/favicons?domain=https://github.com&sz=32)

GitHub

對工業設備而言，如果你要監控的是非常短暫的高速動作，單靠低頻率 frame sampling 可能會完全錯過故障瞬間。

## 9. VLM Training：完整訓練 Pipeline

這是 Senior AI Engineer 面試與實際模型開發非常重要的一部分。

不能只說「使用 PyTorch Fine-tune VLM」。必須了解不同階段到底訓練什麼。

#chatgpt-mermaid-_r_3j6_{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;fill:rgb(13, 13, 13);}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#chatgpt-mermaid-_r_3j6_ .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-_r_3j6_ .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-_r_3j6_ .error-icon{fill:rgb(243, 243, 243);}#chatgpt-mermaid-_r_3j6_ .error-text{fill:rgb(13, 13, 13);stroke:rgb(13, 13, 13);}#chatgpt-mermaid-_r_3j6_ .edge-thickness-normal{stroke-width:1px;}#chatgpt-mermaid-_r_3j6_ .edge-thickness-thick{stroke-width:3.5px;}#chatgpt-mermaid-_r_3j6_ .edge-pattern-solid{stroke-dasharray:0;}#chatgpt-mermaid-_r_3j6_ .edge-thickness-invisible{stroke-width:0;fill:none;}#chatgpt-mermaid-_r_3j6_ .edge-pattern-dashed{stroke-dasharray:3;}#chatgpt-mermaid-_r_3j6_ .edge-pattern-dotted{stroke-dasharray:2;}#chatgpt-mermaid-_r_3j6_ .marker{fill:rgb(143, 143, 143);stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_3j6_ .marker.cross{stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_3j6_ svg{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;}#chatgpt-mermaid-_r_3j6_ p{margin:0;}#chatgpt-mermaid-_r_3j6_ .label{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_3j6_ .cluster-label text{fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_3j6_ .cluster-label span{color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_3j6_ .cluster-label span p{background-color:transparent;}#chatgpt-mermaid-_r_3j6_ .label text,#chatgpt-mermaid-_r_3j6_ span{fill:rgb(13, 13, 13);color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_3j6_ .node rect,#chatgpt-mermaid-_r_3j6_ .node circle,#chatgpt-mermaid-_r_3j6_ .node ellipse,#chatgpt-mermaid-_r_3j6_ .node polygon,#chatgpt-mermaid-_r_3j6_ .node path{fill:rgb(222, 234, 251);stroke:rgb(83, 154, 248);stroke-width:1px;}#chatgpt-mermaid-_r_3j6_ .rough-node .label text,#chatgpt-mermaid-_r_3j6_ .node .label text,#chatgpt-mermaid-_r_3j6_ .image-shape .label,#chatgpt-mermaid-_r_3j6_ .icon-shape .label{text-anchor:middle;}#chatgpt-mermaid-_r_3j6_ .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#chatgpt-mermaid-_r_3j6_ .rough-node .label,#chatgpt-mermaid-_r_3j6_ .node .label,#chatgpt-mermaid-_r_3j6_ .image-shape .label,#chatgpt-mermaid-_r_3j6_ .icon-shape .label{text-align:center;}#chatgpt-mermaid-_r_3j6_ .node.clickable{cursor:pointer;}#chatgpt-mermaid-_r_3j6_ .root .anchor path{fill:rgb(143, 143, 143)!important;stroke-width:0;stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_3j6_ .arrowheadPath{fill:rgb(143, 143, 143);}#chatgpt-mermaid-_r_3j6_ .edgePath .path{stroke:rgb(143, 143, 143);stroke-width:1px;}#chatgpt-mermaid-_r_3j6_ .flowchart-link{stroke:rgb(143, 143, 143);fill:none;}#chatgpt-mermaid-_r_3j6_ .edgeLabel{background-color:rgb(252, 252, 252);text-align:center;}#chatgpt-mermaid-_r_3j6_ .edgeLabel p{background-color:rgb(252, 252, 252);}#chatgpt-mermaid-_r_3j6_ .edgeLabel rect{opacity:0.5;background-color:rgb(252, 252, 252);fill:rgb(252, 252, 252);}#chatgpt-mermaid-_r_3j6_ .labelBkg{background-color:rgba(252, 252, 252, 0.5);}#chatgpt-mermaid-_r_3j6_ .cluster rect{fill:rgb(243, 243, 243);stroke:rgba(0, 0, 0, 0.1);stroke-width:1px;}#chatgpt-mermaid-_r_3j6_ .cluster text{fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_3j6_ .cluster span{color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_3j6_ div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:12px;background:rgb(243, 243, 243);border:1px solid rgba(0, 0, 0, 0.1);border-radius:2px;pointer-events:none;z-index:100;}#chatgpt-mermaid-_r_3j6_ .flowchartTitleText{text-anchor:middle;font-size:18px;fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_3j6_ rect.text{fill:none;stroke-width:0;}#chatgpt-mermaid-_r_3j6_ .icon-shape,#chatgpt-mermaid-_r_3j6_ .image-shape{background-color:rgb(252, 252, 252);text-align:center;}#chatgpt-mermaid-_r_3j6_ .icon-shape p,#chatgpt-mermaid-_r_3j6_ .image-shape p{background-color:rgb(252, 252, 252);padding:2px;}#chatgpt-mermaid-_r_3j6_ .icon-shape .label rect,#chatgpt-mermaid-_r_3j6_ .image-shape .label rect{opacity:0.5;background-color:rgb(252, 252, 252);fill:rgb(252, 252, 252);}#chatgpt-mermaid-_r_3j6_ .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#chatgpt-mermaid-_r_3j6_ .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#chatgpt-mermaid-_r_3j6_ .node .neo-node{stroke:rgb(83, 154, 248);}#chatgpt-mermaid-_r_3j6_ [data-look="neo"].node rect,#chatgpt-mermaid-_r_3j6_ [data-look="neo"].cluster rect,#chatgpt-mermaid-_r_3j6_ [data-look="neo"].node polygon{stroke:url(#chatgpt-mermaid-_r_3j6_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_3j6_ [data-look="neo"].swimlane.cluster rect{filter:none;}#chatgpt-mermaid-_r_3j6_ [data-look="neo"].node path{stroke:url(#chatgpt-mermaid-_r_3j6_-gradient);stroke-width:1px;}#chatgpt-mermaid-_r_3j6_ [data-look="neo"].node .outer-path{filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_3j6_ [data-look="neo"].node .neo-line path{stroke:rgb(83, 154, 248);filter:none;}#chatgpt-mermaid-_r_3j6_ [data-look="neo"].node circle{stroke:url(#chatgpt-mermaid-_r_3j6_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_3j6_ [data-look="neo"].node circle .state-start{fill:#000000;}#chatgpt-mermaid-_r_3j6_ [data-look="neo"].icon-shape .icon{fill:url(#chatgpt-mermaid-_r_3j6_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_3j6_ [data-look="neo"].icon-shape .icon-neo path{stroke:url(#chatgpt-mermaid-_r_3j6_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_3j6_ .node text{font-size:14px;font-weight:600;letter-spacing:normal;fill:rgb(65, 65, 65);}#chatgpt-mermaid-_r_3j6_ .edgeLabels text{font-size:13px;font-weight:600;letter-spacing:-0.08px;fill:rgb(65, 65, 65);}#chatgpt-mermaid-_r_3j6_ .node tspan[font-weight="normal"],#chatgpt-mermaid-_r_3j6_ .edgeLabels tspan[font-weight="normal"]{font-weight:600;}#chatgpt-mermaid-_r_3j6_ .edgeLabel .label rect{opacity:1;rx:13px;ry:13px;fill:rgb(252, 252, 252);stroke:rgb(219, 219, 219);stroke-width:1px;}#chatgpt-mermaid-_r_3j6_ .node rect,#chatgpt-mermaid-_r_3j6_ .node circle,#chatgpt-mermaid-_r_3j6_ .node ellipse,#chatgpt-mermaid-_r_3j6_ .node polygon,#chatgpt-mermaid-_r_3j6_ .node path{fill:rgb(243, 243, 243);stroke:rgba(0, 0, 0, 0.1);stroke-width:1px;}#chatgpt-mermaid-_r_3j6_ .node rect{rx:16px;ry:16px;}#chatgpt-mermaid-_r_3j6_ .node.mermaid-decision .label-container{fill:rgb(249, 249, 249);stroke:rgb(219, 219, 219);stroke-dasharray:2,2;}#chatgpt-mermaid-_r_3j6_ .edgePaths .flowchart-link{stroke:rgb(143, 143, 143);stroke-width:1px;stroke-linecap:round;stroke-linejoin:round;}#chatgpt-mermaid-_r_3j6_ .marker{fill:rgb(143, 143, 143);stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_3j6_ :root{--mermaid-font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";}Image / Text / Video DatasetsVisual RepresentationPretrainingVision-Language AlignmentMultimodal PretrainingInstruction SFTPreference / RL Post-trainingDomain AdaptationEvaluation + CalibrationDeployment

這是常見訓練階段的概念流程，不代表每個 VLM 都依照完全相同的順序訓練。

### Stage 1 — Vision Representation Pretraining

目標：讓 Vision Encoder 學到有效的視覺特徵。

常見方法包括：

Supervised Learning

\[ \mathcal L_{\text{cls}} = -\sum_c y_c\log p_c \]

利用有標籤的影像進行分類訓練。

Self-supervised Learning

不一定需要人工分類標籤，可以使用 self-distillation、masked image modeling 或其他自監督任務。

DINOv3 就是這一研究路線的代表，尤其重視可用於 Dense Prediction 的視覺特徵。

![](https://www.google.com/s2/favicons?domain=https://ai.meta.com&sz=32)

Research - AI at Meta

### Stage 2 — Image-Text Alignment

目標：讓影像和文字的向量表示能互相對應。

CLIP-style Contrastive Learning 常使用 InfoNCE 類型的 Loss。

假設一個 batch 有 \(N\) 組 image-text pairs：

\[ \mathcal L_{I\rightarrow T} = -\frac1N\sum_{i=1}^{N} \log \frac{\exp(s_{ii}/\tau)} {\sum_{j=1}^{N}\exp(s_{ij}/\tau)} \]

其中：

- \(s_{ii}\)：正確 image-text pair 的相似度。
    
- \(s_{ij}\)：image \(i\) 與 text \(j\) 的相似度。
    
- \(\tau\)：temperature parameter。
    

一般也會加入 Text-to-Image 的反向 loss。

而 SigLIP 系列使用不同的 sigmoid-based pairwise loss 設計；SigLIP 2 又加入其他訓練目標，例如自監督學習與改善細粒度特徵的任務。

![](https://www.google.com/s2/favicons?domain=https://google-research.github.io&sz=32)

big_vision

### Stage 3 — Vision-to-LLM Alignment

Vision Encoder 與 LLM 的 embedding 維度不一定相同。

例如：

\[ H_v\in\mathbb R^{N\times1024} \]

但 LLM 的 hidden dimension 是 4096。

可以使用：

\[ Z_v=H_vW_P \]

其中：

\[ W_P\in\mathbb R^{1024\times4096} \]

也可以使用多層 MLP、Cross-Attention Resampler、Q-Former 類型模組，而不只是單層 Linear Projector。

訓練策略可能包含：

- Freeze Vision Encoder。
    
- Freeze LLM。
    
- 只訓練 Projector。
    
- 將 Projector 訓練穩定後，再解凍部分模組共同訓練。
    

具體順序依模型設計而不同。

### Stage 4 — Multimodal Instruction SFT

SFT（Supervised Fine-Tuning）讓模型學習按照使用者指令回答。

一筆資料可以長這樣：

```
{
  "images": [
    "watch_front.png",
    "watch_logo_crop.png"
  ],
  "instruction": "Identify the dial text and report any visible inconsistencies.",
  "response": {
    "dial_text": ["ROLEX", "SUBMARINER"],
    "suspected_regions": [
      {
        "image_id": "watch_logo_crop.png",
        "region": [84, 65, 230, 120],
        "finding": "Requires comparison with reference"
      }
    ],
    "status": "needs_verification"
  }
}
```

訓練時使用 Token-level Cross Entropy：

\[ \mathcal L_{\text{SFT}} = -\sum_{t=1}^{T} \log P_\theta(y_t\mid I,X,y_{<t}) \]

通常只對需要模型學習生成的 target tokens 計算 loss，而不是無差別地把整個 prompt 都當成預測目標。

對於 Domain-specific VLM，訓練品質往往取決於能否提供精確的 visual grounding、完整的標註規範，以及正確的拒答示例。

### Stage 5 — Post-training：DPO、RL 與 Verifiable Rewards

這是 2025–2026 非常值得關注的方向。

SFT 教模型模仿正確答案，但不一定讓模型具備良好的驗證行為。

例如模型可能學會：

「遇到一張手錶圖片，就產生一篇看起來合理的真偽分析。」

但如果實際上沒有足夠證據，它仍可能自信地猜測。

Post-training 可以針對下列特性最佳化：

- 答案正確性。
    
- Grounding 是否正確。
    
- 工具使用是否有幫助。
    
- 是否能在缺乏資訊時拒絕判斷。
    
- JSON 是否符合 Schema。
    
- 是否能找出可驗證的參考證據。
    

#### DPO（Direct Preference Optimization）

可以使用偏好配對：

- Preferred：指出圖片區域、引用來源、無法確認時不作結論。
    
- Rejected：沒有來源，卻直接宣稱手錶一定是假貨。
    

DPO 學習提高 preferred response 相對於 rejected response 的機率。

#### RL / GRPO

Reinforcement Learning 則可以使用可計算的 reward。

一個工程上可以設計的概念性 reward：

\[ R= 0.35R_{\text{accuracy}} +0.25R_{\text{grounding}} +0.15R_{\text{tool}} +0.15R_{\text{calibration}} +0.10R_{\text{format}} \]

這些權重是設計示例，並非任何特定模型的已公布設定。

例如：

\[ R_{\text{grounding}}= \operatorname{IoU}(B_{\text{pred}},B_{\text{GT}}) \]

如果模型預測的區域與 Ground Truth 重疊較多，便可以給較高 reward。

2025 年 Visual-RFT 研究就將 IoU 等可驗證的視覺任務 reward 結合 GRPO，探索少量資料下的視覺模型強化微調。

![](https://www.google.com/s2/favicons?domain=https://arxiv.org&sz=32)

arXiv

Senior Engineer 還需要注意 reward hacking：模型可能學會滿足評分器，而不是真正提高影像判斷能力。因此必須使用獨立評估集和人工抽查。

### Stage 6 — PEFT：LoRA / QLoRA

假設有一個數十億參數的 VLM，不一定需要對全部權重訓練。

LoRA 將權重更新表示成低秩分解：

\[ W'=W+\Delta W \]

\[ \Delta W=BA \]

若：

\[ W\in\mathbb R^{d_{\text{out}}\times d_{\text{in}}} \]

則：

\[ B\in\mathbb R^{d_{\text{out}}\times r},\quad A\in\mathbb R^{r\times d_{\text{in}}} \]

其中：

\[ r\ll \min(d_{\text{in}},d_{\text{out}}) \]

這使需要訓練的參數數量大幅降低。

對手錶工業影像領域，可考慮三種做法：

|訓練方式|優點|缺點|
|---|---|---|
|Projector-only|成本低，容易開始|改善視覺表徵的能力有限|
|LLM LoRA|容易適應專業術語、輸出格式和任務|無法保證微小特徵被 Vision Encoder 看見|
|Vision Adapter + LLM LoRA|可同時調整視覺域與語言域|更複雜，需謹慎防止 overfitting|

如果真實問題出在 Image Encoder 無法辨識細小雕刻，只對 LLM 加 LoRA 不一定有效。

## 10. Multimodal RAG：讓 VLM 使用外部視覺知識

傳統 Text RAG：

`Question → Text Embedding → Vector DB → Text Documents → LLM`

Multimodal RAG：

`Question + Image → Multimodal Retrieval → Reference Images / Documents / Charts → VLM`

對專業的視覺檢驗，這是很大的差異。

例如你要比較 Rolex 不同年份的文字排版：

資料庫可能包含：

- 官方技術文件。
    
- 同系列不同年份的參考影像。
    
- 字體特徵的統計分布。
    
- 零件替換紀錄。
    
- 已確認的假貨案例。
    

這些知識不適合只依賴模型訓練時的記憶。

### Multimodal RAG 的三種主要檢索方式

|方法|適用情境|
|---|---|
|Text / Metadata Retrieval|型號、年份、零件編號等精確條件|
|Image Embedding Retrieval|搜尋外觀類似的圖片|
|Visual Document Retrieval|搜尋包含版面、圖表、照片的技術文件|

例如 ICLR 2025 的 ColPali 使用頁面影像的 Multi-vector Representation 與 Late Interaction 進行文件檢索，減少單純文字化文件可能遺失的版面和視覺資訊。

![](https://www.google.com/s2/favicons?domain=https://proceedings.iclr.cc&sz=32)

ICLR Proceedings

但是對製造業的精確料號匹配，Vector Search 不能完全取代 SQL 或 metadata filtering。

實務上更合理的是：

\[ \text{Exact Filter} \rightarrow \text{Vector Retrieval} \rightarrow \text{Reranking} \rightarrow \text{Verified Evidence} \]

例如先限定 Rolex Series、Family、Reference、Production Period，再做視覺相似度搜尋。

## 11. Agentic VLM：Reasoning + Tool Calling + Verification

Senior AI Engineer 需要理解，Agent 不是一種特定神經網路架構，而是模型與工具、記憶及控制流程組合起來的系統。

一個 Visual Agent 可以分為：

#chatgpt-mermaid-_r_3ki_{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;fill:rgb(13, 13, 13);}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#chatgpt-mermaid-_r_3ki_ .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-_r_3ki_ .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-_r_3ki_ .error-icon{fill:rgb(243, 243, 243);}#chatgpt-mermaid-_r_3ki_ .error-text{fill:rgb(13, 13, 13);stroke:rgb(13, 13, 13);}#chatgpt-mermaid-_r_3ki_ .edge-thickness-normal{stroke-width:1px;}#chatgpt-mermaid-_r_3ki_ .edge-thickness-thick{stroke-width:3.5px;}#chatgpt-mermaid-_r_3ki_ .edge-pattern-solid{stroke-dasharray:0;}#chatgpt-mermaid-_r_3ki_ .edge-thickness-invisible{stroke-width:0;fill:none;}#chatgpt-mermaid-_r_3ki_ .edge-pattern-dashed{stroke-dasharray:3;}#chatgpt-mermaid-_r_3ki_ .edge-pattern-dotted{stroke-dasharray:2;}#chatgpt-mermaid-_r_3ki_ .marker{fill:rgb(143, 143, 143);stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_3ki_ .marker.cross{stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_3ki_ svg{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;}#chatgpt-mermaid-_r_3ki_ p{margin:0;}#chatgpt-mermaid-_r_3ki_ .label{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_3ki_ .cluster-label text{fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_3ki_ .cluster-label span{color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_3ki_ .cluster-label span p{background-color:transparent;}#chatgpt-mermaid-_r_3ki_ .label text,#chatgpt-mermaid-_r_3ki_ span{fill:rgb(13, 13, 13);color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_3ki_ .node rect,#chatgpt-mermaid-_r_3ki_ .node circle,#chatgpt-mermaid-_r_3ki_ .node ellipse,#chatgpt-mermaid-_r_3ki_ .node polygon,#chatgpt-mermaid-_r_3ki_ .node path{fill:rgb(222, 234, 251);stroke:rgb(83, 154, 248);stroke-width:1px;}#chatgpt-mermaid-_r_3ki_ .rough-node .label text,#chatgpt-mermaid-_r_3ki_ .node .label text,#chatgpt-mermaid-_r_3ki_ .image-shape .label,#chatgpt-mermaid-_r_3ki_ .icon-shape .label{text-anchor:middle;}#chatgpt-mermaid-_r_3ki_ .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#chatgpt-mermaid-_r_3ki_ .rough-node .label,#chatgpt-mermaid-_r_3ki_ .node .label,#chatgpt-mermaid-_r_3ki_ .image-shape .label,#chatgpt-mermaid-_r_3ki_ .icon-shape .label{text-align:center;}#chatgpt-mermaid-_r_3ki_ .node.clickable{cursor:pointer;}#chatgpt-mermaid-_r_3ki_ .root .anchor path{fill:rgb(143, 143, 143)!important;stroke-width:0;stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_3ki_ .arrowheadPath{fill:rgb(143, 143, 143);}#chatgpt-mermaid-_r_3ki_ .edgePath .path{stroke:rgb(143, 143, 143);stroke-width:1px;}#chatgpt-mermaid-_r_3ki_ .flowchart-link{stroke:rgb(143, 143, 143);fill:none;}#chatgpt-mermaid-_r_3ki_ .edgeLabel{background-color:rgb(252, 252, 252);text-align:center;}#chatgpt-mermaid-_r_3ki_ .edgeLabel p{background-color:rgb(252, 252, 252);}#chatgpt-mermaid-_r_3ki_ .edgeLabel rect{opacity:0.5;background-color:rgb(252, 252, 252);fill:rgb(252, 252, 252);}#chatgpt-mermaid-_r_3ki_ .labelBkg{background-color:rgba(252, 252, 252, 0.5);}#chatgpt-mermaid-_r_3ki_ .cluster rect{fill:rgb(243, 243, 243);stroke:rgba(0, 0, 0, 0.1);stroke-width:1px;}#chatgpt-mermaid-_r_3ki_ .cluster text{fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_3ki_ .cluster span{color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_3ki_ div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:12px;background:rgb(243, 243, 243);border:1px solid rgba(0, 0, 0, 0.1);border-radius:2px;pointer-events:none;z-index:100;}#chatgpt-mermaid-_r_3ki_ .flowchartTitleText{text-anchor:middle;font-size:18px;fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_3ki_ rect.text{fill:none;stroke-width:0;}#chatgpt-mermaid-_r_3ki_ .icon-shape,#chatgpt-mermaid-_r_3ki_ .image-shape{background-color:rgb(252, 252, 252);text-align:center;}#chatgpt-mermaid-_r_3ki_ .icon-shape p,#chatgpt-mermaid-_r_3ki_ .image-shape p{background-color:rgb(252, 252, 252);padding:2px;}#chatgpt-mermaid-_r_3ki_ .icon-shape .label rect,#chatgpt-mermaid-_r_3ki_ .image-shape .label rect{opacity:0.5;background-color:rgb(252, 252, 252);fill:rgb(252, 252, 252);}#chatgpt-mermaid-_r_3ki_ .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#chatgpt-mermaid-_r_3ki_ .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#chatgpt-mermaid-_r_3ki_ .node .neo-node{stroke:rgb(83, 154, 248);}#chatgpt-mermaid-_r_3ki_ [data-look="neo"].node rect,#chatgpt-mermaid-_r_3ki_ [data-look="neo"].cluster rect,#chatgpt-mermaid-_r_3ki_ [data-look="neo"].node polygon{stroke:url(#chatgpt-mermaid-_r_3ki_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_3ki_ [data-look="neo"].swimlane.cluster rect{filter:none;}#chatgpt-mermaid-_r_3ki_ [data-look="neo"].node path{stroke:url(#chatgpt-mermaid-_r_3ki_-gradient);stroke-width:1px;}#chatgpt-mermaid-_r_3ki_ [data-look="neo"].node .outer-path{filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_3ki_ [data-look="neo"].node .neo-line path{stroke:rgb(83, 154, 248);filter:none;}#chatgpt-mermaid-_r_3ki_ [data-look="neo"].node circle{stroke:url(#chatgpt-mermaid-_r_3ki_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_3ki_ [data-look="neo"].node circle .state-start{fill:#000000;}#chatgpt-mermaid-_r_3ki_ [data-look="neo"].icon-shape .icon{fill:url(#chatgpt-mermaid-_r_3ki_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_3ki_ [data-look="neo"].icon-shape .icon-neo path{stroke:url(#chatgpt-mermaid-_r_3ki_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_3ki_ .node text{font-size:14px;font-weight:600;letter-spacing:normal;fill:rgb(65, 65, 65);}#chatgpt-mermaid-_r_3ki_ .edgeLabels text{font-size:13px;font-weight:600;letter-spacing:-0.08px;fill:rgb(65, 65, 65);}#chatgpt-mermaid-_r_3ki_ .node tspan[font-weight="normal"],#chatgpt-mermaid-_r_3ki_ .edgeLabels tspan[font-weight="normal"]{font-weight:600;}#chatgpt-mermaid-_r_3ki_ .edgeLabel .label rect{opacity:1;rx:13px;ry:13px;fill:rgb(252, 252, 252);stroke:rgb(219, 219, 219);stroke-width:1px;}#chatgpt-mermaid-_r_3ki_ .node rect,#chatgpt-mermaid-_r_3ki_ .node circle,#chatgpt-mermaid-_r_3ki_ .node ellipse,#chatgpt-mermaid-_r_3ki_ .node polygon,#chatgpt-mermaid-_r_3ki_ .node path{fill:rgb(243, 243, 243);stroke:rgba(0, 0, 0, 0.1);stroke-width:1px;}#chatgpt-mermaid-_r_3ki_ .node rect{rx:16px;ry:16px;}#chatgpt-mermaid-_r_3ki_ .node.mermaid-decision .label-container{fill:rgb(249, 249, 249);stroke:rgb(219, 219, 219);stroke-dasharray:2,2;}#chatgpt-mermaid-_r_3ki_ .edgePaths .flowchart-link{stroke:rgb(143, 143, 143);stroke-width:1px;stroke-linecap:round;stroke-linejoin:round;}#chatgpt-mermaid-_r_3ki_ .marker{fill:rgb(143, 143, 143);stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_3ki_ :root{--mermaid-font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";}Image + GoalVLM PlannerNeed more evidence?Tool SelectionCrop / OCR / Segmentation /RAGEvidence ValidationFinal Structured OutputTool / Step BudgetYesNo

在生產系統中，需要限制：

1. 最大工具呼叫次數。
    
2. 最大 visual token 數量。
    
3. 最大 GPU 推論成本。
    
4. 哪些工具可以讀取資料。
    
5. 哪些工具可以修改資料或控制硬體。
    
6. 任務失敗時如何停止。
    
7. 不確定時如何轉交人工。
    

這種流程也被稱為 Agentic Visual Reasoning 或 Tool-augmented VLM。

一個 2026 年值得注意的方向，是讓工具不只協助推理，也協助模型驗證自己的推理。例如 ICML 2026 的 Agent0-VL 研究了 tool-integrated self-evaluation 與 self-repair。

![](https://www.google.com/s2/favicons?domain=https://proceedings.mlr.press&sz=32)

Proceedings of Machine Learning Research

不過在工業設備上，不能讓 VLM 直接不受限制地控制 Zaber stage、雷射或光源。動作必須通過獨立的 safety constraints 和 deterministic controller。

## 12. Vision-Language-Action（VLA）與 World Models

這兩種技術有關係，但並不是相同東西。

### VLM

\[ P(Y_{\text{text}}\mid I,T) \]

主要理解視覺內容並生成回答。

### VLA

\[ \pi_\theta(a_t\mid o_{\le t},l,a_{<t}) \]

主要依照 observation、語言指令和過去動作，生成下一步 action。

其中 \(a_t\) 可以是：

- 機械手臂關節控制量。
    
- 末端執行器位姿。
    
- Gripper 開合。
    
- 一段連續動作序列。
    

某些 VLA 將動作離散 token 化，某些則使用 Diffusion 或 Flow-based Action Head 產生連續動作。

### World Model

目標是預測施加某個 action 後，環境狀態可能如何變化：

\[ \hat{s}_{t+1}=f_\theta(s_t,a_t) \]

也可能在 latent space 中建模：

\[ \hat{z}_{t+1}=f_\theta(z_t,a_t) \]

例如在攝影機下面移動機械手臂時，模型要預測移動後物體可能出現的位置和遮擋情況。

NVIDIA GR00T N1 的設計使用較高階的 VLM 理解與規劃模組，搭配產生連續動作的 Diffusion Transformer；Physical Intelligence 的 π0.5 則研究 VLA 在新環境中的泛化能力。

![](https://www.google.com/s2/favicons?domain=https://developer.nvidia.com&sz=32)

NVIDIA Technical Blog

+1

這條路線非常值得有 Computer Vision + Robotics 背景的工程師關注。

## 13. Inference：真正部署 VLM 時，性能瓶頸在哪裡？

VLM 推論的成本，不只是 LLM parameter 數量。

可粗略拆成：

\[ T_{\text{total}}= T_{\text{preprocess}}+ T_{\text{vision}}+ T_{\text{prefill}}+ T_{\text{decode}}+ T_{\text{tools}} \]

其中：

- \(T_{\text{preprocess}}\)：影像解碼、resize、crop、normalization。
    
- \(T_{\text{vision}}\)：Vision Encoder。
    
- \(T_{\text{prefill}}\)：Multimodal token processing。
    
- \(T_{\text{decode}}\)：生成答案。
    
- \(T_{\text{tools}}\)：外部影像處理與檢索工具。
    

對大量相機影像，前面三項可能占很大的比例。

### GPU Memory 估算

單看語言模型權重，理論上：

|LLM 大小|BF16 / FP16 權重|4-bit 權重理想下限|
|---|---|---|
|3B|約 6 GB|約 1.5 GB|
|8B|約 16 GB|約 4 GB|
|32B|約 64 GB|約 16 GB|

這些數值尚未包含 Vision Encoder、Quantization Metadata、KV Cache、Activations、Batching 及框架額外需求。

真正部署時，不可以直接把表格中的權重大小當成最低 GPU VRAM 需求。

重要的最佳化方法包括：

- Quantization：INT8、4-bit、FP8 等。
    
- Visual Token Pruning / Pooling。
    
- Dynamic Batching。
    
- KV Cache 管理。
    
- Prefix Caching。
    
- 分離快模型和慢模型。
    
- Local / Cloud Hybrid Inference。
    
- Distillation 至專用小模型。
    

例如大量手錶影像可以先用輕量化 CV 模型完成常規檢查，只有疑似異常時才送到較大型的 Reasoning VLM。

這通常比每張圖片都呼叫大型模型更有效率。

## 14. VLM Evaluation：2026 年不能只看 Accuracy

對 Senior Engineer，我會把 Evaluation 分成五個層級。

|層級|評估內容|常見指標|
|---|---|---|
|Perception|物件、文字、局部特徵是否正確|mAP、IoU、CER、WER|
|Reasoning|是否正確比較、推論|Task Accuracy、Consistency|
|Grounding|回答有沒有對應正確影像區域|Grounding IoU、Recall|
|Reliability|會不會幻覺、過度自信|Hallucination Rate、ECE、Brier|
|System / Agent|整個工作流程是否成功|Task Success、Latency、Cost、Safety|

其中 ECE（Expected Calibration Error）主要用來檢查預測信心與實際正確率是否一致。

### 常見公開 Benchmark

- MMMU / MMMU-Pro： 專業領域的多模態推理。
    
- MathVista / MathVision： 數學圖像推理。
    
- DocVQA / ChartQA： 文件與圖表理解。
    
- Video-MME： 影片與多模態理解。
    
- OSWorld： 操作真實桌面軟體的多模態 Agent。
    
- RefCOCO 類型評估： 文字描述對應物件的位置。
    

MMMU-Pro 對原先 Benchmark 加入更嚴格的設計，例如排除不看圖也能回答的題目；OSWorld 則使用實際軟體環境與執行結果評估 Agent，而不是只看模型回答是否流暢。

![](https://www.google.com/s2/favicons?domain=https://aclanthology.org&sz=32)

ACL Anthology

+1

### Senior Engineer 必須特別注意的 Failure Modes

Visual Hallucination： 沒有看見的文字或零件卻聲稱存在。

Grounding Failure： 文字解釋看似正確，但指到錯誤位置。

Perception Bottleneck： 推理再強，影像中的必要資訊已在 resize 時遺失。

Cross-image Confusion： 混淆不同照片、相機或時間點的證據。

Shortcut Learning： 模型靠背景、拍攝環境、序號或資料來源猜標籤，而不是依據真正的缺陷。

Domain Shift： 換相機、光源、focus 或新錶款後，準確率大幅下降。

Visual Prompt Injection： 影像或文件裡的文字試圖命令 AI 忽略原本任務，甚至呼叫不應使用的工具。

此外，生成出來的自然語言解釋並不等於模型內部推理一定忠實；必須另外測試解釋與真正視覺證據的一致性。

這些問題在一般聊天場景可能只是回答錯誤，但在工業自動檢測會直接影響產品品質和客戶信任。

# Part III — 完整實際案例：利用 VLM 建立高精度 Rolex 手錶真偽分析系統

接下來用一個具體的工業影像案例，把前面的技術全部串起來。

假設要設計的系統能自動拍攝一支 Rolex Submariner，並回答：

> 這支手錶的 Dial、Hands、Hour Markers、Rehaut、Case 和 Movement 是否與該型號、年份及原廠規格一致？哪些區域值得懷疑？能否提出可以驗證的證據？

這個例子會同時用到：

- Multi-camera Computer Vision
    
- High-Resolution VLM
    
- Visual Reasoning
    
- Multimodal RAG
    
- Tool Calling
    
- Segmentation
    
- Anomaly Detection
    
- Bayesian Evidence Fusion
    
- Model Evaluation
    

## 15. 首先決定：為什麼不直接用一個 VLM？

有三種主要方案。

|方案|架構|評價|
|---|---|---|
|A|40 張影像全部送進 VLM，要求判斷真假|實作簡單，但精度、成本與可驗證性不足|
|B|CNN / UNet / OCR / 傳統統計模型|精細量測強，但跨零件語義推理不靈活|
|C|CV + VLM + Retrieval + Bayesian Fusion|最適合需要可解釋、可稽核的專業檢測|

我建議使用 C。

但要明確區分職責：

VLM 負責理解、選擇工具、整理跨影像證據與解釋；專用 CV 模型負責精確量測；經過校準的統計模型負責正式證據融合。

VLM 不應以一個自然語言生成的「95% 真品」分數，直接取代可追溯的統計決策模型。

## 16. 完整系統 Architecture

#chatgpt-mermaid-_r_3lt_{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;fill:rgb(13, 13, 13);}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#chatgpt-mermaid-_r_3lt_ .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-_r_3lt_ .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-_r_3lt_ .error-icon{fill:rgb(243, 243, 243);}#chatgpt-mermaid-_r_3lt_ .error-text{fill:rgb(13, 13, 13);stroke:rgb(13, 13, 13);}#chatgpt-mermaid-_r_3lt_ .edge-thickness-normal{stroke-width:1px;}#chatgpt-mermaid-_r_3lt_ .edge-thickness-thick{stroke-width:3.5px;}#chatgpt-mermaid-_r_3lt_ .edge-pattern-solid{stroke-dasharray:0;}#chatgpt-mermaid-_r_3lt_ .edge-thickness-invisible{stroke-width:0;fill:none;}#chatgpt-mermaid-_r_3lt_ .edge-pattern-dashed{stroke-dasharray:3;}#chatgpt-mermaid-_r_3lt_ .edge-pattern-dotted{stroke-dasharray:2;}#chatgpt-mermaid-_r_3lt_ .marker{fill:rgb(143, 143, 143);stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_3lt_ .marker.cross{stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_3lt_ svg{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;}#chatgpt-mermaid-_r_3lt_ p{margin:0;}#chatgpt-mermaid-_r_3lt_ .label{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_3lt_ .cluster-label text{fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_3lt_ .cluster-label span{color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_3lt_ .cluster-label span p{background-color:transparent;}#chatgpt-mermaid-_r_3lt_ .label text,#chatgpt-mermaid-_r_3lt_ span{fill:rgb(13, 13, 13);color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_3lt_ .node rect,#chatgpt-mermaid-_r_3lt_ .node circle,#chatgpt-mermaid-_r_3lt_ .node ellipse,#chatgpt-mermaid-_r_3lt_ .node polygon,#chatgpt-mermaid-_r_3lt_ .node path{fill:rgb(222, 234, 251);stroke:rgb(83, 154, 248);stroke-width:1px;}#chatgpt-mermaid-_r_3lt_ .rough-node .label text,#chatgpt-mermaid-_r_3lt_ .node .label text,#chatgpt-mermaid-_r_3lt_ .image-shape .label,#chatgpt-mermaid-_r_3lt_ .icon-shape .label{text-anchor:middle;}#chatgpt-mermaid-_r_3lt_ .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#chatgpt-mermaid-_r_3lt_ .rough-node .label,#chatgpt-mermaid-_r_3lt_ .node .label,#chatgpt-mermaid-_r_3lt_ .image-shape .label,#chatgpt-mermaid-_r_3lt_ .icon-shape .label{text-align:center;}#chatgpt-mermaid-_r_3lt_ .node.clickable{cursor:pointer;}#chatgpt-mermaid-_r_3lt_ .root .anchor path{fill:rgb(143, 143, 143)!important;stroke-width:0;stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_3lt_ .arrowheadPath{fill:rgb(143, 143, 143);}#chatgpt-mermaid-_r_3lt_ .edgePath .path{stroke:rgb(143, 143, 143);stroke-width:1px;}#chatgpt-mermaid-_r_3lt_ .flowchart-link{stroke:rgb(143, 143, 143);fill:none;}#chatgpt-mermaid-_r_3lt_ .edgeLabel{background-color:rgb(252, 252, 252);text-align:center;}#chatgpt-mermaid-_r_3lt_ .edgeLabel p{background-color:rgb(252, 252, 252);}#chatgpt-mermaid-_r_3lt_ .edgeLabel rect{opacity:0.5;background-color:rgb(252, 252, 252);fill:rgb(252, 252, 252);}#chatgpt-mermaid-_r_3lt_ .labelBkg{background-color:rgba(252, 252, 252, 0.5);}#chatgpt-mermaid-_r_3lt_ .cluster rect{fill:rgb(243, 243, 243);stroke:rgba(0, 0, 0, 0.1);stroke-width:1px;}#chatgpt-mermaid-_r_3lt_ .cluster text{fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_3lt_ .cluster span{color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_3lt_ div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:12px;background:rgb(243, 243, 243);border:1px solid rgba(0, 0, 0, 0.1);border-radius:2px;pointer-events:none;z-index:100;}#chatgpt-mermaid-_r_3lt_ .flowchartTitleText{text-anchor:middle;font-size:18px;fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_3lt_ rect.text{fill:none;stroke-width:0;}#chatgpt-mermaid-_r_3lt_ .icon-shape,#chatgpt-mermaid-_r_3lt_ .image-shape{background-color:rgb(252, 252, 252);text-align:center;}#chatgpt-mermaid-_r_3lt_ .icon-shape p,#chatgpt-mermaid-_r_3lt_ .image-shape p{background-color:rgb(252, 252, 252);padding:2px;}#chatgpt-mermaid-_r_3lt_ .icon-shape .label rect,#chatgpt-mermaid-_r_3lt_ .image-shape .label rect{opacity:0.5;background-color:rgb(252, 252, 252);fill:rgb(252, 252, 252);}#chatgpt-mermaid-_r_3lt_ .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#chatgpt-mermaid-_r_3lt_ .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#chatgpt-mermaid-_r_3lt_ .node .neo-node{stroke:rgb(83, 154, 248);}#chatgpt-mermaid-_r_3lt_ [data-look="neo"].node rect,#chatgpt-mermaid-_r_3lt_ [data-look="neo"].cluster rect,#chatgpt-mermaid-_r_3lt_ [data-look="neo"].node polygon{stroke:url(#chatgpt-mermaid-_r_3lt_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_3lt_ [data-look="neo"].swimlane.cluster rect{filter:none;}#chatgpt-mermaid-_r_3lt_ [data-look="neo"].node path{stroke:url(#chatgpt-mermaid-_r_3lt_-gradient);stroke-width:1px;}#chatgpt-mermaid-_r_3lt_ [data-look="neo"].node .outer-path{filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_3lt_ [data-look="neo"].node .neo-line path{stroke:rgb(83, 154, 248);filter:none;}#chatgpt-mermaid-_r_3lt_ [data-look="neo"].node circle{stroke:url(#chatgpt-mermaid-_r_3lt_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_3lt_ [data-look="neo"].node circle .state-start{fill:#000000;}#chatgpt-mermaid-_r_3lt_ [data-look="neo"].icon-shape .icon{fill:url(#chatgpt-mermaid-_r_3lt_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_3lt_ [data-look="neo"].icon-shape .icon-neo path{stroke:url(#chatgpt-mermaid-_r_3lt_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_3lt_ .node text{font-size:14px;font-weight:600;letter-spacing:normal;fill:rgb(65, 65, 65);}#chatgpt-mermaid-_r_3lt_ .edgeLabels text{font-size:13px;font-weight:600;letter-spacing:-0.08px;fill:rgb(65, 65, 65);}#chatgpt-mermaid-_r_3lt_ .node tspan[font-weight="normal"],#chatgpt-mermaid-_r_3lt_ .edgeLabels tspan[font-weight="normal"]{font-weight:600;}#chatgpt-mermaid-_r_3lt_ .edgeLabel .label rect{opacity:1;rx:13px;ry:13px;fill:rgb(252, 252, 252);stroke:rgb(219, 219, 219);stroke-width:1px;}#chatgpt-mermaid-_r_3lt_ .node rect,#chatgpt-mermaid-_r_3lt_ .node circle,#chatgpt-mermaid-_r_3lt_ .node ellipse,#chatgpt-mermaid-_r_3lt_ .node polygon,#chatgpt-mermaid-_r_3lt_ .node path{fill:rgb(243, 243, 243);stroke:rgba(0, 0, 0, 0.1);stroke-width:1px;}#chatgpt-mermaid-_r_3lt_ .node rect{rx:16px;ry:16px;}#chatgpt-mermaid-_r_3lt_ .node.mermaid-decision .label-container{fill:rgb(249, 249, 249);stroke:rgb(219, 219, 219);stroke-dasharray:2,2;}#chatgpt-mermaid-_r_3lt_ .edgePaths .flowchart-link{stroke:rgb(143, 143, 143);stroke-width:1px;stroke-linecap:round;stroke-linejoin:round;}#chatgpt-mermaid-_r_3lt_ .marker{fill:rgb(143, 143, 143);stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_3lt_ :root{--mermaid-font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";}Macro1 / Macro2 / MicroCamerasImage AcquisitionQuality ControlImage Registration / Stitch /ROIUNet / SAM / OCR / FeatureExtractionHigh-Resolution VLMMeasured FeaturesVisual Reasoning / ToolOrchestrationReference RetrievalReference Images /SpecificationsStatistical Evidence ModelHierarchical Bayesian FusionAnomaly Detection / ExpertPolicyComponent Result / WatchReportExpert Review / FeedbackTraining / Evaluation Pipeline

整個系統可以拆成六個工程模組。

### Module 1 — Image Acquisition

假設一支手錶拍攝 40 張影像，包含：

|Camera|主要目的|假設影像尺寸|
|---|---|---|
|Macro1|錶面全貌、結構、較大零件|4512 × 4512|
|Macro2|局部雕刻、精細幾何|4512 × 4512|
|Micro|文字、表面紋理、微小細節|2856 × 2848|

同時保存：

- Camera ID。
    
- Capture Position。
    
- Illumination Settings。
    
- Focus Position / Autofocus Result。
    
- Watch Series / Family。
    
- Capture Timestamp。
    
- Image Registration Metadata。
    

為什麼要保存這些？

因為模型看到的細節受到成像條件影響。

同樣一個刻度，在不同光線、角度和對焦設定下，看起來可能完全不同。

不能把 Imaging Variation 誤認成 Manufacturing Defect。

### Module 2 — Image Quality Gate

在使用 VLM 前，先做影像品質驗證。

例如：

\[ Q(I)= \{q_{\text{focus}},q_{\text{blur}}, q_{\text{exposure}},q_{\text{glare}}\} \]

可以使用：

- Laplacian Variance：模糊度的初步指標。
    
- Tenengrad：梯度型清晰度指標。
    
- Saturation Ratio：是否過曝。
    
- Registration Residual：影像對齊誤差。
    
- ROI Coverage：目標是否完整出現在視野。
    

如果 12 點鐘刻度已經失焦，就不應要求 VLM 猜測它的細微形狀。

應先觸發重新拍攝，或在不可能補拍時標記為 insufficient evidence。

## 17. Step-by-step：分析一張 Rolex 錶面

假設拍到一支標記為 Rolex Submariner 16613T 的手錶。

注意：真正可比較的參考資料，必須依照已驗證的型號、年份區間、零件版本與維修替換資訊建立，不能認為所有同型號的原廠錶面完全一樣。

![Buy Rolex Submariner Steel and Gold (two tone) 16613 | SwissWatchExpo](https://images.openai.com/static-rsc-4/HCLd4EP_UbkKc3BmaciLes0D8KJb7g0jZ1cTw4LfmNo9dWtaGbQUDhAzu6E_KYvWFemTu_arFNMVN3AO-y8b5AtLxS30ebOTvoTZf4D1PU-jlTUBIE5dg8ZXOlDwbsEzVWsbd2DN8rmuyXJNJH7NNv5kwl_IYl2WXQ59SYtDfxs?purpose=inline)

![Rolex Submariner Date Nos Stainless Steel and Yellow Gold Blue Dial 16613 at 1stDibs | rolex 16613 for sale, pre owned rolex, rolex watches](https://images.openai.com/static-rsc-4/NrWoEFhn4xZmbhVkDQVyEzdyvA_-oa403hnWmqFKo6Cv68nkEg3ROnsroWahQARYO7XLHFMBhWChG9rA3jWAOqDVSn_d1YbBQjVi3Q8iCLxK86hfpTt9Uj360_ttO1eurmilAmobRSdWpynUu3eyseDKLSqdtvuldNkJaGBHbAM?purpose=inline)

示意影像：全錶面、文字區域與 12 點鐘刻度。實際判定必須使用同型號、適用年份且已確認來源的參考資料。

### Step A — 進行 Segmentation

先使用既有的 UNet 或其他專用 Segmentation 模型。

輸出包括：

```
{
  "component": "Dial",
  "segmentation": {
    "dial": "mask_dial.png",
    "hour_markers": "mask_markers.png",
    "hands": "mask_hands.png",
    "dial_text": "mask_text.png"
  }
}
```

對於已經有大量精確標註的固定工業零件，專用 UNet 可能比開放式 segmentation foundation model 更有效率。

而 SAM 3 這類模型可用於輔助新類別的區域標註或開放詞彙物件定位。

### Step B — 擷取高解析度 ROI

假設 Micro image 大小為：

\[ 2856\times2848 \]

其中 `ROLEX` 文字位於：

```
{
  "image_id": "front_micro_0001.png",
  "roi": [1190, 420, 1630, 560],
  "coordinate_system": "original_pixels"
}
```

這只是示例座標。

ROI 尺寸為 440 × 140 pixels。

接著把原始 ROI 直接送到 VLM 或專用 OCR 模型，而不是先將整張原始影像縮得非常小。

對幾何分析，則保留未縮放的 pixel coordinates 與實體尺度校正。

### Step C — OCR + Geometry Features

假設辨識到：

`ROLEX`

對文字進一步計算：

- Character Bounding Box。
    
- Character Spacing。
    
- Stroke Width。
    
- Skeleton Length。
    
- Hu Moments。
    
- Contour Shape。
    
- Position Relative to Dial Center。
    

例如某個字元 `R`：

```
{
  "character": "R",
  "bbox": [1204, 438, 1252, 514],
  "features": {
    "normalized_stroke_width": 0.084,
    "skeleton_length": 173.6,
    "hu_moments": [
      0.193,
      0.0081,
      0.00042
    ]
  }
}
```

以上都是示意測量值，並非真實 Rolex 字體的標準值。

對每個 feature 建立 reference distribution，再分析與同類原廠樣本的差距。

## 18. VLM 在這個環節真正做什麼？

假設我們讓 VLM 接收：

- 整張 Dial Image。
    
- `ROLEX` 高解析度 ROI。
    
- 12 點鐘 Marker ROI。
    
- 同版本參考影像。
    
- OCR 結果。
    
- 幾何量測結果。
    

讓模型回答：

「哪些地方與參考資料有差異？每個差異對應哪張影像、哪個區域？是否需要進一步檢查？」

一個理想的結構化輸出可以是：

VLM Evidence Report

假設輸出

|   |   |
|---|---|
|Component|Dial|
|Region|ROLEX lettering|
|Observation|字母間距與選定參考影像略有不同|
|Grounding|Image ID + Bounding Box|
|Evidence|OCR、字距測量、參考版本|
|Recommended Action|確認版本、重新量測或專家複核|

注意，這個結果是異常候選證據，而不是最終真偽分類。

模型可以幫忙提出可驗證的疑點，但應由其他模組驗證它是否真的成立。

## 19. Agentic VLM：讓模型自己決定下一個分析工具

假設模型認為字體可能有異常，但無法確認。

可以設計以下 Tool API：

```
tools = [    crop_image,    run_ocr,    run_segmentation,    measure_character_geometry,    retrieve_reference_images,    compare_geometric_features,    request_additional_capture]
```

Orchestrator 可採用下列概念性程式：

```
def analyze_watch(image_set, metadata):    qc = validate_image_quality(image_set)    valid_images = qc.accepted_images    cv_features = extract_cv_features(valid_images)    evidence = []    for component in supported_components:        context = build_component_context(            component=component,            images=valid_images,            features=cv_features,            metadata=metadata,        )        references = retrieve_verified_references(context)        result = visual_agent.analyze(            context=context,            references=references,            tools=approved_tools,            max_tool_calls=5,        )        verified = verify_grounded_evidence(result)        evidence.extend(verified)    return authentication_engine.evaluate(        evidence=evidence,        metadata=metadata,    )
```

這是架構示意程式，不是可以直接執行的現成套件。

核心是：

VLM 能要求使用工具，但工具產生的結果仍然要經過獨立驗證。

例如 VLM 若要求移動實體相機進行另一個角度的拍攝，必須交給具有 movement limits、collision checking 和明確權限的硬體控制器處理。

## 20. Hybrid Statistical–Bayesian Authentication：完整數值例子

這一步把 VLM 的推理結果與可量化的 Computer Vision 特徵結合起來。

假設目標是判斷：

\[ H_G=\text{Dial is genuine} \]

\[ H_F=\text{Dial is forged} \]

這裡為了說明數學，暫時簡化為二元假設。真正產品必須處理替換件、後市場零件、修改及其他分類。

### 20.1 收集三組證據

這些數值全部是假設已經由參考樣本統計得到的示範資料，不是任何真實 Rolex 的判斷數據。

|Evidence Group|Observation|Likelihood Ratio \(P(E\mid G)/P(E\mid F)\)|
|---|---|---|
|Geometry|Hour marker geometry 高度吻合|8.0|
|Typography|字體輪廓與字距吻合|4.0|
|Surface|表面紋理存在可疑差異|0.25|

Likelihood Ratio 大於 1 表示這個證據較支持 \(H_G\)，小於 1 表示較支持 \(H_F\)。

假設先驗：

\[ P(H_G)=0.60 \]

因此 Prior Odds：

\[ O(H_G)=\frac{0.6}{0.4}=1.5 \]

為了展示計算，假設這三組證據在各假設條件下相互獨立。

\[ O(H_G\mid E)=1.5\times8\times4\times0.25 \]

\[ O(H_G\mid E)=12 \]

所以：

\[ P(H_G\mid E)=\frac{12}{1+12} \]

假設模型的 Bayesian Calculation

# 92.31%

在指定先驗、似然比與條件獨立假設下，得到的示範性後驗機率。不是實測準確率，也不是手錶的正式鑑定結果。

### 20.2 為什麼不能直接使用這個 92.31%？

因為 Typography、Geometry 和 Surface 可能具有相關性。

例如拍攝失焦可能同時影響：

- Character Stroke Width。
    
- Character Contour。
    
- OCR Confidence。
    
- Geometry Features。
    

若把高度相關的特徵當成完全獨立，便可能重複計算同一個證據。

生產系統應該考慮：

\[ P(E_1,E_2,E_3\mid H) \]

而不一定能簡化成：

\[ P(E_1\mid H)P(E_2\mid H)P(E_3\mid H) \]

可使用 Hierarchical Bayesian Model、Multivariate Density Estimation、Covariance Modeling，或者先把相關特徵整合為 evidence groups。

另外，類別先驗必須反映已定義的使用情境，不能將訓練資料集中各類別的比例直接視為真實市場先驗。

### 20.3 擴展到八種 Component 狀態

正式系統可能需要輸出：

- Original
    
- Authentic replacements
    
- Forgery
    
- Aftermarket
    
- Modified
    
- Incorrect Authentic
    
- Missing
    
- Not applicable
    

其中有一個非常重要的工程原則：

這八種標籤並非全部互斥的自然屬性，也不是同一種真假尺度。

例如 `Missing`、`Not applicable` 與 `Forgery` 在語義上完全不同。

比較合理的設計，是將 Component 狀態、證據完整度，以及 Watch-level 的最終政策分開。

即：

\[ P(C_i\mid E_i,S,F) \]

其中：

- \(C_i\)：Component \(i\) 的狀態。
    
- \(E_i\)：該 Component 的證據。
    
- \(S\)：Series。
    
- \(F\)：Family。
    

然後再使用 Expert Policy 把 Component 結果組合為 Watch-level 判斷，而不是把所有 Component 的分數平均。

## 21. 這套 VLM 應該如何訓練？

假設進入第一階段試點，已有 600 支實際手錶，每支約 40 張影像：

\[ 600\times40=24,000\text{ images} \]

這個資料量可以開始做 Domain Adaptation 試驗，但不能保證足以訓練穩定的八類真偽分類，特別是罕見仿品和不同年份零件。

### Training Dataset 設計

一筆訓練樣本應包含：

```
{
  "watch_id": "WATCH_000321",
  "series": "Series_A1",
  "family": "Submariner",
  "component": "Dial",
  "images": [
    "front_macro1.png",
    "front_micro1.png"
  ],
  "metadata": {
    "camera_id": "Micro",
    "focus_qc": "pass",
    "reference_version": "ref_set_004"
  },
  "expert_label": "Original",
  "evidence": [
    {
      "feature": "text_spacing",
      "image_id": "front_micro1.png",
      "bbox": [1190, 420, 1630, 560]
    }
  ]
}
```

### Dataset Split

不能直接將全部圖片隨機分成 Train / Validation / Test。

因為同一支手錶可能有 40 張高度相關的圖片。

如果其中 30 張進 Train、10 張進 Test，就會產生嚴重的 Data Leakage。

應以實體 Watch ID 為群組分割。

例如：

|Dataset|Watches|Images|
|---|---|---|
|Train 70%|420|約 16,800|
|Validation 15%|90|約 3,600|
|Test 15%|90|約 3,600|

這是示例分割。正式資料集還應另外保留跨年份、跨設備或新錶款的外部測試資料。

### Training Strategy

1. Phase A — Zero-shot Baseline
    
    先測試未經微調的 VLM 能否正確識別零件、閱讀文字和定位疑似異常區域。
    
2. Phase B — Domain SFT / LoRA
    
    訓練專業手錶術語、標註 Schema、Evidence Report、拒答機制。
    
3. Phase C — Vision Adaptation
    
    如果主要錯誤來自細微圖像特徵，再針對 Vision Encoder 或專用 Dense Feature Model 進行調整。
    
4. Phase D — Tool-using Reasoning
    
    訓練模型選擇 Crop、OCR、Segmentation 和 Reference Retrieval 等工具。
    
5. Phase E — Verifiable Post-training
    
    使用 Grounding、OCR 與其他可驗證的結果作為 reward，並特別測試模型的校準和拒答能力。
    
6. Phase F — Production Evaluation
    
    完整驗證各 Series、Family、Camera、Capture Condition 下的結果，再決定是否部署。
    

其中 Phase A 很重要。如果 Zero-shot VLM 本來就不能可靠讀取細微特徵，不應立即投入昂貴的完整 Fine-tuning，而應先檢查影像輸入策略和 Vision Encoder。

## 22. 如何評估整套系統是否能用在 Production？

假設要正式部署，至少建立以下 Evaluation Matrix。

|測試|要驗證什麼|參考 Metric|
|---|---|---|
|Dial Segmentation|刻度、指針、文字區域|Dice、mIoU|
|OCR|字元正確性|CER、Exact Match|
|Feature Extraction|幾何量測誤差|MAE、Repeatability|
|VLM Grounding|異常區域是否正確|IoU、Recall|
|Component Classification|八類元件狀態|Per-class Precision / Recall|
|Anomaly Detection|未知仿品、罕見缺陷|AUROC、Recall@Fixed FPR|
|Bayesian Calibration|分數與真實頻率是否一致|ECE、Brier Score|
|Agent Workflow|是否呼叫正確工具|Tool Success Rate|
|End-to-End|最終檢驗品質|False Accept / False Reject|
|Deployment|系統延遲與穩定性|P95 Latency、GPU Memory|

手錶真偽系統尤其應將 False Accept（將不符合要求的錶或零件錯誤通過）列為關鍵風險。

此外還需要分開評估：

- 已見過與未見過的 Series。
    
- 相同與不同的 Camera。
    
- 有無失焦。
    
- 有無過曝。
    
- 新的外觀版本。
    
- 真品但更換過原廠零件。
    
- 缺少部分影像。
    
- 不確定或無法判斷的案例。
    

不應只用單一 Overall Accuracy 宣稱整套系統已達商業鑑定等級。

## 23. 如何結合 Local DB、S3、AWS 和 Athena？

如果這套系統有大量高解析度影像，我會把即時判斷與離線訓練分開。

#chatgpt-mermaid-_r_3od_{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;fill:rgb(13, 13, 13);}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#chatgpt-mermaid-_r_3od_ .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-_r_3od_ .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-_r_3od_ .error-icon{fill:rgb(243, 243, 243);}#chatgpt-mermaid-_r_3od_ .error-text{fill:rgb(13, 13, 13);stroke:rgb(13, 13, 13);}#chatgpt-mermaid-_r_3od_ .edge-thickness-normal{stroke-width:1px;}#chatgpt-mermaid-_r_3od_ .edge-thickness-thick{stroke-width:3.5px;}#chatgpt-mermaid-_r_3od_ .edge-pattern-solid{stroke-dasharray:0;}#chatgpt-mermaid-_r_3od_ .edge-thickness-invisible{stroke-width:0;fill:none;}#chatgpt-mermaid-_r_3od_ .edge-pattern-dashed{stroke-dasharray:3;}#chatgpt-mermaid-_r_3od_ .edge-pattern-dotted{stroke-dasharray:2;}#chatgpt-mermaid-_r_3od_ .marker{fill:rgb(143, 143, 143);stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_3od_ .marker.cross{stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_3od_ svg{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;}#chatgpt-mermaid-_r_3od_ p{margin:0;}#chatgpt-mermaid-_r_3od_ .label{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_3od_ .cluster-label text{fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_3od_ .cluster-label span{color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_3od_ .cluster-label span p{background-color:transparent;}#chatgpt-mermaid-_r_3od_ .label text,#chatgpt-mermaid-_r_3od_ span{fill:rgb(13, 13, 13);color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_3od_ .node rect,#chatgpt-mermaid-_r_3od_ .node circle,#chatgpt-mermaid-_r_3od_ .node ellipse,#chatgpt-mermaid-_r_3od_ .node polygon,#chatgpt-mermaid-_r_3od_ .node path{fill:rgb(222, 234, 251);stroke:rgb(83, 154, 248);stroke-width:1px;}#chatgpt-mermaid-_r_3od_ .rough-node .label text,#chatgpt-mermaid-_r_3od_ .node .label text,#chatgpt-mermaid-_r_3od_ .image-shape .label,#chatgpt-mermaid-_r_3od_ .icon-shape .label{text-anchor:middle;}#chatgpt-mermaid-_r_3od_ .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#chatgpt-mermaid-_r_3od_ .rough-node .label,#chatgpt-mermaid-_r_3od_ .node .label,#chatgpt-mermaid-_r_3od_ .image-shape .label,#chatgpt-mermaid-_r_3od_ .icon-shape .label{text-align:center;}#chatgpt-mermaid-_r_3od_ .node.clickable{cursor:pointer;}#chatgpt-mermaid-_r_3od_ .root .anchor path{fill:rgb(143, 143, 143)!important;stroke-width:0;stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_3od_ .arrowheadPath{fill:rgb(143, 143, 143);}#chatgpt-mermaid-_r_3od_ .edgePath .path{stroke:rgb(143, 143, 143);stroke-width:1px;}#chatgpt-mermaid-_r_3od_ .flowchart-link{stroke:rgb(143, 143, 143);fill:none;}#chatgpt-mermaid-_r_3od_ .edgeLabel{background-color:rgb(252, 252, 252);text-align:center;}#chatgpt-mermaid-_r_3od_ .edgeLabel p{background-color:rgb(252, 252, 252);}#chatgpt-mermaid-_r_3od_ .edgeLabel rect{opacity:0.5;background-color:rgb(252, 252, 252);fill:rgb(252, 252, 252);}#chatgpt-mermaid-_r_3od_ .labelBkg{background-color:rgba(252, 252, 252, 0.5);}#chatgpt-mermaid-_r_3od_ .cluster rect{fill:rgb(243, 243, 243);stroke:rgba(0, 0, 0, 0.1);stroke-width:1px;}#chatgpt-mermaid-_r_3od_ .cluster text{fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_3od_ .cluster span{color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_3od_ div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:12px;background:rgb(243, 243, 243);border:1px solid rgba(0, 0, 0, 0.1);border-radius:2px;pointer-events:none;z-index:100;}#chatgpt-mermaid-_r_3od_ .flowchartTitleText{text-anchor:middle;font-size:18px;fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_3od_ rect.text{fill:none;stroke-width:0;}#chatgpt-mermaid-_r_3od_ .icon-shape,#chatgpt-mermaid-_r_3od_ .image-shape{background-color:rgb(252, 252, 252);text-align:center;}#chatgpt-mermaid-_r_3od_ .icon-shape p,#chatgpt-mermaid-_r_3od_ .image-shape p{background-color:rgb(252, 252, 252);padding:2px;}#chatgpt-mermaid-_r_3od_ .icon-shape .label rect,#chatgpt-mermaid-_r_3od_ .image-shape .label rect{opacity:0.5;background-color:rgb(252, 252, 252);fill:rgb(252, 252, 252);}#chatgpt-mermaid-_r_3od_ .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#chatgpt-mermaid-_r_3od_ .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#chatgpt-mermaid-_r_3od_ .node .neo-node{stroke:rgb(83, 154, 248);}#chatgpt-mermaid-_r_3od_ [data-look="neo"].node rect,#chatgpt-mermaid-_r_3od_ [data-look="neo"].cluster rect,#chatgpt-mermaid-_r_3od_ [data-look="neo"].node polygon{stroke:url(#chatgpt-mermaid-_r_3od_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_3od_ [data-look="neo"].swimlane.cluster rect{filter:none;}#chatgpt-mermaid-_r_3od_ [data-look="neo"].node path{stroke:url(#chatgpt-mermaid-_r_3od_-gradient);stroke-width:1px;}#chatgpt-mermaid-_r_3od_ [data-look="neo"].node .outer-path{filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_3od_ [data-look="neo"].node .neo-line path{stroke:rgb(83, 154, 248);filter:none;}#chatgpt-mermaid-_r_3od_ [data-look="neo"].node circle{stroke:url(#chatgpt-mermaid-_r_3od_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_3od_ [data-look="neo"].node circle .state-start{fill:#000000;}#chatgpt-mermaid-_r_3od_ [data-look="neo"].icon-shape .icon{fill:url(#chatgpt-mermaid-_r_3od_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_3od_ [data-look="neo"].icon-shape .icon-neo path{stroke:url(#chatgpt-mermaid-_r_3od_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_3od_ .node text{font-size:14px;font-weight:600;letter-spacing:normal;fill:rgb(65, 65, 65);}#chatgpt-mermaid-_r_3od_ .edgeLabels text{font-size:13px;font-weight:600;letter-spacing:-0.08px;fill:rgb(65, 65, 65);}#chatgpt-mermaid-_r_3od_ .node tspan[font-weight="normal"],#chatgpt-mermaid-_r_3od_ .edgeLabels tspan[font-weight="normal"]{font-weight:600;}#chatgpt-mermaid-_r_3od_ .edgeLabel .label rect{opacity:1;rx:13px;ry:13px;fill:rgb(252, 252, 252);stroke:rgb(219, 219, 219);stroke-width:1px;}#chatgpt-mermaid-_r_3od_ .node rect,#chatgpt-mermaid-_r_3od_ .node circle,#chatgpt-mermaid-_r_3od_ .node ellipse,#chatgpt-mermaid-_r_3od_ .node polygon,#chatgpt-mermaid-_r_3od_ .node path{fill:rgb(243, 243, 243);stroke:rgba(0, 0, 0, 0.1);stroke-width:1px;}#chatgpt-mermaid-_r_3od_ .node rect{rx:16px;ry:16px;}#chatgpt-mermaid-_r_3od_ .node.mermaid-decision .label-container{fill:rgb(249, 249, 249);stroke:rgb(219, 219, 219);stroke-dasharray:2,2;}#chatgpt-mermaid-_r_3od_ .edgePaths .flowchart-link{stroke:rgb(143, 143, 143);stroke-width:1px;stroke-linecap:round;stroke-linejoin:round;}#chatgpt-mermaid-_r_3od_ .marker{fill:rgb(143, 143, 143);stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_3od_ :root{--mermaid-font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";}On-Premise CamerasLocal ProcessingUNet / Feature ExtractionLocal VLM InferenceAuthentication EngineLocal Reference DBOperator UI / ReportAWS S3: Raw Images / ArtifactsAWS DB: Results / MetadataParquet / Glue / AthenaTraining / Reference BuildModel Registry / ValidationApproved Model Deployment

建議的職責如下：

|模組|職責|
|---|---|
|Local CV / VLM|即時影像分析、ROI、Evidence Report|
|Local Reference DB|Production 使用的已核准參考資料|
|Local Bayesian Engine|真偽證據融合|
|S3|原始圖片、特徵、模型、訓練資料|
|AWS DB|Watch、Capture、Analysis 的 Metadata|
|Athena|歷史資料分析、Coverage、Dataset Selection|
|AWS Training Pipeline|模型訓練、評估、版本管理|
|Model Registry|保存已通過驗證的模型版本|

Athena 不需要參與每一次即時推論。

如此一來，即使遠端網路不穩定，Local Authentication 仍可在事先部署的模型與 Reference DB 上運作。

# Part IV — 2027 年以後，Multimodal AI 可能往哪裡發展？

以下是基於 2025–2026 研究成果的技術趨勢判斷，而不是已經確定會實現的產品功能。

1. 更深入的 Multimodal Reasoning

AI 將不只根據單張圖回答，而是結合多張圖片、影片、外部工具和可驗證的計算結果。重點會從「生成合理解釋」轉向「生成能被檢查的證據」。

2. 兼具全域理解與細節精度的 Vision Foundation Model

以更有效率的方式保存 high-resolution 和 dense features，降低小型物件、微小文字和局部缺陷的資訊損失。

3. Real-time Streaming Multimodal AI

將 Video、Audio、Sensor Data 與持續更新的記憶整合，進行長時間事件追蹤、製程監控及即時互動。

4. Spatial Intelligence + World Models

模型將更重視三維幾何、遮擋、動作、物理變化與時間關係，成為機器人和自主系統的重要研究基礎。

5. 更有效率的 On-device / Edge Multimodal AI

透過量化、蒸餾、Sparse Compute、MoE 和專用模型路由，讓更多視覺推論能在有限的 GPU 資源下完成。

6. Reliable / Verifiable Multimodal AI

對工業和高風險應用而言，Calibration、Out-of-distribution Detection、Grounding Verification、Abstention 和 Human Review 會越來越重要。

# Part V — Senior AI Engineer 應優先學習哪些技能？

如果已經熟悉傳統 Computer Vision、Python、PyTorch、Image Processing 和模型部署，我建議下一步不是直接開始訓練自己的大型 VLM，而是依照以下順序建立能力。

|優先順序|技術|應具備的實際能力|
|---|---|---|
|1|Transformer / ViT / Attention|能解釋 Patch Embedding、Multi-head Attention、Position Encoding|
|2|Vision Foundation Models|能使用 SigLIP 2、DINOv3、SAM 3 做 Retrieval、Feature Extraction、Segmentation|
|3|VLM Architecture|能理解 Vision Encoder、Projector、Fusion、LLM Decoder|
|4|High-Resolution VLM|能設計 ROI、Tiling、Dynamic Resolution、Token Budget|
|5|Multimodal Fine-tuning|能建立 Dataset，實作 LoRA / QLoRA 與可靠評估|
|6|Multimodal RAG|能整合 Image Embeddings、Metadata Search、Reference Images|
|7|Agentic VLM|能實作 Crop、OCR、Segmentation、Retrieval 的 Tool Calling|
|8|Multimodal Reasoning / RL|理解 Verifiable Rewards、GRPO、Preference Optimization|
|9|Evaluation / Calibration|能測試 Hallucination、Grounding、Domain Shift、False Accept|
|10|Edge / Production Deployment|能最佳化 GPU、Latency、Batching、Observability 和模型版本管理|

### 我認為最值得投入的三個交叉方向

Computer Vision + VLM + Visual Reasoning

特別適合工業影像、精密製造、醫學影像研究、半導體檢測等需要細粒度特徵與跨影像推理的領域。

Multimodal AI + Agentic Systems

適合開發能夠呼叫影像工具、分析檢測結果、自動生成報告並完成專業流程的 AI 系統。

Computer Vision + Spatial AI + Robotics / VLA

適合未來的視覺導引機器人、自動化實驗設備、實體世界 AI Agent 與智能製造。

## 最後總結

對 2026 年的 Senior AI Engineer，應該建立的關鍵認知是：

VLM 不等於 Computer Vision 的替代品，而是讓 Computer Vision、Language、Reasoning、Knowledge Retrieval 和 Agentic Automation 更容易組合成完整系統的重要核心技術。

對高解析度工業影像尤其如此。很多任務真正困難的地方不是「能不能寫出一段合理的解釋」，而是能否保留極細小的視覺特徵、量化誤差、驗證證據，以及在模型不確定時避免做出錯誤判斷。

從長期技術能力來看，能同時掌握 Fine-grained Computer Vision、VLM、Multimodal Reasoning、Model Evaluation 與 Production AI Systems 的工程師，將能處理比單純影像分類或單純 LLM 應用更複雜的問題。