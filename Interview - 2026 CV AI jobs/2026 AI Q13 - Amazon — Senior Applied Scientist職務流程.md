
|                                                |     |
| ---------------------------------------------- | --- |
| [[#### Amazon — Senior Applied Scientist職務流程]] |     |
|                                                |     |
|                                                |     |

#### Amazon — Senior Applied Scientist職務流程
```
請深入詳細解釋譬如Amazon — Senior Applied Scientist, AI Lab position, 以下的每一步驟是甚麼怎麼具體實現:
需要 LLM、Diffusion、Agents、Reinforcement Learning，以及大規模模型訓練和研究能力。應徵 Senior Applied AI Engineer，不一定需要自己從頭 Pretrain 一個 LLM；但應徵 Foundation Model Research Engineer，就不能只會呼叫 API
```

# Amazon — Senior Applied Scientist, AI Lab：完整技術解析與實際研究開發流程

這個職位與一般的 Senior AI Engineer 有相當大的差異。

Senior Applied AI Engineer 的核心任務，通常是運用現有 AI 模型，建立可靠、可擴展且能真正部署的 AI 系統；Senior Applied Scientist 的核心任務，則是研究如何改進 AI 模型、訓練方法與系統能力，並將研究成果轉化成實際產品。

而 Foundation Model Research Engineer 更可能直接參與模型架構、Pretraining、Post-training 和大規模分散式訓練的核心工作。

這次我會以 Amazon 真實職缺為基礎，詳細說明：

1. Amazon Senior Applied Scientist 到底在研究、開發什麼。
    
2. LLM：如何從模型架構、資料準備、訓練到提升 Reasoning 能力。
    
3. Diffusion / Flow Models：如何訓練生成式影像模型，以及與 LLM 的差異。
    
4. Agents：如何設計具有 Tool Calling、多步驟規劃及自主執行能力的系統。
    
5. Reinforcement Learning：如何設計 Reward Function、訓練環境，並用 PPO、GRPO 等演算法改進模型。
    
6. Large-scale Training：如何使用 PyTorch、GPU Cluster、FSDP、Tensor Parallelism 等技術，實際訓練數十億參數的模型。
    
7. Research：如何提出假設、設計實驗、比較 Baseline、完成 Ablation Study，並形成能發表的研究成果。
    
8. 三種職位的技術門檻，以及應徵時需要展現什麼實作能力。
    

## 一、先確認 Amazon 2026 年真實職缺要求

我查到與你的描述幾乎完全一致的 Amazon 職缺：

Amazon — Senior Applied Scientist, ASCS AI Lab Team

Job ID：10567981

[Amazon 官方職缺頁面](https://amazon.jobs/en-gb/jobs/10567981/senior-applied-scientist-ascs-ai-lab-team)

該職缺主要針對 Amazon Selection and Catalog Systems（商品選品與目錄系統），提出以下研究及工程要求：

|技術領域|Amazon 職缺中的工作內容|
|---|---|
|LLM|開發、訓練大型語言模型|
|Diffusion / Flow Models|研究與訓練生成式模型|
|Agentic AI|建立能在 Amazon Production 環境運作的 Autonomous Agents|
|Reinforcement Learning|研究 RL 演算法及其應用|
|Multimodal AI / CV|結合影像、文字和多模態資訊|
|Large-scale Training|讓模型處理跨語言、跨地區的數十億商品|
|Applied Research|提出新演算法、設計實驗、改進模型|
|Production|將研究成果整合進商業系統|
|Publications|發表論文、分享研究成果|

這不是單純呼叫 Bedrock API 的工作。職缺明確包含模型訓練、演算法研究與 Agent 部署。

![](https://www.google.com/s2/favicons?domain=https://amazon.jobs&sz=32)

Amazon.jobs

不過，必須注意：職缺列出多個研究方向，不代表每位 Senior Applied Scientist 每天都要同時訓練 LLM、Diffusion 和 RL 模型。 實際上通常會依照團隊與研究主題專精其中一部分。

另外，Amazon 的另一個職缺 Senior Applied Scientist, Real-Time Conversational AI, AGI，更直接要求候選人具有實際訓練 Large-scale Foundation Models 的經驗，明確指出不只是使用預訓練模型，還包含 Architecture Design、Pretraining、Reward Modeling、RL 和 Distributed Training。

![](https://www.google.com/s2/favicons?domain=https://amazon.jobs&sz=32)

Amazon.jobs

因此，必須把兩種工作區分：

Applied AI Engineering

已有模型 → 建立 RAG / Agents / AI Application → 測試、部署、監控與優化。

Applied AI Research

已有模型或演算法 → 提出改進假設 → 修改模型或訓練方法 → 實驗驗證 → 大規模訓練 → 部署。

Foundation Model Research

資料與模型架構 → Pretraining → Post-training → Alignment / RL → Scaling Research → 新模型能力。

三者最大的差別不是使用哪個程式語言或哪個框架，而是你負責創造、改進或整合 AI 能力的哪一層。

接下來以一個完整的 Amazon AI Lab 模擬研究專案，逐步拆解每項技術如何實際完成。

## 二、完整案例：Amazon AI Lab 開發「下一代智慧商品目錄 AI」

假設你剛加入 Amazon ASCS AI Lab，主管交給你一個研究專案：

> Design and train a multimodal foundation-model-based autonomous agent that can understand product information, identify catalog inconsistencies, generate accurate descriptions, and improve through reinforcement learning.

中文：

設計並訓練一套多模態 AI Agent，能理解商品文字與影像、發現錯誤商品資訊、自動補齊可驗證的屬性、產生商品描述，並透過 Reinforcement Learning 持續改善決策能力。

這是用來解釋技術的假設性 Amazon 研究專案，不是宣稱 Amazon 現有內部系統就是這樣實作。

### 2.1 為什麼這個問題需要 AI Research？

假設 Amazon 商品目錄有以下資料：

商品 A：無線耳機

模擬資料

![Home](https://images.openai.com/static-rsc-4/w3Q8WYSSrcyZJWQxVrHy5HUwTJnT4p-EmX04iaG3QVh2bV-z8nkkl-cIN_RDW6LZYKklQyyrKDhHA3vKjgfAb7SzV7b1pnu-DHG7DPuisdvm1nRi-py7I2IMi9Y1b7UgO4DKgRvno5miAJqA9cj34LjMniiW8TN6BxtZGICnU4w?purpose=inline)

商品原始資料

名稱：Wireless Headphones

品牌：Acme Audio

藍牙版本：5.3

電池續航：未知

顏色：Black

降噪：有爭議

需要 AI 解決的問題

- 圖片與描述是否一致？
    
- 商品是否支援 ANC 主動降噪？
    
- 能否從製造商文件找出電池續航？
    
- 能否自動撰寫可信的商品介紹？
    
- 當資料不足時，Agent 是否知道要停止並交給人工審核？
    

傳統 Rule-based System 很難處理跨語言、不同格式、不完整資料和複雜推理。

如果只使用 LLM API，也會遇到問題：模型可能把「Wireless Headphones」推斷成支援 ANC，或憑空產生「40 小時續航」。

這就是 Applied Scientist 必須研究的地方：

如何讓模型具有更可靠的 Multimodal Reasoning、Grounded Generation、Tool Use 和 Sequential Decision-Making 能力，而不只是把 Prompt 寫得更長？

### 2.2 完整架構

#chatgpt-mermaid-_r_akv_{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;fill:rgb(13, 13, 13);}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#chatgpt-mermaid-_r_akv_ .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-_r_akv_ .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-_r_akv_ .error-icon{fill:rgb(243, 243, 243);}#chatgpt-mermaid-_r_akv_ .error-text{fill:rgb(13, 13, 13);stroke:rgb(13, 13, 13);}#chatgpt-mermaid-_r_akv_ .edge-thickness-normal{stroke-width:1px;}#chatgpt-mermaid-_r_akv_ .edge-thickness-thick{stroke-width:3.5px;}#chatgpt-mermaid-_r_akv_ .edge-pattern-solid{stroke-dasharray:0;}#chatgpt-mermaid-_r_akv_ .edge-thickness-invisible{stroke-width:0;fill:none;}#chatgpt-mermaid-_r_akv_ .edge-pattern-dashed{stroke-dasharray:3;}#chatgpt-mermaid-_r_akv_ .edge-pattern-dotted{stroke-dasharray:2;}#chatgpt-mermaid-_r_akv_ .marker{fill:rgb(143, 143, 143);stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_akv_ .marker.cross{stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_akv_ svg{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;}#chatgpt-mermaid-_r_akv_ p{margin:0;}#chatgpt-mermaid-_r_akv_ .label{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_akv_ .cluster-label text{fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_akv_ .cluster-label span{color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_akv_ .cluster-label span p{background-color:transparent;}#chatgpt-mermaid-_r_akv_ .label text,#chatgpt-mermaid-_r_akv_ span{fill:rgb(13, 13, 13);color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_akv_ .node rect,#chatgpt-mermaid-_r_akv_ .node circle,#chatgpt-mermaid-_r_akv_ .node ellipse,#chatgpt-mermaid-_r_akv_ .node polygon,#chatgpt-mermaid-_r_akv_ .node path{fill:rgb(222, 234, 251);stroke:rgb(83, 154, 248);stroke-width:1px;}#chatgpt-mermaid-_r_akv_ .rough-node .label text,#chatgpt-mermaid-_r_akv_ .node .label text,#chatgpt-mermaid-_r_akv_ .image-shape .label,#chatgpt-mermaid-_r_akv_ .icon-shape .label{text-anchor:middle;}#chatgpt-mermaid-_r_akv_ .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#chatgpt-mermaid-_r_akv_ .rough-node .label,#chatgpt-mermaid-_r_akv_ .node .label,#chatgpt-mermaid-_r_akv_ .image-shape .label,#chatgpt-mermaid-_r_akv_ .icon-shape .label{text-align:center;}#chatgpt-mermaid-_r_akv_ .node.clickable{cursor:pointer;}#chatgpt-mermaid-_r_akv_ .root .anchor path{fill:rgb(143, 143, 143)!important;stroke-width:0;stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_akv_ .arrowheadPath{fill:rgb(143, 143, 143);}#chatgpt-mermaid-_r_akv_ .edgePath .path{stroke:rgb(143, 143, 143);stroke-width:1px;}#chatgpt-mermaid-_r_akv_ .flowchart-link{stroke:rgb(143, 143, 143);fill:none;}#chatgpt-mermaid-_r_akv_ .edgeLabel{background-color:rgb(252, 252, 252);text-align:center;}#chatgpt-mermaid-_r_akv_ .edgeLabel p{background-color:rgb(252, 252, 252);}#chatgpt-mermaid-_r_akv_ .edgeLabel rect{opacity:0.5;background-color:rgb(252, 252, 252);fill:rgb(252, 252, 252);}#chatgpt-mermaid-_r_akv_ .labelBkg{background-color:rgba(252, 252, 252, 0.5);}#chatgpt-mermaid-_r_akv_ .cluster rect{fill:rgb(243, 243, 243);stroke:rgba(0, 0, 0, 0.1);stroke-width:1px;}#chatgpt-mermaid-_r_akv_ .cluster text{fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_akv_ .cluster span{color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_akv_ div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:12px;background:rgb(243, 243, 243);border:1px solid rgba(0, 0, 0, 0.1);border-radius:2px;pointer-events:none;z-index:100;}#chatgpt-mermaid-_r_akv_ .flowchartTitleText{text-anchor:middle;font-size:18px;fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_akv_ rect.text{fill:none;stroke-width:0;}#chatgpt-mermaid-_r_akv_ .icon-shape,#chatgpt-mermaid-_r_akv_ .image-shape{background-color:rgb(252, 252, 252);text-align:center;}#chatgpt-mermaid-_r_akv_ .icon-shape p,#chatgpt-mermaid-_r_akv_ .image-shape p{background-color:rgb(252, 252, 252);padding:2px;}#chatgpt-mermaid-_r_akv_ .icon-shape .label rect,#chatgpt-mermaid-_r_akv_ .image-shape .label rect{opacity:0.5;background-color:rgb(252, 252, 252);fill:rgb(252, 252, 252);}#chatgpt-mermaid-_r_akv_ .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#chatgpt-mermaid-_r_akv_ .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#chatgpt-mermaid-_r_akv_ .node .neo-node{stroke:rgb(83, 154, 248);}#chatgpt-mermaid-_r_akv_ [data-look="neo"].node rect,#chatgpt-mermaid-_r_akv_ [data-look="neo"].cluster rect,#chatgpt-mermaid-_r_akv_ [data-look="neo"].node polygon{stroke:url(#chatgpt-mermaid-_r_akv_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_akv_ [data-look="neo"].swimlane.cluster rect{filter:none;}#chatgpt-mermaid-_r_akv_ [data-look="neo"].node path{stroke:url(#chatgpt-mermaid-_r_akv_-gradient);stroke-width:1px;}#chatgpt-mermaid-_r_akv_ [data-look="neo"].node .outer-path{filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_akv_ [data-look="neo"].node .neo-line path{stroke:rgb(83, 154, 248);filter:none;}#chatgpt-mermaid-_r_akv_ [data-look="neo"].node circle{stroke:url(#chatgpt-mermaid-_r_akv_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_akv_ [data-look="neo"].node circle .state-start{fill:#000000;}#chatgpt-mermaid-_r_akv_ [data-look="neo"].icon-shape .icon{fill:url(#chatgpt-mermaid-_r_akv_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_akv_ [data-look="neo"].icon-shape .icon-neo path{stroke:url(#chatgpt-mermaid-_r_akv_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_akv_ .node text{font-size:14px;font-weight:600;letter-spacing:normal;fill:rgb(65, 65, 65);}#chatgpt-mermaid-_r_akv_ .edgeLabels text{font-size:13px;font-weight:600;letter-spacing:-0.08px;fill:rgb(65, 65, 65);}#chatgpt-mermaid-_r_akv_ .node tspan[font-weight="normal"],#chatgpt-mermaid-_r_akv_ .edgeLabels tspan[font-weight="normal"]{font-weight:600;}#chatgpt-mermaid-_r_akv_ .edgeLabel .label rect{opacity:1;rx:13px;ry:13px;fill:rgb(252, 252, 252);stroke:rgb(219, 219, 219);stroke-width:1px;}#chatgpt-mermaid-_r_akv_ .node rect,#chatgpt-mermaid-_r_akv_ .node circle,#chatgpt-mermaid-_r_akv_ .node ellipse,#chatgpt-mermaid-_r_akv_ .node polygon,#chatgpt-mermaid-_r_akv_ .node path{fill:rgb(243, 243, 243);stroke:rgba(0, 0, 0, 0.1);stroke-width:1px;}#chatgpt-mermaid-_r_akv_ .node rect{rx:16px;ry:16px;}#chatgpt-mermaid-_r_akv_ .node.mermaid-decision .label-container{fill:rgb(249, 249, 249);stroke:rgb(219, 219, 219);stroke-dasharray:2,2;}#chatgpt-mermaid-_r_akv_ .edgePaths .flowchart-link{stroke:rgb(143, 143, 143);stroke-width:1px;stroke-linecap:round;stroke-linejoin:round;}#chatgpt-mermaid-_r_akv_ .marker{fill:rgb(143, 143, 143);stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_akv_ :root{--mermaid-font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";}商品資料：文字、影像、規格文件Data Cleaning / LabelingPretraining / ContinuedPretrainingSupervised Fine-TuningMultimodal LLMAgent Planning & Tool CallingCatalog / Search / ValidationToolsRL Training EnvironmentReward / Policy OptimizationDiffusion / Flow Model商品展示圖生成或編輯Evaluation & Safety GatesProduction DeploymentMonitoring / Human Feedback

這張圖描述了多條協作的工作流，並不是所有模型都必須從頭訓練。尤其 Diffusion 模型與商品事實判斷模型應有清楚的責任界線。

實際專案可以拆成九個階段：

|Phase|研究／工程工作|主要產出|
|---|---|---|
|0|Define Problem & Hypothesis|Research Proposal|
|1|Dataset & Ground Truth|Versioned Dataset|
|2|LLM / Multimodal Training|Model Checkpoints|
|3|Diffusion / Flow Research|Generative Model|
|4|Agent Architecture|Tool-using Agent|
|5|RL & Reward Modeling|RL-trained Policy|
|6|Distributed Training|Scalable Training Pipeline|
|7|Evaluation & Research|Experiment Results / Paper|
|8|Deployment & Monitoring|Production Model|

下面開始逐步說明。

## 三、Phase 0：Research Problem Definition——研究員如何開始一個專案？

這是 Senior Applied Scientist 與一般 AI Application Engineer 第一個明顯不同的地方。

### 3.1 不是先選模型，而是先建立研究假設

例如：

Observation（觀察）

一般 Multimodal LLM 可以識別商品圖片，但遇到相互矛盾的商品屬性時，容易產生不可靠的判斷。

Research Question（研究問題）

是否能透過 Verifiable Tool-based Reinforcement Learning，讓 Agent 更準確地驗證商品屬性，並降低 hallucination？

Hypothesis（研究假設）

相較於只做 SFT 的模型，加入具有可驗證獎勵的 Agent RL 訓練後，可以改善商品屬性判斷的正確率，同時減少未經證實的資訊輸出。

### 3.2 必須建立 Baseline

假設團隊先對 10,000 個獨立測試商品評估以下系統：

|Model|方法|Verified Attribute Accuracy|Unsupported Claim Rate|
|---|---|---|---|
|A|Rules + Retrieval|78%|4%|
|B|Multimodal LLM Zero-shot|84%|12%|
|C|Multimodal LLM + SFT|90%|7%|
|D|SFT + Agent Tools|93%|3%|
|E|SFT + Agent RL|96%|1.5%|

以上全部為教學用假設數字，並非 Amazon 真實研究結果。

研究員的工作並不是看到 Model E 的 Accuracy 最高就直接宣稱成功，而是要證明：

- 改善是否來自 RL，而不是使用更多資料或更好的 Retrieval？
    
- 是否花費更多 Inference Tokens 才換到 Accuracy？
    
- 在未見過的商品、品牌和語言上是否仍有提升？
    
- 是否只是學會利用評分器漏洞？
    
- 是否降低錯誤卻大量選擇「不知道」，導致 Coverage 下降？
    

因此還要同時測量：

\[ \text{Coverage}=\frac{\text{自動完成的商品數}}{\text{全部商品數}} \]

\[ \text{Automation Risk} =P(\text{錯誤}\mid\text{系統選擇自動完成}) \]

這才是 Scientific Research，而不只是 Implementation。

## 四、Phase 1：Dataset、Ground Truth 與 Data Curation

Foundation Model Research 的成敗很大程度取決於訓練資料。

### Step 1：收集資料

對 Amazon Catalog Research，可能使用的資料類型：

|Data|內容|用途|
|---|---|---|
|Product Text|商品名稱、描述、規格|Language Modeling|
|Product Images|正面、背面、側面與細節|Vision Encoder|
|Manufacturer Documents|官方產品文件|Grounded Verification|
|Product Relationships|SKU、品牌、產品系列|Entity Resolution|
|Human Labels|屬性標註、矛盾判斷|SFT / Evaluation|
|Verified Agent Trajectories|工具呼叫與執行紀錄|Agent Training|
|Human Preferences|好／壞答案比較|DPO / Reward Modeling|

實際使用前需要檢查授權、資料用途、個資、商業機密與資料來源可靠性。

### Step 2：清洗資料

假設有 1 億筆商品資料，不代表可以直接全部拿去 Pretrain。

需要處理：

- Duplicates：同一 SKU 不同商家反覆上傳相同內容。
    
- Near Duplicates：商品文字略有差異，但本質相同。
    
- Incorrect Labels：產品屬性標錯。
    
- Image/Text Mismatch：標題與影像不一致。
    
- Language Distribution：英文商品遠多於其他語言。
    
- Synthetic Data Contamination：其他 LLM 生成的錯誤描述再次進入訓練資料。
    

例如，兩筆相同耳機：

```
Record 001
Brand: Acme
Model: X200
ANC: Yes

Record 002
Brand: Acme
Model: X200
ANC: No
```

不應該讓模型自行猜測哪筆正確。

較好的流程是：

```
Conflict Detection
      ↓
Search Authoritative Source
      ↓
Resolve / Mark Unknown
      ↓
Create Verified Ground Truth
      ↓
Training Dataset
```

無法確認的資料可以標成 `unknown`，而不是硬湊一個答案。

### Step 3：防止 Data Leakage

這在研究面試非常常見。

如果 X200 黑色款進入 Training，而相同產品的銀色款進入 Test，模型可能只是記住產品規格。

所以要依照 Product Family、Model、近似重複內容等群組來切分資料，而不是隨機切分每筆 Record。

例如：

- Training：商品家族 A、B、C。
    
- Validation：商品家族 D。
    
- Test：商品家族 E、F。
    
- Temporal Holdout：在切分時間點以後才上市的新商品。
    

這可以檢查模型是否真的具備跨產品的 Generalization，而不是依靠記憶。

### Step 4：建立不同訓練目的的資料

同一份資料不能不加區分地全部塞給所有模型。

Pretraining Dataset

```
產品規格、文件、影像文字配對、大規模一般語料
```

SFT Dataset

```
{
  "instruction": "Verify whether product X200 supports ANC",
  "evidence": "Official specification: ANC supported",
  "response": {
    "anc": true,
    "source": "manufacturer_spec",
    "status": "verified"
  }
}
```

Preference Dataset

```
Prompt: Does product X200 support ANC?

Answer A:
Yes, according to verified specifications.

Answer B:
Yes, and it supports 60 hours of battery life.

Preferred: A
Reason: B includes an unsupported battery claim.
```

RL Training Environment

```
Goal: Verify ANC support

Actions:
- search_catalog()
- retrieve_manufacturer_spec()
- compare_product_images()
- submit_verified_attribute()
- escalate_to_human()

Reward:
Correct verification → positive
Incorrect unsupported claim → penalty
Unnecessary tool calls → small cost
```

這四種資料與環境，分別對應不同的 Training Objective。

## 五、Phase 2：LLM Research——從 Transformer 到模型訓練

### 5.1 先理解 LLM Architecture

典型 Decoder-only LLM 可以簡化為：

#chatgpt-mermaid-_r_amd_{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;fill:rgb(13, 13, 13);}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#chatgpt-mermaid-_r_amd_ .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-_r_amd_ .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-_r_amd_ .error-icon{fill:rgb(243, 243, 243);}#chatgpt-mermaid-_r_amd_ .error-text{fill:rgb(13, 13, 13);stroke:rgb(13, 13, 13);}#chatgpt-mermaid-_r_amd_ .edge-thickness-normal{stroke-width:1px;}#chatgpt-mermaid-_r_amd_ .edge-thickness-thick{stroke-width:3.5px;}#chatgpt-mermaid-_r_amd_ .edge-pattern-solid{stroke-dasharray:0;}#chatgpt-mermaid-_r_amd_ .edge-thickness-invisible{stroke-width:0;fill:none;}#chatgpt-mermaid-_r_amd_ .edge-pattern-dashed{stroke-dasharray:3;}#chatgpt-mermaid-_r_amd_ .edge-pattern-dotted{stroke-dasharray:2;}#chatgpt-mermaid-_r_amd_ .marker{fill:rgb(143, 143, 143);stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_amd_ .marker.cross{stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_amd_ svg{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;}#chatgpt-mermaid-_r_amd_ p{margin:0;}#chatgpt-mermaid-_r_amd_ .label{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_amd_ .cluster-label text{fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_amd_ .cluster-label span{color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_amd_ .cluster-label span p{background-color:transparent;}#chatgpt-mermaid-_r_amd_ .label text,#chatgpt-mermaid-_r_amd_ span{fill:rgb(13, 13, 13);color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_amd_ .node rect,#chatgpt-mermaid-_r_amd_ .node circle,#chatgpt-mermaid-_r_amd_ .node ellipse,#chatgpt-mermaid-_r_amd_ .node polygon,#chatgpt-mermaid-_r_amd_ .node path{fill:rgb(222, 234, 251);stroke:rgb(83, 154, 248);stroke-width:1px;}#chatgpt-mermaid-_r_amd_ .rough-node .label text,#chatgpt-mermaid-_r_amd_ .node .label text,#chatgpt-mermaid-_r_amd_ .image-shape .label,#chatgpt-mermaid-_r_amd_ .icon-shape .label{text-anchor:middle;}#chatgpt-mermaid-_r_amd_ .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#chatgpt-mermaid-_r_amd_ .rough-node .label,#chatgpt-mermaid-_r_amd_ .node .label,#chatgpt-mermaid-_r_amd_ .image-shape .label,#chatgpt-mermaid-_r_amd_ .icon-shape .label{text-align:center;}#chatgpt-mermaid-_r_amd_ .node.clickable{cursor:pointer;}#chatgpt-mermaid-_r_amd_ .root .anchor path{fill:rgb(143, 143, 143)!important;stroke-width:0;stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_amd_ .arrowheadPath{fill:rgb(143, 143, 143);}#chatgpt-mermaid-_r_amd_ .edgePath .path{stroke:rgb(143, 143, 143);stroke-width:1px;}#chatgpt-mermaid-_r_amd_ .flowchart-link{stroke:rgb(143, 143, 143);fill:none;}#chatgpt-mermaid-_r_amd_ .edgeLabel{background-color:rgb(252, 252, 252);text-align:center;}#chatgpt-mermaid-_r_amd_ .edgeLabel p{background-color:rgb(252, 252, 252);}#chatgpt-mermaid-_r_amd_ .edgeLabel rect{opacity:0.5;background-color:rgb(252, 252, 252);fill:rgb(252, 252, 252);}#chatgpt-mermaid-_r_amd_ .labelBkg{background-color:rgba(252, 252, 252, 0.5);}#chatgpt-mermaid-_r_amd_ .cluster rect{fill:rgb(243, 243, 243);stroke:rgba(0, 0, 0, 0.1);stroke-width:1px;}#chatgpt-mermaid-_r_amd_ .cluster text{fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_amd_ .cluster span{color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_amd_ div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:12px;background:rgb(243, 243, 243);border:1px solid rgba(0, 0, 0, 0.1);border-radius:2px;pointer-events:none;z-index:100;}#chatgpt-mermaid-_r_amd_ .flowchartTitleText{text-anchor:middle;font-size:18px;fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_amd_ rect.text{fill:none;stroke-width:0;}#chatgpt-mermaid-_r_amd_ .icon-shape,#chatgpt-mermaid-_r_amd_ .image-shape{background-color:rgb(252, 252, 252);text-align:center;}#chatgpt-mermaid-_r_amd_ .icon-shape p,#chatgpt-mermaid-_r_amd_ .image-shape p{background-color:rgb(252, 252, 252);padding:2px;}#chatgpt-mermaid-_r_amd_ .icon-shape .label rect,#chatgpt-mermaid-_r_amd_ .image-shape .label rect{opacity:0.5;background-color:rgb(252, 252, 252);fill:rgb(252, 252, 252);}#chatgpt-mermaid-_r_amd_ .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#chatgpt-mermaid-_r_amd_ .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#chatgpt-mermaid-_r_amd_ .node .neo-node{stroke:rgb(83, 154, 248);}#chatgpt-mermaid-_r_amd_ [data-look="neo"].node rect,#chatgpt-mermaid-_r_amd_ [data-look="neo"].cluster rect,#chatgpt-mermaid-_r_amd_ [data-look="neo"].node polygon{stroke:url(#chatgpt-mermaid-_r_amd_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_amd_ [data-look="neo"].swimlane.cluster rect{filter:none;}#chatgpt-mermaid-_r_amd_ [data-look="neo"].node path{stroke:url(#chatgpt-mermaid-_r_amd_-gradient);stroke-width:1px;}#chatgpt-mermaid-_r_amd_ [data-look="neo"].node .outer-path{filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_amd_ [data-look="neo"].node .neo-line path{stroke:rgb(83, 154, 248);filter:none;}#chatgpt-mermaid-_r_amd_ [data-look="neo"].node circle{stroke:url(#chatgpt-mermaid-_r_amd_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_amd_ [data-look="neo"].node circle .state-start{fill:#000000;}#chatgpt-mermaid-_r_amd_ [data-look="neo"].icon-shape .icon{fill:url(#chatgpt-mermaid-_r_amd_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_amd_ [data-look="neo"].icon-shape .icon-neo path{stroke:url(#chatgpt-mermaid-_r_amd_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_amd_ .node text{font-size:14px;font-weight:600;letter-spacing:normal;fill:rgb(65, 65, 65);}#chatgpt-mermaid-_r_amd_ .edgeLabels text{font-size:13px;font-weight:600;letter-spacing:-0.08px;fill:rgb(65, 65, 65);}#chatgpt-mermaid-_r_amd_ .node tspan[font-weight="normal"],#chatgpt-mermaid-_r_amd_ .edgeLabels tspan[font-weight="normal"]{font-weight:600;}#chatgpt-mermaid-_r_amd_ .edgeLabel .label rect{opacity:1;rx:13px;ry:13px;fill:rgb(252, 252, 252);stroke:rgb(219, 219, 219);stroke-width:1px;}#chatgpt-mermaid-_r_amd_ .node rect,#chatgpt-mermaid-_r_amd_ .node circle,#chatgpt-mermaid-_r_amd_ .node ellipse,#chatgpt-mermaid-_r_amd_ .node polygon,#chatgpt-mermaid-_r_amd_ .node path{fill:rgb(243, 243, 243);stroke:rgba(0, 0, 0, 0.1);stroke-width:1px;}#chatgpt-mermaid-_r_amd_ .node rect{rx:16px;ry:16px;}#chatgpt-mermaid-_r_amd_ .node.mermaid-decision .label-container{fill:rgb(249, 249, 249);stroke:rgb(219, 219, 219);stroke-dasharray:2,2;}#chatgpt-mermaid-_r_amd_ .edgePaths .flowchart-link{stroke:rgb(143, 143, 143);stroke-width:1px;stroke-linecap:round;stroke-linejoin:round;}#chatgpt-mermaid-_r_amd_ .marker{fill:rgb(143, 143, 143);stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_amd_ :root{--mermaid-font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";}Input TextTokenizerToken IDsToken Embedding + PositionInformationTransformer Blocks × NFinal Normalization / LM HeadNext-token ProbabilitiesSampling or DecodingGenerated Token

例如：

```
Input:
This headphone supports active noise

Output:
cancellation
```

模型不是儲存一條固定的英文句子，而是學習依據前面 Token 的 Context 預測後續 Token 的分布。

### 5.2 Transformer 裡面實際在計算什麼？

核心是 Self-Attention：

\[ \operatorname{Attention}(Q,K,V) = \operatorname{softmax} \left( \frac{QK^T}{\sqrt{d_k}}+M \right)V \]

其中：

- Q（Query）：目前 Token 想取得什麼資訊。
    
- K（Key）：其他 Token 可以被搜尋的表示。
    
- V（Value）：其他 Token 所提供的資訊。
    
- M：例如 Causal Mask，防止模型在預測時偷看未來 Token。
    

假設輸入：

> The headphone is black, but the product title says white.

Attention 能學習連結 `black`、`white` 和 `product title`，形成對矛盾文字的內部表示。

但要注意，Attention 本身不會自動驗證世界上的事實。若模型需要知道商品實際顏色，還得利用可信資料或影像證據。

### 5.3 Pretraining：從大量資料訓練模型

在最典型的 Causal Language Modeling 中，損失函數為：

\[ \mathcal{L}_{LM} = -\sum_{t=1}^{T} \log P_\theta(x_t\mid x_{<t}) \]

也就是讓模型學會預測下一個 Token。

一個簡化的 PyTorch / Transformers Training Loop：

```
import torchfrom transformers import AutoModelForCausalLMmodel = AutoModelForCausalLM.from_config(config)model.train()optimizer = torch.optim.AdamW(    model.parameters(), lr=3e-4)for batch in train_loader:    input_ids = batch["input_ids"].to("cuda")    attention_mask = batch["attention_mask"].to("cuda")    optimizer.zero_grad()    outputs = model(        input_ids=input_ids,        attention_mask=attention_mask,        labels=input_ids    )    loss = outputs.loss    loss.backward()    torch.nn.utils.clip_grad_norm_(        model.parameters(), max_norm=1.0    )    optimizer.step()
```

這是概念性單 GPU 示範。實際執行還需要完整的 `config`、`train_loader`、device 配置與資料處理；Production Pretraining 還要補上 Scheduler、Mixed Precision、Distributed Training、Checkpoint、Resume 與 Logging 等。

Senior Applied Scientist 不一定需要自己手寫所有 Transformer Layers，但必須能說明：

為什麼這個 Loss 合理？模型為什麼不收斂？如何改進訓練策略？模型規模與資料量如何影響性能？

### 5.4 Multimodal LLM：文字與影像如何結合？

因為 Amazon Catalog 有 Product Images，僅使用文字 LLM 不夠。

一種常見 Multimodal Architecture 是：

典型 Vision Encoder + Language Model 融合架構（示意）

具體執行：

1. 使用 ViT（Vision Transformer）把商品圖片轉換成 Visual Features。
    
2. 使用 Tokenizer 把商品文字轉換成 Tokens。
    
3. 利用 Projection Layer 或 Connector 將 Visual Features 對齊 LLM 的 Hidden Dimension。
    
4. 將文字與影像表示送入 Multimodal Transformer。
    
5. 使用監督資料訓練模型輸出商品分類、屬性或判斷結果。
    

例如：

```
{
  "product_category": "Headphones",
  "color": "Black",
  "anc_support": "unknown",
  "evidence": [
    "product_image",
    "catalog_metadata"
  ],
  "requires_verification": true
}
```

Senior Scientist 可能研究的不是單純串接 Vision Encoder，而是：

- 應該使用哪種 Vision Token Compression？
    
- 高解析度影像怎麼避免 Token 數量過大？
    
- Cross-attention 是否比直接 Concatenation 更有效？
    
- 不同語言的商品名稱是否會影響影像理解？
    
- 如何讓模型區分看得見的視覺屬性與需要外部證據的技術規格？
    

### 5.5 SFT：將 Base Model 變成專業商品模型

Pretraining 學到語言與世界知識，但並不代表模型會按照業務要求輸出可信 JSON。

因此需要 Supervised Fine-Tuning（SFT）。

使用人工或經驗證的 Input / Output Pairs：

```
Input:
Determine the product color and explain
any uncertainty.

Expected Output:
Color: Black
Evidence: Product front image
Confidence: High
```

SFT 仍然主要使用 Token-level Cross-Entropy Loss，但訓練時通常只針對指定的目標 Response Tokens 計算 Loss，避免錯誤地學習某些 Prompt 區段。

研究員需要做的實驗：

|Experiment|要研究什麼|
|---|---|
|SFT 10k examples|少量資料是否足夠|
|SFT 100k examples|資料擴張是否有效|
|LoRA vs Full Fine-tuning|參數效率與品質|
|Text-only vs Multimodal|影像是否真正提供增益|
|Clean Labels vs Weak Labels|標註品質對結果的影響|
|3B vs 8B vs 30B|模型規模的效益與成本|

這就是 Model Research 與單純呼叫 LLM API 的差別：你需要自己控制訓練過程，分析模型能力為什麼改變。

## 六、Phase 3：Diffusion / Flow Models——如何研究與訓練生成式模型？

這是 Amazon ASCS AI Lab 職缺中特別值得注意的技術。

LLM 主要學習 Token Sequence 的機率分布；Diffusion Model 則常用於影像、影片、音訊等高維資料生成。

Amazon 自家的 Amazon Nova Canvas 就是 Text-conditioned Diffusion Model 的實際產品例子。

![](https://www.google.com/s2/favicons?domain=https://docs.aws.amazon.com&sz=32)

AWS AI Service Cards

### 6.1 Diffusion 的基本原理

假設你要訓練模型生成一張商品展示圖。

![AI Product Image Generator — Free Photo Maker | ListIQ](https://images.openai.com/static-rsc-4/xMU35Tb8p_mbsuK6vf5p7pnOVMX3xfQ3aDwDNwaNopys-4BXqqrvDHLupv09fcB5aLRIEcdLkw4H36K0p_xRmg0fS0mKLCtl_o6d8H966Lh9agXRZJwfeR-X03cFaZGhGHipFqaEuSYfddNVpgBo0uR7HCcKtMxPmILDn0na564?purpose=inline)

原始影像

![Audio headphones on a white background](https://images.openai.com/static-rsc-4/urg1RbZwjk-KTJ1NErWHYy_PX0-ZGQIenuRwCLCSRIWa51MEpQ-jz8DY-6YZGlxafxq8zMSidfHUDJo60iswHQjr9oYuwJLDVDvCSjy2mDFtv698xdaLgd163VoE0eLk4GG-Hw3diYTHI8lAYAZsshR-gdNJoMLoMG2H7FeyiAtgXooIlYDshbXH1HjuuAgt?purpose=inline)

加上雜訊

![The Best Headphones I’ve Tried (and Why I'd Buy Them All) | WIRED](https://images.openai.com/static-rsc-4/I3hZOnwyHRcCQ9TvRZBL9n4WhLb74NSzJrRs0QUfCkAvND_x3xgHP4Uk9NNSaUK4REScbLPY2cfMMiJLm2-lEuaOz0AUkoBE4Q_LWMlEPjzd17kYjcs3Jokgnrt2-xDyYyRQqg85GrNaGB8NlIl024kUnu9E6AS4L52z1QZZBdU?purpose=inline)

高度雜訊

![Stream White Noise TV Static [4 min loop] by Mattia Cenacchi | Listen online for free on SoundCloud](https://images.openai.com/static-rsc-4/W0obGpZSHiCCS5b_2k0HdDiFOa33bmpCy7f_8kmHO3cuoY2mxornZIfPqXrickzccjsRX3vjfieIeY9mr5n7FcL1Qnwm79gl4tIC6UsMVCxQB-SGnfc3hU-zUODCxutc_TBFiQ5VPxY36CEDkmDhCzqymw7mKbIUIaGh47GzYhE?purpose=inline)

接近純雜訊

示意圖；各張為概念圖，不是同一張照片的實際 Forward Diffusion 步驟。

訓練時，通常先把乾淨影像 \(x_0\) 加入隨機 Gaussian Noise：

\[ x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon \]

其中：

- \(x_0\)：原始乾淨影像。
    
- \(t\)：所抽樣的雜訊時間步。
    
- \(\epsilon\)：Gaussian Noise。
    
- \(\bar{\alpha}_t\)：由 Noise Schedule 決定。
    

模型的目標是根據 Noisy Image、Time Step 與文字條件，預測需要去除的雜訊。

典型 Noise Prediction Loss：

\[ \mathcal{L}_{diffusion} = \mathbb{E}_{x_0,t,\epsilon} \left[ \left\| \epsilon- \epsilon_\theta(x_t,t,c) \right\|^2 \right] \]

其中 \(c\) 是文字或其他 Conditioning。

這是 DDPM 類 Noise Prediction 訓練的一種形式；不同 Diffusion / Flow 模型可能使用不同的 Prediction Target。

### 6.2 具體怎麼用 PyTorch 訓練？

以下用 Hugging Face Diffusers 的概念展示核心步驟：

```
import torchimport torch.nn.functional as F# 假設：# vae, text_encoder, denoiser, noise_scheduler# 已經初始化並放置於適當裝置# train_loader 提供影像與文字 token# optimizer 負責更新 denoiserfor batch in train_loader:    # 1. Encode images into latent space    with torch.no_grad():        latents = vae.encode(            batch["pixel_values"]        ).latent_dist.sample()        latents = latents * vae.config.scaling_factor        text_features = text_encoder(            batch["input_ids"]        )[0]    # 2. Sample random Gaussian noise    noise = torch.randn_like(latents)    # 3. Sample diffusion timesteps    t = torch.randint(        0,        noise_scheduler.config.num_train_timesteps,        (latents.shape[0],),        device=latents.device    )    # 4. Add noise    noisy_latents = noise_scheduler.add_noise(        latents, noise, t
```

這是採用 Noise Prediction 的 Latent Diffusion 教學骨架，假設模型架構和 Scheduler 相容；若是 `v_prediction` 或 Flow Matching 模型，Loss Target 必須修改。正式訓練還需處理 Mixed Precision、Gradient Accumulation、Accelerate、Checkpoint 等。

Diffusers 官方也提供可修改的訓練腳本，涵蓋資料處理、加雜訊、預測、計算 Loss 與儲存模型。

![](https://www.google.com/s2/favicons?domain=https://huggingface.co&sz=32)

Hugging Face

### 6.3 Flow Matching 又是什麼？

職缺同時寫了 Diffusion and Flow Models，這不只是兩個不同名稱。

Flow Matching 的一種簡化形式，是讓模型學習把初始 Noise 分布連續轉換到 Data 分布的 Velocity Field。

假設：

\[ x_t=(1-t)x_{\text{noise}}+tx_{\text{data}} \]

對應的目標速度：

\[ u_t=x_{\text{data}}-x_{\text{noise}} \]

模型 \(v_\theta(x_t,t,c)\) 學習預測這個速度：

\[ \mathcal{L}_{FM} = \mathbb{E} \left[ \|v_\theta(x_t,t,c)-u_t\|^2 \right] \]

這是 Straight-line Conditional Flow Matching 的簡化例子。更一般的 Flow Matching 可以使用不同的 Probability Paths，而非只有直線插值。

![](https://www.google.com/s2/favicons?domain=https://arxiv.org&sz=32)

arXiv

直觀比較：

||Diffusion（典型 Noise Prediction）|Flow Matching|
|---|---|---|
|訓練目標|預測雜訊或其他 Denoising Target|預測 Velocity Field|
|生成過程|逐步 Denoising|沿學到的 Flow 積分|
|常見模型|DDPM、Latent Diffusion|Rectified Flow、Flow-based Generative Transformer|
|實際研究問題|Noise Schedule、Guidance、Sampling|Path Design、Solver、Sampling Efficiency|

兩者並非完全互斥：部分現代生成方法之間有密切的數學聯繫。

### 6.4 Amazon AI Lab 可能實際研究什麼？

對商品目錄而言，不能只是生成「好看的」耳機照片。

模型可能把兩個耳罩變成三個、改變 Logo，或生成不存在的按鈕。對 E-commerce，這會成為重大 Product Truthfulness 問題。

因此研究方向可能是：

Experiment A： 使用 Text-to-Image 生成一般生活情境商品圖。

Experiment B： 使用 Reference Image + Masked Editing，僅修改背景與光線。

Experiment C： 加入 Identity Preservation / Product Consistency Constraint，限制商品外觀變化。

實驗指標：

- Image Quality / Human Preference
    
- Text-Image Alignment
    
- Product Identity Preservation
    
- Geometry Consistency
    
- Unsupported Feature Rate
    
- Inference Latency / GPU Cost
    

例如，研究新 Loss：

\[ \mathcal{L} = \mathcal{L}_{generation} + \lambda \mathcal{L}_{identity} \]

其中 Identity Loss 可以由可信的商品影像特徵、區域對應或經驗證的屬性檢查建構，但不能單憑一般的 Image Embedding Similarity 就認定產品外觀完全正確。

這正是 Senior Scientist 的研究能力：找到現有模型在特定問題上的限制，設計可測量的改進方法，證明其有效。

## 七、Phase 4：Autonomous AI Agents——如何讓 LLM 真正自主完成任務？

Agent 不只是 LLM 多呼叫幾次 API。

它需要具有：

- State：目前有哪些已知事實。
    
- Policy：下一步應採取哪個 Action。
    
- Tools：可使用哪些外部能力。
    
- Memory：目前任務的歷史資訊。
    
- Environment：執行操作後會產生什麼回饋。
    
- Termination：什麼時候應該完成或停止。
    

### 7.1 用商品驗證 Agent 當例子

假設任務：

> Determine whether Acme X200 has Active Noise Cancellation, using verified evidence only.

### 一次 Agent Execution Trace

1. Understand Task
    
    LLM 確認需要驗證的是 ANC，不能只依商品名稱推論。
    
2. Tool Call：search_catalog()
    
    搜尋商品資料，得到兩個互相矛盾的欄位值。
    
3. Plan Next Action
    
    Agent 判斷目前 Evidence 不足，應查詢製造商規格。
    
4. Tool Call：retrieve_manufacturer_spec()
    
    找到官方規格：`ANC: Supported`。
    
5. Tool Call：validate_product_identity()
    
    確認官方規格所述的 Model Number 與目前 SKU 相符。
    
6. Submit Result
    
    { "anc": true, "verification": "passed", "source": "manufacturer_spec", "status": "verified" }
    

這樣 Agent 才算真正完成有證據的多步驟任務。

### 7.2 Agent Execution Loop 如何實作？

一個實務架構：

#chatgpt-mermaid-_r_api_{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;fill:rgb(13, 13, 13);}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#chatgpt-mermaid-_r_api_ .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-_r_api_ .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-_r_api_ .error-icon{fill:rgb(243, 243, 243);}#chatgpt-mermaid-_r_api_ .error-text{fill:rgb(13, 13, 13);stroke:rgb(13, 13, 13);}#chatgpt-mermaid-_r_api_ .edge-thickness-normal{stroke-width:1px;}#chatgpt-mermaid-_r_api_ .edge-thickness-thick{stroke-width:3.5px;}#chatgpt-mermaid-_r_api_ .edge-pattern-solid{stroke-dasharray:0;}#chatgpt-mermaid-_r_api_ .edge-thickness-invisible{stroke-width:0;fill:none;}#chatgpt-mermaid-_r_api_ .edge-pattern-dashed{stroke-dasharray:3;}#chatgpt-mermaid-_r_api_ .edge-pattern-dotted{stroke-dasharray:2;}#chatgpt-mermaid-_r_api_ .marker{fill:rgb(143, 143, 143);stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_api_ .marker.cross{stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_api_ svg{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;}#chatgpt-mermaid-_r_api_ p{margin:0;}#chatgpt-mermaid-_r_api_ .label{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_api_ .cluster-label text{fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_api_ .cluster-label span{color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_api_ .cluster-label span p{background-color:transparent;}#chatgpt-mermaid-_r_api_ .label text,#chatgpt-mermaid-_r_api_ span{fill:rgb(13, 13, 13);color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_api_ .node rect,#chatgpt-mermaid-_r_api_ .node circle,#chatgpt-mermaid-_r_api_ .node ellipse,#chatgpt-mermaid-_r_api_ .node polygon,#chatgpt-mermaid-_r_api_ .node path{fill:rgb(222, 234, 251);stroke:rgb(83, 154, 248);stroke-width:1px;}#chatgpt-mermaid-_r_api_ .rough-node .label text,#chatgpt-mermaid-_r_api_ .node .label text,#chatgpt-mermaid-_r_api_ .image-shape .label,#chatgpt-mermaid-_r_api_ .icon-shape .label{text-anchor:middle;}#chatgpt-mermaid-_r_api_ .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#chatgpt-mermaid-_r_api_ .rough-node .label,#chatgpt-mermaid-_r_api_ .node .label,#chatgpt-mermaid-_r_api_ .image-shape .label,#chatgpt-mermaid-_r_api_ .icon-shape .label{text-align:center;}#chatgpt-mermaid-_r_api_ .node.clickable{cursor:pointer;}#chatgpt-mermaid-_r_api_ .root .anchor path{fill:rgb(143, 143, 143)!important;stroke-width:0;stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_api_ .arrowheadPath{fill:rgb(143, 143, 143);}#chatgpt-mermaid-_r_api_ .edgePath .path{stroke:rgb(143, 143, 143);stroke-width:1px;}#chatgpt-mermaid-_r_api_ .flowchart-link{stroke:rgb(143, 143, 143);fill:none;}#chatgpt-mermaid-_r_api_ .edgeLabel{background-color:rgb(252, 252, 252);text-align:center;}#chatgpt-mermaid-_r_api_ .edgeLabel p{background-color:rgb(252, 252, 252);}#chatgpt-mermaid-_r_api_ .edgeLabel rect{opacity:0.5;background-color:rgb(252, 252, 252);fill:rgb(252, 252, 252);}#chatgpt-mermaid-_r_api_ .labelBkg{background-color:rgba(252, 252, 252, 0.5);}#chatgpt-mermaid-_r_api_ .cluster rect{fill:rgb(243, 243, 243);stroke:rgba(0, 0, 0, 0.1);stroke-width:1px;}#chatgpt-mermaid-_r_api_ .cluster text{fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_api_ .cluster span{color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_api_ div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:12px;background:rgb(243, 243, 243);border:1px solid rgba(0, 0, 0, 0.1);border-radius:2px;pointer-events:none;z-index:100;}#chatgpt-mermaid-_r_api_ .flowchartTitleText{text-anchor:middle;font-size:18px;fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_api_ rect.text{fill:none;stroke-width:0;}#chatgpt-mermaid-_r_api_ .icon-shape,#chatgpt-mermaid-_r_api_ .image-shape{background-color:rgb(252, 252, 252);text-align:center;}#chatgpt-mermaid-_r_api_ .icon-shape p,#chatgpt-mermaid-_r_api_ .image-shape p{background-color:rgb(252, 252, 252);padding:2px;}#chatgpt-mermaid-_r_api_ .icon-shape .label rect,#chatgpt-mermaid-_r_api_ .image-shape .label rect{opacity:0.5;background-color:rgb(252, 252, 252);fill:rgb(252, 252, 252);}#chatgpt-mermaid-_r_api_ .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#chatgpt-mermaid-_r_api_ .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#chatgpt-mermaid-_r_api_ .node .neo-node{stroke:rgb(83, 154, 248);}#chatgpt-mermaid-_r_api_ [data-look="neo"].node rect,#chatgpt-mermaid-_r_api_ [data-look="neo"].cluster rect,#chatgpt-mermaid-_r_api_ [data-look="neo"].node polygon{stroke:url(#chatgpt-mermaid-_r_api_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_api_ [data-look="neo"].swimlane.cluster rect{filter:none;}#chatgpt-mermaid-_r_api_ [data-look="neo"].node path{stroke:url(#chatgpt-mermaid-_r_api_-gradient);stroke-width:1px;}#chatgpt-mermaid-_r_api_ [data-look="neo"].node .outer-path{filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_api_ [data-look="neo"].node .neo-line path{stroke:rgb(83, 154, 248);filter:none;}#chatgpt-mermaid-_r_api_ [data-look="neo"].node circle{stroke:url(#chatgpt-mermaid-_r_api_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_api_ [data-look="neo"].node circle .state-start{fill:#000000;}#chatgpt-mermaid-_r_api_ [data-look="neo"].icon-shape .icon{fill:url(#chatgpt-mermaid-_r_api_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_api_ [data-look="neo"].icon-shape .icon-neo path{stroke:url(#chatgpt-mermaid-_r_api_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_api_ .node text{font-size:14px;font-weight:600;letter-spacing:normal;fill:rgb(65, 65, 65);}#chatgpt-mermaid-_r_api_ .edgeLabels text{font-size:13px;font-weight:600;letter-spacing:-0.08px;fill:rgb(65, 65, 65);}#chatgpt-mermaid-_r_api_ .node tspan[font-weight="normal"],#chatgpt-mermaid-_r_api_ .edgeLabels tspan[font-weight="normal"]{font-weight:600;}#chatgpt-mermaid-_r_api_ .edgeLabel .label rect{opacity:1;rx:13px;ry:13px;fill:rgb(252, 252, 252);stroke:rgb(219, 219, 219);stroke-width:1px;}#chatgpt-mermaid-_r_api_ .node rect,#chatgpt-mermaid-_r_api_ .node circle,#chatgpt-mermaid-_r_api_ .node ellipse,#chatgpt-mermaid-_r_api_ .node polygon,#chatgpt-mermaid-_r_api_ .node path{fill:rgb(243, 243, 243);stroke:rgba(0, 0, 0, 0.1);stroke-width:1px;}#chatgpt-mermaid-_r_api_ .node rect{rx:16px;ry:16px;}#chatgpt-mermaid-_r_api_ .node.mermaid-decision .label-container{fill:rgb(249, 249, 249);stroke:rgb(219, 219, 219);stroke-dasharray:2,2;}#chatgpt-mermaid-_r_api_ .edgePaths .flowchart-link{stroke:rgb(143, 143, 143);stroke-width:1px;stroke-linecap:round;stroke-linejoin:round;}#chatgpt-mermaid-_r_api_ .marker{fill:rgb(143, 143, 143);stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_api_ :root{--mermaid-font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";}Task + Current StateLLM PolicyChoose ActionRetrieval ToolVerification ToolHuman ReviewFinal ResultUpdate StateBudget / Safety / CompletionCheckIndependent Output ValidatorAccepted / RejectedSearchValidateEscalateFinishContinueStop

簡化的 Agent Controller：

```
def run_agent(task, policy, tools, max_steps=8):    state = {        "task": task,        "evidence": [],        "history": []    }    for step in range(max_steps):        action = policy.choose_action(state)        if action["type"] == "finish":            result = action["result"]            if validate_result(result, state):                return {                    "status": "completed",                    "result": result                }            return {                "status": "needs_review",                "reason": "unverified_result"            }        tool_name = action["tool"]        # Only allow explicitly registered tools        if tool_name not in tools:            return {"status": "blocked"}        observation = tools[tool_name](            **action["arguments"]        )        state["history"].append({
```

這是用來展示 Controller 邏輯的偽實作。完整 Production 版本還需要工具參數 Schema Validation、Permissions、Timeout、Error Handling、Evidence Provenance、Idempotency、Audit Logs 等。

### 7.3 Senior Applied Scientist 與 Agent Engineer 的差異

一般 Agent Engineer 可能選用現成 LLM，設計 Tools、Workflow 和 State Machine，讓系統能正常執行。

Senior Applied Scientist 則可能研究：

問題 1：Tool Selection Policy

為什麼模型會使用不必要的 Tools？能否訓練它選擇更有效率的 Action？

問題 2：Long-horizon Planning

當一個任務需要 10–20 步時，模型為什麼容易遺漏前面發現的矛盾？

問題 3：Error Recovery

Tool 回傳空結果，模型是應該重試、換工具，還是直接要求人工介入？

問題 4：Agent Generalization

在訓練時只有 20 種商品類型，是否能處理 200 種從未見過的商品？

問題 5：Learning from Interaction

能否讓模型透過任務成敗學會更好的 Tool-usage Policy，而不是全靠人類撰寫規則？

第五項，就直接連到 Reinforcement Learning。

## 八、Phase 5：Reinforcement Learning——如何真正訓練 Agent 變聰明？

這部分是近年 LLM Research 與 Agent Research 很重要的交集。

### 8.1 先解釋 RL 與 SFT 的本質差異

假設要訓練模型驗證商品。

SFT 的方式：

人類先準備正確答案或理想工具操作序列。

```
Input → Ideal Action Sequence
```

模型透過模仿這些 Examples 學習。

RL 的方式：

讓模型在受控環境中自己選擇 Actions，依照任務執行結果取得 Reward，再調整 Policy。

```
State → Action → Environment → Reward
                    ↓
                Next State
                    ↓
             Policy Improvement
```

不需要每一步都有唯一標準操作。例如查詢規格可以有多種正確工具順序。

### 8.2 將 Agent 問題定義成 MDP

Markov Decision Process 包含：

\[ (\mathcal S,\mathcal A,P,R,\gamma) \]

|元素|在商品 Agent 中的意義|
|---|---|
|State \(s_t\)|商品資料、目前證據、工具歷史|
|Action \(a_t\)|Search、Validate、Submit、Escalate|
|Transition \(P\)|執行工具後產生新的 State|
|Reward \(R\)|正確率、安全性、成本與任務成果|
|Discount \(\gamma\)|對未來 Reward 的重視程度|

在 LLM Agent 中，Policy 通常是模型所定義的 Token / Action Probability：

\[ \pi_\theta(a_t\mid s_t) \]

模型不一定直接輸出抽象的 `search` Action；也可以輸出結構化 Tool Call 所對應的 Tokens。

### 8.3 建立 RL Gym（Training Environment）

Amazon 的 AGI 職缺也明確提到設計 RL Gyms：讓模型在可驗證的任務環境中練習 Reasoning、Tool Use 和多步驟執行。

![](https://www.google.com/s2/favicons?domain=https://www.amazon.jobs&sz=32)

Amazon.jobs

假設某個 Episode 如下：

```
Initial State:
Product X200
ANC field = unknown

Action 1:
search_catalog()
Reward = 0

Action 2:
retrieve_manufacturer_spec()
Reward = 0

Action 3:
validate_model_number()
Reward = 0

Action 4:
submit(ANC=True, evidence=official_doc)
Terminal Reward = +1
```

另一個 Episode：

```
Action 1:
submit(ANC=True, evidence=None)

Terminal Reward = -1
```

第三種：

```
Action 1:
retrieve_manufacturer_spec()

Observation:
Source unavailable

Action 2:
escalate_to_human()

Terminal Reward = +0.4
```

上述 Reward 數值只是教學設定。

重要的是：Agent 應學到「沒有足夠資訊時不要猜」，也要避免讓模型認為「永遠交給人工」就是最佳策略。

### 8.4 Reward Function 如何設計？

一個教學用的 Composite Reward：

\[ R = 0.65R_{\text{correctness}} + 0.25R_{\text{evidence}} + 0.10R_{\text{completion}} - 0.02N_{\text{toolcalls}} \]

其中，前面幾項的值必須有明確定義、適當尺度與實際可驗證的來源。

例如：

```
def compute_reward(result, ground_truth, tool_calls):    if result["action"] == "escalate":        return 0.3    if result["action"] != "submit":        return -0.5    # Hard constraint: unsupported claims are invalid    if not result["evidence_verified"]:        return -1.0    correct = (        result["predicted_value"]        == ground_truth["verified_value"]    )    if not correct:        return -1.0    reward = 1.0    reward -= 0.02 * tool_calls    return reward
```

這個 Reward Function 是簡化版本，但展示了重要的設計原則：關鍵的安全／事實限制，可以設成 Hard Constraint，而不是只給很小的扣分。

完整 RL 環境還應支援 Ground Truth 不可判定的任務、Evidence Quality、Timeout、Action Validity 和不同商品的任務難度。

### 8.5 PPO 如何更新 LLM？

PPO（Proximal Policy Optimization）的重要概念，是限制每次 Policy Update 的幅度，避免訓練不穩定。

其常見 Clipped Objective：

\[ L^{CLIP}(\theta)= \mathbb{E}_t \left[ \min\left( r_t(\theta)A_t, \operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t \right) \right] \]

其中：

\[ r_t(\theta)= \frac{\pi_\theta(a_t\mid s_t)} {\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)} \]

- \(A_t\)：Advantage，衡量 Action 比預期好多少。
    
- \(r_t\)：新舊 Policy 對同一 Action 的機率比。
    
- \(\epsilon\)：控制更新幅度。
    

例如：

Agent 原本有 30% 機率會去查官方規格，70% 機率直接回答。

多次訓練後發現，先查規格能得到更高 Reward。

RL 便逐漸提高模型選擇這條路徑的機率。

當然，並非只靠這個單一機率就能完成完整 LLM RL 訓練；實際需要對整段 Rollout 的 Actions、Tokens 和 Reward 進行 Credit Assignment。

### 8.6 GRPO：為什麼 LLM Research 會用到？

GRPO（Group Relative Policy Optimization）的直觀概念，是針對同一個 Prompt 產生多個 Candidate Responses，使用它們之間的相對 Reward 來建立 Advantage，進而更新 Policy。

例如同一任務產生四種 Trajectories：

|Rollout|Agent 行為|Reward|
|---|---|---|
|A|直接猜 ANC = True|-1.0|
|B|查 Catalog 後直接回答|0.3|
|C|查官方規格並驗證型號|1.0|
|D|查三次重複資料後才完成|0.7|

模型可以學習 C 比其他行為更好。

一種簡化的 Group-relative Advantage：

\[ A_i= \frac{R_i-\operatorname{mean}(R)} {\operatorname{std}(R)+\delta} \]

這裡省略了正式 GRPO Objective 的 Probability Ratio、Clipping 和 KL 等細節。

GRPO 不必另外訓練傳統 PPO 所用的 Value / Critic Model，但仍需要考量 Rollout 成本、Reward 品質與訓練穩定性。

目前 Hugging Face TRL 提供 `GRPOTrainer`，並支援 Tool-based Agent Training 和 Stateful Environments。

![](https://www.google.com/s2/favicons?domain=https://huggingface.co&sz=32)

Hugging Face

### 8.7 真正實作 RL Agent Training 的步驟

1. 準備已有基本 Tool Calling 能力的 SFT Model。
    
2. 建立數千至數萬個可重設的任務環境，每個任務都具有獨立的 Ground Truth 或可靠評分機制。
    
3. 由 Policy Model 在環境中生成多條 Rollout Trajectories。
    
4. 執行真實或沙盒化的工具，儲存 Actions、Observations、Log Probabilities 和 Reward。
    
5. 使用 PPO、GRPO 或其他適合的 RL Algorithm 更新 Model Weights。
    
6. 在獨立的 Held-out Environments 測試 Task Success、Tool Efficiency、Generalization 和安全性。
    
7. 檢查 Reward Hacking、Training Collapse 和對舊任務能力的 Regression。
    

要特別理解：一般 Agent 執行時呼叫工具，不等於 RL Training。

如果執行後沒有將互動經驗用於 Policy Optimization，模型參數就不會因為這次經驗而自動改善。

### 8.8 RL 最大的研究難題之一：Reward Hacking

假設你的 Reward 設計成：

```
只要 submit() 回傳 status=success，就給 +1
```

模型可能學會跳過真正驗證，只是想辦法讓系統回傳 `success`。

所以必須使用獨立的 Ground Truth、可信的 Final-state Verification，並限制 Agent 對評分器和測試資料的存取。

常見研究方法包括：

- Hidden Test Environments
    
- Adversarial Tasks
    
- Counterfactual Evaluation
    
- Reward-model Auditing
    
- Held-out Tool Schemas
    
- Long-horizon Trajectory Analysis
    
- Human Evaluation
    

好的 RL 研究，不能只證明 Reward 上升；要證明實際 Task Success 和安全性也同步改善。

## 九、Phase 6：Large-scale Model Training——如何真正訓練幾十億參數的模型？

這是 Senior Applied Scientist 與 Foundation Model Research Engineer 最重要的技術門檻之一。

如果只會：

```
response = client.responses.create(...)
```

你可以做出很有價值的 AI Application，但你沒有直接處理 Model Weights、Backward Propagation、Gradient Synchronization、Optimizer States 或 Distributed Training 的經驗。

對 Foundation Model Research，這些都是核心能力。

### 9.1 為什麼需要 Distributed Training？

假設我們要訓練一個 7B Parameter LLM，也就是 70 億參數。

先估算 GPU Memory。

使用典型 BF16 + FP32 AdamW 訓練配置：

|記憶體項目|每參數記憶體|7B 模型|
|---|---|---|
|BF16 Model Weights|2 bytes|14 GB|
|Gradients|2 bytes|14 GB|
|FP32 Adam First Moment|4 bytes|28 GB|
|FP32 Adam Second Moment|4 bytes|28 GB|
|FP32 Master Weights（若使用）|4 bytes|28 GB|
|合計|16 bytes|112 GB|

這還沒有包含 Activation、Temporary Buffers、Communication Buffers 和其他 Training Overhead。

不同 Framework、Optimizer 和 Mixed Precision 實作的實際記憶體分配可能不同。

所以即使一張 GPU 有 80 GB VRAM，也不代表能直接使用普通的 Full-parameter AdamW Training 訓練這個 7B Model。

### 9.2 四種核心 Parallelism

### Parallel Training：資料與模型如何分配到 GPU？

1. Data Parallelism

相同模型，各 GPU 處理不同 Batch。

2. Tensor Parallelism

將單一 Layer 的計算分散到多張 GPU。

3. Pipeline Parallelism

不同 GPU 負責不同 Transformer Layers。

4. FSDP / ZeRO

切分參數、Gradient 和 Optimizer State，降低冗餘記憶體。

### 9.3 Data Parallelism（DDP）

假設 8 張 GPU：

```
GPU 0 → Batch 0 → Full Model
GPU 1 → Batch 1 → Full Model
GPU 2 → Batch 2 → Full Model
...
GPU 7 → Batch 7 → Full Model
```

每張 GPU 都執行自己的 Forward / Backward，然後透過 All-Reduce 同步 Gradients，使模型保持一致更新。

優點是容易理解和擴展資料吞吐量。

問題是每張 GPU 通常仍須保存完整的 Model Weights，以及相應的訓練狀態，因此大模型很容易超過 VRAM。

### 9.4 FSDP（Fully Sharded Data Parallelism）

FSDP 會把模型狀態分散儲存在不同 GPU。

例如：

```
GPU 0 → Parameter Shard 0
GPU 1 → Parameter Shard 1
GPU 2 → Parameter Shard 2
GPU 3 → Parameter Shard 3
```

執行某層 Forward 時，透過 All-Gather 取得必要的完整參數；完成後可重新 Shard。Backward 也會利用相應的 Gather 和 Reduce-Scatter 操作。

這樣能顯著減少每張 GPU 儲存完整模型狀態的需求，但增加 Communication Overhead。

PyTorch FSDP 官方文件也明確區分 Full Sharding 的 Parameters、Gradients 和 Optimizer States 行為。

![](https://www.google.com/s2/favicons?domain=https://docs.pytorch.org&sz=32)

PyTorch 2.14 documentation

### 9.5 Tensor Parallelism

假設 Transformer Layer 裡有大型 Linear Transformation：

\[ Y=XW \]

如果 Weight Matrix \(W\) 太大，可以將其切分給不同 GPU。

例如 Column Parallelism：

\[ W=[W_1,W_2,W_3,W_4] \]

四張 GPU 分別計算：

\[ Y_i=XW_i \]

再依模型結構進行必要的 Concatenation 或 Collective Communication。

此方法可解決 Layer 太大的問題，但 GPU 之間需要高頻寬、低延遲的通訊。

### 9.6 Pipeline Parallelism

假設模型有 48 層 Transformer：

```
GPU 0: Layers  1–12
GPU 1: Layers 13–24
GPU 2: Layers 25–36
GPU 3: Layers 37–48
```

一個 Microbatch 從 GPU 0 經過 GPU 1、2、3。

為減少 GPU 等待時間，可以安排多個 Microbatches 同時在不同 Pipeline Stages 工作。

這就產生了新的研究與工程問題：

- Pipeline Bubble
    
- Load Balancing
    
- Microbatch Scheduling
    
- Activation Memory
    
- Pipeline Communication
    

### 9.7 實際怎麼組合？

假設有 64 張 GPU，可能設定：

|Strategy|Degree|
|---|---|
|Tensor Parallelism|4|
|Pipeline Parallelism|2|
|Data Parallelism|8|
|Total GPUs|64|

因為：

\[ 4\times 2\times 8=64 \]

這是用來說明配置方式的例子，並非 Amazon 真實 Training Cluster 設定。

真實模型是否適用，仍取決於 Architecture、Model Size、Sequence Length、GPU Memory 和 Network Topology。

NVIDIA Megatron Core 文件介紹如何組合 Data、Tensor、Pipeline、Context 與 Expert Parallelism，並針對不同模型尺寸提出配置方向。

![](https://www.google.com/s2/favicons?domain=https://docs.nvidia.com&sz=32)

Megatron Core

### 9.8 Senior Scientist 還要理解 Training Scaling

設：

- \(N\)：模型參數數量。
    
- \(D\)：Training Tokens。
    
- \(C\)：訓練計算量。
    

對 Dense Transformer，一個常見的粗略估算：

\[ C\approx 6ND \]

例如：

\[ N=7\times10^9 \]

\[ D=10^{12} \]

則：

\[ C\approx4.2\times10^{22}\text{ FLOPs} \]

這是近似估計，並未完整反映 Attention、Context Length、實際 Hardware Efficiency 等因素。

Senior Scientist 要回答的問題不只是「可以用幾張 GPU 訓練」，而是：

在固定 Compute Budget 下，應該增加 Parameters、Data Tokens、Training Steps，還是改善資料品質？

這涉及 Scaling Laws、Compute-optimal Training、Data Mixture 與實驗設計。

### 9.9 一個真正的 AWS Large-scale Training Pipeline

假設要在 AWS 上進行 Full Fine-tuning 或大型模型訓練：

#chatgpt-mermaid-_r_au3_{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;fill:rgb(13, 13, 13);}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#chatgpt-mermaid-_r_au3_ .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-_r_au3_ .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-_r_au3_ .error-icon{fill:rgb(243, 243, 243);}#chatgpt-mermaid-_r_au3_ .error-text{fill:rgb(13, 13, 13);stroke:rgb(13, 13, 13);}#chatgpt-mermaid-_r_au3_ .edge-thickness-normal{stroke-width:1px;}#chatgpt-mermaid-_r_au3_ .edge-thickness-thick{stroke-width:3.5px;}#chatgpt-mermaid-_r_au3_ .edge-pattern-solid{stroke-dasharray:0;}#chatgpt-mermaid-_r_au3_ .edge-thickness-invisible{stroke-width:0;fill:none;}#chatgpt-mermaid-_r_au3_ .edge-pattern-dashed{stroke-dasharray:3;}#chatgpt-mermaid-_r_au3_ .edge-pattern-dotted{stroke-dasharray:2;}#chatgpt-mermaid-_r_au3_ .marker{fill:rgb(143, 143, 143);stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_au3_ .marker.cross{stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_au3_ svg{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;}#chatgpt-mermaid-_r_au3_ p{margin:0;}#chatgpt-mermaid-_r_au3_ .label{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_au3_ .cluster-label text{fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_au3_ .cluster-label span{color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_au3_ .cluster-label span p{background-color:transparent;}#chatgpt-mermaid-_r_au3_ .label text,#chatgpt-mermaid-_r_au3_ span{fill:rgb(13, 13, 13);color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_au3_ .node rect,#chatgpt-mermaid-_r_au3_ .node circle,#chatgpt-mermaid-_r_au3_ .node ellipse,#chatgpt-mermaid-_r_au3_ .node polygon,#chatgpt-mermaid-_r_au3_ .node path{fill:rgb(222, 234, 251);stroke:rgb(83, 154, 248);stroke-width:1px;}#chatgpt-mermaid-_r_au3_ .rough-node .label text,#chatgpt-mermaid-_r_au3_ .node .label text,#chatgpt-mermaid-_r_au3_ .image-shape .label,#chatgpt-mermaid-_r_au3_ .icon-shape .label{text-anchor:middle;}#chatgpt-mermaid-_r_au3_ .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#chatgpt-mermaid-_r_au3_ .rough-node .label,#chatgpt-mermaid-_r_au3_ .node .label,#chatgpt-mermaid-_r_au3_ .image-shape .label,#chatgpt-mermaid-_r_au3_ .icon-shape .label{text-align:center;}#chatgpt-mermaid-_r_au3_ .node.clickable{cursor:pointer;}#chatgpt-mermaid-_r_au3_ .root .anchor path{fill:rgb(143, 143, 143)!important;stroke-width:0;stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_au3_ .arrowheadPath{fill:rgb(143, 143, 143);}#chatgpt-mermaid-_r_au3_ .edgePath .path{stroke:rgb(143, 143, 143);stroke-width:1px;}#chatgpt-mermaid-_r_au3_ .flowchart-link{stroke:rgb(143, 143, 143);fill:none;}#chatgpt-mermaid-_r_au3_ .edgeLabel{background-color:rgb(252, 252, 252);text-align:center;}#chatgpt-mermaid-_r_au3_ .edgeLabel p{background-color:rgb(252, 252, 252);}#chatgpt-mermaid-_r_au3_ .edgeLabel rect{opacity:0.5;background-color:rgb(252, 252, 252);fill:rgb(252, 252, 252);}#chatgpt-mermaid-_r_au3_ .labelBkg{background-color:rgba(252, 252, 252, 0.5);}#chatgpt-mermaid-_r_au3_ .cluster rect{fill:rgb(243, 243, 243);stroke:rgba(0, 0, 0, 0.1);stroke-width:1px;}#chatgpt-mermaid-_r_au3_ .cluster text{fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_au3_ .cluster span{color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_au3_ div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:12px;background:rgb(243, 243, 243);border:1px solid rgba(0, 0, 0, 0.1);border-radius:2px;pointer-events:none;z-index:100;}#chatgpt-mermaid-_r_au3_ .flowchartTitleText{text-anchor:middle;font-size:18px;fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_au3_ rect.text{fill:none;stroke-width:0;}#chatgpt-mermaid-_r_au3_ .icon-shape,#chatgpt-mermaid-_r_au3_ .image-shape{background-color:rgb(252, 252, 252);text-align:center;}#chatgpt-mermaid-_r_au3_ .icon-shape p,#chatgpt-mermaid-_r_au3_ .image-shape p{background-color:rgb(252, 252, 252);padding:2px;}#chatgpt-mermaid-_r_au3_ .icon-shape .label rect,#chatgpt-mermaid-_r_au3_ .image-shape .label rect{opacity:0.5;background-color:rgb(252, 252, 252);fill:rgb(252, 252, 252);}#chatgpt-mermaid-_r_au3_ .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#chatgpt-mermaid-_r_au3_ .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#chatgpt-mermaid-_r_au3_ .node .neo-node{stroke:rgb(83, 154, 248);}#chatgpt-mermaid-_r_au3_ [data-look="neo"].node rect,#chatgpt-mermaid-_r_au3_ [data-look="neo"].cluster rect,#chatgpt-mermaid-_r_au3_ [data-look="neo"].node polygon{stroke:url(#chatgpt-mermaid-_r_au3_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_au3_ [data-look="neo"].swimlane.cluster rect{filter:none;}#chatgpt-mermaid-_r_au3_ [data-look="neo"].node path{stroke:url(#chatgpt-mermaid-_r_au3_-gradient);stroke-width:1px;}#chatgpt-mermaid-_r_au3_ [data-look="neo"].node .outer-path{filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_au3_ [data-look="neo"].node .neo-line path{stroke:rgb(83, 154, 248);filter:none;}#chatgpt-mermaid-_r_au3_ [data-look="neo"].node circle{stroke:url(#chatgpt-mermaid-_r_au3_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_au3_ [data-look="neo"].node circle .state-start{fill:#000000;}#chatgpt-mermaid-_r_au3_ [data-look="neo"].icon-shape .icon{fill:url(#chatgpt-mermaid-_r_au3_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_au3_ [data-look="neo"].icon-shape .icon-neo path{stroke:url(#chatgpt-mermaid-_r_au3_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_au3_ .node text{font-size:14px;font-weight:600;letter-spacing:normal;fill:rgb(65, 65, 65);}#chatgpt-mermaid-_r_au3_ .edgeLabels text{font-size:13px;font-weight:600;letter-spacing:-0.08px;fill:rgb(65, 65, 65);}#chatgpt-mermaid-_r_au3_ .node tspan[font-weight="normal"],#chatgpt-mermaid-_r_au3_ .edgeLabels tspan[font-weight="normal"]{font-weight:600;}#chatgpt-mermaid-_r_au3_ .edgeLabel .label rect{opacity:1;rx:13px;ry:13px;fill:rgb(252, 252, 252);stroke:rgb(219, 219, 219);stroke-width:1px;}#chatgpt-mermaid-_r_au3_ .node rect,#chatgpt-mermaid-_r_au3_ .node circle,#chatgpt-mermaid-_r_au3_ .node ellipse,#chatgpt-mermaid-_r_au3_ .node polygon,#chatgpt-mermaid-_r_au3_ .node path{fill:rgb(243, 243, 243);stroke:rgba(0, 0, 0, 0.1);stroke-width:1px;}#chatgpt-mermaid-_r_au3_ .node rect{rx:16px;ry:16px;}#chatgpt-mermaid-_r_au3_ .node.mermaid-decision .label-container{fill:rgb(249, 249, 249);stroke:rgb(219, 219, 219);stroke-dasharray:2,2;}#chatgpt-mermaid-_r_au3_ .edgePaths .flowchart-link{stroke:rgb(143, 143, 143);stroke-width:1px;stroke-linecap:round;stroke-linejoin:round;}#chatgpt-mermaid-_r_au3_ .marker{fill:rgb(143, 143, 143);stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_au3_ :root{--mermaid-font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";}Raw Data — S3ETL / Filtering / DeduplicationVersioned Training Shards —S3Training Job OrchestratorSageMaker / GPU ClusterPyTorch Distributed / FSDP /MegatronCheckpoint + Optimizer StateS3 Checkpoint StorageValidation / Model EvaluationRelease CriteriaModel Registry / StagingAnalysis / New ExperimentCanary DeploymentPassFail

AWS SageMaker AI 支援 Distributed Data Parallelism 與 Model Parallelism，也能使用 PyTorch DDP、`torchrun` 等常見訓練工具。

![](https://www.google.com/s2/favicons?domain=https://docs.aws.amazon.com&sz=32)

Amazon SageMaker AI

具體的執行任務：

Step 1：Prepare Training Data

把資料轉換成可高效率載入的 Dataset Shards，儲存在 S3，並固定 Dataset Version。

Step 2：Build Training Container

建立包含 PyTorch、CUDA、NCCL、Megatron / FSDP 程式碼和相依套件的 Docker Image。

Step 3：Configure GPU Cluster

確認 GPU 數量、VRAM、Interconnect、EFA、NCCL 配置與訓練節點的網路通訊。

Step 4：Distributed Initialization

在每個 Process 啟動：

```
import torch.distributed as distdist.init_process_group(backend="nccl")rank = dist.get_rank()world_size = dist.get_world_size()
```

實際還需要正確的 Rank / Device Mapping、Launcher 與 Process Group 設定。

Step 5：Training Monitoring

監控：

|Metric|檢查目的|
|---|---|
|Training / Validation Loss|是否收斂、是否 Overfit|
|Tokens / Second|實際訓練吞吐量|
|Model FLOPs Utilization|是否有效使用 GPU|
|GPU Memory / Utilization|是否有記憶體或計算瓶頸|
|Communication Time|NCCL / Network 是否成為瓶頸|
|Gradient Norm|是否有不穩定更新|
|NaN / Inf Rate|是否有數值溢位|
|Data Loader Wait Time|GPU 是否在等資料|
|Checkpoint Time|儲存是否拖慢訓練|

Step 6：Checkpoint & Recovery

不只儲存 Model Weights，還需保留可恢復訓練的 Optimizer / Scheduler 狀態、Training Step、RNG State，以及必要的 Distributed Sharding Metadata。

AWS SageMaker Checkpoint 機制支援因工作中斷後繼續訓練。

![](https://www.google.com/s2/favicons?domain=https://docs.aws.amazon.com&sz=32)

Amazon SageMaker AI

如果你在 Amazon 面試中說「用 SageMaker 訓練」，Senior 級面試官可能立刻追問：

> What happens if 1 GPU fails after 500,000 training steps?

有經驗的回答必須涵蓋 Job Failure Detection、Checkpoint Integrity、Resume、Distributed State Restoration、資料讀取位置、重複訓練與資料順序，以及恢復後的數值驗證。

這與單純知道如何點選 AWS Console，差別非常大。

## 十、Phase 7：Research Methodology——研究成果怎麼證明？

這是 Research Scientist / Senior Applied Scientist 最具代表性的工作。

你不能只說：

> I fine-tuned a model and accuracy improved.

你需要證明自己的方法有明確的 Scientific Contribution。

### 10.1 提出可以驗證的研究方法

延續前面的 Catalog Agent：

Research Question

Can verifiable reinforcement learning improve multimodal catalog agents beyond supervised fine-tuning under matched training and inference budgets?

Proposed Method

Verifiable Evidence-guided Agent RL。

核心想法：

在 Agent 訓練時，除了使用 Final Task Success，還使用具備 Evidence Provenance 的 Verification Signal，減少 Agent 無根據回答。

### 10.2 設計 Controlled Experiments

假設四種模型：

|Model|SFT|Tool Use|RL|Evidence Reward|
|---|---|---|---|---|
|A|✓|—|—|—|
|B|✓|✓|—|—|
|C|✓|✓|✓|—|
|D|✓|✓|✓|✓|

比較時要控制：

- 相同 Base Model。
    
- 相同 Training / Validation / Test Split。
    
- 相近的 Training Compute Budget。
    
- 相同 Tool Access。
    
- 相同 Inference Budget，或明確報告成本差異。
    
- 相同 Evaluation Protocol。
    

不然就無法知道改善究竟來自演算法，還是因為模型使用了更多 GPU、更多 Tokens 或更多 Tool Calls。

### 10.3 Ablation Study

假設 Model D 加入了幾個新元件。

要個別移除它們：

|Experiment|Task Success|Unsupported Claim Rate|
|---|---|---|
|Full Method|96%|1.5%|
|Without Evidence Reward|93%|4%|
|Without Tool Cost Penalty|95%|1.6%|
|Without RL|92%|5%|
|Without Multimodal Input|89%|4.5%|

假設性研究結果，只用來說明 Ablation Study 如何進行。

如果拿掉 Evidence Reward 後，Unsupported Claim Rate 大幅上升，就能支持 Evidence Reward 確實發揮作用。

但仍需要檢查 Confounding Factors、不同 Random Seeds 和統計不確定性。

### 10.4 不只看 Average Accuracy

大型企業 AI 系統必須做 Slice Analysis。

例如：

|Data Slice|特別要觀察|
|---|---|
|English Products|英文商品能力|
|Japanese Products|多語言泛化|
|Low-quality Images|視覺雜訊魯棒性|
|Missing Specifications|不完整資料處理|
|Conflicting Sources|證據衝突|
|New Product Families|Out-of-distribution|
|Long Tool Trajectories|Long-horizon Reliability|

一個模型的整體 Accuracy 即使達到 96%，如果在新品牌商品上只有 70%，仍可能不能推出 Production。

### 10.5 Statistical Significance

假設兩個模型在同一批 10,000 筆商品上：

- Model A Accuracy：93.2%
    
- Model B Accuracy：94.0%
    

改善是 0.8 Percentage Points。

這個差異是否可靠？

可以用 Paired Bootstrap 估計改善值的 Confidence Interval：

1. 以商品家族或適當的獨立群組為重抽樣單位。
    
2. 每次重新抽樣後計算 B−A 的差異。
    
3. 重複數千次。
    
4. 建立 95% Confidence Interval。
    
5. 檢查改善是否穩定，並同時分析業務上的實質效果。
    

如果測試樣本之間高度相關，不能把每張照片或每筆變體都當成完全獨立的觀察值。

### 10.6 形成 Research Paper

一篇研究論文通常包含：

- Problem & Motivation
    
- Related Work
    
- Proposed Method
    
- Theoretical / Algorithmic Formulation
    
- Experimental Setup
    
- Main Results
    
- Ablation Study
    
- Failure Analysis
    
- Limitations
    

例如研究論文可以有這樣的標題：

Verifiable Evidence-Guided Reinforcement Learning for Multimodal Catalog Agents

這只是教學範例，不是已存在或由 Amazon 發表的論文。

Senior Scientist 面試很常會深問：

> What was your novel contribution?

好的回答應該能明確區分自己提出的新方法，以及只是組合既有 Framework 的工程工作。

## 十一、Phase 8：把 Research Model 交付 Production

即使是研究職位，Amazon 的 Applied Scientist 通常也不能完全忽略 Production。

但他負責的 Production 深度可能與 Applied AI Engineer 不同。

### 11.1 研究成果如何進入真實服務？

假設研究驗證 Model D 表現最好：

#chatgpt-mermaid-_r_av2_{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;fill:rgb(13, 13, 13);}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#chatgpt-mermaid-_r_av2_ .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-_r_av2_ .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-_r_av2_ .error-icon{fill:rgb(243, 243, 243);}#chatgpt-mermaid-_r_av2_ .error-text{fill:rgb(13, 13, 13);stroke:rgb(13, 13, 13);}#chatgpt-mermaid-_r_av2_ .edge-thickness-normal{stroke-width:1px;}#chatgpt-mermaid-_r_av2_ .edge-thickness-thick{stroke-width:3.5px;}#chatgpt-mermaid-_r_av2_ .edge-pattern-solid{stroke-dasharray:0;}#chatgpt-mermaid-_r_av2_ .edge-thickness-invisible{stroke-width:0;fill:none;}#chatgpt-mermaid-_r_av2_ .edge-pattern-dashed{stroke-dasharray:3;}#chatgpt-mermaid-_r_av2_ .edge-pattern-dotted{stroke-dasharray:2;}#chatgpt-mermaid-_r_av2_ .marker{fill:rgb(143, 143, 143);stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_av2_ .marker.cross{stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_av2_ svg{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;}#chatgpt-mermaid-_r_av2_ p{margin:0;}#chatgpt-mermaid-_r_av2_ .label{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_av2_ .cluster-label text{fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_av2_ .cluster-label span{color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_av2_ .cluster-label span p{background-color:transparent;}#chatgpt-mermaid-_r_av2_ .label text,#chatgpt-mermaid-_r_av2_ span{fill:rgb(13, 13, 13);color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_av2_ .node rect,#chatgpt-mermaid-_r_av2_ .node circle,#chatgpt-mermaid-_r_av2_ .node ellipse,#chatgpt-mermaid-_r_av2_ .node polygon,#chatgpt-mermaid-_r_av2_ .node path{fill:rgb(222, 234, 251);stroke:rgb(83, 154, 248);stroke-width:1px;}#chatgpt-mermaid-_r_av2_ .rough-node .label text,#chatgpt-mermaid-_r_av2_ .node .label text,#chatgpt-mermaid-_r_av2_ .image-shape .label,#chatgpt-mermaid-_r_av2_ .icon-shape .label{text-anchor:middle;}#chatgpt-mermaid-_r_av2_ .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#chatgpt-mermaid-_r_av2_ .rough-node .label,#chatgpt-mermaid-_r_av2_ .node .label,#chatgpt-mermaid-_r_av2_ .image-shape .label,#chatgpt-mermaid-_r_av2_ .icon-shape .label{text-align:center;}#chatgpt-mermaid-_r_av2_ .node.clickable{cursor:pointer;}#chatgpt-mermaid-_r_av2_ .root .anchor path{fill:rgb(143, 143, 143)!important;stroke-width:0;stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_av2_ .arrowheadPath{fill:rgb(143, 143, 143);}#chatgpt-mermaid-_r_av2_ .edgePath .path{stroke:rgb(143, 143, 143);stroke-width:1px;}#chatgpt-mermaid-_r_av2_ .flowchart-link{stroke:rgb(143, 143, 143);fill:none;}#chatgpt-mermaid-_r_av2_ .edgeLabel{background-color:rgb(252, 252, 252);text-align:center;}#chatgpt-mermaid-_r_av2_ .edgeLabel p{background-color:rgb(252, 252, 252);}#chatgpt-mermaid-_r_av2_ .edgeLabel rect{opacity:0.5;background-color:rgb(252, 252, 252);fill:rgb(252, 252, 252);}#chatgpt-mermaid-_r_av2_ .labelBkg{background-color:rgba(252, 252, 252, 0.5);}#chatgpt-mermaid-_r_av2_ .cluster rect{fill:rgb(243, 243, 243);stroke:rgba(0, 0, 0, 0.1);stroke-width:1px;}#chatgpt-mermaid-_r_av2_ .cluster text{fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_av2_ .cluster span{color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_av2_ div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:12px;background:rgb(243, 243, 243);border:1px solid rgba(0, 0, 0, 0.1);border-radius:2px;pointer-events:none;z-index:100;}#chatgpt-mermaid-_r_av2_ .flowchartTitleText{text-anchor:middle;font-size:18px;fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_av2_ rect.text{fill:none;stroke-width:0;}#chatgpt-mermaid-_r_av2_ .icon-shape,#chatgpt-mermaid-_r_av2_ .image-shape{background-color:rgb(252, 252, 252);text-align:center;}#chatgpt-mermaid-_r_av2_ .icon-shape p,#chatgpt-mermaid-_r_av2_ .image-shape p{background-color:rgb(252, 252, 252);padding:2px;}#chatgpt-mermaid-_r_av2_ .icon-shape .label rect,#chatgpt-mermaid-_r_av2_ .image-shape .label rect{opacity:0.5;background-color:rgb(252, 252, 252);fill:rgb(252, 252, 252);}#chatgpt-mermaid-_r_av2_ .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#chatgpt-mermaid-_r_av2_ .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#chatgpt-mermaid-_r_av2_ .node .neo-node{stroke:rgb(83, 154, 248);}#chatgpt-mermaid-_r_av2_ [data-look="neo"].node rect,#chatgpt-mermaid-_r_av2_ [data-look="neo"].cluster rect,#chatgpt-mermaid-_r_av2_ [data-look="neo"].node polygon{stroke:url(#chatgpt-mermaid-_r_av2_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_av2_ [data-look="neo"].swimlane.cluster rect{filter:none;}#chatgpt-mermaid-_r_av2_ [data-look="neo"].node path{stroke:url(#chatgpt-mermaid-_r_av2_-gradient);stroke-width:1px;}#chatgpt-mermaid-_r_av2_ [data-look="neo"].node .outer-path{filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_av2_ [data-look="neo"].node .neo-line path{stroke:rgb(83, 154, 248);filter:none;}#chatgpt-mermaid-_r_av2_ [data-look="neo"].node circle{stroke:url(#chatgpt-mermaid-_r_av2_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_av2_ [data-look="neo"].node circle .state-start{fill:#000000;}#chatgpt-mermaid-_r_av2_ [data-look="neo"].icon-shape .icon{fill:url(#chatgpt-mermaid-_r_av2_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_av2_ [data-look="neo"].icon-shape .icon-neo path{stroke:url(#chatgpt-mermaid-_r_av2_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_av2_ .node text{font-size:14px;font-weight:600;letter-spacing:normal;fill:rgb(65, 65, 65);}#chatgpt-mermaid-_r_av2_ .edgeLabels text{font-size:13px;font-weight:600;letter-spacing:-0.08px;fill:rgb(65, 65, 65);}#chatgpt-mermaid-_r_av2_ .node tspan[font-weight="normal"],#chatgpt-mermaid-_r_av2_ .edgeLabels tspan[font-weight="normal"]{font-weight:600;}#chatgpt-mermaid-_r_av2_ .edgeLabel .label rect{opacity:1;rx:13px;ry:13px;fill:rgb(252, 252, 252);stroke:rgb(219, 219, 219);stroke-width:1px;}#chatgpt-mermaid-_r_av2_ .node rect,#chatgpt-mermaid-_r_av2_ .node circle,#chatgpt-mermaid-_r_av2_ .node ellipse,#chatgpt-mermaid-_r_av2_ .node polygon,#chatgpt-mermaid-_r_av2_ .node path{fill:rgb(243, 243, 243);stroke:rgba(0, 0, 0, 0.1);stroke-width:1px;}#chatgpt-mermaid-_r_av2_ .node rect{rx:16px;ry:16px;}#chatgpt-mermaid-_r_av2_ .node.mermaid-decision .label-container{fill:rgb(249, 249, 249);stroke:rgb(219, 219, 219);stroke-dasharray:2,2;}#chatgpt-mermaid-_r_av2_ .edgePaths .flowchart-link{stroke:rgb(143, 143, 143);stroke-width:1px;stroke-linecap:round;stroke-linejoin:round;}#chatgpt-mermaid-_r_av2_ .marker{fill:rgb(143, 143, 143);stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_av2_ :root{--mermaid-font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";}Research CheckpointOffline BenchmarkSafety / Robustness EvaluationServing OptimizationShadow DeploymentCanary ReleaseControlled A/B TestLaunch CriteriaGradual RolloutRollbackMonitoring / FeedbackPassFail

Senior Applied Scientist 通常要與 Software、Infrastructure、Inference 和 Product 團隊合作，確保研究模型能真正服務使用者。

### 11.2 Model Quality 不是唯一的上線指標

除了品質，還要考慮：

|指標|目的|
|---|---|
|Task Success|Agent 是否完成任務|
|False / Unsupported Claims|是否產生無根據資訊|
|Human Escalation Rate|需要多少人工|
|P95 / P99 Latency|長尾回應速度|
|Tokens per Task|Token 消耗|
|GPU Cost per Task|推論成本|
|Failure Recovery|工具或模型出錯後能否恢復|
|Safety Violations|是否違反系統限制|
|Cross-language Performance|跨國市場表現|

更進一步需要評估：SFT 後的 8B Model 是否能達到與 30B Model 相近的 Task Success，但用更低的推論成本？

這可能導向 Model Distillation、Quantization、Speculative Decoding 或更精簡的 Agent Architecture。

其中 Distillation 特別重要，因為 Research 團隊可以用高能力模型產生經過驗證的 Training Signals，再把能力轉移到較小模型。

## 十二、三種職位到底需要掌握到什麼程度？

這是你最初提出的核心問題。

### 12.1 技術能力比較

|技術|Senior Applied AI Engineer|Senior Applied Scientist|Foundation Model Research Engineer|
|---|---|---|---|
|LLM API / Prompt|精通|精通|熟悉|
|RAG / Retrieval|精通|視研究方向|視研究方向|
|Agent Architecture|精通系統設計|研究 Agent 能力|研究模型的 Agent 能力|
|Transformer Internals|能解釋與調整|深入理解|非常深入|
|Fine-tuning / LoRA|具實作能力|熟悉並能改進方法|深入|
|Full-parameter Training|加分|視職位可能必要|核心|
|Pretraining from Scratch|通常不必要|依團隊|通常高度相關|
|PPO / GRPO / RL|能整合或實作|能研究與訓練|核心或依專精|
|Diffusion / Flow|專案有需要才學|職缺相關方向需要|依研究領域|
|Distributed Training|理解基礎|實作能力很有價值|深入掌握|
|Scaling Laws|理解概念|研究級理解|核心研究能力|
|Novel Algorithms|非主要要求|重要|重要|
|Research Publications|通常非必要|重要加分或要求|經常重要|
|Production Deployment|核心能力|需要與團隊協作交付|視團隊|
|Latency / Cost / Monitoring|核心|重要|重要但重點不同|

這是依典型工作範圍做的比較，不是固定的職稱定義。實際上 Amazon AGI 的 Senior Applied Scientist 可能比某些公司掛名 Foundation Model Engineer 的職位更偏向 Pretraining Research。

### 12.2 同一個需求，三種人會怎麼做？

假設公司要求：

> Improve an agent's ability to detect incorrect product attributes.

Senior Applied AI Engineer

可能先使用現有高能力模型，建立 Retrieval、Tool Calling、Validation、Fallback 和 Monitoring。

主要目標：在成本、可靠性與延遲限制下，把正確率提高。

Senior Applied Scientist

可能研究為什麼 Agent 使用工具後仍然出錯，提出新的 Evidence Reward、Tool-selection Training 或 Agent Learning 方法。

主要目標：證明新的訓練方法能夠系統性改善模型能力。

Foundation Model Research Engineer

可能研究 Transformer 的多模態表示、Reasoning Post-training、大規模 RL 訓練、Context / Memory Architecture，或設計新的 Training Objective。

主要目標：提升基礎模型的通用能力，讓不同下游任務都受益。

## 十三、Amazon Senior Applied Scientist 技術面試可能怎麼問？

以下是根據職位研究內容推導的代表性考題，不是 Amazon 官方公布的特定題庫。

Question 1：How would you pretrain a 7B language model from scratch?

不能只回答使用 PyTorch。你需要討論資料清洗、Tokenizer、Architecture、Compute Budget、Loss、Optimizer、Distributed Training、Scaling Laws、Checkpoint 和 Validation。

Question 2：How does diffusion training work, and how is it different from flow matching?

需要解釋 Forward Noising、Training Objective、Sampling、Conditioning、Noise / Velocity Prediction 和實際生成效率。

Question 3：How would you design an RL environment for an LLM agent?

必須從 State、Action、Transition、Reward、Termination、Ground Truth 和 Reward Hacking 解釋，而不能只說使用 PPO。

Question 4：How do you prove that reinforcement learning improved reasoning rather than just increasing computation?

必須設計 Matched Compute / Inference Budget 的 Controlled Experiments、Baselines、Ablation、Held-out Evaluation、統計分析。

Question 5：Your 70B model cannot fit into GPU memory. What would you do?

比較 FSDP、ZeRO、Tensor / Pipeline Parallelism、Activation Checkpointing、Mixed Precision、Optimizer Sharding 與可能的 Offloading，說明選擇依據。

Question 6：Training loss suddenly becomes NaN after 100,000 steps. How do you debug?

檢查 Gradient Norm、Learning Rate、Mixed Precision Overflow、Bad Samples、Optimizer State、Distributed Synchronization，並利用 Checkpoint 重現和隔離問題。

Question 7：Your RL reward increases, but real task success decreases. Why?

討論 Reward Hacking、Reward Misspecification、Overfitting、Distribution Shift、Evaluation Leakage 和 Reward Model Exploitation。

Question 8：Describe a research contribution you personally developed.

面試官會追問你的 Novelty、與 Existing Work 的差別、Ablation、Statistical Significance、Failure Cases，以及哪些核心實驗由你親自設計。

其中第 8 題非常重要。

對 Senior Scientist，能夠展示一個真正有研究深度、完整驗證的專案，比列出自己使用過 15 個 AI Framework 更有說服力。

## 十四、如果要轉向 Amazon AI Lab，應該準備什麼作品？

以你既有的多相機影像擷取、Image Processing、CV / Segmentation 和 AI 系統經驗來看，一個合理的切入點是 Multimodal Applied Research，再逐步擴展到 Foundation Model Training 與 Agent RL。

例如，不一定要放棄熟悉的 Computer Vision 領域，直接跳去從零訓練超大型文字模型。

反而可以把熟悉的影像檢測問題提高到 Research Level：

> A Multimodal Inspection Agent that learns to request additional camera views, verify evidence, and minimize inspection cost through reinforcement learning.

### 專案技術架構

#chatgpt-mermaid-_r_b04_{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;fill:rgb(13, 13, 13);}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#chatgpt-mermaid-_r_b04_ .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-_r_b04_ .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-_r_b04_ .error-icon{fill:rgb(243, 243, 243);}#chatgpt-mermaid-_r_b04_ .error-text{fill:rgb(13, 13, 13);stroke:rgb(13, 13, 13);}#chatgpt-mermaid-_r_b04_ .edge-thickness-normal{stroke-width:1px;}#chatgpt-mermaid-_r_b04_ .edge-thickness-thick{stroke-width:3.5px;}#chatgpt-mermaid-_r_b04_ .edge-pattern-solid{stroke-dasharray:0;}#chatgpt-mermaid-_r_b04_ .edge-thickness-invisible{stroke-width:0;fill:none;}#chatgpt-mermaid-_r_b04_ .edge-pattern-dashed{stroke-dasharray:3;}#chatgpt-mermaid-_r_b04_ .edge-pattern-dotted{stroke-dasharray:2;}#chatgpt-mermaid-_r_b04_ .marker{fill:rgb(143, 143, 143);stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_b04_ .marker.cross{stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_b04_ svg{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;}#chatgpt-mermaid-_r_b04_ p{margin:0;}#chatgpt-mermaid-_r_b04_ .label{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_b04_ .cluster-label text{fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_b04_ .cluster-label span{color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_b04_ .cluster-label span p{background-color:transparent;}#chatgpt-mermaid-_r_b04_ .label text,#chatgpt-mermaid-_r_b04_ span{fill:rgb(13, 13, 13);color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_b04_ .node rect,#chatgpt-mermaid-_r_b04_ .node circle,#chatgpt-mermaid-_r_b04_ .node ellipse,#chatgpt-mermaid-_r_b04_ .node polygon,#chatgpt-mermaid-_r_b04_ .node path{fill:rgb(222, 234, 251);stroke:rgb(83, 154, 248);stroke-width:1px;}#chatgpt-mermaid-_r_b04_ .rough-node .label text,#chatgpt-mermaid-_r_b04_ .node .label text,#chatgpt-mermaid-_r_b04_ .image-shape .label,#chatgpt-mermaid-_r_b04_ .icon-shape .label{text-anchor:middle;}#chatgpt-mermaid-_r_b04_ .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#chatgpt-mermaid-_r_b04_ .rough-node .label,#chatgpt-mermaid-_r_b04_ .node .label,#chatgpt-mermaid-_r_b04_ .image-shape .label,#chatgpt-mermaid-_r_b04_ .icon-shape .label{text-align:center;}#chatgpt-mermaid-_r_b04_ .node.clickable{cursor:pointer;}#chatgpt-mermaid-_r_b04_ .root .anchor path{fill:rgb(143, 143, 143)!important;stroke-width:0;stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_b04_ .arrowheadPath{fill:rgb(143, 143, 143);}#chatgpt-mermaid-_r_b04_ .edgePath .path{stroke:rgb(143, 143, 143);stroke-width:1px;}#chatgpt-mermaid-_r_b04_ .flowchart-link{stroke:rgb(143, 143, 143);fill:none;}#chatgpt-mermaid-_r_b04_ .edgeLabel{background-color:rgb(252, 252, 252);text-align:center;}#chatgpt-mermaid-_r_b04_ .edgeLabel p{background-color:rgb(252, 252, 252);}#chatgpt-mermaid-_r_b04_ .edgeLabel rect{opacity:0.5;background-color:rgb(252, 252, 252);fill:rgb(252, 252, 252);}#chatgpt-mermaid-_r_b04_ .labelBkg{background-color:rgba(252, 252, 252, 0.5);}#chatgpt-mermaid-_r_b04_ .cluster rect{fill:rgb(243, 243, 243);stroke:rgba(0, 0, 0, 0.1);stroke-width:1px;}#chatgpt-mermaid-_r_b04_ .cluster text{fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_b04_ .cluster span{color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_b04_ div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:12px;background:rgb(243, 243, 243);border:1px solid rgba(0, 0, 0, 0.1);border-radius:2px;pointer-events:none;z-index:100;}#chatgpt-mermaid-_r_b04_ .flowchartTitleText{text-anchor:middle;font-size:18px;fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_b04_ rect.text{fill:none;stroke-width:0;}#chatgpt-mermaid-_r_b04_ .icon-shape,#chatgpt-mermaid-_r_b04_ .image-shape{background-color:rgb(252, 252, 252);text-align:center;}#chatgpt-mermaid-_r_b04_ .icon-shape p,#chatgpt-mermaid-_r_b04_ .image-shape p{background-color:rgb(252, 252, 252);padding:2px;}#chatgpt-mermaid-_r_b04_ .icon-shape .label rect,#chatgpt-mermaid-_r_b04_ .image-shape .label rect{opacity:0.5;background-color:rgb(252, 252, 252);fill:rgb(252, 252, 252);}#chatgpt-mermaid-_r_b04_ .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#chatgpt-mermaid-_r_b04_ .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#chatgpt-mermaid-_r_b04_ .node .neo-node{stroke:rgb(83, 154, 248);}#chatgpt-mermaid-_r_b04_ [data-look="neo"].node rect,#chatgpt-mermaid-_r_b04_ [data-look="neo"].cluster rect,#chatgpt-mermaid-_r_b04_ [data-look="neo"].node polygon{stroke:url(#chatgpt-mermaid-_r_b04_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_b04_ [data-look="neo"].swimlane.cluster rect{filter:none;}#chatgpt-mermaid-_r_b04_ [data-look="neo"].node path{stroke:url(#chatgpt-mermaid-_r_b04_-gradient);stroke-width:1px;}#chatgpt-mermaid-_r_b04_ [data-look="neo"].node .outer-path{filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_b04_ [data-look="neo"].node .neo-line path{stroke:rgb(83, 154, 248);filter:none;}#chatgpt-mermaid-_r_b04_ [data-look="neo"].node circle{stroke:url(#chatgpt-mermaid-_r_b04_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_b04_ [data-look="neo"].node circle .state-start{fill:#000000;}#chatgpt-mermaid-_r_b04_ [data-look="neo"].icon-shape .icon{fill:url(#chatgpt-mermaid-_r_b04_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_b04_ [data-look="neo"].icon-shape .icon-neo path{stroke:url(#chatgpt-mermaid-_r_b04_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_b04_ .node text{font-size:14px;font-weight:600;letter-spacing:normal;fill:rgb(65, 65, 65);}#chatgpt-mermaid-_r_b04_ .edgeLabels text{font-size:13px;font-weight:600;letter-spacing:-0.08px;fill:rgb(65, 65, 65);}#chatgpt-mermaid-_r_b04_ .node tspan[font-weight="normal"],#chatgpt-mermaid-_r_b04_ .edgeLabels tspan[font-weight="normal"]{font-weight:600;}#chatgpt-mermaid-_r_b04_ .edgeLabel .label rect{opacity:1;rx:13px;ry:13px;fill:rgb(252, 252, 252);stroke:rgb(219, 219, 219);stroke-width:1px;}#chatgpt-mermaid-_r_b04_ .node rect,#chatgpt-mermaid-_r_b04_ .node circle,#chatgpt-mermaid-_r_b04_ .node ellipse,#chatgpt-mermaid-_r_b04_ .node polygon,#chatgpt-mermaid-_r_b04_ .node path{fill:rgb(243, 243, 243);stroke:rgba(0, 0, 0, 0.1);stroke-width:1px;}#chatgpt-mermaid-_r_b04_ .node rect{rx:16px;ry:16px;}#chatgpt-mermaid-_r_b04_ .node.mermaid-decision .label-container{fill:rgb(249, 249, 249);stroke:rgb(219, 219, 219);stroke-dasharray:2,2;}#chatgpt-mermaid-_r_b04_ .edgePaths .flowchart-link{stroke:rgb(143, 143, 143);stroke-width:1px;stroke-linecap:round;stroke-linejoin:round;}#chatgpt-mermaid-_r_b04_ .marker{fill:rgb(143, 143, 143);stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_b04_ :root{--mermaid-font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";}Multi-camera ImagesVision EncoderSpecifications / Reference DataText EncoderMultimodal ModelAgent PolicyNext ActionRequest New Image / ViewRetrieve Reference EvidenceSubmit / AbstainInspection RL EnvironmentReward + Policy Training

這類專案可以具有四個不同層次：

|Level|具體研究／實作內容|
|---|---|
|1. Applied AI Engineering|使用現有 VLM + Tools 完成檢測工作流|
|2. Model Fine-tuning|用自建多模態資料 Fine-tune Vision-Language Model|
|3. Applied Research|研究 Active Perception、Reward、Missing-view Handling、Uncertainty|
|4. Foundation Model Research|修改跨視角 Transformer / Vision Tokenization / Multimodal Training Objective，研究跨資料集泛化|

這樣做還有一個很大的優點：你可以建立有客觀 Ground Truth 的 RL Environment。

例如 Agent 選擇「再拍另一個角度」有成本，但可能提高缺陷檢出率。

設計 Reward：

\[ R= R_{\text{decision}} -\lambda_1N_{\text{extra images}} -\lambda_2T_{\text{inspection}} -\lambda_3R_{\text{unsafe errors}} \]

再研究：

- 哪些影像需要額外拍攝？
    
- 模型是否能學會 Active View Selection？
    
- 降低拍攝數量是否會增加 False Negatives？
    
- RL 能否改善 Generalization？
    
- 使用不同 Multimodal Encoder，結論是否一致？
    

這會比單純做一個 VLM Demo 更接近 Amazon Applied Scientist 的研究形式。

## 十五、實際準備路線：從 Applied AI 到 Foundation Model Research

如果希望達到上述 Amazon Senior Applied Scientist 職位的技術要求，我會建議依序累積以下四層能力，而不是一開始就追求從零訓練 70B 模型。

Stage 1：Model Internals

親自用 PyTorch 實作小型 Transformer，能解釋 Attention、Backpropagation、Cross-Entropy、Optimizer、Training Stability。

成果：Train a small language model from scratch。

Stage 2：Modern Generative Model Training

親自 Fine-tune 一個 Open-weight LLM、訓練小型 Diffusion Model，理解 SFT、LoRA、DPO、GRPO、Evaluation。

成果：LLM + Diffusion Training Repositories，附實驗結果。

Stage 3：RL Agents + Distributed Training

建立可驗證 RL Gym，完成 Tool-use RL Training，使用多 GPU 的 FSDP / Distributed Training，處理 Checkpoint、Profiling 和恢復訓練。

成果：Reproducible Agent RL Experiment + Multi-GPU Training Pipeline。

Stage 4：Independent Research

找出一個現有方法的限制、提出新 Hypothesis、實作方法、完成 Baseline / Ablation / Held-out Evaluation，撰寫 Research Paper 或 Technical Report。

成果：具有明確 Novel Contribution 的研究作品。

達到 Stage 1–2，通常已能在不少 Senior Applied AI Engineer 職位展現有價值的 Model Training 能力；Stage 3–4 才會比較接近研究密集型 Senior Applied Scientist 職位對獨立研究與大規模訓練的期望。

但 Senior 職位仍要求工作經驗、研究領導、技術決策與跨團隊合作，不能只用完成幾個練習專案取代。

## 十六、最後總結：到底需不需要自己 Pretrain 一個 LLM？

可以分成三種情況。

|目標職位|是否需要從零 Pretrain LLM？|更重要的能力|
|---|---|---|
|Senior Applied AI Engineer|通常不需要|利用現有模型建立可靠 Production AI System|
|Senior Applied Scientist, AI Lab|依具體團隊，不一定需要自己完成從零 Pretraining|演算法研究、實際模型訓練、實驗設計、模型改進|
|Foundation Model Research Engineer|通常應具備直接參與 Pretraining 或大型 Post-training 的能力|模型內部、訓練演算法、Distributed Training、Scaling Research|

關鍵細節：不必曾經獨自從零訓練一個商業級 LLM，並不代表可以只會呼叫 API。

Foundation Model Research 團隊中的工程師可能專精 RL Post-training、Multimodal Architecture、Data Mixture、Distributed Optimization 或 Model Evaluation，不一定每個人都負責整條 Pretraining Pipeline。

但如果職缺明確寫著 Hands-on experience training large-scale foundation models，那你通常必須有直接參與模型訓練、優化或研究的可信經驗，而不是只有使用預訓練模型做產品的經驗。

Amazon 目前的 AGI Senior Applied Scientist 職缺就有這種明確要求。

![](https://www.google.com/s2/favicons?domain=https://amazon.jobs&sz=32)

Amazon.jobs

對於求職，最有說服力的技術展示不是「我知道什麼是 PPO、Diffusion、FSDP」，而是能完整回答：

> 我發現了什麼模型問題？提出了什麼技術假設？修改了哪個模型或訓練方法？如何以可重現的實驗證明改善？如何排除其他因素？模型在大規模訓練時發生過什麼問題？最後如何把研究成果安全地交付使用？

能夠深入回答這些問題，就真正開始具備從 Applied AI Engineering 走向 Senior Applied Scientist 或 Foundation Model Research 的核心競爭力。
