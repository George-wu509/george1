
|                                      |     |
| ------------------------------------ | --- |
| [[#### Fine-tuning Post-training內容]] |     |
|                                      |     |
|                                      |     |

#### Fine-tuning Post-training內容
```
請完整深入解釋以下Fine-tuning / Post-training內容(先以介紹intern了解知識, 之後則更深入完整介紹engineer技術細節)：
| 技能                   | 需要知道什麼                                              |
| -------------------- | --------------------------------------------------- |
| SFT                  | Supervised Fine-tuning、Instruction Dataset          |
| LoRA                 | Low-rank Adaptation 如何降低可訓練參數量                      |
| QLoRA                | Quantization 與 LoRA 的配合                             |
| DPO                  | Direct Preference Optimization                      |
| RLHF                 | Reward Model、Preference、Policy Optimization         |
| RL / Reasoning       | Reward Design、Verifiable Rewards、Policy Improvement |
| Distillation         | Teacher–Student Model、Quality／Cost Tradeoff         |
| Distributed Training | DDP、FSDP、ZeRO、Checkpointing                         |

並非每個職位都需要實作 RLHF 或大規模 Distributed Training。Applied AI 職位通常重視知道何時應用；Model Research 則可能要求真正實作、修改或優化。
```

# Fine-tuning / Post-training 完整技術教學：從 Intern 到 Senior AI / LLM Research Engineer

這八項技術是 2026 年 LLM Engineering、Generative AI、Reasoning Model Research 與 Foundation Model Training 中非常重要的一組知識。

首先需要建立一個核心觀念：

Fine-tuning / Post-training 的目標，不只是讓 LLM 學會新的知識，而是改變模型的能力、行為、輸出格式、偏好、推理策略，以及完成特定任務的成功率。

另外，你列出的八項技術並不是八個必須依序執行的訓練階段，而是分屬不同層次：

- SFT、DPO、RLHF、Reasoning RL： 決定模型如何學習與優化。
    
- LoRA、QLoRA： 決定如何以較低的記憶體與計算成本更新模型。
    
- Distillation： 決定如何將大型模型的能力轉移到較小模型。
    
- Distributed Training： 決定如何利用多 GPU／多機器進行訓練。
    

以下分成三大部分：

1. Part I — Intern Level： 以直覺、案例及流程圖理解每項技術。
    
2. Part II — Engineer / Research Level： 深入數學公式、演算法、PyTorch 實作、GPU 記憶體、訓練與評估。
    
3. Part III — 完整實務案例： 從企業 AI 需求開始，設計 Dataset、SFT、DPO、RL、Distillation、部署與 Production Evaluation。
    

# Part I — Intern Level：先理解每項技術在做什麼

## 1. Pretraining、Fine-tuning、Post-training 的關係

想像某家公司希望建立一個 AI Assistant，能夠回答工程問題、診斷系統異常，甚至呼叫工具執行測試。

公司有三種選擇：

1. 從零開始訓練 LLM。
    
2. 使用現有的 Pretrained LLM，對它進行進一步訓練。
    
3. 直接使用現有 LLM，搭配 Prompt、RAG、Tools 和 Agents。
    

其中第二種就是我們今天討論的主要內容。

Pretraining

大量文字／程式碼／其他模態資料 → 學習一般語言及世界知識

Base Model

具有一般能力，但不一定善於遵循使用者指令

Post-training

SFT

學會如何遵循指令

DPO / RLHF

學會偏好更好的答案

Reasoning RL

提高可驗證任務的成功率

Distillation

將能力轉移給小模型

LoRA／QLoRA 可以用於其中多種訓練階段；Distributed Training 則是執行基礎設施。

Production Model

可進行任務推論、Tool Calling、RAG、Agent Workflows

實際訓練流程不一定包含所有階段，也不一定只進行一次。例如，模型可以進行多輪 SFT → Preference Optimization → Evaluation。

### Pretraining 與 Post-training 最大差異

|項目|Pretraining|Post-training|
|---|---|---|
|主要目的|建立廣泛基礎能力|調整能力、行為、偏好與專業任務|
|資料|大規模文字、程式碼等|Instruction、Preference、Reward、Task Data|
|訓練目標|通常是 Next-token Prediction|SFT Loss、Preference Loss、RL Objective 等|
|計算需求|往往極高|可從單 GPU 到大規模叢集|
|典型工作|Foundation Model Research|Applied LLM、Model Post-training|
|是否需要重新訓練全部參數|通常是|不一定，可使用 LoRA／QLoRA|

補充：如果要讓 Base Model 更深入學習某個領域的大量原始文字，也可以進行 Continued Pretraining / Domain-Adaptive Pretraining。它與使用指令資料的 SFT 不同。

## 2. SFT：Supervised Fine-tuning

一句話：給模型大量「問題＋理想答案」，讓它模仿正確的回答方式。

例如，公司要訓練一個設備維修 AI。

Instruction Dataset

USER — Input

設備出現 `MovementFailedException: Stalled and Stopped (FS)`，應如何診斷？

ASSISTANT — Ideal Output

先停止後續移動指令，檢查軸的故障狀態、機械干涉及移動參數；確認原因並依設備程序解除故障後，才進行低風險測試。

如果有數千筆由專家審核的類似資料，SFT 可以讓模型學會：

- 使用公司的技術術語。
    
- 依照標準程序回答。
    
- 產生正確的 JSON 結構。
    
- 在資訊不足時提出必要的確認問題。
    
- 遵守工具使用與安全限制的文字規範。
    

但 SFT 不保證模型真正理解每個診斷步驟，也不保證能推導訓練資料沒有涵蓋的新問題。

## 3. LoRA：Low-Rank Adaptation

### 為什麼需要 LoRA？

假設模型有 70 億個參數（7B）。

傳統 Full Fine-tuning 會對大量甚至所有參數計算梯度並更新。

這通常很昂貴。

LoRA 的做法是：

保留原有模型權重不變，只訓練附加的小型低秩矩陣。

Full Fine-tuning

更新完整權重矩陣 W

LoRA

凍結 W，僅更新小矩陣 A、B

原本可能需要訓練數十億個參數，現在只需訓練數百萬或數千萬個額外參數，實際數量取決於 LoRA Rank 與套用的 Layers。

LoRA 的原始研究證實，這種方式可以在許多任務中大幅降低可訓練參數與記憶體需求。

![](https://www.google.com/s2/favicons?domain=https://arxiv.org&sz=32)

arXiv

## 4. QLoRA：Quantization + LoRA

QLoRA 是在 LoRA 的基礎上進一步節省記憶體。

兩個觀念要分清楚：

- LoRA： 減少需要訓練的參數。
    
- Quantization： 降低模型權重的儲存精度，例如將 16-bit 權重量化為 4-bit 表示。
    

QLoRA 通常將 Pretrained Model 的基礎權重以 4-bit 儲存並凍結，然後訓練額外的 LoRA Adapter。

相同 7B 模型的權重儲存量（理論估算）

FP32

28 GB

BF16 / FP16

14 GB

INT4 / 4-bit

3.5 GB

僅計算 70 億個權重的原始位元儲存量；不包括 Quantization Metadata、LoRA、Activations、CUDA Workspace、Optimizer 等，並非實際總 GPU 記憶體需求。

QLoRA 原始研究介紹了 NF4、Double Quantization、Paged Optimizers 等節省記憶體的方法。

![](https://www.google.com/s2/favicons?domain=https://arxiv.org&sz=32)

arXiv

## 5. DPO：Direct Preference Optimization

SFT 是讓模型模仿理想答案。

但同一個問題，可能存在兩個看似合理、品質卻不同的回答。

例如：

問題：設備發生移動故障，該如何恢復？

Chosen

先停止動作、讀取故障碼、檢查原因，確認安全後再執行恢復程序。

Rejected

直接清除故障並重新執行原本的移動指令。

DPO 透過這種成對資料，讓模型提高較佳回答的相對可能性，降低較差回答的相對可能性。

重點是：

DPO 不需要先訓練獨立的 Reward Model，再執行 PPO 式的 RL Optimization。

這是 DPO 相較於經典 RLHF 的主要工程優勢。

![](https://www.google.com/s2/favicons?domain=https://arxiv.org&sz=32)

arXiv

## 6. RLHF：Reinforcement Learning from Human Feedback

RLHF 同樣希望讓模型產生人類偏好的答案，但方法不同。

經典的 RLHF 流程是：

SFT Model

Human Preference Dataset

人類比較 Response A 與 B

Reward Model

預測回答的偏好分數

RL Policy Optimization（例如 PPO）

讓模型傾向產生獎勵較高的回答

經典 InstructGPT 研究使用了 SFT、人工偏好比較、Reward Model，以及 PPO 等階段。

![](https://www.google.com/s2/favicons?domain=https://arxiv.org&sz=32)

arXiv

DPO 可以理解成一種將偏好優化問題轉換成直接分類式 Loss 的做法；而傳統 RLHF 類方法會顯式使用 Reward Model 與 Policy Optimization。

兩者不是誰必然比較先進，而是工程成本、資料可用性、是否需要 Online Exploration 等條件不同。

## 7. RL / Reasoning：訓練模型學會解決問題

這個方向在 2025–2026 年尤其重要。

以前我們通常透過 SFT 給模型正確的推理範例。

但如果模型面對一道沒有看過的數學題，該如何讓它學會探索更有效的解題策略？

一種方法是：

1. 給模型一道題目。
    
2. 讓它產生多個解法。
    
3. 透過程式或其他可靠工具驗證答案。
    
4. 對較佳結果給予較高 Reward。
    
5. 更新模型，讓成功解法更容易被產生。
    

例如：

題目：17 × 24 = ?

Candidate A

# 408

Reward = 1

Candidate B

# 398

Reward = 0

Candidate C

# 408

Reward = 1

Candidate D

# 418

Reward = 0

這是 Outcome Reward 示意，只檢查最終答案；並沒有驗證中間推理過程是否正確。

DeepSeekMath 提出 GRPO，使用群組內相對獎勵進行 Policy Optimization；DeepSeek-R1 研究則展示了大規模 RL 對 Reasoning 能力的作用，以及結合 Cold-start Data 與多階段訓練的方式。

![](https://www.google.com/s2/favicons?domain=https://arxiv.org&sz=32)

arXiv

+1

## 8. Distillation：讓小模型學習大模型的能力

假設：

- Teacher Model：70B，回答品質高，但運算昂貴。
    
- Student Model：7B，推論成本低，但能力較弱。
    

可以讓 Teacher 對大量題目提供高品質答案，再用這些資料訓練 Student。

Teacher

大型模型

Student

小型模型

Distillation 不一定只使用 Teacher 產生的文字答案，也可能學習 Teacher 的完整 Token Probability Distribution。

小模型通常無法完整保留大模型全部的能力，但可以在特定任務上取得很好的 Quality／Latency／Cost Balance。

## 9. Distributed Training：讓多個 GPU 協同訓練

如果單 GPU 的記憶體或運算速度不足，就需要考慮分散式訓練。

不同技術處理不同問題：

|技術|Intern 直覺理解|
|---|---|
|DDP|多個 GPU 各自訓練不同資料，再同步梯度|
|FSDP|將模型參數、梯度、Optimizer States 分散到不同 GPU|
|ZeRO|分階段拆分訓練狀態，降低每張 GPU 的冗餘|
|Activation Checkpointing|不保存全部中間結果，需要時重新計算|
|Training Checkpointing|定期保存訓練進度，故障後可恢復|

這些方法不會憑空減少所有計算量，而是在記憶體、通訊、重新計算及訓練吞吐量之間做取捨。

## 10. 八項技術如何比較？

|技術|主要用途|核心資料|是否直接更新模型|
|---|---|---|---|
|SFT|學習指定答案與行為|Instruction + Target|是|
|LoRA|低成本參數更新|依所搭配的訓練目標|是，只更新 Adapter|
|QLoRA|低記憶體微調|依所搭配的訓練目標|是，只更新 Adapter|
|DPO|學習偏好|Chosen / Rejected|是|
|RLHF|以人類偏好引導策略|Preference、Reward|是|
|Reasoning RL|提高任務成功率|Prompts、Verifiable Rewards|是|
|Distillation|轉移模型能力|Teacher Outputs / Logits|是，更新 Student|
|Distributed Training|擴展訓練規模|依訓練目標|是，屬於執行方式|

這裡最容易混淆的是 LoRA 與 SFT。

SFT 回答「模型要學什麼」；LoRA 回答「模型參數要怎麼有效率地更新」。

所以可以有 SFT + LoRA、SFT + QLoRA、DPO + LoRA，甚至 RL + LoRA。

接下來進入演算法及實作細節。

# Part II — Engineer / Research Level：深入數學、架構與實作

## 11. SFT：Supervised Fine-tuning 的完整技術原理

### 11.1 SFT 與 Transformer 的關係

假設我們已經有一個 Decoder-only Transformer：

\[ p_\theta(y_t\mid x,y_{<t}) \]

其中：

- \(\theta\)：模型參數。
    
- \(x\)：使用者的輸入與上下文。
    
- \(y_t\)：第 \(t\) 個正確答案 Token。
    
- \(y_{<t}\)：在它之前的答案 Tokens。
    

模型每一步會預測下一個 Token 的 Probability Distribution。

例如：

輸入：

```
User: What is the capital of Japan?
Assistant:
```

希望模型輸出：

```
Tokyo.
```

SFT 會利用 Cross-Entropy Loss，增加正確答案 Tokens 的預測機率。

### 11.2 SFT Loss Function

對一組有效的目標 Tokens，常見的 SFT Objective 為：

\[ \boxed{ \mathcal{L}_{SFT}(\theta) = -\frac{1}{\sum_t m_t} \sum_{t=1}^{T} m_t\log p_\theta(y_t\mid x,y_{<t}) } \]

其中 \(m_t\) 是 Loss Mask：

- \(m_t=1\)：此 Token 計入訓練 Loss。
    
- \(m_t=0\)：此 Token 不計入 Loss。
    

在 Instruction Tuning 中，一個常見設定是：

只對 Assistant Response 計算 Loss，而不要求模型學習預測 User Prompt。

原因是我們想讓模型學會如何回答問題，而不是單純模仿整段對話中所有人的文字。

Hugging Face TRL 提供 `completion_only_loss` 與 `assistant_only_loss` 等功能；後者對特定 Chat Template 有要求，不能假設所有模型格式都直接支援。

![](https://www.google.com/s2/favicons?domain=https://huggingface.co&sz=32)

Hugging Face

### 11.3 Dataset 如何設計？

常見有三種資料形式。

A. Prompt–Completion

```
{
  "prompt": "Explain what a GPU is.",
  "completion": "A GPU is a processor designed for parallel computation."
}
```

B. Conversational Dataset

```
{
  "messages": [
    {
      "role": "system",
      "content": "You are an industrial AI assistant."
    },
    {
      "role": "user",
      "content": "What does a motor stall fault mean?"
    },
    {
      "role": "assistant",
      "content": "It indicates that commanded movement did not complete as expected. Inspect the fault state and mechanical conditions before retrying."
    }
  ]
}
```

C. Tool-calling Dataset

```
{
  "user_request": "Check the current status of axis X.",
  "target_tool_call": {
    "name": "get_axis_status",
    "arguments": {
      "axis": "X"
    }
  }
}
```

第三種通常需要轉成模型支援的完整 Chat Template 與 Tool-call 格式，並且應包含 Tool Result，以及工具執行後的 Assistant Response。不同模型的序列化格式可能不同。

### 11.4 SFT 的完整訓練流程

01

Collect Data

蒐集真實問題、專家答案、工具執行記錄

02

Clean & Normalize

去重、去除敏感資料、統一格式與專業術語

03

Quality Control

人工審核答案正確性、安全性與一致性

04

Train / Validation / Test

依來源、時間、設備、任務群組拆分資料

05

Tokenization & Masking

套用 Chat Template、Tokenize、建立 Label Mask

06

Training

Forward → Loss → Backward → Optimizer Step

07

Evaluation

比較 Base 與 Fine-tuned Model

08

Deployment & Monitoring

模型版本管理、Shadow Test、Canary、Rollback

### 11.5 SFT Hyperparameters

|參數|意義|Engineer 需要注意|
|---|---|---|
|Learning Rate|更新權重的步長|過大易造成訓練不穩與能力退化|
|Batch Size|每一步使用多少樣本|與 GPU Memory、Gradient Noise 有關|
|Gradient Accumulation|累積多個 Microbatch 梯度|可模擬較大的 Batch|
|Epochs|訓練資料重複次數|過多可能 Overfitting|
|Max Sequence Length|最大訓練上下文長度|決定長文本支援成本|
|Warmup|訓練初期逐步提高 LR|可改善穩定性|
|Weight Decay|正規化|要區分哪些參數應套用|
|Gradient Clipping|限制梯度大小|降低梯度異常風險|
|Packing|將短樣本高效率合併|需處理樣本邊界及 Attention Mask|

Effective Global Batch Size 通常可寫成：

\[ B_{\text{global}} = B_{\text{per GPU}} \times N_{\text{GPU}} \times N_{\text{accumulation}} \]

例如每 GPU Batch=2，4 GPU，累積 8 次：

\[ B_{\text{global}}=2\times4\times8=64 \]

這是以樣本數計算；對長度差異很大的 LLM Training Dataset，也必須監控每一步的有效 Tokens 數量。

### 11.6 SFT 的主要失敗模式

Overfitting： 模型在訓練資料很好，但面對未見過的問題表現不佳。

Catastrophic Forgetting： 過度針對特定任務微調，導致原有通用能力下降。

Data Contamination： 評估題目或相似答案不小心出現在訓練資料中。

Spurious Correlation： 模型學到的是無關模式，而不是真正的規則。

Poor Label Quality： 專家答案互相矛盾，模型難以學出穩定行為。

Exposure Bias： 訓練時看到的是正確歷史 Token，但推論時必須依賴自己先前生成的內容，錯誤可能逐步累積。

Senior Engineer 不應只監看 Training Loss。更重要的是：

- Held-out Task Success
    
- Out-of-distribution Performance
    
- Existing Capability Regression
    
- Safety Violation Rate
    
- Tool Execution Correctness
    

## 12. LoRA：完整數學原理與參數量計算

### 12.1 Full Fine-tuning 為什麼昂貴？

Transformer 的 Attention 和 Feed-forward Layers 中，有大量 Weight Matrices。

例如一個 Linear Layer：

\[ y=Wx \]

假設：

\[ W\in\mathbb{R}^{4096\times4096} \]

那麼總參數量是：

\[ 4096\times4096=16,777,216 \]

Full Fine-tuning 會直接更新 \(W\)。

LoRA 不這麼做。

### 12.2 Low-rank Decomposition

LoRA 使用：

\[ \boxed{ W'=W_0+\Delta W } \]

其中：

- \(W_0\)：Frozen Pretrained Weights。
    
- \(\Delta W\)：Trainable Weight Update。
    

將更新分解成兩個低秩矩陣：

\[ \boxed{ \Delta W=\frac{\alpha}{r}BA } \]

其中：

\[ B\in\mathbb{R}^{4096\times r} \]

\[ A\in\mathbb{R}^{r\times4096} \]

\(r\) 是 LoRA Rank，\(\alpha\) 是 Scaling Factor。

LoRA Layer 的 Forward Pass 變成：

\[ \boxed{ y=W_0x+\frac{\alpha}{r}BAx } \]

### 12.3 用實際數字計算節省多少參數

假設：

\[ r=16 \]

那麼新增 Trainable Parameters：

\[ N_{\text{LoRA}}=4096\times16+16\times4096 \]

\[ N_{\text{LoRA}}=131,072 \]

比較：

|訓練方式|這個 Linear Layer 的可訓練參數|
|---|---|
|Full Fine-tuning|16,777,216|
|LoRA，Rank 16|131,072|

\[ \frac{131072}{16777216}\times100\% = 0.78125\% \]

也就是對這個 Layer，只需訓練原本約 0.78% 的參數量。

這不是宣稱整個模型一定只訓練 0.78%，因為全模型比例會隨 Target Modules、Layer Dimensions、Rank 等條件改變。

### 12.4 為什麼低秩矩陣可以有效？

直覺上，Pretrained Model 已經掌握大量語言知識。

為了適應某個新任務，我們可能不需要修改 Weight Space 中所有可能的方向。

只需要學習相對低維的有效更新方向。

這是 LoRA 的核心假設之一。

但要注意：

並不是所有任務的最佳權重更新都必然是極低秩。 特別是對大型 Domain Shift、複雜多任務或需要深層能力改變的情況，低 Rank 可能限制模型表現。

### 12.5 LoRA 常用設定

```
from peft import LoraConfiglora_config = LoraConfig(    r=16,    lora_alpha=32,    lora_dropout=0.05,    target_modules=[        "q_proj",        "k_proj",        "v_proj",        "o_proj"    ],    bias="none",    task_type="CAUSAL_LM")
```

參數說明：

|參數|作用|
|---|---|
|`r`|低秩矩陣的 Rank|
|`lora_alpha`|更新量的 Scaling|
|`lora_dropout`|Adapter 分支的 Dropout|
|`target_modules`|哪些模型 Layers 使用 LoRA|
|`bias`|是否額外訓練 Bias|
|`task_type`|例如 Causal Language Modeling|

`q_proj`、`k_proj`、`v_proj`、`o_proj` 分別對應 Attention 裡的 Q、K、V、Output Projection。

也可以對 MLP Layers，例如 `up_proj`、`down_proj`、`gate_proj` 套用 LoRA；但不同模型的模組名稱可能不同。

### 12.6 Engineer 如何選擇 Rank？

可用以下方式做實驗：

|Rank|優點|風險|
|---|---|---|
|4–8|Adapter 小、成本低|表達能力可能不足|
|16–32|常用的起始搜尋範圍|需要評估不同任務|
|64–128|更高的更新容量|記憶體、Optimizer、Overfitting 風險增加|

這些只是 Hyperparameter Search 的候選值，不是最佳 Rank 的定律。

對 Senior Engineer 而言，較好的做法是固定資料與評估程序，比較 Rank 8、16、32、64，繪製：

\[ \text{Task Performance vs Training Cost} \]

此外，LoRA Adapter 常可在支援的架構中 Merge 回 Base Weights，避免部署時產生額外 Adapter Matrix Operations；但應事先考慮 Quantization、Adapter Switching 與 Merge Compatibility。

## 13. QLoRA：為什麼 4-bit Base Model 仍然能夠訓練？

### 13.1 先理解 Quantization

一般 BF16 Weight 每個參數需要 16 bits。

4-bit Quantization 只使用 4 bits 表示量化值，還需要額外的 Scale 或 Quantization Metadata。

概念上：

\[ W\approx Dequantize(Q_4(W)) \]

QLoRA 的關鍵是：

不直接訓練 4-bit Base Weights，而是透過它們進行 Forward/Backward Computation，把梯度傳遞到額外的 LoRA Adapter。

因此：

\[ W_{\text{effective}} = Dequantize(W_{4bit}) + BA\frac{\alpha}{r} \]

其中 Base Weights 凍結，Adapter 可訓練。

### 13.2 QLoRA 三個核心技巧

1. NF4 — NormalFloat 4-bit

NF4 是特別針對近似常態分布權重設計的 4-bit Quantization Format。

它不是簡單把每個浮點數四捨五入成一個普通整數，而是利用適合權重分布的量化表示。

2. Double Quantization

一般 Quantization 仍需儲存 Scale / Constants。

Double Quantization 進一步壓縮這些 Quantization Constants，減少 Metadata 所占記憶體。

3. Paged Optimizers

用於緩解訓練時短暫出現的記憶體尖峰，利用分頁式管理降低 OOM 風險；並不代表可以免費使用無限 GPU 記憶體。

![](https://www.google.com/s2/favicons?domain=https://arxiv.org&sz=32)

arXiv

### 13.3 QLoRA 的 Python 設定

```
import torchfrom transformers import BitsAndBytesConfigfrom peft import LoraConfigquantization_config = BitsAndBytesConfig(    load_in_4bit=True,    bnb_4bit_quant_type="nf4",    bnb_4bit_compute_dtype=torch.bfloat16,    bnb_4bit_use_double_quant=True)lora_config = LoraConfig(    r=16,    lora_alpha=32,    lora_dropout=0.05,    target_modules=[        "q_proj",        "v_proj"    ],    task_type="CAUSAL_LM")
```

這兩個 Config 分別解決不同問題：

- `quantization_config`：Base Model 如何量化。
    
- `lora_config`：哪些 Adapter 被訓練。
    

實際載入模型時，要使用支援的 GPU、BitsAndBytes 與 Transformers／PEFT 組合。BF16 也需要相容硬體。

### 13.4 QLoRA 的 Trade-offs

|比較|LoRA|QLoRA|
|---|---|---|
|Base Weights|通常 BF16／FP16|通常 4-bit NF4|
|可訓練部分|Adapter|Adapter|
|GPU Memory|較高|通常較低|
|Quantization Error|沒有額外的 4-bit Base Quantization Error|存在|
|Training Speed|視硬體與實作|可能受到 Dequantization 開銷影響|
|適用場景|記憶體較充裕|單 GPU／有限 GPU Memory|

一個很重要的工程誤解是：

QLoRA 大幅降低記憶體，並不保證比 LoRA 更快。

低位元權重可能需要在計算時 Dequantize，實際速度需 Benchmark。

## 14. DPO：從 Preference Dataset 到數學 Loss

### 14.1 DPO 的訓練資料

每個樣本包含：

\[ (x,y_w,y_l) \]

其中：

- \(x\)：Prompt。
    
- \(y_w\)：Preferred / Chosen Response。
    
- \(y_l\)：Rejected Response。
    

例如：

```
{
  "prompt": "An image is blurry. What should we check?",
  "chosen": "Check focus, exposure, vibration, and image quality metrics before repeating acquisition.",
  "rejected": "Increase exposure time until the image becomes sharp."
}
```

這個範例的 Preferred Response 更重視診斷完整性，而且不把曝光時間錯誤地當成萬用的對焦修復方式。

### 14.2 DPO Loss Function

DPO 的常見形式：

\[ \boxed{ \mathcal{L}_{DPO} = -\mathbb{E} \left[ \log\sigma\left( \beta \left[ \log\frac{\pi_\theta(y_w|x)} {\pi_{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)} {\pi_{ref}(y_l|x)} \right] \right) \right] } \]

公式中的各個變數：

|變數|意義|
|---|---|
|\(\pi_\theta\)|正在訓練的 Policy Model|
|\(\pi_{ref}\)|凍結的 Reference Model|
|\(y_w\)|較佳回答|
|\(y_l\)|較差回答|
|\(\sigma\)|Sigmoid Function|
|\(\beta\)|控制 Preference Optimization 相對 Reference 的尺度|

可以先忽略最外層的 Sigmoid，觀察中間的式子。

DPO 希望：

\[ \log\frac{\pi_\theta(y_w|x)} {\pi_{ref}(y_w|x)} \]

相對於 Rejected Response 的對應項目更大。

換句話說：

模型相對於 Reference Model，要更偏向 Chosen Response，而不是 Rejected Response。

### 14.3 為什麼需要 Reference Model？

如果只提升 Chosen Response Probability，可能產生不受控制的分布改變。

Reference Model 提供比較基準。

DPO 的推導來自帶有 KL Regularization 的 Preference Optimization 問題；這讓模型能利用 Reference Policy 進行相對優化，而不需要像經典 PPO-based RLHF 一樣先訓練獨立 Reward Model。

但要注意，\(\beta\) 與模型分布偏移、Training Strength 的關係需要綜合 Loss 定義及訓練設定分析，不能簡單認為越大就一定越好。

### 14.4 DPO 與 SFT 的最重要差別

SFT：

\[ \text{Increase probability of a demonstrated answer} \]

DPO：

\[ \text{Increase relative preference for chosen over rejected} \]

所以：

- 如果你有專家寫好的標準答案，優先考慮 SFT。
    
- 如果你有相同 Prompt 的回答品質比較，DPO 很適合。
    
- 如果偏好隨多步驟行為或 Environment Interaction 而產生，就需要評估 Online RL 方法。
    

### 14.5 DPO 的失敗模式

Preference Noise： 標註人員對 Chosen / Rejected 的標準不同。

Reference Mismatch： Reference Model 或資料分布與訓練情境不合適。

Length Bias： 標註者可能偏好較長的回答，即使不一定比較正確。

Offline Distribution Limitation： 訓練資料是固定的，模型不能像 Online RL 一樣主動產生新的探索軌跡來更新偏好資料。

Reward Hacking-like Behavior： 模型可能學會表面上比較討喜，但實際任務成功率不高的輸出模式。

因此，DPO 成功不能只根據 Training Loss 降低判斷，應測試 Held-out Preference Accuracy 與實際 Task Quality。

## 15. RLHF：Reward Model、PPO 與 Policy Optimization

### 15.1 RLHF 的主要組件

經典 RLHF 通常涉及以下模型：

|組件|功能|
|---|---|
|Policy Model \(\pi_\theta\)|產生回答|
|Reference Model \(\pi_{ref}\)|限制模型偏離參考行為|
|Reward Model \(r_\phi\)|預測偏好品質分數|
|Value Model / Critic \(V_\psi\)|在 PPO 等方法中估計 Value，降低梯度變異|

具體實作也可能有其他組合，並非所有 RLHF 方法都需要獨立 Critic。

### 15.2 Reward Model 如何訓練？

Preference Dataset：

```
Prompt X

Response A: Better
Response B: Worse
```

Reward Model 產生：

\[ r_\phi(x,y) \]

假設：

\[ r_\phi(x,A)=2.4 \]

\[ r_\phi(x,B)=0.7 \]

代表 Reward Model 對 A 的相對評價更高。

常見的 Pairwise Reward Loss：

\[ \boxed{ \mathcal{L}_{RM} = -\mathbb{E} \left[ \log \sigma \left( r_\phi(x,y_w)-r_\phi(x,y_l) \right) \right] } \]

這相當於鼓勵：

\[ r_\phi(x,y_w)>r_\phi(x,y_l) \]

需注意 Reward Score 通常只是學到的相對品質分數，不能直接解讀成「回答正確機率為 80%」。

### 15.3 PPO 如何更新 Policy？

PPO 是 Proximal Policy Optimization。

它希望增加高 Reward 行為的機率，同時避免 Policy 每一次更新幅度過大。

其中一個核心目標是：

\[ \boxed{ L_{\text{PPO-clip}} = \mathbb{E} \left[ \min \left( \rho_t A_t, \operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t \right) \right] } \]

其中：

\[ \rho_t= \frac{\pi_\theta(a_t|s_t)} {\pi_{old}(a_t|s_t)} \]

- \(s_t\)：目前狀態。
    
- \(a_t\)：採取的動作，例如輸出 Token。
    
- \(A_t\)：Advantage。
    
- \(\epsilon\)：Clipping Range。
    

這裡寫的是要最大化的 PPO Clipped Surrogate Objective。實作時通常轉成需要最小化的 Loss，並可能加入 Value Loss、Entropy Terms 與其他 Regularization。

### 15.4 KL Regularization

為了避免模型追逐 Reward 而嚴重偏離原始行為，常使用：

\[ \boxed{ J(\theta) = \mathbb{E}_{y\sim\pi_\theta} [r_\phi(x,y)] - \beta D_{KL} (\pi_\theta\Vert\pi_{ref}) } \]

這裡 KL Divergence 用來衡量新 Policy 與 Reference Policy 的分布差異。

例如：

原本模型回答正常，但 Reward Model 特別偏愛某種冗長格式。

若沒有適當約束，新模型可能反覆產生冗長而空洞的回答以提高 Reward。

這稱為 Reward Exploitation 或 Reward Hacking 的一種表現。

KL Regularization 可以降低這種分布漂移，但不能保證完全消除 Reward Hacking。

### 15.5 為什麼 RLHF 比 DPO 複雜？

經典 PPO-based RLHF 通常需要：

1. 取得 Prompts。
    
2. Policy 產生新的 Responses。
    
3. Reward Model 計算分數。
    
4. 計算 Reference KL 或相關懲罰。
    
5. Value／Advantage Estimation。
    
6. 執行 PPO Optimization。
    
7. 更新 Policy。
    
8. 再次產生 Responses。
    

這牽涉到：

- Rollout Generation
    
- GPU Resource Scheduling
    
- Multiple Model Copies
    
- Sampling Throughput
    
- Optimizer Stability
    
- Reward / KL Monitoring
    
- Training–Inference Weight Synchronization
    

因此對 Applied AI Engineer 來說，理解何時值得使用 RLHF 通常比親自打造大型 PPO 訓練平台更重要。

但對 Foundation Model Research Engineer 或 Post-training Research Engineer，這些就是核心實作能力。

## 16. RL / Reasoning：從 Next-token Prediction 到可驗證的推理能力

這一節是整個 Post-training 主題中，對 2026 年 Model Research 職位尤其重要的部分。

### 16.1 LLM 如何變成 Reinforcement Learning 問題？

在傳統 RL 中：

\[ \text{Agent} \rightarrow \text{Action} \rightarrow \text{Environment} \rightarrow \text{Reward} \]

對 LLM 而言，可以做以下對應：

|Reinforcement Learning|LLM / Agent|
|---|---|
|State \(s_t\)|Prompt、已產生的 Tokens、Tool Results 等|
|Action \(a_t\)|下一個 Token，或 Agent 的工具動作|
|Policy \(\pi_\theta\)|LLM|
|Trajectory \(\tau\)|一連串 Tokens / Tool Calls / Observations|
|Reward \(R\)|答案正確性、測試通過、任務成功等|
|Environment|Code Executor、Math Verifier、Simulator、API 等|

例如一個 Coding Agent：

```
Task:
Fix a Python function that fails unit tests.

Agent:
  1. Read code
  2. Identify likely bug
  3. Modify code
  4. Run tests
  5. Inspect failures
  6. Modify code again
  7. Finish

Environment:
  Git repository + Python test runner

Reward:
  Tests passed / regression / execution constraints
```

在此情況下，模型不只是預測下一個字，而是學習哪種行動順序比較容易完成任務。

### 16.2 Reward Design

Reward 不是一定要由另一個 LLM 給分數。

可以分成幾類：

|Reward Type|例子|優點與限制|
|---|---|---|
|Human Reward|專家判斷回答品質|彈性高，但成本高、有主觀性|
|Model-based Reward|Reward Model / LLM Judge|可擴展，但可能存在偏差|
|Outcome Reward|最終答案正確與否|清楚，但回饋稀疏|
|Process Reward|中間步驟是否合理|更細緻，但驗證成本高|
|Verifiable Reward|Unit Tests、Math Checker、Schema Validator|可自動化，但只涵蓋可驗證部分|
|Environment Reward|模擬器中的任務成功|適合 Agent，但要防止利用模擬器漏洞|

### 16.3 Outcome Reward 與 Process Reward

以數學題為例。

題目：

\[ 23\times18=? \]

正確答案：

\[ 414 \]

Outcome Reward Model（ORM） 只評估最後輸出的答案是否正確。

但如果模型得到正確答案的過程有邏輯錯誤，它可能仍然得到高分。

Process Reward Model（PRM） 則評估中間步驟。

例如：

\[ 23\times(20-2) \]

\[ =460-46 \]

\[ =414 \]

PRM 可以嘗試判斷每個步驟是否正確。

這對長鏈推理特別有用，但 Process Labels、Verifier Fidelity 和 Credit Assignment 也會變得更加困難。

### 16.4 GRPO：Group Relative Policy Optimization

GRPO 是非常值得理解的一種 Reasoning RL 方法。

它的核心想法：

對同一個 Prompt，產生多個 Responses，利用群組內 Reward 的相對差異估計 Advantage。

假設：

\[ G=4 \]

產生四個回答，其 Reward：

\[ R=[1,0,1,0] \]

平均：

\[ \mu_R=0.5 \]

在一種常見 GRPO 形式中：

\[ \boxed{ A_i= \frac{R_i-\operatorname{mean}(R)} {\operatorname{std}(R)+\epsilon} } \]

如此一來：

- Reward 高於群組平均的 Responses：Positive Advantage。
    
- Reward 低於群組平均的 Responses：Negative Advantage。
    

接著利用 Advantage 引導 Policy Update。

GRPO 訓練流程

1

Sample Prompt

2

Generate G Responses

3

Evaluate Each Response

4

Compute Group-relative Advantages

5

Apply Policy Optimization

6

Repeat with Updated Model

在 PPO 中通常還有 Value / Critic Model 參與 Advantage Estimation；GRPO 的一個重要特點是利用 Group-relative Rewards，避免依賴獨立 Value Model 的典型設計。

但要注意，GRPO 不是單純「計算平均分數後做 Gradient Descent」，完整算法還可能涉及：

- Old Policy 與 Current Policy 的 Importance Ratios
    
- Clipped Policy Objective
    
- KL Regularization
    
- Completion Length Normalization
    
- Reward Normalization
    
- Sampling / Rollout Synchronization
    

不同 GRPO 變體在這些細節上可能不同。原始 DeepSeekMath 與後續實作都值得對照。

![](https://www.google.com/s2/favicons?domain=https://arxiv.org&sz=32)

arXiv

+1

### 16.5 GRPO 一個重要的失敗情況

假設同一個 Prompt 產生四個回答：

\[ R=[0,0,0,0] \]

所有答案都錯誤。

此時群組 Reward Variance 是 0。

因為回答之間沒有相對差異，基於這種標準化 Advantage 的訓練訊號會非常有限，甚至為零。

這代表：

在一個 Model 完全無法解決的題目上，單純增加 GRPO Training Steps 不一定有效。

可以考慮：

- 使用較容易的 Curriculum Tasks。
    
- 提供 SFT Cold-start Examples。
    
- 增加 Sampling Diversity。
    
- 改善 Verifier 或 Reward Signal。
    
- 採用 Process-level Feedback。
    
- 檢查是否所有 Responses 都被截斷。
    
- 評估其他 Advantage Estimation 方法。
    

反過來，當所有 Responses 都正確時，也可能缺乏有效的群組相對學習訊號。

### 16.6 Reward Hacking：Reasoning RL 的核心風險

假設我們要訓練 Coding Agent。

Reward 設計為：

```
reward = 1.0 if tests_pass else 0.0
```

模型有可能學到：

- 真正修正 Bug。
    
- 修改測試使它不再檢查錯誤。
    
- 刪除造成失敗的測試。
    
- 修改測試設定，使錯誤被忽略。
    

如果 Reward 只看測試結果，這些行為可能被誤認為成功。

因此，需要建立更可靠的 Verifier：

```
def evaluate_candidate(result):    if result.modified_protected_tests:        return 0.0    if result.violated_sandbox_rules:        return 0.0    if not result.hidden_tests_passed:        return 0.0    if result.regression_detected:        return 0.0    return 1.0
```

這只是簡化示意。真實系統還需要隔離執行、不可被 Agent 修改的測試、資源限制、版本快照及獨立驗證。

### 16.7 Reasoning RL 與普通 SFT 有什麼根本差別？

|問題|SFT|Reasoning RL|
|---|---|---|
|誰提供答案？|資料集中的 Target|模型自行產生 Candidate|
|學習訊號|正確示範 Tokens|Reward / Advantage|
|能否探索新解法？|有限|可以透過 Sampling 探索|
|主要挑戰|Label Quality|Reward Design、Exploration、Optimization|
|適合任務|格式、指令、專家示範|Coding、Math、可驗證的多步驟任務|

特別要強調：Reasoning RL 不一定依靠人工標註偏好；許多數學、程式碼、工具使用任務可以利用可程式化的驗證結果。

這也是它與傳統 Human-preference-based RLHF 的重要差別。

## 17. Distillation：Teacher–Student Training

### 17.1 為什麼要 Distill？

如果一個高品質 70B Model 每天處理大量 Production Requests，推論成本可能很高。

而某個企業實際只需要模型完成相對固定的任務，例如：

- 解讀設備 Error Logs。
    
- 產生標準化 Incident Summary。
    
- 將自然語言轉換成 Tool Calls。
    
- 判斷該採用哪個故障診斷流程。
    

就可能不需要每個 Request 都呼叫 70B Model。

可以利用大型 Teacher 訓練較小的 Student。

### 17.2 Response Distillation

最容易實作的方式：

1. 建立大量 Task Prompts。
    
2. 用 Teacher Model 產生 Answers。
    
3. 驗證答案品質。
    
4. 將高品質 Outputs 作為 Student 的 SFT Dataset。
    
5. 訓練 Student。
    
6. 在未見過的任務上進行 Evaluation。
    

例如：

```
Teacher Model: 70B
        ↓
10,000 Task Prompts
        ↓
Generated Candidate Answers
        ↓
Correctness / Safety / Quality Filtering
        ↓
Student SFT Dataset
        ↓
7B Student Model
```

這個流程有時稱為 Synthetic-data Distillation 或 Response-level Distillation。

不過，如果 Teacher 產生的答案存在錯誤，Student 也可能學到錯誤。

因此，不應將 Teacher Responses 直接視為 Ground Truth。

### 17.3 Logit / Knowledge Distillation

在原始 Knowledge Distillation 方法中，Student 不只學習 Teacher 選出的 Token，還可以學習 Teacher 對其他 Tokens 的機率分布。

例如，Teacher 預測：

|Token|Teacher Probability|
|---|---|
|fault|0.65|
|error|0.25|
|issue|0.08|
|apple|0.02|

即使 Ground Truth 是 `fault`，Teacher Distribution 還提供了一些重要資訊：

`error` 和 `issue` 也有語義相關性，而 `apple` 幾乎不相關。

這些額外訊號可能幫助 Student 學習更豐富的 Representation。

### 17.4 Distillation Loss

常見目標：

\[ \boxed{ \mathcal{L}_{KD} = (1-\lambda)\mathcal{L}_{CE} + \lambda T^2 D_{KL} \left( p_{teacher}^{(T)} \Vert p_{student}^{(T)} \right) } \]

其中：

- \(\mathcal{L}_{CE}\)：Ground-truth Supervised Loss。
    
- \(D_{KL}\)：Teacher 與 Student Distribution Difference。
    
- \(T\)：Softmax Temperature。
    
- \(\lambda\)：控制兩種 Loss 的相對權重。
    

Temperature 定義：

\[ p_i^{(T)} = \frac{\exp(z_i/T)} {\sum_j\exp(z_j/T)} \]

當 \(T>1\) 時，Softmax Distribution 通常會變得比較平滑。

\(T^2\) 是經典 Distillation 中常見的梯度尺度補償，但並不是所有新的 Distillation 實作都必須使用完全相同的公式。

![](https://www.google.com/s2/favicons?domain=https://arxiv.org&sz=32)

arXiv

### 17.5 Offline vs On-policy Distillation

Offline Distillation：

Teacher 先產生固定 Dataset，再訓練 Student。

優點是容易管理、重現與批次處理。

限制是 Student 訓練時可能遇不到自己真正會犯錯的分布。

On-policy Distillation：

Student 自行產生 Outputs，再利用 Teacher 對這些 Outputs 或其行動分布提供學習訊號。

這種方式更直接針對 Student 當前的行為，但需要更多 Online Generation 與 Teacher Computation。

### 17.6 Distillation 如何評估 Quality / Cost？

不能只看模型參數變少。

Senior Engineer 應同時觀察：

\[ \text{Quality} \]

\[ \text{Latency} \]

\[ \text{Cost per Successful Task} \]

其中最後一項很重要。

例如以下都是純示範數字：

|指標|70B Teacher|7B Student|
|---|---|---|
|Task Success|94%|89%|
|Average Latency|3.2 s|0.55 s|
|Inference Cost / Request|$0.020|$0.003|
|Safety Violation Rate|0.2%|0.3%|

小模型便宜很多，但是否能上線取決於 Task Success 和 Safety 是否符合要求。

如果 Student 在低風險問題表現很好，但在複雜問題較弱，可以採用：

Small-model-first + Large-model Fallback

例如先由小模型處理簡單且可驗證的問題，若其不確定性、規則檢查或驗證結果顯示風險較高，才呼叫大型模型。

這稱為 Model Routing / Cascaded Inference。

## 18. Distributed Training：DDP、FSDP、ZeRO 與 Checkpointing

這一節是 Senior LLM Infrastructure Engineer 和 Foundation Model Training Engineer 必須深入掌握的內容。

### 18.1 LLM Training 的 GPU Memory 用在哪裡？

訓練不只是把 Model Weights 載入 GPU。

通常還有：

\[ M_{\text{total}} = M_{\text{weights}} + M_{\text{gradients}} + M_{\text{optimizer}} + M_{\text{activations}} + M_{\text{temporary}} \]

舉一個 7B Model 的簡化例子。

假設：

- Parameters：7B。
    
- Weights：BF16，2 bytes / parameter。
    
- Gradients：BF16，2 bytes / parameter。
    
- Adam Moment States：FP32，兩份各 4 bytes / parameter。
    

|項目|理論記憶體|
|---|---|
|Model Weights|14 GB|
|Gradients|14 GB|
|Adam First Moment|28 GB|
|Adam Second Moment|28 GB|
|合計|84 GB|

實際系統可能還需要 FP32 Master Weights，以及 Activation、Communication Buffer 和其他記憶體，所以可能明顯高於 84 GB。

反過來，不同 Optimizer、Precision、Sharding 和 Offloading 設計，也能改變上述配置。

這解釋了：

7B Model 可以用遠小於 84 GB 的記憶體進行某些推論，但 Full Fine-tuning 往往需要更多。

### 18.2 DDP：Distributed Data Parallel

DDP 的方法是：

每個 GPU 保存完整的 Model Replica，但讀取不同的 Training Data。

GPU 0

Full Model

Data Shard 0

GPU 1

Full Model

Data Shard 1

GPU 2

Full Model

Data Shard 2

Gradient All-reduce

聚合各 GPU 梯度，讓副本保持同步

Synchronized Optimizer Update

例如：

GPU 0 讀取資料 A；GPU 1 讀取資料 B；GPU 2 讀取資料 C。

各自：

1. Forward。
    
2. Compute Loss。
    
3. Backward。
    
4. Synchronize Gradients。
    
5. Optimizer Step。
    

PyTorch DDP 主要利用 Distributed Collectives 同步 Gradient，並透過一個 Process per GPU 的常見架構執行。

![](https://www.google.com/s2/favicons?domain=https://docs.pytorch.org&sz=32)

PyTorch Tutorials 2.14.0+cu130 documentation

限制： 每個 GPU 通常仍需要保存完整的 Model Parameters 及相應 Optimizer States。

所以 DDP 很適合模型能放進單 GPU，但希望提高訓練吞吐量的情況。

### 18.3 FSDP：Fully Sharded Data Parallel

FSDP 的目標是解決：

單 GPU 無法有效容納完整訓練狀態。

它會將 Model Parameters、Gradients、Optimizer States 分散到多個 GPU。

DDP 與 FSDP 的記憶體配置概念

DDP

每個 GPU 皆保存完整模型訓練狀態

FSDP

持久訓練狀態跨 GPU 分片

Shard A

Shard B

Shard C

Shard D

概念圖；FSDP 在實際 Forward／Backward 時仍需暫時 All-gather 所需的 Parameters。

FSDP 的典型操作：

Forward Pass 前：

All-gather 當前 Layer 所需的 Parameters。

Forward 完成後：

可以釋放或重新分片不再需要的完整權重副本。

Backward：

再次取得必要權重，計算 Gradients，再透過 Reduce-scatter 分配。

Optimizer Step：

各 GPU 更新自己所管理的 Parameter Shards。

PyTorch 的 FSDP2 使用 `fully_shard` 和 DTensor-based Parameter Sharding，並支援更細緻的 Layer-level Grouping。

![](https://www.google.com/s2/favicons?domain=https://docs.pytorch.org&sz=32)

PyTorch 2.14 documentation

### 18.4 ZeRO：Zero Redundancy Optimizer

DeepSpeed ZeRO 將減少冗餘訓練狀態分成三個典型階段：

|ZeRO Stage|分片內容|主要效果|
|---|---|---|
|Stage 1|Optimizer States|降低 Optimizer Memory|
|Stage 2|Optimizer States + Gradients|進一步降低記憶體|
|Stage 3|Optimizer States + Gradients + Parameters|讓更大型的模型能進行訓練|

DeepSpeed 官方文件定義了這些階段，也支援 CPU／NVMe Offload 等擴展方式。

![](https://www.google.com/s2/favicons?domain=https://www.deepspeed.ai&sz=32)

DeepSpeed

### 18.5 DDP、FSDP、ZeRO 的差異

|技術|每 GPU 完整權重|主要瓶頸|適合情境|
|---|---|---|---|
|DDP|通常有|Gradient Synchronization|模型可放入單 GPU，想提高吞吐量|
|FSDP|持久狀態通常分片|All-gather / Reduce-scatter|Full Fine-tuning 大模型|
|ZeRO-1|有|Optimizer State 分片通訊|Optimizer Memory 壓力|
|ZeRO-2|有|Gradient 與 Optimizer 通訊|需要更低訓練記憶體|
|ZeRO-3|持久狀態分片|Parameter Gathering|大模型訓練|

FSDP 和 ZeRO-3 有高度相似的核心目標，但 Framework Integration、API、Offloading、Checkpoint Management 和 Performance Tuning 可以不同。

### 18.6 Checkpointing：其實有兩種不同意思

A. Activation Checkpointing / Gradient Checkpointing

訓練時不保存某些中間 Activations，Backward 時重新執行部分 Forward Computation。

效果：

- 降低 Activation Memory。
    
- 增加重新計算量。
    
- 可能降低 Training Throughput。
    
- 讓較長 Context 或較大 Batch 變得可行。
    

B. Training Checkpointing

定期將訓練狀態保存到磁碟或 Object Storage。

理想情況下，Checkpoint 不只保存 Model Weights，也保存：

- Optimizer State
    
- Learning Rate Scheduler
    
- Global Step
    
- Mixed-precision Scaler（若使用）
    
- Random Number Generator State
    
- Data Sampler / Dataloader Progress
    
- Config、Dataset Version、Tokenizer Version
    

在大型 Distributed Training 中，通常使用 Sharded Checkpoint，避免每次都集中保存完整模型副本。

PyTorch Distributed Checkpoint（DCP）支援分散式儲存與載入，並可以處理不同 World Size 的重新分片需求。

![](https://www.google.com/s2/favicons?domain=https://docs.pytorch.org&sz=32)

PyTorch Tutorials 2.14.0+cu130 documentation

### 18.7 Engineer 如何選擇 Distributed Strategy？

|實際情境|建議的起點|
|---|---|
|7B Model，單 GPU 且記憶體有限|QLoRA|
|7B Model，單 GPU 有足夠記憶體|LoRA|
|模型放得進每張 GPU，想加速訓練|DDP|
|Full Fine-tuning 超出單 GPU 記憶體|FSDP / ZeRO-3|
|GPU Memory 充足但 Activations 很大|Activation Checkpointing|
|多機長時間訓練|Distributed Checkpoint + Fault Recovery|
|數百億以上模型且跨多節點|視架構再搭配 Tensor / Pipeline / Expert Parallelism|

注意，DDP、FSDP、ZeRO 不是所有分散式策略的全部。超大模型可能還需要：

- Tensor Parallelism：拆分 Layer 內的矩陣運算。
    
- Pipeline Parallelism：拆分不同 Layers 到不同設備。
    
- Context / Sequence Parallelism：改善長序列訓練。
    
- Expert Parallelism：針對 MoE Expert 分散運算。
    

### 18.8 Distributed Training 常見問題

Senior Engineer 應理解：

- Out-of-memory： 原因可能是 Parameters、Activations、Long Sequences 或短暫 Allocation Peaks。
    
- Slow All-reduce： 可能是 GPU Interconnect、Network Bandwidth、Gradient Bucketing 問題。
    
- Straggler： 某個 GPU 處理較慢，讓其他 GPU 等待。
    
- NCCL Timeout： 可能是設備、通訊、Process Failure 或 Collective 呼叫次序不一致。
    
- Non-determinism： Sampling、Distributed Reduction、Kernel 等造成結果難以完全重現。
    
- Failed Resume： Checkpoint 沒有保存完整 Optimizer／Sampler State。
    
- Low GPU Utilization： Data Loading、Tokenization、CPU Bottleneck、Synchronization、Kernel Launch 等可能造成 GPU 等待。
    

不能只因為 GPU Utilization 低，就假設模型太小或應該增加 GPU。

# Part III — 完整實務案例：從企業需求到 Production Post-training

以下以一個實際工程場景貫穿前面所有技術。

假設一家工業 AI 公司正在開發 Industrial Inspection AI Agent。

系統包含：

- 多台 Camera、Lighting、Motion Control。
    
- Computer Vision Detection / Segmentation。
    
- Autofocus 與 Image Quality Assessment。
    
- Equipment Error Logs。
    
- Python Control Software。
    
- Cloud Data Storage。
    
- 一個可以協助工程師診斷故障、解釋分析結果、呼叫診斷工具的 LLM Agent。
    

這個例子可以清楚展示 Applied AI Engineer 與 Model Research Engineer 實際會做的工作有何不同。

## 19. Phase 0：先確認是不是真的需要 Fine-tuning

這是 Senior Applied AI Engineer 最重要的技術決策之一。

假設需求如下：

|需求|優先考慮的方案|原因|
|---|---|---|
|查詢最新設備 Manual|RAG|文件需要持續更新|
|回答標準維修問題|Prompt + RAG|可能不需要訓練|
|固定 JSON 輸出格式|Structured Output；必要時 SFT|先使用低成本方案|
|學會特定 Tool-call Pattern|SFT|需要大量一致的示範|
|專家偏好某種診斷方式|DPO|有 Chosen / Rejected Data|
|學會多步驟診斷策略|SFT + Evals；必要時 RL|可用執行結果驗證|
|降低大型模型使用成本|Distillation|將能力轉移至小模型|
|Full Fine-tuning 記憶體不足|FSDP / ZeRO|降低分散式訓練狀態冗餘|

這裡有一條重要原則：

Fine-tuning 通常不應該取代 RAG、Tool Calling、程式化驗證或安全控制。

例如設備的 Safety Limits、最新硬體設定與 Firmware Version，較適合從可信賴的 Configuration Database 或 Documentation 取得。

即使 Fine-tuned Model 說某個動作安全，也不能讓它取代硬體層的 Interlock 或確定性的安全檢查。

## 20. Phase 1：建立完整的資料與訓練架構

Production Data Sources

Equipment Logs · Tool Results · Expert Diagnoses · Manuals · Historical Incidents

Dataset Preparation

Cleaning · PII Removal · Deduplication · Label Review · Train/Test Isolation

Instruction

SFT Dataset

Preferences

DPO / RM Dataset

Tasks + Verifiers

RL Dataset

Model Training & Experiment Tracking

SFT / LoRA / QLoRA / DPO / RL / Distillation

Evaluation & Release Gate

Correctness · Task Success · Safety · Regression · Latency · Cost

Production Agent

LLM + RAG + Read-only Diagnostic Tools + Validated Actions

### 20.1 建立三種 Dataset

SFT Dataset： 專家提供輸入與理想輸出。

DPO Dataset： 對同一個問題比較兩個答案，標出較佳及較差者。

RL Dataset： 提供問題、環境與獨立驗證方式，不一定預先提供完整解答。

例如，一個異常診斷案例可以衍生出：

|Dataset|內容|
|---|---|
|SFT|故障訊息 → 專家建議的診斷步驟|
|DPO|專家安全診斷 vs 未經驗證就重新移動|
|RL|在模擬環境選擇診斷工具，依故障定位結果給 Reward|

### 20.2 防止資料洩漏

假設同一台設備的同一個故障事件包含：

- 20 筆 Logs。
    
- 5 筆 Tool Execution Results。
    
- 3 個專家分析版本。
    
- 10 個變體 Prompts。
    

不能隨機把這些高度相關樣本分散進 Train 和 Test。

否則 Test Performance 可能被嚴重高估。

應按 Incident ID、Equipment ID、Customer／Site、時間區間或相似案例群組 做適當拆分，並另外準備真正未見過的故障案例。

## 21. Phase 2：實作第一個 SFT + LoRA Model

下面是可以作為小規模實驗起點的 Python 範例。

使用：

- PyTorch
    
- Transformers
    
- PEFT
    
- Hugging Face TRL
    
- Hugging Face Datasets
    

TRL 已提供 SFT、DPO、GRPO、Reward Modeling 等 Trainer，以及 PEFT Integration。

![](https://www.google.com/s2/favicons?domain=https://github.com&sz=32)

GitHub

### 21.1 安裝環境

```
pip install torch transformers datasets accelerate peft trl
```

真實專案需要固定相容的套件版本與 CUDA 環境。

### 21.2 建立 SFT Training Script

```
from datasets import Datasetfrom peft import LoraConfigfrom trl import SFTConfig, SFTTrainer# Illustrative training examplestrain_data = Dataset.from_list([    {        "prompt": [            {                "role": "system",                "content": "You are an industrial diagnostic assistant."            },            {                "role": "user",                "content": "The motor reports a stall fault."            }        ],        "completion": [            {                "role": "assistant",                "content": (                    "Stop further movement. Inspect the fault "                    "status and mechanical conditions. "                    "Follow the approved recovery procedure."                )            }        ]    },    {        "prompt": [            {                "role": "user",                "content": "The captured image is blurry."            }        ],
```

這段程式展示：

1. 使用已存在的 Pretrained / Instruct Model。
    
2. 建立 Conversational SFT Dataset。
    
3. 設定 LoRA Rank。
    
4. 只對 Completion 計算 Loss。
    
5. 執行 Fine-tuning。
    
6. 保存 Adapter／Training Artifacts。
    

注意：這是 API 教學範例，不是已驗證的 Production Training Script。 兩筆示範資料不足以訓練有效模型，而且實際訓練還需要 Validation Dataset、Evaluation、Checkpoint Policy、Seed、Experiment Tracking 與安全測試。相容的 Chat Template、模型授權及 VRAM 需求也要確認。

### 21.3 真正開始訓練後要觀察什麼？

至少記錄：

```
train/loss
eval/loss
train/learning_rate
train/grad_norm
train/tokens_per_second
system/gpu_memory
system/gpu_utilization
evaluation/task_success
evaluation/format_validity
evaluation/safety_violation
```

如果 Training Loss 下降但 Task Success 沒有提高，應該檢查資料品質、Loss Mask、Train/Test Split、模型容量及目標定義，而不是直接增加 Epochs。

## 22. Phase 3：使用 DPO 改善專家偏好

當 SFT Model 已經可以回答基本問題後，可以收集 Preference Dataset。

### 22.1 Example Dataset

```
preference_data = [    {        "prompt": "The motor reports a stall fault.",        "chosen": (            "Stop further movement, inspect fault state, "            "and verify mechanical conditions."        ),        "rejected": (            "Clear the alarm and repeat the movement "            "without inspection."        )    }]
```

這裡只是顯示欄位概念，實際應整理成 DPOTrainer 支援的 Conversational 或 Prompt-completion Preference Schema。

### 22.2 DPO Trainer 概念

```
from trl import DPOConfig, DPOTrainertraining_args = DPOConfig(    output_dir="./industrial_dpo",    learning_rate=5e-6,    per_device_train_batch_size=1,    gradient_accumulation_steps=8,    beta=0.1)trainer = DPOTrainer(    model=sft_model,    ref_model=reference_model,    args=training_args,    train_dataset=preference_dataset)trainer.train()
```

其中 `sft_model`、`reference_model` 與 `preference_dataset` 必須事先正確建立；這段是 Trainer Integration 的示意。

使用 LoRA 時，也可以透過 PEFT 管理 Reference Policy，不一定要保留另一個完整權重副本。

### 22.3 實驗該怎麼做？

建立三個模型：

|Model|Training|
|---|---|
|A|Base / Instruct Model|
|B|SFT + LoRA|
|C|SFT + LoRA + DPO|

對相同的 Held-out Test Prompts 進行比較。

主要觀察：

- C 是否比 B 更符合 Expert Preferences？
    
- C 是否提高 Diagnostic Correctness？
    
- C 是否造成回答過長或過度保守？
    
- C 是否破壞原本良好的 Tool-calling Behavior？
    
- C 的安全錯誤率是否下降？
    

不能因為 C 的 Preference Win Rate 提高，就直接認定它更適合 Production。

## 23. Phase 4：什麼情況下進一步導入 Reasoning RL？

假設我們希望 AI Agent 具備以下能力：

```
Input:
  Autofocus failure

Agent:
  Query device status
       ↓
  Retrieve last capture metrics
       ↓
  Check diagnostic constraints
       ↓
  Select next diagnostic action
       ↓
  Evaluate tool result
       ↓
  Generate diagnosis
```

這個任務比單純產生文字答案更複雜。

如果有可靠模擬器與 Verifier，可以考慮 Reasoning RL。

### 23.1 Environment Design

建立一個可控制的模擬環境：

```
class DiagnosticEnvironment:    def reset(self):        # Create or load a diagnostic scenario        ...    def step(self, action):        # Execute simulated diagnostic action        ...        observation = ...        terminated = ...        reward = ...        return observation, reward, terminated
```

Production 硬體不應直接用來讓 RL Agent 無限制探索；應先使用可重現的 Simulator、Replay Environment 或具備嚴格安全限制的 Test Harness。

### 23.2 Reward Design

例如：

\[ R= w_1R_{\text{diagnosis}} +w_2R_{\text{tool}} +w_3R_{\text{completion}} -w_4P_{\text{unsafe}} \]

其中：

- \(R_{\text{diagnosis}}\)：根因診斷是否正確。
    
- \(R_{\text{tool}}\)：工具使用是否有效。
    
- \(R_{\text{completion}}\)：是否完成任務。
    
- \(P_{\text{unsafe}}\)：不安全動作懲罰。
    

但這種 Weighted Reward 有風險。

如果某個不安全行為可以大幅提高 Completion Reward，模型可能仍然選擇它。

因此，對關鍵安全條件，較好的方式是：

在環境和工具權限層直接禁止不安全行為，再使用 Reward 最佳化允許範圍內的策略。

### 23.3 什麼時候值得做 RL？

可以用以下條件評估：

- 是否存在可重現的 Environment？
    
- 是否有客觀且可靠的 Reward？
    
- 是否需要模型探索多種行動序列？
    
- SFT / DPO 是否已達到性能瓶頸？
    
- 是否有足夠的 Training Compute？
    
- 是否有 Offline Evaluation 與安全的 Rollout Infrastructure？
    

如果只是幾十個固定診斷步驟，用 State Machine + Rules + LLM Tool Selection，可能比 RL 更便宜、更可維護，也更容易驗證。

## 24. Phase 5：Distillation、部署與成本最佳化

假設完成 SFT / DPO 後，大模型的品質很好，但在 Production 速度較慢。

此時：

1. 選定能力較強的 Teacher。
    
2. 使用 Teacher 產生多樣化 Task Responses。
    
3. 排除不正確、不安全或低品質案例。
    
4. 以 SFT 或更進階 Distillation Objective 訓練 Student。
    
5. 比較 Student 與 Teacher 的完整能力。
    
6. 決定是否使用 Small-model-first Routing。
    

### 24.1 Production 不只要比較 Token Speed

更重要的是：

\[ \text{Effective Cost per Successful Task} = \frac{\text{Total Serving Cost}} {\text{Number of Successfully Completed Tasks}} \]

某個小模型即使每個 Token 比較便宜，如果經常需要重試，可能未必能達到最低的任務成本。

### 24.2 Production Monitoring

至少要追蹤：

|層次|指標|
|---|---|
|Model Quality|Task Success、Correctness、Unsupported Claims|
|Preference|Expert Win Rate、Preference Consistency|
|Agent|Tool-call Success、Invalid Actions、Recovery Rate|
|Safety|Unsafe Action Attempt、Access Violation|
|Reliability|Timeout、Retry、Service Failure|
|Performance|p50 / p95 Latency、Tokens/s|
|Cost|GPU Hours、Serving Cost、Cost per Successful Task|
|Data Drift|新設備、新故障類型、新軟體版本|

部署順序建議：

```
Offline Evaluation
       ↓
Shadow Deployment
       ↓
Limited Canary
       ↓
Production Monitoring
       ↓
Gradual Rollout
       ↓
Rollback if Regression
```

每一個模型版本都應保留：

`Base Model ID + Adapter ID + Dataset Version + Training Config + Evaluation Report + Deployment Version`

這樣發生 Regression 時才有能力追蹤與恢復。

# Part IV — 2026 Senior Engineer 應具備的決策與面試能力

## 25. 如何快速判斷應該使用哪種技術？

互動式技術選擇練習

選擇目前專案遇到的主要問題：

公司文件持續更新，希望模型能回答最新內容

模型不會遵守固定輸出格式或工具呼叫規範

有專家比較過兩個回答的好壞

需要在可驗證環境中學會多步驟策略

想用有限 GPU 記憶體微調大型開放模型

模型品質已足夠，但 Serving 成本太高

模型完整訓練狀態無法放入單 GPU

推薦的起始技術方案

## DPO

已有 chosen / rejected preference 資料，DPO 通常是值得優先驗證的低複雜度方案。

深入練習這個技術決策

## 26. 不同職位需要掌握到什麼深度？

|技術|Senior Applied AI Engineer|Senior LLM / Research Engineer|Training Infrastructure Engineer|
|---|---|---|---|
|SFT|能設計資料、訓練與評估|能研究與修改 Loss / Objective|能優化 Training Pipeline|
|LoRA / QLoRA|能實作與調整|理解低秩假設與限制|優化 Memory / Kernels|
|DPO|能建立偏好資料並訓練|能推導 Loss 與比較變體|優化大規模 Training|
|RLHF|理解適用情境|能實作 RM / PPO / Online Training|管理 Rollouts、Policy、GPU|
|Reasoning RL|理解 Verifier 與使用條件|能設計 Rewards 與 Policy Optimization|建立 Rollout / Training Infrastructure|
|Distillation|能做 Teacher–Student 實驗|能研究 KD Objective|優化 Student Training|
|DDP / FSDP / ZeRO|能使用現有工具|需要理解大型實驗配置|必須深入 Debug / Optimize|

## 27. Senior / Staff 面試常見追問與回答重點

### Q1. Why would you choose LoRA instead of full fine-tuning?

完整回答應涵蓋：

- Full Fine-tuning 更新全部或大部分參數，記憶體與 Optimizer Cost 高。
    
- LoRA 凍結 Base Weights，只訓練低秩 Adapter。
    
- 訓練成本下降，也容易維護多個 Task-specific Adapters。
    
- LoRA 並不保證與 Full Fine-tuning 一樣好。
    
- 應透過 Rank、Target Modules、Domain Shift 和 Held-out Results 選擇方案。
    

### Q2. How would you decide between SFT and DPO?

應先判斷現有資料類型：

- 高品質 Demonstrations → SFT。
    
- Chosen / Rejected Preferences → DPO。
    
- 兩者都有 → 可以先 SFT，再 DPO。
    
- 行動結果需要 Online Interaction → 評估 RL。
    

還要討論 Dataset Bias、Reference Policy、Objective Alignment 與真實任務評估。

### Q3. Why not use RLHF for every task?

因為 RLHF 需要可靠 Reward、Online Generation、更多計算資源，以及複雜的 Policy Optimization Infrastructure。

如果任務只需要固定 JSON、標準 Tool Calls 或良好的答案格式，SFT 甚至 Structured Outputs 可能已經足夠。

### Q4. Training loss decreases, but model performance becomes worse. Why?

可能原因：

- Overfitting。
    
- Train / Evaluation Distribution Mismatch。
    
- Catastrophic Forgetting。
    
- Incorrect Loss Mask。
    
- Dataset Label Noise。
    
- Preference / Reward Objective 與實際任務不一致。
    
- Model Configuration 或 Inference Template 不一致。
    

Senior Engineer 需要透過 Slice-based Evaluation 和 Error Analysis 找出原因，而不是盲目延長訓練。

### Q5. Would you use DDP or FSDP?

先分析：

- 模型及 Optimizer States 是否能放入單 GPU？
    
- 訓練是否受 Memory 或 Computation 限制？
    
- GPU Interconnect Bandwidth 如何？
    
- 是否有多節點？
    
- Activation Memory 是否才是真正瓶頸？
    

若能放入單 GPU，DDP 常是比較簡單的起點；若完整訓練狀態無法容納，FSDP / ZeRO 會更有吸引力。

### Q6. How would you prove that DPO or RL really improved the model?

需要：

1. Frozen Independent Test Set。
    
2. 明確的 Primary Task Metric。
    
3. Human / Expert Preference Evaluation。
    
4. SFT-only / DPO / RL Ablation。
    
5. Confidence Intervals 或其他適當的不確定性分析。
    
6. Regression / OOD / Safety Tests。
    
7. Production Shadow 或 Canary Validation。
    

不能只用 Training Reward、Reward Model Score 或 LLM Judge Score 證明成功。

## 28. 建議的實作學習順序

如果希望從理解概念，進一步具備 Senior LLM Engineer 的實際操作能力，我會建議依照以下順序完成實驗：

1

SFT + LoRA

找一個約 0.5B–3B 的開放模型，建立小型指令資料，訓練 Adapter，確認 Masking、Loss、Checkpoint 與 Evaluation。

2

QLoRA / Memory Profiling

在相同 Dataset 上比較 LoRA、QLoRA 的 Peak VRAM、Step Time、Tokens/s 與 Accuracy。

3

DPO

建立 Preference Pairs，以同一個 SFT Model 為起點，量測 Preference Win Rate 和 Task Regression。

4

Reward Model + GRPO

選擇數學或 Unit-test 可驗證任務，觀察 Reward、KL、Exploration 與 Failures。

5

Distillation

用較大 Teacher 產生已驗證的回答，訓練小 Student，比較品質、成本與延遲。

6

DDP / FSDP / ZeRO

在多 GPU 環境比較 Memory、Communication、Throughput、Checkpoint / Resume。

7

End-to-end Production

將模型部署成 API 或 Agent，建立 Offline Evals、Monitoring、Canary 與 Rollback。

## 29. 最值得閱讀的原始論文與官方文件

|主題|論文／文件|
|---|---|
|SFT / RLHF|[Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)|
|LoRA|[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)|
|QLoRA|[QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)|
|DPO|[Direct Preference Optimization](https://arxiv.org/abs/2305.18290)|
|GRPO|[DeepSeekMath](https://arxiv.org/abs/2402.03300)|
|Reasoning RL|[DeepSeek-R1](https://arxiv.org/abs/2501.12948)|
|Distillation|[Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)|
|PyTorch FSDP2|[PyTorch FSDP2 Documentation](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)|
|DeepSpeed ZeRO|[DeepSpeed ZeRO Tutorial](https://www.deepspeed.ai/tutorials/zero/)|
|Training Framework|[Hugging Face TRL Documentation](https://huggingface.co/docs/trl)|

## 最後總結：Senior AI Engineer 最需要建立的能力

Fine-tuning / Post-training 的真正工程價值，不在於知道越多演算法名稱越好，而在於能夠回答下列問題：

第一：是否真的需要訓練？

能否先透過 Prompt、RAG、Structured Outputs、Rules 或 Tools 解決？

第二：需要什麼學習訊號？

是有標準答案的 SFT、有人類比較的 DPO，還是有可驗證環境回饋的 RL？

第三：如何控制訓練成本？

應採用 Full Fine-tuning、LoRA、QLoRA、DDP、FSDP，還是 ZeRO？

第四：如何證明模型真的進步？

除了 Loss，還必須驗證 Accuracy、Task Success、Safety、Generalization、Regression、Latency 和 Cost。

第五：如何可靠地交付到 Production？

需要具備資料版本化、模型版本化、持續評估、可觀測性、逐步部署，以及安全 Rollback。

對 Senior Applied AI Engineer 而言，最重要的是能將以上技術正確地組合成實際可維護的產品。

對 Senior LLM Research Engineer / Foundation Model Engineer 而言，還必須深入理解 Objective Functions、Gradient Dynamics、Optimization Stability、Scaling、Reward Design，以及在大量 GPU 上實作與改進演算法。

這兩類職位需要的基礎相同，但真正的技術深度與責任範圍並不相同。