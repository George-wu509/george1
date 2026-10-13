
|                                 |     |
| ------------------------------- | --- |
| [[#### 建立MiniLLM 並實際跑通完整訓練]]    |     |
| [[#### 解釋這個MiniLLM repo的檔案結構]]  |     |
| [[#### 詳細解釋parameters的意義及執行順序]] |     |
| [[#### 怎麼訓練 Tokenizer]]         |     |
|                                 |     |
|                                 |     |
|                                 |     |

#### 建立MiniLLM 並實際跑通完整訓練
```
我想在 Laptop 上建立一套完整的 Mini LLM在C:\LLM\MiniLLM  
從零預訓練 → 後訓練 → 微調 → 推理.  
要從 GitHub 的 MiniMind 開始，搭配 nanoGPT 與 Hugging Face TRL，建立一套能在 laptop 實際運作的 LLM Training Laboratory。  
目標不是單純下載一個小模型並與它聊天，而是親手操作完整的 LLM Lifecycle，包括：

- 從零建立 Transformer Model，理解 Pretraining 前的模型如何初始化。
- 準備 Raw Dataset，進行 Cleaning、Tokenization、Train/Validation Split。
- 執行 Pretraining，觀察 Loss、Gradient、Optimizer 與 Checkpoint。
- 執行 Supervised Fine-Tuning（SFT），讓 Base Model 學習遵循指令。
- 執行 LoRA Fine-Tuning，理解 Parameter-Efficient Fine-Tuning。
- 執行 DPO Preference Alignment，理解 Post-training。
- 執行 Inference，觀察 Temperature、Top-p、KV Cache 等設定。
- 保存所有 Dataset、Model Weights、Tokenizer、Configuration、Metrics，並比較各階段模型的差異。

要建立的完整 Pipeline  
Raw Text Dataset  
TXT / JSONL / Parquet

Data Processing  
Clean · Deduplicate · Split

Tokenizer Training  
Vocabulary · Token IDs

STAGE 1  
Pretraining from Scratch  
Random Weights → Next-token Prediction  
輸出：Base Model Checkpoint

STAGE 2  
Supervised Fine-Tuning (SFT)  
Instruction / Response Dataset  
輸出：Instruction Model

STAGE 3  
Preference Alignment (DPO)  
Prompt · Chosen · Rejected  
輸出：Aligned Model

STAGE 4 · OPTIONAL BRANCH  
Domain Adaptation (LoRA)  
Base/SFT Model + Domain Dataset  
輸出：LoRA Adapter

Evaluation → Inference → Deployment  
Metrics · Generation · API · Monitoring

注意：這是一條方便教學的主要流程，SFT、LoRA、DPO 不一定只能依照圖中的順序進行。LoRA 是參數更新方法，可以應用於 SFT 或其他訓練階段；DPO 通常從已完成 SFT 的模型開始。  
在每個階段，你都應保存獨立的 Model Checkpoint 與 Evaluation Results，才能清楚觀察每一種訓練對模型產生的影響
```
# 在 Laptop 上建立一套完整的 Mini LLM：從零預訓練 → 後訓練 → 微調 → 推理

我最推薦你從 GitHub 的 MiniMind 開始，搭配 nanoGPT 與 Hugging Face TRL，建立一套能在 laptop 實際運作的 LLM Training Laboratory。

你的目標不是單純下載一個小模型並與它聊天，而是親手操作完整的 LLM Lifecycle，包括：

- 從零建立 Transformer Model，理解 Pretraining 前的模型如何初始化。
    
- 準備 Raw Dataset，進行 Cleaning、Tokenization、Train/Validation Split。
    
- 執行 Pretraining，觀察 Loss、Gradient、Optimizer 與 Checkpoint。
    
- 執行 Supervised Fine-Tuning（SFT），讓 Base Model 學習遵循指令。
    
- 執行 LoRA Fine-Tuning，理解 Parameter-Efficient Fine-Tuning。
    
- 執行 DPO Preference Alignment，理解 Post-training。
    
- 執行 Inference，觀察 Temperature、Top-p、KV Cache 等設定。
    
- 保存所有 Dataset、Model Weights、Tokenizer、Configuration、Metrics，並比較各階段模型的差異。
    

這套流程可以在沒有昂貴 GPU 的情況下學習，但必須依照 laptop 的記憶體及 GPU 選擇模型規模。

## 一、我推薦的 5 個 GitHub Repository

![MiniMind: 단돈 3달러, 2시간 만에 64M GPT를 처음부터 학습시키며 학습하는 오픈소스 초경량 언어 모델 프로젝트 - 읽을거리&정보공유 - PyTorchKR](https://images.openai.com/static-rsc-4/PsDYZluQjVYNo7FWLu0S0UjLejQSYoe8BuEN0MRoRtQsi5a0kQyz6vQHBIoMPenNBeGiKh0EtF-mDQOa1yayY4EAw6DnARB3RnhUqamUo9fNzTO6Tr8oQt-OJerhLw-AuSxFGj-JSFl7Y_QKiZsIjBMjYDN9EKVdMCjgER7yyLM?purpose=inline)

1. [MiniMind](https://github.com/jingyaogong/minimind)

最推薦

完整從零訓練小型 LLM 的教學專案。 包含模型架構、Tokenizer、Pretraining、SFT、LoRA、DPO、RL 及 Inference。2026 年版本提供 64M 參數的 MiniMind-3，也有 26M 的舊版小模型，非常適合練習完整 pipeline。

![](https://www.google.com/s2/favicons?domain=https://github.com&sz=32)

GitHub

+1

![GitHub - karpathy/nanoGPT: The simplest, fastest repository for training/finetuning medium-sized GPTs. · GitHub](https://images.openai.com/static-rsc-4/RVT1IBk3lxmu5t9MG8JYlKPLzl1f_RMTBmZ0n1s1-mZSkzM6FLHFrsEsvs35Imek28lOlvyWGcesRcHol7LBkITV11cVWnCE9BhC6NRVAVZCi0ohx8dgBTwDqUAYiuNsH5zXQz9vrWav9lB_oc4oIoMpaZs9fcYD5g39zVKiBVU?purpose=inline)

2. [nanoGPT](https://github.com/karpathy/nanoGPT)

最適合理解 Transformer 和 Pretraining 的底層原理。 程式架構簡單，可以把模型縮到數百萬參數，在 CPU 上訓練。提供 Tiny Shakespeare Dataset、Training Script 與 Inference Script。原作者已將它定位為舊版教學專案。

![](https://www.google.com/s2/favicons?domain=https://github.com&sz=32)

GitHub

+1

![GitHub - karpathy/nanochat: The best ChatGPT that $100 can buy.](https://images.openai.com/static-rsc-4/6wcn_cIzzyvtu_3cNBYElTeFDP0qwbRk7bdrN0Y8McgPi5B1KTXixxqBPs20GhOdwRmP2sNoAi5hl0ZcTNGPMYFKNO_TYXzdzThy51_cQ0txnzYf42_DIueIkRI4Io7ygbec5TB0ilmFxh6mbLTsDa01HyeTpOsRFMq9pzBMdkk?purpose=inline)

3. [nanochat](https://github.com/karpathy/nanochat)

更現代的端到端 LLM Training Framework。 涵蓋 Tokenization、Pretraining、Finetuning、Evaluation、Inference，並能研究 Scaling Laws。官方主要 benchmark 針對高階 GPU，普通 laptop 應縮小模型及資料規模。

![](https://www.google.com/s2/favicons?domain=https://github.com&sz=32)

GitHub

+1

![HuggingFaceTB/SmolLM2-1.7B · Hugging Face](https://images.openai.com/static-rsc-4/NnVrlY_63w6L9PugEWRu5yo7e4cXQY_WM9PWm6cTQRzPHeXTrkls5d8R-Ro93fTalpwyLxFgEMbiLB5OOEQ15ci77ZpF7hY-b_e-ATUaJ2dEsvGN0wG-tMHpTw-VJKPI-OnCITgm6-xL9blRjjRaRPR7JThuON6sGBLfrtYn5-0?purpose=inline)

4. [SmolLM / SmolLM2](https://github.com/huggingface/smollm)

最適合學習如何使用現成 Pretrained Model。 具有 135M、360M 與 1.7B 等模型版本，可以練習 Load Weights、SFT、Evaluation、Inference，也提供訓練相關程式碼。

![](https://www.google.com/s2/favicons?domain=https://github.com&sz=32)

GitHub

![Hands-On Exercises: Fine-Tuning SmolLM3 - Hugging Face a smol course](https://images.openai.com/static-rsc-4/oZPAr3fFKwn09_5A3zxqSRP7RyKiFvJsPK5KINtHqLTy6ny8I6E7J5eYtJFwVWgoJOkJeM6G9P4JV96vhwWo2TQrnL_3JsI_IDn1GZictOXv4aaKbkNiXe78npf9jCcNnmgkmH00AG3dr017Ydr9fp921GjAD0GUVtH6Q5qF4cE?purpose=inline)

5. [Hugging Face Smol Course](https://github.com/huggingface/smol-course)

最適合學習現代 Post-training。 涵蓋 SFT、Evaluation、DPO、RL 與 Synthetic Data，可與 Hugging Face Transformers、PEFT、TRL 配合。部分課程預設較大模型，需要縮小規模才能在普通 laptop 訓練。

![](https://www.google.com/s2/favicons?domain=https://github.com&sz=32)

GitHub

+1

### 如何選擇？

|你的學習目標|最佳 Repo|
|---|---|
|看懂並修改 Transformer、Attention、Loss、Backpropagation|nanoGPT|
|親手從 Random Weights 訓練出 Chat Model|MiniMind|
|學習較現代的 Pretraining Pipeline|nanochat|
|使用已預訓練模型練習 Fine-tuning|SmolLM2|
|熟悉 SFT、DPO、PEFT、Alignment|Smol Course|

我的建議是：

主要專案使用 MiniMind，第一個練習使用 nanoGPT，後續再把 Hugging Face TRL 整合進來。

這樣既可以看到完整訓練流程，也能理解真實企業 LLM Engineer 使用的軟體工具。

## 二、你要建立的完整 Pipeline

Raw Text Dataset

TXT / JSONL / Parquet

Data Processing

Clean · Deduplicate · Split

Tokenizer Training

Vocabulary · Token IDs

STAGE 1

Pretraining from Scratch

Random Weights → Next-token Prediction

輸出：Base Model Checkpoint

STAGE 2

Supervised Fine-Tuning (SFT)

Instruction / Response Dataset

輸出：Instruction Model

STAGE 3

Preference Alignment (DPO)

Prompt · Chosen · Rejected

輸出：Aligned Model

STAGE 4 · OPTIONAL BRANCH

Domain Adaptation (LoRA)

Base/SFT Model + Domain Dataset

輸出：LoRA Adapter

Evaluation → Inference → Deployment

Metrics · Generation · API · Monitoring

注意：這是一條方便教學的主要流程，SFT、LoRA、DPO 不一定只能依照圖中的順序進行。LoRA 是參數更新方法，可以應用於 SFT 或其他訓練階段；DPO 通常從已完成 SFT 的模型開始。

在每個階段，你都應保存獨立的 Model Checkpoint 與 Evaluation Results，才能清楚觀察每一種訓練對模型產生的影響。

## 三、Laptop 硬體需要多強？

要先區分兩個任務：

1. 訓練一個小模型以了解演算法：1M–30M 參數就很有學習價值。
    
2. 訓練一個可以正常回答各種問題的 LLM：即使只有 100M–500M 參數，也需要相對大量的高品質資料與運算。
    

你現在的目標是第一種，因此不需要一開始就使用 1B、7B 或 8B 模型。

依 Laptop 的 GPU 選擇模型

CPU only

4 GB VRAM

8 GB VRAM

16+ GB VRAM

建議：1M–10M 參數的 Tiny GPT使用 nanoGPT 的小模型設定。能操作 Pretraining、Checkpoint、Inference；小資料 SFT、DPO 可以嘗試，但訓練會比較慢。

以上是保守的起始實驗規模，不是硬體上限或保證能完整訓練的規格。實際可行性受到 RAM、序列長度、Batch Size、Optimizer、訓練精度與 Windows 環境影響。

在 Windows 11 可以先使用 PowerShell 查詢硬體：

```
# CPU 與記憶體
Get-CimInstance Win32_Processor |
  Select-Object Name

Get-CimInstance Win32_ComputerSystem |
  Select-Object TotalPhysicalMemory

# NVIDIA GPU（若已安裝 NVIDIA 驅動）
nvidia-smi
```

如果 laptop 沒有 NVIDIA GPU，我建議先以 CPU 跑 nanoGPT；如果有 NVIDIA GPU，則可以直接開始 MiniMind。

## 四、第一個實驗：用 nanoGPT 從零訓練 Transformer

這是我最推薦你先做的實際動手練習。

目標是建立約 1M–2M 參數級的 Tiny GPT，從 Random Initialization 開始 Pretraining，而不是下載 GPT-2 權重。

### Step 1：建立 Python 環境

在 Windows PowerShell 執行：

```
mkdir D:\LLM-Lab
cd D:\LLM-Lab

git clone https://github.com/karpathy/nanoGPT.git
cd nanoGPT

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install torch numpy transformers datasets tiktoken tqdm
```

這裡先以 CPU 版本作為通用起點；有 NVIDIA GPU 時，應依 [PyTorch 官方安裝頁](https://pytorch.org/get-started/locally/) 選擇對應 CUDA 版本，並驗證：

```
import torchprint(torch.__version__)print(torch.cuda.is_available())
```

### Step 2：產生 Pretraining Dataset

執行：

```
python data/shakespeare_char/prepare.py
```

這份小型 Shakespeare 資料會進行字元級 Tokenization，產生：

```
nanoGPT/
├── data/
│   └── shakespeare_char/
│       ├── input.txt
│       ├── train.bin
│       ├── val.bin
│       ├── meta.pkl
│       └── prepare.py
├── model.py
├── train.py
└── sample.py
```

各檔案的用途：

|檔案|作用|
|---|---|
|`input.txt`|原始文字|
|`prepare.py`|Tokenization 與 Train/Val 切分|
|`train.bin`|訓練 Token IDs|
|`val.bin`|驗證 Token IDs|
|`meta.pkl`|字元 Vocabulary 與映射資訊|
|`model.py`|Transformer 架構|
|`train.py`|Optimizer、Forward、Backward、Training Loop|
|`sample.py`|Autoregressive Inference|

這是實際存在於 nanoGPT 的入門資料處理路徑。

![](https://www.google.com/s2/favicons?domain=https://github.com&sz=32)

GitHub

### Step 3：開始 Pretraining

下面是我建議的 CPU 小模型設定：

```
python train.py config/train_shakespeare_char.py `
  --device=cpu `
  --compile=False `
  --dtype=float32 `
  --init_from=scratch `
  --n_layer=4 `
  --n_head=4 `
  --n_embd=64 `
  --block_size=64 `
  --batch_size=8 `
  --max_iters=500 `
  --lr_decay_iters=500 `
  --eval_interval=100 `
  --eval_iters=10 `
  --always_save_checkpoint=True
```

這個設定的重點是縮小：

- Transformer Layers：4
    
- Attention Heads：4
    
- Embedding Dimension：64
    
- Context Window：64 字元
    
- Training Iterations：500
    

它不是一個有實用聊天能力的模型，但非常適合觀察 Pretraining 的基本機制。

### Step 4：觀察訓練過程

在每次 Iteration，主要發生：

\[ X\rightarrow Embedding\rightarrow Transformer\rightarrow Logits \]

再計算：

\[ L=-\frac{1}{N}\sum_{i=1}^{N}\log P(x_{i+1}\mid x_{\le i}) \]

接著執行：

```
logits, loss = model(X, Y)loss.backward()optimizer.step()optimizer.zero_grad()
```

這段是概念化的 PyTorch 訓練循環；實際 nanoGPT 還有 Gradient Accumulation、Learning Rate Scheduling、Evaluation、Checkpoint Saving 等細節。

請特別觀察：

1. Training Loss 是否下降。
    
2. Validation Loss 是否下降。
    
3. Learning Rate 如何變化。
    
4. 每次訓練 Iteration 的耗時。
    
5. 是否發生 Overfitting。
    

### Step 5：執行 Inference

完成訓練後：

```
python sample.py `
  --out_dir=out-shakespeare-char `
  --device=cpu `
  --start="ROMEO:"
```

你可能得到一些模仿 Shakespeare 對話格式的文字，雖然語意未必通順。

這時你已經完成了一個真正的 LLM 原型：從零建立權重、準備 Dataset、Pretraining、儲存 Checkpoint、重新載入並產生文字。

### Step 6：認識 Model Checkpoint

Checkpoint 一般儲存在：

```
out-shakespeare-char/
└── ckpt.pt
```

可以用以下程式觀察模型參數（只載入可信來源的 checkpoint）：

```
import torchckpt = torch.load(    "out-shakespeare-char/ckpt.pt",    map_location="cpu",    weights_only=False,)print(ckpt.keys())print(ckpt["model_args"])print(ckpt.get("iter_num"))print(ckpt.get("best_val_loss"))
```

由於這個舊版 nanoGPT 使用了包含 Python 物件的 checkpoint 格式，上例設定 `weights_only=False`；不要用它讀取不可信的 `.pt` 檔案。

在這個階段，建議你再做三次實驗：改變 Transformer Layers（2、4、6）、改變 Context Length（32、64、128），以及比較不同 Learning Rate 對 Loss 的影響。

## 五、第二個實驗：MiniMind 完整 Pretraining → SFT → DPO

完成 nanoGPT 後，接下來就不只是模仿文章，而是要讓模型學習接受 Instructions，並嘗試偏好對齊。

MiniMind 是比較符合這個目標的選擇。

### Step 1：下載 MiniMind

```
cd D:\LLM-Lab

git clone https://github.com/jingyaogong/minimind.git

cd minimind

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

MiniMind 2026 年已經有新版模型與新的 Dataset 格式，因此應遵照當前版本 README 對應的訓練程式，不應混用舊版 MiniMind2 的檔案配置與新版 MiniMind-3 模型。

![](https://www.google.com/s2/favicons?domain=https://github.com&sz=32)

GitHub

### Step 2：準備三種不同的 Dataset

這是理解完整 pipeline 最重要的地方。

A. Pretraining Dataset

學習目標：預測下一個 Token。

```
{"text":"The Earth revolves around the Sun."}
{"text":"A neural network consists of layers."}
{"text":"Machine learning uses training data."}
```

模型學的是文字分布與語言規律，而不是明確的問答規則。

B. SFT Dataset

學習目標：根據 User Instruction 產生 Assistant Response。

```
{
  "messages": [
    {"role":"user","content":"What is a GPU?"},
    {"role":"assistant","content":"A GPU is a processor designed for parallel computation."}
  ]
}
```

這是通用對話資料結構的示例。實際進入 Training Script 前，需要轉成該模型 Chat Template 接受的格式。

C. DPO Preference Dataset

學習目標：讓模型提高 Preferred Response 的相對機率。

```
{
  "prompt":"Explain overfitting.",
  "chosen":"Overfitting occurs when a model learns training-specific patterns and generalizes poorly.",
  "rejected":"Overfitting means training is always successful."
}
```

這三個欄位是典型 DPO 格式。實際使用前，應配合 Tokenizer 與 Chat Template 做前處理。

### Step 3：執行各階段訓練

MiniMind 的實際流程可以整理成：

```
Random Initialization
       |
       v
Pretraining
       |
       +--> pretrained_model/
       |
       v
Supervised Fine-tuning
       |
       +--> sft_model/
       |
       v
Direct Preference Optimization
       |
       +--> dpo_model/
       |
       v
Inference / Evaluation
```

另外可以由 Pretrained 或 SFT Model 分支進行：

```
Pretrained / SFT Model
       |
       v
LoRA Training
       |
       +--> lora_adapter/
       |
       v
Load Base + Adapter
       |
       v
Inference
```

這裡我建議你先以各階段 100–500 Steps 進行 Pipeline Smoke Test，確認完整執行、儲存、載入與推理，再決定是否增加訓練步數。

100–500 Steps 不代表模型會訓練成功，而是確認程式、資料、GPU 與 Checkpoint 的完整鏈路能運作。

## 六、我建議你建立的 LLM Laboratory 資料夾

實際學習時，我建議把程式、Dataset、模型權重、訓練紀錄分開保存，而不是全部混在 Repository 裡。

以下是你可以自行建立的實驗室結構，不是 MiniMind 原始 Repo 的預設目錄。

D:\LLM-Lab

repos/

nanoGPT、MiniMind、Smol Course 原始碼

datasets/raw/

原始文字資料

datasets/processed/

清理、切分及 Tokenization 後的資料

datasets/sft/

Instruction / Response 資料

datasets/preference/

DPO Chosen / Rejected 資料

tokenizer/

Tokenizer、Vocabulary、Chat Template

configs/

Model、Training、Optimizer、Scheduler 設定

checkpoints/pretrain/

Base Model 權重

checkpoints/sft/

SFT Model 權重

checkpoints/dpo/

DPO Model 權重

adapters/lora/

LoRA Adapter 權重

evaluations/

不同模型的測試結果

logs/

Loss、Learning Rate、Memory、Throughput

inference/

Inference Scripts、API、生成結果

### 各種訓練應保留哪些檔案？

|階段|主要輸入|主要輸出|
|---|---|---|
|Tokenizer|Raw text|`tokenizer.json`、Vocabulary|
|Pretraining|Token IDs、Model Config|Base Model Weights、Optimizer State|
|SFT|Base Model、Instruction Dataset|SFT Checkpoint|
|LoRA|Base Model、Domain Dataset|Adapter Weights、Adapter Config|
|DPO|SFT Model、Preference Dataset、Reference Policy|Aligned Model|
|Evaluation|Model、Test Dataset|Loss、Accuracy、Preference Metrics|
|Inference|Weights、Tokenizer、Generation Config|Generated Text、Latency Metrics|

還應保存資料版本、Random Seed、Git Commit、Package Versions、Training Config，以及用來恢復訓練的 Scheduler 和 RNG State（適用時）。這些是確保實驗可重現的關鍵。

## 七、建議完成的 8 個實驗

LLM Training Lab 進度

0 / 8

Experiment 1 — 從 Random Weights 建立 Tiny GPT

nanoGPT · 理解 Transformer、Embedding、Attention、Next-token Prediction

Experiment 2 — 研究 Tokenizer

nanoGPT / Hugging Face · 比較 Character、BPE、Vocabulary Size

Experiment 3 — Pretraining 與 Scaling

MiniMind · 比較模型參數量、資料量、Loss、Training Time

Experiment 4 — SFT Instruction Tuning

MiniMind / TRL · 比較 Base Model 與 SFT Model 的問答能力

Experiment 5 — Full Fine-tuning vs LoRA

MiniMind / PEFT · 比較 Trainable Parameters、VRAM、訓練結果

Experiment 6 — DPO Preference Alignment

TRL · 比較 Chosen / Rejected Response 的 Log Probability

Experiment 7 — Inference Optimization

PyTorch / Transformers · 測試 KV Cache、Temperature、Top-p、Tokens/sec

Experiment 8 — Evaluation 與 Deployment

Transformers / FastAPI · 比較 Checkpoints、建立本地 API

## 八、額外建議：用同一組問題比較各階段的 Model

訓練不同階段後，建議使用完全相同的 Held-out Test Set 評估。

例如在開始之前先保留 50–100 道測試問題，這些問題不能進入任何訓練資料。

|測試題目|Pretrained|SFT|DPO|
|---|---|---|---|
|What is machine learning?|記錄輸出|記錄輸出|記錄輸出|
|Explain gradient descent.|記錄輸出|記錄輸出|記錄輸出|
|What is overfitting?|記錄輸出|記錄輸出|記錄輸出|
|Explain a Transformer.|記錄輸出|記錄輸出|記錄輸出|

以上是測試設計，而不是預期結果。對非常小、訓練資料有限的模型，SFT 或 DPO 後未必會產生合理回答。

你應該比較的指標包括：

- Pretraining：Train Loss、Validation Loss、Perplexity。
    
- SFT：Held-out Answer Quality、Instruction Following。
    
- DPO：Preference Accuracy、Chosen vs Rejected Log-probability Margin。
    
- Inference：Time to First Token、Tokens/sec、Peak Memory。
    
- Training Efficiency：Steps/sec、Trainable Parameters、Peak VRAM。
    

另外，DPO 不是 Reinforcement Learning 的完整替代練習。如果未來想理解 RLHF、PPO、GRPO，應再增加 Reward Model、Rollout、Policy Optimization 等實驗。

## 九、最後的技術路線建議

第一階段：nanoGPT（CPU 也能開始）

先從 1M–2M 參數左右的小模型開始，真正理解 Model Architecture、Forward/Backward、Optimizer、Checkpoint、Inference。

第二階段：MiniMind（完整 Training Pipeline）

使用 26M–64M 模型，完成 Pretraining、SFT、LoRA、DPO，建立完整的 Dataset 與 Model Versioning。

第三階段：Hugging Face（較接近 Production）

使用 Transformers、Datasets、PEFT、TRL、Accelerate，學習模型部署、量化與訓練評估。從 SmolLM2-135M 等小模型開始。

這三個階段會讓你從「理解模型如何訓練」逐步進入「能設計、實驗和評估完整 LLM Pipeline」的程度。

### 我還需要確認你的 Laptop 規格

這會直接決定適合 2M、26M、64M 還是 135M 的模型，以及能不能在本機完成 SFT、DPO 與 LoRA。

CPU 型號

RAM

8 GB

16 GB

32 GB

64 GB+

GPU 型號或顯示記憶體

依硬體規格設計完整實作教學

最重要的是：我建議你最後不只得到一個可以聊天的模型，而是得到一套可以重複實驗的 LLM Training Laboratory。 每次修改 Dataset、Model Architecture、Loss Function、Learning Rate、SFT/DPO 策略，都能重新訓練、比較結果，這才是對學習 LLM Engineering 最有價值的部分。



#### 解釋這個MiniLLM repo的檔案結構
```
請詳細解釋這個repo的檔案結構每個檔案的作用是甚麼, 以及詳細解釋完整的 LLM Lifecycle的的每一步驟是甚麼以及跟那些檔案有關
```
1. **實驗程式**：定義資料處理、模型初始化、訓練、評估與推理怎麼執行。
2. **實驗設定與原始資料**：決定模型大小、訓練步數，以及模型要學習什麼。
3. **實驗產物**：保存 tokenizer、權重、checkpoint、metrics 和各階段的比較結果。

其中，**MiniMind 提供主模型架構；我們的程式負責預訓練迴圈；TRL 負責 SFT、DPO；PEFT 負責 LoRA；nanoGPT 是獨立的預訓練對照模型。**

以下依照目前磁碟上的實際程式說明。我們建立的程式檔會逐一解釋；第三方套件、上游圖片與快取則按用途說明。

---

先看整個目錄的分工：

```
C:\LLM\MiniLLM\
│
├─ README.md                    操作手冊
├─ requirements.txt             主要 Python 套件版本
├─ requirements.lock.txt        完整套件版本快照
├─ upstream-lock.json           MiniMind / nanoGPT 的固定版本
├─ .gitignore                   不納入 Git 的檔案規則
│
├─ configs\                     實驗設定
│  ├─ smoke.json
│  ├─ laptop.json
│  └─ laptop-check.json
│
├─ minillm\                     實驗室的主要 Python 程式
│  ├─ __init__.py
│  ├─ __main__.py
│  ├─ common.py
│  ├─ data.py
│  ├─ modeling.py
│  ├─ pretrain.py
│  ├─ posttrain.py
│  ├─ evaluate.py
│  ├─ inference.py
│  └─ api.py
│
├─ scripts\                     PowerShell 入口與驗證工具
│  ├─ setup.ps1
│  ├─ lab.ps1
│  ├─ smoke.ps1
│  ├─ monitor.ps1
│  ├─ verify_services.py
│  └─ report_validation.py
│
├─ data\
│  └─ demo\                     原始教學資料
│
├─ artifacts\                   每次實驗的獨立產物
│  ├─ smoke\
│  └─ laptop-check\
│
├─ reports\                     環境、測試與服務驗證結果
├─ tests\                       行為正確性測試
├─ vendor\                      MiniMind / nanoGPT 原始碼
├─ .venv\                       本專案的 Python 執行環境
└─ .cache\                      套件、資料與測試快取
```

**最外層的五個檔案，負責操作與版本重現。**

|檔案|作用|
|---|---|
|[README.md](C:/LLM/MiniLLM/README.md)|繁中操作手冊，包含環境安裝、資料格式、各階段命令、續訓、推理與 API 操作。|
|[requirements.txt](C:/LLM/MiniLLM/requirements.txt)|固定主要依賴，例如 Transformers、TRL、PEFT、Datasets、TensorBoard、FastAPI。|
|[requirements.lock.txt](C:/LLM/MiniLLM/requirements.lock.txt)|記錄實際安裝後的完整套件版本，包含間接依賴，讓環境更容易重建。|
|[upstream-lock.json](C:/LLM/MiniLLM/upstream-lock.json)|固定 MiniMind、nanoGPT 的 GitHub URL、commit 和授權資訊，避免上游更新後行為改變。|
|[.gitignore](C:/LLM/MiniLLM/.gitignore)|排除虛擬環境、快取、模型產物、大型原始資料和 vendor 等檔案。這些檔案仍保存在磁碟，只是不預設交給 Git 管理。|

需要分清楚：**套件版本、上游程式版本、資料版本、模型設定，都是重現實驗的一部分。** 只保存模型權重，無法完整回答「這個模型是怎麼訓練出來的」。

---

**`minillm` 是整套 laboratory 的核心。**

|檔案|作用與主要內容|
|---|---|
|[__init__.py](C:/LLM/MiniLLM/minillm/__init__.py)|將目錄標示為 Python package，目前主要放套件說明。|
|[__main__.py](C:/LLM/MiniLLM/minillm/__main__.py)|命令列總入口。解析 `prepare`、`init`、`pretrain`、`sft`、`dpo`、`lora`、`infer` 等命令，再呼叫對應模組。|
|[common.py](C:/LLM/MiniLLM/minillm/common.py)|共用工具：JSON/JSONL 讀寫、SHA256、隨機種子、CPU/CUDA 選擇、精度選擇、實驗路徑、版本紀錄和硬體檢測。|
|[data.py](C:/LLM/MiniLLM/minillm/data.py)|原始資料讀取、格式統一、清理、去重、切分、教學資料生成、BPE tokenizer 訓練、token binary 建立，以及 SFT prompt/completion 格式化。|
|[modeling.py](C:/LLM/MiniLLM/minillm/modeling.py)|載入 MiniMind 模型類別，提供 `LabMiniMind` 相容層、隨機初始化、權重統計，以及完整模型／LoRA adapter 載入。|
|[pretrain.py](C:/LLM/MiniLLM/minillm/pretrain.py)|原生 PyTorch 預訓練：抽取 token windows、forward、loss、backward、AdamW 更新、梯度裁切、validation、checkpoint 和續訓。也負責 nanoGPT 對照訓練。|
|[posttrain.py](C:/LLM/MiniLLM/minillm/posttrain.py)|使用 TRL 執行 SFT、DPO，使用 PEFT 執行 LoRA。處理 completion masking、batch padding、累積梯度 loss、reference model、訓練紀錄與續訓身分檢查。|
|[evaluate.py](C:/LLM/MiniLLM/minillm/evaluate.py)|在相同評估資料上比較各階段模型：raw loss、perplexity、SFT/domain completion loss、偏好配對、固定 prompt 生成和 cache benchmark。|
|[inference.py](C:/LLM/MiniLLM/minillm/inference.py)|`Engine` 推理介面：載入模型與 tokenizer、套用 chat template、處理 sampling 設定、產生文字、計時，以及 cached/uncached 比較。|
|[api.py](C:/LLM/MiniLLM/minillm/api.py)|將推理引擎包成 FastAPI，提供健康檢查、文字生成與簡單服務 metrics。|

這些模組的呼叫關係是：

````
```mermaid
flowchart TD
    A["PowerShell 命令"] --> B["命令列總入口"]
    B --> C["資料處理與 tokenizer"]
    B --> D["隨機初始化與模型載入"]
    B --> E["PyTorch 預訓練"]
    B --> F["TRL / PEFT 後訓練"]
    B --> G["評估與模型比較"]
    B --> H["推理引擎"]
    B --> I["本機 API"]

    E --> D
    F --> D
    F --> C
    G --> H
    H --> D
    I --> H
```
````

幾個特別值得閱讀的函式：

|函式／類別|你可以學到什麼|
|---|---|
|`clean()`|原始文字如何正規化。|
|`split_rows()`|為何相同文件或 prompt 必須留在同一個 partition。|
|`prepare()`|資料到 tokenizer、token stream 的完整轉換。|
|`fresh_model()`|模型如何依架構設定和實際 vocabulary 建立。|
|`LabMiniMind._init_weights()`|預訓練前的權重如何初始化。|
|`pretrain.train()`|一個真正的訓練 step 做了哪些事。|
|`CompletionCollator`|不同長度資料如何 padding，以及如何遮罩 prompt。|
|`completion_ce()`|如何只對回答計算 loss，並正確處理 gradient accumulation。|
|`posttrain.train()`|SFT、DPO、LoRA 的分支差異。|
|`Engine.generate()`|推理如何從 prompt 走到生成 token。|

---

**`scripts` 讓你不用每次手動處理 Python 路徑和啟動細節。**

|檔案|作用|
|---|---|
|[setup.ps1](C:/LLM/MiniLLM/scripts/setup.ps1)|檢查 Python 3.12、下載固定上游版本、建立 `.venv`、安裝 PyTorch 與依賴，最後執行環境檢查。|
|[lab.ps1](C:/LLM/MiniLLM/scripts/lab.ps1)|日常操作入口。切到專案根目錄，使用專案的 Python 執行 `python -m minillm`，並傳遞你的命令參數。|
|[smoke.ps1](C:/LLM/MiniLLM/scripts/smoke.ps1)|一次執行微型完整流程，再跑 nanoGPT，最後更新比較報告。|
|[monitor.ps1](C:/LLM/MiniLLM/scripts/monitor.ps1)|啟動 TensorBoard，讀取指定 run 的訓練曲線。|
|[verify_services.py](C:/LLM/MiniLLM/scripts/verify_services.py)|暫時啟動 API 和 TensorBoard，透過真正 HTTP 驗證，結束後關閉自己啟動的服務。|
|[report_validation.py](C:/LLM/MiniLLM/scripts/report_validation.py)|從已存在的實測產物整理驗證報告；它不會重新訓練模型。|

例如：

```
cd C:\LLM\MiniLLM
.\scripts\lab.ps1 pretrain --run my-lab
```

實際執行路徑是：

```
lab.ps1
  → 專案 .venv 的 Python
  → python -m minillm
  → __main__.py 解析 pretrain
  → pretrain.py 的 train()
```

`all` 命令包含 MiniMind 主流程和 LoRA 分支；**nanoGPT 是由 `smoke.ps1` 額外呼叫的**。`merge`、API 和 TensorBoard 也需要另外啟動。

---

**`configs` 決定一次實驗的規模。**

|檔案|設定與用途|
|---|---|
|[smoke.json](C:/LLM/MiniLLM/configs/smoke.json)|hidden 128、2 layers，執行 40 steps 預訓練、20 SFT、10 DPO、15 LoRA，快速驗證機制。|
|[laptop.json](C:/LLM/MiniLLM/configs/laptop.json)|hidden 512、8 layers，設定較長訓練：2000／500／200／300 steps。|
|[laptop-check.json](C:/LLM/MiniLLM/configs/laptop-check.json)|保留 laptop 架構與 batch 設定，但每階段只跑 2 steps，用於硬體適配檢查。|

各設定欄位的意義如下：

|欄位|意義|
|---|---|
|`seed`|隨機種子，影響資料切分、初始化、抽樣等。|
|`device`|`auto` 自動使用 CUDA，沒有 CUDA 時使用 CPU。|
|`vocab_size`|BPE 訓練的 vocabulary 目標上限。|
|`hidden_size`|每個 token 的內部向量維度。|
|`num_hidden_layers`|Transformer block 數量。|
|`num_attention_heads`|Query attention heads 數量。|
|`num_key_value_heads`|Key/Value heads 數量；少於 Query heads 時使用 GQA。|
|`intermediate_size`|Feed-forward network 的中間維度。|
|`block_size`|預訓練 prediction window，以及 SFT/DPO 配對長度限制。|
|`max_position_embeddings`|模型設定的最大位置容量。|
|`batch_size`|一次 microbatch 的資料數量。|
|`gradient_accumulation_steps`|累積幾個 microbatches 才更新一次權重。|
|`*_steps`|各階段的 optimizer update 次數。|
|`*_lr`|各階段 learning rate。|
|`warmup_ratio`|前面多少比例的 steps 用來逐漸提高 learning rate。|
|`weight_decay`|AdamW 的權重衰減強度。|
|`max_grad_norm`|梯度裁切門檻。|
|`save_steps` / `eval_steps`|保存與評估的間隔。|
|`lora_r` / `lora_alpha`|LoRA 的低秩維度與更新縮放設定。|
|`dpo_beta`|DPO loss 中，policy/reference 相對 log-probability 的尺度參數。|

**訓練長度與推理容量要分開看。** 例如 smoke 的 `block_size=128`，模型的位置容量是 512。推理可以接受容量內的序列，但短序列訓練不代表模型已經學會良好處理更長上下文。

另外，設定 `vocab_size=1024` 不表示最後一定有 1024 個 tokens。目前教學語料的實際 vocabulary 是 **835**；模型 embedding 依照這個實際數量建立。

---

**原始資料與處理後資料放在不同位置。**

目前原始教學資料位於 [data/demo](C:/LLM/MiniLLM/data/demo)：

|檔案|內容與用途|
|---|---|
|[pretrain.jsonl](C:/LLM/MiniLLM/data/demo/pretrain.jsonl)|一般文字文件，用來學習 next-token prediction。|
|[sft.jsonl](C:/LLM/MiniLLM/data/demo/sft.jsonl)|指令與正確回答，用來做 instruction tuning。|
|[dpo.jsonl](C:/LLM/MiniLLM/data/demo/dpo.jsonl)|同一 prompt 的 chosen/rejected 回答，用來學習偏好。|
|[domain.jsonl](C:/LLM/MiniLLM/data/demo/domain.jsonl)|LLM laboratory 領域問答，用於 LoRA domain SFT。|
|[source.json](C:/LLM/MiniLLM/data/demo/source.json)|資料來源、授權與教學用途說明。|

這些資料是本地程式生成的，目的在於驗證流程。預訓練資料還故意加入空白、控制字元和重複文件，以驗證清理功能。

你自己的語料可以另放在 `C:\LLM\MiniLLM\data\raw\`，再透過 `prepare` 指定來源。

---

**`artifacts` 是一次實驗的完整紀錄。**

`--run smoke` 對應 [artifacts/smoke](C:/LLM/MiniLLM/artifacts/smoke)，`--run laptop-check` 對應另一個獨立目錄。

```
C:\LLM\MiniLLM\artifacts\smoke\
│
├─ config.json
├─ data\
│  ├─ manifest.json
│  ├─ pretrain\
│  ├─ sft\
│  ├─ dpo\
│  └─ domain\
│
├─ tokenizer\
├─ initialized\
├─ base\
├─ sft\
├─ dpo\
├─ lora\
├─ merged\
├─ nanogpt\
├─ tensorboard\
├─ evaluation\
└─ api_requests.jsonl
```

各目錄代表模型生命週期的一個狀態：

|目錄|保存什麼|
|---|---|
|`initialized`|尚未訓練的隨機初始化模型。|
|`base`|完成預訓練的完整模型。|
|`sft`|完成指令微調的完整模型。|
|`dpo`|完成偏好對齊的完整模型。|
|`lora`|LoRA adapter，使用時需要它的 parent model。|
|`merged`|將 LoRA 更新合併後的完整模型。|
|`nanogpt`|獨立 GPT 架構的初始化、訓練結果和對照 metrics。|
|`tensorboard`|訓練曲線 event files。|
|`evaluation`|各階段評估、生成樣本與共同報告。|

產物裡有些檔名會反覆出現，因為每個階段都需要自己的獨立版本：

|檔案類型|作用|
|---|---|
|run 根目錄的 `config.json`|整次實驗的設定快照。|
|模型目錄的 `config.json`|模型架構設定，供載入權重時重建網路。|
|`model.safetensors`|完整模型權重。|
|`adapter_model.safetensors`|LoRA adapter 權重，不包含完整 parent model。|
|`adapter_config.json`|LoRA rank、alpha、target modules 等設定。|
|`tokenizer.json`|Vocabulary、BPE merges、編碼與解碼規則。|
|`tokenizer_config.json`|Hugging Face tokenizer 行為設定。|
|`special_tokens_map.json`|PAD、BOS、EOS 等特殊 token 的對應。|
|`chat_template.jinja`|user／assistant 訊息如何轉成文字序列。|
|`generation_config.json`|模型附帶的生成設定；本實驗室推理時還會明確傳入 CLI 設定。|
|`weight_stats.json`|每個參數 tensor 的名稱、形狀、數量、mean/std 與是否可訓練。|
|`lineage.json`|模型的 parent、階段、套件版本、上游 commit，以及後訓練的資料／parent 身分紀錄。|
|`metrics.jsonl`|隨訓練進行追加的逐步紀錄。|
|`initial_eval.json`|預訓練開始時的 validation 結果。|
|`final_metrics.json`|該階段完成後的 metrics 摘要。|
|`trainer_state.json`|HF Trainer 的 global step、log history 等狀態。|
|`training_args.bin`|HF Trainer 的訓練參數序列化檔。|
|模型產物中的 `README.md`|Trainer／PEFT 產生的模型卡或 adapter 說明，不是專案根目錄的操作手冊。|

處理後資料也有明確用途：

|檔案|作用|
|---|---|
|各資料類型的 `train.jsonl`|已清理、格式統一的訓練資料。|
|各資料類型的 `validation.jsonl`|保留用於評估的資料。|
|預訓練的 `train.bin` / `validation.bin`|`int32` token IDs，可用 memory map 讀取。|
|[manifest.json](C:/LLM/MiniLLM/artifacts/smoke/data/manifest.json)|資料來源 SHA256、清理計數、切分方式、token 數與 tokenizer 身分。|

**模型權重與訓練 checkpoint 的差別尤其重要。**

模型權重保存「模型現在知道什麼」；可續訓 checkpoint 還要保存「optimizer 現在走到哪裡」。

原生預訓練的 checkpoint，例如 [base/checkpoint-10](C:/LLM/MiniLLM/artifacts/smoke/base/checkpoint-10)，包含：

- 完整模型權重與 tokenizer。
- `training_state.pt`：optimizer、scheduler、step、CPU/CUDA RNG、配置與資料 manifest 身分。
- `metrics.json`：checkpoint 對應的評估紀錄。

TRL 的 checkpoint，例如 [sft/checkpoint-10](C:/LLM/MiniLLM/artifacts/smoke/sft/checkpoint-10)，使用分開的檔案：

|檔案|作用|
|---|---|
|`optimizer.pt`|AdamW 的動量與平方梯度等狀態。|
|`scheduler.pt`|Learning-rate scheduler 狀態。|
|`rng_state.pth`|隨機數狀態。|
|`trainer_state.json`|訓練步數與歷史紀錄。|
|`training_args.bin`|訓練參數。|
|模型或 adapter 權重|checkpoint 當下的可學習參數。|

**只載入權重再建立新 optimizer，屬於重新開始一段訓練；不等於精確續訓。**

nanoGPT 另使用 `initialized.pt`、`final.pt`、`model_config.json` 和 `generation.json`，因為它是另一套原生 PyTorch 模型。

---

**`reports` 與 `tests` 用來驗證實驗室本身。**

|檔案|作用|
|---|---|
|[tests/test_lab.py](C:/LLM/MiniLLM/tests/test_lab.py)|測試資料切分、tokenizer、masking、causal attention、KV cache、保存重載、LoRA 凍結、續訓一致性與 API。|
|[environment.json](C:/LLM/MiniLLM/reports/environment.json)|實際 GPU、RAM、CUDA 與套件版本。|
|[tests.txt](C:/LLM/MiniLLM/reports/tests.txt)|unittest 執行輸出。|
|[services.json](C:/LLM/MiniLLM/reports/services.json)|API HTTP 回應、服務 metrics 和 TensorBoard scalar tags 的驗證結果。|
|[api-server.log](C:/LLM/MiniLLM/reports/api-server.log)|API 驗證時的服務日誌。|
|[tensorboard-server.log](C:/LLM/MiniLLM/reports/tensorboard-server.log)|TensorBoard 驗證時的服務日誌。|
|[validation.md](C:/LLM/MiniLLM/reports/validation.md)|人類可閱讀的實测摘要。|
|[validation_summary.json](C:/LLM/MiniLLM/reports/validation_summary.json)|程式可讀取的驗證摘要。|

這裡要區分：

- **程式正確性測試**：確認梯度、mask、cache、續訓等機制正確。
- **模型能力評估**：確認模型回答是否有用、正確、符合需求。

目前 11 項測試通過，證明實作機制可運作；模型能力還需要更完整的資料與評估。

---

**`vendor` 保存上游原始碼，主流程只使用其中一部分。**

MiniMind 最重要的是 [model_minimind.py](C:/LLM/MiniLLM/vendor/minimind/model/model_minimind.py)，它定義實際的 Transformer：

|類別／函式|作用|
|---|---|
|`MiniMindConfig`|模型架構設定。|
|`RMSNorm`|對 hidden states 做 RMS normalization。|
|RoPE 相關函式|將位置資訊加入 Query/Key。|
|`Attention`|Q/K/V projections、causal attention、GQA 和 KV cache。|
|`FeedForward`|使用 gated activation 的前饋網路。|
|`MiniMindBlock`|Attention、normalization、MLP 和 residual connections。|
|`MiniMindModel`|Embedding、多個 blocks、最後 normalization。|
|`MiniMindForCausalLM`|加入輸出 head、next-token loss 和生成迴圈。|
|`MOEFeedForward`|上游的 MoE 實作；目前本實驗室設定沒有啟用。|

其他上游檔案的用途：

|上游檔案／群組|用途與目前關係|
|---|---|
|[model_lora.py](C:/LLM/MiniLLM/vendor/minimind/model/model_lora.py)|MiniMind 自己的 LoRA 實作；目前主流程使用 PEFT。|
|[lm_dataset.py](C:/LLM/MiniLLM/vendor/minimind/dataset/lm_dataset.py)|上游各種訓練資料類別；目前主流程使用我們的資料處理模組。|
|[train_tokenizer.py](C:/LLM/MiniLLM/vendor/minimind/trainer/train_tokenizer.py)|上游 tokenizer 訓練。|
|[train_pretrain.py](C:/LLM/MiniLLM/vendor/minimind/trainer/train_pretrain.py)|上游原生預訓練。|
|[train_full_sft.py](C:/LLM/MiniLLM/vendor/minimind/trainer/train_full_sft.py)|上游全參數 SFT。|
|[train_dpo.py](C:/LLM/MiniLLM/vendor/minimind/trainer/train_dpo.py)|上游原生 DPO。|
|[train_lora.py](C:/LLM/MiniLLM/vendor/minimind/trainer/train_lora.py)|上游原生 LoRA 訓練。|
|[trainer_utils.py](C:/LLM/MiniLLM/vendor/minimind/trainer/trainer_utils.py)|上游的 LR、logging、checkpoint、distributed training 等工具。|
|[train_distillation.py](C:/LLM/MiniLLM/vendor/minimind/trainer/train_distillation.py)|Teacher/student 蒸餾訓練。|
|[train_ppo.py](C:/LLM/MiniLLM/vendor/minimind/trainer/train_ppo.py)、[train_grpo.py](C:/LLM/MiniLLM/vendor/minimind/trainer/train_grpo.py)|上游更進階的 reinforcement learning 訓練。|
|[train_agent.py](C:/LLM/MiniLLM/vendor/minimind/trainer/train_agent.py)、[rollout_engine.py](C:/LLM/MiniLLM/vendor/minimind/trainer/rollout_engine.py)|Agent/tool-use 訓練與 rollout 支援。|
|[eval_llm.py](C:/LLM/MiniLLM/vendor/minimind/eval_llm.py)|上游模型測試入口。|
|[convert_model.py](C:/LLM/MiniLLM/vendor/minimind/scripts/convert_model.py)|上游模型格式轉換工具。|
|[serve_openai_api.py](C:/LLM/MiniLLM/vendor/minimind/scripts/serve_openai_api.py)、[chat_api.py](C:/LLM/MiniLLM/vendor/minimind/scripts/chat_api.py)、[web_demo.py](C:/LLM/MiniLLM/vendor/minimind/scripts/web_demo.py)|上游服務、API 客戶端和網頁展示工具。|
|[eval_toolcall.py](C:/LLM/MiniLLM/vendor/minimind/scripts/eval_toolcall.py)|上游 tool calling 評估。|

上游附帶的 tokenizer 檔也保留下來，但**目前實驗使用的是自己重新訓練的 tokenizer**。

nanoGPT 的主要檔案如下：

|檔案／群組|作用|
|---|---|
|[model.py](C:/LLM/MiniLLM/vendor/nanogpt/model.py)|GPT 模型定義；我們的對照訓練直接使用它。|
|[train.py](C:/LLM/MiniLLM/vendor/nanogpt/train.py)|上游原始訓練入口；我們目前使用共用預訓練迴圈。|
|[sample.py](C:/LLM/MiniLLM/vendor/nanogpt/sample.py)|上游文字生成入口。|
|[bench.py](C:/LLM/MiniLLM/vendor/nanogpt/bench.py)|訓練效能 benchmark。|
|[configurator.py](C:/LLM/MiniLLM/vendor/nanogpt/configurator.py)|上游的設定檔／命令列覆寫機制。|
|[config](C:/LLM/MiniLLM/vendor/nanogpt/config)|Shakespeare、GPT-2 訓練、微調和評估範例設定。|
|[data](C:/LLM/MiniLLM/vendor/nanogpt/data)|Shakespeare/OpenWebText 的原始資料準備範例。|
|[scaling_laws.ipynb](C:/LLM/MiniLLM/vendor/nanogpt/scaling_laws.ipynb)、[transformer_sizing.ipynb](C:/LLM/MiniLLM/vendor/nanogpt/transformer_sizing.ipynb)|模型大小、計算需求與 scaling 的探索 notebook。|

兩個上游的 README、LICENSE、圖片和 assets 是說明、授權與展示資料；它們不參與目前的訓練計算。

TRL、PEFT 等已安裝套件在 [`.venv`](C:/LLM/MiniLLM/.venv) 裡，不在 vendor。`__pycache__` 是 Python 自動產生的編譯快取，通常不需要閱讀或修改。

---

接著把這些檔案串成完整 LLM lifecycle：

````
```mermaid
flowchart TD
    A["Raw Text"] --> B["清理、去重、切分"]
    B --> C["訓練 Tokenizer"]
    C --> D["Token IDs"]
    C --> E["建立 Transformer 與隨機權重"]
    D --> F["Pretraining"]
    E --> F
    F --> G["Base Model"]
    H["Instruction / Response"] --> I["SFT"]
    G --> I
    I --> J["SFT Model"]
    K["Prompt / Chosen / Rejected"] --> L["DPO"]
    J --> L
    L --> M["Aligned Model"]

    G --> N["LoRA Domain SFT"]
    J --> N
    O["Domain Dataset"] --> N
    N --> P["Adapter"]
    P --> Q["Parent + Adapter 或 Merge"]

    G --> R["共同評估"]
    J --> R
    M --> R
    Q --> R
    R --> S["Inference"]
    S --> T["本機 API 與 Monitoring"]
```
````

**第一步：定義實驗與確認硬體。**

在訓練前，你先決定：

- 模型架構多大。
- 要使用哪個 tokenizer。
- 每次看多少 tokens。
- 一次更新使用多少資料。
- 訓練幾個 steps。
- 要記錄與比較哪些指標。

本 repo 由設定檔決定這些條件，`doctor` 則執行真實 CUDA 運算並記錄 GPU、RAM 和版本。

```
cd C:\LLM\MiniLLM
.\scripts\lab.ps1 doctor
```

相關程式是共用工具模組的 `doctor()`；結果寫到環境報告。

`prepare` 會將選用設定複製到該 run。之後各階段讀取這份快照，而不是一直依賴外部設定檔。

---

**第二步：取得 Raw Dataset。**

不同訓練目標需要不同資料。

預訓練使用一般文字(./data/demo/pretrain.jsonl)：

```
{"text":"Transformer uses attention to connect tokens."}
```

SFT 使用指令與回答(./data/demo/sft.jsonl)：

```
{"instruction":"Explain SFT.","output":"SFT trains a model on instruction-response examples."}
```

DPO 使用偏好配對(./data/demo/dpo.jsonl)：

```
{"prompt":"What is 2 + 3?","chosen":"5","rejected":"9"}
```

LoRA domain SFT 使用特定領域的指令與回答，格式可以和 SFT 相同。

資料模組的 `read_rows()` 支援：

- TXT：空白行分隔文件，供預訓練使用。
- JSONL：每行一個 JSON record。
- Parquet：以 batches 讀取欄位資料。

Raw data 在這個階段仍然是文字，還不是神經網路能直接計算的 tensor。

---

**第三步：Cleaning、Deduplication 和 Train/Validation Split。**

資料模組依序做三件事。

首先是清理：

- Unicode NFKC 正規化。
- 統一換行。
- 移除控制字元。
- 壓縮空白。
- 移除空紀錄。

接著統一資料格式，例如把 `instruction/output` 轉成 `messages`。

最後做精確去重和分組切分。切分概念是：

```
SHA256(seed + normalized document/prompt)
    → 固定分配到 train 或 validation
```

約 10% 的 hash buckets 用於 validation，因此不是精確的 90/10 筆數比例。

同一 prompt 的不同回答必須留在同一側，否則模型可能訓練時已見過相同題目，評估結果就容易被高估。

目前 smoke 的資料結果是：

|類型|Train|Validation|
|---|---|---|
|Pretraining|262 文件|19 文件|
|SFT|102 筆|18 筆|
|DPO|102 配對|18 配對|
|Domain|75 筆|5 筆|

這裡做的是**精確去重與相同 prompt 分組**，尚未消除近似、語意相同或模板相似的資料。目前也只有 train/validation，沒有另外建立獨立 test set。

---

**第四步：訓練 Tokenizer。**

Tokenizer 決定文字如何變成整數序列：

```
文字 → tokens → token IDs
```

Token 不一定是一個完整單字，也不一定是一個中文字；byte-level BPE 可以把文字拆成較小片段，再學習常見片段的合併規則。

本 repo 的 `prepare()`：

1. 建立 byte-level BPE。
2. 使用完整 byte alphabet。
3. 只從 pretraining 的 train partition 學習 vocabulary。
4. 加入 PAD、BOS、EOS 和 chat special tokens。
5. 保存 tokenizer 和 chat template。

**訓練 tokenizer 與訓練 Transformer 是兩件不同的事。** Tokenizer 學習文字切分規則；Transformer 學習 token 序列的條件機率。

所有模型階段使用相同 token IDs。若換 tokenizer，原本 embedding 的每一列就可能對應到不同文字，不能直接沿用原權重。

---

**第五步：Tokenization 與建立訓練輸入。**

預訓練文字經 tokenizer 轉換後，在每份文件末尾加入 EOS，接成 token stream，保存為 binary。

例如概念上：

```
文件 A 的 IDs → EOS → 文件 B 的 IDs → EOS
```

Memory map 讓程式能按位置讀取 token windows，而不必先把整個 binary 檔複製進一般 Python list。

目前 smoke 有：
- 8,388 個訓練 tokens。
- 678 個 validation tokens。

SFT 和 DPO 的格式不同，因此不直接使用這份預訓練 binary；它們在後訓練時，將 prompt 和回答 tokenization。

這個 repo 會對過長的 SFT/DPO 配對報錯，讓你明確決定增加 context 或縮短資料。

---

**第六步：建立 Transformer 並隨機初始化。**

這一步決定網路結構，但模型還沒有透過資料學習。

一個 token 經過模型的主要路徑是：

```
Token ID
  → Embedding vector
  → Transformer blocks
  → Final normalization
  → Output projection
  → Vocabulary logits
```

Transformer block 裡：

- **Attention**：讓目前位置利用前面 tokens 的資訊。
- **Causal mask**：阻止位置看到後面的答案。
- **RoPE**：提供位置資訊。
- **GQA**：讓多個 Query heads 分享較少的 Key/Value heads。
- **Feed-forward network**：對每個位置的表示做非線性轉換。
- **Residual connections**：幫助資訊與梯度通過多層網路。

`fresh_model()` 根據配置及实际 vocabulary 建立模型。

初始化規則在 `LabMiniMind._init_weights()`：

- Linear、Embedding：Normal(0, 0.02)。
- RMSNorm scale：1。
- Linear bias：若存在則設為 0。
- 輸入 embedding 和輸出 head 分享權重。

所以「隨機初始化」不表示每個 tensor 都是隨機數。

```
.\scripts\lab.ps1 init --run my-lab
```

輸出是 `initialized` 模型。你可以查看權重統計，或對它做推理，觀察尚未訓練時的輸出。

---

**第七步：Pretraining from Scratch。**

目標是學習：

> 已知前面的 tokens，下一個 token 應該是什麼？

假設一段 token sequence 是：

```
[t0, t1, t2, t3]
```

訓練目標包含：

```
看到 t0         → 預測 t1
看到 t0,t1      → 預測 t2
看到 t0,t1,t2   → 預測 t3
```

所有位置可以在同一次 forward 裡計算，causal attention 保證每個位置只使用允許的前文。

Loss 是 next-token cross entropy：

$L_{\text{pretrain}} = -\frac{1}{N}\sum_t \log p_\theta(x_{t+1}\mid x_{\le t})$

一次 optimizer step 的工作順序是：

1. 從訓練 token stream 抽取 windows。
2. Forward，取得 logits。
3. 計算正確下一個 token 的 loss。
4. Backward，計算每個可訓練參數的 gradient。
5. 累積 microbatch gradients。
6. 裁切 gradient norm。
7. AdamW 更新權重。
8. 更新 learning-rate scheduler。
9. 依間隔評估與保存 checkpoint。

**Gradient 告訴 optimizer，參數變動會如何影響 loss；optimizer 利用 gradient 與歷史狀態，決定實際更新。**

單 GPU 的有效 batch 數大致是：

$effective batch = \text{batch size} \times \text{gradient accumulation steps}$

Laptop 設定是 \(2\times8=16\) 個 windows／optimizer update。預訓練每個 window 有 256 個 next-token targets，因此每次更新處理 4,096 個 targets。

MiniMind 使用 `L+1` 個輸入 tokens 計算 `L` 個 shifted targets；nanoGPT 則以 `L` 個輸入和右移 targets 計算相同數量的預測目標。

```
.\scripts\lab.ps1 pretrain --run my-lab
```

輸出為 Base Model。此時模型可能學會文字模式和一些知識，但不一定會遵循使用者指令。

---

**第八步：Supervised Fine-Tuning，讓 Base Model 學習回答指令。**

SFT 從 Base weights 接著訓練，架構與 tokenizer 不變。

指令會先套用 chat template：

```
<|im_start|>user
What is 2 + 3?<|im_end|>
<|im_start|>assistant
5<|im_end|>
```

我們希望模型讀懂問題，並學習生成回答。因此：

- Prompt 參與 forward，提供上下文。
- Prompt 不計入直接的 token loss。
- 回答與 EOS 計入 loss。
- Padding 不計入 loss。

`CompletionCollator` 將 prompt 和 padding 的 labels 設為 `-100`。這是 loss 的忽略標記，不是 vocabulary 中的 token ID。

Prompt 被遮罩後仍然能影響模型計算；它只是沒有被當成需要生成的監督目標。

`completion_ce()` 對回答 tokens 的 loss 求和，再按監督 token 數正規化。這能讓不同回答長度下的 gradient accumulation 對應到完整 batch 的計算。

TRL 的 `SFTTrainer` 負責訓練排程、optimizer、evaluation 和 checkpoint；completion-only training 也是 TRL 支援的正式訓練方式。[TRL SFT 文件](https://huggingface.co/docs/trl/v0.24.0/en/sft_trainer)

```
.\scripts\lab.ps1 sft --run my-lab
```

輸出為完整的 Instruction Model，這個分支更新全模型參數。

---

**第九步：DPO，從指令能力進一步學習偏好。**

SFT 示範一個回答；DPO 提供同一題的兩個回答，以及哪一個比較好。

本 repo 預設載入兩份 SFT 模型：

- **Policy model**：接受更新。
- **Reference model**：凍結，作為比較基準。

令 prompt 為 \(x\)，chosen 為 \(y^+\)，rejected 為 \(y^-\)，DPO 比較的是：

$\Delta = \left[ \log \pi_\theta(y^+\mid x) - \log \pi_{\text{ref}}(y^+\mid x) \right] - \left[ \log \pi_\theta(y^-\mid x) - \log \pi_{\text{ref}}(y^-\mid x) \right]$

本 repo 使用 sigmoid loss：

\[ L_{\text{DPO}}=-\log \sigma(\beta\Delta) \]

它促使 policy 相對於 reference，更偏向 chosen 而不是 rejected。

DPO 不要求另外訓練一個獨立 reward model，也不需要像 PPO 那樣在這個訓練流程中執行 rollout。這是 DPO 原始方法的重要特性。[DPO 論文](https://arxiv.org/abs/2305.18290)

```
.\scripts\lab.ps1 dpo --run my-lab
```

輸出為 Aligned Model。要觀察的 metrics 包含 DPO loss、reward margin 和偏好 accuracy。

**DPO loss 和預訓練 cross entropy 是不同目標，數字不能直接互相比大小。** 而目前教學偏好是程式生成的加法配對，還不能代表廣泛的人類偏好。

---

**第十步：LoRA Domain Adaptation。**

LoRA 是「怎麼更新參數」的方法；SFT、DPO 是「用什麼目標訓練」。

本 repo 的 LoRA 分支使用 domain 指令資料，做 SFT：

```
SFT Model + Domain instruction data
    → PEFT LoRA
    → Adapter
```

對原始矩陣 \(W\)，LoRA 使用：

\[ W'=W+\frac{\alpha}{r}BA \]

其中原始 \(W\) 凍結，只有低秩矩陣 \(A\)、\(B\) 被更新。[LoRA 論文](https://arxiv.org/abs/2106.09685)

例如 \(W\) 是 \(512\times512\)：

- 全矩陣有 262,144 個參數。
- 若 \(r=8\)，兩個低秩矩陣合計 \(8\times512+512\times8=8,192\) 個參數。

本 repo 對 attention 和 MLP projections 加入 LoRA，包括 `q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj`。

```
.\scripts\lab.ps1 lora --run my-lab --parent sft
```

也可以選 Base 或 DPO 作 parent。

目前 smoke 的 adapter 有 38,912 個可訓練參數，約占包含 adapter 後模型的 7.21%。低比例可減少 optimizer state，但 forward 和 activation 計算仍然存在。

使用 adapter 推理時，載入程式會確認 parent weights 的 SHA256，避免把 adapter 接到錯誤模型。

若要產生獨立完整模型：

```
.\scripts\lab.ps1 merge --run my-lab
```

合併後權重寫到 `merged`，原始 parent 和 adapter 仍保留。

---

**第十一步：用共同評估資料比較各階段。**

各階段的訓練 loss 不同，因此需要共同評估協議。

評估程式對每個模型使用相同的：

- Raw validation windows。
- Held-out SFT 回答。
- Held-out domain 回答。
- DPO 配對。
- 固定生成 prompts。
- KV cache benchmark。

```
.\scripts\lab.ps1 evaluate --run my-lab
```

主要指標：

|指標|說明|
|---|---|
|Raw validation loss|模型對保留文字的 next-token prediction 表現。|
|Perplexity|在這個評估中是 `exp(loss)`；只有相同 tokenizer、資料與協議下才適合比較。|
|SFT completion loss|對保留指令回答的預測表現。|
|Domain completion loss|對領域回答的預測表現。|
|Chosen win rate|chosen 的回答 log-probability 是否高於 rejected。|
|Preference margin|chosen/rejected 的分數差異。|
|Generation samples|真正看模型生成什麼，補足單一數字看不到的問題。|

目前 [比較報告](C:/LLM/MiniLLM/artifacts/smoke/evaluation/comparison.md) 顯示：

|階段|Raw loss|SFT 回答 loss|Domain 回答 loss|
|---|---|---|---|
|Initialized|6.7693|6.8037|6.7083|
|Base|4.2932|6.4739|6.4639|
|SFT|4.3272|4.8764|6.0428|
|DPO|4.3283|4.9068|6.0253|
|LoRA|4.3388|4.9434|5.8047|

這可以觀察到各訓練目標的影響：預訓練降低 raw loss；SFT 改善指令回答；domain LoRA 改善領域回答。

但資料量很小且模板相似，所以這些結果是機制示範。偏好分數也使用整段回答的 log-probability sum，會受回答長度影響。

---

**第十二步：Inference，使用權重生成文字。**

推理不做 backward，也不更新模型權重。

過程是：

```
Prompt
  → Chat template
  → Token IDs
  → Forward
  → 選出下一個 token
  → 加到序列末端
  → 重複
  → Decode 成文字
```

訓練通常一次計算多個位置；生成則需要依序決定下一個 token，因為後面的輸入依賴剛產生的結果。

Sampling 設定控制如何選 token：

|設定|作用|
|---|---|
|Temperature|調整 logits 尺度；較低通常更集中，較高通常更分散。|
|Top-k|僅保留機率最高的 k 個候選；本實驗 `0` 表示關閉。|
|Top-p|保留累積機率達到門檻的候選集合。|
|Greedy|每次取最高分 token。|
|Max new tokens|限制本次新增 token 數。|
|EOS|生成終止標記。|

```
.\scripts\lab.ps1 infer --run smoke --stage sft `
  --prompt "Explain SFT." --temperature 0.8 --top-p 0.9
```

**KV cache 保存已處理 tokens 的 Key/Value tensors**，讓下一次生成不用重算全部前文的 K/V。它仍需要計算新 token，並讓新的 Query 與前文 Key/Value 做 attention。

這裡有三種完全不同的「cache／保存」：

- 磁碟 `.cache`：套件與資料快取。
- Training checkpoint：長期保存模型和訓練狀態。
- KV cache：一次生成過程中的暫時 attention 狀態。

`*_cache.json` 保存的是 benchmark 結果，不是實際 KV tensors。目前 KV cache 也不跨 API requests 沿用。

---

**第十三步：Deployment 與 Monitoring。**

API 將推理引擎變成可被其他程式呼叫的本機服務：

```
.\scripts\lab.ps1 serve --run smoke --stage dpo --port 8000
```

提供：

|Endpoint|作用|
|---|---|
|`/health`|查看服務與模型狀態。|
|`/v1/chat/completions`|發送單一 user message，取得生成結果。|
|`/metrics`|查看 request 數、生成 tokens 和累積生成時間。|

目前支援非串流、單一 user message，並以 lock 序列化模型生成。它是本機教學部署，不包含完整多使用者排程與公開服務功能。

訓練監控另由 TensorBoard 提供：

```
.\scripts\monitor.ps1 -Run smoke
```

你可以觀察 loss、gradient norm、learning rate，以及各階段的訓練指標。

服務監控的紀錄則存於 `api_requests.jsonl`，保存 token 數、延遲與生成設定，不保存 prompt/response。

**推理和監控不會自動讓模型學習。** 若要形成下一輪 lifecycle，需要另外整理新資料或偏好標註，建立新 run，重新訓練與評估。目前 repo 沒有自動回訓機制。

---

**nanoGPT 對照實驗是在同一份資料上，換一種模型架構重新訓練。**

```
.\scripts\lab.ps1 nanogpt --run smoke
```

它與 MiniMind 共用 tokenizer、token stream 和預訓練設定，但使用 GPT 的 LayerNorm、GELU 與 learned position embeddings 等設計；MiniMind 使用 RMSNorm、RoPE、GQA 和 gated MLP。

因此可以研究「相同資料與訓練規模下，不同架構的表現」，但兩者權重不能互換。nanoGPT 分支目前沒有接上本 repo 的 SFT/DPO/LoRA 主流程，也沒有提供 MiniMind 分支的 KV cache 功能。




#### 詳細解釋parameters的意義及執行順序
```
請詳細解釋configs/smoke.json裡面的parameters的意義及用在哪裡並舉例說明(data是data/demo/pretrain.jsonl裡面有283筆資料):  
"seed", "vocab_size", "block_size"=128, batch_size=4是1 bacth是跑四次pretrain.jsonl data=283x4筆資料是一個batch? pretrain_steps=40等於training stage跑40個batch資料? save_steps/eval_steps是甚麼? eval_batches=8代表eval stage跑8個batch? hidden_size是指token的dimension嗎? num_hidden_layers是指在Transformer有多少個block嗎? max_position_embeddings是甚麼?

在哪裡決定要使用哪個 tokenizer? 每次看多少 tokens? 一次更新使用多少資料? 訓練幾個 steps? 要記錄與比較哪些指標? 那統一資料格式，例如把 `instruction/output` 轉成 `messages`甚麼時候需要做要怎麼做? 做精確去重和分組切分。切分概念是SHA256(seed + normalized document/prompt)→ 固定分配到 train 或 validationg是甚麼意思? 這裡做的是精確去重與相同 prompt 分組，那要怎麼做消除近似、語意相同或模板相似的資料

所以要執行pretrain, sft, dpo, LORA有特別的順序嗎? 然後各要怎麼執行.
```
先釐清你最核心的疑問：

**`batch_size=4` 是一次處理 4 個訓練樣本；在這個 repo 的預訓練中，一個樣本是從 token 串流抽出的文字視窗。`pretrain_steps=40` 則是做 40 次 optimizer 更新。**

因此，這裡一次更新使用的預測目標數為：

```
batch_size × block_size × gradient_accumulation_steps
= 4 × 128 × 1
= 512 個 next-token 預測目標
```

下面都依照你目前 repo 的實際程式解釋，特別是 [smoke.json](C:/LLM/MiniLLM/configs/smoke.json)、[data.py](C:/LLM/MiniLLM/minillm/data.py)、[pretrain.py](C:/LLM/MiniLLM/minillm/pretrain.py) 與 [posttrain.py](C:/LLM/MiniLLM/minillm/posttrain.py)。

---

**一、283 筆原始資料，怎麼變成訓練 batch？**

你目前的實際資料統計保存在 [manifest.json](C:/LLM/MiniLLM/artifacts/smoke/data/manifest.json)：

|處理階段|數量|
|---|---|
|`data/demo/pretrain.jsonl` 原始紀錄|283 筆|
|移除空白紀錄|1 筆|
|移除清理後完全相同的紀錄|1 筆|
|保留下來的文件|281 筆|
|分配到 train|262 筆|
|分配到 validation|19 筆|
|train 編碼後，包含文件結尾 EOS|8,388 tokens|
|validation 編碼後，包含 EOS|678 tokens|

接下來，程式會把 train 文件依序編碼，串接成一條 token 串流：

```
文件 A 的 tokens → EOS →
文件 B 的 tokens → EOS →
文件 C 的 tokens → EOS → ……

共 8,388 tokens
```

保存在：

```
artifacts/smoke/data/pretrain/train.bin
```

**到了預訓練時，程式讀取的是這條 token 串流，不是逐筆讀取原始 283 筆 JSONL。**

`get_batch()` 會隨機選出 4 個起點，再從每個起點取出一段視窗。假設某次抽到：

```
起點 100：取 tokens[100:229]
起點 900：取 tokens[900:1029]
起點 3000：取 tokens[3000:3129]
起點 6000：取 tokens[6000:6129]
```

每段先取 **129 tokens**，才能組成 128 組「目前 token → 下一個 token」：

```
原始視窗：t0, t1, t2, …, t127, t128

輸入 x：  t0, t1, t2, …, t127
目標 y：  t1, t2, t3, …, t128
```

所以一個 batch 的概念是：

```
x.shape = [4, 128]
y.shape = [4, 128]
```

這 4 段可以重疊，也可以跨越文件邊界。這份程式利用 EOS 表示文件結尾，沒有另外建立文件邊界 attention mask。

**283 筆是原始文件數；4 是每個 batch 的視窗數；128 是每個視窗的預測目標數。它們是不同單位。**

---

**二、batch、step、epoch 分別是甚麼？**

|名詞|意義|在此 repo 的例子|
|---|---|---|
|Sample|一個訓練樣本|預訓練：一個 token 視窗；SFT：一組對話；DPO：一組偏好配對|
|Microbatch|一次 forward/backward 使用的樣本集合|`batch_size=4`|
|Optimizer step|累積梯度後，呼叫 optimizer 更新一次|`pretrain_steps=40`|
|Epoch|將訓練資料完整走訪一次|SFT 等使用 DataLoader 時有此概念|

目前：

```
gradient_accumulation_steps = 1
```

所以：

```
1 microbatch → backward → 1 optimizer step
```

如果改成：

```
"batch_size": 4,
"gradient_accumulation_steps": 8
```

則是：

```
microbatch 1：4 段文字 → backward
microbatch 2：4 段文字 → backward
……
microbatch 8：4 段文字 → backward

累積完 → optimizer 更新一次
```

單 GPU 的有效 batch 為：

```
4 × 8 = 32 個樣本／optimizer step
```

預訓練每次更新的預測目標數為：

```
4 × 8 × 128 = 4,096 tokens
```

但 `pretrain_steps=40` 仍然代表 **40 次 optimizer 更新**，此時總共會執行 320 個 microbatches。

對目前 smoke 配置：

```
40 steps × 4 視窗 × 128 預測目標
= 20,480 個訓練目標出現次數
```

相對於 8,388 個 train tokens：

```
20,480 ÷ 8,388 ≈ 2.44
```

這只是約 2.44 倍的 token 訓練量。**因為視窗是隨機抽取、有重疊，所以不能說程式把整份資料完整走訪了 2.44 個 epochs。**

---

**三、`smoke.json` 每個參數的意義，以及用在哪裡**

先看資料與執行設定：

|參數|目前值|意義與程式位置|
|---|---|---|
|`seed`|42|控制資料切分、隨機權重初始化、預訓練視窗抽樣、後訓練資料抽樣等。見 `common.seed_all()` 與 `data.split_rows()`|
|`device`|`"auto"`|有可用 CUDA 時使用 GPU，否則 CPU。見 `common.device_for()`|
|`vocab_size`|1024|訓練 BPE tokenizer 時的目標詞表大小。見 `data.prepare()`|
|`block_size`|128|預訓練每段視窗的預測目標數；SFT／LoRA 的完整序列長度上限；DPO 每個 prompt＋回答分支的長度上限|
|`batch_size`|4|一次 microbatch 的樣本數。預訓練為 4 段視窗，SFT／LoRA 為 4 組對話，DPO 為 4 組偏好配對|
|`gradient_accumulation_steps`|1|累積多少個 microbatches 後更新一次 optimizer|

`seed=42` 的主要作用是讓實驗比較有一致基準。相同資料、設定、軟體與硬體環境下，能盡量重現同樣的初始化、切分與抽樣；單靠 seed 並不保證跨硬體結果逐位元相同。

另外，`vocab_size=1024` 不代表一定產生 1,024 個 tokens。BPE 能學到多少合併，還受到資料量與 `min_frequency=2` 影響。你目前實際得到：

```
requested_vocab = 1024
actual_vocab = 835
```

模型建立時採用的是 **`len(tokenizer)=835`**，因此輸出層會預測 835 種 token。這個 BPE 參數的定義可參考 [Hugging Face Tokenizers 文件](https://huggingface.co/docs/tokenizers/en/api/trainers)。

模型結構參數如下：

|參數|目前值|意義|
|---|---|---|
|`hidden_size`|128|每個 token 的 embedding／hidden state 向量有 128 個維度|
|`num_hidden_layers`|2|堆疊 2 個 Transformer blocks|
|`num_attention_heads`|4|每個 block 使用 4 個 Query attention heads|
|`num_key_value_heads`|2|使用 2 組 Key／Value heads，由 4 個 Query heads 分組共用，稱為 GQA|
|`intermediate_size`|384|Feed-forward network 內部的展開維度|
|`max_position_embeddings`|512|模型位置編碼所支援的序列長度；目前推理程式也以此限制完整 context|

這些參數由 `modeling.fresh_model()` 傳入 MiniMind，實際結構在 [model_minimind.py](C:/LLM/MiniLLM/vendor/minimind/model/model_minimind.py)。

**你的理解是對的：`hidden_size` 是 token 表示向量的維度；`num_hidden_layers` 是 Transformer block 的數量。**

概念上的資料流：

```
token ID
   ↓
Embedding：128 維
   ↓
Transformer block 1
   ↓
Transformer block 2
   ↓
最終 normalization
   ↓
輸出 835 個 vocabulary logits
```

token ID 本身只是一個整數，例如 `317`。Embedding 才把它轉成：

```
[0.012, -0.028, 0.007, ……]  共 128 個數值
```

經過每個 block，同一個 token 的向量會根據前文改變。

每個 attention head 的維度為：

```
hidden_size ÷ num_attention_heads
= 128 ÷ 4
= 32
```

因此每個 token 有：

```
Query：4 heads × 32 維
Key：  2 heads × 32 維
Value：2 heads × 32 維
```

Feed-forward 則大致是：

```
128 維 → 384 維內部運算 → 128 維
```

**`block_size=128` 與 `max_position_embeddings=512` 的差別是：**

```
block_size：
這次訓練使用多長的序列／視窗。

max_position_embeddings：
模型的位置編碼與本 repo 推理介面支援多長的完整序列。
```

MiniMind 使用 RoPE，這裡不是建立一個傳統的可訓練 `512 × 128` position embedding 表，而是預先建立位置旋轉所需的數值。

例如推理時：

```
套用 chat template 後的 prompt：100 tokens
max_new_tokens：48

100 + 48 = 148 ≤ 512 → 允許
```

但只用短視窗訓練，不代表模型在接近 512 tokens 時已經學會良好使用長上下文。

預訓練程式會實際送入 129 tokens，再由模型內部 shift 計算 128 個目標，因此調整設定時也要讓這個額外 token 放得進模型 context。

---

訓練步數與 learning rate：

|參數|目前值|用途|
|---|---|---|
|`pretrain_steps`|40|預訓練總共呼叫 40 次 optimizer 更新|
|`sft_steps`|20|SFT 總共 20 次 optimizer 更新|
|`dpo_steps`|10|DPO 總共 10 次 optimizer 更新|
|`lora_steps`|15|LoRA 總共 15 次 optimizer 更新|
|`pretrain_lr`|0.0005|預訓練 learning rate 設定|
|`sft_lr`|0.0002|SFT learning rate 設定|
|`dpo_lr`|0.00005|DPO learning rate 設定|
|`lora_lr`|0.0005|LoRA learning rate 設定|
|`warmup_ratio`|0.1|前約 10% steps 使用 learning rate warmup|
|`weight_decay`|0.01|AdamW 的權重衰減強度|
|`max_grad_norm`|1.0|將整體梯度 norm 裁切到至多 1.0|

`learning_rate` 控制 optimizer 更新參數的幅度，但 AdamW 也會參考梯度的移動平均，因此不能簡化成「每個權重固定減去 learning rate」。

在原生預訓練迴圈中：

```
40 × 0.1 = 4 個 warmup steps
```

learning rate 先升高，再以 cosine schedule 下降。這份預訓練實作保留設定 learning rate 的 10% 作為下降下限；SFT／DPO／LoRA 使用 HF Trainer 的 cosine scheduler，細節不同。

`max_grad_norm=1.0` 的例子：

```
裁切前 grad norm = 3.0
→ 將所有梯度共同縮放，使整體 norm 約為 1.0
→ optimizer 再使用裁切後的梯度更新
```

log 中的 `grad_norm_before_clip` 記錄裁切前數值，方便觀察梯度是否突然放大。

---

保存與評估設定：

|參數|目前值|意義|
|---|---|---|
|`save_steps`|10|每隔 10 個 optimizer steps 保存 checkpoint|
|`eval_steps`|10|每隔 10 個 optimizer steps 執行 validation 評估|
|`eval_batches`|8|**目前只限制 raw text 評估最多使用的固定視窗數**|

`save_steps=10` 的預訓練產物：

```
base/checkpoint-10/
base/checkpoint-20/
base/checkpoint-30/
base/checkpoint-40/
```

其中包含模型與續訓需要的 optimizer、scheduler、step、RNG 等狀態。stage 根目錄另外保存最終模型。

`eval_steps=10` 則表示：

```
更新到 step 10 → 暫停更新，跑 validation
更新到 step 20 → 暫停更新，跑 validation
……
```

validation 使用 `eval()` 與不計算梯度的模式，**不會更新模型權重**。

預訓練還會在開始前評估一次，並在最後一步評估、保存。

---

**四、`eval_batches=8` 是否代表評估 8 個 batches？**

這個參數名稱容易誤導，必須依目前程式分三種情況看。

**1. Raw text 評估**

`pretrain.raw_loss()` 一次只處理 **1 個固定視窗**，沒有使用訓練的 `batch_size=4`。

你目前 validation 有 678 tokens，程式抽取：

```
起點 0：  tokens[0:129]
起點 128：tokens[128:257]
起點 256：tokens[256:385]
起點 384：tokens[384:513]
起點 512：tokens[512:641]
```

共 5 個視窗，每個計算 128 個預測目標：

```
5 × 128 = 640 個 validation 預測目標
```

所以目前 `eval_batches=8` 的實際效果是：

> 最多評估 8 個固定視窗，但資料只足以提供 5 個。

尾端未形成完整視窗的 tokens 不納入這個 raw loss。每次、每個模型階段都使用相同視窗，便於比較。

**2. SFT／DPO／LoRA 訓練期間的評估**

這些使用 HF Trainer：

```
per_device_eval_batch_size = batch_size = 4
```

而且會跑完整的各自 validation dataset，**不受 `eval_batches=8` 限制**。

|階段|Validation 數量|每 batch 4 筆時|
|---|---|---|
|SFT|18 組對話|5 batches：4＋4＋4＋4＋2|
|DPO|18 組偏好配對|5 batches|
|LoRA|5 組 domain 對話|2 batches：4＋1|

**3. 最後的共同比較**

`evaluate.py` 的 SFT／domain completion loss 和 preference 指標，也使用完整 held-out 資料；其中 completion loss 是逐筆計算，再依回答 token 數彙總。

---

**五、LoRA 與 DPO 的專用參數**

|參數|目前值|意義|
|---|---|---|
|`lora_r`|8|LoRA 更新矩陣的 rank|
|`lora_alpha`|16|LoRA 更新的縮放參數|
|`dpo_beta`|0.1|DPO 相對 reference model 的 log-probability 差值縮放係數|

LoRA 大致使用：

```
W_new = W_original + (alpha / r) × B × A
```

這裡：

```
alpha / r = 16 / 8 = 2
```

假設原本 Linear 的權重是 `128 × 128`：

```
全量權重：128 × 128 = 16,384 parameters

LoRA A：8 × 128
LoRA B：128 × 8

可訓練參數：1,024 + 1,024 = 2,048
```

原始 `W` 凍結，只更新 A、B。目前程式對 attention 與 feed-forward 的指定 projection 加入 LoRA。

DPO 則同時有：

```
policy model：接受更新
reference model：凍結，作為比較基準
```

`beta` 會乘上 policy 相對 reference 的 chosen／rejected 優勢，再進入 DPO loss。它同時影響 loss 尺度與最佳解，不能簡單理解成「越大就一定越準」或「越小就一定改得更多」。實際公式與用途見 [TRL 0.24 DPO 文件](https://huggingface.co/docs/trl/v0.24.0/en/dpo_trainer)。

---

**六、在哪裡決定 tokenizer？各階段會重新訓練嗎？**

決定 tokenizer 的地方是：

[data.py 的 `prepare()` (line 126)](C:/LLM/MiniLLM/minillm/data.py:126)

目前程式明確選擇：

```
Tokenizer(models.BPE(...))
pre_tokenizers.ByteLevel(add_prefix_space=False)
trainers.BpeTrainer(
    vocab_size=cfg["vocab_size"],
    min_frequency=2,
    ...
)
```

所以目前使用的是：

> 在本地資料上重新訓練的 Byte-level BPE tokenizer。

訓練語料只使用：

```
pretrain/train 的 262 筆 raw text
```

沒有使用 validation，也沒有拿 SFT／DPO／domain 回答一起訓練 tokenizer。

保存位置：

```
artifacts/<run>/tokenizer/
```

後續由：

[modeling.py 的 `tokenizer_at()` (line 60)](C:/LLM/MiniLLM/minillm/modeling.py:60)

載入同一個 tokenizer，供初始化、預訓練、SFT、DPO、LoRA、評估、推理共用。

**這條 pipeline 在 `prepare` 時訓練一次 tokenizer；後續各 stage 不重新訓練。**

原因是 token ID 與 embedding 有固定對應。例如：

```
token ID 317 ↔ 某個字串片段 ↔ embedding 第 317 列
```

如果中途换 tokenizer，ID 317 可能變成另一個片段，原來模型學到的對應就會錯亂。

想改成現成 tokenizer，需要修改 `prepare()` 的建立方式，同時確認特殊 tokens、chat template 和模型 vocabulary。從零訓練實驗則應建立新 run，重新初始化模型。

---

**七、每次看多少 tokens、一次更新多少資料？各階段不同**

|階段|一個樣本代表甚麼|`batch_size=4`、accumulation=1 的一次更新|
|---|---|---|
|Pretrain|隨機 token 視窗|4 段，固定共 512 個預測目標|
|SFT|一組 prompt＋answer 對話|4 組對話，實際 token 數隨長度不同|
|DPO|prompt＋chosen＋rejected|4 組偏好配對，policy 與 reference 都需要計算回答分數|
|LoRA|一組 domain prompt＋answer|4 組 domain 對話，只更新 adapter|

SFT／LoRA 的 `block_size=128` 是**長度上限**：

```
chat template + prompt + answer + EOS ≤ 128 tokens
```

例如某個 batch 的 4 筆長度為：

```
32、41、35、50 tokens
```

collator 會 padding 到：

```
input_ids.shape = [4, 50]
```

這裡沒有把每筆補到 128，也沒有 packing 多筆樣本成一段。

此外，prompt 與 padding 的 labels 設為 `-100`，只計算最後 assistant 回答與 EOS 的 loss。Prompt 仍然是模型產生回答所需的上下文。

如果某筆對話超過 128 tokens，目前程式會報錯，要求增加 context 或縮短資料，不會默默截斷。

DPO 長度限制則是分別检查：

```
prompt + chosen + EOS ≤ 128
prompt + rejected + EOS ≤ 128
```

不是把三個欄位全部串成一個 128-token 序列。

---

**八、`instruction/output` 何時轉成 `messages`？要怎麼做？**

在這個 repo，**執行 `prepare` 時已經自動轉換**。

位置是 `data.canonical_row()`，而且在去重、分組、切分之前執行。

原始 SFT 資料：

```
{"instruction":"Calculate 1 + 4. Answer with a number.","output":"5"}
```

會變成：

```
{
  "messages": [
    {
      "role": "user",
      "content": "Calculate 1 + 4. Answer with a number."
    },
    {
      "role": "assistant",
      "content": "5"
    }
  ]
}
```

若有 `input`：

```
{
  "instruction": "將文字翻譯成英文",
  "input": "你好",
  "output": "Hello"
}
```

目前程式會把 user content 組成：

```
將文字翻譯成英文
你好
```

核心做法是：

```
prompt = row["instruction"]

if row.get("input"):
    prompt += "\n" + row["input"]

messages = [
    {"role": "user", "content": clean(prompt)},
    {"role": "assistant", "content": clean(row["output"])}
]
```

此 repo 也接受現成的 `messages`，不需要先另寫轉檔程式。TRL 支援多種資料格式；這裡為了統一清理、切分和 loss masking，先轉成自己的 canonical 格式，再預先 tokenize 後交給 Trainer。相關格式見 [TRL 0.24 SFT 文件](https://huggingface.co/docs/trl/v0.24.0/en/sft_trainer)。

不同訓練目標保留不同格式：

```
Pretrain：{"text": "..."}
SFT／domain：{"messages": [...]}
DPO：{"prompt": "...", "chosen": "...", "rejected": "..."}
```

若資料集用 `question/answer`、`human/gpt` 等不同欄位或角色名稱，就要先轉成程式支援的欄位／角色，或擴充 `canonical_row()`。

多輪 `messages` 也可以使用，但目前只把**最後一個 assistant 回答**作為 completion，前面的訊息都作為 context。

---

**九、精確去重與「SHA256 分组切分」是甚麼？**

它們是兩個不同動作。

**精確去重：清理後，整筆紀錄完全相同才移除。**

先做：

```
Unicode NFKC 正規化
→ 統一換行
→ 移除部分控制字元
→ 整理空白
→ 移除空資料
```

再把紀錄轉成欄位順序固定的 JSON 字串，比較是否相同。

例如：

```
{"instruction":"解釋 LoRA","output":"LoRA 使用低秩矩陣微調。"}
{"instruction":"  解釋 LoRA  ","output":"LoRA 使用低秩矩陣微調。"}
```

清理後相同，留下其中一筆。

但如果同一問題有不同回答：

```
{"instruction":"解釋 LoRA","output":"LoRA 使用低秩矩陣微調。"}
{"instruction":"解釋 LoRA","output":"LoRA 凍結原始權重，只更新 adapter。"}
```

會保留兩筆，因為整筆內容不同。

**分組切分：相同問題的所有回答，放在同一側。**

上面兩筆會形成同一個 prompt group：

```
group =「解釋 LoRA」
  ├─ 回答 A
  └─ 回答 B
```

整組分配到 train 或 validation，避免：

```
train 已看過問題＋回答 A
validation 再用相同問題＋回答 B
```

對多輪 SFT，group key 是最後回答之前的完整 context。對 pretrain，key 是清理後的完整文件；對 DPO，key 是 prompt。

單輪 user prompt 在 SFT／domain／DPO 使用相同 key，所以同一個精確 prompt 在這些資料集中會分到同一側。

切分程式的核心是：

```
digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
number = int.from_bytes(digest[:8], "big")
bucket = number % 10

if bucket == 0:
    validation
else:
    train
```

例如你問過的這個 prompt：

```
Calculate 2 + 3. Answer with a number.
```

用 seed 42 計算：

```
SHA256 的輸入：
42:Calculate 2 + 3. Answer with a number.

前 8 bytes：
b9750d778ffc1b10

轉成整數後 % 10：
2

因此分到 train
```

SHA256 在此提供的是**固定的分桶規則，不是在判斷語意**。

它的優點是：

- 相同 seed＋key 永遠分到同一側。
- 重排資料列，不會改變已有群組的歸屬。
- 加入新資料，不會重新洗牌所有舊資料。
- 同一 group 不會跨越 train／validation。

10 個 buckets 中選 1 個當 validation，所以比例期望約 10%，但不是強制精確 90／10。你目前是：

```
train：262
validation：19

validation 比例約 6.76%
```

還要留意：**這只能防止相同 key 跨 split，不能保證預訓練語料沒有包含 validation 問題的答案。** 示範資料的 raw arithmetic 文件，本來就包含與 SFT 題目相關的算式，所以這些評估適合驗證流程。

---

**十、怎麼消除近似、語意相同、模板相似資料？**

目前 repo 尚未實作這三類去重。可以在 `canonical_row()` 清理完成後、正式 train／validation 切分前加入。

建議把「刪掉冗餘資料」與「讓相似資料同側」分開處理：

```
清理與統一格式
→ 精確去重
→ 近似／語意／模板比對
→ 確認哪些是真的冗餘，選擇保留樣本
→ 對仍保留的相關樣本建立 group
→ 以 group key 分配 train／validation
→ 訓練 tokenizer
```

|類型|例子|可採用的方法|
|---|---|---|
|文字近似|同一篇文章改幾個字、加編號|字元／詞 n-gram、Jaccard；大資料量用 MinHash＋LSH|
|語意重複|「解釋預訓練」與「甚麼是從零訓練語言模型？」|用獨立 embedding model 產生向量，再找相似候選|
|模板相似|`Lab question 0/1/2: explain LoRA.`|移除非語意性的編號、網址、樣板欄位，建立模板 key|

近似比對可以先切成字元 n-grams，計算：

```
Jaccard(A, B) = 共有 n-grams 數量 ÷ 合併後 n-grams 數量
```

在 283 筆的小資料上，可以直接兩兩比較，最多約 4 萬組。資料大量增加後，再用 MinHash／LSH 先找候選，避免所有資料兩兩比較；[Hugging Face DataTrove](https://github.com/huggingface/datatrove) 提供相關 pipeline 範例。

語意比對則需要一個額外的 embedding model：

```
文本 A → 向量 A
文本 B → 向量 B
→ cosine similarity
→ 相似候選
→ 確認是否冗餘
```

這個模型只用於資料處理；你的 MiniMind 仍然可以從隨機權重開始訓練。向量相似度只是候選判斷依據，需要人工抽查或其他規則確認，尤其要保護否定詞、數字與不同事實。這類方法可參考 [SemDeDup 論文](https://arxiv.org/abs/2303.09540)。

你目前 demo 很適合觀察模板重複：

```
Lab question 0: explain Transformer.
Lab question 8: explain Transformer.
```

移除 `Lab question <number>:` 後，它們的問題与回答相同，可以考慮只保留一筆。

但：

```
Calculate 1 + 2.
Calculate 1 + 3.
```

雖然模板相似，答案不同，數字又是任務核心，不能直接刪成一筆。

這類資料可依評估目標選擇：

```
測試同一題目的不同說法：
相同運算元的改寫題目放同一組。

測試對新數值的泛化：
保留部分數值或運算元組合，專門作 validation／test。
```

也不宜把所有算術題僅因共用同一模板就全部放進一個 group，否則可能無法形成有效的 train／validation 切分。

若要擴充 repo，主要修改 `data.split_rows()` 的分組邏輯，並在 manifest 增加去重方法、門檻、保留策略與移除數量。新流程也應以新 run 重新準備資料。

---

**十一、要記錄與比較哪些指標？**

目前已有的訓練記錄：

|指標|用途|
|---|---|
|Train loss|訓練目標是否下降|
|Validation loss|在未用來更新權重的資料上表現如何|
|Gradient norm|是否出現梯度突然放大、長期大量裁切|
|Learning rate|對照 warmup／scheduler 與 loss 變化|
|Tokens／samples per second|訓練吞吐量|
|Runtime|實驗所需時間|
|Peak VRAM|比較各階段配置的 GPU 記憶體需求|
|SFT token accuracy／entropy|觀察回答 token 的預測行為|
|DPO chosen／rejected rewards、margin|觀察偏好學習是否朝 chosen 移動|

記錄位置：

```
artifacts/<run>/<stage>/metrics.jsonl
artifacts/<run>/<stage>/final_metrics.json
artifacts/<run>/tensorboard/<stage>/
```

跨階段比較則使用同一評估基準：

|比較項目|希望觀察的改變|
|---|---|
|Raw validation loss／perplexity|預訓練是否改善語言建模|
|SFT completion loss|是否更會產生指定回答|
|Domain completion loss|LoRA 是否改善領域回答|
|Chosen win rate／log-prob margin|偏好回答是否更受模型偏好|
|固定 prompts 的生成結果|實際輸出是否符合期待|
|KV cache token 一致性與時間|cache 是否正確、是否加速|
|參數數量與可訓練比例|比較全量更新與 LoRA|

目前結果可見 [comparison.md](C:/LLM/MiniLLM/artifacts/smoke/evaluation/comparison.md)。例如：

```
initialized raw loss：6.7693
base raw loss：       4.2932
```

表示在同一 raw validation protocol 上，預訓練有改善。

```
base SFT completion loss：6.4739
sft SFT completion loss： 4.8764
```

表示 SFT 改善了這份 instruction／response 評估。

**不同訓練目標的 loss 不能直接比數字大小。** DPO loss 0.69 與預訓練 loss 4.29，意義與尺度不同。

如果目標是算術正確性，還應另外加入「回答是否等於正確數值」的 task accuracy。目前的 token loss 或 chosen win rate，不能直接取代這個指標。

---

**十二、Pretrain、SFT、DPO、LoRA 有固定順序嗎？**

對從零建立模型的主要教學流程，建議：

```
prepare
   ↓
隨機初始化 initialized
   ↓
pretrain → base
   ↓
SFT → sft
   ├─ DPO → dpo
   └─ 領域 LoRA → lora
```

這也正是目前 repo 的預設：

|執行階段|預設 parent|資料|
|---|---|---|
|Pretrain|initialized|pretrain|
|SFT|base|sft|
|DPO|sft|dpo|
|LoRA|sft|domain|

所以即使先執行 DPO，再執行 LoRA，**預設 LoRA 仍從 SFT 載入，不會自動接在 DPO 後面。**

概念上：

- Pretraining：訓練目標是 next-token prediction。
- SFT：訓練目標是學習指定的回答。
- DPO：訓練目標是學習 chosen／rejected 偏好。
- LoRA：決定用甚麼方式更新參數。

因此 LoRA 可以用來執行 SFT 或 DPO，並不一定是一個最後才做的階段。

不過目前 repo 的 `lora` 命令具體代表：

> 使用 domain instruction／response 資料，做 LoRA SFT。

目前 `dpo` 命令做的是全量參數更新；要做 LoRA DPO，需要另外擴充 `posttrain.py`。

你也可以選擇：

```
base → SFT → DPO → domain LoRA
```

對應命令：

```
.\scripts\lab.ps1 lora --run <run名稱> --parent dpo
```

領域微調可能改變原本的偏好表現，因此最後仍要重新評估兩者。

---

**十三、各階段怎麼執行？**

在 `C:\LLM\MiniLLM` 的 PowerShell 執行。下面用新的實驗名稱，保留已完成的 `smoke` 結果：

```
$labRun = 'lesson-20261013'
```

如果這個名稱已使用，換成另一個未使用的名稱。

**1. 準備資料、去重、切分、訓練 tokenizer**

```
.\scripts\lab.ps1 prepare --run $labRun --config .\configs\smoke.json
```

預設讀取：

```
data/demo/pretrain.jsonl
data/demo/sft.jsonl
data/demo/dpo.jsonl
data/demo/domain.jsonl
```

輸出：

```
artifacts/lesson-20261013/config.json
artifacts/lesson-20261013/data/
artifacts/lesson-20261013/tokenizer/
```

**2. 建立與保存隨機初始化模型**

```
.\scripts\lab.ps1 init --run $labRun
```

保存 `initialized/`，方便與後續模型比較。

**3. 執行預訓練**

```
.\scripts\lab.ps1 pretrain --run $labRun
```

使用 40 steps，輸出 `base/`。

**4. 從 base 執行 SFT**

```
.\scripts\lab.ps1 sft --run $labRun --parent base
```

使用 20 steps，輸出 `sft/`。

**5. 從 SFT 執行 DPO**

```
.\scripts\lab.ps1 dpo --run $labRun --parent sft
```

使用 10 steps，輸出 `dpo/`。

**6. 從 SFT 建立 domain LoRA 分支**

```
.\scripts\lab.ps1 lora --run $labRun --parent sft
```

使用 15 steps，輸出 `lora/` adapter。

如果你希望 LoRA 接在 DPO 後面，就在執行此階段前把 parent 改成 `dpo`。同一個 run 的 `lora/` 已完成後，程式會阻止覆寫。

**7. 重新產生共同比較**

```
.\scripts\lab.ps1 evaluate --run $labRun
```

結果位於：

```
artifacts/lesson-20261013/evaluation/comparison.md
```

各個訓練命令也會自動評估剛完成的 stage，並保留其他已保存的比較結果。

**8. 推理比較**

Base 可以先觀察原始文字續寫：

```
.\scripts\lab.ps1 infer --run $labRun --stage base --raw-prompt --prompt "Arithmetic example:" --greedy
```

再比較聊天階段：

```
.\scripts\lab.ps1 infer --run $labRun --stage sft --prompt "Calculate 2 + 3. Answer with a number." --greedy

.\scripts\lab.ps1 infer --run $labRun --stage dpo --prompt "Calculate 2 + 3. Answer with a number." --greedy

.\scripts\lab.ps1 infer --run $labRun --stage lora --prompt "Explain LoRA." --greedy
```

**9. 查看訓練曲線**

在另一個 PowerShell 視窗：

```
.\scripts\monitor.ps1 -Run lesson-20261013
```

如果想一次執行預設主流程：

```
.\scripts\lab.ps1 all --run lesson-all-20261013 --config .\configs\smoke.json
```

它依序執行：

```
prepare → init → pretrain → SFT → DPO → LoRA(parent=SFT) → compare
```

nanoGPT 是另外的架構對照分支：

```
.\scripts\lab.ps1 nanogpt --run $labRun
.\scripts\lab.ps1 evaluate --run $labRun
```

最後，`prepare` 會把設定保存到該 run 的 `config.json`，後續訓練使用這份設定。**修改原始 `configs/smoke.json` 不會改變已準備好的 run；調整詞表、模型結構、context 或資料處理規則時，應建立新 run。**




#### 怎麼訓練 Tokenizer
```
請詳細介紹要怎麼訓練 Tokenizer, 怎麼理解這句話: Token 不一定是一個完整單字，也不一定是一個中文字；byte-level BPE 可以把文字拆成較小片段，再學習常見片段的合併規則, 以及在repo的prepare(). 1. 建立 byte-level BPE。

2. 使用完整 byte alphabet。

3. 只從 pretraining 的 train partition 學習 vocabulary。

4. 加入 PAD、BOS、EOS 和 chat special tokens。

5. 保存 tokenizer 和 chat template。

以具體例子詳細介紹. 以及是否有通用 tokenizer, 不然如果做transfer learning或因要對齊兩個model或其他原因換 tokenizer, 那不就embedding 的每一列就可能對應到不同文字, 要怎麼解決這類問題?
```
訓練 Tokenizer 的核心目標是建立一本「字典」，讓模型能將人類的文字轉換成神經網路能理解的數字（Token IDs）。

### 第一部分：理解「Token 不一定是完整單字或中文字，Byte-level BPE 是如何運作的」

**Byte-Level BPE (Byte-Pair Encoding)** 的核心思想是：**由下而上，從最基本的 Byte（位元組）開始，統計相鄰片段出現的頻率，將最常一起出現的片段合併成一個新的 Token。**

#### 具體例子解析：

1. **為什麼不一定是完整單字（以英文為例）：**
    
    - 假設我們有一句話："unhappiness is everywhere"。
        
    - 如果字典裡沒有 "unhappiness" 這個完整單字，BPE 演算法在訓練時會發現 "un"（字首）、"happi"（字根）、"ness"（字尾）這幾個片段在語料庫中很常出現。
        
    - 因此，"unhappiness" 可能會被切分成三個 Token：`["un", "happi", "ness"]`。
        
    - **好處：** 模型可以用有限的字典大小（例如 32,000 個 Token），透過組合字根字首，去理解並生成無限的罕見字或新造詞，而不會遇到 OOV（Out-Of-Vocabulary，未知詞）問題。
        
2. **為什麼不一定是完整中文字（以中文與 Byte 為例）：**
    
    - 在電腦底層，文字是透過 UTF-8 編碼儲存的。一個英文字母佔 1 個 Byte，而**一個中文字通常佔 3 個 Bytes**。
        
    - 例如：「我」的 UTF-8 十六進位編碼是 `E6 88 91`（三個 Bytes）。
        
    - **常見字：** 如果「我」在訓練語料中出現頻率極高，BPE 會將 `E6`、`88`、`91` 這三個 Bytes 合併，並在字典中給它一個獨立的 Token ID（例如 ID: 5312）。這時，一個 Token 就是一個完整的中文字。
        
    - **罕見字或 Emoji：** 假設遇到極罕見的字「龘」（編碼為 `E9 BE 98`）。因為出現頻率太低，BPE 演算法沒有將它們合併。這時，「龘」就會被拆解成 3 個基礎的 Byte Token：`[<Byte_E9>, <Byte_BE>, <Byte_98>]`。這時，Token 就不是一個完整的中文字，而是位元組片段。
        

### 第二部分：詳解 `prepare()` 腳本中的 5 個步驟

這五個步驟是從零訓練一個標準 LLM Tokenizer 的標準流程（例如 Llama、Qwen 等模型底層的做法）：

#### 1. 建立 byte-level BPE

- **實作意義：** 選擇 BBPE 演算法作為基礎架構。相比於 WordPiece（BERT 使用）或 Unigram，BBPE 能完美處理多語言和程式碼，並且透過 Byte 級別保證資訊不遺失。
    

#### 2. 使用完整 byte alphabet (0~255)

- **實作意義：** 字典的最初始狀態（Base Vocabulary）就是這 256 個基本的 Bytes（`0x00` 到 `0xFF`）。
    
- **具體例子：** 即使模型遇到完全沒見過的外星文字或亂碼，只要它是電腦檔案，底層一定是這 256 個 Bytes 的組合。這樣做徹底消滅了 `<UNK>`（未知 Token），確保模型可以編碼與解碼任何二進位資料。
    

#### 3. 只從 pretraining 的 train partition 學習 vocabulary

- **實作意義：** 嚴格遵守機器學習的資料隔離原則（Data Leakage Prevention）。
    
- **具體例子：** Tokenizer 也是從資料中「學習」出來的（學習哪些片段最常合併）。如果你把 Validation 或 Test set 的資料也拿去訓練 Tokenizer，Tokenizer 就會對測試集中的特定專有名詞產生「過度優化」的合併規則，導致評估模型真實泛化能力時失準。
    

#### 4. 加入 PAD、BOS、EOS 和 chat special tokens

- **實作意義：** 語言模型需要結構化的控制訊號，這些訊號不屬於日常自然語言，必須手動強行加入字典，並且**設定為不可被分割（never_split）**。
    
- **具體例子：**
    
    - `<pad>`：補齊長度用的填充符號（Padding）。
        
    - `<s>` (BOS - Begin of Sentence)：告訴模型句子開始。
        
    - `</s>` (EOS - End of Sentence)：告訴模型句子結束，模型生成這個 Token 時就會停止輸出。
        
    - `<|im_start|>` / `<|im_end|>`：Chat 模式專用的角色標記。例如 `<|im_start|>user\n你好<|im_end|>`，幫助模型區分人類提示和系統預設詞。
        

#### 5. 保存 tokenizer 和 chat template

- **實作意義：** 訓練完成後，將合併規則（Merges）和字對應表（Vocabulary）存成檔案（通常是 `tokenizer.json` 或 `tokenizer.model`）。同時保存 Chat Template（通常是 Jinja2 語法），確保推論時，使用者的對話能被正確格式化加上 Special Tokens。
    

### 第三部分：是否有「通用 Tokenizer」與 Transfer Learning 的 Embedding 對齊問題

你的直覺**完全正確**。如果強行換掉 Tokenizer，原本 Pre-trained Model 的 Embedding Layer（詞向量矩陣）每一列對應的意義就會徹底錯亂，模型會瞬間變成智障，因為權重和詞彙對不上了。

#### 1. 有沒有通用的 Tokenizer？

**嚴格來說，沒有單一的通用 Tokenizer。** 每個模型家族都有自己訓練出來的 Tokenizer。

- OpenAI 有 `tiktoken` (例如 `cl100k_base`，涵蓋 10 萬個 Token)。
    
- LLaMA 有自己的 SentencePiece 模型 (32k 詞表)。
    
- Qwen 有自己優化的 BBPE (高達 152k 詞表，對多語言支援極佳)。
    

雖然沒有絕對通用，但很多新模型為了省麻煩或為了相容性，會**直接沿用** OpenAI 的 `tiktoken` 規則，或者 LLaMA 的 Tokenizer。

#### 2. 如果要做 Transfer Learning 或 Domain Adaptation，該怎麼解決 Embedding 對不上的問題？

業界解決這類問題，絕對不會「憑空換掉」整個 Tokenizer，而是採用「詞表擴充」（Vocabulary Extension）的策略。

**具體解決方案與例子（以 Chinese-LLaMA 為例）：** 原始的 LLaMA 主要是針對英文訓練的，詞表大小只有 32,000。它懂中文，但效率極差（一個中文字可能會被拆成 3 個 Byte Tokens，導致算力浪費且 Context Length 縮水）。

如果你想把 LLaMA 遷移到中文領域（Transfer Learning），你會這樣做：

1. **保留原始 Tokenizer 與 Embedding：** 原來的 32,000 個 Tokens 及其在模型中的 Embedding 矩陣（Shape: `[32000, 4096]`）**完全保持不變**。這保留了模型原本強大的英文與邏輯能力。
    
2. **訓練並加入新的 Tokens：** 拿大量的中文語料去訓練一個新的 Tokenizer，選出最常見的 20,000 個中文詞彙（例如：「人工」、「智能」、「台灣」）。
    
3. **合併 Tokenizer（擴充詞表）：** 把這 20,000 個新詞彙加到原來的字典後面，新的 Tokenizer 詞表大小變成 `52,000`。
    
4. **擴充 Embedding 矩陣（關鍵對齊步驟）：** 將模型的 Embedding 矩陣擴充為 `[52000, 4096]`。
    
    - 前 32,000 列：直接複製 LLaMA 原本的預訓練權重。
        
    - 後 20,000 列（新加的中文詞）：通常會**將它們初始化為組成這些詞的原本 Byte Token 的 Embedding 平均值**（Mean Pooling），或者採用較小的隨機常態分佈初始化。
        
5. **繼續 Pre-training (Continual Pre-training, CPT)：** 凍結或以極小的 Learning Rate 更新前 32k 的權重，重點訓練那 20,000 個新詞的 Embedding，讓模型學會將新詞向量對齊到原有的語義空間中。
    

總結來說：**你無法在保留模型權重的情況下「替換」Tokenizer，你只能**「擴充」它。擴充時，舊矩陣不動，新矩陣追加在後方並重新訓練對齊，這是目前 LLM 在做跨語言或特殊領域（如醫療、程式碼）遷移學習的標準作法。





```

```
