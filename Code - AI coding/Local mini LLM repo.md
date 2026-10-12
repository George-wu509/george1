
|     |     |
| --- | --- |
|     |     |


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