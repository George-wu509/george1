
|                                       |     |
| ------------------------------------- | --- |
| [[#### LLM Inference / Production內容]] |     |
|                                       |     |
|                                       |     |

#### LLM Inference / Production內容
```
請完整深入解釋以下LLM Inference / Production內容(先以介紹intern了解知識, 之後則更深入完整介紹engineer技術細節)：
|技術	|Senior 應理解的問題|
|vLLM / TensorRT-LLM	|如何提高模型 Serving Efficiency|
|Continuous Batching	|如何同時處理大量 Requests|
|KV Cache		|GPU Memory 與 Decode Efficiency|
|Quantization		|INT8／INT4 的效能與品質交換|
|Model Parallelism	|Tensor、Pipeline、Data Parallelism|
|Latency		|TTFT、Time per Output Token、P95|
|Throughput		|Tokens/sec、Concurrent Users|
|Cloud Deployment	|Scaling、Monitoring、Rollback|
|Reliability		|Rate Limits、Retries、Fallback、Circuit Breakers|
|Cost Optimization	|Model Routing、Caching、Token Budget|
```

# LLM Inference / Production 完整技術解析

從 Intern 基礎知識 → Senior AI Engineer 技術深度 → Production System Design → 實際部署與效能優化

LLM Inference / Production 是 2026 年 Senior LLM Engineer、Senior Applied AI Engineer、LLM Infrastructure Engineer，以及 AI Platform Engineer 非常重要的核心能力。

這個領域與 LLM Training 最大的不同在於：

- Training： 如何讓模型學得更好，包含 Pretraining、Fine-tuning、RLHF。
    
- Inference： 模型訓練完成後，如何快速、有效率地產生回答。
    
- Production Serving： 如何讓成千上萬名使用者同時使用模型，維持低延遲、高可靠性、合理成本與安全性。
    

Senior Engineer 不只需要知道如何執行一個 LLM，還需要能回答：

> 公司準備部署一個 70B LLM，預計每天有 100,000 個 Requests。如果希望 P95 TTFT 小於 2 秒、每個使用者都能流暢取得回答，要使用多少 GPU？如何配置 vLLM、KV Cache、Batching、Quantization？如何 Scaling、Monitoring、Rollback？每個 Request 的成本是多少？

這才是 LLM Inference / Production System Design 的核心。

## Part 1 — Intern Level：先建立完整概念

### 1. LLM Inference 是什麼？

假設使用者輸入：

> Explain how a Rolex automatic movement works.

一個 LLM 要產生回答，基本流程如下：

User Request

Explain how a Rolex automatic movement works.

Tokenization

將文字轉換成 Token IDs

Prefill Phase

Transformer 同時處理輸入 Tokens，計算 Attention 並建立 KV Cache

Decode Phase

根據先前 Tokens 與 KV Cache，逐步生成下一個 Token

An → automatic → movement → uses → ...

Streaming Response

逐步將文字傳回使用者，不必等全部回答完成

這裡最重要的是 Prefill 和 Decode 是兩種不同的 GPU 工作負載。

|階段|Prefill|Decode|
|---|---|---|
|目標|理解全部輸入 Context|逐個產生輸出 Token|
|平行程度|大量輸入 Tokens 可以一起計算|同一序列通常需要逐步生成|
|主要限制|常偏向 Compute-bound|常偏向 Memory-bandwidth-bound|
|重要指標|TTFT|TPOT / ITL|
|常用優化|Chunked Prefill、Prefix Cache|KV Cache、Batching、Speculative Decoding|

這個區分非常重要，因為後面所有的效能優化幾乎都圍繞這兩個階段展開。

### 2. 用餐廳理解 LLM Serving

想像 LLM 是一家餐廳：

- GPU 是廚房設備。
    
- Model Weights 是廚師需要使用的完整食譜。
    
- User Requests 是客人的訂單。
    
- Prefill 是閱讀訂單、準備食材。
    
- Decode 是一道一道製作餐點。
    
- KV Cache 是已完成準備、可以直接使用的中間成果。
    
- Continuous Batching 是讓廚房動態安排每個訂單，不用等整批訂單做完。
    
- Load Balancer 是安排客人到不同廚房。
    
- Autoscaling 是客人增加時開設更多廚房。
    
- Quantization 是讓食譜與設備所需的資源更少，但必須確認品質不受太大影響。
    

如果一家餐廳有 100 個客人，工程師真正要解決的不是「廚師能不能煮出一道菜」，而是如何安排整個廚房，讓等待時間短、處理量高、品質一致，而且不浪費成本。

### 3. 十項技術的整體關係

|技術|Intern 應理解|Senior 應深入掌握|
|---|---|---|
|vLLM / TensorRT-LLM|高效率執行 LLM 的 Serving Engine|GPU Kernels、Scheduler、Memory Management|
|Continuous Batching|動態合併多個請求|Request Scheduling、Admission Control、Prefill/Decode Interference|
|KV Cache|避免重算已經處理過的 Tokens|GQA、PagedAttention、Memory Estimation、Eviction|
|Quantization|用較少 Bits 儲存模型|W4A16、W8A8、Calibration、Quality Regression|
|Model Parallelism|多 GPU 一起執行模型|TP、PP、DP、NCCL、Communication Bottlenecks|
|Latency|使用者等待多久|TTFT、TPOT、ITL、P95/P99、Latency Breakdown|
|Throughput|每秒能處理多少工作|Output Tokens/sec、Requests/sec、Saturation|
|Cloud Deployment|在雲端提供服務|GPU Kubernetes、Autoscaling、Canary、Rollback|
|Reliability|故障時維持服務|Timeout、Retry、Fallback、Circuit Breaker|
|Cost Optimization|降低服務成本|Routing、Caching、Batch Size、Cost per Successful Task|

接下來進入真正需要具備的工程技術細節。

## Part 2 — Senior Engineer Level：Inference Engine 與 GPU Optimization

## 4. vLLM / TensorRT-LLM：如何提高 Serving Efficiency？

### 4.1 先理解為什麼不能直接用 PyTorch Serving

假設已經有一個 Hugging Face Transformer Model。

最簡單的方式是：

```
from transformers import AutoTokenizer, AutoModelForCausalLMmodel_id = "Qwen/Qwen2.5-7B-Instruct"tokenizer = AutoTokenizer.from_pretrained(model_id)model = AutoModelForCausalLM.from_pretrained(    model_id,    device_map="auto",    torch_dtype="auto")inputs = tokenizer(    "Explain how an automatic watch works.",    return_tensors="pt").to(model.device)outputs = model.generate(    **inputs,    max_new_tokens=200)print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

這可以成功執行 LLM Inference，但不代表適合高流量的 Production。

問題在於：

1. 如果每個 Request 都獨立執行，GPU 計算資源容易閒置。
    
2. 不同 Requests 有不同 Prompt Length 和 Output Length。
    
3. 長時間執行的 Requests 可能妨礙其他 Requests。
    
4. 大量 Context 需要佔用 GPU 記憶體。
    
5. 模型生成每個 Token 都可能受到 GPU Memory Bandwidth 限制。
    
6. 當使用者數量增加時，需要有效控制排隊、記憶體、Batch Size 和 GPU 資源。
    

一般的 `model.generate()` 並非完全不能進行 batching 或部署，但它本身不提供與專用 Serving Engine 相同程度的整體請求排程、KV Cache 管理及 Production Optimization。

### 4.2 vLLM 是什麼？

vLLM 是專門為 LLM Inference 設計的 Serving Engine。

它主要透過以下方法提升效能：

- Continuous Batching
    
- Paged KV Cache Management
    
- Optimized Attention Kernels
    
- Prefix Caching
    
- Chunked Prefill
    
- Quantization
    
- Tensor / Pipeline Parallelism
    
- Speculative Decoding
    

現行 vLLM 也支援 Disaggregated Prefill/Decode 等較進階的 Serving 架構。

![](https://www.google.com/s2/favicons?domain=https://docs.vllm.ai&sz=32)

vLLM

+1

其中最著名的技術之一是 PagedAttention，但不要把 vLLM 理解成只有一個 PagedAttention Algorithm。它是一整套包含 Scheduler、Model Runner、KV Cache Manager 與 API Server 的系統。

### 4.3 TensorRT-LLM 是什麼？

TensorRT-LLM 是 NVIDIA 提供的 LLM Inference Framework。

特別針對 NVIDIA GPU 架構進行效能優化，例如：

- Optimized CUDA Kernels
    
- Optimized GEMM (Matrix Multiplication)
    
- In-flight Batching
    
- Quantized Inference
    
- Optimized Attention
    
- KV Cache Management
    
- Multi-GPU Parallelism
    
- Speculative Decoding
    

TensorRT-LLM 支援 NVIDIA GPU 上多種精度與執行最佳化方式，例如 FP8、FP4、AWQ、GPTQ，以及部分針對特定硬體的低精度 Kernels。

![](https://www.google.com/s2/favicons?domain=https://developer.nvidia.com&sz=32)

NVIDIA Developer

+1

### 4.4 vLLM vs TensorRT-LLM

|比較|vLLM|TensorRT-LLM|
|---|---|---|
|主要定位|彈性、高效率的 LLM Serving|NVIDIA GPU 優化的 LLM Runtime|
|Model Integration|Hugging Face 生態整合方便|支援多種模型及 NVIDIA 最佳化路徑|
|GPU|多種後端，依支援矩陣而定|NVIDIA GPU 為核心|
|Batching|Continuous Batching|In-flight Batching|
|KV Cache|Paged KV Cache|Contiguous / Paged KV Cache|
|Quantization|INT4、INT8、FP8 等|FP4、FP8、INT4 等|
|適合場景|快速導入、靈活部署、多模型實驗|深入優化 NVIDIA GPU 工作負載|
|是否一定比較快|不一定|不一定|

Senior Engineer 必須理解：不能直接宣稱 TensorRT-LLM 一定比 vLLM 快，或反過來。

實際效能取決於 GPU 架構、模型、Context Length、Batching、Quantization、Kernel Implementation、Engine Version 及 Traffic Pattern。

例如：

- 8B 模型、低 Concurrency、短 Context
    
- 70B 模型、高 Concurrency、長 Context
    
- Mixture-of-Experts 模型
    
- FP8 或 INT4 模型
    

這些情況可能產生不同的結果。

正確工程方法是對相同工作負載進行 Benchmark，而不是只根據某個 Framework 的理論性能選擇。

## 5. Continuous Batching：如何同時處理大量 Requests？

這是 LLM Serving 最重要的技術之一。

### 5.1 Static Batching 的問題

假設 GPU 一次處理三個 Requests：

|Request|需要產生的 Tokens|
|---|---|
|A|100|
|B|20|
|C|200|

傳統 Static Batching 可能將三個 Requests 組成固定 Batch。

問題是 B 很快完成，A 後來完成，但是 C 還在執行。

如果系統需要整個 Batch 結束才安排新的 Requests，GPU 的工作效率就會降低。

### 5.2 Continuous Batching 如何運作？

Continuous Batching 允許 Scheduler 在 Generation Iteration 之間動態加入及移除 Requests。

Request Scheduling 示意圖

每個方塊代表一段推論排程期間，並非固定長度的真實 GPU Kernel。

Request

T1

T2

T3

T4

T5

T6

A

B

C

D

E

B 完成後，D 可以加入；A 完成後，E 可以加入。其他尚未完成的 Request 不必重新開始。

這種設計也稱為 Iteration-level Scheduling。

### 5.3 Scheduler 實際需要管理什麼？

Senior Engineer 要理解 Scheduler 並不只是把多個 Requests 放進 Batch。

它需要管理：

Running Queue

目前正在 GPU 中處理的 Requests。

Waiting Queue

等待分配計算或 KV Cache 資源的 Requests。

Admission Control

判斷是否有足夠資源接受新 Request。

Preemption

如果某個 Request 無法繼續分配所需資源，可能需要暫停，並依 Engine 設計重新計算或移動部分狀態。

Fairness

避免長 Context Requests 長期阻擋短 Requests，或高優先級 Requests 讓其他工作持續挨餓。

### 5.4 Chunked Prefill

假設某 Request 輸入了 32,000 Tokens。

如果一次執行全部 Prefill，可能產生很大的 GPU Compute Spike，拖慢同時正在生成回答的 Requests。

Chunked Prefill 把長 Prompt 拆成多個較小的 Prefill 工作單位。

例如：

```
32,000-token Prompt
       |
       v
Chunk 1: 8,000
Chunk 2: 8,000
Chunk 3: 8,000
Chunk 4: 8,000
```

實際 Chunk Size 由 Token Budget、Scheduler 和配置決定。

這讓 Prefill 工作較容易與 Decode 工作交錯安排。

好處包括：

- 降低長 Prefill 對其他 Requests 的干擾。
    
- 改善 GPU Scheduling Flexibility。
    
- 在某些工作負載下增加 Throughput。
    

但如果 Chunk Size 設定不當，也可能讓 Prefill Completion Time 增加，因此仍需 Benchmark。

TensorRT-LLM 與 vLLM 都具有這類 Prefill/Batching 優化機制。

![](https://www.google.com/s2/favicons?domain=https://github.com&sz=32)

GitHub

+1

### 5.5 Senior Interview 問題

> If increasing batch size improves throughput, why don't we always use the largest possible batch?

回答應包含：

1. Batch 越大，KV Cache 消耗通常越高。
    
2. GPU Memory 有容量限制。
    
3. 更高的 Concurrency 可能提高每位使用者的 Inter-token Latency。
    
4. Request Queue 可能累積，造成 TTFT 上升。
    
5. 某些 Kernel 在不同 Batch Size 的效能不會線性增加。
    
6. Production 應尋找符合 SLO 的最高 Sustainable Throughput，而不是只追求最大 Batch。
    

## 6. KV Cache：GPU Memory 與 Decode Efficiency

### 6.1 為什麼需要 KV Cache？

回到 Transformer Self-Attention。

核心計算是：

\[ Q=XW_Q,\quad K=XW_K,\quad V=XW_V \]

以及：

\[ Attention(Q,K,V)= Softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V \]

在 Autoregressive Decoding 中，模型每次產生下一個 Token，都需要參考之前的 Context。

例如：

```
Step 1: The
Step 2: The watch
Step 3: The watch uses
Step 4: The watch uses an
```

如果每一步都重新計算所有之前 Tokens 的 K 和 V，便會產生大量重複運算。

KV Cache 的想法就是：

> 把之前已經計算好的 Key 和 Value 儲存起來。下一步只需要計算新增 Token 的相關表示，再與之前儲存的 K、V 進行 Attention。

注意：KV Cache 不是直接把過去所有 Attention Matrix 儲存起來，而是保存各層需要重用的 K/V Tensors。

### 6.2 KV Cache 如何佔用 GPU Memory？

對典型使用標準 KV Cache 的 Transformer，可以使用以下估算：

\[ \boxed{ M_{KV} = 2 \times L \times H_{kv} \times D_h \times T \times B } \]

其中：

|符號|意義|
|---|---|
|\(2\)|Key 與 Value|
|\(L\)|Transformer Layers 數量|
|\(H_{kv}\)|KV Heads 數量|
|\(D_h\)|每個 Head 的 Dimension|
|\(T\)|已 Cache 的 Tokens 數量|
|\(B\)|每個元素使用的 Bytes|

這是邏輯 Cache Payload 的近似公式，尚未計入 Block Allocation、Metadata、Alignment 等 Runtime Overhead。

### 6.3 實際計算範例

假設模型有：

- 32 Transformer Layers
    
- 8 KV Heads
    
- 每個 Head Dimension = 128
    
- BF16，每個元素 2 Bytes
    
- 每個 Request 保留 4,096 Tokens
    

則：

\[ M_{KV} =2\times32\times8\times128\times4096\times2 \]

\[ \boxed{M_{KV}=512\ \text{MiB / Request}} \]

這代表：一個 Request 在這個假設模型中，4,096 Tokens 就需要約 512 MiB KV Cache。

如果 32 個 Requests 同時佔用相同長度：

\[ 32\times512\ \text{MiB}=16\ \text{GiB} \]

單是 KV Cache Payload 就約 16 GiB。

KV Cache Memory Estimator

Transformer Layers32

KV Heads8

Tokens / Request4,096

Concurrent Requests32

KV Cache Precision

BF16 / FP16

8-bit

Estimated total KV Cache

# 16.00 GiB

Per Request

# 512.0 MiB

假設 Head Dimension = 128，且每個 Request 使用相同 Token 數量；忽略 Cache Metadata、Block Overhead、Sharding 與額外 Runtime Memory。8-bit 選項假設每個 K/V 元素佔 1 Byte。

上面的計算有個關鍵細節：如果模型使用 Tensor Parallelism，KV Cache 通常可以按 KV Heads 或其他方法分散到多張 GPU，但要依模型架構與 Framework 的實際分配方式判斷，不能直接假設永遠平均除以 GPU 數量。

### 6.4 MHA、MQA、GQA 如何影響 KV Cache？

這是 Senior Interview 很常深入追問的地方。

MHA — Multi-Head Attention

假設有 32 個 Query Heads，也有 32 個 KV Heads。

MQA — Multi-Query Attention

多個 Query Heads 共用一組 K/V Heads。

GQA — Grouped-Query Attention

多組 Query Heads 共用較少數量的 K/V Heads。

例如：

|Architecture|Query Heads|KV Heads|KV Cache 相對大小|
|---|---|---|---|
|MHA|32|32|100%|
|GQA|32|8|25%|
|MQA|32|1|3.125%|

以上是其他參數與 Precision 相同時，KV Cache Payload 的相對值，並非所有模型都能直接自由切換架構。

GQA 能顯著降低 KV Cache 的記憶體需求和資料讀取量，也因此非常適合大型 LLM Inference。

### 6.5 PagedAttention 如何管理 KV Cache？

傳統方式可能替每個 Sequence 保留大片連續記憶體空間。

問題是：

- Sequence Length 不固定。
    
- 提前保留的空間可能用不到。
    
- 不同大小的 Memory Allocations 容易造成浪費。
    
- Request 隨時可能結束，造成動態資源管理困難。
    

PagedAttention 使用類似作業系統 Virtual Memory 的 Block-based 方法管理 KV Cache。

Logical KV Blocks

L0

L1

L2

L3

Block Table / Mapping

Physical GPU Cache Blocks

P7

P2

P9

P4

邏輯上連續的 Tokens 不需要對應連續的實體 GPU Cache Blocks。這是概念性示意，不代表特定 Runtime 的實際 Block Table 排列。

當 Request 完成，相關 KV Blocks 可以釋放或根據 Cache Policy 保留供日後重用。

這改善的是 Memory Management Efficiency，而不是讓 Transformer 的 Attention 計算複雜度自動消失。

### 6.6 Prefix Caching

假設公司有 1,000 個使用者，每個人都使用相同的 3,000-token System Prompt。

如果每個 Request 都重新 Prefill 這些相同 Tokens，會產生不必要的重複計算。

Prefix Caching 可以重用相同 Prefix 已經建立的 KV Cache。

例如：

```
Request A:
[Company Policy 3000 tokens] + [Question A]

Request B:
[Company Policy 3000 tokens] + [Question B]
```

如果 Prefix 完全一致，而且符合模型、Tokenization、Cache Isolation 等條件，第二個 Request 可能重用第一個 Request 的 Prefix KV Cache。

但要注意：

- Prefix Cache 主要減少 Prefill 工作。
    
- 它不會直接消除新答案的 Decode。
    
- Dynamic Prompt 順序或不同 Tokenization 可能降低 Cache Hit Rate。
    
- 多租戶企業系統必須正確隔離使用者資料及 Cache 權限。
    

這些限制也在 vLLM 官方 Prefix Caching 文件中有明確區分。

![](https://www.google.com/s2/favicons?domain=https://docs.vllm.ai&sz=32)

vLLM

+1

## 7. Quantization：INT8 / INT4 的效能與品質交換

### 7.1 Quantization 是什麼？

在 Training 或 Inference 中，模型的 Weights 常用 BF16 或 FP16 儲存。

每個 Weight 使用 16 Bits。

但若改用 INT8，只需要 8 Bits；INT4 則只需要 4 Bits。

因此模型權重需要的記憶體就可以降低。

### 7.2 70B Model 的記憶體例子

假設模型有 70 Billion Parameters。

|Weight Precision|理論每 Parameter 大小|純權重記憶體|
|---|---|---|
|FP32|4 Bytes|280 GB|
|BF16 / FP16|2 Bytes|140 GB|
|INT8 / FP8|1 Byte|70 GB|
|INT4 / FP4|0.5 Byte|35 GB|

這是十進位 GB 的理論權重大小，並非完整 GPU VRAM Requirement。實際 Quantized Model 還有 Scales、Zero-points、未量化層、KV Cache、Activations 與 Runtime Buffer。

因此即使 70B 模型的 INT4 權重理論上約 35 GB，也不代表放進一張 40 GB GPU 就一定可以穩定 Serving。

### 7.3 Weight Quantization 與 Activation Quantization

這兩種不能混為一談。

W8A8

代表：

- Weights：8-bit
    
- Activations：8-bit
    

如果硬體具備合適的低精度 Matrix Multiplication，可能同時降低 Memory Traffic 與 Compute Cost。

W4A16

代表：

- Weights：4-bit
    
- Activations：16-bit
    

這種方法主要降低模型權重記憶體需求。

但計算時仍可能需要進行 Dequantization，最後實際速度取決於 Kernel 和硬體。

KV Cache Quantization

這是另一個獨立維度。

即使模型使用 INT4 Weights，KV Cache 仍可能使用 BF16。

所以：

\[ \text{INT4 Weights}\neq\text{INT4 KV Cache} \]

這是面試中很重要的區分。

### 7.4 為什麼 INT4 不一定比 INT8 快？

例如在某些 GPU 上：

1. INT4 權重雖然更小，但需要額外 Unpacking / Dequantization。
    
2. GPU 可能對 FP8 或 INT8 提供更高效率的原生計算路徑。
    
3. 某個 INT4 Kernel 可能在 Batch Size 1 很有效率，但在 Batch Size 64 反而缺乏優勢。
    
4. Quantization Metadata 與 Kernel Layout 也會影響 Performance。
    

NVIDIA 的量化技術文件同樣強調，量化方式的效能及品質交換與 Batch Size、Hardware、Model 等因素有關。

![](https://www.google.com/s2/favicons?domain=https://github.com&sz=32)

GitHub

### 7.5 PTQ vs QAT

PTQ — Post-Training Quantization

模型已完成訓練後再進行 Quantization。

優點是成本相對低、導入方便。

例如 AWQ、GPTQ 常用於 Weight Quantization。

QAT — Quantization-Aware Training

在訓練或調整過程中模擬量化效果，讓模型學會適應低精度。

優點是部分情況可以降低品質損失，但需要額外訓練及驗證成本。

### 7.6 Senior Engineer 如何評估 Quantization？

不能只執行 Benchmark 然後宣布 INT4 成功。

至少應該建立四種比較：

|項目|評估方式|
|---|---|
|Quality|Task Accuracy、Groundedness、Hallucination、Tool Calling、JSON Validity|
|Speed|TTFT、TPOT、Throughput、P95/P99|
|Memory|Weights、KV Cache、Peak VRAM、OOM Rate|
|Cost|GPU-hours、Cost / Request、Cost / Successful Task|

如果企業 Agent 使用 INT4 後，推論成本降低 35%，但工具呼叫錯誤率增加，導致需要更多 Retry，最終成功完成任務的成本不一定更低。

所以 Quantization 的 Production KPI 不應只有 Token Generation Speed。

## 8. Model Parallelism：Tensor、Pipeline、Data Parallelism

### 8.1 為什麼需要 Parallelism？

假設 70B Model 使用 BF16，純 Weights 需要約 140 GB。

如果 GPU 只有 80 GB VRAM，模型就無法完整放入單張 GPU。

這時需要多 GPU 協同運作。

### 8.2 Tensor Parallelism（TP）

將某個 Layer 中的大型 Matrix Operation 分散到多張 GPU。

例如：

\[ Y=XW \]

假設把 \(W\) 分割成四部分：

\[ W=[W_1,W_2,W_3,W_4] \]

四張 GPU 各自計算部分結果：

\[ Y_i=XW_i \]

之後透過必要的 Collective Communications，例如 All-reduce 或 All-gather，組合後續計算所需要的結果。

Transformer Layer

GPU 0

Shard 1

GPU 1

Shard 2

GPU 2

Shard 3

GPU 3

Shard 4

Collective Communication

Next Layer Computation

優點：

- 多張 GPU 可以共同承載大型模型。
    
- 在適合的情況下可以降低單一 GPU 的計算或 Memory Pressure。
    

缺點：

- GPU 之間需要頻繁通訊。
    
- 通訊成本可能成為 Bottleneck。
    
- Interconnect Bandwidth 與 Latency 很重要。
    

因此 TP 並不是 GPU 越多就一定越快。

NVLink、NVSwitch、PCIe 等互連方式可能產生明顯差異。

### 8.3 Pipeline Parallelism（PP）

不是把一個 Layer 的 Matrix 分割，而是把不同的 Layers 放到不同 GPU。

例如有 80 個 Transformer Layers：

|GPU|負責 Layers|
|---|---|
|GPU 0|Layer 1–20|
|GPU 1|Layer 21–40|
|GPU 2|Layer 41–60|
|GPU 3|Layer 61–80|

資料依序經過 GPU 0 → GPU 1 → GPU 2 → GPU 3。

如果可以安排多個 Microbatches 或 Requests 交錯工作，就能提高硬體使用率。

但 Pipeline 會有 Bubble：某些 GPU 可能正在等待上游階段完成，因此 Pipeline Depth 和 Stage Balance 會直接影響效率。

PP 不一定能降低單一 Request 的 End-to-end Latency，甚至可能增加跨 GPU 傳輸與同步的延遲。

### 8.4 Data Parallelism（DP）

Data Parallelism 是讓不同 GPU 或 GPU Groups 擁有完整的模型 Replica，各自處理不同 Requests。

例如：

```
           Load Balancer
                |
      +---------+---------+
      |         |         |
   Replica A Replica B Replica C
      |         |         |
  Request 1 Request 2 Request 3
```

DP 特別適合提高 Serving Throughput，前提是單個 Replica 能容納完整的模型（或者每個 Replica 本身使用 TP / PP）。

### 8.5 三種 Parallelism 比較

|類型|分割什麼|主要目的|主要代價|
|---|---|---|---|
|TP|Layer 內 Tensor|大模型運算及記憶體分攤|頻繁 Collective Communication|
|PP|Transformer Layers|跨 GPU 放置模型|Pipeline Bubble、跨 Stage 傳輸|
|DP|Requests / Model Replicas|增加總 Serving Capacity|重複模型權重、較多 GPU|

### 8.6 混合式平行化

例如：

\[ TP=4,\quad PP=2,\quad DP=3 \]

理論總 GPU 數量：

\[ 4\times2\times3=24 \]

這代表：

- 每個模型 Replica 使用 8 張 GPU。
    
- 每個 Replica 內使用 TP=4、PP=2。
    
- 一共有 3 個平行 Serving Replicas。
    

實務還需要考量：

- 哪些 GPUs 在同一 Node。
    
- 哪些 GPUs 透過 NVLink 連線。
    
- Cross-node Network Bandwidth。
    
- NCCL Collective Performance。
    
- 每個 Replica 的 KV Cache Budget。
    
- Failure Domain 與 Replica Placement。
    

vLLM 官方已提供 TP、PP 及 Multi-node Distributed Serving 配置方式。

![](https://www.google.com/s2/favicons?domain=https://docs.vllm.ai&sz=32)

vLLM

對於 Mixture-of-Experts 模型，還可能需要 Expert Parallelism（EP），將不同 Experts 分散部署，並處理 Token Routing 與 All-to-all Communication。

## Part 3 — Latency、Throughput 與 GPU Performance Engineering

## 9. Latency：TTFT、TPOT、P95 如何衡量？

Senior Engineer 必須能分辨不同種類的 Latency，而不是只說「這個模型回答需要 5 秒」。

### 9.1 三個最重要的 Latency Metrics

TTFT — Time to First Token

從使用者發出 Request，到收到第一個輸出 Token 的時間。

\[ TTFT=t_{\text{first token}}-t_{\text{request start}} \]

TTFT 可能包含：

- Network Latency
    
- Request Queue Waiting
    
- Tokenization
    
- Prefill Computation
    
- First Decode Step
    
- API / Streaming Overhead
    

如果是完整 Agent Application，使用者感受到的 TTFT 可能還包含 RAG Retrieval、Tools Execution 等前置步驟。

因此要明確區分 Model TTFT 和 Application TTFT。

TPOT — Time per Output Token

常見定義是：

\[ TPOT=\frac{T_{\text{end}}-TTFT}{N_{\text{output}}-1} \]

其中：

- \(T_{\text{end}}\)：相對 Request Start 的最終 Token 到達時間。
    
- \(N_{\text{output}}\)：輸出 Token 數量。
    

它表示第一個 Token 出現之後，平均每個額外 Token 需要多少時間。

ITL — Inter-Token Latency

相鄰兩次 Token Output Event 之間的時間。

當 Streaming 一次回傳多個 Tokens，或使用 Speculative Decoding 時，ITL 與 TPOT 可能不同。

vLLM 官方 Metrics 文件也特別區分這些指標的統計方式。

![](https://www.google.com/s2/favicons?domain=https://docs.vllm.ai&sz=32)

vLLM

### 9.2 實際計算例子

假設一個 LLM Request：

- TTFT = 800 ms
    
- TPOT = 35 ms
    
- Output Tokens = 200
    

則 End-to-end Generation Latency 約為：

\[ T_{\text{total}} \approx TTFT+(N-1)\times TPOT \]

\[ =0.8+199\times0.035 \]

\[ \boxed{T_{\text{total}}\approx7.765\text{ seconds}} \]

這代表即使 TTFT 小於 1 秒，使用者仍可能需要將近 8 秒才能收到完整回答。

因此：

- 聊天介面特別重視 TTFT 與 Streaming Smoothness。
    
- 長文生成更加重視 TPOT 和總完成時間。
    
- 非同步大量資料處理通常更重視 Throughput。
    

### 9.3 P50、P95、P99 是什麼？

假設一個系統有 1,000 次 Requests。

|指標|意義|
|---|---|
|P50|約 50% Requests 的 Latency 不超過此值|
|P95|約 95% Requests 的 Latency 不超過此值|
|P99|約 99% Requests 的 Latency 不超過此值|

例如：

|Metric|P50|P95|P99|
|---|---|---|---|
|TTFT|0.5 s|1.8 s|4.2 s|
|TPOT|25 ms|55 ms|90 ms|
|E2E Latency|5 s|13 s|21 s|

假設的測試結果，不代表特定模型或硬體的實際 Benchmark。

即使 P50 很好，P99 仍可能讓部分使用者覺得服務很慢。

Senior Engineer 應特別調查 Tail Latency。

例如：

- 長 Prompt 導致 Prefill 負載增加。
    
- GPU 記憶體不足，發生 Request Preemption。
    
- Request Queue 積壓。
    
- 網路傳輸抖動。
    
- 某個 Replica 處理過多 Requests。
    
- Cold Start 或模型重新載入。
    

另一個重要細節：P95 TTFT 加上 P95 TPOT 並不等於 P95 E2E Latency，因為 Percentiles 不能像平均值一樣直接相加。

### 9.4 Senior 應了解的 Latency Breakdown

```
User Request
    |
    v
Network / API Gateway
    |
    v
Authentication / Validation
    |
    v
RAG / Context Construction
    |
    v
LLM Queue Waiting
    |
    v
Prefill
    |
    v
First Token
    |
    v
Decode / Streaming
    |
    v
Postprocessing / Final Response
```

每一段都應有獨立 Tracing 與 Metrics。

否則 Production 發現 TTFT 上升時，很難知道是 Retrieval、GPU Scheduler、網路還是模型本身變慢。

## 10. Throughput：Tokens/sec、Concurrent Users

### 10.1 Throughput 有哪些定義？

至少要區分：

Request Throughput

\[ \text{Requests/sec} = \frac{\text{Completed Requests}}{\text{Elapsed Time}} \]

Output Token Throughput

\[ \text{Output Tokens/sec} = \frac{\text{Generated Tokens}}{\text{Elapsed Time}} \]

Total Token Throughput

\[ \text{Total Tokens/sec} = \frac{\text{Input Tokens + Output Tokens}}{\text{Elapsed Time}} \]

三種 Metrics 不能互相取代。

例如，模型每秒可以處理很多 Input Tokens，不代表它也可以用同樣速度生成 Output Tokens。

### 10.2 為什麼要同時測量 Input 與 Output？

假設兩種 Workload：

||Workload A|Workload B|
|---|---|---|
|Input Tokens|500|16,000|
|Output Tokens|1,000|100|
|主要負載|Decode-heavy|Prefill-heavy|
|主要風險|長時間佔用 Decode Capacity|長 Context Prefill 與 KV Memory|

即使兩個 Requests 的總 Token 數量接近，也不能視為相同 GPU Cost。

### 10.3 Concurrent Users 不等於 Requests/sec

假設系統每秒收到 10 個 Requests，每個 Request 平均持續 8 秒。

根據 Little's Law：

\[ L=\lambda W \]

其中：

- \(L\)：平均同時存在的 Requests。
    
- \(\lambda\)：平均 Request Arrival Rate。
    
- \(W\)：平均 Request Response Time。
    

因此：

\[ L=10\times8=80 \]

代表穩態下平均有約 80 個 In-flight Requests。

但這不是說系統只能支援 80 個登入使用者。可能有數千人在線上，卻只有 80 個 Requests 正在等待或執行。

### 10.4 Throughput 與 Latency 的 Tradeoff

Latency–Throughput Tradeoff（示意）

虛構測試資料；隨負載接近系統容量，吞吐量可能趨於飽和，而 P95 延遲迅速增加。

Completed Requests/secP95 Latency（秒）

04812162468101214

兩條曲線使用同一視覺座標軸，但單位不同，僅用來示意 Saturation 趨勢；實際容量分析應分開繪製 Requests/sec 與秒數。

在低負載時，增加 Concurrency 通常有助於提高 GPU Utilization。

但當 GPU 接近 Saturation：

1. 新 Requests 進入 Waiting Queue。
    
2. Queue Waiting Time 增加。
    
3. KV Cache Pressure 上升。
    
4. 每個 Request 可能取得較少計算資源。
    
5. P95 和 P99 Latency 開始顯著上升。
    

因此 Production Capacity 不應定義成：

> GPU 最多可以跑出多少 Tokens/sec？

而應定義成：

> 在符合 P95 TTFT、TPOT、Error Rate 及 Quality SLO 的條件下，系統最高可以穩定處理多少 Requests/sec？

### 10.5 Compute-bound vs Memory-bound

這是 Senior Inference Engineer 需要理解的重要 GPU 概念。

Compute-bound

GPU Arithmetic Units 的計算能力限制了效能。

常見於大型 Prefill Matrix Operations。

Memory-bandwidth-bound

GPU 花大量時間從 HBM 讀取 Model Weights、KV Cache 或其他資料。

許多 Decode Workloads 容易受 Memory Bandwidth 限制，尤其是低 Batch Size 時。

可以利用 Roofline Model 理解：

\[ Performance\leq \min( PeakCompute,\, MemoryBandwidth\times ArithmeticIntensity ) \]

其中：

\[ ArithmeticIntensity=\frac{FLOPs}{BytesTransferred} \]

當 Arithmetic Intensity 低時，即使 GPU 有很高的理論 TFLOPS，也不代表可以完全利用。

因此以下做法的效果必須依瓶頸判斷：

- 如果是 Memory-bound：Quantization、Batching、KV Cache Optimization 可能很有幫助。
    
- 如果是 Compute-bound：優化 Kernels、Tensor Cores、Model Architecture、Parallelism 可能更重要。
    
- 如果是 Queue-bound：增加有效 Serving Capacity 或控制 Admission Rate 才是重點。
    

## Part 4 — Cloud Deployment、Reliability、Cost Optimization

## 11. Cloud Deployment：Scaling、Monitoring、Rollback

### 11.1 一個 Production LLM Serving System 應該長什麼樣？

以下是一個部署在 AWS 的示例架構，也可以對應到 GCP 或 Azure。

```
                     Users / Applications
                              |
                              v
                     API Gateway / WAF
                              |
                              v
                 Authentication / Rate Limit
                              |
                              v
                       LLM API Service
                              |
                +-------------+-------------+
                |                           |
                v                           v
           RAG Service                Model Router
                |                           |
         Vector DB / Search       +---------+---------+
                                  |                   |
                                  v                   v
                           Small Model Pool    Large Model Pool
                             vLLM / TRT-LLM      vLLM / TRT-LLM
                                  |                   |
                                  v                   v
                              GPU Replicas        GPU Replicas
                                  |                   |
                                  +---------+---------+
                                            |
                                            v
                                    Stream / Response

      Observability:
      Prometheus / Grafana / Tracing / GPU Metrics

      Deployment:
      ECR / Kubernetes / EKS / Autoscaling / Canary

      Artifacts:
      S3 / Model Registry / Versioned Configuration
```

在這個系統中，API Gateway、RAG、Inference Engine、GPU Cluster、Observability 各自有不同的責任。

Senior Engineer 應避免把所有功能寫進同一個單體 Python Server，尤其當模型推論、工具呼叫與背景工作負載的 Scaling Characteristics 不同時。

### 11.2 Autoscaling 應根據什麼決定？

一般 Web Server 常根據 CPU Utilization Scaling。

但 LLM Serving 的 CPU 使用率不一定能反映真正瓶頸。

可以觀察：

|Metric|作用|
|---|---|
|Request Queue Depth|是否有大量 Requests 等待|
|Queue Waiting Time|是否開始違反 TTFT SLO|
|KV Cache Utilization|是否接近 Memory Capacity|
|Running Requests|同時處理的 Requests|
|Output Tokens/sec|GPU 實際生成吞吐量|
|GPU Utilization|GPU 是否繁忙|
|HBM Bandwidth|是否受到 Memory Bandwidth 限制|
|P95 TTFT / TPOT|使用者效能是否退化|
|Request Rejection Rate|是否已超出 Capacity|

Kubernetes HPA 可以根據 Custom Metrics 進行 Scaling，但實際整合時需要 Metrics Adapter 或其他適當的 Metrics Pipeline。

![](https://www.google.com/s2/favicons?domain=https://kubernetes.io&sz=32)

Kubernetes

### 11.3 為什麼 GPU Autoscaling 比 Web Autoscaling 困難？

因為新增一個 GPU Model Replica 可能需要：

1. 配置 GPU Node。
    
2. Pull Container Image。
    
3. 下載或掛載 Model Weights。
    
4. 初始化 Runtime。
    
5. 配置 GPU Memory / KV Cache。
    
6. 執行 Warm-up。
    
7. 通過 Readiness Check。
    
8. 才能開始承接 Requests。
    

這通常比啟動一般 Web API Container 複雜。

因此生產系統可能需要：

- Warm Replicas
    
- Min Replica Count
    
- Predictive Scaling
    
- Queue-based Admission Control
    
- Load Shedding
    
- Graceful Drain
    

特別注意：不應該讓尚未完成模型載入的 GPU Pod 通過 Readiness Probe。

### 11.4 Monitoring 與 Alerting

建議至少有四層：

Application Layer

- API Request Count
    
- HTTP 4xx / 5xx
    
- Authentication Failure
    
- Per-tenant Rate Limit
    
- Task Success Rate
    

Inference Layer

- TTFT / TPOT / ITL
    
- Prefill / Decode Time
    
- Queue Duration
    
- Running / Waiting Requests
    
- KV Cache Utilization
    
- Cache Hit Rate
    

GPU Infrastructure Layer

- GPU Utilization
    
- GPU Memory Usage
    
- Memory Bandwidth
    
- GPU Temperature
    
- GPU Errors
    
- Node Health
    

Business / Quality Layer

- Correct Answer Rate
    
- Hallucination Rate
    
- Tool Call Success
    
- Escalation Rate
    
- Cost per Successful Task
    

vLLM 提供 `/metrics` Endpoint，可整合 Prometheus 的 Metrics Collection。

![](https://www.google.com/s2/favicons?domain=https://github.com&sz=32)

GitHub

這裡尤其需要避免一個常見錯誤：只監控 GPU Utilization，沒有監控模型服務品質。

### 11.5 Canary Deployment 與 Rollback

假設公司現在使用 Model V1，希望升級到 V2。

安全做法：

```
Production Traffic
        |
        v
   Load Balancer
        |
   +----+--------------------+
   |                         |
   v                         v
Model V1                  Model V2
95% Traffic               5% Canary
   |                         |
   +------------+------------+
                |
                v
      Monitoring / Evaluation
                |
                v
      Promote or Rollback
```

可以使用 5% → 10% → 25% → 100% 的漸進流量策略，但實際比例需要依測試流量、風險和統計檢定能力決定。

每個階段應比較：

- Task Success
    
- P95 / P99 Latency
    
- GPU Cost
    
- Error Rate
    
- OOM
    
- JSON Schema Validity
    
- Security Behavior
    
- Quality Regression
    

如果 V2 表現不佳，應能立即停止新 Request Routing，將流量切回 V1，並讓既有 Streaming Requests 正常 Drain 或依故障情況安全終止。

Rollback 也應包含 Model Weights、Tokenizer、Prompt Template、Quantization Configuration 和 Runtime Version，而不只是切換 Docker Image Tag。

## 12. Reliability：Rate Limits、Retries、Fallback、Circuit Breakers

可靠性不是指服務永遠不出錯，而是當故障發生時能快速偵測、限制影響、恢復服務。

### 12.1 Rate Limits

可以使用以下條件限制請求：

- Requests per minute
    
- Input Tokens per minute
    
- Output Tokens per minute
    
- Concurrent Requests
    
- 每個 Tenant 的 GPU Budget
    

為什麼只限制 Requests/sec 不夠？

因為一個 50,000-token Prompt 可能比一個 500-token Prompt 需要更多 Prefill 計算及 KV Cache。

因此可以建立 Token-aware Admission Control。

例如：

```
def estimate_request_tokens(prompt_tokens, max_output_tokens):    return prompt_tokens + max_output_tokensdef admit_request(    prompt_tokens,    max_output_tokens,    token_budget_remaining):    estimated = estimate_request_tokens(        prompt_tokens,        max_output_tokens    )    return estimated <= token_budget_remaining
```

這是簡化示意。正式系統還需分別考慮 Prefill Compute Budget、Decode Capacity、KV Cache Reservation、Tenant Quota 與公平排程。

### 12.2 Retries 如何設計？

假設 Inference Request 收到 HTTP 503。

可以使用：

- Exponential Backoff
    
- Random Jitter
    
- Retry Limit
    
- Overall Deadline
    

例如：

```
import randomdef retry_delay(attempt):    base = min(0.5 * (2 ** attempt), 8.0)    return random.uniform(0, base)
```

這是 Full Jitter Backoff 的簡化例子。

但不能對所有失敗都無條件 Retry。

|Failure|建議處理|
|---|---|
|HTTP 429|遵守 Retry-After、限次 Retry 或回傳限流|
|HTTP 503|嘗試健康 Replica，並限制 Retry 次數|
|Invalid Request / 400|修正請求，不應盲目 Retry|
|GPU OOM|隔離不健康 Replica、調整 Capacity，不應反覆重試同一路徑|
|Timeout|檢查是否可安全重試，遵守 Deadline|
|Streaming 中斷|不應直接重新送出並假裝能無縫接續|

重點是 Retry 本身會增加系統負載。

如果 GPU 已經超載，大量 Clients 同時 Retry 可能產生 Retry Storm，讓整個系統更難恢復。

### 12.3 Circuit Breaker

Circuit Breaker 有三個主要狀態：

Closed

正常送出

Open

快速拒絕

Half-open

少量探測

探測成功後可恢復 Closed；失敗則重新 Open。

例如：

- 某個 GPU Replica 連續多次 OOM。
    
- Circuit Breaker 暫時停止導入新的 Requests。
    
- 系統將請求轉給其他健康 Replica。
    
- 經過一段時間後只送少量探測 Requests。
    
- 確認恢復正常後才重新增加流量。
    

### 12.4 Fallback Model

例如主要使用 70B Model。

如果該 Model 暫時無法服務，可以考慮 Fallback 到 8B Model。

但這必須考量任務風險。

|Task|是否適合 Smaller Model Fallback|
|---|---|
|FAQ Summary|通常可以評估採用|
|一般文字分類|可以，需驗證準確率|
|複雜 Legal Analysis|未必適合|
|高風險 Authentication Decision|不應未經驗證直接降低模型能力|
|Structured Data Extraction|取決於 Schema Compliance 和 Error Rate|

Production Reliability 不代表一定要產生答案，有時應該明確拒絕、要求人工處理或暫時降級服務。

## 13. Cost Optimization：Model Routing、Caching、Token Budget

### 13.1 Cost Optimization 的核心問題

使用 API 時，常見成本基礎是 Input / Output Tokens 與服務價格。

Self-hosted LLM 則需要考慮：

- GPU Instance Cost
    
- CPU、Memory、Storage
    
- Network
    
- Model Replicas
    
- Idle Capacity
    
- Observability
    
- Operations / Maintenance
    

一個簡化的單位成本：

\[ Cost/Request= \frac{TotalInfrastructureCost}{SuccessfulRequests} \]

更有價值的指標是：

\[ \boxed{ CostPerSuccessfulTask= \frac{\text{Total Serving + Retry + Tool Cost}} {\text{Successfully Completed Tasks}} } \]

因為便宜的模型如果經常需要 Retry，最終不一定划算。

### 13.2 Model Routing

假設企業使用兩個模型：

- Small Model：8B
    
- Large Model：70B
    

Router 依任務難度、風險與品質需求選擇模型。

```
                Incoming Request
                        |
                        v
               Task Classification
                        |
              +---------+---------+
              |                   |
              v                   v
        Simple / Low Risk    Complex / High Risk
              |                   |
              v                   v
          8B Model            70B Model
              |                   |
              +---------+---------+
                        |
                        v
                 Quality Check
                        |
                        v
                     Response
```

例如：

- 80% 簡單任務由 8B 處理。
    
- 20% 複雜任務使用 70B。
    

這些比例必須透過實際 Task Dataset、Offline Evaluation、Shadow Testing 和 Production Feedback 驗證。

不應單純因為 8B 比較便宜，就把所有任務都交給 8B。

### 13.3 Caching 的三種不同層次

|Cache|儲存內容|主要效益|
|---|---|---|
|Prefix KV Cache|已計算的 K/V States|減少重複 Prefill|
|Exact Response Cache|完全相同請求的已驗證結果|避免重新生成|
|Semantic Cache|語意相似 Query 的可重用結果|在適合的場景減少 LLM Calls|

但 Semantic Caching 特別需要注意：

- 使用者與 Tenant 的 Access Control。
    
- 文件版本是否已更新。
    
- 答案是否含有使用者特定資料。
    
- 不同問題語意相近卻不完全相同。
    
- 是否需要重新檢索即時資訊。
    

不能為了省錢而向使用者返回錯誤或過期的答案。

### 13.4 Token Budget Management

假設企業 RAG 系統每次都加入：

- 5,000 Tokens System Prompt
    
- 20,000 Tokens Retrieved Documents
    
- 5,000 Tokens Conversation History
    

總共 30,000 Input Tokens。

但經過 Retrieval Quality Evaluation，可能發現只需要：

- 1,000 Tokens System Prompt
    
- 6,000 Tokens Relevant Documents
    
- 2,000 Tokens Necessary History
    

總共 9,000 Input Tokens。

這可能減少 Prefill Cost、TTFT 與 KV Cache Pressure。

但不應只為減少 Tokens 而刪除重要證據。

Senior Engineer 應衡量：

\[ Quality=f(\text{Model},\text{Relevant Context},\text{Token Budget}) \]

目標是保留足夠的相關證據，而不是盲目把 Context 縮到最短。

## Part 5 — 實際 Production Case Study：企業手錶鑑定 AI Assistant

下面使用一個具體系統，將前面的技術串起來。

假設公司有多相機手錶影像檢測系統，CV Pipeline 已完成 Segmentation、Feature Extraction、Anomaly Detection 與 Authentication Evidence Generation。

現在希望增加 LLM Assistant，可以讀取分析結果、檢索技術資料，產生人類可閱讀的鑑定報告，並回答客戶問題。

LLM 主要負責 解釋與整合證據，而不是無條件取代既有的檢測、統計驗證與鑑定決策流程。

### 14.1 Step 1 — 定義系統 Requirements

假設有以下業務需求：

|Requirement|假設目標|
|---|---|
|Daily Requests|100,000|
|Peak Request Rate|10 Requests/sec|
|Average Input|2,000 Tokens|
|Average Output|250 Tokens|
|P95 TTFT|小於 2 秒|
|P95 TPOT|小於 50 ms|
|Availability|99.9%|
|Deployment|AWS|
|Models|8B + 70B|
|Serving|vLLM / TensorRT-LLM|

這是示範性 Requirements，不是實際系統測得的數據。

### 14.2 Step 2 — Capacity Estimation

Peak Output Token Demand：

\[ 10\text{ req/s}\times250\text{ tokens/req} \]

\[ \boxed{2500\text{ output tokens/sec}} \]

Peak Input Token Demand：

\[ 10\times2000=20000 \]

即每秒約需處理 20,000 Input Tokens。

這兩種負載要分別估算，不能只看總 Token 數量。

### 14.3 Step 3 — 初步 GPU Capacity Planning

假設經過實際 Benchmark，我們得到以下示範結果：

某個 70B Model Replica 使用 TP=4、4 張 80 GB GPUs，在符合目標 Latency SLO 且輸入長度分布相似的情況下，可提供 900 Output Tokens/sec。

注意：900 Tokens/sec 是假設的 Benchmark 結果，不是對任何特定 GPU 的效能保證。

為了保留容量緩衝，假設長期目標負載為測得容量的 70%。

則：

\[ Capacity_{\text{replica}}=900\times0.7=630 \]

所需 Replicas：

\[ N=\left\lceil\frac{2500}{630}\right\rceil=4 \]

需要：

\[ 4\text{ replicas}\times4\text{ GPUs}=16\text{ GPUs} \]

如果希望保留一個額外 Replica 作為 N+1 Buffer，可以從 5 個 Replicas、20 張 GPU 的方案開始評估。

但以上還未證明 Prefill Capacity、KV Cache、尾端延遲與突發流量一定符合 SLO。

正式容量規劃必須用混合 Prefill/Decode 的壓力測試確認，而不是只用 Output Throughput 除法。

### 14.4 Step 4 — 決定 Serving Architecture

初期可以選擇：

```
App / Client
    |
    v
API Gateway
    |
    v
Authentication + Tenant Quotas
    |
    v
Application Backend
    |
    +----> Existing CV / Authentication Results
    |
    +----> RAG Retrieval
    |
    v
Model Router
    |
    +----> Small LLM Pool (8B)
    |
    +----> Large LLM Pool (70B)
                  |
                  v
         vLLM Continuous Batching
                  |
                  v
         Tensor Parallel GPUs
                  |
                  v
           Streaming Response
```

如果 8B 已足以正確生成大部分簡單報告，將其分流至 Small Model Pool，可以降低 70B Pool 的負載。

但高風險鑑定結論應依驗證通過的決策流程處理，不應因模型可用性或成本而自動改用未經核准的 Fallback。

### 14.5 Step 5 — 實際部署 vLLM

以下示範在支援的 Linux / NVIDIA CUDA 環境中啟動一個 7B 模型 API Server。

示範指令：參數需依 vLLM 版本、GPU 記憶體及模型相容性確認。

```
pip install vllm

vllm serve Qwen/Qwen2.5-7B-Instruct \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90
```

參數說明：

|Parameter|用途|
|---|---|
|`--tensor-parallel-size 2`|使用兩張 GPU 進行 TP|
|`--dtype bfloat16`|使用 BF16 模型精度|
|`--max-model-len 8192`|最大模型 Context Length|
|`--max-num-seqs 64`|單次 Scheduler Iteration 的 Sequence 上限|
|`--max-num-batched-tokens 8192`|一次 Iteration 的 Token Scheduling Budget|
|`--gpu-memory-utilization 0.90`|Engine 可用 GPU Memory Budget 的比例|

這些參數可以在 vLLM 官方 Serving CLI 文件中查閱。

![](https://www.google.com/s2/favicons?domain=https://docs.vllm.ai&sz=32)

vLLM

`--gpu-memory-utilization` 不是將 GPU 運算使用率固定在 90%，而是限制 Engine 可使用的 GPU 記憶體比例。

接著使用 API 測試：

```
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [
      {
        "role": "user",
        "content": "Explain how an automatic watch works."
      }
    ],
    "max_tokens": 200
  }'
```

Production 時不應直接公開未受保護的模型 Endpoint；應透過適當的網路隔離、Authentication、Authorization 和 Gateway Controls 提供服務。

### 14.6 Step 6 — Benchmark Serving Performance

接下來使用 vLLM Benchmark 工具測試。

```
vllm bench serve \
  --backend vllm \
  --model Qwen/Qwen2.5-7B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --dataset-name random \
  --random-input-len 2048 \
  --random-output-len 256 \
  --request-rate 4 \
  --max-concurrency 32 \
  --num-prompts 500 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --save-result
```

這會模擬一定數量的 Requests，並收集 TTFT、TPOT、ITL 與 E2E Latency 等資訊。

相關參數和更進階的 Load Patterns 可參考 vLLM 官方 Benchmark CLI。

![](https://www.google.com/s2/favicons?domain=https://docs.vllm.ai&sz=32)

vLLM

注意：這裡使用 Random Dataset 方便初步檢測 Serving 能力；正式 Benchmark 應再使用真實 Traffic Replay 或具有代表性的 Prompt Length / Output Length Distribution。

### 14.7 Step 7 — 建立 Benchmark Matrix

Senior Engineer 不應只跑一次測試，而是設計實驗矩陣。

|Test|Model Precision|Concurrency|Input / Output|
|---|---|---|---|
|A|BF16|1|2K / 256|
|B|BF16|16|2K / 256|
|C|BF16|64|2K / 256|
|D|INT8|16|2K / 256|
|E|INT4|16|2K / 256|
|F|BF16|16|16K / 256|
|G|BF16|16|2K / 2K|

除了這些參數，還要固定或記錄：

- GPU Hardware
    
- Model Revision
    
- vLLM / TensorRT-LLM Version
    
- CUDA / Driver Version
    
- Tensor Parallel Configuration
    
- KV Cache Precision
    
- Prompt Caching State
    
- Warm-up Conditions
    
- Request Arrival Pattern
    

最後建立比較：

不同設定的 P95 TTFT 比較（示意數據）

數字完全為教學假設，不是 vLLM、TensorRT-LLM 或任何 GPU 的實測結果。

0 s0.6 s1.2 s1.8 s2.4 sBF16 / C1BF16 / C16BF16 / C64INT8 / C16INT4 / C16Long Context

如果結果顯示：

- BF16 / C16 的 P95 TTFT = 0.8s
    
- BF16 / C64 的 P95 TTFT = 2.3s
    

則 C64 雖然可能有較高 Throughput，卻已違反前面設定的 P95 TTFT 小於 2 秒的 SLO。

這就是為什麼 Production Optimization 不能只比較 Tokens/sec。

### 14.8 Step 8 — Deployment、Monitoring、Rollback

將經過驗證的 Serving Image 部署到 AWS EKS 或其他 GPU Infrastructure，並建立以下流程：

```
Code / Model Configuration
          |
          v
       CI Pipeline
          |
          v
   Security + Unit Tests
          |
          v
   Build Container Image
          |
          v
      Push to ECR
          |
          v
  Deploy to Staging GPU
          |
          v
   Functional + Load Tests
          |
          v
      Canary Release
          |
          v
     Production Metrics
          |
          v
    Promote / Rollback
```

每個模型版本應有可追蹤的：

- Model Artifact Hash
    
- Tokenizer Revision
    
- Serving Engine Version
    
- Quantization Method
    
- Deployment Manifest
    
- Evaluation Dataset Version
    
- Benchmark Results
    
- Approval / Rollback Record
    

### 14.9 Step 9 — 成本估算

假設前面估算的 5 個 Replicas，需要 20 張 GPU。

為方便教學，假設每張 GPU 的全部計費成本為每小時 US$3。這只是計算假設，不是任何 AWS Instance 的即時報價。

30 天持續運作：

\[ 20\times3\times24\times30 \]

\[ \boxed{US\$43,200/month} \]

假設每天有 100,000 Requests，30 天共 3,000,000 Requests。

則單純 GPU 成本：

\[ \frac{43200}{3000000} = \boxed{US\$0.0144/Request} \]

也就是每次 Request 的平均 GPU Capacity Cost 約 1.44 美分。

但尚未計入其他 Infrastructure、RAG、失敗重試、閒置效率差異，以及應用開發與維運成本。

如果採用 Model Routing，讓適合的簡單任務交由小模型處理，整體成本可能進一步下降。

此時要比較：

\[ \text{Cost Savings} \quad\text{vs}\quad \text{Quality Regression} \]

而不是只看 GPU Bill。

## Part 6 — 進階 Inference 技術：Senior / Staff Engineer 應該再了解什麼？

除了使用者列出的十項技術，2026 年還有幾項值得深入研究。

### 15. Speculative Decoding

標準 Autoregressive Decoding 通常一次產生一個 Token。

Speculative Decoding 讓一個較小的 Draft Model 先提出多個候選 Tokens，再由較大的 Target Model 平行驗證。

例如：

```
Draft Model proposes:
Token A -> B -> C -> D

Target Model verifies candidates
           |
           v
Accept valid prefix
           |
           v
Continue generation
```

如果 Draft Model 的預測經常被接受，可能降低生成多個 Tokens 的總時間。

但需注意：

- Draft Model 本身也有運算成本。
    
- Acceptance Rate 直接影響效能。
    
- 高 Concurrency 或某些 Hardware Configuration 未必有利。
    
- 使用正確的驗證與採樣演算法時，可以保持 Target Model 的目標輸出分布，而不只是近似生成。
    

這項技術適合進一步研究 Decode Latency Optimization。

### 16. Disaggregated Prefill / Decode

由於 Prefill 與 Decode 使用 GPU 的方式不同，可以考慮分開部署：

```
Incoming Requests
        |
        v
   Prefill GPU Pool
        |
        v
  KV Cache Transfer
        |
        v
    Decode GPU Pool
        |
        v
 Streaming Response
```

好處：

- 可以獨立調整 Prefill 和 Decode Capacity。
    
- 長 Prompt 的 Prefill 比較不容易直接干擾 Decode。
    
- 有機會更精準控制 TTFT 與 Tail ITL。
    

缺點：

- KV Cache Transfer 需要額外 Bandwidth。
    
- 必須處理跨 GPU / Node 的資料移動。
    
- Scheduler 與 Fault Recovery 更複雜。
    
- 不一定適合小型 Deployment。
    

vLLM 官方文件把這項功能列為持續演進的進階部署能力；實際使用時必須驗證所選版本與 KV Transfer Connector 的相容性。

![](https://www.google.com/s2/favicons?domain=https://github.com&sz=32)

GitHub

### 17. Hardware-aware Inference Optimization

Senior Inference Engineer 還應理解：

|主題|重點|
|---|---|
|CUDA Graphs|減少部分 CPU Launch Overhead|
|Kernel Fusion|合併運算，降低中間資料搬移|
|FlashAttention|改善 Attention 的 IO Efficiency|
|Memory Coalescing|提高 GPU Memory Access Efficiency|
|NCCL|多 GPU Collective Communication|
|HBM Bandwidth|Decode 常見瓶頸之一|
|NUMA / PCIe / NVLink|GPU 與 Host / GPU 間資料交換|
|CUDA Profiling|找出 Kernel、Memory、Communication 瓶頸|

若面試的是 Senior Applied AI Engineer，通常不一定要求自己撰寫全部 CUDA Kernels。

但如果是 Senior LLM Inference Engineer 或 Inference Infrastructure Engineer，就可能需要深入理解 GPU Profiling、Kernel Selection、Memory Layout，甚至自行優化 CUDA 或 Triton Kernel。

## Part 7 — Senior Engineer 面試：五個深入追問

### Q1. 為什麼 GPU Utilization 很高，但 Output Tokens/sec 很低？

應檢查：

1. GPU 是 Compute-bound 還是 Memory-bound。
    
2. Prefill 是否佔用了大部分計算時間。
    
3. Batch Size 是否太低。
    
4. KV Cache 是否造成 Memory Pressure。
    
5. Tensor Parallel Communication 是否太昂貴。
    
6. 是否存在頻繁的 Kernel Launch 或同步成本。
    

進一步使用 NVIDIA Nsight Systems、Nsight Compute、PyTorch Profiler 或 Runtime 提供的 Profiling 能力定位 Bottleneck。

### Q2. P95 TTFT 突然從 0.8 秒變成 4 秒，但 TPOT 沒有明顯增加，代表什麼？

很可能與 First-token 之前的階段有關，例如：

- Queue Waiting
    
- Prefill Overload
    
- Long Context Requests
    
- Retrieval Delay
    
- Request Admission
    
- Cold Replica
    

應依 Trace 分析 Queue、Prefill 與 Application 前處理，而不是直接認定 Decode 速度下降。

### Q3. 模型使用 INT4 後，為什麼還會 GPU OOM？

因為 INT4 主要降低 Weight Memory。

仍然有：

- KV Cache
    
- Activations
    
- Temporary Tensors
    
- CUDA Graph Memory
    
- Workspace Buffers
    
- Fragmentation / Allocation Overhead
    

例如模型權重從 16 GB 降到 4 GB，不代表 64 個長 Context Requests 的 KV Cache 也減少四倍。

### Q4. 如果 GPU 不夠，應優先增加 Tensor Parallelism 還是 Data Parallelism？

取決於瓶頸。

如果單一模型無法放入 GPU，需要 TP / PP 來分割模型。

如果單一 Replica 已能容納模型，但 Aggregate Request Load 太高，通常優先考慮增加 DP Replicas。

若單個 Request 的 Latency 太高，才進一步評估 TP、Kernels、Quantization、Prefill/Decode Separation 等。

不能把加 GPU 當成所有問題的唯一答案。

### Q5. 如何證明新 Serving Engine 比舊版更好？

不能只提供一張 Throughput Benchmark 表。

應使用相同模型版本、資料分布、硬體與 SLO，測量：

- TTFT P50 / P95 / P99
    
- TPOT / ITL
    
- E2E Latency
    
- Sustainable Requests/sec
    
- Output Tokens/sec
    
- VRAM Usage
    
- OOM / Failure Rate
    
- Task Quality
    
- Cost per Successful Task
    

最後再進行 Canary、Shadow Traffic、Regression Evaluation 和 Production Monitoring。

這才是完整的工程驗證。

## Part 8 — Intern、Senior Applied Engineer、Senior Inference Engineer 的技能深度差異

|技術|Intern|Senior Applied AI Engineer|Senior LLM Inference Engineer|
|---|---|---|---|
|Transformer Inference|理解 Prefill/Decode|能分析應用瓶頸|深入運算與 GPU Execution|
|vLLM / TRT-LLM|能啟動模型|能部署及 Benchmark|能調整 Runtime / Kernels|
|Continuous Batching|理解概念|能調整 Scheduling 設定|能分析 Scheduler Internals|
|KV Cache|知道避免重算|能估算 VRAM 與 Concurrency|能研究 Cache Layout / Offloading|
|Quantization|知道 INT4/INT8|能比較品質與成本|能優化低精度 Kernels|
|TP / PP / DP|理解差異|能設計 Multi-GPU 部署|能分析 Communication Overhead|
|Latency / Throughput|能解讀 Metrics|能建立 SLO 與 Benchmark|能做 GPU-level Bottleneck Analysis|
|Cloud|知道 Deployment|能建立可靠 Production Serving|能優化 GPU Cluster Infrastructure|
|Reliability|理解 Retry/Fallback|能設計完整 Fault Handling|能處理 Distributed Serving Failures|
|Cost|知道 Token Cost|能優化模型選擇與容量|能最大化 Hardware Efficiency|

## 最後：Senior 以上真正需要具備的能力

對於 Senior Applied AI Engineer，重點不一定是自己實作 PagedAttention 或 CUDA Kernel，而是能把整套系統設計、部署、測試與優化完成。

對於 Senior LLM Inference Engineer，則需要進一步深入 GPU Architecture、Kernel Performance、Distributed Scheduling 與記憶體效率。

可以把整個領域濃縮成五個核心問題：

1. Memory： 模型權重與 KV Cache 能否在 GPU 中高效率運作？
    
2. Scheduling： 如何同時服務大量 Requests，而不造成過高 Tail Latency？
    
3. Performance： 如何在 TTFT、TPOT、Throughput 之間選擇最佳工作點？
    
4. Reliability： 如何在 GPU OOM、過載、模型更新、網路異常時維持服務？
    
5. Economics： 如何在品質達標的前提下，把每個 Successful Task 的成本降到最低？
    

真正成熟的 Production Optimization，應同時最大化三個面向：

\[ \boxed{ \text{Task Quality} \quad+\quad \text{Performance / Reliability} \quad+\quad \text{Cost Efficiency} } \]

它們之間存在交換關係，因此不存在適合所有 LLM、GPU 與工作負載的單一最佳配置。

如果是準備 2026 年美國 Senior / Staff AI Engineer 技術面試，最值得練習的能力，就是拿到一份實際的 Traffic Requirement 後，能自己推導 GPU 容量、選擇 Inference Engine、設計 Benchmark、確認 SLO，最後將系統安全地部署至 Production。