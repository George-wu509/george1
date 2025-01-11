
Llama Recipes: [Examples to get started using the Llama models from Meta](https://github.com/meta-llama/llama-recipes/tree/main)

[Unsloth github](https://github.com/unslothai/unsloth)  inetune Llama 3.2, Mistral, Phi-3.5, Qwen 2.5 & Gemma 2-5x faster with 80% less memory!

LLama3.2 [mycolab](https://colab.research.google.com/drive/1T5-zKWM_5OD21QHwXHiV9ixTRR7k3iB9?usp=sharing#scrollTo=2eSvM9zX_2d3) (Llama-3.2 1B+3B Conversational + 2x faster finetuning.ipynb)

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct", # or choose "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)
```
- **`unsloth`** 是一個專注於高效模型加載和微調的庫，特別針對現代大型語言模型（如 Llama 系列）。
- **`FastLanguageModel`** 是該庫的一個核心類別，用於加載和操作快速語言模型。
#### **參數解析**

1. **`model_name`**
    - 指定要加載的模型名稱。這裡選擇了 `unsloth/Llama-3.2-3B-Instruct` 或 `unsloth/Llama-3.2-1B-Instruct`。
    - **`3B` 和 `1B`** 分別表示模型的參數大小（30億和10億參數）。
2. **`max_seq_length`**
    - 設置模型支持的最大序列長度，影響模型處理的上下文範圍。
3. **`dtype`**
    - 指定模型權重的數據類型，比如 `torch.float16` 或 `torch.float32`。
    - 減小數據類型（如 `float16`）可以節省顯存，但可能影響精度。
4. **`load_in_4bit`**
    - 如果設置為 `True`，模型將以 4-bit 精度加載，進一步減少顯存佔用。
    - 適用於資源受限的場景。
5. **`token`**
    - 如果使用受限的模型（例如需要 API 憑證的模型），需提供對應的 Hugging Face Token。


```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

```
`FastLanguageModel.get_peft_model()` 函數的主要功能是為現有的語言模型添加參數高效微調（PEFT, Parameter-Efficient Fine-Tuning）適配器，特別是 LoRA（Low-Rank Adaptation）適配器。這種方法允許在不改變模型主要結構的情況下，通過微調少量參數來適應特定任務，從而大幅降低計算資源需求。

**作用：** 該函數將 LoRA 適配器整合到指定的模型中，使其能夠在保持原始模型大部分權重不變的情況下，通過微調少量新增的參數來適應新的任務需求。這種方法有效地減少了訓練所需的計算資源和時間，同時保持模型性能。

**原理：** LoRA 的核心思想是將大型模型中的權重更新表示為兩個低秩矩陣的乘積。具體而言，對於需要微調的權重矩陣 $W$，LoRA 引入兩個較小的矩陣 $A$ 和 $B$，使得權重更新 $\Delta W$ 可以表示為 $A \times B$。在訓練過程中，凍結原始權重 $W$，僅訓練 $A$ 和 $B$，從而大幅減少需要更新的參數數量。這種低秩近似方法有效地降低了微調的計算成本和存儲需求。
#### **參數解析**
1. **`r`**
    - 表示 LoRA 的秩（rank）。秩越高，模型的表示能力越強，但計算資源需求越高。
    - 通常選擇 8、16、32、64 或 128。
2. **`target_modules`**
    - 指定要應用微調的模型模塊。
    - 包括 QKV 投影模塊（`q_proj`, `k_proj`, `v_proj`）和其他權重模塊（如 `o_proj` 等）。
3. **`lora_alpha`**
    - LoRA 的比例因子，控制微調權重的影響力。
4. **`lora_dropout`**
    - LoRA 微調過程中的 dropout 率。設置為 `0` 可以最佳化結果。
5. **`bias`**
    - 指定微調是否影響模型的偏置參數。通常設置為 `"none"`，因為這是最佳化的選擇。
6. **`use_gradient_checkpointing`**
    - 使用梯度檢查點來節省內存。設置為 `"unsloth"` 可進一步優化，特別適合處理長上下文的任務。
7. **`random_state`**
    - 隨機種子，用於確保微調的結果可重現。
8. **`use_rslora`**
    - 是否啟用 Rank Stabilized LoRA，進一步提升微調穩定性。
9. **`loftq_config`**
    - 用於啟用 LoftQ（量化技術）的配置
---
### 4. **特點與優化說明**

1. **節省 VRAM**
    - 使用 `unsloth` 的優化技術（例如梯度檢查點），可節省顯存，支持更大的批量大小。
2. **支持長上下文**
    - `unsloth` 提供的上下文處理優化適合長文本的場景。
3. **LoRA 高效微調**
    - 使用 LoRA 技術對模型進行參數高效微調，只需更新部分權重，大幅降低訓練資源需求。
4. **靈活性**
    - 支持多種設置，如梯度檢查點、量化配置（LoftQ），可以根據需求調整模型的性能與資源需求。

---

### 總結

這段程式碼展示了如何利用 `unsloth` 庫來高效加載和微調大型語言模型，並通過 LoRA 等技術進行優化。整體流程非常靈活，適合於資源有限但需要處理高效文本生成、長上下文或多樣化微調任務的場景。