

|                                              |     |
| -------------------------------------------- | --- |
| [[#### 建立最簡單的Vision Language Model repo]]    |     |
| [[#### 比較使用nanoVLM, PaliGemma, Fine-tuning]] |     |
| [[#### 確定一下LLM的完整的 LLM Lifecycle]]           |     |
| [[#### 基於nanoVLM建立一套完整的VLM repo]]            |     |
|                                              |     |
|                                              |     |

#### 建立最簡單的Vision Language Model repo
```
如果我想要建立最簡單的Vision Language Model repo可以連結image跟Language. 從零預訓練 → 後訓練 → 微調 → 推理, 有完整的資料結構. 是否有推薦的github repo以及提供意見並總結. 另外也關於這些不同功能: **影像描述（Image Captioning), 視覺問答（VQA), 視覺定位（Visual Grounding）是否要有不同的資料結構還是可以在同個repo一起訓練或個別訓練?**
```
# 建立自己的 Vision-Language Model（VLM）：從零預訓練、後訓練、微調到推理

如果你的目標是建立一個架構簡單、能看懂完整原始碼、可以親自訓練，而且同時支援 Image Captioning、VQA、Visual Grounding 的 VLM GitHub Repository，我最推薦從 Hugging Face 的 nanoVLM 開始，再參考 PaliGemma 和 Florence-2 擴充多任務能力。

但首先有一個非常重要的區別：

- 真正從零預訓練（Training from Scratch）：Vision Encoder、Language Model 都從隨機權重開始訓練。
    
- 多模態預訓練（Multimodal Pretraining）：利用已經訓練好的 Vision Encoder 和 Language Model，從零訓練連接兩者的 Projection Layer，再逐步聯合訓練。
    
- 後訓練（Post-training）：透過 Instruction Tuning、Preference Optimization 等方法，改善模型遵循指令和回答問題的能力。
    
- 微調（Fine-tuning）：針對特定應用資料進一步優化，例如讓模型專門回答手錶零件位置的問題。
    

對個人研究與建立第一個 VLM repo 而言，我建議主要實驗採取第二種方法，同時保留第一種真正 From Scratch 的小模型實驗。

這樣才能兼顧理解模型架構、可實際執行，以及合理的 GPU 成本。

## 一、2026 年值得參考的 GitHub Repository

以下是我核對過原始 GitHub 專案與文件後，最值得比較的幾個選擇。

|Repository|適合用途|Captioning / VQA / Grounding|我的評價|
|---|---|---|---|
|[Hugging Face nanoVLM](https://github.com/huggingface/nanoVLM)|從底層學習 VLM 架構和訓練|前兩者較直接；Grounding 需擴充|首選|
|[TinyLLaVA Factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory)|模組化、多階段訓練、LoRA|Captioning、VQA；Grounding 需擴充|第二推薦|
|[Google big_vision / PaliGemma](https://github.com/google-research/big_vision)|真正多任務 VLM 研究|三者皆有對應能力|研究進階選擇|
|[Microsoft Florence-2 Fine-tuning](https://github.com/microsoft/dstoolkit-finetuning-florence-2)|Caption、Detection、Grounding 實作|多種視覺任務，VQA 需額外設計|適合空間定位|
|[Salesforce LAVIS](https://github.com/salesforce/LAVIS)|學習經典 Image-Text 預訓練方法|Captioning、VQA，其他需擴充|理論與研究參考|
|[LLaVA](https://github.com/haotian-liu/LLaVA)|學習經典 Multimodal Pretraining + Instruction Tuning|Captioning、VQA；Grounding 需擴充|學習完整訓練流程|

其中 nanoVLM、TinyLLaVA、PaliGemma 和 LLaVA 都公開了可以參考的模型與訓練實作，但它們並不全部提供從隨機權重訓練 Vision Encoder 和 LLM 的一鍵式流程。

![](https://www.google.com/s2/favicons?domain=https://github.com&sz=32)

GitHub

+5

## 二、為什麼最推薦 nanoVLM？

![huggingface/documentation-images at main](https://images.openai.com/static-rsc-4/zSwTXqU2z2RIue9ujX0k7-vdwwBsFY8gqmZboo7VlaYgk3eNdLJzhQZuH8qU3PN7KpL_6I3qvOhMfCQxL_rAXfzWNmkqC0gcWvWVVZLoxAV1dmrBPp9kvLc17Ksz00GVoemQuIfTGl70hZgXiz9H-4NDaL5-Wpzy_2Y1gi6qSHw?purpose=inline)

[huggingface/nanoVLM](https://github.com/huggingface/nanoVLM)

這是一個以 PyTorch 撰寫、專門幫助開發者理解及訓練小型 VLM 的開源專案。它刻意把主要模型架構與訓練程式保持簡潔。

適合學習模型內部運作，而不只呼叫現成 API。

官方介紹中的主要程式碼模組包括：

|原始檔|功能|
|---|---|
|`models/vision_transformer.py`|將影像轉換成 Visual Tokens|
|`models/language_model.py`|Transformer Language Decoder|
|`models/modality_projection.py`|將 Visual Features 映射到 LLM 維度|
|`models/vision_language_model.py`|組合 Vision + Language|
|`models/config.py`|模型與訓練設定|
|`train.py`|訓練流程|
|`generate.py`|推理與文字生成|

nanoVLM 官方指出主要模型定義和訓練邏輯約 750 行，而且可以直接使用預訓練的 Vision Backbone 和 Language Backbone。官方最早的 222M 教學配置結合 SigLIP 與 SmolLM2，後續也推出了 450M 配置。

![](https://www.google.com/s2/favicons?domain=https://github.com&sz=32)

GitHub

+1

特別提醒： nanoVLM 的 `main` 分支曾有不相容的重大更新。官方表示，如果希望使用較簡單、容易研究的早期版本，可以選擇 `v0.1` release。這也是我建議第一次實驗先固定 Git tag 或 commit，而不要無條件追最新版的原因。

### 另外兩個值得一起研究的專案

![PaliGemma: A lightweight open vision-language model (VLM) | DigitalOcean](https://images.openai.com/static-rsc-4/j8Cn8AF7RY5hAYKJP_QFyOFyqOGZlGB58gfZXo95Ft5O-EkzDWNYGDUTcgeDfSo3QhaMQaRgZDZ1Ei26Ds0NLZDhbQj7XxOVu8bjESF7Tf1EgAe9-DWLQjGxdOjzEZv2SM0nsJwOxVfi4rEv6_QZUg2nPh4UpNS-ULIaWPl_bTM?purpose=inline)

PaliGemma

很適合研究如何讓同一個 VLM 處理 Caption、VQA、Object Detection、Segmentation 等不同任務。官方提供多階段預訓練說明及相關程式。

![](https://www.google.com/s2/favicons?domain=https://github.com&sz=32)

GitHub

+1

![Florence-2: Vision-language Model](https://images.openai.com/static-rsc-4/Ox_E47GUeKYblo9nd8pE_tdDy_9ypNfGVGq6ySYWmG9MlSX9FR84ggGX1v06a5EDM6AU6iLyHuOnXLn9pFH9fDWcJe-6qDnmDngr4YkCf94aqbMho1ef5Aci-LvBX1eEApogLhT6si5NE-fCfydCsNZZPesGfLFANvchW1t9R0o?purpose=inline)

Florence-2

特別適合研究 Visual Grounding。它使用不同 Task Prompt 區分 Captioning、Detection、Phrase Grounding、Segmentation 等功能，提供多任務架構的實際參考。

![](https://www.google.com/s2/favicons?domain=https://huggingface.co&sz=32)

Hugging Face

+1

需要注意的是，適合研究模型原始碼與模型本身已原生支援三種任務是兩個不同的評選標準。nanoVLM 在前者最好；PaliGemma 和 Florence-2 在理解多任務資料表示方面更有參考價值。

## 三、Captioning、VQA、Visual Grounding 能否放在同一個 Repo？

可以，而且我建議使用同一套模型架構、同一套 Dataset Interface，支援三種任務共同訓練及個別訓練。

但需要區分兩個概念：

1. 共同 Repository：三種任務共用模型、Data Loader、Trainer、Evaluation 框架。
    
2. 共同 Model：同一組模型權重能夠完成三種任務，而不需要載入三個不同模型。
    

兩者都可以做到，卻不代表三種任務的原始 Annotation 必須完全相同。

### 三種任務比較

|項目|Image Captioning|VQA|Visual Grounding|
|---|---|---|---|
|Input|Image|Image + Question|Image + Referring Expression|
|Output|文字描述|問題答案|Bounding Box / Point / Mask|
|範例 Input|描述這張圖片|錶盤是什麼顏色？|找出時針的位置|
|範例 Output|一支藍色錶盤的手錶|Blue|`[x1,y1,x2,y2]`|
|Ground Truth|Caption Text|Answer Text|Spatial Annotation|
|常見 Dataset|COCO Captions|VQAv2、GQA|RefCOCO、RefCOCO+|
|Evaluation|CIDEr、BLEU、SPICE|Accuracy、VQA Score|IoU、Acc@0.5|
|特殊處理|多參考描述|多個可接受答案|座標、影像縮放|

實際資料集的標註形式可參考 Microsoft 的 [COCO Data Format](https://github.com/microsoft/vision-datasets/blob/main/COCO_DATA_FORMAT.md)，其中也有 Multitask Dataset 的資料結構說明。RefCOCO 則包含文字指涉與目標位置的對應標註。

![](https://www.google.com/s2/favicons?domain=https://github.com&sz=32)

GitHub

+1

### 同一模型如何理解三種任務？

建議的 Shared VLM Architecture：以不同 Task Prompt 控制輸出內容

例如同一張手錶圖片可以輸入：

|Task Prompt|Model Target Output|
|---|---|
|`Describe this image.`|`A wristwatch with a blue dial.`|
|`What color is the dial?`|`Blue`|
|`Locate the hour hand.`|`<box>0.40,0.25,0.55,0.52</box>`|

上述定位座標是格式示例，不是實際測量結果。

三種任務都可以轉化為：

\[ P(\text{Answer}\mid \text{Image},\text{Prompt}) \]

模型依靠 Task Prompt 決定要回答文字、問題答案，還是空間座標。

這類方法稱為 Unified Multimodal Generative Modeling。Florence-2 和 PaliGemma 都有利用提示與結構化輸出支援多種視覺任務的設計。

![](https://www.google.com/s2/favicons?domain=https://huggingface.co&sz=32)

Hugging Face

+1

### 四、我推薦的統一資料結構

我建議不要將所有資料強制塞進只有 `image`、`question`、`answer` 的簡單格式，因為 Grounding 需要額外保留 Bounding Box、原始尺寸及其他空間資訊。

更好的設計是：

Common Schema + Task-specific Annotations + Task-specific Serializer

例如三種訓練資料可以分別存在 JSONL 檔案中，但使用相同的欄位規則。

#### 1. Captioning 資料

`caption_train.jsonl`

```

{
  "id": "caption_0001",
  "image_id": "watch_0001",
  "image": "images/watch_0001.jpg",
  "task": "caption",
  "instruction": "Describe this image.",
  "label": {
    "text": "A watch with a blue dial and silver hands."
  }
}
```

#### 2. VQA 資料

`vqa_train.jsonl`

```

{
  "id": "vqa_0001",
  "image_id": "watch_0001",
  "image": "images/watch_0001.jpg",
  "task": "vqa",
  "instruction": "What color is the dial?",
  "label": {
    "text": "Blue",
    "accepted_answers": ["Blue", "Dark blue"]
  }
}
```

#### 3. Visual Grounding 資料

`grounding_train.jsonl`

```

{
  "id": "grounding_0001",
  "image_id": "watch_0001",
  "image": "images/watch_0001.jpg",
  "task": "grounding",
  "instruction": "Locate the hour hand.",
  "image_size": {
    "width": 640,
    "height": 480
  },
  "label": {
    "phrase": "hour hand",
    "boxes": [[0.40, 0.25, 0.55, 0.52]],
    "box_format": "xyxy",
    "coordinate_system": "relative_0_1"
  }
}
```

以上是我建議你自行定義的 Repo Schema，並非 nanoVLM 原本已內建支援的 JSONL 格式。實際使用時，需要建立 Dataset Adapter。

其核心好處是 `caption`、`vqa`、`grounding` 能共用 Loader，卻又能完整保留不同形式的 Ground Truth。

Grounding 特別需要注意的地方

- `xyxy` 代表左上角與右下角：`[x1,y1,x2,y2]`。
    
- `relative_0_1` 代表座標以原始影像寬高正規化，範圍介於 0 和 1。
    
- 同一段文字可能對應多個 Boxes，因此我用 `boxes` 陣列。
    
- Resize、Crop、Padding、Flip 時，Bounding Box 都必須同步轉換。
    
- 必須保存座標格式與原始影像大小，以便推理後正確畫回圖片。
    

這個資料結構還能擴充 OCR、Segmentation、Object Detection、Region Captioning，不需要重新設計整個 Repository。

## 五、從零建立 Repository：我建議的完整目錄結構

如果要真正建立自己的 GitHub 專案，我會以 nanoVLM 為基礎，建立一個新的 `simple-vlm` Repository。

重點是保留最重要的功能，但不要一開始就加入 DeepSpeed、分散式訓練、複雜 Agent 或 Deployment Infrastructure。

```

simple-vlm/
│
├── README.md
├── pyproject.toml
├── requirements.txt
│
├── configs/
│   ├── scratch.yaml
│   ├── pretrain_alignment.yaml
│   ├── posttrain_sft.yaml
│   ├── finetune.yaml
│   └── inference.yaml
│
├── models/
│   ├── vision_encoder.py
│   ├── language_decoder.py
│   ├── projector.py
│   └── vlm.py
│
├── data/
│   ├── raw/
│   │   ├── coco/
│   │   ├── vqa/
│   │   └── refcoco/
│   │
│   ├── images/
│   │
│   └── processed/
│       ├── train/
│       │   ├── caption.jsonl
│       │   ├── vqa.jsonl
│       │   └── grounding.jsonl
│       ├── val/
│       └── test/
│
├── datasets/
│   ├── dataset.py
│   ├── adapters.py
│   ├── task_serializer.py
│   └── collator.py
│
├── train.py
├── generate.py
├── evaluate.py
│
├── evaluation/
│   ├── caption_metrics.py
│   ├── vqa_metrics.py
│   └── grounding_metrics.py
│
├── checkpoints/
│   ├── scratch/
│   ├── alignment/
│   ├── sft/
│   └── finetuned/
│
└── tests/
    ├── test_dataset.py
    ├── test_model.py
    └── test_inference.py
```

這是建議自行建立的專案架構，不是已存在 nanoVLM Repo 的完整目錄清單。

各個關鍵模組的職責如下：

|模組|職責|
|---|---|
|`vision_encoder.py`|Image → Visual Embeddings|
|`language_decoder.py`|Tokens → Text Generation|
|`projector.py`|Vision Embedding Dimension → LLM Dimension|
|`vlm.py`|整合模型並定義 Forward Pass|
|`dataset.py`|載入圖片與標註|
|`adapters.py`|將 COCO、VQA、RefCOCO 格式轉為統一 Schema|
|`task_serializer.py`|將不同任務標註轉換成模型需要的 Target Tokens|
|`collator.py`|Image Preprocessing、Tokenization、Padding、Label Masking|
|`train.py`|控制不同訓練階段|
|`generate.py`|推理與回答生成|
|`evaluate.py`|執行三種任務的驗證|

### Task Serializer 怎麼運作？

例如 `task_serializer.py` 可以包含類似以下程式：

```
def serialize_target(sample):    task = sample["task"]    label = sample["label"]    if task in ("caption", "vqa"):        return label["text"]    if task == "grounding":        boxes = label["boxes"]        if not boxes:            return "<no_box>"        return " ".join(            "<box>" + ",".join(f"{v:.3f}" for v in box) + "</box>"            for box in boxes        )    raise ValueError(f"Unknown task: {task}")
```

如此一來，三種資料雖然有不同的 Ground Truth，卻都能變成 Language Decoder 可以訓練的文字目標。

`<box>` 和 `<no_box>` 可以註冊成專用 Special Tokens，但必須確保 Tokenizer Vocabulary 與 Embedding Matrix 正確擴充及保存。

這種方法最簡單；如果以後對 Bounding Box 精度有更高要求，還能另外加上 Box Regression Head。

## 六、完整 Training Pipeline：從零預訓練到推理

我建議將整個專案分成六個階段，其中真正從零訓練的實驗可以獨立執行。

Phase 0 — Unimodal Pretraining

Vision Encoder / Language Model 從零訓練，或載入現成權重

Phase 1 — Multimodal Alignment

Image-Text Pairs，訓練 Projection Layer

Phase 2 — Supervised Post-training

Caption + VQA + Grounding Instruction Tuning

Phase 3 — Task / Domain Fine-tuning

LoRA 或 Full Fine-tuning，專門資料優化

Phase 4 — Evaluation

Accuracy、CIDEr、IoU、Error Analysis

Phase 5 — Inference

Image + Prompt → Caption / Answer / Bounding Box

### Phase 0：真正從零預訓練

假設你完全不用預訓練權重：

Vision Encoder

建立小型 ViT，將影像分割成 Patches，再經過 Transformer Encoder 產生 Visual Features。

Language Decoder

建立小型 Decoder-only Transformer，利用 Text Dataset 做 Next-token Prediction。

典型語言模型損失：

\[ \mathcal{L}_{LM}=-\sum_t\log P(x_t\mid x_{<t}) \]

Vision Encoder 可以先進行 Image Classification、Masked Image Modeling 等訓練，也可以直接透過後續 Image-to-Text 目標聯合學習。

如果兩個模組都從隨機權重開始，並沒有必要先獨立預訓練完畢才能連接；但先做 Unimodal Pretraining 通常是比較容易成功的訓練策略。

最小實驗設定（教學用途）：

|參數|建議|
|---|---|
|Image Resolution|128 × 128|
|Vision Encoder|小型 ViT|
|Language Model|約 10M–50M Parameters|
|Embedding Dimension|256|
|Image Patch Size|16|
|Dataset|少量簡單 Image-Text Pairs|
|目標|驗證可學習、可生成文字|

這個設定的目的只是驗證整個 Backpropagation 與 Cross-modal Learning Pipeline，不能期待用少量資料訓練出具有廣泛理解能力的 VLM。

### Phase 1：Multimodal Alignment

在這個階段，我推薦使用：

- 預訓練的 SigLIP Vision Encoder。
    
- 預訓練的 SmolLM2 Language Model。
    
- 隨機初始化的 MLP Projector。
    

初期 Freeze Vision Encoder 和 Language Decoder，只訓練 Projector。

假設 Vision Encoder 輸出：

\[ V\in\mathbb{R}^{N_v\times D_v} \]

MLP Projection：

\[ Z_v=MLP(V)\in\mathbb{R}^{N_v\times D_{LLM}} \]

接著將 Image Tokens 與 Prompt Tokens 組合，讓 Language Decoder 預測 Caption：

\[ \mathcal{L}_{align} =-\sum_{t\in \text{answer}}\log P(y_t\mid I,q,y_{<t}) \]

這裡要注意，這不是 CLIP 的 Contrastive Loss，而是使用 Generative Language Modeling Objective 訓練 Image-to-Text Alignment。

CLIP 類方法可以利用對比學習對齊影像與文字向量，但如果目的是生成 Caption、回答 VQA，使用上述 Generative Alignment 更符合這個最小架構。

LLaVA 官方的初始訓練也採用凍結 Vision Encoder 和 LLM、訓練 Connector 的方式。

![](https://www.google.com/s2/favicons?domain=https://github.com&sz=32)

GitHub

+1

### Phase 2：Post-training / Instruction Tuning

這個階段開始將三種任務放在同一個訓練過程。

例如 Dataset Sampling：

|Task|初始抽樣比例（示例）|
|---|---|
|Captioning|40%|
|VQA|40%|
|Grounding|20%|

這些比例不是最佳化結果，而是為了方便最初的 Experiment。

訓練時可以：

- Vision Encoder：Freeze 或部分更新。
    
- Projector：Trainable。
    
- Language Decoder：LoRA 或 Full Fine-tuning。
    

使用 Cross-entropy Loss，並將 Image/Prompt/Padding 對應的非預測位置設為 `-100`，讓 Loss 主要計算 Assistant Answer Tokens。

需要注意 Grounding 的數值輸出可能不夠精確，模型可能生成不合法座標，因此必須有 Bounding Box Parser 和 Validation。

Hugging Face 的 TRL `SFTTrainer` 已經支援 VLM 的 image-based training datasets，但其資料格式與上述自訂 JSONL 不同，需要經過轉換。

![](https://www.google.com/s2/favicons?domain=https://github.com&sz=32)

GitHub

+1

### Phase 3：Task-specific Fine-tuning

假設模型完成三任務共同訓練後，你希望它專門辨識手錶。

這時可用你自己的資料進行 Domain Adaptation。

例如：

|Task|專用資料|
|---|---|
|Captioning|手錶外觀、錶盤、指針描述|
|VQA|錶面文字、零件、位置問答|
|Grounding|時針、分針、日期窗、刻度 Bounding Boxes|

可以使用 LoRA 減少 Trainable Parameters。

但我不建議只微調 Grounding 後就直接替換共用模型，因為可能會降低原本的 Captioning、VQA 能力。

比較合理的是先測試三種方案：

1. Joint Model：三種任務一起微調。
    
2. Joint Model + Task LoRA：共用 Backbone，但不同任務使用不同 Adapter。
    
3. Separate Models：每個任務訓練獨立模型。
    

如果單一模型的 Grounding Precision 一直無法滿足要求，才考慮專用 Box Head 或獨立 Grounding Model。

### 那 DPO、RLHF、GRPO 需要嗎？

對第一個 VLM Repo，完全不需要一開始就實作。

SFT 本身就是 Post-training 的重要形式，而 Fine-tuning 是較廣泛的概念；兩者不是完全獨立、互不重疊的階段。

等基本模型成功後，再考慮使用 Preference Data：

```

{
  "image": "images/watch_0001.jpg",
  "prompt": "What color is the watch dial?",
  "chosen": "The dial is blue.",
  "rejected": "The dial is red."
}
```

這是概念化的 Preference Record。實際接入 DPOTrainer 時，還須轉換為其要求的 Image 與 Prompt/Chosen/Rejected Dataset Format。

Hugging Face TRL 目前已有 VLM DPO 和 GRPO 的訓練支援，但並非所有自訂 VLM 架構都能直接使用。

![](https://www.google.com/s2/favicons?domain=https://huggingface.co&sz=32)

Hugging Face

+1

## 七、同時訓練與個別訓練，究竟如何選？

這是設計 VLM Repository 時很重要的決策。

|方法|優點|缺點|建議|
|---|---|---|---|
|三任務完全共用模型|架構簡單、參數共用|任務互相干擾|第一版首選|
|共用 Backbone + 不同 LoRA|可以個別優化、參數量較低|Adapter 管理較複雜|第二版|
|共用 Backbone + 不同 Task Heads|有利結構化 Grounding|模型和 Loss 較複雜|定位精度要求高時|
|三個獨立模型|每個任務可單獨最佳化|記憶體、維護成本較高|有充分理由才採用|

我會先從 Single Generative VLM + Task Prefixes 開始。

不過有兩個非常重要的實作細節。

第一，Joint Training 不等於簡單合併所有資料。

如果 Captioning 有 100 萬筆、Grounding 只有 1 萬筆，直接合併後隨機抽樣，Grounding 很可能因為資料量太少而學不好。

應使用 Task-aware Sampler，設定抽樣權重並做實驗比較。

第二，Dataset Split 必須避免資料洩漏。

同一張影像可能同時存在 Captioning、VQA、Grounding Dataset。

不能讓一張圖片出現在 Caption Training，卻又出現在 VQA 或 Grounding Test，然後將測試表現解讀為完全沒見過影像的泛化能力。

應盡可能以 `image_id` 或原始影像群組進行一致的切分，跨資料集還要處理重複影像。

## 八、建議的資料集與模型評估方式

|任務|推薦資料來源|核心 Metrics|
|---|---|---|
|Image Captioning|[MS COCO](https://cocodataset.org/)|CIDEr、BLEU、SPICE|
|VQA|[VQAv2](https://visualqa.org/)、GQA|VQA Accuracy、Exact Match|
|Visual Grounding|[RefCOCO 系列](https://github.com/lichengunc/refer)|IoU、Grounding Accuracy@0.5|
|跨任務綜合|自行建立 Mixed Test Set|Per-task Score、Regression|

其中 Grounding Accuracy@0.5 通常代表預測 Bounding Box 與 Ground Truth 的 IoU 至少達 0.5 的比例。

例如：

\[ IoU=\frac{|B_{pred}\cap B_{gt}|}{|B_{pred}\cup B_{gt}|} \]

如果你的目標物很小，例如錶盤上的細指針，IoU@0.5 可能不足以衡量精度，需要再看 IoU@0.75、Center Error、不同大小物體的分組表現。

除了模型品質，也應該固定記錄 Training Loss、Validation Loss、GPU Memory、Tokens/sec、Inference Latency，確保每次模型變更都能比較。

## 九、具體如何開始實作？

我建議分成兩條路線。

### 路線 A：先讓現成 nanoVLM 跑起來

這條路最適合第一週的研究。

Windows PowerShell：

```

git clone https://github.com/huggingface/nanoVLM.git
cd nanoVLM

# 研究用：固定較簡潔的舊版
git checkout v0.1

python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 先依照 pytorch.org 指示安裝對應 CUDA 的 PyTorch
pip install numpy pillow datasets transformers huggingface-hub wandb

# 按版本 README 設定所需的 Hugging Face / W&B 登入
python train.py

# 推理
python generate.py
```

上述是啟動流程，不代表下載和訓練資料在任何環境都能直接成功。尤其需要檢查該版本的依賴、資料集存取和 GPU 設定。

如果希望先看已訓練好的模型結果，最新 nanoVLM Repo 也提供了相應 Checkpoint 與 `generate.py` 介面。注意不要將新版 450M Checkpoint 直接套用到不相容的 `v0.1` 程式。

### 路線 B：建立自己的 `simple-vlm`

這是我更建議的長期方向。

先從 nanoVLM 參考 `vision_transformer.py`、`language_model.py`、`modality_projection.py` 和 `train.py` 的核心邏輯。

再逐步加入：

|里程碑|完成標準|
|---|---|
|M1 — Model|Image + Text 能 Forward，Loss 能 Backpropagate|
|M2 — Alignment|模型能根據簡單圖片生成描述|
|M3 — Multi-task|Caption / VQA / Grounding 都能讀取 Dataset 和訓練|
|M4 — Evaluation|三任務都有獨立 Test Metrics|
|M5 — Fine-tuning|能載入上階段 Checkpoint 進行 LoRA 或 Full Tuning|
|M6 — Inference|上傳圖片、選擇 Task、顯示 Answer / Bounding Box|

這樣就能將整個專案整理成一個可以向其他 AI Engineer 展示的完整 VLM System。

### 建議的硬體配置

|硬體|適合的工作|
|---|---|
|無獨立 GPU|Dataset Processing、小模型 CPU Smoke Test|
|NVIDIA 8–12GB VRAM|小型 nanoVLM、低 Batch Size 的實驗|
|NVIDIA 16–24GB VRAM|較完整的小型 VLM 訓練、LoRA|
|NVIDIA 48–80GB VRAM|更大的模型、更長序列及高解析度實驗|
|多 GPU|更大規模的全參數預訓練|

這是規劃用的粗略分類，不是保證可運行的硬體門檻。實際 VRAM 取決於模型、Optimizer、Precision、Image Resolution、Sequence Length 和 Activation Checkpointing。

nanoVLM 曾公開 222M 配置在 H100 上的 VRAM 測試：Batch Size 1 約 4.45GB，但測試時序列長度僅 128 Tokens，不能直接視為較長序列或完整多任務訓練的需求。

![](https://www.google.com/s2/favicons?domain=https://github.com&sz=32)

GitHub

## 十、我的最終建議與總結

對你想做的第一個完整 VLM Repo，我會選擇下列設計：

## Simple Unified VLM

建議第一版架構

|   |   |
|---|---|
|Framework|PyTorch + Hugging Face|
|Reference Repo|nanoVLM|
|Vision Encoder|SigLIP / Small ViT|
|Language Decoder|SmolLM2 / Tiny Decoder|
|Connector|MLP Projector|
|Training Objective|Generative Cross-entropy|
|Tasks|Captioning + VQA + Grounding|
|Dataset|Unified JSONL Schema|
|Fine-tuning|LoRA / Full Fine-tuning|
|Evaluation|Per-task Metrics|
|Inference|Image + Task Prompt|

最重要的三個結論：

1. 如果重點是研究從零訓練的原理，選 nanoVLM。 它的程式碼量較小，容易閱讀與修改。保留一個獨立的隨機初始化實驗，以真正理解 Vision Encoder、LLM 和 Projector 的訓練。

2. 如果重點是支援 Captioning、VQA、Grounding 的共同模型，研究 PaliGemma 和 Florence-2。 它們能幫助你設計 Task Prompt、Spatial Tokens 和多任務資料表示，而不必從頭發明整套方法。

3. 如果目標是建立可持續擴充的研究專案，用同一個 Repo、同一套資料介面，但允許 Joint Training、Task-specific Fine-tuning 和不同 Head/LoRA。 不要在一開始就把三任務分成完全獨立的模型，也不要強迫不同 Ground Truth 使用完全相同的資料欄位。

尤其你之後如果想將這個 VLM 延伸到精密影像分析、物件定位和手錶零件辨識，這套架構也能保留現有的 Visual Grounding 能力，再逐步加入 Region-level Features、High-resolution Crops 和 Segmentation。

我會優先做出能正確完成三任務的 200M–500M 級小模型，而不是直接嘗試從零訓練數十億參數的大模型。 對學習完整 Training Pipeline、比較實驗、展示可重現成果來說，這通常更有價值。




#### 比較使用nanoVLM, PaliGemma, Fine-tuning
```
那請幫我比較使用nanoVLM, PaliGemma, Fine-tuning 為基礎從零建立 Repository的優缺點? 是否需要對本來repo做很多變動才能有多種功能, 以及哪個repo的資料結構是最好, 以及最適合理解. 請幫我比較
```

# nanoVLM vs PaliGemma vs Florence-2 Fine-tuning：如何選擇最適合自己的 VLM Repository？

根據你希望建立的系統：

Image + Language → 從零預訓練 → 多模態訓練 → 後訓練 → Fine-tuning → Inference，而且能在同一個 Repository 中分別或共同訓練：

- Image Captioning（影像描述）
    
- VQA（Visual Question Answering，視覺問答）
    
- Visual Grounding（利用文字定位影像中的物件）
    

我會把「最容易理解模型架構」、「最少修改就能完成三種任務」以及「Dataset 結構是否清楚且容易擴充」分開評估。

這三個條件不一定會指向同一個 Repository。

以下以截至 2026 年 10 月能查證的 GitHub 官方文件和實作為基礎。

## 一、先給你比較結果

![Understanding Multimodal LLMs - by Sebastian Raschka, PhD](https://images.openai.com/static-rsc-4/oOdhRJQszBIcd9Q80rqmqKjAD57E7CycpcbxYJfa4IszlWjEjG0M-z_EGxTQBMVZM7A6egZ6PUWWb43P5JmqRZAdxZIKwXXpZkcfWGP76lKeqncbzFUoLFJOR8lDN-nkVS2YsgrDF9woZQ1gXkc56yGPw-I72ype-ZZSsPuFFiA?purpose=inline)

1. [nanoVLM](https://github.com/huggingface/nanoVLM)

最推薦學習架構

最適合從零建立自己的 VLM 專案、讀懂每個 Module，並控制 Pretraining、Post-training、Fine-tuning 和 Inference。

缺點是 Grounding 的 Token Format、Dataset Adapter、Evaluation 等功能需要自行擴充。

![PaliGemma: A lightweight open vision-language model (VLM) | DigitalOcean](https://images.openai.com/static-rsc-4/j8Cn8AF7RY5hAYKJP_QFyOFyqOGZlGB58gfZXo95Ft5O-EkzDWNYGDUTcgeDfSo3QhaMQaRgZDZ1Ei26Ds0NLZDhbQj7XxOVu8bjESF7Tf1EgAe9-DWLQjGxdOjzEZv2SM0nsJwOxVfi4rEv6_QZUg2nPh4UpNS-ULIaWPl_bTM?purpose=inline)

2. [PaliGemma / big_vision](https://github.com/google-research/big_vision)

最適合研究真正的多任務 VLM Training Design，已具備 Captioning、VQA、Detection、Segmentation 等任務的模型設計和結構化 Tokens。

缺點是官方研究程式庫使用 JAX/Flax，架構較大，對初學者不如 nanoVLM 直觀。

![What is Phrase Grounding?](https://images.openai.com/static-rsc-4/9_GJW53EH_U1HC-IbFdBBqA4EaG7pY-QeNaqqTBmrpSIDkUrso52mPxTSip5J4z0iDIKBsP7-RtVmLwoM2kO66Fw6ASJFFsrsBLxa19K-bzuHkJEwRN02rwXYTaCyoAG4vXvu5E-QGP8I5mSrAXne5qtoky4vi_42rl1NP1fctA?purpose=inline)

3. [Florence-2 Fine-tuning](https://github.com/microsoft/dstoolkit-finetuning-florence-2)

最適合快速實驗 Image Captioning、Visual Grounding、Object Detection 和 OCR。

但這個 Fine-tuning Repository 主要是範例 Notebook，並不是完整的 From-scratch Pretraining Framework，而且一般任意問題的 VQA 並非 Florence-2 官方列出的標準 Task Prompt。

依據各專案官方 README、PaliGemma 訓練文件以及 Florence-2 模型文件。

![](https://www.google.com/s2/favicons?domain=https://github.com&sz=32)

GitHub

+3

### 重要釐清

你說的第三個「Fine-tuning」，我這裡以之前提到的 Microsoft Florence-2 Fine-tuning Repository 為準。

另外，這三者其實不是完全相同層級的東西：

- nanoVLM 是小型 VLM 的模型與訓練程式庫。
    
- PaliGemma 是模型家族，`big_vision` 是 Google 的研究訓練程式庫。
    
- Florence-2 是模型；Microsoft `dstoolkit-finetuning-florence-2` 是基於該模型的微調教學程式庫。
    

所以如果目標是建立一個真正完整、從 Pretraining 到 Inference 的新 Repository，不能只比較它們現在能回答哪些問題，也必須比較你要自己補多少訓練基礎設施。

## 二、完整功能比較表

以下的「原生」代表現有模型設計或官方程式碼已有對應能力；不代表下載 Repository 後，所有任務都能不經設定直接開始訓練。

|比較項目|nanoVLM|PaliGemma / big_vision|Florence-2 Fine-tuning|
|---|---|---|---|
|核心 Framework|PyTorch|JAX / Flax|PyTorch / Transformers|
|理解 Model Source|容易|困難|中等，模型核心主要在依賴套件|
|從 Random Weights 實驗|最方便改造|技術上可行，但複雜|不適合|
|Multimodal Pretraining|已有基本訓練流程|完整研究架構|Notebook 未涵蓋|
|SFT / Fine-tuning|可擴充|已有 Task Transfer|已有特定任務範例|
|LoRA|需整合|需額外整合|需額外整合|
|Captioning|支援生成式任務|原生支援|原生支援|
|VQA|支援生成式任務|原生支援|非標準內建 Prompt，需額外處理|
|Visual Grounding|需增加結構化訓練與輸出|原生支援定位 Tokens|模型原生支援|
|Bounding Box Postprocessing|自己寫|需解析位置 Tokens|Processor 已有對應功能|
|JSONL Training|需要 Dataset Adapter|官方提供 JSONL Data Source|範例依任務準備資料|
|多任務共同訓練|需擴充 Sampler / Schema|最成熟的研究參考|Notebook 要較大幅重構|
|訓練可讀性|最佳|中等偏低|Notebook 流程容易入門|
|適合從零建立新 Repo|最佳|適合進階研究|適合應用型原型|

這是針對你要建立的新 Repository 所做的工程評估，而不是官方 Benchmark 排名。

特別值得注意：nanoVLM 官方於 2025 年 6 月與 9 月曾更新模型及訓練資料流程，而且提醒部分輔助腳本可能不相容。如果目的是學習最簡單的模型，應評估固定 `v0.1`；如果目的是訓練新一代多圖片模型，再考慮較新的實作。

![](https://www.google.com/s2/favicons?domain=https://github.com&sz=32)

GitHub

## 三、三個 Repository 的資料結構，哪個最好？

這部分我特別檢查了 nanoVLM 的 `data/datasets.py`、PaliGemma 的 `forkme.py` 和 Florence-2 的 Processor。

### 1. nanoVLM：以 Image + Conversation 為中心

[查看 nanoVLM datasets.py 原始碼](https://github.com/huggingface/nanoVLM/blob/main/data/datasets.py)

nanoVLM 目前的 Dataset Loader 使用 `images` 和 `texts` 欄位，`texts` 中包含 `user` 與 `assistant`，再轉換為 Language Model 的 Chat Template。

以下是其主要欄位概念的簡化範例：

```

{
  "images": ["images/watch001.jpg"],
  "texts": [
    {
      "user": "What color is the dial?",
      "assistant": "Blue."
    },
    {
      "user": "Describe the watch.",
      "assistant": "A watch with a blue dial."
    }
  ]
}
```

這個範例只是展示來源資料格式。實際餵入官方 Dataset Class 時，`images` 必須經過資料載入程序轉換成 PIL Image，而不能直接把上述路徑字串傳給 Image Processor。

這種結構的優點是非常容易理解 VQA 與 Captioning 如何統一成文字生成問題。

對 Captioning 而言，User 提供描述指令，Assistant 輸出 Caption。

對 VQA 而言，User 提供 Question，Assistant 輸出 Answer。

但如果是 Grounding：

```
User: Locate the hour hand.
Assistant: [0.40, 0.25, 0.55, 0.52]
```

雖然能將 Bounding Box 當成文字訓練，卻需要增加解析、座標驗證與 Evaluation。

更重要的是，原始 Dataset 並沒有明確的 `task`、`boxes`、`coordinate_system` 等欄位。

因此，我的評價是：

nanoVLM 的 Dataset 很適合學習 Generative VLM，但不是現成最完整的多任務視覺標註結構。

![](https://www.google.com/s2/favicons?domain=https://github.com&sz=32)

GitHub

### 2. PaliGemma：以 Image + Prefix + Suffix 為中心

[查看 PaliGemma JSONL 訓練設定](https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/transfers/forkme.py)

PaliGemma 的資料格式很值得參考，因為它將各種任務轉換成同一種基本結構：

```
Image + Prefix → Suffix
```

`prefix` 是輸入的指令或問題，`suffix` 是期待生成的答案。

例如以下是對應其資料介面的簡化示意：

```

{
  "image": "images/watch001.jpg",
  "prefix": "caption en",
  "suffix": "A watch with a blue dial."
}
```

VQA 可以寫成：

```

{
  "image": "images/watch001.jpg",
  "prefix": "What color is the dial?",
  "suffix": "Blue."
}
```

Grounding 則可以使用它已定義的位置 Tokens。

例如：

```
Prefix: detect hour hand

Suffix:
<loc0256><loc0128><loc0768><loc0640>hour hand
```

PaliGemma 定義了 1,024 個位置 Tokens（`<loc0000>` 到 `<loc1023>`），用來表達正規化位置。

特別注意，PaliGemma Bounding Box Token 的順序是：

```
y_min, x_min, y_max, x_max
```

不是一般 COCO 常見的 `x_min, y_min, x_max, y_max`。 這是將其他 Dataset 轉換成 PaliGemma 格式時必須處理的地方。

![](https://www.google.com/s2/favicons?domain=https://github.com&sz=32)

GitHub

+1

PaliGemma 的優點很明顯：

- Captioning、VQA 可以直接使用文字答案。
    
- Grounding 已有結構化位置 Tokens。
    
- 同一個 Generative Decoder 能學習三種目標。
    
- 官方 `big_vision` 已有 JSONL Data Source 和多種任務的 Preprocessing Config。
    

但也要注意，簡單的是它的 Input/Target 表示方式，不是整個 `big_vision` Training Framework。

例如，官方 `forkme.py` 就涉及 JAX、TensorFlow Input Pipeline、Preprocessing Operators、Distributed Sharding 等機制，不如 nanoVLM 容易從上到下讀懂。

![](https://www.google.com/s2/favicons?domain=https://github.com&sz=32)

GitHub

+1

### 3. Florence-2：以 Task Token + Structured Output 為中心

Florence-2 的設計跟前兩者不同。

它使用固定的 Task Prompt，例如：

|任務|Florence-2 Prompt|
|---|---|
|Captioning|`<CAPTION>`|
|Detailed Captioning|`<DETAILED_CAPTION>`|
|Object Detection|`<OD>`|
|Visual Grounding|`<CAPTION_TO_PHRASE_GROUNDING>`|
|Segmentation|`<REFERRING_EXPRESSION_SEGMENTATION>`|
|OCR|`<OCR>`|

例如你要尋找 Hour Hand：

```
Image: watch001.jpg

Task Prompt:
<CAPTION_TO_PHRASE_GROUNDING>

Text Input:
hour hand
```

Florence-2 會生成模型自己的結構化序列，而它的 `Florence2Processor.post_process_generation()` 可以將序列轉成：

```

{
  "bboxes": [
    [120, 80, 240, 170]
  ],
  "labels": ["hour hand"]
}
```

以上座標是示意值。

這對 Visual Grounding 很方便，因為無須自己從零建立 Bounding Box Tokenizer 與 Postprocessor。

但 Florence-2 官方支援的標準 Task Prompts 並未包含一般開放式 VQA。你可以自行定義 Question Answering 的 Prompt/Target 並微調，但必須另外驗證回答能力，不能假設它已經與 PaliGemma 一樣支援任意問答。

![](https://www.google.com/s2/favicons?domain=https://huggingface.co&sz=32)

Hugging Face

+1

### 三種資料結構最終評比

|評估角度|最推薦|原因|
|---|---|---|
|最容易看懂原始 Dataset Class|nanoVLM|Image + Conversation|
|最簡潔的 Unified Input/Output|PaliGemma|Image + Prefix + Suffix|
|Grounding 的位置表示|PaliGemma|現成 Location Tokens|
|Grounding 的推理後處理|Florence-2|Processor 已實作|
|學習 Chat-based VLM|nanoVLM|跟現代對話式 VLM 概念相近|
|研究多任務 Preprocessing|PaliGemma|不同任務有獨立 Config|
|自行擴充新任務|nanoVLM|內部模組最容易掌控|

如果只問資料結構哪個最好，我會選擇 PaliGemma 的 Prefix/Suffix 設計作為訓練資料的概念基礎。

但這不代表應該直接複製它全部的 `big_vision` 程式庫。

## 四、從現有 Repository 加上三種任務，要修改多少？

這裡我將修改程度分成「低、中、高」三級，表示預估的工程複雜度，而非已實際修改及測量的結果。

|要加入的功能|nanoVLM|PaliGemma / big_vision|Florence-2 Fine-tuning|
|---|---|---|---|
|Captioning Dataset|低|低|低～中|
|VQA Dataset|低|低|高|
|Grounding Dataset|中～高|低～中|中|
|Grounding Special Tokens|高|低，已有定義|低，已有定義|
|Bounding Box Parser|中|中|低|
|多任務混合訓練|中|中|高|
|任務個別 Fine-tuning|中|低～中|中|
|不同 Task Evaluation|中|低～中|中～高|
|從 Random Weights 訓練小型 VLM|中|高|高|
|完整分階段 Training Pipeline|中|中～高|高|

### 使用 nanoVLM，需要修改哪些檔案？

建議的修改範圍

|   |   |
|---|---|
|`models/vision_transformer.py`|原則上不需要為三任務修改|
|`models/language_model.py`|加入 Special Tokens 時須確保 Embedding 相容|
|`models/modality_projection.py`|原則上不需要修改|
|`data/datasets.py`|加入 Task-aware Dataset Loading|
|`data/collators.py`|調整 Padding、Labels、不同資料的 Batch 處理|
|`train.py`|加入 Task Sampling、不同 Training Stage、Checkpoint Management|
|`generate.py`|加入 Grounding Parser 與輸出驗證|
|新增 `task_serializers.py`|將三種任務轉換成 Training Target|
|新增 `evaluation/`|Caption CIDEr、VQA Accuracy、Grounding IoU|

這裡有個很重要的優點：

只要採用 Generative Grounding，不額外建立 Regression Head，Vision Encoder、Projector 和主要 Language Decoder 架構基本可以保持不變。

因此從 nanoVLM 擴充三種任務，工程工作主要集中在 Data、Tokenizer、Training、Evaluation，而不是重新設計整個 Transformer。

如果真的想做更精確的 Bounding Box Regression，才需要進一步改 Model Architecture 和 Loss Function。

### 使用 PaliGemma，需要修改哪些地方？

PaliGemma 比較多工作會落在 Dataset 和 Configuration：

|模組|修改項目|
|---|---|
|JSONL Data Source|讀取你的 Image、Question、Caption、Grounding|
|Preprocessing Config|不同 Task 產生不同 Prefix/Suffix|
|Task Sampling|控制 Caption/VQA/Grounding 比例|
|Training Config|設定 Stage、Freeze、Learning Rate|
|Evaluation|三種任務各自 Metrics|
|Deployment|匯出權重、推理與解析輸出|

它的 Location Tokens 已經存在，所以若使用相容的 PaliGemma Checkpoint，加入 Grounding 通常不需要重新設計 Transformer。

但若直接使用原始 `big_vision`，你仍需要理解 JAX、Flax、Preprocessing Operator 和 Sharding。

因此，我會將「模型功能的改動量」評為較低，但「理解與維護整個研究框架的成本」評為較高。

另外還有一條更簡單的路：使用 Hugging Face Transformers 內的 PaliGemma，而不是直接 Fork Google `big_vision`。這樣能留在 PyTorch 生態系，利用現成模型與 Processor 自行建立 Trainer。

這個方案很適合以多任務 Fine-tuning 為優先的人，但如果目標是理解模型底層，每個核心算子都親自修改，仍不如 nanoVLM 直觀。

### 使用 Florence-2 Fine-tuning，需要修改哪些地方？

我不建議把 Microsoft 的 Fine-tuning Notebook Repository 直接當成完整 VLM Research Framework。

原因很具體：

它的 README 主要提供 `01-Base_Model.ipynb` 與 `02-Object_Detection_Finetuning.ipynb`，後者是 Object Detection 的微調教學。

你若要建立完整系統，需要把 Notebook 邏輯重新組織成可重用的 Dataset、Trainer、Evaluation、Checkpoint 和 Inference Modules。

尤其若加入一般 VQA，還需要：

- 設計自己的 Question/Answer Training Samples。
    
- 建立適當的 Prompt Template。
    
- 驗證模型是否學會回答，不只是生成固定 Task Output。
    
- 在 Joint Fine-tuning 中防止原有 Captioning/Grounding 能力退化。
    

因此它最適合快速學習 Florence-2 的下游任務微調，而不是作為從零 Pretraining 的主程式庫。

![](https://www.google.com/s2/favicons?domain=https://github.com&sz=32)

GitHub

## 五、如果真的要「從零預訓練」，三者差多少？

我想特別強調這點，因為這會決定你要建立的 Repo 究竟是哪一種類型。

|Training 階段|nanoVLM|PaliGemma|Florence-2 Fine-tuning|
|---|---|---|---|
|Vision Encoder 隨機初始化|方便改造|技術上可做，設定複雜|需另建完整訓練|
|Language Model 隨機初始化|方便改造|可在研究框架中實作，成本高|需另建完整訓練|
|Vision-Language Alignment|既有訓練流程可延伸|官方模型訓練設計包含此階段|Notebook 未提供完整流程|
|Multimodal Training|有小模型訓練程式|有研究訓練架構|需自行建立|
|SFT / Task Fine-tuning|容易延伸|有官方任務遷移 Config|已有特定任務示範|
|LoRA / Domain Adaptation|可額外整合|可透過 PyTorch 路線整合|可額外整合|
|Inference|有|有|有|

注意：三個 Repository 都不應被視為提供一鍵完成所有 Foundation Model Pretraining 階段的完整可重現方案。

例如 PaliGemma 官方清楚描述 Stage 0 使用已預訓練的 SigLIP 和 Gemma，後續才進行多模態訓練及高解析度訓練。官方公開的多數執行範例也聚焦於下游 Fine-tuning，而非完整重現大規模 Foundation Model Pretraining 所需的全部資料。

![](https://www.google.com/s2/favicons?domain=https://github.com&sz=32)

GitHub

對你的需求，我建議把「真正隨機初始化」作為一個獨立可執行模式，而不是要求每次實驗都從頭訓練全部權重。

## 六、我會怎麼設計你自己的新 Repository？

我最推薦的做法是：

以 nanoVLM 的 Model 和 Trainer 為骨架，採用 PaliGemma 的 Prefix/Suffix 思想，並參考 Florence-2 的 Grounding Postprocessing。

不是把三個 Repository 的原始碼全部合併在一起，而是取三者最容易理解、最有價值的設計。

### 建議的 Architecture

Unified Raw Dataset

Caption / VQA / Grounding Annotation

Task Adapters & Serializers

Image + Task Prompt + Target Tokens

Image

Visual Input

Task Prefix

Text Input

Target

Supervision

nanoVLM-based Model

Vision Encoder

Projector

LLM Decoder

Training Controller

Scratch / Alignment / SFT / LoRA / Joint Training

Generation + Structured Parser

Text Answer / Location Tokens / Bounding Boxes

Caption

VQA Answer

Grounding Box

### 推薦資料結構：Canonical Schema + Model-specific Adapter

假設同一張影像有三種不同 Annotation，我不會直接將所有資料轉成 PaliGemma Tokens 儲存。

原因是未來如果換成 Florence-2 或新的 VLM，這些模型的 Special Tokens、座標格式與 Output Parser 可能不同。

我會保存模型無關的原始 Annotation。

例如：

```

{
  "sample_id": "watch001_grounding_01",
  "image": {
    "path": "images/watch001.jpg",
    "width": 2048,
    "height": 2048
  },
  "task": "grounding",
  "input": {
    "text": "Locate the hour hand."
  },
  "target": {
    "text": null,
    "boxes": [
      {
        "phrase": "hour hand",
        "xyxy": [0.40, 0.25, 0.55, 0.52]
      }
    ]
  },
  "metadata": {
    "coordinate_system": "relative_0_1",
    "group_id": "watch001",
    "split": "train",
    "source": "human_annotation_v1"
  }
}
```

再由不同的 Serializer 轉成特定模型的輸入格式。

|Adapter|轉換結果|
|---|---|
|`NanoVLMAdapter`|Image + User / Assistant Messages|
|`PaliGemmaAdapter`|Image + Prefix / Suffix + Location Tokens|
|`FlorenceAdapter`|Image + Florence Task Prompt + 對應 Spatial Tokens|

如此可以讓你將來比較三種模型在同一份 Ground Truth 上的表現，不必重新標註圖片。

同樣重要的是，Grounding 的 `xyxy` 和 `relative_0_1` 必須在轉換程序中明確指定，避免將 PaliGemma 的 `yxyx` Token 次序誤當成 `xyxy`。

### 最終 Repository 結構

```

simple-vlm/
│
├── configs/
│   ├── model.yaml
│   ├── train_scratch.yaml
│   ├── train_alignment.yaml
│   ├── train_sft.yaml
│   └── train_finetune.yaml
│
├── models/
│   ├── vision_encoder.py
│   ├── language_decoder.py
│   ├── projector.py
│   └── vlm.py
│
├── data/
│   ├── raw/
│   ├── annotations/
│   └── splits/
│
├── datasets/
│   ├── unified_dataset.py
│   ├── canonical_schema.py
│   ├── task_sampler.py
│   └── adapters/
│       ├── nanovlm_adapter.py
│       ├── paligemma_adapter.py
│       └── florence_adapter.py
│
├── training/
│   ├── trainer.py
│   ├── losses.py
│   ├── checkpoint.py
│   └── stages/
│       ├── scratch.py
│       ├── alignment.py
│       ├── sft.py
│       └── finetune.py
│
├── inference/
│   ├── generate.py
│   ├── grounding_parser.py
│   └── visualize.py
│
├── evaluation/
│   ├── caption.py
│   ├── vqa.py
│   └── grounding.py
│
├── tests/
│   ├── test_dataset.py
│   ├── test_model.py
│   ├── test_training.py
│   └── test_inference.py
│
└── README.md
```

我會保留各個 Training Stage 的明確入口，但共用 Dataset、Model、Checkpoint 與 Trainer 的核心程式，不為每個任務重寫一整套訓練系統。

第一次實作時，可以只保留 `NanoVLMAdapter`，另外兩個留作未來比較模型的擴充接口。

## 七、如果依你的實際目標選擇，我會怎麼決定？

選擇你的優先目標

主要想理解 Transformer / VLM 從零預訓練

希望少改程式就有 Captioning + VQA + Grounding

重點是 Grounding、Detection、OCR

希望完整理解，也建立可持續擴充的多任務 Repo

建議選擇

## nanoVLM + Unified Dataset + PaliGemma-style Tokens

保留最簡單的 Model Source，吸收 PaliGemma 的多任務資料表示與 Florence-2 的空間解析設計。

第一步

以 nanoVLM 為唯一初始模型，分階段完成 Joint Training、Evaluation、Inference。

## 八、最終總結

若完全以你的原始目標——建立最簡單但完整、能理解所有訓練階段、又能支援多種 Image-Language Tasks 的 VLM Repository——作為評估標準，我的結論是：

|你的問題|我的答案|
|---|---|
|哪個 Repo 最適合理解 Model Architecture？|nanoVLM|
|哪個 Repo 最適合學習完整多任務 Training Design？|PaliGemma / big_vision|
|哪個有最容易理解的 Unified Training Target？|PaliGemma 的 Prefix / Suffix|
|哪個現成 Grounding 支援最方便？|PaliGemma 或 Florence-2|
|哪個最適合快速 Fine-tuning Detection？|Florence-2|
|哪個最適合從 Random Weights 學習？|nanoVLM|
|哪個最適合成為你自己的新 Repo？|nanoVLM + 自建 Unified Dataset|

我不建議直接 Fork 完整 `big_vision` 作為你的第一個 VLM 專案，也不建議把 Florence-2 的示範 Notebook 強行擴充成所有階段的訓練框架。

最適合的方案是 保留 nanoVLM 約數百行核心模型與訓練邏輯的教學優勢，再逐步加入 PaliGemma 式多任務輸出表示。

這樣你最終得到的會是一個真正能展示 AI Engineer 能力的研究專案：不只是會使用現成 VLM，而是能理解 Model Architecture、設計多任務 Dataset、建立從預訓練到微調的 Pipeline，並驗證不同任務的 Training Quality。



#### 確定一下LLM的完整的 LLM Lifecycle
```
確定一下LLM的完整的 LLM Lifecycle, - 從零建立 Transformer Model，理解 Pretraining 前的模型如何初始化。
- 準備 Raw Dataset，進行 Cleaning、Tokenization、Train/Validation Split。
- 執行 Pretraining，觀察 Loss、Gradient、Optimizer 與 Checkpoint。
- 執行 Supervised Fine-Tuning（SFT），讓 Base Model 學習遵循指令。
- 執行 LoRA Fine-Tuning，理解 Parameter-Efficient Fine-Tuning。
- 執行 DPO Preference Alignment，理解 Post-training。
- 執行 Inference，觀察 Temperature、Top-p、KV Cache 等設定。
- 保存所有 Dataset、Model Weights、Tokenizer、Configuration、Metrics，並比較各階段模型的差異。 那如果建立VLM上面那些步驟是不需要的, 那完整VLM Lifecycle是甚麼
```

# 完整 VLM Lifecycle：從零建立模型、Pretraining、Post-training、Fine-tuning 到 Inference

首先確認：你列出的八個 LLM Lifecycle 步驟，在 VLM（Vision-Language Model）中幾乎全部仍然需要。

VLM 不是另一套取代 LLM 的訓練流程，而是在 LLM Lifecycle 的基礎上，增加了影像模型、視覺資料處理、Vision-Language Alignment、Multimodal Training，以及視覺任務的評估。

如果你希望建立一個 Repository，能夠真正從 Random Weights 開始，完整實驗各個訓練階段，最後支援 Image Captioning、VQA、Visual Grounding，那我建議將 VLM Lifecycle 設計成以下 14 個階段。

## 一、LLM Lifecycle vs VLM Lifecycle

|原本 LLM 的步驟|VLM 是否需要？|VLM 新增的部分|
|---|---|---|
|1. 建立 Transformer Model|需要|Vision Transformer、Multimodal Projector|
|2. Raw Dataset、Cleaning、Tokenization|需要|Image Cleaning、Resize、Image Patches、Image-Text Pairing|
|3. Pretraining|需要|Vision Pretraining、Multimodal Pretraining|
|4. SFT|需要|Multimodal Instruction Tuning|
|5. LoRA Fine-tuning|可選|Vision / Projector / LLM 各自決定哪些權重更新|
|6. DPO Preference Alignment|可選|Image-conditioned Preference Optimization|
|7. Inference|需要|Image Encoding、Image Tokens、Visual Grounding Decoding|
|8. 保存 Dataset、Weights、Metrics|需要|Image Processor、Vision Weights、Projector、Coordinate Formats|
|9. Vision Pretraining|VLM 新增|ViT、DINO / MAE / Contrastive Learning 等|
|10. Vision-Language Alignment|VLM 新增|對齊 Visual Embeddings 和 LLM Embeddings|
|11. Multimodal Data Mixing|VLM 新增|Caption、VQA、Grounding 等資料混合|
|12. Spatial / Structured Output|視任務需要|Bounding Boxes、Points、Masks、Location Tokens|
|13. Multimodal Evaluation|需要|Caption Accuracy、VQA、IoU、Hallucination|
|14. Multimodal Deployment|部署時需要|Image Preprocessing、Serving、Postprocessing、Monitoring|

其中，SFT、LoRA 和 DPO 不是三個每次都必須依序完成的獨立訓練階段：

- SFT 是訓練目標／後訓練方法。
    
- LoRA 是參數更新方式，可以用在 SFT、DPO 或 Domain Fine-tuning。
    
- DPO 是 Preference Optimization 方法，通常接在 SFT 後，但不是每個 VLM 都需要。
    

這和 LLM 完全相同，只是 VLM 需要讓這些方法同時正確處理影像與文字。

## 二、完整 VLM Lifecycle 架構圖

Phase 0 — Define Model Architecture

Vision Encoder + Projector + Language Decoder

Random Weight Initialization

Phase 1 — Image Data

Quality、Resize、Patchify、Annotation

Phase 1 — Text Data

Cleaning、Tokenizer、Sequence

Phase 2 — Vision Pretraining

Train Image Encoder

Phase 3 — LLM Pretraining

Train Language Decoder

Phase 4 — Vision-Language Alignment

Connect Vision Encoder → Projector → LLM

Phase 5 — Multimodal Pretraining

Large-scale Image-Text Learning

Phase 6 — Multimodal SFT

Captioning + VQA + Grounding Instructions

Phase 7 — Domain Fine-tuning / LoRA

Specialized Data, Optional PEFT

Phase 8 — DPO / Preference Optimization

Optional Multimodal Alignment

Phase 9 — Evaluation & Model Selection

Quality、Grounding Accuracy、Safety、Regression

Phase 10 — Multimodal Inference

Image + Prompt → Generated Answer / Boxes

Phase 11–13 — Artifacts / Deployment / Continuous Improvement

Checkpoint、Versioning、Monitoring、Retraining

這是一條教學及研究用途的完整參考路線，不是所有大型 VLM 都必須遵守的唯一流程。

例如 LLaVA 的經典訓練流程，主要是把已預訓練的 Vision Encoder 和 LLM 接起來，完成 Feature Alignment，再做 Visual Instruction Tuning；不需要從零重新訓練兩個 Backbone。Google PaliGemma 也使用預訓練的 SigLIP 和 Gemma 初始化。

![](https://www.google.com/s2/favicons?domain=https://github.com&sz=32)

GitHub

+1

因此我會讓你自己的 Repo 同時支援以下兩種模式。

||Mode A：Full From Scratch|Mode B：Pretrained Backbones|
|---|---|---|
|Vision Encoder|隨機初始化|載入 SigLIP|
|LLM|隨機初始化|載入 SmolLM2|
|Projector|隨機初始化|隨機初始化|
|Image Pretraining|需要自行完成或改用聯合訓練|可跳過|
|Text Pretraining|需要自行完成或改用聯合訓練|可跳過|
|Vision-Language Alignment|需要|需要|
|Multimodal SFT|需要|需要|
|LoRA、DPO|可選|可選|
|適合目的|完整理解每一項技術|訓練可實際使用的 VLM|

我的建議：這兩個模式使用同一份 Model Architecture 和 Dataset API，只改 Config 中的權重初始化方式。

這樣才能真正比較「隨機初始化的模型」和「預訓練模型」經過相同 Multimodal Training 後的差異。

## 三、VLM 每個 Phase 的技術內容與實際工作

## Phase 0 — Model Architecture 與 Random Initialization

這個階段與 LLM 最大的差別，是你不只需要建立一個 Transformer，而是需要建立能同時處理 Image 和 Text 的 Model Architecture。

對第一個 Repo，我建議選擇 nanoVLM / LLaVA 這類 Vision Encoder + Projector + Decoder-only LLM 的架構。

### 0.1 模型的三個核心元件

|元件|主要功能|從零開始需要實作|
|---|---|---|
|Vision Encoder|Image → Visual Features|Patch Embedding、Positional Embeddings、Self-Attention、MLP、LayerNorm|
|Multimodal Projector|Vision Dimension → LLM Dimension|Linear / MLP Projection|
|Language Decoder|根據 Image 與 Prompt 預測 Answer|Token Embedding、Causal Attention、RoPE、FFN、Output Head|

例如：

輸入一張 224 × 224 的 RGB 影像，使用 Patch Size 16：

\[ N_{\text{patches}}=(224/16)^2=196 \]

影像經過 Vision Encoder：

\[ V=f_{\text{vision}}(I) \]

假設：

\[ V\in\mathbb{R}^{196\times768} \]

而你的 Language Decoder 的 Hidden Dimension 是 512，那 Projector 就需要進行：

\[ Z_v=\operatorname{MLP}(V) \]

\[ Z_v\in\mathbb{R}^{196\times512} \]

之後將 Visual Token Embeddings 和 Text Token Embeddings 合併：

\[ H_0=[Z_v;E_{\text{text}}] \]

交給 Language Decoder 預測文字。

注意：不同 VLM 可能使用 Cross-Attention、Perceiver Resampler、Q-Former 等不同 Fusion Architecture。上面的方式是最簡單、最適合第一個 Repo 的 Prefix-Fusion 實作。

### 0.2 Random Initialization 應該理解什麼？

從零建立時，你需要比較：

|Parameter|常見初始化方式|
|---|---|
|Linear Weights|Xavier / Kaiming 或模型指定初始化|
|Attention Q、K、V|隨機初始化|
|Embedding Matrix|隨機初始化|
|LayerNorm Scale|通常初始化為 1|
|LayerNorm Bias|通常初始化為 0|
|Projector Weights|隨機初始化|
|Pretrained Weights|不載入（Scratch Mode）|

具體初始化規則應與模型架構相容，而不是把所有 Linear Layer 無差別套用同一方式。

這時應檢查 Model Forward 能否執行、Output Dimension 是否正確、Gradient 是否正常傳遞，以及在少量資料上能否 Overfit。

完成標準： 隨機初始化的 Vision Encoder、Projector、Language Decoder 能接受 Image + Text，執行 Forward/Backward，但尚未具有實用的視覺理解能力。

## Phase 1 — Raw Dataset Preparation

這是 VLM 比 LLM 多出大量工作的一個階段。

原本 LLM 的核心資料為：

```
Raw Text
   ↓
Cleaning
   ↓
Tokenizer
   ↓
Input IDs
```

VLM 則需要兩條 Data Pipeline：

Image Pipeline

Raw Image

Decode / Validate

Resize / Crop / Normalize

Pixel Tensor

Text Pipeline

Raw Text

Cleaning / Normalization

Tokenizer / Special Tokens

Token IDs

兩條 Pipeline 最後由 Multimodal Data Collator 組合。

這裡必須正確理解：一般 ViT 不需要像 LLM 一樣使用文字 Tokenizer 來處理 Image。 它通常將 Pixel Tensor 經 Patch Embedding 轉換成 Visual Tokens。

若要從零建立文字端的模型，也應該明確決定 Text Tokenizer 是自行從 Raw Corpus 訓練，還是採用已存在的 Tokenizer。

### 需要準備哪些資料？

|Dataset|用途|標註內容|
|---|---|---|
|Raw Images|Vision Pretraining|Image|
|Raw Text Corpus|LLM Pretraining|Text|
|Image-Text Pairs|Multimodal Pretraining|Image + Caption|
|Caption Dataset|Instruction Tuning|Image + Caption|
|VQA Dataset|Instruction Tuning|Image + Question + Answer|
|Grounding Dataset|Spatial Training|Image + Phrase + Box|
|Preference Dataset|DPO|Image + Prompt + Chosen / Rejected|

這些 Dataset 可以存在同一個 Repo 的資料索引中，但不應將不同任務的原始標註強制轉成相同的簡單文字。

我會建議保存三層資料：

- Raw Dataset：原始 Image、Text、Annotations。
    
- Canonical Dataset：標準化後、模型無關的 JSONL/Parquet。
    
- Model-ready Dataset：由 Adapter 在訓練時產生 Input IDs、Pixel Values、Labels。
    

另外，需要在 Dataset Preparation 就固定 Training / Validation / Test Split。

如果同一支手錶拍了 40 張不同角度的照片，最好以 `watch_id` 分組切分，而不只是隨機以 `image_id` 切分，避免模型在 Training 見過相同物件，卻在 Test 被當成未見過的樣本。

## Phase 2 — Vision Encoder Pretraining

這是標準文字 LLM Lifecycle 中沒有的階段。

如果從 Random Weights 開始，你的 Vision Encoder 原本並不知道 Edge、Shape、Texture、Object 或 Spatial Features 是什麼。

可以考慮下列訓練策略。

|方法|訓練概念|是否需要文字|
|---|---|---|
|Supervised Image Classification|Image → Class Label|不需要|
|MAE（Masked Autoencoder）|遮住部分 Image Patches，學習重建|不需要|
|DINO-style Self-distillation|讓不同影像視角的 Representation 保持一致|不需要|
|CLIP / SigLIP-style Learning|學習 Image-Text Representation Alignment|需要|

這些方法不是同一個 Loss，也不一定適合混在同一次實驗。

我建議第一版 Full Scratch Mode 可以採用小型 ViT + MAE 或簡單 Classification Pretraining。

你需要觀察的是 Vision Encoder 是否逐步學會提取有用的影像特徵，而不只是記住 Training Images。

也可以省略獨立的 Image-only Pretraining，直接由隨機初始化的 Vision Encoder 參與 Image-to-Text Joint Training。這仍然是 From Scratch，但通常需要更多資料與計算成本。

## Phase 3 — Language Decoder Pretraining

這個階段與你列出的 LLM Lifecycle 基本相同。

使用 Raw Text Corpus 訓練 Decoder-only Transformer：

\[ \mathcal{L}_{text} =-\sum_t\log P(x_t\mid x_1,\ldots,x_{t-1}) \]

你需要觀察 Training Loss、Validation Loss、Gradient Norm、Learning Rate、Optimizer State、Checkpoint，以及 Generated Samples。

這個階段完成後，Language Decoder 應具有基本文字生成能力，但尚不理解影像。

這個階段不需要 Image Dataset。

若使用預訓練 SmolLM2 / Gemma Backbone，則可以直接跳過這一步。但這也代表整體系統並不是所有權重都從零訓練。

## Phase 4 — Vision-Language Alignment

這是 VLM 特別重要的一步。

即使 Vision Encoder 和 LLM 都已經預訓練，它們的 Hidden States 也不會自動具有相同語義。

因此要將兩者連接起來。

### 建議訓練設定

|元件|初始訓練方式|
|---|---|
|Vision Encoder|Frozen|
|Projector|Trainable|
|Language Decoder|Frozen|
|Dataset|Image-Text Pairs|
|Loss|Caption Token Cross-entropy|

例如：

```
Image:
  A blue wristwatch

Prompt:
  Describe this image.

Target:
  A wristwatch with a blue dial.
```

Projector 的參數會透過 Language Decoder 的 Loss 接收到 Gradient，使 Image Embeddings 逐漸對應到 Language Decoder 能使用的 Representation。

需要區分：

- Embedding Dimension Alignment：Projection Layer 讓維度相容。
    
- Semantic Alignment：訓練讓影像特徵與文字語義建立關係。
    

單純將 768 維轉成 512 維，並不代表模型已理解這些影像。

這種訓練階段與 LLaVA 的 Feature Alignment 相近。nanoVLM 的官方訓練程式也會為 Projector 和 Backbones 設定不同的 Learning Rates。

![](https://www.google.com/s2/favicons?domain=https://github.com&sz=32)

GitHub

+1

## Phase 5 — Multimodal Pretraining

Phase 4 比較像先建立 Image-Text 的連接能力。

Phase 5 則允許更多元件參與訓練，提升真正的跨模態 Representation Learning。

例如：

|元件|訓練策略|
|---|---|
|Vision Encoder|Frozen、Partial Unfreeze 或 Trainable|
|Projector|Trainable|
|LLM Decoder|Trainable 或部分更新|
|Dataset|大規模 Image-Text Pairs|
|Objective|Multimodal Next-token Prediction|

生成式訓練的核心目標是：

\[ \mathcal{L}_{mm} =-\sum_{t\in A}\log P(y_t\mid I,q,y_{<t}) \]

其中 \(I\) 是 Image，\(q\) 是 Text Prompt，\(A\) 是要計算 Loss 的答案 Tokens。

模型此時要學會利用 Image Information 預測文字，而不是只靠 Language Prior 猜答案。

例如兩張圖片：

![Saat Kadranı Nedir?](https://images.openai.com/static-rsc-4/71DFt_AIY2GlpNv7Mn1CdRWmcCu51k1KXVPZSh3ssioJkRMhBehvWPvHgTbGP8osmlwVeCr6eS_usWpaOLwdnPN9wpnFMotiCpflY2HRMB6-uMDvYEkEh_Hc-Z8TqAs0sw7Ck2RB1fYvByrQ_hs-_S8WSEgniC1uYm0tL7G12aw?purpose=inline)

Prompt: What color is the dial?

Target: Blue

![Grossiste en ligne | Mode, Accessoires, Bazar – Grossiste-Pro](https://images.openai.com/static-rsc-4/UVS5hRGP4Lo5KoChN2_lgWtxQoaPHA3pGN08U9UWq-VYSknWKmWkAibiSLL6dA2XAmQqxWMoQvVMbGcQmtiAS5FZpjA0k5ZmW0saMJjrHdkQWAmHOyxHaAlF_t0EiwkWAl6yf0OQOtAF_e_0ERqwMqVE4i6Cr5qqgUHpOkTFo_Y?purpose=inline)

Prompt: What color is the dial?

Target: Black

如果模型忽略影像，只根據 Prompt 回答，很難對這兩筆樣本都正確預測。

實際評估時可以用 Image Swap Tests：保留相同問題但替換影像，確認答案是否隨影像內容正確改變。

要注意：Phase 4 和 Phase 5 並不是所有 VLM 都嚴格分開的兩個 Stage。對簡單模型，可以合併成一次 Multimodal Training。對較大的模型，分階段通常有助於穩定訓練及保護預訓練 Backbone。

## Phase 6 — Multimodal Supervised Fine-tuning（SFT）

這個階段讓模型不只會根據圖片產生文字，而能理解並遵循不同類型的指令。

例如：

|Task|User Input|Assistant Target|
|---|---|---|
|Captioning|Describe this image.|A wristwatch with a blue dial.|
|VQA|What color is the dial?|Blue.|
|Grounding|Locate the date window.|`<box>...</box>`|
|OCR（可擴充）|Read the text on the dial.|SUBMARINER|
|Counting（可擴充）|How many hands are visible?|3|

這裡 VLM 和 LLM 的 SFT Objective 可以使用相同的 Cross-entropy Loss，但 Input Tensor 不同：

```
LLM SFT:
  input_ids
  attention_mask
  labels

VLM SFT:
  pixel_values
  input_ids
  attention_mask
  labels
  image_metadata
```

具體欄位名稱依模型而異，部分 VLM 還需要 image grid 或 image position metadata。

### Grounding 的額外工作

假設你要模型輸出：

```
<box>0.40,0.25,0.55,0.52</box>
```

你必須確保 Ground Truth、Token Serialization、Parsing 和 Bounding Box Metrics 都使用一致的座標定義。

Grounding 也可以使用專用的 `<loc0000>` 至 `<loc1023>` 類型的位置 Tokens，PaliGemma 就有這類設計。

![](https://www.google.com/s2/favicons?domain=https://ai.google.dev&sz=32)

Google AI for Developers

因此，相較於 LLM SFT，VLM SFT 增加的工程項目主要是 Image Processor、Multimodal Collator、Task-specific Serializer、Spatial Label Validation 和 Image-conditioned Evaluation。

目前 Hugging Face TRL 的 `SFTTrainer` 已支援 VLM 的 Image Dataset。官方也特別提醒不要在未驗證的情況下截斷 Input Sequence，否則可能切掉 Image Tokens。

![](https://www.google.com/s2/favicons?domain=https://github.com&sz=32)

GitHub

## Phase 7 — LoRA / Domain Fine-tuning

這裡要特別更正一個常見理解：

LoRA 不是在 SFT 之後必須另做的訓練階段。

你可以直接使用 LoRA 執行 Phase 6 的 SFT，也可以先 Full SFT，再針對特殊 Domain 使用 LoRA。

VLM 比 LLM 多了一個重要決策：LoRA 應套在哪些元件？

|設定|Vision Encoder|Projector|Language Decoder|
|---|---|---|---|
|Projector-only|Frozen|Trainable|Frozen|
|LLM LoRA|Frozen|可訓練|LoRA|
|Vision + LLM LoRA|LoRA / Partial|可訓練|LoRA|
|Full Fine-tuning|Trainable|Trainable|Trainable|

例如你要將通用 VLM 微調成手錶影像專用模型：

- Captioning：描述 Dial、Hands、Bezel。
    
- VQA：判斷錶盤顏色、日期窗位置、文字內容。
    
- Grounding：定位 Hour Hand、Minute Hand、Hour Markers。
    

如果只修改 Language Decoder，模型可能學會新的手錶相關詞彙，但原本 Vision Encoder 若無法辨識非常微小的零件特徵，效果仍可能受到限制。

因此當 Domain 差異很大時，應評估是否需要 Partial Vision Unfreeze、High-resolution Crops 或 Vision-side LoRA。

## Phase 8 — DPO / Preference Alignment

LLM DPO 的資料是：

```
Prompt
Chosen Answer
Rejected Answer
```

VLM DPO 則必須包含同一張 Image：

```
Image + Prompt
       ├── Chosen Answer
       └── Rejected Answer
```

例如模型看到藍色錶盤：

```

{
  "image": "watch001.jpg",
  "prompt": "What color is the dial?",
  "chosen": "The watch has a blue dial.",
  "rejected": "The watch has a red dial."
}
```

這是示意性 Preference Data。

DPO 的重點不是再學習普通的 Question/Answer，而是讓模型更偏好符合圖像、指令及特定品質標準的回答。

在 VLM 中可以用來優化：

|問題|偏好學習目標|
|---|---|
|Visual Hallucination|不描述圖中不存在的物件|
|Ungrounded Answers|優先回答有影像依據的內容|
|Unsupported Certainty|無法判斷時表達不確定|
|Instruction Following|按照所需格式回答|
|Detailed Captioning|避免不必要的臆測|

不過 DPO 不能保證改善所有視覺任務，特別是不能假設它一定能提高 Bounding Box IoU。

對精確定位問題，使用有監督的 Grounding Loss 和 Spatial Evaluation 通常應優先於 Preference Optimization。

Hugging Face TRL 的 `DPOTrainer` 目前也支援 VLM 的圖片輸入與 Preference Dataset。

![](https://www.google.com/s2/favicons?domain=https://github.com&sz=32)

GitHub

## Phase 9 — Multimodal Evaluation

原本 LLM 可能重點評估 Perplexity、Instruction Following、Reasoning Accuracy、Hallucination。

VLM 還必須評估影像理解、物件定位，以及回答是否真正受到影像內容支持。

我建議你在 Repo 中分開建立下列 Evaluation Pipeline。

|Evaluation|Metric|要驗證什麼|
|---|---|---|
|Language Modeling|Validation Loss、Perplexity|LLM 是否學會語言分布|
|Image Representation|Linear Probe / k-NN Accuracy|Vision Encoder 特徵是否有用|
|Image Captioning|CIDEr、SPICE、BLEU|描述是否符合影像|
|VQA|Accuracy、Exact Match|是否正確回答問題|
|Visual Grounding|IoU、Acc@0.5、Center Error|Bounding Box 是否正確|
|Hallucination|Object Hallucination Rate、人工審查|是否虛構不存在的影像內容|
|Robustness|Crop、Blur、Lighting Shift|影像變化是否影響模型|
|Efficiency|VRAM、Latency、Throughput|是否能有效推理|

除了 Evaluation Score，你還應該做跨階段比較。

例如：

|Checkpoint|Captioning CIDEr|VQA Accuracy|Grounding Acc@0.5|
|---|---|---|---|
|Multimodal Pretraining|45|31%|12%|
|Multimodal SFT|83|58%|47%|
|Domain LoRA|91|65%|61%|
|DPO|92|67%|60%|

以上全部是說明如何比較的假設數值，不是 nanoVLM、PaliGemma 或任何實際模型的 Benchmark。

重點是觀察某個 Training Stage 是否真的帶來提升。

例如 DPO 後 VQA 變好、Grounding 卻下降，就應調查偏好資料是否讓模型更重視文字回答，而犧牲了空間定位精度。

## Phase 10 — Multimodal Inference

LLM 推理的主要流程是：

```
Text Prompt
   ↓
Tokenizer
   ↓
LLM Prefill
   ↓
Autoregressive Decoding
   ↓
Text Output
```

VLM 推理則需要：

```
Image ──> Image Processor ──> Vision Encoder
                                      │
                                      ▼
                                  Projector
                                      │
Text ──> Tokenizer ────────────────> Fusion
                                      │
                                      ▼
                                 LLM Prefill
                                      │
                                      ▼
                             Autoregressive Decode
                                      │
                                      ▼
                               Generated Tokens
                                      │
                       ┌──────────────┴──────────────┐
                       ▼                             ▼
                  Text Output                 Spatial Parser
                                                     │
                                                     ▼
                                                Bounding Box
```

### Temperature、Top-p、KV Cache 還需要嗎？

全部仍然有用，而且與 LLM 基本相同。

|參數|VLM 的用途|
|---|---|
|Temperature|控制生成答案的隨機性|
|Top-p|控制 Nucleus Sampling|
|Max New Tokens|限制輸出長度|
|KV Cache|減少 Autoregressive Decoding 的重複 Attention 計算|
|Beam Search|某些 Captioning 任務可比較|
|Image Resolution|影響細節、Visual Token 數量及成本|
|Image Cropping|讓模型聚焦重要 Region|
|Batch Size|控制吞吐量與 GPU Memory|

需要特別理解 KV Cache 的作用：

它主要快取 Transformer Decoder 過去 Tokens 的 K/V，而不是取代 Vision Encoder 的 Image Processing。

如果影像在 Prefill 時已被轉換成 Visual Tokens，之後生成文字可以利用相關 KV Cache，避免每個新 Token 都重新執行整段 Prompt 的 Decoder Attention。

對 Grounding，我會建議優先使用確定性或低隨機性的 Decoding，因為輸出座標必須能解析成合法的空間結構。

## Phase 11 — Checkpoint、Artifact 與 Experiment Versioning

你列出的 LLM 最後一步「保存所有 Dataset、Model Weights、Tokenizer、Configuration、Metrics」在 VLM 更加重要。

原因是 VLM 不只有 LLM Model Weights，還涉及 Image Processor 和 Multimodal Projector。

例如，以下資訊必須與 Checkpoint 相容：

|Artifact|為什麼必須保存|
|---|---|
|Vision Encoder Weights|決定 Image Features|
|Language Decoder Weights|決定 Text Generation|
|Projector Weights|決定兩種模態如何連接|
|Text Tokenizer|確保 Token IDs 一致|
|Image Processor Config|確保 Resize、Normalization 正確|
|Spatial Tokenizer Config|確保 Bounding Boxes 正確解碼|
|Model Configuration|Hidden Dimension、Heads、Layers|
|Training Configuration|LR、Optimizer、Batch Size|
|Optimizer / Scheduler State|支援真正恢復訓練|
|RNG / Sampler State|提高中斷續訓的可重現性|
|Dataset Manifest|記錄訓練資料版本與 Split|
|Metrics / Evaluation Results|比較不同 Checkpoints|

我建議將每個訓練階段保存成獨立的 Versioned Checkpoint。

例如：

```
checkpoints/
│
├── 00_random_init/
│
├── 01_vision_pretrained/
│
├── 02_llm_pretrained/
│
├── 03_vlm_aligned/
│
├── 04_multimodal_pretrained/
│
├── 05_multimodal_sft/
│
├── 06_domain_lora/
│
└── 07_dpo_aligned/
```

每一個 Checkpoint 都應能重新載入、執行 Inference、獨立評估，而不依賴上一個階段仍在執行。

另外，LoRA Adapter 不能只假設保存一份 Adapter Weights 就足夠，還要明確記錄它依賴的 Base Model 版本。

## Phase 12 — Deployment

如果你希望 Repo 不只是一個 Training Demo，而是一個完整 AI Engineering 專案，就應加入 Deployment。

對最小系統而言，不需要一開始就建立 Kubernetes 或大量雲端服務。

我會先建立 FastAPI Inference Service：

```
POST /v1/inference

Input:
  image
  task
  prompt
  generation_config

Output:
  answer
  boxes
  model_version
  latency_ms
```

你需要考慮的工程問題包括：

|項目|要處理的問題|
|---|---|
|Image Upload|大小、格式、安全驗證|
|Model Loading|權重版本、GPU Placement|
|Batch Inference|不同圖片大小和序列長度|
|GPU Memory|Visual Tokens + KV Cache|
|Timeout|大圖片或過長生成|
|Monitoring|Quality、P95 Latency、Errors|
|Rollback|新模型效果差時恢復舊版本|

這些與 LLM Serving 有很多共通點，但 VLM 還必須管理影像解析、高解析度輸入和 Spatial Postprocessing。

## Phase 13 — Continuous Improvement / Retraining

最後還需要建立模型改進循環：

```
Production Inference
         ↓
Collect Failure Cases
         ↓
Human Annotation / Review
         ↓
Dataset Versioning
         ↓
Training / Fine-tuning
         ↓
Evaluation
         ↓
Compare Against Baseline
         ↓
Deployment Approval
         ↓
New Model Version
```

例如你發現模型對正常光線下的錶盤可以正確回答 VQA，但在強烈反光的圖片上經常失敗，就可以建立專門的 Hard Example Dataset。

接著比較 Data Augmentation、Vision-side Fine-tuning、High-resolution Image Processing 是否能改善結果。

這才算是完整的 VLM Engineering Lifecycle，而不是只把模型訓練完並保存一份 `.safetensors`。

# 四、如果要建立 GitHub Repo，完整的程式碼結構應如何設計？

我會將上述 14 個 Phase 合併成四個主要系統：Model、Data、Training、Evaluation/Inference。這樣不會因為階段多，就變成十四套不同的訓練程式。

```

simple-vlm/
│
├── configs/
│   ├── model/
│   │   ├── tiny_vlm.yaml
│   │   └── pretrained_vlm.yaml
│   ├── training/
│   │   ├── vision_pretrain.yaml
│   │   ├── llm_pretrain.yaml
│   │   ├── alignment.yaml
│   │   ├── multimodal_pretrain.yaml
│   │   ├── sft.yaml
│   │   ├── lora.yaml
│   │   └── dpo.yaml
│   └── inference.yaml
│
├── models/
│   ├── vision_encoder.py
│   ├── language_decoder.py
│   ├── projector.py
│   └── vlm.py
│
├── tokenization/
│   ├── train_tokenizer.py
│   └── spatial_tokens.py
│
├── data/
│   ├── raw/
│   │   ├── images/
│   │   └── text/
│   ├── annotations/
│   │   ├── image_text.jsonl
│   │   ├── caption.jsonl
│   │   ├── vqa.jsonl
│   │   ├── grounding.jsonl
│   │   └── preference.jsonl
│   ├── processed/
│   └── manifests/
│
├── datasets/
│   ├── image_dataset.py
│   ├── text_dataset.py
│   ├── multimodal_dataset.py
│   ├── task_sampler.py
│   ├── adapters.py
│   └── collator.py
│
├── training/
│   ├── trainer.py
│   ├── optimizer.py
│   ├── scheduler.py
│   ├── losses.py
│   ├── checkpoint.py
│   └── stages/
│       ├── vision_pretrain.py
│       ├── llm_pretrain.py
│       ├── alignment.py
│       ├── multimodal_pretrain.py
│       ├── sft.py
│       ├── lora.py
│       └── dpo.py
│
├── evaluation/
│   ├── caption.py
│   ├── vqa.py
│   ├── grounding.py
│   ├── hallucination.py
│   └── compare_checkpoints.py
│
├── inference/
│   ├── generate.py
│   ├── image_processor.py
│   ├── grounding_parser.py
│   └── visualization.py
│
├── serving/
│   └── api.py
│
├── artifacts/
│   ├── checkpoints/
│   ├── metrics/
│   └── runs/
│
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── infer.py
│
├── tests/
│   ├── test_model_shapes.py
│   ├── test_dataset.py
│   ├── test_gradients.py
│   ├── test_training_resume.py
│   └── test_inference.py
│
├── requirements.txt
└── README.md
```

這是我推薦你建立的新 Repo 架構，而不是 nanoVLM 官方原始目錄。

## 五、應該讓每個 Phase 都有一個獨立 Training Command

例如以下是建議設計的 CLI，屬於規格示範，不是目前已有的可直接執行程式。

```

# 0. Initialize model
python scripts/train.py --stage init --config configs/model/tiny_vlm.yaml

# 1. Prepare dataset
python scripts/prepare_data.py

# 2. Vision pretraining
python scripts/train.py --stage vision_pretrain

# 3. Language pretraining
python scripts/train.py --stage llm_pretrain

# 4. Vision-Language alignment
python scripts/train.py --stage alignment

# 5. Multimodal pretraining
python scripts/train.py --stage multimodal_pretrain

# 6. Multimodal SFT
python scripts/train.py --stage sft

# 7. LoRA domain fine-tuning
python scripts/train.py --stage lora

# 8. DPO preference optimization
python scripts/train.py --stage dpo

# 9. Evaluate all checkpoints
python scripts/evaluate.py --compare-all

# 10. Inference
python scripts/infer.py --task vqa
```

最關鍵的是：每個 Stage 都應能明確指定輸入 Checkpoint，並保存自己的輸出 Checkpoint。

如此你才能測試「跳過 Vision Pretraining」、「跳過 DPO」、「改用 LoRA」是否會產生不同結果。

## 六、最建議的實作順序

如果你的目的是完整理解每個環節，而不只是訓練一個能回答問題的 VLM，我不建議一開始就實作全部 14 個 Phase。

應該分成三個 Development Milestones。

Milestone A — 完全 From Scratch 的 Tiny VLM

實作 Random Initialization、Text Tokenizer、ViT、Language Decoder、Projector、Pretraining、Multimodal Training 和 Inference。

此時使用少量合成影像、簡單圖形與短句，例如辨識紅色正方形、藍色圓形。

目標：理解 Forward、Backward、Gradient、Loss、Optimizer、Checkpoint 和 Vision-Language Interaction。不是追求高 Accuracy。

Milestone B — 使用 Pretrained Backbones 的完整 VLM

使用 nanoVLM 的架構，載入 SigLIP + SmolLM2，訓練 Projector，再做 Multimodal SFT。

加入 Captioning、VQA、Grounding 三種任務的 Unified Dataset。

目標：得到真正可以測試三種視覺語言任務的模型，並能與 Scratch Model 比較。

Milestone C — 完整 Post-training 與 Production Lifecycle

增加 LoRA、DPO、Checkpoint Comparison、Multimodal Evaluation、Inference API、Experiment Tracking 和 Continuous Improvement。

目標：不只能訓練模型，還能分析各階段效果，重現實驗及完成部署。

# 最後總結：VLM 與 LLM Lifecycle 的真正差異

完整的 VLM Lifecycle = LLM Lifecycle + Vision Learning + Multimodal Alignment + Visual Task Supervision + Multimodal Evaluation。

從 Training Objective 角度來看，VLM 很多階段仍然使用 LLM 已有的方法，例如 Causal Language Modeling、SFT、LoRA、DPO。

真正不同的是：

- Input Representation：從 Text Tokens 擴充到 Image Tokens + Text Tokens。
    
- Model Architecture：增加 Vision Encoder、Projector 或其他 Fusion Mechanism。
    
- Pretraining：除了語言能力，還需要學習視覺特徵與跨模態關係。
    
- Datasets：Image-Text Pairs、Question/Answer、Bounding Boxes、Preference Data。
    
- Training Decisions：需要決定 Vision Encoder、Projector、LLM 各自 Freeze 或 Unfreeze。
    
- Evaluation：除了文字正確性，還要測試 Image Grounding、定位精度、視覺 Hallucination 等。
    

因此對你希望建立的 GitHub 專案，我仍推薦 nanoVLM 作為 Model Architecture 的學習基礎，但新增完整的 Lifecycle Training Framework。

這會比直接以 PaliGemma 或 Florence-2 的既有 Fine-tuning Pipeline 作為起點，更適合你想親自理解從 Random Initialization 到 Model Deployment 每個階段的目標。尤其 Full Scratch Mode 和 Pretrained Mode 可以在同一個 Repo 中平行比較，使這個專案不只是 VLM Demo，而是一個真正完整、可重現的 Multimodal AI Research Repository。




#### 基於nanoVLM建立一套完整的VLM repo
```
我想在 Laptop 上基於huggingface/nanoVLM v0.1 release 建立一套完整的VLM repo在C:\LLM\VLM_base. 從零預訓練 → 後訓練 → 微調 → 推理. 利用已經訓練好的 Vision Encoder 和 Language Model，從零訓練連接兩者的 Projection Layer，再逐步聯合訓練, 保留 nanoVLM 約數百行核心模型與訓練邏輯的教學優勢，再逐步加入 PaliGemma 式多任務輸出表示

將這些Phase(Phase 1 — Raw Dataset Preparation, Phase 2 — Vision Encoder Pretraining, Phase 3 — Language Decoder Pretraining, Phase 4 — Vision-Language Alignment, Phase 5 — Multimodal Pretraining, Phase 6 — Multimodal Supervised Fine-tuning（SFT), Phase 7 — LoRA / Domain Fine-tuning, Phase 8 — DPO / Preference Alignment, Phase 9 — Multimodal Evaluation, Phase 10 — Multimodal Inference, Phase 11 — Checkpoint、Artifact 與 Experiment Versioning, Phase 12 — Deployment, Phase 13 — Continuous Improvement / Retraining) 合併成四個主要系統：Model、Data、Training、Evaluation/Inference。這樣不會因為階段多，就變成數套不同的訓練程式

推薦建立的新 Repo 架構:

simple-vlm/  
│  
├── configs/  
│ ├── model/  
│ │ ├── tiny_vlm.yaml  
│ │ └── pretrained_vlm.yaml  
│ ├── training/  
│ │ ├── vision_pretrain.yaml  
│ │ ├── llm_pretrain.yaml  
│ │ ├── alignment.yaml  
│ │ ├── multimodal_pretrain.yaml  
│ │ ├── sft.yaml  
│ │ ├── lora.yaml  
│ │ └── dpo.yaml  
│ └── inference.yaml  
│  
├── models/  
│ ├── vision_encoder.py  
│ ├── language_decoder.py  
│ ├── projector.py  
│ └── vlm.py  
│  
├── tokenization/  
│ ├── train_tokenizer.py  
│ └── spatial_tokens.py  
│  
├── data/  
│ ├── raw/  
│ │ ├── images/  
│ │ └── text/  
│ ├── annotations/  
│ │ ├── image_text.jsonl  
│ │ ├── caption.jsonl  
│ │ ├── vqa.jsonl  
│ │ ├── grounding.jsonl  
│ │ └── preference.jsonl  
│ ├── processed/  
│ └── manifests/  
│  
├── datasets/  
│ ├── image_dataset.py  
│ ├── text_dataset.py  
│ ├── multimodal_dataset.py  
│ ├── task_sampler.py  
│ ├── adapters.py  
│ └── collator.py  
│  
├── training/  
│ ├── trainer.py  
│ ├── optimizer.py  
│ ├── scheduler.py  
│ ├── losses.py  
│ ├── checkpoint.py  
│ └── stages/  
│ ├── vision_pretrain.py  
│ ├── llm_pretrain.py  
│ ├── alignment.py  
│ ├── multimodal_pretrain.py  
│ ├── sft.py  
│ ├── lora.py  
│ └── dpo.py  
│  
├── evaluation/  
│ ├── caption.py  
│ ├── vqa.py  
│ ├── grounding.py  
│ ├── hallucination.py  
│ └── compare_checkpoints.py  
│  
├── inference/  
│ ├── generate.py  
│ ├── image_processor.py  
│ ├── grounding_parser.py  
│ └── visualization.py  
│  
├── serving/  
│ └── api.py  
│  
├── artifacts/  
│ ├── checkpoints/  
│ ├── metrics/  
│ └── runs/  
│  
├── scripts/  
│ ├── prepare_data.py  
│ ├── train.py  
│ ├── evaluate.py  
│ └── infer.py  
│  
├── tests/  
│ ├── test_model_shapes.py  
│ ├── test_dataset.py  
│ ├── test_gradients.py  
│ ├── test_training_resume.py  
│ └── test_inference.py  
│  
├── requirements.txt  
└── README.md
```