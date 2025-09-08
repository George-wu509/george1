
|                                |     |
| ------------------------------ | --- |
| [[#### Qwen2-VL的功能輸出]]         |     |
| [[#### Qwen2-VL的輸入message]]    |     |
| [[#### Qwen2-VL重要其他functions]] |     |
| [[#### Qwen2-VL整体架构]]          |     |
|                                |     |
|                                |     |
Qwen2-VL [github](https://github.com/QwenLM/Qwen2.5-VL)


![[Pasted image 20250828102001.png]]


#### Qwen2-VL的功能輸出

| Cookbook                                                                                                                        | Description                                                                                                                                                                              | Open                                                                                                                                                                                                                                                                                                                                         |
| ------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Universal Recognition](https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/universal_recognition.ipynb)                   | Not only identify animals, plants, people, and scenic spots but also recognize various objects such as cars and merchandise.不僅能辨識動物、植物、人物、景點，還能辨識汽車、商品等各種物品                              | [![Colab](https://camo.githubusercontent.com/96889048f8a9014fdeba2a891f97150c6aac6e723f5190236b10215a97ed41f3/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/github/QwenLM/Qwen2.5-VL/blob/main/cookbooks/universal_recognition.ipynb) |
| [Powerful Document Parsing Capabilities](https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/document_parsing.ipynb)       | The parsing of documents has reached a higher level, including not only text but also layout position information and our Qwen HTML format.文件的解析達到了更高的層次，不僅包括文本，還包括佈局位置資訊和我們的Qwen HTML格式 | [![Colab](https://camo.githubusercontent.com/96889048f8a9014fdeba2a891f97150c6aac6e723f5190236b10215a97ed41f3/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/github/QwenLM/Qwen2.5-VL/blob/main/cookbooks/document_parsing.ipynb)      |
| [Precise Object Grounding Across Formats](https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/spatial_understanding.ipynb) | Using absolute position coordinates, it supports both boxes and points, allowing for diverse combinations of positioning and labeling tasks. 使用絕對位置座標，它同時支援框和點，允許定位和標記任務的多種組合            | [![Colab](https://camo.githubusercontent.com/96889048f8a9014fdeba2a891f97150c6aac6e723f5190236b10215a97ed41f3/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/github/QwenLM/Qwen2.5-VL/blob/main/cookbooks/spatial_understanding.ipynb) |
| [General OCR and Key Information Extraction](https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/ocr.ipynb)                | Stronger text recognition capabilities in natural scenes and multiple languages, supporting diverse key information extraction needs. 更強的自然場景、多語言文字辨識能力，支援多樣化的關鍵資訊擷取需求                   | [![Colab](https://camo.githubusercontent.com/96889048f8a9014fdeba2a891f97150c6aac6e723f5190236b10215a97ed41f3/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/github/QwenLM/Qwen2.5-VL/blob/main/cookbooks/ocr.ipynb)                   |
| [Video Understanding](https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/video_understanding.ipynb)                       | Better video OCR, long video understanding, and video grounding. 更好的視訊OCR、長視訊理解和視訊接地                                                                                                     | [![Colab](https://camo.githubusercontent.com/96889048f8a9014fdeba2a891f97150c6aac6e723f5190236b10215a97ed41f3/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/github/QwenLM/Qwen2.5-VL/blob/main/cookbooks/video_understanding.ipynb)   |
| [Mobile Agent](https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/mobile_agent.ipynb)                                     | Locate and think for mobile phone control. 定位並思考手機控制                                                                                                                                     | [![Colab](https://camo.githubusercontent.com/96889048f8a9014fdeba2a891f97150c6aac6e723f5190236b10215a97ed41f3/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/github/QwenLM/Qwen2.5-VL/blob/main/cookbooks/mobile_agent.ipynb)          |
| [Computer-Use Agent](https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/computer_use.ipynb)                               | Locate and think for controlling computers and Web. 定位並思考如何控制電腦和網路                                                                                                                       | [![Colab](https://camo.githubusercontent.com/96889048f8a9014fdeba2a891f97150c6aac6e723f5190236b10215a97ed41f3/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)](https://colab.research.google.com/github/QwenLM/Qwen2.5-VL/blob/main/cookbooks/computer_use.ipynb)          |

|                                                                                                                                                                |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [QA](https://colab.research.google.com/github/QwenLM/Qwen2.5-VL/blob/main/cookbooks/universal_recognition.ipynb#scrollTo=9596c50d-80a8-433f-b846-1fbf61145ccc) | 1. <br>from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor<br><br>checkpoint = "Qwen/Qwen2.5-VL-7B-Instruct"<br>model = Qwen2_5_VLForConditionalGeneration.from_pretrained(checkpoint)<br>processor = AutoProcessor.from_pretrained(checkpoint)<br><br>2.<br>prompt = "What kind of bird is this? Please give its name."<br>image = Image.open(image_path)<br><br>3.<br>text = processor.==apply_chat_template==(messages)<br>inputs = processor(text=[text], images=[image])<br><br>4.<br>output_ids = model.==generate==(**inputs)<br>generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]<br>output_text = processor.==batch_decode==(generated_ids)                                                                                                                                                                                                                                                                         |
|                                                                                                                                                                |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| [bbox](https://colab.research.google.com/github/QwenLM/Qwen2.5-VL/blob/main/cookbooks/spatial_understanding.ipynb)                                             | 1. <br>from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor<br><br>checkpoint = "Qwen/Qwen2.5-VL-7B-Instruct"<br>model = Qwen2_5_VLForConditionalGeneration.from_pretrained(checkpoint)<br>processor = AutoProcessor.from_pretrained(checkpoint)<br><br>2.<br>prompt = "Outline the position of each small cake and output all the coordinates"<br>messages = [{"role":"system","content":prompt},{"role":"user","content":[{"type": "text","text": prompt},{"image": img_url}]}]<br><br>3.<br>text = processor.==apply_chat_template==(messages)<br>inputs = processor(text=[text], images=[image])<br><br>4.<br>output_ids = model.==generate==(**inputs)<br>generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]<br>output_text = processor.==batch_decode==(generated_id)<br>bounding_boxes = output_text[0]                                <br><br>5.<br>plot_bounding_boxes(image,bounding_boxes,input_width,input_height) |
|                                                                                                                                                                |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| [video](https://colab.research.google.com/github/QwenLM/Qwen2.5-VL/blob/main/cookbooks/video_understanding.ipynb)                                              | 1. <br>from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor<br>from qwen_vl_utils import process_vision_info<br><br>checkpoint = "Qwen/Qwen2.5-VL-7B-Instruct"<br>model = Qwen2_5_VLForConditionalGeneration.from_pretrained(checkpoint)<br>processor = AutoProcessor.from_pretrained(checkpoint)<br><br>2.<br>prompt = "请用表格总结一下视频中的商品特点"<br>video_path, _, _ = get_video_frames(video_url)<br><br>3.<br>messages = [{"role":"system","content":""},{"role":"user","content":[{"type": "text","text": prompt},{"image": img_url}]}]<br>text = processor.==apply_chat_template==(messages)<br>inputs = processor(text=[text], images=[image])<br><br>4.<br>output_ids = model.==generate==(**inputs)<br>generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]<br>output_text = processor.==batch_decode==(generated_id)<br>bounding_boxes = output_text[0]                                                                        |
|                                                                                                                                                                |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
|                                                                                                                                                                |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |






#### Qwen2-VL的輸入message
```python

  messages = [
    {"role": "system", "content": sys_prompt},
    {"role": "user", "content": [
      {"type": "text", "text": prompt},
      {"image": img_url},]
    },
  ]


  messages = [
    {"role": "system", "content": [{"type":"text","text": sys_prompt}]},
    {"role": "user", "content": [
      {"type": "image_url","min_pixels": min_pixels,"max_pixels": max_pixels,"image_url":{"url": f"data:image/jpeg;base64,{base64_image}"}},
      {"type": "text", "text": prompt},]
    },
  ]


  messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
      {"type": "text", "text": prompt},
      {"video": video_path, "total_pixels": total_pixels, "min_pixels": min_pixels},]
    },
  ]
```

![[Pasted image 20250907154244.png]]
您提了幾個非常深入且關鍵的問題，這表示您已經對 Qwen2.5-VL 的基本使用方式有了很好的掌握，並且開始探索其靈活性和邊界。讓我來一一為您詳細解釋。

### 1. `content` 列表中元素的順序 (先圖後文 vs. 先文後圖)

**為什麼會不同？** 因為模型被設計成能夠理解自然、流暢的多模態對話，就像人類一樣。在日常對話中，您可以先展示一張圖片再提問（「看這張照片，<圖片>，這是哪裡？」），也可以先提問再附上圖片（「我想知道這是哪裡，照片如下：<圖片>」）。

**結論：順序是靈活的，並且會影響模型的理解方式。**

`content` 列表中的元素（文字、圖片、影片）最終會被處理成一個線性的序列輸入給模型。因此，它們的相對順序很重要，它構成了**上下文 (Context)**。

- **`[圖片]` -> `[文字]`**：模型會先「看到」圖片，然後再讀到您的問題。這適用於「看圖說話」或對圖片內容進行直接提問的場景。
- **`[文字]` -> `[圖片]`**：模型會先讀到您的文字，帶著這個問題或敘述的意圖去「看」圖片。這適用於「請幫我辨識這張圖片裡的物體」這類場景。

在大部分情況下，這兩種順序的結果可能很相似，但對於需要精細引導的複雜任務，合理安排順序可以讓模型更好地理解您的意圖。

### 2. 關於 `"type"` 關鍵字的使用

您的理解基本上是正確的，但有些細微差別。

**官方推薦的格式是明確的。** 根據 `README.md` 的範例，標準格式如下：

- **圖片**: `{"type": "image", "image": "..."}`
- **影片**: `{"type": "video", "video": "..."}`
- **文字**: `{"type": "text", "text": "..."}`

**但底層處理有兼容性。** 在 `qwen_vl_utils/vision_process.py` 檔案的 `extract_vision_info` 函式中，程式碼的判斷邏輯是： `if "image" in ele or "image_url" in ele or "video" in ele or ele.get("type","") in ("image", "image_url", "video")`

這段程式碼的意思是，只要字典裡**包含 `"image"`、`"image_url"` 或 `"video"` 這幾個鍵 (key)，或者 `"type"` 的值是這幾項之一**，系統就會將其識別為視覺資料。

**結論：**

- **`type` 不是絕對必要，但強烈建議使用。** 遵循官方範例，使用 `{"type": "...", "key": "..."}` 的格式是最標準、最不容易出錯的做法。
- 您看到的某些範例可能利用了底層的兼容性，省略了 `"type"`，但這不是推薦的最佳實踐。

### 3. 對話輪次 (`role`) 的靈活性

#### 一定要 `system` -> `user` 嗎？

不一定。`system` 角色是**選用 (Optional)** 的。

- **使用 `system` 的好處**：它可以為模型設定一個整體的行為框架或角色，對於需要穩定、特定風格輸出的應用很有幫助。
- **不使用 `system`**：如果省略，模型會使用其預設的、通用的助理行為模式。對於簡單的單次問答，完全可以省略 `system` 角色。

#### 只用一輪可以嗎？

**絕對可以。** 大部分的教學範例，例如最基本的圖文問答，都只包含一輪 `user` 輸入，這也是最常見的使用方式。

#### 順序可以倒反嗎？（先 `user` 後 `system`）

**不可以。** 模型的訓練遵循著固定的對話格式。`system` 指令必須在所有對話的最開始，作為初始設定。如果將順序顛倒，會破壞模型對對話結構的理解，導致無法預測的錯誤結果。

#### 可以用三輪或更多輪嗎？

**可以。** Qwen2.5-VL 支援多輪對話。您可以構建一個包含多輪 `user` 和 `assistant` 交流的 `messages` 列表，模型能夠理解並根據完整的上下文進行回應。例如 `README.md` 中的 `Add ids for Multiple Image Inputs` 範例就展示了一個多輪對話的結構。

### 4. 沒有視覺輸入時的行為

您的猜測完全正確。

**如果在 `user` 的 `content` 中沒有提供任何 `image` 或 `video`，Qwen2.5-VL 就會作為一個純粹的大型語言模型 (LLM) 來運作。**

它會忽略視覺處理部分，只處理文字輸入並生成文字回應。這使得 Qwen2.5-VL 非常通用，您可以用同一套程式碼和同一個模型來處理純文字任務和多模態任務，無需切換。`README.md` 的批次推論 (`Batch inference`) 範例中就包含了一個純文字的對話。

您提了幾個非常深入且關鍵的問題，這表示您已經對 Qwen2.5-VL 的基本使用方式有了很好的掌握，並且開始探索其靈活性和邊界。讓我來一一為您詳細解釋。

### 5. 種媒體來源的支援

#### 可以同時使用 `image` 和 `video` 嗎？

**可以。** 您可以在同一個 `user` 的 `content` 列表中放入多個圖片和影片。例如：

```Python
"content": [
    {"type": "text", "text": "比較一下這張圖和這段影片的風格。"},
    {"type": "image", "image": "path/to/image.jpg"},
    {"type": "video", "video": "path/to/video.mp4"}
]
```

模型會接收所有視覺輸入，並根據您的文字指令進行綜合性的回答。




#### Qwen2-VL重要其他functions

除了基本的使用流程，Qwen2.5-VL 的 `model` 和 `processor` 的確提供了許多進階功能，讓開發者可以更精細地控制模型的行為、效能和資源使用。

這些進階功能主要可以分為以下幾大類，大部分資訊都可以在專案的 `README.md` 和 `qwen-vl-utils` 的原始碼中找到：

### 1. 效能與效率優化

#### a. Flash Attention 2 加速

這是在載入模型時可以啟用的一項重要效能優化，特別是在處理多圖像和影片時，能顯著提升速度並節省記憶體。

- **功能**：使用更高效的注意力計算機制。
    
- **如何使用**：在 `from_pretrained` 中加入 `attn_implementation="flash_attention_2"` 參數。
    
    Python
    
    ```
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", # 啟用 Flash Attention 2
        device_map="auto"
    )
    ```
    

#### b. 批次推論 (Batch Inference)

您可以將多個不同的請求（無論是純文字還是多模態）組合成一個批次，一次性送入模型處理，這比逐一處理請求的效率要高得多。

- **功能**：同時處理多個獨立的對話。
    
- **如何使用**：將多個 `messages` 組成一個列表，並將處理後的 `texts` 列表傳遞給 `processor`。
    
    Python
    
    ```
    # 範例：一個多模態請求和一個純文字請求
    messages1 = [{"role": "user", "content": [{"type": "image", ...}, {"text": "..."}]}]
    messages2 = [{"role": "user", "content": "你是誰？"}]
    
    all_messages = [messages1, messages2]
    
    # Processor 會處理好多個 text 輸入
    texts = [processor.apply_chat_template(msg, ...) for msg in all_messages]
    inputs = processor(text=texts, images=...) # images 參數會自動對應
    
    # 一次生成所有回應
    outputs = model.generate(**inputs)
    ```
    

### 2. 精細化的視覺輸入控制

這些功能主要在 `messages` 的視覺資料字典中設定，由 `qwen-vl-utils` 套件在後端進行處理。

#### a. 自訂圖片解析度

您可以精確控制輸入圖片的尺寸，以在效能和效果之間取得平衡。`README.md` 提到了兩種方式。

- **功能**：避免使用預設解析度，改用自訂尺寸。
    
- **如何使用**：
    
    1. **直接指定尺寸**：
        
        Python
        
        ```
        "content": [{
            "type": "image",
            "image": "path/to/image.jpg",
            "resized_height": 280,
            "resized_width": 420
        }]
        ```
        
    2. **定義像素範圍**：讓 `qwen-vl-utils` 自動在保持長寬比的前提下縮放。
        
        Python
        
        ```
        "content": [{
            "type": "image",
            "image": "path/to/image.jpg",
            "min_pixels": 50176,  # 224 * 224
            "max_pixels": 100352 # 224 * 448
        }]
        ```
        

#### b. 自訂影片取樣與解析度

對於影片，控制參數更多，因為它涉及時間維度（幀率）和空間維度（解析度）。

- **功能**：控制影片的取樣幀數、幀率以及每一幀的解析度。
    
- **如何使用**：在 `video` 物件中加入參數。
    
    Python
    
    ```
    "content": [{
        "type": "video",
        "video": "path/to/video.mp4",
        "fps": 1.0,  # 設定每秒取 1 幀 (預設是 2.0)
        "min_frames": 4, # 至少取 4 幀
        "max_frames": 32, # 最多取 32 幀
        "total_pixels": 20480 * 28 * 28, # 限制整個影片的總像素量，防止OOM
        "max_pixels": 768 * 28 * 28 # 限制單幀的最大像素
    }]
    ```
    

### 3. 進階的生成與解碼控制

#### a. 串流式輸出 (Streaming)

在 `web_demo_mm.py` 中有展示，可以讓模型邊生成邊輸出，而不是等全部生成完畢再回傳，這對於互動式應用體驗至關重要。

- **功能**：即時回傳模型生成的文字流。
    
- **如何使用**：使用 `transformers` 的 `TextIteratorStreamer`。
    
    Python
    
    ```
    from transformers import TextIteratorStreamer
    from threading import Thread
    
    streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True)
    
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=512)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # 逐字或逐詞獲取輸出
    for new_text in streamer:
        print(new_text, end="")
    ```
    

#### b. 其他 `generate` 參數

`model.generate()` 函式本身繼承自 Hugging Face Transformers，支援非常多參數來控制生成邏輯，例如：

- `do_sample=True`: 是否進行取樣，設為 `False` 則使用貪婪搜索。
    
- `temperature`: 控制輸出的隨機性，值越小越確定。
    
- `top_k`, `top_p`: 控制取樣的範圍，用於改善生成品質。
    
- `repetition_penalty`: 對重複的詞彙進行懲罰。
    

### 4. 對話模板與上下文控制

#### a. 為多個視覺輸入添加 ID

當一輪對話中包含多張圖片或影片時，為了在提示中能明確指代它們，可以讓 `processor` 自動添加編號。

- **功能**：在對話模板中自動生成 "Picture 1:", "Picture 2:", "Video 1:" 等標籤。
    
- **如何使用**：在 `apply_chat_template` 中設定 `add_vision_id=True`。
    
    Python
    
    ```
    prompt_with_id = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        add_vision_id=True # 啟用此功能
    )
    ```
    

### 5. 模型微調 (Fine-tuning)

這是一個更進階的主題，`qwen-vl-finetune` 資料夾提供了完整的微調腳本，允許您在自己的資料集上進一步訓練模型。

- **功能**：讓模型適應特定領域的資料或任務。
    
- **可控參數** (`qwenvl/train/argument.py`):
    
    - `tune_mm_vision`: 是否微調視覺編碼器 (ViT) 的參數。
        
    - `tune_mm_mlp`: 是否微調連接視覺和語言模組的 MLP 投影層。
        
    - `tune_mm_llm`: 是否微調語言模型部分的參數。
        

透過組合這些進階功能，您可以更靈活、高效地將 Qwen2.5-VL 應用於各種複雜的場景中。






#### Qwen2-VL整体架构

[视觉语言模型](https://zhida.zhihu.com/search?content_id=250493249&content_type=Article&match_order=1&q=%E8%A7%86%E8%A7%89%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B&zhida_source=entity)（VLM）是人工智能领域的重要突破，它能够同时理解和处理图像与文本信息，实现类似人类的多模态认知能力。这类模型通过将强大的[视觉编码器](https://zhida.zhihu.com/search?content_id=250493249&content_type=Article&match_order=1&q=%E8%A7%86%E8%A7%89%E7%BC%96%E7%A0%81%E5%99%A8&zhida_source=entity)（如[CLIP](https://zhida.zhihu.com/search?content_id=250493249&content_type=Article&match_order=1&q=CLIP&zhida_source=entity)、[ViT](https://zhida.zhihu.com/search?content_id=250493249&content_type=Article&match_order=1&q=ViT&zhida_source=entity)）与大型语言模型（如[GPT](https://zhida.zhihu.com/search?content_id=250493249&content_type=Article&match_order=1&q=GPT&zhida_source=entity)、[LLaMA](https://zhida.zhihu.com/search?content_id=250493249&content_type=Article&match_order=1&q=LLaMA&zhida_source=entity)）相结合，创造出能够进行视觉理解和自然语言交互的智能系统。

典型的VLM通常包含三个核心组件：

- 视觉编码器：将图像转换为特征表示
- 语言模型：处理文本信息并生成响应
- [多模态融合模块](https://zhida.zhihu.com/search?content_id=250493249&content_type=Article&match_order=1&q=%E5%A4%9A%E6%A8%A1%E6%80%81%E8%9E%8D%E5%90%88%E6%A8%A1%E5%9D%97&zhida_source=entity)：实现视觉和语言特征的有效结合

2024年8月开源的Qwen2-VL模型在各种分辨率和长宽比的视觉理解任务中均达到领先水平，在[DocVQA](https://zhida.zhihu.com/search?content_id=250493249&content_type=Article&match_order=1&q=DocVQA&zhida_source=entity)（文档问答）、[InfoVQA](https://zhida.zhihu.com/search?content_id=250493249&content_type=Article&match_order=1&q=InfoVQA&zhida_source=entity)（信息问答）、[RealWorldQA](https://zhida.zhihu.com/search?content_id=250493249&content_type=Article&match_order=1&q=RealWorldQA&zhida_source=entity)（真实场景问答）、[MTVQA](https://zhida.zhihu.com/search?content_id=250493249&content_type=Article&match_order=1&q=MTVQA&zhida_source=entity)（多任务视觉问答）和[MathVista](https://zhida.zhihu.com/search?content_id=250493249&content_type=Article&match_order=1&q=MathVista&zhida_source=entity)（数学视觉理解）等多个基准测试中表现卓越。能够理解超过20分钟的长视频内容，显著提升了视频问答、对话和内容创作等任务的质量。其主要创新点如下：

**动态分辨率处理**

- 引入naive dynamic resolution技术
- 能够灵活处理不同分辨率的输入

**多模态位置编码**

- 创新性提出[多模态旋转位置编码](https://zhida.zhihu.com/search?content_id=250493249&content_type=Article&match_order=1&q=%E5%A4%9A%E6%A8%A1%E6%80%81%E6%97%8B%E8%BD%AC%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81&zhida_source=entity)（M-RoPE）
- 实现更有效的跨模态信息融合

**图像和视频的统一理解框架**

- 图像被处理为两个相同帧，保持与视频处理的一致性
- 使用3D tubes替代2D patches处理方式


|                                                                         |                    |
| ----------------------------------------------------------------------- | ------------------ |
| **1. Installing Qwen2-VL**                                              |                    |
| pip 安裝 github.com/ huggingface/transformers                             |                    |
| pip 安裝qwen-vl-utils                                                     |                    |
|                                                                         |                    |
| **2. Video set up**                                                     |                    |
| conda install -c conda-forge ffmpeg -y                                  |                    |
| pip install ffmpeg -q                                                   |                    |
| pip install gdown -q                                                    | download the video |
| !ffmpeg -i {video_path} -q:v 2 -start_number 0 {output_path}/'%05d.jpg' |                    |
|                                                                         |                    |
| **3. Qwen2-VL for Video Understanding**                                 |                    |
|                                                                         |                    |
|                                                                         |                    |
|                                                                         |                    |



![[Pasted image 20250828102105.png]]

Qwen2-VL模型架构打印下来如下，可以发现它主要由Qwen2VisionTransformerPretrainedModel（视觉编码器）和[Qwen2VLModel](https://zhida.zhihu.com/search?content_id=250493249&content_type=Article&match_order=1&q=Qwen2VLModel&zhida_source=entity)（语言模型）两部分组成，并没有明显的Connector部分，视觉编码向量没有经过太多的处理直接进入了语言模型。精致的Connector似乎正变得没那么重要，早在多模态模型诞生之初就有ViLT这样的将图片直接用线性层投影作为Transformer输入的架构，现在也有一些轻量的多模态模型用MLP对图片进行处理，直接删除了视觉编码器，可见多模态信息的融合fusion不一定需要太复杂的结构。




#### Qwen2-VL內部運作原理

深入理解 Qwen2.5-VL 的內部運作原理。我將分兩部分為您詳細解釋：首先是模型的整體架構，其次是以您提供的鳥類辨識範例，一步步追蹤資料在模型中的完整旅程。

### 第一部分：Qwen2.5-VL 模型架構詳解

Qwen2.5-VL (Vision-Language) 的核心是一個由三個主要部分組成的**端到端 (End-to-End)** 深度學習模型。您可以將它想像成一個團隊協作系統：一個「眼睛」，一個「大腦」，以及一個「翻譯官」來連接它們。

1. **視覺編碼器 (Vision Encoder) - 團隊的「眼睛」**
    
    - **組件**：這部分採用了**視覺 Transformer (Vision Transformer, ViT)** 架構。
        
    - **作用**：它的唯一職責是「看懂」輸入的圖像或影片。它會將一張圖片或影片的每一幀分解成許多小方塊 (Patches)，然後將這些方塊轉換成一系列富含資訊的數字向量 (Feature Vectors)。每一個向量都代表了圖像某個部分的視覺特徵。
        
    - **Qwen2.5-VL 的優化**：為了提升效率和性能，這個 ViT 經過了特別優化，例如使用了**視窗注意力 (Window Attention)** 來減少計算量，並採用了與語言模型相同的 **SwiGLU** 和 **RMSNorm** 結構，使其與「大腦」的內部語言更加協調。
        
2. **大型語言模型 (Large Language Model, LLM) - 團隊的「大腦」**
    
    - **組件**：這是基於 **Qwen2.5** 的大型語言模型。
        
    - **作用**：這是模型的核心，負責**思考、推理和生成文字**。它本身是一個極其強大的純文字處理模型，學習了海量的文本資料，懂得語法、語意、邏輯和世界知識。但如果沒有「眼睛」和「翻譯官」，它自己是看不懂圖片的。
        
3. **橋接模組 (Bridge Module / Merger) - 團隊的「翻譯官」**
    
    - **組件**：通常是一個或多個小型神經網路層，例如 **MLP (多層感知器)**。在程式碼中，它被稱為 `merger`。
        
    - **作用**：它的任務至關重要，是連接「眼睛」和「大腦」的橋樑。視覺編碼器輸出的視覺特徵向量，其「語言」和格式與語言模型所理解的文字特徵向量是不同的。橋接模組的作用就是將這些視覺特徵**「翻譯」或「投影」**到語言模型可以理解的空間中。經過這個模組處理後，視覺資訊就變成了一種特殊的「視覺詞彙」，可以無縫地插入到語言模型的文字序列中。
        

### 第二部分：從輸入到輸出的完整流程詳解

現在，讓我們以 `Universal Recognition` 的範例來走一遍完整的流程：

- **輸入圖像**: `unireco_bird_example.jpg` (一張翠鳥的圖片)
    
- **輸入文字**: `prompt = "What kind of bird is this? Please give its name in Chinese and English."`
    

這個過程可以分為 **前端處理 (Processor)**、**模型推論 (Model)** 和 **後端處理 (Processor)** 三大階段。

#### 階段一：前端處理 (Processor 執行)

這一步發生在您的電腦或客戶端，目的是將原始資料轉換為模型可以接受的格式。

1. **載入與預處理圖像**:
    
    - `qwen_vl_utils` 工具包會讀取 `unireco_bird_example.jpg` 這個檔案。
        
    - 圖像會被轉換為 RGB 格式，並透過 `smart_resize` 函式調整大小，使其尺寸能被 ViT 的小方塊大小 (例如 28x28) 整除。
        
    - 最後，圖像被轉換成一個標準化的數字矩陣（PyTorch Tensor），數值介於 0 和 1 之間。
        
2. **處理與符號化文字**:
    
    - `processor` 會調用內部的 Tokenizer (分詞器)。
        
    - 它會先將您的 `prompt` 根據對話模板 `apply_chat_template` 轉換成類似 `<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What kind of bird is this? ...<|im_end|>\n<|im_start|>assistant\n` 的格式。這裡的 `<|image_pad|>` 是一個特殊的**佔位符**，告訴模型這個位置將會插入視覺資訊。
        
    - 接著，Tokenizer 會將這個格式化的字串分割成一個個的詞元 (Token)，並將每個詞元轉換成一個唯一的數字 ID。例如，`"What"` -> `15496`, `"kind"` -> `3292`, `...`。
        
3. **整合多模態輸入**:
    
    - `processor` 會將預處理好的圖像 Tensor 和符號化後的文字 ID 序列整合在一起。它會將圖像 Tensor 與 `<|image_pad|>` 佔位符關聯起來，準備好一個統一的輸入包，發送給模型。
        

#### 階段二：模型推論 (Model 在 GPU 上執行)

這是最核心的計算部分，所有組件開始協同工作。

1. **視覺特徵提取 (眼睛工作)**:
    
    - 整合輸入包中的圖像 Tensor 被送入 **ViT (視覺編碼器)**。
        
    - ViT 將圖像分解、計算，最終輸出一系列代表這張翠鳥圖像的**視覺特徵向量**。
        
2. **特徵對齊 (翻譯官工作)**:
    
    - 這些視覺特徵向量被送入**橋接模組 (Merger)**。
        
    - 橋接模組將它們轉換成與語言模型詞彙空間維度相同的向量。現在，這張圖片在模型內部被表示為一組特殊的「視覺詞彙」嵌入 (Embeddings)。
        
3. **深度融合與理解 (大腦工作)**:
    
    - **LLM (大型語言模型)** 接收一個合併後的序列：一部分是您問題文字的詞元嵌入，另一部分是來自橋接模組的「視覺詞彙」嵌入。
        
    - 這個序列會流經 LLM 的多層 Transformer 網路。在每一層的**自注意力 (Self-Attention)** 機制中，文字詞元可以「關注」到視覺詞元，視覺詞元也可以「關注」到文字詞元。
        
        - 這一步是實現**圖文理解**的關鍵。例如，問題中的 `"bird"` 這個詞元會強烈地「關注」到圖像中代表鳥的視覺特徵上。
            
        - 模型會綜合分析：「使用者在問『鳥』(來自文字) + 這裡有一隻鳥的視覺特徵 (來自圖像)」，從而準確理解任務是要辨識這隻鳥。
            
4. **自迴歸生成 (大腦輸出)**:
    
    - 在完全理解了輸入後，LLM 開始生成答案，這個過程是一詞一詞進行的 (Autoregressive)。
        
    - **第一步**: 模型預測出最可能的第一個詞，可能是「這」。
        
    - **第二步**: 模型將「這」作為新的輸入，繼續預測下一個最可能的詞，可能是「是」。
        
    - **第三步**: 模型將「這是」作為輸入，預測出「一」。
        
    - ...這個過程不斷重複，模型會根據其知識庫中關於鳥類和語言的知識，逐步生成「這是一隻翠鳥。Its English name is Kingfisher.」，直到生成一個代表句子結束的 `[EOS]` 符號為止。
        

#### 階段三：後端處理 (Processor 執行)

1. **反符號化**:
    
    - 模型輸出的是一串數字 ID 序列。
        
    - `processor` 的 `batch_decode` 函式會調用 Tokenizer，根據詞彙表將這串 ID 翻譯回人類可讀的文字。
        
    - 例如 `[234, 345, 890, ...]` -> `"這是一隻翠鳥。..."`。
        

最終，您就得到了螢幕上看到的、準確且格式完整的答案。整個過程流暢地結合了視覺感知和語言智能，展現了 Qwen2.5-VL 強大的多模態能力。





















Reference:
多模态技术梳理：Qwen-VL系列 - 姜富春的文章 - 知乎
https://zhuanlan.zhihu.com/p/25267823390

多模态大模型学习笔记(一)--Qwen2.5-VL - HangYu的文章 - 知乎  
[https://zhuanlan.zhihu.com/p/1943676322443400076](https://www.google.com/url?q=https://zhuanlan.zhihu.com/p/1943676322443400076&sa=D&source=calendar&usd=2&usg=AOvVaw0Ucd_73hZzByXyjPYc1HUz)

【多模态大模型】Qwen2-VL解剖 - Plunck的文章 - 知乎
https://zhuanlan.zhihu.com/p/7352653203

Qwen2-VL源码解读：从准备一条样本到模型生成全流程图解 - 姜富春的文章 - 知乎
https://zhuanlan.zhihu.com/p/28205969434

多模态技术梳理：Qwen-VL系列 - 姜富春的文章 - 知乎
https://zhuanlan.zhihu.com/p/25267823390

Qwen2-VL：提升视觉语言模型对任意分辨率世界的感知能力 - AI专题精讲的文章 - 知乎
https://zhuanlan.zhihu.com/p/1928028373483000545

【多模态模型学习】qwen2-vl模型代码技术学习 - 威化饼的一隅的文章 - 知乎
https://zhuanlan.zhihu.com/p/19107424324

Qwen2-VL技术解析（一）-原生支持任意分辨率图像 - mingming的文章 - 知乎
https://zhuanlan.zhihu.com/p/718515978


