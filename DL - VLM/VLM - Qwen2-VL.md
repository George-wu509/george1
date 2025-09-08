
|                                |     |
| ------------------------------ | --- |
| [[#### Qwen2-VL的功能輸出]]         |     |
| [[#### Qwen2-VL的輸入message]]    |     |
| [[#### Qwen2-VL重要其他functions]] |     |
| [[#### Qwen2-VL整体架构]]          |     |
| [[#### Qwen2-VL內部模型推論]]        |     |
| [[#### QWen2.5-VL 架構圖解析]]      |     |
| [[#### Merger 的位置與工作流程]]       |     |
|                                |     |
|                                |     |
|                                |     |
|                                |     |
Qwen2-VL [github](https://github.com/QwenLM/Qwen2.5-VL)


![[Pasted image 20250828102001.png]]

![[Pasted image 20250828102105.png]]
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

1. **反符號化**
    - 模型輸出的是一串數字 ID 序列。
    - `processor` 的 `batch_decode` 函式會調用 Tokenizer，根據詞彙表將這串 ID 翻譯回人類可讀的文字。
    - 例如 `[234, 345, 890, ...]` -> `"這是一隻翠鳥。..."`。

最終，您就得到了螢幕上看到的、準確且格式完整的答案。整個過程流暢地結合了視覺感知和語言智能，展現了 Qwen2.5-VL 強大的多模態能力。





#### Qwen2-VL內部模型推論

```
請詳細中文解釋階段二：模型推論, 假設從輸入image經過ViT已經產生patch size * 1024(embedding dimension), 那接下來在Merger的特徵對齊的每個步驟是如何進行, 特徵對齊應該不只是轉換維度? 所以LLM (大型語言模型) 接收一個合併後的序列, 這個序列就是之前prompt經過對話模板轉換並由Tokenizer分割的Token序列(含有image_pad), 再轉換成數字id, 並把Merger的輸出插進去image_pad的位置. 然後進入到多層Encoder-decoder中. 所以Encoder跟decoder的詳細內部結構是?
```

已經觸及了模型內部運作的核心機制。您的猜測和提問都非常準確，現在讓我們來揭開這層神秘的面紗。

首先，我需要先澄清一個您提到的關鍵點，這會影響後續所有的解釋：

**Qwen2.5-VL 的大型語言模型部分是一個「Decoder-only (僅解碼器)」架構，而不是「Encoder-Decoder (編碼器-解碼器)」架構。**(也有講法是Qwen2.5-VL是**Encoder-Decoder混和**因為有Vision encoder負責處理影像, LLM則是Decoder-only生成)

這是一個非常重要的區別：

- **Encoder-Decoder 架構** (如傳統的翻譯模型 T5、BART)：通常有兩個獨立的部分。Encoder 負責將輸入序列（例如一句德文）完整讀取並壓縮成一個富含語意的上下文向量。然後 Decoder 再根據這個上下文向量，生成輸出序列（例如一句英文）。
    
- **Decoder-only 架構** (如 GPT 系列、Llama、Qwen)：只有一個部分，就是 Decoder。它將輸入和要生成的輸出視為一個**連續的序列**來處理。它在生成每一個新詞時，都可以回看（關注到）前面所有已經存在的詞（無論是原始輸入還是已經生成的詞）。

Qwen2.5-VL 採用 Decoder-only 架構，這意味著圖像和文字被視為一個統一的、連續的輸入序列，模型從左到右處理這個序列來生成答案。

現在，讓我們詳細拆解階段二的每一個步驟。

### 1. Merger (橋接模組) 的特徵對齊：不只是轉換維度

您的直覺完全正確，特徵對齊遠不止是簡單的維度轉換。ViT 輸出的視覺特徵是冗長且原始的，直接全部丟給 LLM 會導致計算災難。Merger 的核心任務是**「降維與精煉」**。

假設 ViT 將圖像分成了 `N` 個 patches (例如 16x16=256 個)，輸出了 `N` 個維度為 `d_vit` (例如 1024) 的視覺向量。Merger 的工作流程如下：

- **步驟一：引入可學習的「查詢向量 (Query Vectors)」**
    - Merger 會初始化一組數量固定且較少的 (例如 `K=32` 或 `K=64` 個) 可學習的向量，我們稱之為「查詢向量」。這些向量的維度可以與 ViT 的輸出維度相同。
    - 您可以將這些查詢向量想像成 `K` 個「視覺問題摘要員」。它們的任務是去「審問」ViT 輸出的所有 `N` 個原始視覺特徵，並從中提取出最精華、最重要的資訊。
        
- **步驟二：交叉注意力機制 (Cross-Attention)**
    - 這是 Merger 的核心。它會進行一次交叉注意力計算：
        - **Query (Q)**: 來自步驟一的可學習的「查詢向量」。
        - **Key (K)** 和 **Value (V)**: 均來自 ViT 輸出的 `N` 個視覺特徵向量。
    - 在這個過程中，每一個「查詢向量」都會去和所有的 `N` 個視覺特徵計算相似度，然後根據相似度加權求和，得到一個新的向量。
    - 這一步的結果是，`N` 個原始、分散的視覺特徵被**精煉和匯總**成了 `K` 個更具代表性、更抽象的**「視覺 Token 嵌入」**。
        
- **步驟三：線性投影 (Linear Projection)**
    - 最後，這 `K` 個精煉後的視覺 Token 嵌入會經過一個或多個線性層 (MLP)。
    - 這一步的主要目的才是**維度對齊**，將它們的維度從 `d_vit` (例如 1024) 轉換為 LLM 詞嵌入的維度 `d_llm` (例如 4096)。

**總結 Merger 的作用**：它像一個漏斗，將 ViT 產生的大量、原始的視覺 patch 特徵，透過交叉注意力的「篩選」和「提純」，壓縮成了數量固定、語意更豐富的視覺 Token，並最終將它們的「語言格式」（維度）調整到與 LLM 一致。

image -> ViT -> (N x d_vit)=(256 x 1024)

-> Query Vectors (K x d_vit)=(32 x 1024)

-> Linear projection (K x d_llm)=(32 x 4096)

---

### 2. 合併序列的構建

您的理解再次完全正確。LLM 接收的最終輸入序列，正是將 Merger 的輸出插入到 `image_pad` 位置的結果。具體流程如下：

1. **文字部分**：`"What kind of bird is this?..."` -> Tokenizer -> `[15496, 3292, ...]` -> 查詞嵌入表 -> `[Embedding_What, Embedding_kind, ...]`。
    
2. **視覺部分**：`Image` -> ViT -> `N` 個 Patch Embeddings -> Merger -> `K` 個精煉後的 Visual Token Embeddings。
    
3. **序列拼接**：模型會構建一個統一的嵌入序列，例如： `[Emb_user, Emb_vision_start, [K 個 Visual Token Embeddings], Emb_vision_end, Emb_What, Emb_kind, ..., Emb_assistant]`
    

這個拼接好的序列就是送入 LLM 的**初始輸入**。

---

### 3. LLM (Decoder-only) 的詳細內部結構

既然我們已經確定了是 Decoder-only 架構，那麼模型內部就只有一個組件：**一疊 Decoder Block (解碼器層)**。Qwen2.5-VL 7B 模型就有數十個這樣的層堆疊而成。

讓我們深入一個 **Decoder Block** 的內部，看看拼接好的序列是如何在其中流動的：

一個標準的 Transformer Decoder Block 包含兩個核心子層：

#### a. 子層一：帶掩碼的多頭自注意力 (Masked Multi-Head Self-Attention)

這是實現圖文深度融合的魔法所在。

- **輸入**：上一層傳來的、混合了文字和視覺 Token 的嵌入序列。
    
- **工作原理**：
    
    1. 序列中的**每一個 Token**（無論是代表 "bird" 的文字 Token，還是代表翠鳥翅膀的視覺 Token）都會生成自己的 Q、K、V 三個向量。
        
    2. **每一個 Token 的 Q 向量**都會去和**序列中它自己以及它前面的所有 Token 的 K 向量**計算注意力分數（相似度）。
        
    3. 這就意味著：
        
        - 文字 Token `"bird"` 可以「關注」到所有的視覺 Token，找出哪些視覺特徵最像「鳥」。
            
        - 視覺 Token（例如代表鳥喙的那個）可以「關注」到文字 Token `"What"` 和 `"kind"`，理解使用者是在對它進行「詢問」和「分類」。
            
        - 視覺 Token 之間也可以相互關注，例如鳥頭的 Token 可以關注到鳥身的 Token，從而形成對「一整隻鳥」的整體認知。
            
- **「多頭 (Multi-Head)」**：這個過程會被分成多個「頭」並行進行，每個頭可以學習關注不同方面的特徵（例如一個頭關注顏色，另一個頭關注紋理），最後再將結果合併，使其能夠捕捉更豐富的資訊。
    
- **「帶掩碼 (Masked)」**：這是 Decoder 的標誌。在**生成答案**時，為了防止模型作弊（看到後面的答案），會使用一個遮罩 (Mask) 來確保當前的 Token 只能關注到它前面的 Token。但在**處理輸入**時，這個遮罩是全開的，允許所有輸入 Token 之間自由互動。
    

#### b. 子層二：前饋神經網路 (Feed-Forward Network, FFN)

- **輸入**：經過自注意力層融合後的序列。
    
- **作用**：可以理解為一個**計算和非線性轉換站**。自注意力層負責「資訊的交互與融合」，而 FFN 則負責對融合後的資訊進行更深層次的「加工與提煉」。它通常由幾個線性層和一個非線性激活函數（如 `README.md` 中提到的 **SwiGLU**）組成。
    

**串聯與殘差連接**： 在每個 Decoder Block 內部，輸入序列會先經過自注意力層，然後將其輸出與原始輸入相加（**殘差連接**），再進行一次**層歸一化 (Layer Normalization)**。接著，結果會被送入 FFN，同樣進行殘差連接和層歸一化。

這個 `(Attention -> Add & Norm -> FFN -> Add & Norm)` 的完整流程就是一個 Decoder Block。

拼接好的序列會從第一個 Decoder Block 開始，逐層向上傳遞。在**每一層**，圖文資訊都會透過自注意力機制進行一次更深層次的融合。經過幾十層的處理後，模型對輸入的理解已經極其深刻和全面，最終頂層的輸出就會被用來預測下一個詞，從而開啟生成答案的過程。







#### QWen2.5-VL 架構圖解析

![[68747470733a2f2f7169616e77656e2d7265732e6f73732d636e2d6265696a696e672e616c6979756e63732e636f6d2f5177656e322e352d564c2f7177656e322e35766c5f6172632e6a706567.jpg]]


這是一張非常詳盡且資訊量巨大的 Qwen2.5-VL 模型架構圖！我將為您中文詳細解釋圖中的每一個技術細節，涵蓋從最底層的視覺輸入到最上層的語言生成。

---

### Qwen2.5-VL 模型架構圖 技術細節詳解

這張圖清晰地展示了 Qwen2.5-VL 如何將圖像、影片和文字有效地融合到一個統一的 Qwen2.5 LM Decoder 中，以實現強大的多模態理解和生成能力。

---

#### 1. 左下角：原始輸入與處理（Native Resolution Input & Vision Partition）

這部分展示了 Qwen2.5-VL 如何處理多種尺寸和類型的視覺輸入，以及其底層視覺處理的基礎。

- **原始高解析度輸入 (Native Resolution Input)**:
    
    - 圖中顯示了三個不同尺寸的圖像 (Picture 1, Picture 2, Picture 3) 和一個影片 (Video 1)。
    - **Picture 1**: 尺寸 (Width: 1092, Height: 8204)。這是一個非常高且窄的長圖，暗示了模型處理長文件、網頁截圖等的能力。
    - **Picture 2**: 尺寸 (Width: 224, Height: 28)。這是一個非常扁平的圖，可能代表橫幅廣告或某個局部細節。
    - **Picture 3**: 尺寸 (Width: 1260, Height: 700)。這是一個較為標準的圖片。
    - **Video 1**: 尺寸 (Width: 644, Height: 392)，時長 8 秒。影片由多個影格組成。
    - **關鍵信息**: Qwen2.5-VL 能夠處理各種不同尺寸和長寬比的圖像和影片，這對於其在文件解析、物體定位和影片理解等方面的應用至關重要。模型不會簡單粗暴地將所有圖像縮放到固定大小，而是會進行智能調整。
        
- **Window Partition (窗口劃分)**:
    
    - 這是 Vision Encoder 內部處理圖像的一種方式。圖像首先被分割成許多小的、固定大小的「窗口」(Windows)，例如圖中示意的一個 112x112 的區域被分割成 4x4 的 28x28 窗口。
    - 每個小窗口內的像素會被單獨處理，然後這些處理後的特徵再進行組合。這是一種常見的技術，用於減少全局注意力的計算複雜度，同時保持對局部細節的感知。
    - **`Conv3D (2x14x14)`**: 這裡的 `Conv3D` 指的是 3D 卷積層。對於影片（Video 1），它不僅在空間維度（Height, Width）進行卷積，還會在時間維度（Frames）進行卷積。
        - `2x14x14` 可能表示時間維度上採樣 2 幀，每幀在空間上被劃分為 14x14 的 Patches。
        - 這表明 Vision Encoder 能夠同時處理影片的空間和時間特徵，捕捉影片中的動作和變化。

---

#### 2. 中間底部：Vision Encoder (視覺編碼器)

這是模型的「眼睛」部分，負責從圖像和影片中提取視覺特徵。

- **Native Resolution Input**: 再次強調了視覺編碼器能夠處理原始解析度輸入。
- **Vision Encoder Box**:
    - 這個模塊內部使用了類似 ViT 的架構。
    - **關鍵技術**：
        - **FFN with SwiGLU**: 前饋網路 (Feed-Forward Network) 中使用了 SwiGLU 激活函數。SwiGLU 是一種近年來在大型語言模型中表現出色的激活函數，能夠提高模型的性能和訓練穩定性。
        - **RMSNorm**: 使用 RMSNorm 進行層歸一化 (Layer Normalization)。相較於傳統的 LayerNorm，RMSNorm 更為簡潔高效，且在某些場景下表現更好。
        - **Full Attention / Window Attention**:
            - **Full Attention**: 在 Vision Encoder 的某些層級，可能仍會使用全局的自注意力機制來捕捉圖像或影片中遠距離的依賴關係。
            - **Window Attention**: 如前所述，這是將注意力計算限制在局部窗口內，以降低計算複雜度，特別適用於處理高解析度圖像。圖中顯示了 `x1` 和 `xM`，暗示了 Vision Encoder 可能有多層，其中一層或幾層使用全局注意力，而其他多層 (`xM`) 則使用窗口注意力。這是一種**混合注意力機制**，旨在兼顧全局感知和局部細節的高效處理。
                
- **時間資訊整合 (Temporal Information Integration) for Video**:
    
    - **`Sampled MVPE Time IDs`**: `(0 15) / (0 3 6 9 12 15)` 這表示影片的影格是如何被採樣的。
        - `0 15`: 可能是影片的起始和結束影格索引。
        - `0 3 6 9 12 15`: 這是一個稀疏採樣的例子，表示從影片中均勻地抽取了一定數量的幀（例如每 3 幀取 1 幀），而不是處理所有幀。
    - **`Conv3D with 2x temporal merging`**: 這再次強調了 3D 卷積在影片處理中的應用，並且提到了「2 倍時間維度合併」。這意味著在處理影片的過程中，可能會在時間維度上進行降採樣或融合，將連續的幾幀資訊合併成一個更抽象的時間特徵，從而處理更長的影片。
    - **`Align with absolute time`**: 這是 Qwen2.5-VL 處理影片的一個重要創新點。它將視覺特徵與**絕對時間戳**對齊。這對於精確的影片事件定位和理解至關重要。
    - **`Dynamic FPS sampling`**: 根據影片內容或任務需求，動態調整每秒取樣的幀數 (Frames Per Second)。
    - **`MP/IPC Time (s)`**: Memory / Instruction Per Cycle Time (s)，可能是一種內部指標，表示在特定時間點的處理效率。
    - **`Absolute Time`**: `(0 1s 3s 5s 7s)`，表示模型處理影片時，能夠感知和利用影格的絕對時間位置信息，這對影片理解（如「在第 5 秒發生了什麼？」）至關重要。
        
- **輸出**：Vision Encoder 將圖像和影片的視覺信息轉換成一系列的**視覺 Token (Visual Tokens)**。這些 Token 數量與原始 Patches 數相關，但經過編碼和可能的時間/空間融合後，它們已經是更抽象、更精煉的特徵。
    

---

#### 3. 中間偏上：Token 序列與 Qwen2.5 LM Decoder 輸入

這部分展示了視覺 Token 如何與文字 Token 合併，形成 Qwen2.5 LM Decoder 的最終輸入序列。

- **方塊序列**: 這是 Qwen2.5 LM Decoder 的輸入序列。每個小方塊代表一個 Token 嵌入。
    - **藍色方塊**: 代表**圖像或影片的視覺 Token**。這些 Token 是 Vision Encoder 的輸出，經過「翻譯官」（即架構圖中未明確標識但概念上存在的 Merger/Bridge Module）調整維度後，插入到語言模型的詞嵌入空間中。
    - **淺色（黃、紅、綠、灰）方塊**: 代表**文字 Token**。這些是使用者 Prompt (文字指令) 經過 Tokenizer 處理後的詞嵌入。
        
- **Token 數量**:
    - `11427 tokens Picture 1`: 即使經過 Vision Encoder 處理，像 Picture 1 這樣的高解析度長圖仍然會產生大量的視覺 Token。
    - `8 tokens Picture 2`: 較小的圖片產生較少的 Token。
    - `1125 tokens Picture 3`: 中等大小圖片的 Token 數量。
    - `644 / 1288 / 2576 tokens Video 1`: 影片的 Token 數量會根據採樣的幀數和每幀處理的 Patches 數而變化，這三個數字可能代表了在不同 `total_pixels` 或 `max_frames` 限制下的取樣結果。
        
- **序列結構**:
    - 文字 `"Images and videos here."` + 藍色方塊 (視覺 Token) + 文字 `"Picture 1 is an image from a blog"`。
    - 這清楚地表明，圖像和影片的視覺 Token **直接嵌入在文字 Token 序列之中**。模型在處理時，可以自由地在文字和視覺 Token 之間進行注意力計算，實現真正的多模態深度融合。
    - `Picture 1 is an image from a blog`: 這是文字指令中對視覺內容的引用或描述，與視覺 Token 共同作為 LLM 的輸入。

---

#### 4. 最上層：Qwen2.5 LM Decoder (大型語言模型解碼器)

這是模型的「大腦」，負責理解、推理和生成答案。

- **LLM Decoder Box**:
    - **架構**：如前所述，這是一個**僅解碼器 (Decoder-only)** 的 Transformer 架構。
    - **輸入**：就是下方組合好的、混合了文字和視覺 Token 的長序列。
    - **內部運作**：由多層 Decoder Block 堆疊而成。每一層的**帶掩碼的多頭自注意力 (Masked Multi-Head Self-Attention)** 機制，允許序列中的每個 Token 關注其之前的所有 Token（包括所有文字 Token 和所有視覺 Token）。
        - 這使得文字 Token 可以參照視覺內容，視覺 Token 也可以理解文字上下文。
        - 模型會在這個融合的空間中進行深層次的推理，理解圖像和影片的內容，並結合文字指令，生成符合語意的回答。
    - **自迴歸生成 (Autoregressive Generation)**：LLM Decoder 會一詞一詞地預測輸出，直到生成完整的答案。

---

### 總結架構亮點

1. **統一的 Decoder-Only 架構**：Qwen2.5-VL 將所有輸入（文本、圖像、影片）統一為一個 Token 序列，並由一個強大的 LLM Decoder 進行處理，簡化了多模態的融合機制。
2. **靈活的視覺編碼器**：支持多種分辨率、長寬比的圖像和時間/空間維度可控的影片，並使用窗口注意力和 3D 卷積等技術高效提取特徵。
3. **智能的視覺 Token 插入**：視覺 Token 被直接嵌入到語言 Token 序列中，使得語言模型可以在其核心 Transformer 層中進行深度的圖文交叉注意力，實現無縫的多模態融合。
4. **對影片時間信息的深度感知**：通過 `Conv3D`、動態採樣和絕對時間對齊，模型能夠更好地理解影片中的動態事件和時間順序。
5. **高效的組件**：採用 SwiGLU 和 RMSNorm 等現代 LLM 中常用的高效組件，提升模型性能。

這張圖描繪了一個非常精巧的多模態模型，它有效地利用了各個組件的優勢，實現了對文字、圖像和影片的全面理解與生成。








#### Merger 的位置與工作流程
###### Merger 是在哪裡？

**Merger 不在 LLM Decoder 內部。** 它是一個獨立的模組，在邏輯上位於 **Vision Encoder 之後，LLM Decoder 之前**。

###### 正確的工作流程是什麼？

您提出的第二個猜想是**完全正確**的。Merger 必須在視覺特徵被插入到文字序列**之前**完成它的工作。

讓我們用一個更精確的、分步的流程來描述：

1. **【使用者端/Processor】準備階段 - 創建「藍圖」**:
    
    - 使用者提供了圖像和文字 Prompt。
    
    - `processor` 將文字 Prompt 轉換為 Token ID 序列，其中包含了特殊的佔位符 `<|image_pad|>`。例如：`[ID_user, ID_vision_start, ID_image_pad, ID_vision_end, ID_What, ID_is, ...]`。
        
    - `processor` 同時將圖像預處理成一個圖像 Tensor。
        
    - 最終，`processor` 創建了一個「統一的輸入包」，這個包裡包含了**文字 Token ID 序列（帶佔位符）**和**圖像 Tensor**。這個包被發送到模型（GPU端）。
        
2. **【模型端】執行階段 - 根據「藍圖」施工**:
    
    - **步驟 A (視覺通路)**：模型首先將輸入包中的**圖像 Tensor**送入 **Vision Encoder**。Vision Encoder 經過計算，輸出一系列原始的、高維的視覺特徵向量 (Visual Features)。
        
    - **步驟 B (特徵對齊)**：緊接著，這些視覺特徵向量**立刻**被送入 **Merger** 模組。Merger 進行降維、精煉和維度對齊，輸出一組數量固定、語意豐富且與 LLM 詞彙空間維度一致的**「視覺 Token 嵌入 (Visual Token Embeddings)」**。
        
    - **步驟 C (序列構建)**：模型根據輸入包中的**文字 Token ID 序列**查找對應的文字嵌入 (Text Embeddings)。
        
    - **步驟 D (關鍵的替換/插入)**：模型將文字嵌入序列中的 `ID_image_pad` 對應的佔位符嵌入，**替換**成步驟 B 中 Merger 產生的那一組**「視覺 Token 嵌入」**。
        
    - 至此，一個混合了文字嵌入和精煉後視覺嵌入的、完整的、無縫的序列才真正構建完成。
        
3. **【LLM Decoder】推理階段**:
    
    - 這個最終構建好的混合序列，作為一個整體，被送入 **LLM Decoder** 的第一層，開始逐層進行深度的圖文融合與自回歸生成。
        

**總結**：Merger 是在模型內部，但在 LLM Decoder 之外的一個**前置處理模塊**。它的工作是在視覺資訊進入 LLM 的「思考迴路」之前，將其從原始的「像素語言」翻譯成 LLM 能聽懂的「詞彙語言」。`processor` 在前端只是預留了一個位置，而 Merger 則負責生成真正要填入這個位置的高品質內容。





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


