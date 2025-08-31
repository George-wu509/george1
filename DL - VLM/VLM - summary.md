

Reference:
视觉大模型（VLM）：2025技术进展与未来展望 - 北方的郎的文章 - 知乎
https://zhuanlan.zhihu.com/p/1923438673086649859


视觉大模型（VLM）是多模态人工智能模型，能够同时处理和理解图像和文本数据。与传统的单模态模型（如仅处理图像的卷积神经网络或仅处理文本的语言模型）相比，VLM通过学习视觉和语言之间的关联，能够处理更复杂的任务。例如，VLM可以根据图像生成描述性文本（图像描述），回答基于图像的问题（视觉问答），甚至根据文本生成图像（多模态生成）。


![[Pasted image 20250828093929.png]]


### **VLM模型全面比較列表 (擴展版)**

| 特性              | LLaVA-NeXT          | Florence-2                | Video-LLaVA   | **Qwen2-VL (開源)**   | **InstructBLIP**    | **GPT-4o (API)** | **Gemini 1.5 Pro (API)** | **Qwen-VL-Max (API)** |
| --------------- | ------------------- | ------------------------- | ------------- | ------------------- | ------------------- | ---------------- | ------------------------ | --------------------- |
| **核心優勢**        | 社群活躍/易微調            | 統一多樣視覺任務                  | 原生影片時序理解      | 中文能力強/性能均衡          | 經典指令視覺模型            | 頂級多模態推理          | **超長上下文/影片理解**           | 中文頂尖/功能全面             |
| **開源狀態**        | ✅ **完全開源**          | ✅ **完全開源**                | ✅ **完全開源**    | ✅ **完全開源**          | ✅ **完全開源**          | ❌ 閉源API          | ❌ 閉源API                  | ❌ 閉源API               |
| **圖像任務**        |                     |                           |               |                     |                     |                  |                          |                       |
| 圖像描述            | ⭐⭐⭐⭐⭐               | ⭐⭐⭐⭐⭐                     | ⭐⭐⭐⭐          | ⭐⭐⭐⭐⭐               | ⭐⭐⭐⭐                | ⭐⭐⭐⭐⭐            | ⭐⭐⭐⭐⭐                    | ⭐⭐⭐⭐⭐                 |
| 視覺問答(VQA)       | ⭐⭐⭐⭐⭐               | ⭐⭐⭐⭐⭐                     | ⭐⭐⭐⭐          | ⭐⭐⭐⭐⭐               | ⭐⭐⭐⭐                | ⭐⭐⭐⭐⭐            | ⭐⭐⭐⭐⭐                    | ⭐⭐⭐⭐⭐                 |
| 多模態推理           | ⭐⭐⭐⭐                | ⭐⭐⭐⭐                      | ⭐⭐⭐           | ⭐⭐⭐⭐                | ⭐⭐                  | ⭐⭐⭐⭐⭐            | ⭐⭐⭐⭐⭐                    | ⭐⭐⭐⭐                  |
| 視覺定位(Grounding) | ⭐⭐                  | ⭐⭐⭐⭐⭐                     | ⭐⭐            | ⭐⭐⭐⭐                | ⭐⭐                  | ⭐⭐⭐⭐             | ⭐⭐⭐⭐                     | ⭐⭐⭐⭐⭐                 |
| **影片任務**        |                     |                           |               |                     |                     |                  |                          |                       |
| 物件偵測/分割         | ❌                   | ⭐⭐⭐⭐                      | ⭐⭐⭐           | ⭐⭐⭐                 | ❌                   | ⭐⭐⭐              | ⭐⭐⭐⭐                     | ⭐⭐⭐⭐                  |
| 物件追蹤            | ❌                   | ⭐⭐                        | ⭐⭐⭐⭐          | ⭐⭐                  | ❌                   | ⭐⭐⭐⭐             | ⭐⭐⭐⭐⭐                    | ⭐⭐⭐⭐                  |
| 影片描述            | ❌                   | ⭐⭐                        | ⭐⭐⭐⭐⭐         | ⭐⭐⭐                 | ❌                   | ⭐⭐⭐⭐⭐            | ⭐⭐⭐⭐⭐                    | ⭐⭐⭐⭐                  |
| **部署與易用性**      |                     |                           |               |                     |                     |                  |                          |                       |
| **部署難易度**       | 中等                  | **容易**                    | 困難            | 中等                  | 中等                  | **極容易**          | **極容易**                  | **極容易**               |
| **上手速度**        | 中等                  | **快**                     | 慢             | 中等                  | 中等                  | **最快**           | **最快**                   | **最快**                |
| **使用方式**        | 自行部署 (Hugging Face) | Hugging Face Transformers | 自行部署 (特定Repo) | 自行部署 (Hugging Face) | 自行部署 (Hugging Face) | 官方API            | 官方API (Google AI Studio) | 官方API (阿里雲)           |
| **社群與文件**       | 非常活躍                | 活躍                        | 一般            | 活躍 (中文社群強)          | 較舊，但穩定              | 非常完整             | 非常完整                     | 完整                    |


單一的 VLM 就像一個只有視覺皮層和語言中樞的大腦，它能「看懂」和「描述」，但要完成更複雜、更專業的任務，就必須與其他功能高度特化的「腦區」——也就是其他 Computer Vision (CV) 模型——協同工作。

這種「VLM 作為認知核心，調度 CV 專家模型作為感官系統」的混合智能架構，是目前最具潛力且被廣泛關注的發展方向。以下是一些極具潛力的領域和任務，它們都迫切需要 VLM 與其他 CV 模型的深度配合。

---


|                                                        |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 機器人學與物理互動 (Robotics & Physical Interaction)            | 輸入: camera2D圖像 + 「幫我從碗裡把那顆最熟的番茄拿過來，注意不要捏壞了。」<br>輸出: 生成機器人可以執行的指令碼<br><br>VLM 可以從 2D 圖像中識別出「番茄」，甚至可能判斷出哪個顏色更紅（代表更熟）。但它完全不理解物體的 3D 幾何形狀、物理屬性（硬度、脆性、摩擦力）、可供抓握的點 (affordance)，也無法規劃出精確的毫米級運動軌跡<br>-><br>3D 物體重建模型 (3D Object Reconstruction)<br>抓取姿態估計模型 (Grasping Pose Estimation)<br>物體可供性分割模型 (Affordance Segmentation)<br><br>steps:<br>使用者輸入promot:「拿那個最熟的番茄」, camera系統根據2D圖像用3D Object Reconstruction生成其 3D 點雲模型. 抓取姿態估計模型分析點雲，輸出結構化數據, 這段結構化數據被「翻譯」成文字，與原始指令一起輸入給 VLM. VLM 進行最終決策，生成機器人可以執行的指令碼：「採用輕柔爪力 (小于 5N)，在世界座標 (0.1, 0.5, 0.2) 處執行抓取動作。」 |
| 自動駕駛與情境感知 (Autonomous Driving & Situational Awareness) | 輸入: 自動駕駛接收fusing訊號<br>輸出: 生成機器人可以執行的指令碼<br><br>VLM 可以識別「小孩」和「皮球」，但它沒有內建的物理世界模型或行為預測能力，無法判斷出「小孩很可能會跟著球衝出來」這種高風險的意圖 (intent)<br>-><br>行人/車輛軌跡預測模型 (Trajectory Prediction Models)<br>行為分析與意圖預測模型 (Behavior & Intent Prediction)<br>3D 場景重建模型 (NeRFs, Gaussian Splatting)<br><br>steps:<br>自動駕駛接收fusing訊號, 系統感知到綠燈，初始決策是「前進」. 軌跡預測模型輸出數據, 行為分析模型輸出數據. 這些資訊被轉換成文字摘要：「ID 12 的小孩正在跑動，視線和頭部都朝向 ID 13 的球。球的預測軌跡將進入我方車道。」. VLM 接收到這個摘要後，觸發了它在訓練中學到的「常識」：小孩通常會追逐他們的玩具，而忽略周圍的危險. VLM 推翻初始決策，輸出新的行動指令：「高風險情況。立即執行預防性減速，並將該小孩標記為最高優先級的觀察目標。」                     |
| 醫學影像分析 <br>(Medical Image Analysis)                    | 輸入: 3D CT/MRI<br>輸出: 生成一份詳細的放射學報告<br><br>VLM 無法直接讀取 DICOM 這類專業的 3D 醫學影像格式。它也無法進行精確的、像素級的病灶體積測量，更無法將兩次不同時間拍攝的 3D 掃描進行精確的空間對齊<br>-><br>3D 醫學影像分割模型<br>影像配準模型 (Image Registration Models)<br>異常檢測模型 (Anomaly Detection Models)<br><br>steps:<br>用Segmentation Model(U-Net)在 3D 影像中精確地分割出腫瘤, 用Image Registration Models比較兩次CT scan在三維空間中對齊, 用Anomaly Detection Models檢測異常區域. 這些資訊被匯總成文字，輸入給 VLM生成報告                                                                                                                                                    |
| 擴增實境與空間計算 <br>(AR & Spatial Computing)                 | 輸入: 一連串 2D 影片幀 + 「幫我把這個虛擬沙發，靠著客廳最長的那面空牆擺放。」<br>輸出: 新的3D 重建模型<br><br>VLM 看到的只是一連串 2D 影片幀，它沒有關於這個房間的幾何結構、表面材質和空間深度的記憶<br>-><br>SLAM (同步定位與地圖構建)<br>即時 3D 場景重建 (Real-time 3D Reconstruction)<br>平面檢測與語義分割模型 (Plane Detection & Semantic Segmentation)<br><br>steps:<br>使用者戴上眼鏡四處走動，SLAM 和 3D 重建模型在後台持續工作，建立並更新一個房間的 3D 數字孿生模型. 平面檢測模型分析這個 3D 模型，輸出語義標籤. 使用者發出指令：「把沙發放在最長的空牆邊」, VLM 接收指令和從 CV 模型得到的場景語義數據。它進行推理，找到 plane_1 是最符合條件的. VLM 輸出指令給 AR 渲染引擎：「將 virtual_sofa.asset 放置在 plane_1 的中心位置。」                                                        |








|     | 用在這些Projects的流程                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1.  | <mark style="background: #FF5582A6;">qwen_model </mark><br>是一個已經被訓練好的、包含數十億個參數的深度學習模型（具體來說是多模態 Transformer）.您給它一組經過處理的數字（張量 Tensors），它會透過其內部複雜的神經網路進行運算，然後輸出另一組代表結果的數字。它不直接處理圖片檔案或原始文字<br><br><mark style="background: #BBFABBA6;">qwen_processor</mark><br>萬能的資料預處理助理 (The Versatile Pre-processing Assistant), Processor（處理器）是一個方便的「容器」，它內部打包了至少三樣重要的工具:<br><br>1. 圖像/影片處理器 (Image/Video Processor) - 會進行縮放、裁切、正規化（normalization）等操作，將圖片轉換為 qwen_model 視覺部分能理解的數字張量<br><br>2. 文字分詞器 (Tokenizer) - 負責處理文字數據。它會將文字（例如 "你好世界"）轉換成一個個的 token，並再將這些 token 轉換成對應的數字 ID，也就是 qwen_model 語言部分能理解的數字張量<br><br>3. 聊天模板格式化器 (Chat Templater) - 負責將對話內容轉換成模型訓練時所遵循的特定格式。例如，它會自動加上 <\|im_start\|>, user, <\|im_end\|> 等特殊標記，確保輸入的格式和模型預期的完全一致 |
| 2.  | text = <mark style="background: #BBFABBA6;">qwen_processor</mark>.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)<br><br>角色：專職的對話格式化工具 (A Specialized Formatting Tool)<br><br>功能：這是從 qwen_processor 中拿出「聊天模板格式化器」這個工具來單獨使用。它的唯一目的是將一個結構化的對話列表（messages）轉換成一個帶有完整格式的單一字串。tokenize=False 這個參數就是在告訴它：「請先不要將文字轉成數字ID，只幫我把格式排好，回傳一個字串就好。」                                                                                                                                                                                                                                                                                                                                                                    |
| 3.  | inputs = <mark style="background: #BBFABBA6;">qwen_processor</mark>(text=[text], return_tensors="pt").to("cuda")<br><br>角色：將格式化文字進行最終處理<br><br>功能：這是調用 qwen_processor 的純文字模式。透過明確指定 text=[text]，我們是在告訴它：「這次沒有任何圖片或影片，請只使用你內部的『文字分詞器 (Tokenizer)』，將這個已經格式化好的文字字串，轉換成最終的數字張量（input_ids 和 attention_mask）。」                                                                                                                                                                                                                                                                                                                                                                                                                          |
| 4.  | generated_ids = <mark style="background: #FF5582A6;">qwen_model</mark>.generate(<br>        inputs.input_ids,<br>        attention_mask=inputs.attention_mask,<br>        max_new_tokens=512 <br>    )<br><br>角色：啟動大腦進行思考與生成<br><br>功能：這是整個流程的終點。我們將 qwen_processor 準備好的、包含 input_ids 和 attention_mask 的 inputs 物件，餵給 qwen_model。模型會根據這些輸入，進行推理計算，並「生成」代表答案的新 token ID 序列，也就是 generated_ids。最後，我們再用 processor.batch_decode 將這些數字ID翻譯回人類可讀的文字。                                                                                                                                                                                                                                                                                     |




首先，我們定義一個可重複使用的函式 `get_qwen_reasoning_response`。這個函式封裝了與 Qwen2-VL 互動的所有樣板程式碼。它會接收來自專家模型的「情境」和來自使用者的「指令」，將它們組合成一個結構化的提示 (Prompt)，然後發送給 Qwen2-VL 模型，最後清理並返回模型生成的文字結果。之後的所有範例都會調用這個函式
```python
# Preamble: A helper function to interact with the loaded Qwen2-VL model.
# Ensure that `qwen_model` and `qwen_processor` are already loaded from previous cells.

def get_qwen_reasoning_response(context_summary: str, user_prompt: str) -> str:
    """
    Generates a response from Qwen2-VL based on system context and a user prompt.

    Args:
        context_summary (str): The text summary generated from specialist CV models.
        user_prompt (str): The user's command or question.

    Returns:
        str: The cleaned text response from the Qwen2-VL model.
    """
    # Create the full prompt with clear instructions for the model.
    # We instruct it to act as an expert system and base its answer ONLY on the provided data.
    full_prompt = f"""
You are an expert AI system. Your task is to make a decision or generate a response based on the structured data provided by your perception modules.

--- PERCEPTION MODULE DATA ---
{context_summary}
--- END OF DATA ---

User Prompt: "{user_prompt}"

Based on the perception data, provide the final output.
"""
    
    # Prepare the inputs for the model using the processor.
    messages = [{"role": "user", "content": full_prompt}]
    text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = qwen_processor(text=[text], return_tensors="pt").to("cuda")

    # Generate the response.
    generated_ids = qwen_model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=512  # Allow for longer, more detailed responses
    )
    
    # Decode and clean up the response.
    response = qwen_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # A simple way to clean the output is to find the user prompt and take what's after.
    # This removes the echoed prompt from the model's output.
    cleaned_response = response.split(user_prompt)[-1].strip()
    # A common pattern is that the model starts its answer after a marker.
    # Let's find a generic way to extract the final part.
    if "Final output:" in cleaned_response:
        cleaned_response = cleaned_response.split("Final output:")[-1].strip()
        
    return cleaned_response
```



### **1. 機器人學與物理互動 (Robotics & Physical Interaction)**

這個領域的目標是讓機器人不僅能「看懂」世界，更能與之進行精細、安全的物理互動。

- **潛力任務**: 讓機器手臂根據自然語言指令，完成複雜的操作。例如：「幫我從碗裡把那顆最熟的番茄拿過來，注意不要捏壞了。」
    
- **VLM 的局限**: VLM 可以從 2D 圖像中識別出「番茄」，甚至可能判斷出哪個顏色更紅（代表更熟）。但它完全不理解物體的 **3D 幾何形狀**、**物理屬性**（硬度、脆性、摩擦力）、**可供抓握的點 (affordance)**，也無法規劃出精確的毫米級運動軌跡。
    
- **需配合的 CV 模型**:
    
    - **3D 物體重建模型 (3D Object Reconstruction)**: 利用深度相機 (Depth Camera) 或多視角圖像，即時重建出番茄的 3D 網格 (Mesh) 或點雲 (Point Cloud)。
        
    - **抓取姿態估計模型 (Grasping Pose Estimation)**: 分析 3D 模型，計算出機器手爪最適合的抓取位置、角度和姿態，生成一個「抓取熱力圖」。
        
    - **物體可供性分割模型 (Affordance Segmentation)**: 分割出物體上可以被「推」、「拉」、「握住」的不同區域。
        
- **協作模式**:
    
    1. 使用者發出指令：「拿那個最熟的番茄」。
        
    2. VLM 理解指令，並指揮相機系統定位到它認為「最熟的番茄」。
        
    3. **3D 重建模型**對準該番茄，生成其 3D 點雲模型。
        
    4. **抓取姿態估計模型**分析點雲，輸出結構化數據：`{"object": "tomato", "ripeness": 0.9, "grasp_points": [{"x":0.1, "y":0.5, "z":0.2, "orientation": ...}], "required_force": "< 5N"}`。
        
    5. 這段結構化數據被「翻譯」成文字，與原始指令一起輸入給 **VLM**。
        
    6. VLM 進行最終決策，生成機器人可以執行的指令碼：「採用輕柔爪力 (小于 5N)，在世界座標 (0.1, 0.5, 0.2) 處執行抓取動作。」
        
```python
# --- 1. Robotics & Physical Interaction ---

# Step 1: Simulate the output from 3D Reconstruction and Grasping Pose Estimation models.
cv_model_output_robotics = {
    "object_id": "tomato_01",
    "class_name": "Tomato",
    "ripeness_score": 0.92, # Score from a color/texture analysis model
    "position_3d": [0.54, -0.12, 1.03], # World coordinates in meters
    "grasp_candidates": [
        {"point": [0.55, -0.11, 1.05], "orientation": [0.0, 0.707, 0.0, 0.707], "score": 0.95},
        {"point": [0.53, -0.13, 1.02], "orientation": [0.1, 0.701, 0.0, 0.699], "score": 0.88}
    ],
    "physical_properties": {
        "estimated_firmness": "soft",
        "recommended_force_limit_newtons": 5.0
    }
}

# Step 2: The "Translation Layer" function.
def translate_robotics_data_to_text(data):
    # This function converts the structured data into a readable text summary.
    summary = f"Object Detected: {data['class_name']} (ID: {data['object_id']}).\n"
    summary += f"Ripeness Score: {data['ripeness_score']}.\n"
    summary += f"3D Position (X,Y,Z): {data['position_3d']}.\n"
    summary += f"Physical Property: Estimated firmness is {data['physical_properties']['estimated_firmness']}.\n"
    summary += f"Recommended Grip Force Limit: {data['physical_properties']['recommended_force_limit_newtons']} Newtons.\n"
    # Select the best grasp point to report
    best_grasp = max(data['grasp_candidates'], key=lambda x: x['score'])
    summary += f"Optimal Grasp Point: {best_grasp['point']} with orientation {best_grasp['orientation']}."
    return summary

# Step 3: Define the user's high-level command.
user_command_robotics = "Pick up the ripe tomato gently. Output the final command as a JSON object."

# Step 4: Qwen2-VL's Core Function: Reasoning and Command Generation
context_robotics = translate_robotics_data_to_text(cv_model_output_robotics)
print("--- Input to Qwen2-VL (Robotics) ---")
print(context_robotics)
print(f"User Prompt: \"{user_command_robotics}\"")

# Get the final command from the VLM
json_command = get_qwen_reasoning_response(context_robotics, user_command_robotics)

print("\n--- Output from Qwen2-VL ---")
print("Function: Decision Making & Structured Command Generation")
print(f"Generated JSON Command:\n{json_command}")
```



---

### **2. 自動駕駛與情境感知 (Autonomous Driving & Situational Awareness)**

自動駕駛需要的遠不止是偵測車輛和行人，它需要對複雜、動態的交通環境有深刻的**預測和推理**能力。

- **潛力任務**: 在十字路口進行人性化決策。例如：「雖然我們的交通號誌是綠燈，但右前方有個小孩在追逐滾向馬路的皮球，我們應該預防性地減速等待，而不是直接開過去。」
    
- **VLM 的局限**: VLM 可以識別「小孩」和「皮球」，但它沒有內建的物理世界模型或行為預測能力，無法判斷出「小孩很可能會跟著球衝出來」這種高風險的**意圖 (intent)**。
    
- **需配合的 CV 模型**:
    
    - **行人/車輛軌跡預測模型 (Trajectory Prediction Models)**: 通常使用 RNN 或 Transformer 架構，根據過去幾秒的運動軌跡，預測未來幾秒內所有動態物體可能的位置。
        
    - **行為分析與意圖預測模型 (Behavior & Intent Prediction)**: 不僅預測軌跡，還分析行人的姿態、頭部朝向、視線方向，來判斷其意圖（例如，準備過馬路、在等人、在看手機）。
        
    - **3D 場景重建模型 (NeRFs, Gaussian Splatting)**: 建立一個照片級真實感的周圍環境 3D 模型，幫助 VLM 更好地理解空間關係。
        
- **協作模式**:
    
    1. 系統感知到綠燈，初始決策是「前進」。
        
    2. **軌跡預測模型**輸出數據：`{"object": "child", "id": 12, "predicted_path": [...], "collision_risk": "low"}, {"object": "ball", "id": 13, "predicted_path": "[...], "collision_risk": "high"}`。
        
    3. **行為分析模型**輸出數據：`{"id": 12, "gaze_target": "ball", "head_pose": "following_ball", "action": "running"}`。
        
    4. 這些資訊被轉換成文字摘要：「ID 12 的小孩正在跑動，視線和頭部都朝向 ID 13 的球。球的預測軌跡將進入我方車道。」
        
    5. **VLM** 接收到這個摘要後，觸發了它在訓練中學到的「常識」：小孩通常會追逐他們的玩具，而忽略周圍的危險。
        
    6. VLM 推翻初始決策，輸出新的行動指令：「高風險情況。立即執行預防性減速，並將該小孩標記為最高優先級的觀察目標。」
        
```python
# --- 2. Autonomous Driving & Situational Awareness ---

# Step 1: Simulate output from Trajectory and Behavior Prediction models.
cv_model_output_adas = [
    {"object_id": "child_01", "class_name": "Pedestrian", "age_group": "child", "action": "running", "gaze_target": "ball_01", "predicted_trajectory_collides": True, "time_to_collision_sec": 1.5},
    {"object_id": "ball_01", "class_name": "Ball", "predicted_trajectory_collides": True},
    {"object_id": "traffic_light_01", "class_name": "TrafficLight", "state": "green"}
]

# Step 2: The "Translation Layer" function.
def translate_adas_data_to_text(data_list):
    summary = "Current scene perception summary:\n"
    for item in data_list:
        summary += f"- Object ID {item['object_id']} ({item['class_name']}):\n"
        for key, value in item.items():
            if key != 'object_id' and key != 'class_name':
                summary += f"    - {key}: {value}\n"
    return summary

# Step 3: Define the situation and the VLM's task.
user_command_adas = "The default driving plan is 'Proceed through intersection at 25 km/h'. Evaluate the current situation based on perception data and decide if the plan should be overridden. Provide a new plan with a reason."

# Step 4: Qwen2-VL's Core Function: Risk Assessment and Reasoning
context_adas = translate_adas_data_to_text(cv_model_output_adas)
print("--- Input to Qwen2-VL (Autonomous Driving) ---")
print(context_adas)
print(f"User Prompt: \"{user_command_adas}\"")

# Get the final decision from the VLM
decision = get_qwen_reasoning_response(context_adas, user_command_adas)

print("\n--- Output from Qwen2-VL ---")
print("Function: Risk Assessment & Situational Reasoning")
print(f"Generated Driving Decision:\n{decision}")
```

---

### **3. 醫學影像分析 (Medical Image Analysis)**

這是 AI 輔助診斷的核心領域，要求極高的準確性和可解釋性。

- **潛力任務**: 自動為一個 3D 的 CT 或 MRI 掃描生成一份詳細的放射學報告，並與該病人過往的掃描影像進行比較，指出病灶的變化。
    
- **VLM 的局限**: VLM 無法直接讀取 DICOM 這類專業的 3D 醫學影像格式。它也無法進行精確的、像素級的病灶體積測量，更無法將兩次不同時間拍攝的 3D 掃描進行精確的空間對齊。
    
- **需配合的 CV 模型**:
    
    - **3D 醫學影像分割模型 (e.g., U-Net, nnU-Net)**: 在 3D 影像中精確地分割出腫瘤、器官、病變區域的輪廓。
        
    - **影像配準模型 (Image Registration Models)**: 將病人本次的 CT 掃描和半年前的 CT 掃描在三維空間中完美對齊，以便逐點比較。
        
    - **異常檢測模型 (Anomaly Detection Models)**: 在影像中高亮顯示與正常組織模式不符的潛在異常區域。
        
- **協作模式**:
    
    1. **配準模型**將新舊兩份 3D 掃描對齊。
        
    2. **分割模型**在新舊掃描上分別運行，輸出結構化數據：`{"scan": "current", "lesion_volume": "15.3 cm³"}, {"scan": "previous", "lesion_volume": "12.1 cm³"}`。
        
    3. **異常檢測模型**發現了一個新的微小高亮點，輸出：`{"new_anomaly_detected": true, "location": "liver_segment_4", "size": "5mm"}`。
        
    4. 這些資訊被匯總成文字，輸入給 **VLM**。
        
    5. VLM 將這些枯燥的數據，生成一份符合放射科醫生行文風格的專業報告：「影像所見，與半年前的掃描相比，肝臟S6段的已知病灶體積增大約26.4%。此外，在肝臟S4段新見一個約5毫米的結節，建議進一步追蹤檢查。」
        
```python
# --- 3. Medical Image Analysis ---

# Step 1: Simulate output from 3D Segmentation and Registration models.
cv_model_output_medical = {
    "patient_id": "P-10358",
    "scan_date_current": "2025-08-30",
    "scan_date_previous": "2025-02-28",
    "findings": [
        {"finding_id": "lesion_A", "location": "liver_segment_6", "type": "known", "volume_current_cm3": 15.3, "volume_previous_cm3": 12.1, "change_percent": 26.4},
        {"finding_id": "lesion_B", "location": "liver_segment_4", "type": "new", "size_mm": 5.1, "description": "hypodense nodule"}
    ]
}

# Step 2: The "Translation Layer" function.
def translate_medical_data_to_text(data):
    summary = f"Comparison report for Patient ID: {data['patient_id']}.\n"
    summary += f"Current scan date: {data['scan_date_current']}. Previous scan date: {data['scan_date_previous']}.\n\n"
    summary += "Quantitative Findings:\n"
    for finding in data['findings']:
        if finding['type'] == 'known':
            summary += f"- Known lesion (ID {finding['finding_id']}) in {finding['location']} has changed volume from {finding['volume_previous_cm3']} cm³ to {finding['volume_current_cm3']} cm³ (a {finding['change_percent']}% increase).\n"
        elif finding['type'] == 'new':
            summary += f"- A new {finding['description']} (ID {finding['finding_id']}) of {finding['size_mm']}mm has been detected in {finding['location']}.\n"
    return summary

# Step 3: Define the VLM's task.
user_command_medical = "Generate the 'Findings' section of a radiology report in a professional, human-readable format."

# Step 4: Qwen2-VL's Core Function: Professional Report Generation
context_medical = translate_medical_data_to_text(cv_model_output_medical)
print("--- Input to Qwen2-VL (Medical Imaging) ---")
print(context_medical)
print(f"User Prompt: \"{user_command_medical}\"")

# Get the report section from the VLM
report_text = get_qwen_reasoning_response(context_medical, user_command_medical)

print("\n--- Output from Qwen2-VL ---")
print("Function: Professional Report Generation")
print(f"Generated Report Section:\n{report_text}")
```
---

### **4. 擴增實境與空間計算 (AR & Spatial Computing)**

為了讓虛擬物體與真實世界無縫融合，系統需要對三維空間有即時、持久的理解。

- **潛力任務**: 使用者戴上 AR 眼鏡，用自然語言與周圍環境互動。例如：「幫我把這個虛擬沙發，靠著客廳最長的那面空牆擺放。」
    
- **VLM 的局限**: VLM 看到的只是一連串 2D 影片幀，它沒有關於這個房間的**幾何結構**、**表面材質**和**空間深度**的記憶。它不知道哪裡是「牆」，哪面牆「最長」，以及牆上是否「空著」。
    
- **需配合的 CV 模型**:
    
    - **SLAM (同步定位與地圖構建)**: 即時追蹤使用者在空間中的位置，並同時建立一個稀疏的 3D 地圖。
        
    - **即時 3D 場景重建 (Real-time 3D Reconstruction)**: 使用 **NeRFs** 或 **Gaussian Splatting** 等技術，將使用者看到的場景轉換成一個高密度的、帶有紋理的 3D 模型。
        
    - **平面檢測與語義分割模型 (Plane Detection & Semantic Segmentation)**: 在 3D 模型中識別出哪些是「牆面」、「地板」、「天花板」、「窗戶」等帶有語義的平面。
        
- **協作模式**:
    
    1. 使用者戴上眼鏡四處走動，**SLAM** 和 **3D 重建模型**在後台持續工作，建立並更新一個房間的 3D 數字孿生模型。
        
    2. **平面檢測模型**分析這個 3D 模型，輸出語義標籤：`{"plane_1": {"type": "wall", "length": "5.2m", "is_empty": true}, "plane_2": {"type": "wall", "length": "3.4m", "is_empty": false}}`。
        
    3. 使用者發出指令：「把沙發放在最長的空牆邊」。
        
    4. **VLM** 接收指令和從 CV 模型得到的場景語義數據。它進行推理，找到 `plane_1` 是最符合條件的。
        
    5. VLM 輸出指令給 AR 渲染引擎：「將 `virtual_sofa.asset` 放置在 `plane_1` 的中心位置。」
        

```python
# --- 4. Augmented Reality & Spatial Computing ---

# Step 1: Simulate output from real-time 3D Reconstruction and Plane Detection models.
cv_model_output_ar = {
    "scene_id": "living_room_01",
    "surfaces": [
        {"surface_id": "plane_1", "type": "wall", "is_vertical": True, "dimensions_m": [5.2, 2.8], "is_empty": True, "center_pos": [0.0, 1.4, -3.0]},
        {"surface_id": "plane_2", "type": "wall", "is_vertical": True, "dimensions_m": [3.4, 2.8], "is_empty": False, "contains": ["window_1"]},
        {"surface_id": "plane_3", "type": "floor", "is_vertical": False, "dimensions_m": [5.2, 3.4], "is_empty": False, "contains": ["table_1", "rug_1"]}
    ]
}

# Step 2: The "Translation Layer" function.
def translate_ar_data_to_text(data):
    summary = f"Semantic map of scene '{data['scene_id']}'. Detected surfaces:\n"
    for surface in data['surfaces']:
        summary += f"- Surface ID {surface['surface_id']} is a {surface['type']}.\n"
        summary += f"  - Dimensions (L, W): {surface['dimensions_m']} meters.\n"
        summary += f"  - Is empty: {surface['is_empty']}.\n"
    return summary

# Step 3: Define the user's command.
user_command_ar = "I want to place a large virtual sofa. Find the best wall for it and suggest a placement position."

# Step 4: Qwen2-VL's Core Function: Spatial Semantic Reasoning
context_ar = translate_ar_data_to_text(cv_model_output_ar)
print("--- Input to Qwen2-VL (Augmented Reality) ---")
print(context_ar)
print(f"User Prompt: \"{user_command_ar}\"")

# Get the placement recommendation from the VLM
recommendation = get_qwen_reasoning_response(context_ar, user_command_ar)

print("\n--- Output from Qwen2-VL ---")
print("Function: Spatial Semantic Reasoning")
print(f"Generated Placement Recommendation:\n{recommendation}")
```