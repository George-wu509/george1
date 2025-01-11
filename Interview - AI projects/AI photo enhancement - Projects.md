
#### Title:
Text based Image enhancement Multimodal generative model


---
##### **Resume Keyworks:
<mark style="background: #FF5582A6;">Stable Diffusion</mark>, <mark style="background: #FF5582A6;">ControlNet</mark>, <mark style="background: #ADCCFFA6;">Real-ESRGAN</mark>, <mark style="background: #FF5582A6;">LLAMA</mark>

##### **STEPS:**
step1. 系統利用LLAMA模型，結合自定義的對話模板（chat_template），將文字描述轉化為ControlNet ID(表示需要用哪一個ControlNet模型), Prompt(生成圖像的正向提示詞), Negative Prompt(避免生成不需要的內容的反向提示詞)

step2. 根據LLAMA模型的輸出ControlNet ID，對原始圖像進行適合的預處理
	Canny Edge Detection（邊緣檢測）：用於增強輪廓信息。
	Normal Map（法線圖）：用於處理三維結構和光線信息。
	Depth Map（深度圖）：用於識別物體與場景的距離信息。
	Segmentation Map（分割圖）：用於區分不同的物體或場景區域。

step3. 根據LLAMA模型的輸出載入Model
	適合版本的Stable diffusion model
	適合版本的ControlNet Model
	StableDiffusionControlNetPipeline
	Pipeline Scheduler

step4. 利用Stable Diffusion模型和ControlNet模型進行圖像生成
1. 使用從LLAMA獲得的**Prompt**、**Negative Prompt**，並設置生成參數如**Steps**（生成步數）、**Generator**（隨機數生成器）
2. 結合圖像預處理結果（如Canny圖、Depth圖）作為ControlNet的控制輸入，增強生成的準確性。
3. 需要超分辨率處理（**Super-Resolution**），則將生成的圖像傳入**Real-ESRGAN模型**進行高質量放大。


---
#### Resume: 
Designed a text-based AI image enhancement system supporting tasks like denoising, sharpening, super-resolution, inpainting, and object/background manipulation. Utilized LLAMA for text-to-prompt conversion, Stable Diffusion with ControlNet for image processing, and Real-ESRGAN for high-quality outputs, ensuring flexibility and precision across diverse applications.

#### Abstract: 
This project proposes a text-based AI image enhancement system that allows users to enhance input images based on natural language instructions. The system supports multiple enhancement applications, including denoising, sharpening, inpainting, deblurring, super-resolution, white balance correction, HDR application, quality improvement, colorization, light correction, object removal, repositioning, addition, and background regeneration. It leverages the LLAMA model combined with a custom chat template to translate text inputs into ControlNet IDs, prompts, and negative prompts. The input images undergo preprocessing (e.g., canny edge detection, depth maps, segmentation maps) based on LLAMA outputs. The system integrates Stable Diffusion and ControlNet models through a StableDiffusionControlNetPipeline with customized schedulers to generate enhanced outputs. For super-resolution tasks, the Real-ESRGAN model is employed to upscale image quality. This pipeline ensures accurate and user-specific image enhancements through efficient prompt generation, preprocessing, and model integration. The system demonstrates high adaptability for applications in creative media, medical imaging, and automated image enhancement.

About SD and controlnet 
[github](https://github.com/RedDeltas/SDForge-Colab/blob/main/RedDeltasSDForge.ipynb)   [mycolab](https://colab.research.google.com/drive/1MPg0ucbvTwX_RQOOSYkN8_-0zH8u50Mc#scrollTo=rEeZ9in5tBhT)   [Run ControlNet Models on Stable Diffusion WebUI](https://github.com/brevdev/notebooks/blob/main/controlnet.ipynb)

About applications:
[photo-enhancer](https://github.com/nuwandda/photo-enhancer)
[https://deep-image.ai/](https://deep-image.ai/)

#### Technique detail: 

建立一個基於文字指令的 AI 圖像增強系統是一個具有挑戰性但非常創新的項目。以下是該項目的原理和流程細節的詳細中文解釋（重要的英文名詞以中英文表示）：

---

### **系統目標**

建立一個**Text-Based AI Image Enhancement System**，通過文字輸入（Text Input）來自動進行圖像增強（Image Enhancement）。增強功能包括：

- **Denoise（去噪）**
- **Sharpen（銳化）**
- **Image Inpainting（圖像修補）**
- **Deblur（去模糊）**
- **Super-Resolution（超分辨率）**
- **Fix White Balance（修正白平衡）**
- **Apply to HDR（應用高動態範圍處理）**
- **Improve Image Quality（提升圖像質量）**
- **Colorize（圖像上色）**
- **Correct Light（光線校正）**
- **Remove Multi Object（移除多物體）**
- **Reposition Multi Object（重新定位多物體）**
- **Add Multi Object（新增多物體）**
- **Remove and Regenerate Background（移除並重建背景）**

---

### **系統設計與流程**

#### **1. 輸入文字處理（Text Input Processing）**

1. 用戶輸入增強需求的文字描述（例如："去掉背景，並將圖像增強至高分辨率"）。
2. 系統利用**LLAMA模型**，結合自定義的對話模板（**chat_template**），將文字描述轉化為：
    - **ControlNet ID**（表示需要用哪一個ControlNet模型）
    - **Prompt**（生成圖像的正向提示詞）
    - **Negative Prompt**（避免生成不需要的內容的反向提示詞）

#### **2. 圖像前處理（Image Preprocessing）**

根據LLAMA模型的輸出，對原始圖像進行適合的預處理：

- **Canny Edge Detection（邊緣檢測）**：用於增強輪廓信息。
- **Normal Map（法線圖）**：用於處理三維結構和光線信息。
- **Depth Map（深度圖）**：用於識別物體與場景的距離信息。
- **Segmentation Map（分割圖）**：用於區分不同的物體或場景區域。

這些處理步驟由對應的圖像處理模塊（如OpenCV或深度學習模型）實現。

#### **3. 模型加載與配置（Model Loading and Configuration）**

根據輸入指令與前處理結果，載入以下模型：

- **Stable Diffusion Model（穩定擴散模型）**：用於圖像生成與增強。
- **ControlNet Model（控制網路模型）**：輔助Stable Diffusion進行特定圖像生成任務。
- **StableDiffusionControlNetPipeline**：集成Stable Diffusion與ControlNet的管道。
- **Pipeline Scheduler（管道調度器）**：決定生成過程的步驟（Steps）與隨機性。

#### **4. 圖像增強與生成（Image Enhancement and Generation）**

1. 利用Stable Diffusion模型和ControlNet模型進行圖像生成：
    - 使用從LLAMA獲得的**Prompt**、**Negative Prompt**，並設置生成參數如**Steps**（生成步數）、**Generator**（隨機數生成器）。
    - 結合圖像預處理結果（如Canny圖、Depth圖）作為ControlNet的控制輸入，增強生成的準確性。
2. 如果需要超分辨率處理（**Super-Resolution**），則將生成的圖像傳入**Real-ESRGAN模型**進行高質量放大。

#### **5. 輸出圖像（Output Generation）**

生成增強後的圖像（**Generated Output**），並返回給用戶。支持保存輸出或進行下一步增強操作。

---

### **關鍵模塊說明**

#### **LLAMA模型與對話模板**

- **LLAMA**是一種基於Transformer架構的大型語言模型，負責將自然語言轉換為具體的圖像生成參數。
- 自定義**chat_template**用於解析輸入文字，產生與圖像增強需求相符的模型配置。

#### **Stable Diffusion與ControlNet**

- **Stable Diffusion**是一種基於擴散模型的生成方法，適用於圖像生成與增強任務。
- **ControlNet**提供控制機制，使生成的圖像能夠精準匹配特定的內容需求（例如邊緣、深度或分割信息）。

#### **Real-ESRGAN**

- 用於圖像超分辨率，實現高質量的細節恢復和圖像放大。

#### **Pipeline Scheduler**

- 調整生成步驟和隨機性，確保生成結果的質量與穩定性。
