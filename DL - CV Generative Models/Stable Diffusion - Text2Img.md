
[Colab 3 (Teach)](https://colab.research.google.com/drive/1wOIvZW4ic6LTg-VkPrSTFuhotJ-bVJnj)
#### 1. Import
```python
import torch
import requests
from PIL import Image
from io import BytesIO
from matplotlib import pyplot as plt

# We'll be exploring a number of pipelines today!
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionDepth2ImgPipeline)
```

#### 2. Generating Images from Text
```python hlredt:pipe(
# Load the pipeline
model_id = "stabilityai/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)

# Set up a generator for reproducibility
generator = torch.Generator(device=device).manual_seed(42)

# Run the pipeline, showing some of the available arguments
pipe_output = pipe(
    prompt="Palette knife painting of an autumn cityscape", # What to generate
    negative_prompt="Oversaturated, blurry, low quality", # What NOT to generate
    height=480, width=640,     # Specify the image size
    guidance_scale=8,          # How strongly to follow the prompt
    num_inference_steps=35,    # How many steps to take
    generator=generator        # Fixed random seed)

# View the resulting image
pipe_output.images[0]
```
- **主要參數**：
    - `prompt` 和 `negative_prompt`：分別描述希望生成的內容和不希望生成的內容。
    - `height` 和 `width`：生成圖像的分辨率。
    - `guidance_scale`：控制生成的圖像是否更接近提示的語意，高值強調提示，低值允許更多創造性自由。
    - `num_inference_steps`：迭代步數，越高生成的圖像越精緻，但代價是計算時間更長。
    - `generator`：用於設置隨機種子，使生成的圖像可重複。
- **流程**：
    - 提示語會自動通過 tokenizer 編碼。
    - 隨機起始點（latent）會初始化。
    - 使用默認的調度器和步數迭代生成圖像。
    - 最終生成圖像會自動解碼並返回。
- **適用場景**： 適合快速生成圖像，不需要對生成過程進行微調。
#### 3a. Component - VAE
```python hlredt:pipe hlbluet:vae
# Create some fake data (a random image, range (-1, 1))
images = torch.rand(1, 3, 512, 512).to(device) * 2 - 1
print("Input images shape:", images.shape)

# Encode to latent space
with torch.no_grad():
  latents = 0.18215 * pipe.vae.encode(images).latent_dist.mean
print("Encoded latents shape:", latents.shape)

# Decode again
with torch.no_grad():
  decoded_images = pipe.vae.decode(latents / 0.18215).sample
print("Decoded images shape:", decoded_images.shape)
```
Input images shape: torch.Size([1, 3, 512, 512])
Encoded latents shape: torch.Size([1, 4, 64, 64])
Decoded images shape: torch.Size([1, 3, 512, 512])

#### 3b. Component - Tokenizer and text encoder
```python hlredt:pipe hlbluet:tokenizer
# Tokenizing and encoding an example prompt manually

# Tokenize
input_ids = pipe.tokenizer(["A painting of a flooble"])['input_ids']
print("Input ID -> decoded token")
for input_id in input_ids[0]:
  print(f"{input_id} -> {pipe.tokenizer.decode(input_id)}")

# Feed through CLIP text encoder
input_ids = torch.tensor(input_ids).to(device)
with torch.no_grad():
  text_embeddings = pipe.text_encoder(input_ids)['last_hidden_state']
print("Text embeddings shape:", text_embeddings.shape)



# Get the final text embeddings using the pipeline's encode_prompt function
text_embeddings = pipe._encode_prompt("A painting of a flooble", device, 1, False, '')
text_embeddings.shape

```
input_ids = \[[49406, 320, 3086, 539, 320, 4062, 1059, 49407]]  
Input ID -> decoded token 
49406 -> <|startoftext|> 
320 -> a 
3086 -> painting 
539 -> of 
320 -> a 
4062 -> floo 
1059 -> ble 
49407 -> <|endoftext|> 
input_ids after todevice = tensor([[49406, 320, 3086, 539, 320, 4062, 1059, 49407]]) 
Text embeddings shape: torch.Size([1, 8, 1024])

torch.Size([1, 77, 1024])
- `pipe._encode_prompt` 是一個高級封裝，內部會自動調用 `pipe.tokenizer`，並處理文本編碼和向量化的完整流程。
- 它提供了一個更簡化的接口，不需要手動處理 tokenization 或進一步的模型推斷。

- 如果目的是獲取文本嵌入向量，建議直接使用 `pipe._encode_prompt`，因為它更簡單且高效。
- `pipe.tokenizer` 和 `pipe.text_encoder` 提供了更靈活的控制，但需要手動處理多個步驟。

#### 3c. The UNet
```python hlredt:pipe
# Dummy inputs
timestep = pipe.scheduler.timesteps[0]
latents = torch.randn(1, 4, 64, 64).to(device)
text_embeddings = torch.randn(1, 77, 1024).to(device)

# Model prediction
with torch.no_grad():
  unet_output = pipe.unet(latents, timestep, text_embeddings).sample
print('UNet output shape:', unet_output.shape) # Same shape as the input latents
```
UNet output shape: torch.Size([1, 4, 64, 64])

#### 3d The Scheduler
```python hlredt:pipe

# Case1: The basic scheduler
plt.plot(pipe.scheduler.alphas_cumprod, label=r'$\bar{\alpha}$')
plt.xlabel('Timestep (high noise to low noise ->)');
plt.title('Noise schedule');plt.legend();


# Case2: The LMSDiscreteScheduler scheduler
from diffusers import LMSDiscreteScheduler

# Replace the scheduler
pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)

# Print the config
print('Scheduler config:', pipe.scheduler)

# Generate an image with this new scheduler
pipe(prompt="Palette knife painting of an winter cityscape", height=480, width=480,
     generator=torch.Generator(device=device).manual_seed(42)).images[0]
```
Scheduler config: LMSDiscreteScheduler {
  "_class_name": "LMSDiscreteScheduler",
  "_diffusers_version": "0.11.1",
  "beta_end": 0.012,
  "beta_schedule": "scaled_linear",
  "beta_start": 0.00085,
  "clip_sample": false,
  "num_train_timesteps": 1000,
  "prediction_type": "epsilon",
  "set_alpha_to_one": false,
  "skip_prk_steps": true,
  "steps_offset": 1,
  "trained_betas": null
}

#### 4 A DIY Sampling Loop
```python hlredt:pipe
guidance_scale = 8 #@param
num_inference_steps = 30 #@param
prompt = "Beautiful picture of a wave breaking" #@param
negative_prompt = "zoomed in, blurry, oversaturated, warped" #@param

# Encode the prompt
text_embeddings = pipe._encode_prompt(prompt, device, 1, True, negative_prompt)

# Create our random starting point
latents = torch.randn((1, 4, 64, 64), device=device, generator=generator)
latents *= pipe.scheduler.init_noise_sigma

# Prepare the scheduler
pipe.scheduler.set_timesteps(num_inference_steps, device=device)

# Loop through the sampling timesteps
for i, t in enumerate(pipe.scheduler.timesteps):

    # Expand the latents if we are doing classifier free guidance
    latent_model_input = torch.cat([latents] * 2)

    # Apply any scaling required by the scheduler
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

    # Predict the noise residual with the UNet
    with torch.no_grad():
       noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=   
          text_embeddings).sample

    # Perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # Compute the previous noisy sample x_t -> x_t-1
    latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

# Decode the resulting latents into an image
with torch.no_grad():
  image = pipe.decode_latents(latents.detach())
```

### **1. 設置參數**
```python
guidance_scale = 8  # 指導比例
num_inference_steps = 30  # 推理步數
prompt = "Beautiful picture of a wave breaking"  # 文本提示
negative_prompt = "zoomed in, blurry, oversaturated, warped"  # 負面提示
```
#### **解釋**
- **`guidance_scale`**：控制圖像生成時模型對提示語的依賴程度。
    - **低值**（如 2~5）：允許生成結果更加多樣化和創意化。
    - **高值**（如 10~20）：生成結果會更接近提示語，但可能犧牲圖像的多樣性。
- **`num_inference_steps`**：指定從隨機噪聲到最終圖像的迭代步數。
    - **步數多**：圖像細節更豐富，但生成速度慢。
    - **步數少**：生成速度快，但圖像質量可能較差。
#### **舉例**
如果你想生成一幅更符合描述的清晰海浪圖片，可以設置：
```python
`guidance_scale = 12 num_inference_steps = 50`
```

---

### **2. 對文本提示進行編碼**
```python
`text_embeddings = pipe._encode_prompt(prompt, device, 1, True, negative_prompt)`
```

#### **解釋**
- **`pipe._encode_prompt`**：將 `prompt` 和 `negative_prompt` 轉換為嵌入向量。
    
    - **`prompt`**：生成圖像的語義描述，例如 "Beautiful picture of a wave breaking"。
    - **`negative_prompt`**：生成圖像時避免的特性描述，例如 "zoomed in, blurry"。
- **參數詳解**：
    - `device`：指定執行設備（如 GPU）。
    - `1`：批量大小（這裡生成一幅圖像）。
    - `True`：啟用 classifier-free guidance（文本引導方式）。
    - `negative_prompt`：用於改進生成圖像的質量。
#### **具體數據轉換示例**

假設 `prompt` 是 `"Beautiful picture of a wave breaking"`，編碼後可能產生類似以下的嵌入向量：
```python
`tensor([[0.12, 0.34, 0.56, ...]])  # 嵌入向量的形狀類似 (1, 768)`
```

---

### **3. 初始化隨機潛在空間**

```python
latents = torch.randn((1, 4, 64, 64), device=device, generator=generator) 
latents *= pipe.scheduler.init_noise_sigma`
```
#### **解釋**

- **`torch.randn`**：生成形狀為 `(1, 4, 64, 64)` 的隨機噪聲。
    - **`1`**：批量大小（生成一幅圖像）。
    - **`4`**：潛在空間的通道數，與模型架構相關。
    - **`64x64`**：潛在特徵圖的空間維度，表示生成圖像的潛在分辨率。
- **`init_noise_sigma`**：初始化噪聲的標準差，用於調整隨機噪聲的尺度。

**舉例**
假設初始噪聲（latents）如下：
`tensor([[[[-0.8, 0.5, ...], [1.2, -0.3, ...]]]])  # 隨機噪聲的值`

---

### **4. 設置調度器**
```python
`pipe.scheduler.set_timesteps(num_inference_steps, device=device)`
```
#### **解釋**
- **`set_timesteps`**：設定生成過程的時間步數（共 `num_inference_steps` 步）。
- **調度器**：用於控制從隨機噪聲逐步還原到圖像的過程。
    - **常見調度器**：
        - DDIM（快速且效果好）
        - PNDM（進階多步調度器）
        - LMS（線性調度器）

---

### **5. 逐步更新潛在向量**

```python
for i, t in enumerate(pipe.scheduler.timesteps):
    latent_model_input = torch.cat([latents] * 2)  # 擴展為 2 倍
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)  # 時間步縮放
    with torch.no_grad():
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

```
#### **逐步解釋**
1. **`for` 循環**：對每個時間步 `t` 進行處理。
2. **`torch.cat([latents] * 2)`**：複製潛在向量，用於 classifier-free guidance。
3. **`scale_model_input`**：將潛在向量縮放到當前步驟需要的範圍。
4. **`pipe.unet`**：使用 UNet 模型預測當前噪聲。
5. **`noise_pred.chunk(2)`**：
    - `noise_pred_uncond`：未經提示影響的噪聲預測。
    - `noise_pred_text`：受到提示影響的噪聲預測。
6. **噪聲引導**：
```python
   `noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)`
```
    
    - 將未經提示的噪聲和提示影響的噪聲結合。
    - `guidance_scale` 決定提示的權重。
7. **更新潛在向量**：
```python
`latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample`
```
    使用調度器計算從當前噪聲到下一步的潛在向量。

#### **舉例**

假設有 30 個步驟，每次循環的潛在向量逐漸從隨機噪聲轉變為更具結構的數據。

---

### **6. 解碼潛在向量生成圖像**
```python
`with torch.no_grad():     
    image = pipe.decode_latents(latents.detach())`
```
#### **解釋**

- **`pipe.decode_latents`**：將潛在向量轉換為最終的圖像數據。
- **`detach()`**：從計算圖中分離，避免多餘的梯度計算。
#### **生成結果**

假設最終的 `image` 是 640x480 的圖像數據，可以保存或顯示：
`from PIL import Image Image.fromarray((image[0] * 255).astype("uint8")).save("output.png")`

---

### **總結**

這段代碼的每個步驟完整展示了 Stable Diffusion 的圖像生成流程，通過具體設置（如 `guidance_scale` 和 `num_inference_steps`）實現高靈活性。對於需要調整生成過程細節的用戶，這種方法是必須掌握的技術。