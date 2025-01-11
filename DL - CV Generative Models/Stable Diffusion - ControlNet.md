[Controlnet.ipynb](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/controlnet.ipynb)

自從Stable Diffusion席捲全球以來，人們一直在尋找對生成過程結果有更多控制的方法。 ControlNet 提供了一個最小的介面，允許使用者在很大程度上自訂生成過程。透過 ControlNet，使用者可以輕鬆地使用不同的空間情境（例如深度圖depth map、分​​割圖segmentation map、塗鴉scribble、關鍵點keypoints等）來調節生成！

引入了一個框架framework，允許支援各種空間上下文spatial contexts，這些空間上下文可以作為擴散模型（例如穩定擴散）的附加條件additional conditionings。


|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **[CASE 1] Basic - Controlnet + SD**<br><br>1. Import<br>from diffusers import StableDiffusionControlNetPipeline, ControlNetModel<br><br>2. Input Image<br>canny_image = cv2.Canny(image, low_th, high_th)<br><br>3. Load ControlNet<br>controlnet = ControlNetModel.from_pretrained("==sd-controlnet-canny==")<br><br>4. Crate pipeline<br>pipe = StableDiffusionControlNetPipeline.from_pretrained(<br>    "stable-diffusion-v1-5", controlnet)<br><br>5. Run the pipeline<br>output = pipe(prompt, canny_image, negative_prompt, generator, steps)<br><br>                                                                                                            |
| **[CASE 2] Finetune - 用finetune的diffusion model** <br><br>1. Import<br>from diffusers import StableDiffusionControlNetPipeline, ControlNetModel<br><br>2. Input Image<br>canny_image = cv2.Canny(image, low_th, high_th)<br><br>3. Load ControlNet<br>controlnet = ControlNetModel.from_pretrained("==sd-controlnet-canny==")<br><br>4. Crate pipeline<br>pipe = StableDiffusionControlNetPipeline.from_pretrained(<br>    =="mr-potato-head"==, controlnet)<br><br>5. Run the pipeline<br>output = pipe(prompt, canny_image, negative_prompt, generator, steps)<br>                                                                                                     |
| **[CASE 3] Open pose controlnet** <br><br>1. Import<br>from diffusers import StableDiffusionControlNetPipeline, ControlNetModel<br>==from controlnet_aux import OpenposeDetector==<br><br>2. Input Image<br>==model = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")==<br>==poses = model(image)==<br><br>3. Load ControlNet<br>controlnet = ControlNetModel.from_pretrained("==stable-diffusion-v1-5-controlnet-openpose==")<br><br>4. Crate pipeline<br>pipe = StableDiffusionControlNetPipeline.from_pretrained(<br>    "stable-diffusion-v1-5", controlnet)<br><br>5. Run the pipeline<br>output = pipe(prompt, poses, negative_prompt, generator, steps) |
|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |



訓練 ControlNet 包含以下步驟：

1. Cloning Diffusion模型的預訓練參數(pre-trained parameters)，例如 Stable Diffusion 的latent UNet（稱為「可訓練副本」(trainable copy)），同時也單獨維護預訓練參數（「鎖定副本」(locked copy)）。這樣做是為了使鎖定的參數副本(locked parameter copy)可以保留從大型資料集中學到的大量知識，而可訓練的副本(trainable copy)則用於學習特定於任務的方面。

2. 參數的可訓練(trainable copy)和鎖定副本(locked parameter copy)透過「零卷積」層(zero convolution layers)，這些層作為 ControlNet 框架的一部分進行了最佳化。這是一種訓練技巧(training trick)，用於在訓練新條件時保留凍結模型(frozen model)已經學到的語義(semantics)。

每一種新的調節類型(conditioning)都需要訓練一個新的ControlNet weights。論文提出了 8 種不同的調節模型(conditioning models)，Diffusers 都支持這些模型！為了進行推理，需要預先訓練的擴散模型權重(pre-trained diffusion models weights)以及經過訓練的 ControlNet 權重(trained ControlNet weights)。例如，與僅使用原始穩定擴散模型相比，將Stable Diffusion v1-5 與 ControlNet 檢查點結合使用大約需要 7 億個參數，這使得 ControlNet 的推理記憶體消耗更大。由於在訓練過程中會查看預先訓練的擴散模型(pre-trained diffusion models)，因此在使用不同的條件時只需切換 ControlNet 參數。這使得在一個應用程式中部署多個 ControlNet 權重變得相當簡單，

### The StableDiffusionControlNetPipeline

#### 1. Import 
```python hlredt:ControlNetModel hlwhitet:pipe hlbluet:prompt
pip install -q diffusers==0.14.0 transformers xformers git+https://github.com/huggingface/accelerate.git

pip install -q opencv-contrib-python
pip install -q controlnet_aux

# Import
from diffusers import StableDiffusionControlNetPipeline
from diffusers.utils import load_image
```
To process different conditionings depending on the chosen ControlNet, we also need to install some additional dependencies:
- [OpenCV](https://www.google.com/url?q=https%3A%2F%2Fopencv.org%2F)
- [controlnet-aux](https://github.com/patrickvonplaten/controlnet_aux#controlnet-auxiliary-models) - a simple collection of pre-processing models for ControlNet

#### 2. Load controlnet and stable diffusion pipeline
```python hlredt:ControlNetModel hlwhitet:pipe hlbluet:prompt
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch

# controlnet
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)

# stable diffusion pipeline
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16)
)
```


#### 3. The scheduler and optimize
```python hlredt:hlredt:ControlNetModel hlwhitet:pipe hlbluet:prompt
from diffusers import UniPCMultistepScheduler

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# smart CPU offloading
pipe.enable_model_cpu_offload()

# attention layer acceleration
pipe.enable_xformers_memory_efficient_attention()
```
我們不使用 Stable Diffusion 的預設 PNDMScheduler，而是使用目前最快的擴散模型調度程序之一，稱為 UniPCMultistepScheduler。選擇改進的調度程序可以大大減少推理時間 - 在我們的例子中，我們能夠將推理步驟數從 50 個減少到 20 個，同時或多或少保持相同的圖像生成品質。

我們不是將管道直接載入到 GPU而是啟用智慧 CPU 卸載, 這可以透過 enable_model_cpu_offload 函數來實現。在使用 ControlNet 進行穩定擴散的情況下，我們首先使用 CLIP 文字編碼器，然後使用擴散模型unet 和控製網絡，然後使用 VAE 解碼器，最後運行安全檢查器

大多數組件在擴散過程中只運行一次，因此不需要一直佔用 GPU 記憶體。透過啟用智慧模型卸載，我們確保每個元件僅在需要時才載入到 GPU 中，這樣我們就可以顯著節省記憶體消耗，而不會顯著減慢攻擊速度

#### 4. run the ControlNet pipeline
```python hlredt:ControlNetModel hlwhitet:pipe hlbluet:prompt

prompt = ", best quality, extremely detailed"
prompt = [t + prompt for t in ["Sandra Oh", "Kim Kardashian", "rihanna", "taylor swift"]]
generator = [torch.Generator(device="cpu").manual_seed(2) for i in range(len(prompt))]

output = pipe(prompt,canny_image,
    negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] * len(prompt),
    generator=generator,
    num_inference_steps=20,)

```
prompt = [
'Sandra Oh, best quality, extremely detailed',
'Kim Kardashian, best quality, extremely detailed',
'rihanna, best quality, extremely detailed', 
'taylor swift, best quality, extremely detailed']

我們仍然提供提示(prompt)來指導影像生成過程(image generation process)，就像我們通常使用穩定擴散影像到影像管道(Stable Diffusion image-to-image pipeline)所做的那樣。然而，ControlNet 將允許對生成的圖像進行更多的控制，因為我們將能夠使用我們剛剛創建的精明邊緣圖像(canny edge image)來控制生成圖像的精確組成。

看到一些當代名人為這幅 17 世紀的畫作擺出姿勢的照片將會很有趣。使用 ControlNet 做到這一點非常容易，我們所要做的就是在提示中包含這些名人的名字！

#### 5. ControlNet with finetune
```python hlredt:ControlNetModel hlwhitet:pipe hlbluet:prompt
model_id = "sd-dreambooth-library/mr-potato-head"
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id,
    controlnet=controlnet,
    torch_dtype=torch.float16,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()

generator = torch.manual_seed(2)
prompt = "a photo of sks mr potato head, best quality, extremely detailed"
output = pipe(
    prompt,
    canny_image,
    negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
    generator=generator,
    num_inference_steps=20,
)
```
我們也可以毫不費力地將 ControlNet 與微調(fine-tuning)結合！例如，我們可以使用fine-tune a model with DreamBooth，並使用它將自己渲染(render)到不同的場景中。我們可以使用相同的 ContrlNet，但我們不使用 Stable Diffusion 1.5，而是將 Mr Potato Head 模型加載到我們的管道中 - Mr Potato Head 是一個使用 Dreambooth 與 Mr Potato Head 概念進行微調的穩定擴散模型 🥔


ControlNet 的另一個獨特應用是，我們可以從一張圖像中獲取一個姿勢(pose)，然後重新使用它來產生具有完全相同姿勢的不同圖像。因此，在下一個範例中，我們將教導超級英雄如何使用 Open Pose ControlNet 進行瑜珈！
#### 6. Open Pose ControlNet
```python hlredt:ControlNetModel hlwhitet:pipe hlbluet:prompt
# Get yoga images for pose
urls = "yoga1.jpeg", "yoga2.jpeg", "yoga3.jpeg", "yoga4.jpeg"
imgs = [
    load_image("https://hf.co/datasets/YiYiXu/controlnet-testing/resolve/main/" + url)
    for url in urls]

# extract yoga poses using the OpenPose pre-processors
from controlnet_aux import OpenposeDetector

model = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
poses = [model(img) for img in imgs]

# Load controlnet
controlnet = ControlNetModel.from_pretrained(
    "fusing/stable-diffusion-v1-5-controlnet-openpose", torch_dtype=torch.float16
)
# Create StableDiffusionControlNetPipeline
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id,
    controlnet=controlnet,
    torch_dtype=torch.float16,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()

# Run the pipeline
generator = [torch.Generator(device="cpu").manual_seed(2) for i in range(4)]
prompt = "super-hero character, best quality, extremely detailed"
output = pipe(
    [prompt] * 4,
    poses,
    negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] * 4,
    generator=generator,
    num_inference_steps=20,
)
```