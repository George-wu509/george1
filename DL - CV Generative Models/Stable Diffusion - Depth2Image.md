
Img2Img 很棒，但有時我們想用原始圖像的構圖但完全不同的顏色colours 或紋理textures來創建新圖像。要找到 Img2Img 強度來保留我們想要的佈局而不同時保留輸入顏色可能很困難。

是時候進行另一個微調模型了！這個在生成時將深度資訊(in depth information)作為附加條件。該管道pipeline使用深度估計模型(depth estimation model)來創建深度圖depth map，然後在生成圖像時將其饋送到經過微調的 UNet，以（希望）保留初始圖像的深度和結構the depth and structure of the initial image，同時填充全新的內容。

```python hlredt:depth_imgpipe
# We'll be exploring a number of pipelines today!
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionDepth2ImgPipeline)

# Load the Depth2Img pipeline (requires a suitable model)
depth_imgpipe = StableDiffusionDepth2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-depth")
depth_imgpipe = depth_imgpipe.to(device)
```

```python hlredt:depth_imgpipe
# Inpaint with a prompt for what we want the result to look like
prompt = "An oil painting of a man on a bench"
image = depth_imgpipe(prompt=prompt, image=init_image).images[0]

# View the result
fig, axs = plt.subplots(1, 2, figsize=(16, 5))
axs[0].imshow(init_image);axs[0].set_title('Input Image')
axs[1].imshow(image);axs[1].set_title('Result');
```