Img2Img pipeline首先將現有影像encodes為一組潛在影像(a set of latents)，然後在潛在影像中添加一些雜訊並將其用作起點。1. 添加的雜訊量amount of noise和2. 應用的去雜訊步驟數 the number of denoising steps決定了 img2img 製程的「強度」'strength'。

僅添加少量雜訊（低強度）將導致很小的變化，而添加最大量的雜訊並運行完整的去雜訊過程將給出除了整體結構上的一些相似之處之外幾乎與輸入幾乎不相似的影像。

```PYTHON hlredt:img2img_pipe
# We'll be exploring a number of pipelines today!
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionDepth2ImgPipeline)

# Loading an Img2Img pipeline
model_id = "stabilityai/stable-diffusion-2-1-base"
img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id).to(device)
```

1. Generating Image from image
```PYTHON hlredt:img2img_pipe hlbluet:init_image
# Apply Img2Img
result_image = img2img_pipe(
    prompt="An oil painting of a man on a bench",
    image=init_image, # The starting image
    strength=0.6, # 0 for no change, 1.0 for max strength
).images[0]

# View the result
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].imshow(init_image);axs[0].set_title('Input Image')
axs[1].imshow(result_image);axs[1].set_title('Result');
```