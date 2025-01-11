
如果我們想保持輸入影像的某些部分不變，但在其他部分產生新的東西怎麼辦？這稱為“修復(inpainting)”。雖然可以使用與先前的演示相同的模型（透過 StableDiffusionInpaintPipelineLegacy）來完成，但我們可以透過使用穩定擴散的自訂微調版本(custom fine-tuned)來獲得更好的結果，該版本將masked image和mask本身作為附加調節。masked image應與輸入影像形狀相同，要替換的區域為白色，且要保持不變的區域為黑色。以下是我們如何加載這樣的管道並將其應用到“設定”部分中加載的image和mask：

```python hlredt:inpaintingpipe
# We'll be exploring a number of pipelines today!
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionDepth2ImgPipeline)

# Load the inpainting pipeline (requires a suitable inpainting model)
inpaintingpipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
inpaintingpipe = inpaintingpipe.to(device)
```

```python hlredt:inpaintingpipe
# Inpaint with a prompt for what we want the result to look like
prompt = "A small robot, high resolution, sitting on a park bench"
image = inpaintingpipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]

# View the result
fig, axs = plt.subplots(1, 3, figsize=(16, 5))
axs[0].imshow(init_image);axs[0].set_title('Input Image')
axs[1].imshow(mask_image);axs[1].set_title('Mask')
axs[2].imshow(image);axs[2].set_title('Result');
```

當與另一個模型結合自動產生mask時，這會特別強大。例如，此示範空間使用名為 CLIPSeg 的模型來根據文字描述屏蔽要替換的物件。
[stable-diffusion-inpainting huggingface notebook](https://huggingface.co/spaces/multimodalart/stable-diffusion-inpainting/blob/main/clipseg/Quickstart.ipynb)
[clipseg github](https://github.com/timojl/clipseg)