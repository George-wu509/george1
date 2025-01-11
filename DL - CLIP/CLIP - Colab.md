
[CLIP github](https://github.com/openai/CLIP)
[CLIP colab link](https://colab.research.google.com/github/openai/clip/blob/master/notebooks/Interacting_with_CLIP.ipynb)

### 1. Import

```python hlt:clip
import numpy as np
import torch
from pkg_resources import packaging
import clip
clip.available_models()
```

### 2. Load CLIP

```python hlredt:model hlbluet:features
# Load CLIP and get CLIP model and preprocess
model, preprocess = clip.load("ViT-B/32")
model.cuda().eval()

# some CLIP setting
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size    
```

### 3. Get features from Image and Text
```python hlredt:model hlbluet:features
# description: {image name: text} pair
descriptions = {
    "page": "a page of text about segmentation"}
image = Image.open(filename).convert("RGB")

# image input and text tokens
image_input = torch.tensor(np.stack(images)).cuda()
text_tokens = clip.tokenize(["This is " + desc for desc in texts]).cuda()

# image and text features
image_features = model.encode_image(image_input).float()
text_features = model.encode_text(text_tokens).float()

```

