
My Qwen2-VL [Colab](https://colab.research.google.com/drive/1Zahrn91uzsndMvaLefk8xQot4qsAQgIS?usp=sharing#scrollTo=N-kIKVdhxczd)

Original Qwen2-VL [Colab](https://colab.research.google.com/drive/1Zahrn91uzsndMvaLefk8xQot4qsAQgIS?usp=sharing)


圖像/影片描述, 視覺問答 (VQA), 多模態推理, 視覺定位 (Visual Grounding), 物件偵測/分割, 影片描述


|                         |     |
| ----------------------- | --- |
| 圖像/影片描述                 |     |
| 視覺問答 (VQA)              |     |
| 多模態推理                   |     |
| 視覺定位 (Visual Grounding) |     |
| 物件偵測/分割                 |     |
| 物件追蹤 (Object Tracking)  |     |
| 影片描述                    |     |
|                         |     |
|                         |     |




[spatial_understanding](https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/spatial_understanding.ipynb).ipynb
```python
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
model_path = "Qwen/Qwen2.5-VL-7B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",device_map="auto")
processor = AutoProcessor.from_pretrained(model_path)


```
