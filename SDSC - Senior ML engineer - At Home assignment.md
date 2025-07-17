

|                                                                                                 |     |
| ----------------------------------------------------------------------------------------------- | --- |
| 1. Python implementation                                                                        |     |
| 2. Embeddings with VideoMAE-2                                                                   |     |
| 3. BONUS: Use the VideoMAE model to predict the outcome of the procedure from the SOCAL dataset |     |

#### 1. Python implementation
dataset: dataset_SOCAL_small_demo
https://drive.google.com/drive/folders/1o8T2rB7Z4lxolHQNtF2-UHHZ9yvlwv9J

ToDO:
1. Put imports you need here --> **add imports**
```python
from google.colab import drive
drive.mount('/content/drive')

import glob
import os
```
2. Check how many images in dataset --> **add codes**
```python
# explore dataset
```
3. Display bounding boxes on images --> **add codes**
```python

```
Create a function that selects 10 random from the dataset, displays the images and overlays the bounding boxes on the frames.
![[downloadsdfvcdftr.png]]

#### 2. Embeddings with VideoMAE-2

dataset: https://drive.google.com/drive/folders/1LVHGKiLZvyFoRh3PGqnBGilH2_rTCAUn

下載我們準備的小型手術影片示範資料集。此資料集包含 4 個視頻，分別來自兩種手術類型：腦下垂體瘤手術和膽囊切除術。Download the small demo dataset of surgical videos we have prepared. This dataset has 4 videos from 2 procedure types: Pituitary Tumor Surgery and Cholecystectomy.

膽囊切除術影片來自 cholec80 資料集。該資料集是一個內視鏡視訊資料集，包含 13 位外科醫生實施的 80 段膽囊切除術影片。這些影片以 25 fps 的幀率拍攝，並經過降採樣至 1 fps 進行處理。整個資料集都標註了相位和工具存在性。相位由法國斯特拉斯堡醫院的資深外科醫生定義。由於有時影像中的工具幾乎不可見，難以透過視覺識別，因此，如果至少一半的刀尖可見，則將工具定義為存在於影像中。The Cholecystectomy videos come from the cholec80 dataset is an endoscopic video dataset containing 80 videos of cholecystectomy surgeries performed by 13 surgeons. The videos are captured at 25 fps and downsampled to 1 fps for processing. The whole dataset is labeled with the phase and tool presence annotations. The phases have been defined by a senior surgeon in Strasbourg hospital, France. Since the tools are sometimes hardly visible in the images and thus difficult to be recognized visually, a tool is defined as present in an image if at least half of the tool tip is visible.

ToDO:
1. Check that you have 4 videos in the dataset, explore the structure of the dataset --> **add codes**
```python
# explore dataset
```
1. Display some random frames from the videos --> **add codes**
```python

```
1. visualize the embeddings graph of the videos from the surgical videos dataset --> **add codes**
```python
!pip install transformers && pip install av
```

```python
# imports you might need for that section

import numpy as np
from numpy.linalg import norm
import torch
import av
from transformers import AutoImageProcessor, VideoMAEModel
import torchvision.transforms as transforms
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
```

```python
from transformers import AutoImageProcessor, VideoMAEModel

image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
model_videomae_base = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base", output_hidden_states=True)
```

使用 huggingface 的 VideoMAE-2 模型：來視覺化手術影片資料集中影片的嵌入圖。請參閱 huggingface 頁面中的範例部分，以了解一些實用函數。
https://huggingface.co/docs/transformers/model_doc/videomae
to visualize the embeddings graph of the videos from the surgical videos dataset. See the Examples section in the huggingface page to find useful functions.

一些幫助：您需要採樣 16 幀，正如 videomae 論文中所述：「我們的主幹模型是 16 幀的 vanilla ViT-B」。You need to sample 16 frames as mentioned in the videomae paper: "Our backbone is 16-frame vanilla ViT-B".

您可以設定採樣率：每 x 幀採樣 1 幀。You can have a sample rate: sample frames every x frames


#### 3. BONUS: Use the VideoMAE model to predict the outcome of the procedure from the SOCAL dataset

```python

```

您可以在 socal_trial_outcomes.csv 檔案中找到「success」欄位。建立一個用於訓練 videomae 模型的機器學習資料集，並執行訓練 + 評估。
You can find the column "success" in the file socal_trial_outcomes.csv. Create an ML dataset ready for training the videomae model and run the training + Evaluation.
![[downloadrtdfhwt.png]]

您可以在此處找到完整的 SOCAL 資料集： You can find the full SOCAL dataset here:
https://www.google.com/url?q=https%3A%2F%2Fdrive.google.com%2Fdrive%2Ffolders%2F1xGcGkbj34wgETuzSafa5hZw5WAdRdtG4%3Fusp%3Dsharing


Some code to help get started with the training can be found on huggingface page: [https://huggingface.co/docs/transformers/v4.34.1/en/model_doc/videomae#transformers.VideoMAEForVideoClassification](https://www.google.com/url?q=https%3A%2F%2Fhuggingface.co%2Fdocs%2Ftransformers%2Fv4.34.1%2Fen%2Fmodel_doc%2Fvideomae%23transformers.VideoMAEForVideoClassification)

This notebook is also useful: [https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/video_classification.ipynb](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/video_classification.ipynb)



