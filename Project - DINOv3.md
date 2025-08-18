
2025.08.14   DINOv3

我們推出了[DINOv3](http://ai.meta.com/dinov3)，它可以擴展影像的自監督學習，以創建通用視覺主幹，在包括網路和衛星影像在內的不同領域實現絕對最先進的性能。DINOv3 主幹網路能夠產生強大的高解析度影像特徵，從而輕鬆訓練輕量級適配器。這使得其在一系列下游視覺任務（包括影像分類、語義分割和影片中的物件追蹤）上表現出色。透過提供較小的模型來增強 DINOv3 的多功能性，這些模型在廣泛的評估套件中表現優於基於 CLIP 的同類衍生產品，以及針對資源受限用例的替代 ConvNeXt 架構

今天，我們發布了[DINOv3](http://ai.meta.com/dinov3)，這是一款通用的、先進的電腦視覺模型，採用 SSL 進行訓練，能夠產生卓越的高解析度視覺特徵。這是首次在多個長期存在的密集預測任務（包括物件偵測和語義分割）上，單一凍結視覺主幹網路的表現優於專用解決方案. 我們建立了 DINOv3，並在比其前身DINOv2大 12 倍的資料集上訓練了一個 7 倍大的模型。為了展示模型的多功能性，我們在 15 個不同的視覺任務和 60 多個基準測試中對其進行了評估。 DINOv3 主幹在所有密集預測任務中表現尤為出色，展現了對場景佈局和底層物理的卓越理解。

將 DINOv3 擴展到 7B 參數展現了 SSL 的全部潛力。然而，7B 模型對於許多下游應用而言並不實用。根據社群的回饋，我們建立了一系列涵蓋廣泛推理運算需求的模型，以賦能研究人員和開發者來應對各種用例。透過將 ViT-7B 模型提煉為更小、性能更高的變體（例如 ViT-B 和 ViT-L），DINOv3 在廣泛的評估套件中均優於基於 CLIP 的同類模型。此外，我們也引進了從 ViT-7B 提煉而來的 ConvNeXt 替代架構（T、S、B、L），以適應不同的運算限制。我們也發布了提煉流程，以便社群在此基礎上進行建置。

![[Pasted image 20250814203608.png]]

![[Pasted image 20250814203653.png]]

Reference: 
[1]
[DINOv3](https://ai.meta.com/blog/dinov3-self-supervised-vision-model/): Self-supervised learning for vision at unprecedented scale

[2] DINOv3 [github](https://github.com/facebookresearch/dinov3)

[3] DINOv3 [paper](https://ai.meta.com/research/publications/dinov3/)

[4] My DINOv3 colab [DINOv3_segmentation_tracking.ipynb](https://colab.research.google.com/drive/1IBQ4chTxowsBE_wYONRCmjcBnuSIOdkv#scrollTo=3S1MyIZucBoD)



```python
# 1. Load Model
model = torch.hub.load(
    repo_or_dir=local_dinov3_repo_path,
    model=MODEL_NAME,
    source='local',
    pretrained=False)
state_dict = torch.load(local_weights_path)
model.load_state_dict(state_dict)
model.to("cuda")
model.eval()

# 2. 


```

|                                                         |                                                                                                                                             |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| [[#### 1. Import]]                                      |                                                                                                                                             |
| [[#### 2. CONST and env setting]]                       |                                                                                                                                             |
| [[#### 3. Localized Loading of Model and Weights]]      |                                                                                                                                             |
| [[#### 4. forward()]]                                   | 封裝了將一張圖片輸入 DINOv3 模型並提取特徵的完整流程                                                                                                              |
| [[#### 5. Loading Video Frames]]                        | 載入影片幀                                                                                                                                       |
| [[#### 6. Displaying Sampled Video Frames]]             |                                                                                                                                             |
| [[#### 7. mask_to_rgb(), load_image_from_url()]]        | 在影片物件分割與追蹤任務中，通常需要給定第一幀的「答案」，也就是一個**遮罩 (mask)**，來告訴模型我們要追蹤哪些物體                                                                              |
| [[#### 8. class ResizeToMultiple]]                      | 完整的影像預處理流程 (pipeline)包括調整尺寸、轉換為張量、以及正規化                                                                                                     |
| [[#### 9. Preparing the Initial Mask]]                  | 將我們在 Cell 7載入的、與原始圖片一樣大的**初始遮罩 (initial mask)**，downsampling處理成能與模型特徵圖 (feature map) 匹配相同尺寸的格式                                              |
| [[#### 10. propagate(), make_neighborhood_mask()]]      | 定義了最核心的演算法函式 `propagate`。它的作用是：根據已知的「上下文幀 (context frames)」的特徵和遮罩，來推斷出「當前幀 (current frame)」的遮罩。它還定義了一個輔助函式 `make_neighborhood_mask` 來提高運算效率 |
| [[#### 11. Visualizing the Neighborhood Mask]]          | 呼叫 `make_neighborhood_mask` 函式，並將生成的鄰域遮罩顯示出來                                                                                                |
| [[#### 12. Performing Single-Frame Propagation]]        | 將前面定義的所有工具（特徵提取、初始機率、傳播函式）組合起來，完成從第 0 幀到第 1 幀的分割預測                                                                                          |
| [[#### 13. Post-processing and Visualizing the Result]] | 負責將 `propagate` 函式輸出的**低解析度機率圖**，轉換成我們可以看的高解析度分割結果圖                                                                                         |
| [[#### 14. Setting Hyperparameters]]                    | 追蹤演算法中所有可調整的核心參數（超參數, Hyperparameters                                                                                                       |
| [[#### 15. Running the Full Video Tracking]]            | 將前面所有準備好的函式和演算法組合起來，應用到整個影片的每一幀，從而完成從頭到尾的物件追蹤與分割任務                                                                                          |
| [[#### 16. Final Result Visualization]]                 | 將儲存的所有結果以多種直觀的方式呈現出來，讓我們可以全面地評估追蹤效果                                                                                                         |
| [[#### 17. Checking GPU Memory Usage]]                  | 檢查 GPU 記憶體使用                                                                                                                                |
|                                                         |                                                                                                                                             |

#### 1. Import
```python
!pip install -q lovely_tensors mediapy

import datetime
import functools
import io
import logging
import math
import os
from pathlib import Path
import tarfile
import time
import urllib
import lovely_tensors
import matplotlib.pyplot as plt
import mediapy as mp
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as TVT
import torchvision.transforms.functional as TVTF
from torch import Tensor, nn
from tqdm import tqdm
from google.colab import drive

DISPLAY_HEIGHT = 200
lovely_tensors.monkey_patch()
torch.set_grad_enabled(False)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
```

這個 cell 的主要目的是設定好整個專案的執行環境，包含了安裝必要的 Python 套件、匯入所有會用到的函式庫，以及設定一些全域參數。

**詳細解析：**
1. **`!pip install -q lovely_tensors mediapy`**:
    
    - 這是在 Colab 環境中安裝兩個第三方 Python 套件。
        
    - `lovely_tensors`: 一個非常好用的除錯工具，它能讓 PyTorch 的張量 (Tensor) 在被印出時，顯示其型別、維度、數值範圍等摘要資訊，比預設的輸出更具可讀性。
        
    - `mediapy`: 一個專門用來在 Colab 或 Jupyter Notebook 中處理和顯示影像與影片的函式庫。
        
2. **`import ...`**:
    
    - 這部分匯入了所有後續程式碼會用到的標準函式庫，例如：
        
        - `torch`, `nn`: PyTorch 核心，用於建立模型與張量運算。
            
        - `torchvision`: PyTorch 的視覺庫，用於影像轉換。
            
        - `PIL.Image`: 處理影像檔案。
            
        - `numpy`: 高效的數值運算。
            
        - `os`, `pathlib`: 處理檔案路徑與目錄操作。
            
        - `google.colab.drive`: 用於掛載 Google Drive。
            
3. **全域設定 (Global Settings)**:
    
    - `DISPLAY_HEIGHT`: 定義一個常數，用於後續顯示圖片時統一高度。
        
    - `lovely_tensors.monkey_patch()`: 執行後，每當你 `print` 一個 PyTorch 張量，它就會以 `lovely_tensors` 的美化格式顯示。
        
    - `torch.set_grad_enabled(False)`: **這是非常關鍵的一步**。在模型推論（預測）階段，我們不需要計算梯度來更新模型權重。關閉它能顯著減少 GPU 記憶體消耗並提升執行速度。
        
    - `logging.basicConfig(...)`: 設定日誌系統，讓程式在執行時能印出帶有時間戳的資訊，方便了解進度與除錯。

#### 2. CONST and env setting
```python
# Define Data folder
DRIVE_BASE_PATH = "/content/drive/MyDrive/Colab Notebooks"
DINOV3_DATA_FOLDER = os.path.join(DRIVE_BASE_PATH, "dinov3_datafolder")
DINOV3_GITHUB_LOCATION = "facebookresearch/dinov3"


# Load and save DINOv3 models:
MODEL_DINOV3_VITS = "dinov3_vits16"
MODEL_DINOV3_VITSP = "dinov3_vits16plus"
MODEL_DINOV3_VITB = "dinov3_vitb16"
MODEL_DINOV3_VITL = "dinov3_vitl16"
MODEL_DINOV3_VITHP = "dinov3_vith16plus"
MODEL_DINOV3_VIT7B = "dinov3_vit7b16"
# Choice DINOv3 model(ex: MODEL_DINOV3_VITS)
MODEL_NAME = MODEL_DINOV3_VITS
WEIGHTS_URL = "........url"
WEIGHTS_NAME = "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
VIDEO_FRAMES_URI = "https://dl.fbaipublicfiles.com/dinov3/notebooks/segmentation_tracking/video_frames.tar.gz"
VIDEO_TAR_NAME = "video_frames.tar.gz"


# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Selected device: {device}")
if torch.cuda.is_available():
  print(f"GPU Name: {torch.cuda.get_device_name(0)}")


# Mount Google Drive
try:
  drive.mount('/content/drive')
  print("Google Drive mounted successfully.")
except Exception as e:
  print(f"Error mounting Google Drive: {e}")
  raise SystemExit("Halting execution: Google Drive mount failed.")
```
這個 cell 負責定義整個專案中會用到的所有重要路徑、檔案名稱、URL 和模型名稱。同時，它也會設定運算裝置 (GPU 或 CPU) 並掛載 Google Drive。

**詳細解析：**

1. **路徑與名稱定義**:
    
    - 程式碼將所有重要的字串（如路徑、檔名、URL）都定義成大寫的常數變數。這是一個很好的程式設計習慣，讓程式碼更容易維護。如果未來需要更改儲存位置，只需修改這裡即可。
        
    - `DINOV3_DATA_FOLDER`: 這是整個專案的核心資料夾，所有下載的模型、程式碼和資料都會存放在這裡。
        
    - `WEIGHTS_URL`: 這是模型權重檔案 (`.pth`) 的直接下載連結。注意這個連結非常長，包含 `Policy`、`Signature` 等參數，這表明它是一個**有時效性的預簽章 URL (pre-signed URL)**。如果未來連結失效，需要去 DINOv3 官方管道尋找新的連結。
        
2. **裝置設定**:
    
    - `torch.device(...)` 是一個標準寫法，它會檢查 Colab 環境是否有可用的 NVIDIA GPU (cuda)。如果有，就使用 GPU；否則，就使用 CPU。GPU 能大幅加速深度學習運算。
        
3. **掛載 Google Drive**:
    
    - `drive.mount(...)` 會彈出一個授權視窗，請求存取你的 Google Drive。
        
    - 這麼做的**核心目的**是為了**持久化儲存**。將下載的模型和資料存放在 Google Drive 中，這樣即使 Colab 執行階段斷線或重啟，下次也不需要重新下載，可以直接從 Drive 讀取，節省大量時間。


#### 3. Localized Loading of Model and Weights
```python
# Build the gdrive dinov2 data and model folder
os.makedirs(DINOV3_DATA_FOLDER, exist_ok=True)


# Copy reop to gdrive repo folder
local_dinov3_repo_path = os.path.join(DINOV3_DATA_FOLDER, "dinov3")
if not os.path.exists(local_dinov3_repo_path):
  print(f"DINOv3 repository not found at '{local_dinov3_repo_path}'.")
  print("Cloning from GitHub... (This will only run once)")
  # Use git clone repo to repo folder
  !git clone https://github.com/facebookresearch/dinov3.git "{local_dinov3_repo_path}"
else:
  print(f"DINOv3 repository already exists at '{local_dinov3_repo_path}'. Skipping clone.")


# DINOv3 model weight file to gdrive folder
local_weights_path = os.path.join(DINOV3_DATA_FOLDER, WEIGHTS_NAME)
if not os.path.exists(local_weights_path):
  print(f"Weights file '{WEIGHTS_NAME}' not found.")
  print(f"Downloading from {WEIGHTS_URL}... (This will only run once)")
  !wget -O "{local_weights_path}" "{WEIGHTS_URL}"
else:
  print(f"Weights file '{WEIGHTS_NAME}' already exists. Skipping download.")


# Load DINOv3 repo and model files from gdrive folder
print("\nStep 3: Loading model and weights from local paths...")
try:
  # repo_or_dir: gdisk git clone folder
  # source='local': torch.hub load from local
  # pretrained=False: Build a empty model without weight
  model = torch.hub.load(
    repo_or_dir=local_dinov3_repo_path,
    model=MODEL_NAME,
    source='local',
    pretrained=False
  )

  # Load weight from local pth file
  state_dict = torch.load(local_weights_path)

  # Lod weight into model file
  model.load_state_dict(state_dict)

  # Setup mode as eval mode
  if torch.cuda.is_available():
      model.to("cuda")
  model.eval()
  patch_size = model.patch_size
  embed_dim = model.embed_dim

  print("\n Success! Model has been loaded completely from your Google Drive.")
  print("   - Architecture from:", local_dinov3_repo_path)
  print("   - Weights from:", local_weights_path)

except Exception as e:
  print(f"\n An error occurred during model loading: {e}")
  print("Please check if the paths and file names are correct.")
```
這是整個設定過程中最核心的 cell。它的目標是**將 DINOv3 的模型程式碼和權重檔案都下載到你的 Google Drive 中，並從你自己的 Drive 中載入模型**。這個策略非常穩健，可以避免因網路問題或 `torch.hub` 限制導致的下載失敗。
**詳細解析：**

1. **下載程式碼**: 首先，它檢查你的 Drive 裡是否已經有了 DINOv3 的原始碼。如果沒有，它會執行 `git clone` 將整個專案複製一份過來。這樣做可以讓 `torch.hub` 在本地找到定義模型的 Python 程式碼。
    
2. **下載權重**: 接著，用同樣的邏輯檢查權重檔案 (`.pth`) 是否存在。如果不存在，就用 `wget` 這個強大的下載工具將它從指定的 URL 下載到你的 Drive。
    
3. **兩階段載入模型 (最關鍵的技巧)**:
    
    - **第一步 (載入架構)**: `torch.hub.load(...)` 被用來建立一個模型的「空殼」。
        
        - `repo_or_dir=local_dinov3_repo_path`: 告訴 `torch.hub` 不要去網路上找，而是去我們剛才 clone 到 Drive 的那個資料夾裡找模型定義。
            
        - `source='local'`: 強調來源是本地路徑。
            
        - `pretrained=False`: 告訴它只要建立模型架構，**不要**去下載任何預訓練權重。
            
    - **第二步 (填入權重)**:
        
        - `torch.load(local_weights_path)`: 從我們下載到 Drive 的 `.pth` 檔案中，將權重數值讀取到記憶體中。
            
        - `model.load_state_dict(state_dict)`: 將這些權重數值載入到剛剛建立的「空殼」模型中，至此模型才算完整。
            

這個「先建構、後填充」的方法，完美地繞過了所有潛在的網路下載問題，讓模型載入過程變得非常可靠。


#### 4. forward()
```python
@torch.compile(disable=True)
def forward(
  model: nn.Module,
  img: Tensor,  # [3, H, W] already normalized for the model
) -> Tensor:
  feats = model.get_intermediate_layers(img.unsqueeze(0), n=1, reshape=True)[0]  # [1, D, h, w]
  feats = feats.movedim(-3, -1)  # [1, h, w, D]
  feats = F.normalize(feats, dim=-1, p=2)
  return feats.squeeze(0)  # [h, w, D]
```
這個 cell 定義了一個名為 `forward` 的函式，它封裝了將一張圖片輸入 DINOv3 模型並提取特徵的完整流程。
**詳細解析：**

1. **`@torch.compile(...)`**: 這是 PyTorch 2.0 引入的即時編譯器 (JIT compiler) 裝飾器。它能將 Python 程式碼轉換成更底層的最佳化程式碼以提升執行速度。這裡 `disable=True` 表示暫時不啟用它。
    
2. **`model.get_intermediate_layers(...)`**: DINOv3 這類自監督學習模型的核心價值在於其網路中間層學到的「特徵 (features)」。這個函式就是用來提取這些特徵圖 (feature map) 的。
    
3. **`feats.movedim(...)`**: 調整張量維度的順序。原始輸出的格式是 `[Batch, Channel, Height, Width]`，將其轉換為 `[Batch, Height, Width, Channel]`，這在某些後續處理中更方便。
    
4. **`F.normalize(...)`**: 進行 L2 正規化。這一步非常重要，它將每個位置的特徵向量都縮放成單位長度 (長度為 1)。這樣做之後，計算兩個特徵之間的「餘弦相似度 (cosine similarity)」就等同於計算它們的「點積 (dot product)」，極大地簡化了後續的相似性比較，這也是物體分割與追蹤的基礎。


#### 5. Loading Video Frames
```python

local_video_tar_path = os.path.join(DINOV3_DATA_FOLDER, VIDEO_TAR_NAME)
if not os.path.exists(local_video_tar_path):
  print(f"Cannot find video file '{VIDEO_TAR_NAME}'，Download...")
  !wget -O "{local_video_tar_path}" "{VIDEO_FRAMES_URI}"
  print("Download finish！")
else:
  print(f"'{VIDEO_TAR_NAME}' Video file is already in google drive folder.")


def load_video_frames_from_local_tar(tar_path: str) -> list[Image.Image]:
  images = []
  indices = []

  with tarfile.open(tar_path, "r:gz") as tar:
    for member in tar.getmembers():
      if member.isfile():
        index_str, _ = os.path.splitext(os.path.basename(member.name))
        image_data_file = tar.extractfile(member)
        image = Image.open(image_data_file).convert("RGB")
        images.append(image)
        indices.append(int(index_str))

  order = np.argsort(indices)
  return [images[i] for i in order]

frames = load_video_frames_from_local_tar(local_video_tar_path)
num_frames = len(frames)
print(f"Number of frames: {num_frames}")

original_width, original_height = frames[0].size
print(f"Original size: width={original_width}, height={original_height}")
```
這個 cell 的功能是下載一個包含範例影片所有影格 (frames) 的壓縮檔，並將其解壓縮、讀取成圖片物件列表，為後續的分析做準備。
**詳細解析：**

1. **下載影片資料**: 和 Cell 3 的邏輯一樣，先檢查 Google Drive 中是否存在 `video_frames.tar.gz` 檔案，如果沒有就下載。
    
2. **`load_video_frames_from_local_tar` 函式**:
    
    - 這個函式展示了如何高效地處理 `.tar.gz` 壓縮檔。
        
    - `tar.extractfile(member)`: 這是一個很棒的技巧，它允許你直接在記憶體中讀取壓縮檔內的單一檔案，而**不是**把整個壓縮檔解開到硬碟上，這樣更快速且不佔用磁碟空間。
        
    - **排序 (Sorting)**: 壓縮檔中的檔案順序不一定是按檔名排序的。因此，程式碼很聰明地從檔名中提取出數字 (`001`, `002` 等)，然後使用 `np.argsort` 來得到正確的順序索引，最後再根據這個索引重新排列圖片列表。這確保了影片的影格是按照時間順序 `1, 2, 3, ...` 排列的。
        
3. **執行與輸出**: 最後，呼叫這個函式，將返回的圖片列表存儲在 `frames` 變數中，並印出總共有多少影格以及它們的原始尺寸。至此，所有需要的模型和資料都已準備就緒。

#### 6. Displaying Sampled Video Frames
```python
num_selected_frames = 4
selected_frames = np.linspace(0, num_frames - 1, num_selected_frames, dtype=int)

mp.show_images(
    [frames[int(i)] for i in selected_frames],
    titles=[f"Frame {i}" for i in selected_frames],
    height=DISPLAY_HEIGHT,
)
```
這個 cell 的目的很單純：從已經載入的所有影片影格 (`frames`) 中，均勻地挑選幾張並顯示出來，讓我們可以快速預覽影片的內容。
**詳細解析：**

1. **`np.linspace(start, stop, num, ...)`**:
    
    - 這是 NumPy 函式庫中一個非常有用的工具，用於在一個指定的區間內產生等差數列。
        
    - `start=0`: 從第 0 幀開始。
        
    - `stop=num_frames - 1`: 到最後一幀結束。
        
    - `num=num_selected_frames`: 總共要產生 4 個數字。
        
    - `dtype=int`: 將產生的浮點數轉換為整數，因為列表的索引必須是整數。
        
    - **範例**: 如果影片總共有 100 幀 (`num_frames = 100`)，那麼 `selected_frames` 的結果會是 `[0, 33, 66, 99]`，代表了在影片開頭、前三分之一、後三分之一和結尾處的四個影格。
        
2. **`mp.show_images(...)`**:
    
    - 這是 `mediapy` 套件提供的便捷函式，專門用於在 Colab 中並排顯示多張圖片。
        
    - 它接收一個圖片列表和一個對應的標題列表，就能將它們整齊地呈現出來。
        

這個 cell 是一個**數據探索 (Data Exploration)** 的步驟，幫助我們確認影片資料是否已正確載入，並對影片內容有個初步的了解。



#### 7. mask_to_rgb(), load_image_from_url()
```python
def mask_to_rgb(mask: np.ndarray | Tensor, num_masks: int) -> np.ndarray:
  if isinstance(mask, Tensor):
    mask = mask.cpu().numpy()

  # Exclude background
  background = mask == 0
  mask = mask - 1
  num_masks = num_masks - 1

  # Choose palette
  if num_masks <= 10:
    mask_rgb = plt.get_cmap("tab10")(mask)[..., :3]
  elif num_masks <= 20:
    mask_rgb = plt.get_cmap("tab20")(mask)[..., :3]
  else:
    mask_rgb = plt.get_cmap("gist_rainbow")(mask / (num_masks - 1))[..., :3]

  mask_rgb = (mask_rgb * 255).astype(np.uint8)
  mask_rgb[background, :] = 0
  return mask_rgb


def load_image_from_url(url: str) -> Image:
  with urllib.request.urlopen(url) as f:
    return Image.open(f)


first_mask_np = np.array(
  load_image_from_url(
    "https://dl.fbaipublicfiles.com/dinov3/notebooks/segmentation_tracking/first_video_frame_mask.png"
  )
)


mask_height, mask_width = first_mask_np.shape  # Abbreviated at [H', W']
print(f"Mask size: {[mask_height, mask_width]}")


num_masks = int(first_mask_np.max() + 1)  # Abbreviated as M
print(f"Number of masks: {num_masks}")


mp.show_images(
  [frames[0], mask_to_rgb(first_mask_np, num_masks)],
  titles=["Frame", "Mask"],
  height=DISPLAY_HEIGHT,
)


try:
  patch_size = model.patch_size
  print(f"成功從模型中取得 patch_size: {patch_size}")
except NameError:
  print("錯誤：'model' 物件不存在。請先執行載入模型的 cell。")
  # 為了讓程式碼能繼續展示，這裡先給定一個預設值，但在您的工作流程中應確保 model 已載入
  patch_size = 16
  print(f"警告：使用預設 patch_size: {patch_size}")
```

載入並視覺化初始遮罩 (Loading and Visualizing the Initial Mask). 這個 cell 是**設定任務起點**的關鍵步驟。在影片物件分割與追蹤任務中，通常需要給定第一幀的「答案」，也就是一個**遮罩 (mask)**，來告訴模型我們要追蹤哪些物體。這個 cell 就是在載入這個初始遮罩，並將它視覺化，以確認其正確性。
**詳細解析：**

1. **`mask_to_rgb` 函式**:
    
    - **目的**: 遮罩檔案本身通常是單通道的灰階圖，其中每個像素的值代表一個類別（例如 0 代表背景，1 代表車子，2 代表行人）。這樣的圖片不直觀。此函式的目的就是將這個「類別圖」轉換成彩色的 RGB 圖片，讓不同的物體顯示不同的顏色，方便我們用肉眼觀察。
        
    - **原理**: 它根據物體的數量選擇一個合適的調色盤 (colormap)，例如 `tab10` 或 `gist_rainbow`，然後將每個類別索引映射到一個具體的顏色。最後，它會將背景（值為 0 的像素）塗成黑色。
        
2. **`load_image_from_url` 函式**: 一個簡單的工具函式，可以直接從網路連結讀取圖片數據並用 PIL 函式庫打開，而不需要先儲存到本地。
    
3. **載入與分析遮罩**:
    
    - 程式碼從 DINOv3 官方提供的 URL 載入了第一幀的遮罩。這個遮罩是預先製作好的，標記了影片中要追蹤的車輛和行人。
        
    - `num_masks = int(first_mask_np.max() + 1)`: 這是一個計算總類別數的常用技巧。如果遮罩中的最大值是 2，代表有標記為 0, 1, 2 的三個類別，所以總數是 `2 + 1 = 3`。
        
4. **視覺化驗證**:
    
    - 程式碼將原始的第一幀圖片與經過 `mask_to_rgb` 上色後的遮罩圖片並排顯示。這是一個非常重要的**健全性檢查 (Sanity Check)**，讓我們可以確認遮罩是否正確地對應到了圖片中的物體。
        
5. **取得 `patch_size`**: 最後，它從之前載入的 `model` 物件中取得了 `patch_size` 這個屬性。這個值對於下一個 cell 的影像預處理至關重要。

#### 8. class ResizeToMultiple
```python
class ResizeToMultiple(nn.Module):
  def __init__(self, short_side: int, multiple: int):
    super().__init__()
    self.short_side = short_side
    self.multiple = multiple

  def _round_up(self, side: float) -> int:
    return math.ceil(side / self.multiple) * self.multiple

  def forward(self, img):
    old_width, old_height = TVTF.get_image_size(img)
    if old_width > old_height:
      new_height = self._round_up(self.short_side)
      new_width = self._round_up(old_width * new_height / old_height)
    else:
      new_width = self._round_up(self.short_side)
      new_height = self._round_up(old_height * new_width / old_width)
    return TVTF.resize(img, [new_height, new_width], interpolation=TVT.InterpolationMode.BICUBIC)


SHORT_SIDE = 960
transform = TVT.Compose(
    [
      ResizeToMultiple(short_side=SHORT_SIDE, multiple=patch_size),
      TVT.ToTensor(),
      TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
first_frame = transform(frames[0]).to("cuda")
print(f"First frame: {first_frame}")

_, frame_height, frame_width = first_frame.shape  # Abbreviated as [H, W]
feats_height, feats_width = frame_height // patch_size, frame_width // patch_size  # Abbreviated as [h, w]
```
影像預處理轉換 (Image Pre-processing Transform)

這個 cell 的核心任務是建立一個完整的**影像預處理流程 (pipeline)**。原始的圖片不能直接輸入到神經網路中，必須經過一系列的轉換，包括調整尺寸、轉換為張量、以及正規化。
**詳細解析：**

1. **`ResizeToMultiple` 類別**:
    
    - **目的**: 這是整個流程中最客製化的一步。Vision Transformer (ViT) 這類模型會將輸入圖片切割成一個個不重疊的**小方塊 (patches)**。為了讓圖片能被完美地切割，圖片的長和寬都必須是 `patch_size` 的整數倍。這個類別就是用來實現這個特殊的 resize 需求的。
        
    - **`_round_up` 函式**: 一個數學輔助函式，功能是「向上取整到指定倍數」。例如，如果 `patch_size` 是 16，`_round_up(961)` 的結果會是 976 (因為 976 是大於 961 且最接近的 16 的倍數)。
        
    - **`forward` 函式**: 這是執行轉換時的核心邏輯。它會先在保持原始長寬比的前提下，將圖片的短邊縮放到 `SHORT_SIDE` (960)，然後再用 `_round_up` 函式確保最終的長和寬都能被 `patch_size` 整除。
        
2. **`TVT.Compose([...])`**:
    
    - 這是 `torchvision` 提供的一個容器，可以將多個轉換步驟串連起來，形成一個單一的轉換管線。當你對一張圖片應用 `transform` 時，它會依序執行列表中的每一個轉換步驟。
        
3. **轉換步驟**:
    
    - **ResizeToMultiple**: 第一步，進行特殊尺寸調整。
        
    - **ToTensor**: 第二步，將圖片格式轉換為 PyTorch 的張量格式，這是所有 PyTorch 模型的標準輸入格式。
        
    - **Normalize**: 第三步，對張量的數值進行正規化。這一步是為了讓輸入圖片的數據分佈與模型在訓練時看到的數據分佈盡可能一致，對於模型的性能至關重要。
        

最後，程式碼將這個建立好的 `transform` 應用到影片的第一幀上，並印出轉換後的張量資訊，同時計算出模型將會產生的特徵圖 (feature map) 的大小。至此，資料預處理的準備工作全部完成。

#### 9. Preparing the Initial Mask
```python
first_mask = torch.from_numpy(first_mask_np).to("cuda", dtype=torch.long)  # [H', W']
first_mask = F.interpolate(
    first_mask[None, None, :, :].float(),  # [1, 1, H', W']
    (feats_height, feats_width),
    mode="nearest-exact",
)[0, 0].long()  # [h, w]
print(f"First mask:  {first_mask}")

first_probs = F.one_hot(first_mask, num_masks).float()  # [h, w, M]
print(f"First probs: {first_probs}")
```
準備初始遮罩 (Preparing the Initial Mask)

這個 cell 的核心任務是將我們在 [Cell 7] 載入的、與原始圖片一樣大的**初始遮罩 (initial mask)**，處理成能與模型特徵圖 (feature map) 匹配的格式。因為特徵圖的尺寸遠小於原始圖片，所以遮罩也必須被**降維 (downsample)** 到相同尺寸。
**詳細解析：**

1. **轉換為張量**: 第一步是基本的格式轉換，將 NumPy 陣列轉為 PyTorch 張量，並放到 GPU 上以利後續運算。
    
2. **`F.interpolate` (降維)**: 這是關鍵步驟。
    
    - **為什麼要降維?** DINOv3 模型輸出的特徵圖 `feats` 的尺寸是 `[h, w, D]`，比原始圖片小 `patch_size` 倍。我們的初始遮罩 `first_mask_np` 尺寸是 `[H', W']`，和原始圖片一樣大。為了讓遮罩和特徵圖能夠對應起來，必須將遮罩縮小到 `[h, w]`。
        
    - **`mode="nearest-exact"`**: 為什麼用最近鄰插值？假設我們要縮小一個像素，它周圍有代表「車子」(值=1) 和「馬路」(值=2) 的像素。如果用雙線性插值 (bilinear)，縮小後的像素值可能會變成 1.5，這沒有任何意義。而最近鄰插值會直接選擇最近的像素值 (1 或 2)，從而保留了遮罩的類別屬性。
        
3. **`F.one_hot` (One-Hot 編碼)**:
    
    - **目的**: 將一個數值標籤轉換成一個機率向量。
        
    - **範例**: 假設有 3 個類別 (背景=0, 車子=1, 行人=2)，即 `num_masks=3`。
        
        - 特徵圖上某個位置的遮罩值是 `1` (車子)。
            
        - 經過 One-Hot 編碼後，它會變成一個長度為 3 的向量 `[0., 1., 0.]`。
            
        - 這個向量可以被解讀為：這個位置是背景的機率為0，是車子的機率為1，是行人的機率為0。
            
    - 最終，`first_probs` 的維度是 `[h, w, M]`，代表了第一幀特徵圖上**每一個位置**的**真實物件類別機率分佈**。這是後續追蹤演算法的**起始點 (Ground Truth)**。



#### 10. propagate(), make_neighborhood_mask()
```python
@torch.compile(disable=True)
def propagate(
    current_features: Tensor,  # [h", w", D], where h=h", w=w", and " stands for current
    context_features: Tensor,  # [t, h, w, D]
    context_probs: Tensor,  # [t, h, w, M]
    neighborhood_mask: Tensor,  # [h", w", h, w]
    topk: int,
    temperature: float,
) -> Tensor:
    t, h, w, M = context_probs.shape

    # Compute similarity current -> context
    dot = torch.einsum(
        "ijd, tuvd -> ijtuv",
        current_features,  # [h", w", D]
        context_features,  # [t, h, w, D]
    )  # [h", w", t, h, w]

    # Restrict focus to local neighborhood
    dot = torch.where(
        neighborhood_mask[:, :, None, :, :],  # [h", w", 1, h, w]
        dot,  # [h", w", t, h, w]
        -torch.inf,
    )

    # Select top-k patches inside the neighborhood
    dot = dot.flatten(2, -1).flatten(0, 1)  # [h"w", thw]
    k_th_largest = torch.topk(dot, dim=1, k=topk).values  # [h"w", k]
    dot = torch.where(
        dot >= k_th_largest[:, -1:],  # [h"w", thw]
        dot,  # [h"w", thw]
        -torch.inf,
    )

    # Propagate probabilities from context to current frame
    weights = F.softmax(dot / temperature, dim=1)  # [h"w", thw]
    current_probs = torch.mm(
        weights,  # [h"w", thw]
        context_probs.flatten(0, 2),  # [thw, M]
    )  # [h"w", M]

    # Propagated probs should already sum to 1, but just in case
    current_probs = current_probs / current_probs.sum(dim=1, keepdim=True)  # [h"w", M]

    return current_probs.unflatten(0, (h, w))  # [h", w", M]


@functools.lru_cache()
def make_neighborhood_mask(h: int, w: int, size: float, shape: str) -> Tensor:
    ij = torch.stack(
        torch.meshgrid(
            torch.arange(h, dtype=torch.float32, device="cuda"),
            torch.arange(w, dtype=torch.float32, device="cuda"),
            indexing="ij",
        ),
        dim=-1,
    )  # [h, w, 2]
    if shape == "circle":
        ord = 2
    elif shape == "square":
        ord = torch.inf
    else:
        raise ValueError(f"Invalid {shape=}")
    norm = torch.linalg.vector_norm(
        ij[:, :, None, None, :] - ij[None, None, :, :, :],  # [h", w", h, w, 2]
        ord=ord,
        dim=-1,
    )  # [h", w", h, w]
    mask = norm <= size  # [h", w", h, w] bool, True inside, False outside
    return mask
```
核心傳播演算法 (The Core Propagation Algorithm)

這個 cell 是整個專案的**大腦和心臟**，定義了最核心的演算法函式 `propagate`。它的作用是：根據已知的「上下文幀 (context frames)」的特徵和遮罩，來推斷出「當前幀 (current frame)」的遮罩。它還定義了一個輔助函式 `make_neighborhood_mask` 來提高運算效率。

**`propagate` 函式詳細解析：**

這個函式的邏輯可以概括為：「**對於當前幀的每一個小區塊(patch)，去所有上下文幀中尋找跟它長得最像的小區塊，然後把那些最像的小區塊的已知物件類別，加權平均後『傳播』過來，作為當前小區塊的物件類別。**」

1. **計算相似度**:
    
    - `dot = torch.einsum("ijd, tuvd -> ijtuv", ...)`: 使用 `einsum` (愛因斯坦求和約定) 高效地計算**當前幀的每個 patch** 與**上下文幀的所有 patch** 之間的**點積 (dot product)**。因為所有特徵向量都已經被正規化，點積就等於**餘弦相似度**。結果 `dot` 是一個巨大的五維張量，儲存了所有 patch 之間的相似度分數。
        
2. **限制搜尋範圍 (Local Neighborhood)**:
    
    - `dot = torch.where(neighborhood_mask, ...)`: 這是一個關鍵的**效能優化**。我們假設物體在相鄰的幀之間不會移動太遠。因此，對於當前幀的 `(i, j)` 位置的 patch，我們只在上下文幀中 `(i, j)` 周圍的一個小區域 (由 `neighborhood_mask` 定義) 內去尋找相似的 patch。對於這個區域之外的所有 patch，直接將它們的相似度設為負無窮，相當於忽略它們。這大大減少了計算量並排除了遠處不相關的干擾。
        
3. **篩選最相似的 k 個點 (Top-K Selection)**:
    
    - `k_th_largest = torch.topk(...)` & `dot = torch.where(...)`: 在上一步的局部鄰域內，可能仍然有很多 patch。這一步進一步篩選，只保留相似度分數最高的 `k` 個 patch (例如 `topk=5`)，將其他的也設為負無窮。這使得傳播更加穩健，只依賴最可靠的幾個匹配點。
        
4. **計算權重並傳播機率**:
    
    - `weights = F.softmax(dot / temperature, dim=1)`: 將最終篩選出的相似度分數通過 Softmax 函數轉換成**權重**，所有權重加總為 1。`temperature` 是一個溫度係數，值越小，Softmax 的輸出會越「尖銳」，也就是更傾向於給相似度最高的那個 patch 極大的權重，反之則更平滑。
        
    - `current_probs = torch.mm(...)`: 這是最終的傳播步驟。它執行一個矩陣乘法，本質上是一個**加權平均**。它用剛才算出的 `weights` 去對 `context_probs` (上下文幀的已知類別機率) 進行加權求和。
        

**`make_neighborhood_mask` 函式詳細解析：**

- **`@functools.lru_cache()`**: 這是一個**快取 (cache)** 裝飾器。由於鄰域遮罩的形狀只取決於特徵圖大小和鄰域半徑，它在整個影片處理過程中是不會變的。這個裝飾器會將第一次計算的結果儲存起來，之後再用相同的參數呼叫此函式時，會直接返回儲存的結果，而不會重新計算，極大地提高了效率。
    
- **功能**: 這個函式會產生一個四維的布林 (boolean) 張量。`mask[i, j, u, v]` 的值為 `True`，若且唯若 `(u, v)` 這個點在以 `(i, j)` 為中心的、半徑為 `size` 的鄰域內。可以生成圓形 (`circle`) 或方形 (`square`) 的鄰域。


#### 11. Visualizing the Neighborhood Mask
```python
neighborhood_mask = make_neighborhood_mask(feats_height, feats_width, size=12, shape="circle")

mp.show_images(
    {f"{(i, j)}": neighborhood_mask[i, j].cpu().numpy() for i, j in [[3, 14], [20, 25]]},
    height=DISPLAY_HEIGHT,
)
```
視覺化鄰域遮罩 (Visualizing the Neighborhood Mask)

這是一個**健全性檢查 (Sanity Check)** cell。它的目的是呼叫 `make_neighborhood_mask` 函式，並將生成的鄰域遮罩顯示出來，以確認其功能是否符合預期。

**詳細解析：**

- 程式碼首先生成了實際會用到的鄰域遮罩。
    
- 然後，它從這個巨大的四維遮罩中，取出了以 `(3, 14)` 點為中心的鄰域，以及以 `(20, 25)` 點為中心的鄰域。
    
- 輸出的圖片應該是兩張黑色的圖，中間分別以 `(3, 14)` 和 `(20, 25)` 為圓心有一個白色的圓圈。這證明了我們的鄰域遮罩函式是正確的，它確實為特徵圖上的每一個點都定義了一個局部的搜尋範圍。

#### 12. Performing Single-Frame Propagation
```python
torch._dynamo.maybe_mark_dynamic(first_frame, (1, 2))
first_feats = forward(model, first_frame)  # [h, w, D]
print(f"First feats:   {first_feats.shape}")

frame_idx = 1
current_frame_pil = frames[frame_idx]
current_frame = transform(current_frame_pil).to("cuda")  # [3, H, W]
torch._dynamo.maybe_mark_dynamic(current_frame, (1, 2))
current_feats = forward(model, current_frame)  # [h", w", D]
print(f"Current feats: {current_feats.shape}")

current_probs = propagate(
    current_feats,  # [h", w", D]
    context_features=first_feats.unsqueeze(0),  # [1, h, w, D]
    context_probs=first_probs.unsqueeze(0),  # [1, h, w, M]
    neighborhood_mask=neighborhood_mask,  # [h", w", h, w]
    topk=5,
    temperature=0.2,
)  # [h", w", M]
print(f"Current probs:  {current_probs}")
```
執行單幀傳播 (Performing Single-Frame Propagation)

這個 cell 是演算法的**首次實戰演練**。它將前面定義的所有工具（特徵提取、初始機率、傳播函式）組合起來，完成從第 0 幀到第 1 幀的分割預測。

**詳細解析：**

1. **提取特徵**: 分別對第 0 幀和第 1 幀執行預處理和 `forward` 函式，得到它們各自的特徵圖 `first_feats` 和 `current_feats`。
    
2. **`torch._dynamo.maybe_mark_dynamic(...)`**: 這是給 PyTorch 2.0 的 JIT 編譯器 `Dynamo` 的一個提示。因為影片的每一幀在預處理後尺寸可能略有不同，這行程式碼告訴編譯器，張量的第 1 和第 2 維度（高度和寬度）是動態變化的，這樣編譯器可以生成更具通用性的最佳化程式碼。
    
3. **呼叫 `propagate`**: 這是本 cell 最核心的一步。它將第 1 幀的特徵 (`current_feats`) 作為目標，將第 0 幀的特徵 (`first_feats`) 和機率 (`first_probs`) 作為線索，執行傳播演算法。
    
4. **輸出**: 函式的返回值 `current_probs` 是一個 `[h", w", M]` 的張量，代表了模型對第 1 幀特徵圖上**每一個位置**屬於**每一個類別**的預測機率。

#### 13. Post-processing and Visualizing the Result
```python
def postprocess_probs(
    probs: Tensor,  # [B, M, H', W']
) -> Tensor:
    vmin = probs.flatten(2, 3).min(dim=2).values  # [B, M]
    vmax = probs.flatten(2, 3).max(dim=2).values  # [B, M]
    probs = (probs - vmin[:, :, None, None]) / (vmax[:, :, None, None] - vmin[:, :, None, None])
    probs = torch.nan_to_num(probs, nan=0)
    return probs  # [B, M, H', W']


p = current_probs.movedim(-1, -3).unsqueeze(0)  # [1, M, h", w"]
p = F.interpolate(p, size=(mask_height, mask_width), mode="nearest")  # [1, M, H', W']
p = postprocess_probs(p).squeeze(0)  # [M, H', W']
current_pred_np = p.argmax(0).cpu().numpy()  # [H', W']
current_probs_np = p.cpu().numpy()  # [M, H', W']
del p

mp.show_images(
    [
        frames[0],
        current_frame_pil,
        mask_to_rgb(first_mask_np, num_masks),
        mask_to_rgb(current_pred_np, num_masks),
    ],
    titles=["First frame", "Second frame", "", ""],
    columns=2,
    height=DISPLAY_HEIGHT,
)

mp.show_images(current_probs_np, titles=[f"Mask {i}" for i in range(num_masks)], height=DISPLAY_HEIGHT)
```
後處理與視覺化結果 (Post-processing and Visualizing the Result) 這個 cell 負責將 `propagate` 函式輸出的**低解析度機率圖**，轉換成我們可以看的**高解析度分割結果圖**

**詳細解析：**

1. **`postprocess_probs` 函式**:
    
    - **目的**: 進行**對比度拉伸 (contrast stretching)**。有時候模型預測的機率值可能都擠在一個很小的範圍內（例如 0.4 到 0.6），導致結果模糊不清。這個函式會對**每一個類別**的機率圖獨立進行 Min-Max 正規化，將其數值範圍拉伸到 `[0, 1]`，使得高機率的區域更亮，低機率的區域更暗，結果更清晰。
        
2. **放大與預測 (Upsampling & Prediction)**:
    
    - `F.interpolate(...)`: 再次使用插值函式，但這次是**放大**，將低解析度的 `[h", w"]` 機率圖恢復到原始遮罩尺寸 `[H', W']`。
        
    - `p.argmax(0)`: 這是做出最終決策的一步。對於放大後的機率圖上的每一個像素，它有 `M` 個機率值（對應 `M` 個類別）。`argmax(0)` 會找到哪個類別的機率最高，並返回該類別的索引。最終得到的 `current_pred_np` 就是一個 `[H', W']` 的整數陣列，即我們預測出的分割遮罩。
        
3. **視覺化**:
    
    - 第一個 `mp.show_images` 以 2x2 的網格展示了結果：上面是前後兩幀的原始圖片，下面是它們對應的遮罩。這讓我們可以非常直觀地比較**真實遮罩 (第 0 幀)** 和**模型預測的遮罩 (第 1 幀)**，評估追蹤的效果。
        
    - 第二個 `mp.show_images` 則顯示了每個類別的機率熱圖。例如，「車子」的熱圖上越亮的地方，代表模型認為該處是車子的可能性越高。這有助於我們理解模型做出決策的內部依據。



#### 14. Setting Hyperparameters
```python
MAX_CONTEXT_LENGTH = 7
NEIGHBORHOOD_SIZE = 12
NEIGHBORHOOD_SHAPE = "circle"
TOPK = 5
TEMPERATURE = 0.2
```
設定超參數 (Setting Hyperparameters)

這個 cell 的功能非常單純但極為重要：它將整個追蹤演算法中所有可調整的**核心參數（超參數, Hyperparameters）**集中定義在一起。這樣做的好處是，當我們想要實驗不同參數設定對結果的影響時，只需要修改這個 cell 即可，而不用在複雜的程式碼中到處尋找，大大提高了程式碼的可維護性和易用性。

**詳細解析：**

- **`MAX_CONTEXT_LENGTH = 7`**:
    
    - **意義**: 這定義了「記憶」的長度。在預測第 N 幀時，模型會參考第 N-1, N-2, ..., 最多到 N-7 幀的資訊。這是一個滑動的窗口。
        
    - **權衡**: 較大的值能提供更豐富的歷史資訊，有助於處理物體被短暫遮擋後再出現的情況，但會消耗更多的 GPU 記憶體和計算時間。較小的值則更輕量，但可能在複雜場景下丟失目標。
        
- **`NEIGHBORHOOD_SIZE = 12`**:
    
    - **意義**: 定義了局部搜尋的半徑。如果一個物體在相鄰幀之間移動的距離超過了 12 個 patch 的範圍，追蹤就可能會失敗。
        
    - **權衡**: 較大的值能追蹤移動更快的物體，但增加了計算成本，也可能因為搜尋範圍太大而匹配到錯誤的相似物體。較小的值計算快，但在快速運動場景下不適用。
        
- **`TOPK = 5`**:
    
    - **意義**: 在局部搜尋範圍內，只信任最相似的 5 個 patch。
        
    - **權衡**: 較小的 K 值（如 3 或 5）讓預測更依賴於最可靠的幾個匹配，結果通常更穩定。較大的 K 值則引入了更多的參考資訊，可能會讓邊緣更平滑，但也可能被一些不太準確的匹配所干擾。
        
- **`TEMPERATURE = 0.2`**:
    
    - **意義**: 控制 Softmax 權重的分佈。溫度值遠小於 1，會讓權重分佈變得非常「尖銳」，意味著模型會給予相似度最高的那個 patch 極大的權重，幾乎忽略其他 patch。
        
    - **權衡**: 低溫（如 0.1-0.3）表示模型對自己的最佳匹配非常有信心，通常能產生更清晰、邊界更明確的結果。高溫（接近 1）則會產生更「模糊」、更平均的權重，結果較為平滑，但也可能缺乏決斷力。

#### 15. Running the Full Video Tracking
```python
mask_predictions = torch.zeros([num_frames, mask_height, mask_width], dtype=torch.uint8)  # [T, H', W']
mask_predictions[0, :, :] = torch.from_numpy(first_mask_np)

mask_probabilities = torch.zeros([num_frames, num_masks, mask_height, mask_width])  # [T, M, H', W']
mask_probabilities[0, :, :, :] = F.one_hot(torch.from_numpy(first_mask_np).long(), num_masks).movedim(-1, -3)

features_queue: list[Tensor] = []
probs_queue: list[Tensor] = []

neighborhood_mask = make_neighborhood_mask(
    feats_height,
    feats_width,
    size=NEIGHBORHOOD_SIZE,
    shape=NEIGHBORHOOD_SHAPE,
)  # [h", w", h, w]

start = time.perf_counter()
for frame_idx in tqdm(range(1, num_frames), desc="Processing"):
    # Extract features for the current frame
    current_frame_pil = frames[frame_idx]
    current_frame = transform(current_frame_pil).to("cuda")  # [3, H, W]
    torch._dynamo.maybe_mark_dynamic(current_frame, (1, 2))
    current_feats = forward(model, current_frame)  # [h", w", D]

    # Prepare the context, marking the time and mask dimensions as dynamic for torch compile
    context_feats = torch.stack([first_feats, *features_queue], dim=0)  # [1+len(queue), h, w, D]
    context_probs = torch.stack([first_probs, *probs_queue], dim=0)  # [1+len(queue), h, w, M]
    torch._dynamo.maybe_mark_dynamic(context_feats, 0)
    torch._dynamo.maybe_mark_dynamic(context_probs, (0, 3))

    # Propagate segmentation probs from context frames
    current_probs = propagate(
        current_feats,
        context_feats,
        context_probs,
        neighborhood_mask,
        TOPK,
        TEMPERATURE,
    )  # [h", w", M]

    # Update queues with current features and probs
    features_queue.append(current_feats)
    probs_queue.append(current_probs)
    if len(features_queue) > MAX_CONTEXT_LENGTH:
        features_queue.pop(0)
    if len(probs_queue) > MAX_CONTEXT_LENGTH:
        probs_queue.pop(0)

    # Upsample and postprocess segmentation probs, argmax to obtain a prediction
    current_probs = F.interpolate(
        current_probs.movedim(-1, -3)[None, :, :, :],
        size=(mask_height, mask_width),
        mode="nearest",
    )  # [1, M, H', W']
    current_probs = postprocess_probs(current_probs)  # [1, M, H', W']
    current_probs = current_probs.squeeze(0)
    mask_probabilities[frame_idx, :, :, :] = current_probs
    pred = torch.argmax(current_probs, dim=0).to(dtype=torch.uint8)  # [H', W']
    mask_predictions[frame_idx, :, :] = pred  # [H', W']

torch.cuda.synchronize()
end = time.perf_counter()
print(f"Processing time:    {datetime.timedelta(seconds=round(end - start))}")
print(f"Mask probabilities: {mask_probabilities}")
print(f"Mask predictions:   {mask_predictions}")
```
執行完整的影片追蹤 (Running the Full Video Tracking)

這是整個 Colab Notebook 的**執行核心**。它將前面所有準備好的函式和演算法組合起來，應用到整個影片的每一幀，從而完成從頭到尾的物件追蹤與分割任務。

**詳細解析：**

1. **初始化**: 在迴圈開始前，程式碼建立了兩個巨大的 PyTorch 張量 `mask_predictions` 和 `mask_probabilities`，用來存放每一幀的最終結果。同時，建立了兩個空的列表 `features_queue` 和 `probs_queue`，它們將作為實現「滑動窗口」的佇列。
    
2. **主迴圈**: 使用 `tqdm` 函式庫來顯示一個進度條，讓我們可以直觀地看到處理進度。
    
    - **準備上下文 (`context`)**: 在每一步，它都會動態地建立上下文。這個上下文**永遠包含第一幀的特徵和機率**，這是一個非常聰明的設計，可以作為一個**穩定的錨點 (anchor)**，防止追蹤結果因為連續誤差累積而完全跑偏。然後，它會把 `features_queue` 中儲存的最近幾幀的資訊也加進來。
        
    - **更新佇列**: 在完成一次傳播後，它會將剛剛計算出的 `current_feats` 和 `current_probs` 加入到佇列的末尾。然後檢查佇列長度是否超過了 `MAX_CONTEXT_LENGTH` 的限制，如果超過了，就用 `.pop(0)` 從佇列的開頭移除最舊的一幀。這一進一出的操作，就實現了**滑動窗口 (sliding window)** 的效果。
        
    - **儲存結果**: 每處理完一幀，就將放大並後處理好的機率圖和最終的預測遮罩儲存到我們一開始建立的大張量中。
        
3. **計時**: 透過在迴圈前後記錄時間，我們可以得知處理整個影片所需的總時間，這有助於評估演算法的效率。

#### 16. Final Result Visualization
```python
import mediapy as mp

mp.show_images(
    [frames[i].convert("RGB") for i in selected_frames]
    + [mask_to_rgb(mask_predictions[i], num_masks) for i in selected_frames],
    titles=[f"Frame {i}" for i in selected_frames] + [""] * len(selected_frames),
    columns=len(selected_frames),
    height=DISPLAY_HEIGHT,
)

mp.show_videos(
    {
        "Input": [np.array(frame) for frame in frames],
        "Pred": mask_to_rgb(mask_predictions, num_masks),
    },
    height=DISPLAY_HEIGHT,
    fps=24,
)
mp.show_videos(
    {f"Prob {i}": mask_probabilities[:, i].numpy() for i in range(num_masks)},
    height=DISPLAY_HEIGHT,
    fps=24,
)
```
最終結果視覺化 (Final Result Visualization)

當 Cell 15 的漫長計算完成後，這個 cell 負責將儲存的所有結果以多種直觀的方式呈現出來，讓我們可以全面地評估追蹤效果。

**詳細解析：**

1. **靜態圖片對比**: 第一部分程式碼顯示了一個 2x4 的圖片網格。上面一排是影片中幾個時間點的原始畫面，下面一排是完全對應的、模型預測出的分割結果。這讓我們可以靜態地、仔細地檢查模型在特定時刻的表現。
    
2. **動態影片對比**: 第二部分是整個 Colab 最直觀的結果展示。它生成了兩個並排播放的影片：左邊是原始輸入影片，右邊是模型生成的彩色分割影片。你可以看到分割出來的物體（如車輛和行人）隨著原始影片的運動而同步運動，這完美地展示了**影片物件追蹤與分割 (Video Object Segmentation and Tracking)** 的效果。
    
3. **機率熱圖影片**: 第三部分提供了更深度的分析視角。它為每一個物件類別（例如背景、車子、行人）都生成了一個獨立的灰階影片。在影片中，像素越白的地方，代表模型在該時刻認為這個像素屬於該類別的機率越高。這可以幫助我們理解模型的「思考過程」，例如當兩個物體靠近時，它們各自的機率熱圖可能會出現一些重疊或不確定性。



#### 17. Checking GPU Memory Usage
```python
print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
```
檢查 GPU 記憶體使用 (Checking GPU Memory Usage) 這是一個收尾的工具性質 cell，功能很簡單，但對於評估資源消耗很有幫助

**詳細解析：**

- `torch.cuda.max_memory_allocated()`: 這是 PyTorch 內建的一個函式，它會回報從程式開始執行到現在，GPU 記憶體被佔用的最高紀錄（峰值）。
    
- `/ 2**30`: 將單位從 Bytes 轉換為 Gigabytes (GB)。
    
- **作用**: 這個數字告訴我們執行這個演算法最少需要多少 GPU 記憶體。例如，如果結果是 `7.5 GB`，就意味著你的 GPU 至少需要有 8GB 的記憶體才能順利運行此 Colab。這對於判斷演算法是否能在不同硬體上運行，或者在調整超參數（如 `MAX_CONTEXT_LENGTH`）時評估其對記憶體的影響，都是一個非常重要的指標。




