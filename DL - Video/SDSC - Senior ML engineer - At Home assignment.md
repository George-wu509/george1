
https://colab.research.google.com/drive/1WVajBi1p8fItt5asogjLpfFD2poH4CmJ

|                                                                                                 |     |
| ----------------------------------------------------------------------------------------------- | --- |
| 1. Python implementation                                                                        |     |
| 2. Embeddings with VideoMAE-2                                                                   |     |
| 3. BONUS: Use the VideoMAE model to predict the outcome of the procedure from the SOCAL dataset |     |
conda activate image_bbox_env
pip install opencv-python numpy matplotlib


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
image size = 1920 x 1080
y軸 1080 x 0.56 = 604.8
y軸 1080 x 0.17 = 183.6
x軸長  350/1920 = 0.18
y軸長  480/1080 = 0.44
所以
S102T1_frame_00000078.txt
8 0.5580729166666667 0.7865740740740741 0.17239583333333336 0.4268518518518518
8 0.56 0.79 0.17 0.43
[ id, y軸start, x軸?, x軸長, y軸長]



#### 2. Embeddings with VideoMAE-2

dataset: https://drive.google.com/drive/folders/1LVHGKiLZvyFoRh3PGqnBGilH2_rTCAUn

conda create -n videoMAE_env python=3.9
conda activate videoMAE_env
pip install opencv-python numpy matplotlib scikit-learn transformers torch torchvision av tqdm

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

```python
import os
import random
import cv2
import numpy as np
import torch
import av # For efficient video loading
from transformers import AutoImageProcessor, VideoMAEModel
import torchvision.transforms as transforms
from tqdm import tqdm # For progress bar
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE # For dimensionality reduction
from sklearn.preprocessing import StandardScaler # For scaling embeddings
from PIL import Image # Needed if get_frames_from_video needs to create dummy frames

# --- 0. 設定模型和影像處理器 ---
# 這將從 Hugging Face 下載預訓練的 VideoMAE-2 模型和其對應的影像處理器
print("載入 VideoMAE-2 模型和影像處理器...")
try:
    image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    model_videomae_base = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base", output_hidden_states=True)
    model_videomae_base.eval() # 設定模型為評估模式 (不進行訓練，關閉 Dropout 等)
    # 將模型移動到 GPU (如果可用)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_videomae_base.to(device)
    print(f"模型載入完成，使用裝置：{device}")
except Exception as e:
    print(f"錯誤：載入 VideoMAE-2 模型失敗。請檢查網路連線或庫安裝。錯誤訊息: {e}")
    exit()

# --- 1. 探索資料集結構 ---
print("\n--- 1. 探索資料集結構與影片驗證 ---")
dataset_base_path = r'D:\Apply jobs\SDSC\surgical_videos_demo'

# 檢查資料夾是否存在
if not os.path.exists(dataset_base_path):
    print(f"錯誤：資料集路徑 '{dataset_base_path}' 不存在。請確認路徑是否正確。")
    exit()

video_files_info = [] # 使用更具描述性的變數名
# 遍歷主資料夾下的所有子資料夾（例如 Cholecystectomy, pituitary_tumor_surgery）
for phase_folder in os.listdir(dataset_base_path):
    phase_folder_path = os.path.join(dataset_base_path, phase_folder)
    if os.path.isdir(phase_folder_path):
        print(f"  進入資料夾：{phase_folder}")
        # 遍歷子資料夾中的所有 .mp4 檔案
        for video_name in os.listdir(phase_folder_path):
            if video_name.lower().endswith('.mp4'):
                video_path = os.path.join(phase_folder_path, video_name)
                video_files_info.append({
                    'path': video_path,
                    'name': video_name,
                    'phase': phase_folder # 將子資料夾名稱作為影片的 phase 標籤
                })

print(f"\n在資料集中找到了 {len(video_files_info)} 個影片。")
if len(video_files_info) == 4:
    print("✓ 確認資料集中有 4 個影片。")
else:
    print(f"✗ 警告：預期有 4 個影片，但只找到 {len(video_files_info)} 個。")

print("\n--- 資料集中的影片列表 ---")
for video_info in video_files_info:
    print(f"  影片名稱: {video_info['name']}, 所屬階段: {video_info['phase']}, 路徑: {video_info['path']}")

# --- 2. 顯示隨機影格 ---
print("\n--- 2. 顯示隨機影格 ---")
num_frames_to_display_per_video = 3 # 每個影片顯示的隨機影格數量 (減少數量，避免圖形過大)

plt.figure(figsize=(15, 4 * len(video_files_info))) # 根據影片數量調整圖形高度
plot_idx = 1

for video_info in video_files_info:
    video_path = video_info['path']
    video_name = video_info['name']
    print(f"\n正在從影片 '{video_name}' 提取隨機影格...")

    container = None 
    try:
        container = av.open(video_path)
        # 使用 streams.video[0].frames 而不是 estimate_length() 更準確
        total_frames = container.streams.video[0].frames if container.streams.video else 0
        
        if total_frames == 0:
            print(f"警告：影片 '{video_name}' 沒有可讀取的影格或視訊流。")
            continue

        # 隨機選擇影格索引
        random_frame_indices = sorted(random.sample(range(total_frames), min(num_frames_to_display_per_video, total_frames)))

        retrieved_frames_count = 0
        for i, frame in enumerate(container.decode(video=0)):
            if i in random_frame_indices:
                img = frame.to_rgb().to_ndarray() # 將影格轉換為 NumPy 陣列
                
                plt.subplot(len(video_files_info), num_frames_to_display_per_video, plot_idx)
                plt.imshow(img)
                plt.title(f"{video_name}\nFrame {i+1}", fontsize=8)
                plt.axis('off')
                plot_idx += 1
                retrieved_frames_count += 1
            if retrieved_frames_count >= num_frames_to_display_per_video:
                break # 達到所需影格數量，停止解碼
            # 如果 i 已經超過最大隨機索引，也可提前終止，但這需要 `random_frame_indices` 是排序的
            if random_frame_indices and i > random_frame_indices[-1]:
                break

    except av.FFmpegError as e:
        print(f"錯誤：無法開啟或解碼影片 '{video_name}'。可能檔案損壞或缺乏編解碼器。錯誤訊息: {e}")
    except Exception as e:
        print(f"處理影片 '{video_name}' 時發生未知錯誤：{e}")
    finally:
        if container:
            container.close() # 確保容器被關閉

plt.tight_layout()
plt.show()

# --- 3. 提取 VideoMAE-2 嵌入並視覺化 ---
print("\n--- 3. 提取 VideoMAE-2 嵌入並視覺化 ---")

# VideoMAE 紙中提到使用 16 幀作為輸入
NUM_FRAMES = 16 
# 影片的採樣率：每 x 幀採樣一幀。
# 這裡假設您的 MP4 檔案本身就是 1 FPS 數據，或者我們想均勻採樣出 16 幀。
# 為了靈活性，讓 get_frames_from_video 根據影片長度來均勻選擇 16 幀。
SAMPLE_RATE = None # 設為 None 讓函數均勻採樣

# 儲存所有影片的嵌入和標籤
video_embeddings = []
video_labels = [] # 用於視覺化時為不同 phase 著色

# 定義預處理轉換 (針對每個從 av 讀取的 NumPy 陣列幀)
# image_processor 會處理大小調整、歸一化等，所以這裡只需要轉換為 PIL Image
transform = transforms.ToPILImage()

# 獲取影片幀的函數
def get_frames_from_video(video_path, num_frames=NUM_FRAMES):
    frames = []
    container = None
    try:
        container = av.open(video_path)
        total_video_frames = container.streams.video[0].frames if container.streams.video else 0
        
        if total_video_frames == 0:
            print(f"警告：影片 '{os.path.basename(video_path)}' 沒有可讀取的影格。")
            return []

        # 均勻採樣 num_frames
        if total_video_frames < num_frames:
            # 如果總幀數不足，則重複最後一幀來達到所需數量
            sample_indices = list(range(total_video_frames))
            last_frame_idx = sample_indices[-1] if sample_indices else 0
            sample_indices.extend([last_frame_idx] * (num_frames - total_video_frames))
        else:
            sample_indices = np.linspace(0, total_video_frames - 1, num_frames, dtype=int).tolist()
        
        # 使用 set 和 sorted 確保唯一性並維持順序
        sample_indices_set = set(sample_indices)
        
        # 實際讀取幀
        frame_buffer = {} # 用於儲存讀取到的幀，避免重複解碼
        for i, frame in enumerate(container.decode(video=0)):
            if i in sample_indices_set:
                frame_buffer[i] = transform(frame.to_rgb().to_ndarray())
                if len(frame_buffer) == len(sample_indices_set): # 讀取到所有需要的唯一幀就停止
                    break
        
        # 根據 `sample_indices` 的順序組裝 `frames` 列表
        frames = [frame_buffer[idx] for idx in sample_indices if idx in frame_buffer]
        
        # 再次檢查並填充，以防萬一（例如部分幀讀取失敗）
        while len(frames) < num_frames:
            if frames:
                frames.append(frames[-1]) # 重複最後一幀
            else:
                # 極端情況：如果影片完全無法讀取幀，填充黑色幀
                print(f"極端警告：影片 '{os.path.basename(video_path)}' 無法獲取任何有效影格，使用黑色幀填充。")
                frames.append(Image.fromarray(np.zeros((image_processor.size['height'], image_processor.size['width'], 3), dtype=np.uint8)))
                
    except av.FFmpegError as e:
        print(f"錯誤：影片 '{os.path.basename(video_path)}' 解碼失敗。錯誤訊息: {e}")
        return []
    except Exception as e:
        print(f"讀取影片 '{os.path.basename(video_path)}' 時發生未知錯誤：{e}")
        return []
    finally:
        if container:
            container.close()
    
    return frames


with torch.no_grad(): # 在推理模式下，不計算梯度以節省記憶體和加速
    for video_info in tqdm(video_files_info, desc="處理影片並提取嵌入"):
        video_path = video_info['path']
        video_phase = video_info['phase']
        
        # 獲取採樣後的幀
        pixel_values_list = get_frames_from_video(video_path, num_frames=NUM_FRAMES)
        
        if not pixel_values_list: # 如果無法獲取幀，則跳過此影片
            print(f"跳過影片 '{video_info['name']}'，因為無法獲取有效幀。")
            continue

        # 將幀列表轉換為批次張量，並應用 image_processor
        # image_processor 期望一個 list of PIL Images 或 numpy arrays (H, W, C)
        # 它會自動處理成 [batch_size, num_channels, num_frames, height, width] 格式
        inputs = image_processor(pixel_values_list, return_tensors="pt")
        
        # 將輸入移動到 GPU (如果可用)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 獲取模型輸出
        outputs = model_videomae_base(**inputs)
        
        # 提取 CLS token 的特徵作為整個影片的嵌入
        # last_hidden_state 的形狀通常是 (batch_size, num_patches*num_frames + 1, hidden_dim)
        # 第一個 token 是 CLS token，代表整個序列的總結
        # 由於我們每次處理一個影片，batch_size=1
        video_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy() # 提取 CLS token，移除 batch_size 維度，轉為 numpy
        video_embeddings.append(video_embedding)
        video_labels.append(video_phase)

# 將所有嵌入轉換為 NumPy 陣列
video_embeddings = np.array(video_embeddings)

# 確保有足夠的樣本進行 t-SNE
if len(video_embeddings) < 2:
    print("\n沒有足夠的影片樣本來生成 t-SNE 嵌入圖 (至少需要 2 個)。")
else:
    # 標準化嵌入 (可選但推薦，有助於 t-SNE 表現)
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(video_embeddings)

    # 使用 t-SNE 進行降維
    # perplexity 應小於樣本數，通常建議在 5 到 50 之間。
    # 這裡確保它不超過樣本數減一。
    perplexity_val = min(5, len(scaled_embeddings) - 1) 
    if perplexity_val <= 0: # 如果只有1個樣本，perplexity 無法計算
        print("\n無法計算 t-SNE，因為樣本數過少 (可能只有 1 個有效影片)。")
    else:
        print(f"\n正在使用 t-SNE (perplexity={perplexity_val}) 將 {video_embeddings.shape[1]} 維嵌入降維到 2D 進行視覺化...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_val)
        embeddings_2d = tsne.fit_transform(scaled_embeddings)

        # 視覺化嵌入圖
        plt.figure(figsize=(10, 8))
        unique_labels = list(set(video_labels))
        
        # 修正這裡: 使用 plt.cm.get_cmap 或直接使用 colormap 物件
        if len(unique_labels) == 1:
            colors = ['blue'] # 給單一類別一個固定顏色
        else:
            # 從 colormap 物件獲取顏色
            cmap = plt.cm.viridis # 正確的 colormap 引用方式
            colors = [cmap(i) for i in np.linspace(0, 1, len(unique_labels))]

        # 創建一個字典，將標籤映射到顏色
        label_to_color = {label: color for label, color in zip(unique_labels, colors)}

        # 收集用於圖例的 handle 和 label，避免重複
        legend_handles = []
        legend_labels = []

        for i, label in enumerate(video_labels):
            color = label_to_color[label]
            # Plot the scatter point
            scatter_handle = plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], color=color, 
                        s=100, alpha=0.8, edgecolors='w', linewidth=0.5) # 增加邊緣線，使點更明顯
            
            # Add text label for the video name, slightly offset
            plt.text(embeddings_2d[i, 0] + 0.05 * (embeddings_2d[:, 0].max() - embeddings_2d[:, 0].min()), 
                     embeddings_2d[i, 1] + 0.05 * (embeddings_2d[:, 1].max() - embeddings_2d[:, 1].min()), 
                     video_files_info[i]['name'], fontsize=8) 
            
            # Add to legend handles/labels if not already present
            if label not in legend_labels:
                legend_handles.append(scatter_handle)
                legend_labels.append(label)

        plt.title("VideoMAE-2 Embeddings of Surgical Videos (t-SNE 2D Projection)")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        # 使用收集到的 handles 和 labels 創建圖例
        plt.legend(legend_handles, legend_labels, title="Surgical Phase", loc='best') 
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

print("\n程式碼執行完畢。🎉")
```

解釋:
[1]
**`image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")`**

- **作用**：這一行程式碼會從 Hugging Face 的模型中心（Hugging Face Hub）下載並載入與 **VideoMAE-2 基礎模型 (videomae-base)** 相匹配的**影像處理器 (Image Processor)**。這個處理器的作用是確保您輸入給模型的影像或影格數據符合模型預期的格式和預處理要求。
    
- **具體例子**：
    - 假設 VideoMAE-2 模型期望的輸入影格大小是 224x224 像素，並且像素值在 0 到 1 之間（歸一化）。
    - 當您給 `image_processor` 一張原始的 1920x1080 像素的 JPG 圖片（像素值 0-255）時，`image_processor` 會自動幫您完成以下操作：
        - **縮放 (Resizing)**：將圖片縮小或放大到 224x224 像素。
        - **中心裁剪 (Center Cropping)** 或其他裁剪方式，以確保圖片比例合適。
        - **歸一化 (Normalization)**：將像素值從 0-255 的範圍轉換到 0-1 或其他模型期望的範圍（通常是減去平均值，除以標準差）。
        - **格式轉換**：將圖片數據轉換為 PyTorch 張量（Tensor）並調整通道順序（例如從 HWC 轉換為 CHW）。

[2]
**`model_videomae_base = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base", output_hidden_states=True)`**

- **作用**：這行程式碼也是從 Hugging Face Hub 下載並載入**預訓練好的 VideoMAE-2 模型**。`"MCG-NJU/videomae-base"` 是這個模型的唯一識別符。
- `output_hidden_states=True` 是一個重要的參數。它告訴模型不僅要計算最終的輸出（例如分類分數），還要保留**模型每一層的隱藏狀態 (hidden states)**。
    
- **具體例子**：
    - VideoMAE-2 是一個 Transformer 模型，它有多個編碼器層。每層都會處理輸入並生成一個「更抽象」的表示。
    - 當 `output_hidden_states=True` 時，模型會返回一個包含所有層隱藏狀態的元組（tuple）。
    - 我們特別關心的是**最後一層的隱藏狀態**，其中的**第一個 token (CLS token)** 通常被訓練成代表整個輸入序列（即整個影片片段）的**總結性嵌入 (summary embedding)**。這個嵌入就是我們用來視覺化影片相似度的關鍵。

[3]
- **作用**：將模型設定為**評估 (evaluation) 模式**。這與訓練模式相對。
    
- **具體例子**：
    - 在深度學習模型中，有些層（如 **Dropout** 和 **Batch Normalization**）在訓練和評估時行為不同。
    - **Dropout**：在訓練時會隨機關閉部分神經元以防止過擬合，但在評估時必須關閉，以確保結果的穩定性和可重現性。
    - **Batch Normalization**：在訓練時使用當前批次的統計數據來歸一化，但在評估時使用訓練時計算的全局平均值和標準差。
    - 呼叫 `model.eval()` 確保這些層以它們在推理（評估）時應有的方式運行，從而獲得預期且一致的結果。

[4]
**`device = "cuda" if torch.cuda.is_available() else "cpu"` 和 `model_videomae_base.to(device)`**

- **作用**：這些行程式碼用於檢測您的電腦是否安裝了 NVIDIA GPU 並配置了 CUDA。
- 如果 `torch.cuda.is_available()` 返回 `True`（表示有 GPU），則 `device` 會被設定為 `"cuda"`。否則，它會被設定為 `"cpu"`。
- `model_videomae_base.to(device)` 的作用是將整個模型（包括其所有參數和計算）從預設的 CPU 記憶體移動到 GPU 記憶體中。
    
- **具體例子**：
    - 如果您的電腦有兼容的 GPU，將模型放到 GPU 上可以**極大地加速**計算。處理一個影片可能從幾秒鐘縮短到幾毫秒。
    - 如果沒有 GPU，模型將在 CPU 上運行，雖然速度較慢，但程式仍然可以正常執行。

[5]
**`with torch.no_grad():`**

- **作用**：這個上下文管理器告訴 PyTorch **不要計算和儲存梯度 (gradients)**。在深度學習中，梯度主要用於訓練模型（更新模型權重），而在推理或特徵提取時不需要。
    
- **具體例子**：
    - 想像您正在烘焙蛋糕。訓練模式就像您在嘗試不同的配方，每調整一點配料都需要記錄效果（梯度）以便下次改進。
    - 而 `torch.no_grad()` 就像您已經有了完美的配方，現在只是按照配方烘焙蛋糕。您不需要記錄每一步的變化，這會**節省大量記憶體和計算時間**，因為不需要建立計算圖來追蹤梯度。

[6]








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


```python
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from transformers import VideoMAEFeatureExtractor, VideoMAEForVideoClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from PIL import Image # 用於載入 JPEG 影像
import glob # 用於查找影像檔案
import warnings

# 抑制來自 transformers 庫的警告
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# --- 配置 ---
# SOCAL 資料集資料夾路徑
SOCAL_DATA_FOLDER = r"D:\Apply jobs\SDSC\SOCAL"
# 成果 CSV 檔案路徑
OUTCOMES_CSV_PATH = os.path.join(SOCAL_DATA_FOLDER, "socal_trial_outcomes.csv")
# JPEG 影像子資料夾名稱
JPEG_IMAGES_SUBFOLDER = "JPEGImages"
# 影像檔案擴展名
IMAGE_FILE_EXTENSION = ".jpeg"

# 模型參數
MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics" # 使用預訓練的 VideoMAE 基礎模型
NUM_FRAMES = 16 # 從每個試驗中採樣的影像幀數量，這是 VideoMAE 模型輸入的典型值
IMAGE_SIZE = 224 # VideoMAE 的輸入影像大小
BATCH_SIZE = 4 # 根據 GPU 記憶體調整。較小的批次大小可減少記憶體使用。
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3 # 從少量 epoch 開始進行初始測試，以獲得更好的性能。
TRAIN_TEST_SPLIT_RATIO = 0.8 # 80% 用於訓練，20% 用於測試

# --- 裝置配置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置: {device}")
if device.type == 'cuda':
    print(f"CUDA 裝置名稱: {torch.cuda.get_device_name(0)}")
    # 如果 GPU 可用，啟用混合精度訓練以加速訓練並減少記憶體使用
    scaler = torch.cuda.amp.GradScaler()
else:
    scaler = None

# --- 用於高效影像載入的自訂資料集類別 ---
class SOCALImageSequenceDataset(Dataset):
    """
    一個用於載入 SOCAL 影像序列資料和對應結果的自訂 PyTorch 資料集。
    透過即時載入影像幀來優化記憶體。
    """
    def __init__(self, dataframe, image_dir, feature_extractor, num_frames=NUM_FRAMES):
        """
        初始化資料集。

        Args:
            dataframe (pd.DataFrame): 包含影像序列元資料和 'success' 標籤的 DataFrame。
            image_dir (str): 儲存 JPEG 影像檔案的目錄。
            feature_extractor (VideoMAEFeatureExtractor): 用於 VideoMAE 的預訓練特徵提取器。
            num_frames (int): 從每個影像序列中採樣的影像幀數量。
        """
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.feature_extractor = feature_extractor
        self.num_frames = num_frames
        # 注意：VideoMAEFeatureExtractor 會處理影像的縮放、歸一化和轉換為張量。
        # 因此，這裡的 self.transform 可能不會直接使用，但保留以供參考或未來需要。
        self.transform = Compose([
            Resize((IMAGE_SIZE, IMAGE_SIZE)),
            ToTensor(),
            Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
        ])

        # 預掃描影像目錄，以建立 trial_id 到其影像幀路徑的映射。
        # 這有助於在 __getitem__ 期間高效地採樣影像幀。
        self.trial_frames_map = self._build_trial_frames_map()
        
        # 過濾 dataframe，只包含找到影像幀的 trial_id
        initial_len = len(self.dataframe)
        self.dataframe = self.dataframe[self.dataframe['trial_id'].isin(self.trial_frames_map.keys())].reset_index(drop=True)
        if len(self.dataframe) < initial_len:
            print(f"警告: 已過濾掉 {initial_len - len(self.dataframe)} 個沒有找到對應影像幀的試驗。")
        print(f"資料集已初始化，包含 {len(self.dataframe)} 個有效樣本（在影像幀映射後）。")

    def _build_trial_frames_map(self):
        """
        掃描 JPEGImages 資料夾，並建立一個將 trial_id 映射到其所有影像幀路徑的字典。
        影像幀路徑會按名稱排序，以確保時間順序。
        """
        trial_map = {}
        # 獲取目錄中所有 JPEG 影像的路徑
        all_image_paths = glob.glob(os.path.join(self.image_dir, f"*{IMAGE_FILE_EXTENSION}"))
        print(f"在 {self.image_dir} 中找到 {len(all_image_paths)} 個 JPEG 影像。")

        for img_path in all_image_paths:
            # 從檔名中提取 trial_id (例如: S102T1_frame_00000002.jpeg -> S102T1)
            filename = os.path.basename(img_path)
            parts = filename.split('_')
            if len(parts) >= 2:
                trial_id = parts[0] # 假設 trial_id 是 _frame_ 前的第一部分
                if trial_id not in trial_map:
                    trial_map[trial_id] = []
                trial_map[trial_id].append(img_path)
            # else:
            #     print(f"警告: 無法從檔名中解析 trial_id: {filename}") # 避免過多警告訊息

        # 對每個 trial_id 的影像幀進行排序，以確保時間順序
        for trial_id in trial_map:
            trial_map[trial_id].sort() # 依檔名排序，這應該會依影像幀編號排序
        return trial_map

    def __len__(self):
        """返回資料集中的總樣本數。"""
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        透過索引檢索單個樣本（影像幀和標籤）。

        Args:
            idx (int): 要檢索的樣本索引。

        Returns:
            dict: 包含 'pixel_values'（處理後的影像幀）和 'labels'（'success' 結果）的字典。
        """
        row = self.dataframe.iloc[idx]
        trial_id = row['trial_id']
        label = row['success']

        frame_paths = self.trial_frames_map.get(trial_id)
        if not frame_paths:
            print(f"錯誤: 未找到 trial_id {trial_id} 的影像幀。返回虛擬樣本。")
            return {
                'pixel_values': torch.zeros(1, 3, self.num_frames, IMAGE_SIZE, IMAGE_SIZE),
                'labels': torch.tensor(-1) # 使用 -1 作為無效樣本的標誌
            }

        total_frames_in_trial = len(frame_paths)
        
        # 從可用的影像幀中均勻採樣 NUM_FRAMES 個索引
        if total_frames_in_trial < self.num_frames:
            # 如果影像幀數量不足，則重複採樣以達到所需數量
            indices = np.random.choice(total_frames_in_trial, self.num_frames, replace=True)
            indices.sort() # 確保重複採樣後仍保持時間順序
            # print(f"警告: trial {trial_id} 的影像幀不足 ({total_frames_in_trial})。所需: {self.num_frames}。重複採樣影像幀。")
        else:
            indices = np.linspace(0, total_frames_in_trial - 1, self.num_frames, dtype=int)

        selected_frames = []
        for i in indices:
            try:
                # 使用 PIL 載入影像並轉換為 RGB 格式
                img = Image.open(frame_paths[i]).convert("RGB")
                selected_frames.append(img)
            except Exception as e:
                print(f"錯誤載入影像幀 {frame_paths[i]}: {e}。跳過此影像幀。")
                # 如果影像幀載入失敗，則填充一個黑色影像或跳過
                selected_frames.append(Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))) # 黑色影像

        if not selected_frames: # 如果所有影像幀都載入失敗
             print(f"錯誤: trial_id {trial_id} 沒有載入任何有效影像幀。返回虛擬樣本。")
             return {
                'pixel_values': torch.zeros(1, 3, self.num_frames, IMAGE_SIZE, IMAGE_SIZE),
                'labels': torch.tensor(-1) # 使用 -1 作為無效樣本的標誌
            }

        # VideoMAEFeatureExtractor 會在內部處理縮放、歸一化和轉換為張量
        # 它期望一個 PIL 影像或 numpy 陣列的列表
        inputs = self.feature_extractor(selected_frames, return_tensors="pt")
        # 移除 feature_extractor 添加的批次維度
        pixel_values = inputs['pixel_values'].squeeze(0) 

        return {
            'pixel_values': pixel_values,
            'labels': torch.tensor(label, dtype=torch.long)
        }

# --- 主腳本 ---
def main():
    # 1. 載入資料
    try:
        df = pd.read_csv(OUTCOMES_CSV_PATH)
        print(f"成功從 {OUTCOMES_CSV_PATH} 載入 {len(df)} 行資料。")
        # 確保 'success' 列存在且為數值型
        if 'success' not in df.columns:
            raise ValueError(f"在 {OUTCOMES_CSV_PATH} 中未找到 'success' 列。")
        if not pd.api.types.is_numeric_dtype(df['success']):
            raise ValueError(f"在 {OUTCOMES_CSV_PATH} 中的 'success' 列不是數值型。")
        # 確保 'trial_id' 列存在以進行影像映射
        if 'trial_id' not in df.columns:
            raise ValueError(f"在 {OUTCOMES_CSV_PATH} 中未找到 'trial_id' 列。請指定正確的影像 ID 列。")
        
    except FileNotFoundError:
        print(f"錯誤: 未找到 CSV 檔案 {OUTCOMES_CSV_PATH}。請檢查路徑。")
        return
    except ValueError as e:
        print(f"資料載入錯誤: {e}")
        return
    except Exception as e:
        print(f"資料載入期間發生意外錯誤: {e}")
        return

    # 2. 初始化特徵提取器和模型
    print("正在初始化 VideoMAE 特徵提取器和模型...")
    feature_extractor = VideoMAEFeatureExtractor.from_pretrained(MODEL_NAME)
    # 模型將自動為 2 個標籤（二元分類）調整分類器頭部
    model = VideoMAEForVideoClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)
    print("模型和特徵提取器已初始化。")

    # 3. 建立資料集和資料載入器
    image_data_dir = os.path.join(SOCAL_DATA_FOLDER, JPEG_IMAGES_SUBFOLDER)
    if not os.path.isdir(image_data_dir):
        print(f"錯誤: 未找到影像目錄 {image_data_dir}。請確保 JPEG 影像位於正確的子資料夾中。")
        return

    full_dataset = SOCALImageSequenceDataset(df, image_data_dir, feature_extractor, NUM_FRAMES)

    # 過濾掉無效樣本（影像幀未找到或載入失敗）
    valid_indices = [i for i, data in enumerate(full_dataset) if data['labels'].item() != -1]
    if len(valid_indices) < len(full_dataset):
        print(f"已過濾掉 {len(full_dataset) - len(valid_indices)} 個無效樣本。")
        # 建立一個只包含有效樣本的新資料集
        df_valid = full_dataset.dataframe.iloc[valid_indices].reset_index(drop=True)
        full_dataset = SOCALImageSequenceDataset(df_valid, image_data_dir, feature_extractor, NUM_FRAMES)
        print(f"資料集現在包含 {len(full_dataset)} 個有效樣本。")
        if len(full_dataset) == 0:
            print("未找到有效樣本。正在退出。")
            return

    # 將資料集分為訓練集和驗證集
    train_size = int(TRAIN_TEST_SPLIT_RATIO * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"訓練樣本: {len(train_dataset)}, 驗證樣本: {len(val_dataset)}")

    # 建立資料載入器，並使用多個 worker 進行並行載入
    # num_workers > 0 有助於 CPU 綁定的資料載入，釋放 GPU 進行訓練
    # 在 Windows 上，num_workers > 0 可能需要 __name__ == '__main__' 保護。
    # 為了簡潔起見，如果不在主保護中或在 CPU 上，則設定 num_workers=0
    num_workers = 4 if device.type == 'cuda' and os.name != 'nt' else 0 # 根據您的 CPU 核心數調整
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True # 在返回之前將張量複製到 CUDA 釘選記憶體
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    print("資料載入器已建立。")

    # 4. 訓練設定
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # 使用 CrossEntropyLoss 進行多類別分類（即使是二元分類，它也適用）
    # 它內部應用 softmax，然後是 NLLLoss。
    loss_fn = torch.nn.CrossEntropyLoss()
    print("訓練設定完成。")

    # 5. 訓練迴圈
    print("開始訓練...")
    for epoch in range(NUM_EPOCHS):
        model.train() # 將模型設定為訓練模式
        total_loss = 0
        for batch_idx, batch in enumerate(train_dataloader):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad() # 清除梯度

            # 使用自動混合精度 (AMP) 進行更快的訓練和減少記憶體使用
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss # Hugging Face 模型在提供標籤時直接返回損失
                scaler.scale(loss).backward() # 為 AMP 縮放損失
                scaler.step(optimizer) # 更新優化器
                scaler.update() # 更新 scaler 以進行下一次迭代
            else:
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Batch {batch_idx+1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} 完成。平均訓練損失: {avg_train_loss:.4f}")

        # 6. 評估迴圈
        model.eval() # 將模型設定為評估模式
        all_preds = []
        all_labels = []
        val_loss = 0
        print("開始評估...")
        with torch.no_grad(): # 在評估期間禁用梯度計算
            for batch_idx, batch in enumerate(val_dataloader):
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)

                # 評估時也使用 AMP，儘管對於記憶體來說並非絕對必要
                if scaler:
                    with torch.cuda.amp.autocast():
                        outputs = model(pixel_values=pixel_values, labels=labels)
                else:
                    outputs = model(pixel_values=pixel_values, labels=labels)

                logits = outputs.logits
                loss = outputs.loss
                val_loss += loss.item()

                predictions = torch.argmax(logits, dim=-1) # 獲取預測類別 (0 或 1)
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)

        print(f"\n--- Epoch {epoch+1} 的評估結果 ---")
        print(f"平均驗證損失: {avg_val_loss:.4f}")
        print(f"準確度 (Accuracy): {accuracy:.4f}")
        print(f"精確度 (Precision): {precision:.4f}")
        print(f"召回率 (Recall): {recall:.4f}")
        print(f"F1-分數 (F1-Score): {f1:.4f}")
        print("\n分類報告:")
        print(classification_report(all_labels, all_preds, target_names=['failure', 'success'], zero_division=0))
        print("-" * 40)

    print("\n訓練和評估完成！")

    # 可選: 儲存訓練好的模型
    # model_save_path = "./videomae_socal_model"
    # model.save_pretrained(model_save_path)
    # feature_extractor.save_pretrained(model_save_path)
    # print(f"模型已儲存到 {model_save_path}")

if __name__ == '__main__':
    # 這個保護對於在 Windows 上進行多處理 (num_workers > 0) 很重要
    main()
```

- **資料載入與預處理**：從 `socal_trial_outcomes.csv` 載入試驗結果，並掃描 `JPEGImages` 資料夾以建立 `trial_id` 到其對應 JPEG 影像幀路徑的映射。
    
- **客製化資料集**：建立一個 PyTorch `Dataset`，它會根據 `trial_id` 動態載入選定的 JPEG 影像幀，而不是預先載入所有資料，從而優化記憶體使用。
    
- **VideoMAE 模型設定**：使用預訓練的 VideoMAE 模型，並將其分類頭部調整為預測二元「成功」結果。
    
- **訓練與評估**：執行模型訓練迴圈，並在驗證集上評估模型的性能，包括準確度、精確度、召回率和 F1 分數。
    
- **優化**：
    - **記憶體優化**：
        - **即時影像幀載入**：`SOCALVideoDataset` 只在需要時載入 JPEG 影像幀，而不是將所有影像幀載入記憶體
        - **批次處理**：`DataLoader` 以批次處理資料，確保在任何時候只有少量資料在記憶體中。
        - **`pin_memory=True`**：當使用 GPU 時，這有助於加速資料從 CPU 到 GPU 的傳輸。
        - **自動混合精度 (AMP)**：如果檢測到 CUDA GPU，將使用 `torch.cuda.amp` 進行混合精度訓練，顯著減少記憶體佔用並加速計算。
            
    - **執行時間優化**：
        - **預訓練 VideoMAE**：使用預訓練模型可以大幅縮短訓練時間。
        - **`DataLoader` 的 `num_workers`**：啟用多個 CPU 進程並行載入資料，減少資料準備造成的瓶頸。
        - **GPU 加速**：自動檢測並使用 CUDA GPU 進行訓練。
        - **`torch.no_grad()`**：在評估期間禁用梯度計算，節省記憶體並加速推斷。


這個修改後的 Python 腳本專為使用 SOCAL 資料集中的 JPEG 影像幀訓練 VideoMAE 模型而設計，同時高度關注記憶體和執行時間的優化。

#### 1. 配置 (Configuration)

- `SOCAL_DATA_FOLDER`: 您的 SOCAL 資料集根目錄。
    
- `OUTCOMES_CSV_PATH`: 包含試驗結果（包括 `trial_id` 和 `success` 欄位）的 CSV 檔案路徑。
    
- `JPEG_IMAGES_SUBFOLDER`: 包含所有 JPEG 影像幀的子資料夾名稱，例如 `JPEGImages`。
    
- `NUM_FRAMES`: 每個影片序列將採樣的影像幀數量。VideoMAE 模型期望固定數量的幀作為輸入。
    
- `IMAGE_SIZE`: 輸入影像幀的解析度，通常為 224x224 像素。
    
- `BATCH_SIZE`: 訓練期間每個批次處理的樣本數。較小的批次大小有助於減少記憶體消耗。
    
- `MODEL_NAME`: 使用預訓練的 VideoMAE 模型名稱。預訓練模型可以大幅加速訓練過程。
    

#### 2. 裝置配置 (Device Configuration)

- 腳本會自動檢測您的系統是否有可用的 CUDA GPU。如果有的話，它將使用 GPU 進行訓練，這比 CPU 快得多。
    
- **`torch.cuda.amp.GradScaler()` (自動混合精度 - AMP)**：這是一個關鍵的記憶體和速度優化。AMP 允許模型在訓練期間使用較低精度的浮點數（例如 FP16），這可以將記憶體使用量減少一半，並在支援的硬體上加速計算。
    

#### 3. `SOCALImageSequenceDataset` (自訂資料集類別)

這是此腳本的核心，專為處理大量 JPEG 影像幀而設計，並進行了記憶體優化：

- **`__init__(...)` 初始化方法**：
    
    - 它接收一個 `DataFrame`（來自 `socal_trial_outcomes.csv`）、影像資料夾路徑和特徵提取器。
        
    - **`_build_trial_frames_map()` (預掃描影像)**：這是記憶體優化的關鍵。在資料集初始化時，腳本會遍歷 `JPEGImages` 資料夾中的所有 JPEG 檔案。它會根據檔名（例如 `S102T1_frame_00000002.jpeg`）解析出 `trial_id`，並建立一個字典 `self.trial_frames_map`，將每個 `trial_id` 映射到其所有對應影像幀的排序列表。這樣，在訓練期間就不需要重複掃描磁碟，提高了效率。
        
    - **過濾 DataFrame**：在建立映射後，腳本會過濾原始 `DataFrame`，只保留那些在 `JPEGImages` 資料夾中實際找到影像幀的 `trial_id`，確保資料的有效性。
        
- **`__len__()` 方法**：返回資料集中的總樣本數（即 `DataFrame` 中的行數）。
    
- **`__getitem__(idx)` 方法**：
    
    - 這是 PyTorch `DataLoader` 在需要時調用以獲取單個樣本的方法。
        
    - 它根據 `idx` 獲取對應的 `trial_id` 和 `success` 標籤。
        
    - **即時影像幀載入**：它從 `self.trial_frames_map` 中獲取該 `trial_id` 的所有影像幀路徑。
        
    - **均勻採樣**：從該 `trial_id` 的所有可用影像幀中，它會**均勻地採樣 `NUM_FRAMES` 個影像幀**。如果可用幀數少於 `NUM_FRAMES`，它會通過重複採樣現有幀來達到所需數量，以確保模型輸入的維度一致性。
        
    - **`PIL.Image.open()` 載入**：每個選定的 JPEG 影像幀都會使用 `PIL.Image.open()` 即時載入到記憶體中，然後轉換為 RGB 格式。這避免了一次性載入所有影像幀到記憶體中。
        
    - **`feature_extractor` 處理**：載入的影像幀列表會傳遞給 `VideoMAEFeatureExtractor`。這個提取器會自動處理影像的縮放、歸一化和轉換為模型所需的張量格式。
        

#### 4. 主腳本 (`main()` 函數)

- **資料載入**：使用 `pandas` 載入 `socal_trial_outcomes.csv`，並進行基本的錯誤檢查，確保 `success` 和 `trial_id` 列存在。
    
- **模型和特徵提取器初始化**：
    
    - 載入預訓練的 `VideoMAEFeatureExtractor` 和 `VideoMAEForVideoClassification` 模型。`num_labels=2` 會自動調整模型的分類頭部以輸出二元預測。
        
    - 將模型移動到檢測到的裝置（GPU 或 CPU）。
        
- **資料集和資料載入器建立**：
    
    - 實例化 `SOCALImageSequenceDataset`。
        
    - **過濾無效樣本**：在建立資料集後，會再次檢查並過濾掉任何在 `__getitem__` 中返回虛擬樣本（例如，因為影像幀完全缺失或載入失敗）的資料。
        
    - 將資料集分為訓練集和驗證集。
        
    - **`DataLoader` 與 `num_workers`**：建立訓練和驗證資料載入器。`num_workers` 參數允許 PyTorch 使用多個子進程來載入資料，這可以顯著加速資料預處理，防止 GPU 在等待資料時閒置。在 Windows 上，使用 `num_workers > 0` 通常需要將主執行邏輯包裹在 `if __name__ == '__main__':` 塊中。
        
    - **`pin_memory=True`**：此選項指示 PyTorch 將載入的資料複製到 CUDA 釘選記憶體中，這可以加速資料從 CPU 到 GPU 的傳輸。
        
- **訓練設定**：使用 `AdamW` 優化器和 `CrossEntropyLoss` 損失函數。
    
- **訓練迴圈**：
    
    - 模型設定為訓練模式 (`model.train()`)。
        
    - 遍歷訓練資料載入器中的每個批次。
        
    - **自動混合精度 (AMP)**：如果 `scaler` 存在（即有 GPU），則使用 `torch.cuda.amp.autocast()` 包裹前向傳播，並使用 `scaler.scale()` 和 `scaler.step()` 進行反向傳播和優化器更新。這最大限度地利用了 GPU 性能和記憶體。
        
    - 計算並列印每個批次的損失。
        
- **評估迴圈**：
    
    - 模型設定為評估模式 (`model.eval()`)。
        
    - 使用 `torch.no_grad()` 禁用梯度計算，這可以節省記憶體並加速評估。
        
    - 遍歷驗證資料載入器中的每個批次。
        
    - 收集所有預測和真實標籤。
        
    - 計算並列印準確度、精確度、召回率、F1 分數和詳細的分類報告。
        

#### 總結優化點

- **記憶體優化**：
    
    - **即時影像幀載入**：避免一次性將所有影像幀載入記憶體，而是根據需要載入選定的幀。
        
    - **預掃描映射**：在資料集初始化時建立 `trial_id` 到影像幀路徑的映射，避免在每個 `__getitem__` 調用中重複掃描磁碟。
        
    - **批次處理**：`DataLoader` 確保每次只處理一小部分資料。
        
    - **自動混合精度 (AMP)**：顯著減少 GPU 記憶體使用。
        
- **執行時間優化**：
    
    - **預訓練模型**：利用預訓練模型的強大表示能力，減少從頭開始訓練所需的時間。
        
    - **多進程資料載入 (`num_workers`)**：在 CPU 上並行載入和預處理資料，確保 GPU 始終有資料可處理。
        
    - **GPU 加速**：利用 CUDA GPU 進行高效的矩陣運算。
        
    - **`pin_memory`**：加速資料從 CPU 到 GPU 的傳輸。
        
    - **`torch.no_grad()`**：在評估期間跳過不必要的梯度計算。