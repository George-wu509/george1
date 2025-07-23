
https://sgfdx5oxvnhbdno.studio.us-east-2.sagemaker.aws/jupyterlab/default/lab/tree/Untitled.ipynb
on jupyterLab

```python
# SageMaker Studio 預裝了很多套件，但我們需要確保有最新版的
# transformers、datasets 和 evaluate (用於 Hugging Face 生態系)
!pip install -q "transformers>=4.38.0" "datasets[video]" "evaluate" "accelerate"
```


```python
from huggingface_hub import notebook_login
notebook_login()
```


```python
import os
import glob
from datasets import Dataset, DatasetDict

# --- 1. 定義資料集根目錄 ---
dataset_path = "./UCF101_subset"
print(f"開始從 '{dataset_path}' 手動掃描檔案...")

# --- 2. 手動獲取所有檔案路徑和標籤 ---
def load_file_paths(split_dir):
    # split_dir 應為 'train' 或 'test'
    full_path = os.path.join(dataset_path, split_dir)
    # 假設影片格式為 .avi，這對 UCF 資料集是常見的
    # 使用 glob 安全地尋找所有 .avi 檔案
    filepaths = glob.glob(f"{full_path}/*/*.avi")
    
    # 從檔案路徑中提取標籤 (即倒數第二層的資料夾名稱)
    labels = [os.path.basename(os.path.dirname(p)) for p in filepaths]
    
    return {"video": filepaths, "label_str": labels}

try:
    train_data = load_file_paths("train")
    test_data = load_file_paths("test")

    print(f"找到 {len(train_data['video'])} 個訓練樣本和 {len(test_data['video'])} 個測試樣本。")

    # --- 3. 將字串標籤轉換為數字 ID ---
    # 建立標籤與 ID 的映射
    all_labels = sorted(list(set(train_data["label_str"])))
    label2id = {label: i for i, label in enumerate(all_labels)}
    id2label = {i: label for i, label in enumerate(all_labels)}

    print(f"\n成功建立標籤映射，共 {len(all_labels)} 個類別: {all_labels}")
    
    # 應用映射
    train_data["label"] = [label2id[name] for name in train_data["label_str"]]
    test_data["label"] = [label2id[name] for name in test_data["label_str"]]

    # --- 4. 從手動建立的清單創建 Dataset 物件 ---
    # 使用 from_dict 從我們的 Python 字典創建 Dataset
    train_dataset = Dataset.from_dict(train_data)
    test_dataset = Dataset.from_dict(test_data)

    # 將它們合併成一個 DatasetDict，這是 Trainer API 偏好的格式
    dataset = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })
    
    # 移除不再需要的字串標籤欄位
    dataset = dataset.remove_columns("label_str")

    print("\n✅ 成功手動建立並載入資料集！")
    print(dataset)

except Exception as e:
    print(f"\n處理檔案時發生錯誤: {e}")
    print("請確保解壓縮後的資料夾結構為 './UCF101_subset/train/[CLASS_NAME]/*.avi'")
```


```python
from transformers import AutoImageProcessor
import torch

# 從模型 checkpoint 載入對應的圖片/影片處理器
model_checkpoint = "OpenGVLab/VideoMAEv2-Base"
image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)

# 建立一個整理函式 (collate_fn)，將每個影片樣本處理成一個批次 (batch)
def collate_fn(examples):
    # 移除影片檔案路徑，因為它們無法轉換成 Tensor
    video_paths = [example.pop("video") for example in examples]

    # 從每個影片中提取 16 幀
    # 這是 VideoMAE 的標準輸入格式
    pixel_values = [image_processor(list(video.iter_frames()), return_tensors="pt").pixel_values for video in video_paths]

    # 將多個影片的 pixel_values 堆疊成一個批次
    pixel_values = torch.stack(pixel_values)

    # 處理標籤
    labels = torch.tensor([example["label"] for example in examples], dtype=torch.long)

    return {"pixel_values": pixel_values, "labels": labels}
```


```python

```


```python

```