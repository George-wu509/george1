
我們將引入 **Amazon SageMaker** 來實現 AWS 上的訓練。SageMaker 是 AWS 提供的託管機器學習平台，能簡化模型訓練、部署與管理的流程。

我們的策略如下：

1. **資料準備**：將您的 `SOCAL` 資料集從 Google Drive 上傳到 Amazon S3 (Simple Storage Service)，這是 AWS 上標準的物件儲存服務，也是 SageMaker 訓練任務慣用的資料來源。
    
2. **程式碼調整**：將您的訓練腳本（主要是模型定義、資料集類別和訓練迴圈）包裝成一個獨立的 Python 檔案。
    
3. **使用 SageMaker SDK**：在 Colab (或任何本地環境) 中，使用 SageMaker Python SDK 來設定並啟動一個遠端的 AWS 訓練任務 (Training Job)。這個任務會在您指定的 AWS 雲端伺服器 (例如，配備 GPU 的 EC2 執行個體) 上執行您的訓練腳本。
    

---

### 第一步：將資料上傳到 Amazon S3

在開始之前，您需要有一個 AWS 帳戶，並完成以下前置作業：

1. **建立 S3 儲存桶 (Bucket)**：登入 AWS 管理控制台，前往 S3 服務，建立一個新的儲存桶。這個儲存桶將用來存放您的 SOCAL 資料集和模型輸出。假設我們將儲存桶命名為 `socal-videomae-dataset`。
    
2. **上傳資料**：將您的 `SOCAL` 資料夾（包含 `JPEGImages` 和 `socal_trial_outcomes.csv`）從 Google Drive 下載到本機，然後再上傳到您剛剛建立的 S3 儲存桶中。上傳後，S3 中的路徑結構應該看起來像這樣：
    
    - `s3://socal-videomae-dataset/SOCAL/JPEGImages/`
        
    - `s3://socal-videomae-dataset/SOCAL/socal_trial_outcomes.csv`
        
3. **設定 AWS 憑證**：為了讓 Colab 能與您的 AWS 帳戶溝通，您需要設定存取金鑰。
    
    - 在 AWS IAM (Identity and Access Management) 服務中，建立一個新使用者，並賦予其 `AmazonSageMakerFullAccess` 和 `AmazonS3FullAccess` 權限。
        
    - 為該使用者產生一組存取金鑰 (Access Key ID 和 Secret Access Key)。
        
    - 在 Colab 中，安裝並設定 AWS CLI：
        
        Python
        
        ```
        !pip install -q awscli
        !aws configure
        ```
        
        接著，依提示輸入您的 Access Key ID, Secret Access Key, 預設區域 (例如 `us-east-1`) 和輸出格式 (`json`)。
        

---

### 第二步：修改後的 Colab 程式碼 (用於啟動 AWS 訓練)

這是您將在 Colab 中執行的**啟動器 (Launcher)** 程式碼。它取代了您原有的 Colab 程式碼，其主要功能不再是直接在本機訓練，而是設定並遠端啟動 AWS SageMaker 訓練任務。

請將以下程式碼儲存為一個新的 Colab Notebook。

Python

```
# 檔案名稱: sagemaker_launcher.ipynb

import sagemaker
from sagemaker.pytorch import PyTorch
import boto3
import os

# --- 1. AWS SageMaker 設定 ---
print("正在設定 AWS SageMaker...")

# 自動獲取 AWS 執行角色 (Execution Role) 和會話 (Session)
# SageMaker Python SDK 需要一個 IAM 角色來授予訓練任務存取 S3 等資源的權限
try:
    sagemaker_session = sagemaker.Session()
    role = sagemaker.get_execution_role()
    bucket = sagemaker_session.default_bucket() # 或者手動指定 bucket = 'your-s3-bucket-name'
    print(f"SageMaker Role: {role}")
    print(f"Default S3 Bucket: {bucket}")
except ValueError:
    # 如果不在 SageMaker 環境中執行 (例如在本地或 Colab)，需要手動指定 role
    # 請將 'arn:aws:iam::YOUR_AWS_ACCOUNT_ID:role/YourSageMakerRole' 替換成您自己的 SageMaker 角色 ARN
    # 您可以在 AWS IAM 控制台中找到您的角色 ARN
    iam_client = boto3.client('iam')
    role_name = 'AmazonSageMaker-ExecutionRole-YYYYMMDDTHHMMSS' # 根據您建立的角色名稱修改
    try:
        role = iam_client.get_role(RoleName=role_name)['Role']['Arn']
    except iam_client.exceptions.NoSuchEntityException:
        print("無法自動找到 SageMaker 角色，請確認角色名稱或手動提供 ARN。")
        # role = "arn:aws:iam::YOUR_AWS_ACCOUNT_ID:role/YourSageMakerRole" # 手動填寫
        raise
    sagemaker_session = sagemaker.Session()
    bucket = 'socal-videomae-dataset' # **請務必替換成您自己的 S3 儲存桶名稱**
    print(f"手動設定 SageMaker Role: {role}")
    print(f"手動設定 S3 Bucket: {bucket}")


# --- 2. 資料路徑設定 ---
# S3 上的資料路徑
s3_prefix = 'SOCAL' # S3 儲存桶內的路徑前綴
s3_data_path = f's3://{bucket}/{s3_prefix}'
s3_output_path = f's3://{bucket}/output' # 模型輸出和 checkpoints 的儲存位置

print(f"資料來源 (S3): {s3_data_path}")
print(f"模型輸出 (S3): {s3_output_path}")


# --- 3. 超參數與設定 ---
# 這些是您原有的 config，現在作為超參數傳遞給遠端的訓練腳本
hyperparameters = {
    "model_name": "MCG-NJU/videomae-base",
    "num_frames": 16,
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "optimizer_choice": "AdamW",
    "learning_rate": 5e-5,
    "batch_size": 4,         # 可以根據您選擇的 AWS 執行個體規格調整
    "num_epochs": 30,
    "weight_decay": 0.01,
    "dropout_rate": 0.2,
    "use_early_stopping": True,
    "early_stopping_patience": 3,
    "use_lr_scheduler": True,
    "lr_scheduler_patience": 1,
    "force_retrain": False
}


# --- 4. 建立 SageMaker Estimator ---
# Estimator 是 SageMaker 的核心，它封裝了所有啟動訓練任務所需的資訊
print("\n正在建立 SageMaker PyTorch Estimator...")
estimator = PyTorch(
    entry_point='train_script.py',           # **指定您的訓練腳本檔案名稱**
    source_dir='./',                         # 腳本所在的本地目錄
    role=role,                               # IAM 角色
    instance_count=1,                        # 使用 1 台機器進行訓練
    instance_type='ml.g4dn.xlarge',          # **選擇 AWS 執行個體類型，'ml.g4dn.xlarge' 是具備 GPU 的選項**
    framework_version='1.13',                # PyTorch 版本
    py_version='py39',                       # Python 版本
    sagemaker_session=sagemaker_session,
    hyperparameters=hyperparameters,         # 傳遞超參數
    output_path=s3_output_path,              # 指定模型輸出路徑
    # 如果您的訓練資料量很大，可以使用 SageMaker 的 FastFile mode
    # input_mode='FastFile'
)


# --- 5. 啟動訓練任務 ---
# .fit() 方法會將您的腳本和資料上傳，並在雲端啟動訓練
print("\n開始啟動遠端 AWS 訓練任務... 這可能需要幾分鐘來準備執行個體。")
print("您可以前往 AWS SageMaker 控制台 > 訓練 > 訓練任務 查看進度。")
estimator.fit({'training': s3_data_path})

print("\n--- 訓練任務已啟動 ---")
# 您可以呼叫 .logs() 來即時查看遠端訓練的 log
estimator.logs()
```

---

### 第三步：建立獨立的訓練腳本

這是您**核心的訓練邏輯**。我們將您原有的資料處理和模型訓練程式碼從 Colab Notebook 中抽取出來，放入一個名為 `train_script.py` 的 Python 檔案中。這個檔案會和上面的 `sagemaker_launcher.ipynb` 放在同一個目錄下。

**主要變動**：

- **路徑修改**：SageMaker 在啟動訓練時，會自動將 S3 上的資料下載到訓練執行個體的一個特定路徑下。預設路徑是 `/opt/ml/input/data/`。我們需要修改程式碼，從這個路徑讀取資料。
    
- **參數傳遞**：所有可變的設定（如學習率、批次大小）都改由 `argparse` 從 `hyperparameters` 字典接收，增加了靈活性。
    
- **移除 Google Drive 相關程式碼**：所有 `drive.mount` 和 Google Drive 的路徑都已被移除。
    
- **儲存點**：模型的 checkpoints 會被儲存到 `/opt/ml/model/`，SageMaker 會在訓練結束後自動將此目錄下的所有檔案壓縮並上傳到您指定的 S3 輸出路徑。
    

Python

```python
# 檔案名稱: train_script.py

import os
import argparse
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from transformers import AutoImageProcessor, AutoModelForVideoClassification
from PIL import Image
from tqdm.auto import tqdm
import warnings
import collections
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
import sys

# --- 基本設定 ---
# 設定日誌，方便在 SageMaker Logs 中查看
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# --- 資料集類別 (與您原始碼幾乎相同) ---
class SOCALImageSequenceDataset(Dataset):
    def __init__(self, dataframe, image_dir, image_processor, num_frames=16, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.image_processor = image_processor
        self.num_frames = num_frames
        self.transform = transform
        self.trial_frames_map = self._build_trial_frames_map()

        initial_len = len(self.dataframe)
        self.dataframe = self.dataframe[self.dataframe['trial_id'].astype(str).isin(self.trial_frames_map.keys())].reset_index(drop=True)
        if len(self.dataframe) < initial_len:
            logger.info(f"Warning: Filtered out {initial_len - len(self.dataframe)} trials with no corresponding image frames found.")
        logger.info(f"Dataset initialized with {len(self.dataframe)} valid samples.")

    def _build_trial_frames_map(self):
        trial_map = {}
        # 注意這裡的 image_dir 已經是 AWS 執行個體上的本地路徑
        all_image_paths = glob.glob(os.path.join(self.image_dir, "*.jpeg"))
        for img_path in tqdm(all_image_paths, desc="Scanning image files..."):
            filename = os.path.basename(img_path)
            trial_id = filename.split('_')[0]
            if trial_id not in trial_map:
                trial_map[trial_id] = []
            trial_map[trial_id].append(img_path)
        for trial_id in trial_map:
            trial_map[trial_id].sort()
        return trial_map

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        trial_id = str(row['trial_id'])
        label = row['success']
        frame_paths = self.trial_frames_map.get(trial_id, [])

        final_height = self.image_processor.crop_size['height']
        final_width = self.image_processor.crop_size['width']
        image_size_tuple = (final_width, final_height)

        if not frame_paths:
            return {
                'pixel_values': torch.zeros(3, self.num_frames, final_height, final_width),
                'labels': torch.tensor(-1)
            }

        total_frames_in_trial = len(frame_paths)
        if total_frames_in_trial < self.num_frames:
            indices = np.random.choice(total_frames_in_trial, self.num_frames, replace=True)
        else:
            indices = np.linspace(0, total_frames_in_trial - 1, self.num_frames, dtype=int)
        indices.sort()

        selected_frames = []
        for i in indices:
            try:
                img = Image.open(frame_paths[i]).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                selected_frames.append(img)
            except Exception:
                selected_frames.append(Image.new('RGB', image_size_tuple, (0, 0, 0)))

        inputs = self.image_processor(selected_frames, return_tensors="pt")
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# --- 主訓練函式 ---
def train(args):
    # --- 1. 設定裝置和路徑 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Selected device: {device}")
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # SageMaker 會將資料下載到 /opt/ml/input/data/<channel_name>
    # 'training' 是我們在 .fit() 中定義的 channel name
    data_dir = args.data_dir
    model_dir = args.model_dir # SageMaker 提供，用於儲存模型

    SOCAL_DATA_FOLDER = os.path.join(data_dir, "SOCAL")
    JPEG_IMAGES_SUBFOLDER = os.path.join(SOCAL_DATA_FOLDER, "JPEGImages")
    OUTCOMES_CSV_PATH = os.path.join(SOCAL_DATA_FOLDER, "socal_trial_outcomes.csv")
    CHECKPOINT_PATH = os.path.join(model_dir, "checkpoint.pth") # 將 checkpoint 存到 SageMaker 的模型目錄

    logger.info(f"Data directory: {SOCAL_DATA_FOLDER}")
    logger.info(f"Model directory: {model_dir}")

    # --- 2. 資料準備 ---
    logger.info(f"\n--- Preparing Dataset for {args.model_name} ---")
    df = pd.read_csv(OUTCOMES_CSV_PATH)
    logger.info("\nAnalyzing class distribution...")
    logger.info(df['success'].value_counts())

    image_processor = AutoImageProcessor.from_pretrained(args.model_name)

    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ])

    full_dataset = SOCALImageSequenceDataset(df, JPEG_IMAGES_SUBFOLDER, image_processor, args.num_frames, transform=train_transforms)

    total_size = len(full_dataset)
    train_size = int(args.train_ratio * total_size)
    val_size = int(args.val_ratio * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, _ = random_split(full_dataset, [train_size, val_size, test_size])

    logger.info(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")

    # --- 3. 模型與優化器設定 ---
    model = AutoModelForVideoClassification.from_pretrained(args.model_name, num_labels=2, ignore_mismatched_sizes=True)
    model.to(device)

    if args.dropout_rate is not None:
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = args.dropout_rate

    if args.optimizer_choice.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)

    scheduler = None
    if args.use_lr_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=args.lr_scheduler_patience, verbose=True)

    # --- 4. 載入 Checkpoint (如果存在) ---
    start_epoch = 0
    best_val_loss = float('inf')
    if os.path.exists(CHECKPOINT_PATH) and not args.force_retrain:
        logger.info(f"Checkpoint found! Loading '{CHECKPOINT_PATH}'...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        logger.info(f"Resuming from Epoch {start_epoch + 1}")

    # --- 5. DataLoaders 和 Loss Function ---
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    train_labels = [train_dataset.dataset.dataframe.iloc[i]['success'] for i in train_dataset.indices]
    class_counts = collections.Counter(train_labels)
    total_samples = len(train_labels)
    class_weights = torch.tensor([total_samples / (2 * class_counts.get(0, 1e-5)), total_samples / (2 * class_counts.get(1, 1e-5))], dtype=torch.float32).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    logger.info(f"Calculated class weights: {class_weights.cpu().numpy()}")

    # --- 6. 訓練與驗證迴圈 ---
    logger.info("\nStarting training...")
    early_stopping_counter = 0

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{args.num_epochs}"):
            optimizer.zero_grad()
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                outputs = model(pixel_values=pixel_values)
                loss = loss_fn(outputs.logits, labels)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_dataloader)

        model.eval()
        all_preds, all_labels = [], []
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Validating Epoch {epoch+1}/{args.num_epochs}"):
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                    outputs = model(pixel_values=pixel_values)
                    loss = loss_fn(outputs.logits, labels)
                total_val_loss += loss.item()
                all_preds.extend(torch.argmax(outputs.logits, dim=-1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_dataloader)
        val_accuracy = accuracy_score(all_labels, all_preds)

        logger.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

        if scheduler:
            scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            logger.info(f"Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving model...")
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, CHECKPOINT_PATH)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            logger.info(f"Validation loss did not improve. Early stopping counter: {early_stopping_counter}/{args.early_stopping_patience}")
            if args.use_early_stopping and early_stopping_counter >= args.early_stopping_patience:
                logger.info("Early stopping triggered. Halting training.")
                break

    # --- 7. 儲存最終模型 ---
    # SageMaker 會自動將 model_dir 的內容打包上傳到 S3
    # 我們可以儲存最終模型狀態以供後續推論使用
    final_model_path = os.path.join(model_dir, "model.pth")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # --- SageMaker 環境變數 ---
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loading workers')

    # --- 從 sagemaker_launcher 傳遞的超參數 ---
    parser.add_argument('--model_name', type=str, default="MCG-NJU/videomae-base")
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--optimizer_choice', type=str, default="AdamW")
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--use_early_stopping', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--early_stopping_patience', type=int, default=3)
    parser.add_argument('--use_lr_scheduler', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--lr_scheduler_patience', type=int, default=1)
    parser.add_argument('--force_retrain', type=lambda x: (str(x).lower() == 'true'), default=False)

    args = parser.parse_args()
    train(args)
```

---

### 中文詳細介紹與執行流程

#### 程式碼結構說明

1. **`sagemaker_launcher.ipynb` (啟動器 Notebook)**:
    
    - **角色**：這是您的“遙控器”。您在 Colab 或本地機器上執行它。
        
    - **功能**：
        
        - **`boto3` 和 `sagemaker`**：引入 AWS SDK，用於和您的 AWS 帳戶互動。
            
        - **SageMaker Session 和 Role**：設定執行訓練任務所需的權限和會話。`role` 是最重要的部分，它像是一把鑰匙，允許 SageMaker 訓練任務存取您在 S3 上的資料。
            
        - **S3 路徑**：定義資料在 S3 上的位置 (`s3_data_path`) 和訓練完成後模型要儲存的位置 (`s3_output_path`)。
            
        - **`hyperparameters` 字典**：集中管理所有可調參數。這使得修改實驗設定非常方便，無需更改核心的 `train_script.py`。
            
        - **`PyTorch` Estimator**：這是 SageMaker SDK 的核心物件。您在這裡定義了訓練任務的所有規格：
            
            - `entry_point`: 指定要執行的 Python 腳本。
                
            - `source_dir`: 該腳本所在的目錄。SageMaker 會將此目錄下的所有檔案打包上傳。
                
            - `instance_type`: **這是關鍵！** 您可以選擇 AWS 提供的各種計算資源，從基本的 CPU 機器到配備多個高性能 GPU 的伺服器（如 `ml.p3.16xlarge`）。`ml.g4dn.xlarge` 是一個性價比不錯的 GPU 選項。
                
            - `framework_version` 和 `py_version`: 確保雲端環境與您的程式碼相容。
                
        - **`estimator.fit({'training': s3_data_path})`**: 這是觸發器。執行此行程式碼後，SageMaker 會在背景完成以下所有工作：
            
            1. 在雲端啟動一台您指定的 `instance_type` 伺服器。
                
            2. 安裝指定的 Python 和 PyTorch 版本。
                
            3. 將您 `source_dir` 中的程式碼 (`train_script.py`) 複製到伺服器上。
                
            4. 將您 S3 上的資料 (`s3_data_path`) 高效地串流或下載到伺服器的 `/opt/ml/input/data/training/` 目錄下。
                
            5. 執行命令 `python train_script.py --model_name "..." --learning_rate ...`，將 `hyperparameters` 字典中的所有鍵值對轉換為命令列參數。
                
            6. 即時捕捉所有 `print` 和 `logging` 輸出，顯示在您的 Colab 中。
                
            7. 訓練結束或中斷後，將伺服器上 `/opt/ml/model/` 目錄的內容（即您的 `checkpoint.pth` 和 `model.pth`）打包成 `model.tar.gz`，並上傳到您指定的 `s3_output_path`。
                
            8. 自動關閉雲端伺服器，**您只需為實際的訓練時間付費**。
                
2. **`train_script.py` (遠端訓練腳本)**:
    
    - **角色**：這是真正在 AWS 雲端伺服器上執行的程式碼。它是一個獨立、乾淨的訓練腳本。
        
    - **功能**：
        
        - **`argparse`**：腳本的開頭使用 `argparse` 來定義和接收所有來自 `sagemaker_launcher.ipynb` 的超參數。這使得腳本本身具有很高的可重用性。
            
        - **SageMaker 環境變數**：`os.environ.get('SM_MODEL_DIR')` 和 `os.environ.get('SM_CHANNEL_TRAINING')` 是 SageMaker 在執行環境中自動設定的標準環境變數。它們分別指向模型儲存目錄 (`/opt/ml/model/`) 和資料輸入目錄 (`/opt/ml/input/data/training/`)。我們的程式碼透過讀取這些變數，就能準確地找到資料和儲存模型，無需硬編碼任何路徑。
            
        - **核心邏輯不變**：您會發現 `SOCALImageSequenceDataset` 類別和訓練迴圈的邏輯幾乎與您原始的程式碼完全相同。我們只是將其封裝在一個 `train()` 函式中，並替換了路徑和參數的獲取方式。這種“最小改動”原則確保了程式碼的穩定性和可讀性。
            
        - **日誌 (Logging)**：使用 Python 的 `logging` 模組取代單純的 `print`，可以提供更結構化的輸出，這在除錯複雜的雲端任務時非常有用。
            

#### 執行步驟總結

1. **本地準備**：
    
    - 在您的本地電腦或 Colab 環境中，建立一個資料夾。
        
    - 將 `sagemaker_launcher.ipynb` 和 `train_script.py` 這兩個檔案放在該資料夾內。
        
    - 確保已安裝 `sagemaker`, `boto3`, `awscli` 等必要的 Python 套件。
        
2. **AWS 設定**：
    
    - 完成 S3 儲存桶的建立和資料上傳。
        
    - 設定好您的 AWS 存取憑證。
        
3. **啟動訓練**：
    
    - 打開並執行 `sagemaker_launcher.ipynb`。
        
    - 程式碼會引導您完成設定，然後呼叫 `estimator.fit()`。
        
    - 一旦執行，您就可以在 Colab 的輸出單元格中看到來自遠端 AWS 伺服器的即時訓練日誌，就像在本地訓練一樣。同時，您也可以登入 AWS SageMaker 控制台，以圖形化介面監控訓練任務的狀態、資源使用率等。
        
4. **獲取結果**：
    
    - 訓練完成後，您的模型 checkpoint 和最終模型會被儲存在 S3 的 `output` 資料夾中。您可以隨時從 S3 下載這些檔案用於後續的評估或部署。
        

這種將“啟動器”和“核心腳本”分離的模式是使用 AWS SageMaker 的標準最佳實踐，它兼顧了本地開發的便利性和雲端訓練的強大能力與彈性。