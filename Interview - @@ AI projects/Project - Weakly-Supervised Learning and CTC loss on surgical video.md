
Dataset:  _Cholec80_ dataset
https://github.com/CAMMA-public/TF-Cholec80

Maybe useful (Surgical-Phase-Recognition github)
https://github.com/CAMMA-public/Surgical-Phase-Recognition

Surgical phases:

- Preparation
- CalotTriangleDissection
- ClippingCutting
- GallbladderDissection
- GallbladderRetraction
- CleaningCoagulation
- GallbladderPackaging

Surgical instruments **(instrument presence binary labels follow the order from this list)**:

- Grasper
- Bipolar
- Hook
- Scissors
- Clipper
- Irrigator
- SpecimenBag


好的，這是一個根據您的要求，使用 PyTorch 和 Hugging Face `VideoMAEModel` 重寫的 Colab Notebook 完整版本。

這個版本包含了以下特點：

- **PyTorch 框架**：所有代碼均使用 PyTorch 實現。
    
- **VideoMAE 模型**：採用了原生的影片模型 `VideoMAE` 作為特徵提取器，取代了之前逐幀處理的 ViT。
    
- **Google Drive 整合**：包含了載入和檢查 Google Drive 路徑的標準實踐。
    
- **數據集模擬**：由於直接下載大數據集不可行，我們將使用 `av` 函式庫在本地創建小型的 `.mp4` 影片文件來模擬真實數據，使代碼可完整運行。
    
- **語言要求**：**代碼內的註釋為英文**，**代碼外的說明文字為中文**。
    
- **代碼優化**：使用 PyTorch 的 `Dataset` 和 `DataLoader`，並自定義 `collate_fn` 來高效處理變長的影片和標籤序列。
    

請在 Google Colab 中打開一個新的 Notebook，然後按順序複製並執行以下程式碼儲存格。

---

### **Colab Notebook：(PyTorch 版) 以 VideoMAE + CTC Loss 進行弱監督手術階段辨識**

#### **第 0 步：環境設定與 Google Drive 整合**

這個儲存格將安裝所有必要的函式庫，設定 PyTorch 設備，並掛載您的 Google Drive。
```python
# ==============================================================================
# Step 0: Setup Environment and Mount Google Drive
# ==============================================================================

# Install required packages quietly
!pip install -q transformers torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q av tqdm

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import av
import numpy as np
from tqdm.notebook import tqdm
import random
import torch.nn.functional as F
from transformers import AutoImageProcessor, VideoMAEModel
from google.colab import drive

# --- Google Drive and Path Configuration ---
try:
    drive.mount('/content/drive')
    print("Google Drive mounted successfully.")
    # Define data folder within Google Drive
    DRIVE_BASE_PATH = "/content/drive/My Drive/Colab_Notebooks"
    # The main folder for our surgical video demonstration data
    SURGICAL_DATA_FOLDER = os.path.join(DRIVE_BASE_PATH, "surgical_videos_wsl_demo")
    os.makedirs(SURGICAL_DATA_FOLDER, exist_ok=True)
    print(f"Data will be stored in: {SURGICAL_DATA_FOLDER}")
except Exception as e:
    print(f"Error mounting Google Drive: {e}")
    # If drive mount fails, use local Colab storage as a fallback
    print("Using local Colab storage as a fallback.")
    SURGICAL_DATA_FOLDER = "/content/surgical_videos_wsl_demo"


# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Selected device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
```

#### **第 1 部分：數據集準備 (Dataset Preparation)**

我們將在指定的 `SURGICAL_DATA_FOLDER` 中創建模擬數據。這包括使用 `av` 函式庫生成真實的 `.mp4` 影片文件，以及對應的弱標籤文字檔。

**1.1 模擬數據集創建**

```python

# ==============================================================================
# Step 1.1: Simulate and Create Dataset
# ==============================================================================

# --- Global Constants ---
NUM_PHASES = 8  # e.g., 7 surgical phases + 1 background/transition phase
CTC_BLANK_IDX = NUM_PHASES # The blank index for CTC is the last class
CTC_NUM_CLASSES = NUM_PHASES + 1 # Total classes including the blank token

def create_dummy_video_dataset(base_path, num_videos=5, duration_secs=5, fps=10):
    """
    Creates a dummy dataset with actual .mp4 video files and weak labels.
    - base_path/
      - videos/
        - video_01.mp4
        - ...
      - labels/
        - video_01.txt
        - ...
    """
    videos_path = os.path.join(base_path, "videos")
    labels_path = os.path.join(base_path, "labels")
    os.makedirs(videos_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)

    print(f"Checking for existing data in '{base_path}'...")
    if len(os.listdir(videos_path)) >= num_videos:
        print("Dataset already exists. Skipping creation.")
        return

    print(f"Creating dummy dataset with {num_videos} videos...")
    for i in range(1, num_videos + 1):
        video_name = f"video_{i:02d}"
        video_file_path = os.path.join(videos_path, f"{video_name}.mp4")

        # 1. Create a dummy .mp4 video file using pyav
        container = av.open(video_file_path, mode='w')
        stream = container.add_stream('libx264', rate=fps)
        stream.width = 224
        stream.height = 224
        stream.pix_fmt = 'yuv420p'

        for frame_i in range(duration_secs * fps):
            # Create a black frame with a moving white square
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            x, y = (frame_i * 4) % 224, (frame_i * 3) % 224
            img[y:y+20, x:x+20, :] = 255
            frame = av.VideoFrame.from_ndarray(img, format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        # Flush stream
        for packet in stream.encode():
            container.mux(packet)
        container.close()

        # 2. Create corresponding weak label file (sequence of phases)
        num_actual_phases = random.randint(3, NUM_PHASES - 1)
        phase_sequence = random.sample(range(NUM_PHASES), num_actual_phases)
        label_file_path = os.path.join(labels_path, f"{video_name}.txt")
        with open(label_file_path, "w") as f:
            f.write(" ".join(map(str, phase_sequence)))

    print("Dummy dataset creation complete.")

# Execute creation
create_dummy_video_dataset(SURGICAL_DATA_FOLDER)
```

**1.2 PyTorch Dataset 與 DataLoader**

這是數據處理的核心。我們定義一個自定義的 `Dataset` 來讀取影片和標籤，並創建一個 `collate_fn` 來處理變長序列的批次化，這對於 CTC Loss 至關重要。

```python
# ==============================================================================
# Step 1.2: Create PyTorch Dataset and DataLoader
# ==============================================================================

class SurgicalVideoDataset(Dataset):
    """Custom PyTorch Dataset for loading surgical videos and weak labels."""
    def __init__(self, root_dir, image_processor):
        """
        Args:
            root_dir (string): Directory with 'videos' and 'labels' subdirectories.
            image_processor: Hugging Face image processor for VideoMAE.
        """
        self.videos_dir = os.path.join(root_dir, "videos")
        self.labels_dir = os.path.join(root_dir, "labels")
        self.image_processor = image_processor
        
        self.video_files = sorted([f for f in os.listdir(self.videos_dir) if f.endswith('.mp4')])

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        # Get video path
        video_filename = self.video_files[idx]
        video_path = os.path.join(self.videos_dir, video_filename)
        
        # 1. Read video frames using pyav
        container = av.open(video_path)
        frames = []
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format='rgb24'))
        container.close()

        # 2. Preprocess frames using the VideoMAE processor
        # The processor handles resizing, normalization, and channel ordering.
        processed_video = self.image_processor(frames, return_tensors="pt")
        # Squeeze to remove the batch dimension added by the processor
        video_tensor = processed_video['pixel_values'].squeeze(0)

        # 3. Read weak label transcript
        label_filename = os.path.splitext(video_filename)[0] + ".txt"
        label_path = os.path.join(self.labels_dir, label_filename)
        with open(label_path, 'r') as f:
            label_str = f.read().strip()
            labels = [int(p) for p in label_str.split()]
        
        label_tensor = torch.LongTensor(labels)
        
        return video_tensor, label_tensor

def collate_fn(batch):
    """
    Custom collate function to handle variable length sequences for CTC Loss.
    It pads videos and labels and returns their original lengths.
    """
    # Unzip the batch
    videos, labels = zip(*batch)
    
    # Get original lengths before padding
    video_lengths = torch.LongTensor([v.shape[0] for v in videos])
    label_lengths = torch.LongTensor([len(l) for l in labels])
    
    # Pad sequences
    # `batch_first=True` means the output shape will be (batch, time, ...)
    padded_videos = nn.utils.rnn.pad_sequence(videos, batch_first=True)
    # Use a padding value that is not a valid label, e.g., -100 for labels
    padded_labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    
    return {
        'videos': padded_videos,
        'labels': padded_labels,
        'video_lengths': video_lengths,
        'label_lengths': label_lengths
    }


# --- Instantiate Dataset and DataLoader ---
# Load the VideoMAE image processor from Hugging Face
image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
train_dataset = SurgicalVideoDataset(root_dir=SURGICAL_DATA_FOLDER, image_processor=image_processor)
train_loader = DataLoader(
    train_dataset,
    batch_size=2, # Keep batch size small for Colab memory
    shuffle=True,
    collate_fn=collate_fn
)

# --- Verify one batch ---
print("Verifying one batch from the DataLoader...")
try:
    batch_sample = next(iter(train_loader))
    print(f"Videos batch shape: {batch_sample['videos'].shape}") # (B, T, C, H, W)
    print(f"Labels batch shape: {batch_sample['labels'].shape}") # (B, L)
    print(f"Video lengths: {batch_sample['video_lengths']}")
    print(f"Label lengths: {batch_sample['label_lengths']}")
except Exception as e:
    print(f"Error while fetching a batch: {e}")
```

#### **第 2 部分：模型訓練 (VideoMAE + CTC Loss)**

現在我們定義模型架構，並編寫 PyTorch 風格的訓練循環。

**2.1 模型架構定義**

模型將使用預訓練的 `VideoMAEModel` 提取時空特徵，然後傳遞給一個雙向 LSTM 來捕捉更長的時間依賴性，最後通過一個線性層輸出 CTC 所需的機率。

```python
# ==============================================================================
# Step 2.1: Define the Model Architecture
# ==============================================================================

class VideoMAE_CTC_Model(nn.Module):
    """
    A model combining VideoMAE for feature extraction and an LSTM for temporal
    modeling, trained with CTC loss for weakly-supervised learning.
    """
    def __init__(self, num_classes, hidden_lstm_size=256):
        super().__init__()
        
        # 1. Load pretrained VideoMAE model
        self.videomae = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
        # Freeze the VideoMAE model's parameters
        for param in self.videomae.parameters():
            param.requires_grad = False
            
        # 2. Temporal model (Bi-LSTM)
        # The input size for LSTM is the feature dimension from VideoMAE's output
        self.lstm = nn.LSTM(
            input_size=self.videomae.config.hidden_size,
            hidden_size=hidden_lstm_size,
            num_layers=2,
            batch_first=True, # Important: input shape is (batch, seq, feature)
            bidirectional=True,
            dropout=0.5
        )
        
        # 3. Output classifier for CTC
        # LSTM output is 2 * hidden_size because it's bidirectional
        self.fc = nn.Linear(hidden_lstm_size * 2, num_classes)

    def forward(self, videos):
        # videos shape: (batch, time, channels, height, width)
        
        # Get features from VideoMAE
        # To avoid GPU memory issues with long videos, process in chunks if needed.
        # Here we assume the whole video fits in memory.
        outputs = self.videomae(videos)
        frame_features = outputs.last_hidden_state # Shape: (batch, time, feature_dim)
        
        # Pass features through LSTM
        lstm_output, _ = self.lstm(frame_features) # Shape: (batch, time, 2*hidden_size)
        
        # Pass through final linear layer
        logits = self.fc(lstm_output) # Shape: (batch, time, num_classes)
        
        # Prepare for CTC Loss: needs (Time, Batch, Classes) and log probabilities
        log_probs = F.log_softmax(logits, dim=2)
        log_probs = log_probs.permute(1, 0, 2) # (T, B, C)
        
        return log_probs

# Instantiate the model and move to the selected device
model = VideoMAE_CTC_Model(num_classes=CTC_NUM_CLASSES).to(device)

# Print a summary of the trainable parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")
```

**2.2 訓練循環**

這是一個標準的 PyTorch 訓練循環，包含了損失計算、反向傳播和優化器步驟。

```python
# ==============================================================================
# Step 2.2: Define Training Loop
# ==============================================================================

# --- Hyperparameters ---
LEARNING_RATE = 1e-4
EPOCHS = 20 # Increase for real training, keep low for demo

# --- Loss Function and Optimizer ---
# `blank` index must match the one used in the model definition
ctc_loss = nn.CTCLoss(blank=CTC_BLANK_IDX, reduction='mean', zero_infinity=True)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)


print("\n--- Starting Training ---")
model.train() # Set model to training mode

for epoch in range(EPOCHS):
    epoch_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for batch in progress_bar:
        # Move all tensors in the batch to the device
        videos = batch['videos'].to(device)
        labels = batch['labels'].to(device)
        video_lengths = batch['video_lengths'].to(device)
        label_lengths = batch['label_lengths'].to(device)
        
        # 1. Zero the gradients
        optimizer.zero_grad()
        
        # 2. Forward pass
        log_probs = model(videos)
        
        # 3. Calculate CTC loss
        # The model output is (Time, Batch, Classes)
        # The labels should be (Batch, Max_Label_Length)
        # The lengths should be 1D tensors
        loss = ctc_loss(log_probs, labels, video_lengths, label_lengths)
        
        # 4. Backward pass
        loss.backward()
        
        # 5. Optimizer step
        optimizer.step()
        
        # Update progress bar
        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}")

print("--- Training Finished ---")
```

**2.3 推論與解碼**

訓練完成後，我們使用模型進行預測。由於 PyTorch 沒有內建的 `ctc_decode`，我們將實現一個簡單的貪婪解碼器。

```python
# ==============================================================================
# Step 2.3: Inference and Decoding
# ==============================================================================

def greedy_ctc_decode(log_probs):
    """
    Decodes the output of a CTC model using a greedy approach.
    Args:
        log_probs (Tensor): The log-probabilities from the model, shape (Time, Batch, Classes).
    Returns:
        list: A list of decoded sequences for each item in the batch.
    """
    decoded_sequences = []
    # Get the best path (most likely class at each time step)
    best_path = torch.argmax(log_probs, dim=2) # Shape: (Time, Batch)
    
    # Iterate over each sample in the batch
    for i in range(best_path.shape[1]):
        path = best_path[:, i]
        decoded_path = []
        
        # Remove consecutive duplicates and blank tokens
        for t, token_idx in enumerate(path):
            if token_idx.item() != CTC_BLANK_IDX:
                if t == 0 or token_idx != path[t-1]:
                    decoded_path.append(token_idx.item())
        decoded_sequences.append(decoded_path)
        
    return decoded_sequences

# --- Run Inference on a Sample Batch ---
print("\n--- Running Inference and Decoding ---")
model.eval() # Set model to evaluation mode
with torch.no_grad():
    # Get one batch from the dataloader
    batch_sample = next(iter(train_loader))
    videos = batch_sample['videos'].to(device)
    labels = batch_sample['labels'] # Keep labels on CPU for comparison

    # Get model predictions
    log_probs = model(videos)
    
    # Decode the predictions
    decoded_preds = greedy_ctc_decode(log_probs)
    
    # --- Print and Compare ---
    for i in range(videos.shape[0]):
        # Filter out padding from true labels for clean printing
        true_label_len = batch_sample['label_lengths'][i]
        true_label = labels[i][:true_label_len].tolist()
        
        print(f"\nSample {i+1}:")
        print(f"  True Weak Label: {true_label}")
        print(f"  Predicted Label: {decoded_preds[i]}")
```


```python

```








