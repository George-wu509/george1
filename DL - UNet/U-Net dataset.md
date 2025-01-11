
### 1. **U-Net 的訓練資料準備**

**How to Prepare Training Dataset for U-Net**

在訓練 **U-Net** 之前，首先需要準備好適合的訓練數據集。U-Net 的設計目標是進行像素級別的分割，因此每張影像通常都需要有對應的標籤（mask），用於指示每個像素所屬的類別。對於 **U-Net**，最常見的應用是 **語義分割（Semantic Segmentation）**，而不是 **實例分割（Instance Segmentation）**。

#### **數據集格式：**

如果您的數據集是以 **COCO** 格式存儲的，那麼通常需要以下幾個檔案：
ref: [COCO数据集的标注格式](https://zhuanlan.zhihu.com/p/29393415)

- **images**: 存放訓練用的原始圖片，通常為 `.jpg` 或 `.png` 格式。
- **annotations**: 存放標註文件，通常為 `.json` 格式，包含每張圖片的標註信息。

#### **COCO 格式所需檔案：**

- **images**：一個文件夾，其中包含所有要訓練的圖像，例如 30 張影像。
- **annotations/instances_train.json**：這是 COCO 格式的標註文件，裡面包含了每張圖像的標註，包括分類、邊界框（bounding box）和分割掩碼（segmentation mask）。對於 U-Net，重點在於 **segmentation mask**。

COCO 標註文件中的基本結構如下：
{
  "images": [
    {
      "id": 1,
      "width": 800,
      "height": 600,
      "file_name": "image1.jpg"
    },
    {
      "id": 2,
      "width": 800,
      "height": 600,
      "file_name": "image2.jpg"
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "segmentation": [[x1, y1, x2, y2, ...]],  // 這是用於分割的多邊形點
      "area": 1500,
      "bbox": [100, 50, 200, 150],  // 可選，這是bounding box
      "iscrowd": 0
    },
    ...
  ],
  "categories": [
    {
      "id": 1,
      "name": "category1"
    },
    ...
  ]
}
#### **標籤的準備（Labels / Masks）：**

U-Net 的輸入圖像需要有對應的像素標籤（mask）。這些標籤圖像是灰度圖，其中每個像素的值對應於該像素的類別。例如，對於語義分割，如果有兩個類別（背景為0，目標為1），那麼對應的mask可能會是像素值為0或1的影像。

### 2. **PyTorch Example Code for U-Net Training**

#### **U-Net 訓練程式碼示例**

這裡是一個簡單的 PyTorch U-Net 訓練例子，假設我們已經有訓練用的影像和對應的 mask：

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import os

U-Net 模型定義
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        Define U-Net architecture here (omitted for brevity)

    def forward(self, x):
        Forward pass of U-Net (omitted for brevity)
        return x

自定義的數據集類別
class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask

訓練過程
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        前向傳播
        outputs = model(images)
        loss = criterion(outputs, masks)

        反向傳播與優化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

設置超參數
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

加載數據集
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = CustomDataset("path_to_images", "path_to_masks", transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

訓練模型
num_epochs = 10
for epoch in range(num_epochs):
    train(model, train_loader, criterion, optimizer, device)
    print(f"Epoch [{epoch+1}/{num_epochs}] complete")

#### **Inference 範例程式碼**

這裡是 U-Net 模型的推論（inference）部分，假設我們有一張新圖像，想要做分割：
import matplotlib.pyplot as plt

def infer(model, image_path, transform, device):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        output = torch.sigmoid(output)
        output = output.squeeze().cpu().numpy()

    return output

推論範例
model = UNet().to(device)
加載訓練好的模型權重
model.load_state_dict(torch.load("unet_model.pth"))

進行推論
output = infer(model, "path_to_test_image.jpg", transform, device)

顯示結果
plt.imshow(output, cmap="gray")
plt.show()


### **Segmentation？**

**U-Net** 的結果主要是 **語義分割（Semantic Segmentation）**，即為每個像素分配一個類別，所有屬於同一類別的物體都會被標註為相同的標籤。因此，U-Net 更適合於 **語義分割** 任務，而非 **實例分割（Instance Segmentation）**。

- **語義分割（Semantic Segmentation）**：區分不同的類別，但不區分同一類別內的不同實例。例如，兩隻貓都會被標註為 "貓"，而不會單獨區分每一隻貓。
    
- **實例分割（Instance Segmentation）**：不僅區分類別，還區分同一類別的不同實例。比如，兩隻貓會分別標註為 "貓1" 和 "貓2"。常用於實例分割的算法有 **Mask R-CNN**。
    

總結來說，U-Net 的結果是 **語義分割**，即針對每個像素進行分類，而不區分同一類別的不同實例。如果需要進行實例分割，通常需要使用 **Mask R-CNN** 或類似的實例分割模型。



這段 PyTorch 代碼主要實現了自定義數據集類別、模型的訓練過程和數據加載的過程。下面我們將一步步詳細解釋其中的每個部分，特別是與數據集相關的部分。

### 1. **自定義數據集類別 (CustomDataset)**

這個部分定義了一個自定義的數據集類別，繼承自 `torch.utils.data.Dataset`，用於處理圖像和對應的標註掩碼（mask）。它允許將自定義的數據加載到 PyTorch 模型中進行訓練。

#### **1.1 初始化函數 (`__init__`)**
def __init__(self, image_dir, mask_dir, transform=None):
    self.image_dir = image_dir
    self.mask_dir = mask_dir
    self.transform = transform
    self.images = os.listdir(image_dir)
- **image_dir**：這是包含訓練圖像的目錄路徑。
- **mask_dir**：這是包含對應標註掩碼的目錄路徑。
- **transform**：用於數據增強或預處理的變換。這裡可以傳入一個 `torchvision.transforms` 組合，用來轉換圖像和掩碼，例如將它們轉換為 PyTorch 張量 (`Tensor`)。
- **self.images**：通過 `os.listdir(image_dir)` 獲取該目錄下所有文件的名稱，這是一個包含所有圖像文件名的列表。

這個函數負責初始化數據集類，存儲圖像目錄、掩碼目錄，以及圖像列表（`self.images`）。

#### **1.2 長度函數 (`__len__`)**
def __len__(self):
    return len(self.images)
- 該方法返回數據集中的圖像數量，這是必須實現的方法，因為 PyTorch 在加載數據時會調用它來確定數據集的大小。

#### **1.3 獲取數據函數 (`__getitem__`)**
def __getitem__(self, idx):
    img_path = os.path.join(self.image_dir, self.images[idx])
    mask_path = os.path.join(self.mask_dir, self.images[idx])
    image = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if self.transform:
        image = self.transform(image)
        mask = self.transform(mask)
    return image, mask
    
- **`__getitem__`**：這是 PyTorch 中的核心方法，用於從數據集中加載單個樣本（圖像和對應的掩碼）。當 `DataLoader` 遍歷數據集時，會調用這個方法來獲取每一個樣本。

具體步驟：

- 根據索引 `idx`，從 `self.images` 中獲取當前圖像的文件名，並使用 `os.path.join` 拼接成圖像和掩碼的路徑。
- 使用 `PIL.Image.open()` 加載圖像和對應的掩碼。
	- `image.convert("RGB")`：將圖像轉換為 RGB 模式（3通道）。
	- `mask.convert("L")`：將掩碼轉換為灰度模式（單通道），這裡灰度值代表類別標籤（如0代表背景，1代表前景）。
- 如果傳入了 `transform`，則對圖像和掩碼應用變換。常見的變換包括將圖像轉換為 PyTorch 張量（`transforms.ToTensor()`），或者進行數據增強（如旋轉、翻轉等）。
- 最終返回圖像和掩碼。

### 2. **訓練過程 (train function)**
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        # 前向傳播
        outputs = model(images)
        loss = criterion(outputs, masks)

        # 反向傳播與優化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
這個函數定義了模型的訓練過程，下面逐步解釋：

- **`model.train()`**：這是 PyTorch 中的模型模式設定，用於啟用訓練模式（與推理模式 `model.eval()` 相對應）。訓練模式會啟用 dropout 層、batch normalization 層等。
- **數據加載循環 (`for images, masks in dataloader`)**：
    - `dataloader` 是一個用於批量加載數據的工具，會調用我們自定義的 `CustomDataset` 中的 `__getitem__()` 方法來加載圖像和掩碼。
    - 每次加載一批（例如4張）的圖像和掩碼。
- **數據移動到設備上 (`images.to(device)`)**：
    - `device` 指定了計算設備（如 GPU 或 CPU），將數據移動到指定設備上進行計算。
- **前向傳播 (`outputs = model(images)`)**：
    - 通過模型進行前向傳播，輸出結果。
- **計算損失 (`loss = criterion(outputs, masks)`)**：
    - `criterion` 是損失函數，這裡使用二元交叉熵損失（`nn.BCELoss()`），比較模型的輸出和真實標籤（掩碼），計算損失。
- **反向傳播和優化 (`optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`)**：
    - `optimizer.zero_grad()`：清除過去的梯度。
    - `loss.backward()`：反向傳播計算梯度。
    - `optimizer.step()`：更新模型參數。

### 3. **超參數設置與模型初始化**

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

- **device**：這裡檢查是否有可用的 GPU（`cuda`），如果有則將設備設置為 GPU，否則使用 CPU。
- **model**：初始化 U-Net 模型，並將其移動到指定設備（GPU 或 CPU）上。
- **criterion**：使用二元交叉熵損失函數（`BCELoss`）作為損失函數，這通常用於二元分類任務（背景和前景）。
- **optimizer**：使用 Adam 優化器來更新模型的參數，學習率設置為 `1e-3`。

### 4. **數據集加載與訓練**
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = CustomDataset("path_to_images", "path_to_masks", transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

- **transform**：使用 `transforms.Compose([transforms.ToTensor()])` 將圖像和掩碼轉換為 PyTorch 的張量格式。
- **train_dataset**：創建 `CustomDataset` 實例，並傳入圖像和掩碼的路徑，以及定義的轉換。
- **train_loader**：通過 `DataLoader` 將數據集 `train_dataset` 加載到內存中。`batch_size=4` 意味著每次加載 4 張圖像，`shuffle=True` 表示每次訓練時隨機打亂數據。

### 5. **模型訓練**

num_epochs = 10
for epoch in range(num_epochs):
    train(model, train_loader, criterion, optimizer, device)
    print(f"Epoch [{epoch+1}/{num_epochs}] complete")

- **num_epochs**：定義訓練的總輪數（例如10輪）。
- **訓練循環**：每一個 `epoch` 都會調用一次 `train()` 函數，對模型進行一個完整的訓練周期。每完成一輪訓練後，會輸出當前的 `epoch` 完成情況。

### 6. **總結**

這段 PyTorch 代碼的主要功能是加載自定義的數據集，將其與對應的掩碼進行匹配，並通過 U-Net 模型對其進行訓練。數據集的定義部分重點是 `CustomDataset` 類，通過它來處理原始圖像和對應的掩碼，並將它們轉換成 PyTorch 張量，進行批量加載後送入模型進行訓練。



