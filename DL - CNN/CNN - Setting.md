

![[Pasted image 20250329022534.png]]

| setting                       |                                           |                     |
| ----------------------------- | ----------------------------------------- | ------------------- |
| [[Normalization and dropout]] | Layer normalization,  Batch normalization | **model之內**(relu之前) |
| [[Activation funs]]           | Sigmoid, tanh, ReLU, Softmax              | **model之內**         |
| Dropout                       | Dropout                                   | **model之內**(relu之後) |
|                               |                                           |                     |
| Criterion 在下面                 |                                           | create model之後      |
| [[optimizer]]                 | SGD, SGDM, Adagrad, Adam, AdamW           | create model之後      |
| [[Regularization]]            | L1, L2 regularization                     | optimizer裡面         |
順序: [[CNN order]]

| [[Criterion]]      | Loss function (create model之後)                                                                     | Index                                                                        |
| ------------------ | -------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **[[圖像分類loss]]**   | 分類損失 classification loss                                                                           | Accuracy, Precision,<br>Recall, F1-Score,<br>Confusion Matrix<br>ROC 曲線與 AUC |
|                    |                                                                                                    |                                                                              |
| **[[目標檢測loss]]**   | 邊界框檢測損失 Bounding Box Detection Loss<br>分類損失 classification loss<br>信度損失 Confidence Loss            | mAP, IoU<br>Recall, FPS<br>Confusion Matrix                                  |
|                    |                                                                                                    |                                                                              |
| **[[目標追蹤loss]]**   | 邊界框檢測損失 Bounding Box Detection Loss<br>分類損失 classification loss<br>特徵相似性損失 Feature Similarity Loss | MOTA<br>MOTP<br>Identity Switches, IDSW<br>Frames Per Second, FPS<br>IoU     |
|                    |                                                                                                    |                                                                              |
| **[[實例分割loss]]**   | 邊界框檢測損失 Bounding Box Detection Loss<br>分割掩碼損失 Segmentation Mask Loss<br>信度損失 Confidence Loss       | mAP, IoU<br>Recall, FPS<br>分割掩碼的質量                                           |
|                    |                                                                                                    |                                                                              |
| **[[語義分割loss]]**   | 分類損失 classification loss<br>區域重疊損失 Region Overlap Loss<br>邊界損失 Boundary Loss                       | IoU, Mean IoU<br>Dice, Pixel Accuracy<br>Confusion Matrix                    |
|                    |                                                                                                    |                                                                              |
| **[[圖像質量增強loss]]** | 像素級損失 Pixel-wise Loss<br>感知損失 Perceptual Loss<br>對抗損失 Adversarial Loss<br>紋理損失Texture loss         | PSNR<br>SSIM<br>MSE<br>MAE                                                   |

|                                         |                                          |
| --------------------------------------- | ---------------------------------------- |
| 分類損失 <br>classification loss            | Cross-Entropy Loss<br>Focal Loss         |
| 邊界框檢測損失 <br>Bounding Box Detection Loss | IoU loss<br>Smooth L1 Loss               |
| 信度損失 <br>Confidence Loss                | BCE loss <br>(Binary Cross-Entropy loss) |
| 區域重疊損失 <br>Region Overlap Loss          | IoU Loss<br>Dice Loss                    |
| 邊界損失 <br>Boundary Loss<br>              | 邊界損失函數                                   |
| 特徵相似性損失 <br>Feature Similarity Loss     | Cosine Similarity Loss<br>Triplet Loss   |
| **圖像質量增強**                              |                                          |
| 像素級損失 <br>Pixel-wise Loss               | L1 Loss<br>L2 Loss<br>Charbonnier Loss   |
| 感知損失 <br>Perceptual Loss                | VGG Loss<br>Feature Matching Loss        |
| 對抗損失 <br>Adversarial Loss               | GAN Loss                                 |
| 紋理損失<br>Texture loss                    | Texture Loss                             |

|        | create model之後                           |
| ------ | ---------------------------------------- |
| 模型複雜度  | Parms, FLOPs                             |
| 性能指标   | Latency, Throughput                      |
| 计算资源消耗 | Memory consumption, MACs                 |
| 影片分析   | Cold start, throughput                   |
| 硬體層級   | GPU usage, Peak Memory, Token throughput |



Normalization(放relu之前)跟dropout(放relu之後)都在model裡面:
```python
class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 10)
        self.batchnorm1 = nn.BatchNorm1d(10) # Batch Normalization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5) # Dropout
        self.fc2 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.batchnorm1(x) # Add batch normalization
        x = self.relu(x)
        x = self.dropout(x) # Add dropout
        x = self.sigmoid(self.fc2(x))
        return x
```
**模型建立 (BinaryClassifier):**
- `nn.Linear()`: 定義全連接層，用於學習輸入特徵的線性組合。
- `nn.BatchNorm1d()`: 批量標準化層，用於加速訓練，提高穩定性。在全連結層後使用BatchNorm1d。
- `nn.ReLU()`: ReLU 激活函數，引入非線性，讓模型能夠學習更複雜的關係。
- `nn.Sigmoid()`: Sigmoid 激活函數，將輸出限制在 0 到 1 之間，適用於二元分類。
- `forward()`函數定義了模型前向傳播的過程，以及使用了那些layer。

Regulation:
```python
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) # Adam optimizer, L2 regularization
```
`weight_decay` 參數會將 L2 正則化添加到損失函數中，以懲罰較大的模型權重。
這有助於防止過度擬合，並提高模型的泛化能力


```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import random

# 1. 簡化版的 YOLOv8 模型 (需要YOLOv8的完整權重與模型，這裡只提供簡易範例，用於理解)
class SimplifiedYOLOv8(nn.Module):
    def __init__(self, num_classes):
        super(SimplifiedYOLOv8, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 32 * 32, 256) #這裡的數值需要依照實際的輸出大小來調整
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes * 5) # 5: (x, y, w, h, confidence)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.maxpool1(x)
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x.view(x.size(0), -1, 5) # Reshape to (batch_size, num_anchors, 5)

# 2. 自訂數據集 (您需要替換為您的數據加載邏輯)
class CustomDataset(Dataset):
    def __init__(self, image_paths, annotations, transform=None):
        self.image_paths = image_paths
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        annotation = self.annotations[idx] # annotation needs to be in YOLO format [class, x_center, y_center, width, height]
        if self.transform:
            image = self.transform(image)

        annotation_tensor = torch.tensor(annotation, dtype=torch.float32)
        return image, annotation_tensor

# 3. 損失函數 (簡化版, 實際YOLO loss更複雜)
class YOLOLoss(nn.Module):
    def __init__(self):
        super(YOLOLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.bce = nn.BCEWithLogitsLoss(reduction='sum')
    def forward(self, predictions, targets):
        # 這裡是簡化版本，要修改成為YOLOv8的完整損失函數。
        # predictions shape (batch_size, num_anchors, 5), targets (batch_size, num_anchors, 5)
        conf_loss = self.bce(predictions[:, :, 4], targets[:, :, 4])
        xy_loss = self.mse(predictions[:, :, :2], targets[:, :, :2])
        wh_loss = self.mse(predictions[:, :, 2:4], targets[:, :, 2:4])
        return conf_loss + xy_loss + wh_loss

# 4. 超參數
num_classes = 1 # 物體類別數量
learning_rate = 0.001
batch_size = 16
num_epochs = 10
hard_negative_ratio = 0.5
hard_negative_buffer = []
hard_negative_buffer_size = 1000

# 5. 模型、損失函數、優化器
model = SimplifiedYOLOv8(num_classes)
criterion = YOLOLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5) # Regularization

# 6. 數據集與數據加載器 (替換為您的實際數據)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # Normalization
image_paths = ["image1.jpg", "image2.jpg", "..."]
annotations = [[[0, 0.5, 0.5, 0.1, 0.1]], [[0, 0.2, 0.2, 0.2, 0.2]], ["..."]] # Yolo Annotation example.
dataset = CustomDataset(image_paths, annotations, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 7. 訓練迴圈
for epoch in range(num_epochs):
    model.train()
    for images, targets in dataloader:
        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        # Hard Negative Mining logic here, implement similarly as in faster rcnn example.

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# 8. 模型儲存
torch.save(model.state_dict(), "yolo_model.pth")
```


Batch Normalization（批次正規化）通常放置在全連接層（`fc1`）和 ReLU 激活函數之間，這是由於以下幾個原因：

**1. 數值穩定性：**

- 全連接層的輸出可能具有非常大的數值，尤其是在網路較深或學習率較高時。
- 這些大的數值可能會導致激活函數的輸入分布不穩定，使得訓練變得困難。
- Batch Normalization 通過將全連接層的輸出進行正規化，確保其具有零均值和單位方差，從而提高了數值穩定性。

**2. 梯度消失與梯度爆炸：**

- 深度神經網路在訓練過程中容易遇到梯度消失或梯度爆炸的問題。
- 這些問題會導致梯度在反向傳播過程中變得非常小或非常大，從而影響模型的訓練效果。
- Batch Normalization 有助於減少內部協變異數偏移（Internal Covariate Shift），也就是說每一層輸出的數值分佈，可以更加的穩定。進而解決梯度消失或梯度爆炸的問題。

**3. 加速訓練：**

- 通過改善數值穩定性和減少內部協變異數偏移，Batch Normalization 可以加速模型的收斂速度。
- 這使得可以使用更高的學習率，從而加快訓練過程。

**4. 為何不放在其他位置：**

- **激活函數之後：**
    - 激活函數的輸出通常已經在一個較小的範圍內（例如，ReLU 的輸出為非負數）。
    - 在激活函數之後應用 Batch Normalization 的效果可能不如在激活函數之前。
    - 放置在relu之後，會影響relu的非線形性質。
- **全連接層之前：**
    - 全連接層的輸入是前一層的輸出，可能已經經過了其他處理（例如，卷積或池化）。
    - 在全連接層之前應用 Batch Normalization 可能會干擾這些處理，並且效果不如在全連接層之後。
- **批次正規化通常會防止在非線形激活函數之前。**
    - 原因是批次正規化會對上一層的輸出做正規化，如此一來會限制了網路的表示能力，因此通常放置在非線性激活函數之前。

**總而言之：**

將 Batch Normalization 放置在全連接層和 ReLU 激活函數之間，是一個經過實驗驗證的有效策略，可以提高訓練的穩定性和速度。


您提出關於 Dropout 位置和 L2 正則化設定的問題，這兩個部分在深度學習中都非常重要，以下我將詳細說明：

**1. Dropout 放置在 ReLU 之後的原因：**

- **避免影響 ReLU 的非線性特性：**
    - ReLU 激活函數引入了非線性，使得模型能夠學習複雜的關係。
    - 如果將 Dropout 放在 ReLU 之前，隨機丟棄神經元可能會破壞 ReLU 引入的非線性，降低模型的表示能力。
    - 將 Dropout 放在 ReLU 之後，可以保留 ReLU 的非線性特性，同時減少神經元之間的共同適應性，從而提高模型的泛化能力。
- **針對激活後的數值進行隨機丟棄：**
    - 放置在relu之後，被隨機關閉的，會是經過relu激活過後的數值，如此一來，relu產生的非線性結果不會被影響，影響的是relu非線性輸出後的數值。
- **實務上的經驗：**
    - 許多研究和實務經驗表明，將 Dropout 放在 ReLU 之後通常能獲得更好的效果。

**2. L2 正則化的設定：**

- 您提到 `weight_decay=1e-5` 是 L1 正則化，這是**錯誤的**。
- `weight_decay` 參數在 PyTorch 的優化器（如 `optim.Adam`）中，**實作的是 L2 正則化**。
- **L2 正則化的原理：**
    - L2 正則化通過在損失函數中添加模型權重的平方和，來懲罰較大的權重。
    - 這有助於防止過度擬合，並使模型學習更平滑的函數。
- **PyTorch 中的 L2 正則化：**
    - 在 PyTorch 中，通過在優化器中設定 `weight_decay` 參數，即可應用 L2 正則化。
    - `weight_decay` 的值越大，L2 正則化的強度就越大。
    - 1e-5 是很常使用的一個數值，這是一個超參數，依然需要根據資料來做調試。
- **L1 正則化的實作：**
    - PyTorch 的優化器本身不直接提供 L1 正則化。
    - 如果需要使用 L1 正則化，您需要自訂損失函數，並將 L1 正則化項添加到其中。
    - 或者是額外撰寫程式碼於每一次的參數更新中，針對模型的權重進行L1的懲罰。

**總結：**

- Dropout 放置在 ReLU 之後是為了保留 ReLU 的非線性特性，並提高模型的泛化能力。
- `weight_decay` 參數在 PyTorch 優化器中實作的是 L2 正則化。
- 如果需要使用 L1 正則化，需要自訂實作。