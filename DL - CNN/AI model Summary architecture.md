


|      |     | backbone                   | neck                               | head                                                         |
| ---- | --- | -------------------------- | ---------------------------------- | ------------------------------------------------------------ |
| RCNN | DET | AlexNet or VGG             | region proposals, RoI pooling, FPN | FC layer，輸出類別分數（softmax）和邊界框回歸（bounding box regression）。     |
| SSD  | DET | VGG16                      | 多尺度特徵層                             | 多尺度特徵圖上設置預定義錨框（default boxes）每個錨框預測類別分數和邊界框偏移量               |
| FCOS | DET | ResNet or ResNetx          | FPN                                | anchor-free, 輸出類別分數（softmax）和邊界框回歸（bounding box regression）。 |
|      |     |                            |                                    |                                                              |
| YOLO | DET | CSPDarknet or EfficientNet | FPN                                | grid, bounding box and confidence                            |
|      |     |                            |                                    |                                                              |
| FCN  | seg | VGG or ResNet              | skip connections                   | upsampling(Transposed Convolution)                           |
| UNet | seg | encoder                    | skip connections                   | decoder(Transposed Convolution)                              |
RCNN, SSD FCOS  vs  YOLO  
[[CNNs backbone vs YOLO backbone]]
1. YOLO 移除FC layers, 減少downsampling, C2f減少計算量 

RCNN, SSD FCOS  vs  FCN, UNet (語義分割)
[[CNNs backbone vs FCN UNet backbone]]
1. FCN移除FC layers (就等於轉為全卷積結構)
2. UNet是FCN的特例, 用對稱設計, 淺層結構和跳躍連接確保低層特徵不丟失
3. UNet 不追求多尺度檢測能力
4. VGG和ResNet為分類設計, 靠 FC 層整合資訊. 當用於檢測與分割：移除 FC 層，保留特徵圖，添加 neck/head（如 FPN、上採樣），使其適應新任務。


|     |                                            |
| --- | ------------------------------------------ |
|     | [[###### Generalized R-CNN]]               |
|     | [[###### Mask R-CNN]]                      |
|     | [[###### YOLOv7]]                          |
|     | [[###### CenterMask2]]                     |
|     | [[###### U-Net]]                           |
|     | [[###### Real-ESRGAN]]                     |
|     | [[###### Transformer]]                     |
|     | [[###### Vision Transformer (ViT)]]        |
|     | [[###### DINOv2]]                          |
|     | [[###### CLIP]]                            |
|     | [[###### SAM (Segment Anything Model)]]    |
|     | [[###### Segment Anything Model 2（SAM 2)]] |
|     | [[###### Stable Diffusion + ControlNet]]   |
|     |                                            |

 ### **Generalized R-CNN**

Generalized R-CNN 是一種基於兩階段目標檢測框架的通用模型。該模型的設計是為了靈活處理多種目標檢測、分割和其他相關任務，如 Faster R-CNN 和 Mask R-CNN 就是 Generalized R-CNN 的具體實現。以下將詳細解釋該模型的設計特點、架構、Block 結構、輸入輸出、目標函數、作用及重要特性，並以具體案例和 PyTorch 代碼示例進行說明。

![[Pasted image 20250113143208.png]]
###### Generalized R-CNN

### **1. 設計特點**

1. **兩階段檢測（Two-Stage Detection）**：
    
    - **第一階段**（RPN）：生成候選框（Region Proposals）。
    - **第二階段**（RoI Heads）：對候選框進行分類、回歸調整（Bounding Box Regression），並可選地進行像素級分割。
2. **模組化設計（Modular Design）**：
    
    - 包括骨幹網絡（Backbone Network）、RPN（Region Proposal Network）和 RoI Heads 等多個可擴展模塊。
3. **靈活性（Flexibility）**：
    
    - 可以擴展為不同的目標檢測任務，如目標分割（Mask R-CNN）和關鍵點檢測（Keypoint R-CNN）。
4. **高效性（Efficiency）**：
    
    - 通過多尺度特徵提取和精確候選框回歸，在準確率和速度之間取得平衡。

---

### **2. 架構（Architecture）**

Generalized R-CNN 的架構由以下幾個部分組成：

#### **(1) Backbone（骨幹網絡）**

- **功能**：提取影像的多尺度特徵。
- **常用網絡**：ResNet、ResNeXt、FPN（Feature Pyramid Network）。
- **輸出**：多層特徵圖（Feature Maps），尺寸逐層縮小，通道數增加。

#### **(2) Region Proposal Network (RPN)**

- **功能**：從 Backbone 特徵圖生成候選框（Region Proposals）。
- **結構**：
    1. **Anchor Generator**：生成多尺度錨框。
    2. **分類分支（Classification Branch）**：預測錨框是否包含目標。
    3. **回歸分支（Regression Branch）**：調整錨框的位置和大小。
- **輸出**：一組候選框（大小為 N×4N \times 4N×4）及其置信度。

#### **(3) RoI Heads**

- **功能**：對候選框進行精確分類和邊界框回歸。
- **結構**：
    1. **RoI Align**：將候選框特徵映射到固定大小（如 7×77 \times 77×7）。
    2. **分類分支（Classification Branch）**：預測類別。
    3. **回歸分支（Regression Branch）**：調整邊界框位置。
    4. **可選分割分支（Mask Branch）**：生成像素級分割掩碼。
- **輸出**：每個目標的類別、精確位置及可選的分割掩碼。

---

### **3. Block 架構**

#### **(1) Backbone Block**

- **輸入**：原始影像，大小為 1024×1024×31024 \times 1024 \times 31024×1024×3。
- **結構**：多層卷積（Convolution）、批量歸一化（Batch Normalization）和激活函數（ReLU）。
- **輸出**：特徵圖，例如大小為 256×256×256256 \times 256 \times 256256×256×256。

#### **(2) RPN Block**

- **輸入**：Backbone 的特徵圖。
- **結構**：
    - 卷積層進行特徵提取。
    - 分別輸出錨框分類和位置回歸結果。
- **輸出**：候選框，大小為 N×4N \times 4N×4。

#### **(3) RoI Head Block**

- **輸入**：RPN 候選框和對應的特徵。
- **結構**：
    - RoI Align：將特徵映射到固定大小。
    - 分類和回歸分支。
- **輸出**：精確的目標類別和邊界框。

---

### **4. 輸入與輸出（Input/Output）**

#### **輸入**

- 一段影片，每幀大小為 1024×1024×31024 \times 1024 \times 31024×1024×3。
- 數據預處理：
    - 縮放至 800×800800 \times 800 800×800。
    - 歸一化到範圍 [0,1][0,1][0,1]。

#### **輸出**

- 每幀的檢測結果：
    - **邊界框（Bounding Boxes）**：每個目標的 x,y,w,hx, y, w, hx,y,w,h。
    - **目標類別（Class Labels）**。
    - **可選分割掩碼（Segmentation Mask）**（如果是 Mask R-CNN）。

---

### **5. 目標函數（Objective Function）**

Generalized R-CNN 的目標函數由 RPN 和 RoI Head 的損失組成，主要包括：

1. **分類損失**：確保候選框正確分類。
2. **回歸損失**：精確調整候選框位置。
3. **分割損失（可選）**：在 Mask R-CNN 中使用。

---

### **6. 作用及重要特性**

1. **靈活性**：
    
    - 可用於多種場景，如物體檢測、分割、關鍵點檢測等。
2. **高準確性**：
    
    - 兩階段結構在準確性上優於單階段方法（如 YOLO）。
3. **模組化**：
    
    - 每個部分可靈活替換和擴展（如更換 Backbone 或添加新分支）。

---

### **7. 具體案例：輸入影片到輸出目標**

#### **步驟**

1. **輸入影片**：
    
    - 一段影片，每幀大小為 1024×1024×31024 \times 1024 \times 31024×1024×3。
2. **數據預處理**：
    
    - 縮放到 800×800800 \times 800 800×800。
    - 歸一化像素值。
3. **Backbone 提取特徵**：
    
    - 生成多層特徵圖（例如 200×200×256200 \times 200 \times 256200×200×256）。
4. **RPN 生成候選框**：
    
    - 每幀輸出 100010001000 個候選框。
5. **RoI Head 精細檢測**：
    
    - 使用 RoI Align 調整候選框特徵。
    - 輸出每個目標的類別與邊界框。
6. **輸出結果**：
    
    - 每幀返回所有目標的邊界框、類別標籤和分割掩碼（可選）。

---

### **8. PyTorch 代碼**

以下是 Generalized R-CNN 的 PyTorch 實現
```python
import torch
import torch.nn as nn
from torchvision.ops import RoIAlign

# Backbone 模塊
class Backbone(nn.Module):
    def __init__(self, out_channels=256):
        super(Backbone, self).__init__()
        # 更複雜的特徵提取網絡
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 添加更多卷積層
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        features = self.layer2(x)
        return features

# RPN 模塊
class RPN(nn.Module):
    def __init__(self, in_channels=256, anchor_num=9):
        super(RPN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.cls_layer = nn.Conv2d(256, anchor_num, kernel_size=1)
        self.reg_layer = nn.Conv2d(256, anchor_num * 4, kernel_size=1)
        
        # 初始化權重
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.constant_(layer.bias, 0)

    def forward(self, features):
        x = self.conv(features)
        objectness = self.cls_layer(x)
        bbox_reg = self.reg_layer(x)
        return objectness, bbox_reg

# RoI Heads 模塊
class RoIHeads(nn.Module):
    def __init__(self, num_classes=21, in_channels=256, roi_size=7):
        super(RoIHeads, self).__init__()
        self.roi_align = RoIAlign(
            output_size=(roi_size, roi_size),
            spatial_scale=1.0,
            sampling_ratio=2
        )
        
        # 添加 dropout
        self.fc = nn.Sequential(
            nn.Linear(in_channels * roi_size * roi_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # 分類和回歸層
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)
        
        # 初始化權重
        for layer in self.fc.modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.constant_(layer.bias, 0)
        
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, features, proposals, image_shapes):
        roi_features = self.roi_align(features, proposals, image_shapes)
        roi_features = roi_features.flatten(start_dim=1)
        fc_features = self.fc(roi_features)
        cls_scores = self.cls_score(fc_features)
        bbox_deltas = self.bbox_pred(fc_features)
        return cls_scores, bbox_deltas

# Generalized R-CNN
class GeneralizedRCNN(nn.Module):
    def __init__(self, num_classes=21, min_size=800, max_size=1333):
        super(GeneralizedRCNN, self).__init__()
        self.backbone = Backbone(out_channels=256)
        self.rpn = RPN(in_channels=256, anchor_num=9)
        self.roi_heads = RoIHeads(num_classes=num_classes, in_channels=256)
        self.min_size = min_size
        self.max_size = max_size
        
    def transform_image(self, images):
        # 圖像預處理
        original_sizes = [(img.shape[-2], img.shape[-1]) for img in images]
        
        # 標準化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        images = [(img / 255.0 - mean) / std for img in images]
        
        return images, original_sizes

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("在訓練模式下必須提供targets")
            
        images, original_sizes = self.transform_image(images)
        
        # 特徵提取
        features = self.backbone(images)
        
        # RPN 處理
        objectness, bbox_reg = self.rpn(features)
        
        # 生成proposals (簡化版本)
        proposals = torch.rand((len(images), 100, 4))
        image_shapes = [(images.size(2), images.size(3))] * len(images)
        
        # RoI 處理
        cls_scores, bbox_deltas = self.roi_heads(features, proposals, image_shapes)
        
        result = {
            "cls_scores": cls_scores,
            "bbox_deltas": bbox_deltas,
            "proposals": proposals
        }
        
        if self.training:
            losses = self.compute_losses(result, targets)
            result.update(losses)
            
        return result


```



### **Mask R-CNN**

Mask R-CNN 是基於 **Faster R-CNN** 的一種拓展，用於進行目標檢測和實例分割（Instance Segmentation）。它在 Faster R-CNN 的基礎上增加了像素級的分割分支，因此不僅可以輸出每個目標的邊界框（Bounding Box），還能生成高分辨率的分割掩碼（Segmentation Mask）。

![[Pasted image 20250113143613.png]]
###### Mask R-CNN


### **1. 設計特點**

1. **實例分割（Instance Segmentation）**：
    
    - 每個目標除了有邊界框外，還會有像素級的分割掩碼。
2. **兩階段結構（Two-Stage Structure）**：
    
    - 第一階段：使用 Region Proposal Network (RPN) 生成候選框（Region Proposals）。
    - 第二階段：將候選框送入 RoI Head 進行精細分類、邊界框回歸和掩碼生成。
3. **RoI Align 技術**：
    
    - 改進了 Faster R-CNN 中的 RoI Pooling，能夠更準確地對齊候選框與特徵圖，特別有助於生成高分辨率分割掩碼。
4. **靈活性**：
    
    - Mask R-CNN 的設計非常模組化，可以擴展為其他任務，如姿態估計（Keypoint Detection）。

---

### **2. 架構（Architecture）**

Mask R-CNN 的架構分為以下幾部分：

#### **(1) Backbone（骨幹網絡）**

- **功能**：提取影像的多尺度特徵。
- **常用網絡**：ResNet、ResNeXt，通常結合 FPN（Feature Pyramid Network）使用。
- **輸出**：多尺度特徵圖，例如 P2,P3,P4,P5,P6P2, P3, P4, P5, P6P2,P3,P4,P5,P6，每層對應不同的分辨率。

#### **(2) Region Proposal Network (RPN)**

- **功能**：從 Backbone 的特徵圖中生成候選框。
- **結構**：
    1. **Anchor Generator**：生成錨框（Anchors）。
    2. **分類分支（Classification Branch）**：判斷錨框是否包含目標。
    3. **回歸分支（Regression Branch）**：對錨框位置進行調整。
- **輸出**：候選框（大小為 N×4N \times 4N×4）及其置信度。

#### **(3) RoI Heads**

- **功能**：對候選框進行精確分類、邊界框回歸和掩碼生成。
- **結構**：
    1. **RoI Align**：將候選框映射到固定大小的特徵圖。
    2. **分類與回歸分支**：輸出目標的類別和精確位置。
    3. **掩碼分支**：
        - 卷積結構生成每個類別的分割掩碼。
- **輸出**：
    - 邊界框和類別標籤。
    - 每個目標的分割掩碼。

---

### **3. Block 架構**

#### **(1) Backbone Block**

- **輸入**：原始影像 1024×1024×31024 \times 1024 \times 31024×1024×3。
- **結構**：ResNet 或 ResNeXt 的殘差模塊。
- **輸出**：多尺度特徵圖，例如 256×256×256256 \times 256 \times 256256×256×256。

#### **(2) RPN Block**

- **輸入**：Backbone 的特徵圖。
- **結構**：
    - 卷積層提取區域特徵。
    - 生成分類和位置回歸結果。
- **輸出**：候選框 N×4N \times 4N×4。

#### **(3) RoI Head Block**

- **輸入**：候選框和對應的特徵圖。
- **結構**：
    - **RoI Align**：將候選框對應的特徵對齊到固定大小（如 7×77 \times 77×7）。
    - **分類與回歸分支**：生成邊界框和類別。
    - **掩碼分支**：
        - 多層卷積，輸出每個類別的分割掩碼。
- **輸出**：目標的類別、精確位置和分割掩碼。

---

### **4. 輸入與輸出（Input/Output）**

#### **輸入**

- 一段影片，每幀大小為 1024×1024×31024 \times 1024 \times 31024×1024×3。
- **數據預處理**：
    1. 縮放影像（例如 800×800800 \times 800800×800）。
    2. 歸一化像素值到範圍 [0,1][0, 1][0,1]。

#### **輸出**

- 每幀的目標檢測和分割結果：
    - **邊界框（Bounding Box）**：每個目標的 x,y,w,hx, y, w, hx,y,w,h。
    - **類別標籤（Class Labels）**。
    - **分割掩碼（Segmentation Mask）**：每個目標的像素級分割結果。

---

### **5. 目標函數（Objective Function）**

Mask R-CNN 的損失函數包括三部分：

1. **分類損失（Classification Loss）**：用於候選框的類別分類。
2. **回歸損失（Regression Loss）**：用於調整邊界框位置。
3. **掩碼損失（Mask Loss）**：用於生成分割掩碼。

---

### **6. 作用及重要特性**

1. **多任務處理**：
    
    - 同時進行目標檢測和分割，適合需要精確定位和分割的場景。
2. **高分辨率分割**：
    
    - 使用 RoI Align 改善分割掩碼的細節，生成更高質量的掩碼。
3. **模組化設計**：
    
    - Backbone、RPN 和 RoI Heads 可靈活替換，適用於多種任務。
4. **應用場景**：
    
    - 自動駕駛、醫學影像分析、智能監控等。

---

### **7. 具體案例：從輸入影片到輸出目標**

#### **步驟**

1. **輸入影片**：
    
    - 一段影片，每幀大小為 1024×1024×31024 \times 1024 \times 31024×1024×3。
2. **數據預處理**：
    
    - 縮放到 800×800800 \times 800800×800 並進行歸一化。
3. **Backbone 提取特徵**：
    
    - 輸出多尺度特徵圖（例如 200×200×256200 \times 200 \times 256200×200×256）。
4. **RPN 生成候選框**：
    
    - 每幀輸出 100010001000 個候選框。
5. **RoI Head 精細檢測與分割**：
    
    - **RoI Align** 調整候選框特徵。
    - 輸出邊界框、類別標籤和分割掩碼。
6. **輸出結果**：
    
    - 每幀返回所有目標的邊界框、類別標籤和分割掩碼。

---

### **8. PyTorch 代碼**

以下是 Mask R-CNN 的 PyTorch 實現：
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign

class Backbone(nn.Module):
    def __init__(self, out_channels=256):
        super(Backbone, self).__init__()
        # 增強特徵提取能力
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 添加更多卷積層
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        features = self.layer2(x)
        return features

class RPN(nn.Module):
    def __init__(self, in_channels=256, anchor_num=9):
        super(RPN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.cls_layer = nn.Conv2d(256, anchor_num, kernel_size=1)
        self.reg_layer = nn.Conv2d(256, anchor_num * 4, kernel_size=1)
        
        # 初始化權重
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.constant_(layer.bias, 0)

    def forward(self, features):
        x = self.conv(features)
        objectness = self.cls_layer(x)
        bbox_reg = self.reg_layer(x)
        return objectness, bbox_reg

class RoIHeads(nn.Module):
    def __init__(self, num_classes=21, in_channels=256, roi_size=7):
        super(RoIHeads, self).__init__()
        self.roi_align = RoIAlign(
            output_size=(roi_size, roi_size),
            spatial_scale=1.0,
            sampling_ratio=2
        )
        
        roi_size = roi_size * roi_size * in_channels
        self.fc = nn.Sequential(
            nn.Linear(roi_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)
        
        # 初始化權重
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.constant_(layer.bias, 0)

    def forward(self, features, proposals, image_shapes):
        roi_features = self.roi_align(features, proposals, image_shapes)
        roi_features = roi_features.flatten(start_dim=1)
        fc_features = self.fc(roi_features)
        cls_scores = self.cls_score(fc_features)
        bbox_deltas = self.bbox_pred(fc_features)
        return cls_scores, bbox_deltas

class MaskBranch(nn.Module):
    def __init__(self, in_channels=256, num_classes=21):
        super(MaskBranch, self).__init__()
        self.mask_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        
        self.roi_align = RoIAlign(
            output_size=(14, 14),
            spatial_scale=1.0,
            sampling_ratio=2
        )

    def forward(self, features, proposals, image_shapes):
        roi_features = self.roi_align(features, proposals, image_shapes)
        masks = self.mask_head(roi_features)
        return masks

class MaskRCNN(nn.Module):
    def __init__(self, num_classes=21, min_size=800, max_size=1333):
        super(MaskRCNN, self).__init__()
        self.backbone = Backbone(out_channels=256)
        self.rpn = RPN(in_channels=256, anchor_num=9)
        self.roi_heads = RoIHeads(num_classes=num_classes, in_channels=256)
        self.mask_branch = MaskBranch(in_channels=256, num_classes=num_classes)
        self.min_size = min_size
        self.max_size = max_size

    def transform_image(self, images):
        # 圖像預處理
        original_sizes = [(img.shape[-2], img.shape[-1]) for img in images]
        
        # 標準化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        images = [(img / 255.0 - mean) / std for img in images]
        
        return images, original_sizes

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("在訓練模式下必須提供targets")
            
        images, original_sizes = self.transform_image(images)
        
        # 特徵提取
        features = self.backbone(images)
        
        # RPN 處理
        objectness, bbox_reg = self.rpn(features)
        
        # 生成proposals (簡化版本)
        proposals = torch.rand((len(images), 100, 4))
        image_shapes = [(images.size(2), images.size(3))] * len(images)
        
        # RoI 處理
        cls_scores, bbox_deltas = self.roi_heads(features, proposals, image_shapes)
        
        # Mask 預測
        masks = self.mask_branch(features, proposals, image_shapes)
        
        result = {
            "cls_scores": cls_scores,
            "bbox_deltas": bbox_deltas,
            "masks": masks,
            "proposals": proposals
        }
        
        if self.training:
            losses = self.compute_losses(result, targets)
            result.update(losses)
            
        return result



```





### **YOLOv7**

**YOLOv7** 是 YOLO 系列模型的最新版本之一，專注於**目標檢測（Object Detection）**。YOLOv7 進一步提升了精度和速度的平衡，通過優化模型結構、訓練策略及引入新技術，在多個基準數據集上達到 SOTA（State-Of-The-Art）性能。

以下是對 **YOLOv7** 的詳細中文解釋，包括設計特點、架構、Block 結構、輸入輸出、目標函數、作用及重要特性，並附帶 PyTorch 實現代碼（模型結構部分）。

![[Pasted image 20250113144434.png]]

---
###### YOLOv7

### **1. 設計特點**

1. **高效性與輕量化（Efficiency and Lightweight Design）**：
    
    - 與 YOLOv5 相比，YOLOv7 提升了推理速度並降低了參數量。
    - 提供多種模型尺寸（如 YOLOv7-tiny，適合嵌入式設備）。
2. **跨層聯結（Cross-Stage Partial Connections, CSP）**：
    
    - 採用 CSPNet（Cross Stage Partial Network）來提高梯度流通性，減少冗餘計算。
3. **新技術引入**：
    
    - **ELAN（Extended Efficient Layer Aggregation Network）**：用於加強多層特徵的融合。
    - **SPP（Spatial Pyramid Pooling）**：對多尺度特徵進行聚合，提高對大目標的檢測能力。
4. **訓練技術改進**：
    
    - **自動錨框選擇（AutoAnchor）**：自動生成最優的錨框。
    - **動態標籤分配（Dynamic Label Assignment）**：更高效的標籤分配策略。

---

### **2. 架構（Architecture）**

YOLOv7 的架構包含以下主要部分：

#### **(1) Backbone（骨幹網絡）**

- **功能**：提取影像的多尺度特徵。
- **結構**：
    - 使用 CSPNet（Cross Stage Partial Network）結構。
    - 引入 ELAN（Extended Efficient Layer Aggregation Network），提高深層特徵的學習能力。
- **輸出**：多尺度特徵圖，用於後續的檢測任務。

#### **(2) Neck（頸部）**

- **功能**：進一步融合來自不同層的特徵。
- **結構**：
    - SPP（Spatial Pyramid Pooling）：聚合多尺度特徵。
    - PAN（Path Aggregation Network）：強化上下文信息傳播。
- **輸出**：融合後的特徵圖。

#### **(3) Head（檢測頭）**

- **功能**：進行目標檢測，包括邊界框和類別的預測。
- **結構**：
    - 每個檢測頭輸出多尺度預測結果。
    - 使用 sigmoid 激活函數輸出每個錨框的目標置信度和類別概率。

---

### **3. Block 架構**

YOLOv7 的架構由以下 Block 組成：

#### **(1) CSP Block**

- **輸入**：特徵圖，例如 1024×1024×31024 \times 1024 \times 31024×1024×3。
- **結構**：
    1. 將輸入通道分為兩部分（Partial Connections）。
    2. 一部分通過卷積提取特徵，另一部分直接傳遞。
    3. 最後進行特徵融合。
- **輸出**：壓縮後的特徵圖。

#### **(2) ELAN Block**

- **功能**：進一步增強特徵學習。
- **結構**：
    1. 多層卷積進行特徵提取。
    2. 添加特徵融合機制。
- **輸出**：經過多層卷積融合的特徵。

#### **(3) SPP Block**

- **功能**：多尺度特徵聚合。
- **結構**：
    1. 使用不同大小的池化核（如 1×11 \times 11×1、5×55 \times 55×5、9×99 \times 99×9）。
    2. 將多尺度池化特徵進行拼接。
- **輸出**：多尺度的聚合特徵圖。

#### **(4) PAN Block**

- **功能**：路徑聚合，加強特徵傳播。
- **結構**：
    1. 自下而上和自上而下的路徑聚合。
    2. 強化對小目標的檢測能力。
- **輸出**：上下文加強的特徵圖。

---

### **4. 輸入與輸出（Input/Output）**

#### **輸入**

- **數據格式**：RGB 圖像或影片。
- **尺寸**：每幀大小 1024×1024×31024 \times 1024 \times 31024×1024×3。
- **預處理**：
    1. 將影像縮放到 640×640640 \times 640640×640 或指定大小。
    2. 歸一化像素值到範圍 [0,1][0, 1][0,1]。

#### **輸出**

- 每個目標的：
    - **邊界框（Bounding Box）**：形如 x,y,w,hx, y, w, hx,y,w,h。
    - **目標置信度（Object Confidence）**：範圍 [0,1][0, 1][0,1]。
    - **類別概率（Class Probability）**：範圍 [0,1][0, 1][0,1]。

---

### **5. 目標函數（Objective Function）**

YOLOv7 的損失函數由三部分組成：

1. **分類損失（Classification Loss）**：
    
    - 確保預測的類別與目標類別一致。
2. **回歸損失（Regression Loss）**：
    
    - 優化邊界框的預測位置和大小。
3. **置信度損失（Objectness Loss）**：
    
    - 判斷每個錨框是否包含目標。

---

### **6. 作用及重要特性**

1. **實時性能**：
    
    - 提供高效的檢測速度，適合實時應用。
2. **多尺度檢測**：
    
    - 能夠檢測不同大小的目標，對小目標的檢測效果提升。
3. **高準確性**：
    
    - 使用更優的訓練策略和結構設計，在準確性上超越 YOLOv4 和 YOLOv5。
4. **靈活性**：
    
    - 模型大小可調節，適合不同硬件設備。

---

### **7. 具體案例：從輸入影片到輸出目標**

#### **步驟**

1. **輸入影片**：
    
    - 一段影片，每幀大小為 1024×1024×31024 \times 1024 \times 31024×1024×3。
2. **數據預處理**：
    
    - 縮放到 640×640640 \times 640640×640，並進行像素值歸一化。
3. **Backbone 提取特徵**：
    
    - 使用 CSPNet 和 ELAN 模塊提取多層次特徵。
4. **Neck 聚合特徵**：
    
    - 使用 SPP 和 PAN 將特徵進一步融合。
5. **Head 檢測目標**：
    
    - 每幀返回所有目標的邊界框、置信度和類別概率。
    - 
### **8. PyTorch 代碼：YOLOv7 模型結構**

以下為 YOLOv7 的模型結構代碼，展示 Backbone 和 Neck 的設計。
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNSiLU(nn.Module):
    """Convolution + BatchNorm + SiLU activation"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, groups=1):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ELAN(nn.Module):
    """Extended Linear Aggregation Node"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels // 2
        self.conv1 = ConvBNSiLU(in_channels, mid_channels, 1)
        self.conv2 = ConvBNSiLU(mid_channels, mid_channels, 3)
        self.conv3 = ConvBNSiLU(mid_channels, mid_channels, 3)
        self.conv4 = ConvBNSiLU(mid_channels, mid_channels, 3)
        self.conv5 = ConvBNSiLU(mid_channels * 4, out_channels, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        out = self.conv5(torch.cat([x1, x2, x3, x4], dim=1))
        return out

class ELAN_Block(nn.Module):
    """ELAN Block with downsampling"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = ConvBNSiLU(in_channels, out_channels, 3, stride=2)
        self.elan = ELAN(out_channels, out_channels)

    def forward(self, x):
        x = self.downsample(x)
        x = self.elan(x)
        return x

class SPPCSPBlock(nn.Module):
    """Spatial Pyramid Pooling CSP block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = in_channels // 2
        self.conv1 = ConvBNSiLU(in_channels, mid_channels, 1)
        self.conv2 = ConvBNSiLU(in_channels, mid_channels, 1)
        self.conv3 = ConvBNSiLU(mid_channels * 4, out_channels, 1)
        self.pools = nn.ModuleList([
            nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2)
            for k in [5, 9, 13]
        ])

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        pools = [x2]
        pools.extend([pool(x2) for pool in self.pools])
        x2 = torch.cat(pools, dim=1)
        return self.conv3(torch.cat([x1, x2], dim=1))

class YOLOv7(nn.Module):
    def __init__(self, num_classes=80, input_channels=3):
        super().__init__()
        
        # Backbone
        self.stem = nn.Sequential(
            ConvBNSiLU(input_channels, 32, 3),
            ConvBNSiLU(32, 64, 3, stride=2),
            ConvBNSiLU(64, 64, 3)
        )
        
        self.stage1 = ELAN_Block(64, 128)
        self.stage2 = ELAN_Block(128, 256)
        self.stage3 = ELAN_Block(256, 512)
        self.stage4 = ELAN_Block(512, 1024)
        
        self.spp = SPPCSPBlock(1024, 1024)
        
        # Head
        self.head = nn.ModuleList()
        for out_channels in [512, 256, 128]:
            self.head.append(
                nn.Sequential(
                    ConvBNSiLU(1024, out_channels, 1),
                    ConvBNSiLU(out_channels, out_channels * 2, 3),
                    ConvBNSiLU(out_channels * 2, out_channels, 1)
                )
            )
        
        # Detection layers
        self.det_layers = nn.ModuleList()
        for out_channels in [128, 256, 512]:
            self.det_layers.append(
                nn.Conv2d(out_channels, 3 * (5 + num_classes), 1)
            )

    def forward(self, x):
        # Backbone
        x = self.stem(x)
        x1 = self.stage1(x)      # 1/4
        x2 = self.stage2(x1)     # 1/8
        x3 = self.stage3(x2)     # 1/16
        x4 = self.stage4(x3)     # 1/32
        
        x4 = self.spp(x4)
        
        # Head
        outputs = []
        for i, head in enumerate(self.head):
            feat = head(x4 if i == 0 else outputs[-1])
            outputs.append(feat)
            
        # Detection
        results = []
        for feat, det_layer in zip(outputs, self.det_layers):
            results.append(det_layer(feat))
            
        if self.training:
            return results
        else:
            return self.postprocess(results)
            
    def postprocess(self, outputs):
        """後處理：將輸出轉換為邊界框"""
        batch_size = outputs[0].shape[0]
        predictions = []
        
        for output in outputs:
            # 重塑輸出為 [batch, anchors, grid_h, grid_w, xywh + obj + classes]
            batch, _, grid_h, grid_w = output.shape
            output = output.view(batch, 3, -1, grid_h, grid_w).permute(0, 1, 3, 4, 2)
            predictions.append(output)
            
        return predictions

def create_yolov7_model(num_classes=80, pretrained=False):
    model = YOLOv7(num_classes=num_classes)
    if pretrained:
        # 載入預訓練權重的邏輯
        pass
    return model

# 測試代碼
if __name__ == "__main__":
    model = create_yolov7_model()
    x = torch.randn(1, 3, 640, 640)
    outputs = model(x)
    
    # 打印每個檢測層的輸出大小
    if isinstance(outputs, list):
        for i, out in enumerate(outputs):
            print(f"Detection layer {i + 1} output shape:", out.shape)


# 創建模型
model = create_yolov7_model(num_classes=80)

# 訓練模式
model.train()
x = torch.randn(1, 3, 640, 640)
outputs = model(x)  # 返回原始檢測輸出

# 推理模式
model.eval()
with torch.no_grad():
    predictions = model(x)  # 返回處理後的預測結果


```



![[Pasted image 20250113145420.png]]
###### CenterMask2


### **CenterMask2**

**CenterMask2** 是基於 **FCOS（Fully Convolutional One-Stage Object Detection）** 和 **Mask R-CNN** 的實例分割（Instance Segmentation）模型，專注於實現高效的單階段目標檢測和分割。它結合了 FCOS 的一階段檢測架構和 Mask R-CNN 的像素級分割功能，實現了高效且準確的目標分割。

---

### **1. 設計特點**

1. **單階段結構（One-Stage Architecture）**：
    
    - 基於 FCOS 的單階段目標檢測模型，取消了傳統的候選框（anchor boxes），減少了計算成本。
2. **實例分割（Instance Segmentation）**：
    
    - 在 FCOS 的基礎上，添加了 Mask 分支，用於生成目標的像素級分割掩碼。
3. **中心感知機制（Center-Aware Features）**：
    
    - 通過中心感知特徵提高了目標的分割精度，特別是對小目標效果更佳。
4. **特徵金字塔網絡（Feature Pyramid Network, FPN）**：
    
    - 提取多尺度特徵，用於同時檢測小目標和大目標。
5. **簡化的設計**：
    
    - 與兩階段模型（如 Mask R-CNN）相比，CenterMask2 的計算成本更低，且推理速度更快。

---

### **2. 架構（Architecture）**

CenterMask2 的架構由以下幾個模塊組成：

#### **(1) Backbone（骨幹網絡）**

- 負責提取影像的多尺度特徵。
- 通常使用 ResNet 或 ResNeXt，並結合 FPN（Feature Pyramid Network）以增強多尺度檢測能力。
- 輸出多層特徵圖 P3,P4,P5,P6,P7P3, P4, P5, P6, P7P3,P4,P5,P6,P7。

#### **(2) FCOS 檢測頭（Detection Head）**

- 使用全卷積結構直接在特徵圖上預測每個位置的目標分類、邊界框和中心度量（centerness）。
- 包括：
    1. **分類分支（Classification Branch）**：預測每個像素點的類別。
    2. **邊界框回歸分支（Regression Branch）**：預測邊界框的尺寸。
    3. **中心度量分支（Centerness Branch）**：衡量目標中心點的置信度。

#### **(3) Mask 分支（Mask Branch）**

- 將 FCOS 的檢測結果和特徵圖結合，生成每個候選框的像素級分割掩碼。
- 包括：
    1. **RoI Align**：將檢測到的候選框特徵對齊到固定大小。
    2. **多層卷積**：生成細粒度的分割掩碼。

---

### **3. Block 架構**

#### **(1) Backbone Block**

- **輸入**：RGB 影像，大小為 1024×1024×31024 \times 1024 \times 31024×1024×3。
- **結構**：
    - ResNet/ResNeXt 提取深度特徵。
    - FPN 融合多尺度特徵。
- **輸出**：多尺度特徵圖，例如 P3P3P3：大小 256×256×256256 \times 256 \times 256256×256×256。

#### **(2) Detection Head Block**

- **輸入**：多尺度特徵圖（如 P3,P4,P5P3, P4, P5P3,P4,P5）。
- **結構**：
    - 多層卷積分支。
    - 輸出分類概率、邊界框回歸值和中心度量。
- **輸出**：
    - 邊界框：大小 N×4N \times 4N×4。
    - 類別概率：大小 N×CN \times CN×C。
    - 中心度量：大小 N×1N \times 1N×1。

#### **(3) Mask Branch Block**

- **輸入**：來自 Backbone 和 Detection Head 的特徵圖及候選框。
- **結構**：
    - **RoI Align**：將候選框特徵對齊到固定大小。
    - 多層卷積處理，生成分割掩碼。
- **輸出**：
    - 每個候選框的分割掩碼，大小 N×28×28N \times 28 \times 28N×28×28。

---

### **4. 輸入與輸出（Input/Output）**

#### **輸入**

- 一段影片，每幀大小為 1024×1024×31024 \times 1024 \times 31024×1024×3。
- **預處理**：
    1. 將影像縮放到指定大小（如 800×800800 \times 800800×800）。
    2. 將像素值歸一化到 [0,1][0, 1][0,1]。

#### **輸出**

- 每個目標的：
    - **邊界框（Bounding Box）**：大小 N×4N \times 4N×4。
    - **目標類別（Class Labels）**：大小 N×CN \times CN×C。
    - **分割掩碼（Segmentation Mask）**：大小 N×28×28N \times 28 \times 28N×28×28。

---

### **5. 目標函數（Objective Function）**

CenterMask2 的目標函數包括：

1. **分類損失（Classification Loss）**：
    
    - 衡量預測類別與真實類別的一致性。
2. **邊界框回歸損失（Regression Loss）**：
    
    - 優化預測邊界框與真實框的位置差距。
3. **中心度量損失（Centerness Loss）**：
    
    - 確保檢測結果的中心點可靠。
4. **掩碼損失（Mask Loss）**：
    
    - 優化分割掩碼與真實分割結果的匹配度。

---

### **6. 作用及重要特性**

1. **高效性**：
    
    - 單階段結構，計算成本低，推理速度快。
2. **分割性能強**：
    
    - 使用 FCOS 的檢測頭，結合 Mask 分支，在實例分割任務中表現優異。
3. **多尺度檢測**：
    
    - 基於 FPN，能夠同時檢測大小不同的目標。
4. **應用場景**：
    
    - 自動駕駛、智能監控、醫學影像分割等場景。

---

### **7. PyTorch 模型代碼**

以下為 CenterMask2 的 PyTorch 模型代碼實現：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        
        for in_channels in in_channels_list:
            inner_block = ConvBNReLU(in_channels, out_channels, 1)
            layer_block = ConvBNReLU(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)

    def forward(self, features):
        results = []
        last_inner = self.inner_blocks[-1](features[-1])
        results.append(self.layer_blocks[-1](last_inner))

        for idx in range(len(features) - 2, -1, -1):
            inner_lateral = self.inner_blocks[idx](features[idx])
            inner_top_down = F.interpolate(last_inner, size=inner_lateral.shape[-2:], mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.layer_blocks[idx](last_inner))

        return results

class Backbone(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        # ResNet-like backbone
        self.stem = nn.Sequential(
            ConvBNReLU(3, 64, 7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        
        # FPN
        self.fpn = FPN([64, 128, 256, 512], out_channels)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ConvBNReLU(in_channels, out_channels, 3, stride=stride, padding=1))
        for _ in range(1, blocks):
            layers.append(ConvBNReLU(out_channels, out_channels, 3, padding=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return self.fpn([c2, c3, c4, c5])

class DetectionHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        
        tower = []
        for _ in range(4):
            tower.append(ConvBNReLU(in_channels, in_channels, 3, padding=1))
        self.tower = nn.Sequential(*tower)
        
        self.cls_logits = nn.Conv2d(in_channels, num_classes, 3, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, 4, 3, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, 3, padding=1)
        
        # 初始化
        for modules in [self.cls_logits, self.bbox_pred, self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        features = self.tower(x)
        pred_cls = self.cls_logits(features)
        pred_bbox = self.bbox_pred(features)
        pred_centerness = self.centerness(features)
        return pred_cls, pred_bbox, pred_centerness

class MaskBranch(nn.Module):
    def __init__(self, in_channels=256, num_classes=80):
        super().__init__()
        self.mask_head = nn.Sequential(
            ConvBNReLU(in_channels, 256, 3, padding=1),
            ConvBNReLU(256, 256, 3, padding=1),
            ConvBNReLU(256, 256, 3, padding=1),
            ConvBNReLU(256, 256, 3, padding=1),
            nn.ConvTranspose2d(256, 256, 2, 2, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        
        self.roi_align = RoIAlign(
            output_size=(14, 14),
            spatial_scale=1.0,
            sampling_ratio=2
        )

    def forward(self, features, proposals, image_shapes):
        roi_features = self.roi_align(features, proposals, image_shapes)
        return self.mask_head(roi_features)

class CenterMask2(nn.Module):
    def __init__(self, num_classes=80, min_size=800, max_size=1333):
        super().__init__()
        self.backbone = Backbone(out_channels=256)
        self.detection_head = DetectionHead(num_classes=num_classes)
        self.mask_branch = MaskBranch(num_classes=num_classes)
        self.min_size = min_size
        self.max_size = max_size

    def transform_image(self, images):
        # 圖像預處理
        original_sizes = [(img.shape[-2], img.shape[-1]) for img in images]
        
        # 標準化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        images = [(img / 255.0 - mean) / std for img in images]
        
        return images, original_sizes

    def forward(self, images, proposals=None, targets=None):
        if self.training and targets is None:
            raise ValueError("在訓練模式下必須提供targets")
            
        images, original_sizes = self.transform_image(images)
        
        # 特徵提取
        features = self.backbone(images)
        
        results = []
        for feature in features:
            # 檢測頭處理每個 FPN 層
            cls_logits, bbox_preds, centerness = self.detection_head(feature)
            results.append({
                "cls_logits": cls_logits,
                "bbox_preds": bbox_preds,
                "centerness": centerness
            })
        
        if proposals is not None:
            # Mask 分支
            masks = self.mask_branch(features[-1], proposals, original_sizes)
            results[-1]["masks"] = masks
        
        if self.training:
            losses = self.compute_losses(results, targets)
            return losses
        
        return results

    def compute_losses(self, predictions, targets):
        # 實現損失計算邏輯
        pass

def create_centermask2_model(num_classes=80, pretrained=False):
    model = CenterMask2(num_classes=num_classes)
    if pretrained:
        # 載入預訓練權重的邏輯
        pass
    return model

# 測試代碼
if __name__ == "__main__":
    model = create_centermask2_model()
    x = torch.randn(1, 3, 800, 800)
    proposals = torch.rand(1, 100, 4)  # 模擬 100 個提議框
    outputs = model(x, proposals)
    
    for i, output in enumerate(outputs):
        print(f"Level {i}:")
        for k, v in output.items():
            print(f"{k}: {v.shape}")


```


![[Pasted image 20250113145750.png]]

###### U-Net
### **U-Net 模型詳細解釋**

**U-Net** 是一種專為醫學影像分割（Medical Image Segmentation）設計的卷積神經網絡。該模型以 "U" 字形結構命名，由一個對稱的編碼器（Encoder）和解碼器（Decoder）組成，並通過跳躍連接（Skip Connections）將高分辨率特徵融入解碼器中，實現了高效且準確的像素級分割。

ref:  [UNet理解，pytorch实现，源码解读](https://zhuanlan.zhihu.com/p/571760241)
ref: [Unet论文超级详解（附图文：超细节超容易理解）](https://zhuanlan.zhihu.com/p/716339396)
ref: [U-Net原理分析与代码解读](https://zhuanlan.zhihu.com/p/150579454)

---

### **1. 設計特點**

1. **全卷積結構（Fully Convolutional Network, FCN）**：
    
    - 模型由卷積層（Convolutional Layers）和反卷積層（Transposed Convolutional Layers）組成，支持任意大小的輸入影像。
2. **對稱結構（Symmetrical Architecture）**：
    
    - 編碼器和解碼器結構對稱，特徵從影像中提取到最小分辨率，再逐步恢復到輸入的分辨率。
3. **跳躍連接（Skip Connections）**：
    
    - 將編碼器中的高分辨率特徵直接與解碼器對應層融合，保留細節特徵，提升分割精度。
4. **特徵強化**：
    
    - 通過多層特徵融合，模型能捕獲局部和全局上下文信息，適合多種分割任務。

---

### **2. 架構（Architecture）**

U-Net 的架構包含兩個主要部分：

#### **(1) Encoder（編碼器）**

- **功能**：逐步下採樣（Downsampling），提取影像的語義特徵。
- **結構**：
    - 每層由兩個卷積層（Convolutional Layers）、ReLU 激活函數和最大池化層（Max Pooling Layer）組成。
- **輸出**：多層下採樣的特徵圖，分辨率逐層減小，通道數逐層增大。

#### **(2) Decoder（解碼器）**

- **功能**：逐步上採樣（Upsampling），恢復影像分辨率。
- **結構**：
    - 每層由一個反卷積層（Transposed Convolutional Layer）和兩個卷積層組成。
    - 與對應的編碼器層通過跳躍連接融合特徵。

#### **(3) Skip Connections（跳躍連接）**

- **功能**：將編碼器的特徵圖與解碼器的特徵圖拼接，保留低層特徵細節。

---

### **3. Block 架構**

#### **(1) Down Block（下採樣模塊）**

- **輸入**：影像特徵圖，大小 H×W×CH \times W \times CH×W×C。
- **結構**：
    1. 卷積層 + ReLU 激活。
    2. 卷積層 + ReLU 激活。
    3. 最大池化（將分辨率減半）。
- **輸出**：縮小分辨率的特徵圖，大小 H/2×W/2×2CH/2 \times W/2 \times 2CH/2×W/2×2C。

#### **(2) Up Block（上採樣模塊）**

- **輸入**：解碼器特徵圖 H×W×CH \times W \times CH×W×C 和跳躍連接的編碼器特徵圖 H×W×2CH \times W \times 2CH×W×2C。
- **結構**：
    1. 反卷積層（將分辨率擴大到編碼器層大小）。
    2. 拼接編碼器的特徵圖。
    3. 卷積層 + ReLU 激活。
    4. 卷積層 + ReLU 激活。
- **輸出**：恢復分辨率的特徵圖，大小 2H×2W×C2H \times 2W \times C2H×2W×C。

#### **(3) Output Block（輸出模塊）**

- **功能**：生成分割結果。
- **結構**：
    - 卷積層輸出每個像素的類別概率。
- **輸出**：分割結果，大小 H \times W \times \text{num_classes}。

---

### **4. 輸入與輸出（Input/Output）**

#### **輸入**

- **數據格式**：RGB 圖像或灰度圖像。
- **尺寸**：例如 1024×1024×31024 \times 1024 \times 31024×1024×3。
- **預處理**：
    1. 將影像縮放到指定大小（如 512×512512 \times 512512×512）。
    2. 像素值歸一化到 [0,1][0, 1][0,1]。

#### **輸出**

- **分割結果**：
    - 大小為 512×512×C512 \times 512 \times C512×512×C，其中 CCC 為類別數。
    - 每個像素對應一個類別概率。

---

### **5. 目標函數（Objective Function）**

1. **交叉熵損失（Cross-Entropy Loss）**：
    
    - 衡量每個像素的預測與真實類別的一致性。
2. **Dice 損失（Dice Loss）**：
    
    - 用於評估分割結果與真實標註之間的重疊部分，特別適合不平衡數據。

---

### **6. 作用及重要特性**

1. **像素級分割**：
    
    - 能夠對影像中的每個像素進行精確分類。
2. **靈活性**：
    
    - 支持不同大小的輸入影像，適用於多種分割任務。
3. **高效性**：
    
    - 使用全卷積結構，參數少，計算高效。
4. **應用場景**：
    
    - 醫學影像分割（CT/MRI 分割）、道路場景理解（如自動駕駛）等。

---

### **7. 具體案例：從輸入影像到輸出分割結果**

#### **步驟**

1. **輸入影像**：
    
    - 一段影片中的單幀，大小為 1024×1024×31024 \times 1024 \times 31024×1024×3。
2. **數據預處理**：
    
    - 縮放影像至 512×512512 \times 512512×512，並進行歸一化。
3. **編碼器提取特徵**：
    
    - 逐步下採樣，生成多層特徵圖：
        - 512×512×64512 \times 512 \times 64512×512×64
        - 256×256×128256 \times 256 \times 128256×256×128
        - 128×128×256128 \times 128 \times 256128×128×256
4. **解碼器逐步上採樣**：
    
    - 將分辨率恢復，並與對應層的特徵圖拼接。
5. **輸出分割結果**：
    
    - 最終輸出大小為 512×512×C512 \times 512 \times C512×512×C。

---

### **8. PyTorch 模型代碼**

以下為完整的 U-Net PyTorch 實現代碼：
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # 如果使用雙線性插值進行上採樣
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # 處理輸入大小不匹配的情況
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                       diff_y // 2, diff_y - diff_y // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """最後的輸出卷積層"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        # 初始化權重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

def create_unet(n_channels=3, n_classes=1, bilinear=True):
    """創建 U-Net 模型"""
    model = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)
    return model

# 測試代碼
if __name__ == "__main__":
    # 創建模型
    model = create_unet()
    
    # 測試輸入
    batch_size = 1
    channels = 3
    height = 572
    width = 572
    x = torch.randn(batch_size, channels, height, width)
    
    # 前向傳播
    output = model(x)
    
    # 打印輸出形狀
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # 計算模型參數量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")


```

---


### **Real-ESRGAN (Real-World Enhanced Super-Resolution Generative Adversarial Networks)**


![[Pasted image 20250127220601.png]]
###### Real-ESRGAN



#### **設計特點**

Real-ESRGAN 是針對真實世界圖像超分辨率 (Super-Resolution, SR) 的改進模型，解決了 ESRGAN 在處理真實低分辨率圖像時的不足。它具有以下設計特點：

1. **通用性與穩定性 (Generalization & Stability)**：
    
    - 適用於真實世界的低質量圖像，包括噪聲、模糊、多種壓縮伪影的情況。
2. **改進的生成器架構 (Enhanced Generator Architecture)**：
    
    - 基於 **RRDB (Residual-in-Residual Dense Block)**，增強特徵提取和多尺度學習能力。
    - 引入 **第二次降質合成 (Second Degradation Process)**，更貼近真實場景。
3. **更高效的判別器 (Efficient Discriminator)**：
    
    - 使用 **U-Net 判別器 (U-Net Discriminator)**，在感知真實性和穩定性間達到平衡。
4. **多尺度感知損失 (Multi-Scale Perceptual Loss)**：
    
    - 使用多尺度的特徵圖損失來增強細節重建。
5. **降質模擬與域無關性 (Degradation Modeling & Domain Independence)**：
    
    - 使用逼真的降質模擬提升模型的泛化能力。

---

#### **模型架構 (Architecture)**

Real-ESRGAN 的架構主要分為兩部分：生成器 (Generator) 和判別器 (Discriminator)。

##### 1. **生成器 (Generator)**

- 輸入大小：低分辨率圖像 (Low-Resolution Image, LR)，形狀如 `(C, H, W)`，例如 `(3, 64, 64)`。
- 輸出大小：高分辨率圖像 (High-Resolution Image, HR)，形狀如 `(3, 256, 256)`。
- **主要組件**：
    - **初始卷積 (Initial Conv)**：提取低層特徵。
    - **RRDB 模塊 (Residual-in-Residual Dense Block)**：主要特徵提取模塊，包含 23 個 RRDB。
    - **上採樣模塊 (Upsampling Module)**：使用像素分層卷積 (PixelShuffle) 將圖像解析度放大。
    - **最終卷積 (Final Conv)**：生成高分辨率圖像。

##### 2. **判別器 (Discriminator)**

- 輸入大小：生成器輸出的高分辨率圖像，形狀為 `(3, 256, 256)`。
- **主要組件**：
    - 採用 **U-Net 判別器**，在不同尺度上進行判別，增強對小範圍細節的判別能力。
    - 包括多層卷積層、跳躍連接 (Skip Connections)、LeakyReLU 和全連接層。

##### **RRDB 模塊結構 (RRDB Block Structure)**

- **結構**：
    
    text
    
    複製編輯
    
    `Input → DenseBlock1 → DenseBlock2 → DenseBlock3 → Add(Input) → Output`
    
- **細節**：
    - 每個 Dense Block 包含多層卷積層和非線性激活函數 (PReLU)。
    - 殘差連接 (Residual Connection) 和密集跳躍連接 (Dense Skip Connections)。

---

#### **目標函數 (Objective Functions)**

1. **對抗損失 (Adversarial Loss)**：
    - 基於 RaGAN，提升生成圖像的真實感。
2. **感知損失 (Perceptual Loss)**：
    - 使用 VGG 網絡提取的多尺度特徵，提升視覺感知質量。
3. **像素損失 (Pixel Loss)**：
    - 使用 $L_1$ 損失確保像素級準確性。
4. **多尺度損失 (Multi-Scale Loss)**：
    - 將不同尺度的特徵圖差異作為附加約束。

---

#### **數據流具體步驟**

假設輸入圖像大小為 `(3, 64, 64)`，以下為數據流具體步驟：

1. **輸入數據 (Input Data)**：
    
    - 輸入低分辨率圖像 `img_LR`，形狀為 `(B, 3, 64, 64)`，其中 B 為批大小 (batch size)。
2. **初始卷積層 (Initial Conv)**：
    
    - Conv2D 運算，輸出特徵圖形狀為 `(B, 64, 64, 64)`。
3. **RRDB 模塊 (RRDB Block)**：
    
    - 通過 23 個 RRDB 模塊，保持特徵大小不變，形狀為 `(B, 64, 64, 64)`。
4. **上採樣模塊 (Upsampling Module)**：
    
    - 第一次像素分層卷積 (PixelShuffle)：將圖像解析度放大 2 倍，形狀變為 `(B, 64, 128, 128)`。
    - 第二次像素分層卷積：再次放大 2 倍，形狀變為 `(B, 3, 256, 256)`。
5. **輸出高分辨率圖像 (Output)**：
    
    - 最終生成的高分辨率圖像 `img_HR`，形狀為 `(B, 3, 256, 256)`。

---

#### **PyTorch Code**

以下是 Real-ESRGAN 生成器的 PyTorch 實現：
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualDenseBlock(nn.Module):
    """Residual Dense Block"""
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        # 初始化
        self.initialize_weights()

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class RRDB(nn.Module):
    """Residual in Residual Dense Block"""
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class RealESRGANGenerator(nn.Module):
    """Real-ESRGAN Generator"""
    def __init__(
        self,
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4
    ):
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.ModuleList()
        
        # RRDB blocks
        for _ in range(num_block):
            self.body.append(RRDB(num_feat, num_grow_ch))
        
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        # Upsample blocks
        self.upsampling = nn.ModuleList()
        for _ in range(self.scale // 2):
            self.upsampling.extend([
                nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ])
        
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # 初始化權重
        self.initialize_weights()

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = feat.clone()
        
        # RRDB blocks
        for block in self.body:
            body_feat = block(body_feat)
        
        # Global residual learning
        feat = self.conv_body(body_feat) + feat
        
        # Upsampling
        for layer in self.upsampling:
            feat = layer(feat)
        
        feat = self.lrelu(self.conv_hr(feat))
        out = self.conv_last(feat)
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

def create_real_esrgan_model(scale=4, pretrained=False):
    """創建 Real-ESRGAN 模型"""
    model = RealESRGANGenerator(scale=scale)
    if pretrained:
        # 載入預訓練權重的邏輯
        pass
    return model

# 測試代碼
if __name__ == "__main__":
    # 創建模型
    model = create_real_esrgan_model()
    model.eval()
    
    # 測試輸入
    x = torch.randn(1, 3, 64, 64)
    
    # 計算輸出
    with torch.no_grad():
        output = model(x)
    
    # 打印形狀
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # 計算參數量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")


```

---

#### **重要特性總結**

- **改進點**：
    - 更具穩定性的生成器設計。
    - 引入多尺度損失和 U-Net 判別器增強效果。
    - 泛化能力強，適用於真實低質量圖像。
- **應用場景**：
    - 手機照片修復、舊照片或視頻的超分辨率處理。
    - 用於低質量圖像的細節恢復，例如噪聲和模糊情況。


### **Transformer**
![[transformer.webp]]

###### Transformer

#### **設計特點**

Transformer 是一種基於注意力機制 (Attention Mechanism) 的深度學習模型，主要用於自然語言處理 (NLP) 和計算機視覺 (Computer Vision) 任務。以下是其設計特點：

1. **自注意力機制 (Self-Attention Mechanism)**：
    
    - 核心機制是通過 `Query (Q)`、`Key (K)` 和 `Value (V)` 的運算，根據輸入序列的關聯性動態計算權重。
2. **多頭注意力 (Multi-Head Attention)**：
    
    - 將注意力分成多個子空間，同時關注不同的特徵維度，提升模型的表達能力。
3. **位置編碼 (Positional Encoding)**：
    
    - 解決序列數據中位置信息的缺失問題，通過對輸入嵌入 (Embedding) 添加位置資訊。
4. **完全基於注意力**：
    
    - 不再依賴循環結構 (如 RNN、LSTM)，採用並行化設計，更高效處理長序列。
5. **殘差連接 (Residual Connection)**：
    
    - 提高梯度流動，解決深層網絡中梯度消失問題。
6. **層歸一化 (Layer Normalization)**：
    
    - 用於穩定訓練過程，提升模型的收斂速度。

---

#### **模型架構 (Architecture)**

Transformer 的架構包含兩個主要模塊：

1. **編碼器 (Encoder)**
2. **解碼器 (Decoder)**

##### **編碼器 (Encoder)**

- 輸入大小：序列數據形狀為 `(batch_size, seq_len, embed_dim)`，例如 `(32, 128, 512)`。
- 主要結構：
    1. **嵌入層 (Embedding Layer)**：
        - 將輸入的單詞或特徵轉換為固定維度的向量。
    2. **位置編碼 (Positional Encoding)**：
        - 添加位置資訊，輸出形狀與嵌入層相同。
    3. **多層編碼器塊 (Stacked Encoder Blocks)**：
        - 每個編碼器塊包含：
            - 多頭注意力機制 (Multi-Head Attention)
            - 前向全連接層 (Feed-Forward Network, FFN)
            - 殘差連接 (Residual Connections) 和層歸一化 (Layer Norm)。

##### **解碼器 (Decoder)**

- 輸入大小：目標序列形狀為 `(batch_size, seq_len, embed_dim)`。
- 主要結構：
    1. **目標嵌入 (Target Embedding)** 和 **位置編碼 (Positional Encoding)**。
    2. **多層解碼器塊 (Stacked Decoder Blocks)**：
        - 每個解碼器塊包含：
            - 自注意力機制 (Masked Multi-Head Attention)：防止解碼時看見未來的資訊。
            - 編碼器-解碼器注意力機制 (Encoder-Decoder Attention)：融合編碼器輸出的上下文資訊。
            - 前向全連接層和殘差結構。

##### **Block 架構**

1. **多頭注意力機制 (Multi-Head Attention Block)**：
    
    - 結構：

        `Input → Linear(Q, K, V) → Attention(Q, K, V) → Concat → Linear → Output`
        
    - 輸入大小：`(batch_size, seq_len, embed_dim)`。
    - 輸出大小：與輸入相同。
2. **前向全連接層 (Feed-Forward Network)**：
    
    - 結構：

        `Input → Linear(d_model → d_ff) → ReLU → Linear(d_ff → d_model) → Output`
        
    - 輸入大小：`(batch_size, seq_len, d_model)`。
    - 輸出大小：與輸入相同。

---

#### **目標函數**

Transformer 的主要目標函數為交叉熵損失 (Cross-Entropy Loss)，用於計算模型輸出序列與目標序列的差異。

---

#### **數據流具體步驟**

假設輸入數據為文本，形狀為 `(batch_size=32, seq_len=128, vocab_size=10000)`：

1. **嵌入層 (Embedding)**：
    
    - 將每個單詞映射為向量，形狀變為 `(32, 128, 512)`，其中 512 是嵌入維度。
2. **位置編碼 (Positional Encoding)**：
    
    - 為嵌入層輸出添加位置資訊，形狀保持 `(32, 128, 512)`。
3. **多層編碼器塊**：
    
    - 每層包含多頭注意力、前向全連接層和殘差連接，形狀保持不變。
4. **解碼器**：
    
    - 自注意力、編碼器-解碼器注意力融合上下文，最後輸出形狀為 `(32, 128, 512)`。
5. **輸出層**：
    
    - 全連接層將解碼器輸出轉換為詞彙概率分佈，形狀變為 `(32, 128, vocab_size)`。

---

#### **PyTorch Code**

以下為 Transformer 編碼器與解碼器的 PyTorch 實現：
```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 創建位置編碼矩陣
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.size()

        # 計算 Q, K, V
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2), qkv)

        # 注意力計算
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
            
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # 輸出
        x = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        x = self.out_proj(x)
        return self.out_dropout(x)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x

class Transformer(nn.Module):
    def __init__(self, 
                 num_layers=6,
                 d_model=512,
                 num_heads=8,
                 d_ff=2048,
                 dropout=0.1,
                 vocab_size=50000,
                 max_seq_length=5000):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        self.dropout = nn.Dropout(dropout)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)

        # 初始化參數
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x, mask=None):
        # 輸入嵌入和位置編碼
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)

        x = self.norm(x)
        x = self.fc(x)
        return x

def create_mask(size):
    """創建用於自注意力的遮罩"""
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return ~mask

# 測試代碼
if __name__ == "__main__":
    # 模型參數
    vocab_size = 1000
    max_seq_length = 100
    batch_size = 16
    seq_length = 50

    # 創建模型
    model = Transformer(
        num_layers=6,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        dropout=0.1,
        vocab_size=vocab_size,
        max_seq_length=max_seq_length
    )

    # 創建輸入數據
    x = torch.randint(0, vocab_size, (batch_size, seq_length))
    mask = create_mask(seq_length)

    # 前向傳播
    output = model(x, mask)

    # 打印輸出形狀
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # 計算參數量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")


# 創建模型
model = Transformer(
    num_layers=6,
    d_model=512,
    num_heads=8,
    d_ff=2048,
    dropout=0.1,
    vocab_size=1000,
    max_seq_length=100
)

# 準備輸入
x = torch.randint(0, 1000, (16, 50))  # [batch_size, seq_length]
mask = create_mask(50)  # 序列長度的遮罩

# 前向傳播
output = model(x, mask)


```

---

#### **重要特性總結**

1. **核心特性：**
    - **並行化：** 提高長序列處理效率。
    - **自注意力：** 動態學習序列內的依賴關係。
2. **應用場景：**
    - NLP：機器翻譯、文本生成。
    - CV：圖像分類、物體檢測 (如 Vision Transformer)。
3. **優勢：**
    - 高效處理長序列數據。
    - 模型具有高度的可擴展性。


### **Vision Transformer (ViT)**

![[Pasted image 20250127224233.png]]


###### Vision Transformer (ViT)
#### **設計特點**

Vision Transformer (ViT) 是將 Transformer 應用於計算機視覺的模型。它以類似處理文本序列的方式處理圖像，核心理念是將圖像分割成不重疊的小塊（patch），並將每個小塊視為序列的輸入。以下是其主要特點：

1. **圖像塊嵌入 (Patch Embedding)**：
    
    - 將圖像劃分為固定大小的小塊，每個小塊展平 (Flatten) 並嵌入為高維特徵向量，類似於 Transformer 的詞嵌入 (Word Embedding)。
2. **位置編碼 (Positional Encoding)**：
    
    - 添加位置資訊來補充序列缺失的空間結構信息。
3. **純 Transformer 架構**：
    
    - 不依賴卷積結構，完全基於自注意力機制 (Self-Attention)。
4. **並行處理 (Parallel Processing)**：
    
    - 相較於卷積網絡的層級操作，Transformer 可以同時處理整個序列，擅長處理長距依賴關係。
5. **大規模數據需求**：
    
    - ViT 的性能依賴於大規模數據集（如 ImageNet-21k 或 JFT-300M）進行預訓練。

---

#### **模型架構 (Architecture)**

ViT 的架構分為以下幾個部分：

1. **圖像塊嵌入層 (Patch Embedding Layer)**：
    
    - 將輸入圖像分割為不重疊的小塊，並嵌入為固定維度的向量。
    - 輸入大小：`(B, C, H, W)`，例如 `(1, 3, 224, 224)`。
    - 每個塊大小為 `P×P`，例如 `16×16`。
    - 輸出大小：`(B, N, D)`，其中 `N=(H×W)/(P×P)` 是塊數量，`D` 是嵌入維度。
2. **位置編碼 (Positional Encoding)**：
    
    - 將固定或可學習的位置編碼添加到塊嵌入，輸出大小保持為 `(B, N, D)`。
3. **Transformer 編碼器 (Transformer Encoder)**：
    
    - 多層堆疊的 Transformer 編碼器，每層包含：
        - **多頭自注意力 (Multi-Head Self-Attention)**。
        - **前向全連接層 (Feed-Forward Network, FFN)**。
        - **殘差連接 (Residual Connection)** 和 **層歸一化 (Layer Norm)**。
4. **分類頭 (Classification Head)**：
    
    - 使用一個額外的學習標籤 [CLS] 標記，輸出對應於分類結果。

---

#### **Block 架構**

##### **Patch Embedding Block**

- 輸入：`(B, C, H, W)`。
- 步驟：
    - 將圖像劃分為不重疊的塊。
    - 每個塊展平並通過線性層嵌入為向量。
- 輸出：`(B, N, D)`。

##### **Transformer Encoder Block**

1. **多頭自注意力機制 (Multi-Head Self-Attention)**：
    
    - 結構：

        `Input → Linear(Q, K, V) → Attention(Q, K, V) → Concat → Linear → Output`
        
    - 輸入大小：`(B, N, D)`。
    - 輸出大小：與輸入相同。
2. **前向全連接層 (Feed-Forward Network, FFN)**：
    
    - 結構：

        `Input → Linear(D → FFN_dim) → GELU → Linear(FFN_dim → D) → Output`
        
    - 輸入/輸出大小：`(B, N, D)`。
3. **殘差結構**：
    
    - 每層包含兩個殘差結構：

        `Input → Self-Attention → Add(Input) → LayerNorm → FFN → Add → LayerNorm`
        

---

#### **目標函數**

ViT 的主要目標函數為交叉熵損失 (Cross-Entropy Loss)，用於分類任務。

---

#### **數據流具體步驟**

假設輸入圖像大小為 `(1, 3, 224, 224)`，塊大小為 `16×16`，嵌入維度 `D=768`：

1. **圖像塊嵌入**：
    
    - 圖像分割為 `16×16` 的塊，共有 `N=(224/16)²=196` 個塊。
    - 每個塊展平成向量並嵌入到 `D=768` 維度。
    - 輸出形狀為 `(1, 196, 768)`。
2. **位置編碼**：
    
    - 添加固定位置編碼，輸出形狀保持為 `(1, 196, 768)`。
3. **Transformer 編碼器**：
    
    - 通過 12 層 Transformer 編碼器，特徵形狀保持不變。
4. **分類頭**：
    
    - 提取 [CLS] 標記的特徵，輸出分類概率，形狀為 `(1, num_classes)`。

---

#### **PyTorch Code**

以下是 ViT 的 PyTorch 實現：
```python
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """將圖像分割成patch並進行線性嵌入"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        # (B, C, H, W) -> (B, D, H/P, W/P) -> (B, H/P*W/P, D)
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class Attention(nn.Module):
    """多頭自注意力機制"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    """多層感知機"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    """Transformer 編碼器塊"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, 
                 drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                            attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, 
                      act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    """Vision Transformer"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, 
                                    in_channels=in_channels, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                  norm_layer=norm_layer)
            for _ in range(depth)
        ])
        
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize cls token
        torch.nn.init.normal_(self.cls_token, std=.02)

        # Initialize position embedding
        torch.nn.init.normal_(self.pos_embed, std=.02)

        # Initialize all other Linear layers
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Apply Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)
        
        return x[:, 0]  # Return class token

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def create_vit_model(model_name='ViT-B/16', num_classes=1000, has_logits=True):
    """創建 ViT 模型"""
    configs = {
        'ViT-B/16': dict(patch_size=16, embed_dim=768, depth=12, num_heads=12),
        'ViT-L/16': dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16),
        'ViT-H/14': dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16),
    }
    
    config = configs[model_name]
    model = VisionTransformer(patch_size=config['patch_size'],
                            embed_dim=config['embed_dim'],
                            depth=config['depth'],
                            num_heads=config['num_heads'],
                            num_classes=num_classes)
    return model

# 測試代碼
if __name__ == "__main__":
    # 創建模型
    model = create_vit_model('ViT-B/16')
    
    # 測試輸入
    x = torch.randn(1, 3, 224, 224)
    
    # 前向傳播
    output = model(x)
    
    # 打印輸出形狀
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # 計算參數量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

# 創建模型
model = create_vit_model('ViT-B/16', num_classes=1000)

# 準備輸入
x = torch.randn(1, 3, 224, 224)

# 前向傳播
output = model(x)


```

---

#### **重要特性總結**

- **核心特性：**
    - 無卷積架構，基於 Transformer，能處理全局關係。
- **應用場景：**
    - 圖像分類、物體檢測、語義分割（與其他模塊如 DETR 結合）。
- **優勢：**
    - 強大的序列建模能力。
    - 可移植於各種數據模態（例如視頻和醫學影像）。



### **DINOv2 (Distillation with No Labels v2)**

![[Pasted image 20250127225004.png]]


###### DINOv2

DINOv2 是 Meta AI 提出的模型，專注於無需標籤的大規模預訓練，自監督學習方式的高效性，使其在視覺分類、物體檢測和語義分割等任務上表現優秀。以下詳細解釋其設計特點、架構和 PyTorch 實現。

---

#### **設計特點**

1. **自監督學習 (Self-Supervised Learning)**：
    
    - 不需要標籤數據，模型通過對數據本身進行預測和匹配學習。
    - 使用教師-學生架構 (Teacher-Student Framework)，學生模型學習教師模型的輸出。
2. **無標籤知識蒸餾 (Self-Distillation)**：
    
    - 教師模型使用過去的權重更新，輸出更穩定，指導學生模型學習。
3. **基於 Vision Transformer (ViT)**：
    
    - 使用 Vision Transformer 作為主幹網絡，捕捉全局上下文信息。
    - 多層堆疊的 Transformer 提取多尺度特徵。
4. **多視角學習 (Multi-View Learning)**：
    
    - 使用不同尺度的圖像進行多視角輸入，學習多層次語義特徵。
5. **穩定訓練技術**：
    
    - 引入動態教師更新、歸一化和加權損失，提升訓練穩定性和收斂速度。

---

#### **模型架構 (Architecture)**

DINOv2 的架構可以分為以下部分：

1. **圖像塊嵌入 (Patch Embedding)**：
    
    - 將圖像劃分為固定大小的小塊，轉換為嵌入特徵。
2. **Transformer 編碼器 (Transformer Encoder)**：
    
    - 多層堆疊的 Transformer 編碼器。
    - 每層包含：
        - **多頭自注意力 (Multi-Head Self-Attention, MHSA)**。
        - **前向全連接層 (Feed-Forward Network, FFN)**。
        - **殘差連接 (Residual Connections)** 和 **層歸一化 (Layer Norm)**。
3. **多視角輸入 (Multi-View Input)**：
    
    - 多個隨機裁剪的圖像輸入，教師和學生模型各自處理不同視角。
4. **對比學習頭 (Projection Head)**：
    
    - 將 Transformer 的輸出嵌入到低維空間，用於對比學習。
5. **教師-學生框架 (Teacher-Student Framework)**：
    
    - 教師模型權重固定，學生模型通過損失函數學習教師模型的輸出。

---

#### **Block 架構**

##### **Patch Embedding Block**

- 輸入大小：`(B, C, H, W)`，例如 `(1, 3, 224, 224)`。
- 由 Conv2D 實現，將圖像分塊並嵌入到 `D` 維度。
- 輸出大小：`(B, N, D)`，其中 `N=(H×W)/(P×P)`。

##### **Transformer Encoder Block**

- 結構：

    `Input → Multi-Head Self-Attention → Add & LayerNorm → Feed-Forward Network → Add & LayerNorm → Output`
    
- 輸入/輸出大小：`(B, N, D)`。

##### **Projection Head Block**

- 結構：

    `Input → Linear → BatchNorm → ReLU → Linear → Output`
    
- 將高維輸出映射到低維空間，用於對比學習。

---

#### **目標函數**

DINOv2 使用基於對比學習的目標函數，確保多視角輸入的學生模型輸出與教師模型輸出一致，並在相似的特徵空間中對齊。

---

#### **數據流具體步驟**

假設輸入為圖片，形狀 `(1, 3, 224, 224)`：

1. **多視角裁剪**：
    
    - 生成兩個不同尺度的裁剪圖像，形狀仍為 `(1, 3, 224, 224)`。
2. **Patch Embedding**：
    
    - 每個圖像劃分為 `16×16` 小塊，嵌入維度為 `D=768`。
    - 輸出形狀為 `(1, 196, 768)`。
3. **Transformer 編碼器**：
    
    - 多層堆疊的 Transformer，輸出多尺度特徵，形狀保持不變。
4. **Projection Head**：
    
    - 將 Transformer 輸出嵌入到低維度空間，如 `(1, 256)`。
5. **對比學習**：
    
    - 計算學生輸出與教師輸出的對比損失。

---

#### **PyTorch Code**

以下是 DINOv2 的核心結構（省略訓練框架）：
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma

class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., 
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 init_values=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                            attn_drop=attn_drop, proj_drop=drop)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                      act_layer=act_layer, drop=drop)

        if init_values is not None:
            self.gamma_1 = LayerScale(dim, init_values)
            self.gamma_2 = LayerScale(dim, init_values)
        else:
            self.gamma_1 = self.gamma_2 = nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1(self.attn(self.norm1(x))))
        x = x + self.drop_path(self.gamma_2(self.mlp(self.norm2(x))))
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DINOv2(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 init_values=None, proj_dim=256):
        super().__init__()
        
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                    in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=dpr[i], norm_layer=norm_layer, init_values=init_values)
            for i in range(depth)
        ])
        
        self.norm = norm_layer(embed_dim)
        
        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, proj_dim),
            nn.BatchNorm1d(proj_dim, affine=False)
        )

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize patch_embed
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize cls token
        torch.nn.init.normal_(self.cls_token, std=.02)

        # Initialize pos_embed
        torch.nn.init.normal_(self.pos_embed, std=.02)

        # Initialize all other Linear layers
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]  # Return CLS token features

    def forward(self, x):
        x = self.forward_features(x)
        x = self.projection_head(x)
        return F.normalize(x, dim=-1)  # L2 normalize

def create_dinov2_model(model_size='small'):
    """Create DINOv2 model with different sizes"""
    configs = {
        'small': dict(depth=12, embed_dim=384, num_heads=6),
        'base': dict(depth=12, embed_dim=768, num_heads=12),
        'large': dict(depth=24, embed_dim=1024, num_heads=16),
    }
    
    config = configs[model_size]
    model = DINOv2(depth=config['depth'],
                   embed_dim=config['embed_dim'],
                   num_heads=config['num_heads'])
    return model

# 測試代碼
if __name__ == "__main__":
    # 創建模型
    model = create_dinov2_model('base')
    
    # 測試輸入
    x = torch.randn(2, 3, 224, 224)
    
    # 前向傳播
    output = model(x)
    
    # 打印輸出形狀
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # 計算參數量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")


# 創建模型
model = create_dinov2_model('base')

# 準備輸入
x = torch.randn(2, 3, 224, 224)

# 前向傳播
output = model(x)  # 輸出已經 L2 正則化


```

---

#### **重要特性總結**

1. **核心特性**：
    
    - 基於 Transformer，無需標籤的大規模預訓練。
    - 通過多視角學習提升泛化能力。
    - 具備強大的多尺度特徵提取能力。
2. **應用場景**：
    
    - 圖像分類、物體檢測、語義分割等。
3. **優勢**：
    
    - 無需大量標籤數據，能在大規模數據上進行自監督學習。
    - 在下游任務上具備出色的遷移學習性能。


### **CLIP (Contrastive Language-Image Pre-training)**


![[Pasted image 20250127225412.png]]

###### CLIP

#### **設計特點**

CLIP 是 OpenAI 提出的一種多模態模型，能將自然語言 (Natural Language) 和圖像 (Image) 映射到同一個特徵空間中，實現圖像與文本的對比學習。以下是其設計特點：

1. **多模態對比學習 (Multi-Modal Contrastive Learning)**：
    
    - CLIP 使用圖像和文本的對應性進行訓練，不需要額外標籤。
    - 將圖像嵌入 (Image Embedding) 和文本嵌入 (Text Embedding) 映射到共享特徵空間。
2. **通用性 (Generalization)**：
    
    - 在零樣本學習 (Zero-Shot Learning) 中表現優異，可以直接對新類別進行推斷。
3. **架構結合**：
    
    - 使用 Vision Transformer (ViT) 或 ResNet 提取圖像特徵。
    - 使用 Transformer 提取文本特徵。
4. **對比損失 (Contrastive Loss)**：
    
    - 通過最大化匹配的圖像-文本對之間的相似性，同時最小化不匹配對之間的相似性。
5. **大規模預訓練**：
    
    - 在互聯網上的 4 億對圖像-文本數據集上進行訓練，學習豐富的多模態表示。

---

#### **模型架構 (Architecture)**

CLIP 包括兩個主要模塊：

1. **圖像編碼器 (Image Encoder)**：
    
    - 提取圖像特徵，生成固定維度的向量。
    - 支持 ResNet 或 Vision Transformer (ViT)。
2. **文本編碼器 (Text Encoder)**：
    
    - 提取文本特徵，將文本轉換為向量表示。
    - 使用 Transformer 架構處理文本。

---

##### **主要 Block 架構**

1. **圖像編碼器 (Image Encoder)**：
    
    - 如果使用 ViT：
        - **Patch Embedding**：將圖像分割為固定大小的小塊，嵌入為特徵向量。
        - **Transformer Encoder**：多層自注意力和前向全連接層提取特徵。
2. **文本編碼器 (Text Encoder)**：
    
    - 使用標準 Transformer：
        - **嵌入層 (Embedding Layer)**：將詞嵌入轉換為固定維度的向量。
        - **位置編碼 (Positional Encoding)**：添加序列信息。
        - **Transformer Encoder**：多層自注意力提取語義特徵。
3. **對比學習頭 (Contrastive Head)**：
    
    - 將圖像和文本特徵通過線性層歸一化為同一空間中的特徵向量。
4. **對比學習 (Contrastive Learning)**：
    
    - 使用相似性度量（如餘弦相似度）比較圖像和文本嵌入，通過對比損失學習。

---

#### **目標函數**

CLIP 的目標函數是對比損失 (Contrastive Loss)，通過拉近匹配的圖像-文本對，拉遠不匹配對，實現嵌入對齊。

---

#### **數據流具體步驟**

假設輸入數據為：

- 圖像：形狀 `(B, 3, 224, 224)`，例如 `(32, 3, 224, 224)`。
- 文本：形狀 `(B, L)`，例如 `(32, 77)`，其中 `L` 是最大文本序列長度。

**步驟**：

1. **圖像處理**：
    
    - 圖像經過 ViT 的 `Patch Embedding`，轉為嵌入大小 `D=512`。
    - 形狀變為 `(B, N, 512)`，其中 `N` 是圖像塊數量。
    - 經過 Transformer 提取全局特徵，最終得到圖像特徵 `(B, 512)`。
2. **文本處理**：
    
    - 文本經過詞嵌入層，形狀為 `(B, L, 512)`。
    - 加入位置編碼，經過 Transformer 提取特徵，取 CLS 標記，最終得到文本特徵 `(B, 512)`。
3. **對比學習頭**：
    
    - 將圖像和文本特徵通過線性層映射到同一特徵空間。
    - 特徵歸一化後形狀為 `(B, 512)`。
4. **對比損失**：
    
    - 計算圖像和文本特徵間的餘弦相似度，形狀為 `(B, B)`。
    - 對每個正確匹配對 (正樣本) 的相似度最大化，錯誤對 (負樣本) 的相似度最小化。

---

#### **PyTorch Code**

以下是 CLIP 模型的 PyTorch 實現（簡化版）：
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
            
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.attn = MultiheadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3,
                 embed_dim: int = 512, depth: int = 12, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        return x[:, 0]  # Return CLS token features

class TextTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_len: int = 77, embed_dim: int = 512,
                 depth: int = 12, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        x = self.token_embedding(x)
        x = x + self.pos_embedding[:, :x.size(1)]
        x = self.dropout(x)
        
        for block in self.blocks:
            x = block(x, attention_mask)
            
        x = self.norm(x)
        return x[:, 0]  # Return CLS token features

class CLIP(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, vocab_size: int = 49408,
                 embed_dim: int = 512, vision_depth: int = 12, text_depth: int = 12,
                 vision_heads: int = 8, text_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        
        self.visual = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=vision_depth,
            num_heads=vision_heads,
            dropout=dropout
        )
        
        self.textual = TextTransformer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            depth=text_depth,
            num_heads=text_heads,
            dropout=dropout
        )
        
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        
    def forward(self, image: torch.Tensor, text: torch.Tensor,
                attention_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        image_features = F.normalize(self.visual(image), dim=-1)
        text_features = F.normalize(self.textual(text, attention_mask), dim=-1)
        
        return image_features, text_features
    
    def get_similarity(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        return image_features @ text_features.t() * self.logit_scale.exp()

def create_clip_model(model_name: str = 'ViT-B/32') -> CLIP:
    """Create a CLIP model with specified configuration"""
    configs = {
        'ViT-B/32': dict(
            img_size=224,
            patch_size=32,
            embed_dim=512,
            vision_depth=12,
            text_depth=12,
            vision_heads=8,
            text_heads=8
        ),
        'ViT-B/16': dict(
            img_size=224,
            patch_size=16,
            embed_dim=512,
            vision_depth=12,
            text_depth=12,
            vision_heads=8,
            text_heads=8
        ),
    }
    
    if model_name not in configs:
        raise ValueError(f"Model {model_name} not found. Available models: {list(configs.keys())}")
        
    return CLIP(**configs[model_name])

# 測試代碼
if __name__ == "__main__":
    # 創建模型
    model = create_clip_model('ViT-B/32')
    
    # 測試輸入
    batch_size = 2
    image = torch.randn(batch_size, 3, 224, 224)
    text = torch.randint(0, 49408, (batch_size, 77))
    attention_mask = torch.ones(batch_size, 77)
    
    # 前向傳播
    image_features, text_features = model(image, text, attention_mask)
    similarity = model.get_similarity(image_features, text_features)
    
    # 打印輸出形狀
    print(f"Image features shape: {image_features.shape}")
    print(f"Text features shape: {text_features.shape}")
    print(f"Similarity matrix shape: {similarity.shape}")
    
    # 計算參數量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

# 創建模型
model = create_clip_model('ViT-B/32')

# 準備輸入
image = torch.randn(1, 3, 224, 224)
text = torch.randint(0, 49408, (1, 77))
mask = torch.ones(1, 77)

# 計算特徵
image_features, text_features = model(image, text, mask)

# 計算相似度
similarity = model.get_similarity(image_features, text_features)


```

---

#### **重要特性總結**

1. **核心特性**：
    
    - 多模態對比學習，使圖像與文本共享特徵空間。
    - 支援零樣本學習，直接應用於新類別。
2. **應用場景**：
    
    - 圖像分類、圖像檢索、文本生成。
3. **優勢**：
    
    - 利用大規模未標註數據進行訓練，通用性強。
    - 高效對齊圖像與文本，支援多模態任務。
4. **挑戰**：
    
    - 訓練依賴大規模數據和計算資源。
    - 對於多樣化或細粒度的類別，可能需要進一步微調。


### **SAM (Segment Anything Model)**

![[Pasted image 20250127231347.png]]


###### SAM (Segment Anything Model)

#### **設計特點**

Segment Anything Model (SAM) 是 Meta AI 提出的先進圖像分割模型，專注於**開放範圍分割任務**。SAM 能夠在沒有預先標註的情況下，根據多種提示（例如點、框、文本）生成高質量的分割結果。以下是其設計特點：

1. **多模態提示 (Multi-Modal Prompting)**：
    
    - SAM 支援點 (points)、邊框 (bounding boxes)、文本描述 (text descriptions) 作為輸入提示。
2. **對象不可知性 (Object Agnostic)**：
    
    - SAM 能對任意類別的對象進行分割，而不依賴於特定的預訓練類別。
3. **基於 Vision Transformer (ViT)**：
    
    - 使用強大的 Vision Transformer 提取全局特徵，支持精細的分割任務。
4. **快速推理 (Fast Inference)**：
    
    - SAM 能在單次推理中生成高質量的分割結果，適合即時應用。
5. **大規模訓練 (Massive Training)**：
    
    - 使用超過 11 億個分割標註進行訓練，具備強大的泛化能力。

---

#### **模型架構 (Architecture)**

SAM 的架構主要分為以下幾個部分：

1. **圖像編碼器 (Image Encoder)**：
    
    - 使用 ViT 提取輸入圖像的全局特徵。
    - 輸入大小：圖像 `(B, 3, H, W)`，例如 `(1, 3, 1024, 1024)`。
    - 輸出大小：特徵圖 `(B, N, D)`，例如 `(1, 196, 768)`。
2. **提示編碼器 (Prompt Encoder)**：
    
    - 根據提示類型（點、框或文本），生成對應的嵌入。
    - 輸入大小取決於提示類型，例如點提示的大小為 `(B, num_points, 2)`。
    - 輸出大小：提示嵌入 `(B, P, D)`。
3. **掩碼解碼器 (Mask Decoder)**：
    
    - 將圖像特徵和提示嵌入進行融合，生成分割掩碼。
    - 輸入大小：圖像特徵 `(B, N, D)` 和提示嵌入 `(B, P, D)`。
    - 輸出大小：分割掩碼 `(B, H, W)`。
4. **損失函數**：
    
    - SAM 使用對象級別和像素級別的損失，確保模型能處理細粒度分割。

---

##### **主要 Block 架構**

1. **Image Encoder Block**：
    
    - 基於 Vision Transformer，使用多頭自注意力機制 (Multi-Head Self-Attention) 和前向全連接層 (Feed-Forward Network, FFN) 提取全局特徵。
2. **Prompt Encoder Block**：
    
    - 包括：
        - **點嵌入層 (Point Embedding Layer)**：將坐標嵌入到高維空間。
        - **框嵌入層 (Box Embedding Layer)**：將框的位置信息嵌入為特徵向量。
3. **Mask Decoder Block**：
    
    - 使用 Transformer 解碼器，結合圖像特徵和提示嵌入生成分割掩碼。

---

#### **目標函數**

SAM 的目標函數結合了：

1. **對象級損失 (Object-Level Loss)**：確保對象的分割結果準確。
2. **像素級損失 (Pixel-Level Loss)**：確保分割邊界的精細度。

---

#### **數據流具體步驟**

假設輸入為：

- 圖像大小 `(1, 3, 1024, 1024)`。
- 提示為點，形狀為 `(1, 2)`，表示兩個點的坐標。

**步驟**：

1. **圖像特徵提取 (Image Encoder)**：
    
    - 輸入圖像 `(1, 3, 1024, 1024)`。
    - 分割為 `16×16` 塊，經過 ViT 提取特徵，輸出特徵大小為 `(1, 196, 768)`。
2. **提示特徵生成 (Prompt Encoder)**：
    
    - 將點 `(1, 2)` 經過嵌入層，轉為提示嵌入 `(1, 2, 768)`。
3. **分割掩碼生成 (Mask Decoder)**：
    
    - 結合圖像特徵 `(1, 196, 768)` 和提示嵌入 `(1, 2, 768)`。
    - 使用 Transformer 解碼器生成分割掩碼 `(1, 1024, 1024)`。
4. **輸出**：
    
    - 高分辨率分割掩碼，形狀為 `(1, 1024, 1024)`。

---

#### **PyTorch Code**

以下是 SAM 的核心結構：
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class MLPBlock(nn.Module):
    def __init__(self, embedding_dim: int, mlp_dim: int, act=nn.GELU):
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embedding_dim, embedding_dim * 3)
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attn = AttentionLayer(embedding_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLPBlock(embedding_dim, mlp_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_channels: int = 3,
        embedding_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2, embedding_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_dim=int(embedding_dim * mlp_ratio),
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        self.neck = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1),
            LayerNorm2d(embedding_dim),
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, padding=1),
        )

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        x = self.neck(x)
        return x

class PromptEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 256,
        point_embedding_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.point_embed = nn.Sequential(
            nn.Linear(2, point_embedding_dim),
            nn.GELU(),
            nn.Linear(point_embedding_dim, embedding_dim),
        )
        
        self.box_embed = nn.Sequential(
            nn.Linear(4, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        
        self.not_a_point_embed = nn.Parameter(torch.randn(1, embedding_dim))

    def forward(
        self,
        points: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        point_embeddings = self.not_a_point_embed.expand(points.shape[0], -1) if points is None \
                         else self.point_embed(points)
        
        if boxes is not None:
            box_embeddings = self.box_embed(boxes)
            return torch.cat([point_embeddings, box_embeddings], dim=1)
        
        return point_embeddings

class MaskDecoder(nn.Module):
    def __init__(
        self,
        transformer_dim: int = 256,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        
        self.transformer_dim = transformer_dim
        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Parameter(torch.zeros(1, 1, transformer_dim))
        self.mask_tokens = nn.Parameter(torch.zeros(1, num_multimask_outputs, transformer_dim))
        
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        
        self.output_hypernetworks_mlps = nn.ModuleList([
            MLPBlock(transformer_dim, transformer_dim, activation)
            for i in range(num_multimask_outputs)
        ])

        self.iou_prediction_head = MLPBlock(transformer_dim, 1, activation)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Upscale image embeddings
        image_embeddings = self.output_upscaling(image_embeddings)
        
        # Generate masks
        masks = []
        iou_pred = self.iou_prediction_head(prompt_embeddings)
        
        for i in range(self.num_multimask_outputs):
            mask_embedding = self.mask_tokens[:, i:i+1]
            hyper_in = prompt_embeddings + mask_embedding
            hyper_in = self.output_hypernetworks_mlps[i](hyper_in)
            mask = (hyper_in @ image_embeddings.flatten(2)).reshape(
                image_embeddings.shape[0], 1, *image_embeddings.shape[-2:]
            )
            masks.append(mask)
        
        masks = torch.cat(masks, dim=1)
        return masks, iou_pred

class SAM(nn.Module):
    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder

    def forward(
        self,
        images: torch.Tensor,
        points: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image_embeddings = self.image_encoder(images)
        prompt_embeddings = self.prompt_encoder(points, boxes)
        masks, iou_predictions = self.mask_decoder(image_embeddings, prompt_embeddings)
        return masks, iou_predictions

def create_sam_model(model_type: str = "base"):
    """Create a SAM model with specified configuration"""
    configs = {
        "base": dict(
            embedding_dim=768,
            num_heads=12,
            depth=12,
        ),
        "large": dict(
            embedding_dim=1024,
            num_heads=16,
            depth=24,
        ),
    }
    
    if model_type not in configs:
        raise ValueError(f"Model type {model_type} not found. Available types: {list(configs.keys())}")
    
    config = configs[model_type]
    
    image_encoder = ImageEncoderViT(
        embedding_dim=config["embedding_dim"],
        num_heads=config["num_heads"],
        depth=config["depth"],
    )
    
    prompt_encoder = PromptEncoder(
        embedding_dim=config["embedding_dim"] // 3,
    )
    
    mask_decoder = MaskDecoder(
        transformer_dim=config["embedding_dim"] // 3,
    )
    
    return SAM(image_encoder, prompt_encoder, mask_decoder)

# 測試代碼
if __name__ == "__main__":
    # 創建模型
    model = create_sam_model("base")
    
    # 測試輸入
    batch_size = 1
    images = torch.randn(batch_size, 3, 1024, 1024)
    points = torch.randn(batch_size, 2)
    boxes = torch.randn(batch_size, 4)
    
    # 前向傳播
    masks, iou_predictions = model(images, points, boxes)
    
    # 打印輸出形狀
    print(f"Input image shape: {images.shape}")
    print(f"Output masks shape: {masks.shape}")
    print(f"IoU predictions shape: {iou_predictions.shape}")
    
    # 計算參數量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")


# 創建模型
model = create_sam_model("base")

# 準備輸入
images = torch.randn(1, 3, 1024, 1024)
points = torch.randn(1, 2)  # 可選
boxes = torch.randn(1, 4)   # 可選

# 生成遮罩
masks, iou_predictions = model(images, points, boxes)


```

---

#### **重要特性總結**

1. **核心特性**：
    
    - 支援多模態提示 (點、框、文本)。
    - 高效處理任意類別的分割任務。
2. **應用場景**：
    
    - 實時物體分割、醫學圖像分析、多模態檢索。
3. **優勢**：
    
    - 不受標註數據限制，泛化能力強。
    - 支援高分辨率輸入和即時推理。
4. **挑戰**：
    
    - 訓練依賴於大規模數據和高性能計算資源。
    - 分割效果對提示準確性敏感。


### Segment Anything Model 2（SAM 2)

![[Pasted image 20250127231625.png]]

###### Segment Anything Model 2（SAM 2)

Segment Anything Model 2（SAM 2）是由 Meta 公司发布的先进图像和视频分割模型，是对原始 SAM 模型的升级版本。SAM 2 引入了统一的模型架构，能够在图像和视频中实现实时的提示对象分割，并达到最先进的性能。


### 设计特性

1. **统一模型架构（Unified Model Architecture）**：
    
    - SAM 2 将图像和视频的分割能力整合到单一模型中，简化了部署流程，并在不同媒体类型间提供一致的性能表现。
2. **实时性能（Real-Time Performance）**：
    
    - 模型达到了每秒约 44 帧的推理速度，适用于需要实时反馈的应用场景，如视频编辑和增强现实。
3. **零样本泛化（Zero-Shot Generalization）**：
    
    - SAM 2 能够分割从未见过的对象，展示出强大的零样本泛化能力，特别适用于多样化或不断变化的视觉领域。
4. **可提示的模型架构（Promptable Model Architecture）**：
    
    - SAM 2 继承了 SAM 的特性，可以根据不同的提示（如点、框、甚至是文本）来生成分割结果。

### 模型架构（Architecture）

SAM 2 的架构主要包括以下组件：

1. **图像和视频编码器（Image and Video Encoder）**：
    
    - 使用基于 Transformer 的架构，从图像和视频帧中提取高阶特征，理解每个时间点的视觉内容。
2. **提示编码器（Prompt Encoder）**：
    
    - 处理用户提供的提示（如点、框、遮罩），以引导分割任务，使模型能够适应用户输入并针对场景中的特定对象。
3. **遮罩解码器（Mask Decoder）**：
    
    - 根据编码的图像特征和提示生成最终的分割遮罩。在视频中，它还使用记忆上下文以确保跨帧的精确追踪。

### 主要模块架构（Block Architecture）

1. **图像和视频编码器（Image and Video Encoder）**：
    
    - 基于 Transformer 的架构，使用多头自注意力机制（Multi-Head Self-Attention）和前向全连接层（Feed-Forward Network, FFN）来提取全局特征。
2. **提示编码器（Prompt Encoder）**：
    
    - 包括：
        - **点嵌入层（Point Embedding Layer）**：将坐标嵌入到高维空间。
        - **框嵌入层（Box Embedding Layer）**：将框的位置信息嵌入为特征向量。
3. **遮罩解码器（Mask Decoder）**：
    
    - 使用 Transformer 解码器，结合图像特征和提示嵌入生成分割遮罩。

### 输入输出

- **输入（Input）**：
    
    - 图像或视频帧，大小为 `(B, 3, H, W)`，例如 `(1, 3, 1024, 1024)`。
    - 提示，如点 `(B, num_points, 2)` 或框 `(B, num_boxes, 4)`。
- **输出（Output）**：
    
    - 分割遮罩，大小为 `(B, H, W)`，例如 `(1, 1024, 1024)`。

### 目标函数

SAM 2 的目标函数结合了对象级别和像素级别的损失，确保模型能处理细粒度分割。

### 数据流具体步骤

假设输入为：

- **图像**：大小为 `(1, 3, 1024, 1024)`。
- **提示**：点，形状为 `(1, 2)`，表示两个点的坐标。

**步骤**：

1. **图像特征提取（Image Feature Extraction）**：
    
    - 输入图像经过图像编码器，提取特征，输出大小为 `(1, N, D)`，例如 `(1, 196, 768)`。
2. **提示特征生成（Prompt Feature Generation）**：
    
    - 将点坐标嵌入到高维空间，生成提示特征，大小为 `(1, 2, 768)`。
3. **分割遮罩生成（Mask Generation）**：
    
    - 结合图像特征和提示特征，使用遮罩解码器生成分割遮罩，输出大小为 `(1, 1024, 1024)`。
4. **输出（Output）**：
    
    - 高分辨率分割遮罩，形状为 `(1, 1024, 1024)`。

### PyTorch 代码

以下是 SAM 2 的核心结构：
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class ImageEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, embed_dim: int = 256, depths: list = [2, 2, 6, 2]):
        super().__init__()
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim//8, kernel_size=7, stride=4, padding=3),
            LayerNorm2d(embed_dim//8),
            nn.GELU(),
            nn.Conv2d(embed_dim//8, embed_dim//4, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(embed_dim//4),
            nn.GELU(),
            nn.Conv2d(embed_dim//4, embed_dim//2, kernel_size=3, stride=2, padding=1),
            LayerNorm2d(embed_dim//2),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim, kernel_size=3, stride=1, padding=1),
        )
        
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, 64, 64))
        self.pos_drop = nn.Dropout(p=0.1)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        return x

class MemoryTransformer(nn.Module):
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.memory_tokens = nn.Parameter(torch.zeros(1, 64, embed_dim))
        self.memory_pos_embed = nn.Parameter(torch.zeros(1, 64, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        memory = self.memory_tokens.expand(B, -1, -1)
        memory = memory + self.memory_pos_embed
        
        # Reshape spatial dimensions to sequence
        h, w = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        
        # Concatenate memory tokens with input
        x = torch.cat([memory, x], dim=1)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Split memory tokens and spatial tokens
        memory, spatial = x[:, :64], x[:, 64:]
        spatial = spatial.transpose(1, 2).reshape(B, -1, h, w)
        
        return memory, spatial

class MaskDecoder(nn.Module):
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim//2, kernel_size=2, stride=2),
            LayerNorm2d(embed_dim//2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim//2, embed_dim//4, kernel_size=2, stride=2),
            LayerNorm2d(embed_dim//4),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim//4, embed_dim//8, kernel_size=2, stride=2),
            LayerNorm2d(embed_dim//8),
            nn.GELU(),
            nn.Conv2d(embed_dim//8, 1, kernel_size=1)
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, memory: torch.Tensor, spatial: torch.Tensor) -> torch.Tensor:
        # Use memory to condition spatial features
        B, N, C = memory.shape
        memory = memory.mean(dim=1).view(B, C, 1, 1)
        x = spatial * memory
        
        # Decode to mask
        mask = self.decoder(x)
        return mask

class SAM2(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        in_channels: int = 3,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(in_channels, embed_dim)
        self.memory_transformer = MemoryTransformer(embed_dim, num_heads, num_layers, dropout)
        self.mask_decoder = MaskDecoder(embed_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        prompts: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode image
        features = self.image_encoder(x)
        
        # Process with memory transformer
        memory, spatial = self.memory_transformer(features)
        
        # Decode mask
        mask = self.mask_decoder(memory, spatial)
        
        # Resize mask to input resolution
        mask = F.interpolate(mask, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        return mask, memory

def create_sam2_model(model_type: str = "base") -> SAM2:
    """Create a SAM2 model with specified configuration"""
    configs = {
        "base": dict(
            embed_dim=256,
            num_heads=8,
            num_layers=2,
        ),
        "large": dict(
            embed_dim=512,
            num_heads=16,
            num_layers=4,
        ),
    }
    
    if model_type not in configs:
        raise ValueError(f"Model type {model_type} not found. Available types: {list(configs.keys())}")
    
    return SAM2(**configs[model_type])

# 測試代碼
if __name__ == "__main__":
    # 創建模型
    model = create_sam2_model("base")
    
    # 測試輸入
    batch_size = 2
    x = torch.randn(batch_size, 3, 1024, 1024)
    
    # 前向傳播
    mask, memory = model(x)
    
    # 打印輸出形狀
    print(f"Input shape: {x.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Memory shape: {memory.shape}")
    
    # 計算參數量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")


# 創建模型
model = create_sam2_model("base")

# 準備輸入
x = torch.randn(1, 3, 1024, 1024)

# 生成遮罩
mask, memory = model(x)


```



### **Stable Diffusion + ControlNet**

![[Pasted image 20250127232140.png]]

###### Stable Diffusion + ControlNet

#### **設計特點**

1. **Stable Diffusion**：
    
    - 是一種基於擴散模型（Diffusion Model）的生成模型，能從噪聲生成高質量的圖像。
    - 使用潛在擴散（Latent Diffusion）的方法，通過壓縮高維圖像至潛在空間，大幅提升運算效率。
    - 主要作用是從文本（Text-to-Image）或其他條件生成圖像。
2. **ControlNet**：
    
    - ControlNet 是一種架構擴展，能將外部條件（如邊緣檢測、姿態圖等）融入到 Stable Diffusion 的生成過程中。
    - 它通過多層控制權重，保證模型在生成過程中能夠精確遵循外部條件。
3. **結合特性**：
    
    - Stable Diffusion 提供強大的生成能力，ControlNet 提供精確控制條件的能力。
    - 支援多種外部條件，如 Canny 邊緣檢測、OpenPose 姿態信息、深度圖等。

---

#### **架構（Architecture）**

Stable Diffusion 和 ControlNet 的結合主要擴展於 U-Net 編碼器，具體架構如下：

1. **Stable Diffusion 基本架構**：
    
    - **文本編碼器（Text Encoder）**：
        - 將輸入文本轉換為潛在表示，用於指導圖像生成。
        - 通常使用 CLIP 文本編碼器。
    - **U-Net**：
        - 潛在擴散過程的核心網絡，負責對潛在特徵進行去噪。
    - **VAE（Variational Autoencoder）**：
        - **編碼器（Encoder）**：將輸入圖像轉換為潛在空間。
        - **解碼器（Decoder）**：將潛在特徵還原為圖像。
2. **ControlNet 擴展**：
    
    - ControlNet 將外部條件引入 Stable Diffusion 的 U-Net，實現對生成過程的精確控制。
    - **條件分支（Conditional Branch）**：
        - 接收外部條件（例如 Canny 邊緣、姿態），通過多層卷積網絡提取特徵。
    - **特徵融合（Feature Fusion）**：
        - 將條件分支特徵與 Stable Diffusion 的內部特徵進行融合，並通過殘差連接保持原始生成能力。

---

#### **Block 架構**

1. **Stable Diffusion 的 U-Net**：
    
    - **編碼器（Encoder Blocks）**：
        - 使用多層卷積和殘差塊（Residual Blocks）提取潛在特徵。
    - **中間層（Middle Block）**：
        - 通過自注意力機制（Self-Attention）和卷積操作捕捉全局上下文信息。
    - **解碼器（Decoder Blocks）**：
        - 通過對稱結構解碼潛在特徵，逐步還原圖像。
2. **ControlNet**：
    
    - **條件處理分支（Condition Branch）**：
        - 通過特徵提取層（多層卷積和激活函數）處理外部條件。
    - **融合層（Fusion Layers）**：
        - 使用殘差連接（Residual Connections）將條件特徵融合進 U-Net 的中間層。

---

#### **輸入輸出**

1. **輸入（Input）**：
    
    - 文本提示（Text Prompt）：大小 `(batch_size, max_seq_len)`。
    - 外部條件（如邊緣檢測圖、姿態圖）：大小 `(batch_size, channels, H, W)`。
    - 噪聲圖像：大小 `(batch_size, channels, H, W)`，例如 `(1, 3, 512, 512)`。
2. **輸出（Output）**：
    
    - 最終生成的圖像：大小與輸入噪聲圖一致，例如 `(1, 3, 512, 512)`。

---

#### **目標函數**

Stable Diffusion 使用擴散模型的損失函數，ControlNet 增加了條件約束：

1. **擴散損失（Diffusion Loss）**：通過逐步去噪學習分布。
2. **條件損失（Condition Loss）**：保證生成結果符合外部條件。

---

#### **數據流具體步驟**

假設輸入為：

- 文本提示：`"A futuristic cityscape"`。
- Canny 邊緣圖：大小 `(1, 1, 512, 512)`。
- 噪聲圖像：大小 `(1, 3, 512, 512)`。

1. **文本處理**：
    
    - 文本編碼器提取文本特徵，生成大小為 `(1, 77, 768)` 的嵌入向量。
2. **外部條件處理（ControlNet 分支）**：
    
    - Canny 邊緣圖通過卷積網絡提取條件特徵，大小為 `(1, 128, 64, 64)`。
3. **潛在特徵處理（Stable Diffusion U-Net）**：
    
    - 噪聲圖經過 U-Net 編碼器提取特徵，融合條件特徵，生成潛在空間特徵。
4. **解碼與生成**：
    
    - U-Net 解碼器結合條件特徵還原圖像，輸出大小為 `(1, 3, 512, 512)`。

---

#### **PyTorch 代码**

以下是 Stable Diffusion 和 ControlNet 的結合示例：
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import math

class SinusoidalPositionEmbeddings(nn.Module):
    """時間步長的位置編碼"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResnetBlock(nn.Module):
    """Resnet 塊"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act = nn.SiLU()
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        
        return x + identity

class AttentionBlock(nn.Module):
    """自注意力塊"""
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        q = q.view(B, self.num_heads, C // self.num_heads, H * W)
        k = k.view(B, self.num_heads, C // self.num_heads, H * W)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W)
        
        scale = (C // self.num_heads) ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).view(B, C, H, W)
        return x + self.proj(x)

class UNet(nn.Module):
    """改進的 U-Net 架構"""
    def __init__(
        self,
        in_channels: int = 3,
        model_channels: int = 128,
        out_channels: int = 3,
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int] = (16, 8),
        dropout: float = 0.0,
        channel_mult: Tuple[int] = (1, 2, 4, 8),
        conv_resample: bool = True,
        num_heads: int = 8
    ):
        super().__init__()
        
        self.time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )
        
        # 輸入投影
        self.input_blocks = nn.ModuleList([
            nn.Conv2d(in_channels, model_channels, 3, padding=1)
        ])
        
        # 下採樣塊
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResnetBlock(ch, mult * model_channels)]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                self.input_blocks.append(nn.Sequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    nn.Conv2d(ch, ch, 3, stride=2, padding=1)
                )
                input_block_chans.append(ch)
                ds *= 2
        
        # 中間塊
        self.middle_block = nn.Sequential(
            ResnetBlock(ch, ch),
            AttentionBlock(ch, num_heads=num_heads),
            ResnetBlock(ch, ch)
        )
        
        # 上採樣塊
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResnetBlock(
                        ch + input_block_chans.pop(),
                        model_channels * mult
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                if level and i == num_res_blocks:
                    layers.append(
                        nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)
                    )
                    ds //= 2
                self.output_blocks.append(nn.Sequential(*layers))
        
        # 輸出投影
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1)
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        # 時間編碼
        emb = self.time_embed(timesteps)
        
        # 下採樣路徑
        h = x
        hs = []
        for module in self.input_blocks:
            h = module(h)
            hs.append(h)
        
        # 中間處理
        h = self.middle_block(h)
        
        # 上採樣路徑
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h)
            
        return self.out(h)

class ControlNet(nn.Module):
    """改進的 ControlNet 架構"""
    def __init__(
        self,
        in_channels: int = 3,
        model_channels: int = 128,
        control_channels: int = 64,
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int] = (16, 8),
        dropout: float = 0.0,
        channel_mult: Tuple[int] = (1, 2, 4, 8),
        conv_resample: bool = True,
        num_heads: int = 8
    ):
        super().__init__()
        
        self.control_encoder = nn.Sequential(
            nn.Conv2d(in_channels, control_channels, 3, padding=1),
            nn.GroupNorm(32, control_channels),
            nn.SiLU(),
            nn.Conv2d(control_channels, control_channels, 3, padding=1),
            nn.GroupNorm(32, control_channels),
            nn.SiLU(),
            nn.Conv2d(control_channels, model_channels, 3, padding=1)
        )
        
        self.control_unet = UNet(
            in_channels=model_channels,
            model_channels=model_channels,
            out_channels=model_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            num_heads=num_heads
        )

    def forward(self, x: torch.Tensor, control: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        control_features = self.control_encoder(control)
        return self.control_unet(x + control_features, timesteps)

class StableDiffusionControlNet(nn.Module):
    """Stable Diffusion 與 ControlNet 的整合"""
    def __init__(
        self,
        unet_channels: int = 128,
        control_channels: int = 64,
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int] = (16, 8),
        channel_mult: Tuple[int] = (1, 2, 4, 8),
        num_heads: int = 8
    ):
        super().__init__()
        
        self.unet = UNet(
            model_channels=unet_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            channel_mult=channel_mult,
            num_heads=num_heads
        )
        
        self.controlnet = ControlNet(
            model_channels=unet_channels,
            control_channels=control_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            channel_mult=channel_mult,
            num_heads=num_heads
        )

    def forward(
        self,
        x: torch.Tensor,
        control: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x)
            
        # 控制網絡的輸出
        control_output = self.controlnet(x, control, timesteps)
        
        # 主要 UNet 的輸出
        unet_output = self.unet(x + control_output, timesteps)
        
        return unet_output

def create_stable_diffusion_controlnet(model_type: str = "base"):
    """創建 Stable Diffusion ControlNet 模型"""
    configs = {
        "base": dict(
            unet_channels=128,
            control_channels=64,
            num_res_blocks=2,
            attention_resolutions=(16, 8),
            channel_mult=(1, 2, 4, 8),
            num_heads=8
        ),
        "large": dict(
            unet_channels=256,
            control_channels=128,
            num_res_blocks=3,
            attention_resolutions=(16, 8, 4),
            channel_mult=(1, 2, 4, 8, 16),
            num_heads=16
        )
    }
    
    if model_type not in configs:
        raise ValueError(f"Model type {model_type} not found. Available types: {list(configs.keys())}")
    
    return StableDiffusionControlNet(**configs[model_type])

# 測試代碼
if __name__ == "__main__":
    # 創建模型
    model = create_stable_diffusion_controlnet("base")
    
    # 測試輸入
    batch_size = 2
    x = torch.randn(batch_size, 3, 256, 256)
    control = torch.randn(batch_size, 3, 256, 256)
    timesteps = torch.randint(0, 1000, (batch_size,))
    
    # 前向傳播
    output = model(x, control, timesteps)
    
    # 打印輸出形狀
    print(f"Input shape: {x.shape}")
    print(f"Control shape: {control.shape}")
    print(f"Output shape: {output.shape}")
    
    # 計算參數量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

# 創建模型
model = create_stable_diffusion_controlnet("base")

# 準備輸入
x = torch.randn(1, 3, 256, 256)
control = torch.randn(1, 3, 256, 256)
timesteps = torch.randint(0, 1000, (1,))

# 生成圖像
output = model(x, control, timesteps)


```

---

#### **重要特性總結**

1. **核心特性**：
    
    - **Stable Diffusion**：高效的生成能力，擴散過程。
    - **ControlNet**：將外部條件融入生成過程，提升控制精度。
2. **應用場景**：
    
    - 文本到圖像生成。
    - 依據條件（如邊緣、深度圖）生成特定風格圖像。
3. **優勢**：
    
    - 高生成質量。
    - 可控性強，支持多種外部條件。