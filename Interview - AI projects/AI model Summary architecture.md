
 ### **Generalized R-CNN**

Generalized R-CNN 是一種基於兩階段目標檢測框架的通用模型。該模型的設計是為了靈活處理多種目標檢測、分割和其他相關任務，如 Faster R-CNN 和 Mask R-CNN 就是 Generalized R-CNN 的具體實現。以下將詳細解釋該模型的設計特點、架構、Block 結構、輸入輸出、目標函數、作用及重要特性，並以具體案例和 PyTorch 代碼示例進行說明。

![[Pasted image 20250113143208.png]]

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

以下是 Generalized R-CNN 的 PyTorch 實現：
```python
import torch
import torch.nn as nn
from torchvision.ops import RoIAlign

# Backbone 模塊
class Backbone(nn.Module):
    def __init__(self, out_channels=256):
        super(Backbone, self).__init__()
        # 模擬一個簡單的卷積網絡作為 Backbone
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(64, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        features = self.layer1(x)
        return features  # 輸出特徵圖

# RPN 模塊
class RPN(nn.Module):
    def __init__(self, in_channels=256, anchor_num=9):
        super(RPN, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.cls_layer = nn.Conv2d(in_channels, anchor_num, kernel_size=1)  # 分類分支
        self.reg_layer = nn.Conv2d(in_channels, anchor_num * 4, kernel_size=1)  # 回歸分支
        self.relu = nn.ReLU(inplace=True)

    def forward(self, features):
        x = self.relu(self.conv(features))
        objectness = self.cls_layer(x)  # 錨框是否包含目標
        bbox_reg = self.reg_layer(x)  # 錨框的位置調整
        return objectness, bbox_reg

# RoI Heads 模塊
class RoIHeads(nn.Module):
    def __init__(self, num_classes=21, in_channels=256, roi_size=(7, 7)):
        super(RoIHeads, self).__init__()
        self.roi_align = RoIAlign(roi_size, spatial_scale=1.0, sampling_ratio=2)
        self.fc1 = nn.Linear(in_channels * roi_size[0] * roi_size[1], 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.cls_score = nn.Linear(1024, num_classes)  # 類別預測
        self.bbox_pred = nn.Linear(1024, num_classes * 4)  # 邊界框回歸

    def forward(self, features, proposals, image_shapes):
        # 將提案框映射到固定大小
        pooled_features = self.roi_align(features, proposals, image_shapes)
        pooled_features = pooled_features.flatten(start_dim=1)  # 拉平成全連接輸入
        fc_out = self.fc2(self.fc1(pooled_features))
        cls_scores = self.cls_score(fc_out)  # 類別預測
        bbox_deltas = self.bbox_pred(fc_out)  # 邊界框調整
        return cls_scores, bbox_deltas

# Generalized R-CNN
class GeneralizedRCNN(nn.Module):
    def __init__(self, num_classes=21):
        super(GeneralizedRCNN, self).__init__()
        self.backbone = Backbone(out_channels=256)
        self.rpn = RPN(in_channels=256, anchor_num=9)
        self.roi_heads = RoIHeads(num_classes=num_classes, in_channels=256)

    def forward(self, images, targets=None):
        # 1. Backbone 提取特徵
        features = self.backbone(images)

        # 2. RPN 生成候選框
        objectness, bbox_reg = self.rpn(features)

        # 模擬候選框輸出（通常需要 NMS 和生成候選框）
        proposals = torch.rand((len(images), 100, 4))  # 模擬 100 個候選框
        image_shapes = [(images.size(2), images.size(3))] * len(images)

        # 3. RoI Heads 提取特徵，進行分類與回歸
        cls_scores, bbox_deltas = self.roi_heads(features, proposals, image_shapes)

        return {"cls_scores": cls_scores, "bbox_deltas": bbox_deltas, "proposals": proposals}


```



### **Mask R-CNN**

Mask R-CNN 是基於 **Faster R-CNN** 的一種拓展，用於進行目標檢測和實例分割（Instance Segmentation）。它在 Faster R-CNN 的基礎上增加了像素級的分割分支，因此不僅可以輸出每個目標的邊界框（Bounding Box），還能生成高分辨率的分割掩碼（Segmentation Mask）。

![[Pasted image 20250113143613.png]]


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
from torchvision.ops import RoIAlign

# Backbone 模塊
class Backbone(nn.Module):
    def __init__(self, out_channels=256):
        super(Backbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(64, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        features = self.layer1(x)
        return features  # 輸出特徵圖

# RPN 模塊
class RPN(nn.Module):
    def __init__(self, in_channels=256, anchor_num=9):
        super(RPN, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.cls_layer = nn.Conv2d(in_channels, anchor_num, kernel_size=1)  # 分類分支
        self.reg_layer = nn.Conv2d(in_channels, anchor_num * 4, kernel_size=1)  # 回歸分支
        self.relu = nn.ReLU(inplace=True)

    def forward(self, features):
        x = self.relu(self.conv(features))
        objectness = self.cls_layer(x)  # 錨框是否包含目標
        bbox_reg = self.reg_layer(x)  # 錨框的位置調整
        return objectness, bbox_reg

# RoI Heads 模塊
class RoIHeads(nn.Module):
    def __init__(self, num_classes=21, in_channels=256, roi_size=(7, 7)):
        super(RoIHeads, self).__init__()
        self.roi_align = RoIAlign(roi_size, spatial_scale=1.0, sampling_ratio=2)
        self.fc1 = nn.Linear(in_channels * roi_size[0] * roi_size[1], 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.cls_score = nn.Linear(1024, num_classes)  # 類別預測
        self.bbox_pred = nn.Linear(1024, num_classes * 4)  # 邊界框回歸

    def forward(self, features, proposals, image_shapes):
        pooled_features = self.roi_align(features, proposals, image_shapes)
        pooled_features = pooled_features.flatten(start_dim=1)  # 拉平成全連接輸入
        fc_out = self.fc2(self.fc1(pooled_features))
        cls_scores = self.cls_score(fc_out)  # 類別預測
        bbox_deltas = self.bbox_pred(fc_out)  # 邊界框調整
        return cls_scores, bbox_deltas

# Mask 分支
class MaskBranch(nn.Module):
    def __init__(self, in_channels=256, num_classes=21):
        super(MaskBranch, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.mask_pred = nn.Conv2d(256, num_classes, kernel_size=1)  # 每個類別生成一個掩碼

    def forward(self, features, proposals, image_shapes):
        pooled_features = RoIAlign((14, 14), spatial_scale=1.0, sampling_ratio=2)(features, proposals, image_shapes)
        x = self.conv1(pooled_features)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        masks = self.mask_pred(x)  # 每個候選框的掩碼
        return masks

# Mask R-CNN 模型
class MaskRCNN(nn.Module):
    def __init__(self, num_classes=21):
        super(MaskRCNN, self).__init__()
        self.backbone = Backbone(out_channels=256)
        self.rpn = RPN(in_channels=256, anchor_num=9)
        self.roi_heads = RoIHeads(num_classes=num_classes, in_channels=256)
        self.mask_branch = MaskBranch(in_channels=256, num_classes=num_classes)

    def forward(self, images, targets=None):
        # 1. Backbone 提取特徵
        features = self.backbone(images)

        # 2. RPN 生成候選框
        objectness, bbox_reg = self.rpn(features)

        # 模擬候選框輸出（通常需要 NMS 和生成候選框）
        proposals = torch.rand((len(images), 100, 4))  # 模擬 100 個候選框
        image_shapes = [(images.size(2), images.size(3))] * len(images)

        # 3. RoI Heads 進行分類與邊界框調整
        cls_scores, bbox_deltas = self.roi_heads(features, proposals, image_shapes)

        # 4. Mask 分支生成分割掩碼
        masks = self.mask_branch(features, proposals, image_shapes)

        return {
            "cls_scores": cls_scores,
            "bbox_deltas": bbox_deltas,
            "masks": masks,
            "proposals": proposals,
        }


```





### **YOLOv7**

**YOLOv7** 是 YOLO 系列模型的最新版本之一，專注於**目標檢測（Object Detection）**。YOLOv7 進一步提升了精度和速度的平衡，通過優化模型結構、訓練策略及引入新技術，在多個基準數據集上達到 SOTA（State-Of-The-Art）性能。

以下是對 **YOLOv7** 的詳細中文解釋，包括設計特點、架構、Block 結構、輸入輸出、目標函數、作用及重要特性，並附帶 PyTorch 實現代碼（模型結構部分）。

![[Pasted image 20250113144434.png]]

---

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

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CSPBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels // 2, 1, 1, 0)
        self.conv2 = ConvBlock(in_channels, out_channels // 2, 1, 1, 0)
        self.conv3 = ConvBlock(out_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y = torch.cat([y1, y2], dim=1)
        return self.conv3(y)

class YOLOv7Backbone(nn.Module):
    def __init__(self, num_classes=80):
        super(YOLOv7Backbone, self).__init__()
        self.stem = ConvBlock(3, 32, 3, 1, 1)
        self.csp1 = CSPBlock(32, 64)
        self.csp2 = CSPBlock(64, 128)
        self.spp = nn.AdaptiveAvgPool2d((1, 1))  # 模擬 SPP 模塊

    def forward(self, x):
        x = self.stem(x)
        x = self.csp1(x)
        x = self.csp2(x)
        x = self.spp(x)
        return x

# 測試 YOLOv7 Backbone
model = YOLOv7Backbone()
dummy_input = torch.randn(1, 3, 640, 640)
output = model(dummy_input)
print("輸出大小：", output.shape)  # 應該輸出 (1, 128, 1, 1)

```



![[Pasted image 20250113145420.png]]

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
from torchvision.ops import RoIAlign

# Backbone 模塊
class Backbone(nn.Module):
    def __init__(self, out_channels=256):
        super(Backbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(64, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        features = self.layer1(x)
        return features

# Detection Head 模塊
class DetectionHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=80):
        super(DetectionHead, self).__init__()
        self.cls_conv = nn.Conv2d(in_channels, num_classes, kernel_size=3, stride=1, padding=1)
        self.reg_conv = nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1)
        self.centerness_conv = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        cls_logits = self.cls_conv(x)
        bbox_preds = self.reg_conv(x)
        centerness = self.centerness_conv(x)
        return cls_logits, bbox_preds, centerness

# Mask 分支
class MaskBranch(nn.Module):
    def __init__(self, in_channels=256, num_classes=80):
        super(MaskBranch, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1)
        self.mask_pred = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, features, proposals, image_shapes):
        pooled_features = RoIAlign((14, 14), spatial_scale=1.0, sampling_ratio=2)(features, proposals, image_shapes)
        x = self.conv1(pooled_features)
        masks = self.mask_pred(x)
        return masks

# CenterMask2 模型
class CenterMask2(nn.Module):
    def __init__(self, num_classes=80):
        super(CenterMask2, self).__init__()
        self.backbone = Backbone()
        self.detection_head = DetectionHead(num_classes=num_classes)
        self.mask_branch = MaskBranch(num_classes=num_classes)

    def forward(self, images, proposals, image_shapes):
        # 1. Backbone 提取特徵
        features = self.backbone(images)

        # 2. 檢測頭
        cls_logits, bbox_preds, centerness = self.detection_head(features)

        # 3. Mask 分支
        masks = self.mask_branch(features, proposals, image_shapes)

        return {
            "cls_logits": cls_logits,
            "bbox_preds": bbox_preds,
            "centerness": centerness,
            "masks": masks,
        }

```


![[Pasted image 20250113145750.png]]