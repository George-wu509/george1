
### LORA (Low-Rank Adaptation) 微调 ViT 大模型原理详细解析

#### 1. 原理概述

LORA 是一种有效的微调技术，主要用于在大模型（如 ViT）上进行高效的参数微调。其核心思想是通过引入低秩矩阵分解的方式，仅调整模型中部分层的参数（如权重矩阵）的低秩部分，从而减少需要更新的参数量。这种方式在内存占用和计算效率上都有明显优势，尤其适用于资源有限的环境。

假设我们对模型的线性层参数 W∈Rd×kW \in \mathbb{R}^{d \times k}W∈Rd×k 进行微调：

- 不直接更新 WWW，而是将其分解为两个小矩阵的乘积： W′=W+ΔW=W+A⋅BW' = W + \Delta W = W + A \cdot BW′=W+ΔW=W+A⋅B 其中：
    - A∈Rd×rA \in \mathbb{R}^{d \times r}A∈Rd×r：低秩矩阵，表示低维嵌入。
    - B∈Rr×kB \in \mathbb{R}^{r \times k}B∈Rr×k：低秩矩阵，表示投影到原始维度。

通过限制 r≪min⁡(d,k)r \ll \min(d, k)r≪min(d,k)，可以显著减少需要训练的参数量。

#### 2. 公式解析

1. 原始权重矩阵 WWW 不变。
2. 添加一个低秩更新项 ΔW=A⋅B\Delta W = A \cdot BΔW=A⋅B，并只训练 AAA 和 BBB：
    - W′=W+A⋅BW' = W + A \cdot BW′=W+A⋅B
3. 微调时，冻结 WWW，优化 {A,B}\{A, B\}{A,B}。

这种方式的优点是：

- 降低了存储和计算成本。
- 可以保留原始预训练权重 WWW，方便在不同任务间共享。

#### 3. PyTorch 代码实现

以下是一个完整的 PyTorch 实现示例，展示如何对 ViT 模型应用 LORA 微调：


```python
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize

# 定义 LORA 层
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super(LoRALinear, self).__init__()
        self.rank = rank
        self.linear = nn.Linear(in_features, out_features, bias=True)
        
        # 冻结原始权重
        self.linear.weight.requires_grad = False
        self.linear.bias.requires_grad = False
        
        # 添加低秩矩阵 A 和 B
        self.A = nn.Parameter(torch.zeros(in_features, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_features))
        
        # 初始化低秩矩阵
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x):
        return self.linear(x) + x @ self.A @ self.B

# 替换 ViT 模型中的线性层为 LoRA 层
def apply_lora_to_vit(vit_model, rank=4):
    for name, module in vit_model.named_modules():
        if isinstance(module, nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            setattr(
                vit_model,
                name,
                LoRALinear(in_features, out_features, rank)
            )
    return vit_model

# 加载预训练的 ViT 模型
vit_model = models.vit_b_16(pretrained=True)
vit_model = apply_lora_to_vit(vit_model, rank=4)

# CIFAR-10 数据集预处理
transform = Compose([ToTensor(), Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
train_set, val_set = random_split(dataset, [45000, 5000])
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

# 定义训练流程
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, vit_model.parameters()), lr=1e-4)

# 训练 LoRA 微调
def train(model, dataloader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")

train(vit_model, train_loader, criterion, optimizer, epochs=5)

# 验证模型
def validate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Validation Accuracy: {100 * correct / total:.2f}%")

validate(vit_model, val_loader)

```

#### 4. 训练过程中的要点

- **冻结预训练权重：** `self.linear.weight.requires_grad = False` 确保只训练 AAA 和 BBB。
- **优化器：** 仅传递 AAA 和 BBB 的参数给优化器，减少内存和计算成本。
- **低秩初始化：** AAA 和 BBB 使用适当的初始化方法（如 Kaiming 初始化）。

#### 5. 总结

通过 LORA 微调 ViT 大模型，能够以极低的成本适配新的任务，而无需对整个模型进行全面更新。这种方法特别适合资源受限的场景，如移动设备或嵌入式系统。


### **1. 定義 LoRALinear 類**

這部分代碼定義了 LoRA 的核心組件，用於替換模型中的線性層，並實現低秩矩陣更新。

python

複製程式碼

`class LoRALinear(nn.Module):     def __init__(self, in_features, out_features, rank=4):         super(LoRALinear, self).__init__()         self.rank = rank`

- **`LoRALinear` 是一個 PyTorch 模塊，繼承自 `nn.Module`。**
- **`in_features` 和 `out_features`**：分別表示輸入和輸出的特徵數（即原始線性層的維度）。
- **`rank`**：低秩矩陣的秩，控制更新參數的大小。`rank` 越小，需要訓練的參數越少。

---

#### **初始化原始線性層和低秩矩陣**

python

複製程式碼

        `self.linear = nn.Linear(in_features, out_features, bias=True)                  # 冻结原始权重         self.linear.weight.requires_grad = False         self.linear.bias.requires_grad = False`

- 創建原始線性層 `self.linear`，但在微調中不訓練它：
    - **`requires_grad = False`**：禁止梯度更新，確保權重 WWW 保持不變。

---

#### **添加低秩矩陣 A 和 B**

python

複製程式碼

        `self.A = nn.Parameter(torch.zeros(in_features, rank))         self.B = nn.Parameter(torch.zeros(rank, out_features))`

- **`A` 和 `B` 是可訓練的參數矩陣**：
    - A∈Rd×rA \in \mathbb{R}^{d \times r}A∈Rd×r：將輸入特徵壓縮到低維空間。
    - B∈Rr×kB \in \mathbb{R}^{r \times k}B∈Rr×k：將壓縮的特徵還原回高維空間。
- 使用 **`nn.Parameter`** 包裝，讓 PyTorch 追蹤它們的梯度，從而進行訓練。

---

#### **初始化低秩矩陣**

python

複製程式碼

        `nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))         nn.init.zeros_(self.B)`

- 使用 Kaiming 初始化 AAA（適合 ReLU 激活函數），將 BBB 初始化為零矩陣，減少對初始模型的干擾。

---

#### **前向傳播的計算**

python

複製程式碼

    `def forward(self, x):         return self.linear(x) + x @ self.A @ self.B`

- 輸入 xxx 先通過原始線性層 `self.linear(x)`，再加上低秩更新項 x@A@Bx @ A @ Bx@A@B：
    - **`x @ self.A`**：將輸入映射到低維空間。
    - **`@ self.B`**：將低維特徵映射回高維。

---

### **2. 替換 ViT 模型中的線性層**

#### **`apply_lora_to_vit` 函數**

python

複製程式碼

`def apply_lora_to_vit(vit_model, rank=4):     for name, module in vit_model.named_modules():         if isinstance(module, nn.Linear):             in_features = module.in_features             out_features = module.out_features             setattr(                 vit_model,                 name,                 LoRALinear(in_features, out_features, rank)             )     return vit_model`

- 遍歷 ViT 模型的所有子模塊，尋找 `nn.Linear` 層。
- 將所有線性層替換為 **`LoRALinear`**，並保持其結構和輸出不變。

---

### **3. 加載預訓練的 ViT 模型**

#### **使用 PyTorch 的 ViT 模型**

python

複製程式碼

`vit_model = models.vit_b_16(pretrained=True) vit_model = apply_lora_to_vit(vit_model, rank=4)`

- 加載一個預訓練的 ViT 模型（基於 ImageNet）。
- 使用 `apply_lora_to_vit` 將模型中的線性層替換為 LoRA 層。

---

### **4. CIFAR-10 數據集預處理**

#### **數據增強和分割**

python

複製程式碼

`transform = Compose([ToTensor(), Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]) dataset = CIFAR10(root="./data", train=True, download=True, transform=transform) train_set, val_set = random_split(dataset, [45000, 5000]) train_loader = DataLoader(train_set, batch_size=32, shuffle=True) val_loader = DataLoader(val_set, batch_size=32, shuffle=False)`

- **數據增強：** 將 CIFAR-10 圖像轉換為 Tensor 並正則化到 [-1, 1] 範圍。
- **數據集分割：** 將訓練數據分為 45000 張訓練圖片和 5000 張驗證圖片。
- **數據加載器：** 使用批次大小為 32 的 `DataLoader` 提高訓練效率。

---

### **5. 定義訓練過程**

#### **模型、損失函數和優化器**

python

複製程式碼

`device = torch.device("cuda" if torch.cuda.is_available() else "cpu") vit_model.to(device) criterion = nn.CrossEntropyLoss() optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, vit_model.parameters()), lr=1e-4)`

- 將模型移動到 GPU（如果可用）。
- 使用交叉熵損失函數計算分類任務的誤差。
- 僅優化可訓練參數（即 AAA 和 BBB），通過 `filter` 避免更新預訓練權重。

---

#### **訓練過程**

python

複製程式碼

`def train(model, dataloader, criterion, optimizer, epochs=5):     model.train()     for epoch in range(epochs):         total_loss = 0         for images, labels in dataloader:             images, labels = images.to(device), labels.to(device)             optimizer.zero_grad()             outputs = model(images)             loss = criterion(outputs, labels)             loss.backward()             optimizer.step()             total_loss += loss.item()         print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")`

- 設置模型為訓練模式。
- 遍歷每個訓練批次：
    1. 將圖像和標籤移動到 GPU。
    2. 前向傳播計算預測。
    3. 計算損失並反向傳播梯度。
    4. 更新低秩矩陣 AAA 和 BBB。

---

### **6. 驗證模型性能**

#### **驗證過程**

python

複製程式碼

`def validate(model, dataloader):     model.eval()     correct = 0     total = 0     with torch.no_grad():         for images, labels in dataloader:             images, labels = images.to(device), labels.to(device)             outputs = model(images)             _, predicted = torch.max(outputs.data, 1)             total += labels.size(0)             correct += (predicted == labels).sum().item()     print(f"Validation Accuracy: {100 * correct / total:.2f}%")`

- 模型設置為評估模式（禁用 Dropout 等訓練特性）。
- 計算驗證數據上的分類準確率。

---

### **總結**

這段代碼通過引入 LORA 技術，實現了對 ViT 模型的高效微調。LORA 的關鍵在於：

1. **低秩分解：** 只訓練小矩陣 AAA 和 BBB，降低參數量和內存佔用。
2. **模塊化設計：** 使用 `apply_lora_to_vit` 函數，輕鬆將 LORA 集成到任意模型中。
3. **高效實現：** 在保留原始模型性能的同時，大幅度降低訓練成本。


Ref: 
LoRA（Low-Rank Adaptation）详解 - 大师兄的文章 - 知乎
https://zhuanlan.zhihu.com/p/663557294

大模型常用微调方法介绍：P-Tuning、Prefix Tuning、Adapter、LoRA等 - 王海的文章 - 知乎
https://zhuanlan.zhihu.com/p/7474042360

【OpenLLM 006】LoRA:大模型的低秩适配-最近大火的lora到底是什么东西？为啥stable diffusion和开源ChatGPT复现都在用？ - OpenLLMAI的文章 - 知乎
https://zhuanlan.zhihu.com/p/620327907