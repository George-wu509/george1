
Paper: https://arxiv.org/abs/1801.07455



![[Pasted image 20250904174144.png]]


![[Pasted image 20250904174417.png]]

![[Pasted image 20250904174432.png]]



### ST-GCN 網路架構 (Network Architecture)

#### 1. 輸入 (Input)

ST-GCN的輸入是一個代表骨架序列的張量 (Tensor)，其維度通常是 `(N, C, T, V, M)`：

- `N`: 批次大小 (Batch size)。
- `C`: 每個關節點的特徵維度。通常是3，即 `(x, y, confidence)`。
- `T`: 時間序列的長度（幀數）。
- `V`: 關節點的數量 (Vertices)。例如，COCO數據集為17，NTU-RGB+D為25。
- `M`: 一幀畫面中的人數。

#### 2. 核心構建塊：ST-GCN Unit

整個網路是由多個 ST-GCN Unit 堆疊而成的。每個 Unit 包含兩個關鍵部分：**空間圖卷積 (GCN)** 和 **時間卷積 (TCN)**。

- **a) 空間圖卷積層 (Spatial GCN Layer):**
    
    - **目標:** 在**單一幀**內，聚合關節的空間鄰域信息。
    - **運作方式:** 對於某一幀的某個關節 `v`，模型會聚合與 `v` 通過骨骼直接相連的關節的特徵。例如，「手腕」節點會聚合來自「手肘」和「手掌」節點的信息。
    - **數學表達 (簡化版):** f_out=sum_k(mathbfA_kodotmathbfM∗k)f∗inmathbfW_k
        
        - f_in: 輸入的特徵張量 `(C, T, V)`。
            
        - mathbfA∗k: **鄰接矩陣 (Adjacency Matrix)**。這是一個 `(V, V)` 的矩陣，如果關節 `i` 和 `j` 相連，則 A∗ij=1，否則為0。這是**圖結構的核心**。ST-GCN將其擴展為多個鄰接矩陣（分區策略），例如 mathbfA_0 代表節點自身，mathbfA_1 代表相鄰的向心節點等。
            
        - mathbfM_k: **可學習的權重遮罩**。這是一個 `(V, V)` 的矩陣，用於學習每條邊的重要性。
            
        - mathbfW_k: 標準的權重矩陣，類似於CNN中的卷積核，用於特徵變換。
            
        - odot: 逐元素相乘。
            
- **b) 時間卷積層 (Temporal Convolutional Layer):**
    
    - **目標:** 在**時間維度**上，聚合每個關節的時序動態信息。
        
    - **運作方式:** 在GCN層完成空間特徵聚合後，TCN層對每個關節的時間序列進行一維卷積（實際上是用一個 `kernel_size = (K_t, 1)` 的二維卷積實現）。例如，模型會觀察「手腕」節點在連續 `K_t` 幀內的運動模式（例如，來回擺動）。
        

#### 3. 整體架構

1. 輸入的骨架序列首先經過一個數據預處理層（批次標準化）。
    
2. 數據流經約9-10個堆疊的 **ST-GCN Unit**。每個Unit都包含 `GCN -> BatchNorm -> ReLU -> TCN -> BatchNorm -> ReLU` 的流程。
    
3. 通常會在某些Unit之間加入類似ResNet的**殘差連接 (Residual Connection)**，並使用步長 (stride) 為2的時間卷積來進行下採樣，以擴大感受野。
    
4. 經過所有ST-GCN Unit後，得到一個高維的特徵張量。
    
5. 使用**全局平均池化 (Global Average Pooling)** 將 `(C', T', V)` 的特徵圖在時間和空間維度上池化為一個 `(C')` 的特徵向量。
    
6. 最後將該向量送入一個**全連接層 (Softmax分類器)**，輸出每個動作類別的概率。
    

---

### 與 RNN 的差別

|特性|RNN / LSTM|ST-GCN|
|---|---|---|
|**數據處理方式**|**序列化 (Sequential)**：逐幀處理，隱藏狀態在時間步之間傳遞。|**並行化 (Parallel)**：一次性處理整個時空圖，類似於CNN。計算效率更高。|
|**空間結構利用**|**隱式學習**：將每幀的關節點展平為向量，丟失了顯式的拓撲結構，模型必須自行從數據中學習關節間的關係。|**顯式建模**：通過預定義的鄰接矩陣，直接將人體結構的先驗知識（哪個關節與哪個關節相連）融入模型，學習更高效、更具可解釋性。|
|**時間依賴性**|擅長捕捉**全局長時序依賴**，但可能受梯度消失/爆炸問題影響。|通過堆疊多層時間卷積來捕捉時序依賴。感受野是**局部**的，但可以通過堆疊層數來擴大，類似於CNN。|
|**模型感受野**|理論上是整個過去的序列。|感受野是**局部**的（由GCN的鄰居階數和TCN的核大小決定），但通過深層堆疊可以覆蓋整個時空圖。|

匯出到試算表

**核心差異：** ST-GCN最大的優勢在於**顯式地利用了人體的空間圖結構**，這是一個非常強的歸納偏置 (inductive bias)，使得模型不必浪費精力去學習這個固定的結構，而是可以專注於學習基於該結構的運動模式。

---

### PyTorch 程式碼範例

這裡提供一個通用GNN和一個簡化版ST-GCN的程式碼，以作對比。

#### 1. 通用 GNN 範例 (使用 `torch_geometric`)

首先，你需要安裝 `torch_geometric`: `pip install torch_geometric`

Python

```
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# 1. 定義圖數據
# 假設一個有4個節點的圖 (e.g., 一個分子)
edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],  # 邊的起始節點
                           [1, 0, 2, 1, 3, 2]], # 邊的結束節點
                          dtype=torch.long)
# 每個節點有2個特徵
x = torch.randn(4, 2) 
data = Data(x=x, edge_index=edge_index)

# 2. 定義一個簡單的GNN模型
class SimpleGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # 第一層圖卷積
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # 第二層圖卷積
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# 3. 使用模型
model = SimpleGNN(in_channels=2, hidden_channels=16, out_channels=4)
output = model(data.x, data.edge_index)

print("通用GNN輸出形狀:", output.shape) # torch.Size([4, 4]), 每個節點的分類概率
```

#### 2. 簡化版 ST-GCN 範例 (使用純 PyTorch)

Python

```
import torch
import torch.nn as nn

class GraphConvolution(nn.Module):
    """
    簡化的空間圖卷積層
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels))

    def forward(self, x, A):
        # x shape: (N, C_in, T, V)
        # A shape: (V, V)
        # 實現 f_out = A * f_in * W
        x = torch.einsum('nctv,vw->nctw', x, A) # 鄰域聚合
        x = torch.einsum('nctw,oc->notw', x, self.weight) # 特徵變換
        return x.contiguous()

class STGCNBlock(nn.Module):
    """
    簡化的ST-GCN基本單元
    """
    def __init__(self, in_channels, out_channels, kernel_size_t):
        super().__init__()
        self.gcn = GraphConvolution(in_channels, out_channels)
        # 時間卷積, padding保證時間維度不變
        self.tcn = nn.Conv2d(out_channels, out_channels, 
                             kernel_size=(kernel_size_t, 1), 
                             padding=( (kernel_size_t - 1) // 2, 0) )
        self.relu = nn.ReLU()

    def forward(self, x, A):
        x = self.gcn(x, A)
        x = self.relu(x)
        x = self.tcn(x)
        x = self.relu(x)
        return x

class SimpleSTGCN(nn.Module):
    """
    一個非常簡化的ST-GCN模型
    """
    def __init__(self, num_classes, in_channels, num_joints):
        super().__init__()
        self.block1 = STGCNBlock(in_channels, 64, kernel_size_t=9)
        self.block2 = STGCNBlock(64, 128, kernel_size_t=9)
        
        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d(1)
        # 分類器
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x, A):
        # x shape: (N, C, T, V)
        # A shape: (V, V)
        x = self.block1(x, A)
        x = self.block2(x, A)
        
        x = self.gap(x) # shape: (N, 128, 1, 1)
        x = x.view(x.size(0), -1) # shape: (N, 128)
        
        x = self.fc(x)
        return x

# 3. 使用模型
# 模擬輸入數據
N, C, T, V = 16, 3, 50, 18 # 16個樣本, 3通道, 50幀, 18個關節
x = torch.randn(N, C, T, V)

# 創建一個簡單的鄰接矩陣 (這裡用單位矩陣代替，實際應為真實骨架連接)
# 加上自環
A = torch.eye(V) 

model = SimpleSTGCN(num_classes=10, in_channels=C, num_joints=V)
output = model(x, A)

print("ST-GCN輸出形狀:", output.shape) # torch.Size([16, 10]), 每個樣本的分類概率
```

---

### 具體舉例說明 ST-GCN 如何工作

**任務：** 識別 "揮手" 動作。

1. **輸入數據：** 一段2秒的影片（假設30fps，共60幀），MMPose提取出18個關節點。輸入張量 `x` 的形狀為 `(1, 3, 60, 18)`。同時，我們有一個固定的 `18x18` 的鄰接矩陣 `A`，它定義了「手」與「手腕」、「手腕」與「手肘」等關節的連接關係。
    
2. **進入第一個ST-GCN Unit：**
    
    - **空間GCN層：**
        
        - 對於第 `t` 幀的「手腕」節點，GCN層會查看鄰接矩陣 `A`，找到它的鄰居——「手肘」和「手掌」。
            
        - 它會將「手肘」和「手掌」的特徵（它們的 `(x, y, conf)` 值）聚合到「手腕」上，並通過可學習的權重 `W` 進行變換。
            
        - **結果：** 「手腕」節點的新特徵不僅包含了自己的位置信息，還**融合了整個手臂的結構信息**。模型學到了一個代表 "手臂姿態" 的局部特徵。
            
    - **時間TCN層：**
        
        - GCN處理完所有幀後，TCN層開始工作。它看的是**同一個關節**在時間上的變化。
            
        - 對於「手腕」這個節點，TCN的9幀卷積核會觀察它在 `t-4` 到 `t+4` 幀的運動模式。
            
        - **結果：** 如果手腕在做左右往復運動，TCN層會捕捉到這個**震盪的時序模式**，並輸出一個強烈的響應。
            
3. **進入更深的ST-GCN Unit：**
    
    - 後面的層會基於前面層提取的「局部時空特徵」進行更高級的組合。
        
    - 例如，深層的GCN可能會學到「手臂擺動」與「身體軀幹穩定」之間的協同關係。
        
    - 深層的TCN會學習這些組合特徵的更長時間的演變規律。
        
4. **最終分類：**
    
    - 在經過所有層後，模型已經提取了非常豐富的、能夠描述「揮手」這個動作的時空特徵。
        
    - 全局池化層將這些複雜的特徵總結成一個單一的特徵向量。
        
    - 最後，分類器看到這個特徵向量，並高概率地將其識別為 "揮手" 類別。
        

通過這種方式，ST-GCN 完美地結合了對身體結構的理解和對運動模式的捕捉，使其在基於骨架的動作識別任務上取得了巨大的成功。