
評估AI模型的量化指標可分為**通用效率指標**與**任務專用指標**兩大類，涵蓋模型效能、運算效率與應用場景表現。以下整合各領域關鍵指標：


|                                    |     |
| ---------------------------------- | --- |
| [[###通用效率指標]]                      |     |
| [[###AI Model Performance 的量化指标 ]] |     |
| [[###計算 FLOPs方法跟公式]]               |     |
| [[###mAP 50 95 是什麼？]]              |     |
|                                    |     |
|                                    |     |

|        |                                          |
| ------ | ---------------------------------------- |
| 模型複雜度  | Parms, FLOPs                             |
| 性能指标   | Latency, Throughput                      |
| 计算资源消耗 | Memory consumption, MACs                 |
| 影片分析   | Cold start, throughput                   |
| 硬體層級   | GPU usage, Peak Memory, Token throughput |



### 通用效率指標

#### **模型複雜度**

1. **參數量 (Params)**  單位: M
    - 定義：模型可訓練參數總數
    - 影響：記憶體占用、儲存需求、推論延遲
    - 公式：直接統計模型層級參數
    - 硬體獨立性
        
2. **浮點運算次數 (FLOPs)**   單位: G
    - 定義：單次推論所需浮點運算總量
    - 細分：包含乘法累加運算(MACs)
    - 應用：評估運算成本與能耗
    - 硬體獨立性
#### **即時效能**

1. **單次推論耗時 (Latency)**
    - 定義：單次推論耗時
    - 影響：模型參數量、FLOPs、硬體架構
    - **Latency（延遲）**：延遲是指完成一個任務所需的時間。在GPU中，延遲通常與內存訪問、計算等因素相關。延遲可以針對整個GPU、SM或SP進行計算，但通常更關注的是SM層面的延遲，因為它直接影響了GPU的整體性能。
        
2. **單位時間處理量 (Throughput)**
    - 定義：單位時間處理量(如幀/秒)
    - 細分：硬體平行化能力、批次處理
    - **Throughput（吞吐量）**：吞吐量是指在單位時間內完成的任務數量。GPU的吞吐量通常由其並行執行能力決定，尤其是在執行大量線程時。吞吐量可以針對整個GPU或SM進行計算，通常用於評估GPU在特定任務中的效率。

| 指標             | 定義            | 關鍵影響因素           | 應用場景       |
| -------------- | ------------- | ---------------- | ---------- |
| **Latency**    | 單次推論耗時        | 模型參數量、FLOPs、硬體架構 | 即時應用(如自駕車) |
| **Throughput** | 單位時間處理量(如幀/秒) | 硬體平行化能力、批次處理     | 高併發服務      |

### **硬體利用率**
- **模型FLOPs利用率 (MFU)**  
    公式：
    MFU  =  Achieved FLOPs/s    /   Theoretical Peak FLOPs/s
    用途：評估訓練管線效率，低值表示記憶體瓶頸或平行化不足
    
## **影像任務專用指標**

## **目標檢測 (Object Detection)**

1. **交并比 (IoU)**
    - 計算：預測框與真值框交集面積/聯集面積
    - 閾值：常用0.5作為檢測合格基準
        
2. **平均精度均值 (mAP)**
    - 方法：在不同召回率下計算平均精度
    - 版本：COCO數據集採用mAP@[0.5:0.95]多閾值平均
        
3. **追蹤指標 (MOTA/IDF1)**
    - 應用：影片物件追蹤效能評估
    - 包含：誤檢率、漏檢率、ID切換次數
        
## **語義分割 (Semantic Segmentation)**

| 指標         | 公式                 | 特性        |
| ---------- | ------------------ | --------- |
| **Dice係數** | $$\frac{2          | X∩Y       |
| **像素準確率**  | 正確像素數總像素數總像素數正確像素數 | 易受類別不平衡影響 |
| **平均IoU**  | 各類別IoU均值           | 常用於多類別評估  |

## **影像分類 (Image Classification)**

1. **Top-1/Top-5準確率**
    - 定義：預測最高/前五機率是否包含真值標籤    
    - 適用：ImageNet等大規模分類
        
2. **混淆矩陣衍生指標**
    - 精確率 (Precision)：TPTP+FPTP+FPTP  
    - 召回率 (Recall)：TPTP+FNTP+FNTP    
    - F1分數：兩者調和平均    

## **影片分析特殊指標**

1. **冷啟動時間 (Cold Start)**
    - 定義：模型初始化到可運作的延遲
    - 影響：伺服器無狀態架構效能
        
2. **幀間一致性**
    - 方法：連續幀預測結果穩定性分
    - 指標：軌跡中斷率、位置漂移量
        
3. **處理吞吐量**
    - 進階：區分空間吞吐量(fps)與時域吞吐量(秒/影片)
    - 優化：如PatchNet減少偵測頻率提升4.9x效率
    
## **硬體層級指標**

| 指標           | 監控重點        | 應用案例    |
| ------------ | ----------- | ------- |
| **GPU利用率**   | 硬體運算資源使用率   | 成本優化    |
| **峰值啟動記憶體**  | 訓練過程最大記憶體消耗 | 分散式訓練規劃 |
| **Token吞吐量** | 生成式模型輸出效率   | LLM服務部署 |

## **指標選擇策略**

- **即時系統**：優先Latency/Throughput，次選mAP
- **資源受限環境**：監控Params/FLOPs與記憶體峰值
- **生成式模型**：需加入Perplexity/FID等生成品質指標
- **影片分析**：需綜合幀級指標與跨幀追蹤指標

最新趨勢顯示**複合指標體系**的採用增加，如YOLOv7同時優化AP與fps，反映實務需平衡精度與效能。硬體感知指標如MFU可協助診斷訓練瓶頸，而邊緣運算場景需特別關注記憶體與能耗指標




### AI Model Performance 的量化指标 

---

## **1. 计算复杂度**

### **1.1. Parameters (参数量)**

- 衡量模型的大小，通常表示为模型的总参数数目。
- 在 PyTorch 中可以通过 `sum(p.numel() for p in model.parameters())` 计算。

### **1.2. FLOPs (浮点运算数)**

- 表示模型在推理时需要执行的总浮点运算次数（Floating Point Operations）。
- 可以使用 `fvcore` 或 `torchinfo` 来计算。
- 
**FLOPs 與參數量的關係**

- **並非直接相關：**
    - 雖然參數量和 FLOPs 之間存在關聯，但它們並非直接的線性關係。
    - 參數量指的是模型中可學習變數的數量，而 FLOPs 則衡量模型執行過程中進行的浮點運算次數。
    - 模型可能具有大量參數，但其運算相對簡單，因此 FLOPs 可能不高。反之亦然。

**FLOPs 的計算方式**

- **基本概念：**
    
    - FLOPs 計算的是模型在一次前向傳播過程中進行的浮點運算總數。
    - 這包括加法、乘法、除法等浮點運算。
- **以簡單 CNN 模型為例：**
    
    - **卷積層（Convolution）：**
        - 卷積運算是 CNN 中主要的運算。
        - 對於每個輸出特徵圖的每個像素，都需要進行多次乘法和加法運算。
        - 計算 FLOPs 時，需要考慮卷積核的大小、輸入/輸出特徵圖的大小和通道數。
        - 具體計算方式：
            - 輸出特徵圖寬度*輸出特徵圖高度*輸出特徵圖通道數*卷積核寬度*卷積核高度*輸入特徵圖通道數*2(乘法和加法)
    - **ReLU 激活函數：**
        - ReLU 函數的運算相對簡單，主要是進行比較操作。
        - 通常，ReLU 運算的 FLOPs 可以忽略不計，或者將其視為一次比較運算。
    - **池化層（Pooling）：**
        - 池化層的運算（例如最大池化或平均池化）也相對簡單。
        - 最大池化主要進行比較運算，平均池化則涉及加法和除法運算。
        - 具體計算方式：
            - 輸出特徵圖寬度*輸出特徵圖高度*輸出特徵圖通道數*池化核寬度*池化核高度(平均池化需要額外進行除法運算)
    - **全連接層（Fully Connected）：**
        - 全連接層的運算主要涉及矩陣乘法。
        - 計算 FLOPs 時，需要考慮輸入和輸出神經元的數量。
        - 具體計算方式：
            - 輸入神經元數量*輸出神經元數量*2(乘法和加法)
- **重要注意事項：**
    
    - FLOPs 衡量的是運算量，而不是實際的運行時間。
    - 實際運行時間還受到硬體、軟體和記憶體訪問等因素的影響。
    - FLOPs 與 FLOPS 不同：
        - FLOPs（小寫 "s"）指浮點運算次數。
        - FLOPS（大寫 "S"）指每秒浮點運算次數，衡量硬體性能。

希望這些資訊能夠幫助您更深入地了解 AI 模型的 FLOPs。

**範例：4x4 影像，3 通道，3x3 卷積核**

假設我們有一個：

- 輸入影像：4x4 像素，3 個通道（例如 RGB 影像）
- 卷積核：3x3 大小
- 為了簡化，我們假設輸出特徵圖的通道數為 1。
- stride=1,並且沒有padding.

**步驟分析：**

1. **輸出特徵圖大小：**
    
    - 因為 stride=1,並且沒有padding,輸出特徵圖的大小將是 (4-3+1) x (4-3+1) = 2x2 像素。
2. **單一輸出像素的運算：**
    
    - 對於輸出特徵圖中的每個像素，我們需要執行以下運算：
        - 將 3x3 卷積核應用於輸入影像的對應區域。
        - 這涉及 3x3x3 = 27 次乘法運算（因為有 3 個輸入通道）。
        - 然後，將這 27 個乘法結果相加，這涉及 26 次加法運算。
        - 因此，單一輸出像素的總運算次數為 27（乘法）+ 26（加法）= 53 次。
3. **總運算次數：**
    
    - 由於輸出特徵圖的大小為 2x2 像素，因此總運算次數為 53（單一像素運算）x 2x2（輸出像素數量）= 212 次。
4. **FLOPs 計算：**
    
    - 在深度學習中，通常將乘法和加法都視為浮點運算。
    - 因此，總 FLOPs 為 212。

---

## **2. 性能指标**

### **2.1. Latency (推理延迟)**

- 表示单个输入样本经过模型所需的时间。
- 通过 `time.time()` 或 `torch.cuda.Event()` 进行测量。

### **2.2. Throughput (吞吐量)**

- 代表模型每秒可以处理的样本数，即 `batch_size / latency`。

---

## **3. 计算资源消耗**

### **3.1. Memory Consumption (显存占用)**

- 衡量模型在推理或训练过程中消耗的 GPU 显存大小。
- 可以使用 `torch.cuda.memory_allocated()` 和 `torch.cuda.max_memory_allocated()` 监测。

### **3.2. MACs (乘加运算数, Multiply-Accumulate Operations)**

- 衡量计算工作量，类似 FLOPs，但只考虑乘加操作。

---

## **如何在 PyTorch 代码中计算这些指标？**

以下提供详细代码：

### **1. 计算参数量**

```python
import torch
import torchvision.models as models

# 以 ResNet50 为例
model = models.resnet50()

# 计算总参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params}")

```
---

### **2. 计算 FLOPs 和 MACs**

使用 `fvcore` 计算 FLOPs：

```python
from fvcore.nn import FlopCountAnalysis, parameter_count_table

# 创建一个模型实例
model = models.resnet50()
model.eval()

# 创建输入张量
input_tensor = torch.randn(1, 3, 224, 224)

# 计算 FLOPs
flops = FlopCountAnalysis(model, input_tensor)
print(f"FLOPs: {flops.total()}")

# 计算参数表
print(parameter_count_table(model))
```

如果没有安装 `fvcore`，可以使用 `torchinfo`：

```python
from torchinfo import summary

summary(model, input_size=(1, 3, 224, 224), col_names=["input_size", "output_size", "num_params", "mult_adds"])
```
---

### **3. 计算 Latency**

```python
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 生成随机输入
input_tensor = torch.randn(1, 3, 224, 224).to(device)

# 预热 GPU，避免初次计算带来的影响
for _ in range(10):
    _ = model(input_tensor)

# 测试推理时间
with torch.no_grad():
    start_time = time.time()
    for _ in range(100):  # 计算 100 次取平均
        _ = model(input_tensor)
    end_time = time.time()

latency = (end_time - start_time) / 100
print(f"Latency per sample: {latency:.6f} sec")

```

---

### **4. 计算 Throughput**

```python
batch_size = 32
input_tensor = torch.randn(batch_size, 3, 224, 224).to(device)

# 计算 throughput
with torch.no_grad():
    start_time = time.time()
    for _ in range(10):  # 运行 10 次
        _ = model(input_tensor)
    end_time = time.time()

throughput = (batch_size * 10) / (end_time - start_time)
print(f"Throughput: {throughput:.2f} samples/sec")

```
---

### **5. 计算 GPU 显存占用**

```python
torch.cuda.reset_peak_memory_stats()

input_tensor = torch.randn(1, 3, 224, 224).to(device)

# 运行推理
with torch.no_grad():
    _ = model(input_tensor)

# 获取显存信息
allocated_memory = torch.cuda.memory_allocated()
max_memory = torch.cuda.max_memory_allocated()

print(f"Current allocated memory: {allocated_memory / 1024**2:.2f} MB")
print(f"Max allocated memory: {max_memory / 1024**2:.2f} MB")

```

---

### **总结**

|指标|含义|计算方法|
|---|---|---|
|**参数量**|模型的参数总数|`sum(p.numel() for p in model.parameters())`|
|**FLOPs**|浮点运算数|`FlopCountAnalysis(model, input_tensor).total()`|
|**Latency**|单次推理时间|`time.time()` 或 `torch.cuda.Event()`|
|**Throughput**|吞吐量 (样本数/秒)|`batch_size / latency`|
|**Memory Consumption**|GPU 显存使用|`torch.cuda.memory_allocated()`|

这些方法适用于 CNN 和 Transformer 结构的模型，如果是 Transformer，可以用 `torchinfo` 计算 `num_params` 和 `mult_adds`，并通过 `fvcore` 获取 FLOPs。

如果你有具体的模型需求，也可以提供你的 PyTorch 代码，我可以帮你优化计算这些指标！



### 計算 FLOPs方法跟公式


當 AI 模型變得複雜且層數眾多時，計算 FLOPs 的確會變得更具挑戰性。不過，我們可以採用一些系統化的方法來應對：

**1. 分解模型結構：**

- 將複雜的模型分解為更小的、更易於管理的組成部分，例如單個卷積層、全連接層、Transformer 層等。
- 針對每個組成部分，使用前面提到的方法（或針對特定層的公式）計算其 FLOPs。

**2. 利用現有工具和庫：**

- 許多深度學習框架（如 PyTorch 和 TensorFlow）都提供了用於計算 FLOPs 的工具和庫。
    - 例如，`torchsummary` 庫可以用於 PyTorch 模型，以提供模型的摘要，包括 FLOPs。
- 這些工具可以自動執行許多繁瑣的計算，使過程更加高效。

**3. 針對常見層的 FLOPs 公式：**

- 熟悉常見層（如卷積層、全連接層、Transformer 層）的 FLOPs 公式。
- 這有助於您快速估算各層的 FLOPs，並更好地理解模型的計算複雜度。
    - 卷積層:
	    -  舉例: 4x4 image, channel=3, 卷積核size=3x
        - 輸出特徵圖寬度(2)*輸出特徵圖高度(2)*輸出特徵圖通道數(3)*卷積核寬度(3)*卷積核高度(3)*輸入特徵圖通道數*2(乘法和加法)
        -   (3x3x3(卷積核)乘法運算+26加法運算) *2x2  輸出特徵圖 = 212 (FLOPs for this)
    - 全連接層:
        - 輸入神經元數量*輸出神經元數量*2(乘法和加法)
    - Transformer層:
        - Transformer層的計算相當複雜，需要考慮到attention的計算，以及多層的堆疊，若需要詳細計算，可以搜尋Transformer FLOPs.

**4. 注意事項：**

- FLOPs 僅衡量浮點運算的次數，並不能完全反映模型的實際運行時間。
- 其他因素（如記憶體訪問、硬體架構等）也會影響模型的性能。
- 在計算複雜模型的FLOPs時，可以使用程式來幫助計算，可以有效的減少人為計算的錯誤。

**總結：**

- 計算複雜模型的 FLOPs 需要系統化的方法和工具的輔助。
- 通過分解模型結構、利用現有工具和熟悉常見層的 FLOPs 公式，您可以更有效地評估模型的計算複雜度。
- 此外，理解FLOPs的本質，以及它與實際運行時間的關係，對於模型優化至關重要。

**範例：4x4 影像，3 通道，3x3 卷積核**

假設我們有一個：

- 輸入影像：4x4 像素，3 個通道（例如 RGB 影像）
- 卷積核：3x3 大小
- 為了簡化，我們假設輸出特徵圖的通道數為 1。
- stride=1,並且沒有padding.

**步驟分析：**

1. **輸出特徵圖大小：**
    
    - 因為 stride=1,並且沒有padding,輸出特徵圖的大小將是 (4-3+1) x (4-3+1) = 2x2 像素。
2. **單一輸出像素的運算：**
    
    - 對於輸出特徵圖中的每個像素，我們需要執行以下運算：
        - 將 3x3 卷積核應用於輸入影像的對應區域。
        - 這涉及 3x3x3 = 27 次乘法運算（因為有 3 個輸入通道）。
        - 然後，將這 27 個乘法結果相加，這涉及 26 次加法運算。
        - 因此，單一輸出像素的總運算次數為 27（乘法）+ 26（加法）= 53 次。
3. **總運算次數：**
    
    - 由於輸出特徵圖的大小為 2x2 像素，因此總運算次數為 53（單一像素運算）x 2x2（輸出像素數量）= 212 次。
4. **FLOPs 計算：**
    
    - 在深度學習中，通常將乘法和加法都視為浮點運算。
    - 因此，總 FLOPs 為 212。

**公式總結：**

- 輸出特徵圖寬度：(輸入寬度 - 卷積核寬度 + 1)
- 輸出特徵圖高度:(輸入高度 - 卷積核高度 + 1)
- 單一輸出像素FLOPs: 卷積核寬度 * 卷積核高度 * 輸入通道數(乘法) + (卷積核寬度 * 卷積核高度 * 輸入通道數 - 1)(加法)
- 總FLOPs: 輸出特徵圖寬度 * 輸出特徵圖高度 * 單一輸出像素FLOPs.

**重要補充：**

- 如果卷積層包含偏置（bias），則每個輸出像素還需要進行一次加法運算。
- 若輸出通道數不為1,需要將上述計算結果，再乘上輸出通道數量。
- 實際的深度學習框架通常會對這些運算進行優化，因此實際運行時間可能會有所不同。
- ReLU 等激活函數的 FLOPs 通常可以忽略不計，但如果需要非常精確的計算，也可以將其納入考慮。

希望這個更具體的範例能夠幫助您更好地理解 CNN 卷積層的 FLOPs 計算。



### mAP 50 95 是什麼？

### 計算 mAP@50:95 的詳細流程

在物件檢測任務中，mAP（mean Average Precision）是一個常用的評估指標，而 mAP@50:95（或 mAP@[.5:.95]）表示在不同 IoU（Intersection over Union）閾值（從 0.5 到 0.95，步長為 0.05）下計算的平均精確度的均值。以下將以你提供的數據集為例（5 張圖，每張圖包含 3 種物件：cat、dog、car），一步步詳細說明如何計算 mAP@50:95。

---

### 前提假設

為了便於說明，我們假設以下條件：

- 數據集：5 張圖片，每張圖有 3 個 ground truth 物件（cat、dog、car），總計 5×3=15 5 \times 3 = 15 5×3=15 個 ground truth。
- 模型預測：每張圖模型會輸出若干檢測框（bounding box），每個框包含物件類別（cat、dog、car）和置信度分數（confidence score）。
- 評估目標：計算所有類別在 IoU 閾值從 0.5 到 0.95 下的 mAP。

---

### 步驟 1：理解基本概念

1. **IoU（Intersection over Union）**：
    - IoU 衡量預測框與 ground truth 框的重疊程度： IoU=預測框與真實框的交集面積預測框與真實框的並集面積\text{IoU} = \frac{\text{預測框與真實框的交集面積}}{\text{預測框與真實框的並集面積}}IoU=預測框與真實框的並集面積預測框與真實框的交集面積​
    - IoU 閾值決定一個預測框是否被視為“正樣本”（True Positive, TP）。
2. **Precision 和 Recall**：
    - Precision = $\large \frac{\text{TP}}{\text{TP + FP}}$​（真陽性 / 所有預測陽性）
    - Recall = $\large \frac{\text{TP}}{\text{TP + FN}}$​（真陽性 / 所有真實陽性）
    - TP（True Positive）：預測正確的檢測框。
    - FP（False Positive）：預測錯誤的檢測框。
    - FN（False Negative）：未被檢測到的 ground truth。
3. **AP（Average Precision）**：
    - AP 是 Precision-Recall 曲線下的面積，通常通過 11 點插值或全點插值計算。
    - 在 COCO 標準中，使用全點插值（對所有 recall 點取最大 precision 的平均值）。
4. **mAP@50:95**：
    - 在 IoU 閾值從 0.5 到 0.95（步長 0.05，共 10 個閾值：0.5, 0.55, 0.6, ..., 0.95）下計算 AP，然後取平均值。
    - 對每個類別單獨計算 AP，再對所有類別取平均值，得到 mAP。

---

### 步驟 2：假設模型預測結果

假設模型對這 5 張圖的預測如下（為了簡化，假設每張圖預測 4 個框）：

#### 圖片 1：

- Ground Truth: cat (框 G1), dog (框 G2), car (框 G3)
- 預測框：
    - P1: cat, score=0.9, IoU(G1)=0.8
    - P2: dog, score=0.85, IoU(G2)=0.7
    - P3: car, score=0.6, IoU(G3)=0.4
    - P4: cat, score=0.5, IoU(G1)=0.3

#### 圖片 2-5：

類似假設，每張圖有 3 個 ground truth 和 4 個預測框，IoU 和置信度分數不同。

總計：

- Ground Truth: 5×3=15 5 \times 3 = 15 5×3=15 個。
- 預測框: 5×4=20 5 \times 4 = 20 5×4=20 個。

---

### 步驟 3：計算單個類別的 AP（以 “cat” 為例）

我們以 “cat” 類別為例，逐步計算 AP。

#### 3.1 收集所有預測框

假設模型對 5 張圖的 “cat” 預測如下（按置信度降序排列）：

- P1: score=0.9, IoU=0.8 (圖片 1)
- P2: score=0.75, IoU=0.6 (圖片 2)
- P3: score=0.65, IoU=0.5 (圖片 3)
- P4: score=0.5, IoU=0.3 (圖片 1, 與 G1 重複)
- P5: score=0.4, IoU=0.7 (圖片 4)
- P6: score=0.2, IoU=0.9 (圖片 5)

總共 5 個 ground truth “cat”，6 個預測框。

#### 3.2 按 IoU 閾值分配 TP 和 FP

以 IoU=0.5 為例（即 mAP@50）：

- 規則：
    - 若 IoU ≥ 0.5，且該 ground truth 未被匹配，則為 TP。
    - 若 IoU < 0.5 或重複匹配，則為 FP。
    - 每個 ground truth 只匹配一次，按置信度從高到低排序。

計算：

1. P1: score=0.9, IoU=0.8 ≥ 0.5 → TP (匹配圖片 1 的 cat)
2. P2: score=0.75, IoU=0.6 ≥ 0.5 → TP (匹配圖片 2 的 cat)
3. P3: score=0.65, IoU=0.5 ≥ 0.5 → TP (匹配圖片 3 的 cat)
4. P4: score=0.5, IoU=0.3 < 0.5 → FP
5. P5: score=0.4, IoU=0.7 ≥ 0.5 → TP (匹配圖片 4 的 cat)
6. P6: score=0.2, IoU=0.9 ≥ 0.5 → TP (匹配圖片 5 的 cat)

- TP = 5, FP = 1, FN = 0（所有 ground truth 都被檢測到）。

#### 3.3 計算 Precision 和 Recall

按置信度排序，逐個計算：

|預測框|Score|TP/FP|TP 累計|FP 累計|Precision|Recall|
|---|---|---|---|---|---|---|
|P1|0.9|TP|1|0|1.0|0.2|
|P2|0.75|TP|2|0|1.0|0.4|
|P3|0.65|TP|3|0|1.0|0.6|
|P4|0.5|FP|3|1|0.75|0.6|
|P5|0.4|TP|4|1|0.8|0.8|
|P6|0.2|TP|5|1|0.83|1.0|

#### 3.4 計算 AP@50

- **全點插值法**（COCO 標準）：
    - 對每個 Recall 點（0.0 到 1.0），取右側最大的 Precision：
        - Recall = 0.0: Precision = 1.0
        - Recall = 0.2: Precision = 1.0
        - Recall = 0.4: Precision = 1.0
        - Recall = 0.6: Precision = 0.83
        - Recall = 0.8: Precision = 0.83
        - Recall = 1.0: Precision = 0.83
    - 平均值： AP@50=1.0+1.0+1.0+0.83+0.83+0.836=0.915\text{AP@50} = \frac{1.0 + 1.0 + 1.0 + 0.83 + 0.83 + 0.83}{6} = 0.915AP@50=61.0+1.0+1.0+0.83+0.83+0.83​=0.915

#### 3.5 對其他 IoU 閾值重複計算

- 以 IoU=0.75 為例：
    - P1: IoU=0.8 ≥ 0.75 → TP
    - P2: IoU=0.6 < 0.75 → FP
    - P3: IoU=0.5 < 0.75 → FP
    - P4: IoU=0.3 < 0.75 → FP
    - P5: IoU=0.7 < 0.75 → FP
    - P6: IoU=0.9 ≥ 0.75 → TP
    - TP = 2, FP = 4, FN = 3
    - Precision 和 Recall 重新計算，得到 AP@75（假設為 0.4）。
- 重複此過程，計算 IoU = 0.5, 0.55, 0.6, ..., 0.95 的 AP。

#### 3.6 計算 “cat” 的 mAP@50:95

假設 10 個 IoU 閾值的 AP 如下：

- [0.915, 0.85, 0.75, 0.65, 0.55, 0.45, 0.4, 0.3, 0.2, 0.1]
- mAP@50:95 = 0.915+0.85+0.75+0.65+0.55+0.45+0.4+0.3+0.2+0.110=0.515\frac{0.915 + 0.85 + 0.75 + 0.65 + 0.55 + 0.45 + 0.4 + 0.3 + 0.2 + 0.1}{10} = 0.515100.915+0.85+0.75+0.65+0.55+0.45+0.4+0.3+0.2+0.1​=0.515

---

### 步驟 4：對所有類別計算 mAP

- 對 “dog” 和 “car” 重複步驟 3，得到各自的 mAP@50:95。
- 假設：
    - “dog”: mAP@50:95 = 0.48
    - “car”: mAP@50:95 = 0.53
- 最終 mAP@50:95 = 0.515+0.48+0.533=0.508\frac{0.515 + 0.48 + 0.53}{3} = 0.50830.515+0.48+0.53​=0.508

---

### 總結計算流程

1. **收集預測與 Ground Truth**：整理模型對每張圖的預測框和真實框。
2. **按類別計算 AP**：
    - 按置信度排序預測框。
    - 根據 IoU 閾值分配 TP/FP/FN。
    - 計算 Precision-Recall 曲線，求 AP。
3. **遍歷 IoU 閾值**：對 0.5 到 0.95 的 10 個閾值計算 AP，取平均值得 mAP@50:95。
4. **平均所有類別**：對 cat、dog、car 的 mAP@50:95 取平均值，得到最終結果。

---

### 注意事項

- **數據量小**：5 張圖的數據集較小，mAP 可能受隨機性影響較大。
- **非最大抑制（NMS）**：實際計算前需應用 NMS 去除重複預測框，這裡簡化未考慮。
- **COCO 標準**：以上流程遵循 COCO 評估協議，與 Pascal VOC（僅計算 mAP@50）不同。





Reference:
Transformer模型的参数量和FLOPs的计算 - 智彦博的文章 - 知乎
https://zhuanlan.zhihu.com/p/583106030

【LLM指北】五、参数量、计算量FLOPS推导 - 小明的HZ的文章 - 知乎
https://zhuanlan.zhihu.com/p/676113501

神经网络中的参数量和FLOPs的计算 - 卓不凡的文章 - 知乎
https://zhuanlan.zhihu.com/p/580166072

CNN 模型所需的计算力（flops）和参数（parameters）数量是怎么计算的？ - 知乎
https://www.zhihu.com/question/65305385


