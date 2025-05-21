

PyTorch AMP (Automatic Mixed Precision) 與量化技術 (PTQ 和 QAT) 之間的差異與關聯非常重要。它們都是模型優化的手段，但著眼點和實現方式有所不同。

首先，我們來詳細解釋一下各個概念：

### PyTorch AMP (自動混合精度)

正如你所描述的，PyTorch AMP 是一種在訓練和推斷過程中，自動將部分計算從全精度浮點數 (FP32) 轉換為半精度浮點數 (FP16) 的技術。

- **核心目標**：
    - **加速訓練/推斷**：FP16 的計算速度通常比 FP32 快 (尤其在支援 Tensor Core 的 NVIDIA GPU 上)。
    - **減少記憶體佔用**：FP16 數字佔用的記憶體是 FP32 的一半，這意味著可以使用更大的批次大小 (batch size) 或訓練更大的模型。
- **工作原理**：
    - **混合精度 (Mixed Precision)**：並非所有操作都轉換為 FP16。AMP 會智能地選擇哪些操作可以在 FP16 下安全執行而不會損失太多精度 (如卷積、矩陣乘法)，哪些操作則需要保持在 FP32 以維持數值穩定性 (如損失計算、批次歸一化中的累加操作)。
    - **動態損失縮放 (Dynamic Loss Scaling)**：由於 FP16 的表示範圍比 FP32 窄，在反向傳播計算梯度時，較小的梯度值容易變成零 (underflow)。損失縮放會將損失值乘以一個較大的縮放因子，從而放大梯度值，使其在 FP16 範圍內，然後在更新權重之前再將梯度縮放回去。PyTorch 中的 `torch.cuda.amp.GradScaler` 就是為此設計的。
    - **自動轉換 (Automatic)**：開發者主要透過 `torch.cuda.amp.autocast` 上下文管理器來啟用 AMP，PyTorch 會自動處理哪些運算轉為 FP16。
- **數據類型**：主要涉及 **FP32 (單精度浮點數)** 和 **FP16 (半精度浮點數)** 之間的轉換。這仍然是浮點數運算。

### 量化 (Quantization)

量化是一種將模型權重和/或激活值從高精度浮點數 (如 FP32 或 FP16) 轉換為低精度定點整數 (如 INT8、INT4，甚至二進制) 的技術。

- **核心目標**：
    - **大幅減小模型體積**：INT8 權重比 FP32 權重小 4 倍。
    - **加速推斷**：許多硬體 (CPU, GPU, Edge AI 晶片) 對 INT8 等整數運算有專門優化，速度遠快於浮點運算。
    - **降低功耗**：整數運算通常比浮點運算更節能。
- **工作原理**：
    - 將浮點數的範圍映射到一個較小的整數範圍。這通常需要兩個參數：
        - **縮放因子 (Scale)**：一個浮點數，用於將反量化後的整數還原到接近原始浮點數的範圍。
        - **零點 (Zero-point)**：一個整數，表示浮點數 0 在量化後的整數值。
    - 公式示意：RealValue≈(QuantizedValue−ZeroPoint)×Scale
- **數據類型**：主要涉及從 **浮點數 (FP32, FP16)** 到 **整數 (INT8, UINT8, INT4 等)** 的轉換。

現在來看兩種主要的量化方法：

#### 1. Post-Training Quantization (PTQ, 後訓練量化)

- **定義**：在模型已經完成 FP32 (或 FP16) 訓練之後，再對其進行量化的過程。
- **流程**：
    1. **準備預訓練模型**：獲取一個已經訓練好的 FP32 模型。
    2. **校準 (Calibration)**：向模型輸入一小部分有代表性的校準數據集 (calibration dataset)。收集模型中各層權重和激活值的統計信息 (如最小值、最大值、分佈等)。
    3. **計算量化參數**：根據收集到的統計信息，為每一層或每個通道計算最佳的縮放因子和零點。
    4. **轉換模型**：使用計算出的參數將模型的權重轉換為 INT8。對於激活值，可以在推斷時動態量化 (dynamic quantization)，或者如果也進行了校準，則可以靜態量化 (static quantization)。
- **優點**：
    - 簡單快捷，不需要重新訓練模型。
    - 不需要原始的完整訓練數據集或訓練流程。
- **缺點**：
    - 通常會導致一定的精度損失，因為模型在訓練時並不知道後續會被量化。對於某些對精度非常敏感的模型，精度下降可能比較明顯。

#### 2. Quantization-Aware Training (QAT, 量化感知訓練)

- **定義**：在模型訓練或微調 (fine-tuning) 過程中，模擬量化操作引入的誤差，讓模型學會適應這種誤差。
- **流程**：
    1. **準備模型**：可以從一個預訓練的 FP32 模型開始，或者從頭開始訓練。
    2. **插入偽量化節點 (Fake Quantization Nodes)**：在模型的前向傳播路徑中，對權重和/或激活值插入模擬量化的操作。這些操作會模擬量化和反量化過程 (float→int→float)，從而引入量化誤差，但梯度計算仍然使用浮點數 (通常通過 Straight-Through Estimator, STE 來傳遞梯度)。
    3. **訓練/微調**：使用帶有偽量化節點的模型進行訓練或微調。模型在優化過程中會學著去減小量化誤差對最終結果的影響。
    4. **轉換為真實量化模型**：訓練完成後，將學習到的權重和偽量化節點中的量化參數 (通常是學習到的 min/max 範圍，進而得到 scale/zero-point) 用於將模型轉換為真正的 INT8 模型。
- **優點**：
    - 通常比 PTQ 能達到更高的精度，因為模型在訓練時已經考慮了量化效應。
    - 對於複雜或對精度敏感的模型，QAT 是更好的選擇。
- **缺點**：
    - 過程比 PTQ 複雜，需要修改訓練流程並進行重新訓練或微調。
    - 訓練時間更長。

### PyTorch AMP 與 PTQ/QAT 的差別與關係

AMP: 自動混合精度(Automatic Mixed Precision)

|特性|PyTorch AMP (自動混合精度)|Quantization (PTQ & QAT)|
|:--|:--|:--|
|**主要目的**|加速訓練/推斷，減少訓練時記憶體佔用|大幅減小模型體積，加速推斷，降低功耗|
|**數據類型轉換**|FP32 ↔ FP16 (浮點數之間)|FP32/FP16 → INT8/INT4 等 (浮點數到整數)|
|**精度影響**|較小，通常能維持接近 FP32 的精度|可能有精度損失，QAT 通常優於 PTQ|
|**實現時機**|訓練時動態進行，或推斷時使用|PTQ: 訓練後；QAT: 訓練/微調過程中模擬|
|**硬體依賴**|依賴支援 FP16 運算的硬體 (如 NVIDIA Tensor Core)|依賴支援 INT8 等整數運算的硬體|
|**複雜度**|相對簡單，透過 API 易於集成|PTQ 較簡單，QAT 較複雜，需修改訓練流程|

匯出到試算表

**關係：**

1. **不同層次的優化**：
    - AMP 主要是在浮點數領域內進行優化，從 FP32 降到 FP16。
    - 量化則是更進一步的優化，從浮點數 (可以是 FP32 或 FP16) 降到定點整數 (如 INT8)。
2. **可以結合使用**：
    - 你可以先使用 AMP (FP16) 來訓練一個模型，以獲得訓練速度和記憶體上的收益。
    - 然後，你可以對這個用 AMP 訓練出來的 FP16 模型（或者更常見的是其 FP32 checkpoints，因為權重更新通常還是以 FP32 累計的）進行 PTQ 或 QAT，將其轉換為 INT8 模型以獲得最終推斷時的極致性能和模型大小優勢。
    - 例如，在 QAT 中，偽量化操作之前的原始浮點模型可以是 FP32，也可以是已經通過 AMP 訓練得到的模型狀態。
3. **目標有所側重**：
    - 如果你主要關心的是**訓練效率** (速度和記憶體)，AMP 是一個很好的選擇。
    - 如果你主要關心的是**推斷效率** (速度、模型大小、功耗)，尤其是在資源受限的邊緣設備上部署，那麼量化 (PTQ/QAT) 是關鍵技術。
4. **AMP 本身不是量化**：AMP 使用的是半精度 _浮點數_，而量化通常指的是轉換為 _整數_。這是最本質的區別。

**總結一下：**

- **PyTorch AMP** 是一種利用半精度浮點數 (FP16) 來加速訓練和推斷並減少記憶體佔用的技術，它仍然在浮點數的範疇內操作。
- **PTQ** 和 **QAT** 是將模型從浮點數轉換為低位元整數 (如 INT8) 的量化技術，旨在大幅縮小模型尺寸並加速推斷，其中 QAT 通常能獲得比 PTQ 更好的精度，但需要重新訓練。

這兩種技術並不互斥，AMP 可以作為訓練階段的加速手段，而量化則可以作為模型部署前的重要優化步驟。理解它們的區別和適用場景，有助於你更有效地優化你的深度學習模型。


以下是 **Post-Training Quantization (PTQ)**、**Quantization-Aware Training (QAT)** 以及 **Mixed Precision Training/Inference** 的詳細比較表，涵蓋原理、優缺點、準確度影響、訓練需求、部署效率等各方面差異：

---

### 🧠 三種量化技術總覽表

|比較項目|**PTQ（後訓練量化）**|**QAT（量化感知訓練）**|**Mixed Precision（混合精度）**|
|---|---|---|---|
|**主要目的**|壓縮模型以加速推論|維持準確率同時加速推論|減少記憶體佔用、加速訓練或推論|
|**量化時間點**|訓練後|訓練過程中即模擬量化|訓練與推論時使用不同精度|
|**量化精度**|一般為 INT8 或更低|INT8, 可自定義|混用 FP16 + FP32（或 BF16）|
|**是否需要重新訓練**|否|是|通常需要|
|**硬體相容性需求**|高（需支援 INT8 推論）|高（需支援 INT8 計算）|中（需支援混合精度）|
|**準確率影響**|易下降，對精度敏感模型不利|可保持接近原始精度|通常不會明顯下降，有時甚至略升|
|**實作複雜度**|低（容易導入）|高（需修改訓練流程）|中（需使用特定框架如 AMP）|
|**推論速度提升**|明顯（INT8加速）|明顯|視硬體支援而定|
|**推論記憶體減少**|顯著（INT8權重小）|顯著|中度（FP16比FP32小）|
|**訓練成本**|低|高（需訓練更多epoch）|中（需支援AMP等工具）|
|**適合場景**|快速部署小型模型、移動端|高精度要求、對性能敏感應用|資源受限、需兼顧速度與準確度|
|**典型使用框架/工具**|PyTorch FX/PTQ, ONNX RT|TensorRT-QAT, PyTorch QAT|NVIDIA AMP, PyTorch autocast|

---

### 📌 說明補充

#### 1. **PTQ（Post-Training Quantization）**

- 優點：不需重新訓練，可快速部署，適用於模型壓縮需求。
    
- 缺點：對於數值敏感或非線性層多的模型（如Attention）會有精度下降。
    

#### 2. **QAT（Quantization-Aware Training）**

- 優點：能在訓練階段模擬量化誤差，使模型在INT8下仍保持高準確度。
    
- 缺點：訓練成本高，實作較複雜，需使用量化模塊與FakeQuant操作。
    

#### 3. **Mixed Precision**

- 優點：能利用FP16/BF16加速，減少顯存佔用，對Transformer類模型特別有效。
    
- 缺點：需硬體支援（如NVIDIA Tensor Core），部分操作仍需FP32來保持穩定性。
    

---

### 📊 推論效能與準確率對比（以Transformer為例）

| 技術                     | 推論速度提升  | 準確率下降（相對FP32） | 額外訓練需求    |
| ---------------------- | ------- | ------------- | --------- |
| PTQ (INT8)             | 約1.5~3倍 | -1% ~ -5%     | 無         |
| QAT (INT8)             | 約1.5~3倍 | <1%           | 需要重新訓練    |
| Mixed Precision (FP16) | 約1.3~2倍 | 幾乎無           | 需使用AMP等工具 |




說得非常好！這幾個問題都是理解 PyTorch AMP 工作方式的關鍵點。我們來逐一詳細解釋：

### 1. PyTorch AMP 是否會改變模型本身的 weight？

**通常情況下，PyTorch AMP 不會永久性地改變模型參數 (weights) 的數據類型。模型的 "主權重" (master weights) 通常仍然以 FP32 格式存儲。**

- **訓練時 (Training)：**
    
    - **FP32 主權重**：優化器 (Optimizer) 負責更新的權重副本通常保持為 FP32。這樣做是為了數值穩定性和精度，避免在多次迭代中因 FP16 的精度限制而累積誤差。
    - **FP16 計算權重**：在 `torch.cuda.amp.autocast` 上下文管理器覆蓋的前向傳播 (forward pass) 和反向傳播 (backward pass) 的特定區域，模型的權重會被 _臨時轉換_ (cast) 為 FP16 進行計算。但這是一個即時的轉換，用於當前的計算步驟。
    - **梯度計算與更新**：梯度計算可能在 FP16 下進行（取決於操作），然後 `GradScaler` 會對損失進行縮放，在反向傳播後對梯度進行反縮放，最後優化器使用這些反縮放後的梯度（通常已轉回 FP32）來更新 FP32 的主權重。
- **推斷時 (Inference)：**
    
    - 如果你加載一個標準的 FP32 模型進行推斷，並使用 `autocast`，那麼權重也同樣會在計算時被臨時轉換為 FP16。原始存儲在模型對象中的權重仍然是 FP32。
    - **例外情況**：你可以手動將整個模型轉換為 FP16 (`model.half()`) 並保存。在這種情況下，模型本身的權重就是 FP16。但這不是 AMP 的標準做法，AMP 的設計是自動和動態地處理精度轉換。如果你這樣做了，那麼在加載模型時權重就是 FP16，`autocast` 可能仍然有用於確保某些操作在 FP32 中運行以保持穩定性，或者如果輸入是 FP32，它會處理轉換。

**總結：AMP 的核心思想是保留 FP32 的主權重以確保精度，並在計算過程中動態地將權重和激活值轉換為 FP16 以獲取性能優勢。**

### 2. 只有在 training 跟 inference 時會將 FP32 轉成 FP16？

是的，基本上是這樣。轉換主要發生在：

1. **訓練循環中的前向傳播和部分反向傳播**：在 `autocast` 啟用的區域內。
2. **推斷過程中的前向傳播**：同樣在 `autocast` 啟用的區域內。

這意味著當模型閒置時，或者在 `autocast` 上下文之外，權重仍然是它們原始的數據類型（通常是 FP32）。轉換是按需、動態地發生的。

### 3. Training 跟 Inference 在 AMP 使用上有什麼不同？

主要的區別在於訓練時需要處理梯度和權重更新，而推斷時則不需要。

- **訓練 (Training) 中的 AMP：**
    
    1. **`torch.cuda.amp.autocast`**：
        - 在前向傳播中，自動將選定的操作 (ops) 的輸入和權重轉換為 FP16 執行。
        - PyTorch 會自動決定哪些操作適合在 FP16 中運行，哪些為了數值穩定性需要保持在 FP32。
        - 損失函數的計算也可能部分在 `autocast` 區域內。
    2. **`torch.cuda.amp.GradScaler`**：
        - **必要性**：由於 FP16 的動態範圍較小，在反向傳播計算梯度時，較小的梯度值很容易下溢 (underflow) 變成零，導致模型無法學習。
        - **工作原理**：
            - 在計算損失的梯度之前，`GradScaler` 會將損失值乘以一個較大的縮放因子 (scaling factor)。
            - 這樣，反向傳播計算出的梯度也會相應地被放大，從而避免下溢。
            - 在優化器更新權重之前，`GradScaler` 會將梯度反縮放回其原始大小（如果沒有溢出 (overflow)）。
            - 如果檢測到梯度溢出 (NaN 或 Inf)，`GradScaler` 會跳過此次權重更新，並在下一次迭代中嘗試一個較小的縮放因子。它會動態調整縮放因子。
    3. **優化器 (Optimizer)**：如前所述，優化器通常更新 FP32 的主權重。
- **推斷 (Inference) 中的 AMP：**
    
    1. **`torch.cuda.amp.autocast`**：
        - 與訓練時類似，在前向傳播中，自動將選定的操作轉換為 FP16 執行以加速並減少記憶體。
    2. **不需要 `GradScaler`**：因為推斷時不計算梯度，也就不存在梯度下溢或溢出的問題，所以不需要 `GradScaler`。
    3. **模型狀態**：模型通常處於評估模式 (`model.eval()`)。

**簡而言之：訓練時是 `autocast + GradScaler`，推斷時通常只需要 `autocast`。**

### 4. 有哪些地方會轉成 FP16？

`torch.cuda.amp.autocast` (或 CPU 上的 `torch.amp.autocast`) 內部維護了一個列表，指明哪些 PyTorch 操作在 AMP 環境下應該以什麼精度運行。這不是由用戶手動指定的，而是由 `autocast` 根據操作的特性自動處理。

通常，以下類型的操作會傾向於轉換到 FP16（在 GPU 上）以利用 Tensor Cores 加速：

- **卷積層 (Convolutions)**：如 `nn.Conv2d`, `nn.Conv3d` 等。
- **線性層/全連接層 (Linear layers / Fully connected layers)**：如 `nn.Linear`。這些本質上是矩陣乘法。
- **矩陣乘法 (Matrix multiplications)**：如 `torch.matmul`, `torch.bmm`。
- **部分循環神經網絡層 (Recurrent Neural Network layers)**：如 `nn.LSTM`, `nn.GRU` 中的某些計算路徑。

而以下類型的操作通常會保持在 FP32，或者即使輸入是 FP16，它們內部也會提升到 FP32 進行計算，以保證數值穩定性：

- **損失函數 (Loss functions)**：如 `nn.CrossEntropyLoss`, `nn.MSELoss` 等，特別是涉及大範圍數值縮減 (reduction) 的部分。
- **批次歸一化 (Batch Normalization)**：`nn.BatchNorm2d` 等，其內部累加統計量通常保持 FP32。
- **Softmax 和 LogSoftmax**：這些操作對數值精度敏感。
- **指數函數 (Exponential function), 對數函數 (Logarithm function)**。
- **大規模的規約操作 (Large reductions)**：如對整個張量求和或求均值。
- **逐點操作 (Pointwise operations)**：如加法、乘法等，它們的輸出類型通常會遵循類型提升規則 (例如，FP16 + FP32 -> FP32)。

**`autocast` 的智能之處在於它會：**

1. **選擇性轉換**：只轉換那些在 FP16 下安全且能帶來性能提升的操作。
2. **類型提升**：當一個操作有多個不同精度的輸入時，會將它們提升到一個共同的、通常是更高精度的類型（如 FP32）來執行操作，以避免精度損失。
3. **緩存轉換後的權重**：為了效率，`autocast` 可能會緩存權重的 FP16 版本，以避免在每次前向傳播中都重複轉換。

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp # 自動混合精度庫

# --- 0. 設定 ---
# 檢查是否有可用的 CUDA 設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用設備: {device}")

# AMP 主要在 CUDA 上使用 FP16 發揮最佳效果
# 如果在 CPU 上，torch.amp.autocast(device_type='cpu', dtype=torch.bfloat16) 是另一種選擇 (如果硬體支援 BF16)
# 此範例主要演示 CUDA 上的 FP16 AMP
use_amp = True if device.type == 'cuda' else False # 只有在 CUDA 上才啟用 AMP
print(f"是否啟用 AMP: {use_amp}")

# --- 1. 定義一個簡單的模型 ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        # 假設輸入圖片大小為 3x32x32，經過兩次 pooling 後變為 64x8x8
        self.fc = nn.Linear(64 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x

# --- 2. 準備訓練所需的組件 ---
model = SimpleCNN(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss() # 損失函數
optimizer = optim.Adam(model.parameters(), lr=0.001) # 優化器

# GradScaler 用於 AMP 訓練，幫助防止梯度下溢
# 僅在 use_amp 為 True 且在 CUDA 上時才有效
scaler = amp.GradScaler(enabled=use_amp)

# --- 3. 模擬訓練數據 ---
# 假設批次大小為 4，圖片為 3 通道，32x32 像素
dummy_inputs = torch.randn(4, 3, 32, 32, device=device)
dummy_labels = torch.randint(0, 10, (4,), device=device)
num_epochs = 3
print_interval = 10 # 每隔多少個批次打印一次信息 (由於這裡只有一個批次，所以每次都會打印)

# --- 4. AMP 訓練過程 ---
print("\n--- 開始 AMP 訓練 ---")
model.train() # 設置為訓練模式
for epoch in range(num_epochs):
    # 在真實訓練中，這裡會有一個 DataLoader 來加載數據
    for batch_idx in range(1): # 假設每個 epoch 只有一個批次的數據
        optimizer.zero_grad()

        # 使用 amp.autocast() 上下文管理器
        # 在 autocast 上下文內，部分操作會自動以 FP16 執行
        with amp.autocast(enabled=use_amp):
            outputs = model(dummy_inputs)
            loss = criterion(outputs, dummy_labels)

        # 使用 scaler.scale() 來縮放損失，然後進行反向傳播
        # 這會產生縮放後的梯度
        scaler.scale(loss).backward()

        # scaler.step() 會先反縮放梯度，然後調用 optimizer.step()
        # 如果梯度沒有溢出 (overflow)，則更新模型參數
        # 如果檢測到梯度溢出 (NaN 或 Inf)，scaler.step() 會跳過參數更新
        scaler.step(optimizer)

        # scaler.update() 更新縮放因子，為下一次迭代做準備
        scaler.update()

        if batch_idx % print_interval == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/1], Loss: {loss.item():.4f}")
            if use_amp:
                print(f"    GradScaler scale: {scaler.get_scale():.2f}")

print("--- AMP 訓練完成 ---")

# --- 5. AMP 推斷過程 ---
print("\n--- 開始 AMP 推斷 ---")
model.eval() # 設置為評估模式
# 準備推斷用的數據
dummy_inference_input = torch.randn(1, 3, 32, 32, device=device)

with torch.no_grad(): # 推斷時不需要計算梯度
    # 同樣使用 amp.autocast() 進行混合精度推斷
    # 注意：推斷時不需要 GradScaler
    with amp.autocast(enabled=use_amp):
        predictions = model(dummy_inference_input)
        predicted_class = torch.argmax(predictions, dim=1)

    print(f"輸入數據維度: {dummy_inference_input.shape}")
    print(f"預測輸出 (logits) 維度: {predictions.shape}")
    print(f"預測輸出 (logits) 數據類型: {predictions.dtype}") # 在 autocast 內，輸出可能是 FP16
    print(f"預測類別: {predicted_class.item()}")

# 如果希望推斷的最終輸出是 FP32 (例如，進行後續非 AMP 操作)
# 可以將 autocast 的結果轉換回來，或者在 autocast 之外進行最後的轉換
if use_amp and predictions.dtype == torch.float16:
    predictions_fp32 = predictions.float()
    print(f"轉換為 FP32 後的預測輸出數據類型: {predictions_fp32.dtype}")

print("--- AMP 推斷完成 ---")

# 額外提示:
# 1. 模型保存與加載：
#    通常情況下，你會保存模型的 state_dict。AMP 訓練時，optimizer 維護的是 FP32 的主權重，
#    所以 `model.state_dict()` 保存的是 FP32 權重。
#    torch.save(model.state_dict(), "model_amp_trained.pth")
#    # 加載時:
#    # loaded_model = SimpleCNN(num_classes=10)
#    # loaded_model.load_state_dict(torch.load("model_amp_trained.pth"))
#    # loaded_model.to(device)

# 2. 檢查性能：
#    可以使用 torch.utils.benchmark 或簡單的 time.time() 來比較啟用和禁用 AMP 時的訓練/推斷速度。
#    但請注意，對於非常小的模型和數據，AMP 的開銷可能使其看起來沒有加速，甚至更慢。
#    AMP 的優勢在於較大的模型和可以利用 Tensor Cores 的 NVIDIA GPU。

# 3. CPU 上的 AMP:
#    如果在 CPU 上，你可以使用 `device_type='cpu'` 和 `dtype=torch.bfloat16` (如果 CPU 支持 bfloat16)。
#    例如: `with torch.amp.autocast(device_type='cpu', dtype=torch.bfloat16, enabled=use_cpu_amp):`
#    CPU 上的 AMP 通常使用 BFloat16 而不是 Float16。
```