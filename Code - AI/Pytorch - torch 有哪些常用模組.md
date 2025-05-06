
|               |                                     |
| ------------- | ----------------------------------- |
| Core          | torch (核心模組)                        |
|               | torch.nn (神經網路模組)                   |
|               | torch.nn.init (參數初始化模組)             |
|               | torch.nn.functional (函數式神經網路模組)     |
|               |                                     |
| Data          | torch.utils.data (數據載入模組)           |
|               |                                     |
| Model setting | torch.optim (優化模組)                  |
|               | torch.autograd (自動微分模組)             |
|               | torch.save` 和 `torch.load (模型保存與載入) |
|               |                                     |
| Performance   | torch.cuda (CUDA 支持模組)              |
|               | torch.jit (Just-In-Time 編譯器)        |
|               | torch.distributed (分散式訓練模組)         |
|               | torch.multiprocessing (多進程支持模組)     |
|               |                                     |
| Others        | torch.hub (預訓練模型中心)                 |
|               | torch.onnx (ONNX 導出模組)              |

**PyTorch 常用模組列表：**

1. **`torch` (核心模組):** 包含 PyTorch 的基本資料結構 (如 tensors) 和數學運算函數。
2. **`torch.nn` (神經網路模組):** 包含用於構建神經網路的各種層、損失函數和激活函數。
3. **`torch.optim` (優化模組):** 包含各種優化演算法，用於訓練神經網路模型。
4. **`torch.utils.data` (數據載入模組):** 提供用於載入和處理數據的工具，如 `Dataset` 和 `DataLoader`。
5. **`torch.cuda` (CUDA 支持模組):** 提供與 NVIDIA CUDA 和 GPU 交互的功能。
6. **`torch.autograd` (自動微分模組):** PyTorch 的核心自動微分引擎，用於計算梯度。
7. **`torch.save` 和 `torch.load` (模型保存與載入):** 用於保存和載入訓練好的模型。
8. **`torch.nn.init` (參數初始化模組):** 提供各種權重和偏置的初始化方法。
9. **`torch.nn.functional` (函數式神經網路模組):** 包含 `torch.nn` 中層的函數式版本，以及其他有用的函數。
10. **`torch.hub` (預訓練模型中心):** 提供載入預訓練模型的便捷方式。
11. **`torch.jit` (Just-In-Time 編譯器):** 用於將 PyTorch 模型轉換為可序列化和優化的形式。
12. **`torch.distributed` (分散式訓練模組):** 用於在多個機器或 GPU 上進行分散式訓練。
13. **`torch.multiprocessing` (多進程支持模組):** 提供類似 Python `multiprocessing` 的功能，但與 PyTorch tensors 共享記憶體。
14. **`torch.onnx` (ONNX 導出模組):** 用於將 PyTorch 模型導出為 ONNX (Open Neural Network Exchange) 格式。

接下來，我們將針對其中幾個最常用的模組提供具體的簡單範例。

**1. `torch` (核心模組):**

- **功能:** 創建和操作 tensors。
    
- **範例:**
    
    Python
    
    ```
    import torch
    
    # 創建一個 3x3 的浮點數 tensor
    x = torch.randn(3, 3)
    print(f"Tensor x:\n{x}")
    
    # 執行基本數學運算
    y = x + 2
    print(f"\nTensor y (x + 2):\n{y}")
    
    # 改變 tensor 的形狀
    z = x.view(9)
    print(f"\nReshaped tensor z:\n{z}")
    ```
    

**2. `torch.nn` (神經網路模組):**

- **功能:** 定義神經網路層和模型。
    
- **範例:**
    
    Python
    
    ```
    import torch.nn as nn
    
    # 定義一個簡單的線性層
    linear = nn.Linear(in_features=10, out_features=5)
    
    # 創建一個輸入 tensor
    input_tensor = torch.randn(1, 10)
    
    # 通過線性層
    output_tensor = linear(input_tensor)
    print(f"線性層的輸出形狀：{output_tensor.shape}")
    
    # 定義一個包含多個層的簡單模型
    class SimpleNet(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(SimpleNet, self).__init__()
            self.linear1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(hidden_size, output_size)
    
        def forward(self, x):
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
            return x
    
    model = SimpleNet(input_size=10, hidden_size=20, output_size=2)
    dummy_input = torch.randn(1, 10)
    model_output = model(dummy_input)
    print(f"\n簡單模型的輸出形狀：{model_output.shape}")
    ```
    

**3. `torch.optim` (優化模組):**

- **功能:** 實現各種優化演算法來更新模型參數。
    
- **範例:**
    
    Python
    
    ```
    import torch.optim as optim
    import torch.nn as nn
    
    # 創建一個簡單的模型
    model = nn.Linear(5, 2)
    
    # 選擇 Adam 優化器，並設定學習率
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # 模擬計算損失
    criterion = nn.MSELoss()
    input_data = torch.randn(1, 5)
    target_data = torch.randn(1, 2)
    output_data = model(input_data)
    loss = criterion(output_data, target_data)
    print(f"初始損失：{loss.item():.4f}")
    
    # 反向傳播計算梯度
    loss.backward()
    
    # 執行一次優化步驟來更新模型參數
    optimizer.step()
    
    # 再次計算損失（損失應該會減少）
    output_data_after_step = model(input_data)
    loss_after_step = criterion(output_data_after_step, target_data)
    print(f"更新參數後的損失：{loss_after_step.item():.4f}")
    ```
    

**4. `torch.utils.data` (數據載入模組):**

- **功能:** 方便地載入和批次處理數據。
    
- **範例:**
    
    Python
    
    ```
    from torch.utils.data import Dataset, DataLoader
    import torch
    
    # 創建一個自定義的 Dataset
    class SimpleDataset(Dataset):
        def __init__(self, num_samples):
            self.num_samples = num_samples
            self.data = torch.randn(num_samples, 10)
            self.labels = torch.randint(0, 2, (num_samples,))
    
        def __len__(self):
            return self.num_samples
    
        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]
    
    # 創建 Dataset 的實例
    dataset = SimpleDataset(num_samples=100)
    
    # 創建 DataLoader 來批次處理數據
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # 迭代遍歷 DataLoader
    for batch_idx, (data, labels) in enumerate(dataloader):
        print(f"批次索引: {batch_idx}, 數據形狀: {data.shape}, 標籤: {labels}")
        if batch_idx == 0:
            break
    ```
    

**5. `torch.cuda` (CUDA 支持模組):**

- **功能:** 使 PyTorch 能夠在 NVIDIA GPU 上運行。
    
- **範例:** (更詳細的範例請參考之前的回答)
    
    Python
    
    ```
    import torch
    
    # 檢查 CUDA 是否可用
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA 可用，使用的裝置：{torch.cuda.get_device_name(0)}")
    
        # 將 tensor 移動到 GPU
        tensor_cpu = torch.randn(3, 3)
        tensor_gpu = tensor_cpu.to(device)
        print(f"CPU tensor 的裝置：{tensor_cpu.device}")
        print(f"GPU tensor 的裝置：{tensor_gpu.device}")
    else:
        print("CUDA 不可用，將使用 CPU。")
        device = torch.device("cpu")
    ```
    

**6. `torch.autograd` (自動微分模組):**

- **功能:** 自動計算 tensors 的梯度，用於反向傳播。
    
- **範例:**
    
    Python
    
    ```
    import torch
    
    # 創建一個需要計算梯度的 tensor
    x = torch.randn(2, 2, requires_grad=True)
    print(f"Tensor x:\n{x}")
    
    # 執行一些運算
    y = x + 2
    z = y * y
    out = z.mean()
    print(f"\nOutput tensor:\n{out}")
    
    # 計算梯度
    out.backward()
    
    # 查看 x 的梯度
    print(f"\nx 的梯度:\n{x.grad}")
    ```
    

這只是 PyTorch 中一些最常用模組的簡要介紹和範例。PyTorch 提供了非常豐富的功能，可以幫助你構建、訓練和部署各種深度學習模型。