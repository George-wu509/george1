
詳細介紹 PyTorch 中 `torch.cuda` 模組的一些常用功能，並提供簡單的程式碼範例。`torch.cuda` 模組提供了與 NVIDIA CUDA 和 GPU 交互的介面，讓 PyTorch 能夠利用 GPU 的強大並行計算能力來加速深度學習任務。

**常用功能：**

1. **檢查 CUDA 是否可用 (`torch.cuda.is_available()`):** 這個函數用於檢查你的系統是否安裝了 NVIDIA 驅動程式，並且 PyTorch 是否能夠找到可用的 CUDA 裝置。
    
    Python
    
    ```
    import torch
    
    if torch.cuda.is_available():
        print("CUDA 可用！")
    else:
        print("CUDA 不可用，將使用 CPU。")
    ```
    
2. **獲取可用的 CUDA 裝置數量 (`torch.cuda.device_count()`):** 如果你的系統有多個 GPU，這個函數可以返回可用的 GPU 數量。
    
    Python
    
    ```
    import torch
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"找到 {num_gpus} 個 CUDA 裝置。")
    else:
        print("CUDA 不可用。")
    ```
    
3. **獲取當前使用的 CUDA 裝置索引 (`torch.cuda.current_device()`):** 這個函數返回當前 PyTorch 正在使用的 GPU 的索引 (從 0 開始)。預設情況下，PyTorch 會使用索引為 0 的 GPU。
    
    Python
    
    ```
    import torch
    
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        print(f"當前使用的 CUDA 裝置索引：{current_device}")
    else:
        print("CUDA 不可用。")
    ```
    
4. **獲取 CUDA 裝置的名稱 (`torch.cuda.get_device_name(device_index)`):** 這個函數可以根據給定的裝置索引，返回該 GPU 的名稱。
    
    Python
    
    ```
    import torch
    
    if torch.cuda.is_available():
        device_index = 0
        device_name = torch.cuda.get_device_name(device_index)
        print(f"CUDA 裝置 {device_index} 的名稱：{device_name}")
    else:
        print("CUDA 不可用。")
    ```
    
5. **設定當前使用的 CUDA 裝置 (`torch.cuda.set_device(device_index)`):** 如果你有多個 GPU，可以使用這個函數來設定當前 PyTorch 要使用的 GPU。
    
    Python
    
    ```
    import torch
    
    if torch.cuda.is_available():
        device_index = 1  # 假設你有至少兩個 GPU
        try:
            torch.cuda.set_device(device_index)
            print(f"已設定使用 CUDA 裝置：{device_index} ({torch.cuda.get_device_name(device_index)})")
        except RuntimeError as e:
            print(f"設定 CUDA 裝置 {device_index} 失敗：{e}")
    else:
        print("CUDA 不可用。")
    ```
    
6. **將 tensors 移動到 CUDA 裝置 (`tensor.to(device)` 或 `tensor.cuda()`):** 這是使用 GPU 加速的關鍵步驟。你需要將你的模型參數和輸入數據 tensors 移動到 GPU 記憶體中。
    
    Python
    
    ```
    import torch
    
    if torch.cuda.is_available():
        device = torch.device("cuda")  # 更通用的寫法
        # device = torch.device("cuda:0") # 指定第一個 GPU
    
        # 創建一個 CPU 上的 tensor
        cpu_tensor = torch.randn(3, 4)
        print(f"CPU tensor:\n{cpu_tensor}")
        print(f"CPU tensor 的裝置：{cpu_tensor.device}")
    
        # 將 tensor 移動到 GPU
        gpu_tensor = cpu_tensor.to(device)
        # 或者可以使用更簡潔的 .cuda() 方法（移動到當前設定的 GPU）
        # gpu_tensor = cpu_tensor.cuda()
        print(f"\nGPU tensor:\n{gpu_tensor}")
        print(f"GPU tensor 的裝置：{gpu_tensor.device}")
    
        # 將 tensor 從 GPU 移動回 CPU
        cpu_tensor_back = gpu_tensor.to("cpu")
        print(f"\n回到 CPU 的 tensor:\n{cpu_tensor_back}")
        print(f"回到 CPU 的 tensor 的裝置：{cpu_tensor_back.device}")
    else:
        print("CUDA 不可用。")
    ```
    
7. **檢查 tensor 是否在 CUDA 上 (`tensor.is_cuda`):** 這個屬性可以檢查一個 tensor 是否儲存在 GPU 記憶體中。
    
    Python
    
    ```
    import torch
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        tensor = torch.randn(2, 2).to(device)
        print(f"Tensor 是否在 CUDA 上：{tensor.is_cuda}")
    
        cpu_tensor = torch.randn(2, 2)
        print(f"CPU Tensor 是否在 CUDA 上：{cpu_tensor.is_cuda}")
    else:
        print("CUDA 不可用。")
    ```
    
8. **CUDA 記憶體管理 (`torch.cuda.empty_cache()` 和 `torch.cuda.memory_allocated(device_index=None)` 等):** PyTorch 會自動管理 GPU 記憶體，但在某些情況下，你可能需要手動清理不再使用的記憶體。
    
    - `torch.cuda.empty_cache()`: 釋放 CUDA 緩存中不再被 tensors 使用的記憶體，有助於避免 CUDA out of memory 錯誤。
    - `torch.cuda.memory_allocated(device_index=None)`: 返回指定裝置已分配的記憶體量 (以位元組為單位)。如果 `device_index` 為 `None`，則返回當前裝置的記憶體使用量。
    - `torch.cuda.memory_reserved(device_index=None)`: 返回指定裝置 PyTorch CUDA 記憶體管理員保留的記憶體量 (以位元組為單位)。這個值通常大於 `memory_allocated`。
    
    Python
    
    ```
    import torch
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        tensor = torch.randn(1024, 1024).to(device)
        allocated_before = torch.cuda.memory_allocated(device)
        print(f"分配記憶體前：{allocated_before / (1024**2):.2f} MB")
    
        del tensor  # 釋放 tensor 的引用
        torch.cuda.empty_cache()  # 清空緩存
        allocated_after = torch.cuda.memory_allocated(device)
        print(f"清空緩存後分配記憶體：{allocated_after / (1024**2):.2f} MB")
    else:
        print("CUDA 不可用。")
    ```
    

**簡單範例：在 GPU 上執行基本運算**

Python

```
import torch
import time

# 檢查 CUDA 是否可用，並設定裝置
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"使用裝置：{torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("CUDA 不可用，使用 CPU。")

# 創建兩個隨機 tensors
size = (1000, 1000)
cpu_tensor1 = torch.randn(size)
cpu_tensor2 = torch.randn(size)

# 將 tensors 移動到指定的裝置
gpu_tensor1 = cpu_tensor1.to(device)
gpu_tensor2 = cpu_tensor2.to(device)

# 在 CPU 上執行加法並計時
start_time = time.time()
cpu_result = cpu_tensor1 + cpu_tensor2
cpu_time = time.time() - start_time
print(f"CPU 加法時間：{cpu_time:.4f} 秒")

# 在 GPU 上執行加法並計時
start_time = time.time()
gpu_result = gpu_tensor1 + gpu_tensor2
torch.cuda.synchronize()  # 等待 GPU 完成操作
gpu_time = time.time() - start_time
print(f"GPU 加法時間：{gpu_time:.4f} 秒")

if torch.cuda.is_available():
    print(f"GPU 加速比：{cpu_time / gpu_time:.2f} 倍")
```

這個簡單的範例展示了如何將 tensors 移動到 GPU 上，並比較在 CPU 和 GPU 上執行相同運算所需的時間。通常情況下，對於大型的張量運算，GPU 的速度會遠遠快於 CPU。

希望這些解釋和範例能夠幫助你理解 `torch.cuda` 模組的常用功能以及如何在 PyTorch 中利用 GPU 加速你的深度學習工作流程。