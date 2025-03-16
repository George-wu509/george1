

|                               |     |
| ----------------------------- | --- |
| [[### CUDA Programming 詳細介紹]] |     |
| [[###CUDA Programming 是什麼？]]  |     |
|                               |     |
|                               |     |
|                               |     |


### CUDA Programming 詳細介紹

CUDA (Compute Unified Device Architecture) 是由 NVIDIA 開發的一種並行計算平台和 API，允許開發者利用 GPU 來加速計算密集型應用。CUDA 主要基於 **SIMT (Single Instruction, Multiple Thread)** 架構，每個 GPU 核心可以同時執行相同的指令，但可以針對不同的數據進行操作。

CUDA 的主要組件包括：

- **Threads (線程)：** 單個 GPU 核心執行的最小單位。
- **Blocks (區塊)：** 一組 GPU 線程。
- **Grids (網格)：** 一組 Blocks，定義了 CUDA 核心的運行範圍。
- **Shared Memory (共享記憶體)：** 可在同一個 Block 內共享數據的記憶體區域，比 Global Memory 更快。
- **Global Memory (全域記憶體)：** 所有線程都能訪問的記憶體，但訪問速度較慢。

---

## 如何使用 CUDA Programming 加速 AI Segmentation Model

假設我們有一個 AI Segmentation Model，在推理過程中，我們希望透過 CUDA 提高模型運行速度。主要有以下幾種加速方式：

1. **使用 PyTorch CUDA Tensor 加速推理**
2. **使用 CUDA Kernel 編寫 GPU 加速函數**
3. **利用 cuDNN 和 TensorRT 進一步優化**

### **方法 1：使用 PyTorch CUDA Tensor 加速推理**

如果你的模型是基於 PyTorch，最簡單的方式就是將模型和數據轉換到 CUDA：
```python
import torch
import torchvision.models as models

# 加載 AI segmentation model (以 DeepLabV3 為例)
model = models.segmentation.deeplabv3_resnet50(pretrained=True)

# 將模型移動到 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# 測試輸入圖像
input_tensor = torch.randn(1, 3, 512, 512).to(device)  # 隨機生成一張圖片
output = model(input_tensor)

# 顯示 segmentation mask
segmentation_mask = output['out'].argmax(dim=1).squeeze().cpu().numpy()
print(segmentation_mask.shape)  # (512, 512)

```

這種方式可以讓 PyTorch 自動將運算轉換為 CUDA 運行，適合 **推理加速**。

---

### **方法 2：使用 CUDA Kernel 編寫 GPU 加速函數**

有時候，我們需要在 CUDA 上自定義運算，例如加速後處理或某些自訂運算。這時候可以使用 `torch.cuda` 來編寫 CUDA 核心函數。例如，我們可以編寫一個簡單的 **2D 卷積操作**：
```python
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# 使用 CUDA Kernel 進行卷積運算
cuda_code = """
extern "C" __global__ void conv2d_cuda(float* input, float* kernel, float* output, 
                                       int H, int W, int KH, int KW) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= H || y >= W) return;

    float sum = 0.0;
    for (int i = 0; i < KH; ++i) {
        for (int j = 0; j < KW; ++j) {
            int xi = x + i - KH / 2;
            int yj = y + j - KW / 2;
            if (xi >= 0 && xi < H && yj >= 0 && yj < W) {
                sum += input[xi * W + yj] * kernel[i * KW + j];
            }
        }
    }
    output[x * W + y] = sum;
}
"""

# 編譯 CUDA 內核函數
cuda_module = load(name="conv2d_cuda", sources=[], extra_cuda_sources=[cuda_code])

# 測試 CUDA 內核函數
H, W = 512, 512
KH, KW = 3, 3
input_tensor = torch.randn(H, W, device="cuda")
kernel = torch.ones(KH, KW, device="cuda") / (KH * KW)  # 平均池化
output_tensor = torch.zeros(H, W, device="cuda")

# 佈局 CUDA 執行參數
threads_per_block = (16, 16)
blocks_per_grid = ((H + threads_per_block[0] - 1) // threads_per_block[0],
                   (W + threads_per_block[1] - 1) // threads_per_block[1])

# 調用 CUDA 核心函數
cuda_module.conv2d_cuda(input_tensor.data_ptr(), kernel.data_ptr(), output_tensor.data_ptr(),
                        H, W, KH, KW, block=threads_per_block, grid=blocks_per_grid)

# 取回輸出
output_tensor = output_tensor.cpu().numpy()
print(output_tensor.shape)  # (512, 512)

```

這種方式適用於 **自定義 CUDA 運算**，可以加速特定的數學運算，例如卷積、池化或後處理步驟。

---

### **方法 3：使用 cuDNN 和 TensorRT 進一步優化**

如果你希望進一步優化 AI segmentation model，NVIDIA 提供 **cuDNN (CUDA Deep Neural Network Library)** 和 **TensorRT** 來加速推理。

使用 TensorRT 可以讓 PyTorch 模型加速，例如：
```python
import torch
import torch_tensorrt

# 轉換 PyTorch 模型為 TensorRT 加速版本
trt_model = torch_tensorrt.compile(model, inputs=[torch_tensorrt.Input((1, 3, 512, 512), dtype=torch.float32)], enabled_precisions={torch.float16})

# 進行推理
output_trt = trt_model(input_tensor)
print(output_trt['out'].shape)  # (1, 21, 512, 512) for DeepLabV3 segmentation

```

這種方法適用於 **生產環境的 AI 推理加速**，可以獲得極致效能。

---

## **其他兩個例子**

### **例子 1：使用 CUDA 加速矩陣乘法**
```python
import torch

# 創建兩個隨機矩陣
A = torch.randn(1000, 1000).cuda()
B = torch.randn(1000, 1000).cuda()

# 在 GPU 上計算矩陣乘法
C = torch.matmul(A, B)

print(C.shape)  # (1000, 1000)

```

這種方式適用於 **大型數據處理**，如神經網路前向傳播計算。

---

### **例子 2：使用 CUDA 加速影像銳化**
```python
import torch
import torch.nn.functional as F

# 定義影像銳化的卷積核
sharpen_kernel = torch.tensor([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]], dtype=torch.float32, device="cuda").unsqueeze(0).unsqueeze(0)

# 載入影像並轉換為 Tensor
image = torch.randn(1, 1, 512, 512, device="cuda")  # 假設為灰階影像

# 執行卷積
sharpened_image = F.conv2d(image, sharpen_kernel, padding=1)

print(sharpened_image.shape)  # (1, 1, 512, 512)

```

這種方式適用於 **影像處理加速**，如邊緣檢測或影像增強。

---

## **結論**

使用 CUDA 可以顯著提升 AI segmentation model 的推理速度，方法包括：

1. **PyTorch CUDA Tensor 加速**
2. **自定義 CUDA Kernel**
3. **TensorRT 進一步優化**

並且 CUDA 也適用於矩陣計算、影像處理等各種應用場景。🚀




### CUDA Programming 是什麼？

CUDA Programming **不僅僅** 指 CUDA Kernel 編寫 GPU 加速函數，還包括使用 CUDA 相關 API（如 `CUDA C/C++`、`cuDNN`、`TensorRT`）來優化 GPU 運算。  
PyTorch CUDA Tensor 加速推理 **確實也是 CUDA Programming 的一部分**，但它的底層仍然依賴於 CUDA Kernel。

---

## **PyTorch CUDA Tensor 加速推理是否內部原理也是用 CUDA Kernel？**

### ✅ **是的，PyTorch CUDA Tensor 內部確實使用 CUDA Kernel**

當你使用 PyTorch 的 `model.to("cuda")` 或 `tensor.to("cuda")` 時，PyTorch 會：

1. **將 Tensor 存放到 GPU 記憶體**（Global Memory）
2. **調用已優化的 CUDA 核心函數**（這些函數來自 NVIDIA cuBLAS、cuDNN 或 PyTorch 本身的 CUDA Kernel）
3. **在 GPU 上執行矩陣運算、卷積等操作**
4. **將結果返回 CPU 或繼續在 GPU 處理**

你可以檢查 PyTorch 內部是否使用 CUDA：

python

複製編輯

`import torch print(torch.cuda.is_available())  # True 表示 PyTorch 已支援 CUDA print(torch.backends.cudnn.enabled)  # True 表示 cuDNN 已啟用`

舉例來說，當你這樣寫：

python

複製編輯

`A = torch.randn(1000, 1000).cuda() B = torch.randn(1000, 1000).cuda() C = torch.matmul(A, B)  # 這裡 PyTorch 會自動調用 CUDA 核心`

PyTorch **底層會調用 cuBLAS (CUDA Basic Linear Algebra Subprograms) 來執行矩陣乘法**，這是 NVIDIA 提供的高效能 CUDA 核心函數。

你可以開啟 **CUDA Profiling** 來查看 PyTorch 實際使用的 CUDA 核心：

python

複製編輯

`import torch with torch.autograd.profiler.profile(use_cuda=True) as prof:     C = torch.matmul(A, B) print(prof)  # 這裡會列出 CUDA Kernel 調用記錄`

總結來說：

- **PyTorch CUDA Tensor** 是 **封裝好的 CUDA 核心**，你不需要自己寫 CUDA Kernel，就能讓運算在 GPU 上執行。
- **PyTorch 內部會根據運算類型調用最佳化的 CUDA Kernel**（例如 `cuDNN` 用於卷積，`cuBLAS` 用於矩陣運算）。

---

## **除了 CUDA Kernel，CUDA 還有哪些加速方式？**

CUDA 提供許多 **不同層級** 的加速技術，不只是自己寫 CUDA Kernel，還可以用 CUDA Library、Graph、TensorRT、Multi-GPU 等技術來進一步優化：

### **1. 使用 cuBLAS/cuDNN 等 CUDA Library 加速**

NVIDIA 提供了許多 CUDA 優化的函式庫：

- **cuBLAS**：用於矩陣計算（如 `torch.matmul`）。
- **cuDNN**：用於深度學習的卷積、池化等運算（如 `torch.nn.Conv2d`）。
- **cuSPARSE**：用於稀疏矩陣計算。
- **cuFFT**：用於 FFT 變換。

你可以直接調用這些 CUDA Library，或者透過 PyTorch 來使用：

python

複製編輯

`import torch  # 這裡 PyTorch 內部會調用 cuDNN 來加速 Conv2D conv = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3).cuda() input_tensor = torch.randn(1, 3, 224, 224).cuda() output = conv(input_tensor)`

---

### **2. CUDA Graph (提高計算效率)**

- 適用於 **重複性計算**，如 batch inference，能夠 **減少 Kernel Launch 開銷**。
- **原理**：將運算變成 **計算圖 (Graph Execution)**，讓 GPU 不用每次都重新解析 Kernel 依賴關係。

python

複製編輯

`import torch  # 準備模型和輸入 model = torch.nn.Conv2d(3, 64, 3).cuda() input_tensor = torch.randn(1, 3, 224, 224, device="cuda")  # 記錄計算圖 stream = torch.cuda.Stream() with torch.cuda.stream(stream):     g = torch.cuda.CUDAGraph()     with torch.cuda.graph(g):         output = model(input_tensor)`

這可以 **顯著減少推理延遲**。

---

### **3. 使用 TensorRT 進一步加速推理**

TensorRT 是 NVIDIA 提供的 **深度學習推理優化框架**，可以將 PyTorch 模型轉換成高效能 CUDA Kernel：

python

複製編輯

`import torch import torch_tensorrt  model = torch.nn.Conv2d(3, 64, 3).cuda() input_tensor = torch.randn(1, 3, 224, 224, device="cuda")  trt_model = torch_tensorrt.compile(model, inputs=[torch_tensorrt.Input(input_tensor.shape, dtype=torch.float32)]) output = trt_model(input_tensor)`

TensorRT 會 **自動融合運算 (Operator Fusion)**，例如把 Conv2D 和 BatchNorm 合併成一個 Kernel，加速計算。

---

### **4. Multi-GPU 分佈式運算**

當 GPU 記憶體不夠用時，可以用 **Data Parallel (DP)** 或 **Distributed Data Parallel (DDP)**：

python

複製編輯

`from torch.nn.parallel import DistributedDataParallel as DDP model = DDP(model, device_ids=[0,1])`

這種方法能夠 **同時使用多張 GPU 加速計算**。

---

### **5. 混合精度 (FP16) 計算**

可以使用 **AMP (Automatic Mixed Precision)** 來減少記憶體佔用並加速計算：

python

複製編輯

`scaler = torch.cuda.amp.GradScaler()  with torch.cuda.amp.autocast():     output = model(input_tensor)  scaler.scale(loss).backward() scaler.step(optimizer) scaler.update()`

這可以 **提升推理速度 2 倍**，特別適用於 Transformer 或 CNN 模型。

---

### **6. 使用 Shared Memory (共享記憶體) 提高計算效率**

當你寫 CUDA Kernel 時，可以使用 **共享記憶體 (Shared Memory)** 來加速讀取：

cpp

複製編輯

`__global__ void shared_memory_example(float *input, float *output) {     __shared__ float smem[1024];  // 共享記憶體      int tid = threadIdx.x;     smem[tid] = input[tid];  // 把數據放入共享記憶體      __syncthreads();  // 同步所有線程      output[tid] = smem[tid] * 2;  // 計算並輸出 }`

這樣 **記憶體存取會比 Global Memory 更快**，提升計算效能。

---

## **總結**

CUDA Programming 不僅僅是自己寫 CUDA Kernel，還包括：

|**技術**|**是否內部用 CUDA Kernel**|**優點**|
|---|---|---|
|**PyTorch CUDA Tensor**|✅|直接使用 GPU 運算|
|**CUDA Kernel (自定義運算)**|✅|完全控制 CUDA 運算|
|**cuBLAS/cuDNN (NVIDIA 函式庫)**|✅|內建優化的高效能 Kernel|
|**CUDA Graph**|✅|減少 Kernel 啟動開銷|
|**TensorRT**|✅|自動優化推理|
|**Multi-GPU (DDP)**|✅|提高計算規模|
|**AMP 混合精度 (FP16)**|✅|降低記憶體使用，加速推理|
|**CUDA Shared Memory**|✅|更快的記憶體存取|

這些技術 **都屬於 CUDA Programming**，你可以根據自己的需求選擇合適的方法來提升 segmentation model 的效能 🚀。



Reference:
CUDA快速入门-01-基础语法概念 - 香草美人的文章 - 知乎
https://zhuanlan.zhihu.com/p/16641687170