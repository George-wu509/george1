
|                                       |     |
| ------------------------------------- | --- |
| [[###Custom CUDA kernel development]] |     |
| [[###實作一個自訂的CUDA核心（Kernel）]]          |     |
|                                       |     |

### Custom CUDA kernel development

詳細解釋一下「自訂 CUDA 核心開發」（Custom CUDA kernel development）這個觀念，以及如何為專門的 AI 和深度學習工作負載編寫和優化自訂 CUDA 核心。

**什麼是 CUDA？**

CUDA (Compute Unified Device Architecture) 是 NVIDIA 開發的平行計算平台和程式設計模型。它允許開發人員利用 NVIDIA GPU（圖形處理單元）強大的平行處理能力來加速計算密集型應用，遠超傳統 CPU 的處理速度。

**什麼是 CUDA 核心（Kernel）？**

在 CUDA 的術語中，「核心」（Kernel）是一個特殊的 C/C++ 函式。當你從主機端（CPU）呼叫一個核心函式時，它會在 GPU 上由大量的執行緒（Threads）同時並行執行。每個執行緒執行相同的程式碼，但通常處理不同的資料。

**什麼是自訂 CUDA 核心開發？**

通常，在使用 GPU 進行 AI/DL 時，我們會依賴高度優化的函式庫，例如 cuDNN（用於深度神經網路）、cuBLAS（用於基本線性代數運算）等。這些函式庫提供了常見操作（如卷積、矩陣乘法、池化）的高效能實作。

然而，在某些情況下，這些標準函式庫可能無法滿足特定的需求，或者無法達到極致的效能。**自訂 CUDA 核心開發** 就是指**不依賴現成的函式庫，而是直接使用 CUDA C/C++ 程式語言，為特定的計算任務編寫自己的 GPU 核心函式**。

**為什麼要為 AI/DL 工作負載開發自訂核心？**

1. **極致效能優化 (Performance Optimization):**
    
    - 標準函式庫為了通用性，可能無法針對 _你特定模型_ 的 _特定層_ 或 _特定資料維度/特性_ 做到最完美的優化。
    - 自訂核心可以讓你精確控制記憶體存取模式、執行緒配置、指令級並行等，榨乾 GPU 的每一分效能，突破標準函式庫的瓶頸。例如，針對非常規的卷積核大小、特殊的稀疏模式或融合多個操作。
2. **實現新穎的操作或層 (Novel Operations/Layers):**
    
    - 當研究人員提出新的神經網路層、新的活化函數、或創新的演算法時，這些操作往往不存在於標準函式庫中。
    - 開發自訂核心是將這些新想法付諸實踐，並在 GPU 上高效執行的唯一途徑。
3. **記憶體效率優化 (Memory Efficiency):**
    
    - 標準函式庫的操作可能產生中間結果，需要額外的記憶體讀寫。
    - 透過「核心融合」（Kernel Fusion），可以將多個連續的操作合併到一個自訂核心中，減少讀寫全域記憶體的次數，將中間結果直接保存在速度更快的共享記憶體或暫存器中，大幅降低記憶體頻寬壓力並提升效能。
4. **減少框架開銷 (Reduced Framework Overhead):**
    
    - 有時，高階框架（如 PyTorch, TensorFlow）的抽象層會帶來一定的效能開銷。直接呼叫自訂 CUDA 核心可以繞過部分開銷，實現更直接的控制。
5. **研究與探索 (Research & Development):**
    
    - 開發自訂核心有助於深入理解底層硬體架構和並行計算原理，對於探索新的模型架構或硬體加速技術至關重要。

**如何實作自訂 CUDA 核心開發？**

這是一個複雜的過程，需要深入理解 GPU 架構和並行程式設計。以下是主要步驟和關鍵概念：

**1. 環境準備：**

- **硬體：** 一塊支援 CUDA 的 NVIDIA GPU。
- **軟體：**
    - NVIDIA CUDA Toolkit：包含 NVCC 編譯器、函式庫、開發工具（如 Nsight Systems, Nsight Compute）、驅動程式。
    - C/C++ 編譯器（如 GCC, MSVC）。
    - （可選）Python 環境及 AI 框架（PyTorch, TensorFlow）：用於整合和呼叫自訂核心。

**2. 理解 CUDA 程式設計模型關鍵概念：**

- **核心函式 (Kernel Function):** 使用 `__global__` 關鍵字修飾的函式，表示它將在 GPU 上執行，並可由 CPU 呼叫。
- **執行緒、區塊、網格 (Threads, Blocks, Grid):**
    - **執行緒 (Thread):** CUDA 執行的最小單位。大量執行緒同時執行核心函式。
    - **區塊 (Block):** 一組執行緒，可以彼此協作（透過共享記憶體和同步）。同一個區塊內的執行緒保證在同一個 Streaming Multiprocessor (SM) 上執行。
    - **網格 (Grid):** 一組區塊，構成一次核心啟動的整體。
    - **內建變數：** `threadIdx` (執行緒在區塊內的索引), `blockIdx` (區塊在網格內的索引), `blockDim` (區塊的大小), `gridDim` (網格的大小)。透過這些變數，每個執行緒可以計算自己需要處理的資料的索引。
- **記憶體階層 (Memory Hierarchy):** 理解不同記憶體的特性至關重要：
    - **暫存器 (Registers):** 每個執行緒私有，速度最快，但數量有限。
    - **共享記憶體 (Shared Memory / L1 Cache):** 每個區塊私有，區塊內執行緒共享，速度非常快（接近暫存器），是優化的關鍵。需要顯式聲明和管理。
    - **全域記憶體 (Global Memory / DRAM):** GPU 主要記憶體，容量最大，但延遲最高，頻寬有限。CPU 和 GPU 都可以讀寫。優化的主要目標是減少對全域記憶體的訪問次數和提高訪問效率。
    - **常數記憶體 (Constant Memory):** 對所有執行緒唯讀，有快取機制，適合存放不變的參數。
    - **紋理記憶體 (Texture Memory):** 針對 2D/3D 空間局部性進行優化的唯讀記憶體，有硬體內建的快取和濾波功能。
- **執行束 (Warp):** GPU 以 32 個執行緒為一組（稱為 Warp）來調度和執行。同一個 Warp 中的執行緒執行相同的指令 (SIMT - Single Instruction, Multiple Threads)。理解 Warp 對於優化分支（Branch Divergence）和記憶體訪問（Memory Coalescing）至關重要。
- **同步 (Synchronization):** `__syncthreads()` 函式用於確保同一個區塊內的所有執行緒都到達某個點後，才能繼續執行。常用於共享記憶體的讀寫同步。

**3. 開發流程：**

- **a. 編寫核心函式 (.cu 文件):**
    - 使用 CUDA C/C++ 編寫 `__global__` 核心函式。
    - 根據計算邏輯，利用 `threadIdx`, `blockIdx` 等計算每個執行緒負責處理的資料索引。
    - 仔細規劃記憶體使用，考慮如何利用共享記憶體減少全域記憶體訪問。
- **b. 編寫主機端程式碼 (.cpp 或整合進 Python):**
    - 分配 GPU 記憶體 (`cudaMalloc`)。
    - 將資料從 CPU 複製到 GPU (`cudaMemcpyHostToDevice`)。
    - 定義 Grid 和 Block 的維度 (決定啟動多少執行緒)。
    - 使用 `<<<gridDim, blockDim, sharedMemSize, stream>>>` 語法從主機端啟動核心函式。
    - 將結果從 GPU 複製回 CPU (`cudaMemcpyDeviceToHost`)。
    - 釋放 GPU 記憶體 (`cudaFree`)。
- **c. 編譯:**
    - 使用 NVCC (NVIDIA CUDA Compiler) 編譯 `.cu` 文件和相關的 C/C++ 文件。NVCC 會分離主機端和裝置端（GPU）程式碼，分別用 C++ 編譯器和 PTX (Parallel Thread Execution) 中間碼編譯器處理。
    - `nvcc my_kernel.cu my_host_code.cpp -o my_program -lcudart` (基本編譯命令)
- **d. 整合到 AI 框架 (例如 PyTorch):**
    - 通常使用框架提供的 C++ 擴充功能 (e.g., PyTorch C++ Extensions)。
    - 編寫 C++/CUDA 程式碼，並提供一個 Python 的介面（使用 `pybind11` 等工具）。
    - 框架會處理編譯和載入，讓你可以像呼叫普通 Python 函數一樣呼叫你的自訂 CUDA 核心。
- **e. 除錯 (Debugging):**
    - GPU 除錯比 CPU 困難。可以使用 `printf` (在核心中使用，但會影響效能且輸出順序不保證)、`cuda-gdb` (CUDA 的 GDB 版本)、或 NVIDIA Nsight Visual Studio Edition。
- **f. 剖析與優化 (Profiling & Optimization):** 這是最關鍵也最耗時的部分。

**4. 優化策略 (Optimization Strategies):**

- **最大化記憶體吞吐量 (Maximize Memory Throughput):**
    - **記憶體合併 (Memory Coalescing):** 讓同一個 Warp 中的 32 個執行緒盡可能連續地訪問全域記憶體。理想情況下，一次載入 128 字節（32 個執行緒 * 4 字節/浮點數）。避免隨機或跨步訪問。
    - **使用共享記憶體 (Use Shared Memory):** 將需要重複訪問的全域記憶體資料塊載入到共享記憶體中（Tiling / Blocking 技術），然後讓區塊內的執行緒從快速的共享記憶體讀取。這是最常用的優化手段之一。
    - **選擇合適的記憶體類型：** 對於唯讀且會被大量執行緒重複讀取的資料，考慮使用常數記憶體或紋理記憶體（如果訪問模式符合）。
- **最大化計算資源利用率 (Maximize Compute Utilization):**
    - **指令級並行 (Instruction-Level Parallelism - ILP):** 編寫可以被編譯器優化以並行執行的指令。使用 FMA (Fused Multiply-Add) 指令 `fmaf()` 可以提高精度和效能。
    - **提高佔用率 (Increase Occupancy):** Occupancy 指的是 GPU 上 SM (Streaming Multiprocessor) 的活躍 Warp 數量與其最大容量的比率。更高的 Occupancy 有助於隱藏記憶體訪問延遲（當一個 Warp 等待記憶體時，SM 可以切換到另一個就緒的 Warp 執行）。但不是越高越好，需要平衡暫存器和共享記憶體的使用。
    - **調整區塊大小 (Tune Block Size):** 區塊大小（執行緒數量）會影響 Occupancy、共享記憶體使用和同步開銷。通常選擇 32 的倍數（如 128, 256, 512），需要實驗找到最佳值。
- **減少執行緒發散 (Minimize Thread Divergence):**
    - 當同一個 Warp 中的執行緒遇到條件分支 (if/else, switch) 且走向不同路徑時，就會發生分支發散。GPU 需要序列化執行所有路徑，導致效能下降。盡量讓同一個 Warp 中的執行緒執行相同的程式碼路徑。
- **核心融合 (Kernel Fusion):**
    - 將多個訪問相同資料的簡單核心合併成一個更複雜的核心，減少核心啟動開銷和全域記憶體讀寫。
- **非同步操作與流 (Asynchronous Operations & Streams):**
    - 使用 CUDA Streams (`cudaStream_t`) 實現資料傳輸 (CPU <-> GPU) 和核心執行的重疊，以及多個核心之間的並行執行。
- **使用數值精度較低的類型 (Use Lower Precision):**
    - 如果模型容忍，使用半精度 (FP16) 或甚至 INT8 進行計算，可以大幅減少記憶體頻寬需求和提高計算吞吐量（需要 Tensor Cores 支援）。

**5. 剖析工具 (Profiling Tools):**

- **NVIDIA Nsight Systems:** 用於觀察應用程式的整體行為，包括 CPU 和 GPU 活動、CUDA API 呼叫、記憶體傳輸、核心啟動等。適合找出效能瓶頸在哪個階段。
- **NVIDIA Nsight Compute:** 用於深入分析單個 CUDA 核心的效能。提供詳細的指標，如 Occupancy、記憶體吞吐量、指令執行情況、快取命中率、分支發散情況等，指導你進行針對性優化。

**範例場景 (AI/DL):**

- **自訂卷積：** 實現非標準的群組卷積、深度可分離卷積的融合版本、或針對特定硬體優化的 Winograd 卷積。
- **自訂活化函數：** 實現一個新的、複雜的活化函數。
- **自訂池化或正規化層：** 實現特殊的池化策略或融合了正規化操作的層。
- **高效的 Attention 機制：** 如 FlashAttention 等，透過 Tiling 和優化記憶體訪問模式，大幅加速 Transformer 模型中的 Attention 計算。
- **稀疏運算：** 為具有特定稀疏模式的張量乘法或卷積編寫核心。
- **圖神經網路 (GNN) 中的聚合操作：** GNN 的訊息傳遞常涉及不規則的記憶體訪問，自訂核心可以針對圖的結構進行優化。

**挑戰與考量：**

- **學習曲線陡峭：** 需要深入理解 GPU 硬體架構、並行程式設計範式和 CUDA C++。
- **開發與除錯困難：** 並行程式的除錯本身就比序列程式複雜，GPU 除錯工具不如 CPU 成熟。
- **維護成本高：** 程式碼通常與特定硬體和 CUDA 版本相關，升級時可能需要修改。
- **可移植性差：** CUDA 程式碼只能在 NVIDIA GPU 上運行。

**結論：**

自訂 CUDA 核心開發是 AI 和深度學習領域中一項高階但強大的技術。它適用於當標準函式庫無法滿足極致效能需求、需要實現新穎操作或需要深度優化特定計算瓶頸的場景。雖然開發過程複雜且具挑戰性，但透過精確控制硬體資源和計算流程，可以實現顯著的效能提升，推動 AI/DL 模型的邊界。對於追求頂尖效能的研究人員和工程師來說，掌握自訂 CUDA 核心開發是一項寶貴的技能。

================================================



### 實作一個自訂的CUDA核心（Kernel）

實作一個自訂的CUDA核心（Kernel），用於加速卷積神經網絡（CNN）的卷積計算（Convolution Calculation）。這個例子會以一個簡單的2D卷積為基礎，逐步說明實作過程，並解釋每個步驟的目的與背後的原理。假設我們要對一張圖片進行卷積運算，並使用CUDA來加速。

---

### 範例目標

實現一個簡單的2D卷積運算：

- **輸入**：一張單通道圖片（例如灰階圖），大小為width x height。
- **卷積核（Kernel/Filter）**：一個小的矩陣，例如3x3。
- **輸出**：卷積後的結果圖（Feature Map）。
- **目的**：利用GPU的並行計算能力，加速卷積運算。

---

### 步驟分解與實作

#### 1. 理解卷積運算的基本原理

在CNN中，2D卷積的基本操作是將一個小的卷積核（例如3x3）滑過輸入圖片，對每個位置的像素與卷積核進行點積運算，然後將結果寫入輸出圖。假設輸入圖片是I，卷積核是K，輸出圖是O，那麼對於輸出圖的每個像素O(x, y)：

text

CollapseWrapCopy

`O(x, y) = ΣΣ I(x+i, y+j) * K(i, j)`

其中i和j是卷積核的相對座標（例如-1到1表示3x3核的範圍）。

**為什麼用CUDA加速？**

- 卷積運算是高度可並行的：每個輸出像素的計算是獨立的，可以同時進行。
- GPU有數千個核心，能並行處理這些計算，而CPU通常只有幾個核心，效率遠低於GPU。

---

#### 2. 設計CUDA核心的基本結構

我們需要寫一個CUDA核心，讓每個GPU執行緒（Thread）負責計算輸出圖中的一個像素。以下是逐步實作的過程：

##### (1) 定義輸入與輸出

假設：

- 輸入圖片大小：width x height（例如32x32）。
- 卷積核大小：3x3。
- 輸出圖大小：為了簡單起見，假設不使用填充（Padding），則輸出大小為(width-2) x (height-2)。

##### (2) CUDA核心的基本想法

- 每個執行緒計算一個O(x, y)。
- 使用threadIdx和blockIdx來確定當前執行緒負責的輸出座標(x, y)。
- 在核心中訪問輸入圖片和卷積核的數據，進行點積運算。

---

#### 3. 實作CUDA程式碼

以下是完整的實作範例，包括主程式和CUDA核心。

##### (1) CUDA核心（Kernel）程式碼

```c
__global__ void conv2dKernel(float* input, float* kernel, float* output, int width, int height, int kernelSize) {
    // 計算當前執行緒負責的輸出座標 (x, y)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 檢查是否在有效範圍內（考慮不使用Padding的情況）
    int outWidth = width - kernelSize + 1;
    int outHeight = height - kernelSize + 1;
    if (x >= outWidth || y >= outHeight) return;

    // 卷積核的半徑（例如3x3核，半徑為1）
    int radius = kernelSize / 2;
    float sum = 0.0f;

    // 對應輸入圖片區域與卷積核進行點積
    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            int inputX = x + i - radius;  // 輸入圖片的x座標
            int inputY = y + j - radius;  // 輸入圖片的y座標
            if (inputX >= 0 && inputX < width && inputY >= 0 && inputY < height) {
                sum += input[inputY * width + inputX] * kernel[i * kernelSize + j];
            }
        }
    }

    // 將結果寫入輸出圖
    output[y * outWidth + x] = sum;
}
```

**解釋：**

- __global__：標記這是一個CUDA核心函數，可在GPU上執行。
- blockIdx和threadIdx：用來計算每個執行緒的全局座標(x, y)。
- 邊界檢查：確保執行緒不會超出輸出圖範圍。
- 雙重迴圈：遍歷卷積核的每個元素，計算點積。
- 座標映射：將卷積核的相對座標映射到輸入圖片的絕對座標。

---

##### (2) 主程式（Host程式碼）

```c
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    // 定義輸入、卷積核和輸出的大小
    int width = 32, height = 32;
    int kernelSize = 3;
    int outWidth = width - kernelSize + 1;
    int outHeight = height - kernelSize + 1;

    // 分配Host記憶體
    float *h_input = (float*)malloc(width * height * sizeof(float));
    float *h_kernel = (float*)malloc(kernelSize * kernelSize * sizeof(float));
    float *h_output = (float*)malloc(outWidth * outHeight * sizeof(float));

    // 初始化輸入數據（這裡簡化為全1）
    for (int i = 0; i < width * height; i++) h_input[i] = 1.0f;
    // 初始化卷積核（例如簡單的平均濾波器）
    for (int i = 0; i < kernelSize * kernelSize; i++) h_kernel[i] = 1.0f / 9.0f;

    // 分配Device記憶體
    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, width * height * sizeof(float));
    cudaMalloc(&d_kernel, kernelSize * kernelSize * sizeof(float));
    cudaMalloc(&d_output, outWidth * outHeight * sizeof(float));

    // 將數據從Host複製到Device
    cudaMemcpy(d_input, h_input, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);

    // 設定執行緒網格與區塊
    dim3 blockSize(16, 16);  // 每個區塊16x16個執行緒
    dim3 gridSize((outWidth + blockSize.x - 1) / blockSize.x, 
                  (outHeight + blockSize.y - 1) / blockSize.y);

    // 啟動CUDA核心
    conv2dKernel<<<gridSize, blockSize>>>(d_input, d_kernel, d_output, width, height, kernelSize);

    // 將結果從Device複製回Host
    cudaMemcpy(h_output, d_output, outWidth * outHeight * sizeof(float), cudaMemcpyDeviceToHost);

    // 檢查結果（這裡簡單打印第一個元素）
    printf("Output[0] = %f\n", h_output[0]);

    // 釋放記憶體
    free(h_input); free(h_kernel); free(h_output);
    cudaFree(d_input); cudaFree(d_kernel); cudaFree(d_output);

    return 0;
}
```

**解釋：**

- 記憶體分配：分別在Host（CPU）和Device（GPU）上分配記憶體。
- 數據傳輸：使用cudaMemcpy將輸入圖片和卷積核傳到GPU。
- 網格與區塊：定義執行緒的組織方式，確保覆蓋所有輸出像素。
- 結果回收：計算完成後，將結果從GPU傳回CPU。

---

#### 4. 為什麼這樣設計？

- **並行性**：每個執行緒獨立計算一個輸出像素，充分利用GPU的數千個核心。例如，對30x30的輸出圖，有900個像素可以同時計算，而CPU可能需要逐一處理。
- **記憶體管理**：將數據放在GPU的Global Memory中，雖然訪問速度不如共享記憶體（Shared Memory），但對於初學, 簡單實現已足夠。進階優化可使用Shared Memory進一步加速。
- **效率**：相比CPU逐一計算，GPU並行處理可將執行時間從秒級縮減到毫秒級，尤其在處理大圖片或多層CNN時效果更顯著。

---

#### 5. 編譯與執行

假設你已安裝CUDA Toolkit，使用以下命令編譯：

```bash
nvcc -o conv2d conv2d.cu
```

然後執行：

```bash
./conv2d
```

---

### 進階優化（選讀）

1. **使用常量記憶體（Constant Memory）**：將卷積核放入__constant__記憶體，提升訪問速度。
2. **共享記憶體（Shared Memory）**：將輸入圖片分塊載入共享記憶體，減少Global Memory訪問。
3. **平鋪（Tiling）**：將輸入數據分成小塊，提升數據局部性。

這些優化需要更複雜的程式碼，但能進一步提升性能，尤其在大規模CNN中。

---

### 總結

這個範例展示了一個基本的CUDA卷積實現，利用GPU並行計算大幅加速CNN的卷積運算。透過將每個像素的計算分配給獨立執行緒，我們充分利用了GPU的並行能力。相比CPU串行計算，這種方法在處理大規模數據時能顯著提升效率，是深度學習加速的基礎技術之一。

如果你有其他具體需求（例如多通道圖片或更大卷積核），我可以進一步調整範例！請問有什麼我可以再幫你的？



Reference:
CUDA编程指北：从入门到实践 - 离心的文章 - 知乎
https://zhuanlan.zhihu.com/p/680075822

实例：手写 CUDA 算子，让 Pytorch 提速 20 倍（某特殊算子） - PENG Bo的文章 - 知乎
https://zhuanlan.zhihu.com/p/476297195

熬了几个通宵，我写了份CUDA新手入门代码 - godweiyang的文章 - 知乎
https://zhuanlan.zhihu.com/p/360441891

Neural Network CUDA Example
https://github.com/godweiyang/NN-CUDA-Example

