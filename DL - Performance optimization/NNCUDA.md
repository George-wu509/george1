
Neural Network CUDA Example github
https://github.com/godweiyang/NN-CUDA-Example/tree/master

#### **完整流程**

下面我们就来详细了解一下PyTorch是如何调用自定义的CUDA算子的。

首先我们可以看到有四个代码文件：

- `main.py`，这是python入口，也就是你平时写模型的地方。
- `add2.cpp`，这是torch和CUDA连接的地方，将CUDA程序封装成了python可以调用的库。
- `add2.h`，CUDA函数声明。
- `add2.cu`，CUDA函数实现。

然后逐个文件看一下是怎么调用的。

|                                      |                                                                        |
| ------------------------------------ | ---------------------------------------------------------------------- |
| /include/add2.h<br><br>CUDA算子实现      | launch_add2()                                                          |
| /kernel /add2.cu <br><br>CUDA算子实现    | __global__ add2_kernel(c,a,b,n)<br>launch_add2(c,a,b,n)  {add2_kernel} |
| /pytorch/add2.cpp<br><br>Torch C++封装 | torch_launch_add2()  {launch_add2}                                     |
| /pytorch/main.py <br>                | show_time(func)<br>run_cuda()  {torch_launch_add2}<br>                 |


#### **CUDA算子实现**

首先最简单的当属`add2.h`和`add2.cu`，这就是普通的CUDA实现。

```c
void launch_add2(float *c,
                 const float *a,
                 const float *b,
                 int n);

__global__ void add2_kernel(float* c,
                            const float* a,
                            const float* b,
                            int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
            i < n; i += gridDim.x * blockDim.x) {
        c[i] = a[i] + b[i];
    }
}

void launch_add2(float* c,
                 const float* a,
                 const float* b,
                 int n) {
    dim3 grid((n + 1023) / 1024);
    dim3 block(1024);
    add2_kernel<<<grid, block>>>(c, a, b, n);
}
```

这里实现的功能是两个长度为n的tensor相加，每个block有1024个线程，一共有n/1024个block。具体CUDA细节就不讲了，本文重点不在于这个。

`add2_kernel`是kernel函数，运行在GPU端的。而`launch_add2`是CPU端的执行函数，调用kernel。注意它是异步的，调用完之后控制权立刻返回给CPU，所以之后计算时间的时候要格外小心，很容易只统计到调用的时间。

### **Torch C++封装**

这里涉及到的是`add2.cpp`，这个文件主要功能是提供一个PyTorch可以调用的接口。

```c
#include <torch/extension.h>
#include "add2.h"

void torch_launch_add2(torch::Tensor &c,
                       const torch::Tensor &a,
                       const torch::Tensor &b,
                       int n) {
    launch_add2((float *)c.data_ptr(),
                (const float *)a.data_ptr(),
                (const float *)b.data_ptr(),
                n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_add2",
          &torch_launch_add2,
          "add2 kernel warpper");
}
```

`torch_launch_add2`函数传入的是C++版本的torch tensor，然后转换成C++指针数组，调用CUDA函数`launch_add2`来执行核函数。

这里用[pybind11](https://zhida.zhihu.com/search?content_id=167734021&content_type=Article&match_order=1&q=pybind11&zhida_source=entity)来对`torch_launch_add2`函数进行封装，然后用cmake编译就可以产生python可以调用的.so库。但是我们这里不直接手动cmake编译，具体方法看下面的章节。

### **Python调用**

最后就是python层面，也就是我们用户编写代码去调用上面生成的库了。

```python
import time
import numpy as np
import torch
from torch.utils.cpp_extension import load

cuda_module = load(name="add2",
                   sources=["add2.cpp", "add2.cu"],
                   verbose=True)

# c = a + b (shape: [n])
n = 1024 * 1024
a = torch.rand(n, device="cuda:0")
b = torch.rand(n, device="cuda:0")
cuda_c = torch.rand(n, device="cuda:0")

ntest = 10

def show_time(func):
    times = list()
    res = list()
    # GPU warm up
    for _ in range(10):
        func()
    for _ in range(ntest):
        # sync the threads to get accurate cuda running time
        torch.cuda.synchronize(device="cuda:0")
        start_time = time.time()
        r = func()
        torch.cuda.synchronize(device="cuda:0")
        end_time = time.time()

        times.append((end_time-start_time)*1e6)
        res.append(r)
    return times, res

def run_cuda():
    cuda_module.torch_launch_add2(cuda_c, a, b, n)
    return cuda_c

def run_torch():
    # return None to avoid intermediate GPU memory application
    # for accurate time statistics
    a + b
    return None

print("Running cuda...")
cuda_time, _ = show_time(run_cuda)
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))

print("Running torch...")
torch_time, _ = show_time(run_torch)
print("Torch time:  {:.3f}us".format(np.mean(torch_time)))
```

这里6-8行的`torch.utils.cpp_extension.load`函数就是用来自动编译上面的几个cpp和cu文件的。最主要的就是`sources`参数，指定了需要编译的文件列表。然后就可以通过`cuda_module.torch_launch_add2`，也就是我们封装好的接口来进行调用。

```python
cuda_module = load(name="add2",
                   sources=["add2.cpp", "add2.cu"],
                   verbose=True)
```

這行程式碼是 PyTorch 中用來**即時編譯並載入 C++ 和 CUDA 擴展 (extensions)** 的關鍵部分。它的作用可以分解如下：

- **`load(...)`:** 這是 `torch.utils.cpp_extension` 模組提供的函式，用於動態地編譯和載入外部程式碼，使其可以在 Python 中被調用。
- **`name="add2"`:** 這個參數指定了編譯後的擴展模組在 Python 中的名稱。在這裡，編譯後的模組將會被命名為 `add2`。之後，你可以通過 `cuda_module.某個函式` 的方式來調用這個擴展模組中定義的函式。
- **`sources=["add2.cpp", "add2.cu"]`:** 這個參數是一個包含原始碼檔案路徑的列表。
    - `"add2.cpp"`: 這是一個 C++ 原始碼檔案。它通常包含一些膠水代碼 (wrapper code)，用於將 CUDA 核心函式暴露給 Python。在這個例子中，它很可能包含一個名為 `torch_launch_add2` 的函式，這個函式會被 Python 調用，並負責啟動 CUDA 核心。
    - `"add2.cu"`: 這是一個 CUDA 原始碼檔案。它包含了在 GPU 上執行的實際計算核心 (kernel)。在這個例子中，它很可能包含一個名為 `add2_kernel` 的 CUDA 函式，用於將兩個輸入張量 `a` 和 `b` 的對應元素相加，並將結果儲存在 `c` 中。
- **`verbose=True`:** 這個參數是一個布林值。如果設定為 `True`，`load` 函式會在編譯過程中輸出詳細的資訊，例如編譯器指令和輸出。這對於 debugging 編譯錯誤很有用。
- 
**總結來說，`cuda_module = load(...)` 這行程式碼的作用是：**

1. **編譯:** 它會使用你的系統上安裝的 CUDA 編譯器 (nvcc) 和 C++ 編譯器 (例如 g++)，將 `add2.cu` 和 `add2.cpp` 檔案編譯成一個動態連結庫 (shared library)。
2. **載入:** 它會將編譯好的動態連結庫載入到 Python 環境中，並將其賦值給變數 `cuda_module`。
3. **暴露介面:** 它會根據 `add2.cpp` 中使用 PyBind11 (PyTorch C++ 擴展使用的綁定庫) 定義的介面，使得 Python 可以調用 C++ 和 CUDA 中定義的函式。在這個例子中，很可能在 `add2.cpp` 中定義了一個可以從 Python 呼叫的函式 `torch_launch_add2`。


接下来的代码就随心所欲了，这里简单写了一个测量运行时间，对比和torch速度的代码，这部分留着下一章节讲解。

总结一下，主要分为三个模块：

- 先编写CUDA算子和对应的调用函数。
- 然后编写torch cpp函数建立PyTorch和CUDA之间的联系，用pybind11封装。
- 最后用PyTorch的cpp扩展库进行编译和调用。