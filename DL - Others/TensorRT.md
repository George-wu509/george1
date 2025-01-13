
Reference:

[1] TensorRT 介紹與安裝教學

[https://medium.com/ching-i/tensorrt-%E4%BB%8B%E7%B4%B9%E8%88%87%E5%AE%89%E8%A3%9D%E6%95%99%E5%AD%B8-45e44f73b25e](https://medium.com/ching-i/tensorrt-%E4%BB%8B%E7%B4%B9%E8%88%87%E5%AE%89%E8%A3%9D%E6%95%99%E5%AD%B8-45e44f73b25e)


ONNX Runtime and TensorRT总结
https://zhuanlan.zhihu.com/p/706031583

想直接使用ONNX模型来做部署的话，有下列几种情况：第一种情况，目标平台是CUDA或者X86的话，又怕环境配置麻烦采坑，比较推荐使用的是微软的ONNXRuntime；第二种情况，而如果目标平台是CUDA又追求极致的效率的话，可以考虑转换成TensorRT；第三种情况，如果目标平台是ARM或者其他IoT设备，那么就要考虑使用端侧推理框架了，例如NCNN、MNN和MACE等。

第一种情况应该是坑最少，但要注意官方ONNXRuntime安装包只支持[CUDA 10](https://zhida.zhihu.com/search?content_id=245026964&content_type=Article&match_order=1&q=CUDA+10&zhida_source=entity)和Python 3，如果是其他环境可能需要自行编译。安装完成之后推理部署的代码可以直接参考官方文档。

第二种情况要稍微麻烦一点，需要先搭建好TensorRT的环境，然后可以直接使用TensorRT对ONNX模型进行推理；**更为推荐的做法是将ONNX模型转换为TensorRT的trt格式的engine文件，这样可以获得最优的性能。**关于ONNX parser部分的代码，NVIDIA是开源出来了的（当然也包括其他parser比如caffe的），不过这一块如果模型中存在自定义OP，会存在一些坑。

第三种情况的话一般问题也不大，由于是在端上执行，计算力有限，所以需要确保模型是经过精简的，能够适配移动端的。几个端侧推理框架的性能到底如何并没有定论，由于大家都是手写汇编优化，以卷积为例，有的框架针对不同尺寸的卷积都各写了一种汇编实现，因此不同的模型、不同的端侧推理框架，不同的ARM芯片都有可能导致推理的性能有好有坏，这都是正常情况。


利用 TensorRT 加速 PyTorch 模型時，通常有兩種主要方法：

1. **直接使用 PyTorch TensorRT (torch-tensorrt)**。
2. **將 PyTorch 模型轉換為 ONNX 格式，然後用 TensorRT 加速**。

這裡，我將詳細說明這兩種方法，並提供完整的代碼和步驟，最後對兩者進行比較。

---

### 方法 1: 使用 Torch-TensorRT 直接加速 PyTorch 模型

#### 安裝必要的庫

`pip install torch-tensorrt`

#### 代碼示例

假設我們有一個簡單的 PyTorch 模型：
```python
import torch
import torch.nn as nn
import torch_tensorrt

# 定義一個簡單的 PyTorch 模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

# 加載模型
model = SimpleModel().eval()

# 準備測試數據
input_data = torch.randn(1, 10).cuda()  # 模型和數據需要在 GPU 上

# 將模型轉換為 TensorRT 優化模型
trt_model = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input(input_data.shape)],
    enabled_precisions={torch.float},  # 可以指定為 torch.half 以支持 FP16
    truncate_long_and_double=True
)

# 使用 TensorRT 模型進行推理
with torch.no_grad():
    output = trt_model(input_data)
print("TensorRT 模型輸出:", output)

```

---

### 方法 2: 先轉為 ONNX 格式，再用 TensorRT 加速

#### 安裝必要的庫

`pip install onnx onnxruntime 
pip install nvidia-pyindex nvidia-tensorrt`

#### 代碼示例
```python
import torch
import torch.nn as nn
import onnx
import tensorrt as trt
import numpy as np
from cuda import cudart

# 定義與上述相同的 PyTorch 模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

# 加載模型
model = SimpleModel().eval()

# Step 1: 將 PyTorch 模型轉換為 ONNX 格式
onnx_file_path = "simple_model.onnx"
input_data = torch.randn(1, 10)  # CPU 上準備數據
torch.onnx.export(
    model,
    input_data,
    onnx_file_path,
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output']
)
print(f"ONNX 模型已保存到 {onnx_file_path}")

# Step 2: 使用 TensorRT 加載 ONNX 模型並進行加速
logger = trt.Logger(trt.Logger.WARNING)
with trt.Builder(logger) as builder, builder.create_network(1) as network, trt.OnnxParser(network, logger) as parser:
    builder.max_workspace_size = 1 << 30  # 設定最大工作空間
    builder.fp16_mode = False  # 不啟用 FP16

    # 讀取 ONNX 模型
    with open(onnx_file_path, "rb") as model_file:
        if not parser.parse(model_file.read()):
            print("ERROR: Failed to parse ONNX model.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()

    # 編譯 TensorRT 引擎
    engine = builder.build_cuda_engine(network)
    if not engine:
        print("ERROR: Failed to build TensorRT engine.")
        exit()

# Step 3: 使用 TensorRT 引擎進行推理
context = engine.create_execution_context()
input_shape = (1, 10)
output_shape = (1, 5)
d_input = cudart.cudaMalloc(np.prod(input_shape) * 4)[1]  # 分配 GPU 記憶體
d_output = cudart.cudaMalloc(np.prod(output_shape) * 4)[1]

# 準備輸入數據
input_data_np = np.random.randn(*input_shape).astype(np.float32)
cudart.cudaMemcpy(d_input, input_data_np.ctypes.data, input_data_np.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

# 推理
context.execute_v2([int(d_input), int(d_output)])

# 獲取輸出結果
output_data_np = np.empty(output_shape, dtype=np.float32)
cudart.cudaMemcpy(output_data_np.ctypes.data, d_output, output_data_np.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
print("TensorRT 模型輸出:", output_data_np)

# 清理記憶體
cudart.cudaFree(d_input)
cudart.cudaFree(d_output)

```

---

### 比較兩種方法

|**項目**|**Torch-TensorRT**|**ONNX + TensorRT**|
|---|---|---|
|**易用性**|簡單直接，對 PyTorch 用戶更友好|相對複雜，需要理解 ONNX 和 TensorRT 的細節|
|**靈活性**|僅限支持的 PyTorch 操作|支持更多框架（TensorFlow、PyTorch 等）|
|**性能**|在簡單場景下性能接近|性能通常更優，特別是在 FP16 或 INT8 的場景|
|**支持的運算精度**|支持 FP16 和 INT8（需特定硬體支持）|支持 FP16、INT8，且配置更靈活|
|**適用場景**|快速加速現有的 PyTorch 模型|當需要跨框架支持或進一步優化時|

如果你只使用 PyTorch，並希望快速加速模型，選擇 **Torch-TensorRT** 更加簡單直接。如果需要最大限度的性能優化或支持多框架，選擇 **ONNX + TensorRT** 是更好的方案。