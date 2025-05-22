
![[Pasted image 20250411224045.png]]

|                            | Parms                      | FLOPs                                |
| -------------------------- | -------------------------- | ------------------------------------ |
| 卷积层 (Conv2D)               | (Kh​×Kw​×Cin​)×Cout​+Cout​ | 2×(Kh​×Kw​×Cin​)×Cout​×Hout​×Wout    |
| Fully connection <br>layer | Nin​×Nout​+Nout​           | 2×Nin​×Nout​                         |
| Batch <br>normalization    | 2×Cin​                     | 4×N×C×H×W    (dataset: N,C,H,W)      |
| 池化层Pooling                 | 0                          | Hout​×Wout​×Cin     (height,width,c) |



Reference:

[1] CNN基础知识——卷积（Convolution）、填充（Padding）、步长(Stride)

[https://zhuanlan.zhihu.com/p/77471866](https://zhuanlan.zhihu.com/p/77471866)


以下是 PyTorch 中 `nn.Conv2d` 函數的基本用法示例代碼，以及詳細解釋:
```python
import torch
import torch.nn as nn

# 定義卷積層
conv_layer = nn.Conv2d(in_channels=3, 
                       out_channels=16, 
                       kernel_size=3, 
                       stride=1, 
                       padding=1)

# 創建一個隨機輸入張量
input_tensor = torch.randn(1, 3, 32, 32)

# 應用卷積操作
output = conv_layer(input_tensor)

print(output.shape)  # 輸出: torch.Size([1, 16, 32, 32])
```
現在讓我們詳細解釋 `nn.Conv2d` 函數的參數和工作原理:

## 參數解釋

1. **in_channels** (3): 輸入的通道數。在這個例子中，我們假設輸入是一個 RGB 圖像，所以有 3 個[輸入通道](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html?highlight=conv2d)
2. **out_channels** (16): 輸出的通道數，也就是我們想要的特徵圖的數量。這裡我們設置為 16，意味著卷積操作後會產生 16 個
3. **kernel_size** (3): 卷積核的大小。這裡我們使用 3x3 的卷積核
4. **stride** (1): 卷積核在輸入上滑動的步長。默認為 1，表示每次移動一個像素
5. **padding** (1): 在輸入周圍添加的零填充量。這裡設置為 1，意味著在輸入的每一邊添加一行/列的零
## 工作原理

1. **初始化**: 當我們創建 `nn.Conv2d` 實例時，PyTorch 會自動初始化權重和偏置（如果 `bias=True`）。權重的形狀為 (out_channels, in_channels, kernel_size, kernel_size)
2. **前向傳播**: 當我們將輸入張量傳遞給卷積層時（如 `output = conv_layer(input_tensor)`），PyTorch 會執行以下操作：
    
    - 對輸入應用填充
    - 使用定義的步長在輸入上滑動卷積核
    - 對每個位置執行點積運算
    - 如果有偏置，則添加偏置
    - 生成輸出特徵圖
    
3. **輸出形狀**: 輸出張量的形狀由以下公式決定：Hout=⌊Hin+2∗padding−kernel sizestride+1⌋Hout​=⌊strideHin​+2∗padding−kernel size​+1⌋Wout=⌊Win+2∗padding−kernel sizestride+1⌋Wout​=⌊strideWin​+2∗padding−kernel size​+1⌋其中 H_in 和 W_in 是輸入的高度和寬度

在我們的例子中，輸入形狀為 (1, 3, 32, 32)，輸出形狀為 (1, 16, 32, 32)。這是因為我們使用了 padding=1，保持了輸入的空間維度不變，同時將通道數從 3 增加到 16。

## 注意事項

- `nn.Conv2d` 是一個可訓練的層，其權重和偏置可以通過反向傳播進行更新
- 可以通過調整 `stride` 和 `padding` 來控制輸出的空間維度
- 對於更高級的用途，還可以使用 `dilation` 和 `groups` 參數來進行擴張卷積和分組卷積

通過理解和靈活運用這些參數，我們可以設計出各種複雜的卷積神經網絡架構，以適應不同的計算機視覺任務。


在卷积神经网络（CNN）中，参数（Parameters）和浮点运算数（FLOPs）是衡量模型大小和计算复杂度的两个关键指标。




### 1. 参数 (Parameters)

|                            | Parms                      | FLOPs                             |
| -------------------------- | -------------------------- | --------------------------------- |
| 卷积层 (Conv2D)               | (Kh​×Kw​×Cin​)×Cout​+Cout​ | 2×(Kh​×Kw​×Cin​)×Cout​×Hout​×Wout |
| Fully connection <br>layer | Nin​×Nout​+Nout​           | 2×Nin​×Nout​                      |
| Batch <br>normalization    | 2×Cin​                     | 4×N×C×H×W                         |
| 池化层Pooling                 | 0                          | Hout​×Wout​×Cin                   |

参数是指模型中可学习的权重和偏置项的总数。模型越大，参数越多，通常意味着模型容量更大，但也更容易过拟合，且需要更多的内存。

- **卷积层 (Conv2D)**
    
    - **权重 (Weights)**: Kh​×Kw​×Cin​×Cout​
        - Kh​: 卷积核的高度
        - Kw​: 卷积核的宽度
        - Cin​: 输入特征图的通道数
        - Cout​: 输出特征图的通道数 (即卷积核的数量)
    - **偏置 (Bias)**: Cout​ (每个输出通道有一个偏置)
    - **总参数**: (Kh​×Kw​×Cin​)×Cout​+Cout​
- **全连接层 (Fully Connected Layer / Linear Layer)**
    
    - **权重 (Weights)**: Nin​×Nout​
        - Nin​: 输入单元的数量
        - Nout​: 输出单元的数量
    - **偏置 (Bias)**: Nout​
    - **总参数**: Nin​×Nout​+Nout​
- **批量归一化层 (Batch Normalization Layer)**
    
    - **可学习参数**: γ (缩放因子) 和 β (偏移因子)。
    - 每个特征通道都有一个 γ 和一个 β。
    - **总参数**: 2×Cin​ (其中 Cin​ 是输入特征图的通道数)
    - 注意：`running_mean` 和 `running_var` 是非可学习参数，不计入总参数。
- **池化层 (Pooling Layer) 和 激活函数 (Activation Functions, e.g., ReLU)**
    
    - 这些层不包含任何可学习参数。它们的运算是固定的，不需要训练。
    - **总参数**: 0

**整个网络的总参数**就是所有层参数的总和。

### 2. 浮点运算数 (FLOPs - Floating Point Operations)

FLOPs (或 GFLOPs，GigaFLOPs) 是衡量模型计算复杂度的指标，代表模型进行一次前向传播所需的浮点运算次数。FLOPs 越低，通常意味着模型推理速度越快、能耗越低。

**需要注意的是**：

- **FLOPs** (Floating Point Operations) 指的是运算的总次数。
    
- **FLOPS** (Floating Point Operations Per Second) 指的是每秒浮点运算次数，是衡量硬件性能的指标。两者容易混淆。在模型复杂度评估中，我们通常指 FLOPs。
    
- 通常一个乘加操作 (MAC - Multiply-Accumulate) 算作 2 个 FLOPs (1个乘法 + 1个加法)。但在某些语境下，可能会将一个 MAC 算作 1 个 FLOP。这里我们以 1 MAC = 2 FLOPs 为准。
    
- **卷积层 (Conv2D)**
    
    - **输出特征图尺寸**:
        - Hout​=⌊(Hin​+2×P−Kh​)/Sh​⌋+1
        - Wout​=⌊(Win​+2×P−Kw​)/Sw​⌋+1
        - Hin​,Win​: 输入特征图的高度和宽度
        - P: 填充 (Padding)
        - Sh​,Sw​: 步长 (Stride)
    - **每个输出像素的计算**:
        - 每个输出像素需要 Kh​×Kw​×Cin​ 次乘法和 Kh​×Kw​×Cin​−1 次加法 (如果考虑偏置，再加1次)。通常简化为 Kh​×Kw​×Cin​ 次乘加 (MAC)。
    - **总 FLOPs (简化版，MACs × 2)**: 2×(Kh​×Kw​×Cin​)×Cout​×Hout​×Wout​
        - 2: 因为一个 MAC 算作 2 个 FLOPs。
        - 如果考虑偏置的加法，可以再加上 Cout​×Hout​×Wout​ 次加法。但通常在 FLOPs 估算中，偏置的加法会被简化忽略或合并到 MACs 中。
- **全连接层 (Fully Connected Layer / Linear Layer)**
    
    - **总 FLOPs (简化版，MACs × 2)**: 2×Nin​×Nout​
        - Nin​: 输入单元的数量
        - Nout​: 输出单元的数量
        - 2: 因为一个 MAC 算作 2 个 FLOPs。
        - 如果考虑偏置的加法，可以再加上 Nout​ 次加法。
- **批量归一化层 (Batch Normalization Layer)**
    
    - 对于每个元素：
        - 减去均值：1次减法
        - 除以标准差：1次除法
        - 乘以 γ: 1次乘法
        - 加上 β: 1次加法
    - 总共 4 个 FLOPs/元素。
    - **总 FLOPs**: 4×N×C×H×W (对于 NCHW 格式的输入)
        - N: 批次大小
        - C,H,W: 特征图的通道、高度、宽度
- **池化层 (Pooling Layer)**
    
    - Max Pooling 或 Average Pooling 都涉及比较或求和操作，这些通常被认为是浮点运算。
    - **总 FLOPs**: Hout​×Wout​×Cin​ (每个输出元素一次操作，例如最大值比较或求平均)
        - Hout​,Wout​: 输出特征图的高度和宽度
        - Cin​: 输入特征图的通道数
    - 有些工具或定义可能将池化层的 FLOPs 计为 0，因为它们不涉及乘法等核心浮点运算，或者其计算量相对于卷积层可以忽略。但严格来说，是存在的。
- **激活函数 (Activation Functions, e.g., ReLU, Sigmoid, Tanh)**
    
    - ReLU: 1次比较操作（通常算作 1 FLOP 或 0 FLOP，因为非常轻量）。
    - Sigmoid/Tanh: 涉及指数和除法，通常算作 4-5 FLOPs/元素。

**整个网络的总 FLOPs**是所有层 FLOPs 的总和。在实际应用中，通常会使用专门的库（如 `thop` 或 `torchstat` for PyTorch）来自动计算模型的 FLOPs 和参数。