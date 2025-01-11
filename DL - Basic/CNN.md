
Reference:

[1] CNN基础知识——卷积（Convolution）、填充（Padding）、步长(Stride)

[https://zhuanlan.zhihu.com/p/77471866](https://zhuanlan.zhihu.com/p/77471866)


以下是 PyTorch 中 `nn.Conv2d` 函數的基本用法示例代碼，以及詳細解釋:
```
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
2. **out_channels** (16): 輸出的通道數，也就是我們想要的特徵圖的數量。這裡我們設置為 16，意味著卷積操作後會產生 16 個[特徵圖](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html?highlight=conv2d)
3. [
    
    1
    
    ](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html?highlight=conv2d).
3. **kernel_size** (3): 卷積核的大小。這裡我們使用 3x3 的卷積核[
    
    1
    
    ](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html?highlight=conv2d).
4. **stride** (1): 卷積核在輸入上滑動的步長。默認為 1，表示每次移動一個像素[
    
    1
    
    ](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html?highlight=conv2d).
5. **padding** (1): 在輸入周圍添加的零填充量。這裡設置為 1，意味著在輸入的每一邊添加一行/列的零[
    
    1
    
    ](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html?highlight=conv2d).

## 工作原理

1. **初始化**: 當我們創建 `nn.Conv2d` 實例時，PyTorch 會自動初始化權重和偏置（如果 `bias=True`）。權重的形狀為 (out_channels, in_channels, kernel_size, kernel_size)[
    
    1
    
    ](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html?highlight=conv2d).
2. **前向傳播**: 當我們將輸入張量傳遞給卷積層時（如 `output = conv_layer(input_tensor)`），PyTorch 會執行以下操作：
    
    - 對輸入應用填充
    - 使用定義的步長在輸入上滑動卷積核
    - 對每個位置執行點積運算
    - 如果有偏置，則添加偏置
    - 生成輸出特徵圖[
        
        1
        
        ](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html?highlight=conv2d)[
        
        2
        
        ](https://www.geeksforgeeks.org/apply-a-2d-convolution-operation-in-pytorch/)
    
3. **輸出形狀**: 輸出張量的形狀由以下公式決定：Hout=⌊Hin+2∗padding−kernel sizestride+1⌋Hout​=⌊strideHin​+2∗padding−kernel size​+1⌋Wout=⌊Win+2∗padding−kernel sizestride+1⌋Wout​=⌊strideWin​+2∗padding−kernel size​+1⌋其中 H_in 和 W_in 是輸入的高度和寬度[
    
    1
    
    ](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html?highlight=conv2d)[
    
    2
    
    ](https://www.geeksforgeeks.org/apply-a-2d-convolution-operation-in-pytorch/).

在我們的例子中，輸入形狀為 (1, 3, 32, 32)，輸出形狀為 (1, 16, 32, 32)。這是因為我們使用了 padding=1，保持了輸入的空間維度不變，同時將通道數從 3 增加到 16。

## 注意事項

- `nn.Conv2d` 是一個可訓練的層，其權重和偏置可以通過反向傳播進行更新[
    
    1
    
    ](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html?highlight=conv2d).
- 可以通過調整 `stride` 和 `padding` 來控制輸出的空間維度[
    
    1
    
    ](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html?highlight=conv2d).
- 對於更高級的用途，還可以使用 `dilation` 和 `groups` 參數來進行擴張卷積和分組卷積[
    
    1
    
    ](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html?highlight=conv2d).

通過理解和靈活運用這些參數，我們可以設計出各種複雜的卷積神經網絡架構，以適應不同的計算機視覺任務。