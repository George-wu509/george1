

### **PyTorch 基礎**

1. 編寫代碼創建一個 `torch.Tensor`，並將其轉換為 NumPy 數組。
2. 實現一個函數，將一個 3x3 的張量與其轉置矩陣相乘。
3. 編寫代碼實現一個隨機初始化的 5x5 張量，並對每列計算均值和標準差。
4. 創建一個 4x4 張量，將其所有元素設置為 0，但對角線元素設置為 1。
5. 使用 PyTorch 計算以下公式的梯度： $y = x^3 + 2x^2 - 5x + 1$
6. 編寫代碼創建一個 GPU 張量，並將其轉移到 CPU。
7. 創建一個 `torch.nn.Linear` 層並初始化其權重和偏置。
8. 使用 PyTorch 編寫一個函數來計算兩個張量的歐幾里得距離。
9. 使用 PyTorch 中的 `torch.cat` 將兩個張量按列拼接。
10. 編寫代碼進行自動求導，計算 y=sin⁡(x)y = \sin(x)y=sin(x) 在 x=π/4x = \pi / 4x=π/4 處的梯度。

---

### **數據處理與加載**

11. 編寫一個自定義的 `Dataset` 類，從本地文件中加載圖像和標籤。
12. 使用 `DataLoader` 編寫代碼進行批量加載，並對數據進行隨機打亂。
13. 實現一個數據增強函數，包括隨機裁剪和水平翻轉操作。
14. 使用 `torchvision.transforms` 加載 CIFAR-10 數據集並進行標準化。
15. 實現一個函數將數據集中的所有圖像轉換為灰度。
16. 使用 PyTorch 的 `collate_fn` 處理不規則長度的序列數據。
17. 編寫代碼將 COCO 格式的數據集轉換為 PyTorch 支持的格式。
18. 使用 `torch.utils.data` 實現一個多進程數據加載器。
19. 將 MNIST 數據集保存為 PyTorch 的 `pt` 格式，並從中加載數據。
20. 編寫代碼對每個批次的數據進行標籤平衡處理。
21. 使用PyTorch的`Dataset`和`DataLoader`類別來加載自定義數據集。
22. 使用`torchvision`來進行數據增強（Data Augmentation）。
23. 使用`torchtext`來處理文本數據。
24. 使用`torch.utils.data.DataLoader`進行大規模數據的批量處理。
25. 使用`torch.nn.utils.rnn`來處理可變長度的序列數據。

---

### **模型設計**

26. 使用 PyTorch 創建一個全連接神經網絡，用於二分類任務。
27. 編寫代碼實現一個 CNN 用於處理 CIFAR-10 圖像分類。
28. 編寫代碼實現一個自定義的激活函數，如 Swish 或 GELU。
29. 實現一個自定義的 PyTorch 損失函數，用於計算 Focal Loss。
30. 使用 PyTorch 創建一個簡單的 RNN 處理時間序列數據。
31. 使用 `torch.nn.Transformer` 實現一個文本分類模型。
32. 實現一個自定義的 `torch.nn.Module`，包含多個子模塊和跳躍連接。
33. 設計一個具有多輸入的模型，例如處理圖像和文本同時進行分類。
34. 實現一個生成對抗網絡（GAN），用於生成手寫數字。
35. 設計一個基於注意力機制的模型，用於機器翻譯。
36. 實現一個Transformer神經網絡
37. 使用預訓練的模型（如ResNet）進行遷移學習。
38. 實現一個vision Transformer神經網絡
39. 實現一個自定義的損失函數，並在模型中使用。
40. 如何在PyTorch中進行模型的序列化和反序列化？
41. 實現一個自注意力機制（Self-Attention）層。
42. 實現一個自編碼器（Autoencoder）來進行數據壓縮。
43. 實現一個圖神經網路（GNN）來進行節點分類。
44. 使用`torch.distributions`來實現變分自編碼器（VAE）。
45. 實現一個時間序列預測模型。
46. 實現一個LSTM網路來進行情感分析。
47. 實現一個Seq2Seq模型來進行機器翻譯。
48. 實現一個批正規化（Batch Normalization）層。
49. 實現一個UNet神經網絡
50. 實現一個YOLO神經網絡
51. 實現一個MaskRCNN神經網絡
52. 實現一個Segment Anything神經網絡

---

### **訓練流程與優化**

53. 編寫代碼實現訓練一個簡單的線性回歸模型。
54. 實現一個訓練循環，包含驗證和早停功能。
55. 使用 `torch.optim.SGD` 訓練模型，並手動調整學習率。
56. 在訓練過程中記錄損失值，並使用 Matplotlib 畫出曲線。
57. 編寫代碼動態調整學習率，例如使用學習率調度器（LR Scheduler）。
58. 使用混合精度訓練來加速模型訓練。
59. 編寫代碼實現模型的 K 折交叉驗證。
60. 使用 `DistributedDataParallel` 加速模型訓練。
61. 編寫代碼實現自定義的訓練日誌記錄系統。
62. 實現一個完整的模型訓練與測試腳本，支持命令行參數。
63. 使用GPU來加速模型的訓練。
64. 在訓練過程中，如何保存和加載模型的權重？
65. 如何在PyTorch中進行模型的早停（Early Stopping）？
66. 如何在多GPU上進行模型的分佈式訓練？
67. 如何在PyTorch中進行模型的超參數調優？
68. 如何在PyTorch中實現梯度截斷（Gradient Clipping）？
69. 使用`torch.optim`模組來實現自適應學習率調整。
70. 使用`torch.autograd`來計算張量的梯度。
71. 如何在PyTorch中進行模型的混合精度訓練（Mixed Precision Training）？

---

### **模型部署與高效推理**

72. 將一個訓練好的 PyTorch 模型轉換為 ONNX 格式。
73. 編寫代碼使用 `torch.jit.trace` 將模型轉換為 TorchScript。
74. 使用 `onnxruntime` 加載 ONNX 模型進行推理。
75. 將 PyTorch 模型量化以減少模型大小。
76. 實現一個多線程的推理服務，處理多個請求。
77. 測試模型在 GPU 和 CPU 上的推理速度，並比較結果。
78. 使用 TorchServe 部署模型並提供 RESTful API。
79. 優化模型的內存占用，處理大批量推理。
80. 使用 `torch.profiler` 分析模型的性能瓶頸。
81. 編寫代碼測量模型的延遲和吞吐量。
82. 如何在PyTorch中進行模型的剪枝（Pruning）？
83. 如何在PyTorch中進行模型的量化（Quantization）？
84. 如何在PyTorch中進行模型的單元測試？
85. 如何在PyTorch中實現模型的可視化？
86. 使用`torch.jit`來加速模型的推理。
87. 如何在PyTorch中進行模型的版本控制？
88. 使用`torch.multiprocessing`來加速數據加載。
89. 如何在PyTorch中進行模型的部署？
90. 使用`torch.utils.tensorboard`來監控訓練過程。
91. 如何在PyTorch中進行模型的梯度檢查（Gradient Checking）？
92. 使用`torch.quantization`來量化模型以減少其大小。



ref:
https://www.finalroundai.com/blog/pytorch-interview-questions
https://www.adaface.com/blog/pytorch-interview-questions/
https://github.com/Devinterview-io/pytorch-interview-questions
https://interview-questions.org/pytorch/
https://github.com/topics/ai-interview-questions
https://github.com/topics/machine-learning-interview-questions
https://github.com/youssefHosni/Data-Science-Interview-Questions-Answers
https://blog.csdn.net/Vampire_2017/article/details/141529546
https://interviewprep.org/pytorch-interview-questions/
https://climbtheladder.com/pytorch-interview-questions/
https://www.devopsschool.com/blog/top-50-interview-questions-and-answers-of-pytorch/
https://github.com/Devinterview-io/pytorch-interview-questions/blob/main/README.md
https://github.com/lcylmhlcy/Awesome-algorithm-interview


---

### **1. 編寫代碼創建一個 `torch.Tensor`，並將其轉換為 NumPy 數組**

#### 代碼
```python
import torch

# 創建一個 1D Tensor
tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])

# 將 Tensor 轉換為 NumPy 數組
numpy_array = tensor.numpy()

print("Torch Tensor:", tensor)
print("NumPy Array:", numpy_array)

```

#### 中文解釋

1. 使用 `torch.tensor` 創建了一個包含浮點數的 1D Tensor。
2. 使用 `numpy()` 方法將 PyTorch Tensor 轉換為 NumPy 數組。這是一個零拷貝操作，兩者共享同一塊內存，修改其中之一會影響另一個。

---

### **2. 實現一個函數，將一個 3x3 的張量與其轉置矩陣相乘**

#### 代碼

```python
import torch

def tensor_multiply_with_transpose():
    # 創建一個 3x3 的隨機 Tensor
    tensor = torch.rand(3, 3)

    # 計算其轉置
    transpose_tensor = tensor.T

    # 將原 Tensor 與其轉置相乘
    result = torch.matmul(tensor, transpose_tensor)

    print("Original Tensor:\n", tensor)
    print("Transpose of Tensor:\n", transpose_tensor)
    print("Result of Multiplication:\n", result)

# 執行函數
tensor_multiply_with_transpose()

```

#### 中文解釋

1. 使用 `torch.rand` 創建一個隨機的 3x3 Tensor。
2. 使用 `.T` 屬性計算 Tensor 的轉置。
3. 使用 `torch.matmul` 將原始 Tensor 與其轉置進行矩陣乘法。
4. 打印原始 Tensor、轉置矩陣以及乘法結果。

---

### **3. 編寫代碼實現一個隨機初始化的 5x5 張量，並對每列計算均值和標準差**

#### 代碼
```python
import torch

# 創建一個隨機初始化的 5x5 Tensor
tensor = torch.rand(5, 5)

# 計算每列的均值
column_means = tensor.mean(dim=0)

# 計算每列的標準差
column_stds = tensor.std(dim=0)

print("Random 5x5 Tensor:\n", tensor)
print("Column Means:\n", column_means)
print("Column Standard Deviations:\n", column_stds)

```
#### 中文解釋

1. 使用 `torch.rand` 創建一個隨機初始化的 5x5 Tensor，其中每個元素都是 [0, 1) 的隨機浮點數。
2. 使用 `mean(dim=0)` 計算張量沿列方向（dim=0）的均值。
3. 使用 `std(dim=0)` 計算張量沿列方向（dim=0）的標準差。
4. 打印隨機張量、每列的均值和標準差。


---

### **4. 創建一個 4x4 張量，將其所有元素設置為 0，但對角線元素設置為 1**

#### 代碼
```python
import torch

# 創建一個 4x4 的零張量
tensor = torch.zeros(4, 4)

# 將對角線元素設置為 1
tensor.fill_diagonal_(1)

print("4x4 Tensor with Diagonal Elements as 1:\n", tensor)

```
#### 中文解釋

1. 使用 `torch.zeros(4, 4)` 創建一個 4x4 的全零張量。
2. 使用 `fill_diagonal_` 函數直接將對角線元素設置為 1。
    - 此方法為原地操作（in-place operation），不會創建新的張量。
3. 最後打印出結果，顯示張量的內容。

---

### **5. 使用 PyTorch 計算以下公式的梯度： y=x3+2x2−5x+1y = x^3 + 2x^2 - 5x + 1y=x3+2x2−5x+1**

#### 代碼
```python
import torch

# 定義 x 為需要計算梯度的張量
x = torch.tensor(2.0, requires_grad=True)

# 定義公式 y = x^3 + 2x^2 - 5x + 1
y = x**3 + 2*x**2 - 5*x + 1

# 計算梯度
y.backward()

# 輸出 x 的梯度 dy/dx
print("Value of y:", y.item())
print("Gradient (dy/dx) at x = 2:", x.grad.item())

```
#### 中文解釋

1. 使用 `torch.tensor` 定義 xxx，並設置 `requires_grad=True` 以啟用梯度計算。
2. 定義公式 y=x3+2x2−5x+1y = x^3 + 2x^2 - 5x + 1y=x3+2x2−5x+1。
    - 這是 PyTorch 的動態計算圖功能，會自動構建計算圖。
3. 使用 `y.backward()` 方法執行反向傳播，計算 dy/dxdy/dxdy/dx 的值。
4. 使用 `x.grad` 獲取 xxx 的梯度值，並打印出來。
    - 在 x=2x = 2x=2 時，梯度為 dy/dx=3x2+4x−5=3(2)2+4(2)−5=23dy/dx = 3x^2 + 4x - 5 = 3(2)^2 + 4(2) - 5 = 23dy/dx=3x2+4x−5=3(2)2+4(2)−5=23。

### **6. 編寫代碼創建一個 GPU 張量，並將其轉移到 CPU**

#### 代碼

```python
import torch

# 創建一個 GPU 張量（如果可用）
if torch.cuda.is_available():
    gpu_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    print("Tensor on GPU:", gpu_tensor)

    # 將 GPU 張量轉移到 CPU
    cpu_tensor = gpu_tensor.to('cpu')
    print("Tensor on CPU:", cpu_tensor)
else:
    print("CUDA is not available. Running on CPU.")

```

#### 中文解釋

1. 使用 `torch.cuda.is_available()` 檢查是否支持 GPU。
2. 使用 `device='cuda'` 將張量創建在 GPU 上。
3. 使用 `.to('cpu')` 方法將 GPU 張量轉移到 CPU。
4. 如果沒有可用的 GPU，直接運行在 CPU 上。

---

### **7. 創建一個 `torch.nn.Linear` 層並初始化其權重和偏置**

#### 代碼

```python
import torch
import torch.nn as nn

# 創建一個線性層，輸入大小為 3，輸出大小為 2
linear_layer = nn.Linear(3, 2)

# 初始化權重和偏置
nn.init.normal_(linear_layer.weight, mean=0.0, std=0.02)  # 正態分布初始化
nn.init.constant_(linear_layer.bias, 0.0)  # 偏置初始化為 0

print("Initialized Weight:\n", linear_layer.weight)
print("Initialized Bias:\n", linear_layer.bias)

```

1. 使用 `torch.nn.Linear` 創建一個線性層，指定輸入大小和輸出大小。
2. 使用 `nn.init.normal_` 對權重進行正態分布初始化，均值為 0，標準差為 0.02。
3. 使用 `nn.init.constant_` 將偏置初始化為 0。
4. 打印初始化後的權重和偏置。

---

### **8. 使用 PyTorch 編寫一個函數來計算兩個張量的歐幾里得距離**

#### 代碼
```python
import torch

def euclidean_distance(tensor1, tensor2):
    # 計算歐幾里得距離
    distance = torch.sqrt(torch.sum((tensor1 - tensor2) ** 2))
    return distance

# 測試函數
tensor1 = torch.tensor([1.0, 2.0, 3.0])
tensor2 = torch.tensor([4.0, 5.0, 6.0])

distance = euclidean_distance(tensor1, tensor2)
print("Euclidean Distance:", distance.item())

```
#### 中文解釋

1. 定義函數 `euclidean_distance`，計算兩個張量的歐幾里得距離：
    - 使用 `(tensor1 - tensor2) ** 2` 計算平方差。
    - 使用 `torch.sum` 求和，並使用 `torch.sqrt` 計算平方根。
2. 測試時創建兩個張量，並計算它們的距離。

---

### **9. 使用 PyTorch 中的 `torch.cat` 將兩個張量按列拼接**

#### 代碼
```python
import torch

# 創建兩個張量
tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor2 = torch.tensor([[5, 6], [7, 8]])

# 按列（dim=1）拼接
concatenated_tensor = torch.cat((tensor1, tensor2), dim=1)

print("Tensor 1:\n", tensor1)
print("Tensor 2:\n", tensor2)
print("Concatenated Tensor:\n", concatenated_tensor)

```
#### 中文解釋

1. 使用 `torch.tensor` 創建兩個 2x2 的張量。
2. 使用 `torch.cat` 在列方向（dim=1）進行拼接。
3. 打印原始張量和拼接後的結果。

---

### **10. 編寫代碼進行自動求導，計算 y=sin⁡(x)y = \sin(x)y=sin(x) 在 x=π/4x = \pi / 4x=π/4 處的梯度**

#### 代碼
```python
import torch

# 定義 x，並啟用梯度計算
x = torch.tensor(torch.pi / 4, requires_grad=True)

# 定義公式 y = sin(x)
y = torch.sin(x)

# 計算梯度
y.backward()

# 輸出 y 值和梯度
print("Value of y:", y.item())
print("Gradient (dy/dx) at x = pi/4:", x.grad.item())

```
#### 中文解釋

1. 使用 `torch.tensor` 定義 x=π/4x = \pi/4x=π/4，並設置 `requires_grad=True` 啟用自動求導。
2. 定義公式 y=sin⁡(x)y = \sin(x)y=sin(x)。
3. 使用 `y.backward()` 計算梯度。
4. 使用 `x.grad` 獲取梯度值，在 x=π/4x = \pi/4x=π/4 時，梯度應為 cos(x)cos(x)cos(x)，即 2/2≈0.707\sqrt{2}/2 \approx 0.7072​/2≈0.707。

### **11. 編寫一個自定義的 `Dataset` 類，從本地文件中加載圖像和標籤**

#### 代碼
```python
import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.labels = self._load_labels(label_file)
        self.transform = transform

    def _load_labels(self, label_file):
        # 從標籤文件加載圖像名稱和對應標籤
        with open(label_file, 'r') as f:
            lines = f.readlines()
        return [line.strip().split() for line in lines]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name, label = self.labels[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')  # 打開圖像
        label = int(label)  # 轉換標籤為整數
        if self.transform:
            image = self.transform(image)
        return image, label

# 測試自定義 Dataset
# 假設標籤文件每行格式為 "image_name.jpg label"
dataset = CustomImageDataset("images/", "labels.txt")
print("Dataset length:", len(dataset))
print("Sample image and label:", dataset[0])

```
#### 中文解釋

1. 繼承 `torch.utils.data.Dataset` 類，創建自定義數據集類 `CustomImageDataset`。
2. `__init__` 加載圖片目錄路徑和標籤文件，並設置可選的數據增強 `transform`。
3. `_load_labels` 方法加載標籤文件，返回圖像名稱和標籤的列表。
4. `__len__` 方法返回數據集大小。
5. `__getitem__` 方法根據索引加載圖像、應用增強，並返回圖像和標籤。

---

### **12. 使用 `DataLoader` 編寫代碼進行批量加載，並對數據進行隨機打亂**

#### 代碼

```python
from torch.utils.data import DataLoader

# 創建 DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 遍歷批量數據
for images, labels in dataloader:
    print("Batch of images:", images.shape)
    print("Batch of labels:", labels)
    break

```
#### 中文解釋

1. 使用 `torch.utils.data.DataLoader` 將數據集轉換為可批量加載的對象。
2. 設置 `batch_size=4` 指定每批包含 4 個數據點。
3. 使用 `shuffle=True` 隨機打亂數據順序。
4. 遍歷 DataLoader，輸出每批數據的圖像和標籤。

---

### **13. 實現一個數據增強函數，包括隨機裁剪和水平翻轉操作**

#### 代碼

```python
import torchvision.transforms as transforms

# 定義數據增強操作
data_transforms = transforms.Compose([
    transforms.RandomCrop(32),  # 隨機裁剪到 32x32
    transforms.RandomHorizontalFlip(p=0.5),  # 以 50% 機率進行水平翻轉
    transforms.ToTensor()  # 轉換為張量
])

# 測試數據增強
sample_image = Image.open("images/sample.jpg").convert('RGB')
augmented_image = data_transforms(sample_image)

print("Original Image Size:", sample_image.size)
print("Transformed Image Shape:", augmented_image.shape)

```
#### 中文解釋

1. 使用 `torchvision.transforms.Compose` 組合多個數據增強操作。
2. `RandomCrop` 隨機裁剪到指定大小。
3. `RandomHorizontalFlip` 以給定概率進行水平翻轉。
4. 測試數據增強操作，輸出原圖像大小和轉換後的形狀。

---

### **14. 使用 `torchvision.transforms` 加載 CIFAR-10 數據集並進行標準化**

#### 代碼
```python
from torchvision import datasets
import torchvision.transforms as transforms

# 定義標準化和數據增強操作
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 將像素值標準化到 [-1, 1]
])

# 加載 CIFAR-10 數據集
cifar10_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)

# 創建 DataLoader
cifar10_loader = DataLoader(cifar10_dataset, batch_size=8, shuffle=True)

# 打印第一批數據
images, labels = next(iter(cifar10_loader))
print("Batch of images shape:", images.shape)
print("Batch of labels:", labels)

```
#### 中文解釋

1. 使用 `torchvision.datasets.CIFAR10` 加載 CIFAR-10 數據集，並設置 `transform`。
2. `Normalize` 將像素值標準化到 [−1,1][-1, 1][−1,1]，均值和標準差根據 RGB 通道設定。
3. 使用 DataLoader 加載數據集，設置批量大小為 8 並隨機打亂。

---

### **15. 實現一個函數將數據集中的所有圖像轉換為灰度**

#### 代碼

```python
def convert_to_grayscale(dataset):
    gray_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 轉換為單通道灰度圖
        transforms.ToTensor()
    ])
    
    # 轉換整個數據集
    gray_images = [gray_transform(Image.fromarray(data)) for data, _ in dataset]
    return gray_images

# 測試轉換函數
cifar10_gray = convert_to_grayscale(cifar10_dataset)
print("First grayscale image shape:", cifar10_gray[0].shape)

```
#### 中文解釋

1. 定義轉換函數，使用 `Grayscale` 將圖像轉換為灰度。
2. 遍歷數據集，對每張圖像應用轉換。
3. 測試函數，輸出灰度圖像的形狀。

### **16. 使用 PyTorch 的 `collate_fn` 處理不規則長度的序列數據**

#### 代碼

```python
import torch
from torch.utils.data import DataLoader, Dataset

# 定義自定義數據集
class VariableLengthDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 定義自定義 collate_fn
def collate_fn(batch):
    # 按序列長度排序
    batch.sort(key=len, reverse=True)
    lengths = torch.tensor([len(seq) for seq in batch])
    padded_batch = torch.zeros(len(batch), lengths.max())  # 創建零填充張量
    for i, seq in enumerate(batch):
        padded_batch[i, :len(seq)] = torch.tensor(seq)
    return padded_batch, lengths

# 測試數據和 DataLoader
data = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
dataset = VariableLengthDataset(data)
dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

# 遍歷批量數據
for batch_data, lengths in dataloader:
    print("Padded Batch:\n", batch_data)
    print("Lengths:", lengths)

```
#### 中文解釋

1. 定義自定義數據集 `VariableLengthDataset`，用於存儲不規則長度的數據。
2. 使用自定義 `collate_fn` 處理批量數據：
    - 按長度排序（由長到短）。
    - 根據最大長度填充零，使每個序列具有相同長度。
    - 返回填充後的張量和原始序列長度。
3. 測試 DataLoader，按批量輸出處理後的結果。

---

### **17. 編寫代碼將 COCO 格式的數據集轉換為 PyTorch 支持的格式**

#### 代碼

```python
import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class COCODataset(Dataset):
    def __init__(self, annotation_file, image_dir, transform=None):
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations['images'])

    def __getitem__(self, idx):
        img_info = self.annotations['images'][idx]
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        annotations = [
            ann for ann in self.annotations['annotations']
            if ann['image_id'] == img_info['id']
        ]
        if self.transform:
            image = self.transform(image)
        return image, annotations

# 測試數據集類
coco_dataset = COCODataset("annotations.json", "images/")
print("Number of images:", len(coco_dataset))
print("Sample image and annotations:", coco_dataset[0])

```

#### 中文解釋

1. 使用 `json` 讀取 COCO 格式的標註文件。
2. 定義 `COCODataset` 類，解析 `images` 和 `annotations` 部分。
3. 對於每個圖像，從 `annotations` 中提取對應的標註數據。
4. 測試數據集類，打印圖像數量和示例數據。

---

### **18. 使用 `torch.utils.data` 實現一個多進程數據加載器**

#### 代碼

```python
from torch.utils.data import DataLoader, Dataset

# 簡單數據集
class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 測試數據
data = list(range(100))
dataset = SimpleDataset(data)

# 創建多進程 DataLoader
dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=4)

# 遍歷批量數據
for batch in dataloader:
    print("Batch:", batch)

```
#### 中文解釋

1. 定義簡單數據集 `SimpleDataset`。
2. 使用 `DataLoader`，設置 `num_workers=4` 啟用多進程加載數據。
3. 測試批量數據加載，打印每個批次的內容。

---

### **19. 將 MNIST 數據集保存為 PyTorch 的 `pt` 格式，並從中加載數據**

#### 代碼

```python
from torchvision import datasets
import torch

# 加載 MNIST 數據集
mnist_train = datasets.MNIST(root='./data', train=True, download=True)
mnist_test = datasets.MNIST(root='./data', train=False, download=True)

# 保存數據集為 .pt 格式
torch.save(mnist_train, 'mnist_train.pt')
torch.save(mnist_test, 'mnist_test.pt')

# 從文件加載數據集
loaded_train = torch.load('mnist_train.pt')
loaded_test = torch.load('mnist_test.pt')

print("Number of training samples:", len(loaded_train))
print("Number of test samples:", len(loaded_test))

```
#### 中文解釋

1. 使用 `torchvision.datasets.MNIST` 加載 MNIST 數據集。
2. 使用 `torch.save` 保存數據集為 `.pt` 格式。
3. 使用 `torch.load` 從保存的文件中加載數據集。
4. 打印加載後的數據集大小。

---

### **20. 編寫代碼對每個批次的數據進行標籤平衡處理**

#### 代碼

```python
from collections import Counter
import random
import torch
from torch.utils.data import Dataset, DataLoader

class ImbalancedDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 自定義 collate_fn 進行標籤平衡
def balanced_collate_fn(batch):
    data, labels = zip(*batch)
    label_counts = Counter(labels)
    max_count = max(label_counts.values())
    balanced_data, balanced_labels = [], []

    for label, count in label_counts.items():
        label_data = [data[i] for i in range(len(labels)) if labels[i] == label]
        label_labels = [label] * len(label_data)
        extra = random.choices(label_data, k=max_count - len(label_data))
        balanced_data.extend(label_data + extra)
        balanced_labels.extend(label_labels + [label] * len(extra))

    return torch.stack(balanced_data), torch.tensor(balanced_labels)

# 測試數據
data = [torch.tensor([i]) for i in range(10)]
labels = [0, 0, 1, 1, 1, 2, 2, 2, 2, 2]  # 標籤不平衡

dataset = ImbalancedDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=5, collate_fn=balanced_collate_fn)

# 測試 DataLoader
for batch_data, batch_labels in dataloader:
    print("Batch Data:", batch_data)
    print("Batch Labels:", batch_labels)

```
#### 中文解釋

1. 定義不平衡數據集 `ImbalancedDataset`。
2. 使用自定義 `collate_fn` 平衡標籤：
    - 計算每個標籤的數量。
    - 對少數類別進行過採樣（使用 `random.choices`）。
3. 測試 DataLoader，確認每個批次中標籤的分佈被平衡。

### **21. 使用 PyTorch 的 `Dataset` 和 `DataLoader` 類別來加載自定義數據集**

#### 代碼
```python
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 自定義數據
data = torch.arange(10).view(-1, 1).float()  # 10 個數據，每個一維
labels = torch.arange(10) % 2  # 二分類標籤

# 創建 Dataset 和 DataLoader
dataset = CustomDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

# 測試 DataLoader
for batch_data, batch_labels in dataloader:
    print("Batch Data:\n", batch_data)
    print("Batch Labels:\n", batch_labels)

```
#### 中文解釋

1. 定義一個繼承自 `Dataset` 的類，實現 `__len__` 和 `__getitem__` 方法。
2. `__len__` 返回數據集大小；`__getitem__` 返回指定索引的數據和標籤。
3. 使用 `DataLoader` 將數據按批量進行加載並隨機打亂。

---

### **22. 使用 `torchvision` 來進行數據增強（Data Augmentation）**

#### 代碼

```python
import torchvision.transforms as transforms
from PIL import Image

# 定義數據增強操作
data_augmentations = transforms.Compose([
    transforms.RandomResizedCrop(224),  # 隨機裁剪到 224x224
    transforms.RandomHorizontalFlip(),  # 隨機水平翻轉
    transforms.ColorJitter(brightness=0.5, contrast=0.5),  # 隨機調整亮度和對比度
    transforms.ToTensor()  # 轉換為張量
])

# 測試數據增強
sample_image = Image.open("sample.jpg")  # 替換為真實圖片路徑
augmented_image = data_augmentations(sample_image)

print("Original Image Size:", sample_image.size)
print("Transformed Image Shape:", augmented_image.shape)

```
#### 中文解釋

1. 使用 `torchvision.transforms` 定義多個數據增強操作。
2. `RandomResizedCrop` 進行隨機裁剪，`RandomHorizontalFlip` 進行隨機翻轉。
3. 使用 `ColorJitter` 調整圖像亮度和對比度。
4. 測試數據增強操作，並打印增強後的圖像形狀。

---

### **23. 使用 `torchtext` 來處理文本數據**

#### 代碼

```python
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# 定義示例文本數據
data = [
    "This is a PyTorch example.",
    "TorchText makes text processing easy.",
    "You can create custom datasets for NLP."
]

# 分詞器
tokenizer = get_tokenizer("basic_english")

# 定義詞彙表構建函數
def yield_tokens(data):
    for sentence in data:
        yield tokenizer(sentence)

# 構建詞彙表
vocab = build_vocab_from_iterator(yield_tokens(data), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# 測試詞彙表
print("Vocabulary size:", len(vocab))
print("Token IDs for a sentence:", vocab(tokenizer("This is an example")))

```
#### 中文解釋

1. 使用 `torchtext.data.utils.get_tokenizer` 定義分詞器。
2. 使用 `build_vocab_from_iterator` 根據數據構建詞彙表，並添加特殊標籤 `<unk>` 表示未知單詞。
3. 測試詞彙表功能，將句子轉換為對應的詞 ID。

---

### **24. 使用 `torch.utils.data.DataLoader` 進行大規模數據的批量處理**

#### 代碼

```python
from torch.utils.data import DataLoader, Dataset

# 定義大規模數據集
class LargeDataset(Dataset):
    def __init__(self, size):
        self.data = torch.arange(size).view(-1, 1).float()
        self.labels = torch.arange(size) % 2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 創建數據集和 DataLoader
dataset = LargeDataset(10000)  # 1 萬條數據
dataloader = DataLoader(dataset, batch_size=256, num_workers=4, shuffle=True)

# 批量處理
for batch_data, batch_labels in dataloader:
    print("Batch Data Shape:", batch_data.shape)
    print("Batch Labels Shape:", batch_labels.shape)
    break

```
#### 中文解釋

1. 定義一個包含 1 萬條數據的數據集。
2. 使用 `DataLoader` 進行批量處理，設置 `num_workers=4` 加快數據加載速度。
3. 測試批量加載，打印每批數據和標籤的形狀。

---

### **25. 使用 `torch.nn.utils.rnn` 來處理可變長度的序列數據**

#### 代碼
```python
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# 模擬可變長度序列
sequences = [torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6])]

# 填充序列到相同長度
padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)

# 序列長度
lengths = torch.tensor([len(seq) for seq in sequences])

# 打包序列
packed_sequences = pack_padded_sequence(padded_sequences, lengths, batch_first=True, enforce_sorted=False)

# 解包序列
unpacked_sequences, unpacked_lengths = pad_packed_sequence(packed_sequences, batch_first=True)

print("Padded Sequences:\n", padded_sequences)
print("Packed Sequences:\n", packed_sequences)
print("Unpacked Sequences:\n", unpacked_sequences)

```
#### 中文解釋

1. 使用 `pad_sequence` 將不同長度的序列填充為相同長度。
2. 使用 `pack_padded_sequence` 將填充的序列壓縮為有效數據，忽略填充部分。
3. 使用 `pad_packed_sequence` 還原壓縮後的序列，適合後續操作。
4. 測試填充、壓縮和解壓的功能，輸出中間結果。

### **26. 使用 PyTorch 創建一個全連接神經網絡，用於二分類任務**

#### 代碼

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 定義全連接神經網絡
class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# 初始化模型
model = BinaryClassifier(input_size=10, hidden_size=16)
criterion = nn.BCELoss()  # 二分類交叉熵損失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 模擬數據
data = torch.randn(8, 10)  # 8 條數據，每條 10 維
labels = torch.randint(0, 2, (8, 1)).float()  # 二分類標籤

# 前向傳播
outputs = model(data)
loss = criterion(outputs, labels)

# 反向傳播
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Loss:", loss.item())

```
#### 中文解釋

1. 定義一個兩層的全連接網絡，使用 ReLU 激活和 Sigmoid 作為輸出層激活函數。
2. 使用 `BCELoss` 作為二分類損失函數。
3. 前向傳播計算損失，後向傳播更新權重。

perplexity
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size1=128, hidden_size2=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size1),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size2),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

# 資料集類別
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 訓練函數
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted.squeeze() == labels).sum().item()
    
    accuracy = 100 * correct / total
    return running_loss / len(train_loader), accuracy

# 評估函數
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted.squeeze() == labels).sum().item()
    
    accuracy = 100 * correct / total
    return running_loss / len(test_loader), accuracy

# 主程式
def main():
    # 設定超參數
    input_size = 10  # 特徵數量
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 50
    
    # 創建隨機數據（實際應用中替換為真實數據）
    X_train = np.random.randn(1000, input_size)
    y_train = np.random.randint(0, 2, 1000)
    X_test = np.random.randn(200, input_size)
    y_test = np.random.randint(0, 2, 200)
    
    # 創建數據加載器
    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 設定設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 創建模型
    model = BinaryClassifier(input_size).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 訓練循環
    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

if __name__ == '__main__':
    main()

```

---

### **27. 編寫代碼實現一個 CNN 用於處理 CIFAR-10 圖像分類**

#### 代碼

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 定義 CNN 模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 加載 CIFAR-10 數據集
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# 初始化模型和優化器
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 訓練迭代
for images, labels in train_loader:
    outputs = model(images)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Loss:", loss.item())
    break  # 僅執行一個批次測試代碼

```
#### 中文解釋

1. 定義一個包含兩個卷積層和全連接層的 CNN，用於 CIFAR-10 的圖像分類。
2. 使用交叉熵損失函數和 Adam 優化器。
3. 測試模型，運行一個批次的數據。

#### **1. `for images, labels in train_loader:`**

##### **作用**

- `train_loader` 是一個 **數據加載器（DataLoader）**，它可以自動從數據集讀取批量（batch）數據。
- 這是一個**迴圈（Loop）**，每次從 `train_loader` 讀取一個 batch（批量）的 `images` 和 `labels`，直到遍歷完整個訓練集。

##### ** 具體步驟**

1. **`train_loader` 內部使用了 `Dataset` 和 `DataLoader`**：
    
    - `Dataset` 會返回 `(image, label)`。
    - `DataLoader` 會自動批量化這些資料。
2. **假設批量大小 `batch_size=32`，則**：
    
    - `images` 形狀（假設輸入為 1024×10241024 \times 10241024×1024 RGB 影像）：`(32, 3, 1024, 1024)`
    - `labels` 形狀（假設是 10 類分類任務）：`(32,)`（每個數值代表對應影像的類別）

---

#### ** 2. `outputs = model(images)`**

##### **作用**

- **進行前向傳播（Forward Pass）**，讓 `images` 經過 `model`，計算預測結果 `outputs`。

##### **具體步驟**

1. **輸入 `images` 進入 `model`**

    `outputs = model(images)`
    
2. **模型執行前向傳播，通常會經過多層結構，如 CNN 或 Transformer**

    `# 假設是一個 CNN x = conv1(images)  # 第一層卷積 x = relu(x)        # 激活函數 x = fc(x)          # 最後的全連接層`
    
3. **`outputs` 是模型對每個影像的預測結果**
    - 假設分類問題，`outputs` 的形狀通常是 `(32, 10)`，代表 32 張圖片，每張圖片有 10 個類別的預測分數。

---

#### ** 3. `loss = criterion(outputs, labels)`**

##### ** 作用**

- 計算模型預測 (`outputs`) 與真實標籤 (`labels`) 之間的損失（Loss）。

##### ** 具體步驟**

1. **損失函數 `criterion` 計算誤差**

    `criterion = nn.CrossEntropyLoss()`
    
2. **計算 `outputs`（預測） 與 `labels`（真實值） 之間的差異**

    `loss = criterion(outputs, labels)`
    
3. **舉例**
    - **假設 `outputs`（模型預測結果）**：

        `tensor([[0.1, 2.5, -1.2, 1.8, 0.6], ...])  # 5 個類別的 logits 分數`
        
    - **假設 `labels`（真實標籤）**：

        `tensor([1, 3, 0, 4, 2, ...])  # 對應影像的真實類別`
        
    - **交叉熵損失（CrossEntropyLoss）會比較 `outputs` 與 `labels`，計算損失值 `loss`**。

---

#### ** 4. `optimizer.zero_grad()`**

##### ** 作用**

- **清除舊的梯度資訊，以免影響下一步的梯度計算。**

##### **具體步驟**

1. **在 PyTorch 中，每次執行 `loss.backward()` 時，梯度會** **累積（accumulate）**。

    `optimizer.zero_grad()`
    
    **這行代碼將 `model.parameters()` 內所有的 `.grad` 屬性歸零。**
2. **如果不加這一步，梯度會累積，導致錯誤的梯度更新！**

---

#### ** 5. `loss.backward()`**

##### **作用**

- **進行反向傳播（Backward Propagation），計算每個參數的梯度。**
- **梯度會儲存在 `model.parameters()` 內的 `.grad` 屬性中。**

##### ** 具體步驟**

1. **計算 `loss` 相對於 `model.parameters()` 的梯度**

    `loss.backward()`
    
2. **PyTorch 自動計算所有可訓練參數的梯度**

    `weight.grad = ∂Loss / ∂weight bias.grad = ∂Loss / ∂bias`
    
3. **每個 `model.parameters()` 內的 `requires_grad=True` 的變數，會獲得梯度資訊。**

---

#### ** 6. `optimizer.step()`**

##### ** 作用**

- **根據計算出的梯度，更新模型的參數（權重和偏置）。**
- **讓模型在下一次前向傳播時使用新的參數。**

##### ** 具體步驟**

1. **Adam 優化器使用梯度來更新參數**

    `optimizer.step()`
    
2. **具體的數學公式（以 SGD 為例）**

    `weight = weight - learning_rate * weight.grad bias = bias - learning_rate * bias.grad`
    
3. **這一步完成後，模型的 `model.parameters()` 內的權重已經更新。**

## **CNN 設計時該用哪一個？**
**✔️ 推薦**
1. **`nn.ReLU()`** → 如果你在 `__init__()` 定義層，並希望 ReLU 作為 **模型的一部分**。
2. **`F.relu()`** → 如果你希望在 `forward()` **直接調用 ReLU，而不額外定義 ReLU 層**。
**❌ 避免**
- **`torch.relu()`** → 不適合 CNN 設計，因為它不屬於 `torch.nn.Module`，不能存儲於 `state_dict()`，也不支持 `inplace=True`。

perplexity
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 定義 CNN 模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 卷積層
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        # 池化層
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全連接層
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        
        # Dropout層
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # 三個卷積-池化層
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        
        # 展平操作
        x = x.view(-1, 128 * 4 * 4)
        
        # 全連接層
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def train_model():
    # 數據預處理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加載數據
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                           shuffle=False, num_workers=2)

    # 設置設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 創建模型
    model = CNN().to(device)
    
    # 定義損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 訓練模型
    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 200 == 199:
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0

    print('訓練完成')
    return model, testloader, device

def evaluate_model(model, testloader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'準確率: {100 * correct / total}%')

if __name__ == "__main__":
    model, testloader, device = train_model()
    evaluate_model(model, testloader, device)

```


---

### **28. 編寫代碼實現一個自定義的激活函數，如 Swish 或 GELU**

#### 代碼

```python
import torch
import torch.nn as nn

# 定義 Swish 激活函數
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# 測試 Swish 激活函數
activation = Swish()
input_tensor = torch.tensor([-1.0, 0.0, 1.0, 2.0])
output_tensor = activation(input_tensor)

print("Input Tensor:", input_tensor)
print("Output Tensor:", output_tensor)

```
#### 中文解釋

1. 定義 Swish 激活函數，公式為 $x * \text{sigmoid}(x)$。
2. 使用自定義激活函數進行測試，輸入張量包含正數和負數。


perplexity
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np

class Swish(nn.Module):
    """
    Swish 激活函數: x * sigmoid(beta * x)
    beta 是可學習參數或固定值
    """
    def __init__(self, beta=1.0, trainable=False):
        super().__init__()
        self.trainable = trainable
        if trainable:
            # 創建可學習的 beta 參數
            self.beta = nn.Parameter(torch.tensor(beta))
        else:
            # 創建固定的 beta 值
            self.register_buffer('beta', torch.tensor(beta))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

class GELU(nn.Module):
    """
    GELU (Gaussian Error Linear Unit) 激活函數
    GELU(x) = x * Φ(x)
    其中 Φ(x) 是標準正態分佈的累積分佈函數
    """
    def __init__(self, approximate='tanh'):
        super().__init__()
        self.approximate = approximate

    def forward(self, x):
        if self.approximate == 'tanh':
            # 使用 tanh 近似
            return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        else:
            # 使用精確計算
            return x * 0.5 * (1 + torch.erf(x / math.sqrt(2)))

def plot_activation_functions():
    """
    繪製激活函數的圖像進行比較
    """
    x = np.linspace(-5, 5, 1000)
    
    # 轉換為 PyTorch tensor
    x_tensor = torch.FloatTensor(x)
    
    # 初始化激活函數
    swish = Swish(beta=1.0)
    swish2 = Swish(beta=2.0)
    gelu = GELU()
    gelu_exact = GELU(approximate=None)
    
    # 計算輸出
    y_swish = swish(x_tensor).numpy()
    y_swish2 = swish2(x_tensor).numpy()
    y_gelu = gelu(x_tensor).numpy()
    y_gelu_exact = gelu_exact(x_tensor).numpy()
    
    # 繪製圖像
    plt.figure(figsize=(12, 8))
    plt.plot(x, y_swish, label='Swish (β=1.0)')
    plt.plot(x, y_swish2, label='Swish (β=2.0)')
    plt.plot(x, y_gelu, label='GELU (tanh approximation)')
    plt.plot(x, y_gelu_exact, label='GELU (exact)', linestyle='--')
    plt.plot(x, x, label='Linear', linestyle=':')
    plt.grid(True)
    plt.legend()
    plt.title('Activation Functions Comparison')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()

# 示例：在簡單網絡中使用自定義激活函數
class SimpleNet(nn.Module):
    def __init__(self, activation='swish'):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
        
        # 選擇激活函數
        if activation == 'swish':
            self.activation = Swish(beta=1.0, trainable=True)
        elif activation == 'gelu':
            self.activation = GELU()
        else:
            self.activation = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

def test_activation_gradients():
    """
    測試激活函數的梯度
    """
    # 創建輸入數據
    x = torch.randn(10, 5, requires_grad=True)
    
    # 測試 Swish
    swish = Swish(beta=1.0, trainable=True)
    y_swish = swish(x)
    loss_swish = y_swish.sum()
    loss_swish.backward()
    print("Swish gradient check:")
    print(f"Input gradient shape: {x.grad.shape}")
    print(f"Beta gradient: {swish.beta.grad}")
    
    # 重置梯度
    x.grad = None
    
    # 測試 GELU
    gelu = GELU()
    y_gelu = gelu(x)
    loss_gelu = y_gelu.sum()
    loss_gelu.backward()
    print("\nGELU gradient check:")
    print(f"Input gradient shape: {x.grad.shape}")

if __name__ == "__main__":
    # 繪製激活函數比較圖
    plot_activation_functions()
    
    # 測試梯度計算
    test_activation_gradients()
    
    # 創建使用自定義激活函數的網絡
    model_swish = SimpleNet(activation='swish')
    model_gelu = SimpleNet(activation='gelu')
    
    # 打印模型結構
    print("\nModel with Swish activation:")
    print(model_swish)
    print("\nModel with GELU activation:")
    print(model_gelu)

```


| 激活函數        | 特點/適用場景               | 注意事項              | PyTorch Code                         |
| ----------- | --------------------- | ----------------- | ------------------------------------ |
| **ReLU**    | 常用，快速，避免梯度消失          | 可能出現 "死亡 ReLU" 問題 | `nn.ReLU()`                          |
| Leaky ReLU  | 解決 ReLU 的 "死亡" 問題     | 增加了超參數 α\alphaα   | `nn.LeakyReLU(negative_slope=0.01)`  |
| Swish       | 性能優異，適合高性能場景          | 計算稍慢              | `nn.SiLU()`                          |
| **Sigmoid** | 二分類輸出，壓縮到 (0,1)       | 容易梯度消失            | `nn.Sigmoid()`                       |
| **Tanh**    | 中心化輸出，適合 RNN          | 容易梯度消失            | `nn.Tanh()`                          |
| **Softmax** | 多分類輸出                 | 必須配合交叉熵損失使用       | `nn.Softmax(dim=1)`                  |
| ELU         | 負輸出不飽和，適合梯度敏感場景       | 增加了計算成本           | `nn.ELU(alpha=1.0)`                  |
| GELU        | 用於 Transformer，梯度流動良好 | 計算稍慢              | `nn.GELU()`                          |
| Softplus    | ReLU 的平滑版本，避免非平滑問題    | 計算開銷稍大            | `nn.Softplus()`                      |
| Hardtanh    | 範圍限制，適合特殊場景           | 功能有限              | `nn.Hardtanh(min_val=-1, max_val=1)` |


---

### **29. 實現一個自定義的 PyTorch 損失函數，用於計算 Focal Loss**

#### 代碼

```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        focal_weight = self.alpha * (1 - probs) ** self.gamma
        return (focal_weight * bce_loss).mean()

# 測試 Focal Loss
loss_fn = FocalLoss()
logits = torch.tensor([0.5, -0.5, 1.5, -1.5])
targets = torch.tensor([1.0, 0.0, 1.0, 0.0])

loss = loss_fn(logits, targets)
print("Focal Loss:", loss.item())

```
#### 中文解釋

1. 定義 Focal Loss，適合處理類別不平衡問題。
2. 使用 `binary_cross_entropy_with_logits` 計算 BCE 損失，並加權計算焦點損失。
3. 測試自定義 Focal Loss，輸出損失值。


perplexity
```python

```


### **常用的損失函數列表**

| **損失函數**                              | **適用場景**                                                   | **注意事項**                                                                       | **PyTorch Code**                |
| ------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------- |
| **MSELoss (均方誤差損失)**                  | 用於回歸任務，度量輸出與目標之間的平方誤差。                                     | 輸出值應與目標值在相同範圍內；對於異常值敏感。                                                        | `nn.MSELoss()`                  |
| **CrossEntropyLoss (交叉熵損失)**          | 多分類任務，例如圖像分類或 NLP 任務。                                      | 自動包含 `Softmax` 操作，輸入值應為未經激活的 logits（例如線性層的輸出）。                                 | `nn.CrossEntropyLoss()`         |
| **BCELoss (二元交叉熵損失)**                 | 二分類任務，例如二元分類模型的輸出為 [0, 1] 的概率。                             | 需要在模型輸出層使用 `Sigmoid` 激活，否則輸入的值不在正確範圍內。                                         | `nn.BCELoss()`                  |
| **BCEWithLogitsLoss**                 | 二分類任務，與 `BCELoss` 類似，但內部自帶 `Sigmoid` 激活，輸入可以為未經激活的 logits。 | 適合直接處理 logits，數值穩定性更好。                                                         | `nn.BCEWithLogitsLoss()`        |
| **SmoothL1Loss (平滑 L1 損失)**           | 用於回歸任務，例如目標檢測中的邊界框回歸。                                      | 在 L1 和 L2 損失之間取得平衡，對異常值的敏感性低於 `MSELoss`。                                       | `nn.SmoothL1Loss()`             |
| **L1Loss (絕對誤差損失)**                   | 回歸任務，度量輸出與目標的絕對差值。                                         | 對異常值敏感，通常用於需要對每個數值差異同等看待的情況。                                                   | `nn.L1Loss()`                   |
| **HuberLoss**                         | 回歸任務，平衡 L1 和 L2 損失的優點，對異常值不敏感。                             | 在異常值處過渡平滑，但計算稍慢。                                                               | `nn.HuberLoss()`                |
| **MarginRankingLoss**                 | 用於排序任務，例如學習輸出之間的排序關係（例如相似度學習）。                             | 輸入兩個值及其目標標籤 yyy，yyy 應為 +1 或 -1。                                                | `nn.MarginRankingLoss()`        |
| **KLDivLoss (Kullback-Leibler 散度損失)** | 用於測量兩個概率分佈之間的差異，例如分佈對齊問題（知識蒸餾）。                            | 輸入應該是 log 概率分佈（可以用 `log_softmax` 計算），目標分佈應是概率分佈（非 log）。                        | `nn.KLDivLoss()`                |
| **CosineEmbeddingLoss**               | 用於度量輸出向量之間的餘弦相似度，適合於相似性學習任務。                               | 當目標標籤 yyy 為 +1 時，表示相似；當目標標籤 yyy 為 -1 時，表示不相似。                                  | `nn.CosineEmbeddingLoss()`      |
| **HingeEmbeddingLoss**                | 用於度量輸出向量之間的嵌入相似性，例如支持向量機（SVM）。                             | 當目標標籤 yyy 為 +1 時，計算正常值；當目標標籤 yyy 為 -1 時，計算懲罰值。                                 | `nn.HingeEmbeddingLoss()`       |
| **CTCLoss (連接時序分類損失)**                | 用於序列任務，例如語音識別、OCR 中的無對齊學習。                                 | 適合輸入長度與目標長度不同的場景，需要提供序列長度。                                                     | `nn.CTCLoss()`                  |
| **NLLLoss (負對數似然損失)**                 | 多分類任務，用於概率分佈的對數損失計算。                                       | 通常與 `log_softmax` 一起使用，輸入應為 log 概率分佈。                                          | `nn.NLLLoss()`                  |
| **TripletMarginLoss**                 | 用於學習嵌入表示，特別是三元組學習任務（例如人臉識別）。                               | 需要提供 Anchor、Positive 和 Negative 的嵌入向量，設計目的是使 Anchor 更接近 Positive 而遠離 Negative。 | `nn.TripletMarginLoss()`        |
| **MultiLabelSoftMarginLoss**          | 用於多標籤分類任務，類似於交叉熵損失，但處理多標籤場景。                               | 自動應用 `Sigmoid` 激活，輸出可以是多個標籤的概率值。                                               | `nn.MultiLabelSoftMarginLoss()` |
| **PoissonNLLLoss**                    | 用於計算目標值服從泊松分佈的損失，例如計數型數據建模。                                | 對數目標值計算損失，適合處理泊松分佈的情況。                                                         | `nn.PoissonNLLLoss()`           |
| **BinaryFocalLoss (需自定義)**            | 用於處理樣本不均衡問題，例如目標檢測。                                        | PyTorch 沒有內建實現，通常需要自己實現 Focal Loss。                                            | 需自定義實現                          |

---

### **30. 使用 PyTorch 創建一個簡單的 RNN 處理時間序列數據**

#### 代碼

```python
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # 僅取最後一個時間步
        return out

# 初始化 RNN 模型
model = SimpleRNN(input_size=1, hidden_size=16, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 模擬時間序列數據
inputs = torch.randn(8, 10, 1)  # 8 條序列，每條 10 步，每步 1 維
targets = torch.randn(8, 1)  # 對應目標

# 前向傳播
outputs = model(inputs)
loss = criterion(outputs, targets)

# 反向傳播
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Loss:", loss.item())

```
#### 中文解釋

1. 定義一個簡單的 RNN，包含一個 RNN 層和一個全連接層。
2. 模擬時間序列數據，輸入為三維張量（批量大小、序列長度、特徵數）。
3. 使用均方誤差損失函數訓練模型。

perplexity
```python

```


### **31. 使用 `torch.nn.Transformer` 實現一個文本分類模型**

#### 代碼

```python
import torch
import torch.nn as nn

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_classes, num_layers, max_seq_length):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(max_seq_length, embed_size))
        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=num_heads,
            num_encoder_layers=num_layers
        )
        self.fc = nn.Linear(embed_size, num_classes)

    def forward(self, x):
        embedded = self.embedding(x) + self.positional_encoding[:x.size(1), :]
        transformer_out = self.transformer(embedded, embedded)
        out = self.fc(transformer_out.mean(dim=1))  # 取序列平均作為輸出
        return out

# 模擬數據
vocab_size, embed_size, num_heads, num_classes, num_layers, max_seq_length = 1000, 128, 4, 3, 2, 50
model = TransformerClassifier(vocab_size, embed_size, num_heads, num_classes, num_layers, max_seq_length)
input_data = torch.randint(0, vocab_size, (8, max_seq_length))  # 8 條序列，每條長度 50
outputs = model(input_data)
print("Output Shape:", outputs.shape)

```
#### 中文解釋

1. 使用嵌入層和位置編碼處理輸入文本。
2. 使用 `torch.nn.Transformer` 實現編碼器，設置多頭注意力和多層結構。
3. 全連接層將 Transformer 輸出映射到分類類別。

---

### **32. 實現一個自定義的 `torch.nn.Module`，包含多個子模塊和跳躍連接**

#### 代碼

```python
class CustomModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        skip = self.fc1(x)
        x = self.relu(skip)
        x = self.fc2(x)
        x = x + skip  # 跳躍連接
        x = self.fc3(x)
        return x

# 測試模型
model = CustomModel(input_size=10, hidden_size=20, output_size=1)
input_data = torch.randn(5, 10)
outputs = model(input_data)
print("Output Shape:", outputs.shape)

```
#### 中文解釋

1. 定義三層網絡，其中第二層與第一層有跳躍連接。
2. 跳躍連接通過加法操作保留前層信息，提升梯度流動能力。
3. 測試模型輸入和輸出形狀。


perplexity
```python

```


---

### **33. 設計一個具有多輸入的模型，例如處理圖像和文本同時進行分類**

#### 代碼

```python
class MultiInputModel(nn.Module):
    def __init__(self, img_feature_size, text_feature_size, hidden_size, num_classes):
        super(MultiInputModel, self).__init__()
        self.img_fc = nn.Linear(img_feature_size, hidden_size)
        self.text_fc = nn.Linear(text_feature_size, hidden_size)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, img_input, text_input):
        img_features = self.img_fc(img_input)
        text_features = self.text_fc(text_input)
        combined = torch.cat((img_features, text_features), dim=1)
        output = self.classifier(combined)
        return output

# 測試模型
model = MultiInputModel(img_feature_size=128, text_feature_size=64, hidden_size=32, num_classes=5)
img_input = torch.randn(8, 128)
text_input = torch.randn(8, 64)
outputs = model(img_input, text_input)
print("Output Shape:", outputs.shape)

```
#### 中文解釋

1. 定義兩個全連接層分別處理圖像和文本特徵。
2. 使用 `torch.cat` 將圖像和文本特徵拼接。
3. 最後使用全連接層進行分類。

perplexity
```python

```

---

### **34. 實現一個生成對抗網絡（GAN），用於生成手寫數字**

#### 代碼

```python
import tensor
import tensor.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim, output_dim):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.fc(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

# 測試 GAN 模型
noise_dim = 100
output_dim = 28 * 28
generator = Generator(noise_dim, output_dim)
discriminator = Discriminator(output_dim)

# 模擬訓練
noise = torch.randn(16, noise_dim)
fake_images = generator(noise)
print("Fake Image Shape:", fake_images.shape)
real_or_fake = discriminator(fake_images)
print("Discriminator Output Shape:", real_or_fake.shape)

```
#### 中文解釋

1. 定義生成器，輸入噪聲生成圖像。
2. 定義判別器，輸入圖像判斷真假。
3. 測試生成和判別過程。

perplexity:
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.main(x).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x.view(-1, 784)
        return self.main(x)

def train_gan(epochs=100, batch_size=64, noise_dim=100):
    # 資料載入
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = torchvision.datasets.MNIST(root='./data', 
                                       train=True,
                                       transform=transform,
                                       download=True)
    
    dataloader = torch.utils.data.DataLoader(dataset, 
                                           batch_size=batch_size,
                                           shuffle=True)
    
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(noise_dim).to(device)
    discriminator = Discriminator().to(device)
    
    # 損失函數和優化器
    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
    
    for epoch in range(epochs):
        for i, (real_images, _) in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_label = torch.ones(batch_size, 1).to(device)
            fake_label = torch.zeros(batch_size, 1).to(device)
            
            # 訓練判別器
            real_images = real_images.to(device)
            d_output_real = discriminator(real_images)
            d_loss_real = criterion(d_output_real, real_label)
            
            noise = torch.randn(batch_size, noise_dim).to(device)
            fake_images = generator(noise)
            d_output_fake = discriminator(fake_images.detach())
            d_loss_fake = criterion(d_output_fake, fake_label)
            
            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            
            # 訓練生成器
            g_output = discriminator(fake_images)
            g_loss = criterion(g_output, real_label)
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')
    
    return generator, discriminator

```

```python
def generate_samples(generator, noise_dim=100, num_samples=16):
    with torch.no_grad():
        noise = torch.randn(num_samples, noise_dim).to(device)
        generated_images = generator(noise)
        generated_images = generated_images.cpu().numpy()
        
    plt.figure(figsize=(4, 4))
    for i in range(num_samples):
        plt.subplot(4, 4, i+1)
        plt.imshow(generated_images[i][0], cmap='gray')
        plt.axis('off')
    plt.show()

```

---

### **35. 設計一個基於注意力機制的模型，用於機器翻譯**

#### 代碼

```python
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden, encoder_outputs):
        attn_weights = self.softmax(self.attn(torch.cat((hidden, encoder_outputs), dim=1)))
        context = torch.sum(attn_weights * encoder_outputs, dim=1)
        return context, attn_weights

class Seq2SeqModel(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim):
        super(Seq2SeqModel, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, src, tgt):
        encoder_outputs, hidden = self.encoder(src)
        decoder_outputs, _ = self.decoder(tgt, hidden)
        context, _ = self.attention(decoder_outputs, encoder_outputs)
        output = self.fc(context)
        return output

# 測試模型
src = torch.randn(8, 10, 16)  # 8 條輸入序列，每條 10 步，每步 16 維
tgt = torch.randn(8, 5, 16)   # 8 條目標序列，每條 5 步，每步 16 維
model = Seq2SeqModel(input_dim=16, hidden_size=32, output_dim=50)
outputs = model(src, tgt)
print("Output Shape:", outputs.shape)

```
#### 中文解釋

1. 定義注意力機制計算上下文向量。
2. 使用 LSTM 作為編碼器和解碼器。
3. 通過注意力機制結合上下文信息，輸出翻譯結果。

perplexity
```python

```

### **36. 實現一個Transformer神經網絡**

#### 代碼

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 線性轉換並分割成多頭
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 注意力計算
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = torch.softmax(scores, dim=-1)
        out = torch.matmul(attention, v)
        
        # 重組並輸出
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 自注意力層
        attention = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention))
        
        # 前向網路層
        ff = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff))
        return x

class Transformer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, num_layers=6, d_ff=2048, 
                 max_seq_length=100, vocab_size=5000):
        super(Transformer, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self.create_positional_encoding(max_seq_length, d_model)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        
        self.fc = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(0.1)
        
    def create_positional_encoding(self, max_seq_length, d_model):
        pos_encoding = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding.unsqueeze(0)
        
    def forward(self, x, mask=None):
        seq_length = x.size(1)
        
        # 嵌入層和位置編碼
        x = self.embedding(x)
        x = x + self.pos_encoding[:, :seq_length, :].to(x.device)
        x = self.dropout(x)
        
        # Transformer層
        for transformer in self.transformer_blocks:
            x = transformer(x, mask)
            
        # 輸出層
        output = self.fc(x)
        return output


```
## 程式碼說明

1. **MultiHeadAttention類別**:

- 實現多頭注意力機制
- 將輸入分割成多個頭進行並行處理
- 包含Query、Key、Value的線性轉換

2. **TransformerBlock類別**:

- 包含一個完整的Transformer層
- 實現自注意力機制和前向網路
- 使用Layer Normalization和殘差連接

3. **Transformer類別**:

- 完整的Transformer模型實現
- 包含嵌入層和位置編碼
- 堆疊多個Transformer層
- 最後通過線性層輸出結果

4. **位置編碼**:

- 使用正弦和餘弦函數生成位置編碼
- 幫助模型理解序列中的位置信息

使用方式:
```python
# 準備輸入數據 x = torch.randint(0, 5000, (32, 100)) # batch_size=32, seq_length=100 output = model(x)
```

這個實現包含了Transformer的核心組件，適用於各種序列處理任務，如機器翻譯、文本生成等。

---

### **37. 使用預訓練的模型（如 ResNet）進行遷移學習**

#### 代碼

```python
from torchvision import models, transforms
import torch.nn as nn
import torch.optim as optim

# 加載預訓練的 ResNet
resnet = models.resnet18(pretrained=True)

# 冻结特徵提取部分的參數
for param in resnet.parameters():
    param.requires_grad = False

# 替換最後的全連接層
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, 10)  # 10 個分類

# 訓練配置
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.fc.parameters(), lr=0.001)

# 模擬數據
inputs = torch.randn(8, 3, 224, 224)  # 模擬 CIFAR-10 圖像數據
labels = torch.randint(0, 10, (8,))

# 訓練一個步驟
outputs = resnet(inputs)
loss = criterion(outputs, labels)
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Loss:", loss.item())

```
#### 中文解釋

1. 加載預訓練的 ResNet 模型，並凍結其特徵提取層。
2. 替換最後的全連接層以適應新任務。
3. 使用 CIFAR-10 類型的模擬數據進行單步訓練。

---

### **38. 實現一個vision Transformer神經網絡

#### 代碼
[[AI model Summary architecture###### Vision Transformer (ViT)]]

```python
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """將圖像分割成patch並進行線性嵌入"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        # (B, C, H, W) -> (B, D, H/P, W/P) -> (B, H/P*W/P, D)
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class Attention(nn.Module):
    """多頭自注意力機制"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    """多層感知機"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    """Transformer 編碼器塊"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, 
                 drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                            attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, 
                      act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    """Vision Transformer"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, 
                                    in_channels=in_channels, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                  norm_layer=norm_layer)
            for _ in range(depth)
        ])
        
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize cls token
        torch.nn.init.normal_(self.cls_token, std=.02)

        # Initialize position embedding
        torch.nn.init.normal_(self.pos_embed, std=.02)

        # Initialize all other Linear layers
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Apply Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)
        
        return x[:, 0]  # Return class token

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def create_vit_model(model_name='ViT-B/16', num_classes=1000, has_logits=True):
    """創建 ViT 模型"""
    configs = {
        'ViT-B/16': dict(patch_size=16, embed_dim=768, depth=12, num_heads=12),
        'ViT-L/16': dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16),
        'ViT-H/14': dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16),
    }
    
    config = configs[model_name]
    model = VisionTransformer(patch_size=config['patch_size'],
                            embed_dim=config['embed_dim'],
                            depth=config['depth'],
                            num_heads=config['num_heads'],
                            num_classes=num_classes)
    return model

# 測試代碼
if __name__ == "__main__":
    # 創建模型
    model = create_vit_model('ViT-B/16')
    
    # 測試輸入
    x = torch.randn(1, 3, 224, 224)
    
    # 前向傳播
    output = model(x)
    
    # 打印輸出形狀
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # 計算參數量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

# 創建模型
model = create_vit_model('ViT-B/16', num_classes=1000)

# 準備輸入
x = torch.randn(1, 3, 224, 224)

# 前向傳播
output = model(x)

```
## 程式碼說明

1. **PatchEmbed類別**:

- 將輸入圖像分割成固定大小的patch
- 使用卷積層進行patch embedding
- 將2D特徵圖轉換為序列形式

2. **Attention類別**:

- 實現多頭自注意力機制
- 包含Query、Key、Value的線性轉換
- 計算注意力權重並進行加權求和

3. **TransformerBlock類別**:

- 包含自注意力層和MLP層
- 使用Layer Normalization
- 實現殘差連接

4. **VisionTransformer類別**:

- 完整的ViT模型實現
- 包含patch embedding、位置編碼
- 加入特殊的CLS token用於分類
- 堆疊多個Transformer層

5. **MLP類別**:

- 實現前向網路
- 使用GELU激活函數
- 包含dropout正則化

使用方式:
```python
model = VisionTransformer(
    img_size=224,
    patch_size=16,
    in_channels=3,
    num_classes=1000,
    embed_dim=768,
    depth=12,
    num_heads=12
)

# 假設輸入圖像大小為 224x224
x = torch.randn(1, 3, 224, 224)
output = model(x)  # 輸出形狀: (1, 1000)

```



---

### **39. 實現一個自定義的損失函數，並在模型中使用**

#### 代碼

```python
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, predictions, targets):
        return torch.mean((predictions - targets) ** 2 + torch.abs(predictions - targets))

# 測試自定義損失
loss_fn = CustomLoss()
predictions = torch.tensor([2.0, 3.0, 4.0])
targets = torch.tensor([2.5, 2.0, 5.0])

loss = loss_fn(predictions, targets)
print("Custom Loss:", loss.item())

```
#### 中文解釋

1. 定義一個自定義損失函數，包括均方誤差和絕對誤差的組合。
2. 測試損失函數，計算給定輸入的損失值。

---

### **40. 如何在 PyTorch 中進行模型的序列化和反序列化？**

#### 代碼

```python
# 保存模型
torch.save(model.state_dict(), "model.pth")

# 加載模型
loaded_model = SimpleFeedForward(input_size=4, hidden_size=8, output_size=2)
loaded_model.load_state_dict(torch.load("model.pth"))

# 確認模型是否加載成功
print("Model Loaded Successfully")

```
#### 中文解釋

1. 使用 `torch.save` 將模型的參數保存到文件。
2. 使用 `load_state_dict` 將保存的參數加載到模型中。
3. 驗證加載成功後的模型是否可用。

### **41. 實現一個自注意力機制（Self-Attention）層**

#### 代碼

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.0):
        """
        初始化自注意力層
        
        參數:
        embed_dim: 輸入特徵的維度
        num_heads: 注意力頭的數量
        dropout: dropout 比率
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必須能被 num_heads 整除"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5  # 縮放因子
        
        # 定義線性變換層
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)  # 用於生成 query, key, value
        self.proj = nn.Linear(embed_dim, embed_dim)     # 輸出投影
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        前向傳播
        
        參數:
        x: 輸入張量, 形狀為 (batch_size, seq_len, embed_dim)
        mask: 可選的注意力遮罩
        
        返回:
        output: 注意力機制的輸出
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # 1. 生成 query, key, value
        qkv = self.qkv(x)  # (batch_size, seq_len, 3 * embed_dim)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 分離 query, key, value
        
        # 2. 計算注意力分數
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (batch_size, num_heads, seq_len, seq_len)
        
        # 3. 應用遮罩（如果有）
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        # 4. Softmax 歸一化
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 5. 與 value 相乘並重組
        x = (attn @ v).transpose(1, 2)  # (batch_size, seq_len, num_heads, head_dim)
        x = x.reshape(batch_size, seq_len, embed_dim)  # (batch_size, seq_len, embed_dim)
        
        # 6. 最終的線性投影
        x = self.proj(x)
        x = self.dropout(x)
        
        return x

# 測試代碼
if __name__ == "__main__":
    # 創建模型
    embed_dim = 256
    num_heads = 8
    attention = SelfAttention(embed_dim, num_heads)
    
    # 生成測試數據
    batch_size = 4
    seq_len = 10
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # 可選：創建注意力遮罩
    mask = torch.ones(batch_size, num_heads, seq_len, seq_len)
    
    # 前向傳播
    output = attention(x, mask)
    
    # 打印形狀
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")


```
#### 中文解釋

1. 自注意力層包含 Query、Key 和 Value 的線性投影。
2. 通過縮放點積計算注意力分數，並使用 Softmax 歸一化。
3. 將注意力分數加權到 Value，生成輸出。
4. 測試時輸入為三維張量（批量、序列長度、嵌入大小）。

perplexity
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        
        # 定義Query、Key、Value的線性轉換層
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        
        # Softmax用於注意力權重的計算
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x):
        # x的形狀: (batch_size, seq_length, input_dim)
        
        # 生成Query、Key、Value矩陣
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        
        # 計算注意力分數
        scores = torch.bmm(queries, keys.transpose(1, 2))
        
        # 縮放注意力分數
        scores = scores / (self.input_dim ** 0.5)
        
        # 應用softmax獲得注意力權重
        attention = self.softmax(scores)
        
        # 計算加權和
        weighted = torch.bmm(attention, values)
        
        return weighted

# 示例使用
batch_size = 32
seq_length = 10
input_dim = 256

# 創建模型實例
attention = SelfAttention(input_dim)

# 創建輸入數據
x = torch.randn(batch_size, seq_length, input_dim)

# 前向傳播
output = attention(x)


```


---

### **42. 實現一個自編碼器（Autoencoder）來進行數據壓縮**

#### 代碼

perplexity
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, encoding_dim=128):
        """
        初始化自編碼器
        
        參數:
        input_dim: 輸入維度 (例如 MNIST: 28*28=784)
        encoding_dim: 壓縮後的維度
        """
        super().__init__()
        
        # 編碼器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, encoding_dim),
            nn.ReLU()
        )
        
        # 解碼器
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()  # 輸出範圍限制在 [0,1]
        )
        
    def forward(self, x):
        # 編碼
        encoded = self.encoder(x)
        # 解碼
        decoded = self.decoder(encoded)
        return encoded, decoded

def train_autoencoder(model, train_loader, num_epochs=10, learning_rate=1e-3):
    """
    訓練自編碼器
    
    參數:
    model: 自編碼器模型
    train_loader: 訓練數據加載器
    num_epochs: 訓練輪數
    learning_rate: 學習率
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for data in train_loader:
            img, _ = data
            img = img.view(img.size(0), -1).to(device)
            
            # 前向傳播
            _, reconstructed = model(img)
            loss = criterion(reconstructed, img)
            
            # 反向傳播和優化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

def visualize_reconstruction(model, test_loader, num_images=5):
    """
    視覺化重建結果
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            img, _ = data
            img = img.view(img.size(0), -1).to(device)
            _, reconstructed = model(img)
            
            # 顯示原始圖像和重建圖像
            plt.figure(figsize=(12, 4))
            for i in range(num_images):
                # 原始圖像
                plt.subplot(2, num_images, i + 1)
                plt.imshow(img[i].cpu().view(28, 28), cmap='gray')
                plt.axis('off')
                if i == 0:
                    plt.title('Original Images')
                
                # 重建圖像
                plt.subplot(2, num_images, i + 1 + num_images)
                plt.imshow(reconstructed[i].cpu().view(28, 28), cmap='gray')
                plt.axis('off')
                if i == 0:
                    plt.title('Reconstructed Images')
            
            plt.tight_layout()
            plt.show()
            break

# 主程序
if __name__ == "__main__":
    # 設置數據轉換
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # 加載 MNIST 數據集
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # 創建模型
    input_dim = 28 * 28  # MNIST 圖像大小
    encoding_dim = 128   # 壓縮後的維度
    model = Autoencoder(input_dim, encoding_dim)
    
    # 訓練模型
    train_autoencoder(model, train_loader)
    
    # 視覺化結果
    visualize_reconstruction(model, test_loader)


# 創建自編碼器
autoencoder = Autoencoder(input_dim=784, encoding_dim=128)

# 準備數據
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST('./data', transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# 訓練模型
train_autoencoder(autoencoder, dataloader)

# 視覺化結果
visualize_reconstruction(autoencoder, dataloader)

```
#### 中文解釋

1. 自編碼器包括編碼器和解碼器兩部分。
2. 編碼器將高維數據壓縮為低維表示，解碼器重構原始數據。
3. 測試時輸入為高維數據，輸出壓縮表示和重構結果。


---

### **43. 實現一個圖神經網路（GNN）來進行節點分類**

#### 代碼

```python
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# 測試圖神經網絡
input_dim, hidden_dim, output_dim = 16, 32, 4
model = GCN(input_dim, hidden_dim, output_dim)
x = torch.randn(10, input_dim)  # 10 個節點的特徵
edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])  # 三條邊
output = model(x, edge_index)
print("Output Shape:", output.shape)

```
#### 中文解釋

1. 使用 PyG 中的 `GCNConv` 定義圖卷積層。
2. 每個節點特徵傳遞經過兩層卷積進行分類。
3. 測試時輸入節點特徵和圖的邊結構。

perplexity
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(GNN, self).__init__()
        # 第一層圖卷積
        self.conv1 = GCNConv(input_dim, hidden_dim)
        # 第二層圖卷積
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # 第一層: 圖卷積 + ReLU + Dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 第二層: 圖卷積 + softmax
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def train_gnn():
    # 加載 Cora 數據集
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]
    
    # 創建模型
    model = GNN(
        input_dim=dataset.num_features,
        hidden_dim=16,
        output_dim=dataset.num_classes
    )
    
    # 優化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # 訓練模型
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        # 前向傳播
        output = model(data.x, data.edge_index)
        # 計算訓練集上的損失
        loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
        # 反向傳播
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1:03d}, Loss: {loss.item():.4f}')
    
    return model, data

def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        output = model(data.x, data.edge_index)
        # 在測試集上計算準確率
        pred = output[data.test_mask].max(1)[1]
        correct = pred.eq(data.y[data.test_mask]).sum().item()
        acc = correct / data.test_mask.sum().item()
    return acc

if __name__ == "__main__":
    # 訓練模型
    model, data = train_gnn()
    
    # 評估模型
    acc = evaluate(model, data)
    print(f'Test Accuracy: {acc:.4f}')

```


---

### **44. 使用 `torch.distributions` 來實現變分自編碼器（VAE）**

#### 代碼

```python
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu, logvar = self.mu(encoded), self.logvar(encoded)
        z = self.reparameterize(mu, logvar)
        decoded = self.decoder(z)
        return decoded, mu, logvar

# 測試 VAE
input_dim, hidden_dim, latent_dim = 100, 64, 16
model = VAE(input_dim, hidden_dim, latent_dim)
data = torch.randn(32, input_dim)  # 批量大小 32
decoded, mu, logvar = model(data)
print("Decoded Shape:", decoded.shape)
print("Mu Shape:", mu.shape)
print("Logvar Shape:", logvar.shape)

```
#### 中文解釋

1. 定義 VAE 包括編碼器、重參數化和解碼器。
2. 重參數化將潛在空間的高斯分佈進行采樣。
3. 測試生成解碼輸出以及編碼器生成的均值和方差。

perplexity
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        """
        初始化變分自編碼器
        
        參數:
        input_dim: 輸入維度 (例如 MNIST: 28*28=784)
        hidden_dim: 隱藏層維度
        latent_dim: 潛在空間維度
        """
        super().__init__()
        
        # 編碼器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 均值和方差預測層
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # 解碼器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        """編碼過程：輸入 -> 潛在空間參數"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """重參數化技巧"""
        std = torch.exp(0.5 * log_var)
        # 使用 torch.distributions 創建正態分佈
        eps = Normal(0, 1).sample(mu.shape).to(mu.device)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """解碼過程：潛在變量 -> 重建"""
        return self.decoder(z)
    
    def forward(self, x):
        """前向傳播"""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

def loss_function(recon_x, x, mu, log_var):
    """
    VAE 損失函數：重建損失 + KL散度
    """
    # 重建損失（二元交叉熵）
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL 散度
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return BCE + KLD

def train_vae(model, train_loader, num_epochs=50, learning_rate=1e-3):
    """訓練 VAE"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.view(data.size(0), -1).to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, log_var = model(data)
            loss = loss_function(recon_batch, data, mu, log_var)
            
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            
        avg_loss = total_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

def visualize_results(model, test_loader, num_images=10):
    """視覺化重建結果和生成樣本"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    with torch.no_grad():
        # 獲取一批測試數據
        data, _ = next(iter(test_loader))
        data = data.view(data.size(0), -1).to(device)
        recon, _, _ = model(data)
        
        # 生成隨機樣本
        sample = torch.randn(num_images, model.fc_mu.out_features).to(device)
        generated = model.decode(sample)
        
        # 顯示原始圖像、重建圖像和生成圖像
        plt.figure(figsize=(15, 5))
        
        # 原始圖像
        for i in range(num_images):
            plt.subplot(3, num_images, i + 1)
            plt.imshow(data[i].cpu().view(28, 28), cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title('Original')
        
        # 重建圖像
        for i in range(num_images):
            plt.subplot(3, num_images, i + 1 + num_images)
            plt.imshow(recon[i].cpu().view(28, 28), cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title('Reconstructed')
        
        # 生成圖像
        for i in range(num_images):
            plt.subplot(3, num_images, i + 1 + 2*num_images)
            plt.imshow(generated[i].cpu().view(28, 28), cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title('Generated')
        
        plt.tight_layout()
        plt.show()

# 主程序
if __name__ == "__main__":
    # 設置數據轉換
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # 加載 MNIST 數據集
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # 創建模型
    input_dim = 28 * 28  # MNIST 圖像大小
    hidden_dim = 400     # 隱藏層維度
    latent_dim = 20      # 潛在空間維度
    model = VAE(input_dim, hidden_dim, latent_dim)
    
    # 訓練模型
    train_vae(model, train_loader)
    
    # 視覺化結果
    visualize_results(model, test_loader)

# 創建 VAE
vae = VAE(input_dim=784, hidden_dim=400, latent_dim=20)

# 準備數據
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST('./data', transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# 訓練模型
train_vae(vae, dataloader)

# 視覺化結果
visualize_results(vae, dataloader)



```


---

### **45. 實現一個時間序列預測模型**

#### 代碼

```python
class TimeSeriesRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TimeSeriesRNN, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # 取最後一個時間步的輸出
        return out

# 測試時間序列模型
input_dim, hidden_dim, output_dim = 1, 64, 1
model = TimeSeriesRNN(input_dim, hidden_dim, output_dim)
data = torch.randn(16, 10, 1)  # 批量大小 16，序列長度 10，每步 1 維
output = model(data)
print("Output Shape:", output.shape)

```
#### 中文解釋

1. 使用 LSTM 模型處理時間序列數據。
2. 輸入為三維張量（批量大小、序列長度、特徵數）。
3. 解碼最後一個時間步的輸出，進行預測。

### **46. 實現一個 LSTM 網絡來進行情感分析**

#### 代碼

```python
import torch
import torch.nn as nn

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, num_layers):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 使用最後一個時間步的輸出
        return out

# 測試模型
vocab_size, embed_size, hidden_size, output_size, num_layers = 5000, 128, 64, 2, 2
model = SentimentLSTM(vocab_size, embed_size, hidden_size, output_size, num_layers)
inputs = torch.randint(0, vocab_size, (8, 50))  # 批次大小 8，序列長度 50
outputs = model(inputs)
print("Output Shape:", outputs.shape)

```
#### 中文解釋

1. 定義嵌入層和 LSTM 層處理文本數據。
2. 使用全連接層將 LSTM 的最後輸出映射到情感類別。
3. 測試模型輸入為單詞 ID 序列，輸出為情感分類結果。

---

### **47. 實現一個 Seq2Seq 模型來進行機器翻譯**

#### 代碼

```python
class Seq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, embed_size, hidden_size):
        super(Seq2Seq, self).__init__()
        self.encoder = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.embedding = nn.Embedding(input_dim, embed_size)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, src, tgt):
        src_embedded = self.embedding(src)
        _, (hidden, cell) = self.encoder(src_embedded)
        tgt_embedded = self.embedding(tgt)
        decoder_output, _ = self.decoder(tgt_embedded, (hidden, cell))
        output = self.fc(decoder_output)
        return output

# 測試模型
input_dim, output_dim, embed_size, hidden_size = 3000, 3000, 128, 256
model = Seq2Seq(input_dim, output_dim, embed_size, hidden_size)
src = torch.randint(0, input_dim, (8, 10))  # 8 條源語句，每條長度 10
tgt = torch.randint(0, output_dim, (8, 10))  # 8 條目標語句，每條長度 10
outputs = model(src, tgt)
print("Output Shape:", outputs.shape)

```
#### 中文解釋

1. 定義編碼器和解碼器，均使用 LSTM。
2. 嵌入層將單詞 ID 映射為嵌入向量，解碼器輸出經全連接層映射為目標詞彙。
3. 測試輸入源語句和目標語句，輸出序列結果。

---

### **48. 實現一個批正規化（Batch Normalization）層**

#### 代碼

```python
class BatchNormExample(nn.Module):
    def __init__(self, num_features):
        super(BatchNormExample, self).__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x):
        return self.bn(x)

# 測試批正規化層
num_features = 16
model = BatchNormExample(num_features)
inputs = torch.randn(8, num_features)  # 批次大小 8，特徵數 16
outputs = model(inputs)
print("Output Shape:", outputs.shape)


```
#### 中文解釋

1. 使用 `BatchNorm1d` 對特徵進行批正規化。
2. 測試模型，輸入為批次數據，輸出為正規化後的數據。

---

### **49. 實現一個UNet神經網絡

#### 代碼
[[AI model Summary architecture#U-Net]]

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # 如果使用雙線性插值進行上採樣
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # 處理輸入大小不匹配的情況
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                       diff_y // 2, diff_y - diff_y // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """最後的輸出卷積層"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        # 初始化權重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

def create_unet(n_channels=3, n_classes=1, bilinear=True):
    """創建 U-Net 模型"""
    model = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)
    return model

# 測試代碼
if __name__ == "__main__":
    # 創建模型
    model = create_unet()
    
    # 測試輸入
    batch_size = 1
    channels = 3
    height = 572
    width = 572
    x = torch.randn(batch_size, channels, height, width)
    
    # 前向傳播
    output = model(x)
    
    # 打印輸出形狀
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # 計算模型參數量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

```
## 程式碼說明

1. **DoubleConv類別**:

- 實現兩個連續的卷積層
- 每個卷積後接BatchNorm和ReLU
- 使用padding保持特徵圖大小不變

2. **UNet架構**:

- **編碼器部分**:
    
    - 5個下採樣階段
    - 每階段包含兩個3x3卷積
    - 使用MaxPooling進行下採樣
    
- **解碼器部分**:
    
    - 4個上採樣階段
    - 使用反卷積進行上採樣
    - 特徵圖與編碼器對應層進行串接
    

3. **跳躍連接**:

- 將編碼器特徵與解碼器特徵串接
- 幫助保留細節信息
- 使用torch.cat進行特徵串接

這個UNet實現適用於多種圖像分割任務，例如:

- 醫學影像分割
- 衛星圖像分割
- 自動駕駛場景分割



---

### **50. 實現一個YOLO神經網絡

#### 代碼
[[AI model Summary architecture###### YOLOv7]]

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )
    
    def forward(self, x):
        return self.conv(x)

class YOLO(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        
        # 主幹網路
        self.layer1 = nn.Sequential(
            ConvBlock(3, 64, 7, 2),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            ConvBlock(64, 192, 3),
            nn.MaxPool2d(2, 2)
        )
        self.layer3 = nn.Sequential(
            ConvBlock(192, 128, 1),
            ConvBlock(128, 256, 3),
            ConvBlock(256, 256, 1),
            ConvBlock(256, 512, 3),
            nn.MaxPool2d(2, 2)
        )
        self.layer4 = nn.Sequential(
            ConvBlock(512, 256, 1),
            ConvBlock(256, 512, 3),
            ConvBlock(512, 256, 1),
            ConvBlock(256, 512, 3),
            ConvBlock(512, 256, 1),
            ConvBlock(256, 512, 3),
            nn.MaxPool2d(2, 2)
        )
        self.layer5 = nn.Sequential(
            ConvBlock(512, 512, 1),
            ConvBlock(512, 1024, 3),
            ConvBlock(1024, 512, 1),
            ConvBlock(512, 1024, 3),
        )
        
        # 檢測頭
        self.detection_head = nn.Sequential(
            ConvBlock(1024, 1024, 3),
            ConvBlock(1024, 1024, 3, stride=2),
            ConvBlock(1024, 1024, 3),
            ConvBlock(1024, 1024, 3),
            nn.Flatten(),
            nn.Linear(1024 * 7 * 7, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, 7 * 7 * (5 * 2 + num_classes))
        )
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.detection_head(x)
        
        batch_size = x.size(0)
        x = x.view(batch_size, 7, 7, -1)
        return x

```
## 程式碼說明

1. **ConvBlock類別**:

- 實現基本的卷積區塊
- 包含卷積層、批次正規化和LeakyReLU激活函數[1](https://www.datacamp.com/blog/yolo-object-detection-explained)
- 使用padding保持特徵圖大小

2. **YOLO架構**:

- 主幹網路包含24個卷積層和4個最大池化層[1](https://www.datacamp.com/blog/yolo-object-detection-explained)[3](https://viso.ai/computer-vision/yolo-explained/)
- 使用1x1卷積降低通道數[1](https://www.datacamp.com/blog/yolo-object-detection-explained)
- 最後使用全連接層進行預測[3](https://viso.ai/computer-vision/yolo-explained/)

3. **網路結構特點**:

- 輸入圖像會被調整為448x448大小[1](https://www.datacamp.com/blog/yolo-object-detection-explained)
- 將圖像分割為7x7的網格進行預測[6](https://encord.com/blog/yolo-object-detection-guide/)
- 每個網格預測多個邊界框和類別機率[6](https://encord.com/blog/yolo-object-detection-guide/)

4. **檢測頭設計**:

- 使用1024個通道的卷積層處理特徵
- 最後輸出包含邊界框座標、置信度和類別機率[4](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/)
- 使用Dropout防止過擬合[3](https://viso.ai/computer-vision/yolo-explained/)

perplexity
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNSiLU(nn.Module):
    """Convolution + BatchNorm + SiLU activation"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, groups=1):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ELAN(nn.Module):
    """Extended Linear Aggregation Node"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels // 2
        self.conv1 = ConvBNSiLU(in_channels, mid_channels, 1)
        self.conv2 = ConvBNSiLU(mid_channels, mid_channels, 3)
        self.conv3 = ConvBNSiLU(mid_channels, mid_channels, 3)
        self.conv4 = ConvBNSiLU(mid_channels, mid_channels, 3)
        self.conv5 = ConvBNSiLU(mid_channels * 4, out_channels, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        out = self.conv5(torch.cat([x1, x2, x3, x4], dim=1))
        return out

class ELAN_Block(nn.Module):
    """ELAN Block with downsampling"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = ConvBNSiLU(in_channels, out_channels, 3, stride=2)
        self.elan = ELAN(out_channels, out_channels)

    def forward(self, x):
        x = self.downsample(x)
        x = self.elan(x)
        return x

class SPPCSPBlock(nn.Module):
    """Spatial Pyramid Pooling CSP block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = in_channels // 2
        self.conv1 = ConvBNSiLU(in_channels, mid_channels, 1)
        self.conv2 = ConvBNSiLU(in_channels, mid_channels, 1)
        self.conv3 = ConvBNSiLU(mid_channels * 4, out_channels, 1)
        self.pools = nn.ModuleList([
            nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2)
            for k in [5, 9, 13]
        ])

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        pools = [x2]
        pools.extend([pool(x2) for pool in self.pools])
        x2 = torch.cat(pools, dim=1)
        return self.conv3(torch.cat([x1, x2], dim=1))

class YOLOv7(nn.Module):
    def __init__(self, num_classes=80, input_channels=3):
        super().__init__()
        
        # Backbone
        self.stem = nn.Sequential(
            ConvBNSiLU(input_channels, 32, 3),
            ConvBNSiLU(32, 64, 3, stride=2),
            ConvBNSiLU(64, 64, 3)
        )
        
        self.stage1 = ELAN_Block(64, 128)
        self.stage2 = ELAN_Block(128, 256)
        self.stage3 = ELAN_Block(256, 512)
        self.stage4 = ELAN_Block(512, 1024)
        
        self.spp = SPPCSPBlock(1024, 1024)
        
        # Head
        self.head = nn.ModuleList()
        for out_channels in [512, 256, 128]:
            self.head.append(
                nn.Sequential(
                    ConvBNSiLU(1024, out_channels, 1),
                    ConvBNSiLU(out_channels, out_channels * 2, 3),
                    ConvBNSiLU(out_channels * 2, out_channels, 1)
                )
            )
        
        # Detection layers
        self.det_layers = nn.ModuleList()
        for out_channels in [128, 256, 512]:
            self.det_layers.append(
                nn.Conv2d(out_channels, 3 * (5 + num_classes), 1)
            )

    def forward(self, x):
        # Backbone
        x = self.stem(x)
        x1 = self.stage1(x)      # 1/4
        x2 = self.stage2(x1)     # 1/8
        x3 = self.stage3(x2)     # 1/16
        x4 = self.stage4(x3)     # 1/32
        
        x4 = self.spp(x4)
        
        # Head
        outputs = []
        for i, head in enumerate(self.head):
            feat = head(x4 if i == 0 else outputs[-1])
            outputs.append(feat)
            
        # Detection
        results = []
        for feat, det_layer in zip(outputs, self.det_layers):
            results.append(det_layer(feat))
            
        if self.training:
            return results
        else:
            return self.postprocess(results)
            
    def postprocess(self, outputs):
        """後處理：將輸出轉換為邊界框"""
        batch_size = outputs[0].shape[0]
        predictions = []
        
        for output in outputs:
            # 重塑輸出為 [batch, anchors, grid_h, grid_w, xywh + obj + classes]
            batch, _, grid_h, grid_w = output.shape
            output = output.view(batch, 3, -1, grid_h, grid_w).permute(0, 1, 3, 4, 2)
            predictions.append(output)
            
        return predictions

def create_yolov7_model(num_classes=80, pretrained=False):
    model = YOLOv7(num_classes=num_classes)
    if pretrained:
        # 載入預訓練權重的邏輯
        pass
    return model

# 測試代碼
if __name__ == "__main__":
    model = create_yolov7_model()
    x = torch.randn(1, 3, 640, 640)
    outputs = model(x)
    
    # 打印每個檢測層的輸出大小
    if isinstance(outputs, list):
        for i, out in enumerate(outputs):
            print(f"Detection layer {i + 1} output shape:", out.shape)


# 創建模型
model = create_yolov7_model(num_classes=80)

# 訓練模式
model.train()
x = torch.randn(1, 3, 640, 640)
outputs = model(x)  # 返回原始檢測輸出

# 推理模式
model.eval()
with torch.no_grad():
    predictions = model(x)  # 返回處理後的預測結果
```
這個YOLO實現適用於多種目標檢測任務，例如:

- 即時物體檢測
- 自動駕駛場景檢測
- 安防監控系統

### **51. 實現一個MaskRCNN神經網絡**

#### 代碼
[[AI model Summary architecture###### Mask R-CNN]]

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BackboneNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # ResNet-like 骨幹網路
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.conv2 = self._make_layer(64, 256, 3)
        self.conv3 = self._make_layer(256, 512, 4)
        self.conv4 = self._make_layer(512, 1024, 6)
        self.conv5 = self._make_layer(1024, 2048, 3)
        
    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(blocks-1):
            layers.extend([
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
        return nn.Sequential(*layers)

class FPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.toplayer = nn.Conv2d(2048, 256, 1)
        self.lateral1 = nn.Conv2d(1024, 256, 1)
        self.lateral2 = nn.Conv2d(512, 256, 1)
        self.lateral3 = nn.Conv2d(256, 256, 1)
        
        self.smooth1 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, 3, padding=1)
        
    def forward(self, c2, c3, c4, c5):
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.lateral1(c4))
        p3 = self._upsample_add(p4, self.lateral2(c3))
        p2 = self._upsample_add(p3, self.lateral3(c2))
        
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        
        return p2, p3, p4, p5
        
    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.shape[2:], mode='bilinear', align_corners=False) + y

class RPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(256, 256, 3, padding=1)
        self.cls_logits = nn.Conv2d(256, 3 * 2, 1)  # 3個錨框 x 2類別
        self.bbox_pred = nn.Conv2d(256, 3 * 4, 1)   # 3個錨框 x 4座標
        
    def forward(self, x):
        x = F.relu(self.conv(x))
        return self.cls_logits(x), self.bbox_pred(x)

class MaskHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.deconv = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.mask_pred = nn.Conv2d(256, num_classes, 1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.deconv(x))
        return self.mask_pred(x)

class MaskRCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = BackboneNetwork()
        self.fpn = FPN()
        self.rpn = RPN()
        self.mask_head = MaskHead(num_classes)
        
    def forward(self, x):
        # 特徵提取
        c2, c3, c4, c5 = self.backbone(x)
        # FPN特徵融合
        p2, p3, p4, p5 = self.fpn(c2, c3, c4, c5)
        # RPN預測
        rpn_logits, rpn_bbox = self.rpn(p2)
        # 這裡需要實現RoI Pooling和proposal選擇
        # 最後進行遮罩預測
        masks = self.mask_head(p2)
        return rpn_logits, rpn_bbox, masks



```
## 程式碼說明

1. **BackboneNetwork類別**:

- 實現類似ResNet的骨幹網路
- 包含5個階段的特徵提取
- 使用批次正規化和ReLU激活函數

2. **FPN (特徵金字塔網路)類別**:

- 實現自頂向下的特徵融合
- 橫向連接不同尺度的特徵
- 生成多尺度特徵圖

3. **RPN (區域建議網路)類別**:

- 生成候選區域建議
- 預測物體/背景分類
- 預測邊界框回歸

4. **MaskHead類別**:

- 實現遮罩預測頭
- 使用多個卷積層提取特徵
- 最後輸出每個類別的遮罩預測

5. **完整MaskRCNN類別**:

- 整合所有組件
- 實現前向傳播流程
- 輸出RPN預測和遮罩預測

Perplexity
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign

class Backbone(nn.Module):
    def __init__(self, out_channels=256):
        super(Backbone, self).__init__()
        # 增強特徵提取能力
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 添加更多卷積層
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        features = self.layer2(x)
        return features

class RPN(nn.Module):
    def __init__(self, in_channels=256, anchor_num=9):
        super(RPN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.cls_layer = nn.Conv2d(256, anchor_num, kernel_size=1)
        self.reg_layer = nn.Conv2d(256, anchor_num * 4, kernel_size=1)
        
        # 初始化權重
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.constant_(layer.bias, 0)

    def forward(self, features):
        x = self.conv(features)
        objectness = self.cls_layer(x)
        bbox_reg = self.reg_layer(x)
        return objectness, bbox_reg

class RoIHeads(nn.Module):
    def __init__(self, num_classes=21, in_channels=256, roi_size=7):
        super(RoIHeads, self).__init__()
        self.roi_align = RoIAlign(
            output_size=(roi_size, roi_size),
            spatial_scale=1.0,
            sampling_ratio=2
        )
        
        roi_size = roi_size * roi_size * in_channels
        self.fc = nn.Sequential(
            nn.Linear(roi_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)
        
        # 初始化權重
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.constant_(layer.bias, 0)

    def forward(self, features, proposals, image_shapes):
        roi_features = self.roi_align(features, proposals, image_shapes)
        roi_features = roi_features.flatten(start_dim=1)
        fc_features = self.fc(roi_features)
        cls_scores = self.cls_score(fc_features)
        bbox_deltas = self.bbox_pred(fc_features)
        return cls_scores, bbox_deltas

class MaskBranch(nn.Module):
    def __init__(self, in_channels=256, num_classes=21):
        super(MaskBranch, self).__init__()
        self.mask_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        
        self.roi_align = RoIAlign(
            output_size=(14, 14),
            spatial_scale=1.0,
            sampling_ratio=2
        )

    def forward(self, features, proposals, image_shapes):
        roi_features = self.roi_align(features, proposals, image_shapes)
        masks = self.mask_head(roi_features)
        return masks

class MaskRCNN(nn.Module):
    def __init__(self, num_classes=21, min_size=800, max_size=1333):
        super(MaskRCNN, self).__init__()
        self.backbone = Backbone(out_channels=256)
        self.rpn = RPN(in_channels=256, anchor_num=9)
        self.roi_heads = RoIHeads(num_classes=num_classes, in_channels=256)
        self.mask_branch = MaskBranch(in_channels=256, num_classes=num_classes)
        self.min_size = min_size
        self.max_size = max_size

    def transform_image(self, images):
        # 圖像預處理
        original_sizes = [(img.shape[-2], img.shape[-1]) for img in images]
        
        # 標準化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        images = [(img / 255.0 - mean) / std for img in images]
        
        return images, original_sizes

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("在訓練模式下必須提供targets")
            
        images, original_sizes = self.transform_image(images)
        
        # 特徵提取
        features = self.backbone(images)
        
        # RPN 處理
        objectness, bbox_reg = self.rpn(features)
        
        # 生成proposals (簡化版本)
        proposals = torch.rand((len(images), 100, 4))
        image_shapes = [(images.size(2), images.size(3))] * len(images)
        
        # RoI 處理
        cls_scores, bbox_deltas = self.roi_heads(features, proposals, image_shapes)
        
        # Mask 預測
        masks = self.mask_branch(features, proposals, image_shapes)
        
        result = {
            "cls_scores": cls_scores,
            "bbox_deltas": bbox_deltas,
            "masks": masks,
            "proposals": proposals
        }
        
        if self.training:
            losses = self.compute_losses(result, targets)
            result.update(losses)
            
        return result
```

這個實現包含了Mask R-CNN的核心組件，但注意以下幾點仍需補充:

- RoI Pooling/Align層的實現
- Proposal生成和選擇邏輯
- 訓練時的損失函數計算
- 推論時的後處理邏輯

---

### **52. 實現一個Segment Anything神經網絡**

#### 代碼
[[AI model Summary architecture###### SAM (Segment Anything Model)]]

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BackboneNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2 = self._make_layer(64, 128, 3)
        self.conv3 = self._make_layer(128, 256, 4)
        self.conv4 = self._make_layer(256, 512, 6)

    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        for _ in range(blocks):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        return c1, c2, c3, c4

# 特徵金字塔網路實現:
class FeaturePyramidNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.toplayer = nn.Conv2d(512, 256, kernel_size=1)
        self.lateral3 = nn.Conv2d(256, 256, kernel_size=1)
        self.lateral2 = nn.Conv2d(128, 256, kernel_size=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

    def forward(self, c2, c3, c4):
        p4 = self.toplayer(c4)
        p3 = self._upsample_add(p4, self.lateral3(c3))
        p2 = self._upsample_add(p3, self.lateral2(c2))
        p3 = self.smooth3(p3)
        p2 = self.smooth2(p2)
        return p2, p3, p4

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.shape[2:], mode='bilinear', align_corners=False) + y

# 遮罩預測頭實現:
class MaskHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.mask_pred = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.deconv(x))
        return self.mask_pred(x)

```
## 程式碼說明

1. **BackboneNetwork類別**:

- 使用類似ResNet的架構提取特徵
- 包含四個階段的特徵提取
- 每個階段使用多個卷積層和批次正規化

2. **FeaturePyramidNetwork類別**:

- 實現特徵金字塔結構
- 自頂向下的特徵融合
- 使用1x1卷積進行通道調整
- 實現特徵圖的上採樣和加法融合

3. **MaskHead類別**:

- 實現遮罩預測頭
- 使用多個3x3卷積層提取特徵
- 使用反卷積層進行上採樣
- 最後輸出每個像素的類別預測

perplexity
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class MLPBlock(nn.Module):
    def __init__(self, embedding_dim: int, mlp_dim: int, act=nn.GELU):
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embedding_dim, embedding_dim * 3)
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attn = AttentionLayer(embedding_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLPBlock(embedding_dim, mlp_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_channels: int = 3,
        embedding_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2, embedding_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_dim=int(embedding_dim * mlp_ratio),
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        self.neck = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1),
            LayerNorm2d(embedding_dim),
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, padding=1),
        )

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        x = self.neck(x)
        return x

class PromptEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 256,
        point_embedding_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.point_embed = nn.Sequential(
            nn.Linear(2, point_embedding_dim),
            nn.GELU(),
            nn.Linear(point_embedding_dim, embedding_dim),
        )
        
        self.box_embed = nn.Sequential(
            nn.Linear(4, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        
        self.not_a_point_embed = nn.Parameter(torch.randn(1, embedding_dim))

    def forward(
        self,
        points: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        point_embeddings = self.not_a_point_embed.expand(points.shape[0], -1) if points is None \
                         else self.point_embed(points)
        
        if boxes is not None:
            box_embeddings = self.box_embed(boxes)
            return torch.cat([point_embeddings, box_embeddings], dim=1)
        
        return point_embeddings

class MaskDecoder(nn.Module):
    def __init__(
        self,
        transformer_dim: int = 256,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        
        self.transformer_dim = transformer_dim
        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Parameter(torch.zeros(1, 1, transformer_dim))
        self.mask_tokens = nn.Parameter(torch.zeros(1, num_multimask_outputs, transformer_dim))
        
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        
        self.output_hypernetworks_mlps = nn.ModuleList([
            MLPBlock(transformer_dim, transformer_dim, activation)
            for i in range(num_multimask_outputs)
        ])

        self.iou_prediction_head = MLPBlock(transformer_dim, 1, activation)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Upscale image embeddings
        image_embeddings = self.output_upscaling(image_embeddings)
        
        # Generate masks
        masks = []
        iou_pred = self.iou_prediction_head(prompt_embeddings)
        
        for i in range(self.num_multimask_outputs):
            mask_embedding = self.mask_tokens[:, i:i+1]
            hyper_in = prompt_embeddings + mask_embedding
            hyper_in = self.output_hypernetworks_mlps[i](hyper_in)
            mask = (hyper_in @ image_embeddings.flatten(2)).reshape(
                image_embeddings.shape[0], 1, *image_embeddings.shape[-2:]
            )
            masks.append(mask)
        
        masks = torch.cat(masks, dim=1)
        return masks, iou_pred

class SAM(nn.Module):
    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder

    def forward(
        self,
        images: torch.Tensor,
        points: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image_embeddings = self.image_encoder(images)
        prompt_embeddings = self.prompt_encoder(points, boxes)
        masks, iou_predictions = self.mask_decoder(image_embeddings, prompt_embeddings)
        return masks, iou_predictions

def create_sam_model(model_type: str = "base"):
    """Create a SAM model with specified configuration"""
    configs = {
        "base": dict(
            embedding_dim=768,
            num_heads=12,
            depth=12,
        ),
        "large": dict(
            embedding_dim=1024,
            num_heads=16,
            depth=24,
        ),
    }
    
    if model_type not in configs:
        raise ValueError(f"Model type {model_type} not found. Available types: {list(configs.keys())}")
    
    config = configs[model_type]
    
    image_encoder = ImageEncoderViT(
        embedding_dim=config["embedding_dim"],
        num_heads=config["num_heads"],
        depth=config["depth"],
    )
    
    prompt_encoder = PromptEncoder(
        embedding_dim=config["embedding_dim"] // 3,
    )
    
    mask_decoder = MaskDecoder(
        transformer_dim=config["embedding_dim"] // 3,
    )
    
    return SAM(image_encoder, prompt_encoder, mask_decoder)

# 測試代碼
if __name__ == "__main__":
    # 創建模型
    model = create_sam_model("base")
    
    # 測試輸入
    batch_size = 1
    images = torch.randn(batch_size, 3, 1024, 1024)
    points = torch.randn(batch_size, 2)
    boxes = torch.randn(batch_size, 4)
    
    # 前向傳播
    masks, iou_predictions = model(images, points, boxes)
    
    # 打印輸出形狀
    print(f"Input image shape: {images.shape}")
    print(f"Output masks shape: {masks.shape}")
    print(f"IoU predictions shape: {iou_predictions.shape}")
    
    # 計算參數量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")


# 創建模型
model = create_sam_model("base")

# 準備輸入
images = torch.randn(1, 3, 1024, 1024)
points = torch.randn(1, 2)  # 可選
boxes = torch.randn(1, 4)   # 可選

# 生成遮罩
masks, iou_predictions = model(images, points, boxes)

```

這個實現提供了Segment Anything的基本架構，適用於通用的圖像分割任務。

---

### **53. 編寫代碼實現訓練一個簡單的線性回歸模型**

#### 代碼

```python
# 定義數據
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# 定義模型
model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 訓練循環
for epoch in range(100):
    outputs = model(x)
    loss = criterion(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

```
#### 中文解釋

1. 使用線性層模擬線性關係。
2. 定義均方誤差損失函數，使用 SGD 優化器。
3. 訓練模型並輸出每 10 個 epoch 的損失值。

---

### **54. 實現一個訓練循環，包含驗證和早停功能**

#### 代碼

```python
best_loss = float('inf')
patience, patience_counter = 5, 0

for epoch in range(50):
    # 訓練階段
    model.train()
    train_loss = 0
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # 驗證階段
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # 早停
    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

```
#### 中文解釋

1. 訓練階段使用 `.train()`，更新模型參數。
2. 驗證階段使用 `.eval()`，不更新參數。
3. 早停功能檢查驗證損失，若連續多次無改善則停止訓練。

---

### **55. 使用 `torch.optim.SGD` 訓練模型，並手動調整學習率**

#### 代碼

```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

for epoch in range(50):
    model.train()
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    scheduler.step()
    print(f"Epoch {epoch+1}, Learning Rate: {scheduler.get_last_lr()[0]:.5f}")

```
#### 中文解釋

1. 使用 SGD 優化器進行模型訓練。
2. 使用學習率調度器 `StepLR`，每隔一定步數減小學習率。
3. 打印每個 epoch 的當前學習率。

### **56. 在訓練過程中記錄損失值，並使用 Matplotlib 畫出曲線**

#### 代碼

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 模擬數據
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# 定義模型
model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 訓練循環
losses = []
for epoch in range(100):
    outputs = model(x)
    loss = criterion(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# 繪製損失曲線
plt.plot(range(1, 101), losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()

```
#### 中文解釋

1. 在每個 epoch 訓練後記錄損失值。
2. 使用 Matplotlib 繪製損失值的變化曲線。
3. 將損失值作為 y 軸，epoch 作為 x 軸進行可視化。

---

### **57. 編寫代碼動態調整學習率，例如使用學習率調度器（LR Scheduler）**

#### 代碼

```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

for epoch in range(30):
    # 模擬訓練步驟
    outputs = model(x)
    loss = criterion(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 更新學習率
    scheduler.step()
    print(f"Epoch {epoch+1}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

```
#### 中文解釋

1. 使用 `StepLR` 調度器，設置每 10 個 epoch 將學習率減小為原來的 0.1 倍。
2. 在每個 epoch 結束後調用 `scheduler.step()` 更新學習率。
3. 使用 `get_last_lr` 查看當前學習率。

---

### **58. 使用混合精度訓練來加速模型訓練**

#### 代碼

```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

for epoch in range(50):
    optimizer.zero_grad()
    with autocast():
        outputs = model(x)
        loss = criterion(outputs, y)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

```
#### 中文解釋

1. 使用 `GradScaler` 自動處理梯度的縮放，避免混合精度下的數值不穩定。
2. 使用 `autocast` 在前向傳播中自動啟用混合精度計算。
3. `scaler.scale` 用於縮放損失，`scaler.step` 和 `scaler.update` 分別用於優化器步驟和更新縮放。

---

### **59. 編寫代碼實現模型的 K 折交叉驗證**

#### 代碼

```python
from sklearn.model_selection import KFold

data = torch.randn(100, 1)
targets = 2 * data + 3 + 0.1 * torch.randn(100, 1)  # 模擬目標
kf = KFold(n_splits=5)

for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
    train_data, val_data = data[train_idx], data[val_idx]
    train_targets, val_targets = targets[train_idx], targets[val_idx]

    # 初始化模型
    model = nn.Linear(1, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # 訓練
    for epoch in range(20):
        model.train()
        outputs = model(train_data)
        loss = criterion(outputs, train_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 驗證
    model.eval()
    with torch.no_grad():
        val_outputs = model(val_data)
        val_loss = criterion(val_outputs, val_targets)
    print(f"Fold {fold+1}, Validation Loss: {val_loss.item():.4f}")

```
#### 中文解釋

1. 使用 `KFold` 將數據集分為 5 折。
2. 在每一折中分別進行訓練和驗證。
3. 驗證階段使用 `model.eval()`，不計算梯度。

---

### **60. 使用 `DistributedDataParallel` 加速模型訓練**

#### 代碼

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化分布式環境
dist.init_process_group("nccl", init_method="env://")

# 設定 GPU
device = torch.device(f"cuda:{dist.get_rank()}")
torch.cuda.set_device(device)

# 定義模型
model = nn.Linear(10, 1).to(device)
model = DDP(model, device_ids=[dist.get_rank()])

# 定義優化器和數據
optimizer = optim.SGD(model.parameters(), lr=0.01)
inputs = torch.randn(16, 10).to(device)
labels = torch.randn(16, 1).to(device)

# 訓練步驟
for epoch in range(10):
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Rank {dist.get_rank()}, Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 清理分布式環境
dist.destroy_process_group()

```
#### 中文解釋

1. 使用 `DistributedDataParallel` 將模型分布到多個 GPU。
2. 使用 `dist.init_process_group` 初始化分布式環境。
3. 每個進程負責一部分數據並進行訓練，結果會同步更新到所有 GPU。
4. 訓練結束後調用 `dist.destroy_process_group` 清理資源。

### **61. 編寫代碼實現自定義的訓練日誌記錄系統**

#### 代碼

```python
import logging

# 配置日誌系統
logging.basicConfig(
    filename="training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# 訓練過程中記錄損失
for epoch in range(5):
    train_loss = 0.01 * (5 - epoch)  # 模擬損失值
    val_loss = 0.01 * (5 - epoch) + 0.002  # 模擬驗證損失
    logging.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

```
#### 中文解釋

1. 使用 `logging` 設置日誌格式，指定輸出文件和日期格式。
2. 在訓練中記錄每個 epoch 的訓練損失和驗證損失。
3. 所有日誌記錄存儲在 `training.log` 文件中。

---

### **62. 實現一個完整的模型訓練與測試腳本，支持命令行參數**

#### 代碼

```python
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

# 定義模型
class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)

# 設置命令行參數
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
args = parser.parse_args()

# 模擬數據
x = torch.randn(100, 1)
y = 2 * x + 3 + 0.1 * torch.randn(100, 1)

# 初始化模型、損失函數和優化器
model = SimpleModel(1, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr)

# 訓練循環
for epoch in range(args.epochs):
    outputs = model(x)
    loss = criterion(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss.item():.4f}")

```
#### 中文解釋

1. 使用 `argparse` 設置命令行參數，用於設置 epoch 和學習率。
2. 定義簡單的模型，使用模擬數據進行訓練。
3. 打印每個 epoch 的損失值，支持通過命令行靈活控制超參數。

---

### **63. 使用 GPU 來加速模型的訓練**

#### 代碼

```python
# 檢查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型和數據到 GPU
model = SimpleModel(1, 1).to(device)
x = torch.randn(100, 1).to(device)
y = 2 * x + 3 + 0.1 * torch.randn(100, 1).to(device)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 訓練
for epoch in range(10):
    outputs = model(x)
    loss = criterion(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

```
#### 中文解釋

1. 使用 `torch.device` 檢查是否可用 GPU，並選擇執行設備。
2. 將模型和數據移動到 GPU。
3. 在 GPU 上訓練模型，提高計算速度。

---

### **64. 在訓練過程中，如何保存和加載模型的權重？**

#### 代碼

```python
# 保存模型
torch.save(model.state_dict(), "model_weights.pth")
print("Model weights saved.")

# 加載模型
loaded_model = SimpleModel(1, 1)
loaded_model.load_state_dict(torch.load("model_weights.pth"))
loaded_model.eval()
print("Model weights loaded.")

```
#### 中文解釋

1. 使用 `torch.save` 保存模型的權重到文件。
2. 使用 `torch.load` 加載模型權重，並使用 `load_state_dict` 應用到模型。
3. 將模型設置為 `eval` 模式，適用於測試。

---

### **65. 如何在 PyTorch 中進行模型的早停（Early Stopping）？**

#### 代碼

```python
# 定義早停參數
patience = 3
best_loss = float('inf')
patience_counter = 0

# 訓練循環
for epoch in range(20):
    # 模擬訓練損失和驗證損失
    train_loss = 0.05 * (20 - epoch) + 0.01  # 模擬訓練損失
    val_loss = 0.05 * (20 - epoch)  # 模擬驗證損失

    # 檢查早停條件
    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1

    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    if patience_counter >= patience:
        print("Early stopping triggered!")
        break

```
#### 中文解釋

1. 設置 `patience`，允許驗證損失連續無改善的最大次數。
2. 在每個 epoch 結束後檢查驗證損失是否有改善。
3. 當連續 `patience` 次驗證損失無改善時，觸發早停。

### **66. 如何在多 GPU 上進行模型的分佈式訓練？**

#### 代碼

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel

# 定義模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleModel().to(device)

# 包裹模型以支持多 GPU
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = DataParallel(model)

# 模擬數據
inputs = torch.randn(64, 10).to(device)
labels = torch.randn(64, 1).to(device)

# 設置損失函數和優化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 訓練步驟
for epoch in range(5):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

```
#### 中文解釋

1. 使用 `torch.nn.parallel.DataParallel` 包裹模型，實現多 GPU 支持。
2. 將數據和模型移動到 GPU。
3. 在多 GPU 上進行前向和反向傳播，計算損失和梯度。

---

### **67. 如何在 PyTorch 中進行模型的超參數調優？**

#### 代碼

```python
from sklearn.model_selection import ParameterGrid

# 定義超參數搜索空間
param_grid = {
    'lr': [0.001, 0.01],
    'batch_size': [16, 32]
}

# 遍歷所有超參數組合
for params in ParameterGrid(param_grid):
    print(f"Training with params: {params}")
    optimizer = optim.SGD(model.parameters(), lr=params['lr'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=params['batch_size'])
    for epoch in range(5):
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

```
#### 中文解釋

1. 使用 `ParameterGrid` 遍歷超參數組合。
2. 每次組合初始化模型、優化器和數據加載器。
3. 訓練模型，並根據超參數比較性能。

---

### **68. 如何在 PyTorch 中實現梯度截斷（Gradient Clipping）？**

#### 代碼
```python
# 設置梯度截斷
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 訓練步驟
for epoch in range(5):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()

    # 截斷梯度
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

```
#### 中文解釋

1. 使用 `torch.nn.utils.clip_grad_norm_` 截斷梯度以防止梯度爆炸。
2. 在每次反向傳播後應用梯度截斷。
3. 確保所有參數的梯度範數不超過設置的 `max_norm`。

---

### **69. 使用 `torch.optim` 模組來實現自適應學習率調整**

#### 代碼
```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

for epoch in range(10):
    # 模擬訓練
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # 更新學習率
    scheduler.step(loss.item())
    print(f"Epoch {epoch+1}, Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

```
#### 中文解釋

1. 使用 `ReduceLROnPlateau` 根據損失變化自動調整學習率。
2. 當驗證損失在多個 epoch 中未改善時，減小學習率。
3. 打印當前學習率，觀察其動態變化。

---

### **70. 使用 `torch.autograd` 來計算張量的梯度**

#### 代碼
```python
# 創建一個需要梯度計算的張量
x = torch.tensor([2.0, 3.0], requires_grad=True)

# 定義一個函數
y = x[0] ** 2 + x[1] ** 3

# 計算梯度
y.backward()

# 查看梯度
print("Gradients:", x.grad)

```
#### 中文解釋

1. 設置張量的 `requires_grad=True`，啟用自動梯度計算。
2. 定義一個函數，對輸入張量執行操作。
3. 使用 `backward` 計算梯度，結果存儲在 `x.grad` 中。

### **71. 如何在 PyTorch 中進行模型的混合精度訓練（Mixed Precision Training）？**

#### 代碼

```python
from torch.cuda.amp import GradScaler, autocast

# 初始化混合精度工具
scaler = GradScaler()

# 訓練循環
for epoch in range(5):
    optimizer.zero_grad()
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, labels)

    # 使用混合精度計算梯度
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

```
#### 中文解釋

1. 使用 `GradScaler` 進行損失縮放，避免數值不穩定。
2. 使用 `autocast` 在前向傳播中啟用混合精度計算，減少內存使用。
3. 使用 `scaler.scale(loss)` 和 `scaler.step(optimizer)` 進行梯度縮放和優化步驟。

---

### **72. 將一個訓練好的 PyTorch 模型轉換為 ONNX 格式**

#### 代碼

```python
import torch

# 模擬輸入數據
dummy_input = torch.randn(1, 3, 224, 224)

# 保存模型為 ONNX 格式
torch.onnx.export(
    model, 
    dummy_input, 
    "model.onnx", 
    input_names=["input"], 
    output_names=["output"], 
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)
print("Model exported to ONNX format.")

```

#### 中文解釋

1. 使用 `torch.onnx.export` 將模型導出為 ONNX 格式。
2. 指定輸入數據和輸出名稱，支持動態 batch_size。
3. ONNX 格式文件保存為 `model.onnx`。

---

### **73. 編寫代碼使用 `torch.jit.trace` 將模型轉換為 TorchScript**

#### 代碼

```python
import torch.jit

# 模擬輸入數據
dummy_input = torch.randn(1, 3, 224, 224)

# 使用 `torch.jit.trace` 轉換模型
traced_model = torch.jit.trace(model, dummy_input)

# 保存 TorchScript 模型
traced_model.save("traced_model.pt")
print("Model saved as TorchScript format.")

```
#### 中文解釋

1. 使用 `torch.jit.trace` 將模型轉換為 TorchScript，適用於前向傳播固定的模型。
2. 使用 `.save` 將 TorchScript 模型保存到文件。
3. 測試輸入數據與模型結構一致。

---

### **74. 使用 `onnxruntime` 加載 ONNX 模型進行推理**

#### 代碼

```python
import onnxruntime as ort
import numpy as np

# 加載 ONNX 模型
session = ort.InferenceSession("model.onnx")

# 構造輸入數據
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

# 推理
outputs = session.run(None, {"input": input_data})
print("Inference Output:", outputs[0])

```
#### 中文解釋

1. 使用 `onnxruntime.InferenceSession` 加載 ONNX 模型。
2. 構造與模型輸入匹配的 NumPy 數據。
3. 使用 `session.run` 進行推理，返回模型輸出的結果。

---

### **75. 將 PyTorch 模型量化以減少模型大小**

#### 代碼

```python
import torch.quantization

# 模型準備量化
model.qconfig = torch.quantization.get_default_qconfig("fbgemm")

# 假量化準備
model = torch.quantization.prepare(model)

# 模擬校準
for _ in range(100):
    model(inputs)

# 量化模型
quantized_model = torch.quantization.convert(model)
print("Quantized model:", quantized_model)

```
#### 中文解釋

1. 設置模型的量化配置，使用 `get_default_qconfig` 選擇合適的後端。
2. 通過 `prepare` 函數執行假量化，收集量化所需的範圍信息。
3. 使用 `convert` 函數將模型轉換為量化版本，減少內存和加速推理。

### **76. 實現一個多線程的推理服務，處理多個請求**

#### 代碼

```python
import torch
from torch.nn import Linear
from concurrent.futures import ThreadPoolExecutor

# 定義簡單模型
model = Linear(10, 1)
model.eval()

# 模擬請求處理函數
def process_request(data):
    with torch.no_grad():
        input_tensor = torch.tensor(data, dtype=torch.float32)
        output = model(input_tensor)
    return output.numpy()

# 多線程推理服務
def inference_service(requests, max_workers=4):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_request, requests))
    return results

# 測試推理服務
requests = [torch.randn(1, 10).tolist() for _ in range(10)]
responses = inference_service(requests)
print("Inference Results:", responses)

```
#### 中文解釋

1. 使用 `ThreadPoolExecutor` 管理多線程，每個線程處理一個請求。
2. 定義 `process_request` 函數進行模型推理。
3. 測試多個請求並返回推理結果。

---

### **77. 測試模型在 GPU 和 CPU 上的推理速度，並比較結果**

#### 代碼
```python
import time

# 測試推理時間的函數
def measure_inference_time(device, inputs):
    model.to(device)
    inputs = inputs.to(device)
    with torch.no_grad():
        start_time = time.time()
        for _ in range(100):  # 重複推理 100 次
            model(inputs)
        end_time = time.time()
    return (end_time - start_time) / 100  # 平均推理時間

# 測試模型
inputs = torch.randn(32, 10)
cpu_time = measure_inference_time("cpu", inputs)
gpu_time = measure_inference_time("cuda", inputs)

print(f"CPU Time: {cpu_time:.6f} seconds")
print(f"GPU Time: {gpu_time:.6f} seconds")

```
#### 中文解釋

1. 測量模型在 CPU 和 GPU 上推理的平均時間。
2. 使用 `time` 計算推理耗時，重複推理 100 次提高精度。
3. 比較 GPU 和 CPU 的推理時間。

---

### **78. 使用 TorchServe 部署模型並提供 RESTful API**

#### 代碼

1. **保存模型為 TorchServe 格式**
    
```python
import torch

model = torch.nn.Linear(10, 1)
example_input = torch.randn(1, 10)
torch.jit.save(torch.jit.trace(model, example_input), "model.pt")
print("Model saved as TorchScript format.")

```
    
2. **創建 `handler.py`**
    
```python
from ts.torch_handler.base_handler import BaseHandler
import torch

class MyModelHandler(BaseHandler):
    def initialize(self, context):
        self.model = torch.jit.load("model.pt")
        self.model.eval()

    def handle(self, data, context):
        inputs = torch.tensor(data[0]["data"])
        outputs = self.model(inputs)
        return outputs.tolist()

```
    
3. **部署模型**
    
```python
torch-model-archiver --model-name my_model --version 1.0 --serialized-file model.pt --handler handler.py --export-path ./model-store
torchserve --start --model-store ./model-store --models my_model=my_model.mar

```
    
4. **測試 RESTful API**
    
```python
curl -X POST http://127.0.0.1:8080/predictions/my_model -T input.json

```

#### 中文解釋

1. 將模型保存為 TorchScript 格式。
2. 定義自定義處理器 `handler.py`，實現推理邏輯。
3. 使用 TorchServe 部署模型並測試 RESTful API。

---

### **79. 優化模型的內存占用，處理大批量推理**

#### 代碼

```python
# 分批次處理推理
def batch_inference(model, data, batch_size):
    model.eval()
    outputs = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = torch.tensor(data[i:i+batch_size], dtype=torch.float32)
            output = model(batch)
            outputs.append(output.numpy())
    return outputs

# 測試批量推理
data = [torch.randn(10).tolist() for _ in range(1000)]
results = batch_inference(model, data, batch_size=32)
print("Batch inference completed.")

```
#### 中文解釋

1. 將數據分批次處理，降低內存占用。
2. 使用 `with torch.no_grad()` 減少不必要的梯度計算。
3. 測試批量推理並返回結果。

---

### **80. 使用 `torch.profiler` 分析模型的性能瓶頸**

#### 代碼

```python
import torch.profiler

# 模擬數據
inputs = torch.randn(32, 10)

# 性能分析
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA
    ],
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for _ in range(10):
        outputs = model(inputs)

# 打印性能報告
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

```
#### 中文解釋

1. 使用 `torch.profiler` 分析模型的性能，包括 CPU 和 CUDA 活動。
2. 使用 `tensorboard_trace_handler` 將結果保存為 TensorBoard 格式進行可視化。
3. 打印分析報告，定位性能瓶頸。

### **81. 編寫代碼測量模型的延遲和吞吐量**

#### 代碼

```python
import torch
import time

# 測量延遲和吞吐量的函數
def measure_performance(model, inputs, num_runs=100):
    model.eval()
    with torch.no_grad():
        # 測量延遲
        start_time = time.time()
        for _ in range(num_runs):
            _ = model(inputs)
        end_time = time.time()
        latency = (end_time - start_time) / num_runs

        # 測量吞吐量
        throughput = num_runs / (end_time - start_time)

    return latency, throughput

# 測試模型性能
model = torch.nn.Linear(10, 1)
inputs = torch.randn(32, 10)
latency, throughput = measure_performance(model, inputs)
print(f"Latency: {latency:.6f} seconds")
print(f"Throughput: {throughput:.2f} samples/second")

```
#### 中文解釋

1. 測量延遲：多次前向傳播求平均時間。
2. 測量吞吐量：每秒處理的數據樣本數。
3. 測試模型的性能，輸入模擬數據。

---

### **82. 如何在 PyTorch 中進行模型的剪枝（Pruning）**

#### 代碼

```python
import torch.nn.utils.prune as prune

# 定義模型
model = torch.nn.Linear(10, 1)

# 應用剪枝
prune.l1_unstructured(model, name="weight", amount=0.5)

# 查看剪枝結果
print("Pruned weights:", model.weight)
print("Mask applied to weights:", model.weight_mask)

# 移除剪枝
prune.remove(model, "weight")
print("Weights after pruning removed:", model.weight)

```
#### 中文解釋

1. 使用 `l1_unstructured` 剪枝，將權重中較小的 50% 值置為零。
2. 剪枝後生成一個掩碼 (`weight_mask`)。
3. 使用 `prune.remove` 從模型中移除剪枝的掩碼。

---

### **83. 如何在 PyTorch 中進行模型的量化（Quantization）**

#### 代碼

```python
import torch.quantization

# 配置模型進行量化
model.qconfig = torch.quantization.get_default_qconfig("fbgemm")

# 假量化準備
model = torch.quantization.prepare(model)

# 模擬校準
inputs = torch.randn(32, 10)
with torch.no_grad():
    for _ in range(100):
        model(inputs)

# 轉換為量化模型
quantized_model = torch.quantization.convert(model)
print("Quantized model:", quantized_model)

```
#### 中文解釋

1. 設置量化配置 (`qconfig`)。
2. 使用 `prepare` 函數執行假量化，收集數據範圍。
3. 使用 `convert` 將模型轉換為量化模型，減少內存占用和推理時間。

---

### **84. 如何在 PyTorch 中進行模型的單元測試**

#### 代碼

```python
import unittest
import torch

# 定義測試類
class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Linear(10, 1)
        self.inputs = torch.randn(8, 10)
        self.labels = torch.randn(8, 1)

    def test_forward(self):
        outputs = self.model(self.inputs)
        self.assertEqual(outputs.size(), (8, 1))

    def test_loss(self):
        criterion = torch.nn.MSELoss()
        outputs = self.model(self.inputs)
        loss = criterion(outputs, self.labels)
        self.assertGreater(loss.item(), 0)

# 運行測試
if __name__ == "__main__":
    unittest.main()

```
#### 中文解釋

1. 使用 `unittest` 編寫模型的單元測試。
2. 測試前向傳播的輸出形狀。
3. 測試損失值是否合理（大於零）。

---

### **85. 如何在 PyTorch 中實現模型的可視化**

#### 代碼

```python
from torchviz import make_dot

# 定義模型和輸入
model = torch.nn.Linear(10, 1)
inputs = torch.randn(1, 10)

# 前向傳播並生成計算圖
outputs = model(inputs)
dot = make_dot(outputs, params=dict(model.named_parameters()))

# 保存計算圖
dot.render("model_visualization", format="png")
print("Model visualization saved as PNG.")

```
#### 中文解釋

1. 使用 `torchviz` 的 `make_dot` 可視化模型計算圖。
2. 使用 `params` 提供模型的參數。
3. 將計算圖保存為 PNG 文件，方便檢查模型結構。

### **86. 使用 `torch.jit` 來加速模型的推理**

#### 代碼

```python
import torch

# 定義模型
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# 原始模型
model = SimpleModel()

# 使用 TorchScript 加速推理
example_input = torch.randn(1, 10)
scripted_model = torch.jit.trace(model, example_input)

# 測試推理
inputs = torch.randn(8, 10)
outputs = scripted_model(inputs)
print("Scripted Model Output:", outputs)

```
#### 中文解釋

1. 使用 `torch.jit.trace` 將模型轉換為 TorchScript 模型。
2. TorchScript 模型可以加速推理，並支持導出到 C++。
3. 測試輸入數據，確保輸出與原模型一致。

---

### **87. 如何在 PyTorch 中進行模型的版本控制**

#### 代碼

```python
import os
import torch

# 保存不同版本的模型
model = torch.nn.Linear(10, 1)
for version in range(1, 4):
    torch.save(model.state_dict(), f"model_v{version}.pth")
    print(f"Model version {version} saved.")

# 加載指定版本的模型
version_to_load = 2
loaded_model = torch.nn.Linear(10, 1)
loaded_model.load_state_dict(torch.load(f"model_v{version_to_load}.pth"))
print(f"Loaded model version {version_to_load}.")

```
#### 中文解釋

1. 通過命名約定（如 `model_v{版本號}.pth`）實現模型的版本控制。
2. 使用 `torch.save` 和 `torch.load` 保存和加載指定版本的模型權重。
3. 測試加載指定版本的模型，檢查其狀態。

---

### **88. 使用 `torch.multiprocessing` 來加速數據加載**

#### 代碼

```python
import torch
from torch.utils.data import DataLoader, Dataset

# 定義自定義數據集
class CustomDataset(Dataset):
    def __init__(self):
        self.data = torch.randn(1000, 10)
        self.labels = torch.randint(0, 2, (1000,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 創建 DataLoader，啟用多進程數據加載
dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True)

# 測試數據加載
for data, labels in dataloader:
    print("Batch Data Shape:", data.shape)
    break

```
#### 中文解釋

1. 使用 `DataLoader` 的 `num_workers` 參數設置多進程數據加載。
2. 每個進程同時處理數據分塊，加速數據預處理。
3. 測試數據加載，檢查批次大小和形狀。

---

### **89. 如何在 PyTorch 中進行模型的部署**

#### 代碼

1. **將模型保存為 TorchScript 格式**
    
```python
model = SimpleModel()
example_input = torch.randn(1, 10)
scripted_model = torch.jit.trace(model, example_input)
scripted_model.save("deployed_model.pt")
print("Model saved for deployment.")

```
    
2. **在 C++ 或其他環境中加載**
    
    cpp
```python
#include <torch/script.h>
#include <iostream>

int main() {
    torch::jit::script::Module module = torch::jit::load("deployed_model.pt");
    at::Tensor input = torch::randn({1, 10});
    at::Tensor output = module.forward({input}).toTensor();
    std::cout << "Model output: " << output << std::endl;
    return 0;
}

```
    

#### 中文解釋

1. 使用 `torch.jit.trace` 將模型保存為 TorchScript 格式。
2. 在 C++ 中加載保存的模型，實現跨平台部署。

---

### **90. 使用 `torch.utils.tensorboard` 來監控訓練過程**

#### 代碼
```python
from torch.utils.tensorboard import SummaryWriter

# 初始化 TensorBoard 寫入器
writer = SummaryWriter("runs/experiment1")

# 模擬訓練過程
for epoch in range(10):
    train_loss = 0.1 * (10 - epoch)  # 模擬訓練損失
    val_loss = 0.1 * (10 - epoch) + 0.01  # 模擬驗證損失
    writer.add_scalar("Loss/Train", train_loss, epoch)
    writer.add_scalar("Loss/Validation", val_loss, epoch)

# 可視化模型結構
model = SimpleModel()
example_input = torch.randn(1, 10)
writer.add_graph(model, example_input)

writer.close()
print("Training process logged to TensorBoard.")

```
#### 中文解釋

1. 使用 `SummaryWriter` 將訓練過程中的損失值記錄到 TensorBoard。
2. 使用 `add_scalar` 添加標量數據（如損失值）。
3. 使用 `add_graph` 可視化模型結構。
4. 打開 TensorBoard 查看結果：
    
    bash
```python
tensorboard --logdir=runs

```
    

### **91. 如何在 PyTorch 中進行模型的梯度檢查（Gradient Checking）**

梯度檢查是用於驗證自動微分的正確性，通過數值梯度與自動梯度的對比實現。

#### 代碼

```python
import torch

# 定義簡單模型
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)

# 數值梯度計算函數
def numerical_gradient_check(model, inputs, targets, epsilon=1e-5):
    inputs.requires_grad = True
    outputs = model(inputs)
    loss = torch.nn.MSELoss()(outputs, targets)

    # 計算自動梯度
    loss.backward()
    auto_grad = inputs.grad.clone()

    # 數值梯度
    num_grad = torch.zeros_like(inputs)
    for i in range(inputs.numel()):
        original_value = inputs.view(-1)[i].item()

        # 增量計算
        inputs.view(-1)[i] = original_value + epsilon
        loss_plus = torch.nn.MSELoss()(model(inputs), targets)

        # 減量計算
        inputs.view(-1)[i] = original_value - epsilon
        loss_minus = torch.nn.MSELoss()(model(inputs), targets)

        # 還原原始值
        inputs.view(-1)[i] = original_value

        # 數值梯度
        num_grad.view(-1)[i] = (loss_plus - loss_minus) / (2 * epsilon)

    return auto_grad, num_grad

# 測試梯度檢查
model = SimpleModel()
inputs = torch.tensor([[1.0, 2.0]], requires_grad=True)
targets = torch.tensor([[3.0]])

auto_grad, num_grad = numerical_gradient_check(model, inputs, targets)
print("Automatic Gradient:", auto_grad)
print("Numerical Gradient:", num_grad)

```
#### 中文解釋

1. 定義簡單模型並設置目標函數（均方誤差）。
2. 使用 `torch.autograd` 計算自動梯度。
3. 使用有限差分公式計算數值梯度，並對比自動梯度與數值梯度。
4. 梯度檢查的數值與自動結果應該接近。

---

### **92. 使用 `torch.quantization` 來量化模型以減少其大小**

#### 代碼

```python
import torch
import torch.quantization

# 定義模型
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# 初始化模型
model = SimpleModel()

# 配置量化
model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
print("QConfig:", model.qconfig)

# 假量化準備
model_prepared = torch.quantization.prepare(model)
print("Model prepared for quantization.")

# 校準模型
data = torch.randn(100, 10)
with torch.no_grad():
    for _ in range(10):
        model_prepared(data)

# 量化模型
quantized_model = torch.quantization.convert(model_prepared)
print("Quantized Model:", quantized_model)

# 查看模型大小對比
original_size = torch.save(model.state_dict(), "original_model.pth")
quantized_size = torch.save(quantized_model.state_dict(), "quantized_model.pth")
print(f"Original Model Size: {original_size / 1024:.2f} KB")
print(f"Quantized Model Size: {quantized_size / 1024:.2f} KB")

```
#### 中文解釋

1. 定義簡單模型，設置量化配置為 `fbgemm`（適合 x86 平台）。
2. 使用 `torch.quantization.prepare` 執行假量化，收集校準數據範圍。
3. 使用 `torch.quantization.convert` 轉換模型為量化版本。
4. 校準數據後量化，並比較量化模型與原始模型的大小。
