


|                            |                                                                                  |
| -------------------------- | -------------------------------------------------------------------------------- |
| 多層感知器（MLP）                 | 這是最基本的前饋神經網路，包含輸入層、隱藏層和輸出層。由於其簡單性，MLP 的實作題目在面試中出現的機率較高。                          |
| CNN                        |                                                                                  |
| 循環神經網路（RNN）                | 特別是長短期記憶網路（LSTM）和門控循環單元（GRU），這些網路適用於處理序列數據。由於其在自然語言處理中的重要性，RNN 的實作題目在面試中也相當常見。   |
| 自編碼器（Autoencoder）          | 這是一種無監督學習模型，用於數據的降維和特徵學習。在面試中，可能會要求候選人實作自編碼器，特別是在需要評估無監督學習能力時。                   |
| 生成對抗網路（GAN）                | 由生成器和判別器組成，用於生成與真實數據相似的數據。由於其複雜性，手寫 GAN 的題目在面試中出現的機率較低，但在某些專注於生成模型的職位中可能會被問及。    |
| 圖神經網路（GNN）                 | 適用於處理圖結構數據，如社交網路或分子結構。由於其專業性，GNN 的實作題目在一般面試中較少出現，但在需要處理圖數據的職位中可能會被問及。            |
| 注意力機制（Attention Mechanism） | 雖然注意力機制通常與 Transformer 相關，但也可以在其他模型中使用。面試中可能會要求候選人實作簡單的注意力機制，特別是在需要評估對序列數據處理能力時。 |
| Transformer                |                                                                                  |
| ViT                        |                                                                                  |
|                            |                                                                                  |
總的來說，MLP 和 RNN 的實作題目在面試中出現的機率較高，而自編碼器和注意力機制次之。GAN 和 GNN 的實作題目相對較少，但在特定領域的職位中可能會被問及。


|             |                                                                               |
| ----------- | ----------------------------------------------------------------------------- |
| conv2d      | nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1) |
| BatchNorm2d | nn.BatchNorm2d(64)                                                            |
| MaxPool2d   | nn.MaxPool2d(kernel_size=3, stride=2, padding=1)                              |


|                                   |                                                                                   |
| --------------------------------- | --------------------------------------------------------------------------------- |
| 激活函數（Activation Functions）        | 如 Sigmoid、ReLU、Tanh 等。這些函數是神經網路的基本組成部分，手寫它們的實現有助於評估候選人對神經網路運作原理的理解。出題機率較高。        |
| 損失函數（Loss Functions）              | 如均方誤差（MSE）、交叉熵損失等。理解並能手寫損失函數的實現，顯示候選人對模型訓練過程的深入理解。出題機率中等。                         |
| 優化算法（Optimization Algorithms）     | 如梯度下降（Gradient Descent）、隨機梯度下降（SGD）、Adam 等。手寫這些算法的實現，能評估候選人對模型參數更新機制的掌握程度。出題機率中等。 |
| 正則化方法（Regularization Techniques）  | 如 L1、L2 正則化、Dropout 等。這些方法用於防止模型過擬合，手寫其實現可評估候選人對模型泛化能力的理解。出題機率較低。                 |
| 批量歸一化（Batch Normalization）        | 這是一種加速神經網路訓練並提高穩定性的方法。手寫其實現可評估候選人對訓練過程中數據分佈變化的理解。出題機率較低。                          |
| 反向傳播算法（Backpropagation Algorithm） | 這是神經網路訓練的核心算法，手寫其實現可評估候選人對梯度計算和參數更新的理解。出題機率較高。                                    |
| 注意力機制（Attention Mechanism）        | 特別是在序列模型中，如自注意力（Self-Attention）等。手寫其實現可評估候選人對序列數據處理和特徵加權的理解。出題機率中等。               |
| 卷積操作（Convolution Operation        | 在卷積神經網路（CNN）中，手寫卷積操作的實現可評估候選人對空間特徵提取的理解。出題機率中等。                                   |
| 池化操作（Pooling Operation）           | 如最大池化（Max Pooling）、平均池化（Average Pooling）等。手寫其實現可評估候選人對特徵降維和信息濃縮的理解。出題機率較低。        |
|                                   |                                                                                   |
總的來說，激活函數、反向傳播算法和優化算法的手寫實現題目在面試中出現的機率較高，因為這些元件是神經網路的核心組成部分，能夠直接反映候選人對深度學習基礎的掌握程度。



[卷积神经网络实战之手写CNN](https://blog.csdn.net/weixin_39451323/article/details/90680549)

[numpy实现卷积神经网络（CNN）](https://github.com/masamibf/numpy-for-CNN)

[Pytorch手写Transformer完整代码](https://www.kaggle.com/code/commxuyun/pytorch-transformer)



在人工智慧工程師的技術面試中，手寫卷積神經網路（CNN）的實作題目相當常見。這些題目可能要求使用或不使用 PyTorch 等深度學習框架來實現。以下是相關的資源連結、完整代碼示例以及中文解釋：

### 1. 用pytorch手寫多層感知器（MLP）

```
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 定義 MLP 模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 超參數設定
input_size = 784  # 28x28
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# 資料預處理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加載 MNIST 資料集
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型、損失函數和優化器
model = MLP(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 訓練模型
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28)

        # 前向傳播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向傳播及優化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

# 測試模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'測試準確率：{100 * correct / total:.2f}%')

```
上述代碼使用 PyTorch 實現了一個簡單的多層感知器（MLP）模型，用於 MNIST 手寫數字識別。

1. **模型定義**：
    
    - `MLP` 類繼承自 `nn.Module`，包含兩個全連接層（`fc1` 和 `fc2`）以及一個 ReLU 激活函數。
    - 在 `forward` 方法中，輸入數據首先經過第一個全連接層，然後經過 ReLU 激活，最後通過第二個全連接層得到輸出。
2. **超參數設定**：
    
    - `input_size`：輸入層大小，對應於 28x28 的圖像像素數。
    - `hidden_size`：隱藏層神經元數量。
    - `num_classes`：分類數量，對應於 10 個數字類別。
    - `num_epochs`：訓練迭代次數。
    - `batch_size`：每次訓練的批次大小。
    - `learning_rate`：學習率。
3. **資料預處理**：
    
    - 使用 `transforms` 對圖像進行張量化並標準化處理。
4. **資料加載**：
    
    - 使用 `torchvision.datasets.MNIST` 加載訓練和測試資料集，並使用 `DataLoader` 進行批次處理。
5. **模型訓練**：
    
    - 對每個 epoch，遍歷訓練資料，將圖像展平為一維向量，然後進行前向傳播、計算損失、反向傳播和優化。
    - 每 100 個步驟輸出一次當前的損失值。
6. **模型測試**：
    
    - 在測試資料集上評估模型的準確率，並輸出最終結果。

### 2. 不用pytorch手寫多層感知器（MLP）

```
import numpy as np

# Sigmoid 激活函數及其導數
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# MLP 類實現
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        # 初始化權重
        self.learning_rate = learning_rate
        self.weights_input_hidden = np.random.rand(input_size, hidden_size) - 0.5
        self.weights_hidden_output = np.random.rand(hidden_size, output_size) - 0.5

    def forward(self, x):
        # 前向傳播計算
        self.input = x
        self.hidden_input = np.dot(self.input, self.weights_input_hidden)
        self.hidden_output = sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output)
        self.final_output = sigmoid(self.final_input)
        return self.final_output

    def backward(self, y_true):
        # 計算輸出層誤差
        output_error = y_true - self.final_output
        output_delta = output_error * sigmoid_derivative(self.final_output)

        # 計算隱藏層誤差
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

        # 更新權重
        self.weights_hidden_output += self.learning_rate * np.dot(self.hidden_output.T, output_delta)
        self.weights_input_hidden += self.learning_rate * np.dot(self.input.T, hidden_delta)

    def train(self, x, y_true, epochs=1000):
        # 進行指定次數的訓練迭代
        for epoch in range(epochs):
            for i in range(len(x)):
                output = self.forward(x[i])
                self.backward(y_true[i])

    def predict(self, x):
        # 預測輸出
        return self.forward(x)

# 建立數據集（如邏輯閘 XOR 問題）
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# 初始化 MLP 並進行訓練
mlp = MLP(input_size=2, hidden_size=2, output_size=1, learning_rate=0.1)
mlp.train(X, y, epochs=10000)

# 測試 MLP
for x in X:
    print(f'輸入: {x}, 預測: {mlp.predict(x)}')

```
1. **激活函數**：
    
    - 使用 sigmoid 函數作為激活函數，這裡定義了 sigmoid 函數及其導數，用於計算每層的輸出以及反向傳播中的梯度。
2. **MLP 類**：
    
    - `__init__` 方法：初始化 MLP 的權重，`weights_input_hidden` 用於輸入層到隱藏層的權重矩陣，`weights_hidden_output` 用於隱藏層到輸出層的權重矩陣。
    - `forward` 方法：進行前向傳播，輸入數據經過隱藏層和輸出層，經過 sigmoid 激活函數的變換得到最終輸出。
    - `backward` 方法：計算輸出層和隱藏層的誤差，並基於 sigmoid 的導數更新權重。
    - `train` 方法：根據指定的 epoch 數進行多次訓練，每次訓練會將數據逐一輸入模型，進行前向和反向傳播來更新權重。
    - `predict` 方法：使用訓練後的模型進行預測。
3. **數據集**：
    
    - 此示例使用 XOR 閘的數據集。`X` 為輸入數據，`y` 為目標輸出，用於測試模型能否正確學習 XOR 邏輯。
4. **訓練與測試**：
    
    - 初始化 MLP，並使用 XOR 數據集進行訓練。在訓練完成後，對每個輸入數據進行預測，並輸出預測結果。

此實作展示了如何使用 NumPy 從頭構建一個簡單的 MLP，適合用於面試中的手寫代碼題目。

### 3. 用pytorch手寫CNN
```
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 定義 CNN 模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 定義卷積層
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # 定義全連接層
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # 32通道, 7x7 大小的 feature map
        self.fc2 = nn.Linear(128, 10)  # 最後的輸出對應於10個類別

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)  # 展平成一維
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 超參數設定
batch_size = 64
learning_rate = 0.001
num_epochs = 5

# 資料預處理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加載 MNIST 資料集
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型、損失函數和優化器
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 訓練模型
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 前向傳播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向傳播及優化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# 測試模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'測試準確率：{100 * correct / total:.2f}%')

```
- **模型定義**：
    
    - `CNN` 類繼承自 `nn.Module`，包含兩個卷積層、池化層和兩個全連接層。
    - `conv1` 和 `conv2` 是卷積層，用於提取圖像的特徵。`nn.Conv2d` 定義了輸入通道數、輸出通道數和卷積核大小。
    - `pool` 是最大池化層，將圖像尺寸減半以縮減特徵圖的大小。
    - `fc1` 和 `fc2` 是全連接層，用於將特徵圖展平並分類。`fc2` 的輸出大小為 10 對應於 10 個數字類別（0-9）。
- **前向傳播**：
    
    - `forward` 方法定義了前向傳播過程。
    - 輸入圖像先經過兩個卷積層和池化層，提取圖像的深層特徵，並將其展平成一維。
    - 經過全連接層後輸出分類結果。
- **超參數設置**：
    
    - `batch_size`：每次訓練的批次大小。
    - `learning_rate`：學習率，用於控制優化器的步伐大小。
    - `num_epochs`：訓練的迭代次數。
- **資料預處理**：
    
    - 使用 `transforms.ToTensor()` 將圖像轉為張量，並用 `transforms.Normalize` 對圖像進行標準化處理，使其均值為 0，方差為 1。
- **模型訓練**：
    
    - 每個 epoch 中，遍歷訓練資料集並進行前向傳播、計算損失、反向傳播及優化。
    - 每 100 步輸出當前的訓練損失。
- **模型測試**：
    
    - 在測試資料集上評估模型的準確率，並輸出最終結果。
### 4. 不用pytorch手寫CNN
```
import numpy as np

# 定義 ReLU 激活函數及其導數
def relu(x):
    return np.maximum(0, x)

# 定義最大池化操作
def max_pooling(x, size=2, stride=2):
    n, h, w, c = x.shape
    h_out = (h - size) // stride + 1
    w_out = (w - size) // stride + 1
    out = np.zeros((n, h_out, w_out, c))
    for i in range(h_out):
        for j in range(w_out):
            out[:, i, j, :] = np.max(x[:, i*stride:i*stride+size, j*stride:j*stride+size, :], axis=(1, 2))
    return out

# 定義卷積操作
def conv2d(x, kernel, stride=1, padding=0):
    n, h, w, c = x.shape
    kh, kw, c, nc = kernel.shape
    h_out = (h + 2*padding - kh) // stride + 1
    w_out = (w + 2*padding - kw) // stride + 1
    x_padded = np.pad(x, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')
    out = np.zeros((n, h_out, w_out, nc))
    for i in range(h_out):
        for j in range(w_out):
            x_slice = x_padded[:, i*stride:i*stride+kh, j*stride:j*stride+kw, :]
            out[:, i, j, :] = np.tensordot(x_slice, kernel, axes=([1, 2, 3], [0, 1, 2]))
    return out

# 建立簡單的 CNN 模型
class SimpleCNN:
    def __init__(self, num_filters, filter_size, input_shape, pool_size=2, pool_stride=2):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.input_shape = input_shape
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.conv_kernel = np.random.randn(filter_size, filter_size, input_shape[-1], num_filters) * 0.1
        self.fc_weights = np.random.randn(num_filters * (input_shape[0] // pool_stride) * (input_shape[1] // pool_stride), 10) * 0.1

    def forward(self, x):
        # 卷積層
        self.conv_out = conv2d(x, self.conv_kernel, stride=1, padding=1)
        self.conv_out_relu = relu(self.conv_out)
        
        # 最大池化層
        self.pool_out = max_pooling(self.conv_out_relu, size=self.pool_size, stride=self.pool_stride)
        
        # 全連接層
        self.fc_input = self.pool_out.reshape(x.shape[0], -1)
        self.fc_output = np.dot(self.fc_input, self.fc_weights)
        
        return self.fc_output

# 示例數據 (假設每個圖像大小為 8x8，灰階單通道)
np.random.seed(0)
x = np.random.randn(5, 8, 8, 1)  # 5 個樣本，大小為 8x8x1
y = np.random.randint(0, 10, size=(5,))  # 5 個樣本的目標標籤

# 初始化並前向傳播
cnn = SimpleCNN(num_filters=3, filter_size=3, input_shape=(8, 8, 1))
output = cnn.forward(x)

print("CNN 輸出：", output)

```
- **ReLU 激活函數**：
    
    - `relu` 函數應用於每個輸出值，將負值轉為 0，保留正值，用於引入非線性。
- **最大池化操作**：
    
    - `max_pooling` 函數接受一個 4D 輸入張量 (批量大小，高度，寬度，通道)。
    - 池化操作將特徵圖中的每個區域取最大值，從而減少特徵圖的尺寸。
- **卷積操作**：
    
    - `conv2d` 函數執行卷積操作。
    - 函數先在每個輸入特徵圖周圍添加零填充，然後通過移動卷積核，逐一計算卷積核與輸入特徵圖重疊部分的點積。
- **SimpleCNN 模型**：
    
    - `SimpleCNN` 類包含卷積層、ReLU 激活層、池化層和一個全連接層。
    - 初始化時，卷積核和全連接層的權重隨機生成。
    - `forward` 方法中，輸入數據先經過卷積層，應用 ReLU 激活，然後通過池化層，最後展平並輸入到全連接層。
- **測試與輸出**：
    
    - 創建一個 SimpleCNN 實例，並對 5 個樣本進行前向傳播，輸出每個樣本的預測值。

### 5. 用pytorch手寫循環神經網路（RNN）
```
import torch
import torch.nn as nn
import torch.optim as optim

# 定義 RNN 模型
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        # 定義 RNN 層
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        # 定義全連接層
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隱藏層狀態為零
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        # RNN 前向傳播
        out, _ = self.rnn(x, h0)
        # 取最後一個時間步的輸出
        out = self.fc(out[:, -1, :])
        return out

# 設定超參數
input_size = 10    # 每個時間步的特徵數量
hidden_size = 20   # 隱藏層大小
output_size = 1    # 輸出層大小
num_epochs = 100
learning_rate = 0.01

# 生成示例數據 (這裡使用隨機數據模擬)
torch.manual_seed(0)
batch_size = 5
sequence_length = 7  # 序列長度
x = torch.randn(batch_size, sequence_length, input_size)
y = torch.randn(batch_size, output_size)  # 模擬目標輸出

# 初始化模型、損失函數和優化器
model = SimpleRNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 訓練模型
for epoch in range(num_epochs):
    # 前向傳播
    outputs = model(x)
    loss = criterion(outputs, y)
    
    # 反向傳播和優化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 測試模型
model.eval()
with torch.no_grad():
    test_output = model(x)
    print("測試輸出：", test_output)

```
- **模型定義**：
    
    - `SimpleRNN` 類繼承自 `nn.Module`，包含一層 RNN 層和一層全連接層。
    - `self.rnn` 是 RNN 層，用於處理序列數據，`input_size` 表示每個時間步的特徵數量，`hidden_size` 表示隱藏層的大小。
    - `self.fc` 是全連接層，用於將 RNN 的隱藏狀態映射到最終的輸出。
- **前向傳播**：
    
    - 在 `forward` 方法中，`h0` 初始化為零，表示 RNN 的初始隱藏狀態。
    - `self.rnn(x, h0)` 執行前向傳播，返回整個序列的輸出 `out` 和最後的隱藏狀態 `_`。
    - `out[:, -1, :]` 選取最後一個時間步的輸出，然後通過全連接層得到最終的輸出。
- **超參數設置**：
    
    - `input_size`：輸入特徵的數量。
    - `hidden_size`：RNN 隱藏層的大小。
    - `output_size`：輸出的維度。
    - `num_epochs`：訓練的迭代次數。
    - `learning_rate`：學習率。
- **生成數據**：
    
    - 此處使用隨機數據來模擬輸入 `x` 和目標輸出 `y`。`x` 的形狀為 `(batch_size, sequence_length, input_size)`，`y` 的形狀為 `(batch_size, output_size)`。
- **訓練模型**：
    
    - 每個 epoch，模型通過前向傳播計算損失，然後執行反向傳播和權重更新。
    - 每隔 10 個 epoch 輸出一次當前損失值。
- **測試模型**：
    
    - 訓練完成後，在不計算梯度的情況下使用測試數據進行前向傳播，並輸出結果。
### 6. 不用pytorch手寫循環神經網路（RNN）
```
import numpy as np

# 定義 Sigmoid 和 Tanh 函數及其導數
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

# RNN 類實現
class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # 初始化權重矩陣
        self.W_xh = np.random.randn(input_size, hidden_size) * 0.01  # 輸入到隱藏層的權重
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01  # 隱藏層到隱藏層的權重
        self.W_hy = np.random.randn(hidden_size, output_size) * 0.01  # 隱藏層到輸出層的權重

        # 初始化隱藏層狀態
        self.h = np.zeros(hidden_size)

    def forward(self, x):
        self.h = tanh(np.dot(x, self.W_xh) + np.dot(self.h, self.W_hh))
        y = sigmoid(np.dot(self.h, self.W_hy))
        return y

    def backward(self, x, y_true, y_pred):
        # 計算輸出層誤差
        dy = (y_pred - y_true) * y_pred * (1 - y_pred)
        
        # 計算隱藏層誤差
        dh = (1 - self.h ** 2) * np.dot(dy, self.W_hy.T)
        
        # 更新權重
        self.W_hy -= self.learning_rate * np.outer(self.h, dy)
        self.W_hh -= self.learning_rate * np.outer(self.h, dh)
        self.W_xh -= self.learning_rate * np.outer(x, dh)

    def train(self, x_seq, y_seq, epochs=1000):
        for epoch in range(epochs):
            total_loss = 0
            for x, y_true in zip(x_seq, y_seq):
                y_pred = self.forward(x)
                self.backward(x, y_true, y_pred)
                total_loss += np.sum((y_pred - y_true) ** 2)
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# 示例數據
np.random.seed(0)
x_seq = np.random.randn(10, 3)  # 10 個序列，每個序列的特徵數為 3
y_seq = np.random.randint(0, 2, (10, 1))  # 模擬目標輸出為二元標籤

# 初始化並訓練 RNN
rnn = SimpleRNN(input_size=3, hidden_size=5, output_size=1, learning_rate=0.1)
rnn.train(x_seq, y_seq, epochs=1000)

# 測試 RNN
for x in x_seq:
    print(f"輸入: {x}, 預測: {rnn.forward(x)}")

```
- **激活函數**：
    
    - `sigmoid` 和 `tanh` 是常用的激活函數，其中 `sigmoid` 用於輸出層，`tanh` 用於隱藏層。
- **SimpleRNN 類**：
    
    - `__init__` 方法：初始化 RNN，設置輸入、隱藏和輸出層的大小，並初始化權重矩陣和隱藏層狀態。
        - `W_xh` 是輸入到隱藏層的權重矩陣。
        - `W_hh` 是隱藏層到隱藏層的循環權重矩陣。
        - `W_hy` 是隱藏層到輸出層的權重矩陣。
- **前向傳播**：
    
    - `forward` 方法中，根據當前的輸入和隱藏層狀態，計算新的隱藏層狀態和輸出。
    - `self.h` 是隱藏層的狀態，`y` 是當前時間步的輸出。
- **反向傳播**：
    
    - `backward` 方法中，計算輸出層誤差 `dy` 和隱藏層誤差 `dh`，然後更新權重。
    - 誤差更新公式基於 `sigmoid` 和 `tanh` 的導數。
- **訓練**：
    
    - `train` 方法中，對每個 epoch 遍歷輸入序列 `x_seq`，計算預測值、反向傳播並累積損失。
    - 每隔 100 個 epoch 輸出當前的總損失。
- **測試**：
    
    - 訓練完成後，對輸入序列進行預測，並輸出每個輸入序列的預測結果。

### 7. 用pytorch手寫自編碼器（Autoencoder）
```
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 定義 Autoencoder 模型
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # 編碼器
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        # 解碼器
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()  # 使用 Sigmoid 將輸出限制在 [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 超參數設定
num_epochs = 5
batch_size = 64
learning_rate = 0.001

# 資料預處理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加載 MNIST 資料集
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# 初始化模型、損失函數和優化器
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 訓練模型
for epoch in range(num_epochs):
    for data, _ in train_loader:
        data = data.view(-1, 28 * 28)  # 展平圖像為 1D 向量

        # 前向傳播
        output = model(data)
        loss = criterion(output, data)  # 計算重建誤差

        # 反向傳播及優化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 測試模型並可視化
sample_data, _ = next(iter(train_loader))
sample_data = sample_data.view(-1, 28 * 28)
with torch.no_grad():
    reconstructed = model(sample_data)

# 可視化原圖和重建圖
fig, axes = plt.subplots(2, 10, figsize=(10, 2))
for i in range(10):
    # 顯示原始圖像
    axes[0, i].imshow(sample_data[i].view(28, 28), cmap='gray')
    axes[0, i].axis('off')
    # 顯示重建圖像
    axes[1, i].imshow(reconstructed[i].view(28, 28), cmap='gray')
    axes[1, i].axis('off')

plt.show()

```
- **模型定義**：
    
    - `Autoencoder` 類繼承自 `nn.Module`，包含編碼器（encoder）和解碼器（decoder）。
    - 編碼器將輸入數據壓縮為更小的維度，解碼器則將其還原至原始輸入尺寸。
    - 編碼器由三層全連接層組成，將 28×28 的圖像壓縮到 32 個特徵。
    - 解碼器將壓縮的 32 維特徵重新解壓至 28×28 的圖像尺寸，並使用 Sigmoid 激活函數將輸出限制在 [0, 1]。
- **超參數設置**：
    
    - `num_epochs`：訓練的迭代次數。
    - `batch_size`：每次訓練的批次大小。
    - `learning_rate`：學習率，用於控制優化步伐。
- **資料預處理**：
    
    - 使用 `transforms.ToTensor()` 將圖像轉為張量，並用 `transforms.Normalize` 對圖像進行標準化處理。
- **模型訓練**：
    
    - 每個 epoch 中，將圖像展平成一維向量，進行前向傳播、計算損失、反向傳播及更新權重。
    - 損失函數使用 `MSELoss`，即均方誤差，用於衡量重建圖像與原始圖像之間的差異。
- **測試和可視化**：
    
    - 使用部分訓練樣本進行測試並生成重建圖像。
    - 使用 Matplotlib 可視化結果，顯示原始圖像和重建圖像，以觀察模型效果。
### 8. 不用pytorch手寫自編碼器（Autoencoder）
```
import numpy as np

# 定義 Sigmoid 激活函數及其導數
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# 自編碼器類
class SimpleAutoencoder:
    def __init__(self, input_size, hidden_size, learning_rate=0.01):
        # 初始化權重
        self.learning_rate = learning_rate
        self.weights_encoder = np.random.randn(input_size, hidden_size) * 0.1
        self.weights_decoder = np.random.randn(hidden_size, input_size) * 0.1

    def forward(self, x):
        # 前向傳播：編碼器和解碼器
        self.input = x
        self.hidden = sigmoid(np.dot(self.input, self.weights_encoder))
        self.output = sigmoid(np.dot(self.hidden, self.weights_decoder))
        return self.output

    def backward(self):
        # 計算誤差
        output_error = self.input - self.output
        output_delta = output_error * sigmoid_derivative(self.output)

        # 計算隱藏層的誤差
        hidden_error = np.dot(output_delta, self.weights_decoder.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden)

        # 更新權重
        self.weights_decoder += self.learning_rate * np.dot(self.hidden.T, output_delta)
        self.weights_encoder += self.learning_rate * np.dot(self.input.T, hidden_delta)

    def train(self, data, epochs=1000):
        for epoch in range(epochs):
            total_loss = 0
            for x in data:
                # 前向傳播
                self.forward(x)
                # 計算損失
                total_loss += np.sum((self.input - self.output) ** 2)
                # 反向傳播
                self.backward()
            if (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch+1}, Loss: {total_loss / len(data):.4f}')

# 示例數據
np.random.seed(0)
data = np.random.rand(10, 5)  # 模擬輸入數據 (10 個樣本，每個樣本 5 個特徵)

# 初始化自編碼器並進行訓練
autoencoder = SimpleAutoencoder(input_size=5, hidden_size=3, learning_rate=0.1)
autoencoder.train(data, epochs=1000)

# 測試自編碼器
for i, x in enumerate(data):
    reconstructed = autoencoder.forward(x)
    print(f"原始輸入: {x}")
    print(f"重建輸出: {reconstructed}")
    print()

```
1. **激活函數**：
    
    - 使用 `sigmoid` 函數進行激活，用於編碼和解碼過程中的非線性轉換。`sigmoid_derivative` 是其導數，用於計算反向傳播中的梯度。
2. **SimpleAutoencoder 類**：
    
    - `__init__` 方法：初始化自編碼器的權重矩陣和學習率。
        - `weights_encoder` 用於將輸入數據投影到較低維度的隱藏層。
        - `weights_decoder` 用於將隱藏層還原回輸入空間。
    - `forward` 方法：執行前向傳播，將輸入數據先經過編碼器（降維），再經過解碼器（還原）。
        - `self.hidden` 表示隱藏層的激活輸出。
        - `self.output` 表示模型的最終輸出，代表重建的輸入數據。
    - `backward` 方法：計算輸出層的誤差並進行反向傳播。
        - `output_delta` 計算輸出層的誤差梯度，用於更新解碼器權重。
        - `hidden_delta` 計算隱藏層的誤差梯度，用於更新編碼器權重。
3. **訓練**：
    
    - `train` 方法中，遍歷輸入數據並對每個樣本進行前向傳播和反向傳播。
    - 每 100 個 epoch 輸出一次損失值，用於監控訓練進程。
4. **測試**：
    
    - 訓練完成後，對每個輸入樣本進行前向傳播，並輸出原始數據和重建數據，以檢查模型性能。

這個自編碼器模型使用簡單的全連接層和 sigmoid 激活函數，展示了如何用 NumPy 實現降維和重建數據。此實作適合用於面試中的手寫代碼題目。
### 9. 用pytorch手寫生成對抗網路（GAN）
```
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 定義生成器 (Generator)
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Tanh()  # 輸出範圍為 [-1, 1]
        )

    def forward(self, x):
        return self.model(x)

# 定義鑑別器 (Discriminator)
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()  # 輸出為真實度概率
        )

    def forward(self, x):
        return self.model(x)

# 超參數設定
latent_size = 64
hidden_size = 256
image_size = 28 * 28  # MNIST 圖像尺寸
batch_size = 100
num_epochs = 200
learning_rate = 0.0002

# 資料預處理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加載 MNIST 資料集
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# 初始化生成器和鑑別器
generator = Generator(latent_size, hidden_size, image_size)
discriminator = Discriminator(image_size, hidden_size, 1)

# 損失函數和優化器
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate)

# 訓練 GAN
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(train_loader):
        # 準備真實數據和標籤
        images = images.view(-1, image_size)
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # 訓練鑑別器
        outputs = discriminator(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs

        # 生成假數據
        z = torch.randn(batch_size, latent_size)
        fake_images = generator(z)
        outputs = discriminator(fake_images.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs

        # 鑑別器總損失
        d_loss = d_loss_real + d_loss_fake
        optimizer_d.zero_grad()
        d_loss.backward()
        optimizer_d.step()

        # 訓練生成器
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)
        optimizer_g.zero_grad()
        g_loss.backward()
        optimizer_g.step()

    # 每隔一定 epoch 顯示損失
    if (epoch+1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}, '
              f'D(x): {real_score.mean().item():.4f}, D(G(z)): {fake_score.mean().item():.4f}')

# 測試生成器並顯示生成圖像
with torch.no_grad():
    z = torch.randn(batch_size, latent_size)
    fake_images = generator(z)
    fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)

    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(fake_images[i][0].cpu(), cmap='gray')
        plt.axis('off')
    plt.show()

```
1. **生成器 (Generator)**：
    
    - `Generator` 類別繼承自 `nn.Module`，用於生成假圖像。
    - 模型由三層全連接層構成。最後一層使用 `Tanh` 激活函數，將輸出範圍限制在 `[-1, 1]` 之間，方便後續輸出圖像。
    - `latent_size` 表示輸入的隱變量的維度，這些隱變量來自標準正態分佈，用於生成圖像。
2. **鑑別器 (Discriminator)**：
    
    - `Discriminator` 類別繼承自 `nn.Module`，用於判斷輸入圖像是真實還是偽造。
    - 模型由三層全連接層構成。最後一層使用 `Sigmoid` 激活函數，輸出真實度概率。
3. **超參數設置**：
    
    - `latent_size`：生成器的輸入維度（隱變量的大小）。
    - `hidden_size`：隱藏層的大小。
    - `image_size`：圖像的展平尺寸，對於 MNIST 圖像來說是 `28*28`。
    - `num_epochs`：訓練的迭代次數。
    - `learning_rate`：學習率。
4. **訓練流程**：
    
    - 訓練過程中，首先訓練鑑別器，使其能夠分辨真實和偽造的圖像。
        - 將真實圖像和標籤傳入鑑別器，計算 `d_loss_real`。
        - 然後生成假圖像並傳入鑑別器，計算 `d_loss_fake`。
        - 鑑別器的總損失為 `d_loss = d_loss_real + d_loss_fake`。
    - 然後訓練生成器，生成逼真的假圖像，使得鑑別器無法區分。
        - 假圖像傳入鑑別器，生成器的損失為 `g_loss = criterion(outputs, real_labels)`，因為希望騙過鑑別器。
5. **生成圖像的可視化**：
    
    - 訓練完成後，利用生成器生成假圖像，並將它們可視化，展示生成器的效果。

這個簡單的 GAN 模型展示了如何用 PyTorch 實現生成器和鑑別器，並進行對抗訓練。此實作適合用於面試中的手寫代碼題目。

### 10. 不用pytorch手寫生成對抗網路（GAN）
```
import numpy as np

# 激活函數和它的導數
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# 生成器類
class Generator:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.1
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.1

    def forward(self, x):
        self.hidden = sigmoid(np.dot(x, self.weights_input_hidden))
        output = sigmoid(np.dot(self.hidden, self.weights_hidden_output))
        return output

    def backward(self, x, grad_output):
        grad_hidden_output = grad_output * sigmoid_derivative(self.hidden)
        grad_input_hidden = np.dot(x.T, grad_hidden_output)

        # 更新權重
        self.weights_hidden_output += np.dot(self.hidden.T, grad_output) * 0.1
        self.weights_input_hidden += grad_input_hidden * 0.1

# 鑑別器類
class Discriminator:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.1
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.1

    def forward(self, x):
        self.hidden = sigmoid(np.dot(x, self.weights_input_hidden))
        output = sigmoid(np.dot(self.hidden, self.weights_hidden_output))
        return output

    def backward(self, x, y, output):
        error = y - output
        grad_output = error * sigmoid_derivative(output)
        grad_hidden_output = grad_output * sigmoid_derivative(self.hidden)
        grad_input_hidden = np.dot(x.T, grad_hidden_output)

        # 更新權重
        self.weights_hidden_output += np.dot(self.hidden.T, grad_output) * 0.1
        self.weights_input_hidden += grad_input_hidden * 0.1
        return error

# 訓練過程
def train_gan(epochs=10000):
    # 初始化生成器和鑑別器
    generator = Generator(input_size=3, hidden_size=5, output_size=1)
    discriminator = Discriminator(input_size=1, hidden_size=5, output_size=1)

    for epoch in range(epochs):
        # 訓練鑑別器
        real_data = np.random.uniform(0, 1, (1, 1))  # 真實數據
        fake_data = generator.forward(np.random.randn(1, 3))  # 生成假數據

        # 訓練鑑別器辨別真實數據
        real_output = discriminator.forward(real_data)
        real_error = discriminator.backward(real_data, np.array([[1]]), real_output)

        # 訓練鑑別器辨別假數據
        fake_output = discriminator.forward(fake_data)
        fake_error = discriminator.backward(fake_data, np.array([[0]]), fake_output)

        # 訓練生成器
        fake_data = generator.forward(np.random.randn(1, 3))
        fake_output = discriminator.forward(fake_data)
        generator.backward(fake_data, fake_output * (1 - fake_output))  # 更新生成器，使其生成更真實的數據

        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, D Real Error: {np.mean(real_error):.4f}, D Fake Error: {np.mean(fake_error):.4f}')

# 訓練 GAN
train_gan()

```
1. **激活函數**：
    
    - 使用 `sigmoid` 作為激活函數，限制輸出在 [0, 1] 區間。`sigmoid_derivative` 用於計算反向傳播中的梯度。
2. **生成器 (Generator)**：
    
    - `Generator` 類中，包含兩層全連接層。輸入為隨機噪聲，目標是生成偽造的數據。
    - `forward` 方法實現前向傳播，將隱藏層輸出計算為最終生成的數據。
3. **鑑別器 (Discriminator)**：
    
    - `Discriminator` 類中，包含兩層全連接層。輸入為真實或假數據，輸出代表該數據為真實的概率。
    - `forward` 方法進行前向傳播，計算數據的真實度。
    - `backward` 方法計算鑑別器的誤差，並更新權重。誤差基於真實數據標籤 `1` 和假數據標籤 `0`。
4. **訓練過程**：
    
    - `train_gan` 函數中，先訓練鑑別器，使其能夠區分真實數據和生成數據。
    - 對於每個 epoch：
        - 生成一組真實數據和一組假數據。
        - 使用鑑別器對真實數據和假數據進行分類，計算誤差，並反向傳播更新權重。
    - 訓練生成器，使生成的假數據越來越真實。生成器的目標是生成能夠騙過鑑別器的數據。
5. **輸出結果**：
    
    - 每 1000 次訓練迭代，打印鑑別器的真實和偽造數據分類誤差，以觀察訓練進展。

這個簡單的 GAN 模型展示了如何用 NumPy 手寫生成器和鑑別器，並進行對抗訓練，適合用於面試中的手寫代碼題目。

### 11. 用pytorch手寫圖神經網路（GNN）
```
import torch
import torch.nn as nn
import torch.optim as optim

# 定義簡單的圖卷積層（Graph Convolution Layer）
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        # 初始化權重
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        # 執行圖卷積操作
        support = torch.mm(x, self.weight)           # xW
        output = torch.spmm(adj, support)            # A * xW
        return output

# 定義 GNN 模型
class SimpleGNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleGNN, self).__init__()
        # 初始化兩層圖卷積層
        self.gc1 = GraphConvolution(input_size, hidden_size)
        self.gc2 = GraphConvolution(hidden_size, output_size)

    def forward(self, x, adj):
        # 第一層圖卷積 + ReLU 激活
        x = torch.relu(self.gc1(x, adj))
        # 第二層圖卷積
        x = self.gc2(x, adj)
        return x

# 構造示例圖數據
# 節點特徵矩陣（4 個節點，每個節點 3 個特徵）
features = torch.FloatTensor([[1, 0, 1],
                              [0, 1, 0],
                              [1, 1, 0],
                              [0, 0, 1]])

# 鄰接矩陣（4x4）表示節點之間的連接情況
adj = torch.FloatTensor([[1, 1, 0, 0],
                         [1, 1, 1, 0],
                         [0, 1, 1, 1],
                         [0, 0, 1, 1]])

# 節點標籤（假設 2 個類別，0 或 1）
labels = torch.LongTensor([0, 1, 0, 1])

# 初始化 GNN 模型和超參數
input_size = features.shape[1]
hidden_size = 4
output_size = 2
num_epochs = 200
learning_rate = 0.01

model = SimpleGNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 訓練 GNN 模型
for epoch in range(num_epochs):
    # 前向傳播
    outputs = model(features, adj)
    loss = criterion(outputs, labels)
    
    # 反向傳播及優化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 測試模型
model.eval()
with torch.no_grad():
    test_outputs = model(features, adj)
    _, predicted = torch.max(test_outputs, 1)
    print("節點預測標籤：", predicted)
    print("真實標籤：", labels)

```
1. **圖卷積層（GraphConvolution）**：
    
    - `GraphConvolution` 類定義了圖卷積層，包含參數權重矩陣 `self.weight`。
    - `forward` 方法實現圖卷積操作：`support = x * W`，再進行 `output = A * support`，其中 `A` 是鄰接矩陣，`x` 是節點特徵矩陣，`W` 是權重矩陣。
    - 這層實現的是一個基礎的圖卷積操作，即透過相鄰節點的特徵來更新中心節點的特徵。
2. **GNN 模型（SimpleGNN）**：
    
    - `SimpleGNN` 類包含兩層圖卷積層。
    - 第一層為圖卷積後接 `ReLU` 激活，第二層為圖卷積。
    - 模型最終輸出節點的特徵，並用於分類任務。
3. **圖數據示例**：
    
    - `features` 是節點特徵矩陣，共 4 個節點，每個節點包含 3 個特徵。
    - `adj` 是鄰接矩陣，表示節點之間的連接情況，包含自環。
    - `labels` 是節點的類別標籤，表示節點所屬的類別。
4. **訓練過程**：
    
    - 在每個 epoch 中，對特徵矩陣進行前向傳播，計算節點標籤的預測值，並使用交叉熵損失函數進行計算。
    - 執行反向傳播，更新權重。
    - 每 20 個 epoch 輸出一次損失值。
5. **測試模型**：
    
    - 訓練完成後，進行測試，對節點標籤進行預測，並與真實標籤比較，檢查模型的分類效果。

此 GNN 實作展示了如何用 PyTorch 實現簡單的圖卷積網路，適合用於面試中的手寫代碼題目。

### 12. 不用pytorch手寫圖神經網路（GNN）
```
import numpy as np

# 定義 ReLU 激活函數
def relu(x):
    return np.maximum(0, x)

# 圖卷積層實現
class GraphConvolution:
    def __init__(self, in_features, out_features):
        # 初始化權重矩陣
        self.weight = np.random.randn(in_features, out_features) * 0.01

    def forward(self, x, adj):
        # 圖卷積: A * X * W
        support = np.dot(x, self.weight)      # X * W
        output = np.dot(adj, support)         # A * (X * W)
        return output

# GNN 模型
class SimpleGNN:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化兩層圖卷積層
        self.gc1 = GraphConvolution(input_size, hidden_size)
        self.gc2 = GraphConvolution(hidden_size, output_size)

    def forward(self, x, adj):
        # 第一層圖卷積 + ReLU 激活
        x = relu(self.gc1.forward(x, adj))
        # 第二層圖卷積
        x = self.gc2.forward(x, adj)
        return x

    def compute_loss(self, predictions, labels):
        # 使用均方誤差 (MSE) 作為損失函數
        return np.mean((predictions - labels) ** 2)

    def backward(self, x, adj, labels, predictions, learning_rate=0.01):
        # 假設為簡單的更新，不進行反向傳播的細節實現
        grad_output = 2 * (predictions - labels) / labels.shape[0]
        # 使用梯度下降更新權重
        self.gc2.weight -= learning_rate * np.dot(relu(self.gc1.forward(x, adj)).T, grad_output)
        self.gc1.weight -= learning_rate * np.dot(x.T, np.dot(adj, grad_output))

# 節點特徵矩陣（4 個節點，每個節點 3 個特徵）
features = np.array([[1, 0, 1],
                     [0, 1, 0],
                     [1, 1, 0],
                     [0, 0, 1]])

# 鄰接矩陣（4x4）表示節點之間的連接
adj = np.array([[1, 1, 0, 0],
                [1, 1, 1, 0],
                [0, 1, 1, 1],
                [0, 0, 1, 1]])

# 節點標籤（假設 2 個類別，0 或 1，並簡單地標記為數值）
labels = np.array([[1], [0], [1], [0]])

# 初始化 GNN 模型和超參數
input_size = features.shape[1]
hidden_size = 4
output_size = 1
num_epochs = 1000
learning_rate = 0.01

# 初始化 GNN 模型
model = SimpleGNN(input_size, hidden_size, output_size)

# 訓練 GNN 模型
for epoch in range(num_epochs):
    # 前向傳播
    predictions = model.forward(features, adj)
    # 計算損失
    loss = model.compute_loss(predictions, labels)

    # 反向傳播及更新權重
    model.backward(features, adj, labels, predictions, learning_rate=learning_rate)

    # 每隔 100 個 epoch 顯示損失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}')

# 測試模型
print("預測值：", predictions)
print("真實標籤：", labels)

```
1. **激活函數**：
    
    - `relu` 函數是 ReLU 激活函數，用於增加模型的非線性。
2. **圖卷積層（GraphConvolution）**：
    
    - `GraphConvolution` 類包含權重矩陣 `self.weight`，初始化為小的隨機值。
    - `forward` 方法實現了圖卷積操作：`support = X * W`，再計算 `output = A * support`。其中 `A` 是鄰接矩陣，`X` 是節點特徵矩陣，`W` 是權重矩陣。
3. **GNN 模型（SimpleGNN）**：
    
    - `SimpleGNN` 類包含兩層圖卷積層。
    - 第一層為圖卷積加上 `ReLU` 激活，第二層為圖卷積，用於最終的節點分類。
    - `compute_loss` 方法計算損失值，使用均方誤差（MSE）作為損失函數。
    - `backward` 方法進行簡單的梯度下降更新。這裡假設了簡化的更新方式，沒有詳細的反向傳播計算。
4. **圖數據示例**：
    
    - `features` 是節點特徵矩陣，有 4 個節點，每個節點包含 3 個特徵。
    - `adj` 是鄰接矩陣，表示節點之間的連接情況，並包含自環。
    - `labels` 是節點的類別標籤，用於訓練模型。
5. **訓練過程**：
    
    - 每個 epoch 中，模型對節點特徵進行前向傳播，計算預測值，並使用 MSE 計算損失。
    - 反向傳播過程中，簡單地基於梯度下降更新權重。
    - 每隔 100 個 epoch 輸出一次損失。
6. **測試模型**：
    
    - 訓練完成後，對節點進行分類預測，並打印預測結果與真實標籤。

這個簡單的 GNN 使用了兩層圖卷積來處理節點分類任務。它展示了如何用 NumPy 實現圖卷積網絡，適合用於面試中的手寫代碼題目。

### 13. 用pytorch手寫注意力機制（Attention Mechanism）
```
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定義 Scaled Dot-Product Attention 機制
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, query, key, value, mask=None):
        # 計算 QK^T / sqrt(d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        # 使用 mask 忽略特定位置的值（通常用於填充部分的遮掩）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 計算 softmax 得到注意力分數
        attention_weights = F.softmax(scores, dim=-1)

        # 計算注意力輸出
        output = torch.matmul(attention_weights, value)
        return output, attention_weights

# 定義多頭注意力機制（Multi-Head Attention）
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必須能夠整除 num_heads"

        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        # 定義線性層，用於將輸入映射到 Q、K、V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

        # Scaled Dot-Product Attention
        self.attention = ScaledDotProductAttention(self.d_k)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 將輸入投影到 Q、K、V
        query = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 計算注意力輸出
        output, attention_weights = self.attention(query, key, value, mask=mask)

        # 將多頭合併回去
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        # 通過最後一層線性變換
        output = self.fc(output)
        return output, attention_weights

# 測試多頭注意力
d_model = 16  # 模型的維度
num_heads = 4  # 注意力頭的數量
batch_size = 2
seq_length = 5

# 模擬輸入數據 (batch_size, seq_length, d_model)
query = torch.randn(batch_size, seq_length, d_model)
key = torch.randn(batch_size, seq_length, d_model)
value = torch.randn(batch_size, seq_length, d_model)

# 定義多頭注意力層並執行前向傳播
multi_head_attention = MultiHeadAttention(d_model, num_heads)
output, attention_weights = multi_head_attention(query, key, value)

print("多頭注意力輸出:", output)
print("注意力權重:", attention_weights)

```
1. **Scaled Dot-Product Attention 機制**：
    
    - `ScaledDotProductAttention` 類別中，`forward` 方法實現了注意力計算。
    - 首先計算 `scores = QK^T / sqrt(d_k)`，其中 `Q` 是查詢（query），`K` 是鍵（key），`d_k` 是 `Q` 和 `K` 的維度。
    - 如果有遮罩（mask），會將被遮掩的位置的分數設置為極低值（`-1e9`），避免這些位置對結果的影響。
    - 然後使用 `softmax` 對 `scores` 進行歸一化，得到注意力權重，最後將權重應用到 `V`（value）上計算出最終輸出。
2. **多頭注意力機制（Multi-Head Attention）**：
    
    - `MultiHeadAttention` 類包含多個頭的注意力計算。
    - `d_k = d_model / num_heads` 確保每個頭的維度一致。
    - 首先通過線性層將輸入的 `Q`、`K`、`V` 投影到多個頭，然後每個頭都使用 `ScaledDotProductAttention` 計算注意力。
    - 最後將各頭的輸出合併，通過全連接層（`fc`）映射回 `d_model` 維度。
3. **測試多頭注意力**：
    
    - `query`、`key` 和 `value` 都是隨機生成的張量，表示輸入序列。
    - 初始化多頭注意力層並執行前向傳播。
    - 輸出 `output` 是注意力機制的結果，`attention_weights` 是每個頭的注意力權重，用於觀察注意力的分佈情況。

這個簡單的實作展示了如何在 PyTorch 中手寫一個基本的多頭注意力機制，適合用於面試中的手寫代碼題目。
### 14. 不用pytorch手寫注意力機制（Attention Mechanism）
```
import numpy as np

# 定義 softmax 函數
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Scaled Dot-Product Attention 實現
def scaled_dot_product_attention(query, key, value, mask=None):
    # 計算 QK^T / sqrt(d_k)
    d_k = query.shape[-1]
    scores = np.dot(query, key.T) / np.sqrt(d_k)

    # 使用 mask 忽略某些位置的值（例如填充的位置）
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    # 計算 softmax 以獲得注意力權重
    attention_weights = softmax(scores)

    # 計算注意力的輸出
    output = np.dot(attention_weights, value)
    return output, attention_weights

# 測試 Scaled Dot-Product Attention
# 定義 Query, Key, Value 隨機數據
np.random.seed(0)
query = np.random.rand(3, 4)  # 3 個詞，每個詞 4 維特徵
key = np.random.rand(3, 4)
value = np.random.rand(3, 4)

# 可選的 Mask（如果需要遮罩某些位置）
mask = np.array([[1, 1, 0],
                 [1, 1, 1],
                 [1, 0, 1]])

# 計算注意力
output, attention_weights = scaled_dot_product_attention(query, key, value, mask=mask)

print("注意力輸出：\n", output)
print("注意力權重：\n", attention_weights)

```
1. **softmax 函數**：
    
    - `softmax` 函數對輸入進行歸一化，使得輸出為概率分佈，且所有元素之和為 1。
    - 這裡 `exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))` 用來避免溢出，`softmax` 用於計算注意力權重的分佈。
2. **Scaled Dot-Product Attention 機制**：
    
    - `scaled_dot_product_attention` 函數實現了基於點積的注意力機制。
    - `scores = np.dot(query, key.T) / np.sqrt(d_k)`：這一步計算查詢和鍵的點積，然後除以 `sqrt(d_k)`，其中 `d_k` 是 `query` 和 `key` 的最後一個維度。
    - 如果有 `mask`，則將被遮蔽的位置設置為一個非常小的值（-1e9），以避免這些位置的值參與計算。
    - 使用 `softmax` 對 `scores` 進行歸一化，得到注意力權重 `attention_weights`，表示 `query` 在各個 `key` 上的注意力分佈。
    - 最後，計算注意力輸出 `output = np.dot(attention_weights, value)`。
3. **測試 Scaled Dot-Product Attention**：
    
    - 定義 `query`、`key` 和 `value`，這裡使用隨機數據來模擬。
    - `mask` 是一個選擇性參數，如果需要遮掩某些位置則可以傳入 `mask` 矩陣。
    - `scaled_dot_product_attention` 函數返回 `output` 和 `attention_weights`。`output` 是注意力機制的輸出，`attention_weights` 是注意力權重矩陣，表示各查詢對應的注意力分佈。

這個簡單的 NumPy 實作展示了如何手寫一個基本的 Scaled Dot-Product Attention 機制，適合用於面試中的手寫代碼題目。

### 15. 用pytorch手寫Transformer 
```
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 定義 Scaled Dot-Product Attention 機制
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, value)
        return output, attention_weights

# 定義 Multi-Head Attention 機制
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(self.d_k)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 將 Q、K、V 投影到多個頭上
        query = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 計算多頭注意力
        output, attention_weights = self.attention(query, key, value, mask=mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        
        # 全連接層
        output = self.fc(output)
        return output

# 前饋層
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

# 位置編碼
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# Transformer 編碼器層
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 多頭自注意力
        attn_output = self.self_attn(x, x, x, mask)
        x = self.layernorm1(x + self.dropout(attn_output))

        # 前饋層
        ff_output = self.feed_forward(x)
        x = self.layernorm2(x + self.dropout(ff_output))

        return x

# 整體 Transformer 編碼器
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, vocab_size, max_len=5000):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, src, mask=None):
        x = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x, mask)

        return self.layernorm(x)

# 測試 Transformer 編碼器
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6
vocab_size = 10000
max_len = 50
batch_size = 2
seq_length = 10

# 模擬輸入數據
src = torch.randint(0, vocab_size, (batch_size, seq_length))

# 定義 Transformer 編碼器並執行前向傳播
encoder = TransformerEncoder(d_model, num_heads, d_ff, num_layers, vocab_size, max_len)
output = encoder(src)

print("Transformer 編碼器輸出:", output)

```
1. **Scaled Dot-Product Attention**：
    
    - `ScaledDotProductAttention` 類中，`forward` 方法計算注意力分數：`scores = QK^T / sqrt(d_k)`。如果有 `mask`，會將被遮蔽位置設為非常小的值（-1e9），避免影響最終結果。
    - 計算 `softmax` 得到注意力權重 `attention_weights`，並用權重加權 `value` 獲得輸出。
2. **Multi-Head Attention**：
    
    - `MultiHeadAttention` 類中，`query`、`key` 和 `value` 分別投影到多個頭，通過 `ScaledDotProductAttention` 計算每個頭的注意力，最後將所有頭的輸出拼接並通過全連接層。
3. **FeedForward 層**：
    
    - `FeedForward` 是一個兩層的全連接網絡，內含 ReLU 激活函數，用於增強非線性建模能力。
4. **位置編碼**：
    
    - `PositionalEncoding` 類為每個詞位置生成一個向量，確保模型對序列位置敏感。它使用正弦和餘弦來生成位置信息。
5. **Transformer 編碼器層**：
    
    - `TransformerEncoderLayer` 包含一個多頭自注意力層、一個前饋層，以及兩個殘差連接和層正則化。每個詞的表示依賴於其他詞，特別是前序詞。
6. **Transformer 編碼器**：
    
    - `TransformerEncoder` 類包含多個 `TransformerEncoderLayer` 層。首先嵌入詞向量，再加上位置編碼，然後逐層進行自注意力和前饋處理。
7. **測試 Transformer 編碼器**：
    
    - 初始化隨機輸入，通過 `TransformerEncoder` 進行前向傳播，最終輸出經過編碼的特徵。

這個簡單的實作展示了如何在 PyTorch 中手寫 Transformer 編碼器，適合用於面試中的手寫代碼題目。
### 16. 不用pytorch手寫Transformer 
```
import numpy as np

# 定義 softmax 函數
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# 定義 ReLU 激活函數
def relu(x):
    return np.maximum(0, x)

# Scaled Dot-Product Attention 機制
def scaled_dot_product_attention(query, key, value, mask=None):
    d_k = query.shape[-1]
    scores = np.dot(query, key.T) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    attention_weights = softmax(scores)
    output = np.dot(attention_weights, value)
    return output, attention_weights

# Multi-Head Attention 機制
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = np.random.randn(d_model, d_model)
        self.w_k = np.random.randn(d_model, d_model)
        self.w_v = np.random.randn(d_model, d_model)
        self.fc = np.random.randn(d_model, d_model)

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.shape
        x = x.reshape(batch_size, seq_length, self.num_heads, self.d_k)
        return np.transpose(x, (0, 2, 1, 3))

    def forward(self, query, key, value, mask=None):
        query = np.dot(query, self.w_q)
        key = np.dot(key, self.w_k)
        value = np.dot(value, self.w_v)

        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        outputs, _ = zip(*[scaled_dot_product_attention(q, k, v, mask) for q, k, v in zip(query, key, value)])
        concat_output = np.concatenate(outputs, axis=-1)
        output = np.dot(concat_output, self.fc)
        return output

# 位置編碼
def positional_encoding(seq_length, d_model):
    pos = np.arange(seq_length)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / d_model)
    pos_encoding = pos * angle_rates
    pos_encoding[:, 0::2] = np.sin(pos_encoding[:, 0::2])
    pos_encoding[:, 1::2] = np.cos(pos_encoding[:, 1::2])
    return pos_encoding

# 前饋網絡
class FeedForward:
    def __init__(self, d_model, d_ff):
        self.w1 = np.random.randn(d_model, d_ff)
        self.w2 = np.random.randn(d_ff, d_model)

    def forward(self, x):
        return np.dot(relu(np.dot(x, self.w1)), self.w2)

# Transformer 編碼器層
class TransformerEncoderLayer:
    def __init__(self, d_model, num_heads, d_ff):
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)

    def forward(self, x, mask=None):
        attn_output = self.self_attn.forward(x, x, x, mask)
        x = x + attn_output
        ff_output = self.feed_forward.forward(x)
        x = x + ff_output
        return x

# Transformer 編碼器
class TransformerEncoder:
    def __init__(self, d_model, num_heads, d_ff, num_layers, seq_length):
        self.layers = [TransformerEncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        self.positional_encoding = positional_encoding(seq_length, d_model)

    def forward(self, src, mask=None):
        x = src + self.positional_encoding
        for layer in self.layers:
            x = layer.forward(x, mask)
        return x

# 測試 Transformer 編碼器
d_model = 16
num_heads = 4
d_ff = 64
num_layers = 2
seq_length = 5
batch_size = 1

# 模擬輸入數據
np.random.seed(0)
src = np.random.rand(batch_size, seq_length, d_model)

# 定義 Transformer 編碼器並執行前向傳播
encoder = TransformerEncoder(d_model, num_heads, d_ff, num_layers, seq_length)
output = encoder.forward(src)

print("Transformer 編碼器輸出:", output)

```
1. **Scaled Dot-Product Attention 機制**：
    
    - `scaled_dot_product_attention` 函數計算注意力分數，公式為 `scores = QK^T / sqrt(d_k)`，然後通過 softmax 歸一化。
    - 如果有 `mask`，會將被遮蔽的位置設置為非常小的值（`-1e9`）以忽略這些位置。
    - `output` 是計算出的加權值。
2. **Multi-Head Attention 機制**：
    
    - `MultiHeadAttention` 類中，`split_heads` 方法將輸入分割成多頭。
    - 每個頭計算一次 `scaled_dot_product_attention`，然後將所有頭的輸出拼接起來，通過最終的線性層得到輸出。
3. **位置編碼**：
    
    - `positional_encoding` 函數生成固定的位置編碼。每個位置根據其位置編碼，使 Transformer 具有序列感知能力。
4. **前饋網絡**：
    
    - `FeedForward` 類使用兩層全連接層和 ReLU 激活來增強非線性建模能力。
5. **Transformer 編碼器層**：
    
    - `TransformerEncoderLayer` 類包含一個多頭注意力機制和一個前饋網絡。
    - 每層的輸出 `x` 都會加入 `self_attn` 和 `feed_forward` 的輸出，實現殘差連接。
6. **Transformer 編碼器**：
    
    - `TransformerEncoder` 包含多個 `TransformerEncoderLayer` 層。
    - `positional_encoding` 使模型在序列數據上保留位置信息。
7. **測試 Transformer 編碼器**：
    
    - 模擬輸入數據，通過 `TransformerEncoder` 前向傳播。
    - 最終的 `output` 是 Transformer 編碼器的輸出特徵。

這個簡單的 NumPy 實作展示了 Transformer 編碼器的基本結構，適合用於面試中的手寫代碼題目。

### 17. 用pytorch手寫ViT   
```
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 定義 Patch Embedding 類
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # 卷積層模擬分割並嵌入每個 patch
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (batch_size, embed_dim, num_patches^(0.5), num_patches^(0.5))
        x = x.flatten(2)  # 展平成 (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        return x

# 定義位置編碼
class PositionalEncoding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super(PositionalEncoding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))

    def forward(self, x):
        return x + self.pos_embedding

# 定義 Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 多頭自注意力機制
        attn_output, _ = self.attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # 前饋神經網絡
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)
        return x

# 定義 ViT 模型
class VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=8, in_channels=3, num_classes=10, embed_dim=64, num_heads=4, d_ff=128, num_layers=6, dropout=0.1):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.pos_embed = PositionalEncoding(self.patch_embed.num_patches, embed_dim)

        # 定義 Transformer Encoder Layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

        # 分類頭
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        
        # Patch Embedding
        x = self.patch_embed(x)

        # 插入 cls_token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, num_patches+1, embed_dim)

        # 加入位置編碼
        x = self.pos_embed(x)

        # 遍歷 Transformer Encoder Layers
        for layer in self.layers:
            x = layer(x)

        # 使用 cls_token 進行分類
        cls_output = x[:, 0]  # (batch_size, embed_dim)
        output = self.fc(cls_output)  # (batch_size, num_classes)
        return output

# 測試 Vision Transformer
img_size = 32
patch_size = 8
num_classes = 10
batch_size = 2

# 模擬輸入數據 (batch_size, channels, height, width)
x = torch.randn(batch_size, 3, img_size, img_size)

# 定義 ViT 模型並執行前向傳播
vit = VisionTransformer(img_size=img_size, patch_size=patch_size, num_classes=num_classes)
output = vit(x)

print("Vision Transformer 輸出:", output)

```
1. **Patch Embedding**：
    
    - `PatchEmbedding` 將圖像劃分為一系列小塊（Patch）並進行嵌入。通過使用卷積層 (`self.proj`) 模擬切片過程，將每個 Patch 投影到高維特徵空間。
    - `x = self.proj(x)` 將輸入圖像映射到嵌入維度。
2. **位置編碼**：
    
    - `PositionalEncoding` 定義了固定的可學習位置編碼。每個 Patch 在序列中有唯一的位置，因此我們添加了一個位置編碼參數 `self.pos_embedding`，以保持序列順序信息。
3. **Transformer 編碼器層**：
    
    - `TransformerEncoderLayer` 類包含多頭自注意力和前饋神經網絡（FeedForward Neural Network, FFN）。
    - `self.attn` 是 PyTorch 自帶的多頭注意力層。
    - `self.ffn` 是前饋神經網絡，由兩層全連接層組成。
    - `forward` 方法使用殘差連接和 Layer Normalization 確保信號穩定。
4. **Vision Transformer**：
    
    - `VisionTransformer` 類將所有部分結合起來，包括 Patch 嵌入、位置編碼、Transformer 編碼器層和最終的分類頭。
    - `self.cls_token` 是一個特殊的分類 token，代表整個圖像的全局信息，最終輸出會基於 `cls_token`。
    - `forward` 方法中，我們首先將輸入劃分為 Patches，嵌入後添加 `cls_token` 和位置編碼，然後經過多層 Transformer Encoder 層，最後使用 `cls_token` 進行分類。
5. **測試 ViT 模型**：
    
    - 模擬一張 32x32 的彩色圖像輸入（3 個通道），通過 ViT 模型前向傳播，最終輸出每個類別的分數。

此實作展示了如何使用 PyTorch 手寫 Vision Transformer，適合用於面試中的手寫代碼題目。
### 18. 不用pytorch手寫ViT   
```
import numpy as np

# 定義 ReLU 激活函數
def relu(x):
    return np.maximum(0, x)

# 定義 softmax 函數
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Scaled Dot-Product Attention
def scaled_dot_product_attention(query, key, value):
    d_k = query.shape[-1]
    scores = np.dot(query, key.T) / np.sqrt(d_k)
    attention_weights = softmax(scores)
    output = np.dot(attention_weights, value)
    return output

# Multi-Head Attention
class MultiHeadAttention:
    def __init__(self, embed_dim, num_heads):
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.query_weights = np.random.randn(embed_dim, embed_dim)
        self.key_weights = np.random.randn(embed_dim, embed_dim)
        self.value_weights = np.random.randn(embed_dim, embed_dim)
        self.output_weights = np.random.randn(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, num_patches, embed_dim = x.shape
        query = np.dot(x, self.query_weights)
        key = np.dot(x, self.key_weights)
        value = np.dot(x, self.value_weights)

        # 分割多頭
        query = query.reshape(batch_size, num_patches, self.num_heads, self.d_k).swapaxes(1, 2)
        key = key.reshape(batch_size, num_patches, self.num_heads, self.d_k).swapaxes(1, 2)
        value = value.reshape(batch_size, num_patches, self.num_heads, self.d_k).swapaxes(1, 2)

        # 計算注意力
        attention_output = np.array([scaled_dot_product_attention(q, k, v) for q, k, v in zip(query, key, value)])
        attention_output = attention_output.swapaxes(1, 2).reshape(batch_size, num_patches, embed_dim)
        
        # 輸出層
        output = np.dot(attention_output, self.output_weights)
        return output

# 前饋網絡
class FeedForward:
    def __init__(self, embed_dim, hidden_dim):
        self.w1 = np.random.randn(embed_dim, hidden_dim)
        self.w2 = np.random.randn(hidden_dim, embed_dim)

    def forward(self, x):
        return np.dot(relu(np.dot(x, self.w1)), self.w2)

# 位置編碼
def positional_encoding(num_patches, embed_dim):
    pos = np.arange(num_patches)[:, np.newaxis]
    i = np.arange(embed_dim)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / embed_dim)
    pos_encoding = pos * angle_rates
    pos_encoding[:, 0::2] = np.sin(pos_encoding[:, 0::2])
    pos_encoding[:, 1::2] = np.cos(pos_encoding[:, 1::2])
    return pos_encoding

# Transformer 編碼器層
class TransformerEncoderLayer:
    def __init__(self, embed_dim, num_heads, hidden_dim):
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.feed_forward = FeedForward(embed_dim, hidden_dim)

    def forward(self, x):
        # 多頭注意力層
        attn_output = self.self_attn.forward(x)
        x = x + attn_output

        # 前饋層
        ff_output = self.feed_forward.forward(x)
        x = x + ff_output
        return x

# Vision Transformer 類
class VisionTransformer:
    def __init__(self, img_size=32, patch_size=8, in_channels=3, num_classes=10, embed_dim=64, num_heads=4, hidden_dim=128, num_layers=6):
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_weights = np.random.randn(self.patch_dim, embed_dim)
        self.cls_token = np.random.randn(1, embed_dim)
        self.pos_embedding = positional_encoding(self.num_patches + 1, embed_dim)

        # Transformer 編碼器層
        self.layers = [TransformerEncoderLayer(embed_dim, num_heads, hidden_dim) for _ in range(num_layers)]
        self.classifier_weights = np.random.randn(embed_dim, num_classes)

    def patch_embedding(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.num_patches, -1)
        x = np.dot(x, self.patch_weights)
        return x

    def forward(self, x):
        # Patch 嵌入
        x = self.patch_embedding(x)
        
        # 加入 cls_token 和位置編碼
        cls_tokens = np.repeat(self.cls_token, x.shape[0], axis=0).reshape(-1, 1, self.embed_dim)
        x = np.concatenate([cls_tokens, x], axis=1) + self.pos_embedding

        # Transformer 編碼器層
        for layer in self.layers:
            x = layer.forward(x)

        # 分類頭
        cls_output = x[:, 0]  # 選取 cls_token 的輸出
        output = np.dot(cls_output, self.classifier_weights)
        return output

# 測試 Vision Transformer
img_size = 32
patch_size = 8
in_channels = 3
num_classes = 10
batch_size = 2

# 模擬輸入數據
np.random.seed(0)
x = np.random.rand(batch_size, img_size, img_size, in_channels)

# 定義 ViT 並執行前向傳播
vit = VisionTransformer(img_size=img_size, patch_size=patch_size, in_channels=in_channels, num_classes=num_classes)
output = vit.forward(x)

print("Vision Transformer 輸出:", output)

```
1. **Patch 嵌入**：
    
    - `VisionTransformer` 類中的 `patch_embedding` 方法將圖像劃分為小塊（patches），每個 patch 展平成一維向量，然後映射到嵌入空間。
2. **位置編碼**：
    
    - `positional_encoding` 函數為每個 patch 生成位置編碼，用來保留每個 patch 的位置信息。位置編碼被加到每個 patch 的嵌入上，讓模型能夠識別不同位置的 patch。
3. **Scaled Dot-Product Attention**：
    
    - `scaled_dot_product_attention` 函數計算自注意力分數，使用 `scores = QK^T / sqrt(d_k)`，然後通過 softmax 歸一化，得到注意力權重。
4. **多頭注意力**：
    
    - `MultiHeadAttention` 類中，對 `query`、`key` 和 `value` 進行多頭分割並計算每個頭的注意力。最後拼接每個頭的輸出並應用全連接層。
5. **前饋網絡**：
    
    - `FeedForward` 類使用兩層全連接層和 ReLU 激活來實現前饋網絡，增強 Transformer 的表達能力。
6. **Transformer 編碼器層**：
    
    - `TransformerEncoderLayer` 類將多頭注意力和前饋層結合，實現 Transformer 的基本結構。
7. **ViT 模型**：
    
    - `VisionTransformer` 類將圖像劃分成 patches，將 `cls_token` 和位置編碼加入嵌入。數據通過多層 Transformer 編碼器層，最後使用 `cls_token` 的輸出作為分類的特徵。
8. **測試 Vision Transformer**：
    
    - 定義一個 32x32 的彩色圖像輸入（3 個通道），通過 ViT 前向傳播，最終輸出每個類別的分數。

這個 NumPy 實作展示了 Vision Transformer 的基礎結構，適合用於面試中的手寫代碼題目。