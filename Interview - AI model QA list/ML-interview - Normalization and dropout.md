
在深度學習網絡中，常用的 **Normalization（正規化）技術** 主要用來加速收斂、提升泛化能力，並穩定網絡訓練。常見的 Normalization 方法有：

---

## 1. 常見的 Normalization 技術

- **Batch Normalization (BN)** 適用於**不同特徵 (features) 之間的數值範圍差異較大**的情況，主要用來解決深度神經網路中的**內部協變數轉移 (Internal Covariate Shift)**，透過對 mini-batch 內的特徵進行標準化來加速訓練並提升穩定性。
- **Layer Normalization (LN)** 適用於**單筆輸入資料內部數值範圍變化較大**的情況，特別適合 RNN 和 Transformer 等場景，因為它對 batch size 不敏感，適用於不同序列長度的輸入。

| 比較項目                | **Batch Normalization (BN)**                                  | **Layer Normalization (LN)**                          |
| ------------------- | ------------------------------------------------------------- | ----------------------------------------------------- |
| **標準化維度**           | 在 **batch 維度** 上做標準化，即對相同特徵（feature）在整個 mini-batch 內的數值進行標準化。 | 在 **單筆輸入的特徵維度** 上做標準化，即針對單個樣本內的所有特徵進行標準化。             |
| **計算方式**            | 計算 mini-batch 中該特徵的均值與標準差，然後標準化。                              | 計算單個樣本的所有特徵均值與標準差，然後標準化。                              |
| **適用場景**            | 適用於 CNN、DNN 等場景，特別是在 batch size 大時效果較好。                       | 適用於 RNN、Transformer、強化學習等場景，特別適用於 batch size 小或變動的情況。 |
| **對 batch size 影響** | 需要較大的 batch size 來獲得穩定的均值與標準差。batch size 過小時效果較差。             | 不受 batch size 影響，適用於小 batch 或 batch size 為 1 的情況。     |
| **與時間序列的關係**        | 在 RNN 類型的模型中表現較差，因為每個時間步的特徵是獨立標準化的，會破壞時間序列關係。                 | 對於時間序列、RNN 更適合，因為它對單筆數據進行標準化，不會影響不同 batch 間的計算。       |
| **計算開銷**            | 需要計算 batch 內的均值與變異數，在推理時需要使用移動平均值，因此會帶來一定的計算開銷。               | 計算均值與變異數僅涉及單個樣本，計算開銷較小，適用於即時推理場景。                     |
| **移動設備適用性**         | 計算涉及整個 batch，在變動的 batch size 下可能不穩定，不適合移動設備。                  | 計算僅基於單筆樣本，不受 batch size 影響，更適合移動設備與在線推理。              |

### **(1) Batch Normalization (BN)**

- **原理**：對 **同一 mini-batch 內** 的數據，在每個通道 (channel) 上計算均值和方差，然後標準化。

- **優勢**：
    - **加速訓練**：減少 internal covariate shift（即中間特徵分佈變化過大）
    - **提升穩定性**：防止梯度爆炸或梯度消失
    - **降低對學習率的敏感度**
- **缺點**：
    - **對小 batch size 影響較大**（因為均值和方差估計不準確）
    - **計算開銷大**，推理時需要額外處理均值和方差
- **應用場景**：適用於 **CNN、RNN、Transformer**，但在 RNN 中效果一般（時間序列難以 batch）

舉例:
X = [ [ 1, 2, 3 ], [ 4, 5, 6 ] ], 有兩個sample A, B, 各是三個feature f1,f2,f3:  
A sample: [ 1, 2, 3 ],  B sample: [ 4, 5, 6 ]
![[Pasted image 20250304153405.png]]
#### 如果用**Batch normalization**:
針對不同的feature f1,f2,f3做normalization (也針對训练时一个 mini-batch)
f1 mean= (1+4)/2 = 2.5,   f1 std  = ((1-2.5)^2+(4-2.5)^2)/2 = 2.25
f2 mean = (2+5)/2 = 3.5   f2 std=2.25
f3 mean = (3+6)/2 = 4.5   f3 std=2.25
After normalization--> [ [ -1, -1, -1 ], [ 1, 1, 1 ] ]
- **優勢**：
    - **加速訓練**：減少 internal covariate shift（即中間特徵分佈變化過大）
    - **提升穩定性**：防止梯度爆炸或梯度消失
    - **降低對學習率的敏感度**
- **缺點**：
    - **對小 batch size 影響較大**（因為均值和方差估計不準確）
    - **計算開銷大**，推理時需要額外處理均值和方差
- **應用場景**：適用於 **CNN、RNN、Transformer**，但在 RNN 中效果一般（時間序列難以 batch）

![[Pasted image 20250304153502.png]]
#### 如果用**Layer normalization**:
針對不同的Sample A, B做normalization 
A mean= (1+2+3)/3 = 2,   A std  = ((1-2)^2+(2-2)^2+(3-2)^2)/3 = 2/3
B mean= (4+5+6)/3 = 5,   B std  = ((4-5)^2+(5-5)^2+(6-5)^2)/3 = 2/3
After normalization--> [ [ -1.22, 0, 1.22 ], [ -1.22, 0, 1.22 ] ]
- **優勢**：
    - **適合 RNN / Transformer**，因為不依賴 batch size
    - **與 batch size 無關**，適合小 batch 訓練
- **缺點**：
    - 在 CNN 可能效果較差，因為不同通道的特徵可能意義不同
- **應用場景**：
    - 主要用於 **RNN（LSTM, GRU）**，Transformer（如 BERT, GPT）


BatchNorm就是通过对batch size这个维度归一化来让分布稳定下来。LayerNorm则是通过对Hidden size这个维度归一化来让某层的分布稳定。


**4.1 理解上**

<mark style="background: #BBFABBA6;">BatchNorm是对一个batch-size样本内的每个特征做归一化，LayerNorm是对每个样本的所有特征做归一化。</mark>BN 的转换是针对单个神经元可训练的：不同神经元的输入经过再平移和再缩放后分布在不同的区间；而 LN 对于一整层的神经元训练得到同一个转换：所有的输入都在同一个区间范围内。如果不同输入特征不属于相似的类别（比如颜色和大小），那么 LN 的处理可能会降低模型的表达能力。

BN抹杀了不同特征之间的大小关系，但是保留了不同样本间的大小关系；LN抹杀了不同样本间的大小关系，但是保留了一个样本内不同特征之间的大小关系。（理解：BN对batch数据的同一特征进行标准化，变换之后，纵向来看，不同样本的同一特征仍然保留了之前的大小关系，但是横向对比样本内部的各个特征之间的大小关系不一定和变换之前一样了，因此抹杀或破坏了不同特征之间的大小关系，保留了不同样本之间的大小关系；LN对单一样本进行标准化，样本内的特征处理后原来数值大的还是相对较大，原来数值小的还是相对较小，不同特征之间的大小关系还是保留了下来，但是不同样本在各自标准化处理之后，两个样本对应位置的特征之间的大小关系将不再确定，可能和处理之前就不一样了，所以破坏了不同样本间的大小关系）

**4.2 使用场景上**

在BN和LN都能使用的场景中，BN的效果一般优于LN，原因是基于不同数据，同一特征得到的归一化特征更不容易损失信息。但是有些场景是不能使用BN的，例如batch size较小或者序列问题中可以使用LN。这也就解答了**RNN 或Transformer为什么用Layer Normalization？**

**首先**RNN或Transformer解决的是序列问题，一个存在的问题是不同样本的序列长度不一致，而Batch Normalization需要对不同样本的同一位置特征进行标准化处理，所以无法应用；当然，输入的序列都要做padding补齐操作，但是补齐的位置填充的都是0，这些位置都是无意义的，此时的标准化也就没有意义了。

**其次**上面说到，BN抹杀了不同特征之间的大小关系；LN是保留了一个样本内不同特征之间的大小关系，这对[NLP](https://zhida.zhihu.com/search?content_id=232088974&content_type=Article&match_order=1&q=NLP&zhida_source=entity)任务是至关重要的。对于NLP或者序列任务来说，一条样本的不同特征，其实就是时序上的变化，这正是需要学习的东西自然不能做归一化抹杀，所以要用LN。

**4.3 训练和预测时有无区别：**

LN针对的是单独一个样本，在训练和预测阶段的使用并无差别；BN是针对一个batch进行计算的，训练时自然可以根据batch计算，但是预测时有时要预测的是单个样本，此时要么认为batch size就是1，不进行标准化处理，要么是在训练时记录标准化操作的均值和方差直接应用到预测数据，这两种解决方案都不是很完美，都会存在偏差。
https://zhuanlan.zhihu.com/p/647813604

https://zhuanlan.zhihu.com/p/696062068

![[Pasted image 20250304150551.png]]
---

### **(2) Layer Normalization (LN)**

- **原理**：對 **同一樣本內** 的所有特徵 (即神經元輸出) 計算均值和方差，而不是針對 mini-batch。

- **優勢**：
    - **適合 RNN / Transformer**，因為不依賴 batch size
    - **與 batch size 無關**，適合小 batch 訓練
- **缺點**：
    - 在 CNN 可能效果較差，因為不同通道的特徵可能意義不同
- **應用場景**：
    - 主要用於 **RNN（LSTM, GRU）**，Transformer（如 BERT, GPT）

---

### **(3) Instance Normalization (IN)**

- **原理**：對 **每個樣本的每個通道單獨** 計算均值和方差，主要用於圖像風格轉換。
- **應用場景**：
    - 風格遷移（Style Transfer）
    - 圖像生成模型（GANs）

---

### **(4) Group Normalization (GN)**

- **原理**：把通道分成 G 個 group，每個 group 計算均值和方差。
- **優勢**：
    - 適合 **小 batch size**（因為 BN 依賴 batch size）
- **應用場景**：
    - 小 batch 訓練（目標檢測、醫學影像）

---

### **(5) Weight Normalization**

- **原理**：對神經網絡的權重進行正規化，提高訓練穩定性。





## 2. 避免 Overfitting 的技術

除了 Normalization，還有一些常見方法來 **防止過擬合 (Overfitting)**：

### **(1) Dropout**

- **原理**：在訓練過程中，隨機將某些神經元的輸出設為 **0**（即丟棄），從而避免過擬合。
- **公式**： yi=xi⋅Bernoulli(p)y_i = x_i \cdot \text{Bernoulli}(p)yi​=xi​⋅Bernoulli(p) 其中 ppp 是保留概率（例如 0.5），測試時則乘 ppp 來保持期望一致。
- **應用場景**：
    - **全連接層 (Fully Connected, FC)**
    - **CNN 深層結構**
    - **RNN（但效果一般）**

---

### **(2) L1/L2 正則化 (Weight Decay)**

- **L1 正則化**：
    - 加入 λ∑∣w∣\lambda \sum |w|λ∑∣w∣ 的懲罰項，使部分權重趨近於 0（稀疏化）
    - 適用於 **特徵選擇**
- **L2 正則化 (Ridge Regression)**
    - 加入 λ∑w2\lambda \sum w^2λ∑w2 的懲罰項，使權重變小
    - 適用於 **深度學習，與 Adam, SGD 結合**
- **應用場景**：
    - 幾乎所有深度學習模型（CNN、RNN、Transformer）

---

### **(3) Data Augmentation**

- **原理**：對訓練數據進行隨機變換（如旋轉、翻轉、裁剪、噪聲等）來擴充數據量，提高模型泛化能力。
- **應用場景**：
    - **計算機視覺（CV）**
    - **語音處理**

---

### **(4) Early Stopping**

- **原理**：監測驗證集的 Loss 或 Accuracy，當表現下降時停止訓練，防止過擬合。
- **應用場景**：
    - 幾乎所有深度學習訓練

---

### **(5) Transfer Learning**

- **原理**：使用預訓練模型來進行遷移學習，減少過擬合風險。
- **應用場景**：
    - CV（如 ResNet, ViT, DINOv2）
    - NLP（如 BERT, GPT）

---

## 3. 結論

- **Normalization 技術**（BN, LN, GN, IN）主要用來加速訓練、提升穩定性。
- **避免 Overfitting 技術**（Dropout, L1/L2, Data Augmentation）主要用來提升泛化能力。
- **不同場景適合不同技術**，如：
    - **CNN：BatchNorm、GroupNorm**
    - **RNN/Transformer：LayerNorm**
    - **小 batch 訓練：GroupNorm, LayerNorm**
    - **風格轉換：InstanceNorm**
    - **小數據：Data Augmentation、Transfer Learning**