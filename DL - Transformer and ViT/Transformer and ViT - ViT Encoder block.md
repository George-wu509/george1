

## **Vision Transformer (ViT) 的 Encoder Block 解析**

在 **Vision Transformer (ViT)** 的 **Encoder Block** 中，流程圖通常會包含：

1. **LayerNorm (LN)**
2. **Multi-Head Self-Attention (MHSA)**
3. **LayerNorm (LN)**
4. **MLP（前饋全連接層，Feedforward Network, FFN）**
5. **殘差連接（Residual Connection）**

這些步驟的設計類似於 **BERT 或 Transformer Encoder** 的結構，但應用在影像上。

---

## **1. Encoder Block 的完整流程**

假設我們有一張影像 XXX：

- **輸入影像尺寸**：(H,W,C)（例如 224 × 224 × 3）
- **將影像切成 P×P 的 Patches**，假設：
    - Patch Size P=16
    - **總 Patch 數量**：$N = \frac{H}{P} \times \frac{W}{P} = 14 \times 14 = 196$
    - **每個 Patch 維度**：$P^2 \times C = 16 \times 16 \times 3 = 768$  (patch dimension應該是可設定參數)

每個 Patch 透過一個線性變換投影到 Transformer 隱藏維度 D（例如 768），因此：

- **輸入 Encoder Block 的特徵矩陣大小**：$X_0 \in \mathbb{R}^{(N+1) \times D}$（額外加了一個 [ CLS ] Token）

---

### **Step 1: LayerNorm (LN)**

**作用**： (normalization 針對單一筆data的所有feature做normalization)

- 在計算 Multi-Head Self-Attention (MHSA) 之前，先做 **LayerNorm** 來穩定數值範圍，提高訓練穩定性。

**數學公式**：

$\Huge \hat{X} = \frac{X - \mu}{\sigma + \epsilon}$​

其中：
- X 是輸入特徵矩陣  (N+1)×D。
- μ,σ 是對每一個 Token（每一行）計算的均值與標準差。
- ϵ 是小常數，防止除零錯誤。

**輸入輸出維度**：

- **輸入**：$X_0 \in \mathbb{R}^{(N+1) \times D}$
- **輸出**：$\hat{X}_0 \in \mathbb{R}^{(N+1) \times D}$

### **為何 Transformer 選擇 LayerNorm 而非 BatchNorm?**

1. **序列長度變化問題**：
    
    - Transformer (特別是 NLP 和 ViT) 的輸入長度不固定，而 BN 需要在固定的 Batch 上計算均值和標準差，導致 **變長序列下效果不佳**。
2. **自注意力機制會擾亂特徵分佈**：
    
    - BN 依賴於 Batch 的統計特性，但 Self-Attention 可能讓不同 Token 之間的關係動態變化，這會影響 BN 的穩定性。
3. **訓練與推理的不一致**：
    
    - BN 在推理時使用 moving average 來估計均值和標準差，這可能與訓練時不同，導致 Transformer 推理效果變差。
4. **LayerNorm 適合變長輸入**：
    
    - LayerNorm 在每個 Token（或 Patch）內計算均值和標準差，因此與 Batch Size 無關，適用於 Transformer。


---

### **Step 2: Multi-Head Self-Attention (MHSA)**

**作用**：

- 讓每個 Patch 透過 Self-Attention 機制與其他 Patch 交互，學習全局關係。

**計算過程**：

1. **計算 Query, Key, Value**

    $\large Q = X W_Q, \quad K = X W_K, \quad V = X W_V$
    
    - 這些權重矩陣 WQ,WK,WV會學習如何將輸入特徵投影到不同的查詢 (Query)、鍵 (Key) 和值 (Value) 空間。
2. **計算 Attention Scores**

    $\large A = \frac{Q K^T}{\sqrt{D}}$​
    
    - 這裡 A 是每個 Patch 對其他 Patch 的注意力權重矩陣，大小為 (N+1)×(N+1)
3. **Softmax 歸一化**
    
    $\large \alpha = \text{softmax}(A)$
    
    - 這樣每個 Patch 會對其他 Patch 產生一組加權係數。
4. **計算加權輸出**
    
    $\large Z = \alpha V$
    
    - 這個步驟讓每個 Patch 獲取來自其他 Patch 的特徵資訊。
5. **多頭機制 (Multi-Head Attention)**
    
    - 在 Transformer 中，通常會用多組 $W_Q, W_K, W_V$​ 計算 **多個不同角度的注意力**，最後 Concatenate 起來： $\large \text{MHSA}(X) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W_O$
    - 這使得 Transformer 可以學習不同層次的特徵關係。


##### **2. Multi-Head Self-Attention (MHSA) 詳細計算**

假設：

- **Patch Size = 4**
- **Patch Dim = 10**
- **總 Patch 數量 N=6N = 6N=6**
- **Transformer Hidden Dimension D=10D = 10D=10**
- **Attention Head 數量 H=2H = 2H=2（每個 Head 的維度是 D/H=5D / H = 5D/H=5）**

### **(1) Q, K, V 的計算**

每個 Patch 都會計算自己的 **Query (Q), Key (K), Value (V)**，透過學習的線性變換：

$\large Q = X W_Q, \quad K = X W_K, \quad V = X W_V$

其中：

- $X \in \mathbb{R}^{6 \times 10}$（6 個 Patches，每個 10 維）
- $W_Q, W_K, W_V \in \mathbb{R}^{10 \times 10}$（學習的權重矩陣）

計算後：

- $Q, K, V \in \mathbb{R}^{6 \times 10}$

**如果是 Multi-Head Attention，則會把 Q, K, V 拆成 2 個 Head：**

- 每個 Head 只會用 D/H=5 維資訊。
- Q,K,V 被 reshape 為 (6,2,5)。

### **(2) 計算 Attention Score**

注意力分數：

$\large A = \frac{Q K^T}{\sqrt{D}}$

假設：

$Q = \begin{bmatrix} Q_1 \\ Q_2 \\ Q_3 \\ Q_4 \\ Q_5 \\ Q_6 \end{bmatrix}, \quad K = \begin{bmatrix} K_1 \\ K_2 \\ K_3 \\ K_4 \\ K_5 \\ K_6 \end{bmatrix}$

則：

$\large A_{ij} = \frac{Q_i \cdot K_j^T}{\sqrt{D}}$

得到 **Attention Score 矩陣** A：

A=$A = \begin{bmatrix} a_{11} & a_{12} & a_{13} & a_{14} & a_{15} & a_{16} \\ a_{21} & a_{22} & a_{23} & a_{24} & a_{25} & a_{26} \\ a_{31} & a_{32} & a_{33} & a_{34} & a_{35} & a_{36} \\ a_{41} & a_{42} & a_{43} & a_{44} & a_{45} & a_{46} \\ a_{51} & a_{52} & a_{53} & a_{54} & a_{55} & a_{56} \\ a_{61} & a_{62} & a_{63} & a_{64} & a_{65} & a_{66} \\ \end{bmatrix}$

然後透過 **Softmax**：

$\alpha = \text{softmax}(A)$

這樣每個 Patch 會分配不同的權重給其他 Patch。

### **(3) 計算輸出**

Z=αV

這樣，每個 Patch 的輸出 ZZZ 就是所有 Patch **根據注意力權重加權求和的結果**。


---

### **Step 3: 殘差連接 (Residual Connection)**

**作用**：

- 讓輸入訊息直接繞過 Attention 層，使得原始資訊不會被完全改變： X1=Z+X0X_1 = Z + X_0X1​=Z+X0​
- 這樣可以解決深度網路中的梯度消失問題，提高模型的可訓練性。

**輸入輸出維度**：

- **輸入**：Z∈R(N+1)×DZ \in \mathbb{R}^{(N+1) \times D}Z∈R(N+1)×D
- **輸出**：X1∈R(N+1)×DX_1 \in \mathbb{R}^{(N+1) \times D}X1​∈R(N+1)×D

---

### **Step 4: LayerNorm (LN)**

**作用**：

- 在送入 MLP 之前再次做 **LayerNorm**，讓數據保持穩定。

**輸入輸出維度**：

- **輸入**：X1∈R(N+1)×DX_1 \in \mathbb{R}^{(N+1) \times D}X1​∈R(N+1)×D
- **輸出**：X^1∈R(N+1)×D\hat{X}_1 \in \mathbb{R}^{(N+1) \times D}X^1​∈R(N+1)×D

---

### **Step 5: MLP（前饋全連接層，Feedforward Network, FFN）**

**作用**：

- 進一步學習高層次的 Patch 特徵，通常是兩層全連接層： X2=ReLU(XW1+b1)W2+b2X_2 = \text{ReLU}(X W_1 + b_1) W_2 + b_2X2​=ReLU(XW1​+b1​)W2​+b2​
- 其中：
    - W1∈RD×4DW_1 \in \mathbb{R}^{D \times 4D}W1​∈RD×4D（擴展 4 倍）
    - W2∈R4D×DW_2 \in \mathbb{R}^{4D \times D}W2​∈R4D×D（壓縮回 D 維）
- 這樣的結構能夠提高 Transformer 表達能力。

**輸入輸出維度**：

- **輸入**：X^1∈R(N+1)×D\hat{X}_1 \in \mathbb{R}^{(N+1) \times D}X^1​∈R(N+1)×D
- **輸出**：X3∈R(N+1)×DX_3 \in \mathbb{R}^{(N+1) \times D}X3​∈R(N+1)×D

---

### **Step 6: 最終殘差連接**

Xout=X3+X1X_{\text{out}} = X_3 + X_1Xout​=X3​+X1​

這讓 Transformer Encoder Block 具有更好的梯度流動。

**輸入輸出維度**：

- **輸入**：X3∈R(N+1)×DX_3 \in \mathbb{R}^{(N+1) \times D}X3​∈R(N+1)×D
- **輸出**：Xout∈R(N+1)×DX_{\text{out}} \in \mathbb{R}^{(N+1) \times D}Xout​∈R(N+1)×D

---

## **完整輸入輸出維度對照**

|**步驟**|**輸入維度**|**輸出維度**|
|---|---|---|
|LayerNorm|(N+1)×D(N+1) \times D(N+1)×D|(N+1)×D(N+1) \times D(N+1)×D|
|Multi-Head Self-Attention|(N+1)×D(N+1) \times D(N+1)×D|(N+1)×D(N+1) \times D(N+1)×D|
|殘差連接|(N+1)×D(N+1) \times D(N+1)×D|(N+1)×D(N+1) \times D(N+1)×D|
|LayerNorm|(N+1)×D(N+1) \times D(N+1)×D|(N+1)×D(N+1) \times D(N+1)×D|
|MLP|(N+1)×D(N+1) \times D(N+1)×D|(N+1)×D(N+1) \times D(N+1)×D|
|殘差連接|(N+1)×D(N+1) \times D(N+1)×D|(N+1)×D(N+1) \times D(N+1)×D|

---

## **總結**

ViT 的 **Encoder Block** 核心包含：

1. **LayerNorm**（數據穩定化）
2. **Self-Attention**（學習全局關係）
3. **MLP**（學習高維特徵）
4. **Residual Connection**（保持梯度流動）

這種結構讓 ViT 在影像理解上能夠學習豐富的全局資訊，優於傳統 CNN！