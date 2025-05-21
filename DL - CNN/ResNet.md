
詳細介紹一下 ResNet (Residual Network) 的特點。ResNet 是一種在深度學習領域具有里程碑意義的卷積神經網路 (CNN) 架構，它有效地解決了在訓練非常深的網路時出現的<mark style="background: #ABF7F7A6;">梯度消失 (vanishing gradient) 和網路退化 (degradation problem)</mark> 問題，從而使得訓練數百甚至數千層的深度網路成為可能。
殘差塊 (Residual Block)可分成兩種: 
1. <mark style="background: #BBFABBA6;">基本殘差塊</mark> (Basic Block)用於較淺的ResNet (例如 ResNet-18 和 ResNet-34)。它通常由兩個 3x3 的卷積層組成，每個卷積層後跟批量歸一化 (Batch Normalization) 和 ReLU 激活函數.
2. <mark style="background: #BBFABBA6;">瓶頸殘差塊</mark> (Bottleneck Block):** 用於更深的 ResNet (例如 ResNet-50 及以上)。它包含三個卷積層：一個 1x1 的卷積層用於降維 (減少通道數)，一個 3x3 的卷積層用於提取空間特徵，以及另一個 1x1 的卷積層用於升維 (恢復通道數)

|                   |     |
| ----------------- | --- |
| [[##### QA-list]] |     |

以下是 ResNet 的主要特點：

**1. 殘差塊 (Residual Block): 核心創新**

- **跳躍連接 (Skip Connection / Shortcut Connection):** 這是 ResNet 最核心的創新。在傳統的深層網路中，每一層都直接將其輸出傳遞給下一層。而在 ResNet 中，<mark style="background: #FFF3A3A6;">殘差塊引入了“跳躍連接(skip connection)”，允許將前面某一層 (或幾層) 的激活值直接添加到後面的層的輸出上</mark>。
    
    數學表示一個殘差塊的輸出 H(x) 通常是： H(x)=F(x)+x 其中，x 是輸入到該殘差塊的激活值，F(x) 是該殘差塊中一系列卷積層等操作的輸出 (稱為“殘差映射” (residual mapping))。+x 就是跳躍連接所添加的輸入。
    
- **恆等映射 (Identity Mapping):** 最簡單的跳躍連接是“恆等映射”，即直接將輸入 x 加到輸出 F(x) 上，沒有經過任何額外的操作。這樣做的好處是，如果網路的某一層學到的最佳操作是恆等映射 (即什麼都不做)，那麼跳躍連接可以很容易地實現這一點，而不會引入額外的參數或複雜性。
    
- **解決梯度消失(Vanishing gradient)問題:** <mark style="background: #BBFABBA6;">當網路很深時，梯度在反向傳播過程中可能會逐漸衰減，導致淺層的權重難以更新</mark>。跳躍連接提供了一條額外的梯度傳播路徑，梯度可以直接跳過中間的層傳播到更淺的層，從而有效地緩解了梯度消失的問題，使得訓練更深的網路成為可能。
    
- **解決網路退化(degradation problem)問題:** 研究發現，隨著網路深度的增加，模型的準確率會先上升然後下降，這並不是由過擬合引起的，<mark style="background: #BBFABBA6;">而是因為深層網路難以學習到有效的映射</mark>。殘差塊通過學習殘差映射 F(x)=H(x)−x，而不是直接學習複雜的映射 H(x)，使得網路更容易學習到逼近恆等映射的函數。如果深層網路的某些層是冗餘的，那麼學習 F(x)≈0 比學習 H(x)≈x 要容易得多。
    

**2. 更深的網路架構:**

- 基於殘差塊的設計，ResNet 可以構建非常深的網路，例如 ResNet-18、ResNet-34、ResNet-50、ResNet-101、ResNet-152，甚至更深的網路 (例如 ResNet-1202)。這些數字表示網路的總層數 (通常只計算卷積層和全連接層)。
- 更深的網路通常具有更強的特徵提取能力，可以捕捉到更複雜的數據模式，從而在許多視覺任務上取得了顯著的性能提升。

![[Resnet_block.jpg]]

Residual block:    3x3 - relu - 3x3
Bottlenet block:   1x1 - relu - 3x3 - relu - 1x1


**3. 不同的殘差塊變體:**

- **基本殘差塊 (Basic Block):** <mark style="background: #FFF3A3A6;">用於較淺的 ResNet (例如 ResNet-18 和 ResNet-34)。它通常由兩個 3x3 的卷積層組成，每個卷積層後跟批量歸一化 (Batch Normalization) 和 ReLU 激活函數</mark>。跳躍連接直接將輸入添加到第二個 ReLU 激活函數之後。
    
    ```
    Input -> Conv2D -> BN -> ReLU -> Conv2D -> BN -> + (Input) -> ReLU -> Output
    ```
    
- **瓶頸殘差塊 (Bottleneck Block):** 用於更深的 ResNet (例如 ResNet-50 及以上)。<mark style="background: #FFB86CA6;">它包含三個卷積層：一個 1x1 的卷積層用於降維 (減少通道數)，一個 3x3 的卷積層用於提取空間特徵，以及另一個 1x1 的卷積層用於升維</mark> (恢復通道數)。這樣做的目的是減少 3x3 卷積層的計算量。跳躍連接在降維之前的輸入和升維之後的輸出之間進行。
    
    ```
    Input -> Conv2D (1x1, reduce channels) -> BN -> ReLU ->
             Conv2D (3x3) -> BN -> ReLU ->
             Conv2D (1x1, increase channels) -> BN -> + (Input, possibly with 1x1 conv) -> ReLU -> Output
    ```
    
    注意：在瓶頸殘差塊中，如果輸入和輸出的通道數不匹配 (例如在不同階段的 ResNet 中)，跳躍連接通常會通過一個 1x1 的卷積層進行通道數調整，以確保可以進行元素級的加法。
    

**4. 網路結構的組織:**

- ResNet 通常由一系列的殘差塊堆疊而成。整個網路的結構通常遵循一定的模式，例如：
    - 一個初始的卷積層和池化層。
    - 幾個階段 (stage)，每個階段包含若干個相同類型的殘差塊 (基本塊或瓶頸塊)。在每個階段的開始，通常會使用一個步長大於 1 的卷積層 (在基本塊中) 或瓶頸塊中的第一個 1x1 卷積層 (並調整跳躍連接) 來進行下採樣，從而減小特徵圖的尺寸並增加通道數。
    - 一個<mark style="background: #FFF3A3A6;">全局平均池化層 (Global Average Pooling) </mark>將最後的特徵圖轉換為固定長度的向量。
    - 一個<mark style="background: #FFF3A3A6;">全連接層 (Fully Connected Layer)</mark> 用於最終的分類輸出。

**5. 批量歸一化 (Batch Normalization):**

- ResNet 的每個卷積層之後通常都會跟隨一個批量歸一化層。批量歸一化有助於加速訓練，穩定梯度，並提高模型的泛化能力。

**總結 ResNet 的主要特點:**

- **殘差塊和跳躍連接:** 允許構建非常深的網路並解決梯度消失和網路退化問題。
- **恆等映射:** 簡化了網路學習恆等函數的過程。
- **更深的網路架構:** 能夠提取更複雜的特徵。
- **瓶頸結構:** 在更深的網路中減少計算量。
- **系統性的網路組織:** 由多個階段的殘差塊堆疊而成。
- **廣泛使用批量歸一化:** 提高訓練效率和模型性能。

ResNet 的提出是深度學習領域的一個重要突破，它不僅在圖像分類任務上取得了巨大的成功，而且其核心思想 (例如跳躍連接) 也被廣泛應用於後來的許多其他深度網路架構中，例如 Transformer、DenseNet 等。理解 ResNet 的特點對於深入學習現代深度學習模型至關重要。






#### ResNet-50 架構概述

ResNet-50 是一個50層的深度殘差網絡,主要由Bottleneck blocks組成。它的整體架構如下:

1. 一個7x7的卷積層
2. 一個最大池化層
3. 4組殘差層(每組包含多個Bottleneck blocks)
4. 一個全局平均池化層
5. 一個全連接層

## Residual Block 解釋

殘差塊(Residual Block)是ResNet的核心組件。它的主要思想是:

1. 讓網絡學習殘差函數F(x) = H(x) - x,而不是直接學習H(x)。
2. 通過skip connection(跳躍連接)將輸入直接加到輸出上。

這樣做的好處是:

1. 緩解了深度網絡的梯度消失問題。
2. 使得網絡更容易學習恆等映射,有助於網絡的訓練。

## Skip Connections 解釋

Skip connections(跳躍連接)是將輸入直接加到某一層的輸出上的連接。在ResNet中,它將輸入x直接加到F(x)上,形成H(x) = F(x) + x。Skip connections的優點:

1. 允許梯度直接流回早期層,緩解梯度消失問題。
2. 使網絡能夠輕鬆學習恆等函數,有助於訓練非常深的網絡。

## Bottleneck Block 解釋

Bottleneck block是ResNet-50及更深層次ResNet使用的一種特殊的殘差塊。它的結構是:

1. 1x1卷積降維
2. 3x3卷積
3. 1x1卷積升維

使用Bottleneck block的原因:

1. 減少參數量和計算量,使得訓練更深的網絡成為可能。
2. 1x1卷積可以有效地改變通道數,實現降維和升維。

這個實現包含了以下幾個關鍵點:

1. `Bottleneck` 類實現了Bottleneck block,包含三個卷積層和skip connection。
2. `ResNet` 類實現了整個ResNet架構,包括初始卷積層、四個殘差層組、全局平均池化和全連接層。
3. `_make_layer` 方法用於創建包含多個Bottleneck blocks的層。
4. `forward` 方法定義了數據在網絡中的前向傳播路徑。
5. `resnet50()` 函數返回一個配置為50層的ResNet模型。

通過這種方式,我們實現了一個完整的ResNet-50模型,其中包含了Residual Block、Skip Connections和Bottleneck結構。這種設計使得ResNet-50能夠在保持較低計算複雜度的同時,實現深層網絡的有效訓練和優秀性能


### Q: resnet的網絡學習殘差函數是在resnet training跟inference時都會作用到嗎?請舉例residual layer在training跟inference時的每個輸入輸出跟gradient運算並中文詳細解釋

是的,ResNet的殘差學習機制在訓練(training)和推理(inference)階段都會發揮作用。讓我們通過一個具體的例子來詳細解釋殘差層在這兩個階段的運作過程:假設我們有一個簡單的殘差層,其中包含兩個卷積層和一個跳躍連接(skip connection)。訓練階段:

1. 輸入: 假設輸入張量 x 的形狀為 [64, 256, 56, 56] (批次大小, 通道數, 高度, 寬度)
2. 前向傳播:
    
    - 主路徑: x 經過兩個卷積層,得到 F(x)
    - 跳躍連接: x 直接傳遞
    - 輸出: y = F(x) + x
    
3. 反向傳播:
    
    - 計算損失函數對輸出 y 的梯度: ∂L/∂y
    - 梯度分流:
        
        - 主路徑梯度: ∂L/∂F(x) = ∂L/∂y
        - 跳躍連接梯度: ∂L/∂x = ∂L/∂y + ∂L/∂F(x) * ∂F(x)/∂x
        
    
    這裡的關鍵是,跳躍連接允許梯度直接流回早期層,緩解了梯度消失問題。

推理階段:

1. 輸入: 同訓練階段
2. 前向傳播:
    
    - 主路徑: x 經過兩個卷積層,得到 F(x)
    - 跳躍連接: x 直接傳遞
    - 輸出: y = F(x) + x
    
3. 沒有反向傳播

在兩個階段中,殘差學習的核心思想都得到了體現:

1. 網絡學習殘差函數 F(x),而不是直接學習目標映射 H(x)。
2. 跳躍連接使得網絡能夠輕鬆學習恆等映射(當 F(x) ≈ 0 時)。
3. 在訓練階段,殘差結構有助於梯度的流動,使深層網絡更容易訓練。
4. 在推理階段,殘差結構允許信息更直接地從淺層傳遞到深層,保留了更多的原始特徵。

這種設計使得 ResNet 能夠訓練非常深的網絡,同時在推理時保持高效的特徵提取和傳遞。殘差學習不僅幫助網絡更容易學習複雜的函數,還在整個網絡中保持了強大的特徵表示能力。

### 1. 殘差塊的基本結構

假設一個簡單的**殘差塊（Residual Block）**結構如下：

y=F(x)+xy = F(x) + xy=F(x)+x

其中：

- xxx 是殘差塊的輸入。
- F(x)F(x)F(x) 是卷積層和非線性激活（ReLU）層的組合輸出，表示網絡學習的殘差函數。
- yyy 是殘差塊的最終輸出。

在這個結構中，F(x)F(x)F(x) 是通過兩層卷積操作和ReLU激活得到的：

1. 第一層：卷積 + 批量歸一化（Batch Normalization, BN） + ReLU激活
2. 第二層：卷積 + 批量歸一化（無ReLU）

### 2. 訓練（Training）階段的運算步驟

在訓練過程中，殘差塊會進行前向傳播和反向傳播，以學習權重的梯度。以下是訓練時的詳細運算步驟：

#### 前向傳播（Forward Pass）

假設：

- 輸入 x=x0x = x_0x=x0​
- 卷積核的權重為 W1W_1W1​ 和 W2W_2W2​
- BN層的縮放係數為 γ1,γ2\gamma_1, \gamma_2γ1​,γ2​ 和偏移量 β1,β2\beta_1, \beta_2β1​,β2​

前向傳播過程如下：

1. **第一層卷積**：x1=W1∗x0+b1x_1 = W_1 * x_0 + b_1x1​=W1​∗x0​+b1​
2. **第一層BN和ReLU**：x2=ReLU(γ1⋅BN(x1)+β1)x_2 = \text{ReLU}(\gamma_1 \cdot \text{BN}(x_1) + \beta_1)x2​=ReLU(γ1​⋅BN(x1​)+β1​)
3. **第二層卷積**：x3=W2∗x2+b2x_3 = W_2 * x_2 + b_2x3​=W2​∗x2​+b2​
4. **第二層BN**：F(x)=γ2⋅BN(x3)+β2F(x) = \gamma_2 \cdot \text{BN}(x_3) + \beta_2F(x)=γ2​⋅BN(x3​)+β2​
5. **殘差塊輸出**：y=F(x)+x0y = F(x) + x_0y=F(x)+x0​

#### 反向傳播（Backward Pass）

在反向傳播中，通過計算損失對輸出的偏導數，將梯度逐層傳遞回去，以更新權重。

假設損失函數為 LLL，那麼：

1. **殘差加法的梯度計算**：
    
    - 對於輸出 y=F(x)+x0y = F(x) + x_0y=F(x)+x0​，其梯度為： ∂L∂y=∂L∂F(x)+∂L∂x0\frac{\partial L}{\partial y} = \frac{\partial L}{\partial F(x)} + \frac{\partial L}{\partial x_0}∂y∂L​=∂F(x)∂L​+∂x0​∂L​
    - 這裡梯度直接分別傳遞到 F(x)F(x)F(x) 和 x0x_0x0​ 的分支，這樣即使在很深的網絡中，梯度仍然能夠直接傳到輸入 x0x_0x0​。
2. **反向傳播到第二層卷積**：
    
    - 對於 F(x)F(x)F(x) 的梯度 ∂L∂F(x)\frac{\partial L}{\partial F(x)}∂F(x)∂L​，將反向傳播到 x3x_3x3​： ∂L∂x3=∂L∂F(x)⋅γ2\frac{\partial L}{\partial x_3} = \frac{\partial L}{\partial F(x)} \cdot \gamma_2∂x3​∂L​=∂F(x)∂L​⋅γ2​
    - 然後繼續向前傳播以更新 W2W_2W2​ 的權重。
3. **反向傳播到第一層卷積**：
    
    - 在此過程中，對於每層權重 W1W_1W1​ 和 W2W_2W2​ 進行權重更新，最終完成殘差函數的學習。

---

### 3. 推理（Inference）階段的運算步驟

在推理階段，由於不再進行反向傳播，只需進行前向傳播即可。在這一過程中，BN層的均值和方差會使用訓練中得到的移動平均值，不再依賴於當前批次的數據。因此，輸入輸出步驟與訓練過程的前向傳播部分一致：

1. **第一層卷積**：x1=W1∗x0+b1x_1 = W_1 * x_0 + b_1x1​=W1​∗x0​+b1​
2. **第一層BN和ReLU**：x2=ReLU(γ1⋅BN(x1)+β1)x_2 = \text{ReLU}(\gamma_1 \cdot \text{BN}(x_1) + \beta_1)x2​=ReLU(γ1​⋅BN(x1​)+β1​)
3. **第二層卷積**：x3=W2∗x2+b2x_3 = W_2 * x_2 + b_2x3​=W2​∗x2​+b2​
4. **第二層BN**：F(x)=γ2⋅BN(x3)+β2F(x) = \gamma_2 \cdot \text{BN}(x_3) + \beta_2F(x)=γ2​⋅BN(x3​)+β2​
5. **殘差塊輸出**：y=F(x)+x0y = F(x) + x_0y=F(x)+x0​

在這裡，BN層不再計算每批數據的均值和方差，而是使用訓練階段的移動平均值，從而在推理過程中提供穩定的輸出。

---

### 4. 舉例說明輸入、輸出和梯度運算

假設：

- 初始輸入 x0=1.0x_0 = 1.0x0​=1.0
- 卷積核 W1,W2W_1, W_2W1​,W2​ 初始化為隨機值
- 損失函數 LLL 對於輸出 yyy 的梯度為1（假設）

在訓練階段的反向傳播中：

1. **殘差梯度傳遞**：
    
    - ∂L∂y=1\frac{\partial L}{\partial y} = 1∂y∂L​=1
    - 對應於 F(x)F(x)F(x) 和 x0x_0x0​ 的梯度分別為 ∂L∂F(x)=1\frac{\partial L}{\partial F(x)} = 1∂F(x)∂L​=1 和 ∂L∂x0=1\frac{\partial L}{\partial x_0} = 1∂x0​∂L​=1。
2. **反向傳播至第二層卷積**：
    
    - 計算 ∂L∂x3\frac{\partial L}{\partial x_3}∂x3​∂L​ 並更新 W2W_2W2​。
3. **反向傳播至第一層卷積**：
    
    - 計算 ∂L∂x1\frac{\partial L}{\partial x_1}∂x1​∂L​ 並更新 W1W_1W1​。

在推理階段，由於不涉及梯度計算，直接依賴已學習的 W1W_1W1​、W2W_2W2​、γ\gammaγ、β\betaβ 參數進行前向傳播。

---

總結來說，**ResNet在訓練和推理階段都會使用殘差結構**。在訓練階段，殘差函數的學習涉及梯度計算和反向傳播；在推理階段，則使用已學習的參數進行推理，使得殘差結構能夠更高效地提取特徵，減少梯度消失的問題，特別適合訓練深層網絡。

##### QA-list

| Q                                                 | Ans |
| ------------------------------------------------- | --- |
| shortcut-connect(使用残差网络Residual network)          |     |
| resnet相比于之前的卷积神经网络模型中，最大的改进点是什么？，解决了什么问题          |     |
| Resnet为啥能解决梯度消失，怎么做的，能推导吗？                        |     |
| resnet第二个版本做了哪些改进，Resnet性能最好的变体是哪个，结构是怎么样的，原理是什么？ |     |
| DenseNet为什么比ResNet有更强的表达能力？                       |     |
