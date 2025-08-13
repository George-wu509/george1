
### CenterMask2 網路架構詳細介紹

**CenterMask2** 是一個基於 **Anchor-Free** 的即時實例分割（Instance Segmentation）網絡，從 **RetinaNet** 的架構演進而來。它融合了 **<mark style="background: #FFB86CA6;">FCOS</mark>（Fully Convolutional One-Stage Object Detection）** 的物體檢測能力，並在此基礎上添加了實例分割的功能。它針對實時性進行了優化，相比 **Mask R-CNN** 有更好的速度表現。

![[Pasted image 20250521155837.png]]

CenterMask2 是一個基於 CenterMask 的實時無錨點（Anchor-Free）實例分割（Instance Segmentation）模型。它是 CenterMask 的升級實現，通常基於 Detectron2 框架。CenterMask 論文發表於 CVPR 2020，其核心目標是實現高效且精確的實例分割，尤其強調「實時性」。

Backbone: <mark style="background: #FFB86CA6;">VoVNet</mark>  (<mark style="background: #FFB86CA6;">OSA module)</mark>
Neck: FPN
Head: <mark style="background: #FFB86CA6;">FCOS head</mark> + <mark style="background: #FFB86CA6;">SAG-Mask head</mark>

### CenterMask2 的主要特點

1. **無錨點（Anchor-Free）**：
    
    - CenterMask 及其後續版本（CenterMask2）的核心特點是捨棄了傳統目標檢測中常用的預設錨點（anchor boxes）。這意味著模型不再需要大量手工設計的錨點來覆蓋不同尺度和長寬比的目標。
    - 無錨點方法通常能簡化模型結構，減少超參數調整的複雜性，並可能減少計算量，因為不需要對每個錨點進行分類和回歸。
    - CenterMask 是第一個基於無錨點目標檢測器實現實例分割的模型。
2. **實時性（Real-Time）**：
    
    - CenterMask 的一個關鍵目標是實現實時的實例分割。它通過優化網絡架構（如使用 VoVNetV2 作為骨幹網絡）和簡潔的設計來實現這一目標。
    - 論文中提到 CenterMask-Lite 版本甚至可以在 Titan Xp GPU 上以超過 35 fps 的速度運行，同時保持競爭力的準確性。
3. **基於 FCOS (Fully Convolutional One-Stage Object Detection)**：
    
    - CenterMask 將其新穎的分割分支（SAG-Mask）整合到一個高效的單階段無錨點目標檢測器 FCOS (Fully Convolutional One-Stage Object Detector) 上。這種設計方式類似於 Mask R-CNN 將 mask 分支插入 Faster R-CNN。
4. **創新的 SAG-Mask 分支**：
    
    - 引入了空間注意力引導的遮罩（Spatial Attention-Guided Mask, SAG-Mask）分支，這是其實現高性能實例分割的關鍵。

### BackBone Model: VoVNetV2

- **VoVNet（One-shot Aggregation for Light-weight Object Detection）**：VoVNet 是一種高效的骨幹網絡，專為實時目標檢測設計。它的核心思想是 **One-shot Aggregation (OSA) module**。
    - **OSA 模塊**：與 ResNet 或 DenseNet 不同，OSA 模塊在一個卷積層中聚合所有輸入特徵，而不是多個層次。這減少了內存訪問次數和計算量，從而提高了 GPU 的計算效率。
    - 這使得 VoVNet 在保持高準確度的同時，具有更快的推理速度和更低的計算成本。
- **VoVNetV2**：
    - CenterMask2 使用的是改進後的 VoVNetV2。VoVNetV2 進一步增強了原始 VoVNet 的性能和穩定性。
    - **主要改進**：
        1. **殘差連接（Residual Connection）**：在更大的 VoVNet 模型中引入了殘差連接，以緩解深度網絡可能出現的飽和問題（類似於 ResNet 的設計）。這有助於梯度傳播和模型優化。
        2. **高效的 Squeeze-Excitation (eSE) 模塊**：改進了原始的 Squeeze-Excitation (SE) 模塊，以處理通道信息丟失問題，同時保持計算效率。eSE 模塊有助於網絡學習不同通道特徵的重要性。
    - VoVNetV2 相較於 ResNe(X)t 或 HRNet 等骨幹網絡，在速度和準確性之間取得了更好的平衡。

### Neck Model (FPN)

CenterMask2 繼承了 FCOS 的架構，因此通常會採用 **FPN (Feature Pyramid Network)** 作為 Neck。

- **FPN 的作用**：FPN 是一種多尺度特徵融合網絡，它將骨幹網絡提取的不同層次的特徵圖進行組合，以生成具有豐富語義信息的高級特徵和具有精確空間信息的低級特徵。
- **多尺度檢測**：FPN 使得模型能夠在不同尺度上檢測目標。較小的目標在較高解析度的特徵圖上檢測，而較大的目標在較低解析度的特徵圖上檢測。這對於實例分割和目標檢測任務都至關重要。

### 分類與回歸頭（Head Model）: FCOS Box Head

- **FCOS Box Head 是否是 Object Detection 用的 FCOS？** **是的，CenterMask 的目標檢測部分正是基於 FCOS 的**。
    - FCOS（Fully Convolutional One-Stage Object Detection）是一個非常成功的無錨點單階段目標檢測器。
    - FCOS 在每個特徵圖上的每個空間位置直接預測：
        - **類別分類（Classification）**：該位置屬於哪個類別的概率。
        - **邊界框回歸（Box Regression）**：一個 4D 向量，表示從該位置到目標邊界框的上下左右四個邊的距離。
        - **Center-ness**：一個額外的分支，用於預測一個點位於目標中心區域的程度。這個分支幫助抑制低質量的邊界框，尤其是那些離目標中心較遠的邊界框，從而提高檢測性能。
    - CenterMask 正是將其 SAG-Mask 分支「即插即用」地整合到 FCOS 的檢測框輸出之後，為每個檢測到的邊界框生成對應的分割 Mask。

### 分割分支：SAG-Mask (Spatial Attention-Guided Mask)

SAG-Mask 是 CenterMask 的核心創新點，專門用於實例分割。

- **目標**：為 FCOS 檢測到的每個物體邊界框生成一個高質量的分割 Mask。
    
- **工作原理**：
    
    1. **ROI Alignment / Pooling**：首先，基於 FCOS 預測的邊界框，從 FPN 特徵圖上提取對應的區域（Region of Interest, ROI）。通常會使用 ROI Align 操作來獲得對齊的固定大小特徵圖，以避免量化誤差。
    2. **空間注意力模塊（Spatial Attention Module, SAM）**：這是 SAG-Mask 的關鍵。
        - SAG-Mask 不僅直接預測一個二值 Mask，它還預測一個**空間注意力圖（Spatial Attention Map）**。
        - 這個注意力圖會引導 Mask 預測分支**聚焦於物體內部的信息性像素**，同時**抑制背景中的噪聲**。這有助於提高 Mask 的質量和邊緣精度。
        - SAM 通過學習像素級的權重，告訴網絡哪些像素對於生成精確的 Mask 更重要。
    3. **Mask 預測**：結合注意力圖和提取的 ROI 特徵，一個輕量級的全卷積網絡（FCN）會預測最終的二值分割 Mask。這個 Mask 通常會被上採樣到原始 ROI 的尺寸。
- **優勢**：
    
    - **效率**：SAG-Mask 分支設計得很輕量，能夠與 FCOS 結合實現實時性能。
    - **精度**：引入空間注意力使得模型能夠更好地聚焦於物體本身，避免了背景干擾，從而提高了 Mask 的準確性，特別是邊緣的細節。

總的來說，CenterMask2 是一個高效且精準的實時無錨點實例分割模型，它巧妙地結合了 VoVNetV2 骨幹、FPN 頸部、FCOS 檢測頭和創新的 SAG-Mask 分支，使其在保持高速度的同時實現了領先的實例分割性能。




詳細比較一下 VoVNet 的 One-Shot Aggregation (OSA) 模塊與傳統 CNN 層和 ResNet 的 Residual Block 之間的主要差異。

### 1. 傳統 CNN Network 的 Layer

- **結構**：最基本的 CNN 層通常由一個或多個卷積層（Convolutional Layer）、激活函數（Activation Function，如 ReLU）和可能有的池化層（Pooling Layer）組成，按順序堆疊。
- **信息流**：信息通常是**單向流動**，從一層的輸出直接作為下一層的輸入。
- **特點**：每一層處理特徵，然後將處理過的特徵傳遞給下一層。隨著網絡深度的增加，信息可能會逐漸丟失，梯度也容易消失或爆炸。

### 2. ResNet 的 Residual Block (殘差塊)

- **結構**：ResNet (Residual Network) 引入了**殘差連接（Residual Connection / Skip Connection）**。一個基本的 Residual Block 通常包含兩到三個卷積層，然後將**輸入**直接添加到這些卷積層的**輸出**上。
    - `Output = F(Input) + Input`
    - 其中 `F(Input)` 代表經過卷積層和激活函數的變換。
- **信息流**：存在**兩條路徑**：一條是經過卷積操作的路徑，另一條是直接跳過卷積層的恆等映射（Identity Mapping）路徑。
- **特點**：
    - **解決梯度消失/爆炸**：殘差連接使得梯度可以直接從深層傳遞到淺層，極大地緩解了訓練深度網絡時的梯度問題。
    - **改善信息流動**：確保原始輸入信息能被保留和利用，防止信息隨網絡深度增加而丟失。
    - **允許構建更深的網絡**：是深度學習領域的一個里程碑，使得訓練百層甚至千層的網絡成為可能。
- **層數**：一個殘差塊通常包含 2 到 3 個卷積層。

### 3. VoVNet 的 One-Shot Aggregation (OSA) Block

VoVNet 的核心創新是 **One-Shot Aggregation (OSA)** 模塊，它的設計目標是**提高 GPU 的計算效率**，同時保持甚至超越 DenseNet 的優勢（特徵重用）。

- **結構**：
    - OSA Block 包含一系列**連續的卷積層**（例如，`l` 個 3x3 卷積層）。
    - 與 DenseNet 類似，OSA Block 會將**所有中間層的輸出特徵圖**與其**初始輸入特徵圖**在**通道維度上進行一次性拼接（Concatenation）**。
    - 但關鍵的區別在於：**這種拼接只發生在整個 OSA Block 的最後一次**，而不是像 DenseNet 那樣在每個中間層都拼接所有前面的特徵。
    - 最後，通常還有一個 1x1 卷積層來調整輸出通道數。
- **信息流**：
    - **「一次性聚合」**：這就是「One-Shot Aggregation」的含義。所有中間層的特徵在一個步驟中被匯集到最終輸出。
    - **通道增加**：像 DenseNet 一樣，通過拼接，特徵圖的通道數會增加，這能帶來更豐富的徵表示。
- **與 ResNet 和 DenseNet 的主要差異點**：
    1. **聚合方式**：
        - **ResNet**：通過**相加（Addition）**的方式聚合特徵（輸入 + 輸出）。
        - **DenseNet**：**密集地拼接（Concatenation）**所有前面層的輸出作為後面每一層的輸入，導致中間層的輸入通道數線性增長。
        - **VoVNet (OSA)**：在**整個 Block 的末端進行一次性拼接**所有中間層的輸出，避免了 DenseNet 中間層輸入通道數過大導致的內存訪問成本高和計算低效問題。
    2. **效率提升**：
        - DenseNet 雖然特徵重用效率高，但其「密集連接」導致每個中間層的輸入通道數不斷增加，這會增加**內存訪問成本（Memory Access Cost, MAC）**，從而降低 GPU 的實際計算效率。
        - OSA 通過只在**最後**進行一次聚合，使得**中間層的輸入通道數保持相對穩定**，顯著降低了 MAC 和計算成本，從而實現更快的推理速度。
    3. **層數差異**：
        - 一個 ResNet 的 Residual Block 通常包含 2-3 個卷積層，通過殘差連接。
        - 一個 VoVNet 的 OSA Block 包含一系列**連續**的卷積層（比如 3~5 層），其輸出在最後才與最初的輸入以及中間層的輸出進行拼接。

### 總結比較表：

|特性 / 模塊|傳統 CNN Layer|ResNet Residual Block|VoVNet One-Shot Aggregation (OSA) Block|
|:--|:--|:--|:--|
|**基本信息流**|單向、串行|殘差連接（Input + F(Input)）|多分支中間層，最後一次性拼接所有中間輸出和原始輸入|
|**聚合方式**|無（直接傳遞）|**相加（Addition）**|**一次性拼接（Concatenation）**|
|**目的**|基本特徵提取|解決梯度問題，允許深度網絡，信息流動更好|提升 GPU 運行效率，減少 MAC，同時保留特徵重用優勢|
|**層數**|1 或多個卷積層|通常 2-3 個卷積層 + 殘差連接|一系列連續的卷積層（例如 `l` 個），最後一次性聚合|
|**中間層輸入通道**|固定|固定|**固定**（與 DenseNet 每次線性增長不同）|
|**計算效率**|相對較低（深層時）|較高（相比無殘差網絡）|**更高**（尤其是在 GPU 上，因為 MAC 降低）|
|**特徵重用**|有限|通過殘差連接間接重用|**高效特徵重用**（類似 DenseNet 但更高效）|

匯出到試算表

VoVNet 的 OSA 模塊可以看作是從 DenseNet 的密集連接中汲取靈感，但同時解決了其在硬件（特別是 GPU）上效率低下的問題，通過精巧的「一次性聚合」設計，在速度和性能之間取得了優異的平衡。