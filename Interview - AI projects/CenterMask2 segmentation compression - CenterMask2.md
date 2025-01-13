

CenterMask 是基於 FCOS 和 MaskRCNN 改進而來的，網路架構由 backbone+ FPN、FCOS Box head、SAG-Mask 所組成，可以看成是在 FCOS 的基礎上加入分割分支。此外，還提供了輕量模型 CenterMask-Lite。

![](https://miro.medium.com/v2/resize:fit:875/1*6SBIniL3XzS63B-neAiAlQ.png)

改進的部分有以下三項：

- 新的 backbone — VoVNetV2

VoVNetV2 是以 [VoVNet](https://arxiv.org/abs/1904.09730) 為基礎添加 Residual connection 和 Effective Squeeze-Excitation (eSE)。

由下圖 (c) 可看到在 OSA module 的輸出加入注意力機制 eSE，再使用 Residual connection。eSE 將原本 [SENet](https://arxiv.org/abs/1709.01507) block 的兩個全連接層改成一個全連接層，能夠防止通道訊息丟失，並減少計算量。

![](https://miro.medium.com/v2/resize:fit:875/1*pZQ913XIjIB1tK693c_Lsg.png)

- Adaptive RoI Assignment

在 Mask RCNN 中會根據檢測的 ROI 大小分配到不同的 FPN 層上，對應的公式為 Equation (1)，其中 224 是 ImageNet pretraining size。

![](https://miro.medium.com/v2/resize:fit:743/1*lP6dqMxxhTxeWnebaw_IgQ.png)

但該公式並不適用於 CenterMask，因為 two-stage detectors 使用的 FPN 層為 P2~P5，但 one-stage detectors 是使用 P3~P7；另外，其他尺寸的圖片按照參數 224 進行分配不太合理。因此提出新的對應公式，能夠提升小目標檢測的 AP，其中 k_max=5, k_min=3、A_input, A_ROI 分別為輸入圖片與 ROI 的面積。

![](https://miro.medium.com/v2/resize:fit:783/1*ndsgik0L3zHHwMpbEEL5mQ.png)

- Spatial Attention-Guided Mask

從 FCOS 輸出的 box 會先使用 RoI Align 提取 14x14 resolution 的特徵，然後送進 SAG-Mask 產生實例分割，SAG-Mask 是一個空間注意力機制的 mask 分支。

網路架構可由下圖看到，在 spatial attention module (SAM) 中，input feature map 會沿著 channel axis 進行 max pooling 及 avg pooling，用於提升對空間特徵的關注，接著經過一個 3x3 卷積層+ sigmoid，與原輸入做 element-wise 相乘。

![](https://miro.medium.com/v2/resize:fit:875/1*UveYbPXSk8Nsau8F57yy6A.png)


CVPR 2020 | CenterMask : Anchor-Free 实时实例分割(长文详解)
https://blog.csdn.net/qq_42722197/article/details/125701270


**CenterMask: Real-Time Anchor-Free Instance Segmentation** 是一個針對即時實例分割設計的深度學習模型。以下是它使用的損失函數與優化方法的詳細解釋：

---

### 1. **損失函數 (Loss Function)**

CenterMask 的損失函數由多個組件構成，因為它是針對實例分割的複雜任務，這裡需要同時考慮對物體類別、位置、以及實例遮罩的準確性。

#### (1) **分類損失 (Classification Loss)**:

- 用於對物體的類別進行分類。
- 通常採用 **Focal Loss**，以解決正負樣本數量不平衡的問題，這在實例分割和目標檢測中是一個常見的挑戰。
- **Focal Loss 定義**: FL(pt)=−αt(1−pt)γlog⁡(pt)\text{FL}(p_t) = -\alpha_t (1-p_t)^\gamma \log(p_t)FL(pt​)=−αt​(1−pt​)γlog(pt​) 其中 ptp_tpt​ 是模型對正確類別的預測機率，γ\gammaγ 控制難例的權重，αt\alpha_tαt​ 是對正負樣本的平衡參數。

#### (2) **中心點損失 (Center-ness Loss)**:

- 用於學習每個物體的中心點，該損失確保了預測框的位置準確性。
- 通常是 **Binary Cross-Entropy Loss (BCE)**，用來評估每個位置是否為物體的中心。

#### (3) **回歸損失 (Regression Loss)**:

- 用於回歸預測框的坐標，確保邊界框覆蓋物體。
- 通常採用 **L1 損失** 或 **Smooth L1 Loss**，這在回歸任務中非常常見。
- **Smooth L1 Loss 定義**: SmoothL1(x)={0.5x2if ∣x∣<1,∣x∣−0.5otherwise.\text{Smooth}_{L1}(x) = \begin{cases} 0.5x^2 & \text{if } |x| < 1, \\ |x| - 0.5 & \text{otherwise}. \end{cases}SmoothL1​(x)={0.5x2∣x∣−0.5​if ∣x∣<1,otherwise.​

#### (4) **遮罩損失 (Mask Loss)**:

- 用於學習物體的遮罩形狀。
- 主要使用 **Dice Loss** 或 **Binary Cross-Entropy Loss (BCE)**，Dice Loss 更加專注於像素級的準確性。
- **Dice Loss 定義**: Dice Loss=1−2∣A∩B∣∣A∣+∣B∣\text{Dice Loss} = 1 - \frac{2|A \cap B|}{|A| + |B|}Dice Loss=1−∣A∣+∣B∣2∣A∩B∣​ 其中 AAA 是預測遮罩，BBB 是真實遮罩。

---

### 2. **優化方法 (Optimization Method)**

CenterMask 的優化過程使用了現代深度學習中常見的優化方法：

#### (1) **優化器 (Optimizer)**:

- **Stochastic Gradient Descent (SGD)**:
    - 這是一種經典的梯度下降方法，常用於計算效率與收斂穩定性之間的平衡。
    - 搭配使用 **動量 (Momentum)** 和 **權重衰減 (Weight Decay)**，提高優化過程的穩定性並避免過擬合。
    - SGD 更新規則: wt+1=wt−η(∇L(wt)+λwt)w_{t+1} = w_t - \eta (\nabla L(w_t) + \lambda w_t)wt+1​=wt​−η(∇L(wt​)+λwt​) 其中 η\etaη 是學習率，λ\lambdaλ 是權重衰減係數。

#### (2) **學習率調度 (Learning Rate Scheduler)**:

- 通常採用 **Warm-up Cosine Annealing** 策略：
    - 在訓練初期逐漸增加學習率，然後使用餘弦退火方式逐漸降低學習率。
    - 有效提高收斂速度，並防止學習率過大導致的震盪。

#### (3) **批量大小 (Batch Size)**:

- 由於 CenterMask 是即時模型，需要高效的計算，因此通常使用較小的批量大小來降低記憶體佔用。

---

### 總結

1. **損失函數**:
    
    - 分類損失：Focal Loss
    - 中心點損失：Binary Cross-Entropy Loss
    - 回歸損失：Smooth L1 Loss
    - 遮罩損失：Dice Loss 或 Binary Cross-Entropy Loss
2. **優化方法**:
    
    - 優化器：SGD + Momentum + Weight Decay
    - 學習率調度：Warm-up Cosine Annealing

這些組合使 CenterMask 能夠在保證準確度的前提下實現即時的實例分割。