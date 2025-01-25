

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
- **Focal Loss 定義**: $\large \text{FL}(p_t) = -\alpha_t (1-p_t)^\gamma \log(p_t)$ 其中 $p_t$​ 是模型對正確類別的預測機率，γ 控制難例的權重，$\alpha_t$​ 是對正負樣本的平衡參數。

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

SAG-Mask（Spatial Attention-Guided Mask）是CenterMask模型的一个核心组件，设计目的是通过引入空间注意力机制来提升实例分割的性能和鲁棒性。以下是SAG-Mask的详细中文解析：

---

### **1. SAG-Mask 的基本设计理念**

SAG-Mask基于空间注意力模块（SAM, Spatial Attention Module）指导分割头关注有意义的像素，同时抑制无用信息。它在CenterMask中作为独立的分割分支，利用由检测器（FCOS）预测的边界框（Bounding Box）生成像素级的分割结果。

- **目标：**
    - 聚焦于重要的特征区域（例如目标边缘和内部）。
    - 减少背景噪声对分割精度的影响。

---

### **2. SAG-Mask 的工作流程**

1. **输入特征提取**：
    
    - RoI Align操作将每个预测边界框内的特征提取为固定尺寸（例如 14×1414 \times 1414×14）的特征图。
2. **特征增强（空间注意力模块 - SAM）**：
    
    - 对每个输入特征图 Xi∈RC×W×HX_i \in \mathbb{R}^{C \times W \times H}Xi​∈RC×W×H：
        - **全局池化**：计算沿通道维度的平均池化 PavgP_{\text{avg}}Pavg​ 和最大池化 PmaxP_{\text{max}}Pmax​，尺寸均为 1×W×H1 \times W \times H1×W×H。
        - **特征融合**：将 PavgP_{\text{avg}}Pavg​ 和 PmaxP_{\text{max}}Pmax​ 在通道维度上拼接，并通过 3×33 \times 33×3 卷积层和Sigmoid激活函数生成空间注意力权重图 Asag(Xi)A_{\text{sag}}(X_i)Asag​(Xi​)。 Asag(Xi)=σ(F3×3([Pavg,Pmax]))A_{\text{sag}}(X_i) = \sigma(F_{3\times3}([P_{\text{avg}}, P_{\text{max}}]))Asag​(Xi​)=σ(F3×3​([Pavg​,Pmax​]))
        - **特征加权**：用 Asag(Xi)A_{\text{sag}}(X_i)Asag​(Xi​) 对输入特征图 XiX_iXi​ 进行加权（逐元素相乘）： Xsag=Asag(Xi)⊙XiX_{\text{sag}} = A_{\text{sag}}(X_i) \odot X_iXsag​=Asag​(Xi​)⊙Xi​
    - 加权后特征图 XsagX_{\text{sag}}Xsag​ 聚焦于目标区域，同时抑制噪声。
3. **分割输出生成**：
    
    - **上采样**：通过 2×22 \times 22×2 转置卷积将特征图从 14×1414 \times 1414×14 放大到 28×2828 \times 2828×28。
    - **分类预测**：使用 1×11 \times 11×1 卷积生成类别特定的分割掩码。

---

### **3. SAG-Mask 的优势**

1. **聚焦目标区域**：
    
    - SAM通过关注空间上重要的区域，抑制背景或不相关区域的特征，从而提升分割的精度。
2. **模块化设计**：
    
    - SAG-Mask可以无缝集成到其他基于RoI的分割框架中，具有良好的通用性。
3. **计算开销小**：
    
    - SAM的额外计算量主要来源于少量池化、卷积和Sigmoid操作，相比其他注意力机制非常轻量。

---

### **4. 实验结果与分析**

论文通过消融实验验证了SAG-Mask的有效性：

- 加入SAM模块后，分割性能（APmask）显著提升（如表1所示）。
- SAM模块不仅提升了分割的准确性，还间接提高了检测性能（APbox），因为改进的特征图在检测头中也被复用。

---

### **5. 数学公式总结**

1. **空间注意力权重图计算**： Asag(Xi)=σ(F3×3([Pavg,Pmax]))A_{\text{sag}}(X_i) = \sigma(F_{3\times3}([P_{\text{avg}}, P_{\text{max}}]))Asag​(Xi​)=σ(F3×3​([Pavg​,Pmax​]))
2. **特征图加权**： Xsag=Asag(Xi)⊙XiX_{\text{sag}} = A_{\text{sag}}(X_i) \odot X_iXsag​=Asag​(Xi​)⊙Xi​

---

### **6. 适用场景与改进方向**

- 适用于实时性要求高的实例分割任务。
- 可以进一步探索多尺度特征图的注意力机制，优化小目标的分割性能。


在 **CenterMask2** 的論文中，損失函數 (Loss Function) 和優化方法 (Optimization Method) 是實現高效且精準的實例分割的關鍵部分。以下將詳細解釋這些內容並配合具體的數學公式和示例：

---

## **1. CenterMask2 的損失函數**

CenterMask2 的損失函數是基於多任務學習的設計，結合了分類、中心度 (Centerness)、邊界框回歸和分割損失，針對目標檢測與分割任務進行聯合優化。

### **損失函數總公式**

$\Large L = L_{\text{cls}} + L_{\text{center}} + L_{\text{box}} + L_{\text{mask}}$

其中：

- $L_{\text{cls}}$：分類損失 (Classification Loss)，用於指導模型正確分類物體類別。
- $L_{\text{center}}$​：中心度損失 (Centerness Loss)，用於強化模型對目標中心像素的重視程度。
- $L_{\text{box}}$：邊界框回歸損失 (Bounding Box Regression Loss)，用於精確回歸目標邊界框。
- $L_{\text{mask}}$​：分割損失 (Segmentation Loss)，用於提升像素級分割的準確性。

---

### **1.1 分類損失 LclsL_{\text{cls}}Lcls​**

- **公式**：
    
    Lcls=−1N∑i=1Nyilog⁡(pi)L_{\text{cls}} = -\frac{1}{N} \sum_{i=1}^N y_i \log(p_i)Lcls​=−N1​i=1∑N​yi​log(pi​)
    
    其中：
    
    - NNN：預測的所有像素數量。
    - yi∈{0,1}y_i \in \{0, 1\}yi​∈{0,1}：第 iii 個像素的真實類別標籤。
    - pip_ipi​：第 iii 個像素屬於前景的預測概率。
- **示例**： 假設有 4 個像素，其真實標籤 Y=[1,0,1,0]Y = [1, 0, 1, 0]Y=[1,0,1,0]，模型的預測概率 P=[0.9,0.2,0.8,0.3]P = [0.9, 0.2, 0.8, 0.3]P=[0.9,0.2,0.8,0.3]。計算損失：
    
    Lcls=−14[1⋅log⁡(0.9)+0⋅log⁡(0.2)+1⋅log⁡(0.8)+0⋅log⁡(0.3)]L_{\text{cls}} = -\frac{1}{4} \left[ 1 \cdot \log(0.9) + 0 \cdot \log(0.2) + 1 \cdot \log(0.8) + 0 \cdot \log(0.3) \right]Lcls​=−41​[1⋅log(0.9)+0⋅log(0.2)+1⋅log(0.8)+0⋅log(0.3)] =−14[log⁡(0.9)+log⁡(0.8)]≈−14[−0.105+(−0.223)]=0.082= -\frac{1}{4} \left[ \log(0.9) + \log(0.8) \right] \approx -\frac{1}{4} \left[ -0.105 + (-0.223) \right] = 0.082=−41​[log(0.9)+log(0.8)]≈−41​[−0.105+(−0.223)]=0.082

---

### **1.2 中心度損失 LcenterL_{\text{center}}Lcenter​**

- **作用**：衡量像素偏離中心的程度。中心像素的預測權重較高，邊緣像素的影響較小。
- **公式**： Lcenter=Binary Cross-Entropy Loss (BCE)L_{\text{center}} = \text{Binary Cross-Entropy Loss (BCE)}Lcenter​=Binary Cross-Entropy Loss (BCE) 與分類損失類似，BCE 適合二元任務，幫助模型聚焦於目標的中心位置。

---

### **1.3 邊界框回歸損失 LboxL_{\text{box}}Lbox​**

- **作用**：回歸預測的邊界框到真實邊界框。
    
- **公式**： 通常採用 GIoU Loss（Generalized IoU）或 L1 損失：
    
    Lbox=Smooth L1 LossL_{\text{box}} = \text{Smooth L1 Loss}Lbox​=Smooth L1 Loss LsmoothL1(x)={0.5x2if ∣x∣<1,∣x∣−0.5otherwise.L_{\text{smoothL1}}(x) = \begin{cases} 0.5x^2 & \text{if } |x| < 1, \\ |x| - 0.5 & \text{otherwise.} \end{cases}LsmoothL1​(x)={0.5x2∣x∣−0.5​if ∣x∣<1,otherwise.​
    
    xxx 是預測框與真實框的偏差。
    
- **示例**： 假設真實框的坐標為 [x1,y1,x2,y2]=[0,0,100,100][x_1, y_1, x_2, y_2] = [0, 0, 100, 100][x1​,y1​,x2​,y2​]=[0,0,100,100]，預測框為 [x1′,y1′,x2′,y2′]=[5,5,95,95][x'_1, y'_1, x'_2, y'_2] = [5, 5, 95, 95][x1′​,y1′​,x2′​,y2′​]=[5,5,95,95]，損失計算：
    
    Lbox=14∑smoothL1([5,5,−5,−5])=14[0.5⋅52+0.5⋅52+0.5⋅(−5)2+0.5⋅(−5)2]=25L_{\text{box}} = \frac{1}{4} \sum \text{smoothL1}([5, 5, -5, -5]) = \frac{1}{4} \left[ 0.5 \cdot 5^2 + 0.5 \cdot 5^2 + 0.5 \cdot (-5)^2 + 0.5 \cdot (-5)^2 \right] = 25Lbox​=41​∑smoothL1([5,5,−5,−5])=41​[0.5⋅52+0.5⋅52+0.5⋅(−5)2+0.5⋅(−5)2]=25

---

### **1.4 分割損失 LmaskL_{\text{mask}}Lmask​**

- **作用**：用於像素級分割。
    
- **公式**： 通常採用 **Binary Cross-Entropy Loss (BCE)** 或 **Dice Loss**：
    
    Lmask=1−2∑i=1Npiyi∑i=1Npi+∑i=1NyiL_{\text{mask}} = 1 - \frac{2 \sum_{i=1}^N p_i y_i}{\sum_{i=1}^N p_i + \sum_{i=1}^N y_i}Lmask​=1−∑i=1N​pi​+∑i=1N​yi​2∑i=1N​pi​yi​​
- **示例**： 假設真實分割標籤為：
    
    Y=[1011]Y = \begin{bmatrix} 1 & 0 \\ 1 & 1 \end{bmatrix}Y=[11​01​]
    
    預測概率為：
    
    P=[0.90.20.80.6]P = \begin{bmatrix} 0.9 & 0.2 \\ 0.8 & 0.6 \end{bmatrix}P=[0.90.8​0.20.6​]
    
    Dice Loss：
    
    Lmask=1−2⋅(0.9+0.8+0.6)(0.9+0.2+0.8+0.6)+(1+0+1+1)=1−2⋅2.32.5+3≈0.095L_{\text{mask}} = 1 - \frac{2 \cdot (0.9 + 0.8 + 0.6)}{(0.9 + 0.2 + 0.8 + 0.6) + (1 + 0 + 1 + 1)} = 1 - \frac{2 \cdot 2.3}{2.5 + 3} \approx 0.095Lmask​=1−(0.9+0.2+0.8+0.6)+(1+0+1+1)2⋅(0.9+0.8+0.6)​=1−2.5+32⋅2.3​≈0.095

---

## **2. 優化方法 (Optimization Method)**

CenterMask2 採用標準的優化方法，包括：

1. **優化器**：隨機梯度下降 (Stochastic Gradient Descent, SGD)。
    
    - 初始學習率：0.01。
    - 動量：0.9。
    - 權重衰減：0.0001。
2. **學習率調整**：
    
    - 在訓練過程中，學習率按指定步驟下降。
    - 具體：在第 60K 和 80K 次迭代時，將學習率縮小為原來的 1/101/101/10。
3. **訓練設置**：
    
    - 批量大小 (Batch Size)：16。
    - 訓練迭代次數：約 90K 次（約 12 個 Epoch）。

---

### **示例：損失函數的聯合優化**

假設一張圖像的計算結果如下：

- Lcls=0.082L_{\text{cls}} = 0.082Lcls​=0.082，
- Lcenter=0.05L_{\text{center}} = 0.05Lcenter​=0.05，
- Lbox=25L_{\text{box}} = 25Lbox​=25，
- Lmask=0.095L_{\text{mask}} = 0.095Lmask​=0.095。

損失總和：

L=Lcls+Lcenter+Lbox+Lmask=0.082+0.05+25+0.095=25.227L = L_{\text{cls}} + L_{\text{center}} + L_{\text{box}} + L_{\text{mask}} = 0.082 + 0.05 + 25 + 0.095 = 25.227L=Lcls​+Lcenter​+Lbox​+Lmask​=0.082+0.05+25+0.095=25.227

隨後，模型基於此損失進行梯度更新，以逐步縮小損失值。

---

### **3. 總結**

1. **損失函數設計**：
    
    - 分類、中心度、邊界框回歸和分割損失協同工作，確保模型在多任務間平衡。
    - 使用 BCE、Smooth L1 和 Dice Loss 等標準方法。
2. **優化方法**：
    
    - 使用 SGD 和動量，結合學習率衰減策略，實現穩定優化。
3. **應用場景**：
    
    - CenterMask2 的損失函數設計非常靈活，可應用於其他實例分割模型。