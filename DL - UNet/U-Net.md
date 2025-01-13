
ref:  [UNet理解，pytorch实现，源码解读](https://zhuanlan.zhihu.com/p/571760241)
ref: [Unet论文超级详解（附图文：超细节超容易理解）](https://zhuanlan.zhihu.com/p/716339396)
ref: [U-Net原理分析与代码解读](https://zhuanlan.zhihu.com/p/150579454)
![[unet.png]]

如上图，Unet 网络结构是对称的，形似英文字母 U 所以被称为 Unet。整张图都是由蓝/白色框与各种颜色的箭头组成，其中，**蓝/白色框表示 feature map；蓝色箭头表示 3x3 卷积，用于特征提取；灰色箭头表示 skip-connection，用于[特征融合](https://zhida.zhihu.com/search?content_id=121594236&content_type=Article&match_order=1&q=%E7%89%B9%E5%BE%81%E8%9E%8D%E5%90%88&zhida_source=entity)；红色箭头表示池化 pooling，用于降低维度；绿色箭头表示上采样 upsample，用于恢复维度；青色箭头表示 1x1 卷积，用于输出结果**。其中灰色箭头`copy and crop`中的`copy`就是`concatenate`而`crop`是为了让两者的长宽一致


ref: [nn.ConvTranspose2d原理，深度网络如何进行上采样？](https://blog.51cto.com/u_15274944/5244229)

**U-Net** 是一種經典的圖像分割模型，主要用於醫學影像分割和其他需要像素級別精確的任務。以下是 U-Net 的損失函數與優化方法的詳細解釋：

---

### 1. **損失函數 (Loss Function)**

U-Net 的損失函數取決於任務的需求，常見的損失函數設計包括二元分類、多類分類、以及專注於像素級別精度的特殊損失函數。以下是幾種常見的損失函數，U-Net 可能根據具體任務使用其中的一種或多種組合。

#### (1) **Binary Cross-Entropy Loss (BCE)**

- 適用於二元分割任務。
- 該損失函數計算每個像素的預測值與真實值之間的二元交叉熵。
- **定義**： BCE Loss=−1N∑i=1N[yilog⁡(pi)+(1−yi)log⁡(1−pi)]\text{BCE Loss} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(p_i) + (1-y_i) \log(1-p_i) \right]BCE Loss=−N1​i=1∑N​[yi​log(pi​)+(1−yi​)log(1−pi​)] 其中 yiy_iyi​ 是第 iii 個像素的真實值，pip_ipi​ 是預測值。

#### (2) **Categorical Cross-Entropy Loss**

- 適用於多類分割（例如每個像素有多個可能的類別）。
- 該損失函數計算每個像素的預測分類與真實類別之間的交叉熵。
- **定義**： CCE Loss=−1N∑i=1N∑c=1Cyi,clog⁡(pi,c)\text{CCE Loss} = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C y_{i,c} \log(p_{i,c})CCE Loss=−N1​i=1∑N​c=1∑C​yi,c​log(pi,c​) 其中 CCC 是類別數，yi,cy_{i,c}yi,c​ 是像素 iii 屬於類別 ccc 的真實值（one-hot 表示），pi,cp_{i,c}pi,c​ 是預測值。

#### (3) **Dice Loss**

- 用於衡量預測遮罩與真實遮罩的相似度，適用於不平衡數據（正負像素數量差異大）。
- 通常與 BCE 結合使用，增強小物體或稀疏數據的分割能力。
- **定義**： Dice Loss=1−2∑i=1Npiyi∑i=1Npi+∑i=1Nyi\text{Dice Loss} = 1 - \frac{2 \sum_{i=1}^N p_i y_i}{\sum_{i=1}^N p_i + \sum_{i=1}^N y_i}Dice Loss=1−∑i=1N​pi​+∑i=1N​yi​2∑i=1N​pi​yi​​ 其中 pip_ipi​ 和 yiy_iyi​ 是第 iii 個像素的預測值和真實值。

#### (4) **Tversky Loss**

- Tversky Loss 是 Dice Loss 的變體，用於處理更嚴重的不平衡數據。
- **定義**： Tversky Loss=1−∑i=1Npiyi∑i=1Npiyi+α∑i=1Npi(1−yi)+β∑i=1N(1−pi)yi\text{Tversky Loss} = 1 - \frac{\sum_{i=1}^N p_i y_i}{\sum_{i=1}^N p_i y_i + \alpha \sum_{i=1}^N p_i (1-y_i) + \beta \sum_{i=1}^N (1-p_i) y_i}Tversky Loss=1−∑i=1N​pi​yi​+α∑i=1N​pi​(1−yi​)+β∑i=1N​(1−pi​)yi​∑i=1N​pi​yi​​ α\alphaα 和 β\betaβ 用於控制對假陽性和假陰性的懲罰權重。

#### (5) **Hybrid Loss**

- 為了提高模型表現，U-Net 通常將多種損失函數結合，如 **BCE + Dice Loss**，平衡分類準確性與形狀重建能力。

---

### 2. **優化方法 (Optimization Method)**

U-Net 的優化過程使用現代優化技術，這些技術可以加速模型收斂並提高分割效果。

#### (1) **優化器 (Optimizer)**

- **Adam Optimizer**:
    
    - Adam 是最常用的優化器之一，結合了動量法與自適應學習率方法。
    - **優化公式**： mt=β1mt−1+(1−β1)∇Ltm_t = \beta_1 m_{t-1} + (1-\beta_1) \nabla L_tmt​=β1​mt−1​+(1−β1​)∇Lt​ vt=β2vt−1+(1−β2)(∇Lt)2v_t = \beta_2 v_{t-1} + (1-\beta_2) (\nabla L_t)^2vt​=β2​vt−1​+(1−β2​)(∇Lt​)2 w^t+1=wt−ηmtvt+ϵ\hat{w}_{t+1} = w_t - \eta \frac{m_t}{\sqrt{v_t} + \epsilon}w^t+1​=wt​−ηvt​​+ϵmt​​ 其中 mtm_tmt​ 和 vtv_tvt​ 分別是動量和梯度平方的指數移動平均，η\etaη 是學習率。
- **SGD + Momentum**:
    
    - 在一些情況下，使用 Stochastic Gradient Descent (SGD) 配合 Momentum，提升收斂穩定性。
    - **更新公式**： vt=μvt−1−η∇L(wt)v_t = \mu v_{t-1} - \eta \nabla L(w_t)vt​=μvt−1​−η∇L(wt​) wt+1=wt+vtw_{t+1} = w_t + v_twt+1​=wt​+vt​ 其中 μ\muμ 是動量係數。

#### (2) **學習率調度 (Learning Rate Scheduler)**

- **Step Decay**:
    - 每隔幾個 epoch 將學習率減小固定比例，幫助模型更穩定地收斂。
- **Cosine Annealing**:
    - 在訓練過程中使用餘弦退火逐漸降低學習率。
- **Warm-up Scheduler**:
    - 訓練初期逐步增大學習率，避免模型不穩定。

#### (3) **批量大小 (Batch Size)**

- 批量大小依硬體資源而定，通常會在 8 到 32 之間選擇。

---

### 總結

1. **損失函數**:
    
    - 二元分割：Binary Cross-Entropy Loss、Dice Loss
    - 多類分割：Categorical Cross-Entropy Loss、Dice Loss
    - 不平衡數據：Tversky Loss、Hybrid Loss (如 BCE + Dice)
2. **優化方法**:
    
    - **優化器**: Adam Optimizer（首選），或 SGD + Momentum
    - **學習率調度**: Cosine Annealing、Step Decay、Warm-up Scheduler

這些設置確保 U-Net 模型在不同的分割場景中表現出色，尤其是在醫學影像分割中。