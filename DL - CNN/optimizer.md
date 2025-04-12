
在深度學習中，**Optimizer（優化器）** 是影響模型訓練效率與最終收斂效果的關鍵因素。優化器的核心目標是**最小化損失函數（Loss Function）**，並更新模型的權重，使其更接近最優解。

Ref: [每天3分钟，彻底弄懂神经网络的优化器optimizer](https://www.zhihu.com/people/luhengshiwo/posts)
Ref: [机器学习2 -- 优化器（SGD、SGDM、Adagrad、RMSProp、Adam](https://zhuanlan.zhihu.com/p/208178763)）
Ref: [深度学习各类优化器](https://github.com/zonechen1994/CV_Interview/blob/main/%E9%80%9A%E7%94%A8%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89%E7%AE%97%E6%B3%95%E9%9D%A2%E7%BB%8F/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80%E9%9D%A2%E8%AF%95%E9%A2%98/%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95.md)
Ref: [十分钟速通优化器原理，通俗易懂（从SGD到AdamW）](https://zhuanlan.zhihu.com/p/686410423)

| Optimizer                                    |                                                                                                                                                                                                                                                              |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **SGD**<br>(Stochastic <br>Gradient Descent) | **随机梯度下降**. 每次选择一个mini-batch，而不是全部样本，使用梯度下降来更新模型参数。这种方式可以近似地等价于原始的梯度下降法。采用SGD的好处就是可以在保证精度的同时降低计算量<br><br>Good: 它解决了随机小批量样本的问题<br>Bad:    仍然有自适应学习率、容易卡在梯度较小点等问题。                                                                                             |
| **SGDM**<br>(SGD with momentum)              | SGDM即为SGD with momentum，它加入**動量機制**. 引入动量的好处是可以**抵消梯度中那些变化剧烈的分量** t迭代的动量，其实是前t-1迭代的梯度的加权和。λ为衰减权重，越远的迭代权重越小. SGDM相比SGD优势明显，<br><br>Good: 参数更新就可以保持之前更新趋势，而不会卡在当前梯度较小的点了。<br>Bad:    SGDM没有考虑对学习率进行自适应更新，故学习率的选择很关键                                            |
| **Adagrad**                                  | adagrad利用迭代次数和累积梯度，对**学习率**进行自动衰减(不同参数应该有不同的学习率, 如何让模型自动地调整学习率呢)，从而使得刚开始迭代时，学习率较大，可以快速收敛。而后来则逐渐减小，精调参数，使得模型可以稳定找到最优点.学习率除以 前t-1 迭代的梯度的平方和。故称为自适应梯度下降。<br><br>Good: 对学习率进行自动衰减(自适应梯度下降)<br>Bad:    没有考虑迭代衰减. 极端情况，如果刚开始的梯度特别大，而后面的比较小，则学习率基本不会变化了，也就谈不上自适应学习率 |
| **Adam**                                     | Adam是SGDM和Adagrad的结合(SGD算法主要在**优化梯度**问题，Adagrad方法在**优化学习率**问题，那么可以不可以把这两种方法结合起来呢?)，它基本解决了之前提到的梯度下降的一系列问题，比如随机小样本、自适应学习率、容易卡在梯度较小点等问题                                                                                                                         |
| **AdamW**                                    | 在AdamW提出之前，Adam算法已经被广泛应用于深度学习模型训练中。但是人们发现，理论上更优的Adam算法，有时表现并不如SGD momentum好，尤其是在**模型泛化性**上. Adam算法弱化了L2范数的作用，所以导致了用Adam算法训练出来的模型泛化能力较弱. AdamW对这个问题的改进就是将权重衰减和Adam算法解耦                                                                                        |

**AdamW收敛得更快，更容易过拟合一点点；SGD收敛得相对慢一些，但是如果能给更长的训练轮次，最后的效果会略好于AdamW一些**

如上所示，SGDM在CV里面应用较多，而Adam则基本横扫NLP、RL、GAN、[语音合成](https://zhida.zhihu.com/search?content_id=134396393&content_type=Article&match_order=1&q=%E8%AF%AD%E9%9F%B3%E5%90%88%E6%88%90&zhida_source=entity)等领域。所以我们基本按照所属领域来使用就好了。比如NLP领域，Transformer、BERT这些经典模型均使用的Adam，及其变种AdamW。

optimizer优化主要有三种方法

1. 让模型探索更多的可能，包括dropout、加入Gradient noise、样本shuffle等
2. 让模型站在巨人肩膀上，包括warn-up、curriculum learning、[fine-tune](https://zhida.zhihu.com/search?content_id=134396393&content_type=Article&match_order=1&q=fine-tune&zhida_source=entity)等
3. 归一化 normalization，包括batch-norm和[layer-norm](https://zhida.zhihu.com/search?content_id=134396393&content_type=Article&match_order=1&q=layer-norm&zhida_source=entity)等
4. 正则化，惩罚模型的复杂度

---

## 🔹 主要優化器整理
![[Pasted image 20250301094614.png]]
---

## 🔹 PyTorch 代碼示例
```python
import torch
import torch.nn as nn
import torch.optim as optim

# 建立簡單的神經網絡
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)

# 定義損失函數
criterion = nn.MSELoss()

# 選擇不同優化器
optimizer_sgd = optim.SGD(model.parameters(), lr=0.01)
optimizer_momentum = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer_adagrad = optim.Adagrad(model.parameters(), lr=0.01)
optimizer_rmsprop = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)
optimizer_adam = optim.Adam(model.parameters(), lr=0.001)
optimizer_adamw = optim.AdamW(model.parameters(), lr=0.001)

# 模擬一個訓練步驟
input_data = torch.randn(10)
target = torch.tensor([1.0])  # 目標值

output = model(input_data)  # 前向傳播
loss = criterion(output, target)  # 計算損失

optimizer_adam.zero_grad()  # 清空梯度
loss.backward()  # 反向傳播
optimizer_adam.step()  # 更新參數

```


---

## 🔹 如何選擇適合的優化器？

1. **小型數據集 & 簡單網絡**
    
    - 🚀 **推薦**：SGD、Momentum
    - ✅ **原因**：計算量小，能快速訓練
2. **深度神經網絡（CNN、RNN）**
    
    - 🚀 **推薦**：Adam、RMSProp
    - ✅ **原因**：學習率自適應，適合梯度變化較大的問題
3. **自然語言處理（NLP）**
    
    - 🚀 **推薦**：Adagrad、Adam
    - ✅ **原因**：Adagrad 適合稀疏數據，而 Adam 可平衡學習速率
4. **大型 Transformer 模型**
    
    - 🚀 **推薦**：AdamW
    - ✅ **原因**：比 Adam 更能避免過擬合，適合大規模預訓練模型（如 BERT）

---

## 🔹 總結

- **SGD 系列（SGD、Momentum、NAG）**：適用於小型數據集與傳統神經網絡，如 LeNet、AlexNet。
- **自適應學習率系列（Adagrad、RMSProp、Adam）**：適合複雜的深度網絡，如 CNN、RNN、GAN。
- **AdamW**：在 Transformer 模型（如 GPT、BERT）中特別有效，能防止過擬合。

不同場景選擇合適的優化器，可以大幅提升訓練效率與收斂效果！



在深度學習中，**Optimizer（優化器）** 是神經網絡訓練過程中的核心部分，其目標是根據**損失函數（Loss Function）** 的梯度，調整模型的參數（如權重和偏置），使損失值最小化，從而提升模型的表現。不同的優化器有不同的更新方式，影響模型的收斂速度和效果。

---

# **🔹 常見優化器詳細介紹**

## **1. SGD（Stochastic Gradient Descent，隨機梯度下降）**

**公式：**

w=w−η∇L(w)w = w - \eta \nabla L(w)w=w−η∇L(w)

其中：

- www 是參數（權重）
- η\etaη 是學習率（learning rate）
- ∇L(w)\nabla L(w)∇L(w) 是損失函數對參數的梯度

**特點：**

- **隨機性**：每次更新僅基於一個或一小批數據（mini-batch）。
- **收斂速度慢**：由於隨機選取樣本，可能會在最優解附近來回震盪。
- **可能卡在局部極小值**：SGD 本身容易受到隨機梯度的影響，無法很好地逃離局部極小值。

**優勢：**

- **計算簡單**，適合大規模數據集。
- **收斂到全局最優解的可能性更大**，但需要適當的學習率調整。

**缺點：**

- **收斂速度慢，容易震盪**。
- **學習率固定，難以適應不同的梯度變化**。

**適用場景：**

- 經典的神經網絡（如 LeNet、AlexNet）。
- 大型數據集但計算資源有限時。

**PyTorch 實現：**

python

複製編輯

`optimizer = torch.optim.SGD(model.parameters(), lr=0.01)`

---

## **2. Momentum（動量梯度下降）**

**公式：**

vt=γvt−1+η∇L(w)v_t = \gamma v_{t-1} + \eta \nabla L(w)vt​=γvt−1​+η∇L(w) w=w−vtw = w - v_tw=w−vt​

其中：

- γ\gammaγ（通常為 0.9）是動量參數，表示過去梯度對當前梯度的影響程度。
- vtv_tvt​ 是累積的梯度。

**特點：**

- **累積過去的梯度信息**，減少震盪。
- **能更快地跨越鞍點（Saddle Points）**，避免陷入局部最小值。

**優勢：**

- **相比 SGD，能更快收斂，適合深度學習模型**。
- **適合非凸優化問題**，因為能夠跳出局部最小值。

**缺點：**

- 需要調整額外的超參數 γ\gammaγ。

**適用場景：**

- 深度神經網絡（如 CNN）。

**PyTorch 實現：**

python

複製編輯

`optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)`

---

## **3. NAG（Nesterov Accelerated Gradient，Nesterov 動量）**

**公式：**

vt=γvt−1+η∇L(w−γvt−1)v_t = \gamma v_{t-1} + \eta \nabla L(w - \gamma v_{t-1})vt​=γvt−1​+η∇L(w−γvt−1​) w=w−vtw = w - v_tw=w−vt​

**特點：**

- **提前計算一個更新方向**，然後再決定如何調整梯度，減少震盪。
- **更具前瞻性**，能比 Momentum 更快地接近最優解。

**優勢：**

- 比 Momentum 更穩定，避免梯度過衝。
- **適合深度學習網絡**，特別是在具有高曲率的優化問題中。

**缺點：**

- 需要額外計算梯度，計算開銷較大。

**適用場景：**

- 需要較快收斂的深度網絡。

**PyTorch 實現：**

python

複製編輯

`optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)`

---

## **4. Adagrad（Adaptive Gradient）**

**公式：**

w=w−ηGt+ϵ∇L(w)w = w - \frac{\eta}{\sqrt{G_t + \epsilon}} \nabla L(w)w=w−Gt​+ϵ​η​∇L(w)

其中：

- GtG_tGt​ 是所有過去梯度平方的累積值。

**特點：**

- **學習率會根據歷史梯度自適應調整**，梯度較小的方向學習率較大，梯度較大的方向學習率較小。
- **適合稀疏數據**，如 NLP 任務。

**優勢：**

- 避免手動調整學習率。
- 適合處理高維特徵稀疏的數據。

**缺點：**

- **學習率會逐漸變小，可能會導致過早收斂**。

**適用場景：**

- 自然語言處理（NLP）、推薦系統等稀疏數據應用。

**PyTorch 實現：**

python

複製編輯

`optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)`

---

## **5. RMSProp（Root Mean Square Propagation）**

**公式：**

vt=βvt−1+(1−β)∇L(w)2v_t = \beta v_{t-1} + (1-\beta) \nabla L(w)^2vt​=βvt−1​+(1−β)∇L(w)2 w=w−ηvt+ϵ∇L(w)w = w - \frac{\eta}{\sqrt{v_t + \epsilon}} \nabla L(w)w=w−vt​+ϵ​η​∇L(w)

**特點：**

- **解決 Adagrad 學習率逐漸變小的問題**，通過滑動平均的方法抑制梯度變化劇烈的方向。
- **適合 RNN 和序列數據**。

**優勢：**

- 比 Adagrad 更穩定，不會讓學習率下降過快。
- 適合非凸函數優化。

**缺點：**

- 需要額外的 β\betaβ 參數調整。

**適用場景：**

- RNN、LSTM、Transformer 模型。

**PyTorch 實現：**

python

複製編輯

`optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)`

---

## **6. Adam（Adaptive Moment Estimation）**

**公式：**

mt=β1mt−1+(1−β1)∇L(w)m_t = \beta_1 m_{t-1} + (1-\beta_1) \nabla L(w)mt​=β1​mt−1​+(1−β1​)∇L(w) vt=β2vt−1+(1−β2)∇L(w)2v_t = \beta_2 v_{t-1} + (1-\beta_2) \nabla L(w)^2vt​=β2​vt−1​+(1−β2​)∇L(w)2 w=w−ηv^t+ϵm^tw = w - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_tw=w−v^t​​+ϵη​m^t​

**特點：**

- **結合 Momentum 和 RMSProp，能適應不同的梯度變化**。
- **目前最流行的優化器**，適用於幾乎所有深度學習應用。

**優勢：**

- 快速收斂，適合大規模數據集。
- 適應不同梯度變化。

**缺點：**

- 容易過擬合，收斂後效果可能不穩定。

**適用場景：**

- 幾乎所有深度學習任務，如 CNN、RNN、Transformer。

**PyTorch 實現：**

python

複製編輯

`optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`

---

## **7. AdamW（Weight Decay Adam）**

**特點：**

- **專門為 Transformer 優化**，避免 Adam 過擬合問題。
- **適用於大規模預訓練模型，如 BERT、GPT**。

**PyTorch 實現：**

python

複製編輯

`optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)`

---

### **🔹 總結**

- **SGD** 適合小型數據集，收斂慢。
- **Momentum、NAG** 加速收斂，適合 CNN。
- **Adagrad、RMSProp** 適合 NLP、RNN。
- **Adam** 適用於大部分深度學習應用。
- **AdamW** 適合 Transformer 類模型。


首先，严格来说[Transformer](https://zhida.zhihu.com/search?content_id=463651099&content_type=Answer&match_order=1&q=Transformer&zhida_source=entity)用的是[AdamW](https://zhida.zhihu.com/search?content_id=463651099&content_type=Answer&match_order=1&q=AdamW&zhida_source=entity)，只不过现在的框架都把Adam偷偷换成了AdamW，所以没什么人在意这件事情了。如果是AdamW和[SGD](https://zhida.zhihu.com/search?content_id=463651099&content_type=Answer&match_order=1&q=SGD&zhida_source=entity)的比较，简单来说就是：AdamW收敛得更快，更容易过拟合一点点；SGD收敛得相对慢一些，但是如果能给更长的训练轮次，最后的效果会略好于AdamW一些。由于基于Transformer的模型都很巨大，考虑到非常难收敛且不容易过拟合的特性，因此很多模型都使用AdamW。而一些[CNN](https://zhida.zhihu.com/search?content_id=463651099&content_type=Answer&match_order=1&q=CNN&zhida_source=entity)模型，相对比较容易收敛，且相比Transformer更容易过拟合一些，因此选择SGD。
https://www.zhihu.com/question/519307910/answer/2384626354
