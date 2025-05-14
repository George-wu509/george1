
在深度學習中，**Activation Function（激活函數）** 是神經網絡中的關鍵組件，它決定了神經元的輸出，賦予神經網絡**非線性**的能力，從而能夠學習和表示更複雜的數據模式。以下是一些常見的激活函數及其特點與應用。梯度消失（vanishing gradient）和梯度爆炸（exploding gradient）

Reference:
激活函数sigmoid、tanh、softmax、relu、swish原理及区别 - 嘎嘣嘎嘣脆的文章 - 知乎
https://zhuanlan.zhihu.com/p/494417245

---

## 🔹 主要激活函數整理


![[Pasted image 20250301133910.png]]

| 激活函數                              | 公式                                                | 特點                                       | 優勢                                     | 缺點                           | 適用場景                          |
| --------------------------------- | ------------------------------------------------- | ---------------------------------------- | -------------------------------------- | ---------------------------- | ----------------------------- |
| **Sigmoid**                       | $\Huge f(x) = \frac{1}{1 + e^{-x}}$​              | - 值域 (0,1)，適用於二元分類  <br>- 有平滑曲線，適合概率輸出   | - 簡單易理解  <br>- 適合前期簡單模型                | - 梯度消失問題  <br>- 輸出範圍過窄（容易飽和） | - 二元分類的最後一層  <br>- 早期神經網絡     |
| **Tanh**                          | $\Huge f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$​ | - 值域 (-1,1)，比 Sigmoid 更強  <br>- 較適用於 RNN | - 中心對稱，梯度較大  <br>- 學習效果優於 Sigmoid      | - 仍有梯度消失問題                   | - 適用於 RNN  <br>- 適合處理負數值輸入    |
| **ReLU**                          | f(x)=max⁡(0,x)                                    | - 非線性，適合深層網絡  <br>- 計算簡單，收斂快             | - 具 sparsity（部分神經元不激活）  <br>- 避免梯度消失問題 | - Dying ReLU 問題（輸出恆為 0）      | - CNN 常用  <br>- 深度神經網絡        |
| **Softmax**                       | $\Huge f(x)_i = \frac{e^{x_i}}{\sum e^{x_j}}$​​   | - 多分類輸出，值域 (0,1) 且總和為 1                  | - 適用於多類別分類                             | - 可能導致梯度消失                   | - 最後一層的多分類問題                  |
|                                   |                                                   |                                          |                                        |                              |                               |
常见和常用的激活函数：

主要用于二分类：[Sigmoid](https://zhida.zhihu.com/search?content_id=198002069&content_type=Article&match_order=1&q=Sigmoid&zhida_source=entity)、[Tanh](https://zhida.zhihu.com/search?content_id=198002069&content_type=Article&match_order=1&q=Tanh&zhida_source=entity)
主要用于多分类：[Relu](https://zhida.zhihu.com/search?content_id=198002069&content_type=Article&match_order=1&q=Relu&zhida_source=entity)、[Swish](https://zhida.zhihu.com/search?content_id=198002069&content_type=Article&match_order=1&q=Swish&zhida_source=entity)、Softmax
主要用于深层网络：Relu、Swish、及Relu变形

**什么情况下适合使用Sigmoid？**
sigmoid函数的输出范围是0到1。由于输出值在0和1之间，它相当于将每个神经元的输出归一化。特别适合用于需要将预测概率作为输出的模型。因为任何概率值的范围是[0,1]，而且我们往往希望概率值尽量确定（即概率值远离0.5），所以s型曲线是最理想的选择。

**什么情况下适合使用tanh？**
值的范围在 -1到1之间，除此之外，tanh 函数的所有其他性质都与 Sigmoid形函数函数的性质相同。与 sigmoid 函数相比，tanh 函数的梯度更陡峭。通常 tanh 优于 sigmoid 函数，因为它以零为中心，并且梯度不限于在某个方向上移动。tanh在特征相差明显时的效果会很好，在循环过程中会不断扩大特征效果。在二分类问题中，一般tanh被用在隐层，而sigmoid被用在输出层

**什么情况下适合使用Relu？**
使用 ReLU 函数的主要优点是它不会同时激活所有神经元. 使用ReLU得到的SGD的收敛速度会比 sigmoid/tanh 快很多。ReLU 的缺点：  训练的时候很脆弱，大部分神经元会在训练中死忙，例如，一个非常大的梯度流过一个 ReLU 神经元，更新过参数之后，这个神经元再也不会对任何数据有激活现象了，那么这个神经元的梯度就永远都会是 0。 如果 learning rate 很大，那么很有可能网络中的 40% 的神经元都死了。

**Softmax**
Softmax 函数通常被描述为多个 sigmoid 的组合。 我们知道 sigmoid 返回 0到1之间的值，可以将其视为属于特定类的数据点的概率。 因此 sigmoid 被广泛用于二分类问题。softmax 函数可用于多类分类问题。此函数返回属于每个单独类的数据点的概率

#### Summary:
sigmoid 容易产生梯度消失问题，ReLU 的导数就不存在这样的问题

sigmoid将一个real value映射到（0,1）的区间，用来做二分类，而 softmax 主要进行进行多分类的任务。二分类问题时 sigmoid 和 softmax 是一样的，求的都是 cross entropy loss，而 softmax 可以用于多分类问题。

softmax建模使用的分布是多项式分布，而logistic则基于伯努利分布，多个logistic回归通过叠加也同样可以实现多分类的效果，但是 softmax回归进行的多分类，类与类之间是互斥的，即一个输入只能被归为一类；多个logistic回归进行多分类，输出的类别并不是互斥的，即"苹果"这个词语既属于"水果"类也属于"3C"类别。

如果使用 ReLU，要小心设置 learning rate，注意不要让网络出现很多 “dead” 神经元，如果不好解决，可以试试 Leaky ReLU，Swish等。


## **Sigmoid 為何容易產生梯度消失問題？**

Sigmoid 函數的定義如下：

$f(x) = \frac{1}{1 + e^{-x}}$

其導數為：

$f'(x) = f(x) (1 - f(x)) = \frac{e^{-x}}{(1 + e^{-x})^2}$

### **1. Sigmoid 的梯度範圍小**

Sigmoid 函數的輸出範圍是 (0,1)，其導數的最大值出現在 x=0：

$f'(0) = \frac{1}{4} = 0.25$

但當 ∣x∣ 很大時（例如 x≫1x \gg 1x≫1 或 x≪−1x \ll -1x≪−1），Sigmoid 的輸出趨於 1 或 0，導致導數趨近於 0：

lim⁡x→∞f′(x)=0,lim⁡x→−∞f′(x)=0\lim_{x \to \infty} f'(x) = 0, \quad \lim_{x \to -\infty} f'(x) = 0x→∞lim​f′(x)=0,x→−∞lim​f′(x)=0

這意味著：

- 當權重較大（正數）時，輸出值接近 1，導數趨於 0。
- 當權重較小（負數）時，輸出值接近 0，導數趨於 0。
- 只有在 xxx 附近時（約 -4 到 4 之間），Sigmoid 才有較大的梯度。

### **2. 反向傳播時梯度消失**

在神經網絡訓練過程中，**反向傳播（Backpropagation）** 會根據鏈式法則計算梯度：

∂L∂w=∂L∂y⋅f′(x)⋅∂x∂w\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot f'(x) \cdot \frac{\partial x}{\partial w}∂w∂L​=∂y∂L​⋅f′(x)⋅∂w∂x​

由於 Sigmoid 的梯度最多只有 0.25，而當網絡層數較多時，梯度會在每層傳播時被不斷地乘上小於 1 的值，導致梯度迅速衰減，最終變得非常接近 0，**影響前層權重的更新，導致訓練難以進行**。

### **3. 梯度消失的影響**

- **前幾層的權重幾乎不更新**：因為梯度在層層相乘後接近 0，前層的權重更新極小，導致網絡學習能力變差。
- **訓練速度變慢**：梯度太小，參數更新變慢，訓練需要很長時間。
- **難以訓練深度網絡**：對於 10 層以上的神經網絡，使用 Sigmoid 幾乎無法有效訓練。

---

## **為什麼 ReLU 不會產生梯度消失問題？**

ReLU（Rectified Linear Unit）的數學定義為：

f(x)=max⁡(0,x)f(x) = \max(0, x)f(x)=max(0,x)

其導數為：

f′(x)={1,x>00,x≤0f'(x) = \begin{cases} 1, & x > 0 \\ 0, & x \leq 0 \end{cases}f′(x)={1,0,​x>0x≤0​

### **1. 梯度為常數，不會消失**

ReLU 的導數在 x>0x > 0x>0 時恆為 1，而在 x≤0x \leq 0x≤0 時為 0，這帶來幾個優勢：

- 當 xxx 為正數時，**梯度恆為 1，不會因層數增加而逐漸消失**。
- 反向傳播時，梯度不會因為鏈式相乘而減小，這使得較深的神經網絡仍然能夠進行有效訓練。

### **2. 訓練收斂更快**

- 由於梯度為常數 1，參數更新速度快，訓練時間縮短。
- 避免 Sigmoid 與 Tanh 中的指數計算，提高計算效率。

### **3. 仍然存在問題：Dying ReLU**

ReLU 也並非完美，它有一個「Dying ReLU」問題，即：

- **當 x≤0x \leq 0x≤0 時，導數為 0，該神經元不會再被激活**。
- 若大量神經元的輸出為 0，網絡可能失去表達能力。
- 為了避免這個問題，通常使用 **Leaky ReLU**（允許小的負梯度），其定義為： f(x)={x,x>00.01x,x≤0f(x) = \begin{cases} x, & x > 0 \\ 0.01x, & x \leq 0 \end{cases}f(x)={x,0.01x,​x>0x≤0​

### **4. 總結：ReLU vs Sigmoid**

|特性|Sigmoid|ReLU|
|---|---|---|
|值域|(0,1)|[0,∞)[0, \infty)[0,∞)|
|導數範圍|(0, 0.25]|{0,1}\{0, 1\}{0,1}|
|梯度消失問題|✅ 存在|❌ 無|
|計算效率|低（指數運算）|高（只需取最大值）|
|訓練速度|慢|快|
|適用場景|二分類輸出層|深度網絡隱藏層|

---

## **結論**

1. **Sigmoid 容易梯度消失**，因為：
    
    - 其梯度範圍較小（最大為 0.25）。
    - 層層相乘後梯度迅速減小，前層幾乎無法更新。
    - 深層網絡訓練困難，收斂慢。
2. **ReLU 解決了梯度消失問題**，因為：
    
    - 導數為 0 或 1，不會隨著層數增加而消失。
    - 計算簡單，加快訓練速度。
    - 但可能存在「Dying ReLU」，可用 Leaky ReLU 改進。

在現代深度學習中，**隱藏層幾乎都使用 ReLU 或其變體（如 Leaky ReLU, PReLU, Swish）來避免梯度消失問題**，而 Sigmoid 則僅在 **輸出層（特別是二元分類）** 中使用。

|                                   |                             |                              |                                |               |                               |
| --------------------------------- | --------------------------- | ---------------------------- | ------------------------------ | ------------- | ----------------------------- |
| **Leaky ReLU**                    | f(x)=max⁡(0.01x,x)          | - 改進 ReLU，允許負數輸出             | - 避免 Dying ReLU  <br>- 適用於深度網絡 | - 仍然不夠平滑      | - 深度學習網絡  <br>- 避免 ReLU 問題的場景 |
| **Parametric ReLU (PReLU)**       | f(x)=max⁡(αx,x)             | - α\alphaα 可學習，適應不同數據        | - 進一步改進 ReLU  <br>- 允許負數信息傳遞   | - 增加參數，計算量稍大  | - 高度非線性場景  <br>- CNN、RNN      |
| **ELU (Exponential Linear Unit)** | f(x)=x,x>0; <br>α(ex−1),x≤0 | - 平滑，避免 Dying ReLU           | - 負輸出有較強的表達能力  <br>- 提供更好的梯度流  | - 計算量比 ReLU 大 | - 深度學習網絡  <br>- 需要更平滑激活函數的場景  |
| **Swish**                         | f(x)=x⋅σ(x)                 | - 自適應激活函數  <br>- 由 Google 提出 | - 比 ReLU 表現更好  <br>- 平滑，可微分    | - 計算較複雜       | - 用於高效深度學習模型                  |

## 
這裡展示了一些主要激活函數的圖形，幫助更直觀理解它們的變化趨勢：
```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)

# 定義激活函數
sigmoid = 1 / (1 + np.exp(-x))
tanh = np.tanh(x)
relu = np.maximum(0, x)
leaky_relu = np.where(x > 0, x, 0.01*x)
elu = np.where(x > 0, x, 0.1 * (np.exp(x) - 1))
swish = x / (1 + np.exp(-x))

# 繪製圖形
plt.figure(figsize=(12, 6))
plt.plot(x, sigmoid, label="Sigmoid", linestyle="--")
plt.plot(x, tanh, label="Tanh", linestyle=":")
plt.plot(x, relu, label="ReLU")
plt.plot(x, leaky_relu, label="Leaky ReLU")
plt.plot(x, elu, label="ELU")
plt.plot(x, swish, label="Swish", linestyle="-.")
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.legend()
plt.title("常見激活函數比較")
plt.show()

```

---

## 🔹 如何選擇適合的激活函數？

1. **一般深度學習（CNN、DNN）**
    
    - 🚀 **推薦**：ReLU、Leaky ReLU、PReLU、Swish
    - ❌ **避免**：Sigmoid、Tanh（因梯度消失）
2. **遞歸神經網絡（RNN、LSTM）**
    
    - 🚀 **推薦**：Tanh、ReLU、Swish
    - ❌ **避免**：Sigmoid（梯度消失嚴重）
3. **輸出層**
    
    - **二元分類**：Sigmoid
    - **多分類**：Softmax
    - **回歸問題**：線性激活（即不加激活函數）
4. **高性能深度學習**
    
    - 🚀 **推薦**：Swish、ELU（在某些情況下比 ReLU 更穩定）

---

## 🔹 總結

- **ReLU 系列（ReLU、Leaky ReLU、PReLU）**：最常用，特別是 CNN 和深度網絡。
- **Tanh**：適合 RNN，比 Sigmoid 更優，但仍有梯度消失問題。
- **Sigmoid**：適合輸出層二元分類，但在深層網絡中容易梯度消失。
- **Softmax**：多分類輸出常用。
- **Swish**：Google 提出的改進版 ReLU，在某些場合比 ReLU 更有效。

選擇適合的激活函數可以提高模型的學習效率和準確度，特別是對於不同的應用場景應做適當調整！