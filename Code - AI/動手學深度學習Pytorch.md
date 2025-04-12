
《动手学深度学习PyTorch版》：全要点笔记 - hadiii的文章 - 知乎
https://zhuanlan.zhihu.com/p/664880302


**【基本完结】本文是对《动手学深度学习PyTorch版》的复习要点记录，以查漏补缺、巩固基础和准备面试为主。笔记把一些有关的章节聚合为一个小专题进行学习，包含概念，图示，公式和部分重要源码解读。内容跨度上，会从最基本的张量到近来必会必知的GPT，侧重NLP方向。文章较长，共十大专题，加上附录已经5.7w字，建议按目录查询阅读。**

《动手学深度学习》这门课程算是真正的良心课程，免费，全面，高质量...因此推荐给每一位想入门DL，或是想复习一些DL基础，也或是准备面试的同学等等。

> 未加额外说明时，本文的图源均为《动手学深度学习PyTorch版》书籍PDF。参考内容和相关的官方链接如下：

[GitHub - d2l-ai/d2l-zh: 《动手学深度学习》：面向中文读者、能运行、可讨论。中英文版被70多个国家的500多所大学用于教学。​github.com/d2l-ai/d2l-zh](https://link.zhihu.com/?target=https%3A//github.com/d2l-ai/d2l-zh)

  

同时也安利“从Llama 3报告出发的LLM基本技术整理”，概括性地介绍了现代LLM的基本技术，放到这里，欢迎来读。需要对DL和LLM有基本了解。

[hadiii：从Llama 3报告出发的LLM基本技术整理192 赞同 · 9 评论文章![](https://picx.zhimg.com/v2-7a80e49ac2d794e2b653f464713ff4f5_r.jpg?source=172ae18b)](https://zhuanlan.zhihu.com/p/713794852)

同样，记录过一些关于LLM Agents的浅浅思考放到这里，欢迎大家来讨论。

[hadiii：写在跨年之前：聊聊LLM Agents的现状，问题与未来322 赞同 · 38 评论文章![](https://picx.zhimg.com/v2-c79fc3af292d7a8ce926afd0e9920487_r.jpg?source=172ae18b)](https://zhuanlan.zhihu.com/p/679177488)

  

## 专题0 张量与梯度

### 张量的基本概念

张量是一种多维数组，支持在GPU上进行计算，是计算图中的节点，并且支持[自动微分](https://zhida.zhihu.com/search?content_id=235880363&content_type=Article&match_order=1&q=%E8%87%AA%E5%8A%A8%E5%BE%AE%E5%88%86&zhida_source=entity)。

张量的广播机制：在进行运算时，如果两个张量在某个维度上的形状不匹配，那么系统会自动地在这个维度上扩展形状较小的张量，使得两个张量在这个维度上具有相同的形状。

广播机制的例子：

```python3
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a + b)
```

输出：

```text
tensor([[0, 1],
        [1, 2],
        [2, 3]])
```

### 梯度和链式法则

假设函数 f:Rn→R 的输入是一个 n 维向量 x=[x1,x2,…,xn]T ，并且输出是一个标量。而函数 f(x) 相对于 x 的梯度是一个包含 n 个偏导数的向量：

∇xf(x)=[∂f(x)∂x1,∂f(x)∂x2,…,∂f(x)∂xn]T

这个梯度向量的每个元素都是函数 f 在对应维度上的偏导数，它描述了函数 f 在该点的局部变化率。在优化问题中，梯度常常被用来指示函数增长最快的方向。

假设可微函数 y 有变量 u1,u2,…,um ，其中每个可微分函数 ui 都有变量 x1,x2,…,xn 。注意，y是 x1,x2,…,xn 的间接函数。我们可以使用链式法则计算 y 相对于 xi 的偏导数。链式法则表明：

∂y∂xi=∑j=1m∂y∂uj∂uj∂xi

### 自动微分

深度学习框架可以自动计算导数：根据设计好的模型，系统会构建⼀个计算图。当定义一个变量并指定它需要计算梯度时，框架会跟踪所有与该变量有关的计算。然后，当计算一个目标值（通常是损失函数）并调用反向传播函数时`.backward()`，框架会沿着这些计算的路径反向传播，使用链式法则来计算每个变量的偏导数。每次调用`.backward()`时，新的梯度会加到已有的梯度上。

例子：

```python3
import torch

# 创建一个包含4个元素的张量（向量）[0., 1., 2., 3.]，并设置requires_grad=True以跟踪对其的所有操作。
x = torch.arange(4.0, requires_grad=True)

# 计算y，它是x和x自身的点积乘以2。这里y是一个标量。
y = 2 * torch.dot(x, x)

# 对y进行反向传播，计算y关于x的梯度。由于y是一个标量，这等价于计算y的导数。
y.backward()

# 打印出x的梯度。由于y = 2 * x^T * x，y关于x的梯度是4 * x。
print(x.grad)

# 检查计算出的梯度是否如我们通过手动计算得出的那样，即4 * x。
print(x.grad == 4 * x)

# 清除之前计算出的梯度值。在PyTorch中，如果不手动清零，梯度会累积。
x.grad.zero_()

# 计算一个新的y，它是x的所有元素的和。
y = x.sum()

# 对新的y进行反向传播，计算关于x的梯度。
y.backward()

# 打印出新的梯度。由于y是x的和，y关于x的梯度是一个全1的向量。
print(x.grad)
```

输出：

```text
tensor([ 0., 4., 8., 12.])
tensor([True, True, True, True])
tensor([1., 1., 1., 1.])
```

## 专题1 线性回归，softmax回归，多层感知机，激活函数

### 线性回归

![](https://pic4.zhimg.com/v2-9473e507628de96f38529653e12b1621_1440w.jpg)

线性回归是⼀种单层神经⽹络

**线性回归方法的四个关键式子：**

1. 线性模型的预测公式：

y^=Xw+b

这个公式表示向量 y^ （预测值）是矩阵 X （特征）和向量 w（权重）的乘积加上偏置项 b 。这里， X∈Rn×d ，其中 n 是样本数量， d 是特征数量。

2. 每个样本的平方误差损失：

l(i)(w,b)=12(y^(i)−y(i))2

3. 整个数据集的平均损失：

L(w,b)=1n∑i=1nl(i)(w,b)=1n∑i=1n12(wTx(i)+b−y(i))2

4. 最优参数的求解：

(w∗,b∗)=arg⁡minw,bL(w,b)

**随机梯度下降方法求解线性回归问题：**

1. 指定超参数，本问题中是批量大小和学习率。
2. 初始化模型参数的值，如从均值为0、标准差为0.01的正态分布中随机采样，偏置参数初始化为零。
3. 从数据集中随机抽取小批量样本且在负梯度的方向上更新参数，并不断迭代这⼀步骤。

权重更新过程的数学表达如下：

w←w−η|B|∑i∈B∂l(i)(w,b)∂w=w−η|B|∑i∈Bx(i)(wTx(i)+b−y(i))

b←b−η|B|∑i∈B∂l(i)(w,b)∂b=b−η|B|∑i∈B(wTx(i)+b−y(i))

### softmax回归

![](https://pic3.zhimg.com/v2-6c33e0f549e081724f04227c328ddd38_1440w.jpg)

softmax回归是⼀种单层神经⽹络

**Softmax回归方法的四个关键式子：**

1. Softmax的定义：Softmax函数将一个实数向量转换为概率分布。对于每个元素，它计算该元素的指数与所有元素的指数之和的比值。这样可以确保输出向量的所有元素都是非负的，并且总和为1，因此可以被视为概率分布。

其中y^=softmax(o)其中y^j=exp⁡(oj)∑kexp⁡(ok)

2. Softmax的输出就是选择最有可能的类别（取概率最大的），尽管softmax函数改变了输出向量的值，但它不改变元素之间的顺序。

arg⁡maxjy^j=arg⁡maxjoj

3.交叉熵损失：在多分类问题中，模型预测的概率分布为 y^ ，而真实的标签分布为 y 。交叉熵损失函数用于度量这两个分布之间的差异。公式如下：

l(y,y^)=−∑j=1qyjlog⁡y^j

4.交叉熵损失的导数：交叉熵损失函数的梯度是softmax模型分配的概率与真实标签（由独热标签向量表示）之间的差异。

∂∂ojl(y,y^)=softmax(o)j−yj

### 多层感知机

![](https://pic3.zhimg.com/v2-51253c4d63dc5bf0b753860c06f23906_1440w.jpg)

⼀个单隐藏层的多层感知机，具有5个隐藏单元

从线性到非线性：如果我们只是将输入通过仿射变换（线性变换和偏置）传递给隐藏层，然后再将隐藏层的输出通过仿射变换传递给输出层，那么整个模型仍然是一个仿射函数，这并没有比单层模型提供更多的表达能力。

为了使多层模型能够表达更复杂的函数，我们需要在隐藏层的仿射变换后应用一个非线性的激活函数 σ 。这样，模型的计算公式变为：

H=σ(XW(1)+b(1))

O=HW(2)+b(2)

### 常用的激活函数

ReLU的求导表现特别好：要么让参数消失，要么让参数通过。当输⼊为负时，ReLU函数的导数为0，而当输入为正时，ReLU函数的导数为1。这使得优化表现得更好，并且ReLU减轻了困扰以往神经网络的梯度消失问题。其数学定义如下：

ReLU(x)=max(x,0)

![](https://pica.zhimg.com/v2-8b0cde28a50cae785080b55a5f58fd76_1440w.jpg)

Sigmoid函数是一种常用的激活函数，它将实数输入映射到(0, 1)的范围内，因此也被称为挤压函数。sigmoid函数是⼀个自然的选择，因为它是⼀个平滑的、可微的阈值单元近似。它的数学定义如下：

sigmoid(x)=11+exp⁡(−x)

![](https://pic2.zhimg.com/v2-44974be0457b614d2d9128dcc9fa2b59_1440w.jpg)

双曲正切函数（tanh）是另一种常用的激活函数，它将实数输入映射到(-1, 1)的范围内。tanh函数是关于原点对称的。它的数学定义如下：

tanh(x)=1−exp⁡(−2x)1+exp⁡(−2x)

![](https://pic3.zhimg.com/v2-c7a9a3c49cd1786b828ba61d34ef47d2_1440w.jpg)

## 专题2 [K折交叉验证](https://zhida.zhihu.com/search?content_id=235880363&content_type=Article&match_order=1&q=K%E6%8A%98%E4%BA%A4%E5%8F%89%E9%AA%8C%E8%AF%81&zhida_source=entity)，欠（过）拟合，[权重衰退](https://zhida.zhihu.com/search?content_id=235880363&content_type=Article&match_order=1&q=%E6%9D%83%E9%87%8D%E8%A1%B0%E9%80%80&zhida_source=entity)，[暂退法](https://zhida.zhihu.com/search?content_id=235880363&content_type=Article&match_order=1&q=%E6%9A%82%E9%80%80%E6%B3%95&zhida_source=entity)

### K折交叉验证

K折交叉验证是一种评估模型性能的常用方法，特别是在数据量较少的情况下。这种方法将数据集分为K个不重叠的子集，每个子集大致具有相同的大小。然后，模型会进行K次训练和验证。**在每次迭代中，模型会在K-1个子集（即训练集）上进行训练，并在剩下的一个子集（即验证集）上进行验证。**这样，每个子集都有一次机会作为验证集，其余次数作为训练集。

K折交叉验证的主要优点是：它允许模型在多个不同的训练和验证集上进行训练和验证，这有助于提供对模型性能更稳健的估计。此外，它还允许我们使用所有的数据进行训练和验证，这在数据量较少的情况下特别有用。

### 欠（过）拟合

欠拟合（Underfitting）：当模型无法充分捕获数据中的模式和关系时，我们称模型为欠拟合。

过拟合（Overfitting）：当模型过度学习训练数据中的模式和噪声，以至于在新的、未见过的数据上表现不佳时，我们称模型为过拟合。

模型复杂性和数据集大小是影响模型过拟合和欠拟合的两个重要因素：

1. 模型过于复杂可能导致过拟合，过于简单可能导致欠拟合。
2. 数据集过小可能导致过拟合，而更大的数据集通常可以帮助模型更好地学习和泛化，减少过拟合。

![](https://pic4.zhimg.com/v2-0edf5d6d7b02e9731b295259fdf530f1_1440w.jpg)

模型复杂度对欠拟合和过拟合的影响

一般通过观察模型在训练集和验证集（或测试集）上的表现来判断模型是否出现欠拟合或过拟合。

- **如果模型在训练集和验证集上的表现都不好，那么模型可能出现了欠拟合。**这意味着模型可能过于简单，无法捕获数据中的所有相关模式。
- **如果模型在训练集上表现很好，但在验证集上表现差，那么模型可能出现了过拟合。**这意味着模型可能过于复杂，或者过度学习了训练数据中的噪声和异常值。

为了解决欠拟合和过拟合，我们可以尝试更换模型、调整模型复杂度、增加更多的训练数据、使用正则化技术、早停等策略。

### 权重衰退（weight decay）

权重衰减是一种正则化技术，用于防止模型过拟合。这种技术通过在模型的损失函数中添加一个惩罚项来实现，这个惩罚项与模型权重的平方值（L2范数）成正比，因而也被称为L2正则化。**使用L2范数的⼀个原因是它对权重向量的大分量施加了巨大的惩罚（如果L1就不如L2明显，因为L2是平方项，对大数敏感）。这使得我们的学习算法偏向于在大量特征上均匀分布权重的模型。如果使用L1惩罚则可能导致模型将权重集中在一小部分特征上，而将其他权重清除为零。**

L(w,b)+λ2∥w∥2

权重衰减的具体实现方式是在更新权重参数时，除了减去梯度之外，还要减去 λ 乘以当前权重。这就使得权重在每次更新时都会衰减一部分，因此得名"权重衰减"。注意L2正则化回归的小批量随机梯度下降更新表达式的 （）（1−nλ） 部分，这是**损失函数的正则化项的梯度**带来的。

w←(1−ηλ)w−η|B|∑i∈Bx(i)(w⊤x(i)+b−y(i))

### 暂退法（dropout）

> 暂退法的原始论⽂提到了⼀个关于有性繁殖的类：**神经网络过拟合与每一层都依赖于前一层激活值相关**，称这种情况为“共适应性”。作者认为，暂退法会破坏共适应性，就像有性生殖会破坏共适应的基因⼀样。

Dropout是一种在深度学习中常用的正则化技术，主要用于防止过拟合。**其基本思想是在训练过程中，随机丢弃一部分神经元（即设置其输出为0），在当前迭代中不参与前向传播和反向传播的过程。**以减少神经元之间的复杂共适应关系，增强模型的泛化能力。

![](https://pic2.zhimg.com/v2-b260c4dc4b3374f9492d0e50eca7ee95_1440w.jpg)

dropout前后的多层感知机

在标准dropout正则化中，通过按保留（未丢弃）的节点的分数进行规范化来消除每一层的偏差。换句话说，每个中间激活值 h 以dropout概率 p 由随机变量 h′ 替换，数学表达如下所示：

概率为其他情况h′={0概率为 ph1−p其他情况

根据此模型的设计，其期望值保持不变，即 E[h′]=h 。如果在dropout后不进行规范化，即不放大保留下来的激活值，那么网络的每一层的输出分布会随着概率 p的变化而变化，这可能导致训练过程不稳定，因为每层的输入分布都在不断变化。

- **暂退法仅在训练期间使用。**在训练时，Dropout层将根据指定的暂退概率随机丢弃上⼀层的输出（相当于下⼀层的输入）。在测试时，Dropout层仅传递数据。

```python3
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    # 在第⼀个全连接层之后添加⼀个dropout层
                    nn.Dropout(dropout1),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    # 在第⼆个全连接层之后添加⼀个dropout层
                    nn.Dropout(dropout2),
                    nn.Linear(256, 10))
```

## 专题3 前（反）向传播，[梯度消失和爆炸](https://zhida.zhihu.com/search?content_id=235880363&content_type=Article&match_order=1&q=%E6%A2%AF%E5%BA%A6%E6%B6%88%E5%A4%B1%E5%92%8C%E7%88%86%E7%82%B8&zhida_source=entity)，[batch normalization](https://zhida.zhihu.com/search?content_id=235880363&content_type=Article&match_order=1&q=batch+normalization&zhida_source=entity)

### 前（反）向传播

在前向传播过程中，每一层神经元都会接收到前一层神经元的输出作为输入，并通过激活函数进行处理，然后将结果传递给下一层神经元。

![](https://pic4.zhimg.com/v2-c451855cd0f70b176e420555420ff597_1440w.jpg)

前向传播的计算图

反向传播是一种在神经网络中计算参数梯度的方法，它是深度学习中的基础算法。该过程从前向传播开始，计算并存储每一层的输出，然后计算损失函数，接着按照相反的顺序计算每一层的梯度，最后使用这些梯度更新网络参数。在上图中的单隐藏层简单网络的参数是 W(1) 和 W(2) 。则反向传播的目的是计算梯度 ∂J/∂W(1) 和 ∂J/∂W(2) 。

∂J∂W(2)=∏(∂J∂o,∂o∂W(2))+∏(∂J∂s,∂s∂W(2))=∂J∂oh⊤+λW(2)

反向传播算法在计算每一层的梯度时，需要使用到前向传播中存储的激活值和后续层的梯度值。这些中间结果必须在内存中保留，以便在更新参数时使用。中间值的大小与网络层的数量和批量大小大致成正比，因此，使用更大的批量或更深的网络可能会导致内存不足的问题。也因此，训练比预测需要更多的内存。

### 梯度消失和爆炸

从数学视角分析梯度消失和爆炸的原因：考虑⼀个具有 L 层、输入为 x 和输出为 o 的深层⽹络。每⼀层 l 由变换 fl 定义，该变换的参数为权重W(l)，其隐藏变量是 h(l) （令 h(0)=x ）。则网络可以表示为：

因此h(l)=fl(h(l−1))因此o=fL∘⋯∘f1(x).

如果所有隐藏变量和输入都是向量，我们可以将输出变量 o 关于任何⼀组参数 W(l) 的梯度写为下式：

∂o∂W(l)=∂o∂h(L)∂h(L)∂h(L−1)…∂h(l+1)∂h(l)∂h(l)∂W(l)

- **梯度是** L−l **个矩阵** M(L)·...·M(l+1) **与梯度向量** v(l) **的乘积。当将太多的概率乘在一起时，这些问题经常会出现。初始化之后，矩阵** M(l) **可能具有各种各样的特征值。他们可能很小，也可能很大。他们的乘积也可能非常大，也可能非常小。这是导致梯度爆炸和梯度消失的共同原因。**
- ReLU激活函数有助于减轻梯度消失问题，是因为当输入为正时，其导数恒为1。因此，在正输入区间内，梯度不会随着网络深度的增加而衰减。
- 不稳定梯度威胁到我们优化算法的稳定性。要么是梯度爆炸问题：参数更新过大，破坏了模型的稳定收敛；要么是梯度消失问题：参数更新过小，在每次更新时几乎不会移动，导致模型无法学习。

### BN层：Batch Normalization

在模型训练过程中，批量规范化利用小批量的均值和标准差，不断调整神经网络的中间输出，这样可以使得网络在训练过程中更加稳定，加速收敛，同时也可以一定程度上防止过拟合。

批量规范化的基本思想是对每一层的输入进行规范化处理，使其满足均值为0，方差为1的标准正态分布，然后通过学习参数来恢复网络层需要的原始分布。对于来自小批量 B 的输入 x ，批量规范化 BN(x) 可以按照以下表达式进行：

BN(x)=γ⊙x−μ^Bσ^B+β

μ^B=1|B|∑x∈Bx

σ^B2=1|B|∑x∈B(x−μ^B)2+ϵ

注意到加上 ϵ 的原因是，在实际操作中，如果批量的方差 ( σ^B2 ) 非常小或者为零（比如，当批量中的所有样本都相等时），直接进行除法会导致除以零的错误。加上一个小的 ϵ 可以保证分母不会为零。

**实践中BN层的位置：**批量规范化层置于全连接层中的仿射变换和激活函数之间。先规范再激活。

h=ϕ(BN(Wx+b))

**预测过程中的BN层：**常用的方法是通过移动平均估算整个训练数据集的样本均值和方差，并在预测时使用它们得到确定的输出。

## 专题4 深度学习计算

### 自定义块

- 实现一个简单的MLP类：

```python3
class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
        super().__init__()
        self.hidden = nn.Linear(20, 256) # 隐藏层
        self.out = nn.Linear(256, 10) # 输出层

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
        return self.out(F.relu(self.hidden(X)))
```

- 实现一个MySequential类：

```python3
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # 变量_modules中。_modules的类型是OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X
```

- 实现一个带参数的MyLinear层：

```python3
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
```

## 专题5 [CNN](https://zhida.zhihu.com/search?content_id=235880363&content_type=Article&match_order=1&q=CNN&zhida_source=entity)，CNN中的注意力，残差连接

### 基本概念

输入为n，卷积核大小为k，计算输出的大小： (nh−kh+1)×(nw−kw+1).

![](https://pic1.zhimg.com/v2-d994c14127ae12fbd6874516d6606150_1440w.jpg)

输⼊张量和核张量通过互相关运算产⽣输出张量

输入为n，卷积核大小为k，填充为p，计算输出的大小： （（nh−kh+ph+1)×(nw−kw+pw+1)

![](https://pic4.zhimg.com/v2-64296cc9e13b02ebd63b15cbca8257cd_1440w.jpg)

带填充的⼆维互相关

加入步幅，计算输出的大小： ⌊(nh−kh+ph+sh)/sh⌋×⌊(nw−kw+pw+sw)/sw⌋

![](https://pic3.zhimg.com/v2-b2db675d5903efc10db4dc48c40d5d88_1440w.jpg)

垂直步幅为 3，⽔平步幅为 2 的⼆维互相关运算

![](https://pic3.zhimg.com/v2-ebf6a2e2f9322bc2a0de6e9bae68a20c_1440w.jpg)

最大池化

- 在CNN中，神经元的感受野是指：在前向传播过程中，可能影响该神经元计算的所有输入元素，这些元素可以来自所有先前的层。随着网络深度的增加，神经元的感受野也会相应地扩大（其元素被反复地卷积过）。**因此，通过增加网络的深度，我们可以使神经元能够感知到输入数据的更大范围，从而检测到更广区域的特征。**

### 外部例子：CBAM注意力模块

CBAM（Convolutional Block Attention Module）是一种用于卷积神经网络的注意力模块，它通过关注重要的特征来提高网络的性能。CBAM依次集成了通道注意力模块（Channel Attention Module）和空间注意力模块（Spatial Attention Module），这两个模块分别聚焦于不同的特征维度。

**下图中，水平线的长度代表通道数，如果垂直地切一刀，就是一个通道的空间信息。注意力方面，可以先简单地认为是其输出是输入的加权和。**

- 通道注意力模块：这部分主要负责获取每个通道的重要性权重。输入特征图`F`首先通过全局平均池化（AvgPool）和全局最大池化（MaxPool）进行处理，这两种池化操作分别捕获不同的统计信息，均在空间维度上进行压缩，从而只保留通道信息。两个池化结果被送入一个共享的多层感知机（MLP），MLP有两个全连接层，其中第一个全连接层的作用是降维，第二个全连接层是升维，以恢复到原始通道数目。然后将两个结果相加获得Channel Attention的输出结果，包含了每个通道的重要性权重。
- 空间注意力模块：这部分主要负责获取空间位置的重要性权重。具体来说，它先通过基于通道的池化操作（类似于点状卷积，改变特征图的通道数，保持空间维度不变）将输入的特征图在通道方向上进行压缩，然后通过一个更大尺寸的卷积操作（比如7x7），获得Spatial Attention的特征图，表示每个空间位置的重要性权重。

得到注意力图`M`后，它会被用来加权输入特征图`F`的每个通道。这是通过逐元素乘法（Hadamard积）完成的，即每个通道的特征图都乘以其对应的通道注意力权重。这样，那些被认为更重要的通道会被强化，而不那么重要的通道则会被削弱。假设我们有一个特征图`F`，其尺寸为`H*W*C`，其中`H`是高度，`W`是宽度，`C`是通道数。

**通道注意力模块**：

- 对于通道注意力模块，我们首先进行全局平均池化和全局最大池化操作。这两个操作将特征图在空间维度上进行压缩，即对每个通道的`H*W`的特征值进行平均或取最大值。结果是两个`1*1*C`的向量，每个向量表示特征图在该通道上的全局统计信息。
- 这两个`1*1*C`的向量分别通过一个共享的多层感知机（MLP），MLP包含两个全连接层，第一个全连接层通常用于降维（例如降到`C/r`），第二个全连接层用于升维（恢复到`C`），其中`r`是一个超参数，用于控制降维的程度。
- 然后，两个MLP的输出向量进行逐元素相加，并通过一个Sigmoid函数，得到一个`1*1*C`的通道注意力图`M_c`，这个向量中的每个值代表相应通道的重要性权重。

**空间注意力模块**：

- 对于空间注意力模块，我们首先对调整过通道权重的特征图`F'`（尺寸依然是`H*W*C`）进行逐通道的最大池化和平均池化，得到两个`H*W*1`的特征图，这两个特征图分别表示每个空间位置在所有通道上的最大值和平均值。
- 这两个`H*W*1`的特征图在通道维度上进行堆叠，形成一个`H*W*2`的特征图，然后通过一个卷积层（使用较大的卷积核，如7x7），以捕获空间上下文信息。
- 最后，通过一个Sigmoid函数得到一个`H*W*1`的空间注意力图`M_s`，这个特征图中的每个值代表相应空间位置的重要性权重。

**特征图的加权和融合**：

- 通道注意力图`M_c`通过逐元素乘法应用于原始特征图`F`的每个通道，得到一个通道加权的特征图`F'`（尺寸`H*W*C`）。
- 空间注意力图`M_s`通过逐元素乘法应用于`F'`的每个空间位置，得到最终的加权特征图`F''`（尺寸`H*W*C`）。
- 这个`F''`包含了经过通道和空间校准的特征，可以被传递到网络的下一层或用于后续的任务处理。

![](https://pic3.zhimg.com/v2-fc94dd00ac0f32cd219d460bf88afd90_1440w.jpg)

Diagram of each attention sub-module

### 例子：残差连接

- 假设存在一类特定的神经网络架构F，它包括学习速率和其他超参数设置。对于所有f∈F，存在一些参数集（例如权重和偏置），这些参数可以通过在合适的数据集上进行训练而获得。如果是f*∈ F，那我们可以轻而易举地训练得到它，但通常我们不会那么幸运。相反，我们将尝试找到⼀个函数f**，这是我们在F中的最佳选择。

![](https://pic4.zhimg.com/v2-fd39fc869e269f925089fa7d76eee4a9_1440w.jpg)

对于非嵌套函数类，较复杂（由较大区域表示）的函数类不能保证更接近“真”函数（f∗ ）

- **如果一个网络结构F'包含了原来的网络结构F（即F ⊆ F'），那么F'至少可以达到F的性能，因为F'可以通过设置一部分参数，使得自身退化为F，即将新添加的层训练成恒等映射f(x) = x。在这种情况下，选择F'是安全的，因为它至少不会比F差。**然而，如果F'不包含F，那么F'可能会比F更差。这是因为F'可能无法表达F能够表达的一些函数，从而导致性能下降。同时， 由于新模型可能得出更优的解来拟合训练数据集，因此添加层似乎更容易降低训练误差。

![](https://pic3.zhimg.com/v2-961c3b872f2d03ab1b444a1559c8da76_1440w.jpg)

残差块

ResNet的核心思想是引入了“残差学习”（Residual Learning）来解决深度网络训练困难的问题。在传统的神经网络中，每一层都直接拟合一个目标函数。但是随着网络层数的增加，这种直接拟合的方式会导致梯度消失或梯度爆炸，从而使得网络难以训练。

残差学习的思想是让网络层拟合一个残差函数，即目标函数与输入的差值，而不是直接拟合目标函数。其核心优点为：

|   |
|---|
|缓解梯度消失：在深度神经网络中，梯度消失是一个常见的问题，它会阻碍网络的训练。当网络深度增加时，梯度在反向传播过程中可能会变得非常小，这使得权重更新变得困难。ResNet通过引入残差块和跳跃连接来解决这个问题。跳跃连接允许梯度直接反向传播到较早的层，从而缓解了梯度消失问题。|
|更加易于优化：残差映射通常比直接拟合原始映射更容易优化。这是因为，如果理想的输出是输入的恒等映射，那么残差映射（即输出与输入的差）就接近于零。这使得网络可以更容易地学习恒等映射的细微变化。从图中看，就是残差块的输出从f（x）变为了f（x）- x，区别就在于神经网络的目标从直接拟合输出f（x）变成了拟合输入输出的差值，即f（x）- x。让网络专注于学习输入和输出之间的差异，而不是直接学习输出，这样做通常会使得优化过程更加容易，尤其是在深层网络中。|
|跨层特征融合：ResNet的跳跃连接不仅解决了梯度消失问题，还使得网络可以在不同的层级之间共享和融合特征。这种跨层的特征融合有助于提高模型的表现力和准确性。|

## 专题6 序列模型，语言模型，RNN

> 循环神经网络（RNN）通过引入状态变量来存储过去的信息和当前的输入，从而确定当前的输出。这种结构使得RNN非常适合处理序列信息，因为它可以捕捉到序列中的时间依赖性。这与卷积神经网络（CNN）的工作方式形成了对比，CNN主要用于处理空间信息，如图像等。

### 序列模型概念

在序列模型中，预测 xt∼P(xt|xt−1,...,x1) 表示在给定前 t−1 个元素的条件下，第 t 个元素的概率分布。

- 自回归模型：这种模型假设当前的观测值只依赖于过去的一定数量（ τ ）的观测值。这样的好处是模型的参数数量始终是固定的，即使用观测序列 xt−1,...,xt−τ 。这使得我们可以训练一个深度网络。自回归模型的名字来源于它对自身的回归，即当前的观测值是过去观测值的函数。
- 隐变量自回归模型：这种模型除了考虑过去的观测值，还保留了一个对过去观测的总结（隐状态）。在这个模型中，我们不仅预测当前的观测值，而且还更新隐状态。这个模型被称为"隐变量自回归模型"，因为它包含了未被观测到的隐状态。在这个模型中，当前的观测值是基于隐状态的条件概率，隐状态则是基于过去的隐状态和过去的观测值的函数。如此一来，计算该概率分布的数学表达即：

P(xi|x1,…,xi−1)=softmax(Whi−1+b)

![](https://pica.zhimg.com/v2-387d10150e64a53147a456458e8b7d5a_1440w.jpg)

隐变量自回归模型

马尔可夫模型是一种统计模型，它假设系统在下一时刻的状态只依赖于它在当前时刻的状态，而与它在过去的所有状态都无关。这种性质被称为马尔可夫性质或马尔可夫条件。

### 例子：拟合正弦函数

在时间序列分析中，我们通常将序列数据转换为**特征-标签对**，以便于机器学习模型进行学习。这里的"嵌入维度" τ 是指我们用来预测当前数据点的历史数据点的数量。我们的特征 xt 是一个包含了前 τ 个数据点的向量，这个向量提供了预测 yt 所需要的上下文信息。

然而，对于序列的前 τ 个数据点，我们没有足够的历史记录来构建这样的特征向量。因此，如果我们有 N 个数据点，我们只能构建 N−τ 个特征-标签对。这就是为什么我们的数据样本比原始序列少了 τ 个的原因。

- 例如，如果我们有一个时间序列 [x1,x2,x3,x4,x5] ，并且 τ=2 ，那么我们可以构建的特征-标签对包括：特征 x3=[x1,x2] ，标签 y3=x3 ；特征 x4=[x2,x3] ，标签 y4=x4 ；特征 x5=[x3,x4] ，标签 y5=x5 。但我们不能为 x1 和 x2 构建特征，因为它们前面没有足够的数据点。代码实现：

```python3
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 数据生成
T = 1000
time = torch.arange(1, T + 1, dtype=torch.float32)  # torch.Size([1000])

# (T,)是一个元组，它只包含一个元素,表示生成一个形状为(T,)的一维张量
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))  # torch.Size([1000])

# 数据预处理
tau = 4
features = torch.zeros((T - tau, tau))  # torch.Size([996, 4])
for i in range(tau):
    features[:, i] = x[i: T - tau + i]  # features[:, i]是取第一列
labels = x[tau:].reshape((-1, 1))  # torch.Size([996, 1])

# 定义数据加载函数
def load_array(data_arrays, batch_size, is_train=True):
    dataset = TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size, shuffle=is_train)

# 加载数据
batch_size, n_train = 16, 600
train_iter = load_array((features[:n_train], labels[:n_train]), batch_size)  # DataLoader object

# 定义模型（单层网络）
def get_net():
    net = nn.Sequential(nn.Linear(4, 10),  
                        nn.ReLU(),
                        nn.Linear(10, 1)) 
    net.apply(init_weights)
    return net

# 初始化网络权重的函数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

# 定义损失函数
loss = nn.MSELoss(reduction='none')

# 定义训练函数
def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:  # X: torch.Size([batch_size, 4]), y: torch.Size([batch_size, 1])
            trainer.zero_grad()
            l = loss(net(X), y)  # l: torch.Size([batch_size, 1])
            l.sum().backward()
            trainer.step()
        print(f'epoch {epoch + 1}, loss: {sum(loss(net(X), y)).item() / len(y):f}')

# 训练模型
net = get_net()
train(net, train_iter, loss, 5, 0.01)

# 预测
onestep_preds = net(features) # torch.Size([996, 1])
plt.plot(time, x, label='data')
plt.plot(time[tau:], onestep_preds.detach().numpy(), label='1-step preds')
plt.xlim([1, 1000])
plt.legend()
plt.show()

# k步预测
max_steps = 64
features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))  # torch.Size([933, 68])
# 列i（i<tau）是来⾃x的观测，其时间步从（i）到（i+T-tau-max_steps+1）
for i in range(tau):
    features[:, i] = x[i: i + T - tau - max_steps + 1]
# 列i（i>=tau）是来⾃（i-tau+1）步的预测，其时间步从（i）到（i+T-tau-max_steps+1）
for i in range(tau, tau + max_steps):
    features[:, i] = net(features[:, i - tau:i]).reshape(-1)

steps = (1, 4, 16, 64)
for i in steps:
    plt.plot(time[tau + i - 1: T - max_steps + i], features[:, tau + i - 1].detach().numpy(), label=f'{i}-step preds')
plt.xlim([5, 1000])
plt.legend()
plt.show()
```

![](https://picx.zhimg.com/v2-71c21df2c4823b6dfdfdffb4d3306ed5_1440w.jpg)

单步预测

![](https://pic1.zhimg.com/v2-29e2e1ad55cfdfe2149fa4fffebb504a_1440w.jpg)

k步预测

对于直到时间步t的观测序列，其在时间步 t+k 的预测输出是“k步预测”。**随着我们对预测时间k值的增加，会造成误差的快速累积和预测质量的极速下降。**

### 语言模型概念

**文本预处理的一般步骤**

1、加载文本：

将文本数据从文件、数据库或网络资源中读取到内存。清理数据，移除无关的元素，如HTML标签、特殊字符等。

2、文本分词

将连续的文本字符串拆分成更小的单元，通常是单词、短语或字符。这个过程称为分词。分词可以根据空格、标点符号等来进行，也可以使用更复杂的方法，如使用NLP库的分词器。

3、构建词表：

创建一个词表，列出文本中出现的所有独特词元，并为它们分配一个唯一的数字索引。有时会添加特殊词元，如`<pad>`用于序列填充，`<unk>`用于未知词元，`<sos>`和`<eos>`分别表示序列的开始和结束。

4、文本编码：

将分词后的文本转换为数字索引序列，以便模型可以处理。这通常涉及到one-hot编码或词嵌入（embedding）的使用。One-hot编码为每个词元创建一个很长的向量，其中只有一个位置是1，其余都是0。词嵌入则是将词元映射到一个固定大小的、更稠密的向量，通常是通过预训练的词嵌入模型（如Word2Vec）获得。

**建模**

在自然语言处理中，我们经常需要对文档或词元序列进行建模。如果我们在单词级别对文本数据进行词元化，我们可以依赖于序列模型的分析。基本的概率规则如下。

P(x1,x2,...,xT)=∏t=1TP(xt|x1,...,xt−1)

例如，一个包含四个单词的文本序列的概率可以表示为：

P(deep,learning,is,fun)=P(deep)P(learning|deep)P(is|deep,learning)P(fun|deep,learning,is)

**训练**

为了训练语言模型，我们需要计算单词的概率，以及给定前几个单词后出现某个单词的条件概率。这些概率本质上就是语言模型的参数。我们通常假设训练数据集是一个大型的文本语料库。训练数据集中词的概率可以根据给定词的相对词频来计算。例如，估计值 P^(deep) 可以计算为任何以单词“deep”开头的句子的概率。一种方法是统计单词“deep”在数据集中的出现次数，然后将其除以整个语料库中的单词总数。这种方法效果不错，特别是对于频繁出现的单词。然后，我们可以尝试估计：

P^(learning|deep)=n(deep,learning)n(deep)

其中 n(x) 和 n(x,x′) 分别是单个单词和连续单词对的出现次数。但由于连续单词对“deep learning”的出现频率要低得多，所以估计这类单词正确的概率要困难得多。特别是对于一些不常见的单词组合，要想找到足够的出现次数来获得准确的估计可能都不容易。

**n元语法**

如果 P(xt+1|xt,...,x1)=P(xt+1|xt) ，则序列上的分布满足一阶马尔可夫性质。阶数越高，对应的依赖关系就越长。这种性质推导出了许多可以应用于序列建模的近似公式：

P(x1,x2,x3,x4)=P(x1)P(x2)P(x3)P(x4)

P(x1,x2,x3,x4)=P(x1)P(x2|x1)P(x3|x2)P(x4|x3)

P(x1,x2,x3,x4)=P(x1)P(x2|x1)P(x3|x1,x2)P(x4|x2,x3)

通常，涉及一个、两个和三个变量的概率公式分别被称为一元语法（unigram）、二元语法（bigram）和三元语法（trigram）模型。

> 注意到，一个简单的bigram（2-gram）模型会考虑一个单词出现的条件概率依赖于它前面的一个单词。如果我们有一个词汇表（vocabulary）的大小为(V)，那么对于bigram模型来说，我们需要存储的概率数量大约是(V^2)，因为对于词汇表中的每个单词，我们需要知道它后面跟着词汇表中每个其他单词的概率。随着n的增加，模型需要存储的概率数量会呈指数增长（大约是(V^n)），这是因为每增加一个单词，我们就需要乘以词汇表的大小来计算新的组合数量。

**齐普夫定律（Zipf's law）：**

语言中单词序列遵循齐普夫定律，同时很多n元组出现次数较少，说明语言中存在相当多的结构。

基于深度学习的模型在语言建模方面具有优势，因为它们能够处理稀疏数据并从中学习有用的特征和模式，而不像拉普拉斯平滑方法那样受限。

![](https://pic2.zhimg.com/v2-568e844e6a79d87a6106842f7737b481_1440w.jpg)

n元语法的词元频率

### RNN概念

> n元语法模型的局限性：在n元语法模型中，单词 xt 在时间步t的条件概率仅取决于前⾯n − 1个单词。对于时间步t − (n − 1)之前的单词，如果我们想将其可能产⽣的影响合并到xt上，需要增加n，然而模型参数的数量也会随之呈指数增长，因为词表V需要存储 |V|n个数字。

注意，RNN的参数是固定的。无论输入序列的长度如何，RNN的参数（包括输入到隐藏层的权重、隐藏层到隐藏层的权重、偏置项以及可能的隐藏层到输出层的权重等）都保持不变。有一些图示是按照时间步展开的，也就是进行了多次输入输出，横轴上的隐藏状态数并不是隐藏单元数，这很容易误导初次理解。RNN在不同的时间步上使用的是相同的参数（权重和偏置）。

**隐状态变量的长度（或者说维度）是由隐藏层的单元数决定的，而与输入序列的长度无关。每个隐藏单元都会有一个对应的隐状态值，因此隐状态变量的维度就等于隐藏层的单元数。这个隐状态变量会在每个时间步被更新，并用于计算下一个时间步的输出和新的隐状态。**

![](https://pic2.zhimg.com/v2-98d1e4900e12c0c272d32f164ef2baad_1440w.jpg)

（图源：忆臻）t-1时刻的隐变量是被保存下来的，它在网络中实体和t时刻的隐变量是同一个

在n元语法模型中，单词 xt 在时间步 t 的条件概率仅取决于前面 n - 1 个单词。对于时间步 t - (n - 1) 之前的单词，如果我们想将其可能产生的影响合并到 xt 上，需要增加 n，然而模型参数的数量也会随之呈指数增长，因为词表 V 需要存储 |V|n 个数字。因此，与其将 P(xt|xt−1,...,xt−n+1) 建模，不如使用隐变量模型，它可以捕获序列中的长期依赖关系，而不仅仅是前 n - 1 个单词的信息：

P(xt|xt−1,…,x1)≈P(xt|ht−1)

其中 ht−1 是隐状态（hidden state），也称为隐藏变量（hidden variable），它存储了到时间步 t - 1 的序列信息。通常，我们可以基于当前输入xt 和先前隐状态 ht−1 来计算时间步 t 处的任何时间的隐状态：

ht=f(xt,ht−1)

在下方提供的公式中，新增了一项 Ht−1∗Whh ，这一项表示前一个时间步的隐状态与一个权重矩阵 Whh 的乘积，这个乘积结果与当前时间步的输入一起决定了当前时间步的隐状态。这样的设计使得隐状态能够在时间步之间传递信息，实现了序列信息的记忆功能：

Ht=ϕ(XtWxh+Ht−1Whh+bh)

Ot=HtWhq+bq

  

![](https://pic3.zhimg.com/v2-15ec484dc1ba8aca4ccb03fdebec042e_1440w.jpg)

按时间步展开，具有隐状态的循环神经⽹络

### 例子：字符级语言模型

![](https://picx.zhimg.com/v2-afbb5b113ffa9e58a6c94dfa7a017ed9_1440w.jpg)

Char RNN

交叉熵函数：l(y,y^)=−∑j=1qyjlog⁡y^j

**困惑度**

困惑度是一个概率模型对给定测试数据集的预测能力的度量，直观上可以理解为模型在预测一个序列时的平均分支数或选择数。一个好的语言模型能够较准确地预测接下来可能出现的单词，因此具有较低的困惑度。

PP(W)=exp⁡(−1N∑i=1Nlog⁡P(wi|w1,…,wi−1))

P(wi|w1,…,wi−1)=softmax(Whi−1+b)

**梯度剪裁**

对于长度为T的序列，我们在迭代中计算这 T 个时间步上的梯度，将会在反向传播过程中产生长度为O(T)的矩阵乘法链。梯度剪裁通过限制梯度的大小来解决梯度爆炸问题，确保梯度在一个合理的范围内，从而避免梯度过大导致的问题。这里的 g 表示梯度向量，( ||g|| )是梯度向量的范数（例如L2范数），而( θ )是预先设定的阈值。如果梯度的范数超过了阈值，这个公式将梯度向量缩放到阈值大小。

g←min(1,θ‖g‖)g

## 专题7 GRU，LSTM，encoder-decoder架构，seq2seq

### 门控记忆单元（GRU）

GRU模型有专门的机制来确定应该何时更新隐状态，以及应该何时重置隐状态。这些机制是可学习的。门控循环单元具有以下两个显著特征：

- 重置门有助于捕获序列中的短期依赖关系。
- 更新门有助于捕获序列中的长期依赖关系。

![](https://pic4.zhimg.com/v2-ec85f1468ea548620424dfa988ea6809_1440w.jpg)

计算门控循环单元模型中的隐状态

**GRU中的四个计算公式(符号⊙是Hadamard积，按元素乘积)：**

1. Rt=σ(XtWxr+Ht−1Whr+br)
2. Zt=σ(XtWxz+Ht−1Whz+bz)
3. Ht~=tanh(XtWxh+(Rt⊙Ht−1)Whh+bh)
4. Ht=Zt⊙Ht−1+(1−Zt)⊙Ht~

### 长短期记忆网络（LSTM）

LSTM是一种特殊的RNN，主要由遗忘门、输入门和输出门三部分构成。

1. 遗忘门负责决定哪些信息从细胞状态中丢弃，其根据当前输入和上一步的隐藏状态生成一个0到1之间的值，通过乘以细胞状态实现遗忘功能。
2. 输入门则决定哪些新信息应添加到细胞状态中，它由一个sigmoid层和一个tanh层组成，sigmoid层决定哪些值需要更新，tanh层创建新的候选值向量。
3. 输出门负责决定细胞状态的哪部分用于计算下一个隐藏状态，它首先通过sigmoid层决定细胞状态的哪些部分需要输出，然后将细胞状态通过tanh函数处理并与sigmoid门的输出相乘，产生最终的输出。

![](https://pic1.zhimg.com/v2-fd729c404b85c1a880e3747218e7c428_1440w.jpg)

在长短期记忆模型中计算隐状态

**LSTM的相关计算公式：**

1. It=σ(XtWxi+Ht−1Whi+bi)
2. Ft=σ(XtWxf+Ht−1Whf+bf)
3. Ot=σ(XtWxo+Ht−1Who+bo)
4. C~t=tanh(XtWxc+Ht−1Whc+bc)
5. Ct=Ft⊙Ct−1+It⊙C~t
6. Ht=Ot⊙tanh(Ct)

### encoder-decoder架构

> 数据集加载：我们可以通过截断（truncation）和填充（padding）方式实现一次只处理一个小批量的文本序列。假设同一个小批量中的每个序列都应该具有相同的长度num_steps，那么如果文本序列的词元数目少于num_steps时，我们将继续在其末尾添加特定的“<pad>”词元，直到其长度达到num_steps；反之，我们将截断文本序列时，只取其前num_steps个词元，并且丢弃剩余的词元。这样，每个文本序列将具有相同的长度，以便以相同形状的小批量进行加载。

机器翻译是序列转换模型的一个核心问题，其输入和输出都是长度可变的序列。为了处理这种类型的输入和输出，我们可以设计一个包含两个主要组件的架构：

1. **编码器（Encoder）：编码器接受一个长度可变的序列作为输入，并将其转换为具有固定形状的编码状态。编码器的主要任务是理解和编码输入序列的信息。**
2. **解码器（Decoder）：解码器将固定形状的编码状态映射到长度可变的序列。解码器的主要任务是根据编码器的输出生成一个新的序列。**

这种架构被称为编码器-解码器架构。在机器翻译中，编码器可能会接受一种语言的句子，然后解码器会生成另一种语言的句子。这种架构也被广泛应用于其他序列生成任务，如语音识别和文本摘要等。

![](https://pic4.zhimg.com/v2-034c4614fe63168cac13ed399d3f3fb1_1440w.jpg)

编码器-解码器架构

### seq2seq

![](https://pic3.zhimg.com/v2-68650bc729cb741437a16fa4d188c7ec_1440w.jpg)

循环神经网络编码器-解码器模型中的层

**嵌入层**

嵌入层的主要作用是将离散的文本数据（如单词、字符或子词）转换为连续的向量表示。这些向量表示可以捕捉词汇之间的语义关系，从而使模型能够更好地理解和处理文本数据。在这里，嵌⼊层的权重是⼀个矩阵，其行数等于输⼊词表的大小（vocab_size），其列数等于特征向量的维度（embed_size）。对于任意输⼊词元的索引 i ，嵌⼊层获取权重矩阵的第 i 行以返回其特征向量。

**编码器实现**

RNN的循环层所做的变换为ht=f(xt,ht−1)。然后编码通过一个函数q将所有隐状态转换为上下文变量 c=q(h1,...,hT) ，例如取 q(h1,...,hT)=hT ，上下文变量仅仅是输⼊序列在最后时间步的隐状态：

> `X = X.permute(1, 0, 2)`的含义是将输入张量`X`的维度重新排列。具体来说，如果`X`的原始维度是`(batch_size, num_steps, embed_size)`，`permute(1, 0, 2)`将会把这些维度重新排列为`(num_steps, batch_size, embed_size)`。

```python3
class Seq2SeqEncoder(nn.Module):
    """用于序列到序列学习的循环神经网络编码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
        super().__init__()
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # GRU层
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X, *args):
        # 输出'X'的形状：(batch_size, num_steps, embed_size)
        X = self.embedding(X)
        # 在循环神经网络模型中，第一个轴对应于时间步
        X = X.permute(1, 0, 2)
        # 如果未提及状态，则默认初始化为0
        output, state = self.rnn(X)
        # output的形状：(num_steps, batch_size, num_hiddens)
        # state的形状：(num_layers, batch_size, num_hiddens)
        return output, state
```

**解码器实现**

st′=g(yt′−1,c,st′−1) **,函数** g **表示解码器隐藏层的变换，它接收三个输入：上一个时间步的输出** yt′−1 **，上下文变量** c **（通常是编码器的最终隐藏状态，包含输入序列的信息），以及上一个时间步的隐藏状态** st′−1 **。**

通过这个函数，解码器可以在每个时间步生成新的隐藏状态 st′ ，并根据这个隐藏状态来预测当前时间步的输出。获得解码器的隐状态之后，可以使用输出层和softmax操作来计算在时间步 t′ 时输出 yt′ 的条件概率分布 ：P(yt′|y1,...,yt′−1,c) 。

- 当实现解码器时，直接使用编码器最后⼀个时间步的隐状态来初始化解码器的隐状态。这要求使用循环神经网络实现的编码器和解码器具有相同数量的层和隐藏单元。为了进⼀步包含经过编码的输⼊序列的信息，上下文变量在所有的时间步与解码器的输入进行拼接（concat）。为了预测输出词元的概率分布，在循环神经网络解码器的最后⼀层使用全连接层来变换隐状态。

```python3
class Seq2SeqDecoder(nn.Module):
    """用于序列到序列学习的循环神经网络解码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super().__init__(**kwargs)  # 简化的super调用
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        # 输出'X'的形状：(batch_size, num_steps, embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        # 广播context，使其具有与X相同的num_steps
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        # output的形状：(batch_size, num_steps, vocab_size)
        # state的形状：(num_layers, batch_size, num_hiddens)
        return output, state
```

**损失函数**

- 在每个时间步，解码器会预测输出词元的概率分布，类似于语言模型的操作。为了优化这个过程，可以使用softmax来获得概率分布，并通过计算交叉熵损失函数来进行优化。为了遮蔽不相关的预测，可以对softmax交叉熵损失函数进行扩展。**最初，所有预测词元的掩码都被设置为1。一旦给定了有效长度，与填充词元对应的掩码将被设置为0。最后，将所有词元的损失乘以掩码，以过滤掉损失中填充词元产生的不相关预测。**

> 数据集加载：我们可以通过截断填充方式实现一次只处理一个小批量的文本序列。但是，我们应该将填充词元的预测排除在损失函数的计算之外。 可以使用sequence_mask函数通过零值化屏蔽不相关的项，以便后⾯任何不相关预测的计算都是与零的乘积，结果都等于零。例如，如果两个序列的有效长度（不包括填充词元）分别为1和2，则第⼀个序列的第⼀项和第⼆个序列的前两项之后的剩余项将被清除为零。

```python3
def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    # 创建一个范围张量，并通过比较操作生成掩码
    mask = torch.arange((maxlen), dtype=torch.float32,device=X.device)[None, :] < valid_len[:, None]
    print(torch.arange((maxlen))[None, :])
    print(valid_len[:, None])
    print(mask)

    X[~mask] = value
    return X

X = torch.tensor([[1, 2, 3], [4, 5, 6]])
# X 是一个包含两个序列的张量，每个序列长度为3

print(sequence_mask(X, torch.tensor([1, 2])))
```

这段代码需要一定理解，先从`torch.tensor([1, 2])`谈起。首先，`torch.tensor([1, 2])`中的`[1, 2]`表示每个序列的有效长度。然后转到`mask = torch.arange((maxlen)`，mask矩阵的初始化就是假设序列所有位置都有效。再使用了`[None, :]`在这里用来增加一个列的维度，`[:, None]`增加一个行的维度，这样才能利用广播机制正确地比较和应用掩码。`X[~mask] = value` 表示将所有需要被屏蔽的元素赋值为 `value`，value的值默认为0。输出结果如下：

```text
tensor([[0, 1, 2]])
tensor([[1],
        [2]])
tensor([[ True, False, False],
        [ True,  True, False]])
tensor([[1, 0, 0],
        [4, 5, 0]])
```

> Decoder的输入，训练和测试时是不一样的。示意图中就是从解码器拉出来，被指向解码器的每一个时间步的箭头，是上下文变量。

- **训练：采用强制教学方法（teacher forcing）。在这种方法中，序列开始词元（“<bos>”）在初始时间步被输入到解码器中，我们使用真实的目标文本作为输入，即将标准答案作为解码器的输入。在每个时间步，解码器会根据当前正确的输出词和上一步的隐状态来预测下一个输出词。这样做的好处是，在训练过程中，模型可以更容易地学习到正确的输出序列，减轻“一步错，步步错”的误差爆炸问题，加快模型的收敛速度。**

![](https://pic3.zhimg.com/v2-4c5a039f5d67f0e0c981f9d6194a8068_1440w.jpg)

使⽤循环神经网络编码器和循环神经⽹络解码器的序列到序列学习

- **预测：每个解码器当前时间步的输入都是来自前一个时间步的预测词元。**与训练过程类似，序列开始词元（“<bos>”）在初始时间步被输入到解码器中。预测过程会一直进行，直到输出序列的预测遇到序列结束词元（“<eos>”），此时预测过程结束。

![](https://pic2.zhimg.com/v2-833e098a3581d74cbfd5a211a6a218a1_1440w.jpg)

使⽤循环神经⽹络编码器-解码器逐词元地预测输出序列

评估：BLEU是⼀种常用的评估方法，它通过测量预测序列和标签序列之间的n元语法的匹配度来评估预测。

**搜索策略**

贪心搜索：贪心搜索是一种简单直接的策略。**在每一步生成序列的过程中，模型会计算下一个最可能的词元（token，可以是词或字符等）的概率，并选择概率最高的那个词元作为当前步骤的输出。**然后，模型继续基于当前已生成的序列来预测下一个词元，直到整个序列生成完毕。贪心搜索的计算量为 O(|Y|T′) 。

![](https://pica.zhimg.com/v2-55c2f3ce73985b7a7e6ee6d7ffae7ca0_1440w.jpg)

贪心编码

穷举搜索是一种理论上的搜索方法。**它会计算所有可能输出序列的条件概率，并选择概率最高的序列作为最终输出。**由于需要计算所有可能的序列，其计算量非常巨大，为 O(|Y|T′) 。

束搜素：束搜索是贪心搜索的一个改进版本，其思路就是权衡贪心搜索和穷举搜索。**它有一个超参数，名为束宽k。在时间步1，我们选择具有最高条件概率的k个词元。**这k个词元将分别是k个候选输出序列的第一个词元。在随后的每个时间步，基于上一个时间步的k个候选输出序列，我们将继续从k |Y|个可能的选择中挑出具有最高条件概率的k个候选输出序列。 束搜素的计算量为O(k|Y|T′) 。

![](https://pic2.zhimg.com/v2-9f3b3d68065b6c3af02c35d8082fee51_1440w.jpg)

## 专题8 优化算法

> 优化的目标是找到能使损失函数最小化的参数。这通常涉及到在训练数据上尽可能减少误差，也就是我们所说的训练误差。然而，深度学习的目标不仅仅是最小化训练误差，更重要的是最小化泛化误差，也就是模型在未见过的数据上的预测误差。

### 挑战：局部最小值、鞍点和梯度消失

深度学习模型的目标函数通常是非凸的，这意味着它可能有许多局部最优解。当优化算法找到一个局部最优解时，由于在该点附近梯度接近或等于零，优化算法可能会停止在那里，而无法找到全局最优解。除了局部最小值之外，鞍点是梯度消失的另⼀个原因。鞍点是指函数的所有梯度都消失但既是全局最小值也不是局部最小值的任何位置。

![](https://picx.zhimg.com/v2-42df6dab63e6415342216b5ceeb6b585_1440w.jpg)

局部最小值与全局最小值

![](https://picx.zhimg.com/v2-62c6f601c039261aefcb52f14e5211eb_1440w.jpg)

鞍点

![](https://pic2.zhimg.com/v2-b8322803e8945db3ed79a3116b3d5445_1440w.jpg)

梯度消失

### 小批量随机梯度下降

在处理单个观测值时，我们需要执行许多单一矩阵-向量（甚至向量-向量）乘法，这在计算和深度学习框架开销上都是相当大的。这既适用于计算梯度以更新参数时，也适用于使用神经网络预测。也就是说，每当我们执行参数更新 w←w−ηtgt 时，都会消耗大量资源。其中:

gt=∂wf(xt,w)

我们可以通过将这个操作应用于一个小批量观测值来提高计算效率。也就是说，我们将梯度 g 替换为一个小批量的平均梯度，而不是单个观测值的梯度。这样，梯度 g 的计算公式变为：

gt=∂w1|Bt|∑i∈Btf(xi,w)

这种做法对梯度 g 的统计属性有两个影响：首先，由于小批量 B 中的所有元素都是从训练集中随机抽取的，因此梯度的期望保持不变。其次，方差显著降低。由于小批量梯度由 b 个独立梯度的平均值组成，其标准差降低了 b−12 。这是好事，因为这意味着更新更接近于完整的梯度。

## 专题9 注意力机制，Transformer

### 注意力机制

> 注意力是一种稀缺资源。

在注意力机制的背景下，自主性提醒被称为查询（query）。给定任何查询，注意力机制通过注意力汇聚（attention pooling）将选择引导至感官输入（sensory inputs，例如中间特征表示）。在注意力机制中，这些感官输入被称为值（value）。更通俗的解释，每个值都与一个键（key）配对，这可以想象为感官输入的非自主提醒。可以通过设计注意力汇聚的方式，便于给定的查询（自主性提醒）与键（非自主性提醒）进行匹配，这将引导得出最匹配的值（感官输入）。

注意力机制与全连接层或汇聚层的主要区别在于它增加了自主提示。这意味着模型不仅仅是简单地处理所有的输入，而是根据其重要性赋予不同的权重。总的来说，注意力汇聚输出就是值的加权。

![](https://pic4.zhimg.com/v2-994c6adfb5f7e48a5c4e18275c9e42fd_1440w.jpg)

查询，键，值

下面的例子以拟合 yi=2sin(xi)+xi0.8+ϵ 这个非线性函数为例子，训练样本和测试样本分别由以下代码给出：

```python3
x_train, _ = torch.sort(torch.rand(50) * 5) # 排序后的训练样本，x在[0,5)之间的50个随机数
x_test = torch.arange(0, 5, 0.1) # 测试样本,为0开始，步长0.1的50步数据

def f(x):
    return 2 * torch.sin(x) + x**0.8

y_train = f(x_train) + torch.normal(0.0, 0.5, (50,)) # 训练样本的输出，加上了噪声
y_truth = f(x_test) # 测试样本的真实输出
```

**简单的例子：平均汇聚**

在回归问题中，可以首先尝试使用最简单的估计器来解决问题。基于平均汇聚，可以计算所有训练样本输出值的平均值。显然，平均汇聚忽略了输入 xi 。导致真实函数f（“Truth”）和预测函数（“Pred”）相差很大。

f(x)=1n∑i=1nyi

![](https://pic2.zhimg.com/v2-4bbf4e4489d886fc860080eef4890ea1_1440w.jpg)

**非参数注意力汇聚**

现在将输入的位置 x 纳入考虑。在这里，我们有一组键值对 (xi, yi)，其中 x 是查询，(xi, yi) 是键值对。注意力汇聚是通过计算 yi 的加权平均来实现的。在这个过程中，我们计算查询 x 和键 xi 之间的关系，并将其建模为注意力权重 α(x, xi)。这个权重将被分配给每一个对应的值 yi。对于任何查询，注意力权重在所有键值对上构成一个有效的概率分布。这意味着注意力权重是非负的，并且它们的总和为1。

f(x)=∑i=1nα(x,xi)yi

- Nadaraya-Watson核回归：考虑下方的汇聚公式，如果⼀个键xi越是接近给定的查询x，那么分配给这个键对应值yi的注意力权重就会越大，也就“获得了更多的注意力”。
- Nadaraya-Watson核回归的注意力汇聚是对训练数据中输出的加权平均。从注意力的角度来看，分配给每个值的注意力权重取决于将值所对应的键和查询作为输⼊的函数。

f(x)=∑i=1nα(x,xi)yi=∑i=1nexp⁡(−12(x−xi)2)∑j=1nexp⁡(−12(x−xj)2)yi=∑i=1nsoftmax(−12(x−xi)2)yi

![](https://picx.zhimg.com/v2-5074a2ba71a2decbd8b52f2bb11f7ee3_1440w.jpg)

**带参数注意力汇聚**

非参数的Nadaraya-Watson核回归具有一致性（consistency）的优点：如果有足够的数据，此模型会收敛到最优结果。尽管如此，我们还是可以轻松地将可学习的参数集成到注意力汇聚中。例如，在下面的查询 x 和键 xi 之间的距离乘以可学习参数 w ：

f(x)=∑i=1nα(x,xi)yi=∑i=1nexp⁡(−12((x−xi)w)2)∑j=1nexp⁡(−12((x−xj)w)2)yi=∑i=1nsoftmax(−12((x−xi)w)2)yi

- 模型定义：

```python3
class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))
    def forward(self, queries, keys, values):
        # queries和attention_weights的形状为(查询个数，“键－值”对个数)
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        self.attention_weights = nn.functional.softmax(-((queries - keys) * self.w)**2 / 2, dim=1)
        # values的形状为(查询个数，“键－值”对个数)
        return torch.bmm(self.attention_weights.unsqueeze(1),values.unsqueeze(-1)).reshape(-1)
```

### 注意力评分函数

**注意力评分函数是注意力机制中的一个重要组成部分，它的作用是计算查询和每个键之间的相似性。给定一个查询和一个键，注意力评分函数将它们作为输入，输出一个标量，表示查询和键之间的相似性。**这个标量通常被解释为注意力权重，它决定了在注意力汇聚阶段，对应的值应该被赋予多大的权重。图中的a表示注意力评分函数。

![](https://pica.zhimg.com/v2-3664b974b00f20208ded750eac0ef362_1440w.jpg)

计算注意力汇聚的输出为值的加权和

用数学语言描述，假设有一个查询 q∈Rq 和 m个“键－值”对 (k1,v1),...,(km,vm) ，其中 ，ki∈Rk，vi∈Rv ，注意力汇聚函数f就被表示成值的加权和：

f(q,(k1,v1),...,(km,vm))=∑i=1mα(q,ki)vi∈Rv

其中查询 q 和键 ki 的注意力权重（标量）是通过注意力评分函数a将两个向量映射成标量，再经过softmax运算得到的。选择不同的注意⼒评分函数a会导致不同的注意力汇聚操作：

α(q,ki)=softmax(a(q,ki))=exp(a(q,ki))∑j=1mexp(a(q,kj))∈R

**加性注意力：**

一般来说，当查询和键是不同长度的向量时，可以使用加性注意力作为评分函数。给定查询 q∈Rq 和键 k∈Rk ，加性注意力的评分函数为：

a(q,k)=wv⊤tanh⁡(Wqq+Wkk)∈R 。

其中可学习的参数是 Wq∈Rh×q 、 Wk∈Rh×k 和 wv∈Rh

### **缩放点积注意力**

使用点积可以得到计算效率更高的评分函数，但是点积操作要求查询和键具有相同的长度 d ，即 dq=dk 。点积操作在数学上定义为两个向量的对应元素的乘积之和。如果两个向量的长度为 d，那么点积的均值为0（因为向量元素的均值为0），但是方差会随着向量长度 d 的增加而线性增加。具体来说，如果向量的元素是独立同分布的随机变量，具有零均值和单位方差，那么两个向量的点积的方差将是 d 。

当有 n 个查询和 m 个键-值对时，查询、键和值通常被表示为矩阵，其中查询矩阵 Q 的大小为 n×dk ，键矩阵 K 的大小为 m×dk ，值矩阵 V 的大小为 m×dv 。其评分函数和输出值分别为：

a(q,k)=q⊤kd

Attention(Q,K,V)=softmax(QKTdk)V

**为了确保点积的规模不随向量长度d的变化而变化，我们需要对点积进行缩放。这是因为在注意力机制中，如果点积的规模过大，经过softmax函数后，概率分布可能会变得非常尖锐，这意味着注意力机制可能会过于集中在某些极端值上，而忽略其他信息。为了避免这种情况，我们通过向量长度d的平方根进行缩放，从而使得缩放后的点积的方差保持为1。这样做可以保证softmax函数的输入具有适当的规模，从而使得注意力分布更加平滑，有助于梯度的稳定和模型的训练。**

![](https://pica.zhimg.com/v2-695000ea1e6a736e917739d273084754_1440w.jpg)

（图源：打工仔）相同比例的输入进行等比例放大后，softmax操作后的比例差异会更大

下面给一个函数方式的实现，Transfomer专题里面再解读一下相关源码。

```python3
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    计算缩放点积注意力。
    
    参数:
    query: 查询的张量，形状为 (batch_size, num_queries, d_k)
    key: 键的张量，形状为 (batch_size, num_keys, d_k)
    value: 值的张量，形状为 (batch_size, num_values, d_v)
    mask: 掩码张量，用于遮蔽不相关的数据，形状为 (batch_size, num_queries, num_keys)
    
    返回:
    attention_output: 注意力机制的输出张量
    attention_weights: 注意力权重
    """
    d_k = query.size(-1)
    
    # 计算查询和键的点积，然后除以缩放因子 sqrt(d_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=query.dtype))
    
    # 如果提供了掩码，则将掩码应用于分数
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # 应用softmax获取注意力权重
    attention_weights = F.softmax(scores, dim=-1)
    
    # 根据注意力权重得到输出
    attention_output = torch.matmul(attention_weights, value)
    
    return attention_output, attention_weights
```

### 使用注意力机制的seq2seq

动机：机器翻译中，每个生成的词可能相关于源句子中不同的词。seq2seq模型中不能对此直接建模，因为解码器默认只关注最后一个隐状态。即使并非所有输入（源）词元都对解码某个词元都有用，在每个解码步骤中仍使用编码相同的上下文变量。因而可以尝试引入注意力机制。Bahdanau等人提出了相关模型。

![](https://picx.zhimg.com/v2-6963bc58c3bb09e456fb4307ee6a946f_1440w.jpg)

⼀个带有Bahdanau注意力的循环神经⽹络编码器-解码器模型

在预测词元时，如果并非所有输入词元都相关，模型将只对齐（或参与）输入序列中与当前预测相关的部分。这是通过将上下文变量视为注意力集中的输出来实现的。

这个新的基于注意力的模型与前面的seq2seq模型相同，只不过在任何解码时间步 t′ 中的上下文变量 c 会被 ct′ 替换。假设输入序列中有 T 个词元，解码时间步 t′ 的上下文变量是注意力集中的输出：

ct′=∑t=1Tα(st′−1,ht)ht

- 其中，时间步 t′−1 时的解码器隐状态 st′−1 是查询（注意图中从解码器循环层拉出来指到注意力的箭头），编码器隐状态 ht 既是键，也是值（注意图中编码器拉出指到注意力的两个箭头）。注意力权重是使用加性注意力函数计算的。

### 多头注意力

动机：对于同一个key，value，query，我们希望抽取到不同的信息，有一些注意力层关注到短距离关系，有一些关注到长距离关系。有一些类似于卷积中“通道”的概念。多头注意力融合了来⾃于多个注意力汇聚的不同知识，这些知识的不同来源于相同的查询、键和值的不同的子空间表示。

我们使用h组不同的线性投影来变换查询，键和值。然后把这组变换后的查询、键和值将并行地送到注意力汇聚中。最后，将注意力汇聚的输出拼接在⼀起，并且通过另⼀个可以学习的线性投影进⾏变换，以产生最终输出。

![](https://pic2.zhimg.com/v2-96e73a55253ed5ccfb3eb862c1acb0a1_1440w.jpg)

多头注意力：多个头连结然后线性变换

**下面使用数学语言把这个模型形式化地描述出来：**

给定查询 q∈Rdq 、键 k∈Rdk 和值 v∈Rdv ，每个注意力头 （）hi（i=1,...,h） 的计算方法为：

hi=f(Wi(q)q,Wi(k)k,Wi(v)v)∈Rpv

其中，可学习的参数包括 Wi(q)∈Rpq×dq 、 Wi(k)∈Rpk×dk 和 Wi(v)∈Rpv×dv ，以及代表注意力汇聚的函数 f。函数 f 可以是上文所提到的加性注意力和缩放点积注意力。多头注意力的输出需要经过另一个线性转换，它对应着 h 个头连结后的结果，因此其可学习参数是 Wo∈Rpo×hpv ：

Wo[h1..hh]∈Rpo

基于这种设计，每个头都可能会关注输入的不同部分，可以表示比简单加权平均值更复杂的函数。

### 自注意力

自注意力的输出是一个具有相同长度的序列，其中每个元素都是原始序列中所有元素的加权组合。给定一个由词元组成的输入序列 x1,...,xn ，其中任意 （）xi∈Rd（1≤i≤n） 。该序列的自注意力输出为一个长度相同的序列 y1,...,yn ，其中：

yi=f(xi,(x1,x1),...,(xn,xn))∈Rd

在这里， f 是一个函数，它接受一个查询和一组键值对，并返回一个输出。在自注意力机制中，查询、键和值都来自同一输入序列。在自注意力机制中，输入序列的每个元素都有机会“关注”序列中的其他元素，以便更好地表示和理解自己。

- CNN，RNN，self-attention的比较：目标都是将由n个词元组成的序列映射到另⼀个长度相等的序列，其中的每个输入词元或输出词元都由d维向量表示。k代表CNN的卷积核大小。**注意到，若将CNN和自注意力相比，可以发现自注意力的“感受野”覆盖了整个序列。**

![](https://pic3.zhimg.com/v2-3dab249441a813813b78676a6a8f9e2a_1440w.jpg)

比较卷积神经网络（填充词元被忽略）、循环神经网络和自注意力三种架构

|CNN,RNN,自注意力|CNN|RNN|自注意力|
|---|---|---|---|
|计算复杂度|O(knd^2)|O(nd^2)|O(n^2d)|
|并行度|O(n)|O(1)|O(n)|
|最大路径长度|O(n/k)|O(n)|O(1)|

### Transformer全解

> 从上到下的链接分别给出了论文原文，李沐的论文精读和源码的参考实现。

[[1706.03762] Attention Is All You Need (arxiv.org)​arxiv.org/abs/1706.03762](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1706.03762)

[Transformer论文逐段精读【论文精读】_哔哩哔哩_bilibili​www.bilibili.com/video/BV1pu411o7BE/?spm_id_from=333.337.search-card.all.click&vd_source=52ea46ec60f0b444383a953319f238d2![](https://pic3.zhimg.com/v2-39ab9056eb84a68ff4038070383da594_ipico.jpg)](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1pu411o7BE/%3Fspm_id_from%3D333.337.search-card.all.click%26vd_source%3D52ea46ec60f0b444383a953319f238d2)

[GitHub - jadore801120/attention-is-all-you-need-pytorch: A PyTorch implementation of the Transformer model in "Attention is All You Need".​github.com/jadore801120/attention-is-all-you-need-pytorch](https://link.zhihu.com/?target=https%3A//github.com/jadore801120/attention-is-all-you-need-pytorch)

**总览**

Transformer模型的主要特点是完全放弃了传统的RNN（循环神经网络）和CNN（卷积神经网络）结构，而是完全依赖于自注意力（Self-Attention）机制来并行地捕捉输入序列中的各个元素之间的依赖关系。这种设计使得Transformer模型在处理长距离依赖问题时具有优势。Transformer模型主要由编码器（Encoder）和解码器（Decoder）两部分组成：

![](https://pica.zhimg.com/v2-007b4796106bc833f31e7ce8dd61a368_1440w.jpg)

The Transformer - model architecture

**编码器**

由N个相同的层堆叠而成，每一层包含两个主要部分，一个是多头自注意力（Multi-Head Attention）机制，另一个是位置全连接的前馈网络（Position-wise Feed-Forward Network）。每个部分都有残差连接和层归一化（Layer Normalization）。

`input_sequence` 通常是一个三维张量，尺寸为：`(batch_size, sequence_length, model_dim)`

- `batch_size`：批次大小，表示同时处理的序列的数量。
- `sequence_length`：序列长度，即输入序列中的元素（如单词、字符）数量。
- `model_dim`：模型维度，也称为隐藏层大小，是模型中所有子层和嵌入层的输出维度。

对于单个batch而言，输入是有序列长度（sequence_length）行，嵌入长度（model_dim）列的二维矩阵。解码器输入的Transformer表示是由词embedding + 位置 embedding得到的。在Transformer模型中，编码器（Encoder）的输入和输出张量的形状通常是一样的，这是由Transformer的自注意力机制和前馈神经网络层的设计决定的。

**源码参考：**

```python3
import torch.nn as nn
import torch
from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward

__author__ = "Yu-Hsiang Huang"

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        # 多头注意力机制模块，用于处理输入序列并计算自注意力
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        # 前馈神经网络模块，用于对注意力机制的输出进行进一步处理
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        # 通过多头注意力机制模块处理输入序列，可能使用自注意力掩码
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        # 将多头注意力的输出传入前馈神经网络模块
        enc_output = self.pos_ffn(enc_output)
        # 返回编码器层的输出和自注意力权重
        return enc_output, enc_slf_attn
```

**解码器**

也是由N个相同的层堆叠而成，但在每一层中包含了一个掩码多头注意力机制，以及一个encoder-decoder attention层（第二层），这一层的K，V来自于编码器，Q来自于解码器。这个设计思想来自于传统RNN的encoder-decoder设计，详见下文论文原文解读。

在Transformer模型中，解码器（Decoder）的输入和输出张量的形状也是一致的，但是要注意，解码器的结构比编码器稍微复杂一些。解码器的输入主要有两部分：

1. 目标序列的嵌入表示，形状为 `[batch_size, target_sequence_length, d_model]`。
2. 编码器的输出，形状为 `[batch_size, source_sequence_length, d_model]`。

解码器最后的部分是利用 Softmax 预测下一个单词。

**源码参考：**

```python3
import torch.nn as nn
import torch
from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward

__author__ = "Yu-Hsiang Huang"

class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        # 自注意力机制模块，用于解码器的输入序列
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        # 编码器-解码器注意力机制模块，用于关注编码器的输出
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        # 前馈神经网络模块，用于对注意力机制的输出进行进一步处理
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(
            self, dec_input, enc_output,
            slf_attn_mask=None, dec_enc_attn_mask=None):
        # 通过自注意力机制模块处理解码器的输入序列，可能使用自注意力掩码
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        # 通过编码器-解码器注意力机制模块处理解码器的输出，并关注编码器的输出，可能使用注意力掩码
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        # 将注意力机制的输出传入前馈神经网络模块
        dec_output = self.pos_ffn(dec_output)
        # 返回解码器层的输出和两种注意力权重
        return dec_output, dec_slf_attn, dec_enc_attn
```

**Encoder-Decoder attention层：**

**在Transfomer解码器的Encoder-Decoder attention中，注意力的K，V来自于编码器，Q来自于解码器。**传统RNN的循环层所做的变换为ht=f(xt,ht−1)，然后编码器通过一个函数 q 将所有隐状态转换为上下文变量 c=q(h1,...,hT) ，例如取 q(h1,...,hT)=hT ，上下文变量仅仅是输⼊序列在最后时间步的隐状态。实现解码器时，直接使用编码器最后⼀个时间步的隐状态来初始化解码器的隐状态。

**Transfomer解码器的Encoder-Decoder attention层实际上也是想让解码器的每一步都能看到编码器所编码的全局信息。**在后来的纯解码器结构如GPT中，则丢弃了这一层。因为要求模型自回归地进行生成，要把未来信息“掩码到底”。

> 以下是作者原文，可以看到encoder-decoder attention的设计就是从 typical encoder-decoder attention mechanisms启发得到的：  
>   
> In "encoder-decoder attention" layers, **the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence.** This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models such as [38, 2, 9].

**缩放点积自注意力**

给定查询（Query）、键（Key）和值（Value）三个输入，首先计算查询和所有键的点积，然后对结果进行缩放（一般是除以根号下键的维度大小），接着通过softmax函数得到权重，最后用这个权重对值进行加权求和。公式表示为：

Attention(Q,K,V)=softmax(QKTdk)V 。其中， dk 是键的维度大小。

![](https://pica.zhimg.com/v2-6c89fa2e78630955e1890e0678230b7a_1440w.jpg)

Scaled Dot-Product Attention

**源码参考：**

```python3
import torch
import torch.nn as nn
import torch.nn.functional as F

__author__ = "Yu-Hsiang Huang"

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        # temperature参数用于缩放点积的结果以避免过大的值导致softmax函数的梯度过小
        self.temperature = temperature
        # dropout用于在softmax之后随机地将一些注意力权重设置为0，以防止过拟合
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        # 计算查询（q）和键（k）的点积，然后除以温度参数进行缩放
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        # 如果提供了掩码，则将掩码处的注意力权重设置为非常小的负数
        # 这样在应用softmax时，这些位置的权重接近于0
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        # 应用softmax函数得到注意力权重，然后应用dropout
        attn = self.dropout(F.softmax(attn, dim=-1))
        # 使用注意力权重对值（v）进行加权求和得到输出
        output = torch.matmul(attn, v)

        return output, attn
```

在注意力机制中使用dropout可以减少模型对于特定注意力权重的依赖，从而防止模型过拟合到训练数据。通过随机“丢弃”一些注意力权重，模型被迫学习更加健壮的特征表示，这有助于提高模型的泛化能力。这里的实现可以自己指定Scale系数（虽然标准公式是 dk ）。可以通过在多头注意力中指定该参数为dk来实现。

**（掩码）多头注意力**

多头注意力融合了来⾃于多个注意力汇聚的不同知识，这些知识的不同来源于相同的查询、键和值的不同的子空间表示。其公式表示如下：

MultiHead(Q,K,V)=Concat(head1,head2,...,headh)WO

每个 head 是一个独立的注意力层，计算方式如下： headi=Attention(QWQi,KWKi,VWVi) 。在这里， Q,K,V 分别代表查询、键和值， WQi,WKi,WVi 和 WO 是模型需要学习的参数。

解码器的掩码多头注意力模块得到QKT之后进行mask。在掩码矩阵中，我们将不希望模型关注的位置设置为一个非常大的负数（例如，-10^9）。这是因为在计算注意力权重时，我们会将掩码矩阵加到 QKT 上，然后再应用softmax函数。在这个过程中，非常大的负数会被softmax函数映射到接近于0的值，从而使得对应位置的注意力权重接近于0。

![](https://pic1.zhimg.com/v2-6157d11c609f9b6d9a39e1a6940e1e28_1440w.jpg)

Multi-Head Attention

**源码参考：**

多头注意力模块中，进入到缩放点积注意力的张量的输入维度是：`(batch_size, num_heads，seq_length, d_key)`

其输出张量和输入张量的维度是一致的。

```python3
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformer.Modules import ScaledDotProductAttention

__author__ = "Yu-Hsiang Huang"

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        # 初始化多头注意力的参数
        self.n_head = n_head  # 头的数量
        self.d_k = d_k  # 键/查询的维度
        self.d_v = d_v  # 值的维度

        # 定义线性层，用于将输入的d_model维度转换为多头的维度（n_head * d_k 或 n_head * d_v）
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        # 定义线性层，用于将多头输出合并回d_model维度
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        # 创建一个缩放点积注意力模块
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        # 定义dropout层，用于正则化
        self.dropout = nn.Dropout(dropout)
        # 定义层归一化，用于稳定训练
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):
        # 获取维度信息
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # 保存输入以便后面进行残差连接
        residual = q

        # 线性变换并重塑以准备多头计算
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # 转置以将头维度提前，便于并行计算
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # 如果存在掩码，则扩展掩码以适应头维度
        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        # 调用缩放点积注意力模块
        q, attn = self.attention(q, k, v, mask=mask)

        # 转置并重塑以合并多头
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        # 应用线性变换和dropout
        q = self.dropout(self.fc(q))
        # 添加残差连接并进行层归一化
        q += residual
        q = self.layer_norm(q)

        # 返回多头注意力的输出和注意力权重
        return q, attn
```

**从批量归一化到层归一化**

BN层，即批量归一化，是对一个小批量数据进行归一化，具体来说，就是在每一层的每个输入通道上，对小批量数据进行独立的归一化。批量归一化在预测时需要用到训练时的平均数据。总的来说是按照某一个“特征”进行切分。公式表达为：

BN(x)=γ⊙x−μ^Bσ^B+β

层归一化则是对单个数据样本进行归一化，具体来说，就是在每个样本上，对所有输入通道进行归一化。总的来说是按照某一个“样本”进行切分。

![](https://pic3.zhimg.com/v2-12d1649ffed23997cd5bd02b3739d29e_1440w.jpg)

蓝色是批量归一化，黄色是层归一化

**Transformer模型通常用于处理序列数据，如文本，这种数据的长度通常是可变的。层归一化对序列长度的变化不敏感，因此更适合用于处理这种数据。批量归一化（BN）是对每个特征维度在不同样本（即批次）上进行归一化，这意味着它依赖于批次中的其他样本来计算均值和方差。层归一化（LN）则是对单个样本内的所有特征维度进行归一化，每个样本独立地计算其特征的均值和方差。**

![](https://pic2.zhimg.com/v2-f3f4d31ad16179fe716cdd0cb0295857_1440w.jpg)

蓝色是批量归一化，黄色是层归一化，批量归一化对没见过的长序列可能会无效，层归一化针对单一样本，可以有效处理

**前馈网络**

FFN 的主要动机是增加模型的复杂度和表达能力。由于 Transformer 模型主要基于自注意力机制，这种机制是线性的，因此需要 FFN 进行非线性变换，以增加模型的复杂度和表达能力。 由于FFN的输入已经汇聚了序列信息，因而每个时间步单独进入FFN也能够获得全局的语义信息。FFN就是一个两层的全连接，第一层Relu激活，第二层不使用激活函数。输出和输入维度保持一致。

FFN(x)=max(0,xW1+b1)W2+b2

**源码参考：**

```python3
import torch.nn as nn
import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        # 第一个线性层将输入从d_in维度映射到d_hid维度
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        # 第二个线性层将隐藏层的输出从d_hid维度映射回d_in维度
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        # 层归一化层，用于稳定训练过程
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        # dropout层，用于正则化，防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 保存输入以便后面进行残差连接
        residual = x

        # 通过第一个线性层，然后应用ReLU激活函数
        x = self.w_1(x)
        x = F.relu(x)
        # 通过第二个线性层
        x = self.w_2(x)
        # 应用dropout
        x = self.dropout(x)
        # 添加残差连接（直接将输入加到输出）
        x += residual
        # 应用层归一化
        x = self.layer_norm(x)

        # 返回输出
        return x
```

**位置编码**

在处理词元序列时，循环神经网络是逐个重复地处理词元的，而自注意力则因为并行计算而放弃了顺序操作。为了使用序列的顺序信息，通过在输入表示中添加位置编码来注入绝对的或相对的位置信息。位置编码可以通过学习得到，也可以直接固定得到。接下来描述基于正弦函数和余弦函数的固定位置编码 ：

> 为什么是三角函数？一个好的位置编码方案应该满足： 1.能够对每一词元（矩阵的行，序列的时间步）输出一个独一无二的编码。 2.编码的值应该有界，使得模型能泛化到更长的句子。

**假设输入表示** X∈Rn×d **包含一个序列中** n **个词元的** d **维嵌入表示。位置编码使用相同形状的位置嵌入矩阵** P∈Rn×d **输出** X+P **，矩阵第** i **行、第** 2j **列和** 2j+1 **列上的元素为：**

pi,2j=sin⁡(i100002j/d),pi,2j+1=cos⁡(i100002j/d)

这种基于正弦和余弦函数的位置编码可以有效地捕捉序列中的位置信息，并且与词嵌入相加后可以通过自注意力机制进行处理。在位置嵌入矩阵P中，行代表词元在序列中的位置，列代表位置编码的不同维度。可以发现，编码频率沿着向量维度（矩阵的列）增大而单调降低。

![](https://pic1.zhimg.com/v2-90ed4eada205a982c470140b3a2b87a6_1440w.jpg)

相对位置编码：通过使用上述的位置编码，模型可以学习到输入序列中的相对位置信息。这是因为，对于任何固定的位置偏移量δ，位置i + δ的位置编码可以通过对位置i的位置编码进行线性变换来表示。

## 专题10 Word2vec，BERT，GPT

> 词嵌入（Word Embeddings）通常是针对单个词元（如单词、字符或子词）的。然而，OpenAI 使用的是预训练的 Transformer 模型（如 GPT 和 BERT），这些模型不仅可以为单个词元生成嵌入，还可以为整个句子生成嵌入，即Text embedding，注意区分。典型的句向量模型可以参考BGE。

- One-Hot Encoding：独热编码生成的向量是稀疏的，它们之间的距离相等，无法捕捉单词之间的语义关系。独热编码是固定的，无法在训练过程中进行调整。
- Embedding Layer：嵌入层生成的向量是密集的，它们可以捕捉单词之间的语义关系。具有相似含义的单词在向量空间中距离较近。嵌入层的权重（即嵌入矩阵）可以在训练过程中进行调整，以便更好地捕捉词汇之间的语义关系。

### Word2Vec：CBOW 和 Skip-gram

> **如果是拿一个词语的上下文作为输入，来预测这个词语本身，则是 CBOW 模型。**  
> **而如果是用一个词语作为输入，来预测它周围的上下文，那这个模型叫做 Skip-gram 模型。**

**CBOW 模型**

连续词袋模型（Continuous Bag of Words, CBOW）是一种常用的词嵌入模型，它与跳元模型有一些相似之处，但也有关键区别。连续词袋模型的主要假设是，中心词是基于其在文本序列中的周围上下文词生成的。例如，在文本序列 "the", "man", "loves", "his", "son" 中，如果我们选择 "loves" 作为中心词，并将上下文窗口设置为2，连续词袋模型会考虑基于上下文词 "the", "man", "his", "son" 生成中心词 "loves" 的条件概率，即：

P("loves"|"the","man","his","son")

![](https://pic1.zhimg.com/v2-c444251ddf82fd20758e0aab8bed6710_1440w.jpg)

连续词袋模型考虑了给定周围上下文词生成中心词条件概率

**Skip-gram模型**

跳元模型（Skip-gram model）是一种常用的词嵌入模型，它的基本假设是一个词可以用来生成其周围的单词。以文本序列 "the", "man", "loves", "his", "son" 为例，如果我们选择 "loves" 作为中心词，并将上下文窗口设置为2，跳元模型会考虑生成上下文词 "the", "man", "his", "son" 的条件概率，即：

P("the","man","his","son"|"loves")

在跳元模型中，我们通常假设上下文词是在给定中心词的情况下独立生成的，这被称为条件独立性。因此，上述条件概率可以被重写为：

P("the"|"loves")·P("man"|"loves")·P("his"|"loves")·P("son"|"loves")

这意味着，我们可以分别计算每个上下文词在给定中心词的情况下的概率，然后将这些概率相乘，得到的结果就是所有上下文词在给定中心词的情况下的联合概率。这是跳元模型的基本工作原理。

![](https://pic3.zhimg.com/v2-410f1c9b07e7f1fdb1b51378a40141da_1440w.jpg)

跳元模型考虑了在给定中心词的情况下生成周围上下文词的条件概率

**两种模型的网络结构**

> 参考：

[https://cs224d.stanford.edu/lecture_notes/notes1.pdf​cs224d.stanford.edu/lecture_notes/notes1.pdf](https://link.zhihu.com/?target=https%3A//cs224d.stanford.edu/lecture_notes/notes1.pdf)

Word2Vec 模型本质上可以就是一个简单的神经网络，它包含一个输入层、一个隐藏层，以及一个输出层。在这个网络中，并没有激活函数应用于隐藏层的节点，而是直接将输入传递到隐藏层，然后再传递到输出层。这种结构可以被视为全连接（fully connected）或密集（dense）层的网络，因为每个输入节点都与隐藏层的每个节点相连，隐藏层的每个节点又都与输出层的每个节点相连。

在 Word2Vec 中，输入层和输出层的节点数等于词汇表的大小（用 one-hot 编码表示），而隐藏层的节点数等于我们想要学习的嵌入向量的维度。尽管结构上类似于全连接网络，但 Word2Vec 的目标并不是执行传统的分类或回归任务，而是学习词的向量表示，这些向量可以捕捉词之间的语义信息。词嵌入（embeddings）是通过两个权重矩阵来学习的：输入矩阵（通常称为`W`）和输出矩阵（通常称为`W'`）。这两个矩阵的维度分别是`V*N`和`N*V`，其中`V`是词汇表的大小，而`N`是嵌入向量的维度。

在训练过程中，每个单词都会通过它的索引与输入矩阵`W`相对应，这样每个单词就会有一个与之对应的嵌入向量。这个向量就是输入矩阵`W`中的一行。当模型完成训练后，这个输入矩阵`W`就可以作为词嵌入矩阵使用。

![](https://pic3.zhimg.com/v2-b6e5e59d8388bca8450994cdf0979b92_1440w.jpg)

![](https://pic4.zhimg.com/v2-ce28580f45d6d8ea2a457ba8cbe6b313_1440w.jpg)

图源：https://cs224d.stanford.edu/lecture_notes/notes1.pdf

**负采样**

在原始的Word2Vec模型中，我们使用softmax函数来计算目标词的概率。这需要对词汇表中的每个词进行权重更新，这在大词汇表中是非常耗时的。而负采样是一种优化训练Word2Vec模型的方法。**它的核心思想是，对于每个训练样本，我们不仅考虑正例（目标词），还随机选取一小部分的负例（即非目标词）进行更新权重，而不是词汇表中的所有单词。如此一来，我们将多分类转变了一系列二分类问题。因而可以只更新部分权重。**

具体来讲：

1. 在训练神经网络时，我们通常使用梯度下降算法来更新权重。在Word2Vec中，权重实际上是单词的向量表示。在不使用负采样的情况下，softmax函数要求我们更新所有单词的向量，因为它需要计算整个词汇表上的概率分布。然而，当使用负采样时，我们改为优化一个简化的问题，即二分类问题，我们只关心目标词（正例）和少量随机选取的非目标词（负例）的概率。
2. 在负采样中，对于每个训练样本（目标词和上下文词对），我们首先更新目标词的向量表示，使得它更可能与上下文词一起出现。接着，我们从词汇表中随机选择K个负例，并更新这些负例的向量表示，使得它们与上下文词一起出现的概率降低。这意味着，对于每个训练样本，我们只更新1个正例和K个负例的向量，而不是整个词汇表的向量。

### BERT：来自Transformers的双向编码器表示

动机：

通过使用Transformer的双向自注意力机制，BERT能够同时考虑一个单词在其上下文中的左侧和右侧的信息，从而更好地理解其含义。此外，通过预训练-微调的方式，BERT能够在大规模未标注数据上学习到丰富的语言知识，然后将这些知识迁移到具体的NLP任务上，从而在许多任务上取得了显著的性能提升。

![](https://picx.zhimg.com/v2-bdbc6a8772e3068fbee21624654f8d83_1440w.jpg)

**预训练任务之一：掩码语言模型（MLM）**

**在MLM任务中，BERT的输入文本中的一部分单词被随机地替换为特殊的[MASK]标记。模型的目标是预测被遮蔽的单词。**这种方法允许BERT在训练过程中学习到双向的上下文信息，因为它需要同时考虑被遮蔽单词左侧和右侧的上下文来预测其原始单词。MLM任务的一个关键优势是，与传统的从左到右或从右到左的预训练方法相比，它可以更好地捕捉到双向上下文信息。

**预训练任务之二：下一句预测（NSP）**

**在NSP任务中，BERT需要预测两个给定句子是否是连续的。**具体来说，模型接收一对句子作为输入，并需要判断第二个句子是否紧跟在第一个句子之后。这个任务的目的是帮助BERT学习句子之间的关系，从而更好地理解句子级别的上下文。这对于理解文章中的句子关系和执行一些需要跨句子推理的任务（如问答和摘要）非常有用。

![](https://pic2.zhimg.com/v2-27e7067c190c2d8edc50dd5cec75f28d_1440w.jpg)

Overall pre-training and fine-tuning procedures for BERT

**架构：**

**BERT模型采用了多层的Transformer编码器（BERT的E就是Encoder）。**每一层都包含一个自注意力机制和一个前馈神经网络。这些层被堆叠在一起，形成了BERT的深度网络结构。

![](https://pica.zhimg.com/v2-d63f9e3c739fcbd99929db055a4a2be0_1440w.jpg)

Transformer编码器块

**BERT的嵌入主要包括三种类型：词嵌入、段落嵌入和位置嵌入。相加后作为网络输入。**

1. **词嵌入：**词嵌入是将单词转换为向量的过程。在BERT中，每个单词首先被转换为一个固定大小的向量，这个向量捕获了单词的语义信息。BERT使用了WordPiece嵌入，这是一种子词嵌入方法，可以有效处理未知词和长尾词。即下图中黄色的那一层，注意到有一些特殊的token（[CLS]，[SEP]等）。
2. **段落嵌入：**BERT可以处理一对句子的输入（例如，问答任务或者自然语言推理任务中的问题和答案）。为了区分这两个句子，BERT引入了段落嵌入。每个句子都有一个对应的段落嵌入，这个嵌入是添加到每个单词的词嵌入上的。即下图中绿色的那一层，注意到第一个[SEP]前全是EA，第二个前全是EB。
3. **位置嵌入：**由于Transformer模型并不考虑单词的顺序，BERT引入了位置嵌入来捕获单词在句子中的位置信息。每个位置（即每个单词在句子中的索引）都有一个对应的位置嵌入，这个嵌入也是添加到每个单词的词嵌入上的。即下图中灰色的那一层，从E0到E10表征了token的位置。

![](https://pic3.zhimg.com/v2-c7a36720f6671a6a45fe1afc4e375278_1440w.jpg)

BERT input representation

**[CLS]标记的意义：**BERT在句子前会加一个[CLS]标志，最后一层中的[CLS]标志向量可以作为整句话的语义表示，从而用于下游任务。[CLS]本身没有什么语义，但最后一层得到的向量是层层注意力模块加权计算后的结果，或许能够更没有偏差地表征整个句子的语义。

**WordPiece**

WordPiece 算法与 BPE 类似，也是迭代地合并子词，但合并的准则不同。WordPiece 在合并过程中会考虑生成的子词对语言模型的贡献。具体来说，它会选择那些合并后能够最大化语言模型概率的子词对进行合并。这意味着，WordPiece 在每次合并时都会评估合并带来的语言模型概率增益，并选择增益最大的合并操作。下方有BPE的相关介绍。

### GPT：Generative Pre-trained Transformer

**动机：**

GPT由OpenAI在2018年提出，它也是建立在Transformer的基础上，但GPT采用的是Transformer的解码器结构（没有第二个自注意力层，即Q来自解码器，K，V来自编码器那一层）。GPT的关键思想是使用大量的文本数据进行无监督预训练，然后在特定任务上进行微调。GPT在预训练阶段使用的是语言模型任务，即给定一个文本序列的前N个单词，预测第N+1个单词。这种方式使得模型能够学习到丰富的语言特征。

**词表压缩：BytePairEncoding(BPE)**

BPE（Byte Pair Encoding）是一种数据压缩技术，后来被应用于自然语言处理（NLP）中的分词任务。在 NLP 中，BPE 的主要目的是将词汇分解成更小的可重用的单元（subwords），这有助于处理未知词汇和减少词汇表的大小。BPE 算法的基本思想是迭代地将最常见的字节对（在 NLP 中通常是字符对）合并成一个新的单元。以下是 BPE 算法的步骤：

1. **准备数据**：  
    准备一份文本数据，并计算所有单词的频率。为了表示单词的边界，在每个单词的末尾添加一个特殊字符，通常是 `</w>`。
2. **初始化词汇表**：  
    将每个字符（包括特殊字符）视为基本单元，并将其作为初始词汇表的一部分。
3. **统计并合并**：  
    统计所有相邻字符对的频率，并找出出现次数最多的对。然后将这个字符对合并为一个新的单元，并更新词汇表和文本数据。
4. **迭代过程**：  
    重复上述合并过程，直到达到预设的词汇表大小或者合并次数。

通过这种方式，BPE能够生成一个固定大小的词汇表，其中包含了单个字符、常见的字符序列以及完整的单词。这个词汇表可以有效地覆盖整个训练语料库，并为OOV（Out of Vocabulary）单词提供合理的表示，因为这些单词可以被分解为词汇表中的子词序列。

### BERT与GPT的区别简介

- BERT 使用的是Transformer的编码器（Encoder）结构。它是设计为深度双向模型，通过同时考虑左右两侧的上下文来预训练语言表示。Encoder结构是自编码的。
- BERT 采用了掩码语言模型（Masked Language Model, MLM）和下一句预测（Next Sentence Prediction, NSP）两种预训练任务。MLM随机掩盖输入序列中的单词并预测这些单词，而NSP预测两个句子是否顺序相邻。
- GPT 使用的是Transformer的解码器（Decoder）结构，并且不使用encoder-decoder注意力。GPT是单向的，它预训练一个生成式的语言模型，主要考虑左侧（之前）的上下文。Decoder是自回归的。
- GPT 主要采用传统的语言模型预训练任务(Next Token Prediction，NTP)，即根据给定的文本前缀预测下一个单词，只利用了左侧的上下文信息。

  

## 附录 PyTorch模型搭建：一般性框架与常用接口

最近也简单整理了一下PyTorch构建模型的一般方法和常用接口，作为对本笔记的补充，欢迎大家来看~

### **1 框架示例**

使用PyTorch构建模型时，通常涉及到自定义块这个概念。自定义块就是模型中的一层或者一块，是用于创建网络实体的类。

这里给出一个基本代码框架，然后进行解释：

```python3
import torch
import torch.nn as nn

class CustomModule(nn.Module):
    # 初始化模块的参数或子模块
    def __init__(self, in_features, out_features):
        super(CustomModule, self).__init__()

        # 示例：定义一个线性层
        self.linear = nn.Linear(in_features, out_features)
        
        # 示例：定义一个可重用的ReLU激活层
        self.relu = nn.ReLU()

    # 定义前向传播的计算
    def forward(self, x):
        # 示例：通过定义的层传递输入
        x = self.linear(x)  # 输入通过线性层
        x = self.relu(x)    # 通过ReLU激活层
        
        return x
```

**类定义**

```python3
class CustomModule(nn.Module):
```

CustomModule 类继承自 nn.Module，即 PyTorch 中所有神经网络模块的基类。继承 nn.Module 允许 CustomModule 使用 PyTorch 的自动梯度计算、参数管理、模型保存/加载等功能。

**构造函数 __init__**

```python3
def __init__(self, in_features, out_features):
    super(CustomModule, self).__init__()
    self.linear = nn.Linear(in_features, out_features)
    self.relu = nn.ReLU() 
```

构造函数 __init__ 初始化 CustomModule 实例。它接收两个参数：in_features 和 out_features，分别表示输入和输出特征的维度。

- super(CustomModule, self).__init__() 调用基类的构造函数来正确初始化 nn.Module。Python3也可写super().__init__() 。
- self.linear 创建了一个 nn.Linear 层，它是一个全连接层，用于将输入特征线性变换到指定的输出特征维度。
- self.relu 创建了一个 nn.ReLU 层，它是一个非线性激活函数，用于增加模型的表达能力。

**前向传播 forward**

```python3
def forward(self, x):
    x = self.linear(x)
    x = self.relu(x)
    return x
```

forward 方法定义了数据通过 CustomModule 时的计算流程。即定义了输入如何转换为输出。

- 输入 x 首先通过 self.linear 全连接层，进行线性变换。
- 接着，变换后的输出通过 self.relu ReLU 激活层，引入非线性，去除负值。

最终，forward 方法返回通过线性层和激活层处理后的结果。

**总结**

**在 PyTorch 中，当继承 nn.Module 创建自定义模块时，通常需要重写至少以下两个方法：**

1. **__init__(self, ...): 构造函数，用于初始化模块的参数或子模块。**在这个方法中，会调用父类的构造函数，并定义模块内部将使用的层和参数。
2. **forward(self, x): 前向传播方法，定义了模块的计算流程，即当模块接收输入数据时应该执行的操作。**在 forward 方法中，会使用在 __init__ 方法中定义的层和参数来指定如何处理输入，并返回输出。

这两个方法是创建自定义模块时最基本和最重要的部分。根据需要，可能还会重写其他方法。

### 2 常用工具函数

> 这里的函数接口不可能列全，只是抛砖引玉，大家可以自行查询官方Doc，靠谱。

### 张量加法和乘法

**1、torch.add(input, other, *, alpha=1, out=None)**

[torch.add — PyTorch 2.1 documentation](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/generated/torch.add.html%23torch.add)

PyTorch 中用于逐元素相加两个张量的函数。如果两个张量的形状不同，PyTorch 将应用广播规则来匹配它们的形状。还可以通过 alpha 参数来缩放第二个张量的值。

**2、torch.matmul(input, other, *, out=None)**

[torch.matmul — PyTorch 2.1 documentation](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/generated/torch.matmul.html%23torch-matmul)

PyTorch 中用于执行张量的矩阵乘法的函数。支持向量、矩阵和高维张量的乘法。对于两个二维张量，它执行矩阵乘法。对于高维张量，它执行批量矩阵乘法。如果其中一个参数是一维张量，它将执行点积。

**代码示例**

```python3
import torch

# 创建张量
tensor1 = torch.tensor([[1, 2], 
                        [3, 4]])
tensor2 = torch.tensor([[5, 6], 
                        [7, 8]])
vector = torch.tensor([1, 2])
batch1 = torch.randn(10, 3, 4)
batch2 = torch.randn(10, 4, 5)

# 使用 torch.add 进行张量相加
result_add = torch.add(tensor1, tensor2)

# 使用 torch.add 时缩放第二个张量
result_add_scaled = torch.add(tensor1, tensor2, alpha=10)

# 使用 torch.matmul 进行矩阵乘法
result_matmul_matrix = torch.matmul(tensor1, tensor2)

# 使用 torch.matmul 进行批量矩阵乘法
result_matmul_batch = torch.matmul(batch1, batch2)

# 使用 torch.matmul 进行向量和矩阵的乘法
result_matmul_vector = torch.matmul(vector, tensor1)

# 验证批量矩阵乘法结果正确性
# 逐个乘以每个批次的矩阵，并验证结果
for i in range(batch1.size(0)):
    individual_result = torch.matmul(batch1[i], batch2[i])
    if not torch.allclose(individual_result, result_matmul_batch[i]):
        print(f"Batch {i} does not match!")
        break
else:
    print("All individual batch multiplications match the torch.matmul result!")

# 打印其他操作的结果
print("Result of torch.add:\n", result_add)
print("Result of torch.add with scaling:\n", result_add_scaled)
print("Result of torch.matmul with matrices:\n", result_matmul_matrix)
print("Result of torch.matmul with vector and matrix:\n", result_matmul_vector)
```

**输出结果**

```python3
All individual batch multiplications match the torch.matmul result!
Result of torch.add:
 tensor([[ 6,  8],
        [10, 12]])
Result of torch.add with scaling:
 tensor([[51, 62],
        [73, 84]])
Result of torch.matmul with matrices:
 tensor([[19, 22],
        [43, 50]])
Result of torch.matmul with vector and matrix:
 tensor([ 7, 10])
```

### 张量重塑

**1. tensor.view(*shape)**

[torch.Tensor.view — PyTorch 2.1 documentation](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/generated/torch.Tensor.view.html%23torch-tensor-view)

view 方法用于重新塑形一个张量而不改变其数据。它返回一个新的张量，新张量的视图与原始张量共享同一块内存（即不复制数据）。因此，对新张量的任何修改都会反映在原始张量上，反之亦然。*shape 参数允许指定新张量的形状，需要确保新形状与原始张量的元素总数相同。

**2. tensor.reshape(*shape)**

[torch.Tensor.reshape — PyTorch 2.1 documentation](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/generated/torch.Tensor.reshape.html%23torch-tensor-reshape)

reshape 方法与 view 非常相似，也用于改变张量的形状。不同之处在于，如果原始张量的内存布局不允许以所需的形状提供视图，reshape 可能会返回一个数据复制后的新张量。这意味着新张量与原始张量可能不共享内存。*shape 参数同样定义了新张量的形状。

**3. tensor.transpose(dim0, dim1)**

[torch.Tensor.transpose — PyTorch 2.1 documentatio](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/generated/torch.Tensor.transpose.html%23torch-tensor-transpose)

transpose 方法用于交换张量中的两个指定维度。它返回一个新的张量，新张量在指定的两个维度上交换了轴，但数据仍然共享内存。dim0 和 dim1 参数是需要交换的两个维度的索引。

**4. tensor.permute(*dims)**

[torch.Tensor.permute — PyTorch 2.1 documentation](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/generated/torch.Tensor.permute.html%23torch-tensor-permute)

permute 方法用于重新排列张量的维度。与 transpose 不同的是，permute 可以一次性重新排列多个维度，而不仅仅是交换两个维度。*dims 参数是一个包含维度索引的序列，定义了每个维度在新张量中的顺序。返回的新张量在内存中的布局可能与原始张量不同。

**代码示例**

```python3
import torch

# 创建一个 3x2x4 的张量
tensor = torch.arange(24).view(2, 3, 4)
print("Original tensor shape:")
print(tensor.size())

# 使用 view 重新塑形为 4x3x2 的张量
view_tensor = tensor.view(4, 3, 2)
print("\nView tensor (4x3x2) shape:")
print(view_tensor.size())
print("Shares memory with original tensor:", tensor.data_ptr() == view_tensor.data_ptr())

# 使用 reshape 重新塑形为 6x4 的张量
reshape_tensor = tensor.reshape(6, 4)
print("\nReshape tensor (6x4) shape:")
print(reshape_tensor.size())
print("Shares memory with original tensor:", tensor.data_ptr() == reshape_tensor.data_ptr())

# 使用 transpose 交换维度 1 和 2
transposed_tensor = tensor.transpose(1, 2)
print("\nTransposed tensor (swap dimensions 1 and 2) shape:")
print(transposed_tensor.size())
print("Shares memory with original tensor:", tensor.data_ptr() == transposed_tensor.data_ptr())

# 使用 permute 重新排列维度顺序
permuted_tensor = tensor.permute(2, 0, 1)
print("\nPermuted tensor (permute dimensions) shape:")
print(permuted_tensor.size())
print("Shares memory with original tensor:", tensor.data_ptr() == permuted_tensor.data_ptr())
```

**输出结果**

```python3
Original tensor shape:
torch.Size([2, 3, 4])

View tensor (4x3x2) shape:
torch.Size([4, 3, 2])
Shares memory with original tensor: True

Reshape tensor (6x4) shape:
torch.Size([6, 4])
Shares memory with original tensor: True

Transposed tensor (swap dimensions 1 and 2) shape:
torch.Size([2, 4, 3])
Shares memory with original tensor: True

Permuted tensor (permute dimensions) shape:
torch.Size([4, 2, 3])
Shares memory with original tensor: True
```

### contiguous()

[torch.Tensor.contiguous — PyTorch 2.1 documentation](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html%23torch.Tensor.contiguous)

contiguous() 是 PyTorch 中的一个张量（Tensor）方法，它用于确保张量在内存中是连续存储的。在内存中连续存储的张量意味着张量的元素在物理内存中是按顺序排列的，没有间隔。

**某些操作，如 transpose、permute 和 view，可能会返回一个原始数据的新视图，这个新视图在内存中可能不是连续的。这是因为这些操作仅改变了张量的形状和步长，而没有改变底层数据的物理布局。当尝试执行某些特定操作，特别是那些期望张量在内存中连续存储的操作时，如果张量不是连续的，可能会遇到错误，或者操作可能不会按预期执行。在这种情况下，需要调用 contiguous() 方法来使张量连续。**

contiguous() 方法会返回一个新的张量，该张量在内存中是连续存储的，并且内容与原始张量相同。如果原始张量已经是连续的，contiguous() 不会执行任何操作，并且会返回原始张量本身。如果原始张量不是连续的，contiguous() 将创建一个数据的物理副本，以确保连续性。例如：

```python3
import torch

x = torch.tensor([[1, 2], [3, 4]])
y = x.transpose(0, 1)
print(y.is_contiguous())  # 输出: False

y_contiguous = y.contiguous()
print(y_contiguous.is_contiguous())  # 输出: True
```

### squeeze() & unsqueeze()

**1、tensor.squeeze(dim)：**

[torch.squeeze — PyTorch 2.1 documentation](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/generated/torch.squeeze.html%23torch.squeeze)

移除张量中所有维度为1的轴。可以指定要考虑的特定轴，例如.squeeze(0)将只考虑第0轴。

**2、tensor.unsqueeze(dim)：**

[torch.unsqueeze — PyTorch 2.1 documentation](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/generated/torch.unsqueeze.html%23torch.unsqueeze)

与squeeze()相反，用于在指定位置添加一个维度为1的轴。

**代码示例**

```python3
import torch

# 创建一个形状为(3, 1, 5)的张量
x = torch.zeros(3, 1, 5)
print("原始张量形状:", x.shape)

# 使用squeeze()移除所有单一维度
x_squeezed = x.squeeze()
print("使用squeeze()后的形状:", x_squeezed.shape)

# 创建一个形状为(3, 5)的张量
y = torch.zeros(3, 5)
print("原始张量形状:", y.shape)

# 使用unsqueeze()在第0维添加一个维度
y_unsqueezed = y.unsqueeze(1)
print("使用unsqueeze()后的形状:", y_unsqueezed.shape)
```

**输出结果**

```text
原始张量形状: torch.Size([3, 1, 5])
使用squeeze()后的形状: torch.Size([3, 5])
原始张量形状: torch.Size([3, 5])
使用unsqueeze()后的形状: torch.Size([3, 1, 5])
```

### 3 常用模型接口

> 相关模型都可以查看要点笔记进行查看相关概念。同样地，这里的模型接口不可能列全，只是抛砖引玉，大家可以自行查询官方Doc。

### **torch.nn.Linear(in_features,out_features,bias=True,device=None,dtype=None)**

[Linear — PyTorch 2.1 documentation](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/generated/torch.nn.Linear.html%23torch.nn.Linear)

- in_features: 输入张量的特征数量。
- out_features: 输出张量的特征数量。
- bias: 是否添加可学习的偏置到输出。默认值为 True。

### torch.nn.Softmax(dim=None)

[Softmax — PyTorch 2.1 documentation](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/generated/torch.nn.Softmax.html%23torch.nn.Softmax)

- dim: 指定计算 Softmax 的维度。在多维张量中，它表示将在哪个维度上进行概率分布的归一化。

### torch.nn.Dropout(p=0.5,inplace=False)

[Dropout — PyTorch 2.1 documentation](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/generated/torch.nn.Dropout.html%23torch.nn.Dropout)

- p: Dropout 概率，即在训练时设置为零的特征的比例。默认值为 0.5。
- inplace: 如果设置为 True，将直接在输入上进行操作，以节省内存。默认值为 False。

### torch.nn.ReLU(inplace=False)

[ReLU — PyTorch 2.1 documentation](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/2.1/generated/torch.nn.ReLU.html%23relu)

- 应用一个逐元素的 ReLU 函数，即将输入中所有负值置为零。
- inplace: 如果设置为 True，将会直接对输入进行操作，节省内存。默认值为 False。

### torch.nn.LayerNorm(normalized_shape,eps=1e-05)

[LayerNorm — PyTorch 2.1 documentation](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/2.1/generated/torch.nn.LayerNorm.html%23torch.nn.LayerNorm)

> 这里的签名不完善，只是给出了最常用的两个参数，详见官方Doc。

torch.nn.LayerNorm 将层归一化应用于一个小批量的输入，计算公式为：

y=x−E[x]Var[x]+ϵ∗γ+β

其中，均值 E[x] 和标准差 Var[x] 是在最后 D 个维度上计算的，这里的 D 是 normalized_shape 的维度。例如，如果 normalized_shape 是 (3, 5)（一个二维形状），那么均值和标准差是在输入的最后两个维度上计算的（即 input.mean((-2, -1))）。

γ 和 β 是 normalized_shape 的可学习的仿射变换参数。

**官方代码示例**

```python3
# NLP Example
batch, sentence_length, embedding_dim = 20, 5, 10
embedding = torch.randn(batch, sentence_length, embedding_dim)
layer_norm = nn.LayerNorm(embedding_dim)
# Activate module
layer_norm(embedding)

# Image Example
N, C, H, W = 20, 5, 10, 10
input = torch.randn(N, C, H, W)
# Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
# as shown in the image below
layer_norm = nn.LayerNorm([C, H, W])
output = layer_norm(input)
```

### 4 实例：Transfomer部分源码

> 源码参考：

[jadore801120/attention-is-all-you-need-pytorch: A PyTorch implementation of the Transformer model in "Attention is All You Need". (github.com)​github.com/jadore801120/attention-is-all-you-need-pytorch](https://link.zhihu.com/?target=https%3A//github.com/jadore801120/attention-is-all-you-need-pytorch)

学习了一些一般框架和常用接口后，这里给出编码器，多头自注意力，缩放点积注意力的注释源码，并将部分命名更清晰化，大家可以尝试阅读理解，尤其注意其中张量的“形变”。

### 编码器

input_sequence 通常是一个三维的张量，其尺寸为：

(batch_size, sequence_length, model_dim)

- batch_size：批次大小，表示同时处理的序列的数量。
- sequence_length：序列长度，即输入序列中的元素（如单词、字符）数量。
- model_dim：模型维度，也称为隐藏层大小，是模型中所有子层和嵌入层的输出维度。

```python3
import torch.nn as nn

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, model_dim, inner_dim, num_heads, key_dim, value_dim, dropout=0.1):
        super().__init__()
        # 多头注意力机制模块，用于处理输入序列并计算自注意力
        self.self_attention = MultiHeadAttention(num_heads, model_dim, key_dim, value_dim, dropout=dropout)
        # 前馈神经网络模块，用于对注意力机制的输出进行进一步处理
        self.feed_forward = PositionwiseFeedForward(model_dim, inner_dim, dropout=dropout)

    def forward(self, input_sequence, self_attention_mask=None):
        # 通过多头注意力机制模块处理输入序列，可能使用自注意力掩码
        output_sequence, self_attention_weights = self.self_attention(
            input_sequence, input_sequence, input_sequence, mask=self_attention_mask)
        # 将多头注意力的输出传入前馈神经网络模块
        output_sequence = self.feed_forward(output_sequence)
        # 返回编码器层的输出和自注意力权重
        return output_sequence, self_attention_weights
```

### 多头自注意力

```python3
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, num_heads, d_model, d_key, d_value, dropout=0.1):
        super().__init__()

        # 初始化多头注意力的参数
        self.num_heads = num_heads  # 头的数量
        self.d_key = d_key  # 键/查询的维度
        self.d_value = d_value  # 值的维度

        # 定义线性层，用于将输入的 d_model 维度转换为多头的维度（n_head * d_k 或 n_head * d_v）
        self.linear_query = nn.Linear(d_model, num_heads * d_key, bias=False)
        self.linear_key = nn.Linear(d_model, num_heads * d_key, bias=False)
        self.linear_value = nn.Linear(d_model, num_heads * d_value, bias=False)

        # 定义线性层，用于将多头输出合并回 d_model 维度
        self.linear_out = nn.Linear(num_heads * d_value, d_model, bias=False)

        # 创建一个缩放点积注意力模块
        self.attention = ScaledDotProductAttention(temperature=d_key ** 0.5)

        # 定义 dropout 层，用于正则化
        self.dropout = nn.Dropout(dropout)
        # 定义层归一化，用于稳定训练
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, query, key, value, mask=None):
        # 获取维度信息
        d_key, d_value, num_heads = self.d_key, self.d_value, self.num_heads
        batch_size, len_query, len_key, len_value = query.size(0), query.size(1), key.size(1), value.size(1)

        # 保存输入以便后面进行残差连接
        residual = query

        # 线性变换并重塑以准备多头计算
        query = self.linear_query(query).view(batch_size, len_query, num_heads, d_key)
        key = self.linear_key(key).view(batch_size, len_key, num_heads, d_key)
        value = self.linear_value(value).view(batch_size, len_value, num_heads, d_value)

        # 转置以将头维度（num_heads）提前，便于并行计算
        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)

        # 如果存在掩码，则扩展掩码以适应头维度
        if mask is not None:
            mask = mask.unsqueeze(1)

        # 调用缩放点积注意力模块
        output, attention = self.attention(query, key, value, mask=mask)

        # 转置并重塑以合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, len_query, -1)
        # 应用线性变换和 dropout
        output = self.dropout(self.linear_out(output))
        # 添加残差连接并进行层归一化
        output += residual
        output = self.layer_norm(output)

        # 返回多头注意力的输出和注意力权重
        return output, attention
```

### 缩放点积注意力

```python3
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, dropout_rate=0.1):
        super().__init__()
        # 温度参数用于缩放点积的结果以避免过大的值导致softmax函数的梯度过小
        self.temperature = temperature
        # dropout_rate用于在softmax之后随机地将一些注意力权重设置为0，以防止过拟合
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key, value, mask=None):
        # 计算查询（query）和键（key）的点积，然后除以温度参数进行缩放
        scaled_dot_product = torch.matmul(query / self.temperature, key.transpose(2, 3))

        # 如果提供了掩码，则将掩码处的注意力权重设置为非常小的负数
        # 这样在应用softmax时，这些位置的权重接近于0
        if mask is not None:
            scaled_dot_product = scaled_dot_product.masked_fill(mask == 0, -1e9)

        # 应用softmax函数得到注意力权重，然后应用dropout
        attention_weights = self.dropout(F.softmax(scaled_dot_product, dim=-1))
        # 使用注意力权重对值（value）进行加权求和得到输出
        output = torch.matmul(attention_weights, value)

        return output, attention_weights
```

编辑于 2024-12-10 10:41・IP 属地北京