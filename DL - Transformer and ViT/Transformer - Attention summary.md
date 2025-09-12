
## 一：简介

如果你想深入理解大语言模型，**[Transformer](https://zhida.zhihu.com/search?content_id=261762936&content_type=Article&match_order=1&q=Transformer&zhida_source=entity) 中的 Attention 机制**是理解中的重中之重。在现代深度学习和自然语言处理领域，Attention 已成为处理序列数据、捕捉长程依赖的核心工具。它通过为输入序列中的每个元素分配动态权重，使模型能够重点关注关键信息，从而提升特征表示能力和整体性能。

本文主要围绕 **Transformer Attention 家族**进行介绍，重点回答以下几个问题：

1. Attention 的作用是什么？
2. Attention 的数学表达式是什么，他们的区别与联系是什么？
3. 为什么要使用 Attention，它解决了哪些问题？
4. 在什么场景下适合使用哪种Attention？
5. 不同的attention和decoder，encoder的关系是什么？

文章列举了 五种最常用的 Attention 机制：

- Attention（普通注意力）
- Self-Attention（自注意力）
- Multi-Head self-Attention（多头自注意力）
- [Cross-Attention](https://zhida.zhihu.com/search?content_id=261762936&content_type=Article&match_order=1&q=Cross-Attention&zhida_source=entity)（跨序列注意力）
- Masked Self-Attention（掩码自注意力）

这些机制覆盖了 Transformer 的核心设计思想，是理解序列建模的基础。需要注意的是，本文未涉及一些更复杂或衍生的 Attention 机制，例如 additive attention**。**

## 二：本文重点：注意力机制对比表

下表为本文将介绍的五类 Attention 机制的对比速览，展示了它们的使用目的，特点及在 Transformer 中的角色。接下来的内容将对每种机制进行详细解读。

|机制|Q/K/V来源|特点/区别|应用位置|
|---|---|---|---|
|注意力attention|Q，K，V可以来自不同的序列|最基本的形式，计算“查询”和“键值”的加权和|基础形式，一般用作概念解释或跨序列注意力的母式|
|自注意力self-attention|Q=K=V=同一个序列X|捕捉序列内部依赖，位置无关（需位置编码）|encoder，decoder（第一层是masked self- attention）|
|多头自注意力multi-head self-attention|Q，K，V来自同一序列|并行多个注意力，捕捉不同自空间的依赖关系|encoder->self,decoder->masked self + cross|
|跨序列注意力cross-attention|Q来自decoder，K/V来自encoder|让decoder关注encoder的输出，融合源序列信息|decoder，在maksed self-attention之后|
|掩码注意力masked self-attention|Q=K=V=decoder内部序列，与self-attention相同，但加上mask|防止模型训练时看到未来的token，保证自回归|decoder，第一步生成时使用|

## 三： Attention家族

Attention 是一种神经网络机制，用于在序列建模中**选择性地关注输入数据的不同部分**。它通过查询（Q）匹配键（K），并用得到的权重加权对应的值（V），实现信息的动态交互。类比来说，就像人在读文章时，会对关键信息分配更多注意力，而不是平均关注每个字。

根据 Q、K、V 的来源不同，Attention 可以衍生出多种机制，如 self-sttention、masked self-attention 和 cross-attention。在 Transformer 中，encoder 主要用 self-attention 来理解输入序列，decoder 在生成时既用 masked self-attention 保证自回归顺序，又用 cross-attention 参考 encoder 输出，实现源序列与目标序列的信息对齐和条件建模。

本文接下来的章节将详细介绍 Transformer 中的五类常用 Attention 机制，包括它们的公式、使用场景及在 Encoder/Decoder 中的角色。

### 3.1 注意力（Attention）

Attention 是一种神经网络机制，其核心目标是让模型在处理序列数据时动态选择性关注输入的不同部分。通过 Query（Q）匹配 Key（K），对 Value（V）进行加权，从而实现信息交互和特征增强。在读一篇文章时，人不会平均关注每个字，而是会根据任务（如找关键信息）对重要段落分配更多注意力。Attention 就像给每个信息片段分配“关注度”，权重越高代表越重要。

注意力有多种形式，Transformer依赖于**[缩放点积注意力](https://zhida.zhihu.com/search?content_id=261762936&content_type=Article&match_order=1&q=%E7%BC%A9%E6%94%BE%E7%82%B9%E7%A7%AF%E6%B3%A8%E6%84%8F%E5%8A%9B&zhida_source=entity)机制**：给定查询矩阵 Q、键矩阵 K 和值矩阵 V，输出是值向量的加权和，其中每个值的权重由查询向量与对应键向量的点积计算得到：

上面的 Attention 函数可以这样理解：softmax(...) 用于计算加权权重，V 是对应的值矩阵，Attention 的结果是一个与 V 维度相同的矩阵，也就是说公式不会改变值矩阵的维度。整个公式可以理解为：

加权权重值矩阵

其中，所有加权权重的和为 1。具体来说，  ​ 表示每个 Q 对所有 K 的相似度（得到实数矩阵），softmax 是一个函数，将每行相似度转换成权重概率，使每行和为 1，从而指导对 V 的加权求和。

对于一个查询向量和一个键向量  （分别是查询矩阵和键矩阵中的行向量），它们的点积会得到一个标量分数：

其中  表示第  个查询可以关注的键的位置集合。

Attention 是所有衍生注意力机制的基础表达式，它提供了一种通用的**信息选择与条件建模框架**：通过 Query 匹配 Key 并加权 Value，模型能够动态关注输入序列中最相关的信息。

从概念上来看，Attention 就像“母机制”，衍生出了 self-attention、masked self-attention、cross-attention 以及 multi-head attention 等不同变体，每个变体都是在 Q/K/V 来源或计算方式上的扩展和优化。Attention 的价值不仅在于捕捉序列内部依赖，也在于它为模型提供了灵活的**信息交互接口**，无论是同一序列内部、跨序列对齐，还是并行多子空间建模，都可以通过调整 Q/K/V 的来源和组合方式实现。

简而言之，Attention 是 Transformer 和现代序列模型的核心“计算引擎”，它的数学公式虽简单，但衍生出的多种机制为深度学习处理序列问题提供了高度灵活、高效且可解释的工具。

### **3.2 自注意力（Self-Attention）**

自注意力是注意力机制的一种特殊形式，模型在处理某个位置的输入时，会参考同一条输入序列中**其他位置的信息**来生成该位置的表示。直观上，这种方法类似于图像处理中的**非局部均值（Non-local Means）**：为了更新一个像素的值，不仅会看它周围的像素，还会看整张图里所有与它相似的区域。此外，自注意力在数学上是**置换不变的（Permutation-invariant）**，也就是说，如果不加位置编码，输入元素的顺序变化并不会影响结果。

自注意力机制是注意力机制的一种特殊形式，本质上他们的公式是相同的，只是由于自注意力中  都来自同一个序列，我们可以将它们统一表示为  ，公式因此简化为：

在实际的 Transformer 中，自注意力通常不会直接用  计算，而是先通过**线性映射**生成  、  、  ：

其中  是可学习的参数矩阵，维度通常为 或  。其作用是将输入向量映射到不同的子空间，从而方便计算注意力。

使用线性投影 W 的原因：

1. 区分  、  、  : 即使输入相同，通过不同的线性映射，每个 token 的  、  、  向量也可以捕捉不同信息。
2. 提升模型表达能力:线性投影使模型能够学习更丰富的关系，而不仅仅是原始输入的点积。
3. 支持多头注意力: 每个注意力头都有自己的一组  ​，允许模型在不同子空间关注不同模式，从而增强模型对序列信息的捕捉能力。

所以实际的self- attention的公式就是：

在 Transformer 中，Self-Attention 广泛应用于 Encoder 层和 Decoder 层。Decoder 的第一层通常使用 Masked Self-Attention，保证自回归生成时，每个位置只能依赖已生成的内容Self-Attention 的优势还包括高并行性和全局上下文捕捉能力，它可以有效建模长序列依赖，并为后续的 Cross-Attention 或 Multi-Head Attention 提供丰富的上下文表示。

简而言之，Self-Attention 是序列内部信息交互的核心机制，是 Transformer 捕捉上下文依赖和实现高效特征表示的基础模块。

### 3.3 多头自注意力机制（[Multi-Head Self-Attention](https://zhida.zhihu.com/search?content_id=261762936&content_type=Article&match_order=1&q=Multi-Head+Self-Attention&zhida_source=entity)）

多头自注意力机制是 Transformer 的关键组成部分。与只计算一次注意力不同，多头机制会通过不同的线性映射将输入投射到多个子空间（heads），然后在每个子空间中并行计算缩放点积注意力。每个头的独立注意力输出会被拼接起来，并通过线性映射转换回预期的维度。 多头注意力其中，

其中  表示拼接操作，  是权重矩阵，用于将输入嵌入向量映射为  大小的查询、键和值矩阵，  是输出的线性转换，每一个head都是一个self-attention单元。所有权重均在训练过程中学习得到。

![](https://pic1.zhimg.com/v2-f9cbaceec3dbee2fd81fe4be6843b232_1440w.jpg)

多头缩放点集注意力机制的解释（来自Lillian Wang）

需要注意的是，本小节讨论的是 **多头自注意力机制**，因此每个 head 都是一个 **自注意力单元**。而在 **多头注意力机制** 中，Q、K、V 可能来自不同序列，此时每个 head 是一个普通的 **注意力单元**。虽然两者的计算本质相同，但 Q、K、V 的来源会导致它们在应用上存在细微差别。

Multi-Head Self-Attention 是 Self-Attention 的扩展机制，每个“头”都是一个独立的 Self-Attention 单元，它们在不同子空间并行捕捉序列内部的多种依赖模式。通过将各头的输出拼接并线性映射回原始维度，模型能够整合多角度信息，形成更丰富的序列表示。换句话说，它 **保留了 Self-Attention 捕捉序列内部依赖的功能**，同时增强了模型在复杂序列中捕获多样化关系的能力。

在 Transformer 中，Multi-Head Self-Attention 广泛用于 Encoder 和 Decoder 的各层，是提高上下文建模能力和处理长序列依赖的核心模块。

### 3.4 跨序列注意力（cross-attention）

Cross-Attention 的核心目的是让一个序列“关注”另一个序列的信息。一个常见的例子就是机器翻译：

- 源语言句子（英文）由 Encoder 编码成隐藏表示；
- 目标语言句子（中文） 由 Decoder 逐步生成。

当 Decoder 要生成下一个中文词时，它需要参考源句子中与之最相关的部分。此时：

- Decoder 的 Query = 当前生成位置的上下文；
- Encoder 的 Key/Value = 整个源句子的编码结果。

这意味着 Cross-Attention 机制会在每一步生成中，让目标序列动态地“查阅”源序列的不同片段，从而保证翻译准确对齐。

如果换一个生活类比：Encoder 就像图书管理员，已经把书本（输入序列）整理好；Decoder 就像学生，正在写文章（输出序列）。Cross-Attention 就是学生在写作时，不断向图书管理员打听“这一段应该参考书的哪一页”。这样写出来的文章（翻译、总结、回答）才能和原书保持一致。

虽然 Cross-Attention 和普通 Attention 的数学表达式本质上相同，但因为 Q 来自 Decoder 而 K/V 来自 Encoder，所以它能够跨序列捕捉依赖关系。其数学表达式可以写作：

-  ：正在生成的句子（如中文翻译的一部分），用于生成 Query
-  ​：原文句子（如英文原文），用于生成 Key 和 Value

Cross-Attention 主要出现在 Transformer 的 Decoder 层，帮助生成目标序列时对齐源序列。相比仅使用自注意力，Cross-Attention 使模型能够在生成时参考源序列的信息，从而保证输出内容和输入语义一致。

### 3.5 掩码注意力（Masked Self—Attention）

Masked Self-Attention 是 Self-Attention 的一种变体，其目的是在序列生成时防止模型看到未来的 token，从而保证自回归的生成过程。一个常见的例子是机器翻译或语言建模：

- 输入序列（如 Decoder 已经生成的中文词）用于生成 Query、Key 和 Value。
- 在生成当前 token 时，我们只允许模型“看到”之前的 token，而不能看到未来的 token。

生活类比：如果学生在写文章，Masked Self-Attention 就像学生只能参考自己已经写过的内容，而不能偷看后面还没写的段落，以保证生成的顺序正确。

公式上，Masked Self-Attention 与 Self-Attention 相似，只是在 softmax 前加了一个遮罩矩阵 M： 

-  ：输入序列（当前 Decoder 已生成部分）
-  ：可学习的权重矩阵
-  ：上三角遮罩矩阵，将未来位置设置为  ，softmax 后对应权重为 0

Masked Self-Attention 主要用于 Transformer Decoder 的第一层或所有自注意力层，保证生成的序列是自回归的，不会提前看到未来 token，从而生成符合顺序和语义的文本。

## 四：总结

本文系统介绍了 Transformer 中的五类 Attention 机制：Attention、Self-Attention、Masked Self-Attention、Cross-Attention 和 Multi-Head Attention。

Attention 的核心思想是为序列中每个元素分配动态权重，从而增强模型对重要信息的关注。Self-Attention 捕捉序列内部依赖关系，Masked Self-Attention 保证生成序列的自回归顺序，Cross-Attention 允许 Decoder 在生成时参考 Encoder 输出，实现跨序列信息对齐，而 Multi-Head Attention 则通过并行多个子空间捕捉多种模式。

在 Transformer 架构中，Encoder 层主要使用 Self-Attention，Decoder 层使用 Masked Self-Attention 并在生成过程中结合 Cross-Attention，从而实现高效的序列建模和生成。这些机制的组合是 Transformer 能够捕捉长程依赖、提升特征表达能力并生成高质量序列输出的关键所在。

非常感谢耐心阅读，希望你在看完这篇文章后有所收获。

相关链接：[https://lilianweng.github.io/po](https://link.zhihu.com/?target=https%3A//lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/)



Reference:
Transformer Attention 家族全览：五大机制解析与对比 - 野蛮生长的文章 - 知乎
https://zhuanlan.zhihu.com/p/1939735076159072227