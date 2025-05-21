
| Model               | Parameters: LR, batch size, Epoch, weight decay                                              |
| ------------------- | -------------------------------------------------------------------------------------------- |
| VGG                 | 3x3 kernel size replace 5x5,7x7                                                              |
| [[ResNet]]深度殘差網絡    | Residual block, skip connection, vanishing gradient, exploding gradient, degradation problem |
| [[FCN]] 全卷積網絡       | (~ UNet) Fully Convolutional Network, Transpose layer(upsampling)                            |
| [[DarkNet]]         | YOLO backbone, CSPDarkNet(Cross stage partial network)                                       |
| [[lightweight CNN]] | MobileNet(Depthwise Separable Convolution), EfficientNet                                     |
|                     |                                                                                              |
| others              | [[vanishing gradient, exploding gradient, overfit]]                                          |
| [[### QA list]]     |                                                                                              |

| Layers              |                                                                                                                                                                                                                                                                                                                                                                                   |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 卷積核 conv<br>[[CNN]] | Parameters: padding, stride<br>Kernel Size, In Channels, Out Channels<br><br>[[CNN architecture]]<br>[[AI model Summary architecture]]<br><br>[[1x1 conv]]<br>[dilated conv 擴張卷積](https://zhuanlan.zhihu.com/p/585500690) (Receptive Field)<br>[transposed Conv 轉置卷積](https://zhuanlan.zhihu.com/p/28186857)<br>[[Depthwise Separable Convolution]]<br>3x3x3x16 -> 3x3x3+3x16<br> |
| Normalization       | Batch Normalization<br>Layer Normalization                                                                                                                                                                                                                                                                                                                                        |
| 激活Activate          | [[Activation funs]]                                                                                                                                                                                                                                                                                                                                                               |
| 池化層 pool            | max pooling                                                                                                                                                                                                                                                                                                                                                                       |
| Dropout層            | [[Normalization and dropout]]                                                                                                                                                                                                                                                                                                                                                     |
| 全連接層 FC             | Fully connected layer                                                                                                                                                                                                                                                                                                                                                             |

| Block                     |                  |
| ------------------------- | ---------------- |
| Convolution Block         |                  |
| Inception                 |                  |
| Residual Block (ResNet)   | Skip connections |
| Bottleneck Block (ResNet) |                  |
| Attention Block           |                  |
|                           |                  |

| Others                          |                                                                                                                                       |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| Receptive Field(感受野)            |                                                                                                                                       |
|                                 |                                                                                                                                       |
| object detection anchor正負樣本比例不均 | 1. object detection/tracking, instance segmentation<br>2. 有Anchor或proposal通常有正負樣本比例不均問題, 也需要NMS<br>3. Anchor free則沒有正負樣本問題也不需要NMS<br> |
| [[Hard Negative Mining]]        | 選擇困難負(hard negative)樣本放入訓練                                                                                                            |
| [[Focal Loss]]                  | 通過調整損失函數的權重，使模型更關注於難分類樣本的學習. <mark style="background: #FF5582A6;">@要會公式跟手寫!</mark>                                                    |
|                                 |                                                                                                                                       |

參考: [[第五章_卷积神经网络(CNN)]]


![[Pasted image 20250316215533.png]]
![[Pasted image 20250316215548.png]]



Reference:
CNN卷积核与通道讲解 - 双手插袋的文章 - 知乎
https://zhuanlan.zhihu.com/p/251068800




### QA list

| Q                                                     | Ans             |
| ----------------------------------------------------- | --------------- |
| PyTorch 中 nn.eval 函数和训练的区别，BN，dropout 训练和测试的区别        | [[##### Ans1]]  |
| == 如何在fine-tuning后避免模型遗忘之前的知识==                       | [[##### Ans2]]  |
| Coding>> **2D convolution**                           |                 |
| 甚麼是overfitting, 甚麼是underfitting. 抑制overfitting的方法     | [[##### Ans4]]  |
| 类别不均衡解决方法, 样本不平衡的处理方法                                 |                 |
| 抑制overfitting: 数据增强(增加样本的数量), early stopping, Dropout |                 |
| 1*1卷积原理和作用？                                           |                 |
| Batch大小如何选择？                                          |                 |
| 如何解决数据集的正负样本不平衡的问题？                                   | [[##### Ans11]] |
| 有时候会采用步长 为2的卷积操作来代替max pooling，比较两者有什么不同？             |                 |
| CNN的经典模型: LeNet，AlexNet，VGG，GoogLeNet，ResNet，DenseNet |                 |
| CNN和传统的全连接神经网络有什么区别？                                  |                 |
| 讲一下CNN，每个层及作用                                         |                 |
| 为什么神经网络使用卷积层？-共享参数，局部连接                               |                 |
| 数据增强有哪些方法                                             |                 |
| padding的作用                                            |                 |
| pooling如何反向传播? Pooling的作用和缺点                          |                 |
| 反向传播的原理                                               |                 |
| 给定卷积核的尺寸，特征图大小计算方法？                                   |                 |
| 共享参数有什么优点                                             |                 |
| 空洞卷积是什么？有什么应用场景？                                      |                 |
| 训练一个二分类任务，其中数据有80%的标注正确，20%标注失败                       |                 |
| 视觉任务中的长尾问题的常见解决方案                                     |                 |
| 有哪些权重初始化的方法                                           |                 |
| Transformer/CNN/RNN的时间复杂度对比                           |                 |
| 深度可分离卷积                                               |                 |
| CNN和MLP的区别                                            |                 |
| 深度学习训练中如何区分错误样本和难例样本                                  |                 |
| 深度学习模型训练时的Warmup预热学习率作用                               |                 |
| PyTorch 节省显存的常用策略                                     |                 |
| PyTorch中的 ModuleList 和 Sequential的区别和使用场景             |                 |
| 深度学习中为什么不对 bias 偏置进行正则化？                              |                 |
| 深度学习模型中如何融入传统图像处理的特征？直接拼接融合有什么问题？                     |                 |

##### Ans1
**請解釋一下 PyTorch 中 `nn.eval()` 函數的作用以及它與訓練模式的主要區別。同時，請具體說明 Batch Normalization 和 Dropout 在訓練和測試階段的行為差異。**

**你：** 好的。`nn.eval()` 函數在 PyTorch 中用於將模型設置為評估（或測試）模式。這會影響模型中特定層（如 Batch Normalization 和 Dropout）的行為，使其在評估時表現出與訓練時不同的方式。與訓練模式的主要區別在於以下幾個方面：

**1. 梯度計算：**

- **訓練模式 (`model.train()`):** 模型會追蹤計算圖，啟用梯度計算。這是因為在訓練過程中，我們需要計算損失函數相對於模型參數的梯度，並使用這些梯度來更新模型權重。
- **評估模式 (`model.eval()`):** 模型會禁用梯度計算。這是因為在評估或測試階段，我們只需要模型的前向傳播結果，而不需要更新權重，因此不需要計算梯度，這樣可以節省計算資源並提高運行速度。

**2. Batch Normalization (BN) 層的行為：** -> model.eval() 不會重新計算Batch normalization

- **訓練模式 (`model.train()`):**
    - BN 層會計算每個批次 (batch) 輸入的均值和標準差，並使用這些統計量對該批次的數據進行歸一化。
    - 同時，BN 層還會維護一個**移動平均 (running average)** 的均值和標準差，這些統計量會在每個訓練批次更新，用於估計整個訓練數據集的均值和標準差。
    - 模型的可學習參數 `weight` (縮放因子) 和 `bias` (平移因子) 會在訓練過程中進行更新。
- **評估模式 (`model.eval()`):**
    - BN 層**不會**重新計算當前批次的均值和標準差。
    - 相反，它會使用在**訓練階段計算並保存下來的移動平均的均值和標準差**來對輸入數據進行歸一化。
    - 模型的可學習參數 `weight` 和 `bias` 仍然會被應用，但它們在評估階段是固定的，不會被更新。

**這樣做的原因是，在評估階段，我們希望模型的輸出僅依賴於輸入數據和學習到的參數，而不是當前批次的統計信息。使用整個訓練集的統計估計可以使模型的預測更加穩定。**

**3. Dropout 層的行為：** -> model.eval() 完全禁用Dropout

- **訓練模式 (`model.train()`):**
    - Dropout 層會以預設的概率隨機將輸入張量的一些元素置零。這是一種正則化技術，有助於防止模型過度擬合訓練數據，迫使模型學習更魯棒的特徵表示。
- **評估模式 (`model.eval()`):**
    - Dropout 層會**完全禁用**。所有輸入元素都會通過 Dropout 層而不被置零。
    - 或者，在一些 Dropout 的變體實現中，訓練時的輸出會在數值上進行縮放 (除以保留概率)，以便在評估時可以直接使用所有神經元，而無需顯式禁用。PyTorch 的 `nn.Dropout` 默認行為是禁用。

**禁用 Dropout 的原因是，在評估階段，我們希望模型能夠利用所有學習到的權重和連接進行預測，以獲得最準確的結果。隨機丟棄神經元會引入不確定性，這在評估時是不希望看到的。**

**總結來說，`nn.eval()` 的主要作用是切換模型的行為模式，使其適用於評估或測試。這主要體現在禁用梯度計算，以及改變 Batch Normalization 層使用全局統計量和禁用 Dropout 層的隨機失活行為。在進行模型評估或測試之前，務必調用 `model.eval()`，以確保模型以正確的方式運行並產生可靠的評估結果。在訓練結束後，保存模型之前通常也會調用 `model.eval()`，以確保保存的模型在部署時的行為是確定的。**

希望這個回答能夠清晰地解釋 `nn.eval()` 的作用以及 BN 和 Dropout 在訓練和測試階段的區別。



##### Ans2
Q: 如何在fine-tuning后避免模型遗忘之前的知识

避免在微调后模型遗忘先前知识（也称为灾难性遗忘）是一个重要的研究领域。这里有一些常用的策略和技巧，可以帮助你在微调过程中更好地保留模型的已有能力：

**1. 数据策略:**

- **混合训练数据 (Mixed Training Data):** <mark style="background: #BBFABBA6;">在微调数据集之外，少量地包含来自原始预训练数据集或类似任务的数据</mark>。这有助于模型回忆起先前学习到的特征和模式。
- **平衡微调数据 (Balanced Fine-tuning Data):** 确保微调数据集在类别和分布上相对平衡，避免模型过度适应新任务而忘记旧任务。
- **回放 (Replay):** 存储少量来自先前任务或预训练阶段的代表性样本，并在微调新任务时与新数据混合训练。这可以帮助模型“回忆”起旧知识。
- **生成回放 (Generative Replay):** 使用一个生成模型（例如，VAE或GAN）来生成先前任务的合成数据，并将其与新任务的数据一起训练。这避免了存储实际旧数据的需求。

**2. 模型架构与正则化:**

- **弹性权重合并 (Elastic Weight Consolidation, EWC):** 在微调过程中，对模型参数的更新施加约束，使得参数不会大幅偏离先前任务的最优值。EWC通过计算先前任务损失的Fisher信息矩阵来估计参数的重要性，并对重要的参数施加更强的惩罚。 L(θ)=LD​(θ)+λi∑​Fi​(θi​−θi∗​)2 其中，LD​(θ)是新任务的损失函数，Fi​是参数θi​的Fisher信息，$ \theta_i^{*}是先前任务学习到的最优参数，\lambda$是控制遗忘程度的超参数。
- **知识蒸馏 (Knowledge Distillation):** 使用预训练模型（教师模型）的输出来指导微调模型（学生模型）的训练。学生模型不仅学习新任务的标签，还学习模仿教师模型的预测分布，这有助于保留教师模型学习到的知识。
- **参数冻结 (Parameter Freezing):** <mark style="background: #BBFABBA6;">冻结预训练模型的部分或大部分层，只微调特定的层</mark>（通常是靠近输出的层）。这可以防止模型的核心知识被新任务的训练完全覆盖。
- **添加专门的记忆模块 (Adding Memory Modules):** 在模型中引入额外的记忆模块（例如，神经图灵机、记忆网络），这些模块可以显式地存储和检索先前学习到的信息，从而减少对模型主体参数的干扰。
- **正则化技术 (Regularization Techniques):** <mark style="background: #BBFABBA6;">应用L1或L2正则化、Dropout等技术，限制模型参数的复杂性，防止过拟合到新任务</mark>，从而在一定程度上保留泛化能力和先前知识。

**3. 训练过程:**

- **逐步微调 (Progressive Fine-tuning):** 如果有多个相关任务，<mark style="background: #BBFABBA6;">可以逐步地在每个任务上进行微调，而不是一次性在一个差异较大的新任务上进行微调</mark>。这有助于模型更平滑地适应新知识，同时保留旧知识。
- **多任务学习 (Multi-task Learning):** 如果新任务与先前学习的任务相关，可以尝试将<mark style="background: #BBFABBA6;">它们一起进行多任务学习</mark>。通过共享模型参数，可以促进知识的迁移和保留。
- **动态结构调整 (Dynamic Architectures):** 一些研究探索在微调过程中动态地调整模型结构，例如添加新的神经元或模块来适应新任务，同时保留旧模块的功能。

**4. 评估指标:**

- **使用综合评估指标 (Comprehensive Evaluation Metrics):** 除了在新任务上的性能，还要定期评估模型在先前任务或代表性数据集上的性能，以监控遗忘的程度。可以使用诸如平均准确率下降、遗忘率等指标。

**选择哪种策略取决于具体的应用场景、模型架构、数据可用性以及计算资源。通常情况下，结合多种策略可以获得更好的效果。** 例如，可以结合混合训练数据和弹性权重合并，或者使用知识蒸馏并冻结部分模型参数。

在实践中，你需要进行实验和调优，找到最适合你特定任务和模型的避免遗忘策略。理解模型在微调过程中是如何学习和遗忘的，将有助于你更有效地应用这些技术。


##### Ans4

Q: 甚麼是overfitting, 甚麼是underfitting. 抑制overfitting的方法

(1) L1/[L2正则化](https://zhida.zhihu.com/search?content_id=237052607&content_type=Article&match_order=1&q=L2%E6%AD%A3%E5%88%99%E5%8C%96&zhida_source=entity)。

(2) 数据增强：平移、旋转、翻转、随机裁剪等。

(3) Dropout：训练时神经元以概率_p_置0（推理时的计算方式：输入乘以1-_p_）。

(4) [Early stopping](https://zhida.zhihu.com/search?content_id=237052607&content_type=Article&match_order=1&q=Early+stopping&zhida_source=entity)：早停法，当模型在验证集上的表现下降的时候，停止训练。

_附加实战经验：_

_(1) [BatchNorm层](https://zhida.zhihu.com/search?content_id=237052607&content_type=Article&match_order=1&q=BatchNorm%E5%B1%82&zhida_source=entity)冻结，否则小模型容易过拟合。_

_(2) 小batch size先训练。_

_(3) 减少可训练层的数目。_


##### Ans5

[L1正则化](https://zhida.zhihu.com/search?content_id=237052607&content_type=Article&match_order=1&q=L1%E6%AD%A3%E5%88%99%E5%8C%96&zhida_source=entity): loss计算时加一项，参数的_L1范数_，各元素的绝对值之和。得到的参数会更加稀疏。

L2正则化： loss计算时加一项，参数的_L2范数_，各元素的平方和。使网络权重倾向于选择更小的值，这样不同特征对结果的影响相对均衡，不会受一些噪点影响，提升了泛化性。

_附加实战经验：_

_(1) [ResNet50](https://zhida.zhihu.com/search?content_id=237052607&content_type=Article&match_order=1&q=ResNet50&zhida_source=entity)衰减系数一般1e-4，移动端小网络如[MobileNet](https://zhida.zhihu.com/search?content_id=237052607&content_type=Article&match_order=1&q=MobileNet&zhida_source=entity)一般1e-5。_

_(2) L2系数太大抑制过拟合，但可能会欠拟合，尤其对于小网络参数量不足的情况。_

_(3) 数据集小时，网络容易过拟合，这时L2系数可以适当调大。_



##### Ans11
**_【22】如何解决数据集的正负样本不平衡的问题？_**

**1、采样**

采样分为[上采样](https://zhida.zhihu.com/search?content_id=167439930&content_type=Article&match_order=1&q=%E4%B8%8A%E9%87%87%E6%A0%B7&zhida_source=entity)（Oversampling）和下采样（Undersampling），上采样是把小众类复制多份，下采样是从大众类中剔除一些样本，或者说只从大众类中选取部分样本；

**2、数据合成**

数据合成方法是利用已有样本生成更多样本（无中生有），这类方法在小数据场景下有很多成功案例，比如[医学图像分析](https://zhida.zhihu.com/search?content_id=167439930&content_type=Article&match_order=1&q=%E5%8C%BB%E5%AD%A6%E5%9B%BE%E5%83%8F%E5%88%86%E6%9E%90&zhida_source=entity)等。

**3、加权**

**4、分类**

对于正负样本极不平衡的场景，我们可以换一个完全不同的角度来看待问题：把它看做一分类（One Class Learning）或异常检测（Novelty Detection）问题。这类方法的重点不在于捕捉类间的差别，而是为其中一类进行建模，经典的工作包括One-class SVM等。

解决数据不平衡问题的方法有很多，上面只是一些最常用的方法，而最常用的方法也有这么多种，如何根据实际问题选择合适的方法呢？接下来谈谈一些实际经验。

  

- 在正负样本都非常之少的情况下，应该采用[数据合成](https://zhida.zhihu.com/search?content_id=167439930&content_type=Article&match_order=3&q=%E6%95%B0%E6%8D%AE%E5%90%88%E6%88%90&zhida_source=entity)的方式；
- 在负样本足够多，正样本非常之少且比例及其悬殊的情况下，应该考虑一分类方法；
- 在正负样本都足够多且比例不是特别悬殊的情况下，应该考虑采样或者加权的方法。
- 采样和加权在数学上是等价的，但实际应用中效果却有差别。尤其是采样了诸如Random Forest等分类方法，训练过程会对训练集进行随机采样。在这种情况下，如果计算资源允许上采样往往要比加权好一些。
- 另外，虽然上采样和下采样都可以使数据集变得平衡，并且在数据足够多的情况下等价，但两者也是有区别的。实际应用中，我的经验是如果计算资源足够且小众类样本足够多的情况下使用上采样，否则使用下采样，因为上采样会增加训练集的大小进而增加训练时间，同时小的训练集非常容易产生过拟合。
- 对于下采样，如果计算资源相对较多且有良好的并行环境，应该选择Ensemble方法