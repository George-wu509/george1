
TrackFormer解读 - 周先森爱吃素的文章 - 知乎
https://zhuanlan.zhihu.com/p/356966934

**TrackFormer就是將DETR的Encoder-Decoder中的Decoder的object detection query改成track query**


在人群密集的场景中跟踪多个目标其实是非常大的挑战，因为对人而言集中注意力关注一个目标已经不是很容易的事情，因此将Transformer这种强大的自注意力引入MOT任务中，是一种很好的选择。而随着图像级别的目标检测器的发展，大部分MOT方法都采用tracking-by-detection的思路，它主要分为两步：首先，在单帧上检测出目标，接着在帧间检测结果之间进行关联，从而形成沿着时间的轨迹。

很多方法在数据关联这一步做了研究，关联其实也才是跟踪的核心问题。传统的TBD（tracking-by-detection）方法关联检测框采用时序的稀疏或密集图优化方法，或者使用卷积神经网络预测检测框之间的匹配分数。最近的一些工作则产生了一些新的思路，如tracking-by-regression方法，这类方法中检测器不仅仅输出逐帧的检测结果，同时还取代了数据关联步骤，以轨迹位置变化的回归来实现。这类方法隐式地进行数据关联，不过还是依赖于额外的图优化或者运动和外观模型。这在很大程度上是由于缺乏对象标识和局部边界框回归的概念。

这篇论文中，作者提出了一种将MOT问题视为tracking-by-attention的方法，即TrackFormer。TrackFormer以一个统一的方式同时实现目标检测和数据关联，如下图所示，TrackFormer基于CNN和Transformer（基于DETR中的结构）实现轨迹的形成。就像我之前在**[Transformer的文章](https://link.zhihu.com/?target=https%3A//zhouchen.blog.csdn.net/article/details/107006263)**中所说，这篇论文的主要工作其实也集中在decoder的query上。它通过新提出的track query以自回归方式在视频序列中在空间和时间上跟踪一个对象。在每一帧上，模型会对多个track query进行变换，这个track query表示对应目标的空间位置。Transformer在帧级特征和track query上执行注意力操作，以推理目标位置和身份（伴随遮挡）以及新物体的出现。新目标的出现是同一个Transformer以统一的方式为新进入场景的目标生成track query来实现的。

![](https://pic1.zhimg.com/v2-9c9d5d65db985d0e06158582fd7dcb1e_1440w.jpg)

TrackFormer可以端到端训练完成检测和跟踪任务，和DETR类似，优化目标也是一个集合预测损失。它实现了与注意力的隐式关联轨迹，不需要额外的匹配，优化或者运动和外观的建模。在benchmark评估中，将TrackFormer应用到MOT17数据集上，它达到了SOTA表现。此外，我们展示了我们的模型输出分割掩模的灵活性，并展示了多目标跟踪和分割（MOTS20）上SOTA成果。

## **TrackFormer**

TrackFormer的核心其实是提出了一个track query的东西，这和TransTrack中的learned query结合track feature的思路非常类似，它的整体结构如下图所示。

![](https://pic2.zhimg.com/v2-badd85ec9093cd3d10fef705298d8c5b_1440w.jpg)

论文首先回顾了DETR并进行了一些数学上的推导，我这里就在已了解DETR的基础上直接这篇论文的工作了。首先我们看上图最左侧初始帧上，其实这就是标准的DETR过程（因此后面CNN特征送入encoder就不讲解了），直接看右侧decoder过程，最下面的白色框框表示learnable object query，共有个（一般大于单帧最大目标数），它会查询到对应数目的output embedding，这个object embedding一方面用于后续检测任务的head中，如边框回归和类别预测。对那些成功预测出目标（即非背景类）的output embedding（图上的红色、绿色和蓝色框），还将其初始化为传入下一帧的track query。**这里需要注意的是，不算head层，其实Transformer结构会将输入进行频繁的自注意力，但是输出获得内容信息的embedding维度是不变的，这里可以理解为一个黑盒，因此这里初始化的track query其实和后面每帧为了新目标检测使用的object query是同维的。**

下面，对于不是第一帧的后续帧而言，decoder输入的query不仅仅有每帧初始化用于检测的个object query，还有上一帧已经成功检测的目标的个track query（显然它不是固定的，而是依赖于上一帧的），共有个query，decoder接受这个query之后查询到当前帧的检测结果（我这里的理解其实是，object query用于空间上查询，track query带有时序信息进行查询），所以指导训练的集合预测损失变为下式，其中表示最接近的预测和GT。

因此，track query查到的目标如果成功检测到了，那就赋予同一个id（如上图中间部分的前面的红绿蓝框），没检测到则表示目标消失（如上图右边部分的蓝色框），那些object query检测成功（非背景类，上图中不打叉的）的则作为新目标。接着，这些新旧目标的output embedding一起作为下一帧的track query。**这样，以一种相对优雅的方式完成了数据关联以致整个跟踪任务。**

![](https://pic3.zhimg.com/v2-fbfae0f314378da946de82378c21bfc2_1440w.jpg)

从Transformer的结构上来看，其结构如上图所示，这很容易看懂，我就不多赘述了。这里唯一需要说明的是，其实track query并非完全和object query同等对待的，它先经过了一个额外的多头自注意力层（track query attention）来对其预处理，这个操作的理由其实是因为前一帧的output embedding用于目标的分类和回归任务，与objectquery并不在一个空间内。经过一个变换再和object query进行concat会好一些。

从整体上来看，TrackFormer可以被理解为一个基于自适应回归track query的持续的重检测跟踪目标的过程。其实无论是object query还是track query，其实query都是潜在的当前帧目标的表示，只是track query带有时序信息而已，而将其引入下一帧的检测，当注意力在整个集合上允许时，会自动避免重复目标的检测。此外，从整体上看，只要轨迹一直存在，其实目标的track query其实会被逐帧更新，TrackFormer因此也实现了一种隐式的多帧注意力。