


**[NMS](https://zhida.zhihu.com/search?content_id=218629256&content_type=Article&match_order=1&q=NMS&zhida_source=entity)是计算机视觉中常用的后处理（Post-Procession）算法。本文主要介绍传统经典的NMS、[Soft NMS](https://zhida.zhihu.com/search?content_id=218629256&content_type=Article&match_order=1&q=Soft+NMS&zhida_source=entity)、[Softer NMS](https://zhida.zhihu.com/search?content_id=218629256&content_type=Article&match_order=1&q=Softer+NMS&zhida_source=entity)的应用场景、基本原理、实现代码、优缺点等 。其他类型将在后续文章陆续进行介绍。**

  
_**一、 标准NMS**_

**1. 概述**

**非极大值抑制（Non-maximum supression）简称NMS，其作用是去除冗余的检测框，去冗余手段是剔除与极大值重叠较多的检测框结果**。 通常我们所说的NMS指的是标准NMS。那么为什么一定要去冗余呢？因为图像中的目标是多种多样的形状、大小和长宽比，目标检测算法中为了更好的保障目标的召回率，通常会使用[SelectiveSearch](https://zhida.zhihu.com/search?content_id=218629256&content_type=Article&match_order=1&q=SelectiveSearch&zhida_source=entity)、[RPN](https://zhida.zhihu.com/search?content_id=218629256&content_type=Article&match_order=1&q=RPN&zhida_source=entity)（例如：[Faster-RCNN](https://zhida.zhihu.com/search?content_id=218629256&content_type=Article&match_order=1&q=Faster-RCNN&zhida_source=entity)）、Anchor（例如：[YOLO](https://zhida.zhihu.com/search?content_id=218629256&content_type=Article&match_order=1&q=YOLO&zhida_source=entity)）等方式生成长宽不同、数量较多的候选边界框（BBOX）。因此在算法预测生成这些边界框后，紧接着需要跟着一个NMS后处理算法，进行去冗余操作，为每一个目标输出相对最佳的边界框，依次作为该目标最终检测结果。

![[Pasted image 20250316142318.png]]

**2. 思路**

**核心思想**是搜索目标局部范围内的边界框置信度最大的这个最优值，去除目标邻域内的冗余边界框。

**3.具体步骤**

**一般NMS后处理算法需要经历以下步骤（不含背景类，背景类无需NMS）：**

step1：先将所有的边界框按照类别进行区分；

step2：把每个类别中的边界框，按照置信度从高到低进行降序排列；

step3：选择某类别所有边界框中置信度最高的边界框bbox1，然后从该类别的所有边界框列表中将该置信度最高的边界框bbox1移除并同时添加到输出列表中；

step4：依次计算该bbox1和该类别边界框列表中剩余的bbox计算[IOU](https://zhida.zhihu.com/search?content_id=218629256&content_type=Article&match_order=1&q=IOU&zhida_source=entity)；

step5：将IOU与NMS预设阈值Thre进行比较，若某bbox与bbox1的IOU大于Thre，即视为bbox1的“邻域”，则在该类别边界框列表中移除该bbox，即去除冗余边界框；

step6：重复step3~step5，直至该类别的所有边界框列表为空，此时即为完成了一个物体类别的遍历；

step7：重复step2~step6，依次完成所有物体类别的NMS后处理过程；

step8：输出列表即为想要输出的检测框，NMS流程结束。












Reference: 
初探NMS非极大值抑制 - PoemAI的文章 - 知乎
https://zhuanlan.zhihu.com/p/587225859

