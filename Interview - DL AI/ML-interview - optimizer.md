
Ref: [每天3分钟，彻底弄懂神经网络的优化器optimizer](https://www.zhihu.com/people/luhengshiwo/posts)
Ref: [机器学习2 -- 优化器（SGD、SGDM、Adagrad、RMSProp、Adam](https://zhuanlan.zhihu.com/p/208178763)）
Ref: [深度学习各类优化器](https://github.com/zonechen1994/CV_Interview/blob/main/%E9%80%9A%E7%94%A8%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89%E7%AE%97%E6%B3%95%E9%9D%A2%E7%BB%8F/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80%E9%9D%A2%E8%AF%95%E9%A2%98/%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95.md)

| Optimizer                                |                                                                                                                                                                                                                                                                                   |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| SGD<br>(Stochastic <br>Gradient Descent) | 随机梯度下降                                                                                                                                                                                                                                                                            |
| **SGDM**<br>(SGD with momentum)          | SGDM即为SGD with momentum，它加入動量機制<br><br>Slow, Better convergence, stable, <br>smaller generalization gap<br><br>**Computer vision**<br>(image classification, <br>segmentation, <br>object detection)                                                                              |
| RMSProp                                  | 它与Adagrad基本类似，只是加入了迭代衰减                                                                                                                                                                                                                                                           |
| Adagrad                                  | 它利用迭代次数和累积梯度，对学习率进行自动衰减. <br>与SGD的区别在于，学习率除以 前t-1 迭代的梯度的平方和。<br>故称为自适应梯度下降。                                                                                                                                                                                                       |
| **Adam**                                 | Adam是SGDM和RMSProp的结合，<br>它基本解决了之前提到的梯度下降的一系列问题，<br>比如随机小样本、自适应学习率、容易卡在梯度较小点等问题<br><br>Fast, Possibly non-convergence,<br>unstable, larger generalization gap<br><br>**NLP**<br>(QA, summary,<br>machine translation)<br><br>Speech synthesis<br>**GAN**<br>Reinforcement learning |

如上所示，SGDM在CV里面应用较多，而Adam则基本横扫NLP、RL、GAN、[语音合成](https://zhida.zhihu.com/search?content_id=134396393&content_type=Article&match_order=1&q=%E8%AF%AD%E9%9F%B3%E5%90%88%E6%88%90&zhida_source=entity)等领域。所以我们基本按照所属领域来使用就好了。比如NLP领域，Transformer、BERT这些经典模型均使用的Adam，及其变种AdamW。

optimizer优化主要有三种方法

1. 让模型探索更多的可能，包括dropout、加入Gradient noise、样本shuffle等
2. 让模型站在巨人肩膀上，包括warn-up、curriculum learning、[fine-tune](https://zhida.zhihu.com/search?content_id=134396393&content_type=Article&match_order=1&q=fine-tune&zhida_source=entity)等
3. 归一化 normalization，包括batch-norm和[layer-norm](https://zhida.zhihu.com/search?content_id=134396393&content_type=Article&match_order=1&q=layer-norm&zhida_source=entity)等
4. 正则化，惩罚模型的复杂度