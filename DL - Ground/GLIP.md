
GLIP：统一目标检测和视觉定位 ground data带来更大飞跃 - 越洋飞机的文章 - 知乎
https://zhuanlan.zhihu.com/p/4291962647


![[Pasted image 20250724145334.png]]
GLIP 模型全貌


甚麼是Grounding?
？grounding的全称是 **phrase grounding** ，即短语定位，更准确地说是 **识别句子中的短语与图像中的对象（或区域）之间的细粒度对应关系** 。这听起来就跟目标检测很像对不对，目标检测所需要的输入是一张图片，而phrase grounding的输入是一句话和一张图片

怎么证明grounding和detection实际上是一件事呢？也很好办，就是把单独的词当成grounding的输入，最后发现训练出来的模型准确度与detection模型一模一样，当然这是在控制了其他变量都一样的情况下。这就说明，grounding和detection确实做的是一件事，我们也可以吧grounding理解成一种更高级的detection。
![[Pasted image 20250724144244.png]]



比較CLIP的預訓練方法  vs phrase grounding当中，计算图像区域与词汇的对齐分数（alignment grade）方法
![[Pasted image 20250724144716.png]]

![[Pasted image 20250724144755.png]]在GLIP当中，将语义信息和图像信息进行协同训练的架构




Prompt tuning对于deep fusion是非常有用的，而对浅层融合则没那么有用。对于过去的模型，他们大多使用的是late fusion，也就是在计算alignment scores时才将图像编码和文本编码汇聚在一起，这就是浅层融合。所以，如果想要利用Prompt的力量，我们自然需要deep fusion，这样有利于模型学习到更深层次的语义信息，并在下游任务上具有更好的应用。其具体做法如下图所示
![[Pasted image 20250724145119.png]]



![[Pasted image 20250724234912.png]]