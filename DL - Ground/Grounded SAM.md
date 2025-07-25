

全自动标注集成项目（Grounded-SAM）技术报告阅读:Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks - 共由的文章 - 知乎
https://zhuanlan.zhihu.com/p/680745667

我们引入了Grounded SAM，它使用[Grounding DINO](https://zhida.zhihu.com/search?content_id=239406813&content_type=Article&match_order=1&q=Grounding+DINO&zhida_source=entity) [38] 作为开放集对象检测器，并与任何分割模型（SAM）[25] 相结合。**这种整合可以根据任意文本输入检测和分割任何区域**，并为连接各种视觉模型打开一扇大门。如图 1 所示，通过使用通用的 Grounded SAM 管道，可以实现多种视觉任务。例如，一个完全基于输入图像的自动注释管道可以通过合并诸如 [BLIP](https://zhida.zhihu.com/search?content_id=239406813&content_type=Article&match_order=1&q=BLIP&zhida_source=entity) [31]和Recognize Anything[83]这样的模型来实现。此外，结合Stable-Diffusion[52]允许可控的图像编辑，而 [OSX](https://zhida.zhihu.com/search?content_id=239406813&content_type=Article&match_order=1&q=OSX&zhida_source=entity) [33]的集成促进了及时的3D 人体运动分析。Grounded SAM 在开放词汇基准方面也表现出卓越的性能，通过 GroundingDINO-Base 和 SAM-Huge 模型的组合，在 SegInW (自然分割)零拍基准上实现了48.7平均 AP。

![[Pasted image 20250725013523.png]]