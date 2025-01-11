[1] 
2024.10.30<mark style="background: #BBFABBA6;"> **LongVU** - Multimodal LLM (QA, summary, identify from video)</mark> - Meta
[Linkedin post](https://www.linkedin.com/feed/update/urn:li:activity:7257400108944699392/)

Let's go!! [Meta](https://www.linkedin.com/company/meta/) released a new video LLM on [Hugging Face](https://www.linkedin.com/company/huggingface/), and it sets a new SOTA (state-of-the-art) for open-source video understanding. 🔥  
  
The model is called LongVU, a new multimodal large language model capable of processing long videos (for things like answering questions about it, summarizing it, identifying important passages, etc).

Reference
[Project link](https://vision-cair.github.io/LongVU/)
[huggingface link](https://huggingface.co/spaces/Vision-CAIR/LongVU)


------------------------------------------------------------------
[2]
2024.10.30  <mark style="background: #BBFABBA6;">NotebookLM and Illuminate - speech generation model</mark> - DeepMind
[Linkedin post ](https://www.linkedin.com/feed/update/urn:li:activity:7257407562084446209/)

We recently helped develop two tools: NotebookLM and Illuminate to narrate articles, generate stories, and even create multi-speaker discussions. Here’s how the technology works:  
  
Our latest speech generation model can produce 2️⃣ minutes of dialogue.

Reference
[DeepMind post](https://deepmind.google/discover/blog/pushing-the-frontiers-of-audio-generation/?utm_source=linkedin&utm_medium=social&utm_campaign=&utm_content=)


------------------------------------------------------------------
[3]
2024.10.30  <mark style="background: #BBFABBA6;">Layer Skip - end-to-end solution for accelerating LLMs</mark> - Meta
[Linkedin post](https://www.linkedin.com/feed/update/urn:li:activity:7257093132419256321/)

We previously shared our research on Layer Skip, an end-to-end solution for accelerating LLMs from researchers at Meta FAIR. It achieves this by executing a subset of an LLM’s layers and utilizing subsequent layers for verification and correction. We’re now releasing inference code and fine-tuned checkpoints for this work.

Reference
[huggingface model link](https://huggingface.co/collections/facebook/layerskip-666b25c50c8ae90e1965727a)
[Meta post](https://ai.meta.com/blog/fair-news-segment-anything-2-1-meta-spirit-lm-layer-skip-salsa-lingua/?utm_source=linkedin&utm_medium=organic_social&utm_content=video&utm_campaign=fair)


------------------------------------------------------------------
[4]
2024.10.30 
OmniParser - turning screenshots of UIs into structured data - Microsoft
Ferret-UI - turning screenshots of UIs into structured data - Apple
[Linkedin](https://www.linkedin.com/feed/update/urn:li:activity:7256617300131287041/)

Both [Microsoft](https://www.linkedin.com/company/microsoft/) and [Apple](https://www.linkedin.com/company/apple/) released interesting new multimodal models on [Hugging Face](https://www.linkedin.com/company/huggingface/) recently which do a very similar thing: turning screenshots of user interfaces (UIs) into structured data.
Microsoft released OmniParser. OmniParser consists of 2 models, applied in sequence:  
an object detection model (YOLOv8), fine-tuned for Interactable Region Detection (identifying interactable regions from the UI screen).

Apple released Ferret-UI (shown below): a new multimodal large language model (MMLM) tailored for enhanced understanding of mobile UI screens, equipped with referring, grounding, and reasoning capabilities

Reference
[OmniParser github](https://github.com/microsoft/OmniParser)  [OmniParser model](https://huggingface.co/microsoft/OmniParser)
[Ferret-UI demo](https://huggingface.co/spaces/jadechoghari/ferret-demo)   [Ferret-UI model](https://huggingface.co/models?search=ferret-ui)

------------------------------------------------------------------
[5]
2024.10.31
NEO Beta - A Humanoid Robot for the Home
[类似提线木偶或者有轨电车？NEO Beta的技术，或许带来一些人形新思路](https://zhuanlan.zhihu.com/p/721299474)
[Introducing NEO Beta youtube](https://www.youtube.com/watch?v=bUrLuUxv9gE&ab_channel=1X)



------------------------------------------------------------------
[6]
2024.11.01
Compression for AI - Open AI Ilya Sutskever

[An Observation on Generalization youtube](https://www.youtube.com/watch?v=AKMuA_TVz3A&ab_channel=SimonsInstitute)
[lya Sutskever 2023伯克利大学演讲回顾](https://www.youtube.com/watch?v=RH-IdE9udMc&ab_channel=%E6%9C%80%E4%BD%B3%E6%8B%8D%E6%A1%A3)
压缩即智能 （compression for AI）随笔 - 陈雄辉的文章 - 知乎
https://zhuanlan.zhihu.com/p/671472916
直接压缩一切！OpenAI首席科学家Ilya Sutskever这么看无监督学习 - 机器之心的文章 - 知乎
https://zhuanlan.zhihu.com/p/651170707

---
[7]
2024.11.27

@@ [Adobe](https://zhuanlan.zhihu.com/p/4948174414):
[1] Clean Machine：藏在视频里的瑕疵，都能清理是能够自动检测并清除瑕疵镜头的 Clean Machine。由镜头眩光造成的光斑、过曝等，当然可以手动消除，那就要一帧一帧的检查，再慢慢修整。Clean Machine 自动检查整个视频，找出所有的瑕疵帧，然后自动修复，同时填补上原有的元素。在经过修复之后，不仅过曝的画面移除了，也没有损伤整个视频的品质，非常丝滑。这项技术应用场景很多，比如下图这种大面积阴影，是因为路人闯进了镜头造成了遮挡。可以看到模型仍然可以给出一个干净、无遮挡的画面。

[2] HiFi：实时生成，还能实时编辑. HiFi 是一个能够完成实时生成和编辑的功能，不算是很新鲜的想法，外头不少创业公司已经做了出来。 这个特性在 Adobe 的演示中，还是能看到基本逻辑是不变的：抽卡。只不过一致性表现得好，而且用户可以精准控制、调节的颗粒度更细。像上面这个树叶，每一笔添上去的时候，其实绿植的整体都在变化，看得出还是在抽卡。不过轮廓、叶片数量这些，基本能保持一致。这在被抽卡困扰的生成工具中，属实难得了。 HiFi 还计划支持镜头捕捉。在大会上，嘉宾在镜头下用纸样搭配了一下家具，扫描之后，立刻就生成了对应的效果图，而且高度还原。

[3] Perfect Blend：一键 PS 指日可待. 另一个海报神器是 Perfect Blend，可能是这次所有 Sneaks 中最实用的一个，期待它能实装进 Photoshop 里，Adobe 搞快点。Perfect Blend 非常接近「一键 PS」，而且不止是套个美颜滤镜那么简单，是真-修图。抠图啥的不用说，重点是调整光效和色调。经过模型计算，原本光线差异极大的元素，都「融合」在了同一个光线逻辑中。尤其是女孩（她是这个项目的开发者），脸上原本的栅格影子都被完美修掉。光线差异极大的情况，也依然可以掌控。这张人像里，脸部几乎都隐在黑暗中，也能根据新的夕阳背景，覆上新的光泽。哪怕所有的人像是由不同的相机拍摄、在不同的时间、有不同的角度、不同的光线，都不是问题，直接一键生成《冰与火之歌》风格海报。

[4] Turntable：能转的矢量图. Turntable 是最有巧思的一个，对画手、动画人也很有用，实装进 Illustrator 指日可待。这是一个转化矢量图像的功能，把已经整体存在的矢量图层，一键转化成「立体」模型。下图中的骑士，被转化之后，角度就可以自由调整。除了横向旋动也可以纵向，综合起来就能够调节视角维度。实际上 Turntable 中，即便图像能够全方位旋转，依然还是矢量图。模型所做的是补全了原始画面没有涉及到的部位。比如下面这匹马： 在原始画面中，这匹马是正侧面，只能看见两条腿。可一旦要换角度，另外两条腿势必要露出来。而有了 Turntable，不必重画，不必补帧，只需要鼠标拖一拖，就是最符合逻辑的画面。又或者是这个薯片侠，它的背面和侧面都没有画出来。如果想改变角度，需要补足这些没有的部分

**

---

[8]
2024.12.03  <mark style="background: #BBFABBA6;">Motion Prompting: Controlling Video Generation with Motion Trajectories</mark> - DeepMind
[Github](https://motion-prompting.github.io/)

Step 1: Train a Track Conditioned Video Model
Step 2: Prompt the model with Motion Prompts
	1. Object Control物件控制
	2. Emergent Physics新興物理學
	3. Control with Geometry幾何控制
	4. Camera Control相機控制
	5. Object + Camera Control物體+相機控制
	6. Motion Transfer動作轉移


---