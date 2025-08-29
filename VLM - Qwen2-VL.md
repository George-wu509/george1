
Qwen2-VL: Expert Vision Language Model for Video Analysis and Q&A
My Qwen2-VL [Colab](https://colab.research.google.com/drive/1Zahrn91uzsndMvaLefk8xQot4qsAQgIS?usp=sharing#scrollTo=N-kIKVdhxczd)

Original Qwen2-VL [Colab](https://colab.research.google.com/drive/1Zahrn91uzsndMvaLefk8xQot4qsAQgIS?usp=sharing)


![[Pasted image 20250828102001.png]]


### Qwen2-VL整体架构

[视觉语言模型](https://zhida.zhihu.com/search?content_id=250493249&content_type=Article&match_order=1&q=%E8%A7%86%E8%A7%89%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B&zhida_source=entity)（VLM）是人工智能领域的重要突破，它能够同时理解和处理图像与文本信息，实现类似人类的多模态认知能力。这类模型通过将强大的[视觉编码器](https://zhida.zhihu.com/search?content_id=250493249&content_type=Article&match_order=1&q=%E8%A7%86%E8%A7%89%E7%BC%96%E7%A0%81%E5%99%A8&zhida_source=entity)（如[CLIP](https://zhida.zhihu.com/search?content_id=250493249&content_type=Article&match_order=1&q=CLIP&zhida_source=entity)、[ViT](https://zhida.zhihu.com/search?content_id=250493249&content_type=Article&match_order=1&q=ViT&zhida_source=entity)）与大型语言模型（如[GPT](https://zhida.zhihu.com/search?content_id=250493249&content_type=Article&match_order=1&q=GPT&zhida_source=entity)、[LLaMA](https://zhida.zhihu.com/search?content_id=250493249&content_type=Article&match_order=1&q=LLaMA&zhida_source=entity)）相结合，创造出能够进行视觉理解和自然语言交互的智能系统。

典型的VLM通常包含三个核心组件：

- 视觉编码器：将图像转换为特征表示
- 语言模型：处理文本信息并生成响应
- [多模态融合模块](https://zhida.zhihu.com/search?content_id=250493249&content_type=Article&match_order=1&q=%E5%A4%9A%E6%A8%A1%E6%80%81%E8%9E%8D%E5%90%88%E6%A8%A1%E5%9D%97&zhida_source=entity)：实现视觉和语言特征的有效结合

2024年8月开源的Qwen2-VL模型在各种分辨率和长宽比的视觉理解任务中均达到领先水平，在[DocVQA](https://zhida.zhihu.com/search?content_id=250493249&content_type=Article&match_order=1&q=DocVQA&zhida_source=entity)（文档问答）、[InfoVQA](https://zhida.zhihu.com/search?content_id=250493249&content_type=Article&match_order=1&q=InfoVQA&zhida_source=entity)（信息问答）、[RealWorldQA](https://zhida.zhihu.com/search?content_id=250493249&content_type=Article&match_order=1&q=RealWorldQA&zhida_source=entity)（真实场景问答）、[MTVQA](https://zhida.zhihu.com/search?content_id=250493249&content_type=Article&match_order=1&q=MTVQA&zhida_source=entity)（多任务视觉问答）和[MathVista](https://zhida.zhihu.com/search?content_id=250493249&content_type=Article&match_order=1&q=MathVista&zhida_source=entity)（数学视觉理解）等多个基准测试中表现卓越。能够理解超过20分钟的长视频内容，显著提升了视频问答、对话和内容创作等任务的质量。其主要创新点如下：

**动态分辨率处理**

- 引入naive dynamic resolution技术
- 能够灵活处理不同分辨率的输入

**多模态位置编码**

- 创新性提出[多模态旋转位置编码](https://zhida.zhihu.com/search?content_id=250493249&content_type=Article&match_order=1&q=%E5%A4%9A%E6%A8%A1%E6%80%81%E6%97%8B%E8%BD%AC%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81&zhida_source=entity)（M-RoPE）
- 实现更有效的跨模态信息融合

**图像和视频的统一理解框架**

- 图像被处理为两个相同帧，保持与视频处理的一致性
- 使用3D tubes替代2D patches处理方式


|                                                                         |                    |
| ----------------------------------------------------------------------- | ------------------ |
| **1. Installing Qwen2-VL**                                              |                    |
| pip 安裝 github.com/ huggingface/transformers                             |                    |
| pip 安裝qwen-vl-utils                                                     |                    |
|                                                                         |                    |
| **2. Video set up**                                                     |                    |
| conda install -c conda-forge ffmpeg -y                                  |                    |
| pip install ffmpeg -q                                                   |                    |
| pip install gdown -q                                                    | download the video |
| !ffmpeg -i {video_path} -q:v 2 -start_number 0 {output_path}/'%05d.jpg' |                    |
|                                                                         |                    |
| **3. Qwen2-VL for Video Understanding**                                 |                    |
|                                                                         |                    |
|                                                                         |                    |
|                                                                         |                    |



![[Pasted image 20250828102105.png]]

Qwen2-VL模型架构打印下来如下，可以发现它主要由Qwen2VisionTransformerPretrainedModel（视觉编码器）和[Qwen2VLModel](https://zhida.zhihu.com/search?content_id=250493249&content_type=Article&match_order=1&q=Qwen2VLModel&zhida_source=entity)（语言模型）两部分组成，并没有明显的Connector部分，视觉编码向量没有经过太多的处理直接进入了语言模型。精致的Connector似乎正变得没那么重要，早在多模态模型诞生之初就有ViLT这样的将图片直接用线性层投影作为Transformer输入的架构，现在也有一些轻量的多模态模型用MLP对图片进行处理，直接删除了视觉编码器，可见多模态信息的融合fusion不一定需要太复杂的结构。