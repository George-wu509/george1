
超强实时跟踪系统首次开源！支持跨镜头、多类别、小目标跟踪！ - 百度大脑的文章 - 知乎
https://zhuanlan.zhihu.com/p/439125383
github: https://github.com/PaddlePaddle/paddledetection

在琳琅满目的视觉应用中，对车辆、行人、飞行器等快速移动的物体进行实时跟踪及分析，可以说是突破安防、自动驾驶、智慧城市等炙手可热行业的利器。

但要实现又快又准的持续跟踪，往往面临被检目标多、相互遮挡、图像扭曲变形、背景杂乱、视角差异大、目标小且运动速度快等产业难题。

那如何快速获得这个能力呢？今天给大家介绍的不仅仅是单独的智能视觉算法，而是一整套多功能多场景的跟踪系统——[PP-Tracking](https://zhida.zhihu.com/search?content_id=185713067&content_type=Article&match_order=1&q=PP-Tracking&zhida_source=entity)。‍

它融合了目标检测、行人重识别、轨迹融合等核心能力，并针对性地优化和解决上述实际业务的痛点难点，提供行人车辆跟踪、跨镜头跟踪、多类别跟踪、小目标跟踪及流量计数等能力与产业应用，还支持可视化界面开发，让你快速上手、迅速落地。

![[Pasted image 20250326222902.png]]

想了解这套超强目标跟踪系统的详细结构、优势亮点及使用方法？下面带大家来快速领略下。

功能丰富效果佳

PP-Tracking 内置 [DeepSORT](https://zhida.zhihu.com/search?content_id=185713067&content_type=Article&match_order=1&q=DeepSORT&zhida_source=entity)[6]、[JDE](https://zhida.zhihu.com/search?content_id=185713067&content_type=Article&match_order=1&q=JDE&zhida_source=entity)[7]与 [FairMOT](https://zhida.zhihu.com/search?content_id=185713067&content_type=Article&match_order=1&q=FairMOT&zhida_source=entity)[8]三种主流高精度多目标跟踪模型，并针对产业痛点、结合实际落地场景进行一系列拓展和优化，覆盖多类别跟踪、跨镜跟踪、流量统计等功能与应用，可谓是精度、性能、功能丰富样样俱全。

>> 单镜头跟踪

单镜头下的单类别目标跟踪是指在单个镜头下，对于同一种类别的多个目标进行连续跟踪，是跟踪任务的基础。针对该任务，PP-Tracking 基于端到端的 One Shot 高精模型 FairMOT[8]，替换为更轻量的骨干网络 HRNetV2-W18，采用多种 Tricks，如 Sync_BN与EMA，保持性能的同时大幅提高了精度，并且扩大训练数据集，减小输入尺寸，最终实现服务端轻量化模型在权威数据集 [MOT17](https://zhida.zhihu.com/search?content_id=185713067&content_type=Article&match_order=1&q=MOT17&zhida_source=entity)上精度达到 MOTA 65.3，在 [NVIDIA Jetson NX](https://zhida.zhihu.com/search?content_id=185713067&content_type=Article&match_order=1&q=NVIDIA+Jetson+NX&zhida_source=entity) 上速度达到23.3FPS，GPU 上速度可达到60FPS！同时，针对对精度要求较高的场景，PP-Tracking 还提供了精度高达 MOTA75.3的高精版跟踪模型。

>> 多类别跟踪

PP-Tracking 不仅高性能地实现了单镜头下的单类别目标跟踪，更针对多种不同类别的目标跟踪场景，增强了特征匹配模块以适配不同类别的跟踪任务，实现跟踪类别覆盖人、自行车、小轿车、卡车、公交、三轮车等上十种目标，精准实现多种不同种类物体的同时跟踪。

>> 跨镜头跟踪

安防场景常常会涉及在多个镜头下对于目标物体的持续跟踪。当目标从一个镜头切换到另一个镜头，往往会出现目标跟丢的情况，这时，一个效果好速度快的跨镜头跟踪算法就必不可少了！PP-Tracking 中提供的跨镜头跟踪能力基于 DeepSORT[6]算法，采用了百度自研的轻量级模型 [PP-PicoDet](https://zhida.zhihu.com/search?content_id=185713067&content_type=Article&match_order=1&q=PP-PicoDet&zhida_source=entity) 和 [PP-LCNet](https://zhida.zhihu.com/search?content_id=185713067&content_type=Article&match_order=1&q=PP-LCNet&zhida_source=entity) 分别作为检测模型和 ReID 模型，配合轨迹融合算法，保持高性能的同时也兼顾了高准确度，实现在多个镜头下紧跟目标，无论镜头如何切换、场景如何变换，也能准确跟踪目标的效果。

>> 流量监测

与此同时，针对智慧城市中的高频场景—人/车流量监测，PP-Tracking 也提供了完整的解决方案，应用服务器端轻量级版 FairMOT[8]模型预测得到目标轨迹与 ID 信息，实现动态人流/车流的实时去重计数，并支持自定义流量统计时间间隔。

为了满足不同业务场景下的需求，如商场进出口人流监测、高速路口车流量监测等，PP-Tracking 更是提供了出入口两侧流量统计方式。

复杂场景覆盖全

>> 行人、车辆跟踪

智慧交通中，行人和车辆的场景尤为广泛，因此 PP-Tracking 针对行人和车辆，提供对应的预训练模型，大幅降低开发成本，节省训练时间和数据成本，实现业务场景直接推理，算法即应用的效果！不仅如此，PP-Tracking 支持显示目标轨迹，更直观地辅助实现高效的路径规划分析。

>> 人头跟踪

不仅如此，除了在日常跟踪任务中拥有极强的通用性，针对实际业务中常常出现目标遮挡严重等问题，PP-Tracking也进行了一系列优化，提供了基于 FairMOT[8]训练的人头跟踪模型，并在 Head Tracking 2021数据集榜单位居榜首，助力 PP-Tracking 灵活适配各类行人场景。

>> 小目标跟踪

针对小目标出现在大尺幅图像中的产业常见难题场景，PP-Tracking 进行了一系列的优化，提供专门针对小目标跟踪的预训练模型，实现在特殊场景，如无人机等航拍场景下，也能达到较为精准的效果。

两种使用模式 训练推理灵活掌握

为了满足不同的开发需求，PP-Tracking 支持两种使用方式，无论是想通过代码调用/训练模型，进行快速推理部署，还是想要零代码直接上手使用功能，PP-Tracking 通通满足你！

>> API 代码调用：

API 简洁易用，支持模型调用、训练与推理部署，最大程度降低开发成本的前提下，灵活适配各类场景与任务。

更贴心的是，PP-Tracking 支持 Python、C++两种部署语言，同时提供使用飞桨原生推理库 Paddle Inference 和飞桨服务化推理框架 Paddle Serving 的保姆级部署教程，真正意义上打通从训练、推理到部署的全流程。

产业场景快速融合

这么厉害的实时跟踪系统在实际落地中的表现如何呢？接下来，让我们看看 PP-Tracking 的实际业务落地效果吧。

以人流量计数为例，在上海音智达公司的实际业务中，使用 PP-Tracking 中的服务端轻量化版 FairMOT[8]，结合人流量计数功能，快速实现商圈出入口的实时人流量去重计数。

