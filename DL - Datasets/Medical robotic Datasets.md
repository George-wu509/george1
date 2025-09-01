
以下是我为你精心整理的 **2D / 3D 医疗机器人（medical robotic）相关 AI 模型训练所需的主流开放资料集（open datasets）**，包含任务内容、应用场景、特点、规模、下载方式与主要的标注形式，全部以中文详解，助你快速掌握资源与落地使用方案。

---

## 一、2D 医疗机器人相关数据集

### 1. **JIGSAWS（JHU-ISI Gesture and Skill Assessment Working Set）**

- **任务内容**：用于评估与分析机器人辅助手术中的手势识别、技术水平评估等；数据同步包含视频帧＋机器人运动（kinematics）。
- **应用**：训练模型识别特定手术手势动作、技能级别评估、阶段分割等。
- **特点**：运营者技能不同的真实执行数据，视频与 kinematics 数据同步，可用于机器人技能学习、示范模仿学习。
- **规模**：101 次试验录像，涵盖 “suturing（缝合）”、“needle passing（穿针）”、“knot tying（打结）” 三类基础任务，由 6 位外科医生执行[PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5559351/?utm_source=chatgpt.com)[Johns Hopkins University+1](https://pure.johnshopkins.edu/en/publications/a-dataset-and-benchmarks-for-segmentation-and-recognition-of-gest?utm_source=chatgpt.com)[arXiv](https://arxiv.org/pdf/2102.03643?utm_source=chatgpt.com)。
- **下载方式**：需前往 Johns Hopkins University 官方页面或 JIGSAWS 官网，通常要求填写简单使用目的后下载。
- **标注形式**：
    - 视频：RGB 视频帧；
    - 运动数据：76 维度的机器人关节 kinematics；
    - 手势动作阶段标记、时间戳标签。

---

### 2. **SurgPose**

- **任务内容**：用于手术器械的姿态估计与跟踪，提供关键点（keypoints）和骨架（skeleton）标注。
- **应用**：训练视觉模型识别手术工具的姿态，支持增强现实（AR）辅助培训或自主操作研究。
- **特点**：使用紫外 UV 液体荧光标记关键点，不影响可见光，提供精准视觉关键点标注。
- **规模**：约 120,000 个器械实例（80k 用于训练，40k 用于验证），6 类器械，每个实例标注 7 个语义关键点[Johns Hopkins University+2PMC+2](https://pure.johnshopkins.edu/en/publications/a-dataset-and-benchmarks-for-segmentation-and-recognition-of-gest?utm_source=chatgpt.com)[arXiv+1](https://arxiv.org/html/2502.11534v1?utm_source=chatgpt.com)。
- **下载方式**：参考 SurgPose 的 arXiv 论文或对应 GitHub 仓库（如 authors 提供链接），部分提供公开下载。
- **标注形式**：
    - 每一帧或每个实例：7 个语义关键点坐标（如尖端、关节等）；
    - 类别标签 + 实例编号。

---

### 3. **EgoSurgery-HTS**

- **任务内容**：用于开腹手术中“第一人称”（egocentric）视频的手部与器械分割。
- **应用**：训练模型在手持器械操作场景准确分割出“手”和“工具”，适用于实时辅助、行为理解等。
- **特点**：开腹手术视角罕见，提供像素级高精度分割，标注颗粒度精细。
- **规模**：目前尚无具体规模数据，但涵盖 14 类工具与手部实例，多帧像素标注[ResearchGate+1](https://www.researchgate.net/publication/389091434_SurgPose_a_Dataset_for_Articulated_Robotic_Surgical_Tool_Pose_Estimation_and_Tracking?utm_source=chatgpt.com)[arXiv](https://arxiv.org/abs/2503.18755?utm_source=chatgpt.com)。
- **下载方式**：论文中提供 GitHub 链接（如 [https://github.com/Fujiry0/EgoSurgery），可按说明获取。](https://github.com/Fujiry0/EgoSurgery%EF%BC%89%EF%BC%8C%E5%8F%AF%E6%8C%89%E8%AF%B4%E6%98%8E%E8%8E%B7%E5%8F%96%E3%80%82)
- **标注形式**：
    - 像素级 mask：工具实例 mask、手部实例 mask；
    - “手工具交互”标注：标注哪些手操作哪些工具。

---

### 4. **CholecInstanceSeg**

- **任务内容**：腹腔镜（laparoscopic）手术工具实例分割。
- **应用**：精确分割每个手术工具实例，支持更高级的场景分析、行为识别。
- **特点**：目前最大规模公开工具实例分割集，全部来自真实临床数据。
- **规模**：41,900 帧，来自 85 个真实手术，包含 64,400 个工具实例，带 semantic mask 和 instance ID[arXiv](https://arxiv.org/abs/2503.18755?utm_source=chatgpt.com)[arXiv](https://arxiv.org/abs/2406.16039?utm_source=chatgpt.com)。
- **下载方式**：访问 arXiv 文章，通常提供数据链接或联系作者申请。
- **标注形式**：
    - 像素级 segmentation masks；
    - 实例 ID + 语义标签。

---

### 5. **SurgT**（Soft-Tissue Tracking Challenge）

- **任务内容**：软组织 tracking（跟踪）benchmark，用于评估软组织在手术视频中的动态跟踪能力。
- **应用**：训练模型实时跟踪器械与组织之间的相对位置变化，用于导航系统或机器人操作校正。
- **特点**：提供带有校正参数的立体视频，可衡量追踪鲁棒性。
- **规模**：157 段立体（stereo）视频以及相机校准参数[OpenCas+15科學直通車+15arXiv+15](https://www.sciencedirect.com/science/article/pii/S1361841523002451?utm_source=chatgpt.com)[ResearchGate](https://www.researchgate.net/publication/389091434_SurgPose_a_Dataset_for_Articulated_Robotic_Surgical_Tool_Pose_Estimation_and_Tracking?utm_source=chatgpt.com)[arXiv+1](https://arxiv.org/abs/2110.12555?utm_source=chatgpt.com)。
- **下载方式**：参访问 EndoVis Challenge 或 SurgT 官方站点，填写表单后获取下载权限。
- **标注形式**：
    - Stereo 视频帧；
    - 相机标定（intrinsics/extrinsics）；
    - 通常包括 ground truth tissue tracking 信息（mask 或 point correspondences）。

---

### 6. **SurgBench（SurgBench-P 和 SurgBench-E）**

- **任务内容**：大规模手术视频基准框架，提供预训练集（SurgBench‑P）与评估集（SurgBench‑E）。
- **应用**：训练基础 surgical foundation models，包含 laparoscopic、内镜、机器人、开放手术。
- **特点**：覆盖四种主要手术模式，规模庞大，统一提供训练与测试指标[OpenCas+1](https://opencas.dkfz.de/endovis/datasetspublications/?utm_source=chatgpt.com)[arXiv](https://arxiv.org/html/2506.07603v1?utm_source=chatgpt.com)。
- **规模**：SurgBench‑P 包括 53M 帧，采自 16 个不同来源。
- **下载方式**：查找 SurgBench 论文或对应作者，通常提供链接获取大规模视频数据。
- **标注形式**：
    - 视频帧，可能包括手术阶段标签、动作标签等。

---

## 二、3D 医疗机器人相关数据集

### 7. **SERV-CT / StereoMIS (基于 da Vinci 机器人)**

- **任务内容**：为立体内窥镜图像提供 CT‑based 解剖分割与 occlusion maps。
- **应用**：训练模型在立体视觉中恢复深度、组织结构以及遮挡关系。
- **特点**：融合立体图像与 CT 解剖学分割，高精度；但样本不多，适用于验证与微调。
- **规模**：SERV‑CT 包含 16 对立体图像；StereoMIS 涵盖 3 份猪体、3 个人体录制。[arXiv](https://arxiv.org/html/2506.07603v1?utm_source=chatgpt.com)[Kinova](https://www.kinovarobotics.com/uploads/Robotic-Arm-Platform-for-Multi-View-Image-Acquisition-and-3D-Reconstruction-in-Minimally-Invasive-Surgery.pdf?utm_source=chatgpt.com)
- **下载方式**：需查找相关论文或资源，一般通过联系作者或 Robot dataset 平台获取。
- **标注形式**：
    - 立体图像对；
    - CT segmentation maps；
    - 遮挡标注（occlusion maps）。

---

### 8. **SARAMIS**

- **任务内容**：提供合成人体解剖部位的高质量 3D 网格、体素（tetrahedral volumes）、纹理等，用于 MIS 与 RAMIS 仿真。
- **应用**：生成视觉模拟数据，例如用于训练视觉感知模型、引导机器人操作或导航。
- **特点**：基于开源 CT 数据制作，包含 104 个解剖靶点，规模大、内容丰富。
- **规模**：104 个解剖目标的 3D rendering assets。
- **下载方式**：开源托管在 GitHub（[https://github.com/NMontanaBrown/saramis/）](https://github.com/NMontanaBrown/saramis/%EF%BC%89)[NeurIPS](https://neurips.cc/virtual/2023/poster/73590?utm_source=chatgpt.com)。
- **标注形式**：
    - 3D meshes（网格）、纹理贴图；
    - 可配合合成渲染生成的数据。

---

### 9. **Visible Human Project**

- **任务内容**：医学影像指导下的完整人体 3D 再现数据集，包括男性与女性的解剖切层图。
- **应用**：可用于机器人针插导航、解剖学识别、模拟训练等。
- **特点**：高分辨切片图，可免费下载（需同意 NLM 条款），并具有标准解剖结构。
- **规模**：完整人体系列切片数据，容量大；男性与女性版本。
- **下载方式**：国家图书馆（National Library of Medicine）网站在线下载，可免费获取[arXiv+6Nature+6en.wikipedia.org+6](https://www.nature.com/articles/s41597-025-05093-7?utm_source=chatgpt.com)[en.wikipedia.org+2github.com+2](https://en.wikipedia.org/wiki/Visible_Human_Project?utm_source=chatgpt.com)。
- **标注形式**：
    - 静态切片图（解剖层面）；
    - 可配合结构化标签或 segmentation maps 使用。

---

## 三、 汇总表格

|数据集名称|类型|任务内容|应用场景|特点|规模|下载方式|标注形式|
|---|---|---|---|---|---|---|---|
|JIGSAWS|2D video + kinematics|手势识别、技能评估|训练手术动作分类 / 机器人技能学习|视频＋运动数据同步|101 trials|官方页面|视频帧 + kinematics + 时序标签|
|SurgPose|2D image|工具姿态估计|AR 培训 / 姿态跟踪|UV 关键点，精确|120k instances|arXiv/GitHub|7 keypoints per instance|
|EgoSurgery-HTS|2D pixel|手/工具分割|第一人称行为理解|手 + 工具 mask|多帧数据|GitHub|像素级 mask，实例标注|
|CholecInstanceSeg|2D pixel|工具实例分割|工具识别 & 后续分析|最大规模实例分割|41.9k 帧|arXiv link|semantic + instance masks|
|SurgT|2D stereo video|软组织 tracking|实时导航|立体视 + 校准|157 段视频|Challenge 网站|视频 + calibration + tracking|
|SurgBench|2D video|基础视频预训练 & 评估|基础模型训练平台|覆盖多种手术模态|53M 帧|论文 / 作者|视频帧 + 可能的标签|
|SERV-CT / StereoMIS|3D stereo|解剖 segmentation|立体视觉 + 导航|CT + stereo 高精度|少量图像|论文 / 联系作者|Stereo pairs + CT maps|
|SARAMIS|3D synthetic|解剖仿真资产|模拟训练 / 合成数据|104 部位 mesh+textures|104 targets|GitHub|3D meshes, textures|
|Visible Human|3D slices|医学解剖图|解剖导航 / 模拟|完整人体|全部切片|