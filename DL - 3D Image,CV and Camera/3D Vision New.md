
這部分是你原本文件最值得增加技術深度的地方之一。

|建議新增|代表技術或模型|原因|
|---|---|---|
|Neural 3D Reconstruction|NeRF、3D Gaussian Splatting|從多視角影像重建及渲染場景|
|Learned Geometry|DUSt3R、MASt3R|Learning-based 3D Reconstruction 和 Matching|
|Geometry Foundation Models|VGGT、VGGT-Omega|從多視角影像直接推論深度、相機與幾何|
|Foundation Depth Estimation|Depth Anything V2、Depth Anything 3|從專門的深度模型走向通用幾何模型|
|6D Object Pose Estimation|FoundationPose|Robotics、物件抓取、工業定位|
|3D Foundation Reconstruction|SAM 3D|單張影像推論物體幾何與外觀|
|Dynamic 3D / 4D Reconstruction|4D Gaussian Splatting 等|動態場景、時間一致性|
|Neural / Learned SLAM|Learned Features、Visual SLAM|Localization、Mapping、Pose Optimization|

#### 特別重要的三個發展

![Introduction to Gaussian Splatting | Can Koz](https://images.openai.com/static-rsc-4/kuZORKHoHkS7Y7wLfP-trBF9r0ExNfeViDnL4tArpdiWRfAZHQQXzRNNnKbOz6zz1iQcVzBKTfgHyChzGaaYmJVE3haGAgehaEU_AkfApVXiSUy6bG7fn4KmV6-FtfkZVZ8pIoPJ14f42MPOc0pq-yeKCCr0fKCS0omh6N28CG4?purpose=inline)

A. NeRF / 3D Gaussian Splatting

從多視角影像學習場景表示並合成新視角。3DGS 透過可微分的 3D Gaussian primitives 表示場景，通常能提供高效率的渲染。這裡要學的是 explicit / implicit representations、camera poses、photometric optimization 和 reconstruction quality。

![CVPR 2025 Best Paper: Visual Geometry Grounded Transformer (VGGT) ❤️ 
VGGT shows that multi-view 3D reconstruction can be handled by a single feed-forward transformer, without relying on heavy… | Kavishka Abeywardana](https://images.openai.com/static-rsc-4/CvftF1mcQhg3CMZ8ctUnfovIb3Tx29RUoquA3Ra8HHkoNC4O79SPJO8wM4IBtTsnXEej2YmDyQMVGM3_RRXNCLEqnLGqWpa6hnJ-kyf8CAZpuVg7XFtbSuRz8Pb6BirHpBLZeaHJ3gLMPdOvG2KRJRbn0u5buqt8dfVkiWhwDTE?purpose=inline)

B. VGGT / Geometry Foundation Models

VGGT 在 CVPR 2025 發表，能從一張或多張影像推論相機參數、深度、point maps 和 3D point tracks。2026 年又有 VGGT-Omega 後續工作。


重要改變是：部分傳統上需要 feature matching、epipolar geometry、triangulation、bundle adjustment 的流程，現在可以利用 pretrained model 提供初始幾何估計，再依任務進行 refinement。

![Jianyuan (@jianyuan_wang) on X](https://images.openai.com/static-rsc-4/IDdgTjWXKDEvqIW8O_ehj0c_okKaKCk6jUmlHATIQPw-_6WnvDhVw5tsy49tPfokvOMyn_QXtT82HMPZE2RzUxWJM2BB-5rDPEwE04rxYVC6W9dC9MvfUh00xVw5WGfPH3tVE5PUPFGQwqlSkQlePiktumRW76pZhuXWgKXG4Es?purpose=inline)

C. Depth Anything 3

2025 年底發布，從單純 depth prediction 進一步處理多視角的 spatially consistent geometry。值得理解它和傳統 monocular depth、stereo depth、SfM 之間的差異。


#### 為什麼這個方向重要？

傳統 3D Vision 比較依賴明確幾何假設：

```
Feature Detection
    ↓
Feature Matching
    ↓
Fundamental / Essential Matrix
    ↓
Camera Pose Estimation
    ↓
Triangulation
    ↓
Bundle Adjustment
    ↓
3D Reconstruction
```

現代方法可以加入 Learned Geometry：

```
Multi-view Images
    ↓
Pretrained Geometry Foundation Model
    ↓
Depth / Point Maps / Camera Pose
    ↓
Geometric Verification and Refinement
    ↓
3D Reconstruction / SLAM / Measurements
```

但這不代表幾何學不重要。

如果你在 Senior CV 面試時被問到：

> A foundation model reconstructs a 3D object, but the dimensions are physically incorrect. How would you fix it?

你需要提到 monocular scale ambiguity、metric depth、camera calibration、coordinate systems、reprojection error、bundle adjustment，以及用已知尺寸或深度感測器來建立實際物理尺度。

這也是為什麼你原本列出的 Fourier Transform、Subpixel Registration、Optics、Camera Calibration 不應被移除：它們仍是建立可量測、可驗證工業影像系統的核心。