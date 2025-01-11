

Ref1: [理解SIFT/SURF算法原理，并进行关键点检测](https://blog.51cto.com/u_8238263/6003416)

ref2: [SIFT算法: 尺度不变特征变换](https://zhuanlan.zhihu.com/p/494983479)


### SIFT（尺度不变特征变换）

**SIFT（Scale-Invariant Feature Transform，尺度不变特征变换）** 是一种在计算机视觉中用于检测和描述局部特征的算法。它由 David Lowe 在 1999 年提出，并在 2004 年进一步完善。SIFT 特征是一种对缩放、旋转、光照变化非常鲁棒的特征描述子，广泛应用于图像匹配、物体识别、图像拼接等领域。

#### SIFT 的步骤

SIFT 算法包括以下主要步骤：

1. **尺度空间极值检测**：首先构建图像的尺度空间，通过对图像进行不同尺度的高斯模糊处理，再在每个尺度上计算高斯差分（Difference of Gaussian, DoG）。在不同尺度的图像上，寻找极值点，这些点可能是潜在的关键点。
    
2. **关键点定位**：通过拟合三维二次函数来精确定位关键点的位置。去除边缘响应和低对比度的点，以提高特征的稳健性。
    
3. **方向分配**：对每个关键点的邻域进行梯度计算，确定主要的方向。这样，SIFT 特征具有旋转不变性。
    
4. **特征描述子生成**：在关键点的邻域内，根据梯度方向生成特征向量，通常会将关键点的邻域划分为 4x4 的网格，每个网格计算 8 个方向的梯度直方图，形成长度为 128 维的特征向量。
    
5. **特征匹配**：通过欧氏距离对特征进行匹配，匹配的原则是寻找最小的距离。
    

#### SIFT 的优点

- 对尺度、旋转和光照变化具有很强的鲁棒性。
- 特征描述子具有高维度（128 维），能够描述复杂的局部特征，适合高精度的图像匹配。

#### SIFT 的缺点

- 计算复杂度较高，特别是在高分辨率图像中，计算时间较长。
- 对视角变化和仿射变换不如对尺度和旋转变化鲁棒。

---

### SURF（加速鲁棒特征）

**SURF（Speeded-Up Robust Features，加速鲁棒特征）** 是 SIFT 的改进版本，由 Herbert Bay 等人在 2006 年提出。SURF 的设计目标是在保持鲁棒性的同时加快特征检测和描述的速度。

#### SURF 的步骤

SURF 的主要步骤与 SIFT 类似，但在实现方式上有所不同：

1. **积分图像**：SURF 使用积分图像来加速计算。这使得在不同尺度上对图像进行卷积的计算速度更快。
    
2. **Hessian 矩阵的行列式**：SURF 采用 Hessian 矩阵的行列式来进行关键点检测。这与 SIFT 中使用的高斯差分类似，但计算速度更快。
    
3. **方向分配**：与 SIFT 类似，SURF 也为每个关键点分配主要方向，使其具有旋转不变性。SURF 使用 Haar 小波响应来计算方向。
    
4. **特征描述子生成**：SURF 特征描述子通常是 64 维的向量，比 SIFT 的 128 维更小。通过对关键点邻域内的 Haar 小波响应进行统计，生成描述子。
    
5. **特征匹配**：SURF 的特征匹配与 SIFT 类似，使用欧氏距离或其他度量标准进行匹配。
    

#### SURF 的优点

- 比 SIFT 计算速度更快，适合实时应用。
- 使用积分图像和 Hessian 矩阵行列式，大幅度提高了关键点检测的效率。
- 特征描述子维度较小（64 维），计算开销较低。

#### SURF 的缺点

- 虽然 SURF 在速度上比 SIFT 快，但在一些复杂场景（如大幅度的视角变化或非刚性变形下）鲁棒性可能不如 SIFT。
- 对光照变化的鲁棒性不如 SIFT。

---

### SIFT 与 SURF 的比较

|特性|SIFT|SURF|
|---|---|---|
|**提出年份**|1999 年（David Lowe）|2006 年（Herbert Bay 等）|
|**特征点检测**|高斯差分（DoG）|Hessian 矩阵行列式|
|**特征描述子**|128 维描述子|64 维描述子|
|**计算速度**|较慢|较快|
|**鲁棒性**|对尺度、旋转、光照变化具有高鲁棒性|对尺度、旋转和轻微仿射变换具有较好鲁棒性|
|**实时性**|不适合实时应用|适合实时应用|
|**应用场景**|高精度的图像匹配、物体识别、图像拼接|实时图像处理、快速物体识别|

---

### 英文版本

### SIFT (Scale-Invariant Feature Transform)

**SIFT (Scale-Invariant Feature Transform)** is an algorithm used in computer vision for detecting and describing local features in images. It was introduced by David Lowe in 1999 and refined in 2004. SIFT features are highly robust to changes in scale, rotation, and illumination, making them widely used in applications such as image matching, object recognition, and image stitching.

#### Steps of SIFT

The SIFT algorithm involves the following steps:

1. **Scale-Space Extrema Detection**: The image is processed across different scales by applying Gaussian blurring, and then <mark style="background: #BBFABBA6;">Difference of Gaussians (DoG)</mark> is calculated. Local extrema in different scales are identified as potential keypoints.
    
2. **Keypoint Localization**: The position of each keypoint is refined using a 3D quadratic function. Points with low contrast or poor localization on edges are discarded to improve robustness.
    
3. **Orientation Assignment**: Each keypoint is assigned an orientation based on the gradient of its neighborhood, ensuring rotational invariance.
    
4. **Descriptor Generation**: A feature descriptor is created for each keypoint by dividing the neighborhood into 4x4 grids and computing 8-bin gradient histograms for each, resulting in a 128-dimensional feature vector.
    
5. **Feature Matching**: Features are matched by comparing the Euclidean distance between descriptors.
    

#### Advantages of SIFT

- Highly robust to changes in scale, rotation, and illumination.
- The 128-dimensional feature descriptor can describe complex local features, making it suitable for high-precision image matching.

#### Disadvantages of SIFT

- High computational complexity, making it slower for high-resolution images.
- Less robust to perspective distortions and affine transformations.

---

### SURF (Speeded-Up Robust Features)

**SURF (Speeded-Up Robust Features)** is an improved version of SIFT, introduced by Herbert Bay in 2006. SURF aims to maintain the robustness of SIFT while significantly improving the speed of detection and description.

#### Steps of SURF

SURF follows similar steps as SIFT but with key differences:

1. **Integral Images**: SURF uses integral images to speed up computations, enabling faster convolution across different scales.
    
2. **Hessian Matrix Determinant**: SURF uses the determinant of the <mark style="background: #BBFABBA6;">Hessian matrix</mark> for keypoint detection, providing a faster alternative to SIFT’s Difference of Gaussians.
    
3. **Orientation Assignment**: Like SIFT, SURF assigns an orientation to each keypoint using Haar wavelet responses, ensuring rotational invariance.
    
4. **Descriptor Generation**: SURF generates 64-dimensional descriptors by computing Haar wavelet responses in keypoint neighborhoods.
    
5. **Feature Matching**: Features are matched using the Euclidean distance or other metrics.
    

#### Advantages of SURF

- Faster computation compared to SIFT, making it suitable for real-time applications.
- Efficient keypoint detection due to the use of integral images and Hessian matrix determinant.
- Lower dimensional descriptors (64 dimensions), reducing computational load.

#### Disadvantages of SURF

- While faster, it may not be as robust as SIFT in complex scenarios such as large viewpoint changes or non-rigid deformations.
- Slightly less robust to illumination changes compared to SIFT.

---

### Comparison Between SIFT and SURF

|Feature|SIFT|SURF|
|---|---|---|
|**Year Introduced**|1999 (David Lowe)|2006 (Herbert Bay et al.)|
|**Keypoint Detection**|Difference of Gaussians (DoG)|Hessian Matrix Determinant|
|**Feature Descriptor**|128-dimensional descriptor|64-dimensional descriptor|
|**Speed**|Slower|Faster|
|**Robustness**|Highly robust to scale, rotation, and illumination changes|Robust to scale, rotation, and slight affine transformations|
|**Real-Time Suitability**|Not suitable for real-time applications|Suitable for real-time applications|
|**Applications**|High-precision image matching,||

4o