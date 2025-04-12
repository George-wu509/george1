

|                                 |     |
| ------------------------------- | --- |
| [[###Geometric transformation]] |     |
| [[###2D Affine Transformation]] |     |
| [[###相机的几何参数标定、内参、外参与图像转换的关系]]  |     |
|                                 |     |



### Geometric transformation

|                                 | 剛體變換 <br>Rigid Transformation | 仿射變換<br>Affine Transformation | 投影變換<br>Projective Transformation | 非線性變換<br>Nonlinear Transformation |
| ------------------------------- | ----------------------------- | ----------------------------- | --------------------------------- | --------------------------------- |
| 平移<br>(Translation)             | Yes                           | Yes                           | Yes                               | Yes                               |
| 旋轉<br>(Rotation)                | Yes                           | Yes                           | Yes                               | Yes                               |
| 鏡像/反射<br>(Mirroring/Reflection) | Yes                           | Yes                           | Yes                               | Yes                               |
| 縮放<br>(Scaling)                 |                               | Yes                           | Yes                               | Yes                               |
| 剪切<br>(Shearing)                |                               | Yes                           | Yes                               | Yes                               |
| 非均勻縮放<br>(Non-unform scaling)   |                               | Yes                           | Yes                               | Yes                               |
| 透視縮放<br>(Perspective scaling)   |                               |                               | Yes                               |                                   |
| 非線性扭曲<br>(Nonlinear Warping)    |                               |                               |                                   | Yes                               |

在計算機視覺和圖像處理領域，2D和3D的**剛體變換（Rigid Transformation）**、**仿射變換（Affine Transformation）**、**投影變換（Projective Transformation）** 以及 **非線性變換（Nonlinear Transformation）** 是一系列幾何變換，這些變換可以包括**平移（Translation）**、**旋轉（Rotation）**、**鏡像（Mirroring）**、**縮放（Scaling）**、**剪切（Shearing）**、**非均勻縮放（Non-uniform Scaling）** 和 **反射（Reflection）** 等基本操作。它們之間有著緊密的關聯性和層次關係。以下是這些變換及其基本操作之間的詳細解釋：

### 1. **剛體變換（Rigid Transformation）**

- **定義**: 剛體變換保持物體的形狀和大小不變，僅允許物體在空間中移動或旋轉。變換過程中不會有尺度的變化（即無縮放），並且物體內部的幾何關係完全保持。
- **基本操作**:
    - **平移（Translation）**: 移動物體在空間中的位置。
    - **旋轉（Rotation）**: 物體繞某一軸線旋轉，但不改變其形狀或大小。
    - **鏡像（Mirroring）/ 反射（Reflection）**: 將物體以某一對稱軸為基準翻轉，形狀不變，但位置相對軸線發生變化。
- **應用**: 剛體變換主要應用於物體識別和配準中，因為其保持了物體的幾何特徵。

### 2. **仿射變換（Affine Transformation）**

- **定義**: 仿射變換是一類包含剛體變換和額外變換操作的線性變換。仿射變換能夠保持直線性質和平行性，但角度和距離可能會改變。
- **基本操作**:
    - **平移（Translation）**: 將物體沿著某一方向平移。
    - **旋轉（Rotation）**: 圍繞原點或其他參考點旋轉。
    - **縮放（Scaling）**: 等比或不等比放大/縮小物體的尺寸。等比縮放時，在所有方向上的縮放比例相同；不等比縮放（Non-uniform Scaling）則在不同方向上有不同的縮放比例。
    - **剪切（Shearing）**: 也稱為"斜切"變換，物體在某個方向上移動，形狀發生傾斜，但直線仍保持直線。
- **與剛體變換的區別**: 仿射變換相比剛體變換多了縮放和剪切操作，因此能夠改變物體的形狀或大小。
- **應用**: 仿射變換常用於圖像旋轉、縮放、平移以及仿射變形，廣泛應用於計算機視覺中的影像處理、幾何校正和仿射變換矩陣計算。

### 3. **投影變換（Projective Transformation）**

- **定義**: 投影變換也稱為**透視變換（Perspective Transformation）**，是更廣泛的線性變換。<mark style="background: #ABF7F7A6;">投影變換能夠將3D空間中的點映射到2D平面，並能夠改變物體的平行性</mark>（即投影後平行線不再平行）。它能處理圖像中的透視效果。
- **基本操作**:
    - **平移（Translation）**: 改變物體的位置。
    - **旋轉（Rotation）**: 繞一個參考點旋轉。
    - **縮放（Scaling）**: 放大或縮小物體的尺寸。
    - **剪切（Shearing）**: 改變物體的形狀。
    - <mark style="background: #FFB86CA6;">**透視縮放（Perspective Scaling</mark>）**: 模擬物體隨著距離的增加而變小，產生透視效果。
- **應用**: 投影變換常用於電腦圖形學中的3D場景渲染，將三維空間中的物體透視投影到二維影像中，也可應用於攝影中的視角校正和建模。

### 4. **非線性變換（Nonlinear Transformation）**

- **定義**: 非線性變換是一類不保持直線性的變換，變換後物體的形狀可能會發生非線性的扭曲或彎曲。這類變換無法用簡單的矩陣表示。
- **基本操作**:
    - **非線性扭曲（Nonlinear Warping）**: 將影像或物體進行任意形式的扭曲，這不再是線性映射，可以產生複雜的變形效果。
    - **多次曲線變換（Polynomial Transformations）**: 通常用多項式表示的非線性變換。
- **應用**: 非線性變換常見於醫學影像配準和圖像處理中的變形矯正。其主要目的是處理非剛體對應問題，如人臉變形、地圖變形等。
- ### 總結：

1. **剛體變換** 是最基本的變換，只允許物體的<mark style="background: #BBFABBA6;">平移、旋轉和反射</mark>，保持<mark style="background: #FFB8EBA6;">物體的大小和形狀不變</mark>。
2. **仿射變換** 是剛體變換的擴展，允許<mark style="background: #BBFABBA6;">平移、旋轉、縮放、剪切</mark>，保持<mark style="background: #FFB8EBA6;">平行線不變</mark>。
3. **投影變換** 更加廣泛，允許物體的<mark style="background: #BBFABBA6;">透視變換</mark>，能夠模擬三維空間的透視效果，適用於攝影和三維場景的轉換。
4. **非線性變換** 可以任意改變物體的形狀允許物體的<mark style="background: #BBFABBA6;">非線性扭曲</mark>，廣泛用於非剛體對應場景，如醫學影像中的形變校正和人臉扭曲等應用。


计算机图形学知识点（二）——变换（仿射变换、齐次坐标） - 乱想的文章 - 知乎
https://zhuanlan.zhihu.com/p/681157857

仿射变换（Affine Transformation）在2D和3D坐标下的变换矩阵和性质及齐次坐标系（Homogeneous Coordinates）的应用 - 啊軒Oo的文章 - 知乎
https://zhuanlan.zhihu.com/p/465490024

Affine transformation 是<mark style="background: #BBFABBA6;">线性变换与平移变换</mark>的组合，常用于图像处理、计算机视觉和计算几何。Affine transformation 包含四种基本操作：<mark style="background: #BBFABBA6;">平移 (Translation)，旋转 (Rotation)，缩放 (Scaling)，剪切 (Shearing)</mark>。

在计算机视觉和图形学中，<mark style="background: #FFB86CA6;">齐次坐标系（Homogeneous Coordinates）</mark>常用于表示仿射变换，因为它可以将平移、旋转、缩放、剪切等变换统一为矩阵乘法形式。通过引入一个额外的维度，可以将线性变换和非线性变换（如平移）结合起来。在二维齐次坐标中，每个二维点 $(x, y)$ 可以用一个三维向量 $(x, y, 1)$ 表示。Affine 变换可以通过 $3 \times 3$ 矩阵完成，具体包括平移、旋转、缩放和剪切。






### 2D Affine Transformation

#### 1. 平移 (Translation)

在二维空间中，平移操作只是简单地将点移动某个固定的距离。给定一个点 $(x, y)$ 和一个平移向量 $(t_x, t_y)$，变换后的点 $(x', y')$ 由以下公式表示：


$\large \begin{pmatrix} x' \\ y' \\ 1 \end{pmatrix} = \begin{pmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \\ 0 & 0 & 1 \end{pmatrix} \begin{pmatrix} x \\ y \\ 1 \end{pmatrix}$

#### 2. 旋转 (Rotation)

二维空间中的旋转可以通过旋转矩阵来实现。给定一个角度 $\theta$，旋转后的坐标 $(x', y')$ 可通过下式计算：


$\large \begin{pmatrix} x' \\ y' \\ 1 \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{pmatrix} \begin{pmatrix} x \\ y \\ 1 \end{pmatrix}$


#### 3. 缩放 (Scaling)

缩放是指沿着坐标轴按比例放大或缩小点的位置。给定缩放因子 $s_x$ 和 $s_y$，缩放后的坐标 $(x', y')$ 计算公式如下：


$\large \begin{pmatrix} x' \\ y' \\ 1 \end{pmatrix} = \begin{pmatrix} s_x & 0 & 0 \\ 0 & s_y & 0 \\ 0 & 0 & 1 \end{pmatrix} \begin{pmatrix} x \\ y \\ 1 \end{pmatrix}​$

#### 4. 剪切 (Shearing)

剪切是指在一个方向上拉伸或压缩物体，同时保持另一个方向不变。二维空间中的剪切变换矩阵为：

$\large \begin{pmatrix} x' \\ y' \\ 1 \end{pmatrix} = \begin{pmatrix} 1 & sh_x & 0 \\ sh_y & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix} \begin{pmatrix} x \\ y \\ 1 \end{pmatrix}​$

### 3D Affine Transformation 使用齐次坐标表示

在三维齐次坐标中，每个三维点 $(x, y, z)$ 用四维向量 $(x, y, z, 1)$ 表示，Affine 变换通过 $4 \times 4$ 矩阵来完成。

#### 1. 平移 (Translation)

在齐次坐标中，三维平移矩阵表示为：

$\large \begin{pmatrix} x' \\ y' \\ z' \\ 1 \end{pmatrix} = \begin{pmatrix} 1 & 0 & 0 & t_x \\ 0 & 1 & 0 & t_y \\ 0 & 0 & 1 & t_z \\ 0 & 0 & 0 & 1 \end{pmatrix} \begin{pmatrix} x \\ y \\ z \\ 1 \end{pmatrix}​$

#### 2. 旋转 (Rotation)

三维旋转在齐次坐标中，可以分别绕 $x$ 轴、$y$ 轴和 $z$ 轴旋转。

- 绕 $x$ 轴旋转的齐次坐标矩阵：

$\large \begin{pmatrix} x' \\ y' \\ z' \\ 1 \end{pmatrix} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & \cos\theta & -\sin\theta & 0 \\ 0 & \sin\theta & \cos\theta & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix} \begin{pmatrix} x \\ y \\ z \\ 1 \end{pmatrix}​$

- 绕 $y$ 轴旋转的齐次坐标矩阵：

$\large  \begin{pmatrix} x' \\ y' \\ z' \\ 1 \end{pmatrix} = \begin{pmatrix} \cos\theta & 0 & \sin\theta & 0 \\ 0 & 1 & 0 & 0 \\ -\sin\theta & 0 & \cos\theta & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix} \begin{pmatrix} x \\ y \\ z \\ 1 \end{pmatrix}​$

- 绕 $z$ 轴旋转的齐次坐标矩阵：

$\large \begin{pmatrix} x' \\ y' \\ z' \\ 1 \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta & 0 & 0 \\ \sin\theta & \cos\theta & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix} \begin{pmatrix} x \\ y \\ z \\ 1 \end{pmatrix}$


#### 3. 缩放 (Scaling)

三维缩放在齐次坐标中，表示为：

$\large \begin{pmatrix} x' \\ y' \\ z' \\ 1 \end{pmatrix} = \begin{pmatrix} s_x & 0 & 0 & 0 \\ 0 & s_y & 0 & 0 \\ 0 & 0 & s_z & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix} \begin{pmatrix} x \\ y \\ z \\ 1 \end{pmatrix}$


#### 4. 剪切 (Shearing)

三维剪切在齐次坐标中的矩阵形式为：

$\large \begin{pmatrix} x' \\ y' \\ z' \\ 1 \end{pmatrix} = \begin{pmatrix} 1 & sh_{xy} & sh_{xz} & 0 \\ sh_{yx} & 1 & sh_{yz} & 0 \\ sh_{zx} & sh_{zy} & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix} \begin{pmatrix} x \\ y \\ z \\ 1 \end{pmatrix}​$



在图像处理、计算机视觉和计算几何中，除了 Affine transformation（仿射变换）之外，还有多种变换类型可以应用于 2D 和 3D 图像。这些变换主要包括：

### 其他常见的图像转换类型

#### 1. **投影变换 (Projective Transformation)**

投影变换是一种更一般的变换，包括了仿射变换作为其子集。它允许平行线不再保持平行，并且可以模拟透视效果。投影变换是通过 3x3 矩阵在齐次坐标中进行的，在 2D 中称为 **透视变换 (Perspective Transformation)**。

**公式：**

$\begin{pmatrix} x' \\ y' \\ w \end{pmatrix} = \begin{pmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{pmatrix} \begin{pmatrix} x \\ y \\ 1 \end{pmatrix}$​

最终的转换结果需要通过齐次坐标的归一化得到 $(x'/w, y'/w)$。

#### 2. **非线性变换 (Non-linear Transformation)**

这些变换包括诸如辐射畸变校正、双曲线变换、球面变换等，它们用于处理诸如鱼眼镜头失真、全景图像处理等问题。这类变换无法用简单的线性矩阵描述，通常需要特定的畸变模型。

#### 3. **薄板样条变换 (Thin Plate Spline, TPS)**

薄板样条变换是一种用于图像形变的非线性变换，用于模拟图像的柔性变形（例如模拟材料的形变）。它被广泛用于医学图像处理。

#### 4. **刚体变换 (Rigid Transformation)**

刚体变换保持物体的形状和大小不变，只涉及旋转和平移。在 3D 图像处理中，刚体变换经常用于对物体进行旋转和平移的操作。

#### 5. **相似变换 (Similarity Transformation)**

相似变换包括旋转、平移和统一缩放。它保持了物体的形状，但允许缩放和旋转。





### 相机的几何参数标定、内参、外参与图像转换的关系

在计算机视觉中，相机标定是用于获取相机的几何参数以描述如何从 3D 世界坐标系投影到 2D 图像平面的过程。这个过程与多种变换紧密相关。


#### (1) Perspective Projection Formula

$\Huge \begin{bmatrix} x \\ y \\ w \end{bmatrix} = \begin{bmatrix} f & 0 & p_{x} & 0 \\ 0 & f & p_{y} & 0 \\ 0 & 0 & 1 & 0\end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix} = \begin{bmatrix} Xf \\ Yf \\ Z \end{bmatrix} ​$ 

Image            Projection Matrix      World
Coordinates                                    Coordinates


#### (2) Decomposing the Perspective Projection

$\Huge  \begin{bmatrix} f & 0 & p_{x} & 0 \\ 0 & f & p_{y} & 0 \\ 0 & 0 & 1 & 0\end{bmatrix} = \begin{bmatrix} f & 0 & p_{x} \\ 0 & f & p_{y} \\ 0 & 0 & 1\end{bmatrix} \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0\end{bmatrix} ​$ 


$\large K = \begin{bmatrix} f & 0 & p_{x} \\ 0 & f & p_{y} \\ 0 & 0 & 1\end{bmatrix}$    K:  相機內參矩陣Intrinsic Matrix 




#### **1. 相机的内参 (Intrinsic Parameters)**

相机的内参描述了相机的内部特性，这些参数将相机坐标系中的 3D 点投影到图像平面上，通常包括焦距、主点（光心的偏移量）以及相机的像素尺度。内参可以表示为一个 3x3 的矩阵：

**相机内参矩阵：**

$\large K = \begin{pmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{pmatrix}$​​

其中 $f_x$ 和 $f_y$ 是相机的焦距，$c_x$ 和 $c_y$ 是主点的坐标。

**作用：**

- 内参负责将 3D 空间中的点投影到 2D 图像平面上，同时影响图像的缩放、透视效果等。透视变换与相机的内参关系密切，通常用于将真实场景与图像之间的投影关系建立起来。

#### **2. 相机的外参 (Extrinsic Parameters)**

外参描述了相机相对于世界坐标系的位姿（位置和方向），包括一个 3x3 的旋转矩阵 $R$ 和一个 3x1 的平移向量 $T$，即表示相机在 3D 世界坐标系中的姿态。

**外参矩阵：**

$\large \begin{pmatrix} R & T \\ 0 & 1 \end{pmatrix}$

其中 $R$ 表示旋转矩阵，$T$ 表示平移向量。

**作用：**

- 外参将 3D 世界坐标中的点转换到相机坐标系中，然后再通过内参矩阵将相机坐标系中的点投影到 2D 图像平面。这一过程可以描述为：世界坐标系 -> 相机坐标系 -> 图像坐标系。

#### **3. 世界模型与相机标定的关系**

世界模型通常是指定义物体在 3D 空间中的位置和形状的模型。在计算机视觉中，世界模型的坐标通过相机的外参变换到相机坐标系中，再通过内参投影到 2D 图像上。

**几何投影模型：** 对于一个 3D 点 $(X_w, Y_w, Z_w)$，它首先通过外参变换到相机坐标系中的点 $(X_c, Y_c, Z_c)$：

$\large \begin{pmatrix} X_c \\ Y_c \\ Z_c \end{pmatrix} = R \begin{pmatrix} X_w \\ Y_w \\ Z_w \end{pmatrix} + T$

然后通过相机内参将 3D 点投影到 2D 图像平面：

$\large \begin{pmatrix} u \\ v \\ 1 \end{pmatrix} = K \begin{pmatrix} X_c / Z_c \\ Y_c / Z_c \\ 1 \end{pmatrix}​$

其中 $(u, v)$ 为图像平面的坐标。

#### **4. 图像转换与相机标定的关系**

图像转换（如旋转、缩放、透视）与相机的几何变换密切相关。具体来说，图像的旋转、缩放可以用相机的内参矩阵表示，而图像的平移和相机在 3D 世界中的位姿则通过外参来描述。比如：

- **投影变换** 可以用来模拟相机的透视投影，它与相机内参直接相关。
- **刚体变换** 描述了物体在 3D 空间中的旋转和平移，这些变换对应着相机外参在标定过程中的计算。
- **非线性变换**（例如畸变校正）是实际相机中由于镜头造成的变形，需要通过特定的相机模型来校正，这个过程属于相机标定的一部分。

总之，图像转换的数学表达与相机标定过程中的内参和外参矩阵密切相关，它们共同决定了如何将 3D 世界中的物体映射到 2D 图像平面上，以及如何从 2D 图像中恢复 3D 信息。

ref: 
[[图形学渲染]大白话推导三维重建-内参(Intrinsic)、外参(extrinsic)、相机坐标转换、3D物体投影归一化、单双目摄像头、视差(Disparity)](https://zhuanlan.zhihu.com/p/681902159)


