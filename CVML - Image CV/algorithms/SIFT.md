

Ref1: [理解SIFT/SURF算法原理，并进行关键点检测](https://blog.51cto.com/u_8238263/6003416)

ref2: [SIFT算法: 尺度不变特征变换](https://zhuanlan.zhihu.com/p/494983479)


### SIFT（尺度不变特征变换）

**SIFT（Scale-Invariant Feature Transform，尺度不变特征变换）** 是一种在计算机视觉中用于检测和描述局部特征的算法。它由 David Lowe 在 1999 年提出，并在 2004 年进一步完善。SIFT 特征是一种对缩放、旋转、光照变化非常鲁棒的特征描述子，广泛应用于图像匹配、物体识别、图像拼接等领域。

#### SIFT 的步骤
SIFT 算法包括以下主要步骤：

1. **尺度空间极值检测**：首先构建图像的尺度空间，通过对图像进行不同尺度的高斯模糊处理，再在每个尺度上计算高斯差分（Difference of Gaussian, DoG）。在不同尺度的图像上，寻找极值点，这些点可能是潜在的关键点。
2. **关键点定位** 通过拟合三维二次函数来精确定位关键点的位置。去除边缘响应和低对比度的点，以提高特征的稳健性。
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
2. **Hessian 矩阵的行列式**：SURF 采用 ==Hessian 矩阵的行列式==来进行关键点检测。这与 SIFT 中使用的高斯差分类似，但计算速度更快。
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




```
能否具體舉個簡單的例子譬如5x5 matrix 來每一步驟解釋SIFT跟SURF的計算過程並中文詳細解釋
```

這是一個非常好的請求，但也是一個巨大的挑戰。**SIFT** 和 **SURF** 都是相當複雜的多階段演算法。

**首先，我必須說明一個現實：** 一個 5x5 的矩陣（影像）實在是**太小了**，小到無法真正展示 SIFT 或 SURF 的**核心精神**，特別是「尺度不變性」(Scale Invariance)。例如：

1. **尺度空間：** SIFT 需要建立高斯金字塔（Octaves），每一層都比前一層模糊且更小。一個 5x5 的影像降採樣一次就變成 2x2 了，無法進行有意義的比較。
    
2. **特徵描述子：** SIFT 的描述子 (Descriptor) 需要在關鍵點周圍取一個 16x16 的鄰域。這在 5x5 的影像上顯然是不可能的。
    

**因此，我們換一個方式：**

我將**高層次地**解釋 SIFT 和 SURF 的每一步。然後，對於演算法中的**特定計算步驟**（例如計算梯度、計算積分圖），我會使用一個 5x5 的矩陣作為範例來展示**該步驟**是如何運作的。

這能讓您在理解整體流程的同時，也能看到具體的數學計算。

---

### 範例 5x5 影像矩陣 (Image, $I$)

假設我們有以下 5x5 的灰階影像（像素值 0-255）：

$$I = \begin{bmatrix} 10 & 20 & 30 & 40 & 50 \\ 20 & 30 & 40 & 50 & 60 \\ 30 & 40 & 90 & 60 & 70 \\ 40 & 50 & 60 & 70 & 80 \\ 50 & 60 & 70 & 80 & 90 \end{bmatrix}$$

注意：中心點 $I(2, 2) = 90$ 是一個亮點（索引從 0 開始）。我們將會重點關注這個區域。

---

### Part 1: SIFT (Scale-Invariant Feature Transform)

SIFT 的目標是找到在尺度和旋轉下都保持穩定的特徵點。

#### 步驟 1：建立尺度空間 (Scale-Space Construction)

- **目的：** 讓影像在不同「尺度」（模糊程度）下都能被檢測。
    
- **作法：**
    
    1. **高斯模糊 (Gaussian Blur)：** 對原始影像 $I$ 進行多次高斯模糊，每次使用的 $\sigma$ 值（標準差，代表模糊程度）都比前一次大。例如：
        
        - $L_1 = G(\sigma_1) * I$
            
        - $L_2 = G(\sigma_2) * I$
            
        - ...
            
    2. **高斯差分 (Difference of Gaussians, DoG)：** 將相鄰的模糊影像相減，得到 DoG 影像。
        
        - $DoG_1 = L_2 - L_1$
            
        - $DoG_2 = L_3 - L_2$
            
        - ...
            
    3. **金字塔 (Pyramid)：** 將影像降採樣（縮小一半），重複步驟 1 和 2，建立好幾個「層」(Octaves)。
        
- **為何這麼做？** DoG 是對「拉普拉斯算子」(Laplacian of Gaussian, LoG) 的一個高效近似。LoG 對「斑點」(Blobs) 非常敏感。DoG 影像中值最大或最小的點，就是潛在的斑點（關鍵點）。
    
- **5x5 範例 (概念)：** 由於 5x5 太小，我們無法進行有意義的模糊和相減。想像一下，我們用 $\sigma_1$ 和 $\sigma_2$ 模糊了 $I$，中心點 $I(2, 2)$ 附近的值會被「平滑」，然後 $DoG(2, 2) = L_2(2, 2) - L_1(2, 2)$。
    

#### 步驟 2：關鍵點定位 (Keypoint Localization)

- **目的：** 在 DoG 影像中精確找到「極值點」（最大值或最小值）。
    
- **作法：**
    
    1. **3D 鄰域比較：** 檢查 DoG 影像中的每一個像素，看它是否比它**周圍的 26 個鄰居**都大或都小。
        
        - 這 26 個鄰居是：當前尺度的 8 個鄰居 + 上一個尺度的 9 個鄰居 + 下一個尺度的 9 個鄰居。
            
    2. **精確定位：** 透過泰勒展開式擬合一個 3D 二次函數，找到真正的極值點，精確到次像素 (sub-pixel) 位置和尺度。
        
    3. **去除低對比度點：** 如果極值點的 DoG 值太低（例如 $|DoG| < 0.03$），說明它對比度太低，丟棄。
        
    4. **去除邊緣響應：** **(這裡用到了 Hessian 矩陣!)** 邊緣在 DoG 中也會產生強烈響應，但它不是穩定的「點」特徵。
        
        - 計算該點在 DoG 影像上的 $2 \times 2$ Hessian 矩陣 $H$。
            
        - Hessian 的特徵值 $\lambda_1, \lambda_2$ 描述了主曲率。
            
        - 如果一個特徵值遠大於另一個（例如 $\frac{|\lambda_1|}{|\lambda_2|} > 10$），表示這是一個「脊線」（邊緣），而不是「山峰」（角點）。丟棄這種點。
            

#### 步驟 3：方向指定 (Orientation Assignment)

- **目的：** 賦予每個關鍵點一個「主方向」，以實現旋轉不變性。
    
- **作法：**
    
    1. 在關鍵點所在的尺度 $L$（注意：不是 DoG 影像）上，取一個鄰域（例如 16x16）。
        
    2. 計算鄰域中每個像素的**梯度大小 $m(x, y)$** 和**梯度方向 $\theta(x, y)$**。
        
    3. 建立一個 36 區間 (bin) 的方向直方圖（每 10 度一個區間）。
        
    4. 將每個像素的梯度大小 $m$ 加權（通常用高斯加權，離中心越近權重越大）後，丟入其對應的方向 $\theta$ 區間。
        
    5. 直方圖中最高的那個峰值，就是該關鍵點的「主方向」。
        
- **5x5 範例 (計算梯度)：** 讓我們計算中心點 $I(2, 2)$ 的梯度。我們使用簡單的一階差分：
    
    - $G_x = I(2, 3) - I(2, 1) = 60 - 40 = 20$
        
    - $G_y = I(3, 2) - I(1, 2) = 60 - 40 = 20$
        
    - **梯度大小** $m = \sqrt{G_x^2 + G_y^2} = \sqrt{20^2 + 20^2} \approx 28.3$
        
    - **梯度方向** $\theta = \text{atan2}(G_y, G_x) = \text{atan2}(20, 20) = 45^\circ$
        
    - 這個 $45^\circ$ 會被丟入 $40^\circ\text{-}49^\circ$ 的區間，並貢獻 28.3 的權重。
        

#### 步驟 4：關鍵點描述 (Keypoint Descriptor)

- **目的：** 為關鍵點建立一個獨特且穩健的「指紋」（128 維向量）。
    
- **作法：**
    
    1. 在關鍵點周圍取一個 16x16 的鄰域。
        
    2. 將這個 16x16 鄰域**旋轉**到步驟 3 中算出的「主方向」，以確保旋轉不變性。
        
    3. 將 16x16 鄰域切成 4x4 共 16 個子區域（每個子區域 4x4 像素）。
        
    4. 在**每一個**子區域中，計算一個 8 區間 (bin) 的梯度方向直方圖（類似步驟 3，但只有 8 個方向）。
        
    5. 將這 16 個子區域的 8 區間直方圖全部串接起來。
        
    6. **結果：** $16 \text{ (子區域)} \times 8 \text{ (方向)} = 128$ 維的特徵向量。
        
    7. 最後對向量進行歸一化 (Normalize)，以消除光照變化的影響。
        

---

### Part 2: SURF (Speeded Up Robust Features)

SURF 的目標與 SIFT 相同，但它在每一步都追求**極致的速度**。

#### 步驟 1：積分圖 (Integral Image)

- **目的：** 這是 SURF 加速的**核心祕密**。它允許在 $O(1)$（常數時間）內計算出**任何矩形區域**的像素總和。
    
- **作法：** 積分圖 $II(x, y)$ 的值是原始影像 $I$ 中，從左上角 $(0, 0)$ 到 $(x, y)$ 的所有像素總和。
    
    - $II(x, y) = \sum_{i \le x, j \le y} I(i, j)$
        
- **5x5 範例 (計算積分圖)：**
    
    - $I = \begin{bmatrix} 10 & 20 & 30 & 40 & 50 \\ 20 & 30 & 40 & 50 & 60 \\ 30 & 40 & 90 & 60 & 70 \\ 40 & 50 & 60 & 70 & 80 \\ 50 & 60 & 70 & 80 & 90 \end{bmatrix}$
        
    - $II(0, 0) = I(0, 0) = 10$
        
    - $II(0, 1) = I(0, 0) + I(0, 1) = 10 + 20 = 30$
        
    - $II(1, 0) = I(0, 0) + I(1, 0) = 10 + 20 = 30$
        
    - $II(1, 1) = I(0,0) + I(0,1) + I(1,0) + I(1,1) = 10+20+20+30 = 80$
        
    - ... 經過計算 ...
        
    - $II = \begin{bmatrix} 10 & 30 & 60 & 100 & 150 \\ 30 & 80 & 150 & 240 & 350 \\ 60 & 150 & 310 & 460 & 640 \\ 100 & 240 & 460 & 680 & 940 \\ 150 & 350 & 650 & 950 & 1290 \end{bmatrix}$
        
- **如何使用？** 假設要計算 $I$ 中間 $3 \times 3$ 區域（以 90 為中心）的總和。
    
    - 該區域的四個角是 (1,1), (1,3), (3,1), (3,3)。
        
    - 對應的 $II$ 點是 A=(0,0), B=(0,3), C=(3,0), D=(3,3)。
        
    - **不對，這裡容易搞錯。** 應該是：
        
    - 區域的四個角 (x,y) 是 (1,1), (3,1), (1,3), (3,3)。
        
    - 對應的積分圖計算點是 A=(0,0), B=(0,3), C=(3,0), D=(3,3)。
        
    - **Sum = $II(3,3) - II(0,3) - II(3,0) + II(0,0)$**
        
    - Sum = $680 - 100 - 100 + 10 = 490$。 (手動驗算: $30+40+50+40+90+60+50+60+70 = 490$。正確！)
        

#### 步驟 2：關鍵點偵測 (Hessian Matrix)

- **目的：** 找到斑點（關鍵點）。
    
- **作法：**
    
    1. SURF **不使用 DoG**，而是使用 **Hessian 矩陣的行列式 (DoH)**。如前所述，DoH 對斑點（兩個特徵值都很大）的響應很強。
        
    2. **加速的祕密：** SURF **不使用高斯濾波器**來計算二階導數 ($I_{xx}, I_{yy}, I_{xy}$)，而是使用**方框濾波器 (Box Filters)**。
        
    3. 這些方框濾波器（由黑色和白色矩形組成）的響應可以**利用積分圖（步驟 1）極快地計算出來**，無論濾波器多大！
        
    4. 計算近似的 DoH：$\det(H_{approx}) = D_{xx}D_{yy} - (0.9 \cdot D_{xy})^2$。
        

#### 步驟 3：建立尺度空間 (Scale Space)

- **目的：** 實現尺度不變性。
    
- **作法：**
    
    - SIFT 的作法：縮小影像，濾波器大小不變。
        
    - SURF 的作法：**影像大小不變**，而是**放大方框濾波器的尺寸**（例如 9x9, 15x15, 21x21...）。
        
    - **為何能加速？** 因為有了積分圖，濾波器從 9x9 放大到 21x21，計算成本**完全相同**！這比 SIFT 不斷降採樣和模糊快得多。
        
- **定位：** 與 SIFT 類似，在 $3 \times 3 \times 3$（x, y, 尺度）鄰域中尋找 DoH 響應的極值點。
    

#### 步驟 4：方向指定 (Orientation Assignment)

- **目的：** 實現旋轉不變性。
    
- **作法：**
    
    1. SURF **不使用梯度直方圖**，而是使用 **Haar 小波 (Haar Wavelet) 響應**。
        
    2. 在關鍵點周圍的一個圓形區域內，計算 60 度扇形內的 $dx$ 和 $dy$ 的 Haar 響應總和。
        
    3. 滑動這個 60 度的扇形，找到響應最強的方向，作為主方向。
        
    4. 這一步也利用積分圖來快速計算 Haar 響應。
        

#### 步驟 5：關鍵點描述 (Keypoint Descriptor)

- **目的：** 建立「指紋」。
    
- **作法：**
    
    1. 在關鍵點周圍取一個 20s x 20s 的鄰域（s 是該點的尺度）。
        
    2. 將鄰域旋轉到主方向。
        
    3. 切成 4x4 的子區域。
        
    4. 在**每一個**子區域中，計算 4 個值：
        
        - $\sum dx$ (x 方向 Haar 響應和)
            
        - $\sum |dx|$ (x 方向 Haar 響應絕對值和)
            
        - $\sum dy$ (y 方向 Haar 響應和)
            
        - $\sum |dy|$ (y 方向 Haar 響應絕對值和)
            
    5. **結果：** $16 \text{ (子區域)} \times 4 \text{ (值)} = 64$ 維的特徵向量。
        
    6. 這比 SIFT 的 128 維更小、更快，且同樣具有區分度。
        

---

### 總結 SIFT vs SURF

|**步驟**|**SIFT (Scale-Invariant Feature Transform)**|**SURF (Speeded Up Robust Features)**|**哪個更快？**|
|---|---|---|---|
|**關鍵點偵測**|高斯差分 (DoG)|**Hessian 行列式 (DoH)**|SURF|
|**加速工具**|無（高斯模糊很慢）|**積分圖 (Integral Image)**|SURF|
|**計算基礎**|高斯濾波器|**方框濾波器 (Box Filters)**|SURF|
|**尺度空間**|縮小影像，濾波器不變|影像不變，**放大濾波器**|SURF|
|**方向指定**|梯度直方圖 (36-bin)|**Haar 小波** 扇形響應|SURF|
|**描述子**|4x4 區域，8-bin 梯度直方圖|4x4 區域，4 個 Haar 響應值|SURF|
|**向量維度**|**128 維**|**64 維**|SURF|

**結論：** SIFT 是開創性的，非常穩健。SURF 透過積分圖和方框濾波器（來近似 Hessian 和 Haar 小波）的巧妙組合，在幾乎所有步驟上都實現了驚人的加速，同時保持了與 SIFT 相當（甚至更好）的穩健性。