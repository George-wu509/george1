#### Title:
Minimal Multi-Camera 3D Reconstruction and Segmentation

---

##### **Resume Keyworks:
<mark style="background: #FF5582A6;"> SfM(Feature detection Stereo matching, Camera pose, sparse point cloud)</mark>, <mark style="background: #FF5582A6;">MVS</mark>, <mark style="background: #FF5582A6;">Surface reconstruction</mark>, <mark style="background: #ADCCFFA6;">3D segmentation</mark>, <mark style="background: #ADCCFFA6;">Minimal selection</mark>, <mark style="background: #FF5582A6;">3D gaussian splatting</mark>**

##### **KEYWORDS:**  
1. ==**Structure from Motion (SfM)==: feature detection and Stereo matching特徵檢測與立體匹配, camera pose相機姿態估算, sparse point cloud稀疏點雲生成**

2. ==**Multi-View Stereo (MVS)==: dense point cloud 高密度點雲**
3. **Surface reconstruction 表面重建**
4. **3D segmentation 3D 分割**
5. **minimal image selection 基於特徵的最小影像選擇**

##### **STEPS:**
step0. Camera calibration
1. Data有20張multi-view camera images, 

step1. <mark style="background: #FF5582A6;">Structure from motion(SfM)</mark> -> 3D sparse point cloud
1. (**detect_features**)載入所有的images, 用opencv的<mark style="background: #BBFABBA6;">SIFT</mark>方法計算每張image上的所有keypoints跟descriptors.

2. (**match_features**)使用opencv的<mark style="background: #BBFABBA6;">FLANN匹配器</mark>的<mark style="background: #BBFABBA6;">KnnMatch</mark>從第1,2張, 第2,3張開始兩兩匹配descriptors, 譬如第1張圖的第1個keypoint的descriptor去找第2張圖最匹配的前2個descriptors, 如果Euler距離最短的比第二短x0.75更短, 則選入這兩張圖匹配的good matches.

3. (**estimate_pose**)用兩張圖匹配的good matches(src_pts第1張圖, dst_pts第2張圖的pixel coordinates各N點匹配點座標), 加上兩個camera的K(intrinsic matrix)用opencv的<mark style="background: #BBFABBA6;">RANSAC</mark>方法計算兩個camera的<mark style="background: #BBFABBA6;">Essential Matrix(E)</mark>. 接著在反推兩個camera的rotation matrix跟transition matrix.

4. (**sparse_points**)用兩個camera相對的rotation matrix跟transition matrix, 再加上各自的K(intrinsic matrix)可以求得兩個camera相對各自的projection matrix. 使用opencv的cv2.<mark style="background: #BBFABBA6;">triangulatePoints function</mark>可以用src_pts第1張圖, dst_pts第2張圖的pixel coordinates各N點匹配點座標跟兩個projection matrix求得3D sparse cloud points. 輸出是齊次座標(Homogeneous Coordinate)
5. (去重)由第1,2張, 第2,3張....計算出來的3D sparse cloud points可能有重複的, 設立3D distance threshold將太接近的sparse points視為重複的並刪除
6. (Minimal images)(optional)設定一個最小Minimal images N. 計算兩兩image pair求得的3D sparse cloud points並計算和其他image pair的重複3D sparse cloud points的個數並排序, 將重複數最高的依序刪除到剩下Minimal images N為止

![[mvs.jpg]]
step2. <mark style="background: #FF5582A6;">Multi-view stereo(MVS)</mark> -> 用openMVS的<mark style="background: #FF5582A6;">DensifyPointCloud function</mark>輸出dense point cloud
1. 將之前的camera pose, sparse 3D point cloud按照openmvs的格式儲存成xxxx.mvs跟資料夾images放置之前的所有images. <mark style="background: #ADCCFFA6;">DensifyPointCloud</mark>的輸入包括sparse 3D point cloud, 原始images, 以及內參K(intrinsic matrix)跟外參matrix. 輸出是dense 3D point cloud. 演算法是<mark style="background: #ADCCFFA6;">Patch-Match Algorithm</mark>
2. (DensifyPointCloud第一步- <mark style="background: #BBFABBA6;">立體聲對選擇Stereo pair selection</mark>)基于图像视角之间的几何关系（如相机位置和视角重叠）选择最佳的图像对, 譬如20 个相机中，假设图像 1 和图像 5 之间视角差异合适，则选择它们作为一个立体对
3. (DensifyPointCloud第二步<mark style="background: #BBFABBA6;"> 深度圖計算Depth-map computation</mark>) 为每个立体图像对计算参考图像的深度图, 为参考图像中的每个像素随机初始化深度 d 和法向量n, 计算当前深度值的匹配代价 CCC，选择代价最小的深度值。参考图像为 img1，源图像为 img5. 假设像素 (100,200)(100, 200)(100,200) 的初始深度为 5 米，经过迭代后更新为 5.3 米. 输出：每张参考图像生成的深度图 D(u,v)D(u, v)D(u,v)。
4. (DensifyPointCloud第三步<mark style="background: #BBFABBA6;">深度圖細化Depth-map refinement</mark>) 目的: 消除深度图中的噪声和错误估计，提升深度图的精度。原理: 在物体边界处进行平滑，避免出现不连续或尖锐的深度变化。利用多视图一致性检查如果重投影误差超过阈值，则认为深度值无效。使用深度的邻域信息调整法向量 nnn，提高局部表面的一致性。
5. (DensifyPointCloud第四步<mark style="background: #BBFABBA6;">深度圖合併Stereo pair selection</mark>) 将所有参考图像的深度图合并，生成场景的稠密点云。点云生成：将每张深度图中的像素点投影到三维空间，生成 3D 点. 点云合并：将所有深度图的点云合并，去除重复点和低质量点。应用点云去噪算法（例如基于密度的滤波），提升点云质量。
6. Summary

| **步骤**                    | **输入**        | **输出**       | **说明**                    |
| ------------------------- | ------------- | ------------ | ------------------------- |
| **Stereo Pair Selection** | 图像、相机姿态       | 图像对列表        | 选择最佳图像对以优化匹配计算效率和准确性。     |
| **Depth-Map Computation** | 立体图像对、稀疏点云    | 每张参考图像的初始深度图 | 使用 PatchMatch 算法快速估计像素深度。 |
| **Depth-Map Refinement**  | 初始深度图、多视图几何关系 | 优化后的深度图      | 应用多视图验证和法向量调整，提升深度图的精度。   |
| **Depth-Map Merging**     | 优化深度图、相机内参    | 稠密点云文件       | 合并深度图生成稠密点云，并去除重复和低质量点。   |
|                           |               |              |                           |

step3. <mark style="background: #ADCCFFA6;">Surface reconstruction</mark> -> 将稠密点云转换为连续的三角网格模型
1. (点云预处理) 输入稠密点云文件（通常为 `.ply` 格式), 稠密点云包含的每个点的信息：(x,y,z,r,g,b)(x, y, z, r, g, b)(x,y,z,r,g,b)，即 3D 坐标和颜色。基于点密度阈值，去除点云中的孤立点或低密度区域
2. (法向量估算) 为点云的每个点估算法向量，确定点所在表面的方向。为每个点搜索其邻域点（例如 10 个最近邻点）。使用最小二乘法拟合点的局部平面。
3. (表面重建) 基于点云和法向量生成三角形网格，描述稠密点云的几何结构. Poisson 表面重建, 通过点云的法向量场，拟合生成闭合的三维表面。对生成的表面进行 Delaunay 三角剖分，分割成多个三角形。
4. (网格优化) 提高网格的质量，通过优化去除冗余三角形和平滑表面。使用 Edge Collapse（边折叠算法）合并冗余三角形，减少模型的复杂度。使用拉普拉斯平滑算法（Laplacian Smoothing）去除网格中的噪声

step4. <mark style="background: #ADCCFFA6;">3D segmentation</mark>
1. 將3D dense point cloud轉換成3D體積數據(Volumetric Data). 將3D空間劃分為每個邊長為1的立方體(體素voxel),將點雲的 (x, y, z) 坐標映射到網格中. 如果某個體素中有點，則設置值為 `1`，否則為 0
2. 使用<mark style="background: #BBFABBA6;">3D U-Net</mark>做3D segmentation.  3D U-Net的輸入是體積數據(Volumetric Data), 形狀為 `(C, D, H, W). `C`：通道數, `D`：深度（3D影像的切片數), `H`：高度, `W`：寬度. example: 單模態MRI影像：`(1, 64, 128, 128). 
3. 3D U-Net跟2D U-Net的差別. 使用Conv3D, MaxPool3D, ReLU, Softmax or Sigmoid, ConvTranspose3D. Loss function:  二分類：Binary Cross-Entropy。, 多分類：Categorical Cross-Entropy,醫學影像中常用 Dice Loss 或 IoU Loss。

step5. <mark style="background: #FF5582A6;">3D gaussian splatting</mark>



---
#### Resume: 
Designed a multi-view dense 3D reconstruction and segmentation pipeline using SfM and MVS, incorporating feature matching, dense point cloud and surface reconstruction, 3D segmentation, and feature-based minimal image selection. Enhanced efficiency and quality with evaluation metrics for geometric accuracy, model completeness, and visual fidelity.
使用 SfM 和 MVS 設計了多視圖密集 3D 重建和分割管道，結合了特徵匹配、密集點雲和表面重建、3D 分割以及基於特徵的最小影像選擇。透過幾何精度、模型完整性和視覺保真度的評估指標提高效率和品質。

---
#### Abstract: 
This project presents an optimized pipeline for **3D reconstruction and segmentation** using minimal images from multi-camera datasets. Leveraging **Structure from Motion (SfM)** and **Multi-View Stereo (MVS)** techniques, the framework combines efficiency and accuracy by selecting a subset of images based on feature point distribution while maintaining high-quality reconstruction results.
該專案提供了一個使用來自多相機資料集的最少影像進行 **3D 重建和分割** 的最佳化管道。利用**運動結構(SfM)** 和**多視圖立體(MVS)** 技術，該框架透過根據特徵點分佈選擇影像子集，同時保持高品質的重建結果，從而將效率和準確性結合起來。

Key features include:

- **Full and Minimal Image Reconstruction:** Perform 3D reconstruction using all images or a minimal subset, selected through heuristic algorithms to cover all feature points with fewer images. 
- **Sparse-to-Dense Workflow:** Generate sparse point clouds via SfM, refine into dense point clouds with MVS, and produce surface meshes for 3D visualization.
- **Evaluation Metrics:** Quantify reconstruction results with **Geometric Accuracy** (Chamfer Distance), **Model Completeness** (coverage ratio), and **Visual Fidelity** (SSIM-like comparison).
- **3D Segmentation:** Perform AI-based segmentation on reconstructed 3D models, enabling semantic understanding of the reconstructed objects or scenes.
- - **完整和最小影像重建：** 使用所有影像或透過啟發式演算法選擇的最小子集執行 3D 重建，以更少的影像覆蓋所有特徵點。
- **稀疏到密集工作流程：** 透過 SfM 產生稀疏點雲，使用 MVS 細化為密集點雲，並產生用於 3D 視覺化的表面網格。
- **評估指標：**透過**幾何精確度**（倒角距離）、**模型完整性**（覆蓋率）和**視覺保真度**（類似SSIM的比較）來量化重建結果。
- **3D 分割：** 對重建的 3D 模型執行基於 AI 的分割，從而實現對重建物件或場景的語義理解。

---
#### Technique detail:
本項目旨在實現一個基於多攝像機影像的三維重建和分割系統，通過以下技術實現：

1. **基於全部影像的完整三維重建：** 使用多視角影像進行高精度的三維建模，生成密集點雲和網格模型。
2. **最少影像子集選擇：** 基於特徵點分布和覆蓋矩陣，實現最小覆蓋問題的解決，選擇最少的影像組合。
3. **影像子集的高效重建：** 使用選擇的子集進行三維重建，並對比與全影像重建的質量差異。
4. **三維分割與語義分析：** 對重建的三維模型進行語義分割，為應用場景提供豐富的物體語義信息。
5. **量化評估指標：** 提供幾何準確性（Geometric Accuracy）、模型完整性（Model Completeness）和視覺一致性（Visual Fidelity）等指標，量化比較重建結果。


Ref: 3D gaussian splatting official [github](https://github.com/graphdeco-inria/gaussian-splatting)
Ref: OpenMVS officialt [github](https://github.com/cdcseacave/openMVS?tab=readme-ov-file)
Ref: awesome_3DReconstruction_list [github](https://github.com/openMVG/awesome_3DReconstruction_list)
Ref: 速通3D Gaussian ：[击败NERF的三维重建](https://zhuanlan.zhihu.com/p/713908565)
Ref: [OpenMVS的初探](https://zhuanlan.zhihu.com/p/566828254)

Ref: 计算机视觉（cv）笔记——第二&六&七讲[Camera Calibration and Epipolar Geometry](https://zhuanlan.zhihu.com/p/7497049628)
Ref: 计算机视觉（cv）笔记——第十讲[Multi-View Stereo（MVS）](https://zhuanlan.zhihu.com/p/9627998239)
Ref: 《GAMES203：三维重建和理解》5 多视图立体视觉[（Multi-View Stereo，MVS](https://zhuanlan.zhihu.com/p/602861595)）
Ref: Multi-View Stereo中的[平面扫描(plane sweep)](https://zhuanlan.zhihu.com/p/363830541)





---

### **技術細節**

#### **1. Structure from Motion (SfM)**

SfM 是一種通過多視角影像恢復三維結構的方法。該系統通過以下步驟完成 SfM 流程：

- **特徵檢測與匹配：** 使用 SIFT 算法提取影像的關鍵點和描述符，並進行影像對之間的匹配。
- **相機姿態估算：** 利用本質矩陣（Essential Matrix）和 RANSAC 方法計算相機外參（旋轉矩陣和平移向量）。
- **稀疏點雲生成：** 通過三角化方法將特徵點轉換為三維坐標，生成初步的稀疏點雲。

#### **2. Multi-View Stereo (MVS)**

MVS 是基於多視角影像生成密集點雲的技術。本系統將 SfM 生成的稀疏點雲作為初始輸入，通過插值和點雲濾波生成高密度的點雲表示，提供三維模型的詳細幾何結構。

#### **3. 特徵點分布與最少影像選擇**

該系統實現了基於特徵點分布的影像選擇：

- **覆蓋矩陣構建：** 建立特徵點和影像的關聯矩陣，標識每張影像覆蓋的特徵點集合。
- **最小覆蓋問題：** 通過線性分配算法解決最小覆蓋問題，選擇最少的影像組合以覆蓋所有特徵點。

#### **4. 三維模型分割**

基於密集點雲的語義分割功能，模擬了簡單的分割方案，並支持替換為深度學習模型進行高級分割。這使得該系統不僅能生成三維模型，還能賦予其語義信息。

#### **5. 量化指標評估**

系統提供了以下指標，用於比較全影像重建和子集影像重建的質量：

- **Geometric Accuracy (Chamfer Distance):** 衡量重建模型與基準模型之間的幾何差異。
- **Model Completeness (Coverage):** 計算子集影像重建對全影像重建的覆蓋率。
- **Visual Fidelity:** 通過點雲數量比例衡量模型的視覺一致性。

---

### **流程細節**

#### **1. 數據準備**

- 將多視角影像存放於指定目錄（`image_dir`）。
- 使用 `load_images()` 加載影像並進行初步處理。

#### **2. 全影像三維重建**

1. 使用 `detect_features()` 提取影像特徵點和描述符。
2. 使用 `match_features()` 匹配影像間的特徵點對。
3. 使用 `estimate_pose()` 計算相機姿態，並生成稀疏點雲。
4. 使用 `dense_reconstruction()` 生成密集點雲。
5. 使用 `surface_reconstruction()` 生成表面網格模型。

#### **3. 最少影像子集選擇與重建**

1. 使用 `select_minimal_image_subset()` 基於特徵點分布選擇最少影像子集。
2. 對選擇的子集影像進行 `detect_features()`、`match_features()` 和 `estimate_pose()`。
3. 使用子集影像完成稀疏點雲、密集點雲和表面模型的重建。

#### **4. 三維分割**

- 使用 `ai_3d_segmentation()` 對密集點雲進行語義分割，生成分割結果文件。

#### **5. 質量評估**

- 使用 `compute_metrics()` 比較全影像重建與子集影像重建結果，生成以下評估指標：
    - 幾何準確性（Chamfer 距離）
    - 模型完整性（Coverage）
    - 視覺一致性（Visual Fidelity）

#### **6. 結果輸出**

- 所有重建結果（點雲、網格模型和分割結果）保存在指定的輸出目錄。
- 評估指標以 JSON 格式輸出。

---

```python

# Load images
img = cv2.imread(img_path)   # img = ndarray(). shape=(3024,4032,3)

# [[ 1. feature detection ]]
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(img, None)  
# keypoints = list(tuple(),tuple()..), len=300. type=cv2.KeyPoint contain: 
#    pt=(x,y), size, class_id, angle, octave, response
# descriptors = ndarray(), shape = (300,128)

```

#### 1. 初始化 SIFT 演算法

`sift = cv2.SIFT_create()`

- **說明**：
    - 使用 OpenCV 的 `cv2.SIFT_create()` 方法來初始化 ==SIFT（Scale-Invariant Feature Transform）==演算法的物件。
    - SIFT 是一種用於檢測和描述影像中局部特徵的演算法，具有尺度不變性和旋轉不變性，常用於影像匹配、物體識別等應用。
- **輸出類型**：
    - `sift` 是一個 SIFT 特徵檢測器的物件，屬於 `cv2.SIFT` 類型。

#### 2. 遍歷影像並檢測特徵
```python
for img in self.images:
    keypoints, descriptors = sift.detectAndCompute(img, None)
    self.keypoints_list.append(keypoints)
    self.descriptors_list.append(descriptors)
```
- **說明**：
    - `self.images` 是一個包含多張影像的列表，每張影像通常是以 NumPy 陣列的形式儲存（通常是三維陣列，形狀為 `(高度, 寬度, 通道數)`）。
    - 使用 `for` 迴圈逐一處理每張影像 `img`。
##### a. 檢測關鍵點和計算描述子
`keypoints, descriptors = sift.detectAndCompute(img, None)`
- **功能**：
    - `sift.detectAndCompute(img, None)` 同時完成關鍵點的檢測和描述子的計算。
    - 第一個參數 `img` 是待處理的影像，第二個參數 `None` 表示不使用掩膜（mask）。
- **輸出**：
    - **`keypoints`**：
        - **類型**：`list`，每個元素是 `cv2.KeyPoint` 物件。
        - **說明**：`keypoints` 列表包含了影像中所有檢測到的關鍵點（特徵點）。每個 `cv2.KeyPoint` 物件包含了關鍵點的位置、尺度、方向等資訊。
        - **具體範例**：
            `[KeyPoint(x=100.0, y=150.0, _size=1.6, _angle=-1, ...),  KeyPoint(x=200.0, y=250.0, _size=2.0, _angle=45, ...),  ...]`
            
    - **`descriptors`**：
        - **類型**：`numpy.ndarray`，形狀為 `(關鍵點數量, 128)`，數據類型通常為 `float32`。
        - **說明**：`descriptors` 是描述子矩陣，每一行對應一個關鍵點的描述子。SIFT 描述子是 128 維的向量，用於表示關鍵點周圍的局部圖像特徵。
        - **具體範例**：
            `array([[0.2, -0.1, ..., 0.5],        [0.3,  0.0, ..., 0.6],        ...], dtype=float32)`         
            假設有 2 個關鍵點，則 `descriptors` 的形狀為 `(2, 128)`。
##### b. 將檢測結果添加到列表中

`self.keypoints_list.append(keypoints) self.descriptors_list.append(descriptors)`
- **說明**：
    - `self.keypoints_list` 和 `self.descriptors_list` 是用來存儲所有影像的關鍵點和描述子的列表。
    - 將每張影像的 `keypoints` 和 `descriptors` 分別添加到對應的列表中。
- **結果**：
    - **`self.keypoints_list`**：
        - **類型**：`list`，每個元素是 `list` 類型，內部包含多個 `cv2.KeyPoint` 物件。
        - **結構**：
            `[     [KeyPoint1_img1, KeyPoint2_img1, ...],  # 第一張影像的關鍵點     [KeyPoint1_img2, KeyPoint2_img2, ...],  # 第二張影像的關鍵點     ... ]`
            
    - **`self.descriptors_list`**：
        - **類型**：`list`，每個元素是 `numpy.ndarray` 類型，形狀為 `(關鍵點數量, 128)`。
        - **結構**：
            `[     array([[...], [...], ...], dtype=float32),  # 第一張影像的描述子     array([[...], [...], ...], dtype=float32),  # 第二張影像的描述子     ... ]`         

### 輸出類型詳細說明

#### 1. `keypoints` 的類型和內容

- **類型**：`list`（Python 列表）
- **元素類型**：`cv2.KeyPoint` 物件
- **`cv2.KeyPoint` 的屬性**：
    
    - `pt`：關鍵點的 `(x, y)` 座標。
    - `size`：關鍵點的尺度。
    - `angle`：關鍵點的方向（以度為單位，-1 表示未計算）。
    - `response`：響應值，表示該關鍵點的強度。
    - `octave`：關鍵點所在的金字塔層數。
    - `class_id`：類別 ID，通常用於識別不同物體的關鍵點。
#### 2. `descriptors` 的類型和內容

- **類型**：`numpy.ndarray`
- **形狀**：`(n, 128)`，其中 `n` 是影像中檢測到的關鍵點數量。
- **數據類型**：通常為 `float32`
- **說明**：
    - 每一行 `descriptors[i]` 對應於 `keypoints[i]` 的描述子，是一個 128 維的向量，用於描述該關鍵點周圍的局部圖像特徵。
    - 描述子可用於後續的影像匹配、物體識別等任務。
### 補充說明

#### 為什麼使用 `detectAndCompute`

- **`detectAndCompute`** 方法同時完成了特徵點的檢測和描述子的計算，這樣可以減少程式碼的複雜度並提高效率。
- 如果只需要檢測關鍵點，可以使用 `sift.detect(img, None)`。
- 如果已經有關鍵點，可以使用 `sift.compute(img, keypoints)` 來計算描述子。

#### 為什麼使用 `numpy.ndarray` 作為描述子儲存形式

- **效能**：NumPy 的多維陣列在處理大量數據時，比 Python 的原生列表更高效，特別是在數值計算和矩陣操作方面。
- **兼容性**：許多機器學習和數據處理庫（如 scikit-learn、TensorFlow、PyTorch 等）都直接支持 NumPy 陣列，方便進行後續的分析和處理。

#### 關鍵點與描述子的應用

- **影像匹配**：使用描述子可以計算不同影像間關鍵點的相似度，從而實現影像匹配。
- **物體識別**：通過比對描述子，可以在不同影像中識別出相同的物體。
- **影像拼接**：在全景圖生成中，通過匹配描述子來找到影像間的對應關鍵點，實現影像拼接。

### 總結

這段程式碼的主要功能是利用 SIFT 演算法檢測影像中的特徵點（關鍵點）並計算其描述子，將這些結果存儲在兩個列表中：

- **`self.keypoints_list`**：儲存每張影像的關鍵點列表。
- **`self.descriptors_list`**：儲存每張影像的描述子矩陣。

透過這些特徵點和描述子，可以進行各種影像分析和處理任務，如影像匹配、物體識別、影像拼接等。理解這些輸出類型及其結構，有助於進一步應用和擴展這段程式碼的功能。

```python

# [[ 2. feature matching ]]

matcher = cv2.FlannBasedMatcher_create()
for i in range(len(self.descriptors_list) - 1):
	matches_i = matcher.knnMatch(self.descriptors_list[i], self.descriptors_list[i + 1], k=2)
	good_matches = [m for m, n in matches_i if m.distance < 0.75 * n.distance]
	self.matches.append(good_matches)

```
`match_features` 方法的主要目的是在多張影像之間進行特徵匹配。具體來說，它使用 ==FLANN（Fast Library for Approximate Nearest Neighbors）==基於描述子的匹配，並應用 Lowe 的比率測試來篩選出「好的匹配」。
#### 1. 初始化特徵匹配器

`matcher = cv2.FlannBasedMatcher_create()`

- **功能**：
    - 使用 OpenCV 的 FLANN 基於匹配器來創建一個匹配器物件。
    - FLANN 是一種高效的近似最近鄰搜索方法，適用於大規模資料集的快速匹配。
- **輸出類型**：
    - `matcher` 是一個 `cv2.FlannBasedMatcher` 類型的物件，具備進行特徵匹配的方法。
    
1. 遍歷描述子列表進行匹配
```python
for i in range(len(self.descriptors_list) - 1):
    matches_i = matcher.knnMatch(self.descriptors_list[i], self.descriptors_list[i + 1], k=2)
    good_matches = [m for m, n in matches_i if m.distance < 0.75 * n.distance]
    self.matches.append(good_matches)    
```

### 詳細解析 `knnMatch` 方法及 `DMatch` 物件

您提供的程式碼片段如下：

python

複製程式碼

`matches_0 = matcher.knnMatch(self.descriptors_list[0], self.descriptors_list[1], k=2)`

其中：

- `self.descriptors_list[0]` 是第一張影像（`img1`）的描述子，形狀為 `(80904, 128)`。
- `self.descriptors_list[1]` 是第二張影像（`img2`）的描述子，形狀為 `(80116, 128)`。

假設 `matches_0` 的結果為：

python

複製程式碼

`matches_0 = [     [DMatch1_img1_img2, DMatch2_img1_img2],     [DMatch3_img1_img2, DMatch4_img1_img2],     ... ]`

### 您的理解是否正確？

您的理解基本上是正確的，以下是更詳細的解釋：

1. **`knnMatch` 方法的功能**
    
    python
    
    複製程式碼
    
    `matches_0 = matcher.knnMatch(self.descriptors_list[0], self.descriptors_list[1], k=2)`
    
    - **功能**：
        
        - `knnMatch` 方法用於查找每個查詢描述子（`self.descriptors_list[0]` 中的描述子）在訓練描述子（`self.descriptors_list[1]` 中的描述子）中最接近的 `k` 個匹配。
        - 在這裡，`k=2` 表示對於每個描述子，將找到其在 `img2` 中最接近的兩個描述子。
    - **參數說明**：
        
        - **`queryDescriptors`** (`self.descriptors_list[0]`)：查詢描述子集，來自第一張影像（`img1`）。
        - **`trainDescriptors`** (`self.descriptors_list[1]`)：訓練描述子集，來自第二張影像（`img2`）。
        - **`k`**：每個查詢描述子要找到的最近鄰數量，此處為 `2`。
    - **返回值**：
        
        - `matches_0` 是一個列表，其中每個元素是一個包含 `k` 個 `cv2.DMatch` 物件的子列表。
        - 每個子列表對應於 `img1` 中的一個描述子，包含其在 `img2` 中的 `k` 個最近鄰匹配。
2. **`DMatch` 物件的含義**
    
    每個 `DMatch` 物件包含以下屬性：
    
    - **`queryIdx`**：查詢描述子（`img1` 中的描述子）的索引。
    - **`trainIdx`**：訓練描述子（`img2` 中的描述子）的索引。
    - **`imgIdx`**：描述子所屬的圖像索引（多圖像匹配時使用）。
    - **`distance`**：兩個描述子之間的距離（通常是歐氏距離），距離越小表示匹配越好。
3. **具體範例說明**
    
    假設：
    
    - `self.descriptors_list[0]`（`img1`）有 80904 個描述子，每個描述子是 128 維向量。
    - `self.descriptors_list[1]`（`img2`）有 80116 個描述子，每個描述子是 128 維向量。
    
    執行 `knnMatch` 後，`matches_0` 的結構如下：
    
    python
    
    複製程式碼
    
    `matches_0 = [     [DMatch1_img1_img2, DMatch2_img1_img2],     [DMatch3_img1_img2, DMatch4_img1_img2],     ... ]`
    
    **解釋**：
    
    - **`DMatch1_img1_img2`**：
        
        - 對應於 `img1` 中的第 0 個描述子（`queryIdx=0`）。
        - 是 `img2` 中最接近的描述子（假設為 `img2` 的第 10 個描述子，`trainIdx=10`）。
        - `distance` 表示 `img1` 的第 0 個描述子與 `img2` 的第 10 個描述子之間的距離。
    - **`DMatch2_img1_img2`**：
        
        - 同樣對應於 `img1` 中的第 0 個描述子（`queryIdx=0`）。
        - 是 `img2` 中第二接近的描述子（假設為 `img2` 的第 20 個描述子，`trainIdx=20`）。
        - `distance` 表示 `img1` 的第 0 個描述子與 `img2` 的第 20 個描述子之間的距離。
    - **後續元素**：
        
        - 每個子列表中的 `DMatch` 物件依次對應於 `img1` 中的下一個描述子及其在 `img2` 中的最近鄰匹配。
4. **“最接近”的定義**
    
    - **計算方式**：
        
        - “最接近”是基於描述子之間的距離計算的。對於 SIFT 描述子，通常使用歐氏距離（Euclidean Distance）來衡量兩個 128 維向量之間的相似性。
        - 距離越小，表示兩個描述子在特徵空間中越接近，匹配程度越高。
    - **具體計算**：
        
        - 對於每個 `img1` 中的描述子 `D_i`，計算其與 `img2` 中所有描述子 `D_j` 的距離。
        - 選取距離最小的兩個描述子作為 `k=2` 的匹配結果。
5. **具體範例**
    
    假設 `img1` 中的第 0 個描述子 `D1` 與 `img2` 中的描述子計算距離後，最接近的兩個描述子為 `D5` 和 `D7`，其距離分別為 `0.5` 和 `0.8`。
    
    那麼：
    
    python
    
    複製程式碼
    
    `matches_0 = [     [DMatch1_img1_img2, DMatch2_img1_img2],  # DMatch1: D1 <-> D5, DMatch2: D1 <-> D7     ... ]`
    
    - **`DMatch1_img1_img2`**：
        
        - `queryIdx=0`（`img1` 的 D1）
        - `trainIdx=4`（`img2` 的 D5，索引從 0 開始）
        - `distance=0.5`
    - **`DMatch2_img1_img2`**：
        
        - `queryIdx=0`（`img1` 的 D1）
        - `trainIdx=6`（`img2` 的 D7）
        - `distance=0.8`
6. **總結您的理解**
    

根據上述解析，您的理解是正確的：

- **`DMatch1_img1_img2`**：
    
    - 代表 `img1` 上的第 1 個描述子（索引為 0）的最佳匹配，即與 `img2` 中最接近的一個描述子匹配。
    - `queryIdx` 對應於 `img1` 的描述子索引（0）。
    - `trainIdx` 對應於 `img2` 的描述子索引（如 10）。
    - `distance` 是兩個 128 維向量之間的距離。
- **`DMatch2_img1_img2`**：
    
    - 代表 `img1` 上的第 1 個描述子（索引為 0）的第二最佳匹配，即與 `img2` 中第二接近的一個描述子匹配。
    - `queryIdx` 同樣對應於 `img1` 的描述子索引（0）。
    - `trainIdx` 對應於 `img2` 的描述子索引（如 20）。
    - `distance` 是兩個 128 維向量之間的距離。
- **“最接近”**：
    
    - 是根據兩個 128 維向量之間的距離計算的，通常使用歐氏距離。
    - 距離越小，匹配越好。

### 補充說明

1. **`knnMatch` 的返回結構**
    
    - **返回值類型**：
        
        - `knnMatch` 返回一個列表，其中每個元素是一個包含 `k` 個 `cv2.DMatch` 物件的子列表。
        - 例如，對於 `k=2`，每個子列表包含兩個 `DMatch` 物件。
    - **結構說明**：
        
        - 每個 `DMatch` 物件對應於 `img1` 中的一個描述子與 `img2` 中的一個描述子的匹配。
        - `matches_0[i][0]` 是第 `i` 個 `img1` 描述子的最佳匹配。
        - `matches_0[i][1]` 是第 `i` 個 `img1` 描述子的第二最佳匹配。
2. **Lowe 的比率測試**
    
    在後續的程式碼中，您使用了 Lowe 的比率測試來篩選「好的匹配」：
    
    python
    
    複製程式碼
    
    `good_matches = [m for m, n in matches_i if m.distance < 0.75 * n.distance]`
    
    - **目的**：
        
        - 確保匹配的唯一性和可靠性，減少錯誤匹配。
        - 僅保留那些第一匹配與第二匹配之間距離比例小於 0.75 的匹配。
    - **運作原理**：
        
        - 對於每個 `img1` 中的描述子，檢查其最佳匹配（`m`）與第二最佳匹配（`n`）之間的距離比。
        - 只有當 `m.distance` 明顯小於 `n.distance` 時，才認為 `m` 是一個好的匹配。
3. **`cv2.DMatch` 的詳細屬性**
    
    - **`queryIdx`**：查詢描述子（來自 `img1`）的索引。
    - **`trainIdx`**：訓練描述子（來自 `img2`）的索引。
    - **`imgIdx`**：描述子所在的圖像索引（在多圖像匹配時使用，單圖像匹配通常為 0）。
    - **`distance`**：描述子之間的距離，表示匹配的相似度。
    
    **範例**：
    
    python
    
    複製程式碼
    
    `DMatch1_img1_img2 = cv2.DMatch(_queryIdx=0, _trainIdx=10, _imgIdx=0, _distance=0.5) DMatch2_img1_img2 = cv2.DMatch(_queryIdx=0, _trainIdx=20, _imgIdx=0, _distance=0.8)`
    
    - **`DMatch1_img1_img2`**：
        
        - `queryIdx=0`：來自 `img1` 的第 0 個描述子。
        - `trainIdx=10`：來自 `img2` 的第 10 個描述子。
        - `distance=0.5`：兩個描述子之間的距離。
    - **`DMatch2_img1_img2`**：
        
        - `queryIdx=0`：來自 `img1` 的第 0 個描述子。
        - `trainIdx=20`：來自 `img2` 的第 20 個描述子。
        - `distance=0.8`：兩個描述子之間的距離。
4. **匹配結果的應用**
    
    - **影像拼接（Image Stitching）**：
        
        - 利用 `good_matches` 對應的特徵點，計算圖像間的變換矩陣，將多張影像拼接成全景圖。
    - **三維重建（3D Reconstruction）**：
        
        - 通過匹配特徵點，計算視差，構建三維點雲。
    - **物體識別（Object Recognition）**：
        
        - 通過匹配描述子，識別和追蹤物體。

### 總結

您的理解基本正確：

- **`DMatch1_img1_img2`** 確實代表 `img1` 中的第 1 個描述子（索引為 0）與 `img2` 中最接近的一個描述子（索引為 10）的匹配。
- **`DMatch2_img1_img2`** 代表同一個 `img1` 的描述子（索引為 0）與 `img2` 中第二接近的一個描述子（索引為 20）的匹配。
- **“最接近”** 是基於兩個 128 維描述子向量之間的距離計算得出的，通常使用歐氏距離來衡量。

這種匹配方式有助於找到兩張影像之間相似的特徵點，並通過篩選可靠的匹配來提高後續處理（如圖像拼接或三維重建）的準確性和穩定性。

如果您有進一步的問題或需要更多範例，請隨時告訴我！

o1-mini

##### 你說：

請中文詳細解釋good_matches = [m for m, n in matches_i if m.distance < 0.75 * n.distance], 請問m, n是否就是指DMatch1_img1_img2跟DMatch2_img1_img2如果是指第一行? 如果符合條件, DMatch1_img1_img2就加入good_matches的list中. 以上是否正確?

ChatGPT

Thought 花的時間為 2 秒

### 詳細解析 `good_matches` 的篩選過程

您提供的程式碼片段如下：

python

複製程式碼

`good_matches = [m for m, n in matches_i if m.distance < 0.75 * n.distance]`

您對這行程式碼的理解基本上是正確的。以下是對這段程式碼的詳細解釋，並確認您的理解：

#### 1. 背景說明

在進行特徵匹配時，通常會為每個描述子找到其在另一張影像中的前 `k` 個最近鄰。這裡使用的是 `k=2`，即每個描述子會找到兩個最接近的描述子。這些匹配結果被存儲在 `matches_i` 中，結構類似於：

python

複製程式碼

`matches_i = [     [DMatch1_img1_img2, DMatch2_img1_img2],     [DMatch3_img1_img2, DMatch4_img1_img2],     ... ]`

每一個子列表（如 `[DMatch1_img1_img2, DMatch2_img1_img2]`）包含了兩個 `cv2.DMatch` 物件，分別代表對應於 `img1` 中某個描述子在 `img2` 中的第一和第二近鄰匹配。

#### 2. `good_matches` 的篩選條件

python

複製程式碼

`good_matches = [m for m, n in matches_i if m.distance < 0.75 * n.distance]`

這行程式碼的目的是篩選出「好的匹配」，具體步驟如下：

1. **解構匹配對**：
    
    - `m` 代表每一對匹配中的第一個 `DMatch` 物件（即最佳匹配）。
    - `n` 代表每一對匹配中的第二個 `DMatch` 物件（即次佳匹配）。
2. **應用 Lowe 的比率測試（Ratio Test）**：
    
    - 比率測試是 David Lowe 提出的一種篩選方法，用於提高匹配的可靠性。
    - 條件 `m.distance < 0.75 * n.distance` 意味著：
        - 如果第一個匹配 (`m`) 的距離明顯小於第二個匹配 (`n`) 的距離（這裡比例設定為 0.75），則認為這是一個可靠的匹配。
        - 這樣做的目的是確保匹配的唯一性，避免因為描述子在特徵空間中太過接近而產生的錯誤匹配。
3. **添加符合條件的匹配**：
    
    - 只有當 `m.distance < 0.75 * n.distance` 時，`m` 會被加入到 `good_matches` 列表中。

#### 3. 您的理解是否正確？

是的，您的理解是正確的。具體來說：

- **`m` 和 `n` 的含義**：
    
    - 在這裡，`m` 和 `n` 分別代表每一對匹配中的第一個和第二個 `cv2.DMatch` 物件。
    - 例如，對於 `matches_i` 中的第一個子列表 `[DMatch1_img1_img2, DMatch2_img1_img2]`：
        - `m` 對應於 `DMatch1_img1_img2`（第一個描述子的最佳匹配）。
        - `n` 對應於 `DMatch2_img1_img2`（第一個描述子的次佳匹配）。
- **篩選條件**：
    
    - 如果 `DMatch1_img1_img2.distance < 0.75 * DMatch2_img1_img2.distance`，則 `DMatch1_img1_img2` 會被加入到 `good_matches` 列表中。
- **距離計算**：
    
    - 距離是基於描述子之間的相似性計算的。對於 SIFT 描述子，通常使用歐氏距離（Euclidean Distance）來衡量兩個 128 維向量之間的距離。
    - 距離越小，表示兩個描述子越相似，匹配越可靠。

#### 4. 具體範例說明

假設我們有兩張影像 `img1` 和 `img2`，並且進行特徵匹配後得到了以下的 `matches_i`：

python

複製程式碼

`matches_i = [     [DMatch1_img1_img2, DMatch2_img1_img2],     [DMatch3_img1_img2, DMatch4_img1_img2],     [DMatch5_img1_img2, DMatch6_img1_img2],     # ...更多匹配對 ]`

每個 `DMatch` 物件包含以下屬性：

- `queryIdx`：來自 `img1` 的描述子索引。
- `trainIdx`：來自 `img2` 的描述子索引。
- `distance`：描述子之間的距離。

假設具體值如下：

python

複製程式碼

`DMatch1_img1_img2 = cv2.DMatch(_queryIdx=0, _trainIdx=100, _imgIdx=0, _distance=0.5) DMatch2_img1_img2 = cv2.DMatch(_queryIdx=0, _trainIdx=200, _imgIdx=0, _distance=0.8)  DMatch3_img1_img2 = cv2.DMatch(_queryIdx=1, _trainIdx=150, _imgIdx=0, _distance=0.4) DMatch4_img1_img2 = cv2.DMatch(_queryIdx=1, _trainIdx=250, _imgIdx=0, _distance=0.6)  DMatch5_img1_img2 = cv2.DMatch(_queryIdx=2, _trainIdx=180, _imgIdx=0, _distance=0.3) DMatch6_img1_img2 = cv2.DMatch(_queryIdx=2, _trainIdx=280, _imgIdx=0, _distance=0.7)`

應用篩選條件後：

python

複製程式碼

`good_matches = [m for m, n in matches_i if m.distance < 0.75 * n.distance]`

具體步驟如下：

1. **第一對匹配**：
    
    - `m = DMatch1_img1_img2`（distance = 0.5）
    - `n = DMatch2_img1_img2`（distance = 0.8）
    - 判斷：`0.5 < 0.75 * 0.8` → `0.5 < 0.6` → **符合條件**
    - `good_matches` 添加 `DMatch1_img1_img2`
2. **第二對匹配**：
    
    - `m = DMatch3_img1_img2`（distance = 0.4）
    - `n = DMatch4_img1_img2`（distance = 0.6）
    - 判斷：`0.4 < 0.75 * 0.6` → `0.4 < 0.45` → **符合條件**
    - `good_matches` 添加 `DMatch3_img1_img2`
3. **第三對匹配**：
    
    - `m = DMatch5_img1_img2`（distance = 0.3）
    - `n = DMatch6_img1_img2`（distance = 0.7）
    - 判斷：`0.3 < 0.75 * 0.7` → `0.3 < 0.525` → **符合條件**
    - `good_matches` 添加 `DMatch5_img1_img2`

最終，`good_matches` 將包含：

python

複製程式碼

`good_matches = [DMatch1_img1_img2, DMatch3_img1_img2, DMatch5_img1_img2]`

#### 5. 總結您的理解

您的理解如下：

- **`m` 和 `n` 的含義**：
    
    - `m` 和 `n` 分別代表每一對匹配中的第一個和第二個 `cv2.DMatch` 物件。
    - 在 `matches_i` 的第一行 `[DMatch1_img1_img2, DMatch2_img1_img2]` 中，`m = DMatch1_img1_img2`，`n = DMatch2_img1_img2`。
- **條件判斷**：
    
    - 如果 `DMatch1_img1_img2.distance < 0.75 * DMatch2_img1_img2.distance`，則 `DMatch1_img1_img2` 被視為一個好的匹配，並被加入到 `good_matches` 列表中。
- **距離計算方式**：
    
    - “最接近” 是根據兩個 128 維向量之間的距離計算的，通常使用歐氏距離（Euclidean Distance）。
- **結果**：
    
    - 符合條件的 `m`（即第一個匹配）會被加入到 `good_matches` 中，而 `n`（第二個匹配）不會被加入，除非它們也符合自己對應的條件。

因此，您的理解完全正確！

#### 6. 補充說明

##### a. 為什麼使用 `k=2`？

Lowe 的比率測試依賴於找到每個描述子的前兩個最近鄰。這樣可以比較最佳匹配和次佳匹配之間的距離，從而判斷匹配的可靠性。如果只找一個最近鄰，無法進行比率測試。

##### b. 比率閾值的選擇

- **0.75** 是一個常見的選擇，但根據具體應用場景，可以調整這個閾值。
- 一般而言，閾值越低，篩選出的匹配越嚴格，可能會減少錯誤匹配，但也可能丟失一些正確匹配。
- 閾值越高，匹配數量越多，但可能會引入更多的錯誤匹配。

##### c. `cv2.DMatch` 物件的屬性

- **`queryIdx`**：來自查詢集（`img1`）的描述子索引。
- **`trainIdx`**：來自訓練集（`img2`）的描述子索引。
- **`imgIdx`**：描述子所屬的圖像索引（在多圖像匹配時使用，單圖像匹配通常為 0）。
- **`distance`**：描述子之間的距離，表示匹配的相似度。距離越小，匹配越好。

##### d. 具體應用

- **影像拼接（Image Stitching）**：
    
    - 使用 `good_matches` 中的匹配對來估計兩張影像之間的變換矩陣，從而進行拼接。
- **三維重建（3D Reconstruction）**：
    
    - 通過匹配多張影像中的特徵點，計算相對位置和視差，構建三維點雲。
- **物體識別（Object Recognition）**：
    
    - 在不同影像中找到相同物體的匹配特徵點，進行物體識別和追蹤。

##### e. 進一步優化

- **使用更高效的描述子**：
    
    - 如 ORB 描述子，它們比 SIFT 更快，但可能在匹配精度上有所不同。
- **多線程或並行處理**：
    
    - 當處理大量描述子時，可以考慮使用多線程或並行處理來加速匹配過程。
- **後續處理**：
    
    - 在獲得 `good_matches` 之後，可以使用 RANSAC（Random Sample Consensus）來進一步篩選匹配點，消除錯誤匹配，提高匹配質量。

### 總結

您對於以下幾點的理解是正確的：

1. **`m` 和 `n` 的含義**：
    
    - `m` 代表每一對匹配中的第一個 `cv2.DMatch` 物件（最佳匹配）。
    - `n` 代表每一對匹配中的第二個 `cv2.DMatch` 物件（次佳匹配）。
2. **篩選條件**：
    
    - 如果 `m.distance < 0.75 * n.distance`，則 `m` 被認為是好的匹配，並加入到 `good_matches` 列表中。
3. **距離計算方式**：
    
    - 基於描述子之間的 128 維向量距離（通常是歐氏距離）來判斷匹配的好壞。

這樣的篩選機制有助於提高匹配的準確性，減少錯誤匹配，從而提升後續影像處理步驟（如影像拼接、三維重建等）的效果和穩定性。

```python
# get src_pts(pts in img1) and dst_pts(pts in img2), N=5272
src_pts = np.float32([self.keypoints_list[i][m.queryIdx].pt for m in self.matches[i]]).reshape(-1, 1, 2)

dst_pts = np.float32([self.keypoints_list[i + 1][m.trainIdx].pt for m in self.matches[i]]).reshape(-1, 1, 2)

# Calculate Essential matrix(E), N=5272 
E, mask = cv2.findEssentialMat(src_pts, dst_pts, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0) 

# Recover pose (R:rotate matrix, t:transition matrix), N=5272
_, R, t, mask_pose = cv2.recoverPose(E, src_pts, dst_pts, self.K)
  
```
src_pts, shape=(5272,1,2) 
dst_pts, shape=(5272,1,2) 
E: numpy.ndarray(3,3),   mask: numpy.ndarray(N,1)
R: numpy.ndarray(3,3),   t: numpy.ndarray(3,1),   mask_pose: numpy.ndarray(N,1)


### 代码逐步拆解

#### 1. 初始化稀疏点云存储

`sparse_points = []`

- **功能**：用于存储从每对相邻图像中计算得到的稀疏三维点云。

---

#### 2. 遍历匹配点对

```python
for i in range(len(self.matches)):
  src_pts = np.float32([self.keypoints_list[i][m.queryIdx].pt for m in self.matches[i]]).reshape(-1, 1, 2)

  dst_pts = np.float32([self.keypoints_list[i + 1][m.trainIdx].pt for m in self.matches[i]]).reshape(-1, 1, 2)
```

- **功能**：将特征点的匹配对从像素坐标提取出来。
    - **`self.matches[i]`**：第 iii 对相邻图像之间的匹配结果，包含匹配的特征点索引。
    - **`self.keypoints_list[i][m.queryIdx].pt`**：获取图像 iii 中的匹配点的像素坐标。
    - **`self.keypoints_list[i + 1][m.trainIdx].pt`**：获取图像 i+1i+1i+1 中的匹配点的像素坐标。
- **输入输出示例**：
    - 假设 `self.matches[i]` 包含匹配对： 匹配对=[(点1, 点2),(点3, 点4)]\text{匹配对} = [\text{(点1, 点2)}, \text{(点3, 点4)}]匹配对=[(点1, 点2),(点3, 点4)]
    - 提取的像素坐标：
        `src_pts = np.array([[[100, 200]], [[150, 250]]]) dst_pts = np.array([[[110, 210]], [[160, 260]]])`
        


```python
# Calculate Essential matrix(E), N=5272 
E, mask = cv2.findEssentialMat(src_pts, dst_pts, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0) 
```
E: numpy.ndarray(3,3),   mask: numpy.ndarray(N,1)

你提出的疑问很有道理，但需要明确一点：**在对极几何中，本质矩阵 EEE 和基础矩阵 FFF 的适用场景不同**，这导致了为什么需要将像素坐标转换为归一化相机坐标。

以下从头再解释 `cv2.findEssentialMat` 的完整流程，并纠正不必要的模糊点。

---

### 为什么需要将像素坐标转换为归一化相机坐标？

1. **基础矩阵 FFF**：
    
    - FFF 是直接作用于像素坐标的对极几何描述。
    - 满足： q2T⋅F⋅q1=0q_2^T \cdot F \cdot q_1 = 0q2T​⋅F⋅q1​=0 其中：
        - q1=[u1,v1,1]Tq_1 = [u_1, v_1, 1]^Tq1​=[u1​,v1​,1]T：相机1的像素坐标。
        - q2=[u2,v2,1]Tq_2 = [u_2, v_2, 1]^Tq2​=[u2​,v2​,1]T：相机2的像素坐标。
2. **本质矩阵 EEE**：
    
    - EEE 是归一化相机坐标系下的对极几何描述。
    - 满足： p2T⋅E⋅p1=0p_2^T \cdot E \cdot p_1 = 0p2T​⋅E⋅p1​=0 其中：
        - p1=[x1,y1,1]Tp_1 = [x_1, y_1, 1]^Tp1​=[x1​,y1​,1]T：相机1的归一化相机坐标。
        - p2=[x2,y2,1]Tp_2 = [x_2, y_2, 1]^Tp2​=[x2​,y2​,1]T：相机2的归一化相机坐标。
3. **转换关系**：
    
    - 像素坐标 q1,q2q_1, q_2q1​,q2​ 通过内参矩阵 KKK 转换为归一化相机坐标 p1,p2p_1, p_2p1​,p2​： p1=K−1⋅q1,p2=K−1⋅q2p_1 = K^{-1} \cdot q_1, \quad p_2 = K^{-1} \cdot q_2p1​=K−1⋅q1​,p2​=K−1⋅q2​
4. **原因**：
    
    - 本质矩阵 EEE 的定义仅在归一化相机坐标系下有效。因此，使用 EEE 时必须将像素坐标 q1,q2q_1, q_2q1​,q2​ 转换为归一化相机坐标 p1,p2p_1, p_2p1​,p2​。
    - 如果直接使用像素坐标，就需要使用基础矩阵 FFF 而不是 EEE。

---

### 函数 `cv2.findEssentialMat` 的完整流程

#### **输入参数**

- src_pts,dst_pts\text{src\_pts}, \text{dst\_pts}src_pts,dst_pts：两个相机中匹配的特征点对（像素坐标）。
- KKK：相机的内参矩阵。

#### **输出参数**

- EEE：计算出的本质矩阵。
- mask\text{mask}mask：内点掩码，标记符合对极几何的点对。

#### **核心步骤**

##### **步骤 1：像素坐标转归一化相机坐标**

将输入的像素坐标转换为归一化相机坐标：

p1=K−1⋅q1,p2=K−1⋅q2p_1 = K^{-1} \cdot q_1, \quad p_2 = K^{-1} \cdot q_2p1​=K−1⋅q1​,p2​=K−1⋅q2​

- q1,q2q_1, q_2q1​,q2​：像素坐标。
- p1,p2p_1, p_2p1​,p2​：归一化相机坐标。

##### **步骤 2：计算基础矩阵 FFF**

通过点对 p1,p2p_1, p_2p1​,p2​ 计算基础矩阵 FFF：

- 使用八点算法或更鲁棒的 RANSAC 方法，确保满足对极几何约束： p2T⋅F⋅p1=0p_2^T \cdot F \cdot p_1 = 0p2T​⋅F⋅p1​=0

##### **步骤 3：将基础矩阵 FFF 转换为本质矩阵 EEE**

根据内参矩阵 KKK，将基础矩阵 FFF 转换为本质矩阵 EEE：

E=KT⋅F⋅KE = K^T \cdot F \cdot KE=KT⋅F⋅K

##### **步骤 4：RANSAC 去除外点**

使用 RANSAC 方法筛选内点：

1. 随机选择点对计算 EEE。
2. 验证所有点对是否符合对极约束： p2T⋅E⋅p1≈0p_2^T \cdot E \cdot p_1 \approx 0p2T​⋅E⋅p1​≈0
3. 迭代优化，输出最佳 EEE 和内点掩码。

---

### 具体例子

#### 输入数据

1. 匹配点对：
    
    - 相机1中的点（像素坐标）： src_pts=[[600,400][650,420]]\text{src\_pts} = \begin{bmatrix} [600, 400] \\ [650, 420] \end{bmatrix}src_pts=[[600,400][650,420]​]
    - 相机2中的点（像素坐标）： dst_pts=[[610,410][660,430]]\text{dst\_pts} = \begin{bmatrix} [610, 410] \\ [660, 430] \end{bmatrix}dst_pts=[[610,410][660,430]​]
2. 相机内参矩阵：
    
    K=[1000050001000500001]K = \begin{bmatrix} 1000 & 0 & 500 \\ 0 & 1000 & 500 \\ 0 & 0 & 1 \end{bmatrix}K=​100000​010000​5005001​​

#### 计算过程

##### **步骤 1：像素坐标转归一化相机坐标**

计算归一化相机坐标：

p1=K−1⋅[6004001]=[0.1−0.11],p2=K−1⋅[6104101]=[0.11−0.091]p_1 = K^{-1} \cdot \begin{bmatrix} 600 \\ 400 \\ 1 \end{bmatrix} = \begin{bmatrix} 0.1 \\ -0.1 \\ 1 \end{bmatrix}, \quad p_2 = K^{-1} \cdot \begin{bmatrix} 610 \\ 410 \\ 1 \end{bmatrix} = \begin{bmatrix} 0.11 \\ -0.09 \\ 1 \end{bmatrix}p1​=K−1⋅​6004001​​=​0.1−0.11​​,p2​=K−1⋅​6104101​​=​0.11−0.091​​

##### **步骤 2：计算基础矩阵 FFF**

通过八点算法计算基础矩阵：

F=[0−0.0010.020.0010−0.5−0.020.51]F = \begin{bmatrix} 0 & -0.001 & 0.02 \\ 0.001 & 0 & -0.5 \\ -0.02 & 0.5 & 1 \end{bmatrix}F=​00.001−0.02​−0.00100.5​0.02−0.51​​

##### **步骤 3：将基础矩阵转换为本质矩阵**

将基础矩阵 FFF 转换为本质矩阵：

E=KT⋅F⋅KE = K^T \cdot F \cdot KE=KT⋅F⋅K

假设结果为：

E=[0−10.510−0.2−0.50.21]E = \begin{bmatrix} 0 & -1 & 0.5 \\ 1 & 0 & -0.2 \\ -0.5 & 0.2 & 1 \end{bmatrix}E=​01−0.5​−100.2​0.5−0.21​​

##### **步骤 4：RANSAC 优化**

通过 RANSAC 选择符合对极几何的点对，输出最佳本质矩阵 EEE 和内点掩码。

---

### 结论

#### **流程总结**

1. 像素坐标通过内参矩阵转换为归一化相机坐标。
2. 计算基础矩阵 FFF。
3. 使用内参矩阵将基础矩阵转换为本质矩阵 EEE。
4. 通过 RANSAC 优化，筛选内点并输出鲁棒的本质矩阵。

#### **本质矩阵的作用**

- 描述相机的旋转和平移。
- 用于后续恢复相机姿态（旋转矩阵 RRR 和平移向量 ttt）。

通过这些步骤，`cv2.findEssentialMat` 将匹配点对和相机内参矩阵结合起来，计算出多视角几何中核心的本质矩阵。

4o



```python
# Recover pose (R:rotate matrix, t:transition matrix), N=5272
_, R, t, mask_pose = cv2.recoverPose(E, src_pts, dst_pts, self.K)
```
R: numpy.ndarray(3,3),   t: numpy.ndarray(3,1),   mask_pose: numpy.ndarray(N,1)

### 函数 `cv2.recoverPose` 的详细解析

`cv2.recoverPose` 是 OpenCV 提供的函数，用于从 **本质矩阵 EEE** 和匹配点对（像素坐标）恢复两个相机之间的相对姿态，即 **旋转矩阵 RRR** 和 **平移向量 ttt**。下面通过数学公式、函数流程以及具体例子详细解释其工作机制。

---

### 1. 背景知识

#### **本质矩阵的作用**

本质矩阵 E 描述两个相机在归一化相机坐标系下的几何关系，满足对极几何约束：

$p_2^T \cdot E \cdot p_1 = 0$

其中：

- $p_1 = [x_1, y_1, 1]^T$：相机1中的归一化相机坐标。
- $p_2 = [x_2, y_2, 1]^T$：相机2中的归一化相机坐标。

#### **旋转矩阵 RRR 和平移向量 ttt**

本质矩阵 EEE 是两个相机之间相对旋转和平移的直接描述：

$E = [t]_\times \cdot R$

- $[t]_\times$​：平移向量 t 的反对称矩阵，用于表示向量积的线性化形式。$\times = \begin{bmatrix} 0 & -t_z & t_y \\ t_z & 0 & -t_x \\ -t_y & t_x & 0 \end{bmatrix}$
- RRR：旋转矩阵。

`cv2.recoverPose` 使用这些关系恢复 RRR 和 ttt。

---

### 2. 函数 `cv2.recoverPose` 的输入输出

#### **输入参数**

1. **`E`**：
    - 本质矩阵，通常由 `cv2.findEssentialMat` 计算得到。
2. **`src_pts` 和 `dst_pts`**：
    - 匹配的特征点对：
        - src_ptssrc\_ptssrc_pts：相机1中的点（像素坐标）。
        - dst_ptsdst\_ptsdst_pts：相机2中的点（像素坐标）。
3. **`self.K`**：
    - 相机的内参矩阵，用于将像素坐标转换为归一化相机坐标： K=[fx0cx0fycy001]K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}K=​fx​00​0fy​0​cx​cy​1​​

#### **输出参数**

1. **`R`**：
    - 两个相机之间的旋转矩阵。
2. **`t`**：
    - 两个相机之间的平移向量。
3. **`mask_pose`**：
    - 标记哪些点对是内点（符合几何约束的点）。

---

### 3. 函数计算流程

#### **步骤 1：将像素坐标转换为归一化相机坐标**

通过内参矩阵 KKK，将匹配的像素坐标转换为归一化相机坐标：

p1=K−1⋅[u1v11],p2=K−1⋅[u2v21]p_1 = K^{-1} \cdot \begin{bmatrix} u_1 \\ v_1 \\ 1 \end{bmatrix}, \quad p_2 = K^{-1} \cdot \begin{bmatrix} u_2 \\ v_2 \\ 1 \end{bmatrix}p1​=K−1⋅​u1​v1​1​​,p2​=K−1⋅​u2​v2​1​​

#### **步骤 2：分解本质矩阵 EEE**

通过奇异值分解（SVD）将本质矩阵 EEE 分解为三个矩阵：

E=U⋅Σ⋅VTE = U \cdot \Sigma \cdot V^TE=U⋅Σ⋅VT

其中：

- U,VU, VU,V：正交矩阵。
- Σ=diag(1,1,0)\Sigma = \text{diag}(1, 1, 0)Σ=diag(1,1,0)：奇异值矩阵。

#### **步骤 3：从分解中恢复旋转矩阵 RRR 和平移向量 ttt**

从 SVD 的结果中恢复 RRR 和 ttt：

1. 旋转矩阵的候选解：
    
    R1=U⋅W⋅VT,R2=U⋅WT⋅VTR_1 = U \cdot W \cdot V^T, \quad R_2 = U \cdot W^T \cdot V^TR1​=U⋅W⋅VT,R2​=U⋅WT⋅VT
    
    其中：
    
    W=[0−10100001]W = \begin{bmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{bmatrix}W=​010​−100​001​​
2. 平移向量的候选解：
    
    t=U[:,2]t = U[:, 2]t=U[:,2]
    
    （取 UUU 的第三列作为平移向量）。
    

#### **步骤 4：选择正确的解**

因为从 EEE 的分解中可能得到 4 种组合 (R1,t),(R1,−t),(R2,t),(R2,−t)(R_1, t), (R_1, -t), (R_2, t), (R_2, -t)(R1​,t),(R1​,−t),(R2​,t),(R2​,−t)，需要通过点对的三角测量验证哪种组合满足实际的几何关系。

---

### 4. 具体例子

#### 假设输入数据

1. **本质矩阵 EEE**：
    
    E=[0−10.510−0.2−0.50.21]E = \begin{bmatrix} 0 & -1 & 0.5 \\ 1 & 0 & -0.2 \\ -0.5 & 0.2 & 1 \end{bmatrix}E=​01−0.5​−100.2​0.5−0.21​​
2. **匹配点对**：
    
    - 相机1中的点： src_pts=[[600,400][650,420]]src\_pts = \begin{bmatrix} [600, 400] \\ [650, 420] \end{bmatrix}src_pts=[[600,400][650,420]​]
    - 相机2中的点： dst_pts=[[610,410][660,430]]dst\_pts = \begin{bmatrix} [610, 410] \\ [660, 430] \end{bmatrix}dst_pts=[[610,410][660,430]​]
3. **相机内参矩阵 KKK**：
    
    K=[1000050001000500001]K = \begin{bmatrix} 1000 & 0 & 500 \\ 0 & 1000 & 500 \\ 0 & 0 & 1 \end{bmatrix}K=​100000​010000​5005001​​

#### 计算步骤

##### **步骤 1：像素坐标转归一化相机坐标**

归一化相机坐标：

p1=K−1⋅[6004001]=[0.1−0.11],p2=K−1⋅[6104101]=[0.11−0.091]p_1 = K^{-1} \cdot \begin{bmatrix} 600 \\ 400 \\ 1 \end{bmatrix} = \begin{bmatrix} 0.1 \\ -0.1 \\ 1 \end{bmatrix}, \quad p_2 = K^{-1} \cdot \begin{bmatrix} 610 \\ 410 \\ 1 \end{bmatrix} = \begin{bmatrix} 0.11 \\ -0.09 \\ 1 \end{bmatrix}p1​=K−1⋅​6004001​​=​0.1−0.11​​,p2​=K−1⋅​6104101​​=​0.11−0.091​​

##### **步骤 2：分解本质矩阵**

对本质矩阵 EEE 进行 SVD 分解：

U,Σ,VT=SVD(E)U, \Sigma, V^T = \text{SVD}(E)U,Σ,VT=SVD(E)

假设分解结果：

U=[0−10100001],VT=[0−10100001]U = \begin{bmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{bmatrix}, \quad V^T = \begin{bmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{bmatrix}U=​010​−100​001​​,VT=​010​−100​001​​

##### **步骤 3：恢复 RRR 和 ttt**

使用公式恢复 RRR 和 ttt 的候选解：

1. 旋转矩阵候选解：
    
    R1=U⋅W⋅VT,R2=U⋅WT⋅VTR_1 = U \cdot W \cdot V^T, \quad R_2 = U \cdot W^T \cdot V^TR1​=U⋅W⋅VT,R2​=U⋅WT⋅VT
    
    假设计算结果：
    
    R=[0.999−0.010.020.010.999−0.03−0.020.030.999]R = \begin{bmatrix} 0.999 & -0.01 & 0.02 \\ 0.01 & 0.999 & -0.03 \\ -0.02 & 0.03 & 0.999 \end{bmatrix}R=​0.9990.01−0.02​−0.010.9990.03​0.02−0.030.999​​
2. 平移向量候选解：
    
    t=[0.10.20.3]t = \begin{bmatrix} 0.1 \\ 0.2 \\ 0.3 \end{bmatrix}t=​0.10.20.3​​

##### **步骤 4：选择正确解**

通过点对验证，选择最终的 RRR 和 ttt。

---

### 5. 总结

#### **函数核心步骤**

1. 使用内参矩阵将像素坐标转换为归一化相机坐标。
2. 对本质矩阵进行 SVD 分解，恢复候选的 RRR 和 ttt。
3. 验证点对几何关系，选择正确的解。

#### **最终结果**

- 恢复的旋转矩阵 RRR： R=[0.999−0.010.020.010.999−0.03−0.020.030.999]R = \begin{bmatrix} 0.999 & -0.01 & 0.02 \\ 0.01 & 0.999 & -0.03 \\ -0.02 & 0.03 & 0.999 \end{bmatrix}R=​0.9990.01−0.02​−0.010.9990.03​0.02−0.030.999​​
- 平移向量 ttt： t=[0.10.20.3]t = \begin{bmatrix} 0.1 \\ 0.2 \\ 0.3 \end{bmatrix}t=​0.10.20.3​​

#### **实际应用**

- 用于多视角几何中的相机姿态估计。
- 是三维重建、SLAM（同时定位与建图）等任务的关键步骤。
---
```python
 # triangulation to sparse point cloud
proj1 = np.hstack((np.eye(3), np.zeros((3, 1))))
proj2 = np.hstack((R, t))
proj1 = self.K @ proj1
proj2 = self.K @ proj2
```
#### **投影矩阵的定义**

投影矩阵 PPP 是将三维点从世界坐标系映射到相机的像素坐标系的工具：

$\huge \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = P \cdot \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}​$

其中：

- $[X, Y, Z, 1]^T$：三维点的齐次坐标（在世界坐标系中）。
- $[u, v, 1]^T$：像素坐标。
- P：投影矩阵，形式为： $\huge P = K \cdot [R | t]$
    - KKK：相机内参矩阵。
    - [R∣t]：相机外参矩阵，描述相机在世界坐标系中的旋转和平移。

#### **两相机投影矩阵的用途**

在立体视觉中，两个相机的投影矩阵 P1P_1P1​ 和 P2P_2P2​ 可以用于三角测量，计算三维点的世界坐标。

---

### 3. 代码逐步解析

#### **第一步：构造第一个相机的投影矩阵**

`proj1 = np.hstack((np.eye(3), np.zeros((3, 1))))`

- **功能**： 构造第一个相机的外参矩阵：
    
    [R1∣t1]
    - I：单位矩阵，表示第一个相机没有旋转。
    - 0：零向量，表示第一个相机没有平移（默认设置为世界坐标系的参考相机）。
- **实现**：
    - 使用 `np.eye(3)` 生成 3×33 \times 33×3 的单位矩阵。
    - 使用 `np.zeros((3, 1))` 生成 3×13 \times 13×1 的零向量。
    - 使用 `np.hstack` 将矩阵和向量水平拼接。
- **结果**：
    $proj1 = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix}$

---

#### **第二步：构造第二个相机的投影矩阵**

`proj2 = np.hstack((R, t))`

- **功能**： 构造第二个相机的外参矩阵：
    
    [R2∣t2]
    - R：描述第二个相机相对于第一个相机的旋转。
    - t：描述第二个相机相对于第一个相机的平移。
- **实现**：
    - 使用 `np.hstack` 将 R 和 t 水平拼接。
- **输入示例**：
    
    - $R = \begin{bmatrix} 0.866 & -0.5 & 0 \\ 0.5 & 0.866 & 0 \\ 0 & 0 & 1 \end{bmatrix}$
    - $t = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}$
- **结果**：
    
    $proj2 = \begin{bmatrix} 0.866 & -0.5 & 0 & 1 \\ 0.5 & 0.866 & 0 & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix}$

---

#### **第三步：将内参矩阵应用于投影矩阵**

`proj1 = self.K @ proj1 proj2 = self.K @ proj2`

- **功能**： 将相机的内参矩阵 K 应用于外参矩阵，生成完整的投影矩阵：
    
    $P_1 = K \cdot [R_1 | t_1], \quad P_2 = K \cdot [R_2 | t_2]$
    
- **公式细节**：
    - 相机内参矩阵 KKK 的形式为： $K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$ 其中：
        - fx, fy​：相机的焦距（以像素为单位）。
        - cx, cy​：相机的主点坐标（图像中心）。
- **输入示例**：
    
    - 假设内参矩阵为： $K = \begin{bmatrix} 1000 & 0 & 500 \\ 0 & 1000 & 500 \\ 0 & 0 & 1 \end{bmatrix}$
- **结果**：
    
    - 第一个相机的投影矩阵： $P_1 = K \cdot [I | 0] = \begin{bmatrix} 1000 & 0 & 500 & 0 \\ 0 & 1000 & 500 & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix}$
    - 第二个相机的投影矩阵： $P_2 = K \cdot [R_2 | t_2] = \begin{bmatrix} 866 & -500 & 500 & 1000 \\ 500 & 866 & 500 & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix}$

---


```python
points_4d = cv2.triangulatePoints(proj1, proj2, src_pts[mask.ravel() == 1], dst_pts[mask.ravel() == 1])
points_3d = points_4d[:3] / points_4d[3]
sparse_points.append(points_3d.T)
```
point_4d = array, size(4, 1690)
point_3d = array, size(3, 1690)


这段代码的目的是使用 **三角测量（Triangulation）** 从两台相机的投影矩阵 P1P_1P1​、P2P_2P2​ 以及匹配的点对（来自两幅图像的像素坐标）计算出这些点在三维空间中的坐标。

#### **代码逻辑**

1. 使用 OpenCV 的 `cv2.triangulatePoints` 函数，通过两台相机的投影矩阵和匹配的点对，计算这些点在三维空间中的 **齐次坐标（Homogeneous Coordinates）**。
2. 将齐次坐标转化为普通三维坐标。
3. 将计算出的三维点存入稀疏点云列表 `sparse_points`。

以下详细解析每一行代码的作用，并用具体例子进行说明。

---

### 1. 背景知识

#### **三角测量的定义**

三角测量是通过多视角的几何关系（通常是两台相机）计算三维点的方法。假设有一组匹配点对：

- q1=[u1,v1,1]Tq_1 = [u_1, v_1, 1]^Tq1​=[u1​,v1​,1]T：相机1中的像素坐标。
- q2=[u2,v2,1]Tq_2 = [u_2, v_2, 1]^Tq2​=[u2​,v2​,1]T：相机2中的像素坐标。

相应的三维点 [X,Y,Z]T[X, Y, Z]^T[X,Y,Z]T 满足：

q1∝P1⋅[XYZ1],q2∝P2⋅[XYZ1]q_1 \propto P_1 \cdot \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}, \quad q_2 \propto P_2 \cdot \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}q1​∝P1​⋅​XYZ1​​,q2​∝P2​⋅​XYZ1​​

- P1P_1P1​：第一个相机的投影矩阵。
- P2P_2P2​：第二个相机的投影矩阵。

通过这两个方程，可以解出三维点的齐次坐标 [X,Y,Z,W]T[X, Y, Z, W]^T[X,Y,Z,W]T。

---

### 2. 代码逐步解析

#### **第一行：三角测量计算齐次坐标**

python

複製程式碼

`points_4d = cv2.triangulatePoints(proj1, proj2, src_pts[mask.ravel() == 1], dst_pts[mask.ravel() == 1])`

- **功能**： 使用 OpenCV 的 `cv2.triangulatePoints` 函数计算匹配点的三维齐次坐标。
    
- **输入参数**：
    
    1. **`proj1` 和 `proj2`**：
        - 两台相机的投影矩阵，分别描述从三维世界到各自图像像素平面的映射关系。
    2. **`src_pts[mask.ravel() == 1]` 和 `dst_pts[mask.ravel() == 1]`**：
        - 筛选出的有效匹配点（内点）。
        - `mask.ravel() == 1` 用于过滤 RANSAC 判断为内点的匹配点。
        - `src_pts` 和 `dst_pts` 是对应的像素坐标。
- **输出**：
    
    - **`points_4d`**：
        - 匹配点的三维齐次坐标，形状为 4×N4 \times N4×N： points4d=[X1X2…XNY1Y2…YNZ1Z2…ZNW1W2…WN]points_4d = \begin{bmatrix} X_1 & X_2 & \dots & X_N \\ Y_1 & Y_2 & \dots & Y_N \\ Z_1 & Z_2 & \dots & Z_N \\ W_1 & W_2 & \dots & W_N \end{bmatrix}points4​d=​X1​Y1​Z1​W1​​X2​Y2​Z2​W2​​…………​XN​YN​ZN​WN​​​

---

#### **第二行：从齐次坐标转化为三维坐标**

python

複製程式碼

`points_3d = points_4d[:3] / points_4d[3]`

- **功能**： 将三维点从齐次坐标 [X,Y,Z,W][X, Y, Z, W][X,Y,Z,W] 转换为普通的三维坐标 [X/W,Y/W,Z/W][X/W, Y/W, Z/W][X/W,Y/W,Z/W]。
    
- **公式**：
    
    [XYZ]=[X/WY/WZ/W]\begin{bmatrix} X \\ Y \\ Z \end{bmatrix} = \begin{bmatrix} X / W \\ Y / W \\ Z / W \end{bmatrix}​XYZ​​=​X/WY/WZ/W​​
- **结果**：
    
    - 输出形状为 3×N3 \times N3×N 的矩阵： points3d=[X1X2…XNY1Y2…YNZ1Z2…ZN]points_3d = \begin{bmatrix} X_1 & X_2 & \dots & X_N \\ Y_1 & Y_2 & \dots & Y_N \\ Z_1 & Z_2 & \dots & Z_N \end{bmatrix}points3​d=​X1​Y1​Z1​​X2​Y2​Z2​​………​XN​YN​ZN​​​

---

#### **第三行：保存三维点云**

python

複製程式碼

`sparse_points.append(points_3d.T)`

- **功能**： 将计算得到的三维点加入到稀疏点云列表中。
    
- **处理**：
    
    - `.T` 是转置操作，将点云的形状从 3×N3 \times N3×N 转换为 N×3N \times 3N×3，更适合存储和操作。
    - 将每次计算得到的点云结果累积到 `sparse_points` 列表中。
- **结果**：
    
    - `sparse_points` 是一个列表，每个元素是形状为 N×3N \times 3N×3 的三维点云。

---

### 3. 具体例子

#### 输入数据

1. **投影矩阵**：
    
    - 第一个相机： P1=[1000050000100050000010]P_1 = \begin{bmatrix} 1000 & 0 & 500 & 0 \\ 0 & 1000 & 500 & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix}P1​=​100000​010000​5005001​000​​
    - 第二个相机： P2=[866−500500100050086650000010]P_2 = \begin{bmatrix} 866 & -500 & 500 & 1000 \\ 500 & 866 & 500 & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix}P2​=​8665000​−5008660​5005001​100000​​
2. **匹配点对（经过 RANSAC 筛选的内点）**：
    
    - 相机1的像素坐标： src_pts=[[600,400],[650,420]]src\_pts = \begin{bmatrix} [600, 400], [650, 420] \end{bmatrix}src_pts=[[600,400],[650,420]​]
    - 相机2的像素坐标： dst_pts=[[610,410],[660,430]]dst\_pts = \begin{bmatrix} [610, 410], [660, 430] \end{bmatrix}dst_pts=[[610,410],[660,430]​]

#### 计算过程

##### **三角测量计算齐次坐标**

使用 `cv2.triangulatePoints` 得到：

points4d=[1.52.02.53.03.54.01.01.0]points_4d = \begin{bmatrix} 1.5 & 2.0 \\ 2.5 & 3.0 \\ 3.5 & 4.0 \\ 1.0 & 1.0 \end{bmatrix}points4​d=​1.52.53.51.0​2.03.04.01.0​​

##### **齐次坐标转普通三维坐标**

计算普通三维坐标：

points3d=[1.52.02.53.03.54.0][1.01.0]=[1.52.02.53.03.54.0]points_3d = \frac{\begin{bmatrix} 1.5 & 2.0 \\ 2.5 & 3.0 \\ 3.5 & 4.0 \end{bmatrix}}{\begin{bmatrix} 1.0 & 1.0 \end{bmatrix}} = \begin{bmatrix} 1.5 & 2.0 \\ 2.5 & 3.0 \\ 3.5 & 4.0 \end{bmatrix}points3​d=[1.0​1.0​]​1.52.53.5​2.03.04.0​​​=​1.52.53.5​2.03.04.0​​

##### **存储点云**

转置并保存：

points3d.T=[1.52.53.52.03.04.0]points_3d.T = \begin{bmatrix} 1.5 & 2.5 & 3.5 \\ 2.0 & 3.0 & 4.0 \end{bmatrix}points3​d.T=[1.52.0​2.53.0​3.54.0​]

存入 `sparse_points` 列表。

---

### 4. 总结

#### **核心流程**

1. **三角测量计算齐次坐标**：
    - 使用投影矩阵和匹配点对，通过 `cv2.triangulatePoints` 计算三维点的齐次坐标。
2. **齐次坐标转普通三维坐标**：
    - 将齐次坐标归一化，得到普通的三维坐标。
3. **存储三维点云**：
    - 转置坐标矩阵并保存到点云列表中。

#### **输出结果**

- 稀疏点云列表 `sparse_points`，每个元素是形状为 N×3N \times 3N×3 的三维点。

#### **应用场景**

- **立体视觉**：恢复三维场景结构。
- **三维重建**：构建稀疏点云，用于后续的密集点云生成。

这段代码是多视角几何中的核心步骤，为三维重建奠定了基础。

```python
for i in range(len(self.matches)):
   ...
   sparse_points.append(points_3d.T)

sparse_points = np.vstack(sparse_points)
```
sparse_points = list = [array(size=3,1690), array(size=3,1293), array(size=3,1060), ..]
--> 
sparse_points = ndarray (size=4198,3)


```python
# OpenMVS generate dense 3D point cloud
DensifyPointCloud /path/to/sparse.mvs --resolution-level 2
```

### OpenMVS `DensifyPointCloud` 构建稠密 3D 点云的详细流程和原理

`DensifyPointCloud` 是 OpenMVS 框架中用于生成稠密 3D 点云的核心模块。它基于多视图立体（Multi-View Stereo, MVS）技术，通过多张图像及其相机姿态和稀疏点云信息，计算出场景中的高密度三维点云。

---

| OMVS                            |                                            |
| ------------------------------- | ------------------------------------------ |
| 1. openMVG_main_openMVG2openMVS | Convert SfM scene<br>轉換 SfM 場景             |
| 2. DensifyPointCloud            | Dense Point-Cloud Reconstruction<br>密集點雲重建 |
| 3. ReconstructMesh              | Rough Mesh Reconstruction <br>粗糙網格重建       |
| 4. RefineMesh                   | Mesh Refinement <br>網格細化                   |
| 5. TextureMesh                  | Mesh Texturing <br>網格紋理                    |

### **1. 核心原理**

#### **1.1 多视图几何与稠密点云**

稠密点云生成的核心思想是基于多视图几何，将从多个视角拍摄的图像进行深度估计和融合。其基本步骤如下：

- **深度图估计**：从每张图像的视角估算场景中每个像素的深度（即其在三维空间中的位置）。
- **多视图一致性**：通过不同图像视角验证每个像素的深度值，从而去除不一致的或错误的估计。
- **深度图融合**：将多视图的深度图合并为稠密点云。

---

### **2. `DensifyPointCloud` 的详细流程**

以下是 OpenMVS `DensifyPointCloud` 的主要步骤及其实现原理。

#### **2.1 输入数据准备**

- **输入要求**：
    1. 稀疏点云（通常由 SfM 工具生成，如 COLMAP）。
    2. 图像数据（多张视角重叠的图像）。
    3. 每张图像的相机参数和姿态（内参、外参）。
- **典型输入文件**：
    - 稀疏点云（如 `sparse.mvs` 文件）。
    - 原始图像及其标定信息。

---

#### **2.2 深度图估计**

- **步骤**：
    - 对每张图像，基于其相机参数和图像序列，生成深度图。
- **算法**：
    1. **代价体积构建（Cost Volume Construction）**：
        
        - 使用匹配代价函数（如 NCC 或 SGM）构建一个 3D 空间的代价体积。
        - 每个像素的深度候选值对应于一个代价。
        
        **公式**： C(p,d)=Cost(Iref(p),Isrc(π(d)))C(p, d) = \text{Cost}(I_{ref}(p), I_{src}(\pi(d)))C(p,d)=Cost(Iref​(p),Isrc​(π(d))) 其中：
        - ppp：参考图像中的像素。
        - ddd：深度假设值。
        - π(d)\pi(d)π(d)：投影到源图像的像素位置。
        - Cost\text{Cost}Cost：匹配代价函数。
    2. **深度优化（Depth Optimization）**：
        
        - 通过全局优化（如基于图割的优化）选择代价最小的深度值。
        
        **示例**：
        - 假设图像分辨率为 1000×10001000 \times 10001000×1000，对每个像素计算深度范围为 1∼101 \sim 101∼10 的代价值，选择最优深度值。
- **输出**： 每张图像生成一个深度图。

---

#### **2.3 多视图一致性检查**

- **目的**： 验证深度值在不同视角中的一致性，剔除错误匹配和噪声。
    
- **方法**：
    
    - 利用视差验证（Photometric Consistency）检查相邻图像间的深度投影是否一致。
    - 例如，参考图像的像素点 ppp 和源图像的点 π(d)\pi(d)π(d) 应该具有相似的颜色或特征。

**公式**：

Consistency(d)=∥Iref(p)−Isrc(π(d))∥\text{Consistency}(d) = \| I_{ref}(p) - I_{src}(\pi(d)) \|Consistency(d)=∥Iref​(p)−Isrc​(π(d))∥

- 如果 Consistency(d)\text{Consistency}(d)Consistency(d) 超过某个阈值，则认为深度值 ddd 不一致。

---

#### **2.4 深度图融合**

- **目的**： 将所有图像的深度图合并，生成全局稠密点云。
    
- **方法**：
    
    - 对于每个深度值，将其投影到全局 3D 空间。
    - 去除冗余和不一致点（如不同视角的深度冲突点）。
- **结果**：
    
    - 输出稠密点云，通常为 `.ply` 格式。

---

#### **2.5 输出稠密点云**

- `DensifyPointCloud` 的最终输出是稠密点云文件。
- 文件格式：
    - `.mvs` 文件：用于后续表面重建。
    - `.ply` 文件：标准点云格式，可用于可视化和分析。

---

### **3. 使用流程示例**

#### **输入准备**

假设已通过 SfM 生成稀疏点云 `sparse.mvs` 和相关图像。

#### **运行命令**

bash

複製程式碼

`DensifyPointCloud sparse.mvs --resolution-level 2 --output-file dense.mvs`

- 参数说明：
    - `sparse.mvs`：输入稀疏点云。
    - `--resolution-level 2`：降低分辨率以加快处理速度。
    - `--output-file dense.mvs`：输出稠密点云文件。

---

### **4. 示例**

#### **输入数据**

1. **稀疏点云**：
    - 100 个稀疏 3D 点，表示基本场景结构。
2. **图像**：
    - 10 张图像，分辨率 1920×10801920 \times 10801920×1080。

#### **生成深度图**

- 假设对图像 1 生成深度图，某像素深度值为 d=5d = 5d=5，表示其在世界坐标系的深度为 5 个单位。

#### **多视图一致性**

- 深度值 d=5d = 5d=5 在图像 2 和图像 3 中投影位置一致，验证通过。

#### **融合稠密点云**

- 合并所有图像的深度图，生成 100 万个点的稠密点云，表示完整场景。

---

### **5. 优缺点分析**

#### **优点**

1. **高精度**：
    - 基于多视图几何和一致性验证，生成高质量的点云。
2. **自动化**：
    - 自动处理多张图像，无需手动调整。
3. **灵活性**：
    - 支持不同分辨率和优化参数。

#### **缺点**

1. **计算复杂度高**：
    - 深度估计和一致性验证需要大量计算。
2. **对纹理依赖强**：
    - 对无纹理区域（如平坦表面）可能生成误差。
3. **硬件要求高**：
    - 需要较高的内存和 GPU 性能。

---

### **6. 总结**

OpenMVS `DensifyPointCloud` 是一个强大的稠密点云生成工具，其核心包括：

1. 深度图估计。
2. 多视图一致性验证。
3. 深度图融合。

通过这些步骤，它能够从多视角图像和稀疏点云生成高质量的稠密点云，为后续表面重建和纹理化奠定基础。对于应用场景，如无人驾驶、3D 建模和场景重建，`DensifyPointCloud` 提供了一种高效且鲁棒的解决方案。



```python
# OpenMVS surface reconstruction
ReconstructMesh /path/to/dense.mvs
```