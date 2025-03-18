
### **傳統機器學習與統計分析問題**

21. 請比較隨機森林（Random Forest）與梯度提升（Boosting）的核心差異與應用場景。
22. 如何利用 SVM 進行醫學影像中腫瘤的分類？
23. 在處理高維影像數據時，主成分分析（PCA）如何幫助降維？
24. 請說明您對 K-means 聚類在醫學影像中應用的理解與挑戰。
25. 如何將特徵選擇與深度學習結合來提高模型性能？
26. 對於醫學影像中的小數據集，如何利用統計方法進行樣本增強？
27. 在生物醫學影像分析中，什麼時候應該選擇傳統機器學習算法而非深度學習？
28. 如何驗證影像分析模型的穩健性和泛化能力？
29. 面對不平衡數據集，如何利用 SMOTE 或其他技術進行處理？
30. 什麼是信息增益？如何在特徵選擇中應用？


### **問題 21：請比較隨機森林（Random Forest）與梯度提升（Boosting）的核心差異與應用場景。**

#### **回答結構：**

1. **基本概念**
2. **核心差異**
3. **優缺點比較**
4. **應用場景**
5. **案例分析**

---

#### **1. 基本概念**

- **隨機森林（Random Forest, RF）：**
    
    - 基於多個決策樹的集合學習算法，通過 Bagging 技術（Bootstrap Aggregating）將多個樹的輸出進行平均或投票。
- **梯度提升（Gradient Boosting, GB）：**
    
    - 基於弱學習器（如決策樹），逐步減少模型的損失，使用加權集成（Boosting）提升模型性能。

---

#### **2. 核心差異**

|特性|隨機森林（RF）|梯度提升（GB）|
|---|---|---|
|**構建方式**|並行訓練多個樹|逐步構建樹，每棵樹依賴於上一棵|
|**目標**|減少方差（Variance Reduction）|減少偏差（Bias Reduction）|
|**模型複雜度**|較低，易於調參|較高，調參複雜|
|**訓練效率**|較快，可並行|較慢，需逐步迭代|

---

#### **3. 優缺點比較**

- **隨機森林：**
    
    - **優點：**
        - 對噪聲和過擬合不敏感。
        - 可並行訓練，計算速度快。
    - **缺點：**
        - 對非線性關係或複雜特徵的處理能力有限。
- **梯度提升：**
    
    - **優點：**
        - 能有效降低偏差，對少量數據或噪聲表現優異。
        - 可處理複雜非線性關係。
    - **缺點：**
        - 訓練速度慢，需仔細調參。

---

#### **4. 應用場景**

- **隨機森林：**
    - 用於需要快速訓練的大數據集，適合分類任務，如影像分類或多模態數據融合。
- **梯度提升：**
    - 用於需要高精度的應用，如醫學數據預測、癌症分期分類。

---

#### **5. 案例分析**

在影像分類任務中，使用隨機森林對手寫數字進行分類，準確率達到 **96%**，訓練時間僅需 **10 分鐘**。在腫瘤預測任務中，使用梯度提升（如 XGBoost），準確率達到 **98%**，但訓練時間較長。

**代碼示例：隨機森林**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
accuracy = rf.score(X_test, y_test)

```

這些回答詳細解釋了技術原理、選擇原則和應用場景，結合代碼示例展現實踐能力。

### **問題 22：如何利用 SVM 進行醫學影像中腫瘤的分類？**

#### **回答結構：**

1. **SVM 的基本概念**
2. **SVM 在腫瘤分類中的應用流程**
3. **特徵選擇與處理**
4. **優化與調參**
5. **案例分析與代碼示例**

---

#### **1. SVM 的基本概念**

**支持向量機（Support Vector Machine, SVM）** 是一種監督學習算法，目的是找到一個超平面（Hyperplane），將不同類別的數據進行最大間隔分割。

- **核心思想：**
    - 最大化類別之間的邊界（Margin）。
    - 支持核方法（Kernel Methods），如線性核（Linear Kernel）、高斯核（RBF Kernel），能處理線性不可分問題。

---

#### **2. SVM 在腫瘤分類中的應用流程**

1. **數據預處理：**
    
    - 讀取醫學影像（如 MRI、CT），進行圖像增強（Image Enhancement）。
    - 將影像轉換為特徵向量（如紋理、形狀、密度特徵）。
2. **特徵提取：**
    
    - 使用 GLCM（灰度共生矩陣）或 HOG（方向梯度直方圖）提取紋理特徵。
    - 或應用深度學習模型（如 ResNet）提取高層特徵。
3. **模型訓練：**
    
    - 使用 SVM 訓練腫瘤分類模型，將數據分為良性和惡性。
4. **模型評估：**
    
    - 通過交叉驗證（Cross Validation）計算準確率、靈敏度（Sensitivity）和特異性（Specificity）。

---

#### **3. 特徵選擇與處理**

- **標準化：** 使用 Z-score 將特徵歸一化：
    
    $\large x' = \frac{x - \mu}{\sigma}$
- **降維：** 若特徵數量過多，使用 PCA（主成分分析）提取關鍵特徵，減少維度。
    

---

#### **4. 優化與調參**

- **核函數選擇：**
    - **線性核：** 適合線性可分問題。
    - **高斯核（RBF Kernel）：** 處理非線性問題效果良好。
- **正則化參數 CCC：** 控制邊界的寬度，調整模型對誤分類的容忍度。

---

#### **5. 案例分析與代碼示例**

**案例：** 在乳腺癌數據集（Breast Cancer Dataset）中，使用 MRI 圖像提取紋理特徵，應用 SVM 區分良性和惡性腫瘤，準確率達到 **94%**。

**代碼示例：**
```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 數據預處理
X = extract_features(images)  # 自定義特徵提取函數
y = labels  # 腫瘤標籤
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 訓練 SVM
model = SVC(kernel='rbf', C=1.0)
model.fit(X_train, y_train)

# 評估
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

```

---

### **問題 23：在處理高維影像數據時，主成分分析（PCA）如何幫助降維？**

#### **回答結構：**

1. **PCA 的基本概念**
2. **PCA 的數學原理**
3. **PCA 在影像數據降維中的作用**
4. **案例分析與代碼示例**

---

#### **1. PCA 的基本概念**

**主成分分析（Principal Component Analysis, PCA）** 是一種降維技術，通過將高維數據轉換為低維數據，同時保留最大的信息量。

- **核心目標：**
    - 簡化數據表示。
    - 減少維度，降低計算負擔。

---

#### **2. PCA 的數學原理**

1. **數據中心化：** 將數據均值歸零：
    
    $\large X' = X - \mu$
2. **計算協方差矩陣：**
    
    $\large \Sigma = \frac{1}{n} X'^\top X'$
3. **特徵分解：** 計算協方差矩陣的特徵值與特徵向量。
    
4. **選擇主成分：** 根據特徵值排序，選擇前 k個最大特徵值對應的特徵向量作為主成分。
    

---

#### **3. PCA 在影像數據降維中的作用**

1. **降噪：**
    - 去除小特徵值對應的噪聲成分。
2. **降維：**
    - 在醫學影像中，將高分辨率圖像壓縮到低維特徵空間，提高計算效率。
3. **可視化：**
    - 將高維數據映射到 2D 或 3D 空間，便於觀察。

---

#### **4. 案例分析與代碼示例**

**案例：** 在 CT 圖像分析中，使用 PCA 將影像的像素矩陣降至 50 維特徵，準確保留 **95%** 的原始數據信息，訓練分類模型的時間縮短 **40%**。

**代碼示例：**
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 數據標準化
X = preprocess_images(images)  # 提取影像特徵
X = StandardScaler().fit_transform(X)

# PCA 降維
pca = PCA(n_components=50)
X_reduced = pca.fit_transform(X)

# 查看保留的方差比例
print(pca.explained_variance_ratio_)

```

---

### **問題 24：請說明您對 K-means 聚類在醫學影像中應用的理解與挑戰。**

#### **回答結構：**

1. **K-means 聚類的基本概念**
2. **K-means 在醫學影像中的應用**
3. **K-means 的挑戰與改進**
4. **案例分析與代碼示例**

---

#### **1. K-means 聚類的基本概念**

**K-means 聚類** 是一種無監督學習算法，通過將數據分為 K 個聚類（Clusters），最小化每個數據點到其聚類中心的距離平方和。

- **目標函數：** $\large J = \sum_{i=1}^K \sum_{x \in C_i} ||x - \mu_i||^2$ 其中 μi 是聚類中心。

---

#### **2. K-means 在醫學影像中的應用**

1. **影像分割（Image Segmentation）：**
    - 將影像中的像素分為不同區域（如病變區和健康區）。
2. **病變檢測（Lesion Detection）：**
    - 在 MRI 或 CT 影像中分離異常區域。
3. **組織分類：**
    - 將病理影像中的細胞或組織分為不同的類型。

---

#### **3. K-means 的挑戰與改進**

- **挑戰：**
    
    1. **初始聚類中心敏感：** 初始中心選擇可能影響結果，導致局部最優。
    2. **K 值選擇困難：** 無法自動確定最佳聚類數 KKK。
    3. **數據特徵分布要求：** 假設聚類是球形且均勻分佈，對非均勻數據效果較差。
- **改進方法：**
    
    1. 使用 **K-means++** 自動選擇初始聚類中心。
    2. 結合層次聚類（Hierarchical Clustering）確定 KKK 值。
    3. 在高維數據上結合 PCA 或 t-SNE 進行降維後聚類。

---

#### **4. 案例分析與代碼示例**

**案例：** 在腦部 MRI 圖像中，使用 K-means 聚類分割腫瘤區域，分割準確率達到 **85%**，使用 K-means++ 改進初始中心，提高穩定性。

**代碼示例：**
```python
from sklearn.cluster import KMeans
import cv2
import numpy as np

# 讀取影像
image = cv2.imread('brain_mri.jpg', 0)
pixels = image.reshape(-1, 1)

# K-means 聚類
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
kmeans.fit(pixels)
segmented_image = kmeans.labels_.reshape(image.shape)

# 顯示分割結果
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)

```

---

這些回答詳細解釋了技術原理、應用場景、挑戰與改進，並結合具體代碼實現提供實踐支持。


### **問題 25：如何將特徵選擇與深度學習結合來提高模型性能？**

#### **回答結構：**

1. **特徵選擇的概念與重要性**
2. **特徵選擇在深度學習中的角色**
3. **結合特徵選擇與深度學習的方法**
4. **案例分析與代碼示例**

---

#### **1. 特徵選擇的概念與重要性**

**特徵選擇（Feature Selection）** 是從數據集中選擇最相關的特徵，去除冗餘或無關特徵，以提升模型性能。

- **目標：**
    - 減少計算負擔。
    - 降低過擬合風險。
    - 提高解釋性。

---

#### **2. 特徵選擇在深度學習中的角色**

- **數據降維：** 從高維輸入中提取關鍵特徵，降低輸入層的維度。
- **訓練加速：** 減少不必要特徵的干擾，提升訓練效率。
- **模型精度提升：** 聚焦於有意義的特徵，有助於模型學習更有效的模式。

---

#### **3. 結合特徵選擇與深度學習的方法**

1. **基於統計的方法（Filter Methods）：**
    
    - 使用卡方檢驗（Chi-Square）、互信息（Mutual Information）篩選特徵，再將選擇的特徵輸入深度模型。
    - 示例：選擇腫瘤影像中的紋理和形狀特徵。
2. **基於模型的方法（Wrapper Methods）：**
    
    - 使用機器學習算法（如隨機森林、Lasso 回歸）評估特徵的重要性。
    - 示例：用隨機森林計算影像特徵的重要性，過濾低權重特徵。
3. **嵌入式方法（Embedded Methods）：**
    
    - 使用深度模型本身的注意力機制（Attention Mechanism）或權重正則化（如 L1 正則化）進行特徵選擇。
    - 示例：在卷積神經網絡中加入注意力層，篩選出關鍵區域。
4. **降維技術輔助（如 PCA、t-SNE）：**
    
    - 通過降維技術將高維特徵映射到低維空間，減少輸入特徵數量。

---

#### **4. 案例分析與代碼示例**

**案例：** 在醫學影像分類任務中，結合統計特徵選擇和深度學習。首先使用互信息篩選腫瘤影像的紋理特徵，然後輸入到 CNN 中，準確率從 **85%** 提升到 **91%**。

**代碼示例：**
```python
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
import numpy as np

# 特徵選擇
X_selected = mutual_info_classif(X, y) > 0.1  # 篩選互信息得分高的特徵
X_filtered = X[:, X_selected]

# 深度學習模型
import torch.nn as nn
class MyModel(nn.Module):
    def __init__(self, input_size):
        super(MyModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

```

---

### **問題 26：對於醫學影像中的小數據集，如何利用統計方法進行樣本增強？**

#### **回答結構：**

1. **小數據集的挑戰**
2. **統計方法進行樣本增強的技術**
3. **應用場景**
4. **案例分析與代碼示例**

---

#### **1. 小數據集的挑戰**

- **數據不足：** 醫學影像數據通常昂貴且難以獲取。
- **模型過擬合：** 小數據集導致模型難以泛化。
- **數據不均衡：** 某些病變樣本數量稀少。

---

#### **2. 統計方法進行樣本增強的技術**

1. **圖像插值（Image Interpolation）：**
    
    - 通過平滑插值技術生成新的樣本。
    - 示例：從 MRI 切片中生成不同層面的影像。
2. **數據合成（Data Synthesis）：**
    
    - 使用生成模型（如高斯模型）基於已有數據分布生成新數據。
    - 示例：在 CT 數據中模擬不同灰度分布的肺結節。
3. **隨機擾動（Random Perturbation）：**
    
    - 給現有數據添加隨機噪聲。
    - 示例：添加高斯噪聲模擬低劑量成像。
4. **數據平衡（SMOTE，Synthetic Minority Oversampling Technique）：**
    
    - 通過插值生成少數類樣本。
    - 示例：平衡良性和惡性腫瘤樣本數量。

---

#### **3. 應用場景**

- **稀有病變檢測：** 增強罕見腫瘤的數據樣本。
- **多模態融合：** 通過合成新樣本模擬不同成像設備的數據。

---

#### **4. 案例分析與代碼示例**

**案例：** 在 CT 影像中使用高斯噪聲增強，數據增強後模型在測試集的靈敏度從 **78%** 提升到 **85%**。

**代碼示例：**
```python
import numpy as np
import cv2

# 添加高斯噪聲
def add_gaussian_noise(image):
    row, col, _ = image.shape
    mean = 0
    sigma = 10
    gauss = np.random.normal(mean, sigma, (row, col, 1)).astype(np.uint8)
    noisy_image = cv2.add(image, gauss)
    return noisy_image

# 應用數據增強
image_augmented = add_gaussian_noise(image)

```

---

### **問題 27：在生物醫學影像分析中，什麼時候應該選擇傳統機器學習算法而非深度學習？**

#### **回答結構：**

1. **傳統機器學習與深度學習的區別**
2. **適合選擇傳統機器學習的場景**
3. **具體應用舉例**
4. **優缺點分析**

---

#### **1. 傳統機器學習與深度學習的區別**

- **傳統機器學習（如隨機森林、SVM）：**
    - 通常需要手工提取特徵（Feature Engineering）。
    - 適合小數據集或低計算資源場景。
- **深度學習（如 CNN、Transformer）：**
    - 能自動學習特徵，但需要大量標註數據和高計算資源。

---

#### **2. 適合選擇傳統機器學習的場景**

1. **小數據集：**
    
    - 當數據量不足時，傳統機器學習能避免深度學習的過擬合問題。
    - 示例：僅有數百張 MRI 影像的分類。
2. **特徵已知的情況：**
    
    - 當某些特徵（如紋理、形狀）已被證明對分類有效時，傳統方法更加高效。
3. **計算資源有限：**
    
    - 當硬件條件限制 GPU 使用時，傳統機器學習更加輕量化。
4. **解釋性需求高：**
    
    - 傳統機器學習（如決策樹）更易解釋，適合醫學場景。

---

#### **3. 具體應用舉例**

- **病變分類：** 使用 SVM 分類乳腺癌病變（良性/惡性）。
- **腫瘤檢測：** 使用隨機森林識別腦部 MRI 影像中的腫瘤。
- **影像配準：** 使用 KNN 或傳統迭代方法進行多模態影像配準。

---

#### **4. 優缺點分析**

|**特點**|**傳統機器學習**|**深度學習**|
|---|---|---|
|**數據需求**|小數據集|大規模數據集|
|**計算資源**|較低|高（需 GPU）|
|**特徵工程**|需要手工提取特徵|自動學習特徵|
|**解釋性**|高|較低|

---

#### **5. 案例分析與代碼示例**

**案例：** 在乳腺癌影像分類中，使用 SVM 和紋理特徵，準確率達到 **90%**，而 CNN 在小數據集下過擬合嚴重，無法超過 **85%**。

**代碼示例：傳統機器學習**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 提取特徵
X = extract_features(images)
y = labels

# 分割數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 訓練模型
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 測試與評估
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

```

---

這些回答詳細說明了技術選擇、應用場景和代碼實現，結合案例展現實踐經驗與技術深度。

### **問題 28：如何驗證影像分析模型的穩健性和泛化能力？**

#### **回答結構：**

1. **穩健性與泛化能力的定義**
2. **驗證模型穩健性的方法**
3. **驗證模型泛化能力的方法**
4. **案例分析與實現**

---

#### **1. 穩健性與泛化能力的定義**

- **穩健性（Robustness）：** 模型在噪聲、失真或其他非預期輸入條件下保持性能穩定的能力。
    
- **泛化能力（Generalization Ability）：** 模型在未見數據（測試集）上的表現，反映模型是否過度擬合訓練數據。
    

---

#### **2. 驗證模型穩健性的方法**

1. **加噪測試（Noise Testing）：**
    
    - 在輸入影像中添加高斯噪聲、椒鹽噪聲或其他干擾，觀察模型性能是否顯著下降。
2. **圖像失真測試（Image Distortion Testing）：**
    
    - 模擬實際應用中的成像變化，如光線、對比度變化、旋轉、模糊等。
3. **對抗攻擊測試（Adversarial Attack Testing）：**
    
    - 使用對抗樣本（Adversarial Examples）測試模型，例如 FGSM（Fast Gradient Sign Method），檢查模型是否易受攻擊。
4. **跨設備測試（Cross-Device Testing）：**
    
    - 使用不同成像設備的數據測試模型性能。

---

#### **3. 驗證模型泛化能力的方法**

1. **交叉驗證（Cross-Validation）：**
    
    - 將數據集分為 k 份，進行 k-折交叉驗證（K-Fold Cross-Validation），平均結果。
2. **測試集評估（Test Set Evaluation）：**
    
    - 使用完全獨立的測試集驗證性能，避免信息洩露。
3. **分布外測試（Out-of-Distribution Testing）：**
    
    - 測試模型在分布不同的數據（如不同地區、不同設備）上的表現。
4. **學習曲線分析（Learning Curve Analysis）：**
    
    - 檢查訓練損失與驗證損失的趨勢，判斷過擬合或欠擬合。

---

#### **4. 案例分析與實現**

**案例：** 在放射學影像分類任務中，對模型添加隨機噪聲測試穩健性，並使用另一家醫院的數據集進行分布外測試。結果顯示噪聲標準差增加到 **0.05** 時，準確率從 **91%** 降至 **85%**，模型對分布外數據的準確率為 **88%**，表現出良好的泛化能力。

**代碼示例：**
```python
import numpy as np

# 加噪測試
def add_noise(image, stddev=0.05):
    noise = np.random.normal(0, stddev, image.shape)
    return np.clip(image + noise, 0, 1)

# 模型評估
noisy_images = [add_noise(img) for img in test_images]
accuracy = model.evaluate(noisy_images, test_labels)
print("Accuracy with noise:", accuracy)

```

---

### **問題 29：面對不平衡數據集，如何利用 SMOTE 或其他技術進行處理？**

#### **回答結構：**

1. **數據不平衡的問題與影響**
2. **SMOTE 的基本原理**
3. **其他處理技術**
4. **案例分析與代碼示例**

---

#### **1. 數據不平衡的問題與影響**

- **問題：** 在醫學影像中，患病樣本數量遠少於健康樣本，導致模型更傾向於預測多數類，忽視少數類。
    
- **影響：**
    
    - 評估指標（如準確率）可能失真。
    - 少數類的檢測性能低。

---

#### **2. SMOTE 的基本原理**

**SMOTE（Synthetic Minority Oversampling Technique）** 是一種過採樣方法，通過在少數類樣本間進行插值生成新樣本。

- **步驟：**
    1. 隨機選擇一個少數類樣本 x。
    2. 在 xxx 的 k 最近鄰樣本中隨機選取一個樣本 $\large x_{nn}$
    3. 通過插值生成新樣本： $\large x_{\text{new}} = x + \lambda \cdot (x_{nn} - x), \quad \lambda \in [0, 1]$

---

#### **3. 其他處理技術**

1. **欠採樣（Under-sampling）：**
    
    - 隨機刪除多數類樣本，使數據平衡。
2. **加權損失函數（Weighted Loss Function）：**
    
    - 給少數類樣本分配更高的損失權重，例如 Focal Loss。
3. **數據增強（Data Augmentation）：**
    
    - 使用旋轉、翻轉等技術生成少數類樣本的新版本。

---

#### **4. 案例分析與代碼示例**

**案例：** 在肺癌 X 光分類任務中，健康樣本與腫瘤樣本比例為 10:1，通過 SMOTE 將比例調整為 1:1，分類準確率從 **80%** 提升到 **88%**。

**代碼示例：**
```python
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# SMOTE 過採樣
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 訓練測試分割
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2)

```

---

### **問題 30：什麼是信息增益？如何在特徵選擇中應用？**

#### **回答結構：**

1. **信息增益的概念**
2. **信息增益的公式與計算**
3. **在特徵選擇中的應用步驟**
4. **案例分析與代碼示例**

---

#### **1. 信息增益的概念**

**信息增益（Information Gain, IG）** 是衡量某個特徵對於目標分類的不確定性減少的程度。高信息增益的特徵對分類具有更大的區分能力。

- **核心思想：** 減少信息熵（Entropy），提高系統確定性。

---

#### **2. 信息增益的公式與計算**

1. **信息熵（Entropy）：**
    
    $\large H(S) = -\sum_{i=1}^n p_i \log_2 p_i$
    
    pi​ 是第 i 類的概率。
    
2. **信息增益：**
    
    $\large IG(T, A) = H(T) - \sum_{v \in A} \frac{|T_v|}{|T|} H(T_v)$
    - H(T)：整體數據集的熵。
    - H(Tv)：特徵 A 按值分割後子集的熵。

---

#### **3. 在特徵選擇中的應用步驟**

1. **計算每個特徵的 IG：** 對所有特徵計算 IG，量化其對目標變量的影響力。
    
2. **篩選高信息增益特徵：** 設定閾值，保留信息增益高於閾值的特徵。
    
3. **應用於模型訓練：** 使用篩選後的特徵進行模型訓練，降低維度，提高性能。
    

---

#### **4. 案例分析與代碼示例**

**案例：** 在腫瘤影像分類任務中，對紋理特徵計算信息增益，選擇 IG 高於 0.2 的特徵進行模型訓練，準確率提升 **5%**。

**代碼示例：**
```python
from sklearn.feature_selection import mutual_info_classif

# 計算信息增益
info_gain = mutual_info_classif(X, y)

# 篩選高 IG 特徵
selected_features = X[:, info_gain > 0.2]

# 構建模型
model.fit(selected_features, y)

```

---

這些回答詳細解釋了技術原理、應用場景和代碼實現，結合實例展示方法的效果和實踐可行性。
