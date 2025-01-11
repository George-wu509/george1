
|                                                                                                                                                                  |                                                                                                                                                                                                                                                                               |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Classification                                                                                                                                                   | 1. [==K-nearest neighbors(KNN)==](https://medium.com/programming-with-data/k-%E8%BF%91%E9%84%B0%E6%BC%94%E7%AE%97%E6%B3%95-k-nearest-neighbors-72b6cec68367), 2. [==SVM==](https://zhuanlan.zhihu.com/p/696741013), 3. Decision Tree, 4. Naive Bayes, 5. Logistic Regression, |
| Cluster                                                                                                                                                          | 1. [==K-means==](https://chih-sheng-huang821.medium.com/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E9%9B%86%E7%BE%A4%E5%88%86%E6%9E%90-k-means-clustering-e608a7fe1b43), 2. Hierarchical Clustering                                                                                |
| Regression                                                                                                                                                       | 1. Linear Regression, 2. XGBoost, 3. MLP                                                                                                                                                                                                                                      |
| Dimension                                                                                                                                                        | 1. PCA                                                                                                                                                                                                                                                                        |
| Optimization                                                                                                                                                     | 1. SGD                                                                                                                                                                                                                                                                        |
| [Ensemble](https://chih-sheng-huang821.medium.com/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-ensemble-learning%E4%B9%8Bbagging-boosting%E5%92%8Cadaboost-af031229ebc3) | 1. Bagging、2. Random Forest, 3. Boosting, 4. AdaBoost                                                                                                                                                                                                                         |
|                                                                                                                                                                  |                                                                                                                                                                                                                                                                               |

Ref: [機器學習: Ensemble learning之Bagging、Boosting和AdaBoost](https://chih-sheng-huang821.medium.com/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-ensemble-learning%E4%B9%8Bbagging-boosting%E5%92%8Cadaboost-af031229ebc3)


以下是機器學習中最常用或在面試中經常出現的 10 種算法的詳細解釋，並提供了對應的 MATLAB 代碼示例。這些算法涵蓋監督學習、無監督學習和增強學習。

### 1. 線性回歸（Linear Regression）

**應用**：線性回歸用於建模目標變量和一個或多個輸入特徵之間的線性關係。這是最簡單的回歸算法之一。

**原理**：假設目標變量 yyy 和輸入特徵 XXX 之間呈線性關係，即 y=wX+by = wX + by=wX+b，其中 www 是權重，bbb 是偏置項。

**MATLAB 範例代碼**：

matlab

複製程式碼

`% 線性回歸模型 x = (1:10)'; y = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10];  % 構建線性回歸模型 mdl = fitlm(x, y);  % 預測 y_pred = predict(mdl, x); plot(x, y, 'o', x, y_pred, '-'); title('線性回歸擬合');`

### 2. 邏輯回歸（Logistic Regression）

**應用**：邏輯回歸主要用於二分類問題，例如癌症檢測（陽性/陰性）。

**原理**：邏輯回歸是一種分類算法，它預測目標變量的概率，通常用 sigmoid 函數將線性回歸轉換為二分類模型。

**MATLAB 範例代碼**：

matlab

複製程式碼

`% 邏輯回歸模型 X = [1, 2; 2, 3; 3, 4; 4, 5; 5, 6]; % 特徵矩陣 Y = [0; 0; 1; 1; 1]; % 標籤  % 構建邏輯回歸模型 mdl = fitglm(X, Y, 'Distribution', 'binomial');  % 預測 Y_pred = predict(mdl, X) > 0.5;  % 分類結果`

### 3. K-近鄰算法（K-Nearest Neighbors, KNN）

**應用**：KNN 用於分類和回歸，特別適合模式識別和異常檢測。

**原理**：KNN 是一種非參數算法，它基於距離度量，通過查找最近的 K 個鄰居來進行分類或回歸。

**MATLAB 範例代碼**：

matlab

複製程式碼

`% KNN 分類器 X = [1,2; 2,3; 3,3; 4,5; 5,6];  % 特徵 Y = [0; 0; 1; 1; 1];             % 標籤 mdl = fitcknn(X, Y, 'NumNeighbors', 3);  % 預測 X_new = [3, 4]; label = predict(mdl, X_new);`

### 4. 支持向量機（Support Vector Machine, SVM）

**應用**：SVM 常用於圖像分類、文本分類等二分類問題，並且也可擴展到多分類。

**原理**：SVM 尋找一個最大化類別間距的超平面來區分不同的類別，並且可以通過核函數應對非線性分類。

**MATLAB 範例代碼**：

matlab

複製程式碼

`% SVM 分類器 X = [1,2; 2,3; 3,3; 4,5; 5,6];  % 特徵 Y = [0; 0; 1; 1; 1];             % 標籤 mdl = fitcsvm(X, Y);  % 預測 X_new = [3, 4]; label = predict(mdl, X_new);`

### 5. 決策樹（Decision Tree）

**應用**：決策樹用於分類和回歸，常見於金融風險評估、醫學診斷等領域。

**原理**：決策樹是一種基於特徵的樹狀結構算法，通過選擇信息增益（或基尼不純度）最大的特徵進行劃分。

**MATLAB 範例代碼**：

matlab

複製程式碼

`% 決策樹分類器 X = [1,2; 2,3; 3,3; 4,5; 5,6];  % 特徵 Y = [0; 0; 1; 1; 1];             % 標籤 mdl = fitctree(X, Y);  % 預測 X_new = [3, 4]; label = predict(mdl, X_new); view(mdl, 'Mode', 'graph');  % 顯示決策樹`

### 6. 隨機森林（Random Forest）

**應用**：隨機森林用於分類和回歸，適用於各種領域，並且在處理缺失數據和非線性問題時表現良好。

**原理**：隨機森林是一種集成算法，它通過構建多棵決策樹來提高分類或回歸的準確性，最終的預測結果是所有決策樹結果的平均值或眾數。

**MATLAB 範例代碼**：

matlab

複製程式碼

`% 隨機森林分類器 X = [1,2; 2,3; 3,3; 4,5; 5,6];  % 特徵 Y = [0; 0; 1; 1; 1];             % 標籤 mdl = TreeBagger(50, X, Y, 'Method', 'classification');  % 預測 X_new = [3, 4]; [label, scores] = predict(mdl, X_new);`

### 7. 朴素貝葉斯（Naive Bayes）

**應用**：朴素貝葉斯經常用於文本分類，如垃圾郵件檢測，並且能夠處理高維數據。

**原理**：朴素貝葉斯基於貝葉斯定理，並假設特徵之間是條件獨立的。這使得它計算效率高且容易實現。

**MATLAB 範例代碼**：

matlab

複製程式碼

`% 朴素貝葉斯分類器 X = [1,2; 2,3; 3,3; 4,5; 5,6];  % 特徵 Y = [0; 0; 1; 1; 1];             % 標籤 mdl = fitcnb(X, Y);  % 預測 X_new = [3, 4]; label = predict(mdl, X_new);`

### 8. K-Means 聚類（K-Means Clustering）

**應用**：K-Means 用於無監督學習中的聚類問題，例如市場分割和圖像壓縮。

**原理**：K-Means 通過最小化每個數據點與最近的簇中心之間的距離來劃分數據。

**MATLAB 範例代碼**：

matlab

複製程式碼

`% K-Means 聚類 X = [1,2; 2,3; 3,3; 4,5; 5,6];  % 特徵 k = 2;  % 聚類數 [idx, C] = kmeans(X, k);  % 顯示結果 scatter(X(:,1), X(:,2), 50, idx, 'filled'); hold on; plot(C(:,1), C(:,2), 'kx', 'MarkerSize', 15, 'LineWidth', 3);`

### 9. 主成分分析（Principal Component Analysis, PCA）

**應用**：PCA 用於數據降維，特別是在高維數據集中，通過投影到低維空間來保留盡可能多的數據變異性。

**原理**：PCA 通過計算數據的主成分（特徵向量）來最大化變異性，並丟棄低變異性的成分。

**MATLAB 範例代碼**：

matlab

複製程式碼

`% 主成分分析 (PCA) X = [1,2; 2,3; 3,3; 4,5; 5,6];  % 特徵矩陣 [coeff, score, latent] = pca(X);  %`

除了之前介紹的常見機器學習算法，以下是一些面試中經常出現的聚類（Clustering）、預測（Prediction）和其他重要的機器學習算法。這些算法不僅常用，還涉及到很多實際應用。每個算法的介紹都附有 MATLAB 代碼範例，以便更好地理解這些技術。

### 1. **層次聚類（Hierarchical Clustering）**

**應用**：層次聚類用於無監督學習的聚類任務，特別適合基因表達分析、客戶群體細分等場景。它不需要事先指定簇數，可以構建層次結構（樹狀圖）。

**原理**：層次聚類從每個數據點作為一個簇開始，然後不斷將最相似的簇合併，直到所有數據點都合併到一個簇中。

**MATLAB 範例代碼**：

matlab

複製程式碼

`% 示例數據 X = rand(10, 2);  % 計算距離矩陣 distanceMatrix = pdist(X);  % 進行層次聚類 tree = linkage(distanceMatrix, 'average');  % 畫出樹狀圖 dendrogram(tree);`

### 2. **DBSCAN（Density-Based Spatial Clustering of Applications with Noise）**

**應用**：DBSCAN 用於發現數據集中任意形狀的簇，特別適合處理具有異常點（噪聲）的數據。它在聚類數未知時也能很好地工作。

**原理**：DBSCAN 基於密度進行聚類，對每個數據點進行密度檢查，將密度相鄰的點分配到相同簇，異常點會被標記為噪聲。

**MATLAB 範例代碼**：

matlab

複製程式碼

`% 示例數據 X = [randn(100, 2) + 1; randn(100, 2) - 1];  % 使用 DBSCAN 聚類 epsilon = 0.5; % 鄰域半徑 minpts = 5;    % 最少鄰居數 labels = dbscan(X, epsilon, minpts);  % 可視化結果 gscatter(X(:,1), X(:,2), labels);`

### 3. **拉索回歸（Lasso Regression）**

**應用**：拉索回歸適用於特徵選擇和避免過擬合。它通過在損失函數中加入 L1 正則化項來約束模型的複雜度。

**原理**：Lasso 回歸的損失函數中加入了一個正則化項，該項懲罰大係數，從而迫使某些特徵的權重變為零，實現特徵選擇。

**MATLAB 範例代碼**：

matlab

複製程式碼

`% 示例數據 X = rand(100, 5); y = rand(100, 1);  % Lasso 回歸 [B, FitInfo] = lasso(X, y);  % 顯示回歸系數 lassoPlot(B, FitInfo, 'PlotType', 'Lambda', 'XScale', 'log');`

### 4. **梯度提升機（Gradient Boosting Machine, GBM）**

**應用**：梯度提升機常用於回歸和分類問題，尤其在比賽中（如 Kaggle 比賽）表現良好。它用於建立一個強學習器，通過迭代地將弱學習器（如決策樹）進行加權組合。

**原理**：GBM 通過一系列決策樹逐步修正前一棵樹的錯誤預測。每次迭代中，新樹的權重是基於當前模型的殘差計算的。

**MATLAB 範例代碼**：

matlab

複製程式碼

`% 示例數據 X = rand(100, 5); y = rand(100, 1);  % 梯度提升回歸模型 Mdl = fitrensemble(X, y, 'Method', 'LSBoost', 'NumLearningCycles', 100);  % 預測 y_pred = predict(Mdl, X);`

### 5. **XGBoost（Extreme Gradient Boosting）**

**應用**：XGBoost 是一種增強型的梯度提升算法，在各種回歸和分類問題中表現優秀。它是基於梯度提升樹的強化版本，能高效處理缺失數據和非線性問題。

**原理**：XGBoost 提供了正則化梯度提升模型，通過加入 L1 和 L2 正則化來防止過擬合，並且使用了樹結構提升演算法。

MATLAB 本身不直接支持 XGBoost，但你可以使用 XGBoost 的 MEX 接口或 Python 版本。

### 6. **貝葉斯優化（Bayesian Optimization）**

**應用**：貝葉斯優化常用於參數調優，特別是當算法參數空間複雜且昂貴時。它能夠高效地搜索最優參數組合。

**原理**：貝葉斯優化基於概率模型（如高斯過程）來預測最有希望的參數，並逐步更新模型以接近全局最優解。

**MATLAB 範例代碼**：

matlab

複製程式碼

`% 設定目標函數 objFcn = @(x) (x.^2 + 3*x + 5);  % 使用貝葉斯優化找到最優解 results = bayesopt(objFcn, [optimizableVariable('x', [-10, 10])]);  % 顯示最優解 bestX = results.XAtMinObjective;`

### 7. **AdaBoost（Adaptive Boosting）**

**應用**：AdaBoost 是一種經典的增強算法，經常用於處理分類問題，如人臉識別和文本分類。它通過組合弱學習器來構建一個強學習器。

**原理**：AdaBoost 通過對錯誤分類的數據點加大權重，使得後續的弱學習器更關注這些錯誤分類點，最終達到提高分類性能的效果。

**MATLAB 範例代碼**：

matlab

複製程式碼

`% 示例數據 X = rand(100, 5); Y = randi([0,1], 100, 1);  % 二分類標籤  % 使用 AdaBoost 進行分類 Mdl = fitcensemble(X, Y, 'Method', 'AdaBoostM1');  % 預測 y_pred = predict(Mdl, X);`

### 8. **多層感知器（Multilayer Perceptron, MLP）**

**應用**：MLP 是一種經典的前向神經網絡，常用於處理分類、回歸以及圖像識別等任務。

**原理**：MLP 包含輸入層、隱藏層和輸出層，通過反向傳播算法來調整權重以最小化誤差。每個神經元使用激活函數（如 ReLU 或 Sigmoid）來實現非線性映射。

**MATLAB 範例代碼**：

matlab

複製程式碼

`% 創建一個 MLP 網絡 net = feedforwardnet(10);  % 隱藏層有 10 個神經元  % 訓練數據 X = rand(100, 5);  % 輸入數據 Y = randi([0,1], 100, 1);  % 標籤  % 訓練神經網絡 net = train(net, X', Y');  % 預測 y_pred = net(X');`

### 9. **隨機梯度下降（Stochastic Gradient Descent, SGD）**

**應用**：SGD 是一種優化算法，廣泛用於深度學習中進行參數更新。特別適合大數據集上的優化問題。

**原理**：SGD 在每次迭代中隨機選取一個或小批量樣本來更新模型參數，從而降低計算成本。它是一種快速且內存效率高的優化算法。

**MATLAB 範例代碼**： MATLAB 本身不直接提供 SGD，但可以手動實現。

matlab

複製程式碼

`% 假設目標函數 loss = (theta - x)^2 theta = 0;  % 初始化參數 learning_rate = 0.01;  % 使用 SGD 更新 theta for i`

4o