#### Title:
Spatial-Temporal ANN Prediction and Classification system


---
##### **Resume Keyworks:
<mark style="background: #ADCCFFA6;">BPN</mark>, <mark style="background: #ADCCFFA6;">SOM</mark>
##### **STEPS:**
step0. Camera calibration

---

#### Resume: 
Developed an AI-based framework combining BPN and SOM to predict and classify arsenic concentrations over 20 years of spatial-temporal data, enabling accurate forecasting, pattern recognition, and actionable insights for environmental monitoring and remediation efforts.

#### Abstract: 
Arsenic contamination poses significant environmental and health risks, necessitating effective monitoring and prediction systems. This project develops an **AI-based spatial-temporal modeling framework** to predict and classify arsenic concentrations using 20 years of monthly data collected from various locations. The prediction model employs a **Back Propagation Neural Network (BPN)**, designed to capture temporal and spatial dependencies for forecasting arsenic concentrations at specific times and locations. Complementing this, a **Self-Organizing Map (SOM)** is utilized to classify spatial-temporal patterns, identifying clusters of arsenic concentration trends across regions and time periods. The workflow includes data preprocessing (cleaning, normalization), BPN-based concentration forecasting, and SOM-based spatial-temporal clustering. Model performance is evaluated using metrics such as Mean Squared Error (MSE) and clustering visualization. Results from this dual-model approach are expected to provide actionable insights into high-risk regions and critical timeframes, supporting environmental monitoring and remediation efforts. The framework's scalability and adaptability make it a valuable tool for long-term arsenic concentration management and decision-making.

#### Technique detail: 

### **專案名稱：Arsenic Spatial-Temporal Prediction and Classification**

---

#### **專案背景**

砷(Arsenic)的污染對環境和健康有重大影響，特別是在地下水和土壤中的累積濃度。監測與預測砷濃度在不同時間與空間的變化對於環境監控和決策至關重要。本專案基於20年來每月的砷濃度數據，使用以下兩種技術：

1. **反向傳播神經網絡 (Back Propagation Neural Network, BPN)**：進行時間和空間的砷濃度預測。
2. **自組織映射網絡 (Self-Organizing Map, SOM)**：對砷濃度數據進行空間-時間分類。

---

#### **專案目標**

1. 建立基於**BPN**的砷濃度預測模型，實現對某時間與地點濃度的準確預測。
2. 利用**SOM**進行數據的空間-時間模式分類，識別不同地點和時間的濃度特徵。

---

### **專案的原理與方法**

#### **(1) 數據處理 (Data Processing)**

1. **數據來源**：
    - 數據包括20年來每個月不同地點的砷濃度，數據形式為時間序列與空間網格點的組合。
    - 假設數據格式為 `(地點ID, 時間(月份), 砷濃度)`。
2. **數據清理**：
    - 處理遺漏值，例如使用插值法 (Interpolation) 填補缺失值。
    - 排除異常值 (Outliers)，使用統計方法如3倍標準差法進行異常值檢測。
3. **數據標準化 (Normalization)**：
    - 為了提高模型的效率，對砷濃度數據進行標準化處理，例如將數據縮放至 `[0, 1]` 或 `[-1, 1]` 範圍。

---

#### **(2) 預測模型 (Prediction Model: BPN)**

1. **反向傳播神經網絡 (Back Propagation Neural Network, BPN)**
    
    - **結構設計**：
        - 輸入層 (Input Layer)：包括時間特徵（如月份或年份編碼）與空間特徵（如地點ID）。
        - 隱藏層 (Hidden Layer)：根據數據的複雜性設計1-2層，每層包含若干神經元。
        - 輸出層 (Output Layer)：對應預測的砷濃度值。
    - **激活函數 (Activation Function)**：
        - 隱藏層使用ReLU或Sigmoid函數。
        - 輸出層使用線性函數 (Linear Function)。
    - **損失函數 (Loss Function)**：
        - 使用均方誤差 (Mean Squared Error, MSE) 作為損失函數。
    - **訓練方式**：
        - 使用時間序列數據進行滑動窗口訓練（如以前6個月預測下一個月）。
        - 優化器選擇：Adam或RMSprop。
2. **輸入與輸出設置**：
    
    - **輸入特徵**：
        - 時間：如月份（1到12）或年份。
        - 空間：地點的經緯度或編碼。
    - **輸出目標**：
        - 目標濃度值。
3. **模型訓練與評估**：
    
    - 使用歷史數據進行訓練，並劃分訓練集 (Training Set) 與測試集 (Testing Set)。
    - 使用均方誤差 (MSE) 與決定係數 (R²) 評估模型準確性。

---

#### **(3) 分類模型 (Classification Model: SOM)**

1. **自組織映射 (Self-Organizing Map, SOM)**：
    
    - **原理**：
        - SOM是一種無監督學習模型，能將高維數據映射到低維（通常是2D）的網格中，並根據相似性進行分類。
    - **結構設計**：
        - 輸入層 (Input Layer)：包括砷濃度數據、時間特徵（如年份、月份）、空間特徵（如地點）。
        - 網格尺寸 (Grid Size)：設置為適當大小，如10×10或15×15。
    - **訓練目標**：
        - 找出數據的空間和時間模式，例如不同地點的濃度群集。
2. **SOM訓練流程**：
    
    - 初始化網格的權重值（隨機或基於數據分布）。
    - 對每個輸入數據點：
        - 計算與網格所有節點的歐幾里得距離 (Euclidean Distance)。
        - 更新最相似節點及其鄰域內節點的權重值。
    - 持續迭代，直至網格收斂。
3. **輸入與輸出設置**：
    
    - **輸入特徵**：
        - 每月的砷濃度數據。
        - 地點的經緯度或地點ID。
    - **輸出結果**：
        - 群集的分類結果，每個網格點代表一種空間-時間濃度模式。

---

### **專案完整流程**

1. **數據收集與清理**：
    
    - 收集20年來的月度砷濃度數據，進行清理和標準化。
2. **建構預測模型 (BPN)**：
    
    - 將數據劃分為訓練集與測試集。
    - 訓練BPN模型，實現砷濃度的時間-空間預測。
3. **構建分類模型 (SOM)**：
    
    - 訓練SOM模型，識別不同地點和時間的濃度特徵。
4. **模型驗證與優化**：
    
    - 比較模型的預測結果與實際值，優化網絡結構與參數。
    - 驗證SOM的分類結果，分析不同分類群集的環境意義。
5. **結果分析與應用**：
    
    - 基於預測模型對未來砷濃度進行預測。
    - 利用分類模型找出高風險區域和時間段，為環境治理提供決策支持。

---

### **工具與技術**

1. **程式語言與框架**：
    - Python：使用TensorFlow/Keras或PyTorch實現BPN。
    - SOM：使用`MiniSom`或`SOM Toolbox`。
2. **數據可視化**：
    - Matplotlib、Seaborn用於可視化分類結果與預測值。
    - 地理信息工具 (如GeoPandas) 繪製空間分布圖。

---

### **專案挑戰與解決方案**

1. **數據稀疏與不均衡**：
    
    - 問題：不同地點的數據可能不完整或稀疏。
    - 解決方案：使用插值法填補缺失值，或設計基於稀疏數據的模型。
2. **多尺度特徵處理**：
    
    - 問題：空間和時間的特徵具有多尺度性。
    - 解決方案：採用多層神經網絡結構，分別提取時間與空間特徵。
3. **模型解釋性**：
    
    - 問題：BPN與SOM的結果可能難以直觀解釋。
    - 解決方案：結合SHAP值或特徵重要性分析，提升模型的解釋能力。