#### Title:
Multi-Objective Neural Architecture Optimization


---
##### **Resume Keyworks:
<mark style="background: #ADCCFFA6;">BPN</mark>, <mark style="background: #ADCCFFA6;">NSGAII</mark>, <mark style="background: #ADCCFFA6;">NN architecture optimization</mark>
##### **STEPS:**
step0. Camera calibration

---


#### Resume: 
Developed a multi-objective optimization framework using fuzzy logic and NSGA-II to optimize BPN architecture for rainfall prediction, achieving enhanced accuracy and reduced complexity. Delivered scalable, resource-efficient solutions by balancing model simplicity and prediction performance through Pareto-optimal design.

#### Abstract: 
This study introduces a **Multi-Objective Neural Architecture Optimization** framework for optimizing Back Propagation Neural Networks (BPN) in rainfall prediction tasks. The proposed method addresses two objectives: (1) minimizing network complexity by reducing the number of layers and nodes, and (2) maximizing prediction accuracy. To balance these competing objectives, we integrate **Fuzzy Logic** to handle uncertainty and prioritize trade-offs between model simplicity and accuracy. The optimization process employs the **Non-Dominated Sorting Genetic Algorithm II (NSGA-II)**, generating a Pareto front of optimal solutions that offer various trade-offs between the objectives. The framework is validated on a meteorological dataset, where each solution balances prediction performance with computational efficiency. Experimental results demonstrate significant improvements in rainfall prediction accuracy while compressing the BPN architecture, achieving robust and efficient predictive performance. This approach provides a scalable and adaptable solution for designing lightweight neural networks in time-series prediction tasks, enabling resource-efficient deployment in real-world applications.

#### Technique detail: 

### 建立多目標神經網路架構優化方法的原理與流程詳細說明

#### 1. **背景與目標**

針對**後向傳播神經網路 (Back Propagation Neural Network, BPN)**，目標是建立一種基於**多目標優化 (Multi-Objective Optimization)** 的架構設計方法，以解決以下兩個目標：

- **目標1**：降低BPN的架構複雜度，例如減少層數（Layer Number）或節點數（Node Number），以提升計算效率和模型壓縮性。
- **目標2**：提升BPN對於降雨量預測（Rainfall Prediction）結果的準確性。

我們採用**模糊邏輯 (Fuzzy Logic)** 結合目標函數進行模糊化處理，以更好地處理這些目標之間的權衡。最後通過**非支配排序遺傳算法II (Non-Dominated Sorting Genetic Algorithm II, NSGA-II)**，獲得一組Pareto解集（Pareto Solutions），在保證預測準確度的前提下壓縮模型架構。

---

#### 2. **方法論與技術詳解**

##### 2.1 **後向傳播神經網路 (BPN)**

BPN 是一種常見的前饋型人工神經網路，使用梯度下降法（Gradient Descent）進行權重更新，適用於時間序列預測任務。在此專案中，BPN用於降雨量預測。

- **輸入 (Input)**：氣象相關數據，例如降雨歷史數據、溫度、濕度、氣壓等。
- **輸出 (Output)**：未來降雨量的預測值。

---

##### 2.2 **多目標優化問題**

- **目標函數1**：最小化神經網路架構複雜度（Complexity\text{Complexity}Complexity）。
    
    f1=總節點數 (Total Nodes)最大允許節點數 (Max Nodes)+總層數 (Total Layers)最大允許層數 (Max Layers)f_1 = \frac{\text{總節點數 (Total Nodes)}}{\text{最大允許節點數 (Max Nodes)}} + \frac{\text{總層數 (Total Layers)}}{\text{最大允許層數 (Max Layers)}}f1​=最大允許節點數 (Max Nodes)總節點數 (Total Nodes)​+最大允許層數 (Max Layers)總層數 (Total Layers)​
- **目標函數2**：最小化預測誤差（Prediction Error\text{Prediction Error}Prediction Error），例如均方誤差 (Mean Squared Error, MSE)：
    
    f2=1N∑i=1N(yi−y^i)2f_2 = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2f2​=N1​i=1∑N​(yi​−y^​i​)2

##### 2.3 **模糊邏輯 (Fuzzy Logic)**

為了更靈活地整合上述目標函數，模糊邏輯將目標函數值轉換為模糊集合，使用語義模糊集進行權重分配，例如：

- **高準確性 (High Accuracy)**：對應較小的 f2f_2f2​。
- **低複雜度 (Low Complexity)**：對應較小的 f1f_1f1​。

透過模糊規則（如「如果 f1f_1f1​ 很小且 f2f_2f2​ 很小，那麼適合度很高」），對候選解進行適合度評估。

---

##### 2.4 **非支配排序遺傳算法 II (NSGA-II)**

NSGA-II 是一種多目標進化算法，用於優化模糊化後的多目標函數，流程如下：

1. **種群初始化**：隨機生成初始種群，每個個體代表一個BPN架構（包括層數和節點數）。
2. **適應度評估**：計算每個個體的 f1f_1f1​ 和 f2f_2f2​，並基於模糊邏輯進行適應度模糊化。
3. **非支配排序**：
    - 排除被其他個體完全支配（即在所有目標上均表現更差）的個體。
    - 分層進行排序，越靠前的層擁有越高的選擇優先權。
4. **擬合度排序 (Crowding Distance Sorting)**：在每層內，基於解的分布密度進行排序，避免解集中於特定區域。
5. **交叉與變異**：選擇種群進行交叉（Crossover）和變異（Mutation），生成新一代候選解。
6. **種群更新**：根據非支配排序和密度排序，保留最優個體進入下一代。
7. **終止條件**：達到預設代數或收斂後，輸出 Pareto 最優解集。

---

#### 3. **專案實現步驟**

##### 3.1 **數據準備**

1. 收集降雨相關數據集（如地面氣象站、衛星遙感數據）。
2. 將數據進行歸一化處理（Normalization）。
3. 分割訓練集與測試集。

##### 3.2 **目標函數設計**

1. 使用公式設計 f1f_1f1​ 和 f2f_2f2​。
2. 定義模糊邏輯的模糊集（如高準確性、低複雜度）及其規則。

##### 3.3 **NSGA-II 優化過程**

1. 初始化種群，每個個體包含：
    - 神經網路層數。
    - 每層節點數。
2. 計算目標函數值 f1,f2f_1, f_2f1​,f2​。
3. 進行非支配排序，交叉與變異操作，並更新種群。

##### 3.4 **Pareto 解集分析**

1. 繪製 Pareto 前沿（Pareto Front）：
    - x 軸：f1f_1f1​（模型複雜度）。
    - y 軸：f2f_2f2​（預測誤差）。
2. 根據實際需求，從 Pareto 解集中選擇平衡點。

---

#### 4. **預期結果**

1. 獲得多個平衡的神經網路架構（如簡化模型但仍保持高準確性）。
2. 在降雨預測任務中，相比原始BPN模型提高準確性並減少計算成本。
3. 提供一組Pareto解，供決策者選擇合適的架構。

---

#### 5. **潛在挑戰與解決方案**

1. **數據質量不足**：使用數據增強（Data Augmentation）方法提高訓練數據質量。
2. **模型過於簡化**：設置最低允許層數與節點數，避免過度壓縮。
3. **計算資源需求高**：採用並行計算加速NSGA-II的收斂。

此專案能實現多目標優化設計，提升降雨預測準確性與模型效率，適用於氣象領域中的實際應用場景。