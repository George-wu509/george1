
What concepts are commonly asked in 𝗠𝗟 𝗘𝗻𝗴𝗶𝗻𝗲𝗲𝗿 𝗶𝗻𝘁𝗲𝗿𝘃𝗶𝗲𝘄𝘀?  
  
These are fair game in interviews at 𝘀𝘁𝗮𝗿𝘁𝘂𝗽𝘀, 𝗰𝗼𝗻𝘀𝘂𝗹𝘁𝗶𝗻𝗴 𝗳𝗶𝗿𝗺𝘀 & 𝗙𝗔𝗔𝗡𝗚𝘀.  
  
Here's a comprehensive list 👇👇👇  
  
𝗠𝗟 𝗙𝘂𝗻𝗱𝗮𝗺𝗲𝗻𝘁𝗮𝗹𝘀  
↳ Variance-Bias Trade-off  
↳ Regression Algorithms (e.g. Linear, Polynomial)  
↳ Classification Algorithms (e.g. GBM)  
↳ Clustering Algorithms  
↳ Deep Learning (e.g. DNN, LSTM, Transformers)  
↳ Reinforcement Learning  
↳ Model Evaluation Metrics (Precision, Recall, AUC-ROC)  
↳ Handling Missing Data  
↳ Feature Scaling (Normalization, Standardization)  
↳ Feature Selection Techniques  
↳ Dimensionality Reduction (PCA, t-SNE)  
↳ Encoding Categorical Data  
↳ Hyperparameter Tuning  
↳ Cross-Validation Techniques  
↳ Regularization Methods (L1, L2)  
  
𝗖𝗼𝗱𝗶𝗻𝗴  
↳ Python for ML  
↳ Writing Efficient Code  
↳ Data Structures and Algorithms  
↳ Implementing ML Algorithms from Scratch  
↳ Working with ML Libraries (scikit-learn, PyTorch)  
  
𝗦𝘆𝘀𝘁𝗲𝗺 𝗗𝗲𝘀𝗶𝗴𝗻  
↳ Data Ingestion and Preprocessing  
↳ ETL/ELT Processes  
↳ Handling Big Data (Hadoop, Spark)  
↳ Kafka for streaming  
↳ Caching  
↳ SQL vs noSQL  
↳ Load Balancing  
↳ Edge Deploymebt  
  
𝗠𝗟 𝗦𝘆𝘀𝘁𝗲𝗺 𝗗𝗲𝘀𝗶𝗴𝗻  
↳ Design Recommender System  
↳ Fraud Detection System  
↳ Real-Time Bidding  
↳ Chatbot Architecture  
↳ Sentiment Analysis Pipeline  
↳ Image Classification System  
↳ Voice Recognition System  
  
𝗠𝗟 𝗗𝗲𝗽𝗹𝗼𝘆𝗺𝗲𝗻𝘁  
↳ Model Serving (Batch, Real-Time)  
↳ Monitoring and Maintaining Models in Production  
↳ Model Retraining Strategies  
↳ Workflow Orchestration (Airflow, Kubeflow)  
↳ Experiment Tracking (MLflow)  
↳ Model Registry and Versioning  
  
Now go ace your next interview👇  
  
📕 𝗜𝗻𝘁𝗲𝗿𝘃𝗶𝗲𝘄 𝗣𝗿𝗲𝗽 𝗖𝗼𝘂𝗿𝘀𝗲𝘀: https://lnkd.in/gzgB-dHT  
📘 𝗝𝗼𝗶𝗻 𝗗𝗦 𝗜𝗻𝘁𝗲𝗿𝘃𝗶𝗲𝘄 𝗕𝗼𝗼𝘁𝗰𝗮𝗺𝗽: https://lnkd.in/eiA5Ntdp  
📙 𝗝𝗼𝗶𝗻 𝗠𝗟𝗘 𝗜𝗻𝘁𝗲𝗿𝘃𝗶𝗲𝘄 𝗕𝗼𝗼𝘁𝗰𝗮𝗺𝗽: https://lnkd.in/e6HbN6dy  
📗 𝗔𝗕 𝗧𝗲𝘀𝘁𝗶𝗻𝗴 𝗖𝗼𝘂𝗿𝘀𝗲: https://lnkd.in/g82dMJ77


### **1. Variance-Bias Trade-off（偏差-方差權衡）**

偏差-方差權衡（Bias-Variance Trade-off）是機器學習中關於**模型複雜度與泛化能力**之間的平衡問題：

- **偏差（Bias）**：指模型對數據的適配能力。如果偏差高，說明模型過於簡單，可能無法捕捉數據的真實模式，導致**欠擬合（underfitting）**。
- **方差（Variance）**：指模型對數據的敏感度。如果方差高，說明模型對訓練數據學得過好，可能記住了數據的細節和噪聲，導致**過擬合（overfitting）**。

**解決方法：**

- 增加數據量可以降低方差
- 適當的正則化（L1, L2）可以降低過擬合
- 使用交叉驗證（Cross-Validation）來選擇最佳模型

---

### **2. Regression Algorithms（回歸演算法）**

回歸（Regression）是用於預測**連續數值**的模型：

- **線性回歸（Linear Regression）**：假設輸入特徵與輸出之間的關係為線性，例如： y=w1x1+w2x2+⋯+wnxn+by = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + by=w1​x1​+w2​x2​+⋯+wn​xn​+b
- **多項式回歸（Polynomial Regression）**：在線性回歸的基礎上加入多項式特徵，例如： y=w1x+w2x2+w3x3+by = w_1 x + w_2 x^2 + w_3 x^3 + by=w1​x+w2​x2+w3​x3+b 用於擬合非線性數據。

---

### **3. Classification Algorithms（分類演算法，例如 GBM）**

分類（Classification）是用來預測**離散標籤**（例如「貓或狗」、「疾病或健康」）的算法：

- **決策樹（Decision Tree）**：根據數據特徵逐層進行分類。
- **梯度提升機（Gradient Boosting Machine, GBM）**：一種集成學習（Ensemble Learning）方法，它透過多棵決策樹來提升分類性能（如 XGBoost, LightGBM）。

---

### **4. Clustering Algorithms（聚類演算法）**

聚類（Clustering）是**無監督學習**方法，將數據點劃分為不同的組：

- **K-Means**：基於質心（centroid）對數據進行分組。
- **DBSCAN**：基於密度，適合非球形分佈的數據。
- **層次聚類（Hierarchical Clustering）**：生成一棵樹狀的數據分類結構。

---

### **5. Deep Learning（深度學習）**

- **深度神經網絡（Deep Neural Network, DNN）**：多層感知機（MLP），適用於結構化數據與簡單的影像處理。
- **長短時記憶網絡（LSTM）**：處理時間序列數據（如語音識別、股市預測）。
- **變換器（Transformers）**：例如 BERT、GPT，用於自然語言處理（NLP）。

---

### **6. Reinforcement Learning（強化學習）**

強化學習（RL）是基於獎勵機制的學習方法：

- **Agent**（智能體）在環境（Environment）中採取行動（Action），獲取獎勵（Reward）。
- 常見算法包括：
    - Q-Learning（基於 Q 值更新）
    - Deep Q Network（DQN）
    - Actor-Critic（策略梯度方法）

---

### **7. Model Evaluation Metrics（模型評估指標）**

- **準確率（Accuracy）**：正確預測數量佔總數的比例。
- **精確率（Precision）**： Precision=TPTP+FP\text{Precision} = \frac{TP}{TP + FP}Precision=TP+FPTP​ 表示模型預測為正類的樣本中有多少是真正的正類。
- **召回率（Recall）**： Recall=TPTP+FN\text{Recall} = \frac{TP}{TP + FN}Recall=TP+FNTP​ 表示實際的正類樣本中有多少被模型正確預測。
- **AUC-ROC**：衡量分類模型的性能，AUC 越接近 1，模型越優秀。

---

### **8. Handling Missing Data（處理缺失數據）**

- **刪除缺失數據**：如果缺失值比例低，可以刪除。
- **填充缺失值（Imputation）**：
    - 使用均值、中位數、眾數填補
    - 使用回歸或 KNN 進行插補

---

### **9. Feature Scaling（特徵縮放）**

- **歸一化（Normalization）**： x′=x−min⁡(x)max⁡(x)−min⁡(x)x' = \frac{x - \min(x)}{\max(x) - \min(x)}x′=max(x)−min(x)x−min(x)​ 把數據映射到 [0,1]。
- **標準化（Standardization）**： x′=x−μσx' = \frac{x - \mu}{\sigma}x′=σx−μ​ 讓數據的均值為 0，標準差為 1。

---

### **10. Feature Selection Techniques（特徵選擇方法）**

- **Filter 方法**（基於統計）：如皮爾森相關係數、信息增益。
- **Wrapper 方法**（基於模型）：如遺傳算法（GA）。
- **嵌入式方法（Embedded）**：如 Lasso 回歸（L1 正則化）。

---

### **11. Dimensionality Reduction（降維）**

- **主成分分析（PCA）**：基於協方差矩陣找到最佳投影方向，保留最大變異數。
- **t-SNE**：非線性降維方法，適合可視化。

---

### **12. Encoding Categorical Data（編碼類別特徵）**

- **獨熱編碼（One-Hot Encoding）**：適用於無序類別變量。
- **標籤編碼（Label Encoding）**：適用於有序變量（如小、中、大）。

---

### **13. Hyperparameter Tuning（超參數調整）**

- **網格搜索（Grid Search）**：嘗試所有可能的參數組合。
- **隨機搜索（Random Search）**：隨機抽樣部分參數組合。
- **貝葉斯優化（Bayesian Optimization）**：根據過去結果來選擇新的參數組合。

---

### **14. Cross-Validation Techniques（交叉驗證技術）**

- **K 折交叉驗證（K-Fold Cross Validation）**：將數據分成 K 個子集，每次用 K-1 個子集訓練，剩餘 1 個子集測試。
- **留一驗證（LOO, Leave-One-Out）**：每次用 n-1 個數據訓練，1 個數據測試。

---

### **15. Regularization Methods（正則化）**

- **L1 正則化（Lasso）**：讓部分權重變 0，實現特徵選擇。
- **L2 正則化（Ridge）**：抑制過大的權重，提升模型穩定性。