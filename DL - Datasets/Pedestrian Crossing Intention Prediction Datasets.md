
「行人過街意圖預測（Pedestrian Crossing Intention Prediction, PCIP）」project 在學界和自駕車研究社群裡非常熱門，我幫你整理：

---

## 一、適合的公開資料集 (Open Datasets)

### 1. **PIE (Pedestrian Intention Estimation Dataset)**

- **內容**：由行車記錄器拍攝，場景在加州。包含行人 bounding boxes、關鍵點、意圖標註 (是否要過馬路)、車速等資訊。
    
- **特色**：有 **bounding box、pose、速度**、意圖標籤，非常對口。
    
- **下載**：PIE Dataset
    

---

### 2. **JAAD (Joint Attention in Autonomous Driving)**

- **內容**：影片片段中標註了行人動作（看左/右、加速/停下）、交通情境。
    
- **特色**：專門設計給行人行為分析，適合做意圖預測。
    
- **下載**：JAAD Dataset
    

---

### 3. **TITAN Dataset**

- **內容**：行人與車輛交互，支援 3D bounding box 與意圖標註。
    
- **特色**：偏向**車輛與行人交互意圖**，支援多類意圖標籤。
    
- **下載**：TITAN Dataset
    

---

### 4. **STIP (Sussex Traffic Intention Prediction)**

- **內容**：英國交通場景，行人意圖與過街行為標註。
    
- **特色**：專門設計給 **intention prediction**。
    
- **下載**：STIP Dataset
    

---

### 5. **NuScenes / Waymo Open Dataset**（雖非專為意圖標註，但可延伸）

- **內容**：多感測器（LiDAR + 多相機）自駕車資料集。
    
- **特色**：可自行標註意圖，且有多視角（前、側、後），適合做 **sensor fusion**。
    
- **下載**：NuScenes，Waymo Open Dataset
    

---

## 二、多視角 / Sensor Fusion 的價值

- **鳥瞰視角 (Top-Down, Drone 或路口攝影機)**：
    
    - 優點：清楚觀察群體軌跡，方便估算 **行人-車輛距離** 與全局路徑。
        
    - 缺點：可能缺乏微表情（head orientation, gaze）細節。
        
    - 適用：**群體行人動態建模**（群體同時過馬路 vs. 個人停下）。
        
- **汽車角度 (Ego-view, 車前相機)**：
    
    - 優點：模擬真實自駕場景，可直接推斷行人 **行進方向 + 速度**。
        
    - 技術：
        
        - 光流 (Optical Flow)
            
        - 追蹤器 (e.g. DeepSORT)
            
        - Pose-based heading estimation（頭部+肩膀朝向 ≈ 未來行進方向）
            
- **多視角融合 (Camera + LiDAR / 多攝影機)**：
    
    - 利用 **3D幾何三角化** 或 **Transformer-based fusion**（如 BEVFormer）將行人位置與朝向放到同一個鳥瞰座標系，提升準確率。
        

---

## 三、其他可提升準確度的 Ideas

1. **行人細粒度特徵**
    
    - 頭部朝向（Head Orientation）與注視方向（Gaze）
        
    - 手部動作（是否看手機、是否舉手示意）
        
    - 可利用 MMPose Whole-Body Pose 模型。
        
2. **時序資訊 (Trajectory + Pose Sequence)**
    
    - 單幀姿態不可靠 → 用 Transformer/LSTM/Graph Neural Network 處理 **pose sequence**。
        
    - 已有研究用 **GCN (Graph Convolutional Network)** 處理 pose sequence，準確率提升明顯。
        
3. **社會互動 (Social Interaction Modeling)**
    
    - 使用 **Social LSTM / Social GAN**，考慮「群體行人」互動，例如一群人同時過馬路。
        
4. **場景語義 (Scene Context)**
    
    - 加入 **交通燈狀態、車流量、車速** → 提高預測準確度。
        
    - 可用 **semantic segmentation (例如 Segment Anything / DeepLab)** 抽取道路區域與斑馬線位置。
        
5. **多模態融合 (Vision + Motion Sensors)**
    
    - 若有車輛自帶 IMU、雷達，可用來估算相對速度，結合影像姿態更穩定。
        
6. **預測方法**
    
    - 使用 **多任務學習**：同時輸出
        
        - (a) 過街意圖（binary / multi-class）
            
        - (b) 未來 1–3 秒軌跡（trajectory forecasting）
            
    - 模型範例：IntentNet、PedFormer。
        

---

## 四、難度分級

- **低難度**：JAAD/PIE dataset，直接跑預訓練模型 + 輸出二分類（要不要過馬路）。
    
- **中難度**：加入 pose sequence + GCN/Transformer，做 trajectory forecasting。
    
- **高難度**：多視角 (multi-camera 或 bird’s-eye view) + sensor fusion，結合場景語義。




```
請詳細中文解釋NuScenes / Waymo Open Dataset, 各有那些video或image, 那些角度, 是如何標註. 並具體舉例說明data. 以及適合做Pedestrian Crossing Intention Prediction, PCIP）嗎
```

詳細解釋 **NuScenes** 和 **Waymo Open Dataset**，分別有哪些影像/影片、多角度攝影機設置、標註格式，並舉具體例子，最後再說明是否適合用來做 **行人過街意圖預測 (Pedestrian Crossing Intention Prediction, PCIP)**。

---

# 🟢 1. NuScenes Dataset

### 基本資訊

- 由 **Motional (原 nuTonomy)** 與 **Aptiv** 收集（新加坡、波士頓）。
    
- **1,000 段 20 秒場景影片**，共約 **1.4M 影像**。
    
- 每段場景長 **20 秒**，以 **2 Hz 取樣 (每秒 2 幀)**，所以每段有 40 幀同步資料。
    

### 感測器配置 (全方位 360°)

- **6 個相機**：
    
    - 前左 (Front Left)
        
    - 前 (Front)
        
    - 前右 (Front Right)
        
    - 後左 (Back Left)
        
    - 後 (Back)
        
    - 後右 (Back Right)
        
    - ⇒ 提供完整 360° 水平視角
        
- **1 個 32-beam LiDAR**（中央旋轉式，360° 覆蓋）。
    
- **5 個雷達 (Radar)**：前、後、四角。
    

### 標註內容

- **3D bounding boxes**：23 種類別，包括：Pedestrian, Vehicle, Bus, Bicycle 等。
    
- **屬性 (Attributes)**：例如對 Pedestrian 有 **standing / moving / sitting / lying down**。
    
- **追蹤 ID**：可跨幀追蹤行人。
    
- **地圖語義 (HD Map)**：包含車道線、斑馬線、路口、交通號誌位置。
    

### 具體例子

一幀影像中的 JSON 標註可能包含：

`{  "sample_token": "xx123",  "annotation": {    "category_name": "human.pedestrian.adult",    "attribute": "pedestrian.moving",    "bbox_3d": [x, y, z, width, length, height, yaw],    "tracking_id": 105,    "visibility_token": 3  } }`

意思是：某一個成人行人正在移動，有對應的 3D 框、ID，可追蹤其路徑。

### 適合 PCIP 嗎？

✅ 優點：

- 有 360° 視角，可以同時觀察車前和車側的行人動作。
    
- 標註有「行人是否在移動 / 停下」，對於 **意圖建模** 有幫助。
    
- 有 **斑馬線資訊 (map data)**，可判斷行人是否接近過街區域。
    

❌ 限制：

- 沒有「explicit 意圖標籤」（例如“要過街/不過街”），需要從「行人狀態 + 地圖位置 + 軌跡」推測。
    

---

# 🟢 2. Waymo Open Dataset (WOD)

### 基本資訊

- 由 **Waymo (Google 旗下)** 收集，美國多州（加州、鳳凰城等）。
    
- **1,150 段影片**，每段 **20 秒**，取樣 **10 Hz**。
    
- 提供 **感測器同步資料**：相機 + LiDAR。
    

### 感測器配置

- **5 個相機**：
    
    - 前 (Front)
        
    - 前左 (Front-Left)
        
    - 前右 (Front-Right)
        
    - 側左 (Side-Left)
        
    - 側右 (Side-Right)
        
    - ⇒ 視角幾乎覆蓋車前方與兩側，但後方缺乏。
        
- **LiDAR**：5 個雷射雷達（1 個 64-beam roof LiDAR + 4 個短距離 200m LiDAR）。
    

### 標註內容

- **3D bounding boxes**：行人、車輛、自行車等。
    
- **追蹤 ID**：可跨幀追蹤，支援 **tracking benchmark**。
    
- **動作狀態**：包含 bounding box + 速度向量 (vx, vy)，所以能估計運動方向。
    

### 具體例子

一幀 JSON 標註可能像：

`{  "object_id": "ped_5678",  "label": "Pedestrian",  "bbox_3d": [x, y, z, width, length, height, heading],  "velocity": [vx, vy],  "num_lidar_points": 24,  "num_camera_points": 15 }`

意思是：某行人有 3D 框與速度 (vx, vy)，可直接推估行進方向。

### 適合 PCIP 嗎？

✅ 優點：

- 高頻率 (10Hz)，更容易分析動作與意圖。
    
- 有「速度向量」，可直接判斷行人行進方向與加速度。
    
- 可跨感測器 (LiDAR + Camera) 做 **fusion**，提升判斷準確度。
    

❌ 限制：

- 沒有直接的「過街意圖」標籤，需要研究者自己定義規則或透過學習推斷。
    

---

# 🔵 NuScenes vs. Waymo Dataset 比較 (針對 PCIP)

|特點|**NuScenes**|**Waymo Open Dataset**|
|---|---|---|
|**場景數據量**|1,000 段 × 20 秒 (2Hz)|1,150 段 × 20 秒 (10Hz)|
|**相機數**|6 個 (360° 視角)|5 個 (前/側，無後方)|
|**LiDAR**|1 × 32-beam + 5 radar|5 × LiDAR (64-beam + 4 短距)|
|**行人標註**|類別 + 3D bbox + 動作屬性 (moving/standing)|類別 + 3D bbox + 速度向量|
|**地圖資訊**|有 (斑馬線、車道線)|無完整 HD map|
|**適合 PCIP?**|適合研究「行人位置 vs. 斑馬線」意圖推斷|適合研究「行人速度/方向」意圖推斷|

---

# 🟠 結論：是否適合做 PCIP？

- **NuScenes**：
    
    - 適合做「行人是否準備進入斑馬線」的研究。
        
    - 可利用斑馬線位置 + 行人動作屬性 + 軌跡 → 預測過街意圖。
        
- **Waymo Open Dataset**：
    
    - 適合做「行人動態行進方向/速度預測」。
        
    - 可直接用速度向量判斷加速或減速，並預測是否走向車道。
        

👉 最佳方案：**把兩者結合**

- 用 NuScenes 學習「場景語境 (context + 斑馬線)」
    
- 用 Waymo 學習「動態運動 (velocity + heading)」
    
- 最後 fusion → 強化 PCIP 模型準確度。
    

---

要不要我幫你設計一個 **PCIP pipeline**，具體說明：

1. 用哪些 dataset → 處理成統一格式
    
2. 哪些模型（Pose Estimation + Trajectory Forecasting + Scene Context Fusion）
    
3. 預測輸出（過街 vs. 不過街 + 未來 3 秒軌跡）



```
請幫我從上面提供的公開資料集推薦可以做行人過街意圖預測（Pedestrian Crossing Intention Prediction, PCIP), 以及多視角 / Sensor Fusion相關model
```

把 **適合 PCIP 的資料集** 和 **多視角 / Sensor Fusion 模型** 梳理清楚，讓你能快速挑選起手式。

---

# 🟢 一、推薦的公開資料集 (適合 PCIP)

|資料集|特點|適合 PCIP 的理由|是否有多視角/多感測器|
|---|---|---|---|
|**PIE (Pedestrian Intention Estimation)**|車前行車紀錄器影片 (ego-view)，有 bounding box、pose、意圖標註|✅ 有明確的「過街意圖」標籤，是 PCIP 的黃金標準|單視角 (前方攝影)|
|**JAAD (Joint Attention in Autonomous Driving)**|行車紀錄器影片，標註行人動作 (看左右、走/停)|✅ 有「head orientation、動作」標籤，可做意圖推斷|單視角|
|**STIP (Sussex Traffic Intention Prediction)**|英國交通場景，標註意圖|✅ 為 Intention Prediction 設計，適合 PCIP|單視角|
|**TITAN**|行人與車輛交互，含 3D bbox|✅ 有交互意圖，可延伸到 PCIP|多視角部分支援|
|**NuScenes**|6 攝影機 (360°)、LiDAR、Radar，標註行人 bbox、屬性 (moving/standing)，含 HD map (斑馬線/車道線)|✅ 可利用「地圖+行人屬性」做過街意圖|多視角 (6 cam + LiDAR + Radar)|
|**Waymo Open Dataset**|5 攝影機、5 LiDAR，標註 3D bbox + velocity|✅ 有速度向量，可推斷是否加速跨入車道|多視角 (5 cam + 5 LiDAR)|

👉 **建議組合**：

- **短期快速研究** → 用 **PIE + JAAD**（因為有直接的意圖標籤）。
    
- **多視角 / Sensor Fusion 研究** → 用 **NuScenes + Waymo**（利用多相機 + LiDAR + map context）。
    
- **進階** → 融合 JAAD/PIE 的「意圖標籤」+ NuScenes/Waymo 的「感測器資訊」，訓練一個泛化 PCIP 模型。
    

---

# 🟢 二、推薦的模型 / 方法

## (A) 單視角 PCIP 模型

1. **IntentNet (ICRA 2019, NVIDIA)**
    
    - 輸入：影像 + 地圖 (Bird’s-Eye View)
        
    - 輸出：行人是否要過街的二分類 + 未來軌跡
        
    - 適合：JAAD、PIE
        
2. **PedFormer (Transformer-based PCIP)**
    
    - 用 Transformer 處理行人關鍵點序列 (pose sequence)
        
    - 適合：PIE、STIP
        

---

## (B) 多視角 / Sensor Fusion 模型

1. **BEVFormer (Bird’s Eye View Transformer)**
    
    - 把多攝影機影像投影到鳥瞰圖 (BEV)，再做檢測與意圖預測
        
    - 適合：NuScenes, Waymo
        
2. **ViT + Trajectory Forecasting (Social-STGCNN / Trajectron++)**
    
    - 使用 **Vision Transformer 提取影像特徵**，再接入 **Graph Neural Network** 處理多行人軌跡交互
        
    - 適合：多人同時過街的場景
        
3. **Multi-Sensor Fusion (LiDAR + Camera)**
    
    - **Fusion Transformer**：同時輸入 LiDAR point cloud (3D) + 影像 (2D)，在 Transformer 裡 cross-attention
        
    - 適合：Waymo, NuScenes
        

---

## (C) 可提升 PCIP 準確度的附加模組

- **Head Orientation / Gaze Estimation** → MMPose Whole-body 模型可抓臉部關鍵點，判斷是否看向車道
    
- **Scene Context** → 利用 segmentation (DeepLab, SAM2) 判斷斑馬線、紅綠燈狀態
    
- **Social Interaction** → Social-LSTM / Social-GAN：考慮多行人群體互動
    
- **Multi-task Learning** → 同時學「是否過街 + 未來 1–3 秒軌跡」，能讓模型更穩定
    

---

# 🟢 三、結論與建議

- **新手起手式**：用 **PIE dataset + IntentNet**，因為有意圖標籤，能快速跑出 baseline。
    
- **進階研究**：用 **NuScenes / Waymo + BEVFormer**，做 **多視角 fusion**，可用於真實自駕情境。
    
- **挑戰方向**：結合 **pose sequence (PedFormer)** + **map context (斑馬線、紅綠燈)** + **多視角 fusion**，提升準確率。