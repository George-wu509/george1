

## 現有 Climbing Video 相關 datasets & projects

雖然不像足球、籃球那樣有大規模開放資料集，但有幾個方向可參考：

### (A) Climbing 專用 datasets

- **ClimbAlong dataset (2022)**
    
    - 來源：法國團隊發表，針對 **bouldering** 動作。
        
    - 包含：多視角影片，標註了攀岩者在 route 上的動作與姿態。
        
    - 用途：動作識別、姿態估計。
        
- **ClimbersPose dataset (2021, research demo)**
    
    - 針對攀岩者姿態估計 (Pose Estimation)。
        
    - 提供關節 keypoints（類似 COCO 17 keypoints，但適配攀岩）。
        
- **Speed Climbing datasets (YouTube / IFSC 比賽影片)**
    
    - 沒有官方標註，但許多研究者利用 IFSC 國際攀岩賽事影片，做 **速度分析**、**關鍵事件 (start / top)** 偵測。
        
    - 可以利用 **pose estimation (OpenPose / HRNet / ViTPose)** 自動生成 label。
        

---

### (B) 通用 Sports datasets 可轉移

- **Sports-1M / FineGym / Diving48**
    
    - 雖然沒有攀岩，但都是 **運動動作分類 (Action Detection)** 資料集，可以轉移學習。
        
    - 特別是 FineGym 與 Diving48 提供 **細粒度動作標註**，很適合借用方法。
        
- **PoseTrack / COCO Keypoints / MPII**
    
    - 提供人體姿態標註，可作為攀岩 keypoint 檢測的 backbone dataset。
        

---

## 🔹 2. 相關研究/Project

- **攀岩自動評分 (Climbing Performance Scoring)**
    
    - 用 action detection 判斷是否完成動作（如 top hold 觸碰）。
        
    - 用 pose estimation 分析「動作是否標準」。
        
- **自動生成攀岩 Beta (Beta = 動作路線策略)**
    
    - 輸入影片，分析使用者抓點順序、腳點使用，與專業選手做比對。
        
- **動作失敗預測 (Fall Prediction)**
    
    - 透過 keypoint + handhold detection，預測「此動作是否不穩」。
        
    - 對安全監控與教練分析很有價值。
        

---

## 🔹 3. 潛力 Ideas 💡

1. **多視角 Sensor Fusion**
    
    - 攀岩館通常有固定攝影機（側視 / 上視）。
        
    - 可以結合多視角 Pose Estimation → 3D pose reconstruction。
        
    - 更精確捕捉「身體重心偏移」與「三點固定」策略。
        
2. **Hold Detection + Pose Estimation 融合**
    
    - 利用 Object Detection 偵測「岩點 (hold)」位置。
        
    - 再結合 Pose Estimation 判斷「手/腳是否抓到/踩到該 hold」。
        
    - 可延伸出 **自動路線重播、Beta 建議系統**。
        
3. **速度攀爬 (Speed Climbing) 自動計時**
    
    - 利用 Action Detection 偵測起步 (離地) 與結束 (拍鈴)。
        
    - 提供比賽自動計時系統。
        
4. **動作品質分析 (Movement Quality)**
    
    - 分析關節角度 (elbow, knee) 與核心穩定性。
        
    - 可做 **教練建議系統**，類似「你的身體重心偏高，應該降低腰部」。
        
5. **AI Route Recommendation**
    
    - 使用者拍攝自己完成某條路線。
        
    - AI 自動生成 **相似難度** 的推薦路線。
        
6. **結合 Wearable IMU/手環數據**
    
    - 攀岩館使用攝影機 + 手環加速度計 → fusion → 更精確的「發力分析」。
        

---

## 🔹 4. 技術建議

- **Keypoint Detection**
    
    - 可直接使用 ViTPose / MMPose (支援自定義 dataset)。
        
- **Action Detection**
    
    - SlowFast, VideoMAE, Timesformer（適合 Video-based Action Recognition）。
        
- **Multi-task Learning**
    
    - Backbone: DINOv2 / DINOv3
        
    - Heads: (1) Pose estimation head (keypoints), (2) Action recognition head。
        

---

👉 結論：  
目前攀岩 (Bouldering / Speed / Top Rope) 的公開 dataset 不多，多數需要自己蒐集 (例如 IFSC YouTube)，但 **ClimbAlong** 和 **ClimbersPose** 可作為起點。  
潛力最大的方向是：**Pose Estimation + Hold Detection 融合**（自動分析 Beta）、**多視角 3D Pose Reconstruction**（專業動作分析）、以及 **動作失敗預測 (fall prediction)**（安全與訓練輔助）。