
## 一、動作檢測（Action Detection / Recognition）相關資料集

### **UCF-101**

- **任務內容**：人類動作分類（單段影片內的動作辨識）。
- **應用場景**：影片中辨識特定動作，如運動、日常行為等。
- **特色**：涵蓋 101 類動作、13,000 多個片段、總長約 27 小時，影片取自網路、內容多元 [kaggle.com](https://www.kaggle.com/code/jyotiprasadpal/dp-proj-action-recognition-using-pose-estimation?utm_source=chatgpt.com)[維基百科](https://en.wikipedia.org/wiki/Activity_recognition?utm_source=chatgpt.com)。
- **規模**：101 類動作、13,000+ 片段。
- **下載方式**：Kaggle 或官方網站搜尋「UCF‑101 dataset」下載。
- **標註形式**：每段影片一個動作 label。

### **HMDB51**

- **任務內容**：動作分類。
- **應用場景**：短影片分類，例如「跳躍」「擁抱」「笑」等動作識別。
- **特色**：收集自電影與網路影片，共 51 類、6,849 個短片，每類至少 101 片段 [維基百科](https://en.wikipedia.org/wiki/Activity_recognition?utm_source=chatgpt.com)。
- **規模**：6,849 影片片段。
- **下載方式**：搜尋 “HMDB51 dataset” 取得下載方式。
- **標註形式**：每片段一個動作 label。

### **Kinetics**

- **任務內容**：動作分類。
- **應用場景**：大規模影片動作辨識。
- **特色**：由 DeepMind 建立，含 400 類動作、每類至少 400 片段，片段長約 10 秒 [arXiv+15維基百科+15arXiv+15](https://en.wikipedia.org/wiki/Activity_recognition?utm_source=chatgpt.com)[kaggle.com+1](https://www.kaggle.com/datasets/meetnagadia/human-action-recognition-har-dataset?utm_source=chatgpt.com)。
- **規模**：至少 160,000 片段。
- **下載方式**：DeepMind 官方或 YouTube API 下載。
- **標註形式**：每片段一動作 label。

### **AVA – Kinetics Localized Actions**

- **任務內容**：動作定位（Action Detection）與分類，在影像的特定框中辨識動作。
- **應用場景**：視頻中人物的動態行為定位與分類。
- **特色**：從 Kinetics-700 處理得來，每張關鍵幀上有 box + 多達 80 類動作標註，含 1.6M 註釋 [維基百科](https://en.wikipedia.org/wiki/List_of_datasets_in_computer_vision_and_image_processing?utm_source=chatgpt.com)。
- **規模**：238,906 片段、624,430 幅關鍵影像。
- **下載方式**：使用 AVA 官方頁面或 Google Research 提供的資料庫下載。
- **標註形式**：bbox + 動作類別 per 人物 per 幀。

### **IKEA ASM Dataset**

- **任務內容**：動作辨識（人組裝家具時）＋姿態與物件分割。
- **應用場景**：家具組裝時的動作理解、動作與物體交互分析。
- **特色**：3 百萬幀多視角影片，具 RGB、深度、動作原子分類（atomic actions）、物件 segmentation 與人類 pose 標註 [維基百科](https://en.wikipedia.org/wiki/List_of_datasets_in_computer_vision_and_image_processing?utm_source=chatgpt.com)[arXiv](https://arxiv.org/abs/2007.00394?utm_source=chatgpt.com)。
- **規模**：超過 3 百萬幀，多視角。
- **下載方式**：論文之 arXiv 或 GitHub 中應有下載連結。
- **標註形式**：每視角帧皆具動作 label、物件分割 mask、人體關鍵點。

---

## 二、姿態估計（Pose Estimation）相關資料集

### **COCO – keypoints**

- **任務內容**：2D 人體關鍵點姿態估計。
- **應用場景**：公共空間中的人形姿態辨識。
- **特色**：330,000+ 張圖片中，有 250,000 張含 17 關鍵點標註，註釋精細且開源廣泛使用 [encord.com](https://encord.com/blog/15-best-free-pose-estimation-datasets/?utm_source=chatgpt.com)。
- **規模**：至少 250,000 張標註圖。
- **下載方式**：COCO 官方網站。
- **標註形式**：每人 17 個關鍵點座標 + 可見性。

### **Human3.6M**

- **任務內容**：3D 姿態估計（RGB 圖像＋3D joint 標註）。
- **應用場景**：室內場景、高解析度 3D 姿態研究。
- **特色**：360 萬 3D 姿態和對應圖片，24 個關節標註，演員在 17 場景中做動作 [維基百科+4encord.com+4openreview.net+4](https://encord.com/blog/15-best-free-pose-estimation-datasets/?utm_source=chatgpt.com)[rose1.ntu.edu.sg](https://rose1.ntu.edu.sg/dataset/actionRecognition/?utm_source=chatgpt.com)。
- **規模**：3.6M frames。
- **下載方式**：Human3.6M 官方網站取得。
- **標註形式**：每張影像對應 24 關節 3D 座標。

### **SportsPose**

- **任務內容**：動態運動中的 3D 姿態估計。
- **應用場景**：運動分析、傷害預防、運動技術評估。
- **特色**：176,000+ 個 3D 姿態，24 人、5 種運動場景，姿態誤差約 34.5 mm [encord.com+3arXiv+3openreview.net+3](https://arxiv.org/abs/2304.01865?utm_source=chatgpt.com)。
- **規模**：176,000+ 個 3D 姿態。
- **下載方式**：ArXiv 頁面或專案網站內下載連結。
- **標註形式**：3D joint coordinates。

### **mRI – Multi-modal Rehabilitation Dataset**

- **任務內容**：3D 姿態估計與動作檢測。
- **應用場景**：復健動作分析、姿態監控。
- **特色**：超過 500 萬幀，提供 RGB-D、mmWave (雷達)、慣性感測模組（IMU），專門用於 HPE 與動作檢測 benchmark [arXiv+8dl.acm.org+8openreview.net+8](https://dl.acm.org/doi/10.1145/3664647.3681055?utm_source=chatgpt.com)[paperswithcode.com+1](https://paperswithcode.com/datasets?page=1&task=3d-pose-estimation&utm_source=chatgpt.com)[Sizhe An / Research Scientist+1](https://sizhean.github.io/mri?utm_source=chatgpt.com)。
- **規模**：5,000,000+ frames, 20 subjects。
- **下載方式**：mRI 官方網站（GitHub／論文頁）。
- **標註形式**：多模態資料 + 3D joint + 動作 label。

### **HMPEAR**

- **任務內容**：3D 姿態估計與動作辨識（HAR）。
- **應用場景**：戶外大場景中同步分析姿態與動作。
- **特色**：300,000+ frames，RGB + LiDAR + 3D poses（250K frames）、40 動作類型、6,000+ action clips [Sizhe An / Research Scientist](https://sizhean.github.io/mri?utm_source=chatgpt.com)[openreview.net+1](https://openreview.net/forum?id=hgRElsBV6v&referrer=%5Bthe+profile+of+Lan+Xu%5D%28%2Fprofile%3Fid%3D~Lan_Xu2%29&utm_source=chatgpt.com)。
- **規模**：300K+ frames，250K 帶 3D pose。
- **下載方式**：HmPEAR 論文 arXiv / OpenReview 附連下載。
- **標註形式**：RGB、LiDAR point clouds、3D poses、action label。

### **PoseTrack**

- **任務內容**：影片中多人姿態估計與追蹤。
- **應用場景**：社交動態、影片內容分析、多人體互動情境。
- **特色**：影片多人關鍵點估計 + 追蹤 ID，提供評測伺服器 [openreview.net+1](https://openreview.net/forum?id=hgRElsBV6v&referrer=%5Bthe+profile+of+Lan+Xu%5D%28%2Fprofile%3Fid%3D~Lan_Xu2%29&utm_source=chatgpt.com)[openreview.net+2openreview.net+2](https://openreview.net/pdf/8bcbd5f375932ca306a5b19acd66d7f6ed085ef2.pdf?utm_source=chatgpt.com)。
- **規模**：不定量大規模，影片形式，含多人姿態與追蹤 annotation。
- **下載方式**：PoseTrack 官方網站 [arXiv](https://arxiv.org/abs/1710.10000?utm_source=chatgpt.com)。
- **標註形式**：每幀多人關鍵點 + 身分 tracking ID。

### **NTU RGB-D (and 120)**

- **任務內容**：3D skeleton-based pose estimation + 動作辨識。
- **應用場景**：日常生活、人與動作情境分析。
- **特色**：60 類動作 / 57,600 片段 (NTU 120 延伸至 120 類、114,480 samples)，包含 RGB/Depth/3D skeleton/IR 數據 [rose1.ntu.edu.sg+1](https://rose1.ntu.edu.sg/dataset/actionRecognition/?utm_source=chatgpt.com)。
- **規模**：NTU: 56,880 samples；NTU120: 114,480 samples。
- **下載方式**：NTU 官方頁面申請下載。
- **標註形式**：RGB 影片、深度圖、3D 關節座標、IR 影像 + 動作類別 label。

### **JRDB-Pose**

- **任務內容**：多人姿態估計與追蹤，適用於社交導航場景。
- **應用場景**：機器人理解人群動態、多人體交互識別。
- **特色**：來自機器人視角的室內外場景影片，多人關鍵點 + occlusion 標註 + track ID [維基百科+1](https://en.wikipedia.org/wiki/NTU_RGB-D_dataset?utm_source=chatgpt.com)[rose1.ntu.edu.sg](https://rose1.ntu.edu.sg/dataset/actionRecognition/?utm_source=chatgpt.com)。
- **規模**：大型多人影片資料集。
- **下載方式**：JRDB 官方網站 [arXiv](https://arxiv.org/abs/2210.11940?utm_source=chatgpt.com)。
- **標註形式**：關鍵點 (含遮擋) + Track ID。

---

## 三、總覽表格：資料集一覽

|資料集名稱|任務內容|應用場景|特色|規模|下載方式|標註形式|
|---|---|---|---|---|---|---|
|UCF-101|動作分類|普通影片動作辨識|101 類、13k+ 片段|13k+ 影片片段|官方 / Kaggle|影片 + 動作 label|
|HMDB51|動作分類|日常動作辨識|51 類、6.8k 片段|6.8k+ 片段|官方網站|影片 + 動作 label|
|Kinetics|動作分類|網路影像動作辨識|400 類、400k+ 片段|≥160k 片段|DeepMind|影片 + 動作 label|
|AVA|動作定位 + 分類|影片內人物動作定位辨識|bbox + 80 類動作|624k 註釋關鍵幀|AVA 官方頁|bbox + 動作 label per 人物|
|IKEA ASM|動作 + Pose + Segmentation|家具組裝動作理解|3M 幀、多視角 + 多標註|3M+ 幀|論文 / GitHub|動作 label、物件 segmentation、人體關鍵點|
|COCO keypoints|2D 姿態估計|通用公共姿態辨識|250k 人臉關鍵點標註|≥250k 圖片|COCO 官方|17 point coordinates + 見可見性|
|Human3.6M|3D 姿態估計|室內高解析姿態估計|3.6M 3D pose + 圖片|3.6M frames|官方網站|24 joints 3D 坐標|
|SportsPose|3D 運動姿態估計|運動分析|真實運動、高動態姿態|176k 3D 姿態|論文 / 網站|3D joint coordinates|
|mRI|3D 姿態 + 動作檢測|復健監控|5M 多模態 frames|5M+ frames|論文 / GitHub|多模態資料 + 3D pose + 動作 label|
|HMPEAR|3D Pose + HAR 多模態|戶外姿態 + 動作理解|RGB + LiDAR + 40 動作|300k+ frames|arXiv / OpenReview|RGB + LiDAR + 3D pose + action label|
|PoseTrack|視頻多人姿態 + 追蹤|多人體互動動態分析|視頻格式 + 人與 tracking ID|大規模多人影片|PoseTrack 官方|關鍵點 + track ID|
|NTU RGB-D|Skeleton pose + 動作辨識|日常生活動作理解|RGB-D + skeleton + IR|56k / 114k samples|NTU 官方申請|RGB-D + 3D skeleton + action label|
|JRDB-Pose|多人姿態估計 + 追蹤|社交導航|機器人視角影片 + 遮擋標註|大規模影片資料集|JRDB 官方網站|關鍵點 (含遮擋) + track ID|