

以下是我為你整理的 **自駕車 (Autonomous Driving)** 相關 AI 模型訓練所需的主流、公開資料集清單，包含其任務內容、應用場景、資料特點、規模、下載方式與標註形式，並以**繁體中文**詳細說明。

---

## 主流公開資料集概覽

### 1. KITTI Vision Benchmark Suite

- **任務內容**：涵蓋包括立體視覺 (Stereo)、光流 (Optical Flow)、視覺里程計 (Visual Odometry)、2D/3D 物體偵測等多個自駕相關任務。
- **應用場景**：訓練與驗證車輛上的感知模型，如物件偵測、深度估計、位姿推估等。
- **資料特點**：真實城市街道、郊區與高速公路影像，支援多種任務的 benchmark。
- **規模**：訓練集中包含 7,481 張註記有 3D bounding boxes 的影像，整套資料超過百 GB [Repli5+1](https://www.repli5.com/post/ultimate-guide-top-free-autonomous-driving-datasets-for-computer-vision-2024?utm_source=chatgpt.com)[paperswithcode.com+4TensorFlow+4arXiv+4](https://www.tensorflow.org/datasets/catalog/kitti?utm_source=chatgpt.com)[cvlibs.net](https://www.cvlibs.net/datasets/kitti/?utm_source=chatgpt.com)。
- **下載方式**：前往 KITTI 官方網站，在各 benchmark 分頁下載相應模組與資料集 [cvlibs.net](https://www.cvlibs.net/datasets/kitti/?utm_source=chatgpt.com)。
- **標註形式**：
    - 單目影像 + 立體影像序列；
    - 2D bounding boxes；
    - 3D bounding boxes with orientation and dimensions；
    - 深度 Ground Truth（部分模組）。

---

### 2. Cityscapes

- **任務內容**：語義分割 (Semantic Segmentation)、實例分割 (Instance Segmentation)、全景分割 (Panoptic Segmentation)、2D 物件偵測等。
- **應用場景**：城市街景理解，車道標示、行人車流辨識、精細語義分割等。
- **資料特點**：來自 50 城市的街景，具高品質像素級標註與部分快速標註影像，並提供影像序列、立體資料、GPS 與車輛里程計資訊 [GitHub+14cityscapes-dataset.com+14TensorFlow+14](https://www.cityscapes-dataset.com/?utm_source=chatgpt.com)。
- **規模**：
    - 高品質標註影像：5,000 張；
    - 粗略標註影像：20,000 張 [GitHub](https://github.com/mcordts/cityscapesScripts?utm_source=chatgpt.com)[cityscapes-dataset.com](https://www.cityscapes-dataset.com/?utm_source=chatgpt.com)。
- **下載方式**：需註冊 Cityscapes 官方網站帳號並登入後下載，也可使用提供的下載腳本自動化完成 [GitHub](https://github.com/mcordts/cityscapesScripts?utm_source=chatgpt.com)。
- **標註形式**：
    - 每像素語義類別標籤、實例 ID；
    - 支援 panoptic segmentation；
    - 附加 stereo, GPS, odometry metadata。

---

### 3. nuScenes

- **任務內容**：3D 物件偵測與追蹤、多傳感器融合感知（包含車載 LiDAR、雷達、影像等）。
- **應用場景**：360° 環景物件偵測與追蹤、多感知融合、行人車輛行為分析等。
- **資料特點**：全球首個完整自駕感知套件，具備 6 鏡頭、5 雷達、1 LiDAR、360° 視角，註釋包含 23 種類別與 8 種屬性，且與 KITTI 相比註釋量 7 倍，影像數量多 100 倍 [TensorFlow+8cityscapes-dataset.com+8paperswithcode.com+8](https://www.cityscapes-dataset.com/?utm_source=chatgpt.com)[arXiv](https://arxiv.org/abs/1903.11027?utm_source=chatgpt.com)。
- **規模**：1,000 個場景，每場景長度約 20 秒。
- **下載方式**：前往 nuScenes 官方網站取得資料與開發套件 (devkit) [維基百科](https://en.wikipedia.org/wiki/List_of_datasets_in_computer_vision_and_image_processing?utm_source=chatgpt.com)[arXiv](https://arxiv.org/abs/1903.11027?utm_source=chatgpt.com)。
- **標註形式**：
    - 3D bounding boxes；
    - 物件分類與屬性標註；
    - 時間的 tracking ID。

---

### 4. Waymo Open Dataset

- **任務內容**：3D 物件偵測與追蹤 (LiDAR + camera)、2D/3D 感知任務。
- **應用場景**：提升科技園區、城市環境中自駕車的感知與追踪精度。
- **資料特點**：大量高品質 LiDAR 與影像資料，覆蓋都市與郊區，註釋包含 2D 影像與 3D LiDAR bounding box，並維持跨 frame 的一致 ID [TensorFlow+5arXiv+5arXiv+5](https://arxiv.org/abs/1912.04838?utm_source=chatgpt.com)。
- **規模**：1,150 個場景，每個場景約 20 秒，涵蓋多樣地理區域與場景特性。
- **下載方式**：訪問 Waymo Open 官方網站下載資料與工具 [cvlibs.net+15arXiv+15GitHub+15](https://arxiv.org/abs/1912.04838?utm_source=chatgpt.com)。
- **標註形式**：
    - 2D 與 3D bounding boxes；
    - 跟踪 ID 跨幀連續；
    - 高精度 LiDAR point clouds + 相機影像同步。

---

### 5. A2D2 (Audi Autonomous Driving Dataset)

- **任務內容**：多傳感器融合感知，包括 2D/3D segmentation、3D bounding boxes，以及 LiDAR + 相機融合。
- **應用場景**：複雜環境中的感知學習、多傳感器融合、多任務訓練與測試。
- **資料特點**：包含 6 鏡頭、5 LiDAR 設備，全方位資料；提供同步註釋：語義分割、實例分割、3D bounding boxes。且非註釋影像序列量大，可用於訓練自監督模型 https://arxiv.org/abs/2004.06320?utm_source=chatgpt.com
- **規模**：
    - 註釋幀：41,277 幀帶語義與點雲標註，其中 12,497 幀帶有 3D bounding boxes；
    - 未註釋幀：392,556 幀序列資料。
- **下載方式**：透過 Audi A2D2 官方網站下載，授權為 CC BY-ND 4.0，可用於研究與商業用途 [arXiv+1](https://arxiv.org/abs/2004.06320?utm_source=chatgpt.com)。
- **標註形式**：
    - 2D 語義分割;
    - 3D bounding boxes;
    - LiDAR 點雲分類與 segmentation。

---

### 6. Zenseact Open Dataset (ZOD)

- **任務內容**：長距離、多模態感知，包含 2D/3D 物件訓練、semantic segmentation、交通訊號識別等。
- **應用場景**：拓展自駕在遠距離感知與多任務學習能力。
- **資料特點**：涵蓋範圍廣、解析度高、有完整 sensor 套裝與長距離註釋（最多達 245m），支援空間—時間訓練。亦是少數對商業應用開放的資料集之一 [arXiv+1](https://arxiv.org/abs/2004.06320?utm_source=chatgpt.com)[arXiv](https://arxiv.org/abs/2305.02008?utm_source=chatgpt.com)。
- **規模**：
    - Frames：100,000 張已挑選過的相機影像；
    - Sequences：1,473 段完整 sensor 記錄 (約 20 秒)；
    - Drives：多段完整的幾分鐘路徑資料。
- **下載方式**：前往 Zenseact ZOD 官方網站下載資料與開發套件 [arXiv+1](https://arxiv.org/abs/2305.02008?utm_source=chatgpt.com)。
- **標註形式**：
    - 2D/3D 物件框；
    - Semantic segmentation；
    - 交通標誌與道路分類。

---

## 總覽表格

|資料集名稱|任務內容|應用場景|特點|規模|下載方式|標註形式|
|---|---|---|---|---|---|---|
|KITTI|立體視、光流、VO、2D/3D 偵測|多項視覺任務訓練|多任務 unified benchmark|約 7,481 幀 3D 標註|官方網站下載|2D/3D bounding, depth|
|Cityscapes|語義、實例、全景分割|城市街景理解|豐富 annotation + metadata|5k 精註 + 20k 粗註|註冊後下載或腳本下載|pixel-label, instance, stereo/metadata|
|nuScenes|3D 偵測與追蹤、多傳感融合|360° 全景物件追蹤|感知 sensor 全套註釋豐富|1,000 場景|官方網站與 devkit|3D bboxes, attributes, tracking ID|
|Waymo Open|2D/3D 偵測與追蹤|城市環境場景感知|高品質 LiDAR + camera，多樣地理區域|1,150 場景|官方下載|2D/3D bboxes, tracking ID|
|A2D2 (Audi)|語義/實例分割 + 3D 偵測 + 自監督序列|多模態融合訓練、長距離感知|影像 + LiDAR 應有盡有，標註豐富|41k 註釋幀 + 392k 序列|官方網站下載 (CC BY-ND 4.0)|2D/3D masks, bboxes, LiDAR|
|Zenseact ZOD|長距離 2D/3D 感知、交通符號、場景分類|長距離、多任務學習|高解析度、商業可用、sensor 套件齊全|100k 幀 + 1,473 段 + Drives|官方網站下載|2D/3D boxes, seg, signage|