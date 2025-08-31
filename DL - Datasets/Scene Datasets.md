
如果要訓練 **2D / 3D scene analysis** 的 AI 模型（也就是帶有 **scene** 任務名稱的任務，例如 Scene Classification / Scene Parsing / Scene Graph / Scene Layout / Scene Generation 等），一定需要公開的 **open dataset** 來支持。下面我幫你分成 **2D scene analysis** 與 **3D scene analysis** 兩大類，整理目前主流且完整的開源數據集，並說明它們的特色、標註內容、應用場景。

---

# 📌 2D Scene Analysis 資料集

## 1. MIT Indoor67 (2010) – 室內場景分類

- **任務**：Scene Classification
- **內容**：67 類室內場景（臥室、廚房、實驗室…），每類約 80 張訓練、20 張測試。
- **應用**：最早用於室內場景分類 benchmark。
- **特點**：雖然規模較小，但仍然是經典分類資料集。
- **規模**：67 類別，共約 15,620 張圖像 [GitHub+1](https://github.com/TUTvision/ScanNet-EfficientPS?utm_source=chatgpt.com)[GitHub](https://github.com/wilys-json/indoor-scene-recognition?utm_source=chatgpt.com)[dspace.mit.edu+5Massachusetts Institute of Technology+5GitHub+5](https://web.mit.edu/torralba/www/indoor.html?utm_source=chatgpt.com)
- **下載方式**：直接從 MIT 官方頁面下載 (tar 檔約 2.4 GB)，包含訓練／測試分割 file lists [Massachusetts Institute of Technology](https://web.mit.edu/torralba/www/indoor.html?utm_source=chatgpt.com)
- **標註形式**：圖像歸類 label + 部分有 LabelMe 格式的分割標註 [GitHub+13Massachusetts Institute of Technology+13GitHub+13](https://web.mit.edu/torralba/www/indoor.html?utm_source=chatgpt.com)
- **特色**：經典室內場景分類資料集，適合入門與 baseline 建置。

---

## 2. Places365 (2017) – 大規模場景分類

- **任務**：Scene Classification
- **內容**：365 個場景類別，>1.8M 圖像（train），5k 驗證，328k 測試。
- **應用**：目前最大規模的 scene classification dataset。
- **特點**：涵蓋室內（臥室、辦公室、廚房…）與室外（街道、公園…）。
- **規模**：365 類，訓練集超過 1.8M 圖像
- **下載方式**：官方網站提供下載（未於搜尋結果中引用，但是常見開放資源，可自行查找）
- **特色**：涵蓋室內與室外多種場景，適合訓練大型分類模型。

---

## 3. ADE20K (2017, MIT CSAIL) – 場景解析 (Scene Parsing)

- **任務**：Scene Parsing / Semantic Segmentation
- **內容**：20k 訓練、2k 驗證、3k 測試圖像，150 個語義標籤。
- **應用**：最常用於場景語義分割 benchmark（Scene Parsing Challenge）。
- **特點**：像素級標註，室內與室外場景皆有。
- **規模**：訓練 20k、驗證 2k、測試 3k 張圖像，150 個語義類別
- **下載方式**：官方提供下載連結與註冊方式（可透過 Semantic‑Aware repo 進入） [kaggle.com+3Massachusetts Institute of Technology+3GitHub+3](https://web.mit.edu/torralba/www/indoor.html?utm_source=chatgpt.com)[arXiv](https://arxiv.org/abs/1702.04405?utm_source=chatgpt.com)[arXiv+4scan-net.org+4kaldir.vc.in.tum.de+4](https://www.scan-net.org/ScanNet/?utm_source=chatgpt.com)
- **特色**：非常細緻且多樣，包括室內／室外場景，是 segmentation 的主流 benchmark。

---

## 4. SUN RGB-D (2015) – 室內 RGB-D 場景理解

- **任務**：Scene Parsing (2D+Depth)
- **內容**：10k 張 RGB-D 圖像（深度由 Kinect、Structure Sensor 取得）。
- **標註**：語義分割、3D 物件框。
- **應用**：常用於室內 segmentation 與 3D layout estimation。
- **特點**：包含 RGB 影像 + 深度圖，是 2D→3D 過渡的重要 dataset。
- **規模**：約 10k 張 RGB-D 圖像
- **下載方式**：官方網站下載（搜尋結果中未直接列出，但可透過 SUN RGB‑D 正式網站取得）
- **特色**：RGB + 深度，支持 2D 語義 + 3D 拓展任務。

---

## 5. Visual Genome (2017) – 場景圖生成

- **任務**：Scene Graph Generation
- **內容**：108k 圖像，>1.7M 物件，2.3M 關係 (subject–predicate–object)。
- **應用**：場景圖生成（Scene Graph Generation）、關係檢測。
- **特點**：每張圖有豐富的物件和關係標註，例如 `(sofa, next to, table)`。
- **規模**：108k 圖像，包含 1.7M 物件與 2.3M 關係三元組 [kaldir.vc.in.tum.de+1](https://kaldir.vc.in.tum.de/scannetpp/documentation?utm_source=chatgpt.com)[arXiv+4kaldir.vc.in.tum.de+4arXiv+4](https://kaldir.vc.in.tum.de/scannetpp/?utm_source=chatgpt.com)
- **下載方式**：官方網站或 GitHub repository 提供，需申請並下載。
- **特色**：豐富物件與關係標註，是 scene graph 任務的主流資料集。

---

## 6. Cityscapes (2016) – 城市街景解析

- **任務**：Scene Parsing / Panoptic Segmentation
- **內容**：5k 精細標註影像，20k 粗標註，30 類別（人、車、道路、建築…）。
- **應用**：城市自駕場景 segmentation。
- **特點**：專注於城市 outdoor scene。

---

# 📌 3D Scene Analysis 資料集

## 1. NYUv2 (2012) – 室內 RGB-D

- **任務**：3D Scene Parsing, Depth Estimation, Layout Estimation
- **內容**：1.4k RGB-D 室內圖像，1449 張精細標註 segmentation。
- **特點**：早期室內場景資料集，仍是深度估計與場景解析基準。
- - **規模**：約 1,449 張 RGB-D 圖像 [W](https://wjiajie.github.io/contents/datasets/scannet/?utm_source=chatgpt.com)
- **下載方式**：官方網站提供下載（常見於 akademik 分享，這裡提供推參）
- **特色**：早期標準室內 RGB-D dataset，適合入門基礎 3D scene tasks。

---

## 2. ScanNet (2017, Stanford) – 室內 3D 場景重建

- **任務**：3D Scene Parsing / Semantic Segmentation / 3D Reconstruction
- **內容**：>2.5M RGB-D 影像（來自 1.5k 室內場景），提供 3D mesh 與語義標註。
- **應用**：最常用的室內 3D segmentation benchmark。
- **特點**：提供原始影片、深度、3D mesh，適合 2D-3D 多模態研究。
- - **規模**：1513 個場景，共 2.5M RGB-D 影像，含相機姿態、表面重建與語義分割 [GitHub+6arXiv+6kaldir.vc.in.tum.de+6](https://arxiv.org/abs/1702.04405?utm_source=chatgpt.com)
- **下載方式**：需註冊並同意條款後，從官方網站下載 [scan-net.org](https://www.scan-net.org/ScanNet/?utm_source=chatgpt.com)[arXiv](https://arxiv.org/abs/1702.04405?utm_source=chatgpt.com)
- **特色**：目前室內 3D 任務中最受歡迎的 dataset，涵蓋重建與 segmentation。

---

## 3. Matterport3D (2017) – 室內 3D 掃描

- **任務**：3D Scene Understanding, Layout Estimation
- **內容**：90 棟建築物（住宅/商業空間），10k 室內 panorama，RGB-D。
- **應用**：常用於室內導航、3D 場景理解。
- **特點**：提供高品質全景與 3D annotation。
- **規模**：90 棟建築、10k 全景 RGB-D 圖像
- **下載方式**：官方網站下載，需填表申請。
- **特色**：提供全景視角，適合導航與全場景理解模型訓練。

---

## 4. Replica (2019, Facebook AI) – 高真實感室內 3D 資料集

- **任務**：3D Scene Parsing, Scene Generation, Simulation
- **內容**：高精度 3D 室內模型（mesh, texture）。
- **應用**：VR/AR 模擬、機器人場景理解。
- **特點**：高真實感、可直接用於模擬與 reinforcement learning。
- - **規模**：高精度室內 mesh + texture
- **下載方式**：官方發佈頁面可下載，常用於模擬。
- **特色**：高品質 Unreal‐style 模型，適合 VR/AR 或 RL 模擬。

---

## 5. 3D-FRONT (2020) – 大規模合成室內場景

- **任務**：3D Scene Generation, Layout Estimation
- **內容**：>18k 合成室內場景（房間），提供結構化 3D 家具與佈局。
- **應用**：常用於室內場景生成與 3D synthesis。
- **特點**：合成資料，適合訓練生成模型。
- - **規模**：>18k 合成場景，含結構化家具與佈局資訊
- **下載方式**：官方網站註冊後下載
- **特色**：合成資料，易於訓練 generative 模型或佈局推理。

---

## 6. Structured3D (2020) – 室內結構化資料集

- **任務**：3D Scene Parsing / Layout Estimation
- **內容**：3.5k 室內場景，196k RGB 影像，帶有房間幾何結構標註。
- **應用**：房間結構估計、室內重建。
- **特點**：結構化標註（牆、門、窗、地板）。
- **規模**：3.5k 場景、196k RGB 圖像，含幾何結構標註
- **下載方式**：官方網站可下載
- **特色**：標註詳盡，適合訓練佈局與解析模型。

---

# 📊 總覽表（2D vs 3D Scene Analysis 資料集）

|類別|資料集|年份|規模|任務|特點|
|---|---|---|---|---|---|
|**2D Scene Classification**|MIT Indoor67|2010|67 類, ~15k 圖|Scene Classification|室內場景分類經典|
||Places365|2017|365 類, 1.8M 圖|Scene Classification|最大規模場景分類|
|**2D Scene Parsing**|ADE20K|2017|25k 圖, 150 類|Scene Parsing|像素級語義分割|
||SUN RGB-D|2015|10k RGB-D|Scene Parsing|室內 RGB-D segmentation|
|**2D Scene Graph**|Visual Genome|2017|108k 圖|Scene Graph|關係標註豐富|
|**2D City Scene**|Cityscapes|2016|25k 圖|Scene Parsing|城市街景 segmentation|
|**3D Scene Parsing**|NYUv2|2012|1.4k 圖|3D Parsing, Depth|室內 RGB-D|
||ScanNet|2017|2.5M 圖, 1.5k 場景|3D Parsing, Segmentation|最大規模室內 3D|
||Matterport3D|2017|90 棟建築, 10k 全景|3D Layout, Parsing|全景高品質|
||Replica|2019|高精度 3D 模型|3D Scene Parsing|VR/AR 模擬用|
||3D-FRONT|2020|18k 合成場景|3D Generation|合成數據, 生成模型|
||Structured3D|2020|3.5k 場景, 196k 圖|3D Layout Estimation|房間幾何結構|

---

✅ **總結**

- **2D Scene Analysis** → MIT Indoor67 / Places365（分類），ADE20K / SUN RGB-D / Cityscapes（解析），Visual Genome（場景圖）。
- **3D Scene Analysis** → NYUv2（經典 RGB-D），ScanNet（最常用室內 3D segmentation），Matterport3D（全景高品質），Replica（模擬），3D-FRONT 與 Structured3D（合成 + 結構化）。
- **用途**：2D 偏向語義與結構理解，3D 偏向幾何、佈局與重建。