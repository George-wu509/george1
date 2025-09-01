
## 2D 視覺檢測（圖像／表面缺陷）

### 1. MVTec Anomaly Detection（MVTec AD）

- **任務內容**：工業表面異常檢測（anomaly detection）。
- **應用範圍**：瑕疵檢測，如 PCB、金屬表面、食物包裝、膠囊。
- **特色**：共有 15 類別的高解析度影像，每類包含無缺陷訓練樣本，以及混有缺陷的測試樣本；適用於評估異常偵測模型在檢測圖像級與像素級瑕疵表現。[picsellia.com](https://www.picsellia.com/post/manufacturing-datasets?utm_source=chatgpt.com)
- **規模**：約 5,000 張高解析度影像，涵蓋多種材質與類別。[picsellia.com](https://www.picsellia.com/post/manufacturing-datasets?utm_source=chatgpt.com)
- **下載方式**：可於 MVTec 官方網站或透過相關開源平台取得。
- **標註形式**：
    - 訓練集：僅包含「正常（defect-free）」影像；
    - 測試集：每張影像含異常位置的像素級 mask，用於 segmentation 或 scoring。

---

### 2. VISION Datasets

- **任務內容**：視覺基礎工業檢測 benchmark，涵蓋多種缺陷檢測任務。
- **應用範圍**：缺陷分類與定位、工業品質控制與自動檢測。
- **特色**：集合 14 個不同工業場景的資料集，共計 18,000 張圖像與 44 種 defect 類型，並提供 instance-segmentation 級標註。[arXiv+1](https://arxiv.org/abs/2505.03412?utm_source=chatgpt.com)[arXiv+4picsellia.com+4kaggle.com+4](https://www.picsellia.com/post/manufacturing-datasets?utm_source=chatgpt.com)[arXiv](https://arxiv.org/abs/2306.07890?utm_source=chatgpt.com)
- **規模**：總計 18,000 張含缺陷與正常樣本影像。[arXiv](https://arxiv.org/abs/2306.07890?utm_source=chatgpt.com)
- **下載方式**：通常附於論文中或配套 GitHub／挑戰賽專頁提供下載連結。
- **標註形式**：
    - 像素級 mask，用於異常分割與實例定位。

---

### 3. CXR-AD: Component X-ray Anomaly Detection

- **任務內容**：X 光影像下工業零件內部異常檢測。
- **應用範圍**：內部缺陷偵測（如隱藏裂縫、氣泡、材質漏洞）。
- **特色**：目前首個公開工業零件 X 光影像資料集，五類零件共 653 張正常影像與 561 張含缺陷影像，並提供 pixel-level 的 defect mask。[arXiv](https://arxiv.org/abs/2505.03412?utm_source=chatgpt.com)
- **規模**：總計 1,214 張圖（653 正常 + 561 缺陷）。[arXiv](https://arxiv.org/abs/2505.03412?utm_source=chatgpt.com)
- **下載方式**：論文中通常附有下載連結或授權方式，可透過 arXiv 文章查詢。
- **標註形式**：
    - 異常像素級 segmentation mask。

---

### 4. 工業機具元件表面缺陷資料集（Industrial Machine Tool Component Surface Defect Dataset）

- **任務內容**：機具元件表面缺陷分類與檢測。
- **應用範圍**：金屬工件、機器零件畢結構疲勞、機械磨損邊界檢測。
- **特色**：提倡真實工業場景標準資料，適用於模型分類與 wear prognostics 建模。[picsellia.com](https://www.picsellia.com/post/manufacturing-datasets?utm_source=chatgpt.com)[arXiv](https://arxiv.org/abs/2103.13003?utm_source=chatgpt.com)
- **規模**：未列明具體數量，需透過 DOI 或原始資料來源查詢。[arXiv](https://arxiv.org/abs/2103.13003?utm_source=chatgpt.com)
- **下載方式**：透過 DOI（如 [https://doi.org/10.5445/IR/1000129520）進行資料存取。](https://doi.org/10.5445/IR/1000129520%EF%BC%89%E9%80%B2%E8%A1%8C%E8%B3%87%E6%96%99%E5%AD%98%E5%8F%96%E3%80%82)
- **標註形式**：
    - 表面缺陷分類 label，可能附有定位資訊。

---

## 其他工業資料類型（聲音／異常監控）

### 5. MIMII Dataset（Malfunctioning Industrial Machine Investigation and Inspection）

- **任務內容**：工業機器異常聲音偵測與分類。
- **應用範圍**：泵浦、風扇、閥門等機器故障預警與聲音監測。
- **特色**：提供正常與異常情境下的聲音錄音，模擬真實工廠場景，非常適合音訊辨識與 anomaly detection。[kaggle.com+3arXiv+3bigdata-ai.fraunhofer.de+3](https://arxiv.org/abs/2103.13003?utm_source=chatgpt.com)[kaggle.com+1](https://www.kaggle.com/datasets/yidazhang07/bridge-cracks-image?utm_source=chatgpt.com)
- **規模**：涵蓋多種類型機器與多種故障聲響錄音，具體數量請參考官方篇章。
- **下載方式**：資料可自 Zenodo 平臺免費下載。[arXiv](https://arxiv.org/abs/1909.09347?utm_source=chatgpt.com)
- **標註形式**：
    - 毫秒級音訊檔案（WAV），每段標註是否異常與異常類型。

---

## 總覽表：智慧製造與工業檢測資料集

|資料集名稱|任務內容|應用場景|特點|規模|下載方式|標註形式|
|---|---|---|---|---|---|---|
|MVTec AD|表面異常檢測|異常影像檢測 → pixel mask|高解析度、多材質、多類別（15 類）|約 5,000 張影像|MVTec 官方網站／平台|正常影像 + 異常 mask|
|VISION Datasets|工業異常 / 缺陷檢測|多場景、多 defect type|14 資料集、44 種 defect|18,000 張影像|論文／GitHub／挑戰賽網站|instance-segmentation mask|
|CXR-AD|工業零件 X 光異常|內部缺陷檢測（X 光）|首個公開 X 光異常資料|正常 653 + 缺陷 561 張|arXiv 論文提供下載|pixel-level mask|
|工具元件表面缺陷集|表面缺陷分類|機具疲勞／磨損檢測|真實場景資料|規模未明，需查 DOI|DOI 登錄下載|defect label (分類)|
|MIMII Dataset|聲音異常偵測|機器故障預警|音訊檢測異常資料|多種類機器錄音|Zenodo 公開下載|WAV + 異常標註|