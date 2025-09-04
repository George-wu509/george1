


|                                            |     |
| ------------------------------------------ | --- |
| [[#### Python library 是否應該要有 `setup.py`?]] |     |
| [[#### Python函式庫的結構設計]]                    |     |
|                                            |     |



#### Python library 是否應該要有 `setup.py`?

**簡短回答**：在現代 Python 專案中，`pyproject.toml` 已經很大程度上取代了 `setup.py` 的必要性。對您的專案來說，**`pyproject.toml` 已經足夠**。

**詳細說明**： Python 的套件打包生態系統近年來有了很大的演進。`pyproject.toml` 是 PEP 518 中引入的標準，用於聲明專案的建構相依性 (build dependencies)。現在，它已經擴展為儲存專案元數據 (metadata)、相依性 (dependencies) 和工具設定的主要位置。

- **`pyproject.toml` (現代標準)**:
    
    - 這是一個靜態的、宣告式的設定檔。
        
    - 它告訴 `pip` 和其他建構工具 (如 `build`) 如何打包您的專案。
        
    - 幾乎所有的專案資訊，包括名稱、版本、相依性等，都可以定義在這裡。
        
- **`setup.py` (傳統方式)**:
    
    - 這是一個可執行的 Python 腳本。
        
    - 在過去，這是定義所有套件資訊的唯一方式。
        
    - 它的動態性（因為是程式碼）雖然靈活，但也可能導致建構過程不夠可靠。
        
    - 在某些非常複雜的情況下（例如，需要編譯 C++ 擴展且邏輯複雜），`setup.py` 仍然是必要的，但它會與 `pyproject.toml` 一起使用。
        

**結論**：對於 `FastTrack` 這個純 Python 函式庫，`pyproject.toml` 是更現代、更簡潔、更標準的選擇。您不需要額外建立 `setup.py` 檔案。

#### 2. `environment.yml` 是否應該是 `environment.yaml`?

兩者都是可以的。`.yml` 和 `.yaml` 都是 YAML 檔案的有效副檔名。

- **`.yml`**: 是更常見的縮寫，尤其在 Conda 和 Docker Compose 的社群中，大多數文件和範例都使用 `.yml`。它更簡潔。
    
- **`.yaml`**: 是官方推薦的副檔名，更具描述性。





#### Python函式庫的結構設計

##### 專案佈局與理念

我們將遵循Python社群的最佳實踐，構建一個現代化且穩健的專案結構。

- **`src`佈局**：我們將強制使用`src`佈局（`src/pedestrian_intent/`），這是Python打包官方（PyPA）和許多專家推薦的結構 。這種佈局將源碼與測試、文檔等明確分離，可以避免潛在的導入路徑問題，並確保測試是針對已安裝的套件執行的。  
- **目錄結構**：將提供一個詳細的檔案系統樹狀圖，參考標準的機器學習專案模板 ，包含  
    `data/`（用於下載模型）、`docs/`、`notebooks/`、`scripts/`（用於訓練、轉換等）、`src/`和`tests/`等目錄。

##### 使用`pyproject.toml`與Poetry進行依賴管理

- 我們將提供一個完整的`pyproject.toml`文件 。  
- 該文件將定義專案的元數據（名稱、版本、作者）、核心依賴項（`torch`, `opencv-python`, `mmpose`, `transformers`等）以及開發依賴項（`pytest`, `black`, `ruff`）。  
- 我們將使用Poetry進行依賴解析和虛擬環境管理，並提供如`poetry install`和`poetry run`等常用指令 。
- Example:
```toml
# pyproject.toml

[tool.poetry]
name = "pedestrian-intent"
version = "0.1.0"
description = "A multi-modal library for pedestrian crossing intention prediction."
authors =
license = "Apache-2.0"
readme = "README.md"

[tool.poetry.dependencies]
python = "^3.9"
torch = "^2.1.0"
torchvision = "^0.16.0"
opencv-python-headless = "^4.8.0"
numpy = "^1.26.0"
mmpose = "^1.2.0"
mmcv = "^2.1.0"
mmdet = "^3.2.0"
transformers = "^4.35.0"
segment-anything = "^1.0"
matplotlib = "^3.8.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"
black = "^23.10.0"
ruff = "^0.1.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

##### 定義公共API

函式庫的接口應設計得簡潔直觀。可以用一個主要的`Pipeline`類來封裝整個處理流程。
Example
```python
# 預期使用範例
from pedestrian_intent import Pipeline

# 初始化管道，可從配置文件載入模型路徑等
pipeline = Pipeline(config_file="configs/default_pipeline.yaml")

# 處理影片並獲取結果
results = pipeline.process_video("path/to/video.mp4")

# results 將是一個結構化的數據對象，包含所有幀、所有行人的特徵和預測
```

**`pedestrian_intent`函式庫API結構提案**

|模組路徑 (Module Path)|類別/函數 (Class/Function)|職責 (Responsibility)|
|---|---|---|
|`pedestrian_intent.pipeline`|`Pipeline`|主要的用戶接口，整合所有模組，執行端到端的推論|
|`pedestrian_intent.modules.detection`|`YOLOv8Detector`|實現物體偵測功能|
|`pedestrian_intent.modules.pose`|`MMPoseEstimator`|實現全身姿態估計功能|
|`pedestrian_intent.modules.gaze`|`ETHGazeEstimator`|實現凝視估計功能|
|`pedestrian_intent.modules.scene`|`SAMClipSegmenter`|實現零樣本語義分割功能|
|`pedestrian_intent.fusion`|`TransformerIntentionModel`|核心的多模態融合與預測模型|
|`pedestrian_intent.visualization`|`Visualizer`|負責將所有結果繪製到圖像或影片上|
|`pedestrian_intent.utils`|`*`|包含各種輔助函數，如座標轉換、數據加載等|