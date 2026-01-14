

|                         |     |
| ----------------------- | --- |
| [[#### lume V2 task解釋]] |     |
|                         |     |
|                         |     |
|                         |     |

#### lume V2 task解釋
```
請詳細解釋在系統的image process tasks(主要是lume_algo.py)關於image裡面有lume的分析除了用unet得到segmentation mask還有其他的output嗎? 所以執c行完這個cli model跟server mode的output各是甚麼?
```
根據您提供的程式碼，特別是 `tasks/task_algos/lume_algo.py`，系統在執行 Lume 分析時，**除了 U-Net 產生的 Segmentation Mask 之外，還有進行幾何形狀分析 (Geometry Analysis)**。

以下詳細解釋它的 Output 內容以及 CLI 與 Server 模式的差異。

---

### 1. Lume Analysis (`lume_algo.py`) 的詳細 Output

在 `process_single_image` 函式中，除了產生 Mask，還執行了以下步驟：

1. **輪廓提取**：從 Mask 中找出輪廓 (`cv2.findContours`)。
    
2. **多邊形擬合**：將輪廓簡化為多邊形 (`cv2.approxPolyDP`)。
    
3. **線段與角點分析**：計算線段長度與夾角。
    

因此，它的 Output 包含以下三類：

#### A. 實體檔案 (儲存在 `output_dir`)

程式會將結果寫入硬碟，檔名會加上原本圖片的名稱作為前綴：

1. **Mask 圖片** (`{filename}_mask_lume.png`):
    
    - 這就是您提到的由 U-Net 預測出的二值化遮罩。
        
2. **幾何視覺化圖片** (`{filename}_geometry_vis.jpg`):
    
    - 這是將分析結果畫在原圖上的圖片。
        
    - **綠色線條**：代表偵測到的線段 (Segments)。
        
    - **紅色圓點**：代表偵測到的直角角點 (Corners, 90度 ± 容許值)。
        
3. **JSON 報告檔** (`{filename}_lume_report.json`):
    
    - 包含完整的分析數據 (線段座標、長度、角度)。
        

#### B. 數據結構 (Python Dictionary / JSON)

這份數據是演算法的核心產出，包含以下欄位：

- **`segments` (線段列表)**: 每一段擬合出的直線。
    
    - `type`: "line"
        
    - `start`: [x, y] (起點座標)
        
    - `end`: [x, y] (終點座標)
        
    - `length`: (線段長度，像素單位)
        
- **`corners` (角點列表)**: 線段之間形成的夾角。
    
    - `location`: [x, y] (角點座標)
        
    - `angle`: (角度值，例如 89.5 度)
        
    - `type`: "right_angle" (如果是直角)
        

---

### 2. CLI Mode vs. Server Mode 的 Output 差異

雖然兩者**底層邏輯完全相同** (都呼叫 `LumeProcessor` 類別)，產生的**實體檔案也完全一樣**，但它們將結果「回傳給呼叫端」的方式不同：

#### CLI Mode (`main_cli.py` -> `run_lume_cli.py`)

這是透過 `subprocess` 執行的。

- **觸發點**: `core/orchestrator.py` 執行 `python tasks/cli_wrappers/run_lume_cli.py ...`。
    
- **Output 方式**: **Standard Output (stdout)**。
    
    - CLI 腳本最後一行執行了 `print(json.dumps(result, cls=NumpyEncoder))`。
        
    - `Orchestrator` 會去抓取這串被 print 出來的文字，並嘗試用 `json.loads()` 解析它。
        
- **回傳給系統的資料**:
    
    JSON
    
    ```
    {
      "image_path": "...",
      "segments": [ ... ],
      "corners": [ ... ],
      "status": "success"
    }
    ```
    

#### Server Mode (`workflow_server.py` -> `server_lume.py`)

這是透過 HTTP API 執行的 (常駐服務)。

- **觸發點**: `core/orchestrator.py` 發送 HTTP POST 請求到 `http://127.0.0.1:5002/process`。
    
- **Output 方式**: **HTTP Response Body**。
    
    - FastAPI 會將 Python Dictionary 自動序列化為 JSON 回傳。
        
- **回傳給系統的資料**:
    
    - 與 CLI 完全相同的 JSON 結構。
        

### 總結

|**特徵**|**Lume Analysis Output**|
|---|---|
|**影像產出**|1. Segmentation Mask (`.png`)<br><br>  <br><br>2. 幾何標註圖 (`_geometry_vis.jpg`)|
|**數據產出**|**Segments**: 線段的起終點座標與長度。<br><br>  <br><br>**Corners**: 角點的位置與角度 (主要用於檢測直角特徵)。|
|**CLI 執行結果**|將上述數據印在螢幕 (stdout) 供主程式抓取，並寫入實體檔案。|
|**Server 執行結果**|將上述數據透過 HTTP JSON 回傳，並寫入實體檔案。|

**結論：** 無論是用 CLI 還是 Server 模式，**產出的檔案和分析數據是一模一樣的**。差異僅在於系統是如何「拿到」這份數據（讀取 print 輸出 vs 讀取 API 回傳）。