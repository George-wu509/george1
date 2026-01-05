

|                                |     |
| ------------------------------ | --- |
| [[#### 新DynamoDB架構]]           |     |
| [[#### 三個DynamoDB的files跟DB內容]] |     |
|                                |     |


#### 新DynamoDB架構
```
如果我的系統是local computer system是連接著camera等hardware, 然後有兩套系統一套在LA, 一套在Germany. 而可以用多個remote client在local or 任何地方操控這些local computer systems. 那如果我們系統裡面的三個DynamoDB(WatchAnalysisResults, WatchCommandLog, WatchTemplates)都放在AWS cloud跟放在local computer system, 或者混和哪個比較適合呢? WatchTemplates管理了在camera take images的位置跟設定. WatchAnalysisResults則管理camera take images在local跟cloud的位置, 以及用image processing tasks的results在local跟cloud的位置. 也能方便搜尋並下載. 而WatchCommandLog則管理在loca, remote的所有操作指令. 以30年tech lead對怎麼配置三個DynamoDB提出建議並深入的分析
```

這是一個非常經典的 **Distributed System (分散式系統)** 與 **Edge Computing (邊緣運算)** 架構問題。

作為 Tech Lead，我的建議非常明確：**絕對不能選「純 Cloud」或「純 Local」，必須採用「混合架構 (Hybrid Architecture)」，也就是 "Cloud Native with Local First" (雲原生但在地優先) 的策略。**

原因很簡單：

1. **物理限制 (Physics)**：德國與 LA 到 AWS 機房（假設在 US-East-1）都有物理延遲 (Latency)。如果每一次快門、每一次馬達移動都要等 AWS 回應，你的系統會慢到無法接受。
    
2. **可用性 (Availability)**：網路一定會斷。如果網路斷了，德國工廠就停擺不能拍照了嗎？這在生產環境是不允許的。
    
3. **全域協作 (Global Collaboration)**：LA 開發好的 Template，德國要能馬上用；Remote Client 要能隨時監控兩地的狀態。
    

以下我針對這三個 DynamoDB Table 的特性，逐一分析它們應該**放在哪裡**以及**如何同步**。

---

### 系統架構總覽

我們將架構定義為：

- **Edge (Local Computer)**: 德國與 LA 的實體電腦。使用輕量級資料庫 (如 **SQLite**) 作為 Local Cache。
    
- **Cloud (AWS)**: 使用 **DynamoDB Global Tables** (若預算允許) 或單一 Region。作為 Single Source of Truth (唯一真理來源)。
    

### 三個 Table 的詳細配置建議

#### 1. `WatchTemplates` (拍攝腳本)

- **性質**：**Read-Heavy (多讀少寫)**。Template 設定好後很少改，但每次拍照都要讀。
    
- **配置建議**：**Cloud Master + Local Cache**
    
- **運作方式**：
    
    - **Cloud (AWS)**: 這是「Master」。所有的 Template 新增/修改都在這裡進行。
        
    - **Local (SQLite/JSON)**: 當 Local Computer 啟動（或透過 `force_sync` 指令）時，從 AWS 下載所有最新的 Templates 存到本地。
        
    - **執行時**：WorkflowManager **只讀取本地 Cache**。這樣確保了毫秒級的讀取速度，且斷網時依然能跑舊有的 Template。
        
- **為什麼這樣做？**
    
    - 避免每次跑 Step 1 到 Step 2 都要去雲端問「下一步是什麼？」，消除網路延遲造成的硬體頓挫感。
        

#### 2. `WatchAnalysisResults` (拍攝結果與索引)

- **性質**：**Write-Heavy (多寫少讀)**。每次拍照都會產生一筆，且伴隨著大檔案上傳 (S3)。
    
- **配置建議**：**Local Buffer -> Async Push to Cloud (非同步上傳)**
    
- **運作方式**：
    
    - **寫入 (Capture)**: 拍照當下，Metadata (路徑、參數) **先寫入 Local DB (SQLite)**。這保證了相機不用等雲端回應就能拍下一張。
        
    - **同步 (Background)**: 背景程式 (`CloudSyncManager`) 掃描 Local DB 中 `synced=False` 的紀錄，將檔案上傳 S3，成功後再將 Metadata 寫入 AWS DynamoDB。
        
    - **讀取 (Search)**:
        
        - 如果 Remote Client 要搜尋：直接查 AWS DynamoDB。
            
        - 如果 Local Computer 要回顧：優先查 Local DB。
            
- **為什麼這樣做？**
    
    - **高吞吐量 (Throughput)**：自動化檢測一秒可能拍好幾張，直接寫 DynamoDB 會成為效能瓶頸（甚至遇到 Throttling）。
        
    - **斷網保護**：網路斷了，工廠繼續拍，資料存在 Local。網路恢復後，背景程式自動補傳。
        

#### 3. `WatchCommandLog` (指令控制)

- **性質**：**Latency-Sensitive (對延遲極度敏感)**、**Transient (暫時性)**。指令做完就沒用了，主要是為了狀態同步。
    
- **配置建議**：**Cloud Broker (作為中介) + Local Polling/WebSocket**
    
- **運作方式**：
    
    - **Remote Client**：因為 Remote Client 可能在任何地方（家裡、咖啡廳），無法直接連到有防火牆保護的 Local Computer。所以 Client 必須把指令寫入 **AWS Cloud**。
        
    - **Local Computer (Agent)**：
        
        - **方法 A (標準 DynamoDB)**: 每 0.5 秒 Poll 一次 AWS DynamoDB 看有沒有新指令 (`state='submitted'`)。
            
        - **方法 B (進階 AWS IoT)**: 透過 MQTT Subscribe，雲端有指令會主動 Push 下來 (延遲更低)。
            
    - **Local Client (在機台旁)**：**例外處理！** 如果操作員就在電腦旁邊用 GUI，GUI 應該直接透過 `localhost` API 呼叫 Engine，**繞過 Cloud**。同時，Agent 會把這筆操作「補寫」回 Cloud Log，讓遠端的人知道「剛剛有人在本地操作了」。
        
- **為什麼這樣做？**
    
    - 解決了「內網穿透」的問題。Remote Client 不需要 VPN 就能控制德國的機器。
        

---

### 深入分析：Hybrid 架構的資料流圖

1. **Remote Client (e.g., in Taiwan)** sends "Start Routine" -> **AWS DynamoDB `WatchCommandLog`**.
    
2. **Local Computer (in Germany)** detects command -> Executes Routine.
    
3. **Local Computer** reads `WatchTemplates` from **Local Cache (SQLite)** (Synced from Cloud previously).
    
4. **Local Computer** captures image -> Saves to SSD -> Writes Metadata to **Local SQLite `WatchAnalysisResults`**.
    
5. **Background Thread** uploads image to **S3** & puts item to **AWS DynamoDB `WatchAnalysisResults`**.
    
6. **Remote Client** sees status update in **AWS**.
    

### 針對你的 Python 代碼的具體修改建議

你目前的 `CloudDatabaseManager` 與 `DataManager` 已經有很好的雛形，但需要明確區分 Local 與 Cloud 的職責。

#### 1. 修改 `DataManager` (Local Storage)

目前的 `DataManager` 裡面有 `self.db` (SQLite)，這是對的。要確保：

- `insert_raw_image`: 寫入 SQLite，標記 `synced=0`。
    
- `get_pending_uploads`: 從 SQLite 撈出 `synced=0` 的資料。
    

#### 2. 修改 `CloudDatabaseManager` (Cloud Storage)

這個 Class 專門負責跟 AWS 對話。

- `index_record`: 這應該由 `CloudSyncManager` 呼叫，而不是在拍照當下呼叫。
    

#### 3. 新增 `TemplateManager` 的同步邏輯 (Sync Logic)

你需要一個明確的同步機制。



#### 三個DynamoDB的files跟DB內容
```
請中文詳細解釋這個system跟DynamoDB相關的所有files包括DB本身, 設定或相關python files, 並中文解釋每個file跟其中的class, functions的作用, 並詳細解釋他們之間是如何合作執行DynamoDB以及跟cloud的關係. 並具體舉例說明. 並詳細描述三個DynamoDB一筆資料的內容
```

這是一個非常宏大的系統架構問題。作為 Tech Lead，我將為你詳細解構這個 **「混合雲端架構 (Hybrid Cloud Architecture)」** 中，DynamoDB 如何作為核心神經系統運作。

這個系統的設計哲學是：**「在地執行 (Local Execution)，雲端管理 (Cloud Management)，非同步同步 (Async Sync)」**。

---

### 第一部分：三個 DynamoDB Tables 的具體內容與結構

在 AWS DynamoDB 中，我們建立了三個 Table，分別對應系統的三大支柱：**配置 (Configuration)**、**數據 (Data)**、**控制 (Control)**。

#### 1. `WatchTemplates` (拍攝腳本庫)

這是一本「食譜」，定義了每種手錶系列該如何拍攝。

- **性質**：設定檔 (Configuration)。
    
- **讀寫模式**：雲端寫入 (管理員)，本地讀取 (機台)。
    
- **資料結構範例**：
    
    - **PK (Partition Key)**: `series_name` (e.g., `"Rolex_Submariner"`)
        
    - **SK (Sort Key)**: `version` (e.g., `"v1"`)
        
    - **Attributes (內容)**: 一個巨大的 JSON，包含 `views` (視角) -> `parts` (部位) -> `HardwareParameters` (馬達座標、曝光)。
        
    
    JSON
    
    ```
    {
      "series_name": "Rolex_Submariner",
      "version": "v1",
      "Brand": "Rolex",
      "views": {
        "Front": {
          "glasspoint": {
            "X": 50.5, "Y": 100.2, "Z": 30.0,
            "camused": "micro_cam_id",
            "exptimes": [5000, 15000] // HDR
          },
          "crown": { "X": 10.0, ... }
        }
      }
    }
    ```
    

#### 2. `WatchAnalysisResults` (數位雙生資料庫)

這是每支手錶的「履歷」，紀錄了它被拍攝的所有細節。採用 **Split-Key (分拆鍵)** 設計以避開 DynamoDB 400KB 限制。

- **性質**：歷史紀錄 (Immutable Record)。
    
- **讀寫模式**：本地寫入 (緩衝)，背景上傳雲端。
    
- **資料結構範例 (一支手錶會有多筆資料)**：
    
    - **Item 1 (基本資料)**:
        
        - **PK**: `"Rolex_Sub_SN12345"` (WatchID)
            
        - **SK**: `"INFO"`
            
        - **Attr**: `{"Brand": "Rolex", "Authenticity": "Real", "EntryTime": 170000...}`
            
    - **Item 2 (正面-鏡面照片紀錄)**:
        
        - **PK**: `"Rolex_Sub_SN12345"`
            
        - **SK**: `"VIEW#Front"`
            
        - **Attr**:
            
            JSON
            
            ```
            {
              "glasspoint": {
                "topID": "a1b2c3d4...", // 圖片的 Hex ID
                "s3_key": "Rolex/Raw/Front/glass.jpg",
                "X": 50.5, // 實際拍攝時的座標
                "status": "captured"
              }
            }
            ```
            
    - **Item 3 (材質分析)**:
        
        - **PK**: `"Rolex_Sub_SN12345"`
            
        - **SK**: `"MAT#Case-Body"`
            
        - **Attr**: `{"materialconcentration": [0.9, 0.05...], "alloytype": "904L"}`
            

#### 3. `WatchCommandLog` (指令中介站)

這是遠端控制的「信箱」。

- **性質**：指令佇列 (Command Queue)。
    
- **讀寫模式**：Remote Client 寫入，Local Agent 讀取並更新狀態。
    
- **資料結構範例**：
    
    - **PK**: `"Rolex_Station_001"` (DeviceID)
        
    - **SK**: `"cmd_uuid_5678"`
        
    - **Attributes**:
        
        - `action`: `"run_routine"`
            
        - `payload`: `{"watch_id": "Rolex_Sub_SN12345", "routine": "Rolex_Submariner"}`
            
        - `state`: `"submitted"` -> `"accepted"` -> `"completed"`
            

---

### 第二部分：相關檔案與其職責詳解

這些 Python 檔案共同構成了一個「資料管線 (Data Pipeline)」。

#### 1. 設定層

- **`config/system_config.yaml`**
    
    - **作用**：系統的總設定檔。
        
    - **DynamoDB 相關設定**：定義了上述三個 Table 的名稱 (`template_table`, `dynamodb_table`, `command_table`) 以及 AWS Region。這讓程式碼與特定 Table 解耦，方便測試環境與生產環境切換。
        

#### 2. 資料定義層 (Model Layer)

- **`core/models/shared_structures.py`**
    
    - **Class `HardwareParameters`**: 定義了「硬體怎麼動」的標準格式 (X, Y, Z, Exposure)。Template 用它來下令，Result 用它來紀錄。
        
- **`core/models/template_structure.py`**
    
    - **Class `WatchTemplate`**: 定義 `WatchTemplates` DB 的資料結構。
        
- **`core/models/watch_structure.py`**
    
    - **Class `WatchRecord`**: 定義 `WatchAnalysisResults` DB 的資料結構 (數位雙生)，包含 View, Material 等巢狀結構。
        

#### 3. 雲端介面層 (Cloud Interface)

- **`data_manager/cloud_db.py`**
    
    - **作用**：唯一有權限直接呼叫 AWS Boto3 SDK 連接 DynamoDB 的模組。
        
    - **關鍵 Functions**:
        
        - `scan_all_templates()`: 下載雲端所有腳本 (用於同步)。
            
        - `index_record(pk, sk, meta)`: 將一筆拍攝紀錄寫入 `WatchAnalysisResults`。
            
        - `put_command_state()`: 更新指令狀態 (例如從 submitted 變成 running)。
            
        - `put_item_split()`: 支援 Split-Key 寫入策略，把大物件拆成多筆 DynamoDB Items。
            

#### 4. 本地快取層 (Local Cache)

- **`DB/db_manager.py`**
    
    - **作用**：本地的 SQLite 資料庫。它是雲端的「緩衝區 (Buffer)」與「快取 (Cache)」。
        
    - **關鍵 Functions**:
        
        - `insert_raw_image(synced=0)`: 拍照瞬間呼叫此函式。資料**秒存**入 SSD，標記為未同步。這讓相機不用等網路。
            
        - `get_pending_uploads()`: 撈出所有 `synced=0` 的資料，準備上傳。
            
        - `save_local_template()`: 將從雲端下載的 Template 存入 SQLite，供斷網時使用。
            
        - `get_local_template()`: 執行時只讀這裡，確保毫秒級回應。
            

#### 5. 同步層 (The Bridge)

- **`data_manager/cloud_sync.py`**
    
    - **作用**：背景執行緒 (Background Thread)，負責將 Local 資料搬到 Cloud。
        
    - **關鍵 Logic (`_worker_loop`)**:
        
        1. 每隔幾秒呼叫 `db_manager.get_pending_uploads()`。
            
        2. 發現有未同步照片 -> 上傳 S3。
            
        3. 上傳成功 -> 呼叫 `cloud_db.index_record()` 寫入 DynamoDB。
            
        4. 寫入成功 -> 更新 Local DB `synced=1`。
            

#### 6. 業務邏輯層 (Business Logic)

- **`core/template_manager.py`**
    
    - **作用**：管理腳本的「大腦」。
        
    - **關鍵 Functions**:
        
        - `sync_from_cloud()`: 呼叫 `cloud_db` 下載 -> 呼叫 `db_manager` 存入 SQLite。
            
        - `get_template()`: **只從** `db_manager` (Local) 讀取。
            
        - `create_from_history()`: 讀取 `cloud_db` 的歷史紀錄 -> 轉換為 Template -> 存回 Cloud。
            
- **`core/watch_record_manager.py`**
    
    - **作用**：管理數位雙生資料結構。
        
    - **關鍵 Functions**:
        
        - `update_imaging_part()`: 當 `WorkflowManager` 拍完一張照，呼叫此函式更新記憶體中的資料結構，並觸發 Local DB 寫入。
            

#### 7. 執行層 (Orchestrator)

- **`core/workflow_manager.py`**
    
    - **作用**：總指揮。
        
    - **與 DB 的互動**:
        
        1. 初始化時，啟動 `cloud_sync`。
            
        2. 執行 `run_routine` 時，呼叫 `template_mgr` 拿腳本 (Local)。
            
        3. 每拍一張照 (`_perform_smart_step`)，呼叫 `data_mgr` 寫入 Local DB。
            
        4. 同時呼叫 `watch_rec_mgr` 更新數位雙生結構。
            

---

### 第三部分：合作執行流程舉例

讓我們用一個具體的例子：**「在德國機台執行 Rolex Submariner 的自動檢測」**。

#### 階段 1: 同步 (Sync Phase) - 發生在早上開機時

1. **Operator** 執行指令 `python main_cli.py --command sync_templates`。
    
2. **`WorkflowManager`** 呼叫 **`TemplateManager.sync_from_cloud()`**。
    
3. **`TemplateManager`** 呼叫 **`CloudDatabaseManager.scan_all_templates()`**。
    
4. **`CloudDatabaseManager`** 連線 AWS DynamoDB `WatchTemplates`，下載所有 JSON。
    
5. **`TemplateManager`** 將下載的 JSON 透過 **`DatabaseManager.save_local_template()`** 存入德國電腦的 SQLite。
    
    - _結果_：現在德國電腦裡有最新的 Rolex 腳本，就算網路斷了也能跑。
        

#### 階段 2: 執行與緩衝 (Execution Phase) - 發生在檢測時

1. **Operator** 執行 `python main_cli.py --command run_routine ...`。
    
2. **`WorkflowManager`** 從 SQLite 讀取腳本。
    
3. **`WorkflowManager`** 控制 Zaber 馬達移動、相機拍照 (HDR)。
    
4. **`WorkflowManager`** 拿到照片 (Image) 與 參數 (Metadata)。
    
5. **`WorkflowManager`** 呼叫 **`DataManager.process_and_sync_raw_image()`**。
    
6. **`DataManager`** 呼叫 **`DatabaseManager.insert_raw_image(synced=0)`**。
    
    - _關鍵_：資料瞬間寫入 SSD 的 SQLite，程式立刻繼續拍下一張。**完全不與 AWS 溝通**。
        

#### 階段 3: 背景同步 (Async Sync Phase) - 持續在背景發生

1. **`CloudSyncManager`** 的 `_worker_loop` 醒來，問 SQLite：「有沒傳的資料嗎？」
    
2. SQLite 回傳：「有，剛剛拍的那張 GlassPoint」。
    
3. **`CloudSyncManager`** 將照片上傳到 AWS S3。
    
4. 上傳成功後，**`CloudSyncManager`** 呼叫 **`CloudDatabaseManager.index_record()`**。
    
5. **`CloudDatabaseManager`** 將 Metadata (PK=WatchID, SK=VIEW#Front) 寫入 AWS DynamoDB `WatchAnalysisResults`。
    
6. 寫入成功後，**`CloudSyncManager`** 告訴 SQLite 把該筆資料標記為 `synced=1`。
    

#### 階段 4: 遠端查詢 (Remote Query Phase) - LA 的經理

1. LA 的經理打開 Dashboard。
    
2. Dashboard 透過 API 查詢 AWS DynamoDB `WatchAnalysisResults`。
    
3. 因為德國的資料已經（幾秒延遲後）同步上來了，經理看到了最新的檢測結果。
    

這就是整個系統如何利用 **Local-First** 策略，達成高效、穩定且可全球協作的 DynamoDB 整合。