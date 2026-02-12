

|                                  |     |
| -------------------------------- | --- |
| [[#### UI 流程控制]]                 |     |
| [[#### UI元件跟動作跟相關的functions對應]]  |     |
| [[#### audit_logger跟一些重複code解釋]] |     |
|                                  |     |


#### UI 流程控制
```
我有一個System有APP可以用App_run.py開啟, 在page_account登入帳號, page_device選擇device, page_checkTemplate會從local DB選擇template, 然後就會跳到page_Front_Run, 在這個頁面按pushButton_TemplateRun跳到page_Front_Loaded, 在這個頁面按pushButton_TemplateRun_3跳到page_Front_CamImgs. 這時listView_FrontCamImgs會顯示images. 這三個頁面是負責拍Watch的"Front"的images, 後面還有"Back"位置的三個頁面, 後面還有"OpenBack"位置的三個頁面, 後面還有"OpenBackCrown"位置的三個頁面, 後面還有"Strap"位置的三個頁面. 而在page_Material_click頁面會從folder讀取csv files, 並在page_view2_components可以決定authentication status, 在page_info page輸入watchentry的information, 最後在page_check確定每個步驟都有正確執行, 最後按pushButton_Temprun_Front_25. 

而在page_template_create page中, 可以在textEdit_account_checkTemplate_10裡從template list選擇template, 並按pushButton_TemplateRun_133, 就會跳到page_template_front, page_template_back, page_template_openback, page_template_openbackcrown, page_template_strap裡面都會有tableView會顯示watch部位('Front','Back'..)裡面的points.., 並在frame_42, frame_43, frame_44, frame_45, frame_46顯示image. 在page_template_summary, 可以看checkbox確保每個page都完成, 也可以按下pushButton_Temprun_Front_44到page_template_info page, 這裡可以修改tempalte的value然後可以按pushButton_TemplateRun_135, 這裡會跳回page_template_summary, 並將template存回local DB, 並上傳到AWS cloud. 

在這裡請分析以下相關的code, 並中文非常詳細介紹在這個流程裡有哪些會在電腦create folder, add or edit files, 跟local DB互動, 使用到AWS Cloud. 也介紹使用那些core/, config/, DB/, Controller/, cloud_relay/ folders裡面的那些python code的功能.
```

這是一個架構非常完整且複雜的軟體系統，融合了**本地端硬體控制 (Edge Computing)**、**本地資料庫快取 (Local Cache/Buffer)**、以及**AWS 雲端同步與遠端遙控 (Cloud Native & IoT)**。

根據您描述的 UI 流程（從登入、選擇設備、載入 Template、拍攝各部位影像、Material CSV 掃描、Component 驗證到最後的總結與上傳），我將為您非常詳細地拆解這個流程中，系統在**檔案系統 (File System)**、**本地資料庫 (Local DB)** 與 **AWS 雲端 (AWS Cloud)** 之間做了哪些互動，並解析各個資料夾中核心 Python 程式碼的功能。

---

### 一、 檔案系統操作 (電腦中 Create Folder, Add or Edit Files)

在您操作 UI 進行拍攝、檢測到最終完成的過程中，系統會在背景執行大量的檔案寫入與資料夾建立：

#### 1. 建立資料夾 (Create Folders)

- **檢測原始資料 (Raw Data)**：`DataManager.get_routine_raw_dir()` 會在設定的根目錄下建立 `Local_Data/{WatchID}/Raw/` 資料夾，用來存放相機拍下的原始圖片。
    
- **分析資料 (Analysis Data)**：`DataManager.create_analysis_dir()` 會建立 `Local_Data/{WatchID}/Analysis/Exp_{Timestamp}_{UUID}/`，用來存放演算法運算後的結果（例如遮罩 Mask）與 YAML 報告檔。
    
- **稽核日誌 (Audit Logs)**：`AuditLogger` 會自動建立 `Local_Data/audit_logs/` 資料夾。
    
- **UI 檢測結果 (Output Results)**：
    
    - 當您在 `page_view2_components` 驗證完零件並儲存時，會在專案根目錄下建立 `output_results/`。
        
    - 當您在 Material 頁面掃描並匯入 CSV 時，會建立 `output_results/material_db/` 資料夾。
        
- **暫存區 (Temp)**：`SimulationHardware` 或加密處理模組會建立 `Local_Data/temp/` 來進行影像複製或解密替換。
    

#### 2. 新增或修改檔案 (Add or Edit Files)

- **圖片檔 (.jpg / .png)**：拍攝或模擬產生的圖片會被存入 `Raw` 資料夾，檔名會被轉換成安全的隨機 UUID (例如 `1234abcd.jpg`)。
    
- **加密與 Metadata (.meta / .enc)**：如果系統開啟了 `enable_data_at_rest_encryption`，`DataEncryptor` 會將原圖的前 4096 bytes (Header) 抹零，並將加密後的 Header 存成附加的 `.meta` 檔，或是將整個 YAML 報告存成 `.enc` 檔。
    
- **檢測報告檔 (.yaml / .json)**：
    
    - `_atomic_write_yaml()` 會將分析報告寫入 `report_{watch_id}.yaml`。
        
    - 組件驗證結果會寫入 `component_results_{timestamp}.json`。
        
    - Material CSV 的解析結果會寫入 `material_record_{timestamp}.json` (及 `.yaml`)。
        
- **稽核日誌檔 (.jsonl)**：`AuditLogger` 會將使用者的每一個動作（如 `PRODUCTION_START`、`ROTATION_LINK`）加上 HMAC-SHA256 防竄改簽章，不斷 Append (新增) 到 `audit_YYYY-MM-DD.jsonl` 檔案中。
    

---

### 二、 與本地資料庫互動 (Local DB: watchDB.db)

`DB/db_manager.py` (SQLite) 在此系統中扮演著**「雲端快取 (Cache)」**與**「上傳緩衝區 (Write Buffer)」**的關鍵角色。

- **讀取 Template (`local_templates`)**：當您在 `page_checkTemplate` 下拉選單或輸入框選擇 Template 時，系統會呼叫 `list_all_templates()` 與 `get_local_template()`，直接從本地資料庫拉取 JSON 格式的檢測計畫，而不是每次都去雲端抓，這保證了離線或網路不穩時的可用性。
    
- **寫入 / 更新 Template (`local_templates`)**：當您在 `page_template_create` 建立新的 Template 並在最後按下儲存時，會呼叫 `save_local_template()`，將修改後的屬性與點位以 `INSERT OR REPLACE` 存回 SQLite。
    
- **註冊手錶與圖片寫入 (`watch_registry`, `raw_images`, `experiment_results`)**：
    
    - 每次拍攝完一張圖片，`DataManager` 會呼叫 `insert_raw_image()`，將圖片路徑寫入 DB，並標記 `synced=0` (尚未上傳)。
        
    - 結果報告或遮罩圖片產出時，會呼叫 `insert_experiment_result()`，同樣標記 `synced=0`。
        
- **身分驗證 (`users`)**：App 啟動時的登入動作，是透過查詢 `users` 表中的 `password_hash` 與 `salt` 來進行驗證。
    

---

### 三、 使用到 AWS Cloud 的環節

當您在流程最後按下 **`pushButton_Temprun_Front_25` (Upload / Finish)**，或是背景的 `CloudSyncManager` 運作時，系統會密集使用 AWS 服務：

- **1. Amazon S3 (圖片與檔案儲存)**
    
    - **背景同步上傳**：`CloudSyncManager` (在 `data_manager/cloud_sync.py`) 有一個 Background Thread，它會不斷輪詢本地 DB 中 `synced=0` 的檔案。找到後，使用 `boto3.client('s3')` 將圖片 (`.jpg`)、加密輔助檔 (`.meta`) 及報告檔 (`.yaml`) 上傳到名為 `watch-analysis-v1` 的 Bucket 中。上傳成功後，將本地 DB 的 `synced` 改為 1。
        
- **2. Amazon DynamoDB (NoSQL 資料庫)**
    
    - **上傳檢測實體 (WatchEntry)**：當您按下 UI 上的上傳按鈕時，`_on_upload_s3_dynamo` 會被觸發。它會將目前這支手錶的所有資訊 (`current_watchentry.to_dynamo_item()`) 直接透過 `boto3` 的 `put_item` 寫入 DynamoDB 的 `WatchAnalysisResults` 資料表。
        
    - **Template 同步**：在 Template Create 流程最後，`upload_template_to_cloud()` 會將新的 Template 同步寫入雲端的 `WatchTemplates` 表。
        
    - **指令與狀態追蹤**：`aws_agent.py` 接收到遠端指令時，會將執行狀態 (running, completed, error) 寫入 `WatchCommandLog` 表。
        
- **3. AWS IoT Core (MQTT 遠端遙控)**
    
    - **訂閱與發佈指令**：`cloud_relay/aws_agent.py` 會透過 MQTT 協定連接到 AWS IoT Endpoint。它訂閱了 `cmd/{device_id}` 主題，等待遠端 (如 Ann Arbor) 透過 `remote_run.py` 發送的指令 (如 `run_routine`, `manual_capture`, `system_control`)。
        
    - **即時影像串流 (Live Preview)**：當自動化流程在執行拍攝時，會觸發 `broadcast_preview`。它會將剛拍下的圖片壓縮、轉為 Base64 格式，並以 QoS 0 發佈到 `preview/{device_id}` 主題，讓遠端監控端可以看見即時縮圖。
        
    - **OTA 軟體更新**：Agent 會監聽 AWS IoT Jobs (`$aws/things/{device_id}/jobs/notify-next`)。當雲端派發軟體更新時，它會從 S3 下載 Zip 檔、解壓縮覆蓋本地檔案，並自動重啟 (Restart)。
        
- **4. Amazon Athena (巨量資料分析)**
    
    - 在 `data_manager/athena_query.py` 中，提供了一個介面可以透過 SQL 語法，直接查詢放置在 S3 中的海量檢測報告，用於未來的全域數據分析。
        

---

### 四、 各資料夾核心 Python 程式碼功能解析

#### 1. `core/` (核心邏輯層)

- **`workflow_manager.py` (WorkflowManager)**：系統的「大腦」。它串接了 UI / CLI 的指令與底層的硬體驅動。
    
    - **決定運行模式**：會根據 Config 判斷目前是 `production` (真機)、`hybrid` (混和，硬體故障時切換模擬) 還是 `simulation` (純軟體模擬)。
        
    - **`execute_routine`**：負責解析 Template 中的拍攝點位與參數，將其展平 (Flatten) 為一步步的指令，然後依序控制硬體移動、打光、拍攝，再將圖片交給 Orchestrator 呼叫演算法 (如 DoctR, YOLO 相關的 features_service)。
        

#### 2. `config/` (設定檔與靜態對照表)

- **`system_config.yaml` / `App_config.yaml` / `cli_config.yaml`**：系統啟動的參數來源。定義了網路模式 (`aws_iot` 或 `local`)、AWS 憑證路徑、資料夾相對路徑、各個 Microservice 演算法的 Port、以及 UI 的權限控管層級。
    
- **`pipeline_sim_map.yaml` & `simulation_map.yaml`**：**「模擬模式的靈魂」**。當系統在 `simulation` 模式，或 `hybrid` 模式下抓不到相機時，系統會查閱這個 YAML，根據目前的 View 與 Point (例如 `Front` 的 `macropoint1`)，從 `assets/sample_images/` 資料夾中找出對應的假圖片 (Dummy Image) 來「假裝」拍攝成功，讓軟體流程能繼續進行除錯。
    

#### 3. `DB/` (本地資料庫層)

- **`db_manager.py` (DatabaseManager)**：封裝了對 SQLite (`watchDB.db`) 的所有 SQL 操作。包含了 `raw_images`, `experiment_results`, `local_templates` 等資料表的建立與 CRUD 操作。它也包含了直接呼叫 `boto3` 寫入 DynamoDB (`WatchTemplates`) 的混合型擴充功能。
    

#### 4. `Controller/` (硬體控制抽象層)

- **`hardware_managers.py`**：定義了四個硬體管理器 (`ZaberManager` 控制馬達, `CameraManager` 控制相機, `LightManager` 控制打光, `IOManager` 控制 Keyence 測距儀)。內建「**優雅降級 (Graceful Degradation)**」機制：如果硬體斷線且允許 Mock (Hybrid 模式)，會自動回傳假資料 (如假座標 0.0) 避免程式崩潰。
    
- **`unified_driver.py` (UnifiedHardwareDriver)**：統一驅動介面。無論底層是真機還是模擬器，`WorkflowManager` 只需要呼叫 `unified_driver.move_compound()` 或 `run_robust_autofocus()`。它實作了真實的對焦邏輯 (讀取測距儀 -> 計算偏差 -> 移動 Z 軸)。
    
- **`simulation_driver.py` (SimulationHardware)**：當設定為純 `simulation` 模式時啟動。用 `time.sleep` 來模擬馬達移動的物理時間，並利用 OpenCV 生成雜訊圖片或複製 Asset 圖片來模擬相機拍攝。
    

#### 5. `cloud_relay/` (雲端中繼層)

- **`aws_agent.py` (AWSAgent)**：系統作為邊緣運算節點 (Edge Device) 的守門員。
    
    - 啟動 AWS IoT MQTT Client，維持長連線。
        
    - **冪等性檢查 (Idempotency)**：透過 `_CommandDedupCache` 與 Cloud DB，確保遠端重複發送的指令不會被執行兩次。
        
    - **安全驗證 (Signature)**：透過 HMAC 驗證遠端指令的合法性 (`_verify_signature`)。
        
    - 根據接收到的 JSON Payload，在背景建立 Thread 去呼叫 `WorkflowManager` 執行對應工作 (`run_routine`, `manual_move` 等)，並將結果寫回 Device Shadow 與 MQTT Topic。



#### UI元件跟動作跟相關的functions對應
```
請幫我更詳細的說明是執行UI的哪個動作, 導致系統執行甚麼動作, 這跟哪個code function有關係? 譬如在step x: page_checkTemplate頁面輸入textEdit_account_checkTemplate, 則會從local DB裡面按照我輸入的text列出相關的, 而選擇之後將這template從local DB載入記憶體, 這是由_setup_search_completer function 執行的. 並中文詳細解釋
```

這是一個非常好的問題。要理解這套龐大系統的運作，我們必須將「**使用者的滑鼠/鍵盤動作**」與「**後端的 Python 函式呼叫**」精確地對應起來。

以下我將依照您描述的操作流程（Run Routine 與 Template Create），詳細拆解每一個步驟的 **UI 動作**、**系統反應** 以及 **對應的程式碼 (Code Function)**。

---

### 第一部分：檢測執行流程 (Run Routine Flow)

這個流程是操作員最常用的，目的是對一隻手錶進行完整的拍攝與數據上傳。

#### Step 1: 搜尋並載入 Template

- **UI 動作**：在 `page_checkTemplate` 頁面的 `textEdit_account_checkTemplate` 輸入文字（例如 "Rolex"）。
    
- **系統反應**：
    
    1. 系統即時監聽輸入內容。
        
    2. 從記憶體中的 `self.search_db`（源自 Local DB）過濾出符合的字串。
        
    3. 顯示一個下拉選單 (`QListWidget`) 列出建議選項。
        
- **對應程式碼**：
    
    - **初始化**：`MainWindow._setup_search_completer()` (在 `App/main.py`)。它將 `textEdit` 的 `textChanged` 信號連接到 `_update_search_suggestions`。
        
    - **執行篩選**：`MainWindow._update_search_suggestions()`。它執行 `[s for s in self.search_db if text in s]` 這種 Python List 過濾邏輯。
        

#### Step 2: 確認選擇 Template

- **UI 動作**：點擊下拉選單中的某個 Template 名稱（例如 "Rolex_Submariner"）。
    
- **系統反應**：
    
    1. 從 Local DB (`watchDB.db`) 的 `local_templates` 資料表讀取該 Template 的完整 JSON 結構。
        
    2. 將 JSON 反序列化 (Deserialize) 為 `WatchTemplate` 物件存入 `self.current_template`。
        
    3. 建立一個新的 `watchentry` 物件（本次檢測的實體），生成唯一的 `WatchID` (例如 `Rolex_Submariner_17158900`)。
        
    4. 跳轉頁面至 `page_Front_Run`。
        
- **對應程式碼**：
    
    - **觸發**：`MainWindow._on_search_item_clicked()`。
        
    - **資料庫讀取**：`MainWindow._load_template_to_memory()` -> 呼叫 `self.local_db.get_local_template()` (在 `DB/db_manager.py`)。
        
    - **建立實體**：`MainWindow._create_new_watchentry()`。
        

#### Step 3: 拍攝影像 (以 Front 為例)

- **UI 動作**：在 `page_Front_Loaded` 頁面按下 **`pushButton_TemplateRun_3`** (Load/Capture)。
    
- **系統反應**：
    
    1. 系統判斷目前的 `system.mode` (Production / Hybrid / Simulation)。
        
    2. **如果是 Simulation**：從 `config/pipeline_sim_map.yaml` 讀取假圖路徑。
        
    3. **如果是 Production/Hybrid**：
        
        - 讀取 Template 中定義的 `Front` 部位所有 Point 的座標 (X, Y, Z) 和曝光參數。
            
        - 指揮 Zaber 馬達移動、控制燈光打亮、觸發相機拍照。
            
        - 將拍到的照片存入 `Local_Data/{WatchID}/Raw/`。
            
        - 將圖片路徑寫入 Local DB (`raw_images` table)。
            
    4. 最後將圖片顯示在 `listView_FrontCamImgs` 上。
        
- **對應程式碼**：
    
    - **UI 入口**：`MainWindow.load_camimgs('front')`。
        
    - **核心拍攝邏輯**：`MainWindow._capture_images_routine('front')`。
        
    - **硬體調度**：`WorkflowManager._perform_smart_step()` (在 `core/workflow_manager.py`) -> 呼叫 `UnifiedHardwareDriver.capture_image()`。
        

#### Step 4: Material CSV 讀取

- **UI 動作**：在 `page_Material_click` 頁面按下 **`pushButton_TemplateRun_157`** (Scan/Finish)。
    
- **系統反應**：
    
    1. 掃描設定好的 `material_csv_folder` 資料夾。
        
    2. 找出修改時間最新的 `.csv` 檔案。
        
    3. 解析 CSV 內容（針對 XRF 金屬分析儀的格式），提取 Au, Pt, Steel 等數值。
        
    4. 將解析結果存成 JSON 到 `output_results/` 並顯示在文字框中。
        
- **對應程式碼**：
    
    - **觸發**：`MainWindow._finish_material_scan()`。
        
    - **解析邏輯**：`MainWindow._read_and_format_csv()` (使用 Python 內建 `csv` module)。
        

#### Step 5: 上傳至 AWS (最關鍵的一步)

- **UI 動作**：在 `page_check` 頁面確認所有 Checkbox 後，按下 **`pushButton_Temprun_Front_25`**。
    
- **系統反應**：
    
    1. **驗證**：確保 `WatchEntry` 物件存在。
        
    2. **背景同步**：觸發 `CloudSyncManager` 開始將本地還沒上傳的圖片 (`synced=0`) 傳到 S3。
        
    3. **寫入 DynamoDB**：將這次檢測的所有屬性（手錶型號、序號、各部位狀態、Material 結果）打包成 JSON，直接寫入 AWS DynamoDB 的 `WatchAnalysisResults` 資料表。
        
- **對應程式碼**：
    
    - **觸發**：`MainWindow._on_upload_s3_dynamo()`。
        
    - **雲端寫入**：使用 `boto3` 套件，呼叫 `table_resource.put_item(Item=item_data)`。
        

---

### 第二部分：Template 建立流程 (Template Create Flow)

這個流程是用來定義新的手錶檢測標準。

#### Step A: 建立新 Template

- **UI 動作**：在 `page_template_create` 輸入新名稱，並按下 **`pushButton_TemplateRun_133`**。
    
- **系統反應**：
    
    1. 如果是 "Scratch"，則建立一個空的 `WatchTemplate` 物件。
        
    2. 如果是選擇既有的 Template，則從 Local DB 載入並深拷貝 (Deep Copy) 一份，並賦予新的 UUID (`templateID`)。
        
- **對應程式碼**：
    
    - **觸發**：`MainWindow._on_template_create_click()`。
        
    - **物件操作**：使用 `uuid.uuid4().hex` 生成新的 ID，防止覆蓋舊資料。
        

#### Step B: 編輯點位 (Edit Points)

- **UI 動作**：在 `page_template_front` 的 `tableView_9` 點選某個點位，並按下 **`pushButton_TemplateRun_121`** (Delete) 或輸入文字按 Enter (Add)。
    
- **系統反應**：
    
    - 直接修改記憶體中 `self.template_front_model` (UI Model) 的資料。
        
    - 同時更新 `self.current_template.watchView.Front` 下的資料結構。
        
- **對應程式碼**：
    
    - **刪除**：`MainWindow._template_delete_item('front')`。
        
    - **新增**：`MainWindow._template_add_item('front', ...)`。
        

#### Step C: 儲存並同步雲端

- **UI 動作**：在 `page_template_summary` 按下 **`pushButton_TemplateRun_135`** (Finish/Save)。
    
- **系統反應**：
    
    1. **本地存檔**：將 `current_template` 轉為 Dictionary，寫入 Local DB 的 `local_templates` 表 (`INSERT OR REPLACE`)。
        
    2. **雲端同步**：直接連線 AWS DynamoDB 的 `WatchTemplates` 表，將此 Template 上傳，讓其他機台也能下載使用。
        
- **對應程式碼**：
    
    - **觸發**：`MainWindow._save_template_info_and_finish()`。
        
    - **本地寫入**：`self.local_db.save_local_template()`。
        
    - **雲端寫入**：`self.local_db.upload_template_to_cloud()` (在 `DB/db_manager.py` 中實作)。
        

---

### 總結對照表

| **UI 頁面**          | **UI 元件 (按鈕/輸入框)**            | **動作**      | **執行的 Python Function (main.py)** | **涉及的後端模組**                                  |
| ------------------ | ----------------------------- | ----------- | --------------------------------- | -------------------------------------------- |
| **Check Template** | `textEdit_...`                | 輸入文字        | `_update_search_suggestions`      | `search_db` (List)                           |
| **Check Template** | `QListWidget`                 | 點擊選項        | `_on_search_item_clicked`         | `DB.db_manager` (讀取 DB)                      |
| **Front/Back...**  | `pushButton_TemplateRun_3`    | 點擊 (Load)   | `load_camimgs`                    | `core.workflow_manager` (硬體控制)               |
| **Material**       | `pushButton_TemplateRun_157`  | 點擊 (Scan)   | `_finish_material_scan`           | Python `csv`, `pathlib`                      |
| **Check (Finish)** | `pushButton_Temprun_Front_25` | 點擊 (Upload) | `_on_upload_s3_dynamo`            | `boto3` (AWS SDK), `data_manager.cloud_sync` |
| **Temp Create**    | `pushButton_TemplateRun_133`  | 點擊 (Create) | `_on_template_create_click`       | `DB.templates.structure`                     |
| **Temp Summary**   | `pushButton_TemplateRun_135`  | 點擊 (Save)   | `_save_template_info_and_finish`  | `DB.db_manager` (寫入 SQLite & DynamoDB)       |




#### audit_logger跟一些重複code解釋
```
我在檢查我的code的時候發現在D:\Provenance Project\ImagingLibWatch\config\keys\跟D:\Provenance Project\ImagingLibWatch\App\config\keys\ folder都有aesgcm.key跟hmac.key. 所以他們的作用是甚麼? 在code裡面真正使用的是哪一個folder裡面的?

然後在D:\Provenance Project\ImagingLibWatch\DB\跟D:\Provenance Project\ImagingLibWatch\App\DB\也都有watchDB file. 在code裡面真正使用的是哪一個folder裡面的? 

然後在D:\Provenance Project\ImagingLibWatch\App\modules\跟D:\Provenance Project\ImagingLibWatch\App\ui\都有ui_main.py, 在code裡面真正使用的是哪一個folder裡面的?

這些有的files跟python codes在App\好像都多了一個copy, 是否也是相對path產生的問題? 請幫我仔細檢查所有code是否有可能會造成這樣的地方.

另外在D:\Provenance Project\ImagingLibWatch\output_results\有component_results_20260211_190721 這是甚麼是怎麼產生的?

也請詳細解釋D:\Provenance Project\ImagingLibWatch\logging_system\audit_logger.py是甚麼?, D:\Provenance Project\ImagingLibWatch\Local_Data\audit_logs\audit_2026-02-11.jsonl是怎麼產生的? 

D:\Provenance Project\ImagingLibWatch\DB\schema file是甚麼?

請中文詳細解釋

```

是一個非常敏銳的觀察！這正是軟體開發中常見的「**相對路徑 (Relative Path) 與工作目錄 (Current Working Directory, CWD) 不一致**」所導致的經典問題。

簡單來說：**您的系統正在產生兩套資料（一套在根目錄，一套在 `App/` 目錄下），這是因為 `App_run.py` 改變了程式執行的基準點。**

以下我將針對您的問題逐一詳細分析原因、副作用，以及如何修正。

---

### 1. 為什麼會有兩份 Key 和 DB？

#### **現象分析**

- **Root Folder**: `D:\Provenance Project\ImagingLibWatch\` (原本預期的位置)
    
- **App Folder**: `D:\Provenance Project\ImagingLibWatch\App\` (多出來的副本位置)
    

#### **根本原因 (Root Cause)**

問題出在 **`App_run.py`** 這行程式碼：

Python

```
# (App_run.py)
os.chdir(str(APP_DIR))  # <--- 兇手在這裡！
```

這行指令將「當前工作目錄 (CWD)」切換到了 `App/` 資料夾內部。然而，您的 `config/system_config.yaml` 裡面的路徑設定寫的是 **相對路徑** (`./`)：

YAML

```
# config/system_config.yaml
paths:
  base_data_dir: "./Local_Data"
  aesgcm_key_path: "./config/keys/aesgcm.key"
  # ...
```

**發生了什麼事？**

1. 當您執行 **`main_cli.py`** (CLI 模式) 時，CWD 是根目錄。`./config/keys` 指向 `ImagingLibWatch/config/keys`。 **(正確)**
    
2. 當您執行 **`App_run.py`** (GUI 模式) 時，程式先切換到了 `App/` 目錄。此時 `./config/keys` 變成了 `ImagingLibWatch/App/config/keys`。因為程式發現該路徑不存在，或為了寫入 Key，它就自動在 `App` 底下建立了一份新的資料夾和檔案。 **(錯誤，產生了副本)**
    

#### **詳細解答您的問題：**

1. **Keys (`aesgcm.key`, `hmac.key`)**
    
    - **作用**：
        
        - `aesgcm.key`：用於加密圖片 Header 和敏感資料 (Data At Rest Encryption)。
            
        - `hmac.key`：用於對 Audit Log 進行數位簽章，防止日誌被竄改。
            
    - **真正使用的是哪個？**
        
        - 如果您跑 GUI (`App_run.py`)，用的是 `App/config/keys/` 裡面的。
            
        - 如果您跑 CLI (`main_cli.py`)，用的是 `config/keys/` 裡面的。
            
        - **嚴重後果**：這會導致 GUI 加密的檔案，CLI 無法解密（因為金鑰不同），反之亦然。
            
2. **Database (`watchDB.db`)**
    
    - **真正使用的是哪個？**
        
        - 同理，GUI 模式用 `App/DB/watchDB.db`，CLI 模式用 `DB/watchDB.db`。
            
    - **嚴重後果**：您的 GUI 操作紀錄（Template、使用者、拍攝紀錄）與 CLI 的紀錄完全分離，導致資料不同步（Data Split Brain）。
        
3. **UI Code (`ui_main.py`)**
    
    - **真正使用的是哪個？**
        
        - 在 `App/main.py` 中有一行 `from ui.ui_main import Ui_MainWindow`。
            
        - 這代表它引用的是 **`App/ui/ui_main.py`**。
            
        - `App/modules/ui_main.py` 極有可能是您在整理檔案或複製貼上時意外留下的殘骸，**程式碼並沒有使用它**，建議刪除以免混淆。
            

---

### 2. 如何修正這個問題？ (Code Fix)

我們必須強制所有路徑都以「專案根目錄」為基準，無論程式是從哪裡啟動的。請依照以下步驟修正程式碼：

#### **修正 1: `App_run.py` (移除 `os.chdir`)**

不要改變工作目錄，而是將 `App` 加入 `sys.path` 即可。

Python

```
# (App_run.py) - 修正版
import os
import sys
from pathlib import Path

# 取得目前檔案 (App_run.py) 的所在目錄，即專案根目錄
PROJECT_ROOT = Path(__file__).resolve().parent
APP_DIR = PROJECT_ROOT / "App"

# 將 App 目錄加入 Python 搜尋路徑，這樣才能 import main
sys.path.insert(0, str(APP_DIR))

# [刪除這行] 不要改變工作目錄，保持在根目錄
# os.chdir(str(APP_DIR)) 

from main import MainWindow
from PySide6.QtWidgets import QApplication

app = QApplication(sys.argv)
w = MainWindow()
sys.exit(app.exec())
```

#### **修正 2: `App/main.py` (確保資源路徑正確)**

因為我們移除了 `os.chdir`，現在 CWD 變回了根目錄，但 `App/main.py` 裡面有些引用 `ui/` 或 `themes/` 的地方可能會因為路徑改變而找不到。

您需要確保 `_abs_path_from_here` 函式能正確運作（它原本就是設計來解決相對路徑問題的，所以應該沒問題）。

但請檢查 `DataManager` 和 `DatabaseManager` 的初始化。您之前的代碼中我已經幫您加上了這段修正（在 `data_manager/local_storage.py`）：

Python

```
# [這是之前給您的代碼，它已經包含修正邏輯]
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.path.isabs(raw_base_dir):
    self.base_dir = raw_base_dir
else:
    # 強制綁定絕對路徑，忽略 CWD
    self.base_dir = os.path.abspath(os.path.join(project_root, os.path.normpath(raw_base_dir)))
```

**建議操作：**

1. **備份資料**：先備份所有 DB 和 Key。
    
2. **清理副本**：刪除 `App/DB/`、`App/config/keys/`、`App/Local_Data/` (如果有)。
    
3. **應用修正**：修改 `App_run.py`。
    
4. **測試**：重新啟動 GUI，確認它是否讀取到了根目錄下的 DB (可以看到舊資料) 和 Key。
    

---

### 3. 其他檔案解釋

#### **A. `output_results/component_results_20260211_190721.json`**

- **這是什麼？**：這是您在 **View 2 (Components)** 頁面進行「組件驗證 (Authentication)」後的存檔結果。
    
- **怎麼產生的？**：
    
    1. 您在該頁面點選了零件（如 "Dial"），並標記狀態（如 "Original"）。
        
    2. 系統將這些狀態存在 `self.comp_auth_states` 字典中。
        
    3. 當您按下 **Save** 或 **Next** 按鈕時，觸發了 `_on_comp_save_and_nav` -> `_save_component_results`。
        
    4. 該函式將字典加上時間戳記，存成 JSON 檔。
        
- **內容範例**：
    
    JSON
    
    ```
    {
      "timestamp": "20260211_190721",
      "learning_mode": true,
      "components": {
        "Dial": "Original",
        "Hands": "Modified"
      }
    }
    ```
    

#### **B. `logging_system/audit_logger.py` 與 `audit_2026-02-11.jsonl`**

這是系統的 **「數位鑑識與稽核系統 (Audit Trail)」**。它的設計目的是為了符合企業或實驗室的高規格安全標準。

- **`audit_logger.py` 是什麼？**
    
    - 它是一個專門寫 Log 的模組，但比一般 Log 更嚴格。
        
    - **Hash Chain (雜湊鏈)**：它寫入的每一行 Log，都包含了**上一行 Log 的簽章 (Signature)**。這意味著，如果有人偷偷打開文字檔刪除了中間某一行，整個鏈條就會斷掉（後一行的 `prev_hash` 會對不上），從而發現資料被竄改。
        
    - **HMAC 簽章**：使用 `hmac.key` 對內容進行加密簽章，確保只有擁有 Key 的系統才能寫入合法的 Log。
        
- **`audit_...jsonl` 是怎麼產生的？**
    
    - **觸發點**：當系統做關鍵動作時（例如：啟動拍攝 `PRODUCTION_START`、完成模擬 `SIMULATION_COMPLETE`、或是登入），程式會呼叫 `self.audit.log_action(...)`。
        
    - **格式**：JSONL (JSON Lines) 格式，每一行是一個完整的 JSON 物件。
        
    - **內容**：包含時間 (`ts`)、使用者 (`uid`)、動作 (`act`)、詳細資料 (`dat`)、上一筆簽章 (`prv`)、本筆簽章 (`sig`)。
        

#### **C. `DB/schema` file**

- **這是什麼？**：這通常不是一個被 Python 程式直接執行的檔案。
    
- **作用**：它通常是一個 **SQL 腳本檔 (.sql)** 或者 **說明文件**，用來定義資料庫的結構（Schema）。
    
- **內容**：它會寫著 `CREATE TABLE raw_images (...)` 這樣的語法。
    
- **用途**：給開發者參考用，或者在第一次部署資料庫時，手動匯入這個檔案來建立資料表結構。但在您的系統中，`db_manager.py` 的 `_init_db()` 函式已經在程式碼裡自動執行了 `CREATE TABLE IF NOT EXISTS`，所以這個 schema file 可能只是一個備份或設計文件。
    

### 總結您的行動清單

1. **修正路徑**：修改 `App_run.py`，移除 `os.chdir`，改用 `sys.path.insert`。
    
2. **清理垃圾**：確認修正後，刪除 `App/` 資料夾下意外產生的 `DB/`, `config/`, `Local_Data/` 資料夾。
    
3. **確認 UI**：`App/modules/ui_main.py` 是多餘的，請確認 `App/main.py` import 的是 `ui.ui_main` 後，刪除 `modules` 裡的那份。

