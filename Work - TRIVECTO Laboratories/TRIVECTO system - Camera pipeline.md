

| [[#### Camera pipeline (focus on autofocus)]] |     |
| --------------------------------------------- | --- |
| [[#### Camera pipeline (focus on Dataflow)]]  |     |
|                                               |     |
| [[#### stress test檢測內容跟硬體Code做的修正]]           |     |
| [[#### 那我修改的硬體code運作機制解答]]                    |     |
|                                               |     |


#### Camera pipeline (focus on autofocus)
```
請幫我詳細列出pushButton_TemplateRun_3之後的camera pipeline流程, 列出function跟所屬的python file 尤其是execute_template_point function跟run_robust_autofocus function. 是不是在hardware_config裡的autofocus_option=0,1,2,3會執行run_robust_autofocus, autofocus_option=4會執行execute_template_point? 是否可畫成樹狀流程圖.
```
### 關於 Autofocus 的迷思釐清

**不，這是一個常見的誤解！** `autofocus_option` (0, 1, 2, 3, 4) 並不是用來決定要走 `run_robust_autofocus` 還是 `execute_template_point`。

這兩個函數的根本差異在於**「使用情境 (Context)」**，而不是選項編號：

- **`execute_template_point()` (位於 `unified_driver.py`):** 這是**「全自動化拍攝流程」**的核心大腦。當你按下 `pushButton_TemplateRun_3` (自動執行 Front 視角流程) 或是跑自動化測試時，系統會呼叫這個函數。它**同時支援了 Option 0, 1, 2, 3, 4**。它將所有的對焦、打光、移動、拍照邏輯（共 7 個 Phase）包裝成一個完整動作。
    
- **`run_robust_autofocus()` (位於 `unified_driver.py`):** 這是一個**「手動/獨立的對焦輔助工具」**。當工程師在控制面板 (Control Panel) 點擊 "Auto Focusing..." 或發送 `disp_img` 動作指令時，才會呼叫這個函數。它只實作了 Option 1, 2, 3，並不包含 Option 4。
    

---

### 📷 Camera Pipeline 流程樹狀圖 (由 pushButton_TemplateRun_3 觸發)

當你按下 `pushButton_TemplateRun_3` (Front 視角執行按鈕) 時，程式的流向如下：

Plaintext

```
[UI Event] User Clicks pushButton_TemplateRun_3
 │
 ├── 1. main.py
 │   ├── _setup_front_logic()
 │   │   └── 綁定按鈕，呼叫 _run_view_sequence("front", ...)
 │   ├── _run_view_sequence()
 │   │   └── 設定狀態並呼叫 load_camimgs()
 │   ├── load_camimgs()
 │   │   └── 啟動背景執行緒 CaptureRoutineWorker 避免卡死 UI
 │   └── CaptureRoutineWorker.run()
 │       └── 呼叫 _capture_images_routine_internal()
 │           └── 遍歷 Template 點位，轉換 Config，交由 WorkflowManager 執行。
 │
 ├── 2. core/workflow_manager.py (未包含在附件中，為中介層)
 │   └── _perform_smart_step()
 │       └── 最終呼叫硬體驅動層 (Hardware Driver)
 │
 ├── 3. unified_driver.py (核心大腦)
 │   └── execute_template_point() 
 │       │
 │       ├── Phase 1: Parse Config (解析 X, Y, Z 座標、曝光、燈光、取得 z_af_option)
 │       ├── Phase 2: Lights OFF & Initial XY Move (關燈並移動 Zaber XY 軸)
 │       │
 │       ├── Phase 3: Z-Axis Mechanical AF (Z軸機械對焦核心)
 │       │   ├── 若 Option = 4: 執行內部寫死的 Keyence 2 Absolute Calculation
 │       │   ├── 若 Option = 1: 執行內部寫死的 Keyence 1 Closed-Loop
 │       │   └── 若 Option = 2 或 3: 呼叫 _af_vision_method()
 │       │
 │       ├── Phase 4 & 5: Final Alignment & Light ON (移動最終Z軸，開燈，準備相機)
 │       │
 │       ├── Phase 6: Liquid Lens Autofocus (液態鏡頭對焦)
 │       │   └── 呼叫 hardware_managers.py 的相機層
 │       │       ├── 優先嘗試: trigger_hardware_liquid_af() (硬體 FPGA 對焦)
 │       │       └── 備案嘗試: run_liquid_lens_autofocus() (軟體多步掃描對焦)
 │       │
 │       └── Phase 7: Image Capture (觸發快門拍照)
 │           └── 呼叫 hardware_managers.py 的相機層
 │               ├── 若啟用 HDR: capture_enhanced_hdr()
 │               └── 一般拍攝: capture_image()
 │
 └── 4. main.py (回傳結果)
     └── _on_single_image_captured()
         └── 將拍好的照片即時更新到 UI 的九宮格與表格中
```

---

### 核心函數深入解析

#### 1. `execute_template_point(self, params)`

- **檔案位置:** `Controller/hardware_drivers/unified_driver.py`
    
- **職責:** 將單一「點位(Point)」的所有參數 (XYZ、打光、對焦模式、相機設定) 轉換為實體硬體動作，並回傳拍好的影像。
    
- **AF 判斷邏輯 (Phase 3):** 它會先從 `hardware_config.yaml` 抓取全局的 `autofocus_option`，如果該點位有 override (例如 `front.micropoint14: 0`)，則套用 override 值。
    
    - **Option 4:** 讀取 `keyence_offset`，移動 XY 後，讀取 `out01` 數值並直接加上 `keyence2_base_z` 算出絕對高度。
        
    - **Option 1:** 移動到測距點，使用 `out01` 進行多次迴圈修正 (Closed-Loop)，直到誤差小於 `tolerance`。
        
    - **Option 2 & 3:** 打開燈光，呼叫同檔案內的 `_af_vision_method` 利用影像銳利度 (Laplacian / Tenengrad) 來找最佳 Z 軸。
        

#### 2. `run_robust_autofocus(self, base_x, base_y, base_z, cam_alias, has_glass=True)`

- **檔案位置:** `Controller/hardware_drivers/unified_driver.py`
    
- **職責:** 獨立的對焦模組。主要被工程模式 UI (例如 `ControlEngWindow` 或 `test_template_ui.py`) 裡面的獨立「Auto Focus 按鈕」呼叫。
    
- **AF 判斷邏輯:** * **Option 0:** 直接回傳 `base_z` (不對焦)。
    
    - **Option 1:** 呼叫 `_af_keyence_method()`。
        
    - **Option 2 & 3:** 呼叫 `_af_vision_method()`。
        
    - _(注意：此函數完全沒有實作 Option 4 的邏輯)_





#### Camera pipeline (focus on Dataflow)
```
請幫我中文詳細解釋執行App main.py的流程, 著重資料流程. 如何用template info到控制拍照, camera taken image存到哪裡, 如何把image跟computer vision task services連結作分析, 結果要放到哪裡? 那些會上傳到AWS S3怎麼控制, 那些傳入local DB, 以及跟AWS DB同步. 也要指出跟哪個python file的哪個function相關, 並畫詳細流程樹狀圖
```
這是一個非常經典且架構嚴謹的邊緣到雲端 (Edge-to-Cloud) AI 檢測系統。系統採用了**非同步處理 (Asynchronous)**、**本地寫入緩衝 (Write Buffer)** 以及**背景同步 (Background Polling Sync)** 的設計模式。

以下我將為你詳細拆解 `main.py` 啟動後的資料流與生命週期，並特別針對你關注的五大核心問題進行解說，最後附上完整的呼叫流程樹狀圖。

---

### 核心資料流解析

#### 1. 從 Template Info 到控制拍照 (Template -> Hardware Capture)

當使用者在 UI 選擇一個手錶型號 (Template) 並按下執行時，資料是這樣流動的：

- **載入 Template：** 在 `main.py` 中，`_load_template_to_memory()` 會從 Local DB 讀取預設的 Template JSON，轉為記憶體中的 `WatchTemplate` 物件。接著透過 `_create_new_watchentry()` 產生這次檢測的唯一實體 `watchentry` 物件，並賦予專屬的 `WatchID` 與各點位的 UUID。
    
- **觸發拍照：** 使用者點擊執行後，會呼叫 `main.py` 的 `_run_view_sequence()`，這會啟動背景執行緒 `CaptureRoutineWorker`，避免卡死 UI。
    
- **解析參數並呼叫硬體：** 核心邏輯在 `main.py` 的 `_capture_images_routine_internal()`。它會遍歷 Template 中的 `watchView`，將 X, Y, Z 座標、相機選擇 (`cam_id`)、曝光時間 (`exptimes`) 提取出來，打包成 `step_config` 字典。
    
- **交給 Workflow 執行：** 將 `step_config` 傳遞給 `core/workflow_manager.py` 的 `_perform_smart_step()`，該函數會呼叫硬體驅動 `self.hw.execute_template_point(hw_cfg)` 真正驅動馬達與相機拍下照片。
    

#### 2. 相機拍下的 Image 存到哪裡？ (Raw Image Storage)

相機回傳 OpenCV 影像矩陣後，資料落地流程如下：

- **暫存區：** `workflow_manager.py` 會先將影像存入 `Local_Data/temp/` 加上 UUID 的暫存檔。
    
- **正式歸檔：** 呼叫 `data_manager/local_storage.py` 的 `process_and_sync_raw_image()`。
    
    - 檔案會被移動到：`Local_Data/{WatchID}/Raw/{UUID}.jpg`。
        
    - **加密 (可選)：** 如果開啟加密，會呼叫 `encryption.py` 對檔案 Header 進行混淆，並產生 `.meta` 備份檔。
        
- **寫入 Local DB：** 呼叫 `DB/db_manager.py` 的 `insert_raw_image()`，將本機路徑、預計的 S3 Key 寫入 SQLite 表格 `raw_images` 中，並將 `synced` 欄位標記為 `0` (尚未同步)。
    

#### 3. 如何將 Image 與 Computer Vision Task 連結並分析？

- **綁定關聯 (Mapping)：** 在系統啟動時，`WorkflowManager._build_service_lookup_map()` 會讀取 `system_config.yaml`，建立「視角點位」對應「CV 演算法」的映射表 (例如 `front.glasspoint1` 對應 `['ocr_service']`)。
    
- **觸發分析：** 拍完照取得本機路徑後，在 `main.py` 裡會接著呼叫 `_trigger_manual_analysis()`。
    
- **安全解密與派發：** 轉交給 `workflow_manager.py` 的 `_run_analysis_safe()`。因為影像可能被加密了，它會先呼叫 `data_manager.prepare_image_for_viewing()` 解密出一個暫存圖，然後呼叫 `core.orchestrator.run_batch()` 將影像路徑與演算法清單派發給微服務架構 (ServiceManager/ZeroMQ 或 HTTP) 進行推論。
    

#### 4. CV 分析結果放到哪裡？ (Analysis Results)

- **推論結果回傳：** Orchestrator 拿到 AI 回傳的 Mask 影像與 JSON 數據 (Metrics)。
    
- **資料夾存放：** 檔案會統一存放在專屬分析目錄：`Local_Data/{WatchID}/Analysis/{ExpID}/` 下。
    
- **紀錄結果：** 呼叫 `workflow_manager.py` 的 `_process_step_results()`，它會調用 `local_storage.py`，將 Mask 影像路徑與 JSON 結果寫入 Local DB 的 `experiment_results` 表格中，同樣標記 `synced=0`。
    

#### 5. 上傳 AWS S3 的控制機制與 Local/Cloud DB 同步

這個系統採用了非常穩健的**非同步寫入緩衝 (Write-Behind Buffer)** 策略，UI 不會因為網路延遲而卡頓。

- **加入佇列 (Queueing)：** 當 `local_storage.py` 完成寫入 SQLite 後，會呼叫 `cloud_sync.add_to_queue(local_path, s3_key)`，但這其實只是一個喚醒信號。
    
- **背景輪詢 (Polling)：** 真正的推手是 `data_manager/cloud_sync.py` 裡面的背景執行緒 `_worker_loop()`。它會不斷向 Local DB (`db_manager.py`) 請求 `get_pending_uploads()`，撈出所有 `synced=0` 的紀錄。
    
- **上傳 S3：** 若檔案存在，`cloud_sync.py` 使用 `boto3` 將檔案直接上傳至 `s3://{bucket_name}/{s3_key}`。
    
- **同步 AWS DynamoDB：** S3 上傳成功後，`cloud_sync.py` 會立刻呼叫 `data_manager/cloud_db.py` 的 `index_record()`，將 metadata、時間戳與 S3 Key 寫入雲端的 DynamoDB `WatchAnalysisResults` 表中。
    
- **標記完成 (Ack)：** 雲端寫入成功後，回頭呼叫 Local DB 的 `mark_as_synced()`，將 SQLite 的該筆紀錄標記為 `synced=1`，完成整個閉環。
    

---

### 詳細執行流程樹狀圖 (Flowchart Tree)

這裡為你梳理從使用者點擊執行，到最終資料上傳 AWS 的完整函數呼叫鏈：

Plaintext

```
[UI] main.py : 使用者點擊 "執行" 某個視角 (例如 Front)
 ├── _run_view_sequence("front")
 │    ├── load_camimgs("front")
 │    │    └── CaptureRoutineWorker(QThread).start()  -> 進入背景執行緒
 │
 [Background Thread] main.py : CaptureRoutineWorker.run()
  └── _capture_images_routine_internal()
       │
       ├── [擷取 Template 參數]
       │    └── 從 self.current_template 取出 X, Y, Z, 曝光時間, 相機 ID
       │
       ├── [1. 觸發拍照與儲存]
       │    └── workflow_manager.py : _perform_smart_step()
       │         ├── hardware_drivers.cam.capture_image()  -> OpenCV 影像
       │         ├── 存入 Local_Data/temp/{uuid}.jpg
       │         └── local_storage.py : process_and_sync_raw_image()
       │              ├── _safe_move() -> 移至 Local_Data/{WatchID}/Raw/{uuid}.jpg
       │              ├── encryption.py : obfuscate_image_header_to_bytes()  [可選：檔案加密]
       │              ├── db_manager.py : insert_raw_image()  [寫入 Local SQLite, synced=0]
       │              └── cloud_sync.py : add_to_queue()  [發送喚醒訊號]
       │
       ├── [2. 觸發 CV 演算法]
       │    └── main.py : _trigger_manual_analysis()
       │         └── workflow_manager.py : _run_analysis_safe()
       │              ├── local_storage.py : prepare_image_for_viewing() [解密為暫存檔]
       │              └── orchestrator.py : run_batch(tasks, image_path) [派發給 AI 服務]
       │
       └── [3. 處理 CV 結果]
            └── workflow_manager.py : _process_step_results()
                 ├── 儲存至 Local_Data/{WatchID}/Analysis/{ExpID}/
                 ├── db_manager.py : insert_experiment_result() [寫入 SQLite, synced=0]
                 └── cloud_sync.py : add_to_queue()

========================================================================================
[獨立的非同步守護執行緒 - 負責雲端同步]
data_manager/cloud_sync.py : _worker_loop()
 ├── db_manager.py : get_pending_uploads()  [撈出 SQLite 中 synced=0 的檔案]
 │
 ├── [For 每一筆待傳資料]
 │    ├── boto3.client('s3').upload_file(...)  [上傳圖片或 JSON 到 AWS S3]
 │    │
 │    ├── 成功後 -> cloud_db.py : index_record() [寫入 AWS DynamoDB 建立關聯索引]
 │    │
 │    └── db_manager.py : mark_as_synced()     [更新 Local SQLite, synced=1, 避免重複上傳]
```











#### stress test檢測內容跟硬體Code做的修正
```
也請幫我中文詳細解釋從原來的版本(hardware_manager, unified_driver, hardware_config)我做了那些改變能讓我的stress test從原本會失敗到100%成功
```


###### 對硬體hardware_manager.py, unified_driver.py的改動

|                                                     |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 流量塑形與資料降載 (Traffic Shaping & Data Reduction)        | 這是解決 1500 MTU 網路環境下 UDP 微突發 (Micro-bursts) 導致死鎖的根本物理原因。<br><br>- **檔案改動**：`hardware_managers.py` 中的 `_apply_default_settings`。<br>    <br>- **舊版問題**：舊版程式強制要求相機傳送 `BGR8` 或 `RGB8` 格式。對於高畫素相機來說，按一次快門會瞬間產生數十 MB 的資料，以全速衝撞頻寬。<br>    <br>- **新版解法**：<br>    <br>    - **Bayer 優先**：加入了 `['BayerRG8', 'BayerGB8', ...]` 作為第一優先選項。這直接把網路傳輸量砍掉了 66%，讓 Switch 得以喘息。<br>        <br>    - **硬體頻寬閥門**：新增了 `DeviceLinkThroughputLimit` 設定為 80MB/s (`80000000`)。這等於在相機的出口裝了一個水龍頭，嚴格限制每秒流出的封包量。<br>        <br>    - **封包間隔延遲**：加入了 `GevSCPD.value = 25000`，強迫相機在發送每個 UDP 封包之間停頓微秒級的時間。                                                                                                                                                                      |
| 異步防護罩與自我療癒 (Asynchronous Protection & Self-Healing) | - **檔案改動**：`hardware_managers.py` 中的 `capture_image` 函數。<br>    <br>- **舊版問題**：舊版直接在主執行緒呼叫 `self.active_camera.fetch()`。只要網路剛好掉了一個封包，C++ 驅動就會進入無限期等待（死鎖），導致整個 Python 腳本永久凍結。<br>    <br>- **新版解法**：<br>    <br>    - **隔離執行緒**：我們用 `concurrent.futures.ThreadPoolExecutor(max_workers=1)` 把危險的底層取像動作包了起來。<br>        <br>    - **絕對 Timeout 斬斷**：主程式設定了 `hard_timeout` (例如 8 秒)。只要底層發生死鎖，主程式時間一到就會無情切斷。<br>        <br>    - **3 次指數退避重試**：加入了 `for attempt in range(3):` 迴圈。第一次失敗後，系統會等待 0.5 秒，觸發 `self.recover_state()` 執行「焦土政策」銷毀並重建相機連線，然後重新拍攝。這賦予了系統強大的自我修復能力。                                                                                                                                                                               |
| 記憶體環形緩衝區與防呆機制 (Buffer Expansion & Failsafes)        | 解決了「狀態損毀 (State Corruption)」與「讀到舊圖」的問題。<br><br>- **檔案改動**：`hardware_managers.py`。<br>    <br>- **舊版問題**：Harvesters 預設分配的緩衝區太少，且舊版清空 Buffer 的寫法沒有妥善處理 Timeout 例外。<br>    <br>- **新版解法**：<br>    <br>    - **擴充緩衝池**：在 `switch_camera` 中加入了 `self.active_camera.num_buffers = 15`。這極大地增加了 Python 吸收瞬間流量的容錯空間。<br>        <br>    - **硬體級掉包偵測**：在取像迴圈中加入了 `if hasattr(buffer, "is_incomplete") and buffer.is_incomplete:`。這讓我們能在第一時間攔截到破損的影像，而不是把它送去 OpenCV 導致崩潰。<br>        <br>    - **安全的 Buffer 洩洪**：在軟體觸發 (`TriggerSoftware`) 前，使用帶有 `try-except` 的微小 timeout (`0.005`) 迴圈來抽乾殘留影像，確保每次拍到的都是最新畫面。                                                                                                                                                |
| 解耦硬體對焦與防手震 (Decoupled AF & Anti-Vibration)          | 為了應對真實機台高速移動時帶來的物理干擾。<br><br>- **檔案改動**：`unified_driver.py` 與 `hardware_config.yaml`。<br>    <br>- **舊版問題**：舊版把自動對焦邏輯混在主流程中，且缺乏對液態鏡頭硬體特性的支援。<br>    <br>- **新版解法**：<br>    <br>    - **液態鏡頭獨立管線 (Phase 6)**：在 `unified_driver.py` 的 `execute_template_point` 中，新增了專屬的液態鏡頭對焦管線。它會優先嘗試速度極快、成功率極高的 `trigger_hardware_liquid_af()` (FPGA 晶片級對焦)。如果硬體對焦失敗，才會無縫降級為軟體掃描 (`run_liquid_lens_autofocus`)。<br>        <br>    - **執行緒安全鎖**：在 `CameraManager` 的所有關鍵硬體操作（包含 `sweep_liquid_lens` 與 `switch_camera`）都加入了 `with self.capture_lock:`，防止多個 API 請求同時搶奪相機資源。<br>        <br>    - **Zaber 防震升級**：在 `hardware_config.yaml` 中新增了 `anti_vibration` 設定，把 `accel_limit` 和 `decel_limit` 壓低到 50.0，並且將 `settle_time_s` 從 1.0 延長到 5.0 秒，大幅減少馬達急煞造成的鏡頭殘影與電源波動。 |

###### Camera_switch_stress_test.py 檢測內容(跟未來todo)

|                                                     |                                                                                                                                                                                                                                                                                                                                                                              |
| --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 網路底層 MTU 診斷 (Network MTU Diagnostic)                | 測試內容： 發送 8972 bytes 的 ping 封包給相機，強制不准分割 (-f)。<br><br>  <br><br>為什麼有幫助： 這是「照妖鏡」。它能一秒鐘確認從電腦網卡、Switch 到相機這段骨幹，是否真的具備 Jumbo Frame 能力。如果這裡失敗，我們就知道必須依賴後續的軟體限速（如 1500 MTU 與 Bayer 格式）來保命。                                                                                                                                                                                          |
| 硬體資源釋放與切換穩定度 (Camera Switch Stability)              | 測試內容： 不斷在 macro_cam_1、macro_cam_2 與 micro_cam 之間切換 (switch_camera(alias))。<br><br>  <br><br>為什麼有幫助： Opto-Engineering 的 GenTL 驅動如果沒有乾淨地釋放記憶體指標，切換相機時就會引發「狀態損毀 (State Corruption)」。這個測試能確保你的 hardware_managers.py 具備完美切換與重新掛載的能力。                                                                                                                                              |
| 高壓連續取像與微突發驗證 (High-Throughput Capture)              | 測試內容： 每台相機連續要求 90 張照片。這數字完美對應了 V2 Moonlight 系統每隻手錶需要拍攝約 90 張照片的生產需求。<br><br>  <br><br>為什麼有幫助： 網路塞車（微突發 Micro-bursts）通常在連續高負載時才會發生 。這個測試驗證了我們寫入相機底層的「80MB/s 頻寬限制」與「BayerRG8 格式降載」是否真的能撫平流量，防止 1500 MTU 塞車死鎖。                                                                                                                                                                  |
| 異步防死鎖保護罩 (Asynchronous Deadlock Protection)         | 測試內容： 將拍照動作放入 ThreadPoolExecutor 中，並設定 8 秒的 hard_timeout。<br><br>  <br><br>為什麼有幫助： 這是系統的「逃生艙」。如果 C++ 底層因為任何原因卡死，它能確保 Python 主程式不會跟著陪葬，並且會自動呼叫 recover_state() 踢醒相機，讓自動化流程能繼續往下走。                                                                                                                                                                                             |
| 延遲與抖動分析 (Latency & Jitter Analysis)                 | 測試內容： 計算 Median（中位數）、Max（最大值）與 P95 延遲。<br><br>  <br><br>為什麼有幫助： 在工業自動化中，系統必須有「確定性」。P95 延遲極低代表系統沒有抖動（Jitter），這能確保未來串接馬達移動與燈光控制時，時序不會錯亂。                                                                                                                                                                                                                                       |
| ==新增TODO==                                          |                                                                                                                                                                                                                                                                                                                                                                              |
| 複合動作干擾測試 (Compound Movement & EMI Stress Test)      | 盲點： 電磁干擾 (EMI) 與電源波動。當 Zaber 馬達高速煞車，或是 LTDVE 控制器瞬間擊發高亮度的 Strobe 閃光時，可能會干擾相機的 PoE 供電或網路訊號。<br><br>  <br><br>測試設計： 寫一支腳本，讓 Zaber 的五軸馬達 (stage_L_X, stage_R_Z 等) 隨機高速移動，同時讓 ADAM/LTDVE 燈光狂閃。在這種惡劣環境下，同步呼叫 safe_capture 拍攝 900 張照片，確認相機不會因為電壓波動而掉包或斷線。                                                                                                                           |
| 液態鏡頭與 HDR 綜合高壓測試 (Liquid Lens & HDR Sweep)          | 盲點： 目前的壓力測試只用了單一的 5000us 曝光。但在你的 unified_driver.py 中，實際執行 Template 時會大量使用 HDR 融合 (多重曝光連續拍攝) 以及液態鏡頭的電流掃描 (sweep_liquid_lens)。<br><br>  <br><br>測試設計： 針對配備液態鏡頭的 micro_cam，寫一支迴圈腳本：不斷改變電流 (例如 -30mA 到 +30mA)，每改一次就拍一組 HDR 照片 (如 2500us, 5000us, 10000us)。這能測試 CameraManager 中 capture_lock 執行緒鎖的穩定度，以及相機在極短時間內不斷切換曝光參數是否會當機。                                                    |
| 24 小時耐久與記憶體洩漏測試 (24/7 Endurance & Memory Leak Test) | 盲點： Python 呼叫 C++ (GenTL/Harvesters) 最容易在邊界發生記憶體洩漏 (Memory Leak)。即使一兩個小時沒事，連跑 24 小時後 RAM 可能會被吃光導致崩潰。<br><br>  <br><br>測試設計： 將目前的 camera_switch_stress_test.py 加上 Python 的 psutil 或 memory_profiler 模組。設定 cycles=100 讓它在洛杉磯主機上連跑一整個晚上。腳本每跑完一個 Cycle 就記錄一次當前 Python process 所佔用的 RAM。如果 RAM 呈現線性穩定增長，就代表 Harvesters 在底層有記憶體洩漏，必須在 recover_state() 中加入定期的 destroy() 焦土政策來釋放記憶體。 |
|                                                     |                                                                                                                                                                                                                                                                                                                                                                              |



#### 那我修改的硬體code運作機制解答
```
所以相機輸出無壓縮的 24-bit RGB8 全彩格式跟BayerRG8的 8-bit 資料差別在哪裡? BayerRG8是最原始的8bit資料, 所以24-bit RGB8 全彩格式是在相機裡面加工成的資料? 那我們改成 BayerRG8會不會最後我們得到的image是比較缺少資訊的image(譬如color range縮小 或已經壓縮過 或比較不精細?) 所以我們加入DeviceLinkThroughputLimit = 80000000跟GevSCPD = 25000會讓每張照片的產生image時間大概多多少呢? 多少%? 所以加入非同步防護罩 (safe_capture)是指如果我有一張image譬如少了幾個封包, 是直接捨棄這張image進行下一張image or 重新執行拍照同一張image? 現在的self.active_camera.num_buffers = 15那本來預設可能是多少? 他們之間的差別在哪裡能否用具體例子解釋?
```
這幾個問題問得非常深入且切中要害！你完全抓住了機器視覺（Machine Vision）底層架構的核心關鍵。

我來為你逐一解開這些硬體與軟體之間的運作機制，保證不會有任何資訊或畫質的妥協：

### 1. BayerRG8 vs. RGB8：真的會損失畫質嗎？

**結論：完全不會損失任何畫質、色彩範圍或精細度，而且還更好。**

- **物理真相：** 所有的工業相機感光元件（Sensor），其物理結構本身**只能感知光線的強弱（灰階），無法感知顏色**。為了拍出彩色照片，感光元件上方會覆蓋一層「拜爾濾色鏡（Bayer Filter）」。也就是說，每一個實體像素其實只負責接收**紅、綠、或藍**其中一種顏色的光，這就是最原始的 8-bit `Bayer` 資料。
    
- **RGB8 怎麼來的（相機端加工）：** 當你設定為 `RGB8` 時，相機會利用內建的微處理器（FPGA）進行「解馬賽克（Demosaicing）」的猜色演算法：利用周圍的紅色和綠色像素，去算出缺少的藍色數值，最後將 1 個像素強行膨脹成 3 個像素（R, G, B 各 8-bit）。
    
- **為什麼我們改用 BayerRG8 更好？**
    
    1. **零壓縮、零失真：** `BayerRG8` 就是感光元件最純粹、最原始的 1:1 物理訊號。
        
    2. **拯救網路：** 既然 `RGB8` 是算出來的，我們把它交給相機算，相機就必須透過網路傳送 **3 倍** 的龐大資料量給你。我們現在改為傳送 `BayerRG8`，網路頻寬直接省下 66%。
        
    3. **電腦算得更好：** 傳到主機後，我們透過 OpenCV (`cv2.cvtColor`) 讓主機強大的 CPU 去做解馬賽克。電腦的浮點運算能力比相機晶片強得多，最終得到的全彩照片品質絕對等於、甚至優於相機直出的 `RGB8`。
        

### 2. 頻寬限制與延遲設定：會拖慢多少時間？

加入了 `DeviceLinkThroughputLimit = 80000000` (80MB/s) 與 `GevSCPD = 25000` 確實是一種「主動降速」來配合 1500 MTU 交換器防塞車的手段。

- **時間增加了多少？** 以你的 2000 萬畫素相機 `macro_cam_2` 為例，一張 `BayerRG8` 的照片大小剛好是 20MB。
    
    - 如果不限速（Gigabit 網路物理極限約 115MB/s）：傳輸時間約為 `20 / 115 = 0.17 秒`。
        
    - 限速在 80MB/s 後：傳輸時間為 `20 / 80 = 0.25 秒`。
        
    - **單張照片的傳輸時間增加了約 0.08 秒（約慢了 40%）。**
        
- **但整體效能是「倒賺」的：** 如果你不用 Bayer 格式，硬傳 60MB 的 `RGB8` 照片，就算不限速也需要 `60 / 115 = 0.52 秒`。所以，透過「降載 + 限速」的組合拳，你現在每張照片反而比舊版快了 0.1 到 0.5 秒以上！
    

### 3. 非同步防護罩 (`safe_capture`)：掉包是捨棄還是重拍？

**是「重拍」，而且有 3 次機會。**

在你的 `hardware_managers.py` 的 `capture_image` 函數中，我們包裝了一個 `for attempt in range(3):` 的迴圈。 如果 `fetch()` 抓回來的 `buffer.is_incomplete` 顯示有封包遺失，或是發生了 Timeout 死鎖，系統的處置流程是：

1. **捨棄這張壞掉的圖**（絕對不讓壞圖進入後端 AI 辨識）。
    
2. **呼叫 `self.recover_state()`**：這會執行焦土政策，切斷並重新建立一次相機連線，把卡在相機或網卡裡的錯誤封包徹底清空。
    
3. **重新執行同一次拍照**。 只有當「連續重拍 3 次都失敗」時，防護罩才會徹底放棄這張照片（回傳 `None` 或假圖），並讓流程繼續往下走。這確保了在生產線上，極小機率的網路抖動不會造成漏檢。
    

### 4. `num_buffers = 15`：預設是多少？差別在哪？

Harvesters 函式庫針對記憶體環形緩衝區（Ring Buffer）的**預設值通常是 3 到 5 個**。我們現在把它強制拉高到了 15 個。

- **具體例子（壽司迴轉台）：** 想像相機是「壽司師傅」，你的 Python 程式是「吃壽司的客人」，網路是「迴轉台」，而 Buffer 就是迴轉台上的「盤子」。
    
    - **預設情況（3 個 Buffer）：** 壽司師傅（相機）連續捏了 3 盤壽司放上台子。但客人（Python）這時剛好在忙（例如作業系統在寫入硬碟、或是 CPU 正在算別的東西卡住了 0.1 秒）。師傅要放第 4 盤時，發現台子上已經沒有空盤子了！相機只好把第 4 張照片直接丟進垃圾桶（這叫 Dropped Frame）。
        
    - **改為 15 個 Buffer 後：** 台子上有 15 個空盤子。即使 Python 客人暫時發呆了半秒鐘，相機依然可以安穩地把連續拍攝的 15 張照片依序放上台子，一張都不會漏。等 Python 醒過來，就可以一口氣把 15 張照片全部收走。這給了作業系統極大的彈性與容錯空間。
        

這四個機制的結合，就是你這套 Moonlight 系統能夠達到 100% 穩定度的底層魔法。
