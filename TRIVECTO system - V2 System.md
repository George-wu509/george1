
|                                                      |     |
| ---------------------------------------------------- | --- |
| [[#### V2整體系統架構]]                                    |     |
| [[#### 費用評估跟多Local system]]                          |     |
| [[#### 具體操作分析- 用remote client操作local system拍照image]] |     |
|                                                      |     |
|                                                      |     |
|                                                      |     |


#### V2整體系統架構
```
請中文詳細介紹系統聚焦在cloud, datamanager, database方面並以30年tech lead分析並提供建議這系統是否有缺點或可優化的地方. 這是一套可以對watch的各個部位拍照(約30~40張)並以此分析真偽的系統, 預計系統應該要能承載30萬watchs的 容量, 在cloud主要就是3套database, 主要imaging processing tasks是放在local system進行.
```
**總體評價：** 這是一套設計相當成熟的 **混合雲（Hybrid Edge-Cloud）** 架構。你做出了正確的架構決策：在邊緣端（Local）處理高頻寬、低延遲的硬體控制與影像處理，僅將關鍵數據（Metadata）與最終檔案（S3）同步至雲端。這種「**Local Execution, Cloud Orchestration**」的模式是工業級 AOI（自動光學檢測）系統的標準且穩健的解法。

針對你的目標 **「承載 30萬隻手錶容量（約 1200萬張影像）」**，目前的系統在邏輯上是可行的，但在 **擴展性（Scalability）**、**吞吐量（Throughput）** 和 **長期維護成本** 上，我看見了幾個潛在的瓶頸與優化空間。

以下是針對 Cloud、DataManager 與 Database 三個維度的詳細分析與 Tech Lead 建議：

---

### 1. Database 架構分析 (DynamoDB & SQLite)

你的設計採用了 **雙層數據庫策略**：Local SQLite 作為寫入緩衝（Write Buffer），Cloud DynamoDB 作為全域索引（Master Index）。

#### ✅ 優點 (Pros)

- **Split-Key Pattern (核心亮點)**：在 `WatchRecordManager` 和 `CloudDatabaseManager` 中，你將單一巨大的 Watch 物件拆解為多個 DynamoDB Items (`INFO`, `VIEW#Front`, `MAT#Case-Body`)。這非常高明，避免了 DynamoDB 單一 Item 400KB 的限制，且允許針對特定部位（Part）進行獨立讀寫，大幅節省 RCU/WCU（讀寫成本）。
    
- **Command Log 模式**：利用 DynamoDB `WatchCommandLog` 來追蹤 AWS IoT 的指令狀態，這解決了 MQTT "Fire and Forget" 不可靠的問題，提供了良好的可追溯性。
    

#### ⚠️ 缺點與風險 (Cons & Risks)

1. **DynamoDB Scan 的隱患 (`scan_all_templates`)**：
    
    - **問題**：在 `scan_all_templates` 中使用了 `table.scan()`。雖然目前 Template 可能不多，但隨著系統運作，版本控制（v1, v2...）會導致 Template 數量增長。`Scan` 是全表掃描，隨著數據量增加，這會變得極慢且昂貴。
        
    - **風險**：當你有 30萬隻錶的數據後，如果在某些查詢誤用了 Scan，會瞬間耗盡你的 Read Capacity，導致生產線停擺。
        
2. **Access Pattern 設計不足**：
    
    - **問題**：目前 PK 是 `watch_id`。如果你想查詢「所有 Rolex Submariner 的數據」或「所有 2025年生產的手錶」，目前的架構無法高效支援。
        
    - **風險**：對於分析師來說，數據變成了「孤島」，只能透過 ID 查找，無法進行聚合分析。
        

#### 🚀 Tech Lead 建議

- **優化 1：消除 Scan，改用 GSI**
    
    - 在 Template 表中，增加 GSI (Global Secondary Index)，例如 PK=`ActiveStatus` (值為 1)。這樣你只需要 `Query` 狀態為活躍的 Template，而不是掃描整張表。
        
- **優化 2：引入 GSI 支援多維查詢**
    
    - 在主表 `WatchAnalysisResults` 增加 GSI：
        
        - GSI1: PK=`SeriesName`, SK=`CreatedTime` (查詢特定系列的時間軸)。
            
        - GSI2: PK=`Status`, SK=`CreatedTime` (查詢待審核或異常的手錶)。
            
- **優化 3：冷熱數據分離 (TTL)**
    
    - 你的 `Command` 表已經設定了 TTL（7天），這很好。建議對 `Raw Images` 的索引也考慮 TTL 或封存策略，因為 30萬隻錶 x 40張圖 = 1200萬條索引，若全部放在 DynamoDB 熱區，儲存成本會隨時間線性增加。
        

---

### 2. Data Manager & Sync 架構分析

這是系統的「心臟」，負責將本地 SQLite 的數據搬運到 S3。

#### ✅ 優點 (Pros)

- **Async Push 策略**：使用 `synced=0` 標記，由背景執行緒（Background Thread）輪詢並上傳。這確保了即使網路斷線，生產線（Local）依然能繼續拍照，網路恢復後自動補傳。這對工廠環境至關重要。
    
- **加密整合**：`DataEncryptor` 支援 AES-GCM 且實作了安全的 `get_decrypted_temp_path`，在安全性與效能間取得了平衡。
    

#### ⚠️ 缺點與風險 (Cons & Risks)

1. **單執行緒同步瓶頸 (Single-Threaded Sync)**：
    
    - **問題**：`CloudSyncManager` 只有一個 worker thread。假設一張圖 5MB，上傳需 1秒。30萬隻錶每天若產出 1000隻，每隻 40張圖 = 40,000張圖。單執行緒一天只有 86,400秒，若遇到網路波動，同步速度會跟不上生產速度（Backpressure），導致本地硬碟被塞滿。
        
    - **現狀**：程式碼中 `cursor.execute('... LIMIT 50')` 每次取 50 筆，但上傳是同步阻塞的 (`s3_client.upload_file`)。
        
2. **SQLite 並發鎖定 (Concurrency Locking)**：
    
    - **問題**：雖然你為每個操作開了新的 connection，但在高負載下（例如：Robot 快速拍照寫入 + Sync Thread 頻繁讀取更新），SQLite 預設的 Journal Mode 可能會導致 `database is locked` 錯誤。
        

#### 🚀 Tech Lead 建議

- **優化 1：實作多執行緒上傳 (Upload Thread Pool)**
    
    - 不要在主 Loop 裡直接 `upload_file`。將待上傳的任務放入一個 `ThreadPoolExecutor` (建議 4-8 workers)，並發上傳至 S3。S3 的吞吐量非常大，單執行緒浪費了頻寬。
        
- **優化 2：開啟 SQLite WAL 模式**
    
    - 在 `_init_db` 連線後，執行 `PRAGMA journal_mode=WAL;`。這允許多個讀取者與一個寫入者同時進行，大幅減少 Sync Thread 卡住拍照流程的機率。
        
- **優化 3：S3 Transfer Acceleration**
    
    - 如果是跨國傳輸（例如工廠在亞洲，Bucket 在美東），建議開啟 S3 Transfer Acceleration，利用 AWS 骨幹網路加速大檔傳輸。
        

---

### 3. Cloud (AWS) 架構與儲存分析

#### ✅ 優點 (Pros)

- **Athena 整合**：你在 `data_manager/athena_query.py` 預留了 Athena 查詢介面。這是處理 30萬隻錶數據分析的正確選擇。不要在 DynamoDB 做分析，要在 S3 上用 SQL (Athena) 做。
    
- **S3 結構清晰**：`watch_id/Raw/Project/Step/File` 的結構有利於 Athena 的 Partitioning。
    

#### ⚠️ 缺點與風險 (Cons & Risks)

1. **S3 List 效能問題**：
    
    - **問題**：當單一資料夾（Prefix）下的檔案數量超過 10萬級別時，S3 的 `ListObjects` 效能會下降。雖然你有分層，但如果 `Analysis` 資料夾下累積了太多小檔（報告、縮圖），管理會變慢。
        
2. **Athena 查詢成本**：
    
    - **風險**：若 S3 數據沒有依照「時間」或「型號」做 Partition (例如 `year=2025/month=01/`)，Athena 每次查詢都會掃描整個 Bucket。對於 1200萬個檔案，這會讓每次 SQL 查詢變得非常昂貴且緩慢。
        

#### 🚀 Tech Lead 建議

- **優化 1：實作 S3 Partitioning**
    
    - 修改 `s3_key` 的生成邏輯。
        
    - **原**：`{watch_id}/Raw/...`
        
    - **改**：`Data/Year=2025/Month=10/Day=15/{watch_id}/...`
        
    - 這讓 Athena 可以 `WHERE year=2025`，只掃描極少部分數據，速度快 100倍，成本降 99%。
        
- **優化 2：S3 Lifecycle Policy**
    
    - 設定規則：Raw Images（大檔）在 30天後轉入 **S3 Intelligent-Tiering** 或 **Glacier Instant Retrieval**。這能幫你在 30萬隻錶的規模下節省 40%-60% 的儲存成本。
        

---

### 4. 總結與下一步 (Summary & Next Steps)

這套系統的代碼品質很高，結構模組化做得很好（Hardware, Logic, Data 分離）。作為 30年 Tech Lead，我給予 **A-** 的初始評價。剩下的 **+** 取決於面對「規模化」時的細節處理。

**關鍵缺點清單 (按優先級排序)：**

1. **同步效能**：單執行緒上傳無法支撐 30萬隻錶的高通量生產。
    
2. **數據庫鎖定**：SQLite 未顯式開啟 WAL 模式，高併發有風險。
    
3. **大數據檢索**：DynamoDB 缺乏 GSI，S3 缺乏 Partition 結構，導致未來數據分析困難且昂貴。
    

**如果我是你的 Tech Lead，我會要求你下週先做這件事：**

> **「請修改 `CloudSyncManager`，將上傳邏輯改為 `ThreadPoolExecutor` 並發處理，並在 SQLite 中啟用 WAL 模式。這兩個改動改動極小，但能立即使系統的吞吐量提升 3-5 倍，確保上線後的穩定性。」**

這套系統基礎非常穩固，經過上述優化後，絕對有能力承載 30萬甚至 100萬隻手錶的檢測任務。




#### 費用評估跟多Local system
```
請幫我評估一下在AWS cloud上有3個database, 還有儲存camera taken images(以10萬watch來算)及image and meta data results, 以及其他的AWS支出, 各每月的支出大概是多少. 另外就是如果我們的local system有兩台(一台在new york, 一台在germany). 這系統是否還適用多個local system, 這樣是不是AWS上本來的DynamoDB就要變成6個? 其他的也要變成雙份
```
這是一個非常務實且關鍵的成本與架構評估問題。作為 Tech Lead，我會將評估分為兩個部分：**「AWS 每月成本估算」** 與 **「多地部署（Multi-Region）架構策略」**。

---

### 第一部分：AWS 每月成本估算 (以 10萬隻手錶為基準)

**假設前提：**

1. **總儲存量**：系統已累積 100,000 隻手錶的歷史數據。
    
2. **單隻手錶數據量**：
    
    - 影像：35 張 x 5MB (高解析度工業相機) = 175 MB。
        
    - Metadata (DynamoDB)：約 50 KB (JSON 文字數據極小)。
        
3. **每月新增量 (Throughput)**：假設工廠每月新生產 **10,000** 隻手錶 (寫入流量)。
    
4. **區域**：AWS US-East-1 (N. Virginia)。
    

#### 1. S3 儲存成本 (成本大頭)

這是此系統最昂貴的部分。

- **總容量**：100,000 隻 x 175 MB = **17,500,000 MB ≈ 17.5 TB**。
    
- **定價 (Standard Tier)**：$0.023 USD / GB。
    
- **計算**：17,500 GB x $0.023 = **$402.5 USD / 月**。
    

> Tech Lead 建議優化：
> 
> 手錶影像通常是「寫入一次，偶爾讀取」。建議啟用 S3 Intelligent-Tiering。若 30 天後未存取，會自動轉入 Infrequent Access 層 ($0.0125/GB)。
> 
> - **優化後預估**：約 **$220 ~ $250 USD / 月** (假設 90% 數據變冷)。
>     

#### 2. DynamoDB 成本 (意外地便宜)

DynamoDB 對於結構化文字數據非常便宜，除非你使用了錯誤的查詢方式 (Scan)。

- **儲存成本**：100,000 隻 x 50 KB = 5 GB。前 25GB 免費，所以 **$0 USD**。
    
- **寫入成本 (WCU)**：每月新增 10,000 隻錶。
    
    - Split-Key 設計下，每隻錶約寫入 50 個 Items。總寫入 500,000 次。
        
    - 使用 On-Demand 模式 ($1.25 / 百萬次寫入)。
        
    - 成本：< **$1.00 USD**。
        
- **讀取成本 (RCU)**：假設每月有 500 萬次讀取操作。
    
    - 成本：< **$2.00 USD**。
        

#### 3. Data Transfer (流量費 - 隱形殺手)

- **上傳 (Inbound)**：免費。
    
- **下載 (Outbound)**：如果你在辦公室 (Local) 查看雲端照片，或用 Athena 分析。
    
    - 假設每月需查看/下載 10% 的數據 (1.75 TB)。
        
    - 定價：$0.09 USD / GB。
        
    - 計算：1,750 GB x $0.09 = **$157.5 USD**。
        

#### 4. 其他雜項 (Athena, IoT Core, CloudWatch)

- **Athena**：$5.00 / TB scanned。如果你沒有做 Partitioning (如前文建議)，每次查詢掃描全桶，費用會爆炸。若有優化，估計每月 **$50 USD**。
    
- **AWS IoT Core**：維持連線與 MQTT 訊息。以你的規模，每月約 **$10 - $20 USD**。
    

#### 💰 每月總支出預估表

|**項目**|**預估費用 (USD)**|**備註**|
|---|---|---|
|**S3 Storage**|**$250 - $400**|取決於是否開啟自動分層存儲|
|**Data Transfer**|**$150**|取決於下載查看照片的頻率|
|**Athena Analysis**|**$50**|取決於 SQL 查詢次數|
|**DynamoDB**|**$5**|幾乎可以忽略不計|
|**IoT & Logs**|**$20**|CloudWatch Logs 需設定過期時間|
|**總計 (Total)**|**約 $475 - $625 / 月**|台幣約 1.5萬 ~ 2萬元|

---

### 第二部分：多地部署架構評估 (New York & Germany)

**問題核心：** 如果有兩套 Local System (NY, Germany)，雲端的 DynamoDB 是否要變成 6 個 (3x2)？

Tech Lead 直接回答：

絕對不需要，也不建議變成 6 個。

#### 1. 為什麼維持「單一」雲端資料庫？

- **數據孤島 (Data Silo)**：如果你建立了 `Table_NY` 和 `Table_DE`，未來老闆問：「我想看全球 Rolex 的總良率」，你就必須寫程式去兩個表抓資料再合併，這維護起來是災難。
    
- **DynamoDB 的設計哲學**：DynamoDB 是 NoSQL，它透過 **Partition Key (PK)** 來區分數據。
    
- **正確做法**：在現有的 Table 中，透過欄位區分來源。
    
    - 在 Metadata 中增加欄位：`"site": "NY"` 或 `"site": "Germany"`。
        
    - 或者在 `device_id` 中區分：`NY_Station_01`, `DE_Station_01`。
        

#### 2. 架構挑戰與建議

雖然 Table 不用變，但跨國架構有兩個必須考慮的挑戰：**延遲 (Latency)** 與 **法規 (GDPR)**。

##### A. 延遲問題 (Latency)

- **場景**：德國的機器上傳照片到美國 (US-East-1) 的 S3。
    
- **影響**：上傳速度會比紐約慢，但在你的架構中，因為採用了 **Async Cloud Sync (背景上傳)**，這對生產線速度 **完全沒有影響**。Local System 拍完照存 Local DB 就繼續下一張，背景慢慢傳即可。
    
- **建議**：如果上傳實在太慢，可以開啟 S3 Bucket 的 **Transfer Acceleration** 功能 (利用 AWS 全球邊緣節點加速上傳)，但這會增加額外費用。
    

##### B. 關鍵紅線：GDPR (歐盟一般資料保護規則)

這是我最擔心的部分。**德國的數據受到嚴格的 GDPR 管轄。**

- **風險**：如果你的手錶數據中包含「操作員資訊 (User ID)」、「客戶資訊 (Client Name)」等個資，將這些數據直接傳到美國 (US-East-1) **可能是違法的**，或者需要極其複雜的合規宣告。
    
- **Tech Lead 架構建議 (合規版)**：
    
    1. **S3 分離 (Data Residency)**：
        
        - 建立一個新的 S3 Bucket 在 **法蘭克福 (eu-central-1)**。
            
        - 德國的 Local System 設定 `system_config.yaml` 指向這個德國 Bucket。
            
        - 美國的 Local System 繼續用美東 Bucket。
            
    2. **DynamoDB 集中 (Metadata Only)**：
        
        - Metadata (分析結果、良率、機器狀態) 通常不含敏感個資，可以集中存在美國的 DynamoDB。
            
        - **注意**：確保 `user_id` 在上傳前進行 Hash 或加密 (如你的 `AuditLogger` 所做的)，不要傳送德國員工的真實姓名。
            

#### 3. 雙地部署架構圖 (Summary)

Plaintext

```
[Local System: NY]  ----(寫入)----> [S3 Bucket: US-East-1] (Images)
        |
        +---------------(寫入)----> [DynamoDB: US-East-1] (Metadata/Command) <---+
                                                                                 |
[Local System: Germany] --(寫入)--> [S3 Bucket: EU-Central-1] (Images/GDPR) -----+
```

**結論建議：**

1. **Database**：維持 **3 個 Table** (集中在美東)。不要拆分，利用 `site` 欄位區分數據。
    
2. **Storage**：建議拆分為 **2 個 Bucket** (美東、法蘭克福) 以符合物理距離優化與 GDPR 法規。
    
3. **Config**：你的 `system_config.yaml` 設計得很好，只需要在德國機台的 Config 中修改 `aws.s3_bucket` 和 `aws.region` 即可，程式碼完全不用改。
    

這套系統的擴展性非常好，只要你在 `CloudSyncManager` 支援不同 Region 的 S3 Client 初始化 (這在 boto3 是預設支援的)，增加地點幾乎是「設定檔」層級的工作，無需重寫代碼。




#### 具體操作分析- 用remote client操作local system拍照image
```
請幫我中文詳細解釋這個具體操作有這系統的那些部分, 哪些classe跟那些重要操作涵蓋在裡面, 以及怎麼交互的, 也請流程一步步解釋: 打開用remote client操作local system拍照image包含上傳images到cloud
```
這是一個非常好的問題。這個流程展現了這套系統的核心架構：**「控制流 (Control Plane) 與 數據流 (Data Plane) 分離」** 以及 **「非同步處理 (Async Processing)」** 的設計哲學。

以下我將針對「**Remote Client 操作 Local System 拍照並上傳 Cloud**」這個具體案例，詳細拆解涉及的 **類別 (Classes)**、**檔案 (Files)** 以及 **詳細交互流程**。

---

### 1. 核心角色與類別 (Key Actors & Classes)

在這個流程中，主要有 5 個關鍵角色互相合作：

| **角色**            | **核心 Class**         | **所在檔案**                                     | **職責**                                            |
| ----------------- | -------------------- | -------------------------------------------- | ------------------------------------------------- |
| **發令官 (Client)**  | `WorkflowProxy`      | `core/workflow_proxy.py`                     | 位於遠端 (Remote)，負責打包指令並發送 (透過 HTTP 或 AWS IoT MQTT)。 |
| **指揮官 (Server)**  | `WorkflowManager`    | `core/workflow_manager.py`                   | 位於本地 (Local)，系統大腦。接收指令，協調硬體與數據層。                  |
| **驅動者 (Driver)**  | `RealHardwareDriver` | `Controller/hardware_drivers/real_driver.py` | 位於本地，直接控制相機與馬達，回傳影像數據 (NumPy Array)。              |
| **倉儲管 (Storage)** | `DataManager`        | `data_manager/local_storage.py`              | 位於本地，負責檔案整理、加密、存入 SQLite (Buffer)。                |
| **搬運工 (Sync)**    | `CloudSyncManager`   | `data_manager/cloud_sync.py`                 | 位於本地背景執行緒，負責掃描 SQLite 並將檔案搬運至 AWS S3 與 DynamoDB。  |

---

### 2. 詳細流程拆解 (Step-by-Step Walkthrough)

假設情境：你在紐約 (Remote)，透過 Client App 按下「**Manual Capture**」按鈕，命令德國的機器 (Local) 拍一張照。

#### **階段一：指令發送 (Remote Side)**

1. **使用者觸發**：
    
    - 呼叫 `WorkflowProxy.manual_capture(watch_id="Rolex001", cam_id="macro")`。
        
2. **打包指令**：
    
    - `WorkflowProxy` 判斷模式為 `aws_iot`。
        
    - 它將參數打包成 JSON Payload：`{"action": "manual_capture", "watch_id": "Rolex001", ...}`。
        
    - **加密簽章**：呼叫 `_sign_payload` 對指令進行 HMAC 簽章 (安全性)。
        
3. **發送 (Publish)**：
    
    - 呼叫 `_submit_iot_command`。
        
    - 使用 `boto3.client('iot-data')` 將指令 Publish 到 MQTT Topic `cmd/Rolex_Station_001`。
        
    - _此時 Client 進入等待狀態 (或非同步回傳)，指令在雲端飛翔。_
        

---

#### **階段二：接收與調度 (Local Side - Control Plane)**

4. **接收指令**：
    
    - (程式碼中未顯示 `AWSAgent` 監聽迴圈，但邏輯上它會收到 MQTT 訊息)。
        
    - Agent 解析 JSON，驗證簽章，然後呼叫 `WorkflowManager.manual_capture(...)`。
        
5. **執行邏輯**：
    
    - **File**: `core/workflow_manager.py`
        
    - `WorkflowManager` 建立一個暫存路徑 `temp_path` (例如 `Local_Data/temp/uuid.jpg`)。
        
    - 它呼叫硬體層：`self.hw.capture_image(cam_id, exposure, temp_path)`。
        

---

#### **階段三：硬體執行 (Local Side - Hardware)**

6. **硬體動作**：
    
    - **File**: `Controller/hardware_drivers/real_driver.py`
        
    - `RealHardwareDriver` 呼叫 `self.cam.capture_image`。
        
    - `CameraManager` (在 `hardware_managers.py`) 控制工業相機曝光、擷取影像數據。
        
    - 影像被寫入上述的 `temp_path`。
        
    - **關鍵點**：此時影像還只是硬碟裡的一個暫存檔，尚未歸檔。
        

---

#### **階段四：數據歸檔與緩衝 (Local Side - Data Plane)**

7. **數據處理**：
    
    - 硬體返回後，`WorkflowManager` 呼叫 `self.data_mgr.process_and_sync_raw_image(...)`。
        
8. **檔案整理**：
    
    - **File**: `data_manager/local_storage.py`
        
    - `DataManager` 計算最終存放路徑：`Local_Data/Rolex001/Raw/Manual/Unknown/image.jpg`。
        
    - 它將 `temp_path` 的檔案 **Move** 到最終路徑。
        
9. **加密 (可選)**：
    
    - 如果 `encrypt_enabled=True`，呼叫 `DataEncryptor` 對圖檔檔頭 (Header) 進行混淆 (Obfuscation)。
        
10. **寫入本地資料庫 (Write Buffer)**：
    
    - `DataManager` 呼叫 `self.db.insert_raw_image(...)`。
        
    - **File**: `DB/db_manager.py`。
        
    - 在 SQLite 的 `raw_images` 表中插入一筆記錄，**重點是設定 `synced = 0`**。
        
    - 這代表：「檔案已安全存在本地，但還沒上傳」。
        

---

#### **階段五：雲端同步 (Local Side - Background Sync)**

_注意：此階段與上面的階段是非同步並行的，不會卡住下一張拍照。_

11. **背景輪詢**：
    
    - **File**: `data_manager/cloud_sync.py`
        
    - `CloudSyncManager` 有一個 `_worker_loop` 線程一直在跑。
        
    - 它呼叫 `self.db.get_pending_uploads()`，發現了剛剛那筆 `synced=0` 的影像。
        
12. **上傳 S3**：
    
    - `CloudSyncManager` 取得 `local_path` 和目標 `s3_key`。
        
    - 使用 `boto3` 執行 `s3_client.upload_file(...)` 將影像傳至 AWS S3 Bucket。
        
13. **建立雲端索引**：
    
    - 上傳成功後，呼叫 `self.cloud_db.index_record(...)`。
        
    - **File**: `data_manager/cloud_db.py`。
        
    - 在 AWS DynamoDB 的 `WatchAnalysisResults` 表中寫入一筆 Metadata (包含 S3 路徑、拍照參數等)。
        
14. **標記完成**：
    
    - 最後，`CloudSyncManager` 呼叫 `self.db.mark_as_synced(...)`。
        
    - SQLite 中的該筆記錄被更新為 `synced = 1`。
        

---

### 3. 交互流程圖 (Sequence Summary)

程式碼片段

```
sequenceDiagram
    participant Proxy as WorkflowProxy (Remote)
    participant Cloud as AWS IoT / S3 / DynamoDB
    participant Manager as WorkflowManager (Local)
    participant HW as HardwareDriver
    participant DM as DataManager
    participant DB as SQLite (Local)
    participant Sync as CloudSyncManager (Thread)

    Note over Proxy, Sync: 動作開始：手動拍照

    Proxy->>Cloud: 1. Publish Command (MQTT)
    Cloud->>Manager: 2. Receive Command
    Manager->>HW: 3. capture_image()
    HW-->>Manager: Image Saved to Temp
    
    Manager->>DM: 4. process_and_sync_raw_image()
    DM->>DM: Move Temp -> Final Path
    DM->>DM: Encrypt Header (Optional)
    DM->>DB: 5. INSERT raw_image (synced=0)
    DM-->>Manager: Return Success (Local path)
    
    Manager->>Cloud: 6. Update Command Status (Done)
    
    Note over Proxy, Sync: 動作結束：Client 收到完成通知，但上傳還在背景進行

    loop Background Sync
        Sync->>DB: 7. Poll: get_pending_uploads()
        DB-->>Sync: Return [Image Record]
        Sync->>Cloud: 8. Upload File to S3
        Sync->>Cloud: 9. Put Item to DynamoDB (Metadata)
        Sync->>DB: 10. UPDATE raw_image SET synced=1
    end
```

### 總結

這個流程的精髓在於 **第 10 步 (寫入本地 DB)** 與 **第 11 步 (背景同步)** 的解耦。

- **優點**：即使 AWS 網路斷線，Local System 依然可以一直拍照，數據會安全地堆積在 Local SQLite，直到網路恢復後自動消化 (Backpressure Handling)。
    
- **交互核心**：`WorkflowManager` 負責發號施令，`DataManager` 負責記帳，`CloudSyncManager` 負責跑腿。
