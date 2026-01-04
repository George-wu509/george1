

|                      |     |
| -------------------- | --- |
| [[#### 新DynamoDB架構]] |     |
|                      |     |
|                      |     |


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