

|                                |     |
| ------------------------------ | --- |
| [[#### IIoT特性的自動化光學檢測AOI系統架構]] |     |
|                                |     |
|                                |     |


#### IIoT特性的自動化光學檢測AOI系統架構
```
請幫我中文詳細解釋這個系統的整體規劃及一步步解釋計畫細節
```

這是一套設計非常嚴謹、具備**企業級（Enterprise-Grade）**與**工業物聯網（IIoT）**特性的自動化光學檢測（AOI）系統架構。

根據您提供的程式碼，這個系統是為了**跨國界、跨網路環境（NAT穿透）所設計的。它解決了核心問題：「操作者在遠端（如 Ann Arbor），設備在本地（如 LA 或德國），如何安全、穩定地控制硬體並同步大量影像數據？」**

以下為您詳細解釋整體規劃及一步步的運作細節。

---

### 1. 整體系統架構規劃 (System Architecture)

這套系統採用了 **混合雲端/邊緣運算 (Hybrid Cloud-Edge)** 架構，主要分為三個層面：

1. **控制層 (Control Plane) - AWS IoT MQTT & DynamoDB**
    
    - **目的：** 傳送指令（如「開始檢測」、「拍照」）與回報狀態。
        
    - **技術：** 使用 MQTT 協定（輕量、低延遲、可穿透防火牆），搭配 DynamoDB 記錄指令狀態（Command Log）確保指令不丟失、不重複執行（Idempotency）。
        
2. **數據層 (Data Plane) - AWS S3 & DynamoDB**
    
    - **目的：** 儲存高解析度原圖、分析報告與數據索引。
        
    - **技術：** 邊緣端（Local）先存硬碟，透過 `CloudSyncManager` 背景非同步上傳 S3。DynamoDB 負責記錄圖片的 metadata (索引)。
        
3. **媒體層 (Media Plane) - WebRTC / Snapshot**
    
    - **目的：** 讓遠端操作者看到即時畫面。
        
    - **技術：** 優先使用 WebRTC（低延遲），若網路受限則降級為 S3 Snapshot 輪詢。
        

---

### 2. 核心模組詳解 (Module Breakdown)

根據您提供的檔案，各模組職責如下：

#### A. 入口與設定 (`main_cli.py` & `config/`)

- **`main_cli.py`**：這是系統的**通用入口**。
    
    - 它不只是用來跑檢測，還能根據 `config/system_config.yaml` 中的 `role` (角色) 決定自己是 **Agent** (被控端/連硬體) 還是 **Client** (控制端)。
        
    - 它會呼叫 `core.get_workflow_engine` 工廠模式來載入正確的引擎。
        

#### B. 雲端代理人 (`cloud_relay/aws_agent.py`)

- **角色：** 這是跑在 **Local Computer (LA/德國)** 上的守護行程 (Daemon)。
    
- **功能：**
    
    - 它**訂閱 (Subscribe)** AWS IoT Core 的 MQTT Topic (`cmd/{device_id}`)。
        
    - 收到遠端指令後，它會進行**簽章驗證 (HMAC Signature)** 確保安全性。
        
    - 透過 `_CommandDedupCache` 與 DynamoDB 防止同一個指令被執行兩次。
        
    - 呼叫內部的 `WorkflowManager` 去驅動硬體。
        

#### C. 業務邏輯核心 (`core/workflow_manager.py` & `orchestrator.py`)

- **`WorkflowManager`**：本地端的總指揮。它負責初始化硬體驅動 (`RealHardwareDriver`)、管理數據 (`DataManager`)、並協調檢測流程。
    
- **`Orchestrator`**：負責執行具體的影像分析任務（如 Lume Segmentation, OCR）。它支援多執行緒 (`ThreadPoolExecutor`) 平行處理以加速運算。
    

#### D. 遠端代理 (`core/workflow_proxy.py`)

- **角色：** 跑在 **Remote Client (Ann Arbor)** 的程式碼。
    
- **功能：** 它不直接連硬體，而是把指令打包成 JSON，加上 HMAC 簽名，透過 `boto3` 發送到 AWS IoT Cloud。
    

#### E. 數據管理 (`data_manager/*`)

- **`CloudSyncManager`**：這是關鍵的**可靠性設計**。它維護一個上傳佇列 (Queue)，即使網路斷線，它會 retry，並支援指數退避 (Exponential Backoff)。
    
- **`DataEncryptor`**：實現 **Data-at-Rest Encryption**。圖片在存入硬碟前會進行 Header 混淆或全檔加密 (AES-GCM)，防止商業機密（如未發布的手錶原型）外洩。
    

---

### 3. 一步步解釋計畫細節 (Step-by-Step Execution Flow)

假設情境：**您人在 Ann Arbor (Client)，要控制位於德國 (Local Agent) 的機台對一支 "Rolex Submariner" 執行標準檢測。**

#### 第一階段：指令發送 (Client Side)

1. **操作者執行命令**： 您在 Ann Arbor 的電腦執行：
    
    Bash
    
    ```
    python main_cli.py --command run_routine --watch_id Rolex_Submariner_001 --name Standard_Check
    ```
    
2. **初始化 Proxy**： `main_cli.py` 讀取設定檔，發現 `app_mode: "aws_iot"` 且 `role: "client"`，於是初始化 `WorkflowProxy`。
    
3. **打包與簽章**： `WorkflowProxy` 產生一個唯一的 `command_id` (UUID)。 它使用 `auth_token` 對指令內容 (Action, Timestamp, Requester) 進行 **HMAC-SHA256 簽章**。
    
4. **發送至雲端**： Proxy 透過 MQTT 將 JSON payload 發布到 AWS IoT Topic `cmd/Rolex_Station_001`。同時，它會在 DynamoDB 的 `WatchCommandLog` 資料表中寫入一筆狀態為 `submitted` 的紀錄。
    

#### 第二階段：指令傳輸與接收 (Cloud & Agent Side)

5. **AWS IoT 轉發**： AWS 收到訊息，即時推送到位於德國、已訂閱該 Topic 的 `aws_agent.py`。
    
6. **Agent 驗證**： 德國的 `aws_agent.py` 收到訊息。
    
    - **驗證簽章**：計算 HMAC 是否相符？(防止駭客偽造指令)。
        
    - **防重放攻擊**：檢查 `timestamp` 是否過期？
        
    - **去重 (Deduplication)**：檢查 DynamoDB 或本地 Cache，確認這個 `command_id` 是否已經執行過？
        
7. **狀態更新**： 驗證通過後，Agent 將 DynamoDB 上的指令狀態更新為 `running`，並透過 MQTT 回報 "Busy" 狀態。
    

#### 第三階段：硬體執行與運算 (Local Execution)

8. **WorkflowManager 接手**： Agent 呼叫 `manager.execute_routine(...)`。
    
9. **硬體動作**： 根據 `Standard_Check` 的計畫檔 (YAML)，`WorkflowManager` 控制 Zaber 滑台移動、相機調整曝光、Keyence 觸發拍照。
    
10. **影像處理**： 拍到的照片存入暫存區。`Orchestrator` 呼叫 AI 模型（如 Lume Service, OCR Service）進行分析。
    
11. **數據保存與加密**： `DataManager` 將原圖與分析結果存入 `Local_Data` 資料夾。
    
    - **加密**：使用 `AES-GCM` 對圖片 Header 進行混淆 (Obfuscation)，確保即使硬碟被偷，照片也無法直接開啟。
        
    - **本地索引**：寫入 SQLite (`watchDB.db`)。
        

#### 第四階段：數據同步與回報 (Sync & Report)

12. **雲端同步**： `CloudSyncManager` (背景執行緒) 發現有新檔案。它將加密後的檔案與 Meta 檔上傳至 AWS S3 Bucket。若網路上傳失敗，它會自動重試 (Retry)。
    
13. **完成回報**： 流程結束。Agent 更新 DynamoDB 的 `WatchCommandLog` 為 `completed`，並附上 S3 的報告路徑。 Agent 透過 MQTT 發布 `status/Rolex_Station_001` 告知任務結束。
    
14. **Client 端接收結果**： Ann Arbor 的 `main_cli.py` (原本在輪詢 DynamoDB) 讀取到狀態變為 `completed`，下載報告並顯示結果。
    

---

### 4. 針對「30年軟體規格」的建議與亮點

您的程式碼已經具備非常高的水準，針對長期維護（30年生命週期）與遠端維修，目前的設計有以下亮點與建議：

#### 現有亮點 (已在程式碼中)：

1. **解耦 (Decoupling)**：控制層 (MQTT) 與 數據層 (S3) 分離。即使 S3 上傳慢，也不會卡住控制指令。
    
2. **韌性 (Resiliency)**：
    
    - `aws_agent.py` 中的 `_CommandDedupCache` 確保網路不穩造成重複送封包時，機器不會重複動作（這對硬體保護至關重要）。
        
    - `CloudSyncManager` 實作了 Queue 和 Backoff retry，不怕德國網路突然斷線。
        
3. **安全性 (Security)**：
    
    - **HMAC 簽章**：防止中間人攻擊篡改指令。
        
    - **Data-at-Rest Encryption**：符合高階商業機密保護需求。
        
4. **可觀測性 (Observability)**：
    
    - `AuditLogger` (Hash Chain) 提供了不可篡改的操作軌跡，這對長期維護和責任歸屬非常重要。
        
    - DynamoDB 記錄了完整的指令生命週期 (Submitted -> Running -> Completed)。
        

#### 未來建議 (針對 30 年維護)：

1. **遠端除錯通道 (Remote Debug Tunneling)**：
    
    - 目前的 `aws_agent` 只能收指令。建議未來整合 **AWS IoT Secure Tunneling** 或 **SSH Reverse Tunnel**。
        
    - _場景：_ 當 `main_cli.py` 報錯但無法判斷原因時，工程師可以直接 SSH 進去德國的電腦看 Log 或更新 Docker Container。
        
2. **OTA 更新機制 (Over-The-Air Update)**：
    
    - 30 年軟體一定會更新。建議在 `aws_agent` 增加一個 `update_software` 的 Action，可以讓 Agent 自動從 S3 下載新版 `main_cli` 或 Docker Image 並重啟。
        
3. **看門狗機制 (Watchdog)**：
    
    - 如果 `aws_agent.py` 當機怎麼辦？建議在 OS 層級 (如 Systemd 或 Windows Service) 設定自動重啟，並在 `aws_agent` 內實作 Heartbeat 發送到 AWS CloudWatch。
        

這套系統的邏輯非常清晰，利用 AWS IoT 解決了 IP/Firewall 問題，利用本地 Queue 解決了網路穩定性問題，是一個非常成熟的架構。
