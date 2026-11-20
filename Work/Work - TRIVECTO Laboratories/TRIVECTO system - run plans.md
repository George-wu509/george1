


|                                        |     |
| -------------------------------------- | --- |
| [[#### 12個操作Plan的設定跟可行性]]              |     |
| [[#### Network Mode, "run_routine"疑問]] |     |
|                                        |     |
|                                        |     |
|                                        |     |
|                                        |     |
|                                        |     |
|                                        |     |


#### 12個操作Plan的設定跟可行性
```
在這裡附上這個system的核心code. 請幫我分析這裡有12個操作plans, 譬如第一個(plan1)就是使用local system使用存在Local database或Cloud的templates操作Zaber, camera等等hardware拍攝images. 請中文解釋要如何用cli_config.yaml, system_config.yaml或其他檔案控制執行這些plans, 這12個plans裡面是否有無法執行的? 並詳細解釋cli_config.yaml, system_config.yaml的各項設定的意義跟會對執行code造成甚麼影響.

plan1 - use local client use Data base templates, control Zaber camera take images

plan2 - use local client use loaded template yaml files, control Zaber camera take images

plan3 - use local client, use Data base templates, hybrid simulation use images

plan4 - use local client use loaded template yaml files, hybrid simulation use images

plan5 - use local client, all simulation use images

plan6 - use local client, all simulation use images

plan7 - use remote client to control local system, use Data base templates, control Zaber camera take images

plan8 - use remote client to control local system use loaded template yaml files, control Zaber camera take images

plan9 - use remote client to control local system, use Data base templates, hybrid simulation use images

plan10 - use remote client to control local system use loaded template yaml files, hybrid simulation use images

plan11 - use remote client to control local system, all simulation use images

plan12 - use remote client to control local system, all simulation use images
```
這是一個非常完整且架構嚴謹的 Watch Inspection System（手錶自動檢測系統）。

根據提供的代碼，這個系統的架構可以分為三層：

1. **Client Layer (操作層):** `main_cli.py` (本地 CLI) 或 `workflow_proxy.py` (遠端/IoT 客戶端)。
    
2. **Core Layer (邏輯層):** `WorkflowManager`，負責協調流程、數據庫 (`DB`) 與硬體。
    
3. **Hardware Layer (驅動層):** `UnifiedHardwareDriver`，負責根據模式 (`mode`) 控制實體硬體或模擬數據。
    

以下我將針對你提出的 **12 個 Plans** 進行詳細的可行性分析，並解釋如何透過 Config 檔案進行控制。

---

### 第一部分：Config 檔案設定詳解

要控制這 12 個 Plans，你主要需要調整兩個檔案：`config/system_config.yaml` (定義環境與硬體行為) 和 `config/cli_config.yaml` (定義當次執行的任務參數)。

#### 1. `config/system_config.yaml` (環境與硬體模式)

此檔案決定系統「**是什麼**」 (Is Production? Is Hybrid?) 以及「**在哪裡**」 (Local? AWS Agent?)。

|**設定項目**|**關鍵欄位**|**設定值意義**|**影響**|
|---|---|---|---|
|**System Mode**|`system.mode`|`"production"`|**全真模式**：必須連接真實 Zaber/相機。硬體報錯會導致崩潰或報錯。|
|||`"hybrid"`|**混合模式**：優先嘗試連接硬體。若硬體斷線或未安裝，自動降級為模擬 (Mock/Asset) 而不崩潰。適合開發測試。|
|||`"simulation"`|**全模擬模式**：完全不嘗試連接硬體。直接使用 `SimulationHardware`，複製 `assets` 圖片。|
|**Network Mode**|`network.app_mode`|`"local"`|**本地執行**：直接呼叫 `WorkflowManager`。 (`main_cli.py` 行為)|
|||`"remote"`|**HTTP Server**：啟動 FastAPI Server (`workflow_server.py`) 等待指令。|
|||`"aws_iot"`|**AWS IoT**：啟動 `aws_agent.py`，透過 MQTT 接收雲端指令。|
|**Role**|`network.role`|`"agent"`|這是連接硬體的電腦 (被控端)。|
|||`"client"`|這是發送指令的電腦 (控制端，如 `workflow_proxy.py`)。|

#### 2. `config/cli_config.yaml` (執行指令與數據來源)

此檔案決定 `main_cli.py` 「**做什麼**」 (Run Routine?) 以及「**用什麼資料**」 (DB Template? YAML File?)。

|**設定項目**|**關鍵欄位**|**設定值意義**|**影響**|
|---|---|---|---|
|**Command**|`command`|`"run_routine"`|執行檢測流程。|
|**Template Source (File)**|`hybrid_import_yaml`|`"template_B4FC.yaml"` (或任何檔名)|**檔案優先策略**：若此欄位有值，且系統模式為 `hybrid` (或 `main_cli` 強制邏輯)，系統會嘗試從 `DB/templates_import/` 讀取此 YAML 檔案作為流程，**忽略** DB 中的設定。|
|||`null` (或留空)|**DB 優先策略**：若此欄位為空，則使用下方的 `routine_name` 從 SQLite DB 讀取。|
|**Template Source (DB)**|`routine_name`|`"Rolex_Submariner"`|從 Local DB (`local_templates` table) 讀取對應的 Template JSON 執行。|

---

### 第二部分：12 個 Plans 的執行分析

這 12 個 Plans 是由 **Client 位置** x **數據來源** x **硬體模式** 組合而成。


|                                                                                                            |     |                                                                                                                                                                                                                                                                                                                                                                                                                |
| ---------------------------------------------------------------------------------------------------------- | --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **plan1** <br>-------------<br>local client<br><br>DB templates<br><br>control Zaber camera take images    | ok  | - **Config 設定**:<br>    <br>    - `system_config`: mode: **==production==**, app_mode: =="local"==<br>        <br>    - `cli_config`: `hybrid_import_yaml: null`, `routine_name: "Target_Routine_Name"`<br>        <br>- **執行**: 系統從 DB 讀取流程，控制真實 Zaber/相機拍攝                                                                                                                                                   |
| **plan2** <br>-------------<br>local client<br><br>yaml templates<br><br>control Zaber camera take images  | ok  | - **Config 設定**:<br>    <br>    - `system_config`: mode: =="production"==<br>        <br>    - `cli_config`: hybrid_import_yaml: "==template_file.yaml==" (或者執行時用參數 `python main_cli.py --template_path "path/to/file.yaml"`)<br>        <br>- **執行**: 系統直接讀取 YAML 檔案解析步驟，控制真實硬體。<br>    <br>- **注意**: 雖然代碼中 `WorkflowManager.execute_routine` 支援 `template_path`，但通常 `production` 模式建議走 DB 以確保版本一致性，但技術上是可行的。 |
| **plan3** <br>-------------<br>local client<br><br>DB templates<br><br>hybrid simulation use images        | ok  | - **Config 設定**:<br>    <br>    - `system_config`: mode: =="hybrid"==<br>        <br>    - `cli_config`: `hybrid_import_yaml: null`, `routine_name: "Target_Routine_Name"`<br>        <br>- **執行**: 從 DB 讀流程。`UnifiedDriver` 會嘗試連硬體，若沒插相機，會自動用 `assets` 裡的圖片 (由 `config/pipeline_sim_map.yaml` 定義) 假裝拍照成功。                                                                                                     |
| **plan4** <br>-------------<br>local client<br><br>yaml templates<br><br>hybrid simulation use images      | ok  | - **Config 設定**:<br>    <br>    - `system_config`: mode: =="hybrid"==<br>        <br>    - `cli_config`: hybrid_import_yaml: =="template_file.yaml"==<br>        <br>- **執行**: 從 YAML 讀流程，硬體走 Hybrid 模式 (有則動，無則模擬)。                                                                                                                                                                                            |
| **plan5** <br>-------------<br>local client<br><br>all simulation use images                               | ok  | - **Config 設定**:<br>    <br>    - system_config: mode: =="simulation"== (注意：這會切換到 `SimulationHardware` class)<br>        <br>    - `cli_config`: `hybrid_import_yaml: null`<br>        <br>- **執行**: 從 DB 讀流程，`SimulationHardware` 僅單純複製圖片，完全不嘗試連接硬體。                                                                                                                                                          |
| **plan6** <br>-------------<br>local client<br><br>all simulation use images                               | ok  | - **Config 設定**:<br>    <br>    - `system_config`: mode: =="simulation"==<br>        <br>    - `cli_config`: hybrid_import_yaml: =="template_file.yaml"==<br>        <br>- **執行**: 從 YAML 讀流程，全模擬執行。                                                                                                                                                                                                           |
|                                                                                                            |     |                                                                                                                                                                                                                                                                                                                                                                                                                |
| **plan7** <br>-------------<br>remote client<br><br>DB templates<br><br>control Zaber camera take images   | ok  | **設定**: Server `system.mode: "production"`. Client 送出 `routine_name`. <br><br>可執行。Server 從自己的 DB 讀取 Template 執行                                                                                                                                                                                                                                                                                                |
| **plan8** <br>-------------<br>remote client<br><br>yaml templates<br><br>control Zaber camera take images | NO  | - **問題**: Remote Client 讀取本地 YAML，想叫 Server 跑這個 YAML。<br>    <br>- **分析**: 目前 API (`start_routine`) 不支援上傳 YAML 內容。<br>    <br>- **解決方案**: 必須分兩步：(1) Client 呼叫 API 將 Template 存入 Server DB (`create_template` 或 `sync` 機制)。(2) Client 呼叫 `run_routine` 使用該名稱。<br>    <br>- **結論**: ❌ **無法直接執行** (As-is Code)。目前的 API 設計不支援「Remote Client 直接指定 Local YAML 內容讓 Server 跑」。必須依賴 Server 端已有的檔案或 DB。                  |
| **plan9** <br>-------------<br>remote client<br><br>DB templates<br><br>hybrid simulation use images       | ok  | **設定**: Server `system.mode: "hybrid"`. Client 送出 `routine_name`                                                                                                                                                                                                                                                                                                                                               |
| **plan10** <br>-------------<br>remote client<br><br>yaml templates<br><br>hybrid simulation use images    | NO  | 同 Plan 8                                                                                                                                                                                                                                                                                                                                                                                                       |
| **plan11** <br>-------------<br>remote client<br><br>all simulation use images                             | ok  | **設定**: Server `system.mode: "simulation"`. Client 送出 `routine_name`.                                                                                                                                                                                                                                                                                                                                          |
| **plan12** <br>-------------<br>remote client<br><br>all simulation use images                             | NO  | 同 Plan 8                                                                                                                                                                                                                                                                                                                                                                                                       |


#### **Local Client (使用 `main_cli.py`)**

這些 Plan 都是在一台電腦上執行 `python main_cli.py`。

- **Plan 1 (Local, DB Template, Production HW)**
    
    - **Config 設定**:
        
        - `system_config`: `mode: "production"`, `app_mode: "local"`
            
        - `cli_config`: `hybrid_import_yaml: null`, `routine_name: "Target_Routine_Name"`
            
    - **執行**: 系統從 DB 讀取流程，控制真實 Zaber/相機拍攝。
        
    - **狀態**: ✅ **可執行**。
        
- **Plan 2 (Local, YAML File, Production HW)**
    
    - **Config 設定**:
        
        - `system_config`: `mode: "production"`
            
        - `cli_config`: `hybrid_import_yaml: "template_file.yaml"` (或者執行時用參數 `python main_cli.py --template_path "path/to/file.yaml"`)
            
    - **執行**: 系統直接讀取 YAML 檔案解析步驟，控制真實硬體。
        
    - **注意**: 雖然代碼中 `WorkflowManager.execute_routine` 支援 `template_path`，但通常 `production` 模式建議走 DB 以確保版本一致性，但技術上是可行的。
        
    - **狀態**: ✅ **可執行**。
        
- **Plan 3 (Local, DB Template, Hybrid Sim)**
    
    - **Config 設定**:
        
        - `system_config`: `mode: "hybrid"`
            
        - `cli_config`: `hybrid_import_yaml: null`, `routine_name: "Target_Routine_Name"`
            
    - **執行**: 從 DB 讀流程。`UnifiedDriver` 會嘗試連硬體，若沒插相機，會自動用 `assets` 裡的圖片 (由 `config/pipeline_sim_map.yaml` 定義) 假裝拍照成功。
        
    - **狀態**: ✅ **可執行**。
        
- **Plan 4 (Local, YAML File, Hybrid Sim)**
    
    - **Config 設定**:
        
        - `system_config`: `mode: "hybrid"`
            
        - `cli_config`: `hybrid_import_yaml: "template_file.yaml"`
            
    - **執行**: 從 YAML 讀流程，硬體走 Hybrid 模式 (有則動，無則模擬)。
        
    - **狀態**: ✅ **可執行**。
        
- **Plan 5 (Local, DB Template, All Simulation)**
    
    - **Config 設定**:
        
        - `system_config`: `mode: "simulation"` (注意：這會切換到 `SimulationHardware` class)
            
        - `cli_config`: `hybrid_import_yaml: null`
            
    - **執行**: 從 DB 讀流程，`SimulationHardware` 僅單純複製圖片，完全不嘗試連接硬體。
        
    - **狀態**: ✅ **可執行**。
        
- **Plan 6 (Local, YAML File, All Simulation)**
    
    - **Config 設定**:
        
        - `system_config`: `mode: "simulation"`
            
        - `cli_config`: `hybrid_import_yaml: "template_file.yaml"`
            
    - **執行**: 從 YAML 讀流程，全模擬執行。
        
    - **狀態**: ✅ **可執行**。
        

---

#### **Remote Client (使用 Proxy 控制 Server/Agent)**

這些 Plan 涉及兩台機器：Client (發送指令) 和 Server/Agent (連接硬體)。

- **Server 端**: 執行 `workflow_server.py` (HTTP) 或 `aws_agent.py` (IoT)。Config 設定在 **Server 端** 的 `system_config.yaml`。
    
- **Client 端**: 執行 `workflow_proxy.py` 或發送 Request。
    

⚠️ 重大限制發現：

查看 workflow_server.py 的 API 定義：

Python

```
class RoutineRequest(BaseModel):
    watch_id: str
    routine_name: str
    user_id: Optional[str] = None
```

以及 `WorkflowManager.execute_routine` 的簽名：

Python

```
def execute_routine(self, ..., template_path: str = None, ...)
```

雖然 Manager 支援 `template_path`，但 Server 的 HTTP API (`/api/routine/start`) **並沒有** 接受 `template_path` 或 `yaml_content` 的參數。它只接受 `routine_name` (依賴 Server 端 DB)。

這意味著 **Client 無法直接傳送一個本地的 YAML 檔案給 Server 執行** (除非該檔案已經存在於 Server 的檔案系統上，且 Client 知道 Server 上的路徑)。

- **Plan 7 (Remote, DB Template, Production HW)**
    
    - **設定**: Server `system.mode: "production"`. Client 送出 `routine_name`.
        
    - **狀態**: ✅ **可執行**。Server 從自己的 DB 讀取 Template 執行。
        
- **Plan 8 (Remote, Loaded YAML File, Production HW)**
    
    - **問題**: Remote Client 讀取本地 YAML，想叫 Server 跑這個 YAML。
        
    - **分析**: 目前 API (`start_routine`) 不支援上傳 YAML 內容。
        
    - **解決方案**: 必須分兩步：(1) Client 呼叫 API 將 Template 存入 Server DB (`create_template` 或 `sync` 機制)。(2) Client 呼叫 `run_routine` 使用該名稱。
        
    - **結論**: ❌ **無法直接執行** (As-is Code)。目前的 API 設計不支援「Remote Client 直接指定 Local YAML 內容讓 Server 跑」。必須依賴 Server 端已有的檔案或 DB。
        
- **Plan 9 (Remote, DB Template, Hybrid Sim)**
    
    - **設定**: Server `system.mode: "hybrid"`. Client 送出 `routine_name`.
        
    - **狀態**: ✅ **可執行**。
        
- **Plan 10 (Remote, Loaded YAML File, Hybrid Sim)**
    
    - **分析**: 同 Plan 8。
        
    - **結論**: ❌ **無法直接執行** (As-is Code)。
        
- **Plan 11 (Remote, DB Template, All Simulation)**
    
    - **設定**: Server `system.mode: "simulation"`. Client 送出 `routine_name`.
        
    - **狀態**: ✅ **可執行**。
        
- **Plan 12 (Remote, Loaded YAML File, All Simulation)**
    
    - **分析**: 同 Plan 8。
        
    - **結論**: ❌ **無法直接執行** (As-is Code)。
        

---

### 第三部分：無法執行的 Plans 總結與解決方案

無法執行的 Plans: Plan 8, Plan 10, Plan 12

原因：workflow_server.py 和 aws_agent.py 的接口設計是基於 Routine Name (DB Key)，並不支援透過網路傳輸 YAML 檔案內容來即時執行 (Ad-hoc execution)。

解決方案 (Workaround):

如果你希望支援這些 Plans，你需要修改 workflow_server.py 的 /api/routine/start 接口，增加一個 template_override_content 欄位，或者：

1. **Sync First**: Client 先把 YAML 透過 `upload` 接口 (目前代碼未見) 傳到 Server。
    
2. **Shared DB**: Client 和 Server 連接同一個 AWS Cloud DynamoDB。Client 將 YAML 存入 Cloud DB (`WatchTemplates` table)，然後通知 Server 用 `routine_name` 去跑 (Server 會從 Cloud 同步下來)。這是本系統架構中最推薦的做法。
    

### 第四部分：詳細 Config 設定指南

#### 若要執行 Plan 1 (Local, DB, Real HW):

1. **`config/system_config.yaml`**:
    
    YAML
    
    ```
    system:
      mode: "production"   # 開啟真實硬體檢查
    network:
      app_mode: "local"    # 使用 main_cli.py
    ```
    
2. **`config/cli_config.yaml`**:
    
    YAML
    
    ```
    hybrid_import_yaml: null       # 設為 null，強制使用 DB
    routine_name: "Rolex_Submariner" # 確保 DB 有此 Template
    command: "run_routine"
    ```
    
3. **執行**: `python main_cli.py`
    

#### 若要執行 Plan 4 (Local, YAML, Hybrid Sim):

1. **`config/system_config.yaml`**:
    
    YAML
    
    ```
    system:
      mode: "hybrid"       # 允許硬體缺失，使用 Mock/Assets
    ```
    
2. **`config/cli_config.yaml`**:
    
    YAML
    
    ```
    hybrid_import_yaml: "template_B4FC.yaml" # 指定檔案名稱 (需在 DB/templates_import/)
    command: "run_routine"
    ```
    
3. **執行**: `python main_cli.py` (系統會優先讀取該 YAML 檔案內容執行)。
    

#### 若要執行 Plan 7 (Remote, DB, Real HW):

1. **Server 端 (連接硬體的電腦) `config/system_config.yaml`**:
    
    YAML
    
    ```
    system:
      mode: "production"
    network:
      app_mode: "remote"   # 或是 "aws_iot"
      role: "agent"
    ```
    
    _啟動 Server_: `python core/workflow_server.py` (或 `aws_agent.py`)
    
2. **Client 端 (控制端) `config/system_config.yaml`**:
    
    YAML
    
    ```
    network:
      app_mode: "remote"   # 配合 Server 模式
      role: "client"
      server_ip: "192.168.X.X" # Server IP
    ```
    
3. Client 執行代碼:
    
    Client 不使用 main_cli.py (那是本地用的)，而是使用 workflow_proxy.py 或類似腳本：
    
    Python
    
    ```
    from core.workflow_proxy import WorkflowProxy
    proxy = WorkflowProxy()
    proxy.run_routine(watch_id="Rolex_001", routine_name="Rolex_Submariner")
    ```
    

### 總結

- **cli_config.yaml**：控制「這一次」要跑什麼任務（跑哪個 routine，用檔案還是 DB）。
    
- **system_config.yaml**：控制「這台機器」的能力（是否連接硬體，是 Server 還是 Client）。
    
- **Plan 8, 10, 12** 在當前架構下，遠端 Client 不能直接「餵」YAML 給 Server 跑，必須透過「先存入 DB (Cloud Sync)，再呼叫名稱」的方式達成。




#### Network Mode, "run_routine"疑問
```
有些疑問關於Config 檔案設定跟系統.

1. Network Mode="aws_iot"是否可以完全取代Network Mode="local", "remote". 我想確認這些會不會是code更新後無用的設定("local", "remote"), 或者他們有特別的作用"aws_iot"無法取代

2. 所以如果network.app_mode設定不是"aws_iot", 那network.role設定是否就沒有意義了?

3. 如果系統模式不是 hybrid, 那hybrid_import_yaml是否設定甚麼都沒關係, 系統都是從Database讀取templates. 

4. command: "run_routine", "manual_capture",
```
這是一個非常深入且切中架構核心的提問。我將針對您的四個疑問逐一分析。

### 1. Network Mode="aws_iot" 是否可以完全取代 "local" 和 "remote"?

**答案：不行。它們各自有無法被取代的應用場景，並非過時設定。**

這三種模式代表了三種完全不同的**控制拓樸結構 (Topology)**，適用於不同階段或環境：

- **`local` (本地模式)**
    
    - **架構**：沒有網路層，`main_cli.py` 直接呼叫 `WorkflowManager`。
        
    - **不可取代性**：這是**開發與除錯**的基石。當你需要單步執行 (Step-through debug)、測試新寫的 Driver、或者在完全沒有網路的環境下進行硬體測試時，只有這個模式能用。它沒有網路延遲，錯誤訊息直接印在 Console 上，最直觀。
        
- **`remote` (局域網/HTTP 模式)**
    
    - **架構**：`Client` <--(HTTP/REST)--> `Server` (在同一區網內)。
        
    - **不可取代性**：這是**工廠內部部署 (On-Premise)** 的最佳選擇。
        
        1. **低延遲**：HTTP 內網傳輸比繞一圈 AWS Cloud 快非常多。
            
        2. **資安與穩定性**：許多工廠產線是禁止連接外網 (Internet) 的。在這種「內網隔離」的環境下，`aws_iot` 完全無法運作，必須依賴 `remote` 模式來進行機台間的控制。
            
- **`aws_iot` (雲端模式)**
    
    - **架構**：`Client` <--(MQTT)--> `AWS Cloud` <--(MQTT)--> `Agent`。
        
    - **特點**：這主要用於**跨國管理**或**遠端監控**。雖然功能最強大，但依賴 Internet 連線，且有計費與延遲問題。
        

**結論**：`local` 是開發底層，`remote` 是產線實戰首選，`aws_iot` 是遠端運維管理。三者並存，缺一不可。

---

### 2. 如果 `network.app_mode` 不是 "aws_iot", 那 `network.role` 設定是否就沒有意義了?

**答案：是的，您是對的。**

- **在 `local` 模式下**：程式直接運行 `WorkflowManager`，本機既是發號施令者也是執行者，不需要區分 Role。
    
- **在 `remote` 模式下**：角色的區分是透過**你執行哪個 Python 檔案**來決定的。
    
    - 執行 `workflow_server.py` = 你就是 Server (Agent)。
        
    - 執行 `workflow_proxy.py` (Client端代碼) = 你就是 Client。
        
    - `core/__init__.py` 中的 Factory 邏輯在 `mode="remote"` 時，只會回傳 `WorkflowProxy`，並沒有去檢查 `role` 參數。
        
- **只有在 `aws_iot` 模式下**：
    
    - 因為 AWS IoT 的 Agent 和 Client 使用的是**同一套通訊協定 (MQTT)** 和類似的代碼結構。程式必須讀取 `network.role` 來決定它是要「訂閱指令 (Subscribe Command) 並控制硬體」(Agent)，還是要「發布指令 (Publish Command)」(Client)。
        

**結論**：`network.role` 確實是專為 `aws_iot` 模式設計的參數，在其他模式下是無效的（冗餘設定）。

---

### 3. 如果系統模式不是 hybrid, 那 `hybrid_import_yaml` 是否設定什麼都沒關係?

**答案：是的，在非 hybrid 模式下，此設定會被忽略。**

讓我們看 `main_cli.py` 的關鍵邏輯：

Python

```
# main_cli.py (簡化版)
if args.command == "run_routine":
    target_template_path = None
    
    # 1. 優先權最高：命令列參數 --template_path
    if args.template_path:
        target_template_path = args.template_path
        
    # 2. 次要優先權：Hybrid 模式且設定檔有填 hybrid_import_yaml
    elif engine.mode == "hybrid" and args.hybrid_yaml:  # <--- 關鍵判斷
        potential_path = ...
        # ... 載入 YAML ...
        
    # 3. 最後：如果上述都沒中，execute_routine 內部會預設去讀 DB
    engine.execute_routine(..., template_path=target_template_path, ...)
```

- **如果 `mode="production"`**：上述的 `elif` 條件不成立，程式會跳過讀取 YAML，直接進入 `execute_routine`。在 `execute_routine` 內部，因為 `template_path` 為 `None`，它會根據 `routine_name` 去 Query Database。
    
- **例外情況**：如果你在執行 CLI 時手動使用了 `--template_path` 參數，那無論什麼模式都會強制讀取檔案。
    

**結論**：`config/cli_config.yaml` 裡的 `hybrid_import_yaml` 只有在 `system.mode: "hybrid"` 時才會生效。

---

### 4. 這些 command 各是什麼？怎麼控制系統？

這些 Command 是 `main_cli.py` 的進入點，用來告訴系統這次啟動要執行什麼任務。分析如下：

|**Command**|**功能描述**|**是否有效/常用**|
|---|---|---|
|**`run_routine`**|**執行自動化檢測流程**。<br><br>  <br><br>這是系統的核心功能。它會根據 Template (DB 或 YAML) 定義的一連串步驟（移動馬達 -> 打光 -> 拍照 -> 分析 -> 存檔），自動跑完所有測試。|✅ **核心功能**。<br><br>  <br><br>這是最主要的執行指令。|
|**`manual_capture`**|**手動單張拍攝**。<br><br>  <br><br>不走流程，直接控制指定相機 (`--cam`) 拍一張照片並存檔。通常用於測試相機焦距、光線或單純想看即時畫面。|✅ **測試常用**。<br><br>  <br><br>工程師調機時很常用。|
|**`force_sync`**|**強制雲端同步**。<br><br>  <br><br>觸發 `CloudSyncManager` 立即掃描本地 DB 中尚未上傳的圖片 (`synced=0`) 並上傳到 S3/DynamoDB。|✅ **維護常用**。<br><br>  <br><br>當背景同步失敗或網路恢復後手動補傳用。|
|**`analyze_local`**|**(代碼缺失)** 本地圖片重分析。<br><br>  <br><br>理論上是用來讀取硬碟裡既有的圖片，重新跑演算法 (不重拍)。|❌ **無效 (Dead Code)**。<br><br>  <br><br>在您提供的 `main_cli.py` 中，雖然 `add_argument` 裡有這個選項，但 `main()` 函式的 `if/elif` 判斷區塊裡**完全沒有實作**這個指令的邏輯。|
|**`create_template`**|**建立/匯入 Template**。<br><br>  <br><br>根據歷史紀錄 (`--source_watch`) 產生新的 Template。|✅ **管理常用**。<br><br>  <br><br>用於從舊資料生成新標準。|

詳細解釋 analyze_local 的問題：

在提供的 main_cli.py 中：

Python

```
# ... 參數定義有 ...
parser.add_argument("--command", ..., choices=[..., "analyze_local", ...])

# ... 但 main() 函數的執行邏輯只有 ...
if args.command == "run_routine": ...
elif args.command == "manual_capture": ...
elif args.command == "create_template": ...
elif args.command == "sync_templates": ...
elif args.command == "force_sync": ...
# 這裡少了 elif args.command == "analyze_local": 的區塊！
```

所以，如果您在 CLI 設定 `command: "analyze_local"`，程式會執行完 `try` 區塊前的準備工作，然後什麼都不做就結束了。這是一個尚未實作完成或被遺棄的功能。
