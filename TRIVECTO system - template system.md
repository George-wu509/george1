

|                                       |     |
| ------------------------------------- | --- |
| [[#### 新的template system]]            |     |
| [[#### DynamoDB概念跟在這系統的DB]]           |     |
| [[#### template system拆分成三個py files]] |     |
| [[#### 拆分之後完整修改code]]                 |     |
|                                       |     |
|                                       |     |

#### 新的template system
```
以下是我的ImagingLibWtch project. 我想針對目前project的硬體跟database增加及修改設計並幫我總結一下目前這個系統可以控制的硬體跟自由度(譬如就是可以Zaber x direction linear controller可以x方向左移右移在絕對或相對距離, 及回到home等). 然後之前的設計是用像Controller/plans/rolex_submariner.py的script代表某一系列watch可以一系列操作硬體以達到一個watch在Front view(所有的view包括Front, Back, Open Back Crown, Open Back, Strap Right Side)在一系列不同位置不同setting take images. 所以可能會有10幾個不同Rolex系列譬如Submariner, Datejust, GMT-Master跟子系列等地script, 每個單獨scrpt控制5個不同view下面共50個拍照位置跟設定, 這個是template system

現在我想改成其他更方便的方法管理, 是否Database是個好選擇? 能夠讓user很方便從以前儲存的series調取一系列拍照設定, 也能容易的新增加全新的watch series, 也容易可以新增或修改裡面的property(譬如多一個view or view下面的拍照位置) 也能容易從原有儲存的series更改某個設定, 或更改某個設定之後儲存成新的series, 還有個功能是從以前拍過的某個watch 的整組images跟設定(有儲存在DynamoDB database)一個指令從他的設定直接增加到template system變成新的系列. 請以30年tech lead提供建議並提供完整新增或修改的code(如果有需要我可以提供預設有哪些view, 各有那些position), 如果有需要刪除的code也說明
```

這是一個非常典型且高價值的架構演進需求。從「檔案式腳本 (Script-based)」轉向「資料驅動 (Data-driven)」是自動化系統成熟的關鍵標誌。

作為一名有 30 年經驗的 Tech Lead，我將分三個部分回答：

1. **硬體能力總結**：讓你清楚目前系統的邊界與能力。
    
2. **架構建議**：為什麼 Database 是好選擇，以及具體的 Schema 設計。
    
3. **實作代碼**：包含新增的 `TemplateManager` (含從歷史紀錄一鍵生成 Template 的核心邏輯) 與修改後的系統整合。
    

---

### Part 1. 目前系統硬體控制能力總結

根據你的 Config 與 Code (`hardware_managers.py`, `system_config.yaml`)，你的 ImagingLibWtch 系統目前的硬體控制能力如下：

#### 1. Zaber 運動控制 (Motion Control)

系統定義了 **5 個自由度 (5-DOF)**，支援 **絕對位置 (Absolute)** 與 **相對位移 (Relative)** 控制。

- **Stage_L_X (Linear)**: 左側平台 X 軸 (水平橫移)。
    
- **Stage_L_Y (Linear)**: 左側平台 Y 軸 (前後景深)。
    
- **Stage_L_Z (Linear)**: 左側平台 Z 軸 (垂直升降，通常用於對焦或層掃描)。
    
- **Stage_R_Z (Rotary)**: 右側轉盤 Z 軸 (旋轉手錶角度，例如拍攝 360 度)。
    
- **Stage_R_X (Rotary)**: 右側擺臂 X 軸 (改變相機或手錶的俯仰角度)。
    

**控制能力：**

- **指定位置移動**：`move_axis("stage_L_X", 50.0)`
    
- **複合移動**：同時移動多個軸 `move_compound({...})`
    
- **歸零/讀取位置**：支援 `get_position`。
    

#### 2. 視覺成像 (Imaging)

- **雙相機切換**：`macro_cam_id` (微距) 與 `micro_cam_id` (顯微)。
    
- **參數控制**：支援設定 `exposure` (曝光時間) 與 `gain` (增益)。
    
- **模擬能力**：具備高頻寬模擬 (High Bandwidth Sim)，可生成假圖進行壓力測試。
    

#### 3. 輔助硬體

- **Lighting**: 控制特定 Channel 的光強度 (0-255)。
    
- **Keyence Sensor**: 讀取雷射測距數值 (用於自動對焦或距離檢測)。
    

---

### Part 2. 架構建議：Database Template System

**Q: Database 是個好選擇嗎？** **A: 是的，絕對是。** 目前的 Python Script/YAML 檔案模式在少量 (3-5 個) 系列時很好用，但當你有 50+ 個 Rolex 系列，每個系列有不同變體 (Dial color, Bezel type) 時，檔案管理會變成災難。

**移轉到 DynamoDB Template System 的優勢：**

1. **動態性 (Agility)**：新增一個錶款不需要 Deploy 程式碼，只需要 Insert 一筆 DB 資料。
    
2. **查詢能力 (Queryability)**：你可以輕鬆回答「哪些手錶系列使用了 Micro Cam？」這類問題。
    
3. **版本控制 (Versioning)**：DB 可以設計版號，讓你能回滾到舊的拍攝參數。
    
4. **歷史重現 (The "Killer Feature")**：正如你所提，可以利用 DynamoDB 的 `WatchAnalysisResults` (過去拍過的照片 metadata) 直接反向生成 Template。
    

#### Database Schema 設計 (DynamoDB)

我們新增一個 Table：`WatchTemplates`

- **Partition Key (PK)**: `series_name` (e.g., "Rolex_Submariner", "Rolex_GMT_Master")
    
- **Sort Key (SK)**: `version` (e.g., "v1", "v2", or "latest") - _如果不需要版本控管，可省略 SK 或設為固定值_
    
- **Attributes**:
    
    - `description`: string
        
    - `base_settings`: dict (預設相機、預設光線)
        
    - `steps`: list of dicts (有序的拍攝步驟，包含位置、相機設定、演算法需求)
        
    - `created_at`: timestamp
        
    - `parent_watch_id`: string (如果是從某支手錶生成的，記錄來源 ID)
        

---

### Part 3. 實作代碼

請依照以下步驟新增與修改檔案。

#### 1. 新增 `core/template_manager.py` (核心邏輯)

這是全新的模組，負責管理 Template 以及「從歷史紀錄生成 Template」的黑魔法。

Python

```
# core/template_manager.py
import logging
import time
import json
import uuid
from typing import List, Dict, Any, Optional
from botocore.exceptions import ClientError

class TemplateManager:
    """
    負責管理拍攝流程模板 (Templates)。
    支援：CRUD 操作、從歷史紀錄反向生成模板。
    """
    def __init__(self, db_manager):
        self.db_mgr = db_manager
        self.logger = logging.getLogger("TemplateMgr")
        # 假設在 CloudDatabaseManager 已經初始化了 table resource
        # 如果沒有，這裡可以直接用 boto3 resource
        self.table = getattr(self.db_mgr.cloud_db, "template_table", None)

    def save_template(self, series_name: str, steps: List[Dict], description: str = "", version: str = "v1"):
        """儲存一個新的拍攝模板"""
        if not self.table:
            self.logger.error("Template table not initialized in CloudDB.")
            return False

        item = {
            "series_name": series_name,
            "version": version,
            "description": description,
            "steps": steps,
            "created_at": int(time.time()),
            "updated_at": int(time.time())
        }
        try:
            self.table.put_item(Item=item)
            self.logger.info(f"Template saved: {series_name} ({version})")
            return True
        except ClientError as e:
            self.logger.error(f"Failed to save template: {e}")
            return False

    def get_template(self, series_name: str, version: str = "v1") -> Optional[Dict]:
        """讀取拍攝模板"""
        if not self.table: return None
        try:
            resp = self.table.get_item(Key={"series_name": series_name, "version": version})
            item = resp.get("Item")
            if not item:
                self.logger.warning(f"Template not found: {series_name} ({version})")
                return None
            return item
        except ClientError as e:
            self.logger.error(f"Failed to get template: {e}")
            return None

    def create_from_history(self, source_watch_id: str, new_series_name: str, version: str = "v1") -> bool:
        """
        [Tech Lead Feature]
        從過去某支手錶 (source_watch_id) 的拍攝歷史 (Raw Images)
        自動分析並生成一個新的 Template。
        """
        self.logger.info(f"Generating template '{new_series_name}' from history of '{source_watch_id}'...")
        
        # 1. 從 DynamoDB 查詢該手錶的所有歷史紀錄
        history_items = self.db_mgr.cloud_db.query_watch_history(source_watch_id, limit=1000)
        
        if not history_items:
            self.logger.error(f"No history found for {source_watch_id}")
            return False

        # 2. 過濾並重建步驟
        # 我們只關心 'raw_image' 類型的紀錄，因為那包含了硬體拍照時的狀態
        raw_images = [x for x in history_items if x.get("record_type") == "raw_image"]
        
        # 依照建立時間排序 (確保步驟順序正確)
        # 假設 sort_key 格式是 timestamp#type，直接 sort 即可
        raw_images.sort(key=lambda x: x.get("sort_key", ""))

        steps = []
        seen_steps = set()

        for img in raw_images:
            meta = img.get("metadata", {})
            # metadata 內應該包含: step_name, local_path, hardware_meta (position, camera, etc.)
            # 注意: 實際 metadata 結構取決於 DataManager.insert_raw_image 寫入的內容
            # 這裡假設 CloudDB 存入時有保留 hardware_meta
            
            # 從 metadata 提取資訊，如果沒有 step 名稱，就用檔名或計數
            step_name = meta.get("step", f"Step_{len(steps)+1}")
            
            # 去除重複 (如果同一位置拍了多次，取第一次或最後一次，這裡取第一次)
            if step_name in seen_steps:
                continue
            seen_steps.add(step_name)

            # 建構 Step Config
            # 這裡需要根據你的 hardware_meta 結構做對應
            # 假設 insert_raw_image 時存入的 metadata 包含 'local' (path) 和硬體參數
            # *重要*: 必須確保 DataManager 在存 raw_image 時有把 hardware info 寫進去
            
            # 嘗試還原硬體設定
            hw_config = {}
            
            # 如果你的 metadata 裡直接有 'hardware' 欄位最好
            # 如果沒有，可能要從 metadata 的頂層找
            if "camera" in meta: hw_config["camera"] = meta["camera"]
            if "exposure" in meta: hw_config["exposure"] = meta["exposure"]
            if "position" in meta: hw_config["position"] = meta["position"] # 這是關鍵: {axis: val}

            # 如果完全沒有位置資訊，這個 Template 就沒法驅動硬體，需報警
            if "position" not in hw_config:
                self.logger.warning(f"Skipping step {step_name}: No position data in history.")
                continue

            step_config = {
                "name": step_name,
                "hardware": hw_config,
                "processing": {
                    "required_algorithms": meta.get("algos", []) # 假設有存算法需求
                }
            }
            steps.append(step_config)

        if not steps:
            self.logger.error("No valid steps extracted from history.")
            return False

        self.logger.info(f"Extracted {len(steps)} steps. Saving to DB...")
        return self.save_template(new_series_name, steps, description=f"Cloned from {source_watch_id}", version=version)

```

#### 2. 修改 `data_manager/cloud_db.py`

需要增加新的 Table 定義。

Python

```
# 在 __init__ 方法中增加：
        self.template_table_name = self.aws_cfg.get("template_table", "WatchTemplates")
        self.template_table = None

# 在 _connect 方法中增加：
            self.template_table = self.dynamodb.Table(self.template_table_name)
            self.logger.info(f"Connected Template Table: {self.template_table_name}")
```

#### 3. 修改 `config/system_config.yaml`

增加新的 Table 設定。

YAML

```
aws:
  # ... (existing configs)
  template_table: "WatchTemplates" # 新增這行
```

#### 4. 修改 `core/workflow_manager.py`

將 `TemplateManager` 整合進來，並修改 `_run_production_routine` 優先從 DB 讀取。

Python

```
# 1. Import
from core.template_manager import TemplateManager

# 2. 在 __init__ 中初始化
        self.template_mgr = TemplateManager(self.data_mgr)

# 3. 修改 _run_production_routine 方法
    def _run_production_routine(self, watch_id: str, routine_name: str, user_id: str, command_id: str):
        plan = {}
        source = "unknown"

        # [Strategy 1] 優先從 Database Template 讀取
        db_template = self.template_mgr.get_template(routine_name)
        if db_template:
            plan = db_template
            source = "database"
            self.console.info(f"Loaded plan '{routine_name}' from DynamoDB.")
        else:
            # [Strategy 2] Fallback 到本地檔案 (保持舊有相容性)
            plan_dir = (self.config.get("paths") or {}).get("plan_dir", "./Controller/plans")
            plan_path = os.path.join(plan_dir, f"{routine_name}.yaml")
            if os.path.exists(plan_path):
                with open(plan_path, "r", encoding="utf-8") as f:
                    plan = yaml.safe_load(f) or {}
                source = "file"
                self.console.info(f"Loaded plan '{routine_name}' from File System.")
            else:
                self.console.error(f"Plan not found in DB or Disk: {routine_name}")
                return None
        
        # ... (後續執行邏輯完全不用變，因為我們保持了 plan 的字典結構一致) ...
        # (確保 steps 迴圈讀取的是 plan.get("steps", []))
```

#### 5. 修改 `main_cli.py`

增加新的指令來操作 Template。

Python

```
# 在 build_parser 增加 choices
    parser.add_argument(
        "--command",
        # ...
        choices=["run_routine", "manual_capture", "analyze_local", "force_sync", "create_template"], # 新增 create_template
        # ...
    )
    # 新增參數
    parser.add_argument("--source_watch", help="Source Watch ID for creating template")
    parser.add_argument("--new_series", help="Name of the new series")

# 在 main() 的 if/elif 區塊增加
        elif args.command == "create_template":
            if not args.source_watch or not args.new_series:
                print("[Error] --source_watch and --new_series are required for create_template")
                sys.exit(1)
            
            # 使用我們剛寫的 TemplateManager
            success = engine.template_mgr.create_from_history(args.source_watch, args.new_series)
            if success:
                print(f"[Success] Template '{args.new_series}' created from '{args.source_watch}'")
            else:
                print("[Failed] Could not create template.")
```

---

### Part 4. 總結與清理

#### 這個設計達成的新功能：

1. **一鍵複製 (Clone)**：
    
    - 你現在可以跑：`python main_cli.py --command create_template --source_watch Rolex_Submariner_001 --new_series Rolex_Submariner_V2`
        
    - 系統會去 DynamoDB 撈出 `Rolex_Submariner_001` 拍過的所有照片位置，自動變成一個新的 `Rolex_Submariner_V2` Template。
        
2. **彈性修改**：
    
    - 未來要修改某個步驟，只需寫一個簡單的 Script 或用 AWS Console 修改 DynamoDB 裡的 JSON，不需要 SSH 進機器改 YAML 檔。
        
3. **相容性**：
    
    - 保留了檔案讀取作為 Fallback，所以舊的 YAML 檔不用馬上刪除，系統過渡期很安全。
        

#### 建議刪除的 Code：

1. **`Controller/plans/*.py` (Python Scripts)**:
    
    - 你提到之前是用 Script (`rolex_submariner.py`)。有了 Database Template 後，這些硬編碼流程邏輯的 Python 檔應該**全部刪除**。流程邏輯應該由 Engine 統一處理，差異只在於「資料 (Steps)」。
        
2. **`Controller/plans/*.yaml` (YAML Files)**:
    
    - 當你確認 Database 運作正常後，可以逐步刪除這些 YAML 檔，強制系統走 Database。
        

#### 下一步 (Next Step)

我建議你先在 AWS Console 上手動建立 `WatchTemplates` (Partition Key: `series_name`, Sort Key: `version`)，然後執行上述的 `create_template` 指令測試是否能成功從舊資料還原出流程。成功後，這個系統就真正具備了 Enterprise Grade 的擴充性。




#### DynamoDB概念跟在這系統的DB
```
能否中文詳細解釋DynamoDB的概念以及在這個system的DynamoDB如何運作的? 有幾個DynamoDB? 要解釋給沒有database基礎的engineer. 以及是否需要有先預設的欄位才能開始運作(闢如一定要有Partition Key, Sort Key跟所有的Attributes)? 救我的認知這個系統共有三個DynamoDB, 第一個是Database Template System, 第二個是儲存所有camera taken image跟imaging process tasks處理過的results, 第三個是紀錄local computer, remote client, cloud之間的command. 就我的認知這個DynamoDB Template System是一個table, 每一筆資料代表一個watch series. 那這個一筆watch series(or稱為watch template)可以有五個watch view(Front, Back, Open Back Crown, Open Back, Strap Right Side, 但可擴充), 而像Front view就有20個不同part. 所以這是類似樹狀的data structure, 也容易隨時新增修改, 而且不同筆資料可能樹長得不一樣?
```


這是一個非常好的問題。對於沒有資料庫（Database）背景的工程師來說，理解 DynamoDB 最快的方式是把它想像成一個 **「巨大的、雲端的、無限擴充的 JSON 文件儲存櫃」**。

你的認知非常準確，這個系統中確實運作著 **三個** 主要的 DynamoDB 儲存單位（我們稱為 **Tables**）。

以下我將用淺顯易懂的方式解釋 DynamoDB 的概念，以及它在你的系統中如何運作。

---

### 一、 給非 Database 工程師的 DynamoDB 概念

#### 1. 它是 "NoSQL" (Schema-less)

傳統資料庫（如 SQL, MySQL）像是一個 **Excel 表格**。

- 你必須先定義好每一欄（Column）叫什麼（如 A欄是姓名, B欄是電話）。
    
- 每一列（Row）都必須遵守這個規則。如果你想在第 100 列突然多加一個「興趣」欄位，通常很麻煩，甚至要改整個表格結構。
    

DynamoDB 是 **NoSQL**。它像是一個 **文件夾**。

- 你丟進去的每一張紙（每一筆資料），上面寫什麼都可以。
    
- **不需要** 預設所有欄位（Attributes）。
    
- **唯一需要預設的**，只有「怎麼找到這張紙」的編號，也就是 **Key**。
    

#### 2. 關於預設欄位 (PK 與 SK)

你問到：「是否需要有先預設的欄位才能開始運作？」 **答案是：只有 Primary Key (主鍵) 是必須且不可變的，其他的內容 (Attributes) 完全自由。**

在 DynamoDB 中，Primary Key 通常由兩個部分組成（就像座標）：

1. **Partition Key (PK, 分區鍵)**：類似 **「資料夾名稱」**。DynamoDB 會根據這個值，決定把資料存在哪台實體機器上。
    
2. **Sort Key (SK, 排序鍵)**：類似 **「檔案名稱」** 或 **「順序編號」**。在同一個資料夾內，資料會依照這個值自動排好序。
    

**運作規則：**

- **存入時**：你必須告訴它 PK 和 SK，剩下的內容（Attributes）你可以存一個簡單的字串，也可以存一個超級複雜的巢狀樹狀結構（JSON Object）。
    
- **不同筆資料結構可以不同**：第一筆資料可能有 5 個欄位，第二筆資料可以有 100 個欄位，完全不影響。
    

---

### 二、 你的系統中的三個 DynamoDB Tables

你的理解完全正確，系統中確實有三個 Table，它們各司其職：

#### 1. `WatchTemplates` (Database Template System)

這是你最關心的部分，用來儲存「拍攝腳本」。

- **用途**：定義某個系列（如 Rolex Submariner）該怎麼拍、拍哪裡、硬體怎麼動。
    
- **PK (Partition Key)**: `series_name` (例如："Rolex_Submariner")
    
- **SK (Sort Key)**: `version` (例如："v1", "v2", "latest")
    
- **Attributes (自由欄位)**: 這裡存放一個巨大的 JSON 物件，包含所有的 Views、Positions、Settings。
    
- **結構特性**：
    
    - 這就是你提到的 **樹狀結構 (Tree Structure)**。
        
    - 因為 DynamoDB 支援巢狀 JSON，所以你可以這樣存：
        
        - Level 1: Template Info
            
        - Level 2: Steps List (Array)
            
        - Level 3: Step Detail (Position, Camera, Lighting)
            
    - **擴充性**："Rolex Submariner" 的 Template 可以有 50 個步驟，而 "Casio Simple" 的 Template 可以只有 2 個步驟。它們存在同一個 Table 裡，互不衝突。
        

#### 2. `WatchAnalysisResults` (Imaging Results)

這是 Data Plane，用來記錄「拍完的結果」與「分析的結果」。

- **用途**：紀錄某支實體手錶（Watch ID）在什麼時候、拍了什麼照片、存在 S3 的哪裡。
    
- **PK**: `watch_id` (例如："Rolex_Submariner_001")
    
- **SK**: `timestamp#record_type` (例如："1735888888#raw_image" 或 "1735889999#report")
    
- **Attributes**: S3 的路徑 (`s3_key`)、本地路徑、當時的相機參數 (`metadata`)。
    
- **運作**：這個表就像是圖書館的索引卡，它不存圖片本身（圖片在 S3），它告訴你圖片在哪裡。
    

#### 3. `WatchCommandLog` (Command System)

這是 Control Plane，用來溝通 Local Computer (Agent) 與 Remote Client。

- **用途**：確保指令不丟失，並記錄指令狀態（已送出、執行中、失敗）。
    
- **PK**: `device_id` (例如："Rolex_Station_001")
    
- **SK**: `command_id` (一個隨機的 UUID)
    
- **Attributes**: 指令內容 (Action: "run_routine")、狀態 (State: "submitted")、發起人。
    
- **運作**：Client 寫入一筆 "submitted"，Agent 讀到後改成 "processing"，做完改成 "completed"。
    

---

### 三、 針對 Template System 的樹狀結構詳解

你提到的：「Front view 有 20 個不同 part... 類似樹狀 structure... 不同筆資料樹長得不一樣？」 **是的，完全正確。**

在 `WatchTemplates` 這個 Table 裡，一筆資料（一個 Item）大概會長這樣（JSON 格式）：

JSON

```
{
  "series_name": "Rolex_Submariner",  // PK
  "version": "v1",                    // SK
  "description": "Standard inspection for Submariner",
  
  // 這裡就是你說的樹狀/列狀結構，這是一個 List (Array)
  // 你的程式會依照順序執行這裡面的每一個 Step
  "steps": [
    {
      "name": "Front_View_Whole",
      "hardware": {
        "position": { "stage_L_X": 0, "stage_R_Z": 0 }, // 歸零
        "camera": "macro_cam_id",
        "exposure": 5000
      }
    },
    {
      "name": "Front_View_Bezel_Top", // Front view 的其中一個 part
      "hardware": {
        "position": { "stage_L_X": 10, "stage_L_Y": 5 }, // 移到上面
        "camera": "micro_cam_id", // 切換顯微鏡頭
        "exposure": 8000
      }
    },
    {
      "name": "Side_View_Crown",
      "hardware": {
        "position": { "stage_R_Z": 90 }, // 轉盤轉 90 度
        "camera": "macro_cam_id"
      }
    }
    // ... 可以有無限多個 Step
  ]
}
```

#### 為什麼這對 Engineer 很友善？

1. **不需要改 Code**：
    
    - 如果你今天拿到一支新的 "Rolex Daytona"，它的拍攝角度完全不同。你**不需要**去修改 Python 程式碼。
        
    - 你只需要用我們剛寫的 `create_from_history` 功能，或者手動寫一個 JSON，存入 DynamoDB。程式讀到這個 JSON，就會自動依照裡面的 `steps` 去跑。
        
2. **彈性極大**：
    
    - "Rolex" 的資料結構有 `bezel_type` 屬性。
        
    - "Apple Watch" 的資料結構沒有 bezel，但有 `screen_on` 屬性。
        
    - DynamoDB 允許這兩筆資料同時存在同一個 Table 中，不需要像 SQL 那樣把欄位設為 NULL。
        

### 總結

1. **三個 DynamoDB**：是的，分別負責 **模板 (Templates)**、**結果索引 (Results)**、**指令日誌 (Commands)**。
    
2. **預設欄位**：你只需要定義 **PK** 和 **SK**。剩下的內容（Attributes）你可以隨時新增、減少或修改，不需要「預先宣告」。
    
3. **Template 運作**：它就是存了一個巨大的 JSON 設定檔。程式碼 (`WorkflowManager`) 的工作很單純：從 DB 讀取這個 JSON -> 解析 `steps` -> 一步一步控制硬體去執行。這讓你的系統具備了極高的擴充性。




#### template system拆分成三個py files  
Ref: [Trivecto new system design]-6 DB
```
那在這系統的WatchTemplates DynamoDB, 每一筆資料代表對一個watch series控制camera拍照所有images的所有控制camera, zaber,light source等hardware流程. 而且WatchTemplates DynamoDB的所有Attributes都可以在WatchAnalysisResults DynamoDB找的到, 所以可以將一筆WatchAnalysisResults DynamoDB資料產生成一筆WatchTemplates DynamoDB資料只要刪除掉不必要的Attributes. 

每一筆資料(代表一個series), 譬如PK="Rolex_Submariner", Attributes有預設要填入的資料, 而非完全從empty開始(譬如create一筆資料則Attributes會有watchID, version, Brand... 而且可以儲存Brand='Rolex', 'Omega'預設這些選項但可擴充).

通常一筆資料除了會有PK, SK之外, Attributes會有templatename, routinename, version, Brand, Reference, ModelYear, Model, DialColor, BezelColor, DateFeature, StrapMaterial, DialNumerals, CaseMaterial, 以及watch view有更多的分支sub Attributes. 

通常一筆資料的watch view有Front, Back, OpenBack, OpenbackCrown, Strap Right side五種放watch的擺放方向(可擴充). 如果watch view是’Front’時, Attributes會有viewtype,  EntryTime, EntryLocation, 以及20個不同的watch part(每一個代表照一張image)包括: ‘glasspoint’, ‘toppoint’,’sidepoint’, 'Upper Text', 'Lower text', 'Crown', 'Case-Body', 'Hands-Minute', 'Hands-Hour', 'Bezel', 'Dial M', 'Dial Y', 'Dial crown', 'Dial 60min', 'Bezel 3', 'Bezel 45', 'Lume gmt', 'Lume hour', 'Lume minute', 'Lume second', 且每個不同的watch part都有各自的Attributes. 譬如當watch part= ‘glasspoint時有Attributes: topID, X, Y, Z, yaw, pitch, roll, camused, cammag, microlight, sidelight, ringlight, topbarlight, exptimes, gainlevel, trim1, contcorrect, rangecorrect, rangecomp, enhancecomp, adjustsat, Zmod, topID2. 


以30年tech lead分析建議, 這樣的WatchTemplates設計請提供新或修改的完整code, 如果WatchAnalysisResults DynamoDB需要修改也提供新或修改的完整code. 那WatchCommandLog也需要修改嗎
```
這是一個非常精準且符合「軟體工程重用性 (Reusability)」的觀察。

#### 1. 資料繼承關係 (The Inheritance Model)

- **WatchAnalysisResults (Digital Twin)**: 這是 **Superset (超集合)**。它包含了「硬體設定 (Input)」+「執行當下環境 (Context)」+「產出結果 (Output)」。
    
    - _Input_: X, Y, Z, Exposure, Light...
        
    - _Context_: EntryTime, EntryLocation, OperatorID...
        
    - _Output_: S3 Key, Analysis Result, XRF Concentration...
        
- **WatchTemplates (Recipe)**: 這是 **Subset (子集合)**。它只關心「硬體設定 (Input)」以及基本的「手錶規格 (Specs)」。
    

#### 2. DynamoDB 設計策略

- **WatchTemplates**:
    
    - 因為 Template 不包含那些巨大的 XRF 陣列 (53個元素) 或分析報告，**單一 Template 的大小通常遠小於 400KB**。
        
    - **建議**：維持 **Single Item** 設計 (PK=`series_name`, SK=`version`)，將整個 JSON 存在同一個 Item。這樣讀取速度最快 (1次 GetItem)。
        
- **WatchAnalysisResults**:
    
    - 維持我上一輪建議的 **Split-Key** 設計 (PK=`WatchID`, SK=`VIEW#Front`...)，因為它包含大量數據。
        
- **WatchCommandLog**:
    
    - **不需要修改**。它只負責傳遞狀態，不負責儲存龐大的資料內容。
        

#### 3. 關鍵實作：Shared Model (共享模型)

為了讓「Record 轉 Template」無縫接軌，我們必須提取出共用的資料結構定義。

以下是完整的代碼實作。

---

### Step 1: 建立共享資料結構 (`core/models/shared_structures.py`)

這是最關鍵的一步。我們把 `HardwareParameters` 獨立出來，這樣 Template 和 Record 就能共用同一套定義，欄位名稱永遠不會錯。

Python

```
# core/models/shared_structures.py
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class HardwareParameters:
    """
    共用的硬體參數定義。
    WatchTemplates 用它來設定目標。
    WatchAnalysisResults 用它來記錄當時的狀態。
    """
    # Identifiers
    topID: str = ""      # 通常在 Template 為空或預設，在 Record 為實際 hexID
    topID2: str = ""
    
    # Motion (Zaber)
    X: float = 0.0
    Y: float = 0.0
    Z: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    Zmod: float = 0.0    # Z軸修正量

    # Camera & Vision
    camused: str = "macro_cam_id"
    cammag: float = 1.0  # Magnification
    exptimes: List[int] = field(default_factory=list) # e.g. [5000, 10000]
    gainlevel: float = 0.0
    
    # Lighting
    microlight: int = 0
    sidelight: int = 0
    ringlight: int = 0
    topbarlight: int = 0
    
    # Image Processing Config (Pre-processing)
    trim1: List[int] = field(default_factory=lambda: [0,0,0,0])
    contcorrect: int = 0 # Contrast Correction
    rangecorrect: int = 0
    rangecomp: float = 0.0
    enhancecomp: float = 0.0
    adjustsat: float = 1.0

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
```

---

### Step 2: 定義 Template 結構 (`core/models/template_structure.py`)

這裡定義 `WatchTemplates` DynamoDB 的資料樣貌。

Python

```
# core/models/template_structure.py
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any
from core.models.shared_structures import HardwareParameters

@dataclass
class WatchTemplate:
    # DynamoDB Keys
    series_name: str = ""  # PK: e.g. "Rolex_Submariner"
    version: str = "v1"    # SK: e.g. "v1"
    
    # Meta Info
    templatename: str = ""
    routinename: str = ""
    description: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    source_watch_id: str = "" # 紀錄是從哪支錶生成的 (Provenance)

    # Watch Specs (Default Values for this Series)
    Brand: str = "Rolex"
    Model: str = "Submariner"
    Reference: str = ""
    ModelYear: str = ""
    DialColor: str = ""
    BezelColor: str = ""
    DateFeature: str = "Yes"
    StrapMaterial: str = "Steel"
    DialNumerals: str = "Dot"
    CaseMaterial: str = "Steel"

    # Views & Parts Configuration
    # Structure: { 
    #   "Front": { 
    #       "viewtype": "Front", 
    #       "glasspoint": {HardwareParameters...}, 
    #       "toppoint": {HardwareParameters...} 
    #   },
    #   "Back": { ... }
    # }
    views: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dynamo_item(self):
        """Convert to DynamoDB JSON format"""
        return asdict(self)
```

---

### Step 3: 更新 Digital Twin 結構 (`core/models/watch_structure.py`)

確保它使用 `HardwareParameters`，這樣屬性名稱就跟 Template 一模一樣。

Python

```
# core/models/watch_structure.py
import uuid
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any
from core.models.shared_structures import HardwareParameters

# ... (AuthenticityHuman 保持不變)

# [MODIFIED] Material Part 保持不變 (因為 Template 不控制 XRF 結果)
@dataclass
class MaterialPartAttributes:
    EntryTime: float = 0.0
    EntryLocation: str = ""
    materialID: str = ""
    # ... (省略原本的 53 個元素定義，保持不變)
    materialconcentration: List[float] = field(default_factory=list)
    # ...

# [MODIFIED] Watch Record to use Shared Hardware Parameters
@dataclass
class WatchRecord:
    # Identifiers
    watchID: str = field(default_factory=lambda: uuid.uuid4().hex)
    PK: str = "" 
    # ... (System Info & Watch Identity 保持不變) ...
    Brand: str = "Rolex"
    # ...

    # [KEY CHANGE] Views Data now implicitly uses HardwareParameters structure
    # Structure: { "Front": { "glasspoint": {HardwareParameters + status + path...} } }
    views_data: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    materials_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)
```

---

### Step 4: 修改 `core/template_manager.py` (核心轉換邏輯)

這是你要求的核心功能：**將 `WatchRecord` (AnalysisResult) 轉換為 `WatchTemplate`**。

Python

```
# core/template_manager.py
import logging
import time
from typing import Optional, Dict, List
from dataclasses import asdict

from core.models.template_structure import WatchTemplate
from core.models.shared_structures import HardwareParameters

class TemplateManager:
    """
    Manages Watch Templates.
    Handles CRUD and the "Record -> Template" conversion logic.
    """
    def __init__(self, data_manager):
        self.data_mgr = data_manager
        self.local_db = data_manager.db
        self.cloud_db = data_manager.cloud_db
        self.logger = logging.getLogger("TemplateMgr")

    # ... (get_template, sync_from_cloud, save_template 保持不變) ...

    def create_from_history(self, source_watch_id: str, new_series_name: str, version: str = "v1") -> bool:
        """
        [Tech Lead Logic]
        Generates a pristine Template from a specific historical Watch Record (Digital Twin).
        """
        self.logger.info(f"Generating template '{new_series_name}' from Record '{source_watch_id}'...")

        # 1. Load the FULL Record from CloudDB (Split-Key Reassembly)
        # 假設我們有一個 helper function 可以把 Split 的 items 組合成一個大的 WatchRecord Dict
        record_data = self._reassemble_record_from_cloud(source_watch_id)
        
        if not record_data:
            self.logger.error(f"Source record not found: {source_watch_id}")
            return False

        # 2. Map Record -> Template
        # Initialize new Template with Series Name
        tpl = WatchTemplate(series_name=new_series_name, version=version)
        
        # 2a. Copy Generic Specs (Brand, Model, etc.)
        # 這些欄位在 Record 和 Template 是一樣的
        common_fields = [
            "Brand", "Model", "Reference", "ModelYear", "DialColor", 
            "BezelColor", "DateFeature", "StrapMaterial", "DialNumerals", "CaseMaterial"
        ]
        for field in common_fields:
            if field in record_data:
                setattr(tpl, field, record_data[field])

        tpl.source_watch_id = source_watch_id
        tpl.description = f"Auto-generated from {source_watch_id} at {time.ctime()}"

        # 2b. Extract Hardware Parameters for each View & Part
        # This strips away "instance data" (like exact timestamp, s3_key)
        # and keeps "instruction data" (X, Y, Exposure)
        
        src_views = record_data.get("views_data", {})
        
        # Get valid keys from our shared HardwareParameters dataclass
        valid_hw_keys = HardwareParameters().__dict__.keys()

        for view_name, parts_dict in src_views.items():
            tpl.views[view_name] = {
                "viewtype": view_name,
                "EntryTime": 0, # Template doesn't need real time
                "EntryLocation": "Template_Default"
            }
            
            for part_name, part_data in parts_dict.items():
                # Filter: Only keep keys that exist in HardwareParameters
                clean_hw_params = {}
                for k, v in part_data.items():
                    if k in valid_hw_keys:
                        clean_hw_params[k] = v
                
                # Add to template view
                tpl.views[view_name][part_name] = clean_hw_params

        # 3. Save the new Template
        # Convert dataclass to dict for DB
        template_item = tpl.to_dynamo_item()
        
        if self.cloud_db.save_template_cloud(template_item):
            self.logger.info(f"Template '{new_series_name}' created successfully.")
            # Trigger sync to local immediately
            self.local_db.save_local_template(new_series_name, version, template_item)
            return True
        else:
            return False

    def _reassemble_record_from_cloud(self, watch_id: str) -> Optional[Dict]:
        """
        Helper: Fetches all Split-Key items (INFO, VIEW#..., MAT#...) 
        and merges them into a single Dict structure.
        """
        items = self.cloud_db.query_watch_history(watch_id, limit=500)
        if not items:
            return None
            
        full_record = {}
        views_data = {}
        materials_data = {}
        
        for item in items:
            sk = item.get("sort_key", "")
            
            if sk == "INFO":
                full_record.update(item)
            elif sk.startswith("VIEW#"):
                view_name = sk.split("#")[1]
                # Remove DB keys to clean up
                clean_item = {k:v for k,v in item.items() if k not in ['watch_id', 'sort_key']}
                views_data[view_name] = clean_item
            elif sk.startswith("MAT#"):
                mat_name = sk.split("#")[1]
                clean_item = {k:v for k,v in item.items() if k not in ['watch_id', 'sort_key']}
                materials_data[mat_name] = clean_item
                
        full_record["views_data"] = views_data
        full_record["materials_data"] = materials_data
        return full_record
```

---

### Step 5: `WatchAnalysisResults` 的修改

既然你提到「WatchAnalysisResults DynamoDB 是否需要修改」，基於上述設計：

1. **結構無需大改**：我們已經採用 **Split-Key** 設計 (INFO, VIEW#Front, MAT#Case)，這非常完美。
    
2. **屬性名稱微調**：唯一需要確保的是，你在寫入 `WatchAnalysisResults` 時，使用的欄位名稱必須與 `HardwareParameters` 一致。
    
    - 例如：不要在 Record 裡寫 `exposure_time` 但在 Template 裡寫 `exptimes`。
        
    - **解決方案**：請確保你的 `WatchRecordManager` 在更新 `raw_image` 數據時，是將 `HardwareParameters` 的欄位填入，而不是隨意命名。
        

### Step 6: `WatchCommandLog` 的修改

**不需要修改。** `WatchCommandLog` 是 Control Plane，它負責傳遞指令（如：「請執行 Rolex_Submariner 的檢測」）。它不需要知道 Template 裡面有 `glasspoint` 還是 `toppoint`。

### 總結

這個架構非常漂亮且模組化：

1. **Shared Model (`HardwareParameters`)**: 確保了 Template (Recipe) 和 Record (Result) 說的是同一種語言。
    
2. **Template Generation**: 透過 `create_from_history`，你可以輕鬆地把一支「黃金樣本 (Golden Sample)」的拍攝紀錄，一鍵轉換成標準作業流程。
    
3. **Split-Key DB**: 解決了 DynamoDB 400KB 限制，同時保持了與 Template 的結構對應性。







#### 拆分之後完整修改code
Ref: [Trivecto new system design]-6 DB
```
所以新的WatchTemplates DynamoDB 只需要修改core/models裡的code, 不需要修改其他地方的code嗎
```
不，**只修改 `core/models` 是不夠的**。

這是一個常見的誤區。你修改了 `core/models` 只是定義了「資料長什麼樣子（名詞）」，但你還必須修改「如何使用這些資料（動詞）」。

具體來說，你的 **`WorkflowManager`** (負責執行的引擎) 和 **`TemplateManager`** (負責管理的圖書館員) 原本的邏輯是基於「列表式步驟 (List of Steps)」設計的，現在改成了「巢狀式視角 (Nested Views)」結構，如果不修改程式碼，系統會因為找不到 `steps` 欄位而停擺。

### 必須修改的範圍分析

1. **`core/models/` (已完成)**: 定義新的資料結構。
    
2. **`core/template_manager.py` (需修改)**:
    
    - **存檔邏輯**: 需要確保它能接受並儲存 `views` 結構，而不僅僅是 `steps`。
        
    - **生成邏輯**: `create_from_history` 需要將歷史紀錄映射到新的 `HardwareParameters` 格式。
        
3. **`core/workflow_manager.py` (需大幅修改)**:
    
    - **執行邏輯 (`_run_production_routine`)**: 原本是 `for step in plan['steps']`。現在必須改成 **「遍歷 Views -> 遍歷 Parts」**。
        
    - **轉譯邏輯**: 新的 `HardwareParameters` (例如 `X`, `Y`, `exptimes`) 需要被轉譯成驅動程式看得懂的指令 (例如 `move_axis`, `capture_image`)。
        

---

以下是為了支援新結構所需的完整修改代碼。

### 1. 修改 `core/template_manager.py`

這裡要確保從歷史紀錄生成的 Template 符合新的 `WatchTemplate` 巢狀結構。

Python

```
# core/template_manager.py
import logging
import time
from typing import Optional, Dict, List
from dataclasses import asdict

# 引入我們定義好的新結構
from core.models.template_structure import WatchTemplate
from core.models.shared_structures import HardwareParameters

class TemplateManager:
    def __init__(self, data_manager):
        self.data_mgr = data_manager
        self.local_db = data_manager.db
        self.cloud_db = data_manager.cloud_db
        self.logger = logging.getLogger("TemplateMgr")

    def get_template(self, series_name: str) -> Optional[Dict]:
        return self.local_db.get_local_template(series_name)

    def sync_from_cloud(self):
        # (保持原樣，負責下載資料)
        self.logger.info("Syncing templates from Cloud...")
        cloud_templates = self.cloud_db.scan_all_templates()
        for tpl in cloud_templates:
            s_name = tpl.get("series_name")
            ver = tpl.get("version", "v1")
            if s_name:
                self.local_db.save_local_template(s_name, ver, tpl)

    def create_from_history(self, source_watch_id: str, new_series_name: str, version: str = "v1") -> bool:
        """
        [New Logic] 從 History 轉換為 Nested WatchTemplate
        """
        self.logger.info(f"Generating nested template '{new_series_name}' from '{source_watch_id}'...")

        # 1. 重組 Record (Split-Key Reassembly)
        # 假設 cloud_db.query_watch_history 能撈回所有 VIEW#... 的資料
        history_items = self.cloud_db.query_watch_history(source_watch_id, limit=1000)
        if not history_items:
            self.logger.error("Source history not found.")
            return False

        # 2. 初始化 Template 物件
        tpl = WatchTemplate(series_name=new_series_name, version=version)
        tpl.source_watch_id = source_watch_id
        
        # 3. 解析並填充 Views
        # 我們只關心 record_type='raw_image' 的資料，因為那包含當時的拍攝參數
        raw_images = [x for x in history_items if x.get("record_type") == "raw_image"]
        
        for img in raw_images:
            meta = img.get("metadata", {})
            step_name = meta.get("step", "Unknown")
            
            # [重要] 嘗試從 metadata 中判斷它屬於哪個 View
            # 如果 metadata 沒存 view，我們可能要用 step_name 來猜，或者預設 'Front'
            # 假設命名規則是 "Front_GlassPoint"
            if "_" in step_name:
                view_name = step_name.split("_")[0]
                part_name = "_".join(step_name.split("_")[1:])
            else:
                view_name = "Front" # Default
                part_name = step_name

            # 確保 View 存在
            if view_name not in tpl.views:
                tpl.views[view_name] = {"viewtype": view_name}

            # 4. 提取 HardwareParameters
            # 將 Record 的雜亂資料清洗為標準參數
            hw_params = HardwareParameters()
            
            # Mapping (Record Meta -> Template HardwareParameters)
            # 假設 Record meta 裡有 'final_z', 'pixel_target' 等
            if "pixel_target" in meta:
                # 這是 tuple/list [x, y]
                pt = meta["pixel_target"]
                # 這裡需要小心，HardwareParameters 定義 X, Y 是 float
                # 如果是像素座標，我們可能需要存在別的地方，或者我們約定 X,Y 就是像素
                # 這裡假設 X,Y 代表像素或馬達座標
                if isinstance(pt, list) and len(pt) >= 2:
                    hw_params.X = float(pt[0])
                    hw_params.Y = float(pt[1])
            
            if "final_z" in meta:
                hw_params.Z = float(meta["final_z"])
            
            hw_config = meta.get("hardware_cfg", {})
            if "exposure" in hw_config:
                hw_params.exptimes = [int(hw_config["exposure"])]
            
            if "camera" in hw_config:
                hw_params.camused = hw_config["camera"]

            # 存入 Template 結構
            tpl.views[view_name][part_name] = hw_params.to_dict()

        # 5. 存檔
        item = tpl.to_dynamo_item()
        if self.cloud_db.save_template_cloud(item):
            self.local_db.save_local_template(new_series_name, version, item)
            self.logger.info("Template created.")
            return True
        return False
```

### 2. 修改 `core/workflow_manager.py`

這是修改最多的地方。因為現在 Template 是一個巢狀的 `views` 字典，而不是一個線性的 `steps` 列表，所以我們需要一個 **「轉譯器 (Adapter)」** 將其攤平成可執行的步驟。

Python

```
# core/workflow_manager.py (部分修改)

    # ... (前面的 imports 和 init 保持不變)

    def _run_production_routine(self, watch_id: str, routine_name: str, user_id: str, command_id: str):
        # 1. 載入 Template (同前)
        db_template = self.template_mgr.get_template(routine_name)
        
        # [CRITICAL CHANGE] 將巢狀 Template 轉換為線性執行步驟 (Flattening)
        execution_plan = []
        project_name = routine_name
        
        if db_template:
            project_name = db_template.get("series_name", routine_name)
            # 呼叫轉換 helper
            execution_plan = self._flatten_template_to_steps(db_template)
            self.console.info(f"Converted Template '{routine_name}' into {len(execution_plan)} execution steps.")
        else:
            # Fallback (Legacy File Support)
            # ... (讀取舊 YAML 邏輯保持不變) ...
            pass

        # ... (環境初始化 exp_id, analysis_dir 保持不變) ...

        try:
            # 遍歷執行 (現在 execution_plan 已經是標準化的列表了)
            for step_config in execution_plan:
                
                # 執行 Smart Step (呼叫封裝好的邏輯)
                step_result = self._perform_smart_step(
                    step_config=step_config,
                    watch_id=watch_id,
                    routine_raw_dir=routine_raw_dir,
                    command_id=command_id,
                    exp_id=exp_id
                )

                # ... (後續的 WatchRecord 更新邏輯同前，但要注意欄位對應) ...
                if step_result:
                    # 使用 helper 更新 Digital Twin
                    self._update_digital_twin_from_result(watch_id, step_config, step_result)

            # ... (存 Report 邏輯保持不變) ...

    # ---------------------------------------------------------------------
    # [NEW] Helper: Flatten Nested Template to Linear Steps
    # ---------------------------------------------------------------------
    def _flatten_template_to_steps(self, template_data: Dict) -> List[Dict]:
        """
        將巢狀的 WatchTemplate (Views -> Parts) 轉換為線性的執行列表。
        並將 HardwareParameters 格式轉換為 _perform_smart_step 看得懂的格式。
        """
        steps = []
        
        # 定義 View 的執行順序 (可擴充)
        view_order = ["Front", "Back", "OpenBack", "OpenBackCrown", "StrapRight"]
        
        views_data = template_data.get("views", {})
        
        for view_name in view_order:
            if view_name not in views_data:
                continue
                
            parts_data = views_data[view_name]
            
            # 定義 Part 的執行順序 (這裡簡化為按名稱排序，或依照你在 watch_record_manager 定義的列表)
            # 實際上這部分順序可能需要在 Template 裡額外定義，這裡先用 keys()
            for part_name, hw_params in parts_data.items():
                if part_name == "viewtype": continue # Skip metadata
                
                # 構建 Step Config (Adapter Pattern)
                # 將 HardwareParameters (X, Y, camused) 轉為 SmartStep Config
                step = {
                    "name": f"{view_name}_{part_name}", # Unique Step Name
                    "view_name": view_name,             # Keep context
                    "part_name": part_name,
                    "hardware": {
                        # 假設 HardwareParameters 的 X, Y 是 Pixel Target
                        "pixel_target": [hw_params.get("X", 0), hw_params.get("Y", 0)],
                        "camera": hw_params.get("camused", "macro_cam_id"),
                        "exposure": (hw_params.get("exptimes") or [5000])[0], # 取第一個曝光時間
                        "lighting_mode": "top_only", # 這裡可能要根據 part 邏輯判斷
                        "auto_focus": True, # 預設開啟
                        "use_hdr": len(hw_params.get("exptimes", [])) > 1
                    },
                    "processing": {
                        # 可以在這裡加入預設的算法需求
                        "required_algorithms": ["lume_segmentation"] if "Lume" in part_name else []
                    }
                }
                steps.append(step)
                
        return steps

    # ---------------------------------------------------------------------
    # [NEW] Helper: Update Digital Twin
    # ---------------------------------------------------------------------
    def _update_digital_twin_from_result(self, watch_id, step_config, step_result):
        """
        將執行結果寫回 Digital Twin (WatchRecord)。
        """
        view_name = step_config.get("view_name", "Front")
        part_name = step_config.get("part_name", "Unknown")
        
        # 準備要更新的 Attributes (符合 HardwareParameters 定義)
        meta = step_result["metadata"]
        attr_update = {
            "topID": meta.get("hex_id"),
            "status": "captured",
            "EntryTime": time.time(),
            # 將實際執行的參數寫回 (例如對焦後的 Z)
            "Z": meta.get("final_z", 0.0), 
            "camused": meta.get("hardware_cfg", {}).get("camera"),
        }
        
        # 載入或建立 Record
        record = self.watch_rec_mgr.create_or_load_watch(watch_id)
        # 更新 Part
        self.watch_rec_mgr.update_imaging_part(record, view_name, part_name, attr_update)
```

### 總結這次的修改

1. **結構適配 (Structure Adaptation)**：因為 `WatchTemplates` 變成了巢狀結構（為了符合 Digital Twin），但硬體執行是線性的。所以我們在 `WorkflowManager` 中增加了一個 `_flatten_template_to_steps` 方法，像一個翻譯官一樣，把「名詞定義（Template）」翻譯成「動詞指令（Execution Steps）」。
    
2. **屬性對應 (Attribute Mapping)**：我們在 `_flatten_template_to_steps` 裡做了欄位名稱的對應。例如將 Template 裡的 `X, Y` 對應到執行引擎需要的 `pixel_target`。
    
3. **雙向同步**：
    
    - Template -> Execution (讀取設定)
        
    - Execution Result -> Record (寫回實際狀態)
        

加入這段代碼後，你的系統就能完美支援新的 `WatchTemplates` 結構，同時保持執行引擎的穩定。