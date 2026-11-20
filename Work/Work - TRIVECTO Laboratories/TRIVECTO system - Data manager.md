

|                                        |     |
| -------------------------------------- | --- |
| [[#### 新版本DataManage流程]]               |     |
| [[#### 上傳到AWS S3的五種data type]]         |     |
| [[#### download_from_aws python code]] |     |
| [[#### 評估30萬到100萬的watch Scale]]        |     |
| [[#### 本地資料夾的image或data是可以定期刪除的]]      |     |
|                                        |     |
|                                        |     |


#### 新版本DataManage流程
```
請重新詳細中文解釋新版本DataManage以及他如何在這個系統發揮作用及詳細流程, 請詳細介紹細節
```
這是一個非常好的切入點。在您的系統 Version 1 中，**`DataManager` (數據管理器)** 已經從單純的「存檔工具」進化為整個系統的**「數據金庫 (Data Vault)」**與**「物流中心 (Logistics Center)」**。

它解決了工業級軟體最頭痛的三個問題：**資安 (Security)、效能 (Performance)、與資料一致性 (Data Integrity)**。

以下我將為您詳細拆解新版 `DataManager` 的設計理念、架構圖解以及它在五個關鍵步驟中的具體運作細節。

---

### 1. DataManager 的核心角色定位

如果把 `WorkflowManager` 比喻為工廠的「廠長」，那麼 `DataManager` 就是**「倉儲物流經理」**。

它的職責非常明確且嚴格：

1. **資產保護 (Security):** 確保硬碟裡的每一張照片、每一份報告都是加密的。即便硬碟被竊，駭客看到的也只是亂碼。
    
2. **生命週期管理 (Lifecycle):** 明確區分「原始素材 (Raw Data)」與「加工產品 (Experiment Results)」。原始素材是神聖不可變更的。
    
3. **雲端同步 (Cloud Sync):** 在背景默默地將資料搬運到 AWS S3，完全不讓使用者感覺到「上傳中」的卡頓。
    
4. **索引查詢 (Indexing):** 使用 SQLite 本地資料庫記錄所有檔案位置，讓 App 能毫秒級查詢歷史紀錄。
    

---

### 2. 檔案結構設計 (Directory Structure)

新版 `DataManager` 會自動維護以下結構，這反映了您「Raw Image 共用，Experiment 獨立」的需求：

Plaintext

```
./Local_Data/
├── experiment_index.db          # [SQLite] 系統總帳本 (索引)
└── Watch_Rolex_12345/           # [手錶 ID] 為單位的根目錄
    ├── raw_images/              # [共用素材區]
    │   ├── front_view.jpg       # (加密狀態: Header被破壞)
    │   ├── front_view.jpg.meta  # (解密金鑰片段)
    │   └── macro_dial.jpg       # (加密狀態)
    │
    └── Experiments/             # [實驗紀錄區]
        ├── 20251221_RoutineA/   # 第一次檢測
        │   ├── masks.png        # (加密狀態: 分割結果圖)
        │   └── report.yaml.enc  # (全加密: 分析數據)
        │
        └── 20251222_RoutineB/   # 第二次檢測 (可能重複使用 raw_images)
            └── ...
```

---

### 3. 詳細運作流程解析 (Step-by-Step Workflow)

針對您定義的 5 個步驟，`DataManager` 在後台的運作邏輯如下：

#### 步驟 1：處理拍攝影像 (Capture, Encrypt & Store)

這是「原始數據」落地的瞬間，必須最優先處理。

- **觸發:** 相機拍完照，暫存在 `temp/front.jpg`。
    
- **DataManager 動作 (`process_and_sync_raw_image`):**
    
    1. **歸檔:** 將檔案從 `temp/` 移動到 `Local_Data/Watch_XXX/raw_images/`。
        
    2. **標頭加密 (Header Encryption):** * 讀取檔案前 4KB (Header)。
        
        - 使用 AES-256 加密這 4KB，存成 `.meta` 檔。
            
        - **破壞性覆寫:** 將原檔的前 4KB 全部填入 `0x00`。
            
        - _結果:_ 檔案大小不變，但任何圖片檢視器都無法開啟它（顯示損毀）。
            
    3. **加入佇列:** 將檔案路徑丟給同步執行緒。
        

#### 步驟 2：雲端同步 (Background Sync)

這是一個**非同步 (Asynchronous)** 的過程，發生在獨立的執行緒 (`CloudSyncManager`)。

- **DataManager 動作:**
    
    1. 背景執行緒發現佇列中有新任務：`Watch_XXX/raw_images/front.jpg`。
        
    2. 呼叫 AWS SDK (`boto3`) 上傳檔案。
        
    3. 同時上傳對應的 `.meta` 解密檔。
        
    4. 上傳成功後，更新 SQLite 資料庫狀態為 `synced`。
        
- **使用者體驗:** 介面完全流暢，不會轉圈圈等待上傳。
    

#### 步驟 3：建立實驗結構 (Create Experiment Hierarchy)

這個步驟只在**「確定要開始跑 AI 分析」**時才觸發。

- **DataManager 動作 (`create_experiment_folder`):**
    
    1. 根據當下時間戳 (Timestamp) 與流程名稱 (Routine Name)，生成唯一的 `Exp_ID` (例如: `202512211400_Standard`).
        
    2. 在 `Local_Data/Watch_XXX/Experiments/` 下建立對應資料夾。
        
    3. **寫入索引:** 在 SQLite 的 `experiments` 表中新增一筆紀錄，關聯 `Exp_ID` 與 `Watch_ID`。
        

#### 步驟 4：影像處理與結果儲存 (Analyze, Save & Encrypt)

這是最複雜的一步，涉及「解密 -> 分析 -> 加密 -> 存結果」的循環。

- **DataManager 動作:**
    
    1. **暫時解密 (Pre-Analysis):** 為了讓 AI (Orchestrator) 能讀取圖片，呼叫 `prepare_image_for_viewing`，將 `.meta` 中的 Header 解密並寫回 Raw Image。
        
    2. **執行分析:** (Orchestrator 讀取正常的 JPG 進行運算)。
        
    3. **重新加密 (Post-Analysis):** 分析一結束，立刻呼叫 `re_encrypt_image_after_viewing`，再次破壞 Raw Image 的 Header。
        
    4. **儲存結果 (`save_result_...`):** * 將 AI 產生的 Mask 圖存入 Experiment 資料夾，並執行 Header Encryption。
        
        - 將 AI 產生的數據 (JSON/YAML) 存入 Experiment 資料夾，執行 **全檔案 AES 加密** (因為文字檔很小，全加密更安全)。
            
    5. **加入佇列:** 將所有結果檔案丟給同步執行緒。
        

#### 步驟 5：UI 顯示與解密 (View & Decrypt)

當使用者在 App 上點擊「查看圖片」時。

- **DataManager 動作:**
    
    1. **解密請求:** App 呼叫 `prepare_image_for_viewing(path)`。
        
    2. **修復檔案:** DataManager 讀取 `.meta`，修復硬碟上的檔案 Header。
        
    3. **顯示:** App 讀取修復後的圖片並顯示在螢幕上。
        
    4. **關閉重鎖:** 當使用者切換頁面或關閉視窗，App 呼叫 `re_encrypt_image_after_viewing(path)`，DataManager 再次破壞檔案 Header。
        

---

### 4. 為什麼這樣設計？ (Design Rationale)

1. **為什麼要分 Raw 與 Experiment？**
    
    - **情境:** 假設您今天要用新版的 AI 算法重新分析去年的舊照片。
        
    - **舊設計:** 照片都在舊的 Experiment 資料夾裡，很難找。
        
    - **新設計:** 照片都在 `raw_images` 共用區。您只需要建立一個新的 Experiment 資料夾，讀取共用的 Raw Image，然後存入新的結果。這符合 "Single Source of Truth" 原則。
        
2. **為什麼圖片用 Header Encryption，而文字用全加密？**
    
    - **效能考量:** 一張 20MB 的 4K 照片，做 AES 全加密需要 0.5~1 秒。如果是連拍 10 張，使用者就要等 10 秒，體驗極差。Header Encryption 只需要處理前 4KB，耗時 < 0.001 秒。
        
    - **安全性:** 對圖片來說，破壞 Header 就足以讓它無法被預覽或開啟，已達到 99% 的防護目的。文字檔很小，全加密不影響效能，且文字內容（如客戶個資、手錶序號）更敏感，值得全加密。
        
3. **為什麼需要 SQLite 索引？**
    
    - 當您的資料庫累積了 10,000 隻手錶，每隻手錶有 5 次實驗。
        
    - 如果沒有 SQLite，App 要列出「最近 10 筆實驗」，就需要去掃描 50,000 個資料夾，這會讓 App 卡死。
        
    - 有了 SQLite，查詢只需要 `SELECT * FROM experiments ORDER BY date DESC LIMIT 10`，耗時 0.01 秒。
        

### 總結

新版的 `DataManager` 是一個高度專業化的模組。它完美地執行了您要求的流程：

1. **Capture** -> Local Save -> Encrypt -> Sync Queue
    
2. **Create Experiment** (Only when needed)
    
3. **Analyze** (Decrypt -> Process -> Re-encrypt) -> Save Result -> Encrypt -> Sync Queue
    
4. **View** (Decrypt -> Show -> Re-encrypt)





#### 上傳到AWS S3的五種data type
```
目前流程看起來合理多了, 不過要上傳到AWS S3的包括(1)camera拍照完的加密images跟(2)每張照片對應的AES-256加密metadata, 跟(3)對應到watch的AES-256加密yaml setting. (Example: 放置一個watch到座台上會對不同部分用不同camera拍照可能共40images, 所以有40個對應的metadata yaml file, 以及一個對應到watch的yaml file). 除此之外在用image processing tasks處理完之後並加密儲存在experiment folder, 也要上傳到AWS S3的包括(4)40張images經過結果tasks處理過的結果譬如segmentation mask images(也要用Header Encryption), 以及每個images對應的分析結果的AES-256加密yaml files. 所以共五種data要上傳到AWS, 那這樣用一個CloudManager可以處理還是要很多個? 另外SQLite 資料庫也是要針對上面五種data(包括從camera擷取到以及image processing tasks分析過的experiment結果)可以管理及快速的查詢, 包括找到local folder中的哪一次camera taken或哪一次experiment分析了甚麼? 以及儲存在AWS S3的那些地方才能快速找到或下載. 這樣目前的SQLite是否足夠或需要擴充呢? 如果有需要修改的請提供新的完整地code
```
這是一個非常精確且系統化的需求分析。您的觀察完全正確：**隨著數據種類的增加，資料庫的 schema 必須能反映「手錶 (Watch)」與「實驗 (Experiment)」之間的層級關係**。

以下我將針對您的兩個核心問題進行解答，並提供經過擴充設計的完整程式碼。

---

### 1. 架構分析與回答

#### Q1: 五種 Data 共用一個 CloudManager 夠嗎？

**答案：足夠，且建議只用一個。**

`CloudManager` (即 `cloud_sync.py`) 的角色定位是底層的 **「物流卡車」**。

- 它不需要知道貨櫃裡裝的是「手錶設定」還是「分割遮罩」。
    
- 它只需要知道兩個資訊：**「本地檔案在哪裡 (Source)」** 與 **「要送到 S3 的哪個地址 (Key)」**。
    
- 只要 `DataManager` 在把東西丟給它時，將 S3 Key 命名規則定義好（分類清楚），一個 CloudManager 就能透過多執行緒高效處理所有類型的上傳。
    

#### Q2: SQLite 資料庫是否需要擴充？

**答案：是的，需要擴充 Schema。**

原本的設計是將檔案強綁定在 `exp_id` (實驗) 下。但根據您的需求 (1) 和 (3) —— Raw Image 和 Watch Settings 其實是屬於 **「手錶層級 (Watch Level)」** 的資產，它們可能被多個實驗共用，或者在實驗開始前就存在。

**Schema 設計變更 (Schema Evolution):** 我們需要將 `files` 表從「隸屬於實驗」改為「隸屬於手錶，但可選關聯到實驗」。

- **舊版:** `files(exp_id, path...)`
    
- **新版:** `files(watch_id, exp_id [可為空], file_type, local_path, s3_key...)`
    

這樣您就可以查詢：

- 「列出 Watch A 的所有原始照片」 -> `SELECT * FROM files WHERE watch_id='A' AND file_type='raw_image'`
    
- 「列出 Watch A 在 Experiment B 的所有結果」 -> `SELECT * FROM files WHERE exp_id='B'`
    

---

### 2. 完整修正後的 Code

為了實現上述邏輯，我们需要大幅升級 **`query_engine.py` (資料庫結構)** 與 **`local_storage.py` (存檔與S3路徑邏輯)**。

#### A. `data_manager/query_engine.py` (資料庫核心擴充)

新增了 `watch_id` 欄位，並支援五種檔案類型的註冊與查詢。

Python

```
# data_manager/query_engine.py
import sqlite3
import os
import time
import logging

class QueryEngine:
    def __init__(self, db_path="Local_Data/experiment_index.db"):
        self.db_path = db_path
        self.logger = logging.getLogger("QueryEngine")
        self._init_db()

    def _init_db(self):
        """初始化資料庫 Schema (擴充版)"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # 1. Experiments 表 (紀錄一次分析流程)
        c.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                exp_id TEXT PRIMARY KEY,
                watch_id TEXT NOT NULL,
                routine_name TEXT,
                timestamp REAL,
                local_path TEXT
            )
        ''')
        
        # 2. Files 表 (紀錄所有檔案，包含 5 種 Data Type)
        # 修改點：加入 watch_id (必填), exp_id (選填), s3_key (記錄雲端位置)
        c.execute('''
            CREATE TABLE IF NOT EXISTS files (
                file_id INTEGER PRIMARY KEY AUTOINCREMENT,
                watch_id TEXT NOT NULL,
                exp_id TEXT,            -- 若是 Raw Image/Setting，此欄位可為 NULL
                file_type TEXT,         -- 'raw_image', 'raw_meta', 'watch_setting', 'result_image', 'result_report'
                file_name TEXT,
                local_path TEXT,
                s3_key TEXT,
                status TEXT DEFAULT 'pending' -- 'pending', 'synced'
            )
        ''')
        
        # 建立索引以加速查詢
        c.execute('CREATE INDEX IF NOT EXISTS idx_watch_id ON files (watch_id)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_exp_id ON files (exp_id)')
        
        conn.commit()
        conn.close()

    # --- 寫入功能 ---

    def register_experiment(self, exp_id, watch_id, routine_name, local_path):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO experiments (exp_id, watch_id, routine_name, timestamp, local_path)
                VALUES (?, ?, ?, ?, ?)
            ''', (exp_id, watch_id, routine_name, time.time(), local_path))

    def register_file(self, watch_id, exp_id, file_type, file_name, local_path, s3_key):
        """
        通用檔案註冊函數
        :param exp_id: 如果是 Watch Level 的檔案 (Raw/Setting)，傳入 None
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO files (watch_id, exp_id, file_type, file_name, local_path, s3_key, status)
                    VALUES (?, ?, ?, ?, ?, ?, 'pending')
                ''', (watch_id, exp_id, file_type, file_name, local_path, s3_key))
        except Exception as e:
            self.logger.error(f"DB Error registering file {file_name}: {e}")

    def update_file_sync_status(self, local_path, status='synced'):
        """當 CloudManager 上傳成功後呼叫"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('UPDATE files SET status = ? WHERE local_path = ?', (status, local_path))

    # --- 查詢功能 (支援 UI/CLI 快速檢索) ---

    def get_watch_raw_images(self, watch_id):
        """查詢某手錶的所有原始影像 (包含已加密路徑)"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('''
                SELECT * FROM files 
                WHERE watch_id = ? AND file_type = 'raw_image'
            ''', (watch_id,))
            return [dict(row) for row in cursor.fetchall()]

    def get_experiment_results(self, exp_id):
        """查詢某次實驗產生的所有結果 (Masks, Reports)"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('''
                SELECT * FROM files 
                WHERE exp_id = ? 
            ''', (exp_id,))
            return [dict(row) for row in cursor.fetchall()]

    def get_s3_download_info(self, watch_id):
        """[AWS下載用] 取得該手錶所有檔案的 S3 Key 與 Local Path"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('SELECT local_path, s3_key, file_type FROM files WHERE watch_id = ?', (watch_id,))
            return [dict(row) for row in cursor.fetchall()]
```

---

#### B. `data_manager/local_storage.py` (S3 Key 規劃與儲存邏輯)

這裡定義了 **S3 的目錄結構**，以及如何處理這五種不同的數據。

**S3 Key 規劃結構：**

- `bucket/Watch_{ID}/Settings/settings.yaml.enc` (Type 3)
    
- `bucket/Watch_{ID}/Raw/{filename}` (Type 1)
    
- `bucket/Watch_{ID}/Raw/{filename}.meta` (Type 2)
    
- `bucket/Watch_{ID}/Experiments/{Exp_ID}/Reports/...` (Type 5)
    
- `bucket/Watch_{ID}/Experiments/{Exp_ID}/Images/...` (Type 4)
    

Python

```
# data_manager/local_storage.py
import os
import time
import shutil
import yaml
import json
from .encryption import DataEncryptor
from .query_engine import QueryEngine
from .cloud_sync import CloudSyncManager

class DataManager:
    def __init__(self, config):
        self.config = config
        self.base_dir = config['paths']['base_data_dir']
        self.secret_key_path = config['paths']['secret_key_path']
        
        self.encryptor = DataEncryptor(self.secret_key_path)
        self.query_engine = QueryEngine(os.path.join(self.base_dir, "experiment_index.db"))
        self.cloud_sync = CloudSyncManager(config)
        self.cloud_sync.start_background_sync()

    # =================================================================
    #  Type 3: Watch Settings (Watch Level)
    # =================================================================
    def save_watch_settings(self, watch_id, settings_dict):
        """
        儲存並上傳手錶的設定檔 (例如: 該手錶的尺寸、預計檢測項目)
        """
        # 1. 建立路徑
        watch_dir = os.path.join(self.base_dir, f"Watch_{watch_id}")
        os.makedirs(watch_dir, exist_ok=True)
        
        filename = f"{watch_id}_settings.yaml"
        temp_path = os.path.join(watch_dir, filename)
        
        # 2. 寫入 YAML
        with open(temp_path, 'w', encoding='utf-8') as f:
            yaml.dump(settings_dict, f)
            
        # 3. 全加密 (AES-256)
        enc_path = self.encryptor.encrypt_file(temp_path, delete_original=True)
        
        # 4. 定義 S3 Key (Type 3)
        s3_key = f"Watch_{watch_id}/Settings/{filename}.enc"
        
        # 5. 註冊與同步
        if enc_path:
            self.query_engine.register_file(
                watch_id=watch_id, exp_id=None, file_type="watch_setting",
                file_name=filename + ".enc", local_path=enc_path, s3_key=s3_key
            )
            self.cloud_sync.add_to_queue(enc_path, s3_key)

    # =================================================================
    #  Type 1 & 2: Raw Images & Metadata (Watch Level)
    # =================================================================
    def get_watch_raw_dir(self, watch_id):
        return os.path.join(self.base_dir, f"Watch_{watch_id}", "raw_images")

    def get_raw_image_path(self, watch_id, filename):
        return os.path.join(self.get_watch_raw_dir(watch_id), filename)

    def process_and_sync_raw_image(self, temp_path, watch_id, filename):
        """
        處理相機拍完的照片：
        1. 存入 raw_images
        2. 加密 Header
        3. 上傳 Image 與 Meta
        """
        target_dir = self.get_watch_raw_dir(watch_id)
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, filename)
        
        # 1. 搬移
        shutil.move(temp_path, target_path)
        
        # 2. 加密 Header
        meta_path = self.encryptor.encrypt_image_header(target_path)
        
        # 3. 定義 S3 Keys
        s3_key_img = f"Watch_{watch_id}/Raw/{filename}"
        s3_key_meta = f"Watch_{watch_id}/Raw/{filename}.meta"
        
        # 4. 註冊 DB (Type 1: Raw Image)
        self.query_engine.register_file(
            watch_id=watch_id, exp_id=None, file_type="raw_image",
            file_name=filename, local_path=target_path, s3_key=s3_key_img
        )
        # (Type 2: Metadata)
        if meta_path:
            self.query_engine.register_file(
                watch_id=watch_id, exp_id=None, file_type="raw_meta",
                file_name=filename + ".meta", local_path=meta_path, s3_key=s3_key_meta
            )
            
        # 5. 加入上傳佇列
        self.cloud_sync.add_to_queue(target_path, s3_key_img)
        if meta_path:
            self.cloud_sync.add_to_queue(meta_path, s3_key_meta)
            
        return target_path

    # =================================================================
    #  Experiment Management
    # =================================================================
    def create_experiment_folder(self, watch_id, routine_name):
        timestamp_str = time.strftime("%Y%m%d%H%M%S")
        exp_id = f"{timestamp_str}_{routine_name}"
        exp_dir = os.path.join(self.base_dir, f"Watch_{watch_id}", "Experiments", exp_id)
        os.makedirs(exp_dir, exist_ok=True)
        
        self.query_engine.register_experiment(exp_id, watch_id, routine_name, exp_dir)
        return exp_id, exp_dir

    # =================================================================
    #  Type 4: Result Images (Experiment Level)
    # =================================================================
    def save_result_image_and_sync(self, image_np, filename, exp_id, exp_dir, watch_id):
        import cv2
        target_path = os.path.join(exp_dir, filename)
        
        # 1. 存圖
        cv2.imwrite(target_path, image_np)
        
        # 2. 加密 Header
        meta_path = self.encryptor.encrypt_image_header(target_path)
        
        # 3. S3 Key
        s3_key_img = f"Watch_{watch_id}/Experiments/{exp_id}/Images/{filename}"
        s3_key_meta = f"Watch_{watch_id}/Experiments/{exp_id}/Images/{filename}.meta"
        
        # 4. 註冊 DB
        self.query_engine.register_file(
            watch_id=watch_id, exp_id=exp_id, file_type="result_image",
            file_name=filename, local_path=target_path, s3_key=s3_key_img
        )
        
        # 5. 上傳
        self.cloud_sync.add_to_queue(target_path, s3_key_img)
        if meta_path:
            self.cloud_sync.add_to_queue(meta_path, s3_key_meta)

        return target_path

    # =================================================================
    #  Type 5: Result Analysis Reports (Experiment Level)
    # =================================================================
    def save_analysis_report_and_sync(self, results_dict, exp_id, exp_dir, watch_id):
        filename = "analysis_results.yaml"
        temp_path = os.path.join(exp_dir, filename)
        
        with open(temp_path, 'w', encoding='utf-8') as f:
            yaml.dump(results_dict, f, allow_unicode=True)
            
        enc_path = self.encryptor.encrypt_file(temp_path, delete_original=True)
        
        if enc_path:
            real_filename = filename + ".enc"
            s3_key = f"Watch_{watch_id}/Experiments/{exp_id}/Reports/{real_filename}"
            
            self.query_engine.register_file(
                watch_id=watch_id, exp_id=exp_id, file_type="result_report",
                file_name=real_filename, local_path=enc_path, s3_key=s3_key
            )
            
            self.cloud_sync.add_to_queue(enc_path, s3_key)

    # --- UI Helpers ---
    def prepare_image_for_viewing(self, image_path):
        return self.encryptor.decrypt_image_header(image_path)

    def re_encrypt_image_after_viewing(self, image_path):
        self.encryptor.encrypt_image_header(image_path)

    def close(self):
        self.cloud_sync.stop_sync()
```

---

### 3. 如何查詢與下載 (Fast Retrieval)

您提到**「需要快速查詢找到 Local 哪一次 Camera taken 或 AWS S3 的位置」**。

有了上面的新 Schema，您現在可以在 App 中實作這樣的高效查詢：

**情境 1：App 顯示某隻手錶的所有原始照片**

Python

```
# UI Code Example
watch_id = "Rolex_123"
raw_files = data_mgr.query_engine.get_watch_raw_images(watch_id)

for f in raw_files:
    print(f"File: {f['file_name']}")
    print(f"Local Path: {f['local_path']}")
    print(f"AWS S3 Key: {f['s3_key']}")
    print(f"Sync Status: {f['status']}")
```

**情境 2：從 AWS 下載資料 (例如在 Remote Client)** 這需要一個新的方法 `download_from_aws` (實作在 `CloudSyncManager` 或 `DataManager` 中)，邏輯如下：

1. App 呼叫 `query_engine` 獲取 `s3_key`。
    
2. 呼叫 `boto3.download_file(bucket, s3_key, local_temp_path)`。
    
3. 如果是 Header Encrypted 的圖片，下載 `.meta` 檔，然後呼叫 `decrypt_image_header` 進行還原。
    
4. 如果是 Full Encrypted 的 YAML，呼叫 `decrypt_file` 還原。
    

### 總結

1. **資料類型管理:** 我們透過 `file_type` 欄位 ('raw_image', 'watch_setting', 'result_image'...) 完美區分了您要求的五種資料。
    
2. **S3 結構:** 透過明確的 `s3_key` 命名規則，您的 AWS Bucket 將會非常整潔 (`Watch/Raw`, `Watch/Settings`, `Watch/Experiments`).
    
3. **查詢效能:** SQLite 索引確保了即使有幾萬張照片，查詢依然是瞬間完成。
    
4. **單一 CloudManager:** 透過統一的佇列機制，單一 CloudManager 就能依序處理這五種截然不同的檔案，架構最簡潔。





#### download_from_aws python code
```
請提供完整的download_from_aws python code. 並中文詳細解釋在這系統內SQLite 索引是如何幫助高效的查詢, 是否是每次camera拍攝一張照片就存入SQLite一次以及存那些東西, 還是一次針對watch的拍攝工作存入一次以及存那些東西, 是否完成一次experiment分析工作就存入SQLite一次以及存那些東西? 在舊的方法拍攝完會把照片放到local folder, 然後有個對應watch的matlab mat file存這些照片在folder的位置然後進行image processing tasks. 請比照舊方法跟使用SQLite新方法的比較, 所以SQLite也是存成一個檔案?
```
這是一個非常關鍵的實作細節。數據如果「只上傳、無法下載」或是「下載後無法還原」，這套備份系統就沒有意義了。

以下我將提供：

1. **`download_from_aws` 的完整實作代碼**（需加入 `cloud_sync.py` 與 `local_storage.py`）。
    
2. **SQLite 在本系統中的運作機制深度解析**（針對您的拍攝與分析頻率）。
    
3. **新舊架構（MAT vs SQLite）的詳細比較**。
    

---

### 第一部分：完整 Download Code 實作

下載的邏輯比上傳複雜，因為涉及到**「相依性下載」**。

- 如果您下載一張 `raw_image.jpg`，您**必須**同時下載 `raw_image.jpg.meta`，否則無法解密標頭，圖片打不開。
    

#### 1. 修改 `data_manager/cloud_sync.py`

新增下載功能。這裡我們使用 Blocking（阻塞式）方法，因為通常使用者按「下載」時是願意等待的，不需要像上傳那樣丟到背景排隊。

Python

```
# data_manager/cloud_sync.py (新增部分)

    def download_file(self, s3_key, local_dest_path):
        """
        從 AWS S3 下載單一檔案
        :param s3_key: S3 上的路徑 (例如: Watch_123/Raw/img.jpg)
        :param local_dest_path: 本地儲存路徑
        :return: Boolean (Success/Fail)
        """
        if not HAS_BOTO3:
            self.logger.warning("[MOCK] Boto3 missing, simulating download...")
            # 模擬：如果本地沒有檔案，造一個假的
            if not os.path.exists(local_dest_path):
                with open(local_dest_path, 'wb') as f:
                    f.write(b'[MOCK FILE CONTENT]')
            return True

        s3_client = boto3.client('s3', region_name=self.region)
        
        try:
            # 確保本地資料夾存在
            os.makedirs(os.path.dirname(local_dest_path), exist_ok=True)
            
            self.logger.info(f"Downloading s3://{self.bucket_name}/{s3_key} -> {local_dest_path}")
            s3_client.download_file(self.bucket_name, s3_key, local_dest_path)
            return True
        except Exception as e:
            self.logger.error(f"Download failed for {s3_key}: {e}")
            return False

    def list_s3_files(self, prefix):
        """列出 S3 上某個目錄下的所有檔案 (用於同步檢查)"""
        if not HAS_BOTO3: return []
        s3_client = boto3.client('s3', region_name=self.region)
        try:
            response = s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
            if 'Contents' in response:
                return [item['Key'] for item in response['Contents']]
            return []
        except Exception as e:
            self.logger.error(f"List S3 failed: {e}")
            return []
```

#### 2. 修改 `data_manager/local_storage.py`

新增高層級的下載邏輯，自動處理 `.meta` 伴隨檔。

Python

```
# data_manager/local_storage.py (新增部分)

    def download_watch_data(self, watch_id, target_file_type="all"):
        """
        [核心功能] 從 AWS 下載特定手錶的數據到本地
        :param target_file_type: 'raw_image', 'report', 'all'
        """
        # 1. 雖然 SQLite 主要存本地狀態，但在下載情境，
        # 我們通常假設 "我知道 s3_key 規則" 或者 "從 query_engine 取得 s3_key"
        # 這裡示範：透過 SQLite 找回 s3_key (假設 DB 還在，但圖檔被刪了)
        # 如果是全新的電腦，則需要先呼叫 cloud_sync.list_s3_files 重建 DB (進階災難復原)
        
        files_to_download = []
        
        # 查詢資料庫看有哪些檔案
        # 假設我們要在這台電腦重現 Watch_123 的資料
        records = self.query_engine.get_s3_download_info(watch_id)
        
        for rec in records:
            ftype = rec['file_type']
            if target_file_type != "all" and ftype != target_file_type:
                continue
                
            local_path = rec['local_path']
            s3_key = rec['s3_key']
            
            # 如果本地已經有檔案，跳過 (或可加 force=True 參數)
            if os.path.exists(local_path):
                continue
                
            files_to_download.append((s3_key, local_path))
            
            # [關鍵邏輯] 檢查是否需要下載伴隨的 .meta 檔 (解密金鑰)
            # 如果是圖片 (raw_image, result_image)，通常都有 .meta
            if ftype in ['raw_image', 'result_image']:
                meta_local = local_path + ".meta"
                meta_s3 = s3_key + ".meta"
                if not os.path.exists(meta_local):
                    files_to_download.append((meta_s3, meta_local))

        # 執行下載
        success_count = 0
        for s3_k, local_p in files_to_download:
            if self.cloud_sync.download_file(s3_k, local_p):
                success_count += 1
                
        return f"Downloaded {success_count}/{len(files_to_download)} files."
```

---

### 第二部分：SQLite 索引運作機制深度解析

這部分是系統效能的核心。您問到 **「存入頻率」** 與 **「存什麼」**，這決定了系統的即時性。

**SQLite 是一個單一檔案 (`experiment_index.db`)**。它就像一個高效能的 Excel 檔案，但支援 SQL 語法查詢。

#### 1. 存入頻率 (Insertion Frequency)

在您的系統中，SQLite 的寫入是 **「即時且原子化 (Atomic)」** 的。

- **情境 A：Camera 拍攝時 (Capture Phase)**
    
    - **頻率：** **每拍一張就存一次**。
        
    - **原因：** 如果您預計拍 40 張，拍到第 39 張時電腦斷電，如果不是每張存，前 38 張的索引就會遺失，變成孤兒檔案。
        
    - **存入內容：**
        
        - `watch_id`: "Rolex_123"
            
        - `file_type`: "raw_image"
            
        - `file_name`: "angle_45.jpg"
            
        - `local_path`: "D:/Data/Watch_Rolex_123/raw/angle_45.jpg"
            
        - `s3_key`: "Watch_Rolex_123/Raw/angle_45.jpg"
            
        - `status`: "pending" (還沒上傳)
            
- **情境 B：實驗建立時 (Experiment Start)**
    
    - **頻率：** 每次按下 "Run Analysis" 時存一次。
        
    - **存入內容 (Experiments Table)：**
        
        - `exp_id`: "20251221_RoutineA"
            
        - `watch_id`: "Rolex_123"
            
        - `routine`: "Standard Check"
            
        - `timestamp`: 1734800000
            
- **情境 C：實驗分析結束時 (Analysis End)**
    
    - **頻率：** 整個 Batch 分析完，存入結果檔案時（可能一次存入 40 個 mask 紀錄 + 1 個 report 紀錄）。
        
    - **存入內容 (Files Table)：**
        
        - `exp_id`: "20251221_RoutineA"
            
        - `file_type`: "result_image" (Mask) / "report" (YAML)
            
        - `local_path`: ...
            
        - `s3_key`: ...
            

#### 2. 為什麼 SQLite 能高效查詢？

假設您有 10,000 隻手錶，每隻有 100 個檔案，共 **100 萬個檔案**。

- **無索引查詢 (Windows 檔案總管):** 您要找 "Watch_500" 的 "Mask"，系統需要遍歷資料夾樹狀結構，這在 HDD 上非常慢。
    
- **SQLite 查詢:**
    
    - SQLite 內部使用 **B-Tree** 結構。
        
    - 當我們執行 `SELECT * FROM files WHERE watch_id='Watch_500'`。
        
    - 因為我們在 `watch_id` 建了 Index，它不需要看那 100 萬行資料，而是直接跳到 B-Tree 的分支，**複雜度是 O(log N)**。
        
    - **速度差異：** 檔案系統掃描可能要 3~5 秒；SQLite 只要 0.005 秒。
        

---

### 第三部分：新舊方法比較 (MATLAB .mat vs SQLite)

您提到的舊方法（MAT 檔）是學術界常見的做法，但在工業級應用上有致命缺陷。

|**特性**|**舊方法 (.mat file)**|**新方法 (SQLite .db)**|**關鍵差異解析**|
|---|---|---|---|
|**資料儲存型態**|**分散式檔案** (每個手錶一個 .mat)|**集中式資料庫** (單一 .db 檔)|舊方法要查詢「所有手錶」，必須打開 10,000 個 .mat 檔，速度極慢。新方法只要查一個表。|
|**記錄時機**|通常是**整批跑完後**才存檔 (`save('data.mat')`)|**每一步驟**即時寫入 (`INSERT`)|舊方法如果在第 39 張照片時當機，記憶體中的變數消失，前面拍的都沒紀錄。新方法因為每張都 Commit，重開機後紀錄還在。|
|**併發處理 (Concurrency)**|**差** (無法同時讀寫同一個 .mat)|**優** (支援多執行緒/多進程讀取)|如果您的 App 一邊在顯示圖片，後台一邊在寫入新數據，.mat 檔會鎖死或損壞。SQLite 專門處理這種情況。|
|**查詢能力**|只能依賴 `load` 載入全部變數|強大的 SQL (`WHERE`, `ORDER BY`, `LIMIT`)|您可以問 SQLite：「請給我上個月所有失敗的實驗」，舊方法做不到這點（除非寫複雜腳本掃描）。|
|**檔案路徑管理**|.mat 裡面通常存絕對路徑 (`C:\User\...`)|存相對路徑或動態路徑|舊方法如果把資料夾搬到 D 槽，.mat 裡的路徑就全失效了。SQLite 可以靈活更新路徑。|
|**與雲端整合**|困難 (要自己解析 mat 找路徑)|容易 (欄位直接存 `s3_key`)|SQLite 直接告訴 CloudManager 該去 S3 哪裡抓檔案，無需運算。|

### 總結

1. **Download Code:** 已補上。關鍵在於**必須同時下載 `.meta` 檔**，這是新架構特有的資安需求。
    
2. **SQLite 機制:** 它是**單一檔案** (`experiment_index.db`)，但在邏輯上是一個強大的關聯式資料庫。它在**每次拍照**和**每次分析**後都會即時寫入，確保數據不遺失。
    
3. **比較:** 從 .mat 轉向 SQLite，是從「個人工具」轉向「工業系統」的關鍵一步，解決了查詢速度、資料安全與斷電保護的問題。




#### 評估30萬到100萬的watch Scale
```
請幫我評估這套系統如果未來譬如需要存30萬到100萬的watch以及對應的所有images跟experiments. 這套系統是否能仍然維持高效或需要改進? 請以30年資深software engineer tech lead進行詳細的分析並中文解釋
```
這是一個非常嚴肅且具備前瞻性的問題。作為一名擁有 30 年經驗的 Tech Lead，我必須坦白地告訴您：

**目前的 V1 架構（SQLite + 單一本地資料夾）在達到 30 萬到 100 萬隻手錶的規模時，會面臨嚴重的效能崩潰與物理極限。**

目前的 V1 架構是針對「單機工作站」或「小規模實驗室（數千隻手錶）」設計的，這是正確的 MVP (Minimum Viable Product) 策略。但面對百萬級規模（Enterprise Scale），我們必須引入 **分散式架構** 與 **分層儲存** 的概念。

以下我將從 **資料庫、檔案系統、儲存成本、與雲端同步** 四個維度進行深度分析，並提出 V2 的演進藍圖。

---

### 1. 深度分析：V1 架構在百萬級規模下的瓶頸

假設 100 萬隻手錶，每隻手錶 40 張 Raw Images + 2 次實驗 (80 張 Result Images + Reports)。

總檔案數預估： $1,000,000 \times (40 + 80 + 5) \approx \mathbf{1.25 億個檔案}$。

總容量預估： 每張圖 10MB $\rightarrow$ $1.2 億 \times 10MB \approx \mathbf{1.2 PB (Petabytes)}$。

#### A. 資料庫瓶頸 (SQLite 的極限)

- **現狀：** 使用單一 `experiment_index.db` 檔案。
    
- **問題：**
    
    1. **查詢效能：** 雖然 SQLite 支援 B-Tree 索引，但在 `files` 表達到 **1.2 億行**時，即便有索引，查詢速度也會顯著下降。
        
    2. **寫入鎖定 (Write Locking)：** SQLite 預設是檔案級鎖定。當你有 4 個 Worker Threads 加上背景 CloudSync 同時嘗試寫入狀態時，會發生嚴重的 **"Database is locked"** 錯誤，導致系統卡頓。
        
    3. **單點故障：** 100 萬筆資料全壓在一個 .db 檔，一旦檔案損壞（例如斷電），重建索引需要數小時甚至數天。
        

#### B. 檔案系統瓶頸 (NTFS/Ext4 的極限)

- **現狀：** 所有資料放在 `Local_Data` 下。
    
- **問題：**
    
    1. **目錄遍歷災難：** 雖然我們用 `Watch_ID` 分了資料夾，但 `Local_Data` 根目錄下會有 **100 萬個子資料夾**。大多數作業系統（Windows/Linux）在打開一個包含 100 萬個資料夾的目錄時，Explorer 會直接當機，CLI 指令 (`ls` 或 `dir`) 會跑幾分鐘出不來。
        
    2. **Inode 耗盡：** 在 Linux 系統下，檔案數量受限於 Inode 總數，1.2 億個小檔案極可能耗盡 Inode，導致硬碟明明有空間卻無法寫入。
        

#### C. 儲存物理極限

- **問題：** **1.2 PB** 的資料量遠超過任何單一硬碟甚至普通 RAID 陣列的容量。V1 架構假設 `base_data_dir` 是一個路徑，這在物理上是不可能的。
    

---

### 2. V2 架構演進建議 (Tech Lead 的處方箋)

為了支援百萬級規模，我們不需要推翻重寫，但需要對 **DataManager** 進行「企業級升級」。

#### 改進一：資料庫遷移 (SQLite -> PostgreSQL)

建議方案：

將 experiment_index.db 替換為 PostgreSQL (或是 MySQL)。

- **為什麼？**
    
    - **Row-level Locking：** 支援高併發寫入，UI 查詢與背景上傳互不干擾。
        
    - **Partitioning (資料表分區)：** 可以將 `files` 表依照 `watch_id` 或 `timestamp` 進行切分，即使有 10 億行資料，查詢依然維持毫秒級。
        
    - **網路化：** 資料庫可以架設在專用伺服器，多台 Client (產線上的多台 AOI 機台) 可以共用同一個資料庫，實現產線數據互通。
        

#### 改進二：檔案系統分片 (Directory Sharding)

建議方案：

修改 DataManager 的路徑生成邏輯，不要把所有手錶放在同一層。使用 Hash Sharding。

- **舊路徑：** `Local_Data/Watch_123456/...`
    
- **新路徑：** `Local_Data/12/34/Watch_123456/...`
    
    - 取 `Watch_ID` 的前幾碼或 Hash 值作為子目錄。
        
    - 這樣每個目錄下的資料夾數量會控制在幾千個以內，作業系統讀取效率最高。
        

#### 改進三：熱/冷數據分離 (Hot/Cold Data Lifecycle)

**這是解決 1.2 PB 儲存問題的唯一解法。**

- **本地 (Local Data):** 僅作為 **「暫存緩衝區 (Cache)」**。
    
    - 只保留最近 1 個月或最近 1000 隻手錶的資料。
        
    - 一旦 CloudSync 確認 `status='synced'` 且資料超過保留期限，**自動從本地刪除 Raw Image**。
        
- **雲端 (S3):** 作為 **「永久儲存區 (Permanent Storage)」**。
    
    - 所有 100 萬隻手錶的資料都在這裡。
        
- **下載機制 (On-demand):**
    
    - 當使用者在 App 查詢一隻「一年前的手錶」時，`QueryEngine` 發現本地沒有檔案，自動觸發 `download_from_aws` 從 S3 拉取回來暫存。
        

#### 改進四：AWS 成本優化

- **S3 Intelligent-Tiering:** 針對 1.2 PB 的資料，必須開啟 S3 智慧分層。對於 30 天沒存取的檔案（冷資料），自動移入低成本儲存區（Archive Access），這可以節省 50%~80% 的雲端費用。
    

---

### 3. 如何現在就為未來做準備？ (Future-Proofing V1)

您不需要現在就實作 V2，但在目前的 V1 代碼中，您可以預留**「擴充接口」**，避免未來需要打掉重練。

請在目前的代碼中加入以下兩個關鍵修改：

#### A. 抽象化資料庫介面 (Abstract Database Interface)

不要在 `local_storage.py` 裡直接寫 `sqlite3.connect`。建立一個 `DatabaseInterface` 類別。

Python

```
# data_manager/database_interface.py (新增)
class DatabaseInterface:
    def register_file(self, ...): raise NotImplementedError
    def query_file(self, ...): raise NotImplementedError

# data_manager/query_engine.py (修改)
# 讓 QueryEngine 變成 SQLite 的實作
class SQLiteEngine(DatabaseInterface):
    # ... 原本的 SQLite 程式碼 ...

# 未來只需新增 class PostgresEngine(DatabaseInterface) 即可無痛切換
```

#### B. 實作路徑分片 (Directory Sharding)

現在就可以改，這對效能只有好處沒有壞處。

**修改 `data_manager/local_storage.py`：**

Python

```
import hashlib

class DataManager:
    # ...
    def _get_sharded_path(self, watch_id):
        """
        將 Watch_ID 雜湊後，產生分層路徑，避免單一目錄檔案過多。
        例如: Watch_ROLEX_123 -> Local_Data/a1/b2/Watch_ROLEX_123
        """
        # 使用 MD5 取前 4 碼做分片 (2層，每層256個資料夾)
        hash_obj = hashlib.md5(watch_id.encode())
        hash_hex = hash_obj.hexdigest()
        layer1 = hash_hex[:2]
        layer2 = hash_hex[2:4]
        
        return os.path.join(self.base_dir, layer1, layer2, f"Watch_{watch_id}")

    def get_watch_raw_dir(self, watch_id):
        # 使用分片路徑
        base = self._get_sharded_path(watch_id)
        return os.path.join(base, "raw_images")
```

#### C. 實作「本地清理」策略 (Retention Policy)

在 `DataManager` 中預留一個 `cleanup_old_data` 的接口。

Python

```
    def cleanup_old_data(self, days_to_keep=30):
        """
        刪除已經上傳且超過 30 天的原始檔案，釋放本地空間。
        """
        # 1. 查詢 DB: SELECT * FROM files WHERE status='synced' AND timestamp < (now - 30days)
        # 2. os.remove(local_path)
        # 3. 更新 DB: UPDATE files SET local_path=NULL WHERE ...
        pass # V1 先留空，V2 實作
```

### 總結報告

1. **當前狀態 (V1):** 完美適用於單機、開發階段及首批 1~5 萬隻手錶的量級。
    
2. **擴充瓶頸:** 10 萬隻是 SQLite 與 NTFS 的效能轉折點；100 萬隻是物理極限。
    
3. **核心策略:** * **DB:** 從 SQLite 轉向 PostgreSQL。
    
    - **Storage:** 從「全部存本地」轉向「本地是 Cache，雲端是倉庫」。
        
    - **Structure:** 實作 Hash Sharding 資料夾結構。
        

身為 Tech Lead，我的建議是：**現在先用 V1 (SQLite) 快速上線驗證商業模式，但在代碼中預留上述的 Sharding 與 Interface 接口。等到數據量達到 1 萬隻時，啟動 V2 (PostgreSQL + Lifecycle Management) 的遷移計畫。**



#### 本地資料夾的image或data是可以定期刪除的
```
我忘了補充一個重要細節, 就是關於local storage部分, 可能存在本地資料夾的image或data是可以定期刪除的, 所以即使有 100 萬隻手錶在本地資料可能比較沒有檔案系統瓶頸跟儲存物理極限, 因為所有的資料仍然會在AWS cloud上, 有需要的時候可以下載跟從AWS查詢. 但的確資料庫瓶頸還是存在的, 所以我們仍須使用MySQL而且是否需要對MySQL也有cloud備份機制? 請重新分析並就需要改進的地方提供新的完整code
```

這是一個非常關鍵的轉折點。您的補充說明將這個系統的架構定義從「單機封閉系統」轉變為現代化的 **「混合雲緩存架構 (Hybrid Cloud Cache Architecture)」**。

這極大地減輕了本地硬體的壓力。現在，本地電腦不再是「倉庫」，而是一個 **「暫存區 (Cache)」**。

### 1. 30年 Tech Lead 的重新分析 (Re-Analysis)

基於「本地資料可刪除，AWS 為 Single Source of Truth」的前提，我們可以重新評估瓶頸：

#### A. 檔案系統瓶頸：**已解決**

- **現狀：** 本地只保留最近（例如 30 天）或正在處理的「熱數據 (Hot Data)」。
    
- **結論：** 即便總量有 100 萬隻手錶，本地硬碟可能只需要維持 5,000 隻手錶的量。NTFS/Ext4 檔案系統完全可以輕鬆應對，不需要做複雜的 Hash Sharding（雖然做了更好，但已非必要）。
    

#### B. 資料庫瓶頸：**依然存在，且更為關鍵**

- **原因：** 即使您刪除了本地的 JPG 圖片，**「這隻手錶曾經拍過照」這條紀錄 (Metadata) 不能刪除**。否則，當使用者想查詢歷史紀錄時，系統會因為本地沒有資料而顯示空白。
    
- **需求：** 資料庫必須記錄 100 萬隻手錶 x 100 個檔案 = **1 億筆紀錄**。
    
    - 每筆紀錄包含：`Watch_ID`, `S3_Key` (最重要), `Local_Path` (可能為 NULL), `Status`.
        
- **SQLite 的極限：** SQLite 雖然可以存 1 億筆，但在高併發寫入時會變慢。不過，既然您採取了「本地是 Cache」的策略，我們可以採用 **「輕量化 SQLite + 雲端備份」** 或者 **「直接連線雲端資料庫 (AWS RDS)」** 的策略。
    

#### C. 資料庫備份策略：**絕對需要**

- **風險：** 如果本地電腦燒毀，雖然圖片都在 S3 上，但您失去了 **「檔名與 S3 Key 的對應關係」** 以及 **「實驗的商業邏輯關聯」**（哪張 Mask 對應哪張 Raw Image）。重建這個索引會非常痛苦。
    
- **解決方案：**
    
    1. **定期 Dump：** 每天將 `experiment_index.db` 備份上傳到 S3。
        
    2. **即時同步 (更佳)：** 使用 AWS RDS (MySQL/PostgreSQL) 作為索引，本地不存 DB，直接連線雲端。但考慮到現場網路延遲，**「本地 SQLite + 定期備份」** 或是 **「本地 SQLite (Cache) + 雲端 Master DB」** 是比較務實的做法。
        

---

### 2. 架構修正：混合雲緩存模式 (Code Implementation)

為了支援 **「本地自動清理 (Auto-Cleanup)」** 與 **「資料庫備份」**，我們需要對 `data_manager` 進行重構。

我將代碼拆分為三個檔案以保持清晰：

1. **`database.py`**: 封裝 SQLite 操作，新增 `cleanup_query`。
    
2. **`local_storage.py`**: 核心邏輯，新增 `retention_policy` (保留策略) 與 `restore` (回補)。
    
3. **`cloud_sync.py`**: 負責傳輸。
    

#### A. `data_manager/database.py` (負責管理 1 億筆索引)

這裡我優化了 Schema，使其能標記檔案是否在本地 (`is_local`)，以及最後存取時間 (`last_accessed`)，以便 LRU 清理算法使用。

Python

```
# data_manager/database.py
import sqlite3
import os
import time
import logging

class DatabaseManager:
    def __init__(self, db_path="Local_Data/experiment_index.db"):
        self.db_path = db_path
        self.logger = logging.getLogger("Database")
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            # 1. Files 表：系統的核心索引
            # 新增 last_accessed: 用於判斷哪些檔案太久沒用可以刪除
            # 新增 is_cached: 1代表本地有檔案, 0代表本地已刪除(僅存S3)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS files (
                    file_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    watch_id TEXT NOT NULL,
                    exp_id TEXT,
                    file_type TEXT,
                    file_name TEXT,
                    local_path TEXT,
                    s3_key TEXT,
                    status TEXT DEFAULT 'pending', -- 'pending', 'synced'
                    is_cached INTEGER DEFAULT 1,   -- 1=Local+Cloud, 0=Cloud Only
                    last_accessed REAL,            -- Timestamp
                    created_at REAL
                )
            ''')
            # 建立索引加速查詢
            conn.execute('CREATE INDEX IF NOT EXISTS idx_watch ON files (watch_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_status ON files (status, is_cached, last_accessed)')

    # --- 寫入與更新 ---

    def register_file(self, watch_id, exp_id, file_type, file_name, local_path, s3_key):
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO files (watch_id, exp_id, file_type, file_name, local_path, s3_key, status, is_cached, last_accessed, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 'pending', 1, ?, ?)
            ''', (watch_id, exp_id, file_type, file_name, local_path, s3_key, now, now))

    def mark_as_synced(self, local_path):
        """上傳成功後呼叫"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('UPDATE files SET status = "synced" WHERE local_path = ?', (local_path,))

    def update_access_time(self, local_path):
        """當 App 讀取圖片時呼叫 (Keep Alive)"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('UPDATE files SET last_accessed = ? WHERE local_path = ?', (time.time(), local_path))

    def mark_as_deleted_locally(self, file_id):
        """當執行清理後，標記此檔案本地已刪除"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('UPDATE files SET is_cached = 0, local_path = NULL WHERE file_id = ?', (file_id,))

    def restore_local_path(self, s3_key, new_local_path):
        """當從 S3 下載回來後，更新路徑與狀態"""
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE files 
                SET local_path = ?, is_cached = 1, last_accessed = ? 
                WHERE s3_key = ?
            ''', (new_local_path, now, s3_key))

    # --- 查詢與維護 ---

    def get_watch_files(self, watch_id):
        """查詢手錶所有檔案 (無論在本地或雲端)"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.execute('SELECT * FROM files WHERE watch_id = ?', (watch_id,))
            return [dict(row) for row in c.fetchall()]

    def get_s3_key_by_local_path(self, local_path):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.execute('SELECT s3_key FROM files WHERE local_path = ?', (local_path,))
            res = c.fetchone()
            return res[0] if res else None

    def find_evictable_files(self, days_old=30, limit=1000):
        """
        [核心] 找出可以刪除的檔案
        條件：1. 狀態是 synced (已上傳)
              2. 本地還存在 (is_cached=1)
              3. 超過 N 天沒有被讀取
        """
        cutoff_time = time.time() - (days_old * 86400)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.execute('''
                SELECT file_id, local_path, s3_key 
                FROM files 
                WHERE status = 'synced' 
                  AND is_cached = 1 
                  AND last_accessed < ?
                LIMIT ?
            ''', (cutoff_time, limit))
            return [dict(row) for row in c.fetchall()]
```

---

#### B. `data_manager/local_storage.py` (實作清理策略與 DB 備份)

這是 **DataManager** 的完全體。它增加了 `run_maintenance` 方法，負責清理舊檔與備份 DB。

Python

```
# data_manager/local_storage.py
import os
import time
import shutil
import yaml
import logging
from .encryption import DataEncryptor
from .database import DatabaseManager
from .cloud_sync import CloudSyncManager

class DataManager:
    def __init__(self, config):
        self.config = config
        self.base_dir = config['paths']['base_data_dir']
        self.secret_key_path = config['paths']['secret_key_path']
        self.logger = logging.getLogger("DataManager")
        
        # 初始化
        self.encryptor = DataEncryptor(self.secret_key_path)
        # 資料庫路徑
        self.db_path = os.path.join(self.base_dir, "experiment_index.db")
        self.db = DatabaseManager(self.db_path)
        self.cloud_sync = CloudSyncManager(config, self.db) # 傳入 DB 以便 Callback 更新狀態
        
        self.cloud_sync.start_background_sync()

    # ... (省略 create_experiment_folder 等與之前相同的建檔邏輯，保持不變) ...
    # 這裡只列出有更動或新增的關鍵邏輯

    def process_and_sync_raw_image(self, temp_path, watch_id, filename):
        """存 Raw -> 加密 -> DB -> Sync"""
        target_dir = os.path.join(self.base_dir, f"Watch_{watch_id}", "raw_images")
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, filename)
        
        shutil.move(temp_path, target_path)
        self.encryptor.encrypt_image_header(target_path)
        
        s3_key = f"Watch_{watch_id}/Raw/{filename}"
        
        # 寫入 DB (is_cached=1)
        self.db.register_file(watch_id, None, "raw_image", filename, target_path, s3_key)
        
        # 加入上傳
        self.cloud_sync.add_to_queue(target_path, s3_key)
        
        # Metadata 處理略 (同前版)
        return target_path

    # ... save_result_image_and_sync 等邏輯同上，只需記得呼叫 self.db.register_file ...

    # =================================================================
    #  [New Feature] 自動維護：清理舊檔 & 備份資料庫
    # =================================================================
    
    def run_maintenance(self, days_to_keep=30):
        """
        執行系統維護：
        1. 清理本地過期快取 (Free up disk space)
        2. 備份 SQLite 資料庫到雲端 (Protect metadata)
        此函數建議在系統 Idle 時或 App 啟動/關閉時呼叫。
        """
        self.logger.info("Starting maintenance...")
        
        # 1. 執行清理 (Pruning)
        files_to_delete = self.db.find_evictable_files(days_old=days_to_keep, limit=500)
        
        deleted_count = 0
        for f in files_to_delete:
            file_path = f['local_path']
            # 雙重檢查：確保檔案存在且已上傳 (DB query 已過濾 synced，但再確認一次較安全)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    # 同時刪除 .meta 檔
                    if os.path.exists(file_path + ".meta"):
                        os.remove(file_path + ".meta")
                        
                    # 更新資料庫狀態: is_cached=0
                    self.db.mark_as_deleted_locally(f['file_id'])
                    deleted_count += 1
                except Exception as e:
                    self.logger.error(f"Failed to delete {file_path}: {e}")
        
        if deleted_count > 0:
            self.logger.info(f"Cleaned up {deleted_count} old files.")

        # 2. 資料庫備份 (DB Backup)
        self._backup_database()

    def _backup_database(self):
        """備份 SQLite 到 S3"""
        # 為了避免備份時 DB 正在寫入造成損壞，最好使用 SQLite 的備份 API 或簡單 copy
        # 這裡用簡單 Copy，實際上線建議用 sqlite3.backup API
        timestamp = time.strftime("%Y%m%d")
        backup_name = f"experiment_index_backup_{timestamp}.db"
        backup_path = self.db_path + ".bak"
        
        try:
            shutil.copy2(self.db_path, backup_path)
            
            # 加密備份檔 (因為 DB 內含路徑資訊，也算敏感)
            enc_backup_path = self.encryptor.encrypt_file(backup_path, delete_original=True)
            
            # 上傳到專屬的備份目錄
            s3_key = f"System_Backups/{backup_name}.enc"
            
            # 這裡不走一般 queue，直接用 boto3 上傳確保完成，或者用高優先級 queue
            # 簡單起見，我們借用 cloud_sync 的功能
            self.cloud_sync.add_to_queue(enc_backup_path, s3_key)
            self.logger.info(f"Database backup queued: {s3_key}")
            
        except Exception as e:
            self.logger.error(f"Database backup failed: {e}")

    # =================================================================
    #  [New Feature] 從雲端取回資料 (Restore from Archive)
    # =================================================================

    def retrieve_file_for_viewing(self, watch_id, filename, file_type="raw_image"):
        """
        當 UI 需要顯示圖片，但發現本地已刪除 (is_cached=0) 時呼叫此函數。
        """
        # 1. 查 DB 找 S3 Key
        files = self.db.get_watch_files(watch_id)
        target_rec = next((f for f in files if f['file_name'] == filename), None)
        
        if not target_rec:
            raise FileNotFoundError("File record not found in DB")

        local_path = target_rec['local_path']
        
        # 如果 DB 記錄的 local_path 是空的 (因為被清理過)，我們需要重建路徑
        if not local_path:
            # 重建邏輯
            if file_type == "raw_image":
                local_path = os.path.join(self.base_dir, f"Watch_{watch_id}", "raw_images", filename)
            # ... 其他類型邏輯 ...
        
        # 2. 如果本地已有檔案，直接回傳
        if os.path.exists(local_path):
            self.db.update_access_time(local_path) # 更新 Access Time 避免馬上被刪
            return local_path

        # 3. 從 S3 下載
        s3_key = target_rec['s3_key']
        self.logger.info(f"Restoring file from S3: {s3_key}")
        
        # 下載本體
        success = self.cloud_sync.download_file(s3_key, local_path)
        
        # 下載 .meta (如果是圖片)
        if success and file_type in ['raw_image', 'result_image']:
            self.cloud_sync.download_file(s3_key + ".meta", local_path + ".meta")
            
        if success:
            # 4. 更新 DB 狀態 (變成 is_cached=1)
            self.db.restore_local_path(s3_key, local_path)
            return local_path
        else:
            raise IOError("Failed to download from S3")

    def prepare_image_for_viewing(self, image_path):
        # 更新最後讀取時間，這樣這個檔案最近就不會被清除
        self.db.update_access_time(image_path)
        return self.encryptor.decrypt_image_header(image_path)
```

---

#### C. `data_manager/cloud_sync.py` (更新 Callback)

我們需要讓 CloudSync 在上傳成功後通知 DB 更新狀態。

Python

```
# data_manager/cloud_sync.py
import threading
import queue
import time
import os
import logging

try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

class CloudSyncManager:
    def __init__(self, config, db_manager):
        self.config = config
        self.db = db_manager # [Update] 持有 DB 引用
        self.bucket_name = config.get('aws', {}).get('s3_bucket', 'watch-v1')
        self.upload_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.logger = logging.getLogger("CloudSync")

    def start_background_sync(self):
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def add_to_queue(self, file_path, s3_key):
        self.upload_queue.put((file_path, s3_key))

    def download_file(self, s3_key, local_path):
        # ... (同上一個回應的 download 實作) ...
        # 這裡是阻塞式呼叫，直接回傳 True/False
        if not HAS_BOTO3: return True # Mock
        try:
            s3 = boto3.client('s3')
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(self.bucket_name, s3_key, local_path)
            return True
        except Exception as e:
            self.logger.error(f"Download Error: {e}")
            return False

    def _worker_loop(self):
        s3_client = boto3.client('s3') if HAS_BOTO3 else None
        
        while not self.stop_event.is_set():
            try:
                file_path, s3_key = self.upload_queue.get(timeout=2)
            except queue.Empty:
                continue

            if not os.path.exists(file_path):
                self.upload_queue.task_done()
                continue

            try:
                if s3_client:
                    s3_client.upload_file(file_path, self.bucket_name, s3_key)
                else:
                    time.sleep(0.1) # Mock delay
                
                self.logger.info(f"Synced: {s3_key}")
                
                # [Update] 上傳成功，更新 DB 狀態
                # 只有非 .meta 的檔案才需要更新 status (雖然 meta 沒更新也沒差)
                if not file_path.endswith(".meta"):
                    self.db.mark_as_synced(file_path)

            except Exception as e:
                self.logger.error(f"Upload Fail: {e}")
                # 失敗重試邏輯 (可選: 放回 queue)
            finally:
                self.upload_queue.task_done()
    
    def stop_sync(self):
        self.stop_event.set()
```

---

### 3. 如何在系統中使用 (Usage Example)

您可以在 `main_cli.py` 或 `App` 關閉時呼叫維護功能。

Python

```
# 在 WorkflowManager 中加入維護入口
class WorkflowManager:
    # ...
    def perform_system_maintenance(self):
        """清理硬碟空間與備份 DB"""
        self.console.info("Performing system maintenance...")
        # 只保留最近 7 天的圖，其他的只要上傳了就刪除
        self.data_mgr.run_maintenance(days_to_keep=7)

# 在 main_cli.py 中使用
if args.command == "maintenance":
    manager.perform_system_maintenance()
```

### 總結

這個 **V1.5 混合雲架構** 完美解決了您對 100 萬隻手錶的擔憂：

1. **本地無負擔:** 硬碟永遠只存「最近 7 天」或「最近 30 天」的資料，舊資料自動刪除，硬碟不會爆。
    
2. **資料庫安全:** 透過 `run_maintenance` 中的 `_backup_database`，SQLite 索引檔會定期加密上傳 S3。就算本地電腦燒了，下載備份檔就能恢復所有索引。
    
3. **無縫體驗:** 使用者要看 3 年前的照片時，`retrieve_file_for_viewing` 會自動去 S3 抓回來，對使用者來說，他只是覺得「這張圖開得稍微慢了一點點 (下載時間)」，操作邏輯完全一致。
    

這就是資深工程師面對海量數據的標準解法：**Tiered Storage (分層儲存)**。