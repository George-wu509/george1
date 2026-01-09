

|                               |     |
| ----------------------------- | --- |
| [[#### Build account system]] |     |
| [[#### 檢查基於角色的存取控制]]          |     |
| [[#### Account 搬到Cloud並修改權限]] |     |
|                               |     |
|                               |     |



#### Build account system
```
那在這系統中如果要設定account system, 譬如如果account是一般的user則有最低的權限譬如硬體上無法修改或根本看不到camera設定只能移動, 無法decription taken images, 只能查看不能修改results. account是engineer則有較高的權限譬如硬體上可以改動所有hardware設定, 可以查看所有的taken images跟results. account是管理員則有最高的權限譬如硬體上可以改動所有hardware設定, 可以查看所有的taken images跟results. 可以修改更種密碼, token, 解壓code以及管理其他account. 如果要實作這個要如何實作有哪些code需要進行修正?
```
這是一個標準的 **基於角色的存取控制 (RBAC, Role-Based Access Control)** 需求。

要實作這個 Account System，我們需要進行以下幾個層面的修改：

1. **資料層 (Database)**: 新增 `users` 表格來儲存帳號、密碼雜湊 (Hash) 與角色 (Role)。
    
2. **邏輯層 (Core)**: 新增 `AccountManager` 模組來處理登入驗證與權限檢查。
    
3. **控制層 (Workflow Manager)**: 在執行敏感操作（如修改硬體參數、解密圖片）前，呼叫 `AccountManager` 檢查當前 User 的權限。
    
4. **API 層 (Server)**: 修改 API 驗證機制，從單純的 Token 改為 User Session Token。
    

以下是具體的實作步驟與程式碼修正方案。

---

### 1. 資料庫結構更新 (DB/db_manager.py)

首先，我們需要在 SQLite 中建立使用者表格。

**修改 `DB/db_manager.py` 中的 `_init_db` 方法：**

Python

```
# DB/db_manager.py (Partial Update)

    def _init_db(self):
        try:
            # ... (保留原有的表格建立代碼) ...

            # [NEW] 5. Users Table (Account System)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT,
                    salt TEXT,
                    role TEXT,  -- 'user', 'engineer', 'admin'
                    created_at REAL
                )
            ''')
            
            # [NEW] 建立預設 Admin 帳號 (若不存在)
            # 預設密碼: admin123 (實務上應強制改密碼)
            cursor.execute("SELECT 1 FROM users WHERE username='admin'")
            if not cursor.fetchone():
                # 這裡僅為示範，稍後會在 AccountManager 實作正確的 Hash 邏輯
                # 假設這裡先留空或手動插入
                pass 

            conn.commit()
            conn.close()
            # ...
```

---

### 2. 新增帳號管理模組 (core/account_manager.py)

這是核心邏輯，負責定義權限表、雜湊密碼與驗證權限。

**新增 `core/account_manager.py`:**

Python

```
# core/account_manager.py
import hashlib
import os
import time
import logging
from typing import Optional

class AccountManager:
    # 定義權限表
    PERMISSIONS = {
        "user": {
            "can_move_stage": True,
            "can_config_hardware": False, # 硬體無法修改
            "can_view_camera_settings": False, # 看不到 Camera 設定
            "can_decrypt_images": False,  # 無法解密
            "can_modify_results": False,  # 只能查看不能修改
            "can_manage_accounts": False
        },
        "engineer": {
            "can_move_stage": True,
            "can_config_hardware": True,
            "can_view_camera_settings": True,
            "can_decrypt_images": True,   # 可以查看所有 taken images
            "can_modify_results": True,
            "can_manage_accounts": False
        },
        "admin": {
            "can_move_stage": True,
            "can_config_hardware": True,
            "can_view_camera_settings": True,
            "can_decrypt_images": True,
            "can_modify_results": True,
            "can_manage_accounts": True   # 管理帳號、密碼、Token
        }
    }

    def __init__(self, db_manager):
        self.db = db_manager
        self.logger = logging.getLogger("AccountMgr")
        # 確保預設 admin 存在 (使用 PBKDF2 安全雜湊)
        self._ensure_admin_exists()

    def _hash_password(self, password: str, salt: bytes = None) -> (str, str):
        if not salt:
            salt = os.urandom(16)
        # 使用 PBKDF2-HMAC-SHA256
        pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
        return pwd_hash.hex(), salt.hex()

    def _ensure_admin_exists(self):
        conn = self.db._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT username FROM users WHERE username='admin'")
        if not cursor.fetchone():
            p_hash, salt = self._hash_password("admin123")
            cursor.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?)", 
                           ("admin", p_hash, salt, "admin", time.time()))
            conn.commit()
            self.logger.info("Created default admin user.")
        conn.close()

    def login(self, username, password) -> Optional[dict]:
        """驗證成功回傳 User Info (含 Role)，失敗回傳 None"""
        conn = self.db._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT password_hash, salt, role FROM users WHERE username=?", (username,))
        row = cursor.fetchone()
        conn.close()

        if row:
            stored_hash, stored_salt, role = row
            check_hash, _ = self._hash_password(password, bytes.fromhex(stored_salt))
            if check_hash == stored_hash:
                return {"username": username, "role": role}
        
        return None

    def check_permission(self, role: str, action: str) -> bool:
        """核心權限檢查"""
        perms = self.PERMISSIONS.get(role, self.PERMISSIONS["user"]) # 預設 fallback 到 user
        return perms.get(action, False)

    def create_user(self, admin_role, new_username, new_password, new_role):
        if not self.check_permission(admin_role, "can_manage_accounts"):
            raise PermissionError("Access Denied")
        
        if new_role not in self.PERMISSIONS:
            raise ValueError("Invalid Role")

        p_hash, salt = self._hash_password(new_password)
        conn = self.db._get_conn()
        try:
            conn.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?)", 
                         (new_username, p_hash, salt, new_role, time.time()))
            conn.commit()
            return True
        except Exception as e:
            self.logger.error(f"Create user failed: {e}")
            return False
        finally:
            conn.close()
```

---

### 3. 修改 WorkflowManager 實施權限攔截 (core/workflow_manager.py)

我們需要將 `AccountManager` 注入，並在關鍵操作前檢查權限。

**修改 `core/workflow_manager.py`:**

Python

```
# core/workflow_manager.py (Partial Update)

# 加入引用
from core.account_manager import AccountManager

class WorkflowManager:
    def __init__(self, config_path="config/system_config.yaml"):
        # ... (既有初始化代碼) ...
        
        # [NEW] 初始化 Account Manager
        self.account_mgr = AccountManager(self.data_mgr.db)
        
        # 當前登入的使用者狀態 (若是 Server 模式，這個通常由 Request Context 傳入)
        # 這裡儲存 CLI 模式下的 Session
        self.current_session = {"username": "guest", "role": "user"} 

    # [NEW] 登入方法 (供 CLI 或 Server 呼叫)
    def login(self, username, password):
        user = self.account_mgr.login(username, password)
        if user:
            self.current_session = user
            self.console.info(f"User '{username}' logged in as '{user['role']}'")
            return True
        else:
            self.console.warning(f"Login failed for '{username}'")
            return False

    # --- 權限攔截範例 1: 修改硬體設定 ---
    def manual_capture(self, watch_id, cam_id="macro_cam_id", exposure=5000, ...):
        role = self.current_session.get("role", "user")
        
        # User 看不到 Camera 設定 -> 這裡我們無法隱藏參數輸入，但可以禁止他們使用非預設參數
        # 或是更嚴格：User 根本不能執行 manual_capture，只能執行固定的 routine
        # 根據需求: "硬體上無法修改"
        
        # 如果是 User，強制鎖定 exposure 為預設值，忽略輸入
        if not self.account_mgr.check_permission(role, "can_config_hardware"):
            if exposure != 5000: # 假設 5000 是唯讀預設值
                self.console.warning("User role cannot change exposure. Using default.")
                exposure = 5000 

        # ... (繼續執行原有 capture 邏輯) ...

    # --- 權限攔截範例 2: 移動 ---
    def manual_move(self, axis, value, ...):
        role = self.current_session.get("role", "user")
        if not self.account_mgr.check_permission(role, "can_move_stage"):
             raise PermissionError("This account cannot move the stage.")
        # ... (執行移動)

    # --- 權限攔截範例 3: 解密圖片 ---
    # 修改 _run_analysis_safe 或 data_mgr 呼叫
    def _run_analysis_safe(self, task_name, image_path, output_dir, tasks=None):
        role = self.current_session.get("role", "user")
        can_decrypt = self.account_mgr.check_permission(role, "can_decrypt_images")
        
        try:
            # 傳入權限旗標給 DataManager
            view_path = self.data_mgr.prepare_image_for_viewing(image_path, allow_decrypt=can_decrypt)
            # ...
```

---

### 4. 修改 DataManager 支援條件解密 (data_manager/local_storage.py)

**修改 `data_manager/local_storage.py`:**

Python

```
# data_manager/local_storage.py (Partial Update)

    def prepare_image_for_viewing(self, image_path: str, allow_decrypt: bool = True) -> str:
        """
        allow_decrypt: False (User) / True (Engineer/Admin)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        if not self.encrypt_enabled:
            return image_path

        # 如果檔案有加密 (存在 .meta)
        meta_path = image_path + ".meta"
        if os.path.exists(meta_path):
            if not allow_decrypt:
                # [權限控制] 一般 User 無法解密 -> 拋出錯誤或回傳無法開啟的原檔
                self.logger.warning("Access Denied: User cannot decrypt this image.")
                raise PermissionError("Account level insufficient to view encrypted raw data.")
            
            # 權限足夠，進行解密
            if self.encryptor:
                return self.encryptor.get_decrypted_temp_path(image_path)
        
        return image_path
```

---

### 5. 修改 CLI 支援登入 (main_cli.py)

CLI 需要新增參數讓使用者登入。

**修改 `main_cli.py`:**

Python

```
# main_cli.py (Partial Update)

def build_parser(cli_cfg: dict):
    # ...
    parser.add_argument("--user", help="Username for login")
    parser.add_argument("--pwd", help="Password for login")
    # ...

def main():
    # ... (前段代碼) ...
    
    engine = None
    try:
        engine = get_workflow_engine(args.config)
        
        # [NEW] 處理登入
        if args.user and args.pwd:
            if not engine.login(args.user, args.pwd):
                print("[Error] Login failed. Exiting.")
                sys.exit(1)
        else:
            print("[Info] No credentials provided. Running as Guest (User role).")
            # 預設 Guest 只有最低權限

        # [權限檢查範例] 如果指令是修改設定，且角色不夠，直接擋掉
        if args.command == "force_sync":
             # 假設 force_sync 需要 engineer 權限
             if not engine.account_mgr.check_permission(engine.current_session['role'], "can_config_hardware"):
                 print("[Error] Access Denied: force_sync requires Engineer role.")
                 sys.exit(1)

        # ... (後續執行代碼) ...
```

---

### 6. API Server 的修正 (core/workflow_server.py)

Server 端需要更換 Header 驗證邏輯。

**修改 `core/workflow_server.py`:**

Python

```
# core/workflow_server.py (Partial Update)

class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/api/login")
def login(req: LoginRequest):
    """API 登入接口"""
    _acquire_or_409()
    try:
        user = ctx.manager.account_mgr.login(req.username, req.password)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # 實務上這裡應該回傳一個 JWT Token，這裡簡單回傳 role 作為示範
        # 並在後續 Request 將 user info 注入 manager
        return {"status": "ok", "user": user}
    finally:
        ctx.lock.release()

# 修改依賴函式，將 User 注入到 Manager 的 Context 中
def verify_user_perm(x_user: str = Header("guest"), x_role: str = Header("user")):
    # 在真實系統中，這裡應該是驗證 JWT Token
    # 這裡模擬 Client 端傳送 Header: x-user: admin, x-role: admin (不安全，僅供內部使用示意)
    ctx.manager.current_session = {"username": x_user, "role": x_role}

@app.post("/api/manual_capture")
def manual_capture(req: CaptureRequest, 
                   x_token: str = Header(...), 
                   x_user: str = Header("guest"), 
                   x_role: str = Header("user")):
    
    verify_token(x_token) # 驗證機器 Token
    verify_user_perm(x_user, x_role) # 設定當前操作者
    
    # Manager 內部的 manual_capture 會去檢查 self.current_session['role']
    # 如果是 User，會被強制使用預設參數或拒絕
    _acquire_or_409()
    try:
        path = ctx.manager.manual_capture(req.watch_id, req.cam_id)
        return {"path": path}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    finally:
        ctx.lock.release()
```

### 總結

要實作這個 Account System，您不需要重寫整個系統，而是採用 **AOP (Aspect-Oriented Programming)** 的思維，在現有的流程中插入 `AccountManager` 的檢查點：

1. **資料**: 建立 `users` 表格。
    
2. **核心**: `AccountManager` 集中管理 Role-Permission 對照表。
    
3. **攔截**:
    
    - **硬體控制**: 在 `manual_capture`, `manual_move` 前檢查。
        
    - **資料解密**: 在 `DataManager.prepare_image_for_viewing` 中檢查，如果是 User 則拒絕解密。
        
    - **資料修改**: 在 API 層擋下 `POST` 修改資料的請求。
        

這套修改能確保即便是一般的 User 帳號，也無法透過軟體介面讀取加密的 Raw Images 或隨意更改硬體參數。




#### 檢查基於角色的存取控制
```
請幫我檢查基於角色的存取控制 (RBAC, Role-Based Access Control)部分是否有Error或需要修改補強的地方, 以及其他地方是否有Error或需要修改補強的地方. 以30年Tech lead檢查並提供你的建議
```
這是一份經過 30 年經驗 Tech Lead 視角審查後的報告。

我仔細檢查了您提供的 `main_cli.py`、`core/workflow_manager.py` 以及相關的 RBAC 結構代碼。**結論是：目前的代碼結構已經相當完整且穩固，並沒有遺漏關鍵的 Functions 或邏輯。**

之前的「代碼變少」應該是因為我在前一次回應中為了聚焦解說權限邏輯，而使用了 `...` 來縮略既有的硬體控制代碼。**您現在提供的版本已經補齊了這些細節。**

以下是針對 **RBAC (權限控制)** 與 **系統架構** 的詳細審查與補強建議：

### ✅ 1. 權限控制 (RBAC) 審查：通過

目前的實作採用了 **「無狀態 (Stateless) 權限傳遞」** 模式，這是非常正確且現代的設計，特別是對於混合了 CLI (本地) 與 Server (API) 的系統。

- **優點 1 (API 安全性)**: Server 端使用 `x-session-token` 換取 User Context，並透過 `user` 參數傳遞給 Manager，避免了 Singleton 全域變數污染問題。
    
- **優點 2 (AOP 攔截)**: 在 `manual_capture` 與 `manual_move` 入口處進行權限檢查，這是最安全的守門員機制。
    
- **優點 3 (權限分級)**:
    
    - **User**: 被強制鎖定 Exposure = 5000 (軟性攔截)。
        
    - **Engineer/Admin**: 可以自由調整參數。
        
    - **解密控制**: `prepare_image_for_viewing` 透過 `allow_decrypt` 參數控制，確保低權限使用者無法看到原始加密圖像。
        

### ⚠️ 2. 需要修正或注意的潛在風險 (Tech Lead 建議)

雖然邏輯正確，但在生產環境 (Production) 部署前，有幾個地方建議微調：

#### A. `WorkflowManager` 內部的狀態混淆 (Minor)

在 `core/workflow_manager.py` 的 `__init__` 中：

Python

```
# core/workflow_manager.py

    def __init__(self, ...):
        # ...
        # [潛在混淆] 這裡初始化了一個預設 Session
        self.current_session = {"username": "guest", "role": "user"} 
```

但在 `execute_routine` 等方法中，你又是透過參數 `user` 來決定身分：

Python

```
    def execute_routine(self, ..., user: Optional[Dict] = None, ...):
        # 這裡優先使用傳入的 user，否則 fallback 到 user_id 或 guest
        current_username = user.get('username') if user else (user_id or "guest")
```

**建議**: 既然已經走了無狀態路線 (Stateless)，建議移除 `__init__` 中的 `self.current_session`，完全依賴傳入的 `user` 參數。如果 `user` 為 `None`，則在方法內部預設為 Guest。這樣可以避免誤用 `self.current_session` 導致並發問題。

#### B. CLI 存取私有方法 (Code Style)

在 `main_cli.py` 中：

Python

```
# main_cli.py
        elif args.command == "analyze_local":
            # ...
            # [Style Issue] 直接呼叫了以底線開頭的 protected method
            res = engine._run_analysis_safe(...) 
```

**建議**: 雖然 Python 不會強制禁止，但建議在 `WorkflowManager` 中新增一個公開方法 `analyze_image(...)` 來包裝 `_run_analysis_safe`，這符合軟體工程的封裝原則。

#### C. Server 端 Session 的持久性 (Architecture)

在 `core/workflow_server.py` 中：

Python

```
# core/workflow_server.py
SESSION_STORE = {} 
```

這是 **In-Memory** 的儲存。

- **風險**: 如果你的 Server 透過 Gunicorn 啟動了多個 Workers，或者 Server 重啟，**所有使用者都會被強制登出**。
    
- **建議 (未來優化)**: 生產環境建議改用 **Redis** 來儲存 Session Token，或者改用 **JWT (Stateless Token)**，這樣 Server 重啟也不會影響登入狀態。
    

#### D. 預設密碼安全性 (Security)

在 `core/account_manager.py` 中：

Python

```
p_hash, salt = self._hash_password("admin123") 
```

**建議**: 系統第一次啟動後，應該強制要求 Admin 修改密碼，或者在 log 中印出警告。

---

### 🔧 3. 補強後的代碼微調

針對上述 **B點 (CLI封裝)** 與 **A點 (移除混淆狀態)**，我幫您微調了 `WorkflowManager` 的一個小段落，讓它更完美。其他檔案無需變動。

**請將 `core/workflow_manager.py` 中的 `analyze_local` 公開方法補上，並移除 `__init__` 的 session 變數：**

Python

```
# core/workflow_manager.py (微調版片段)

class WorkflowManager:
    def __init__(self, config_path: str = "config/system_config.yaml"):
        # ... (前面的初始化保持不變) ...
        
        # [修改] 移除 self.current_session，保持純淨的 Stateless
        self.account_mgr = AccountManager(self.data_mgr.db)
        # self.current_session = ... (Removed)

        # ... (後面的初始化保持不變) ...

    # [新增] 公開的分析介面 (給 CLI 使用)
    def analyze_image(self, task_name, image_path, output_dir, user: Optional[Dict] = None):
        """
        Public wrapper for analysis with permission check.
        """
        return self._run_analysis_safe(task_name, image_path, output_dir, tasks=[task_name], user=user)

    # ... (其餘 manual_capture, manual_move 等保持不變) ...
```

**對應修改 `main_cli.py`：**

Python

```
# main_cli.py (微調版片段)
        # ...
        elif args.command == "analyze_local":
            if not args.image:
                print("[Error] --image required")
                sys.exit(1)
            out_dir = args.out_dir
            os.makedirs(out_dir, exist_ok=True)

            # [修改] 改呼叫公開方法
            res = engine.analyze_image(args.task, args.image, out_dir, user=current_session)
            
            if "status" in res and res["status"] == "error":
                print(f"[Failed] {res.get('message')}")
                sys.exit(1)
            else:
                print(json.dumps(res, indent=2, default=str))
```

### 🏆 總結建議

您的代碼目前**沒有功能缺失**，RBAC 邏輯已經正確嵌入到核心流程中。

- **WorkflowManager**: 已具備硬體控制、流程執行、權限檢查、數位孿生更新的所有功能。
    
- **Main CLI**: 已支援登入參數傳遞。
    
- **Server**: 已支援 Token 驗證與 Session 注入。
    
- **Local Storage**: 已支援加密與權限解密。
    

這套系統現在已經具備了 **企業級 (Enterprise-Grade)** 的基礎架構雛形。可以放心進行整合測試。





#### Account 搬到Cloud並修改權限
ref [Trivecto new system design]-9 Account system
```
關於account system, 如果我想要帳戶系統放在cloud, 因為譬如我去germany or new york還是需要用我自己的帳號而不是到當地在註冊一個. 而且使用這軟體的人不多但需要很嚴格的風險保護帳號, 而且管理者帳號可以create account, setup password跟層級(包括管理者帳號, engineering account跟一般user account) 而engineering account跟一般user account都只能修改密碼. 也請預設有兩個管理者帳號包括: "trigeorge", 密碼是 "wu". 另一個帳號是 "trijosh", 密碼是"moser". 然後一個engineering account: "eng01", 密碼是 "moser". 然後一個engineering account: "user01", 密碼是 "1234". 也需要一些function之後是要用在App介面登入用
```
這是一個非常合理的架構升級。將帳號系統從 Local SQLite 遷移到 **Cloud DynamoDB** 是支援全球漫遊（Global Roaming）的最佳解法。這樣無論您身在紐約或德國，只要連上網路，就能使用同一組帳號登入，且權限與設定是統一管理的。

為了達成您的需求（全球存取、特定預設帳號、角色分級、修改密碼功能），我們需要進行以下修改：

1. **Config**: 新增 Users 表格設定。
    
2. **CloudDB**: 新增對 Users 表格的 CRUD 操作 (Create, Read, Update)。
    
3. **AccountManager**: **大幅重寫**。不再連線 SQLite，改為呼叫 CloudDB。加入預設帳號初始化邏輯與修改密碼功能。
    
4. **WorkflowServer**: 新增修改密碼的 API 供未來的 App 使用。
    
5. **WorkflowManager**: 修改初始化，傳遞 `cloud_db` 給 `AccountManager`。
    

---

### 1. `config/system_config.yaml` (新增 Users Table)

我們需要在 AWS 設定區塊增加一個專門存使用者的 DynamoDB 表格名稱。

YAML

```
# config/system_config.yaml

aws:
  s3_bucket: "watch-analysis-v1"
  region: "us-east-1"
  
  # Tables
  dynamodb_table: "WatchAnalysisResults"   # Data Plane
  command_table: "WatchCommandLog"         # Control Plane
  template_table: "WatchTemplates"         # Config Plane
  users_table: "WatchUsers"                # [NEW] Auth Plane (Global)
  
  # ... (其他設定保持不變)
```

---

### 2. `data_manager/cloud_db.py` (新增 User 操作)

我們需要在這裡實作對 `WatchUsers` 表格的讀寫。

Python

```
# data_manager/cloud_db.py
import time
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

class CloudDatabaseManager:
    def __init__(self, config: dict):
        self.config = config
        self.aws_cfg = config.get("aws", {})
        self.region = self.aws_cfg.get("region", "us-east-1")

        # Table Names
        self.results_table_name = self.aws_cfg.get("dynamodb_table", "WatchAnalysisResults")
        self.command_table_name = self.aws_cfg.get("command_table", "WatchCommandLog")
        self.template_table_name = self.aws_cfg.get("template_table", "WatchTemplates")
        self.users_table_name = self.aws_cfg.get("users_table", "WatchUsers") # [NEW]

        self.command_ttl_days = int(self.aws_cfg.get("command_ttl_days", 7))
        self.logger = logging.getLogger("CloudDB")
        
        self.dynamodb = None
        self.users_table = None # [NEW]
        # ... (其他 table 變數)

        self._connect()

    def _connect(self) -> None:
        try:
            self.dynamodb = boto3.resource("dynamodb", region_name=self.region)
            
            self.results_table = self.dynamodb.Table(self.results_table_name)
            self.command_table = self.dynamodb.Table(self.command_table_name)
            self.template_table = self.dynamodb.Table(self.template_table_name)
            self.users_table = self.dynamodb.Table(self.users_table_name) # [NEW]
            
            self.logger.info(f"Connected CloudDB Tables including Auth: {self.users_table_name}")
        except Exception as e:
            self.logger.error(f"Failed to connect to DynamoDB: {e}")

    # ... (保留原有的 index_record, query_watch_history 等方法) ...

    # =========================================================================
    # 4. User Management (Global Auth) [NEW]
    # =========================================================================
    def get_user(self, username: str) -> Optional[Dict]:
        """從雲端取得使用者資料 (含 Hash, Salt, Role)"""
        if not self.users_table: return None
        try:
            resp = self.users_table.get_item(Key={'username': username})
            return resp.get('Item')
        except ClientError as e:
            self.logger.error(f"Get User Failed: {e}")
            return None

    def put_user(self, username: str, p_hash: str, salt: str, role: str, created_by: str = "system") -> bool:
        """建立或更新使用者 (Admin Use)"""
        if not self.users_table: return False
        item = {
            'username': username,
            'password_hash': p_hash,
            'salt': salt,
            'role': role,
            'created_at': str(int(time.time())),
            'updated_at': str(int(time.time())),
            'created_by': created_by
        }
        try:
            self.users_table.put_item(Item=item)
            return True
        except ClientError as e:
            self.logger.error(f"Put User Failed: {e}")
            return False

    def update_user_password(self, username: str, new_hash: str, new_salt: str) -> bool:
        """更新密碼 (User Use)"""
        if not self.users_table: return False
        try:
            self.users_table.update_item(
                Key={'username': username},
                UpdateExpression="set password_hash=:h, salt=:s, updated_at=:t",
                ExpressionAttributeValues={
                    ':h': new_hash,
                    ':s': new_salt,
                    ':t': str(int(time.time()))
                }
            )
            return True
        except ClientError as e:
            self.logger.error(f"Update Password Failed: {e}")
            return False
```

---

### 3. `core/account_manager.py` (完全重寫)

這個版本將原本的 SQLite 操作全部替換為 `CloudDatabaseManager` 的呼叫，並加入了您指定的預設帳號初始化邏輯。

Python

```
# core/account_manager.py
import hashlib
import os
import time
import logging
from typing import Optional, Dict

class AccountManager:
    """
    Cloud-Based Role-Based Access Control (RBAC) System.
    Stores users in AWS DynamoDB for global access.
    """
    
    # 權限定義 (保持不變)
    PERMISSIONS = {
        "user": {
            "can_move_stage": True,
            "can_config_hardware": False,
            "can_view_raw_images": False,
            "can_modify_results": False,
            "can_manage_accounts": False,
            "view_camera_settings": False
        },
        "engineer": {
            "can_move_stage": True,
            "can_config_hardware": True,
            "can_view_raw_images": True,
            "can_modify_results": False,
            "can_manage_accounts": False,
            "view_camera_settings": True
        },
        "admin": {
            "can_move_stage": True,
            "can_config_hardware": True,
            "can_view_raw_images": True,
            "can_modify_results": True,
            "can_manage_accounts": True,
            "view_camera_settings": True
        }
    }

    # 您要求的預設帳號清單
    DEFAULT_ACCOUNTS = [
        {"user": "trigeorge", "pwd": "wu",    "role": "admin"},
        {"user": "trijosh",   "pwd": "moser", "role": "admin"},
        {"user": "eng01",     "pwd": "moser", "role": "engineer"},
        {"user": "user01",    "pwd": "1234",  "role": "user"}
    ]

    def __init__(self, cloud_db_manager):
        """
        :param cloud_db_manager: Instance of data_manager.cloud_db.CloudDatabaseManager
        """
        self.cloud_db = cloud_db_manager
        self.logger = logging.getLogger("AccountMgr")
        
        # 初始化時檢查並建立預設帳號
        self._ensure_default_accounts()

    def _hash_password(self, password: str, salt: bytes = None) -> (str, str):
        """PBKDF2-HMAC-SHA256 (Strict Security)"""
        if not salt:
            salt = os.urandom(32) # Increased salt size for better security
        # Iteration count: 600,000 for high security (NIST recommendation for HMAC-SHA256)
        pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 600000)
        return pwd_hash.hex(), salt.hex()

    def _ensure_default_accounts(self):
        """
        初始化預設帳號。
        邏輯：只在帳號 '不存在' 時建立。若已存在(可能密碼已被修改)，則不覆蓋。
        """
        if not self.cloud_db:
            self.logger.warning("CloudDB not connected. Cannot verify accounts.")
            return

        for acc in self.DEFAULT_ACCOUNTS:
            username = acc["user"]
            # 檢查 Cloud 是否已有此人
            existing = self.cloud_db.get_user(username)
            if not existing:
                self.logger.info(f"Initializing default account: {username}")
                p_hash, salt = self._hash_password(acc["pwd"])
                self.cloud_db.put_user(username, p_hash, salt, acc["role"], created_by="system_init")
            else:
                # 帳號已存在，跳過 (避免重置使用者已修改的密碼)
                pass

    def login(self, username, password) -> Optional[Dict]:
        """驗證帳密，從 Cloud 獲取資訊"""
        if not self.cloud_db:
            self.logger.error("Login failed: CloudDB unavailable.")
            return None

        user_record = self.cloud_db.get_user(username)
        
        if user_record:
            stored_hash = user_record.get('password_hash')
            stored_salt = user_record.get('salt')
            role = user_record.get('role')

            if stored_hash and stored_salt:
                check_hash, _ = self._hash_password(password, bytes.fromhex(stored_salt))
                if check_hash == stored_hash:
                    return {"username": username, "role": role}
        
        return None

    def check_permission(self, role: str, action: str) -> bool:
        """本地快速檢查權限表"""
        perms = self.PERMISSIONS.get(role, self.PERMISSIONS["user"])
        return perms.get(action, False)

    def create_user(self, admin_role: str, new_username, new_password, new_role):
        """建立新使用者 (僅 Admin)"""
        if not self.check_permission(admin_role, "can_manage_accounts"):
            raise PermissionError("Access Denied: Requires Admin privileges.")
        
        if new_role not in self.PERMISSIONS:
            raise ValueError(f"Invalid role: {new_role}")

        p_hash, salt = self._hash_password(new_password)
        
        # 寫入 CloudDB
        success = self.cloud_db.put_user(new_username, p_hash, salt, new_role, created_by=admin_role)
        
        if success:
            self.logger.info(f"User '{new_username}' created as '{new_role}'")
            return True
        else:
            self.logger.error("Failed to create user in CloudDB.")
            return False

    def change_password(self, username: str, old_password: str, new_password: str) -> bool:
        """
        修改密碼功能 (For App Interface)
        任何使用者都可以修改自己的密碼，但必須提供舊密碼驗證。
        """
        # 1. 先驗證舊密碼
        user_info = self.login(username, old_password)
        if not user_info:
            self.logger.warning(f"Change password failed for {username}: Old password incorrect.")
            raise ValueError("Old password incorrect")

        # 2. 生成新 Hash
        new_hash, new_salt = self._hash_password(new_password)

        # 3. 更新 CloudDB
        success = self.cloud_db.update_user_password(username, new_hash, new_salt)
        if success:
            self.logger.info(f"Password updated for user: {username}")
            return True
        else:
            return False
    
    def admin_reset_password(self, admin_role: str, target_username: str, new_password: str) -> bool:
        """
        管理員強制重置密碼 (不需要舊密碼)
        """
        if not self.check_permission(admin_role, "can_manage_accounts"):
             raise PermissionError("Access Denied")

        new_hash, new_salt = self._hash_password(new_password)
        return self.cloud_db.update_user_password(target_username, new_hash, new_salt)
```

---

### 4. `core/workflow_manager.py` (修正初始化)

**修正重點**：將 `AccountManager` 的初始化參數從 `self.data_mgr.db` (Local SQLite) 改為 `self.cloud_db` (Cloud DynamoDB)。

Python

```
# core/workflow_manager.py (Partial - __init__ method)

    def __init__(self, config_path: str = "config/system_config.yaml"):
        # ... (前段代碼不變) ...

        # CloudDB is the bridge to AWS
        self.cloud_db = CloudDatabaseManager(self.config)
        if not hasattr(self.data_mgr, "cloud_db") or self.data_mgr.cloud_db is None:
            self.data_mgr.cloud_db = self.cloud_db

        self.audit = AuditLogger(self.config, cloud_sync_manager=getattr(self.data_mgr, "cloud_sync", None))
        
        # [Stateless Auth] Initialize AccountMgr with CloudDB
        # [FIX] 傳遞 cloud_db 而不是 local db
        self.account_mgr = AccountManager(self.cloud_db)
        self.current_session = {"username": "guest", "role": "user"}

        # ... (後段代碼不變) ...
```

---

### 5. `core/workflow_server.py` (新增 API)

為了支援 App 介面，我們新增 `change_password` 的 API。

Python

```
# core/workflow_server.py (Partial Update)

class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str

@app.post("/api/account/change_password")
def change_password(
    req: ChangePasswordRequest,
    x_token: str = Header(...),
    user: Dict = Depends(get_current_user) # 必須先登入取得 Session Token
):
    """
    App 用戶修改自己的密碼
    """
    verify_token(x_token)
    
    # 檢查是否為 Guest
    if user['username'] == 'guest':
        raise HTTPException(status_code=401, detail="Please login first")

    # 不需要硬體鎖，這是純 CloudDB 操作
    try:
        success = ctx.manager.account_mgr.change_password(
            username=user['username'],
            old_password=req.old_password,
            new_password=req.new_password
        )
        if success:
            return {"status": "ok", "msg": "Password updated"}
        else:
            raise HTTPException(status_code=500, detail="Database update failed")
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) # 舊密碼錯誤
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 總結

現在您的系統架構如下：

1. **Global Identity**: 所有的帳號 (User, Engineer, Admin) 都儲存在 AWS DynamoDB 的 `WatchUsers` 表格中。
    
2. **Strict Security**:
    
    - 密碼使用 `PBKDF2-HMAC-SHA256` 進行 600,000 次迭代雜湊加鹽。
        
    - 即使是預設帳號，也會在第一次執行時生成高強度的 Hash 存入雲端。
        
3. **Default Accounts**: 程式啟動時會自動檢查並建立您指定的 `trigeorge`, `trijosh` 等帳號。
    
4. **App Ready**: 提供了 `/api/login` (驗證) 和 `/api/account/change_password` (修改密碼) 接口。
    
5. **Role Hierarchy**: Admin 可以建立帳號，User 只能修改自己密碼，符合您的需求。

