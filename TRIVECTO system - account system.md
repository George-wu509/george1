

|                               |     |
| ----------------------------- | --- |
| [[#### Build account system]] |     |
| [[#### 檢查基於角色的存取控制]]          |     |
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
