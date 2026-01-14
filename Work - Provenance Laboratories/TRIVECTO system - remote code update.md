
|                                           |     |
| ----------------------------------------- | --- |
| [[#### 遠端更新Code兩個方案]]                     |     |
| [[#### 用方案一GitHub但卻是Private]]             |     |
| [[#### 方案二：使用 AWS Systems Manager (SSM)]] |     |
|                                           |     |



#### 遠端更新Code兩個方案
```
像這樣安裝在Germany有接上camera, hardware的local system, 我的軟體版本升級(修改了或升級某些python code), 有辦法從remote client控制對方也升級code成新版本嗎? 按照目前的系統架構? 是透過Cloud? 或者有甚麼辦法可以做得到?
```
這是一個非常實際且關鍵的維運（DevOps）問題。既然你的系統已經跨國部署（紐約 vs. 德國），**「遠端更新 (OTA, Over-The-Air Update)」** 機制是絕對必須的。你不能每次改一行 code 就買機票飛去德國。

作為 Tech Lead，我的答案是：**是的，利用目前的 AWS IoT 架構，完全可以做到遠端更新。**

你不需要額外的 VPN 或 TeamViewer，利用現有的 **AWS IoT MQTT Command Channel** 就能達成。

以下我提供兩種方案，並強烈建議採用 **方案一**（最貼合你現有架構）或 **方案二**（AWS 原生解法）。

---

### 方案一：利用現有架構 (MQTT Command + Git) - 推薦

**核心概念**： 把「軟體更新」當作一種特殊的「指令」。就像你傳送 `manual_capture` 叫相機拍照一樣，你傳送 `system_update` 叫系統去 GitHub 拉最新的 Code 並重啟。

#### 1. 架構邏輯 (Architecture Flow)

1. **Remote Client**：開發者 Push 新 Code 到 GitHub/GitLab。
    
2. **Remote Client**：透過 `WorkflowProxy` 發送 MQTT 指令：`{"action": "system_update", "branch": "main"}`。
    
3. **Local System (Germany)**：收到指令。
    
4. **Local System**：執行 Shell Script (`git pull`, `pip install`).
    
5. **Local System**：自我重啟 (Restart Service)。
    

#### 2. 需要修改的 Code (Implementation)

**Step A: 修改 `core/workflow_proxy.py` (新增更新指令)**

Python

```
    def trigger_system_update(self, branch: str = "main", wait: bool = False) -> Dict[str, Any]:
        """
        發送 OTA 更新指令給德國機台
        """
        if self.mode != "aws_iot":
            self.logger.error("System update is only supported via AWS IoT.")
            return None
            
        payload = {"branch": branch}
        return self._submit_iot_command(
            action="system_update",
            payload=payload,
            wait=wait,
            wait_timeout_sec=300 # 更新可能需要幾分鐘
        )
```

**Step B: 修改 `core/workflow_server.py` 或 `core/workflow_manager.py` (接收並執行)**

這裡有個**關鍵技巧**：如果讓 `WorkflowManager` 更新自己，可能會因為檔案被鎖定或記憶體殘留而出錯。建議寫一個獨立的小腳本 `updater.py`，或在 `WorkflowManager` 裡呼叫外部 Shell 指令。

在 `WorkflowManager` 中增加處理邏輯：

Python

```
    # 在 _perform_smart_step 或 command handler 增加分支
    def handle_system_update(self, branch="main"):
        import subprocess
        import sys
        
        self.console.info(f"!!! SYSTEM UPDATE TRIGGERED (Branch: {branch}) !!!")
        
        try:
            # 1. Git Pull
            self.console.info("Executing git pull...")
            subprocess.check_call(["git", "fetch", "origin"])
            subprocess.check_call(["git", "reset", "--hard", f"origin/{branch}"])
            
            # 2. Update Dependencies (如果 requirements.txt 有變)
            self.console.info("Updating dependencies...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            
            # 3. Restart Service
            # 假設你在 Linux 上跑，並用 systemd 管理服務 (例如 watch_system.service)
            self.console.info("Restarting service...")
            # 注意：這行執行後，目前的 Python Process 會被殺掉，連線會斷開
            subprocess.Popen(["sudo", "systemctl", "restart", "watch_system"])
            
            return {"status": "updating", "msg": "System is restarting with new code..."}
            
        except Exception as e:
            self.console.error(f"Update Failed: {e}")
            return {"status": "error", "msg": str(e)}
```

---

### 方案二：使用 AWS Systems Manager (SSM) - 企業級解法

如果你不想自己寫 `git pull` 的邏輯，或者擔心 Python 自己重啟自己會掛掉，可以使用 AWS 的標準服務。

**核心概念**： AWS 提供一個叫 **SSM Agent** 的小程式，預裝在大多數 AWS EC2 上，但也支援 **On-Premise (地端) 伺服器**。

#### 操作流程：

1. **安裝**：在德國的電腦上安裝 AWS SSM Agent 並註冊 (Activation)。
    
2. **操作**：你在紐約的 AWS Console (網頁) 上，找到這台 "Managed Instance"。
    
3. **指令**：點選 "Run Command"，選擇 `AWS-RunShellScript`。
    
4. **輸入**：
    
    Bash
    
    ```
    cd /home/operator/ImagingLibWatch
    git pull
    pip install -r requirements.txt
    sudo systemctl restart watch_system
    ```
    
5. **執行**：AWS 會幫你透過安全通道執行這些指令，並回傳 Log 給你。
    

**優點**：

- **安全**：不需要在 Python code 裡開 `sudo` 權限。
    
- **穩定**：即使你的 Python 程式掛了（Crash），SSM Agent 通常還活著，你可以遠端重啟它或進行救援。
    

---

### Tech Lead 的 30年經驗警告 (Critical Advice)

要在跨國系統做遠端更新，最大的恐懼是 **「變磚 (Bricking)」** —— 更新後系統起不來，且因為網路連線依賴於該系統，導致你再也連不進去。

為了避免這種災難，請務必實作以下 **「安全網 (Safety Net)」**：

#### 1. 雙行程架構 (Watchdog Pattern)

不要讓主程式 (`WorkflowManager`) 負責與 AWS IoT 的連線。

- **Process A (Watchdog/Updater)**：一個極簡的 Python 腳本，**只負責** 連線 AWS IoT、接收更新指令、重啟 Process B。這個腳本幾乎不更新，極度穩定。
    
- **Process B (Main App)**：你的 `WorkflowManager`，負責複雜的相機控制。
    
- **運作**：Process A 收到更新指令 -> 殺掉 Process B -> Git Pull -> 啟動 Process B。如果 Process B 啟動失敗，Process A 回報錯誤給雲端。
    

#### 2. Rollback 機制

在執行 `git pull` 之前，先記錄目前的 Commit Hash。

Python

```
current_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip()
try:
    # update...
except:
    # 如果失敗，立刻還原
    subprocess.check_call(["git", "reset", "--hard", current_hash])
```

#### 3. 設定檔保留

確保你的 `config/system_config.yaml` 和 `config/keys/` 被列在 `.gitignore` 中，或者是透過環境變數注入。否則 `git pull` 可能會覆蓋掉德國當地的特定設定（例如 S3 Bucket 是 `eu-central-1`），導致更新後連不上網。

### 總結建議

1. **短期**：採用 **方案一 (MQTT + Git Pull)**。因為你已經有現成的 `WorkflowProxy` 和 AWS IoT 通道，這是開發成本最低的路徑。
    
2. **實作細節**：
    
    - 在德國電腦上設定 `systemd` 服務來跑你的 Python 程式 (確保掛掉會自動重啟)。
        
    - 給予執行 Python 的 User 足夠的權限去執行 `systemctl restart` (透過 `visudo`)。
        
3. **長期**：當機器超過 10 台時，轉用 **AWS SSM**，統一管理 fleet updates。
    

這樣你就可以坐在紐約喝咖啡，一行指令讓德國的系統升級到最新版了。





#### 用方案一GitHub但卻是Private
```
如果我要用方案一：利用現有架構 (MQTT Command + Git) 做遠端軟體版本升級. 但這個github是公司github是private那這樣還可以嗎
```
這是一個非常好的問題。對於公司內部的 Private Repo，**絕對不能**把你的個人帳號密碼寫在 Python 程式碼裡。

作為 Tech Lead，標準且安全的做法是使用 **GitHub Deploy Keys (部署金鑰)** 搭配 **SSH** 協定。

這也是目前業界 CI/CD 和邊緣設備自動更新的標準配置。它的好處是：

1. **權限最小化**：這個 Key **只能讀取 (Read-Only)** 這一個專案，無法存取公司其他專案，也無法修改代碼。
    
2. **無需密碼**：設定好後，Python 腳本執行 `git pull` 不需要輸入任何東西，完全自動化。
    

以下是具體的實作步驟：

---

### 第一階段：在德國的機器 (Local System) 上配置 SSH

這是一次性的設定工作（Provisioning）。你需要先連進德國那台電腦（透過 TeamViewer 或 SSH）做一次設定。

#### 1. 生成專用的 SSH Key

在德國電腦的終端機 (Terminal / PowerShell) 執行：

Bash

```
# 產生一個新的 key，不要設定密碼 (Passphrase 按 Enter 跳過)
ssh-keygen -t ed25519 -C "germany_station_01" -f ~/.ssh/id_github_deploy
```

這會產生兩個檔案：

- `~/.ssh/id_github_deploy` (私鑰，**絕對保密**，留在這台電腦)
    
- `~/.ssh/id_github_deploy.pub` (公鑰，等等要給 GitHub)
    

#### 2. 設定 SSH Config

為了讓 `git` 指令知道連線 GitHub 時要用這把鑰匙，建立或修改 `~/.ssh/config` 檔案：

Plaintext

```
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_github_deploy
  IdentitiesOnly yes
```

_(Windows 用戶路徑通常在 `C:\Users\User\.ssh\`)_

---

### 第二階段：在 GitHub 上授權 (Remote Client / Tech Lead)

1. 打開公司的 GitHub Repo 頁面。
    
2. 進入 **Settings** -> **Deploy keys**。
    
3. 點擊 **Add deploy key**。
    
4. **Title**: `Germany Station 01` (清楚標示是哪台機器)。
    
5. **Key**: 把剛剛那台電腦上的 `id_github_deploy.pub` (公鑰) 內容貼上去。
    
6. **Allow write access**: **不要勾選** (只要更新代碼，不需要從機器寫回 GitHub)。
    
7. 點擊 **Add key**。
    

---

### 第三階段：切換連線方式與測試

回到德國那台電腦，把原本可能的 HTTPS 連線改成 SSH 連線。

1. **修改 Remote URL**：
    
    Bash
    
    ```
    cd /path/to/ImagingLibWatch
    # 將 https://github.com/company/repo.git 改為 SSH 格式
    git remote set-url origin git@github.com:Company/ImagingLibWatch.git
    ```
    
2. **手動測試一次 (關鍵步驟)**： 第一次連線時，SSH 會問你是否信任 GitHub 的指紋。這一步必須人工確認，否則 Python 腳本會卡死。
    
    Bash
    
    ```
    git pull
    # 出現: The authenticity of host 'github.com' can't be established...
    # 輸入: yes
    ```
    
    如果顯示 `Already up to date` 或成功拉取代碼，表示配置成功！
    

---

### 第四階段：Python 自動更新腳本 (`updater.py`)

現在，你的 Python 腳本就可以在沒有密碼的情況下更新 Private Repo 了。

這是整合了 MQTT 指令的完整更新腳本範例：

**(Controller/updater_service.py)**

Python

```
import subprocess
import time
import sys
import os
import logging
# 假設你用前面提到的 WorkflowProxy 或 boto3 來監聽 MQTT
# 這裡簡化為偽代碼邏輯

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Updater")

def perform_update(branch="main"):
    repo_dir = "/path/to/ImagingLibWatch" # 專案根目錄
    
    try:
        logger.info(f"Starting update for branch: {branch}...")
        
        # 1. 確保在正確目錄
        os.chdir(repo_dir)
        
        # 2. 獲取目前 Hash (為了失敗回滾)
        current_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        
        # 3. Git Fetch (更新索引)
        # 這裡會自動使用 SSH Key，不需要密碼
        subprocess.check_call(["git", "fetch", "origin"])
        
        # 4. Git Reset (強制覆蓋本地修改，避免 Merge Conflict 卡住)
        # 注意：這會丟棄德國本地直接改動的 code，這是好事，保持環境乾淨
        subprocess.check_call(["git", "reset", "--hard", f"origin/{branch}"])
        
        # 5. 更新依賴 (如果有新套件)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        logger.info("Code updated successfully.")
        
        # 6. 重啟主程式 (WorkflowManager)
        # 這裡假設是用 systemd 管理
        # 需確保執行此 script 的 user 有 sudo systemctl restart 的權限 (visudo NOPASSWD)
        subprocess.call(["sudo", "systemctl", "restart", "watch_system"])
        
        return True

    except Exception as e:
        logger.error(f"Update failed: {e}")
        logger.info("Rolling back...")
        # 回滾機制
        try:
            subprocess.check_call(["git", "reset", "--hard", current_hash])
        except:
            logger.critical("Rollback failed! System may be unstable.")
        return False

# 模擬接收 MQTT 指令
def on_mqtt_message(client, userdata, msg):
    payload = json.loads(msg.payload)
    if payload.get("action") == "system_update":
        branch = payload.get("branch", "main")
        perform_update(branch)

# ... MQTT 連線與監聽 loop ...
```

### 為什麼這比用 HTTPS + 帳號密碼好？

1. **安全性**：如果德國電腦被駭，駭客拿到這個 SSH Key，他**只能讀取**這個專案的程式碼，無法修改，也無法存取你公司的其他專案。
    
2. **管理方便**：如果這台機器報廢了，你只要去 GitHub Settings 把那個 Deploy Key 刪除，這台機器就立刻失去權限，完全不需要改密碼。
    
3. **穩定性**：SSH 連線比 HTTPS 帶 Token 更穩定，不會有 Token 過期的問題。
    

### 總結

即使是 Private Repo，利用 **SSH Deploy Key** 配合 **MQTT 指令**，你依然可以坐在紐約，一鍵讓德國的系統自動拉取最新的私有代碼並重啟。這是最標準且安全的做法。





#### 方案二：使用 AWS Systems Manager (SSM)
```
那如果要採用方案二：使用 AWS Systems Manager (SSM) - 企業級解法進行遠端的code升級. 如果對方user是完全沒有電腦背景的人, 要如何在一開始安裝code到對方的local system時就能有所有需要的東西, 到時候要遠端升級在user那邊完全不需要操作
```
這是一個非常經典的 **「零接觸部署 (Zero-Touch Provisioning)」** 場景。針對完全沒有技術背景的操作員（Operator），你的目標是將所有的複雜度（SSH Key, AWS 憑證, Python 環境）封裝在一個 **「一鍵安裝包」** 裡。

作為 Tech Lead，我推薦使用 **AWS Systems Manager (SSM) 的 "Hybrid Activations" (混合啟用)** 機制。這是 AWS 專門為地端伺服器（On-Premise）設計的方案。

以下是具體的實施藍圖：

---

### 核心策略：工廠模式 (The Factory Pattern)

我們將流程分為兩個階段：

1. **紐約端準備 (You)**：製作一個包含所有憑證與腳本的 `.zip` 壓縮包。
    
2. **德國端執行 (User)**：下載 -> 解壓縮 -> 雙擊 `install.bat` (或 `.sh`) -> 結束。
    

---

### 第一階段：紐約端準備 (你的工作)

你需要準備好「三把鑰匙」並放入安裝包。

#### 1. 準備 AWS SSM 門票 (Activation Code)

這是一組臨時的帳號密碼，讓德國的電腦有權限向 AWS 註冊自己，變成一台「受管實例 (Managed Instance)」。

- **操作**：
    
    1. 進入 AWS Console -> Systems Manager -> **Hybrid Activations**。
        
    2. 點擊 **Create an activation**。
        
    3. **Description**: `Germany_Station_01_Setup`.
        
    4. **IAM Role**: 建立一個新 Role (如 `SSMServiceRoleForIoT`)，賦予 `AmazonSSMManagedInstanceCore` 權限（這允許它執行遠端指令）。
        
    5. **Expiry Date**: 設定 1 天後過期（為了安全，只要夠安裝時間就好）。
        
    6. **建立後**：你會拿到 `Activation Code` 和 `Activation ID`。**這很重要，存下來**。
        

#### 2. 準備 GitHub 門票 (Deploy Key)

- 如同上一篇所述，生成一組 SSH Key (`id_rsa_germany`)，並將公鑰加入 GitHub Repo 的 Deploy Keys。
    
- 將私鑰檔案準備好。
    

#### 3. 撰寫「一鍵安裝腳本」 (`setup.sh` 或 `install.bat`)

這是給德國 User 執行的唯一檔案。它要負責安裝 Python, Git, SSM Agent 並註冊。

**(假設德國電腦是 Linux/Ubuntu，如果是 Windows 邏輯相同但指令不同)**

Bash

```
#!/bin/bash
# install.sh - 德國端一鍵安裝腳本

echo ">>> 開始安裝 ImagingLibWatch 系統..."

# 1. 參數設定 (你在紐約填好這些)
SSM_CODE="你的_ACTIVATION_CODE"
SSM_ID="你的_ACTIVATION_ID"
REGION="us-east-1"
REPO_URL="git@github.com:Company/ImagingLibWatch.git"

# 2. 安裝基礎軟體 (無需 User 介入)
echo ">>> 安裝 Python, Git, AWS SSM Agent..."
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip git curl

# 3. 安裝 AWS SSM Agent (關鍵步驟)
mkdir /tmp/ssm
curl https://s3.amazonaws.com/ec2-downloads-windows/SSMAgent/latest/debian_amd64/amazon-ssm-agent.deb -o /tmp/ssm/amazon-ssm-agent.deb
sudo dpkg -i /tmp/ssm/amazon-ssm-agent.deb

# 4. 註冊到 AWS (讓這台電腦出現在你的 Console 裡)
echo ">>> 註冊裝置到 AWS Cloud..."
sudo service amazon-ssm-agent stop
sudo amazon-ssm-agent -register -code $SSM_CODE -id $SSM_ID -region $REGION
sudo service amazon-ssm-agent start

# 5. 配置 GitHub SSH Key (自動化)
echo ">>> 配置 GitHub 權限..."
mkdir -p ~/.ssh
# 假設你的 zip 檔裡有名為 id_rsa_germany 的私鑰
cp ./id_rsa_germany ~/.ssh/id_github_deploy
chmod 600 ~/.ssh/id_github_deploy
# 自動加入 known_hosts 避免跳出詢問 Yes/No
ssh-keyscan github.com >> ~/.ssh/known_hosts

# 寫入 SSH Config
cat <<EOF > ~/.ssh/config
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_github_deploy
  IdentitiesOnly yes
EOF

# 6. 下載程式碼
echo ">>> 下載主程式..."
cd ~
git clone $REPO_URL ImagingLibWatch
cd ImagingLibWatch
pip3 install -r requirements.txt

# 7. 設定自動啟動 (Systemd)
# (省略具體寫 service 檔的步驟，概念是寫入 /etc/systemd/system/watch.service)

echo ">>> 安裝完成！系統已上線並連線至 AWS。"
```

#### 4. 打包 (Packaging)

將以下檔案打包成 `Germany_Installer.zip`：

1. `install.sh` (上面的腳本，填好了 Code)
    
2. `id_rsa_germany` (私鑰)
    
3. `requirements.txt` (可選，或是 git clone 下來也有)
    

---

### 第二階段：德國端執行 (User 的視角)

User 只需要做這件事：

1. 收到你的 USB 或下載連結。
    
2. 解壓縮。
    
3. 右鍵點擊 `install.sh` -> **「以終端機執行」** (或者你教他在 terminal 打 `./install.sh`)。
    
4. 看著螢幕跑一堆字，最後顯示「安裝完成」。
    

**結果：**

- User **不需要**知道什麼是 AWS、SSH 或 Python。
    
- 這台電腦現在已經安裝了 SSM Agent 並且**主動連線**回你的 AWS Console。
    
- 你在紐約打開 AWS Systems Manager -> Fleet Manager，就會看到一台名為 `mi-0123456789abcdef` 的新機器上線了。
    

---

### 第三階段：未來的遠端升級 (你的視角)

當你要升級軟體時，User **完全不需要做任何事**（甚至不需要在電腦前）。

1. 你在紐約，打開 AWS Console -> **Systems Manager** -> **Run Command**。
    
2. 選擇目標機器 (德國那台)。
    
3. 輸入指令 (Shell Script)：
    
    Bash
    
    ```
    cd /home/operator/ImagingLibWatch
    
    # 因為我們已經在安裝時設好了 SSH config，這裡不需要密碼
    git fetch origin
    git reset --hard origin/main
    
    # 更新依賴
    pip3 install -r requirements.txt
    
    # 重啟服務
    sudo systemctl restart watch.service
    ```
    
4. 點擊 **Run**。
    
5. 你會看到指令狀態：`Pending` -> `Success`。
    

---

### Tech Lead 的安全與優化建議

#### 1. "焚毀後不理" 的安裝包 (Burn After Reading)

- 那個 `Activation Code` 只有 24 小時效期。就算安裝包洩漏，一天後那個 Code 就失效了，駭客無法註冊新機器。
    
- SSH Key 是 Read-Only 的。
    
- 安裝完成後，可以在腳本最後一行加上 `rm id_rsa_germany` 把私鑰原始檔刪除（因為已經 copy 到 `~/.ssh` 了）。
    

#### 2. Windows 支援

如果德國接硬體的電腦是 Windows (這在工業界更常見)，流程完全一樣，只是：

- 腳本改成 PowerShell (`install.ps1`)。
    
- 下載的是 `AmazonSSMAgent.exe`。
    
- 註冊指令是 `.\amazon-ssm-agent.exe -register ...`。
    
- 使用 Windows Task Scheduler 代替 systemd。
    

#### 3. 為什麼這樣做比 TeamViewer 好？

- **TeamViewer/AnyDesk**：需要對方告訴你 ID/密碼，需要對方電腦開著且有人顧，且畫面會被佔用。
    
- **AWS SSM**：是**背景執行**。對方甚至可以在用電腦做別的事，你在背景默默更新軟體，他完全無感（除了服務重啟的一瞬間）。
    

### 總結

要讓不懂電腦的 User 配合，關鍵在於**「預先配置 (Pre-configuration)」**。 你利用 **AWS Hybrid Activations** 預先產生好「身分證」，並把「鑰匙 (SSH Key)」藏在安裝腳本裡。User 唯一做的就是執行腳本，之後這台電腦就歸你管轄了。