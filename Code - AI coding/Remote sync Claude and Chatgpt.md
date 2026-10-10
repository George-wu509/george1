
|                                                                     |     |
| ------------------------------------------------------------------- | --- |
| [[#### 一步步教我怎麼設定codex的Remote Control]]                              |     |
| [[#### Codex在遠端電腦B的repo ImagingLibWatch?]]                          |     |
| [[#### Codex windows選擇remote folder]]                               |     |
| [[#### 可以使用claude remote control的網頁教學]]                             |     |
| [[#### 'claude' is not recognized]]                                 |     |
| [[#### ANTHROPIC_API_KEY is set]]                                   |     |
| [[#### 存入 Windows Credential Manager]]                              |     |
| [[#### Windows Credential Manager + 每站獨立 key方案]]                    |     |
| [[#### Windows Credential Manager 修改code]]                          |     |
| [[#### 介紹給初學者的方式 Credential Manager 方案]]                            |     |
| [[#### Claude申請好多key]]                                              |     |
| [[#### cmd line 把 key 存進 Windows Credential Manager?]]              |     |
| [[#### Windows Credential Manager是把api key存取跟python environment有關]] |     |
| [[#### 移除ANTHROPIC_API_KEY跟OPENAI_API_KEY]]                         |     |
|                                                                     |     |

#### 一步步教我怎麼設定codex的Remote Control
```
一步步教我怎麼設定codex的Remote Control
```
Codex 遠端控制 (Remote Control) 已於 2026 年 6 月全面開放支援 Windows 系統。設定過程主要透過官方的 ChatGPT 桌面應用程式與行動版 App 進行掃碼配對。

以下是完整的設定與連線步驟：

### 階段一：事前準備

- **更新應用程式**：確保 Windows 主機上的 ChatGPT 桌面應用程式與手機端 (iOS/Android) 的 ChatGPT App 都已更新至最新版本。
    
- **帳號一致**：兩部裝置必須登入完全相同的 ChatGPT 帳號與 Workspace (工作區)。
    

### 階段二：Windows 主機端設定

1. 打開 Windows 上的 ChatGPT 桌面應用程式。
    
2. 導覽至 **Settings (設定) > Connections (連線)**。
    
3. 找到並點擊 **Control this PC (控制此電腦)** 下的 **Set up (設定)** 或 **Add (新增)**。
    
4. 同意遠端存取權限並完成身分驗證後，畫面上會顯示一組專屬的 QR Code。
    

### 階段三：手機端綁定與驗證

1. 使用手機相機掃描螢幕上的 QR Code，這會自動引導開啟手機上的 ChatGPT App。
    
2. 依照 App 畫面提示，確認授權兩端裝置連線。若帳號有安全性設定，此時會要求進行多重身分驗證 (MFA) 或通行密鑰驗證。
    
3. 綁定完成後，在手機版 App 導覽列中開啟 **Remote (遠端)** 區塊，你的 Windows 電腦就會顯示在可控制的主機清單中。
    

### 疑難排解：設定檔強制啟用 (針對選項未顯示的情況)

如果你的 Windows 應用程式沒有出現上述的 Connections 設定，可以透過手動修改 Codex 的本機設定檔來強制啟用連線功能：

1. 徹底關閉並退出 ChatGPT/Codex 桌面應用程式。
    
2. 前往使用者資料夾，使用文字編輯器打開設定檔：`C:\Users\你的使用者名稱\.codex\config.toml`。
    
3. 找到 `[features]` 區段，手動加入或修改以下兩行設定：
```
remote_connections = true
remote_control = true
```
1. （選用）為了防止應用程式在啟動時自動覆寫這兩行，你可以右鍵點擊該 `config.toml` 檔案，在內容中勾選「唯讀」。
    
2. 儲存檔案後重啟桌面應用程式，並確保你的帳號已開啟 MFA 驗證，此時功能便會啟動。
    
設定完成後，只要 Windows 電腦保持喚醒與連網狀態，你就可以隨時從手機的 Remote 介面進入主機內的專案目錄，發起新任務或接管執行中的程式，直接用語音或文字指揮電腦工作。





#### Codex在遠端電腦B的repo ImagingLibWatch?
```
我的電腦A,B裡面都有repo - ImagingLibWatch. 我在電腦B的codex windows已經設定好Control this PC完成身分驗證, 所以現在在電腦A的codex windows是可以連線到電腦B. 如果我想在電腦A的codex windows chat用在遠端電腦B的repo, 我在電腦Acodex windows chat看projects選ImagingLibWatch, 我看到在對話框上面有ImagingLibWatch, local, dev. 那我應該要從哪裡開始選work在遠端電腦B的repo ImagingLibWatch?
```
你目前在對話框上方看到的 `ImagingLibWatch, local, dev`，其中的 **`local`** 是關鍵——這代表你目前的對話依然綁定在電腦 A（本機端）的執行環境。如果你現在下達修改指令，Codex 會直接去改電腦 A 裡面的程式碼，而不是電腦 B 的。

要在電腦 A 的 Codex Windows 介面中切換並操作遠端電腦 B 的 Repo，你需要更改這個對話的「宿主機（Host）/執行環境」。請依照以下步驟操作：

### 第一步：切換執行環境 (Host)

1. 在對話框上方的標籤（Breadcrumb/Status 區塊），直接**點擊 `local` 這個字眼**，或是尋找介面左側邊欄的 **「Remote / Connections」** 圖示。
    
2. 點擊後，系統會展開一個下拉選單或連線面板，列出所有已授權且目前在線的主機。
    
3. 在清單中找到並選擇「電腦 B 的名稱」（例如 `Desktop-XXX` 或你在設定時給它取的名稱）。
    

### 第二步：重新載入遠端專案 (Project)

當你把環境切換到電腦 B 後，原本標籤上的狀態會重新整理。

1. 這時候，對話環境已經身處電腦 B 內部，但你需要告訴它去哪裡找 Repo。
    
2. 再次點擊 **Projects**（或 Open Folder / Workspace）。
    
3. 此時彈出的路徑選擇器或是專案清單，**已經是電腦 B 的檔案系統了**。
    
4. 請在電腦 B 的目錄中，選擇電腦 B 存放 `ImagingLibWatch` 的正確路徑並開啟。
    

### 第三步：確認狀態改變

選好之後，回到對話框，你應該會看到上方的狀態列變成類似： 👉 `ImagingLibWatch, [電腦 B 的名稱], dev`

看到這個狀態，就代表連線已完全確立。接下來你在這個對話框裡輸入的所有提示詞（例如：「幫我跑一次這個模組的測試」、「解釋某個檔案的邏輯」），Codex 都會將指令傳送到電腦 B 執行，讀取電腦 B 上的檔案，然後把結果與修改差異（Diff）串流回你的電腦 A 螢幕上！




#### Codex windows選擇remote folder
```
我按下local按鈕, 但我只有看到幾個選項: local, new local worktree connect codex web, cloud(不能選). 沒看到你說的電腦B. 但我在setting/connection/control other devices有看到Nitro2025(這就是電腦B), 狀態寫connected. 那我要怎麼切換
```
既然在 Settings 裡已經看到「Nitro2025」顯示 connected，代表底層的點對點連線已經成功建立。

你在對話框的 `local` 按鈕沒看到它，是因為該選單（顯示 `new local worktree...` 與 `cloud`）是用來切換「當前電腦 A 的本機分支」或「雲端沙盒」。**Codex 的設計架構不允許將一個已經在「本機 (local)」初始化的對話直接熱切換 (Hot-swap) 到另一台遠端實體機器。**

你需要開啟一個全新的「遠端工作階段 (Remote Session)」來存取電腦 B。請嘗試以下三種標準的啟動方式：

### 方法一：從設定介面直接啟動（最直覺）

1. 回到 **Settings > Connections > Control other devices**。
    
2. 找到顯示 connected 的 **Nitro2025**。
    
3. 點擊該裝置名稱，或是點擊它旁邊的 **連線圖示 / 齒輪 / ⋯ 選單**。
    
4. 選擇 **Connect**、**Open Window** 或 **Start Chat**。
    
5. 系統會彈出一個全新的 Codex 視窗，該視窗的底層環境就會直接綁定在 Nitro2025 上。
    

### 方法二：使用命令面板 (Command Palette)

這是多數開發者工具（如 VS Code、Cursor 等）共通的快捷連線方式：

1. 在 Codex Windows 介面中按下快捷鍵：**`Ctrl + Shift + P`**。
    
2. 在彈出的搜尋列輸入：**`Remote`** 或 **`Connect`**。
    
3. 在搜尋結果中尋找類似 **`Codex: Connect to Remote Device...`** 或 **`Open Remote Session`** 的選項。
    
4. 點擊後，清單會列出 **Nitro2025**，選擇它即可開啟遠端視窗。
    

### 方法三：從側邊欄 (Sidebar) 的專屬圖示啟動

1. 觀察 Codex 介面的最左側邊欄（通常有 Chat、Projects、Settings 等圖示）。
    
2. 尋找一個看起來像 **「兩台電腦連線」或「天線/訊號」的 Remote 圖示**。
    
3. 點開該面板，裡面會列出你的 Remote targets (遠端目標)。
    
4. 游標停留在 **Nitro2025** 上，點擊旁邊出現的 **「→」或「+」** 圖示來建立新連線。
    

**確認連線成功與載入 Repo：** 透過上述任一方法開啟新視窗後，對話框上方的狀態列應該會預設顯示 `[Nitro2025]` 或空狀態。此時你再去點擊 **Projects** 開啟資料夾，彈出的路徑選擇器就會是電腦 B 的硬碟，選擇裡面的 `ImagingLibWatch` 即可開始工作。




#### 可以使用claude remote control的網頁教學
```
根據這個網頁看起來是可以使用claude remote control. 請解釋如果要在電腦A的claude windows的chat可以分析或修正電腦B的repo. 那我在兩台電腦的claude windows要怎麼設定
```
根據這份官方文件，Claude 的遠端控制設計非常輕量且安全。它的核心運作邏輯是：**電腦 B（有 Repo 的主機）作為 Host 運行底層，而電腦 A（你想遠端工作的地方）只需要透過瀏覽器前往 `claude.ai/code` 就能接管對話。**

以下是根據官方文件為你整理的標準設定步驟：

### 📌 步驟一：在電腦 B（擁有 Repo 的被控端）啟動伺服器

請確保電腦 B 已經使用 `/login` 登入具備 Pro、Max、Team 或 Enterprise 資格的帳號。

**方法 1：純命令列模式（推薦，適合背景常駐）**

1. 在電腦 B 開啟終端機，切換到專案目錄：`cd 路徑/ImagingLibWatch`
    
2. 輸入啟動指令：`claude remote-control`
    
3. 首次啟動會詢問是否啟用，輸入 `y` 確定。
    
4. 接下來，電腦 B 的終端機會保持在伺服器模式運作（等待連線），畫面上會顯示一組 Session URL，你也可以按空白鍵顯示 QR Code。
    

**方法 2：從正在進行的對話中途開啟** 如果在電腦 B 的終端機、VS Code 擴充套件、或是 Claude Desktop 桌面版的 Code 標籤頁中已經在對話了，只要在對話框輸入 `/remote-control` 或 `/rc`，該對話就會立刻轉為可以被遠端接管的狀態。

### 📌 步驟二：在電腦 A（控制端）接手工作

官方設計的 Client 端非常方便，你**不需要**在電腦 A 安裝任何特殊的設定檔：

1. 在電腦 A 打開瀏覽器，前往專屬的開發者控制台：**`claude.ai/code`**。
    
2. 登入與電腦 B 完全相同的 Claude 帳號。
    
3. 在左側的 **Session list（對話清單）** 中，你就會直接看到由電腦 B 廣播出來的 `ImagingLibWatch` 專案對話。
    
4. 點擊進入該對話！
    

### 💡 在電腦 A 遠端操作的體驗

一旦在電腦 A 的網頁端連線成功：

- **無縫接軌**：你在電腦 A 網頁上打的每一句指令，都會傳回電腦 B 執行。
    
- **讀取本機環境**：官方強調，你在電腦 A 聊天框輸入 `@` 時，它會直接**自動補齊 (autocomplete) 電腦 B 本機的檔案路徑**。
    
- **斷線重連機制**：如果在電腦 A 關掉網頁或網路斷線，只要電腦 B 的終端機還沒關閉，隨時連回 `claude.ai/code` 都可以從中斷的地方繼續工作。
    

總結來說，你不需要在兩台電腦間弄什麼複雜的網域或實體 IP 同步，只要在電腦 B 下達 `claude remote-control` 讓它進入待命，接著在任何地方用瀏覽器打開 `claude.ai/code` 就能完美遙控修改 Repo！





#### 'claude' is not recognized
```
我輸入claude remote-control 顯示'claude' is not recognized as an internal or external command,

operable program or batch file.
```

會出現這個錯誤訊息，代表你的電腦 B **尚未安裝 Claude Code 命令列工具**，或者你剛安裝完但**還沒有重新啟動終端機**（導致系統抓不到 `claude` 的環境變數）。

請在電腦 B 依照以下步驟進行安裝與初始化：

### 步驟一：安裝 Claude Code

官方提供了幾種安裝方式，請開啟電腦 B 的終端機（Terminal）或命令提示字元（CMD），選擇其中一種執行：

**方法 1：使用官方 Windows PowerShell 腳本（最簡單）** 開啟 PowerShell 並貼上以下指令：

PowerShell

```
irm https://claude.ai/install.ps1 | iex
```

**方法 2：使用官方 Windows CMD 腳本** 如果你習慣用傳統的 CMD (命令提示字元)，請貼上以下指令：

DOS

```
curl -fsSL https://claude.ai/install.cmd -o install.cmd && install.cmd && del install.cmd
```

**方法 3：使用 npm（如果你是前端開發者，電腦已有 Node.js）**

DOS

```
npm install -g @anthropic-ai/claude-code
```

### 步驟二：重啟終端機並登入

1. 安裝完成後，**請務必關閉目前的終端機視窗，然後重新開一個新的**，這樣系統才會載入最新的 `PATH` 環境變數。
    
2. 在新的終端機輸入 `claude`。
    
3. 系統會提示你登入，請跟隨畫面指示在瀏覽器完成登入（注意：必須是 Pro, Max, Team 或 Enterprise 的付費帳號才能使用此功能）。
    

### 步驟三：啟動遠端伺服器

登入完成並授權後，切換到你的專案資料夾（例如 `ImagingLibWatch`），再次輸入我們剛剛討論的指令：

DOS

```
claude remote-control
```

這時它就會順利啟動，並問你是否允許開啟遠端控制（輸入 `y`）。接著，你就可以走到電腦 A，打開瀏覽器前往 `claude.ai/code` 無縫接管了！




#### ANTHROPIC_API_KEY is set
```
執行laude remote-control 得到Error: Remote Control requires claude.ai subscription auth. ANTHROPIC_API_KEY is set
```
這個錯誤是因為你的系統中已經設定了 `ANTHROPIC_API_KEY` 環境變數。當 Claude Code 偵測到這個金鑰時，會自動切換為開發者 API 計費模式；但根據官方規定，**遠端控制 (Remote Control) 功能只能透過 claude.ai 的訂閱帳號（Pro, Max, Team 或 Enterprise）登入來使用，不支援 API Key 模式。**

你需要清除這個環境變數，讓 Claude 改用網頁登入驗證。請根據你目前使用的終端機類型，選擇以下對應的解法：

**方法一：在當前終端機暫時清除（最快，不影響其他專案）** 如果你只是現在想測試，請在目前的終端機輸入以下指令清除變數：

- 如果你使用的是 **傳統 CMD (命令提示字元)**：
    
    DOS
    
    ```
    set ANTHROPIC_API_KEY=
    ```
    
- 如果你使用的是 **PowerShell**：
    
    PowerShell
    
    ```
    $env:ANTHROPIC_API_KEY=""
    ```
    

清除後，請立刻在同一個視窗重新執行 `claude remote-control`。這時系統就不會報錯了，並會引導你開啟瀏覽器完成帳號登入授權。

**方法二：從 Windows 系統永久移除（如果你不常直接用 API Key 開發）** 如果你希望以後每次打開終端機都不會遇到這個衝突：

1. 按下 **Windows鍵**，搜尋「**環境變數**」並開啟設定。
    
2. 點擊右下角的 **環境變數 (Environment Variables...)**。
    
3. 在上半部的「使用者變數」或下半部的「系統變數」清單中，尋找 `ANTHROPIC_API_KEY`。
    
4. 點選它，然後按 **刪除 (Delete)**。
    
5. 按 **確定** 關閉所有視窗。
    
6. **務必關閉目前的終端機並重新開一個**，切換回專案目錄後再執行 `claude remote-control` 即可。




#### 存入 Windows Credential Manager
```
或者有沒有其他更簡單的方案可以保存或使用App使用的ANTHROPIC_API_KEY? 有沒有優缺點?
```
以目前 App 是 Windows/PySide6 桌面程式、會部署到多個站點來看，我最建議的是：

> 每個站點使用獨立 Anthropic API key，存入 Windows Credential Manager；Anthropic 端以 Workspace 做站點分組、費用上限與撤銷管理。

這是「改動小、安全性有實質提升、維運難度合理」的折衷。

## 各方案比較

|方案|安全性|改動量|多站點管理|主要缺點|
|---|---|---|---|---|
|Windows Credential Manager + 每站獨立 key|中高|小|中高|被完全控制的 station 仍可取得 key|
|DPAPI 加密檔案|中|小至中|中|綁定 Windows user/machine，備份與重灌較麻煩|
|AWS Secrets Manager 由 App 直接讀取|中高|中|高|Station 還需要 AWS 身分；攻陷 station 仍能讀取 key|
|Windows environment variable|中低|幾乎零|低|同帳號程式與 child process 容易讀取|
|`.env`／YAML 檔案|低|零|低|容易複製、備份或誤 commit|
|LLM Gateway|最高|大|最高|需要額外服務與維運|

## 建議方案：Credential Manager + 每站獨立 key

### Anthropic 端

建議建立：

```
Organization
├── Workspace: Site-01
│   ├── Key: site01-station01
│   └── Key: site01-station02
└── Workspace: Site-02
    ├── Key: site02-station01
    └── Key: site02-station02
```

每個站點使用自己的 API key，不要讓全部 station 共用一把。

Anthropic Workspace 可以：

- 個別建立、停用或刪除 API key
- 設 Workspace spend limit
- 設 rate limit
- 設費用通知
- 分別查看各 Workspace 的 usage/cost

這代表某一台 station 的 key 外洩時，可以只停用該 key；若每個 site 使用不同 Workspace，也能限制單一 site 的最大損失。[Anthropic Workspace 管理文件](https://support.claude.com/en/articles/9796807-creating-and-managing-workspaces-in-the-claude-console)

### Windows 站點端

將 key 存成：

```
Credential target: Moonlight/Anthropic/site01-station01
Username: site01-station01
Password: sk-ant-...
```

App 啟動時使用 Windows `CredRead` 讀取，設定工具使用 `CredWrite` 寫入。

Microsoft 建議桌面程式優先使用 Windows Credential Manager；它會以目前 Windows 登入使用者的憑證保護資料。若需要自行保存加密內容，第二選擇才是 DPAPI。[Microsoft Windows credential guidance](https://learn.microsoft.com/en-us/windows/win32/secbp/handling-passwords)

### 優點

- Key 不會出現在 Git、YAML、`.env` 或 registry environment。
- 不會被一般 `Get-ChildItem Env:` 或 child process 自動讀到。
- 不需要部署另一個 server。
- 可以獨立撤銷一台 station。
- 現有 LLM 呼叫流程只需替換 `resolve_api_key()`。
- 不影響 OCR、image analysis 或 Claude SDK 的使用方法。

### 缺點

- Windows 管理員、惡意程式、注入 App process 的程式，最終仍可能取得 key。
- Credential 是綁定 Windows 帳號的；必須在「實際執行 App 的帳號」下 provision。
- Windows 重灌、更換 service account 時需要重新 provision。
- Key rotation 要更新各站 Credential Manager。

桌面 App 無法百分之百隱藏一個可重複使用的 API key。只要 App 能直接呼叫 Anthropic，App process 就必須在某個時間點取得 plaintext key。Credential Manager 能防止意外外洩與靜態檔案竊取，但無法防止 station 已完全被控制。只有 Gateway 能真正讓 vendor key 完全不進入 station。

## 對目前 repo 的改動量

這個方案改動很小，預估約 2–4 個檔案：

- 新增 Windows Credential Manager wrapper
- 修改 `resolve_api_key()`：
    1. Production 優先讀 Credential Manager
    2. Development 才允許讀環境變數
    3. 不再允許 YAML 內直接放 `api_key`
- 修改 environment UI：
    - 移除 production API key environment 設定
    - 加入「設定／更新／刪除 Anthropic credential」
    - 不顯示現有 key
- 加入測試與 deployment 文件

約 1–2 個工程日，不需要更動 image analysis pipeline。

## 次佳方案：AWS Secrets Manager 直接讀取

App 啟動時：

```
Station App
   │ AWS temporary identity
   ▼
AWS Secrets Manager
   │ GetSecretValue
   ▼
ANTHROPIC_API_KEY
```

優點：

- 中央管理、輪替、CloudTrail audit。
- Key 不在 repo 或 station config。
- 多站部署比手動更新 Credential Manager方便。
- 可透過 IAM 控制哪些 station 能讀取。

AWS 建議 secret 使用 KMS 加密、最小權限、rotation、監控與私有網路存取。[AWS Secrets Manager best practices](https://docs.aws.amazon.com/secretsmanager/latest/userguide/best-practices.html)

缺點：

- Station 必須先有 AWS authentication。
- 如果用 static AWS access key 來取得 Anthropic key，只是把一個 secret 換成另一個 secret。
- 最好使用 AWS IoT certificate 換取短效 AWS credentials，設定複雜度會提高。
- 被攻陷的 station 仍可執行 `GetSecretValue`。
- App 啟動依賴 AWS 網路；需要安全快取與失敗策略。

所以 Secrets Manager 主要改善的是集中維運與 audit，不等於 vendor key 不會抵達 station。

## 不建議當正式方案

### Persistent environment variable

目前 repo 的 UI 會將 `ANTHROPIC_API_KEY` 寫入 `HKCU\Environment`。

優點是完全不必改 LLM code，而且 Anthropic 一般也建議 API key 不要寫死、可透過 environment 注入。[Anthropic API key best practices](https://support.claude.com/en/articles/9767949-api-key-best-practices-keeping-your-keys-safe-and-secure)

但 Windows persistent environment variable 並不是 secret vault：

- 同一使用者的其他程式可以讀取。
- 會傳給 child processes。
- 可能進入診斷資訊或 support script。
- 所有 station 若共用同一 key，外洩影響很大。

可以暫時使用，但至少要做到「每站不同 key + Workspace spend limit」。

### `.env` 或 YAML

只適合開發環境。即使 `.gitignore` 已排除，仍可能經由備份、壓縮檔、客服資料或錯誤打包外洩。

### 把 key 加密後連同解密 key 放進 App

沒有實質安全性。如果 executable 內同時包含密文與解密方式，攻擊者最終仍能還原；程式碼混淆也只能增加少量分析成本。

## 我的建議優先順序

現階段採用：

1. Anthropic 每個 site 建立獨立 Workspace。
2. 每一台 station 建立獨立 API key。
3. 設定 Workspace spend limit 與通知。
4. Key 存 Windows Credential Manager。
5. App 只在 development mode 才允許 environment variable fallback。
6. 每 60–90 天輪替，或在人員／設備異動時立即輪替。
7. Log 只能記錄 station ID、Workspace、model、request ID，不能記錄 key。
8. 未來站點數量或安全要求增加，再升級成 Gateway。

這個方案不能達到 Gateway 的最高安全等級，但以目前架構來說，投入成本和安全收益的比例最好。此次我沒有修改 repo。




#### Windows Credential Manager + 每站獨立 key方案
```
如果App採用Windows Credential Manager + 每站獨立 key方案, 那如果我這台開發的電腦也用這個, 那我還可以額外使用我的claude code用remote control嗎? 因為要用到ANTHROPIC_API_KEY. 或者我使用remote control可以使用其他方法避開使用ANTHROPIC_API_KEY?
```
可以，而且這正是建議的分離方式：

- App：從 Windows Credential Manager 讀取專屬的 `ANTHROPIC_API_KEY`
- Claude Code Remote Control：使用 Claude.ai 帳號／訂閱的 OAuth 登入
- Windows 全域環境變數：不要設定 `ANTHROPIC_API_KEY`

兩者可以在同一台開發電腦、同一個 Windows 帳號下共存，不會互相衝突。

## 建議架構

```
Windows Credential Manager
└── Moonlight App credential
    └── App 專用 Anthropic API key
          │
          ▼
       Moonlight App

Claude Code credential
└── Claude.ai OAuth login
          │
          ▼
Claude Code + Remote Control
```

Claude Code 不會自動搜尋 App 自訂的 Credential Manager 項目；只要 Credential Target 使用獨立名稱，例如：

```
Moonlight/Anthropic/development
```

App 可以讀到它，但 Claude Code 不會把它當成自己的認證。

## Claude Code 不需要 `ANTHROPIC_API_KEY`

如果你有以下任一方案：

- Claude Pro
- Claude Max
- Claude Team
- Claude Enterprise

Claude Code 可以直接使用 Claude.ai 帳號登入，而不是 API key：

```
claude
/login
```

登入完成後就能使用：

```
/remote-control
```

或直接從終端啟動：

```
claude remote-control
```

Remote Control 官方支援 Pro、Max、Team 和 Enterprise；Team／Enterprise 管理員也可以在組織設定中控制是否開放。[Claude Code Remote Control 說明](https://support.claude.com/en/articles/14554000-claude-code-power-user-tips)

## 最重要的環境變數衝突

Claude Code 的認證優先順序是：

```
ANTHROPIC_API_KEY environment variable
            ↓ 優先於
Claude.ai subscription / OAuth login
```

所以如果 Windows 還有全域 `ANTHROPIC_API_KEY`，即使你已用 `/login` 登入，Claude Code 仍可能使用該 API key，並產生 API pay-as-you-go 費用。[Anthropic 認證優先順序](https://support.claude.com/en/articles/12304248-manage-api-key-environment-variables-in-claude-code)

因此 App 改用 Credential Manager 後，應移除全域環境變數。

### 不顯示 key，只檢查是否存在

PowerShell：

```
if ($env:ANTHROPIC_API_KEY) {
    "ANTHROPIC_API_KEY is set"
} else {
    "ANTHROPIC_API_KEY is not set"
}
```

檢查永久 User variable：

```
if ([Environment]::GetEnvironmentVariable("ANTHROPIC_API_KEY", "User")) {
    "User variable is set"
} else {
    "User variable is not set"
}
```

### 移除目前終端的變數

```
Remove-Item Env:ANTHROPIC_API_KEY -ErrorAction SilentlyContinue
```

### 移除永久 User environment variable

```
[Environment]::SetEnvironmentVariable(
    "ANTHROPIC_API_KEY",
    $null,
    "User"
)
```

如果 Machine environment 也曾設定，需要管理員 PowerShell：

```
[Environment]::SetEnvironmentVariable(
    "ANTHROPIC_API_KEY",
    $null,
    "Machine"
)
```

之後關閉並重新開啟終端。

這不會刪除 Windows Credential Manager 裡的 App key，所以 App 仍可使用 Claude API。

## Claude Code 設定步驟

1. 確保沒有全域 `ANTHROPIC_API_KEY`。
    
2. 更新 Claude Code：
    
    ```
    claude update
    ```
    
3. 啟動 Claude Code：
    
    ```
    claude
    ```
    
4. 執行：
    
    ```
    /login
    ```
    
5. 選擇 Claude.ai 的 Pro、Max、Team 或 Enterprise 帳號。
    
6. 執行：
    
    ```
    /status
    ```
    
7. 確認顯示的是 Claude.ai subscription／登入帳號，而不是 Console API key。
    
8. 啟用：
    
    ```
    /remote-control
    ```
    

Anthropic 官方也建議，如果想使用訂閱額度，應保持 `ANTHROPIC_API_KEY` 未設定，並透過 `/status` 確認目前認證來源。[Claude Pro/Max 與 Claude Code](https://support.claude.com/en/articles/11145838-use-claude-code-with-your-pro-or-max-plan)

## 如果沒有 Claude 訂閱

Claude Code 本身可以使用：

- Anthropic Console API key
- Amazon Bedrock
- Google Vertex AI

但 Remote Control 官方目前主要提供給 Claude.ai 的 Pro、Max、Team、Enterprise 帳號。Bedrock／Vertex 適合一般 Claude Code API inference，不建議把它們當成避開 Remote Control 登入要求的方法。

如果只有 Console API key，最乾淨的作法是：

- App 使用 App 專屬 key，存在 Credential Manager
    
- Claude Code 使用另一把開發者專屬 key
    
- 僅在啟動 Claude Code 的單一 PowerShell session 暫時設定：
    
    ```
    $env:ANTHROPIC_API_KEY = "developer-key"
    claude
    Remove-Item Env:ANTHROPIC_API_KEY
    ```
    

但這仍是 API 計費，而且 Remote Control 功能可能取決於訂閱資格。

## 最推薦的開發機配置

|用途|認證方式|
|---|---|
|Moonlight App|App 專屬 API key，Windows Credential Manager|
|Claude Code|Claude.ai `/login` OAuth|
|Claude Code Remote Control|同一個 Claude.ai subscription|
|`ANTHROPIC_API_KEY` User/Machine environment|不設定|
|Production station|每站獨立 API key，Credential Manager|
|Anthropic Workspace|開發、測試、各 production site 分開|

這樣 App API 費用和 Claude Code 使用量也會自然分開，最不容易出現誤用或計費混淆。此次沒有修改 repo。





#### Windows Credential Manager 修改code
```
了解, 那請幫我code修改成Windows Credential Manager + 每站獨立 key方案, 並解釋除了code修改之後, 我要做甚麼設定改變
```
已完成 Windows Credential Manager + 每站獨立 key 改造。App 現在會優先且預設只從 Credential Manager 讀取 Anthropic/OpenAI key，不再使用全域環境變數或 YAML plaintext key。

目前尚未放入任何真實 API key，因此在完成下列 provisioning 前，App 的 Claude/OpenAI 功能會顯示 credential missing。

## 程式修改內容

- LLM credential resolution 改為 Credential Manager：[LLM_models.py (line 80)](D:/Provenance Laboratories projects/ImagingLibWatch/algorithms/LLM_models.py:80)
    
- 新增 Win32 Credential Manager wrapper：[windows_credentials.py (line 65)](D:/Provenance Laboratories projects/ImagingLibWatch/core/windows_credentials.py:65)
    
- Anthropic Credential Target：
    
    ```
    Moonlight/LLM/Anthropic
    ```
    
- OpenAI Credential Target：
    
    ```
    Moonlight/LLM/OpenAI
    ```
    
    設定位置：[LLM_config.yaml (line 24)](D:/Provenance Laboratories projects/ImagingLibWatch/config/LLM_config.yaml:24)
    
- 新增安全 provisioning CLI，不接受 command-line key 值：[manage_llm_credentials.py (line 105)](D:/Provenance Laboratories projects/ImagingLibWatch/tasks/cli_wrappers/manage_llm_credentials.py:105)
    
- Environment UI 不再提供 Anthropic/OpenAI key 欄位，避免重新寫回 `HKCU\Environment`：[update_env_ui.py (line 234)](D:/Provenance Laboratories projects/ImagingLibWatch/config/update_env_ui.py:234)
    
- 新增完整部署與 rotation 文件：[5.54_llm_credentials.md](D:/Provenance Laboratories projects/ImagingLibWatch/helper/docs/05_configuration_architecture/5.54_llm_credentials.md)
    

相容性方面：

- 有 `credential_target` 的 OpenAI／Claude：只讀 Credential Manager。
- Environment fallback 與 YAML plaintext fallback 都預設關閉。
- 舊的自訂 LLM service 若沒有 `credential_target`，仍保留原本 environment → YAML 行為。
- OCR、image analysis、task config 與輸出介面沒有改變。

## 你的開發電腦現在要做什麼

### 1. 在 Anthropic Console 建立開發機專用 key

建議命名：

```
development-laptop-NITRO2025
```

不要沿用 production station key。若可行，放在獨立 Development Workspace，並設定 spend limit 與通知。

### 2. 寫入 Windows Credential Manager

請在「平常執行 App 的同一個 Windows 帳號」開啟 PowerShell：

```
conda activate watch_env
cd "D:\Provenance Laboratories projects\ImagingLibWatch"

python tasks\cli_wrappers\manage_llm_credentials.py set anthropic
```

程式會以隱藏輸入提示：

```
New claude API key:
Confirm API key:
```

Key 不會出現在 command line、PowerShell history 或輸出中。

確認：

```
python tasks\cli_wrappers\manage_llm_credentials.py status anthropic
```

應顯示：

```
claude: present (target: Moonlight/LLM/Anthropic)
```

如果 App 也使用 OpenAI：

```
python tasks\cli_wrappers\manage_llm_credentials.py set openai
python tasks\cli_wrappers\manage_llm_credentials.py status openai
```

### 3. 重新啟動 App 並執行一個受控分析

Credential Manager 不需要重啟 Windows，但已經在執行中的 App 要完全關閉再啟動。

## Claude Code Remote Control 設定

我檢查到目前狀態：

- Process-level `ANTHROPIC_API_KEY`：存在
- User-level 永久變數：不存在
- Machine-level 永久變數：不存在

App 已確認會忽略 process environment 中的 `ANTHROPIC_API_KEY`，所以不會誤用 Claude Code 的 key。

但 Claude Code 本身仍會優先使用目前 terminal 的環境變數。啟動 Claude Code 前，在該 PowerShell 執行：

```
Remove-Item Env:ANTHROPIC_API_KEY -ErrorAction SilentlyContinue
```

接著：

```
claude
```

在 Claude Code 裡：

```
/login
/status
/remote-control
```

`/status` 應顯示 Claude.ai subscription／OAuth，而不是 Console API key。

刪除 environment variable 不會刪除：

```
Moonlight/LLM/Anthropic
```

所以 App 與 Claude Code 可以完全分開：

|程式|認證來源|
|---|---|
|Moonlight App|Windows Credential Manager 的 App 專屬 key|
|Claude Code|Claude.ai `/login` OAuth|
|Remote Control|Claude.ai subscription|
|全域 `ANTHROPIC_API_KEY`|不需要|

## 每個 production station 要做什麼

### 1. 每站建立不同 API key

例如：

```
site01-station01
site01-station02
site02-station01
```

建議一個 site 一個 Anthropic Workspace、一台 station 一把 key。Workspace 設 spend limit 與通知。

### 2. 更新 station 外部設定

既有 production station 使用：

```
D:\Moonlight\Protected\config\LLM_config.yaml
```

要確認 Claude service 至少包含：

```
api_key: ""
api_key_env: "ANTHROPIC_API_KEY"
credential_target: "Moonlight/LLM/Anthropic"
allow_env_fallback: false
allow_plaintext_api_key: false
```

只更新 EXE 不保證既有外部 YAML 被覆寫。不要隨意使用 `--overwrite-config`，因為它可能連其他 station-specific YAML、硬體與校正設定一起覆寫；既有站點建議只合併上述 LLM 欄位。

### 3. 使用執行 App 的帳號 provision

對 packaged App：

```
& "D:\Moonlight\Production\MoonlightApp.exe" `
  --moonlight-script tasks/cli_wrappers/manage_llm_credentials.py `
  set anthropic
```

檢查：

```
& "D:\Moonlight\Production\MoonlightApp.exe" `
  --moonlight-script tasks/cli_wrappers/manage_llm_credentials.py `
  status anthropic
```

如果 App 是由專用 Windows service account 執行，必須以該帳號 provision。其他管理員帳號存入的 Credential Manager 項目，App service account 看不到。

### 4. 驗證後移除舊環境變數

先確認 App 能完成一次 Claude image analysis，再移除舊變數：

```
[Environment]::SetEnvironmentVariable(
    "ANTHROPIC_API_KEY",
    $null,
    "User"
)
```

若 Machine scope 也曾設定，以管理員身分執行：

```
[Environment]::SetEnvironmentVariable(
    "ANTHROPIC_API_KEY",
    $null,
    "Machine"
)
```

然後重啟 App。

## Key rotation 流程

1. Anthropic Console 建立同一 station 的新 key。
2. 在 station 再執行 `set anthropic`，它會取代舊 Credential Manager 值。
3. 重啟 App。
4. 執行一次受控分析。
5. 成功後才在 Anthropic Console 撤銷舊 key。

退役 station 時，除了撤銷 Anthropic key，也可清除本機 credential：



#### 介紹給初學者的方式 Credential Manager 方案
```
請用介紹給初學者的方式詳細一步步解釋 "你的開發電腦現在要做什麼"
```
下面以「你在這台開發電腦從 source 執行 App」為主要情境，一步一步操作。

完成後會是：

```
Moonlight App
    └─ 從 Windows Credential Manager 取得 App 專用 API key

Claude Code
    └─ 使用 Claude.ai /login 登入
       不使用 App 的 API key
```

兩者互不影響。

---

## 第 0 步：先確認你如何執行 App

有兩種情況。

### 情況 A：從 repo 執行

例如你平常在這個目錄開發：

```
D:\Provenance Laboratories projects\ImagingLibWatch
```

然後使用 Python、IDE 或 `App_run.py` 啟動。

這種情況使用 repo 裡的：

```
config\LLM_config.yaml
```

目前程式修改已經完成，不需要再手動改 YAML。

### 情況 B：執行安裝好的 EXE

例如：

```
D:\Moonlight\Production\MoonlightApp.exe
```

這種情況使用外部設定：

```
D:\Moonlight\Protected\config\LLM_config.yaml
```

除了設定 Credential Manager，還要確認這個外部 YAML 也已更新。本文後面有說明。

---

# 第一部分：在 Anthropic 建立這台電腦專用的 API key

## 第 1 步：登入 Claude Console

開啟 Claude Console：

[Claude Console](https://console.anthropic.com/)

注意，這是 API 管理 Console，不是一般聊天用的 `claude.ai`。

兩者用途不同：

|系統|用途|
|---|---|
|Claude Console|建立 API key、管理 API 費用|
|claude.ai|Claude 聊天、Claude Code subscription、Remote Control|

## 第 2 步：建立開發用 Workspace

登入後進入：

```
Settings → Workspaces
```

按：

```
Add Workspace
```

建議名稱：

```
Moonlight-Development
```

如果你沒有建立 Workspace 的權限，可以暫時使用 `Default Workspace`。只有 Organization Admin 能建立新 Workspace。[Anthropic Workspace 官方說明](https://support.claude.com/en/articles/9796807-creating-and-managing-workspaces-in-the-claude-console)

## 第 3 步：設定開發 Workspace 費用上限

進入剛建立的 Workspace，選擇：

```
Limits
```

建議至少設定：

- Spend limit
- Email notification
- Rate limit

初期可以設定較小額度，等確認正常使用後再提高。

目的不是限制 App 功能，而是萬一 key 外洩，不會無限制產生費用。

## 第 4 步：建立這台電腦專用的 key

在 Workspace 裡選擇：

```
API Keys → Create Key
```

建議命名：

```
development-laptop-NITRO2025
```

或：

```
moonlight-dev-a3146
```

不要使用太模糊的名字，例如：

```
test
key1
new-key
```

未來看到帳單或要停用 key 時，清楚的名稱會很重要。

## 第 5 步：複製 API key

建立完成後，Anthropic 會顯示 key。

通常外觀類似：

```
sk-ant-...
```

此時：

1. 按 Copy。
2. 不要貼到記事本。
3. 不要放進 YAML。
4. 不要放進 `.env`。
5. 不要貼到聊天、Slack、email 或 ticket。
6. 保持瀏覽器頁面開著，接著進行下一部分。

API key 通常只會完整顯示一次。如果遺失，不需要設法找回；建立新 key 並撤銷舊 key即可。

---

# 第二部分：把 key 存進 Windows Credential Manager

## 第 6 步：完全關閉目前正在執行的 Moonlight App

如果 App、OCR worker 或其他 Moonlight process 正在執行，先關閉。

Credential 寫入後雖然不需要重開 Windows，但已啟動的程式最好全部重啟。

## 第 7 步：開啟正確的 PowerShell

非常重要：

> 必須使用「平常執行 Moonlight App 的同一個 Windows 帳號」。

先開啟 PowerShell，執行：

```
whoami
```

你會看到類似：

```
computer-name\a3146
```

記住這個帳號。

Windows Credential Manager 的資料跟 Windows 帳號綁定。如果用管理員 A 寫入，但用操作員 B 執行 App，操作員 B 讀不到。

## 第 8 步：進入 repo

在 PowerShell 執行：

```
cd "D:\Provenance Laboratories projects\ImagingLibWatch"
```

確認目前位置：

```
Get-Location
```

應看到：

```
D:\Provenance Laboratories projects\ImagingLibWatch
```

## 第 9 步：啟用 Python 環境

執行：

```
conda activate watch_env
```

如果 PowerShell 顯示找不到 `conda`，可以改用：

- Anaconda Prompt
- Miniconda Prompt
- 已初始化 Conda 的 PowerShell

確認 Python 正常：

```
python --version
```

不需要特定輸出，只要不是錯誤即可。

## 第 10 步：先檢查 Credential 是否已存在

執行：

```
python -m tasks.cli_wrappers.manage_llm_credentials status anthropic
```

第一次通常會看到：

```
claude: not set (target: Moonlight/LLM/Anthropic)
```

這是正常的，代表目前還沒有存入 key。

## 第 11 步：寫入 Anthropic key

執行：

```
python -m tasks.cli_wrappers.manage_llm_credentials set anthropic
```

畫面會出現：

```
New claude API key:
```

現在貼上剛才從 Anthropic Console 複製的 key。

### 貼上後畫面沒有出現任何文字是正常的

這是安全輸入，PowerShell 不會顯示：

- 字母
- 星號
- 圓點
- key 長度

貼上後直接按 Enter。

接著會出現：

```
Confirm API key:
```

再貼上同一個 key，按 Enter。

成功時應看到：

```
Stored claude credential in Windows Credential Manager target: Moonlight/LLM/Anthropic
Restart MoonlightApp before testing the provider.
```

### 常見問題：兩次輸入不一致

如果看到：

```
API key confirmation does not match
```

重新執行：

```
python -m tasks.cli_wrappers.manage_llm_credentials set anthropic
```

再貼兩次。

## 第 12 步：清除 Windows clipboard

避免 API key 一直留在剪貼簿：

```
Set-Clipboard -Value ""
```

## 第 13 步：確認 Credential 存在

執行：

```
python -m tasks.cli_wrappers.manage_llm_credentials status anthropic
```

應看到：

```
claude: present (target: Moonlight/LLM/Anthropic)
```

它只會顯示 `present`，不會顯示 API key。

## 第 14 步：從 Windows UI 確認，可選

在 Windows 搜尋：

```
Credential Manager
```

接著開啟：

```
Credential Manager
→ Windows Credentials
→ Generic Credentials
```

應能找到：

```
Moonlight/LLM/Anthropic
```

不要按顯示密碼，也不需要手動編輯。

---

# 第三部分：啟動 App 並測試

## 第 15 步：啟動 App

使用你平常的方式啟動，例如：

```
python App_run.py
```

或者透過 IDE 啟動。

重要的是：

- 使用相同 Windows 帳號
- 不要使用另一個 administrator account
- 如果用 Windows service，service 也必須使用已 provision 的帳號

## 第 16 步：執行一個 Claude image analysis

選擇一個明確設定為：

```
service: "claude"
```

的 OCR 或 image-analysis 功能進行測試。

如果成功，代表：

```
App
→ Windows Credential Manager
→ Moonlight/LLM/Anthropic
→ Anthropic API
```

整條路徑正常。

## 如果出現 credential missing

錯誤可能類似：

```
Claude API credential is missing.
Provision Windows Credential Manager target
'Moonlight/LLM/Anthropic'
```

依序檢查：

### 檢查 1：Credential status

```
python -m tasks.cli_wrappers.manage_llm_credentials status anthropic
```

必須是：

```
present
```

### 檢查 2：Windows 帳號是否相同

在設定 Credential 的 PowerShell 執行：

```
whoami
```

再確認 App 是不是由同一帳號啟動。

### 檢查 3：App 是否真的使用新的 LLM config

從 source 執行時，使用：

```
D:\Provenance Laboratories projects\ImagingLibWatch\config\LLM_config.yaml
```

裡面 Claude 應包含：

```
claude:
  provider: "anthropic"
  api_key: ""
  api_key_env: "ANTHROPIC_API_KEY"
  credential_target: "Moonlight/LLM/Anthropic"
  allow_env_fallback: false
  allow_plaintext_api_key: false
```

### 檢查 4：是否執行 installed EXE

如果你執行的是：

```
D:\Moonlight\Production\MoonlightApp.exe
```

它會讀：

```
D:\Moonlight\Protected\config\LLM_config.yaml
```

而不是 repo 裡的 config。

你需要把上述三個新欄位合併到 external config：

```
credential_target: "Moonlight/LLM/Anthropic"
allow_env_fallback: false
allow_plaintext_api_key: false
```

---

# 第四部分：設定 Claude Code 與 Remote Control

App key 已經安全放進 Credential Manager。現在要確保 Claude Code 不使用 `ANTHROPIC_API_KEY` environment variable。

## 第 17 步：開一個新的 PowerShell

建議不要沿用剛才啟動 App 的終端。

開一個全新的 PowerShell。

## 第 18 步：不顯示 key，只檢查目前 terminal 是否有環境變數

執行：

```
if ($env:ANTHROPIC_API_KEY) {
    "ANTHROPIC_API_KEY is set in this terminal"
} else {
    "ANTHROPIC_API_KEY is not set"
}
```

目前我檢查到：

- User-level 永久變數：未設定
- Machine-level 永久變數：未設定
- 目前 Codex process：有暫時的 process-level 值

因此一般新開的獨立 PowerShell 很可能不會有，但仍建議檢查。

## 第 19 步：如果 terminal 中存在，移除它

```
Remove-Item Env:ANTHROPIC_API_KEY -ErrorAction SilentlyContinue
```

再檢查一次：

```
if ($env:ANTHROPIC_API_KEY) {
    "Still set"
} else {
    "Not set — ready for Claude Code login"
}
```

這個指令只移除目前 terminal 的 environment variable，不會刪除：

```
Moonlight/LLM/Anthropic
```

所以不會影響 App。

## 第 20 步：確認沒有永久 User/Machine environment variable

不要輸出 key，只檢查有沒有：

```
[bool][Environment]::GetEnvironmentVariable(
    "ANTHROPIC_API_KEY",
    "User"
)
```

預期：

```
False
```

再檢查 Machine：

```
[bool][Environment]::GetEnvironmentVariable(
    "ANTHROPIC_API_KEY",
    "Machine"
)
```

預期：

```
False
```

如果是 `True`，可以移除。

User：

```
[Environment]::SetEnvironmentVariable(
    "ANTHROPIC_API_KEY",
    $null,
    "User"
)
```

Machine，需要管理員 PowerShell：

```
[Environment]::SetEnvironmentVariable(
    "ANTHROPIC_API_KEY",
    $null,
    "Machine"
)
```

Anthropic 官方說明，Claude Code 會優先使用環境變數中的 API key，即使已登入 subscription 也是如此；因此要使用 subscription，應保持該變數未設定。[Claude Code environment key 說明](https://support.claude.com/en/articles/12304248-manage-api-key-environment-variables-in-claude-code)

## 第 21 步：檢查 Claude Code 版本

```
claude --version
```

Remote Control 需要 Claude Code CLI 2.1.51 或更新版本。

如果太舊：

```
claude update
```

## 第 22 步：登入 Claude Code

啟動：

```
claude
```

進入 Claude Code 後輸入：

```
/login
```

選擇你的 Claude.ai Pro、Max、Team 或 Enterprise 帳號。

不要選擇用 Anthropic Console API key 驗證。

## 第 23 步：確認 Claude Code 使用正確認證

在 Claude Code 輸入：

```
/status
```

確認顯示的是：

- Claude.ai account
- Pro／Max／Team／Enterprise
- subscription authentication

而不是：

- API key
- Console PAYG
- `ANTHROPIC_API_KEY`

## 第 24 步：啟用 Remote Control

在 Claude Code 內輸入：

```
/remote-control
```

也可以從 PowerShell直接啟動：

```
claude remote-control
```

Remote Control 支援 Pro、Max、Team、Enterprise，CLI 需 2.1.51 以上。[Claude Code Remote Control 官方說明](https://support.claude.com/en/articles/14554000-claude-code-power-user-tips)

---

# 如果 App 還會使用 OpenAI

Anthropic 設定完成不代表 OpenAI 也完成。

如果某些功能設定：

```
service: "openai"
```

需要另外建立 OpenAI key，然後執行：

```
python -m tasks.cli_wrappers.manage_llm_credentials set openai
```

確認：

```
python -m tasks.cli_wrappers.manage_llm_credentials status openai
```

對應 Credential Target 是：

```
Moonlight/LLM/OpenAI
```

---

# 最終檢查清單

完成後逐項確認：

- [ ] Anthropic Console 有開發機專用 API key
- [ ] API key 名稱可以辨識這台電腦
- [ ] Development Workspace 有 spend limit
- [ ] `status anthropic` 顯示 `present`
- [ ] Windows Credential Manager 有 `Moonlight/LLM/Anthropic`
- [ ] YAML 的 `api_key` 保持空白
- [ ] `allow_env_fallback` 是 `false`
- [ ] App Claude image analysis 成功
- [ ] User-level `ANTHROPIC_API_KEY` 是 `False`
- [ ] Machine-level `ANTHROPIC_API_KEY` 是 `False`
- [ ] Claude Code `/status` 顯示 subscription login
- [ ] `/remote-control` 可以正常連線



#### Claude申請好多key
```
我在我這台laptop登入了Claude Console. 然後也使用Workspace. 如果有多台安裝App production code以及硬體會放在多個站點, 以及我有兩台laptop開發用, 以這樣在這一步: "第 4 步：建立這台電腦專用的 key" 是不是要申請好多key?
```
是的，如果採用「每站／每台 App 主機獨立 key」，確實會建立多把 API key。但不需要每台電腦建立一個 Workspace；同一個 Workspace 裡可以建立多把 key。

最簡單的計算方式是：

> API key 數量 = Production App 主機數量 + 需要實際測試 App LLM 功能的開發電腦數量

例如：

|用途|電腦數量|建議 key 數量|
|---|---|---|
|Production 站點|5|5|
|開發 Laptop|2|2|
|Claude Code Remote Control|2|0|
|合計||7|

## 建議的 Workspace 配置

不建議每個站點建立一個 Workspace。比較適合的是建立兩個：

### `Moonlight-Production`

每台 Production App 主機一把 key，例如：

```
prod-site-taipei-01
prod-site-taipei-02
prod-site-newyork-01
prod-site-factory-a-01
```

### `Moonlight-Development`

兩台開發 Laptop 各一把 key：

```
dev-laptop-nitro2025
dev-laptop-surface01
```

Anthropic 的 API key 會綁定建立它的 Workspace，不能移到其他 Workspace；Workspace 也可以設定整體的 spending/rate limits。[Anthropic Workspace 文件](https://support.claude.com/en/articles/9796807-creating-and-managing-workspaces-in-the-claude-console)

## 為什麼建議多把 key？

多 key 的主要目的不是增加功能，而是縮小事故影響範圍。

假設 `prod-site-taipei-02` 的電腦遺失或遭入侵：

- 只需停用 `prod-site-taipei-02` 的 key
- 其他站點可以繼續運作
- 不需要重新設定所有電腦
- 可以在 Claude Console 查出是哪一把 key 產生異常用量

Claude Console 的 Usage 報告可以按 API key 篩選，因此清楚的 key 名稱很有價值。[Anthropic Cost and Usage Reporting](https://support.claude.com/en/articles/9534590-cost-and-usage-reporting-in-the-claude-console)

如果所有站點共用同一把 key：

- 一台洩漏，所有站點都必須換 key
- 無法快速判斷是哪個站點產生費用
- 停用時所有站點會一起停止
- 一個站點可能消耗掉其他站點需要的額度

## 「每站」更精確的定義

最好把它理解成：

> 每一台實際執行 Production App 的 Windows 主機一把 key。

例如同一個工廠有兩台獨立 App 電腦，建議兩把 key，而不是因為都在同一站點就共用。

如果一台電腦控制多個相機或硬體，但只有一個 Moonlight App instance，則通常一把 key 就可以。

Windows Credential Manager 的 target 名稱在每台電腦可以相同：

```
Moonlight/LLM/Anthropic
```

但裡面儲存的 key 值不同：

```
站點 A Credential Manager
Moonlight/LLM/Anthropic → prod-site-a 專用 key

站點 B Credential Manager
Moonlight/LLM/Anthropic → prod-site-b 專用 key
```

因此 App code 與設定檔都不需要針對站點修改。

## 兩台開發 Laptop 是否都需要 key？

分成兩種情況。

### Laptop 需要執行 App 並測試圖片分析

需要。兩台各建立一把 Development key：

```
dev-laptop-01
dev-laptop-02
```

不要共用同一把，這樣才能獨立撤銷與追蹤使用量。

### Laptop 只使用 Claude Code / Remote Control

不需要額外建立 App API key。Claude Code 可以使用帳號登入／OAuth，而不是使用 `ANTHROPIC_API_KEY`。

建議保持 Windows 環境變數 `ANTHROPIC_API_KEY` 未設定，因為 Claude Code 會優先使用環境變數裡的 API key，可能造成 API 計費，而不是使用登入帳號包含的額度。[Anthropic 認證說明](https://support.claude.com/en/articles/12304248-manage-api-key-environment-variables-in-claude-code)

目前程式設計是：

```
Moonlight App
  → Windows Credential Manager
  → 每台電腦自己的 App API key

Claude Code / Remote Control
  → Claude 帳號登入
  → 不讀取 Credential Manager 裡的 Moonlight key
```

兩者可以同時存在，不會互相衝突。

## 實際建議

假設目前有 3 個 Production 站點和 2 台開發 Laptop：

```
Workspace: Moonlight-Production
- prod-site-001
- prod-site-002
- prod-site-003

Workspace: Moonlight-Development
- dev-laptop-001
- dev-laptop-002
```

總共建立 5 把 API key。

不必預先建立備用 key。需要輪替時：

1. 建立新的 key，例如 `prod-site-001-2026-rotation`
2. 在該站點更新 Windows Credential Manager
3. 測試 App
4. 確認成功後停用舊 key

這個方案在站點數量是數台到數十台時仍然合理。如果未來發展到數十或數百台，手動發放與輪替會變得麻煩，那時才比較值得導入 Gateway、集中式 Secret Manager 或自動化佈署。



#### cmd line 把 key 存進 Windows Credential Manager?
```
在這部分'第二部分：把 key 存進 Windows Credential Manager"提到在 PowerShell 執行conda activate watch_env, 但python --version, 但顯示Python was not found... 因為我執行App都是在cmd line, 在cmd下就可以正確執行conda activate watch_env執行python version App.

那我是否可以在cmd line 把 key 存進 Windows Credential Manager?
```
可以，完全可以在 CMD（命令提示字元）把 key 存進 Windows Credential Manager。

PowerShell 顯示 `Python was not found`，只是因為 PowerShell 沒有正確載入 Conda 環境；與 Windows Credential Manager 無關。既然你的 CMD 可以正常啟用 `watch_env`，直接用 CMD 操作即可。

## 在 CMD 執行

### 1. 開啟一般 CMD

使用平常執行 App 的 Windows 帳號開啟 CMD。

不建議特別使用「以系統管理員身分執行」，因為 Credential Manager 會跟 Windows 使用者帳號綁定。App 平常以哪個帳號執行，就用那個帳號儲存 key。

### 2. 切換到專案目錄

```
cd /d "D:\Provenance Laboratories projects\ImagingLibWatch"
```

`/d` 的作用是讓 CMD 同時從目前磁碟切換到 `D:`。

### 3. 啟用 Conda 環境

```
conda activate watch_env
```

確認 Python 正常：

```
python --version
```

如果看到類似：

```
Python 3.x.x
```

就代表環境正確。

### 4. 先檢查 Anthropic key 狀態

```
python -m tasks.cli_wrappers.manage_llm_credentials status anthropic
```

如果尚未設定，應該會顯示尚未儲存或不存在。

這個指令只顯示狀態，不會印出真正的 key。

### 5. 儲存 Anthropic API key

```
python -m tasks.cli_wrappers.manage_llm_credentials set anthropic
```

程式會要求輸入兩次：

```
Enter API key:
Confirm API key:
```

貼上 key 後按 Enter。

注意：輸入或貼上時，畫面不會出現文字、星號或游標移動，這是正常的安全設計。不要重複貼上，貼一次後直接按 Enter。

第二次再貼上同一把 key並按 Enter。

### 6. 再次確認

```
python -m tasks.cli_wrappers.manage_llm_credentials status anthropic
```

應該會顯示 Credential 已存在。仍然不會顯示 key 內容。

接著關閉並重新啟動 App，再測試圖片 LLM 分析功能。

## 如果也要設定 OpenAI

使用相同方式：

```
python -m tasks.cli_wrappers.manage_llm_credentials status openai
python -m tasks.cli_wrappers.manage_llm_credentials set openai
```

如果目前沒有使用 OpenAI，就不需要設定。

## 重要觀念

這些指令：

```
python -m tasks.cli_wrappers.manage_llm_credentials set anthropic
```

會把 key 寫進：

```
Windows Credential Manager
└── Moonlight/LLM/Anthropic
```

它不是設定 CMD 環境變數，也不是把 key 寫進 YAML、程式碼或 repo。

因此請不要另外執行：

```
set ANTHROPIC_API_KEY=...
```

也不要使用：

```
setx ANTHROPIC_API_KEY ...
```

這樣可以讓 Moonlight App 使用 Credential Manager 裡的 key，同時避免 `ANTHROPIC_API_KEY` 環境變數干擾 Claude Code 的登入方式。

## Windows 帳號必須一致

例如：

```
Windows 使用者：MYPC\john
```

如果你用 `MYPC\john` 儲存 key，App 也必須由 `MYPC\john` 執行才能讀到。

可在 CMD 查看目前帳號：

```
whoami
```

請確保：

```
執行儲存指令的帳號
=
平常啟動 App 的帳號
```

如果 App 未來改成 Windows Service，並由另一個 service account 執行，就必須在該 service account 的 Credential Manager 中另外設定。

## 完整指令順序

可以依序執行：

```
cd /d "D:\Provenance Laboratories projects\ImagingLibWatch"
conda activate watch_env
python --version
whoami
python -m tasks.cli_wrappers.manage_llm_credentials status anthropic
python -m tasks.cli_wrappers.manage_llm_credentials set anthropic
python -m tasks.cli_wrappers.manage_llm_credentials status anthropic
```

整個過程都可以只使用 CMD，不需要使用 PowerShell。





#### Windows Credential Manager是把api key存取跟python environment有關
```
在python -m tasks.cli_wrappers.manage_llm_credentials status anthropic指令的anthropic是甚麼? 是否是api key的命名譬如development-laptop-NITRO2025? 還是在cmd輸入api key存進 Windows Credential Manager不會用到api key的命名? 另外api key存進 Windows Credential Manager為什麼要先開啟python environment? 是否這Windows Credential Manager是把api key存取跟python environment有關?
```
`anthropic` 不是 API key 的名稱，也不是 API key 本身。它只是告訴管理程式：「我要操作 Anthropic/Claude 這個服務的 Credential」。

可以把這幾個名稱分開理解：

|項目|範例|用途|
|---|---|---|
|Claude Console 裡的 key 名稱|`development-laptop-NITRO2025`|方便你在 Console 辨識是哪台電腦|
|指令中的服務名稱|`anthropic`|告訴程式要設定 Anthropic|
|Credential Manager 儲存名稱|`Moonlight/LLM/Anthropic`|App 用來尋找 Credential 的固定名稱|
|真正的 API key|`sk-ant-...`|實際呼叫 Anthropic API 的秘密內容|

## `anthropic` 代表什麼？

下面這個指令：

```
python -m tasks.cli_wrappers.manage_llm_credentials status anthropic
```

可以拆成：

```
python -m
    執行一個 Python module

tasks.cli_wrappers.manage_llm_credentials
    執行專案裡的 Credential 管理工具

status
    查詢狀態

anthropic
    查詢 Anthropic/Claude 的 Credential
```

所以 `anthropic` 是服務代號。

如果查 OpenAI，則使用：

```
python -m tasks.cli_wrappers.manage_llm_credentials status openai
```

## `development-laptop-NITRO2025` 用在哪裡？

這個名稱是在 Claude Console 建立 API key 時填寫的名稱，例如：

```
development-laptop-NITRO2025
```

它的用途是讓你以後可以在 Claude Console 裡辨認：

- 這把 key 屬於哪台電腦
- 哪台電腦產生 API usage
- 電腦遺失時應該撤銷哪一把 key
- 哪一把 key 需要輪替

例如 Console 裡可能有：

```
development-laptop-NITRO2025
development-laptop-SURFACE
production-site-001
production-site-002
```

但在每一台 Windows 電腦裡，App 尋找的 Credential 名稱都固定是：

```
Moonlight/LLM/Anthropic
```

差別在於裡面存的 key 不同。

例如：

```
NITRO2025
Moonlight/LLM/Anthropic
→ development-laptop-NITRO2025 的真正 key

SURFACE
Moonlight/LLM/Anthropic
→ development-laptop-SURFACE 的真正 key

Production Site 001
Moonlight/LLM/Anthropic
→ production-site-001 的真正 key
```

固定名稱的好處是 App 不需要知道這台電腦在 Console 裡叫什麼名字，只要固定尋找：

```
Moonlight/LLM/Anthropic
```

就能取得該電腦自己的 key。

## 在 CMD 儲存時，會不會輸入 key 名稱？

不會。

你執行：

```
python -m tasks.cli_wrappers.manage_llm_credentials set anthropic
```

程式只會要求輸入：

1. 真正的 API key
2. 再輸入一次確認

大致會是：

```
Enter API key:
Confirm API key:
```

不會要求輸入：

```
development-laptop-NITRO2025
```

因為這個名稱已經在 Claude Console 建立 key 時設定了。Windows Credential Manager 只需要保存真正的 key。

整個關係是：

```
Claude Console
├── 顯示名稱：development-laptop-NITRO2025
└── 真正 key：sk-ant-xxxxxxxx
                    │
                    ▼
Windows Credential Manager
├── Credential 名稱：Moonlight/LLM/Anthropic
└── Credential 內容：sk-ant-xxxxxxxx
                    │
                    ▼
Moonlight App
└── 尋找 Moonlight/LLM/Anthropic，取得真正 key
```

## 為什麼需要啟用 Python environment？

不是因為 Windows Credential Manager 跟 Python environment 有關。

原因只是：

> 我們在 repo 裡寫了一個 Python 管理工具，必須用 Python 執行它。

也就是這個程式：

```
tasks.cli_wrappers.manage_llm_credentials
```

負責：

1. 讀取 `LLM_config.yaml`
2. 找出應使用的 Credential 名稱
3. 安全地讓你輸入 API key
4. 檢查兩次輸入是否相同
5. 呼叫 Windows API
6. 把 key 寫入 Windows Credential Manager
7. 驗證是否成功

`watch_env` 則確保執行這個工具時：

- 使用專案正確的 Python
- 有專案需要的 Python package
- 可以正確 import 專案程式碼
- 使用和 App 相同的執行環境

所以：

```
Conda/Python environment
→ 只是用來執行管理工具

Windows Credential Manager
→ 真正保存 key 的 Windows 系統功能
```

## 關閉 Conda environment 後，key 會不會消失？

不會。

執行：

```
conda deactivate
```

甚至關閉 CMD、重新啟動 Windows，Credential 都還會保留。

因為 key 最後是保存在：

```
Windows Credential Manager
```

而不是保存在：

```
watch_env
CMD
Python 記憶體
環境變數
```

## 如果刪除 `watch_env`，key 會不會消失？

也不會。

刪除或重新建立 Conda environment 不會刪除 Windows Credential Manager 裡的 key。

但是，沒有可用的 Python environment 時，你可能無法執行我們的管理工具來查看、更新或刪除它。Credential 本身仍然存在。

## Credential 跟什麼綁定？

它主要跟「Windows 使用者帳號」綁定，而不是跟 Python environment 綁定。

例如：

```
Windows 帳號：MYLAPTOP\alice
```

如果你用 `alice` 執行：

```
python -m tasks.cli_wrappers.manage_llm_credentials set anthropic
```

Credential 會存在 `alice` 的 Credential Manager 裡。

之後 Moonlight App 也需要以 `alice` 執行，才能讀取它。

它不在意你是從：

- `watch_env`
- 另一個 Conda environment
- 打包後的 EXE

啟動 App；只要 App 是由同一個 Windows 使用者執行，而且 App 使用相同的 Credential target，就可以讀取。

## 為什麼不用其他方式直接寫 Credential Manager？

你也可以從 Windows UI 開啟 Credential Manager，但手動新增 Generic Credential 比較容易發生：

- Credential 名稱拼錯
- 儲存到錯誤的帳號
- 把 username/password 欄位搞混
- 不確定 App 是否成功讀到
- 更新時留下錯誤內容

專案提供的 Python 工具能使用 App 相同的設定和讀取方法進行驗證，因此比較不容易出錯。

另外，請不要把真正的 key 直接放在指令後面，例如：

```
REM 不要這樣做
some-command sk-ant-xxxxxxxx
```

因為指令可能留在 command history 或系統紀錄裡。我們的工具會互動式要求輸入，而且不顯示輸入內容。

總結：

```
anthropic
= 服務代號，不是 key 名稱

development-laptop-NITRO2025
= Claude Console 裡方便管理的名稱

Moonlight/LLM/Anthropic
= Windows Credential Manager 裡的固定名稱

sk-ant-...
= 實際被安全保存的 API key

watch_env
= 執行管理工具需要的 Python 環境，不是保存 key 的地方
```





#### 移除ANTHROPIC_API_KEY跟OPENAI_API_KEY
```
我把claude跟openai的api key 都存入Windows Credential Manager, 那是不是可以移除environment variables: ANTHROPIC_API_KEY跟OPENAI_API_KEY
```
可以。既然 Anthropic 和 OpenAI key 都已成功存入 Windows Credential Manager，就可以移除：

```
ANTHROPIC_API_KEY
OPENAI_API_KEY
```

目前修改後的 App 設定優先使用 Credential Manager，而且已停用環境變數 fallback，因此這兩個環境變數不再是 App 必需的。

## 移除前先確認 Credential

在 CMD 執行：

```
cd /d "D:\Provenance Laboratories projects\ImagingLibWatch"
conda activate watch_env

python -m tasks.cli_wrappers.manage_llm_credentials status anthropic
python -m tasks.cli_wrappers.manage_llm_credentials status openai
```

確認兩者都顯示 Credential 已存在後，再移除環境變數。

## 從 Windows 永久移除

最簡單的方法是使用 Windows UI：

1. 按下 Windows 鍵。
2. 搜尋「環境變數」。
3. 選擇「編輯帳戶的環境變數」。
4. 在「使用者變數」尋找：
    - `ANTHROPIC_API_KEY`
    - `OPENAI_API_KEY`
5. 分別選取後按「刪除」。
6. 也檢查「系統變數」中是否存在同名變數。
7. 如果系統變數也存在，刪除時可能需要系統管理員權限。
8. 按「確定」關閉所有視窗。

不要把變數設定成空白，最好直接刪除該變數。

## 清除目前 CMD 視窗裡的副本

即使已從 Windows UI 刪除，已經開啟的 CMD 可能還保留舊值。

在目前 CMD 執行：

```
set ANTHROPIC_API_KEY=
set OPENAI_API_KEY=
```

這只會清除目前 CMD process 裡的值。最穩妥的方法是刪除後關閉所有 CMD、PowerShell、IDE 和 App，再重新開啟。

## 不顯示 key 地檢查

在新的 CMD 中執行：

```
if defined ANTHROPIC_API_KEY (echo ANTHROPIC_API_KEY is still set) else (echo ANTHROPIC_API_KEY is not set)

if defined OPENAI_API_KEY (echo OPENAI_API_KEY is still set) else (echo OPENAI_API_KEY is not set)
```

請不要使用以下指令檢查：

```
echo %ANTHROPIC_API_KEY%
echo %OPENAI_API_KEY%
```

因為這會把真正的 key 顯示在畫面上。

## 最後驗證 App

在新的 CMD：

```
cd /d "D:\Provenance Laboratories projects\ImagingLibWatch"
conda activate watch_env

python -m tasks.cli_wrappers.manage_llm_credentials status anthropic
python -m tasks.cli_wrappers.manage_llm_credentials status openai
```

然後啟動 App，分別測試：

- Claude 圖片分析
- OpenAI 相關功能

如果功能正常，就證明 App 已經完全從 Windows Credential Manager 讀取 key。

## 一個重要注意事項

移除這些環境變數可能影響這台電腦上的其他程式：

- 其他 Python 專案
- OpenAI CLI
- 自己寫的測試 scripts
- 第三方工具
- 批次檔或排程工作

如果這些工具原本依賴 `OPENAI_API_KEY` 或 `ANTHROPIC_API_KEY`，移除後它們會無法呼叫 API。這不影響目前的 Moonlight App，但需要針對那些工具另外設定。

對 Claude Code 而言，移除 `ANTHROPIC_API_KEY` 通常反而是正確的：Claude Code 就不會誤用 App 的付費 API key，而會使用你登入的 Claude 帳號。可在 Claude Code 裡執行：

```
/status
```

確認它使用的是帳號登入，而不是環境變數 API key。





