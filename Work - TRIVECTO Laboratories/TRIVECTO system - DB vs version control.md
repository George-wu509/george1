
|                                                    |     |
| -------------------------------------------------- | --- |
| [[#### Multiple Local DB how to upload to AWS ]]   |     |
| [[#### system_config.yaml加入在.gitignore]]           |     |
| [[#### production level code的local database是存在哪裡]] |     |
| [[#### Production code 找不到watchDB.db]]             |     |
| [[#### Why local DB no upload to AWS DynamoDB]]    |     |
|                                                    |     |

#### Multiple Local DB how to upload to AWS 
```
請幫我檢查如果我github有main branch, dev branch, usr_ajp user branch, 然後在LA downtown機台也各有local dev branch的workspace在D:\Moonlight\ImagingLibWatch_dev\, 也有usr_ajp branch在D:\Moonlight\ImagingLibWatch\, 還有個production level code在D:\Moonlight\Production\(有MoonlightApp.exe)跟D:\Moonlight\Protected\(有DB\). 那我在LA downtown機台用D:\Moonlight\ImagingLibWatch_dev\, D:\Moonlight\ImagingLibWatch or D:\Moonlight\Production\執行App, 是否都會跟AWS database同步, 然後可能混在一起? 有沒有辦法避免? 或應該怎麼管理這樣多branch甚至還有不同地點(在remote我的laptop也是work on dev branch)
```
直接回答:**會,而且目前的架構完全沒有防止這件事發生的機制** — 這不是我猜測,是我實際看了這個 repo 的 config 與雲端同步程式碼後確認的。

### 為什麼三個資料夾都會打到同一個 AWS

我查了 config/system_config.yaml 和相關文件,關鍵事實:

1. **AWS 資源是寫死在 config 檔裡的固定名稱**,不是依環境切換的:
    - S3 bucket: `moonlight-system-v2`
    - DynamoDB tables: `moonlight-WatchAnalysisResults`、`moonlight-WatchTemplates`、`moonlight-WatchUsers`…
    - Region: `us-east-2`
    - 這些值在 `D:\Moonlight\ImagingLibWatch_dev\`、`D:\Moonlight\ImagingLibWatch\`、`D:\Moonlight\Production\` 三份 checkout 裡**預設是一模一樣的**,因為它們本來就是同一份 repo 複製出來的檔案,沒有 build-time templating 或環境變數覆寫機制(22.4 Configuration and Secret Injection 明確寫著:"a station's `config/` directory is... simply whatever files exist in that station's own checkout, edited directly per-station as needed")。
2. **雲端同步預設是開的**:`security.enable_cloud_sync: true`、`force_mock_cloud_sync: false`。只要 App 啟動時能拿到有效的 AWS 憑證(透過 boto3 credential chain,例如環境變數或 profile),背景執行緒就會自動把本地 SQLite 裡 `synced=0` 的資料每 ~2 秒 poll 一次並上傳到 S3 + 寫入 DynamoDB(13.4 cloud_sync.py)。這跟你是從哪個資料夾、哪個 git branch 執行**完全無關**。
3. **AWS IoT 裝置身分也是共用的**:`network.DeviceID: "Moonlight system v1"` 搭配同一組憑證檔(`AmazonRootCA1.pem`/`private.pem.key`/`certificate.pem.crt`)。如果 dev 和 prod 兩份 checkout 用同一組證書連線,還會在 IoT topic(`cmd/{DeviceID}`、shadow topics)上互相衝突,不只是資料庫混淆而已。
4. **沒有 CI/CD、沒有部署流程**(22.9 Deployment Procedure 明講:"No deployment script... deploying a change to a running station is... a manual process")。也就是說目前完全依賴人工記得「這份是 dev 要關掉 sync」,沒有任何系統性防呆。
5. 補充風險:config/system_config.yaml 裡目前**直接寫著 AWS access key / secret key**,而且這個檔案是被 git 追蹤進版控的(5.5 system_config.md 的 Security warning 段落也警告了這件事)。這是另一個獨立但相關的問題,順便提醒你。

### 建議做法

#### 立即可做(不改 code,純 config 隔離)

- 在 `D:\Moonlight\ImagingLibWatch_dev\config\system_config.yaml` 和 `D:\Moonlight\ImagingLibWatch\config\system_config.yaml`(usr_ajp 那份)裡把 `security.enable_cloud_sync` 設成 `false`(或 `force_mock_cloud_sync: true`),只讓 `D:\Moonlight\Production\` 那份保留 `true`。
- **注意**:因為這個檔案被 git 追蹤,`git pull`/merge 之後這個手動修改可能被覆蓋回去(尤其你在 dev branch 上開發,若哪次 commit 動到這個檔案就會被沖掉)。建議每次 pull 完都養成習慣重新檢查這個 block,或考慮用 `git update-index --skip-worktree config/system_config.yaml` 之類的方式讓本地這處修改不被同步覆蓋(但要小心別把這招用在會漏掉真正需要的 config 更新上)。
- laptop 上的 dev branch workspace 同理處理,而且風險更高(離站更遠、更難注意到誤觸)。

#### 中期建議(架構層級,比較根本)

- **用 IAM 而不是 YAML flag 做隔離**:給 dev/測試用一組只有 dev AWS 資源權限的 `AWS_PROFILE`(或獨立 AWS 帳號),prod 用另一組只有 prod 資源權限的 profile/credentials。這樣就算 config 檔裡的 `enable_cloud_sync` 被不小心留成 `true`,dev 憑證也**技術上就是寫不進 prod 的 bucket/table**——不用依賴人記得關開關。這是文件裡也建議的方向(secrets 不該放 YAML,該走 credential chain)。
- **不同環境用不同的 AWS 資源名稱**,例如 `moonlight-system-v2-dev` 這種 bucket、獨立的 DynamoDB table prefix,徹底避免 dev 資料混進 prod 分析結果表。
- **每個實體站點/每個角色要有自己的 IoT DeviceID + 憑證**,而不是共用同一張證書 —— 這在文件裡本來就是為了未來多站部署(LA/NY/Germany)而規劃的必要條件(22.5 Station Provisioning)。

#### Branch / workspace 管理建議

- **Production 資料夾只能部署 `main` branch 的內容**(或未來如果有 tag/release 機制,用 tag),絕對不要讓它跑 `dev` 或 `usr_ajp` 分支的 code。
- `ImagingLibWatch_dev` 和 `usr_ajp` workspace 定位為沙盒,原則上不該持有能寫入 prod AWS 資源的憑證——這點應該用上面 IAM 的方式強制,而不是靠自覺。
- laptop 上工作時,同樣的 dev-only AWS 憑證原則要套用,且因為 laptop 不在受控站點內,建議乾脆預設關閉 cloud sync,只在需要驗證雲端行為時才手動開、開完立刻關回去。

一句話總結:目前程式碼本身沒有任何 dev/prod 隔離機制,三個資料夾預設會打到同一個 AWS,唯一的防線是「你自己記得改 config」。要真正安全,建議把隔離做到 IAM/AWS 資源層級,而不是只靠 YAML 裡的開關。




#### system_config.yaml加入在.gitignore
```
是否建議把system_config.yaml加入在.gitignore讓它不會被追蹤, 然後設定只有在production code才會上傳到AWS, 其他的dev branch, user branch都是開發用不需要上傳到AWS 這樣比較好管理, 所以在各自的system_config.yaml設成不要同步AWS?
```
方向是對的,而且我看了程式碼——你的想法其實跟現有機制的設計意圖一致。但這件事要拆成兩塊講,因為有一塊比你原本想的更急迫。

### 好消息:code 本來就支援你這個做法

data_manager/local_storage.py:221-254 裡 `_init_cloud_sync()` 已經有雙重防線:

enable_cloud_sync = bool(sec.get("enable_cloud_sync", True))
...
if not enable_cloud_sync:
    return NullCloudSync()
...
if sys_mode in ["simulation", "hybrid"] and not force_s3:
    # 強制 NullCloudSync,保護 data lake

也就是說:`security.enable_cloud_sync: false`(或把 `system.mode` 設成 `simulation`/`hybrid`)本來就是設計好要拿來擋掉真實上傳的開關。你「dev/user branch 各自的 system_config.yaml 設不同步」這個想法,踩的正是現成的正確按鈕,方向沒問題。

### gitignore 這件事要拆成兩塊看

**第一塊(你問的):以後別再讓 dev 的設定被 git 動到**  
把 `system_config.yaml` 加進 `.gitignore` 確實能避免「因為 pull/merge main 或 dev 的變更,把你手動關掉的 `enable_cloud_sync` 又蓋回 `true`」。但要注意一個代價:這個檔案同時也是 ~600 行的**共用 app 設定**(services、tasks、ports、paths 全在裡面),一旦整檔不追蹤,以後別人在 `dev` 合法新增一個 service 或改個 port,不會再自動同步到你的 checkout——變成另一種手動維護的 config drift。折衷做法是拆檔:把 AWS 憑證跟 `enable_cloud_sync` 這種「環境相關/敏感」欄位搬到一個獨立、gitignore 掉的小檔案,其餘 services/tasks/paths 繼續留在版控裡共用。目前 code 是整份 yaml 一次讀入,沒有 overlay 機制(5.3 environment_variables.md 也寫了「環境變數目前不構成對 YAML 的通用 overlay」),所以要做到這樣需要一點小改動,不是純設定就能達成。

**第二塊(比你問的更嚴重,而且是現在進行式)**

我剛查了這個檔案的 git 歷史和 20.4 Secret Storage 文件,發現:

- config/system_config.yaml **從 2025-12-29 到現在的 20 幾次 commit 裡,一直帶著真實明文的 AWS access key / secret key**(現在的檔案裡就看得到 `AKIAWGS7••••••••••••` 這組)。
- config/certs/private.pem.key(AWS IoT 裝置私鑰)、`certificate.pem.crt`、config/keys/hmac.key(稽核日誌簽章金鑰)、`aesgcm.key`(本地資料加密金鑰)也全部被 commit 進 git,**完全沒有被 `.gitignore` 排除**。
- 這代表:**任何 clone 過這個 repo 的人,現在手上就已經有這些真金鑰**——不是「未來可能外洩」的風險,是「已經外洩」的事實。

單純把 `system_config.yaml` 加進 `.gitignore`,只能防止**以後**再 commit 新的明文,**不會**移除已經存在於 git history 裡的舊金鑰。只要 history 裡還查得到,任何有 repo 存取權的人都還是能直接拿那組 key 打你的正式 AWS 資源——不管你之後有沒有把 dev branch 的 `enable_cloud_sync` 設成 `false`,因為那個人根本不需要透過你的 App 執行,直接用 `aws-cli` 或一支腳本就能用同一把 key 讀寫 prod 的 S3/DynamoDB。

### 建議的完整處理順序

1. **先把這些 key/憑證當作已外洩處理**(不是選項,是必須):在 AWS IoT Core 重發一組新證書並撤銷舊的、重新產生 `hmac.key`/`aesgcm.key`、輪替 AWS access key。
2. **把這些檔案(連同 git history)徹底清除**——只在新 commit 裡刪除檔案不夠,舊版本還在歷史裡可還原;需要 `git filter-repo` 或等效方式,並協調所有其他 clone/fork 一起處理(20.11 Credential Leak Response 有寫完整流程)。
3. **AWS access_key_id/secret_access_key 這兩個欄位建議直接從 yaml 拿掉**,改用 boto3 credential chain(`AWS_PROFILE` 環境變數或 IAM role)——文件裡本來就註記這兩個欄位是「prohibited target-state secrets」。這樣一來,不同 workspace 可以直接用**不同權限的 AWS profile**做隔離(dev profile 甚至可以在 IAM 層級就沒有寫入 prod bucket/table 的權限),比單純一個 YAML flag 更保險——就算哪次不小心 `enable_cloud_sync` 被設成 `true`,dev 的憑證在 AWS 端也寫不進去。
4. **之後**再把 `config/certs/`、`config/keys/`、以及拆分出來的 AWS 憑證/`enable_cloud_sync` override 檔案加進 `.gitignore`,防止歷史重演。

一句話:你的「prod 上傳、dev/user 不上傳」的分工方向沒問題,gitignore 也該做,但它解決的是「以後別再犯」;現在更急的是「已經外流的 4 組金鑰」要立刻輪替並從 history 清除,這個優先權應該排在 gitignore 之前。



#### production level code的local database是存在哪裡
```
那production level code的local database是存在哪裡?
```
我追進 core/frozen_runtime.py 和 data_manager/local_storage.py:139-158,找到確切機制了。

### 結論:Local DB 的位置取決於「是不是 frozen exe」,不是取決於哪個 branch

App 用 core/frozen_runtime.py:55-68 的 `runtime_root()` 決定資料根目錄,邏輯是:

def protected_root():
    if 環境變數 MOONLIGHT_PROTECTED_ROOT 有設定: 用它
    elif is_frozen():          # 也就是跑的是 PyInstaller 打包出來的 .exe
        return D:\Moonlight\Protected     # 寫死的預設值
    else:
        return source_root()   # 就是這次執行時的原始碼所在資料夾

def runtime_root():
    if 環境變數 MOONLIGHT_RUNTIME_ROOT 有設定: 用它
    else:
        return protected_root() if is_frozen() else source_root()

而本地 SQLite 的路徑是 `db_cfg.get("db_path", "DB/watchDB.db")`,這個相對路徑會被接到 `runtime_root()` 後面(local_storage.py:154-158)。所以實際落點是:

|執行方式|`is_frozen()`|Local DB 實際位置|
|---|---|---|
|**Production**:`D:\Moonlight\Production\MoonlightApp.exe`(PyInstaller 打包)|`True`|**`D:\Moonlight\Protected\DB\watchDB.db`**|
|**ImagingLibWatch_dev**:直接跑 `python App/main.py` 之類(source 執行)|`False`|`D:\Moonlight\ImagingLibWatch_dev\DB\watchDB.db`|
|**usr_ajp workspace**(`D:\Moonlight\ImagingLibWatch\`,source 執行)|`False`|`D:\Moonlight\ImagingLibWatch\DB\watchDB.db`|

這也解釋了你觀察到的現象——為什麼 `MoonlightApp.exe` 裝在 `Production\`,但 DB 卻出現在 `Protected\` 底下:因為打包成單檔 exe 後,PyInstaller 會把程式解壓到暫存的 `_MEI...` 目錄,寫入的資料不能留在那裡(重啟就沒了),所以 `frozen_runtime.py` 特別把 frozen 模式的可寫入根目錄導向一個獨立、由管理員預先建置好的外部資料夾 `D:\Moonlight\Protected`(同一份程式碼註解也直接寫了這個理由)。同理,`config\system_config.yaml`、`Local_Data\`(原始影像/分析結果)、`checkpoints\`、`assets\` 在 Production 模式下,也都是讀寫 `D:\Moonlight\Protected\` 底下對應的子資料夾,不是 `D:\Moonlight\Production\` 裡面——這點 helper/deployment/README.md 的目錄圖也印證了。

### 好消息:local DB 天生就是三個獨立檔案

跟前面討論的 AWS 雲端資料庫不同,**local SQLite DB 預設本來就不會混在一起**——因為它是相對路徑接到「執行當下的 runtime_root」,而三種執行方式(frozen exe / 兩份 source checkout)的 runtime_root 天生就是三個不同資料夾。這算是現有架構已經做對的部分。

### 一個要注意、值得你去確認的例外

`runtime_root()` 和 `protected_root()` 都會**優先看環境變數** `MOONLIGHT_RUNTIME_ROOT` / `MOONLIGHT_PROTECTED_ROOT`,不管是不是 frozen 模式都一樣優先。也就是說:**如果 LA downtown 那台機台上,系統層級(System,不是 User)設了這兩個環境變數指向 `D:\Moonlight\Protected`**,那麼就算你用 source 直接跑 `ImagingLibWatch_dev` 或 `usr_ajp` 那份,也會被導去跟 Production 共用同一個 `Protected\DB\watchDB.db`,local 層級的隔離就失效了。

建議在那台機台上開 PowerShell 確認一下:

[System.Environment]::GetEnvironmentVariable("MOONLIGHT_RUNTIME_ROOT","Machine")
[System.Environment]::GetEnvironmentVariable("MOONLIGHT_PROTECTED_ROOT","Machine")
[System.Environment]::GetEnvironmentVariable("MOONLIGHT_RUNTIME_ROOT","User")
[System.Environment]::GetEnvironmentVariable("MOONLIGHT_PROTECTED_ROOT","User")

如果這兩個都是空的,你目前的三份 local DB 是天然隔離的,不用額外處理。

最後提醒一點:這個 local SQLite DB 本身只是本機佇列/紀錄,它跟不跟 AWS 的雲端 DB(S3/DynamoDB)同步,還是取決於我們前面討論的 `security.enable_cloud_sync` 那層開關——local DB 隔離跟 AWS 混不混是兩個獨立的問題,前者現在沒事,後者才是真正要處理的部分。




#### Production code 找不到watchDB.db
```
可是我遠端登入在LA downtown的主機, 但在D:\Moonlight\Protected\DB\看不到watchDB.db, 也不是隱藏檔, 但我執行D:\Moonlight\Production\MoonlightApp.exe 在App卻可以看到存在local DB裡面的templates? 為何
```
這代表「你看到的資料夾」跟「App 實際在讀寫的檔案」不是同一個位置。我在程式碼裡多找到一層,先講關鍵發現,再給你怎麼精確定位的方法。

### 關鍵發現:DB 路徑其實還可以被 config 覆寫

data_manager/local_storage.py:153-158:

db_cfg = (self.config.get("db") or {})
db_path = db_cfg.get("db_path", "DB/watchDB.db")

也就是說,DB/watchDB.db 只是**沒有設定時的預設值**。如果 `D:\Moonlight\Protected\config\system_config.yaml`(這是 production 實際在讀的那份 live config,不是這個 repo 追蹤的版本)裡有一段:

db:
  db_path: 某個其他路徑

那 DB 實際上會落在那個別的路徑,而不是 `Protected\DB\watchDB.db`。值得注意的是,helper/deployment/README.md 列出的目標目錄結構寫的是 `D:\Moonlight\Protected\data\`,**不是 `DB\`**——這暗示 production 的實際 config 很可能把 `db_path` 導向 `Protected\data\...` 底下的某處,而不是走預設值。

### 最可靠的驗證方法:直接問正在跑的那個 process

與其猜 config,不如直接看 `MoonlightApp.exe` 這個 process 現在手上開著哪個檔案控制代碼——這個資訊不會騙你:

resmon

開啟後到 **CPU 分頁 → Associated Handles**,在搜尋框打 `watchDB`,勾選 `MoonlightApp.exe` 對應的那一行,就會列出它目前實際開啟的檔案完整路徑。這是最準確、不用猜的方法。

如果你的環境沒有 GUI 或想用命令列,也可以用 [Sysinternals handle.exe](https://learn.microsoft.com/sysinternals/downloads/handle)(微軟官方工具,需要你自己下載,我不會幫你下載執行未知檔案):

handle.exe -p MoonlightApp.exe watchDB

### 其次要檢查的兩件事

**1. Production 的 live config 裡有沒有自訂 `db:` 區塊**

直接在那台機器上開:

Get-Content "D:\Moonlight\Protected\config\system_config.yaml" | Select-String -Context 2,2 "db_path|^db:"

**2. 環境變數是不是「User 範圍」設定,而 App 實際執行的 Windows 帳號跟你遠端登入的帳號不是同一個**

如果 `MoonlightApp.exe` 是被某個服務帳號、排程工作、或另一個 Windows 使用者身分啟動的(常見於站點機台是某個 operator 帳號常駐執行),而你是用自己的帳號遠端登入去「看」`D:\Moonlight\Protected\DB\`,兩者看到的環境變數可能不同,也可能導致你有讀取權限上的落差(即使檔案存在,你的帳號可能就是看不到)。先確認 App 實際跑在哪個帳號下:

Get-CimInstance Win32_Process -Filter "Name='MoonlightApp.exe'" |
    ForEach-Object { $_.GetOwner() }

再對照你自己這個遠端登入 session 的帳號(`whoami`),如果不一樣,接著檢查該資料夾/檔案的 ACL:

Get-Acl "D:\Moonlight\Protected\DB" | Format-List

helper/deployment/README.md:86-91 裡其實已經提到這個資料夾未來會限制成「`operator1: No Access`」這類權限管控,說明這個資料夾本來就預期會有帳號層級的存取限制——如果目前已經套用了類似規則,而你的遠端登入帳號剛好不在允許清單裡,就會出現「檔案確實存在、App 讀得到,但你的 Explorer 看不到/看不到內容」的狀況。

**建議順序**:先用 `resmon` 看 handle(一次到位、不用猜),如果確認路徑跟你以為的不一樣,再回頭查 production config 的 `db:` 區塊;如果路徑真的就是 `Protected\DB\watchDB.db` 但你就是看不到,那基本上就是帳號/ACL 問題,查 `Get-Acl` 會告訴你答案。




#### Why local DB no upload to AWS DynamoDB
```
而且我遠端登入在LA downtown主機, 在dev branch執行App 並create new template, 我隨後在那裏打開test_template_ui.py, 可以看到我新create的template, 但我過了一小時打開AWS DynamoDB的moonlight-WatchTemplates, 有看到以前的template但看不到最近create的template包括最新的, 那可能是出甚麼問題
```
### 首先要澄清:Template 上傳跟你創建的時間點是「同步、不重試」的

跟圖片/分析結果不同,template 存雲端**不是**走那個有背景 worker、失敗會不斷重試的 outbox/CloudSyncManager 佇列。它是在你按下儲存的那一刻直接同步呼叫的(App/main.py:14211-14220):

cloud_db = getattr(getattr(self, "workflow_engine", None), "cloud_db", None)
saved_cloud = bool(
    saved_local
    and cloud_db
    and cloud_db.save_template_cloud(cloud_payload)
)

data_manager/outbox_dispatcher.py:165-167 的註解也明講:「GUI 和 CLI 的 cloud-owned/cache 流程會 opt out」這個有重試機制的 outbox 路徑。**意思是:如果當下上傳失敗,它就是失敗了,不會像圖片那樣每 2 秒自動重試——等一小時不會讓它自己冒出來。** `saved_local` 只要成功就會馬上反映在 `test_template_ui.py`(讀本地 SQLite),但 `saved_cloud` 是完全獨立、當下就決定生死的另一件事。

### 最可疑的具體原因:region / table 名稱預設值不一樣

data_manager/cloud_db.py:36,40:

self.region = self.aws_cfg.get("region", "us-east-1")
self.template_table_name = self.aws_cfg.get("template_table", "WatchTemplates")

但這個 repo 追蹤的 config/system_config.yaml 裡實際值是 `region: us-east-2`、`template_table: moonlight-WatchTemplates`。**如果你在 LA downtown 那台機器上、dev branch checkout 裡的 `system_config.yaml`(或它讀到的那份 config)`aws:` 區塊缺了 `region` 或 `template_table` 這兩個 key**(比如被手動精簡過、或是舊版沒同步到最新欄位),`CloudDatabaseManager` 會**悄悄地**寫進 `us-east-1` 區域裡一個叫 `WatchTemplates`(不是 `moonlight-WatchTemplates`)的表——你在 AWS Console 開 `us-east-2` 底下的 `moonlight-WatchTemplates` 當然永遠找不到,因為東西根本不在那裡。這是我覺得最值得先查的,因為它完全符合「舊 template 看得到(可能是別的環境/時期用完整 config 寫的),新的看不到」的現象。

**檢查方法**:在那台機器上直接看 dev checkout 實際載入的 config:

Select-String -Path "D:\Moonlight\ImagingLibWatch_dev\config\system_config.yaml" -Pattern "region:|template_table:"

再去 AWS Console 右上角確認你正在看的是哪個 region,跟這個值比對。如果 dev 那份缺這兩個 key 或值不一樣,順手也去 `us-east-1` 找找看有沒有一個叫 `WatchTemplates` 的表,新 template 很可能就在那裡。

### 第二可能:`cloud_db` 根本是 `None`,上傳從沒被嘗試過

13.5 cloud_db.md 提到 `system.offline_mode` 設 `true` 時,`CloudDatabaseManager` 在建構階段就直接跳過連線。如果你(或先前我們討論「dev branch 不要同步 AWS」時)在這份 dev checkout 裡動過 `system.offline_mode`,`workflow_engine.cloud_db` 可能整個是 `None`——上面那段 `saved_local and cloud_db and ...` 短路求值,直接得到 `False`,**不會丟錯誤、不會有例外**,UI 上很容易被忽略。

**檢查方法**:確認 dev checkout 的 `system.offline_mode` 是否為 `true`,以及當時的應用程式日誌裡有沒有 `CloudDB` logger 的訊息(`save_template_cloud` 失敗一定會 log 一行)。

### 第三種可能(較低機率但值得排除)

- **憑證/網路瞬斷**:如果那一刻 AWS session token 剛好過期或網路短暫中斷,`save_template_cloud` 內部 try/except 只會記一行 error、回傳 `False`,同樣不重試。
- **`TEMPLATE_SYNC_BLOCKLIST` 或 schema 檢查誤判**:`save_template_cloud` 重用了跟本地層一樣的防呆邏輯(13.5 有列出保留字清單),正常新建的 template 應該不會誤觸,但可以在 log 裡確認排除。
- **380KB 大小限制**:如果這個 template 內嵌了較大的 watchshift 參考影像資料,超過 `DYNAMODB_ITEM_SOFT_LIMIT_BYTES` 會被拒絕並記錄錯誤,而不是拋出明顯例外。

### 建議下一步

1. 先查 dev checkout 的 `aws.region` / `aws.template_table` 是否跟你在 Console 上查的一致(最可疑)。
2. 查當時的 log 有沒有 `CloudDB` logger 的錯誤訊息。
3. 因為這個路徑不會自動重試,**如果確認是失敗了,得回到 App 裡重新開啟該 template 再存一次**,而不是等它自己補上去。


