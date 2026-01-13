

|                                            |     |
| ------------------------------------------ | --- |
| [[#### AWS S3 account setting]]            |     |
| [[#### 完整AWS Cloud各種服務整理]]                 |     |
| [[#### 設定AWS S3的每一流程步驟step-by-step]]       |     |
| [[#### 如果保管AWS 金鑰]]                        |     |
| [[#### 設定AWS DynamoDB的每一流程步驟step-by-step]] |     |
| [[#### 設定AWS IoT Core的每一流程步驟step-by-step]] |     |
|                                            |     |
|                                            |     |
|                                            |     |





#### AWS S3 account setting
```
我之前已經有了我自己的AWS account有Root user email, AWS account name跟IAM Role name. 但目前我開發的App會有camera拍攝images並同步到AWS S3上. 請解釋AWS這些account之間的關係, 並一步步非常詳細解釋從AWS主控台如何create AWS S3 account以及所有一系列設定, 以及如何讓image可以上傳到Amazon S3. 以下是我用來同步AWS S3的code cloud_sync.py跟system_config.yaml
```

下面我用「你的情境＝桌面 App 在本機拍照、再把檔案同步到同一個 AWS 帳號底下的 S3」來講清楚 **AWS 帳號 / Root user / IAM Role / S3** 的關係，並且**一步步**帶你從 AWS Console 建立 S3（正確說法是 _建立 S3 Bucket_，不是 S3 account），再到讓你這份 `cloud_sync.py` 能順利上傳。

---

## 1) AWS 這些身分/資源之間的關係（用你提到的名詞解釋）

### A. AWS Account（帳號）

- 你只有 **一個 AWS account**（例如一個 12 位數 account ID）。
    
- **所有資源**（S3 buckets、IAM users/roles、EC2…）都屬於這個 account。
    

### B. Root user（Root email / Root user）

- Root user 是「這個 AWS account 的最高權限身分」，用你註冊帳號時的 email 登入。
    
- **強烈建議：不要用 root 做日常操作，也不要用 root 建 access keys。** [AWS 文件+2AWS 文件+2](https://docs.aws.amazon.com/IAM/latest/UserGuide/root-user-best-practices.html?utm_source=chatgpt.com)
    
- Root 最常做的事：帳務、付款、關閉帳號、某些只有 root 才能做的設定等。
    

### C. IAM（Identity and Access Management）

IAM 裡面你會看到：

- **IAM User**：給「人或應用程式」用的長期身分（可有 console password、可有 access keys）。
    
- **IAM Role**：給「被信任的實體去 _assume_」的權限集合（例如 EC2 / ECS / Lambda / 或某個 IAM user 透過 STS 去 assume）。Role 本身**不是帳號**，也不是固定某一個人。 [Amazon Web Services, Inc.](https://aws.amazon.com/blogs/security/guidelines-for-protecting-your-aws-account-while-using-programmatic-access/?utm_source=chatgpt.com)
    

> 你的桌面 App 若在「客戶端電腦 / 本機」跑：最常見是用 **IAM User access key**（或用 AWS IAM Identity Center/SSO）。  
> 若你的 App 之後改成跑在 AWS（EC2/ECS/Lambda）：最佳實務是用 **IAM Role** 拿短期憑證，而不是放 long-term access key。 [AWS 文件+1](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html?utm_source=chatgpt.com)

### D. S3（Amazon Simple Storage Service）

- S3 是一個服務；你要建立的是 **Bucket**（像一個「頂層資料夾/磁碟機」），Bucket 底下放 objects（檔案）。 [AWS 文件+1](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html?utm_source=chatgpt.com)
    
- Bucket 名稱必須 **全球唯一**（全 AWS 所有人共用命名空間）。
    

---

## 2) 先決定你的最佳做法（依你 App 的部署方式）

### 情境 1：App 在本機 / 客戶端 Windows 電腦跑（你現在最像這個）

✅ 建議作法：

1. 建立 **S3 bucket**
    
2. 建立一個 **IAM user（例如 watch-app-uploader）**
    
3. 給它「只允許上傳/下載/列舉你指定 bucket」的最小權限
    
4. 把 access key 存在本機的 `~/.aws/credentials`（或環境變數、Windows Credential Manager）
    
5. 你的 `boto3.client('s3')` 就會自動吃到憑證鏈
    

### 情境 2：App 之後跑在 EC2/ECS/Lambda

✅ 建議作法：

- 用 **IAM Role** 掛到 EC2 instance profile 或 ECS task role，boto3 會自動從 metadata 取短期憑證。 [boto3.amazonaws.com+1](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html?utm_source=chatgpt.com)
    

下面我先以「情境 1（本機桌面 App）」做完整 step-by-step。

---

## 3) AWS Console：建立 S3 Bucket（逐步點選）

1. 登入 AWS Management Console（建議用 IAM user/SSO；避免用 root 進行日常操作） [AWS 文件+1](https://docs.aws.amazon.com/IAM/latest/UserGuide/root-user-best-practices.html?utm_source=chatgpt.com)
    
2. 進入 **S3** 服務
    
3. 左側點 **Buckets**
    
4. 點 **Create bucket**
    

### 在 Create bucket 頁面，照這樣填（重要選項我都寫原因）

**(1) Bucket name**

- 例如：`watch-analysis-v1`
    
- 如果跳「已被使用」，代表全球有人用過，要換成更獨特：  
    `watch-analysis-v1-<公司名縮寫>-<一串數字>`
    

**(2) AWS Region**

- 你 `system_config.yaml` 現在是 `us-east-1`，那你就選 **us-east-1**（避免你日後管理/合規/延遲混亂）。
    
- 注意：Bucket 名稱全球唯一，但 bucket 是有 region 概念的。 [AWS 文件+1](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html?utm_source=chatgpt.com)
    

**(3) Object Ownership**

- 建議保留預設：通常會是 **Bucket owner enforced（ACL disabled）**，用 policy 管權限更乾淨。 [AWS 文件](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html?utm_source=chatgpt.com)
    

**(4) Block Public Access settings**

- **維持全部勾選（Block all public access）**  
    你的影像資料通常不該公開。 [AWS 文件+1](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html?utm_source=chatgpt.com)
    

**(5) Bucket Versioning（建議開）**

- 建議：**Enable**  
    好處：誤刪、覆蓋可以救回（但要注意成本：舊版本也算儲存）。 [AWS 文件](https://docs.aws.amazon.com/AmazonS3/latest/userguide/GetStartedWithS3.html?utm_source=chatgpt.com)
    

**(6) Default encryption（建議開）**

- 建議：**Enable**，通常用 SSE-S3（AES256）就夠；或用 SSE-KMS（更細的金鑰控管）。 [AWS 文件](https://docs.aws.amazon.com/AmazonS3/latest/userguide/default-bucket-encryption.html?utm_source=chatgpt.com)
    

**(7) Create bucket**

- 檢查無誤後按 **Create bucket**
    

> 到這一步，你的「S3 bucket」就建立好了。

---

## 4) IAM：建立最小權限的上傳身分（Console 步驟）

> 你提到你已經有 “IAM Role name”。  
> 但對「本機桌面 App」最直覺的是建立 IAM user access key；Role 比較常用在 AWS 內部服務（EC2/ECS/Lambda）。我先給你 IAM user 方案，後面再補「如果你堅持要用 Role」要怎麼做。

### Step A：建立一個 IAM Policy（只允許存取你的 bucket）

1. 進 IAM 服務
    
2. 左側 **Policies**
    
3. 點 **Create policy**
    
4. 選 **JSON**，貼上下面（把 `watch-analysis-v1` 換成你的 bucket 名）
    

`{   "Version": "2012-10-17",   "Statement": [     {       "Sid": "ListBucketInPrefix",       "Effect": "Allow",       "Action": ["s3:ListBucket"],       "Resource": "arn:aws:s3:::watch-analysis-v1"     },     {       "Sid": "ObjectRW",       "Effect": "Allow",       "Action": [         "s3:PutObject",         "s3:GetObject",         "s3:AbortMultipartUpload",         "s3:ListBucketMultipartUploads",         "s3:ListMultipartUploadParts"       ],       "Resource": "arn:aws:s3:::watch-analysis-v1/*"     }   ] }`

這類 policy/permissions 的寫法是 AWS 官方文件範圍內的典型模式（bucket 本身 vs bucket 內 objects 需要不同 Resource ARN）。 [AWS 文件+1](https://docs.aws.amazon.com/AmazonS3/latest/userguide/access-policy-language-overview.html?utm_source=chatgpt.com)

5. Next → 命名，例如：`WatchAnalysisS3RWPolicy`
    
6. Create policy
    

> 你之後如果想限制「只能寫入某個 prefix（例如只准寫 `images/`）」也可以做到（會稍微複雜一點，需要搭配 `s3:prefix` 條件）。有需要我再幫你改成 prefix 限制版本。

### Step B：建立 IAM User（給你的桌面 App 用）

1. IAM → **Users** → **Create user**
    
2. User name：`watch-app-uploader`
    
3. Permissions：
    
    - 選 **Attach policies directly**
        
    - 勾你剛做的 `WatchAnalysisS3RWPolicy`
        
4. Create user
    

### Step C：建立 Access Key（讓程式用）

1. 點進 `watch-app-uploader`
    
2. **Security credentials** 分頁
    
3. **Access keys** → **Create access key**
    
4. Use case 選「Application running outside AWS」（類似選項）
    
5. 建立後會看到：
    
    - `AWS_ACCESS_KEY_ID`
        
    - `AWS_SECRET_ACCESS_KEY`  
        **這個 secret 只會顯示一次，務必保存。** [AWS 文件](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html?utm_source=chatgpt.com)
        

---

## 5) 在你的 Windows 本機設定 AWS 憑證（讓 boto3 讀得到）

你的 `cloud_sync.py` 這行：

`s3_client = boto3.client('s3', region_name=self.region)`

boto3 會走「標準憑證鏈」去找 credentials（環境變數、shared credentials file、IAM role 等）。 [boto3.amazonaws.com+2boto3.amazonaws.com+2](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html?utm_source=chatgpt.com)

### 方案 1（最常用）：用 AWS CLI 寫入 `~/.aws/credentials`

1. 安裝 AWS CLI
    
2. 打開 PowerShell / CMD，執行：
    

`aws configure`

依序輸入：

- AWS Access Key ID：你剛建立的
    
- AWS Secret Access Key：你剛建立的
    
- Default region name：`us-east-1`
    
- Default output format：`json`（或留空）
    

Windows 的 `~/.aws/credentials` 通常會在 `%UserProfile%\.aws\credentials`。 [AWS 文件](https://docs.aws.amazon.com/cli/v1/userguide/cli-configure-files.html?utm_source=chatgpt.com)

### 方案 2：用環境變數（不落地檔案）

設定：

- `AWS_ACCESS_KEY_ID`
    
- `AWS_SECRET_ACCESS_KEY`
    
- `AWS_DEFAULT_REGION`（或程式指定 region_name 也可）
    

---

## 6) 讓你的程式真的上傳成功：你現有程式的關鍵檢查點

你在 `system_config.yaml` 已經有：

`aws:   s3_bucket: "watch-analysis-v1"   region: "us-east-1"`

在 `CloudSyncManager.__init__()` 也讀得到：

`self.bucket_name = config.get('aws', {}).get('s3_bucket', 'watch-analysis-v1') self.region = config.get('aws', {}).get('region', 'us-east-1')`

上傳也正確用：

`s3_client.upload_file(file_path, self.bucket_name, s3_key)`

所以真正會卡住的通常是以下幾種：

### A) 憑證不存在 / 沒被 boto3 找到

典型錯誤：`NoCredentialsError`  
你已經 `from botocore.exceptions import NoCredentialsError`，但目前沒有特別 catch 它、也沒有在 log 裡把「當下使用哪種 credential」印出來。

✅ 快速驗證方法（不用改太多 code）：

- 在同一台機器、同一個使用者下，開 terminal 執行：
    

`aws sts get-caller-identity`

如果這個成功，代表你的本機憑證設定 OK，boto3 多半也 OK。

### B) Bucket 名稱或 Region 不一致

- 你 region 用 `us-east-1` 是 OK 的（S3 client 的 region 不一定要跟 bucket 完全一致也能運作，但建議一致，避免某些情境出錯或管理混亂）。
    

### C) 權限不足（AccessDenied）

常見是 policy 沒給到：

- `s3:PutObject` 到 `bucket/*`
    
- 或 bucket policy 另外有 Deny（較少見，除非你自己加過）
    

---

## 7) 建議你對 `cloud_sync.py` 做的 3 個「很實用」小強化（讓同步更穩）

> 不改也能用；但下面這些會讓你更容易 debug、也更像 production。

### 強化 1：把 `NoCredentialsError` 與 `ClientError` 分開記錄

這樣你一看 log 就知道是「本機沒憑證」還是「權限/資源問題」。

### 強化 2：加上 ExtraArgs（ContentType / ServerSideEncryption）

即便你已開 default encryption，仍可在 put 時明確指定（可選）。

### 強化 3：加 retry（或失敗重回 queue）

你已經註解提到 retry，很適合加：例如每個檔案最多重試 3 次，並且延遲退避。

如果你希望我**直接給你一份修改完成、可直接替換**的 `cloud_sync.py`（包含 retry、清楚錯誤分類、可選 profile/role、以及更完整 log），我也可以用你現有架構改好。

---

## 8) 如果你「一定要用你已有的 IAM Role name」在本機上傳（可行，但概念要對）

本機要用 Role，通常是：

- 你本機先有一個「來源身分」（例如一個 IAM user 或 SSO）
    
- 再透過 `~/.aws/config` 設定 `role_arn` + `source_profile`，讓 AWS SDK/CLI 幫你去 assume role 取得短期憑證（STS）
    

boto3/CLI 的設定方式屬於「configuration / credential provider chain」的一部分。 [boto3.amazonaws.com+2boto3.amazonaws.com+2](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html?utm_source=chatgpt.com)

> 但就你現在的需求（桌面 App 上傳），**直接用最小權限 IAM user access key**通常更直覺、也更好控管。

---

## 9) 你接下來照做的最短路徑（建議照這個順序）

1. ✅ S3 建 bucket（Block public access + Encryption + Versioning 建議打開） [AWS 文件+2AWS 文件+2](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html?utm_source=chatgpt.com)
    
2. ✅ IAM 建 policy（只允許你的 bucket） [AWS 文件+1](https://docs.aws.amazon.com/AmazonS3/latest/userguide/access-policy-language-overview.html?utm_source=chatgpt.com)
    
3. ✅ IAM 建 user + access key（不要用 root key） [AWS 文件+1](https://docs.aws.amazon.com/IAM/latest/UserGuide/root-user-best-practices.html?utm_source=chatgpt.com)
    
4. ✅ Windows `aws configure`，用 `aws sts get-caller-identity` 驗證
    
5. ✅ 跑你的 App，確認 log 出現 `Upload success: ...`
    
6. （可選）我幫你把 `cloud_sync.py` 加上 retry + 更清楚的錯誤分類
    

---

如果你願意，我可以直接依你現在 `system_config.yaml` 的風格，幫你補兩個設定欄位（例如 `aws.profile`、`aws.s3_prefix`），並給你一版「更 production」的 `cloud_sync.py`，讓你同時支援：

- 本機 IAM user profile
    
- 之後改成 EC2/ECS role 時不用改 code（只改部署方式）





#### 完整AWS Cloud各種服務整理
```
這個系統需要AWS Cloud 上面有WatchTemplates, WatchAnalysisResults, WatchCommandLog

等DynamoDB並會需要跟Local SQLite同步, 以及將Camera taken images上傳到Cloud上面儲存, 以及Images and metadata results上傳到Cloud上面儲存, 以及AuditLogger也在Cloud上. 除此之外還有Remote client跟Local computers之間的AWS IoT (MQTT)聯繫, 傳送Local computers的camera畫面到Remote Client, 以及透過AWS cloud可以做到遠端幫local computers(可能在Germany or New York)升級軟體版本.

請中文詳細解釋這System的code關於Cloud部分的技術及設計. 並以30年以上的Cloud架構師及software tech lead做出專業的規劃設計及建議, 要使用哪些AWS服務並分析為何.
```
這是一個非常完整且具備企業級潛力的 Hybrid Cloud（混合雲）AI/IoT 架構。根據你提供的 Python 代碼，我將以 **30 年資深雲端架構師 (Cloud Architect) 及軟體技術負責人 (Software Tech Lead)** 的角度，為你詳細剖析現有的 Cloud 技術實作，並提出針對 AWS 的專業規劃建議。

---

### 第一部分：現有代碼的 Cloud 技術架構分析

目前的代碼實現了一套 **"Cloud-Managed, Edge-Executed"（雲端管理，邊緣執行）** 的架構。核心在於讓 Local Computer（位於德國或紐約的邊緣設備）保有自主運算能力，同時接受雲端的指揮與資料備份。

#### 1. 控制平面 (Control Plane) - AWS IoT & Command Log

- **代碼對應:** `cloud_relay/aws_agent.py`, `core/workflow_proxy.py`, `data_manager/cloud_db.py` (`WatchCommandLog`)
    
- **技術分析:**
    
    - **通訊協定 (MQTT):** 系統使用 `AWSIoTPythonSDK` 透過 MQTT 協定 (TLS 1.2, Port 8883) 連接 AWS IoT Core。這是正確的選擇，因為 MQTT 輕量且支援雙向通訊，能讓位於防火牆後的 Local Computer (NAT 環境) 不需要開放入站端口 (Inbound Ports) 即可接收指令。
        
    - **指令冪等性 (Idempotency):** 在 `aws_agent.py` 中使用了 `_CommandDedupCache` 和 DynamoDB `claim_command_once`。這是一個非常資深的設計，防止網路重發導致機械手臂重複移動或重複拍照。
        
    - **狀態同步 (Shadow):** 代碼中使用了 Device Shadow (`$aws/things/.../shadow/update`) 來回報 `busy` / `idle` 狀態。這讓 Remote Client 即使在設備短暫斷線時，也能知道設備最後的狀態。
        
    - **Command Log (DynamoDB):** 使用 `WatchCommandLog` Table 記錄每一筆指令的狀態 (Running -> Completed/Failed)。這不僅是 Log，更是 **分散式系統的狀態機**，讓 Client 端可以 polling 查詢長任務的執行進度。
        

#### 2. 資料平面 (Data Plane) - S3 & Metadata Indexing

- **代碼對應:** `data_manager/cloud_sync.py`, `data_manager/local_storage.py`, `data_manager/cloud_db.py` (`WatchAnalysisResults`)
    
- **技術分析:**
    
    - **非同步上傳 (Async Push):** `CloudSyncManager` 採用背景執行緒 (Background Thread) 掃描 Local SQLite (`synced=0`)，然後上傳 S3 並寫入 DynamoDB。這種 **"Store-and-Forward"** 模式保證了即使斷網，邊緣設備也能繼續生產數據，待網路恢復後自動同步。
        
    - **Split-Key Pattern:** 在 `WatchRecordManager` 中，看到將龐大的 `watchentry` 拆解為 `INFO`, `VIEW#Front`, `MAT#CaseBody` 等多個 DynamoDB Items。這是為了繞過 DynamoDB 單一 Item 400KB 的限制，是非常專業的 NoSQL 資料建模技巧。
        
    - **S3 預簽名 URL (Presigned URL):** 在 `aws_agent.py` 的 snapshot 處理中，使用了 `generate_presigned_url`。這讓 Remote Client 可以安全地讀取私有 S3 Bucket 的圖片，而不需要長期憑證。
        

#### 3. 配置平面 (Configuration Plane) - Template Sync

- **代碼對應:** `DB/templates/template_manager.py`, `data_manager/cloud_db.py` (`WatchTemplates`)
    
- **技術分析:**
    
    - **雲端作為 Single Source of Truth:** `WatchTemplates` 在雲端維護，Local 端只做 Cache。這解決了多站點（德國、紐約）檢測標準不一致的問題。只要在雲端更新 Template，所有站點同步後皆使用最新標準。
        

#### 4. 安全性 (Security)

- **代碼對應:** `logging_system/audit_logger.py`, `data_manager/encryption.py`
    
- **技術分析:**
    
    - **HMAC 簽章:** 指令 payload 包含 `sig` 與 `sig_ts`，防止中間人重放攻擊 (Replay Attack) 或竄改指令。
        
    - **Data-at-Rest Encryption:** 圖片 Header 被混淆 (Obfuscation) 且使用 AES-GCM 加密。這對於高價值資產（如 Rolex 原型機照片）的保密性至關重要。
        
    - **Audit Log Chain:** 稽核日誌使用上一筆的 Hash 進行串接 (Hash Chain)，確保日誌不可被篡改（Immutable Logs）。
        

---

### 第二部分：架構師的規劃與建議 (AWS Architecture Proposal)

基於上述代碼與需求，我建議採用以下 AWS 架構來支撐生產環境。

#### 1. 核心服務選型與理由

| **需求領域**        | **建議 AWS 服務**                                   | **架構師分析理由**                                                                                                                               |
| --------------- | ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **IoT 連線與通訊**   | **AWS IoT Core**                                | **必選。** 它是唯一能在大規模跨國部署中，安全穿越防火牆管理數千台設備的方案。提供 MQTT Broker、Device Shadow 與憑證管理 (X.509)。                                                      |
| **資料庫 (NoSQL)** | **Amazon DynamoDB**                             | **必選。** 你的資料結構 (`WatchView`, `Points`) 是高度巢狀且變動的 JSON。RDBMS (如 RDS) 處理這種 Schema 變更很痛苦，且 DynamoDB 的 Global Tables 可在德國與美國 Region 間進行毫秒級同步。 |
| **物件儲存**        | **Amazon S3**                                   | **必選。** 儲存 Raw Image 和 Analysis Report。配合 Lifecycle Policy (如 30 天後轉 Glacier) 可大幅降低成本。                                                    |
| **即時影像串流**      | **Amazon Kinesis Video Streams (KVS) (WebRTC)** | **建議。** 目前代碼是用 `client_binary` 跑 WebRTC。AWS KVS 提供託管的 TURN/STUN 伺服器，解決複雜的 NAT 穿透問題，這在跨國企業網路中是剛需。                                          |
| **遠端軟體更新**      | **AWS IoT Jobs** (配合 Systems Manager)           | **關鍵建議。** 單靠 MQTT 指令做 `git pull` 很危險且難以追蹤進度。IoT Jobs 專門處理 OTA (Over-The-Air) 更新，支援版本控制、灰度發布 (Canary Rollout) 和失敗回滾。                       |
| **稽核與分析**       | **Amazon Athena**                               | 你的 `AuditLogger` 會上傳 JSONL 到 S3。使用 Athena 可以直接用 SQL 查詢 S3 上的日誌，無需建立昂貴的 Log Server (如 ELK Stack)。                                          |

#### 2. 針對 "遠端軟體升級" (Software OTA) 的詳細設計

目前的 `force_sync` 或 `git pull` 邏輯在生產環境是不夠的。對於德國或紐約的機器，建議實作以下流程：

- **機制:** 使用 **AWS IoT Jobs**。
    
- **流程:**
    
    1. **Build:** 在 CI/CD (如 GitHub Actions) 建置完成後，將新的代碼打包 (Zip/Docker Image) 上傳至 S3。
        
    2. **Create Job:** 管理員在 AWS Console 建立一個 IoT Job document (JSON)，內容包含新版本的 S3 URL 和版本號。
        
    3. **Notify:** AWS IoT Core 推送通知給 Local Computer 的 `AWSAgent`。
        
    4. **Execute:** `AWSAgent` 收到 Job，下載套件，執行安裝腳本 (e.g., `pip install -r requirements.txt` 或 `docker-compose up -d`)。
        
    5. **Report:** 安裝完成後，Agent 回報狀態 (Succeeded) 給 AWS。
        
- **代碼修改建議:** `AWSAgent` 需要增加對 `jobs` topic 的訂閱與處理邏輯，而不僅僅是 `cmd` topic。
    

#### 3. 資料庫設計優化 (DynamoDB Schema Design)

你目前的 `CloudDatabaseManager` 設計已經很棒，針對 Split-Key 我有以下優化建議：

- **WatchAnalysisResults Table:**
    
    - **Partition Key (PK):** `watch_id` (e.g., `Rolex_Submariner_001`)
        
    - **Sort Key (SK):** 使用 **Hierarchical Key Design**。
        
        - `METADATA` (手錶基本資料)
            
        - `IMG#<Timestamp>#<StepName>` (原始圖索引)
            
        - `RES#<Timestamp>#<TaskName>` (分析結果)
            
        - `VIEW#<ViewName>#<Version>` (視圖資料)
            
    - **GSI (Global Secondary Index):** 建議建立一個以 `series_name` 為 PK 的 GSI，這樣你可以快速查詢 "所有 Submariner 系列的檢測結果"，而不僅僅是單一手錶。
        

#### 4. 全球部署架構圖 (文字描述)

Plaintext

```
[Remote Client (Web/App)]
       |
       v
[AWS Cloud (us-east-1 / eu-central-1)]
   |-- IoT Core (MQTT Broker) <-----> [IoT Jobs (OTA Update)]
   |-- DynamoDB (Global Tables) <---> [Data Sync]
   |-- S3 Bucket (Cross-Region Replication)
   |-- Kinesis Video Streams (WebRTC Signaling)
       ^
       | (MQTT / HTTPS / WebRTC)
       |
[Local Computer (Germany/NY)]
   |-- AWSAgent (Python Daemon)
   |-- WorkflowManager
   |-- Hardware Drivers (Camera/Stage)
   |-- Local SQLite (Buffer)
```

### 第三部分：針對代碼的具體改進建議 (Tech Lead Code Review)

作為 Tech Lead，我看這份代碼有幾個地方需要針對 "穩定性" 做增強：

1. **AWS Agent 的守護進程化 (Daemonizing):**
    
    - 目前 `aws_agent.py` 是一個 Python Script。在生產環境中，它應該被包裝成 **Systemd Service** (Linux) 或 **Windows Service**。
        
    - **理由:** 確保當程式崩潰或電腦重啟時，Agent 能自動重啟，否則遠端就失聯了。
        
2. **改善 Cloud Sync 的錯誤處理:**
    
    - 在 `CloudSyncManager` (`cloud_sync.py`) 中，如果 `upload_file` 失敗，目前的邏輯是 `time.sleep(0.5)` 然後重試。
        
    - **建議:** 實作 **Exponential Backoff (指數退避)**。如果網路長期中斷，固定的 sleep 會造成 CPU 空轉浪費。
        
3. **Audit Log 的雲端一致性:**
    
    - `AuditLogger` 目前是寫入本地檔案再由 `CloudSync` 上傳。
        
    - **建議:** 考慮將 Audit Log 直接串接 **AWS CloudWatch Logs** (透過 boto3)，或者保持現狀但增加 `flush` 機制，確保在程式崩潰前最後一筆 Log 能寫入磁碟。
        
4. **WebRTC 的整合:**
    
    - 目前 WebRTC 依賴外部 binary (`client_binary`).
        
    - **建議:** 如果 Python 整合 C++ SDK 有困難，可以考慮使用 **GStreamer** 配合 **Amazon Kinesis Video Streams Producer SDK for C++** 的 GStreamer plugin，這樣可以在 Python 中透過 `subprocess` 更穩定地控制串流，甚至疊加 AI 辨識結果在影像上 (Overlay)。
        

### 總結

你的系統代碼品質很高，已經具備了現代化 IoT Edge AI 的雛形 (Local First, Cloud Sync)。

- **雲端選擇:** AWS 是最佳解，特別是 AWS IoT Core + DynamoDB + S3 的組合，完美契合你的代碼邏輯。
    
- **關鍵下一步:**
    
    1. 實作 **AWS IoT Jobs** 邏輯來處理 `遠端軟體升級`。
        
    2. 將 `AWSAgent` 包裝為系統服務 (System Service) 以確保 24/7 在線。
        
    3. 開啟 DynamoDB 的 PITR (Point-in-time Recovery) 以防資料誤刪。
        

這個架構足以支撐從單一實驗室擴展到全球數百個檢測站點的規模。



#### 設定AWS S3的每一流程步驟step-by-step
```
我有AWS root user account現在在Console Home, 但我對AWS是初學者. 請一步步中文詳細解釋教我怎麼設定系統需要的Amazon S3 儲存 Raw Image 和 Analysis Report, 以及如何和我的系統做連結可以開始上傳到Amazon S3.
```
這是一個非常好的開始。身為 AWS 初學者，直接使用 **Root User (根帳號)** 來執行程式是非常危險的。

因此，我們的第一步不是直接去 S3，而是先建立一個「專門給程式用的帳號 (IAM User)」，這在 AWS 是標準的安全規範。

請跟隨以下 **4 個步驟**，我會帶您完成從權限設定到 S3 建立，最後修改您的 `system_config.yaml`。

---

### 第一步：建立程式專用的「鑰匙」 (IAM User)

我們需要產生一組 `Access Key ID` 和 `Secret Access Key` 給您的 Python 系統使用，而不是用您現在登入的 Root 帳號密碼。

1. **進入 IAM 控制台**：
    
    - 在 Console Home 上方的搜尋列輸入 **"IAM"**，點擊進入。
        
2. **建立使用者**：
    
    - 左側選單點擊 **Users (使用者)** -> 右上角 **Create user (建立使用者)**。
        
    - **User name**: 輸入 `WatchSystemAgent` (或您喜歡的名字)。
        
    - **Next (下一步)**。
        
3. **設定權限**：
    
    - 選擇 **"Attach policies directly" (直接連接政策)**。
        
    - 在下方的搜尋框輸入 `S3Full`。
        
    - 勾選 **`AmazonS3FullAccess`**。
        
        - _(註：這給予該帳號對所有 S3 的完全權限。為了初學者方便我們先這樣設，未來熟悉後可以限縮權限)_。
            
    - **Next (下一步)** -> **Create user (建立使用者)**。
        
4. **取得金鑰 (Access Keys)**：
    
    - 點擊剛建立的 `WatchSystemAgent` 名稱進入詳細頁面。
        
    - 點擊 **"Security credentials" (安全憑證)** 分頁。
        
    - 往下滑到 **Access keys** 區域，點擊 **Create access key (建立存取金鑰)**。
        
    - 選擇 **"Application running outside AWS"** (或是選 Other 也可以)，點擊 Next。
        
    - 點擊 **Create access key**。
        
    - **【非常重要】**：您會看到 `Access key ID` 和 `Secret access key`。請點擊 **"Download .csv"** 下載保存，或是複製這兩串亂碼存到記事本。
        
    - _注意：Secret key 只會出現這一次，關掉視窗後就看不到了。_
        

---

### 第二步：建立 S3 Bucket (水桶)

現在我們去建立存放圖片的「水桶」。

1. **進入 S3 控制台**：
    
    - 上方搜尋列輸入 **"S3"**，點擊進入。
        
2. **建立 Bucket**：
    
    - 點擊橘色的 **Create bucket (建立儲存貯體)** 按鈕。
        
3. **設定 Bucket 詳細資訊**：
    
    - **Bucket name**: 輸入 `watch-analysis-v1`。
        
        - _注意：S3 名字是全球唯一的。如果被別人用走了，請改為 `watch-analysis-[您的名字]-v1`，例如 `watch-analysis-trigeorge-v1`。_
            
    - **AWS Region**: 建議選擇 **`US East (N. Virginia) us-east-1`**。
        
        - _原因：您的程式碼 `config/system_config.yaml` 預設是 `us-east-1`，選這個可以少改很多設定。_
            
4. **其餘保持預設**：
    
    - Object Ownership: ACLs disabled (預設)。
        
    - Block Public Access: **勾選 Block all public access** (預設)。這很重要，確保您的圖片不會被公開在網路上。
        
5. **完成**：
    
    - 滑到最下方點擊 **Create bucket**。
        

---

### 第三步：將 AWS 金鑰放入您的電腦

您的 Python 系統 (使用 `boto3` library) 會自動去電腦的特定位置尋找金鑰。我們需要將第一步取得的金鑰放在那裡。

#### 如果您是用 Windows (Local Computer):

1. 打開檔案總管，進入您的使用者資料夾 `C:\Users\您的使用者名稱\`。
    
2. 建立一個新資料夾，命名為 `.aws` (注意前面有點)。
    
3. 進入 `.aws` 資料夾，建立一個文字檔，命名為 `credentials` (注意**沒有** .txt 副檔名)。
    
4. 用記事本打開 `credentials` 檔案，貼上以下內容：
    

Ini, TOML

```
[default]
aws_access_key_id = 貼上第一步取得的Access_Key_ID
aws_secret_access_key = 貼上第一步取得的Secret_Access_Key
region = us-east-1
```

5. 儲存並關閉。
    

_(如果您已經安裝了 AWS CLI 工具，可以在 cmd 輸入 `aws configure` 直接設定，效果一樣。)_

---

### 第四步：修改系統設定檔並連結

現在回到您的程式碼專案，我們需要確保 `config/system_config.yaml` 對應到正確的 S3 設定。

1. 打開 `config/system_config.yaml`。
    
2. 找到 `aws` 區塊，修改如下：
    

YAML

```
aws:
  # 把這裡改成您剛剛建立的 Bucket 名字
  # 如果剛剛 'watch-analysis-v1' 被用走了，這裡要填您取的新名字
  s3_bucket: "watch-analysis-v1" 
  
  # 確保這裡是 us-east-1 (如果您剛剛選了別區，這裡也要改)
  region: "us-east-1"
  
  # (以下暫時不用動，這是之後進階功能用的)
  dynamodb_table: "WatchAnalysisResults" 
  command_table: "WatchCommandLog"
  # ...
```

3. 找到 `security` 區塊，確保雲端同步功能有開啟：
    

YAML

```
security:
  enable_cloud_sync: true       # 確保這裡是 true
  force_mock_cloud_sync: false  # 確保這裡是 false (不要用模擬模式)
  # ...
```

---

### 第五步：測試連結

現在可以做一個簡單的測試，確認您的系統能否上傳檔案到 S3。

您可以在專案目錄下建立一個名為 `test_s3.py` 的檔案，貼上以下代碼來測試：

Python

```
import boto3
import yaml
import os

# 讀取設定檔
with open("config/system_config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

bucket_name = config['aws']['s3_bucket']
region = config['aws']['region']

print(f"正在測試連線到 Bucket: {bucket_name} ({region})...")

try:
    # 建立 S3 客戶端 (它會自動讀取我們在第三步設定的 .aws/credentials)
    s3 = boto3.client('s3', region_name=region)
    
    # 建立一個測試檔案
    with open("test_upload.txt", "w") as f:
        f.write("Hello AWS S3 from Watch System!")
    
    # 上傳檔案
    s3.upload_file("test_upload.txt", bucket_name, "Test/connection_check.txt")
    print("✅ 上傳成功！請去 AWS S3 Console 查看您的 Bucket 裡面是否有 'Test' 資料夾。")
    
    # 清理測試檔案
    os.remove("test_upload.txt")

except Exception as e:
    print("❌ 連線失敗。錯誤訊息：")
    print(e)
    print("\n請檢查：")
    print("1. C:\\Users\\您的名稱\\.aws\\credentials 檔案內容是否正確")
    print("2. config/system_config.yaml 裡的 bucket 名稱是否正確")
```

執行這個腳本。如果顯示 ✅ 上傳成功，恭喜您！您的系統已經與 AWS S3 成功連結。接下來當您執行 `main_cli.py` 進行拍照或分析時，`CloudSyncManager` 就會自動將 Raw Image 和 Analysis Report 上傳到這個 Bucket 了。



#### 如果保管AWS 金鑰
```
在第三步：將 AWS 金鑰放入您的電腦步驟, 無法把aws_access_key_id, aws_secret_access_key 用system)config 設定嗎? 因為如果要遠端安裝在local system, 或客戶不懂電腦 這樣是否會有問題使用AWS 金鑰
```
這是一個非常好的實務問題！您的擔憂完全正確：

1. **使用者體驗問題**：要求客戶或非技術人員手動去建立 `.aws` 資料夾並編輯檔案，出錯率極高。
    
2. **自動化部署問題**：如果要批量部署到遠端電腦，手動步驟是不可行的。
    

**答案是：可以的。** 您絕對可以將金鑰放在 `system_config.yaml`，然後在程式碼中讀取它。

但身為架構師，我必須提醒您一個**嚴重安全性風險**：**絕對不要將包含真實金鑰的 `system_config.yaml` 上傳到 GitHub 或任何版本控制系統。**

以下提供兩種解決方案，建議您根據場景選擇：

---

### 方案一：修改程式碼，直接從 Config 讀取金鑰 (最簡單，適合您的需求)

這是最直接解決「客戶不懂電腦」的方法。您將金鑰寫在設定檔中，程式碼讀取後直接傳給 `boto3`，這樣就不需要依賴作業系統的 `.aws` 資料夾。

#### 1. 修改 `config/system_config.yaml`

在 `aws` 區塊下新增金鑰欄位：

YAML

```
aws:
  s3_bucket: "watch-analysis-v1"
  region: "us-east-1"
  
  # [新增] 直接在這裡設定金鑰
  access_key_id: "AKIAxxxxxxxxxxxx"     # 填入您的 Access Key
  secret_access_key: "xxxxxxxxxxxxxxx"  # 填入您的 Secret Key
  
  # 其他設定...
```

#### 2. 修改程式碼 `data_manager/cloud_sync.py`

我們需要修改 `_make_s3_client` 函式，讓它優先從 Config 讀取金鑰。

找到 `data_manager/cloud_sync.py` 中的 `_make_s3_client` 方法，修改如下：

Python

```
    def _make_s3_client(self):
        if not HAS_BOTO3:
            return None
        try:
            # [修改開始] 優先從 Config 讀取金鑰
            aws_cfg = self.config.get("aws", {})
            ak = aws_cfg.get("access_key_id")
            sk = aws_cfg.get("secret_access_key")
            region = aws_cfg.get("region", "us-east-1")

            if ak and sk:
                # 如果 Config 有金鑰，直接使用，不依賴本機 .aws 檔案
                self.logger.info("Using AWS credentials from system_config.yaml")
                return boto3.client(
                    "s3",
                    region_name=region,
                    aws_access_key_id=ak,
                    aws_secret_access_key=sk
                )
            else:
                # Config 沒有金鑰，才嘗試讀取本機 .aws/credentials
                return boto3.client("s3", region_name=self.region)
            # [修改結束]

        except Exception as e:
            self.logger.error(f"Failed to create S3 client: {e}")
            return None
```

**同樣的邏輯也需要應用到 `cloud_relay/aws_agent.py`** (如果有用到 boto3 的話)，或是任何直接呼叫 `boto3.client` 的地方。

---

### 方案二：提供「一鍵安裝腳本」 (最專業，推薦用於量產)

如果您不想把金鑰明碼寫在 `system_config.yaml` (怕不小心 git push 出去)，但又不想讓客戶手動設定。您可以寫一個 `setup.py` 腳本。

當您要把軟體給客戶時，您給他們一個 `install_credentials.py`，客戶只要點兩下執行，這個腳本就會自動幫他們建立 `.aws/credentials` 檔案。

#### 建立 `install_credentials.py`

Python

```
import os
import sys
from pathlib import Path

def setup_aws_credentials():
    print("=== AWS 金鑰自動設定工具 ===")
    
    # 這裡可以由您預先填好，或者讓客戶輸入
    # 如果是發給客戶的包，您可以直接寫死在這裡，然後編譯成 .exe
    access_key = input("請輸入 AWS Access Key ID (或直接按 Enter 使用預設值): ").strip() or "您的預設AK"
    secret_key = input("請輸入 AWS Secret Access Key (或直接按 Enter 使用預設值): ").strip() or "您的預設SK"
    
    # 1. 找到使用者的家目錄 (Windows/Linux/Mac 通用)
    home_dir = Path.home()
    aws_dir = home_dir / ".aws"
    
    # 2. 建立 .aws 資料夾
    if not aws_dir.exists():
        print(f"建立目錄: {aws_dir}")
        aws_dir.mkdir(parents=True, exist_ok=True)
        
    # 3. 寫入 credentials 檔案
    cred_file = aws_dir / "credentials"
    
    content = f"""[default]
aws_access_key_id = {access_key}
aws_secret_access_key = {secret_key}
region = us-east-1
"""
    
    try:
        with open(cred_file, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ 成功！金鑰已寫入: {cred_file}")
        print("系統現在可以連線 AWS 了。")
    except Exception as e:
        print(f"❌ 失敗: {e}")

if __name__ == "__main__":
    setup_aws_credentials()
    input("\n按 Enter 鍵結束...")
```

---

### 架構師建議總結

考慮到您提到「遠端安裝」與「客戶不懂電腦」：

1. **短期/開發階段**：使用 **方案一**。
    
    - 直接修改 `config/system_config.yaml` 和程式碼。
        
    - **關鍵動作**：請務必將 `config/system_config.yaml` 加入 `.gitignore`，或者建立一個 `config/system_config.template.yaml` (不含金鑰) 上傳 Git，而含金鑰的真實檔案只存在於部署機器上。
        
2. **長期/生產環境 (Production)**：
    
    - 不建議將 Access Key/Secret Key 發送給客戶端。
        
    - 真正的企業級做法是使用 **AWS IoT Certificate (憑證)** 進行認證，然後透過 AWS IoT 的 "Credential Provider" 功能來換取暫時的 S3 上傳權限。
        
    - 這比較複雜，**目前建議您先採用方案一**，讓系統先跑起來，日後資安需求提高時再升級架構。
        

您可以先依照 **方案一** 修改您的代碼，這樣部署時只要把 `yaml` 檔丟過去就搞定了。




#### 設定AWS DynamoDB的每一流程步驟step-by-step
```
我有AWS root user account現在在Console Home, 但我對AWS是初學者. 請一步步中文詳細解釋教我怎麼設定系統需要的所有Amazon DynamoDB 包括WatchTemplates, WatchAnalysisResults, WatchCommandLog和其他需要用到的地方, 以及如何和我的系統做連結並對這些DynamoDB做相應的setting.
```
好的，沒問題！繼 S3 之後，我們現在來設定系統的核心資料庫 **Amazon DynamoDB**。

由於您的系統設計非常完整（包含了權限管理、指令控制、分析結果存檔），我們需要建立 **4 個 Table (資料表)**。

請跟隨以下步驟，我會帶您在 AWS Console 上逐一建立，並講解每個設定的原因。

---

### 第一步：進入 DynamoDB 控制台

1. 登入 AWS Console。
    
2. 在上方搜尋列輸入 **"DynamoDB"**，點擊進入。
    
3. 點擊左側選單的 **"Tables" (資料表)**。
    
4. 點擊橘色的 **"Create table" (建立資料表)** 按鈕。
    

---

### 第二步：逐一建立 4 個資料表

我們需要分別建立 `WatchAnalysisResults`, `WatchCommandLog`, `WatchTemplates`, `WatchUsers`。請依照下方表格的數值填寫：

#### 1. 建立 `WatchAnalysisResults` (儲存分析結果與圖片索引)

這是資料量最大的表，建議設定自動擴展。

- **Table name (資料表名稱)**: `WatchAnalysisResults`
    
- **Partition key (分割區索引鍵)**: `watch_id` (String / 字串)
    
- **Sort key (排序索引鍵)**: `sort_key` (String / 字串)
    
- **Table settings (資料表設定)**: 選擇 **Customize settings (自訂設定)**
    
- **Capacity mode (容量模式)**: 選擇 **On-demand (隨需)**
    
    - _原因：隨需模式雖然單價稍高，但適合流量不穩定的 IoT 場景（例如半夜沒人檢測），且完全不用管理容量，對初學者最友善。_
        
- 點擊 **Create table**。
    

#### 2. 建立 `WatchCommandLog` (儲存遠端指令狀態)

這是控制平面，需要設定 TTL (自動過期刪除) 以節省空間。

- **Table name**: `WatchCommandLog`
    
- **Partition key**: `device_id` (String)
    
- **Sort key**: `command_id` (String)
    
- **Table settings**: 選擇 **Customize settings** -> **On-demand**
    
- 點擊 **Create table**。
    

**(建立後的重要步驟：設定 TTL)** 這張表會累積很多指令紀錄，我們希望系統自動刪除 7 天前的舊紀錄。

1. 建立完成後，點擊列表中的 `WatchCommandLog` 進入詳細頁。
    
2. 點擊 **"Additional settings" (其他設定)** 分頁。
    
3. 找到 **"Time to Live (TTL)"** 區塊，點擊 **Turn on (開啟)**。
    
4. **TTL attribute name**: 輸入 `ttl`。
    
5. 點擊 **Turn on TTL**。
    

#### 3. 建立 `WatchTemplates` (儲存檢測模板)

- **Table name**: `WatchTemplates`
    
- **Partition key**: `series_name` (String)
    
- **Sort key**: `seriesversion` (String)
    
    - _注意：這裡的 Sort key 名稱與其他表不同，請務必填寫 `seriesversion`。_
        
- **Table settings**: 選擇 **Customize settings** -> **On-demand**
    
- 點擊 **Create table**。
    

#### 4. 建立 `WatchUsers` (使用者帳號與權限)

- **Table name**: `WatchUsers`
    
- **Partition key**: `username` (String)
    
- **Sort key**: (留空，不需要)
    
- **Table settings**: 選擇 **Customize settings** -> **On-demand**
    
- 點擊 **Create table**。
    

---

### 第三步：連結系統與測試

只要您之前在 **S3 設定步驟** 中已經完成了 `config/system_config.yaml` 的金鑰設定（填入了 `access_key_id` 和 `secret_access_key`），您的系統現在**已經自動連結**到這些 DynamoDB 了！

這是因為您的 `data_manager/cloud_db.py` 程式碼已經寫好會自動去讀取那些設定。

我們來做一個簡單的 Python 腳本測試，確認這 4 張表都能正常寫入。請在專案目錄下建立 `test_dynamo.py`：

Python

```
import boto3
import yaml
import time

# 1. 讀取設定
with open("config/system_config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

aws_cfg = config.get("aws", {})
region = aws_cfg.get("region", "us-east-1")
ak = aws_cfg.get("access_key_id")
sk = aws_cfg.get("secret_access_key")

print(f"正在連線到 DynamoDB (Region: {region})...")

try:
    # 2. 建立連線
    db = boto3.resource(
        "dynamodb",
        region_name=region,
        aws_access_key_id=ak,
        aws_secret_access_key=sk
    )

    # 3. 測試寫入 WatchUsers (最簡單的表)
    table_users = db.Table("WatchUsers")
    
    print("正在寫入測試資料到 WatchUsers...")
    table_users.put_item(Item={
        "username": "test_admin",
        "role": "admin",
        "created_at": str(int(time.time())),
        "note": "這是連線測試，可刪除"
    })
    
    # 4. 測試讀取
    resp = table_users.get_item(Key={"username": "test_admin"})
    item = resp.get("Item")
    
    if item and item['username'] == 'test_admin':
        print("✅ 成功！DynamoDB 讀寫功能正常。")
        print(f"讀取到的資料: {item}")
        
        # 5. 清理測試資料
        table_users.delete_item(Key={"username": "test_admin"})
        print("測試資料已清理。")
    else:
        print("❌ 寫入似乎成功，但讀取失敗。")

except Exception as e:
    print("❌ 連線失敗。錯誤訊息：")
    print(e)
    print("\n請檢查：")
    print("1. config/system_config.yaml 的 Table 名稱是否與 AWS 上一致")
    print("2. IAM User 是否有 AmazonDynamoDBFullAccess 權限 (如果之前只給了 S3，需要再去 IAM 加開 DynamoDB 權限)")
```

### 常見問題：權限不足

如果在執行測試時出現 `AccessDeniedException`，代表您之前建立的 IAM User (`WatchSystemAgent`) 只有 S3 的權限，沒有 DynamoDB 的權限。

**解決方法：**

1. 回到 AWS Console -> **IAM** -> **Users**。
    
2. 點擊 `WatchSystemAgent`。
    
3. 點擊 **Add permissions** -> **Attach policies directly**。
    
4. 搜尋 `DynamoDB`。
    
5. 勾選 **`AmazonDynamoDBFullAccess`**。
    
6. 點擊 **Next** -> **Add permissions**。
    

這樣您的系統就具備了完整的資料庫存取能力了！





#### 設定AWS IoT Core的每一流程步驟step-by-step
```
我有AWS root user account現在在Console Home, 但我對AWS是初學者. 請一步步中文詳細解釋教我怎麼設定系統需要的所有AWS IoT Core能實現控制平面 (Control Plane) - AWS IoT & Command Log 以及如何和我的系統做連結對這些AWS IoT Core做相應的setting.
```
這是一個非常關鍵的步驟。**AWS IoT Core** 是您系統的中樞神經，它負責讓遠端（Remote Client）和本地端（Local Computer）進行即時的「對話」。

不同於 S3 或 DynamoDB 只需要 Access Key，**IoT Core 需要「數位憑證 (Certificates)」**。這就像是給您的機器發一張「身分證」，只有持有這張身分證的機器才能連線。

請跟隨以下 **5 個步驟**，完成設定並連結您的 Python 系統。

---

### 第一步：進入 IoT Core 並建立「物件 (Thing)」

在 AWS 中，每一個連線的實體裝置（您的 Local Computer）都被稱為一個 **Thing**。

1. **進入控制台**：
    
    - 在 AWS Console 搜尋列輸入 **"IoT Core"**，點擊進入。
        
2. **建立 Thing**：
    
    - 左側選單選擇 **Manage (管理)** -> **All devices (所有裝置)** -> **Things (物件)**。
        
    - 點擊橘色的 **Create things (建立物件)**。
        
3. **設定流程**：
    
    - 選擇 **Create single thing (建立單一物件)** -> **Next**。
        
    - **Thing name (物件名稱)**：請輸入 `Rolex_Station_001`。
        
        - _注意：這個名字必須跟您 `config/system_config.yaml` 裡的 `device_id` 完全一樣。_
            
    - 其餘保持預設 (No Shadow, etc.) -> **Next**。
        
4. **設定憑證**：
    
    - 選擇 **Auto-generate a new certificate (自動產生新憑證)** (這是最簡單的選項)。
        
    - 點擊 **Next**。
        

---

### 第二步：建立並綁定「策略 (Policy)」

有了身分證 (Certificate)，還需要權限 (Policy) 才能通行。我們現在要建立一個允許裝置「說話 (Publish)」和「聆聽 (Subscribe)」的規則。

1. 在剛剛的畫面中，您會看到 **Policies (政策)** 的區塊。
    
2. 點擊右側的 **Create policy (建立政策)** 按鈕 (會開新分頁)。
    
3. **Policy properties**：
    
    - **Policy name**: 輸入 `WatchSystem_FullAccess`。
        
4. **Policy document (權限設定)**：
    
    - 選擇 **JSON** 模式 (比較快)，複製貼上以下內容：
        
    
    JSON
    
    ```
    {
      "Version": "2012-10-17",
      "Statement": [
        {
          "Effect": "Allow",
          "Action": "iot:*",
          "Resource": "*"
        }
      ]
    }
    ```
    
    - _註：這是給開發用的全開權限。生產環境我們會限制它只能訂閱特定的 Topic。_
        
5. 點擊 **Create (建立)**。
    
6. **回到上一個分頁 (建立 Thing 的精靈)**：
    
    - 勾選剛剛建立的 `WatchSystem_FullAccess`。
        
    - 點擊 **Create thing (建立物件)**。
        

---

### 第三步：【關鍵】下載憑證與金鑰

這是最重要的一步！AWS 會彈出一個視窗讓您下載憑證。**這些檔案只會出現這一次，關掉就沒了。**

請在您的專案資料夾中，建立一個路徑：`專案目錄/config/certs/`。然後下載以下檔案並改名放入：

1. **Device certificate (裝置憑證)**：
    
    - 下載畫面上的 `xxxxxx-certificate.pem.crt`。
        
    - **改名存為**：`certificate.pem.crt` -> 放入 `config/certs/`。
        
2. **Private key (私有金鑰)**：
    
    - 下載畫面上的 `xxxxxx-private.pem.key`。
        
    - **改名存為**：`private.pem.key` -> 放入 `config/certs/`。
        
3. **Root CA certificates (根憑證)**：
    
    - 點擊 "RSA 2048 bit key: Amazon Root CA 1" 的下載連結。
        
    - **改名存為**：`AmazonRootCA1.pem` -> 放入 `config/certs/`。
        
4. (Public key 不需要下載)。
    

完成後，點擊 **Done**。

---

### 第四步：取得連線網址 (Endpoint)

您的 Python 程式需要知道要連線到哪個伺服器。

1. 在 AWS IoT Core 左側選單，點擊最下方的 **Settings (設定)**。
    
2. 在 **Device data endpoint (裝置資料端點)** 區塊。
    
3. 複製那個網址，看起來像：`xxxxxxxxxxxxx-ats.iot.us-east-1.amazonaws.com`。
    

---

### 第五步：修改系統設定並連結

現在回到您的程式碼，修改 `config/system_config.yaml` 讓 `AWSAgent` 讀取正確的憑證。

1. 打開 `config/system_config.yaml`。
    
2. 找到 `network` 區塊，修改如下：
    

YAML

```
network:
  app_mode: "aws_iot"  # 確保模式是 aws_iot
  role: "agent"        # 如果是在 Local Computer 執行，這裡是 agent
  
  # [填入第四步取得的網址]
  aws_endpoint: "xxxxxxxxxxxxx-ats.iot.us-east-1.amazonaws.com"
  
  # [填入第一步設定的名稱]
  device_id: "Rolex_Station_001" 

  # 憑證路徑 (確保檔案名稱跟第三步下載改名的一致)
  certs:
    root_ca: "config/certs/AmazonRootCA1.pem"
    private_key: "config/certs/private.pem.key"
    cert_file: "config/certs/certificate.pem.crt"
  
  # Topic 設定 (保持預設即可，這些是我們在代碼裡定義好的通訊頻道)
  topics:
    command: "cmd/{device_id}"
    status: "status/{device_id}"
    shadow_update: "$aws/things/{device_id}/shadow/update"
```

---

### 最終測試：驗證「控制平面」

我們現在要模擬「從雲端發送指令」，看您的電腦是否收得到。

**1. 啟動您的 Agent 程式** 在終端機執行：

Bash

```
python cloud_relay/aws_agent.py
```

如果成功，您應該會看到類似 `Connecting to AWS IoT Core... Connected!` 的訊息。

**2. 使用 AWS MQTT Test Client 監聽狀態**

1. 回到 AWS IoT Core 控制台。
    
2. 左側選單點擊 **MQTT test client (MQTT 測試用戶端)**。
    
3. 在 "Subscribe to a topic" 分頁，輸入 `status/Rolex_Station_001`。
    
4. 點擊 **Subscribe (訂閱)**。
    
    - _這時如果您重啟 Python 程式，您應該會在這裡看到一條 JSON 訊息顯示 `status: idle`，代表連線成功！_
        

**3. 發送測試指令**

1. 點擊 **"Publish to a topic" (發布至主題)** 分頁。
    
2. **Topic name**: 輸入 `cmd/Rolex_Station_001`。
    
3. **Message payload**: 輸入以下測試指令 (模擬 force_sync)：
    
    JSON
    
    ```
    {
      "command_id": "test_cmd_001",
      "action": "force_sync",
      "requester": "console_admin"
    }
    ```
    
4. 點擊 **Publish**。
    

**4. 觀察結果**

- **在您的終端機 (Python)**：應該會看到 `Received action=force_sync, command_id=test_cmd_001`，並且開始執行同步。
    
- **在 AWS Console (Subscribe 分頁)**：應該會收到 `status` 回報，顯示 `state: running` 然後變為 `completed`。
    

**恭喜！** 您已經成功建立了完整的 **AWS IoT 控制平面**。現在您可以透過雲端遙控位於世界任何角落的檢測機台了。