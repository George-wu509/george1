


|                           |     |
| ------------------------- | --- |
| [[#### 從頭開始AWS很多服務]]      |     |
| [[#### 需要建立 Access Key嗎]] |     |
|                           |     |
|                           |     |
|                           |     |



#### 從頭開始AWS很多服務
```
新的公司上司申請了AWS root account, 並給我開了IAM account以及一些權限. 我的工作開發軟體會需要用到AWS S3存放很多images跟結果, AWS IoT Core讓數個remote client可以操控數個local agent也用到MQTT, Amazon DynamoDB可以儲存軟體用到的四個Database(Templates table, images and results table, command table, account DB)並跟SQLite同步. 也用到AWS IoT Jobs做到遠端軟體升級, 以及用Amazon Athena管理AufitLogger.

我需要用那個新申請的有限權限IAM account在新開的
```

這是一個非常典型的從 0 到 1 建置 AWS 架構的過程。由於你提到你擁有的是「有限權限的 IAM Account」且「預設不懂 AWS」，我們的首要任務是確認權限，然後透過 AWS 管理控制台 (Management Console) — 也就是網頁介面 — 來一步步建立這些資源。

這份指南分為六大步驟。在開始之前，請先確認你已經可以登入 AWS Console (你的上司應該給了你一個登入網址、使用者名稱和密碼)。

---

### 第一步：權限確認與開發環境 (最重要)

因為你的帳號權限有限，在開始之前，你必須確認上司是否賦予了你足夠的權限去**創建**這些資源，而不僅僅是**使用**它們。

1. **權限檢核表：** 請將此清單傳給你的上司，確認你的 IAM User 擁有以下策略 (Policies) 或對等的權限：
    
    - **AmazonS3FullAccess** (創建與管理 Bucket)
        
    - **AWSIoTFullAccess** (創建 Thing, Policy, Jobs)
        
    - **AmazonDynamoDBFullAccess** (創建 Table)
        
    - **AmazonAthenaFullAccess** (查詢與管理 Workgroup)
        
    - (選用) **IAMReadOnlyAccess** (讓你能在 IoT Core 中查看或附加 Role，如果需要的話)
        
2. **取得 Access Keys (給寫程式用)：**
    
    - 登入 AWS Console，右上角點擊你的帳號名稱 -> **Security credentials (安全憑證)**。
        
    - 找到 **Access keys** 區塊，點擊 **Create access key**。
        
    - 選擇 **Local code** (本地代碼)，勾選確認，然後點擊 Create。
        
    - **務必下載 .csv 檔案**，裡面有 `Access Key ID` 和 `Secret Access Key`。這是你的軟體 (Boto3 或 AWS SDK) 連線 AWS 的鑰匙，弄丟了就無法找回，只能重開。
        

---

### 第二步：設定 Amazon S3 (儲存圖片與結果)

S3 是物件儲存服務，類似雲端硬碟，但透過 API 操作。

1. 在 AWS Console 上方搜尋欄輸入 **S3** 並進入。
    
2. 點擊橘色的 **Create bucket**。
    
3. **Bucket name (名稱)：** 取一個全球唯一的名稱 (例如：`companyname-app-data-prod`)。注意：S3 名稱全網通用，不能跟別人重複。
    
4. **Region (區域)：** 選擇離你或客戶最近的區域 (例如 `us-east-1` 或 `ap-northeast-1`)。**請記住這個區域，後續所有服務都要選同一個。**
    
5. **Object Ownership：** 保持預設 (ACLs disabled)。
    
6. **Block Public Access settings for this bucket：** **務必勾選 Block all public access**。你的軟體會透過 Access Key (IAM 權限) 存取，不需要公開給大眾。
    
7. 點擊底部的 **Create bucket**。
    
8. (建議) 重複上述步驟，如果你想將「軟體升級檔(Firmware)」與「使用者圖片」分開，可以建兩個 Bucket。
    

---

### 第三步：設定 Amazon DynamoDB (資料庫)

DynamoDB 是 NoSQL 資料庫，你需要建立四張表。AWS 的 NoSQL 沒有 "Database" 的概念，只有 "Table"。

1. 搜尋並進入 **DynamoDB**。
    
2. 點擊側邊欄的 **Tables** -> **Create table**。
    
3. **Table 1: Templates**
    
    - Table name: `Templates`
        
    - **Partition key (主鍵):** 輸入你軟體設計的唯一識別碼，例如 `template_id` (String)。
        
    - **Sort key (排序鍵 - 選用):** 如果你需要複合查詢，例如「同一類別下的不同版本」，可以設 `version`。如果不需要，留空即可。
        
    - Capacity settings: 選 **On-demand** (按需付費)。開發初期流量不穩，這模式最省錢且不用管理效能。
        
    - 點擊 **Create table**。
        
4. **重複步驟建立其他三張表** (`ImagesResults`, `Commands`, `Accounts`)，重點是**確認好每一個 Table 的 Partition Key 是什麼** (這必須跟你 SQLite 的設計對應)。
    

---

### 第三步：設定 AWS IoT Core (MQTT 通訊核心)

這是最複雜的一步，這是讓你的 Remote Client 和 Local Agent 溝通的橋樑。

**3.1 建立 Policy (權限策略)** 首先要規定連上來的設備「能做什麼」。

1. 搜尋並進入 **IoT Core**。
    
2. 側邊欄選 **Security** -> **Policies** -> **Create policy**。
    
3. Name: `App_Device_Policy` (範例)。
    
4. **Policy document:** 為了開發方便，你可以先設寬鬆一點 (生產環境需緊縮)。
    
    - Policy effect: **Allow**
        
    - Policy action: `iot:*` (代表允許所有 IoT 動作，如 Publish, Subscribe, Connect)
        
    - Policy resource: `*`
        
5. 點擊 **Create**。
    

**3.2 建立 Thing (你的裝置)** 每一個 Local Agent 或 Remote Client 在 AWS 裡都對應一個 "Thing"。

1. 側邊欄選 **Manage** -> **All devices** -> **Things** -> **Create things**。
    
2. 選 **Create single thing** -> Next。
    
3. **Thing name:** 輸入裝置名稱 (例如 `LocalAgent_01`)。此名稱通常會用作 MQTT 的 Client ID。 -> Next。
    
4. **Device certificate:** 選 **Auto-generate a new certificate** (最簡單) -> Next。
    
5. **Attach policies:** 勾選剛剛建立的 `App_Device_Policy` -> **Create thing**。
    
6. **下載憑證 (關鍵步驟)：**
    
    - 畫面會跳出下載連結。你**必須**下載：
        
        - **Device Certificate** (`xxx-certificate.pem.crt`)
            
        - **Private Key** (`xxx-private.pem.key`)
            
        - **Amazon Root CA 1** (這是公開的，用來驗證 AWS 伺服器)
            
    - _注意：Public Key 可以不用下載。_
        
    - **這三個檔案要存放在你的軟體資料夾中**，Python 連線 MQTT 時需要路徑指向它們。
        
7. 重複此步驟為你的 Remote Client 也建立一個 Thing (或共用憑證，但建議分開)。
    

**3.3 取得連線端點 (Endpoint)**

1. 在 IoT Core 左側選單最下方點 **Settings**。
    
2. 複製 **Device data endpoint** (格式如 `xxx-ats.iot.us-east-1.amazonaws.com`)。這就是你軟體 MQTT Broker 的網址。
    

---

### 第四步：設定 AWS IoT Jobs (軟體升級)

IoT Jobs 不需要像 Database 那樣「建立」一個實體，它是一個功能。你需要在 S3 準備好更新檔，並確認裝置有權限下載。

1. **準備更新檔：** 將你的軟體更新包 (例如 `update_v1.zip`) 上傳到你在第二步建立的 S3 Bucket。
    
2. **IAM Role (給 Job 用)：**
    
    - 因為你權限有限，這步若卡住需找上司。你需要一個 IAM Role 允許 IoT Jobs 去 S3 讀取檔案 (Role 需包含 `AmazonS3ReadOnlyAccess`)。
        
3. **使用流程 (開發軟體時的邏輯)：**
    
    - 你不需在 Console 按什麼按鈕來「啟動」服務。
        
    - 你的後端程式碼會呼叫 AWS SDK (boto3) 的 `create_job` API。
        
    - 你的 Local Agent (設備端) 會訂閱 MQTT Topic `$aws/things/{thingName}/jobs/notify` 來接收更新通知。
        

---

### 第五步：設定 Amazon Athena (日誌審計)

Athena 是用來「查詢放在 S3 裡的檔案」的工具。假設你的 AuditLogger 已經把 log 存成 `.json` 或 `.csv` 丟到 S3 了。

1. **建立儲存查詢結果的 Bucket：**
    
    - Athena 執行查詢時，會產生 CSV 結果檔，它需要一個地方放。
        
    - 去 S3 建立一個新資料夾或 Bucket，例如 `company-athena-results`。
        
2. **設定 Athena：**
    
    - 搜尋並進入 **Athena**。
        
    - 點擊 **Query editor**。
        
    - 點擊 **Settings** -> **Manage**。
        
    - **Query result location:** 選擇剛剛建的 S3 路徑 (例如 `s3://company-athena-results/`) -> Save。
        
3. **建立 Database 與 Table (對應你的 Log)：**
    
    - 在 Query Editor 中，你需要寫 SQL 來定義你的 Log 長什麼樣子。
        
    - 範例 (假設你的 Log 是 JSON 格式)：
        
    
    SQL
    
    ```
    CREATE DATABASE audit_db;
    ```
    
    - 然後建立 Table 指向你的 Log S3 Bucket：
        
    
    SQL
    
    ```
    CREATE EXTERNAL TABLE IF NOT EXISTS audit_db.logs (
      timestamp string,
      user_id string,
      action string,
      details string
    )
    ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
    LOCATION 's3://company-app-data-prod/audit-logs/';
    ```
    
    - 按下 **Run**。如果成功，你就可以用 SQL 語法 `SELECT * FROM audit_db.logs` 來查 Log 了。
        

---

### 總結開發者需要的參數

完成以上步驟後，你需要將下列資訊寫入你的軟體設定檔 (Config) 中：

1. **AWS Credentials:** Access Key ID, Secret Access Key.
    
2. **Region:** (例如 `us-east-1`).
    
3. **S3:** Bucket Names.
    
4. **DynamoDB:** Table Names.
    
5. **IoT Core:**
    
    - Endpoint URL.
        
    - 路徑指向: Certificate file, Private Key file, Root CA file.
        
    - Thing Name.
        

現在你的基礎建設已經搭建完成，可以開始寫 Python/Boto3 代碼來串接這些服務了。如果在建立過程中看到紅色錯誤訊息 "User is not authorized to perform: ...", 請直接截圖給你的上司，請求開放對應權限。




#### 需要建立 Access Key嗎
```
那在進行這些步驟之前, 我需要我的剛開啟的IAM有在Security credentials裡面create access key嗎? 這是做甚麼的呢? 有甚麼我之前提到的服務或任務是需要access key才能做到的? 所以access key是我專屬的還是我開發的軟體專屬的或者AWS帳戶專屬的? 請中文詳細解釋
```
這是一個非常關鍵且核心的問題。不用擔心，這些觀念對於初次接觸 AWS 的人來說確實容易混淆。我會將「網頁操作」與「程式開發」分開來講解。

簡單的回答是：**為了在網頁上建立服務（上一篇回答的 Step 2~5），你「不需要」Access Key；但是為了讓你開發的軟體能運作，你「絕對需要」Access Key。**

以下詳細解釋每一個細節：

### 1. Access Key 到底是做什麼的？

你可以把 AWS 的驗證方式想像成進公司的兩種方式：

1. **使用者名稱 + 密碼 (Console Access)：** 這是給**人類**用的。
    
    - 當你要打開瀏覽器，登入 AWS 網頁介面去「點擊按鈕」建立 S3 Bucket 或 DynamoDB Table 時，你用的是這組。
        
2. **Access Key ID + Secret Access Key (Programmatic Access)：** 這是給**程式/機器**用的。
    
    - 你的 Python 程式碼不會「打開瀏覽器輸入密碼」，它需要一種方式告訴 AWS：「嘿，我是那個 IAM User，我有權限上傳圖片」。這個「身分證」就是 Access Key。
        
    - 它由兩部分組成：
        
        - **Access Key ID:** 類似帳號 (公開一點沒關係，例如 `AKIAIOSFODNN7EXAMPLE`)。
            
        - **Secret Access Key:** 類似密碼 (絕對不能給別人看，很長一串亂碼)。
            

### 2. 在進行哪些步驟前需要它？

- **在 AWS 網頁上設定服務時 (上一篇的步驟)：** **不需要**。你只需要用上司給你的帳號密碼登入網頁即可。
    
- **當你開始寫第一行 Code 連接 AWS 時：** **需要**。
    

**建議：** 雖然設定服務時不用，但我建議你**現在就先去申請下來並存好**（存成 .csv 檔）。因為一旦你在寫程式時卡住，或者要測試連線，馬上就會用到。

### 3. 你提到的服務中，哪些任務需要 Access Key？

基本上，只要是你的「軟體後端」或「管理端程式」要執行的動作，全部都需要 Access Key。

具體對應你提到的功能：

- **S3 (存放 Images/結果)：**
    
    - 你的軟體要執行 `upload_file` (上傳圖片) 或 `download_file` (下載結果) 時，AWS SDK (Boto3) 會檢查你的環境變數裡有沒有 Access Key。如果沒有，程式會直接報錯 `NoCredentialsError`。
        
- **DynamoDB (資料庫同步)：**
    
    - 當你的軟體要把 SQLite 的資料 `put_item` (寫入) 到 AWS DynamoDB 時，需要 Access Key 來證明你有寫入權限。
        
- **AWS IoT Core (管理端)：**
    
    - **注意區分：**
        
        - **你的 Local Agent (設備端)：** 通常是用 **憑證 (Certificates)** 來連線 MQTT，**不使用** Access Key（為了安全）。
            
        - **你的控制軟體 (發送指令端)：** 如果你有一個後台軟體要發送 MQTT 訊息給設備，或者要透過 API 查詢設備狀態，這個軟體需要 Access Key。
            
- **AWS IoT Jobs (遠端升級)：**
    
    - 你的軟體要「建立一個新的升級任務 (Create Job)」時，是透過 API 呼叫的，這需要 Access Key。
        
- **Amazon Athena (AuditLogger)：**
    
    - 當你的軟體要送出 SQL 語法去查詢 Log 時，這是一個 API 請求，必須附帶 Access Key。
        

### 4. Access Key 是專屬於誰的？

這是一個層級關係，理解這點對資安很重要：

1. **AWS Root Account (公司帳號)：** 最上層。
    
2. **IAM User (你的員工帳號)：** 上司開給你的那個帳號。
    
3. **Access Key (你的程式鑰匙)：** **它是專屬於「你的 IAM User」的。**
    

**詳細釐清：**

- **不是 AWS 帳戶專屬：** 一個 AWS Root Account 底下可以有 100 個員工 (IAM Users)，每個員工都可以有自己的 Access Key。
    
- **不是軟體專屬 (但由軟體使用)：** 實際上，Access Key 是綁定在你這個「人(IAM User)」身上的。但是，你會把這把鑰匙「借給」你開發的軟體使用。
    
    - 當軟體使用這把 Key 去存取 S3 時，AWS 會視為「**是你 (IAM User) 在操作**」。
        
    - 如果你的 IAM User 被刪除了，這把 Key 也會立刻失效，軟體就會壞掉。
        

### 總結操作建議

1. **現在：** 先用帳號密碼登入 AWS Console，把所有服務 (S3, DynamoDB 等) 建置好。
    
2. **接著：** 去 **Security Credentials** 建立 Access Key，下載 `.csv` 檔。
    
3. **開發時：** 在你的電腦上安裝 AWS CLI (`aws configure`) 或在程式碼的設定檔 (`config.py` 或 `.env`) 中填入這組 Key。
    
4. **注意：** 永遠不要把 Access Key 直接寫死在程式碼裡上傳到 GitHub (這叫 Hardcoding)，這非常危險。最好放在環境變數或獨立的設定檔中。