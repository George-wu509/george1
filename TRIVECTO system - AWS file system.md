
|                                           |          |
| ----------------------------------------- | -------- |
| [[#### S3 檔案結構跟DB是否需要更新 - thinkpad]]      | thinkpad |
| [[#### S3 檔案結構跟DB是否需要更新 - laptop]]        | laptop   |
| [[#### 多個branch有各自的local database. 怎麼合併]] | laptop   |
| [[#### [ AWS ] 分析S3與資料庫架構 (laptop)  ]]    | laptop   |
|                                           |          |
|                                           |          |
|                                           |          |
|                                           |          |
|                                           |          |
|                                           |          |


#### S3 檔案結構跟DB是否需要更新 - thinkpad
```
在目前的App main.py當在watchentry scan完一個watch的所有views包含front, back....會將所有image files存放在<watchentry id>的Raw folder, 而用./tasks/裡面對每個images分析的task service產生的結果包括image results跟json files也會存在<watchentry id>的Analysis folder, 同時這些raw folder跟Analysis folder的所有images, results也會將資訊存到local database. 而這些Raw folder跟Analysis folder裡面的images, files也會上傳到AWS S3儲存, local DB也會跟AWS DynamoDB同步. 請幫我整理這些Raw folder跟Analysis folder裡面的images, files上傳到AWS S3儲存的檔案結構? 是統一存放在同一個AWS裡面的folder? 還是按照不同watchentry or template or 時間不同folder? 然後local DB以及AWS DynamoDB又是怎麼管理這些data資料, 譬如是否有指標可以容易找到local database裡面每張image或結果image對應的UUID? 是否有指標可以容易找到AWS DynamoDB裡面每張image或結果image對應的UUID?

然後現在資料跟結果存放在AWS S3的檔案結構是否在管理上方便? 以及要支援多站點都進行掃描可以順暢的上傳到AWS S3管理? 而這樣的AWS檔案結構跟database設定對在local跟AWS協同的authentication pipeline是否也是優良的? 這樣的AWS檔案結構跟database設定對在local跟AWS協同的authentication pipeline是否也是優良的? 這樣的AWS檔案結構跟database設定對檔案, images等地搜尋功能(譬如尋找AWS S3上某個uuid的image file是屬於哪個watchid, 是屬於哪個template)是否也是優良的? 請深入分析並將結果整理, 並提供建議
```
我讀完了 S3 key 生成、兩條同步路徑、local SQLite schema、DynamoDB 寫入和各個查詢服務。總結一句：**設計方向是對的（UUID 檔名、`sites/{site}/watches/{watch}/runs/{run}` 分層、transactional outbox、UUID GSI），但目前 App 主流程只做了一半。** 至少有 4 種 S3 key 格式同時存在，App 的分析結果也沒帶 lineage 欄位。所以多站點管理、authentication pipeline 和 UUID 反查，目前都只對部分資料有效。

---

### 1. 現況：S3 檔案結構

S3 **不是**放在同一個 folder，也**沒有**統一規則。bucket 根目錄下同時存在以下格式：

|資料|S3 key 格式|產生位置|
|---|---|---|
|Raw 影像（App 主流程）|`sites/{site}/watches/{watch}/runs/{run}/raw/{uuid}.jpg`|local_storage.py:392|
|Raw（`structured_s3_keys=false`，舊版）|`Raw/{uuid}.jpg`（全部攤平）|同上|
|Report / result（DataManager 版）|`sites/.../runs/{run}/experiments/{exp}/reports|results/{task}/{uuid}`|local_storage.py:573, :632|
|**App 的 Analysis 結果**|`{DeviceID}/{watch}/Analysis/{exp}/{原始檔名}`|App/main.py:34228, :34444, workflow_manager.py:1852, :2524|
|Mask（workflow）|`{watch}/Analysis/{exp}/{hex}/{file}`|workflow_manager.py:3546|
|Sidepoint|`{watch}/runs/{run}/analysis/sidepoint_service_xxxxxx.png`|workflow_manager.py:1618|
|Authentication features/results|`{watch}/runs/{scan}/authentication/...`|runtime.py:754, :1022|
|Training corpus|`authentication/training/samples/feature_schema_version=…/…/site_id=…/dt=…/`（Hive）|storage_layout.py|
|Lake facts|`lake/analysis_facts`（Iceberg，依 brand/dt 分區）|lake_etl.py|
|WatchShift 參考圖|`watchshift/...`|cloud_sync.py:151|

也就是說，**同一個 watch 的 Raw 在 `sites/…`，它的 Analysis 在 `{DeviceID}/…`，authentication 結果又在 `{watch}/…`**。三個地方沒有共同的根。

Local 端的結構則簡單一致：`Local_Data/{watchid}/Raw/{uuid}.jpg`，以及 `Local_Data/{watchid}/Analysis/Exp_{時間}_{8hex}/{Task}_{rawUUID}_{suffix}.jpg`。

另外要注意：PDF 報告是以 `s3_key=""` 寫入的（App/main.py:36243），**目前不會上傳 S3**。

---

### 2. Local SQLite 怎麼管理

資料庫裡有兩代 schema 並存（db_manager.py:47）：

- **Legacy**：`raw_images`、`experiment_results`。主鍵是自增整數 `id`，**沒有 UUID 欄位**，UUID 只放在 `metadata` JSON 或檔名裡。
- **V3**：`watch_runs`、`experiments`、`point_instances`、`capture_instances`、`image_assets`（`asset_id` PK）、`analysis_results_v2`（`result_id` PK）、`artifact_records`。有 template、site、run 等索引，設計完整。
- **同步**：`sync_outbox`，目標分成 s3、catalog、lake。

**問題在於 App 主流程走的是 legacy 路徑**：

- Raw 用的是 `process_and_sync_raw_image`，只寫 `raw_images`，不寫 `image_assets`。回傳的 `raw_image_id` 是 **SQLite rowid**，不是 UUID。
- Analysis 用的是 `insert_experiment_result(...)`，**沒有傳 `exp_id` 和 `context`**，而且 `raw_image_id=None`（App/main.py:34246）。所以結果記錄沒有 run、experiment、template、site、source_asset。它的 id 會變成 `legacy_result_{rowid}`（db_manager.py:1556）。

**UUID 反查（local）**：有 `LocalProvenanceQueryService.lookup_uuid`（provenance_query.py:140），但它對 App 的資料會退回**逐行掃描整個 `raw_images` 並解析 JSON**（`LIKE '%uuid%'`），不走索引。Analysis 結果和 raw 的關聯只靠檔名裡嵌入的 raw UUID，DB 本身沒有記錄這個關聯。

---

### 3. DynamoDB 怎麼管理

Results table（`moonlight-WatchAnalysisResults`）由 `index_record` 寫入（cloud_db.py:165）：

- `PK = WatchID`，`SK = sort_key = asset_id`（UUID）
- 屬性攤平：`view_name`、`point_name`、`template_id`、`site_id`、`run_id`、`content_sha256`…
- GSI 用的鍵：`uuid_pk = UUID#{id}`（給 UuidIndex）、`source_asset_pk = ASSET#{raw uuid}`（給 SourceAssetIndex）
- 同一張 table 還混放了 watchentry 的 `INFO`、`VIEW#Front`、`MAT#…`、`AUTH` 等項目（watch_record_manager.py）

這裡的問題：

- **`UuidIndex`、`SourceAssetIndex`、`BrandModelIndex` 都不在任何 IaC 裡**，只能靠在 console 手動建立。
- 沒有任何程式寫入 `GSI1_PK`，所以 `query_watches_by_gsi`（cloud_db.py:509）永遠查不到東西。
- 走 outbox 時，App 的 analysis 結果在 DynamoDB 的 SK 是 `legacy_result_17` 這類 rowid。只要換站點掃描同一個 watch，或重建 local DB，就可能**覆寫**別筆記錄，也無法用檔名反查。
- 走 legacy polling 時，結果記錄送出的 metadata 是**任務的輸出資料**，不是 context（cloud_sync.py:245，配合 `get_pending_uploads`），所以 template、site 等欄位全是空的。
- 兩條路徑的 `record_type` 命名不一致：`raw_image/report/mask_image` 對上 `analysis_result/analysis_report/pdf_report`。

---

### 4. 同步機制

目前有兩個 worker，由設定擇一啟用。兩個都開時會都關掉，這個保護是好的。

- **Legacy polling**（`synced=0` → S3 → DynamoDB → `synced=1`）：只掃 `raw_images` 和 `experiment_results`。
- **Outbox**（建議用這個）：寫 DB 和寫 outbox 在同一個 transaction；catalog 會等 S3 完成後才寫；上傳後用 sha256 和檔案大小驗證；有 dead-letter。

Outbox 的缺點：

- 用 outbox 時，**各表的 `synced` 欄位永遠不會變成 1**。UI 或報表如果看這個欄位，會一直顯示未同步。
- 重試退避是以 `created_at + 2^n` 計算（outbox_dispatcher.py:254），所以舊項目失敗後會立即重試，等於沒有 backoff。
- 範例設定裡兩個 worker 都是 `false`，而且 `simulation`/`hybrid` 模式預設阻擋上傳。

---

### 5. 評估

|面向|評價|原因|
|---|---|---|
|**管理便利性**|⚠️ 中下|key 格式不統一；lifecycle、權限、成本無法用單一 prefix 管理；Analysis 不在 `sites/` 下|
|**多站點**|⚠️ 半成品|Raw 已依 site 分區是好的，但 `site_id` 缺值時退回 `"local"`（local_storage.py:358），多站可能全寫進 `sites/local/`；Analysis 用 DeviceID 當前綴；WatchID 是 `型號_epoch秒`，跨站可能撞名；無法用 IAM 限制「每站只能寫自己的 prefix」|
|**Authentication pipeline**|⚠️ 基礎好、資料斷鏈|training corpus 的 Hive 分區（有 site_id、dt）、Iceberg lake、以內容 hash 當 auth result id 都設計得好；但 App 分析結果進 lake 時 `site_id`、`template_id`、`run_id`、`source_asset_id` 是空的 → **依站點做 drift 分析、追溯證據回 raw 影像都會斷**；auth 結果 key 也不在 `sites/` 下|
|**搜尋：S3 UUID → watch**|✅ Raw 可以 / ❌ Analysis 不行|Raw 從路徑或 UuidIndex 就能查到；Analysis 檔名不是 UUID，DynamoDB SK 又是 rowid|
|**搜尋：UUID → template**|⚠️|路徑裡沒有 template；App 的 raw 在 metadata 有 template_id，App 的 analysis 沒有|
|**搜尋：template → 所有影像**|❌|DynamoDB 沒有 template GSI，只能全表 Scan；local 的 template 索引建在 `image_assets`，但 App 寫的是 `raw_images`|
|**搜尋：site → 所有資料**|⚠️|只有 Raw 能用 S3 prefix；DynamoDB 沒有 site 索引|
|**S3 物件自描述**|❌|物件上只有 `sha256` 一個 metadata，沒有 watch、site、template 的 tag，S3 Inventory 無法反查|

---

### 6. 建議（依優先順序，採最小改動）

#### P0：修正資料正確性（不做的話資料會繼續斷鏈）

1. **把 App 所有 Analysis 輸出改用 `build_artifact_s3_key`**，取代那 4 處手寫的 f-string，並傳入 `exp_id` 和 context（`result_id=uuid4().hex`、`run_id`、`template_id`、`template_version`、`site_id`、`source_asset_id`、`view_name`、`point_name`）。`source_asset_id` 可以從檔名裡的 `{Task}_{rawUUID}_…` 解析出來，或從 capture 流程直接帶入。
2. **`source_asset_id` 必須用 raw 的 UUID**（metadata 裡的 `asset_id`），不要用 rowid。
3. **正式環境 `site_id` 缺值要 fail-closed**：值為 `local` 或 `CHANGE_ME*` 且 `mode=production` 時，拒絕啟用 cloud sync。
4. PDF 報告要設 `s3_key`（例如 `…/experiments/{exp}/reports/{uuid}.pdf`），否則永遠不會上傳。

#### P1：統一結構，支援多站點

5. **所有資料統一放在 `sites/{site}/` 之下**：
    
    sites/{site}/watches/{watch}/runs/{run}/raw/{asset_uuid}.{ext}
    sites/{site}/watches/{watch}/runs/{run}/experiments/{exp}/results/{task}/{result_uuid}.{ext}
    sites/{site}/watches/{watch}/runs/{run}/experiments/{exp}/reports/{uuid}.{yaml|pdf}
    sites/{site}/watches/{watch}/runs/{run}/authentication/{features|results}/{id}.json
    
    全域資料（`authentication/training/`、`lake/`、`watchshift/`）維持在根目錄。  
    **template 不要放進路徑**，改用索引查詢，因為 template 版本會變，路徑應該保持不變。
    
6. **上傳時加上 S3 object metadata 和 tag**：`watchid`、`site_id`、`run_id`、`template_id`、`template_version`、`asset_id`、`artifact_class=raw|analysis|report`。tag 可以給 lifecycle 規則用（例如 raw 90 天後轉 Glacier IA），也能讓 S3 Inventory 不經 DynamoDB 直接反查。
7. **每站使用獨立權限**：你們已經有 AWS IoT 憑證，可以用 **IoT Credential Provider** 換取臨時憑證，IAM policy 設成 `s3:PutObject` 只允許 `sites/${iot:Connection.Thing.ThingName}/*`，取代 config 裡的長期 access key。
8. **WatchID 或 run_id 要能全域唯一**：WatchID 的 epoch 秒改成加上 site 或短 UUID；`Exp_{ts}_{8hex}` 可以保留，但確認它一定在 `sites/{site}` 之下。

#### P1：DynamoDB 索引（用 IaC 建立）

9. 用 CloudFormation 或 Terraform 建好 `UuidIndex`、`SourceAssetIndex`，並新增：
    
    - `TemplateIndex`：`template_pk = TEMPLATE#{id}#V#{ver}`，`SK = captured_at`
    - `SiteIndex`：`site_pk = SITE#{site}#DT#{yyyy-mm-dd}`，`SK = ts`
    
    四個都用 sparse、`INCLUDE` projection 控制成本。刪除或補上沒有被寫入的 `BrandModelIndex`（`GSI1_PK`）。
    
10. 統一 `record_type` 的命名，並考慮把 SK 加上型別前綴（`ASSET#`、`RESULT#`）。要改 SK 前需要先做 migration，避免破壞既有資料。

#### P2：Local DB 與同步

11. App 的 raw 改成同時寫 `image_assets`（`register_image_asset` 已經有 double-write 到 legacy 表的邏輯），或至少在 `raw_images` 加一欄有索引的 `asset_id`，讓 UUID 查詢不再全表掃描。
12. 改為正式使用 outbox，並在 `_mark_done` 時一併更新來源表的 `synced`；backoff 改成以 `now` 為基準計算。
13. 寫一支 backfill 腳本：替現有的 `{DeviceID}/…/Analysis/…` 物件補上 DynamoDB 欄位，或複製到新 key（S3 `CopyObject` 加上 metadata，保留舊 key 相容）。

另外有一件事：config/secret.key（32 bytes）已經被 commit 進 git。如果它是正式環境的加密金鑰，建議輪替並移出版本控制。我沒有讀它的內容。

---

**建議先驗證的地方**：到 AWS console 確認 `moonlight-WatchAnalysisResults` 是否真的有 `UuidIndex` 和 `SourceAssetIndex`；在 bucket 根目錄列一下，看目前實際有哪些頂層 prefix（`sites/`、DeviceID、watchid 各佔多少）；再查 local DB 裡 `sites/local/` 有多少筆。

需要的話，我可以從 P0 第 1、2 項開始實作（統一 App Analysis 的 key 並帶入 lineage）。如果這份分析要給團隊看，我也可以整理成一個可分享的頁面。





#### S3 檔案結構跟DB是否需要更新 - laptop
```
在目前的App main.py當在watchentry scan完一個watch的所有views包含front, back....會將所有image files存放在<watchentry id>的Raw folder, 而用./tasks/裡面對每個images分析的task service產生的結果包括image results跟json files也會存在<watchentry id>的Analysis folder, 同時這些raw folder跟Analysis folder的所有images, results也會將資訊存到local database. 而這些Raw folder跟Analysis folder裡面的images, files也會上傳到AWS S3儲存, local DB也會跟AWS DynamoDB同步.

而以下是儲存在S3的檔案結構:

S3 **不是**放在同一個 folder，也**沒有**統一規則。bucket 根目錄下同時存在以下格式：

|資料|S3 key 格式|產生位置|
|---|---|---|
|Raw 影像（App 主流程）|`sites/{site}/watches/{watch}/runs/{run}/raw/{uuid}.jpg`|local_storage.py:392|
|Raw（`structured_s3_keys=false`，舊版）|`Raw/{uuid}.jpg`（全部攤平）|同上|
|Report / result（DataManager 版）|`sites/.../runs/{run}/experiments/{exp}/reports|results/{task}/{uuid}`|
|**App 的 Analysis 結果**|`{DeviceID}/{watch}/Analysis/{exp}/{原始檔名}`|App/main.py:34228, :34444, workflow_manager.py:1852, :2524|
|Mask（workflow）|`{watch}/Analysis/{exp}/{hex}/{file}`|workflow_manager.py:3546|
|Sidepoint|`{watch}/runs/{run}/analysis/sidepoint_service_xxxxxx.png`|workflow_manager.py:1618|
|Authentication features/results|`{watch}/runs/{scan}/authentication/...`|runtime.py:754, :1022|
|Training corpus|`authentication/training/samples/feature_schema_version=…/…/site_id=…/dt=…/`（Hive）|storage_layout.py|
|Lake facts|`lake/analysis_facts`（Iceberg，依 brand/dt 分區）|lake_etl.py|
|WatchShift 參考圖|`watchshift/...`|cloud_sync.py:151|

也就是說，**同一個 watch 的 Raw 在 `sites/…`，它的 Analysis 在 `{DeviceID}/…`，authentication 結果又在 `{watch}/…`**。三個地方沒有共同的根。

Local 端的結構則簡單一致：`Local_Data/{watchid}/Raw/{uuid}.jpg`，以及 `Local_Data/{watchid}/Analysis/Exp_{時間}_{8hex}/{Task}_{rawUUID}_{suffix}.jpg`。

另外要注意：PDF 報告是以 `s3_key=""` 寫入的（App/main.py:36243），**目前不會上傳 S3**。

如果我要重新設計儲存在S3的檔案結構讓資料跟結果存放在AWS S3的檔案結構是否在管理上方便? 以及要支援多站點都進行掃描可以順暢的上傳到AWS S3管理? 而這樣的AWS S3檔案結構對在local跟AWS協同的authentication pipeline是否也是優良的? 這樣的AWS S3檔案結構跟database設定對檔案, images等地搜尋功能(譬如尋找AWS S3上某個uuid的image file是屬於哪個watchid, 是屬於哪個template)是否也是優良的? 請深入分析並將結果整理, 並提供建議
```
我讀了程式碼，也用唯讀方式查過 DB/watchDB.db 的實際資料。你表格裡列的程式位置全部正確。不過真實資料的狀況比表格看起來還糟。這次只做分析，沒有改任何檔案。

### 一、結論

|問題|現況|主因|
|---|---|---|
|管理是否方便|❌|同一次掃描散在 3 個以上沒有共同根目錄的地方，無法用一個 prefix 做刪除、匯出、保存期限或權限設定|
|多站點上傳|⚠️|Raw 路徑有站點，但 Analysis 和 authentication 路徑沒有；同一台機器還用了兩個不同的身分|
|Authentication 協同|⚠️|training corpus 和 bundles 的設計是好的；但每次掃描的證據路徑沒有站點，而且 IAM 的萬用字元權限過大|
|搜尋（uuid → watch/template）|❌|主要障礙不在 S3 key，而在 **catalog 記錄沒有寫入 lineage 和 template 欄位**|

最重要的一點：光改 S3 key 解決不了搜尋問題。要先修 DB 寫入的內容，再改 key。

### 二、核實結果

#### 程式碼層面

- **同一站點有兩個身分。** 設定檔中 `site_id: laptop`，而 `DeviceID: Moonlight system v1`。
    - Raw 用 `sites/laptop/...`（local_storage.py:346）。
    - Analysis 用 `Moonlight_system_v1/...`（main.py:34228）。
    - 同一台機器因此在 S3 出現兩個根目錄。
- **App 的分析結果完全沒有 lineage。** main.py:34246 呼叫 `insert_experiment_result` 時帶的是 `raw_image_id=None`，而且沒有傳 `context`。結果是寫進 catalog 的資料中：
    - `template_id`、`run_id`、`site_id`、`source_asset_id`、`view_name` 全部是空的；
    - `asset_id` 變成 `legacy_result_{本機自增id}`（db_manager.py:1557）。多個站點會產生相同的 `legacy_result_12`，UuidIndex 查詢會回傳多筆、無法判斷是哪一筆。
- **WatchID 是 `{template_name}_{epoch秒}`**（main.py:28205）。
    - 會含空白和括號，例如 `run_16613T (V2)_...`。
    - 兩個站點在同一秒用同一個 template 建立 watch，DynamoDB 的 PK 就會撞在一起。
- **Authentication 的 IAM 權限過大。** Cloud role 對 `*/runs/*` 有 `s3:PutObject`（authentication.yaml:264）。S3 的 `*` 可以匹配 `/`，所以這個權限也涵蓋 `sites/.../runs/.../raw/`。等於雲端的 authentication 角色可以覆寫 Raw 原始影像，這對 provenance 系統是完整性風險。
- **Preprod 的 catalog 表沒有 GSI**（authentication.yaml:183），所以 `query_by_uuid` 在 preprod 一定失敗。Production 的 `UuidIndex` 和 `SourceAssetIndex` 不在 IaC 裡，我無法確認它們是否真的存在。
- **本機 DB 的索引不足。** `experiment_results` 沒有任何索引，`raw_images` 在 `s3_key` 上也沒有索引。

#### 實際資料（DB/watchDB.db，2026-05-03 至 2026-08-12）

|資料表|實際情況|
|---|---|
|`raw_images` 7,277 筆|**全部是舊的平鋪格式 `Raw/{uuid}.png`**，`sites/...` 格式一筆都沒有|
|同上|3,521 個 `s3_key` 各出現 2 次。這是舊 bug，目前程式碼在 db_manager.py:579 已有去重|
|`experiment_results` 9,944 筆|**9,519 筆（95.7%）的 `raw_image_id` 是 NULL**；157 筆的 `s3_key` 是空的|
|`image_assets` 3,521 筆|`site_id` 全部是 NULL|
|`analysis_results_v2`、`artifact_records`、auth 相關表、`sync_outbox`|0 筆，新的 pipeline 在這台機器上還沒產生過資料|

所以你表格裡描述的「新格式」目前只存在於程式碼中。S3 上實際的歷史資料是 `Raw/` 平鋪加 `Moonlight_system_v1/`。遷移時要以這兩種格式為準。

### 三、建議的 S3 結構

#### 原則

1. S3 key 只放不會變的身分欄位：env、site、watch、run、uuid。brand、template、view 這類業務欄位放在 catalog 和 object metadata，因為 template 可能被重新指派，名稱也會改。
2. 依存取模式分成不同的根目錄。每次掃描的證據放在同一棵樹；跨 watch 的資料（training、lake、參考圖）各自有根目錄。IAM、Lifecycle、Object Lock、Replication 和 S3 事件通知都只能用 prefix 來設定。
3. 站點放在最上層分區，這樣 IAM 可以直接寫成 `captures/site=${aws:PrincipalTag/site_id}/*`。每個站點只能寫自己的 prefix，天生不會互相覆寫。
4. 分區用 Hive 的 `key=value` 寫法。這樣就算 DynamoDB 出問題，也能用 S3 Inventory 加 Athena 直接從 key 重建 catalog，同時多一條稽核管道。

#### 結構

s3://moonlight-{prod|nonprod}/            ← 環境用不同 bucket 分開（見第五節）
  captures/site={site_id}/watch={watch_id}/run={run_id}/
      raw/{asset_uuid}.{ext}               ← 不可覆寫（Object Lock / Deny overwrite）
      raw/{asset_uuid}.{ext}.meta
      derived/exp={experiment_id}/{task}/{result_uuid}.{ext}
      reports/{report_uuid}.pdf            ← 若要讓 PDF 上雲
      authentication/features/{batch_id}.json
      authentication/results/{result_id}.json
      _manifest.json                       ← 最後寫入，代表這次 run 已完整上傳
  authentication/training/samples/...      ← 維持現有 Hive 結構，不需要改
  authentication/bundles/...               ← 維持現狀
  lake/analysis_facts/                     ← 維持現狀
  reference/watchshift/{template_id}/...   ← 原本的 watchshift/，搬到 reference/ 底下

幾個關鍵選擇：

- **衍生檔用 `{result_uuid}` 命名，不用原始檔名。** 目前的 `{Task}_{rawUUID}_{suffix}.jpg` 無法反查結果本身的 id。原始檔名放到 metadata 的 `original_filename`。
- **站點放在 watch 之前。** 同一支錶在 A、B 兩站掃描，資料會分在兩個 prefix。這是可以接受的，因為 run 本來就是在某個站點產生的；要看一支錶的完整歷史，應該查 catalog，而不是列 S3。
- **不需要做 prefix 雜湊。** 這個資料量遠低於每個 prefix 每秒 3,500 次 PUT 的上限。

### 四、對 authentication pipeline 的評估

**已經做對的，建議保留：**

- `storage_layout.py` 的 training corpus 是 Hive 分區，而且兩個版本欄位排在最前面。
- bundle 是不可變的 prefix，簽章檔獨立放置。

這部分就是 Athena 跨站點取用資料的正確設計，不需要改動。

**要修的地方：**

1. **每次掃描的證據要放回 `captures/site=/watch=/run=/authentication/`，和 raw 放在同一棵樹。**
    - 將來 `extractor_version` 升級、要重新抽取特徵時，雲端需要從 feature batch 找回原始 raw。同一個 prefix 就能直接列出來，不必依賴 catalog 做 join。
    - IAM 可以收緊成讀 `captures/*` 加寫 `captures/*/authentication/*`。這樣就不再能寫入 raw。
2. **S3 事件通知只能用固定的 prefix 或 suffix 過濾。** 在現有的 `{watch}/runs/...` 格式下，沒辦法只對 authentication 結果觸發。Replay Lambda 已經有 invoke 權限，但 IaC 裡找不到通知設定，所以無法確認目前是怎麼接的。
3. **統一 run 的識別名稱。** 目前 `scan_id`、`run_id`、`exp_id` 三個名稱並存；main.py:13949 已經把它們統一成 `run_id`，key 裡只用 `run=`。
4. **learning sample 已經帶有 `site_id` 和 `camera_id`**，這一點是正確的。前提是 `image_assets.site_id` 要真的寫進去，目前全部是 NULL。

### 五、DB 與 catalog 的建議（比改 key 更重要）

1. **App 的分析結果一律帶完整 context 寫入。** 在 main.py:34246、main.py:34457 及 `workflow_manager.py` 的三處呼叫，都要傳入：
    
    - `result_id=uuid4().hex`
    - `source_asset_id=<raw asset uuid>`
    - `site_id`、`run_id`、`experiment_id`、`template_id`、`template_version`、`view_name`、`point_name`
    
    同時停止產生 `legacy_result_{lid}`。這一步做完，「uuid → watch/template」和「raw → 所有衍生結果」這兩種查詢才會成立。
    
2. **S3 key 只由一個函式產生。** 現在 key 散在七處用 f-string 各自拼接，要全部改成呼叫 `build_artifact_s3_key`（或新的 `storage_layout`）。
3. **只保留一個站點身分。** 統一用 `site_id`，`DeviceID` 只作為 MQTT topic 使用。
4. **DynamoDB 的 GSI 要寫進 IaC。** `UuidIndex`（`uuid_pk`）和 `SourceAssetIndex` 都要定義，建議再加一個 `TemplateIndex`（`template_id#version` 對應 `created_at`）。
5. **本機 DB 要補的：**
    - 為 `experiment_results(s3_key)`、`raw_images(s3_key)`、`image_assets(template_id, template_version)` 建索引；
    - `experiment_results` 加上 `result_uuid` 和 `source_asset_uuid` 欄位。
6. **每個物件都寫入 S3 object metadata**：`asset-id`、`watch-id`、`site-id`、`run-id`、`template-id`、`sha256`。這樣即使是 catalog 查不到的孤兒檔案，也能自己說明來源。
7. **WatchID 的碰撞問題。** 最小的修改方式是在後面加 4 個 hex：`{template}_{epoch}_{4hex}`，並把空白和括號清理掉。

### 六、遷移步驟（向下相容）

1. **先修寫入端。** 做第五節的第 1、2、3 點。新 key 用 `storage.s3_layout: v3` 開關控制，舊的 `structured_s3_keys=false` 繼續保留。
2. **補 catalog 資料。** 從本機 DB 重新計算 lineage：用 `{rawUUID}` 從衍生檔的檔名解析出 `source_asset_id`，再從 `watch_runs` 補上 template。
3. **搬移舊物件。** 用 S3 Batch Operations 依 manifest 複製 `Raw/`、`Moonlight_system_v1/`、`{watch}/runs/`；複製完才更新 `s3_key`。舊物件先保留 N 天，用 Lifecycle 自動刪除，不要手動刪。
4. **調整權限與規則。** 收緊 IAM、對 `captures/*/raw/` 啟用 Object Lock 或 Deny overwrite、設定 Lifecycle，把 raw 超過 90 天的物件移到 Glacier IR。
5. **最後才改 UI 和 CLI 的查詢方式。**

### 七、需要你決定的事項

- **環境隔離。** 目前這台機器是 `mode: simulation`，但 bucket 是 production 的 `moonlight-system-v2`。我建議 prod 和 nonprod 用不同 bucket，而不是在同一個 bucket 用 prefix 區分，因為 Object Lock 和 Replication 都是 bucket 層級的設定。這會影響部署方式，所以由你決定。
- **PDF 報告。** 程式碼裡明確寫著 `export_policy: "local_only"`（main.py:36265），看起來是刻意不上傳。如果要上雲，要先確認報告裡是否含客戶資料。
- **無法核實的部分。** 我沒有連 AWS，所以 S3 上實際有哪些 prefix、production 表是否有 GSI 都未確認。可以用 DB/DB_test/check_s3.py 或 `aws s3 ls` 對照確認。

需要的話，我可以把這份分析做成可分享的頁面，或從第六節第 1 步（修分析結果的 lineage 和統一 key 產生函式）開始實作。





#### 多個branch有各自的local database. 怎麼合併
```
@"C:\Users\a3146\OneDrive\桌面\other\DB backup\watchDB_debugbezel_09271015.db" 現在狀況是在LA downtown的Moonlight system主機(真實接上硬體在拍照watch的)有幾個branch, 每個branch有各自的local database. 請比較我的local DB跟我最新上傳的local database(從Moonlight system主機 copy過來)檔案格式類似或更糟? 之後我們會需要優化local DB(就像我們之前對AWS S3預計做的一樣), 並會合併數個local DB成一個主要local DB, 然後數個branch也會都merge成dev branch. 所以可以這樣合併local DB嗎? 最後dev branch的local DB應該就是這個final合併過的local DB嗎? 所以這個Moonlight system 以後只會有個dev branch跟單一個local DB. 但同時我們會生成production code, 那在Moonlight system又會有個local DB. 那在Moonlight system要怎麼管理這兩個local DB(因為在dev branch開發時預計會有很多測試性的拍照image, 所以應該要跟正式production code拍照的分開)?
```
我比較了兩個 DB（原檔沒動，複製到 scratchpad 後以唯讀方式分析）。先說結論：兩者格式相同，但 LA 那份在幾個地方更糟。兩份 DB 可以合併，但不能直接把資料表 copy 過去。另外，「dev branch 的 DB」這個說法需要修正，DB 其實不屬於 branch。

### 一、比較結果

你的 local DB 路徑多半是 `D:/Moonlight/ImagingLibWatch/Local_Data`，所以它其實是早期從 Moonlight 主機 main 資料夾複製過來的（5/3 到 8/12）。LA 這份則來自 `ImagingLibWatch_debugbezel` 資料夾（9/3 到 9/25）。

|項目|你的 local|LA debug_bezel|判定|
|---|---|---|---|
|Schema|較新|**較舊**：少 6 張表、少 9 個欄位、少 18 個 index|LA 較差|
|Raw S3 key|全部是 `Raw/{uuid}.png`|同左|相同|
|raw_images 重複記錄|3,521 筆，每張都重複|3,876 筆，也是每張都重複|相同|
|分析結果沒有 `raw_image_id`|95.7%|93.3%|相同|
|`image_assets.metadata` 缺 site/template/sha256|100%|100%|相同|
|`experiment_results.data` 平均大小|28 KB，共 279 MB|**166 KB，共 530 MB（占整個 DB 的 82%）**|LA 較差|
|`local_path='.'` 的壞路徑|142 筆|234 筆|LA 較差|
|`sync_outbox`|0 筆|**7,791 筆 pending，從未處理**|LA 較差|
|`local_path` 的根目錄|3 種不同根目錄|1 種|LA 較好|

`debug_bezel` 比 dev 少 64 個 commit，其中包括 9/8 修掉 raw_images 重複寫入的那個 commit（`9866987`）。所以 LA 這份 DB 到 9/25 為止還在持續產生重複記錄。

還有一點：LA 的 `synced=1` 不能證明檔案真的上傳了。在 simulation mode 下，mock 模式也會把記錄標成 synced。要確認的話，需要抽幾個 key 到 S3 上查。

### 二、可以合併嗎？

可以，但用 `ATTACH` 加 `INSERT` 直接倒進去會壞掉。實際衝突如下：

|衝突|數量|處理方式|
|---|---|---|
|`raw_images.id`|**7,277 筆 ID 全部重疊**|必須重新編號，同時修正 `experiment_results.raw_image_id` 和 `sync_outbox.source_id` 的對應|
|`experiment_results.id`|3,195 筆全部重疊|同上|
|asset_id、run_id、WatchID、s3_key 等文字或 UUID 主鍵|0 筆重疊|可以安全合併|
|`local_templates`|**27 組 (TemplateID, version) 相同，其中 24 組內容不同**|無法自動合併，要你逐一決定。這代表修改 template 時沒有升版號|
|users|1 個 admin 帳號的密碼 hash 不同|要決定保留哪一邊|
|檔案路徑|絕對路徑，每個資料夾都不同|DB 只是索引，**`Local_Data/` 裡的實體檔案要一起合併**，路徑也要改寫成相對路徑|

另外，兩份 DB 的時間是 8/12 以前和 9/3 以後，中間有空檔。主機上其他 branch 資料夾的 DB 應該也要一起納入。

**建議的順序**（先改 schema 再合併，才不會做兩次）：

1. 先把各個程式 branch merge 進 dev，讓 schema 只剩一個版本。
2. 設計優化後的 schema：補上 lineage 和 template 欄位、把大的 `data` 搬到 JSON 檔、加上 `schema_version`。
3. 寫一支合併 ETL。它先以 dry-run 產出衝突報告，再把每份來源 DB 匯入**一個新的 DB**。匯入時要：
    - 重新編號 ID；
    - 標上 `source_db` 和 branch；
    - 依 s3_key 去除重複；
    - 把路徑改成相對路徑。
4. 驗證資料筆數和實體檔案是否存在，確認無誤後才切換。

### 三、「dev branch 的 local DB」這個說法要修正

/DB/watchDB.db 和 `/Local_Data/` 都在 `.gitignore` 裡（.gitignore:2、.gitignore:17）。`db_path` 又是相對路徑，所以 **DB 綁的是資料夾，不是 branch**。LA 有好幾份 DB，是因為有好幾個資料夾；在同一個資料夾裡切換 branch，用的還是同一份 DB。

所以合併後的 DB 應該稱為「**dev 環境的 DB**」，而不是 dev branch 的 DB。這不只是名稱問題，還有實際風險：程式開啟 DB 時會自動加欄位（`_ensure_column`），而兩份 DB 的 `user_version` 都是 0。只要新版 code 開到哪一份 DB，就會在不知不覺中把它 migrate 掉。

### 四、Moonlight 主機上 dev 和 production 兩份 DB 怎麼管理

原則是**依環境分開，而不是依 branch 分開**，而且資料要放在程式資料夾之外。程式已經支援用絕對路徑設定 `db.db_path` 和 `paths.base_data_dir`（local_storage.py:154），並寫在 gitignored 的 `system_config.local.yaml` 裡。

D:/Moonlight/
  code/dev/          ← git clone，checkout dev（開發用）
  code/prod/         ← 只 checkout release tag，不在這裡開發
  data/dev/   watchDB.db, Local_Data/     ← 合併後的 DB 放這裡
  data/prod/  watchDB.db, Local_Data/     ← 從空的 DB 開始

|項目|dev|production|
|---|---|---|
|`system.mode`|simulation / hybrid|production|
|`site_id`|`la-downtown-dev`|`la-downtown`|
|S3 / DynamoDB|nonprod bucket 和 table|prod bucket 和 table|
|Templates|可以編輯|只能拉取已核准的版本，每次修改都要升版號|
|Users|各自一套，不共用密碼|各自一套|
|Schema migration|開啟時自動進行|**不自動**：先備份，再明確執行 migration|

需要加的防呆機制：

1. **DB 記錄自己的身分。** 在 DB 裡放一張 `db_meta` 表，記錄 `env`、`site_id`、`schema_version`。App 啟動時比對 config，不一致就拒絕啟動，避免 dev code 誤開 prod DB。
2. **硬體同一時間只能給一個 App 用。** dev 和 prod 不能同時運作，需要一個硬體 lock。我還沒確認目前 App 有沒有單一實例保護。
3. **備份。** Prod 用 SQLite 的 online backup API（`.backup`），不要在 App 執行中直接複製檔案。這次的檔案 `quick_check` 雖然通過，但那是運氣。
4. **推送到 prod 要有工具。** 如果 dev 期間拍到真正有價值的錶，用一個明確的 promote 工具推到 prod，不要直接複製 DB。
5. **合併後的歷史資料留在 dev。** 它混了大量測試拍攝，不應該當作 production 的初始資料。

### 需要你決定

- 那 24 組同名同版號、內容卻不同的 templates，要保留哪一邊，還是改成不同版號？
- LA 那 7,791 筆從未處理的 outbox 記錄：先確認 S3 上是否已經有這些檔案，再決定要不要丟棄。
- 主機上還有哪些 branch 資料夾的 DB 要一起合併？

我這次沒有修改任何檔案。下一步我建議先寫合併 ETL 的 dry-run 版本，它只讀取各份 DB、輸出衝突報告，不會寫入任何東西。






```
DB 綁的是資料夾沒錯, 不過是把每個branch建立worktree. 而目前預計在moonlight system應該建立兩個branch(在LA downtown的hardware engineer的他的user branch負責拍照跟硬體一些code), 以及我在Michigan主要工作在dev branch負責主要coding工作, 我定期會把他的user branch的code changes merge進去我的dev branch. 而當在Moonlight system要交付給顧客的版本時, 我會在Moonlight system從dev branch建立production code, 會在Moonlight system生成production folder裡面有packed的code, 以及protected folder裡面有DB, checkpoints, local_Data等sub folders裡面也有它自己的local DB. 那目前即使已經更新的新的優化格式, 但仍有三個local DB在Moonlight system. 那我們要怎麼管理這些local DB, 特別是那位hardware engineer在測試硬體會拍出很多測試性的images, 我們可能不希望這些進去local DB or 上傳到AWS S3, 因為這些也會對之後的authentication造成不好影響. 所以考慮到這些因素, 請提供建議, 在日常開發工作merge branch時, local DB要怎麼處理?(譬如也要做local DB的合併)? 以及生成production code時, dev branch的local DB跟production code的local DB是否要做合併? 譬如把這一次production code的local DB合併進去上一次production code的local DB? 或者是跟dev branch的合併? 請深入分析並整理規劃適合的方案. 不要修改code. 另外目前local DB跟outbox沒有上傳/同步到AWS, 我們有在config設定關閉, 很多images跟results也沒有上傳到AWS S3. 那我們如果在config設定enable是否只會之後拍照的images跟results會開始自動上傳, 但之前的不會? 如果我們想把之前沒上傳的images跟results跟DB上傳/同步到AWS, 是否要另外寫code額外進行這動作?
```
### 一、開啟 config 之後，只會上傳之後的資料嗎？

**不是。** 上傳哪些舊資料取決於你開的是哪一個 worker，而且兩種都無法控制範圍。

要真正開始上傳，config 要同時滿足這幾個條件：

- `security.enable_cloud_sync: true`
- 如果 `mode` 是 simulation 或 hybrid，還要 `force_upload_to_S3: true`
- `sync.legacy_polling_enabled` 和 `sync.outbox_enabled` 只能開其中一個
- 通過 cloud policy 檢查，並設定好 region 和 bucket

|Worker|會上傳哪些舊資料|以 LA 那份 DB 為例|
|---|---|---|
|Outbox（outbox_dispatcher.py:57）|**所有 `status='pending'` 的舊記錄，不論時間多久**|開啟後會立刻推出 **7,791 筆**：3,876 張測試 raw 影像、3,876 筆 catalog，以及 **39 筆 template，會覆寫 DynamoDB 上的 templates**（其中 24 組和你的版本內容不同）|
|Legacy polling（db_manager.py:1677）|所有 `synced=0` 的舊記錄|LA 的記錄全部是 `synced=1`，所以一筆都不會傳|

還有三個風險：

- **舊記錄會用舊格式的 key 上傳。** S3 key 在寫入 DB 的當下就已經固定在 payload 或 row 裡，所以舊資料會以 `Raw/{uuid}.png` 上傳，和我們規劃的新 S3 結構衝突。
- **有些資料永遠不會上傳。** LA 的 `experiment_results` 當時的程式碼沒有產生 outbox 記錄；被標成 `synced=1` 但其實沒傳的資料也一樣。
- **LA 的 `synced=1` 不能相信。** Sync 關閉時用的是 NullCloudSync，它不會標記 synced。所以 `synced=1` 只可能來自 MockCloudSync（只標記、不上傳）或真的上傳過。
    - 另外，`debug_bezel` 的程式在 config 缺少設定時，預設就是 `enable_cloud_sync=True`、`mode=production`。
    - 所以**有可能測試影像已經在 S3 上了**。要抽樣用 `head_object` 查才能確定；這需要用到 AWS 憑證，所以我沒有做。

**結論：** 對，歷史資料需要另外寫 backfill 工具。而且在開啟任何開關之前，要先把現有的 pending 記錄擋下來，否則一開啟就會把所有測試歷史用舊格式推上去。

### 二、根本問題

- **程式碼裡沒有任何欄位能區分「測試拍攝」和「正式拍攝」。** 我搜尋了整個專案，找不到 `capture_class`、`is_test` 之類的欄位；outbox payload 也沒有記錄 `system_mode`。資料一旦混進同一個 DB，事後就只能靠人工判斷。
- **Authentication 有一道真正的防線：** learning sample 必須有專家標註才會建立（runtime.py:589），沒有標註的測試拍攝不會進入 training corpus。
- **但這道防線不涵蓋其他資料。** Raw 影像、分析結果、lake 都沒有這道檢查，而且 App 預設 `learning_mode = True`（main.py:5828）。只要有人在測試拍攝上填了標註，就會污染訓練資料。

因此，隔離要在**拍攝當下依環境決定**，而不是等到合併時再挑出來。

### 三、三個 DB 的角色

|DB|環境身分|內容|上雲|生命週期|
|---|---|---|---|---|
|硬體工程師 worktree（他的 user branch）|`hwlab`|全部視為測試資料|**永遠不上傳**|可以隨時清空，建議保留 30 天|
|你的 dev worktree|`dev`|整合測試用，以及挑選過的開發用資料|只能上傳到 nonprod bucket，或完全不上傳|長期保存|
|Production protected|`prod`|**只有正式拍攝**|唯一會上傳到 prod S3 和 DynamoDB 的 DB|永久保存，跨版本延續|

硬體工程師的測試影像還是要寫進 `hwlab` DB，因為 App 要靠 DB 來顯示和分析影像。重點是這份 DB 永遠不會離開那個環境。規則是：**真實客戶的錶或參考用的錶，只能在 production App 裡拍。**

### 四、日常 merge branch 時，DB 怎麼處理

**答案是不合併 DB。** 程式碼透過 git 流動，資料不跟著 merge。每次 merge 要處理的是這些：

1. **Schema 變更放在程式碼裡，做成有版本號的 migration**（例如 `PRAGMA user_version` 或 `schema_migrations` 表）。每個 worktree 下次啟動時各自 migrate 自己的 DB：hwlab 和 dev 可以自動進行，prod 必須手動執行。
2. **舊程式碼不能寫入比它新的 DB。** 如果 DB 的 schema 版本高於程式碼認得的版本，就拒絕寫入。
    - `debug_bezel` 比 dev 少 64 個 commit，其中包括 9/8 的去重修正，而且它的 sync 預設值是危險的。
    - 建議硬體工程師**每週把 dev merge 進他的 branch**，同時你也把他的 branch merge 進 dev，避免兩邊差距太大。
3. **Templates 不是 capture，而是設定。** Source of truth 應該只有一個，也就是 DynamoDB 的 template 表，每個 DB 只是 cache。修改 template 一定要升版號。那 24 組同版號、內容卻不同的 templates，就是把 local DB 當成 source of truth 造成的。
4. **例外情況：** 如果 hwlab 確實拍到有價值的資料（例如你請他拍的參考錶），用「指定 `run_id` 的匯出加匯入工具」連同檔案一起搬移，並重新分類。這是 promote，不是合併 DB。
5. **從 hwlab 流向 prod 的應該是校正資料和硬體設定**，透過 git 或 release artifact 傳遞，而不是 captures。

### 五、生成 production 時

**兩種合併都不要做：**

- 不要把 dev DB 合併進 prod。
- 也不要每次 release 產生一個新 DB，再合併進上一版。

**Production DB 應該只有一份，跨版本延續，每次升級時就地 migrate。** 每次 release 替換的只有 code 資料夾。

Release 流程：

1. 從 dev 打 tag，例如 `v1.2.0`，再打包程式碼。
2. 停止 prod App。用 SQLite 的 backup API 備份 prod DB（不要在執行中直接複製檔案），並保留 N 份。
3. 部署到 `production/releases/v1.2.0/`，再切換 `current` 指向它，保留之前的版本以便 rollback。
4. 明確執行 migration，先 dry-run，再從 N 版升到 M 版。版本不符時 prod App 拒絕啟動。
5. Smoke test：拍一張校正用的目標物，並標記為 `calibration`，不要拍真實的錶。
6. Rollback 的做法是切回上一版的資料夾，並還原備份。

|從 dev 帶到 prod|不帶到 prod|
|---|---|
|程式碼、有版本號和 hash 驗證的 checkpoints、已核准的 template 版本、校正資料|captures、分析結果、users、outbox|

**第一次上 production 時，prod DB 從空的開始。** 歷史資料裡如果有真正的參考錶掃描（例如 LA 在 9/3 到 9/25 拍的那批 Rolex），用**一次性、依 `run_id` 人工審核**的方式匯入。上次討論的合併 DB 就用在這一步。

```
D:/Moonlight/
  worktrees/hw-<name>/     ← user branch；config.local: env=hwlab，DB 指向 data/hwlab
  worktrees/dev/           ← dev branch；env=dev，DB 指向 data/dev
  data/hwlab/  watchDB.db, Local_Data/     (保留 30 天)
  data/dev/    watchDB.db, Local_Data/
  production/releases/v1.x.y/   production/current → 指向目前版本
  protected/   DB/  Local_Data/  checkpoints/  backups/   (只有 prod 服務帳號能寫入)
```
**DB 一定要移出 worktree。** 目前 DB 是用相對路徑放在 worktree 裡面。`git worktree remove` 只把未追蹤檔案和修改過的追蹤檔案算作「不乾淨」，**被 ignore 的 `DB/` 和 `Local_Data/` 不會阻止移除，會跟著整個資料夾一起被刪掉。** 程式已經支援用絕對路徑設定 `db.db_path` 和 `paths.base_data_dir`，只要改 gitignored 的 `system_config.local.yaml` 就可以。

### 七、之後要加的防呆機制（這次不改程式碼）

1. **DB 記錄自己的身分。** 在 DB 裡放一張 `db_meta` 表，記錄 env、site_id、schema_version。App 啟動時如果和 config 不一致就拒絕啟動；`hwlab` DB 在程式碼層級禁止開啟 cloud sync。
2. **加上 `capture_class` 欄位，以 run 為單位**，值為 real、reference、test 或 calibration。
    - 預設值由環境強制決定：hwlab 一律是 test，prod 一律是 real。
    - 這個值要一路帶到 S3 metadata、catalog 和 lake。
    - Training corpus 和 lake ETL 只收 real 和 reference。
3. **Outbox payload 記錄 env 和 `capture_class`。** Dispatcher 發現記錄的 env 和目前設定的目標 env 不符時，拒絕上傳。
4. **在 hwlab 環境停用專家標註的 UI**，從源頭避免測試拍攝被標註後進入訓練資料。
5. **硬體 lock：** 同一時間只能有一個 App 控制硬體。我還沒確認目前是否已有這個保護。
6. **Prod 和 nonprod 的 S3 bucket、DynamoDB 表完全分開。**

### 八、歷史資料的補傳順序

1. **先擋住 pending 記錄。** 把每個 DB 現有的 pending outbox 記錄改為 held 或封存，特別是 LA 那 39 筆 template。
2. **抽樣查 S3**，確認 LA 的資料到底有沒有上傳過。
3. **人工分類**：依 `run_id` 產出一份清單，標明哪些是正式拍攝、哪些是測試。這是目前唯一的辦法，因為程式碼裡沒有相關欄位。
4. **先完成 S3 結構重新設計，再做 backfill**，否則會用舊 key 上傳，之後還得再搬一次。
5. **寫新的 backfill 工具。** 它依照清單產生新格式的 key、上傳後用 sha256 驗證、寫入 lineage 完整的 catalog。這支工具要可以重複執行、可以 dry-run、中斷後可以續傳，並且先在 nonprod 跑過。
    - 現有的 `backfill_lake_from_dynamodb.py` 是從 DynamoDB 補資料到 lake，方向相反，不能拿來用。
6. **最後只在 prod 開啟 outbox**（不開 legacy polling），給之後的新拍攝使用。






#### [ AWS ] 分析S3與資料庫架構 (laptop)  
```
在目前的App main.py當在watchentry scan完一個watch的所有views包含front, back....會將所有image files存放在<watchentry id>的Raw folder, 而用./tasks/裡面對每個images分析的task service產生的結果包括image results跟json files也會存在<watchentry id>的Analysis folder, 同時這些raw folder跟Analysis folder的所有images, results也會將資訊存到local database. 而這些Raw folder跟Analysis folder裡面的images, files也會上傳到AWS S3儲存, local DB也會跟AWS DynamoDB同步. 請幫我整理這些Raw folder跟Analysis folder裡面的images, files上傳到AWS S3儲存的檔案結構? 是統一存放在同一個AWS裡面的folder? 還是按照不同watchentry or template or 時間不同folder? 然後local DB以及AWS DynamoDB又是怎麼管理這些data資料, 譬如是否有指標可以容易找到local database裡面每張image或結果image對應的UUID? 是否有指標可以容易找到AWS DynamoDB裡面每張image或結果image對應的UUID?

而我現在AWS S3上面看的主要資料夾結構是:  
Amazon S3/Buckets/的下面有兩個:  
auth-isolated-lab-001-426476636376-us-east-2  
moonlight-system-v2

Amazon S3/Buckets/moonlight-system-v2的下面有  
AUTH_UI_20260908_164328_961/runs/

authentication/training/

Moonlight_system_v1/下面有307 folders

Raw/下面有999+個png files

sites/Moonlight system v1/

watchshift/下面有這些資料夾  
test_116613_T_v1/  
test_16613_T_v1/  
test1/  
test1201/  
test1351/  
test1419_openbackcrown/  
test1428/  
tets16613T_v2/  
test2/

然後現在資料跟結果存放在AWS S3的檔案結構是否在管理上方便? 以及要支援多站點都進行掃描可以順暢的上傳到AWS S3管理? 而這樣的AWS檔案結構跟database設定對在local跟AWS協同的authentication pipeline是否也是優良的? 這樣的AWS檔案結構跟database設定對在local跟AWS協同的authentication pipeline是否也是優良的? 這樣的AWS檔案結構跟database設定對檔案, images等地搜尋功能(譬如尋找AWS S3上某個uuid的image file是屬於哪個watchid, 是屬於哪個template)是否也是優良的? 請深入分析並將結果整理, 並提供建議, 譬如像Raw似乎有兩個新舊版本或其他有新舊版本, 請建議只選一個. 所以最後要總結你建議的AWS S3上面看的主要資料夾結構
```
## 結論

目前並不是一套統一的 S3 結構，而是至少四代／四種寫法同時存在：

1. 舊版扁平結構：`Raw/<uuid>`、`Analysis/<uuid>`
2. App task 結果結構：`Moonlight_system_v1/<WatchID>/Analysis/<ExpID>/...`
3. 新版結構：`sites/<SiteID>/watches/<WatchID>/runs/<RunID>/...`
4. Authentication 專用結構：`<WatchID>/runs/<ScanID>/authentication/...` 與 `authentication/...`

因此目前：

- Raw image 的新版 UUID/provenance 管理已經相對完整。
- Analysis task 的 image/json 結果仍主要走舊表、舊 prefix，UUID 與來源 image 關聯不完整。
- 多站點在「新版 Raw」部分已有雛形，但 Analysis、Authentication 還沒有完全納入同一個 Site/Run namespace。
- 本機設定目前關閉 AWS sync；SQLite 裡有大量待同步 outbox，不代表資料已經上傳到 S3/DynamoDB。
- 我建議 operational inspection data 最後只保留 `sites/<site>/watches/<watch>/runs/<run>/...` 這一套。`Raw/`、`Moonlight_system_v1/` 等只保留為歷史唯讀 prefix，停止新寫入。

另外，`auth-isolated-lab-001-...` 是刻意隔離的 authentication PREPROD/qualification bucket，不應和 production bucket 合併。

---

## 目前實際的本機與 S3 結構

本機 App capture 現在主要存成：

```
Local_Data/<WatchID>/
├── Raw/
│   └── <asset_uuid>.png
└── Analysis/
    └── Exp_<timestamp>_<random8>/
        ├── <TaskName>_<source_uuid>_<result>.png
        ├── <TaskName>_<source_uuid>_report.json
        ├── debug images
        └── yaml/json/result files
```

Raw image 由 [`process_and_sync_raw_image` (line 442)](D:/Provenance Laboratories projects/ImagingLibWatch/data_manager/local_storage.py:442) 產生 32 字元 UUID 檔名，再存進 `<WatchID>/Raw/`。

新版 S3 key builder 在 [`build_artifact_s3_key` (line 392)](D:/Provenance Laboratories projects/ImagingLibWatch/data_manager/local_storage.py:392)，預設產生：

```
sites/<SiteID>/watches/<WatchID>/runs/<RunID>/raw/<asset_uuid>.png

sites/<SiteID>/watches/<WatchID>/runs/<RunID>/
  experiments/<ExperimentID>/<artifact_type>/<filename>
```

但 App 目前執行 task analysis 時，仍直接手工組合另一套 key：

```
<DeviceID>/<WatchID>/Analysis/<ExpID>/<task-output-filename>
```

實作在 [App/main.py (line 34923)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:34923)，所以您的 bucket 才會出現：

```
Moonlight_system_v1/<很多 WatchID folders>/Analysis/...
```

而不是全部進入 `sites/...`。

### 您看到的各個 prefix 代表什麼

| S3 prefix                                    | 來源／用途                                                | 判定                        |
| -------------------------------------------- | ---------------------------------------------------- | ------------------------- |
| `Raw/`                                       | 舊版 `structured_s3_keys=false` 的 raw images           | 舊版，停止新寫入                  |
| `Analysis/`                                  | 舊版 DataManager analysis reports/results              | 舊版，停止新寫入                  |
| `Moonlight_system_v1/<WatchID>/Analysis/...` | App/task service 目前仍使用的 DeviceID 路徑                  | 仍在使用，但應遷移                 |
| `sites/<SiteID>/watches/...`                 | 新版 Raw、camera pipeline reports 等                     | 建議保留為唯一 operational 結構    |
| `<WatchID>/runs/<ScanID>/authentication/...` | 每次掃描的 authentication features/results                | 應併入 `sites/.../runs/...`  |
| `authentication/training/...`                | 跨站點 training corpus、bundle、training runs             | Authentication 全域資料，應獨立管理 |
| `watchshift/<TemplateID>/...`                | Template/view 對應的 watchshift reference               | 合理，但建議加 template version  |
| `AUTH_UI_.../runs/`                          | 很可能是 authentication UI/test 產生、以 WatchID 作為根目錄的 scan | 推論；repo 找不到這個固定名稱         |
| `auth-isolated-lab-001-...` bucket           | Authentication 隔離測試／PREPROD                          | 應保留獨立 bucket              |

Watchshift 的正式 key 是：

```
watchshift/<TemplateID>/<view>.toppoint1.png
```

定義在 [internalnum_config.py (line 1897)](D:/Provenance Laboratories projects/ImagingLibWatch/DB/templates/internalnum_config.py:1897)。

---

## Local SQLite 現況

資料庫同時保留兩代 schema。

### 舊表

```
raw_images
experiment_results
```

### 新版 normalized/provenance 表

```
watch_runs
  └── experiments
       └── point_instances
            └── capture_instances
                 └── image_assets
                      └── analysis_results_v2

artifact_records
feature_observation_batches
authentication_results
sync_outbox
```

Schema 位於 [DB/db_manager.py (line 47)](D:/Provenance Laboratories projects/ImagingLibWatch/DB/db_manager.py:47)。

這個新版模型方向是正確的：

- point metadata 與 image metadata 分開。
- 同一個 watch point 可以有多個 capture/image。
- `asset_role` 可表達 `raw_single`、`raw_bracket`、`hdr_fused`。
- 每個 image 有 `asset_id`。
- analysis result 可用 `result_id` 並透過 `asset_id/source_asset_id` 指回來源 image。

### 目前資料庫實際盤點

這份 `DB/watchDB.db` snapshot 中：

|項目|數量／狀態|
|---|---|
|`raw_images`|69|
|`image_assets`|68|
|採新 `sites/...` raw key|67|
|舊 `Raw/...` raw key|2|
|`experiment_results`|178|
|`analysis_results_v2`|**0**|
|`artifact_records`|69|
|`feature_observation_batches`|32|
|`authentication_results`|2|
|`sync_outbox`|**907，全部 pending**|

這表示 Raw 已大致進入新模型，但目前 App 產生的 task results 尚未透過 `register_analysis_result_v2()` 寫入新版 analysis 表。

檔案和 DB 也不是完全一對一：

- `Local_Data` 中有 534 個 Analysis files。
- 只有 91 個不同檔案路徑出現在 `experiment_results`。
- 443 個 Analysis files 在這份 DB snapshot 找不到對應 row。
- 178 個 result rows 中，有 86 個是重複指向已出現過的檔案路徑。

部分可能是舊資料庫遺留或 task debug outputs，但至少證明「所有 Analysis files 都已確實入庫」目前不成立。

---

## UUID 是否容易查找？

### Raw image：新版相對良好

Raw image 的：

```
檔名 UUID
= image_assets.asset_id
= DynamoDB sort_key
= uuid_pk 中的 UUID
```

新版 `image_assets` 也有：

- `watchid`
- `run_id`
- `template_id`
- `template_version`
- `view_name`
- `point_name`
- `capture_id`
- `internalnum1/internalnum2`
- `asset_role`
- `s3_key`
- `content_sha256`

本機已有 `local_lookup --identifier <uuid>`，底層是 [`LocalProvenanceQueryService.lookup_uuid` (line 140)](D:/Provenance Laboratories projects/ImagingLibWatch/data_manager/provenance_query.py:140)。

所以對新 Raw image 而言，從 UUID 找到 WatchID、Run、Template、Point、S3 key 是可行的。

### Analysis result：目前不夠好

目前 App 將 task outputs 寫入 `experiment_results` 時，沒有傳完整的：

- `result_id`
- `source_asset_id`
- `run_id`
- `experiment_id`
- `template_id/version`

結果 outbox 只臨時產生：

```
legacy_result_178
legacy_result_177
...
```

這不是全域 UUID，而只是本機 SQLite autoincrement 衍生值。多站點都可能產生 `legacy_result_178`。

而 task output 檔名中常見的 32 字元 UUID，多數是「來源 Raw image UUID」，不是這張 result image 自己的 UUID。

因此目前：

- 可從某些 result filename 猜到來源 raw UUID。
- 不能保證每個 result file 都有自己的全域唯一 UUID。
- 很多 result row 沒有直接 source-asset linkage。
- 本機 `lookup_uuid()` 主要查 `image_assets` 和 `artifact_records`，並不完整涵蓋這些 legacy result IDs。

---

## DynamoDB 現況

`moonlight-WatchAnalysisResults` 的設計是：

```
PK: WatchID
SK: sort_key = asset_id / result_id
```

每個 index item 另寫入：

```
uuid_pk = UUID#<asset_or_result_id>
uuid_sk = WATCH#<WatchID>#TYPE#<type>#TS#<timestamp>

source_asset_pk = ASSET#<source_asset_id>
source_asset_sk = TYPE#...#ID#...
```

實作在 [cloud_db.py (line 165)](D:/Provenance Laboratories projects/ImagingLibWatch/data_manager/cloud_db.py:165)。

程式提供：

- `query_by_uuid()`，預設查 `UuidIndex`
- `query_by_source_asset()`，預設查 `SourceAssetIndex`

但是這兩個 GSI 只是程式假設存在；repo 沒有 main results table 的 IaC provisioning。此次 AWS CLI 也因本機沒有 `default` AWS profile，無法確認實際 table 是否真的有：

```
UuidIndex
SourceAssetIndex
BrandModelIndex
```

所以目前應理解為：

- 程式「準備好寫入 UUID GSI 欄位」。
- 但不能確認 AWS table「已建立對應 GSI」。
- 若 GSI 不存在，`query_by_uuid()` 會直接失敗。
- 即使 GSI 存在，legacy Analysis result 的 `legacy_result_N` 仍不是合格的全域 UUID。
- 目前也沒有 TemplateID lookup GSI；依 template 查 assets 可能需要 DynamoDB Scan 或 Athena。

---

## Sync 與多站點評估

目前機器設定為：

```
security.enable_cloud_sync: false
sync.legacy_polling_enabled: false
sync.outbox_enabled: false
system.mode: simulation
system.site_id: laptop
```

因此這台機器目前不會自動同步。907 筆 outbox 全部 pending，正是這個結果。

新版 transactional outbox 本身設計不錯：

1. 先寫 SQLite 與 outbox。
2. 先上傳 S3。
3. 驗證檔案大小和 SHA-256 metadata。
4. S3 完成後才寫 DynamoDB catalog。
5. 最多嘗試五次，失敗進 dead-letter 狀態。

邏輯在 [outbox_dispatcher.py (line 47)](D:/Provenance Laboratories projects/ImagingLibWatch/data_manager/outbox_dispatcher.py:47)。

但要支援多站點，還缺少：

- 每台 production station 的穩定且唯一 `site_id/station_id`，不能使用 `laptop`。
- 所有 task output 必須共同使用中央 S3 key builder。
- Authentication operational keys 必須包含 SiteID。
- 只能啟用 outbox writer；不要同時啟用 legacy polling。
- DynamoDB GSI 必須由 IaC 正式建立及驗證。
- `legacy_result_<local integer>` 必須淘汰。
- Site、WatchID、RunID、ObjectID 都要具備跨站點唯一性。

所以目前架構「有多站點基礎」，但還不能稱為可安全擴展的多站點 production 架構。

---

## Authentication pipeline 評估

較好的部分：

- `auth-isolated-lab-001-...` 使用獨立 bucket，是正確的 fail-closed PREPROD 隔離。
- Training corpus 已使用適合 Athena/Glue 的 Hive partition：

```
authentication/training/samples/
  feature_schema_version=<...>/
  extractor_version=<...>/
  series=<...>/
  site_id=<...>/
  dt=YYYY-MM-DD/
  <sample_id>.json
```

定義在 [storage_layout.py (line 22)](D:/Provenance Laboratories projects/ImagingLibWatch/core/authentication/storage_layout.py:22)。

- Bundle 有 immutable version prefix、signature 與 target pointer。
- Authentication result/feature batch 都有穩定 hash-based ID。

需要改善的部分：

- Operational authentication data 目前是：

```
<WatchID>/runs/<ScanID>/authentication/features/...
<WatchID>/runs/<ScanID>/authentication/results/...
```

見 [runtime.py (line 748)](D:/Provenance Laboratories projects/ImagingLibWatch/core/authentication/integration/runtime.py:748)。

它沒有 SiteID，且和 Raw 的 `sites/<site>/watches/...` 不同根。

- Analysis source lineage 不完整，會削弱 authentication evidence 的可追溯性。
- 目前本機只有 provisional `NOT_EVALUATED` results，且相關 outbox 尚未同步。
- Production authentication bundles/corpora 最好放在獨立 private bucket，不要和 UI/site hosting、一般 scan data 混用相同權限與 lifecycle。

---

## 我建議最後只保留的 operational S3 結構

S3 並沒有真正的 folder，以下是 object-key prefix。建議唯一的新寫入結構為：

```
moonlight-system-v2/
└── sites/
    └── <site_id>/
        └── watches/
            └── <watch_id>/
                └── runs/
                    └── <run_id>/
                        ├── manifest/
                        │   └── run.json
                        │
                        ├── raw/
                        │   └── <view>/
                        │       └── <point>/
                        │           └── <capture_id>/
                        │               └── <asset_role>/
                        │                   └── <asset_uuid>.<ext>
                        │
                        ├── analysis/
                        │   └── <algorithm_name>/
                        │       └── <result_type>/
                        │           └── <result_uuid>.<ext>
                        │
                        ├── reports/
                        │   └── <artifact_type>/
                        │       └── <artifact_uuid>.<ext>
                        │
                        └── authentication/
                            ├── features/
                            │   └── <batch_id>.json
                            └── results/
                                └── <authentication_result_id>.json
```

Template 不建議放進每個 image key，因為 TemplateID/version 是 run metadata，不是 object ownership。應存在：

```
run.json
watch_runs
image_assets / analysis_results_v2
DynamoDB catalog item
```

時間也不必再做一層 folder；以 `run_id`、`started_at`、`captured_at` 和 DynamoDB time index 管理。日期 partition 只用在 Athena lake 與 authentication training corpus。

Watchshift 建議改成：

```
references/
└── watchshift/
    └── templates/
        └── <template_id>/
            └── versions/
                └── <template_version>/
                    └── <view>/
                        └── toppoint1.png
```

Authentication production 資料最好使用另一個 bucket：

```
moonlight-auth-production/
└── authentication/
    ├── authored/
    ├── training/samples/...
    ├── training-runs/...
    └── bundles/
        ├── <bundle_version>/...
        ├── <bundle_version>.sig
        ├── current.json
        └── targets/...
```

`auth-isolated-lab-001-...` 繼續作為 PREPROD 隔離 bucket，不合併。

---

## 哪些舊 prefix 應停止使用

停止所有新寫入：

```
Raw/
Analysis/
Moonlight_system_v1/<WatchID>/Analysis/
<WatchID>/Analysis/
<WatchID>/runs/...              # operational authentication 舊根
```

保留但逐步搬遷：

```
watchshift/                     → references/watchshift/...
authentication/training/...     → authentication 專用 bucket
```

不要直接在 S3 Console 用「移動」處理。S3 move 實際上是 copy + delete，會讓 SQLite/DynamoDB 裡的 `s3_key` 全部失效。正確方式是：

1. 產生 S3 Inventory。
2. 以 SQLite、DynamoDB、S3 Inventory 建 migration manifest。
3. 複製到新 key。
4. 驗證 size、SHA-256、UUID、WatchID、RunID。
5. 更新 catalog。
6. 經過觀察期後，用 lifecycle 移除舊 object。

---

## 資料庫最重要的改善

短期、最小安全修改：

1. App 所有 task outputs 改用 `register_analysis_result_v2()`。
2. 每個 output file 自己取得真正的 `result_uuid`。
3. 必須填入 `source_asset_id`、`watchid`、`run_id`、`experiment_id`、`template_id/version`。
4. 所有 S3 key 只能由一個中央 builder 產生，刪除散落的手工 f-string。
5. 加入 indexes：

```
analysis_results_v2(result_identifier)
analysis_results_v2(s3_key)
analysis_results_v2(watchid, run_id, algorithm_name)
artifact_records(s3_key)
artifact_records(capture_instance_id)
```

6. 對 `s3_key` 加合理的 unique/partial-unique constraint。
7. 將 legacy tables 降為 compatibility mirror，不再當主要 writer。

DynamoDB 長期建議拆開：

```
MoonlightWatchEntries
MoonlightAssetCatalog
MoonlightWatchTemplates
MoonlightAuthentication...
```

其中 `MoonlightAssetCatalog` 最好直接：

```
PK: ObjectID = <asset/result/artifact UUID>
```

如此從 UUID 查找不需要 GSI，直接 `GetItem`。另建立：

- WatchRunIndex
- TemplateVersionIndex
- SourceAssetIndex
- SiteCapturedAtIndex

每筆 catalog item必須包含：

```
object_id
object_type
source_object_id
site_id
station_id
watch_id
run_id
experiment_id
template_id
template_version
view_name
point_name
capture_id
asset_role/result_type
s3_bucket
s3_key
content_sha256
captured_at
sync_status
```

---

## 最後判斷

目前架構不是完全不良；新版 Raw、normalized DB、outbox、SHA-256 驗證與 authentication isolation 都是正確方向。但現在處於「遷移一半」：

- Raw 已進新版。
- Analysis 仍在舊版。
- Authentication 有自己的第三套路徑。
- DynamoDB UUID lookup 依賴未確認存在的 GSI。
- 本機 cloud sync 完全停用。
- S3 bucket 中混有歷史、production、test、reference、authentication 等多種 lifecycle。

所以我建議選定並只保留：

```
sites/<site_id>/watches/<watch_id>/runs/<run_id>/...
```

作為所有 inspection operational data 的唯一結構；不要再寫入頂層 `Raw/` 或 `Moonlight_system_v1/`。Template 用 DB/manifest 關聯，時間用 Run 與 timestamp index，UUID 用 Asset Catalog 查詢。

這次是只讀分析，沒有修改任何檔案。最大未驗證項是 AWS 上實際 DynamoDB GSI、bucket versioning/lifecycle/policy，因本機沒有可用 AWS profile；正式遷移前必須在 AWS Console 或有權限的 profile 下核對。

10:09 AM





```

```





```

```



