
|     |     |
| --- | --- |
|     |     |
|     |     |
|     |     |



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


```

```


```

```