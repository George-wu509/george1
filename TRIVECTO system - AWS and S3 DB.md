
|                         |     |
| ----------------------- | --- |
| [[#### WatchDB.db詳細分析]] |     |
|                         |     |
|                         |     |
|                         |     |
|                         |     |
|                         |     |

#### WatchDB.db詳細分析
```
在./DB資料夾裡面的watchDB.db是local DB(因為目前git沒有追蹤所以付上example file). 在這local DB file裡面有16個Tables跟9個Indices. Indices跟Tables的關係是甚麼? 是table裡面的flat column嗎? 那這些indics的作用是甚麼? 也請詳細解釋這些16個tables是甚麼? 請具體舉例以App裡面做一次完整watchentry scan或template creation來解釋這些tables的作用以及怎麼建立的. 並用兩種版本詳細解釋給intern去了解概念以及介紹給senior engineer技術細節. 最後則是這些tables那些會跟AWS同步? 甚麼時候同步及怎麼變成AWS DynamoDB資料格式
```

## 先講結論

我以附件 DB 為主要分析對象，並和目前 repository 的程式碼及 `DB/watchDB.db` 交叉比對。

附件實際包含：

- 15 個應用程式 tables
- 1 個 SQLite 自動建立的 `sqlite_sequence`
- 合計 16 tables
- 9 個程式明確建立的 indices
- 另外有 13 個 SQLite 因 `PRIMARY KEY` 自動建立的 `sqlite_autoindex_*`

因此：

> Index 不是 table 裡的 flat column。  
> 它是 SQLite 另外維護的查找結構，通常是 B-tree，內容類似「某個 column value → 對應資料列位置」。

另外有 schema 版本差異：

- 附件：15 個 app tables + `sqlite_sequence`
- 目前工作目錄的 `DB/watchDB.db`：21 個 app tables + `sqlite_sequence`，以及 27 個明確 indices
- 現行程式多了 `experiments`、`artifact_records`、`feature_observation_batches`、`authentication_results`、`authentication_learning_samples`、`authentication_bundle_releases`

如果用現在的 App 開啟附件 DB，[`DatabaseManager._init_db()` (line 47)](D:/Provenance Laboratories projects/ImagingLibWatch/DB/db_manager.py:47) 會用 `CREATE TABLE IF NOT EXISTS` 和 `_ensure_column()` 自動把舊 DB 往目前 schema 擴充。

附件本身是 template-oriented example：有 32 筆 templates、8 筆 users、1 筆 pending template outbox；scan、image、analysis tables 都是空的。

---

# Version A：給 Intern 的概念解釋

## 1. 可以把 DB 想成一座工廠

- `local_templates`：工作說明書／拍攝 recipe
- `watch_metadata`：這支錶的基本資料卡
- `watch_runs`：某一次完整掃描工作單
- `point_instances`：這次掃描到了哪一個物理位置
- `capture_instances`：在該位置執行哪一組拍攝，例如 Standard 或 HDR
- `image_assets`：真正產生的每一張影像檔案
- `analysis_results_v2`：AI 對某張影像產生的結果
- `ui_asset_registry`：App 畫面要顯示哪個結果檔的捷徑
- `sync_outbox`：等待寄到 AWS 的寄件匣

核心關係是：

```
一支錶 WatchID
└─ 一次掃描 watch_runs
   └─ 多個拍攝位置 point_instances
      └─ 每個位置可有多個 capture_instances
         ├─ Standard capture
         │  └─ 一張 raw_single image_asset
         └─ HDR capture
            ├─ 多張 raw_bracket image_assets
            └─ 一張 hdr_fused image_asset
               └─ 多個 analysis_results_v2
```

這些目前只是程式建立的邏輯關係。附件所有 tables 的 `PRAGMA foreign_key_list` 都是空的，也就是沒有真正的 SQLite `FOREIGN KEY` constraint。

例如 `image_assets.capture_instance_id` 雖然應該指向 `capture_instances.capture_instance_id`，SQLite 本身不會阻止一個不存在的 ID 被寫入。

---

## 2. Index 和 column 的差別

假設 `raw_images` 有一百萬筆：

```
SELECT *
FROM raw_images
WHERE local_path = 'D:\...\abc.png';
```

沒有 index 時，SQLite 可能從第一列一路掃到最後一列。

有：

```
CREATE INDEX idx_path ON raw_images(local_path);
```

SQLite 可以先查 `idx_path`，快速找到符合的 row。

它不是：

```
raw_images
├─ id
├─ watchid
├─ local_path
└─ idx_path       ← 不是這樣
```

而比較像：

```
raw_images table             idx_path B-tree
row 1: ... abc.png           abc.png → row 1
row 2: ... xyz.png           xyz.png → row 2
```

代價是：

- 查詢通常變快
- insert/update/delete 變慢一些，因為 index 也要更新
- DB 檔案變大
- composite index 的 column 順序很重要

例如：

```
(watchid, internalnum1, internalnum2)
```

適合：

```
WHERE watchid=?
WHERE watchid=? AND internalnum1=?
WHERE watchid=? AND internalnum1=? AND internalnum2=?
```

但只用 `WHERE internalnum2=?` 通常不能有效利用這個 index。

---

# 3. 附件中的 16 個 Tables

| Table                 | 一筆 row 代表什麼                        | 何時建立／寫入                                                   | AWS                                                                |
| --------------------- | ---------------------------------- | --------------------------------------------------------- | ------------------------------------------------------------------ |
| `local_templates`     | 一個 template 的一個 version            | 建立、修改、下載 template 時                                       | 同步至 `WatchTemplates` DynamoDB                                      |
| `watch_metadata`      | 一支 WatchEntry 的目前基本資料與完整 JSON      | WatchEntry 完成／儲存時 upsert                                  | row 本身不直接 mirror；同一份 WatchEntry JSON 由 App 直接送 DynamoDB            |
| `watch_registry`      | 某 WatchID 首次與最後出現時間                | 呼叫 `register_watch()` 時                                   | <span style="color:rgb(0, 0, 0)">不同步；目前主要流程沒有找到實際 call site</span> |
| `watch_runs`          | 一次 scan/routine execution          | scan 開始時 insert、結束時改 status                               | 本表不直接同步；run_id 被放進 image/result 的 Dynamo item                      |
| `point_instances`     | 某 run 中一個 watch point 的執行 instance | 第一次進入某 view/point 時                                       | 不直接同步                                                              |
| `capture_instances`   | 某 point 的一次 Standard/HDR capture   | 每個 capture 開始前                                            | 不直接同步                                                              |
| `image_assets`        | 一個實體影像檔                            | 每一張 standard、HDR bracket、HDR fused 圖產生時                   | 檔案到 S3，metadata 到 Dynamo catalog                                   |
| `analysis_results_v2` | AI 對一個 asset 的一個分析結果               | algorithm 執行完成時                                           | 檔案到 S3、索引到 Dynamo，也可送 Data Lake                                    |
| `raw_images`          | 舊版的一張 raw image record             | 舊 workflow 直接寫；V3 也會 compatibility double-write           | legacy sync 或相容 outbox 路徑                                          |
| `experiment_results`  | 舊版分析結果、report 或 WatchEntry JSON    | 分析、YAML/PDF report、WatchEntry 完成時                         | legacy/outbox sync；V3 分析也會 double-write                            |
| `ui_asset_registry`   | 某 watch/view/point/task 在 UI 顯示的檔案 | capture 或分析完成後 upsert                                     | 不同步                                                                |
| `users`               | 離線登入使用者 cache                      | 建帳號或從 AWS users table 下載時                                 | 透過明確 user API，不走 outbox                                            |
| `sync_outbox`         | 一件尚待處理或已完成的 AWS 工作                 | domain data 與 outbox 在同一 transaction 建立                   | 本身不傳上 AWS                                                          |
| `sync_outbox_archive` | 超過 retention 的已完成 outbox 工作        | done 超過 30 天後搬入                                           | 不同步                                                                |
| `lake_etl_batches`    | 一批 Data Lake outbox IDs            | Lake ETL 取 batch 時                                        | 控制 Glue/Iceberg commit，本表本身不上傳                                     |
| `sqlite_sequence`     | SQLite 的 `AUTOINCREMENT` 計數器       | 因 `raw_images`、`experiment_results` 使用 AUTOINCREMENT 自動建立 | SQLite 內部使用，不同步                                                    |

其中 `local_templates.data`、`watch_metadata.Full_JSON`、`point_instances.hardware_cfg`、`image_assets.metadata`、`analysis_results_v2.data_json`、`sync_outbox.payload` 都是 JSON serialized 成 SQLite `TEXT`。

所以這個 DB 是混合模型：

- 常查詢的欄位，例如 `WatchID`、`Brand`、`run_id`，放 flat columns
- 結構複雜、常演進的資料，放 JSON text
- 現有 indices 都建立在 flat columns 上，沒有 index JSON 內部欄位

---

# 4. 9 個明確 Indices 的用途

|Index|Table(columns)|用途|
|---|---|---|
|`idx_path`|`raw_images(local_path)`|用路徑尋找 legacy image record，例如解密或結果反查|
|`idx_watch_meta_brand_model`|`watch_metadata(Brand, Model)`|App 用品牌、型號搜尋 WatchEntry|
|`idx_asset_cap_inst`|`image_assets(capture_instance_id)`|從一次 capture 找出全部 assets|
|`idx_asset_internalnums`|`image_assets(watchid, internalnum1, internalnum2)`|依錶與 point/capture identity 找影像|
|`idx_outbox_status`|`sync_outbox(status, target)`|找 pending S3/catalog/lake 工作|
|`idx_outbox_source`|`sync_outbox(source_table, source_id, target)`|找某 domain record 的特定同步工作與 dependency|
|`idx_outbox_retention`|`sync_outbox(status, updated_at)`|找已完成且超過保留期限的工作|
|`idx_outbox_archive_source`|`sync_outbox_archive(source_table, source_id, target)`|稽核某 record 的歷史同步結果|
|`idx_lake_etl_batch_status`|`lake_etl_batches(status, created_at)`|找最舊的 prepared/processing Lake batch|

附件還有 13 個 `sqlite_autoindex_*`，負責 `PRIMARY KEY` uniqueness。例如：

- `local_templates(TemplateID, version)`
- `ui_asset_registry(WatchID, ViewName, PointName, TaskName)`
- `image_assets(asset_id)`
- `watch_runs(run_id)`

附件 schema 的效能缺口包括：

- `analysis_results_v2.asset_id` 沒有 index
- `image_assets.run_id/view_name/point_name` 沒有 index
- `point_instances.run_id`、`capture_instances.point_instance_id` 沒有 index
- `watch_runs(template_id, template_version)` 沒有 index

現行程式已補上其中一部分，例如 `idx_analysis_asset`、`idx_asset_run_point`、`idx_watch_runs_template`。

---

# 5. 完整 WatchEntry Scan 範例

假設執行：

```
WatchID = W000123
Template = Rolex_16613T / v1
Point = Front.macropoint1
Standard = std_1
HDR = hdr_1，3 張 bracket
```

## Step 1：載入 template

App 從 `local_templates` 讀出：

```
TemplateID = Rolex_16613T
version = v1
data = {
  watchView: {
    Front: {
      macropoint1: {
        hardware settings,
        standard_captures: [...],
        hdr_captures: [...]
      }
    }
  }
}
```

Template 本身是一個 JSON document，而不是把每個 template point 拆成 DB rows。

## Step 2：建立 scan

[`execute_routine()` (line 560)](D:/Provenance Laboratories projects/ImagingLibWatch/core/workflow_manager.py:560) 會建立：

```
watch_runs
run_id = Run_20261002_...
watchid = W000123
template_id = Rolex_16613T
template_version = v1
status = running
```

現行版還會建立 `experiments`；附件版本尚未包含這張 table。

## Step 3：建立 point

進入 `Front.macropoint1`：

```
point_instances
point_instance_id = pt_abcd1234
run_id = Run_...
view_name = Front
point_name = macropoint1
hardware_cfg = {...JSON...}
```

同一個 point 如果連續執行 Standard 和 HDR，可以共用同一個 `point_instance_id`。

## Step 4：Standard capture

建立：

```
capture_instances
capture_instance_id = cap_std1234
capture_id = std_1
capture_type = single
```

硬體執行 `execute_template_point()`，產生一張影像，然後建立：

```
image_assets
asset_id = asset_std...
capture_instance_id = cap_std1234
asset_role = raw_single
asset_index = NULL
local_path = ...png
s3_key = sites/.../raw/...png
```

同時 double-write 一筆 `raw_images`，供舊 UI、CLI、report 或 legacy sync 使用。

## Step 5：HDR capture

建立第二筆：

```
capture_instances
capture_instance_id = cap_hdr1234
capture_id = hdr_1
capture_type = hdr
```

如果拍三張 exposure，會產生：

```
image_assets:
1. raw_bracket, asset_index=0
2. raw_bracket, asset_index=1
3. raw_bracket, asset_index=2
4. hdr_fused,   asset_index=NULL
```

因此「一個 watch point」不再等於「一張 image」：

```
1 point
→ 2 captures
→ 5 image assets
```

這就是目前 refactor 的核心模型。HDR 實作在 [`_execute_capture_instance_v3()` (line 1039)](D:/Provenance Laboratories projects/ImagingLibWatch/core/workflow_manager.py:1039)。

## Step 6：AI analysis

Runtime 按 `analysis_target_role` 選：

- Standard 通常選 `raw_single`
- HDR 通常選 `hdr_fused`
- 不會把所有 bracket 都送給一般 single-image algorithm

每個 algorithm result 寫入：

```
analysis_results_v2
result_id = res_...
asset_id = 被分析的 image asset
algorithm_name = ocr_service
result_type = image_mask/json
data_json = {...}
```

並 compatibility double-write `experiment_results`。

這裡仍存在有意保留的 single-image adapter：很多 algorithm API 只接受一個 `local_path`，所以 workflow 從多個 assets 中選定一張 primary asset，而不是讓 algorithm 自己接收整個 list。

## Step 7：UI 與 report

`ui_asset_registry` 會記錄：

```
(W000123, Front, macropoint1, raw_image) → 路徑
(W000123, Front, macropoint1, ocr_service) → overlay 路徑
```

它是 UI shortcut，不是完整 image history。

它的主鍵沒有 `capture_id`、`asset_id` 或 `asset_index`，所以同一 point/task 的後一次結果會覆蓋前一次。完整歷史必須查 `image_assets`／`analysis_results_v2`。

DB 的「選一張圖給舊功能」邏輯在 [`get_preferred_raw_image_path()` (line 2314)](D:/Provenance Laboratories projects/ImagingLibWatch/DB/db_manager.py:2314)：

```
hdr_fused → raw_single → 其他可用 asset
```

這是 backward compatibility，不代表 DB 只能存一張。

## Step 8：完成 WatchEntry

App 完成頁會：

1. 將完整 WatchEntry 寫入 `experiment_results`
2. 將 Brand、Model、Reference 等 flat 欄位及完整 JSON 寫入 `watch_metadata`
3. 直接在 worker 中送 WatchEntry JSON 到 DynamoDB
4. 將 `watch_runs.status` 更新為 completed

程式位置在 [`_on_upload_s3_dynamo()` (line 37040)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:37040) 和 [`_upload_watchentry_metadata_core()` (line 36992)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:36992)。

---

# 6. Template Creation 範例

假設 intern 建立 `Rolex_16613T / v1`。

## Template 內容

App 建立一個 `WatchTemplate` object，內容包含：

- Template identity
- Brand、Reference
- 各 watch views
- 每個 point 的位置、角度、light、camera
- `standard_captures`
- `hdr_captures`
- material points
- optional reference-image manifest

資料模型在 [template_structure.py (line 16)](D:/Provenance Laboratories projects/ImagingLibWatch/DB/templates/template_structure.py:16)。

## 儲存至 SQLite

App serialize 成 dict 後呼叫：

```
save_local_template(
    TemplateID="Rolex_16613T",
    version="v1",
    data={...}
)
```

最後是：

```
INSERT OR REPLACE INTO local_templates
    (TemplateID, version, data, updated_at)
VALUES (?, ?, ?, ?)
```

所以 20 個 template points 不會產生 20 筆 `point_instances`。它們全部存在同一筆 `local_templates.data` JSON；只有真正執行 scan 時才建立 runtime instances。

相關流程在：

- [`_serialize_template_to_payload()` (line 14629)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:14629)
- [`_save_template_to_db_and_cloud()` (line 14953)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:14953)
- [`save_local_template()` (line 1776)](D:/Provenance Laboratories projects/ImagingLibWatch/DB/db_manager.py:1776)

## Template 上 AWS

正式 Save/Upload 流程是：

```
GUI thread
→ serialize
→ 先寫 local_templates
→ background worker
→ upload watchshift reference images 到 S3
→ 更新 manifest
→ save_template_cloud()
→ WatchTemplates.put_item()
```

DynamoDB item 大致是：

```
{
  "TemplateID": "Rolex_16613T",
  "version": "v1",
  "schema_version": "3.1",
  "watchView": {
    "Front": {
      "macropoint1": {
        "standard_captures": [...],
        "hdr_captures": [...]
      }
    }
  },
  "updated_at": "DynamoDB Decimal"
}
```

`to_dynamo_item()` 會移除可由 canonical config 重建的 manifest 和未使用 factory placeholders，以避免超過 DynamoDB item size；見 [`WatchTemplate.to_dynamo_item()` (line 70)](D:/Provenance Laboratories projects/ImagingLibWatch/DB/templates/template_structure.py:70)。

---

# Version B：給 Senior Engineer 的技術細節

## 1. 關聯模型是 application-enforced，不是 relationally enforced

附件沒有任何 declared foreign keys。主要 logical joins 是：

```
watch_metadata.WatchID
  → watch_runs.watchid
  → point_instances.watchid
  → image_assets.watchid

local_templates.(TemplateID, version)
  → watch_runs.(template_id, template_version)

watch_runs.run_id
  → point_instances.run_id
  → capture_instances.run_id
  → image_assets.run_id

point_instances.point_instance_id
  → capture_instances.point_instance_id

capture_instances.capture_instance_id
  → image_assets.capture_instance_id

image_assets.asset_id
  → analysis_results_v2.asset_id

raw_images.id
  → experiment_results.raw_image_id
```

最後一個 legacy relation 已變得不可靠：V3 `register_analysis_result_v2()` compatibility double-write 時把 `raw_image_id` 寫成 `NULL`。新 provenance 應以 `asset_id`、`run_id`、`capture_instance_id` 為主。

## 2. Point-level 與 image-level metadata

目前模型的正確分工是：

- Point-level：
    - `point_instances.view_name`
    - `point_instances.point_name`
    - `point_instances.hardware_cfg`
    - `internalnum1`
- Capture-level：
    - `capture_instances.capture_id`
    - `capture_type`
    - `internalnum2`
- Image-level：
    - `image_assets.asset_role`
    - `asset_index`
    - `local_path`
    - `s3_key`
    - exposure、Z、hash 等 metadata
- Result-level：
    - `analysis_results_v2.algorithm_name`
    - `result_type`
    - `data_json`

這符合「one point → multiple images + optional HDR」的方向。

## 3. Local DB 不會逐 table 複製到 DynamoDB

真正架構是：

```
SQLite domain row
   │
   ├─ local file ───────────────→ S3 object
   │
   └─ metadata snapshot
       └─ sync_outbox
          └─ CloudDatabaseManager.index_record()
             └─ flattened DynamoDB catalog item
```

Image/result 的 DynamoDB item 在 [`index_record()` (line 156)](D:/Provenance Laboratories projects/ImagingLibWatch/data_manager/cloud_db.py:156) 建構：

```
{
  "WatchID": "W000123",
  "sort_key": "asset-or-result-id",
  "s3_key": "sites/site/watches/W000123/runs/Run_x/raw/file.png",
  "record_type": "raw_image",
  "view_name": "Front",
  "point_name": "macropoint1",
  "capture_id": "hdr_1",
  "internalnum1": "0002",
  "internalnum2": "0001",
  "site_id": "station_01",
  "run_id": "Run_x",
  "experiment_id": "Exp_x",
  "capture_instance_id": "cap_x",
  "template_id": "Rolex_16613T",
  "template_version": "v1",
  "uuid_pk": "UUID#asset-id",
  "uuid_sk": "WATCH#W000123#TYPE#raw_image#TS#...",
  "metadata_raw": {
    "...": "full metadata"
  }
}
```

如果是 analysis result：

- Dynamo `sort_key` 使用 `result_id`
- `source_asset_id` 指回原影像
- 可建立 `source_asset_pk/source_asset_sk` 供 GSI 查詢

Python `float` 不能直接安全送 DynamoDB，所以 [`_float_to_decimal()` (line 136)](D:/Provenance Laboratories projects/ImagingLibWatch/data_manager/cloud_db.py:136) 會遞迴轉成 `Decimal(str(value))`；非有限數值會變 `None`。

## 4. 三種不同 AWS 資料形狀

### Asset/result catalog

目的地：`moonlight-WatchAnalysisResults`

- flat query fields
- nested `metadata_raw`
- 檔案本身在 S3
- 大型 metadata 超過約 380 KB 時只留 compact pointer

### Template

目的地：`moonlight-WatchTemplates`

- `TemplateID` + `version`
- 保存 nested `watchView`
- 上傳前 normalize
- float → Decimal
- 超過約 380 KB 拒絕寫入

見 [`save_template_cloud()` (line 444)](D:/Provenance Laboratories projects/ImagingLibWatch/data_manager/cloud_db.py:444)。

### 完整 WatchEntry

完成按鈕使用：

```
item_data_aws = json.loads(json.dumps(item_data), parse_float=Decimal)
results_table.put_item(Item=item_data_aws)
```

也就是直接傳整個 `watchentry.to_dynamo_item()`，不是經過 `index_record()` flatten。

這裡存在需要在 AWS 驗證的 schema 風險：

- `index_record()` 假設 key 包含 `WatchID` + `sort_key`
- `WatchRecordManager.put_item_split()` 也寫 `WatchID` + `sort_key`
- App 完整 WatchEntry upload 特別確保的是 `version`，卻沒有明確補 `sort_key`

因此必須以 AWS `DescribeTable.KeySchema` 為準確認 `moonlight-WatchAnalysisResults` 真正是：

- 只有 `WatchID` partition key，或
- `WatchID + sort_key`，或
- 舊版的 `WatchID + version`

單靠 repository 註解目前有互相不一致之處。

---

# 7. 哪些 Tables 會同步、何時同步

|Local table|S3|DynamoDB|時機|
|---|---|---|---|
|`image_assets`|是|`WatchAnalysisResults` index|asset registration transaction 後|
|`analysis_results_v2`|有結果檔時|`WatchAnalysisResults` index|analysis result registration 後|
|`raw_images`|legacy 是|legacy catalog|legacy capture path 或相容流程|
|`experiment_results`|有檔案時|catalog|legacy result/report path|
|`local_templates`|reference images 才上 S3|`WatchTemplates`|template Save/Upload 或 template outbox|
|`watch_metadata`|否|不直接 mirror；完整 WatchEntry 由 App 直接 put|Finish/Upload 按鈕|
|`users`|否|`WatchUsers`|explicit user API／startup cache refresh|
|`watch_runs`|否|不直接；欄位 denormalize 到 asset/result items|local only|
|`point_instances`|否|不直接|local only|
|`capture_instances`|否|不直接|local only|
|`ui_asset_registry`|否|否|local UI cache|
|`watch_registry`|否|否|local only|
|`sync_outbox`|否|否|local delivery control|
|`sync_outbox_archive`|否|否|local audit|
|`lake_etl_batches`|否|否|local Iceberg commit control|
|`sqlite_sequence`|否|否|SQLite internal|

## Transactional outbox 時序

[`OutboxDispatcher` (line 9)](D:/Provenance Laboratories projects/ImagingLibWatch/data_manager/outbox_dispatcher.py:9) 的行為是：

1. domain row 和 outbox row 同一個 SQLite transaction commit
2. worker 每 2 秒 poll
3. 同一 source record 必須先完成 `target=s3`
4. 才允許執行 `target=catalog`
5. 最多嘗試 5 次
6. 失敗以 exponential delay retry
7. 5 次後轉 `dead_letter`
8. S3 dead-letter 會使相依 catalog 工作也轉 dead-letter
9. done 超過 30 天移到 `sync_outbox_archive`
10. `target=lake` 不由 dispatcher 處理，而由 `LakeETLJob` 寫 Glue Iceberg

S3 key 的現行格式在 [`build_artifact_s3_key()` (line 392)](D:/Provenance Laboratories projects/ImagingLibWatch/data_manager/local_storage.py:392)：

```
sites/{site}/watches/{watch}/runs/{run}/raw/{uuid}.png
```

或：

```
sites/{site}/watches/{watch}/runs/{run}/
experiments/{experiment}/{artifact_type}/{uuid}.yaml
```

## Legacy polling

如果開啟 `legacy_polling_enabled`，舊 worker 會查：

```
raw_images WHERE synced=0 LIMIT 50
experiment_results WHERE synced=0 LIMIT 50
```

完成 S3/Dynamo 後把 `synced=1`。

在 outbox 模式下，dispatcher 只更新 `sync_outbox.status`，不一定更新 source table 的 `synced`。所以 `synced=0` 在 outbox 架構中不能單獨證明「沒有上傳」，應以 outbox status 與 S3/Dynamo 驗證為準。

---

# 8. 目前環境的實際同步狀態

目前 local config 是：

- `system.mode: simulation`
- `security.enable_cloud_sync: false`
- `security.force_upload_to_S3: false`
- `sync.legacy_polling_enabled: false`
- `sync.outbox_enabled: false`

見 [system_config.local.yaml (line 1)](D:/Provenance Laboratories projects/ImagingLibWatch/config/system_config.local.yaml:1)。

因此目前設定下：

> 不會自動將 capture/analysis outbox 傳到 AWS。

目前工作目錄的 `DB/watchDB.db` 有 907 筆 outbox，全部是 `pending`；附件有 1 筆 pending `local_templates → catalog`。這和同步 worker 被關閉的設定一致。

此外，simulation/hybrid mode 在沒有 `force_upload_to_S3=true` 時會使用 `NullCloudSync`，避免測試資料污染 production Data Lake。

---

# 9. Single-image assumptions 的剩餘風險

搜尋 App、CLI、analysis、report 與 DB 後，主要殘留是：

1. `ui_asset_registry` 一個 point/task 只能保存一個 path，會覆蓋舊 capture。
2. `get_preferred_raw_image_path()` 刻意只回傳一張最佳圖片。
3. 多數 algorithm CLI/API 仍接受單一 `local_path`；目前由 workflow 先選 `raw_single` 或 `hdr_fused`。
4. 若一個 capture 產生兩張同 role 的 primary assets，目前多處使用 `next(...)` 只取第一張。
5. `raw_images` 與 `experiment_results` 是 single-file-per-row，但可用多 rows 表達多圖；它們不是新模型的 authoritative relation。
6. report/UI 若只查 `ui_asset_registry`，看不到全部 bracket history。
7. 附件 schema 對 `run_id`、`asset_id` 等常用 logical join 欄位缺少部分 indices。
8. 沒有 foreign keys，transaction rollback 以外的 crash 或 manual SQL 仍可能留下 orphan rows。
9. template 的「local save」路徑預設會建立 template catalog outbox；若日後開啟 outbox，需確認 Draft/Local-only 按鈕是否真的允許自動上雲。
10. direct WatchEntry put、direct template put、legacy polling、outbox 四條 cloud path 必須避免同時啟用造成 duplicate writes。

新版 [`CaptureImageResult` (line 22)](D:/Provenance Laboratories projects/ImagingLibWatch/DB/templates/watchentry_structure.py:22) 已經有：

```
image_paths[]
source_image_paths[]
asset_ids[]
s3_keys[]
hdr_image_path
```

表示 shared model 已經往 multi-image 方向調整；目前最大的剩餘差距在 UI shortcut、單圖 algorithm boundary、report selection，以及 cloud writer 一致性。

---

## 建議的人工驗證

1. 在 staging 執行一個 Standard + 3-bracket HDR point。
2. 驗證 cardinality：
    - `watch_runs`: 1
    - `point_instances`: 1
    - `capture_instances`: 2
    - `image_assets`: 5
    - `raw_images`: 5 compatibility rows
3. 確認 `hdr_fused` 和三張 `raw_bracket` 都有不同 `asset_id`／S3 key。
4. 確認 analysis result 的 `source_asset_id` 指向 `hdr_fused`。
5. 驗證 outbox 一定是 S3 done 後才 catalog done。
6. 用 AWS `DescribeTable` 確認 `WatchAnalysisResults` 和 `WatchTemplates` 的真正 key schema。
7. 驗證完整 WatchEntry direct put 是否符合該 key schema。
8. 驗證 PDF/report 使用的是 intended primary asset，而非任意第一張 bracket。
9. 檢查所有 `dead_letter` 與長期 pending outbox。
10. 為 logical FK 做 orphan audit query。

本次沒有修改任何檔案或 DB；所有 SQLite 檢查都使用 read-only connection。
