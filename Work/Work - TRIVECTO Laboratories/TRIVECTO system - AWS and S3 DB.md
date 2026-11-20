
|                                        |     |
| -------------------------------------- | --- |
| [[#### WatchDB.db詳細分析]]                |     |
| [[#### 新AWS S3建議結構]]                   |     |
| [[#### 但不同的站點為什麼要分site_id跟station_id]] |     |
| [[#### 修改code新AWS S3建議結構]]             |     |
|                                        |     |
|                                        |     |

#### WatchDB.db詳細分析
```
在./DB資料夾裡面的watchDB.db是local DB(因為目前git沒有追蹤所以付上example file). 在這local DB file裡面有16個Tables跟9個Indices. Indices跟Tables的關係是甚麼? 是table裡面的flat column嗎? 那這些indics的作用是甚麼? 也請詳細解釋這些16個tables是甚麼? 請具體舉例以App裡面做一次完整watchentry scan或template creation來解釋這些tables的作用以及怎麼建立的. 並用兩種版本詳細解釋給intern去了解概念以及介紹給senior engineer技術細節. 最後則是這些tables那些會跟AWS同步? 甚麼時候同步及怎麼變成AWS DynamoDB資料格式
```

																															## 先講結論`Q

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





```
那請幫我比較AWS S3架構([1]最新跟本機儲存方式相似的架構, [2]舊版結構, [3]新版結構, 以及[4]operational S3 結構), 如果目前所有Raw image, Analysis task 的 image/json 結果都放在Amazon S3/Buckets/Moonlight-system-v2之下因為目前只有一台device(Moonlight-system-v2). 然後之後Raw images存進<WatchID>/Raw/, Analysis task 的 image/json 結果也都存進<WatchID>/Analysis/Exp_<timestamp>_<random8>/, 也就是跟本機 App capture儲存方式把Local_Data換成Amazon S3/Buckets/Moonlight-system-v2一樣. 那這樣在AWS S3儲存的方式跟目前的舊版扁平結構的S3儲存方式(也就是用Raw/<uuid>, Analysis/<uuid>), 以及用新版結構sites/<SiteID><WatchID>/run...

這幾種結構在App執行watchentry拍攝上傳到AWS S3, 以及多device deploy站點都 為上傳到AWS S3(這時最新跟本機儲存方式相似的架構Amazon S3/Buckets/後面接的就是不同device), 以及是否容易搜尋到譬如要找到所有屬於template_id=Rolex_16613T or watchid = W000123的所有images, 或找到template_id=Rolex_16613T的所有internalnum1/internalnum2 = 0004/0001的images以及results. 或者這4種結構對authentication process是否哪個有明顯優缺點? 請完整全面分析這4種結構對目前App功能性包括儲存, 搜尋, 分析等功能優缺點及是否推薦哪一種? 或者只要S3 key builder建好或database等或Athena有配合好則4種都可以?

這是operational S3 結構  
moonlight-system-v2/  
└── sites/  
└── <site_id>/  
└── watches/  
└── <watch_id>/  
└── runs/  
└── <run_id>/  
├── manifest/  
│ └── run.json  
│  
├── raw/  
│ └── <view>/  
│ └── <point>/  
│ └── <capture_id>/  
│ └── <asset_role>/  
│ └── <asset_uuid>.<ext>  
│  
├── analysis/  
│ └── <algorithm_name>/  
│ └── <result_type>/  
│ └── <result_uuid>.<ext>  
│  
├── reports/  
│ └── <artifact_type>/  
│ └── <artifact_uuid>.<ext>  
│  
└── authentication/  
├── features/  
│ └── <batch_id>.json  
└── results/  
└── <authentication_result_id>.json
```

## 結論

我建議把第 4 種 operational 結構作為未來唯一的 canonical S3 layout，但做兩項調整：

1. 在 `analysis/` 下保留 `experiment_id`。
2. `device_id/station_id` 不要直接取代 `site_id`；預設放在 manifest、DB 與 object metadata。若需要嚴格的 device 級 IAM 隔離，再增加 `stations/<station_id>/` 或發放 run-scoped STS credentials。

建議結構：

```
s3://<deployment-bucket>/operational/v1/
└── sites/<site_id>/
    └── watches/<watch_id>/
        └── runs/<run_id>/
            ├── manifest/
            │   └── run.json
            ├── raw/
            │   └── <view>/<point>/<capture_id>/<asset_role>/<asset_uuid>.<ext>
            ├── analysis/
            │   └── <experiment_id>/<algorithm_name>/<result_type>/<result_uuid>.<ext>
            ├── reports/
            │   └── <artifact_type>/<artifact_uuid>.<ext>
            └── authentication/
                ├── features/<batch_id>.json
                └── results/<authentication_result_id>.json
```

但必須強調：**S3 key 不是搜尋資料庫**。`template_id`、`internalnum1/internalnum2` 這類跨 watch、跨 site 查詢，應由 DynamoDB/Iceberg/Athena 負責，不應全部塞入 object key。

---

## 目前 repo 的實際狀態

目前不是單一 S3 架構，而是多種結構同時存在：

- 核心 builder 預設產生：
    
    ```
    sites/<site>/watches/<watch>/runs/<run>/raw/<uuid>
    sites/<site>/watches/<watch>/runs/<run>/experiments/<exp>/<kind>/<uuid>
    ```
    
    見 [local_storage.py (line 392)](D:/Provenance Laboratories projects/ImagingLibWatch/data_manager/local_storage.py:392)。
    
- 本機仍使用：
    
    ```
    Local_Data/<watch_id>/Raw/
    Local_Data/<watch_id>/Analysis/<Exp_timestamp_random8>/
    ```
    
    見 [local_storage.py (line 418)](D:/Provenance Laboratories projects/ImagingLibWatch/data_manager/local_storage.py:418)。
    
- App 與 workflow 部分分析路徑繞過 builder，直接建立：
    
    ```
    <device_id>/<watch_id>/Analysis/<exp_id>/...
    ```
    
    見 [App/main.py (line 34934)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:34934)、[workflow_manager.py (line 1852)](D:/Provenance Laboratories projects/ImagingLibWatch/core/workflow_manager.py:1852)。
    
- authentication runtime 又是：
    
    ```
    <watch_id>/runs/<scan_id>/authentication/features/...
    <watch_id>/runs/<scan_id>/authentication/results/...
    ```
    
    見 [runtime.py (line 754)](D:/Provenance Laboratories projects/ImagingLibWatch/core/authentication/integration/runtime.py:754)。
    

所以現況最大的問題不是「選錯其中一種」，而是**寫入路徑沒有全部經過同一個 key builder**。

另外，目前 checked-in config 是：

- bucket：`moonlight-system-v2`
- `site_id: laptop`
- `station_id: laptop-NITRO2025`
- `DeviceID: Moonlight system v1`
- cloud sync 與 outbox 預設都關閉

見 [system_config.yaml (line 3)](D:/Provenance Laboratories projects/ImagingLibWatch/config/system_config.yaml:3)、[system_config.yaml (line 58)](D:/Provenance Laboratories projects/ImagingLibWatch/config/system_config.yaml:58)。實際 production override 可能不同，但這份設定本身不會上傳。

---

## 四種結構總比較

|面向|1. 每 device bucket、本機鏡像|2. 舊版扁平 Raw/Analysis|3. 現有 sites/watch/run|4. Operational|
|---|---|---|---|---|
|App 寫入複雜度|低|最低|中|中高|
|人工瀏覽|容易|很差|容易找到 run|最好|
|多圖片/HDR 表達|依賴 DB|完全依賴 DB|UUID 不衝突，但 key 看不出角色|最清楚|
|WatchID prefix 查找|bucket 已知時容易|不行|site 已知時容易|site 已知時容易|
|template/internalnum 查找|仍需 DB|需 DB|需 DB|仍需 DB|
|多 device 集中管理|差|好|好|最好|
|device 級 IAM 隔離|每 bucket 很強|很差|通常只能到 site|可設計到 site/station/run|
|Lifecycle/事件路由|每 bucket 管理繁瑣|只能 Raw/Analysis|可到 site/run/type|最精細|
|DB 遺失後人工重建|尚可|最差|中等|最好|
|與現有 refactor 契合|普通|差|好|最好|
|遷移成本|中|最低|最低|中|

S3 的「資料夾」其實只是 object key prefix，不是真正的目錄；多幾層本身不會產生檔案系統式的昂貴 traversal。[AWS S3 object key 文件](https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-keys.html)

以目前工作量來看，四種結構的 PUT/GET 效能差距不會是決策主因。S3 每個 partitioned prefix 至少可支援數千次每秒請求，真正比較容易影響 Athena 的是大量小 JSON、小檔案與錯誤 partition 策略。[AWS S3 效能文件](https://docs.aws.amazon.com/AmazonS3/latest/userguide/optimizing-performance.html)

---

## 1. 每 device bucket，加上本機鏡像結構

例如：

```
s3://moonlight-system-v2/
└── W000123/
    ├── Raw/
    └── Analysis/Exp_20261005_.../
```

未來每台 device 都有不同 bucket。

優點：

- 和本機 `Local_Data` 一致，開發者及現場人員最容易理解。
- 已知 device 與 WatchID 時，可以直接 prefix-list。
- 每台 device 使用獨立 bucket policy、KMS key、retention，隔離清楚。
- 一台 device 設定錯誤，理論上不會寫入另一台 device bucket。

缺點：

- S3 bucket 名稱是全 AWS 全球唯一，而且只能使用符合 bucket 規則的名稱；目前 `DeviceID` 包含空白，不能直接當 bucket 名稱。
- 每增加 device，就要重複設定 bucket policy、versioning、encryption、lifecycle、event notification、inventory、metadata table。
- 同一支 watch 若在兩台設備執行，資料會散落在兩個 bucket。
- device 換機、改名或退役時，歷史資料的 business ownership 會和實體設備生命週期綁死。
- Athena/Glue 必須跨多個 bucket/location，集中查詢和災難復原較複雜。
- `template_id`、`internalnum` 仍然不在 key 裡，搜尋能力沒有實質改善。

這種方式適合完全獨立、彼此不共享資料的 appliance，但不適合日後多站點、集中 authentication/training、跨 device 查詢。

若採用一個 bucket、device 作為第一層 prefix，會減少 bucket 管理成本，但仍有 watch 被 device 拆散的問題。

---

## 2. 舊版扁平結構

```
Raw/<uuid>.<ext>
Analysis/<uuid>.<ext>
```

優點：

- builder 最簡單。
- UUID object key 和 business metadata 完全解耦，template 或 watch metadata 更正時不需要搬 object。
- 對舊版相容最容易。
- key 不暴露 WatchID、template 等資訊。

缺點：

- S3 Console 幾乎無法判斷 object 屬於誰。
- DB/DynamoDB index 遺失或漏寫後，object 幾乎無法恢復分類。
- 無法從 prefix 判斷 run、capture、HDR bracket、fused image、algorithm 或 result type。
- IAM 最小權限只能大致切成 `Raw/*`、`Analysis/*`，無法限制某 site/device/watch。
- Lifecycle、S3 event notification、法務保留也只能做很粗的分類。
- orphan object 很難修復。

這個結構不是不能運作；只要 DB 永遠完整，它可以正常執行 App、分析與報告。但它把所有營運可靠性押在 catalog 上，對需要 audit、authentication evidence 與災難復原的系統不理想。

---

## 3. 現有 `sites/<site>/watches/<watch>/runs/<run>` 結構

現有核心 builder：

```
sites/<site>/watches/<watch>/runs/<run>/raw/<uuid>
sites/<site>/watches/<watch>/runs/<run>/
  experiments/<experiment>/<kind>/<uuid>
```

優點：

- 已具備 stable site、watch、run、experiment 邊界。
- 同一 watch 的歷史 run 容易列出。
- run 層級很適合 retry、resume、完成狀態與 retention。
- 已經存在於核心 builder 和測試中，遷移成本最低。
- UUID filename 保留 immutable identity。

缺點：

- raw key 看不出 `view/point/capture/asset_role`。
- 多張 HDR brackets 與 fused image 雖不會撞名，但只能查 DB 才知道各自角色。
- 缺少 manifest，無法只靠 S3 判斷某次 run 是否完整。
- App、workflow、authentication 尚有多個 direct f-string writer，實際資料不一定進到這個結構。
- `get_site_id()` 會 fallback 到 DeviceID，容易混淆「實體 site」和「station/device」。
- analysis、reports、authentication 的分類尚未統一。

這是合理的過渡架構。如果現在只想做最小安全修正，可以先保留它，加入 manifest、raw semantic segments，並清掉所有 bypass builder 的 writer，逐步演進成第 4 種。

---

## 4. Operational 結構

你提出的 operational tree 最符合目前「一個 point 多圖片，加上可選 HDR」的 refactor。

特別是：

```
raw/<view>/<point>/<capture_id>/<asset_role>/<asset_uuid>
```

可自然表達：

```
.../std_1/raw_single/<uuid>.png
.../hdr_1/raw_bracket/<uuid0>.png
.../hdr_1/raw_bracket/<uuid1>.png
.../hdr_1/raw_bracket/<uuid2>.png
.../hdr_1/hdr_fused/<uuid>.png
```

目前 capture workflow 已經使用 `raw_single`、`raw_bracket`、`hdr_fused` 和 `asset_index`，見 [workflow_manager.py (line 1039)](D:/Provenance Laboratories projects/ImagingLibWatch/core/workflow_manager.py:1039)。因此 operational tree 和 shared model 是對齊的。

主要優點：

- 多圖與 HDR 不再只是藏在 JSON/DB 中。
- 可以用 manifest 完整描述 run 的 template、station、assets、checksums、analysis lineage。
- 容易依 site、watch、run、artifact type 設 IAM、Lifecycle 和事件規則。AWS 支援用 prefix 限制 `ListBucket` 與 object access。[AWS prefix IAM policy](https://docs.aws.amazon.com/AmazonS3/latest/userguide/amazon-s3-policy-keys.html)
- Raw、analysis、report、authentication evidence 可以使用不同保存期限；S3 Lifecycle 可依 prefix 或 object tags 過濾。[AWS Lifecycle filters](https://docs.aws.amazon.com/AmazonS3/latest/userguide/intro-lifecycle-filters.html)
- DB 暫時不可用時，仍能從 key 與 manifest 重建大部分 provenance。
- 最適合報告、authentication、法務 audit 和資料修復。

需要注意：

- 你原本的 operational tree 沒有 `experiment_id`。目前程式有獨立 `runs` 和 `experiments`，而且同一 run 可能重新執行 manual analysis 或新版 model，所以建議保留：
    
    ```
    analysis/<experiment_id>/<algorithm>/<result_type>/<uuid>
    ```
    
- 不要把 `template_id`、`internalnum` 全部加入路徑。這些是查詢維度，不是 storage ownership boundary。
    
- algorithm 名稱改名不應搬動舊結果；舊 object 應保持 immutable。
    
- `manifest/run.json` 若會不斷覆寫，需要 S3 Versioning；更嚴謹的方式是 immutable manifest revisions 加 final completion marker。
    
- analysis result 必須保存 `source_asset_id` 或 `source_asset_ids`，不能只靠資料夾位置推論 lineage。
    

---

## 三種實際搜尋需求

### 找到 `WatchID=W000123` 的所有 images/results

目前最有效的方法其實已經存在：

- DynamoDB `WatchID` 是 partition key。
- `query_watch_history()` 可以直接 Query，不需要 Scan。

見 [cloud_db.py (line 300)](D:/Provenance Laboratories projects/ImagingLibWatch/data_manager/cloud_db.py:300)。

四種 S3 layout 的影響：

- 第 1 種：必須先知道 device bucket。
- 第 2 種：S3 prefix 完全無法查，只能查 DynamoDB。
- 第 3/4 種：若知道 site，可直接 list watch prefix；跨 site 仍應查 DB。

### 找到 `template_id=Rolex_16613T` 的所有 images/results

現在 DynamoDB item 已存 `template_id`，但沒有 template GSI/query method；不建立 GSI 時只能 Scan。

Athena `analysis_facts` 有 `template_id` 欄位，也有 template filter，見 [analytics_query.py (line 435)](D:/Provenance Laboratories projects/ImagingLibWatch/data_manager/analytics_query.py:435)。但目前 raw `image_assets` 不會寫入 lake，只寫 S3 與 catalog，因此「所有 raw images」不能單靠現有 `analysis_facts` 找齊。

### 找到 template + `internalnum1/internalnum2=0004/0001`

這是目前真正的缺口：

- 本機 `image_assets` 已有 internalnum 欄位與 index，見 [db_manager.py (line 237)](D:/Provenance Laboratories projects/ImagingLibWatch/DB/db_manager.py:237)。
- raw image 的 DynamoDB metadata 可帶 internalnum。
- 但 `register_analysis_result_v2()` 沒有從 source asset 繼承 internalnum 到 result snapshot。
- Lake ETL 雖把 internalnum 列在排除/上下文 key 清單，實際 Iceberg dimensions/schema 沒有 `internalnum1/internalnum2` 欄位，見 [lake_etl.py (line 375)](D:/Provenance Laboratories projects/ImagingLibWatch/data_manager/lake_etl.py:375)、[lake_etl.py (line 492)](D:/Provenance Laboratories projects/ImagingLibWatch/data_manager/lake_etl.py:492)。

所以目前無論採哪種 S3 key，這個查詢都不能保證完整。

建議新增一個 DynamoDB GSI：

```
GSI_Template_PK = TEMPLATE#Rolex_16613T

GSI_Template_SK =
  IN1#0004#IN2#0001#
  TS#<captured_at>#
  WATCH#W000123#
  TYPE#raw_image|analysis_result#
  ID#<uuid>
```

如此可以：

- Query partition key：取得 template 全部 assets/results。
- 再以 `begins_with(IN1#0004#IN2#0001#)` 找指定 image identity。
- raw 與 analysis result 使用相同索引格式。
- analysis result 必須繼承 source asset 的 internalnum。

Athena/Iceberg 方面，建議分成：

- `asset_catalog`：每個 raw/derived image 一列。
- `analysis_facts`：每個 measurement/metric 一列。
- 用 `source_asset_id` join。

不要用 WatchID 或 template_id 做大量高基數實體 partition；建議 partition 使用 `dt`、必要時加 `site_id`，其餘維度留成 Iceberg columns。AWS 也建議 partition key 應符合常用 filter、避免過多 partition 與大量小檔案。[Athena data optimization](https://docs.aws.amazon.com/athena/latest/ug/performance-tuning-data-optimization-techniques.html)

---

## Authentication 的影響

要區分兩種 authentication。

### Watch authenticity 流程

Operational 結構最適合：

```
runs/<run>/authentication/features/<batch>.json
runs/<run>/authentication/results/<result>.json
```

每個結果仍需記錄：

- source asset UUID/hash
- template ID/version
- feature schema version
- extractor/model/bundle version
- site/station/camera
- final/provisional 狀態
- run/scan ID
- content SHA-256

但是 authentication training corpus 與 model bundles 不應塞進單一 watch run：

```
authentication/training/samples/
  feature_schema_version=.../
  extractor_version=.../
  series=.../
  site_id=.../
  dt=.../

authentication/bundles/<bundle_version>/
authentication/authored/
authentication/training-runs/
```

目前 repo 已經使用這類 Hive-style training layout，見 [storage_layout.py (line 52)](D:/Provenance Laboratories projects/ImagingLibWatch/core/authentication/storage_layout.py:52)。這些是跨 watch 的共享模型資產，應保持獨立 top-level namespace。

### AWS/device 身分驗證與 IAM

S3 layout 不會取代：

- IAM role/temporary credentials
- AWS IoT X.509 identity
- bucket policy
- KMS
- application user login

它只能改善 authorization scope。

安全排序大致為：

- 每 device bucket：device 隔離最直接，但管理成本最高。
- flat layout：最難做 site/device 級最小權限。
- sites layout：可限制到 site。
- operational：可限制到 site/run/type；若加入 station segment，可限制到單一 station。

如果每台 station 必須完全不能寫到其他 station 的 namespace，建議二選一：

```
sites/<site>/stations/<station>/watches/...
```

或由中央服務發放只允許特定 run prefix 的短期 STS credentials。

單純把 `device_id` 寫進 key 並不代表該 device 已被驗證；client 可以偽造字串。

---

## 是否「只要 key builder、DB、Athena 做好，四種都可以」

答案是：**功能上接近可以，營運上不等價。**

只要 catalog 完整，四種都可以完成：

- App 拍攝
- S3 upload
- analysis dispatch
- report generation
- WatchID/template/internalnum 查詢

但 DB/Athena 無法完全補償：

- flat layout 的 IAM 粗粒度。
- catalog 遺失後的重建困難。
- prefix-based lifecycle/event routing。
- 人工事故調查與 orphan recovery。
- 每 device bucket 的跨設備資料碎片化與管理成本。
- run 完整性與 manifest 問題。

所以不應說「四種都一樣」。對這個 repo：

- 第 2 種只適合 legacy read。
- 第 1 種適合單機，但不適合成長成集中式多站點。
- 第 3 種適合作為短期過渡。
- 第 4 種加上 experiment 層、中央 catalog，是長期推薦。

---

## 建議實作順序

依目前 AGENTS.md 的安全原則，建議：

1. 定義共同 artifact metadata contract：
    
    ```
    key_schema_version
    site_id
    station_id/device_id
    watch_id
    run_id
    experiment_id
    template_id/template_version
    view_name
    point_name/internalnum1
    capture_id/internalnum2
    asset_role/asset_index
    asset_id
    source_asset_id/source_asset_ids
    algorithm_name/version
    result_type
    content_sha256
    captured_at
    ```
    
2. 建立唯一 `S3KeyBuilder`，支援：
    
    ```
    legacy_flat
    device_local_mirror
    structured_v3
    operational_v1
    ```
    
    所有 App、workflow、authentication writer 都必須呼叫它，移除目前的 direct f-string。
    
3. 先修 DB/serialization：
    
    - analysis result 繼承 internalnum、asset role、site/template。
    - DynamoDB 新增 template/internalnum GSI。
    - raw assets 進 `asset_catalog` lake table。
    - Iceberg schema 加 internalnums、asset role、result type。
4. 再加入 run manifest 與完成 marker。
    
5. 新資料只寫 operational；舊資料不必立即搬動：
    
    - DB 保留舊 `s3_key`。
    - 新增 `key_schema_version`。
    - reader 支援所有 legacy keys。
    - 有需要再用 S3 Batch Operations 複製，而不是在 App 啟動時搬移。
6. 更新 IAM、Lifecycle、event notifications 和 authentication infra。
    

---

## 必做人工驗證

- 一個 point：兩張 standard images。
- 一個 HDR capture：多張 brackets、fused image、fusion fallback。
- 中斷網路後恢復，上傳不得重複或漏失。
- 同一 WatchID 在兩台 station 執行。
- `WatchID` 查詢包含 raw、analysis、reports、authentication。
- `template_id + internalnum1/internalnum2` 查詢同時找到 source images 和 results。
- report 仍能取得 preferred primary image，同時保留完整 image list。
- authentication result 可以追溯到 exact source asset 和 model bundle。
- station A 不可寫入 station B namespace。
- 舊 flat keys 仍能被 gallery、CLI、report 和 provenance query 讀取。

這次是架構分析，沒有修改任何檔案。最值得先處理的不是搬 S3 objects，而是統一 builder、補齊 analysis result 的 internalnum lineage，以及建立 raw asset catalog。




#### 新AWS S3建議結構
```
請具體舉例解說你的S3建議結構, 為什麼比其他三種更好, 並解釋這句話"S3 key 不是搜尋資料庫。template_id、internalnum1/internalnum2 這類跨 watch、跨 site 查詢，應由 DynamoDB/Iceberg/Athena 負責，不應全部塞入 object key ". 那由本來的本機的Local_Data/<watch_id>/Raw/跟Local_Data/<watch_id>/Analysis/<Exp_timestamp_random8>/是怎麼上傳到S3變成S3建議結構? 請在仔細檢查S3建議結構是否還有缺點或會有Error? 如果沒有則準備下一步修正code

結論  
我建議把第 4 種 operational 結構作為未來唯一的 canonical S3 layout，但做兩項調整：

1. 在 analysis/ 下保留 experiment_id。
2. device_id/station_id 不要直接取代 site_id；預設放在 manifest、DB 與 object metadata。若需要嚴格的 device 級 IAM 隔離，再增加 stations/<station_id>/ 或發放 run-scoped STS credentials。  
    建議結構：  
    s3://<deployment-bucket>/operational/v1/  
    └── sites/<site_id>/  
    └── watches/<watch_id>/  
    └── runs/<run_id>/  
    ├── manifest/  
    │ └── run.json  
    ├── raw/  
    │ └── <view>/<point>/<capture_id>/<asset_role>/<asset_uuid>.<ext>  
    ├── analysis/  
    │ └── <experiment_id>/<algorithm_name>/<result_type>/<result_uuid>.<ext>  
    ├── reports/  
    │ └── <artifact_type>/<artifact_uuid>.<ext>  
    └── authentication/  
    ├── features/<batch_id>.json  
    └── results/<authentication_result_id>.json

但必須強調：S3 key 不是搜尋資料庫。template_id、internalnum1/internalnum2 這類跨 watch、跨 site 查詢，應由 DynamoDB/Iceberg/Athena 負責，不應全部塞入 object key。

建議實作順序  
依目前 AGENTS.md 的安全原則，建議：

1. 定義共同 artifact metadata contract：  
    key_schema_version  
    site_id  
    station_id/device_id  
    watch_id  
    run_id  
    experiment_id  
    template_id/template_version  
    view_name  
    point_name/internalnum1  
    capture_id/internalnum2  
    asset_role/asset_index  
    asset_id  
    source_asset_id/source_asset_ids  
    algorithm_name/version  
    result_type  
    content_sha256  
    captured_at
2. 建立唯一 S3KeyBuilder，支援：  
    legacy_flat  
    device_local_mirror  
    structured_v3  
    operational_v1  
    所有 App、workflow、authentication writer 都必須呼叫它，移除目前的 direct f-string。
3. 先修 DB/serialization：
    - analysis result 繼承 internalnum、asset role、site/template。
    - DynamoDB 新增 template/internalnum GSI。
    - raw assets 進 asset_catalog lake table。
    - Iceberg schema 加 internalnums、asset role、result type。
4. 再加入 run manifest 與完成 marker。
5. 新資料只寫 operational；舊資料不必立即搬動：
    - DB 保留舊 s3_key。
    - 新增 key_schema_version。
    - reader 支援所有 legacy keys。
    - 有需要再用 S3 Batch Operations 複製，而不是在 App 啟動時搬移。
6. 更新 IAM、Lifecycle、event notifications 和 authentication infra。  
    必做人工驗證

- 一個 point：兩張 standard images。
- 一個 HDR capture：多張 brackets、fused image、fusion fallback。
- 中斷網路後恢復，上傳不得重複或漏失。
- 同一 WatchID 在兩台 station 執行。
- WatchID 查詢包含 raw、analysis、reports、authentication。
- template_id + internalnum1/internalnum2 查詢同時找到 source images 和 results。
- report 仍能取得 preferred primary image，同時保留完整 image list。
- authentication result 可以追溯到 exact source asset 和 model bundle。
- station A 不可寫入 station B namespace。
- 舊 flat keys 仍能被 gallery、CLI、report 和 provenance query 讀取。
```

結論：第 4 種 `operational/v1` 方向正確，也確實比另外三種好；但目前版本仍有幾個會造成權限漏洞、覆寫或錯誤分類的問題，不能原封不動直接實作。我建議修正後再把它定為唯一 canonical layout。

## 修正版建議結構

因為驗證需求明確包含「station A 不可寫入 station B namespace」，建議把 `station_id` 納入路徑，而不是只放 metadata：

```
s3://<deployment-bucket>/operational/v1/
└── sites/<site_id>/
    └── stations/<station_id>/
        └── watches/<watch_id>/
            └── runs/<run_id>/
                ├── manifest/
                │   ├── revisions/
                │   │   └── <revision_id>.json
                │   ├── run.final.json
                │   └── _SUCCESS
                ├── raw/
                │   └── <view_key>/<point_key>/<capture_id>/
                │       └── <asset_role>/<asset_id>.<ext>
                ├── analysis/
                │   └── <experiment_id>/<algorithm_name>/
                │       └── <result_type>/<result_id>.<ext>
                ├── reports/
                │   └── <artifact_type>/<artifact_id>.<ext>
                └── authentication/
                    ├── features/<batch_id>.json
                    └── results/<authentication_result_id>.json
```

如果未來確定採用「每次 run 發放限定 prefix 的 STS credentials」，才可以省略 `stations/<station_id>`。目前 repository 看不到完整的 run-scoped STS 機制，所以預設加入 station 比較安全。

IAM 可限制為類似：

```
arn:aws:s3:::<bucket>/operational/v1/sites/${site_id}/stations/${station_id}/*
```

這樣 station A 的 credentials 從 S3 policy 層就無法寫入 station B。

---

## 具體範例：一個 point 有兩張標準圖和一組 HDR

假設：

```
site_id       = taipei-lab
station_id    = station-a
watch_id      = W-16610-00042
run_id        = Run_20261006_143012_a1b2c3
experiment_id = Exp_20261006_143015_deadbeef
view_key      = front
point_key     = macropoint22
```

### 兩張 Standard images

```
.../raw/front/macropoint22/std_1/raw_single/
    8e83f4....jpg
    93c81a....jpg
```

兩張圖都屬於同一個 capture，但有不同的：

```
asset_id
asset_index = 0 / 1
content_sha256
captured_at
```

`asset_index` 不一定需要放入 key，因為 `asset_id` 已保證唯一；它應保留在 DB、manifest 和 catalog metadata。

### HDR brackets 與 fused image

```
.../raw/front/macropoint22/hdl_1/raw_bracket/
    a101....jpg
    a102....jpg
    a103....jpg

.../raw/front/macropoint22/hdl_1/hdr_fused/
    b201....tiff
```

metadata 中記錄：

```
{
  "asset_id": "b201...",
  "asset_role": "hdr_fused",
  "source_asset_ids": ["a101...", "a102...", "a103..."],
  "hdr_fused_generated": true
}
```

如果 fusion 失敗、改用某張 bracket 當 fallback，不能仍假裝它是真正 fused image。建議：

```
{
  "asset_role": "hdr_fused",
  "source_asset_ids": ["a102..."],
  "quality_flags": {
    "hdr_fused_generated": false,
    "hdr_fallback_used": true
  }
}
```

或者進一步使用明確角色 `hdr_fallback`。目前 repository 的 fallback 訊息沒有完整持久化，這是必須修正的缺口。

### 分析輸出

例如 HDR texture 分析：

```
.../analysis/Exp_20261006_143015_deadbeef/
    lume_hand_texture_service/
        mask/
            res_71ce....png
        measurements/
            res_983a....json
```

DB/metadata 另外保存：

```
{
  "source_asset_id": "b201...",
  "source_asset_ids": ["a101...", "a102...", "a103..."],
  "internalnum1": "0022",
  "internalnum2": "0002",
  "algorithm_name": "lume_hand_texture_service",
  "algorithm_version": "2.3.1",
  "result_type": "mask"
}
```

### 報告

```
.../reports/pdf/report_f923....pdf
.../reports/watchentry/watchentry_3d1a....json
```

---

## 本機路徑如何轉成 S3 key

不是把本機路徑直接做字串取代。

正確流程是：

```
本機檔案
  + DB/run context
  + image-level metadata
          ↓
     S3KeyBuilder
          ↓
  canonical S3 key
          ↓
    sync_outbox
          ↓
       S3 upload
```

### Raw 範例

原本本機：

```
Local_Data/W-16610-00042/Raw/8e83f4.jpg
```

上傳時從 `image_assets` 或 capture context 取得：

```
site_id       = taipei-lab
station_id    = station-a
run_id        = Run_20261006_143012_a1b2c3
view_key      = front
point_key     = macropoint22
capture_id    = std_1
asset_role    = raw_single
asset_id      = 8e83f4...
```

產生：

```
operational/v1/sites/taipei-lab/stations/station-a/
watches/W-16610-00042/runs/Run_20261006_143012_a1b2c3/
raw/front/macropoint22/std_1/raw_single/8e83f4....jpg
```

實際上新版 capture workflow 已經有更細的本機結構：

```
Local_Data/<watch_id>/runs/<run_id>/
views/<view>/points/<point>/captures/<capture_id>/<uuid>
```

但同樣不能依靠相對路徑推算 S3 key；DB metadata 才是權威來源。

### Analysis 範例

本機：

```
Local_Data/W-16610-00042/Analysis/
Exp_20261006_143015_deadbeef/71ce.png
```

加上 DB context：

```
algorithm_name = lume_hand_texture_service
result_type    = mask
result_id      = res_71ce
```

產生：

```
.../analysis/Exp_20261006_143015_deadbeef/
lume_hand_texture_service/mask/res_71ce.png
```

### 舊資料怎麼辦

已經有 `s3_key` 的舊紀錄不能在 App 啟動時重新推導或搬移：

- 舊 DB 繼續保留原 `s3_key`。
- reader 根據 `key_schema_version` 支援舊 key。
- 新資料才寫 `operational_v1`。
- 尚未上傳、但 metadata 不完整的舊檔案，不可虛構 `view/point/capture`。
- 真正需要搬移時，透過獨立 migration inventory 和 S3 Batch Operations 做 copy。
- copy 驗證 hash 成功後才更新 catalog；不要立即刪除舊 object。

---

## 為什麼第 4 種比另外三種好

|結構|優點|主要問題|
|---|---|---|
|`legacy_flat`：`Raw/<uuid>`、`Analysis/<uuid>`|最簡單|無 site/watch/run 邊界；IAM、Lifecycle、事故調查都困難|
|`device_local_mirror`：`<device>/<watch>/Analysis/<exp>`|容易對照本機目錄|把 device 當資料所有者；同一 watch 在兩台 station 執行會被拆散；本機目錄變動會污染雲端 contract|
|`structured_v3`：目前部分 `sites/.../runs/.../experiments/...`|已有 site/watch/run 概念|沒有 schema version；raw 缺少 view/point/capture/role；不同 writer 仍各自拼 key|
|`operational_v1`|有版本、run 邊界、artifact 類別、HDR role、manifest、IAM 和 lifecycle 邊界|必須配合 DB contract、唯一 builder、manifest 完成語意與 immutable upload|

第 4 種的真正優勢不是「路徑比較漂亮」，而是：

- `run` 成為完整 provenance、保留、刪除與重跑的操作單位。
- Standard、bracket、fused、fallback 不會全混在同一個 `Raw/`。
- `experiment_id` 不會在分析結果中消失。
- IAM 可以限制 site/station。
- Lifecycle 可以依 raw、analysis、reports、authentication 分類。
- S3 event consumer 可以從固定 artifact 類別判斷事件，而不是搜尋檔名。
- `operational/v1` 讓未來 schema 變更能新增 `v2`，不用破壞舊 reader。

---

## 「S3 key 不是搜尋資料庫」的具體意思

S3 最有效率的查找方式本質上是：

```
bucket + 完整 key
```

或：

```
ListObjectsV2(Prefix="已知前綴")
```

它不能直接執行：

```
WHERE template_id = 'TPL-Rolex-01'
  AND internalnum1 = '0022'
  AND internalnum2 = '0002'
  AND algorithm_name = 'lume_hand_texture_service'
```

假設 key 是：

```
sites/taipei/stations/a/watches/W1/runs/R1/raw/...
```

當你只知道 `template_id=TPL-01` 時，根本不知道該列哪個 site、station、watch、run prefix。

如果把 `template_id` 放到最前面：

```
templates/TPL-01/sites/.../watches/...
```

那麼按照 `watch_id` 查詢又失去有效 prefix。再加入 internalnum 後，會面臨不同查詢順序：

```
template → internalnum → watch
watch → run → internalnum
site → template → algorithm
internalnum → algorithm → captured_at
```

單一樹狀 key 不可能同時服務所有查詢順序。若為每種順序複製一份 object，又會造成：

- 重複儲存。
- lifecycle 不一致。
- 修改與刪除困難。
- provenance 指向多份「相同來源」。
- 遺漏其中一份 index object。

此外，S3 沒有 rename。修改 key 等同於 copy 新 object，再 delete 舊 object。因此把會改名或重新分類的欄位塞進 key，會把 metadata 修改變成大量物件搬移。

合理分工應是：

### S3

負責：

- 保存 bytes。
- immutable object address。
- coarse IAM namespace。
- lifecycle、retention、event notification。
- 依已知 run/watch prefix 瀏覽或災難復原。

### DynamoDB

負責即時操作查詢，例如：

```
給我 W-16610-00042 的所有 runs
給我 TPL-01 + internalnum1=0022 + internalnum2=0002 的最新 assets
給我 asset b201 的所有分析結果
給我 authentication result 的 exact source assets
```

可能的索引概念：

```
主索引：
PK = WATCH#<watch_id>
SK = RUN#<run_id>#ASSET#<asset_id>

Template/Internalnum GSI：
GSI_PK = TEMPLATE#<template_id>
GSI_SK = I1#<internalnum1>#I2#<internalnum2>#<captured_at>#<asset_id>
```

實際上可能需要兩個 GSI，因為「只查 template」和「只查 internalnum」是不同 access pattern，不能把所有需求勉強塞進一個 GSI。

### Iceberg/Athena

負責：

- 跨 site、跨 watch 的歷史分析。
- template/version 比較。
- 演算法版本品質趨勢。
- HDR 與 Standard 結果比較。
- 大量 joins、aggregations 和 audit。

簡單說：

> S3 key 回答「這個 object 放在哪裡」；DynamoDB/Athena 回答「哪些 object 符合我的條件」。

---

## 審核後發現的實際問題

目前 code 尚未符合上述 canonical layout。

1. 現有 builder 沒有 `operational/v1`，raw 仍只有 `raw/<uuid>`，分析仍採 `experiments/<experiment>/<kind>`。[local_storage.py (line 392)](D:/Provenance Project/ImagingLibWatch/data_manager/local_storage.py:392)
    
2. `artifact_type="results/<algorithm>"` 會先經過 `_safe_component()`，其中 `/` 被轉成 `_`，實際會變成 `results_algorithm`，不是預期的兩層目錄。[local_storage.py (line 1001)](D:/Provenance Project/ImagingLibWatch/data_manager/local_storage.py:1001)
    
3. `get_site_id()` 會 fallback 到 `station_id`、`DeviceID`。這可能把 station 誤當成 site，造成相同實體 site 出現多個 namespace。[local_storage.py (line 346)](D:/Provenance Project/ImagingLibWatch/data_manager/local_storage.py:346)
    
4. 還有大量 direct f-string，沒有統一走 builder，例如：
    
    - [App/main.py (line 34934)](D:/Provenance Project/ImagingLibWatch/App/main.py:34934)
    - [workflow_manager.py (line 1618)](D:/Provenance Project/ImagingLibWatch/core/workflow_manager.py:1618)
    - [workflow_manager.py (line 1852)](D:/Provenance Project/ImagingLibWatch/core/workflow_manager.py:1852)
    - [workflow_manager.py (line 3546)](D:/Provenance Project/ImagingLibWatch/core/workflow_manager.py:3546)
    - [authentication runtime (line 754)](D:/Provenance Project/ImagingLibWatch/core/authentication/integration/runtime.py:754)
5. `analysis_results_v2` 沒有正式保存 `internalnum1/internalnum2`、asset role、algorithm version 與完整 source asset list；目前主要靠來源 asset 或 JSON 間接回查。[db_manager.py (line 97)](D:/Provenance Project/ImagingLibWatch/DB/db_manager.py:97)
    
6. uploader 使用 `upload_file()`，相同 key 已存在時會直接覆寫。上傳後才驗 SHA-256，無法發現「先前不同內容已被覆蓋」的事故。[outbox_dispatcher.py (line 101)](D:/Provenance Project/ImagingLibWatch/data_manager/outbox_dispatcher.py:101)
    
7. catalog record type 仍有根據 key 是否包含 `report` 或 `mask` 的字串推斷，改 layout 後容易誤分類；應以正式 `record_type/result_type` 為準。[outbox_dispatcher.py (line 185)](D:/Provenance Project/ImagingLibWatch/data_manager/outbox_dispatcher.py:185)
    
8. 固定 `manifest/run.json` 若在執行期間反覆更新，會造成覆寫、S3 event 重複與 reader 看到半完成狀態。因此建議：
    
    - 執行中只寫 immutable revision/event。
    - 所有 artifacts 上傳並驗證後才寫 `run.final.json`。
    - 最後才建立 `_SUCCESS`。
    - `run.final.json` 和 `_SUCCESS` 都必須使用「不存在才建立」語意。
9. 加密產生的 `.meta` sidecar 目前會直接上傳成 `<s3_key>.meta`，但建議結構沒有定義這種 artifact。應正式定義為 encryption sidecar，或改採 S3 SSE-KMS，避免 lineage/lifecycle 漏管。
    
10. `view_name`、`point_name` 等 segment 不能採用會造成碰撞的靜默清洗。例如 `A/B` 和 `A\B` 都可能變成 `A_B`。穩定 ID 應嚴格驗證，不合法就報錯；display label 留在 metadata。
    

---

## 下一步改碼建議

目前不應先改 UI 或大量替換 f-string。第一批修正應是：

1. 建立獨立 `S3KeyBuilder` 與 `ArtifactMetadata` contract。
    
2. 明確支援：
    

```
legacy_flat
device_local_mirror
structured_v3
operational_v1
```

3. DB migration 新增：

```
key_schema_version
station_id
internalnum1/internalnum2（analysis results）
asset_role/asset_index（analysis lineage）
source_asset_id/source_asset_ids
algorithm_version
quality_flags
object_version_id/etag
```

4. 先讓新 builder 只產生 key、加完整 unit tests，不立即切換 production writer。
    
5. 再逐一移除 App、workflow、authentication 的 direct f-string。
    
6. Outbox 加 immutable/idempotent upload：
    
    - key 不存在：上傳。
    - key 已存在且 SHA-256 相同：視為成功重試。
    - key 已存在但 SHA-256 不同：拒絕覆寫並進 dead letter。
7. 最後才啟用 `operational_v1` 新寫入，reader 保持 legacy 相容。
    

這次只完成架構與 code audit，尚未修改檔案；worktree 目前乾淨。修正上述設計問題後，這個 `operational/v1` 就可以作為下一階段改碼基準。




#### 但不同的站點為什麼要分site_id跟station_id
```
但不同的站點為什麼要分site_id跟station_id? 因為一個site不會有兩個station. 然後raw/下面是用<view_key>/<point_key>/<capture_id>/, 跟local比較有多加<point_key>/<capture_id>/是單純容易定位容易搜尋考量嗎? 還是有其他理由?(因為如果是存在local就全部放在raw folder下面) 然後這是甚麼請解釋: └── <asset_role>/<asset_id>.<ext>儲存. 那這句話是甚麼? 如果未來確定採用「每次 run 發放限定 prefix 的 STS credentials」，才可以省略 `stations/<station_id>`。目前 repository 看不到完整的 run-scoped STS 機制，所以預設加入 station 比較安全。
```

如果一個 `site` 永遠只對應一個 station，我同意：S3 key 不需要同時放 `site_id` 和 `station_id`。

建議改成：

```
operational/v1/
└── sites/<site_id>/
    └── watches/<watch_id>/
        └── runs/<run_id>/
            ├── manifest/
            ├── raw/
            ├── analysis/
            ├── reports/
            └── authentication/
```

`station_id/device_id` 可以保留在 DB、manifest 和 object metadata，用來記錄「實際是哪台電腦／設備執行」，但不必成為 S3 路徑的一部分。

## site_id 和 station_id 原本的差別

一般定義是：

- `site_id`：業務或部署位置，例如 `taipei-lab-01`。
- `station_id`：該位置中的某一套實體設備，例如 `nitro-pc-01`。

即使同一時間只有一台 station，兩者仍可能因換機而不同：

```
site_id = taipei-lab-01       # 部署位置不變

2026 舊電腦：
station_id = imaging-pc-01

2027 換電腦：
station_id = imaging-pc-02
```

但這只是 audit metadata 的價值，不代表兩者都必須進 S3 key。

你目前的明確規則是：

```
一個 site 不會有兩個 station
```

因此可把 `site_id` 當成：

- IAM 權限邊界。
- S3 namespace。
- 部署單位。

而把 `station_id` 當成：

- 設備稽核資訊。
- 硬體更換追蹤。
- 問題排查資訊。

另外，code 中目前 `get_site_id()` 會在沒有 `site_id` 時拿 `station_id` 或 `DeviceID` 代替，這仍建議修掉，因為會混淆兩種欄位的語意。[local_storage.py (line 346)](D:/Provenance Project/ImagingLibWatch/data_manager/local_storage.py:346)

---

## 為什麼 S3 raw 下多了 view/point/capture

原本本機是：

```
Local_Data/<watch_id>/Raw/<uuid>.jpg
```

建議 S3 是：

```
raw/<view_key>/<point_key>/<capture_id>/<asset_role>/<asset_id>.<ext>
```

這些層級不是為了跨 watch「搜尋」。跨 watch、template、internalnum 的搜尋仍然應該走 DynamoDB/Athena。

主要有三個理由。

### 1. 人工排錯和災難復原

看到：

```
raw/front/macropoint22/hdl_1/raw_bracket/abc.jpg
```

可以立即知道它屬於：

```
Front
→ macropoint22
→ HDR capture hdl_1
→ 一張 bracket 原圖
```

如果只有：

```
raw/abc.jpg
```

就一定要先查 DB 才知道它是什麼。

當 DB index、manifest 或同步程序發生問題時，路徑本身保留一點最低限度的操作資訊，會比較容易救援。

### 2. 可以按 capture 做批次處理

例如只下載某個 capture：

```
raw/front/macropoint22/hdl_1/
```

可以一次取得：

```
所有 brackets
HDR fused image
fallback image
```

也能對這個 prefix 做檢查、重新處理或重建 manifest。

### 3. 清楚表達「一個 point 有多個 capture，每個 capture 有多個 assets」

目前重構目標正是：

```
point
└── capture
    └── assets[]
```

S3 路徑直接反映這個關係，對人工檢查很直觀。

但必須強調：

> `view/point/capture` 對唯一性和資料正確性不是必要的，它們主要提供 operational grouping。

真正保證不衝突的是 `asset_id`；真正的資料關聯仍以 DB 和 manifest 為準。

如果希望 key 更精簡，也可以改成：

```
raw/<capture_instance_id>/<asset_role>/<asset_id>.<ext>
```

例如：

```
raw/cap_86a29f731c21/raw_bracket/abc123.jpg
raw/cap_86a29f731c21/hdr_fused/def456.tiff
```

這種做法的優缺點是：

|設計|優點|缺點|
|---|---|---|
|`view/point/capture`|人工看得懂，方便依邏輯 capture 批次操作|路徑較長；名稱必須穩定|
|`capture_instance_id`|簡單、穩定、不受改名影響|必須查 manifest/DB 才知道是哪個 point|
|全部平鋪 `raw/<asset_id>`|最簡單|人工維運與災難復原最困難|

就目前系統而言，我仍偏向保留：

```
raw/<view_key>/<point_key>/<capture_id>/
```

但前提是這三者使用穩定的 canonical key，不是可任意修改的 UI 顯示名稱。

---

## `<asset_role>/<asset_id>.<ext>` 是什麼

S3 本身其實沒有真正的資料夾。以下整串都是 object key：

```
raw/front/macropoint22/hdl_1/raw_bracket/abc123.jpg
```

S3 Console 只是把 `/` 顯示成資料夾。

其中：

### `asset_role`

表示這個檔案在一次 capture 中扮演什麼角色。

建議使用固定 enum，例如：

```
raw_single       一般 Standard image
raw_bracket      HDR 的其中一張曝光原圖
hdr_fused        成功融合的 HDR 圖
hdr_fallback     HDR 融合失敗後選用的替代圖
```

例如 HDR capture：

```
raw/front/macropoint22/hdl_1/
├── raw_bracket/
│   ├── 111aaa.jpg       # -2 EV
│   ├── 222bbb.jpg       #  0 EV
│   └── 333ccc.jpg       # +2 EV
└── hdr_fused/
    └── 444ddd.tiff
```

`asset_role` 的用途是區分檔案語意，不是用來保證唯一。

### `asset_id`

`asset_id` 是每個 image asset 的唯一 ID，通常是 UUID：

```
444ddd...
```

不能用原始檔名，因為可能重複：

```
image.jpg
image.jpg
image.jpg
```

改用 asset ID 後：

```
7b66c5....jpg
82ca11....jpg
f17a82....jpg
```

重試、不同 station 或不同 watch 之間不容易互相覆蓋。

### `<ext>`

是真正的檔案格式：

```
.jpg
.png
.tiff
.enc
```

因此：

```
hdr_fused/444ddd.tiff
```

意思是：

> 這是一個角色為 `hdr_fused`、asset ID 為 `444ddd`、檔案格式為 TIFF 的 image asset。

曝光順序、EV、camera、internalnum 等不需要塞進檔名，應保存在 metadata：

```
{
  "asset_id": "111aaa",
  "asset_role": "raw_bracket",
  "asset_index": 0,
  "exposure_ev": -2,
  "internalnum1": "0022",
  "internalnum2": "0002"
}
```

還有一個命名問題：`hdr_fused` 嚴格來說不是真正的 raw sensor image。若希望語意非常嚴格，可以把 `raw/` 政名為：

```
captures/
```

變成：

```
captures/<view>/<point>/<capture>/<asset_role>/<asset_id>.<ext>
```

這樣 Standard、brackets、fused、fallback 都可合理視為「capture 階段產生的 assets」。我認為 `captures/` 在語意上其實比 `raw/` 更準確。

---

## 「run-scoped STS credentials」是什麼

STS 是 AWS Security Token Service。它可以發放一組短效、權限受限的臨時 AWS credentials。

一般 station credentials 可能可以寫整個 site：

```
operational/v1/sites/taipei-lab/*
```

run-scoped STS credentials 則只允許寫這一次 run：

```
operational/v1/sites/taipei-lab/
watches/W-16610-00042/runs/Run_20261006_143012_a1b2c3/*
```

也就是說，即使程式發生 bug，拿著這組 credentials 也不能寫到：

```
另一個 watch
另一個 run
另一個 site
```

完整概念是：

```
1. Station 請求開始一個 run
2. 後端建立 run_id
3. 後端透過 STS 發放臨時 credentials
4. Credentials 只允許寫該 run prefix
5. Run 結束或 credentials 過期後，不能再上傳
```

這是比較強的「最小權限」設計，但需要：

- 一個可信任的 credential broker／後端。
- STS AssumeRole 流程。
- 每個 run 動態產生 IAM session policy。
- credentials 更新與過期恢復邏輯。
- 離線拍攝後恢復同步的重新授權機制。

目前 repository 的上傳方式不是這種模式，而是由現有 AWS client 直接 `upload_file()`；因此我才說目前看不到完整的 run-scoped STS 機制。

不過在你補充「一個 site 永遠只有一個 station」之後，前一句需要修正：

> 省略 `stations/<station_id>` 不需要以 run-scoped STS 為前提。

你們目前可以直接採用 site-scoped IAM：

```
Station Taipei credentials
只允許：
operational/v1/sites/taipei-lab/*
```

run-scoped STS 是額外加強措施，不是省略 station 層級的必要條件。

## 更新後的建議

依你的實際部署模型，我建議 canonical layout 改為：

```
operational/v1/
└── sites/<site_id>/
    └── watches/<watch_id>/
        └── runs/<run_id>/
            ├── manifest/
            ├── captures/
            │   └── <view_key>/<point_key>/<capture_id>/
            │       └── <asset_role>/<asset_id>.<ext>
            ├── analysis/
            ├── reports/
            └── authentication/
```

並採取：

- `site_id`：S3 namespace 與 IAM 邊界。
- `station_id/device_id`：只放 DB、manifest、object metadata。
- `captures/`：比 `raw/` 更準確涵蓋 Standard、HDR brackets、fused 與 fallback。
- `view/point/capture`：用於 operational grouping，不作為搜尋資料庫。
- `asset_role`：固定 enum，描述檔案角色。
- `asset_id`：唯一 object 身分，避免覆寫。




#### 修改code新AWS S3建議結構
```
了解請修改code把S3的 架構改成這個, 那是否local DB, AWS DB也需要做相對應的修改?
```
已完成程式修改，新資料會寫入：

```
operational/v1/sites/<site_id>/watches/<watch_id>/runs/<run_id>/
├── manifest/run.final.json
├── captures/<view>/<point>/<capture_id>/<asset_role>/<asset_id>.<ext>
├── analysis/<experiment_id>/<algorithm>/<result_type>/<result_id>.<ext>
├── reports/<artifact_type>/<artifact_id>.<ext>
└── authentication/{features|results}/<id>.json
```

其中 `station_id` 不放進 key，只保存在 DB/metadata；本機既有 `Raw/`、`Analysis/` 目錄也不搬動。

是的，DB 必須同步修改，這次已包含：

- Local SQLite：加入 `site_id`、`station_id`、`key_schema_version`、internal numbers、asset role、source asset lineage、checksum 等欄位與索引，啟動時自動 migration。
- DynamoDB：新 item 寫入完整 provenance，以及：
    - `template_internalnum_pk/sk`
    - `internalnum_pk/sk`
- Iceberg/Athena：ETL 已輸出 internal numbers、asset role、schema version、algorithm version 等欄位。
- 舊資料：不搬 S3、不改舊 `s3_key`；仍直接依 DB 儲存的完整 key 讀取。
- HDR fallback：現在使用獨立的 `hdr_fallback`，不再冒充 `hdr_fused`。

主要修改位於 [s3_layout.py](D:/Provenance Project/ImagingLibWatch/data_manager/s3_layout.py)、[local_storage.py](D:/Provenance Project/ImagingLibWatch/data_manager/local_storage.py)、[db_manager.py](D:/Provenance Project/ImagingLibWatch/DB/db_manager.py)、[cloud_db.py](D:/Provenance Project/ImagingLibWatch/data_manager/cloud_db.py)、[lake_etl.py](D:/Provenance Project/ImagingLibWatch/data_manager/lake_etl.py)、[workflow_manager.py](D:/Provenance Project/ImagingLibWatch/core/workflow_manager.py) 與 [App/main.py](D:/Provenance Project/ImagingLibWatch/App/main.py)。

驗證結果：

- 核心 S3、DB、HDR、outbox、authentication、provenance、Lake ETL：92 項通過。
- task UI：20 項通過。
- continue-watch 相容性：9 項通過。
- Python compile、YAML 解析、`git diff --check` 均通過。

AWS 上線前仍需人工部署：

1. 執行 [migrate_analysis_facts_operational_v1.sql](D:/Provenance Project/ImagingLibWatch/DB/DB_test/migrate_analysis_facts_operational_v1.sql)。
2. 在 DynamoDB 實際建立上述兩組 GSI；程式目前只會寫入 GSI attributes。
3. 更新 IAM、Lifecycle 與 event notification prefix。
4. 做實機驗證：多張 standard、HDR fused/fallback、斷網恢復、舊 key 讀取、PDF/manifest、authentication lineage。
5. `_SUCCESS` key builder 已支援，但目前沒有產生 completion marker；必須等所有 outbox uploads 完成後才能安全加入。

另外，全量 `App.test_template_create_rules` 目前仍有至少 6 項位於本次 diff 外的 fixture/config 失敗，且一項測試 worker 會掛住，因此不能宣稱整個 repository 全綠；這些錯誤不是此次 S3/DB 路徑修改所觸發的程式區段。





```
那如果我現在有一些在LA downtown執行的watchentry, 然後我把folder copy到我的laptop的這個folder(D:\Provenance Project[ Trivecto ]@21. GCIdata\ImagingLibWatch_debugbezel\Rolex_16610 T_1788559897), 那我可以把這個folder上傳到AWS S3用你建議的新的架構儲存, 以及database也有紀錄, 就像真正在App main.py執行一樣. 這是否做得到? 因為很多以前掃描的data要用新的架構重新上傳到AWS S3, 這樣AWS S3才可以統一架構也不會需要省略舊的data. 請評估不要修改code
```
可以做到，而且這個資料夾的完整度相當高；但不能只是把資料夾複製進 `Local_Data` 後啟動 `main.py`。目前 repository 沒有一個能把舊 WatchEntry 完整轉換成新 S3 layout、補建 SQLite 關聯、再同步 DynamoDB 的正式匯入工具，需要下一步製作專用的 legacy migration/importer。

本次只做讀取檢查，沒有修改程式、DB 或 AWS。

## 這個資料夾的實際狀況

我檢查了[這個 WatchEntry 資料夾](D:/Provenance Project/[ Trivecto ]/@21. GCIdata/ImagingLibWatch_debugbezel/Rolex_16610 T_1788559897)：

|內容|數量|可遷移性|
|---|---|---|
|Raw 圖片|65|65 張全部能配對 report|
|CameraPipelineReports|65|每份都有必要的 capture metadata|
|Analysis|53|14 JSON、29 PNG、10 JPG|
|Analysis 對應來源|16 個 Raw UUID|全部能找到來源圖片|
|SQLite DB／watchentry JSON／manifest|0|無法百分之百重建原始 DB 狀態|

65 份 camera report 全部包含：

- `asset_id`
- `view_name`
- `point_name`
- `capture_id`
- `internalnum1/internalnum2`
- `asset_role`
- `run_id`
- `template_id`
- `created_at`
- 舊 `raw_image_id`
- 舊 `s3_key`

資產角色也很清楚：

- `raw_single`: 50 張
- `hdr_fused`: 15 張

因此 Raw 圖片本身可以高可信度地遷移，不需要根據檔名猜測 point 或 capture。

## 實際轉換範例

例如目前的圖片：

```
Raw/ddc6af07e09243d6b11b419d851ee07e.png
```

對應 report 顯示：

```
watchid: Rolex_16610 T_1788559897
template_id: Rolex_16610 T
run_id: Exp_20260904_151137_eecd35fd
view_name: Front
point_name: macropoint1
capture_id: std_1
internalnum1: 0004
internalnum2: 0001
asset_role: raw_single
```

若確認歷史站點 ID 為 `la-downtown`，新 S3 key 會是：

```
operational/v1/sites/la-downtown/
  watches/Rolex_16610_T_1788559897/
  runs/Exp_20260904_151137_eecd35fd/
  captures/front/macropoint1/std_1/raw_single/
  ddc6af07e09243d6b11b419d851ee07e.png
```

HDR 也能正確分類。例如 `micropoint6/hdr_1` 的 report 明確記錄：

```
asset_id: adde50ef5b98424d8657939c95ae6935
asset_role: hdr_fused
internalnum1: 0016
internalnum2: 0002
```

所以可存成：

```
.../captures/front/micropoint6/hdr_1/hdr_fused/
adde50ef5b98424d8657939c95ae6935.png
```

這批舊資料只有 fused HDR 成品，沒有 bracket 原圖，因此不能虛構 `raw_bracket` 資產。只登記實際存在的 `hdr_fused` 是正確做法。

## 能否像 App 執行一樣記錄 DB？

儲存結果可以做到幾乎相同，包括：

- SQLite `watch_registry`
- `watch_runs`
- `experiments`
- `point_instances`
- `capture_instances`
- `image_assets`
- `analysis_results_v2`
- `artifact_records`
- 相容舊 reader 的 `raw_images`、`experiment_results`
- `sync_outbox`
- S3 objects
- DynamoDB catalog records
- 最終 `run.final.json`

但「歷史事實完全等同當時 App 現場執行」做不到，因為這個資料夾沒有原始 SQLite DB，也沒有：

- 原始 `site_id/station_id`
- operator/user
- 完整 WatchEntry JSON
- 明確 template version
- 原始 point/capture instance UUID
- run 最終完成狀態
- authentication records
- HDR bracket 原始影像
- 部分 analysis 的 algorithm version 與精確產生時間

缺少的欄位不能由現在的 laptop 設定代填，否則會製造錯誤 provenance。

正確做法是：

```
site_id = la-downtown                 # 歷史拍攝地點
station_id = 歷史 station，若未知則 null/unknown
ingested_by_station_id = 目前 laptop # 本次搬運者，不能冒充拍攝 station
migration_origin = legacy_folder
original_run_status = unknown
```

目前 laptop 的 `site_id` 絕對不能直接套用到這批資料。

## DB schema 是否需要再修改？

若使用目前已更新的 `operational_v1` schema，SQLite 表結構本身大致不需要再改；需要的是匯入資料與建立關聯。

但是不能直接使用現有的 `capture_and_register_asset()`，因為它會：

- 產生新的 asset UUID
- 重新安排本機檔案
- 可能破壞舊 analysis 檔名所指向的 source UUID

相關行為可見 [local_storage.py (line 764)](D:/Provenance Project/ImagingLibWatch/data_manager/local_storage.py:764)。

legacy importer 應保留原始 `asset_id`，並直接透過 DB registration/outbox 流程登記。舊 report 中的數字 `raw_image_id` 不應強制寫入現在 SQLite 的自增主鍵；應保存為：

```
legacy_raw_image_id
original_s3_key
original_local_path
```

AWS 方面：

- S3 不需要 schema migration。
- DynamoDB 應透過 outbox/catalog sync 建立新紀錄，不建議另外手寫兩套同步流程。
- DynamoDB GSI 與 Iceberg/Athena schema migration 必須先部署，跨 watch/template/internalnum 查詢才會完整。
- Raw assets 目前主要進 DynamoDB catalog；analysis facts 才進 lake/Athena 流程。

## 建議的安全匯入流程

1. `--dry-run` 掃描整個舊資料夾。
2. 驗證每張 Raw 都有唯一 report；本案例已確認為 65/65。
3. 計算每個檔案的 SHA-256。
4. 由操作者指定歷史 `site_id`，不能使用目前 laptop 預設值。
5. 保留原本 65 個 `asset_id`。
6. 對缺失的 run/point/capture/result ID 使用「可重複產生」的 deterministic ID。
7. 先檢查 SQLite、DynamoDB 與 S3 是否已存在：
    - key 與 hash 相同：跳過。
    - key 相同但 hash 不同：停止，不能覆寫。
8. 寫入 SQLite 與 `sync_outbox`，讓既有同步機制上傳 S3、更新 DynamoDB。
9. CameraPipelineReports 登記到 `reports/camera_pipeline/`。
10. 53 個 analysis 檔逐一登記，並保留 source asset lineage。
11. 所有 object 與 DB record 驗證完成後，最後才產生 `run.final.json` 和 migration `_SUCCESS`。
12. 產生匯入報告，列出 uploaded/skipped/conflict/missing 數量。

## 批次遷移前還要處理的風險

目前 [S3KeyBuilder (line 132)](D:/Provenance Project/ImagingLibWatch/data_manager/s3_layout.py:132) 會清理 key segment，例如空白會轉成底線。因此：

```
Rolex_16610 T_1788559897
Rolex_16610_T_1788559897
```

可能得到相同的 S3 watch segment。這個單一資料夾沒有顯示衝突，但大量歷史資料匯入前必須先做全域 collision scan；同樣也要檢查只差大小寫或特殊符號的 view、point、capture ID。

另外，舊 S3 key 不應立即刪除。建議：

- 新 canonical key 上傳完成。
- DB 保存 `original_s3_key`。
- 驗證 gallery、report、CLI、provenance query 都能使用新 key。
- 最後才決定保留、封存或 lifecycle 清理舊 objects。

結論是：這批資料可以高品質地匯入新 S3 架構並建立新的 SQLite/DynamoDB 紀錄；但應標示為「由 legacy folder 重建」，不能偽裝成原始 App live run。下一步適合實作一個具備 `dry-run → apply → verify` 三階段、可重複執行且不覆寫衝突資料的 legacy WatchEntry importer。