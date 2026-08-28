

| [[#### 整理這些檔案的data or index等有哪些會存入local DB方便我們管理所有的files跟後續的分析]]<br> |     |
| -------------------------------------------------------------------- | --- |
| [[#### 整理all files 連結 Database]]                                     |     |
| [[#### database支援上萬watch?]]                                          |     |
| [[#### 「邊緣到雲端 (Edge-to-Cloud)」與「OLTP / OLAP 分離」架構]]                  |     |
| [[#### 舉例解釋SQLite vs. DynamoDB的差別]]                                  |     |
| [[#### 文詳細解釋 DB/watchDB.db的內部資料結構]]                                  |     |
| [[#### DB/watchDB.db是如何跟AWS同步]]                                      |     |
| [[#### DB browser for SQLite打開watchDB.db]]                           |     |
| [[#### Claude分析新DB架構]]                                               |     |
| [[#### 給Codex具體修改計畫]]                                                |     |
| [[#### 簡單解釋Sync Outbox跟Parquet / Iceberg Metric Facts]]              |     |
| [[#### 詳細解釋Sync Outbox跟Parquet / Iceberg Metric Facts]]              |     |
| [[#### 開通那些AWS服務]]                                                   |     |
|                                                                      |     |
|                                                                      |     |

#### 整理這些檔案的data or index等有哪些會存入local DB方便我們管理所有的files跟後續的分析

```
**

之前有整理了這個App main.py如果有儲存images or 其他files有包括1. 正式拍攝影像, 2. 分析結果, 3. Camera pipeline TXT 報告, 4. Template 建立影像, 5. WatchShift 參考影像, 6. Material／XRF 檔案, 7. PDF 與報告檔, 8. SQLite 資料庫, 9. Log 與稽核檔案, 10. 其他零散輸出與快取, 及yaml files.

請幫我整理這些檔案的data or index等有哪些會存入local DB方便我們管理所有的files跟後續的分析譬如:

尋找某個uuid的image file是屬於哪個watchid, 是屬於哪個template, 是甚麼時候拍的, 是哪個watchpoint, 以及跟這個uuid相關的analysis結果是存在哪裡哪個experiment? 跟這個uuid相關的analysis結果的某個結果數值value? 跟這個uuid相關的pdf report的identifier是哪一個? 有哪個另外的uuid跟這個同屬於同一次拍照, 有哪個另外的uuid跟這個同屬於同個watchpoint?, 有哪個另外的uuid跟這個同屬於同個reference number? 跟這個uuid相關的watchpoint不同capture的uuid是? 找到所有屬於同個reference number的所有這個watchpoint的uuid, 找到所有屬於同個watchpoint的所有的uuid, 找到找到所有屬於同個reference number的所有這個watchpoint且authentication是真品的uuid

Analysis\Exp_YYYYMMDD_HHMMSS_<8字元ID>有包含哪些analysis results, analysis結果的某個結果數值value? 跟這個uuid相關的pdf report的某個analysis的identifier是哪一個? 這個experiment是從哪一次watchid來的? 是從哪個template? 這個watchid還有沒有其他experiments, 找到所有屬於同個reference number的所有這個watchpoint的uuid, 找到所有屬於同個analysis result的所有的uuid, 找到找到所有屬於同個reference number的所有這個analysis result且authentication是真品的values, 要對這個experiment裡面某個analysis找到所有相關values的distribution.

要怎麼找到Camera pipeline TXT 報告是屬於哪次拍照, 是屬於哪個template建立的, 其中的某個數值譬如拍照的exposure

尋找某個template是屬於哪個reference number, 以及用在哪個watchid, 是甚麼時候建立的, 跟哪些watchpoint有關, 以及跟這個template相關的analysis結果是存在哪裡哪個experiment? 跟這個template相關的analysis結果的某個結果數值value? 有哪個另外的template跟這個同屬於同個reference number? 找到所有屬於同個template的所有這個watch scan

是否有儲存WatchShift 參考影像相關index在database裡面

關於material data是屬於哪一次scan? 是屬於哪個reference number? 跟同個reference number的material values是否能比較不同的成分組成%? 跟同個reference number的material values是否能比較分析特定成分%?

能否找到pdf file是從哪次watch scan? 這個pdf file是屬於哪種reference number? pdf file裡面的某個數值是甚麼? pdf file裡面某數值跟另外一份pdf file數值的比較

local DB能否查詢裡面有哪些種類跟內容? 譬如有哪些template? 有哪些watch scan?

以及還有甚麼是local DB有儲存的

**
```

## 結論

我檢查了實際的 [watchDB.db](/D:/Provenance Laboratories projects/ImagingLibWatch/DB/watchDB.db)、DB schema、App 拍攝流程、analysis、WatchShift、material、PDF 與 template 程式碼。

目前 Local DB 已經有不錯的拍攝階層：

```
watch_runs
  → point_instances
    → capture_instances
      → image_assets
```

所以「影像屬於哪個 WatchID、哪個 template、哪次 run、哪個 watchpoint、哪個 capture」大致可查。

但以下關聯目前尚未完整資料庫化：

- image UUID → analysis result
- analysis result → experiment
- analysis value 的結構化查詢與 distribution
- image／analysis → PDF report identifier
- material/XRF 的 scan 與 element values
- Camera Pipeline TXT 作為獨立 file asset
- Template 建立影像
- WatchShift reference image 的完整 file index
- log、YAML、cache、零散輸出的通用 file registry
- reference number／authentication 的可靠索引

因此答案是：目前可以做「部分追溯」，但還不能可靠管理所有 files 與後續跨 scan 分析。

## 實際 Local DB 內容

目前 DB 約 359 MB，資料量如下：

|Table|筆數|用途|
|---|---|---|
|`watch_runs`|1,371|每次 watch scan／run、WatchID、template|
|`point_instances`|4,861|該 run 執行過的 watchpoint|
|`capture_instances`|3,471|watchpoint 下的 standard/HDR capture|
|`image_assets`|3,471|新版影像 asset|
|`raw_images`|7,182|舊版影像紀錄／S3 sync queue|
|`experiment_results`|9,813|舊版 analysis、PDF、Final Report|
|`analysis_results_v2`|0|新版 image→analysis 關聯表，目前沒有實際資料|
|`ui_asset_registry`|6,215|App UI 顯示用的最新 image/result path|
|`local_templates`|21|Template 完整 JSON|
|`watch_metadata`|113|WatchID、Brand、Model、Reference、完整 WatchEntry JSON|
|`watch_registry`|0|目前未實際使用|
|`users`|5|Local login cache|

Schema 定義在 [db_manager.py (line 54)](/D:/Provenance Laboratories projects/ImagingLibWatch/DB/db_manager.py:54)。

另外有幾個重要資料品質現況：

- 1,371 個 `watch_runs` 全部仍是 `running`。
- 3,471 個 `capture_instances` 全部仍是 `processing`。
- 3,471 個 capture 每個都只有一筆 `image_assets`。
- `image_assets` 包含：
    - `raw_single`：3,237
    - `hdr_fused`：234
    - `raw_bracket`：0
- 目前實際上還沒有做到一個 capture 對多張 bracket image。
- 140 個有 run 的 WatchID 中，只有 110 個能 join 到 `watch_metadata`。
- 只有 7 個 run WatchID 有非空的 Reference。
- 只有 405／1,371 個 run 能精確對上相同 TemplateID＋version。

App 拍攝流程目前只註冊一個 `raw_single` 或 `hdr_fused`，而且固定 `asset_index=0`，見 [main.py (line 27242)](/D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:27242)。

## 十類檔案目前是否有 DB index

|檔案類型|目前 DB 狀態|結論|
|---|---|---|
|1. 正式拍攝影像|`image_assets`＋`raw_images`|有，但存在雙寫與兩種 ID|
|2. Analysis 結果|`experiment_results`|有 path/task/data，但 image 關聯多數缺失|
|3. Camera Pipeline TXT|path 藏在 `image_assets.metadata`|部分有，TXT 本身不是 asset row|
|4. Template 建立影像|`TemplateScratch` filesystem|沒有正式 DB index|
|5. WatchShift reference|Template JSON 的 flags＋manifest|部分有，沒有獨立 asset table|
|6. Material／XRF|WatchEntry JSON／`material_records.json`|沒有 dedicated DB table|
|7. PDF／報告|`experiment_results` 的 `Auto_PDF_Report`|有 PDF path，但缺 report lineage|
|8. SQLite DB|`DB/watchDB.db`|本體|
|9. Log／稽核|`Local_Data/audit_logs/*.jsonl`|不在 DB|
|10. 其他輸出／快取|filesystem|沒有 generic file catalog|
|YAML|部分 analysis API 支援，template process YAML 不入 DB|實際 DB 目前沒有 `data_yaml` rows|

Audit log 是獨立 JSONL hash chain，不寫 SQLite，見 [audit_logger.py (line 98)](/D:/Provenance Laboratories projects/ImagingLibWatch/logging_system/audit_logger.py:98)。

## Image UUID 的實際問題

正式 image filename 使用 32 字元 UUID，例如：

```
1b71bfcc4f604c51b35d081fb2c0e4d9.png
```

這個 UUID 由 [local_storage.py (line 263)](/D:/Provenance Laboratories projects/ImagingLibWatch/data_manager/local_storage.py:263) 產生，並放在：

```
image_assets.metadata["asset_id"]
```

但 `image_assets.asset_id` 本身卻是另一個 ID：

```
ast_504d394de793
```

所以現在同一張 image 有至少三種識別方式：

- file UUID：`1b71bf...`
- DB asset ID：`ast_504d...`
- legacy `raw_images.id`：SQLite integer

此外，同一張 image 會先寫一次 `raw_images`，註冊 `image_assets` 時又雙寫一次 `raw_images`。目前：

- `raw_images`：7,182 rows
- 實際不同 path：3,711
- 有 3,471 個 path 各出現兩次

雙寫邏輯位於 [db_manager.py (line 144)](/D:/Provenance Laboratories projects/ImagingLibWatch/DB/db_manager.py:144)。

### 目前由 UUID 查 image lineage

可以使用：

```
WITH target AS (
    SELECT
        a.*,
        LOWER(json_extract(a.metadata, '$.asset_id')) AS file_uuid
    FROM image_assets a
    WHERE LOWER(json_extract(a.metadata, '$.asset_id')) = LOWER(:uuid)
)
SELECT
    t.file_uuid,
    t.asset_id,
    t.local_path,
    t.s3_key,
    t.watchid,
    t.run_id,
    r.template_id,
    r.template_version,
    datetime(t.created_at, 'unixepoch', 'localtime') AS captured_at,
    p.view_name,
    p.point_name,
    p.internalnum1,
    c.capture_id,
    c.internalnum2,
    c.capture_type,
    m.Reference
FROM target t
LEFT JOIN capture_instances c
       ON c.capture_instance_id = t.capture_instance_id
LEFT JOIN point_instances p
       ON p.point_instance_id = c.point_instance_id
LEFT JOIN watch_runs r
       ON r.run_id = t.run_id
LEFT JOIN watch_metadata m
       ON m.WatchID = t.watchid;
```

這可以回答：

- 哪個 WatchID
- 哪個 TemplateID/version
- 拍攝時間
- view／watchpoint
- standard 或 HDR
- capture ID
- Reference，如果 `watch_metadata.Reference` 有值

## 同一次拍照、同一 watchpoint、同 Reference 的 UUID

### 同一次 capture

結構上可以：

```
SELECT peer.*
FROM image_assets target
JOIN image_assets peer
  ON peer.capture_instance_id = target.capture_instance_id
WHERE LOWER(json_extract(target.metadata, '$.asset_id')) = LOWER(:uuid);
```

但目前每個 capture 都只有一個 `image_assets`，因此通常找不到「另外的 UUID」。

未來 HDR 應該是：

```
capture_instance_id
  ├─ raw_bracket index 0
  ├─ raw_bracket index 1
  ├─ raw_bracket index 2
  └─ hdr_fused
```

### 同一個 watchpoint

建議優先用：

```
watchid + internalnum1
```

次要 fallback 才是：

```
watchid + view_name + point_name
```

但目前只有 81／3,471 個 image asset 有非空 `internalnum1`，所以大多只能依靠 view/point 名稱。

### 同 Reference 的同一 watchpoint

可以 join：

```
image_assets.watchid
→ watch_metadata.WatchID
→ watch_metadata.Reference
```

但目前 Reference coverage 很低，而且存在 `16613T`、`16613 T` 等格式差異，因此正式查詢前應增加 `reference_normalized`。

### Authentication＝真品

Authentication 現在藏在：

```
watch_metadata.Full_JSON
  → watchauthenticity.EntireWatch
  → watchauthenticity.<component>
```

但實際 113 筆 watch metadata 中，沒有任何非空 `EntireWatch`。所以目前不能可靠執行：

> 同 Reference＋同 watchpoint＋authentication 為真品的 UUID

Authentication 結構定義在 [watchauthenticity_structure.py (line 8)](/D:/Provenance Laboratories projects/ImagingLibWatch/DB/templates/watchauthenticity_structure.py:8)，但需要真正存入 DB 並改成標準 enum，例如：

```
AUTHENTIC
NOT_AUTHENTIC
INCONCLUSIVE
NOT_EVALUATED
```

而不是自由文字 `"Authentic"`、`"Full authenticity confirmed"` 等。

## Image UUID 與 Analysis 的關係

這是目前最大的斷點。

`experiment_results` 有 9,813 rows，但：

- `raw_image_id` 非 NULL：425
- `raw_image_id` 為 NULL：9,388
- 425 筆只對應 31 個 raw image ID
- 新版 `analysis_results_v2`：0 rows

App 主分析流程明確把 `raw_image_id=None` 寫入，見 [main.py (line 22571)](/D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:22571)。

雖然 3,409 個 analysis rows 的 JSON blob 中可搜尋到 raw UUID，但那是文字推導，不是可索引 foreign key。

因此：

|問題|目前能力|
|---|---|
|UUID 相關的 analysis 在哪個 experiment|部分可從 JSON/path 推導，不可靠|
|UUID 相關的某個 analysis value|JSON 有值時可抽，但沒有可靠 image FK|
|所有屬於同 analysis result 的 UUID|無法可靠完成|
|對某 analysis value 做 distribution|可以離線解析，但 DB SQL 會重複計數|
|image → analysis_results_v2|理論上可以，實際 table 為空|

另一個問題是 analysis report data 的放置方式：

- 大部分 `data_json` row 的 `data` 是 `{}`。
- 完整 analysis JSON 反而重複存放在數個 `image_mask` rows。
- 相同 payload 會因為多個輸出圖而重複。

因此直接對 `experiment_results.data` 做 distribution 會產生重複樣本。

## Experiment 關係

實際 analysis path 中找到 119 個：

```
Exp_YYYYMMDD_HHMMSS_<8-char>
```

其中：

- 114 個可對上 `watch_runs.run_id`
- 5 個沒有相對應的 `watch_runs`

當可以對上時，能得到：

```
experiment/run
→ WatchID
→ TemplateID/version
→ point/capture/image
```

但 `experiment_results` 本身沒有 `experiment_id` 或 `run_id` 欄位，只能由 `local_path` 解析。

目前可暫時查：

```
SELECT
    er.*,
    wr.watchid,
    wr.template_id,
    wr.template_version,
    wm.Reference
FROM experiment_results er
LEFT JOIN watch_runs wr
       ON wr.run_id = :experiment_id
LEFT JOIN watch_metadata wm
       ON wm.WatchID = wr.watchid
WHERE instr(
    replace(er.local_path, '/', '\'),
    '\Analysis\' || :experiment_id || '\'
) > 0;
```

某個 value 可以用：

```
SELECT DISTINCT
    watchid,
    json_extract(data, '$.metrics.bump_density') AS value
FROM experiment_results
WHERE task_name = :analysis_name
  AND data <> '{}'
  AND json_extract(data, '$.metrics.bump_density') IS NOT NULL;
```

但目前只能算「探索性查詢」，不適合正式統計，因為缺少 `analysis_run_id` 與 `source_asset_uuid`，且 payload 有重複。

程式已有各 analysis identifier/value path mapping，例如 `metrics.bump_density`、`metrics.feature_area`、OCR、lume、crown 等，見 [report_identifiers.py (line 4)](/D:/Provenance Laboratories projects/ImagingLibWatch/core/report_identifiers.py:4)。這些 mapping 很適合拿來建立 normalized `analysis_values` table。

## Camera Pipeline TXT

Camera Pipeline TXT 內容包含：

- watchpoint
- capture parameters
- exposure
- gain/light settings
- autofocus/HDR config
- output image path
- raw image ID
- S3 key
- run ID／experiment ID

Payload 定義在 [camera_pipeline_report.py (line 91)](/D:/Provenance Laboratories projects/ImagingLibWatch/core/camera_pipeline_report.py:91)。

目前 TXT path 存在：

```
image_assets.metadata.camera_pipeline_report_path
```

實際 coverage：

- 568／3,471 image assets 有 TXT path
- exposure 則 3,471／3,471 都已存在 `image_assets.metadata.exposure`
- 同時也可從 `point_instances.hardware_cfg.exposure` 查 template/hardware 原始設定

查詢：

```
SELECT
    json_extract(metadata, '$.asset_id') AS file_uuid,
    watchid,
    run_id,
    view_name,
    point_name,
    capture_id,
    json_extract(metadata, '$.camera_pipeline_report_path') AS txt_path,
    json_extract(metadata, '$.exposure') AS actual_exposure
FROM image_assets
WHERE LOWER(json_extract(metadata, '$.asset_id')) = LOWER(:uuid);
```

缺點是 TXT 沒有自己的 UUID、hash、size、S3 key 或獨立 asset row。

## Template

`local_templates` 已保存：

- TemplateID
- version
- Reference
- Brand/Model
- watchView/watchpoints
- hardware/capture conditions
- material scan point設定
- WatchShift flags/manifest
- updated_at

Template model 見 [template_structure.py (line 21)](/D:/Provenance Laboratories projects/ImagingLibWatch/DB/templates/template_structure.py:21)。

可以查 template 用在哪些 scans：

```
SELECT
    t.TemplateID,
    t.version,
    json_extract(t.data, '$.Reference') AS Reference,
    datetime(t.updated_at, 'unixepoch', 'localtime') AS updated_at,
    r.run_id,
    r.watchid,
    datetime(r.started_at, 'unixepoch', 'localtime') AS scan_started_at
FROM local_templates t
LEFT JOIN watch_runs r
       ON r.template_id = t.TemplateID
WHERE t.TemplateID = :template_id;
```

限制：

- 只有 `updated_at`，沒有可靠 `created_at`。
- App 建立 run 時常未傳正確 template version，因此預設成 `v1`。
- TemplateScratch images 沒有 dedicated index。
- Template → analysis 仍需經過 run，再由 analysis path 推導 experiment。

Template 建立影像目前寫到：

```
Local_Data/<TemplateID>/TemplateScratch/
```

見 [main.py (line 5224)](/D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:5224)，但不會註冊到 SQLite。

## WatchShift reference

目前有部分 index：

```
local_templates.data.watchshift_images
local_templates.data.watchshift_image_manifest
```

21 個 templates 中：

- 8 個有 WatchShift flags
- 8 個有非空 manifest

Manifest 包含：

- template_id
- view_name
- S3 key
- bucket
- etag
- version ID（如果 S3 有）
- updated_at

Manifest 定義在 [internalnum_config.py (line 1725)](/D:/Provenance Laboratories projects/ImagingLibWatch/DB/templates/internalnum_config.py:1725)。

但 reference image 沒有獨立 DB asset；local path 是依規則推導：

```
DB/watchshift/<TemplateID>/<View>.toppoint1.png
```

每次 scan 的 WatchShift 計算結果則寫成：

```
Analysis/<ExpID>/watchshift_<View>.json
```

見 [main.py (line 3511)](/D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:3511)，通常沒有註冊進 `experiment_results`。

所以答案是：有 template-level manifest，但沒有完整的 WatchShift file/run-level index。

## Material／XRF

目前 Material CSV 解析內容很完整，包括：

- source file/path
- scan part
- scan mode
- instrument metadata
- alloy
- gold plated／karat
- element
- concentration %
- error %
- raw values

結構見 [material_structure.py (line 78)](/D:/Provenance Laboratories projects/ImagingLibWatch/DB/templates/material_structure.py:78)。

Material session JSON 也會記：

- WatchID
- ExpID
- material CSV folder
- records
- element values

但它只寫到：

```
Analysis/<ExpID>/material_records.json
```

見 [main.py (line 23665)](/D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:23665)，沒有 dedicated DB insert。

資料也會暫時放到 `current_watchentry.watchmaterial` 與 `material_results_by_internalnum`，見 [material_reader.py (line 576)](/D:/Provenance Laboratories projects/ImagingLibWatch/algorithms/material_reader.py:576)；只有後續成功存 Final Report 時才可能進 JSON blob。

實際 DB 現況：

- 113 個 `watch_metadata` 沒有任何已填 material data
- 139 個 Final Report 沒有任何非空 `material_results_by_internalnum`
- 沒有 `material_scans` 或 `material_values` table

因此目前無法直接用 SQL：

- 比較同 Reference 的組成 %
- 比較不同 scan 的 Au、Fe、Ni 等成分
- 取得某元素的 distribution

## PDF 與 report identifier

目前 DB 有 15 個 PDF rows：

```
task_name = Auto_PDF_Report
result_type = data_pdf
```

保存：

- WatchID
- PDF path
- generated_at
- source

註冊程式見 [main.py (line 24438)](/D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:24438)。

問題是：

- 沒有 `report_id` 欄位
- 沒有 experiment ID
- 沒有 template ID
- 沒有 source image/analysis link
- PDF 裡的數值沒有寫成 DB values
- report identifier JSON 檔沒有註冊到 DB
- PDF 顯示的 `report_id` 目前是 hard-coded `"5473454756"`，見 [main.py (line 28248)](/D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:28248)

PDF filename 的 8 字元 suffix 雖然可以暫時當 file identifier，但不是正式 report ID。

因此：

- PDF → WatchID：可以
- PDF → Reference：需再 join `watch_metadata`，而 Reference coverage 很低
- PDF → experiment/image/analysis：目前不可靠
- 比較兩份 PDF 裡的數值：必須重讀 PDF 或 `report_identifiers_*.json`，不能直接由 DB 查

## 建議的目標資料模型

應保留現有表以維持 App／CLI／cloud 相容，再 add-only 增加以下結構：

````
```mermaid
flowchart LR
    T["Template version"] --> TP["Template points / internalnum1"]
    T --> TA["Template assets<br/>scratch / WatchShift"]

    W["Watch / normalized reference"] --> R["Watch run / experiment"]
    R --> P["Point instance"]
    P --> C["Capture instance"]
    C --> A["Image assets<br/>multiple images + optional HDR"]

    A --> AR["Analysis run"]
    AR --> AV["Analysis values / identifiers"]

    C --> CP["Camera pipeline report"]
    R --> MS["Material scan"]
    MS --> MV["Element concentration values"]

    R --> PDF["PDF report"]
    PDF --> RI["Report identifiers"]
    RI --> AR

    FA["Generic file assets"] --- A
    FA --- TA
    FA --- CP
    FA --- MS
    FA --- PDF
```
````

最少需要新增：

1. `file_assets`
    
    - `file_uuid`
    - `file_type`
    - `local_path`
    - `s3_key`
    - `sha256`
    - `mime_type`
    - `byte_size`
    - `created_at`
    - `metadata_json`
2. `experiments`
    
    - `experiment_id`
    - `run_id`
    - `analysis_dir`
    - `started_at`
    - `completed_at`
    - `status`
3. `analysis_runs`
    
    - `analysis_run_id`
    - `experiment_id`
    - `source_asset_uuid`
    - `algorithm_name`
    - `algorithm_version`
    - `result_document_asset_uuid`
    - `created_at`
4. `analysis_values`
    
    - `analysis_run_id`
    - `identifier_key`
    - `value_path`
    - `value_numeric`
    - `value_text`
    - `unit`
    - `ordinal`
    - `passed`
5. `reports`＋`report_analysis_links`
    
    - 正式 `report_id`
    - PDF file UUID
    - run/experiment/template/reference snapshot
    - 關聯 analysis run／identifier
6. `material_scans`＋`material_values`
    
    - scan ID
    - run／watchpoint／internalnum1
    - source CSV asset
    - element
    - concentration %
    - error %
    - unit
7. `template_assets`
    
    - TemplateID/version
    - view/point
    - asset type：scratch、WatchShift reference、calibration
8. `authenticity_assessments`
    
    - scope：watch/component/point/analysis
    - normalized status
    - source analysis/report
    - evaluated_at

## 必要 indexes

建議至少增加：

```
file_assets(file_uuid) UNIQUE
file_assets(local_path) UNIQUE
file_assets(sha256)

watch_runs(watchid, started_at)
watch_runs(template_id, template_version, started_at)

point_instances(run_id, internalnum1)
point_instances(watchid, view_name, point_name)

capture_instances(point_instance_id, internalnum2)
capture_instances(run_id, capture_id)

image_assets(file_uuid) UNIQUE
image_assets(capture_instance_id, asset_role, asset_index)
image_assets(watchid, internalnum1, internalnum2)

experiments(run_id, experiment_id)
analysis_runs(source_asset_uuid, algorithm_name)
analysis_runs(experiment_id, algorithm_name)
analysis_values(identifier_key, value_numeric)
analysis_values(analysis_run_id, identifier_key)

watch_metadata(reference_normalized)
material_scans(run_id, internalnum1)
material_values(element, concentration_percent)

reports(run_id, report_id)
report_analysis_links(report_id, analysis_run_id)
```

並應啟用真正的 foreign keys；目前現有 schema 沒有任何 FK constraint。

## 建議實作順序

為維持舊 App、CLI、DB、S3、DynamoDB 與報告相容，建議：

1. 先統一 `file_uuid`，保留原 `asset_id` 和 legacy integer ID。
2. 新增 `file_assets`，回填正式 image、analysis、PDF、TXT、WatchShift、material、template files。
3. 新增 explicit `experiment_id` 與 `analysis_run_id`。
4. 將 analysis identifiers/value mappings 寫入 `analysis_values`。
5. 新增 material/report tables。
6. 建立 compatibility views，讓舊 `raw_images`／`experiment_results` 繼續可用。
7. 修改 dual-write，避免同一 image path 重複兩筆 `raw_images`。





#### 整理all files 連結 Database
```
之前有整理了這個App main.py如果有儲存images or 其他files有包括1. 正式拍攝影像, 2. 分析結果, 3. Camera pipeline TXT 報告, 4. Template 建立影像, 5. WatchShift 參考影像, 6. Material／XRF 檔案, 7. PDF 與報告檔, 8. SQLite 資料庫, 9. Log 與稽核檔案, 10. 其他零散輸出與快取, 及yaml files.

請幫我整理這些檔案的data or index等有哪些會存入local DB方便我們管理所有的files跟後續的分析譬如:

尋找某個uuid的image file是屬於哪個watchid, 是屬於哪個template, 是甚麼時候拍的, 是哪個watchpoint, 以及跟這個uuid相關的analysis結果是存在哪裡哪個experiment? 跟這個uuid相關的analysis結果的某個結果數值value? 跟這個uuid相關的pdf report的identifier是哪一個? 有哪個另外的uuid跟這個同屬於同一次拍照, 有哪個另外的uuid跟這個同屬於同個watchpoint?, 有哪個另外的uuid跟這個同屬於同個reference number? 跟這個uuid相關的watchpoint不同capture的uuid是? 找到所有屬於同個reference number的所有這個watchpoint的uuid, 找到所有屬於同個watchpoint的所有的uuid, 找到找到所有屬於同個reference number的所有這個watchpoint且authentication是真品的uuid

Analysis\Exp_YYYYMMDD_HHMMSS_<8字元ID>有包含哪些analysis results, analysis結果的某個結果數值value? 跟這個uuid相關的pdf report的某個analysis的identifier是哪一個? 這個experiment是從哪一次watchid來的? 是從哪個template? 這個watchid還有沒有其他experiments, 找到所有屬於同個reference number的所有這個watchpoint的uuid, 找到所有屬於同個analysis result的所有的uuid, 找到找到所有屬於同個reference number的所有這個analysis result且authentication是真品的values, 要對這個experiment裡面某個analysis找到所有相關values的distribution.

要怎麼找到Camera pipeline TXT 報告是屬於哪次拍照, 是屬於哪個template建立的, 其中的某個數值譬如拍照的exposure

尋找某個template是屬於哪個reference number, 以及用在哪個watchid, 是甚麼時候建立的, 跟哪些watchpoint有關, 以及跟這個template相關的analysis結果是存在哪裡哪個experiment? 跟這個template相關的analysis結果的某個結果數值value? 有哪個另外的template跟這個同屬於同個reference number? 找到所有屬於同個template的所有這個watch scan

是否有儲存WatchShift 參考影像相關index在database裡面

關於material data是屬於哪一次scan? 是屬於哪個reference number? 跟同個reference number的material values是否能比較不同的成分組成%? 跟同個reference number的material values是否能比較分析特定成分%?

能否找到pdf file是從哪次watch scan? 這個pdf file是屬於哪種reference number? pdf file裡面的某個數值是甚麼? pdf file裡面某數值跟另外一份pdf file數值的比較

local DB能否查詢裡面有哪些種類跟內容? 譬如有哪些template? 有哪些watch scan? 以及還有甚麼是local DB有儲存的. 請幫我設計提供建議先不要改code 及修改database能夠有以上功能性可以進行那些查詢跟分析, 這database也要跟AWS D做同步
```
目前 local SQLite **只能部分支援**你列出的查詢。近期正式拍攝的「watch → run → watchpoint → capture → image」已有雛形，但還不能可靠地從任意 UUID 一路追到 analysis、experiment、PDF、material、authentication 與 AWS。

本次只做唯讀盤點，**沒有修改任何 code 或 database**；目前 Git working tree 也沒有新增變更。

### 最重要的現況問題

1. 檔案 UUID 與 DB `asset_id` 是兩套 ID  
    3,472 筆 `image_assets.asset_id` 都是 `ast_xxx`，沒有一筆等於影像檔名 UUID。真正的檔名 UUID 藏在 `metadata.asset_id` JSON 裡，且沒有索引。
    
2. 新 analysis 關聯表完全沒有資料  
    `analysis_results_v2 = 0`；9,813 筆結果仍在舊 `experiment_results`。
    
3. 9,813 筆分析結果中，9,388 筆沒有 `raw_image_id`  
    只有 425 筆 `sidepoint_service` 有直接 source-image 關聯。因此多數「UUID → analysis」只能靠路徑、檔名或 JSON 猜測。
    
4. 沒有正式的 `experiments` table  
    `Exp_YYYYMMDD...` 目前主要只是資料夾名稱。App 某條路徑會讓 `run_id = exp_id`，另一條 workflow 路徑則把兩者分開，所以不能視為穩定 contract。[App/main.py (line 34376)](/D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:34376)、[core/workflow_manager.py (line 479)](/D:/Provenance Laboratories projects/ImagingLibWatch/core/workflow_manager.py:479)
    
5. 狀態沒有結束  
    1,372 筆 `watch_runs` 全是 `running`；3,472 筆 `capture_instances` 全是 `processing`。
    
6. `reference number`、authentication 與 internalnum 覆蓋率很低  
    只有 82/3,472 個 image assets 有 `internalnum1/2`；只有 232 個 image assets 能 join 到有 Reference 的 watch metadata。113 個 `watch_metadata` 中只有 8 筆有 Reference，且目前沒有實際 authentication identifier snapshot。
    
7. 很多檔案完全沒有進 DB  
    TemplateScratch、material JSON、WatchShift reference file、identifier JSON、camera TXT 本身、audit/runtime logs、run YAML、部分 PDF 都沒有統一 asset row。
    
8. AWS 同步不是通用同步  
    worker 只直接輪詢舊 `raw_images`、`experiment_results`；`add_to_queue()` 本身是 no-op。[DB/db_manager.py (line 412)](/D:/Provenance Laboratories projects/ImagingLibWatch/DB/db_manager.py:412)、[data_manager/cloud_sync.py (line 74)](/D:/Provenance Laboratories projects/ImagingLibWatch/data_manager/cloud_sync.py:74)
    
9. 設定檔內有明文 AWS 長期憑證  
    建議立即撤銷、輪替，改用 IAM role／environment／Secrets Manager，並清除 Git 歷史中的舊值。此項優先級高於 DB refactor。
    

---

## 實際 SQLite 盤點

目前 [watchDB.db](/D:/Provenance Laboratories projects/ImagingLibWatch/DB/watchDB.db) 約 343 MiB，`integrity_check=ok`，但：

- `PRAGMA user_version = 0`
- 沒有 migration version
- 沒有宣告任何 foreign key
- `foreign_keys=OFF`
- journal mode 為 `DELETE`

|Table|筆數|現況|
|---|---|---|
|`raw_images`|7,185|舊影像 index；包含雙寫重複|
|`experiment_results`|9,813|主要分析、entry、PDF 資料|
|`watch_runs`|1,372|run/template 關聯，但全為 running|
|`point_instances`|4,862|run 中的 watchpoint|
|`capture_instances`|3,472|capture group，但全為 processing|
|`image_assets`|3,472|新影像表|
|`analysis_results_v2`|0|schema 存在但沒有實際資料|
|`local_templates`|23|template JSON cache|
|`watch_metadata`|113|watch 基本資料及完整 JSON|
|`ui_asset_registry`|6,218|UI 最新顯示檔案，會覆蓋歷史|
|`users`|8|local account cache|
|`watch_registry`|0|未實際使用|

建表與雙寫邏輯在 [DB/db_manager.py (line 54)](/D:/Provenance Laboratories projects/ImagingLibWatch/DB/db_manager.py:54)。

另外，3,472 個 V3 image path 在 `raw_images` 都被再登錄一次，因此舊表出現一份 capture 兩筆相同路徑的情況。

---

## 目前各類檔案能否查詢

|類別|現況|
|---|---|
|正式拍攝影像|部分可查。可經 `image_assets` 找 watch/run/view/point/capture/time，但檔名 UUID 藏在 JSON|
|Analysis results|不完整。結果在舊表，95.7% 沒有 source raw FK；V2 表為空|
|Camera pipeline TXT|TXT 內容完整，含 template、watchpoint、exposure、capture/run context；但 TXT 沒有獨立 DB asset row。569 個 image metadata 曾存其 path|
|Template 建立影像|`TemplateScratch` 直接寫檔，未進 DB。現有檔案樹抽查到 70 個未登錄檔案。[App/main.py (line 6762)](/D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:6762)|
|WatchShift reference|23 個 template 都有 flags，8 個有非空 manifest；檔案本身沒有 asset row。目前檔案樹有 3 個 reference images。[App/main.py (line 4436)](/D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:4436)|
|Material/XRF|parser model 很完整，但目前實際 DB 沒有 source file 或 element measurements；只找到未登錄的 `material_records.json`。[App/main.py (line 31225)](/D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:31225)|
|PDF/report|只有 15 筆 auto PDF row，全部沒有 S3 key；manual PDF 不一定登錄。PDF 內 `report_id` 目前還是固定常數，不是唯一 ID。[App/main.py (line 32023)](/D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:32023)、[App/main.py (line 36261)](/D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:36261)|
|SQLite DB|可列 table，但 active DB 不應把自己登錄在自己裡；應登錄 DB backup|
|Log/audit|audit、runtime JSONL 都沒有 DB index。audit rotation 的 cloud enqueue 實際不工作|
|YAML/run manifests|多數直接寫檔，沒有 DB asset row|
|Cache/temp/debug|沒有統一 retention/index policy|

Camera TXT 本身的資料結構足以提取 exposure 等欄位，見 [core/camera_pipeline_report.py (line 42)](/D:/Provenance Laboratories projects/ImagingLibWatch/core/camera_pipeline_report.py:42)。

---

## 你列出的查詢，目前能做到什麼

### UUID 查詢

目前可部分回答：

- 哪個 WatchID
- view／watchpoint
- capture ID
- run ID
- 拍攝時間
- 同一 `capture_instance_id` 的其他 image
- 同一 watchpoint 的其他 image
- 使用哪個 template：經 `image_assets.run_id → watch_runs.template_id`

但必須先解決 UUID：

```
filename UUID
  ≠ image_assets.asset_id
  = image_assets.metadata.asset_id
```

所以目前需要未索引的 `json_extract(metadata, '$.asset_id')`，不適合作為正式管理 API。

以下目前不能可靠回答：

- 該 UUID 的所有 analysis result
- analysis 所屬 experiment
- PDF report identifier
- 同 reference 且 authenticity=真品的 UUID
- multi-image/stitch analysis 的所有 input UUID

### Experiment 查詢

可以從 `experiment_results.local_path` 推測 `Exp_...` 並列出內容，但 DB 沒有：

- experiment row
- experiment → scan/run FK
- experiment → template FK
- experiment → input assets 多對多表
- experiment status/version/config snapshot

Analysis value 目前在 `data` JSON 中。已發現 2,161 筆含 `metrics`、805 筆含 `identifiers`，可以針對已知 JSON path 查，但無法穩定列出所有 metric、單位或做高效率 distribution。

### Template

目前可查：

- 23 個 template
- 17 個 template 有非空 Reference
- template JSON 裡的 watchpoints
- `watch_runs.template_id` 可查曾用在哪些 run
- `updated_at`

不足之處：

- 沒有可靠 `created_at`
- template image 沒有 DB index
- 沒有 immutable revision/content hash
- 1,372 個 run 中只有 1,234 個能對到目前 local template cache

### Material／PDF

Material parser 已能保存元素名稱、濃度%、error、unit 等結構，[material_structure.py (line 50)](/D:/Provenance Laboratories projects/ImagingLibWatch/DB/templates/material_structure.py:50)；問題是這些值沒有正規化寫進 DB。

PDF 可生成 identifier JSON，但該 JSON 及 identifier values 沒有進統一 DB catalog。[App/main.py (line 35577)](/D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:35577)

---

# 建議的 V4 database 設計

不要刪除或直接修改舊表。建議新增 V4 tables、views 與 dual-write，舊 App/CLI 先繼續讀舊表。

核心階層應固定為：

```
Reference / Watch Item
  → Scan Session
    → Point Visit
      → Capture Event
        → 多個 Artifacts（standard、HDR brackets、HDR fused）
          → Analysis Run
            → 多個 Analysis Inputs / Outputs / Metric Values
              → Report
```

這樣 point-level metadata 與 image-level metadata可以保持分離。

## 核心 tables

|Table|主要用途|
|---|---|
|`watch_items`|實體手錶／reference／serial identity|
|`scan_sessions`|一次完整 watch scan；保存 reference、template、authentication snapshot|
|`template_revisions`|immutable template version、Reference、content hash、created_at|
|`watchpoint_definitions`|template 中穩定的 point identity/internalnum1|
|`point_visits`|某次 scan 實際拜訪該 point|
|`capture_events`|一次 standard/HDR acquisition；internalnum2、曝光、狀態|
|`artifacts`|所有 image、TXT、JSON、YAML、PDF、material、log、DB backup|
|`artifact_aliases`|filename UUID、`ast_xxx`、legacy raw ID、舊 S3 key 對到 canonical UUID|
|`artifact_relations`|`source_of`、`derived_from`、`hdr_member_of`、`report_includes`|
|`experiments`|明確登錄 `Exp_...`、scan、run、template、folder、status|
|`analysis_runs`|algorithm/config/model version、experiment、started/completed|
|`analysis_inputs`|analysis 與多個 source asset 的 many-to-many 關聯|
|`analysis_results`|一次 task result、JSON artifact、status|
|`metric_definitions`|穩定 metric key、label、type、unit、version|
|`metric_values`|numeric/text/bool/json value；連回 result、asset、point|
|`reports`|唯一 report UUID、PDF artifact、scan/experiment/template|
|`report_metric_snapshots`|PDF 當時實際渲染的 identifier/value/page/section|
|`material_scans`|XRF source file、scan、point、instrument、method|
|`material_values`|element、concentration%、error%、unit|
|`auth_evaluations`|scan/component/point 的 authentic/not_authentic/unknown|
|`config_snapshots`|該次 scan/analysis 使用過的 YAML hash 與 snapshot artifact|
|`sync_outbox`|通用 AWS sync queue、retry/error/idempotency|

`artifacts` 建議至少包含：

```
artifact_uuid PK
artifact_kind
asset_role
scan_id
point_visit_id
capture_event_id
experiment_id
local_path
s3_bucket
s3_key
mime_type
size_bytes
sha256
created_at
retention_class
local_status
cloud_status
metadata_json
```

最重要規則是：

> 新資料的 `artifact_uuid` 必須等於檔名 UUID，並同時成為 local DB、S3 metadata 與 DynamoDB identity。

舊 `ast_xxx`、numeric `raw_image_id` 全部放到 `artifact_aliases`，不要再產生第三套 ID。

## 必要 indexes

```
UNIQUE artifacts(artifact_uuid);
UNIQUE artifacts(local_path);
UNIQUE artifacts(s3_bucket, s3_key);

INDEX artifacts(capture_event_id, asset_role);
INDEX artifacts(scan_id, created_at);
INDEX point_visits(scan_id, watchpoint_id);
INDEX scan_sessions(reference_number, started_at);
INDEX scan_sessions(template_revision_id, started_at);

INDEX analysis_inputs(artifact_uuid, analysis_run_id);
INDEX analysis_results(analysis_run_id, result_type);
INDEX metric_values(metric_definition_id, numeric_value);
INDEX metric_values(artifact_uuid, metric_definition_id);

INDEX material_values(element_symbol, concentration_percent);
INDEX auth_evaluations(scan_id, status);
INDEX sync_outbox(status, next_retry_at);
```

另建四個 read views：

- `v_asset_context`
- `v_analysis_metric_fact`
- `v_report_context`
- `v_material_fact`

---

## 新 schema 可直接支援的查詢

同一次拍照的全部 UUID：

```
SELECT artifact_uuid, asset_role
FROM artifacts
WHERE capture_event_id = (
    SELECT capture_event_id
    FROM artifacts
    WHERE artifact_uuid = ?
);
```

同 reference、同 watchpoint、authentication 為真品：

```
SELECT a.artifact_uuid, s.scan_id, s.started_at
FROM artifacts a
JOIN point_visits pv ON pv.point_visit_id = a.point_visit_id
JOIN scan_sessions s ON s.scan_id = a.scan_id
JOIN auth_evaluations ae ON ae.scan_id = s.scan_id
WHERE s.reference_number = ?
  AND pv.watchpoint_id = ?
  AND ae.status = 'authentic';
```

Experiment 某 analysis metric 的 distribution：

```
SELECT mv.numeric_value
FROM metric_values mv
JOIN analysis_results ar ON ar.analysis_result_id = mv.analysis_result_id
JOIN analysis_runs run ON run.analysis_run_id = ar.analysis_run_id
WHERE run.experiment_id = ?
  AND mv.metric_definition_id = ?;
```

Camera pipeline exposure：

```
SELECT ce.exposure_us, ce.camera_id, ce.gain,
       a.local_path AS camera_report_path
FROM capture_events ce
LEFT JOIN artifacts a
  ON a.capture_event_id = ce.capture_event_id
 AND a.artifact_kind = 'camera_pipeline_report'
WHERE ce.capture_event_id = ?;
```

同 reference 的 XRF 元素比較：

```
SELECT s.started_at, ms.scan_part, mv.element_symbol,
       mv.concentration_percent, mv.error_percent
FROM material_values mv
JOIN material_scans ms ON ms.material_scan_id = mv.material_scan_id
JOIN scan_sessions s ON s.scan_id = ms.scan_id
WHERE s.reference_number = ?
  AND mv.element_symbol = ?
ORDER BY s.started_at;
```

PDF 中使用的 identifier/value：

```
SELECT r.report_uuid, rms.identifier_key,
       rms.rendered_value, rms.page_number
FROM reports r
JOIN report_metric_snapshots rms ON rms.report_id = r.report_id
WHERE r.report_uuid = ?;
```

---

# AWS 同步建議

## S3

S3 保存檔案 bytes，建議新 key：

```
artifacts/v2/
  reference=<reference>/
  scan=<scan_uuid>/
  capture=<capture_uuid>/
  <artifact_uuid>.<ext>
```

舊 `Raw/<uuid>`、`Analysis/<uuid>` 保留，不需搬動；在 `artifact_aliases` 保存 legacy key。

## DynamoDB

DynamoDB 用於 operational lookup，不應塞大型 analysis JSON 或做整表 distribution scan。

建議 catalog access patterns：

```
PK=ASSET#<uuid>                  SK=META
PK=SCAN#<scan_id>               SK=POINT#...#CAPTURE#...#ASSET#...
PK=EXP#<experiment_id>          SK=RESULT#<analysis_run_id>#...
PK=REPORT#<report_uuid>         SK=METRIC#...
```

GSIs：

- `SCAN#scan_id → captured_at/asset`
- `REFPOINT#reference#watchpoint → captured_at/asset`
- `TEMPLATE#template#version → scan`
- `ASSET#uuid → experiment/result/report`

大量 metric distribution 建議同步成 S3 Parquet，由 Athena 查詢；DynamoDB保留單一 UUID、scan、experiment、report 的快速查找。

## 同步可靠性

不要再只用 `synced INTEGER`。每次 local transaction 同時新增 `sync_outbox`：

```
entity_type
entity_id
operation
revision
payload_hash
status
retry_count
next_retry_at
last_error
```

AWS write 必須使用 deterministic idempotency key；成功後記錄 DynamoDB version、S3 ETag/version ID。另做每日 reconciliation：

- DB 有、local file 無
- local file 有、DB 無
- DB 有、S3 無
- S3 有、DynamoDB 無
- checksum 不一致

---

# 建議導入順序

1. **先輪替明文 AWS credentials，備份 SQLite。**
2. 新增 migration framework、`schema_version`、foreign keys 與 V4 tables；不碰舊表資料。
3. 回填 canonical UUID 與 `artifact_aliases`：
    - `metadata.asset_id`
    - 檔名 UUID
    - `ast_xxx`
    - numeric `raw_image_id`
4. 由 `watch_runs/point_instances/capture_instances` 回填 scan lineage。
5. 從 `Exp_...` path 建立 experiments；無法確定的 row 標記 `lineage_confidence='inferred'`，不要假裝精確。
6. 新 capture 流程 dual-write 舊表與 V4，並正確完成 run/capture status。
7. Analysis adapter 統一登錄 inputs、outputs、metric values；stitch/multi-image analysis 必須用 junction table。
8. 接入 Camera TXT、TemplateScratch、WatchShift、Material、PDF、identifier JSON、YAML、audit/log。
9. 改成 universal outbox → S3/DynamoDB，完成後再考慮逐步停用舊表。

## 人工驗證重點

- 同一 watchpoint 拍 2 張 standard 加 1 組 HDR，確認 capture siblings、brackets、fused lineage。
- 同 reference 用不同 template 做兩次 scan，確認查詢不互相覆蓋。
- 一次 multi-image stitch analysis，確認所有 input UUID 可回溯。
- 產生 PDF，確認唯一 report UUID、identifier snapshot 與來源 metric。
- 匯入兩次 XRF，確認同 reference／同元素可比較 concentration distribution。
- offline capture 後恢復網路，確認 outbox 可重試且不重複上傳。
- 驗證 App、CLI、report、S3、DynamoDB 舊流程仍可運作。
- 對目前大量 stale DB paths 做 dry-run reconciliation；在人工確認前不要刪資料。




#### database支援上萬watch?
```
所以新的database不但要能夠執行以上提到的查詢跟分析, 而且我們可能有上千個watch系列, 十萬了watch. 跟你你設計的新database是否能否順暢的運行support這些功能性
```
可以順暢支援，但必須採用「分層式 database architecture」。如果把十萬支 watch、數千萬 image assets 和數億筆 analysis values 全塞在單一 `watchDB.db`，最後一定會遇到檔案過大、寫入鎖、備份緩慢及 distribution 查詢過慢。

## 建議的正式架構


```mermaid
flowchart LR
    App["App / CLI / Hardware"] --> Local["每台設備 Local SQLite V4"]
    Local --> Outbox["Transactional Sync Outbox"]
    Outbox --> Catalog["DynamoDB V4 Operational Catalog"]
    Outbox --> S3["S3 Artifact Storage"]
    Outbox --> Facts["Parquet / Iceberg Metric Facts"]
    Facts --> Athena["Athena Analytics"]
    Catalog --> API["UUID / Watch / Capture / Experiment API"]
    Athena --> Analysis["Distribution / Comparison / Statistics"]
```


各層職責：

|儲存層|保存內容|主要查詢|
|---|---|---|
|Local SQLite V4|本機最近資料、未同步資料、capture lineage|即時拍攝、UUID、watchpoint、experiment|
|DynamoDB|所有 entity／artifact 的精簡 index|UUID、scan、capture、template、report 快速查找|
|S3|image、TXT、JSON、YAML、PDF、XRF、log 原始檔|檔案保存與下載|
|S3 Parquet/Iceberg + Athena|正規化 analysis metrics、material values|distribution、reference 比較、authentication 統計|

DynamoDB 單一 item 上限是 400 KB，因此完整 analysis JSON、PDF、影像不能直接放入 item；AWS 也建議大型資料存 S3，在 DynamoDB 保存 object identifier。[AWS DynamoDB constraints](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Constraints.html)、[AWS large-item guidance](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/bp-use-s3-too.html)

---

## 十萬支 watch 的大致資料量

假設：

- 100,000 次 watch scan
- 每次 50 個 watchpoints
- 每個 watchpoint 平均 2 個 capture events
- 每個 capture 平均 2 個 artifacts，含 standard/HDR brackets/fused
- 每個 point 跑 5 個 analysis
- 每個 analysis 平均 20 個 metric values

大約會得到：

|資料|預估筆數|
|---|---|
|`scan_sessions`|100,000|
|`point_visits`|5,000,000|
|`capture_events`|10,000,000|
|`artifacts`|約 20,000,000|
|`analysis_results`|約 25,000,000|
|`metric_values`|約 500,000,000|

100,000 個 watch 和幾千個 template 本身很小；真正造成壓力的是數千萬 artifacts 與數億 metric values。

因此：

- `scan_sessions/templates/watch metadata`：SQLite、DynamoDB 都很輕鬆。
- `artifacts/capture lineage`：DynamoDB適合，local SQLite 只保留 station 所需資料。
- 500 million metrics：不適合全部留在每台設備的 SQLite，也不適合為 distribution 直接掃 DynamoDB；應進 Parquet/Iceberg/Athena。

---

## 各類查詢會走哪一層

### 即時 UUID 查詢

例如：

> 這個 UUID 屬於哪個 WatchID、template、watchpoint、capture、experiment、PDF？

走 Local SQLite 或 DynamoDB：

```
PK = ASSET#<uuid>
```

這是直接 key lookup，不需要 scan。正確建立 key 與 index 後，100,000 watches 和 20 million assets 不會構成問題。

### 同一次 capture 的其他 UUID

使用：

```
GSI1PK = CAPTURE#<capture_event_id>
GSI1SK = ROLE#<asset_role>#ASSET#<uuid>
```

可以一次取得：

- raw standard
- HDR brackets
- HDR fused
- camera TXT
- derived preview

### 同 reference、同 watchpoint 的全部 UUID

使用：

```
GSI2PK = REFPOINT#<reference>#<watchpoint_id>#<shard>
GSI2SK = <captured_at>#<scan_id>#<uuid>
```

必須分頁，不能一次傳回數十萬筆。

如果某個熱門 reference 特別大，可以用 16 或 32 個 hash shards，避免所有 request 集中在同一 DynamoDB partition。AWS 建議 partition key 應讓讀寫均勻分散；單一 partition 有固定 throughput 上限。[AWS partition-key guidance](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/bp-partition-key-design.html)

### Authentication 為真品的 UUID

不要在查詢時解析巨大 `watchentry.Full_JSON`，應把 authentication 正規化：

```
scan_sessions.authentication_status
auth_evaluations.component
auth_evaluations.watchpoint_id
auth_evaluations.status
```

DynamoDB GSI 可設：

```
GSI3PK = REFAUTH#<reference>#AUTHENTIC
GSI3SK = POINT#<watchpoint>#<captured_at>#<uuid>
```

### Analysis value 與 distribution

單一 UUID 的 analysis value：

- Local SQLite `metric_values`
- DynamoDB `ASSET#uuid / METRIC#metric_key`

大量 distribution，例如：

> 同 reference、同 analysis result、authentication=真品的 value distribution

應走 Athena：

```
SELECT
    approx_percentile(numeric_value, 0.5) AS median,
    avg(numeric_value),
    stddev_pop(numeric_value),
    min(numeric_value),
    max(numeric_value)
FROM analysis_metric_facts
WHERE reference_number = ?
  AND watchpoint_id = ?
  AND metric_key = ?
  AND authentication_status = 'authentic';
```

Athena 應使用 Parquet/Iceberg，依月份或資料類型 partition，並對高 cardinality 的 reference／watchpoint 使用 bucketing。AWS 官方建議 columnar format、適當 partition、壓縮及避免大量 small files。[Athena optimization](https://docs.aws.amazon.com/athena/latest/ug/performance-tuning-data-optimization-techniques.html)、[Athena Iceberg optimization](https://docs.aws.amazon.com/athena/latest/ug/querying-iceberg-data-optimization.html)

---

## Local SQLite 能保留多少資料

SQLite 可以管理數百萬到數千萬個有索引的 metadata rows，但不建議每個 station 永久保存全公司的完整 metrics。

建議 local retention：

- 所有未同步資料
- 最近 6–12 個月的完整 scan metadata
- 最近一段期間的 analysis metrics
- 歷史 artifact catalog 的必要摘要
- 使用者指定 pin 的 watches
- template cache

已成功同步且超過 retention 的：

- 原始檔留在 S3
- metric facts 留在 Iceberg
- DynamoDB保留永久 operational index
- local 將大型 analysis JSON／cache 清理，但保留 UUID、S3 key、checksum、基本 lineage

如果某台 station 必須離線查詢全部十萬 watches，可另外下載 Parquet snapshot 並使用 local analytical engine；不應讓 GUI 直接掃 500 million-row SQLite table。

---

## SQLite 必須做的效能調整

新的 local DB 至少需要：

- `WAL` journal mode
- `foreign_keys=ON`
- 單一 writer queue，多個 read-only connections
- capture／assets／outbox 使用短 transaction
- business fields 正規化，不把 UUID、Reference、metric key 藏在 JSON
- 所有 UI list 強制 pagination
- `ANALYZE`／`PRAGMA optimize`
- migration `user_version`
- partial index，例如只索引未同步資料

關鍵 indexes：

```
CREATE UNIQUE INDEX ux_artifacts_uuid
ON artifacts(artifact_uuid);

CREATE INDEX ix_artifacts_capture
ON artifacts(capture_event_id, asset_role);

CREATE INDEX ix_artifacts_scan_time
ON artifacts(scan_id, created_at);

CREATE INDEX ix_scans_reference_time
ON scan_sessions(reference_number, started_at);

CREATE INDEX ix_point_visits_scan_point
ON point_visits(scan_id, watchpoint_id);

CREATE INDEX ix_analysis_input_asset
ON analysis_inputs(artifact_uuid, analysis_run_id);

CREATE INDEX ix_metrics_asset_key
ON metric_values(artifact_uuid, metric_definition_id);

CREATE INDEX ix_metrics_result_key
ON metric_values(analysis_result_id, metric_definition_id);

CREATE INDEX ix_outbox_pending
ON sync_outbox(status, next_retry_at)
WHERE status IN ('pending', 'retry');
```

目前 [watchDB.db](/D:/Provenance Laboratories projects/ImagingLibWatch/DB/watchDB.db) 使用 `DELETE` journal、沒有 foreign keys、沒有 schema version；正式擴充前必須先補齊。

---

## DynamoDB 不可以這樣設計

不建議：

```
PK = Reference
SK = Asset UUID
```

因為熱門系列可能有數十萬至數百萬 assets，容易形成 hot partition。

也不能：

- 把整個 watchentry 與所有 image/results 塞成一個 item
- 使用 DynamoDB `Scan` 尋找 UUID
- 使用隨機 UUID 當 chronology sort key
- 把所有 metrics 存成一個巨大 list
- 每個 value 寫一個獨立 S3 JSON small file

建議一個 entity 一個小 item，通常控制在數 KB 到數十 KB；大型 JSON 放 S3。

---

## 是否需要 Aurora PostgreSQL

以目前已經使用 DynamoDB 的系統來說，十萬 watches 可以先使用：

> SQLite V4 + DynamoDB Catalog + S3 + Iceberg/Athena

這套足以完成你列出的全部查詢。

但如果未來要求：

- 任意拖拉組合條件
- 大量 relational joins
- 使用者自訂 SQL
- 跨 reference/template/partner/material/report 的即時複合查詢
- 每個查詢都希望低於一秒

那麼中央 metadata database 建議增加 Aurora PostgreSQL，而 DynamoDB保留 UUID/capture 的 operational lookup。這不是十萬 watches 一定需要，而是「任意互動式 relational analytics」才需要。

---

## 建議效能目標

在正式 benchmark 時可以設定：

|查詢|目標|
|---|---|
|UUID → 完整 context|local p95 < 50 ms|
|UUID → DynamoDB context|p95 < 100 ms|
|同 capture assets|p95 < 100 ms|
|同 watchpoint 最近 100 筆|p95 < 200 ms|
|Experiment 全部內容|p95 < 500 ms|
|單一 metric 歷史值|p95 < 300 ms|
|數億 values distribution|Athena 約數秒至數十秒|
|Capture local DB transaction|< 50 ms，不等待 AWS|

Distribution 查詢可以預先產生 daily/monthly summary：

- count
- mean
- standard deviation
- min/max
- p05/p25/p50/p75/p95
- histogram bins

App 先顯示 cached summary，需要細節時才觸發 Athena。

## 最終判斷

**新的 V4 data model 本身可以支援數千 watch 系列及十萬 watches。**  
但 production architecture 應明確限制為：

- SQLite：station-local operational DB
- DynamoDB：global asset/entity catalog
- S3：所有檔案的 authoritative storage
- Iceberg/Athena：大型 metric/material distribution analytics
- Outbox：可靠且可重試的同步層

只要不把十萬 watches 的全部影像 metadata、完整 JSON 和數億 metrics 永久塞進單一 local SQLite，這個規模不會是架構瓶頸。





#### 「邊緣到雲端 (Edge-to-Cloud)」與「OLTP / OLAP 分離」架構
```
以下是Moonlight system的建議的正式架構架構圖, 根據我的需求: "所以新的database不但要能夠執行以上提到的查詢跟分析, 而且我們可能有上千個watch系列, 十萬了watch. 跟你你設計的新database是否能否順暢的運行support這些功能性"
可以順暢支援，但必須採用「分層式 database architecture」。如果把十萬支 watch、數千萬 image assets 和數億筆 analysis values 全塞在單一 `watchDB.db`，最後一定會遇到檔案過大、寫入鎖、備份緩慢及 distribution 查詢過慢。

請更詳細解釋各個單元以及其作用, 詳細解釋SQLite跟DynamoDB的差別, 以及這系統是怎麼運作的

flowchart LR
    App["App / CLI / Hardware"] --> Local["每台設備 Local SQLite V4"]
    Local --> Outbox["Transactional Sync Outbox"]
    Outbox --> Catalog["DynamoDB V4 Operational Catalog"]
    Outbox --> S3["S3 Artifact Storage"]
    Outbox --> Facts["Parquet / Iceberg Metric Facts"]
    Facts --> Athena["Athena Analytics"]
    Catalog --> API["UUID / Watch / Capture / Experiment API"]
    Athena --> Analysis["Distribution / Comparison / Statistics"]
```

```mermaid
flowchart LR
    App["App / CLI / Hardware"] --> Local["每台設備 Local SQLite V4"]
    Local --> Outbox["Transactional Sync Outbox"]
    Outbox --> Catalog["DynamoDB V4 Operational Catalog"]
    Outbox --> S3["S3 Artifact Storage"]
    Outbox --> Facts["Parquet / Iceberg Metric Facts"]
    Facts --> Athena["Athena Analytics"]
    Catalog --> API["UUID / Watch / Capture / Experiment API"]
    Athena --> Analysis["Distribution / Comparison / Statistics"]
```

這是一個非常經典且具備高度擴展性的「邊緣到雲端 (Edge-to-Cloud)」與「OLTP / OLAP 分離」架構。

針對你提到的規模（上千個 Watch 系列、十萬支 Watch、數千萬的影像與數億筆分析數據），這套架構是能夠**順暢運行且經得起未來擴展**的最佳實踐。如果將這些資料全塞在一個 `watchDB.db`（SQLite）中，你一定會遇到「寫入鎖死 (Write Locks)」、「查詢超時 (Timeout)」與「備份災難」等瓶頸。

以下為你詳細拆解這個架構的各個單元、SQLite 與 DynamoDB 的差異，以及資料是如何在這套系統中流動的。

## 1. 系統單元與作用解析

這個架構將系統分為三個主要層級：**Edge（邊緣端）、Sync（同步層）、Cloud（雲端層）**。

### Edge 層 (設備端)

- **App / CLI / Hardware (硬體設備/控制介面):** 第一線與硬體互動的程式。負責拍攝 Watch 的影像、執行 Experiment（實驗）並收集原始數據。
    
- **Local SQLite V4 (邊緣資料庫):** 位於每台設備本機的資料庫。它**只儲存該台設備當下需要的熱資料 (Hot Data)** 或最近幾天的測試紀錄。它的存在讓硬體在網路斷線時依然能全速運行，不用等待雲端回應。
    

### Sync 層 (同步層)

- **Transactional Sync Outbox (交易同步發件匣):** 這是一個保證資料「不遺失、不重複」的關鍵設計（Outbox Pattern）。當設備產生新資料時，會先寫入 Local SQLite，同時在 Outbox 記錄一筆「待同步任務」。背景程式會依序將 Outbox 裡的資料往雲端送，成功後才標記完成。這解決了網路不穩時的資料同步問題。
    

### Cloud 層 (雲端儲存與運算)

- **DynamoDB V4 Operational Catalog (營運目錄庫):** 負責處理「高頻率的單筆讀寫」。例如：查詢某支 Watch 的當前狀態、設備清單、或是特定 Capture 的 metadata。它是雲端系統的操作大腦。
    
- **S3 Artifact Storage (大檔案物件儲存):** 專門用來存放影像 (Images)、原始日誌檔或大型 JSON。S3 的儲存成本極低且容量無限，將大檔案從資料庫剝離，是保證資料庫效能的關鍵。
    
- **Parquet / Iceberg Metric Facts (大數據分析格式):** 這是將分析數據（Values）轉換成適合「欄位導向 (Columnar)」查詢的格式並存放在 S3。Parquet/Iceberg 非常適合壓縮大量重複的數據（例如幾億筆的測量點），且只讀取需要的欄位，速度極快。
    
- **Athena Analytics (無伺服器分析引擎):** 針對 Parquet 檔案執行標準 SQL 查詢的雲端服務。當你要分析「過去三個月內，特定 Watch 系列的良率分佈」時，Athena 可以在幾秒內掃描數億筆資料並給出結果。
    
- **API & Analysis (應用介面):** 面對終端使用者或 Web 儀表板的接口，將背後的複雜邏輯封裝起來。
    

## 2. 深入比較：SQLite vs. DynamoDB

為何不全用 SQLite？或者為何不全用 DynamoDB？這兩者的設計哲學完全不同，在架構中扮演互補的角色：

|**比較維度**|**Local SQLite V4 (邊緣端)**|**DynamoDB V4 (雲端營運端)**|
|---|---|---|
|**核心架構**|**單機關聯式資料庫** (檔案型)。所有資料存在一個本地檔案中。|**全託管分散式 NoSQL 資料庫**。資料自動分散在多個伺服器節點。|
|**資料關聯性**|支援複雜的 JOIN (關聯表查詢) 與 ACID 交易。|適合 Key-Value 或 Document 結構，不支援跨表 JOIN，強調極速的單筆讀取。|
|**規模上限**|GB 級最佳。檔案超過幾十 GB 後，查詢與備份速度會明顯下降。|**無上限 (PB 級)**。即便存入十萬支 Watch 或千萬筆 Metadata，效能依然穩定。|
|**併發能力 (Concurrency)**|較弱。寫入時會產生檔案鎖 (File/WAL Lock)，多個寫入者會互相排隊等待。|**極強**。支援每秒數十萬次的並行讀寫，完全不會互相卡死。|
|**在此架構的定位**|**「快取與單機操作」**。只存特定設備近期的擷取與測試資料，保證硬體反應速度。|**「全域目錄」**。儲存所有設備、十萬支 Watch 的總目錄與最新狀態，供 Web API 快速查詢。|

> **總結：** 讓 **SQLite** 處理單台設備的即時反應，讓 **DynamoDB** 扛下十萬支 Watch 的全局併發查詢。

## 3. 系統是如何運作的 (End-to-End Flow)

這套系統的威力在於「資料分流」**。我們以一個具體的場景為例：**「設備對一支 Watch 進行拍攝與分析，然後主管在雲端查看整體數據的分佈」。

1. **資料產生 (Capture & Analyze)：**
    
    - 硬體拍攝了 Watch 的影像，並產生了 10,000 個測量數據點。
        
    - App 將影像暫存在本地，並將 Metadata 與 10,000 個數據點快速寫入 **Local SQLite**。硬體立即進入下一次測試，零延遲。
        
2. **背景同步 (Sync Outbox)：**
    
    - Outbox 偵測到新資料，開始工作：
        
        - **大檔案分流：** 將影像上傳到 **S3 Artifact Storage**，並取得一組 S3 URL。
            
        - **目錄分流：** 將「這支 Watch 剛完成一次測試」的狀態、時間戳記、影像 URL 上傳到 **DynamoDB** 更新狀態。
            
        - **分析數據分流：** 將這 10,000 個測量數據點打包，轉成 **Parquet** 格式，丟入資料湖 (Data Lake) 中。
            
3. **日常查詢 (Operational API)：**
    
    - 當工程師打開網頁想看「清單上有哪些 Watch 剛測試完？」，**API** 會向 **DynamoDB** 查詢。因為 DynamoDB 只有輕量的 Metadata，即使有十萬支 Watch，網頁載入時間也只有幾十毫秒。
        
4. **巨量分析 (Analytics & Distribution)：**
    
    - 當資料科學家想知道「這 10 萬支 Watch、數億個數據點的分佈 (Distribution) 狀況」時。
        
    - 系統**不會**去查詢 DynamoDB，而是呼叫 **Athena**。Athena 會平行處理 S3 裡的 **Parquet** 檔案，在一兩秒內算出統計學上的均值、標準差與分佈曲線，並回傳給 **Analysis** 前端。
        
    - **關鍵好處：** 這種超級沈重的分析運算，完全不會影響到線上設備同步資料與 API 網頁的流暢度（這就是 OLTP 與 OLAP 分離的價值）。



#### 舉例解釋SQLite vs. DynamoDB的差別
```
能否舉幾個簡單的例子解釋SQLite vs. DynamoDB的差別, 並以例子解釋核心架構, 資料關聯性, 併發能力, 在此架構的定位以及其他跟我們系統有關的價
```
我們可以把 **SQLite** 想像成「**工廠檢測站工程師隨身攜帶的電子筆記本**」，而 **DynamoDB** 則是「**跨國企業總部的超大型自動化檔案中心**」。

以下我用具體、生活化的例子，為你對比這兩者在不同維度上的差異，以及它們為何在你的系統中必須這樣搭配。

### 1. 核心架構 (Core Architecture)

- **SQLite (單機檔案型):**
    
    - **例子：** 你的筆記型電腦裡有一個名為 `watchDB.db` 的 Excel 檔案。所有的資料都在這「一個檔案」裡。
        
    - **特點：** 超級輕便，不需要架設伺服器，隨開隨用。但如果這台電腦壞了，或者你想把這個檔案分享給遠在美國的同事，你必須把整個檔案傳過去。
        
- **DynamoDB (全託管分散式):**
    
    - **例子：** 像是 Google 搜尋引擎背後的超級資料中心。資料被自動切碎、複製並分散儲存在幾十台甚至幾百台伺服器上。
        
    - **特點：** 你看不到任何「實體檔案」。當你存入第十萬支 Watch 的資料時，AWS 雲端會自動增加伺服器來容納，容量與效能是「無限延伸」的。
        

### 2. 併發能力 (Concurrency) ── 這是單一資料庫最容易崩潰的地方

- **SQLite (排隊機制 / File Locks):**
    
    - **例子：** 想像一間只有「一把鑰匙」的廁所。雖然大家上廁所都很快（寫入速度快），但如果 10 個檢測儀器在「同一個毫秒」都要把測試數據寫入 `watchDB.db`，系統會把檔案鎖住。第 1 個寫入時，剩下 9 個必須排隊等待。如果資料量大，排隊太久就會引發 `Timeout` (超時錯誤)。
        
- **DynamoDB (極高併發):**
    
    - **例子：** 想像一個有 10,000 個收費閘道的高速公路收費站。全台各地的設備（幾千台儀器）同時把 Watch 的狀態回傳給總部，DynamoDB 可以瞬間讓大家同時通過，完全不需要排隊，互不干擾。
        

### 3. 資料關聯性 (Data Relationships) ── 查詢方式的不同

- **SQLite (關聯式 SQL):**
    
    - **例子：** 像是精密的試算表。你有一個「手錶表單 (Watches)」和一個「測試人員表單 (Users)」。你可以下一個指令（JOIN 查詢）：「_幫我找出『張三』在『昨天』測試的所有『Series 5 手錶』_」。SQLite 擅長做這種跨表格的複雜關聯比對。
        
- **DynamoDB (NoSQL 鍵值對):**
    
    - **例子：** 像是一本「字典」。你只要輸入 Watch 的身分證號碼（UUID），它會在一毫秒內把這支手錶的所有狀態（目錄）整包丟給你。
        
    - **特點：** 它**不擅長**做複雜的跨表 JOIN。所以我們在架構中，只把它當作「營運目錄 (Operational Catalog)」，用最簡單直接的方式（Key-Value）來取得特定 Watch 的最新狀態。
        

### 4. 在此系統架構中的精確定位

回到你的需求：「有上千個 watch 系列，十萬支 watch，數千萬 images，數億筆 values」。

- **Local SQLite 的定位：** **「前線的防護罩與快取」**
    
    - **情境：** 設備正在進行精密拍攝，如果此時網路斷線，設備不能卡死。所以設備只管把當下的 1 萬筆測試結果「啪」一下寫入本地的 SQLite（因為是本機，保證低於 1 毫秒）。SQLite 裡面可能只存這台設備「這三天」測過的幾百支 Watch，負載極輕，永遠不會變慢。
        
- **DynamoDB 的定位：** **「總部的戰情看板 (Catalog)」**
    
    - **情境：** 主管在網頁端想要查詢 UUID 為 `W-99999` 的手錶狀態。因為十萬支 Watch 的資料都在 DynamoDB，無論系統裡有 100 支還是 10 萬支手錶，DynamoDB 抓取單一支手錶的時間永遠是**平穩的個位數毫秒**。
        

### 5. 其他跟你們系統有關的巨大價值 (擴展與備份)

如果把十萬支 Watch 和數億筆數據都硬塞進一個雲端 SQLite (或單一 MySQL)：

1. **資料庫過度肥大 (The Fat DB Problem)：** 一個 `.db` 檔案可能膨脹到 50GB 甚至 100GB。每次你想備份，就得複製一個 100GB 的檔案，不但極慢，還可能在備份時影響效能。
    
    - _架構解法：_ 在這套 Moonlight 架構中，最佔空間的 Image 全丟給 S3，最龐大的數億筆 Values 丟給 Parquet/Athena。DynamoDB 裡只存「文字標籤和網址 (Metadata)」。你的 DynamoDB 可能只有幾百 MB 到幾 GB，輕巧無比。
        
2. **效能衰退 (Performance Degradation)：** 傳統資料庫資料達到千萬級別後，如果沒有精準設定 Index (索引)，查詢速度會從 0.1 秒掉到 5 秒甚至 30 秒。
    
    - _架構解法：_ DynamoDB 的特性是「效能與資料量無關」。查詢第 1 筆資料和查詢第 1 億筆資料的速度是完全一樣的。而數億筆數據的分析，交由專門做大數據掃描的 Athena 處理，這確保了系統未來成長到百萬支 Watch 時，網頁依然秒開。
        

**總結來說：** 我們利用 SQLite 在邊緣端提供「無視網路狀態的極速寫入」**；利用 DynamoDB 在雲端提供**「十萬支手錶等級的高併發全局目錄」；最後把真正巨大且沉重的檔案與數據交給 S3 和 Athena。這就是為何這套系統能順暢運行的根本原因。







```
以下是構想的下一代Moonlight system的的database架構圖, 根據我的需求: "所以新的database不但要能夠執行以上提到的查詢跟分析, 而且我們可能有上千個watch系列, 十萬了watch. 跟你你設計的新database是否能否順暢的運行support這些功能性"  
可以順暢支援，但必須採用「分層式 database architecture」。如果把十萬支 watch、數千萬 image assets 和數億筆 analysis values 全塞在單一 `watchDB.db`，最後一定會遇到檔案過大、寫入鎖、備份緩慢及 distribution 查詢過慢。

請幫我深入分析根據我目前的架構比較這下一代的架構, 以及各有甚麼優缺點? 以及是否建議用下一代Moonlight system的的database架構圖取代現在的database? 以及如果現在在local機台的SQLite是否需要更新,

詳細要升新的步驟是甚麼? 以及如果這一兩天就要開始大量create template, 掃描watchentry, 是否來得及更新整個database包括AWS? 如果來不及有沒有優先要做的變動?

flowchart LR  
App["App / CLI / Hardware"] --> Local["每台設備 Local SQLite V4"]  
Local --> Outbox["Transactional Sync Outbox"]  
Outbox --> Catalog["DynamoDB V4 Operational Catalog"]  
Outbox --> S3["S3 Artifact Storage"]  
Outbox --> Facts["Parquet / Iceberg Metric Facts"]  
Facts --> Athena["Athena Analytics"]  
Catalog --> API["UUID / Watch / Capture / Experiment API"]  
Athena --> Analysis["Distribution / Comparison / Statistics"]
```

## 結論

我建議採用下一代 Moonlight 分層架構作為目標架構，但**不建議一次性取代現有 database**。正確做法是：

1. 保留 SQLite，將它升級為每台設備的 edge operational database／durable outbox。
2. 建立全新的 AWS V4 資源，與現有 AWS V2/V3 並行。
3. 新舊 schema 雙寫、核對，最後逐步切換讀取。
4. 數億筆 metrics 只進 Parquet/Iceberg，不進 SQLite 或 DynamoDB。

完整、production-grade 的本機加 AWS 遷移，**1–2 天內來不及安全完成**。但來得及完成一組 P0 保護措施，讓你先大量建立 template，並在有限條件下開始掃描，同時保證資料未來可回填 V4。

---

## 一、目前架構其實已經有部分 V4 能力

目前並不是完全的「一個 point 一張圖」：

- 已有 `watch_runs → point_instances → capture_instances → image_assets`。
- `image_assets` 有 `capture_id`、`asset_role`、`asset_index`、`internalnum1/2`。
- HDR 可產生多張 `raw_bracket` 加一張 `hdr_fused`。
- `watchentry` 已有 `PointImageResult.captures`，每個 capture 支援多張 image 及 optional HDR。[watchentry_structure.py (line 24)](D:/Provenance Project/ImagingLibWatch/DB/templates/watchentry_structure.py:24)
- 舊 template 仍可轉成新結構。
- 我執行了三個針對多圖、HDR、舊 template 相容性的測試，全部通過。

所以真正要做的不是重新發明 multi-image model，而是把已經存在的 V4 雛形變成唯一可靠的資料路徑。

---

## 二、目前 database 的實際問題

### 1. SQLite 已經明顯膨脹

目前 `watchDB.db`：

- 約 359 MB。
- 7,277 筆 `raw_images`。
- 9,944 筆 `experiment_results`。
- 其中 `experiment_results.data` JSON 合計約 279 MB。
- 單筆最大 JSON 約 500 KB。
- 約 139–146 個 distinct watch IDs。

依目前樣本線性估算，十萬支 watch 會產生約 **200–260 GB 的 SQLite DB**，還不包含影像本身。這不是 SQLite 檔案大小的硬限制問題，而是：

- 備份、restore、migration 太慢。
- JSON distribution query 必須逐列 parse。
- index 和 VACUUM 成本增加。
- 多執行緒 capture、analysis、sync 會出現寫入競爭。

主要問題是把完整 analysis payload 重複存進 SQLite，不是影像索引本身。

### 2. 新舊 tables 雙寫，但新 table 尚未真正成為主資料源

目前 `register_image_asset()` 同時寫：

- `image_assets`
- `raw_images`

`register_analysis_result_v2()` 同時寫：

- `analysis_results_v2`
- `experiment_results`

但實際 DB 中：

- `image_assets`：3,521 rows
- `analysis_results_v2`：0 rows
- `experiment_results`：9,944 rows

表示 analysis 還是大量走 legacy path。[db_manager.py (line 144)](D:/Provenance Project/ImagingLibWatch/DB/db_manager.py:144)

### 3. 現在的 sync queue 不是 transactional outbox

目前只用 `synced=0/1`：

- 沒有 `event_id`。
- 沒有每個 destination 的獨立狀態。
- 沒有 retry count、next retry time、last error、dead-letter。
- S3、DynamoDB、Facts 無法分別追蹤。
- `get_pending_uploads()` 仍然讀 legacy tables。[db_manager.py (line 412)](D:/Provenance Project/ImagingLibWatch/DB/db_manager.py:412)

更嚴重的是：

- S3 key 空白時會直接標記 synced。
- local file 遺失時，程式目前仍可能將這筆視為成功，甚至建立 DynamoDB index。

這會造成「DB 顯示已同步，但 S3 根本沒有 object」的永久資料洞。[cloud_sync.py (line 187)](D:/Provenance Project/ImagingLibWatch/data_manager/cloud_sync.py:187)

Transactional outbox 必須把 domain row 與 outbox event 放在同一個 SQLite transaction，消費端則必須允許重送並具備冪等性；這也符合 [AWS transactional outbox guidance](https://docs.aws.amazon.com/prescriptive-guidance/latest/cloud-design-patterns/transactional-outbox.html)。

### 4. Run/capture 狀態沒有正常結束

目前資料中：

- 1,378 個 `watch_runs` 全部是 `running`。
- 3,521 個 `capture_instances` 全部是 `processing`。

程式有 start methods，卻沒有一致使用的 complete/failed/cancelled transition。[db_manager.py (line 114)](D:/Provenance Project/ImagingLibWatch/DB/db_manager.py:114)

因此以下查詢目前會失真：

- 成功率。
- 失敗 capture。
- 未完成 run。
- 可安全重跑的 capture。
- 每支 watch 最新的完整 run。

### 5. SQLite index 不足

現在主要只有四個自建 indexes。實際 query plan 顯示這些查詢會 full scan 或建立 temporary sort：

- `raw_images WHERE synced=0`
- `experiment_results WHERE synced=0`
- analysis results by watch/task/time
- runs by watch/time
- analysis by asset

此外：

- `journal_mode=DELETE`
- `foreign_keys=OFF`
- 沒有 schema migration version table

對單機少量資料尚可，但不是長期大量掃描的理想設定。

### 6. DynamoDB 現行 query model 有缺口

目前 DynamoDB `WatchAnalysisResults` 混合：

- asset index rows
- `INFO`
- `VIEW#...`
- `MAT#...`
- `AUTH`

主要問題：

- asset 的 sort key 是隨機 UUID；`query_watch_history(... ScanIndexForward=False)` 並不會依時間倒序，只是依 UUID 排序。[cloud_db.py (line 127)](D:/Provenance Project/ImagingLibWatch/data_manager/cloud_db.py:127)
- 程式查詢 `BrandModelIndex`，但 repository 中沒有任何實際寫入 `GSI1_PK/GSI1_SK` 的程式碼。
- 一千個以上 templates 時，`scan_all_templates()` 會下載整張 template table，而不是先讀 summary 或 delta。[cloud_db.py (line 305)](D:/Provenance Project/ImagingLibWatch/data_manager/cloud_db.py:305)
- DynamoDB 單 item 上限仍是 400 KB，因此完整 watchentry、template 或 analysis JSON 不適合持續膨脹。[AWS DynamoDB constraints](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Constraints.html)

### 7. Athena 現在只是 query wrapper

目前 `AthenaQueryEngine` 只負責送出任意 SQL：

- 沒有 metric facts exporter。
- 沒有 Parquet writer。
- 沒有 Iceberg schema／table provisioning。
- 沒有 Glue/Iceberg maintenance。
- 沒有 compaction。
- 沒有明確 partition strategy。

所以「Athena 已 enabled」不等於已經有可支援 distribution analysis 的 data lake。

---

## 三、現行與下一代架構比較

|面向|現行架構|下一代分層架構|
|---|---|---|
|離線操作|很好，SQLite 可獨立工作|保留相同優點|
|實作複雜度|低至中|高，需要資料契約、IaC、監控|
|多圖/HDR|已部分支援，但新舊路徑並存|原生 point → captures → assets|
|同步可靠性|單一 `synced` flag，容易誤標成功|Transactional outbox、冪等重送、DLQ|
|十萬 watch catalog|本機全塞不可取|DynamoDB 適合 operational lookup|
|數億 metrics|JSON row 不適合|Parquet/Iceberg 適合|
|Distribution/比較|需讀大量 JSON|Athena column pruning、aggregation|
|備份與 restore|DB 越來越大|本機小型 DB；artifact/facts 分別管理|
|一致性|多處直接雙寫|明確 eventual consistency 與狀態|
|維運成本|初期低、後期技術債高|AWS 成本及維運較高，但可控|
|Cutover 風險|已運作|Big-bang 高風險，side-by-side 可控|

---

## 四、建議修正後的下一代架構

原圖的主要修改是：**設備端不要直接 commit Iceberg**。

````
```mermaid
flowchart LR
    App["App / CLI / Hardware"] --> Local["每台設備 Local SQLite V4"]
    Local -->|"同一個 transaction"| Outbox["Transactional Outbox"]

    Outbox --> Artifact["S3 Artifact Storage"]
    Outbox --> Landing["S3 Event Landing / SQS"]

    Landing --> Projector["Idempotent Cloud Projectors"]
    Projector --> Catalog["DynamoDB V4 Operational Catalog"]
    Projector --> FactWriter["Batch Fact Writer / Compaction"]
    FactWriter --> Facts["Parquet / Iceberg Metric Facts"]

    Catalog --> API["Watch / Run / Capture / Asset API"]
    Artifact --> API
    Facts --> Athena["Athena Analytics"]
    Athena --> Analysis["Distribution / Comparison / Statistics"]
```
````

原因是 Athena 官方也明確指出，大量小型 Parquet files 會降低效能；應批次合併成較大的 columnar files，並按常用查詢條件合理 partition。[Athena data optimization](https://docs.aws.amazon.com/athena/latest/ug/performance-tuning-data-optimization-techniques.html)

Iceberg 還需要定期 `OPTIMIZE` compaction 及 `VACUUM` snapshot/orphan cleanup。[Athena Iceberg optimization](https://docs.aws.amazon.com/athena/latest/ug/querying-iceberg-data-optimization.html)

---

## 五、哪些查詢應該去哪一層

|查詢|正確資料層|
|---|---|
|用 UUID／serial／reference 找 watch|DynamoDB GSI|
|取得某 watch 最新 run|DynamoDB，time-based sort key|
|顯示 run、point、capture、image gallery|DynamoDB + S3|
|找 HDR source brackets 與 fused image|DynamoDB asset lineage|
|取得 template 最新版本|Local cache + Templates DynamoDB|
|某系列某 metric 的 distribution|Athena/Iceberg|
|不同系列／reference／年份比較|Athena/Iceberg|
|平均、標準差、percentile、outlier|Athena/Iceberg|
|單支 watch 產生 report|DynamoDB manifest + S3 report|
|本機等待／失敗同步項目|SQLite outbox|
|即時 sub-second distribution UI|預先聚合 summary/cache；不要每次掃 Athena|

十萬 watch 對 DynamoDB 不是問題，因為 `WatchID/watch_uuid` 可自然分散 partition。真正不能放進 DynamoDB 的是數億個 analysis values；這些應維持 narrow fact schema。

---

## 六、Local SQLite 是否需要更新

**需要，而且應該比 AWS 遷移更早做。** 但不必取消 SQLite，也不需要每支 watch 一個 DB。

建議 V4 local schema 至少有：

- `schema_migrations`
- `watches`
- `watch_runs`
- `point_instances`
- `capture_instances`
- `image_assets`
- `analysis_runs`
- `metric_summaries`：只存少量本機 UI 必要 metrics
- `sync_outbox`
- `sync_deliveries`

資料責任應明確分離：

- Point metadata：座標、component、description、`internalnum1`
- Capture metadata：曝光計畫、capture type、`internalnum2`
- Image metadata：checksum、mime、尺寸、role、index、S3 key
- Analysis metadata：algorithm/model/config version
- Metric facts：獨立 long-form rows，最終寫入 Iceberg
- 完整 analysis report：local file/S3 artifact，不再塞進 SQLite JSON

應加入的主要 indexes：

- `raw_images(synced, created_at)`
- `experiment_results(synced, created_at)`
- `watch_runs(watchid, started_at DESC)`
- `capture_instances(run_id, point_instance_id, capture_id)`
- `image_assets(watchid, view_name, point_name, capture_id, created_at DESC)`
- `analysis_results_v2(asset_id, algorithm_name, created_at DESC)`
- `sync_outbox(status, next_attempt_at, created_at)`

在確認 DB 位於本機磁碟且完成 backup 後，可評估：

- `journal_mode=WAL`
- `foreign_keys=ON`
- `busy_timeout`
- 一致的 transaction boundaries
- 定期 WAL checkpoint

---

## 七、完整升級步驟

### Phase 0：定義資料契約

先固定 immutable IDs：

- `series_uuid`
- `template_uuid` + `template_version`
- `watch_uuid`
- `run_id`
- `point_instance_id`
- `capture_instance_id`
- `asset_id`
- `analysis_run_id`
- `event_id`

現有 `internalnum1/2` 應保留作為穩定 business identity，但不要用它取代 UUID。

同時列出正式 access patterns 和 SLO，例如：

- operational lookup < 300 ms
- gallery load < 1 s
- Athena distribution query < 10–30 s
- outbox eventual sync < 5 min

### Phase 1：備份與 additive SQLite migration

- 使用 SQLite online backup 建立可驗證 snapshot。
- 匯出所有 template YAML 並記錄 checksum。
- 建立 `schema_migrations`。
- 新增 V4 tables，不刪除舊 tables。
- 舊資料保持可讀。
- 新 writer 在同一 transaction 寫 canonical row 和 outbox event。

### Phase 2：修正 capture lifecycle

每層都要有：

- `pending`
- `running`
- `completed`
- `failed`
- `cancelled`
- `completed_with_warnings`

並增加：

- `completed_at`
- `error_code`
- `error_message`
- `attempt`
- `quality_status`

### Phase 3：真正的 transactional outbox

Outbox 至少需要：

- `event_id`
- `event_type`
- `aggregate_type`
- `aggregate_id`
- `payload_json`
- `created_at`
- `status`
- `attempts`
- `next_attempt_at`
- `last_error`

S3、Dynamo、Facts 必須有獨立 delivery status。相同 `event_id` 重送不得建立重複 asset/fact。

### Phase 4：建立 AWS V4 side-by-side

不要原地修改 production tables，建立：

- `moonlight-v4-catalog`
- `moonlight-v4-templates`
- 新 S3 V4 prefixes/bucket
- Glue catalog
- Iceberg metric tables
- alarms、PITR、lifecycle、IAM roles

DynamoDB sort key 應包含時間，例如：

```
PK = WATCH#<watch_uuid>
SK = RUN#<captured_at>#<run_id>
SK = ASSET#<captured_at>#<asset_id>
```

另建 series/reference/serial/date 所需 GSIs。

### Phase 5：Metric fact pipeline

建議 narrow schema：

```
event_id
watch_uuid
series_uuid
run_id
internalnum1
internalnum2
capture_instance_id
asset_id
analysis_run_id
algorithm_name
algorithm_version
metric_name
metric_index
value_double
value_text
unit
captured_at
station_id
template_version
quality_status
```

不要按 `watch_uuid` partition，否則會產生十萬個以上的小 partitions。通常按：

- capture date/month
- metric family 或 algorithm family
- deployment/tenant（若需要）

並在 partition 內保留 series/watch 作 column/filter。

### Phase 6：Shadow write、backfill、reconciliation

- 所有新 capture 同時寫 legacy 和 V4。
- 從現有 SQLite/S3 reports 抽取 metric facts。
- 使用 S3 Inventory 和 Dynamo export 對帳。
- 比較每個 run 的 point/capture/asset 數量。
- 比較 checksum、S3 existence、Dynamo status。
- 任一 destination 不一致不得標記 global complete。

### Phase 7：逐讀路徑 cutover

順序建議：

1. 新 template reads
2. gallery/asset reads
3. watch operational API
4. report reads
5. distribution/analytics
6. 停止 legacy writes
7. 舊 DB 轉 read-only archive

---

## 八、未來 1–2 天能做甚麼

### 可以安全完成的 P0

1. **立即旋轉 AWS credentials。**  
    [system_config.yaml (line 55)](D:/Provenance Project/ImagingLibWatch/config/system_config.yaml:55) 目前包含 plaintext AWS access credentials。應立刻停用、rotate，改用環境變數、AWS profile 或 device IAM mechanism；若曾提交 Git，還要視為已外洩處理。
    
2. 備份 `watchDB.db`，執行 integrity check，匯出所有 template YAML。
    
3. 大量 template 建立時：
    
    - 統一 `schema_version=3.1`
    - `internalnum_schema_version=4`
    - 使用 `standard_captures[]` / `hdr_captures[]`
    - 保持 `internalnum1/2` 不變
    - 每次修改建立新 version，不覆寫舊 version
    - 每份 YAML 保存 checksum 和建立時間
4. 將 template 保存改為 local-first + outbox。  
    現在 `TemplateManager.save_template()` 是 cloud 成功後才存 local；AWS 暫時故障會讓新 template 無法可靠保存。[template_manager.py (line 309)](D:/Provenance Project/ImagingLibWatch/DB/templates/template_manager.py:309)
    
5. 修正兩個資料遺失風險：
    
    - empty S3 key 不得標記 synced
    - missing local file 不得視為成功
6. 增加必要 SQLite indexes 和 run/capture completion updates。
    
7. 所有掃描保留：
    
    - raw assets
    - HDR source brackets
    - fused image
    - 完整 analysis report
    - algorithm/config/template versions
    - checksum

只要完整 artifacts 沒有遺失，metric facts 可以在 AWS V4 完成後再 backfill。

### 不建議 1–2 天內做的事

- 直接改掉現有 DynamoDB production table keys。
- 一次搬完全部 S3 keys。
- 讓設備直接寫 Iceberg。
- 刪除 legacy tables。
- 在未 load-test 前切換 report/gallery 全部讀 V4。
- 邊大量掃描邊執行大型 `VACUUM` 或 destructive migration。

---

## 九、是否可以現在開始大量建立 template／掃描

### Template：可以開始，但先完成保存保護

現有 schema 已能表達 multi-image/HDR，舊 template 也能相容。只要：

- append-only versioning
- local backup
- YAML export
- stable internal IDs
- 不依賴 cloud-first save

大量 template 建立不必等待完整 AWS V4。

### Watch scanning：有條件可以

至少先完成：

- DB snapshot
- sync false-success 修正
- run/capture status 修正
- stable IDs
- 保留完整 raw/report artifacts
- 確認本機磁碟容量與 pending sync queue

如果這些 P0 尚未完成，我不建議直接進入不可重做的大量 production scanning。

---

## 最終建議

- **架構方向：採用下一代分層架構。**
- **遷移方式：side-by-side、additive、shadow write；不要 big-bang replacement。**
- **SQLite：保留，但改成小型 edge DB/outbox，必須立即升級。**
- **DynamoDB：只做 operational catalog，不存數億 metrics。**
- **S3：保存不可取代的 images/reports。**
- **Iceberg/Athena：承擔 distribution、comparison、statistics。**
- **1–2 天：先保護資料與 template identity；完整 AWS V4 遷移來不及。**

本次只做了分析與驗證，沒有修改 repository 檔案。SQLite `integrity_check` 通過，三個 multi-image/HDR/legacy compatibility 測試通過；尚未實際讀取你 AWS account 內已部署 tables、GSIs、Glue catalog 和 S3 inventory，因此正式 migration 前仍須先完成一次 read-only AWS inventory。












#### 文詳細解釋 DB/watchDB.db的內部資料結構
```
能否中文詳細解釋 DB/watchDB.db的內部資料結構, 是否裡面有很多template, 很多watchentry以及其他等等, 我以為watchDB.db是SQLite database, 不是應該類似excel一樣有row, column..
```
我已經確認了 `DB/watchDB.db` 檔案本身確實是 SQLite 3 格式(用 `file` 指令驗證過:`SQLite 3.x database ... database pages 87740`),檔案大小 359MB,並且對照了建表的原始碼 `DB/db_manager.py` 逐一確認了裡面的資料表結構。以下詳細說明。

## 為什麼「看起來不像 Excel」

你的直覺沒錯——SQLite **確實**有 row(列)/column(欄)的結構,但它跟 Excel 不同的地方是:

- Excel 檔案通常只有一張或幾張「表」(sheet),你打開就整個看到。
- SQLite 資料庫檔案裡可以包含**多達十幾個獨立的表(table)**,每個表有自己的欄位定義,彼此用外鍵/ID關聯起來(這叫「關聯式資料庫」)。
- 如果你直接用文字編輯器打開 `.db` 檔案,看到的是二進位格式,當然看不出 row/column——要用 **DB Browser for SQLite** 這類 GUI 工具,或指令列 `sqlite3 DB/watchDB.db` 才能看到熟悉的表格畫面。

所以正確理解是:`watchDB.db` = 一個容器,裡面裝了 **13 張表**,每張表才是「像 Excel 一樣有 row/column」的東西。

## 實際的表結構(依 `_init_db()` 逐一列出)

**舊版相容表(Legacy)**

|表名|用途|
|---|---|
|`raw_images`|每張拍攝的原始照片一筆記錄(watchid, step_name, local_path, s3_key, metadata JSON, synced)|
|`experiment_results`|每個分析結果一筆記錄(watchid, task_name, result_type, data JSON)|

**核心設定/快取表**

|表名|用途|
|---|---|
|`watch_registry`|極簡的 watchid 首次/最後出現時間|
|`users`|本地帳號快取(username, password_hash, salt, role)|
|**`local_templates`**|**你問的 template 就在這裡** — Primary Key 是 `(TemplateID, version)`,`data` 欄位是整包 template 的 JSON 字串|
|`watch_metadata`|每支錶一筆(WatchID 為主鍵),含 Brand/Model/Reference/Version + `Full_JSON`(完整快照)—用來做搜尋索引|
|`ui_asset_registry`|UI 顯示用的路徑對照表|

**V3 資產導向表(新架構,拆得更細)**

|表名|用途|
|---|---|
|`watch_runs`|一次拍攝流程(routine)的執行記錄|
|`point_instances`|流程中每個拍攝點(view+point)的執行實例|
|`capture_instances`|每個點底下每次拍照(capture)的實例|
|`image_assets`|**每一張照片一筆**(取代 raw_images 的正規化新版)|
|`analysis_results_v2`|**每一個分析結果一筆**(取代 experiment_results 的正規化新版)|

## 針對你的問題:「是否有很多 template、很多 watchentry」

- **Template:確實存在,且很多** — 全部存在 `local_templates` 表裡,一列 = 一個 (TemplateID, version) 組合,內容是 JSON 字串(不是攤平成很多欄位,而是整包塞進一個 `data` TEXT 欄位)。這是典型的「關聯式表 + JSON blob 欄位」混合設計。
- **`watchentry`(手錶檢測執行記錄):並沒有一張叫 `watchentries` 的表** — 我查了 `core/workflow_manager.py` 和 `App/main.py:32625`,`watchentry` 這個 Python dataclass(定義在 `DB/templates/watchentry_structure.py`)的完整內容主要是寫到**雲端 DynamoDB**,SQLite 這邊只做兩件事:
    1. `watch_metadata` 表存一份**扁平化快取**(Brand/Model/Reference + Full_JSON 快照),方便離線快速搜尋;
    2. 過程中產生的**照片與分析結果**(不是整個 entry)則逐筆寫進 `image_assets` / `analysis_results_v2`(以及舊版 `raw_images` / `experiment_results` 做雙寫相容)。

也就是說,watchDB.db 的角色是:「**本地寫入緩衝區 + 雲端 DynamoDB 的鏡射快取**」,不是唯一真實來源(source of truth)——這點在文件 `helper/docs/14_db/14.3_watchDB_db.md` 裡也有明確說明,而且它被 `.gitignore` 排除,是每台工作站各自產生的執行期資料。

## 如果你想親眼看實際的 row 數量

我這台環境沒有現成的 `python`/`sqlite3` 指令列可以直接查詢實際筆數(只確認了 schema 是否吻合)。你可以自己執行,例如:

sqlite3 "DB/watchDB.db" ".tables"

或者更方便的是下載 **DB Browser for SQLite**(免費 GUI 工具),直接打開 `watchDB.db`,點「Browse Data」分頁,選任一張表,就會看到你熟悉的 row/column 表格畫面了。需要的話我可以幫你寫一個小腳本把每張表的筆數和大小統計出來。

Background shell completedSearch for a real python executable outside the store alias

This is just a background task finishing (the earlier `find` search for a Python executable) — nothing requiring action, and my previous answer already covered the question without needing it. No further action needed here.




#### DB/watchDB.db是如何跟AWS同步
```
那請中文詳細解釋DB/watchDB.db是如何跟AWS同步, 而且watchDB.db是SQLite, 而AWS上的應該是有四個DynamoDB(目前有moonlight-WatchAnalysisResults, moonlight-WatchCommandLog, moonlight-WatchTemplates, moonlight-WatchUsers), 他們之間是怎麼互動? 怎麼由local的SQLite database跟AWS 不同型態database DynamoDB互動
```
我對照了實際設定檔 `config/system_config.yaml:57-70` 確認你講的四張表名稱完全正確,以及原始碼 `data_manager/cloud_db.py`、`data_manager/cloud_sync.py`、`core/watch_record_manager.py` 的實際邏輯。以下詳細說明。

## 核心觀念:SQLite 和 DynamoDB 「不是互相同步的兩份一樣的資料庫」

這是最關鍵的一點——這套系統**沒有底層自動的表對表複製機制**(不是 SQLite table ↔ DynamoDB table 逐欄同步)。而是**應用層(Python 程式碼)在每個關鍵操作點,手動把同一份資料分別轉換成兩邊各自適合的格式**去寫入:

- SQLite 扮演的角色:**本地寫入緩衝區(write buffer)+ 離線快取(offline cache)**
- DynamoDB 扮演的角色:**雲端的「可查詢索引」或「主要真實來源(master)」**,依表而異

兩者資料模型完全不同,靠中間的 Python dict/JSON 物件當「轉接介面」,分別呼叫 `sqlite3` 的 INSERT 和 `boto3` 的 `put_item` API。

---

## 四張 DynamoDB 表,對應到系統裡的三個「平面(plane)」

|DynamoDB 表|Partition Key / Sort Key|平面|跟本地 SQLite 的關係|
|---|---|---|---|
|`moonlight-WatchAnalysisResults`|`WatchID` / `sort_key`|資料平面(Data)|**單向推送**:本地 `image_assets`/`raw_images`/`analysis_results_v2` 等表 → 雲端(見下方流程①②)|
|`moonlight-WatchCommandLog`|`DeviceID` / `CommandID`|控制平面(Control)|**幾乎無關**:這是 AWS IoT 遠端遙控指令的日誌,直接由雲端讀寫,不經過本地 SQLite|
|`moonlight-WatchTemplates`|`TemplateID` / `version`|設定平面(Config)|**雲端為主、本地為快取**:對應本地 `local_templates` 表(見下方流程③)|
|`moonlight-WatchUsers`|`username`|帳號|**雲端為主、本地為快取**:對應本地 `users` 表(見下方流程④)|

---
## 流程① 拍照/分析結果同步 —— 「Async Push」模式

這是最主要、最常跑的同步路徑,對應 `WatchAnalysisResults` 表:
```
本地拍照/分析完成
  → 寫入 SQLite (image_assets / raw_images / analysis_results_v2 表, synced=0)
        │  (這一步是同步、立即完成的,不等網路)
        ▼
CloudSyncManager 背景執行緒 (daemon thread)
  → 每 ~2 秒輪詢一次 SQLite:SELECT * WHERE synced=0
    (或呼叫 kick_sync_worker() 立刻喚醒,不用等 2 秒)
        ▼
對每一筆 pending 資料 (_sync_one_item):
  1) 把本地檔案上傳到 S3 (Raw/<uuid>.ext 或 Analysis/<uuid>.ext)
  2) 呼叫 CloudDatabaseManager.index_record()
        ▼
index_record() 把 metadata「攤平(flatten)」成一個 DynamoDB item:
  {
    "WatchID": watchid,          ← Partition Key
    "sort_key": asset_id,        ← Sort Key (用 UUID)
    "s3_key": ...,
    "view_name": ..., "point_name": ..., "capture_id": ...,
    "internalnum1": ..., "internalnum2": ...,
    "metadata_raw": {...}        ← 完整 JSON 也保留一份
  }
        ▼
S3 上傳 + DynamoDB 寫入都成功
  → 才把 SQLite 那一列 UPDATE synced=1
```
重點:**SQLite 這邊的一列,對應 DynamoDB 那邊的一個 item**——但欄位名稱、欄位數量完全不同,是靠 `index_record()` 這個 Python 函式手動做「欄位對映」轉換的,不是資料庫層級的自動同步。而且這條路徑目前**沒有上傳失敗次數上限**(unbounded retry),卡住的資料會一直重試到成功為止。

---

## 流程② 完整 watchentry 記錄 —— 「拆分成多個 item」

一個 `watchentry`(一次完整檢測記錄,包含所有 view、材質、真偽鑑定結果)如果整包塞進一個 DynamoDB item,很容易超過 DynamoDB **單一 item 400KB 的硬限制**。所以 `core/watch_record_manager.py:103` 的 `save_full_record_to_cloud()` 把它拆開,寫進**同一張** `WatchAnalysisResults` 表,但用不同的 `sort_key` 前綴區分:
|sort_key|內容|
|---|---|
|`INFO`|Brand/Model/Reference 等根層級欄位|
|`VIEW#Front`、`VIEW#Back`...|每個 view 的檢測結果|
|`MAT#CaseBody`、`MAT#BoxLidFront`...|每個材質部位的量測結果|
|`AUTH`|真偽鑑定(authenticity)結果|

所以同一個 `WatchID` 底下,`WatchAnalysisResults` 表裡其實混雜著**兩種完全不同形狀的 item**:一堆單張照片的 flat index item(流程①),加上幾筆這個 entry 本體的 split-key item(流程②)。之後如果要重組回完整的 watchentry,就用 `_reconstruct_split_aws_entries` 依 `sort_key` 前綴反向拼回去。

這條路徑**不經過本地 SQLite 表**——`watchentry` 本體是直接被組成 DynamoDB item 送上雲端的,SQLite 這邊只留一份扁平化的搜尋快取在 `watch_metadata` 表(我們上次談過的)。

## 流程③ Template 同步 —— 雲端是「master」,本地只是 cache

跟①②方向相反,Template 是**雲端優先**的設計(`helper/docs/06_domain_model/6.16_template_loading_and_saving.md`):

儲存 (TemplateManager.save_template)
  1) 先寫 DynamoDB WatchTemplates 表 (PK=TemplateID, SK=version)
  2) 只有雲端寫入成功，才寫回本地 SQLite local_templates 表當快取
     （如果雲端寫失敗，本地也不會更新——"fail together" 設計，
       跟①②的"本地一定先成功、雲端是best-effort"完全相反）

讀取 (TemplateManager.get_template)
  1) 優先讀本地 SQLite local_templates（離線也能用）
  2) 找不到才去雲端拉

兩邊都各自有一份 `TEMPLATE_SYNC_BLOCKLIST`(拒絕把 `local_db_all_objects`、`local_watchentry_list` 這類匯出檔名當成真的 Template 存進去的防呆機制)——是**各自獨立實作**的相同邏輯,不是共用同一段程式碼,這點文件裡也特別點出是維護風險。

## 流程④ User 帳號同步 —— 整批覆寫快取

`WatchUsers` 表跟本地 `users` 表的關係最簡單:`update_local_user_cache()` 把雲端整份使用者清單抓下來,整批 `INSERT OR REPLACE` 覆寫進 SQLite,讓站點離線時也能用密碼雜湊比對登入(`get_user_offline`)。這是**單向、批次**的快取更新,不是逐筆同步。

---

## 流程⑤ Command Log —— 跟 SQLite 完全無關的獨立通道

`WatchCommandLog` 是遠端遙控(AWS IoT / MQTT)專用,`claim_command_once()` 用 DynamoDB 的 conditional update(`ConditionExpression`)做「這個指令只能被執行一次」的冪等判斷,直接雲端讀寫,**不透過本地 SQLite**,概念上跟前面幾條資料同步路徑完全獨立。

## 為什麼「型態不同」也能互動:關鍵在於資料轉換,不是資料庫互通

1. **沒有 ORM 或跨資料庫同步工具**——就是 Python 函式手動把同一份 dict 資料,分別組成 `INSERT INTO ... VALUES (?,?,?)` 和 `table.put_item(Item={...})` 兩種呼叫。
2. **型別轉換是手動處理的**:DynamoDB 不接受 Python 原生 `float`,所有數值寫入前都要跑 `_float_to_decimal()` 轉成 `Decimal`(且 `NaN`/`inf` 會被轉成 `None`,因為 DynamoDB 完全拒絕非有限數值)。SQLite 這邊反而簡單,直接把整包 JSON 字串塞進一個 `TEXT` 欄位就好。
3. **兩邊「正規化程度」不同**:SQLite 傾向把常查詢的欄位拆成獨立 column(如 `watch_metadata.Brand/Model`),但大 JSON 直接整包存一欄；DynamoDB 則刻意把 asset 索引「攤平」成多個 top-level attribute(方便 Athena/GSI 查詢),但完整原始資料仍保留一份在 `metadata_raw`。
4. **一致性模型不同且鬆散**:SQLite 寫入是同步、立即的;DynamoDB 是背景執行緒非同步輪詢推送過去的——中間會有幾秒到「不確定多久」的延遲(尤其是 S3 資料要等到 Athena/Glue catalog 抓到才能被 SQL 查到,這一段完全不受這套系統控制)。

---

## 值得注意的現況/風險(來自文件裡的既有發現)

- 上傳失敗**沒有重試次數上限**,卡住的項目會一直在背景重試,沒有 dead-letter 機制。
- `query_watches_by_gsi()`(依 Brand/Model 查詢)依賴一個名叫 `BrandModelIndex` 的 GSI,**必須手動在 AWS Console 建立**,程式本身不會自動建立這個索引。
- `system.offline_mode = true` 時,`CloudDatabaseManager` 完全跳過連線,所有寫入雲端的呼叫都變成 no-op——這時候資料就只留在本地 SQLite,直到離線模式關閉才會補上傳。

如果你想看某一條路徑(例如 template 同步或 watchentry 拆分/重組)更細部的程式碼片段,我可以再深入挖給你看。









#### DB browser for SQLite打開watchDB.db
```
我用DB browser for SQLite打開watchDB.db, 看到裡面有13個tables - analysis_results_v2, capture_instances, experiment_results, image_assets, local_templates, point_instances, raw_images, sqlite_sequence, ui_asset_registry, users, watch_metadata, watch_registry, watch_runs, 以及四個indices - Idx_asset_cap_inst, Idx_asset_cap_internalnums, Idx_path, Idx_watch_meta_brand_model這些各是甚麼? 以及有沒有哪個是現在沒在用的. 我選local_templates並選Database Structure看到裡面有TemplateID, version, data, update_at. 這些各是甚麼? 然後選Browse Data看到有很多筆TemplateID譬如run_16613T(V3), test_hdr1, test_Josh...等. 那local_templates是怎麼把資料傳給AWS的DynamoDB - moonlight-WatchTemplates? 在裡面的data structure是怎麼樣的? 能否以實際例子譬如當local_template新創了一個template, 之後怎麼存到moonlight-WatchTemplates
```
我對照了 `DB/db_manager.py`、`DB/templates/template_manager.py:309`、`data_manager/cloud_db.py:325`、`DB/templates/template_structure.py`、`DB/templates/internalnum_config.py:1515` 的實際程式碼,並且用 grep 逐一確認了每張表/每個索引在整個專案裡「有沒有被真的呼叫到」,而不是只憑建表語法猜測。以下分三部分回答。

---

## 一、13 個表 + 4 個索引逐一說明,以及哪些沒在用

|表名|用途|使用狀態(用 grep 實際驗證)|
|---|---|---|
|`raw_images`|每張照片一筆(舊版)|✅ **主力表**——`get_pending_uploads()` 就是從這裡 `WHERE synced=0` 撈出待上傳清單,是 CloudSyncManager 背景執行緒**實際輪詢的來源**|
|`experiment_results`|每個分析結果一筆(舊版)|✅ 同上,`get_pending_uploads()` 也從這裡撈,同樣是真正驅動上傳的表|
|`image_assets`|照片的「V3 正規化版」|✅ 有寫也有讀(`get_image_assets_for_point`、`get_preferred_raw_image_path`),但**不是**雲端上傳佇列的來源(下面細講)|
|`analysis_results_v2`|分析結果的「V3 正規化版」|⚠️ **只寫不讀**——全專案沒有任何地方對它下 `SELECT`,寫進去後目前沒人讀回來|
|`watch_runs`|一次拍攝流程(routine)的執行記錄|⚠️ **只寫不讀**——`start_watch_run()` 被 `core/workflow_manager.py`、`App/main.py` 呼叫寫入,但沒有任何查詢讀取它|
|`point_instances`|每個拍攝點的執行實例|⚠️ **只寫不讀**,同上|
|`capture_instances`|每次拍照的執行實例|⚠️ **只寫不讀**,同上|
|`local_templates`|Template 快取(你的問題主角)|✅ 大量讀寫|
|`watch_metadata`|Brand/Model/Reference 搜尋索引快取|⚠️ 只有 `upsert_watch_metadata()`(寫)被 `App/main.py:33065` 呼叫;對應的讀取函式 `search_watches_by_metadata()` **在全專案沒有任何呼叫者**——等於寫了但沒人查|
|`ui_asset_registry`|UI 顯示用的路徑對照表|✅ 大量讀寫|
|`users`|帳號本地快取|✅ 讀寫皆有(離線登入用)|
|`watch_registry`|錶的首次/最後出現時間|❌ **完全死表**——`register_watch()` 這個方法本身除了自己定義那行之外,**全專案沒有任何地方呼叫它**,連寫入都沒發生|
|`sqlite_sequence`|—|不是應用程式自己建的表,是 SQLite **內建系統表**——因為 `raw_images`/`experiment_results` 用了 `INTEGER PRIMARY KEY AUTOINCREMENT`,SQLite 會自動生出這張表來記錄各表目前的自動遞增計數器,DB Browser 才會列出來,不用管它|

**索引部分**(對照 `DB/db_manager.py:55,63,85,97`):

|索引|建在哪張表/欄位|有沒有被實際查詢用到|
|---|---|---|
|`idx_path`|`raw_images(local_path)`|✅ 有用——`get_image_record_by_path()` 用 `WHERE local_path=?` 精確比對會吃到這個索引|
|`idx_watch_meta_brand_model`|`watch_metadata(Brand, Model)`|❌ **目前沒用到**——因為它服務的 `search_watches_by_metadata()` 本身就是死代碼,沒人呼叫,索引也就空轉|
|`idx_asset_cap_inst`|`image_assets(capture_instance_id)`|❌ **目前沒用到**——我找了所有對 `image_assets` 下的查詢,沒有一個是用 `capture_instance_id` 當篩選條件(實際查詢都是用 `watchid`/`view_name`/`point_name`)|
|`idx_asset_internalnums`|`image_assets(watchid, internalnum1, internalnum2)`|❌ **目前沒用到**——同樣沒有查詢用 `internalnum1`/`internalnum2` 當篩選條件|

**小結,回答「有沒有現在沒在用的」**:有,而且不少——`watch_registry` 是完全的死表;`watch_runs`/`point_instances`/`capture_instances`/`analysis_results_v2` 是「持續累積但沒人讀回來」的單向紀錄表(可能是為未來報表/稽核功能預留,或只是設計時多做了正規化但 UI 端還沒接上);4 個索引裡有 3 個目前沒有任何查詢會用到它們,只有 `idx_path` 是真正有效的。這些空轉的索引/表不會造成錯誤,但每次寫入都要多付一點索引維護成本,是可以考慮清理的地方。

---

## 二、`local_templates` 的四個欄位

CREATE TABLE local_templates (
    TemplateID TEXT,
    version    TEXT,
    data       TEXT,
    updated_at REAL,
    PRIMARY KEY (TemplateID, version)
)

|欄位|說明|
|---|---|
|`TemplateID`|樣板的唯一識別碼(你看到的 `run_16613T(V3)`、`test_hdr1`、`test_Josh` 這些字串)。是複合主鍵的一半。|
|`version`|同一個 `TemplateID` 可以有多個版本(`v1`, `v2`...)。`TemplateID` + `version` 兩者合起來才是唯一鍵——這就是為什麼你會看到同名 TemplateID 出現好幾筆,只是 version 不同。|
|`data`|**整包 Template 內容的 JSON 字串**——不是攤平成很多欄位,而是把整個 Python `WatchTemplate` dataclass(所有 view、point、材質、量測設定…)序列化成一大串 JSON 文字塞進這一個 TEXT 欄位。你點開任一筆的 `data`,會看到一整包巢狀 JSON。|
|`updated_at`|最後更新時間,epoch seconds(Unix timestamp,浮點數)|

---

## 三、`local_templates` 怎麼傳到 AWS `moonlight-WatchTemplates`——實際流程 + 具體範例

### 關鍵:雲端優先寫,本地只在雲端成功後才寫

跟你可能猜測的「本地先存,背景再同步過去」**不一樣**——Template 儲存是 `DB/templates/template_manager.py:309` 的 `save_template()` 這個函式**同步、順序**執行的,順序是「先雲端、後本地」:

```
def save_template(self, template_obj):
    item = normalize_template_payload(
        template_obj.to_dynamo_item(),
        template_obj.TemplateID,
        template_obj.version,
        refresh_updated_at=True,
    )
    if self.cloud_db.save_template_cloud(item):        # ① 先寫 DynamoDB
        self.local_db.save_local_template(              # ② 雲端成功才寫 SQLite
            item["TemplateID"], item["version"], item
        )
        return True
    else:
        return False   # 雲端失敗 → 本地也不會存，整個操作視為失敗
```

### 具體例子:假設你在 UI 新建一個叫 `run_16613T` 的樣板(version `V3`)

**步驟 1 —— 記憶體中的 Python 物件**

UI 填完表單後,建立一個 `WatchTemplate` dataclass 實例,大致長這樣(欄位定義在 `DB/templates/template_structure.py:22`):

```
WatchTemplate(
    templatename = "run_16613T",
    TemplateID   = "run_16613T",
    version      = "V3",
    template_type= "watch",
    schema_version = "3.1",
    Brand = "Rolex", Reference = "16613", ...,
    watchView = WatchView(Front=..., Back=..., ...),   # 巢狀物件
    watchmaterial = WatchMaterialTemplate(...),
    ...
)
```


**步驟 2 —— `to_dynamo_item()` 轉成純 dict**

`asdict(self)` 把整個 dataclass(含巢狀的 `watchView`/`watchmaterial`)攤成一個 Python `dict`,並且會**刪掉沒被實際填寫/確認過的 point 欄位**(避免空樣板佔太多空間),結果大概是:

```
{
  "templatename": "run_16613T",
  "TemplateID": "run_16613T",
  "version": "V3",
  "schema_version": "3.1",
  "Brand": "Rolex", "Reference": "16613",
  "watchView": {
      "Front": {"Point1": {"pointID": "...", "complete": 1, ...}},
      "Back":  {...}
  },
  "watchmaterial": {...},
  "updated_at": 0.0
}
```

**步驟 3 —— `normalize_template_payload(..., refresh_updated_at=True)`**

補齊/修正一些欄位(`DB/templates/internalnum_config.py:1515`):把 `updated_at` 蓋成 `time.time()` 目前時間、確保 `TemplateID`/`version` 是乾淨字串、跑內部編號 schema migration、移除巨大的衍生欄位 `internalnum_manifest`(避免超過 DynamoDB 400KB item 上限)。

**步驟 4a —— 寫進 DynamoDB `moonlight-WatchTemplates`**

`data_manager/cloud_db.py:325` 的 `save_template_cloud()`:先把所有 `float` 轉成 `Decimal`(DynamoDB API 硬性要求),然後直接 `self.template_table.put_item(Item=safe_item)`。這一步寫進 DynamoDB 的**是一個原生的多屬性(multi-attribute)item**,結構大概像這樣(用 AWS 的 JSON 表示法):

```
{
  "TemplateID": {"S": "run_16613T"},        ← Partition Key
  "version":    {"S": "V3"},                 ← Sort Key
  "templatename": {"S": "run_16613T"},
  "Brand": {"S": "Rolex"},
  "Reference": {"S": "16613"},
  "watchView": {"M": {                       ← 巢狀 dict 存成 DynamoDB 的 "Map" 型別
      "Front": {"M": {"Point1": {"M": {...}}}},
      "Back":  {"M": {...}}
  }},
  "watchmaterial": {"M": {...}},
  "updated_at": {"N": "1755840000.123"}
}
```

**這是跟 SQLite 最大的結構差異**:同一包 Python dict,DynamoDB 是把**每個 top-level key 變成一個獨立的屬性(attribute)**,巢狀 dict/list 也保留原生的 Map/List 結構(可以直接對 `watchView.Front.Point1` 這種路徑做屬性層級查詢);而 SQLite 那邊完全不拆——整包直接 `json.dumps()` 變成一整串文字塞進**一個** `data` TEXT 欄位。

**步驟 4b —— 雲端成功後,才寫回本地 `local_templates`**

`DB/db_manager.py:519` 的 `save_local_template()`:

INSERT OR REPLACE INTO local_templates (TemplateID, version, data, updated_at)
VALUES ('run_16613T', 'V3', '{"templatename":"run_16613T","TemplateID":"run_16613T",...}', 1755840000.123)

也就是你在 DB Browser 看到的那一列——`data` 欄位裡那一大串 JSON,其實就是**跟剛才寫進 DynamoDB 那個 item 完全同一份內容**,只是 DynamoDB 把它拆成多個原生屬性存,SQLite 把它整包序列化成一段文字存。

### 如果雲端寫入失敗會怎樣?

`save_template_cloud()` 回傳 `False`(可能是超過 380KB 軟限制、驗證沒過、或網路/AWS 認證失敗),`save_template()` 就直接 `return False`,**完全不會執行第 4b 步**——也就是說,你新建的樣板即使在畫面上看起來存好了,只要雲端那一步沒成功,SQLite `local_templates` 裡根本不會有這一筆。這跟你之前問的照片同步(先本地寫、雲端是背景 best-effort)剛好是相反的設計哲學,前面文件裡也特別點出這是刻意的「fail-together」設計。





#### Claude分析新DB架構
```
以下是Moonlight system的建議的正式架構架構圖, 根據我的需求: "所以新的database不但要能夠執行以上提到的查詢跟分析, 而且我們可能有上千個watch系列, 十萬了watch. 跟你你設計的新database是否能否順暢的運行support這些功能性" 可以順暢支援，但必須採用「分層式 database architecture」。如果把十萬支 watch、數千萬 image assets 和數億筆 analysis values 全塞在單一 `watchDB.db`，最後一定會遇到檔案過大、寫入鎖、備份緩慢及 distribution 查詢過慢。 請深入分析這套系統的優缺點以及是否建議這樣設計或有更好的設計?為什麼 更詳細解釋各個單元以及其作用, 詳細解釋SQLite跟DynamoDB的差別, 以及這系統是怎麼運作的, 也以實際例子解釋你的新架構 flowchart LR App["App / CLI / Hardware"] --> Local["每台設備 Local SQLite V4"] Local --> Outbox["Transactional Sync Outbox"] Outbox --> Catalog["DynamoDB V4 Operational Catalog"] Outbox --> S3["S3 Artifact Storage"] Outbox --> Facts["Parquet / Iceberg Metric Facts"] Facts --> Athena["Athena Analytics"] Catalog --> API["UUID / Watch / Capture / Experiment API"] Athena --> Analysis["Distribution / Comparison / Statistics"]
```
# 深度分析:單站 SQLite + 四表 DynamoDB → 你提出的「分層式 V4 架構」

我先講結論,再逐層拆解:**你朋友/團隊建議的方向是對的,而且跟現有系統相比不是「推倒重來」,而是「補齊現有系統本來就欠缺的那一塊(結構化分析層)」**。理由詳述如下。

---

## 一、先釐清一個容易誤解的地方:瓶頸在「本地端」還是「中央端」?

這點很重要,因為它決定了你要優化的其實不是 `watchDB.db` 本身,而是雲端那一側。

回顧上一輪我們確認過的架構:`watchDB.db` 是**每一台工作站各自一份**,角色是「本地寫入緩衝區 + 離線快取」,理論上資料一旦 `synced=1` 上傳成功,本地就沒有義務永久保留全部歷史(現有程式碼沒有主動清除機制,但架構設計的意圖是如此)。

所以:

- **如果需求只是「單站要能順暢拍照、離線也能跑」**——現有 SQLite 單機檔案完全夠用,十萬支錶的「總量」跟單一站點的本地檔案大小沒有直接關係,因為每支錶只會短暫停留在某一台站點的本地 DB,同步完就是雲端的事了。
- **如果需求是「要能對全部十萬支錶、跨所有站點做統計分析/比對」**——瓶頸從來就不在 `watchDB.db`,而在**中央端**:目前系統唯一具備「集中式、可分析」能力的,是 DynamoDB(單筆查詢)+ Athena(直接掃 S3 上零散的 JSON/圖片 metadata)。而 Athena 直接掃描非結構化/半結構化的 S3 物件,在資料量到千萬、億級筆數時,查詢費用和延遲會急遽惡化——這才是你這次問題真正要解決的痛點。

你提出的架構圖裡新增的 **Parquet/Iceberg Metric Facts** 這個節點,正是針對這個「中央端分析瓶頸」設計的,不是針對本地 SQLite。這是我認為這個提案抓對重點的地方。

---

## 二、SQLite 與 DynamoDB 的本質差異(這是理解整張圖為什麼要「分層」的基礎)

|維度|SQLite|DynamoDB|
|---|---|---|
|資料模型|關聯式(Table + Schema,支援 JOIN)|NoSQL Key-Value/Document(PK/SK 為主,其餘欄位可自由增減,無強制 schema)|
|查詢能力|完整 SQL:`JOIN`、`GROUP BY`、子查詢、聚合函數,由查詢優化器自動選索引|只能用 PK 做等值查詢,或 PK+SK 做 range 查詢;要用第二維度就得另外建 GSI(如 `BrandModelIndex`);**沒有原生 JOIN,沒有跨 item 的 `GROUP BY`/聚合**|
|擴展性|單機檔案型,沒有原生水平擴展——要自己刻 sharding|雲端全託管,依 partition key 自動水平擴展,理論上可支撐近乎無限並發讀寫(前提是 key 設計均勻、不產生 hot partition)|
|交易(Transaction)|完整 ACID,可跨多表交易|有 `TransactWriteItems`,但限制較多(單次最多 100 item、額外計費),不像關聯式資料庫那樣自然好用|
|延遲/一致性|本地讀寫微秒~毫秒級,單機|個位數毫秒級,但要走網路;預設是最終一致性讀取,要付額外成本才有強一致性讀|
|成本模型|幾乎零成本(就是一個檔案)|依讀寫容量單位(RCU/WCU)或 on-demand 計費 + 儲存費用|
|適合場景|單機、需要複雜關聯查詢、離線優先|高並發、低延遲的「已知 key」點查詢(OLTP 型的 operational lookup)|
|**分析型查詢(OLAP)能力**|可以用 SQL 做,但單機無法平行掃描 TB 級資料|**完全不適合**——沒有 `GROUP BY`,沒有跨 item 聚合,想做統計分析必須另外搭配 Athena/EMR/Redshift 之類的分析引擎|

**這一行「分析型查詢能力」正是整套架構為什麼一定要多加一層 Athena(甚至 Parquet/Iceberg)的原因**——DynamoDB 天生就不是用來回答「這一萬支 Daytona 的 dial color 分布是什麼」這種問題的,無論資料量大小都一樣做不到,不是效能問題,是能力問題。

---

## 三、現有系統在「查詢跟分析」需求下,具體會卡在哪裡

用你提到的十萬支錶、上千系列規模具體推算:

- 假設每支錶平均 6 個 view × 5 個 point,每個 point 2~3 張照片 → 單支錶 **50~80 張影像資產**,十萬支錶就是 **500 萬~800 萬筆 `image_assets`**。
- 每張照片跑 3~5 種演算法分析 → **上千萬~上億筆 analysis 結果**。

現有架構下,這些資料最終落在 DynamoDB `WatchAnalysisResults` 表(flat item)+ S3(圖片/報表原始檔)。如果你要做「distribution/comparison/statistics」這類分析:

1. DynamoDB 做不到——只能一筆一筆用 PK 查,或全表 `Scan`(非常慢且昂貴,DynamoDB 的 Scan 是逐頁讀取全部資料再在應用端過濾,完全不是為分析設計的)。
2. 只能靠 Athena 直接對 S3 掃描。但目前 S3 上存的是**零散的原始 JSON/圖片**,沒有欄式(columnar)結構、沒有 partition 規劃——Athena 每次查詢都要整批掃過所有物件、解析 JSON,費用是按「掃描的資料量」計費,億級筆數規模下,一次查詢可能就要掃 TB 級資料,又慢又貴。
3. 前面文件也提過:S3 → Athena/Glue catalog 抓到新資料**沒有主動 refresh 機制**,新寫入的資料要多久才查得到是不確定的——這在「即時分析」場景下是硬傷。

這就是你提出的架構裡 **Parquet/Iceberg Metric Facts** 要解決的問題。

---

## 四、逐一解釋新架構每個節點的作用

App/CLI/Hardware
    │
    ▼
每台設備 Local SQLite V4  ──── 本地寫入緩衝區(跟現在角色相同,離線也能拍照/分析)
    │
    ▼
Transactional Sync Outbox ──── 新增的可靠投遞層(關鍵改進點,見下方詳述)
    │
    ├──▶ DynamoDB V4 Operational Catalog ──▶ UUID/Watch/Capture/Experiment API
    │         (單筆查詢用途:「這支錶現在狀態如何」「這次拍攝的 asset 有哪些」)
    │
    ├──▶ S3 Artifact Storage
    │         (存放原始影像、報表等大型二進位檔案,跟現在角色相同)
    │
    └──▶ Parquet / Iceberg Metric Facts ──▶ Athena Analytics ──▶ Distribution/Comparison/Statistics
              (結構化、欄式儲存的分析資料湖,這是新增的核心)

### ① Local SQLite V4(每台設備)

角色跟現在的 `watchDB.db` 一樣——本地寫入緩衝區。**這一層應該保留,不需要換掉**,因為離線可靠性、低延遲寫入的需求不會因為中央資料量變大而改變。差異只在於:V4 版本裡它應該更明確地只扮演「暫存」角色,同步完成後的資料應該有機制定期歸檔/清理,而不是無限累積在本地(這點現有系統目前沒有做,值得在新架構落地時一併補上)。

### ② Transactional Sync Outbox(新增,關鍵改進)

這是「Outbox Pattern」——一種分散式系統裡確保「本地資料寫入」和「對外發送同步事件」**要嘛一起成功、要嘛一起失敗**的標準做法。

跟現有系統對比:現在的 `raw_images`/`experiment_results` 表其實已經是一種簡化版 outbox(用 `synced` 欄位當作「還沒送出去的佇列」),但我們上次分析時發現了它的弱點——**上傳失敗沒有重試次數上限,也沒有 dead-letter 機制**,卡住的項目會無限期在背景重試,沒人知道它卡住了。

正式的 Outbox 設計會補上:

- 獨立的 outbox 事件表,跟業務資料**在同一個本地 SQLite transaction 裡一起 commit**(保證不會發生「資料寫了但同步事件漏記」的情況)。
- 一個獨立的 relay/dispatcher process,對每個下游目標(Catalog / S3 / Facts)分別追蹤送達狀態、重試次數,超過門檻就丟進 dead-letter 佇列讓人工介入,而不是無限重試。
- 同一份本地資料**同時扇出(fan-out)給三個不同下游**,而不是像現在只有一條「上傳到 S3 + 寫 DynamoDB」的單一路徑。

### ③ DynamoDB V4 Operational Catalog

延續現在四張表的角色,但聚焦在「Operational」——也就是**只做它擅長的事**:單筆、已知 key 的高並發查詢(某支錶現在的狀態、某次拍攝的 asset 清單、某個指令的執行狀態)。不試圖用它做統計分析,這正是吸取了現有系統「把所有東西都塞進 DynamoDB,包括拆分後的完整 watchentry」這種設計的教訓。

### ④ UUID/Watch/Capture/Experiment API

在 DynamoDB 之上包一層 API,把底層儲存細節封裝起來。現有系統目前是 App 端直接用 `boto3` SDK 操作 DynamoDB(`data_manager/cloud_db.py`),這樣做的問題是:未來如果要換底層儲存、加快取層、或做存取權限控管,所有呼叫端都要跟著改。加一層 API 是好的工程實踐,值得採納。

### ⑤ S3 Artifact Storage

角色不變,存放原始影像、HDR 合成圖、PDF 報表這類大型二進位檔案——這向來是 S3 的強項(幾乎無限容量、低成本、高耐久),不需要換。

### ⑥ Parquet/Iceberg Metric Facts(新增,整個架構的核心)

這是解決「十萬支錶、億級分析數值要做統計/分佈/比對」問題的關鍵元件。需要說明兩個技術概念:

- **Parquet**:一種「欄式(columnar)」檔案格式。跟現在 Athena 直接讀 JSON 的「列式(row-based)」相比,欄式儲存在做統計分析時效能差非常多——例如你只要 `dial_color` 這一欄做分布統計,Parquet 可以只讀這一欄,完全跳過其他幾十個欄位;而且每個欄位有自己的 min/max 統計值,可以直接跳過不相關的資料區塊(row group)。實務上同樣查詢,Parquet 通常比掃描原始 JSON 快 10~100 倍,Athena 費用也等比例下降(因為費用是按掃描量計費)。
- **Iceberg**:架在 Parquet 之上的「table format」,提供 ACID 交易、schema 演進(以後加新欄位不用重寫整個資料湖)、time travel(查詢某個時間點的歷史快照)、以及自動化的 partition 管理和小檔案合併(compaction)。對於「資料會持續增長、schema 偶爾會變」的分析資料湖場景特別合適——你的系統本來就有「v1 到 v2 schema migration」這類經驗,Iceberg 的 schema evolution 正好對應這個需求。

**重要提醒**:圖上「Outbox → Facts」這條線不是自動發生的,實務上需要一個 ETL/streaming 轉換元件(例如 AWS Glue ETL job、Kinesis Firehose、或定期跑的 batch job),把 DynamoDB/S3 上的原始 metadata 轉換、攤平、寫成 Parquet/Iceberg 格式,並依查詢模式設計 partition(例如依 `Brand`/`Model`/日期分區)。這是落地時容易被低估工作量的一塊。

### ⑦ Athena Analytics + Distribution/Comparison/Statistics

對 Parquet/Iceberg 資料做 SQL 分析,對應你最初提到的「查詢跟分析」需求本體。因為底層是結構化欄式資料且有 partition,查詢速度和成本相較「直接掃 S3 JSON」會有數量級的改善。

---

## 五、跟現有系統的對照表:保留 / 新增 / 取代

|現有系統|新架構|變化|
|---|---|---|
|每站 `watchDB.db`(SQLite,write buffer)|每站 Local SQLite V4|**保留角色**,建議補上定期歸檔/清理機制|
|`raw_images`/`experiment_results` 的 `synced=0` 輪詢|Transactional Sync Outbox|**升級**:同一 transaction 保證、多目標 fan-out、dead-letter 機制|
|4 張 DynamoDB 表(`WatchAnalysisResults`/`WatchCommandLog`/`WatchTemplates`/`WatchUsers`)|DynamoDB V4 Operational Catalog|**保留但收斂角色**——只做單筆 operational 查詢,不再承載「完整 watchentry 用 split-key 塞進同一張表」這種混用分析/交易的設計|
|App 直接用 boto3 呼叫 DynamoDB|UUID/Watch/Capture/Experiment API|**新增封裝層**|
|S3 存原始影像|S3 Artifact Storage|**保留不變**|
|Athena 直接掃 S3 上零散 JSON|Parquet/Iceberg Metric Facts + Athena|**新增結構化分析層**,這是解決你「查詢跟分析」+「十萬支錶規模」需求的核心|

---

## 六、兩個具體例子,走過新架構的兩條不同路徑

**例子 1——「查某支特定錶(WatchID = RLX-16613-00042)現在的狀態跟已拍攝的 asset 清單」**

這是「已知 key、單筆查詢」,走 **DynamoDB Catalog** 這條路:App 呼叫 `Watch API`,API 內部對 `WatchID` 做一次 `Query`(不是 Scan),個位數毫秒回應。**完全不會碰到 Athena/Parquet**,因為那條路是為分析設計的,拿來做單筆查詢反而更慢更貴。

**例子 2——「過去一年所有 Rolex Daytona 系列,dial color 的分布統計,並跟 Submariner 系列比較」**

這是「跨十萬支錶、要做 GROUP BY/統計」的分析型查詢,DynamoDB 做不到,走 **Parquet/Iceberg + Athena** 這條路:

1. 資料已經依 `Brand=Rolex, Model=Daytona/Submariner` 做 partition,Athena 查詢時先做 partition pruning,只掃相關分區,完全跳過其他上千個系列的資料。
2. Parquet 欄式儲存讓 Athena 只讀 `dial_color`、`Brand`、`Model` 這幾欄,不用把整包巢狀 JSON 全部解析出來。
3. 查詢秒級內回應,費用只跟實際掃描的資料量(而不是整個資料湖)成正比。

相較之下,如果現有架構要回答同一個問題,得讓 Athena 對 S3 上億筆零散 JSON 物件做全表掃描,解析每一筆的巢狀結構,查詢可能要跑數分鐘、掃描費用也高出一到兩個數量級。

---

## 七、我的建議:方向正確,但落地時要注意幾個風險點

1. **Outbox 要真的做到「同 transaction」**——如果只是換個名字繼續用 polling + `synced` flag,沒有解決「無限重試無 dead-letter」的根本問題,那就只是換皮,沒有真正解決現有系統文件裡點出的可靠性缺口。
2. **DynamoDB partition key 設計要避免 hot partition**——現有用 `WatchID` 當 PK 分散度還算合理,但如果新增的 Catalog 有「依系列批次查詢」的場景,要注意別讓某個熱門系列集中在少數 partition 上造成流量瓶頸。
3. **Parquet/Iceberg 的小檔案問題(small file problem)**——如果 Outbox 是即時、逐筆把資料寫進資料湖,很容易產生大量小 Parquet 檔案,反而拖慢 Athena 查詢。建議搭配 Iceberg 的自動 compaction,或改用小批次(micro-batch)方式寫入。
4. **需要明確定義 ETL 轉換邏輯**——目前現有系統裡類似 `normalize_template_payload` 這種「攤平巢狀資料」的邏輯,在新架構裡要在「Outbox → Facts」這一段重新設計一次(要決定哪些欄位攤平進 Parquet schema、怎麼處理 schema 演進)。
5. **Local SQLite 的角色要跟現有系統一樣明確保持「暫存」而非「歷史庫」**——不然十萬支錶的規模問題依然可能悄悄回到本地端。

**總結**:這套「本地 SQLite(OLTP write buffer)+ DynamoDB(operational catalog)+ S3(物件)+ Parquet/Iceberg + Athena(OLAP 分析)」的分層架構,是業界處理「高並發交易型讀寫 + 大規模分析型查詢」這兩種截然不同負載時的標準做法(關鍵字:OLTP/OLAP 分離、Lakehouse 架構)。以你們的規模(十萬支錶、上億筆分析值)來說,這是必要的演進方向,而不是過度設計。





#### 給Codex具體修改計畫
```
請將你的建議化成具體修改計畫, 並提供給Codex的prompt, 如果有需要在AWS設定的也詳細說明
```
# V4 架構導入計畫:具體修改步驟 + AWS 設定 + Codex Prompts

我把上一輪的建議拆成一個**分階段、風險可控**的落地計畫。原則是:**每個階段都是「新增」而非「取代」,舊路徑先保留、新路徑跑通驗證過再切換**,避免大改一次全部壞掉。

---

## 一、總體導入策略(4 個階段)

|階段|目標|風險|現有系統是否受影響|
|---|---|---|---|
|Phase 0|本地 Transactional Outbox|低——純新增表 + 新 dispatcher,舊 `raw_images`/`experiment_results` 輪詢先並行(shadow mode)|不影響現有讀寫|
|Phase 1|DynamoDB Catalog 收斂 + API 層|中——需要重構 `cloud_db.py` 的呼叫端|影響 App 端呼叫方式,但可用 API 層做相容包裝|
|Phase 2|Parquet/Iceberg 資料湖 + ETL|中——需要新 AWS 資源 + 批次/增量 ETL job|不影響現有讀寫,純新增|
|Phase 3|Athena on Iceberg + 分析查詢層|低——在 Phase 2 資料湖之上做查詢,不動舊資料|不影響現有讀寫|

建議順序:**0 → 2/3(可平行做,因為互不依賴) → 1(最後做,因為風險最高、牽動最多呼叫端)**。

---

## 二、具體修改計畫(逐階段對應實際檔案)

### Phase 0:Transactional Sync Outbox

**要解決的問題**:現有 `data_manager/cloud_sync.py` 的 `get_pending_uploads()` 直接輪詢 `raw_images`/`experiment_results` 的 `synced=0`,沒有重試上限、沒有 dead-letter,而且只有單一「上傳到 S3 + 寫 DynamoDB」路徑,沒辦法同時扇出給未來的資料湖 ETL。

**修改內容**:

1. `DB/db_manager.py` 新增一張表:

```
CREATE TABLE sync_outbox (
    outbox_id TEXT PRIMARY KEY,      -- uuid
    source_table TEXT,               -- 'image_assets' / 'analysis_results_v2' / 'local_templates'
    source_id TEXT,                  -- 對應來源表的主鍵
    target TEXT,                     -- 'catalog' / 's3' / 'lake'
    payload TEXT,                    -- JSON snapshot
    status TEXT DEFAULT 'pending',   -- pending / in_flight / done / dead_letter
    attempt_count INTEGER DEFAULT 0,
    last_error TEXT,
    created_at REAL,
    updated_at REAL
)
CREATE INDEX idx_outbox_status ON sync_outbox(status, target)
```

2. 修改 `register_image_asset()`、`register_analysis_result_v2()`、`save_local_template()`:在**同一個 SQLite transaction** 裡,除了寫業務表,也 `INSERT` 一筆(或多筆,依 target 數量)`sync_outbox` 記錄——這是 Outbox Pattern 的核心保證。
3. 新增 `data_manager/outbox_dispatcher.py`:取代 `cloud_sync.py` 目前的 `_worker_loop` 輪詢方式,改成讀 `sync_outbox WHERE status='pending'`,依 `target` 分派給不同下游 handler(S3 upload / DynamoDB catalog / 未來的 lake ETL),失敗時累加 `attempt_count`,超過門檻(例如 5 次)轉 `dead_letter` 並發告警,而不是無限重試。
4. `cloud_sync.py` 的舊輪詢邏輯**先保留、不刪**,兩條路徑並行跑一段時間(shadow mode),用 log 比對兩邊上傳結果是否一致,確認新路徑穩定後再讓舊路徑 no-op。

---

### Phase 2:Parquet/Iceberg 資料湖 + ETL

**要解決的問題**:分析數值資料目前只以零散 JSON 存在 S3/DynamoDB `metadata_raw` 裡,Athena 查詢要整批掃描解析,無法做高效的 distribution/comparison 分析。

**修改內容**:

1. 新增 `data_manager/lake_etl.py`:一個獨立的 batch job(可以先寫成本機可跑的 Python script,之後包成 AWS Glue Python Shell job 或 Lambda),職責:
    - 讀取 `sync_outbox WHERE target='lake' AND status='pending'`
    - 把 `analysis_results_v2`/`image_assets` 的 metadata 攤平成固定 schema(例如:`watch_id, brand, model, reference, view_name, point_name, algo_task_name, result_key, result_value, captured_at, dt`)
    - 寫成 Parquet 檔,依 `brand`(或 `dt` 日期)做 partition,上傳到 `s3://moonlight-system-v2/lake/analysis_facts/brand=.../dt=.../*.parquet`
    - 成功後把對應 outbox 記錄標記 `done`
2. 一次性 backfill script:把現有 DynamoDB `WatchAnalysisResults` 裡累積的歷史資料,批次匯出轉成同樣的 Parquet schema,補進資料湖(只做一次)。
3. 小檔案問題處理:ETL job 用**批次(例如每 5 分鐘或每累積 1000 筆)**寫一次 Parquet,而不是逐筆即時寫,避免小檔案氾濫;搭配 Iceberg 的 compaction(見 AWS 設定章節)。

---

### Phase 3:Athena on Iceberg + 查詢層

**修改內容**:

1. 在 Athena 建立 Iceberg table(DDL 見下方 AWS 設定章節)。
2. 新增 `data_manager/analytics_query.py`,包裝常用的分析查詢(取代/擴充現有的 `data_manager/athena_query.py`):
    - `get_distribution(brand, model, field)` → 對應「dial color 分布統計」這類需求
    - `compare_series(brand_a, model_a, brand_b, model_b, field)` → 對應「兩系列比較」
3. 對應到你的架構圖裡的「Distribution/Comparison/Statistics」這個應用層節點。

---

### Phase 1:DynamoDB Catalog 收斂 + API 層(放最後做)

**修改內容**:

1. 新增 `api/catalog_api.py`(建議用 FastAPI,因為系統已經有 `helper/docs/04_developer_onboarding/4.7_http_fastapi_...` 提到 FastAPI 相關基礎):包裝 `data_manager/cloud_db.py` 的讀取方法,對外提供 `GET /watch/{watch_id}`、`GET /watch/{watch_id}/assets`、`GET /experiment/{experiment_id}` 這類 REST 端點。
2. App 端(`App/main.py`)原本直接 `import boto3` 呼叫 DynamoDB 的地方,逐步改成呼叫這層 API——**這步影響面最大,建議留到最後、且用 feature flag 逐步切換**。

---

## 三、AWS 設定步驟(詳細)

以下每一步我會註明是「主控台手動設定」還是「可以用 CLI/IaC 自動化」。

### 1. S3 資料湖路徑規劃

沿用現有 bucket `moonlight-system-v2`(見 `config/system_config.yaml:57`),新增一個獨立前綴,跟現有 `Raw/`、`Analysis/` 分開,避免 Glue Crawler 掃到不相關的物件:
```
s3://moonlight-system-v2/lake/analysis_facts/brand=Rolex/dt=2026-08-25/part-0001.parquet
s3://moonlight-system-v2/lake/analysis_facts/_iceberg_metadata/...   ← Iceberg 自動管理的 metadata
```
這一步不需要主控台操作,ETL job 寫入時自動建立路徑。

### 2. 建立 Glue Database

主控台操作(AWS Glue → Databases → Add database),或用 CLI:

aws glue create-database --database-input '{"Name":"moonlight_lake"}'

### 3. 用 Athena 建立 Iceberg Table

在 Athena Query Editor 執行(這是**主控台/CLI 皆可**,推薦用 Athena 而不是手動建 Glue table,因為 Iceberg table 由 Athena 建立會自動處理好 table format metadata):

```
CREATE TABLE moonlight_lake.analysis_facts (
    watch_id string,
    brand string,
    model string,
    reference string,
    view_name string,
    point_name string,
    algo_task_name string,
    result_key string,
    result_value double,
    captured_at timestamp,
    dt date
)
PARTITIONED BY (brand, dt)
LOCATION 's3://moonlight-system-v2/lake/analysis_facts/'
TBLPROPERTIES (
    'table_type' = 'ICEBERG',
    'format' = 'parquet',
    'optimize_rewrite_delete_file_threshold' = '10'
);
```


### 4. Athena Workgroup(建議跟現有查詢隔離,避免互相搶資源/混費用)

主控台:Athena → Workgroups → Create workgroup,名稱建議 `moonlight-analytics-v4`,設定:

- Query result location:`s3://moonlight-system-v2/athena-results/v4/`
- 開啟 "Query result reuse"(相同查詢短期內重複執行可省錢)
- 設定 Data usage control(每次查詢掃描上限,例如 5GB,避免誤用全表掃描炸費用)

### 5. IAM 權限(ETL job 用的角色)

新增一個 IAM Role(例如 `moonlight-lake-etl-role`),附加最小權限 policy:

```
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::moonlight-system-v2/lake/*",
        "arn:aws:s3:::moonlight-system-v2"
      ]
    },
    {
      "Effect": "Allow",
      "Action": ["dynamodb:Query", "dynamodb:Scan", "dynamodb:GetItem"],
      "Resource": "arn:aws:dynamodb:*:*:table/moonlight-WatchAnalysisResults*"
    },
    {
      "Effect": "Allow",
      "Action": ["glue:GetTable", "glue:GetDatabase", "glue:UpdateTable"],
      "Resource": "*"
    }
  ]
}
```

主控台:IAM → Roles → Create role,選 "Glue" 或 "Lambda" 作為信任的服務(依你選 Glue Python Shell job 還是 Lambda 執行 ETL 而定),掛上這個 policy。

### 6. ETL 排程(Glue Job 或 EventBridge + Lambda)

如果用 Glue Python Shell Job(建議,因為系統已經在用 boto3/Python 生態):

- 主控台:AWS Glue → Jobs → Create job → Python Shell
- 上傳 `data_manager/lake_etl.py`
- 設定觸發排程(Glue Trigger,例如每 15 分鐘跑一次)

### 7. DynamoDB GSI 確認

你系統文件裡已經提過 `BrandModelIndex` **需要手動在 AWS Console 建立**,程式不會自動建立。如果 Phase 1 的 Catalog API 要支援「依 Brand/Model 查詢」,要確認這個 GSI 已存在:

主控台:DynamoDB → Tables → `moonlight-WatchAnalysisResults` → Indexes → Create index

- Partition key:`GSI1_PK`(String)
- Sort key:`GSI1_SK`(String)
- Index name:`BrandModelIndex`

### 8. 成本控管(現有系統文件提過這塊幾乎沒做,建議這次一併補上)

- Athena workgroup 設定每查詢資料掃描上限(見步驟 4)
- S3 生命週期規則:`lake/` 前綴下,舊 partition 超過 N 天可轉 S3 Glacier(如果冷資料不常查)
- CloudWatch 告警:ETL job 失敗次數、`sync_outbox` 表 `dead_letter` 累積數量超過門檻時告警

---

## 四、給 Codex 的 Prompt(分階段,可直接複製貼上)

> 這幾個 prompt 是設計成**依序、獨立的工作階段**餵給 Codex(或其他 coding agent),每個 prompt 都是自包含的(不假設 Codex 記得前一個階段的對話)。建議一次只跑一個階段,驗證通過再進下一個。

### Prompt 1 —— Phase 0:Transactional Outbox

你正在修改一個 Python 專案(位於 D:\Provenance Laboratories projects\ImagingLibWatch），這是一套手錶鑑定/檢測站的本地 SQLite + AWS 雲端同步系統。

背景：
- DB/db_manager.py 的 DatabaseManager 類別管理本地 SQLite (DB/watchDB.db)，其中 register_image_asset()、register_analysis_result_v2()、save_local_template() 這幾個方法目前分別直接寫入 image_assets / analysis_results_v2 / local_templates 表。
- data_manager/cloud_sync.py 的 CloudSyncManager 目前用 db.get_pending_uploads() 輪詢 raw_images/experiment_results 表的 synced=0 欄位來找出待上傳項目，上傳到 S3 後呼叫 CloudDatabaseManager.index_record() 寫入 DynamoDB。這個機制沒有重試次數上限、沒有 dead-letter，失敗項目會無限期重試。

任務：實作 Transactional Outbox Pattern，取代現有輪詢機制，具體要求：

1. 在 DB/db_manager.py 的 _init_db() 新增一張表：
   sync_outbox(outbox_id TEXT PRIMARY KEY, source_table TEXT, source_id TEXT,
   target TEXT, payload TEXT, status TEXT DEFAULT 'pending',
   attempt_count INTEGER DEFAULT 0, last_error TEXT,
   created_at REAL, updated_at REAL)
   並建立 idx_outbox_status ON sync_outbox(status, target) 索引。

2. 修改 register_image_asset()、register_analysis_result_v2()、save_local_template()：
   在同一個 SQLite connection/transaction 裡（也就是同一次 conn.commit() 之前），
   額外 INSERT 對應的 sync_outbox 記錄：
   - image_asset 寫入時，插入 target='s3' 和 target='catalog' 兩筆 outbox 記錄
   - analysis_result 寫入時，插入 target='s3', target='catalog', target='lake' 三筆
   - template 寫入時，插入 target='catalog' 一筆
   payload 欄位存這筆來源資料的 JSON snapshot（不要之後再回頭查一次來源表）。
   如果任何一步失敗，整個 transaction 要 rollback（不能只有業務表寫入成功而 outbox 遺漏）。

3. 新增 data_manager/outbox_dispatcher.py，定義 OutboxDispatcher 類別：
   - 建構子接收 db_manager、cloud_db_manager、s3_client（或既有的 CloudSyncManager 依賴）
   - dispatch_pending(batch_size=32) 方法：查詢 sync_outbox WHERE status='pending' ORDER BY created_at LIMIT batch_size，依 target 分派到對應 handler：
     - target='s3'：呼叫現有 S3 上傳邏輯（可以複用 cloud_sync.py 裡既有的上傳程式碼，抽成共用函式）
     - target='catalog'：呼叫 cloud_db_manager 對應方法（index_record 或 save_template_cloud）
     - target='lake'：先實作成 no-op（TODO: Phase 2 補上），只把狀態標記處理過
   - 每筆處理成功後 UPDATE status='done'；失敗則 attempt_count += 1，若 attempt_count >= 5 則 status='dead_letter' 並記錄 last_error，否則維持 pending 讓下次輪詢重試（可加一個簡單的 exponential backoff：下次處理時間 = created_at + 2^attempt_count 秒，用 updated_at 欄位判斷是否已到重試時間）
   - 提供 start_background_worker()：daemon thread，每 2 秒呼叫一次 dispatch_pending()（架構參考現有 CloudSyncManager 的 _worker_loop 寫法）
   - 提供 get_dead_letter_items() 方便之後做告警或人工檢視

4. 不要刪除或修改 cloud_sync.py 現有的輪詢邏輯——它要跟新的 OutboxDispatcher 並行執行一段時間（shadow mode），方便日後比對兩邊行為是否一致。只需要在檔案開頭加上註解說明這是舊路徑，新路徑是 outbox_dispatcher.py。

5. 幫我寫對應的單元測試（可以用 pytest + 一個暫存的 SQLite 檔案），至少涵蓋：
   - 寫入 image_asset 時確認同一 transaction 內 outbox 記錄有一起寫入
   - dispatch_pending 成功處理一筆並標記 done
   - 模擬 handler 拋例外，確認 attempt_count 遞增，超過 5 次後變成 dead_letter

請先讀過 DB/db_manager.py 和 data_manager/cloud_sync.py 的完整內容再開始修改，維持現有的 code style（例如 logger 用法、connection 管理方式 self._get_conn()）。修改完後跑一次現有測試（如果有的話）確保沒有破壞既有功能。

### Prompt 2 —— Phase 2:Parquet/Iceberg ETL

你正在修改同一個專案（D:\Provenance Laboratories projects\ImagingLibWatch），延續 Phase 0 已經完成的 Transactional Outbox（DB/db_manager.py 的 sync_outbox 表、data_manager/outbox_dispatcher.py 的 OutboxDispatcher）。

背景：sync_outbox 表裡 target='lake' 的記錄目前是 no-op，需要真正實作把資料寫進一個 S3 上的 Iceberg 資料湖，供 Athena 做統計分析用。

AWS 端已經準備好：
- S3 bucket: moonlight-system-v2，資料湖路徑前綴 lake/analysis_facts/
- Glue database: moonlight_lake
- Athena 已建立 Iceberg table moonlight_lake.analysis_facts，schema 為：
  watch_id string, brand string, model string, reference string,
  view_name string, point_name string, algo_task_name string,
  result_key string, result_value double, captured_at timestamp, dt date
  PARTITIONED BY (brand, dt)

任務：
```
1. 新增 data_manager/lake_etl.py，實作 LakeETLJob 類別：
   - 建構子接收 db_manager（本地 SQLite）、aws config（bucket 名稱、region）
   - run_batch(limit=1000) 方法：
     a. 查詢 sync_outbox WHERE target='lake' AND status='pending' ORDER BY created_at LIMIT limit
     b. 對每筆記錄，解析 payload JSON（這是 analysis_results_v2 或相關表的資料快照），
        攤平成符合上面 Iceberg schema 的欄位（watch_id, brand, model, reference,
        view_name, point_name, algo_task_name, result_key, result_value, captured_at, dt）
        注意：一筆 analysis_results_v2 的 data_json 裡可能包含多個 result_key/result_value
        （例如同時有 diameter_mm、weight_g 等），需要展開成多列（每個 key-value 一列）
     c. 用 pyarrow 或 pandas 把這批展開後的資料寫成一個 Parquet 檔案，
        依 brand 和 dt 分區寫到 s3://moonlight-system-v2/lake/analysis_facts/brand=<brand>/dt=<date>/
        （檔名用 uuid 避免衝突，例如 part-<uuid>.parquet）
     d. 成功後把這批 sync_outbox 記錄 status 改成 'done'
     e. 任何一步失敗，這批記錄要維持 pending（不要標記 done），並記錄錯誤 log
   - 提供 run_forever(interval_seconds=300)：每 5 分鐘跑一次 run_batch，用於本機測試或未來包成排程 job
```


1. 新增一次性 backfill script：DB/DB_test/backfill_lake_from_dynamodb.py
   - 用 boto3 分頁 Scan（或 Query，如果能依 WatchID 分批）現有 DynamoDB moonlight-WatchAnalysisResults 表
   - 只處理 record_type 為分析結果類型的 item（過濾掉純 asset index 或 split-key 的 INFO/VIEW#/MAT#/AUTH 項目，除非你判斷這些也該納入 lake，需要的話請在程式裡明確列出納入哪些 record_type）
   - 用跟 run_batch 一樣的攤平/寫 Parquet 邏輯，把歷史資料一次性補進資料湖
   - 加上 --dry-run 參數，先印出會處理幾筆、會產生幾個 Parquet 檔案，不真的寫入，方便先確認範圍再正式執行

3. 依賴套件：如果專案目前沒有 pyarrow，請加進 requirements.txt（或對應的依賴管理檔案，先檢查專案用的是什麼）

4. 寫單元測試：至少測試「一筆包含多個 result_key 的 analysis_results_v2 payload，正確展開成多列」的攤平邏輯，可以不用真的連 S3（用 mock 或本地檔案系統模擬）。

請先讀過 DB/db_manager.py 的 sync_outbox 表結構和 register_analysis_result_v2() 的 payload 實際格式，再開始寫攤平邏輯，確保欄位對應正確。

### Prompt 3 —— Phase 3:分析查詢層

你正在修改同一個專案，延續 Phase 2 已經建好的 Iceberg 資料湖表 moonlight_lake.analysis_facts
（schema: watch_id, brand, model, reference, view_name, point_name, algo_task_name,
result_key, result_value, captured_at, dt，依 brand/dt 分區）。

現有專案已經有 data_manager/athena_query.py，裡面有一個 AthenaQueryEngine 類別
（負責送 Athena 查詢、輪詢查詢狀態、取回結果，你可以先讀過這個檔案理解現有的呼叫模式）。

任務：新增 data_manager/analytics_query.py，實作 AnalyticsQueryService 類別，
包裝針對 moonlight_lake.analysis_facts 的統計分析查詢，複用 AthenaQueryEngine 的送查詢/取結果邏輯：

1. get_distribution(brand: str, model: str = None, result_key: str = None,
   date_from: str = None, date_to: str = None) -> list[dict]
   對指定 brand（可選 model）在指定日期區間內，某個 result_key（例如 'dial_color' 或
   'diameter_mm'）的數值分布。回傳每個 distinct value（若是數值型則自動分桶/bucket）
   及其出現次數，SQL 大致邏輯：
   SELECT result_key, result_value, COUNT(*) as cnt
   FROM moonlight_lake.analysis_facts
   WHERE brand = ? [AND model = ?] AND result_key = ? [AND dt BETWEEN ? AND ?]
   GROUP BY result_key, result_value ORDER BY cnt DESC

2. compare_series(series_a: tuple[str, str], series_b: tuple[str, str],
   result_key: str) -> dict
   比較兩個系列（各自是 (brand, model) tuple）在同一個 result_key 上的統計量
   （count、avg、min、max、標準差），用一次 SQL 或兩次查詢都可以，回傳結構清楚標示
   哪個統計量屬於哪個系列。

3. 兩個方法都要：
   - 用參數化查詢方式組 SQL（避免字串直接拼接使用者輸入造成 SQL injection，
     Athena 的 boto3 SDK 本身不支援 bind parameter，所以請對輸入做白名單/型別檢查，
     例如 brand/model 只允許英數字與空白，result_key 檢查是否在允許清單內）
   - 查詢送出後如果 Athena 回傳 QueryExecutionState 是 FAILED 或 CANCELLED，
     要拋出明確的例外並附上 Athena 回傳的錯誤訊息
   - 加上查詢逾時控制（例如最多輪詢 30 秒沒有結果就拋 TimeoutError）

4. 寫幾個範例呼叫，證明可以回答這兩個具體問題：
   - 「Rolex Daytona 系列的 dial_color 分布」
   - 「Rolex Daytona vs Rolex Submariner 的 diameter_mm 比較」
   可以寫成一個 DB/DB_test/test_analytics_query.py 的手動驗證腳本（不一定要是自動化 pytest，
   因為需要真的連 Athena，但至少要能被人手動執行來驗證功能）。

請先讀過 data_manager/athena_query.py 的完整內容，複用它現有的 boto3 Athena client
設定和查詢輪詢邏輯，不要重複造輪子。

---
## 五、風險與驗收標準(交給 Codex 前先確認)

在跑上述任何一個 Prompt 之前,建議先確認:

1. **每個 Phase 都先在 `dev` 分支上跑**,不要直接對 `main` 動手——你目前在 `dev` 分支上,這點是符合的。
2. **Phase 0 完成後的驗收**:寫一筆新的 `image_asset`,確認 `sync_outbox` 裡同時出現對應記錄;手動讓 handler 拋例外 5 次,確認變成 `dead_letter` 而不是無限重試。
3. **Phase 2 完成後的驗收**:跑 `backfill_lake_from_dynamodb.py --dry-run`,先看預估筆數是否合理(對照你在 DynamoDB 主控台看到的表項目數量級),確認沒問題再正式執行。
4. **AWS 費用護欄**:在 Phase 2/3 正式對接生產資料前,先在 Athena workgroup 設好查詢掃描上限(前面 AWS 設定步驟 4),避免一次不小心的 `SELECT *` 全表掃描產生意外費用。

這幾個 Prompt 我刻意保持「先讀現有程式碼再動手」「不要刪除舊路徑」「要有測試」這幾條規則,是因為現有系統的文件裡多次提到「兩邊各自獨立實作同一段防呆邏輯」「上傳無限重試」這類技術債,新架構如果重蹈覆轍(讓 Codex 自由發揮又沒有明確邊界),很容易複製同樣的問題。



#### 簡單解釋Sync Outbox跟Parquet / Iceberg Metric Facts
```
這是目前Moonlight system database架構圖. 是基於這些需求重新設計的
用你提到的十萬支錶、上千系列規模具體推算:

假設每支錶平均 6 個 view × 5 個 point,每個 point 2~3 張照片 → 單支錶 50~80 張影像資產,十萬支錶就是 500 萬~800 萬筆 image_assets。
每張照片跑 3~5 種演算法分析 → 上千萬~上億筆 analysis 結果。

現有架構下,這些資料最終落在 DynamoDB WatchAnalysisResults 表(flat item)+ S3(圖片/報表原始檔)。如果你要做「distribution/comparison/statistics」這類分析:

DynamoDB 做不到——只能一筆一筆用 PK 查,或全表 Scan(非常慢且昂貴,DynamoDB 的 Scan 是逐頁讀取全部資料再在應用端過濾,完全不是為分析設計的)。
只能靠 Athena 直接對 S3 掃描。但目前 S3 上存的是零散的原始 JSON/圖片,沒有欄式(columnar)結構、沒有 partition 規劃——Athena 每次查詢都要整批掃過所有物件、解析 JSON,費用是按「掃描的資料量」計費,億級筆數規模下,一次查詢可能就要掃 TB 級資料,又慢又貴。
前面文件也提過:S3 → Athena/Glue catalog 抓到新資料沒有主動 refresh 機制,新寫入的資料要多久才查得到是不確定的——這在「即時分析」場景下是硬傷。

先假設我是對database不了解的人. 請非常詳細解釋Transactional Sync Outbox是甚麼, 以及在這裡扮演的作用. 請非常詳細解釋Parquet / Iceberg Metric Facts是甚麼, 以及在這裡扮演的作用. 
```
這是一張非常經典且解決了實際痛點的現代資料架構圖。從你提供的 `structure.png` 可以看出來，這個設計完全是針對「海量資料的高效能分析」與「系統穩定性」量身打造的。

對於不熟悉資料庫底層運作的人來說，這些名詞聽起來很像火星文。別擔心，我們拋開生硬的技術術語，用日常生活的比喻來拆解這兩個關鍵核心。

### 一、 Transactional Sync Outbox（新增的可靠投遞層）

你可以把 **Transactional Sync Outbox** 想像成一個「超級可靠的郵局發送中心」。

- **痛點在哪裡？** 在舊架構或沒有這個設計的情況下，設備（SQLite）拍完照、做完分析後，必須自己「同時」把資料寫入 DynamoDB、上傳圖片到 S3，還要送到分析系統。萬一遇到網路不穩，可能圖片傳上 S3 了，但 DynamoDB 沒寫入成功。這會導致資料「對不起來」（有圖片卻沒紀錄），這在工程上叫做「資料不一致」。
    
- **它是什麼？**
    
    - **Transactional（交易性）：** 意思是「要嘛全部成功，要嘛全部失敗，沒有中間值」。
        
    - **Outbox（寄件匣）：** 就像電子郵件的寄件匣。
        
- **它在這裡扮演的作用：** 如 `structure.png` 所示，設備現在只需要把資料交給這個「郵局發送中心」，設備的任務就結束了（可以繼續去拍下一支錶）。這個 Outbox 會負責把資料**分發到三個不同的地方**：
    
    1. 把單筆查詢用的資料送進 **DynamoDB**。
        
    2. 把原始大檔案送進 **S3**。
        
    3. 把分析用的數據送進 **Parquet / Iceberg**。 如果途中某個環節失敗了，Outbox 會記住並自動重試，保證三邊的資料最終一定會完美同步，徹底解決了跨資料庫寫入的風險。
        

### 二、 Parquet / Iceberg Metric Facts（結構化、欄式儲存的分析資料湖）

你可以把這個組合想像成一座「擁有超強索引系統的現代化圖書館」。

- **痛點在哪裡？** 過去你們把原始 JSON 檔案丟在 S3，這就像把幾億張寫著數據的「便條紙」全部堆在一個大倉庫裡。當你想查「某個特定演算法的平均值」時，Athena 必須把幾億張便條紙**全部讀過一遍**才能算出來，這就是為什麼每次查詢都要掃描 TB 級資料，既慢又燒錢。
    
- **它是什麼？**
    
    - **Parquet（欄式儲存格式）：** 這是一種檔案格式。傳統資料是一列一列存的（像發票明細），Parquet 則是一欄一欄存的。當分析師只想查「總金額」這一欄時，系統只會讀取「總金額」的資料，完全跳過其他不相關的資訊。這能將讀取量縮減 90% 以上。
        
    - **Iceberg（資料表格式）：** 這是一個智慧型目錄與索引系統。它會精準記錄每一筆新資料放在哪裡。
        
    - **Metric Facts（事實數據）：** 指的就是你們那上千萬到上億筆的演算法分析結果。
        
- **它在這裡扮演的作用：** 如 `structure.png` 所示，這是為「Distribution/Comparison/Statistics」量身打造的核心。當資料進入這裡後，Iceberg 會立刻更新目錄（解決了你提到「S3 不知道何時才有新資料」的硬傷，實現即時分析）。當 Athena 要做統計時，Iceberg 會直接告訴 Athena：「你要的數據在第 3 到第 5 號櫃子的第二層」，然後 Athena 利用 Parquet 的特性，只抽出需要的「那一欄」來算。 **結果就是：查詢時間從幾分鐘變成幾秒鐘，掃描費用也大幅暴跌。**
    

這個架構非常精準地解決了「單筆查詢（DynamoDB）」與「宏觀大數據分析（Iceberg）」無法共用同一種資料庫的矛盾。

目前在看這張架構圖時，針對這三個資料流的分發機制，你們團隊有打算使用哪種雲端服務（例如 AWS SQS, Kinesis 或 EventBridge）來實作這個 Outbox 嗎？




#### 詳細解釋Sync Outbox跟Parquet / Iceberg Metric Facts
```
現在假設我是對database了解的engineer. 請更仔細非常詳細解釋Transactional Sync Outbox是甚麼, 以及在這裡扮演的作用. 請更仔細非常詳細解釋Parquet / Iceberg Metric Facts是甚麼, 以及在這裡扮演的作用.
```

既然切換到工程師視角，我們就直接從 System Design、分散式架構與 Data Lakehouse 的底層機制來拆解 `structure.png` 中這兩個關鍵元件的實作細節與價值。這個設計非常漂亮地融合了 CQRS (Command Query Responsibility Segregation) 與 Event-Driven 概念。

### Transactional Sync Outbox：解決 Dual-Write 與解耦

在微服務或邊緣運算架構中，最怕的就是「Dual-Write (雙重寫入)」問題。過去設備可能要同時呼叫 DynamoDB API 與 S3 API，一旦中途遇到 Network Partition，就會產生髒資料 (Dirty Data) 或狀態不一致。

- **ACID 本地保證：** 設備端的 Local SQLite 實作了 Transactional Outbox Pattern。當寫入一筆 Watch Analysis 紀錄時，會在同一個 DB Transaction 內將業務資料更新，同時 Insert 一筆 Event 到本地的 Outbox 表。這保證了「業務操作」與「事件生成」的原子性 (Atomicity)。
    
- **非同步的 At-Least-Once 投遞：** 系統背景會有一個 Sync Daemon (或 CDC worker) 輪詢/監聽這個 Outbox 表。它負責將 Event 拋到雲端的 Message Broker (如 Kinesis, EventBridge 或 SQS)。只有當雲端確認接收 (ACK) 後，本地 Outbox 的紀錄才會被標記完成或刪除。這容忍了設備的離線狀態，並保證資料不遺失。
    
- **Fan-out 與寫入隔離：** 如 `structure.png` 所示，Outbox 成為資料上雲的單一入口。後端的訂閱者再將事件 Fan-out 到 DynamoDB (供 UUID 快速 Point Query) 與 S3 (存放 Binary Artifact)。這將邊緣設備的寫入壓力與雲端三個異質 Data Sink 完全解耦。
    

### Parquet / Iceberg Metric Facts：具備 ACID 的現代化湖倉

這是將傳統 Data Lake 升級為 Data Lakehouse 的核心，專門解決上億筆資料在 S3 上做 OLAP (線上分析處理) 時，I/O 成本失控與 Data Staleness (資料延遲) 的硬傷。

- **Columnar I/O 與 Predicate Pushdown (Parquet)：** 捨棄 JSON 的 Row-based 儲存。Parquet 的欄式結構與 Dictionary Encoding 讓資料緊密壓縮。當 Athena 查詢特定維度的統計 (例如 `SELECT AVG(score) FROM metrics WHERE defect_type='scratch'`) 時，底層引擎只需要讀取該 Column 的 Chunk，並利用 Parquet footer 的 min/max 統計值直接略過不相關的 Block，將掃描量從 TB 級驟降至 MB 級。
    
- **Atomic Commits 與 Snapshot Isolation (Iceberg)：** Iceberg 是 Open Table Format 的中介資料層 (Metadata Layer)。它透過 Manifest files 來管理 S3 上龐大的 Parquet 檔案群。當 Outbox 將新批次的分析資料寫入時，Iceberg 會執行 Atomic Commit 產生新的 Snapshot。這徹底解決了過去 Glue Catalog 緩慢且不同步的問題，實現近乎即時的 Read-After-Write Consistency。
    
- **Hidden Partitioning：** 面對破億筆資料，Iceberg 可以動態管理 Partition (例如依據 `ingest_time` 或 `watch_model`)，而不需要改變使用者的查詢 SQL。這讓 Athena 的 Query Planner 能精準執行 Partition Pruning，避免了毀滅性的全表 Scan。
    

這個架構徹底把 OLTP (DynamoDB 單筆查詢) 與 OLAP (Iceberg 大量聚合) 的物理儲存分開了。

針對 Transactional Sync Outbox 的實作，你們目前的邊緣設備是打算採用哪一種機制來實作 Outbox 的 polling 與上傳 (例如自建的背景 Cron Job 還是使用像 Debezium 這類的 CDC 工具) 呢？





#### 開通那些AWS服務
```
我目前已經開啟AWS Athena, DynamoDB, IAM, S3, IoT Core服務, 那如果要實現新的database架構包含Parquet, iceberg, outbox, 需要在開通那些AWS服務? 請一步步解釋如何開通那些服務
```
為了實現包含 Iceberg 資料湖與 Transactional Sync Outbox 的現代化架構，你需要補齊中介資料目錄與雲端非同步處理的運算資源。

**AWS Glue (必備：Iceberg 的核心目錄)** Athena 無法自己記住 S3 上千百個 Parquet 檔案的狀態，它必須依賴 AWS Glue Data Catalog 作為 Iceberg 的 Metastore，用來追蹤 Snapshot 與 Schema 演化。

- **建立 Database：** 進入 AWS Glue 主控台，在左側導覽列選擇 Data Catalog > Databases，點擊 Add database（例如命名為 `moonlight_lake`）。
    
- **無伺服器配置：** Glue Catalog 是完全 Serverless 的服務，建立目錄不需要開啟實體伺服器。
    
- **權限設定：** 確保你現有的 IAM 策略中，允許後續處理資料的應用程式具備 `glue:CreateTable`, `glue:UpdateTable`, `glue:GetTable` 權限。
    

**AWS Lambda (必備：Outbox 的雲端處理工廠)** 本地機台 SQLite Outbox 的資料上雲後，需要一個無狀態的運算單元來執行 `lake_etl.py` 的邏輯：將狀態寫入 DynamoDB、將影像特徵轉為 Parquet，並透過 PyIceberg 呼叫 Glue 完成 Snapshot Commit。

- **建立 Function：** 進入 Lambda 主控台，點選 Create function，選擇 Python 執行環境。
    
- **綁定 IAM 角色：** 為該 Lambda 賦予能夠同時存取 S3、DynamoDB 與 Glue 的執行角色 (Execution Role)。
    
- **建立 Layer：** Iceberg 寫入需要依賴 `pyarrow` 與 `pyiceberg` 套件，必須將這些依賴庫打包成 Lambda Layer 才能順利執行。
    

**Amazon SQS (強烈建議：Outbox 的雲端避震器)** 當洛杉磯的遠端測試機台進行高頻率的光學檢測時，瞬間產生的巨量 Outbox 訊息若直接打向資料庫，容易引發寫入衝突或 API 限流。SQS 能作為可靠的雲端緩衝區。

- **建立 Queue：** 進入 SQS 主控台，建立一個 Standard Queue。
    
- **對接現有路由：** 在你已開通的 IoT Core 中，新增一條 Message Routing Rule，設定當機台透過 MQTT 發布 Outbox 訊息時，直接轉發進此 SQS 佇列。
    
- **設定觸發器：** 在 SQS 設定頁面中，將剛才建立的 Lambda 設為觸發器 (Event Source)。Lambda 就會依照系統負載，平穩地從佇列抓取資料並寫入 Iceberg 與 DynamoDB。
    

對於邊緣機台將 Outbox 實體資料拋上雲端的第一步，目前是打算直接呼叫 AWS API Gateway，還是沿用既有的 MQTT 協定透過 IoT Core 傳輸？









