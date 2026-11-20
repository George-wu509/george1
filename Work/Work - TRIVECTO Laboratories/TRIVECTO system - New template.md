

|                                                              |     |
| ------------------------------------------------------------ | --- |
| [[#### 幫我設計新的Human_authenticity跟Box_authenticity]]           |     |
| [[#### 實作修改class Human_authenticity跟class Box_authenticity]] |     |
|                                                              |     |
|                                                              |     |


#### 幫我設計新的Human_authenticity跟Box_authenticity
```
請提供建議幫我設計(目前還不需要修改code)可以把watch所有view以及material的所有identifier都存入到Human_authenticity(watchauthenticity_structure.py), 以及box所有identifier以及material都存入到Box_authenticity(boxauthenticity_structure.py), 而且可能identifier數量也一直在擴充中. 因為之後每次做完scan的watch的watchentry以及box的都會保存到local DB然後跟AWS同步(應該是moonlight-WatchAnalysisResults), 之後譬如scan了越來越多手錶也要針對每個identify統計以此判定authentication, 也要能方便查詢各個identify.
```

可以，我會建議用「雙層設計」：

1. `Human_authenticity` / `Box_authenticity` 裡保存完整 snapshot，讓每個 `watchentry` / `boxentry` 自己帶著當次 scan 的所有 authentication evidence。
2. 另外在 SQLite / DynamoDB 建 identifier index，讓之後大量手錶統計、查詢、比對時不用每次 parse 整包 JSON。

只把所有 identifier 塞進 `watchauthenticity` JSON 是可以保存，但不適合「查詢各個 identify」和「跨很多支錶做統計」。DynamoDB/SQLite 都會痛苦。

**建議資料結構**  
目前 [Human_authenticity (line 9)](D:/Provenance Project/ImagingLibWatch/DB/templates/watchauthenticity_structure.py:9) 和 [Box_authenticity (line 6)](D:/Provenance Project/ImagingLibWatch/DB/templates/boxauthenticity_structure.py:6) 都只有 component 判定欄位。建議保留既有欄位，新增 dynamic 欄位，例如：

```
@dataclass
class Human_authenticity:
    EntireWatch: str = ""
    Dial: str = ""
    AllHands: str = ""
    Crown: str = ""
    Bracelet: str = ""
    Box: str = ""
    Case: str = ""
    Crystal: str = ""
    Bezel: str = ""
    Caseback: str = ""
    EntireMovement: str = ""
    Source: str = ""

    schema_version: str = "auth-v2"
    identifier_summary: Dict[str, Any] = field(default_factory=dict)
    identifiers_by_component: Dict[str, Any] = field(default_factory=dict)
    identifiers_by_view: Dict[str, Any] = field(default_factory=dict)
    material_identifiers: Dict[str, Any] = field(default_factory=dict)
```

`Box_authenticity` 類似：

```
@dataclass
class Box_authenticity:
    EntireBox: str = ""
    OuterBox: str = ""
    InnerBox: str = ""
    CrownLogo: str = ""
    Pillow: str = ""
    Papers: str = ""
    Source: str = ""

    schema_version: str = "box-auth-v2"
    identifier_summary: Dict[str, Any] = field(default_factory=dict)
    identifiers_by_component: Dict[str, Any] = field(default_factory=dict)
    identifiers_by_view: Dict[str, Any] = field(default_factory=dict)
    material_identifiers: Dict[str, Any] = field(default_factory=dict)
```

**identifier row 建議統一格式**  
每個 identifier 不要只存 `{key: value}`，要存成可追蹤、可比對、可統計的 row：

```
{
  "identifier_id": "uuid-or-stable-hash",
  "scope": "watch",
  "component": "DIAL",
  "view": "Front",
  "point": "micropoint1",
  "capture_id": "std_1",
  "task_name": "front_stitch_service",
  "key": "hour_hand_length",
  "label": "Hour Hand Length",
  "value": 123.45,
  "value_type": "number",
  "unit": "px",
  "value_path": "details.hands_analysis.hour.length",
  "source_json_path": "...",
  "source_image_asset_id": "...",
  "passed": true,
  "comparison": {
    "standard": 120.0,
    "diff_pct": 2.875,
    "tolerance_pct": 5.0
  },
  "created_at": "2026-06-28T..."
}
```

這樣 identifier 數量一直增加也沒問題，因為不是固定 dataclass 欄位，而是 list/dict 裝 dynamic records。

**在 authenticity 裡怎麼分層**  
我會建議 `watchauthenticity.identifiers_by_component` 長這樣：

```
{
  "DIAL": {
    "status": "Authentic",
    "identifier_count": 86,
    "passed_count": 80,
    "pass_rate": 93.0,
    "identifiers": [...]
  },
  "BEZEL": {
    "status": "Authenticity not confirmed",
    "identifier_count": 31,
    "passed_count": 20,
    "pass_rate": 64.5,
    "identifiers": [...]
  }
}
```

`identifiers_by_view` 則用原始來源排序：

```
{
  "Front": {
    "micropoint1": [...],
    "macropoint1": [...],
    "toppoint1": [...]
  },
  "Back": {},
  "Strap": {}
}
```

`material_identifiers` 建議不要混進 image identifiers，而是保持材料檢測獨立：

```
{
  "CaseBody": {
    "component": "CASE",
    "scan_type": "XRF",
    "identifiers": [
      {"key": "AuFineness", "value": "..."},
      {"key": "GoldPlating", "value": "..."},
      {"key": "PluginResult", "value": "..."}
    ],
    "raw_material_snapshot": {...}
  }
}
```

這符合你們 AGENTS 裡「point-level metadata separate from image-level metadata」的方向，也避免 material 結果跟 image processing metric 混在一起。

**Local DB 建議**  
目前 SQLite 有 `experiment_results`、`analysis_results_v2`、`watch_metadata`、`ui_asset_registry`。我建議新增一張 normalized table，例如：

`auth_identifier_index`

欄位大概：

- `record_id`
- `watchid`
- `entry_type`: `watch` / `box`
- `scope`: `watch` / `box` / `material`
- `component`
- `view_name`
- `point_name`
- `capture_id`
- `task_name`
- `identifier_key`
- `identifier_label`
- `value_number`
- `value_text`
- `value_bool`
- `unit`
- `passed`
- `diff_pct`
- `source_asset_id`
- `source_json_path`
- `created_at`

常用 index：

- `(watchid)`
- `(entry_type, component)`
- `(task_name, identifier_key)`
- `(component, identifier_key)`
- `(identifier_key, value_number)`
- `(Brand, Model, Reference, identifier_key)` 如果你們願意 denormalize metadata

這張表是之後「掃了越來越多手錶後，針對每個 identifier 統計 authentication」的核心。`watchauthenticity` 裡的 JSON 是完整快照，`auth_identifier_index` 是查詢/統計用。

**AWS / DynamoDB 建議**  
`moonlight-WatchAnalysisResults` 不建議只放一個超大的 `watchentry` item，DynamoDB item 有大小限制，而且 identifier 會越來越多。你們其實已經有 split item 概念，`put_item_split(pk, sk, attributes)` 會用 `WatchID` + `sort_key` 存不同類別。

建議 AWS 寫法：

- `PK = WatchID`
- `SK = AUTH#SUMMARY`
- `SK = AUTH#WATCH#COMPONENT#DIAL`
- `SK = AUTH#WATCH#VIEW#Front#micropoint1`
- `SK = AUTH#BOX#COMPONENT#CrownLogo`
- `SK = AUTH#MATERIAL#WATCH#CaseBody`
- `SK = AUTH#IDENTIFIER#<task_name>#<identifier_key>#<hash>`

如果要方便跨錶查詢，再加 GSI：

- `GSI1_PK = IDENTIFIER#hour_hand_length`
- `GSI1_SK = BRAND#ROLEX#MODEL#SUBMARINER#VALUE#000123.45`
- 或 `GSI1_PK = COMPONENT#DIAL#IDENTIFIER#hour_hand_length`

這樣可以查「所有 DIAL 的 hour_hand_length 分布」、「某 reference 的某 identifier 平均值」、「某支錶有哪些 failed identifiers」。

**我會避免的設計**  
不要把 identifier 做成 dataclass 固定欄位，例如 `DialIdentifier1`, `DialIdentifier2`。因為 mapping 目前已經有 240 specs，而且 `expand: True` 會讓實際數量不固定。

也不要只存在 PDF 或 `report_identifiers_<WatchID>.json`。那份檔案適合 report artifact，但不適合做資料庫主資料。

**推薦落地順序**

1. 先定義通用 `AuthenticationIdentifier` record schema。
2. 擴充 `Human_authenticity` / `Box_authenticity`，只加 dynamic dict/list 欄位，保留舊欄位相容。
3. 在 PDF 生成前後，把目前 `identifier_doc` merge 回 `current_watchentry.watchauthenticity` 或 box entry 的 `watchauthenticity`。
4. Local DB 新增 `auth_identifier_index`，每次完成 scan/finish entry 時 bulk upsert。
5. DynamoDB 用 split item 儲存 summary + component + identifier index，不把所有資料塞單一 item。
6. Report 改成優先讀 `watchauthenticity` / identifier index，找不到再 fallback 掃 task JSON。

總結：`Human_authenticity` 和 `Box_authenticity` 可以當「該次 scan 的完整 authentication snapshot」，但真正支撐未來統計和查詢的應該是 normalized identifier index。這樣既保留 watchentry/boxentry 自包含，也能讓 Moonlight 後面累積很多 scan 後做 authentication learning。



#### 實作修改class Human_authenticity跟class Box_authenticity
```
補充一點我們所有的raw images跟結果包括image based results跟yaml file results都會保留在local driver也上傳到AWS S3. 那你這樣設計的是否已經優化符合我們的設計? 如果是請先修正class Human_authenticity跟class Box_authenticity部分及相關code, 其他跟AWS相關的我以後再修正code
```
符合你補充的設計：raw images、image results、YAML results 繼續由 local driver/S3 保存；`watchauthenticity`/`Box_authenticity` 現在只保存「可查詢的 authentication snapshot + source path/reference」，不複製大檔本體。AWS split/index 的部分我沒有動，留給後續。

已完成修改：

- [watchauthenticity_structure.py](D:/Provenance Project/ImagingLibWatch/DB/templates/watchauthenticity_structure.py): `Human_authenticity` 新增 `identifier_summary`、`identifiers_by_component`、`identifiers_by_view`、`material_identifiers`、`source_artifacts`，並加入從 `identifier_doc`/material dataclass 建 snapshot 的 helper。
- [boxauthenticity_structure.py](D:/Provenance Project/ImagingLibWatch/DB/templates/boxauthenticity_structure.py): `Box_authenticity` 加同樣欄位與 helper，scope 會標成 `box`。
- [template_manager.py](D:/Provenance Project/ImagingLibWatch/DB/templates/template_manager.py): 載入舊 YAML/JSON 時，會把 `watchauthenticity` dict 還原成 `Human_authenticity` 或 `Box_authenticity` dataclass。
- [App/main.py](D:/Provenance Project/ImagingLibWatch/App/main.py): PDF identifier doc 經 goldstand evaluation 更新後，會同步回 `current_watchentry.watchauthenticity`，並同步 watch/box material identifiers。

驗證已跑過：

```
python -m py_compile DB\templates\watchauthenticity_structure.py DB\templates\boxauthenticity_structure.py DB\templates\template_manager.py App\main.py
```

也做了 smoke test，確認 identifier count、pass rate、box scope、material identifier extraction 都正常。

注意事項：目前只是把 snapshot 回填到 entry object；只要後續流程呼叫既有 `to_dynamo_item()` / local DB save，就會包含這些欄位。AWS/DynamoDB query index、SQLite normalized `auth_identifier_index` 尚未建立，這部分照你說的先不動。另有兩個原本就存在的 untracked files 我沒有碰：`algorithms/watchband_stitcher.py`、`config/watchband_stitch.yaml`。
