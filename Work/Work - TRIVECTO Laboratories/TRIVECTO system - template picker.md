

|                                   |     |
| --------------------------------- | --- |
| [[#### reassign internum1 and 2]] |     |
|                                   |     |
|                                   |     |
|                                   |     |


#### reassign internum1 and 2
```
在之前的template版本watch的每個特定拍照位置是以watchpoint定義譬如front.macropoint1, back.micropoint2等, 每個watchpoint還有數個standard_capture, hdr_capture, focus_capture. 之前我們定義了internalnum(config/internalnum_config.yaml). 現在要重新規劃internalnum1, internalnum2根據下列的規則: internalnum1都是四位數字. 我希望Front是0開頭(譬如0xxx), Back是1開頭(譬如1xxx), OpenBack是2開頭(譬如2xxx), OpenBackCrown是3開頭(譬如3xxx), Strap是4開頭(譬如4xxx), Box是5開頭(譬如5xxx). 之前的每個watchpoint還有數個standard_capture, hdr_capture, focus_capture. 現在把這些同個watchpoint(現在是同個internalnum1)記錄成同個internalnum1但不同的internalnum2, 譬如本來front.micropoint1(假設internalnum1=0012)有std_1, std_2, hdr_1, 及一個focus_capture, 則他們則都是internalnum1=0012, internalnum2=0001, internalnum2=0002, internalnum2=0003, internalnum2=0004. internalnum1的命名順序大致是這樣 toppoint ->sidepoint(如果有的話)->macropoint->micropoint->distancepoint(如果有的話). 譬如front.toppoint的internalnum1應該就是0001. 而在config/internalnum_config.yaml裡面每個entries為什麼有兩套的internalnum1跟internalnum1是否會造成困擾及造成不一致的可能問題. 譬如 "0001": "0001": internalnum1: "0001" internalnum2: "0001" 請詳細檢查原本使用watchpoint, capture的功能性(拍照, template, APP UI等)是否都由完整的使用internalnum1跟internalnum2取代. 像front.macropoint1這樣的名稱在code裡面應該只出現在internalnum_config.yaml的description. 以及template包括過去建立的template是否也需要改變資料結構? 要如何修改才保險不會破壞過去建立的template? 或應該有保險不會刪除或遺失原有功能性的方式可以提升template
```


完成了這一輪比較保守的 internalnum refactor。重點是：現在 `internalnum1` 是 point-level，`internalnum2` 是 capture-level；同一 watchpoint 的 `std_1/std_2/hdr_1/focus_1` 會共用同一個 `internalnum1`，並依序得到 `0001/0002/0003/0004`。

**主要修改**

- [DB/templates/internalnum_config.py (line 36)](D:/Provenance Project/ImagingLibWatch/DB/templates/internalnum_config.py:36)：新增 Front=0xxx、Back=1xxx、OpenBack=2xxx、OpenBackCrown=3xxx、Strap=4xxx、Box=5xxx 的編號規則；排序為 top -> side -> macro -> micro -> distance -> glass。
- [DB/templates/watchview_structure.py (line 11)](D:/Provenance Project/ImagingLibWatch/DB/templates/watchview_structure.py:11)：capture dataclass 現在也有 `internalnum1/internalnum2`。
- [config/internalnum_config.yaml (line 1)](D:/Provenance Project/ImagingLibWatch/config/internalnum_config.yaml:1)：升到 `schema_version: 3`，移除 entry body 內重複的 `internalnum1/internalnum2`，改成 YAML key 為權威來源，避免 `"0001": {"0001": {internalnum1: "0001"}}` 這種可能不一致的雙來源。
- [core/workflow_manager.py (line 2463)](D:/Provenance Project/ImagingLibWatch/core/workflow_manager.py:2463)、[App/main.py (line 12235)](D:/Provenance Project/ImagingLibWatch/App/main.py:12235)、[Controller/test_hardware/test_template_ui.py (line 4984)](D:/Provenance Project/ImagingLibWatch/Controller/test_hardware/test_template_ui.py:4984)：capture 流程會帶著 `internalnum1/internalnum2`。
- [DB/db_manager.py (line 92)](D:/Provenance Project/ImagingLibWatch/DB/db_manager.py:92)、[data_manager/local_storage.py (line 396)](D:/Provenance Project/ImagingLibWatch/data_manager/local_storage.py:396)、[data_manager/cloud_db.py (line 147)](D:/Provenance Project/ImagingLibWatch/data_manager/cloud_db.py:147)：SQLite、raw image metadata、DynamoDB serialization 都會保存 internalnums。
- [config/system_config.yaml (line 345)](D:/Provenance Project/ImagingLibWatch/config/system_config.yaml:345)：service internalnums 已依新 mapping 更新。

**Template 相容性**  
舊 template 不需要破壞性改資料結構。`normalize_template_payload()` 會在 load/save/flatten 前補齊 point 與 capture internalnums；也同時處理 legacy `views` 和 canonical `watchView`，避免兩者分岔。舊的 `capture_id` 仍保留，因為 App、報告、舊 selector、演算法 config 仍需要它做相容橋接。

我不建議直接批次覆寫所有歷史 template。比較保險的做法是：讀取舊 template -> normalize 成新結構 -> 另存新 version；確認 App capture、report、analysis 都正常後，再視需要做正式 migration。

**重要檢查結果**  
還沒有完全達到「`front.macropoint1` 只出現在 `internalnum_config.yaml` description」這個狀態。它仍然是實際功能依賴，包含：

- `config/hdr_config.yaml`
- `config/front_stitch_config.yaml`
- `config/pdf_config.yaml`
- `algorithms/image_stitcher.py`
- `tasks/task_algos/front_stitch_algo.py`
- `core/workflow_manager.py`
- `App/main.py`
- template factory/dataclass 屬性路徑

所以這次是先建立 internalnum canonical layer 與相容轉譯，不硬拔 legacy watchpoint selector。下一步若要完全 internalnum-only，需要逐一改演算法 config、PDF config、UI selector、analysis lookup，否則會破壞現有功能。

**驗證**  
已跑過：

- `python -m py_compile ...` 通過
- `config/internalnum_config.yaml` / `config/system_config.yaml` YAML load 通過
- 範例驗證：`Front.toppoint1=0001`、`Back.toppoint1=1001`、`OpenBack=2001`、`OpenBackCrown=3001`、`Strap=4001`、`Box=5001`
- 範例驗證：`Front.macropoint1` 的 `std_1/std_2/hdr_1/focus_1` -> `internalnum2=0001/0002/0003/0004`
- service selector 驗證：`internalnum1=0004, internalnum2=0003` 解析為 `front.macropoint1.hdr_1`
- 臨時 SQLite insert/read 驗證 internalnums 可寫入 `image_assets`

殘留手動驗證建議：用一個舊 template 在 App 裡 load -> save as new version -> 實拍含 std/std/hdr/focus 的點 -> 檢查 SQLite、raw metadata、S3/DynamoDB、PDF 報告與 analysis pipeline。