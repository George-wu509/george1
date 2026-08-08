

|     |     |
| --- | --- |
|     |     |
|     |     |
|     |     |
|     |     |
|     |     |
|     |     |
|     |     |


## 1. `hardware_config.yaml` 的基準 AF

嚴格來說，[hardware_config.yaml (line 785)](D:/Provenance Laboratories projects/ImagingLibWatch/config/hardware_config.yaml:785) 沒有定義 `af_mode_id` 清單；它定義的是 mode 0 的基準值，以及底層粗對焦策略：

|`autofocus_option`|粗對焦策略|
|---|---|
|0|關閉粗對焦|
|1|Keyence MATLAB-like empirical AF|
|2|Vision Laplacian AF|
|3|Vision Tenengrad AF|
|4|Keyence2 legacy absolute AF|

目前 mode 0 的主要基準：

- 三台相機 `macro_cam_1`、`macro_cam_2`、`micro_cam` 都是 `coarse_af_option: 1`，即 Keyence 粗對焦。
- 三台都開啟液態鏡頭 FPGA AF。
- Keyence 量測起始 Z 為 `20.0 mm`。
- `force_reautofocus_each_capture: true`。
- 預設 AOI：`L_128x128`。
- Macro 液態鏡頭範圍：`-10 ～ +10 mA`、40 frames。
- Micro 液態鏡頭範圍：`-10 ～ +10 mA`、20 frames。
- 預設沒有 best-of-N、baseline validation、額外 Z boundary search。

## 2. 所有 `af_mode_id`

來源：[af_mode_config.yaml (line 8)](D:/Provenance Laboratories projects/ImagingLibWatch/config/af_mode_config.yaml:8)

|ID|名稱|對焦行為摘要|
|---|---|---|
|0|`default (simple af with liquid lens)`|完全使用 `hardware_config.yaml`；Keyence 粗 AF + 一次液態鏡頭 AF。|
|1|`crown`|基於 mode 0；只把 `micro_cam` 液態鏡頭範圍縮成 `-3 ～ +3 mA`。|
|2|`advanced`|Keyence 後執行兩階段液態 AF；第一次廣域掃描後，以結果為中心再做 ±5 mA 細掃描。三台相機皆啟用；另外 `macro_cam_2` Keyence 起始 Z=65。|
|3|`strap`|基於預設 AF；全域 Keyence 量測起點改成 Y=65、Z=65。|
|4|`strap endlink number`|mode 3，加上 `macro_cam_1` 液態鏡頭範圍 `-50 ～ +50 mA`。|
|5|`lume guarded af`|Micro lume 專用；AF 時使用 spotlight、10 ms exposure、±10 mA；啟用 FPGA result/baseline validation、baseline fallback，拒絕 boundary result。|
|6|`keyence autofocus only (all cameras)`|三台相機只做 Keyence coarse AF；完全關閉液態鏡頭 AF。|
|7|`side zaber 95 to 80 two-stage af`|關閉 Keyence；在 Y=80/85/90/95 與多個液態鏡頭電流組合取樣，再對最佳候選做局部 FPGA 細掃描。三台相機皆設定。|
|8|`side af two-stage wide then fine`|關閉 Keyence；Macro 先掃 ±60 mA、Micro 先掃 ±120 mA，再以結果為中心做 ±10 mA 細掃描。|
|9|`keyence+two-stage wide then fine`|完整繼承 mode 8，但重新開啟三台相機的 Keyence coarse AF；即 Keyence → 廣域液態 AF → 細液態 AF。|
|10|`side z 85 to 95 aoi 64x64`|R_X=90 side 專用；關閉 Keyence及液態鏡頭調整，在 Zaber Y=85～95 每 1 mm 用 `M_64x64` AOI 評分，停在最清楚的 Y。|
|11|`reuse side reference y with per-family offset`|不重新做 Keyence/液態 AF；重用 mode 10 找出的 side reference Y，topside/bottomside/sidecrown 各加 3 mm。若 reference 不存在，driver 會先跑來源 capture。|
|12|`strap side keyence z68.12 plus two-stage liquid af`|繼承 mode 8；只對 `macro_cam_1` 重新開啟 Keyence，量測 Z=68.12，套用 Strap side 專用 OUT1 線性公式，再做兩階段液態 AF。|

目前沒有在 canonical `internalnum_config.yaml` 直接配置的模式是：**4、6、7**。  
Mode 12 則由 Strap macro-camera-1 掃描設定使用。

## 3. `App/main.py` 拍照 task 與 AF mode

### 正式 WatchEntry 拍攝

`CaptureRoutineWorker → _capture_images_routine_internal()` 是正式 Front/Back/OpenBack/OpenBackCrown/Strap/Box 拍攝主流程。它不固定單一 mode，而是從目前 template 的 point/capture 取得 `af_mode_id`，再傳給 `execute_template_point()`。

目前 canonical internal-number 對應如下：

|View|目前 mode 分布|
|---|---|
|Front|預設 mode 0；`sidepoint1/2`、`macropoint3` 使用 mode 10；`micropoint12–15` 使用 mode 5；`glasspoint1` 使用 mode 1；`micropoint18–23` 使用 mode 11；`macropoint8–19` 使用 mode 8。|
|Back|預設 mode 0；`macropoint1` 使用 mode 9；`micropoint1` 使用 mode 2。|
|OpenBack|全部 mode 0。|
|OpenBackCrown|預設 mode 0；`micropoint3` 使用 mode 2。|
|StrapRightSide|預設 mode 3；`macropoint29–32` 使用 mode 8。部分自動 component task 會動態改寫，見下方。|
|Box|全部 mode 0。|

這些是目前 [internalnum_config.yaml (line 1)](D:/Provenance Laboratories projects/ImagingLibWatch/config/internalnum_config.yaml:1) 的 canonical defaults；舊 DB template 或使用者編輯過的 template 仍可能帶不同 mode。

### Template pre-capture

`TemplatePreCaptureWorker` 定義於 [App/main.py (line 1507)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:1507)。

|Task / 影像|AF mode|AF 開關|
|---|---|---|
|`template_create_front`|0|Keyence + liquid AF|
|`template_create_side1`|8|`use_autofocus=false`、`use_liqud_af=true`，所以只做 mode 8 液態鏡頭 wide/fine|
|`template_create_side2`|8|同上|
|`template_create_side3`|8|同上|
|`template_create_side4`|8|同上|
|`template_create_back`|0|繼承 internalnum `1001`|
|`template_create_openback`|0|繼承 internalnum `2001`|
|`template_create_openbackcrown`|0|繼承 internalnum `3001`|
|`template_create_box`|0|繼承 internalnum `5001`|

### Template point / scratch capture

`template_point_capture:{side}.{point}`、`_execute_prepared_point_capture()`：

- 使用目前選取 capture 的 `af_mode_id`。
- 一般 precedence 為 capture 設定優先。
- 若 point 的 `af_unity=1`，point-level 的 `af_mode_id`、AF 開關及 XYZ modifiers 會重新蓋過 capture。
- 因此實際模式就是上方 canonical point 分布，或使用者/template 儲存的 override。

### Rehaut 自動拍攝

`rehaut_auto_capture:0039–0050`：

- 對應 `Front.macropoint8–19`。
- 全部使用 **mode 8**。
- Standard 後面的 HDR capture 會重用 Standard 的 pose/focus，不重新執行 AF。

### Front template suggestion OCR preflight

`front_template_suggestion → Front.sidepoint1`：

- Payload 使用 **mode 10**。
- Canonical point 同時是 `use_autofocus=false`、`use_liqud_af=false`；因此需要注意，mode ID 雖是 10，實際 AF 是否執行仍受 capture-level master switches 及 driver 的 specialized side-scan routing控制。

### Strap button 213 / 217

這兩個 task 有兩套可切換 pipeline。

**Legacy macro_cam_2 stitched pipeline**

- Button 213：主要拍 `4001–4021`。
- Button 217：主要拍 `4022–4028`。
- 這些 canonical points 都是 **mode 3 `strap`**。

**Macro-camera-1 scan pipeline**

設定來源：[strap_macro_cam1_scan.yaml (line 295)](D:/Provenance Laboratories projects/ImagingLibWatch/config/strap_macro_cam1_scan.yaml:295)

|Strap scan view|Anchor mode|Tile mode|
|---|---|---|
|front|3|8|
|side|12|8|
|back|3|8|
|9clock|12|8|

也就是：

- Front/Back anchor：mode 3。
- Side/9 O’Clock anchor：mode 12，使用 Strap side Z=68.12 Keyence 公式。
- 所有非 anchor tiles：mode 8。

### Strap component capture

`strap_component_capture` 會在 [App/main.py (line 14123)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:14123) 動態改寫 mode：

|Component capture 角色|AF mode|實際開關|
|---|---|---|
|Micro camera point|3|Mechanical/Keyence + liquid AF|
|Dynamic group anchor|3|Mechanical/Keyence + liquid AF|
|Dynamic group non-anchor|8|不跑 mechanical AF；重用預測 Z/locked current，必要時依流程做液態 AF|
|固定點|使用原 point/capture mode|不由 planner 改寫|
|`4029/4030` 固定 reference|mode 8 保留|明確關閉 mechanical 與 liquid AF，直接用已驗證 pose/current|

## 4. 容易誤認為拍照 task 的類別

- `CaptureTaskRunnable` 名字雖有 Capture，但實際只處理 simulation image、儲存、DB/sync 與分析；production mode 直接 skip，**不操作實體相機，也沒有 AF mode**。
- `WorkflowTaskWorker` 是通用背景 wrapper，本身沒有 AF mode；mode 由包在裡面的 capture job 決定。
- `control_autofocus` 不儲存正式照片，但會用目前選取 point/capture 的 `af_mode_id` 暫時套用對應設定。
- Remote `manual_capture(watchid, cam_id, exposure)` 沒有 `af_mode_id` 參數。
- `_execute_point_capture()` 是舊的 direct camera 路徑，目前 `main.py` 中沒有 call site，也沒有把 `af_mode_id` 傳入 driver。