

|                                                 |     |
| ----------------------------------------------- | --- |
| [[#### af_mode_config.yaml裡面定義的所有af_mode並列表整理]] |     |
| [[#### 列表整理每個af_mode_id(0~12)類別]]               |     |
| [[#### 在執行App的create template拍照的autofocus]]     |     |
|                                                 |     |
|                                                 |     |
|                                                 |     |
|                                                 |     |

#### af_mode_config.yaml裡面定義的所有af_mode並列表整理
```
請幫我整理目前系統的所有config/hardware_config.yaml, config/af_mode_config.yaml裡面定義的所有af_mode並列表整理, 並整理App main.py裡面有拍照的tasks各是使用哪個af_mode
```
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






#### 列表整理每個af_mode_id(0~12)類別
```
請列表整理每個af_mode_id(0~12)類別, 個有哪些在internalnum_config裡面的internalnum1屬於這些類別. 而在真正拍照時, 甚麼時候會不依照這些在internalnum_config裡面point的設定, 而是依照其他的設定進行拍照?
```

以下依照目前工作樹的 [internalnum_config.yaml (line 1)](D:/Provenance Laboratories projects/ImagingLibWatch/config/internalnum_config.yaml:1) 重新統計。這裡列的是 `point_defaults.af_mode_id`，不是 capture-level override。

範圍例如 `0007–0021` 代表首尾皆包含。

## 一、各 `af_mode_id` 對應的 `internalnum1`

|ID|名稱|數量|`internalnum1`|
|---|---|---|---|
|0|default|76|`0001`, `0004–0005`, `0007–0021`, `0026–0031`, `0051–0054`; `1001`, `1003`, `1005–1011`; `2001–2012`; `3001–3008`, `3010–3017`; `6001–6011`|
|1|crown|1|`0032`|
|2|advanced|2|`1004`, `3009`|
|3|strap|268|`4001–4028`; `4101–4120`; `4201–4220`; `4301–4320`; `4401–4420`; `4501–4520`; `4601–4620`; `4701–4720`; `4801–4820`; `5001–5020`; `5101–5120`; `5201–5220`; `5301–5320`|
|4|strap endlink number|0|無|
|5|lume guarded af|4|`0022–0025`|
|6|keyence autofocus only|0|無|
|7|side zaber 95 to 80 two-stage af|0|無|
|8|side AF two-stage wide then fine|16|`0039–0050`, `4029–4032`|
|9|keyence + two-stage wide then fine|1|`1002`|
|10|side Z 85 to 95 AOI 64×64|3|`0002`, `0003`, `0006`|
|11|reuse side reference Y|6|`0033–0038`|
|12|strap side Keyence Z=68.12 + two-stage liquid AF|0|無|

AF mode 名稱來源：[af_mode_config.yaml (line 8)](D:/Provenance Laboratories projects/ImagingLibWatch/config/af_mode_config.yaml:8)。

### 依 View 展開

- Front `0xxx`
    
    - mode 0：`0001`, `0004–0005`, `0007–0021`, `0026–0031`, `0051–0054`
    - mode 1：`0032`
    - mode 5：`0022–0025`
    - mode 8：`0039–0050`
    - mode 10：`0002`, `0003`, `0006`
    - mode 11：`0033–0038`
- Back `1xxx`
    
    - mode 0：`1001`, `1003`, `1005–1011`
    - mode 2：`1004`
    - mode 9：`1002`
- OpenBack `2xxx`
    
    - mode 0：`2001–2012`
- OpenBackCrown `3xxx`
    
    - mode 0：`3001–3008`, `3010–3017`
    - mode 2：`3009`
- Strap `4xxx/5xxx`
    
    - mode 3：上表中的大部分 Strap ranges
    - mode 8：`4029–4032`
- Box `6xxx`
    
    - mode 0：`6001–6011`

### 沒有 `af_mode_id` 的 internalnum

`7001–7013` 與 `8001–8010` 沒有 `point_defaults.af_mode_id`：

- `7xxx` 是 Material/Alloy metadata。
- `8xxx` 是外部量測資料。
- 它們不是一般相機 WatchPoint，所以不能視為 mode 0。

---

## 二、重要觀念：正式拍照不是每次重新讀取 internalnum point defaults

正式拍照時，[App/main.py (line 33312)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:33312) 呼叫 `apply_internalnums_to_watchview()`，但這個函式只補齊：

- `internalnum1`
- `internalnum2`
- point/capture identity

它明確不會把 `internalnum_config.yaml` 的 AF、曝光、燈光等 defaults 強制覆蓋到已載入的 template；可見 [internalnum_config.py (line 1470)](D:/Provenance Laboratories projects/ImagingLibWatch/DB/templates/internalnum_config.py:1470)。

所以一般正式拍照的真正來源是：

```
internalnum_config
    ↓ 建立新 point/capture 時提供初始值
已儲存的 current template
    ↓
point 設定 + capture 設定
    ↓
App runtime 特殊政策
    ↓
hardware_config + af_mode_config
    ↓
execute_template_point()
```

也就是說，修改 `internalnum_config.yaml` 不一定會改變既有 DB template 下一次拍照的 AF mode。既有 template 若已保存不同的 `af_mode_id`，通常會依 template 值執行。

---

## 三、Point 與 Capture 的優先順序

正式 payload 的合併邏輯在 [App/main.py (line 8274)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:8274)。

### `af_unity = 0`

Capture-level 設定優先：

```
capture.af_mode_id
    > point.af_mode_id
```

Capture 可自行覆蓋：

- `af_mode_id`
- `use_autofocus`
- `use_liqud_af`
- `Xmod/Ymod/Zmod`
- 相機、曝光、燈光等

目前 `internalnum_config` 有一個實際差異：

|internalnum1|Capture|Point mode|Capture mode|真正使用|
|---|---|---|---|---|
|`0008` Front.macropoint5|`0002`|0|1|**1**，因為 `af_unity=0`|

相關設定位於 [internalnum_config.yaml (line 353)](D:/Provenance Laboratories projects/ImagingLibWatch/config/internalnum_config.yaml:353)。

### `af_unity = 1`

Point-level AF policy 強制優先：

```
point.af_mode_id
    > capture.af_mode_id
```

同時 point-level 的下列設定都會覆蓋 capture：

- `af_mode_id`
- `use_autofocus`
- `use_liqud_af`
- `Xmod`
- `Ymod`
- `Zmod`

目前有一個看似不一致但會被 `af_unity` 修正的資料：

|internalnum1|Point mode|Capture mode|`af_unity`|真正使用|
|---|---|---|---|---|
|`3008`|0|2|1|**0**|

設定位置：[internalnum_config.yaml (line 4459)](D:/Provenance Laboratories projects/ImagingLibWatch/config/internalnum_config.yaml:4459)。

---

## 四、會不依照 internalnum point AF 設定的拍照情況

### 1. 已儲存 Template 與 internalnum defaults 不同

這是最常見情況。

正式拍攝直接讀 `current_template.watchView` 裡面的 point/capture 物件，不會每次把 internalnum defaults 重新套上去。

因此以下來源都可能讓拍照值不同：

- 舊版本 DB template
- 從 YAML 匯入的 template
- App UI 修改並儲存過的 point/capture
- Capture-level 自訂值
- 由舊資料 migration 產生的值

只有「建立或 materialize point/capture」時，才會主動以 `internalnum_config` 作為 default。

### 2. Capture-level `af_mode_id` 覆蓋 Point

當 `af_unity=0` 時，單張 Standard/HDR capture 可以有自己的 AF mode。

目前明確案例是：

- `0008/0002`：point mode 0，capture mode 1，實拍使用 mode 1。

所以只看 point 分類表，不能百分之百代表每一張 capture。

### 3. `focus_capture` 使用另一套 AF 設定

一個 point 可以有：

- `standard_captures`
- `hdr_captures`
- `focus_capture`

若存在 `focus_capture`，App 會建立獨立的 `focus_hardware` payload；對焦階段可以使用 focus capture 的：

- 相機
- 曝光
- 燈光
- `af_mode_id`
- `use_autofocus`
- `use_liqud_af`

完成對焦後才用正式 capture 的曝光/燈光拍照。也就是「對焦所用 mode」和「最終影像 capture 上看到的 mode」可能不是同一組設定。

### 4. HDR 重用前一張 Standard 的對焦

當同一 point 的 HDR 緊接成功的 Standard capture，App 會設定：

- `reuse_previous_focus = true`
- `skip_xyz_move = true`
- `use_autofocus = 0`
- `use_liqud_af = false`
- 移除 `focus_hardware`

見 [capture_policy.py (line 50)](D:/Provenance Laboratories projects/ImagingLibWatch/core/capture_policy.py:50)。

此時 HDR 雖然仍帶有原本的 `af_mode_id`，但不會重新執行該 mode，而是直接重用 Standard 的最終 Z 與液態鏡頭鎖定結果。

### 5. HDR 指定另一張 Standard 作為 AF 來源

`hdr_config.yaml` 可以透過 `hdr_use_std_af` 類型設定，指定另一張 Standard capture 作為 HDR 的 `focus_hardware`。

這時 HDR 對焦依照被引用的 Standard point/capture，而不是 HDR 自己或當前 point 的 internalnum AF 設定。

相關組裝位置：[App/main.py (line 33704)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:33704)。

### 6. Bezel runtime policy

當 point component 被辨識為 `Bezel`，App 在送進 driver 前強制改成：

- `use_autofocus = false`
- `use_liqud_af = true`
- `has_glass = false`
- `angle_pose_compensation = false`

即使 template/internalnum 要求 Keyence mechanical AF，也會被關掉，只執行液態鏡頭 AF。

見 [App/main.py (line 7366)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:7366)。

### 7. Strap component 自動拍攝

`strap_component_capture` 不單純依照 Strap internalnum 的 mode 3。App 會依當次 component plan 動態改寫 point、capture 和 focus capture：

|Runtime 角色|強制 mode|Mechanical AF|Liquid AF|
|---|---|---|---|
|Micro screw point|3|開|開|
|Macro anchor|3|開|開|
|Dynamic non-anchor|8|關|關，直接使用預測 Z/locked current|
|Fixed point|不改 mode|依原設定|依原設定|
|`4029/4030` reference|mode 8 保留|關|關|

程式位置：[App/main.py (line 15122)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:15122)。

因此，許多 `internalnum1` 雖然 point default 是 mode 3，作為 dynamic non-anchor 拍攝時會被改為 mode 8，而且兩種 AF 都關閉。

另外 Strap component anchor 還會注入 runtime-only `liquid_af_overrides`，因此即使 ID 仍然是 mode 3，液態 AF 的 fallback/validation 行為仍可能和純 mode 3 不同。

### 8. Strap Macro Cam 1 scan pipeline

Button 213/217 的 Macro Cam 1 pipeline 使用 [strap_macro_cam1_scan.yaml (line 295)](D:/Provenance Laboratories projects/ImagingLibWatch/config/strap_macro_cam1_scan.yaml:295)，不是 internalnum point mode：

|Scan view|Anchor|Tile|
|---|---|---|
|front|mode 3|mode 8|
|side|mode 12|mode 8|
|back|mode 3|mode 8|
|9clock|mode 12|mode 8|

因此 mode 12 雖然在 `internalnum_config` 中沒有任何 point，仍會在 Side/9 O’Clock anchor 正式拍照中使用。

### 9. Template pre-capture 使用 `template_create_config.yaml`

建立模板的 Front overview sequence 中：

- `template_create_front`：由 `0001` defaults 得到 mode 0。
- `template_create_side1–4`：直接由 `template_create_config.yaml` 指定 mode 8。
- Back/OpenBack/OpenBackCrown/Box overview：透過其指定 internalnum defaults。

所以 side1–4 pre-capture 不屬於 normal internalnum point 拍照，是由 template-create 專用 config 控制。

### 10. Sidepoint 額外角度拍攝

Sidepoint 主影像完成後，額外 R_Z angle captures 會重用中心 XYZ，並強制：

- `use_autofocus = 0`
- `use_liqud_af = 1`
- `has_glass = 0`
- 清除 `focus_hardware`
- XYZ modifiers 歸零

見 [App/main.py (line 29472)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:29472)。

所以中心影像可能完整執行 mode 10；其他角度影像則只跑液態鏡頭 AF，不重新做 mechanical/Keyence AF。

### 11. 其他 R_Z angle capture 路徑

`core/workflow_manager.py` 的另一個 angle capture 路徑會直接強制：

- `use_autofocus = 0`
- `use_liqud_af = 0`
- `has_glass = 0`
- 移除 `focus_hardware`

見 [workflow_manager.py (line 1247)](D:/Provenance Laboratories projects/ImagingLibWatch/core/workflow_manager.py:1247)。

這些額外角度影像完全不執行 point 指定的 AF mode。

### 12. Capture-level `liquid_af_overrides`

即使最終 `af_mode_id` 沒改，capture payload 還能帶 `liquid_af_overrides`。

Driver 的實際合併順序是：

```
hardware_config 基準
→ af_mode_config 對應 mode overrides
→ capture.liquid_af_overrides
```

最後一層可以再改變：

- FPGA/Software fallback
- 掃描範圍
- frame count
- validation
- boundary handling
- AOI 等

Driver 合併位置：[unified_driver.py (line 10008)](D:/Provenance Laboratories projects/ImagingLibWatch/Controller/hardware_drivers/unified_driver.py:10008)。

---

## 五、反過來：哪些情況會強制重新依照 internalnum YAML？

有兩個特別嚴格的 Strap reference：

- `4029`
- `4030`

在 prepared capture 時，App 會重新讀取 point defaults 和 capture defaults，防止舊 template/capture 改掉：

- XYZ
- 相機
- 電流
- 曝光
- 燈光
- AF
- rotation

見 [App/main.py (line 8730)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:8730)。

這兩個是少數「正式拍照前再次以 internalnum YAML 為 authoritative」的例外。

## 結論

判斷真正拍照使用哪個 AF，不應只看 `internalnum_config.point_defaults.af_mode_id`。正確順序是：

1. 確認目前載入的 template point 值。
2. 檢查該張 capture 自己的 `af_mode_id`。
3. 檢查 point 的 `af_unity`。
4. 檢查是否有 `focus_capture/focus_hardware`。
5. 檢查 HDR 是否重用 Standard focus。
6. 檢查 Bezel、Sidepoint、Strap component、Strap Macro1 等 runtime policy。
7. 檢查 `use_autofocus`、`use_liqud_af`；mode ID 存在不代表兩個 AF 階段都會執行。
8. 最後檢查 `liquid_af_overrides`。




#### 在執行App的create template拍照的autofocus
```
而在執行App的create template拍照及把autofocus設定存進去template時, 甚麼時候會不依照這些在internalnum_config裡面point的設定, 而是依照其他的設定進行拍照?
```
最核心的結論是：

`internalnum_config.yaml` 在 Create Template 裡是「初始化／補值來源」，不是持續綁定的唯一真值。實際拍照會先使用目前記憶體中的 point/capture 設定，再套用特殊拍照策略；儲存 template 時，也不會重新拿 `internalnum_config` 覆蓋一次，而是直接序列化目前物件。

## 一般 Create Template 拍照的 AF 優先順序

對一般 point 的正式拍照：

1. `af_unity = 0`
    
    - `capture.af_mode_id`
    - `capture.use_autofocus`
    - `capture.use_liqud_af`
    
    優先於 point 設定。
    
2. `af_unity = 1`
    
    - `point.af_mode_id`
    - `point.use_autofocus`
    - `point.use_liqud_af`
    - `point.Xmod/Ymod/Zmod`
    
    會強制覆蓋 capture 設定。
    

這套規則實作在 [_canonical_capture_payload() (line 8274)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:8274)。

因此，即使 `internalnum_config` 的 point 設定是 `af_mode_id: 0`，若：

- `af_unity=0`
- 該 point 的某個 `internalnum2` capture 是 `af_mode_id: 1`

真正拍這張 capture 時仍會使用 mode 1。這不是忽略 internalnum，而是使用同一個 internalnum 下更細層級的 capture 設定。

## 會改用其他設定的情況

|情況|真正拍照使用來源|是否會存進 template|
|---|---|---|
|Copy from existing template|原 template 內既有 point/capture AF|會|
|既有 capture 已被 UI 或程式修改|記憶體內的 capture 值|會|
|`af_unity=0`|capture AF，非 point AF|會同時保留 point/capture 各自數值|
|`focus_capture` 存在|對焦階段使用 `focus_capture`，曝光則使用目前 Standard/HDR capture|`focus_capture` 本身會存|
|UI 手動新增 capture|UI 值及 dataclass 預設：通常 mode 0、mechanical AF 開、liquid AF 開|會|
|Strap 動態 component|程式產生的 anchor/non-anchor 策略|會|
|Bezel|強制 mechanical AF 關、liquid AF 開、`has_glass=false`|視建立路徑而定|
|HDR 接續成功的 Standard|重用前一張 Standard focus，當次不重新 AF|不會把這個 runtime override 存回 HDR|
|Template pre-capture|`template_create_config.yaml`，有 internalnum 時再由 internalnum 覆蓋|不會成為一般 point 的 AF 設定|
|Strap Macro Cam1 scan|`strap_macro_cam1_scan.yaml` 的 anchor/tile AF|不會直接成為一般 point 設定|

### 1. Copy from existing template

Create Template 選擇既有 template 當來源時，只載入原 template，沒有再次用 `internalnum_config` 全面覆蓋。

所以原 template 裡已修改過的：

- `af_mode_id`
- `use_autofocus`
- `use_liqud_af`
- `af_unity`
- `cammag`
- `focus_capture`

會繼續被使用並存入新 template。對照 [_on_template_create_click() (line 17663)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:17663)。

### 2. 從零建立，但 capture 已存在或經過 UI 修改

Scratch 初次建立時確實會用：

```
override_existing=True
```

把 internalnum defaults 強制寫入，見 [default_template_factory.py (line 357)](D:/Provenance Laboratories projects/ImagingLibWatch/DB/templates/default_template_factory.py:357)。

但後續選擇 point 或 finalize point 時多數使用：

```
override_existing=False
```

也就是只補空值，不覆蓋已存在值，見：

- [_materialize_scratch_point_defaults() (line 5537)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:5537)
- [_finalize_pending_point() (line 41753)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:41753)
- [apply_internalnum_defaults_to_point() (line 997)](D:/Provenance Laboratories projects/ImagingLibWatch/DB/templates/internalnum_config.py:997)

所以只要 capture 已被 UI／程式建立，internalnum 的 point AF 就不一定能蓋回去。

尤其手動 Add Capture 會建立自己的 CaptureCondition，然後 finalize 時因 `override_existing=False` 而保留，見 [_add_capture_to_pending_point() (line 41625)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:41625)。

### 3. `focus_capture` 與曝光 capture 不同

若 point 有 `focus_capture`：

- Autofocus 階段使用 `focus_capture.af_mode_id`
- 最終曝光使用目前 Standard/HDR capture
- 兩者可以是不同 mode、camera、cammag 或 AF switch

建立 `focus_hardware` 的位置在 [App/main.py (line 8998)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:8998)。

所以不能只看 point 的 `af_mode_id` 判斷真正對焦採用哪個 mode。

### 4. Strap 動態 component 會改寫 internalnum AF

Strap component 流程會根據角色動態改 AF：

- Micro screw：mode 3，mechanical AF 開，liquid AF 開
- Macro anchor：mode 3，mechanical AF 開，liquid AF 開
- Dynamic non-anchor：mode 8，mechanical AF 關，liquid AF 關，使用預測 focus plane/current
- `4029/4030`：兩種 AF 都關閉，使用固定 pose/current

程式會直接修改 point 和 capture，所以這些 mode/switch 會存入 template，見 [_configure_strap_component_focus_policy() (line 15052)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:15052)。

但動態注入的 `liquid_af_overrides` 屬於 runtime 附加屬性，不是 dataclass schema 欄位，不一定隨 template 序列化保存。

### 5. Bezel 規則

Bezel 正式拍照會強制：

```
use_autofocus = false
use_liqud_af = true
has_glass = false
angle_pose_compensation = false
```

見 [_apply_bezel_capture_policy() (line 7366)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:7366)。

需要分兩種情況：

- 新建 Bezel 且完成 Bezel surface preparation：point/capture 也會被修改，因此會存入 template。
- 舊 template Bezel 或沒有重新完成 preparation：最後拍照仍會 runtime 強制套用上述規則，但原 template 欄位可能仍保持舊值。

因此可能出現「template 顯示 mechanical AF 開，但真正 Bezel 拍照時 mechanical AF 關」的情況。

### 6. HDR 重用 Standard focus

Create Template 同一 point 先成功拍 Standard，再拍 HDR 時，HDR 會：

```
reuse_previous_focus = true
skip_xyz_move = true
use_autofocus = false
use_liqud_af = false
```

見 [capture_policy.py (line 50)](D:/Provenance Laboratories projects/ImagingLibWatch/core/capture_policy.py:50)。

這只修改當次送給 driver 的 payload，不會回寫 HDR capture object。因此：

- template 內 HDR 可能仍顯示 AF 開啟或保留原 mode
- 但當次 Create Template HDR 實際不重新 Autofocus

如果前一張 Standard 沒有成功，HDR 才會回到自己的正常 AF 設定。

### 7. Template pre-capture 不一定看 internalnum point

進入 Create Template 前的 overview/pre-capture 影像由 `template_create_config.yaml` 控制。

合併順序是：

```
point_d = template_create_config point_defaults
point_d.update(internal_point_defaults)
```

所以：

- 有對應 `internalnum1`：internalnum 設定勝出
- 沒有 internalnum：使用 `template_create_config.yaml`

見 [TemplatePreCaptureWorker (line 1590)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:1590)。

例如部分 `template_create_side*` 會使用 `template_create_config.yaml` 指定的 mode 8，而非某個正式 watch point 的 internalnum。這些 overview 拍照設定不會直接存成一般 point AF。

### 8. Strap Macro Cam1 scan

Strap overview/拼接掃描使用 [strap_macro_cam1_scan.yaml (line 312)](D:/Provenance Laboratories projects/ImagingLibWatch/config/strap_macro_cam1_scan.yaml:312)：

- Front/Back anchor：mode 3
- Side/9-clock anchor：mode 12
- 一般 tile：mode 8

這些是掃描影像的 AF 策略，不是普通 internalnum point 的設定。

## Autofocus 結果是否存回 template

正式 `_execute_prepared_point_capture()` 拍照完成後：

- driver 回傳的 `final_z`
- liquid lens `locked_current`

只記錄為執行結果。  
程式會把 point/capture 的 pose 恢復為拍照前 requested pose，避免將 Autofocus 結果當成 template 設定，見 [App/main.py (line 9063)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:9063)。

但有一個例外：在 Control UI 按下 Autofocus 按鈕時，成功後會直接寫入：

```
point.Z = best_z
capture.Z = best_z
capture.cammag = best_mA
```

見 [App/main.py (line 28589)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:28589)。這些值之後會存進 template。

## 儲存時的最終原則

[_serialize_template_to_payload() (line 12628)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:12628) 只序列化目前的 template object，沒有在 Save 前重新套一次 `internalnum_config`。

所以最終可以簡化成：

```
internalnum_config
    ↓ 初始建立／缺值補齊
記憶體中的 point + capture
    ↓ UI、copy template、Strap/Bezel preparation 修改
存入 template

真正拍照
    = 已存 point/capture
    + focus_capture
    + Bezel/HDR/Strap 等 runtime policy
```

另外，Front 的 `0032`、`0039–0050` 會在準備拍照時強制重新套用 internalnum 設定，屬於「特別避免偏離 internalnum」的保護案例，見 [App/main.py (line 8258)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:8258)。





```
那特別幫我完整整理在Strap時, 包括macro_cam_2拍的stitch用的images(front, side, back, 9oclock side)的對焦, macro_cam_1拍的stitch用的images(front, side, back, 9oclock side)的對焦, 以及使用macro_cam_2的stitched image(front, side, back, 9oclock side)用SAM model推測links, endlinks, screw座標之後拍照的images的對焦, 使用macro_cam_1的stitched image(front, side, back, 9oclock side)用SAM model推測links, endlinks, screw座標之後拍照的images的對焦, 以及兩個endlinks reference images的對焦. 這些autofocus是用哪個af_modes, 是否有用keyence af, keyence af的z start or y start, 是否有用liquid lens af等等.
```

先講最重要的結論：

- 目前正式預設是 **macro_cam_2 legacy stitch pipeline**，因為 [`use_macro_cam_1_pipeline: false` (line 7)](D:/Provenance Laboratories projects/ImagingLibWatch/config/strap_macro_cam1_scan.yaml:7)。
- macro_cam_1 stitch pipeline 已完整實作，但必須把該開關改成 `true` 才會取代 macro_cam_2 的 4001–4028 流程；兩套不會在同一次 Strap session 混用。
- SAM stitched source 用 macro_cam_2 或 macro_cam_1，主要影響「座標轉換與可選 component」；SAM 後的正式 component 照片，links/endlinks 都是用 `macro_cam_1`，screws 用 `micro_cam`。

## Keyence start 軸判讀原則

Strap 四個 view 的物理方向如下：

|View|`R_X`|stitched 平面|對焦法向軸|
|---|---|---|---|
|Front / underside|90°|XZ|Stage Y|
|Side / 3-clock|0°|XY|Stage Z|
|Back / outer surface|90°|XZ|Stage Y|
|9-clock side|0°|XY|Stage Z|

定義在 [strap_registration.py (line 13)](D:/Provenance Laboratories projects/ImagingLibWatch/core/strap_registration.py:13)。

因此：

- Front、Back 的 Keyence AF，實際是沿 **Y 軸方向**決定表面焦點。
- Side、9-clock 的 Keyence AF，實際是沿 **Z 軸方向**決定焦點。
- 程式仍可能同時移動 Y/Z 去保持光軸；這裡的「Y start／Z start」指的是主要對焦參考軸。

另外 Keyence start 設定的解析優先順序是：

1. camera-specific `keyence1_measure_start_*`
2. global `af_settings.keyence1_measure_start_*`
3. 目前 point pose

見 [unified_driver.py (line 2634)](D:/Provenance Laboratories projects/ImagingLibWatch/Controller/hardware_drivers/unified_driver.py:2634)。

---

# 一、macro_cam_2 拍 stitch source images

這是目前預設啟用的 legacy pipeline：

- Front：internalnum 4001–4007
- Side：4008–4014
- Back：4015–4021
- 9-clock：4022–4028

Stitch group 對應定義在 [watchband_stitch.yaml (line 31)](D:/Provenance Laboratories projects/ImagingLibWatch/config/watchband_stitch.yaml:31)。

所有這些點都是：

```
camera = macro_cam_2
af_mode_id = 3
use_liqud_af = true
af_unity = 1
```

但 `use_autofocus` 並不是每張都開，所以不能只看到 mode 3 就認為每張都有 Keyence AF。

## macro_cam_2 stitch AF 明細

|View|Source IDs|Keyence AF images|其他 images|Liquid-lens AF|
|---|---|---|---|---|
|Front|4001–4007|4001、4002|4003–4007 不跑 Keyence|每張都跑|
|Side|4008–4014|4009|4008、4010–4014 不跑 Keyence|每張都跑|
|Back|4015–4021|4016|4015、4017–4021 不跑 Keyence|每張都跑|
|9-clock|4022–4028|4023|4022、4024–4028 不跑 Keyence|每張都跑|

來源設定例如：

- [4001 (line 4920)](D:/Provenance Laboratories projects/ImagingLibWatch/config/internalnum_config.yaml:4920)
- [4002 (line 4966)](D:/Provenance Laboratories projects/ImagingLibWatch/config/internalnum_config.yaml:4966)
- [4008 (line 5242)](D:/Provenance Laboratories projects/ImagingLibWatch/config/internalnum_config.yaml:5242)
- [4009 (line 5288)](D:/Provenance Laboratories projects/ImagingLibWatch/config/internalnum_config.yaml:5288)
- [4015 (line 5564)](D:/Provenance Laboratories projects/ImagingLibWatch/config/internalnum_config.yaml:5564)
- [4016 (line 5610)](D:/Provenance Laboratories projects/ImagingLibWatch/config/internalnum_config.yaml:5610)
- [4022 (line 5886)](D:/Provenance Laboratories projects/ImagingLibWatch/config/internalnum_config.yaml:5886)
- [4023 (line 5932)](D:/Provenance Laboratories projects/ImagingLibWatch/config/internalnum_config.yaml:5932)

實際執行前還會用 `override_existing=True` 強制重新 materialize 4001–4028，見 [_materialize_strap_scan_point() (line 39997)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:39997)。

## Keyence start

### Front、Back

- AF mode：3
- 物理焦點軸：Y
- `keyence1_measure_start_y = 65.0`
- Keyence：只有上述指定 anchor images 執行
- Liquid lens：每張執行

Mode 3 在 [af_mode_config.yaml (line 37)](D:/Provenance Laboratories projects/ImagingLibWatch/config/af_mode_config.yaml:37) 同時設 start Y/Z 為 65；在 `R_X=90°` 時，有效重點是 Y=65。

### Side、9-clock

- AF mode：3
- 物理焦點軸：Z
- `keyence1_measure_start_z = 65.0`
- Keyence：4009／4023
- Liquid lens：每張執行
- 沒有額外的 ±Y Keyence search

macro_cam_2 沒有 camera-specific mode-3 start Z 覆蓋，因此使用 mode 3 的 global Z=65。

## Liquid-lens 細節

macro_cam_2 legacy stitch 每張都設 `use_liqud_af=true`：

- FPGA liquid AF
- 預設 current 範圍約 `-10 ～ +10 mA`
- 預設沒有 software fallback
- 落在 boundary 可判定為失敗

見 [hardware_config.yaml (line 1179)](D:/Provenance Laboratories projects/ImagingLibWatch/config/hardware_config.yaml:1179)。

所以 legacy pipeline 的非 Keyence 圖片不是 fixed-current capture，而是：

```
Keyence：不跑
Liquid lens：仍然每張重跑
```

## Extension 對實際圖片數量的影響

4001、4007、4008、4014、4015、4021、4022、4028 是否納入，會受左右延伸檢測影響。

一般沒有 extension 時通常是：

- Front：4002–4006
- Side：4009–4013
- Back：4016–4020
- 9-clock：4023–4027

因此 4002／4009／4016／4023 是各 view 最穩定會存在的 Keyence anchor；4001 是 Front 額外 extension anchor。

---

# 二、macro_cam_1 拍 stitch source images

目前設定檔存在，但預設關閉。

這套不使用 4001–4028 的固定 X 點，而是：

1. Keyence 探測 Strap 左右端點
2. 依 14 mm step 動態建立 scan positions
3. 每個 view 建立 5 個 AF anchors：10%、30%、50%、70%、90%
4. 其他 tile 用 anchors 插值出的 Z/Y 與 liquid current
5. anchor 圖片直接重用於 stitch

主要設定在 [strap_macro_cam1_scan.yaml (line 283)](D:/Provenance Laboratories projects/ImagingLibWatch/config/strap_macro_cam1_scan.yaml:283)。

## macro_cam_1 stitch AF 明細

|View|Anchor mode|Anchor Keyence|Anchor liquid AF|非 anchor tile|
|---|---|---|---|---|
|Front|3|有，Y start 65|有|mode 8，但兩種 AF 都關|
|Side|12|有，calibrated Z start 68.12|有|mode 8，但兩種 AF 都關|
|Back|3|有，Y start 65|有|mode 8，但兩種 AF 都關|
|9-clock|12|有，calibrated Z start 68.12|有|mode 8，但兩種 AF 都關|

設定位置：

- Front：[anchor 3 / tile 8 (line 312)](D:/Provenance Laboratories projects/ImagingLibWatch/config/strap_macro_cam1_scan.yaml:312)
- Side：[anchor 12 / tile 8 (line 361)](D:/Provenance Laboratories projects/ImagingLibWatch/config/strap_macro_cam1_scan.yaml:361)
- Back：[anchor 3 / tile 8 (line 417)](D:/Provenance Laboratories projects/ImagingLibWatch/config/strap_macro_cam1_scan.yaml:417)
- 9-clock：[anchor 12 / tile 8 (line 463)](D:/Provenance Laboratories projects/ImagingLibWatch/config/strap_macro_cam1_scan.yaml:463)

## Front、Back anchors

- mode 3
- Keyence AF：有
- 對焦主軸：Y
- start Y：65 mm
- Liquid AF：有
- Anchor 位置：10%、30%、50%、70%、90%
- 設定有 Y attempts：`[0, -1, +1]`

這裡的 `[0,-1,+1]` 是 anchor capture 候選位置；不是每張 tile 都掃 Y。

## Side、9-clock anchors

使用 mode 12：

```
strap side keyence z68.12 plus two-stage liquid af
```

見 [af_mode_config.yaml (line 371)](D:/Provenance Laboratories projects/ImagingLibWatch/config/af_mode_config.yaml:371)。

實際行為：

- camera：macro_cam_1
- Keyence：有
- Keyence start Z：68.12 mm
- 使用專屬 no-glass 線性公式
- Liquid AF：有
- Anchor 位置：5 個
- capture 本身的 Y 固定在 view 的 calibrated camera line

50% 中央 anchor 會先做 Keyence probe Y 搜尋：

```
0, +1, -1, +2, -2, +3, -3 mm
```

見 [strap_macro_cam1_scan.yaml (line 369)](D:/Provenance Laboratories projects/ImagingLibWatch/config/strap_macro_cam1_scan.yaml:369)。

這個 Y search 是：

- 只用在 Side／9-clock 中央 anchor
- 用來找得到有效 Keyence OUT1 的 probe Y
- 不是以影像 sharpness 掃 Y
- 最終 Keyence focus 仍是 Z=68.12 公式

## macro_cam_1 anchor liquid AF

每個 anchor：

1. Keyence mechanical AF
    
2. FPGA liquid-lens AF
    
3. FPGA 失敗時允許 software sharpness fallback
    
4. Software fallback 範圍 `-20 ～ +20 mA`、61 steps
    
5. Anchor AF 結果必須能提供：
    
    - Keyence focus height
    - locked liquid current

否則 view 停止，不會靜默改用 nominal pose。

設定在 [strap_macro_cam1_scan.yaml (line 18)](D:/Provenance Laboratories projects/ImagingLibWatch/config/strap_macro_cam1_scan.yaml:18)。

## macro_cam_1 非 anchor tiles

非 anchor：

```
af_mode_id = 8
use_autofocus = false
use_liqud_af = false
```

實際不執行 Keyence，也不執行 liquid AF，而是：

- 從前後 anchors 插值／外插 Keyence focus height
- 從 anchors 插值 locked liquid current
- 移到 predicted pose
- 直接設定 predicted `cammag`
- 拍照

目前：

```
liquid_autofocus_every_n_tiles = 0
liquid_autofocus_retry_count = 0
```

所以正常情況下沒有 periodic liquid AF，也沒有 quality-triggered liquid retry。

Payload 的兩個 AF master switch 在 [strap_macro1_scan.py (line 1594)](D:/Provenance Laboratories projects/ImagingLibWatch/core/strap_macro1_scan.py:1594)。

---

# 三、SAM 後正式拍攝 links／endlinks／screws

## 來源相機和最終拍照相機不同

不論 stitched source 是：

- macro_cam_2 stitched image
- macro_cam_1 stitched image

SAM 後：

|Component|最終拍照相機|
|---|---|
|Regular links|macro_cam_1|
|Endlinks|macro_cam_1|
|Screw source-link 中間分析圖|macro_cam_1|
|最終 screw 特寫|micro_cam|
|4029/4030 endlink references|macro_cam_1|

相機 mapping 定義在 [strap_link_capture_plan.py (line 59)](D:/Provenance Laboratories projects/ImagingLibWatch/core/strap_link_capture_plan.py:59)。

## stitched source 是 macro_cam_2 時

可選：

- Front/underside links
- Side links
- Back/outer-surface links
- 9-clock links
- Endlinks
- Endlink references
- Side／9-clock screws

SAM 座標先由 macro_cam_2 stitched pixel 轉換至 macro_cam_1／micro_cam stage pose，之後套用同一套 component focus policy。

## stitched source 是 macro_cam_1 時

目前 UI 限制只允許：

- Endlinks
- Endlink references
- Side screws
- 9-clock screws

不允許把四組 regular-link surfaces 當成正式 template component 選擇，見 [MACRO1_AVAILABLE_COMPONENTS (line 90)](D:/Provenance Laboratories projects/ImagingLibWatch/core/strap_link_capture_plan.py:90)。

但選 screws 時，系統仍會先用 macro_cam_1 拍 regular link 的中間分析圖，做第二階段 fine SAM；這些 link images 是 `analysis_only`，不會成為 template 正式 point。

---

## Links／Endlinks macro_cam_1 AF policy

每個 group 會選最多 5 個 anchors，約位於 X 範圍：

```
10%、30%、50%、70%、90%
```

優先從 regular links 選 anchor。

### Macro anchors

```
camera = macro_cam_1
af_mode_id = 3
use_autofocus = true
use_liqud_af = true
```

見 [_configure_strap_component_focus_policy() (line 15052)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:15052)。

|View|Keyence 軸/start|Liquid AF|
|---|---|---|
|Front|Y start 65|有|
|Side|Z start 20|有|
|Back|Y start 65|有|
|9-clock|Z start 20|有|

注意 Side／9-clock component anchors 使用的是 **mode 3，不是 macro_cam_1 stitch anchor 的 mode 12**。

macro_cam_1 有 camera-specific：

```
keyence1_measure_start_z = 20.0
```

而 camera-specific 優先於 mode 3 的 global Z=65，見 [hardware_config.yaml (line 1090)](D:/Provenance Laboratories projects/ImagingLibWatch/config/hardware_config.yaml:1090)。

### Side／9-clock 中央 component anchor

中央 anchor 如果 Keyence miss，會依序重試整個 capture：

```
Y offset = 0, +1, -1, +2, -2, +3, -3 mm
```

見 [App/main.py (line 15150)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:15150)。

這仍是：

- Keyence Z AF
- Z start 20
- Y offset 只是 lateral target recovery

### Macro component anchor liquid AF

與 macro_cam_1 stitch anchors 不同，component anchor 使用：

- FPGA liquid AF
- 開啟 baseline validation
- 不做 61-step software sweep
- FPGA 不可靠時，比較 configured cammag／安全 boundary candidate
- 選 sharpness 較佳 current
- Liquid AF 失敗可使用 manual cammag 繼續拍照
- 不會因 macro liquid AF 單獨失敗而停止整個 component plan

設定在 [STRAP_COMPONENT_ANCHOR_LIQUID_AF_OVERRIDES (line 263)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:263)。

### 非 anchor links/endlinks

```
af_mode_id = 8
use_autofocus = false
use_liqud_af = false
```

不執行 Keyence，也不執行 liquid AF，而是：

- Front／Back：插值 Stage Y focus coordinate
- Side／9-clock：插值 Stage Z focus coordinate
- 插值 liquid-lens current
- 設定 `cammag`
- 直接拍照

實作在 [App/main.py (line 16138)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:16138)。

## Endlink 是 anchor 還是 predicted point？

這取決於當次選了哪些 component：

- 如果同一 group 有 regular links：5 個 anchors 從 regular links 選，endlinks 通常是非 anchor，使用 mode 8 + predicted coordinate/current。
- 如果只選 Endlinks、沒有 regular link rows：每個 group 通常只有左右兩個 endlinks，因為總數 ≤5，兩個 endlinks 都會被當成 anchor，使用 mode 3 + Keyence + liquid AF。
- 如果選 Endlinks + Screws：screw 流程會加入 analysis-only regular links，因此 anchors 會從那些 regular links 選，endlinks通常回到 predicted non-anchor。

---

# 四、SAM 推測 screws 後的正式 micro images

Screws 只支援：

- 3-clock side screws
- 9-clock side screws

流程不是直接拿 stitched SAM 座標就拍 micro：

1. stitched image 找出 link
2. macro_cam_1 拍該 link
3. 在 macro image 做 finer screw detection
4. 將座標轉成 micro_cam pose
5. micro_cam 正式拍 screw

Screw capture plan 在 [strap_link_capture_plan.py (line 353)](D:/Provenance Laboratories projects/ImagingLibWatch/core/strap_link_capture_plan.py:353)。

## 每一張 screw image

```
camera = micro_cam
af_mode_id = 3
use_autofocus = true
use_liqud_af = true
```

也就是每顆 screw 都是完整 AF，不走 anchor interpolation。

|項目|行為|
|---|---|
|Keyence|有|
|Keyence focus 軸|Z|
|Keyence start Z|65 mm|
|Y search|無，正常只有 offset 0|
|Liquid lens AF|有，必須成功|
|FPGA AF fallback|有|
|Software fallback|`-20～+20 mA`，61 steps|
|Mechanical AF failure|該 screw capture 失敗|
|Liquid AF failure|該 screw capture 失敗|

因為 micro_cam 沒有 camera-specific `keyence1_measure_start_z`，所以 mode 3 的 global Z=65 生效。

Micro screw 的 fallback 在 [App/main.py (line 15058)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:15058)。

---

# 五、兩張 Endlink reference images

就是：

- 4029：12-clock Bracelet Endlink reference
- 4030：6-clock Bracelet Endlink reference

設定：

```
camera = macro_cam_1
af_mode_id = 8
use_autofocus = false
use_liqud_af = false
cammag = 0
has_glass = false
```

來源：

- [4029 (line 6208)](D:/Provenance Laboratories projects/ImagingLibWatch/config/internalnum_config.yaml:6208)
- [4030 (line 6254)](D:/Provenance Laboratories projects/ImagingLibWatch/config/internalnum_config.yaml:6254)

因此兩張 reference：

|項目|4029 / 4030|
|---|---|
|AF mode ID|8|
|Keyence AF|不使用|
|Keyence start Y/Z|不適用|
|Liquid AF|不使用|
|Liquid current|固定 `cammag=0`|
|Pose|完全使用 internalnum 固定 XYZ/RX/RZ|
|執行時是否重套 internalnum|是，strict fixed reference|

雖然 mode 8 本身定義為 liquid-only wide/fine mode，但 capture master switch `use_liqud_af=false`，所以實際不會跑 liquid AF。

---

# 最終總表

|拍照類型|Camera|Mode|Keyence|Start／軸|Liquid AF|
|---|---|---|---|---|---|
|macro_cam_2 Front stitch anchor|macro_cam_2|3|有|Y=65|有|
|macro_cam_2 Front 其他 tile|macro_cam_2|3|無|—|每張有|
|macro_cam_2 Side stitch anchor|macro_cam_2|3|有|Z=65|有|
|macro_cam_2 Side 其他 tile|macro_cam_2|3|無|—|每張有|
|macro_cam_2 Back stitch anchor|macro_cam_2|3|有|Y=65|有|
|macro_cam_2 Back 其他 tile|macro_cam_2|3|無|—|每張有|
|macro_cam_2 9-clock anchor|macro_cam_2|3|有|Z=65|有|
|macro_cam_2 9-clock 其他 tile|macro_cam_2|3|無|—|每張有|
|macro_cam_1 Front/Back stitch anchors|macro_cam_1|3|有|Y=65|有，FPGA→software fallback|
|macro_cam_1 Side/9 stitch anchors|macro_cam_1|12|有|Z=68.12；中央 anchor 有 probe-Y search|有，FPGA→software fallback|
|macro_cam_1 stitch 非 anchors|macro_cam_1|8|無|anchors 插值|無，直接套插值 current|
|SAM macro links anchors|macro_cam_1|3|有|Front/Back Y=65；Side/9 Z=20|有，best-effort FPGA|
|SAM macro links 非 anchors|macro_cam_1|8|無|插值 Y 或 Z|無，直接套插值 current|
|SAM endlinks|macro_cam_1|3 或 8|依是否被選成 anchor|同所在 view|anchor 有；predicted 無|
|SAM screws|micro_cam|3|每張有|Z=65|每張有，含 software fallback|
|4029/4030 references|macro_cam_1|8|無|不適用|無，固定 current 0|

補充：macro_cam_1 scan 開始前的 Strap endpoint detection 也會使用 Keyence OUT1，但那是「找 Strap 左右邊界」的 probe，不是任何 stitch image 的 Autofocus，也不會執行 liquid-lens AF。