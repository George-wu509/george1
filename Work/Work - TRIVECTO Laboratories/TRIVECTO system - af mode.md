

|                                                 |     |
| ----------------------------------------------- | --- |
| [[#### af_mode_config.yaml裡面定義的所有af_mode並列表整理]] |     |
| [[#### 列表整理每個af_mode_id(0~12)類別]]               |     |
| [[#### 在執行App的create template拍照的autofocus]]     |     |
| [[#### 完整整理在Strap時Af_mode]]                     |     |
| [[#### autofocus為什麼比預期的更慢]]                     |     |
| [[#### 甚麼原因會造成硬體 FPGA AF失敗]]                    |     |
| [[#### 特別針對Strap使用到的autofocus方法]]               |     |
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

| ID  | 名稱                                               | 數量  | `internalnum1`                                                                                                                                                          |
| --- | ------------------------------------------------ | --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0   | default                                          | 76  | `0001`, `0004–0005`, `0007–0021`, `0026–0031`, `0051–0054`; `1001`, `1003`, `1005–1011`; `2001–2012`; `3001–3008`, `3010–3017`; `6001–6011`                             |
| 1   | crown                                            | 1   | `0032`                                                                                                                                                                  |
| 2   | advanced                                         | 2   | `1004`, `3009`                                                                                                                                                          |
| 3   | strap                                            | 268 | `4001–4028`; `4101–4120`; `4201–4220`; `4301–4320`; `4401–4420`; `4501–4520`; `4601–4620`; `4701–4720`; `4801–4820`; `5001–5020`; `5101–5120`; `5201–5220`; `5301–5320` |
| 4   | strap endlink number                             | 0   | 無                                                                                                                                                                       |
| 5   | lume guarded af                                  | 4   | `0022–0025`                                                                                                                                                             |
| 6   | keyence autofocus only                           | 0   | 無                                                                                                                                                                       |
| 7   | side zaber 95 to 80 two-stage af                 | 0   | 無                                                                                                                                                                       |
| 8   | side AF two-stage wide then fine                 | 16  | `0039–0050`, `4029–4032`                                                                                                                                                |
| 9   | keyence + two-stage wide then fine               | 1   | `1002`                                                                                                                                                                  |
| 10  | side Z 85 to 95 AOI 64×64                        | 3   | `0002`, `0003`, `0006`                                                                                                                                                  |
| 11  | reuse side reference Y                           | 6   | `0033–0038`                                                                                                                                                             |
| 12  | strap side Keyence Z=68.12 + two-stage liquid AF | 0   | 無                                                                                                                                                                       |

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









#### 完整整理在Strap時Af_mode
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

| View    | Anchor mode | Anchor Keyence             | Anchor liquid AF | 非 anchor tile    |
| ------- | ----------- | -------------------------- | ---------------- | ---------------- |
| Front   | 3           | 有，Y start 65               | 有                | mode 8，但兩種 AF 都關 |
| Side    | 12          | 有，calibrated Z start 68.12 | 有                | mode 8，但兩種 AF 都關 |
| Back    | 3           | 有，Y start 65               | 有                | mode 8，但兩種 AF 都關 |
| 9-clock | 12          | 有，calibrated Z start 68.12 | 有                | mode 8，但兩種 AF 都關 |

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






```
也請幫我分析所有的af_mode的keyence autofocus跟liquid lens autofocus為什麼比預期的更慢? 我使用官方的ITALA View 使用liquid lens都非常快都少於1sec, 但這裡都很慢, 請深入分析並提供優化建議
```
## Autofocus 效能分析:為什麼比 ITALA View 慢

我已經深入追蹤了 Keyence AF 與 Liquid Lens AF(硬體 FPGA 版)的完整呼叫鏈,並實際讀取原始碼驗證了關鍵行號。結論:**兩條路徑都不是單一瓶頸造成的,而是"每一步都疊加了保守的 sleep / 診斷 / 安全檢查",而 ITALA View 官方工具幾乎只是直接下 FPGA 指令、輪詢一次狀態就結束。** 以下分兩部分拆解。

---

### 1. Liquid Lens AF(硬體 FPGA 版)—— 這是與 ITALA View 直接對標的路徑

**呼叫鏈:** unified_driver.py:7957 `trigger_liquid_lens_hardware_autofocus` → hardware_managers.py:2914 `CameraManager.trigger_hardware_liquid_af`

#### 1.1 單次呼叫的硬編碼 sleep 鏈(我已逐行核對)

| 位置                                  | Sleep                          | 說明                                                   |
| ----------------------------------- | ------------------------------ | ---------------------------------------------------- |
| hardware_managers.py:2938,3087      | 80ms                           | 重置 lens 電流為 0                                        |
| hardware_managers.py:3104           | 50ms                           | 關閉 `oeLiquidLensAutofocusEnable` 後                   |
| hardware_managers.py:3115           | 50ms                           | 設定 `CurrentMode` 後                                   |
| hardware_managers.py:3684           | **150ms**                      | 關閉 `TriggerMode` 讓 FPGA free-run 後                   |
| hardware_managers.py:3687-3692      | 最多100ms                        | 清空舊 frame buffer                                     |
| hardware_managers.py:3707           | 50ms                           | 啟用 `oeLiquidLensAutofocusEnable` 後                   |
| hardware_managers.py:3905           | **150ms**                      | 下達 `AutofocusStart` 指令後(確認過,這行就是 `time.sleep(0.15)`) |
| 輪詢迴圈 hardware_managers.py:3924-3936 | 每輪 20ms drain + 讀 status + 讀電流 | 見下 1.2                                               |
| hardware_managers.py:3951,3982-3983 | 50+20+30ms                     | AF 成功後 latch 電流                                      |
| Fallback 逾時路徑                       | 最多 **2.0s**                    | `status_fallback_wait_s`,只有輪詢逾時才會走到                  |

**光是不含輪詢迭代的固定 sleep 就有 ~0.6–0.8 秒**,這還沒算 FPGA 內部真正掃描電流所花的時間。使用者提到 ITALA View 全程 <1 秒,這代表軟體層的"儀式性等待"幾乎吃掉了整個時間預算。

#### 1.2 每次輪詢都做了不必要的重量級動作

輪詢迴圈(hardware_managers.py:3924-3936)在等 FPGA 從 Busy 變成 Ready 的過程中,**每一輪**除了讀 status,還額外呼叫 `_read_liquid_lens_current()`(另一次同步 GenICam node 讀取),外加 20ms 的 buffer drain。這個電流讀值在迴圈裡除了寫進 trace 之外沒有被用來做決策,純粹是診斷用途卻跑在熱路徑裡。

#### 1.3 診斷 dump 是隱藏的大成本(我新發現、之前探索沒抓到)

hardware_managers.py:3007-3021 的 `_dump_af_diagnostics()` 在 PRE-START、POST-START、AF-SUCCESS/FAILED 等**每個階段都會呼叫**(至少 2-4 次/次 AF),每次呼叫 hardware_managers.py:2994-3005 的 `_read_af_diagnostics()`,而該函式會**逐一同步讀取 21 個 GenICam node**(`oeLiquidLensAutofocusEnable`、`Status`、`Current`、`Width/Height/WidthMax/HeightMax/SensorWidth/SensorHeight`…)。

更糟的是,每次 dump 同時做 `self.logger.info(log_msg)` **和** `print(log_msg)`(hardware_managers.py:3020-3021),輸出一段 20 多行的格式化字串。在 console 沒有重導向、或 logger 掛了慢速 handler(檔案/網路)的情況下,`print()` 的 stdout flush 本身就可能造成數十毫秒的阻塞,而這完全跟"對焦"無關,是純粹的除錯輸出、且看起來沒有 debug flag 保護、是無條件執行的。

**這是全部發現中風險最低、CP 值最高的優化點**——4 次 × 21 個 node 讀取 + 4 次 print/log,在生產環境的每次 AF 都在跑,卻只在真的要除錯時才有用。

#### 1.4 真正致命的是"乘法效應":同一次邏輯 AF 被整套重跑多次

unified_driver.py:8705 `_run_hardware_af_attempt` 包住上面整套 ~0.7-0.9 秒的流程,但以下開關會讓它**重複執行整個流程**:

- **`fpga_af_best_of_n_enable`**(unified_driver.py:8186-8192,迴圈於 8741-8791):整套流程重跑 `best_n_count` 次取最銳利的結果
- **`fpga_af_repeat_check`**(unified_driver.py:8182-8183, 8853-8874):再跑一次驗證電流值是否穩定
- **`hardware_two_stage_enable`**(unified_driver.py:8116-8166, 9399-9490):先跑一次粗掃(預設 30 frames),再跑一次**完全獨立**的細掃(預設 30 frames)——等於整套流程跑 2 次
- **`liquid_af_extensionN`**:結果落在邊界時額外重掃

我核對了 config/af_mode_config.yaml 的第 25-33、119-273、403-539 行,確認有多個 `af_mode_id` 同時開啟 `hardware_two_stage_enable: true` **加上** `liquid_af_extensionN: 2`。這種組合下,單一邏輯 AF 呼叫可能把上述整套 ~0.7-0.9 秒的流程跑到 **4-6 次**,累積成 3-5 秒以上——這很可能就是你觀察到"比 ITALA View 慢很多"的主因,值得先確認慢速觀測點用的是哪個 `af_mode_id`。

---

### 2. Keyence AF(粗定位,`autofocus_option: 1` MATLAB-like)

**核心函式:** unified_driver.py:5259 `_read_keyence_absolute_output_with_expansion`

- `keyence_sample_settle_s` 設計上可調,但被 unified_driver.py:5609,5702 `max(0.05, settle_s)` **強制下限鎖在 50ms**,調再低也沒用
- `read_stable_value(samples=5, delay=0.01)`(hardware_managers.py:4474-4498)每次讀 OUT1 都是 **5 次序列式 TCP 往返** + sleep,單次讀值成本就 ≥50ms + socket RTT
- `read_out3_for_debug` 預設是 **`True`**(unified_driver.py:2809),等於每次都多打一次 Keyence 往返,但這個值只在 `has_glass` 時才會用到公式裡——多數情況純屬浪費
- 每一次探針移動(甚至 dx=0,dy=0 的中心點)都走 `_move_xyz_with_safety` → `_plan_motion_segments` 的**完整安全規劃路徑**(unified_driver.py:985-1030):重新讀 5 軸 Zaber 位置、規劃"狗腿"退避-重配置-趨近路徑、re-validate against digital twin——對一個公厘級的探針微動來說是過度設計
- 我還確認了 `_af_keyence_matlab_like`(unified_driver.py:3106-3111)先移到探針 XY,接著呼叫的 `sample_offsets()`(unified_driver.py:5400-5407)對中心點(offset 0,0)**又重複移動了一次到同一座標**——這是一個明確的重複動作 bug

若 ROI/Z expansion(`keyence_read_roi_expan`/`keyence_read_z_expan`)被任何 mode override 打開(shipped config 預設是 0,關閉的),整套"移動+settle+5樣本讀值"會被乘上 `(2·roi+1)² × (2·z+1)` 倍。

---

### 3. 每次拍照都會疊加的全域開銷(跟 AF 演算法無關,但直接墊高你感受到的總延遲)

- unified_driver.py:12298-12310:AF 完成後、每次擷取影像前,**無條件** `time.sleep(min(capture_settle_time_s, 10.0))`,而 config/hardware_config.yaml:88 設的是 **1.0 秒**(程式碼內建 fallback 是 0.5 秒,但 config 覆蓋成 1.0)。這是按"每次拍照"算,不是按"每個對焦點"算——如果一個點要拍多張(HDR/多曝光),這 1 秒會被重複收取
- 每次 `capture_image()` 都會重新 arm software trigger 並 drain buffer 最多 150ms(hardware_managers.py:1339-1358)。這點有新加入的測試檔 test_camera_capture_state_machine.py:88-125 明確驗證是刻意設計(因為 AF 讓相機處於 free-run 模式,之後每次拍照都要重新 arm),但代價是它套用到**每一張**照片,不只是 AF 後的第一張

---

### 優化建議(依「效益/風險比」排序)

1. **把 `_dump_af_diagnostics` 的 21-node 讀取 + print/log 輸出改成用 debug flag 包起來,預設關閉**(hardware_managers.py:3007-3021)。零風險,直接砍掉每次 AF 呼叫 2-4 次 × 21 個同步 GenICam 讀取。
2. **確認生產環境用的 `af_mode_id` 是否開了 `hardware_two_stage_enable` + `best_of_n` + `extensionN` 的組合**(config/af_mode_config.yaml 第 25-33、119-273、403-539 行)。這是唯一能解釋"比 ITALA View 慢好幾倍"而非"慢 0.5-1 秒"的原因,先確認是否真的需要這些精度保證,否則關掉能立即減少 2-6 倍的呼叫次數。
3. **重新檢視 FPGA AF 前後那串 50/80/150ms 的固定 sleep**(hardware_managers.py:2938-3707)。這些是軟體等待,不是光學/馬達物理限制,值得跟硬體廠商(Opto Engineering/ITALA)的規格書核對實際所需的最短穩定時間,很可能可以砍到目前的 1/2-1/3。
4. **輪詢迴圈(hardware_managers.py:3924-3936)拿掉每輪都讀電流的呼叫**,只在狀態變成 success/failure 時才讀一次,减少每輪的 node round-trip。
5. **`capture_settle_time_s` 從 1.0 秒下修**(config/hardware_config.yaml:88),或改成只在真正有 Z 軸移動/AF 動作後才 settle,而非每次拍照都套用。
6. **修掉 Keyence AF 中心點的重複移動**(unified_driver.py:5400-5407)與 `read_out3_for_debug` 預設值改為 `False`(unified_driver.py:2809)。
7. **針對 AF 用的公厘級微動探針,提供一條輕量直接移動路徑,繞過 `_move_xyz_with_safety` 的完整安全規劃器**(unified_driver.py:985-1030),把安全規劃保留給真正的大範圍點位跳轉。

我建議先做 #1(零風險)跟 #2(找出你實際踩到的 mode 設定),這兩項不需要改動核心邏輯就能驗證影響有多大,再決定要不要動 sleep 時間常數。需要我直接動手改哪一項嗎?



#### autofocus為什麼比預期的更慢
```
也請幫我分析所有的af_mode的keyence autofocus跟liquid lens autofocus為什麼比預期的更慢? 我使用官方的ITALA View 使用liquid lens都非常快都少於1sec, 但這裡都很慢, 請深入分析並提供優化建議, 那如果我用c++重寫keyence autofocus跟liquid lens autofocus functions之後在python裡呼叫是否也能加速, 以及那這樣也可以用python helper\deployment\build_moonlight.py這樣指令compiler成dll, exe並正常運作嗎
```
## 分析結果:Keyence AF 與 Liquid Lens AF 為什麼比 ITALA View 慢

我完整讀過 Controller/hardware_managers.py、Controller/hardware_drivers/unified_driver.py 的核心 AF 邏輯，以及 helper/deployment/build_moonlight.py。結論先講:**慢的根源幾乎都是「軟體/機械掃描迴圈 + I/O 等待」,不是 Python 運算太慢** — 這也直接決定了 C++ 重寫能不能幫上忙。

---

### 一、Liquid Lens AF 慢在哪(這是主要瓶頸)

系統其實**已經有**跟 ITALA View 一樣的硬體原生路徑,但實務上經常走到慢路徑:

| 路徑                                                                                                     | 機制                                                                   | 每次操作成本                                                | 速度             |
| ------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------- | ----------------------------------------------------- | -------------- |
| **硬體 FPGA AF**  <br>Controller/hardware_managers.py:2914                                               | 寫入 GenICam node 下單一 AF 指令,然後 50ms 週期 poll 狀態                         | 由 FPGA 內部掃描時間決定,基本上等同 ITALA View                      | **快**(<1~2s 級) |
| **軟體逐點掃描 AF**  <br>Controller/hardware_managers.py:2688                                                | Python for 迴圈跑 **61 個(最多到 121 個,af_mode 14)** 鏡頭電流值,每個位置真的觸發拍一張圖算清晰度 | 61 × (settle 25ms + `capture_image` 完整觸發/曝光/fetch 流程) | **慢**(數秒到數十秒)  |
| **Side-Zaber 合併掃描**  <br>(af_mode 7/8/10,側面視角)  <br>Controller/hardware_drivers/unified_driver.py:7466 | Y 軸機械移動(預設 4 點,最多 11 點)× 鏡頭電流(預設 11 點)× 每點拍 3 張                      | 預設 **4×11×3 = 132 次拍照**,每次前面還有 Zaber 物理移動             | **最慢**         |

關鍵程式碼(Controller/hardware_managers.py:2770):

values = list(np.linspace(start_val, end_val, steps))   # steps 預設 61
for idx, v in enumerate(values):
    ok, applied = self.set_liquid_lens_raw(v, settle_time_s=settle_s, verify=False)
    for _ in range(max(0, warmup_frames)):
        _ = self.capture_image(...)         # 額外的暖機拍照,直接丟棄
    img = self.capture_image(...)           # 真正的一張
    score = self.get_image_sharpness(img, roi_cfg)  # 這段是 cv2 Laplacian,已向量化,不是瓶頸

而每一次 `capture_image`(Controller/hardware_managers.py:1279)本身就包含:寫曝光/增益 → arm software trigger → drain 殘留 buffer(最多到 0.15s,通常較短)→ 等 trigger armed → `TriggerSoftware.execute()` → fetch(timeout 依曝光時間動態算,最多到 5s)→ 失敗會 `for attempt in range(3)` 重試。**這一整套流程被重複 61~132 次**,是實際慢下來的主因,不是 sharpness 計算本身。

**為什麼會走到慢路徑**:

1. 硬體 FPGA AF 失敗/驗證不通過時,系統設計上會 fallback 到軟體掃描(`software_fallback_enable: true`,多個 af_mode 都開啟這個),這是**預期會發生**的行為,不是 bug。
2. 側面/strap 視角(af_mode 7/8/10/11/16)因為 `mechanical_af_disabled`,**預設就直接走**最貴的 side-Zaber 合併掃描,而不是先試硬體 AF。
3. `af_mode_config.yaml:450-452` 甚至有一條註解明講這是**刻意的 speed-for-accuracy 取捨**(拒絕弱的 FPGA lock,強制用 121 步影像掃描驗證)。

---

### 二、Keyence AF 慢在哪

Keyence AF 的核心不是拍照算清晰度,而是讀雷射位移感測器的數值算 Z(`_af_keyence_matlab_like`,Controller/hardware_drivers/unified_driver.py:2718)。**單點測量本身很快**(幾次 channel 讀值 + 一次 stage 移動)。

但我進一步往下讀發現一個容易被忽略的地方——Controller/hardware_drivers/unified_driver.py:5272 的取樣函式,預設參數是:

roi_expansion = af_cfg.get("keyence_read_roi_expan", ...)  # 預設 1
z_expansion   = af_cfg.get("keyence_read_z_expan", ...)    # 預設 2  → z 方向會取 5 個點 (0, ±1, ±2 圈)

一旦觸發 ROI 信心度擴展或 Z 擴展(信心度檢查沒過、角度感知模式等情況),它會對**每一個取樣點都做一次真實的 Zaber 探針移動**(Controller/hardware_drivers/unified_driver.py:5388/`_move_keyence_probe_xy`)+ `time.sleep(settle_s)`(預設 50ms)+ 對每個 channel 讀 5 次樣本、每次間隔 10ms。而且文件裡明確提到 Z 軸移動 profile(`z_focus`)是**系統中刻意設得最慢的**(helper/docs/24_performance_capacity_reliability/24.2_capture_and_motion_timing.md),上限 10mm/s,為了避免震動影響量測精度。

所以 Keyence AF 慢,通常不是「演算法迴圈太多次」,而是**當信心度檢查沒過,退到多點網格重試時,每一點都要等一次「刻意放慢的機械移動」**——這是物理限制,不是程式效率問題。

---

### 三、為什麼 ITALA View 用 liquid lens 都 <1 秒

合理推測(ITALA View 我們看不到原始碼,但從行為反推):它幾乎肯定只走**單一次 FPGA 硬體 AF 指令 + 輕量 poll**,沒有軟體逐點掃描 fallback、沒有多點信心度驗證重試、沒有側面 Y 軸機械掃描。這套系統為了要支援更嚴苛的良率驗證(拒絕弱鎖定、驗證信心度、side-Zaber 補救),主動加了很多「慢但準」的保護機制。**這是設計取捨造成的差異,不是實作效率差**。

---

### 四、優化建議(不需要 C++,純邏輯層面就能大幅提速)

優先序由高到低:

1. **把 61 步全域線性掃描改成粗到細二分搜尋**:先跑 7~9 步粗掃定出大致峰值區間,再在附近跑 5~7 步精掃。61 步 → 約 12~16 步,理論上可以砍掉 70-80% 的拍照次數,精度影響很小(清晰度曲線通常是單峰的)。
2. **重新檢視軟體 fallback 觸發條件**,而不是條件放寬就無腦掃 61~121 步。先確認硬體 FPGA AF 為什麼常失敗(profile mismatch?timeout 太短?),把根因修掉可能比優化 fallback 更有效。
3. **side-Zaber 掃描(af_mode 7/8/10)砍掉大量無效組合**:4 Y × 11 電流 × 3 frame = 132 張是暴力全網格。可以先用較少 Y 點粗定位,再用二分搜尋鎖電流,frame 數也可視情況降到 1~2(median 只是為了抗雜訊)。
4. **把 lens 移動的 settle 時間跟上一步的 sharpness 運算重疊**(pipeline 化):目前是完全序列的 `set_liquid_lens_raw → sleep(settle) → capture → sharpness`,可以用一條背景 thread 讓「移動下一個電流值+settle」跟「算上一張圖的清晰度」同時進行——這用 Python threading 就能做,不需要 C++。
5. **降低每步 log 開銷**:Controller/hardware_managers.py:2787 每步都 `logger.info` 格式化字串,加上 FPGA 路徑的 `_dump_af_diagnostics` 同時 `logger.info` + `print()` 一大段診斷(Controller/hardware_managers.py:3007)。單次影響不大,但在 61~132 次迴圈裡疊加起來是實測前該先關掉的干擾項。
6. **對重複測同一位置的情況啟用 focus reuse**(文件已經有設計:helper/docs/07_capture_workflow/7.8_autofocus_routing_and_focus_reuse.md),避免對同一個站位重複跑完整 AF。
7. 儲存庫**完全沒有實測過 AF 各路徑的真實耗時**(文件 helper/docs/24_performance_capacity_reliability/24.2_capture_and_motion_timing.md 明講"未找到已記錄的比較")。建議先加簡單的 timing log(每個 AF 呼叫記錄各階段耗時),實測出真正的熱點,再決定要優化哪一段——目前這份分析是基於程式碼結構推論,不是量測數據。

---

### 五、C++ 重寫 Keyence/Liquid Lens AF function 會加速嗎?

**大機率不會有顯著效果,而且很可能不值得投入。** 原因:

- 這些函式的耗時主體是 **I/O 等待**:相機曝光時間、GenTL frame fetch、`time.sleep()` 的機械 settle 時間、Zaber 物理移動時間、FPGA 內部掃描時間。這些都是**硬體本身的物理延遲**,C++ 跟 Python 呼叫底層 SDK 時等待的時間完全一樣——CPU 在這段時間根本沒有在算東西,只是在等硬體回應。
- 唯一真正做運算的部分(`get_image_sharpness` 用 `cv2.Laplacian(...).var()`)本來就是**已經向量化的 C/C++(OpenCV 底層)**,Python 這層只是呼叫介面,重寫成 C++ 頂多省掉幾微秒的呼叫開銷,對總耗時(以百毫秒~秒計)完全無感。
- 相機(harvesters/GenICam)跟 Zaber(zaber_motion)本身**已經是 Python 包著原生 C/C++ SDK**(zaber_motion 底層就是一顆 ctypes DLL:`zaber-motion-core-windows-*.dll`),不是純 Python 重新實作協議——所以「重寫成 C++」實際上只是把 Python 的**控制流程/迴圈邏輯**改寫,而控制流程本身不是瓶頸,迴圈裡的 `time.sleep()` 和 SDK 阻塞呼叫才是。

真正該優化的是**演算法層(掃描點數、粗細二分、pipeline 重疊)**,這在 Python 就能做,C++ 重寫在這裡屬於「優化了不是瓶頸的部分」。

如果你之後想用多執行緒把「移動 lens」跟「算清晰度」重疊(建議 4),C++ 或許能省掉 Python GIL 造成的執行緒排程開銷,但這個量級跟硬體延遲比起來也是次要的。

---

### 六、`build_moonlight.py` 能不能把 C++ 編譯進 DLL/EXE?

**目前不行,而且它現在做的事完全是另一回事**——讀過整份 helper/deployment/build_moonlight.py 後可以確認:

- 它用的是 **PyInstaller**(helper/deployment/build_moonlight.py:33、`PyInstaller.__main__.run(args)`)。PyInstaller **只是把 Python 直譯器 + bytecode + 相依套件打包成一個 exe**,執行時仍然是逐行解讀 Python bytecode,**完全不會編譯成原生機器碼**,跑起來速度跟直接 `python App_run.py` 一模一樣。它不是 Cython、不是 Nuitka,沒有任何 AOT 編譯步驟。
- 整個 repo 裡**沒有任何 C++ extension 基礎建設**:沒有 `.pyx`、`CMakeLists.txt`、`pybind11`、`setup.py Extension()`、`.pyd` 檔案,也沒有 `src/`/`cpp/`/`native/` 目錄。硬體存取全部是純 Python binding(`harvesters`、`zaber_motion`)。

如果你真的要把某段邏輯用 C++ 重寫並在 Python 呼叫(即使如上所述效益有限),流程會是:

1. 用 **pybind11 + CMake**(或 `setup.py` 的 `Extension()` / scikit-build)另外寫一個獨立的 build 步驟,產出一個 `.pyd`(Windows 上 Python C extension 的副檔名)。這跟 `build_moonlight.py` 是兩個完全不同的階段——`build_moonlight.py` 只負責"打包成 exe",不負責"編譯 C++"。
2. 先確認這個 `.pyd` 能被單獨 `import` 且正常運作。
3. 修改 `build_moonlity.py`,仿照它現在處理 Zaber native DLL 的方式(`zaber_native_binaries()`,helper/deployment/build_moonlight.py:130、`--add-binary`,helper/deployment/build_moonlight.py:227),把新的 `.pyd` 加進 `--add-binary`/`--collect-all` 清單,PyInstaller 才會把它一起打包進最終的 `MoonlightApp.exe`。

這條路技術上可行、也不複雜,但鑑於第五點的分析——**瓶頸是硬體等待時間,不是 CPU 運算**——投入 C++/pybind11 的建置成本換來的實際加速可能非常有限。我會建議先做第四節的邏輯層優化(掃描點數精簡、fallback 觸發條件檢討、pipeline 重疊),用時間量測驗證效果後,再評估是否真的需要走 C++。

所以先針對liquid lens af, 甚麼時候會用硬體 FPGA AF? 甚麼時候會用軟體逐點掃描 AF?甚麼時候會用Side-Zaber 合併掃描? 是看不同的af_mode嗎? 是看不同的位置嗎? 那如果把軟體逐點掃描 AF都用硬體 FPGA AF替代是否可行?

Traced liquid lens AF routing logicSonnet 5

### 一、決定用哪個路徑的機制:是「per-camera 設定」,被 af_mode 覆寫,而不是位置直接判斷

路徑選擇不是程式碼裡寫死的位置判斷式(例如 `if view=="side"`),而是每個 af_mode 的 YAML 針對每一台相機分別覆寫一組設定值,實際判斷邏輯只在 Controller/hardware_drivers/unified_driver.py:11837 這一段:

af_method = str(af_liquid_af_cfg.get("af_method", "hardware")).strip().lower()
use_hardware_first = af_method in ["hardware", "fpga", "hardware_first", "fpga_first"]

if use_hardware_first:
    side_zaber_two_stage = af_liquid_af_cfg.get("side_zaber_two_stage_enable", False)
    if side_zaber_two_stage:
        hw_success, ... = self._run_side_zaber_liquid_lens_autofocus(...)   # side-Zaber
    else:
        hw_success, ... = self.trigger_liquid_lens_hardware_autofocus(...)  # 純硬體 FPGA
else:
    # 直接跑軟體逐點掃描(primary,不是 fallback)

# hw_success == False 時:
allow_software_fallback = af_liquid_af_cfg.get("software_fallback_enable", not use_hardware_first)
if liquid_failure_reason == "boundary_z_retry_exhausted":
    allow_software_fallback = False   # 這個失敗原因無論如何都不會 fallback
if (not hw_success) and allow_software_fallback:
    self.trigger_liquid_lens_software_autofocus(...)   # 軟體掃描當 fallback

`af_liquid_af_cfg` 讀的是 `hardware.autofocus.cameras.<相機>.liquid_af`(Controller/hardware_drivers/unified_driver.py:11019),所以**同一個 af_mode 底下,三台相機(macro_cam_1/macro_cam_2/micro_cam)可以各自被設成不同路徑**——這不是巧合,是設計上刻意允許的粒度。

**判斷順序**:

1. `af_method`(每相機設定)→ 決定走「硬體優先」還是「軟體優先(primary)」
2. 若硬體優先,`side_zaber_two_stage_enable` → 決定是「純硬體 FPGA」還是「side-Zaber 機械 Y 掃描 + 內部硬體 FPGA 精掃」
3. 若硬體嘗試失敗,`software_fallback_enable` → 決定要不要退到軟體逐點掃描(**除非失敗原因是 `boundary_z_retry_exhausted`,這種情況無論設定都不會 fallback**)

### 二、實測整理:17 個 af_mode 各自用哪條路徑

|af_mode|用途|Liquid Lens AF 實際路徑|
|---|---|---|
|0(預設)|一般拍攝|**純硬體 FPGA AF**,單次嘗試,`software_fallback_enable: false`|
|1|crown|純硬體 FPGA(僅調電流範圍)|
|2|advanced|硬體 FPGA,`hardware_two_stage_enable: true`(粗+精兩段內建重試,仍是硬體)|
|3, 4|strap 相關|純硬體 FPGA(只調 Keyence Y/Z)|
|5|lume guarded af|硬體 FPGA + baseline 驗證/邊界拒絕,無軟體/side-Zaber|
|6|keyence only|Liquid lens AF **整個關閉**(`enable: 0`),只跑 Keyence|
|**7**|side zaber two-stage|**side-Zaber 機械 Y 掃描 + 內部硬體 FPGA 精掃**(`side_zaber_two_stage_enable: true`)|
|8|side af two-stage|**純硬體 FPGA,兩段式**,不是 side-Zaber(容易跟 mode7 搞混,但沒有機械 Y 掃描)|
|9|keyence+two-stage|同 mode 8 + Keyence|
|10|side z aoi scan|`liquid_af.enable: 0`,**完全不碰 liquid lens 電流**,靠機械 Y 軸當對焦軸|
|11|reuse side reference|`liquid_af.enable: 0`,liquid lens **完全不跑**|
|12, 13|strap component|硬體 FPGA 兩段式,mode 13 明確 `software_fallback_enable: false`|
|**14**|strap component side adaptive|硬體 FPGA 兩段式 **+ 明確開啟 `software_fallback_enable: true`**(121 步密集掃描)|
|15|strap Endlink safe|同 13,失敗直接不拍(`continue_capture_on_liquid_af_failure: false`)|
|16|strap fast liquid only|純硬體 FPGA,單次嘗試,無任何重試/fallback|

**關鍵發現:全部 17 個 af_mode 裡,只有 mode 14 的 macro_cam_1 明確開啟軟體逐點掃描當 fallback。** 其餘所有模式,`software_fallback_enable` 在 base config(config/hardware_config.yaml)都是 `false`,軟體掃描根本不會被觸發。也沒有任何 af_mode 把 `af_method` 設成純軟體當 primary。

### 三、位置依賴是「透過 af_mode 間接表達」,不是硬體限制

你問的「side view 是不是就一定要用軟體/side-Zaber」——答案是**不一定,是設定選擇,不是硬體限制**:

- Mode 7(side,two-stage)**確實**用 side-Zaber。
- Mode 8(同樣是 side 相關名稱)**卻是純硬體 FPGA**,完全沒有機械 Y 掃描。
- Mode 16(strap,側面)也是**純硬體 FPGA 單次嘗試**。

我specifically 找過程式碼裡有沒有「因為這個角度/視角,FPGA AF 硬體上做不到,所以才需要 side-Zaber」這種限制——**沒有找到**。唯一在程式碼裡寫明的物理限制是關於 **Keyence**(雷射位移感測器),不是 liquid lens:

> `af_mode_config.yaml:104-106`:「R_X=90 的側面拍攝無法可靠使用 Keyence,因為相機參考點在較淺探針角度下難以重新對準,所以停用 mechanical/Keyence Z-AF,改用純 liquid lens AF」

Mode 10/11 停用 liquid lens AF 也是**設計選擇**(用機械 Y 軸本身當對焦軸,刻意不碰電流),註解(config/af_mode_config.yaml)寫的是「不要改動/自動對焦 liquid lens」,不是「做不到」。

**結論:side-Zaber 存在的真正理由是 mechanical_af_disabled({7,8,10,11,16})這幾個模式停用了 Keyence,而 side-Zaber 是為了在沒有 Keyence 輔助時,用機械掃描去補足對焦精度/魯棒性——不是因為 liquid lens 硬體 FPGA AF 在那些角度用不了。**

### 四、如果把所有軟體逐點掃描都換成硬體 FPGA AF,可行嗎?

**先講重點:你其實已經幾乎在那個狀態了。** 17 個模式裡有 16 個本來就是純硬體 FPGA(或硬體兩段式),軟體掃描目前只在 **mode 14 的 macro_cam_1** 上被啟用。所以「軟體逐點掃描造成的慢」在整體 17 個模式的使用比例上其實佔比不高——除非你的產線實際大量跑的是 mode 14,或是硬體 AF 在其他模式下經常失敗到需要靠 `use_manual_cammag_on_af_failure`/`best_effort_current_fallback_enable` 這類次要 fallback(這些不是軟體掃描,但也是失敗後的補救路徑,值得你確認一下實際觸發頻率)。

**如果你想連 mode 14 也關掉 `software_fallback_enable`,會有明確記載的風險**——mode 14 的註解(config/af_mode_config.yaml:450)直接寫明了觀察到的失敗案例:

> 「Point 93 顯示即使 sharpness 比 baseline 提升了 13.9%,整張影像仍然是全域模糊的。拒絕弱的 FPGA lock,讓影像掃描去證明確實存在一個明顯的峰值。」

這是原作者實測踩過的坑,而不是理論上的擔心。根本原因是**結構性的**:硬體 FPGA AF 對驅動程式來說是黑盒子——它只回傳「最終鎖定電流 + 一個粗略狀態字串」,`fpga_af_baseline_validation_enable` 頂多拿這個結果去跟**單一一張** baseline 電流的清晰度比較,沒辦法看到完整的「電流 vs 清晰度」曲線。而軟體逐點掃描因為每一步都真的拍照算分數,才能做到:

# hardware_managers.py:2824-2832
curve_median = np.median(smoothed_scores)
minimum_peak = curve_median * peak_median_min_gain
curve_validation_passed = best_score >= minimum_peak   # 判斷整條曲線是不是「平的」(沒有真正峰值)

這種「整條曲線是否平坦/沒有明顯峰值」的判斷,硬體 FPGA AF **拿不到資料去做**,除非韌體本身有暴露內部逐點掃描資料給驅動程式讀(我在程式碼裡沒找到任何 node/API 名稱暗示這個能力存在)。

**所以我的建議是**:

1. **可以放心做**:確認其餘 16 個模式(尤其如果你的產線常用到 mode 2/8/9/12/13 這種硬體兩段式)沒有意外開啟軟體 fallback,並檢查實際失敗率——如果硬體 AF 失敗率本來就低,現狀已經接近你要的「全硬體 FPGA」。
2. **mode 14 不建議直接關掉軟體 fallback**,除非你能先驗證:硬體 FPGA AF 在 mode 14 對應的位置/相機上,已經不會再重現「Point 93」那種弱鎖定但全域模糊的狀況。比較安全的折衷做法:
    - 先用 `fpga_af_best_of_n_enable`(重複跑 N 次 FPGA 掃描取一致結果)或 `liquid_af_extensionN`(邊界時延伸掃描範圍)這些**硬體側**現有的重試/驗證機制去加強,取代軟體掃描的角色。
    - 或是把軟體掃描的步數從 121 大幅降低(例如降到 15~21 步的粗掃,只用來做「曲線平坦度快篩」,通過就直接信任硬體結果,不通過才精掃)——這樣即使保留軟體 fallback,單次觸發的代價也會小很多。
3. 若要精準驗證「硬體 FPGA AF 失敗率到底多高」,目前程式碼裡**沒有任何統計/紀錄機制**在追蹤這個數字(只有 mode 14 那條質化註解),建議先加一個輕量 counter/log(記錄每次 AF 呼叫的 `af_method`、成功與否、`failure_reason`),量測幾天實際產線數據後再決定要不要進一步收斂 fallback。


#### 甚麼原因會造成硬體 FPGA AF失敗
```
甚麼原因會造成硬體 FPGA AF失敗? 因為目前很慢的軟體逐點掃描 AF只有當硬體 FPGA AF失敗才會使用, 所以預計要刪除軟體逐點掃描 AF, 因為即使使用runningtime需求絕對無法達成. 我發覺17個af_mode即使寫純硬體 FPGA AF, 仍然有單次嘗試, (僅調電流範圍), 粗+精兩段內建重試,仍是硬體, + baseline 驗證/邊界拒絕,無軟體等等不同設定, 這些不同設定是否有的也造成running time大幅增加? 請深入分析並提供建議如何優化速度
```
### 一、硬體 FPGA AF 失敗的根本原因

先講清楚 `trigger_hardware_liquid_af`(Controller/hardware_managers.py:2914)內部各失敗原因的判斷邏輯,以及背後真正的觸發條件:

|失敗原因|判斷邏輯|真正根因|
|---|---|---|
|**`timeout`**|15s(`hardware_timeout_s`)內狀態機沒回報明確的 success/failure 字串,也沒出現 Busy→Ready 轉態(Controller/hardware_managers.py:3916)|三種可能:(1) FPGA 真的需要接近/超過 15s 掃完 `hardware_frame_count` 張(見下方分析);(2) FPGA 內部狀態機卡在上一次結果沒重置(程式碼註解明講:「Make sure FPGA Internal status machines have been reset」Controller/hardware_managers.py:3099);(3) GenICam/USB3Vision 連線 flaky,status node 讀不到值。**這三種原因程式碼本身分不出來**,只能靠實際跑起來的 poll_trace 診斷資料判斷。|
|**`boundary`**|鎖定電流落在配置範圍邊緣 ±約 0.25~0.5mA 內(`_is_boundary_current`, 3576-3586)|有兩種可能且**程式碼分不出來**:(a) 真的物理上對焦點超出 `hardware_start/stop_current` 掃描範圍(例如工件 Z 位置超出液態鏡可對焦範圍);(b) 掃描範圍本身設太窄。這正是 `liquid_af_extensionN` 存在的理由——用「擴大範圍重跑」去試探到底是哪一種。|
|**`out_of_range`**|讀回電流完全在範圍外(`_is_out_of_range_current`, 3566-3574)|邏輯上比較怪——FPGA 應該只在配置範圍內掃描。最可能是**讀值時機競態**(讀到還沒完全 settle 的過衝值),或前一次操作殘留的電流值在 reset 完成前被讀到,而不是掃描範圍配置問題。|
|**`fpga_af_profile_mismatch`**|寫入值 vs 讀回值超過容許誤差(`_validate_af_profile`, 3811-3883)|程式碼註解直接寫明:`FrameCount` 這個暫存器在 AF Busy 狀態下會從「設定值」變成「即時遞減的剩餘張數計數器」,所以 start 後立刻讀回會「看起來不match」但其實是正常行為(所以 post_start 驗證故意跳過 frame_count,3907-3910)。真正的硬體 bug 只在 start/stop current 或 AOI size 這幾個「不該變」的欄位在 0.15s settle 後仍對不上時才算數。|
|**`aoi_not_centered`/`_readback_unavailable`**|AOI offset 讀回值跟預期偏差超過 4px 或完全讀不到(3400-3498)|AOI size 選項的 offset node `min/max/inc` 可能根本不允許真正置中,或是 GenICam node 在 AOI size 剛改變後還沒被填值。|

**重點結論**:`boundary`/`timeout` 是最模糊、最需要人工判斷的兩種失敗,而系統目前的應對方式(`liquid_af_extensionN` 擴大範圍重跑、15s+2s 的超時等待)本質上是在**用時間去換取對失敗原因的猜測**——這正是你要優化速度時該優先動手的地方。

---

### 二、各種「純硬體」設定層,實際上會不會拖慢速度

我逐一查證了 `unified_driver.py` 裡包住 `trigger_hardware_liquid_af` 的 6 層機制,結果分兩類:

#### 會增加 FPGA 呼叫次數的(真正拖慢速度)

|機制|額外開銷|目前 17 個 af_mode 有沒有用|
|---|---|---|
|`hardware_two_stage_enable`(粗+精)|**固定 2 次完整 FPGA AF 呼叫**(coarse 全範圍 + fine 在鎖定值 ±10mA 內再掃一次,`frame_count` 通常一樣多,不是快速版)|mode 2, 8, 9, 12, 13, 14 有用|
|`liquid_af_extensionN`|每次邊界失敗 → **再跑一次完整 FPGA AF**(範圍逐次加倍),且**不會關掉** best-of-N/baseline 驗證,是最大的組合爆炸源|mode 7、14 用(`extensionN: 2`)|
|`fpga_af_best_of_n_enable`|N 次完整 FPGA AF 呼叫|**17 個模式全部沒開**(config 預設關閉)|
|`fpga_af_repeat_check`|2 次完整 FPGA AF 呼叫|**17 個模式全部沒開**|
|`stage_retry_*`|每個候選點 = 1 次物理 Zaber 移動 + 1 次 FPGA AF 呼叫,`hybrid` 模式預設 4 candidates/step|**17 個模式全部沒開**(`stage_retry_enable: false`)|

**這是個很重要的發現:best-of-N、repeat-check、stage-retry 這三個最貴的乘數層,目前產線設定完全沒在用。** 所以你不需要擔心它們——它們目前是休眠的,不用特別去關。真正在吃時間的,是 `hardware_two_stage_enable` 和 `liquid_af_extensionN`。

#### 只加額外拍照、不加 FPGA 呼叫次數的(影響較小)

- `fpga_af_baseline_validation_enable`:多拍 1~3 張(mode 13/14 設到 3 張)參考電流的影像去算清晰度比較,不會重跑 FPGA 掃描本身。

#### 單次 FPGA AF 呼叫本身的固定開銷

- `hardware_timeout_s = 15.0s`(全部 3 台相機)+ `status_fallback_wait_s = 2.0s` → **失敗案例最多要等到 17s 才會判定失敗**,這是成功案例不受影響、但失敗/邊界案例會被拖累的固定成本。
- `hardware_frame_count`:macro_cam 40 張、micro_cam 20 張。我驗證出一個明確的數學關係:`(hardware_stop_current - hardware_start_current) / hardware_frame_current` **精確等於** `hardware_frame_count`(20mA / 0.5mA = 40,20mA / 1mA = 20)。這強烈暗示 **FPGA 內部做的就是跟軟體逐點掃描邏輯相同的「電流步進 + 每步拍一張評分」,只是用韌體實作而非 Python**。這代表:**掃描範圍越寬、電流步階越細,FPGA 內部要處理的張數就越多,單次呼叫本身就會越慢**——這是你目前手上唯一能直接控制、且可能大幅影響單次呼叫耗時的旋鈕(即使我們拿不到 FPGA 內部真實耗時的量測數據)。
- 另外我發現一個配置矛盾值得你確認:config/af_mode_config.yaml:117 註解寫「這台相機在側面模式下最多接受 30 張 FPGA AF frame」,但 base config(config/hardware_config.yaml:1195)預設是 40 張——這可能是韌體限制沒同步更新到 base config,建議跟硬體規格對一下。

---

### 三、17 個 af_mode 按硬體 AF 疊加開銷排名

|排名|Mode|疊加了什麼|備註|
|---|---|---|---|
|1|**14**|two-stage(2 次 FPGA)+ extension×2(疊加 baseline 驗證)+ baseline validation(3 frame)|目前唯一 two-stage + extension + baseline 三層疊加的模式,最壞情況約 4 次 raw FPGA 呼叫 + 額外拍照,失敗後還會掉到 121 步軟體掃描|
|2|**13**|two-stage + baseline validation(3 frame)|2 次 FPGA 呼叫 + 額外拍照,無 extension|
|3|**7**|side-Zaber 粗掃(**4 Y × 11 電流 × 3 frame = 132 張拍照 + 4 次物理 Y 軸移動**)+ extension×2|這不算在「two_stage/best_of_n」這組計數裡,但**很可能是全系統實際 wall-clock 最貴的模式**,而且**不屬於你要刪除的「軟體逐點掃描 AF」**(它呼叫的是 `_run_side_zaber_liquid_lens_autofocus`,不是 `run_liquid_lens_autofocus`)——**這是你必須注意的地方,見下方第五節**。|
|4|8, 9, 12|two-stage only|2 次 FPGA 呼叫,乾淨|
|5|2|two-stage only|同上|
|6|5|baseline validation only,但把 `hardware_frame_count` 拉到 80(單次掃描本身變慢)|無疊加重跑,但單次呼叫更慢|
|—|0,1,3,4,6,10,11,15,16|無疊加|mode 16 是最乾淨的:單次嘗試、無任何重試|

---

### 四、優化建議(針對你要刪除軟體逐點掃描 AF 的計畫)

#### 1. 軟體逐點掃描 AF 目前只在 mode 14 的 macro_cam_1 被啟用,砍掉風險可控但不是零

刪掉它之前,你需要處理 mode 14 存在的**具體、已記錄的真實案例**(config/af_mode_config.yaml:450):「Point 93 顯示即使 sharpness 比 baseline 提升 13.9%,整張影像仍然全域模糊」——這是硬體 FPGA AF 誤判成功的真實案例,不是理論風險。

**建議替代方案**(用硬體側現有機制取代,不用軟體掃描):

- 開啟 `fpga_af_repeat_check`(跑 2 次 FPGA AF,比對兩次鎖定電流是否一致)——增加 1 次 FPGA 呼叫(約 1 個 timeout 週期內完成,遠比 121 步軟體掃描快),可以抓到「掃描結果不穩定」這種假成功。
- 或開啟 `fpga_af_best_of_n_enable`(N=2~3),取最銳利的一次——同樣是 N 次 FPGA 呼叫而非 N 次「拍照+移動」。
- 這兩個機制都不會重現軟體掃描「整條清晰度曲線平坦度驗證」的能力,但可以用更低的成本抓住「單次結果不可信」的訊號,值得先試,再看實際案例是否被攔下來。

#### 2. `hardware_timeout_s`(15s)+ `status_fallback_wait_s`(2s)是失敗案例的固定拖累,建議先量測再調

目前完全沒有 FPGA AF 實際耗時的量測數據(程式碼裡沒有,只有 runtime 才產生的 `poll_trace`)。**在動任何參數之前,建議先加輕量 log**:每次 `trigger_hardware_liquid_af` 呼叫記錄「成功耗時」「失敗類型+耗時」,跑個幾百次拿到真實分布,再決定：

- 如果成功案例通常 2-4s 完成,15s timeout 可以大膽砍到 5-6s,失敗案例就能提早判定、提早重試,而不是白等 15s+2s。
- 如果沒有這個數據就直接砍 timeout,可能把「掃描比較久但最終會成功」的案例錯殺成 timeout,反而增加失敗率、逼出更多 retry。

#### 3. 檢視 `hardware_start_current`/`hardware_stop_current` 範圍寬度和 `hardware_frame_current` 步階

因為驗證出 `frame_count = 範圍寬度 / 步階`,如果目前 ±10mA(20mA 全範圍)、0.5mA 步階(40 張)對實際產品的對焦範圍來說太保守,**縮小範圍或加粗步階能直接降低單次 FPGA 呼叫要處理的張數**,這是少數能直接影響「單次成功案例」耗時的槓桿(相對於前面談的都是「失敗/重試案例」的耗時)。建議先跟良率/精度要求核對,不要一刀切縮小導致更容易 boundary 失敗。

#### 4. Mode 14/13 的 `hardware_two_stage_enable` + extension,建議收斂 fine stage 的 frame_count

目前 fine stage 預設沿用跟 coarse 一樣多的 `hardware_frame_count`(除非模式明確覆寫成 30,如 mode 8/9/12/13/14 已經有做)。既然 fine stage 只在鎖定值 ±10mA 內精掃,**沒有必要用跟粗掃一樣密的步階**,可以把 fine stage 的 frame_count 進一步降低(例如 10-15 張),因為搜尋範圍已經窄很多。

#### 5. ⚠️ **重要提醒:你說要刪除的是「軟體逐點掃描 AF」,但別忘了 mode 7 的 side-Zaber 掃描**

Side-Zaber 掃描(mode 7)雖然內部精掃用的是 `trigger_hardware_liquid_af`(硬體 FPGA),但它的**粗掃階段是純 Python 迴圈:4 個 Y 位置 × 11 個電流 × 3 張 frame = 132 次真實拍照 + 4 次物理 Zaber Y 軸移動**(Controller/hardware_drivers/unified_driver.py:7523)。這在架構上跟你要刪的「軟體逐點掃描 AF」(`run_liquid_lens_autofocus`)是**不同的函式**,砍掉軟體逐點掃描**不會影響到它**,但它極可能是你目前系統裡**最慢的單一路徑**,如果你的產線有跑到 mode 7,單單看它就足以讓 runtime 需求無法達成。

**建議**:

- 把 4 Y 位置降到 2 個(或用二分搜尋代替全網格)
- 11 個電流值改成先粗掃 3-4 點定範圍,再交給硬體 FPGA 精掃(反正精掃本來就已經是硬體了)
- `side_zaber_probe_frames` 從 3 降到 1(median 抗雜訊的邊際效益在拿掉 2/3 拍照次數面前不划算)
- 這樣理論上能把 132 次拍照砍到 20-30 次以下,是比刪軟體逐點掃描 AF 更立即見效的動作。

---

**總結行動順序建議**:(1)先加 timing log 量測硬體 AF 真實成功/失敗耗時分布;(2)刪除 mode 14 的軟體 fallback,同時開啟 `fpga_af_repeat_check` 或 `best_of_n=2` 當替代保護;(3)砍掉/精簡 mode 7 的 side-Zaber 132 次粗掃;(4)根據第 2 步的量測數據調整 `hardware_timeout_s`;(5)視良率數據評估是否收斂 `hardware_frame_count`/電流範圍。




#### 特別針對Strap使用到的autofocus方法
```
那如果我現在是要特別針對Strap使用到的autofocus方法(liquid lens af, keyence af)要非常非常大的降低running time, 要用最單一優化的liquid lens autofocus跟優化的keyence autofocus, 並利用cpython or c++方法加速, 評估要如何設計對strap plan1到plan4都加速的計畫以及預估可以加速多少. 也深入分析除了autofocus之外, 還有甚麼可以再減少strap plan1到plan4的方法? 或有沒有其他更優化plan5,6等等
```
### 重要前提說明

先講一個關鍵發現:Plan1~4 用的**是同一套執行引擎**——`UnifiedHardwareDriver.execute_template_point`(Controller/hardware_drivers/unified_driver.py:10481)。Plan1/2/3/4 只是「要走哪些站點、用哪個 af_mode」的**計畫層**差異(定義在 core/strap_workflow),真正執行 AF、拍照、settle 的**是同一段程式碼**。所以你要做的「統一優化 liquid lens/keyence AF」**本來就會自動套用到所有 4 個 plan**,不需要分別改。

也要先提醒:**Plan 1 目前是 `production_enabled: false`(純工程/dry-run,從未真的跑在硬體上)**,Plan 4 是目前 active production plan。所以效能優化實質上是針對 Plan 4(順帶讓未來若重啟 Plan 2/3 也受益)。

---

### 一、統一優化的 Liquid Lens AF 設計

#### 現況問題

Plan 4 的 component AF 用 mode 13(前/後)、mode 14(側面),兩者都開 `hardware_two_stage_enable`——**固定跑 2 次完整 FPGA 掃描**(粗 40 幀 + 精 30 幀),mode 13/14 還加 `fpga_af_baseline_validation_enable`(額外 1-3 張參考影像)。而根據我們前面驗證出的關係式 `frame_count = 電流範圍寬度 / frame_current 步階`,**掃描範圍越寬,單次掃描本身就越慢**。

#### 建議的單一優化設計

**核心思路:用「位置延續性」取代「兩段式重掃」**——strap 上相鄰的 link 位置(4001→4028)幾何上是連續的,對焦電流值理論上也會連續變化,不需要每個點都從頭掃 ±10mA 全範圍。

單一 FPGA AF 呼叫(取代 two-stage 雙掃):
  1. 若有上一個相鄰點的鎖定電流 → seed_current = 上一點結果
     否則 → seed_current = 該位置的出廠校正查找表(LUT)中心值
  2. 用 seed_current ± 3~5mA(而非全範圍 ±10mA)做「單次」FPGA 掃描
  3. hardware_frame_count 相應降到約 12~15 張(而非 40)
  4. 只有當結果落在這個窄範圍的邊界(boundary)時,才觸發一次 liquid_af_extensionN
     擴大範圍重掃——而不是不論如何都固定跑兩次

- 這需要建立一個**每個 internalnum 位置的電流校正 LUT**(工程模式先跑一輪全範圍掃描記錄每個位置的鎖定電流,存成 config,量產時直接當 seed)——這是一次性投資,換來長期單次掃描張數減半以上。
- mode 14 目前解決「弱鎖定/全域模糊」問題(Point 93 案例)靠的是軟體逐點掃描 fallback。前一輪分析建議過:改用 `fpga_af_repeat_check`(2 次 FPGA 掃描比對一致性)取代,現在加上「窄範圍掃描」後,repeat-check 的 2 次呼叫總成本仍然遠低於「兩段式(2 次×40幀)+ baseline 驗證」的現狀。
- 光源穩定 sleep(`unified_driver.py:11755` 的 0.15s)如果相鄰點燈光設定沒變,不需要每點都重新 settle——可以在同一個 view 的連續 component 點之間**保持燈常亮**,只在真的切換燈光 profile 時才 settle。

#### 統一優化的 Keyence AF 設計

好消息:我深入追蹤了 `_read_keyence_raw_m_channels`/`sample_offsets` 的實際邏輯(Controller/hardware_drivers/unified_driver.py:5689),確認了**正常成功案例(happy path)其實已經很快**——先測中心點一次,只有讀值失敗才會進入 ROI/Z 擴展重試網格。單次成功測量大約是:1 次移動 + settle(50-80ms)+ OUT2 讀值 + OUT1 的 5-sample 穩定讀值(~50ms),總計約 150-250ms 額外開銷,**這部分不是主要瓶頸**。

**真正的 Keyence 時間成本在「表面幾何掃描」,不是 component AF**:Plan 4 每個 view(側/前/後)有 **18~21 個純幾何量測站**(不拍照,只讀 Keyence 定義表面),三個 view 加起來約 55-65 站。這些站目前是**逐點全部量測**,沒有用插值。

**建議**:把幾何掃描也改成「錨點 + 插值」——例如每個 view 只在 5-7 個關鍵位置做真實 Keyence 量測,其餘位置用插值估算表面(Plan 2/3 的 component AF 本來就是這樣做,只是還沒套用到幾何掃描上)。這能把 ~60 站砍到 ~20 站左右。

---

### 二、C++ / CPython 加速的定位:只在一個地方真正划算

延續前面的結論——AF 本身的瓶頸是硬體等待/機械移動,C++ 幫不上忙。但這次深入 Strap 執行流程後,我找到一個**真正 CPU-bound、且在關鍵路徑上同步執行**的地方:

**`unified_driver.py:12349-12354`**——每次拍照後,**同一個執行緒**同步做 `apply_color_calibration`、`apply_post_color_tone`、`_apply_capture_rotation`,做完才能進入下一站。這段是否已經向量化(numpy/cv2)還是逐 pixel Python 迴圈,值得先檢查——如果是逐 pixel,這才是這個系統裡少數 C++/Cython 重寫真正有意義的地方。

**但建議順序是**:

1. **先用背景執行緒把這段處理跟下一站的移動重疊**(零風險,不用 C++,Python threading 就能做)——如果影像處理時間本來就比移動時間短,重疊後這段成本幾乎完全被藏起來,C++ 就沒有必要了。
2. 只有在重疊之後**還是**測出這段比移動時間長、拖累節奏時,才值得評估把色彩校正/旋轉這幾個函式用 pybind11/Cython 重寫——而且大機率它們已經是 `cv2`/`numpy` 向量化操作,重寫效益有限。

SAM/screw ML 推論(可能 CPU-only,取決於有沒有 CUDA)如果真的是 CPU 跑,優化方向也不是「手寫 C++」,而是確認 GPU 有沒有被正確使用,或考慮縮小輸入解析度/量化模型——這些比重寫底層運算划算得多。

**結論:C++ 在 Strap AF 加速計畫裡不是主戰場**,真正該投入的是流程重疊(threading/pipeline)跟前面的 AF 掃描範圍收斂。

---

### 三、AF 之外還能砍哪裡(依影響力排序)

| 排名  | 項目                                         | 現況                                                                                                                                                                     | 建議                                                           | 預估影響                                                                                                  |
| --- | ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------- |
| 1   | **固定 1.0s capture settle**                 | Controller/hardware_drivers/unified_driver.py:12310 每次拍照無條件 sleep 1.0s,Strap 沒有覆寫這個值(禁用中的 macro_cam_1 fast pipeline 已證明 0.25s 可行,config/strap_macro_cam1_scan.yaml:21) | 幫 Strap component 拍照路徑加一個專屬 override,降到 0.25-0.3s            | ~30+ 次拍照 × 0.7s 節省 ≈ **20-25s**                                                                       |
| 2   | **`keyence_policy: every_physical_point`** | Plan 4 對全部 ~30 個 component 點都做完整 Keyence+liquid AF;Plan 2/3 只在 5 個錨點做完整 AF,其餘插值                                                                                        | 評估是否所有位置都需要完整 AF精度,若可以改回錨點+插值(但保留每個點都拍照)                     | **最大槓桿**,但需要良率/精度團隊確認,不是免費午餐                                                                          |
| 3   | **前/後 view 轉場做了兩次**                        | stitch 順序是 側→前→後,component 順序卻是 後→前→側(相反)(core/strap_workflow/plan4.py:142)→ 需要**兩趟**完整 RX/RZ 轉場,每趟約 17s                                                               | 檢查能否讓 component 順序跟 stitch 順序一致,只轉場一次                        | ~**17-34s**,**但這個順序是刻意改的**(commit 訊息「strap plan4 reorder - err2 fix」暗示是修 bug 才改的,動之前務必先搞清楚原因,別引入舊 bug |
| 4   | **表面幾何掃描站數(~55-65站)**                      | 逐點量測,無插值                                                                                                                                                               | 改錨點+插值(同第一節 Keyence 建議)                                      | ~**15-20s**                                                                                           |
| 5   | **Y-offset 重試迴圈**                          | [`focus_y_search_offsets_mm`] 側面訊號不佳時最多重試到 8 個 offset,每次都要重新移動+settle+拍照                                                                                               | 先加 log 統計實際觸發頻率,若常觸發,代表初始定位/校正需要改善(治本),而不是單純縮短重試列表(治標可能犧牲良率) | 依觸發率而定,可能是隱藏的大成本                                                                                      |
| 6   | **每段動作前後的 Zaber 讀值**                       | 每個 motion segment 都有獨立多軸序列讀值(Controller/hardware_drivers/unified_driver.py:11356 等多處)                                                                                  | 評估哪些讀值是防禦性/非必要,改成抽查而非每站必讀                                    | 每站省 0.1-0.2s,累積 ~90 站 ≈ **10-15s**                                                                    |
| 7   | **後處理與下一站移動重疊**                            | 目前色彩校正/旋轉是同步阻塞(Controller/hardware_drivers/unified_driver.py:12349)                                                                                                    | 背景執行緒重疊(見上節)                                                 | 視處理時間而定,可能隱藏掉這整段                                                                                      |

---

### 四、粗估總加速效果(重要:這是根據程式碼常數推算的示意估計,repo 裡完全沒有實測 timing 數據,務必先加 telemetry 驗證)

|階段|現況粗估|優化後粗估|備註|
|---|---|---|---|
|Capture settle|~35s|~10s|高信心|
|Liquid lens AF(~30 component 點,兩段式→單次窄範圍)|~90s(*)|~35-45s|(*)FPGA 內部真實掃描時間未知,這個數字是用 frame_count 比例反推的粗估,信心較低|
|Keyence component 量測|~6s|~4s|本來就快,空間有限|
|幾何表面掃描(錨點化)|~30s|~10s|中信心|
|前後轉場(若能只做一次)|~34s|~17s|**需先確認安全**,不確定能不能改|
|其他讀值/雜項|~15s|~8s||
|**合計**|**~210s(~3.5分)**|**~85-95s(~1.5分)**|**約 50-55% 縮短,但這是示意值**|

**強烈建議先做的第一步**:在 `execute_template_point` 跟 `trigger_hardware_liquid_af` 加入實際耗時 log(進站/AF開始/AF結束/拍照/下一站,各自時間戳),跑一輪真實產線資料,把上表換成真數字——目前所有「多少秒」的估計都是根據程式碼裡的迴圈次數/sleep 常數反推,FPGA 內部真實掃描時間完全是黑盒。沒有這個數據,任何優化投入的優先順序都可能賭錯方向。

---

### 五、有沒有更好的 Plan 5 / 6 架構?

在現有 Plan 1-4 的基礎上,我認為有兩個值得評估的新方向,但**都需要跟量測精度/良率團隊確認可行性**,不是純工程單方面能決定:

#### Plan 5 構想:「Geometry-first 順序 + Anchor-based AF」混合架構

沿用 Plan 4 已經修好的站點順序/bug fix(不要重新引入 Plan 4 修過的問題),但把 `keyence_policy` 從 `every_physical_point` 改回 Plan 2/3 的錨點+插值——**每個點依然拍照**(不犧牲影像涵蓋率),但只有 5-7 個錨點做完整 AF,其餘用插值/延續電流值。這是影響力最大的一項,值得優先評估。

#### Plan 6 構想:「全流水線重疊(Pipeline Overlap)架構」

目前整個站點迴圈是完全序列化(移動→settle→AF→拍照→後處理→下一站移動),只有 SAM/screw 分析在背景跑。真正的架構升級是讓:

- 站點 N 的影像後處理,跟站點 N+1 的移動**同時進行**(目前只有 SAM 這樣做,色彩校正/旋轉沒有)
- 若 Zaber 控制器與相機觸發是獨立匯流排,評估「預先規劃/預先定位下一站」跟「當前站的 settle 收尾」是否能重疊,把 settle 從關鍵路徑上移除

這兩個方向都不是「調參數」層級,是架構調整,實作前建議先用 telemetry 數據確認理論估計的加速效果值不值得投入這個工程量。




```
了解, 那在以修改python code的前提下根據你的統一優化的 Liquid Lens AF 設計的建議, 是否會影響到其他不是strap的watch的autofocus? 是否有風險讓原本已經正確對焦的image變成out of focus? 另外除了autofocus之外, 在其他zaber, light, 或流程上還有其他可以縮短running time的任何地方? 或許是拍照流程, 等待時間, 或image傳輸或程式life cycle等等因為目前瓶頸就是running time大大超出需求5倍以上, 所以這是第一需求, 而其次是不要影響到其他autofocus除非可以確定可以縮短time但不會有out of focus風險, 請深入分析
```
### 一、優化的 Liquid Lens AF 設計是否會影響 Watch AF——結論:架構上安全,但有兩個關鍵前提要處理

#### 好消息:config 隔離機制本身是乾淨的

我讓 agent 追蹤了 `af_mode_id` 從「哪個capture模板讀出來」到「怎麼合併成最終設定」的完整路徑,結論很明確:

# unified_driver.py:10058-10072  _effective_config_for_af_mode()
cfg = copy.deepcopy(self.config)                # 每次都從乾淨的 base config 複製
self._apply_af_mode_overrides_to_config(cfg, overrides)  # 只套用「這一個」mode 的 override

**每次 AF 呼叫都是「複製一份乾淨 base config + 只套用當下這個 af_mode 的 override」,不會累積、不會互相污染。** `af_mode_id` 本身也是每個 capture 模板自己帶的值(`unified_driver.py:10764`),Watch 的 capture 模板不可能意外拿到 Strap 的 af_mode 設定。**所以只要新設定只加在正確的 af_mode 裡,理論上不會動到 Watch。**

#### 但有一個關鍵地雷:**mode 8 不是 Strap 專屬**

我原本假設的「seed from previous point」適用範圍需要修正——追查後發現 **mode 8 同時被 Watch 跟 Strap 使用**:

- Watch:config/internalnum_config.yaml 裡 Front `0039`-`0050`(側邊 40° 拍攝點)明確用 `af_mode_id: 8`
- Strap:`strap_macro_cam1_scan.yaml:318`、`strap_component_autofocus.yaml:27` 的「非錨點/tile 掃描」也用 mode 8

**這代表:如果你把「窄範圍種子掃描」這個優化加在 mode 8 上,會同時改到 Watch Front 0039-0050 的對焦行為。** 這是唯一一個真正有風險的交集點,必須避開。

#### 另一個要修正的前提:4001-4028 目前根本沒在跑 two-stage

我原本以為 Strap link 位置(4001-4028)用的是 two-stage 雙掃描——實際查證發現**不是**:這些點用的是 mode 3,mode 3 的 override 只碰 Keyence 的 Y/Z,**沒有設定 `hardware_two_stage_enable`**,而 base config 裡這個值預設也是 `False`。也就是說 **4001-4028 目前本來就已經是「單次」FPGA 掃描**(±10mA、40幀)。所以我們要優化的目標不是「拿掉 two-stage」,而是「把這個單次掃描的範圍,從固定 ±10mA 收窄成用前一點種子值 ±3-5mA」。

#### 安全的實作方式:嚴格 opt-in,只加在確認為 Strap 專屬的 mode

# 沿用現有 unified_driver.py 裡其他可選旋鈕(如 liquid_af_extensionN)的模式
effective_robust_cfg.get("liquid_af_seed_from_previous_locked_current", False)  # 預設 False
effective_robust_cfg.get("liquid_af_seed_half_range_ma", 5.0)

**這兩個新 key 只加進 mode 3、12、13、14、16 的 override 區塊,絕對不加進 mode 8。** 因為隔離機制是「每個 mode 各自 merge」,沒設定這個新 key 的所有其他 mode(包含 mode 8、以及全部 Watch 用的 mode)**設定值會跟現在一模一樣,行為完全不變**。這是這個 config 架構的優點——加新旋鈕本身不會動到既有行為。

#### 真正的風險不在 config,在「新邏輯本身」——這裡才是會不會 out of focus 的關鍵

有兩個必須處理的工程細節,這兩個沒做好,才是真正會讓對焦跑掉的地方:

1. **目前沒有跨呼叫的「前一點鎖定電流」狀態**:`self._last_liquid_af_report` 是單一個 instance attribute,會被任何相機、任何點的 AF 呼叫覆蓋掉,不是按 internalnum/相機分開存的。要做這個功能,需要新增一個真正的「按位置索引」的快取,而且要確認 capture 迴圈**真的是嚴格按 4001→4002→...→4028 順序跑**(不是被 retry 或多相機交錯打亂順序)——如果順序不保證,種子就可能對錯位置,反而害後面的點對焦跑掉。
2. **窄範圍掃描如果種子錯了,可能找到局部最亮點而不是真正焦點**——這才是「image變成out of focus」的真實風險機制。**解法**:窄範圍掃描結果如果落在邊界(boundary),必須自動觸發 `liquid_af_extensionN` 擴大範圍重掃(這個保護機制現在就有,不用新建),絕對不能讓窄範圍掃描的結果「無條件被接受」。只要保留這個 fallback,最壞情況也只是「這一點多花一點時間重掃」,不會變成「安靜地鎖錯焦」。

**結論:這個優化架構上可以做到不影響 Watch,但要正確實作「種子快取 + 邊界擴展保護」這兩個機制,不是單純改幾個 config 值就結束。** 建議先在 Strap 專屬的 mode(3/12/13/14/16)上小範圍試,並保留完整的邊界擴展 fallback 當安全網。

---

### 二、AF 之外的新發現——這批對「5倍超標」這個當前最優先問題可能影響更大

這輪深挖找到幾個**之前完全沒分析過**的地方,而且有幾個看起來影響不小:

#### 🔴 最大新發現:**燈光每個拍照點都關再開,每次開關是 30-60 次同步 Modbus 網路通訊**

`apply_lights`(Controller/hardware_drivers/unified_driver.py:11286)**每個拍照點都無條件執行**,一個點最多循環 ON→OFF 2-4 次。而每一次 `apply_lights("ON")` 實際上是 **9 次獨立的同步阻塞 Modbus TCP 交易**(2 個 ring light × 3 個暫存器寫入 + 2 個 spot light + 1 個 side light,Controller/hardware_managers.py:4284),每一次 `apply_lights("OFF")`(`emergency_stop`)又是至少 4 次交易。**這代表單一個拍照點,光是燈光控制就要跑 30-60 次同步網路來回**,而且程式碼裡**完全沒有比對「這次要設定的燈光值,跟上一點是不是一樣」**——連續兩個用同樣燈光設定的 Strap link 點(例如 4001→4002),也會整套重新關燈開燈。我沒找到任何「必須每點重新校正燈光(例如防閃爍/熱漂移)」的註解或說明。

**這是目前為止找到、風險最低但潛在影響可能最大的一項**——只要加一個「跟上一點燈光設定比對,一樣就跳過」的判斷,理論上不會動到任何對焦邏輯,純粹是省掉不必要的硬體通訊。**強烈建議優先處理**,但**建議跟你確認一下燈光每次都重開是不是有你們知道、但沒寫在程式碼裡的硬體理由**(例如燈具需要重新歸零校正)——如果沒有,這是免費的午餐。

#### 🟠 `af_mode_config.yaml`(542行)每次 AF 判斷都重新讀檔+重新解析 YAML,完全沒有快取

Controller/hardware_drivers/unified_driver.py:9994 的 `_load_af_mode_config` **每次呼叫都重新開檔+重新 `yaml.safe_load`**,而單一個 capture 點內部就會呼叫到這個函式**最多 7-8 次**(`unified_driver.py:10775,11449,11463,11486,11799,11855,11920,11932`)。這是**純粹的效能問題,不影響對焦正確性**——加一個 mtime-based 快取(檔案沒變就用記憶體裡的結果),邏輯完全不變,零風險,是最安全的第一個優化項目。

#### 🟠 拍照後的圖片存檔是同步阻塞,擋住下一站的移動

App/main.py:7020-7050 的 `_save_scratch_template_image` 直接同步呼叫 `cv2.imwrite`,沒有背景執行緒/queue。下一站的移動要等這次寫檔完成才能開始。建議改成背景寫入佇列(這也是零風險,純粹排程調整,不影響對焦邏輯或畫質)。

#### 🟡 效能記錄(perf log)也是同步寫檔,單一 capture 可能觸發 5-10 次

`RuntimePerformanceRecorder.record()`(core/workflow_manager.py:61)每次都是「開檔→寫入→關檔」,雖然是小檔案 append,單次可能很快,但頻率高(單一拍照可能觸發 5-10 次)。建議改成緩衝批次寫入。

#### ✅ 好消息:Zaber 多軸移動、影像傳輸/Bayer 解碼,都已經是乾淨、正確的實作

- **多軸移動確實是同時發出、同時等待完成**(`move_axes_absolute`,Controller/hardware_managers.py:582),不是你可能擔心的「一軸一軸依序等待」,這部分沒有額外可省的時間。
- **影像格式轉換(Bayer demosaic)用的是 `cv2.cvtColor` 原生實作**,沒有多餘的複製或轉換——GenTL buffer 只複製一次(而且是必要的,因為底層 buffer 馬上要被回收重用)。這部分沒有隱藏的浪費。

---

### 三、建議優先順序(針對「5倍超標」這個第一優先目標,且都不影響對焦正確性)

|順序|項目|風險|預期效益|
|---|---|---|---|
|1|**燈光同配置跳過重複開關**|極低(純比對邏輯,除非有隱藏硬體理由需先跟你確認)|**可能是這輪找到影響最大的一項**——每點省下 30-60 次網路交易裡的絕大部分|
|2|**`af_mode_config.yaml` 加 mtime 快取**|零(純 I/O 快取,不改變任何解析結果)|每點省 7-8 次檔案 I/O+YAML 解析|
|3|**圖片存檔改背景佇列**|低(純排程,不影響畫質/內容)|讓下一站移動不用等寫檔完成|
|4|**perf log 改批次寫入**|低|視觸發頻率而定|
|5|**Liquid lens AF 種子快取(前述設計)**|中(需要正確實作快取索引+邊界擴展保護,只加在 mode 3/12/13/14/16,避開 mode 8)|單次掃描張數可望減半以上|

建議先做 1-4(都跟 AF/對焦邏輯完全無關,可以放心先動,累積起來對「5倍超標」的貢獻可能不小於單獨優化 AF),同時要求量測實際數字驗證效果;AF 種子快取這項因為牽涉到對焦正確性,建議排在後面,而且需要先加 timing/成功率 telemetry,確認前面幾項優化後還有多少落差需要靠它補。



```
了解, 那請幫我根據你之前提的統一優化的 Liquid Lens AF 設計的建議提出完整的修改code計畫修改plan2, plan3, plan4, 儘量侷限在strap所以如果有同時在watch及strap都適用的可以定義新的af_mode, 除非確定不會對對焦有影響只影響到time, 譬如running time.

在liquid lens autofocus方面都使用單次硬體 FPGA不要額外輪詢迭代, 不要做`_dump_af_diagnostics()`, `self.logger.info, 或其他. 也使用`hardware_two_stage_enable: false`以及`liquid_af_extensionN: 1`, 不要軟體逐點掃描 fallback、不要多點信心度驗證重試, **把 lens 移動的 settle 時間跟上一步的 sharpness 運算重疊**(pipeline 化):目前是完全序列的` set_liquid_lens_raw → sleep(settle) → capture → sharpness`,可以用一條背景 thread 讓「移動下一個電流值+settle」跟「算上一張圖的清晰度」同時進行

而且分析單次呼叫的硬編碼 sleep 鏈是否還有可以優化的地方(單次呼叫的硬編碼 sleep 鏈->關閉 `oeLiquidLensAutofocusEnable` 後->設定 `CurrentMode` 後->關閉 `TriggerMode` 讓 FPGA free-run 後->清空舊 frame buffer->啟用 `oeLiquidLensAutofocusEnable` 後->下達 `AutofocusStart` 指令後(確認過,這行就是 `time.sleep(0.15)`)->每輪 20ms drain + 讀 status + 讀電流->AF 成功後 latch 電流->`status_fallback_wait_s`,只有輪詢逾時才會走到)儘量優化, 重新檢視 FPGA AF 前後那串 50/80/150ms 的固定 sleep

在Keyence autofocus方面檢查是否有zaber重複行為或其他異常行為, 譬如每次讀 OUT1 都是5 次序列式 TCP 往返這是正常的測量媽還是重複五次? read_out3_for_debug=True是否真的打兩次?因為strap絕對是no glass所以strap測量不需要out3. `_af_keyence_matlab_like`裡面是否有重複移動了一次到同一座標?

而拍照時每次 capture_image()都會重新 arm software trigger 並 drain buffer 最多 150ms這是正常的嗎還是是多餘的? unified_driver.py:12298-12310:AF 完成後、每次擷取影像前,無條件time.sleep(min(capture_settle_time_s, 10.0))這是正常的嗎還是是多餘的?

在Strap的autofocus即使liquid lens af失敗也不fallback軟體逐點掃描, 如果沒有特別設定失敗要使用的方法則預設用current=0.

而在strap的zaber移動除了下一步是要在camera position拍照, 不然不要刻意放慢機械移動, 以及固定 1.0s strap capture settle改成0.25-0.3s
```
## Strap Liquid Lens / Keyence AF 優化——完整修改計畫

### 設計原則(貫穿全部修改)

1. **能用「只針對特定 af_mode 的 config override」解決的,絕不碰共用程式碼的預設行為**——因為 `_effective_config_for_af_mode` 每次都是「深copy base config + 只套用該 af_mode 的 override」(前輪已驗證),所以只要新 key 只加在 Strap 專屬的 mode(3/12/13/14/16),Watch 用的 mode(0/1/2/5/6/8/9/10/11)行為完全不變。
2. **必須改共用函式硬編碼常數的地方(FPGA sleep chain、Keyence 5x取樣),一律改成「可由 config 覆寫,預設值=現狀」**——這樣程式碼變動本身不改變任何人的行為,只有 Strap mode 顯式覆寫新 key 才會生效。
3. **唯一允許「全域套用、不特別限制在 Strap」的例外**:Keyence 重複移動的 bug fix(第四節)——因為修正後最終停留座標完全相同,只是省略掉多餘的移動指令,可以證明「只影響時間、不影響對焦結果」,符合你設的判斷標準,所以我建議全域套用而不是只給 Strap。

---

### 一、Liquid Lens AF 統一快速路徑——mode 3 / 12 / 13 / 14 / 16 的 config 改動

不需要新增 af_mode——mode 12/13/14 目前雖然透過 `<<: *af_mode_8_overrides`/`<<: *af_mode_3_overrides` 繼承共用 mode 的設定,但**它們自己的 override 區塊已經在後面重新覆寫過所有關鍵 key**(現有程式碼就是這個 pattern,例如 mode 13 已經重新設定 `hardware_two_stage_enable` 等等),所以我們只要在 mode 3/12/13/14/16 自己的 override 裡加上/改成你要的值即可,**完全不用動 mode 8**。

#### 各 mode 要改的 key(對照 config/af_mode_config.yaml 現況)

# mode 13(前/後 component)—— 現況 hardware_two_stage_enable: true, liquid_af_extensionN: 0
# 改為:
hardware.autofocus.cameras.macro_cam_1.liquid_af.hardware_two_stage_enable: false      # 現況 true → 改 false
hardware.autofocus.cameras.macro_cam_1.liquid_af.liquid_af_extensionN: 1               # 現況 0 → 改 1(單次邊界重試,不是雙倍兩段掃描)
hardware.autofocus.cameras.macro_cam_1.liquid_af.fpga_af_baseline_validation_enable: false   # 移除多點信心度驗證重試
hardware.autofocus.cameras.macro_cam_1.liquid_af.fpga_af_baseline_fallback_enable: false
hardware.autofocus.cameras.macro_cam_1.liquid_af.best_effort_current_fallback_enable: false  # 不要额外的 best-effort fallback
hardware.autofocus.cameras.macro_cam_1.liquid_af.software_fallback_enable: false       # 現況已經是 false,保留
hardware.autofocus.af_settings.read_out3_for_debug: false   # 新增:Strap 無玻璃,關掉 OUT3(見第三節)

# mode 14(側面 component)—— 現況 hardware_two_stage_enable: true, extensionN: 2, baseline_validation: true, software_fallback: true
# 改為:
hardware.autofocus.cameras.macro_cam_1.liquid_af.hardware_two_stage_enable: false      # 現況 true → 改 false
hardware.autofocus.cameras.macro_cam_1.liquid_af.liquid_af_extensionN: 1               # 現況 2 → 改 1
hardware.autofocus.cameras.macro_cam_1.liquid_af.fpga_af_baseline_validation_enable: false   # 現況 true → 改 false
hardware.autofocus.cameras.macro_cam_1.liquid_af.fpga_af_baseline_fallback_enable: false     # 現況已 false,保留
hardware.autofocus.cameras.macro_cam_1.liquid_af.best_effort_current_fallback_enable: false  # 現況已 false,保留
hardware.autofocus.cameras.macro_cam_1.liquid_af.software_fallback_enable: false       # 現況 true → 改 false(移除 121 步軟體 fallback)
hardware.autofocus.af_settings.read_out3_for_debug: false
# 移除(或保留但改小)電流範圍:目前 -60..60(121步),既然不再靠軟體掃描救援弱鎖定,
# 建議收斂回跟 mode 13 一致的 -20..20,避免單次硬體掃描本身變慢(frame_count 隨範圍線性增加)
hardware.autofocus.cameras.macro_cam_1.liquid_af.hardware_start_current: -20.0
hardware.autofocus.cameras.macro_cam_1.liquid_af.hardware_stop_current: 20.0
hardware.autofocus.cameras.macro_cam_1.liquid_af.hardware_frame_count: 30

# mode 3(4001-4028 link 位置的 Keyence Y/Z override)—— 目前沒碰 liquid_af,只需加:
hardware.autofocus.af_settings.read_out3_for_debug: false

# mode 12(側面 macro_cam_1 Keyence,繼承 mode 8 tree)—— 同 mode 13 邏輯:
hardware.autofocus.cameras.macro_cam_1.liquid_af.hardware_two_stage_enable: false
hardware.autofocus.cameras.macro_cam_1.liquid_af.liquid_af_extensionN: 1
hardware.autofocus.af_settings.read_out3_for_debug: false

# mode 16(fast liquid-lens-only)—— 現況已經是 hardware_two_stage_enable: false,
# liquid_af_extensionN: 0 → 改成 1(你要求的值),其餘已經符合:
hardware.autofocus.cameras.macro_cam_1.liquid_af.liquid_af_extensionN: 1               # 現況 0 → 改 1
hardware.autofocus.af_settings.read_out3_for_debug: false

**重要提醒**:mode 14 目前開 `software_fallback_enable: true` 是因為 mode 14 的 comment 明確記載了「Point 93」這個真實案例(弱 FPGA lock 但整張圖模糊)。**移除它之後,弱鎖定的風險不會消失,只是不再有軟體掃描去攔截它**。你在需求裡說「Strap 即使 liquid lens AF 失敗也不 fallback 軟體掃描,沒設定就用 current=0」——這代表你是接受這個風險、選擇用「明確失敗 + current=0」取代「軟體掃描硬救」。這個取捨我會照做,但**強烈建議你們良率團隊知情且同意**,因為這是唯一一項會實質改變失敗時行為(而不只是時間)的修改。

---

### 二、FPGA 單次呼叫的 sleep chain——逐項分析與建議

我把 `trigger_hardware_liquid_af`(Controller/hardware_managers.py:2914)裡**成功路徑上**的每一個 sleep 都列出來了,並標註哪些有硬體理由、哪些是防禦性緩衝:

|#|位置|現值|有無硬體理由註解|建議|
|---|---|---|---|---|
|1|reset settle(2547-2548)|0.08s(config `af_reset_settle_ms`)|無,但已經是 config 可調|已可調,建議 Strap mode 收到 0.05s(需驗證)|
|2|關閉殘留 AF-enable node 後(3104)|**0.05s**|只有「Give the hardware a moment」這種通用註解|**改成 config-driven,Strap 收到 0.02s**|
|3|設定 CurrentMode 後(3115)|**0.05s**|無|**改成 config-driven,Strap 收到 0.02s**|
|4|關閉 TriggerMode 讓 FPGA free-run 後(3684)|**0.15s**|無量化理由,但這是最大宗、也最可能有真實 pipeline 反應時間需求的一個|**不建議在沒有實測驗證前縮短**——這是風險/效益比最差的一項|
|5a|清空舊 frame buffer(3688)|上限 0.1s,實際通常遠低於此(空 buffer 時第一次 fetch 就跳出)|—|不用動,已經是自我限速|
|6|重新啟用 AF-enable node 後(3707)|**0.05s**|無|**改成 config-driven,Strap 收到 0.02s**|
|7|下達 AutofocusStart 後(你確認的行)|**0.15s(line 3905)**|無量化理由,同樣是關鍵風險項|**不建議在沒有實測前縮短**|
|8|輪詢迴圈每輪(4084)|0.05s + 0.02s drain sub-loop|輪詢頻率,不是正確性風險|**改成 config-driven,Strap 收到 0.03s**(輕度縮短,只影響狀態偵測解析度,不影響最終鎖定電流)|
|9|Latch 電流(成功後)(3951/3982)|0.05s + 0.02s|無|**改成 config-driven,Strap 收到各 0.02s/0.01s**|
|10|最終 settle current(2547-2548透過`settle_time_s=0.03`)|0.03s|無|保留(數值已經很小)|

**加總估算**:項目 2/3/6/8/9 這幾個「無明確硬體理由」的緩衝,現況合計約 0.3-0.4s,如果驗證後能收斂到 Strap 建議值,大約可以省下 **0.15-0.2s / 次成功呼叫**。項目 4 跟 7(各 0.15s,合計 0.3s)是**風險最高、效益未知**的兩項——我的建議是**先不動**,除非你們願意在實驗室用示波器/實測驗證 FPGA 真正需要多少反應時間,盲目縮短這兩個有較高機率造成 AF 失敗率上升(進而觸發更多 boundary/timeout,反而更慢)。

**實作方式**:把每一個 `time.sleep(0.05)` 這種硬編碼常數,改成 `time.sleep(af_cfg.get("hw_af_<step_name>_sleep_s", 0.05))`,預設值維持現狀不變。Strap mode 的 override 再顯式設定較小的值。**這個改動本身(把常數變成可設定)不影響任何人的行為**,是最安全的第一步。

---

### 三、Keyence AF——你問的四個問題,答案都確認了

1. **OUT1 是不是真的 5 次序列式往返?** 確認是——`read_stable_channel(channel, samples=5, delay=0.01)`(Controller/hardware_managers.py:4474),內部真的是 5 次獨立 socket 往返,取 median。這是真的抗雜訊機制,但**找不到任何註解說明為什麼是 5 次而不是 3 次**。我的建議:改成 config-driven(`keyence_out1_read_samples`,預設 5 保留現狀),Strap 先保守改成 3,**但這項需要你們拿實際訊號做驗證**,因為這會改變量測噪聲,跟前面「只影響時間」的安全項目不同,屬於「可能有 out-of-focus 風險」類別,我不會自作主張改成 1。
2. **`read_out3_for_debug=True` 是不是真的打兩次?** **確認是,而且是個真正的 bug 等級浪費**——同一次測量裡,OUT3 被讀了兩次:一次在診斷 dump 裡(Controller/hardware_drivers/unified_driver.py:3116,不管 `do_z_af` 是不是 true 都會跑),一次在真正的 Z-AF 分支裡明確又讀一次(Controller/hardware_drivers/unified_driver.py:3179)。**好消息是這兩處都已經被同一個既有的 `read_out3_for_debug` config 開關控制**,不需要改程式碼——**Strap mode 直接把 `hardware.autofocus.af_settings.read_out3_for_debug` 設成 `false`,兩次 OUT3 讀取就都會消失,而且因為是透過 af_mode override 機制套用,不會影響 Watch。** 這是這次分析裡風險最低、最快就能拿到效果的一項。
3. **`_af_keyence_matlab_like` 是不是真的移動到同一座標兩次?** **確認是,而且比你想的更多**——非 angle-aware 路徑(Strap 預設走這條):先移動到 `keyence_pose`,接著呼叫的 `_read_keyence_absolute_output_with_expansion` 內部又對同一個 Z 做一次 Z-only 移動(Controller/hardware_drivers/unified_driver.py:5691),然後又對同一個 XY 做一次 XY-only 移動(Controller/hardware_drivers/unified_driver.py:5402)——**每次量測都有 2 次多餘的移動指令,而且每個都帶著完整的 safety/collision-check/route-planning 開銷**,不是輕量操作。Angle-aware 路徑(非 Strap 預設,但其他 mode 可能用到)甚至有 3 次移動到同一姿態。**這個修正符合你的「確定不影響對焦、只影響時間」標準——因為目標座標完全相同,修正後最終停留位置一模一樣**,所以我建議這項**全域套用**(不限 Strap),做法是在 `_move_keyence_probe_pose`/`_move_xyz_with_safety`/`_move_keyence_probe_xy` 加一個「目標座標跟已知目前座標在誤差範圍內(例如 0.001mm)就跳過實際移動指令」的短路判斷,但安全鎖檢查(`check_safety_lock()`)保留。

---

### 四、拍照流程——你問的兩個 sleep 是不是多餘

1. **`capture_image` 每次都重新 arm+drain(最多150ms)是正常的嗎?** 確認**沒有找到任何「自從上次拍照後 TriggerMode 有沒有被動過」的狀態旗標**,所以確實是每次都無條件跑一遍。但實測邏輯上這個開銷本身**已經是自我限速的**:`_arm_software_trigger` 如果 node 值本來就對就不會真的寫入(Controller/hardware_managers.py:2071),`_drain_camera_buffers` 沒有殘留 frame 時通常在 ~5ms 內就跳出,不是真的每次都吃滿 150ms。**唯一真正每次都固定發生的,是沒有 `TriggerArmed` node 的相機上那個無條件 0.02s sleep**(Controller/hardware_managers.py:2119)。這個項目**不是主要目標**,可以先不動,效益有限。
2. **`unified_driver.py:12298-12310` 那個無條件 `time.sleep(capture_settle_time_s)` 是正常的嗎?** **這才是真正的大目標**——我確認它跟上面 `capture_image` 內部的 arm/drain **是完全不同、額外疊加的機制**,發生在呼叫 `capture_image` **之前**,而且**附近完全沒有任何註解說明這個 1.0 秒是為了什麼**(不是機構穩定、不是曝光穩定,程式碼旁邊的註解只解釋「數值 clamp 邏輯」,不是「為什麼要等」)。既然找不到任何有記載的理由,而你也已經確認要改成 0.25-0.3s,這個改動風險相對可控——**建議做法是新增 Strap 專屬 override**(不改全域預設 1.0s,避免影響 Watch):

# unified_driver.py:12299 附近,capture_settle_time_s 目前讀 hardware.zaber.capture_settle_time_s(全域,1.0s)
# 改為:優先讀 per-af_mode override,沒有才 fallback 到全域值

# 在 mode 3/12/13/14/16 的 override 加:
hardware.zaber.capture_settle_time_s: 0.3

---

### 五、Zaber 移動——好消息,你要求的行為現況已經大致符合

我確認過 `z_focus` profile 現在就是 **80mm/s,跟 `default` 一樣**,程式碼裡的註解甚至明講「不要用不對稱的慢速 motor profile,Keyence 精度靠的是 step/range 配置跟同步讀值,不是靠刻意放慢」——這部分**已經是你要的狀態,不用改**。唯一比 default 慢的兩處都**確實只發生在最後靠近拍照姿態的收尾段**:`capture` profile(20mm/s)只用在最後 ~1mm/0.3mm 的精準逼近(Controller/hardware_drivers/unified_driver.py:5764),以及一個非常局部的例外(`endlink_4030_low_y`,40mm/s,只在 internalnum=4030、af_mode=15、Y 在 140-150mm 這個已驗證安全走廊內生效)。**我沒有找到任何 Keyence 表面掃描站、side-Zaber Y 移動、或轉場移動被刻意放慢的地方**——這部分你要求的「非到拍照位置前不要刻意放慢」現況已經符合,不需要修改。

---

### 六、關於「pipeline 化 settle + sharpness」的澄清

你提到的「`set_liquid_lens_raw → sleep(settle) → capture → sharpness` 用背景 thread 重疊」這個技巧,**在這輪修改後,Strap 的 component AF 主流程已經不會再出現這個 pattern 了**——因為 `hardware_two_stage_enable: false` + 移除軟體逐點掃描 fallback,component 對焦變成單次 FPGA 呼叫,不再有 Python 迴圈逐步調電流拍照。

**唯一還會出現這個 pattern 的地方,是 mode 7 的 side-Zaber 探測迴圈**(`_capture_sharpness`,Controller/hardware_drivers/unified_driver.py:7631,用於側面 Y 軸粗掃時逐一測試 11 個電流值)——這次你的需求沒有提到 mode 7,所以我沒有把它排進這輪修改範圍。**如果你也想套用這個 pipeline 化技巧,需要另外告訴我,我會把它排進 mode 7 的修改裡**,做法是背景 thread 提前送出下一個電流值的 `set_liquid_lens_raw` + settle,同時主執行緒對上一張已經拍好的圖算 sharpness。

---

### 七、失敗時 fallback 政策——current=0

現況追蹤下來,`af_reset_current` 預設本來就是 `0.0`,失敗路徑上多處都會 reset 回這個值。要讓「沒特別設定就是 current=0」這件事**明確、不依賴隱含的多層 fallback鏈**,建議在 mode 3/12/13/14/16 統一顯式設定:

hardware.autofocus.cameras.macro_cam_1.liquid_af.best_effort_current_fallback_enable: false
hardware.autofocus.cameras.macro_cam_1.liquid_af.af_reset_current: 0.0   # 顯式寫出,不依賴預設值隱含生效
hardware.autofocus.cameras.macro_cam_1.liquid_af.use_manual_cammag_on_af_failure: false

**例外**:mode 15(4029/4030 endlink 參考點)目前設定 `continue_capture_on_liquid_af_failure: false`——也就是「這兩個特定安全關鍵參考點,對焦失敗就整個不拍」。這是刻意的嚴格設計,我不會動它,除非你確認也要放寬。

---

### 八、修改檔案總表

|檔案|修改內容|風險|
|---|---|---|
|config/af_mode_config.yaml|mode 3/12/13/14/16 override 區塊(第一、三、四、七節列出的 key)|低(config-only,已驗證隔離機制)|
|Controller/hardware_managers.py|`trigger_hardware_liquid_af` 內 5 處 sleep 常數改成 `af_cfg.get(..., 現狀預設值)`|低(預設值不變,行為零改動除非顯式覆寫)|
|Controller/hardware_drivers/unified_driver.py|①`_move_keyence_probe_pose`/`_move_xyz_with_safety`/`_move_keyence_probe_xy` 加「同座標跳過移動」短路判斷(全域套用)②`capture_settle_time_s` 讀取邏輯加 per-af_mode override 優先權|①低(同座標保證同結果)②低(config-scoped)|
|Controller/hardware_drivers/keyence_angle_aware.py|若 angle-aware 路徑也要修同座標重複移動問題,需要在此檔案對應處也加短路判斷|低|

---

### 九、`_dump_af_diagnostics()` / `logger.info` 移除

這兩個是純日誌/除錯輸出,不影響對焦結果——建議加一個 `af_cfg.get("af_verbose_diagnostics_enable", True)` 開關(預設 True 維持 Watch 現狀不變),Strap mode 顯式設 `false`。這比直接刪除程式碼安全,萬一 Strap 現場除錯需要,隨時可以臨時開回來,不用重新部署程式碼。
