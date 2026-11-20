

|                                  |     |
| -------------------------------- | --- |
| [[#### active_plan: 3` 的實際執行路徑]] |     |
| [[#### 規劃active_plan: 4 ]]       |     |
| [[#### 實作active_plan: 4 ]]       |     |
|                                  |     |


#### active_plan: 3` 的實際執行路徑
```
當在App main.py page\_template\_new\_strap\_pre執行了pushButton\_TemplateRun\_213, page\_template\_new\_strap\_pre2執行了pushButton\_TemplateRun\_217, 目前預設是active\_plan: 3

請幫我把pushButton\_TemplateRun\_213到最後的流程, 分成數個不同階段(譬如一開始是在side view測量長度, 在side view拍照stitch source images, 然後side view合成stitch...)並加以邊號, 並列表詳細解釋每個階段zaber如何移動轉段, 是否有退到safety place, 是否使用Keyence autofocus, 是否使用liquid lens autofocus, af\_mode等以便後續要create active\_plan: 4
```

已完整追過目前 `active_plan: 3` 的實際執行路徑。最重要的結論是：

- `pushButton_TemplateRun_213`：先量測 Side strap 範圍，再依序完成 `Side → Front → Back` 三張 overview stitch；每張 stitch 完立即背景跑 SAM；全部 stitch 完成後，才依實際程式順序 `Back → Front → Side` 拍 component。
- 操作者翻面後按 `pushButton_TemplateRun_217`：完成 `9-clock overview → stitch → SAM → 9-clock components`，最後回 watch center。
- Side、Front、Back overview 的 source capture 都不再逐張跑 Keyence；只有 liquid-lens AF。
- 9-clock overview 才是每張 source 都跑 Keyence AF + liquid-lens AF。
- Plan 3 的所有動態 component 都重用 overview Keyence 曲面，不再跑 Keyence。

目前 workflow 選擇位於 [strap_workflow.yaml (line 142)](D:/Provenance Laboratories projects/ImagingLibWatch/config/strap_workflow.yaml:142)，實際 runtime 排序則來自 [plan3.py (line 17)](D:/Provenance Laboratories projects/ImagingLibWatch/core/strap_workflow/plan3.py:17) 與 [main.py (line 42797)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:42797)。

## 213 流程

| 編號     | 階段                              | Zaber 移動與 safety place                                                                                                                                                                                                                     | Keyence / liquid lens / AF mode                                                                                                                                                                                                          |
| ------ | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 213-00 | 按鈕入口與流程鎖定                       | 取得單一 camera-operation guard，等待既有 holder move 完成。此階段本身不移動 Zaber。強制使用 `macro_cam_2` overview pipeline。                                                                                                                                       | 按鈕雖有 load-verification wrapper，但目前 [hardware_config.yaml (line 134)](D:/Provenance Laboratories projects/ImagingLibWatch/config/hardware_config.yaml:134) 設為 `enabled: false`，所以不會進行裝載 Keyence 驗證。                                       |
| 213-01 | Side Wide-FoV：定位 Pin1           | 直接移到 Side Pin1 `4014` 的 Keyence pose：`X=128.9, Y=140.693, Z=73.213, RX=0, RZ=0`。流程明確要求「不要先做 Z approach」。沒有明確先退到 Y=160；使用 Plan-3 direct-Keyence route，但 target validator、Digital Twin 與 readback 仍有效。若當前角度不符合 direct route，planner 仍可分段或拒絕。 | 這是 Keyence 距離量測，不是正常 autofocus；沒有 liquid AF，也沒有 `af_mode_id`。讀取 5 samples，OUT1 必須在含端點的 `[-8,+8] mm`。                                                                                                                                     |
| 213-02 | Side Wide-FoV：Pin1 Y recovery   | 若精確 Pin1 pose 無有效 OUT1，只改 Y，依序嘗試 `+0.5,+1,+1.5,+2,-0.5,-1,-1.5,-2 mm`；X/Z/RX/RZ 不變。                                                                                                                                                        | 只讀 Keyence。第一個有效位置成為正式 Pin1。                                                                                                                                                                                                             |
| 213-03 | Side Wide-FoV：沿 X 找 Pin2／量測跨度   | 從 Pin1 每站 `X +10 mm`；每站先讀原 Y，失敗時再用同一組 Y offset 搜尋。最後一個有效站保存為 Pin2；第一個找不到有效表面的站表示 strap 結束。最大掃描跨度 210 mm；到界限仍未找到結束會 fail closed。沒有站間 safety retract。                                                                                        | 只讀 Keyence；表面高度必須與 Pin1 baseline 相差不超過 8 mm。這裡保存 Pin1、Pin2、全部測點及 `camera_x → final_z`；主要用來決定 stitch 範圍及 extension flag，不是單純只輸出一個 length 數值。[main.py (line 42280)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:42280) |
| 213-04 | Wide-FoV → Side overview 轉段     | **明確退 safety place**：Y 移到 160 mm、保留目前 Z、RX/RZ 折回 0°，並驗證 readback。                                                                                                                                                                          | 無 AF。                                                                                                                                                                                                                                    |
| 213-05 | Side stitch source images       | 依 Wide-FoV 曲面，以 source X 線性插值/外插 Z，再直接移到各 `macro_cam_2` camera pose。正常範圍為 4008–4014，但端點會依 Pin2/extension 動態裁切。source 間沒有明確退 Y=160。                                                                                                         | **Keyence mechanical AF 關閉**；每張跑 liquid-lens AF。保留 point 的 `af_mode_id=3`，但 `use_autofocus=0`、`use_liqud_af=1`，因此 mode 3 的 Keyence 部分不會執行。                                                                                               |
| 213-06 | Side stitch + 背景 SAM            | Zaber 不動；產生 `strap_right_side` stitched image。完成後立即把此 view 的 SAM 丟進獨立 thread pool，可與下一 view capture 重疊。                                                                                                                                    | 無 AF。                                                                                                                                                                                                                                    |
| 213-07 | Side → Front group 轉段           | **明確退 Y=160**、保留 Z、RX/RZ 折回 0，再由 planner 轉到 RX=90 的 Front group。                                                                                                                                                                           | 無 AF。                                                                                                                                                                                                                                    |
| 213-08 | Front Keyence anchor batch      | 先選 source X 的 10%、30%、50%、70%、90% 共最多五個 anchor。第一個 anchor 經安全 planner 到 Keyence；之後 probe-to-probe 必須只有 X 改變，Y/Z/RX/RZ 不得改變。batch 完成後回到第一張 camera source pose。                                                                              | 只跑 `macro_cam_2` Keyence MATLAB-like/angle-aware focus；沒有 liquid AF，也不是 watchpoint `af_mode_id` 流程。結果保存 `X → corrected camera Y` 曲面。[main.py (line 45049)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:45049)        |
| 213-09 | Front sources → stitch → SAM    | 各 source 以 anchor 曲面插值 Y。source 之間不退 safety place；完成 stitch 後背景啟動 SAM。                                                                                                                                                                     | source capture：Keyence 關閉、每張 liquid AF、`af_mode_id=3`。                                                                                                                                                                                   |
| 213-10 | Front → Back group 轉段           | 再次**明確退 Y=160**、保留 Z、折回 RX/RZ，之後轉入 Back group。                                                                                                                                                                                             | 無 AF。                                                                                                                                                                                                                                    |
| 213-11 | Back anchors、sources、stitch、SAM | 與 Front 相同：先五個 Keyence anchors，再以 X 插值 Y 拍 source，最後 stitch 並背景跑 SAM。anchor 之間只准 X move。                                                                                                                                                   | Anchor：只有 Keyence。Source：Keyence 關閉、每張 liquid AF、`af_mode_id=3`。                                                                                                                                                                         |
| 213-12 | Back components                 | 等待 Back SAM cache；依操作者在 tableView_24 選擇，拍 outer surface/endlinks 等 macro components，再視需要從真實 macro 圖第二次跑 screw SAM、拍 micro screws。macro phase 結束會**明確退 Y=160**；整個 view component phase 結束時 finally 又會確認一次 safety retract。                   | 所有動態點都不跑 Keyence；詳細 AF 規則見下表。                                                                                                                                                                                                            |
| 213-13 | Front components                | 從前一階段的安全位置進入 Front，等待 Front SAM，拍 underside/endlinks 等選定項目。macro 結束與整個 phase 結束都會安全 retract。                                                                                                                                               | 不跑 Keyence；使用 Front overview 的 Y 曲面。                                                                                                                                                                                                     |
| 213-14 | Side components                 | 等待 Side SAM，拍 3-clock link/endlink，以及選定時的 screws。macro 結束及 phase finally 都會安全 retract。                                                                                                                                                     | 不跑 Keyence；使用 Wide-FoV 取得的 Z 曲面。                                                                                                                                                                                                         |
| 213-15 | 固定 Endlink references 4029/4030 | 只有選取 `Bracelet-Endlink reference` 才執行。這兩點是特殊 validated pose；每次進出包含 4029→4030 都使用 **Y=190、Z=70 special staging**，不是一般 Y=160/保留 Z。                                                                                                           | `af_mode_id=15`：特殊安全 Keyence X-Z 搜尋 + liquid AF；兩者都必須成功。4029/4030 都是 RX=90，Keyence 結果修正 camera Y。                                                                                                                                        |
| 213-16 | 回 holder，等待翻面                   | 經 `move_compound` 安全 planner 回 `start_position = X0, Y292, Z150, RX0, RZ0`。這裡沒有另外呼叫一次一般 safe-retract，但目標經 Transition Guard 分段與 readback 驗證。完成後切到 `page_template_new_strap_pre2`。                                                           | 無 AF。操作者在此翻到 9-clock load。                                                                                                                                                                                                               |

### 213 component 的 AF 規則

|Component 類型|Keyence|Liquid lens|`af_mode_id`|
|---|---|---|---|
|Front/Back regular-link liquid anchors|否，焦平面來自 overview cache|是|13|
|Side regular-link liquid anchors|否|是|14|
|非 anchor regular links|否|否；直接套用插值後的 liquid current|8|
|動態 Endlinks|否|是，每點獨立 AF，不加入 regular-link current model|Front/Back=13、Side=14|
|Micro screw points|否|是，每一個 accepted screw 都 strict AF|3|
|固定 4029/4030|是，special-safe Keyence|是，strict|15|

Regular-link liquid anchors 使用 X 位置 `0%、25%、50%、75%、100%`。程式先拍這些 anchors，建立 liquid-current 曲線，其他 link 才使用插值 current；相關設定在 [strap_component_autofocus.yaml (line 19)](D:/Provenance Laboratories projects/ImagingLibWatch/config/strap_component_autofocus.yaml:19)，Plan-3 component override 在 [main.py (line 16600)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:16600)。

## 217 翻面後流程

|編號|階段|Zaber 與 safety|Autofocus|
|---|---|---|---|
|217-00|按 `pushButton_TemplateRun_217`|再取得 camera-operation guard，確認 213 的三個 stitched views 與 extension report 存在。load verification wrapper 同樣因 config disabled 而略過。|無 AF。|
|217-01|進入 9-clock source capture|Plan-3 217 的第一個 view 沒有額外呼叫 Y=160 retract；正常情況 Zaber 此時位於 holder `0,292,150,0,0`，由 Transition Guard 直接規劃到第一個 9-clock source。範圍為 4022–4028，並依 213 extension flags 動態裁切。|—|
|217-02|9-clock source images|每張 source 都執行 camera ↔ Keyence probe cycle。因 RX/RZ 相同且 RZ=0，可使用 Plan-3 direct-Keyence corridor，不需每張先退 Y=160；source 間依 X 掃描。|**每張都跑 Keyence AF + liquid AF，`af_mode_id=3`**。這是 Plan 3 overview 中唯一逐張 Keyence AF 的 view。|
|217-03|9-clock stitch + SAM|產生第四張 `strap_9clock_side` stitch，與原三張組成完整 frame_82；啟動/等待該 view SAM。|無 AF。|
|217-04|9-clock components|拍選取的 9-clock links/endlinks；若選 screws，先拍 macro link，再對真實 macro 圖做第二次 screw SAM，最後拍 micro screw。macro phase 後及 finally 都明確退 Y=160。|動態 component 不跑 Keyence。Regular/endlink liquid AF 用 mode 14；非 anchors mode 8 且兩套 AF 都關閉；micro screws 用 mode 3、只跑 strict liquid AF。|
|217-05|最終回 center|經安全 planner 回 `home_positions.watch = X322, Y78, Z21.779, RX0, RZ0`，不是 strap home。完成後進 template-name page 或 `page_template_new_strap`。|無 AF。|

217 入口與 dispatch 位於 [main.py (line 44136)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:44136)，最後返回邏輯位於 [main.py (line 42685)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:42685)。

## Safety place 的精確定義

一般 Plan-3 view/component retract 是：

- `Y = 160 mm`
- Z 保留當前值
- `RX = 0°`
- `RZ = 0°`
- readback tolerance：translation 0.05 mm、rotation 0.5°

雖然 config 另有 `safe_z_mm: 70`，目前 `stage_z_for_rotation: false`，所以一般 retract 不會為了轉 view 強制改成 Z=70。只有 4029/4030 special staging 明確使用 `Y=190, Z=70`。[hardware_config.yaml (line 417)](D:/Provenance Laboratories projects/ImagingLibWatch/config/hardware_config.yaml:417)

另外，Plan 2/3 的 distance-only top-down staging 被停用，但這不代表取消 safety；target validation、wall envelope、angle-aware transition、Digital Twin、axis limits 與每段 readback 仍然有效。

## 建立 active_plan 4 前需要先處理的落差

1. **Component order 設定與實際程式不一致。**  
    YAML 寫 `button_213_component_order: [front, side, back]`，但 runtime 常數是 `Back → Front → Side`。目前實際以 Python 常數為準。Plan 4 應只保留一個 authoritative source。
    
2. **Front/Back 命名容易誤判。**  
    `strap_right_front` 使用 4001–4007，但 internal description 寫的是 `Strap back / underside`；`strap_right_back` 使用 4015–4021，description 是 `Strap front / Outer surface`。Plan 4 最好以 `group_id + component surface + internalnum range` 定義，不要只寫 Front/Back。
    
3. **診斷 timeline 沒完整表示 runtime 尾段。**  
    `build_plan3_stage_plan()` 有 overview/components，但實際 orchestrator 另外增加 4029/4030 fixed references、holder return、217 center return。Plan 4 的 timeline 與 runtime 應由同一份 stage definition 產生。
    
4. **不能只複製 af_mode。**  
    Mode 13/14 本身包含 Keyence 能力，但 Plan 3 component 是靠 `use_autofocus=0` 強制不跑 Keyence。若 Plan 4 只複製 mode ID、漏掉 capture flags，就會意外重新啟動 mechanical AF。
    
5. **大量 allowlist 目前只接受 `{2,3}`。**  
    建立 Plan 4 時必須同步更新 App、SafetyRule、UnifiedHardwareDriver、direct-Keyence qualification、focus-cache persistence、diagnostic UI 與測試；否則可能被當成未知 plan，或失去 Plan-3 safety/AF 特例。
    

驗證方面，我執行了 16 個 Strap workflow unittest，加上 5 個 Plan-3 component/AF 核心測試，共 21 個全部通過。這次沒有修改任何檔案。



#### 規劃active_plan: 4 
```
假設我現在要把active_plan: 3然後修改成儘量依照我上傳的word裡面的規劃的pipeline, 要修改active_plan: 3那些地方要怎麼改, 請不要修改code而是提出完整新plan, 並指出不同的地方
```
結論：建議把 `active_plan: 3` 改成「先建立完整 strap 幾何模型，再依模型拍攝」的 Plan 3 revision 2，而不是沿用目前「Side 掃描一次、Front/Back 只量 5 個 anchor」的架構。

我沒有修改任何程式；工作目錄目前是乾淨的。

## 一、我對 Word 文件的解讀

Word 的主要精神是：

1. 以已知 Pin1 為基準。
2. 用 Keyence 每 10 mm 建立 strap 的三維表面資料。
3. 找出 Pin2，計算 strap 長度及旋轉後的位置。
4. 依表面資料把相機移到 liquid lens 約 0 current 的焦平面。
5. 拍 Side、Back、Front，並在背景執行 stitching／segmentation。
6. SAM 找到 link、screw 後，再對實際 component 逐點測高和拍攝。
7. Endlink 不是完全依賴固定座標，而是分析角度後計算旋轉位置。  
    strap_collection_pseudocode_ajp_082120261854.docx (page 1)  
    strap_collection_pseudocode_ajp_082120261854.docx (page 2)

文件裡「可直接使用 Zaber API」和「不實作 Digital Twin」是設計建議，不應解讀成停用目前已有的 motion guard。新版 Plan 3 應保留現有：

- safe retract
- RX/RZ 旋轉限制
- stage readback
- travel envelope
- Keyence target validation
- Digital Twin／safety rule 檢查

尤其文件自己也標示部分位置可能 collision、移動方向正負號尚待確認。

---

## 二、建議的完整新版 Plan 3

以下假設：

- `pushButton_TemplateRun_213` 執行 Word 的主要 Side／Back／Front／component／endlink 流程。
- Word 沒有定義目前 217 的完整 9-clock pipeline，因此先保留 `pushButton_TemplateRun_217`，避免破壞既有流程。
- 每次 RZ 旋轉前必須 `RX=0` 並進入已驗證的 rotation-safe pose。

### P3-00：載入、校正與 preflight

|項目|新規劃|
|---|---|
|Zaber|尚不移動；讀取目前 X/Y/Z/RX/RZ|
|Safety place|驗證目前位置可否安全進入 Side 起點；否則先走一般 safe retract|
|Keyence|不使用|
|Liquid lens|設為 calibrated neutral/0 current|
|AF mode|不適用|

必須確認：

- Pin1 calibration 有效。
- stage rotation center／fixture center 已校正。
- Keyence 有回應。
- camera-to-Keyence offset 版本正確。
- safety config 與 template revision 相符。

任何必要 calibration 缺失，不能只使用 Word 的假設值繼續。

### P3-01：Side Pin1 定位與第一個高度

Word 原案是在 Pin1 附近 `+2X`，由低 Z 往表面方向搜尋 Keyence。

|項目|新規劃|
|---|---|
|Zaber|先 safe retract；移到 Pin1 X/Y 的 Keyence staging position，再做 bounded Z approach|
|Safety place|進入前有；搜尋過程不需要每一步退 safe，但必須有硬體 Z 上下限|
|Keyence|使用，屬於幾何量測，不是相機 capture AF|
|Liquid lens|不使用|
|AF mode|不適用|

搜尋順序建議：

1. Pin1 nominal X/Y。
2. Y offset：`+0.5, -0.5, +1, -1, +2, -2, +4, -4 mm`。
3. 每個候選點做有限次 Keyence sample，取 median。
4. OUT1 必須在設定窗口內。
5. 成功後把 stage 移到 Keyence 中央量測範圍。

與目前不同：目前 Plan 3 直接到固定 `Z=73.213`，沒有 Word 所要求的 bounded Z approach。

### P3-02：Side 每 10 mm 掃描、建立 Pin2

|項目|新規劃|
|---|---|
|Zaber|維持 `RX=0/RZ=0`，主要沿 X 每 10 mm；只有無效 station 才搜尋 Y|
|Safety place|station 之間不退 safe；掃描結束、準備 RX 旋轉前才退|
|Keyence|每個 station 使用|
|Liquid lens|只有 Pin2 確認照片才使用|
|AF mode|幾何掃描不適用；Pin2 確認照片用 mode 3、Keyence AF off、liquid AF on|

每個 station 保存：

- requested X/Y/Z
- actual X/Y/Z/RX/RZ
- Keyence raw samples、median、validity
- 實際使用的 Y recovery offset
- `measured / interpolated / rejected-holder`
- uncertainty

停止條件不能只是第一次 miss。建議：

- 一個 miss：標記 missing，繼續下一站。
- 連續多站 miss，且已超過合理 strap 長度：以前一個最後有效站作為 Pin2 候選。
- 若後面重新取得有效值，missing station 只能由前後有效值 interpolation；不能用 missing 值帶領下一個 Y。
- holder rejection 必須有明確規則，例如 stage XY exclusion zone、OUT1/OUT2 特徵或影像 classifier，不能只留下文字判斷。

之後移到 Pin2 camera position：

1. 根據 Keyence 結果把機械焦距調到 liquid lens 約 0 current。
2. liquid AF。
3. 拍 Pin2 endlink overview。
4. machine vision 找 Pin2 中心。
5. 失敗時才 fallback 到最後有效 Keyence station，並標記低信心。

Word 在這裡還要求從 stage center、Pin1、Pin2 計算旋轉後位置。  
strap_collection_pseudocode_ajp_082120261854.docx (page 2)

### P3-03：Side source images、stitch、SAM

|項目|新規劃|
|---|---|
|Zaber|依 Side station curve 插值 Z；X 依 stitch overlap 計算|
|Safety place|同 view 連續拍攝不退；完成後退 safe|
|Keyence AF|關閉，因為已完成 Side surface scan|
|Liquid lens AF|configurable；建議 first/middle/last anchor 開，其餘插值|
|AF mode|mode 3，`use_autofocus=0`；liquid 開啟時 `use_liqud_af=1`|

每個 source position 先用 Zaber 把機械焦距調到預測的 0-current focal plane，再決定是否執行 liquid AF。這和「只設定 liquid current=0」不同。

拍完立即：

- 保存 raw source images。
- 啟動 Side stitch。
- stitch 完成後背景啟動 SAM segmentation。
- 主流程可以進入 Back，不必等待 SAM。

### P3-04：Side → Back 安全轉段

|項目|新規劃|
|---|---|
|Zaber|Side 結束 → Y safe/retract → readback → RX90|
|Safety place|必須|
|Keyence／Liquid|不使用|
|AF mode|不適用|

預設沿用目前經驗值 `Y=160 mm`，但它應被定義為 calibrated safe envelope，而不是只看單一 Y 數字。

### P3-05：Back surface acquisition

|項目|新規劃|
|---|---|
|Zaber|`RX=90/RZ=0`；從 Pin1 一端沿 X 每 10 mm 到 Pin2|
|Safety place|進入 view 前有；station 間沒有|
|Keyence|每 10 mm 使用|
|Liquid lens|不使用|
|AF mode|不適用|

起點：

- X 使用 Pin1。
- Z 使用 Side 平均 HoS 的保守值。
- Y 從 Pin1 加文件的 nominal offset（約 70 mm）開始，做 bounded Y search。
- 找到表面後移到 Keyence 中央範圍。

無效 station：

- 不用無效值改變後續 Y。
- 暫時使用前一有效趨勢繼續掃。
- 有前後有效點後再 interpolation。
- 若缺口太長，不允許自動拍攝該區。

### P3-06：Back source images、stitch、SAM

|項目|新規劃|
|---|---|
|Zaber|X 依 stitch positions；Y 由 Back surface curve 插值|
|Safety place|同 view 不退；結束後退|
|Keyence AF|關閉|
|Liquid lens AF|可切換；建議 anchor-only|
|AF mode|mode 3，Keyence off，liquid optional|

完成後立刻 background stitch／SAM。

### P3-07：Back → Front 安全翻面

文件明確要求「RZ 旋轉時 RX 必須為 0」。  
strap_collection_pseudocode_ajp_082120261854.docx (page 3)

固定序列應是：

1. Back safe retract。
2. `RX90 → RX0`。
3. 移到 wall/rotation clearance。
4. readback 驗證 RX=0。
5. `RZ0 → RZ180`。
6. readback。
7. `RX0 → RX90`，進入 Front view。

不能把這段縮成一次沒有中間 readback 的 compound rotation。

### P3-08：Front pins 轉換與 surface acquisition

先從 Side Pin1/Pin2 計算 `Pin1_180`、`Pin2_180`。

建議不要只使用 Word 的「半徑相同」假設，而使用：

- calibrated rotation center
- 2D homogeneous rotation
- calibrated translation/backlash correction
- RZ actual readback

到達 Front 後，再由 Keyence 實測修正 Pin1/Pin2。若預測與實測差異超過例如 2 mm，停止並要求人工確認，不能把錯誤 transform 傳給所有 source/component。

之後從 Pin1 朝 Pin2 的相反 X 方向每 10 mm 掃描，規則與 Back 相同。

|項目|新規劃|
|---|---|
|Keyence|每 station|
|Liquid lens|不使用|
|AF mode|不適用|
|Safe place|進入前已完成；station 間不退|

### P3-09：Front source images、stitch、SAM

和 Back 相同：

- Y 由 Front curve 插值。
- Zaber 先把機械焦距置於 liquid lens nominal 0。
- Keyence AF off。
- liquid AF anchor-only 或全部，使用 mode 3。
- 完成後 background stitch／SAM。

### P3-10：Top links、screws、SAM components

在三個 stitch 都完成後，等待相關 SAM 結果。文件希望對每個 screw／feature 重新送到 Keyence 下量測，再移到相機位置拍攝。  
strap_collection_pseudocode_ajp_082120261854.docx (page 3)

建議每個「實體位置」只做一次 Keyence，接著連續完成該位置的 macro/micro captures：

|Capture|Keyence|Liquid|AF mode|
|---|---|---|---|
|Front/Back 第一張 macro|是|是|13|
|Side 第一張 macro|是|是|14|
|同一實體位置的額外 capture|重用剛才結果|是|3|
|非 anchor、可信幾何區|可重用鄰近量測|視需要|8|
|micro screw|重用同點 mechanical focus；liquid strict|是|3|

重要改動：目前 Plan 3 的 dynamic component 原則上重用 overview Keyence cache，並不對每個 component 做 Keyence。新版為貼近 Word，預設應改成 `every_physical_point`；若要達成 20 分鐘，再另設 `anchor_interpolate` speed profile，而不是默默跳過。

### P3-11：Endlink 1 動態角度 reference

1. 從 endlink overview 分析 `Ang1`。
2. 失敗則 fallback `Ang1=0`，但 metadata 記錄 fallback。
3. safe retract。
4. RX 回 0。
5. 計算旋轉後 Pin1。
6. `RZ = 90° - Ang1`。
7. conservative move 到 Keyence，量測並修正 Pin1 position。
8. 移到 camera pose。
9. liquid lens AF。
10. capture。
11. safe retract、RX0。

|項目|設定|
|---|---|
|Keyence|是，使用 special-safe bounded X/Z search|
|Liquid|是，camera position 執行|
|AF mode|15|
|Safety place|RZ 前、capture 後都必須|

### P3-12：Endlink 2 動態角度 reference

1. 取得 `Ang2`，失敗 fallback 0。
2. RX0、rotation-safe。
3. `RZ = 270° - Ang2`。
4. 若該 view 需要 RX90，只能在 RZ 到位後執行。
5. 重複 Pin2 Keyence → camera → liquid AF → capture。
6. safe retract。
7. RX0。
8. RZ0。

文件的 Endlink 步驟在第四頁，但其中「rotate strap」和後面的 RZ 指令有些重複；實作時應正規化成上述單一、可稽核的旋轉序列。  
strap_collection_pseudocode_ajp_082120261854.docx (page 4)

### P3-13：結束與 213 → 217 邊界

建議不要停在 Word 的「RZ0 即完成」，而是：

1. RX0、RZ0。
2. safe retract。
3. 回到目前 213 使用的 holder/start pose。
4. 保存 geometry、focus、transform、AF audit。
5. 顯示 `page_template_new_strap_pre2`。

`pushButton_TemplateRun_217` 暫時保留現有 9-clock 工作：

- 4022–4028 overview sources
- 9-clock stitch/SAM
- 9-clock link/screw/endlink component captures
- 最後回 watch center

因為 Word 沒有清楚定義 9-clock overview；直接刪除 217 會是不相容變更。

---

## 三、目前 Plan 3 與 Word／新版的主要差異

|項目|目前 Plan 3|Word／建議新版|
|---|---|---|
|Side 起始 Z|直接到固定 Z，不做 approach|由保守 Z 做 bounded approach|
|Side Y recovery|最大約 ±2 mm|Word 到 ±4 mm|
|Pin2|最後有效 Keyence station|加 endlink camera image／vision refine|
|旋轉模型|未建立完整 Pin1/Pin2 transform|建立 Pin1/Pin2、HoS、RZ transform|
|Back/Front Keyence|10/30/50/70/90%，共 5 anchors|每 10 mm surface acquisition|
|Source mechanical AF|依 cache interpolation|相同精神，但先明確移到 liquid 0-current plane|
|Source liquid AF|每張都做|Word 允許 toggle；建議 anchor-only|
|拍攝順序|Side → Front → Back|Side → Back → Front|
|Component Keyence|多數重用 overview cache|每個 physical feature 量一次|
|Endlink|固定 4029/4030 pose、mode 15|先分析 Ang1/Ang2，再算動態 pose|
|SAM|三 view stitch 後才做 component，SAM 背景執行|基本相容|
|217|獨立 9-clock 流程|Word 未描述；建議保留|
|最終位置|213 回 start、217 回 center|保留現行明確 park，不採用只有 RZ0|
|Safety/Digital Twin|現有 motion guard|必須保留，不依文件文字停用|

另有一個現行不一致需要一併解決：

- YAML 寫的是 `button_213_component_order: [front, side, back]`
- runtime 常數實際是 `back → front → side`

新版 stage graph 必須只有一個 authoritative order，不能讓設定檔和 runtime 各自定義。

---

## 四、預計要修改的位置

這裡只是修改規格，尚未動 code。

1. [config/strap_workflow.yaml (line 142)](D:/Provenance Laboratories projects/ImagingLibWatch/config/strap_workflow.yaml:142)
    
    把目前 5-anchor overview config 改為：
    
    - `pipeline_revision`
    - Pin1 bounded Z search
    - 每個 view 的 10 mm scan policy
    - invalid gap／holder rejection
    - rotation transform
    - source liquid AF toggle
    - component `every_physical_point`
    - 217 compatibility policy
2. [core/strap_workflow/plan3.py (line 17)](D:/Provenance Laboratories projects/ImagingLibWatch/core/strap_workflow/plan3.py:17)
    
    現有 stage plan 太簡化，應改成上述 P3-00～P3-13，並明確表示：
    
    - geometry data dependency
    - safe barriers
    - stitch/SAM background jobs
    - component wait barriers
    - button 213／217 boundary
    - failure/fallback path
3. [App/main.py (line 42280)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:42280)
    
    現有 Wide-FoV、anchor batch、source capture 與 component runner 要重新組成 geometry-first runner。重點不是再加更多 callback，而是讓每個 stage consume／produce 明確資料。
    
4. [Controller/safety_rules.py (line 1528)](D:/Provenance Laboratories projects/ImagingLibWatch/Controller/safety_rules.py:1528) 和 [unified_driver.py (line 4465)](D:/Provenance Laboratories projects/ImagingLibWatch/Controller/hardware_drivers/unified_driver.py:4465)
    
    現有 direct Keyence transition 還帶有 `plan2` 命名。應泛化為安全的 strap geometry scan transition，但仍由 driver 和 safety rule 驗證，不能直接從 UI 呼叫裸 Zaber motion。
    
5. [config/strap_component_autofocus.yaml (line 15)](D:/Provenance Laboratories projects/ImagingLibWatch/config/strap_component_autofocus.yaml:15) 與 [config/af_mode_config.yaml (line 387)](D:/Provenance Laboratories projects/ImagingLibWatch/config/af_mode_config.yaml:387)
    
    現有 mode 3、13、14、15 基本足夠，不一定需要新 AF mode；主要是改每階段的：
    
    - `use_autofocus`
    - `use_liqud_af`
    - 是否重用同一 physical point 的 mechanical focus
    - liquid anchor policy
6. [db/templates/watchentry_structure.py (line 24)](D:/Provenance Laboratories projects/ImagingLibWatch/db/templates/watchentry_structure.py:24)
    
    現有模型已經把 `point_metadata` 和 `capture_metadata` 分開，可以直接延伸，不必回到「一點一張圖」。
    

---

## 五、應新增保存的幾何與影像 metadata

建議每次 213 產生一份 versioned `strap_geometry`：

- Pin1、Pin2 nominal／measured／vision-refined pose
- Side、Back、Front 每 10 mm station
- measured/interpolated/invalid 狀態
- Keyence raw samples、median、valid window
- HoS
- RZ transform matrix、rotation center、residual
- Ang1、Ang2、fallback reason
- calibration/version identifiers

Point-level metadata 保存：

- component semantic identity
- geometry station/reference
- derived target pose
- transform version
- confidence

Capture-level metadata保存：

- actual pose/readback
- Keyence result
- liquid current before/after
- `af_mode_id`
- `use_autofocus`
- `use_liqud_af`
- source/stitch/SAM IDs
- raw/HDR image assets

這樣 DB、S3、DynamoDB、report 才能知道一張圖是如何由哪一份 strap geometry 推導出來。

---

## 六、建議導入順序與驗收門檻

1. 先只建立 geometry model，不拍 component。
2. 使用 simulation／舊 log 驗證 Pin1、Pin2、station interpolation。
3. 執行 hardware dry-run，只移動和量 Keyence，不拍攝。
4. 開放 Side → Back → Front sources。
5. 最後開放 component 與動態 Endlink。
6. 計時並調整 liquid AF 為 all／anchor-only。

最低驗收條件：

- 所有 RZ move 發生時 RX 都是 0。
- 所有跨 view transition 都經過 safe pose 和 readback。
- Front transform 實測 residual 不超過設定值。
- 不能用 invalid Keyence station 引導下一站。
- geometry、source、component metadata 能寫入現有 1:N image model。
- 213 與 217 都能正常結束及 park。
- 舊 template 仍可 fallback 到目前 fixed-position Plan 3。
- 完整流程實測接近 Word 的 `<20 min` 目標；若超時，優先減少 liquid AF 次數，不應刪除安全轉段。




#### 實作active_plan: 4 
```
當在App main.py page_template_new_strap_pre執行了pushButton_TemplateRun_213, page_template_new_strap_pre2執行了pushButton_TemplateRun_217, 目前預設是active_plan: 3

現在要依照你的建議create active_plan: 4以active_plan, 並詳細解釋指出是基於上傳文件哪段, 或有何不同
```
已完成 `active_plan: 4`，並將它設為預設 Plan；Plan 0–3 都仍保留，可隨時切回。

主要入口：

- [strap_workflow.yaml (line 13)](D:/Provenance Laboratories projects/ImagingLibWatch/config/strap_workflow.yaml:13)：預設 `active_plan: 4`
- [Plan 4 設定 (line 223)](D:/Provenance Laboratories projects/ImagingLibWatch/config/strap_workflow.yaml:223)
- [Plan 4 階段模型 (line 41)](D:/Provenance Laboratories projects/ImagingLibWatch/core/strap_workflow/plan4.py:41)
- [Button 213 執行入口 (line 43618)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:43618)
- [Button 217 執行入口 (line 43690)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:43690)

## Button 213：Plan 4 流程

Plan 4 共定義 25 個可稽核 stage，以下依實際工作階段合併說明。

|階段|工作內容|Zaber／Safety place|Keyence AF|Liquid lens AF／af_mode|
|---|---|---|---|---|
|213-00|Preflight：檢查設定、相機、Keyence、安全規則|不移動|否|否|
|213-10|Side view 安全進場|先到 `Y >= 160`；Z 保持原位；折回 `RX=0、RZ=0`，並檢查 readback|否|否|
|213-11|尋找 Pin1|從 Z=70.213 開始，每次 +0.5 mm，最大 8 mm，硬限制 70–78.5；若沒訊號只依 Word 嘗試 Y `+0.5、-0.5`|直接讀 OUT1；有效範圍含 ±8 mm|不使用；非 capture，因此 af_mode 不適用|
|213-12～13|Side 每 10 mm 掃描至 Pin2|X 每次 +10 mm；失敗時依序試 Y `+0.5,-0.5,+1,-1,+2,-2,+4,-4`；無效點不影響下一站 Y|每站量測|不使用；Pin2 暫採最後有效 Keyence station|
|Side sources|拍 Side stitch source images|使用 Side surface 的 X→Z 插值；source 間不重新做機械 AF|不重測，重用前段 surface|每張都從 current 0 開始做 liquid AF；現有 overview capture 為 mode 3|
|Side stitch/SAM|Side 合成並背景啟動 SAM|不移動|否|否|
|Back transition|轉到 Back|先退 Y160、保持 Z、RX0/RZ0，再由安全 planner 部署到 Back `RX90/RZ0`|否|否|
|Back surface|Back 全長每 10 mm 建 surface|名義 station 間為 X-only；Keyence 修正 focus Y；失敗點只用前後有效資料插值，最大容許 gap 30 mm|每站量測|surface 階段不使用|
|Back sources/stitch/SAM|拍來源、合成、背景 SAM|source 使用 X→Y surface 插值|source 不重測 Keyence|每張 liquid AF、current 0、目前 mode 3|
|Front transition|轉到 Front|強制 `safe retract → RX0 → wall clearance → RZ180 → RX90`；禁止 RX 未折回時直接改 RZ|否|否|
|Front surface|反方向每 10 mm 建 surface|與 Back 相反方向掃描；X→Y 插值|每站量測|surface 階段不使用|
|Front sources/stitch/SAM|拍來源、合成、背景 SAM|同上|source 不重測 Keyence|每張 liquid AF、current 0、目前 mode 3|
|213-60|SAM barrier|不移動；只等待所選 component 需要的分析|否|否|
|Component phases|順序為 Front→Side→Back，讓完成 Front stitch 後先拍當前方向|同 view 局部移動走安全 planner；切換方向仍需 Y160、RX0，再改 RZ|**每一個實體 macro/micro 點都重新 Keyence AF**|Macro Front/Back mode 13；Side mode 14；Micro mode 3；每點 liquid AF|
|213-80～81|Endlink angle及固定 reference|若 UI 選了 Endlink reference，4029/4030 使用特殊安全路徑，通常先到 Y190、Z 保持，再旋轉|是|4029/4030 固定 mode 15，Keyence＋strict liquid AF|
|213-90|結束停車|經 shared safety planner 回到 holder/start position；所有軸 readback 驗證|否|否|

Pin1、Side 10 mm 掃描及 Y recovery 主要來自 Word「Wide FoV Position Acquisition 1–2」。strap_collection_pseudocode_ajp_082120261854.docx (page 1)

Side source interpolation、Back surface、Back source images 與 background stitch/SAM 來自「Wide FoV Position Acquisition 3–4」及「Back Link Pictures 5–10」。strap_collection_pseudocode_ajp_082120261854.docx (page 2)

Front 的 RX0/RZ180 順序、反向 surface scan，以及每個 Top screw/link 放到雷射下量測，來自「Front Link Pictures 11–21」與「Top Screws/Links 22–25」。strap_collection_pseudocode_ajp_082120261854.docx (page 3)

4029/4030、Ang1/Ang2 fallback、RX0 後才改 RZ，以及 liquid-only finishing movement，對應「End link reference 26a–q」。strap_collection_pseudocode_ajp_082120261854.docx (page 4)

## Button 217

Word 沒有定義目前 App 的 9-clock 翻面流程，因此 Plan 4 保留它作為明確的相容性 extension：

1. 安全進場：Y160、Z 保持、RX0/RZ0。
2. 9-clock overview sources。
3. Stitch。
4. Background SAM。
5. 所選 components。
6. 回到 verified center position。

Overview source 保留既有逐張 Keyence＋liquid AF；Plan 4 component 則套用 Side 規則：macro mode 14、micro mode 3。這一整段屬於「與 Word 不同、為維持 Button 217 相容性而保留」。

## 與 Word 不同之處

- Word 在第一個無效 X station 後即可判定超過 Pin2；Plan 4 要求連續兩個無效 station，降低單次反光或讀值遺失造成的誤判。
- 尚無可用的 Pin2 image detector，因此目前明確記錄 `last_valid_keyence_station` fallback，不宣稱已做 vision refinement。
- Word 的 Pin1/Pin2 旋轉中心換算已建立數學函式及 metadata，但未實際控制 Zaber；目前沒有完成 commissioning 的 strap-specific stage center。Front 改採獨立實測 surface。
- Endlink Ang1/Ang2 vision 尚未實作，依 Word 允許的 fallback 使用 0°，並在 geometry report 標示 `vision_status: not_implemented`。
- Word 把 liquid AF 描述為 optional；Plan 4 overview sources 選擇每張執行，且從 current 0 開始。
- Word 提到可考慮跳過 component Keyence；Plan 4 採較嚴格策略，每一個實體 component 都做 Keyence＋liquid AF。
- Fixed 4029/4030 仍尊重現有 UI 的「Bracelet-Endlink reference」選擇，不強迫拍攝。
- 保留既有 SafetyRule、Digital Twin、axis limits、逐段 readback；安全退避目前是移動 Y、保留 Z，沒有照 Word 推測新的 safety Z。
- Button 217 是現有產品相容 extension，不是 Word pipeline 的內容。

## 資料紀錄

每次執行後會新增獨立的 `strap_plan4_geometry.yaml`，內容包括：

- 25 個編號 stage 與 `document_basis`
- Pin1／Pin2
- Side、Back、Front surface stations
- Keyence 成功／失敗 attempts
- RZ180 transform 的 commissioning 狀態
- Endlink angle fallback
- 與 Word 的差異

它與 image-level capture metadata 分開，沒有改動 DB、S3、DynamoDB 或 report 的既有多圖片資料模型。

## 驗證結果

- Python 編譯檢查通過。
- 最終受影響回歸測試：`181/181 OK`。
- 完整 Strap discovery 載入 294 個案例時，293 個通過；唯一未執行的是環境缺少 `pytest`，導致 `test_strap_registration.py` 匯入失敗。
- 尚未執行實體 Zaber／Keyence movement。正式使用前應先跑 simulation，再以 supervised hardware run 驗證 Pin1 bounded-Z、Y160/RX0/RZ transition、Back/Front 10 mm scan 及 4029/4030 mode 15。