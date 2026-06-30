
|                                   |     |
| --------------------------------- | --- |
| [[#### Template creation in App]] |     |
|                                   |     |
|                                   |     |


#### Template creation in App
```
我重新執行App main.py測試校正目前位置大致上準確了(macro_cam_1, micro_cam). 幫我確認一下當在最後滑鼠點擊要拍照macro_cam_1, micro_cam時, 除了x,y是來自於在UI上面點擊的位置, 其他應該是來自internalnum_config.yaml裡面每個internalnum1的設定? 另外在左邊有列出在internalnum_config.yaml裡面Front的所有internalnum1, 不過請不要顯示displayname顯示backup的因為這些是空的為外來方便增加做準備所以請刪除. 而在page_template_new_front的左上角frame_100會顯示縮圖. 所以這裡縮圖的連結要在哪裡設定, 圖要放在哪裡呢? 另外在front.top的frame_50顯示的toppoint image設定是internalnum_config.yaml裡面的internalnum1=0001所以拍攝時應該會用這個internalnum1=0001的設定譬如包括postcolor: 1, 還在滑鼠按一下出現的放大圖應該也用同樣的設定譬如postcolor: 1. 而在在最後frame_50滑鼠點擊要拍照macro_cam_1, micro_cam時預設應該顯示internalnum_config.yaml裡面internalnum2=0001所以應該是第一個standard capture的image. 而拍照完當按下pushButton_TemplateRun_174就代表要顯示下一個capture也就是internalnum2=0002(如果有存在的話), 所以拍照用internalnum2=0002裡面的拍照設定, 而按下pushButton_TemplateRun_195 "Confirm按鈕"就是把這個internalnum2=0002的也加入到template, 而這時按下pushButton_TemplateRun_159也是針對internalnum2=0002這個capture進行修正. 所以之前我們做的修改是針對page_template_new_front是處理"front view"的template creation. 所以我們要把我們對page_template_new_front做的修正也擴充到: page_template_new_back page是針對"Back view", 所以tableView_16顯示的是internalnum_config.yaml裡面"Back view"的internalnum1的items, 也不要顯示displayname顯示backup的. 在frame_79也是一開始顯示Back.toppoint的image(拍照按照internalnum_config.yaml裡面internalnum1=1001的設定), 而之後滑鼠按一下的放大圖跟最後macro_cam_1, micro_cam image拍攝的設定也來自於internalnum_config.yaml, 而macro_cam_2切換到macro_cam_1, micro_cam校正應該跟Front view一樣的數值. 而這時按下pushButton_TemplateRun_175就代表要顯示下一個capture. 而按下pushButton_TemplateRun_196就是"Confirm按鈕", 按下pushButton_TemplateRun_162也是針對這個view的Control panel. page_template_new_openback page是針對"openback view", 所以tableView_17顯示的是internalnum_config.yaml裡面這個view的internalnum1的items, 也不要顯示displayname顯示backup的. 在frame_80也是一開始顯示openback.toppoint的image(拍照按照internalnum_config.yaml裡面internalnum1=2001的設定), 而之後滑鼠按一下的放大圖跟最後macro_cam_1, micro_cam image拍攝的設定也來自於internalnum_config.yaml, 而macro_cam_2切換到macro_cam_1, micro_cam校正應該跟Front view一樣的數值. 而這時按下pushButton_TemplateRun_177就代表要顯示下一個capture. 而按下pushButton_TemplateRun_197就是"Confirm按鈕", 按下pushButton_TemplateRun_165也是針對這個view的Control panel. page_template_new_openbackcrown page是針對"openbackcrown view", 所以tableView_18顯示的是internalnum_config.yaml裡面這個view的internalnum1的items, 也不要顯示displayname顯示backup的. 在frame_81也是一開始顯示openbackcrown.toppoint的image(拍照按照internalnum_config.yaml裡面internalnum1=3001的設定), 而之後滑鼠按一下的放大圖跟最後macro_cam_1, micro_cam image拍攝的設定也來自於internalnum_config.yaml, 而macro_cam_2切換到macro_cam_1, micro_cam校正應該跟Front view一樣的數值. 而這時按下pushButton_TemplateRun_179就代表要顯示下一個capture. 而按下pushButton_TemplateRun_198就是"Confirm按鈕", 按下pushButton_TemplateRun_168也是針對這個view的Control panel.
```

已處理。重點結論如下：

最後第二次滑鼠點擊要拍 `macro_cam_1` / `micro_cam` 時，現在邏輯是：

- `X/Y`：來自 UI 點擊位置，先轉回未裁切 toppoint raw pixel，再用前面校正好的 affine 算出 Zaber 座標。
- 其他設定：來自目前選中的 point/capture，也就是由 `internalnum_config.yaml` 的 `internalnum1 + internalnum2` materialize 出來的設定，例如 `exptimes`、燈光、gain、`use_autofocus`、`use_liqud_af`、`has_glass`、`postcolor` 等。
- target camera 會依點位強制修正：`macropoint*` -> `macro_cam_1`，`micropoint*` -> `micro_cam`。

我做的修改：

- [App/main.py (line 3190)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:3190)：scratch template 初始化時，Front/Back/OpenBack/OpenBackCrown/Strap/Box 的 `toppoint1` 都會套用自己的 `internalnum_config.yaml` 設定，不再只處理 Front。
- [App/main.py (line 6122)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:6122)：new-template 左側 table 現在會跳過 `displayname: backup` 的項目。
- [App/main.py (line 6230)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:6230)：`To Next Capture` 現在會切到下一個 capture，例如 `internalnum2=0002`，然後走 `_execute_prepared_point_capture()` / `execute_template_point()`，不再走舊的直接 `capture_image()`。
- [App/main.py (line 6590)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:6590)：backup 過濾做成共用判斷，所以 Back/OpenBack/OpenBackCrown 也會一致套用。
- [config/internalnum_config.yaml (line 1454)](D:/Provenance Laboratories projects/ImagingLibWatch/config/internalnum_config.yaml:1454)：OpenBack `2001` toppoint 改成 `macro_cam_2`。
- [config/internalnum_config.yaml (line 1664)](D:/Provenance Laboratories projects/ImagingLibWatch/config/internalnum_config.yaml:1664)：OpenBackCrown `3001` toppoint 改成 `macro_cam_2`。

關於 `frame_100` 左上角縮圖：目前不是在 `internalnum_config.yaml` 設定。它找的是：

`App/images/images/<View>.<point_key>.png`

例如 Front lower text/macropoint2 是：

`App/images/images/Front.macropoint2.png`

如果那裡沒有，再 fallback 到：

`images/images/<View>.<point_key>.png`

`internalnum_config.yaml` 裡的 `simulation_file` 是給 simulation/sample asset pipeline 用，不是目前 `frame_100` 的 instruction thumbnail 來源。

驗證已跑：

- `conda run python -m py_compile App\main.py DB\templates\internalnum_config.py`
- YAML parse `hardware_config.yaml` / `internalnum_config.yaml`
- `git diff --check`
- 非 GUI 檢查確認：
    - `0001 Front.toppoint1 -> macro_cam_2, postcolor=1`
    - `1001 Back.toppoint1 -> macro_cam_2`
    - `2001 OpenBack.toppoint1 -> macro_cam_2`
    - `3001 OpenBackCrown.toppoint1 -> macro_cam_2`
    - backup display names 可被識別並隱藏

還沒做實機 GUI/hardware validation。建議你重開 App 後依序測 Front、Back、OpenBack、OpenBackCrown：選 point、點兩次拍 `internalnum2=0001`，再按對應的 `To Next Capture` 看 log 是否切到 `internalnum2=0002` 並仍走 `execute_template_point()`。