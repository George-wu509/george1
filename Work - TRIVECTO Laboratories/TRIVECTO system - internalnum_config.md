

|                                                      |     |
| ---------------------------------------------------- | --- |
| [[#### internalnum_config設定1 - focus and hdr image]] |     |
| [[#### internalnum_config設定2 - focus and hdr image]] |     |
| [[#### internalnum_config設定3 - focus and hdr image]] |     |
|                                                      |     |

#### internalnum_config設定1 - focus and hdr image
```
在internalnum_config裡面每個watchpoint(or internalnum1)的point_defaults有use_liqud_af, 而在之下譬如'0001','0002'可能是standard capture or hdr capture也有use_liqud_af. 那真的拍照standard capture or hdr capture時是看自己的use_liqud_af設定or point_defaults的use_liqud_af 在internalnum_config裡面每個watchpoint的point_defaults有af_mode_id, 所以在這裡選擇af_mode_id譬如0,1..就會用在standard capture or hdr capture. 所以當hdr capture時會有exposure time的bracket有不同的exposure time, 所以拍攝hdr image的focus(Keyence af and liquid lens af)到底怎麼決定, 還是直接用standard capture的autofocus的z跟current? 在internalnum_config裡面每個watchpoint如果下面的某個capture是capture_type: hdr代表是拍hdr image. 這裡設定有postcolor, num_images, hdr_expo_mode, hdr_process_mode. 在hdr_expo_mode裡面hdr_expo_mode=0是代表auto決定exposure bracket, hdr_expo_mode=2,3是代表之前已經設定好的exposure bracket. 所以當hdr_expo_mode=0,2,3...那是不是exptimes參數就不會用到了. 而在hdr_process_mode也有包括hdr image專用的postcolor設定, 那這會不會跟postcolor參數重複或沒必要? 在internalnum_config裡面每個watchpoint下面的某個capture譬如'0001','0002'都有各自的has_glass, 但應該這是一致的可以放到point_defaults. 在internalnum_config裡面每個watchpoint的point_defaults也應該增加xmod, ymod, zmod.
```

大致上：**用以前的 template 拍照應該不會出問題**，因為這次改法刻意把邊界留住了。

舊 template 已經存好的 `standard_captures` / `hdr_captures` 裡面，本來就有 exposure、light、AF、HDR mode 等值；runtime 真實拍照時仍然是讀 template 內的 capture 設定，不會重新拿 `config/internalnum_config.yaml` 去覆蓋舊 template。`normalize_template_payload()` 也沒有被改成自動套 default，所以舊 template load/save 不會被偷偷改拍照參數。

會需要注意的情況主要有三種：

1. **舊 template 某些點沒有 capture list**  
    這種以前可能本來就不完整。現在 App 在 create/edit 流程遇到沒有 capture 時，會嘗試用 `internalnum1` 從 config 補一個 default capture；如果找不到對應 default，仍會 fallback 成原本那種基本 `CaptureCondition`。
    
2. **舊 template 裡 Front.sidepoint1 曾經是 `internalnum1=0003`**  
    我這次把新 canonical 修成 `0002`。舊 template 如果已經保存 `0003`，實際拍照的曝光/燈光仍用舊 template capture，不會因此壞掉；但若某些報表/顯示/metadata 完全依賴 internalnum lookup，可能會把 `0003` 對到新的 config 說明。這比較像 metadata 對應風險，不是硬體拍照風險。
    
3. **HDR 細節**  
    舊 template 如果存了 `hdr_exposure_policy` / `hdr_processing_policy` 等欄位，runtime 仍會照舊處理；新 internalnum default 只負責 `hdr_expo_mode` / `hdr_process_mode`，細節由 `hdr_config.yaml` 展開。這對舊 template 是相容的。
    

所以結論是：**老 template 的拍照行為應該維持原樣；新 defaults 主要影響「建立/編輯 template 時補齊預設值」**。我建議手動拿一個舊 template 做一次 simulation/hybrid dry run，特別看 `Front.sidepoint1` 和有 HDR 的 micropoint metadata 是否符合預期。



#### internalnum_config設定2 - focus and hdr image
```
在internalnum_config裡面每個watchpoint(or internalnum1)的point_defaults有use_liqud_af, 而在之下譬如'0001','0002'可能是standard capture or hdr capture也有use_liqud_af. 那真的拍照standard capture or hdr capture時是看自己的use_liqud_af設定or point_defaults的use_liqud_af 在internalnum_config裡面每個watchpoint的point_defaults有af_mode_id, 所以在這裡選擇af_mode_id譬如0,1..就會用在standard capture or hdr capture. 所以當hdr capture時會有exposure time的bracket有不同的exposure time, 所以拍攝hdr image的focus(Keyence af and liquid lens af)到底怎麼決定, 還是直接用standard capture的autofocus的z跟current? 在internalnum_config裡面每個watchpoint如果下面的某個capture是capture_type: hdr代表是拍hdr image. 這裡設定有postcolor, num_images, hdr_expo_mode, hdr_process_mode. 在hdr_expo_mode裡面hdr_expo_mode=0是代表auto決定exposure bracket, hdr_expo_mode=2,3是代表之前已經設定好的exposure bracket. 所以當hdr_expo_mode=0,2,3...那是不是exptimes參數就不會用到了. 而在hdr_process_mode也有包括hdr image專用的postcolor設定, 那這會不會跟postcolor參數重複或沒必要? 在internalnum_config裡面每個watchpoint下面的某個capture譬如'0001','0002'都有各自的has_glass, 但應該這是一致的可以放到point_defaults. 在internalnum_config裡面每個watchpoint的point_defaults也應該增加xmod, ymod, zmod.
```
我剛剛已經順手把 schema 往你說的方向收斂了。

現在規則是：

`use_liqud_af`  
實際拍照時，如果 capture 裡有自己的 `use_liqud_af`，runtime 會用 capture 的值覆蓋 point 的值。這是原本 App 合成 `hw_cfg` 的行為。  
但我已把 config default 整理成：`use_liqud_af` 放在 `point_defaults`，capture default 會在 materialize 時繼承它；除非未來某個 capture 真的要 override，才需要在 capture 裡放自己的 `use_liqud_af`。

`af_mode_id` / HDR focus  
`af_mode_id` 是 point-level default，會進入每個 capture 的 `hw_cfg`。Keyence AF 和 liquid-lens AF 都會用這個 mode 包住執行。  
HDR 時目前邏輯是：

- 如果 `config/hdr_config.yaml` 的 `hdr_use_std_af` 有指定，HDR capture 會先用對應 standard capture 的 `focus_hardware` 做 autofocus。
- 也就是 focus 用 standard capture 的 camera/lights/exposure/AF 設定，HDR bracket 仍用 HDR capture 自己的 exptimes/HDR mode 拍。
- 如果沒有 `hdr_use_std_af` mapping，就用 HDR capture 自己的 AF 設定。
- HDR 多張曝光時，autofocus 會用 bracket 裡的 middle exposure 做 AF，不是每個 exposure 都重新 focus。

`hdr_expo_mode` / `exptimes`  
不是所有 HDR mode 都完全不用 `exptimes`：

- `hdr_expo_mode=0`：auto，會用 preview + `hdr_config.yaml` 決定 bracket；template/config 裡的 `exptimes` 主要變成 requested/fallback，不是最終權威。
- manual mode：會直接用 template 裡的 `exptimes`。
- fixed mode，例如 `2`：用 `hdr_config.yaml` 對應 mode 裡設定好的 fixed bracket。
- 未來 `3...` 也是看 `hdr_config.yaml` 該 mode 的 policy。

`postcolor` vs `hdr_process_mode`  
這兩個有重疊風險。`postcolor` 是 capture/image-level color tone；`hdr_process_mode` 是 HDR fusion/tone/post pipeline 的 mode。  
所以對 HDR capture，我建議 `postcolor=0`，由 `hdr_process_mode` + `hdr_config.yaml` 負責 HDR 專用處理。我也驗證過目前 HDR defaults 沒有 nonzero `postcolor`。

我做的修改：

- [DB/templates/internalnum_config.py (line 147)](D:/Provenance Laboratories projects/ImagingLibWatch/DB/templates/internalnum_config.py:147)：新增 `xmod/ymod/zmod` alias，`has_glass` 可放 point defaults。
- [DB/templates/internalnum_config.py (line 477)](D:/Provenance Laboratories projects/ImagingLibWatch/DB/templates/internalnum_config.py:477)：capture defaults 會繼承 point-level `use_liqud_af/has_glass`。
- [config/internalnum_config.yaml (line 8)](D:/Provenance Laboratories projects/ImagingLibWatch/config/internalnum_config.yaml:8)：每個 `point_defaults` 補了 `Xmod/Ymod/Zmod`，並把共同的 `has_glass` 移到 point-level；capture-level 重複的 `use_liqud_af/has_glass` 已移除。

驗證已過：

- `py_compile`
- config parse：`174` entries、`112` capture defaults、沒有 `0000`
- watch factory：`97 standard + 9 HDR`
- box factory：`11 standard`
- 代表點確認 capture 仍能繼承 `use_liqud_af/has_glass/Xmod/Ymod/Zmod`
- HDR capture `postcolor` 全部是 `0` 或未設定
- `git diff --check` 只有 CRLF 提示，沒有 whitespace error.



#### internalnum_config設定3 - focus and hdr image
```
那請幫我評估以下的修改, 關於auto focus的部分我在point_defaults level我新增了些autofocus相關參數: af_unity, use_autofocus, use_liqud_af, af_mode_id, Xmod, ymod, zmod. 當af_unity=1代表在這個watchpoint下面captures都用同樣的autofocus由point_defaults level下的use_autofocus, use_liqud_af, af_mode_id, Xmod, ymod, zmod控制. 如果只跑standard capture沒有hdr capture, 或有跑standard capture跟hdr capture, 或沒有跑standard capture只有hdr capture都跑一次autofocus就好, 然後都用同樣的focus z跟current (這時每個capture下面的autofocus參數就沒有用). 當af_unity=0代表在這個watchpoint下面captures都用各自的autofocus參數, 所以每個capture都有各自的use_autofocus, use_liqud_af, af_mode_id, Xmod, ymod, zmod然後用這個參數控制. 而這時point_defaults level的autofocus參數就沒有用. 而在HDR capture裡面的postcolor直接刪除就不會有重疊風險. 而每個capture的visible_default是甚麼? 0跟1有差別嗎? 然後再重新檢查一下internum_config.yaml, 我有看到在entries下面有些watchpoint用'0001','0002'也有看到0008, 0009這樣格式不統一請修正變成類似這樣格式'0001', 而且每個point_defaults level的參數跟每個standard capture, 每個hdr capture的參數要統一(譬如internalnum1='0001', internalnum1='0002'...下面的每個point_defaults level的參數跟每個standard capture, 每個hdr capture的參數要一樣)
```
已完成這輪修改與驗證。

`visible_default` 的意思是：在同一個 watchpoint 多張 capture 裡，`1` 代表預設/代表圖片，會用在顯示、upload metadata、default capture 標記；`0` 就不是代表圖。如果全部都是 `0`，App runtime 會自動把第一張當 default。它不影響曝光、HDR、AF。

我改了這些：

- [DB/templates/watchview_structure.py (line 44)](D:/Provenance Laboratories projects/ImagingLibWatch/DB/templates/watchview_structure.py:44)：capture model 補 `Xmod/Ymod/Zmod/af_mode_id`；point model 補 `af_unity/use_autofocus`。
- [DB/templates/internalnum_config.py (line 142)](D:/Provenance Laboratories projects/ImagingLibWatch/DB/templates/internalnum_config.py:142)：loader 支援 `af_unity` aliases，並讀取 point-level `use_autofocus/af_unity`。
- [App/main.py (line 14258)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:14258)：拍照 payload 帶入 `af_unity`；當 `af_unity=1` 時，capture 自己的 AF 欄位會被 point-level 值覆蓋。
- [Controller/hardware_drivers/unified_driver.py (line 6098)](D:/Provenance Laboratories projects/ImagingLibWatch/Controller/hardware_drivers/unified_driver.py:6098)：`af_unity=1` 時，AF cache signature 不再被 `capture_id`、曝光、燈光打散，所以同 watchpoint 的 standard/HDR capture 可共用同一次 AF 的 z/current。
- [config/internalnum_config.yaml (line 7)](D:/Provenance Laboratories projects/ImagingLibWatch/config/internalnum_config.yaml:7)：統一四碼 key quote 格式、補齊 point_defaults schema、補齊 capture AF schema、HDR capture 移除 `postcolor`，並修正 `0003` 為 `Front.sidepoint2`。

關於舊 template：一般用舊 template 拍照不應該出問題。舊 template 沒有 `af_unity` 時，App/driver 預設是 `0`，所以會維持舊的 per-capture AF 行為；只有新建 template 從 `internalnum_config.yaml` copy 出 `af_unity: 1` 時，才會啟用共用 AF。

驗證已跑：

- `conda run python -m py_compile ...` 通過。
- YAML schema 檢查通過：174 entries、112 captures、9 HDR captures。
- HDR capture 已確認沒有 `postcolor`。
- `Front.sidepoint1 -> 0002`、`Front.sidepoint2 -> 0003` 正常。
- default template smoke test 通過：scratch template 可建立，Front.micropoint13 有 standard + HDR。

剩餘風險/手動驗證：建議用 production 或 hardware simulation 跑一個有 standard+HDR 的 watchpoint，確認 log 裡第二個 capture 的 `same_position_cache_hit=True`，且 focus report 的 `af_unity=True`。這是硬體行為層，compile/smoke test 無法完全替代。
