
|                                               |     |
| --------------------------------------------- | --- |
| [[#### 自動決定exposure series的方法, fuse相關或者後處理]]  |     |
| [[#### 實作: 自動決定exposure series, fuse相關或者後處理]] |     |
|                                               |     |
|                                               |     |
|                                               |     |
|                                               |     |

#### 自動決定exposure series的方法, fuse相關或者後處理
```
在目前的拍攝hdr image主要是結合數張不同exposure time拍攝的images fuse成hdr image以分析錶面各區域的texture並以algorithm/bump_wrapper的方法分析. 在目前的hdr config mode有三種特別的hdr mode以及default, 所以這幾種mode是跟hdr exposure相關 or fuse相關或者後處理相關呢? 我想要加入能自動決定exposure series的方法, 以及能自動決定 fuse相關或者後處理的方法讓user 不需要決定這些設定就能拍出適合texture analysis的hdr image, 除此之外也要保留可以自行決定的幾種不同mode. 所以先不要修改code先建議適合這樣的code structure, 以及要怎麼做到自動決定exposure series的方法, 以及能自動決定 fuse相關或者後處理的方法
```

有看過目前程式，先不改 code。結論是：目前 `hdr_mode_1/2/3/default` 主要是 **fusion + tone mapping + post-processing preset**，不是 exposure series preset。

在 [config/hdr_config.yaml (line 117)](D:/Provenance Laboratories projects/ImagingLibWatch/config/hdr_config.yaml:117) 裡，`hdr_modes` 會 deep-merge 覆蓋 default 的這些 section：

- `fusion`: algorithm、registration、Mertens weights
- `per_frame_preprocess`: CLAHE/blur
- `hdr_merge`: reliable pixel min/max
- `tone_mapping`: local tone mapping/gamma/white percentile
- `post`: white balance、LAB CLAHE、denoise、sharpen、saturation

真正決定拍哪幾張 exposure 的是 `CaptureConditionHDR.exptimes`，定義在 [DB/templates/watchview_structure.py (line 64)](D:/Provenance Laboratories projects/ImagingLibWatch/DB/templates/watchview_structure.py:64)。而 `hdr_fusion_mode` 的 `exposure / z_stack / mixed` 比較像「bracket 類型」，跟 `hdr_mode_1/2/3` 不是同一件事。

**建議結構**

我會把 HDR 拆成三個獨立層，不要把 auto exposure 塞進 `hdr_mode_1/2/3`：

1. **Capture / Exposure Plan**
    
    - 負責決定 `exptimes`
    - 新增 `hdr_exposure_policy: manual | auto_texture | auto_full_range`
    - manual 時保留目前 `exptimes`
    - auto 時 runtime 產生 `resolved_exptimes`
2. **Fusion / Processing Plan**
    
    - 負責決定 `hdr_mode` 或 resolved `hdr_cfg`
    - 新增 `hdr_processing_policy: manual | auto_texture`
    - manual 時保留 user 選的 `hdr_mode`
    - auto 時從 `default/hdr_mode_1/2/3/...` 候選中挑一個
3. **Result / Metadata**
    
    - bracket frame metadata：`bracket_index`, `exposure_time_us`, `z_position_mm`
    - fused image metadata：`selected_hdr_mode`, `selected_algorithm`, `auto_score`, `auto_report`
    - 不要把 runtime auto result 靜默寫回 point-level template，除非 user 明確按「save tuned settings」

可以新增幾個小模組：

```
algorithms/hdr_exposure_planner.py
algorithms/hdr_quality.py
algorithms/hdr_auto_selector.py
algorithms/hdr_pipeline.py  # 保留現有 fuse/postprocess
```

`hdr_pipeline.py` 現在的 `hdr_postprocess_pipeline(...)` 已經是好的共用入口：[algorithms/hdr_pipeline.py (line 761)](D:/Provenance Laboratories projects/ImagingLibWatch/algorithms/hdr_pipeline.py:761)。auto processing 應該包在它外面，多跑幾個 candidate config，選分數最高的 output。

**Auto Exposure Series 做法**

建議先做「plan-first」版本，簡單穩定：

1. 在最終燈光、camera、gain、focus pose 下拍 1 張 preview。
2. 對 ROI 算 luminance stats：`p1/p5/p50/p95/p99.9`、black clipped、white clipped。
3. 算短曝光：
    - 目標是高光不要爆
    - 例如讓 `p99.9 <= 245`，white clipped `< 0.5%`
4. 算長曝光：
    - 目標是暗部 texture 不要埋掉
    - 例如讓 `p5/p10 >= 12~20`
5. 在 `short_exp` 到 `long_exp` 間用 geometric series 產生 3~5 張：
    - range 小：3 張
    - range 大：5 張
    - 避免超過 camera max exposure 或造成太慢
6. 拍完 bracket 後驗證 coverage：
    - 每個 ROI pixel 至少有一張 frame 落在 reliable range，例如 `[5, 248]`
    - coverage 不夠就擴 bracket 或標 warning

簡化公式大概是：

```
short_exp = current_exp * 245 / max(p99_9, 1)
long_exp = current_exp * 20 / max(p5, 1)
exptimes = geometric_series(short_exp, long_exp, max_frames=5)
```

texture analysis 特別要用 ROI，不要只看全圖。因為 `bump_wrapper` 主要用 HSV V channel、entropy mask、KAZE keypoints，所以 auto exposure 的目標應該是：V channel 有足夠中間調、局部熵/梯度夠、但不要靠過度 sharpen 或 noise 產生假 keypoints。

**Auto Fuse / Postprocess 做法**

這部分不需要重新拍照。用同一組 bracket frames，對候選 mode 都跑一次 fusion：

```
candidates = [default, hdr_mode_1, hdr_mode_2, hdr_mode_3]

for mode in candidates:
    cfg = resolve_hdr_mode_config(raw_hdr_cfg, mode)
    fused = hdr_postprocess_pipeline(frames, exptimes, contcorrect, cfg)
    score = texture_quality_score(fused)
pick best score
```

texture score 不建議只看「看起來亮」或 KAZE 數量，會被 noise/sharpen 騙。建議混合：

- clipping penalty：黑/白 clipped percentage
- midtone score：p10~p90 是否落在可分析範圍
- entropy / local contrast：V channel 局部紋理
- KAZE keypoint stability：太少不好，暴增也可能是 noise
- sharpness：Laplacian variance，但加上 noise penalty
- saturation penalty：避免顏色/亮度處理太極端

**我會優先保留 manual mode**

UI 上建議概念改成：

- `Exposure Series`: Manual / Auto Texture
- `Processing Preset`: Manual selected mode / Auto Texture
- `HDR Config Preset`: default / hdr_mode_1 / hdr_mode_2 / hdr_mode_3
- `Bracket Type`: exposure / z_stack / mixed

這樣 user 可以完全自動，也可以只自動 exposure、手動選 fuse/post，或完全手動。

**要注意的風險**

目前真實硬體路徑在 [UnifiedHardwareDriver (line 3266)](D:/Provenance Laboratories projects/ImagingLibWatch/Controller/hardware_drivers/unified_driver.py:3266) 會在 driver 裡 capture+fuse 後回傳單張 fused image。若要完整支援「一個 watch point 多張 raw bracket + optional HDR」，最好之後讓 driver 回傳 structured HDR result，或至少暴露 `_last_hdr_bracket_frames/_last_hdr_auto_report` 給 `WorkflowManager` 註冊 bracket assets。這跟你現在的 refactor 方向是同一件事。

手動驗證建議先用 HDR bucket：對同一組 bracket frames 跑 auto selector，比較 `default/hdr_mode_1/2/3` 和 texture metrics，再上真機做 auto exposure。


核心判斷：

|Moonlight 建議|對目前系統的價值|我的建議|
|---|---|---|
|不要用單一 ROI 平均亮度迭代曝光|非常正確。陰影、黑白背景、小位移會讓 exposure drift|不採用舊式 average-pixel iterative exposure|
|拍一組從很暗到很亮的固定 bracket|很適合 texture/HDR，結果更穩定|作為 `auto_texture` 的預設策略|
|User 不應該選 HDR/exposure/mode|很符合目前 template/refactor 方向|template/point 決定是否拍 standard + HDR|
|保存所有 bracket images|對 debug/訓練/調參很重要|初期一定要保存 raw brackets + hdr_fused|
|HDR fusion 可背景處理|架構上合理|但需要先把 raw bracket asset 註冊清楚，再 async fuse|

目前系統其實已經有一部分很接近這個方向：

- [DB/templates/watchview_structure.py (line 113)](D:/Provenance Laboratories projects/ImagingLibWatch/DB/templates/watchview_structure.py:113) 已經有 `standard_captures` / `hdr_captures`，適合讓「某個 watch point 自動拍 standard + HDR」。
- [algorithms/hdr_pipeline.py (line 761)](D:/Provenance Laboratories projects/ImagingLibWatch/algorithms/hdr_pipeline.py:761) 已經有統一 `hdr_postprocess_pipeline(...)`，可以拿同一組 bracket frames 跑多個 `hdr_mode` candidate。
- [config/hdr_config.yaml (line 117)](D:/Provenance Laboratories projects/ImagingLibWatch/config/hdr_config.yaml:117) 的 `hdr_mode_1/2/3` 本質是 fusion/tone/postprocess preset，不是 exposure preset。

最大的 gap 是：真實硬體 HDR path 目前在 driver 裡 capture+fuse 後，多半只回傳 fused image。這和「保存所有五張 bracket images 到 DB」的建議不完全一致。V3 asset model 支援 `raw_bracket` / `hdr_fused`，但 driver 最好改成回傳 structured result，例如：

```
{
    "frames": [...],
    "exptimes": [...],
    "fused": fused_img,
    "auto_report": {...}
}
```

而不是只回傳單張 image。

我建議現在的設計方向是：

1. **Exposure Policy**
    
    - `manual`: 使用 user/template 的 `exptimes`
    - `fixed_texture_bracket`: 用固定大範圍 bracket，例如 4~5 張
    - `auto_texture_bracket`: 用一次 preview + robust percentiles 微調 bracket，但不要用平均亮度迭代
2. **Processing Policy**
    
    - `manual`: 使用 user 選的 `hdr_mode`
    - `auto_texture`: 對 `default/hdr_mode_1/2/3` 都 fuse，根據 texture score 選最佳
3. **User Experience**
    
    - 普通 user 不選 HDR mode、不選 exposure
    - template/routine 決定哪些 point 需要 standard、哪些需要 HDR
    - advanced UI 保留 manual exposure/mode，用於 debug 和調參
4. **Texture Auto Score**  
    因為 `texture_algo`/`bump_wrapper` 主要看 HSV V channel、entropy mask、KAZE、Delaunay，所以自動選擇 HDR 時不要只看「圖片亮不亮」。應該看：
    
    - clipped black / white ratio
    - V channel 中間調覆蓋
    - local entropy / gradient
    - KAZE keypoint count 是否穩定
    - 過度 sharpen/noise penalty

我會採納 Moonlight 的主要精神：**穩定的大範圍 bracket 比每次追亮度更可靠**。但我會把它升級成目前系統可維護的三層架構：`exposure policy`、`fusion/post policy`、`asset/result metadata` 分開。這樣既能讓 user 不用決定，又能保留 manual mode，並且符合現在「一個 watch point -> multiple images plus optional HDR image」的 refactor 目標。



#### 實作: 自動決定exposure series, fuse相關或者後處理
```
請按照建議修改code, 而且解釋user要如何在App main.py及test_template_ui.py 在create template時可以create要拍hdr image要做的設定, 或者可以先預設定某些watchpoint就是要拍hdr image, 以及如何最自動的拍hdr image
```

已完成 HDR auto 設計的 code wiring。這次把 HDR 設定拆成兩層：

- `hdr_exposure_policy`: 決定 exposure series，預設可用 `auto_texture_bracket`
- `hdr_processing_policy`: 決定 fuse / postprocess，預設可用 `auto_texture`
- 原本 `hdr_mode` 保留，當作 manual 或 auto candidate 的 HDR config mode

主要改動在 [algorithms/hdr_auto.py](D:/Provenance Laboratories projects/ImagingLibWatch/algorithms/hdr_auto.py)、[config/hdr_config.yaml](D:/Provenance Laboratories projects/ImagingLibWatch/config/hdr_config.yaml)、[core/workflow_manager.py](D:/Provenance Laboratories projects/ImagingLibWatch/core/workflow_manager.py)、[Controller/hardware_drivers/unified_driver.py](D:/Provenance Laboratories projects/ImagingLibWatch/Controller/hardware_drivers/unified_driver.py)、[Controller/test_hardware/test_template_ui.py](D:/Provenance Laboratories projects/ImagingLibWatch/Controller/test_hardware/test_template_ui.py)、[App/main.py](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py)。

**怎麼用**  
在 `test_template_ui.py` create template 時：

1. 選 `Capture Mode = HDR Capture`
2. `HDR Exposure` 選 `Auto texture bracket`
3. `HDR Processing` 選 `Auto texture mode`
4. 儲存 capture/template

這樣 user 不需要選曝光序列，也不需要選 fuse/postprocess mode。Runtime 會用 `config/hdr_config.yaml` 目前預設的 `[25000, 75000, 200000, 500000]`，再用候選 `default, hdr_mode_1, hdr_mode_2, hdr_mode_3` 合成並用 texture score 挑最適合 bump/texture analysis 的結果。

若要手動控制，則在 `test_template_ui.py`：

- `HDR Exposure = Manual exposures`
- `HDR Processing = Manual selected mode`
- 再手動設定 HDR frames / `HDR Config Mode`

在 `App/main.py` 的 create-template UI 裡，按 HDR toggle 或 Add HDR capture 現在會預設建立 auto HDR capture，不再是舊的 `0.5x / 1x / 2x` 三張 bracket。

**預設哪些 watchpoint 會拍 HDR**  
`create_default_scratch_template()` 現在會預設替 `Front.micropoint1-7, 13, 16` 加上 auto HDR capture。要改這份清單，改 [DB/templates/default_template_factory.py](D:/Provenance Laboratories projects/ImagingLibWatch/DB/templates/default_template_factory.py) 裡的 `AUTO_TEXTURE_HDR_DEFAULT_SELECTORS`。

**驗證**  
已跑：

- `py_compile` 檢查所有改到的 Python 檔案，通過
- smoke test 建立 default scratch template，確認 `Front.micropoint1.hdr_1` 是 `[25000, 75000, 200000, 500000] auto_texture_bracket auto_texture`
- 已依 repo 指示搜尋 single-image assumptions；V3 `WorkflowManager`/DB asset path 已支援 `raw_bracket` + `hdr_fused`

**Residual Risks**  
尚未做實機硬體拍攝驗證。建議下一步用一個 front micropoint 實拍，確認 DB `image_assets` 有多張 `raw_bracket` 和一張 `hdr_fused`，並確認 texture/bump_wrapper 用的是 `hdr_fused`。另外，legacy App/report path 仍以 primary image 為主，會使用 fused HDR，不會完整展示每張 bracket。