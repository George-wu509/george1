
|                                             |     |     |
| ------------------------------------------- | --- | --- |
| [[#### 2026 1001 debug_bezel merge to dev]] |     |     |
|                                             |     |     |
|                                             |     |     |

#### 2026 1001 debug_bezel merge to dev
```
我的GitHub repo - ImagingLibWatch目前有多個branch, 目前我主要在開發的是dev branch, 這在我的Nitro2025電腦的D:\Provenance Laboratories projects\ImagingLibWatch, 而另一個branch debug_bezel是從dev branch分支出去由其他人開發, 目前在我的D:\Provenance Laboratories projects\ImagingLibWatch_debugbezel. 我預計要將branch debug_bezel merge到我的dev branch. 請幫我檢查branch debug_bezel有哪些dev branch沒有的code變動是dev branch沒有的, 或有哪些是雙方不同的地方請指出, 並幫我評估直接merge是否適合要確保dev branch的變動不會被蓋掉, 先不要修改code.
```
已完成唯讀比較，沒有修改、切換或合併任何程式碼。

## 結論

目前不適合直接合併後立即提交。一般 `git merge debug_bezel` 會產生 3 個衝突檔案、共 5 個 conflict hunks；此外還有 8 個檔案雖然 Git 能自動合併，但包含硬體動作、校正值和工作流程順序，仍需人工審查。

不過，`dev` 的大部分變動不會被覆蓋：

- `dev` 自共同祖先後修改 382 個檔案。
- `debug_bezel` 修改 34 個檔案。
- 其中 360 個檔案只由 `dev` 修改，正常三方合併會保留。
- `debug_bezel` 沒有相對共同祖先刪除或重新命名檔案。
- 不應使用 `-X theirs`、整檔接受 theirs，或 GitHub 的無審查自動解衝突。

## Branch 狀態

|項目|結果|
|---|---|
|`dev`|`5ed3b42`，2026-10-01|
|`debug_bezel`|`81fb386`，2026-09-28|
|共同祖先|`bd4eee4`，2026-09-03|
|分岔後 commit|`dev` 79、`debug_bezel` 25|
|Patch 等價分析|`debug_bezel` 有 5 個 commit 已等價存在於 `dev`，20 個含獨有或不同 patch|
|工作目錄|兩邊 tracked files 都乾淨；`dev` 另有既存未追蹤 `.claude/`|
|Origin 狀態|都與本機的 `origin/*` tracking ref 相同，0 ahead / 0 behind|

本次沒有執行 `git fetch`，因此最後一項代表「與上次 fetch 的 origin ref 一致」，不是即時驗證 GitHub 遠端。

## 實際 merge 衝突

1. [App/main.py](D:\\Provenance Laboratories projects\\ImagingLibWatch\\App\\main.py)
    
    `_handle_zero_capture_preflight` 行為不同：
    
    - `dev`：空 view 顯示確認訊息，使用者確認後才標示完成。
    - `debug_bezel`：把空 view 視為合法定義，靜默標示完成並繼續。
    
    這是產品行為選擇，不是單純程式碼格式衝突。相應測試也必須選同一套預期。
    
2. [config/hardware_config.yaml](D:\\Provenance Laboratories projects\\ImagingLibWatch\\config\\hardware_config.yaml)
    
    兩個衝突：
    
    - micro offset 數值相同，只有註解空格不同。
    - `dev` 新增玻璃厚度 OUT1/OUT4 tolerance；`debug_bezel` 在同位置新增 Keyence OUT6／per-internalnum channel 說明。
    
    正確處理應保留兩邊設定，而不是選一邊整段覆蓋。
    
3. [Controller/test_hardware/template_test.json](D:\\Provenance Laboratories projects\\ImagingLibWatch\\Controller\\test_hardware\\template_test.json)
    
    兩邊實際上是不同用途的 fixture：
    
    - `dev`：`Rolex_SubMariner`、`3.1-box`。
    - `debug_bezel`：`Rolex_16610` watch template，並包含約 6,000 行額外 point 資料。
    
    接受 theirs 會實質取代 dev 的 box fixture。較安全做法是保留 dev fixture，另存 debug watch fixture或人工抽取必要 point。
    

## `debug_bezel` 真正帶來的主要變動

- 硬體與高度量測：
    
    - 修改 Keyence probe movement 流程。
    - 高度量測失敗時不再把 fallback 當成功。
    - 新增 OUT1/OUT4 反射誤判 cross-check。
    - 修改 macro/micro offset、部分 Y／rotation 校正。
    - 六個 internal number 改為 `use_position: 1`。
    - 新增 GMT 點 `0059/0060`、movement 點 `2013–2017`。
- 拍攝順序：
    
    - 新增 [core/point_visit_order_template.py](D:\\Provenance Laboratories projects\\ImagingLibWatch_debugbezel\\core\\point_visit_order_template.py)。
    - 此檔會重新排序 production capture 和 simulation，不只是測試工具。
    - 未列出的點會附加在後面，但硬編碼列表日後可能與 dataclass 不同步。
- App 手動重拍：
    
    - 開啟 control panel 時立即移動實體 stage 到 point pose。
    - 把 WatchEntry 的設定同步回 current template。
    - 清除該 point 的 thumbnail 後重新拍攝。
    
    對目前「一個 point 多張圖＋HDR」架構有一項風險：操作是 point-scoped，會清除該 point 的所有 gallery rows並重跑 point，但只同步 active capture 的部分設定。需要驗證多 standard capture／HDR 時不會誤清或混合 point-level 與 image-level metadata。
    
- Template/UI：
    
    - 分離 watch template 與 box template 搜尋。
    - 阻止 box template 從 watch workflow 開啟。
    - 移除輸入完整名稱時自動執行的行為，改由 Enter／選擇觸發。
- 新增工具及資料：
    
    - direct probe GUI、stage click mapping bench。
    - bezel／stage notebooks。
    - 兩張 `Claude outputs` 圖片。
    - 修改 `AGENTS.md`，加入特定硬體測試偏好。

續拍與 PDF viewer 的大部分程式其實已等價存在於 `dev`；有 11 個共同修改檔案在兩個 branch 的最終內容完全相同。

## Git 可自動合併、但仍應人工檢查的檔案

- `App/test_template_create_rules.py`
- `Controller/hardware_drivers/unified_driver.py`
- `DB/db_manager.py`
- `DB/templates/watchentry_structure.py`
- `config/internalnum_config.yaml`
- `core/workflow_manager.py`
- `helper/deployment/build_moonlight.py`
- `helper/deployment/deploy_moonlight.py`

前三個高風險點是 `unified_driver.py`、`internalnum_config.yaml`、`workflow_manager.py`，因為它們直接影響實機移動、校正位置及拍攝順序。

## 建議合併方式

建議從 `dev` 建一個臨時 integration branch，再使用 `--no-commit --no-ff` 合併並人工解決上述 3 個檔案。若目標其實只是 bezel／movement 點位，不需要手動重拍、PDF、box search 等功能，選擇性 cherry-pick 會比整支 branch merge 更可控。

合併後至少要驗證：

- 舊 template 與多 image／HDR template 的 load-save round trip。
- WatchEntry resume、SQLite insert、DynamoDB/S3 serialization 與 report smoke test。
- 一個 point 有多張圖時的手動重拍與 thumbnail 清理。
- Keyence probe movement、OUT1/OUT4 判斷及反光件實機量測。
- `0059/0060`、`2013–2017` 座標；branch 註解本身已標示部分為 placeholder／需實機確認。
- production 與 simulation 的 point visit order。
- watch/box 搜尋及空 view 的最終 UX 決定。

靜態檢查方面，`debug_bezel` 修改的 22 個 Python 檔案都可解析，兩邊的相關 JSON/YAML 也可解析；目前只發現少量 trailing
