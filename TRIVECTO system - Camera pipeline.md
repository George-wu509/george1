

|     |     |
| --- | --- |
|     |     |
|     |     |
|     |     |
|     |     |
|     |     |


Camera pipeline
```
請幫我詳細列出pushButton_TemplateRun_3之後的camera pipeline流程, 列出function跟所屬的python file 尤其是execute_template_point function跟run_robust_autofocus function. 是不是在hardware_config裡的autofocus_option=0,1,2,3會執行run_robust_autofocus, autofocus_option=4會執行execute_template_point? 是否可畫成樹狀流程圖.
```
### 關於 Autofocus 的迷思釐清

**不，這是一個常見的誤解！** `autofocus_option` (0, 1, 2, 3, 4) 並不是用來決定要走 `run_robust_autofocus` 還是 `execute_template_point`。

這兩個函數的根本差異在於**「使用情境 (Context)」**，而不是選項編號：

- **`execute_template_point()` (位於 `unified_driver.py`):** 這是**「全自動化拍攝流程」**的核心大腦。當你按下 `pushButton_TemplateRun_3` (自動執行 Front 視角流程) 或是跑自動化測試時，系統會呼叫這個函數。它**同時支援了 Option 0, 1, 2, 3, 4**。它將所有的對焦、打光、移動、拍照邏輯（共 7 個 Phase）包裝成一個完整動作。
    
- **`run_robust_autofocus()` (位於 `unified_driver.py`):** 這是一個**「手動/獨立的對焦輔助工具」**。當工程師在控制面板 (Control Panel) 點擊 "Auto Focusing..." 或發送 `disp_img` 動作指令時，才會呼叫這個函數。它只實作了 Option 1, 2, 3，並不包含 Option 4。
    

---

### 📷 Camera Pipeline 流程樹狀圖 (由 pushButton_TemplateRun_3 觸發)

當你按下 `pushButton_TemplateRun_3` (Front 視角執行按鈕) 時，程式的流向如下：

Plaintext

```
[UI Event] User Clicks pushButton_TemplateRun_3
 │
 ├── 1. main.py
 │   ├── _setup_front_logic()
 │   │   └── 綁定按鈕，呼叫 _run_view_sequence("front", ...)
 │   ├── _run_view_sequence()
 │   │   └── 設定狀態並呼叫 load_camimgs()
 │   ├── load_camimgs()
 │   │   └── 啟動背景執行緒 CaptureRoutineWorker 避免卡死 UI
 │   └── CaptureRoutineWorker.run()
 │       └── 呼叫 _capture_images_routine_internal()
 │           └── 遍歷 Template 點位，轉換 Config，交由 WorkflowManager 執行。
 │
 ├── 2. core/workflow_manager.py (未包含在附件中，為中介層)
 │   └── _perform_smart_step()
 │       └── 最終呼叫硬體驅動層 (Hardware Driver)
 │
 ├── 3. unified_driver.py (核心大腦)
 │   └── execute_template_point() 
 │       │
 │       ├── Phase 1: Parse Config (解析 X, Y, Z 座標、曝光、燈光、取得 z_af_option)
 │       ├── Phase 2: Lights OFF & Initial XY Move (關燈並移動 Zaber XY 軸)
 │       │
 │       ├── Phase 3: Z-Axis Mechanical AF (Z軸機械對焦核心)
 │       │   ├── 若 Option = 4: 執行內部寫死的 Keyence 2 Absolute Calculation
 │       │   ├── 若 Option = 1: 執行內部寫死的 Keyence 1 Closed-Loop
 │       │   └── 若 Option = 2 或 3: 呼叫 _af_vision_method()
 │       │
 │       ├── Phase 4 & 5: Final Alignment & Light ON (移動最終Z軸，開燈，準備相機)
 │       │
 │       ├── Phase 6: Liquid Lens Autofocus (液態鏡頭對焦)
 │       │   └── 呼叫 hardware_managers.py 的相機層
 │       │       ├── 優先嘗試: trigger_hardware_liquid_af() (硬體 FPGA 對焦)
 │       │       └── 備案嘗試: run_liquid_lens_autofocus() (軟體多步掃描對焦)
 │       │
 │       └── Phase 7: Image Capture (觸發快門拍照)
 │           └── 呼叫 hardware_managers.py 的相機層
 │               ├── 若啟用 HDR: capture_enhanced_hdr()
 │               └── 一般拍攝: capture_image()
 │
 └── 4. main.py (回傳結果)
     └── _on_single_image_captured()
         └── 將拍好的照片即時更新到 UI 的九宮格與表格中
```

---

### 核心函數深入解析

#### 1. `execute_template_point(self, params)`

- **檔案位置:** `Controller/hardware_drivers/unified_driver.py`
    
- **職責:** 將單一「點位(Point)」的所有參數 (XYZ、打光、對焦模式、相機設定) 轉換為實體硬體動作，並回傳拍好的影像。
    
- **AF 判斷邏輯 (Phase 3):** 它會先從 `hardware_config.yaml` 抓取全局的 `autofocus_option`，如果該點位有 override (例如 `front.micropoint14: 0`)，則套用 override 值。
    
    - **Option 4:** 讀取 `keyence_offset`，移動 XY 後，讀取 `out01` 數值並直接加上 `keyence2_base_z` 算出絕對高度。
        
    - **Option 1:** 移動到測距點，使用 `out01` 進行多次迴圈修正 (Closed-Loop)，直到誤差小於 `tolerance`。
        
    - **Option 2 & 3:** 打開燈光，呼叫同檔案內的 `_af_vision_method` 利用影像銳利度 (Laplacian / Tenengrad) 來找最佳 Z 軸。
        

#### 2. `run_robust_autofocus(self, base_x, base_y, base_z, cam_alias, has_glass=True)`

- **檔案位置:** `Controller/hardware_drivers/unified_driver.py`
    
- **職責:** 獨立的對焦模組。主要被工程模式 UI (例如 `ControlEngWindow` 或 `test_template_ui.py`) 裡面的獨立「Auto Focus 按鈕」呼叫。
    
- **AF 判斷邏輯:** * **Option 0:** 直接回傳 `base_z` (不對焦)。
    
    - **Option 1:** 呼叫 `_af_keyence_method()`。
        
    - **Option 2 & 3:** 呼叫 `_af_vision_method()`。
        
    - _(注意：此函數完全沒有實作 Option 4 的邏輯)_