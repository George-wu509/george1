


|                                                                 |     |
| --------------------------------------------------------------- | --- |
| [[#### 三個相機 (Cameras) 的對焦計畫]]                                   |     |
| [[#### hardware_config的參數autofocus_camera方案(0-4)]]              |     |
| [[#### 「粗對焦 (Coarse Focus)」與「細對焦 (Fine Focus)」]]                |     |
| [[#### 封裝成一支獨立的通用 API：perform_two_stage_autofocus]]             |     |
| [[#### 重新reorganize hardware_config.yaml]]                      |     |
| [[#### stand capture, hdr capture跟focus capture如何控制auto focus]] |     |
| [[#### af_modes=0(default)的詳細autofocus流程]]                      |     |
| [[#### af_modes=1~8的autofocus]]                                 |     |
|                                                                 |     |


#### 三個相機 (Cameras) 的對焦計畫

```
我的系統裡面有三個camera(macro_cam_1, macro_cam_2, micro_cam). 目前micro_cam有很詳細的兩階段focus plan(Keyence autofocus + liquid lens auto focus以及很多因應特殊情況的做法譬如往x,y平面或z平面搜尋). 除此之外還有Vision(Laplacian), Vision(Tenengrad)等方法

請幫我整理目前三個camera所使用的focus planu以及中文詳細其細節, 甚麼時候會用到auto focus?
```

根據您系統中的設定檔 ([hardware_config.yaml](code-assist-path:d:\Provenance%20Project\ImagingLibWatch\config\hardware_config.yaml "d:\Provenance Project\ImagingLibWatch\config\<mark style="background: #FFF3A3A6;">hardware_config.yaml</mark>")) 與驅動程式 (`UnifiedHardwareDriver`, `CameraManager`) 的原始碼，我為您整理了三個相機目前的對焦計畫 (Focus Plan)，以及各種對焦技術的詳細運作細節與觸發時機。

### 一、 三個相機 (Cameras) 的對焦計畫

系統透過 hardware_config.yaml中的 `autofocus_camera` 與 `devices` 區塊，為不同的相機指派了不同的預設對焦行為：

1. **`macro_cam_1` (微距頂部相機)**
    
    - **預設對焦計畫：** **Option 4** (Keyence Absolute Calculation / 絕對座標計算式)
    - **說明：** 依賴 Keyence 雷射測距儀進行實體 Z 軸馬達的粗調/精調對焦。不使用液態鏡頭。
2. **`macro_cam_2` (微距側面相機)**
    
    - **預設對焦計畫：** **Option 0** (Disable / 關閉自動對焦)
    - **說明：** 側面相機通常固定景深與位置，預設不進行動態自動對焦，而是直接前往檢測範本 (Template) 中紀錄的固定 Z 軸座標。
3. **`micro_cam` (顯微液態鏡頭相機)**
    
    - **預設對焦計畫：** **兩階段混合對焦** (Option 4 + FPGA Liquid Lens AF)
    - **說明：**
        - **第一階段 (機械 Z 軸粗調)：** 先使用與 `macro_cam_1` 相同的 Option 4 (Keyence 雷射)，將實體 Z 軸快速移動到接近焦平面的高度。同時也會補償這兩顆鏡頭間的機構高低差 (`parfocal_z_offset`)。
        - **第二階段 (液態鏡頭微調)：** 觸發 Opto-Engineering 相機內建的 FPGA 硬體液態鏡頭對焦 (`af_method: "hardware"`)，透過瞬間改變鏡頭電流來達成極高速的精準對焦，此時 Z 軸馬達保持靜止。

---

### 二、 對焦演算法細節與特殊情況處理

在 UnifiedHardwareDriver.<mark style="background: #FF5582A6;">run_robust_autofocus</mark>中，實作了多種對焦策略：

#### 1. 機械 Z 軸對焦 (雷射測距)

- **Option 1: Keyence Closed-Loop (迴圈逼近式)**
    - **細節：** Z 軸先移動到標準測量高度，讀取雷射數值後，計算與「完美焦距讀數 (`target_out1`)」的誤差，然後移動 Z 軸。移動後再讀、再算、再移動，直到誤差小於容許值 (`tolerance`)。這是一個不斷微調的閉迴路過程。
- **Option 4: Keyence Absolute Calculation (絕對座標計算式) - _系統目前主要使用的雷射對焦法_**
    - **細節：** 預先在系統中校正出一個純光學焦點對應的機械常數 (`keyence2_base_z` 或有玻璃的 `keyence2_base_z_glass`)。運作時，雷射只讀取「一次」數值，直接用 `基準常數 + 雷射讀數` 一次性算出目標 Z 軸高度並移動到位，速度極快。
    - **特殊情況 (XYZ 擴展搜尋)：** 當雷射打到手錶孔洞、反光太強或吸光材質導致讀不到數值時，系統會觸發 `_read_keyence_absolute_output_with_expansion`：
        - **XY 平面搜尋 (`keyence_read_roi_expan`)：** 機台會以目標為中心，像九宮格一樣向外擴展 (`keyence_read_roi_dist`)，尋找附近是否有實體表面可以反射雷射。
        - **Z 平面搜尋 (`keyence_read_z_expan`)：** 如果 XY 都找不到，代表機台可能太高或太低，Z 軸會往上/往下分層移動，再次嘗試捕捉雷射訊號。

#### 2. 影像視覺對焦 (Vision AF)

這是不依賴雷射，純粹靠分析相機拍攝出來的畫面「清晰度」來決定焦點。系統會控制 Z 軸在一個範圍內 (`z_search_range`) 上下步進拍照。

- **Option 2: Vision (Laplacian)**
    - **細節：** 計算影像灰階的「拉普拉斯變異數 (Laplacian Variance)」，也就是二階導數。這能敏銳地抓出影像邊緣的高頻細節。變異數最高的那張照片，就代表對焦最精準。
- **Option 3: Vision (Tenengrad)**
    - **細節：** 使用 Sobel 算子計算水平與垂直方向的梯度 (一階導數)。將梯度平方加總，分數最高的即為焦點。這對某些金屬拉絲紋理的判斷有時比 Laplacian 穩定。

#### 3. 液態鏡頭自動對焦 (Liquid Lens AF - 專屬 `micro_cam`)

- **硬體對焦 (FPGA Hardware AF)：** 由相機內建的 FPGA 晶片主導。軟體送出一個掃描範圍 (如 -10mA 到 10mA)，相機硬體會用極高的幀率自己掃描、自己計算清晰度並鎖定電流。
    - **特殊情況 (邊界重試 `z_boundary_retries`)：** 如果硬體回報的最佳電流剛好卡在範圍的極限邊界 (例如剛好是 10mA)，代表真正的焦點可能在範圍之外。系統會微調實體 Z 軸高度，然後重新叫硬體再掃一次。
    - **特殊情況 (Best-of-N 策略)：** 為防止硬體被金屬反光欺騙造成「假對焦」，代碼中啟用了 `fpga_af_best_of_n_enable`。它會連續讓硬體對焦 N 次，每次對焦完拍一張圖算 Laplacian 銳利度，最後強制採用銳利度最高的那次電流。
- **軟體對焦 (Software Sweep AF)：** 作為硬體的備案或校正工具 (`software_fallback_enable`)。由 Python 程式端主控，逐步改變電流 -> 拍照 -> 算 Laplacian 銳利度，找出最佳電流。

---

### 三、 什麼時候會用到 Auto Focus (自動對焦)？

在系統的運行生命週期中，自動對焦主要在以下幾個時機被觸發：

1. **常規自動化檢測流程 (Routine Execution)：** 當系統透過 WorkflowManager.<mark style="background: #ABF7F7A6;">execute_routine</mark> 執行一連串的檢測點位時，針對每一個點位 (`CaptureCondition`)，只要設定檔中的 **`use_autofocus: 1`** (預設通常為 1)，系統在移動馬達到達該 (X, Y) 座標後、正式拍照之前，就會強制呼叫 <mark style="background: #ABF7F7A6;">run_robust_autofocus</mark>。
    
    - _註：如果只是同一個位置連拍三張不同曝光做 HDR，或者切換打光，只要 X, Y 沒動，就不會重複對焦。_
2. **UI 介面手動檢測與調校 (Template Testing)：** 在工程師使用的 <mark style="background: #FFF3A3A6;">test_template_ui.py</mark> 中：
    
    - 當點擊 **"EXECUTE SEQUENCE & VERIFY"** 或 **"TEST APP PIPELINE"** 時，會走標準流程，依照勾選的狀態執行雷射與液態鏡頭對焦。
    - 當獨立點擊 **"Test FPGA Liquid Lens AF"** 按鈕時，會不移動 Z 軸，純粹命令 `micro_cam` 進行一次液態鏡頭硬體對焦。
3. **系統校正模式 (Calibration)：** 在 <mark style="background: #FFF3A3A6;">test_keyence_optimize_ui.py</mark> 或自動校正腳本中，為了找尋基準常數 `Z_ref`，系統會主動使用 **Vision AF (Option 2)** 去尋找「光學真理 (絕對清晰的物理平面)」，然後再拿這個平面與 Keyence 的雷射讀數做數學反解校正。
    
4. **建立或匯入檢測範本 (Template Authoring)：** 當使用者教導一個新手錶點位並存入 JSON 時，會將當下對焦好的最佳 Z 軸與液態鏡頭電流 (`cammag`) 記錄下來。如果未來將該點位的 `use_autofocus` 設為 0，系統就會**不執行**自動對焦，直接盲飛到這個記憶的焦點進行拍攝。



#### hardware_config的參數autofocus_camera方案(0-4)
```
在hardware_config的參數有autofocus_camera:
      macro_cam_1: 4
      macro_cam_2: 0
      micro_cam: 4 

請整理如果這三個camera各選擇0,1,2,3,4他們的focus方案各是甚麼? 都會使用liquid lens auto focus嗎
```
在系統設計中，設定檔裡的 `0, 1, 2, 3, 4` 方案控制的是**「實體 Z 軸馬達 (Zaber)」**的對焦行為（對應程式碼中的 Phase 2）；而**「液態鏡頭對焦 (Liquid Lens AF)」**則是獨立運作的另一個階段（對應程式碼中的 Phase 4）。

這代表**這五個選項與是否觸發液態鏡頭沒有絕對關聯**，液態鏡頭是否啟動取決於相機本身的硬體設定。

以下為您詳細整理這 5 種方案的差異，以及 3 個 Camera 對應的液態鏡頭運作邏輯：

### 一、 `0, 1, 2, 3, 4` 各代表什麼 Focus 方案？

這五個數字定義了實體 Z 軸馬達如何去尋找物理焦平面：

- **`0` (Disable / 關閉 Z 軸對焦)** 系統不會進行任何測距或視覺掃描，直接將 Z 軸馬達移動到 Template 預先記錄的固定座標。
- **`1` (Keyence Closed-Loop / 雷射閉迴路微調)** 依賴 Keyence 雷射測距儀。Z 軸先移動到測量起點，讀取雷射誤差後移動馬達，接著「再讀、再算、再移」，不斷在一個迴圈內重複，直到誤差小於容許值 (`tolerance: 0.005`) 才停止。
- **`2` (Vision Laplacian / 視覺對焦 - 拉普拉斯)** 不使用雷射。控制 Z 軸馬達在設定範圍（如上下 ±0.5mm）內分層步進移動，每動一步拍一張照片。透過 Laplacian 演算法計算影像銳利度（二階導數），最後把 Z 軸停在銳利度最高的高度。
- **`3` (Vision Tenengrad / 視覺對焦 - 梯度算子)** 運作方式與 `2` 完全相同，只是評分演算法改用 Sobel 一階梯度算子 (Tenengrad) 來判斷紋理邊緣的清晰度。
- **`4` (Keyence Absolute Calculation / 雷射絕對計算式)** **（系統主力方案）**。依賴 Keyence 雷射測距儀，只讀取「一次」數值，利用預先校正好的絕對光學基準常數 (`keyence2_base_z`)，套用公式直接算出目標 Z 軸高度，然後馬達「一步到位」，速度極快，並帶有防呆（找不到讀數時擴展 XY/Z 搜尋）的機制。

---

### 二、 這些 Camera 都會使用 Liquid Lens Auto Focus 嗎？

**答案是：只有 `micro_cam` 會使用，`macro_cam_1` 與 `macro_cam_2` 絕對不會使用。**

根據 `[unified_driver.py](code-assist-path:d:\Provenance%20Project\ImagingLibWatch\Controller\hardware_drivers\unified_driver.py "d:\Provenance Project\ImagingLibWatch\Controller\hardware_drivers\unified_driver.py")` 中的 `execute_template_point` 邏輯，系統會檢查硬體設定中的 `lens_type`，只有當類型包含 `"liquid"` 時，才會觸發液態鏡頭對焦。

1. **`macro_cam_1` 與 `macro_cam_2` (微距相機)**
    - **`lens_type: "fixed"`**：這兩顆相機是定焦鏡頭。
    - **結果**：無論你為它們選擇 0, 1, 2, 3, 4 的哪一個，系統執行完 Z 軸馬達對焦後，都會直接跳過 Phase 4（液態鏡頭對焦），進入拍攝階段。
2. **`micro_cam` (顯微相機)**
    - **`lens_type: "liquid_tcel350"`**：具備 Optotune 液態鏡片。
    - **結果**：這顆相機採取的是**「兩階段混合對焦」**。不管你選哪個選項，系統都會在 Z 軸馬達停定後，強制啟動相機內建的 FPGA 硬體去微調液態鏡頭電流 (`trigger_liquid_lens_hardware_autofocus`)。

#### `micro_cam` 的各選項運作細節：

- **如果設為 `4` (目前設定)**： 第一階段 Z 軸馬達利用雷射算式快速飛到正確高度（粗調） -> 接著馬達靜止，第二階段 FPGA 瞬間改變電流，精準鎖定焦平面（極微調） -> 拍照。
- **如果設為 `0`**： Z 軸馬達直接盲飛到定點不做對焦 -> 馬達靜止，依賴 FPGA 液態鏡頭靠改變電流把焦點「拉」回來 -> 拍照。（前提是誤差不能超出液態鏡頭的變形極限）。
- **如果設為 `2` 或 `3` (視覺對焦)**： 系統會**先強制將液態鏡頭的電流設為 0.0mA (鎖死鏡頭)** -> 進行 Z 軸馬達的上下影像掃描，找出最佳物理高度 -> 馬達靜止，再啟動 FPGA 液態鏡頭做最後的電流微調掃描 -> 拍照。



#### 「粗對焦 (Coarse Focus)」與「細對焦 (Fine Focus)」

```
目前在run_robust_autofocus，實作了多種對焦策略: 
	Option 0: 不進行auto focus
	Option 1: Keyence Closed-Loop (迴圈逼近式)
	Option 2: Vision (Laplacian)
	Option 3: Vision (Tenengrad)
	Option 4: Keyence Absolute Calculation

所以這些不同策略是個有專屬的function嗎? 如果不是應該封裝成個別的functions比較好管理?

然後還有liquid lens的autodocus, 握們可以把camera pipeline的autofocus分成兩階段(粗對焦跟細對焦)之前run_robust_autofocus的策略都歸於粗對焦, 而liquid lens的autodocus就是細對焦. liquid lens的autodocus包含特殊情況 (XYZ 擴展搜尋)是否有封裝成一個function? 在目前的code structure, liquid lens的autodocus function是否可接在不同的run_robust_autofocus的Option 0-4後面? 

以及目前micro_cam (顯微液態鏡頭相機)兩階段混合對焦 (Option 4 + FPGA Liquid Lens AF)是寫在同個functions裡面嗎? 還是是結合兩個functon執行?
```

這是一個非常好的架構問題！您對「粗對焦 (Coarse Focus)」與「細對焦 (Fine Focus)」的拆分觀念，完全符合現代高精度光學檢測設備（尤其是搭配液態鏡頭的微距系統）的設計模式。

在目前的 codebase  中，這個兩階段對焦的架構**已經被完美解耦且封裝成獨立的 functions**。以下為您詳細拆解目前的程式碼結構：

### 1. 粗對焦策略 (Option 0-4) 是否有專屬 function？

**是的，已經完全封裝。** `run_robust_autofocus` 實際上只是一個 **「路由/分發器 (Router/Hub)」**，負責根據相機設定與使用者選擇的 `af_option` 來決定呼叫哪一個底層的專屬 function。

在 \Controller\hardware_drivers\ <mark style="background: #FFF3A3A6;">unified_driver.py </mark>中，各策略對應的專屬函數如下：

- **Option 1 (Keyence Closed-Loop):** 封裝在<mark style="background: #FFF3A3A6;">unified_driver.py </mark>中 <mark style="background: #ABF7F7A6;">_af_keyence_closed_loop</mark>(...)
- **Option 2 & 3 (Vision Laplacian/Tenengrad):** 封裝在<mark style="background: #FFF3A3A6;">unified_driver.py </mark>中 self.<mark style="background: #ABF7F7A6;">_af_vision_method</mark>(...)
- **Option 4 (Keyence Absolute):** 封裝在<mark style="background: #FFF3A3A6;">unified_driver.py </mark>中 self._<mark style="background: #ABF7F7A6;">af_keyence_absolute</mark>(...)
- **Option 0 (Disable):** 直接在 Router 內攔截並回傳原 Z 軸座標。

這種設計非常容易管理，未來如果要新增 Option 5（例如基於 AI 的 Z 軸預測），只需新增一個 `_af_ai_predict(...)` 函數並在 Hub 中加入一個 `elif` 即可。

### 2. 液態鏡頭細對焦 (Liquid Lens AF) 的封裝與通用性

**有獨立封裝，且完全可以接在 Option 0-4 任何一個粗對焦之後。**

液態鏡頭的對焦邏輯（包含您提到的 XYZ 邊界擴展搜尋、硬體容錯、連續 N 次取最佳值等複雜情境），都被完整封裝在以下兩個專屬函數中：

- **硬體 (FPGA) 驅動：** 在<mark style="background: #FFF3A3A6;">unified_driver.py </mark>中 self.<mark style="background: #ABF7F7A6;">trigger_liquid_lens_hardware_autofocus</mark>(...)
- **軟體 (Software Sweep) 驅動：**在<mark style="background: #FFF3A3A6;">unified_driver.py </mark>中 self.<mark style="background: #ABF7F7A6;">trigger_liquid_lens_software_autofocus</mark>(...)

**為何能接在任何粗對焦之後？** 因為這兩個函數**「不依賴」**前面的 Z 軸是怎麼走到目標位置的。只要 Z 軸（粗對焦）停在一個合理範圍內，呼叫這個函數時，它就會在當下的物理 Z 軸基礎上，進行液態鏡頭的微調電流對焦。這保證了極高的模組化與重用性。

### 3. `micro_cam` 兩階段混合對焦是寫在同一個 function 嗎？

**不是，它們是「組合兩個 function」來執行，統一交由上層的 Pipeline 函數進行編排 (Orchestration)。**

在系統中，有兩個主要的地方展示了這種完美的結合：

**情境 A：UI 手動交互式拍攝 (`execute_interactive_micro_capture`)** 在這個函數中，您可以看到非常明顯的兩階段分離寫法：

```
`# Step 4: 呼叫粗對焦 (Keyence Z-Sweep 等) rough_z, success, _ = self.run_robust_autofocus(...)  # Step 5: 馬達移動到粗對焦高度 self.move_compound({"stage_L_X": ..., "stage_L_Y": ..., "stage_L_Z": rough_z})  # Step 8: 呼叫細對焦 (FPGA Liquid Lens) af_success, locked_current = self.trigger_liquid_lens_hardware_autofocus(...)`
```


**情境 B：App 全自動量產流程 (`execute_template_point`)** 這個函數是系統的核心大腦，它將整個流程拆分為不同 Phase：
```
- **Phase 2:** 機械位移與實體 Z 軸粗對焦 (Safety Planner + Mechanical Z-AF)
    - 這裡會根據 Template 設定，呼叫對應的粗對焦專屬函數 (`_af_keyence_absolute` 等)。
- **Phase 3:** 相機與光源整備。
- **Phase 4:** Liquid Lens AF (細對焦)。
    - 在這裡會呼叫 `trigger_liquid_lens_hardware_autofocus`。
```

### 總結

目前的結構不僅清晰，且極度彈性。

- **解耦 (Decoupling)：** 機械 Z 軸 (Zaber) 與光學 Z 軸 (Opto Engineering Liquid Lens) 各自有獨立的驅動函數與驗證邏輯。
- **組合 (Composition)：** 上層的 Pipeline 函數 (`execute_template_point`) 像積木一樣，根據 Template 傳入的設定 (`z_af_option`, `liquidlens_autofocus` 開關)，自由排列組合要先用哪種粗對焦，再決定是否疊加細對焦。

您不需要將它們寫成同一個巨大的函數，目前的這種「各司其職，由上層編排」的設計正是大型自動化設備軟體最佳的設計模式。






#### 封裝成一支獨立的通用 API：perform_two_stage_autofocus
```
目前的兩階段混合對焦 (Option 4 + FPGA Liquid Lens AF)只適用於micro_cam, 那如果我想要micro_cam 在粗對焦階段能很容易切換不同粗對焦策略(option0-4), 而且容易通過hardware_config開啟細對焦 liquid lens的autodocus. 然後micro_cam, macro_cam_1, macro_cam_2都可以有同樣的auto focus流程, 三個cameras的autofocus的設定也都同樣可以由hardware_config控制. 請建議如何實作, 或這樣修改是否更好
```

這樣修改 **絕對是更好且最標準的作法**！這在軟體工程中稱為 **「設定驅動架構 (Config-Driven Architecture)」**。

將所有 Camera 的對焦邏輯統一，把「要不要粗對焦 (Option 0-4)」、「要不要細對焦 (Liquid Lens)」的決定權完全交還給 `[hardware_config.yaml](code-assist-path:d:\Provenance%20Laboratories%20projects\ImagingLibWatch\config\hardware_config.yaml "d:\Provenance Laboratories projects\ImagingLibWatch\config\hardware_config.yaml")`，能帶來極大的好處：

1. **程式碼高度重用**：`macro_cam_1`, `macro_cam_2`, `micro_cam` 呼叫同一支 API，不再需要為不同相機寫 hardcode。
2. **未來擴充性極強**：如果某天為 `macro_cam` 加裝了液態鏡頭，您完全不需要改 Python 程式碼，只要去 YAML 把 `liquidlens_autofocus` 改成 `true`，流程就會自動啟動細對焦。

### 建議的實作方式

我們可以將這個邏輯封裝成一支獨立的通用 API：`perform_two_stage_autofocus`。這支 API 會動態讀取 YAML，自動判斷該相機目前需要執行哪些對焦階段。

#### 1. YAML 設定檔的對應方式

未來您的 hardware_config.yaml就可以這樣優雅地控制：

```
# 1. 控制「粗對焦 (Coarse AF)」的策略 (Option 0-4)
autofocus:
  autofocus_camera:
    macro_cam_1: 4  # 使用 Option 4 (Keyence Absolute)
    macro_cam_2: 0  # 使用 Option 0 (Disabled)
    micro_cam: 4    # 使用 Option 4 (Keyence Absolute)

# 2. 控制「細對焦 (Fine AF)」的開關
cameras:
  devices:
    macro_cam_1:
      lens_type: "fixed"
      liquidlens_autofocus: false  # 不觸發細對焦
    micro_cam:
      lens_type: "liquid_tcel350"
      liquidlens_autofocus: true   # 觸發細對焦

```

#### 2. 程式碼實作 (Unified Driver)

我為您在 unified_driver.py函數。它會完美地串接粗對焦與細對焦，並自動處理 Z 軸的位移與 Parfocal (齊焦) 補償。

以下是針對 unified_driver.py 的程式碼修改：
###### 這樣修改帶來的好處

加入 `perform_two_stage_autofocus` 後，原本在 UI 測試中冗長且被 Hardcode 的 `execute_interactive_micro_capture`，現在可以重構成非常乾淨的邏輯（您甚至可以把它改名為 `execute_interactive_capture` 給所有鏡頭用）：**總結來說：** 您這個想法非常精準，將這兩段對焦統一封裝並用 Config 控管，是大型設備軟體邁向成熟的關鍵一步。目前的 `run_robust_autofocus` 和 `trigger_liquid_lens_hardware_autofocus` 的解耦狀態非常良好，用這層 Wrapper (包裝函數) 把他們串起來，是最優雅的解法！



#### 重新reorganize hardware_config.yaml
```
請幫我把hardware_config.yaml裡面原本放在cameras關於autofocus的部分統一放在autofocus之下, 而在autofocus之下先寫跟對所有cameras(譬如autofocus_point....)的設定, 之後再分別分成三個cameraszz放各自專屬的設定. 也註定三個camera都要有完整的粗對焦跟細對焦的個別的parameters. 另外如果hardware_config.yaml 對focus parameters做了修正, 也要對相對應的code/functions作相對應的修正
```





```
**
Claude code 3 stage prompt

我上傳的hardware_config.yaml是前一代的版本, 但我檢查過最新git pull後的最新版本似乎把很多我在hardware_config.yaml裡的autofocus的相關parameters都刪除了, 譬如在hardware.cameras.devices的macro_cam_1, macro_cam_2, micro_cam裡面liquidlens_autofocus跟liquid_af下面的parameters本來應該放到後面autofocus下面但都消失了. 

這是我上一次git pull的主要需求: "請幫我把hardware_config.yaml裡面原本放在cameras關於autofocus的部分統一放在autofocus之下, 而在autofocus之下先寫跟對所有cameras(譬如autofocus_point....)的設定, 之後再分別分成三個cameraszz放各自專屬的設定. 也註定三個camera都要有完整的粗對焦跟細對焦的個別的parameters. 另外如果hardware_config.yaml 對focus parameters做了修正, 也要對相對應的code/functions作相對應的修正"

請幫我比對我目前的hardware_config.yaml跟前一版本hardware_config.yaml重新edit, 並確保使用到這些parameters的functions都有正確修改, 也並檢查perform_two_stage_autofocus function是否有正確做到三個cameras的autofocus在粗對焦階段能很容易切換不同粗對焦策略(option0-4), 而且容易通過hardware_config開啟細對焦 liquid lens的autodocus. 然後micro_cam, macro_cam_1, macro_cam_2都可以有同樣的auto focus流程, 三個cameras的autofocus的設定也都同樣可以由hardware_config控制

**
```




#### stand capture, hdr capture跟focus capture如何控制auto focus
```
請中文詳細解釋watchpoint的stand capture, hdr capture跟focus capture如何控制auto focus? 譬如只有stand capture跟stand capture + focus capture的autofocus差異在哪裡? 如果有focus capture, 會用哪些focus capture的parameters進行autofocus(所以這時stand capture的的autofocus parameters都無用?)
```

我看了目前程式碼後，結論是：

`focus_capture` 的定位不是「多拍一張 focus 圖」，而是「給 autofocus 用的一組替代硬體參數」。真正輸出的影像還是 `standard_captures` 或 `hdr_captures` 裡的 capture。

主要路徑在：

[core/workflow_manager.py (line 2542)](D:\\Provenance Project\\ImagingLibWatch\\core\\workflow_manager.py:2542)  
[Controller/hardware_drivers/unified_driver.py (line 5533)](D:\\Provenance Project\\ImagingLibWatch\\Controller\\hardware_drivers\\unified_driver.py:5533)

**資料模型**  
在 [DB/templates/watchview_structure.py (line 113)](D:\\Provenance Project\\ImagingLibWatch\\DB\\templates\\watchview_structure.py:113)，每個 watchpoint 有三種 capture 容器：

```
standard_captures: List[CaptureCondition]
hdr_captures: List[CaptureConditionHDR]
focus_capture: Optional[CaptureCondition]
```

`standard_captures` / `hdr_captures` 是真正會拍照、存圖、跑分析的 capture。  
`focus_capture` 是 point-level 的「AF 前置條件」，會被套到每個 standard/HDR capture 上，作為 autofocus 的參數來源。

**只有 standard capture 時**  
如果某個 point 只有 `standard_captures`，沒有 `focus_capture`：

1. `workflow_manager` 會把 standard capture 的參數放進 `step["hardware"]`。
2. `execute_template_point()` 裡面沒有 `focus_hardware`，所以：
    
    ```
    af_source = params
    ```
    
3. 也就是 autofocus 和 final capture 都使用同一組 standard capture 參數。

這時 standard capture 的這些欄位會同時影響 AF 與拍照：

```
cam_id
cammag
ringlight1 / ringlight2
spotlight1 / spotlight2
sidelight
exptimes
gainlevel
use_autofocus
use_liqud_af
has_glass
Z_micro_offset
```

簡單說：沒有 `focus_capture` 時，standard capture 自己決定「怎麼對焦」也決定「怎麼拍」。

**standard capture + focus_capture 時**  
如果 point 有 `focus_capture`，`workflow_manager` 會把它轉成：

```
step["hardware"]["focus_hardware"] = {...}
```

位置在 [core/workflow_manager.py (line 2591)](D:\\Provenance Project\\ImagingLibWatch\\core\\workflow_manager.py:2591)。

進入 `execute_template_point()` 後：

```
raw_focus_params = params.get("focus_hardware") or params.get("focus_params")
focus_params = params.copy()
focus_params.update(focus_hardware)
af_source = focus_params
```

也就是 autofocus 會改用 `focus_capture` 的參數；但 final capture 仍然使用 standard capture 的參數。

差異可以這樣理解：

|階段|沒有 focus_capture|有 focus_capture|
|---|---|---|
|機械 Z AF / Keyence / vision AF|用 standard capture 參數|用 focus_capture 參數|
|liquid lens AF 的相機、曝光、gain、cammag|用 standard capture 參數|用 focus_capture 參數|
|AF 時打的燈|用 standard capture 燈光|用 focus_capture 燈光|
|最後真正拍照的相機|standard capture|standard capture|
|最後真正拍照的曝光/gain|standard capture|standard capture|
|最後真正拍照的燈光|standard capture|standard capture|
|影像 capture_id / 儲存 / 分析|standard capture|standard capture|

所以回答你的例子：

「只有 standard capture」  
standard capture 的 autofocus parameters 會直接控制 autofocus。

「standard capture + focus_capture」  
autofocus 主要改由 focus_capture 控制；standard capture 主要控制最後拍照。

**focus_capture 會用哪些 parameters 做 autofocus**  
目前程式實際會用到 focus_capture 裡這些欄位：

```
X / Y / Z 或 x / y / z
cam_id
cammag
ringlight1
ringlight2
spotlight1
spotlight2
sidelight
exptimes
gainlevel
use_autofocus
has_glass
af_mode_id
```

用途大概是：

`x/y/z`：AF 先移動到 focus pose。  
`cam_id`：指定用哪個 camera 做 AF。  
`exptimes`：AF 用的曝光；若是 HDR 且 exptimes 有多個，會選中間曝光做 AF。  
`gainlevel`：AF 用的 gain。  
`ringlight/spotlight/sidelight`：AF 時開的燈。  
`use_autofocus`：是否做機械 Z AF / Keyence / vision AF。  
`has_glass`：是否做玻璃相關 Keyence 量測。  
`cammag`：liquid lens AF 的 baseline/manual current。  
`af_mode_id`：選用 `config/af_mode_config.yaml` 裡的 AF mode override。

**那 standard capture 的 autofocus parameters 都沒用了嗎？**  
不是全部沒用，但大部分「AF 用參數」會被 `focus_capture` 覆蓋。

有 `focus_capture` 時，這些 standard capture 欄位仍然很重要：

```
cam_id
exptimes
gainlevel
ringlight1 / ringlight2
spotlight1 / spotlight2
sidelight
postcolor
rotation
capture_id
hdr_* parameters
z_positions
```

因為它們控制最後拍出來的圖。

但對 AF 來說，`use_autofocus`、`has_glass`、AF 用 `cam_id/exptimes/gain/lights/cammag` 會改看 `focus_capture`。

不過有一個重要細節：照目前程式碼，`use_liqud_af` 的總開關仍然是從原本 capture 的 `params` 讀，而不是從 `af_source` 讀：

[Controller/hardware_drivers/unified_driver.py (line 5689)](D:\\Provenance Project\\ImagingLibWatch\\Controller\\hardware_drivers\\unified_driver.py:5689)

所以即使 `focus_capture.use_liqud_af=true`，如果 standard capture 的 `use_liqud_af=false`，liquid lens AF 仍可能被關掉。這看起來像目前設計上的不一致或潛在 bug。

**HDR capture 的 autofocus**  
HDR capture 分兩層：

第一層：進入 HDR capture 前，`execute_template_point()` 會先跑一次正常 AF。  
如果有 `focus_capture`，這次 AF 也會用 `focus_capture`。

第二層：真正拍 HDR bracket 時， `_capture_and_fuse_hdr_v2()` 會看 `z_positions`。如果 `z_positions` 有明顯變化，而且 `use_autofocus` / `use_liqud_af` / `has_glass` 任一個啟用，就可能在不同 Z bucket 重新跑 autofocus。

相關位置：

[Controller/hardware_drivers/unified_driver.py (line 3318)](D:\\Provenance Project\\ImagingLibWatch\\Controller\\hardware_drivers\\unified_driver.py:3318)  
[Controller/hardware_drivers/unified_driver.py (line 3387)](D:\\Provenance Project\\ImagingLibWatch\\Controller\\hardware_drivers\\unified_driver.py:3387)

如果 HDR 有 varying `z_positions`，但 AF 沒被要求：

```
use_autofocus: false
use_liqud_af: false
has_glass: false
hdr_skip_autofocus: true
```

那程式會跳過 `focus_capture` autofocus，直接用 capture pose / z_positions 拍 bracket。

**一個要注意的 HDR + focus_capture 細節**  
目前 `_capture_and_fuse_hdr_v2()` 在每個 Z bucket 重新 AF 時，會複製整個 `hw_cfg`，因此 `focus_hardware` 也會一起被帶入。這代表如果 HDR capture 有 `focus_capture`，bucket AF 仍可能優先用固定的 focus_capture pose，而不是該 bucket 的 `z_target`。

這點我會標成 residual risk：如果你的期待是「HDR 每個 z_position 都在該 Z 上重新 AF」，目前有 `focus_capture` 時可能不完全符合這個語意。






#### af_modes=0(default)的詳細autofocus流程
```
請中文詳細解釋在af_mode_config定義af_modes=0(default)的詳細autofocus流程, zaber怎麼移動怎麼跟Keyence, light, liquid lens協同作autofocus?
```

以下是目前程式裡 `af_mode_id = 0` / `af_modes: 0` 的實際 default autofocus 流程。

主要依據：

[config/af_mode_config.yaml (line 1)](D:\\Provenance Project\\ImagingLibWatch\\config\\af_mode_config.yaml:1)  
[config/hardware_config.yaml (line 355)](D:\\Provenance Project\\ImagingLibWatch\\config\\hardware_config.yaml:355)  
[Controller/hardware_drivers/unified_driver.py (line 5403)](D:\\Provenance Project\\ImagingLibWatch\\Controller\\hardware_drivers\\unified_driver.py:5403)

**核心結論**  
`af_modes: 0` 本身沒有 override：

```
0:
  af_mode_name: default (simple af with liquid lens)
  overrides: {}
```

所以 `af_mode_id=0` 時，不會改寫任何 AF 設定，完全使用 `config/hardware_config.yaml` 裡的 default autofocus 設定。

目前 default 是兩階段：

1. Stage 1 粗對焦：Zaber + Keyence，使用 `coarse_af_option/autofocus_option = 1`，也就是 `Keyence MATLAB-like empirical AF`
2. Stage 2 細對焦：camera FPGA liquid lens autofocus，使用液態鏡頭掃 current，鎖住最佳 current 後拍照

default mode 的特性是「簡單模式」：沒有 best-of-N、沒有 baseline validation、沒有 liquid lens stage retry、沒有 software fallback。

**Phase 0：決定 AF 使用哪組參數**  
`execute_template_point(params)` 進來後，會先決定 autofocus 的參數來源：

- 如果沒有 `focus_capture` / `focus_hardware`：AF 使用這次 standard/HDR capture 自己的參數
- 如果有 `focus_capture`：AF 使用 `focus_capture` 的 camera、曝光、gain、燈光、cammag、use_autofocus、has_glass 等參數；最後真正拍照仍回到 standard/HDR capture 參數

接著會取得：

```
af_mode_id = 0
effective_config = self.config
z_af_option = autofocus.af_settings.autofocus_option
```

因為 mode 0 沒 override，所以 `z_af_option` default 是 `1`。之後又會看 camera-specific 設定：

```
hardware.autofocus.cameras.{camera}.coarse_af_option: 1
```

所以 macro/micro default 都是 Keyence option 1。

**Phase 1：Zaber 先移到 AF pose**  
程式會先讀目前 Zaber 位置：

```
stage_L_X
stage_L_Y
stage_L_Z
stage_R_X
stage_R_Z
```

然後用目標 AF pose 建立：

```
target_q = {
  x: af_base_x,
  y: af_base_y,
  z: af_base_z,
  rotate_x: af_rotate_x,
  rotate_z: af_rotate_z
}
```

這裡的 `af_base_*` 如果有 `focus_capture` 就來自 focus capture；否則來自 standard/HDR capture。

Zaber 移動不是直接斜線衝過去，而是走 `_plan_motion_segments()` 的安全路徑。一般 Front/Back 情況：

- 如果目標 Z 數值比較小，代表要先遠離工件，就先移 `stage_L_Z`
- 再 compound move：`stage_R_X`, `stage_R_Z`, `stage_L_X`, `stage_L_Y`
- 如果目標 Z 數值比較大，代表更靠近工件，就先 XY/rotation，再最後下 Z

側邊或高傾角 view 會走 dog-leg：

1. `stage_L_Z` 退到 safe height
2. 在安全高度移 XY / R_X / R_Z
3. `stage_L_Z` 再下降到目標 focus height

這段是避免 Z 軸、XY、旋轉一起走斜線時撞到工件。

**Phase 2：Keyence 粗對焦**  
default `z_af_option == 1`，所以進：

```
_af_keyence_matlab_like(...)
```

位置在 [unified_driver.py (line 1358)](D:\\Provenance Project\\ImagingLibWatch\\Controller\\hardware_drivers\\unified_driver.py:1358)。

這階段的重點是：Zaber 把 Keyence probe 移到可量測的位置，Keyence 讀 OUT1/OUT2/OUT3，程式用 empirical formula 算出應該拍照的 Z。

default 重要設定：

```
autofocus_option: 1
out1_min: -6.0
out1_max: 8.0
keyence1_measure_start_z: 20.0
keyence_read_roi_expan: 0
keyence_read_z_expan: 0
```

流程大概是：

1. 套用 Zaber `z_focus` motion profile
2. 計算 Keyence probe pose  
    也就是從 camera pose 加上 camera-to-Keyence offset；如果 angle-aware 開啟，還會根據 `stage_R_X` 角度修正 offset
3. Zaber 移到 Keyence pose
4. Keyence 讀值：
    - OUT1：主要高度/焦距訊號
    - OUT2：有玻璃時用來估玻璃厚度
    - OUT3：debug 或 secondary formula 用
5. 檢查 OUT1 是否在 window 內，目前 default 是 `-6.0 < OUT1 < 8.0`
6. 用 `hardware.keyence.matlab_like_focus` 的公式算 `target_z`
7. 如果成功，回傳 `best_z`
8. finally 裡會把 Zaber 從 Keyence probe pose 移回 camera reference pose

這個 default Keyence 粗對焦階段本身不開相機燈；它靠 Keyence sensor 讀值，不靠影像亮度。只有 Vision AF option 2/3 才會先 `apply_lights("ON", af_source)`。

**Keyence 算 Z 的公式**  
在 `_calc_matlab_like_keyence_z()` 裡，核心公式是線性的：

```
target_z = zbase + zdisp + sensor_coeff * OUT1
```

如果 `has_glass=True`，還會加玻璃修正：

```
glass_thick = OUT2 * out2_thickness_scale
glasscorrect = (glass_thick - glass_base_mm) * glass_mod_z_per_mm
target_z += glasscorrect_coefficient * glasscorrect
```

micro camera 會用 micro 的 `zdisp_glass` / `zdisp_no_glass`，macro camera 用 macro 的對應參數。算出的 `target_z` 還會檢查是否在 valid Z range 內。

**Phase 3：從 Keyence Z 轉成真正 capture / liquid AF pose**  
Keyence 回傳的是粗對焦結果 `best_z`。接著程式會算最後要移到哪裡做 liquid lens AF。

如果 Keyence AF 成功：

- 一般情況：
    
    ```
    target_z = best_z + parfocal_offset + Z_micro_offset
    ```
    
- 如果 angle-aware 有 corrected_camera_pose：
    
    ```
    target_x/y/z = corrected_camera_pose + angle-aware parfocal compensation + Z_micro_offset
    ```
    

這裡 `parfocal_offset` 來自 camera device 設定，`Z_micro_offset` 來自 capture/focus capture 設定。

然後 Zaber 走 `_do_final_approach(target_x, target_y, target_z)`：

- 先切 Zaber motion profile 到 `default`
- 如果 `final_approach.enable=false`，直接 X/Y/Z 移到 target
- 如果 enable，先移到 near point，再切 `capture` profile 做最後進刀
- 最後一定恢復 `default` profile

這一步結束時，stage 已經在準備拍照/液態鏡頭細對焦的位置。

**Phase 4：開燈、切相機、設曝光/gain**  
粗對焦和 Zaber final approach 完成後，才進 Phase 3 camera/light preparation：

```
apply_lights("ON")
self.cam.switch_camera(cam_alias)
ExposureTime = primary_exp
Gain = gain
```

燈光邏輯是：

```
ringlight1 -> ring_light_1 continuous
ringlight2 -> ring_light_2 continuous
spotlight1 > 0 -> spot_light_1 PWM 100
spotlight2 > 0 -> spot_light_2 PWM 100
sidelight > 0 -> side_light_1 PWM 100
```

注意：這裡先用 final capture 的燈光與相機設定。如果有 `focus_capture`，在 Phase 4 liquid AF 前會再切一次 focus light：

```
apply_lights("ON", af_source)
time.sleep(0.15)
```

這是為了讓 FPGA liquid AF 在 focus_capture 的照明條件下工作。

**Phase 5：Liquid lens FPGA 細對焦**  
default 每個 camera 在 `hardware.autofocus.cameras.{cam}` 裡大致是：

```
coarse_af_option: 1
liquidlens_autofocus: true
liquid_af:
  enable: 1
  af_method: "hardware"
  hardware_start_current: -20.0
  hardware_stop_current: 20.0
  hardware_frame_current: 1.0
  hardware_frame_count: 10/20/30
  fpga_af_validation_enable: false
  fpga_af_best_of_n_enable: false
  software_fallback_enable: false
  liquid_af_extensionN: 0
  stage_retry_enable: false
```

只要：

```
af_cam_af_enabled == True
af_device_capable == True
use_liqud_af == True
```

就會呼叫：

```
trigger_liquid_lens_hardware_autofocus(
  cam_alias=af_cam_alias,
  exposure_time=af_primary_exp,
  gain=af_gain
)
```

這裡會：

1. 切到 AF camera
2. 檢查 `liquid_af.enable == 1`
3. 使用目前燈光、曝光、gain
4. 交給 `CameraManager.trigger_hardware_liquid_af()` 做 FPGA AF
5. FPGA 在設定 current range 內掃 liquid lens current
6. 找到最佳 sharpness/current 後鎖住 liquid lens current
7. 成功時不做 Z compensation/reset，直接保留該 current 給最後拍照

default 如果 FPGA AF 失敗：

- 不跑 software fallback，因為 `software_fallback_enable: false`
- 不跑 stage retry，因為 `stage_retry_enable: false`
- 不用 manual cammag fallback，因為 `use_manual_cammag_on_af_failure: false`
- 最後把 liquid lens reset 到 `af_reset_current`，default 是 `0.0mA`

**Phase 6：manual X/Y/Zmod，再拍照**  
Liquid lens AF 結束後，程式會檢查 `Xmod/Ymod/Zmod`。這些不是粗對焦的一部分，而是「liquid lens AF 後、真正拍照前」的最後人工修正。

如果非零：

```
final_target = liquid_af_completed_pose + Xmod/Ymod/Zmod
```

Zaber 再移一次到 final target。

然後如果前面用過 `focus_capture`，會恢復 final capture 的燈光、camera、exposure、gain。

最後：

- 等 `capture_settle_time_s`
- 如果是 HDR，進 HDR bracket capture
- 否則 `self.cam.capture_image(exposure_time=primary_exp, gain=gain)`
- finally `apply_lights("OFF")`

**整體流程一句話**  
`af_mode=0` 是最直覺的兩段式 AF：Zaber 先安全移到 AF 點，Keyence 在固定 reference Z 量 OUT 值算出機械 Z 粗焦；Zaber 再把相機移到這個粗焦後的 capture pose；接著開燈、切相機、設定曝光/gain，讓 liquid lens FPGA 在目前畫面上掃電流找最清楚的 current；成功就鎖住 current 拍照，失敗就 reset 到 0mA 拍照或按設定 fallback。

9:27 AM



#### af_modes=1~8的autofocus
```
請中文詳細解釋在af_mode_config定義af_modes=1~8(default)跟default(af_modes=0)差別在哪裡? 會對autofocus造成甚麼影響
```

`af_mode_id=0` 是 baseline：不覆蓋任何設定，直接用 `config/hardware_config.yaml` 的 default AF。  
`af_mode_id=1~8` 則是在 AF 執行期間臨時覆蓋部分 config key，拍完後恢復原 config。

實作位置是 [Controller/hardware_drivers/unified_driver.py (line 5240)](D:\\Provenance Project\\ImagingLibWatch\\Controller\\hardware_drivers\\unified_driver.py:5240) 的 `_effective_config_for_af_mode()` / `_temporary_af_mode_config()`，設定來源是 [config/af_mode_config.yaml (line 1)](D:\\Provenance Project\\ImagingLibWatch\\config\\af_mode_config.yaml:1)。

**先講 default: mode 0**  
`af_modes: 0`：

```
af_mode_name: default (simple af with liquid lens)
overrides: {}
```

所以它不改任何東西。default 流程是：

1. Zaber 安全移動到 focus pose
2. Keyence 做粗對焦，default `autofocus_option/coarse_af_option = 1`
3. 使用 Keyence MATLAB-like empirical formula 算 `best_z`
4. Zaber 移到 capture/liquid lens AF pose
5. 開燈、切相機、設曝光/gain
6. liquid lens FPGA AF 掃 current
7. 成功就鎖住 current 拍照；失敗就 reset 到 `0mA`，default 不做 software fallback、不做 stage retry

也就是註解說的：

```
simple Keyence AF plus one liquid-lens AF attempt
```

**mode 1: lume fast af with liquid lens**  
主要改 Keyence 粗對焦公式，尤其是 `micro_cam` 的 lume 場景：

```
hardware.keyence.matlab_like_focus.micro_cam.formula_name: lume_linear_out1
hardware.keyence.matlab_like_focus.micro_cam.zdisp_glass: 8.1146
hardware.keyence.matlab_like_focus.micro_cam.out1_coefficient_glass: 0.9894
hardware.autofocus.af_settings.out1_min: -1.2
hardware.autofocus.af_settings.out1_max: 0.8
hardware.autofocus.af_settings.keyence_sample_settle_s: 0.08
```

影響：

- Keyence OUT1 有效範圍從 default `-6.0 ~ 8.0` 變成很窄的 `-1.2 ~ 0.8`
- micro + glass 的 Z 計算公式改成 lume 專用 calibration
- Keyence 讀值更嚴格，錯誤表面/離焦讀值比較容易被 reject
- liquid lens 還是 default simple AF，沒有啟用 best-of-N / stage retry

適合：lume 類型、已知 Keyence formula 需要特殊校正的點。  
代價：Keyence OUT1 window 窄，若位置偏太多可能更容易 fallback。

**mode 2: crown**  
這是比 default 強很多的 robust AF 模式，偏 crown 類難對焦點。

它主要啟用：

```
fpga_af_best_of_n_enable: true
fpga_af_best_of_n_stop_on_boundary_failure: true
hardware_frame_count: 20
stage_retry_enable: true
stage_retry_max_steps: 1
stage_retry_fast_scan: true
stage_retry_validate_success_sharpness: true
stage_retry_keep_best_sharpness: true
stage_retry_accept_best_effort_on_exhausted: true
keyence_read_roi_expan: 1
keyence_roi_confidence_enable: true
axis_retry_max_attempts: 9
fpga_af_validation_enable: true
fpga_af_baseline_validation_enable: true
fpga_af_baseline_fallback_enable: true
fpga_af_baseline_failure_triggers_stage_retry: true
reject_boundary_result: true
```

影響：

- Keyence 粗對焦不只讀單點，會允許 ROI expansion，讀不到或讀值不穩時有更多機會補救
- angle-aware Keyence retry 從 default 較少嘗試變成最多 9 次
- liquid lens 不再只相信一次 FPGA AF
- 如果 FPGA AF 結果落在 current 邊界，會更傾向判定為失敗
- AF 結果會做 validation / baseline validation
- 失敗時允許 Zaber stage retry 微調後再試
- 比 default 慢，但成功率與可靠度更高

適合：crown、反光強、表面曲率高、容易出現 liquid lens 邊界結果的點。

**mode 3: fast af**  
名字叫 fast，但實際上它的重點是：保留一些 robust config，關掉 liquid lens AF。

關鍵 override：

```
hardware.autofocus.cameras.micro_cam.liquidlens_autofocus: false
hardware.autofocus.cameras.macro_cam_1.liquidlens_autofocus: false
hardware.autofocus.cameras.micro_cam.liquid_af.enable: 0
hardware.autofocus.cameras.macro_cam_1.liquid_af.enable: 0
```

同時它也設定了一堆 validation/stage retry 相關 key，但因為 liquid lens 被關掉，這些 liquid lens 相關的 retry 多半不會真正跑。

影響：

- 會做 Keyence 粗對焦
- 不跑 liquid lens FPGA 細對焦
- 拍照時主要依賴 Keyence 算出的 Z
- 速度通常比有 liquid lens AF 快
- 細對焦精度可能比 mode 0/2/4/5 差，尤其對 micro/liquid lens 場景

適合：想省時間、或某些點 liquid lens AF 容易誤判時。

**mode 4: advance search fs**  
這是目前最 aggressive 的 advanced search 類模式。

它啟用：

```
fpga_af_best_of_n_count: 3
fpga_af_best_of_n_enable: true
hardware_frame_current: 0.5
hardware_frame_count: 40
boundary_margin_current_ma: 0.5
liquid_af_extensionN: 1
stage_retry_enable: true
stage_retry_mode: z_only
stage_retry_step_mm: 0.25
stage_retry_max_steps: 4
stage_retry_confirm_best_of_n: true
stage_retry_validate_success_sharpness: true
stage_retry_min_sharpness_gain: 1.01
fpga_af_repeat_check: true
fpga_af_repeat_tolerance_ma: 1.5
software_fallback_enable: true
steps: 81
settling_time_ms: 40.0
af_reset_settle_ms: 150.0
```

影響：

- liquid lens 掃描更細：`0.5mA` step、更多 frame
- 最多跑 best-of-N 3 次
- 檢查 repeat stability，兩次結果差太多會視為不穩
- 邊界容忍更嚴格
- AF 失敗時 Zaber 可以做最多 4 次 Z-only stage retry，每次 0.25mm
- 硬體 AF 失敗後允許 software sweep fallback
- 最慢，但最有補救能力

適合：非常難對焦、Z 高度不確定、liquid lens 常打到邊界的點。  
代價：時間明顯增加，硬體動作更多。

**mode 5: lume**  
這像是 mode 1 + robust liquid lens 的組合。

它有 mode 1 的 lume Keyence formula：

```
formula_name: lume_linear_out1
zdisp_glass: 8.1146
out1_coefficient_glass: 0.9894
out1_min: -1.2
out1_max: 0.8
```

又有 robust liquid lens：

```
fpga_af_best_of_n_count: 10
fpga_af_best_of_n_enable: true
stage_retry_enable: true
stage_retry_max_steps: 1
keyence_read_roi_expan: 1
axis_retry_max_attempts: 9
fpga_af_validation_enable: true
fpga_af_baseline_validation_enable: true
fpga_af_baseline_fallback_enable: true
```

影響：

- Keyence 粗對焦用 lume 專用公式
- OUT1 window 很嚴格
- liquid lens 最多 best-of-N 10 次，可靠但較慢
- 有 stage retry 和 baseline validation

適合：lume 點中又很容易 AF 不穩的情況。  
比 mode 1 慢，但比 mode 1 更可靠。

**mode 6: band side**  
只有一個 override：

```
hardware.autofocus.cameras.macro_cam_2.keyence1_measure_start_z: 65.0
```

影響：

- 只影響 `macro_cam_2`
- Keyence option 1 粗對焦時，固定 reference measurement Z 改成 `65.0`
- liquid lens 還是 default simple behavior
- 其他相機幾乎不受影響

適合：band side 這類 macro_cam_2 的側邊拍攝，Keyence 需要在不同 reference Z 才讀得到正確表面。

**mode 7: band front**  
也是針對 `macro_cam_2` 的 Keyence reference pose：

```
hardware.autofocus.cameras.macro_cam_2.keyence1_measure_start_y: 60.0
hardware.autofocus.cameras.macro_cam_2.keyence1_measure_start_z: 73.0
```

影響：

- macro_cam_2 的 Keyence 量測 reference Y/Z 都改掉
- 粗對焦時 Keyence probe 會用這組 band front 專用量測位置
- liquid lens 還是 default simple behavior

適合：band front，表面/幾何位置跟一般 watch front 不同，Keyence 需要換 Y/Z 才讀得準。

**mode 8: strap**  
只有：

```
hardware.autofocus.af_settings.keyence1_measure_start_z: 70.0
```

影響：

- 全域 default Keyence option 1 reference Z 改成 `70.0`
- 但注意：如果 camera-specific 有自己的 `keyence1_measure_start_z`，camera-specific 會優先
- liquid lens 還是 default simple behavior

適合：strap 類點位，整體 Z 高度基準跟一般 watch point 不同。

**簡表**

|mode|名稱|主要改變|對 AF 的效果|
|---|---|---|---|
|0|default|無 override|Keyence 粗 AF + 一次 liquid lens FPGA AF|
|1|lume fast|micro lume Keyence formula、窄 OUT1 window|Keyence 更針對 lume、更嚴格；liquid lens 仍簡單|
|2|crown|robust Keyence ROI + robust liquid lens + stage retry|更可靠、較慢|
|3|fast af|關閉 micro/macro1 liquid lens AF|更快，但少了細對焦|
|4|advance search fs|best-of-N、repeat check、Z retry、software fallback|最強補救能力，最慢|
|5|lume|lume formula + robust liquid lens|lume 專用且更穩，較慢|
|6|band side|macro_cam_2 Keyence start Z=65|band side 專用 Keyence reference Z|
|7|band front|macro_cam_2 Keyence start Y=60/Z=73|band front 專用 Keyence reference pose|
|8|strap|global Keyence start Z=70|strap 類點位改 Keyence reference Z|

最重要的實務判斷是：

- 想要最快：mode 3
- 想要一般 default：mode 0
- lume 點：mode 1 或 mode 5
- crown / 難對焦：mode 2
- 最難、願意花時間換成功率：mode 4
- macro_cam_2 band 幾何特殊：mode 6/7
- strap 高度基準特殊：mode 8



#### af_modes=0(default)詳細流程圖
```
請將af_modes=0(default)的詳細autofocus流程畫成詳細流程圖
```

下面是 `af_modes=0` 的 default autofocus 詳細流程圖。這個 mode 本身沒有 override，實際走 `hardware_config.yaml` 的 default：Keyence 粗對焦 + 一次 liquid lens FPGA 細對焦。

```
flowchart TD
    A["execute_template_point(params) 開始"] --> B["Safety check: check_safety_lock()"]
    B --> C["解析 capture / focus 參數"]

    C --> C1{"有 focus_capture / focus_hardware ?"}
    C1 -- "有" --> C2["AF 參數來源 = focus_capture<br/>cam_id, x/y/z, lights, exptimes, gain, cammag,<br/>use_autofocus, has_glass"]
    C1 -- "沒有" --> C3["AF 參數來源 = 目前 standard/HDR capture"]
    C2 --> D
    C3 --> D

    D["讀取 af_mode_id"] --> E{"af_mode_id == 0 ?"}
    E -- "是" --> F["不套用 af_mode_config overrides<br/>使用 hardware_config.yaml default"]
    F --> G["決定 coarse AF strategy<br/>autofocus_option / coarse_af_option = 1"]
    G --> H["Stage 1: Zaber 移動到 AF pose"]

    H --> H1["讀目前 Zaber pose<br/>stage_L_X/Y/Z, stage_R_X/Z"]
    H1 --> H2["建立 target_q<br/>x/y/z = AF pose<br/>rotate_x/rotate_z = AF pose"]
    H2 --> H3["_plan_motion_segments()<br/>產生安全移動路徑"]

    H3 --> H4{"一般 Front/Back ?"}
    H4 -- "是, 目標 Z 較高<br/>數值較小" --> H5["先 Z_ONLY 退 Z<br/>再 COMPOUND 移 XY + R_X/R_Z"]
    H4 -- "是, 目標 Z 較低<br/>數值較大" --> H6["先 COMPOUND 移 XY + R_X/R_Z<br/>再 Z_ONLY 下 Z"]
    H4 -- "Side/Crown/高傾角" --> H7["Dog-leg path<br/>1. Z 退到 safe height<br/>2. 高空移 XY + rotate<br/>3. Z 下探到 AF height"]

    H5 --> I
    H6 --> I
    H7 --> I

    I["執行 motion segments"] --> I1["Z_ONLY: zaber.move_axis(stage_L_Z)"]
    I --> I2["COMPOUND: zaber.move_axes_absolute()"]
    I1 --> J
    I2 --> J

    J["Stage 1 粗對焦: Keyence MATLAB-like AF"] --> K{"use_autofocus 或 has_glass ?"}
    K -- "否" --> K0["跳過 Keyence Z-AF<br/>target_z = base_z + Z_micro_offset"]
    K -- "是" --> L["_af_keyence_matlab_like()"]

    L --> L1["套用 Zaber z_focus motion profile"]
    L1 --> L2["計算 Keyence probe pose<br/>camera pose + camera_to_keyence_offset"]
    L2 --> L3{"angle-aware enabled ?"}
    L3 -- "是" --> L4["根據 R_X 角度修正<br/>Keyence offset / focus axis / pose compensation"]
    L3 -- "否" --> L5["使用固定 Keyence XY offset"]
    L4 --> L6
    L5 --> L6

    L6["Zaber 移到 Keyence probe pose"] --> L7["讀 Keyence OUT1 / OUT2 / OUT3"]
    L7 --> L8{"OUT1 在有效範圍 ?<br/>default: -6.0 到 8.0"}
    L8 -- "否" --> L9["Keyence AF failed<br/>fallback_z = template z_default 或 base_z"]
    L8 -- "是" --> L10{"has_glass ?"}

    L10 -- "是" --> L11["用 OUT2 算 glass_thick<br/>計算 glasscorrect"]
    L10 -- "否" --> L12["glass_thick = 0<br/>glasscorrect = 0"]
    L11 --> L13
    L12 --> L13

    L13["用 empirical formula 算 target_z<br/>target_z = zbase + zdisp + coeff*OUT1 + glass correction"] --> L14{"target_z 在 valid Z range ?"}
    L14 -- "否" --> L9
    L14 -- "是" --> L15["Keyence AF success<br/>best_z = target_z"]

    L9 --> L16["Keyence cleanup<br/>Zaber 回 camera reference pose<br/>motion profile 恢復 default"]
    L15 --> L16

    L16 --> M["計算 liquid lens AF 前的 capture pose"]
    K0 --> M

    M --> M1{"Keyence angle-aware 有 corrected_camera_pose ?"}
    M1 -- "有" --> M2["target_x/y/z = corrected_camera_pose<br/>+ parfocal compensation<br/>+ Z_micro_offset"]
    M1 -- "沒有, Keyence 成功" --> M3["target_x = base_x<br/>target_y = base_y<br/>target_z = best_z + parfocal_offset + Z_micro_offset"]
    M1 -- "Keyence 失敗 fallback" --> M4["target_z = z_default<br/>跳過 parfocal/Z_micro_offset 前置補償"]
    M2 --> N
    M3 --> N
    M4 --> N

    N["Zaber final approach 到 liquid lens AF pose"] --> N1{"final_approach.enable ?"}
    N1 -- "否" --> N2["直接 move X/Y/Z 到 target"]
    N1 -- "是" --> N3["先 default profile 移到 near point<br/>再 capture profile 慢速進到 target<br/>最後恢復 default profile"]
    N2 --> O
    N3 --> O

    O["Phase 3: 相機與燈光準備"] --> O1["apply_lights(ON)<br/>使用 final capture 燈光"]
    O1 --> O2["switch_camera(final capture camera)"]
    O2 --> O3["設定 final capture ExposureTime / Gain"]

    O3 --> P["Stage 2: Liquid lens FPGA AF"]
    P --> P1{"有 focus_capture ?"}
    P1 -- "有" --> P2["切成 focus_capture 燈光<br/>等待 0.15s 穩定"]
    P1 -- "沒有" --> P3["沿用目前 capture 燈光"]
    P2 --> Q
    P3 --> Q

    Q{"liquid lens AF enabled ?<br/>liquidlens_autofocus=true<br/>liquid_af.enable=1<br/>use_liqud_af=true"}
    Q -- "否, 但有 cammag" --> Q1["套 manual cammag<br/>不跑 FPGA AF"]
    Q -- "否, 無 cammag" --> Q2["不做 liquid lens AF"]
    Q -- "是" --> R["trigger_liquid_lens_hardware_autofocus()"]

    R --> R1["switch_camera(AF camera)"]
    R1 --> R2["確認 liquid_af.enable == 1"]
    R2 --> R3["使用 AF exposure / gain"]
    R3 --> R4["FPGA 掃 liquid lens current<br/>default range 通常 -20mA 到 20mA"]
    R4 --> R5["相機/FPGA 計算 sharpness<br/>尋找最佳 current"]
    R5 --> R6{"FPGA AF success ?"}

    R6 -- "成功" --> R7["讀回 locked_current<br/>保持此 liquid lens current<br/>不做 Z compensation/reset"]
    R6 -- "失敗" --> R8{"software_fallback_enable ?<br/>default false"}
    R8 -- "true, 非 mode 0 default" --> R9["跑 software sweep fallback"]
    R8 -- "false" --> R10["reset liquid lens current<br/>default af_reset_current = 0mA"]

    R7 --> S
    R9 --> S
    R10 --> S
    Q1 --> S
    Q2 --> S

    S["Liquid AF 完成後"] --> S1{"有 Xmod/Ymod/Zmod ?"}
    S1 -- "有" --> S2["Zaber 再移到<br/>liquid pose + manual mods"]
    S1 -- "沒有" --> S3["保持目前 pose"]
    S2 --> T
    S3 --> T

    T{"之前使用 focus_capture ?"}
    T -- "有" --> T1["恢復 final capture 燈光<br/>切回 final capture camera<br/>恢復 final exposure/gain"]
    T -- "沒有" --> T2["保持 final capture 設定"]
    T1 --> U
    T2 --> U

    U["等待 capture_settle_time_s"] --> V{"HDR capture ?"}
    V -- "是" --> V1["拍 HDR bracket / fuse"]
    V -- "否" --> V2["capture_image(primary_exp, gain)"]
    V1 --> W["apply_lights(OFF)"]
    V2 --> W
    W --> X["回傳 image, final_z, glass_thick"]
```

簡短抓重點：

- `af_mode=0` 不改設定，只用 default。
- 粗對焦靠 Zaber + Keyence 算出機械 Z。
- 細對焦靠 liquid lens FPGA 掃 current。
- Keyence 階段主要控制 Zaber 的機械高度。
- Liquid lens 階段通常不再移 Z，而是鎖定最佳鏡頭電流。
- 最後拍照前才套 `Xmod/Ymod/Zmod`，然後恢復 final capture 的燈光、曝光與相機設定。