


|                                                    |     |
| -------------------------------------------------- | --- |
| [[#### 三個相機 (Cameras) 的對焦計畫]]                      |     |
| [[#### hardware_config的參數autofocus_camera方案(0-4)]] |     |
| [[#### 「粗對焦 (Coarse Focus)」與「細對焦 (Fine Focus)」]]   |     |
|                                                    |     |
|                                                    |     |
|                                                    |     |
|                                                    |     |


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




```
請幫我把hardware_config.yaml裡面原本放在cameras關於autofocus的部分統一放在autofocus之下, 而在autofocus之下先寫跟對所有cameras(譬如autofocus_point....)的設定, 之後再分別分成三個cameraszz放各自專屬的設定. 也註定三個camera都要有完整的粗對焦跟細對焦的個別的parameters. 另外如果hardware_config.yaml 對focus parameters做了修正, 也要對相對應的code/functions作相對應的修正
```