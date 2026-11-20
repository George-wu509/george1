
|                                            |     |
| ------------------------------------------ | --- |
| [[#### matlab functionality/ matlab code]] |     |
| [[#### Matlab code跟python code比對]]         |     |
| [[#### Matlab的template參數意義]]               |     |
|                                            |     |
|                                            |     |

#### matlab functionality/ matlab code
```
這是一些matlab m files. 有已經整合進去ImagingLibWatch project或準備要整合進去ImagingLibWatch project. 請幫我中文詳細解釋這些code並檢查ImagingLibWatch project裡面有哪些python files是這些matlab改寫的或整合的? 並建議其他的matlab files要如何整合進去ImagingLibWatch project? 請以提供30年tech lead提供建議
```

這是一份非常有價值的 Legacy Code (遺產代碼) 分析請求。作為一名 Tech Lead，我會先詳細拆解這些 MATLAB 程式碼的邏輯，然後與目前的 Python 架構進行比對，最後給出具體的遷移 (Migration) 與整合建議。

這些 MATLAB code 代表了這個專案的 **核心領域知識 (Domain Knowledge)**，特別是在 **自動對焦邏輯 (Auto-focus heuristics)**、**HDR 成像** 與 **座標換算** 上。

---

### 第一部分：MATLAB 代碼詳細解析

#### 1. `watch_cap.m` (HDR 影像擷取核心)

- **功能**：這是最底層的相機驅動封裝。它不只是拍照，還包含了 **HDR (高動態範圍)** 合成。
    
- **邏輯**：
    
    - 接受 `exptimes` (一組曝光時間陣列)。
        
    - 迴圈拍攝多張不同曝光的照片 (`microcaps{i}`).
        
    - 支援 `adapthisteq` (自適應直方圖均衡化) 來增強對比。
        
    - 最後使用 `makehdr` (MATLAB 內建函式) 合成 HDR，並用 `localtonemap` 進行色調映射 (Tone mapping)。
        
- **關鍵價值**：手錶金屬表面反光嚴重，單次曝光通常看不清楚細節，這個 HDR 流程是成像品質的關鍵。
    

#### 2. `watch_detail_point_collect_double.m` (互動式採樣流程)

- **功能**：這是一個「人機互動」的腳本。
    
- **流程**：
    
    1. 顯示一張廣角 (Tele) 照片。
        
    2. 讓使用者用滑鼠點擊 (`drawpoint`) 一個感興趣的點。
        
    3. 將點擊的像素座標轉換為 Zaber 馬達的物理座標 (`watch_tele2detail`)。
        
    4. 移動馬達，使用雷射找高度 (`watch_findsurface3`)。
        
    5. 控制燈光 (關 Ring light, 開 Top light)。
        
    6. 呼叫 `watch_camcap` 進行 HDR 拍照。
        

#### 3. `watch_distance_point_collect.m` (距離/厚度量測流程)

- **功能**：量測兩個點之間的 Z 軸高度差（例如量測錶鏡厚度或指針高度）。
    
- **流程**：使用者點兩個點 -> 系統移動去量測點 1 高度 -> 移動去量測點 2 高度 -> 計算差值。
    
- **邏輯細節**：包含了很多錯誤處理 (Error Handling)，例如讀數 `>8` 或 `<-6` 會被視為無效並重試。
    

#### 4. `watch_findsurface*.m` 系列 (自動對焦/尋找表面核心)

這系列是 **最有價值** 的硬體控制邏輯，包含了處理反光、玻璃干擾的經驗法則 (Heuristics)。

- **`watch_findsurface.m`**: 基礎版。移動到 `Zbase`，讀取雷射，計算目標 `Znew`。
    
- **`watch_findsurface_direct.m`**: 用於 Top Cam。包含 `while` 迴圈，如果讀數不合理 (Out of range)，會自動調整 `Zbase` (每次下降 8mm) 重試，這是典型的 **Search Strategy**。
    
- **`watch_findsurface2.m` & `3.m`**: 用於 Micro Cam。邏輯更複雜，考慮了 **玻璃折射率修正** (`glasscorrect`)。如果雷射打在玻璃上 (`hasglass==1`)，它會根據玻璃厚度修正對焦距離。包含了多層 `if-else` 的 fallback 機制 (Zbase1 失敗試 Zbase2...)。
    

#### 5. `watch_glass_thickness.m` (玻璃厚度量測)

- **功能**：在三個特定的點 (形成三角形) 量測玻璃厚度，取平均值。
    
- **價值**：用於建立手錶的幾何模型，避免鏡頭撞到玻璃。
    

#### 6. `watch_micro_detail_point_collect_triple2.m` (多重光照採集)

- **功能**：全自動流程。
    
- **流程**：移動 -> 對焦 -> **拍三種不同設定的照片** (正常光、高對比光用於看刮痕/Bumps、另一種光)。
    
- **價值**：這是 Dataset 建立的關鍵，透過不同光照來提取不同特徵 (紋理 vs 形狀)。
    

#### 7. `watch_name256.m` (UUID 生成)

- **功能**：生成 32 碼 Hex 字串。
    

#### 8. `watch_watchshift_tele2.m` (影像對位/指紋識別)

- **功能**：計算兩張照片的位移 (Shift) 和旋轉 (Rotation)。
    
- **技術**：使用特徵點匹配 (`watch_tele_fing`，推測是 SIFT/SURF 類演算法)。
    
- **邏輯**：過濾掉錯誤的匹配點 (RANSAC 概念的簡化版)，計算向量角度差來求旋轉量。
    

---

### 第二部分：Python 專案 (ImagingLibWatch) 目前整合狀態

根據之前的對話紀錄，目前的 Python 架構 (`WorkflowManager`, `HardwareManager`) 整合狀態如下：

| **MATLAB 功能**            | **對應 Python 檔案**                                     | **狀態評估 (30-year Tech Lead View)**                                          |
| ------------------------ | ---------------------------------------------------- | -------------------------------------------------------------------------- |
| **Zaber 移動**             | `Controller/hardware_managers.py` (ZaberManager)     | **已完成**。基礎移動功能已具備。                                                         |
| **相機擷取**                 | `Controller/hardware_managers.py` (CameraManager)    | **部分完成**。目前只有單張擷取 (`capture_image`)，**缺 HDR 邏輯**。                          |
| **UUID 生成**              | `uuid` (Python 內建庫)                                  | **已完成**。Python 直接用 `uuid.uuid4()` 取代。                                      |
| **流程控制** (`collect`)     | `Controller/plans/*.yaml` + `WorkflowManager`        | **架構已完成**。用 YAML/DB 定義步驟取代了 MATLAB 的寫死腳本。                                  |
| **自動對焦** (`findsurface`) | `Controller/controller_server.py` (`run_auto_focus`) | **過於簡化**。目前的 Python 版只是簡單的 `Z_base + dist`，**缺少 MATLAB 版的 Retry 機制與玻璃修正**。 |
| **影像對位** (`watchshift`)  | `tasks/api_servers/server_features.py` (推測)          | **待確認**。Python 生態系有強大的 OpenCV，通常用 `cv2.findHomography` 或 `ORB` 取代。         |
| **座標轉換** (`tele2detail`) | **缺失**                                               | 目前沒看到 Python 版有實作「廣角像素 -> 馬達座標」的校正矩陣邏輯。                                    |

---

### 第三部分：整合建議與實作路線圖

作為 Tech Lead，我建議不要直接翻譯 MATLAB code，而是將其邏輯 **重構 (Refactor)** 進現有的 Python 物件導向架構中。

#### 1. 優先整合：增強型自動對焦 (Robust Auto-Focus)

目前的 Python `run_auto_focus` 太天真 (Naive)。你需要將 `watch_findsurface2.m` 的 Retry 與 Fallback 邏輯移植過去。

建議做法：

在 Controller/hardware_managers.py 的 ZaberManager 或 IOManager 之上，建立一個高階控制層，例如 FocusController。

Python

```
# 建議新增 Controller/focus_controller.py
class FocusController:
    def __init__(self, zaber_mgr, io_mgr):
        self.zaber = zaber_mgr
        self.io = io_mgr
        # 定義搜尋高度 (來自 MATLAB Zbase1~6)
        self.search_heights = [48, 40, 32, 24, 16, 8] 
        self.glass_correction = -0.4

    def run_robust_autofocus(self):
        """移植 watch_findsurface2.m 的邏輯"""
        for z_base in self.search_heights:
            self.zaber.move_axis('stage_L_Z', z_base)
            time.sleep(0.1)
            dist = self.io.read_keyence()
            
            # MATLAB 邏輯: if sensordist<2 && sensordist>-6
            if -6 < dist < 2: 
                # Good read, calculate target
                # 這裡需要移植 hasglass 的判斷邏輯
                target_z = self._calculate_target(z_base, dist)
                if 0 < target_z < 60:
                    self.zaber.move_axis('stage_L_Z', target_z)
                    return target_z
                
        raise Exception("Auto-focus failed after exhaustive search")
```

#### 2. 核心整合：HDR 成像流水線

`watch_cap.m` 的 HDR 是品質關鍵。Python 的 `CameraManager` 目前只回傳單張 Raw。

建議做法：

修改 Controller/hardware_managers.py 中的 CameraManager，或在 tasks 中新增一個 Pre-processing task。

推薦使用 OpenCV 的 Mertens Fusion (不需要相機響應曲線，效果好且快)。

Python

```
# 修改 CameraManager 或新增 ImageProcessor
import cv2
import numpy as np

def capture_hdr(self, exposures=[5000, 15000, 40000]):
    images = []
    for exp in exposures:
        img = self.capture_image(exposure=exp) # 基礎拍攝
        images.append(img)
    
    # OpenCV HDR Merge (Mertens) - 取代 MATLAB makehdr
    merge_mertens = cv2.createMergeMertens()
    hdr = merge_mertens.process(images)
    
    # 轉換回 8-bit (0-255)
    res_8bit = np.clip(hdr * 255, 0, 255).astype('uint8')
    return res_8bit
```

#### 3. 流程整合：座標轉換 (Calibration Matrix)

`watch_tele2detail` (點擊像素 -> 馬達座標) 依賴一個校正矩陣（Affine Transformation Matrix）。這在 Python 專案中目前缺失。

**建議做法**：

1. 在 `config/system_config.yaml` 中新增 `calibration` 區塊，儲存矩陣參數。
    
2. 在 `WorkflowManager` 或前端 UI 中實作轉換函式：
    
    Python
    
    ```
    def pixel_to_stage_coords(pixel_x, pixel_y):
        # 讀取 config 中的矩陣 M
        # [StageX, StageY] = M * [PixelX, PixelY, 1]
        pass
    ```
    

#### 4. 遺產廢棄：互動式腳本

`watch_detail_point_collect_double.m` 這種「跳出視窗讓使用者點」的邏輯，**不應該** 寫在 Backend Engine (WorkflowManager) 裡。

**建議做法**：

- **前端 (Client)**：在 PySide6 或 Web UI 顯示即時影像，捕捉滑鼠點擊事件 (x, y)。
    
- **通訊**：Client 發送指令 `{"action": "move_to_pixel", "pixel": [x, y]}` 給 Server。
    
- **後端 (Server)**：Server 收到後，呼叫上面的座標轉換函式，再驅動馬達。
    

### 總結

目前的 `ImagingLibWatch` 已經搭建好了骨架 (Architecture)，但肌肉 (Logic) 還不夠強壯。

1. **已整合**：基礎控制、檔案管理、UUID。
    
2. **需重寫 (高優先級)**：
    
    - `watch_findsurface*.m` -> **`FocusController` (Python Class)**: 這是硬體穩定運作的靈魂。
        
    - `watch_cap.m` -> **`CameraManager.capture_hdr`**: 這是影像品質的靈魂。
        
3. **需重寫 (中優先級)**：
    
    - `watch_watchshift_tele2.m` -> **`OpenCV Feature Matching`**: 用於校正手錶擺放誤差。
        
4. **架構調整**：
    
    - MATLAB 的 `collect` 腳本邏輯應轉化為 DynamoDB 的 `Templates` 資料。
        

作為 Tech Lead，我會建議你先從 **HDR** 和 **Auto-Focus** 這兩個功能著手，因為它們直接影響「能不能拍出清楚的照片」。互動部分則留給 UI 層處理。


#### Matlab code跟python code比對
```
也請整理一下列出的matlab code是對應到哪些我已有的python code並中文解釋
```
這是一項非常關鍵的盤點工作。將過去實驗室或初代的 MATLAB 腳本，映射到你現在這套 V2 Moonlight Python 物件導向架構中，是確保量產不翻車的必經之路。

你現在的 Python 架構設計得很好，把過去 MATLAB 裡「寫死（Hardcoded）」的流程，成功抽象化成了 Controller、Managers 和 YAML 驅動的 Workflow。

以下我幫你整理的「**MATLAB 腳本 vs. Python 模組映射與狀態解析表**」，並針對每一塊提供詳細的 Tech Lead 視角解說：

### 🗺️ MATLAB 到 Python 的架構映射總表

|**MATLAB 原始腳本**|**核心功能描述**|**對應的 Python 模組 / 類別**|**目前移植狀態 (Status)**|
|---|---|---|---|
|`watch_cap.m`|影像擷取、多曝光 HDR 合成、CLAHE 對比增強與色調映射|`Controller.hardware_managers.CameraManager`|🟡 **部分完成** (已有 HDR，缺 CLAHE 增強)|
|`watch_findsurface*.m` (系列)|利用 Keyence 雷射進行 Z 軸階梯式搜尋、玻璃折射補償、閉迴圈對焦|`Controller.focus_controller.FocusController`<br><br>  <br><br>`Controller.hardware_drivers.unified_driver`|🟡 **部分完成** (已有基本雷射讀取，缺完整階梯搜尋與玻璃折射公式)|
|`watch_glass_thickness.m`|利用雷射在三個點測量錶面玻璃厚度並取平均，用於後續 Z 軸補償|無 (需實作於 `unified_driver.py`)|🔴 **完全缺失** (需新增)|
|`watch_gain_adjust4.m`|根據畫面亮度 (Mean Intensity) 動態計算並調整相機 Gain/Exposure|無 (需實作於 `CameraManager`)|🔴 **完全缺失** (需新增自動曝光演算法)|
|`watch_*_point_collect_*.m` (系列)|**檢測 SOP 腳本**：移動到位 -> 測高 -> 切換光源 -> 拍多張不同設定的影像|`Controller.controller_client.AutoInspector`<br><br>  <br><br>`core.workflow_manager` (搭配 YAML)|🟢 **架構升級** (Python 做法更優秀，已將硬編碼轉為 YAML Template 驅動)|

---

### 🔍 模組功能詳細中文解釋與架構對比

#### 1. 光學成像與增強 (Imaging & Enhancement)

- **MATLAB**: `watch_cap.m`
    
- **Python**: `CameraManager` 內的 `capture_image()` 與 `capture_hdr()`。
    
- **Tech Lead 解析**：
    
    MATLAB 版本不只是單純拍照，它還包含了 `adapthisteq` (限制對比度自適應直方圖均衡化，CLAHE) 以及 `localtonemap`。手錶的錶盤 (Dial) 與金屬指針 (Hands) 反光極強，如果只做標準的 OpenCV Mertens HDR，暗部細節（如微小灰塵或指針邊緣的毛刺）會出不來。
    
    👉 **你的 Python 程式碼目前只有 `cv2.createMergeMertens()`，必須把 CLAHE 補回 `CameraManager` 裡，否則送給後端 UNet / SAM 的影像品質會不如預期。**
    

#### 2. 精密對焦與光學補償 (Autofocus & Physics Compensation)

- **MATLAB**: `watch_findsurface.m`, `watch_findsurface2.m`, `watch_findsurface3.m`, `watch_findsurface_direct.m`
    
- **Python**: `UnifiedHardwareDriver.run_robust_autofocus()` 與 `focus_controller.py`。
    
- **Tech Lead 解析**：
    
    這是整套 AOI 系統最難的地方。MATLAB 代碼裡用了「階梯式降高搜尋」(從 48mm 降到 8mm) 來防止感測器抓到錯誤的反光面。更重要的是 `glasscorrect = (glassthick1 - glassbase) * glassmod` 這段折射補償物理公式 (`glassmod = -0.4`)。
    
    👉 **你目前的 Python 已經搭好了 `FocusController` 的骨架，但裡面缺乏完整的「階梯式 While Loop」以及「藍寶石玻璃厚度帶來的折射補償」，這會導致你的 Micro Camera 永遠對不準機芯內部。**
    

#### 3. 玻璃厚度測量 (Glass Thickness Calibration)

- **MATLAB**: `watch_glass_thickness.m`
    
- **Python**: 目前沒有這段邏輯。
    
- **Tech Lead 解析**：
    
    MATLAB 腳本透過在錶面上打三個點（構成正三角形），計算平均玻璃厚度（排除表面弧度或放置不平的誤差），這個數值會餵給前面的對焦補償使用。
    
    👉 **這是一項前置校正任務。你需要在 Python 的 `UnifiedDriver` 中新增一個 `measure_glass_thickness()` function，並在每一隻手錶的檢測 Routine 一開始就先呼叫它。**
    

#### 4. 自動曝光控制 (Auto Exposure / Auto Gain)

- **MATLAB**: `watch_gain_adjust4.m`
    
- **Python**: 目前沒有這段邏輯。
    
- **Tech Lead 解析**：
    
    雖然你有光控系統，但不同型號、甚至同型號但不同顏色的手錶（例如黑水鬼 vs. 綠水鬼），反光率差異極大。MATLAB 腳本會先拍一張，計算畫面中心的亮度 (Mean Intensity)，如果太暗或太亮，會自動算出修正係數 (`lightcorrection`) 來動態調整 Gain 重新拍攝。
    
    👉 **AOI 系統不能依賴寫死 (Hardcoded) 的曝光時間。你必須在 `CameraManager` 加入一個 `auto_adjust_exposure_gain(target_intensity=115)` 的閉迴圈控制功能。**
    

#### 5. 檢測流程邏輯 (Workflow & Automation SOP)

- **MATLAB**: `watch_detail_point_collect_double.m`, `watch_micro_detail_point_collect_triple2.m`, `watch_distance_point_collect.m`
    
- **Python**: `core.workflow_manager` (搭配 `main_cli.py` 讀取的 `DB/templates_import/*.yaml`) 與 `AutoInspector`。
    
- **Tech Lead 解析**：
    
    這是你 **Python 系統大勝 MATLAB** 的地方！過去 MATLAB 把移動座標 (X, Y)、開關幾號燈、要拍幾次全部寫死在 `.m` 檔裡，這根本無法維護。你現在的 V2 系統將這些流程抽離成了 YAML Template（例如你提到的 `pipeline_sim_map.yaml` 和例行檢查任務）。
    
    👉 **不用把這幾支 MATLAB 腳本直譯成 Python 程式碼，而是把它們的「執行邏輯」（先開燈 A 拍一張，再開燈 B 拍一張）寫成你的 YAML Template 任務節點即可。**
    

---

### 下一步建議

從這份健檢清單來看，底層的硬體通訊你都搞定了，現在欠缺的是**「演算法與物理補償」**的最後一哩路。

這幾項缺失中，**「對焦階梯搜尋與玻璃補償 (watch_findsurface)」** 以及 **「CLAHE 影像對比增強 (watch_cap)」** 對於你後續的 AI 影像辨識影響最大。

需要我先幫你把 `CameraManager` 裡的 `capture_enhanced_hdr()` (包含 MATLAB 的 CLAHE 邏輯) 寫出來，還是先處理 `FocusController` 的玻璃厚度補償演算法？




#### Matlab的template參數意義
```
在matlab code裡面用mat裡面儲存的watchtemplate去控制拍照到image的所有流程的參數, watchtemplate裡面的watchvie有分成'Front','Back','Open back', 'Open back','Strap Right Side'五個view, 在這五個view裡面還有分'glasspoint','toppoint','sidepoint','macropoint','micropoint'.是那些view從不同的point拍照. 這些point有的參數包括. 請依照那些matlab code跟新上傳的matlab App file中文詳細解釋每個參數的意義跟用處以及用在那些functions

glasspoint: 
etchID, X, Y, Z, yaw, pitch, roll, camused, cammag, microlight, sidelight, ringlight, topbarlight, exptimes, gainlevel, trim1, contcorrect, rangecorrect, rangecomp, enhancecomp, adjustsat, Zmod, glassthick, X0, Y0, glassflat, etchID2

toppoint:
多了topID, topID2

sidepoint:
多了sideID, sideID2, Xmod, Ymod

macropoint跟micropoint:
多了pointID, pointID2, parttype, detailedparttype, heightmod, EntryTime, EntryLocation
```

這是一個非常好的問題。`watchtemplate` 這個 `struct` 在原本的 MATLAB 系統中扮演著 **「檢測配方 (Recipe / SOP)」** 的核心角色。它定義了每一隻手錶、每一個角度 (View)、每一個特徵點 (Point) 該「怎麼走到、怎麼打光、怎麼拍、怎麼修圖」。

在移植到 Python 時，這些參數就是你的 `Template YAML` 或資料庫 schema 的核心欄位。

以下我將針對你列出的 `glasspoint`, `toppoint`, `sidepoint`, `macropoint`, `micropoint` 內的參數，進行詳細的中文技術解析與程式碼對應分析。

---

### 1. 核心參數詳解 (通用於大部分 Points)

這些參數控制著機械運動、相機成像與光源，是自動化檢測的基礎。

#### A. 運動控制參數 (Positioning)

- **`X`, `Y`**:
    
    - **意義**: 該特徵點在載台上的絕對座標 (mm)。
        
    - **用處**: 驅動 Zaber 馬達將手錶移動到相機視野中心。
        
    - **對應 Code**: 用於 `watch_zaber_move(connection, deviceList, X, Y, Z)`。
        
- **`Z`**:
    
    - **意義**: 該特徵點的預設對焦高度 (mm)。
        
    - **用處**: 作為 `watch_findsurface` (自動對焦) 的起始搜尋高度 (`Zbase`)。
        
    - **對應 Code**: `watch_findsurface*.m` 中的 `Zbase` 初始值。
        
- **`yaw` (Z-Rot), `pitch` (X-Rot), `roll`**:
    
    - **意義**: 旋轉軸的角度 (度)。`yaw` 是載台旋轉 (Stage R-Z)，`pitch` 是懸臂旋轉 (Stage R-X)。
        
    - **用處**: 讓相機與手錶特徵面保持垂直或特定觀測角度。
        
    - **對應 Code**: `watch_zaber_move`，對應到你在 `hardware_config.yaml` 裡的 `stage_R_Z` 與 `stage_R_X`。
        
- **`X0`, `Y0`**:
    
    - **意義**: 參考座標或該 View 的原點座標。
        
    - **用處**: 用於計算相對位移，或者在 `watch_glass_thickness.m` 中作為測量玻璃厚度三角形的基準中心點。
        

#### B. 硬體選擇參數 (Hardware Selection)

- **`camused`**:
    
    - **意義**: 指定使用哪一顆相機。
        
    - **值域**: 1=Macro (Tele), 2=Micro, 3=Top, 4=Side (依據 `watch_cap.m` 的邏輯)。
        
    - **對應 Code**: `watch_camcap.m` 開頭的 `if camID==1...` 判斷邏輯。
        
- **`cammag`**:
    
    - **意義**: 相機倍率 (Magnification) 或液態鏡頭的屈光度設定。
        
    - **用處**: 紀錄該次拍攝的光學倍率，可能用於後續計算 "每像素代表多少mm (um/pixel)"。
        

#### C. 光源控制參數 (Lighting)

- **`microlight`, `sidelight`, `ringlight`, `topbarlight`**:
    
    - **意義**: 各個光源的開關狀態 (0/1) 或強度值 (0-255)。
        
    - **用處**: 決定拍攝當下的打光策略（例如：拍表面刮痕要開側光，拍機芯要開環形光）。
        
    - **對應 Code**: `watch_light_control2` (未上傳但被呼叫) 或 `watch_detail_point_collect_*.m` 中設定 `lightnum` 與 `lighton` 的邏輯。
        

#### D. 影像擷取與增強參數 (Image Capture & Enhancement)

- **`exptimes`**:
    
    - **意義**: 曝光時間陣列 (Array)。例如 `[5000, 12000, 70000]`。
        
    - **用處**: 告訴相機要連拍幾張、每張曝光多久，用於 HDR 合成。
        
    - **對應 Code**: `watch_camcap.m` 中的 `src.ExposureTime = exptimes(i)`。
        
- **`gainlevel`**:
    
    - **意義**: 類比增益 (Analog Gain)。
        
    - **用處**: 在低光環境下增加訊號強度（但會增加雜訊）。
        
    - **對應 Code**: `watch_camcap.m` 中的 `src.Gain = gainlevel`。
        
- **`trim1`**:
    
    - **意義**: 裁切參數 (ROI - Region of Interest)。格式通常是 `[x, y, width, height]`。
        
    - **用處**: 拍完後只保留重要區域，減少儲存空間並加快 AI 處理。
        
    - **對應 Code**: 在 `watch_detail_point_collect_double.m` 中被傳入 `watch_camcap` (雖然你上傳的 `watch_cap.m` 沒顯示這個參數，但呼叫端有傳，表示舊版或新版 function 有支援 `imcrop`)。
        
- **`contcorrect`**:
    
    - **意義**: 對比度修正開關 (Contrast Correction Flag)。
        
    - **用處**: `1` = 啟用 CLAHE (自適應直方圖均衡化)。
        
    - **對應 Code**: `watch_camcap.m` 中的 `if contcorrect==1 ... adapthisteq(...)`。
        
- **`rangecorrect`, `rangecomp`, `enhancecomp`**:
    
    - **意義**: HDR 色調映射 (Tone Mapping) 參數。
        
    - **用處**:
        
        - `rangecomp` (Range Compression): 壓縮動態範圍的程度。
            
        - `enhancecomp` (Enhance Contrast): 增強細節對比的程度。
            
    - **對應 Code**: `watch_camcap.m` 中的 `localtonemap(..., 'RangeCompression', rangecomp, 'EnhanceContrast', enhancecomp)`。
        
- **`adjustsat`**:
    
    - **意義**: 飽和度調整係數 (Saturation)。
        
    - **用處**: 增強金屬或寶石的顏色鮮豔度。
        
    - **對應 Code**: `watch_camcap.m` (需實作 HSV 轉換調整 S 通道)。
        

#### E. 物理補償參數 (Physics Compensation)

- **`glassthick`**:
    
    - **意義**: 藍寶石玻璃厚度 (mm)。
        
    - **用處**: 用於雷射對焦時的折射率補償計算。
        
    - **對應 Code**: `watch_findsurface2.m` 中的 `glasscorrect = (glassthick1 - glassbase) * glassmod`。
        
- **`Zmod`**:
    
    - **意義**: Z 軸微調量 (Z-Modifier)。
        
    - **用處**: 針對該特定錶款的微小高度修正（例如這隻錶比較厚）。
        
- **`glassflat`**:
    
    - **意義**: 玻璃平整度標記 (Boolean)。
        
    - **用處**: `1` = 平面玻璃，`0` = 凸面/透鏡玻璃 (Domed)。凸面玻璃會嚴重影響雷射測距與折射計算，可能需要不同的對焦演算法。
        

---

### 2. 各 Point Type 的特殊參數詳解

除了通用參數外，不同類型的點有其特殊用途：

#### A. Glass Point (玻璃定位點)

這是所有檢測的**基準點**。

- **`etchID`, `etchID2`**: 可能是雷射防偽標籤 (Laser Etching) 的 ID，用於定位玻璃表面。
    
- **功能**:
    
    1. 用來測量玻璃的高度與傾斜度 (`watch_glass_thickness.m`)。
        
    2. 確認玻璃是否存在 (`hasglass` flag in `watch_findsurface*.m`)。
        
    3. 建立 `Zbase` 基準面，後續所有透過玻璃拍攝機芯的動作，都要扣掉這個基準面並加上折射補償。
        

#### B. Top Point / Side Point (外觀檢測點)

主要用於檢測錶殼 (Case)、錶圈 (Bezel) 與錶帶 (Strap) 的外觀瑕疵。

- **`topID`, `topID2`, `sideID`, `sideID2`**: 資料庫中該瑕疵點的唯一識別碼。
    
- **`Xmod`, `Ymod`**:
    
    - **意義**: X/Y 軸修正量。
        
    - **用處**: 某些錶款的直徑不同，Top/Side view 需要根據錶徑微調相機位置，保持對焦距離一致。
        

#### C. Macro Point / Micro Point (細節與機芯檢測點)

這是最複雜的部分，通常用於拍攝錶盤 (Dial) 細節與機芯 (Movement) 零件。

- **`pointID`, `pointID2`**: 特徵點 ID。
    
- **`parttype`, `detailedparttype`**:
    
    - **意義**: 部位描述字串 (例如: `parttype="Movement"`, `detailedparttype="Balance Wheel"` (擺輪))。
        
    - **用處**:
        
        1. **存檔命名**: 生成如 `Rolex_Submariner_Movement_BalanceWheel.jpg` 的檔名。
            
        2. **AI 模型路由**: Python 程式碼會根據 `parttype` 決定要呼叫哪個 AI 模型 (例如 `parttype="Text"` -> 呼叫 OCR 模型；`parttype="Lume"` -> 呼叫 Segmentation 模型)。
            
- **`heightmod` (極重要! 🌟)**:
    
    - **意義**: **特徵深度 (Depth)**。即「該零件位於玻璃表面下方多少 mm 處」。
        
    - **用處**: 雷射感測器只能測到「玻璃表面」。要對焦到玻璃下方的「指針」或更深處的「擺輪」，必須依賴此參數。
        
    - **對應 Code**: `watch_detail_point_collect_double.m`：
        
        Matlab
        
        ```
        % 1. 找玻璃表面
        [Znew, ...] = watch_findsurface3(...);
        % 2. 往下潛 heightmod 的深度
        Ztarget = Znew - heightmod;
        watch_zaber_move(..., Ztarget);
        ```
        
    - **Python 移植注意**: 這就是為什麼你的 Python 需要 `FocusController` 能夠處理 `offset`。
        

---

### 3. Python 移植建議總結

在你的 Python `template_B4FC.yaml` (或其他 Template) 中，結構應該長這樣才能對應上述功能：

YAML

```
# template_Rolex_Submariner.yaml 範例

points:
  - name: "Movement_Balance_Wheel"
    type: "micro_point"
    # 對應 parttype, detailedparttype
    tags: ["Movement", "BalanceWheel"] 
    
    # 對應 X, Y, Z, yaw, pitch
    position: 
      x: 50.5
      y: 100.2
      z_base: 48.0
      rot_z: 0
      rot_x: 90
    
    # 對應 heightmod (相對於玻璃表面的深度)
    depth_offset: 2.5 
    
    # 對應 camused
    camera: "micro_cam" 
    
    # 對應 lighting (microlight, sidelight...)
    lighting:
      ring: 0
      coaxial: 100
      top_spot: 0
    
    # 對應 exptimes, gainlevel, contcorrect, rangecomp
    capture_settings:
      exposures: [5000, 12000, 70000]
      gain: 15
      enhance_clahe: true  # contcorrect
      hdr_tone_map:        # rangecomp/enhancecomp
        range_compression: 0.6
        enhance_contrast: 0.4

    # 對應 glassthick (若為固定值)
    glass_properties:
      thickness: 2.0
      is_flat: true
```

這份解析應該能幫助你將 `appModel.mat` 裡的黑盒子參數，轉化為 Python 中清晰可讀的 YAML 設定檔。這些參數是系統能否「自動適應」不同手錶的關鍵。