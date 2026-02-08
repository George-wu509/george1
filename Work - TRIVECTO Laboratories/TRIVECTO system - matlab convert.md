
|                                            |     |
| ------------------------------------------ | --- |
| [[#### matlab functionality/ matlab code]] |     |
|                                            |     |
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

|**MATLAB 功能**|**對應 Python 檔案**|**狀態評估 (30-year Tech Lead View)**|
|---|---|---|
|**Zaber 移動**|`Controller/hardware_managers.py` (ZaberManager)|**已完成**。基礎移動功能已具備。|
|**相機擷取**|`Controller/hardware_managers.py` (CameraManager)|**部分完成**。目前只有單張擷取 (`capture_image`)，**缺 HDR 邏輯**。|
|**UUID 生成**|`uuid` (Python 內建庫)|**已完成**。Python 直接用 `uuid.uuid4()` 取代。|
|**流程控制** (`collect`)|`Controller/plans/*.yaml` + `WorkflowManager`|**架構已完成**。用 YAML/DB 定義步驟取代了 MATLAB 的寫死腳本。|
|**自動對焦** (`findsurface`)|`Controller/controller_server.py` (`run_auto_focus`)|**過於簡化**。目前的 Python 版只是簡單的 `Z_base + dist`，**缺少 MATLAB 版的 Retry 機制與玻璃修正**。|
|**影像對位** (`watchshift`)|`tasks/api_servers/server_features.py` (推測)|**待確認**。Python 生態系有強大的 OpenCV，通常用 `cv2.findHomography` 或 `ORB` 取代。|
|**座標轉換** (`tele2detail`)|**缺失**|目前沒看到 Python 版有實作「廣角像素 -> 馬達座標」的校正矩陣邏輯。|

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