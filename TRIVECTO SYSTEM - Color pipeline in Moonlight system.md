

|                         |     |
| ----------------------- | --- |
| [[#### Color pipeline]] |     |
|                         |     |
|                         |     |
|                         |     |
|                         |     |


#### Color pipeline
```
- **相機 raw / Bayer 階段**：保留 sensor 原始格式資訊
- **OpenCV 影像處理主流程**：可以統一用 **BGR**
- **Qt 顯示層**：統一走一個 helper，明確處理成 `QImage`
- **AI / ML 輸入**：每個 model 入口自己明確轉成它要的格式（很多模型其實要 RGB）
- **存檔**：明確規定「凡是交給 `cv2.imwrite` 的 3-channel image，一律是 BGR」
```


這份關於色彩空間（Color Space）管理的系統設計建議**非常精確且切中要害**。對於一個高度依賴 OpenCV 進行機器視覺（CV）和硬體控制的系統來說，建立「Single Source of Truth（單一真理來源）」是避免色彩錯亂（阿凡達效應）的唯一解法。

深入檢視你目前的程式碼，可以發現系統中**已經存在色彩空間定義不一致的潛在危機**：

1. **相機解碼層（正確）：** 在 `hardware_managers.py` 的 `_decode_pixel_format` 中，你已經非常正確地將 Bayer 格式統一轉換為 **BGR**（`cv2.COLOR_BAYER_RG2BGR` 等）。
    
2. **儲存層與 OpenCV 處理層（正確）：** `unified_driver.py` 中的 HDR 合成（Mertens）、LAB 色彩空間轉換（CLAHE），以及 `cv2.imwrite`，都是建立在 **BGR** 的基礎上完美運作。
    
3. **UI 顯示層（致命錯誤）：** 在 `main.py` 中（如 `update_stream_frame` 等多處），你直接將相機傳來的 BGR 陣列塞給 `QImage.Format_RGB888`，**完全沒有進行 `BGR2RGB` 的轉換**。這意味著目前 UI 上顯示的顏色，紅藍通道是顛倒的。
    

為了徹底根除這個問題，並落實你提出的優化建議，以下是完整的深度分析與程式碼重構方案。

---

### 一、 核心系統設計原則 (The Color Contract)

在系統中建立嚴格的「色彩契約（Color Contract）」：

- **內部流通標準 (Backend Standard)：** 系統內所有流動的 `np.ndarray` 3-channel 影像，**絕對且唯一是 BGR 格式**。
    
- **相機出口 (Camera Output)：** 負責將 Raw/Bayer 解碼為 BGR。
    
- **顯示終端 (UI Display)：** 負責將 BGR 轉為 RGB 並交給 Qt 渲染。
    
- **AI 預處理 (AI Gateway)：** 負責將 BGR 轉為 RGB（或其他模型指定格式）。
    
- **存檔終端 (File I/O)：** `cv2.imwrite` 信任輸入為 BGR，直接存檔。
    

---

### 二、 完整程式碼修改方案

#### 1. 建立全局 Qt 顯示 Helper 函式

在 `main.py` 的頂部（可以在 `_abs_path_from_here` 等輔助函式附近），新增一個專門處理 OpenCV BGR 轉 QImage 的 Helper。這將成為全系統**唯一**建立 `QImage` 的入口。

Python

```
# =========================================================
# 全域輔助函式 (Global Helper Functions)
# =========================================================
import numpy as np
import cv2
from PySide6.QtGui import QImage, QPixmap

def convert_bgr_to_qimage(img_array: np.ndarray) -> QImage:
    """
    [Single Source of Truth] 統一影像顯示轉換器
    嚴格假設傳入的 3-channel 影像為 OpenCV 原生 BGR 格式。
    將其安全轉換為 Qt 支援的 RGB888 格式。
    """
    if img_array is None or not isinstance(img_array, np.ndarray):
        return QImage()

    # 確保記憶體連續性，避免 QImage 讀取崩潰
    img_safe = np.ascontiguousarray(img_array)
    
    if len(img_safe.shape) == 3:
        h, w, ch = img_safe.shape
        # 關鍵：在此處將 BGR 轉換為 RGB
        img_rgb = cv2.cvtColor(img_safe, cv2.COLOR_BGR2RGB)
        bytes_per_line = ch * w
        return QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        
    elif len(img_safe.shape) == 2:
        h, w = img_safe.shape
        bytes_per_line = w
        return QImage(img_safe.data, w, h, bytes_per_line, QImage.Format_Grayscale8).copy()
        
    return QImage()
```

#### 2. 重構 `main.py` 中所有的顯示邏輯

你需要搜尋 `main.py` 中所有使用 `QImage.Format_RGB888` 的地方，並將繁瑣的陣列處理替換為我們剛建好的 Helper。

**修改點 A：Live Stream 畫面更新 (`update_stream_frame`)**

Python

```
    @Slot(object)
    def update_stream_frame(self, img_array):
        frame, lbl = self._get_preview_label()
        if not frame or not lbl: return

        # [修改前]
        # img_array = np.ascontiguousarray(img_array)
        # if len(img_array.shape) == 3:
        #     h, w, ch = img_array.shape
        #     qimg = QImage(img_array.data, w, h, ch * w, QImage.Format_RGB888)
        # else: ...

        # [修改後] 統一交給 helper
        qimg = convert_bgr_to_qimage(img_array)

        pixmap = QPixmap.fromImage(qimg)
        lbl.setPixmap(pixmap.scaled(frame.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        if not frame.isVisible(): frame.show()
```

**修改點 B：Control Window 預覽 (`_update_control_preview`)**

Python

```
    def _update_control_preview(self):
        # ... 前略 ...
            with driver.cam.capture_lock:
                img_array = driver.cam.capture_image(exposure_time=int(exposure), gain=float(gain))
                
            if img_array is None: return
            
            # [修改前] 直接塞入 Format_RGB888 (導致色偏)
            # [修改後] 統一交給 helper
            qimg = convert_bgr_to_qimage(img_array)

            pixmap = QPixmap.fromImage(qimg)
            lbl.setPixmap(pixmap.scaled(frame.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            if not frame.isVisible(): frame.show()
```

**修改點 C：Template 互動拍攝預覽 (`_scratch_next_capture` & `_capture_and_show_macro_top_view` & `_process_interactive_macro_click`)** 這三個函式中都有類似的重複代碼，全部精簡為一行。

Python

```
                # 找到類似以下的代碼區塊：
                # img_safe = np.ascontiguousarray(img)
                # h_c, w_c = img_safe.shape[:2]
                # fmt = QImage.Format_RGB888 if len(img_safe.shape) == 3 else QImage.Format_Grayscale8
                # bytes_per_line = w_c * 3 if len(img_safe.shape) == 3 else w_c
                # qimg = QImage(img_safe.data, w_c, h_c, bytes_per_line, fmt).copy()

                # 全部替換為：
                qimg = convert_bgr_to_qimage(img)  # 如果變數名叫 img 或 cropped_img，請對應替換
```

#### 3. 建立檔案存取與 AI 分析的防護網 (Architecture Guidelines)

這部分主要是系統設計規範，建議在 `data_manager` 或工具類別中建立 Wrapper，確保未來的開發者不會踩雷。

**A. 存檔防護網 (File Save Guard)** 在處理圖片存檔的模組（如 `DataManager.process_and_sync_raw_image`）中，確立規範：

Python

```
import cv2

def safe_save_image(filepath: str, img_bgr: np.ndarray):
    """
    明確約定：凡是交給 cv2.imwrite 的 3-channel image，一律是 BGR。
    這層 Wrapper 用於語意聲明，並可用於後續統一添加壓縮、浮水印等邏輯。
    """
    if img_bgr is None:
        raise ValueError("Cannot save empty image.")
    # OpenCV imwrite natively expects BGR
    cv2.imwrite(filepath, img_bgr)
```

**B. AI/ML 模型入口防護 (AI Gateway)** 在 `Orchestrator._run_analysis_safe` 或個別 Model 的 Inference 類別中，負責橋接 OpenCV BGR 與模型預期的格式：

Python

```
def prepare_image_for_model(img_bgr: np.ndarray, model_type: str) -> np.ndarray:
    """
    根據不同的 Model 需求，在入口處將內部流通的 BGR 轉換為正確格式。
    """
    if model_type == "pytorch_resnet":
        # PyTorch pre-trained models typically expect RGB
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
    elif model_type == "legacy_opencv_dnn":
        # Some old OpenCV models expect BGR
        return img_bgr.copy()
        
    elif model_type == "monochrome_inspection":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
    else:
        # Default to RGB for most modern ML pipelines (TensorFlow/PyTorch)
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
```

### 三、 總結分析效益

透過上述重構，你的系統將獲得極高的穩定性：

1. **解耦 (Decoupling)：** 影像獲取（硬體）、影像處理（CV 演算法）、影像顯示（UI）三者的色彩空間責任完全脫鉤。
    
2. **防呆 (Fail-Safe)：** 移除了 `main.py` 中四處散落的 `QImage.Format_RGB888` 強制轉型，未來無論是新增視窗還是修改 UI，只要呼叫 `convert_bgr_to_qimage` 就不會發生阿凡達效應。
    
3. **擴展性 (Scalability)：** 當未來導入更多 AI 模型時，工程師不需要去翻找相機底層到底吐出什麼格式，因為「內部絕對是 BGR」的鐵律已經建立，只需要在 AI Gateway 做一次性轉換即可。


