

```python
OpenCV

import cv2

# 打開影片檔案
cap = cv2.VideoCapture('path/to/your/surgical_video.mp4')

# 逐幀讀取和處理
ret, frame = cap.read()

# 顯示處理後的影像
cv2.imshow()

# 複製影像
output_image = image.copy()

# BGR to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 建立遮罩
mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

# 腐蝕操作去除小的噪點
mask = cv2.erode(mask, kernel, iterations=1)

# 膨脹操作恢復物體大小
mask = cv2.dilate(mask, kernel, iterations=2)

# 尋找輪廓
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 輪廓
cv2.contourArea(main_contour)
```


## 應用 OpenCV 於手術影片偵測與分割的深度解析

**OpenCV (Open Source Computer Vision Library)** 是一個開源的電腦視覺和機器學習軟體庫。它包含了超過 2500 種最佳化的演算法，為影像和影片處理提供了極其豐富的工具集。對於手術影片分析，OpenCV 是執行基礎任務（如讀取影片、影像前處理）和實現傳統電腦視覺演算法（如顏色偵測、輪廓分析）的基石。

雖然目前最先進 (State-of-the-art) 的方法大多基於深度學習（如前述的 SageMaker 方案），但使用 OpenCV 的傳統方法在以下場景中依然非常有價值：

- **快速原型驗證**：快速測試一個想法的可行性。
    
- **運算資源有限**：在不具備強大 GPU 的環境中運行。
    
- **特定且穩定的環境**：當手術環境的光照、器械顏色等條件相對固定時。
    
- **作為深度學習的前/後處理步驟**：例如，使用 OpenCV 擷取感興趣區域 (ROI) 後再送入神經網路。
    

### 核心工作流程：處理影片檔案

無論是偵測還是分割，處理手術影片的第一步都是讀取影片並逐幀 (frame-by-frame) 處理。

```Python
import cv2
import numpy as np

# 打開影片檔案
cap = cv2.VideoCapture('path/to/your/surgical_video.mp4')

# 檢查影片是否成功打開
if not cap.isOpened():
    print("錯誤: 無法打開影片檔案")
    exit()

# 逐幀讀取和處理
while True:
    # ret 是一個布林值，表示是否成功讀取到畫格
    # frame 是讀取到的單一畫格 (一個 NumPy 陣列)
    ret, frame = cap.read()

    # 如果 ret 是 False，表示影片已結束或讀取錯誤
    if not ret:
        print("影片處理完畢或讀取錯誤")
        break

    # --- 在這裡插入你的偵測或分割程式碼 ---
    # processed_frame = your_processing_function(frame)


    # 顯示處理後的影像
    cv2.imshow('Surgical Video Analysis', frame) # 或者顯示 processed_frame

    # 按下 'q' 鍵退出迴圈
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源並關閉所有視窗
cap.release()
cv2.destroyAllWindows()
```

---

### 應用範例一：手術器械偵測 (Detection)

偵測的目標是找到物體在影像中的位置，通常以一個邊界框 (Bounding Box) 來表示。

#### 方法 A：基於顏色的偵測

許多手術器械的手柄或尖端帶有特定的顏色（例如藍色、綠色），這為偵測提供了簡單有效的特徵。使用 HSV 色彩空間比傳統的 RGB 更能抵抗光照變化的影響。

**具體步驟：**

1. 將影像從 BGR (OpenCV 的預設格式) 轉換到 HSV。
    
2. 定義目標顏色在 HSV 空間中的範圍（例如，藍色的 H, S, V 最小值和最大值）。
    
3. 使用 `cv2.inRange()` 函數建立一個遮罩 (Mask)，只保留在顏色範圍內的像素。
    
4. 對遮罩進行形態學操作（腐蝕和膨脹）以去除噪點。
    
5. 使用 `cv2.findContours()` 找到遮罩中的所有獨立輪廓。
    
6. 過濾輪廓，例如選擇面積最大的輪廓，它最可能代表我們想找的器械。
    
7. 使用 `cv2.boundingRect()` 計算該輪廓的最小正交邊界框。
    
8. 在原始影像上繪製這個邊界框。
    

**完整範例程式碼 (處理單張圖片):**

```Python
import cv2
import numpy as np

def detect_blue_instrument(image):
    """
    在一張影像中偵測藍色的手術器械。
    :param image: 輸入的影像 (BGR格式)
    :return: 繪製了邊界框的影像
    """
    # 複製影像，以免修改原始影像
    output_image = image.copy()
    
    # 1. BGR to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 2. 定義藍色的 HSV 範圍
    # 這個範圍需要根據實際影片中的藍色進行調整
    lower_blue = np.array([90, 80, 50])
    upper_blue = np.array([130, 255, 255])
    
    # 3. 建立遮罩
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    
    # 4. 形態學操作 (可選，但建議)
    # 腐蝕操作去除小的噪點
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    # 膨脹操作恢復物體大小
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # 5. 尋找輪廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 6. 假設面積最大的輪廓是我們的目標
        main_contour = max(contours, key=cv2.contourArea)
        
        # 過濾掉太小的輪廓，避免誤判
        if cv2.contourArea(main_contour) > 500: # 面積閾值可調
            # 7. 計算邊界框
            x, y, w, h = cv2.boundingRect(main_contour)
            
            # 8. 在原始影像上繪製邊界框和標籤
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(output_image, 'Surgical Instrument', (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
    return output_image, mask

# --- 主程式 ---
# 載入一張手術場景的範例圖片
# 請將 'surgical_scene.jpg' 換成你自己的圖片路徑
try:
    frame = cv2.imread('surgical_scene.jpg')
    if frame is None:
        raise FileNotFoundError
except FileNotFoundError:
    print("錯誤: 找不到 'surgical_scene.jpg'。正在建立一個假的範例圖片。")
    # 建立一個模擬影像：黑色背景上有一個大的藍色矩形
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (200, 150), (400, 200), (255, 0, 0), -1) # BGR 藍色

# 執行偵測
processed_frame, result_mask = detect_blue_instrument(frame)

# 顯示結果
cv2.imshow('Original Frame', frame)
cv2.imshow('Blue Mask', result_mask)
cv2.imshow('Detection Result', processed_frame)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### 應用範例二：器官或病灶分割 (Segmentation)

分割的目標是找到物體所佔據的**精確像素區域**，而不僅僅是一個方框。分割的結果是一個遮罩影像。

#### 方法 B：互動式前景分割 (GrabCut 演算法)

GrabCut 是一種基於圖割 (Graph Cut) 的強大分割演算法。它需要使用者提供一個包含目標物體的初始矩形，然後演算法會自動迭代地將矩形內的像素區分為前景和背景。

**適用場景**：在需要精確分割但形狀不規則的物體（如器官、組織）時非常有用，可以作為半自動的標註工具。

**具體步驟：**

1. 定義一個矩形 (ROI, Region of Interest) 包圍你想要分割的物體。
    
2. 建立一個與原圖大小相同的遮罩，用於存放演算法的中間結果。
    
3. 呼叫 `cv2.grabCut()` 函數，傳入原圖、遮罩和矩形。
    
4. 演算法會更新遮罩，將像素標記為「確定背景」、「確定前景」、「可能背景」、「可能前景」。
    
5. 根據更新後的遮罩，將所有確定前景和可能前景的像素提取出來，形成最終的分割結果。
    

**完整範例程式碼 (處理單張圖片):**

```Python
import cv2
import numpy as np

# 載入一張手術場景的範例圖片
# 請將 'surgical_scene_organ.jpg' 換成你自己的圖片路徑
try:
    img = cv2.imread('surgical_scene_organ.jpg')
    if img is None:
        raise FileNotFoundError
except FileNotFoundError:
    print("錯誤: 找不到 'surgical_scene_organ.jpg'。正在建立一個假的範例圖片。")
    # 建立一個模擬影像：綠色背景上有一個紅色的圓形（模擬器官）
    img = np.full((480, 640, 3), (0, 100, 0), dtype=np.uint8) # BGR 綠色
    cv2.circle(img, (320, 240), 100, (0, 0, 200), -1) # BGR 紅色

# 1. 定義一個包含前景物體的矩形 (x, y, w, h)
# 這個矩形需要手動或透過其他偵測方法初步確定
roi_rect = (200, 100, 250, 250) 

# 2. 初始化 GrabCut 需要的遮罩和模型
mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# 3. 執行 GrabCut 演算法
# cv2.GC_INIT_WITH_RECT 表示我們使用矩形來初始化
try:
    cv2.grabCut(img, mask, roi_rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
except Exception as e:
    print(f"GrabCut 執行錯誤: {e}")
    exit()

# 4. 建立一個新的遮罩，將所有確定前景和可能前景的像素設為 1，其餘為 0
# 這樣我們就可以用它來提取前景
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# 5. 將新遮罩應用到原始影像上，提取分割出的前景
segmented_img = img * mask2[:, :, np.newaxis]

# 在原圖上畫出初始矩形以供參考
cv2.rectangle(img, (roi_rect[0], roi_rect[1]), (roi_rect[0]+roi_rect[2], roi_rect[1]+roi_rect[3]), (0, 255, 0), 2)
cv2.putText(img, 'Initial ROI', (roi_rect[0], roi_rect[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


# 顯示結果
cv2.imshow('Original with ROI', img)
cv2.imshow('Segmented Object', segmented_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### 綜合應用：處理完整影片並輸出結果

現在，我們將範例一中的「顏色偵測」方法應用到完整的影片處理流程中，並將帶有偵測框的影片保存下來。

**完整範例程式碼 (處理影片檔案):**

```Python
import cv2
import numpy as np

def detect_in_video_frame(frame):
    """在單一影片畫格中偵測藍色物體並回傳處理後的畫格"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 80, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(main_contour) > 500:
            x, y, w, h = cv2.boundingRect(main_contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(frame, 'Instrument', (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame

# --- 主影片處理流程 ---
input_video_path = 'input_surgical_video.mp4'
output_video_path = 'output_detected_video.mp4'

cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print(f"錯誤: 無法打開影片 {input_video_path}")
    # 如果找不到影片，建立一個假的輸入影片用於演示
    print("正在建立一個假的輸入影片 'input_surgical_video.mp4'")
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_fake = cv2.VideoWriter(input_video_path, fourcc, 20.0, (width, height))
    for i in range(100): # 建立一個5秒的影片
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # 讓藍色方塊移動
        x_pos = 100 + int(200 * np.sin(i * 0.1))
        cv2.rectangle(frame, (x_pos, 200), (x_pos + 150, 250), (255, 0, 0), -1)
        out_fake.write(frame)
    out_fake.release()
    print("假影片建立完成，請重新執行腳本。")
    exit()


# 取得影片的屬性以建立 VideoWriter
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 定義影片編碼器並建立 VideoWriter 物件
# 使用 'mp4v' for .mp4, 'XVID' for .avi
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

print("正在處理影片...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 對每一幀進行處理
    processed_frame = detect_in_video_frame(frame)

    # 將處理後的畫格寫入輸出影片
    out.write(processed_frame)

    # (可選) 顯示即時處理畫面
    cv2.imshow('Processing...', processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"處理完成！影片已儲存至 {output_video_path}")

# 釋放所有資源
cap.release()
out.release()
cv2.destroyAllWindows()
```

### 總結與限制

使用 OpenCV 的傳統方法可以快速實現手術影片中特定物體的偵測與分割。然而，這些方法也存在明顯的局限性：

- **對環境變化敏感**：光照的改變、陰影、反光都會嚴重影響顏色偵測的穩定性。
    
- **缺乏語意理解**：演算法只認識顏色和形狀，不理解「這是一把剪刀」或「這是肝臟」。
    
- **難以泛化**：為某個手術場景調整好的顏色範圍，換到另一個場景可能就完全失效。
    
- **處理遮擋困難**：當器械被組織部分遮擋時，輪廓會不完整，可能導致偵測失敗。
    

因此，在實際的商業或臨床應用中，通常會將 OpenCV 作為工具，並結合 **OpenCV 的 DNN 模組**來載入預先訓練好的深度學習模型（如 YOLO, U-Net），以達到更高、更穩健的準確度。