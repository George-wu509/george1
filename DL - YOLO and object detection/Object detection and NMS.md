
NMS: <mark style="background: #FFF3A3A6;">RCNN series, YOLO series, SSD</mark>
No NMS: <mark style="background: #BBFABBA6;">No Anchor(CenterNet..), YOLOv10, DETR</mark>

許多現代目標檢測模型都在模型內部集成了 NMS，以提高效率。以下是您在使用這些模型時可能需要配置的相關參數：

|                          | 模型內部集成了 NMS                        |                                                                                                                                                                                                                                    |
| ------------------------ | ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| YOLO                     | 模型的後處理階段或檢測層中                      | # 調整置信度閾值<br>model.conf = 0.5  # 只保留置信度 >= 0.5 的檢測框<br><br># 調整 IoU 閾值<br>model.iou = 0.45 # NMS 的 IoU 閾值<br><br># 調整最大檢測數量<br>model.max_det = 100 # 每張圖片最多保留 100 個檢測框                                                             |
| Faster RCNN<br>Mask RCNN | 模型的 RoI (Region of Interest) 後處理階段 | # 設定置信度閾值<br>cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5<br><br># 設定 IoU 閾值<br>cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.45<br><br># 設定最大檢測數量<br>cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMAGE = 100                                      |
| FCOS                     | 模型的後處理階段                           | # 設定置信度閾值<br>cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5<br><br># 設定 IoU 閾值<br>cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.45<br><br># 設定最大檢測數量<br>cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMAGE = 100                                      |
|                          |                                    |                                                                                                                                                                                                                                    |
|                          | NMS在模型之後處理                         |                                                                                                                                                                                                                                    |
| EfficientDet             | 模型的後處理階段                           | predictions = model.predict(image)<br><br>detections = postprocess_boxes(<br>      predictions[0], image.shape[1], <br>      image.shape[2], score_threshold=0.5, <br>      iou_threshold=0.45,<br>      max_detection_points=100) |
| MobileNet-SSD            | 模型的後處理階段                           |                                                                                                                                                                                                                                    |
|                          |                                    |                                                                                                                                                                                                                                    |



NMS: 將所有的框(boundary box)依照confidence排序, 先選出最高confidence框, 然後計算這個框跟其他框的IoU, 如果高於門檻則刪除. 接下來再繼續下一個第二高confidence框 

**AI 物件偵測模型是否有的 AI 模型得到結果之後還需要用 NMS？**

是的，許多主流的 AI 物件偵測模型在得到初步的預測結果後，通常**需要**使用非極大值抑制（NMS）來進行後處理。這是因為模型可能會對同一個物體產生多個重疊的預測框。NMS 的作用就是從這些重疊的預測框中選擇出最優的一個，並抑制掉其餘的冗餘框。

然而，近年來也出現了一些**不需要** NMS 的物件偵測模型。這些模型通常採用不同的預測和匹配機制，例如直接預測物體的中心點和尺寸，並使用一對一的匹配策略，從而在模型架構上避免了產生大量重疊框的可能性，因此在後處理階段也就不需要 NMS。

**有哪些常用 AI based object detection model 需要 NMS, 那些不需要?**

**需要 NMS 的常用模型：**

- **基於錨框（Anchor-based）的模型：**
    - **R-CNN 系列 (R-CNN, Fast R-CNN, Faster R-CNN)：** 這些是經典的兩階段偵測器，會先生成候選區域，然後對這些區域進行分類和邊界框回歸。由於錨框的設計，可能對同一個物體產生多個候選框，因此需要 NMS 來篩選。
    - **YOLO 系列 (YOLOv1, YOLOv2, YOLOv3, YOLOv4, YOLOv5, YOLOv7, YOLOv8 等)：** YOLO 是一種單階段偵測器，它將圖像劃分為網格，並在每個網格單元中預測多個帶有不同尺寸和比例的錨框。因此，對於同一個物體，可能會有多個錨框被預測為包含該物體，需要 NMS 來去除冗餘框。
    - **SSD (Single Shot MultiBox Detector)：** 類似於 YOLO，SSD 也是一種單階段偵測器，它在不同尺度的特徵圖上使用一系列預定義的錨框進行預測，因此也需要 NMS。
    - **RetinaNet：** 這是一種單階段偵測器，旨在解決單階段偵測器在訓練過程中前景-背景類別不平衡的問題。它依然基於錨框進行預測，因此需要 NMS。

**不需要 NMS 的常用模型（或正在發展此方向的模型）：**

- **基於中心點（Center-based）的模型：**
    
    - **CenterNet：** 這個模型將每個物體視為一個點——其邊界框的中心點。模型預測每個物體的中心點heatmap，以及到其邊界框邊緣的偏移量。由於每個物體只預測一個中心點，因此在後處理中不需要 NMS 來去除重複框。它通常通過提取 heatmap 的局部峰值來確定物體中心。
    - **Objects as Points (ExtremeNet 的基礎)：** 這個方法也預測物體的中心點和其他關鍵點，並將它們組合起來形成邊界框，避免了使用錨框和 NMS。
    - **YOLOv10：** 最新的 YOLO 版本 YOLOv10 正在探索和採用無 NMS 的訓練方法，旨在提高推理速度。
- **基於 Transformer 的模型：**
    
    - **DETR (DEtection TRansformer)：** DETR 是一種開創性的使用 Transformer 進行物件偵測的模型。它直接預測一組固定數量的物體邊界框，並使用二分匹配（Bipartite Matching）將預測結果與真實標註進行匹配。由於其預測是直接且唯一的，因此在推理階段不需要 NMS。
    - **Deformable DETR 及其後續版本 (如 DINO)：** 這些模型在 DETR 的基礎上進行了改進，但仍然保留了基於 Transformer 的架構和二分匹配的特性，因此也不需要 NMS。

**請繁體中文解釋甚麼是 NMS（Non-Maximum Suppression）**

**非極大值抑制（Non-Maximum Suppression，簡稱 NMS）** 是一種在物件偵測任務中常用的**後處理技術**。<mark style="background: #FFF3A3A6;">當一個訓練好的物件偵測模型對一張圖片進行預測時，可能會對同一個物體產生多個邊界框（bounding box）。這些邊界框可能彼此重疊</mark>，並且都帶有一個**置信度分數**（confidence score），表示模型認為這個框內包含特定物體的可能性有多大。

NMS 的目標是**從這些重疊的預測框中選擇出最優的一個**，並**抑制（去除）掉其餘的冗餘框**，從而確保每個真實的物體只被一個最準確的邊界框標註出來。

NMS 的典型運作步驟如下：

1. **根據置信度分數對所有的預測框進行排序**，將置信度最高的框放在最前面。
2. **選擇置信度最高的預測框**，並將其加入到最終的偵測結果列表中。
3. **計算這個被選中的預測框與其餘所有預測框之間的重疊程度**（通常使用**交並比，Intersection over Union，IoU** 來衡量）。IoU 的值越高，表示兩個框的重疊程度越大。
4. **對於所有與當前選中框的 IoU 值大於某個預先設定的閾值（例如 0.5）的預測框，將它們從待處理的預測框列表中移除**。這是因為這些框很可能都是檢測到同一個物體，而我們已經選擇了置信度最高的那個。
5. **重複步驟 2-4**，直到待處理的預測框列表為空。
6. **最終偵測結果列表中的框就是經過 NMS 處理後得到的、每個物體一個的最佳預測框。**

**總結來說，NMS 的作用是去除物件偵測模型產生的冗餘和重疊的預測框，保留最可信的預測結果，從而提高偵測結果的準確性和可讀性。**




# 一、什麼是 NMS（Non-Maximum Suppression）？

NMS 是 **非極大值抑制**，在物件偵測中常用來**消除多餘重疊的預測框**。

### 📌 為什麼需要 NMS？

大多數物件偵測模型（尤其是 anchor-based）會對同一個物件產生多個重疊框


# ✅ 二、NMS 的具體流程是什麼？

假設你有多個預測框，每個都有一個信心分數（score）與座標：

### 📌 NMS 處理步驟：

1. **將所有預測框按 score 從高到低排序**。
    
2. **選取最高分框 A，保留作為輸出**。
    
3. **計算其餘所有框與 A 的 IoU**：
    
    - 如果 IoU ≥ NMS 門檻（如 0.5），則**刪除這些重疊的框**。
        
4. 重複步驟 2~3，直到沒有框為止。
    

這樣保證：

- **對每個物件只保留一個最佳框**
    
- **避免重複預測**
    

---

# ✅ 三、哪些模型需要 NMS？

|模型類型|是否使用 Anchor|是否需要 NMS？|原因說明|
|---|---|---|---|
|**Faster R-CNN**|✅ 有|✅ 需要 NMS（2 次）|Proposal 跟 Final 預測都會重複預測|
|**RetinaNet**|✅ 有|✅ 需要|對每個 anchor 輸出預測，數量龐大且重疊|
|**SSD / YOLO v1~v4**|✅ 有|✅ 需要|每格位置可能預測多物件，多 anchor 導致重複預測|
|**FCOS / CenterNet**|❌ 無|✅ 通常仍會用|雖然 anchor-free，但一張圖的像素會預測多個 bbox|
|**DETR / Deformable DETR**|❌ 無|❌ 不需要|使用固定數量的 object queries 且透過 matching 避免重疊預測|

---

# ✅ 四、NMS 在模型中在哪一步執行？（以 Faster R-CNN 為例）

### 🔁 流程圖：

text

複製編輯

`輸入圖像  → Backbone + FPN  → RPN（產生數千個 anchor）  → 回歸修正 → Proposal（還是數千個框）  → 🔶 NMS（第一次，保留 Top-N Proposal）  → RoIAlign  → Classification + BBox Regression  → 🔶 NMS（第二次，過濾最終輸出重疊框）`

---

### ✅ NMS 在哪兩個階段執行？

|階段|NMS 是否使用|說明|
|---|---|---|
|**RPN 階段（第一次）**|✅|從上萬個 anchor proposals 中保留 top-N|
|**最終輸出（第二次）**|✅|從 classification head 預測結果中，過濾同類別重疊的框|

---

## ✅ 舉例說明：Faster R-CNN 的 NMS 應用

假設輸入一張圖片，模型預測下列框（同一類別）：

|預測框 ID|Score|IoU（相對於框 #1）|
|---|---|---|
|1|0.95|--|
|2|0.90|0.75|
|3|0.85|0.60|
|4|0.70|0.40|

若 NMS 門檻設為 0.5：

- 框 1 被保留（score最高）
    
- 框 2 IoU=0.75 → 移除（因為與框1重疊太多）
    
- 框 3 IoU=0.60 → 移除
    
- 框 4 IoU=0.40 → 保留
    

輸出為框 1 和 4，成功消除重複偵測。

---

## ✅ 為什麼 **DETR 不需要 NMS？**

|模型|輸出方式|為何不需要 NMS|
|---|---|---|
|**DETR**|使用固定數量（例如 100 個）的 Object Queries，輸出固定數量預測框|採用 **Hungarian Matching** 演算法，將每個 query 與 ground truth 進行一一最佳配對，且預測分散，無需過濾重疊|

這種設計本質上就**避免了多框預測同一物件的情況**，因此不需要 NMS。

---

# ✅ 小結

| 問題                       | 解釋                                                                  |
| ------------------------ | ------------------------------------------------------------------- |
| 有 Anchor 的模型是否一定需要 NMS？  | ✅ **幾乎都需要**，因為會對同一物件產生多個預測框                                         |
| 沒有 Anchor 的模型是否就不需要 NMS？ | ❌ **不一定**，如 FCOS 還是會對同一物件預測多框，需要 NMS 處理；但 **DETR 因設計避免重複預測，**不需 NMS |
| NMS 在模型中哪裡使用？            | 1. RPN Proposal 後；2. Final 預測輸出前                                    |





## ✅ 表格整理：各模型 NMS 作用位置

|**模型名稱**|**是否使用 Anchor？**|**是否使用 NMS？**|**NMS 用途與位置**|
|---|---|---|---|
|**YOLO v1–v4**|✅ 是|✅ 使用|**輸出階段使用**，篩選掉重疊的高分框，保留每類別最佳框|
|**YOLO v5–v8**|❌（v8為 Anchor-Free）|✅ 使用|**最終預測階段使用**，在每個 grid cell 或中心點預測後去除重複框|
|**FCOS**|❌ 否|✅ 使用|**像素級預測後使用**，因為每個像素可預測一框，需用 NMS 避免大量重疊框|
|**EfficientDet**|✅ 是|✅ 使用|**RPN 最終預測階段使用**，大量 anchor 預測後需進行 NMS 過濾重複物件|
|**MobileNet-SSD**|✅ 是|✅ 使用|**輸出階段使用**，與 YOLO 類似，從所有預測框中篩選出最可能正確且不重疊的|
|**DETR**|❌ 否（無 Anchor）|❌ 不使用|不需 NMS，使用 **Hungarian Matching** 一一配對 Ground Truth，自動學會避免重複預測|
|**Deformable DETR**|❌ 否|❌ 不使用|同 DETR，依靠 matching 訓練策略自動消除重複預測|

---

## ✅ NMS 運作的位置視覺化說明：

text

複製編輯

`【YOLO / SSD / FCOS / EfficientDet】    預測階段（所有位置預測數千框）         ↓    NMS 過濾（保留不重疊的高分框）         ↓    輸出結果  【DETR / Deformable DETR】    預測固定數量的 object queries（例如 100 個）         ↓    每個預測與 GT 配對（Hungarian Matching）         ↓    輸出結果（無需 NMS）`

---

## ✅ 小結與對比

|模型類型|是否需要 NMS|原因與情境說明|
|---|---|---|
|**Anchor-Based**（如 YOLOv4, EfficientDet, SSD）|✅ 需要|大量 anchor 預測框重疊，需 NMS 篩選|
|**Anchor-Free**（如 FCOS）|✅ 通常需要|雖無 anchor，但每像素預測仍可能重疊|
|**Query-Based**（如 DETR）|❌ 不需要|訓練過程透過匹配機制，自動學會不重複預測|

---

## ✅ 建議用途（是否保留 NMS）

| **應用場景**     | **建議**               |
| ------------ | -------------------- |
| 多物件、需高精度     | 使用 NMS 或考慮用 DETR     |
| 即時應用，需快速處理   | 簡化版 NMS（Top-k + IoU） |
| DETR-like 架構 | 可完全移除 NMS            |




我將依據這些模型的 **開源實作（如 PyTorch、TensorFlow、Ultralytics）** 與原始論文，**逐一回答是否內建 NMS、是否需要手動處理**，再總結出哪些模型需要你自己接 NMS。

---

## ✅ 各模型 NMS 是否內建整理表

|**模型**|**是否內建 NMS？**|**說明**|
|---|---|---|
|**YOLO v1–v4**|❌ 不內建|論文與大部分實作中，NMS 需自己加在推理後處理，篩選預測框|
|**YOLO v5 / v7 / v8 (Ultralytics)**|✅ 內建|Ultralytics 的 `model.predict()` 預設會自動加上 NMS|
|**FCOS**|❌ 不內建|輸出為 dense 預測，需自行在推理後執行 NMS，如 `torchvision.ops.nms()`|
|**EfficientDet (TF, PyTorch)**|❌ 不內建|官方與第三方實作通常將 NMS 作為後處理，需要手動套用（如使用 `TFLite` 也要明確加入）|
|**MobileNet-SSD (Caffe, PyTorch)**|❌ 不內建|Caffe 和 Torch 版本通常會在推理後外部加 NMS，如 MobileNet-SSD with OpenCV DNN|
|**DETR / Deformable DETR**|✅ 不需要 NMS|訓練過程中使用 Hungarian Matching，自動避免重疊，不需使用任何形式的 NMS|

---

## ✅ 哪些模型 **需要手動** 加上 NMS？

這些模型在推理階段會輸出 **大量框（或 dense map）**，你需要自己在後處理階段接上 NMS：

|模型|手動加 NMS？|推理時輸出為…|推薦用法|
|---|---|---|---|
|YOLO v1–v4|✅|Dense bbox list|用 IoU 門檻+Top-k 過濾|
|FCOS|✅|每個像素點預測框|用 `torchvision.ops.nms()`|
|EfficientDet|✅|所有 anchor-based 預測框|可搭配 score threshold + NMS|
|MobileNet-SSD|✅|Dense 預測 + 類別分數|OpenCV 提供 `cv2.dnn.NMSBoxes`|
|DETR|❌ 不需要|固定 object queries|不需 NMS，直接輸出結果|

---

## ✅ 哪些模型 **已內建 NMS，不需自己加？**

|模型|說明|
|---|---|
|**YOLOv5~v8 (Ultralytics)**|預設 `model.predict()`、`model.forward()`（`inference=True`）都會自動做 NMS。若不想使用，可在 `model.forward(inference=False)` 中關閉。|
|**DETR / Deformable DETR**|完全不需要，設計上避免重疊預測（透過 Matching 訓練學會）|





**通用 NMS Python 程式碼**

以下是一個通用的 NMS 函數，您可以將其應用於任何輸出包含邊界框和置信度的目標檢測模型。

```Python
import numpy as np

def non_max_suppression(boxes, scores, iou_threshold):
    """
    對檢測到的邊界框執行非極大值抑制。

    Args:
        boxes (np.ndarray): 形狀為 (N, 4) 的 NumPy 陣列，其中 N 是檢測到的邊界框數量。
                           每個框的格式為 (x_min, y_min, x_max, y_max)。
        scores (np.ndarray): 形狀為 (N,) 的 NumPy 陣列，包含每個框的置信度分數。
        iou_threshold (float): IoU (Intersection over Union) 閾值，用於確定重疊框。

    Returns:
        np.ndarray: 保留的邊界框的索引。
    """
    # 如果沒有檢測到框，則返回一個空列表
    if len(boxes) == 0:
        return []

    # 將框的坐標轉換為浮點數
    boxes = boxes.astype(np.float32)

    # 獲取按置信度分數降序排列的框的索引
    indices = np.argsort(scores)[::-1]

    # 用於存儲保留的框的索引的列表
    keep_indices = []

    while indices.size > 0:
        # 選擇置信度最高的框
        current_index = indices[0]
        keep_indices.append(current_index)

        # 計算當前框與剩餘框的 IoU
        current_box = boxes[current_index]
        remaining_indices = indices[1:]
        remaining_boxes = boxes[remaining_indices]

        x1 = np.maximum(current_box[0], remaining_boxes[:, 0])
        y1 = np.maximum(current_box[1], remaining_boxes[:, 1])
        x2 = np.minimum(current_box[2], remaining_boxes[:, 2])
        y2 = np.minimum(current_box[3], remaining_boxes[:, 3])

        # 計算交集區域
        intersection_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        # 計算每個框的區域
        current_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        remaining_areas = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * (remaining_boxes[:, 3] - remaining_boxes[:, 1])

        # 計算 IoU
        iou = intersection_area / (current_area + remaining_areas - intersection_area + 1e-6)

        # 移除 IoU 大於閾值的框
        indices = remaining_indices[iou <= iou_threshold]

    return np.array(keep_indices)

# 示例用法
if __name__ == '__main__':
    boxes = np.array([[10, 10, 100, 100],
                      [20, 20, 110, 110],
                      [150, 150, 250, 250],
                      [160, 160, 260, 260]])
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    iou_threshold = 0.5
    keep_indices = non_max_suppression(boxes, scores, iou_threshold)
    print("保留的框的索引:", keep_indices)
    print("保留的框:", boxes[keep_indices])
```

**模型內 NMS 設定**

許多現代目標檢測模型都在模型內部集成了 NMS，以提高效率。以下是您在使用這些模型時可能需要配置的相關參數：

**1. YOLO 系列 (YOLOv3, YOLOv5, YOLOv8 等)**

- **設定位置:** 通常在模型的後處理階段或檢測層中。
    
- **主要參數:**
    
    - `conf_thres` (或 `confidence_threshold`): 置信度閾值。只有置信度高於此閾值的檢測框才會被考慮進行 NMS。
    - `iou_thres` (或 `iou_threshold`): IoU 閾值。IoU 高於此閾值的重疊框將被抑制。
    - `max_det` (或 `max_predictions`): 每張圖像輸出的最大檢測框數量。即使在 NMS 之後，也只會保留置信度最高的前 `max_det` 個框。
- **程式碼範例 (PyTorch YOLOv5):**
    
    ```Python
    import torch
    
    # 載入模型
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    
    # 推理
    img = 'https://ultralytics.com/images/zidane.jpg'
    results = model(img)
    
    # results.xyxy[0] 包含每個檢測到的物件的 [x_min, y_min, x_max, y_max, confidence, class]
    # NMS 已經在模型內部執行，可以使用以下參數進行調整：
    
    # 調整置信度閾值
    model.conf = 0.5  # 只保留置信度 >= 0.5 的檢測框
    
    # 調整 IoU 閾值
    model.iou = 0.45 # NMS 的 IoU 閾值
    
    # 調整最大檢測數量
    model.max_det = 100 # 每張圖片最多保留 100 個檢測框
    
    results = model(img)
    print(results.xyxy[0])
    ```
    

**2. FCOS (Fully Convolutional One-Stage Object Detection)**

- **設定位置:** 通常在模型的後處理階段。
    
- **主要參數:**
    
    - `score_thresh`: 置信度閾值，與 YOLO 的 `conf_thres` 類似。
    - `nms_thresh`: IoU 閾值，與 YOLO 的 `iou_thres` 類似。
    - `detections_per_img`: 每張圖像輸出的最大檢測框數量，與 YOLO 的 `max_det` 類似。
- **程式碼範例 (使用 Detectron2 框架):**
    
    ```Python
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    import cv2
    
    # 載入 FCOS 模型配置
    cfg = get_cfg()
    cfg.merge_from_file(
        "detectron2://COCO-Detection/fcos_R_50_FPN_1x.yaml"
    )
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/fcos_R_50_FPN_1x/137038246/model_final_f96b54.pkl"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 設定置信度閾值
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.45 # 設定 IoU 閾值
    cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMAGE = 100 # 設定最大檢測數量
    predictor = DefaultPredictor(cfg)
    
    # 進行預測
    im = cv2.imread("path/to/your/image.jpg")
    outputs = predictor(im)
    print(outputs["instances"].pred_boxes)
    print(outputs["instances"].scores)
    ```
    

**3. EfficientDet**

- **設定位置:** 通常在模型的後處理階段。
    
- **主要參數:**
    
    - `score_threshold`: 置信度閾值。
    - `iou_threshold`: IoU 閾值。
    - `max_detection_points`: 每張圖像的最大檢測框數量。
- **程式碼範例 (使用 `tf.keras` 實現):**
    
    ```Python
    import tensorflow as tf
    from efficientdet import EfficientDetModel
    
    # 載入 EfficientDet 模型
    model = EfficientDetModel(model_name='efficientdet-d0')
    model.load_weights('efficientdet-d0.h5') # 載入預訓練權重
    
    # 進行預測 (假設你已經有了預處理後的圖像 `image`)
    predictions = model.predict(image)
    
    # 後處理函數通常會包含 NMS
    # 你可能需要查看 EfficientDet 實現的程式碼來找到具體的參數名稱和設定方式。
    # 一些實現可能會將這些參數作為後處理函數的輸入。
    
    # 範例 (可能需要根據具體實現調整):
    # from efficientdet.postprocess import postprocess_boxes
    # detections = postprocess_boxes(predictions[0], image.shape[1], image.shape[2],
    #                                score_threshold=0.5, iou_threshold=0.45,
    #                                max_detection_points=100)
    # print(detections)
    ```
    

**4. MobileNet-SSD**

- **設定位置:** 通常在模型的後處理階段。
    
- **主要參數:**
    
    - `confidence_threshold`: 置信度閾值。
    - `iou_threshold`: IoU 閾值。
    - `top_k`: 在 NMS 之前保留的最高置信度框的數量。
- **程式碼範例 (使用 TensorFlow Object Detection API):**
    
    ```Python
    import tensorflow as tf
    import tensorflow_hub as hub
    
    # 載入 MobileNet-SSD 模型
    module_handle = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
    detector = hub.load(module_handle)
    
    # 進行預測 (假設你已經有了圖像 `image_np`)
    detections = detector(image_np)
    
    # 在 TensorFlow Object Detection API 中，NMS 通常是模型圖的一部分。
    # 你可以在配置檔案 (pipeline.config) 中找到相關的 NMS 設定。
    # 常見的配置路徑可能類似於：
    # model {
    #   ssd {
    #     post_processing {
    #       batch_non_max_suppression {
    #         score_threshold: 0.5
    #         iou_threshold: 0.45
    #         max_detections_per_class: 100
    #         max_total_detections: 100
    #       }
    #     }
    #   }
    # }
    
    # 要在程式碼中調整這些參數，你通常需要修改配置檔案並重新訓練或使用提供的工具進行配置。
    # 直接在推理時更改這些內部 NMS 參數可能比較複雜。
    ```
    

**5. Faster R-CNN 和 Mask R-CNN**

- **設定位置:** 通常在模型的 RoI (Region of Interest) 後處理階段。
    
- **主要參數:**
    
    - `score_thresh`: 用於過濾低置信度 RoI 的閾值。
    - `nms_thresh`: 用於對每個類別的 RoI 執行 NMS 的 IoU 閾值。
    - `detections_per_img`: 每張圖像輸出的最大檢測框數量。
- **程式碼範例 (使用 Detectron2 框架 - 與 FCOS 類似):**
    
    ```Python
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    import cv2
    
    # 載入 Faster R-CNN 或 Mask R-CNN 模型配置
    # 例如 Faster R-CNN:
    cfg = get_cfg()
    cfg.merge_from_file(
        "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"
    )
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_1x/137257794/model_final_b27c81.pkl"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 設定置信度閾值
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.45 # 設定 IoU 閾值
    cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMAGE = 100 # 設定最大檢測數量
    predictor = DefaultPredictor(cfg)
    
    # 或者 Mask R-CNN:
    cfg = get_cfg()
    cfg.merge_from_file(
        "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
    )
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137259228/model_final_b83cb5.pkl"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.45
    cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMAGE = 100
    predictor = DefaultPredictor(cfg)
    
    # 進行預測
    im = cv2.imread("path/to/your/image.jpg")
    outputs = predictor(im)
    print(outputs["instances"].pred_boxes)
    print(outputs["instances"].scores)
    if "pred_masks" in outputs["instances"]:
        print(outputs["instances"].pred_masks)
    ```
    

**總結**

- 對於通用的 NMS 需求，您可以使用提供的 Python 程式碼。
- 對於模型內部的 NMS，參數名稱通常包括 `score_thresh` (或 `confidence_threshold`) 和 `iou_thresh` (或 `iou_threshold`)，以及限制輸出數量的參數 (如 `max_det` 或 `detections_per_img`)。
- 具體的參數名稱和設定方式取決於您使用的模型框架 (如 PyTorch, TensorFlow, Detectron2 等)。請參考您所用框架和模型的官方文檔或程式碼來進行更精確的配置。