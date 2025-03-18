


|                                        |     |
| -------------------------------------- | --- |
| [[###將 YOLO 模型應用於 Flock Safety 和 LPR]] |     |
| [[###YOLO 模型中 NMS（非極大值抑制）的作用]]         |     |
| [[###YOLO輕量化模型的設計]]                    |     |
|                                        |     |
|                                        |     |
|                                        |     |




### 將 YOLO 模型應用於 Flock Safety 和 LPR

在將 YOLO 模型應用於 Flock Safety 和 LPR（車牌識別，License Plate Recognition）攝影機時，需要考慮多種環境挑戰（如天氣、物體太小、速度太快）以及邊緣設備（edge device）的硬體限制。以下我會詳細解釋如何通過調整 YOLO 設計中的 **grid size**、**參數（parameters）**、**超參數（hyperparameters）** 和 **前處理（preprocessing）** 來最佳化其在這些場景下的 **object detection** 和 **tracking** 能力。這些最佳化將針對速度、準確度和硬體限制進行平衡。

---

### 挑戰分析

1. **天氣影響**：雨、霧、雪或強光可能降低圖像對比度，影響車牌檢測。
2. **物體太小**：車牌或遠處物體在圖像中佔比小，容易被忽略。
3. **速度太快**：快速移動的車輛導致模糊，影響檢測和追蹤。
4. **邊緣設備硬體限制**：低功耗、低記憶體的設備需要輕量化模型和高效推理。

---

### 最佳化策略

#### 1. Grid Size 調整

- **YOLO 的網格分割**：
    - YOLO 將圖像分割為 S×S 網格，每個網格負責檢測其中心落在該區域的物體。Grid size 的大小直接影響檢測精度和計算成本。
    - 原始 YOLOv1 使用 7×7 網格，適合較大物體，但對於小物體（如車牌）效果不佳。
- **最佳化建議**：
    - **增加網格密度**：例如從 7×7 調整到 13×13 或 26×26（如 YOLOv3/v5 的多尺度預測）。這能提升對小物體（如車牌）的檢測能力，因為每個網格覆蓋的區域變小，解析度更高。
    - **多尺度網格**：採用類似 YOLOv3 的 FPN（Feature Pyramid Network）結構，在不同層級（例如 13×13、26×26、52×52）進行預測，適應不同大小的物體。
    - **硬體限制考量**：增加網格密度會提升計算量，因此在邊緣設備上可選擇輕量化模型（如 YOLOv5n 或 YOLOv8n），並限制最高網格層級（例如僅使用 26×26）。

#### 2. Parameters（參數）調整

- **Bounding Box 數量 (B)**：
    - 每個網格預測 B 個框，默認 B=2（YOLOv1）。對於快速移動的車輛或多車牌場景，增加 B（例如 B=3 或 5）可以提升召回率。
    - 但 B 增加會提高計算負擔，因此需根據硬體限制平衡。
- **Anchor Boxes**：
    - YOLO 使用預定義的 anchor boxes 來預測 bounding box 的形狀。對於車牌（通常長寬比固定），應根據訓練數據集（如 CCPD 或 UFPR-ALPR）運行 k-means 聚類，自訂 anchor box 尺寸。
    - 例如，車牌可能是長條形（如 2:1 比例），自訂 anchor 可提升檢測精度。
- **通道數與層數**：
    - 減少 backbone（如 CSPDarknet）的通道數（如從 512 降到 256）或層數，輕量化模型以適應邊緣設備。
    - 使用高效卷積（如 Depthwise Separable Convolution）替代標準卷積，降低參數量。

#### 3. Hyperparameters（超參數）調整

- **學習率 (Learning Rate)**：
    - 在多變天氣條件下，模型需要更好泛化能力。使用動態學習率（如 cosine annealing）或較小的初始學習率（例如 0.001），避免過擬合。
- **IoU 閾值**：
    - 調整訓練時的 IoU 閾值（例如從 0.5 提高到 0.7），提升對小物體（如車牌）的定位精度。
- **Confidence 閾值**：
    - 在檢測快速移動物體時，降低 confidence 閾值（例如從 0.5 降到 0.3），避免漏檢，但需搭配更強的 NMS（非極大值抑制）過濾冗餘框。
- **NMS 參數**：
    - 快速移動的車輛可能導致多個重疊框，調整 NMS 的 IoU 閾值（例如從 0.45 調到 0.6），減少冗餘檢測。
- **Batch Size**：
    - 邊緣設備記憶體有限，選擇小 batch size（如 1 或 2）進行推理，同時在訓練時使用較大 batch size（若硬體允許）提升穩定性。

#### 4. 前處理（Preprocessing）最佳化

- **圖像增強**：
    - **天氣適應**：加入數據增強，如隨機亮度、對比度調整、添加霧效或雨滴模擬，提升模型對惡劣天氣的魯棒性。
    - **小物體增強**：使用 mosaic 增強（YOLOv4/v5 引入），將多張圖像拼接，增加小物體的曝光率。
    - **模糊處理**：模擬快速移動的運動模糊（motion blur），訓練模型適應高速車輛場景。
- **圖像分辨率**：
    - 提高輸入分辨率（例如從 416×416 到 640×640），改善小車牌檢測，但需權衡推理速度。
    - 在邊緣設備上，可使用自適應分辨率（根據硬體動態調整輸入大小）。
- **ROI 提取**：
    - 在 LPR 場景中，先使用車輛檢測（較大物體）定位車輛，再對車牌區域進行裁剪和放大，減少背景干擾，提升車牌檢測精度。

#### 5. Object Detection 與 Tracking 的最佳化

- **檢測 (Detection)**：
    - **輕量化模型**：選擇 YOLOv5n、YOLOv8n 或 YOLOv7-tiny，這些模型在邊緣設備上有較低的計算需求，同時保持合理精度。
    - **量化與剪枝**：應用 INT8 量化或濾波器剪枝（filter pruning），減少模型大小和推理時間。例如，YOLOv7-tiny 在量化後參數量可減少約 40%。
    - **小物體專注**：在 loss function 中增加小物體的權重（如 focal loss），提升對車牌的檢測能力。
- **追蹤 (Tracking)**：
    - **追蹤算法**：結合 DeepSORT 或 ByteTrack，提升快速移動車輛的追蹤穩定性。
        - DeepSORT 使用特徵嵌入（appearance embedding）匹配框，適合 LPR 中車牌的連續識別。
        - ByteTrack 利用高低置信度框，提升追蹤魯棒性。
    - **關鍵幀間隔**：在高速場景中，減少追蹤算法的更新間隔（例如每 2 幀更新一次），避免因模糊丟失目標。
    - **硬體加速**：利用邊緣設備的 NPU（如 RK3568 的 1 TOPS）或 GPU 加速追蹤計算。

---

### 針對具體挑戰的解決方案

1. **天氣**：
    - 前處理：圖像去噪（如中值濾波）、對比度增強（CLAHE）。
    - 模型：訓練時加入多天氣數據集（如 RainyCityscapes）。
2. **物體太小**：
    - Grid Size：增加到 26×26 或更高。
    - Anchor：自訂車牌比例。
    - Loss：調整小物體的權重。
3. **速度太快**：
    - 前處理：運動模糊增強。
    - 追蹤：使用 ByteTrack 處理低置信度框。
    - 推理：提高 FPS（如 YOLOv8n 可達 100+ FPS）。
4. **邊緣設備限制**：
    - 模型：選擇 YOLOv8n 或 YOLOv7-tiny（參數量約 6M）。
    - 量化：INT8 或 FP16 推理。
    - 分辨率：動態調整至 320×320 或 416×416。

---

### 實例設計（以 YOLOv8 為例）

- **模型結構**：
    - Backbone：CSPDarknet（輕量化版本）。
    - Neck：E-BiFPN（高效特徵融合）。
    - Head：Anchor-free 檢測（減少計算量）。
- **參數與超參數**：
    - Grid Size：26×26。
    - B=3，Anchor 基於車牌比例。
    - Learning Rate：0.001（cosine 衰減）。
    - Confidence 閾值：0.3，NMS IoU：0.6。
- **前處理**：
    - 輸入：640×640（檢測時可降至 416×416）。
    - 增強：mosaic、亮度調整、運動模糊。
- **推理**：
    - 量化：INT8。
    - 追蹤：ByteTrack（每 2 幀更新）。

---

### 總結

通過調整 **grid size**（增加密度）、**parameters**（自訂 anchor 和 B）、**hyperparameters**（優化 IoU 和 confidence）以及 **前處理**（增強天氣和小物體適應性），YOLO 可以有效應對 Flock Safety 和 LPR 的挑戰。同時，針對邊緣設備，採用輕量化模型和硬體加速技術，能在速度和精度間取得平衡。這些最佳化需根據具體數據集（例如車牌數據）和硬體規格進一步微調，建議在實際應用中測試並迭代。




### YOLO 模型中 NMS（非極大值抑制）的作用

**NMS 的作用**

在 YOLO 模型中，NMS 的主要作用是：

- **消除重複的邊界框：** YOLO 模型會產生大量的邊界框預測，其中許多邊界框會重疊並指向同一個目標。
- **選擇最佳邊界框：** NMS 會根據邊界框的置信度（confidence score）和交並比（IoU），篩選出最佳的邊界框，並抑制其他重複的邊界框。

**NMS 的位置**

- NMS 屬於 YOLO 模型的**後處理（post-processing）**階段，而不是 Backbone、Neck 或 Head 模型的一部分。
- 它在模型輸出所有邊界框預測後執行。

**NMS 如何影響 YOLO**

- **提高檢測精度：** 透過消除重複的邊界框，NMS 可以提高檢測結果的準確性。
- **改善檢測結果的可讀性：** NMS 可以減少檢測結果中邊界框的數量，使結果更清晰易懂。
- **影響檢測速度：** NMS 的計算會增加推論時間，但通常影響不大。

**NMS 出現的 YOLO 版本**

- NMS 是目標檢測中常用的後處理技術，因此它出現在所有 YOLO 版本中，包括 YOLOv1、YOLOv2、YOLOv3、YOLOv4、YOLOv5、YOLOX、YOLOv6、YOLOv7 和 YOLOv8。

**Python 程式碼範例（簡化版）**

以下是一個簡化的 NMS 程式碼範例：

```Python
import torch

def nms(boxes, scores, iou_threshold):
    """
    執行 NMS。

    參數：
        boxes (torch.Tensor): 邊界框座標 (x1, y1, x2, y2)。
        scores (torch.Tensor): 邊界框的置信度。
        iou_threshold (float): IoU 閾值。

    回傳：
        keep (torch.Tensor): 保留的邊界框索引。
    """

    # 按照置信度排序
    sorted_scores, sorted_indices = torch.sort(scores, descending=True)
    keep = []

    while sorted_indices.numel() > 0:
        # 選擇置信度最高的邊界框
        best_index = sorted_indices[0]
        keep.append(best_index)

        # 計算 IoU
        best_box = boxes[best_index]
        remaining_indices = sorted_indices[1:]
        remaining_boxes = boxes[remaining_indices]

        x1 = torch.max(best_box[0], remaining_boxes[:, 0])
        y1 = torch.max(best_box[1], remaining_boxes[:, 1])
        x2 = torch.min(best_box[2], remaining_boxes[:, 2])
        y2 = torch.min(best_box[3], remaining_boxes[:, 3])

        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        best_area = (best_box[2] - best_box[0]) * (best_box[3] - best_box[1])
        remaining_areas = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * (remaining_boxes[:, 3] - remaining_boxes[:, 1])
        union = best_area + remaining_areas - intersection
        iou = intersection / union

        # 移除 IoU 大於閾值的邊界框
        mask = iou <= iou_threshold
        sorted_indices = remaining_indices[mask]

    return torch.tensor(keep)
```

**程式碼說明**

1. **排序：** 按照邊界框的置信度對其進行排序。
2. **選擇最佳邊界框：** 選擇置信度最高的邊界框，並將其添加到保留列表中。
3. **計算 IoU：** 計算最佳邊界框與其他邊界框之間的 IoU。
4. **移除重複邊界框：** 移除 IoU 大於閾值的邊界框。
5. **重複：** 重複步驟 2-4，直到所有邊界框都被處理。

**重要注意事項**

- 實際應用中，您可能需要使用更高效的 NMS 實作，例如基於 GPU 的實作。
- YOLOv8中NMS已經集成到模型的後處理當中，所以不需要單獨寫NMS程式碼。




### YOLO輕量化模型的設計

輕量化模型（例如 YOLOv5n、YOLOv8n）的目標是減少計算複雜度（FLOPs）、參數量和記憶體使用，同時盡量保持檢測精度。這對於邊緣設備（如 Flock Safety 或 LPR 攝影機）至關重要，因為這些設備通常有低功耗和有限的計算資源。輕量化主要通過以下部分實現：

---

### 1. Backbone 的輕量化

**Backbone** 是 YOLO 的特徵提取網絡，負責從輸入圖像中提取多尺度特徵。輕量化模型在這部分做了顯著優化：

- **減少通道數**：
    - 在 YOLOv5/YOLOv8 中，backbone 使用 CSPDarknet（Cross Stage Partial Darknet）結構。
    - 輕量化版本（如 YOLOv5n）將通道數大幅減少。例如：
        - YOLOv5s（小型版）：基礎通道數約為 64。
        - YOLOv5n（超輕量版）：基礎通道數降至 32 或更低。
    - 效果：減少通道數直接降低卷積層的參數量和計算量（參數量與通道數平方成正比）。
- **減少層數**：
    - YOLOv5n 減少了 CSP 模塊的層數。例如，YOLOv5l（大型版）可能有 10+ 個 CSP 塊，而 YOLOv5n 可能只有 3-5 個。
    - 效果：層數減少意味著更少的計算層，降低推理時間。
- **高效卷積**：
    - 使用 **Depthwise Separable Convolution**（深度可分離卷積）替代標準卷積。
    - 標準卷積：每次計算所有輸入通道和輸出通道的點積。
    - Depthwise Separable Convolution：分為 depthwise（逐通道卷積）和 pointwise（1×1 卷積）兩步，減少約 8-9 倍的計算量。
    - YOLOv8n 中部分採用此技術，提升效率。
- **具體例子**：
    - YOLOv5n 的 backbone 參數量約為 1.9M（百萬），而 YOLOv5s 為 7.2M，YOLOv5l 為 46.5M。這種差距主要來自通道數和層數的縮減。

---

### 2. Neck 的輕量化

**Neck** 負責融合 backbone 提取的多尺度特徵（例如 FPN 或 PANet），將特徵傳遞給 head。輕量化模型在這部分的優化包括：

- **簡化特徵融合**：
    - YOLOv5 使用 PANet（Path Aggregation Network），輕量化版本（如 YOLOv5n）減少了融合層的數量。
    - 例如，YOLOv5l 可能融合 3-4 個尺度的特徵（52×52、26×26、13×13），而 YOLOv5n 可能只保留 2 個尺度（26×26、13×13）。
    - 效果：減少特徵融合的計算成本。
- **通道數壓縮**：
    - Neck 中的卷積層通道數也隨 backbone 減少。例如，若 backbone 輸出通道從 512 降到 256，neck 的通道數也相應縮減。
    - 效果：降低參數量和記憶體需求。
- **高效模塊**：
    - YOLOv8n 引入 **E-ELAN**（Efficient Extended-Long-range Attention Network）或簡化的 BiFPN，替代傳統的密集連接，減少冗餘計算。
    - 效果：保持特徵融合能力的同時降低 FLOPs。

---

### 3. Head 的輕量化

**Head** 是 YOLO 的檢測頭，負責將特徵轉換為 bounding box 和類別預測。輕量化模型在這部分的優化如下：

- **減少預測分支**：
    - YOLOv5/YOLOv8 的 head 通常有多個檢測分支（例如 3 個尺度）。輕量化版本可能減少分支數，例如只保留中間尺度（26×26）。
    - 效果：減少輸出張量的維度，降低後處理成本。
- **Anchor-free 設計**（YOLOv8）：
    - 傳統 YOLO 使用 anchor boxes，每個網格預測 B 個框（例如 B=3）。YOLOv8n 引入 anchor-free 設計，每個網格直接預測一個框。
    - 效果：減少參數量（無需預定義 anchor）和計算量（從 B*5 降到 5）。
- **簡化卷積**：
    - Head 中的 1×1 卷積或 3×3 卷積通道數減少，與 backbone 和 neck 保持一致。
    - 效果：降低檢測頭的計算負擔。
- **具體例子**：
    - YOLOv5n 的 head 輸出張量可能從 [batch_size, 3*(5+C), 52, 52]（YOLOv5s）簡化為更小的尺度或更少的預測值。

---

### 4. 其他設計與機制

輕量化不僅限於 backbone、neck 和 head，還包括以下機制：

- **模型剪枝（Pruning）**：
    - 在訓練後移除不重要的濾波器或通道（例如 L1 規範剪枝），YOLOv5n/v8n 常應用此技術。
    - 效果：參數量減少 20%-50%，推理速度提升。
- **量化（Quantization）**：
    - 將模型從 FP32（32 位浮點）轉為 INT8（8 位整數）或 FP16。
    - 效果：減少記憶體使用（約 4 倍）和推理時間，特別適合邊緣設備的 NPU 或 TPU。
- **知識蒸餾（Knowledge Distillation）**：
    - 用大型模型（如 YOLOv5l）訓練小型模型（如 YOLOv5n），讓小模型學習大模型的特徵。
    - 效果：提升輕量模型的精度，彌補結構簡化帶來的損失。
- **高效激活函數**：
    - 使用 SiLU 或 Hard-Swish 替代 ReLU，提升性能同時保持低計算成本。

---

### YOLO 是否有限制輸入 Image Size？

#### 基本原則

- **YOLOv1**：有限制。
    - YOLOv1 使用全連接層（例如 nn.Linear(512 * S * S, 4096)），要求輸入圖像大小固定（例如 448×448）。因為全連接層的輸入維度與特徵圖大小綁定，若輸入大小改變，特徵圖大小也變，無法匹配固定維度的全連接層。
    - 例如，若輸入從 448×448 變為 416×416，7×7 的特徵圖可能變為 6×6，導致 512 * S * S 不匹配。
- **現代 YOLO（如 YOLOv3、v5、v8）**：無嚴格限制。
    - 現代 YOLO 去除全連接層，採用全卷積結構（Fully Convolutional Network, FCN）。輸出張量的大小由輸入圖像大小和下採樣倍數（stride）動態決定。
    - 例如：
        - 輸入 416×416，stride=32，輸出網格為 13×13。
        - 輸入 640×640，stride=32，輸出網格為 20×20。
    - 只要輸入尺寸是 stride 的倍數（通常為 32），模型就能正常運行。

#### 實際限制與注意事項

- **Stride 倍數要求**：
    - YOLO 的 backbone 和 neck 使用多次池化或卷積（stride=2），總下採樣倍數通常為 32。因此，輸入寬度和高度必須是 32 的倍數（例如 320、416、608、640），否則會報錯或需要填充（padding）。
- **Anchor 適配**：
    - 雖然輸入大小可變，但 anchor boxes 是基於訓練時的輸入尺寸設計的。若推理時輸入大小變化太大（例如從 416×416 變到 1280×1280），可能影響檢測精度，需重新調整 anchor。
- **硬體限制**：
    - 邊緣設備記憶體有限，高分辨率輸入（例如 1280×1280）可能超出資源，需動態調整輸入大小。

#### 輕量化模型的輸入處理

- YOLOv5n/v8n 支援動態輸入：
    - 在推理時，可通過 --img-size 參數指定任意符合 stride 的尺寸（例如 320×320 或 416×416）。
    - 模型會自動調整輸出網格大小，例如：
        - 輸入 320×320，輸出 10×10（stride=32）。
        - 輸入 416×416，輸出 13×13。
- 輕量化版本通常偏好較小輸入（例如 320×320），以減少計算量。

---

### 總結

#### 輕量化部分

- **Backbone**：減少通道數、層數，使用高效卷積。
- **Neck**：簡化特徵融合層，壓縮通道數。
- **Head**：減少分支，採用 anchor-free，降低卷積複雜度。
- **其他機制**：剪枝、量化、知識蒸餾、高效激活函數。

#### 輸入 Image Size 限制

- **YOLOv1**：有限制，需固定輸入大小（例如 448×448）。
- **現代 YOLO**：無嚴格限制，輸入需為 stride（通常 32）的倍數，支持動態調整。

在應用於 Flock Safety 或 LPR 時，輕量化模型（如 YOLOv8n）通過上述優化，能在邊緣設備上高效運行，並適應不同輸入大小。建議根據具體硬體和場景（如車牌大小）選擇合適的輸入尺寸（例如 416×416）並測試性能。