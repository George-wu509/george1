
| 分類損失 <mark style="background: #ADCCFFA6;">classification</mark> loss | Image classification<br>Object detection<br>Instance segmentation<br> |
| -------------------------------------------------------------------- | --------------------------------------------------------------------- |
|                                                                      | Cross-Entropy Loss                                                    |
|                                                                      | Focal Loss                                                            |

在物件偵測模型（如 Faster R-CNN）中，Focal Loss 是一種專門設計來解決類別不平衡問題的損失函數，尤其適用於正負樣本比例懸殊的情況。以下是 Focal Loss 的詳細解釋，以及如何在物件偵測模型中使用它：

![[Pasted image 20250521111435.png]]

|                    | 假設training image有一個positive ahchor(真實標籤[0,1])的類別概率是[0.1, 0.9]. 有一個negative ahchor(真實標籤[1,0])的類別概率是[0.8, 0.2].                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Cross-Entropy loss | positive anchor CE: - (0×log(0.1) + 1×log(0.9)) = 0.105<br><br>negative anchor CE: - (1×log(0.8) + 0×log(0.2)) = 0.223<br><br>CEtotal = (0.105 + 0.223)/2        <br><br>p.s 要平均所有的positive and negative anchor<br>可能會有很多的negative anchor會影響loss function<br><br>>>>  **CE(yij,pij) = - sum ( yij log pij )**<br>>>>  ** 所有CE相加(正負) / 正負樣本anchor數量  ** <br>(anchor/proposal based objection detection/instance segmentation)<br><br>>>>  ** 所有CE相加(正負) / 正負樣本邊界框數量  **<br>(anchor free objection detection/instance segmentation)                                      |
| Focal loss         | positive anchor FL: −0.25×(1−0.9)^2 ×log(0.9)=0.00026<br><br>negative anchor FL: −0.75×(1−0.8)^2 ×log(0.8)=0.007<br><br>FLtotal = (0.00026 + 00.007)/2  <br><br>p.s 要平均所有的positive and negative anchor<br>這個容易分類的正樣本負樣本的 Focal Loss 非常小  <br><br>>>>  <mark style="background: #BBFABBA6;">**FL(pt​)=−αt​(1−pt​)^γ log(pt​)** </mark>(αt平衡因子,γ聚焦參數)<br>>>>  ** 所有FL相加(正負) / 正負樣本anchor數量  **<br>(anchor/proposal based objection detection/instance segmentation)<br><br>>>>  ** 所有FL相加(正負) / 正負樣本邊界框數量  **<br>(anchor free objection detection/instance segmentation) |
|                    |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |





**1. Focal Loss 的原理**
**易分類負樣本**: 這些是模型能夠以高置信度正確預測為背景的負樣本。也就是說，模型很容易區分這些區域不是目標物件。 ~ True Negative (預測為negative, 但結果正確)

**難分類負樣本**: 這些是模型容易錯誤預測為目標物件的負樣本。這些區域可能看起來與目標物件相似，或者具有複雜的背景，使得模型難以區分。 ~ False Positive (預測為positive, 但結果錯誤)

- **問題：**
    - 在物件偵測中，背景區域（負樣本）通常遠遠多於目標物件（正樣本）。
    - 這會導致模型在訓練過程中過度關注於大量的易分類負樣本，而忽略了少數但重要的難分類正負樣本。
- **解決方案：**
    -<mark style="background: #FFF3A3A6;"> Focal Loss 通過降低易分類樣本的損失權重，並提高難分類樣本的損失權重，從而使模型更專注於學習難分類樣本。</mark>
    - 它的核心思想是在標準的交叉熵損失函數中引入一個調節因子，該因子會隨著樣本被正確分類的置信度增加而降低。
- **數學公式：**
    - Focal Loss 的數學公式如下：
        - `FL(pt) = -αt (1 - pt)^γ log(pt)`
        - 其中：
            - `pt` 是模型預測的類別概率。
            - `αt` 是一個平衡因子，用於調整正負樣本的權重。
            - `γ` 是一個調節因子，用於調整易難分類樣本的權重。

**2. 在 Faster R-CNN 中使用 Focal Loss**

- 在 Faster R-CNN 中，Focal Loss 主要應用於 R-CNN Head 部分，用於分類候選區域（region proposals）。
- 它取代了標準的交叉熵損失函數，用於訓練分類器。

**3. Python 程式碼實作**

以下是一個簡化的 PyTorch 程式碼範例，展示了如何在 Faster R-CNN 中實作 Focal Loss：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, prediction, target):
        prediction = prediction.clamp(1e-7, 1.0 - 1e-7) #為了數值的穩定加入的clip
        pt = prediction.sigmoid()
        #因為要處理的是物件偵測的問題，所以target需要轉化成one hot encoding。
        one_hot_target = F.one_hot(target, num_classes=prediction.shape[1])
        alpha_t = self.alpha * one_hot_target + (1 - self.alpha) * (1 - one_hot_target)
        loss = -alpha_t * (1 - pt) ** self.gamma * one_hot_target * torch.log(pt) - \
            (1 - alpha_t) * (pt) ** self.gamma * (1 - one_hot_target) * torch.log(1 - pt)
        return loss.sum()

#在Faster R-CNN中的使用範例
#假設 prediction 是模型預測的類別概率，target 是真實標籤。
focal_loss = FocalLoss()
loss = focal_loss(prediction, target)
```

**程式碼解釋：**

- `FocalLoss` 類別定義了 Focal Loss 損失函數。
- `forward` 函數計算損失值。
- `alpha` 和 `gamma` 是 Focal Loss 的超參數，需要根據具體問題進行調整。

**重要注意事項：**

- 超參數調整：`alpha` 和 `gamma` 的最佳值取決於您的數據集。通常，`gamma` 的值在 2 左右，`alpha` 的值在 0.25 左右。
- Focal Loss的使用，要應用在含有類別的loss計算當中，因此要在RCNN head裡面的class loss進行Focal Loss的替換。
- 將 Focal Loss 應用於 Faster R-CNN 的 R-CNN Head 部分，替換標準的交叉熵損失函數。

通過使用 Focal Loss，您可以有效地解決物件偵測中的正負樣本不平衡問題，並提高模型的檢測準確性。


您對於 Hard Negative Mining 和 Focal Loss 的理解基本正確，以下我將更詳細地解釋它們如何解決物件偵測中的正負樣本不平衡問題：

**1. Hard Negative Mining 的作用**

- Hard Negative Mining 專注於選擇對模型來說「困難」的負樣本進行訓練。
- 這些困難的負樣本（Hard Negatives）是模型容易誤判為目標物件的背景區域。
- 通過增加這些困難負樣本在訓練中的比例，模型可以更好地學習區分目標物件和複雜的背景。
- 因此，Hard Negative Mining 的核心是通過樣本選擇來提高模型對難負樣本的判別能力。

**2. Focal Loss 的作用**

- Focal Loss 則通過修改損失函數來調整不同樣本的權重。
- 它降低了容易分類樣本（包括容易分類的負樣本和正樣本）的損失權重，並提高了難以分類樣本的損失權重。
- 通過這種方式，模型在訓練過程中會更加關注難以分類的樣本。
- 也就是説Focal Loss是通過修改損失函數，讓模型更加注重困難樣本的學習。
- 因此，Focal Loss 的核心是通過損失函數的調整來提高模型對難分類樣本的判別能力。

**3. 兩者的比較**

- 雖然 Hard Negative Mining 和 Focal Loss 都旨在提高模型對難分類樣本的判別能力，但它們的方法不同。
    - Hard Negative Mining 通過選擇特定的樣本進行訓練，改變的是參與訓練的樣本構成。
    - Focal Loss 通過調整損失函數的權重，改變的是不同樣本對損失函數的貢獻。
- Focal loss 是調整loss權重， Hard Negative Mining 是調整訓練的樣本構成。

**總結：**

- 您的理解是正確的。Hard Negative Mining 和 Focal Loss 都是為了提高模型對難分類樣本的判別能力。
- Focal Loss 通過調整損失函數的權重，使模型更關注於難分類樣本的學習。
- Hard negative mining 則是選擇困難負樣本放入訓練。




**Focal Loss 的設計理念與解決類別不平衡問題**

在目標檢測任務中，一個常見的挑戰是前景類別（例如：行人、車輛、貓、狗等我們感興趣的目標）的樣本數量通常遠遠少於背景類別（圖像中不包含我們感興趣目標的區域）。這種嚴重的類別不平衡會導致標準的交叉熵損失函數在訓練過程中出現以下問題：

1. **梯度主導 (Gradient Dominance):** 大量的容易分類的負樣本（背景樣本）會產生很大的損失值，從而主導了總體的損失函數。模型在訓練的早期階段很容易被這些簡單的負樣本所驅動，而對那些更難分類的正樣本（前景樣本）的學習不足。
2. **訓練效率低下:** 模型將大量的精力花費在學習已經可以很好分類的負樣本上，而對那些真正需要學習的難樣本（通常是正樣本，但也可能是容易混淆的負樣本）的關注度不夠。

**Focal Loss 的設計核心思想：降低容易分類樣本的損失貢獻**

Focal Loss 的核心思想是通過引入一個**調製因子 (modulating factor)** 來降低那些容易分類的樣本（無論是正樣本還是負樣本）的損失貢獻，從而讓模型更專注於那些難以分類的樣本。

Focal Loss 的公式如下：

FL(pt​)=−(1−pt​)γlog(pt​)

讓我們逐步解析這個公式：

- **pt​:** 這個變數代表了模型預測的屬於真實類別的概率。
    
    - 對於正樣本（真實標籤為 1），pt​ 就是模型預測為正類的概率 p。
    - 對於負樣本（真實標籤為 0），pt​ 就是模型預測為負類的概率 1−p。 因此，pt​ 的取值範圍是 [0,1]。當模型對一個樣本的預測越準確，pt​ 的值就越接近 1。
- **γ (gamma):** 這是一個可調節的**聚焦參數 (focusing parameter)**，γ≥0。這個參數控制了降低容易分類樣本權重的程度。
    
- **(1−pt​)γ:** 這就是**調製因子**。它的作用是：
    
    - **對於容易分類的樣本 (pt​→1):** (1−pt​)γ→0。調製因子趨近於 0，因此這些容易分類的樣本的損失會被極大地降低。
    - **對於難以分類的樣本 (pt​→0):** (1−pt​)γ→1。調製因子趨近於 1，這些難以分類的樣本的損失幾乎不受影響。
- **−log(pt​):** 這就是標準的二元交叉熵損失函數的形式。對於正樣本，我們希望 pt​ 接近 1，−log(pt​) 就接近 0；對於負樣本，我們希望 pt​ 接近 1（即預測為負類的概率接近 1），−log(pt​)=−log(1−p) 也接近 0。
    

**γ 的作用機制**

- 當 γ=0 時，Focal Loss 退化為標準的交叉熵損失函數。
- 當 γ>0 時，調製因子的作用開始顯現。γ 的值越大，容易分類的樣本的損失被降低的程度就越高，模型就越加關注難以分類的樣本。

**如何解決類別不平衡問題**

Focal Loss 通過以下方式來解決類別不平衡問題：

1. **降低負樣本的損失貢獻:** 由於背景樣本通常是容易分類的負樣本，它們的 pt​ 值會比較高（接近 1）。調製因子 (1−pt​)γ 會將這些樣本的損失顯著降低，避免了大量簡單負樣本主導總體損失的情況。
2. **提升難分類樣本的重要性:** 難分類的樣本（通常是前景樣本，但也可能是容易混淆的背景樣本）的 pt​ 值會比較低（接近 0）。調製因子 (1−pt​)γ 對這些樣本的損失影響較小，使得模型在訓練過程中更加關注這些難以分類的樣本，從而提升了模型的性能。

**為了進一步平衡正負樣本的損失，Focal Loss 通常會結合一個平衡因子 α∈[0,1]:**

FL(pt​)=−αt​(1−pt​)γlog(pt​)

其中，αt​ 的定義如下：

- 對於正樣本，αt​=α。
- 對於負樣本，αt​=1−α。

通過調整 α 的值，我們可以給予正樣本更高的權重，從而進一步平衡正負樣本之間的損失貢獻。通常，對於類別不平衡嚴重的問題，α 會設置為一個較小的值（例如 0.25），以增加稀有正樣本的權重。

**使用了 Focal Loss 的常用 AI 模型**

Focal Loss 最初由 Lin 等人在他們的論文 "Focal Loss for Dense Object Detection" 中提出，並被成功應用於 **RetinaNet** 這個單階段目標檢測器中，取得了當時最先進的性能。此後，Focal Loss 被廣泛應用於各種目標檢測模型中，特別是那些旨在處理類別不平衡問題的場景。以下是一些常用的 AI 模型中使用了 Focal Loss 的例子：

1. **RetinaNet:** 這是 Focal Loss 最早也是最著名的應用。RetinaNet 是一個單階段的目標檢測器，它使用 Focal Loss 來解決密集目標檢測中前景和背景類別的極端不平衡問題。
    
2. **其他單階段目標檢測器:** 儘管 YOLO 系列在後來的版本中可能沒有直接使用原始的 Focal Loss，但其設計思想（關注難樣本）在一些變體和相關工作中有所體現。一些其他的單階段檢測器，例如 **EfficientDet** 的某些變體，也可能會考慮使用 Focal Loss 或其變體來提升性能。
    
3. **Anchor-free 目標檢測器:** 像 **FCOS (Fully Convolutional One-Stage Object Detection)** 這樣的 anchor-free 檢測器，由於其密集的預測方式，也面臨著嚴重的類別不平衡問題。因此，Focal Loss 或其變體經常被應用於 FCOS 的損失函數中，以提高檢測性能。
    
4. **一些二階段目標檢測器的變體:** 雖然二階段檢測器（如 Faster R-CNN）通常通過 RoI pooling 等機制來減少類別不平衡，但在某些需要處理極端不平衡的場景下，Focal Loss 也可能被引入到分類分支的損失函數中。例如，在一些針對小目標檢測或長尾分佈數據集的 Faster R-CNN 變體中可能會看到 Focal Loss 的應用。
    
5. **其他視覺任務:** 雖然 Focal Loss 最初是為目標檢測設計的，但其降低容易分類樣本權重、關注難分類樣本的思想也可以應用於其他存在類別不平衡問題的視覺任務中，例如圖像分割、少樣本學習等。
    

**總結**

Focal Loss 通過引入調製因子 (1−pt​)γ 來動態地調整每個樣本的損失權重。對於容易分類的樣本，其損失權重會被降低，而對於難以分類的樣本，其損失權重幾乎不受影響。這種設計使得模型在訓練過程中能夠更專注於那些真正具有挑戰性的樣本，從而有效地解決了目標檢測中由於前景和背景類別數量懸殊而導致的類別不平衡問題，提升了模型的檢測性能，尤其是在那些容易被大量背景樣本淹沒的稀疏前景目標的檢測上。RetinaNet 是 Focal Loss 的一個里程碑式的應用，而其思想也影響了後續許多目標檢測模型和相關研究。





具體舉例解釋圖像目標檢測中 Cross-Entropy Loss 和 Focal Loss 的計算過程，並提供您要求的 COCO 格式的訓練數據集範例。

**1. COCO 格式的訓練數據集範例**

假設我們有一張名為 `image.jpg` 的圖片，其中有三個 "person" 物體。這張圖片的 ID 為 123。以下是符合 COCO 格式的 `annotations` 部分的範例：

JSON

```
{
  "images": [
    {
      "id": 123,
      "file_name": "image.jpg",
      "width": 640,
      "height": 480
    }
  ],
  "annotations": [
    {
      "id": 456,
      "image_id": 123,
      "category_id": 1,
      "bbox": [100, 100, 50, 150],
      "area": 7500,
      "iscrowd": 0
    },
    {
      "id": 789,
      "image_id": 123,
      "category_id": 1,
      "bbox": [250, 50, 80, 200],
      "area": 16000,
      "iscrowd": 0
    },
    {
      "id": 101,
      "image_id": 123,
      "category_id": 1,
      "bbox": [400, 150, 60, 180],
      "area": 10800,
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "person",
      "supercategory": "human"
    }
  ]
}
```

**解釋：**

- **`images`:** 包含圖像的元數據。
    - `id`: 圖像的唯一 ID (123)。
    - `file_name`: 圖像的文件名 (`image.jpg`).
    - `width`: 圖像的寬度 (640 像素)。
    - `height`: 圖像的高度 (480 像素)。
- **`annotations`:** 包含圖像中每個物體的標註信息。
    - `id`: 標註的唯一 ID (456, 789, 101)。
    - `image_id`: 該標註對應的圖像 ID (都是 123)。
    - `category_id`: 物體的類別 ID (1，對應 "person" 類別)。
    - `bbox`: 物體的邊界框，格式為 `[x_min, y_min, width, height]`。
        - 第一個人的邊界框：左上角座標 (100, 100)，寬度 50 像素，高度 150 像素。
        - 第二個人的邊界框：左上角座標 (250, 50)，寬度 80 像素，高度 200 像素。
        - 第三個人的邊界框：左上角座標 (400, 150)，寬度 60 像素，高度 180 像素。
    - `area`: 邊界框的面積。
    - `iscrowd`: 一個標誌，表示該區域是否代表一組擁擠的物體 (0 表示不是)。
- **`categories`:** 定義了數據集中的所有類別。
    - `id`: 類別的唯一 ID (1)。
    - `name`: 類別的名稱 ("person")。
    - `supercategory`: 類別的父類別 ("human")。

**2. 計算 Cross-Entropy Loss 和 Focal Loss 的步驟**

在訓練目標檢測模型時，對於每個生成的 anchor，模型會預測兩個主要部分：

1. **類別概率 (Class Probabilities):** 模型預測這個 anchor 框內包含每個類別的概率。在我們的例子中，如果我們只檢測 "person" 和 "background"，那麼對於每個 anchor，模型會輸出一個包含兩個概率值的向量，例如 `[P(background), P(person)]`。
2. **邊界框偏移量 (Bounding Box Offsets):** 模型預測這個 anchor 框需要調整多少才能更精確地包圍目標物體。這通常表示為相對於 anchor 框的中心、寬度和高度的偏移量。






**我們這裡重點討論類別預測部分的損失計算。**

**步驟 1: 確定正負樣本 Anchor**

在訓練過程中，需要將 100 個生成的 anchor 與圖像中的真實標註框（ground truth bounding boxes）進行匹配，以確定哪些 anchor 是正樣本（負責預測某個真實物體），哪些是負樣本（負責預測背景）。這個匹配過程通常基於 Intersection over Union (IoU)。

- **正樣本 Anchor:** 通常，與任何一個真實標註框的 IoU 大於某個閾值（例如 0.5）的 anchor 會被視為正樣本，負責預測該真實標註框對應的類別。一個真實標註框可能有多個正樣本 anchor 與之對應。
- **負樣本 Anchor:** 通常，與所有真實標註框的 IoU 都小於某個閾值（例如 0.3）的 anchor 會被視為負樣本，負責預測背景類別。
- **忽略的 Anchor:** IoU 介於正負樣本閾值之間的 anchor 通常在訓練中被忽略，不參與損失計算。

**假設我們的 100 個 anchor 中，有以下情況：**

- 5 個 anchor 被分配為正樣本，分別負責預測三個 "person" 物體（可能一個真實物體有多個 anchor 負責）。
- 90 個 anchor 被分配為負樣本（背景）。
- 5 個 anchor 被忽略。


**步驟 2: 計算 Cross-Entropy Loss**

對於每個被分配的正樣本和負樣本 anchor，我們計算其 Cross-Entropy Loss。

**對於一個正樣本 anchor i：**

- **真實標籤 (Ground Truth Label):** 假設這個 anchor i 被分配負責預測第一個 "person" 物體，那麼其真實類別標籤是一個 one-hot 編碼的向量。如果 "person" 的類別 ID 是 1，背景的類別 ID 是 0，那麼真實標籤可能是 [0,1]。
- **模型預測概率:** 假設模型對於這個 anchor i 預測的類別概率是 [P(background)i​,P(person)i​]=[0.1,0.9]。
- **Cross-Entropy Loss (CEi​):**  其中， CEi​=−(0×log(0.1)+1×log(0.9))=−log(0.9)≈0.105

**對於一個負樣本 anchor k：**

- **真實標籤:** 這個 anchor 負責預測背景，所以其真實標籤是 [1,0]。
- **模型預測概率:** 假設模型對於這個 anchor k 預測的類別概率是 [P(background)k​,P(person)k​]=[0.8,0.2]。
- **Cross-Entropy Loss (CEk​):** CEk​=−(1×log(0.8)+0×log(0.2))=−log(0.8)≈0.223

**總體的 Cross-Entropy Loss:**

總體的 Cross-Entropy Loss 是所有被分配的正負樣本 anchor 的損失的平均值（或者加權平均）。在我們的例子中，有 5 個正樣本和 90 個負樣本，所以總損失可能是：

CEtotal​=5+90∑i=15​CEpositive_i​+∑k=190​CEnegative_k​​



**步驟 3: 計算 Focal Loss**

Focal Loss 的公式是：

FL(pt​)=−αt​(1−pt​)γlog(pt​)

其中：

- pt​ 是模型預測的屬於真實類別的概率。
    - 對於正樣本，如果真實類別是 "person"，pt​=P(person)。
    - 對於負樣本，如果真實類別是 "background"，pt​=P(background)。
- γ 是聚焦參數 (通常在 0 到 5 之間，常見值為 2)。
- αt​ 是平衡因子，用於平衡正負樣本的權重。
    - 對於正樣本，αt​=α (例如 0.25)。
    - 對於負樣本，αt​=1−α (例如 0.75)。

**對於我們之前的正樣本 anchor i：**

- 真實類別是 "person"，模型預測 P(person)i​=0.9，所以 pt​=0.9。
    
- 假設 γ=2，$ \alpha = 0.25$，那麼 αt​=0.25。
    
- **Focal Loss (FLi​):** FLi​=−αt​(1−pt​)γlog(pt​)=−0.25×(1−0.9)2×log(0.9)=−0.25×(0.1)2×(−0.105)=0.0002625
    
    可以看到，與 Cross-Entropy Loss (0.105) 相比，這個容易分類的正樣本的 Focal Loss 非常小。
    

**對於我們之前的負樣本 anchor k：**

- 真實類別是 "background"，模型預測 P(background)k​=0.8，所以 pt​=0.8。
    
- 假設 γ=2，$ \alpha = 0.25$，那麼 αt​=1−α=0.75。
    
- **Focal Loss (FLk​):** FLk​=−αt​(1−pt​)γlog(pt​)=−0.75×(1−0.8)2×log(0.8)=−0.75×(0.2)2×(−0.223)=0.00669
    
    與 Cross-Entropy Loss (0.223) 相比，這個相對容易分類的負樣本的 Focal Loss 也顯著減小。
    

**總體的 Focal Loss:**

總體的 Focal Loss 是所有被分配的正負樣本 anchor 的 Focal Loss 的平均值（或者加權平均）：

FLtotal​=5+90∑i=15​FLpositive_i​+∑k=190​FLnegative_k​​

**總結：Focal Loss 的效果**

通過 (1−pt​)γ 這個調製因子，Focal Loss 降低了那些模型能夠以高置信度正確分類的樣本（無論是正樣本還是負樣本）的損失貢獻。γ 的值越大，降低的程度越高。這樣做的目的是讓模型在訓練過程中更專注於那些難以分類的樣本，從而解決類別不平衡問題，並提高模型對難樣本的學習能力。αt​ 平衡因子則進一步調整了正負樣本之間的權重，以應對數量上的不平衡。