
hard negative - 模型預測錯誤的負樣本(譬如是background但被誤認為object)

|                               |     |
| ----------------------------- | --- |
| [[###anchor box 正負樣本比例不均的問題]] |     |
| [[###Hard Negative Mining]]   |     |
|                               |     |
|                               |     |

### anchor box 正負樣本比例不均的問題

在目標檢測模型（如 Faster R-CNN）訓練中，anchor box 正負樣本比例不均的問題，是目標檢測領域中一個非常常見且重要的挑戰。這種不平衡確實會對模型訓練產生顯著影響。以下我將詳細說明問題的本質、可能產生的影響，以及解決方案：

**問題的本質：Anchor Box 的正負樣本不平衡**

- 在像 Faster R-CNN 這樣的目標檢測模型中，會生成大量的 anchor box 作為候選區域。
- 由於一張圖像中目標物體的數量通常遠少於背景區域，因此絕大多數的 anchor box 都會被判定為負樣本（背景），只有少數幾個與真實標註框（ground truth）重疊度高的 anchor box 會被判定為正樣本（目標物體）。
- 這就導致了極端的正負樣本比例不平衡，例如，幾百個正樣本對上萬個負樣本。

**可能產生的影響：**

- **模型訓練偏差：**
    - 模型在訓練過程中會過度關注於大量的負樣本，因為這些樣本對損失函數的貢獻佔據主導地位。
    - 這會導致模型傾向於將所有候選區域都預測為背景，從而降低了對目標物體的檢測能力。
- **訓練效率低下：**
    - 大量的負樣本會增加計算量，減慢訓練速度。
    - 由於模型主要從負樣本中學習，因此訓練過程的效率不高。
- **模型泛化能力差：**
    - 模型在訓練數據上表現良好，但在實際應用中，由於對目標物體的檢測能力不足，因此泛化能力較差。

**解決方案：**

為了應對這種不平衡問題，研究人員提出了多種解決方案：

- **樣本採樣：**
    - **硬負樣本挖掘 (Hard Negative Mining)：**
        - 在訓練過程中，只選擇那些模型預測錯誤的負樣本進行訓練，也就是那些「難以分辨的負樣本」。
        - 這樣可以讓模型更專注於學習那些容易混淆的背景區域。
    - **隨機採樣：**
        - 減少負樣本的數量。
        - 將正負樣本的比例控制在可接受的範圍內，例如 1:3。
- **損失函數調整：**
    - **Focal Loss：**
        - 通過調整損失函數的權重，使模型更關注於難以分類的樣本（包括正樣本和難負樣本）。
        - Focal Loss 可以有效地減少易分類負樣本對損失函數的貢獻，從而平衡正負樣本的影響。
- **資料增強：**
    - 通過增加影像中較少物件出現的頻率。
    - 通過平移,放大,縮小，遮蔽等等手段增加資料的多樣性，增加正樣本的數量，或者改變物件的比例。

**總結：**

正負樣本比例不平衡是目標檢測模型訓練中一個必須解決的問題。通過採用樣本採樣、損失函數調整或數據增強等方法，可以有效地提高模型的檢測性能。



### Hard Negative Mining

以下是硬負樣本挖掘在目標檢測模型中實作的一般步驟：

- **初始訓練：**
    - 使用初始的正負樣本集訓練模型。
    - 此時，負樣本集可以包含隨機選擇的背景區域。
- **負樣本篩選：**
    - 使用已經訓練的模型，對大量的候選框進行預測。
    - 選擇那些<mark style="background: #BBFABBA6;">模型預測為目標物體，但實際上是背景的候選框，作為「硬負樣本」(hard negative mining)。</mark>
    - 判斷硬負樣本的方式，可以透過這個框的confidence score是否足夠高，來判斷這個負樣本是不是模型容易誤判的負樣本。
- **負樣本集更新：**
    - 將篩選出的硬負樣本加入到負樣本集中。
    - 可以維持一個負樣本的Buffer，設定負樣本的數量上限，每次放入新的負樣本，就將confidence score最低的負樣本給移除。
- **重新訓練：**
    - 使用更新後的負樣本集，重新訓練模型。
    - 重複負樣本篩選和重新訓練的步驟，直到模型性能達到滿意的水平。

**3. 在 PyTorch 中的使用：**

- 在 PyTorch 中，硬負樣本挖掘並不是一個內建的函數或層。
- 它通常作為一個自定義的訓練策略來實現。
- 具體的實現方式會根據目標檢測模型的架構和訓練流程而有所不同。
- Faster R-CNN實作的過程中，可以在RPN layer，與RCNN Head的layer來分別做Hard negative mining。
- 一般實作的流程如下:
    - 在訓練過程中，於每個訓練的batch過後，去篩選出hard negative。
    - 把這些hard negative放進hard negative buffer裡面。
    - 並且在下一次的batch訓練時，從hard negative buffer中，拿出部分或者全部的樣本，合併到原先的負樣本中，一同進行模型的訓練。

解釋:
step1
在每個epoch, 先照run一次 outputs = model(image, target). 
outputs裡面是所有的anchors包括lables=0 or 1
更新weights

step2
所有的anchors, 如果是labels=0而且scores>0.5 則為hard negative samples(難負樣本)
從所有hard negative samples按照buffer size選入hard negative buffer

step3
在這epoch, 再跑一次hard_outputs = model(hard negative buffer, target)
更新weights, 到下一個epoch


**4. 重要注意事項：**

- 硬負樣本挖掘可以有效地提高目標檢測模型的性能，但也會增加訓練的複雜度和計算量。
- 硬負樣本的選擇策略對模型的性能有很大影響，需要根據具體問題進行調整。
- 硬負樣本挖掘可能會導致模型過度擬合於難負樣本，因此需要適當的平衡。

```python
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
import numpy as np
import random
from PIL import Image

# 1. 準備數據集 (您需要替換為您的數據加載邏輯)
def load_data(image_path, annotation_path):
    # 這裡的邏輯需要您根據您的數據格式來實作
    # 返回一個圖片的PIL Image，與標註框的tensor。
    image = Image.open(image_path).convert("RGB")
    #annotation的label 與 boxes， 需要替換成你的annotation reader。
    boxes = torch.tensor([[10, 20, 100, 120], [150, 200, 250, 300]], dtype=torch.float32)  # 示例標註框
    labels = torch.tensor([1, 2], dtype=torch.int64) # 示例標籤

    target = {}
    target["boxes"] = boxes
    target["labels"] = labels
    return image, target

# 2. 建立 Faster R-CNN 模型
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 3  # 背景 + 您的物體類別數量
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.train()

# 3. 準備優化器與損失函數
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# 4. 硬負樣本挖掘的參數
hard_negative_buffer = []  # 存儲難負樣本
hard_negative_buffer_size = 1000 #負樣本buffer的size上限
hard_negative_ratio = 0.5  # 加入到訓練的硬負樣本比例

# 5. 訓練迴圈
num_epochs = 10
for epoch in range(num_epochs):
    #data loader,這部分要替換成讀取您的圖片與annotation的邏輯
    image_path = "./your_image.jpg"
    annotation_path = "./your_annotation.txt"
    image, target = load_data(image_path, annotation_path)

    image_tensor = F.to_tensor(image).unsqueeze(0)
    targets = [{"boxes": target["boxes"], "labels": target["labels"]}]

    # 標準訓練過程
    outputs = model(image_tensor, targets)
    loss = sum(l for l in outputs.values())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 硬負樣本挖掘部分
    model.eval()
    with torch.no_grad():
        proposal = model(image_tensor)
        proposal_boxes = proposal[0]['boxes']
        proposal_labels = proposal[0]['labels']
        proposal_scores = proposal[0]['scores']

    # 篩選難負樣本（基於預測分數）
    hard_negative_candidates = proposal_boxes[(proposal_labels == 0) & (proposal_scores > 0.5)]

    # 把hard negative 放進buffer.
    for box in hard_negative_candidates:
      hard_negative_buffer.append(box.detach())

    #限制buffer的size
    while len(hard_negative_buffer) > hard_negative_buffer_size:
        hard_negative_buffer.pop(0)

    # 創建用於訓練的新targets (包含難負樣本)
    model.train()
    if hard_negative_buffer:
        num_hard_negatives = int(len(hard_negative_buffer) * hard_negative_ratio)
        hard_negatives_to_use = random.sample(hard_negative_buffer, num_hard_negatives)
        all_proposal_boxes = torch.cat([proposal_boxes, torch.stack(hard_negatives_to_use)])
        all_proposal_labels = torch.cat([proposal_labels, torch.zeros(len(hard_negatives_to_use))])

        hard_target = [{"boxes": target["boxes"], "labels": target["labels"],"proposals":all_proposal_boxes,"proposal_labels":all_proposal_labels}]
        hard_output = model(image_tensor,hard_target)
        hard_loss = sum(l for l in hard_output.values())
        optimizer.zero_grad()
        hard_loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# 6. 儲存模型
torch.save(model.state_dict(), "fasterrcnn_trained.pth")
```
**重要注意事項：**

1. **數據集加載:** 您需要根據您的數據格式（例如，COCO、Pascal VOC）來實作 `load_data` 函數。
2. **類別數量:** 確保 `num_classes` 與您的數據集中的物體類別數量一致（包括背景類別）。
3. **超參數調整:** 您可能需要調整學習率、批次大小、難負樣本的篩選閾值等超參數，以獲得最佳性能。
4. **計算資源:** 訓練 Faster R-CNN 需要較高的計算資源，建議使用 GPU。
5. **測試與評估:** 訓練完成後，請使用測試集對模型進行評估，並調整超參數。
6. **簡化版本:** 這個版本的程式碼是高度簡化的版本，許多錯誤處理都省略了，僅供參考，實際部署上請添加需要的錯誤處理程序。
7. **效率**: 實際運用上, hard negative mining 的負樣本Buffer需要使用更加高效率的資料結構來維護.

此程式碼提供了一個基本的硬負樣本挖掘實作框架。您需要根據您的具體需求和數據集來進行調整。