
| 掩码損失 <mark style="background: #FF5582A6;">Mask</mark> Loss |                                   |
| ---------------------------------------------------------- | --------------------------------- |
|                                                            | IoU Loss                          |
|                                                            | Dice Loss                         |
|                                                            | Pixel-wise <br>Cross-Entropy Loss |
|                                                            | Focal Loss                        |

| GT      |         |     | Mask   |        |
| ------- | ------- | --- | ------ | ------ |
| 1(back) | 1(back) |     | 0      | 0      |
| 2(car)  | 2(car)  |     | 1(car) | 1(car) |

| Model    |          |     | Mask   |        |
| -------- | -------- | --- | ------ | ------ |
| 0.1(car) | 0.5(car) |     | 0      | 1(car) |
| 0.8(car) | 0.7(car) |     | 1(car) | 1(car) |

|                                   |                                                                                                       |
| --------------------------------- | ----------------------------------------------------------------------------------------------------- |
| IoU Loss                          | IoU Loss = 1 - IoU = 1 - (2/3)<br><br>>>>  **Total IoU Loss = Sum(IoU Loss) all pixel**               |
| Dice Loss                         | Dice Loss = 1 - (2 x 2)/(2+3+e)<br><br>>>>  **Total Dice Loss = Sum(Dice Loss) all pixel**            |
| Pixel-wise <br>Cross-Entropy Loss | CE Loss = -log(0.8)<br><br>>>>  **Total CE Loss = Sum(CE Loss) all pixel**                            |
| Focal Loss                        | Focal Loss = -alpha (1-0.8)^r x log(0.8)<br><br>>>>  **Total Focal Loss = Sum(Focal Loss) all pixel** |


**二、由 Instance 或 Semantic Segmentation Model 的輸出計算損失**

假設對於一個包含 N 個像素的圖像，模型輸出了每個像素屬於 C 個類別的概率分佈 Pij​ (像素 i 屬於類別 j 的概率)，真實標籤為 Yi​∈{1,...,C} (像素 i 的真實類別索引)。

**1. IoU Loss (用於分割)**

在分割任務中，IoU 通常針對每個類別計算。對於類別 c，預測的分割掩碼 Mpredc​ 和真實掩碼 Mgtc​ 都是二元掩碼（像素屬於類別 c 為 1，否則為 0）。

IoUc​=∑i=1N​(Mpredc​(i)+Mgtc​(i)−Mpredc​(i)×Mgtc​(i))∑i=1N​(Mpredc​(i)×Mgtc​(i))​

**IoU Loss 的計算公式為：**

LIoUc​=1−IoUc​

總的 IoU Loss 可以是所有類別 IoU Loss 的平均值。

**具體範例 (針對一個類別):**

假設一個 5x5 的區域，真實掩碼和預測掩碼如下 (1 表示屬於該類別，0 表示不屬於)：





- **交集 (逐元素相乘並求和):** (0×0)+(1×1)+(1×0)+...+(0×0)=6
- **並集 (逐元素相加再減去乘積並求和):** (0+0−0)+(1+1−1)+(1+0−0)+...+(0+0−0)=10
- IoU=106​=0.6
- LIoU​=1−0.6=0.4

**2. Dice Loss**

Dice Loss 也是針對每個類別計算的：

DiceLossc​=1−∑i=1N​Mpredc​(i)+∑i=1N​Mgtc​(i)+ϵ2∑i=1N​(Mpredc​(i)×Mgtc​(i))​

其中 ϵ 是一個很小的數值，用於防止分母為零。

**具體範例 (使用上面的掩碼):**

- ∑i=1N​(Mpredc​(i)×Mgtc​(i))=6 (交集)
- ∑i=1N​Mpredc​(i)=8 (預測為該類別的像素數)
- ∑i=1N​Mgtc​(i)=9 (真實屬於該類別的像素數)

DiceLoss=1−8+9+ϵ2×6​≈1−1712​≈0.294

**3. Pixel-wise Cross-Entropy Loss**

對於每個像素 i，模型輸出一個概率分佈 Pi​ over C 個類別。真實標籤 Yi​ 是該像素的真實類別索引。

LCE​=−N1​i=1∑N​log(Pi,Yi​​)

其中 Pi,Yi​​ 是像素 i 屬於其真實類別 Yi​ 的預測概率。

**具體範例 (一個像素，3 個類別):**

- 模型輸出概率: Pi​=[P(背景)=0.1,P(汽車)=0.8,P(行人)=0.1]
- 真實標籤: Yi​=1 (汽車類別的索引)
- LCE_i​=−log(Pi,1​)=−log(0.8)≈0.223

總的 Pixel-wise Cross-Entropy Loss 是所有像素損失的平均值。

**4. Focal Loss (用於分割)**

Focal Loss 也可以應用於像素級分割，以處理類別不平衡。對於像素 i 屬於類別 j，Focal Loss 的形式為：

FL(Pij​,yij​)=−αj​(1−Pij​)γlog(Pij​)

其中：

- yij​ 是像素 i 屬於類別 j 的真實標籤 (1 如果屬於，0 如果不屬於)。
- Pij​ 是模型預測像素 i 屬於類別 j 的概率。
- γ 是聚焦參數 (通常大於 0)。
- αj​ 是類別權重，用於平衡不同類別的損失。

**具體範例 (一個像素，汽車類別):**

- 真實標籤: yi,1​=1 (是汽車)
- 模型預測概率: Pi,1​=0.8
- 假設 γ=2, α1​=0.25 (汽車類別的權重)
- FL(0.8,1)=−0.25(1−0.8)2log(0.8)=−0.25×0.04×(−0.223)≈0.00223

總的 Focal Loss 是所有像素和所有類別損失的總和或平均值。

希望這些詳細的解釋和具體範例能夠幫助您理解這些常見的損失函數是如何計算的。在實際應用中，這些損失函數通常會在深度學習框架中自動計算。