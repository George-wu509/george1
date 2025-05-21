
| 邊界框檢測損失 <br><mark style="background: #FFB86CA6;">Bounding Box Detection</mark> Loss |                |
| ----------------------------------------------------------------------------------- | -------------- |
|                                                                                     | IoU Loss       |
|                                                                                     | L1 Loss        |
|                                                                                     | Smooth L1 Loss |

|                | 對於一個特定的 anchor box，其輸出的預測邊界框為 Bpred​=(100,100,50,50)，而真實標註框為 Bgt​=(120,120,60,60)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| IoU Loss       | Intersection(Bpred, Bgt) = 900<br><br>Union(Bpred, Bgt) = 2500<br><br>IoU Loss = 1 - 900/5200 = 0.827<br><br>>>>  **IoU Loss = 1 - Intersection/Union**<br>>>>  ** 所有IoU Loss相加(正) / 正樣本anchor數量  **  (沒有負)<br>(anchor/proposal based objection detection/instance segmentation)<br><br>>>>  ** 所有IoU Loss相加(正) / 正樣本邊界框數量  ** (沒有負)<br>(anchor free objection detection/instance segmentation)                                                                                                                                                                                                      |
| L1 Loss        | L1 Loss =<br>​<br>= ∣125−150∣ + ∣125−150∣ + ∣50−60∣ + ∣50−60∣<br>= 25+25+10+10=70<br><br>>>>  **L1 Loss = 中心點\|dx\| + 中心點\|dy\| + 寬\|dw\| + 高\|dh\|** <br>>>>  ** 所有L1 Loss相加(正) / 正樣本anchor數量  ** (沒有負)<br>(anchor/proposal based objection detection/instance segmentation)<br><br>>>>  ** 所有L1 Loss相加(正) / 正樣本邊界框數量  ** (沒有負)<br>(anchor free objection detection/instance segmentation)                                                                                                                                                                                                          |
| Smooth L1 Loss | center x loss = \|25\|-0.5<br>center y loss = \|25\|-0.5<br>width loss = \|10\|-0.5<br>height loss = \|10\|-0.5<br><br>Smooth L1 Loss total = 24.5+24.5+9.5+9.5 = 68<br><br>>>> **Smooth L1 Loss** <br>    = <mark style="background: #BBFABBA6;">0.5x^2 if \|x\|<1</mark>\|\|<br>    =<mark style="background: #BBFABBA6;"> \|x\|-0.5</mark>\|\|<br>>>>  ** 所有Smooth L1 Loss相加(正) / 正樣本anchor數量  ** (沒有負)<br>(anchor/proposal based objection detection/instance segmentation)<br><br>>>>  ** 所有Smooth L1 Loss相加(正) / 正樣本邊界框數量  ** (沒有負)<br>(anchor free objection detection/instance segmentation) |


詳細解釋如何根據不同模型的輸出計算 IoU Loss、L1 Loss、Smooth L1 Loss、BCE Loss、Dice Loss、Pixel-wise Cross-Entropy Loss 和 Focal Loss，並提供具體範例。

**一、由 Object Detection Model 的輸出計算損失**

假設我們有一個目標檢測模型，對於一個特定的 anchor box，其輸出的預測邊界框為 $Bpred​=(xpred​,ypred​,wpred​,hpred​)$，而真實標註框為 $Bgt​=(xgt​,ygt​,wgt​,hgt​)$。

**1. IoU Loss (Intersection over Union Loss)**

IoU Loss 直接基於預測框和真實框的交並比 (Intersection over Union)。IoU 的計算公式為：

IoU=Area(Bpred​∪Bgt​)Area(Bpred​∩Bgt​)​=Area(Bpred​)+Area(Bgt​)−Area(Bpred​∩Bgt​)Area(Bpred​∩Bgt​)​

IoU 的值介於 0 到 1 之間，1 表示完美重合，0 表示完全不重疊。

**IoU Loss 的計算公式為：**

LIoU​=1−IoU

**具體範例:**

假設 Bpred​=(100,100,50,50) 和 Bgt​=(120,120,60,60)。

- **計算交集:**
    
    - xoverlap_min​=max(100,120)=120
    - yoverlap_min​=max(100,120)=120
    - xoverlap_max​=min(100+50,120+60)=min(150,180)=150
    - yoverlap_max​=min(100+50,120+60)=min(150,180)=150
    - widthoverlap​=max(0,xoverlap_max​−xoverlap_min​)=max(0,150−120)=30
    - heightoverlap​=max(0,yoverlap_max​−yoverlap_min​)=max(0,150−120)=30
    - Area(Bpred​∩Bgt​)=widthoverlap​×heightoverlap​=30×30=900
- **計算並集:**
    
    - Area(Bpred​)=wpred​×hpred​=50×50=2500
    - Area(Bgt​)=wgt​×hgt​=60×60=3600
    - Area(Bpred​∪Bgt​)=Area(Bpred​)+Area(Bgt​)−Area(Bpred​∩Bgt​)=2500+3600−900=5200
- **計算 IoU:**
    
    - IoU=5200900​≈0.173
- **計算 IoU Loss:**
    
    - LIoU​=1−0.173=0.827

**2. L1 Loss**

L1 Loss 直接計算預測框和真實框中心點以及寬高之間的絕對值差異。

LL1​=∣xpred​−xgt​∣+∣ypred​−ygt​∣+∣wpred​−wgt​∣+∣hpred​−hgt​∣

**具體範例 (使用上面的邊界框):**

假設中心點為 (x+w/2,y+h/2)。

- centerx_pred​=100+50/2=125, centery_pred​=100+50/2=125
- centerx_gt​=120+60/2=150, centery_gt​=120+60/2=150

LL1​=∣125−150∣+∣125−150∣+∣50−60∣+∣50−60∣=25+25+10+10=70

**3. Smooth L1 Loss**

Smooth L1 Loss 是 L1 Loss 的平滑版本，在 0 附近使用平方函數，以减小梯度，防止訓練初期梯度過大。

Lsmooth_L1​(x)={0.5x2∣x∣−0.5​if ∣x∣<1otherwise​

其中，x 可以是 (xpred​−xgt​), (ypred​−ygt​), (wpred​−wgt​), 或 (hpred​−hgt​)。

**具體範例 (使用上面的中心點和寬高差):**

- ∣xpred​−xgt​∣=25>1⟹Lsmooth_L1​(25)=25−0.5=24.5
- ∣ypred​−ygt​∣=25>1⟹Lsmooth_L1​(25)=25−0.5=24.5
- ∣wpred​−wgt​∣=10>1⟹Lsmooth_L1​(10)=10−0.5=9.5
- ∣hpred​−hgt​∣=10>1⟹Lsmooth_L1​(10)=10−0.5=9.5

Lsmooth_L1_total​=24.5+24.5+9.5+9.5=68


