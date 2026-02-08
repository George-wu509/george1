

|                                           |     |
| ----------------------------------------- | --- |
| [[#### Project 4 - Dialtext]]             |     |
| [[#### Project 5 - Lume]]                 |     |
| [[#### Project 6 - Features]]             |     |
| [[#### Project 7 - Movement dimensions1]] |     |
| [[#### Project 8 - Movement dimensions2]] |     |
|                                           |     |


#### Project 4 - Dialtext




```
以下是一段python script code跟他的yaml config file. 可以從錶面detect OCR及一系列分析. 請根據下列需求根據原始code儘量少變動提供新的python script code跟yaml config file, code comments全部用英文.

(需求)

新增一個parameters TEXT_HEIGHT_THRESHOLD從0~1. 如果detect的text height小於全部text height平均值乘以TEXT_HEIGHT_THRESHOLD, 則刪除這個text.

對每個character, 新增x 跟y方向每隔PROFILE_SPACING的projection profile值並輸出在txt file.

對每個character, 新增沿著skeleton每隔PROFILE_SPACING的thickness值並輸出在txt file.

(yaml config file)
(python script code)
```




#### Project 5 - Lume


```
以下是一段python script code跟他的yaml config file. 可以從image用Example-Based Segmentation (Histogram Backprojection)得到lume的segmentation mask及一系列分析. 請根據下列需求根據原始code儘量少變動提供新的python script code跟yaml config file, code comments全部用英文.

(需求)

新增一個parameter BUILD_MODEL在config file. BUILD_MODEL=true代表由positive reference跟negative referecne建立新的model並用JSON_MODEL_PATH存下來, 並使用這個model計算segmentation mask. BUILD_MODEL=false代表直接讀取JSON_MODEL_PATH載入model並計算segmentation mask.

將得到的最大segmentation mask 嘗試用最長直線或最長曲線去fit輪廓, 也就是如果一段輪廓可能是fit兩個直線線段, 或更長的一段直線線段則取最長的為主. 計算出mask輪廓直線佔總輪廓長的比例(contour straight line ratio), 計算出mask輪廓曲線佔總輪廓長的比例(contour curve line ratio). 請修正接下來的分析. 這裡我們以下面的規則做classifications. 

1. 如果 contour straight line ratio > 30% and contour curve line ratio > 30% 類別則是"hours".

2. 如果 contour straight line ratio > 90% and 兩條或多於兩條直線延長線夾角介於80~100度 類別則是"minutes".  

3. 如果 contour curve line ratio > 90% 類別則是"seconds".

4. 如果 contour straight line ratio > 90% and 一條或小於一條直線夾角介於80~100度 類別則是"GMT". 

如果是直線要記錄頭尾座標, 長度. 如果是曲線則紀錄頭尾座標, 長度, 圓心跟radius. 如果有兩條線以上也記錄兩條直線延長線夾角. 如果有接近平行的(夾角角度<10度)標記是平行並計算兩條線之間距離. 輸出的figures要畫上straight line和curve line在image上, 以及線的頭尾端點 . 另一張圖是曲線可以用圓表示, 畫上那個園以及標上圓心. 這些mask類別以及直線跟曲線各項information可以儲存在txt file.

(yaml config file)
(python script code)
```


#### Project 6 - Features


```

```

#### Project 7 - Movement dimensions1



```
以下是一段python script code跟他的yaml config file. 可以segment雕刻文字3及一系列分析. 請根據下列需求根據原始code儘量少變動提供新的python script code跟yaml config file, code comments全部用英文.

(需求)

對這個segmentation mask後續的skeleton, 新增沿著skeleton每隔PROFILE_SPACING的thickness值並輸出在txt file.

關於這個雕刻文字"3"(大小約1500x1500pixel, thickness約250pixel)有關於skeleton兩條直線(頭跟尾)跟中間輪廓直線計算角度的分析用下面的方法. 雕刻文字"3"的skeleton line應該會有三個頂點,一個是3這個字的頭跟尾, 以及中間. 請在這三個頂點的200pixel(SKELETON_ANALYSIS_SEARCH_RADIUS)範圍內的輪廓上的點, 設法去fit至少100pixel(SKELETON_ANALYSIS_MIN_LINE_LENGTH)長的最長直線, 如果沒有夠長直線則嘗試用曲線去fit. 如果有直線就提供頭尾座標跟長度, 如果是曲線則提供頭尾座標跟長度根圓心跟radius. 並把直線跟曲線標示在圖上並如果有直線計算他們之間夾角. 結果也要輸出在txt file

另外檢查這個mask的skeleton line的1/3*thickness區域在image上有明顯的黑色區域, 則這個mask類型是"type 1", 如果沒有則是"type 2". 結果也要輸出在txt file

(yaml config file)
(python script code)
```

#### Project 8 - Movement dimensions2


```
以下是一段python script code跟他的yaml config file. 可以從image用DINOv3加上pca得到印刷字體的外圍輪廓的Outer contour segmentation mask及一系列分析. 請根據下列需求根據原始code儘量少變動提供新的python script code跟yaml config file, code comments全部用英文.

(需求)
由印刷letter的Outer contour segmentation mask填滿內部就可以得到這個letter的full segmentation mask. 由full segmentation mask減去Outer contour segmentation mask就得到central engraving segmentation mask.

針對每個獨立的letter的full segmentation mask(只取這區域ROI)並再順時針旋轉90度三次成為四個方向的文字ROI. 這時用easyOCR去辨認這個letter是哪個英文或數字. 四張得到最高的confidence就代表是這個letter. 取這個最高的confidence ROI image其他不考慮(譬如如果這個原始letter在順時針90度後辨認"W"並有最高的confidence), 則保留這個順時針90度的所有masks(full segmentation mask, Outer contour segmentation mask, central engraving segmentation mask). 並依旋轉過的full segmentation mask的x_min, x_max, y_min, y_max計算letter height and letter width. 並記錄在txt file

計算旋轉過後的full segmentation mask的skeleton line. 計算skeleton line每隔PROFILE_SPACING的thickness, 以及平均thickness. 並記錄在txt file.

計算每個letter旋轉過後的central engraving segmentation mask的skeleton line. 計算skeleton line每隔PROFILE_SPACING的thickness, 以及平均thickness. 並記錄在txt file.

對每個letter旋轉過後的full segmentation mask, 計算Hu moments, 新增x 跟y方向每隔PROFILE_SPACING的projection profile值並輸出在txt file. 

(yaml config file)
(python script code)
```


