
|                                           |     |
| ----------------------------------------- | --- |
| [[#### Project 4 - Dialtext]]             |     |
| [[#### Project 5 - Lume]]                 |     |
| [[#### Project 6 - Features]]             |     |
| [[#### Project 7 - Movement dimensions1]] |     |
| [[#### Project 8 - Movement dimensions2]] |     |
|                                           |     |




#### Project 4 - Dialtext

|                                                                                | All | Best | Note |
| ------------------------------------------------------------------------------ | --- | ---- | ---- |
| # ----- 1. Import dependent -----                                              | 1   | 1    |      |
| # ----- 2. Project folder and device/device setup -----                        | 1   | 1    |      |
| # ----- 3. images loading: dial text -----                                     | 1   | 1    |      |
| # ----- 4. (version 2)create OCR region segmentation masks and analysis  ----- | 2   | 2    |      |

1031
create OCR region segmentation masks and analysis
```
這個Colab code 用easyOCR跟pytesseract兩階段偵測text. 但在pytesseract有時候在較緊密的字間隔時可能會漏掉字, 以及明明easyOCR已經偵測出一串字譬如"GMT"的box, 但pytesseract都辨識不出來. 或者有時候easyOCR偵測的"PERPETUAL"但pytesseract把"UA"框成一個. 請想辦法解決上述issue並提供code. 

這個colab codefigure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分使用now = datetime.datetime.now(ZoneInfo("America/New_York")). 然後再把這個subfolder名字後面加上"_dialtext_result(v2)". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文. 並注意不要犯這個錯誤: module 'cv2' has no attribute 'COLOR_BGR_RGB'
```
-> ----- 4. (version 2)create OCR region segmentation masks and analysis  -----



#### Project 5 - Lume

|                                                                                    | All | Best | Note |
| ---------------------------------------------------------------------------------- | --- | ---- | ---- |
| # ----- 1. Import dependent -----                                                  | 1   | 1    |      |
| # ----- 2. Project folder and device/device setup -----                            | 1   | 1    |      |
| # ----- 3. images loading: DATA_LUME_PATH -----                                    | 1   | 1    |      |
| # ----- 4. calculate image features summary                                        | 1   | 1    |      |
| # ----- 5. (version 2) Example-Based Segmentation (Histogram Backprojection) ----- | 1,2 | 2    |      |
| # ----- 6. (version 1) Lume mask classification and analysis -----                 | 1   | 1    |      |
|                                                                                    |     |      |      |

1031
Features: Hue, Sarturation contrast looks good, Texture not working
FFT not working
[Action] should try Histogram backprojection using Hue or Sarturation or contrast
[New code result] 


```
對「基於範例的分割（直方圖反向投影）」(Example-Based Segmentation / Histogram Backprojection) 方法的詳細中文解釋。

什麼是直方圖反向投影？
直方圖反向投影 (Histogram Backprojection) 是一種影像分割技術，其核心思想是：使用一個「範例」（Model）來尋找一張「目標影像」（Target Image）中所有與該範例相似的區域。它是一種"有監督" (supervised) 的方法，因為您需要先提供一個您感興趣的物體（即前景，Foreground）的範例。想像一下，您有一個物體（例如一塊特定的藍色布料）的「特徵指紋」。這個「指紋」就是直方圖(Histogram)。然後，您拿著這個指紋去掃描整張目標影像，為每個像素打分，分數高低代表該像素與「指紋」的匹配程度。這張「分數地圖」就是機率圖 (Probability Map)。最後，您設定一個分數門檻（Threshold），高於這個門檻的像素就被認為是您要找的物體。

核心流程詳解
1. 建立「模型」（特徵指紋）
最簡單的反向投影是使用「顏色」。

選定範例 (ROI)： 從一張參考影像中，手動框選一小塊您感興趣的區域 (Region of Interest, ROI)，例如您想找的皮膚、草地或天空。

計算直方圖： 計算這個 ROI 區域的顏色直方圖（例如，在 HSV 色彩空間中，只計算 H (色相) 和 S (飽和度) 的 2D 直方圖）。

正規化 (Normalize)： 將這個直方圖正規化（數值縮放到 0-1 或 0-255 之間）。

這個正規化後的直方圖就是您的模型 (Model)。直方圖中的每個「bin」（區間）現在代表的是「一個像素具有 這個 特徵組合（例如 Hue=50, Sat=100）的機率有多高」。

2. 執行「反向投影」
載入目標影像： 讀取您想要進行分割的完整影像。

遍歷像素： 對目標影像中的每一個像素執行以下操作：

取得該像素的特徵值（例如，它的 H 和 S 值）。

查找模型： 以這個 (H, S) 值作為索引，去您在步驟 1 建立的「模型直方圖」中查找對應的機率分數。

繪製機率圖： 將這個分數寫入到一張新的、空白的單通道影像（機率圖）的相同位置。

取得機率圖： 當所有像素都處理完畢後，您會得到一張灰階影像。這就是反向投影機率圖 (Back-Projection Probability Map)。

在這張圖上，越亮的像素代表其特徵（顏色、紋理等）與您的「範例 ROI」越相似，屬於前景的機率越高。
越暗的像素代表越不相似。

3. 後處理與分割
二值化 (Thresholding)： 對這張機率圖設定一個閾值（Threshold）。

高於閾值的像素設為 255 (白色，前景)。

低於閾值的像素設為 0 (黑色，背景)。

您現在得到了一個二值遮罩 (Binary Mask)。

形態學操作 (Morphological Operations)： 這個遮罩通常會有很多雜訊（小黑點或小白點）。

開啟 (Opening)： (先侵蝕再膨脹) 可以去除小的白色雜訊點（像「鹽巴」）。

關閉 (Closing)： (先膨脹再侵蝕) 可以填補前景物體內部的小黑洞（像「胡椒」）。
```
-> # ----- 5. (version 1) Example-Based Segmentation (Histogram Backprojection) -----

11031521結果:
對很多image可正確segment lume region但有些外面的小顆粒 or 內部有些小區域沒分割到, 只需要用opening or closing處理或改變threshold. 但有些images幾乎所有區域都分割到, 要檢查哪個ROI或bin number, 或threshold可以有最robust的結果適用所有images.

--> (1) open or close (2) smooth contour
```
以下是colab code可以用Example-Based Segmentation / Histogram Backprojection在image上偵測相似區域並建立segmentation mask. 因為這樣生成出來的mask外面有很多小顆粒, 內部也有空洞. 在這個colab cell code請稍微修改加上proprocessing用opening operator並把mask內部的hole填補成為mask, 而如果有多餘一個連續的masks出現則取最大面積為main segmentation mask. 而這個main mask在下一個Colab cell 6會繼續進一步的分析.
```
-> # ----- 5. (version 2) Example-Based Segmentation (Histogram Backprojection) -----



```
在這個Colab cell 6中會判斷這個mask的類別. 有四種mask類別包括1."hours", 2."minutes", 3."seconds", 4."GMT". 可以從計算mask的輪廓去判斷. 計算mask的輪廓然後可以用最長直線跟最長圓型的去嘗試fit mask的輪廓(譬如如果是長方形應該就會有四條直線). 並計算輪廓上的直線跟曲線(可以用有圓心跟固定radius去逼近)的佔比, 以及直線跟直線之間的角度. 另外如果是直線則要儲存兩端點的座標以及長度, 如果是曲線, 則儲存圓心跟固定radius, 以及曲線兩端點.

如果mask的輪廓曲線佔30%以上, 以及輪廓直線佔30%以上, mask則屬於"hours"
如果mask的輪廓曲線佔5%以下, 以及輪廓直線佔95%以上, 而且有兩個直線mask角度是80度到100度之間則屬於"minutes"
如果mask的輪廓曲線佔90%以上, 以及輪廓直線佔10%以下, mask則屬於"seconds"
如果mask的輪廓曲線佔5%以下, 以及輪廓直線佔95%以上, 而且只有一個或沒有直線mask角度是80度到100度之間mask則屬於"GMT"
這些mask類別以及直線跟曲線各項information可以儲存在txt file.

這個新增加的colab code跟上一個colab code一樣figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分使用now = datetime.datetime.now(ZoneInfo("America/New_York")). 然後再把這個subfolder名字後面加上"_lume_analysis_result". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文. 並注意不要犯這個錯誤: module 'cv2' has no attribute 'COLOR_BGR_RGB'
```
-> # ----- 6. (version 1) Lume mask classification and analysis -----

11031521結果:





#### Project 6 - Features

|                                                                          | All | Best | Note |
| ------------------------------------------------------------------------ | --- | ---- | ---- |
| # ----- 1. Import dependent -----                                        | 1   | 1    |      |
| # ----- 2. Project folder and device/device setup -----                  | 1   | 1    |      |
| # ----- 3. images loading: Features -----                                | 1   | 1    |      |
| # ----- 4. calculate image features summary                              | 1   | 1    |      |
| # ----- 5.(version 6) - Image automatic mask generation using SAM  ----- | 1-6 | 6    |      |
|                                                                          |     |      |      |

1031
SAM can do good segmentation but how to decide foreground and background?
DINOv3 sometimes have not bad results in first or third PCA
Features, sometimes texture and saturation works sometimes not
FFT good in some images, but all image type?
[Action] try SAM and think good way to detect foreground

[New code result] 
```
以下是colab code用SAM生成一張圖片的automatic mask generation. 所以一張圖上面會有很多個獨立masks, 是有幾個獨立的字母是frontground, 然後有background. 很多masks可能同屬於background, 也有很多masks是屬於同樣或獨立的frontground. 請依照TextTure(Std Dev)將所有的masks分成數個clusters(background跟frontground), 希望這數個cluster能做到cluster與cluster之間的TextTure差距要遠大於cluster內部的平均TextTure. 分完cluster之後. 如果是分成2個cluster如果這兩組Saturation差距不大譬如是差距是100之內, 則以TextTure(Std Dev)平均最大的cluster為background, 另一個則為frontground. 2個cluster如果這兩組Saturation差距大於100, 則以TextTure(Std Dev)平均最小的cluster為background, 另一個則為frontground. 而如果是分成3個cluster以上則以TextTure(Std Dev)平均最大的cluster為background, 另一個則為frontground. 在這裡先輸出background的segmentation mask(background_mask), 以及background_mask的overlay figure, 以及background_mask的Texture(std dev)數值.

當分完background跟frontground之後先把background的segmentation masks移除, 這時剩下的區域可能包含數個獨立的frontground區域(譬如可能是印刷字體A,B,C三個獨立區域). 這時要選擇在最中心的frontground區域(譬如只有中間的B)並輸出這個獨立segmentation mask(center_mask). 方法是先從中心的1/16*height, 1/16*width區域開始, 看這個區域內含有A,B,C哪個區域的pixel最多, 譬如這個搜尋區域屬於B的pixel最多, 則B的整體segmentation mask為center_mask. 如果1/16*height, 1/16*width區域都沒有frontground, 則搜尋1/8*height, 1/8*width區域, 如果再沒有則搜尋1/4*height, 1/4*width區域, 如果再沒有則搜尋1/2*height, 1/2*width區域. 只要找到center_mask則停止搜尋並輸出center_mask. 以及center_mask的overlay figure.

這個新增加的colab code跟上一個colab code一樣figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分, 然後再把這個subfolder名字後面加上"_dinov3(v2)". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. code儘量function化, code中的comments也都用英文
```
--> # ----- 5.(version 2) - Image automatic mask generation using SAM  -----

11010925結果:
結果發現在分割跟cluster segment背景區域時會找到某些小區域, 所以新增限制條件background一定要大於30% image pixel numbers. 另外結果也發現center_mask有包含background區域, 也重寫code.

```
以下是colab code用SAM生成一張圖片的automatic mask generation. 並找尋background_mask跟center_mask. 請新增輸出SAM的original image with SAM masks overlaid並把code裡面會影響segmentation及結果的parameter集中放在import下方方便設定. 請新增條件background必須是pixel number至少占原image面積30%以上, 如果不達到就以順位下一位為background(譬如以TextTure(Std Dev)平均最大的cluster為background如果pixel只佔原image的10%, 則以TextTure(Std Dev)平均第二大的cluster為background). 另外在從frontground尋找center_mask的步驟, 發現有幾張image的center_mask結果竟包含原來被辨認為background的區域? 請檢查code為何會如此並修正.

這個新增加的colab code跟上一個colab code一樣figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分, 然後再把這個subfolder名字後面加上"_dinov3(v2)". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. code儘量function化, code中的comments也都用英文

```
--> # ----- 5.(version 3) - Image automatic mask generation using SAM  -----

11011024結果:
結果發現:
有時候分成2 cluster時, 會把邊緣(很明顯的邊緣)分成一個cluster但pixel可能只占5%以下. 所以可能要限制初期cluster 有pixel number限制. 另外發現有些frontground一開始SAM就沒有分割到, 應該修改SAM parameter讓分割更細.

```
以下是colab code用SAM生成一張圖片的automatic mask generation. 並找尋background_mask跟center_mask. 請在將SAM 一開始cluster階段並自動決定cluster number階段, 如果有某個cluster有小於10% (新增加這個parameter) pixel面積, 則先排除這部分視為例外, 等到之後計算出background之後再把例外區域加入frontground.另外SAM在初期segment時有些物體沒有被分割出來, 建議調整參數

這個新增加的colab code跟上一個colab code一樣figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分, 然後再把這個subfolder名字後面加上"_dinov3(v2)". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文
```
--> # ----- 5.(version 4) - Image automatic mask generation using SAM  -----

11011024結果:
結果如果刪除例外區域應該重新再clustering, 所以修正

```
以下是colab code用SAM生成一張圖片的automatic mask generation. 並找尋background_mask跟center_mask. 請在將SAM 一開始cluster階段並自動決定cluster number階段, 如果有某個cluster有小於10% (新增加這個parameter) pixel面積, 則先排除這部分視為例外, 這裡排除這部分之後要重新進行自動決定cluster number跟把剩下的區域clustering直到沒有例外區域. 請重修code. 等到之後計算出background之後再把例外區域加入frontground.另外SAM在初期segment時有些物體沒有被分割出來, 建議調整參數

這個新增加的colab code跟上一個colab code一樣figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分使用now = datetime.datetime.now(ZoneInfo("America/New_York")). 然後再把這個subfolder名字後面加上"_dinov3(v2)". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文
```
--> # ----- 5.(version 5) - Image automatic mask generation using SAM  -----

11011024結果:

```
以下是colab code想要增加一點功能. 在最後得到Background mask之後計算background的Texture(std dev), texture(entropy), contrast(range). 以及三種texture類別. "highly textured", "lightly textured", or "smooth". 如果Texture(std dev) >= 50則為textured", 如果20 < Texture(std dev) >50則為"lightly textured", 如果Texture(std dev) <= 20則為"smooth"

這個新增加的colab code跟上一個colab code一樣figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分使用now = datetime.datetime.now(ZoneInfo("America/New_York")). 然後再把這個subfolder名字後面加上"_features_feature result". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文
```
--> # ----- 5.(version 6) - Image automatic mask generation using SAM  -----











#### Project 7 - Movement dimensions1

|                                                                           | All | Best | Note |
| ------------------------------------------------------------------------- | --- | ---- | ---- |
| # ----- 1. Import dependent -----                                         | 1   | 1    |      |
| # ----- 2. Project folder and device/device setup -----                   | 1   | 1    |      |
| # ----- 3. images loading: movement dimensions1 -----                     | 1   | 1    |      |
| # ----- 4. calculate image features summary                               | 1   | 1    |      |
| # ----- 5a. Clone and install DINOv3 repo and load checkpoints  -----     | 1   | 1    |      |
| # ----- 5b. Create DINOv3 model with weights and related functions  ----- | 1   | 1    |      |
| # ----- 5c.(version 3) use self-supervised learning model (DINOv3) -----  | 3   | 3    |      |
| # ----- 5d. (version 12) DINOv3 Mask Analysis -----                       | 12  | 12   |      |

1031
Looks like DINOv3 works great, but sometimes need to automatically decide the “positive” or “negative” value to generate segmentation masks
We found foreground masks are always Texture(std) small and saturation small
[Action]: Should use DINOv3, and may use first PCA and solve positive, negative
[Action]: can use Texture(std) low value to define foreground masks
14:18  Add two colab cells including modified DINOv2 using texture for mask, and another colab cell to calculate skeleton, boundary and remove rectangle.

[New code result] 
```
以下是用DINOv3然後用pca的第一個component生成segmentation masks. 要改動的地方是本來cluster是用lower PC1 value來決定segmentation masks. 現在增加計算兩個cluster每個pixel的texture(std), 哪個cluster是有比較小的texture(std)平均值, 就代表是這張圖片的segmentation mask. 另外也新增這個mask的類別. 如果這個有比較小的texture(std)的cluster也是有lower PC1 value, 則這個類別是"print type 1", 如果比較小的texture(std)的cluster也是big PC1 value則這個類別是"print type 2".

新增接續的下一個colab code cell, 這是利用上面得到的segmentation mask進行下一步analysis. 因為這是一個數字很有可能是3(數字高可能有照片長寬的1/2以上)數字有外面有長方形框都有一定的寬度, 可能還有其他數字的一小部分. 所以利用長方形框的boundary很多是直線來偵測這個長方形框, 如果某一連續塊segmentation mask的boundary幾乎是直線, 從原本segmentation mask移除這個有寬度的長方形框的segmentation mask, 接下來則保留segmentation mask最大連續塊應該就是3, 其他小塊segmentation mask也刪除. 接下來把數字的segmentation mask計算skeleton, 然後計算這個skeleton line到數字mask boundary的距離(thickness)以及數字的boundary. 

這個新增加的colab code跟上一個colab code一樣figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分, 然後再把這個subfolder名字後面加上"_dinov3(v2)". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 並儘量function化, code中的comments也都用英文
```
->  # ----- 8d. DINOv3 Mask Analysis (Frame Removal & Skeleton) -----

11011024結果:
看起來DINOv3 PCA1的結果還不錯, 但請修改find_and_remove_frame() function 裡面的方法不是用多邊形近似輪廓形狀去判斷他是否是矩形框


```
在前一個colab cell用DINOv3然後用pca的第一個component生成segmentation masks. DINOv3生成segmentation mask的部分沒問題. 而這段colab code cell是從原本segmentation mask移除這個有寬度的長方形框的segmentation mask以及後續分析. 但請修改find_and_remove_frame() function 裡面的方法不是用多邊形近似輪廓形狀去判斷他是否是矩形框, 而是得到最外層輪廓之後, 判斷是直線的比例佔最外層輪廓的比例多少去判斷是長方形框或數字. 因為這可能只是長方形框的部分. 從原本segmentation mask移除這個有寬度的長方形框的segmentation mask, 接下來則保留segmentation mask最大連續塊應該就是3, 其他小塊segmentation mask也刪除. 接下來把數字的segmentation mask計算skeleton, 然後計算這個skeleton line到數字mask boundary的距離(thickness)以及數字的boundary. 進一步分析數字3個外層輪廓, 把外層輪廓的直線部分跟明顯轉折點以及轉折的角度都計算出來並輸出, 多增加figure把直線部分跟明顯轉折點也標示在figure上

這個新增加的colab code跟上一個colab code一樣figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分使用now = datetime.datetime.now(ZoneInfo("America/New_York")). 然後再把這個subfolder名字後面加上"_dinov3(v2)". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文.
```
->  # ----- 8d. (version 2) DINOv3 Mask Analysis (Frame Removal & Skeleton) -----

11011024結果:
輸出的segmentation mask不夠精細不能支持後續的分析. 用2 step 方式DINOv3 -> otsu 得到精細的
segmentation mask在原圖尺寸上


```
請幫我分析這段code, 因為這段code涉及將image縮小到size=768當成dinov3的輸入, 但在接續的下一個Colab會從這個colab cell得到的結果generated_masks繼續後續的分析, 但這段code是否有保留原始image的height跟height? 如果有的話在下一個colab要如何讀取? 如果沒有請修改code 保留這個原始image的height跟height

已修改上一個colab cell code 保留在讀取影像時捕捉原始的 (height, width) 可以在

for img_path, (initial_mask, original_image_np, safe_filename, original_shape) in tqdm(generated_masks.items(), ...):

original_h, original_w = original_shape

知道每個image原始的 (height, width). 因為在下列的code裡從原本segmentation mask移除這個有寬度的長方形框的segmentation mask, 接下來則保留segmentation mask最大連續塊應該就是3, 其他小塊segmentation mask也刪除. 對這個只保留3的mask的image(IMAGE_SIZE = 768)我們要回復成原始的 (height, width)大小. 接下來我們要在這個已經恢復原始大小的3的segmentation mask先用pixel=5的dilation morphology filter擴大, 然後在這個新的segmentation mask區域內對原圖用otsu method得到精細版本的segmentation mask. 接下來把數字的segmentation mask計算skeleton, 然後計算這個skeleton line到數字mask boundary的距離(thickness)以及數字的boundary. 進一步分析數字3個外層輪廓, 把外層輪廓的直線部分跟明顯轉折點以及轉折的角度都計算出來並輸出, 多增加figure把直線部分跟明顯轉折點也標示在figure上

這個新增加的colab code跟上一個colab code一樣figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分使用now = datetime.datetime.now(ZoneInfo("America/New_York")). 然後再把這個subfolder名字後面加上"_dinov3(v2)". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文.
```
-> # ----- 8c.(version 3) use self-supervised learning model (DINOv3) -----
-> # ----- 8d. (version 3) DINOv3 Mask Analysis (Frame Removal & Skeleton) -----

110113251結果:
DINOv3得到3跟矩形框沒有問題, 但要將矩形框移除還是有問題, 另外用otsu去做精細的segmentation mask似乎也效果不好

```
在下面code試過偵測矩形的效果都不好. 改成計算segmentation mask連續塊中, 計算連續塊內最大最小的x跟y, 如果x方向距離跟y方向距離相加為最大則為有寬度的長方形框的segmentation mask, 從原來的segmentation mask移除之後接下來則保留segmentation mask最大連續塊應該就是3, 其他小塊segmentation mask也刪除. 對這個只保留3的mask的image(IMAGE_SIZE = 768)我們要回復成原始的 (height, width)大小. 接下來我們要在這個已經恢復原始大小的3的segmentation mask先用pixel=10的dilation morphology filter擴大, 然後在這個新的segmentation mask區域以每個pixel的Texture(std dev)為基準(3的區域應該是texture較低, 3的外面區域texture較高). 從segmentation mask外圍向內部包圍得到新的3的精細segmentation mask. 接下來把數字的segmentation mask計算skeleton, 然後計算這個skeleton line到數字mask boundary的距離(thickness)以及數字的boundary. 進一步分析數字3個外層輪廓, 把外層輪廓的直線部分跟明顯轉折點以及轉折的角度都計算出來並輸出, 多增加figure把直線部分跟明顯轉折點也標示在figure上

這個新增加的colab code跟上一個colab code一樣figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分使用now = datetime.datetime.now(ZoneInfo("America/New_York")). 然後再把這個subfolder名字後面加上"_dinov3(v2)". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文.
```
-> # ----- 8d. (version 4) DINOv3 Mask Analysis (Frame Removal & Skeleton) -----

11011531結果:
矩形框正確移除及其他小塊區域也可移除, 3已經可以在未恢復原圖size的image得到segmentation mask. 但之後的運算應該都完全在原始image的size下運算. 


```
矩形框正確移除及其他小塊區域也可移除, 3已經可以在未恢復原圖size的image得到segmentation mask. 但之後的運算應該都完全在原始image的size下運算包括原始image以及segmentation mask的image. 包括在原始圖大小的segmentation mask image先用pixel=10的dilation morphology filter處理, 然後在這個新的segmentation mask區域以每個pixel的Texture(std dev)為基準(3的區域應該是texture較低, 3的外面區域texture較高). 從segmentation mask外圍向內部包圍得到新的3的精細segmentation mask. 最後再針對segmentation mask的輪廓做平滑化為最終的segmentation mask. 在這計算得到最終的segmentation mask的過程盡量展示過程每一個figure都要輸出image size. 接下來把數字的最終的segmentation mask計算skeleton, 然後計算這個skeleton line到數字mask boundary的距離(thickness)以及數字的boundary. 進一步分析數字3個外層輪廓, 把外層輪廓的直線部分跟明顯轉折點以及轉折的角度都計算出來並輸出, 多增加figure把直線部分跟明顯轉折點也標示在figure上. 

這個新增加的colab code跟上一個colab code一樣figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分使用now = datetime.datetime.now(ZoneInfo("America/New_York")). 然後再把這個subfolder名字後面加上"_dinov3(v2)". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文.
```
-> # ----- 8d. (version 5) DINOv3 Mask Analysis (Frame Removal & Skeleton) -----

11011531結果:
在3的計算精細segmentation mask仍然有patch化, 但它顯示是操作在原始image上? 或者我們用texture map應該用gaussian 平滑化之後再去計算新的segmentation mask? 另外texture計算太久, 或者用其他的features?

```
矩形框正確移除及其他小塊區域也可移除, 3已經可以在未恢復原圖size的image得到segmentation mask. 但之後的運算應該都完全在原始image的size下運算包括原始image以及segmentation mask的image. 包括在原始圖大小的segmentation mask image先平滑化segmentation mask的邊緣, 再用pixel=40的dilation morphology filter處理, 然後在這個新的segmentation mask區域以每個pixel的contrast為基準(3的區域應該是contrast較低, 3的外面區域contrast較高)先建立contrast map, 並用大小為10的gaussian filter平滑化contrast map, 再從segmentation mask外圍向內部包圍得到新的3的精細segmentation mask. 最後再針對segmentation mask的輪廓做平滑化為最終的segmentation mask. 在這計算得到最終的segmentation mask的過程盡量展示過程每一個figure都要輸出image size. 接下來把數字的最終的segmentation mask計算skeleton, 然後計算這個skeleton line到數字mask boundary的距離(thickness)以及數字的boundary. 進一步分析數字3個外層輪廓, 把外層輪廓的直線部分跟明顯轉折點以及轉折的角度都計算出來並輸出, 多增加figure把直線部分跟明顯轉折點也標示在figure上. 

這個新增加的colab code跟上一個colab code一樣figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分使用now = datetime.datetime.now(ZoneInfo("America/New_York")). 然後再把這個subfolder名字後面加上"_dinov3(v2)". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文.
```
-> # ----- 8d. (version 6) DINOv3 Mask Analysis (Frame Removal & Skeleton) -----

11011531結果:
結果有比較平滑化了, 而在之後用


```
矩形框正確移除及其他小塊區域也可移除, 3已經可以在未恢復原圖size的image得到segmentation mask. 但之後的運算應該都完全在原始image的size下運算包括原始image以及segmentation mask的image. 包括在原始圖大小的segmentation mask image先平滑化segmentation mask的邊緣, 再用pixel=40的dilation morphology filter處理, 然後在這個新的segmentation mask區域以每個pixel的contrast為基準(3的區域應該是contrast較低, 3的外面區域contrast較高)先建立contrast map, 並用大小為20的gaussian filter平滑化contrast map, 再從segmentation mask外圍向內部包圍得到新的3的精細segmentation mask. 最後再針對segmentation mask用大小為10的opening filter之後再對輪廓做平滑化為最終的segmentation mask. 接下來把數字的最終的segmentation mask計算skeleton, 然後計算這個skeleton line到數字mask boundary的距離(thickness)以及數字的boundary. 進一步分析數字3外層輪廓, 把外層輪廓的直線部分跟明顯轉折點以及轉折的角度都計算出來並輸出. 顯示的figure只要顯示initial DINOv3 mask的overlap figure, Resized coarse mask的overlap figure, Final smoothed mask的overlap figure, skeleton on final mask, 以及thickness map.

這個新增加的colab code跟上一個colab code一樣figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分使用now = datetime.datetime.now(ZoneInfo("America/New_York")). 然後再把這個subfolder名字後面加上"_dinov3(v2)". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文.
```
-> # ----- 5d. (version 7) DINOv3 Mask Analysis (Frame Removal & Skeleton) -----

```
矩形框正確移除及其他小塊區域也可移除, 3已經可以在未恢復原圖size的image得到segmentation mask. 但之後的運算應該都完全在原始image的size下運算包括原始image以及segmentation mask的image. 包括在原始圖大小的segmentation mask image先平滑化segmentation mask的邊緣, 再用pixel=40的dilation morphology filter處理, 然後在這個新的segmentation mask區域以每個pixel的contrast為基準(3的區域應該是contrast較低, 3的外面區域contrast較高)先建立contrast map, 並用大小為20的gaussian filter平滑化contrast map, 再從segmentation mask外圍向內部包圍得到新的3的精細segmentation mask. 在這裡對segmentation mask用大小為10的opening filter之後再對輪廓做平滑化, 再用Stage 1：Morphological closing 修補凹痕, 去除 jagged edges、鋸齒或過於複雜的邊界, 平滑後的邊界貼合真實影像邊緣，而不只是幾何平均化(使用 DenseCRF (Krähenbühl & Koltun))為最終的segmentation mask. 

接下來把數字的最終的segmentation mask計算skeleton, 然後計算這個skeleton line到數字mask boundary的距離(thickness)以及數字的boundary. 進一步分析數字3外層輪廓, 把外層輪廓的直線部分跟明顯轉折點以及轉折的角度都計算出來並輸出. 顯示的figure只要顯示initial DINOv3 mask的overlap figure, Resized coarse mask的overlap figure, Final smoothed mask的overlap figure, skeleton on final mask, 以及thickness map.

這個新增加的colab code跟上一個colab code一樣figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分使用now = datetime.datetime.now(ZoneInfo("America/New_York")). 然後再把這個subfolder名字後面加上"_movement1_dinov3_bone_result". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文.
(version 8)

那我想稍微修改這段Colab code, 除了已經儲存pca3 value image之外, 新增加可以把每張image儲存pca3 value image在matlab讀取並轉換成pca直接輸出數值需要的數字(譬如min 和 max 值)統一存在txt file裡並存在同一個folder. 本來在Colab輸出的一些cluster資訊也可以存在同個txt file裡.
(version 9)

```
-> # ----- 5d. (version 10) DINOv3 Mask Analysis (Frame Removal & Skeleton) -----

11011531結果:
試驗結果對一些case有很平滑的輪廓, 但某些case還是凹下去或形狀奇怪. 需要繼續改進平滑化. 


```
這個Colab code新增及修改一些功能請提供新的code: 對最後得到的final mask得到輪廓線, 並用直線和曲線去fit這條輪廓線(某些線段是直線, 有些線段是曲線有圓心跟固定radius), 記錄下每個直線段的頭尾座標以及距離, 直線部分跟明顯轉折點以及轉折的角度都計算出來. 曲線的頭尾座標, 圓心座標跟radius以及線段長. 接著計算mask的skeleton以及這個skeleton line到數字mask boundary的距離(thickness), 可以用在skeleton每隔一段距離的座標點並記下thickness. 還有skeleton line的頭跟尾各有一條接近垂直於skeleton line的直線, 輸出這兩條直線(頭跟尾)的夾角. 另外檢查這個mask的skeleton line的1/3*thickness區域在image上有明顯的黑色區域, 則這個mask類型是"type 1", 如果沒有則是"type 2"

這個colab code一樣figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分使用now = datetime.datetime.now(ZoneInfo("America/New_York")). 然後再把這個subfolder名字後面加上"_movement1_dinov3_bone_result". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文.


```
-> # ----- 5d. (version 11) DINOv3 Mask Analysis -----



```
這個Colab code新增及修改一個功能: 在已得到的skeleton, 找尋skeleton中間最靠近的中間輪廓直線. 並計算之前計算過的skeleton兩條直線(頭跟尾)接近垂直於skeleton line的直線跟中間輪廓直線的夾角

這個colab code一樣figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分使用now = datetime.datetime.now(ZoneInfo("America/New_York")). 然後再把這個subfolder名字後面加上"_movement1_dinov3_bone_result(v12)". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文.
```
-> # ----- 5d. (version 12) DINOv3 Mask Analysis -----










#### Project 8 - Movement dimensions2

|                                                                                   | All | Best | Note |
| --------------------------------------------------------------------------------- | --- | ---- | ---- |
| # ----- 1. Import dependent -----                                                 | 1   | 1    |      |
| # ----- 2. Project folder and device/device setup -----                           | 1   | 1    |      |
| # ----- 3. images loading: movement dimensions1 -----                             | 1   | 1    |      |
| # ----- 4. calculate image features summary                                       | 1   | 1    |      |
| # ----- 5a. Clone and install DINOv3 repo and load checkpoints  -----             | 1   | 1    |      |
| # ----- 5b. Create DINOv3 model with weights and related functions  -----         | 1   | 1    |      |
| # ----- 5c.(version 8) use self-supervised learning model (DINOv3) -----          | 8   | 8    |      |
| # ----- 5d. (version 1) Smooth  DINOv3 segmentation mask and follow analysis----- | 1-7 | 3    |      |
| ----- 5e. (version 1) DINOv3 fine segmentation Text Mask Analysis -----           | 1   | 1    |      |

1031
DINOv3 first and second PCA affect by light and texture, but 3nd PCA may works, still have positive and negative issue
SAM not working
Feature: Hue, Saturation, Texture all not working
For positive or negative issue, maybe try Texture(entropy) to decide text boundary or OCR to decide
[Action]: Should try DINOv3, and may use 3nd PCA and solve positive, negative
15:03 Add tow colab cells including modified DINOv3 using pca 3 to create segmentation masks, and use two step OCR and find skeleton bone.

[New code result] 
```
以下是用DINOv3然後用pca的第一個component生成segmentation masks. 要改動的地方是本來cluster是用lower PC1 value來決定segmentation masks. 現在改成用pca的principal component 3用clustering分成兩個cluster, 然後取pixel number較少的為這張圖的segmentation mask. 

新增接續的下一個colab code cell, 這是利用上面得到的segmentation mask進行下一步analysis. 因為這些是有一定的寬度的數字以及英文字, 但可能是直的或橫的. 我們把這張image的segmentation mask 90度旋轉3次在四個角度都用easy OCR偵測數字或英文字, 以偵測到以及confidence最高的那個角度為主而去辨識數字或英文字. 之後用第二步OCR用pytesseract在easyOCR的detection box裡面偵測character level 數字以及英文字, 並用segmentation mask計算x_min, x_max, y_min, y_max可得到字的height跟width. 接下來把字的segmentation mask計算skeleton, 然後計算這個skeleton line到數字mask boundary的距離(thickness)以及數字的boundary. 

這個新增加的colab code跟上一個colab code一樣figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分, 然後再把這個subfolder名字後面加上"_dinov3(v2)". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. code儘量function化, code中的comments也都用英文
```



```
以下是用DINOv3然後用pca的principal component 3用clustering分成兩個cluster, 然後取pixel number較少的為這張圖的segmentation mask. 現在改成自動決定clustering數(最大cluster number=5)希望cluster之間的principal component 3 value差距要遠大於cluster內部的principal component 3 value. 而且每個cluster pixel number不得小於全部image pixel number的10%. 接著計算這些clusters和四個邊緣(上下左右)有pixel接觸的邊數. 譬如如果這個cluster有和上邊界跟左邊界相鄰, 則邊數=2. 計算完之後選邊數等於0或1的cluster中有最大面積數的cluster, 當成segmentation mask. 接著將這個把這個segmentation mask從IMAGE_SIZE復原到原始image的大小. 並保留這個原始image大小的segmentation mask跟原始image以供下一個colab cell使用. 

這個新增加的colab code跟上一個colab code一樣figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分使用now = datetime.datetime.now(ZoneInfo("America/New_York")). 然後再把這個subfolder名字後面加上"_movement2_dinov3_result". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文.
(version 3)

請在檢查方法跟code, 因為最後的segmentation mask很明顯有和邊緣的兩個邊或三個邊相接, 明顯和我的需求不合, 請再重寫
(version 4)

用上面的code執行的結果, 有的可以得到正確的segmentation mask, 有的完全沒有mask. 請新增figure, 把image用自動cluster的所有的cluster的mask的overlap figure都顯示, 並在colab顯示每個cluster的mean principal component 3 value, 面積以及邊數都顯示. 用這樣檢查問題在哪裡. 其他地方都不變請提供新的code
(version 5)

all 3 PC3 clusters (for debugging)中的masks太不明顯, 請用不同的透明顏色, 綠, 藍, 紅, 黃, 橙色並標示是哪個cluster (0,1,2).... 請提供新的code
(version 6)
```
-> # ----- 8c. (version 6) use self-supervised learning model (DINOv3) -----

11012312結果:
目前用DINOv3 pca3生出的segmentation mask有的是可以的, 有的是完全沒有mask. 多加檢查



```
以下是用DINOv3然後用pca的principal component 3自動決定clustering數(最大cluster number=5)希望cluster之間的principal component 3 value差距要遠大於cluster內部的principal component 3 value. 現在改成先用radius=1的gaussian filter用在這張圖的principal component 3 value map, 如果在這map value大於0的面積大於value小於0的面積, 則map value大於0的區域都是background, 只把跟value小於0的連續區域分割出來為frontground. 如果在這map value大於0的面積小於value小於0的面積, 則map value小於0的區域都是background, 只把跟value大於0的連續區域分割出來為frontground. 接下來frontground應該由幾個獨立的連續區域組成, 檢查如果pixel number小於全部image pixel number的10%則刪除, 接著計算這些獨立的連續區域和四個邊緣(上下左右)有pixel接觸的邊數. 譬如如果這個區域有和上邊界跟左邊界相鄰, 則邊數=2. 計算完之後選邊數等於0或1的區域就是segmentation mask. 接著將這個把這個segmentation mask從IMAGE_SIZE復原到原始image的大小. 並保留這個原始image大小的segmentation mask跟原始image以供下一個colab cell使用. 

這個新增加的colab code跟上一個colab code一樣figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分使用now = datetime.datetime.now(ZoneInfo("America/New_York")). 然後再把這個subfolder名字後面加上"_movement2_dinov3_result". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文.
```
-> # ----- 8c. (version 7) use self-supervised learning model (DINOv3) -----

11012312結果:
gaussian選擇1應該比較適合, 不過出來的結果完全沒有masks. 進一步檢查


```
以下是用DINOv3然後用pca的principal component 3自動決定clustering數(最大cluster number=5)希望cluster之間的principal component 3 value差距要遠大於cluster內部的principal component 3 value. 現在改成先用radius=1的gaussian filter用在這張圖的principal component 3 value map, 接著在這map再取 Laplacian，邊界就是 zero-crossing 的地方做邊界 + flood fill 建獨立區域. 這些區域是frontground. 接著檢查這些區域如果pixel number小於全部image pixel number的10%則刪除, 接著計算這些獨立的連續區域和四個邊緣(上下左右)有pixel接觸的邊數. 譬如如果這個區域有和上邊界跟左邊界相鄰, 則邊數=2. 計算完之後選邊數等於0或1的區域就是segmentation mask. 接著將這個把這個segmentation mask從IMAGE_SIZE復原到原始image的大小. 並保留這個原始image大小的segmentation mask跟原始image以供下一個colab cell使用. 請多提供一個figure包含這些獨立區域的overlap用不同的透明顏色(綠, 藍, 紅, 橙色...). 

這個新增加的colab code跟上一個colab code一樣figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分使用now = datetime.datetime.now(ZoneInfo("America/New_York")). 然後再把這個subfolder名字後面加上"_movement2_dinov3_result". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文.
(version 8)

為什麼執行這個code只有顯示三個pca的一張圖跟原圖跟combined pca的一張圖. 然後計算每個image 顯示這個: 
cv2.error: OpenCV(4.12.0) /io/opencv/modules/imgproc/src/filter.simd.hpp:3250: error: (-213:The function/feature is not implemented) Unsupported combination of source format (=5), and destination format (=6) in function 'getLinearFilter'
(version 9)

```
-> # ----- 8c. (version 8) use self-supervised learning model (DINOv3) -----

11012312結果:
用LoG無法準確切割邊界, 會有太多邊界


```
以下是用DINOv3然後用pca的principal component 3自動決定clustering數(最大cluster number=5)希望cluster之間的principal component 3 value差距要遠大於cluster內部的principal component 3 value. 現在改成先用radius=1的gaussian filter用在這張圖的principal component 3 value map, 接著用map>0跟map<0可以在image切出很多獨立區域. 接著檢查這些區域如果pixel number小於全部image pixel number的10%則刪除, 接著計算這些獨立的連續區域和四個邊緣(上下左右)有pixel接觸的邊數. 譬如如果這個區域有和上邊界跟左邊界相鄰, 則邊數=2. 計算完之後選邊數等於0或1的區域就是segmentation mask. 接著將這個把這個segmentation mask從IMAGE_SIZE復原到原始image的大小. 並保留這個原始image大小的segmentation mask跟原始image以供下一個colab cell使用. 請多提供一個figure包含這些獨立區域的overlap用不同的透明顏色(綠, 藍, 紅, 橙色...). 

這個新增加的colab code跟上一個colab code一樣figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分使用now = datetime.datetime.now(ZoneInfo("America/New_York")). 然後再把這個subfolder名字後面加上"_movement2_dinov3_result". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文. 並注意不要犯這個錯誤: module 'cv2' has no attribute 'COLOR_BGR_RGB'
```
-> # ----- 8c. (version 9) use self-supervised learning model (DINOv3) -----

11012312結果:
用單純threshold=0也無法準確切割邊界


回到用Version 6
```
以下是用DINOv3然後用pca的principal component 3用clustering分成兩個cluster, 然後取pixel number較少的為這張圖的segmentation mask. 現在改成自動決定clustering數(最大cluster number=5)希望cluster之間的principal component 3 value差距要遠大於cluster內部的principal component 3 value. 而且每個cluster pixel number不得小於全部image pixel number的10%. 接著計算這些clusters和四個邊緣(上下左右)有pixel接觸的邊數. 譬如如果這個cluster有和上邊界跟左邊界相鄰, 則邊數=2. 計算完之後選邊數等於0或1的cluster中有最大面積數的cluster, 當成segmentation mask. 接著將這個把這個segmentation mask從IMAGE_SIZE復原到原始image的大小. 並保留這個原始image大小的segmentation mask跟原始image以供下一個colab cell使用. 

這個新增加的colab code跟上一個colab code一樣figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分使用now = datetime.datetime.now(ZoneInfo("America/New_York")). 然後再把這個subfolder名字後面加上"_movement2_dinov3_result". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文.
(version 3)

請在檢查方法跟code, 因為最後的segmentation mask很明顯有和邊緣的兩個邊或三個邊相接, 明顯和我的需求不合, 請再重寫
(version 4)

用上面的code執行的結果, 有的可以得到正確的segmentation mask, 有的完全沒有mask. 請新增figure, 把image用自動cluster的所有的cluster的mask的overlap figure都顯示, 並在colab顯示每個cluster的mean principal component 3 value, 面積以及邊數都顯示. 用這樣檢查問題在哪裡. 其他地方都不變請提供新的code
(version 5)

all 3 PC3 clusters (for debugging)中的masks太不明顯, 請用不同的透明顏色, 綠, 藍, 紅, 黃, 橙色並標示是哪個cluster (0,1,2).... 請提供新的code
(version 6)
```
-> # ----- 8c. (version 6) use self-supervised learning model (DINOv3) -----



```
以下是用DINOv3然後用pca的principal component 3自動決定clustering數(最大cluster number=5)希望cluster之間的principal component 3 value差距要遠大於cluster內部的principal component 3 value. 現在改成先用radius=1的gaussian filter用在這張圖的principal component 3 value map, 接著用map>0跟map<0可以在image切出很多獨立區域. 接著檢查這些區域如果pixel number小於全部image pixel number的10%則刪除, 接著計算這些獨立的連續區域和四個邊緣(上下左右)是否有pixel接觸. 如果有接觸的邊界是相鄰的, 譬如(上邊界跟左邊界or上邊界跟右邊界), 則這區域刪除. 如果有接觸的邊界是不相鄰的或只有一個邊界或0個邊界接觸, 則區域加入segmentation mask. 接著將這個把這個segmentation mask從IMAGE_SIZE復原到原始image的大小. 並保留這個原始image大小的segmentation mask跟原始image以供下一個colab cell使用. 請多提供一個figure包含這些獨立區域的overlap用不同的透明顏色(綠, 藍, 紅, 橙色...). 

這個新增加的colab code跟上一個colab code一樣figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分使用now = datetime.datetime.now(ZoneInfo("America/New_York")). 然後再把這個subfolder名字後面加上"_movement2_dinov3_result". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文. 並注意不要犯這個錯誤: module 'cv2' has no attribute 'COLOR_BGR_RGB'
```
-> # ----- 8c. (version 7) use self-supervised learning model (DINOv3) -----

11012312結果:
用單純threshold=0也無法準確切割邊界

```
以下是用DINOv3然後用pca的principal component 3 value map(在這裡將principal component 3 value map獨立儲存成同size的image) 然後自動決定clustering數(最大cluster number=3)希望cluster之間的principal component 3 value差距要遠大於cluster內部的principal component 3 value. 而且每個cluster pixel number不得小於全部image pixel number的10%. 接著檢查這些區域如果pixel number小於全部image pixel number的10%則刪除, 

每個cluster內部會有一個或多個不連接的pixel獨立連續區域, 如果其中有一個獨立連續區域和四個邊緣(上下左右)有pixel接觸. 如果有接觸的邊界是相鄰的, 譬如(上邊界跟左邊界or上邊界跟右邊界), 則整個cluster刪除. 如果一個cluster包含所有的獨立連續區域接觸的邊界是不相鄰的或只有一個邊界或0個邊界接觸, 則cluster加入segmentation mask. 接著將這個把這個segmentation mask從IMAGE_SIZE復原到原始image的大小. 並保留這個原始image大小的segmentation mask跟原始image以供下一個colab cell使用. 請多提供一個figure包含這些獨立區域的overlap用不同的透明顏色(綠, 藍, 紅, 橙色...). 

這個新增加的colab code跟上一個colab code一樣figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分使用now = datetime.datetime.now(ZoneInfo("America/New_York")). 然後再把這個subfolder名字後面加上"_movement2_dinov3_result". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文. 並注意不要犯這個錯誤: module 'cv2' has no attribute 'COLOR_BGR_RGB'
```
-> # ----- 8c. (version 8) use self-supervised learning model (DINOv3) -----

11021724結果:
對大部分image是ok的, 有一張缺一個text, 有一張都沒有mask, 不過其他還算ok. 暫定用version8
然後cluster number=7


```
以下的colab code是用DINOv3然後用pca的principal component 3 value map(在這裡將principal component 3 value map獨立儲存成同size的image) 然後自動決定clustering數並以此一系列計算輸出原image尺寸的segmentation mask跟原尺寸image(請檢查輸出的segmentation mask跟image是原image尺寸). 接下來請提供接續下去的下一個colab code, 是要在原image size下得到更精細及平滑化的segmentation mask. 這segmentation mask是frontground是有寬度的英文字或數字. 英文或數字的輪廓有一層厚度是contrast明顯比較低的區域.

首先在原始圖大小的segmentation mask image先平滑化segmentation mask的邊緣, 再用pixel=40的dilation morphology filter處理, 然後在這個新的segmentation mask區域以每個pixel的contrast為基準(frontground的邊緣是contrast較低, frontground的外面區域contrast較高)先建立contrast map, 並用大小為10的gaussian filter平滑化contrast map, 再從segmentation mask外圍向內部包圍得到新的frontground的精細segmentation mask. 最後再針對segmentation mask的輪廓做平滑化為最終的segmentation mask. 

這個新增加的colab code跟上一個colab code一樣figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分使用now = datetime.datetime.now(ZoneInfo("America/New_York")). 然後再把這個subfolder名字後面加上"_movement2_dinov3_ske_result". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文. 並注意不要犯這個錯誤: module 'cv2' has no attribute 'COLOR_BGR_RGB'
```
-> # ----- 8d. (version 1) Smooth  DINOv3 segmentation mask and follow analysis-----


11021747結果:
這個Text的contrast邊緣上方有破洞, 所以似乎由外往內侵蝕到內部, 而且在text外部有很多小顆粒.


```
這個colab code是在原image size下得到更精細及平滑化的segmentation mask. 這segmentation mask是frontground是有寬度的英文字或數字. 英文或數字的輪廓有一層厚度是contrast明顯比較低的區域. 首先在原始圖大小的segmentation mask image先平滑化segmentation mask的邊緣, 再用pixel=40的dilation morphology filter處理, 然後在這個新的segmentation mask區域以每個pixel的saturation為基準(frontground的邊緣是saturation較低, frontground的外面區域saturation較高)先建立saturation map, 並用大小為10的gaussian filter平滑化saturation map, 再從segmentation mask外圍向內部包圍得到新的frontground的精細segmentation mask, 並用open filter處理去除外面的小顆粒. 最後再針對segmentation mask的輪廓做平滑化為最終的segmentation mask. 

這個新增加的colab code跟上一個colab code一樣figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分使用now = datetime.datetime.now(ZoneInfo("America/New_York")). 然後再把這個subfolder名字後面加上"_movement2_dinov3_ske_result". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文. 並注意不要犯這個錯誤: module 'cv2' has no attribute 'COLOR_BGR_RGB'
```
-> # ----- 8d. (version 2) Smooth  DINOv3 segmentation mask and follow analysis-----

11021823結果:
Text的輪廓segmentation mask有抓到, 但內部有洞, 外部也有獨立小顆粒.


```
這個colab code是在原image size下得到更精細及平滑化的segmentation mask. 這segmentation mask是frontground是有寬度的英文字或數字. 英文或數字的輪廓有一層厚度是contrast明顯比較低的區域. 首先在原始圖大小的segmentation mask image先平滑化segmentation mask的邊緣, 再用pixel=40的dilation morphology filter處理, 然後在這個新的segmentation mask區域以每個pixel的saturation為基準(frontground的邊緣是saturation較低, frontground的外面區域saturation較高)先建立saturation map, 並用大小為15的gaussian filter平滑化saturation map, 這時low-saturation mask在英文或數字的輪廓之外background有很多獨立的小顆粒, 試著用morphology filtering處理消除這些小顆粒. 再從segmentation mask外圍向內部包圍得到新的frontground的精細segmentation mask, 並用morphology filtering先填補輪廓區往內凹的隙縫或小洞, 再處理去除外面的小顆粒. 最後再針對segmentation mask的輪廓做平滑化為最終的segmentation mask. 這就是新的----- 8d. (version 3) Smooth  DINOv3 segmentation mask and follow analysis-----

這個新增加的colab code跟上一個colab code一樣figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分使用now = datetime.datetime.now(ZoneInfo("America/New_York")). 然後再把這個subfolder名字後面加上"_movement2_dinov3_ske_result". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文. 並注意不要犯這個錯誤: module 'cv2' has no attribute 'COLOR_BGR_RGB'
```
-> # ----- 8d. (version 3) Smooth  DINOv3 segmentation mask and follow analysis-----

11021844結果(8d. (version 3)): 11021858_movement2_dinov3_ske_result

11021940結果(8d. (version 2)): 11021939_movement2_dinov3_ske_result
	- SATURATION_GAUSSIAN_RADIUS = 20 default
	- FINAL_SMOOTHING_KERNEL_SIZE = 20 default

11021944結果(8d. (version 2)): 11021939_movement2_dinov3_ske_result
	- SATURATION_GAUSSIAN_RADIUS = 15
	- FINAL_SMOOTHING_KERNEL_SIZE = 15
	- 
11021957結果(8d. (version 2)): 11021957_movement2_dinov3_ske_result
	- DILATION_KERNEL_SIZE = 100
	- SATURATION_GAUSSIAN_RADIUS = 15
	- FINAL_SMOOTHING_KERNEL_SIZE = 15

11022008結果(8d. (version 3)): 11022008_movement2_dinov3_ske_result
	- DILATION_KERNEL_SIZE = 100
	- SATURATION_GAUSSIAN_RADIUS = 30


```
這個colab code是在原image size下得到更精細及平滑化的segmentation mask. 這segmentation mask是frontground是有寬度的英文字或數字. 英文或數字的輪廓有一層厚度是contrast明顯比較低的區域. 首先在原始圖大小的segmentation mask image先平滑化segmentation mask的邊緣, 再用pixel=40的dilation morphology filter處理, 然後在這個新的segmentation mask區域(coarse mask)以每個pixel的saturation為基準(frontground的邊緣是saturation較低, frontground的外面區域saturation較高)先建立saturation map, 並用大小為15的gaussian filter平滑化saturation map再進行反轉使低saturation 區（字體邊界）成為高值區, 用基於 gradient magnitude 的形態學內縮 (morphological inward shrinking)用 coarse mask 當起始，做逐步形態學侵蝕直到字體邊界. 並用morphology filtering填補輪廓區往內凹的隙縫或小洞, 處理去除外面的小顆粒. 最後再針對segmentation mask的輪廓做平滑化為最終的segmentation mask. 這就是新的----- 8d. (version 4) Smooth  DINOv3 segmentation mask and follow analysis-----

這個新增加的colab code跟上一個colab code一樣figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分使用now = datetime.datetime.now(ZoneInfo("America/New_York")). 然後再把這個subfolder名字後面加上"_movement2_dinov3_ske_result". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文. 並注意不要犯這個錯誤: module 'cv2' has no attribute 'COLOR_BGR_RGB'
```
-> # ----- 8d. (version 4) Smooth  DINOv3 segmentation mask and follow analysis-----

11022150結果(8d. (version 4)): 11022156_movement2_dinov3_ske_result

11022204結果(8d. (version 4)): 11022204_movement2_dinov3_ske_result
	- SATURATION_GAUSSIAN_RADIUS = 5

11022229結果(8d. (version 4)): 11022229_movement2_dinov3_ske_result
	- SATURATION_GAUSSIAN_RADIUS = 5
	- SURE_FG_ERODE_KSIZE = 20

11022232結果(8d. (version 4)): 11022231_movement2_dinov3_ske_result
	- SATURATION_GAUSSIAN_RADIUS = 5
	- SURE_FG_ERODE_KSIZE = 50


```
我想把上面# ----- 8d.(version 17) Refine Mask with Saturation and Watershed -----的code做一些改變. 關於Watershed Markers的Sure Foreground (標記 2), 我選擇用計算這個粗糙 Mask的的中心點設為Sure Foreground (標記 2). 然後最後的輪廓還是不夠平滑有馬賽克狀應該如何解決. 提供新的code

那如果watershed我們有確定的背景以及Saturation計算出的低飽和度山脊, 但沒有seed已可以用來找字體的輪廓嗎? 那用level set / geodesic active contour 方法可以嗎
```
-> # ----- 8d. (version 5) Smooth  DINOv3 segmentation mask and follow analysis-----

11030002結果(8d. (version 5)): 11030002_movement2_dinov3_ske_result


```
先建立saturation map, 並用大小為15的gaussian filter平滑化saturation map用otsu計算出segmentation mask再用morphology filtering先填補輪廓區往內凹的隙縫或小洞, 再處理去除外面的小顆粒得到一個mask. 接著我們開始行分水嶺算法. 我們把剛得到的mask將其向內侵蝕 7 像素當成watershed的Sure Foreground (標記 2), 之後再對最原始的粗糙 Mask往內侵蝕7 pixel成確定的Sure Background (標記 1), 然後我們把saturation map反轉地圖 (Invert Map)使得低飽和度 (字體邊緣) 變成了高數值 (255) 的「山脊」再執行執行分水嶺 (Run Watershed). 最後我們只提取所有被演算法歸類為「前景」(label 2) 的像素. 並在最後針對最後segmentation mask的邊緣平滑化解決馬賽克. 

這個新增加的colab code跟上一個colab code一樣figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分使用now = datetime.datetime.now(ZoneInfo("America/New_York")). 然後再把這個subfolder名字後面加上"_movement2_dinov3_ske_result". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文. 並注意不要犯這個錯誤: module 'cv2' has no attribute 'COLOR_BGR_RGB'

```
-> # ----- 8d. (version 6) Smooth  DINOv3 segmentation mask and follow analysis-----

11030049結果(8d. (version 6)): 11030049_movement2_dinov3_ske_result


```
以下是colab code可以得到一個最終的segmentation mask. 接下來我進行分水嶺算法. 把剛剛得到的最終的segmentation mask將其向內侵蝕 7 像素當成watershed的Sure Foreground (標記 2), 之後再對原始的粗糙 Mask經過Dilation得到的dilated_mask_roi往內侵蝕7 pixel成確定的Sure Background (標記 1), 建立saturation map, 並用大小為15的gaussian filter平滑化saturation map再進行反轉使低saturation 區（字體邊界）成為高值區, 用基於 gradient magnitude 的形態學內縮 (morphological inward shrinking)用 coarse mask 當起始，做逐步形態學侵蝕直到字體邊界. 並用morphology filtering填補輪廓區往內凹的隙縫或小洞, 處理去除外面的小顆粒. 最後再針對segmentation mask的輪廓做平滑化為最終的segmentation mask. 

這個新增加的colab code跟上一個colab code一樣figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分使用now = datetime.datetime.now(ZoneInfo("America/New_York")). 然後再把這個subfolder名字後面加上"_movement2_dinov3_ske_result". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文. 並注意不要犯這個錯誤: module 'cv2' has no attribute 'COLOR_BGR_RGB'
```
-> # ----- 8d. (version 7) Smooth  DINOv3 segmentation mask and follow analysis-----

11030127結果(8d. (version 6)): 11030127_movement2_dinov3_ske_result


看起來目前使用11021835_movement2_dinov3_ske_result  (應該是5d. (version 3))
![[Pasted image 20251103075610.png]]



```
這個Colab code (5d.(version 3)) 會輸出一個final_refined_mask. 請新創一個新的colab cell繼續final_refined_mask下面的分析請提供code: 對最後得到的final_refined_mask得到輪廓線, 並用直線和曲線去fit這條輪廓線(某些線段是直線, 有些線段是曲線有圓心跟固定radius), 記錄下每個直線段的頭尾座標以及距離, 直線部分跟明顯轉折點以及轉折的角度都計算出來. 曲線的頭尾座標, 圓心座標跟radius以及線段長. 接著計算mask的skeleton以及這個skeleton line到數字mask boundary的距離(thickness), 可以用在skeleton每隔一段距離的座標點並記下thickness. 另外檢查這個mask的skeleton line的1/3*thickness區域在image上有明顯的黑色區域, 則這個mask類型是"type 1", 如果沒有則是"type 2". 

然後這個final mask是由幾個獨立mask(連續有接連的pixel區域)組成. 這是水平方向的幾個英文字數字, 或者是字體轉90度垂直方向的幾個英文字數字. 我們可以判斷這些字體是水平方向字體或垂直方向字體可以把這些獨立mask各計算中心座標, 然後看他們如果中心座標主要x方向間距大於 y方向間距那代表就是水平. x方向間距小於 y方向間距那代表就是垂直. 當判斷是水平方向字體, 則final mask的每個獨立mask(連續有接連的pixel區域)用x,y的min, max得到box的四個端點座標以及個別的height跟width, 以及用OCR去辨識是甚麼英文字或數字, 對每個box用原來的box以及轉180度再去用OCR去辨識並採最高confidence去決定text並輸出. 當判斷是垂直方向字體時, 先把每個box先轉90度, 之後的流程就跟水平方向字體流程一樣.

這個新增加的colab code跟上一個colab code一樣figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分使用now = datetime.datetime.now(ZoneInfo("America/New_York")). 然後再把這個subfolder名字後面加上"_movement2_dinov3_text_result(v1)". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文. 並注意不要犯這個錯誤: module 'cv2' has no attribute 'COLOR_BGR_RGB'
```
-> # ----- 5e. (version 1) DINOv3 fine segmentation Text Mask Analysis -----
