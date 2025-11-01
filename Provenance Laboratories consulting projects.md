
|                                           |     |
| ----------------------------------------- | --- |
| [[#### Project 4 - Dialtext]]             |     |
| [[#### Project 5 - Lume]]                 |     |
| [[#### Project 6 - Features]]             |     |
| [[#### Project 7 - Movement dimensions1]] |     |
| [[#### Project 8 - Movement dimensions2]] |     |




#### Project 4 - Dialtext

1031
create OCR region segmentation masks and analysis




#### Project 5 - Lume


1031
Features: Hue, Sarturation contrast looks good, Texture not working
FFT not working
[Action] should try Histogram backprojection using Hue or Sarturation or contrast
[New code result] 




#### Project 6 - Features

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
--> # ----- 10.(version 2) - Image automatic mask generation using SAM  -----

11010925結果:
結果發現在分割跟cluster segment背景區域時會找到某些小區域, 所以新增限制條件background一定要大於30% image pixel numbers. 另外結果也發現center_mask有包含background區域, 也重寫code.

```
以下是colab code用SAM生成一張圖片的automatic mask generation. 並找尋background_mask跟center_mask. 請新增輸出SAM的original image with SAM masks overlaid並把code裡面會影響segmentation及結果的parameter集中放在import下方方便設定. 請新增條件background必須是pixel number至少占原image面積30%以上, 如果不達到就以順位下一位為background(譬如以TextTure(Std Dev)平均最大的cluster為background如果pixel只佔原image的10%, 則以TextTure(Std Dev)平均第二大的cluster為background). 另外在從frontground尋找center_mask的步驟, 發現有幾張image的center_mask結果竟包含原來被辨認為background的區域? 請檢查code為何會如此並修正.

這個新增加的colab code跟上一個colab code一樣figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分, 然後再把這個subfolder名字後面加上"_dinov3(v2)". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. code儘量function化, code中的comments也都用英文

```
--> # ----- 10.(version 3) - Image automatic mask generation using SAM  -----

11011024結果:
結果發現:
有時候分成2 cluster時, 會把邊緣(很明顯的邊緣)分成一個cluster但pixel可能只占5%以下. 所以可能要限制初期cluster 有pixel number限制. 另外發現有些frontground一開始SAM就沒有分割到, 應該修改SAM parameter讓分割更細.

```
以下是colab code用SAM生成一張圖片的automatic mask generation. 並找尋background_mask跟center_mask. 請在將SAM 一開始cluster階段並自動決定cluster number階段, 如果有某個cluster有小於10% (新增加這個parameter) pixel面積, 則先排除這部分視為例外, 等到之後計算出background之後再把例外區域加入frontground.另外SAM在初期segment時有些物體沒有被分割出來, 建議調整參數

這個新增加的colab code跟上一個colab code一樣figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分, 然後再把這個subfolder名字後面加上"_dinov3(v2)". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文
```
--> # ----- 10.(version 4) - Image automatic mask generation using SAM  -----

11011024結果:
結果如果刪除例外區域應該重新再clustering, 所以修正

```
以下是colab code用SAM生成一張圖片的automatic mask generation. 並找尋background_mask跟center_mask. 請在將SAM 一開始cluster階段並自動決定cluster number階段, 如果有某個cluster有小於10% (新增加這個parameter) pixel面積, 則先排除這部分視為例外, 這裡排除這部分之後要重新進行自動決定cluster number跟把剩下的區域clustering直到沒有例外區域. 請重修code. 等到之後計算出background之後再把例外區域加入frontground.另外SAM在初期segment時有些物體沒有被分割出來, 建議調整參數

這個新增加的colab code跟上一個colab code一樣figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分使用now = datetime.datetime.now(ZoneInfo("America/New_York")). 然後再把這個subfolder名字後面加上"_dinov3(v2)". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文
```
--> # ----- 10.(version 5) - Image automatic mask generation using SAM  -----

11011024結果:



#### Project 7 - Movement dimensions1

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




#### Project 8 - Movement dimensions2

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
