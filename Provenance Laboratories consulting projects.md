
|                                           |     |
| ----------------------------------------- | --- |
| [[#### Project 4 - Dialtext]]             |     |
| [[#### Project 5 - Lume]]                 |     |
| [[#### Project 6 - Features]]             |     |
| [[#### Project 7 - Movement dimensions1]] |     |
| [[#### Project 8 - Movement dimensions2]] |     |
|                                           |     |



#### Project 4 - Dialtext


|                                                                                    | All | Best | Note |
| ---------------------------------------------------------------------------------- | --- | ---- | ---- |
| # ----- 1. Import dependent -----                                                  | 1   | 1    |      |
| # ----- 2. Project folder and device/device setup -----                            | 1   | 1    |      |
| # ----- 3. images loading: dial text -----                                         | 1   | 1    |      |
| ----- 4. (version 6) create OCR region segmentation masks and analysis ----- ----- | 6   | 6    |      |




create OCR region segmentation masks and analysis
```
這個Colab code 用easyOCR跟pytesseract兩階段偵測text. 但在pytesseract有時候在較緊密的字間隔時可能會漏掉字, 以及明明easyOCR已經偵測出一串字譬如"GMT"的box, 但pytesseract都辨識不出來. 或者有時候easyOCR偵測的"PERPETUAL"但pytesseract把"UA"框成一個. 請想辦法解決上述issue並提供code. 

這個colab codefigure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分使用now = datetime.datetime.now(ZoneInfo("America/New_York")). 然後再把這個subfolder名字後面加上"_dialtext_result(v2)". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文. 並注意不要犯這個錯誤: module 'cv2' has no attribute 'COLOR_BGR_RGB'
```

```
以下是Colab code可以從錶面用2 step OCR去先用easyOCR辨識單詞, 並在這個單詞的box內利用pytesseract辨識character level 英文字或數字.

但我使用時step OCR去先用easyOCR辨識單詞結果不錯, 但接下來character level似乎出一些issue. 譬如easyOCR辨識的"OYSTER", pytesseract無法辨識其中的"O","Y","S","T","E","R"的所有character. easyOCR辨識的"GMT", pytesseract無法辨識其中的"G","M","T"所有character. 譬如easyOCR辨識的"300m", pytesseract無法辨識其中的"3","0","0","m"的所有character.或者pytesseract辨識"PERPETUAL"他會把"UA"放在同個字, 或者pytesseract會把"ER", "AT"放在同個字,或者有些稍微傾斜類似稍微草寫的字也無法辨識. 

這個colab codefigure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分使用now = datetime.datetime.now(ZoneInfo("America/New_York")). 然後再把這個subfolder名字後面加上"_dialtext_result(v3)". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文. 並注意不要犯這個錯誤: module 'cv2' has no attribute 'COLOR_BGR_RGB'
```



參考# ----- 4.(version 1) create OCR region segmentation masks and analysis  -----
```
以下是Colab code可以從錶面用2 step OCR去先用easyOCR辨識單詞, 並在這個單詞的box內利用pytesseract辨識character level 英文字或數字. 但我使用時step OCR去先用easyOCR辨識單詞結果不錯, 但接下來character level似乎出一些issue. 譬如easyOCR辨識的"OYSTER", pytesseract無法辨識其中的"O","Y","S","T","E","R"的所有character. easyOCR辨識的"GMT", pytesseract無法辨識其中的"G","M","T"所有character. 譬如easyOCR辨識的"300m", pytesseract無法辨識其中的"3","0","0","m"的所有character.或者pytesseract辨識"PERPETUAL"他會把"UA"放在同個字, 或者pytesseract會把"ER", "AT"放在同個字,或者有些稍微傾斜類似稍微草寫的字也無法辨識. 請增加一些設定可以讓OCR可以解決這些問題(譬如ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_C等等等), 其他不改變. 
```
-> # ----- 4.(version 2) create OCR region segmentation masks and analysis  -----






#### Project 5 - Lume

|                                                                                    | All | Best | Note |
| ---------------------------------------------------------------------------------- | --- | ---- | ---- |
| # ----- 1. Import dependent -----                                                  | 1   | 1    |      |
| # ----- 2. Project folder and device/device setup -----                            | 1   | 1    |      |
| # ----- 3. images loading: DATA_LUME_PATH -----                                    | 1   | 1    |      |
| # ----- 4. calculate image features summary                                        | 1   | 1    |      |
| # ----- 5. (version 4) Example-Based Segmentation (Histogram Backprojection) ----- | 1-8 | 8    |      |
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
直方圖反向投影 (Histogram Backprojection) 是一種影像分割技術，其核心思想是：使用一個「範例」（Model）來尋找一張「目標影像」（Target Image）中所有與該範例相似的區域。它是一種"有監督" (supervised) 的方法，因為您需要先提供一個您感興趣的物體（即前景，Foreground）的範例。想像一下，您有一個物體（例如一塊特定的藍色布料）的「特徵指紋」。這個「指紋」就是直方圖(Histogram)。然後，您拿著這個指紋去掃描整張目標影像，為每個像素打分，分數高低代表該像素與「指紋」的匹配程度。這張「分數地圖」就是機率圖 (Probability Map)。最後，您設定一個分數門檻（Threshold），高於這個門檻的像素就被認為是您要找的物體

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

11031521結果:



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



```
以下是colab code可以用Example-Based Segmentation / Histogram Backprojection在image上偵測相似區域並建立segmentation mask. 請幫我新增功能可以每個POSITIVE_REFERENCE or NEGATIVE_REFERENCE的image都輸出figure並標示ROI在上面. 另外我想要把從POSITIVE_REFERENCE or NEGATIVE_REFERENCE最後計算的2D histogram以及所有需要用來在target計算segmentation masks的所有data跟參數存在txt file, 目的是往後用在其他target images不需要載入POSITIVE_REFERENCE or NEGATIVE_REFERENCE的image, 只需要txt上面的參數就好.

這個新增加的colab code跟上一個colab code一樣figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分使用now = datetime.datetime.now(ZoneInfo("America/New_York")). 然後再把這個subfolder名字後面加上"_lume_analysis_result(v3)". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文. 並注意不要犯這個錯誤: module 'cv2' has no attribute 'COLOR_BGR_RGB'
```
-> # ----- 5. (version 3) Example-Based Segmentation (Histogram Backprojection) -----


結果:11041459_histBack_model(v3) 
BINS_CH1 = 15, BINS_CH2 = 15

結果:11041459_histBack_model(v3) 
BINS_CH1 = 10, BINS_CH2 = 10

結果:11041524_histBack_model(v3) 
BINS_CH1 = 10, BINS_CH2 = 10
POSITIVE_REFERENCE_PATHS = [hour_image_paths[0], hour_image_paths[3]]
POSITIVE_ROI_FACTORS_LIST = [[0.50, 0.70, 0.4, 0.6],[0.16, 0.79, 0.10, 0.86]]
NEGATIVE_REFERENCE_PATHS = [hour_image_paths[6]]
NEGATIVE_ROI_FACTORS_LIST = [[0.18, 0.30, 0.26, 0.38]]
好像加入negative samples幾乎所有segmentation masks都消失


結果:11041529_histBack_model(v3) 
BINS_CH1 = 15, BINS_CH2 = 15
NEGATIVE_REFERENCE_PATHS = [hour_image_paths[6]]
NEGATIVE_ROI_FACTORS_LIST = [[0.18, 0.30, 0.26, 0.38]]

結果:11041538_histBack_model(v3) 
BINS_CH1 = 15, BINS_CH2 = 15
POSITIVE_REFERENCE_PATHS = [hour_image_paths[0]]
POSITIVE_ROI_FACTORS_LIST = [[0.50, 0.70, 0.4, 0.6]]
NEGATIVE_REFERENCE_PATHS = [hour_image_paths[6]]
NEGATIVE_ROI_FACTORS_LIST = [[0.18, 0.30, 0.26, 0.38]]


```
以下是colab code可以用Example-Based Segmentation / Histogram Backprojection在image上偵測相似區域並建立segmentation mask. 想改變機制 POSITIVE_ROI_FACTORS_LIST跟NEGATIVE_ROI_FACTORS_LIST內部代表每個ROI的list多加一個參數weight(0~1)所以現在每個list有五個數字. 這個weight代表權重, 權重越大越dominate直方圖. 另外positive_ROI跟negative_ROI都作用在同個直方圖上, positive_ROI會增加對應bin的數值(也受權重影響), neative_ROI會減少對應bin的數值(也受權重影響). 請提供新的code.

這個新增加的colab code跟上一個colab code一樣figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分使用now = datetime.datetime.now(ZoneInfo("America/New_York")). 然後再把這個subfolder名字後面加上"_lume_histBack_result(v4)". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文. 並注意不要犯這個錯誤: module 'cv2' has no attribute 'COLOR_BGR_RGB'
```
-> # ----- 5. (version 4) Example-Based Segmentation (Histogram Backprojection) -----

結果:11041613_lume_histBack_result(v4)
	POSITIVE_REFERENCE_PATHS = [hour_image_paths[0], hour_image_paths[1]]
	POSITIVE_ROI_FACTORS_LIST = [[0.50, 0.70, 0.4, 0.6, 1.0],[0.16, 0.79, 0.10, 0.86, 1.0]]
	NEGATIVE_REFERENCE_PATHS = [hour_image_paths[6]]
	NEGATIVE_ROI_FACTORS_LIST = [[0.18, 0.3, 0.26, 0.38, 0.2]]
	
結果:11041622_lume_histBack_result(v4)
	POSITIVE_REFERENCE_PATHS = [hour_image_paths[0], hour_image_paths[1]]
	POSITIVE_ROI_FACTORS_LIST = [[0.50, 0.70, 0.4, 0.6, 1.0],[0.16, 0.79, 0.10, 0.86, 1.0]]
	NEGATIVE_REFERENCE_PATHS = [hour_image_paths[6]]
	NEGATIVE_ROI_FACTORS_LIST = [[0.18, 0.3, 0.26, 0.38, 1.0]]
--> 全部都是segmentation masks?

結果:11041622_lume_histBack_result(v4)
	POSITIVE_REFERENCE_PATHS = [hour_image_paths[0], hour_image_paths[1]]
	POSITIVE_ROI_FACTORS_LIST = [[0.50, 0.70, 0.4, 0.6, 1.0],[0.16, 0.79, 0.10, 0.86, 1.0]]
	NEGATIVE_REFERENCE_PATHS = [hour_image_paths[6]]
	NEGATIVE_ROI_FACTORS_LIST = [[0.18, 0.3, 0.26, 0.38, 1.0]]

結果:11041636_lume_histBack_result(v4)
	POSITIVE_REFERENCE_PATHS = [hour_image_paths[6]]
	POSITIVE_ROI_FACTORS_LIST = [[0.40, 0.56, 0.21, 0.39, 1.0]]
	NEGATIVE_REFERENCE_PATHS = [hour_image_paths[6]]
	NEGATIVE_ROI_FACTORS_LIST = [[0.18, 0.3, 0.26, 0.38, 1.0]]
	
結果:11041643_lume_histBack_result(v4)
		Select: [ 2, 3 ]
	POSITIVE_REFERENCE_PATHS = [hour_image_paths[6]]
	POSITIVE_ROI_FACTORS_LIST = [[0.40, 0.56, 0.21, 0.39, 1.0]]
	NEGATIVE_REFERENCE_PATHS = [hour_image_paths[6]]
	NEGATIVE_ROI_FACTORS_LIST = [[0.18, 0.3, 0.26, 0.38, 1.0]]

結果:11041959_lume_histBack_result(v4)
		Select: [ 2, 2 ]
	POSITIVE_REFERENCE_PATHS = [hour_image_paths[6]]
	POSITIVE_ROI_FACTORS_LIST = [[0.40, 0.56, 0.21, 0.39, 1.0]]
	NEGATIVE_REFERENCE_PATHS = [hour_image_paths[6]]
	NEGATIVE_ROI_FACTORS_LIST = [[0.18, 0.3, 0.26, 0.38, 1.0]]


在這裡還是以5. (version 3) Example-Based Segmentation (Histogram Backprojection)做修正
```
以下是colab code可以用Example-Based Segmentation / Histogram Backprojection在image上偵測相似區域並建立segmentation mask. 基於以下的code做修正, POSITIVE_ROI_FACTORS_LIST跟NEGATIVE_ROI_FACTORS_LIST內部代表每個ROI的list多加一個參數weight(0~1)所以現在每個list有五個數字. 這個weight代表權重, 權重越大越dominate直方圖. BACKPROJ_THRESHOLD_FG跟BACKPROJ_THRESHOLD_BG
有沒有方法可以自動選擇? post processing部分在用了opening跟closing之後用三種不同method進行(有parameter選擇哪個method) method 1. 提取邊界cv2.findContours, 接著用Gaussian filter 平滑邊界座標, 接著用重建 mask. method 2. 提取邊界cv2.findContours, 接著用B-spline 平滑邊界座標, 接著用重建 mask. method 3. 用from scipy.ndimage import distance_transform_edt, gaussian_filter使用Distance transform + Gaussian filter + threshold

這個新增加的colab code跟上一個colab code一樣figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分使用now = datetime.datetime.now(ZoneInfo("America/New_York")). 然後再把這個subfolder名字後面加上"_lume_histBack_result(v5)". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文. 並注意不要犯這個錯誤: module 'cv2' has no attribute 'COLOR_BGR_RGB'

```
-> # ----- 5. (version 5) Example-Based Segmentation (Histogram Backprojection) -----

結果:11042147_lume_histBack_result(v5)
	POSITIVE_REFERENCE_PATHS = [hour_image_paths[0], hour_image_paths[10]]
	POSITIVE_ROI_FACTORS_LIST = [ [0.50, 0.70, 0.4, 0.6, 1.0],[0.45, 0.65, 0.22, 0.42, 0.8]]
	AUTO_THRESHOLD = False
	SMOOTHING_METHOD = "gaussian_contour"

結果:11042154_lume_histBack_result(v5)
	POSITIVE_REFERENCE_PATHS = [hour_image_paths[0], hour_image_paths[10]]
	POSITIVE_ROI_FACTORS_LIST = [ [0.50, 0.70, 0.4, 0.6, 1.0],[0.45, 0.65, 0.22, 0.42, 0.8]]
	AUTO_THRESHOLD = True
	SMOOTHING_METHOD = "gaussian_contour"

結果:11042159_lume_histBack_result(v5)
	POSITIVE_REFERENCE_PATHS = [hour_image_paths[0], hour_image_paths[10]]
	POSITIVE_ROI_FACTORS_LIST = [ [0.50, 0.70, 0.4, 0.6, 1.0],[0.45, 0.65, 0.22, 0.42, 0.8]]
	AUTO_THRESHOLD = True
	SMOOTHING_METHOD = "bspline_contour"

結果:11042206_lume_histBack_result(v5)
	POSITIVE_REFERENCE_PATHS = [hour_image_paths[0], hour_image_paths[10]]
	POSITIVE_ROI_FACTORS_LIST = [ [0.50, 0.70, 0.4, 0.6, 1.0],[0.45, 0.65, 0.22, 0.42, 0.8]]
	AUTO_THRESHOLD = False
	BACKPROJ_THRESHOLD_FG = 2
	BACKPROJ_THRESHOLD_BG = 2
	SMOOTHING_METHOD = "bspline_contour"
	SMOOTHING_SPLINE_S = 5.0
	
結果:11042224_lume_histBack_result(v5)
	POSITIVE_REFERENCE_PATHS = [hour_image_paths[0], hour_image_paths[10]]
	POSITIVE_ROI_FACTORS_LIST = [ [0.50, 0.70, 0.4, 0.6, 1.0],[0.45, 0.65, 0.22, 0.42, 0.8]]
	AUTO_THRESHOLD = True
	AUTO_THRESHOLD_RATIO = 0.6
	SMOOTHING_METHOD = "bspline_contour"
	SMOOTHING_SPLINE_S = 50.0

結果:11042235_lume_histBack_result(v5)
	POSITIVE_REFERENCE_PATHS = [hour_image_paths[0]]
	POSITIVE_ROI_FACTORS_LIST = [ [0.50, 0.70, 0.4, 0.6, 1.0]]
	AUTO_THRESHOLD = True
	AUTO_THRESHOLD_RATIO = 0.6
	SMOOTHING_METHOD = "bspline_contour"
	SMOOTHING_SPLINE_S = 100.0


在這裡還是以5. (version 5) Example-Based Segmentation (Histogram Backprojection)做修正
```
以下是colab code可以用Example-Based Segmentation / Histogram Backprojection在image上偵測相似區域並建立segmentation mask. 這個code也會將code2D histogram以及所有需要用來在target計算segmentation masks的所有data跟參數存在segmentation_model.json, 目的是往後用在其他target images不需要載入POSITIVE_REFERENCE or NEGATIVE_REFERENCE的image, 只需要segmentation_model.json上面的參數就好.基於以下的code加入這個功能其他不改變. 可以用JSON_PATH這個參數載入segmentation_model.json的path, 如果JSON_PATH不是empty, 則載入JSON上面的parameters而在target iamges建立segmentation masks, 也代表本來colab參數就不會影響譬如POSITIVE_REFERENCE_PATHS,  POSITIVE_ROI_FACTORS_LIST等

這個新增加的colab code跟上一個colab code一樣figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分使用now = datetime.datetime.now(ZoneInfo("America/New_York")). 然後再把這個subfolder名字後面加上"_lume_histBack_result(v6)". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文. 並注意不要犯這個錯誤: module 'cv2' has no attribute 'COLOR_BGR_RGB'
```
-> # ----- 5. (version 6) Example-Based Segmentation (Histogram Backprojection) -----

結果:11042325_lume_histBack_result(v6)
	POSITIVE_REFERENCE_PATHS = [hour_image_paths[2], hour_image_paths[5]]
	POSITIVE_ROI_FACTORS_LIST = [[0.4, 0.70, 0.10, 0.40, 1.0] , [0.30, 0.60, 0.25, 0.55, 0.8]]
	AUTO_THRESHOLD = True
	AUTO_THRESHOLD_RATIO = 0.6
	SMOOTHING_METHOD = "bspline_contour"
	SMOOTHING_SPLINE_S = 500.0

結果:11042336_lume_histBack_result(v6)
	POSITIVE_REFERENCE_PATHS = [hour_image_paths[2], hour_image_paths[5]]
	POSITIVE_ROI_FACTORS_LIST = [[0.4, 0.70, 0.10, 0.40, 1.0] , [0.30, 0.60, 0.25, 0.55, 0.4]]
	AUTO_THRESHOLD = True
	AUTO_THRESHOLD_RATIO = 0.8
	SMOOTHING_METHOD = "bspline_contour"
	SMOOTHING_SPLINE_S = 500.0
	
結果:11042343_lume_histBack_result(v6)
	POSITIVE_REFERENCE_PATHS = [hour_image_paths[2], hour_image_paths[5]]
	POSITIVE_ROI_FACTORS_LIST = [[0.4, 0.70, 0.10, 0.40, 1.0] , [0.30, 0.60, 0.25, 0.55, 0.4]]
	AUTO_THRESHOLD = True
	AUTO_THRESHOLD_RATIO = 0.6
	SMOOTHING_METHOD = "bspline_contour"
	SMOOTHING_SPLINE_S = 2000.0
	
結果:11042350_lume_histBack_result(v6)
	POSITIVE_REFERENCE_PATHS = [hour_image_paths[2], hour_image_paths[5]]
	POSITIVE_ROI_FACTORS_LIST = [[0.4, 0.70, 0.10, 0.40, 1.0] , [0.30, 0.60, 0.25, 0.55, 0.4]]
	AUTO_THRESHOLD = True
	AUTO_THRESHOLD_RATIO = 0.3
	SMOOTHING_METHOD = "bspline_contour"
	SMOOTHING_SPLINE_S = 10000.0

結果:11042357_lume_histBack_result(v6)
	POSITIVE_REFERENCE_PATHS = [hour_image_paths[2], hour_image_paths[5]]
	POSITIVE_ROI_FACTORS_LIST = [[0.4, 0.70, 0.10, 0.40, 1.0] , [0.30, 0.60, 0.25, 0.55, 0.4]]
	AUTO_THRESHOLD = True
	AUTO_THRESHOLD_RATIO = 0.1
	SMOOTHING_METHOD = "bspline_contour"
	SMOOTHING_SPLINE_S = 50000.0

結果:11050010_lume_histBack_result(v6)
	POSITIVE_REFERENCE_PATHS = [hour_image_paths[2], hour_image_paths[5]]
	POSITIVE_ROI_FACTORS_LIST = [[0.4, 0.70, 0.10, 0.40, 1.0] , [0.30, 0.60, 0.25, 0.55, 0.4]]
	AUTO_THRESHOLD = True
	AUTO_THRESHOLD_RATIO = 0.2
	SMOOTHING_METHOD = "bspline_contour"
	SMOOTHING_SPLINE_S = 100000.0

結果:11050017_lume_histBack_result(v6)
	POSITIVE_REFERENCE_PATHS = [hour_image_paths[2], hour_image_paths[5]]
	POSITIVE_ROI_FACTORS_LIST = [[0.4, 0.70, 0.10, 0.40, 1.0] , [0.30, 0.60, 0.25, 0.55, 0.4]]
	AUTO_THRESHOLD = True
	AUTO_THRESHOLD_RATIO = 0.3
	SMOOTHING_METHOD = "bspline_contour"
	SMOOTHING_SPLINE_S = 100000.0

結果:11050024_lume_histBack_result(v6)
	POSITIVE_REFERENCE_PATHS = [hour_image_paths[2], hour_image_paths[5]]
	POSITIVE_ROI_FACTORS_LIST = [[0.4, 0.70, 0.10, 0.40, 1.0] , [0.30, 0.60, 0.25, 0.55, 0.4]]
	AUTO_THRESHOLD = True
	AUTO_THRESHOLD_RATIO = 0.3
	OPEN_KERNEL_SIZE = (30, 30)
	SMOOTHING_METHOD = "bspline_contour"
	SMOOTHING_SPLINE_S = 100000.0


```
**在 `(version 6)` 中，我已經修正了這個問題：**

1. **修正 B-spline**：我將 `SMOOTHING_SPLINE_S` (絕對值) 替換為 `SMOOTHING_SPLINE_FACTOR` (比例因子)。`s_value = m * spline_factor`。現在，`SMOOTHING_SPLINE_FACTOR = 1.5` 將在所有情況下都提供一致的平滑效果。
    
2. **新增 `JSON_PATH`**：在「Parameters」區塊頂部新增了 `JSON_PATH = None`。
    
3. **新增 `load_segmentation_model` 函數**：用於讀取 JSON 並**覆蓋** (overwrite) 程式碼中的全域參數。
    
4. **新增「Build vs. Load」邏輯**：在「Block 1」中，程式碼現在會檢查 `JSON_PATH`：
    
    - **If `not JSON_PATH` (Build Mode)**：執行 `v5` 的所有模型建立、儲存 `json` 的邏輯。
        
    - **Else (Inference Mode)**：跳過建立，呼叫 `load_segmentation_model` 載入所有設定和直方圖。
```
-> # ----- 5. (version 7) Example-Based Segmentation (Histogram Backprojection) -----

結果:11050054_lume_histBack_result(v6)
	POSITIVE_REFERENCE_PATHS = [hour_image_paths[2], hour_image_paths[5]]
	POSITIVE_ROI_FACTORS_LIST = [[0.4, 0.70, 0.10, 0.40, 1.0] , [0.30, 0.60, 0.25, 0.55, 0.4]]
	AUTO_THRESHOLD = True
	AUTO_THRESHOLD_RATIO = 0.3
	OPEN_KERNEL_SIZE = (30, 30)
	SMOOTHING_METHOD = "bspline_contour"
	SMOOTHING_SPLINE_S = 100000.0



```
這將是一個 `(version 7)`，它會重寫 `apply_smoothing` 函數，使其能夠：

1. 使用 `cv2.approxPolyDP` 找到輪廓的「斷點」(breakpoints)。 
2. 將_原始_輪廓（有鋸齒的）分割成由這些斷點定義的多個「點集」。 
3. 對每個「點集」**嘗試擬合一條直線**。 
4. **如果** 擬合得足夠好（低於 `PIECEWISE_LINE_FIT_THRESH` 閾值），則用一條**完美的直線**取代該線段。
5. **如果** 擬合得不好（即它是一條曲線），則**退回** (fallback) 到使用 B-spline **僅對該線段**進行平滑。
6. 最後，將所有這些「完美的直線」和「平滑的曲線」無縫地縫合回一個單一的、極度平滑的輪廓。

我還會保留您在 `(version 6)` 中要求的所有功能（`JSON_PATH` 載入、權重、自動閾值），並**修正** `B-spline` 的平滑因子問題。

```
-> # ----- 5. (version 8) Example-Based Segmentation (Histogram Backprojection) -----

結果:11050128_lume_histBack_result(v7)
	PIECEWISE_APPROX_EPSILON = 0.005 # 0.5% of arc length
	PIECEWISE_LINE_FIT_THRESH = 0.5
	PIECEWISE_FALLBACK_SPLINE_FACTOR = 1.0

結果:11050136_lume_histBack_result(v7)
	PIECEWISE_APPROX_EPSILON = 0.005 # 0.5% of arc length
	PIECEWISE_LINE_FIT_THRESH = 1.5
	PIECEWISE_FALLBACK_SPLINE_FACTOR = 2.0

結果:11050149_lume_histBack_result(v7)
	AUTO_THRESHOLD_RATIO = 0.3
	PIECEWISE_APPROX_EPSILON = 0.005 # 0.05% of arc length
	PIECEWISE_LINE_FIT_THRESH = 3
	PIECEWISE_FALLBACK_SPLINE_FACTOR = 3.0

結果:11050200_lume_histBack_result(v7)
	AUTO_THRESHOLD_RATIO = 0.5
	PIECEWISE_APPROX_EPSILON = 0.005 # 0.01% of arc length
	PIECEWISE_LINE_FIT_THRESH = 3
	PIECEWISE_FALLBACK_SPLINE_FACTOR = 3.0
--> **用Version 8 跟這個parameters對不少的images表現很好, 先用這個**



#### Project 6 - Features

|                                                                          | All | Best | Note |
| ------------------------------------------------------------------------ | --- | ---- | ---- |
| # ----- 1. Import dependent -----                                        | 1   | 1    |      |
| # ----- 2. Project folder and device/device setup -----                  | 1   | 1    |      |
| # ----- 3. images loading: Features -----                                | 1   | 1    |      |
| # ----- 4. calculate image features summary                              | 1   | 1    |      |
| # ----- 5.(version 6) - Image automatic mask generation using SAM  ----- | 1-9 | 9    |      |
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

```
3. 決策：決定 Background (BG)
這是第二層決策，在 get_bg_fg_from_main_clusters 函數中執行。此時，程式只專注於 main_features (主要群組)。

計算主要特徵： 程式會計算這些「主要群組」各自的平均 avg_texture 和 avg_saturation。

排序候選者 (The Decision Tree)： 程式根據「主要群組」的數量 (k_main) 來決定**「背景候選者」的排序**：

如果 k_main == 1： 只有一個主要群組，它自動成為背景候選者。

如果 k_main == 2：

計算兩個群組的飽和度差異 (sat_diff)。

如果 sat_diff <= 100 (飽和度相似)：程式假設紋理是主要差異。背景候選者 = 紋理最高 (Max Texture) 的群組。

如果 sat_diff > 100 (飽和度差異大)：程式假設飽和度是主要差異。背景候選者 = 紋理最低 (Min Texture) 的群組。

如果 k_main >= 3：

邏輯簡化。背景候選者 = 紋理最高 (Max Texture) 的群組。

資格審查 (Area Veto)：

程式會從「排序第一」的背景候選者開始檢查。

這個候選者必須通過面積審查：Area >= MIN_BG_AREA_PERCENT (在您的 code 中設定為 30%)。

如果「候選者 #1」的面積 >= 30%：它被正式選為 Background。

如果「候選者 #1」的面積 < 30%：它失去資格。程式會接著檢查「候選者 #2」（如果有的話），看它是否 >= 30%。

如果所有候選者都 < 30%：程式會「妥協」，並強制選擇「候選者 #1」 (無論其面積多小) 作為背景，並在日誌中留下一條警告。
```


參考 # ----- 5.(version 6) - Image automatic mask generation using SAM  -----
```
關於Colab code想做些修正請提供新的code: 在遞迴式例外排除 (Iterative Outlier Removal), 分群後，程式會計算每一個 cluster面積, 如果一個群組的總面積 < 10%(這是個參數)被存放到例外遮罩而且就不會再加入到main_features. 當決策：決定 Background (BG)時我們只專注於 main_features (主要群組), 這時我們選主要群組的前兩大面積群組. 如果第一大群組跟第二大群組的面積差不大於20%, 我們選取紋理最高 (Max Texture) 的群組為background. 如果第一大群組跟第二大群組的面積差大於20%, 我們選取由x_min跟x_max, y_min跟y_max定義的box最大群組為background. 而在選取Foreground時也不考慮例外遮罩.

```
-> # ----- 5.(version 7) - Image automatic mask generation using SAM  -----



參考 # ----- 5.(version 7) - Image automatic mask generation using SAM  -----
```
請根據以下的code根據修正提供新的code: 

請修改SAM參數因為有些細節的區域沒有正確segmentation.

情境 B (面積懸殊)：如果 area_diff > 0.20 (20%), 邏輯：既然面積差異很大，就用「空間佔用率 (Bbox)」來決勝負。

決策： 比較 cluster_1 和 cluster_2 的 bbox_area，Bbox 面積最大的那個群組被選為 Background。這裡Bbox改成用cluster pixel number也就是面積最大的那個群組被選為 Background而不是bbox area.
```
-> # ----- 5.(version 8) - Image automatic mask generation using SAM  -----


-> # ----- 5.(version 8) - Image automatic mask generation using SAM  -----
```
以下是version 8colab code. 想多增加一個小功能. 就是最後決定了background跟其他的frontground, 請計算texture(std), texture(entropy)跟saturation跟contrast. 並把結果列在txt file

這個colab codefigure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分使用now = datetime.datetime.now(ZoneInfo("America/New_York")). 然後再把這個subfolder名字後面加上"_dialtext_result(v6)". 把figuers儲存存在這個新subfolder. 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文. 並注意不要犯這個錯誤: module 'cv2' has no attribute 'COLOR_BGR_RGB'
```
-> # ----- 5.(version 9) - Image automatic mask generation using SAM  -----






#### Project 7 - Movement dimensions1

|                                                                                       | All | Best | Note |
| ------------------------------------------------------------------------------------- | --- | ---- | ---- |
| # ----- 1. Import dependent -----                                                     | 1   | 1    |      |
| # ----- 2. Project folder and device/device setup -----                               | 1   | 1    |      |
| # ----- 3. images loading: movement dimensions1 -----                                 | 1   | 1    |      |
| # ----- 4. calculate image features summary                                           | 1   | 1    |      |
| ~~# ----- 5a. Clone and install DINOv3 repo and load checkpoints  -----~~                 | ~~1~~   | ~~1~~    |      |
| ~~# ----- 5b. Create DINOv3 model with weights and related functions  -----~~             | ~~1~~   | ~~1~~    |      |
| ~~# ----- 5c.(version 3) use self-supervised learning model (DINOv3) -----~~              | ~~3~~   | ~~3~~    |      |
| ~~# ----- 5d. (version 12) DINOv3 Mask Analysis -----~~                                   | ~~15~~  | ~~15~~   |      |
| # ----- 6. (version 6) Example-Based Segmentation (Histogram Backprojection) -----    | 6   | 6    |      |
| # ----- 7. (version 1) Example-Based Segmentation (Skeleton Geometric Analysis) ----- | 1   | 1    |      |

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



基於# ----- 5d. (version 12) DINOv3 Mask Analysis -----
```
這個Colab將 Step 9: Apply Spline Fitting for final smoothing使用fit_spline_to_contour做輪廓平滑化參考下列方法:
1.使用 cv2.approxPolyDP 找到輪廓的「斷點」(breakpoints)。
2.將原始輪廓（有鋸齒的）分割成由這些斷點定義的多個「點集」。
3.對每個「點集」嘗試擬合一條直線。
4.如果 擬合得足夠好（低於 PIECEWISE_LINE_FIT_THRESH 閾值），則用一條完美的直線取代該線段。
5.如果 擬合得不好（即它是一條曲線），則退回 (fallback) 到使用 B-spline 僅對該線段進行平滑。
6.最後，將所有這些「完美的直線」和「平滑的曲線」無縫地縫合回一個單一的、極度平滑的輪廓。

這個colab code一樣figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分使用now = datetime.datetime.now(ZoneInfo("America/New_York")). 然後再把這個subfolder名字後面加上"_movement1_dinov3_bone_result(v13)". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文.
```
-> # ----- 5d. (version 13) DINOv3 Mask Analysis -----

11050230_movement1_dinov3_result(v3)
11050233_movement1_dinov3_bone_result(v12)  比較v12跟v13
11050245_movement1_dinov3_bone_result(v13)
還是很不平滑

基於-> # ----- 5d. (version 13) DINOv3 Mask Analysis -----
```
請將這Colab code修改將 Step 9: Apply Spline Fitting for final smoothing使用fit_spline_to_contour做輪廓平滑化參考下列方法:
用了opening跟closing之後用三種不同method進行(有parameter選擇哪個method) method 1. 提取邊界cv2.findContours, 接著用Gaussian filter 平滑邊界座標, 接著用重建 mask. method 2. 提取邊界cv2.findContours, 接著用B-spline 平滑邊界座標, 接著用重建 mask. method 3. 用from scipy.ndimage import distance_transform_edt, gaussian_filter使用Distance transform + Gaussian filter + threshold

這個colab code一樣figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分使用now = datetime.datetime.now(ZoneInfo("America/New_York")). 然後再把這個subfolder名字後面加上"_movement1_dinov3_bone_result(v13)". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文.


這個colab code一樣figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分使用now = datetime.datetime.now(ZoneInfo("America/New_York")). 然後再把這個subfolder名字後面加上"_movement1_dinov3_bone_result(v13)". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文.
```
-> # ----- 5d. (version 14) DINOv3 Mask Analysis -----

11050257_movement1_dinov3_bone_result(v14)

11050257_movement1_dinov3_bone_result(v14)
	FINAL_SMOOTHING_METHOD = "method_2"
	"method_2_spline_s": 10000.0,
	method_2_spline_n_pts": 1000,
還是很不平滑

11050307_movement1_dinov3_bone_result(v14)
	FINAL_SMOOTHING_METHOD = "method_2"
	"method_2_spline_s": 50000.0,
	method_2_spline_n_pts": 1000,
還是很不平滑

11050314_movement1_dinov3_bone_result(v14)
	INITIAL_SMOOTHING_KERNEL_SIZE = (31, 31)
	DILATION_KERNEL_SIZE = (40, 40)
	CONTRAST_GAUSSIAN_SMOOTH_KERNEL_SIZE = (101, 101)
	OPENING_FILTER_KERNEL_SIZE = (21, 21)
	CLOSING_FILTER_KERNEL_SIZE = (61, 61)
	FINAL_SMOOTHING_METHOD = "method_3"
	"method_3_dt_gaussian_sigma": 50.0

11050328_movement1_dinov3_bone_result(v14)
	INITIAL_SMOOTHING_KERNEL_SIZE = (31, 31)
	DILATION_KERNEL_SIZE = (40, 40)
	CONTRAST_GAUSSIAN_SMOOTH_KERNEL_SIZE = (101, 101)
	OPENING_FILTER_KERNEL_SIZE = (21, 21)
	CLOSING_FILTER_KERNEL_SIZE = (61, 61)
	FINAL_SMOOTHING_METHOD = "method_2"
	"method_2_spline_s": 1000000.0
	"method_2_spline_n_pts": 1000

參考 # ----- 5d. (version 14) DINOv3 Mask Analysis -----
```
請將這Colab code修改為不是參考contrast map來進行fine segmentation, 而是有選項可以選1. Texture(std)(字體輪廓texture小), 2. Hue(字體輪廓texture大), 3. Saturation(字體輪廓texture大), 4. Contrast(range)(字體輪廓texture小), 5. sharpness(字體輪廓texture小). 其他地方保持不變. 

這個colab code一樣figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分使用now = datetime.datetime.now(ZoneInfo("America/New_York")). 然後再把這個subfolder名字後面加上"_movement1_dinov3_bone_result(v14)". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文.

version 14的code怎麼沒有輪廓光滑的流程: 用了opening跟closing之後用三種不同method進行(有parameter選擇哪個method) method 1. 提取邊界cv2.findContours, 接著用Gaussian filter 平滑邊界座標, 接著用重建 mask. method 2. 提取邊界cv2.findContours, 接著用B-spline 平滑邊界座標, 接著用重建 mask. method 3. 用from scipy.ndimage import distance_transform_edt, gaussian_filter使用Distance transform + Gaussian filter + threshold?
```
-> # ----- 5d. (version 15) DINOv3 Mask Analysis -----


11050408_movement1_dinov3_bone_result(v15)
	FEATURE_MAP_METHOD = "texture"
	FEATURE_MAP_GAUSSIAN_SMOOTH_KERNEL = (101, 101)
	    "texture": {
	        "kernel_size": (11, 11),
	        "threshold_type": cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
結果平滑很多

11050748_movement1_dinov3_result(v3)
	20張images裡面有一張DINOv3 PCA1 segmentation有問題, 其他都還好
11050756_movement1_dinov3_bone_result(v15)
	FEATURE_MAP_METHOD = "texture"
	FEATURE_MAP_GAUSSIAN_SMOOTH_KERNEL = (101, 101)
	    "texture": {
	        "kernel_size": (11, 11),
	        "threshold_type": cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
	INITIAL_SMOOTHING_KERNEL_SIZE = (31, 31)
	DILATION_KERNEL_SIZE = (40, 40)
	OPENING_FILTER_KERNEL_SIZE = (21, 21)
	CONTOUR_SMOOTHING_KERNEL_SIZE_MID = (11, 11)
	CLOSING_FILTER_KERNEL_SIZE = (61, 61)
	FINAL_SMOOTHING_METHOD = "method_3"
	"method_3_dt_gaussian_sigma": 50.0

11050918_movement1_dinov3_bone_result(v15)
	FEATURE_MAP_METHOD = "texture"
	FEATURE_MAP_GAUSSIAN_SMOOTH_KERNEL = **(51, 51)**
	    "texture": {
	        "kernel_size": (11, 11),
	        "threshold_type": cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
	INITIAL_SMOOTHING_KERNEL_SIZE = (31, 31)
	DILATION_KERNEL_SIZE = (40, 40)
	OPENING_FILTER_KERNEL_SIZE = (21, 21)
	CONTOUR_SMOOTHING_KERNEL_SIZE_MID = (11, 11)
	CLOSING_FILTER_KERNEL_SIZE = (61, 61)
	FINAL_SMOOTHING_METHOD = "method_3"
	"method_3_dt_gaussian_sigma": 50.0

11050932_movement1_dinov3_bone_result(v15)
	FEATURE_MAP_METHOD = "texture"
	FEATURE_MAP_GAUSSIAN_SMOOTH_KERNEL = (101, 101)
	    "texture": {
	        "kernel_size": (11, 11),
	        "threshold_type": cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
	INITIAL_SMOOTHING_KERNEL_SIZE = (31, 31)
	DILATION_KERNEL_SIZE = (40, 40)
	OPENING_FILTER_KERNEL_SIZE = (21, 21)
	CONTOUR_SMOOTHING_KERNEL_SIZE_MID = (11, 11)
	CLOSING_FILTER_KERNEL_SIZE = (61, 61)
	FINAL_SMOOTHING_METHOD = **"method_2"**
	"method_2_spline_s": 1000000.0
	"method_2_spline_n_pts": 1000

11051017_movement1_dinov3_bone_result(v15)
	FEATURE_MAP_METHOD = "texture"
	FEATURE_MAP_GAUSSIAN_SMOOTH_KERNEL = (101, 101)
	    "texture": {
	        "kernel_size": (11, 11),
	        "threshold_type": cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
	INITIAL_SMOOTHING_KERNEL_SIZE = (31, 31)
	DILATION_KERNEL_SIZE = (40, 40)
	OPENING_FILTER_KERNEL_SIZE = (21, 21)
	CONTOUR_SMOOTHING_KERNEL_SIZE_MID = (11, 11)
	CLOSING_FILTER_KERNEL_SIZE = (61, 61)
	FINAL_SMOOTHING_METHOD = **"method_3"**
	"method_3_dt_gaussian_sigma": 50.0

11051032_movement1_dinov3_bone_result(v15)
	FEATURE_MAP_METHOD = "texture"
	FEATURE_MAP_GAUSSIAN_SMOOTH_KERNEL = (101, 101)
	    "texture": {
	        "kernel_size": (11, 11),
	        "threshold_type": cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
	INITIAL_SMOOTHING_KERNEL_SIZE = (31, 31)
	DILATION_KERNEL_SIZE = **(80, 80)**
	OPENING_FILTER_KERNEL_SIZE = (21, 21)
	CONTOUR_SMOOTHING_KERNEL_SIZE_MID = (11, 11)
	CLOSING_FILTER_KERNEL_SIZE = (61, 61)
	FINAL_SMOOTHING_METHOD = "method_3"
	"method_3_dt_gaussian_sigma": 50.0

11051032_movement1_dinov3_bone_result(v15)
	FEATURE_MAP_METHOD = **"saturation"**
	FEATURE_MAP_GAUSSIAN_SMOOTH_KERNEL = (101, 101)
	    "texture": {
	        "kernel_size": (11, 11),
	        "threshold_type": cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
	INITIAL_SMOOTHING_KERNEL_SIZE = (31, 31)
	DILATION_KERNEL_SIZE = (80, 80)
	OPENING_FILTER_KERNEL_SIZE = (21, 21)
	CONTOUR_SMOOTHING_KERNEL_SIZE_MID = (11, 11)
	CLOSING_FILTER_KERNEL_SIZE = (61, 61)
	FINAL_SMOOTHING_METHOD = "method_3"
	"method_3_dt_gaussian_sigma": 50.0

11051056_movement1_dinov3_bone_result(v15)
	**THREAD_RATIO = 0.6**
	FEATURE_MAP_METHOD = "saturation"
	FEATURE_MAP_GAUSSIAN_SMOOTH_KERNEL = (101, 101)
	    "texture": {
	        "kernel_size": (11, 11),
	        "threshold_type": cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
	INITIAL_SMOOTHING_KERNEL_SIZE = (31, 31)
	DILATION_KERNEL_SIZE = (80, 80)
	OPENING_FILTER_KERNEL_SIZE = (21, 21)
	CONTOUR_SMOOTHING_KERNEL_SIZE_MID = (11, 11)
	CLOSING_FILTER_KERNEL_SIZE = (61, 61)
	FINAL_SMOOTHING_METHOD = "method_3"
	"method_3_dt_gaussian_sigma": 50.0

11051105_movement1_dinov3_bone_result(v15)
	**THREAD_RATIO = 0.6**
	FEATURE_MAP_METHOD = "saturation"
	FEATURE_MAP_GAUSSIAN_SMOOTH_KERNEL = (21, 21)
	    "texture": {
	        "kernel_size": (11, 11),
	        "threshold_type": cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
	INITIAL_SMOOTHING_KERNEL_SIZE = (31, 31)
	DILATION_KERNEL_SIZE = (80, 80)
	OPENING_FILTER_KERNEL_SIZE = (21, 21)
	CONTOUR_SMOOTHING_KERNEL_SIZE_MID = (11, 11)
	CLOSING_FILTER_KERNEL_SIZE = (61, 61)
	FINAL_SMOOTHING_METHOD = "method_2"
	"method_2_spline_s": 5000000.0
	"method_2_spline_n_pts": 1000

11051108_movement1_dinov3_bone_result(v15)
	**THREAD_RATIO = 0.6**
	FEATURE_MAP_METHOD = **"hue"**
	FEATURE_MAP_GAUSSIAN_SMOOTH_KERNEL = (21, 21)
	    "texture": {
	        "kernel_size": (11, 11),
	        "threshold_type": cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
	INITIAL_SMOOTHING_KERNEL_SIZE = (31, 31)
	DILATION_KERNEL_SIZE = (80, 80)
	OPENING_FILTER_KERNEL_SIZE = (21, 21)
	CONTOUR_SMOOTHING_KERNEL_SIZE_MID = (11, 11)
	CLOSING_FILTER_KERNEL_SIZE = (61, 61)
	FINAL_SMOOTHING_METHOD = "method_2"
	"method_2_spline_s": 5000000.0
	"method_2_spline_n_pts": 1000

11051110_movement1_dinov3_bone_result(v15)
	**THREAD_RATIO = 0.6**
	FEATURE_MAP_METHOD = **"contrast"**
	FEATURE_MAP_GAUSSIAN_SMOOTH_KERNEL = (21, 21)
	    "texture": {
	        "kernel_size": (11, 11),
	        "threshold_type": cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
	INITIAL_SMOOTHING_KERNEL_SIZE = (31, 31)
	DILATION_KERNEL_SIZE = (80, 80)
	OPENING_FILTER_KERNEL_SIZE = (21, 21)
	CONTOUR_SMOOTHING_KERNEL_SIZE_MID = (11, 11)
	CLOSING_FILTER_KERNEL_SIZE = (61, 61)
	FINAL_SMOOTHING_METHOD = "method_2"
	"method_2_spline_s": 5000000.0
	"method_2_spline_n_pts": 1000

11051115_movement1_dinov3_bone_result(v15)
	**THREAD_RATIO = 0.6**
	FEATURE_MAP_METHOD = **"texture"**
	FEATURE_MAP_GAUSSIAN_SMOOTH_KERNEL = (21, 21)
	    "texture": {
	        "kernel_size": (11, 11),
	        "threshold_type": cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
	INITIAL_SMOOTHING_KERNEL_SIZE = (31, 31)
	DILATION_KERNEL_SIZE = (80, 80)
	OPENING_FILTER_KERNEL_SIZE = (21, 21)
	CONTOUR_SMOOTHING_KERNEL_SIZE_MID = (11, 11)
	CLOSING_FILTER_KERNEL_SIZE = (61, 61)
	FINAL_SMOOTHING_METHOD = "method_2"
	"method_2_spline_s": 5000000.0
	"method_2_spline_n_pts": 1000


```
用Example-Based Segmentation (Histogram Backprojection)方法
但改成不取最大的segment, 而改以刪除min_max box最大的(frame)
```
-> # ----- 6. (version 1) Example-Based Segmentation (Histogram Backprojection) -----

11051143_lume_histBack_result(v7)
	只有positive images不少都不錯, 但有一些background也在裡面

11051155_lume_histBack_result(v7)
	加入negative images還是沒有segmentation masks

11051157_lume_histBack_result(v7)
	POSITIVE_REFERENCE_PATHS = [movement_image_paths[1], movement_image_paths[0]]
	POSITIVE_ROI_FACTORS_LIST = [[0.41, 0.49, 0.26, 0.35, 1.0],[0.56, 0.64, 0.25, 0.33, 1.0]]
	AUTO_THRESHOLD = True
	AUTO_THRESHOLD_RATIO = 0.8
	結果很好


-參考 # ----- 6. (version 1) Example-Based Segmentation (Histogram Backprojection) -----
```
這是用2D histogram方法建立segmentation mask請基於這個做修正提供新的code. 在Step 4.6: Find Largest Mask Component這一步(main_mask = find_largest_mask(final_mask))變成計算每個獨立連續的segment區域計算x_min, x_max, y_min, y_max. 將這四個座標定義的box面積最大的獨立連續的segment先刪除, 然後取第二大的獨立連續的segment為final segmentaiton mask(也就是數字3). 對最後得到的final mask得到輪廓線, 並用直線和曲線去fit這條輪廓線(某些線段是直線, 有些線段是曲線有圓心跟固定radius), 記錄下每個直線段的頭尾座標以及距離, 直線部分跟明顯轉折點以及轉折的角度都計算出來. 曲線的頭尾座標, 圓心座標跟radius以及線段長. 接著計算mask的skeleton以及這個skeleton line到數字mask boundary的距離(thickness), 可以用在skeleton每隔一段距離的座標點並記下thickness. 還有skeleton line的頭跟尾各有一條接近垂直於skeleton line的直線, 輸出這兩條直線(頭跟尾)的夾角. 另外檢查這個mask的skeleton line的1/3*thickness區域在image上有明顯的黑色區域, 則這個mask類型是"type 1", 如果沒有則是"type 2".

在已得到的skeleton, 找尋skeleton中間最靠近的中間輪廓直線. 並計算之前計算過的skeleton兩條直線(頭跟尾)接近垂直於skeleton line的直線跟中間輪廓直線的夾角
```
-> # ----- 6. (version 2) Example-Based Segmentation (Histogram Backprojection) -----


參考# ----- 6. (version 2) Example-Based Segmentation (Histogram Backprojection) -----
```
這是用2D histogram方法建立segmentation mask請基於這個做修正提供新的code. 將找尋final mask的方式從計算x_min, x_max, y_min, y_max去掉最大區域的方式修正. 先找由2D histogram方法得到的coarse segmentation mask尋找五個最大獨立連續的segment. 從image的1/32*size,1/32*size中心區域開始從這五個之內找尋獨立連續的segment區域. 如果在這區域有找到獨立連續的segment的pixel(如果有兩個同時出現則取獨立連續的segment有最大pixel面積)則這獨立連續的segment就是final segmentation mask. 如果都沒有則擴大搜尋區域到1/16size, 搜尋區域到1/8size, 搜尋區域到1/4size, 搜尋區域到1/2size,搜尋區域到all image. final segmentation mask用B-spline（樣條曲線）平滑 segmentation mask 的邊界, spline_s參數用5000.

對最後得到的final mask得到輪廓線, 並用直線和曲線去fit這條輪廓線(某些線段是直線, 有些線段是曲線有圓心跟固定radius), 記錄下每個直線段的頭尾座標以及距離, 直線部分跟明顯轉折點以及轉折的角度都計算出來. 曲線的頭尾座標, 圓心座標跟radius以及線段長. 接著計算mask的skeleton以及這個skeleton line到數字mask boundary的距離(thickness), 可以用在skeleton每隔一段距離的座標點並記下thickness. 還有skeleton line的頭跟尾各有一條接近垂直於skeleton line的直線, 輸出這兩條直線(頭跟尾)的夾角. 另外檢查這個mask的skeleton line的1/3*thickness區域在image上有明顯的黑色區域, 則這個mask類型是"type 1", 如果沒有則是"type 2".在已得到的skeleton, 找尋skeleton中間最靠近的中間輪廓直線. 並計算之前計算過的skeleton兩條直線(頭跟尾)接近垂直於skeleton line的直線跟中間輪廓直線的夾角. 也請提供skeleton及相關的figures. 另外執行時有出現WARNING:matplotlib.font_manager:findfont: Generic family 'sans-serif' not found because none of the following families were found: Noto Sans CJK TC請解決

這個新增加的colab code跟上一個colab code一樣figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分使用now = datetime.datetime.now(ZoneInfo("America/New_York")). 然後再把這個subfolder名字後面加上"_movement1_histBack_result(v3)". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文. 並注意不要犯這個錯誤: module 'cv2' has no attribute 'COLOR_BGR_RGB'

```
-> # ----- 6. (version 3) Example-Based Segmentation (Histogram Backprojection) -----

11051308_movement1_histBack_result(v4)
	FEATURE_CHOICES = [1, 2]
	POSITIVE_REFERENCE_PATHS = [movement_image_paths[1], movement_image_paths[0]]
	POSITIVE_ROI_FACTORS_LIST = [[0.41, 0.49, 0.26, 0.35, 1.0], ][0.56, 0.64, 0.25, 0.33, 1.0]
	FINAL_SPLINE_S = 5000

11051312_movement1_histBack_result(v4)
	FEATURE_CHOICES = [1, 2]
	POSITIVE_REFERENCE_PATHS = [movement_image_paths[1], movement_image_paths[0]]
	POSITIVE_ROI_FACTORS_LIST = [[0.41, 0.49, 0.26, 0.35, 1.0], ][0.56, 0.64, 0.25, 0.33, 1.0]
	FINAL_SPLINE_S = 50000

11051328_movement1_histBack_result(v3)
	FEATURE_CHOICES = [1, 2]
	POSITIVE_REFERENCE_PATHS = [movement_image_paths[1], movement_image_paths[0]]
	POSITIVE_ROI_FACTORS_LIST = [[0.41, 0.49, 0.26, 0.35, 1.0], ][0.56, 0.64, 0.25, 0.33, 1.0]
	SMOOTHING_METHOD = "bspline_contour"

11051338_movement1_histBack_result(v3)
	FEATURE_CHOICES = [1, 2]
	POSITIVE_REFERENCE_PATHS = [movement_image_paths[1], movement_image_paths[0]]
	POSITIVE_ROI_FACTORS_LIST = [[0.41, 0.49, 0.26, 0.35, 1.0], ][0.56, 0.64, 0.25, 0.33, 1.0]
	SMOOTHING_METHOD = "distance_transform"

11051348_movement1_histBack_result(v3)
	FEATURE_CHOICES = [1, 2]
	POSITIVE_REFERENCE_PATHS = [movement_image_paths[1], movement_image_paths[0]]
	POSITIVE_ROI_FACTORS_LIST = [[0.41, 0.49, 0.26, 0.35, 1.0], ][0.56, 0.64, 0.25, 0.33, 1.0]
	SMOOTHING_METHOD = "bspline_contour"
	CLOSE_KERNEL_SIZE = (15, 15)
	OPEN_KERNEL_SIZE = (3, 3)
	SMOOTHING_SPLINE_S = 100000

11051352_movement1_histBack_result(v3)
	FEATURE_CHOICES = [1, 2]
	POSITIVE_REFERENCE_PATHS = [movement_image_paths[1], movement_image_paths[0]]
	POSITIVE_ROI_FACTORS_LIST = [[0.41, 0.49, 0.26, 0.35, 1.0], ][0.56, 0.64, 0.25, 0.33, 1.0]
	SMOOTHING_METHOD = "bspline_contour"
	CLOSE_KERNEL_SIZE = (15, 15)
	OPEN_KERNEL_SIZE = (3, 3)
	SMOOTHING_SPLINE_S = 1000



參考 # ----- 6. (version 3) Example-Based Segmentation (Histogram Backprojection) -----
```
這是一段平滑化final segmentation mask輪廓的code為接下來skeleton做準備. 請用這段code的方法替代# ----- 6. (version 5 - MODIFIED) Example-Based Segmentation (Histogram Backprojection) -----裡面平滑化final segmentation mask輪廓的方法並提供新的code:def smooth_contour_gaussian(contour, sigma):

    """Smooths contour coordinates using a 1D Gaussian filter."""

    # Separate x and y coordinates

    x = contour[:, 0, 0].astype(float)

    y = contour[:, 0, 1].astype(float)

    # Apply Gaussian filter. 'wrap' handles closed contours.

    x_smooth = gaussian_filter(x, sigma, mode='wrap')

    y_smooth = gaussian_filter(y, sigma, mode='wrap')

    # Re-stack and format

    new_contour = np.vstack((x_smooth, y_smooth)).T.astype(np.int32).reshape((-1, 1, 2))

    return new_contour

  

# --- [NEW] Final Smoothing Method 2: B-Spline (v10/v11/v12 logic) ---

def smooth_contour_spline(contour, s, n_pts):

    """Smooths contour coordinates using a B-spline fit."""

    if len(contour) < 5: # splprep needs at least k+1

        return contour

    x = contour[:, 0, 0]

    y = contour[:, 0, 1]

    try:

        tck, u = splprep([x, y], s=s, per=True)

    except Exception as e:

        print(f"  Spline Fitting Warning: splprep failed ({e}). Returning original contour.")

        return contour

    u_new = np.linspace(u.min(), u.max(), n_pts)

    x_new, y_new = splev(u_new, tck, der=0)

    new_contour = np.vstack((x_new, y_new)).T.astype(np.int32).reshape((-1, 1, 2))

    return new_contour

  

# --- [NEW] Final Smoothing Method 3: Distance Transform Blur ---

def smooth_mask_distance_transform(mask, sigma):

    """Smooths a mask by blurring its Signed Distance Function (SDF)."""

    # Positive distance (inside)

    dist_in = distance_transform_edt(mask)

    # Negative distance (outside)

    dist_out = distance_transform_edt(cv2.bitwise_not(mask))

    # Create Signed Distance Function (SDF)

    signed_dist = dist_in - dist_out

    # Blur the SDF

    smoothed_sdf = gaussian_filter(signed_dist, sigma)

    # Re-threshold to get the new binary mask

    new_mask = (smoothed_sdf > 0).astype(np.uint8) * 255

    return new_mask

  

# --- [NEW] Final Smoothing Selector Function ---

def apply_final_smoothing(mask, method, params):

    """

    Applies the selected final smoothing method to the mask.

    """

    print(f"  Applying final smoothing with: {method}")

    if method == "method_3":

        # This method operates on the whole mask

        sigma = params["method_3_dt_gaussian_sigma"]

        return smooth_mask_distance_transform(mask, sigma)

  

    # Methods 1 and 2 operate on the contour

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:

        print("  Final Smoothing: No contour found.")

        return mask

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    cnt = contours[0]

    if method == "method_1":

        sigma = params["method_1_gaussian_sigma"]

        new_contour = smooth_contour_gaussian(cnt, sigma)

    elif method == "method_2":

        s = params["method_2_spline_s"]

        n_pts = params["method_2_spline_n_pts"]

        new_contour = smooth_contour_spline(cnt, s, n_pts)

    else:

        print(f"  Warning: Unknown smoothing method '{method}'. Returning original mask.")

        return mask

    # Re-draw the new contour to create the final mask

    new_mask = np.zeros_like(mask)

    cv2.drawContours(new_mask, [new_contour], -1, 255, -1)

    return new_mask
```
-> # ----- 6. (version 4) Example-Based Segmentation (Histogram Backprojection) -----

11051411_movement1_histBack_result(v4)
	FINAL_SMOOTHING_METHOD = "method_2"
	"method_2_spline_s": 50000
	"method_2_spline_n_pts": 1000

11051424_movement1_histBack_result(v4)
	FINAL_SMOOTHING_METHOD = "method_3"
	"method_2_spline_s": 50000
	"method_2_spline_n_pts": 1000


參考 # ----- 6. (version 4) Example-Based Segmentation (Histogram Backprojection) -----
```
修改下面的code把apply_smoothing 平滑化輪廓三種方法除了原本的方法之外多加幾種多階段整合方法, 因為結果顯示出現局部突起/毛刺(segmentation mask是雕刻文字"3"(大小約1500x1500pixel, thickness約250pixel)的輪廓及使用spline方法輪廓上還是有些20~30pixel大小突起, 以及一些細長型突起~10x40pixel.). 所以第四種方法就是先用用 B-spline 平滑大形狀, 再用 distance-map + Gaussian 修掉細小突起, 最後用中型 kernel 的 opening 清乾淨毛邊. 如果有其他推薦的方法也可以加進來. code其他部分不改變. 另外我用"method_3"時候內部常出現小hole. 請加上把內部hole填滿.

  

這個新增加的colab code跟上一個colab code一樣figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分使用now = datetime.datetime.now(ZoneInfo("America/New_York")). 然後再把這個subfolder名字後面加上"_movement1_histBack_result(v5)". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文. 並注意不要犯這個錯誤: module 'cv2' has no attribute 'COLOR_BGR_RGB':
```
-> # ----- 6. (version 5) Example-Based Segmentation (Histogram Backprojection) -----

11051441_movement1_histBack_result(v5)
	FINAL_SMOOTHING_METHOD = "method_4"
	# (NEW v7) Params for Method 4 (Spline -> SDF -> Opening)
	"method_4_spline_s": 50000
	"method_4_spline_n_pts": 1000
	"method_4_dt_gaussian_sigma": 3
	"method_4_opening_kernel_size": 7

11051447_movement1_histBack_result(v5)
	FINAL_SMOOTHING_METHOD = "method_4"
	# (NEW v7) Params for Method 4 (Spline -> SDF -> Opening)
	"method_4_spline_s": 50000
	"method_4_spline_n_pts": 1000
	"method_4_dt_gaussian_sigma": 3
	"method_4_opening_kernel_size": 7
**segmentation mask及平滑結果很不錯. 使用這個方法跟設定**



-> # ----- 6. (version 5) Example-Based Segmentation (Histogram Backprojection) -----
```
目前這個Colab code segmentation mask跟輪廓平滑化結果還不錯. 至於後續的skeleton跟geometric analysis我想把它放在下一個colab cell裡. 也就是在原本的colab cell完成segmentation mask跟輪廓平滑化跟輸出, 將結果在下一個colab cell執行後續分析. 另外關於這個雕刻文字"3"(大小約1500x1500pixel, thickness約250pixel)有關於skeleton兩條直線(頭跟尾)跟中間輪廓直線計算角度的分析用下面的方法取代. 雕刻文字"3"的skeleton line應該會有三個頂點,一個是3這個字的頭跟尾, 以及中間. 請在這三個頂點的200pixel範圍內的輪廓上的點, 設法去fit至少100pixel長的最長直線, 如果沒有夠長直線則嘗試用曲線去fit. 如果有直線就提供頭尾座標跟長度, 如果是曲線則提供頭尾座標跟長度根圓心跟radius. 並把直線跟曲線標示在圖上並如果有直線計算他們之間夾角.

```
-> # ----- 6. (version 6) Example-Based Segmentation (Histogram Backprojection) -----
-> # ----- 7. (version 1) Example-Based Segmentation (Skeleton) -----


11051517_movement1_histBack_result(v8)
測試ok

11051535_movement1_histBack_result(v8)
	20張images for presentation












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
| # ----- 5d. (version 1) Smooth  DINOv3 segmentation mask and follow analysis----- | 1-8 | 8    |      |
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





參考# ----- 5d.(version 3) Refine Mask with Advanced Saturation Morphology -----
```
那請幫我按照你的方法步驟 1：準備 S-Map(不過這裡不只用saturation, 還有texture(std), contrast, sharpness可以有選項選擇), 步驟 2：應用「黑帽」轉換, 步驟 3：二值化並清理, 步驟 4：連接並填充輪廓. 另外另個選項就是cv2.grabCut以及preprocessing及proprocessing. 也提供詳細流程figures.
```
-> # ----- 5d.(version 8) Refine Mask with Advanced Saturation Morphology -----


11052010_movement2_dinov3_ske_result(v4_blackhat)
	METHOD_CHOICE = 'BLACKHAT'
	BH_SATURATION_BLUR_RADIUS = 5
	BH_KERNEL_SIZE = 41
	BH_CLEAN_KERNEL_SIZE = 7
	BH_CONNECT_KERNEL_SIZE = 21
	BH_FINAL_SMOOTH_KERNEL_SIZE = 5
	GC_DEFINITE_FG_ERODE_KSIZE = 15
	GC_ITERATIONS = 5
	GC_CONNECT_KERNEL_SIZE = 21
	GC_FINAL_SMOOTH_KERNEL_SIZE = 5

11052015_movement2_dinov3_ske_result(v4_blackhat)
	METHOD_CHOICE = 'GRABCUT'
Too slow and result bad

11052043_movement2_dinov3_ske_result(v5_blackhat_texture_std)
	METHOD_CHOICE = 'BLACKHAT'
	FEATURE_MAP_CHOICE = TEXTURE_STD

11052048_movement2_dinov3_ske_result(v5_blackhat_texture_std)
	METHOD_CHOICE = 'BLACKHAT'
	FEATURE_MAP_CHOICE = 'CONTRAST_GRADIENT'



```
# ----- 5d.(version 3) Refine Mask with Advanced Saturation Morphology -----
表現好, 試著調整參數
```

11052104_movement2_dinov3_ske_result(v3)
	DILATION_KERNEL_SIZE = 100
	SATURATION_GAUSSIAN_RADIUS = 10
	INITIAL_SMOOTH_KSIZE = 5
	PARTICLE_CLEAN_KSIZE = 20
	HOLE_FILL_KSIZE = 15
	FINAL_SMOOTH_KSIZE = 40


參考# ----- 5d.(version 3) Refine Mask with Advanced Saturation Morphology -----
```
版本 version 9 = version 3
```
-> # ----- 5d.(version 9) Refine Mask with Advanced Saturation Morphology -----

11052104_movement2_dinov3_ske_result(v9)
	分割輪廓結果好 但沒有連結


參考 # ----- 5d.(version 9) Refine Mask with Advanced Saturation Morphology -----
```
不應該更換分割方法，而是應該修復 v3 的後處理邏輯
**主要變更：**
1. **全新的參數 (v6)**：我們刪除了 v3 的錯誤參數，換成了 `CLEAN_NOISE_KSIZE`、`CONNECT_GAP_KSIZE` 和 `FILL_SMOOTH_KSIZE`，並設定了正確的值。
    
2. **重寫 `refine_mask_based_on_saturation`**：
    - **保留** 步驟 1-5 (Otsu 分割)。
    - **步驟 6 (清理)**：使用 `CLEAN_NOISE_KSIZE = 7` (遠小於 15px)，只清除小噪點。
    - **步驟 8 (連接)**：使用 `CONNECT_GAP_KSIZE = 31` (遠大於 15px)，強力連接斷裂的輪廓。  
    - **步驟 9 (填充)**：新增 `cv2.findContours` + `cv2.drawContours(thickness=cv2.FILLED)` 來填充內部。
    - **步驟 10 (平滑)**：使用 `FILL_SMOOTH_KSIZE = 9` (小於 15px)，只平滑最終遮罩的邊緣，而不會刪除它。
        
3. **更新 `save_refinement_visualizations`**：使其顯示新的、正確的處理步驟。
```
-> # ----- 5d.(version 10) Refine Mask with Advanced Saturation Morphology -----

11052136_movement2_dinov3_ske_result(v10)
	THRESHOLD_RATIO = 1.0

11052144_movement2_dinov3_ske_result(v10)
	THRESHOLD_RATIO = 0.8







看起來目前使用11021835_movement2_dinov3_ske_result  (應該是5d. (version 3))
![[Pasted image 20251103075610.png]]



```
這個Colab code (5d.(version 3)) 會輸出一個final_refined_mask. 請新創一個新的colab cell繼續final_refined_mask下面的分析請提供code: 對最後得到的final_refined_mask得到輪廓線, 並用直線和曲線去fit這條輪廓線(某些線段是直線, 有些線段是曲線有圓心跟固定radius), 記錄下每個直線段的頭尾座標以及距離, 直線部分跟明顯轉折點以及轉折的角度都計算出來. 曲線的頭尾座標, 圓心座標跟radius以及線段長. 接著計算mask的skeleton以及這個skeleton line到數字mask boundary的距離(thickness), 可以用在skeleton每隔一段距離的座標點並記下thickness. 另外檢查這個mask的skeleton line的1/3*thickness區域在image上有明顯的黑色區域, 則這個mask類型是"type 1", 如果沒有則是"type 2". 

然後這個final mask是由幾個獨立mask(連續有接連的pixel區域)組成. 這是水平方向的幾個英文字數字, 或者是字體轉90度垂直方向的幾個英文字數字. 我們可以判斷這些字體是水平方向字體或垂直方向字體可以把這些獨立mask各計算中心座標, 然後看他們如果中心座標主要x方向間距大於 y方向間距那代表就是水平. x方向間距小於 y方向間距那代表就是垂直. 當判斷是水平方向字體, 則final mask的每個獨立mask(連續有接連的pixel區域)用x,y的min, max得到box的四個端點座標以及個別的height跟width, 以及用OCR去辨識是甚麼英文字或數字, 對每個box用原來的box以及轉180度再去用OCR去辨識並採最高confidence去決定text並輸出. 當判斷是垂直方向字體時, 先把每個box先轉90度, 之後的流程就跟水平方向字體流程一樣.

這個新增加的colab code跟上一個colab code一樣figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分使用now = datetime.datetime.now(ZoneInfo("America/New_York")). 然後再把這個subfolder名字後面加上"_movement2_dinov3_text_result(v1)". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文. 並注意不要犯這個錯誤: module 'cv2' has no attribute 'COLOR_BGR_RGB'
```
-> # ----- 5e. (version 1) DINOv3 fine segmentation Text Mask Analysis -----
