


|                                                 |     |
| ----------------------------------------------- | --- |
| [[#### Project 9 - Consulting 1]]               |     |
| [[#### Project 10 - Matlab Image Extraction]]   |     |
| [[#### Project 11 - Image Processing Pipeline]] |     |
| [[#### Project 12 - Database of results]]       |     |
| [[#### Project 13 - Crystal Crown]]             |     |
| [[#### Project 14 - Links]]                     |     |
| [[#### Project 15 - Movement Isolation]]        |     |
| [[#### Project 16 - Bracelet Pin]]              |     |
| [[#### Project 17 - Reading Case Side]]         |     |





#### Project 9 - Consulting 1




#### Project 10 - Matlab Image Extraction

Create and deliver python code that takes matlab structures of file information to extract the desired image for analysis. Matlab 圖像提取：編寫並提交 Python 程式碼，該程式碼能夠讀取 Matlab 文件結構信息，並從中提取所需圖像以進行分析。

a.      This should work for a specified general part type, detailed part type, image type, view, and/or other sets of 1 or more parameters. The image should be extracted to be used in a given processing algorithm. Edge cases and fault tolerances should be handled. 程式碼應適用於指定的通用零件類型、詳細零件類型、影像類型、檢視和/或其他一個或多個參數組合。提取的圖像應用於給定的處理演算法。程式碼應考慮邊界情況和容錯機制。

![[Pasted image 20251111074904.png]]


參考\ImagingLibWatch\App\ImagingWatch_ui.py
```
以下的code是我用pyside6-uic ImagingWatch.ui -o ImagingWatch_ui.py產生的ImagingWatch_ui.py. 請幫我依下面需求加上一些ui互動的functions.

1.  RadioButton_latest跟RadioButton_folder跟RadioButton_file是一組的而且跟stackedWidget_extraction connect, 當RadioButton_latest click的時候顯示stackedWidget_extraction connect的page_connect_latest, 當RadioButton_folder click的時候顯示stackedWidget_extraction connect的page_connect_folder, 當RadioButton_file click的時候顯示stackedWidget_extraction connect的page_connect_list.

2. 新增一個Search_image_folder(file_name, search_folder) function. 輸入有兩個: file_name是string, search_folder是path也是string. 輸出也是一個image_folder也是string, 代表在search_folder這個folder內找所有的sub folders(包含sub folders及之下的sub folders), 如果有找到則輸出這個sub folder path, 如果沒有則傳回None. 

3. 如果RadioButton_folder click, 按下pushButton_folder_image會跳出視窗可以選擇folder並儲存或更新這個path為變數image_extract_folder. 按下pushButton_folder_DB會跳出視窗可以選擇folder並儲存或更新這個path為變數db_extract_folder, 並新創或清空list變數db_extract_data, 並將db_extract_folder folder內的所有mat files的[file name, file path, []]存進去db_extract_data, 每個mat file i就是db_extract_data[i]. 並在listWidget_image加入這些db_extract_data[i][0]為項目, 而且CheckState為Checked. 按下PushButton_folder_extraction則會針對每個db_extract_data[i]執行Search_image_folder(db_extract_data[i][0], image_extract_folder)並將輸出存入db_extract_data[i][2].

4. 如果RadioButton_file click, 按下pushButton_file_image會跳出視窗可以選擇folder並儲存或更新這個path為變數image_file_folder. 按下pushButton_file_DB會跳出視窗可以選擇folder內的一個mat file並儲存或更新這個folder為db_extract_folder, 並新創或清空list變數db_extract_data, 並將這個mat files的[file name, file path, []]存進去db_extract_data, 這個mat file i就是db_extract_data[0]. 並在listWidget_image加入這個db_extract_data[0][0]為項目, 而且CheckState為Checked. 按下PushButton_file_extraction則會針對這個db_extract_data[0]執行Search_image_folder(db_extract_data[0][0], image_extract_folder)並將輸出存入db_extract_data[0][2]. 並將這個db_extract_data存入ImagingWatch_config.yaml

5. 如果RadioButton_latest click, 讀取ImagingWatch_config.yaml的變數db_extract_data. 然後textEdit_latest_text會顯示db_extract_data[0][0]. 按下PushButton_file_extraction則會針對這個db_extract_data[0]執行Search_image_folder(db_extract_data[0][0], image_extract_folder)並將輸出存入db_extract_data[0][2].
```


```
以下的code是我用pyside6-uic ImagingWatch.ui -o ImagingWatch_ui.py產生的ImagingWatch_ui.py. 請幫我依下面需求修改並加上一些ui互動的functions.

1. 新增一個變數App_root為ImagingWatch_ui.py所在的絕對path. ImagingWatch_config.yaml存取的path就是這裡. 


2. 在ImagingWatch_config.yaml裡面新增加另一個dictionary watchview_id. watchview_id = 
{
'glasspoint 1':'["watchentry"]["watchview"][0]["glasspoint"]["etchID"]'
'toppoint 1':'["watchentry"]["watchview"][0]["toppoint"]["topID"]'
'toppoint 2':'["watchentry"]["watchview"][1]["toppoint"]["topID"]'
'toppoint 3':'["watchentry"]["watchview"][2]["toppoint"]["topID"]'
'toppoint 4':'["watchentry"]["watchview"][3]["toppoint"]["topID"]'
'toppoint 5':'["watchentry"]["watchview"][4]["toppoint"]["topID"]'
}


3. 加入新function load_mat_auto()
def load_mat_auto(filename):
    try:
        watchentry = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
        return watchentry
    except NotImplementedError:
        watchentry = h5py.File(filename, 'r')
        return watchentry


4. 加入新function search_watchview_id(watchview, watchview_id, watchview_image_path). 這個function輸出另一個dictionary search_watchview_id跟watchview_id有一樣的大小. 如果在watchview_id裡面有'glasspoint 1':'["watchentry"]["watchview"][0]["glasspoint"]["etchID"]', 而且watchview["watchentry"]["watchview"][0]["glasspoint"]["etchID"] = "C84EEDCE31308831A8D4F8BD522A7507", 且watchview_image_path = “\ImagingLibWatch\images\matlab_images\”. 則search_watchview_id對應的就是'glasspoint 1':"\ImagingLibWatch\images\matlab_images\C84EEDCE31308831A8D4F8BD522A7507”


5. 如果RadioButton_folder click or 如果RadioButton_file click or 如果RadioButton_latest click, 則image_extract_folder, db_extract_folder跟db_extract_data為empty.


6. 如果RadioButton_folder click, 將"在listWidget_image加入這些db_extract_data[i][0]為項目, 而且CheckState為Checked."移到按下PushButton_folder_extraction, 而且image_extract_folder跟db_extract_data不為empty才觸發(而且listWidget_image裡的項目如果在image_extract_folder找不到subfolder, 則項目關閉Enabled). 同時觸發對於db_extract_data[i]的每個項目, 執行Watchentry = load_mat_auto(db_extract_data[i][1]), 以及search_watchview_id = search_watchview_id(watchview, watchview_id, db_extract_data[i][2]).


7. 如果RadioButton_file click, 將"在listWidget_image加入這些db_extract_data[i][0]為項目, 而且CheckState為Checked."移到按下PushButton_file_extraction, 而且image_extract_folder跟db_extract_data不為empty才觸發(而且listWidget_image裡的項目如果在image_extract_folder找不到subfolder, 則項目關閉Enabled). 同時觸發對於db_extract_data[i]的每個項目, 執行Watchentry = load_mat_auto(db_extract_data[i][1]), 以及search_watchview_id = search_watchview_id(watchview, watchview_id, db_extract_data[i][2]).


8. 如果RadioButton_latest click, 將"在listWidget_image加入這些db_extract_data[i][0]為項目, 而且CheckState為Checked."移到按下PushButton_latest_extraction, 而且image_extract_folder跟db_extract_data不為empty才觸發(而且listWidget_image裡的項目如果在image_extract_folder找不到subfolder, 則項目關閉Enabled). 同時觸發對於db_extract_data[i]的每個項目, 執行Watchentry = load_mat_auto(db_extract_data[i][1]), 以及search_watchview_id = search_watchview_id(watchview, watchview_id, db_extract_data[i][2]).


9. 按下pushButton_image_selectall時, listWidget_image裡的所有項目都click. 按下pushButton_image_deselectall時, listWidget_image裡的所有項目都unclick. 按下pushButton_view_selectall時, listWidget_view裡的所有項目都click. 按下pushButton_view_deselectall時, listWidget_view裡的所有項目都unclick. 


10. 最後當按下pushButton_view_image按鈕時, 根據listWidget_image跟listWidget_view enable打開以及click, 則跳出視窗顯示search_watchview_id對應的image path的image. 譬如這就是image path= \ImagingLibWatch\images\matlab_images\C84EEDCE31308831A8D4F8BD522A7507.png, 跳出視窗顯示這個image, 跳出視窗可以調整大小讓image放大縮小.

10.最後code裡面所有的comments都是英文.


```


```
我的python code會load mat file(闢如A03FF1C13FCE053512F5686451B5CC23.mat)裏面的watchentry變數, 並讀取這個watchentry變數的很多value. 譬如在matlab裡面的watchentry.watchview(1).glasspoint.etchID = 'C84EEDCE31308831A8D4F8BD522A7507'. 請幫我檢查這個python function: def search_watchview_id(self, mat_data, watchview_id, watchview_image_path) 我執行的時候mat_data就等於watchentry變數, watchview_id是之前我在ImagingWatch_config.yaml寫下的
watchview_id:
  glasspoint 1: '["watchentry"]["watchview"][0]["glasspoint"]["etchID"]'
  toppoint 1: '["watchentry"]["watchview"][0]["toppoint"]["topID"]'
  toppoint 2: '["watchentry"]["watchview"][1]["toppoint"]["topID"]'
  toppoint 3: '["watchentry"]["watchview"][2]["toppoint"]["topID"]'
  toppoint 4: '["watchentry"]["watchview"][3]["toppoint"]["topID"]'
  toppoint 5: '["watchentry"]["watchview"][4]["toppoint"]["topID"]'

watchview_image_path則是'D:/Provenance Project/ImagingLibWatch/images/matlab_images\\A03FF1C13FCE053512F5686451B5CC23'這個folder裡面有很多png file. 譬如C84EEDCE31308831A8D4F8BD522A7507.png. 

所以在for key, eval_str in watchview_id.items()迴圈裏面result_dict[key]應該可以正確指到那張png file的path 'D:/Provenance Project/ImagingLibWatch/images/matlab_images\\A03FF1C13FCE053512F5686451B5CC23\\C84EEDCE31308831A8D4F8BD522A7507.png' 但似乎會有Error 請幫我修改

```


```

目前Error已經解決可以正確定位到image並跳出視窗展示image. 想做個小修正, 就是跳出image視窗後可以縮放, 可能用滑鼠滾論縮放.

還有另一個新增功能是按下pushButton_view_info之後要跳出視窗顯示很多值(這些值也是來自於watchentry變數), 譬如
X: 80
Y: 140等...

以function search_watchview_id(self, mat_data, watchview_id, watchview_image_path)來說在for key, eval_str in watchview_id.items()迴圈裡, key='glasspoint 1', eval_str='["watchentry"]["watchview"][0]["glasspoint"]["etchID"]'. 那要顯示的value就是["watchentry"]["watchview"][0]["glasspoint"]下面所有的值譬如["watchentry"]["watchview"][0]["glasspoint"]["X"] = 80, ["watchentry"]["watchview"][0]["glasspoint"][Y] = 140 等等. 

```

```
目前可以用視窗縮放, 但效果很不理想請修正. 目前用滾輪控制是只擷取image從左上開始擷取小區域並沒有image放大縮小. 我想要的效果是如果滑鼠滾輪往上滾, 在滑鼠指的地方為中心放大. 譬如本來image是100x100, 顯示的區域是x的40到60, y的40到60. 滑鼠滾輪往上滾原來image變成新的放大image 200x200, 但顯示的區域仍然新的放大image的x的40到60, y的40到60區域. 但如果是縮小, 譬如顯示的區域是x的40到60, y的40到60, 但縮小後的image不到20pixel x 20pixel, 則視窗也進行縮小. 請提供新的ImageViewer
```


#### Project 11 - Image Processing Pipeline

Create and deliver python code that A) extracts relevant images (in conjunction with task 1), B) performs an analysis or preliminary analysis of the image, C) selects one or more other analysis methods to perform on the on image, D) Stores and/or displays the outputs, and E) continues to the next image(s) until complete. 影像處理流程：建立並提交 Python 程式碼，程式碼能夠：A) 擷取相關影像（與任務 1 結合使用）；B) 對影像進行分析或初步分析；C) 選擇一種或多種其他分析方法對目前影像執行；D) 儲存和/或顯示輸出結果；E) 繼續處理下一張或多張影像，直至完成。

a.      This task should be executable from a Matlab command, or if deemed not possible, by a simple button push 此任務應可透過 Matlab 指令執行；如果無法透過 Matlab 指令執行，則可透過簡單的按鈕操作執行。

b.     Output images and data, to be specified by Client, should be put into a human readable output file (e.g. a word doc, PDF, image, or other format) 輸出影像和資料（由客戶指定）應儲存為易於閱讀的輸出檔案（例如 Word 文件、PDF、影像或其他格式）。


#### Project 12 - Database of results

Create and deliver python code that runs all the analysis methods in Project Assignment 1 and 2, and aggregates the results. The results should then be split by watch reference, which should be extracted from the Matlab structure. 結果資料庫：建立並提交 Python 程式碼，該程式碼執行專案作業 1 和 2 中的所有分析方法，並彙總結果。然後，應按監視引用拆分結果，監視引用應從 Matlab 結構中提取。

a.      The mean, median, std, 25%, 75%, min, and max should be found for each metric (for numeric metrics). 應計算每個指標（對於數值指標）的平均值、中位數、標準差、25% 偏差、75% 偏差、最小值和最大值。 i. 應依時期、型號及參考編號進行總結分析。This should be done in aggregate, per era, per model, and per reference.

b.     The values should be plotted directly and as histograms (for numeric metrics). This should be done in aggregate, per era, per model, and per reference. 數值指標應直接繪製成圖表，並以直方圖的形式呈現。應依時期、型號及參考編號進行總結分析。

c.      For metrics that are lines, the data should be plotted as overlapping values. This should be done in aggregate, per era, per model, and per reference. 對於折線圖指標，資料應以重疊值的形式繪製。應依時期、型號及參考編號進行總結分析。

![[Pasted image 20251111074722.png]]

![[Pasted image 20251111074813.png]]

![[Pasted image 20251111074836.png]]


#### Project 13 - Crystal Crown

Create and deliver python code that processes images of the crystal (aka sapphire or glass) of watches and isolates and analyzes the etching. This should be done in a manner that is robust to dust, uneven lighting, highlights and shadows, specular reflections, background images, small occlusions, and all variations of background and lume colors, textures, and reflectivity. 表鏡：編寫並提交 Python 程式碼，用於處理腕錶錶鏡（又稱藍寶石或玻璃）的圖像，並提取和分析蝕刻痕跡。程式碼應能有效應對灰塵、光照不均、高光和陰影、鏡面反射、背景影像、輕微遮蔽以及背景和夜光顏色、紋理和反射率的各種變化。

a.      You will be provided with Matlab code as a starting point 我們將提供 Matlab 程式碼作為起點。

b.     The location of the dots and any that are out of place, missing, or very dim should be recorded. 應記錄點的位置，以及任何位置錯誤、缺失或非常暗淡的點。

c.      Method should be made more robust to bright spots than what is currently in the Matlab code. 該方法應該比目前 Matlab 程式碼中的方法更能有效應對亮點。

![[Pasted image 20251113135921.png]]


#### Project 14 - Links

Create and deliver python code that processes images of the bracelet (aka strap) of watches and isolates the centrally shown link. This should be done in a manner that is robust to dust, uneven lighting, highlights and shadows, specular reflections, uneven feature edge shape, small occlusions, and all variations of background and feature colors, textures, and reflectivity. 錶鍊：編寫並提交 Python 程式碼，用於處理手錶錶帶（又稱錶鍊）的圖像，並提取圖中中心顯示的錶鍊節。此過程應能有效應對灰塵、光照不均、高光和陰影、鏡面反射、不規則邊緣形狀、輕微遮擋以及背景和特徵顏色、紋理和反射率的各種變化。

a.      Determine the area 確定面積

b.     Determine the length (absolute and along curvature) 決定長度（絕對長度和沿曲率方向的長度）

c.      Determine thickness (absolute and along curvature) 確定厚度（絕對厚度和沿曲率方向的厚度）

d.     If a pin is present, determine the radius and center of it 如果存在銷釘，則確定其半徑和中心

![[Pasted image 20251114075629.png]]

#### Project 15 - Movement Isolation

Create and deliver python code that processes images of the microscopic number feature of the movement and provides 機芯隔離：編寫並提交 Python 程式碼，用於處理機芯微觀數位特徵的圖像，並提供以下資訊：

a.      A mask of the textured region 紋理區域的遮罩

b.     A description of the type of texturing (e.g. bumpy or striated) 紋理類型的描述（例如，凹凸不平或條紋狀）

![[Pasted image 20251111074949.png]]

![[Pasted image 20251111075008.png]]
![[Pasted image 20251111075037.png]]
#### Project 16 - Bracelet Pin

Create and deliver python code that processes images of the microscopic pin of the bracelet and provides 錶帶銷釘：編寫並提交 Python 程式碼，用於處理錶帶微觀銷釘的圖像，並提供以下資訊：

a.      A determination if the pin is A) Fully visible and in focus B) Partially visible and in focus, or C) Out of focus and/or not visible 判斷銷釘的狀態：A) 完全可見且清晰；B) 部分可見且清晰；C) 失焦和/或不可見

b.     An analysis for case A (fully visible in focus) and determines 分析情況 A（完全可見且清晰），並確定：
b1. The radius of the pin 銷釘半徑
b2. The radius of the hole 孔半徑
b3. The concentricity of the pin to the hole (e.g. offset of centers) 銷釘與孔的同心度（例如，中心偏移）
b4. The variation in radius of the pin 銷釘半徑的變化
b5. The variation in radius of the hole 孔半徑的變化
b6. An analysis of case B for the same as for case A. 分析情況 B，與情況 A 相同。

![[Pasted image 20251111075202.png]]


#### Project 17 - Reading Case Side

Create and deliver python code that processes images of the sides of the case (aka body) of watches. This should be done in a manner that is robust to dust, uneven lighting, highlights and shadows, specular reflections, uneven feature edge shape, small occlusions, and all variations of background and feature colors, textures, and reflectivity 讀取錶殼側面：編寫並提交 Python 程式碼，用於處理手錶錶殼（又稱表身）側面的圖像。這項工作應能有效應對灰塵、光照不均、高光和陰影、鏡面反射、特徵邊緣形狀不規則、輕微遮擋以及背景和特徵顏色、紋理和反射率的各種變化。

a.      Extract the text, especially focused on the reference and where applicable the serial number. 提取文本，特別關注參考資訊以及適用的序號。

b.     Determine the spacing of the words “Stainless”, “Steel”, “Orig.”, “Rolex”, and “Design” (where applicable). 決定「Stainless」、「Steel」、「Orig.」、「Rolex」和「Design」（如適用）等字詞的間距。

c.      Determine the height of the text 確定文字高度。

d.     Determine if the text is engraved or laser etched. 確定文字是雕刻還是雷射蝕刻。

e.      Create metrics that can be repeatably seen in the engravings, even with wear and challenging lighting conditions. 創建即使在磨損和光照條件不佳的情況下，也能在雕刻中重複觀察到的度量標準。

![[Pasted image 20251111075230.png]]


參考text_run.py 跟text_config.yaml
```
11/11 2226
以下是一段python script code跟他的yaml config file. 可以從錶面detect OCR及一系列分析. 想按照以下python script code跟他的yaml config file新寫一個python script code(ocr_compare_run.py)跟他的yaml config file(ocr_compare_config.yaml)目的是比較不同的OCR library. 將OCR library新增加PaddleOCR, DocTR, Surya, mmOCR, 希望將不同library儘可能模組化方便未來擴充新的OCR library. 希望就像下面python script code進行text level detection跟同樣的輸出. 如果OCR library也有提供character level detection也進行, 並同樣的輸出. 包括figure跟txt file儲存到output folder. 在yaml config file有選項可以針對每個OCR library選擇要不要執行. 最後並附上environment file create python environment.

請根據原始code儘量少變動提供新的python script code跟yaml config file, code comments全部用英文.
```



```
我在surya github看到下列到OCR (text recognition)使用方式, 請修改surya相關的import及python script code使用方法: 

OCR (text recognition)
from PIL import Image
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor

image = Image.open(IMAGE_PATH)
foundation_predictor = FoundationPredictor()
recognition_predictor = RecognitionPredictor(foundation_predictor)
detection_predictor = DetectionPredictor()

predictions = recognition_predictor([image], det_predictor=detection_predictor)


(Text line detection)
from PIL import Image
from surya.detection import DetectionPredictor

image = Image.open(IMAGE_PATH)
det_predictor = DetectionPredictor()

# predictions is a list of dicts, one per image
predictions = det_predictor([image])
```


```
我查到DocTR, Surya, mmOCR 都有character level OCR. 所以我想比較的是easyOCR(text level)+pytesseract(character level), PaddleOCR(text level)+pytesseract(character level), DocTR (text level and character level), surya (text level and character level), mmOCR (text level and character level). 提供character level box, text level box以及其他資訊譬如text, character, confidence等等. 請中文詳細解釋如何用「主控腳本 (Controller Script)」使用 Python 的 subprocess 模組來呼叫**其他環境的 Python 執行檔來做到將原有的（EasyOCR + Tesseract）分析流程，擴展為一個模組化的比較框架，以評估多個OCR library（PaddleOCR, DocTR, Surya, mmOCR）個別的text level OCR跟character level OCR. 請提供新的code. 以下是我查到的:

(DocTR library) github: https://github.com/mindee/doctr

DocTR支援 character level OCR：DocTR 採兩階段架構，先偵測文字區域再將每一區域經過 recognition 模型辨識所有字元，結果可提供到字元層級

  

from doctr.models import ocr_predictor

model = ocr_predictor(pretrained=True)

result = model(["your_image.jpg"])

# result 內包含偵測到的每個文字區域，以及區域內的文字（可拆分為每個字元）

  

如要針對單個字元進行細緻辨識，可將圖片切割為單字圖，傳入 recognition_predictor

from doctr.io import DocumentFile

from doctr.models import recognition_predictor

doc = DocumentFile.from_images("char_image.png")

model = recognition_predictor(pretrained=True)

result = model(doc)

print(result)

  

  

  

(surya library) github: https://github.com/datalab-to/surya

The results.json file will contain a json dictionary where the keys are the input filenames without extensions. Each value will be a list of dictionaries, one per page of the input document. Each page dictionary contains:

chars - the individual characters in the line

text - the text of the character

bbox - the character bbox (same format as line bbox)

polygon - the character polygon (same format as line polygon)

confidence - the confidence of the model in the detected character (0-1)

bbox_valid - if the character is a special token or math, the bbox may not be valid

  

  

(mmOCR library) github: https://github.com/open-mmlab/mmocr

支援 character level OCR：mmOCR 支援多種模型（如 CRNN、SAR），這些模型底層字元編碼，可自訂導出每個 word 的 character

from mmocr.apis import MMOCR

ocr = MMOCR(det='DB_r18', rec='CRNN')

result = ocr.readtext('your_image.jpg')

# result['text'] 會回傳識別到的字串，可拆為每個字元
```


```
在這裡我想做一下變動. 將controller.py裡的功能跟設定, 將mmocr OCR library部分獨立成類似run_mmocr_subprocess.py可獨立運行的python file請提供code. 可以讀取folder裡面的images, 然後用那個OCR library進行text level OCR跟character level OCR, 如果這個OCR library輸出character level OCR結果是empty或根本沒這功能, 則用Tesseract library從text的box內部去辨識到character(要標示用哪個library)進行後續character level分析. 如果這個OCR library有character level OCR結果, 就同時計算並輸出這個OCR library跟Tesseract library的兩個內容並進行後續character level分析.

輸出text bbox(Individual Detections), text跟confidence score. 如果text(Individual Detections)是在同個水平位置用group_words_into_lines變成Grouped Lines, 也輸出box coordinates跟text. 而在Character-Level OCR則輸出letter, character的bbox, height, width, Skeleton Endpoints, junctions, Thickness and totla length, 並計算Hu Moments (log), 並計算x direction projection(每隔PROFILE_SPACING=5計錄一次), 並計算y direction projection(每隔PROFILE_SPACING=5計錄一次), 並計算Skeleton thickness(每隔PROFILE_SPACING=5計錄一次). Figure則輸出原圖跟text-level bbox結果的overlay figure, 原圖跟character-level bbox結果的overlay figure, 原圖跟text-level bbox + character-level bbox 結果的overlay figure, 對每個character bbox用otsu方法得到的所有letter 的segmentation masks的overlay figure, 圖跟skeleton的overlay figure. 

figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分使用now = datetime.datetime.now(ZoneInfo("America/New_York")). 然後再把這個subfolder名字後面加上"_mmocr_result(v1)". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文.


請把所有跟參數設定都放在獨立的mmocr_config.yaml 管理, 請提供新的code包括run_mmocr_pipeline.py, mmocr_config.yaml還有設定python environment的files(conda, pip)



這是是我之前確認成功建立且mmocr可以用的environemt file. 請參考這個提供conda , pip 建立python environment的files: conda create -n ocr_mmocr_env python=3.8 -y

conda activate ocr_mmocr_env
conda install pytorch==1.10.0 torchvision cudatoolkit=11.3 -c pytorch -y

pip install -U openmim
mim install "mmengine>=0.7.1,<1.1.0"
mim install "mmcv>=2.0.0rc4,<2.1.0"
mim install "mmdet>=3.0.0rc5,<3.2.0"
pip install "mmocr==1.0.1"
pip install opencv-python numpy pyyaml pillow pytesseract
```


```
在這裡我想做一下變動. 將controller.py裡的功能跟設定, 將paddle OCR library部分獨立成類似run_paddle_subprocess.py可獨立運行的python file請提供code. 可以讀取folder裡面的images, 然後用那個OCR library進行text level OCR跟character level OCR, 如果這個OCR library輸出character level OCR結果是empty或根本沒這功能, 則用Tesseract library從text的box內部去辨識到character(要標示用哪個library)進行後續character level分析. 如果這個OCR library有character level OCR結果, 就同時計算並輸出這個OCR library跟Tesseract library的兩個內容並進行後續character level分析.

輸出text bbox(Individual Detections), text跟confidence score. 如果text(Individual Detections)是在同個水平位置用group_words_into_lines變成Grouped Lines, 也輸出box coordinates跟text. 而在Character-Level OCR則輸出letter, character的bbox, height, width, Skeleton Endpoints, junctions, Thickness and totla length, 並計算Hu Moments (log), 並計算x direction projection(每隔PROFILE_SPACING=5計錄一次), 並計算y direction projection(每隔PROFILE_SPACING=5計錄一次), 並計算Skeleton thickness(每隔PROFILE_SPACING=5計錄一次). Figure則輸出原圖跟text-level bbox結果的overlay figure, 原圖跟character-level bbox結果的overlay figure, 原圖跟text-level bbox + character-level bbox 結果的overlay figure, 對每個character bbox用otsu方法得到的所有letter 的segmentation masks的overlay figure, 圖跟skeleton的overlay figure. 

figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分使用now = datetime.datetime.now(ZoneInfo("America/New_York")). 然後再把這個subfolder名字後面加上"_paddle_result(v1)". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文.
```


```
在這裡我想做一下變動. 將controller.py裡的功能跟設定, 將Surya OCR library部分獨立成類似run_paddle_subprocess.py可獨立運行的python file請提供code. 可以讀取folder裡面的images, 然後用那個OCR library進行text level OCR跟character level OCR, 如果這個OCR library輸出character level OCR結果是empty或根本沒這功能, 則用Tesseract library從text的box內部去辨識到character(要標示用哪個library)進行後續character level分析. 如果這個OCR library有character level OCR結果, 就同時計算並輸出這個OCR library跟Tesseract library的兩個內容並進行後續character level分析. 請把所有跟參數設定都放在獨立的paddle_config.yaml 管理, 也請提供建立python environment code.

輸出text bbox(Individual Detections), text跟confidence score. 如果text(Individual Detections)是在同個水平位置用group_words_into_lines變成Grouped Lines, 也輸出box coordinates跟text. 而在Character-Level OCR則輸出letter, character的bbox, height, width, Skeleton Endpoints, junctions, Thickness and totla length, 並計算Hu Moments (log), 並計算x direction projection(每隔PROFILE_SPACING=5計錄一次), 並計算y direction projection(每隔PROFILE_SPACING=5計錄一次), 並計算Skeleton thickness(每隔PROFILE_SPACING=5計錄一次). Figure則輸出原圖跟text-level bbox結果的overlay figure, 原圖跟character-level bbox結果的overlay figure, 原圖跟text-level bbox + character-level bbox 結果的overlay figure, 對每個character bbox用otsu方法得到的所有letter 的segmentation masks的overlay figure, 圖跟skeleton的overlay figure. 

figure file存在OUTPUT_PATH下面的新開一個sub folder以月分日期跟時間命名譬如"10301639"代表10月30日16點39分使用now = datetime.datetime.now(ZoneInfo("America/New_York")). 然後再把這個subfolder名字後面加上"_Surya_result(v1)". 把figuers儲存存在這個新subfolder. 在之前colab已經定義OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output", "Lume images"). 如果有輸出txt file則存在同個subfolder. 把code裡面會影響結果的parameter集中放在import下方方便設定, code儘量function化, code中的comments也都用英文.
```



```
mmocr_env
---------------------------
conda create -n ocr_mmocr_env python=3.8 -y
conda activate ocr_mmocr_env
conda install pytorch==1.10.0 torchvision cudatoolkit=11.3 -c pytorch -y
pip install -U pip
pip install -U openmim
mim install "mmengine>=0.7.1,<1.1.0"
mim install "mmcv>=2.0.0rc4,<2.1.0"
mim install "mmdet>=3.0.0rc5,<3.2.0"
pip install "mmocr==1.0.1"
pip install opencv-python numpy pyyaml pillow pytesseract scipy scikit-image tqdm tomli platformdirs
python -m pip install backports.zoneinfo




paddle_env
---------------------------
conda create -n ocr_paddle_env python=3.10 -y
conda activate ocr_paddle_env
python -m pip install -U pip
python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install "paddleocr==3.2.0"
python -m pip install opencv-python numpy pyyaml tqdm scikit-image scipy Pillow
python -m pip install "torch==2.2.2" --index-url https://download.pytorch.org/whl/cpu
python -m pip install pytesseract




Surya_env
---------------------------
conda create -n ocr_surya_env python=3.10 -y
conda activate ocr_Surya_env
pip install surya-ocr
python -m pip install scikit-image scipy pytesseract



doctr_env
---------------------------
conda create -n ocr_doctr_env python=3.10 -y
conda activate ocr_doctr_env   
pip install python-doctr
python -m pip install scikit-image scipy pytesseract





```



```
這裡是run_doctr_subprocess.py跟設定檔doctr_config.yaml. 這裡IMAGE_SCALE控制image進行縮放(譬如IMAGE_SCALE=0.5則200x200pixel變成100x100pixel). 請按照下列需求更改doctr_config.yaml跟run_doctr_subprocess.py

1. 這裡要修改縮放的方式. IMAGE_SCALE=0.5則200x200pixel image縮小成100x100pixel的區域在200x200pixel的左上角, 其他區域則補0. 而當OCR library detect到text and letter得到bbox之後, 將bbox恢復原來尺寸到他本來要對應的地方. 並以原始尺寸輸出figures跟後續的結果. 

2. 在preprocess_image時cv2.imread載入image之後, 進行局部對比增強（CLAHE）, 適度 gamma 調整(預設0.40), Unsharp (預設σ=1.5, α=5.0), Second CLAHE, IMADJUST, IMBINARIZE等步驟. 每個步驟都有parameter可以選擇true or false, 設定都放在yaml設定檔.

請按照這需求更改doctr_config.yaml跟run_doctr_subprocess.py
```

```11190930 gemini

以下是run_doctr_subprocess.py跟設定yaml file 會從image進行前處理並用OCR library讀取image中的文字並輸出含有文字的bbox的一系列figures跟txt files. 現在要在儘量不改變原本code的前提下基於需求做稍微修改並提供新的完整code:

需求: 
在apply_advanced_preprocessing() function, 在# --- 3. Scaling (Canvas Method) ---部分目前是用Canvas Method方式, 現在要加入另一種方式就是單純的resize method. 譬如IMAGE_SCALE = 0.5, 則原本200 x 200pixel的image就變成100 x 100pixel image. 有選項在doctr_config.yaml 可以選擇用哪種方式. 

run_doctr_subprocess.py輸出含有text or character bbox的figures時, 尤其在有resize跟rotate時bbox的位置明顯不match. 請特別檢查code這方面並進行修正.

(doctr_config.yaml file)

```

目前doctr_config.yaml可以提供最好的效果

PATHS:
  DATA_TEXT_PATH: "./images/text_images3"
  OUTPUT_PATH: "./output/doctr_standalone_results"

SYSTEM:
  TESSERACT_CMD_PATH: 'D:\Program Files\Tesseract-OCR\tesseract.exe'

OUTPUT:
  SUFFIX: "_DocTR_result(v6)(images3_r0_CANVAS_scale0.3)"
  TIMEZONE: "America/New_York"

PREPROCESSING_PARAMS:
  # 1. Scaling
  # Options: 'CANVAS' (Keep original canvas size, shrink image to corner) 
  #          'RESIZE' (Physically resize the image dimensions)
  SCALING_MODE: 'CANVAS'
  IMAGE_SCALE: 0.3
  
  # 2. Rotation (0, 90, 180, 270)
  ROTATION_ANGLE: 0

  # 3. Preprocessing Steps (True/False switches & parameters)
  ENABLE_CLAHE: true
  CLAHE_CLIP_LIMIT: 0.04
  CLAHE_TILE_GRID_SIZE: [20, 20]

  ENABLE_GAMMA: true
  GAMMA_VALUE: 1.2

  ENABLE_UNSHARP: true
  UNSHARP_SIGMA: 1.5
  UNSHARP_ALPHA: 5.0

  ENABLE_CLAHE_2: false
  CLAHE_2_CLIP: 0.04
  CLAHE_2_GRID: [15, 15]

  ENABLE_IMADJUST: false
  ENABLE_IMBINARIZE: false

# --- Visualization Parameters ---
VISUALIZATION_PARAMS:
  OVERLAY_OPACITY: 0.4
  TEXT_BOX_COLOR: [0, 255, 0]
  TEXT_BOX_THICKNESS: 3
  CHAR_BOX_TESS_COLOR: [255, 0, 0]
  CHAR_BOX_TESS_THICKNESS: 3
  CHAR_BOX_DOCTR_COLOR: [255, 255, 0]
  CHAR_BOX_DOCTR_THICKNESS: 3

DOCTR_PARAMS:
  DEVICE: "cuda"
  MIN_CONFIDENCE: 0.1
  PRETRAINED: true

GROUPING_PARAMS:
  LINE_GROUPING_Y_THRESHOLD: 0.5

POLARITY_PARAMS:
  GAUSSIAN_BLUR_KERNEL: [5, 5]
  NUM_SAMPLES: 5

TESSERACT_CHAR_PARAMS:
  TESSERACT_CHAR_CONFIG: "--oem 1 --psm 8"
  TESSERACT_PADDING: 10

FILTERING_PARAMS:
  ENABLE_TILT_FILTER: true
  MAX_TEXT_TILT_DEGREES: 30.0
  TEXT_HEIGHT_THRESHOLD: 0.0
  ENABLE_LARGEST_COMPONENT_FILTER: true
  CHAR_EXCEPTIONS_TO_FILTERING: ['i', 'j', '!', ':', ';', '=']

ANALYSIS_PARAMS:
  GAUSSIAN_BLUR_KERNEL: [3, 3]
  MORPH_OPEN_KERNEL_SIZE: [3, 3]
  UPSCALING_FACTOR: 2.0
  BINARIZATION_MODE: 'AUTO_OTSU'
  
  ADAPTIVE:
    BLOCK_SIZE: 19
    C_CONSTANT: 9
  GLOBAL_OTSU:
    OTSU_RATIO: 1.0
  AUTO_OTSU:
    RATIO_SEARCH_LIST: [0.8, 1.0, 0.6, 1.2]
  
  N_FOURIER_DESCRIPTORS: 16
  PROFILE_SPACING: 5


``` 11191036 gemini
以下是python script run_mmocr_subprocess.py跟設定file mmocr_config.yaml會從image進行前處理並用OCR library讀取image中的文字並輸出含有文字的bbox的一系列figures跟txt files. 現在要在儘量不改變原本code的前提下基於需求做稍微修改並提供新的完整code:

需求: 
1. 在main function下的detect_words_with_mmocr之前是執行apply_preprocessing()進行前處理. 現在改成參考下面的apply_advanced_preprocessing()對Image進行一系列前處理並在yaml進行設定. 這系列前處理對image的OCR效果影響很大所以儘量照apply_advanced_preprocessing處理
def apply_advanced_preprocessing(img, config, MAIN_RUN_OUTPUT_PATH):
    """
    Applies Rotation -> Enhancement -> Scaling (Canvas or Resize).
    Returns: processed_img, meta_data (essential for coordinate mapping)
    """
    params = config.get('PREPROCESSING_PARAMS', {})
    
    # Store original shape
    h_orig, w_orig = img.shape[:2]
    meta = {'h_orig': h_orig, 'w_orig': w_orig}
    
    # --- 1. Rotation ---
    angle = params.get('ROTATION_ANGLE', 0)
    rotation_code = None
    if angle == 90: rotation_code = cv2.ROTATE_90_CLOCKWISE
    elif angle == 180: rotation_code = cv2.ROTATE_180
    elif angle == 270: rotation_code = cv2.ROTATE_90_COUNTERCLOCKWISE
    
    if rotation_code is not None:
        img = cv2.rotate(img, rotation_code)
        logging.info(f"  [Preproc] Applied rotation: {angle} deg")
        cv2.imwrite(os.path.join(MAIN_RUN_OUTPUT_PATH, "0_Rotation.png"), img)
    
    # Capture dimensions AFTER rotation but BEFORE scaling (Crucial for un-rotating)
    h_rotated, w_rotated = img.shape[:2]
    meta['rotation'] = angle
    meta['h_rotated'] = h_rotated
    meta['w_rotated'] = w_rotated

    # --- 2. Enhancements ---
    # CLAHE 1
    if params.get('ENABLE_CLAHE', False):
        clip = params.get('CLAHE_CLIP_LIMIT', 2.0)
        grid = tuple(params.get('CLAHE_TILE_GRID_SIZE', [20, 20]))
        img = apply_clahe(img, clip, grid)
        cv2.imwrite(os.path.join(MAIN_RUN_OUTPUT_PATH, "1_clahe1.png"), img)
        
    # Gamma
    if params.get('ENABLE_GAMMA', False): 
        gamma = params.get('GAMMA_VALUE', 1.2)
        img = apply_gamma_correction(img, gamma)
        cv2.imwrite(os.path.join(MAIN_RUN_OUTPUT_PATH, "2_gamma.png"), img)

    # Unsharp
    if params.get('ENABLE_UNSHARP', False): 
        sigma = params.get('UNSHARP_SIGMA', 1.5)
        alpha = params.get('UNSHARP_ALPHA', 5.0)
        img = apply_unsharp_mask(img, sigma, alpha)
        cv2.imwrite(os.path.join(MAIN_RUN_OUTPUT_PATH, "3_unsharp.png"), img)
        
    # CLAHE 2
    if params.get('ENABLE_CLAHE_2', False):
        clip = params.get('CLAHE_2_CLIP', 0.04)
        grid = tuple(params.get('CLAHE_2_GRID', [15, 15]))
        img = apply_clahe(img, clip, grid)
        cv2.imwrite(os.path.join(MAIN_RUN_OUTPUT_PATH, "4_clahe2.png"), img)
        
    # Imadjust & Imbinarize
    if params.get('ENABLE_IMADJUST', False):
        img = apply_imadjust(img)
        cv2.imwrite(os.path.join(MAIN_RUN_OUTPUT_PATH, "5_imadjust.png"), img)
    if params.get('ENABLE_IMBINARIZE', False):
        img = apply_imbinarize(img)
        cv2.imwrite(os.path.join(MAIN_RUN_OUTPUT_PATH, "6_imbinarize.png"), img)

    # --- 3. Scaling (Updated for SCALING_MODE) ---
    scaling_mode = params.get('SCALING_MODE', 'CANVAS').upper() # RESIZE or CANVAS
    scale = params.get('IMAGE_SCALE', 1.0)
    
    if scale != 1.0:
        w_curr = img.shape[1]
        h_curr = img.shape[0]
        w_scaled = int(w_curr * scale)
        h_scaled = int(h_curr * scale)
        
        # Step A: Resize the content
        resized_img = cv2.resize(img, (w_scaled, h_scaled), interpolation=cv2.INTER_AREA)
        
        if scaling_mode == 'RESIZE':
            # Mode 1: Pure Resize (Image becomes smaller)
            img = resized_img
            logging.info(f"  [Preproc] Mode: RESIZE. Scaled {scale}x to {w_scaled}x{h_scaled}.")
            
        else: # Default to CANVAS
            # Mode 2: Canvas (Image placed on original-sized canvas)
            # Create canvas of ROTATED size (so we don't lose cropping if rotated)
            # Actually, logic says: "canvas of ORIGINAL size". 
            # However, if rotated 90, dims swapped. Using h_rotated/w_rotated is safer for "Canvas" logic on rotated image.
            canvas = np.zeros((h_rotated, w_rotated, 3), dtype=np.uint8)
            
            # Place at top-left
            h_place = min(h_rotated, h_scaled)
            w_place = min(w_rotated, w_scaled)
            canvas[0:h_place, 0:w_place] = resized_img[0:h_place, 0:w_place]
            
            img = canvas
            logging.info(f"  [Preproc] Mode: CANVAS. Scaled content {scale}x on {w_rotated}x{h_rotated} canvas.")
    else:
        logging.info("  [Preproc] No scaling applied.")
        
    meta['scale'] = scale
    meta['scaling_mode'] = scaling_mode

    return img, meta


2. 
python script輸出含有text or character bbox的figures時, 尤其在有resize跟rotate時bbox的位置明顯不match. 請特別檢查code這方面並進行修正.



(mmocr_config.yaml)


(run_mmocr_subprocess.py)
```