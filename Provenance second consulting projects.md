


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






#### Project 11 - Image Processing Pipeline

Create and deliver python code that A) extracts relevant images (in conjunction with task 1), B) performs an analysis or preliminary analysis of the image, C) selects one or more other analysis methods to perform on the on image, D) Stores and/or displays the outputs, and E) continues to the next image(s) until complete. 影像處理流程：建立並提交 Python 程式碼，程式碼能夠：A) 擷取相關影像（與任務 1 結合使用）；B) 對影像進行分析或初步分析；C) 選擇一種或多種其他分析方法對目前影像執行；D) 儲存和/或顯示輸出結果；E) 繼續處理下一張或多張影像，直至完成。

a.      This task should be executable from a Matlab command, or if deemed not possible, by a simple button push 此任務應可透過 Matlab 指令執行；如果無法透過 Matlab 指令執行，則可透過簡單的按鈕操作執行。

b.     Output images and data, to be specified by Client, should be put into a human readable output file (e.g. a word doc, PDF, image, or other format) 輸出影像和資料（由客戶指定）應儲存為易於閱讀的輸出檔案（例如 Word 文件、PDF、影像或其他格式）。


#### Project 12 - Database of results

Create and deliver python code that runs all the analysis methods in Project Assignment 1 and 2, and aggregates the results. The results should then be split by watch reference, which should be extracted from the Matlab structure. 結果資料庫：建立並提交 Python 程式碼，該程式碼執行專案作業 1 和 2 中的所有分析方法，並彙總結果。然後，應按監視引用拆分結果，監視引用應從 Matlab 結構中提取。

a.      The mean, median, std, 25%, 75%, min, and max should be found for each metric (for numeric metrics). 應計算每個指標（對於數值指標）的平均值、中位數、標準差、25% 偏差、75% 偏差、最小值和最大值。 i. 應依時期、型號及參考編號進行總結分析。This should be done in aggregate, per era, per model, and per reference.

b.     The values should be plotted directly and as histograms (for numeric metrics). This should be done in aggregate, per era, per model, and per reference. 數值指標應直接繪製成圖表，並以直方圖的形式呈現。應依時期、型號及參考編號進行總結分析。

c.      For metrics that are lines, the data should be plotted as overlapping values. This should be done in aggregate, per era, per model, and per reference. 對於折線圖指標，資料應以重疊值的形式繪製。應依時期、型號及參考編號進行總結分析。


#### Project 13 - Crystal Crown

Create and deliver python code that processes images of the crystal (aka sapphire or glass) of watches and isolates and analyzes the etching. This should be done in a manner that is robust to dust, uneven lighting, highlights and shadows, specular reflections, background images, small occlusions, and all variations of background and lume colors, textures, and reflectivity. 表鏡：編寫並提交 Python 程式碼，用於處理腕錶錶鏡（又稱藍寶石或玻璃）的圖像，並提取和分析蝕刻痕跡。程式碼應能有效應對灰塵、光照不均、高光和陰影、鏡面反射、背景影像、輕微遮蔽以及背景和夜光顏色、紋理和反射率的各種變化。

a.      You will be provided with Matlab code as a starting point 我們將提供 Matlab 程式碼作為起點。

b.     The location of the dots and any that are out of place, missing, or very dim should be recorded. 應記錄點的位置，以及任何位置錯誤、缺失或非常暗淡的點。

c.      Method should be made more robust to bright spots than what is currently in the Matlab code. 該方法應該比目前 Matlab 程式碼中的方法更能有效應對亮點。


#### Project 14 - Links

Create and deliver python code that processes images of the bracelet (aka strap) of watches and isolates the centrally shown link. This should be done in a manner that is robust to dust, uneven lighting, highlights and shadows, specular reflections, uneven feature edge shape, small occlusions, and all variations of background and feature colors, textures, and reflectivity. 錶鍊：編寫並提交 Python 程式碼，用於處理手錶錶帶（又稱錶鍊）的圖像，並提取圖中中心顯示的錶鍊節。此過程應能有效應對灰塵、光照不均、高光和陰影、鏡面反射、不規則邊緣形狀、輕微遮擋以及背景和特徵顏色、紋理和反射率的各種變化。

a.      Determine the area 確定面積

b.     Determine the length (absolute and along curvature) 決定長度（絕對長度和沿曲率方向的長度）

c.      Determine thickness (absolute and along curvature) 確定厚度（絕對厚度和沿曲率方向的厚度）

d.     If a pin is present, determine the radius and center of it 如果存在銷釘，則確定其半徑和中心


#### Project 15 - Movement Isolation

Create and deliver python code that processes images of the microscopic number feature of the movement and provides 機芯隔離：編寫並提交 Python 程式碼，用於處理機芯微觀數位特徵的圖像，並提供以下資訊：

a.      A mask of the textured region 紋理區域的遮罩

b.     A description of the type of texturing (e.g. bumpy or striated) 紋理類型的描述（例如，凹凸不平或條紋狀）

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

#### Project 17 - Reading Case Side

Create and deliver python code that processes images of the sides of the case (aka body) of watches. This should be done in a manner that is robust to dust, uneven lighting, highlights and shadows, specular reflections, uneven feature edge shape, small occlusions, and all variations of background and feature colors, textures, and reflectivity 讀取錶殼側面：編寫並提交 Python 程式碼，用於處理手錶錶殼（又稱表身）側面的圖像。這項工作應能有效應對灰塵、光照不均、高光和陰影、鏡面反射、特徵邊緣形狀不規則、輕微遮擋以及背景和特徵顏色、紋理和反射率的各種變化。

a.      Extract the text, especially focused on the reference and where applicable the serial number. 提取文本，特別關注參考資訊以及適用的序號。

b.     Determine the spacing of the words “Stainless”, “Steel”, “Orig.”, “Rolex”, and “Design” (where applicable). 決定「Stainless」、「Steel」、「Orig.」、「Rolex」和「Design」（如適用）等字詞的間距。

c.      Determine the height of the text 確定文字高度。

d.     Determine if the text is engraved or laser etched. 確定文字是雕刻還是雷射蝕刻。

e.      Create metrics that can be repeatably seen in the engravings, even with wear and challenging lighting conditions. 創建即使在磨損和光照條件不佳的情況下，也能在雕刻中重複觀察到的度量標準。