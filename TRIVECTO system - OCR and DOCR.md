

|                                                  |     |
| ------------------------------------------------ | --- |
| [[#### Watch side image OCR and shape analysis]] |     |
|                                                  |     |
|                                                  |     |
|                                                  |     |
|                                                  |     |
|                                                  |     |


#### Watch side image OCR and shape analysis
```
我想要對這張watch side image上面的文字進行OCR analysis and shape analysis, 也就是要辨認上面的字, 也要得到segmentation mask這樣才能針對每個char的形狀進行分析. 目前在系統有ocr_algo.py, doctr_algo.py兩個OCR專用analysis. 其中ocr_algo.py目前很好的用在錶面上的文字分析, 而且有結合了SAM model跟ocr. doctr_algo.py之前就是針對watch side image對這張image進行很多前處理. 請先簡單比較ocr_algo.py, doctr_algo.py方法, 之後中文詳細解釋doctr_algo.py詳細流程, 並請根據image提出建議要如何修改doctr_algo.py讓他可進行OCR analysis and segmentation
```
我看完了 ocr_algo.py、doctr_algo.py 和兩份 config。這張 side image 上的文字大致是：

ORIG. ROLEX DESIGN  
16613 T

**簡單比較**  
ocr_algo.py 是比較新的「OCR + SAM segmentation」路線。它用 DocTR 找 word box，然後依照文字長度和字元寬度權重推估每個 char box，再用 SAM 對每個 char box 做像素級 mask，最後抽 skeleton、stroke thickness、projection 等 shape features。核心在 ocr_algo.py (line 49)、SAM segmentation 在 ocr_algo.py (line 227)。

doctr_algo.py 則是比較傳統、偏 side image preprocessing 的 pipeline。它先做旋轉、CLAHE、gamma、unsharp、縮放等前處理，再用 DocTR 找文字，再用 Tesseract 或 DocTR native char boxes 找 char，最後用 local Otsu 在 char polygon 裡切出 mask 並做 shape analysis。流程在 doctr_algo.py (line 808)，char segmentation 在 doctr_algo.py (line 445)，feature extraction 在 doctr_algo.py (line 581)。

**doctr_algo.py 詳細流程**

1. 載入圖片後跑 apply_advanced_preprocessing()：可旋轉、CLAHE、gamma、unsharp、第二次 CLAHE、imadjust、binarize、scale。現在 config 是 rotation_angle: 180、image_scale: 0.4、scaling_mode: CANVAS，在 doctr_config.yaml (line 11)。
    
2. 用 DocTR 對 preprocessed image 做 OCR detection/recognition，取得 word text、confidence、word box，以及如果模型輸出支援，取得 native char boxes。這在 doctr_algo.py (line 205)。
    
3. 因為 OCR 是在 preprocessed image 上做的，所以 map_detections_to_original() 會把座標從旋轉/縮放後的圖映射回原圖座標。
    
4. group_words_into_lines() 把 word detections 依 y center 分行。對這張圖理想上應該得到兩行：ORIG. ROLEX DESIGN 和 16613 T。
    
5. auto_detect_text_polarity() 判斷文字是亮字還是暗字。這裡目前只得到一個全圖/全批次的 TEXT_IS_LIGHT，但這張圖同一行有亮刻字、暗刻字和金屬反光混合，所以這一步很可能不穩。
    
6. 對每個 word，先跑 Tesseract image_to_boxes() 取 char boxes，再跑 DocTR native char boxes。如果成功，就對每個 char box 呼叫 run_feature_extraction()。
    
7. run_feature_extraction() 會在 char polygon 的 bounding crop 內做 Otsu threshold，得到 local char mask，再做 largest connected component filtering、morph open、skeletonize、Hu moments、Fourier descriptors、projection profiles、stroke width 等分析。
    
8. paste_masks_to_global() 把每個 char 的 cleaned mask 和 skeleton mask 貼回全圖，最後輸出 segmentation overlay、skeleton overlay、annotated boxes、JSON/PDF report。
    

**針對這張圖的修改建議**  
我會建議把 doctr_algo.py 改成 side engraving 專用模式，而不是只調幾個 threshold：

1. 先修 config：這張圖不應該預設旋轉 180 度，也不應縮到 0.4。改成 rotation_angle: 0、image_scale: 1.0 或 1.5，scaling_mode: RESIZE。目前縮小會直接傷害細字 mask。
    
2. 加入 text ROI detection：先定位中間 brushed metal plate 區域，只在刻字帶上 OCR。這可以避開上方齒圈、黑色背景和底部模糊區，DocTR/Tesseract 會乾淨很多。
    
3. 改成 per-word/per-char polarity，不要全圖單一 TEXT_IS_LIGHT。這張圖 ORIG. 偏亮，ROLEX 和 16613 偏暗，單一 polarity 會讓某些字 mask 反相錯誤。
    
4. 在 Otsu 前加入金屬背景 flattening：例如 grayscale 後用 large-kernel blur 或 morphological opening 估背景，再做 gray - background 或 black-hat/top-hat。這會壓掉垂直刷痕，讓字元筆畫比較突出。
    
5. 把 ocr_algo.py 的 SAM 字元 segmentation 整合進 doctr_algo.py 作為可選模式：DocTR/Tesseract 負責 OCR 和 char box，SAM 負責 mask refinement。對這種刻字和反光背景，SAM box-prompt 通常比純 Otsu 更穩。
    
6. 若不想馬上加 SAM，至少加入 hybrid mask：local Otsu + adaptive threshold + connected components + char-box center filtering，不要只取 largest component。像 R、O、G、6 這些字可能有內洞或斷裂，largest-only 可能吃掉有效筆畫。
    
7. 儲存真正的 mask，不只 overlay。現在 doctr_algo.py 輸出 segmentation overlay，但沒有明確輸出 raw global char mask。建議新增：  
    *_4_segmentation_mask.png  
    *_5_skeleton_mask.png  
    以及每個 char 的 local mask metadata 或 RLE/path。
    
8. app_overlay 目前是 UNet background mask，不是 OCR char segmentation overlay。若目標是 OCR/shape analysis，建議讓 app_display_image 指向 char segmentation overlay，或新增 ocr_segmentation_overlay 作為 UI 預設。
    

總結：doctr_algo.py 已經有 OCR + char shape analysis 的大部分骨架，但要讓它可靠處理這張 side engraving image，最重要是「ROI + 不縮小 + per-char polarity + 背景 flattening + SAM 或 hybrid mask」。最小安全改法是先調 config 和加入 raw mask 輸出；效果若仍不穩，再把 ocr_algo.py 的 SAM char-mask 路線移植進 doctr_algo.py。