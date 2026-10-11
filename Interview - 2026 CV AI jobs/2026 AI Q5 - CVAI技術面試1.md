

|                    |     |
| ------------------ | --- |
| [[#### CVAI技術面試1]] |     |
|                    |     |
|                    |     |

#### CVAI技術面試1
```
如果在技術面試上被問到下列問題要如何完整深入回答:
Laplacian 與 Tenengrad 如何用於 Autofocus？(以及解釋甚麼是Focus Measure、Gradient、Noise、搜尋策略)
HDR Merge 如何處理不同曝光影像？(以及解釋甚麼是Exposure、Alignment、Weighting、Ghosting)
如何校正 Camera Lens Distortion？(以及解釋甚麼是Intrinsic Matrix、Radial／Tangential Distortion)
Image Stitching 有哪幾個主要步驟？(以及解釋甚麼是Feature Matching、RANSAC、Homography、Blending)
如何衡量影像 Sharpness？(以及解釋甚麼是Laplacian Variance、MTF、頻率響應)
RGB、HSV、Lab 各適合什麼分析？(以及解釋甚麼是色彩表示、亮度分離、Color Distance)
為什麼更高解析度不一定代表更好的辨識率？(以及解釋甚麼是Optical Resolution、SNR、Blur、Data Quality)
```

# Senior Computer Vision Engineer 技術面試：7 大影像處理問題完整解析

這七題是美國 Senior／Staff Computer Vision Engineer、Imaging Engineer、Machine Vision Engineer，以及部分 Senior AI Engineer 技術面試中相當重要的基礎問題。

面試官通常不只是想知道你是否使用過 OpenCV，而是希望判斷你是否具備以下能力：

1. Mathematical Fundamentals： 理解影像演算法背後的數學與物理原理。
    
2. Algorithm Selection： 知道不同方法的優勢、限制及適用條件。
    
3. System Design： 能把多種演算法整合成完整的 Imaging Pipeline。
    
4. Failure Analysis： 能診斷 Noise、Motion Blur、Distortion、Misalignment 等問題。
    
5. Production Engineering： 能制定實驗方法、評估指標及驗收標準，確保系統可重複、可靠地運作。
    

我會針對每題提供：

- 面試口述答案： 面試官問這題時，可以怎樣回答。
    
- 深入技術原理： 從基本概念、數學公式到演算法流程。
    
- 具體實例： 以精密相機、手錶檢測或微距影像為案例。
    
- Senior／Staff 層級延伸： 如何處理實際產品的 Performance、Robustness 與 Failure Modes。
    

# Q1. Laplacian 與 Tenengrad 如何用於 Autofocus？

## 1.1 面試時的完整核心回答

> Autofocus 的基本概念，是在不同的 Lens Position 或 Camera Position 擷取影像，計算每張影像的 Focus Measure，並搜尋使影像清晰度最大的對焦位置。
> 
> Laplacian 與 Tenengrad 都是基於影像空間變化的 Passive Autofocus 方法。
> 
> Laplacian 使用影像亮度的二階空間導數，評估高頻細節的強度，常用 Laplacian Variance 作為 Focus Measure。
> 
> Tenengrad 則使用 Sobel Operator 計算水平與垂直方向的一階 Gradient，並將 Gradient Magnitude 的平方加總或平均，衡量影像邊緣的強度。
> 
> Laplacian 計算簡單，對細節變化敏感，但也容易受到 Noise 影響。Tenengrad 通常對雜訊相對穩健一些，但其結果仍會受到影像內容、光照、對比及 ROI 的影響。
> 
> 在 Production System 中，我會將 Focus Measure 與 ROI Selection、Noise Reduction、Coarse-to-Fine Search、Motor Backlash Compensation 以及 Focus Confidence Validation 結合，而不是單純取分數最高的一張影像。

這段回答約需 1–2 分鐘，已經涵蓋演算法原理、比較與實際系統設計。

## 1.2 什麼是 Focus Measure？

Focus Measure（對焦評估函數） 是一個數值指標，用來估計影像有多清晰。

理想情況下：

- Defocused Image → 邊緣模糊，高頻細節減少。
    
- Focused Image → 邊緣清楚，局部亮度變化較劇烈。
    
- 過了最佳對焦點 → 影像再次模糊。
    

假設 Camera 沿 Z 軸移動：

|Camera Z Position|Focus Measure|
|---|---|
|25.0 mm|110|
|25.2 mm|260|
|25.4 mm|580|
|25.6 mm|920|
|25.8 mm|610|
|26.0 mm|310|

Autofocus Focus Curve

示意數據：Focus Measure 隨相機 Z 軸位置先上升後下降。

02505007501K25.025.225.425.625.826.0

最高分出現在 Z = 25.6 mm，因此可以將它作為候選最佳對焦位置。

但要注意：Focus Measure 不是物理上的絕對清晰度，也不保證任何影像中最高分的位置就是真正最佳焦點。

它是一個依賴影像內容與演算法的 Optimization Objective。

## 1.3 Gradient 是什麼？

Gradient（梯度）代表影像亮度在空間中的變化率。

假設灰階影像為：

\[ I(x,y) \]

影像的 Gradient 為：

\[ \nabla I = \begin{bmatrix} \frac{\partial I}{\partial x}\\ \frac{\partial I}{\partial y} \end{bmatrix} \]

其中：

- \(G_x=\frac{\partial I}{\partial x}\)：水平方向的亮度變化率。
    
- \(G_y=\frac{\partial I}{\partial y}\)：垂直方向的亮度變化率。
    

Gradient Magnitude：

\[ G(x,y)=\sqrt{G_x^2+G_y^2} \]

想像手錶錶盤上的黑色印刷文字，與白色錶盤背景：

Blurred — 邊緣過渡緩慢

Focused — 邊緣變化劇烈

概念示意：兩種影像的邊緣變化。實際 Gradient 數值也取決於曝光、對比和雜訊。

清楚的黑白邊界會在少數 Pixel 內完成亮度變化，因此 Gradient 通常比較大。

模糊影像則會讓同樣的黑白過渡分散到更多 Pixel，Gradient 峰值下降。

這就是 Gradient-based Autofocus 的核心原理。

## 1.4 Laplacian Autofocus 的原理

Laplacian 是影像的二階空間導數：

\[ \nabla^2 I= \frac{\partial^2I}{\partial x^2} + \frac{\partial^2I}{\partial y^2} \]

常見的離散 Laplacian Kernel：

\[ K= \begin{bmatrix} 0&1&0\\ 1&-4&1\\ 0&1&0 \end{bmatrix} \]

透過 Convolution：

\[ L(x,y)=I*K \]

接著計算 Laplacian Variance：

\[ F_L=\frac{1}{N}\sum_{i=1}^{N} (L_i-\bar L)^2 \]

其中：

- \(L_i\)：Laplacian Response。
    
- \(\bar L\)：所有 Response 的平均。
    
- \(N\)：Pixel 數目。
    

直覺解釋： 清楚影像通常包含較強烈的局部亮度曲率變化，使 Laplacian Response 的變異程度增加。

Python 範例：

```
import cv2import numpy as npdef laplacian_focus(image):    gray = cv2.cvtColor(        image, cv2.COLOR_BGR2GRAY    )    gray = cv2.GaussianBlur(        gray, (3, 3), 0.6    )    lap = cv2.Laplacian(        gray, cv2.CV_64F    )    return lap.var()
```

Gaussian Blur 在此是為了抑制部分 Noise，但也會損失高頻細節，所以 Kernel 與 Sigma 不宜過大。

Laplacian 的主要問題：

影像 Noise 本身通常也包含高頻成分，因此即使影像沒有對焦成功，Noise 也可能提高 Laplacian Variance。

例如某張暗部影像因高 Gain 出現大量雜訊，其 Laplacian 分數可能高於一張較清楚但較乾淨的影像。

這是典型的 False Focus Peak。

## 1.5 Tenengrad Autofocus 的原理

Tenengrad 是基於 Sobel Gradient 的 Focus Measure。

Sobel Operator 通常使用兩個 Kernel：

\[ S_x= \begin{bmatrix} -1&0&1\\ -2&0&2\\ -1&0&1 \end{bmatrix} \]

\[ S_y= \begin{bmatrix} -1&-2&-1\\ 0&0&0\\ 1&2&1 \end{bmatrix} \]

計算：

\[ G_x=I*S_x,\qquad G_y=I*S_y \]

再計算：

\[ G^2=G_x^2+G_y^2 \]

Tenengrad Score 可以定義為：

\[ F_T= \frac{1}{N} \sum_{x,y} G^2(x,y) \]

某些實作只累計超過 Threshold 的 Gradient：

\[ F_T= \frac{1}{N} \sum_{x,y} G^2(x,y)\, \mathbf{1}\{G(x,y)>T\} \]

這個 Threshold 可以減少弱 Gradient 與部分雜訊的干擾。

Python：

```
def tenengrad_focus(image):    gray = cv2.cvtColor(        image, cv2.COLOR_BGR2GRAY    )    gray = cv2.GaussianBlur(        gray, (3, 3), 0.6    )    gx = cv2.Sobel(        gray, cv2.CV_64F, 1, 0, ksize=3    )    gy = cv2.Sobel(        gray, cv2.CV_64F, 0, 1, ksize=3    )    gradient_sq = gx**2 + gy**2    return gradient_sq.mean()
```

### Laplacian 與 Tenengrad 比較

|項目|Laplacian Variance|Tenengrad|
|---|---|---|
|導數|二階|一階|
|主要訊號|局部曲率、高頻細節|Edge Gradient|
|計算成本|低|低|
|Noise 敏感性|通常較高|相對較低，但仍有影響|
|適合影像|細紋、文字、細微結構|清楚邊界、線條、刻字|
|限制|高頻 Noise 易造成假峰值|強反光與高對比邊緣可能主導分數|
|關鍵設計|Denoising 與 ROI|Threshold 與 ROI|

不能簡單宣稱 Tenengrad 一定比 Laplacian 準確，因為最佳演算法與使用的 Lens、Noise Characteristics、Illumination、Target Texture 有關。

## 1.6 Search Strategy：如何找到最佳 Focus Position？

演算法算出 Focus Score 後，還需要 Search Strategy。

常見策略如下：

|搜尋方式|原理|適用情境|
|---|---|---|
|Exhaustive Search|掃描所有位置|搜尋範圍小、需要可靠 baseline|
|Coarse-to-Fine|大步搜尋，再小步細找|工業 Autofocus 常用|
|Golden-section Search|透過區間縮減搜尋峰值|Focus Curve 近似單峰時|
|Hill Climbing|沿分數增加方向移動|起始位置已接近焦點|
|Bayesian Optimization|用 Probabilistic Model 決定下一個位置|每次拍攝成本很高時|
|Sensor-assisted AF|先用距離感測器估計，再影像微調|可搭配 Laser、Depth Sensor|

其中我會優先選擇 Coarse-to-Fine + Focus Curve Validation。

例如：

Initial Z = 25.0 mm

Coarse Scan

25.0–26.0 mm，Step = 0.1 mm

Candidate Peak = 25.6 mm

Fine Scan

25.5–25.7 mm，Step = 0.02 mm

Best Z = 25.58 mm

Move Back → Recapture → Verify

這些 Step Size 是示意數值；實際必須依 Depth of Field、Motor Repeatability 與 Imaging Magnification 決定。

### Senior Engineer 必須考慮的問題

第一：Noise 與 Exposure

Autofocus 期間盡可能固定 Exposure、Gain、Lighting 與 Image Processing，以免 Focus Score 的變化不是來自 Focus，而是來自曝光條件。

第二：ROI Selection

例如手錶有大面積平滑錶盤與局部文字。如果直接評估整張影像，強反光可能比目標刻字產生更大的 Focus Score。

因此應該選擇具有足夠紋理、與任務相關，而且在整個掃描過程中保持可見的 ROI。

第三：Mechanical Hysteresis 與 Backlash

Z 軸從上方移到某位置，與從下方移到同一位置，實際定位可能存在差異。

因此最後定位最好遵循一致的 Approach Direction，必要時先越過目標再從同一方向接近。

第四：Focus Confidence

不是每次找到最大值就代表成功。

可以驗證：

\[ \text{Peak Prominence} = F_{\max}-F_{\text{baseline}} \]

以及 Focus Curve 是否有可辨識的峰值、峰值是否位於搜尋邊界、重新拍攝是否能重現結果。

如果 Focus Score 幾乎不變，可能代表 ROI 缺乏紋理、目標不在搜尋範圍，或對焦致動器沒有正確動作。

## 1.7 具體應用：手錶微距影像 Autofocus

假設要拍攝 Rolex 錶盤上的極小印刷字母。

系統包含：

- Motorized Z Stage。
    
- Macro Camera。
    
- 可調式 Ring Light。
    
- Keyence Distance Sensor。
    
- Laplacian 或 Tenengrad Focus Measure。
    

我會設計：

1. 用 Keyence 讀取表面距離，估計初始 Z Position。
    
2. 使用固定曝光與適合的照明角度。
    
3. 在文字區域建立 ROI，避開飽和反光。
    
4. 使用 Tenengrad 執行 Coarse Scan。
    
5. 在最佳候選附近執行 Fine Scan。
    
6. 重新拍攝最佳位置，檢查 Focus Score、Peak Confidence 和 Image Quality。
    
7. 如果沒有有效峰值，改用另一 ROI、擴大搜尋範圍或啟動安全的 Fallback Strategy。
    

面試時再補充一個關鍵：

最好的 Autofocus 不一定是使整張影像最清晰，而是使下游重要特徵最容易被可靠辨識。

例如任務是辨識極細文字，文字筆畫區域的 Focus Quality 可能比整個錶盤的平均 Sharpness 更重要。

# Q2. HDR Merge 如何處理不同曝光影像？

## 2.1 面試核心回答

> HDR Imaging 的目標是擷取單一曝光無法同時保留的亮部與暗部細節。
> 
> 我會先控制 Exposure Bracketing，取得不同 Exposure Time 的影像，然後進行 Image Alignment，再根據每個 Pixel 的 Exposure Reliability 進行 Weighting 和 Fusion。
> 
> 傳統 HDR Radiance Reconstruction，例如 Debevec，會使用 Camera Response Function 和 Exposure Time 將影像轉換成 Scene Radiance，再執行 Tone Mapping。
> 
> 另一種方法是 Mertens Exposure Fusion，它直接依照 Contrast、Saturation 與 Well-exposedness 組合不同曝光的影像，不一定需要重建真實 Radiance。
> 
> Production 系統還必須處理 Saturation、Noise、Alignment Error、Motion Ghosting，以及 Fusion 後對色彩和細節的影響。

## 2.2 Exposure 是什麼？

Exposure（曝光）是影像感測器接收到的光能量。

在簡化且光照固定的情況下：

\[ H \propto L\cdot t \]

其中：

- \(H\)：感測器曝光量。
    
- \(L\)：到達感測器的照度。
    
- \(t\)：Exposure Time。
    

實際還受 Aperture、光學透射率與場景亮度等因素影響。

一張影像如果曝光太短：

- 暗部訊號很弱。
    
- 細節接近 Noise Floor。
    
- SNR 可能很差。
    

曝光太長：

- 亮部 Pixel Saturation。
    
- 高光細節變成無法區分的最大值。
    
- Motion Blur 風險增加。
    

例如金屬手錶：

|Exposure Time|亮部金屬|暗部刻字|
|---|---|---|
|0.05 ms|保留強反光區細節|非常暗|
|0.12 ms|高光部分保留|部分可見|
|0.50 ms|多處過曝|清楚|
|3.20 ms|大面積飽和|很清楚，但可能模糊|

這四個曝光時間只是示範不同動態範圍的取捨。

如果同一張照片無法同時保留亮部與暗部，就可以考慮 Exposure Bracketing。

## 2.3 什麼是 Dynamic Range？

Dynamic Range（動態範圍）代表一個系統能夠保留的最大與最小有效訊號的比例。

\[ DR_{\mathrm{dB}} = 20\log_{10} \left(\frac{S_{\max}}{S_{\min}}\right) \]

其中 \(S_{\min}\) 一般需要根據可接受的 Noise 或 SNR 定義。

例如手錶表面可能同時包含：

- 高反射的 Stainless Steel。
    
- 深色錶盤。
    
- 金色刻度。
    
- 很細的黑色文字。
    

這些部位的亮度差異可能很大，因此單次曝光不一定能完整擷取所有資訊。

需要注意：影像是 8-bit 還是 16-bit，不會單獨決定感測器的實際 Dynamic Range。

Bit Depth 是數值表示能力；實際可保留的動態範圍還取決於感測器 Full-well Capacity、Read Noise、曝光條件與影像處理鏈。

## 2.4 Alignment：為什麼不同曝光需要對齊？

在多張曝光之間，可能出現：

- Camera Motion。
    
- Stage Vibration。
    
- Subject Motion。
    
- 微小的定位偏差。
    
- 不同曝光造成的亮度變化。
    

因此同一個物理特徵，在不同影像中的 Pixel Coordinates 未必完全一樣。

例如短曝光影像的刻字位於：

\[ (x,y)=(100,200) \]

長曝光影像可能位於：

\[ (x,y)=(102,201) \]

如果直接融合，刻字可能產生雙邊緣或模糊。

### 常見 Alignment 方法

|方法|適合情況|主要限制|
|---|---|---|
|Translation Alignment|相機只有平移|無法處理旋轉或變形|
|ECC Alignment|強度或幾何變化較小|初始化與影像亮度差異會影響收斂|
|Feature-based Alignment|影像具有可辨識特徵|低紋理或飽和區域匹配困難|
|Optical Flow|局部非剛性運動|可能產生錯誤變形|
|AlignMTB|不同曝光影像|適合特定曝光對齊場景，不保證微米級精度|

OpenCV 提供 `createAlignMTB()` 等工具。

在固定工業機台中，我會先盡量消除物理運動，再使用 Software Alignment 修正殘餘偏差，因為過度依賴非剛性 Warp 可能改變缺陷與量測幾何。

## 2.5 Weighting：為什麼不是直接把影像平均？

假設同一個 Pixel 在四張影像中的亮度為：

\[ I=[20,\ 80,\ 180,\ 255] \]

如果直接 Average：

\[ I_{\mathrm{avg}}=133.75 \]

問題是最後一張的 255 可能已經 Saturated，第一張的 20 可能太接近 Noise Floor。

所以每個 Exposure 的 Pixel 應該有不同的 Reliability Weight。

典型權重設計會降低：

- 太暗的 Pixel。
    
- 太亮或已飽和的 Pixel。
    
- 受到 Motion 影響的 Pixel。
    
- Noise 特別大的 Pixel。
    

例如：

\[ w(z)= \begin{cases} z,&z\le 0.5\\ 1-z,&z>0.5 \end{cases} \]

這是對正規化亮度 \(z\in[0,1]\) 的簡化三角權重函數。

它偏好不接近 0 或 1 的 Pixel。

### Debevec HDR Radiance Reconstruction

假設：

\[ Z_{ij}=f(E_i t_j) \]

其中：

- \(Z_{ij}\)：第 \(i\) 個 Pixel 在第 \(j\) 張影像的值。
    
- \(E_i\)：場景 Radiance 對應的訊號。
    
- \(t_j\)：Exposure Time。
    
- \(f\)：Camera Response Function。
    

定義：

\[ g=f^{-1}\text{ 的對數表示} \]

經校正後可估計：

\[ \ln E_i= \frac{ \sum_j w(Z_{ij}) [g(Z_{ij})-\ln t_j] }{ \sum_j w(Z_{ij}) } \]

這個式子的重點是：不同曝光值必須轉換到可比較的輻射量尺度，才適合做物理上有意義的融合。

之後使用 Tone Mapping，將高動態範圍的 Radiance Image 轉換成顯示設備能呈現的影像。

## 2.6 Mertens Exposure Fusion 是什麼？

Mertens Exposure Fusion 不要求先重建物理 Radiance Map。

它直接從不同 Exposure 的影像計算局部品質權重。

典型有三項：

\[ W_k(x,y)=C_k^\alpha S_k^\beta E_k^\gamma \]

其中：

- \(C_k\)：Contrast。
    
- \(S_k\)：Saturation。
    
- \(E_k\)：Well-exposedness。
    
- \(\alpha,\beta,\gamma\)：對應權重的指數。
    

Contrast 偏好局部細節強的區域。

Saturation 偏好色彩較豐富的區域。

Well-exposedness 偏好不太暗、不太亮的區域。

接著正規化權重：

\[ \widehat{W}_k= \frac{W_k}{\sum_j W_j+\epsilon} \]

在實際 Mertens 方法中，通常會建立 Gaussian Weight Pyramids 與 Laplacian Image Pyramids，做 Multi-resolution Fusion：

\[ L_{\mathrm{fused}}^{(l)} = \sum_k G(W_k)^{(l)} L(I_k)^{(l)} \]

最後重建影像。

使用 Pyramid 的原因之一，是減少直接以局部權重合成時可能出現的不自然邊界。

OpenCV 範例：

```
import cv2images = [    cv2.imread("exp1.png"),    cv2.imread("exp2.png"),    cv2.imread("exp3.png"),    cv2.imread("exp4.png")]merge = cv2.createMergeMertens(    contrast_weight=1.0,    saturation_weight=1.0,    exposure_weight=1.0)fused = merge.process(images)cv2.imwrite(    "fused.png",    (fused.clip(0, 1) * 255).astype("uint8"))
```

這裡明確設定三項權重皆為 1；不同 OpenCV 版本或 API 文件中的預設參數應分別確認。

很重要的面試區別：

|方法|Debevec HDR|Mertens Fusion|
|---|---|---|
|目標|Radiance Reconstruction|視覺品質融合|
|需要曝光時間|是|通常不需要|
|需要 Camera Response|通常需要估計或校正|不需要|
|輸出|HDR Radiance Map|通常是 Display-oriented LDR|
|Tone Mapping|需要|通常不需要額外 Tone Mapping|
|定量光度分析|較適合經完整校正的流程|不宜直接當作物理 Radiance|

OpenCV 的 HDR 教學也明確區分 HDR Radiance Reconstruction 與 Mertens Exposure Fusion。

![](https://www.google.com/s2/favicons?domain=https://docs.opencv.org&sz=32)

OpenCV Documentation

+1

## 2.7 Ghosting 是什麼？如何解決？

Ghosting 是多張影像融合後，同一個物體出現在不同位置而形成的重影。

例如：

- 秒針在四次曝光之間移動。
    
- 相機受到震動。
    
- 光滑金屬上的反光隨視角改變。
    
- 自動對焦鏡片尚未完全穩定。
    

可能出現雙層邊緣、重複文字或不自然的影像結構。

需要區分：

Alignment 解決的是不同影像之間的幾何位置偏差。

Deghosting 解決的是場景內容在不同曝光之間發生改變，無法用單一全域幾何轉換消除的問題。

常見 Deghosting 策略包括建立 Motion Mask，將不一致區域改由參考曝光影像提供，或採用 Motion-aware Fusion。

對手錶檢測，我會特別注意秒針可能移動，但錶盤文字不會移動。因此可對 Motion Region 與 Static Region 使用不同的 Fusion Policy。

## 2.8 Senior／Staff 層級的系統設計

在精密手錶檢測中，假設需要辨識微小刻字與表面缺陷，我會把兩種影像分開保存：

Raw Exposure Images

用於可追溯的量測、模型訓練、色彩分析與重新處理。

HDR／Exposure-fused Images

用於改善人員檢視、部分 Feature Extraction 或降低飽和區域造成的資訊缺失。

因為 Mertens 可能透過 Contrast Weighting 改變局部細節強度，所以我不會直接假設 Fused Image 適合所有 Defect Detection 或 Color Measurement。

尤其對極微小刮痕，Fusion 過程可能提高、降低，甚至產生類似缺陷的局部紋理。

我會另外建立驗證數據集，比較 Raw Single Exposure 與 HDR Fusion 對下游 Detection、Segmentation 及 Measurement Error 的實際影響。

高階面試的關鍵回答是：HDR 應該改善真實可用資訊，而不是只讓影像看起來比較漂亮。

# Q3. 如何校正 Camera Lens Distortion？

## 3.1 面試核心回答

> Camera Lens Distortion Calibration 的目標是建立實際 Camera Projection Model，估計 Intrinsic Parameters 和 Distortion Coefficients，再將 Distorted Pixel Coordinates 映射到校正後的影像。
> 
> Intrinsic Matrix 描述 Focal Length 與 Principal Point 等相機內部投影參數。
> 
> Radial Distortion 主要造成 Barrel 或 Pincushion 型的非線性變形；Tangential Distortion 則描述鏡片與感測器不理想對準所引起的非對稱變形。
> 
> 我通常會使用已知幾何尺寸的 Checkerboard 或 Charuco Calibration Target，拍攝不同位置及傾斜角度的多張影像，使用 Nonlinear Optimization 估計參數。
> 
> 最後不只看 Reprojection Error，還會用獨立的 Calibration Target 和實際量測任務驗證 Geometric Accuracy。

## 3.2 先理解 Camera Projection

理想 Pinhole Camera Model：

\[ x_n=\frac{X_c}{Z_c},\qquad y_n=\frac{Y_c}{Z_c} \]

其中：

- \(X_c,Y_c,Z_c\)：Camera Coordinate System 中的 3D 位置。
    
- \(x_n,y_n\)：Normalized Image Coordinates。
    

將 Normalized Coordinates 轉成 Pixel Coordinates：

\[ u=f_x x_n+c_x \]

\[ v=f_y y_n+c_y \]

這可寫成：

\[ s \begin{bmatrix}u\\v\\1\end{bmatrix} = K[R|t] \begin{bmatrix}X\\Y\\Z\\1\end{bmatrix} \]

其中 \(K\) 是 Intrinsic Matrix，而 \(R,t\) 是 Camera Extrinsic Parameters。

## 3.3 Intrinsic Matrix 是什麼？

最常使用的 Camera Intrinsic Matrix：

\[ K= \begin{bmatrix} f_x&0&c_x\\ 0&f_y&c_y\\ 0&0&1 \end{bmatrix} \]

其中：

- \(f_x\)：X 方向以 Pixel 為單位的有效焦距。
    
- \(f_y\)：Y 方向以 Pixel 為單位的有效焦距。
    
- \(c_x\)：Principal Point 的水平座標。
    
- \(c_y\)：Principal Point 的垂直座標。
    

例如：

\[ K= \begin{bmatrix} 2500&0&1428\\ 0&2505&1424\\ 0&0&1 \end{bmatrix} \]

代表這台示意相機的有效焦距約 2500 pixels，而 Principal Point 接近 \((1428,1424)\)。

這不是實際相機的校正結果，只是範例。

另外，有些更一般化的模型還會加入 Skew Parameter，但現代數位相機常假設 Skew 為 0。

### Intrinsic 與 Extrinsic 有什麼不同？

Intrinsic 描述相機的內部投影特性。

Extrinsic 描述相機相對於世界座標系的旋轉與平移。

例如同一台相機從 Z = 20 mm 移動到 Z = 30 mm，理想情況下它的相機內參不會只因為外部位置改變就必然改變，但外參會改變。

不過，如果伴隨重新對焦、變焦或內部鏡片位置改變，Intrinsic Parameters 也可能改變。

## 3.4 Radial Distortion 是什麼？

Radial Distortion（徑向畸變）是隨著影像點距離 Distortion Center 增加而改變的畸變。

![Questions About Lenses I Was Afraid to Ask | by Michael Alford | Live View | Medium](https://images.openai.com/static-rsc-4/7vJ4NGZWD2b-82NUaDSW0Txd9MqFy4iibSXWw0ggZ2jrsHX7NlQoqsv7XIwOe8x0O8ASK_El0oE-G1_Fht3mLqHXcGPdVzcA8GAYewmu0DBqnL2iIuY8UviVPkEM6_n4Rt4LOdadxUiowe6jdByUqYjTHSn8_PBtkkqSqkgBsIk?purpose=inline)

Barrel Distortion

影像中的直線向外鼓起

![OpenCV: Camera Calibration](https://images.openai.com/static-rsc-4/hhjukUFVLw0KCQmfqulK5M3-3RP1kMLuaGjGt3RehUEIgtPTXpUd43xixWEJXfetQV6gP2PEavhxERuiWHqiEwCr2gG9ioutLuuscxnZe02DNyLZqgB22rWsFDU1Ky35iD54Vow4dkyKpW6LjxfGdiglW1adIVmwZdV-RH6l4sY?purpose=inline)

Undistorted

經模型校正後，直線更接近理想幾何

常見畸變包含：

Barrel Distortion（桶狀畸變）：直線看起來向外鼓起，常見於部分廣角鏡頭。

Pincushion Distortion（枕狀畸變）：直線看起來向中心凹入，常見於某些鏡頭配置。

使用 Brown–Conrady 類型的模型：

\[ r^2=x^2+y^2 \]

\[ x_r=x(1+k_1r^2+k_2r^4+k_3r^6) \]

\[ y_r=y(1+k_1r^2+k_2r^4+k_3r^6) \]

其中 \(x,y\) 是理想的 Normalized Coordinates。

\(k_1,k_2,k_3\) 是 Radial Distortion Coefficients。

越靠近影像外圍，畸變通常越值得注意。

## 3.5 Tangential Distortion 是什麼？

Tangential Distortion（切向畸變）主要描述光學系統元件未完全對準造成的非對稱變形。

常見模型：

\[ x_t=2p_1xy+p_2(r^2+2x^2) \]

\[ y_t=p_1(r^2+2y^2)+2p_2xy \]

其中 \(p_1,p_2\) 是 Tangential Distortion Coefficients。

最後 Distorted Coordinates：

\[ x_d=x_r+x_t \]

\[ y_d=y_r+y_t \]

所以一般五參數模型包含：

\[ D=[k_1,k_2,p_1,p_2,k_3] \]

OpenCV 支援這種 Radial／Tangential Model，也支援更多高階參數和其他投影模型。

![](https://www.google.com/s2/favicons?domain=https://docs.opencv.org&sz=32)

OpenCV Documentation

+1

## 3.6 如何執行 Camera Calibration？

### Step 1：準備已知幾何尺寸的 Calibration Target

例如 9 × 6 個 Internal Corners 的 Checkerboard，每個 Square 大小為 2 mm。

建立每個 Corner 的實際世界座標：

\[ P_i=(X_i,Y_i,0) \]

因為 Checkerboard 在同一平面，Z = 0。

### Step 2：拍攝不同位置與角度

例如拍攝 20–30 張不同姿態的 Checkerboard 影像。

需要覆蓋：

- 影像中心。
    
- 四角及邊緣。
    
- 不同 Tilt Angles。
    
- 不同適當的 Orientation。
    

重點是取得充分的幾何變化，而不只是增加影像張數。

### Step 3：偵測 Checkerboard Corners

```
found, corners = cv2.findChessboardCorners(    gray, (9, 6))
```

再使用 `cornerSubPix()` 或較新的高精度角點偵測方法改善座標估計。

### Step 4：估計 Camera Parameters

```
rms, K, dist, rvecs, tvecs = \    cv2.calibrateCamera(        object_points,        image_points,        image_size,        None,        None    )
```

估計過程通常會最小化 Reprojection Error：

\[ \min_{\theta} \sum_{i,j} \left\| p_{ij}^{\mathrm{observed}}- \pi(\theta,P_{ij}) \right\|^2 \]

\(\theta\) 包含 Intrinsic、Distortion 和各張影像的 Extrinsic Parameters。

### Step 5：Undistort Image

```
corrected = cv2.undistort(    image, K, dist)
```

在高吞吐量 Production 系統中，可以使用：

```
cv2.initUndistortRectifyMap()cv2.remap()
```

預先建立 Remapping Map，避免每張影像重新計算全部映射。

## 3.7 如何衡量 Calibration 是否成功？

第一個指標是 Reprojection Error。

\[ RMSE= \sqrt{ \frac{1}{N} \sum_i \left\| p_i-\hat p_i \right\|^2 } \]

例如：

|Calibration 結果|Reprojection RMSE|
|---|---|
|Model A|0.80 pixel|
|Model B|0.25 pixel|
|Model C|0.12 pixel|

一般來說，較低 Reprojection Error 表示模型更能解釋觀測到的 Calibration Data。

但不能只看 RMSE。

例如一個過度複雜的 Distortion Model，可能在 Training Calibration Images 上得到很低的 Error，卻無法在新影像上維持準確。

因此我會用獨立的 Target 測試：

- 校正後的直線殘餘彎曲程度。
    
- 已知長度的 Measurement Error。
    
- 不同影像位置的 Geometric Error。
    
- 不同 Working Distance 的誤差。
    
- 重複校正的 Parameter Stability。
    

## 3.8 Senior／Staff 延伸：微距與精密量測有什麼特殊問題？

在微距手錶檢測中，不能假設一次 Calibration 就適用所有情境。

例如液態鏡片改變焦距、不同工作距離、不同鏡頭、不同 ROI Crop，都可能使有效投影參數改變。

對某些近似 Telecentric Lens 的系統，正交投影模型可能比標準 Pinhole Model 更適合。

另外，Lens Distortion Calibration 不等於完整的世界座標量測校正。

如果希望精準測量兩個刻字之間的毫米距離，除了畸變修正，還可能需要：

- Pixel-to-mm Scale Calibration。
    
- Camera-to-Stage Extrinsic Calibration。
    
- 影像平面與物體平面的幾何校正。
    
- Stage Movement Accuracy Validation。
    
- Measurement Uncertainty Analysis。
    

例如某系統達到 0.2 Pixel 的 Reprojection RMSE，並不代表它在任何物體深度都能達到 0.2 Pixel 的真實量測精度。

面試重點：Distortion Calibration 的成功標準，應由最終 Geometric Measurement Accuracy 決定，而不能只看 Calibration Software 輸出的 RMSE。

# Q4. Image Stitching 有哪幾個主要步驟？

## 4.1 面試核心回答

> Image Stitching 是將具有 Overlap 的多張影像對齊到共同座標系，再融合成一張較大視野的影像。
> 
> 典型 Pipeline 包含 Feature Detection、Feature Description、Feature Matching、Robust Transformation Estimation、Image Warping、Seam Finding 與 Blending。
> 
> 我會使用 SIFT 或 ORB 等 Feature Detector 找出不同影像之間的 Correspondences，再利用 RANSAC 排除錯誤 Matching，估計 Homography 或其他適合的 Geometric Transformation。
> 
> 對齊後，使用 Warping 將影像轉移到同一座標系，接著做 Exposure Compensation、Seam Optimization 及 Multi-band Blending。
> 
> 如果是 Precision Imaging，我還會額外評估 Registration Error、Parallax、Geometric Distortion，以及 Stitching 對下游 Measurement 的影響。

## 4.2 完整 Image Stitching Pipeline

Image A

Image B

Image C

1. Preprocessing

Undistortion、Color / Exposure Normalization

2. Feature Detection

SIFT、ORB、AKAZE

3. Feature Matching

Descriptor Matching、Ratio Test

4. RANSAC

Outlier Rejection

5. Transformation Estimation

Homography / Affine / Translation

6. Warping

Common Coordinate System

7. Blending

Seam、Multi-band Fusion

Stitched Image

## 4.3 Feature Detection 是什麼？

Feature Detection 是從影像中找出容易重複辨識的位置，例如：

- Corners。
    
- 高對比的局部紋理。
    
- 特殊幾何交會點。
    
- 穩定的局部影像結構。
    

假設要 Stitch 兩張 Rolex 錶面照片。

第一張照片有部分文字：

`SUBMARINER`

第二張照片與第一張部分重疊，也包含部分相同文字。

Feature Detector 可以從字母筆畫的端點、交叉點或其他局部結構找出 Candidate Keypoints。

常見方法：

|方法|核心特色|優點|缺點|
|---|---|---|---|
|Harris Corner|利用局部梯度變化找角點|快速、原理簡單|缺少原生 Scale Invariance|
|SIFT|Scale-space 特徵與 Descriptor|對 Scale、Rotation 較穩健|計算較昂貴|
|ORB|FAST Keypoint + Binary Descriptor|快速，易部署|某些尺度或外觀變化下較不穩定|
|AKAZE|Nonlinear Scale Space|局部特徵描述能力良好|依設定有不同計算成本|

需要特別區分兩個名詞：

Keypoint：影像中值得比對的位置。

Descriptor：用向量或二進位特徵來描述 Keypoint 周圍的局部影像。

例如 SIFT Descriptor 典型為 128 維特徵向量。

ORB 常用 Binary Descriptor。

## 4.4 Feature Matching 是什麼？

Feature Matching 的目標是找出兩張影像中對應同一實際物理位置的 Keypoints。

假設：

Image A：

\[ P_A=(300,500) \]

Image B：

\[ P_B=(120,510) \]

如果兩者是同一刻字的角點，它們就是一組 Candidate Correspondence。

對 Descriptor 進行距離比對：

SIFT 常用 Euclidean Distance：

\[ d(a,b)=\sqrt{\sum_i(a_i-b_i)^2} \]

ORB Binary Descriptor 常用 Hamming Distance。

但只選擇距離最小的 Descriptor 並不可靠。

因此常使用 Lowe's Ratio Test：

\[ \frac{d_1}{d_2}<\tau \]

其中 \(d_1,d_2\) 是最接近與第二接近的 Descriptor Distance。

例如：

\[ d_1=50,\quad d_2=100 \]

Ratio = 0.5，代表最佳匹配比第二匹配明顯更接近。

但是：

\[ d_1=90,\quad d_2=100 \]

Ratio = 0.9，表示兩者非常接近，匹配具有較高歧義。

常見 Ratio Threshold 約為 0.7–0.8，但需要根據資料驗證，而非固定不變。

## 4.5 RANSAC 是什麼？為什麼重要？

RANSAC（Random Sample Consensus）是用來在存在錯誤 Correspondences 時，穩健估計幾何模型的方法。

例如：

100 組 Feature Matches：

- 75 組是真正正確的 Matches。
    
- 25 組是 Outliers。
    

如果直接以全部 Matches 估計 Homography，錯誤匹配可能造成嚴重的 Transformation Error。

RANSAC 的流程：

1. 隨機選擇足夠數量的 Matching Points。
    
2. 建立候選 Transformation Model。
    
3. 計算其他 Matching Points 的 Reprojection Error。
    
4. 將符合 Threshold 的點標記為 Inliers。
    
5. 重複多次，選出高共識的候選模型。
    
6. 用選出的 Inliers 重新估計最佳 Transformation。
    

### RANSAC 的數學原理

若假設每次隨機取 \(s\) 個點，Inlier Ratio 為 \(w\)，希望至少有一次抽樣全部為 Inliers 的機率達到 \(p\)，所需迭代次數約為：

\[ N= \frac{\log(1-p)} {\log(1-w^s)} \]

以 Homography 為例，非退化的最小樣本需要 4 組 Correspondences。

假設：

\[ w=0.7,\quad s=4,\quad p=0.99 \]

得到約：

\[ N\approx17 \]

所以至少約 17 次隨機抽樣，才能在理想假設下達到約 99% 的成功機率。

實際 RANSAC 還必須考慮匹配點的分布、退化幾何、Reprojection Threshold、雜訊與停止策略。

### 為什麼 RANSAC 不能解決所有問題？

想像錶盤上有 12 個外形非常相似的 Hour Markers。

錯誤 Matching 可能並非完全隨機，而是由重複且對稱的結構造成。

RANSAC 有機會選到數量很多、幾何上看似一致、但實際位置錯誤的 Correspondences。

所以 Precision Stitching 還需要額外的幾何限制，例如 Stage Position Prior、Rotation Range、Expected Scale 與固定相機結構。

## 4.6 Homography 是什麼？

Homography 是 3×3 Projective Transformation Matrix：

\[ H= \begin{bmatrix} h_{11}&h_{12}&h_{13}\\ h_{21}&h_{22}&h_{23}\\ h_{31}&h_{32}&h_{33} \end{bmatrix} \]

將一個影像座標映射到另一個影像座標：

\[ s \begin{bmatrix} x'\\y'\\1 \end{bmatrix} = H \begin{bmatrix} x\\y\\1 \end{bmatrix} \]

因為矩陣可乘任意非零 Scale 而不改變 Projective Mapping，所以 Homography 有 8 個獨立自由度。

在 OpenCV：

```
H, inliers = cv2.findHomography(    src_points,    dst_points,    cv2.RANSAC,    3.0)
```

其中 `3.0` 是示意的 Reprojection Error Threshold，單位為 Pixels。

### Homography 什麼時候有效？

Homography 特別適合：

1. 拍攝近似同一個平面物體。
    
2. 相機只有純旋轉，而沒有造成深度視差的平移。
    
3. 特定可以用單一平面投影描述的場景。
    

例如拍攝平坦的錶盤表面，Homography 可能是一個合理模型。

但如果手錶時針、分針、錶盤與水晶玻璃位於不同深度，並且相機發生平移，可能產生 Parallax。

這時單一 Homography 不一定能同時正確對齊所有結構。

OpenCV 的 Homography 文件也將它定位為平面之間的 Projective Mapping，並搭配 RANSAC 處理 Outlier Correspondences。

![](https://www.google.com/s2/favicons?domain=https://github.com&sz=32)

GitHub

+2

## 4.7 Warping 與 Blending 有什麼不同？

Warping 解決幾何問題；Blending 解決外觀接縫問題。

Warping：

```
warped = cv2.warpPerspective(    image,    H,    output_size)
```

將來源影像的 Pixel 重新取樣到共同的 Coordinate System。

但是兩張影像就算幾何對齊，也可能因為：

- Brightness Difference。
    
- White Balance Difference。
    
- Lens Vignetting。
    
- Local Reflections。
    
- Image Noise。
    

導致接縫很明顯。

因此需要 Blending。

常見方法：

|Blending|原理|特性|
|---|---|---|
|Alpha Blending|以權重平均重疊區域|簡單，但殘餘錯位會模糊|
|Feather Blending|權重隨接縫距離變化|改善邊界不連續|
|Multi-band Blending|使用不同頻率的影像金字塔融合|能較好地處理不同尺度的接縫|
|Seam Optimization|選擇較不顯眼的接縫路徑|可避開重要特徵或局部錯位|

### 為什麼 Multi-band Blending 有效？

因為不同空間頻率代表不同類型的影像資訊。

低頻資訊主要包含大尺度亮度與色調變化。

高頻資訊則包含邊緣、文字筆畫與細微紋理。

Multi-band Blending 可以在不同尺度上使用不同的權重分布，避免直接平均造成大範圍亮度跳變。

但它並不會從根本上修正錯誤的幾何配準。

## 4.8 具體案例：五張手錶 Front Images Stitching

假設使用五張具有 Overlap 的 Front Images：

`0004、0005、0008、0009、0010`

目標是產生一張完整錶盤影像。

我會這樣設計：

第一階段：Acquisition Validation

確認五張影像都存在，且 Exposure、Focus、Pixel Format 和 Capture Metadata 正確。

第二階段：Geometric Calibration

對相機進行 Undistortion，並讀取 Stage Position 作為相鄰影像的初始 Transformation Prior。

第三階段：Image Registration

先使用已知 Stage Motion 初始化，再用 Feature Matching 或 ECC 作精細對齊。

第四階段：Transformation Estimation

根據運動模型選擇 Translation、Affine 或 Homography；如果存在高度差與 Parallax，則不能強行假設同一個 Homography 足夠。

第五階段：Global Optimization

以所有重疊影像的幾何一致性優化 Transformation，避免逐張接合造成 Cumulative Drift。

第六階段：Image Blending

在保留文字與標記邊界的前提下做 Exposure Compensation、Seam Finding 和 Multi-band Blending。

第七階段：Quality Validation

檢查 Matching Inlier Ratio、Overlap Reprojection Error、Stitching Seams、Missing Region 與重複文字。

對於錶盤上的文字、Hands、Hour Markers，我還會加入 Semantic Validation，確認 Stitching 沒有讓同一個 Marker 出現兩次。

### Senior／Staff 的關鍵取捨

如果 Stage Position 具有良好的 Repeatability，而且 Camera Orientation 固定，我可能不會把 Feature-based Homography 當成唯一的 Registration 方法。

更適合的方案可能是：

Stage Geometry Prior + Image-based Fine Registration + Quality Gate

這會比完全依賴 SIFT／RANSAC 更可預測，也更容易分析 Production Failure。

另外，若 Stitched Image 後續用於幾何量測，必須保留 Warp Transform、Scale 與原始影像座標對應關係，不能直接假設 Stitched Image 中所有 Pixel 都具有相同且準確的毫米尺度。

# Q5. 如何衡量影像 Sharpness？

## 5.1 面試核心回答

> Image Sharpness 代表影像系統保留細節與邊緣對比的能力。
> 
> 如果是即時 Autofocus，我會使用 Laplacian Variance、Tenengrad 或其他 Gradient-based Metrics，因為這些方法計算快速。
> 
> 如果需要定量比較相機、鏡頭或 Imaging System 的 Optical Performance，我會使用 MTF，也就是 Modulation Transfer Function。
> 
> MTF 量測不同 Spatial Frequency 的 Contrast Transfer，通常使用 Slanted-edge 方法估計 MTF50 等指標。
> 
> 我也會區分 Sharpness 與 Image Quality，因為 Sharpness 高不代表 Noise 低、曝光正確或下游辨識效果一定較好。

## 5.2 Sharpness 與 Resolution 有何不同？

這兩個名詞經常被混淆。

Resolution（解析能力） 是系統能否區分足夠接近的兩個細節。

Sharpness（清晰度） 則通常描述影像對邊緣與不同尺度細節的呈現能力。

例如兩台 Camera 都是 4000 × 4000 Pixels：

Camera A 使用高品質光學鏡頭，對焦準確。

Camera B 使用較差鏡頭，產生明顯的 Optical Blur。

兩台 Camera 的 Pixel Count 相同，但 Camera A 可能具有更高的有效解析能力與 Sharpness。

因此：

\[ \text{Pixel Count}\neq\text{Optical Resolution} \]

## 5.3 Laplacian Variance 如何衡量 Sharpness？

這個方法在 Autofocus 已經介紹過。

公式：

\[ F_{\mathrm{sharp}}= \mathrm{Var}(\nabla^2I) \]

直覺上：

清楚影像 → 邊緣變化劇烈 → Laplacian Response 變化大。

模糊影像 → 高頻細節減少 → Laplacian Response 變化小。

優點是計算快，適合 Real-time Monitoring。

缺點是它不是標準化的物理光學品質指標，也容易受到 Noise、Contrast、Exposure、Scene Content 和 Sharpening 影響。

如果 Image A 是純白背景，而 Image B 有大量黑白條紋，即使兩者對焦程度相同，Laplacian Variance 也可能差異很大。

因此它比較適合在同一場景、同一 ROI、固定成像條件下做相對比較。

## 5.4 MTF 是什麼？

MTF（Modulation Transfer Function）代表光學或成像系統，對不同 Spatial Frequency 的 Contrast Transfer 能力。

### 第一步：理解 Spatial Frequency

Spatial Frequency（空間頻率）代表影像明暗變化在單位距離內重複的頻率。

例如：

- 大面積明暗區域 → Low Spatial Frequency。
    
- 細線條 → High Spatial Frequency。
    
- 很密集的黑白條紋 → Very High Spatial Frequency。
    

常見單位：

- Cycles / Pixel。
    
- Line Pairs / mm（lp/mm）。
    
- Cycles / mm。
    

一個 Line Pair 是一條亮線與一條暗線組成的完整週期。

### 第二步：理解 Contrast

對正弦條紋，常用 Michelson Contrast：

\[ C=\frac{I_{\max}-I_{\min}} {I_{\max}+I_{\min}} \]

如果原始物體的 Contrast 為：

\[ C_{\mathrm{input}}=0.8 \]

經過 Camera Imaging 後：

\[ C_{\mathrm{output}}=0.4 \]

則該 Spatial Frequency 的 MTF：

\[ MTF(f)= \frac{C_{\mathrm{output}}(f)} {C_{\mathrm{input}}(f)} =0.5 \]

也就是原始對比只保留了 50%。

### 第三步：MTF Curve

示意 MTF 曲線：隨空間頻率增加，對比逐漸下降

MTF50 為相對低頻響應下降至 50% 所對應的空間頻率。

00.250.50.75100.050.10.150.20.250.30.350.40.450.5

此曲線為示意數據。

當 Spatial Frequency 增加，成像系統通常越難保留原本的 Contrast。

MTF50 是 MTF 下降至低頻參考值 50% 時的 Spatial Frequency。

MTF10 是下降至 10% 時的 Spatial Frequency。

一般而言，在相同且受控的測試條件下，MTF50 越高，代表系統可以在更高 Spatial Frequency 保留一半的初始對比。

但 MTF50 不是唯一的影像品質指標。

## 5.5 PSF、LSF、ESF 與 MTF 的關係

這是面試官判斷你是否真正理解 Optical Imaging 的常見追問。

### PSF：Point Spread Function

假設真實世界有一個理想的光學點。

經過鏡頭成像後，這個點通常不會成為完美的單一 Pixel，而會擴散成某種分布。

這個分布稱為 PSF。

Optical Blur、Diffraction、Defocus 等因素都會改變 PSF。

### MTF 與 PSF 的數學關係

對近似線性且空間不變的成像系統：

\[ I_{\mathrm{observed}} = I_{\mathrm{ideal}}*PSF+N \]

其中 \(N\) 是 Noise。

在 Frequency Domain：

\[ \mathcal{F}\{I_{\mathrm{observed}}\} = \mathcal{F}\{I_{\mathrm{ideal}}\} \cdot OTF +\mathcal{F}\{N\} \]

OTF 是 Optical Transfer Function。

\[ MTF(f)= \frac{|OTF(f)|}{|OTF(0)|} \]

所以 MTF 可以理解為正規化後的 Frequency Response Magnitude。

### Slanted-edge Method

工業 Camera 常用 Slanted-edge Target 量測 MTF。

基本流程：

Capture Slanted Edge

拍攝已知高品質傾斜邊緣

ESF

估計 Edge Spread Function

LSF

對 ESF 微分得到 Line Spread Function

FFT

將 LSF 轉換到 Frequency Domain

MTF

正規化頻率響應，取得 MTF50 / MTF10

Slanted-edge 之所以使用傾斜邊緣，是為了利用不同 Scan Lines 的 Subpixel Sampling Phase，更穩定地估計 Edge Response。

ISO 12233 是影像解析度與 Spatial Frequency Response 測試的重要標準。Imatest 的文件也詳細說明了 Slanted-edge、ESF、LSF 及 MTF 的計算關係。

![](https://www.google.com/s2/favicons?domain=https://www.imatest.com&sz=32)

Imatest

+1

## 5.6 頻率響應（Frequency Response）為什麼重要？

單純的 Sharpness Score 只能描述一個數字。

MTF Curve 則讓你知道影像系統在不同尺度上保留多少細節。

例如兩種相機鏡頭：

|項目|Lens A|Lens B|
|---|---|---|
|MTF50|0.30 cycles/pixel|0.18 cycles/pixel|
|MTF at 0.4 cycles/pixel|0.18|0.04|
|細微文字辨識的潛在表現|較有利|較不利|

以上為假設數據，且前提是兩台相機使用相同的 Sampling Scale 與可比較的處理鏈。

Lens A 在較高 Spatial Frequency 能保留更多 Contrast，因此可能更適合辨識微小文字。

但最終效果仍須結合 Noise、Pixel Scale、Training Data 與模型測試。

## 5.7 Senior／Staff 層級：如何建立完整 Sharpness 評估？

我不會只使用一個 Sharpness Metric。

而會建立兩層不同目標的品質驗證：

|目的|推薦指標|
|---|---|
|即時 Autofocus|Tenengrad、Laplacian Variance|
|Camera／Lens Characterization|MTF50、MTF10、MTF Curve|
|Image Noise 評估|SNR、Temporal Noise、Noise Power|
|是否飽和或曝光不足|Saturation Ratio、Exposure Statistics|
|Downstream OCR|Character Accuracy、CER|
|Downstream Segmentation|Dice、IoU|
|Production Stability|Repeatability、Failure Rate|

舉例：

一套機台拍攝很細的錶盤文字。

某次調整 Unsharp Mask 後，Laplacian Variance 增加 40%。

這看似 Sharpness 改善。

但如果 OCR Character Error Rate 反而增加，說明 Sharpening 可能產生 Halo 或 False Edges。

因此我會把物理 Sharpness、影像處理效果與最終 Task Performance 分開評估。

這是一個非常重要的 Senior Engineer 觀點：

增加 Sharpness Score 不一定代表系統真正獲得更多有效資訊。

# Q6. RGB、HSV、Lab 各適合什麼影像分析？

## 6.1 面試核心回答

> RGB、HSV 和 CIELAB 都是 Color Representations，但它們使用不同方式描述顏色。
> 
> RGB 是以 Red、Green、Blue 三個通道表示顏色，是數位相機與影像處理最常見的表示方式。
> 
> HSV 將顏色轉換成 Hue、Saturation、Value，在某些 Color Segmentation 任務中特別方便，例如根據 Hue 範圍偵測特定顏色的物體。
> 
> CIELAB 將 Lightness 與 Chromatic Components 分開，通常更適合 Color Difference Measurement、Color Quality Control 和 Perceptual Color Comparison。
> 
> 但我會特別注意 Color Calibration、White Balance、Illumination、Gamma 及 Device Color Space，因為同一個物體在不同拍攝條件下，RGB、HSV 和 Lab 數值都可能改變。

## 6.2 什麼是 Color Representation？

Color Representation（色彩表示）就是使用某種數學座標來描述顏色。

相同的顏色，可以在不同 Color Space 中表示成不同數值。

例如一種標準的純紅色：

|Color Space|數值示例|
|---|---|
|sRGB|(255, 0, 0)|
|HSV|(0°, 100%, 100%)|
|CIELAB（D65）|約 (53.2, 80.1, 67.2)|

這些數值是在指定 Color Space 和 Reference White 的條件下對應同一種顏色，不能忽略其 Color Space 定義。

RGB

Red / Green / Blue

HSV

Hue / Saturation / Value

Lab

L* / a* / b*

概念示意：RGB 表示通道、HSV 表示色相與飽和度、Lab 將明度與色度分開；不是完整的色域座標圖。

## 6.3 RGB：什麼時候使用？

RGB 的三個主要成分：

\[ C=(R,G,B) \]

例如：

\[ (255,0,0)=\text{Red} \]

\[ (0,255,0)=\text{Green} \]

\[ (0,0,255)=\text{Blue} \]

### RGB 的優點

RGB 最接近多數彩色 Camera 的輸出與電腦顯示、影像處理工作流程。

適合：

- Camera Image Processing。
    
- CNN／Deep Learning Model Inputs。
    
- Color Correction Matrix。
    
- Channel Statistics。
    
- 特定通道的 Defect Analysis。
    
- 原始與校正後色彩訊號比較。
    

但有一個經常被忽略的問題：

RGB 不一定是 Linear RGB。

一般 sRGB 影像經過非線性的 Transfer Function 編碼，因此 sRGB 的 128 並不等於 sRGB 255 對應線性光強度的一半。

如果要進行與 Radiance、Exposure 或物理反射相關的運算，應先了解 Camera Response 與 Linearization。

### RGB 的限制

RGB 三個通道共同包含亮度與色彩資訊。

假設同一塊藍色錶盤：

在強光下測到：

\[ RGB_1=(40,80,180) \]

在較暗光線下：

\[ RGB_2=(20,40,90) \]

雖然兩者來自同一個物體，但 Euclidean RGB Distance：

\[ d=\sqrt{(R_1-R_2)^2+(G_1-G_2)^2+(B_1-B_2)^2} \]

仍然可能很大。

這代表 RGB Distance 不能直接當作穩定的 Perceptual Color Difference。

## 6.4 HSV：什麼時候使用？

HSV 包含：

- H = Hue： 色相，例如紅色、黃色、藍色。
    
- S = Saturation： 飽和度。
    
- V = Value： RGB 最大通道所代表的明亮程度。
    

通常：

\[ V=\max(R,G,B) \]

這裡假設 RGB 已正規化到 0–1。

\[ S= \frac{\max(R,G,B)-\min(R,G,B)} {\max(R,G,B)} \]

當最大值為 0 時，通常定義 S = 0。

Hue 由最大通道及通道差異計算。

### HSV 的實際用途

假設某錶款有綠色 Bezel。

如果希望辨識綠色區域，直接使用 RGB Threshold 可能比較不方便，因為綠色在不同亮度下，RGB 數值會變化。

HSV 可以將主要 Color Category 與 Value 分開。

OpenCV：

```
hsv = cv2.cvtColor(    image, cv2.COLOR_BGR2HSV)lower_green = (35, 40, 40)upper_green = (85, 255, 255)mask = cv2.inRange(    hsv,    lower_green,    upper_green)
```

以上 Threshold 只是一個示意設定，需要根據實際 Camera 與光源測試。

注意 OpenCV 對一般 8-bit HSV 的 Hue 通常編碼為 0–179，而不是 0–360。

### HSV 的限制

HSV 並不完全不受光照影響。

它的 Hue 在以下情況可能不穩定：

- Saturation 很低。
    
- 影像接近白色或灰色。
    
- 光源色溫改變。
    
- 有強烈 Specular Reflection。
    
- 相機發生 Color Clipping。
    

例如銀色金屬表面的反光區域接近白色，其 Hue 可能失去明確的物理意義。

因此 HSV 很適合某些 Color Segmentation，但並不是所有 Color Measurement 的最佳選擇。

## 6.5 Lab：什麼時候使用？

CIELAB 由三個成分組成：

\[ (L^*,a^*,b^*) \]

其中：

- \(L^*\)：Perceptual Lightness。
    
- \(a^*\)：綠色到紅色方向。
    
- \(b^*\)：藍色到黃色方向。
    

Lab 的重要特色是把 Lightness 與 Chromatic Components 分開。

但要注意：Lab 的 \(L^*\) 是感知明度，不是絕對物理亮度或 Radiance。

### 為什麼 Lab 常用於 Color Difference？

假設某手錶錶盤的標準顏色：

\[ Lab_{\mathrm{reference}}=(50,10,20) \]

待測錶盤：

\[ Lab_{\mathrm{sample}}=(52,12,23) \]

簡單的 CIE76 Color Distance：

\[ \Delta E_{ab}^* = \sqrt{ (\Delta L^*)^2+ (\Delta a^*)^2+ (\Delta b^*)^2 } \]

計算：

\[ \Delta E_{ab}^* = \sqrt{2^2+2^2+3^2} = \sqrt{17} \approx4.12 \]

這是一個 Color Difference 指標。

但 CIE76 並不是在所有色域都具有完全均勻的 Perceptual Distance。

如果是精準色差品質控制，通常還會考慮 CIEDE2000：

\[ \Delta E_{00} \]

它對 Lightness、Chroma 和 Hue 等因素使用更複雜的修正。

### Lab 特別適合哪些分析？

例如：

Case 1：檢查錶盤是否有顏色異常

用已知 Authentic Reference Dial 建立 Lab Color Distribution，再比較待測區域的 Color Difference。

Case 2：檢查金色錶殼色差

在一致的 Illumination、Camera Calibration 與反光控制條件下，比較 Gold-tone Color Consistency。

Case 3：影像區域分割

某些材料或顏色差異，在 Lab 的 a*、b* 空間中可能比原始 RGB 更容易區分。

## 6.6 RGB、HSV、Lab 的完整比較

|項目|RGB|HSV|CIELAB|
|---|---|---|---|
|核心表示|三色通道|色相、飽和度、明亮程度|明度、色度|
|常見用途|影像處理、Deep Learning|Color Segmentation|Color Difference、品質控制|
|分離亮度|否|部分分離，V 非物理亮度|分離 Perceptual Lightness|
|Perceptual Uniformity|否|否|近似|
|Color Distance|直接 RGB 距離通常不理想|HSV 距離需特別設計|ΔE76、ΔE00|
|對 Illumination 的穩健性|有限|有限|仍需控制與校正|
|工業色彩量測|需要完整校正|多用於分類／分割|常用於可重複的色差比較|

## 6.7 Senior／Staff 層級：如何設計可靠的 Color Analysis Pipeline？

對精密手錶 Authenticity Analysis，我會建立：

Controlled Illumination

RAW Capture + Exposure / Gain Metadata

Black Level / White Balance / Demosaicing

Camera Color Calibration / CCM

Convert to Defined Color Space

ROI Segmentation / Specular Mask

Color Features / ΔE / Distribution

Compare with Reference + Uncertainty

這裡的 White Balance、Demosaicing、CCM 與 Color Transfer Function 必須按照資料的實際編碼及相機處理鏈安排。

不能假設任何 RGB 影像都能直接套用同樣的處理順序。

### 這個案例最重要的三個實驗

Experiment A：Repeatability

同一支錶在相同條件下重複拍攝 20 次。

確認 Lab Mean、Standard Deviation 和 Color Difference 是否穩定。

Experiment B：Lighting Robustness

改變光源強度與角度，分析 Color Features 的變化。

如果微小的照明角度差異就能讓 Authenticity Score 大幅改變，代表 Color Measurement 不夠可靠。

Experiment C：Reference Validation

使用 ColorChecker 或其他已知反射特性的標準色彩目標，建立 Color Accuracy Baseline。

這對具有金屬鏡面反射的手錶尤其重要，因為鏡面反射所測到的顏色包含強烈的光源成分，未必等於材料本身的色彩特性。

面試重點：選擇 Lab 不等於解決了 Color Calibration。色彩空間與可靠的色彩量測，是兩個不同層面的問題。

# Q7. 為什麼更高解析度不一定代表更好的辨識率？

## 7.1 面試核心回答

> Higher Resolution 不一定帶來更高的 Recognition Accuracy，因為 Pixel Count 只是影像的 Sampling Density，不代表相機真正取得了更多有用的 Optical Information。
> 
> 最終的辨識表現取決於 Optical Resolution、Lens MTF、Focus、Motion Blur、Noise、SNR、Illumination、Effective Object Size 及 Training Data Quality。
> 
> 如果 Lens 無法解析足夠細的結構，增加 Sensor Pixels 只會更密集地取樣相同的模糊資訊。
> 
> 更高解析度還可能增加 Computation、Memory、Storage 和 Inference Latency。
> 
> 我會透過 Controlled Experiments，比較不同 Resolution、Optical Setup 與 Model Input Size 下的 Task-level Performance，而不是只比較 Image Dimensions。

## 7.2 Pixel Resolution 與 Optical Resolution

這是面試非常重要的區別。

### Pixel Resolution

例如：

Camera A：

\[ 2048\times2048 \]

約 4.2 Megapixels。

Camera B：

\[ 4512\times4512 \]

約 20.4 Megapixels。

Camera B 的 Pixel 數量約是 Camera A 的 4.85 倍。

但是這不代表它一定可以辨識約五倍細小的物體。

### Optical Resolution

Optical Resolution 取決於：

- Lens Aberration。
    
- Diffraction。
    
- Numerical Aperture。
    
- Optical MTF。
    
- Magnification。
    
- Working Distance。
    
- Focus Accuracy。
    
- Sensor Sampling。
    

如果影像進入 Sensor 以前，已經被鏡頭模糊，高 Pixel Count 不可能憑空恢復所有失去的細節。

## 7.3 用實際數字解釋

假設兩台相機都拍攝寬度為 20 mm 的視野。

Camera A：

\[ 2048\text{ pixels} \]

Camera B：

\[ 4512\text{ pixels} \]

則 Object-space Sampling Scale：

\[ s_A=\frac{20}{2048} \approx9.77\ \mu m/\text{pixel} \]

\[ s_B=\frac{20}{4512} \approx4.43\ \mu m/\text{pixel} \]

假設要辨識一條 20 µm 的微小刮痕：

Camera A：

\[ \frac{20}{9.77}\approx2.05\text{ pixels} \]

Camera B：

\[ \frac{20}{4.43}\approx4.51\text{ pixels} \]

Camera B 可以使用更多 Pixels 描述這條刮痕。

但是這只能說明 Sampling 比較密，並不能保證刮痕更容易辨識。

假設光學系統本身的 Blur Diameter 已經達到 25 µm，兩台相機取得的刮痕訊號都可能受到嚴重模糊。

Camera B 的更多 Pixels 只會更加細緻地取樣模糊的影像。

所以必須同時看 Optical MTF 和 Sampling Scale。

## 7.4 Nyquist Sampling Theory 是什麼？

Nyquist Theory 說明，如果希望正確取樣某個 Spatial Frequency，Sampling Frequency 必須至少為訊號最高頻率的兩倍。

在理想的一維情況下：

\[ f_{\mathrm{sampling}}>2f_{\mathrm{signal}} \]

對空間取樣而言：

\[ f_{\mathrm{Nyquist}}= \frac{1}{2s} \]

其中 \(s\) 是物體空間每 Pixel 的距離。

假設：

\[ s=5\ \mu m/\text{pixel} \]

則：

\[ f_{\mathrm{Nyquist}} = 100\text{ lp/mm} \]

這代表感測器的理論 Nyquist Frequency 是 100 lp/mm。

但並不代表 Lens 在 100 lp/mm 可以保留足夠 Contrast。

如果 Lens 在該 Spatial Frequency 的 MTF 幾乎為 0，更多取樣也無法提供有效的原始光學細節。

此外，Nyquist 是避免頻譜混疊的理論條件，不代表只用兩個 Pixel 就一定可以可靠地辨識、量測複雜的文字或微小缺陷。

對精密 Machine Vision，通常需要更多 Pixels 覆蓋目標特徵，以支援穩健的分類、分割或尺寸量測。

## 7.5 SNR 是什麼？為什麼會影響辨識？

SNR（Signal-to-Noise Ratio）是有效訊號相對於 Noise 的強度。

簡單定義：

\[ SNR=\frac{\text{Signal}}{\text{Noise}} \]

對相機中的電子數，簡化模型為：

\[ SNR= \frac{N_e} {\sqrt{N_e+\sigma_{\mathrm{read}}^2+\sigma_{\mathrm{dark}}^2}} \]

其中：

- \(N_e\)：接收到的有效光電子數。
    
- \(\sigma_{\mathrm{read}}\)：Read Noise。
    
- \(\sigma_{\mathrm{dark}}\)：Dark-related Noise。
    

在 Shot-noise Dominated 條件下：

\[ SNR\approx\sqrt{N_e} \]

因為 Photon Shot Noise 的標準差約隨接收光電子數的平方根增加。

### 為什麼更多 Pixels 可能降低 Per-pixel SNR？

假設相同 Sensor Area 被分成更多、更小的 Pixels。

在相同光照、曝光及量子效率下，單個較小 Pixel 通常接收到更少 Photons。

因此 Per-pixel Shot-noise-limited SNR 可能下降。

例如單個 Pixel 的有效電子數從 1000 降為 250：

\[ SNR_1\approx\sqrt{1000}=31.6 \]

\[ SNR_2\approx\sqrt{250}=15.8 \]

這是 Shot-noise-limited 的簡化例子，不代表所有高解析度感測器都有較差 SNR。

因為實際表現還取決於 Sensor Technology、Pixel Area、Read Noise、Quantum Efficiency、Binning 與 Downsampling。

如果在相同總感測器面積上，把相鄰 Pixels 合併，有時可以恢復或改善目標尺度上的有效 SNR。

所以不能單純認為 Pixel 越小一定越差。

## 7.6 Blur 有哪幾種類型？

Blur 不是只有 Defocus。

|Blur 類型|原因|解決方向|
|---|---|---|
|Defocus Blur|目標不在最佳焦平面|Autofocus、Depth of Field|
|Motion Blur|曝光時 Camera 或物體移動|降低 Exposure、控制 Motion|
|Optical Blur|Lens Aberration、光學品質限制|改善 Lens、Optical Design|
|Diffraction Blur|光學孔徑造成的繞射限制|選擇合適 Aperture／NA|
|Atmospheric Blur|光傳播介質不穩定|控制環境|
|Processing Blur|Denoising、Interpolation、Compression|調整 Processing Pipeline|

### Motion Blur 例子

假設相機拍攝期間物體以：

\[ v=1\text{ mm/s} \]

的相對速度移動，曝光：

\[ t=10\text{ ms} \]

則曝光期間移動距離：

\[ d=vt=10\ \mu m \]

若 Object-space Pixel Scale 是：

\[ s=5\ \mu m/\text{pixel} \]

則 Motion Blur 約跨越：

\[ \frac{10}{5}=2\text{ pixels} \]

這會影響微小刮痕或文字筆畫。

Camera Resolution 再高，如果曝光期間存在 Motion Blur，也無法自動消除這個問題。

## 7.7 Data Quality 為什麼比 Resolution 更重要？

對 Machine Learning，Image Quality 與 Training Data Quality 是兩個不同概念。

影像非常清楚，模型仍可能辨識錯誤。

因為還有：

- Incorrect Labels。
    
- Insufficient Training Data。
    
- Class Imbalance。
    
- Distribution Shift。
    
- Data Leakage。
    
- Missing Edge Cases。
    
- Train／Inference Preprocessing Mismatch。
    

例如 Authenticity Detection：

Dataset A：

20,000 張高解析度影像，但其中部分 Forgery 被誤標成 Original，而且不同類別的拍攝條件不一致。

Dataset B：

10,000 張解析度較低，但 Label 準確、拍攝條件一致，而且各類別涵蓋充分。

Dataset B 完全可能訓練出更好的模型。

Data Quality 通常包含：Label Correctness、Coverage、Consistency、Diversity 與 Representativeness，而不只是 Image Sharpness。

## 7.8 Higher Resolution 的計算成本

假設 Model 的 Input：

Input A

## 1024 × 1024

約 1.05M Pixels

Input B

## 2048 × 2048

約 4.19M Pixels

影像的長與寬各增加兩倍，Pixel Count 增加四倍。

許多 Full-resolution Image Processing Operations 的計算量與 Memory Usage 會顯著增加；對某些 Attention-based Architecture，成本甚至可能呈現更不利的尺度增長。

因此你可能面臨：

- 更高 GPU Memory。
    
- 更慢的 Inference。
    
- 更大的 Storage。
    
- 更長的 Data Transfer Time。
    
- 更高的 Model Serving Cost。
    

所以最佳解析度是 在辨識品質與系統成本之間，能滿足需求的解析度，不一定是硬體能輸出的最大解析度。

## 7.9 Senior／Staff 層級：如何設計 Resolution Experiment？

如果要判斷應該使用 2048 還是 4512 Resolution，我會設計 Controlled Ablation Study。

### Experiment 1：固定 Optical Setup

保持：

- Same Camera and Lens。
    
- Same Field of View。
    
- Same Lighting。
    
- Same Exposure。
    
- Same Focus。
    
- Same Data Samples。
    

使用同一張高解析度原始影像，產生不同 Downsampled Resolution，測試解析度變化本身對 Model Performance 的影響。

### Experiment 2：比較 Optical Setup

如果要比較兩台不同解析度的相機，還要分別量測：

- Optical MTF。
    
- SNR。
    
- Dynamic Range。
    
- Distortion。
    
- Depth of Field。
    
- Object-space Pixel Scale。
    

避免把 Lens 和 Sensor 差異錯誤歸因於 Resolution。

### Experiment 3：Task-level Evaluation

例如任務是手錶 Dial Text Defect Detection：

|指標|2048 Model|4512 Model|
|---|---|---|
|Recall|94%|96%|
|Precision|96%|95%|
|P95 Inference Latency|70 ms|280 ms|
|Relative Input Pixel Count|1×|4.85×|
|Model Memory|較低|較高|

以上為示意結果，不是實際測量數據。

假設系統要求：

\[ Recall\ge95\% \]

且：

\[ P95\ Latency\le150\text{ ms} \]

那麼：

- 2048 Model 沒達到 Recall Requirement。
    
- 4512 Model 沒達到 Latency Requirement。
    

這表示不應只是選擇其中一個，而需要考慮新的 Architecture。

例如：

Low-resolution Whole-image Detection + High-resolution ROI Inspection

第一階段使用低解析度影像定位目標，第二階段只對相關 ROI 進行高解析度分析。

這種 Multi-scale 或 Hierarchical Inference Pipeline，可以在保留微小特徵的同時降低不必要的計算成本。

另一個選項是 Tiling，但必須處理 Tile Boundary、Context Loss 和 Detection Deduplication。

面試的高階結論：

Resolution Selection 是一個 End-to-End System Optimization Problem，不只是 Camera Specification 的比較。

# Q8. 如何把這七個概念整合成完整的 Production Imaging System？

這並不是你原本列出的第八題，而是建議在 Senior／Staff 面試時，用來展示 System Design 能力的加分延伸。

假設面試官提出：

> Design an imaging pipeline for inspecting fine details on luxury watches. It must handle reflective surfaces, autofocus, geometric measurements, color inspection, image stitching, and AI-based defect detection.

我會將這七個技術概念整合成以下系統。

## 8.1 End-to-End Imaging Pipeline

01

1. Hardware Acquisition

Camera + Lens + Lighting + Motorized Stages

02

2. Camera Calibration

Intrinsic + Distortion + Scale + Extrinsic

03

3. Autofocus

Sensor-assisted AF + Tenengrad / Laplacian

04

4. Exposure Bracketing

RAW Single Images + Multi-exposure Capture

05

5. HDR / Exposure Fusion

Alignment + Weighting + Deghosting

06

6. Multi-view Stitching

Feature Matching + RANSAC + Warping + Blending

07

7. Color & Quality Analysis

RGB / HSV / Lab + MTF / SNR / Sharpness

08

8. AI Inference

Detection + Segmentation + OCR + Authentication

09

9. Quality Gate

Confidence + Image Quality + Fail / Retry

10

10. Storage & Monitoring

Raw / Derived Assets + Metadata + Metrics

這張圖表示主要的資料依賴關係，不一定是所有處理都必須依序執行。

例如 Single Exposure、HDR 和 Stitched Image 可以是不同的平行影像分支。當畫面被用於定量色彩或幾何量測時，必須選用合適的 Raw 或校正後資料，不能無條件使用經視覺增強的融合影像。

## 8.2 Production Failure Modes

真正的 Senior Engineer 面試官經常追問：

What happens when your algorithm fails in production?

可以從以下角度回答：

|Pipeline Stage|Failure Mode|Detection / Mitigation|
|---|---|---|
|Acquisition|Camera frame missing|Frame count、Timestamp、Retry|
|Autofocus|False focus peak|Peak Confidence、Recapture、Fallback|
|HDR|Ghosting|Motion Mask、Single-exposure Fallback|
|Camera Calibration|Distortion Parameters Drift|Periodic Calibration、Reference Target|
|Stitching|Wrong Feature Matches|Inlier Distribution、Stage Geometry Prior|
|Stitching|Seam Artifacts|Overlap Quality Check、Semantic Validation|
|Color Analysis|Illumination Shift|Reference Color Patch、Lighting Monitoring|
|Sharpness|Noise inflates score|SNR Check、MTF／Task-level Validation|
|AI Inference|Distribution Shift|Data Drift、Uncertainty、Model Evaluation|

例如 Autofocus 失敗，不應該讓不清楚的影像直接進入 Authentication Model，再把模型的低信心誤認為產品缺陷。

應明確區分：

\[ \text{Image Acquisition Failure} \]

與：

\[ \text{Actual Object Defect} \]

這對於任何工業 AI 系統都是重要的設計原則。

## 8.3 如何建立 Ground Truth？

Production 系統不能只靠演算法分數自我驗證。

應該為不同階段建立獨立 Ground Truth。

|系統|Ground Truth|Evaluation Metric|
|---|---|---|
|Autofocus|人工精細調焦或標準對焦 Target|Focus Error（µm）、Success Rate|
|HDR|參考曝光／Radiometric Target|Recoverable Detail、Ghosting Rate|
|Distortion|已知精度的 Calibration Target|Reprojection RMSE、Measurement Error|
|Stitching|Ground Truth Correspondences／Stage Geometry|Registration Error、Seam Error|
|Sharpness|標準 Slanted-edge Target|MTF50、MTF Curve|
|Color|Calibration Color Target|ΔE00、Repeatability|
|AI Detection|專家標註與可驗證缺陷樣本|Precision、Recall、F1、False Negative Rate|

例如 Autofocus 系統可以用 Independent Ground Truth Focus Position 評估：

\[ e_z= |z_{\mathrm{AF}}-z_{\mathrm{GT}}| \]

再定義：

\[ Success = \mathbf{1} \left( e_z < \delta \right) \]

其中 \(\delta\) 應由物鏡 Depth of Field 與實際下游任務需求決定。

如果某個 Autofocus Algorithm 的平均誤差很小，但 5% 的 Case 偶爾發生非常大的 Focus Error，Production 仍可能不可靠。

因此除了 Mean Error，我也會看：

- P95／P99 Focus Error。
    
- Failure Rate。
    
- Retry Rate。
    
- Average and P95 Autofocus Time。
    
- Downstream Image Acceptance Rate。
    

# 面試前的七題快速複習表

以下是面試時最應該記住的核心概念。

|題目|最重要原理|必須提到的限制|Senior 層級加分點|
|---|---|---|---|
|Autofocus|Laplacian 二階導數；Tenengrad Sobel Gradient|Noise、ROI、False Peak|Coarse-to-Fine、Backlash、Confidence|
|HDR|Multi-exposure、Alignment、Weighted Fusion|Saturation、Ghosting、Color Fidelity|Radiance HDR vs Mertens、Raw Preservation|
|Distortion|Intrinsic Matrix + Radial／Tangential Model|Calibration Drift、Model Mismatch|Reprojection + Independent Geometric Accuracy|
|Stitching|Matching → RANSAC → Homography → Blending|Wrong Matches、Parallax、Seams|Stage Prior、Global Optimization|
|Sharpness|Laplacian Variance、MTF／Frequency Response|Noise、Sharpening、Content Dependence|MTF50 + Task-level Quality|
|Color|RGB、HSV、CIELAB、Color Distance|Lighting、Gamma、Specular Reflection|CCM、ΔE00、Color Calibration|
|Resolution|Optical MTF、Nyquist、SNR|Blur、Data Quality、Compute Cost|Controlled Ablation、Multi-scale Inference|

## 美國 Senior／Staff 面試的最後建議

這七題的回答，可以依照同一套順序組織：

Principle → Mathematics → Algorithm → Trade-offs → Failure Modes → Validation → Production Design

Junior Engineer 通常能解釋一個演算法怎麼運作。

Senior Engineer 應該能說明為什麼選擇它、什麼時候會失敗、怎麼驗證，以及如何部署成可靠的系統。

Staff Engineer 則應進一步說明如何跨 Camera Hardware、Motion Control、Imaging Software、AI Models 和 Quality Engineering 制定整體的技術架構、介面契約、驗收標準與長期維護策略。

這七題最值得優先深入準備的是 Q1 Autofocus、Q3 Camera Calibration、Q4 Stitching，以及 Q7 Resolution vs Recognition Accuracy。 因為這四題最容易從基礎影像數學延伸到 Hardware、Software、Optics、AI 和完整 Production System Design，能充分展示你的跨領域工程能力。