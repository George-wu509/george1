
|                                         |                                               |
| --------------------------------------- | --------------------------------------------- |
| [[### 1. 光照 (Illumination)]]            | 圖像的低頻成分，代表緩慢變化的明暗                             |
| [[### 2. 紋理 (Texture)]]<br>   - Std Dev | 像素鄰域內的強度變化程度                                  |
| 3. 紋理 (Texture)<br>   - Entropy         | 像素鄰域內的隨機性或資訊量                                 |
| 4. [[###### 色相 (Hue)]]                  | 顏色的種類（紅/綠/藍等）                                 |
| 5. [[###### 飽和度 (Saturation)]]          | 顏色的純度或鮮豔度 -  ** 3 channel (MAX - MIN) / MAX** |
| 6. [[###### 對比度 (Contrast)]]            | 像素鄰域內的亮度範圍  - **1 channel (MAX - MIN)**       |
| 7. [[###### 頻率 (Frequency)]]            | 局部圖案的頻率響應強度 - **FFT**                         |
| 8. [[###### 相位 (Phase)]]                | 局部圖案的空間位置 - **FFT**                           |
| 9. [[###### 清晰度 (Sharpness)]]           | 邊緣和細節的強度  - **Laplacian**                     |
| 10. [[###### 信噪比 (SNR)]]                | 局部訊號強度（平均值）與雜訊（標準差）的比值                        |
| 11. [[###### Granularity(DoG)]]         | 特定尺度紋理結構的響應（高斯差分）Difference of Gaussians      |
| GLCM 對比度                                | 像素對之間灰度差異的劇烈程度                                |
| GLCM 同質性                                | 像素對之間灰度相似的程度                                  |


### 1. 光照 (Illumination)

- **中文解釋**： 光照特徵圖旨在==估計圖像中每個像素所受到的光照強度==。通常，光照被認為是圖像中的**低頻成分**，它變化緩慢、平滑，構成了物體的基本明暗關係。陰影就是光照值極低的區域。
- **常用計算方法**： 最直接的方法是==透過一個**大型的低通濾波器**對原始圖像進行平滑處理==。濾波後的結果可以近似為光照圖。
    
    - **高斯濾波 (Gaussian Filtering)**：使用一個較大的標準差 (sigma) 的高斯核對圖像進行卷積，可以得到非常平滑的光照估計。
        
    - **型態學操作 (Morphological Operations)**：使用一個大的結構元素對圖像進行「開運算」(`imopen`) 或「閉運算」(`imclose`) 也能有效估計背景光照。
        
- **MATLAB 實作**：
    ```Matlab
    % --- Illumination Feature Map ---
    % Assumes 'img' is the input image
    if size(img, 3) == 3
        gray_img = rgb2gray(img);
    else
        gray_img = img;
    end
    
    % Method: Large-scale Gaussian filtering
    % The sigma value should be large to capture slow-changing illumination.
    sigma = 20; 
    illumination_map = imgaussfilt(gray_img, sigma);
    
    % --- Visualization ---
    % figure;
    % subplot(1,2,1); imshow(img); title('Original Image');
    % subplot(1,2,2); imshow(illumination_map); title('Estimated Illumination Map');
    ```
    

---

### 2. 紋理 (Texture)

- **中文解釋**： 紋理特徵圖描述了==每個像素周圍鄰域的**強度變化劇烈程度**或**空間排佈規律**==。平滑區域的紋理值低，而充滿細節或雜訊的區域紋理值高。
- **常用計算方法**：
    
    - **局部標準差 (Local Standard Deviation)**：計算每個像素鄰域內的像素值標準差。標準差越大，表示該區域紋理越複雜或對比越強烈。
        
    - **局部熵 (Local Entropy)**：計算每個像素鄰域內的熵。熵衡量的是區域內的隨機性或資訊量，是紋理複雜度的一個絕佳指標。
        
    - **Gabor 濾波器響應**：Gabor 濾波器可以偵測特定方向和頻率的紋理，其幅度響應圖本身就是一種強大的紋理特徵圖。
        
- **MATLAB 實作**：
    ```
    % --- Texture Feature Maps ---
    nhood_size = 9; % Define the neighborhood size (e.g., 9x9)
    
    % Method 1: Local Standard Deviation
    texture_std_map = stdfilt(gray_img, ones(nhood_size, nhood_size));
    
    % Method 2: Local Entropy
    texture_entropy_map = entropyfilt(gray_img, ones(nhood_size, nhood_size));
    
    % --- Visualization ---
    % figure;
    % subplot(1,3,1); imshow(img); title('Original');
    % subplot(1,3,2); imshow(texture_std_map, []); title('Texture (Std Dev)');
    % subplot(1,3,3); imshow(texture_entropy_map, []); title('Texture (Entropy)');
    ```
    

---
###### 色相 (Hue)     
###### 飽和度 (Saturation)
### 3. & 4. 色相 (Hue) & 飽和度 (Saturation)

- **中文解釋**： 這兩個特徵來自 HSV (Hue, Saturation, Value) 色彩空間，只能從彩色圖像計算。
    
    - **色相 (Hue)**：代表顏色的「種類」，例如紅色、綠色、藍色。它是一個環形的值（0-360度）。
        
    - **飽和度 (Saturation)**：代表顏色的「純度」或「鮮豔度」。飽和度為 0 時是灰色，飽和度越高顏色越鮮豔。
        
- **常用計算方法**： 將圖像從 RGB 色彩空間轉換到 HSV 色彩空間。轉換後的圖像包含三個通道，分別就是 H、S 和 V。
- **MATLAB 實作**：
    ```Matlab
    % --- Hue and Saturation Feature Maps ---
    % This requires a color image as input
    if size(img, 3) ~= 3
        error('Input image must be a color image (RGB) to extract Hue and Saturation.');
    end
    
    % Convert from RGB to HSV color space
    hsv_image = rgb2hsv(img);
    
    % The first channel is Hue, the second is Saturation
    hue_map = hsv_image(:, :, 1);
    saturation_map = hsv_image(:, :, 2);
    
    % --- Visualization ---
    % figure;
    % subplot(1,3,1); imshow(img); title('Original RGB');
    % subplot(1,3,2); imshow(hue_map); colormap hsv; title('Hue Channel');
    % subplot(1,3,3); imshow(saturation_map); title('Saturation Channel');
    ```

理解 RGB 到 HSV 的轉換以及 HSV 各通道的意義，對於許多進階的影像處理任務（例如顏色分割、光照校正）至關重要。

我將為您詳細解釋這整個過程。

### HSV 色彩空間的直觀解釋

首先，我們要明白為什麼需要 HSV。RGB 色彩模型雖然是螢幕顯示的基礎，但它並不符合人類感知顏色的方式。我們通常不會用「紅色有多少，綠色有多少」來描述顏色，而是用「這是什麼顏色？深還是淺？鮮豔還是暗淡？」來描述。

HSV 正是這樣一個更符合人類直覺的模型，它包含三個通道：

- **H (Hue)**：**色相**。代表顏色的種類，也就是「這是什麼顏色？」。


- **S (Saturation)**：**飽和度**。代表顏色的純度或鮮豔度，也就是「顏色有多濃？」。
	例如純紅色 `[255, 0, 0]`，它的 `MAX=255`, `MIN=0`。`MAX` 和 `MIN` 的差距非常大，計算出的飽和度就很高

- **V (Value)**：**明度**。代表顏色的明亮程度，也就是「有多亮？」。
    例如 `RGB = [200, 50, 30]`，最亮的分量是 `200`，所以它的明度就很高

您可以將 HSV 想像成一個圓錐體：

- **H (色相)** 是沿著圓錐頂部圓周的**角度**（0° 到 360°），代表了彩虹上的所有顏色。
    
- **S (飽和度)** 是從圓錐中心軸到邊緣的**距離**。中心軸上是沒有色彩的灰色，越往邊緣越鮮豔。
    
- **V (明度)** 是圓錐的**高度**。底部尖端是純黑色，越往上越明亮。
    

---

### H, S, V 是如何從 RGB 計算的？

接下來，我們來回答您的核心問題：如何用 RGB 三個通道的數值去解釋 H, S, V。

為了方便計算，我們首先將 `R, G, B` 的值從 `[0, 255]` 的範圍正規化到 `[0, 1]` 的範圍。然後找出這三個值中的最大值 `MAX` 和最小值 `MIN`。

#### 1. V (明度, Value) 的計算

**V = MAX(R, G, B)**

- **解釋**：明度非常簡單，它就是 R, G, B 三個通道中最亮的那一個的值。一個顏色的「明亮程度」取決於它最強的光分量。
    
    - 例如 `RGB = [200, 50, 30]`，最亮的分量是 `200`，所以它的明度就很高。
        

#### 2. S (飽和度, Saturation) 的計算

**S = (MAX - MIN) / MAX** (如果 MAX 不為 0)

- **解釋**：飽和度衡量的是顏色距離灰色的程度。
    
    - 當一個顏色非常「純」時，例如純紅色 `[255, 0, 0]`，它的 `MAX=255`, `MIN=0`。`MAX` 和 `MIN` 的差距非常大，計算出的飽和度就很高（接近 1）。
        
    - 當一個顏色是灰色、白色或黑色時，例如灰色 `[128, 128, 128]`，它的 `R=G=B`，所以 `MAX = MIN`。`MAX - MIN = 0`，計算出的飽和度就是 `0`。
        
    - **簡單來說，飽和度就是衡量 R, G, B 三個值之間的「貧富差距」。差距越大，顏色越鮮豔；差距越小，顏色越趨近於灰色。**
        

#### 3. H (色相, Hue) 的計算

**H 的計算最複雜，它是一個分段函式，取決於 R, G, B 中哪一個是最大值。**

- **解釋**：色相是一個 0° 到 360° 的角度，代表顏色在色輪上的位置。它的計算公式旨 在衡量最強的顏色分量，以及另外兩個分量之間的相對比例，從而確定具體的顏色種類。
    
    - **如果 R 是最大值**：色相就在黃色和品紅色之間。具體公式約為 `H = 60 * (G - B) / (MAX - MIN)`。
        
    - **如果 G 是最大值**：色相就在黃色和青色之間。具體公式約為 `H = 120 + 60 * (B - R) / (MAX - MIN)`。
        
    - **如果 B 是最大值**：色相就在青色和品紅色之間。具體公式約為 `H = 240 + 60 * (R - G) / (MAX - MIN)`。
        
- 您不需要記住這些公式，只需要理解它的核心思想：**透過比較非主要顏色分量的相對大小，來確定顏色在光譜上的精確位置。**
    

---

### HSV 的數值範圍（非常重要！）

您問到 HSV 是否也是 3 個通道，數值範圍是否為 `0-255`。

是的，HSV 也是 3 個通道，但它的數值範圍在不同的軟體和函式庫中**定義不同**，這是一個常見的混淆點。

|通道|理論/數學範圍|**OpenCV 中的 `uint8` 範圍**|
|---|---|---|
|**H (色相)**|`[0, 360)` (度)|`[0, 179]`|
|**S (飽和度)**|`[0, 1]` (百分比)|`[0, 255]`|
|**V (明度)**|`[0, 1]` (百分比)|`[0, 255]`|

匯出到試算表

**為什麼 OpenCV 中的 H 是 `[0, 179]`？** 因為一個 8-bit 的整數 (`uint8`) 只能儲存 0 到 255 的值。360 這個數字無法存入。為了將 360 度的完整色輪資訊存入 `uint8`，OpenCV 選擇將原始的 Hue 角度**除以 2**（`360 / 2 = 180`），所以範圍就變成了 `[0, 179]`。在處理時需要特別注意這一點。

---

### 解讀 Hue (色相) - 您的問題回答

> Hue 的數值是代表他比較傾向哪個顏色嗎？所以是某個Hue值代表正紅, 某個Hue值代表正藍, 然後紫色應該就在這兩個中間?

**您說的完全正確！** Hue 值就是顏色在色輪上的位置。

以 OpenCV 的 `[0, 179]` 範圍為例：

- **紅色 (Red)**：在 `0` 附近 (以及 `179` 附近，因為它是一個環)。
    
- **黃色 (Yellow)**：在 `30` 附近 (`60/2`)。
    
- **綠色 (Green)**：在 `60` 附近 (`120/2`)。
    
- **青色 (Cyan)**：在 `90` 附近 (`180/2`)。
    
- **藍色 (Blue)**：在 `120` 附近 (`240/2`)。
    
- **品紅/紫色 (Magenta/Purple)**：在 `150` 附近 (`300/2`)。
    

所以，紫色 (品紅) 的 Hue 值 `~150`，正好就在藍色 (`~120`) 和紅色 (環繞回 `0` 或 `180`) 之間。

---

### 高飽和度 (High Saturation) 的 RGB 特徵

> 怎樣的 RGB 數值代表高 Saturation?

根據飽和度的計算公式 `S = (MAX - MIN) / MAX`，要使 `S` 的值很高（接近 1），就需要 `MAX - MIN` 的值非常接近 `MAX`。這意味著 `MIN` 的值必須非常接近 `0`。

因此，一個顏色具有高飽和度的 RGB 特徵是： **至少一個通道的值非常高（接近 255），同時至少一個通道的值非常低（接近 0）。**

**範例**：

- **高飽和度**：
    
    - 純紅色 `[255, 0, 0]` (MAX=255, MIN=0, S=1)
        
    - 亮黃色 `[255, 255, 0]` (MAX=255, MIN=0, S=1)
        
    - 天藍色 `[0, 150, 255]` (MAX=255, MIN=0, S=1)
        
- **低飽和度**：
    
    - 白色 `[255, 255, 255]` (MAX=255, MIN=255, S=0)
        
    - 灰色 `[128, 128, 128]` (MAX=128, MIN=128, S=0)
        
    - 粉色 `[255, 180, 180]` (MAX=255, MIN=180, S ≈ 0.29)。雖然很亮，但因為 R,G,B 值差距不大，所以不鮮豔。



---

###### 對比度 (Contrast)
### 5. 對比度 (Contrast)

- **中文解釋**： 局部對比度特徵圖衡量了==每個像素與其周圍鄰域的亮度差異==。高對比度區域通常意味著存在邊緣或劇烈的紋理變化。
- **常用計算方法**：
    
    - **局部範圍 (Local Range)**：計算每個像素鄰域內的 `最大值 - 最小值`。這是對比度最直接的定義。
    - **局部標準差 (Local Standard Deviation)**：`stdfilt` 的結果同樣可以作為局部對比度的優秀指標。 = 也等於是基於std的texture
        
- **MATLAB 實作**：
    ```Matlab
    % --- Contrast Feature Map ---
    nhood_size = 9;
    
    % Method: Local Range
    contrast_map = rangefilt(gray_img, ones(nhood_size, nhood_size));
    
    % Note: The texture_std_map calculated earlier is also a valid contrast map.
    
    % --- Visualization ---
    % figure;
    % subplot(1,2,1); imshow(img); title('Original Image');
    % subplot(1,2,2); imshow(contrast_map); title('Local Contrast Map (Range)');
    ```
    

---

###### 頻率 (Frequency)
###### 相位 (Phase)
### 6. & 7. 頻率 (Frequency) & 相位 (Phase)

- **中文解釋**： 這些是在頻率域中描述圖像的特徵。要在像素級別得到它們，需要使用局部頻率分析方法。

    - **局部頻率 (Local Frequency)**：衡量每個像素鄰域內訊號變化的快慢。高頻對應邊緣和細密紋理，低頻對應平滑區域。
        
    - **局部相位 (Local Phase)**：描述特徵（如邊緣）在鄰域內的位置。
        
- **常用計算方法**：
    - **Gabor 濾波器**：Gabor 濾波器是分析局部頻率和方向的最佳工具之一。`imgaborfilt` 函式可以直接返回每個像素對濾波器的**幅度 (Magnitude)** 和 **相位 (Phase)** 響應。幅度圖可以視為局部頻率強度圖。
        
    - **小波變換 (Wavelet Transform)**：靜態小波變換 (`swt2`) 可以將圖像分解成不同層級的頻率分量，且每個分量都與原圖大小相同。
        
- **MATLAB 實作**：
    ```Matlab
    % --- Local Frequency and Phase Feature Maps ---
    
    % Method: Using a single Gabor filter
    wavelength = 8;
    orientation = 45;
    g = gabor(wavelength, orientation);
    
    % 'imgaborfilt' can return both magnitude and phase
    [magnitude_map, phase_map] = imgaborfilt(gray_img, g);
    
    % magnitude_map represents the local frequency content at the specified
    % orientation and wavelength.
    % phase_map represents the local phase.
    
    % --- Visualization ---
    % figure;
    % subplot(1,3,1); imshow(img); title('Original');
    % subplot(1,3,2); imshow(magnitude_map, []); title('Local Frequency (Gabor Mag)');
    % subplot(1,3,3); imshow(phase_map, []); title('Local Phase (Gabor Phase)');
    ```
    

---

###### 清晰度 (Sharpness)
### 8. 模糊度 (Blurriness) / 清晰度 (Sharpness)

- **中文解釋**： 模糊度特徵圖用來==衡量每個像素鄰域的清晰程度==。邊緣清晰的區域模糊度低（清晰度高），平滑或失焦的區域模糊度高（清晰度低）。
- **常用計算方法**：
    
    - **梯度幅度 (Gradient Magnitude)**：圖像的梯度在邊緣處值很大，在平滑區域值很小。因此，梯度幅度圖可以作為一張「清晰度」圖。
        
    - **拉普拉斯算子的變異數 (Variance of Laplacian)**：先用拉普拉斯濾波器突顯邊緣，然後計算結果的局部變異數。清晰區域的拉普拉斯響應變化劇烈（高變異數），模糊區域則變化平緩（低變異數）。
        
- **MATLAB 實作**：
    ```Matlab
    % --- Sharpness Feature Map ---
    
    % Method: Gradient Magnitude
    % 'imgradient' computes the gradient magnitude and direction.
    [Gmag, ~] = imgradient(gray_img, 'sobel');
    
    % The Gmag is the sharpness map. We can normalize it for visualization.
    sharpness_map = mat2gray(Gmag);
    
    % --- Visualization ---
    % figure;
    % subplot(1,2,1); imshow(img); title('Original Image');
    % subplot(1,2,2); imshow(sharpness_map); title('Sharpness Map (Gradient Mag)');
    ```
    

---

###### 信噪比 (SNR)
### 9. 信噪比 (Signal-to-Noise Ratio, SNR)

- **中文解釋**： 信噪比通常是一個衡量整張圖像品質的**全域指標**。要在像素級別計算它，我們需要重新定義其局部意義。一個常見的定義是：將每個鄰域的**平均值**視為「訊號」，將**標準差**視為「雜訊」。
    
- **常用計算方法**：
    1. 計算圖像的局部平均值圖 (Local Mean Image)。
    2. 計算圖像的局部標準差圖 (Local Standard Deviation Image)。
    3. 將平均值圖除以標準差圖。
        
- **MATLAB 實作**：
    ```Matlab
    % --- Local SNR Feature Map ---
    nhood_size = 9;
    nhood = ones(nhood_size, nhood_size);
    
    % Calculate local mean using a simple averaging filter
    img_double = im2double(gray_img);
    local_mean_map = imfilter(img_double, fspecial('average', nhood_size), 'replicate');
    
    % Calculate local standard deviation
    local_std_map = stdfilt(img_double, nhood);
    
    % Calculate Local SNR = Local Mean / Local Std
    % Add a small epsilon to avoid division by zero
    epsilon = 1e-6;
    local_snr_map = local_mean_map ./ (local_std_map + epsilon);
    
    % --- Visualization ---
    % figure;
    % subplot(1,2,1); imshow(img); title('Original Image');
    % subplot(1,2,2); imshow(local_snr_map, []); title('Local SNR Map');
    ```


###### Granularity(DoG)
### 關於 Granularity (顆粒度) 和 GLCM

這兩者是新的、更進階的紋理分析特徵，之前的函式中**沒有包含**。我將為您新增這兩項功能。

- **Granularity (顆粒度)**：這個特徵用來描述紋理的尺度大小。例如，沙灘的紋理是「細顆粒」的，而一堆鵝卵石的紋理則是「粗顆粒」的。一個常見的計算方法是使用**高斯差分 (Difference of Gaussians, DoG)**，透過比較不同尺度濾波後的圖像差異，來突顯特定大小的紋理結構。
    
- **GLCM (灰度共生矩陣, Gray-Level Co-occurrence Matrix)**：這是一種經典且強大的統計紋理分析方法。它描述了圖像中具有特定空間關係（如相鄰、相隔5個像素等）的像素對的灰度分佈。從 GLCM 中可以提取出多個二階統計特徵，如**對比度 (Contrast)**、**相關性 (Correlation)**、**能量 (Energy)** 和 **同質性 (Homogeneity)**。為了得到像素級的特徵圖，我們需要在每個像素周圍的鄰域內（滑動窗口）計算這些 GLCM 特徵。
    
##### 新增功能後的 MATLAB 程式碼

我已經更新了 `computeAllPixelFeatures` 函式，加入了 **Granularity (DoG)** 和 **GLCM (Contrast, Homogeneity)** 的計算。

**請注意**：像素級的 GLCM 計算需要對每個像素的鄰域進行操作，這在 MATLAB 中**非常耗時**。我已經在程式碼中加入了相關的性能警告。
```Matlab
function result = computeAllPixelFeatures(img, output_folder, options)
    % computeAllPixelFeatures Computes a comprehensive set of pixel-level feature maps.
    %
    % VERSION 2: Added Granularity (DoG) and GLCM features.

    % --- Argument Validation and Default Parameter Setup ---
    arguments
        img (:,:,3) {mustBeNumeric}
        output_folder (1,1) string
        options.IlluminationSigma (1,1) {mustBeNumeric, mustBePositive} = 20
        options.TextureNeighborhoodSize (1,1) {mustBeInteger, mustBePositive, mustBeOdd} = 9
        options.GaborWavelength (1,1) {mustBeNumeric, mustBePositive} = 8
        options.GaborOrientation (1,1) {mustBeNumeric} = 45
        options.SnrNeighborhoodSize (1,1) {mustBeInteger, mustBePositive, mustBeOdd} = 9
        % New parameters for new features
        options.GranularitySigmas (1,2) {mustBeNumeric, mustBePositive} = [2, 4] % Sigmas for Difference of Gaussians
        options.GlcmNeighborhoodSize (1,1) {mustBeInteger, mustBePositive, mustBeOdd} = 7 % For sliding window GLCM
    end

    % --- Initialization ---
    fprintf('Starting feature computation...\n');
    result = struct(); 
    if ~exist(output_folder, 'dir'), mkdir(output_folder); end
    
    gray_img = im2gray(img);
    img_double = im2double(gray_img);

    % (Previous features 1-9 are calculated here as before)
    fprintf('1-9. Computing standard features (Illumination, Texture, Color, etc.)...\n');
    result.Illumination = imgaussfilt(gray_img, options.IlluminationSigma);
    nhood_texture = ones(options.TextureNeighborhoodSize);
    result.texture_std = stdfilt(gray_img, nhood_texture);
    result.texture_entropy = entropyfilt(gray_img, nhood_texture);
    hsv_image = rgb2hsv(img);
    result.Hue = hsv_image(:, :, 1);
    result.Saturation = hsv_image(:, :, 2);
    result.Contrast = rangefilt(gray_img, nhood_texture);
    g = gabor(options.GaborWavelength, options.GaborOrientation);
    [result.Frequency, result.Phase] = imgaborfilt(gray_img, g);
    [Gmag, ~] = imgradient(gray_img, 'sobel');
    result.Sharpness = mat2gray(Gmag);
    nhood_snr = ones(options.SnrNeighborhoodSize);
    local_mean_map = imfilter(img_double, fspecial('average', options.SnrNeighborhoodSize), 'replicate');
    local_std_map = stdfilt(img_double, nhood_snr);
    result.SNR = local_mean_map ./ (local_std_map + 1e-6);

    % --- 10. Granularity (NEW) ---
    fprintf('10. Computing Granularity map (Difference of Gaussians)...\n');
    sigma1 = options.GranularitySigmas(1);
    sigma2 = options.GranularitySigmas(2);
    img_gauss1 = imgaussfilt(img_double, sigma1);
    img_gauss2 = imgaussfilt(img_double, sigma2);
    result.Granularity = mat2gray(img_gauss1 - img_gauss2);
    visualize_and_save(result.Granularity, 'Granularity (DoG)', fullfile(output_folder, '10_granularity.jpg'));

    % --- 11. GLCM Features (NEW) ---
    % WARNING: This is computationally very expensive due to the pixel-wise sliding window.
    fprintf('11. Computing GLCM maps (this may take a long time)...\n');
    
    glcm_nhood_size = options.GlcmNeighborhoodSize;
    [rows, cols] = size(gray_img);
    
    % Initialize maps
    glcm_contrast_map = zeros(rows, cols);
    glcm_homogeneity_map = zeros(rows, cols);
    
    % Pad the image to handle borders
    pad_size = floor(glcm_nhood_size / 2);
    img_padded = padarray(gray_img, [pad_size pad_size], 'replicate');
    
    % Convert to uint8 with 8 gray levels for faster GLCM computation
    img_quantized = uint8(imquantize(img_padded, linspace(0, 255, 8)));

    % Process each pixel
    for r = 1:rows
        for c = 1:cols
            % Extract neighborhood
            row_start = r;
            row_end = r + 2 * pad_size;
            col_start = c;
            col_end = c + 2 * pad_size;
            neighborhood = img_quantized(row_start:row_end, col_start:col_end);
            
            % Compute GLCM. [0 1] means pixel pairs are horizontal neighbors.
            glcm = graycomatrix(neighborhood, 'Offset', [0 1], 'NumLevels', 8, 'GrayLimits', [0 7]);
            
            if ~isempty(glcm)
                % Compute properties from the GLCM
                stats = graycoprops(glcm, {'Contrast', 'Homogeneity'});
                glcm_contrast_map(r, c) = stats.Contrast;
                glcm_homogeneity_map(r, c) = stats.Homogeneity;
            end
        end
    end
    result.glcm_contrast = mat2gray(glcm_contrast_map);
    result.glcm_homogeneity = glcm_homogeneity_map;
    
    visualize_and_save(result.glcm_contrast, 'GLCM Contrast', fullfile(output_folder, '11a_glcm_contrast.jpg'));
    visualize_and_save(result.glcm_homogeneity, 'GLCM Homogeneity', fullfile(output_folder, '11b_glcm_homogeneity.jpg'));
    
    fprintf('All features computed and visualizations saved.\n');
end

% (The helper function 'visualize_and_save' remains the same as before)
function visualize_and_save(feature_map, title_str, file_path, cmap)
    if nargin < 4, cmap = 'parula'; end
    h = figure('Visible', 'off');
    imshow(feature_map, []);
    title(title_str, 'FontSize', 14);
    if strcmp(cmap, 'hsv'), colormap(h, hsv); else, colormap(h, parula); end
    colorbar;
    saveas(h, file_path);
    close(h);
    fprintf('   ...saved ''%s''\n', file_path);
end
```