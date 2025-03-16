**圖像增強技術（Image Enhancement Techniques）** 是一組用來改善圖像質量的處理方法，通過增強圖像中的特定特徵（如邊緣、對比度）來提高其可視性或質量。這些技術廣泛應用於醫學影像、監控系統、衛星圖像、工業檢測等領域。常見的圖像增強技術包括對比度調整、濾波、直方圖均衡化等。以下是一些常用的圖像增強技術及其 MATLAB 示例代碼。

### 1. **直方圖均衡化（<mark style="background: #FF5582A6;">Histogram Equalization</mark>）**

直方圖均衡化是一種對比度增強技術，它通過調整圖像的灰度分布，將其直方圖均勻化，從而<mark style="background: #BBFABBA6;">提高低對比度圖像的對比度</mark>。

**應用場景**：醫學影像、低光或霧霾環境下拍攝的圖像。

`% 讀取灰度圖像 
img = imread('low_contrast_image.jpg'); 
grayImg = rgb2gray(img);  
eqImg = <mark style="background: #FFB86CA6;">histeq</mark>(grayImg);    % 直方圖均衡化 
`
### 2. **對比度拉伸（<mark style="background: #FF5582A6;">Contrast Stretching</mark>）**

對比度拉伸通過拉伸圖像的灰度範圍，使得暗區變暗、亮區變亮，從而增加對比度。這種方法常用於<mark style="background: #BBFABBA6;">改善圖像中對比度不足的情況</mark>。

**應用場景**：需要強化邊緣或對比度不明顯的圖像。

% 讀取灰度圖像 
img = imread('low_contrast_image.jpg'); 
grayImg = rgb2gray(img);  
stretchedImg = <mark style="background: #FFB86CA6;">imadjust</mark>(grayImg, stretchlim(grayImg), []);   % 對比度拉伸 

### 3. **噪聲去除（<mark style="background: #FF5582A6;">Noise Removal</mark>）**

噪聲去除是圖像增強中的關鍵技術，常見的噪聲包括高斯噪聲、椒鹽噪聲等。去噪技術包括均<mark style="background: #BBFABBA6;">值濾波、中值濾波、雙邊濾波</mark>等。高斯濾波是一種低通濾波器，用於去除高頻噪聲，同時保持圖像中的主要結構。它通過與高斯核進行卷積來平滑圖像。

- **均值濾波**：用於去除隨機噪聲，但會模糊圖像邊緣。
- **中值濾波**：對椒鹽噪聲有良好的去除效果，能保持邊緣細節。
- 高斯濾波（Gaussian Filtering）

**應用場景**：醫學影像、監控圖像、工業檢測圖像中的噪聲去除。

img = imread('noisy_image.jpg');  
denoisedImg = <mark style="background: #FFB86CA6;">medfilt2</mark>(img, [3 3]);  % 應用中值濾波去除椒鹽噪聲 
or
smoothedImg = <mark style="background: #FFB86CA6;">imgaussfilt</mark>(img, 2);  % sigma = 2  % or 應用高斯濾波 


### 4. **邊緣增強（<mark style="background: #FF5582A6;">Edge Enhancement</mark>）**

邊緣增強技術用於突出圖像中的邊緣，使得邊界更加清晰。常用方法包括拉普拉斯濾波、Sobel 邊緣檢測等。

**應用場景**：工業檢測、目標識別、醫學影像。

img = imread('image.jpg'); 
grayImg = rgb2gray(img);  
edgeImg = <mark style="background: #FFB86CA6;">edge</mark>(grayImg, 'Sobel');   % 使用 Sobel 邊緣檢測 `

### 6. **頻域增強（<mark style="background: #FF5582A6;">Frequency Domain Enhancement</mark>）**

頻域增強技術通過將圖像轉換到頻域進行操作（如使用傅里葉變換），可以增強特定的頻率分量，達到去噪或強化邊緣的效果。

**應用場景**：精細結構的增強，如遙感圖像、醫學成像。

img = imread('image.jpg'); 
grayImg = rgb2gray(img);  

% 將圖像轉換到頻域 
F = <mark style="background: #FFB86CA6;">fft2</mark>(double(grayImg)); Fshift = <mark style="background: #FFB86CA6;">fftshift</mark>(F);  

% 構建高通濾波器 
[M, N] = size(grayImg); D0 = 30; % 截止頻率  
[u, v] = meshgrid(1:N, 1:M); D = sqrt((u - N/2).^2 + (v - M/2).^2); H = double(D > D0);  

% 應用濾波器並轉換回空域 
F_filtered = Fshift .* H;  
img_filtered = <mark style="background: #FFB86CA6;">ifft2</mark>(ifftshift(F_filtered));
img_filtered = abs(img_filtered); 


### 7. **非局部平均濾波（Non-Local Means Filter, NLM）**

NLM 濾波基於圖像中相似像素塊的平均來去除噪聲，它能保留圖像的細節，同時有效去除噪聲。

**應用場景**：醫學影像、去噪處理。

**MATLAB 示例代碼**： MATLAB 並未提供內置的 NLM 濾波器，但可以使用 **Image Processing Toolbox** 或其他自定義實現。

### 8. **CLAHE（Contrast Limited Adaptive Histogram Equalization）**

CLAHE 是自適應直方圖均衡化的一種變體，通過限制對比度來避免過度增強局部區域的噪聲。

**應用場景**：X 光圖像、CT 圖像等醫學圖像增強。

img = imread('low_contrast_image.jpg'); 
grayImg = rgb2gray(img);  
% 應用 CLAHE 
claheImg = <mark style="background: #FFB86CA6;">adapthisteq</mark>(grayImg, 'ClipLimit', 0.02); 

### 總結

這些圖像增強技術適用於不同的應用場景，包括醫學圖像分析、工業檢測和遙感圖像處理。每一種技術都有其特定的應用，根據不同的需求選擇合適的增強技術，能顯著改善圖像質量。