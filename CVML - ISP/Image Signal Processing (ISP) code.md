
詳細解釋相機影像訊號處理（Image Signal Processing, ISP）的完整流程。我會說明每一步驟的目的、主流方法，並提供對應的 Python /OpenCV 範例程式碼，讓您能更清晰地理解。

### 什麼是 Camera ISP？

相機的感光元件（Sensor）捕捉到的原始（RAW）數據並不是我們平常看到的照片。它包含了許多雜訊、顏色偏差和感光元件本身的物理缺陷。**ISP 的核心任務就是將感光元件輸出的原始 RAW 數據，經過一系列的演算法處理，轉換成一張人眼觀看舒適、色彩準確、畫質清晰的影像（如 JPEG、PNG 格式）。**

ISP 可以是相機內部的一顆專用晶片（硬體 ISP），也可以在電腦上用軟體來實現（例如 Adobe Lightroom 或手機上的計算攝影演算法）。

### ISP 處理流程概覽

一個典型（且簡化）的 ISP 流程如下圖所示。請注意，不同廠商的 ISP 流程順序和具體步驟可能略有不同，但核心模組是相似的。

**RAW Data → 黑電平校正 → 鏡頭陰影校正 → 去馬賽克 → 白平衡 → 顏色校正 → Gamma 校正 → 雜訊抑制 → 邊緣增強 → 顏色空間轉換 → 影像壓縮**

---

### 各步驟詳解與程式碼範例

在開始之前，請先確保您已安裝必要的 Python 函式庫：

Bash

```
pip install numpy opencv-python matplotlib
```

我們將使用一個模擬的 RAW 影像數據（通常是單通道的 Bayer 格式影像）來進行展示。

#### 0. 原始 RAW 數據 (RAW Data)

感光元件上覆蓋著一片**拜耳濾色鏡 (Bayer Filter Array)**，它只允許特定顏色的光通過。最常見的排列是 RGGB。這意味著每個像素只記錄 R、G、B 其中一種顏色資訊。因此，RAW 影像是一個單通道的灰階影像，但每個像素的值代表了不同顏色的光的強度。

Python

```
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 假設我們有一個 12-bit 的 RAW 影像 (值域 0-4095)
# 這裡我們用一個已有的彩色圖來模擬產生 RAW data
# 實際的 RAW 檔案需要專門的函式庫 (如 rawpy) 讀取
color_image = cv2.imread('some_image.jpg') # 請替換成你的圖片路徑
color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB) / 255.0

# 模擬 Bayer RGGB 格式的 RAW 影像
height, width, _ = color_image.shape
raw_image = np.zeros((height, width), dtype=np.float32)

# R pixels
raw_image[0::2, 0::2] = color_image[0::2, 0::2, 0]
# G pixels (top-right)
raw_image[0::2, 1::2] = color_image[0::2, 1::2, 1]
# G pixels (bottom-left)
raw_image[1::2, 0::2] = color_image[1::2, 0::2, 1]
# B pixels
raw_image[1::2, 1::2] = color_image[1::2, 1::2, 2]

# 轉換回 12-bit 整數範圍
raw_image_12bit = (raw_image * 4095).astype(np.uint16)

print("RAW 影像的維度:", raw_image_12bit.shape)
print("RAW 影像的數據類型:", raw_image_12bit.dtype)
```

#### 1. 黑電平校正 (Black Level Correction, BLC)

- **目的**：感光元件即使在完全沒有光線的情況下，由於暗電流（dark current）的存在，輸出的像素值也不會是 0。這個最低的數值就是黑電平（Black Level）。此步驟的目的就是從所有像素中減去這個基礎值，讓黑色真正歸零。
    
- **主流方法**：直接減去一個固定的值。這個值通常由相機製造商在校準時測得，或寫在 RAW 檔案的元數據 (Metadata) 中。
    
- **Example Code**:
    

Python

```
def black_level_correction(raw_img, black_level):
    # 確保數據類型可以處理負數
    corrected_img = raw_img.astype(np.int32) - black_level
    # 裁切掉小於 0 的值
    corrected_img = np.maximum(corrected_img, 0)
    return corrected_img.astype(raw_img.dtype)

# 假設黑電平值為 64 (在 12-bit 影像中)
black_level = 64
blc_output = black_level_correction(raw_image_12bit, black_level)
```

#### 2. 鏡頭陰影校正 (Lens Shading Correction, LSC)

- **目的**：由於鏡頭的物理特性，影像的中心通常比四周的角落更亮，這種現象稱為「暗角」或「暈影」(Vignetting)。LSC 的目的就是補償這種亮度不均，讓整個畫面的亮度更均勻。
    
- **主流方法**：生成一個增益圖（Gain Map）。這個圖的中心值接近 1，而四周的值大於 1。將原始影像與這個增益圖相乘，即可提亮角落。增益圖通常是透過拍攝一張均勻的白板或灰板來校準生成的。
    
- **Example Code** (使用一個簡化的、程式產生的增益圖):
    

Python

```
def lens_shading_correction(img):
    height, width = img.shape
    # 創建一個徑向的增益圖 (Gain Map) 來模擬 LSC
    center_x, center_y = width / 2, height / 2
    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    
    # 計算每個像素到中心的距離
    distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    
    # 增益從中心 1.0 增加到角落 1.4 (可調參數)
    gain = 1.0 + 0.4 * (distance / max_dist)
    
    corrected_img = img * gain
    # 避免溢出，裁切到原始數據最大值 (4095-64)
    corrected_img = np.clip(corrected_img, 0, 4095 - black_level) 
    return corrected_img.astype(img.dtype)

lsc_output = lens_shading_correction(blc_output)
```

#### 3. 去馬賽克 (Demosaicing / Debayering)

- **目的**：這是 ISP 中最關鍵的步驟之一。它將只有單一顏色資訊的 Bayer 格式影像，還原成每個像素都擁有 R、G、B 三個顏色通道的標準彩色影像。
    
- **主流方法**：
    
    - **鄰近插值**：簡單，但效果差，會產生鋸齒。
        
    - **雙線性插值 (Bilinear Interpolation)**：最常見的基礎方法。一個像素缺失的顏色值，由其周圍同色像素的平均值來計算。例如，一個 R 像素點的 G 值，由其上下左右四個 G 像素的平均值得到。
        
    - **高階演算法**：如 Malvar-He-Cutler (MHC)、AHD (Adaptive Homogeneity-Directed) 等，它們會考慮邊緣方向，以獲得更好的細節並減少偽色 (Color Artifacts)。
        
- **Example Code** (使用 OpenCV 內建的雙線性插值):
    

Python

```
# OpenCV 的 demosaicing 函數需要 8-bit 或 16-bit 輸入
# 我們先將數據正規化到 16-bit 範圍
demosaic_input = (lsc_output / (4095 - black_level) * 65535).astype(np.uint16)

# cv2.COLOR_BayerBG2RGB 表示我們的 Bayer 格式是 BGGR
# 根據你的感光元件，可能是 COLOR_BayerRG2RGB, COLOR_BayerGB2RGB 等
# 這裡我們假設是 RGGB
demosaiced_img_bgr = cv2.cvtColor(demosaic_input, cv2.COLOR_BayerRG2RGB)

# 轉換為 RGB 以便顯示
demosaiced_img_rgb = cv2.cvtColor(demosaiced_img_bgr, cv2.COLOR_BGR2RGB)

# 正規化到 0-1 浮點數，方便後續處理
demosaiced_img_float = demosaiced_img_rgb / 65535.0
```

#### 4. 白平衡 (White Balance, WB)

- **目的**：不同光源（如日光、燈泡、螢光燈）的色溫不同，會導致影像偏色（例如在燈泡下拍攝會偏黃）。白平衡的目的就是校正這種偏色，讓影像中的白色物體在最終照片裡也呈現為白色。
    
- **主流方法**：
    
    - **灰世界演算法 (Gray World)**：假設整個場景的平均顏色是中性灰。計算 R、G、B 三個通道的平均值，然後對 R 和 B 通道進行縮放，使其平均值與 G 通道相等。
        
    - **完美反射演算法 (White Patch / Max-RGB)**：假設影像中最亮的點是白色或高光，並以此為基準來縮放 R 和 B 通道。
        
    - **基於機器學習的方法**：現代手機常用，能更準確地識別場景和光源。
        
- **Example Code** (使用灰世界演算法):
    

Python

```
def gray_world_white_balance(img_float):
    # 計算 R, G, B 通道的平均值
    avg_r = np.mean(img_float[:, :, 0])
    avg_g = np.mean(img_float[:, :, 1])
    avg_b = np.mean(img_float[:, :, 2])
    
    # 以 G 通道為基準，計算 R 和 B 通道的增益
    gain_r = avg_g / avg_r
    gain_b = avg_g / avg_b
    
    corrected_img = img_float.copy()
    corrected_img[:, :, 0] *= gain_r  # R channel
    corrected_img[:, :, 1] *= 1.0      # G channel
    corrected_img[:, :, 2] *= gain_b  # B channel
    
    # 裁切到 [0, 1] 範圍
    return np.clip(corrected_img, 0, 1)

wb_output = gray_world_white_balance(demosaiced_img_float)
```

#### 5. 顏色校正矩陣 (Color Correction Matrix, CCM)

- **目的**：感光元件的濾色片光譜響應與人眼的標準響應（CIE XYZ 色彩空間）存在差異。為了讓顏色看起來更真實、更討喜，需要透過一個 3x3 的矩陣將影像從感光元件的 RGB 空間轉換到一個標準的色彩空間（如 sRGB）。
    
- **主流方法**：`[R_out, G_out, B_out]^T = CCM * [R_in, G_in, B_in]^T`。這個 CCM 矩陣由廠商在嚴格的光照條件下，拍攝標準色卡（Color Checker）後計算得出。
    
- **Example Code**:
    

Python

```
def apply_ccm(img_float, ccm):
    # 將影像從 (H, W, 3) 轉換為 (H*W, 3)
    pixels = img_float.reshape(-1, 3)
    # 進行矩陣乘法
    corrected_pixels = np.dot(pixels, ccm.T)
    # 將影像還原為 (H, W, 3)
    corrected_img = corrected_pixels.reshape(img_float.shape)
    return np.clip(corrected_img, 0, 1)

# 一個示例 CCM (通常由相機校準提供)
# 這個矩陣會增強飽和度並微調色相
ccm = np.array([
    [1.2, -0.1, -0.1],
    [-0.1, 1.2, -0.1],
    [-0.1, -0.1, 1.2]
])

ccm_output = apply_ccm(wb_output, ccm)
```

#### 6. Gamma 校正 (Gamma Correction)

- **目的**：人眼對亮度的感知不是線性的，我們對暗部細節的變化比亮部更敏感。而顯示器顯示亮度的方式也是非線性的。Gamma 校正就是應用一個冪函數 Output=Input1/γ，將線性的影像數據轉換為非線性的，使其在螢幕上看起來更自然，同時能更有效地利用數據位元來儲存暗部細節。
    
- **主流方法**：應用冪函數。對於 sRGB 色彩空間，γ 的值通常約為 2.2。
    
- **Example Code**:
    

Python

```
def gamma_correction(img_float, gamma=2.2):
    return np.power(img_float, 1/gamma)

gamma_output = gamma_correction(ccm_output)
```

#### 7. 雜訊抑制 (Noise Reduction / Denoising)

- **目的**：由於感光元件和電路中的熱噪聲和讀取噪聲，影像中會存在隨機的雜訊點，尤其是在低光照環境下。此步驟旨在去除這些雜訊，同時盡可能地保留影像細節。
    
- **主流方法**：
    
    - **空間濾波**：高斯模糊（簡單但會模糊邊緣）、中值濾波（對椒鹽雜訊有效）、**雙邊濾波 (Bilateral Filter)**（能很好地在降噪的同時保持邊緣）。
        
    - **高階演算法**：如 BM3D (Block-matching and 3D filtering)。
        
    - **深度學習**：基於 CNN 的降噪器在現代手機和相機中越來越普遍。
        
- **Example Code** (使用雙邊濾波):
    

Python

```
# 先將影像轉回 8-bit 以便使用 OpenCV 濾波器
img_8bit = (gamma_output * 255).astype(np.uint8)

# d: 鄰域直徑, sigmaColor: 顏色空間標準差, sigmaSpace: 座標空間標準差
# sigmaColor 值越大，表示顏色相近的像素會被一起平均，降噪效果更強
# sigmaSpace 值越大，表示距離更遠的像素也會被納入計算
denoised_img = cv2.bilateralFilter(img_8bit, d=9, sigmaColor=75, sigmaSpace=75)
```

#### 8. 邊緣增強 (Edge Enhancement / Sharpening)

- **目的**：降噪過程或鏡頭本身可能會使影像變得有些模糊。邊緣增強可以使物體的輪廓和細節看起來更清晰。
    
- **主流方法**：**反銳化遮罩 (Unsharp Masking, USM)**。其原理是：
    
    1. 對原圖進行高斯模糊，得到一張模糊的影像。
        
    2. 用原圖減去模糊影像，得到包含邊緣細節的「遮罩」。
        
    3. 將這個遮罩按一定權重加回到原圖上。
        
- **Example Code**:
    

Python

```
def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

sharpened_output = unsharp_mask(denoised_img, amount=1.2)
```

#### 9. 顏色空間轉換與影像壓縮

- **目的**：這是 ISP 的最後步驟，將處理好的 RGB 影像轉換並壓縮成通用的儲存格式。
    
- **主流方法**：
    
    - **顏色空間轉換**：對於 JPEG 壓縮，通常會將 RGB 轉換為 YCbCr 空間。Y 代表亮度（Luma），Cb 和 Cr 代表色度（Chroma）。人眼對亮度比對色度敏感，因此可以對 Cb 和 Cr 通道進行**色度抽樣 (Chroma Subsampling)**（如 4:2:2 或 4:2:0），在人眼幾乎無法察覺差異的情況下，大幅減少數據量。
        
    - **影像壓縮**：應用如 JPEG（有損壓縮）、PNG（無損壓縮）等演算法。
        
- **Example Code**:
    

Python

```
# 1. 顏色空間轉換 (可選，但 JPEG 內部會做)
# ycrcb_img = cv2.cvtColor(sharpened_output, cv2.COLOR_RGB2YCrCb)

# 2. 儲存為 JPEG 檔案 (壓縮)
# quality 參數從 0 到 100，越高代表品質越好，檔案越大
cv2.imwrite('output_image.jpg', cv2.cvtColor(sharpened_output, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])

print("ISP 流程完成，影像已儲存為 output_image.jpg")

# 顯示最終結果
plt.figure(figsize=(10, 10))
plt.imshow(sharpened_output)
plt.title("Final ISP Output")
plt.axis('off')
plt.show()
```

### 總結

以上就是一個完整且詳細的相機 ISP 流程。從感光元件捕捉到的最原始、充滿缺陷的數據開始，經過一步步精密的校正、還原和優化，最終才產出我們在螢幕上看到的美麗照片。每一步都至關重要，而各家廠商的 ISP 技術實力，也正是其相機或手機拍照品質的核心競爭力所在。