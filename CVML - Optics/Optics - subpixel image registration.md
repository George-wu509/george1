
Efficient subpixel image registration by cross-correlation
https://www.mathworks.com/matlabcentral/fileexchange/18401-efficient-subpixel-image-registration-by-cross-correlation

Registers two images (2-D rigid translation) within a fraction of a pixel specified by the user. Instead of computing a zero-padded FFT (fast Fourier transform), this code uses selective upsampling by a matrix-multiply DFT (discrete FT) to dramatically reduce computation time and memory without sacrificing accuracy. With this procedure all the image points are used to compute the upsampled cross-correlation in a very small neighborhood around its peak.

### 核心思想：兩階段「先粗後精」的對位策略

這個演算法的精髓在於避免了傳統方法中「暴力」升採樣（upsampling）帶來的高計算量和記憶體消耗。傳統方法為了達到 `1/20` 像素的精度，會將整個影像的傅立葉變換（FFT）補零擴大20倍，然後再做反傅立葉變換，這個過程非常耗時。

此演算法則採用了更聰明的兩階段策略：

1. **粗定位（Coarse Registration）**：先用標準的 FFT 互相關（cross-correlation）方法，快速找到一個整數像素或半個像素精度的初始位移估計。這就像先用低解析度地圖找到目標山峰所在的大致區域。
    
2. **精定位（Fine Registration）**：在第一步找到的峰值附近，利用一個計算成本極低的「矩陣乘法離散傅立葉變換（matrix-multiply DFT）」，只對這個小鄰域進行高倍率的升採樣，從而精確定位峰值，得到次像素級的位移。這就像只派出高解析度無人機在目標山峰周圍的小範圍內飛行，來確定最高點的精確座標。
    

---

### 原理詳解與程式碼對照

我們將基於傅立葉變換的 **相位相關（Phase Correlation）** 原理來逐步解析。

#### 傅立葉位移定理 (Fourier Shift Theorem)

這是整個演算法的理論基石。該定理指出，如果影像 I2​ 是影像 I1​ 經過位移 (Δx,Δy) 後得到的，即 $I2​(x,y)=I1​(x−Δx,y−Δy)$，那麼它們的傅立葉變換 F1​ 和 F2​ 之間存在以下關係：
$$
F2​(u,v)=F1​(u,v)⋅e−i2π(uΔx+vΔy)
$$
其中 (u,v) 是頻率座標。

為了找出位移 (Δx,Δy)，我們可以計算兩者的 **互功率頻譜（cross-power spectrum）**：
$$
C(u,v)=∣F1​(u,v)⋅F2∗​(u,v)∣F1​(u,v)⋅F2∗​(u,v)​=ei2π(uΔx+vΔy)
$$

其中 F2∗​ 是 F2​ 的共軛複數。對 C(u,v) 進行反傅立葉變換（IFFT），會在對應位移 (Δx,Δy) 的位置得到一個清晰的脈衝峰值。

在實際應用中，為了對雜訊更穩健，通常省略分母的正規化步驟，直接計算 F1​⋅F2∗​，然後取 IFFT，峰值位置同樣對應著影像間的位移。

---

### 演算法步驟對照

現在我們來看看 `dftregistration.m` 是如何實現這個過程的。

#### 1. 初始設定與整數像素對位（Coarse Registration - Level 1）

程式碼接收兩個已經過傅立葉變換的影像 `buf1ft` 和 `buf2ft` 作為輸入。

**如果 `usfac == 1` (整數像素精度):**

```Matlab
% Single pixel registration
CC = ifft2(buf1ft.*conj(buf2ft)); % (1)
CCabs = abs(CC);
[row_shift, col_shift] = find(CCabs == max(CCabs(:))); % (2)
```

1. `buf1ft.*conj(buf2ft)`：這是在頻域中計算互相關，完全對應 F1​⋅F2∗​。
2. `ifft2(...)`：將互相關結果從頻域轉回空間域，得到一個「相關性地圖」 `CC`。
3. `find(CCabs == max(CCabs(:)))`：在地圖上尋找亮度最高的點（即相關性最強的點），其座標就代表了兩個影像之間的 **整數像素位移**。

#### 2. 0.5 像素精度對位（Coarse Registration - Level 2）

**如果 `usfac > 1`:**

```Matlab
% Start with usfac == 2
CC = ifft2(FTpad(buf1ft.*conj(buf2ft),[2*nr,2*nc])); % (3)
CCabs = abs(CC);
[row_shift, col_shift] = find(CCabs == max(CCabs(:)),1,'first');
% ...
row_shift = Nr2(row_shift)/2; % (4)
col_shift = Nc2(col_shift)/2;
```

1. 這一步是為了得到一個比整數像素更準確的初始估計。`FTpad` 函式將互功率頻譜 `buf1ft.*conj(buf2ft)` 在頻域進行補零，使其尺寸擴大兩倍。
2. **頻域補零等效於空間域插值**。對擴大兩倍的頻譜做 IFFT，就等於在空間域得到了 2 倍升採樣的相關性地圖。
3. 在這個 2 倍大的地圖上找峰值，然後將座標除以 2，就能得到一個 **0.5 像素精度** 的位移估計。這為下一步的精細搜索提供了更準確的起點。

#### 3. 次像素精定位（Fine Registration using Matrix-Multiply DFT）

**如果 `usfac > 2`:** 這是演算法最核心的部分。

Matlab

```
%%% DFT computation %%%
% Initial shift estimate in upsampled grid
row_shift = round(row_shift*usfac)/usfac; 
col_shift = round(col_shift*usfac)/usfac;     
dftshift = fix(ceil(usfac*1.5)/2); %% Center of output array

% Matrix multiply DFT around the current shift estimate
CC = conj(dftups(buf2ft.*conj(buf1ft),ceil(usfac*1.5),ceil(usfac*1.5),usfac,...
    dftshift-row_shift*usfac,dftshift-col_shift*usfac)); % (5)

% Locate maximum and map back to original pixel grid 
[rloc, cloc] = find(CCabs == max(CCabs(:)),1,'first');
% ...
row_shift = row_shift + rloc/usfac; % (6)
col_shift = col_shift + cloc/usfac;
```

1. **關鍵呼叫 `dftups`**：這裡不再對整個頻譜進行補零。而是直接呼叫 `dftups` 函式，傳入原始的互功率頻譜 (`buf2ft.*conj(buf1ft)`）。
2. `dftups` 的作用是：在給定的偏移量（`dftshift-row_shift*usfac`, ...）附近，計算一個大小僅為 `ceil(usfac*1.5) x ceil(usfac*1.5)` 的小區域的升採樣結果。例如，如果 `usfac=20`，它只計算一個約 `30x30` 大小的網格點上的值，而這個網格代表了原始相關峰周圍一個極小區域（約 1.5x1.5 像素）內部的 `20` 倍升採樣結果。
3. **`dftups` 的內部原理**：

    ```Matlab
    function out=dftups(in,nor,noc,usfac,roff,coff)
    % ...
    % Compute kernels and obtain DFT by matrix products
    kernc=exp((-1i*2*pi/(nc*usfac))*(...));
    kernr=exp((-1i*2*pi/(nr*usfac))*(...));
    out=kernr*in*kernc; % (7)
    return
    ```
    
    它沒有使用 FFT。相反，它直接根據 DFT 的數學定義式來計算。`kernc` 和 `kernr` 是預先計算好的 DFT 變換核（即 e−i2π(…) 項）。`out = kernr * in * kernc` 透過矩陣乘法，高效地計算出輸入頻譜 `in` 在指定位置（由 `roff`, `coff` 決定）和指定解析度（由 `usfac` 決定）下的 DFT 值。因為輸出區域 `(nor, noc)` 非常小，所以這個矩陣乘法的計算量遠小於做一個巨大的 FFT。
4. **疊加位移**：在 `dftups` 輸出的高解析度小地圖上找到新的峰值位置 `(rloc, cloc)`，將其縮放 (`rloc/usfac`) 後，與之前得到的粗估計 `row_shift` 相加，就得到了最終的**高精度次像素位移**。

### 總結

這個演算法的巧妙之處在於，它深刻理解了「頻域補零」與「空間域插值」的等價關係，並意識到我們並不需要完整的、高解析度的相關性地圖，而只需要峰值附近的局部高解析度視圖。它透過以下步驟實現了效率與精度的完美結合：

1. **FFT 互相關**：快速獲得一個整數或半像素精度的粗略位移。
2. **局部 DFT**：鎖定粗略位移的小鄰域。
3. **矩陣乘法**：用極小的計算代價，對這個小鄰域進行高倍率的「虛擬」升採樣，並找到最精確的峰值位置。

這種方法避免了處理巨大陣列的記憶體和算力開銷，使其在需要高精度影像對位的場景中（如粒子圖像測速、天文學、醫學影像分析等）非常高效和實用。



```Matlab
function [output, Greg] = dftregistration(buf1ft,buf2ft,usfac)
% function [output Greg] = dftregistration(buf1ft,buf2ft,usfac);
% Efficient subpixel image registration by crosscorrelation. This code
% gives the same precision as the FFT upsampled cross correlation in a
% small fraction of the computation time and with reduced memory 
% requirements. It obtains an initial estimate of the crosscorrelation peak
% by an FFT and then refines the shift estimation by upsampling the DFT
% only in a small neighborhood of that estimate by means of a 
% matrix-multiply DFT. With this procedure all the image points are used to
% compute the upsampled crosscorrelation.
% Manuel Guizar - Dec 13, 2007
%
% Rewrote all code not authored by either Manuel Guizar or Jim Fienup
% Manuel Guizar - May 13, 2016
%
% Citation for this algorithm:
% Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup, 
% "Efficient subpixel image registration algorithms," Opt. Lett. 33, 
% 156-158 (2008).
%
% Inputs
% buf1ft    Fourier transform of reference image, 
%           DC in (1,1)   [DO NOT FFTSHIFT]
% buf2ft    Fourier transform of image to register, 
%           DC in (1,1) [DO NOT FFTSHIFT]
% usfac     Upsampling factor (integer). Images will be registered to 
%           within 1/usfac of a pixel. For example usfac = 20 means the
%           images will be registered within 1/20 of a pixel. (default = 1)
%
% Outputs
% output =  [error,diffphase,net_row_shift,net_col_shift]
% error     Translation invariant normalized RMS error between f and g
% diffphase     Global phase difference between the two images (should be
%               zero if images are non-negative).
% net_row_shift net_col_shift   Pixel shifts between images
% Greg      (Optional) Fourier transform of registered version of buf2ft,
%           the global phase difference is compensated for.
%
%
% Copyright (c) 2016, Manuel Guizar Sicairos, James R. Fienup, University of Rochester
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are
% met:
% 
%     * Redistributions of source code must retain the above copyright
%       notice, this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above copyright
%       notice, this list of conditions and the following disclaimer in
%       the documentation and/or other materials provided with the distribution
%     * Neither the name of the University of Rochester nor the names
%       of its contributors may be used to endorse or promote products derived
%       from this software without specific prior written permission.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.
if ~exist('usfac','var')
    usfac = 1;
end
[nr,nc]=size(buf2ft);
Nr = ifftshift(-fix(nr/2):ceil(nr/2)-1);
Nc = ifftshift(-fix(nc/2):ceil(nc/2)-1);
if usfac == 0
    % Simple computation of error and phase difference without registration
    CCmax = sum(buf1ft(:).*conj(buf2ft(:)));
    row_shift = 0;
    col_shift = 0;
elseif usfac == 1
    % Single pixel registration
    CC = ifft2(buf1ft.*conj(buf2ft));
    CCabs = abs(CC);
    [row_shift, col_shift] = find(CCabs == max(CCabs(:)));
    CCmax = CC(row_shift,col_shift)*nr*nc;
    % Now change shifts so that they represent relative shifts and not indices
    row_shift = Nr(row_shift);
    col_shift = Nc(col_shift);
elseif usfac > 1
    % Start with usfac == 2
    CC = ifft2(FTpad(buf1ft.*conj(buf2ft),[2*nr,2*nc]));
    CCabs = abs(CC);
    [row_shift, col_shift] = find(CCabs == max(CCabs(:)),1,'first');
    CCmax = CC(row_shift,col_shift)*nr*nc;
    % Now change shifts so that they represent relative shifts and not indices
    Nr2 = ifftshift(-fix(nr):ceil(nr)-1);
    Nc2 = ifftshift(-fix(nc):ceil(nc)-1);
    row_shift = Nr2(row_shift)/2;
    col_shift = Nc2(col_shift)/2;
    % If upsampling > 2, then refine estimate with matrix multiply DFT
    if usfac > 2,
        %%% DFT computation %%%
        % Initial shift estimate in upsampled grid
        row_shift = round(row_shift*usfac)/usfac; 
        col_shift = round(col_shift*usfac)/usfac;     
        dftshift = fix(ceil(usfac*1.5)/2); %% Center of output array at dftshift+1
        % Matrix multiply DFT around the current shift estimate
        CC = conj(dftups(buf2ft.*conj(buf1ft),ceil(usfac*1.5),ceil(usfac*1.5),usfac,...
            dftshift-row_shift*usfac,dftshift-col_shift*usfac));
        % Locate maximum and map back to original pixel grid 
        CCabs = abs(CC);
        [rloc, cloc] = find(CCabs == max(CCabs(:)),1,'first');
        CCmax = CC(rloc,cloc);
        rloc = rloc - dftshift - 1;
        cloc = cloc - dftshift - 1;
        row_shift = row_shift + rloc/usfac;
        col_shift = col_shift + cloc/usfac;    
    end
    % If its only one row or column the shift along that dimension has no
    % effect. Set to zero.
    if nr == 1,
        row_shift = 0;
    end
    if nc == 1,
        col_shift = 0;
    end
    
end  
rg00 = sum(abs(buf1ft(:)).^2);
rf00 = sum(abs(buf2ft(:)).^2);
error = 1.0 - abs(CCmax).^2/(rg00*rf00);
error = sqrt(abs(error));
diffphase = angle(CCmax);
output=[error,diffphase,row_shift,col_shift];
% Compute registered version of buf2ft
if (nargout > 1)&&(usfac > 0),
    [Nc,Nr] = meshgrid(Nc,Nr);
    Greg = buf2ft.*exp(1i*2*pi*(-row_shift*Nr/nr-col_shift*Nc/nc));
    Greg = Greg*exp(1i*diffphase);
elseif (nargout > 1)&&(usfac == 0)
    Greg = buf2ft*exp(1i*diffphase);
end
return
function out=dftups(in,nor,noc,usfac,roff,coff)
% function out=dftups(in,nor,noc,usfac,roff,coff);
% Upsampled DFT by matrix multiplies, can compute an upsampled DFT in just
% a small region.
% usfac         Upsampling factor (default usfac = 1)
% [nor,noc]     Number of pixels in the output upsampled DFT, in
%               units of upsampled pixels (default = size(in))
% roff, coff    Row and column offsets, allow to shift the output array to
%               a region of interest on the DFT (default = 0)
% Recieves DC in upper left corner, image center must be in (1,1) 
% Manuel Guizar - Dec 13, 2007
% Modified from dftus, by J.R. Fienup 7/31/06
% This code is intended to provide the same result as if the following
% operations were performed
%   - Embed the array "in" in an array that is usfac times larger in each
%     dimension. ifftshift to bring the center of the image to (1,1).
%   - Take the FFT of the larger array
%   - Extract an [nor, noc] region of the result. Starting with the 
%     [roff+1 coff+1] element.
% It achieves this result by computing the DFT in the output array without
% the need to zeropad. Much faster and memory efficient than the
% zero-padded FFT approach if [nor noc] are much smaller than [nr*usfac nc*usfac]
[nr,nc]=size(in);
% Set defaults
if exist('roff', 'var')~=1, roff=0;  end
if exist('coff', 'var')~=1, coff=0;  end
if exist('usfac','var')~=1, usfac=1; end
if exist('noc',  'var')~=1, noc=nc;  end
if exist('nor',  'var')~=1, nor=nr;  end
% Compute kernels and obtain DFT by matrix products
kernc=exp((-1i*2*pi/(nc*usfac))*( ifftshift(0:nc-1).' - floor(nc/2) )*( (0:noc-1) - coff ));
kernr=exp((-1i*2*pi/(nr*usfac))*( (0:nor-1).' - roff )*( ifftshift([0:nr-1]) - floor(nr/2)  ));
out=kernr*in*kernc;
return
function [ imFTout ] = FTpad(imFT,outsize)
% imFTout = FTpad(imFT,outsize)
% Pads or crops the Fourier transform to the desired ouput size. Taking 
% care that the zero frequency is put in the correct place for the output
% for subsequent FT or IFT. Can be used for Fourier transform based
% interpolation, i.e. dirichlet kernel interpolation. 
%
%   Inputs
% imFT      - Input complex array with DC in [1,1]
% outsize   - Output size of array [ny nx] 
%
%   Outputs
% imout   - Output complex image with DC in [1,1]
% Manuel Guizar - 2014.06.02
if ~ismatrix(imFT)
    error('Maximum number of array dimensions is 2')
end
Nout = outsize;
Nin = size(imFT);
imFT = fftshift(imFT);
center = floor(size(imFT)/2)+1;
imFTout = zeros(outsize);
centerout = floor(size(imFTout)/2)+1;
% imout(centerout(1)+[1:Nin(1)]-center(1),centerout(2)+[1:Nin(2)]-center(2)) ...
%     = imFT;
cenout_cen = centerout - center;
imFTout(max(cenout_cen(1)+1,1):min(cenout_cen(1)+Nin(1),Nout(1)),max(cenout_cen(2)+1,1):min(cenout_cen(2)+Nin(2),Nout(2))) ...
    = imFT(max(-cenout_cen(1)+1,1):min(-cenout_cen(1)+Nout(1),Nin(1)),max(-cenout_cen(2)+1,1):min(-cenout_cen(2)+Nout(2),Nin(2)));
imFTout = ifftshift(imFTout)*Nout(1)*Nout(2)/(Nin(1)*Nin(2));
return
```



Here is a detailed English explanation of the "Efficient Subpixel Image Registration by Cross-Correlation" algorithm based on the provided overview and MATLAB code.

### **Core Concept: A "Coarse-to-Fine" Strategy** 

The main goal of this algorithm is to find the precise translational shift between two images with an accuracy much better than a single pixel (subpixel).

Instead of using the traditional, "brute-force" method of upsampling the entire images—which is incredibly slow and memory-intensive—this algorithm employs a much smarter and more efficient **two-step, coarse-to-fine strategy**.

Think of it like finding the highest peak of a mountain range:

1. **Coarse Search**: First, you use a low-resolution satellite map to quickly identify the general area of the highest mountain.
    
2. **Fine Search**: Then, instead of re-mapping the entire world in high resolution, you send a high-resolution drone to scan _only_ that specific mountain to pinpoint its exact summit.
    

This algorithm does the exact same thing for image registration, saving enormous amounts of computation.

---

### **The Foundation: Cross-Correlation via FFT**

The algorithm is built upon a fundamental principle in signal processing: the **Fourier Shift Theorem**. This theorem states that a shift in the spatial domain (the image) corresponds to a linear phase change in the frequency domain.

We can find the shift between two images, `image1` and `image2`, by finding the peak of their cross-correlation. The FFT makes this process extremely fast:

1. Compute the 2D FFT of both images, let's call them `F1` and `F2`.
    
2. Calculate the cross-power spectrum: `C = F1 * conj(F2)`, where `conj(F2)` is the complex conjugate of `F2`.
    
3. Compute the inverse FFT of `C`. The result is a correlation map where the location of the brightest pixel corresponds to the integer shift (Δx,Δy) between the two images.
    

This is precisely what the MATLAB code does for integer-pixel registration (`usfac == 1`):

Matlab

```
% Single pixel registration
CC = ifft2(buf1ft .* conj(buf2ft)); % Step 2 & 3
CCabs = abs(CC);
[row_shift, col_shift] = find(CCabs == max(CCabs(:))); % Find the peak
```

---

### **The Efficient Solution: Selective Upsampling with a Matrix-Multiply DFT**

The real innovation of this algorithm is how it achieves subpixel accuracy without the massive overhead of traditional methods.

#### **Step 1: Get a Better Initial Guess (0.5 pixel accuracy)**

For any upsampling factor `usfac > 1`, the code first gets a more refined initial guess. It does this by zero-padding the cross-power spectrum to twice its original size using the `FTpad` function.

Matlab

```
% Start with usfac == 2
CC = ifft2(FTpad(buf1ft.*conj(buf2ft), [2*nr, 2*nc]));
```

In Fourier theory, **zero-padding in the frequency domain is equivalent to ideal interpolation (upsampling) in the spatial domain**. By doubling the size of the Fourier data, we get a 2x upsampled correlation map. Finding the peak here gives us an initial shift estimate with **0.5 pixel accuracy**.

#### **Step 2: The Key Innovation – Localized High-Resolution Refinement**

This is the "drone survey" part of our analogy. We now have a good idea of where the peak is. Instead of computing a massive, fully upsampled correlation map (e.g., 20x larger), we only need to calculate the values on a high-resolution grid in a tiny neighborhood around our 0.5-pixel estimate.

This is where the `dftups` function comes in. It performs an **Upsampled Discrete Fourier Transform (DFT) using direct matrix multiplication**.

Matlab

```
% Matrix multiply DFT around the current shift estimate
CC = conj(dftups(buf2ft.*conj(buf1ft), ceil(usfac*1.5), ceil(usfac*1.5), usfac, ...
      dftshift-row_shift*usfac, dftshift-col_shift*usfac));
```

Let's break down what `dftups` does internally:

Matlab

```
function out = dftups(in, nor, noc, usfac, roff, coff)
    % ...
    % Compute kernels and obtain DFT by matrix products
    kernc = exp((-1i*2*pi / (nc*usfac)) * ...);
    kernr = exp((-1i*2*pi / (nr*usfac)) * ...);
    out = kernr * in * kernc;
    return
```

- Instead of using an FFT algorithm, it directly implements the mathematical formula for the DFT.
    
- `kernr` and `kernc` are the DFT "kernels" (the complex exponential terms) for the rows and columns. They are specifically calculated for the desired high-resolution grid points (`usfac`) in a small output region (`nor`, `noc`) at a specific offset (`roff`, `coff`).
    
- The line **`out = kernr * in * kernc`** is a separable 2D DFT computed via matrix multiplication. Because the output region is very small (e.g., a 30x30 grid for a `usfac` of 20), this calculation is incredibly fast compared to a massive FFT.
    

Finally, the algorithm finds the peak in this small, high-resolution `CC` map and adds its subpixel offset to the coarse estimate to get the final, highly precise shift.

Matlab

```
% Locate maximum and map back to original pixel grid
[rloc, cloc] = find(CCabs == max(CCabs(:)), 1, 'first');
% ...
row_shift = row_shift + rloc/usfac;
col_shift = col_shift + cloc/usfac;
```

---

### **Summary**

This algorithm achieves its remarkable efficiency and accuracy by:

1. **Starting with a fast FFT-based cross-correlation** to get a coarse estimate of the shift (to 0.5 pixel accuracy).
    
2. **Avoiding full upsampling** by intelligently identifying that only the area around the correlation peak matters.
    
3. **Using a localized, matrix-multiply DFT (`dftups`)** to compute a high-resolution grid _only_ in that small, critical neighborhood.
    

This provides the same precision as brute-force methods but with dramatically reduced computation time and memory requirements, making it a powerful and practical tool for high-precision image alignment.



中文詳細解釋「頻域補零」與「空間域插值」的等價關係。這是一個在數位訊號處理 (Digital Signal Processing) 中非常核心且巧妙的概念。

---

### **核心概念：用「頻率配方」來增加「像素細節」**

我們可以把這句話用一個比喻來理解：

- **空間域 (Spatial Domain)**：就是我們眼睛看到的影像，由一個個像素點組成。
- **頻域 (Frequency Domain)**：是影像的「頻率配方」。傅立葉變換 (FFT) 能告訴我們，這張影像是由哪些不同頻率（細節的疏密程度）和振幅（對比度）的正弦波/餘弦波疊加而成的。低頻成分代表影像的輪廓和緩慢變化的背景，高頻成分代表邊緣、紋理等劇烈變化的細節。

**「頻域補零」與「空間域插值」的等價關係** 的意思是：

> 我們想要在空間域的影像上增加更多像素點（插值），使其看起來更平滑、解析度更高。要達到這個目的，一個高效的方法是，先算出影像的「頻率配方」，然後在這個配方的基礎上「添油加醋」（補零），再把加料後的配方還原成影像。最終得到的影像，就和直接在原始影像上做理想插值的效果一模一樣。

---

### **為什麼它們是等價的？從原理上理解**

要理解其背後的原理，我們需要了解幾個關鍵點：

#### **1. 數位影像是「採樣」的結果**

一張數位影像，本質上是對一個連續的真實世界場景進行「採樣」的結果。例如，一張 `8x8` 的影像，就是在一個連續的畫面上取了 64 個離散的點。

#### **2. 離散傅立葉變換 (DFT) 的本質**

當我們對這張 `8x8` 的影像做 DFT (通常用 FFT 快速實現)，我們會得到一個 `8x8` 的頻譜圖。這 64 個頻譜點，可以看作是對影像 **連續傅立葉頻譜** 的 64 個「採樣點」。

**關鍵點**：空間域的採樣點數量，決定了我們在頻域能得到的採樣點數量。

#### **3. 插值 (Interpolation) 的目標**

假設我們想把 `8x8` 的影像放大成 `16x16`。我們的目標是在原有的像素點之間，創造出新的、合理的像素點。最理想的插值方法，是根據訊號處理理論，完美地從離散的採樣點還原出其背後的「連續訊號」，然後再從這個連續訊號上進行更密集的採樣。

這個理想的插值過程，在數學上被證明等同於 **Sinc 函數插值**。

#### **4. 「補零」操作如何實現理想插值**

現在，我們來看「頻域補零」是如何巧妙地達成這個目的的：

1. **原始影像 (空間域)**：一張 `8x8` 的影像。
2. **計算頻譜 (FFT)**：對其進行 FFT，得到一個 `8x8` 的頻譜 `F(u,v)`。這個頻譜包含了構成原始影像的所有頻率資訊。
3. **頻域補零 (Zero-Padding)**：
    - 我們創建一個更大的、比如 `16x16` 的全零矩陣。
    - 然後，將原始的 `8x8` 頻譜 `F(u,v)` 複製到這個 `16x16` 矩陣的 **中央**。其餘位置保持為零。
    - **重要**：這個操作的意義是，我們增加了頻譜的「點數」或「解析度」，但 **沒有增加任何新的高頻資訊**。我們只是在原有的頻譜採樣點之間插入了大量的零。
    
    [Diagram showing an 8x8 DFT result being placed in the center of a larger 16x16 zero matrix]
    
4. **還原影像 (Inverse FFT)**：對這個補零後的 `16x16` 頻譜進行反傅立葉變換 (IFFT)。
    
5. **結果 (空間域)**：我們會得到一張 `16x16` 的影像。這張影像就是對原始 `8x8` 影像進行 **Sinc 函數插值** 的結果。
    

---

### **數學上的連結：卷積定理 (Convolution Theorem)**

為什麼會這樣？這背後有深刻的數學原理：
- **卷積定理** 指出：兩個函數在一個域中的乘積，等價於它們在另一個域中的卷積 (Convolution)。
- 在頻域中，我們的「補零」操作，相當於將原始頻譜 `F(u,v)` 乘以一個 **矩形函數 (Rectangular function)**。（矩形函數在中心區域值為1，其他區域為0）。
- 根據卷積定理，頻域的乘法，對應到空間域就是 **卷積**。
- 矩形函數的傅立葉反變換正好是 **Sinc 函數**。

所以，整個流程可以翻譯成：
> [IFFT( 原始頻譜 × 矩形函數 )] = [ IFFT(原始頻譜) ∗ IFFT(矩形函數) ]
> **[ `16x16` 插值影像 ] = [ 原始 `8x8` 影像 ∗ Sinc 函數 ]**
而「與 Sinc 函數進行卷積」正是 **理想插值** 的數學定義。

### **結論**

「頻域補零」與「空間域插值」的等價關係，可以總結為：
**在頻域對訊號的傅立葉變換進行補零，然後再進行反傅立葉變換，其結果等同於在空間域對原始訊號進行理想的 Sinc 函數插值。**

這不僅僅是一個有趣的數學巧合，更是一種計算上極為高效的方法。在 `Efficient subpixel image registration` 演算法中，它被用來在互相關峰值的周圍，生成一個高解析度的網格，從而能以次像素的精度定位峰值的確切位置，而這一切都無需在空間域進行複雜且耗時的插值運算。