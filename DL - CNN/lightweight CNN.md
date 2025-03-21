
**輕量化 CNN 的核心概念：**

輕量化 CNN 的設計目標是在保持模型準確度的前提下，大幅降低模型的計算複雜度和參數數量，使其能夠在資源有限的裝置（例如手機、嵌入式系統）上高效運行。

**1. MobileNet**

- **核心特點：**
    - **深度可分離卷積（Depthwise Separable Convolution）：** 這是 MobileNet 的核心創新。它將標準卷積分解為兩個步驟：
        - **深度卷積（Depthwise Convolution）：** 對每個輸入通道獨立進行卷積。
        - **逐點卷積（Pointwise Convolution）：** 使用 1x1 卷積組合深度卷積的輸出。
    - 這種分解方式大幅減少了計算量和參數數量。
    - 引入 Width Multiplier 與 Resolution Multiplier，讓使用者可以更彈性的去調整模型的大小與準確度。
- **與傳統 CNN 的差異：**
    - 傳統 CNN 使用標準卷積，同時進行空間和通道的卷積，計算量較大。
    - MobileNet 的深度可分離卷積大大降低了計算成本。
- **應用場景：**
    - 行動裝置上的圖像分類、物件偵測、人臉辨識等任務。

**2. EfficientNet**

- **核心特點：**
    - **複合縮放（Compound Scaling）：** EfficientNet 提出了一種統一的縮放方法，同時調整模型的深度（depth）、寬度（width）和解析度（resolution），以達到最佳的效能。
    - 使用 AutoML 自動搜尋最佳模型結構。
    - 使用 MobileNetV2 所使用的反向殘差結構(Inverted residuals)與壓縮激活(Squeeze-and-Excitation Networks)。
- **與傳統 CNN 的差異：**
    - 傳統 CNN 通常只單獨調整模型的某個維度（例如深度）。
    - EfficientNet 的複合縮放方法能夠更有效地平衡模型的各個維度，提高效能。
- **應用場景：**
    - 需要高精度和高效率的圖像分類任務。

**3. 常用的輕量物件偵測模型：**

- **YOLOv5-Lite/YOLOv8-Nano:**
    - YOLO 系列的輕量版本，針對行動裝置和嵌入式系統進行了優化。
    - 使用輕量化的骨幹網路和高效的偵測頭。
    - 輕量化模型在精度上可能稍遜於大型模型，但在速度上具有明顯優勢。
- **SSD-MobileNet:**
    - 結合 SSD（Single Shot MultiBox Detector）物件偵測框架和 MobileNet 骨幹網路。
    - 在速度和準確度之間取得了良好的平衡。

**輕量化 CNN 的共同特點：**

- **更少的參數和計算量：** 適用於資源有限的裝置。
- **更快的推理速度：** 實現即時應用。
- **模型壓縮技術：**
    - 例如模型剪枝(model pruning)與量化(quantization)，常常被應用在輕量化模型上面，更加減少模型的參數與計算。

**與一般 CNN 的差異總結：**

- 一般 CNN 追求更高的準確度，通常具有更深的網路結構和更多的參數。
- 輕量化 CNN 追求更高的效率，通過創新的網路結構和優化方法，在保持一定準確度的前提下，大幅降低計算成本。



理解 SSD-MobileNet 的結構，需要將兩個關鍵組件分開來看，然後理解它們是如何結合的：SSD（Single Shot MultiBox Detector）和 MobileNet。

**1. SSD（Single Shot MultiBox Detector）**

- **核心特點：**
    - SSD 是一種單階段（single-shot）目標偵測器，這意味著它直接從輸入圖像預測目標的邊界框和類別，而不需要像兩階段偵測器（例如 Faster R-CNN）那樣先生成候選區域。
    - **多尺度特徵圖（Multi-scale Feature Maps）：** SSD 的一個關鍵特點是使用來自網路不同層次的多個特徵圖進行預測。這使其能夠檢測不同尺寸的目標。較淺的層次可以檢測較小的目標，而較深的層次可以檢測較大的目標。這某種程度上達到類似Feature Pyramid Network的效果，但是作法上稍有不同。
    - **預設框（Default Boxes）：** SSD 在每個特徵圖單元位置放置一組預設框，這些預設框具有不同的尺寸和長寬比。模型學習調整這些預設框，以更好地匹配真實目標。
- **與 Feature Pyramid Network (FPN) 的關係：**
    - 雖然 SSD 通過使用多個特徵圖實現了多尺度檢測，但它與 FPN 的方法有所不同。FPN 通過自上而下的方式構建特徵金字塔，增強了低層次特徵的語義資訊。
    - SSD 直接利用網絡前向運算的各層Feature map, FPN則是將高層的語意特徵向底層傳遞。
    - 因此，可以說 SSD 具有類似 FPN 的多尺度檢測能力，但實現方式不同。

**2. MobileNet**

- **核心特點：**
    - MobileNet 是一種輕量級的 CNN 架構，專為行動和嵌入式裝置設計。
    - 它使用深度可分離卷積，顯著減少了計算成本和參數數量。

**3. SSD-MobileNet 的結合**

- **結構：**
    - SSD-MobileNet 將 MobileNet 作為其骨幹網路（backbone network）。這意味著 MobileNet 負責從輸入圖像中提取特徵。
    - 然後，SSD 的多尺度預測層被添加到 MobileNet 的不同層次的特徵圖上。
    - 具體來說，在 MobileNet 的幾個關鍵層次之後，添加了額外的卷積層，這些層輸出的特徵圖用於預測目標的邊界框和類別。
    - 因此MobileNet負責輕量化的特徵提取，SSD負責多尺度的目標檢測。
- **優勢：**
    - 這種結合使得 SSD-MobileNet 能夠在資源有限的裝置上實現實時目標檢測。
    - MobileNet 提供了輕量級的特徵提取，而 SSD 提供了高效的多尺度目標檢測能力。

**總結：**

- SSD-MobileNet 是一種高效的目標檢測模型，通過結合 MobileNet 的輕量級特徵提取和 SSD 的多尺度預測能力，實現了在資源有限的裝置上的實時目標檢測。
- SSD 本身透過利用深淺不同的feature map 來達到多尺度檢測的效果，因此，跟FPN的作用類似，但是實作方式不同。