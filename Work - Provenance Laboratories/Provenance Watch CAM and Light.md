
![[Pasted image 20251210170006.png]]


## 1. 整體概念：三個成像通道 + 高度量測 + 多組光源

- **系統用途**：拍手錶各個部位（面盤、外圈、錶殼側面、刻字、微小特徵…）做後續 CV/AI 分析。
    
- **移動 & 旋轉**：實際的 watch 放在 Zaber X-RST120AK（旋轉）+ X-LRQ300BP（平移）上，不在這張 CAD 圖裡。
    
- **這張圖裡有的主角**：
    
    - 3 個相機：**macro 1 cam、macro 2 cam、micro cam**
        
    - 1 組 **Keyence CL-3000 confocal 高度量測頭**
        
    - 多個 **液體透鏡（liquid lenses）** 做快速自動對焦
        
    - 多組 **LED 光源**
        
        - 一部分由 **光源控制器 TICGR1000-D1** 控制
            
        - 另一部分由 **ADAM-6266-B I/O 模組** 控制
            

PC（client/控制端電腦）就是所有東西的「大腦」，透過 Ethernet / USB / RS-232 去控整套系統。

---

## 2. 中央「Macro 1」主光學通道

### 2.1 大圓錐 & 上方 macro 1 cam

- 圖中央大圓錐狀那一組是 **主拍照光學通道**：大多是 telecentric / 同軸照明鏡組。
    
- 上面那台寫 **macro 1 cam** 的相機：
    
    - 功能：拍整隻錶或大範圍（例如整個 dial + bezel）。
        
    - 優點：視野大，幾何變形小，適合量測外徑、圓心、同心度等等。
        

**連線關係：**

- macro 1 相機 → 直接用 **GigE / USB** 接到 **PC**。
    
- 這顆鏡頭的照明（通常是 telecentric ring/back light） →
    
    - 光源本體接到 **TICGR1000-D1 光源控制器**，
        
    - TICGR1000-D1 再透過 **Ethernet 或 RS-232** 接到 PC，
        
    - 或用 ADAM 的 I/O 給它 trigger / on-off。
        

### 2.2 右側支架上的 liquid lenses

- 圖右邊支架上畫的 **liquid lenses** 是放在 macro 光路中的可變焦元件。
    
- 用途：
    
    - 不用動 Zaber 的 Z 軸，就可以透過改變液體透鏡焦距快速調前後景（例如玻璃面、指針、面盤文字、外圈刻度）。
        
    - 可以做**多焦平面拍攝 / 快速對焦掃描**，方便後面做 focus stacking 或多平面檢測。
        

**連線 / 控制：**

- 液體透鏡本體 → 連到 **liquid lens driver**（驅動器），
    
- driver 再透過 ** USB / RS-232 / 模擬電壓** 被 PC 控制，  
    或是有些設計是讓 **相機提供 I2C / 電流輸出** 去直接控制（你圖上有註記 “liquid lens controled via camera API” 就是這種）。
    

---

## 3. Macro 2 cam：側視 / 斜視通道

- 你在 macro 1 cam 旁邊畫一顆 **macro 2 cam**（在側邊，看起來像斜視角度）。
    
- 功能大多是：
    
    - 拍錶殼側面、錶耳、錶冠、側邊刻字、錶圈斜面等。
        
    - 角度不同，避免反光，輔助 macro 1 的 top-down 視角。
        

**連線：**

- macro 2 cam 同樣 → 直接 **GigE / USB → PC**。
    
- 它自己的照明（可能是側面條形燈 / spot light）→
    
    - 通常由 **ADAM-6266-B 或 TICGR1000-D1** 控制亮度 / 開關，
        
    - 視實際接線看是走 I/O（ADAM）還是專用光源控制器。
        

---

## 4. Micro cam + 微距光學通道

左側立柱上的那套標「micro cam」、「micro」、「liquid lens controled via camera API」的就是**微距高倍率通道**：

- **micro cam**：
    
    - 視野很小、放大倍率高，用來看非常細的特徵：
        
        - 刻字的邊緣、micro-printing、logo 紋理、指針斷面、鑽石爪細節等等。
            
- 下方的 **micro objective + liquid lens**：
    
    - 可能是固定焦的微距鏡 + 1 顆或多顆液體透鏡，
        
    - 藉由液體透鏡來做快速對焦 / 掃 z。
        

**連線 & 控制：**

- micro 相機 → **GigE / USB 直接接 PC**。
    
- micro 通道的液體透鏡：
    
    - 你標註「**liquid lens controled via camera API**」→
        
        - 代表液體透鏡 driver 接在相機上（很多工業相機有內建 liquid lens port），
            
        - 然後 PC 透過 **相機 SDK** 去設定焦距（例如設參數 focus current 等）。
            
- micro 通道的光源：
    
    - 可能是小型 ring light / coaxial light，
        
    - 常由 **ADAM-6266-B** 或光源控制器控制。
        

---

## 5. Keyence CL-3000 confocal 高度感測

中間直立那個細長柱寫 **Keyence CL-3000 lens**：

- 這是 Keyence CL-3000 系列的 **共焦位移感測頭**。
    
- 功能：
    
    - 精確量測高度 / 距離（Z 方向），例如：
        
        - 表面高度作為 auto-focus reference，
            
        - 量測錶鏡、面盤的高度差，
            
        - 計算元件厚度、平整度。
            

**連線：**

- 這個 lens head 會有線連到一個 **CL-3000 控制器機箱**（不在圖上，通常放在機櫃裡）。
    
- 控制器再：
    
    - 用 **Ethernet / RS-232** 接到 PC，回傳數值，
        
    - 也可以用 I/O 跟 **ADAM 或 Zaber stage** 做同步，例如：
        
        - 旋轉到某角度 → 量一次高度 → 調整液體透鏡 / stage Z。
            

**為什麼要有這個高度 sensor？**

- 可以做**自動對焦閉迴路**：
    
    - 先用 Keyence 掃高度 → 算出目標表面高度 → 換算到液體透鏡的最佳電流 / stage Z 位置。
        
- 也能當作**量測設備**：不是只有影像，而是提供非常準的幾何 Z-data，對一些 QC 指標（厚度、彎曲）很有用。
    

---

## 6. 左側兩個光源：Controlled by ADAM

圖左邊兩個長得像散熱片 / 鋁擠殼的東西，你標**“controlled by ADAM”**：

- 這應該是 **高亮度 LED spot light / bar light**，用來做：
    
    - 斜向打光、遮蔽反射、強調刻印紋理、邊緣 highlight。
        
- 它們的電源 / driver 接到機櫃裡的 **LED driver**，
    
- **ADAM-6266-B** 則是用來：
    
    - 開關 / strobe（數位輸出 DO），
        
    - 或選擇不同照明模式。
        

**連線：**

- ADAM-6266-B → **Ethernet → PC**（Modbus/TCP 或其 API）
    
- ADAM 的 DO → LED driver → 光源。
    
- 這樣 PC 就可以在拍照序列裡對每張圖設定：
    
    - 哪顆燈開 / 關、
        
    - 先開側光再開背光、或同步觸發相機。
        

---

## 7. 光源控制器 TICGR1000-D1：中央/底部光源

你在底部錶座附近寫 **“controlled by light controller”**，並註明是 Opto-engineering **TICGR1000-D1**：

- TICGR1000-D1 是 Opto Engineering 的 **多通道高精度光源控制器**。
    
- 主要用途：
    
    - 控制主 telecentric/dome light 的亮度、脈衝、觸發，
        
    - 可能還順便控制一些 backlight / ringlight。
        
- 與 ADAM 的差異：
    
    - ADAM 比較像「通用的 I/O box」，
        
    - TICGR1000-D1 是專門給 Opto Engineering 光源，用於**精準電流 / PWM 控制、同步 strobe**。
        

**連線：**

- 光源 → 直接接 TICGR1000-D1 的輸出端。
    
- TICGR1000-D1 → 用 **Ethernet / RS-232** 接 PC，
    
    - 或由 ADAM 的 I/O 給它 trigger（例如曝光瞬間 strobe）。
        
- PC 在你的 Main Controller/Orchestrator 裡會：
    
    1. 設定某 channel 的亮度 / pulse width，
        
    2. 發送拍照命令（或經由 ADAM 發送 trigger），
        
    3. 同步讓相機曝光。
        

---

## 8. ADAM-6266-B 的角色：I/O 中樞

雖然本圖只有標「controlled by ADAM」，但在整個系統裡 ADAM 的典型角色會是：

- **多光源 on/off & strobe 控制**（數位輸出 DO）。
    
- **接收感測器 / 極限開關 / 安全門開關**（數位輸入 DI）。
    
- 有需要時，也可以用 DO 去當 **相機硬體 trigger** 或跟 Zaber stage 同步。
    

**連線：**

- ADAM-6266-B → Ethernet → PC，
    
- DO → 光源 driver / 相機 trigger / Keyence input，
    
- DI ← 極限開關 / 安全迴路等。
    

---

## 9. 誰直接跟「客戶端電腦」相連？誰不會？

**直接跟客戶端電腦（PC）連線的：**

1. 三個相機（macro 1, macro 2, micro）  
    → GigE / USB，傳影像 + 透過 SDK 控曝光、gain、trigger mode、液體透鏡（micro 通道）。
    
2. Zaber X-RST120AK & X-LRQ300BP  
    → USB / RS-232，PC 用 Zaber API 控制移動、定位。
    
3. Keyence CL-3000 控制器  
    → Ethernet / RS-232 回報高度資料、接受設定。
    
4. Opto-Engineering TICGR1000-D1 光源控制器  
    → Ethernet / RS-232 控制亮度 / 模式。
    
5. ADAM-6266-B  
    → Ethernet（Modbus TCP）控制 I/O。  
    6.（如果有）液體透鏡專用 driver  
    → USB / RS-232，被 PC 送指令。
    

**不直接跟 PC 連的：**

- 各種照明 head（LED ring、bar、backlight…）  
    → 它們只跟 **TICGR1000-D1 或 LED driver / ADAM 的 DO 端子** 相連。
    
- 液體透鏡本體  
    → 只連到 **相機的 liquid lens port 或專用 driver**。
    
- Keyence 的 lens head  
    → 只連回自己的 CL-3000 控制器。
    

---

## 10. 為什麼要這樣分成 3 個 camera？

1. **macro 1**：
    
    - 高精度、正視、幾何穩定 → 適合整體尺寸量測與外觀檢查。
        
2. **macro 2**：
    
    - 斜角 / 側視 → 看到 macro 1 看不到的側邊結構與刻字，減少反光。
        
3. **micro**：
    
    - 超高倍率 → 做防偽 / 瑕疵檢測（毛邊、刮痕、微紋理），不會為了大視野而犧牲空間解析度。
        

搭配 **液體透鏡 + Keyence 高度資訊 + 多種照明 + Zaber 旋轉/平移**，可以在一隻錶的不同位置、不同角度、不同高度，一口氣完成所有需要的 image tasks。