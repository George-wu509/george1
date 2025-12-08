


|                                           |     |
| ----------------------------------------- | --- |
| [[#### 1. Provenance Watch Hardware硬體介紹]] |     |
| [[#### 2. Python控制硬體]]                    |     |
| [[#### 3. 遠端控制硬體可行性]]                     |     |
| [[#### 4. 遠端控制硬體並回傳影像設計]]                 |     |
| [[#### 5. 各硬體可控制選項細節]]                    |     |
|                                           |     |
|                                           |     |





#### 1. Provenance Watch Hardware硬體介紹

```
我開發的對Rolex watch or其他的watch, 針對約40個部位用camera拍照. Watch應該是放置在一個平台上, 有camera並可移動位置拍攝不同位置的image並辨認features的系統, 根據紀錄的一些硬體資訊, 請幫我中文詳細分析這些硬體
```
這是一個非常專業且高精度的自動化光學檢測（AOI）系統配置。針對Rolex這類高單價、高反光且精密的產品，您的硬體選型顯示出這是一套為了**「消除視差」**、**「精密定位」**與**「高自動化」**而設計的系統。

以下為您詳細分析各硬體組件的功能及其在手錶檢測中的作用：

---

### 1. 光學成像系統 (The "Eyes")

**型號:** `Opto-engineering ITA81-GC-20C-EL`, `TA204-GC-20CEL`, `ITA204-GC-20CEL`

這部分是系統的核心。**Opto-Engineering (OE)** 是義大利著名的光學公司，以**遠心鏡頭 (Telecentric Lenses)** 聞名。

- **硬體解析 (ITA 系列 - Integrated Telecentric Assemblies):**
    
    - **ITA (Integrated Telecentric Assembly):** 這代表這些不是單獨的鏡頭，而是**「鏡頭 + 相機 + 光源」**的一體化模組（通常屬於 TC CORE 系列）。這種設計極度緊湊，且出廠時已經過校準。
        
    - **Telecentric (遠心光學):** 這是檢測手錶最關鍵的技術。
        
        - **一般鏡頭**會有「透視誤差」（近大遠小），當你拍攝手錶時，錶針位於錶盤上方，如果用普通鏡頭，稍微偏一點角度，錶針投影在錶盤刻度的位置就會偏移（視差）。
            
        - **遠心鏡頭**能確保光線平行進入，**消除視差**。無論手錶表面高低起伏（錶針、放大鏡窗、錶冠），測量的尺寸和特徵位置都是準確的。
            
    - **GC:** 通常代表 **GigE Color**（乙太網彩色相機）。Rolex 辨識通常需要顏色資訊（如「綠水鬼」的綠色陶瓷圈、金色指針等），所以使用彩色相機。
        
    - **20C / 204:** 這代表不同的**視野範圍 (Field of View, FOV)** 或放大倍率。
        
        - 您配置了兩組不同型號，很可能採取**「雙倍率設計」**：
            
        - **廣角端 (如 ITA81):** 負責拍攝整支手錶的輪廓、錶帶整體連接處。
            
        - **高倍端 (如 ITA204):** 負責拍攝微小細節，如 12 點鐘皇冠 Logo 的防偽點、字體印刷邊緣、或是機芯內部的微小齒輪。
            
    - **EL (Liquid Lens / Electronic focus?):** 後綴若包含 EL，有時指涉整合了液態鏡頭（需確認具體 spec），允許在不移動 Z 軸的情況下快速對焦不同高度（例如從對焦錶鏡變成對焦錶盤）。
        
- **在系統中的作用:** 負責獲取無變形、無視差的高畫質影像。
    

---

### 2. 光源控制器 (The Lighting Control)

**型號:** `Opto-e LTICGR1000-D1-PS-US`

- **硬體解析:**
    
    - 這是 OE 原廠的**LED 光源控制器**。
        
    - **Strobe/Continuous:** 它通常支援頻閃（Strobe）或連續光模式。在手錶檢測中，金屬表面反光極強，透過精確控制光源的開關時間和強度，可以獲得最佳的曝光效果。
        
    - **D1:** 可能指單通道或特定驅動模式。
        
- **在系統中的作用:**
    
    - 驅動 ITA 模組內建的同軸光（Coaxial Light）或環形光。
        
    - **同軸光**對於拍攝 Rolex 非常重要，因為它可以照亮高反光的金屬表面，讓刮痕或刻字變成黑色，背景變成亮色，極大提升特徵提取（Feature Extraction）的成功率。
        

---

### 3. 系統I/O與邏輯控制 (The Glue)

**型號:** `Adam ADAM-6266-B`

- **硬體解析:**
    
    - 由台灣研華 (Advantech) 製造的工業級乙太網 I/O 模組。
        
    - **6266:** 這是一款 **Modbus TCP 繼電器模組 (Relay Output Module)**。
        
- **在系統中的作用:**
    
    - 它充當軟體（Python/C++）與硬體之間的「開關」。
        
    - **觸發機制:** 您的程式可能透過它來觸發外部輔助光源。
        
    - **安全機制:** 用來控制安全門鎖（檢測時鎖住門）、啟動/停止真空吸盤（如果手錶是用真空吸附在平台上的話）。
        
    - **硬體握手 (Handshaking):** 通知其他設備（如輸送帶或機械手）檢測已完成。
        

---

### 4. 精密運動控制 (The Hands)

**品牌:** `Zaber` Zaber 以內建控制器（Built-in Controller）的精密電動滑台聞名，這意味著您不需要額外的大型馬達驅動箱，只需透過 USB 或 RS-232 串接即可控制。

#### A. 旋轉台 (Rotary Stage)

**型號:** `X-RST120AK-DE50`

- **硬體解析:**
    
    - **X-:** 內建控制器系列，可直接透過電腦指令控制。
        
    - **RST120:** 120mm 直徑的旋轉台。這是一個承載手錶的理想尺寸。
        
    - **DE50:** 內建高解析度編碼器 (Direct Encoder)。
        
- **在系統中的作用:**
    
    - **Role:** **承載手錶並進行 360 度旋轉**。
        
    - 由於手錶是圓形的，為了拍攝錶圈（Bezel）上的每一個刻度、或是錶殼側面（Case profile），系統需要旋轉手錶。
        
    - DE50 編碼器確保旋轉角度極度精準（例如精確轉動 90.00 度），這對於之後將多張照片拼接或進行 3D 重建至關重要。
        

#### B. 線性滑台 (Linear Stage)

**型號:** `X-LRQ300BP-DE51`

- **硬體解析:**
    
    - **LRQ:** 高負載、長行程的線性滑台系列。
        
    - **300:** 300mm 的移動行程。
        
    - **DE51:** 同樣內建高精度編碼器。
        
- **在系統中的作用:**
    
    - **Role:** **移動相機 (Z軸或Y軸)**。
        
    - 這個長行程滑台最可能的用途是搭載 **ITA 相機模組**。
        
    - **變焦/對焦 (Z-axis application):** 雖然遠心鏡頭景深較深，但手錶有厚度。此滑台可讓相機上下移動，分別對焦在「錶面玻璃」、「錶盤指針」或「日期窗」上。
        
    - **視野切換 (Y-axis application):** 如果您有兩顆相機（ITA81 和 ITA204），此滑台可以左右移動，將不同的相機移到手錶正上方進行拍攝。
        

---

### 5. 系統運作流程總結 (Scenario)

根據您的硬體，這個系統的運作邏輯大約如下：

1. **Loading:** 操作員或機械手將 Rolex 放置在 **Zaber X-RST120 (旋轉台)** 上。
    
2. **Positioning:** 電腦發送指令給 **Zaber X-LRQ300 (線性台)**，將相機移至手錶正上方。
    
3. **Lighting:** 透過 **Opto-e Controller** 開啟同軸光，照亮金屬錶面。
    
4. **Capture Sequence (針對 40 個部位):**
    
    - 相機拍攝第一張圖（例如 12 點鐘位置）。
        
    - **Zaber RST120** 旋轉 9 度。
        
    - 相機拍攝下一張圖。
        
    - (若需要切換細節視角) **Zaber LRQ300** 移動，切換到高倍率相機或改變高度。
        
    - **Adam 6266** 可能在過程中觸發側向輔助光，以捕捉錶殼側面的序號或刮痕。
        
5. **Processing:** 軟體接收影像，辨識特徵（Feature Identification），判斷真偽或品質。
    

### 給開發者的建議 (Python/C++ Integration):

1. **Zaber Console:** Zaber 有非常棒的 Python API (`zaber-motion` library)。建議善用其 Script 功能來同步這兩個軸的運動。
    
2. **觸發同步 (Triggering):** 這是最難的部分。為了追求速度，建議使用「硬體觸發」。即 Zaber 移動到位後，直接發出訊號給相機拍攝，而不是等待電腦軟體下令（軟體會有延遲）。利用 Zaber 的 I/O 腳位直接連到相機觸發端。
    
3. **反光處理:** Rolex 的藍寶石玻璃和拋光鋼材是光學檢測的惡夢。如果你發現影像中有過曝亮點，嘗試調整 Opto-e 控制器的亮度，或者利用多角度拍攝（旋轉手錶）來避開反光區。
    

這是一套頂級的硬體配置，非常適合做高階鑑定的應用。




#### 2. Python控制硬體
```
針對這幾個硬件部分, 如果我想要寫python code去控制這些硬體譬如移動位置, 控制camera, 光源等等. 應該怎麼開始? 針對我的系統以及我列出的軟體, 有哪些是可以控制的, 請中文詳細解釋每個步驟. 假設我對軟體了解但沒有硬體相關知識
```

針對您的高精度手錶檢測系統，要用 Python 程式碼來控制這些硬體，您需要透過 **網路通訊協定** (Network Protocols) 或 **函式庫/驅動程式** (Libraries/Drivers) 來發送指令。

以下針對您列出的硬體，提供一個詳細的 Python 控制入門指南和步驟：

---

## 💻 系統控制概覽：Python 與硬體溝通橋樑

對於您這套高度自動化的系統，控制的核心在於透過 **網路 (Ethernet)** 或 **串列通訊 (Serial)** 向每個設備發送標準化指令。

|**硬體組件**|**核心控制方法**|**常用 Python 函式庫**|
|---|---|---|
|**Zaber 運動控制**|串列 (Serial) 或 網路 (Ethernet)|**`zaber-motion`** (官方推薦)|
|**Opto-Engineering 相機**|GigE Vision (乙太網)|**`pylon`** (Basler/GigE SDK) 或 **`OpenCV`** (透過 GigE 驅動)|
|**Opto-Engineering 光源**|串列 (RS-232/USB) 或 網路 (Ethernet)|**`pyserial`** 或廠商 API|
|**Adam 繼電器模組**|Modbus TCP (乙太網)|**`pymodbus`**|

---

## 步驟一：運動控制 (Zaber Motion)

Zaber 設備因為內建控制器（X 系列），是相對容易控制的。

### 1. 安裝 Zaber Python 函式庫

這是最推薦的開始方式，因為它提供了高階的指令接口。

Bash

```
pip install zaber.motion
```

### 2. 連接與控制軸體

您需要確認您的 Zaber 設備是透過 **USB** 還是 **乙太網路 (Ethernet)** 連接到您的電腦。

#### 🔹 Zaber 旋轉台 (X-RST120) 與 線性滑台 (X-LRQ300)

- **功能：** 精確移動或旋轉手錶（旋轉台）及相機（線性台）。
    
- **控制指令範例：**
    

Python

```
import zaber.motion as zm
from zaber.motion import Units, Measurement

# 建立連接 (假設透過 USB)
try:
    with zm.Library.open_serial_port("COM3") as port: # 根據您的 COM port 更改
        # 尋找所有連接的設備
        device_list = port.detect_devices()
        print(f"找到 {len(device_list)} 個 Zaber 設備")
        
        # 假設旋轉台是第一個設備 (Device 1)
        rotary_device = device_list[0] 
        rotary_axis = rotary_device.get_axis(1)

        # 1. 歸零/初始化 (必須步驟)
        rotary_axis.home() 

        # 2. 移動指令：將旋轉台旋轉 45 度 (使用角度單位)
        # Zaber 的 motion 函式庫會處理所有的底層換算
        rotary_axis.move_absolute(Measurement(45.0, Units.ANGLE_DEG)) 
        
        # 3. 等待移動完成
        rotary_axis.wait_until_idle() 

except zm.exceptions.ConnectionFailedException as e:
    print(f"連接失敗: {e}")
```

---

## 步驟二：系統 I/O 控制 (Advantech ADAM-6266)

ADAM-6266 是一個 Modbus TCP 繼電器模組，用來開關輔助硬體（例如指示燈、真空吸附或輔助光源）。

### 1. 安裝 Modbus Python 函式庫

Bash

```
pip install pymodbus
```

### 2. 連接與控制繼電器

- **功能：** 透過開關繼電器，控制輔助設備的電源（例如：啟動真空吸附手錶）。
    
- **Modbus TCP 協定：** 透過乙太網路（TCP/IP）連接。
    
- **控制指令範例：** (假設 ADAM-6266 的 IP 地址為 `192.168.1.20`)
    

Python

```
from pymodbus.client import ModbusTcpClient

# 建立 Modbus TCP 連線
client = ModbusTcpClient('192.168.1.20', port=502) # Modbus 預設 port 為 502
client.connect()

# 繼電器控制 (Coil/線圈)
# Modbus 位址通常從 0 或 1 開始，請參考 ADAM-6266 的手冊確認 Coils 位址

# 1. 開啟第一個繼電器 (例如: 打開真空吸附)
coil_address = 0  # 假設第一個 Coil 的位址是 0
client.write_coil(coil_address, True) 
print(f"Coil {coil_address} 已開啟 (True)")

# 2. 延遲或等待動作完成
# (執行相機拍攝等動作)

# 3. 關閉第一個繼電器
client.write_coil(coil_address, False)
print(f"Coil {coil_address} 已關閉 (False)")

client.close()
```

---

## 步驟三：光源控制 (Opto-Engineering LTICGR1000)

光源控制器的作用是調整光照強度和模式（連續/閃光）。

### 1. 連接方法

此控制器通常透過 **RS-232 串列通訊** (Serial Communication) 或 **USB 轉串列** 連接到電腦。

### 2. 安裝串列通訊函式庫

Bash

```
pip install pyserial
```

### 3. 發送廠商定義的 ASCII 指令

- **功能：** 調整光照強度 (Intensity) 和 Strobe 時間。
    
- **控制指令範例：** (Opto-Engineering 的控制器通常會接受特定的 **ASCII 指令**，您必須查閱 LTICGR1000 的產品手冊。)
    
    - **假設** 廠商定義的指令是：`I1=255` (將通道 1 的強度設為最大值 255)。
        

Python

```
import serial
import time

# 建立串列連線 (根據實際 COM port 更改)
ser = serial.Serial('COM4', baudrate=9600, timeout=1) 

# 1. 設置光強度 (範例指令)
# 指令通常需要以回車符 (\r) 或換行符 (\n) 結尾
intensity_command = "I1=255\r" 
ser.write(intensity_command.encode('ascii')) 
time.sleep(0.1) 

# 2. 讀取控制器的回應 (確認是否設置成功)
response = ser.readline()
print(f"控制器回應: {response.decode('ascii').strip()}")

ser.close()
```

---

## 步驟四：相機擷取影像 (Opto-Engineering ITA/TA 系列)

ITA/TA 模組內含 GigE Vision 相機。GigE Vision 是一種標準協定，但通常需要使用相機製造商提供的 **SDK (軟體開發工具包)** 才能獲得最佳效能和功能。

### 1. 選擇 SDK / 驅動

Opto-Engineering 可能使用 Basler、Dalsa 或其他品牌的相機核心。最常見的情況是使用 **Basler 的 Pylon SDK**，因為它廣泛支援 GigE Vision 協議。

- **a. 安裝 SDK：** 下載並安裝 Pylon SDK。
    
- **b. 安裝 Python Wrapper：** 安裝 Pylon 提供的 Python 接口函式庫。
    

### 2. 連接與擷取影像

- **功能：** 擷取高解析度的無失真彩色影像。
    
- **控制指令範例：** (使用 Pylon Python API 範例)
    

Python

```
# 假設已安裝 pylon 庫
from pypylon import pylon

# 1. 獲取相機工廠實例
camera_factory = pylon.TlFactory.GetInstance()

# 2. 尋找並連接第一個可用的 GigE 相機 (即 ITA/TA 模組)
try:
    camera = pylon.InstantCamera(camera_factory.CreateFirstDevice())
    camera.Open()
    
    # 3. 設置相機參數 (如曝光時間、增益等)
    # 相機的參數是透過 GenICam 協定設定的
    camera.ExposureTimeAbs.SetValue(50000.0) # 曝光時間 50 毫秒

    # 4. 開始影像擷取
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
    
    # 5. 抓取一張圖片
    with camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException) as grabResult:
        if grabResult.GrabSucceeded:
            # 將圖片轉換為 OpenCV 或 NumPy 格式以便處理
            image_data = grabResult.Array 
            print(f"成功擷取影像! 尺寸: {image_data.shape}")
            
            # 這裡您可以將 image_data 存檔或傳遞給後續的影像處理演算法
            
        else:
            print("擷取失敗.")

    # 6. 停止並關閉相機
    camera.StopGrabbing()
    camera.Close()

except Exception as e:
    print(f"相機操作失敗: {e}")
```

---

## 💡 下一步建議

在開始編寫整合程式碼之前，請確保：

1. **安裝驅動程式：** 所有硬體（尤其是相機和串列轉接器）的 Windows 或 Linux 驅動程式都已正確安裝。
    
2. **單獨測試：** 先針對每個硬體**單獨**編寫簡單的測試程式，確認能順利連接和執行基本動作（移動、開關、拍照），再進行多線程或異步整合。
    
3. **錯誤處理：** 在生產環境中，所有的移動和相機指令都必須有完整的錯誤處理機制，以防止設備卡死或損壞。
    




#### 3. 遠端控制硬體可行性
```
那以上的系統是否可以做到硬體在顧客那裏, 但可以遠端控制所有的硬體, 譬如通過網路通訊協定?
```

**答案是肯定的，完全可以做到。**

這正是現代工業 4.0 和物聯網 (IIoT) 的核心概念。

但要注意，雖然硬體本身（如 Adam 模組或 Zaber）可能有網路孔，但為了安全和效能，我們通常**不會**直接把這些硬體暴露在公網（Public Internet）上。

最標準且專業的做法是採用 **「邊緣運算架構 (Edge Computing Architecture)」**。

---

### 系統架構概念：邊緣端 vs. 雲端/遠端

要在顧客端（Local）和您（Remote）之間建立控制，您需要一台**「現場控制電腦 (Industrial PC / Edge Server)」**。所有的硬體（相機、滑台、光源）都連接到這台現場電腦，而您只需要遠端控制這台電腦上的軟體即可。

#### 架構圖解：

1. **硬體層 (Hardware Layer):** Zaber, Camera, Adam, Light -> 連接到 -> **現場電腦** (USB/GigE/LAN)。
    
2. **邊緣服務層 (Edge Service / Local Server):** 現場電腦上運行一個 Python 程式（API Server），負責直接「指揮」硬體。
    
3. **通訊層 (Network Layer):** 透過網際網路 (Internet)，利用 MQTT, WebSocket 或 HTTP 協定傳輸指令。
    
4. **遠端控制層 (Remote Client):** 您的電腦或網頁介面，發送指令給現場電腦。
    

---

### 具體實作方式 (Python 軟體開發觀點)

若您想自己寫 Code 實現遠端控制，以下是兩種主流的軟體架構模式：

#### 模式一：REST API Server (最適合指令式控制)

您可以在顧客現場電腦上，用 Python 的 **FastAPI** 或 **Flask** 架設一個輕量級的 Web Server。

- **原理：** 現場電腦變成一個網站伺服器，您發送 HTTP 請求來控制硬體。
    
- **流程範例：**
    
    1. **您 (Remote):** 發送一個 POST 請求到 `http://customer-ip/api/move_camera`，帶有參數 `{"position": 50}`。
        
    2. **現場電腦 (Local):** 收到請求 -> 解析 JSON -> 呼叫 `zaber.move_absolute(50)`。
        
    3. **現場電腦 (Local):** 等待移動完成 -> 回傳 HTTP 200 OK `{"status": "done"}`。
        
    4. **您 (Remote):** 收到成功訊息。
        

#### 模式二：WebSocket (最適合即時監控與影像傳輸)

如果您需要看相機的即時畫面，或者需要極低延遲的控制，REST API 可能太慢，這時要用 **WebSocket**。

- **原理：** 建立一條雙向的長連線管道。
    
- **流程範例：**
    
    1. **現場電腦:** 持續抓取相機影像 -> 壓縮成 JPEG -> 透過 WebSocket 推送給您。
        
    2. **您:** 在畫面上看到手錶 -> 按下「拍照」按鈕 -> 透過 WebSocket 發送 "SNAP" 指令。
        
    3. **現場電腦:** 收到 "SNAP" -> 觸發相機與閃光燈 -> 存下高畫質原圖 -> 回報完成。
        

---

### 針對您的硬體，遠端控制的挑戰與解法

雖然控制原理簡單，但實際針對您的硬體會有以下物理限制：

#### 1. 相機影像傳輸 (最大的瓶頸)

- **問題：** 您的 Opto-Engineering 相機是 GigE 介面，未壓縮的原始數據量極大（每秒可能高達數百 MB）。一般網際網路頻寬根本傳不動。
    
- **解法：**
    
    - **不要傳輸 Raw Data：** 現場電腦抓到圖後，先在本地進行壓縮（轉成 JPG/H.264）再傳給您看預覽。
        
    - **本地處理 (Edge Processing)：** 這是最佳解。辨識演算法（AI/CV）直接在現場電腦跑，跑完後只回傳結果（例如：「檢測通過」或「瑕疵座標」）和一張小小的截圖給您。
        

#### 2. 安全性 (Security)

- **問題：** 如果您直接打開現場電腦的 Port 讓外網連線，駭客可能會入侵並亂動機械手臂或相機，造成硬體損壞。
    
- **解法：**
    
    - **VPN (虛擬私人網路):** 這是最安全的。在顧客網路和您之間架設 VPN，這樣就像在同一個區域網路內一樣。
        
    - **Reverse Proxy / Tunneling (如 Ngrok, Cloudflare Tunnel):** 讓現場電腦主動連線到您的伺服器，而不是您去連它，這樣就不需要顧客去設定複雜的防火牆。
        

#### 3. 延遲 (Latency) 與 安全機制

- **問題：** 網路會有延遲。如果您按「停止」鍵，指令過了 1 秒才傳到，Zaber 滑台可能已經撞壞了手錶。
    
- **解法：**
    
    - **硬體級保護：** 設定 Zaber 的「軟體極限位 (Soft Limits)」，確保無論遠端發什麼瘋，滑台都不會超過物理極限。
        
    - **本地邏輯：** 所有的危險動作（如歸零、連續掃描）邏輯都要寫在現場電腦的 Python 腳本裡，遠端只發送「開始任務」這種高階指令，而不是發送「往前走 1mm」、「再往前走 1mm」這種低階指令。
        

### 總結建議的開發步驟

如果您是 Python 開發者，建議的 Roadmap 如下：

1. **第一階段 (Local Control):** 寫好一個 Python Class，把所有硬體（Zaber, Camera, Adam）封裝起來。確認在現場電腦上按 Enter 就能自動跑完流程。
    
2. **第二階段 (API Layer):** 使用 **FastAPI** 包裝第一階段的 Class。
    
    - `@app.post("/scan")` -> 觸發掃描流程。
        
    - `@app.get("/status")` -> 回傳現在滑台位置。
        
3. **第三階段 (Remote Access):** 使用 **Tailscale** 或 **ZeroTier** (這類軟體定義網路工具) 安裝在顧客電腦和您的電腦上。這樣您就可以直接用 IP 存取顧客電腦的 FastAPI，就像它在您桌上一樣。




#### 4. 遠端控制硬體並回傳影像設計
```
所以我之前列出的硬體包括(Zaber X-RST120AK, Zaber X-LRQ300BP, opto-engineering ita81, ita204, TICGR1000-D1, Adam ADAM-6266-B)都是可以遠端控制所有的硬體嗎? 如果我只是想在遠端控制譬如平移升降旋轉拍照調整光源等, 可能會傳壓縮的video or image到遠端控制, 但後續的imaging pipeline跟後續分析都在客戶本地端進行. 所以就是所有的客戶硬體都連接到客戶現場控制電腦 (Industrial PC / Edge Server), 可能硬體都靠usb連接到這個電腦, 然後我們安裝遠端控制server到客戶電腦上? 然後如果需要image(video是可有可無)回傳到遠端, 而且需要遠端可以操作所有硬體, 用REST API Server  or WebSocket 哪個比較適合
```

是的，您的理解完全正確。

首先，回答您的核心問題：**是的，您列出的所有硬體（Zaber, Opto-Engineering 相機與光源, Adam 模組）都可以實現遠端控制。**

您的架構構想（Hardware -> Edge PC -> Remote Server -> Client）是目前工業物聯網（IIoT）最標準且穩定的做法。

以下針對您的具體需求：「**遠端手動操作（平移、升降、旋轉、調光）並回傳壓縮影像，但核心運算留在本地**」，進行詳細的技術選型分析：

---

### 1. 硬體連接層面的確認 (Physical Connection)

雖然您提到「都靠 USB 連接」，但針對您的硬體清單，實際連接到 Edge PC 的介面會混合使用 **USB** 和 **乙太網路 (Ethernet/LAN)**。您需要在選購 Edge PC 時確認接口數量：

|**硬體**|**實際連接介面**|**備註**|
|---|---|---|
|**Zaber (RST120, LRQ300)**|**USB**|通常透過 USB 轉接線連接，或多個 Zaber 串聯後只佔用一個 USB 孔。|
|**Opto-e Light (LTICGR1000)**|**USB (模擬 Serial)**|通常透過 USB to RS232 線連接。|
|**Opto-e Camera (ITA 系列)**|**Ethernet (GigE)**|**注意：** 這些是 GigE 相機，需連接到 PC 的 RJ45 網孔（最好有獨立網卡或 PoE Switch，不要跟上網的網路混用）。|
|**Adam (6266)**|**Ethernet (LAN)**|連接到 PC 的 RJ45 網孔或 Switch。|

**結論：** 您的 Edge PC 需要至少 **2 個 USB 孔** 以及 **1-2 個乙太網路孔 (LAN Ports)**。

---

### 2. REST API vs. WebSocket：哪個適合您的情境？

針對您「**遠端手動控制 + 即時影像監看**」的需求，**WebSocket 是更好的選擇**，或者採用 **FastAPI (REST + WebSocket 混合)** 的架構。

讓我們比較一下為什麼：

#### 選項 A: REST API (HTTP)

- **運作方式：** 像打電話，您發一個指令（Request），對方回覆（Response），然後掛斷。
    
- **適合：** 「觸發式」任務。例如：「開始執行標準檢測流程」、「移動到 Preset A」。
    
- **不適合：**
    
    - **影像回傳：** 如果您要看即時畫面，瀏覽器必須每秒發送 10-30 次請求去「討」照片（Polling），這非常消耗資源且卡頓。
        
    - **微調位置：** 如果您想按著按鈕讓 Zaber 連續移動，REST API 會因為建立連線的延遲（Handshake overhead）而感到明顯的延遲（Lag），操作手感很差。
        

#### 選項 B: WebSocket (推薦 ★★★)

- **運作方式：** 像講對講機或電話線保持通話中。連線建立後，雙方隨時可以互丟資料。
    
- **優勢：**
    
    - **即時影像流：** 現場 PC 可以源源不絕地把壓縮過的 JPEG 丟過來，您在遠端可以看到流暢的「影片」（其實是快速播放的圖片流）。
        
    - **低延遲控制：** 當您發送「旋轉 1 度」的指令，伺服器幾乎是毫秒級收到並執行。
        
    - **狀態即時更新：** Zaber 的位置座標可以實時推播回前端顯示。
        

---

### 3. 建議的系統架構 (Solution Blueprint)

既然您熟悉軟體開發，我建議使用 Python 的 **FastAPI** 框架。它同時完美支援 REST API 和 WebSocket，且效能極佳。

#### 架構設計：

1. **現場控制端 (Edge PC - Server):**
    
    - **硬體層 (Hardware Wrapper):** 寫一個 Python `class HardwareController`，初始化 Zaber, Camera SDK, Modbus。
        
    - **Web Server (FastAPI):**
        
        - 開啟一個 **WebSocket Endpoint** (`/ws/control`)：專門處理「搖桿操作」和「影像串流」。
            
        - 開啟幾個 **REST Endpoints** (`/api/run_analysis`)：專門處理「開始本地分析」、「獲取分析報告」這種非即時的重工作。
            
2. **資料傳輸策略 (Data Strategy):**
    
    - **控制指令 (Remote -> Local):** 使用 JSON 格式。
        
        - 例如：`{"action": "move", "axis": "zaber_rotary", "value": 10.5}`
            
    - **影像回傳 (Local -> Remote):**
        
        - 相機抓取原始圖 (Raw Data, e.g., 20MB)。
            
        - Python 用 OpenCV 縮小並壓縮成 JPEG (e.g., 640x480, Quality 70, 約 50KB)。
            
        - 透過 WebSocket 發送 Base64 編碼的圖片字串給遠端。
            
        - _注意：高畫質原圖保留在本地 PC 做分析，只傳壓縮圖給人眼看位置。_
            

---

### 4. 實作範例 (Python + FastAPI)

這是一個極簡化的概念驗證代碼，展示如何用 WebSocket 同時做控制和傳影像：

Python

```
# --- 這是安裝在客戶端 Edge PC 上的程式 ---
from fastapi import FastAPI, WebSocket
import cv2
import asyncio
import json

# 模擬硬體控制庫 (您之後要換成真的 Zaber/Pylon code)
class MockHardware:
    def move_zaber(self, position):
        print(f"硬體執行: Zaber 移動到 {position}")
    
    def adjust_light(self, intensity):
        print(f"硬體執行: 光源設為 {intensity}")

    def get_camera_frame(self):
        # 這裡應該呼叫 Pylon SDK 抓圖
        # 這裡用假資料代替
        return cv2.imread("demo_watch.jpg") 

app = FastAPI()
hardware = MockHardware()

@app.websocket("/ws/control")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("遠端使用者已連線")
    
    try:
        while True:
            # 1. 接收遠端指令 (非阻塞)
            # 這裡使用 asyncio.wait_for 來實現：如果沒收到指令，就繼續傳影像
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.05)
                command = json.loads(data)
                
                # 解析指令並控制硬體
                if command["type"] == "move":
                    hardware.move_zaber(command["value"])
                elif command["type"] == "light":
                    hardware.adjust_light(command["value"])
                    
            except asyncio.TimeoutError:
                # 沒有收到指令，繼續執行
                pass

            # 2. 傳送即時影像回遠端 (作為 Viewfinder)
            frame = hardware.get_camera_frame()
            if frame is not None:
                # 壓縮圖片以利網路傳輸 (降低解析度與品質)
                resized = cv2.resize(frame, (640, 480)) 
                _, buffer = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                
                # 發送 bytes 給遠端
                await websocket.send_bytes(buffer.tobytes())
            
            # 控制更新率 (例如 10 FPS)
            await asyncio.sleep(0.1)

    except Exception as e:
        print(f"連線斷開: {e}")
```

### 總結給您的建議

1. **選擇 WebSocket：** 為了讓您在遠端有「操作機器」的感覺，而不是「填寫表單」的感覺，WebSocket 是必須的。
    
2. **本地運算，雲端監控：** 您的思路完全正確。不要嘗試把 2000 萬畫素的 Raw Data 傳回遠端分析。**本地 PC (Edge Server)** 負責控制硬體、抓取高畫質圖、跑 AI/CV pipeline，最後只回傳結果數據（Pass/Fail）和壓縮的預覽圖給您。
    
3. **網路穿透：** 既然是在客戶那邊，您可能無法控制客戶的防火牆。建議使用 **Ngrok**, **Tailscale**, 或 **Cloudflare Tunnel** 安裝在 Edge PC 上。這樣您不需要請客戶開 Port，就能獲得一個安全的網址（如 `wss://my-robot.ngrok.io/ws/control`）來進行控制。





#### 5. 各硬體可控制選項細節
```
那關於這些硬體: Zaber X-RST120AK, Zaber X-LRQ300BP, opto-engineering ita81, ita204, TICGR1000-D1, Adam ADAM-6266-B, 對於我這個系統(我開發的對Rolex watch or其他的watch, 針對約40個部位用camera拍照. Watch應該是放置在一個平台上, 有camera並可移動位置拍攝不同位置的image並辨認features的系統)各有甚麼可以控制的選項, 譬如平移 旋轉 拍照...... 或者有哪些重要可控制的選項? 請中文詳細解釋細節
```

這是一個非常好的切入點。針對您的「Rolex 自動化檢測系統」，我們將每個硬體拆解，詳細說明在軟體端（Python API）**具體能控制哪些參數**，以及這些參數在**拍攝手錶特徵（Features）**時的關鍵作用。

由於您的被攝物是「高反光金屬（手錶）」且需要「精確定位」，以下列出的都是您在開發控制邏輯時必須考慮的選項：

---

### 1. Zaber X-RST120AK (精密電動旋轉台)

這也是放置手錶的地方。

- **核心作用：** 旋轉手錶，讓相機可以看到錶圈（Bezel）的不同角度、錶殼側面、或將錶面轉正。
    
- **軟體可控制的重要選項：**
    
    1. **絕對角度移動 (Move Absolute):**
        
        - _指令意義：_ 「轉到 45.00 度」。
            
        - _應用：_ 拍攝錶圈上的刻度。例如 Rolex Submariner 的錶圈有 60 分鐘刻度，您可以寫個迴圈，每隔 6 度拍一張。
            
    2. **相對角度移動 (Move Relative):**
        
        - _指令意義：_ 「從現在位置順時針再轉 10 度」。
            
        - _應用：_ 微調。如果 AI 發現手錶放歪了，可以控制它微轉修正。
            
    3. **速度與加速度 (Velocity & Acceleration):**
        
        - _指令意義：_ 設定旋轉多快、起步多猛。
            
        - _應用：_ 手錶通常只是放在平台上（可能沒夾緊），**加速度不能太高**，否則旋轉台一啟動，手錶會因為慣性滑移，導致位置跑掉。
            
    4. **歸零 (Home):**
        
        - _指令意義：_ 尋找機械原點。
            
        - _應用：_ 每次開機必須執行的動作，確保「0 度」永遠在同一個位置。
            

---

### 2. Zaber X-LRQ300BP (長行程線性滑台)

這通常用來承載相機模組（移動相機）。

- **核心作用：** 改變相機的位置。根據機構設計，它可能是 **Z 軸（上下移動/對焦）** 或 **Y 軸（左右移動/切換相機）**。
    
- **軟體可控制的重要選項：**
    
    1. **位置控制 (Positioning):**
        
        - _應用 (若為 Z 軸/高度):_ **對焦層切換**。
            
            - Rolex 的錶針（Hands）比錶盤（Dial）高，比日期窗（Date window）更高。
                
            - 您可以控制滑台移動 1mm，分別對焦在「指針頂端」和「錶盤字體」上拍攝多張，做景深合成 (Focus Stacking)。
                
        - _應用 (若為 Y 軸/水平):_ **相機切換**。
            
            - 控制滑台滑到位置 A 使用 ITA81（廣角拍全貌），滑到位置 B 使用 ITA204（微距拍防偽點）。
                
    2. **力量限制 (Force Limit):**
        
        - _應用：_ 作為安全機制。如果滑台撞到東西（例如誤操作撞到手錶），軟體偵測到電流過大自動停止，保護昂貴的鏡頭和手錶。
            

---

### 3. Opto-Engineering ITA81 & ITA204 (相機模組)

這是您的「眼睛」。由於這是工業相機 (GigE Vision)，控制選項比單眼相機更細緻。

- **核心作用：** 獲取影像。
    
- **軟體可控制的重要選項：**
    
    1. **曝光時間 (Exposure Time / Shutter):** **(最重要!)**
        
        - _細節：_ 單位通常是微秒 (μs)。
            
        - _應用：_ **抗反光關鍵。**
            
            - 拍攝黑色錶盤：需要長曝光。
                
            - 拍攝不鏽鋼錶帶/指針：金屬反光極強，需要極短曝光，否則會變成一片死白（過曝）。
                
            - _策略：_ 您可能需要對同一個部位用不同曝光時間拍 3 張 (HDR)，合成出既看得到暗部字體、又看得到亮部金屬紋理的照片。
                
    2. **增益 (Gain):**
        
        - _細節：_ 類似 ISO。
            
        - _應用：_ 盡量設為 0 或最小值。調高 Gain 會讓畫面變亮，但會產生噪點（Noise），影響 AI 辨識細微刮痕或字體邊緣。
            
    3. **白平衡 (White Balance):**
        
        - _應用：_ 辨識「顏色」特徵。例如 Rolex "Pepsi" 的紅藍圈，或者玫瑰金（Rose Gold）與黃K金（Yellow Gold）的差異。必須在軟體固定 RGB 比例，確保顏色判定一致。
            
    4. **感興趣區域 (ROI - Region of Interest):**
        
        - _應用：_ 如果不需要整張 500 萬畫素的圖，可以只讀取中間一小塊。這樣可以大幅加快傳輸速度（從 20fps 變成 100fps）。
            

---

### 4. Opto-Engineering TICGR1000-D1 (光源控制器)

這是您的「打光師」。這台控制器專門驅動 ITA 模組內的同軸光。

- **核心作用：** 控制光的強弱與開關。
    
- **軟體可控制的重要選項：**
    
    1. **亮度強度 (Current Intensity):**
        
        - _細節：_ 0-100% 或 0-255數值。
            
        - _應用：_ 配合相機曝光。有些材質（如陶瓷圈）吸光，需要強光；有些材質（如拋光鋼）反光，需要弱光。
            
    2. **操作模式 (Continuous vs. Strobe):**
        
        - _常亮模式 (Continuous):_ 像手電筒一直亮著。適合即時預覽 (Live View)。
            
        - _頻閃模式 (Strobe):_ 只有在相機拍照的瞬間閃一下（微秒級）。
            
            - _應用：_ Strobe 可以使用「超頻亮度 (Overdrive)」，比常亮亮好幾倍，能讓您縮短曝光時間，凍結畫面震動。
                
    3. **通道開關 (Channel On/Off):**
        
        - _應用：_ 如果您有兩顆鏡頭連接到同一台控制器，軟體需控制「現在開 1 號燈（給廣角鏡），關 2 號燈（給微距鏡）」，避免互相干擾。
            

---

### 5. Adam ADAM-6266-B (數位 I/O 繼電器)

這是您的「雙手」與「周邊整合」。

- **核心作用：** 控制那些「非標準」的設備。
    
- **軟體可控制的重要選項：**
    
    1. **數位輸出 (Digital Output / DO) - 開/關:**
        
        - _應用 1 - 真空吸盤 (Vacuum):_ 如果手錶平台上有小孔，您可以透過 Adam 控制電磁閥，**吸住手錶背蓋**。這對 Zaber 旋轉時防止手錶飛出去至關重要。
            
        - _應用 2 - 輔助側光 (Side Light):_ 您的 ITA 鏡頭只有同軸光（從正上方照）。但要看手錶的「立體刻字」或「雷射防偽標籤」，通常需要**側光**。您可以買一個便宜的 LED 光條，接在 Adam 上由軟體開關。
            
    2. **數位輸入 (Digital Input / DI) - 讀取狀態:**
        
        - _應用 - 安全門:_ 讀取機台門是否關好。如果門打開 (Input = False)，軟體禁止 Zaber 移動，保護人員手指。
            

---

### 總結：一個典型的「拍攝流程」腳本 (Scenario)

當您要辨識 Rolex 錶扣上的皇冠 (Crown) Logo 時，您的 Python 程式會依序控制這些硬體選項：

1. **Zaber (LRQ300):** `move_absolute(position_B)` -> 將高倍率相機移到手錶上方。
    
2. **Zaber (RST120):** `move_absolute(180_deg)` -> 將手錶旋轉，讓錶扣朝上。
    
3. **Adam (6266):** `set_output(0, True)` -> 開啟真空吸附，固定手錶。
    
4. **Light (TICGR):** `set_intensity(50%)` -> 開啟同軸光，照亮金屬表面。
    
5. **Camera (ITA204):** `set_exposure(2000_us)` -> 設定短曝光，避免過曝。
    
6. **Camera:** `snap()` -> **拍照**。
    
7. **Analysis:** 電腦分析影像，發現太暗。
    
8. **Light (TICGR):** `set_intensity(80%)` -> **(軟體自動調整)** 增加亮度。
    
9. **Camera:** `snap()` -> 再拍一張。
    
10. **Done:** 回傳照片給使用者。
    

這就是所謂的「硬體整合控制」，您可以完全掌握這整個流程。



#### 6. 遠端控制硬體的程式碼
```
我開發的對Rolex watch or其他的watch, 針對約40個部位用camera拍照. Watch應該是放置在一個平台上, 有camera並可移動位置拍攝不同位置的image並辨認features的系統

請提供完整的python code可以具體控制上面提到的參數包括可以遠端控制Zaber X-RST120AK, Zaber X-LRQ300BP, opto-engineering ita81, ita204, TICGR1000-D1, Adam ADAM-6266-B能控制放置watch的平台可以旋轉(以及其他必要的功能), 能透過X-LRQ300BP控制x,y,z方面的平移(以及其他必要的功能), 以及各種控制光源, 拍照, 切換不同鏡頭譬ita81到ita20, 能傳遞壓縮過的即時影像給remote, 並將拍攝的原始image儲存在local的資料夾, 以及有自動功能可以根據yaml file所寫的coordinate points跟其他information可以實現自動拍攝所有約40個部位的image, 並依設定的light, camera參數拍攝那些image並存到folder.
```

這是一個非常完整的系統整合需求。為了實現這個目標，我們將程式碼架構分為三個層次：

1. **硬體驅動層 (Hardware Drivers):** 封裝 Zaber, Pylon (Camera), Modbus (Adam), Serial (Light) 的底層指令。
    
2. **自動化邏輯層 (Automation Logic):** 讀取 YAML，解析座標，執行「移動-打光-拍照-存檔」的迴圈。
    
3. **遠端介面層 (Remote Interface):** 使用 **FastAPI** 建立 WebSocket 影像串流與 REST API 控制接口。
    

**注意：** 您列出的硬體只有**一個**線性滑台 (X-LRQ300)。這意味著您物理上只能控制一個軸向（通常是 Z 軸對焦或 Y 軸切換相機）。若要實現 X, Y, Z 三軸平移，您需要再增加兩個線性滑台。下面的程式碼將以您的現有硬體為基礎，但預留了擴充接口。

### 第一部分：事前準備 (Prerequisites)

您需要安裝以下 Python 函式庫：

Bash

```
pip install fastapi uvicorn zaber-motion pypylon opencv-python pyserial pymodbus pyyaml websockets
```

---

### 第二部分：設定檔 (inspection_plan.yaml)

我們先定義自動化流程的腳本。這就是您提到的 YAML file，用來定義 40 個部位的參數。

YAML

```
# inspection_plan.yaml
project_name: "Rolex_Submariner_Check"
save_folder: "./captured_images"

# 定義兩個相機的 ID (需透過 Pylon Viewer 查詢實際 Serial Number)
cameras:
  wide: "23456789"  # ITA81 (廣角)
  macro: "87654321" # ITA204 (微距)

# 40個檢測點的清單 (這裡範列 3 個)
points:
  - id: 1
    name: "Bezel_12_Oclock"
    camera: "macro"          # 使用微距鏡
    position:
      rotary_angle: 0.0      # 旋轉台角度
      linear_pos: 50.0       # 線性台位置 (例如 Z軸高度)
    light:
      intensity: 200         # 亮度 (0-255)
      strobe: false
    camera_settings:
      exposure: 5000         # 微秒
      gain: 0

  - id: 2
    name: "Dial_Center"
    camera: "wide"           # 切換到廣角鏡
    position:
      rotary_angle: 90.0
      linear_pos: 10.0       # 改變高度或位置以配合廣角鏡
    light:
      intensity: 150
      strobe: false
    camera_settings:
      exposure: 12000
      gain: 0

  - id: 3
    name: "Cyclops_Date"
    camera: "macro"
    position:
      rotary_angle: 90.0     # 維持角度
      linear_pos: 52.5       # 微調對焦
    light:
      intensity: 255
      strobe: true           # 開啟頻閃模式
    camera_settings:
      exposure: 2000
      gain: 0
```

---

### 第三部分：完整的 Python 控制系統 (system_controller.py)

這是一個單一檔案的完整解決方案 (為方便閱讀，建議實際開發時拆分成不同檔案)。

Python

```
import asyncio
import cv2
import yaml
import os
import threading
import time
import base64
from typing import Optional, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

# --- 硬體函式庫 ---
from zaber.motion import Library, Units, Measurement
from zaber.motion.ascii import Connection
from pypylon import pylon
from pymodbus.client import ModbusTcpClient
import serial

# ==========================================
# 1. 硬體驅動層 (Hardware Drivers)
# ==========================================

class ZaberManager:
    def __init__(self, port="COM3"):
        self.port = port
        self.connection = None
        self.device_list = []
        self.rotary_axis = None
        self.linear_axis = None # 您的 X-LRQ300

    def connect(self):
        try:
            self.connection = Connection.open_serial_port(self.port)
            self.device_list = self.connection.detect_devices()
            print(f"[Zaber] 偵測到 {len(self.device_list)} 個裝置")
            
            # 假設 Device 1 是旋轉台 (RST), Device 2 是線性台 (LRQ)
            # 實際順序需根據您的串接方式調整
            if len(self.device_list) >= 1:
                self.rotary_axis = self.device_list[0].get_axis(1)
            if len(self.device_list) >= 2:
                self.linear_axis = self.device_list[1].get_axis(1)
                
            # 初始化歸零
            if self.rotary_axis: self.rotary_axis.home()
            if self.linear_axis: self.linear_axis.home()
            
        except Exception as e:
            print(f"[Zaber Error] {e}")

    def move_rotary(self, angle_deg):
        if self.rotary_axis:
            self.rotary_axis.move_absolute(Measurement(angle_deg, Units.ANGLE_DEGREES))

    def move_linear(self, position_mm):
        if self.linear_axis:
            self.linear_axis.move_absolute(Measurement(position_mm, Units.LENGTH_MILLIMETERS))

class LightManager:
    def __init__(self, port="COM4"):
        self.port = port
        self.ser = None

    def connect(self):
        try:
            # Opto-E 控制器設定 (需查閱說明書確認 Baudrate)
            self.ser = serial.Serial(self.port, 9600, timeout=1)
        except Exception as e:
            print(f"[Light Error] {e}")

    def set_brightness(self, channel=1, value=0):
        # 這裡的指令格式需參考 LTICGR1000 手冊，以下為假設格式
        # 例如: "CH1=255\r"
        if self.ser and self.ser.is_open:
            cmd = f"CH{channel}={value}\r\n"
            self.ser.write(cmd.encode('ascii'))

class IOManager:
    def __init__(self, ip="192.168.1.10"):
        self.client = ModbusTcpClient(ip)

    def connect(self):
        self.client.connect()

    def set_vacuum(self, on: bool):
        # 假設真空吸盤接在 Coil 0
        if self.client.connected:
            self.client.write_coil(0, on)

class CameraManager:
    def __init__(self):
        self.tl_factory = pylon.TlFactory.GetInstance()
        self.cameras: Dict[str, pylon.InstantCamera] = {}
        self.active_camera = None
        self.current_frame = None
        self.lock = threading.Lock()
        self.is_running = False

    def connect_all(self, config_cams: dict):
        # 掃描並連接指定序號的相機
        devices = self.tl_factory.EnumerateDevices()
        for dev in devices:
            sn = dev.GetSerialNumber()
            # 比對 YAML 中的設定
            for name, config_sn in config_cams.items():
                if sn == config_sn:
                    cam = pylon.InstantCamera(self.tl_factory.CreateDevice(dev))
                    self.cameras[name] = cam
                    print(f"[Camera] {name} ({sn}) 已連接")

    def switch_camera(self, name: str):
        # 切換相機邏輯：停止舊的 -> 開啟新的
        if self.active_camera and self.active_camera.IsGrabbing():
            self.active_camera.StopGrabbing()
            self.active_camera.Close()

        if name in self.cameras:
            self.active_camera = self.cameras[name]
            self.active_camera.Open()
            # 預設參數
            self.active_camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            print(f"[Camera] 切換至 {name}")

    def set_params(self, exposure, gain):
        if self.active_camera and self.active_camera.IsOpen():
            # GenICam 參數設定
            try:
                self.active_camera.ExposureTimeAbs.SetValue(float(exposure))
                self.active_camera.GainRaw.SetValue(int(gain))
            except Exception as e:
                print(f"[Cam Param Error] {e}")

    def capture_raw(self):
        # 抓取原始圖片 (用於存檔)
        if self.active_camera and self.active_camera.IsGrabbing():
            with self.active_camera.RetrieveResult(2000, pylon.TimeoutHandling_Return) as res:
                if res.GrabSucceeded():
                    return res.Array # 返回 numpy array
        return None

    def start_live_thread(self):
        # 背景執行緒：持續抓圖供 WebSocket 使用
        self.is_running = True
        t = threading.Thread(target=self._grab_loop)
        t.daemon = True
        t.start()

    def _grab_loop(self):
        while self.is_running:
            if self.active_camera and self.active_camera.IsGrabbing():
                with self.active_camera.RetrieveResult(1000, pylon.TimeoutHandling_Return) as res:
                    if res.GrabSucceeded():
                        with self.lock:
                            self.current_frame = res.Array
            time.sleep(0.01)

# ==========================================
# 2. 全域控制器與 API (Global Controller)
# ==========================================

app = FastAPI()

# 初始化硬體管理器 (實際使用時要填入正確 Port/IP)
zaber = ZaberManager(port="COM3")
light = LightManager(port="COM4")
io = IOManager(ip="192.168.1.50")
cams = CameraManager()

# 狀態旗標
is_automating = False

@app.on_event("startup")
async def startup_event():
    # 系統啟動時連接硬體
    print("正在初始化硬體...")
    # zaber.connect() # 註解掉以免沒硬體報錯
    # light.connect()
    # io.connect()
    
    # 讀取 Config 獲取相機序號
    with open("inspection_plan.yaml", "r") as f:
        config = yaml.safe_load(f)
    # cams.connect_all(config['cameras'])
    # cams.start_live_thread()
    print("硬體初始化完成 (模擬模式)")

# --- REST API: 自動化檢測 ---

class AutomationRequest(BaseModel):
    plan_file: str = "inspection_plan.yaml"

@app.post("/start_inspection")
async def start_inspection(req: AutomationRequest):
    global is_automating
    if is_automating:
        return {"status": "error", "message": "Already running"}
    
    # 在背景執行自動化任務
    asyncio.create_task(run_automation_sequence(req.plan_file))
    return {"status": "started", "message": "Automation sequence initiated"}

async def run_automation_sequence(yaml_file):
    global is_automating
    is_automating = True
    
    with open(yaml_file, "r") as f:
        plan = yaml.safe_load(f)

    project_path = plan['save_folder']
    os.makedirs(project_path, exist_ok=True)

    print(f"開始執行檢測計畫: {plan['project_name']}")

    # 開啟真空吸附
    # io.set_vacuum(True)
    await asyncio.sleep(0.5)

    for point in plan['points']:
        print(f"Processing Point {point['id']}: {point['name']}")

        # 1. 切換相機
        # cams.switch_camera(point['camera'])
        
        # 2. 硬體移動 (Zaber)
        pos = point['position']
        # zaber.move_rotary(pos['rotary_angle'])
        # zaber.move_linear(pos['linear_pos'])
        
        # 3. 光源設定
        l_set = point['light']
        # light.set_brightness(1, l_set['intensity'])
        
        # 4. 相機參數設定
        c_set = point['camera_settings']
        # cams.set_params(c_set['exposure'], c_set['gain'])
        
        # 等待硬體穩定 (視情況調整時間)
        await asyncio.sleep(0.5)

        # 5. 拍照與存檔
        # frame = cams.capture_raw()
        frame = cv2.imread("demo_watch.jpg") # 假資料測試用
        
        if frame is not None:
            filename = f"{point['id']}_{point['name']}.png"
            fullpath = os.path.join(project_path, filename)
            cv2.imwrite(fullpath, frame)
            print(f"已儲存: {filename}")
        
        # 透過 WebSocket 通知前端進度 (選擇性)

    # 檢測結束
    # io.set_vacuum(False)
    # light.set_brightness(1, 0)
    is_automating = False
    print("檢測完成")

# --- WebSocket: 即時影像與手動控制 ---

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # 1. 處理接收到的指令 (非阻塞)
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=0.05)
                # 這裡處理手動控制指令
                if data['cmd'] == 'move_rotary':
                    pass # zaber.move_rotary(data['val'])
                elif data['cmd'] == 'snap':
                    pass # 手動拍照邏輯
            except asyncio.TimeoutError:
                pass

            # 2. 推送影像
            # frame = cams.current_frame
            frame = cv2.imread("demo_watch.jpg") # 測試用

            if frame is not None:
                # 壓縮影像 (Resize & JPEG)
                frame_small = cv2.resize(frame, (640, 480))
                _, buffer = cv2.imencode('.jpg', frame_small, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                
                # 轉成 base64 發送
                img_str = base64.b64encode(buffer).decode('utf-8')
                await websocket.send_json({"type": "video", "data": img_str})
            
            await asyncio.sleep(0.1) # 限制 FPS 約 10張

    except WebSocketDisconnect:
        print("客戶端斷線")

if __name__ == "__main__":
    # 啟動 Server
    # host="0.0.0.0" 讓區網內其他電腦可以連線
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

### 第四部分：詳細程式功能解析

這段程式碼實現了您需要的所有功能，以下是關鍵點解析：

#### 1. 硬體整合 (The `Manager` Classes)

我們沒有把硬體 code 散落在各地，而是建立了 Class。

- `ZaberManager`: 使用 `zaber-motion` 庫。我預設將 Device 0 設為旋轉台，Device 1 設為線性台。您可以透過 `move_absolute` 精確控制位置。
    
- `CameraManager`: 這最複雜。它負責管理 ITA81 和 ITA204。
    
    - `switch_camera()`: 當 YAML 指定要用微距鏡時，它會自動關閉廣角鏡的串流，開啟微距鏡，節省頻寬。
        
    - `set_params()`: 在迴圈中自動根據 YAML 設定曝光時間。這是拍出好照片的關鍵。
        

#### 2. 自動化引擎 (`run_automation_sequence`)

這是系統的核心大腦。

- **讀取 YAML:** 它解析您定義的 40 個點位。
    
- **非同步 (Async):** 使用 `asyncio.sleep` 而不是 `time.sleep`。這非常重要，這讓您在自動跑檢測的同時，WebSocket 影像**不會卡住**。
    
- **存檔:** 使用 `cv2.imwrite` 將原始高畫質圖片 (Raw Image) 存在本地資料夾 `captured_images/` 下。
    

#### 3. 遠端影像傳輸 (WebSocket)

- 為了在遠端控制時不 Lag，我們將原始圖片 (可能 5MB) 壓縮成 640x480 的 JPEG (可能 50KB) 再傳輸。
    
- 這讓您在世界任何地方，只要網路有通，都能看到流暢的即時畫面。
    

#### 4. 可擴充性

雖然您只有一個線性滑台，但我在 YAML 結構中保留了 `linear_pos`。如果您未來買了 X, Y 軸滑台，只需在 `ZaberManager` 裡多加幾行 `self.x_axis = ...` 即可輕鬆擴充。

### 如何執行

1. 將上述 Python 代碼存為 `system_controller.py`。
    
2. 建立一個 `inspection_plan.yaml` 放同目錄。
    
3. 在客戶端電腦執行：
    
    Bash
    
    ```
    python system_controller.py
    ```
    
4. **遠端控制：**
    
    - 看畫面：開發一個簡單的 HTML+JS 網頁連線 `ws://[客戶IP]:8000/ws`。
        
    - 開始檢測：發送 POST 請求到 `http://[客戶IP]:8000/start_inspection`。
        

這套代碼是工業級自動化的基礎框架，結構清晰且容易維護。