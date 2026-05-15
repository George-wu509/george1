


|                                         |     |
| --------------------------------------- | --- |
| [[#### 目前有的Control hardware code跟修改建議]] |     |
| [[#### Base64 JSON 傳圖是否該修正?]]           |     |
| [[#### USW-pro-max-16-poe設定]]           |     |
| [[#### 為 Ethernet 2 設定靜態 IP]]           |     |
| [[#### 測試ita81跟liquid lens]]            |     |
|                                         |     |
|                                         |     |
|                                         |     |


#### 目前有的Control hardware code跟修改建議
```
我現在開發這套V2 Moonlight 系統, 自動化手錶光學檢測系統 (python based imaging processing system在windows platform並支援GPU)。這是一套對每一次一個watch由自動camera system拍攝錶面不同部位約90張images (2048 x 2048)總共約2.0GB傳入, 並針對每張image有不同的imaging processing task分析上面的features並儲存結果可能包括segmentation masks及這種values並傳出某個matrics, 大部分會使用opencv等傳統影像library但其中1/3有些可能需要segmentation AI models or OCR(UNet, SAM, OCR)辨認上面文字. 最後整合90幾個這些matrics並和某些reference matrics做比對, 最後確認這watch是真的或假的, 每月大概會scan 500隻手錶.

我的系統是自動光學檢測 (AOI) 系統包括多相機系統以及精密運動平台可控光源多通道燈光控制. 對watch約40個部位用camera拍照. Watch應該是放置在一個平台上, 有camera並可移動位置拍攝不同位置的image並辨認features的系統. 使用 3個Zaber線性控制器(Zaber X-LRQ300BP)進行 X, Y, Z 三軸移動, 以及兩個Zaber旋轉控制器(Zaber X-RST120AK)有旋轉軸指向z方向讓放手表的平台旋轉, 以及旋轉軸指向x方向讓上方相機支架旋轉. 有三個相機跟一個LTDVE2CH-20F light controller跟3個Adam ADAM-6266-B. 相機有兩個macro camera(ITA204-GC-20C)共用同個光路及liquid lenses跟light controller(LTDVE2CH-20F), 一個macro camera用於拍攝整個手錶，另一個用於拍攝較小的部分. 另一個micro camera(ITA81-GC-20C-EL)走另一個光路包含另一個liquid lenses跟同一個light controller(因為有兩個通道，可以連接兩個光路的light source環形燈). 另外還有兩個Adam ADAM-6266-B控制兩個非常強大的光源（每個40瓦），它們連接到micro camera的光路就像有一個非常明亮的聚光燈照在焦平面上. 透過相機 API 控制liquid lenses。一條線連接液體鏡頭和相機，然後相機透過網路線連接到電腦。除此之外還有一個背光light source也是用ADAM-6266-B控制. 因此，液體鏡頭是相機 API 中可以控制的額外設定. 除此之外還有雷射/位移感測器Keyence CL-3000用於測量表面高度和玻璃厚度. 流程大概是這樣:「先用大範圍相機定位 -> 移動到目標位置 -> 利用感測器進行精密對焦（需補償玻璃折射） -> 自動調整曝光與增益 -> 拍攝 HDR 影像 -> 用於特徵分析。」

請幫我檢查Zaber跟camera, light, liquid lenses等系統拍攝image相關的硬體有哪些跟操作移動,拍攝跟對焦相關的functions跟parameters? 我的Controller folder裡面的目前的python code functions有哪些相關功能性? 以及建議我要再為這system加入那些functions? 請以40年Tech lead提供建議
```

你好！我是你的 Tech Lead。首先，這套系統的架構設計得相當有水準。你將硬體抽象化（Hardware Managers）、支援優雅降級（Graceful Degradation）與混合模擬模式（Hybrid/Simulation Mode），並且分離了 Client/Server 架構，這對於一套需要高度客製化、且每月要穩定產出 500 隻奢侈品手錶檢測結果的 AOI 系統來說，是相當成熟的起手式。

針對你的問題，我先幫你盤點目前程式碼中已具備的硬體控制功能與參數，接著以我 40 年的自動化與光學檢測經驗，給你「上線量產前」的強烈建議。

---

### 第一部分：目前的硬體控制功能與參數盤點

從你提供的 `Controller` 資料夾代碼中，我看到系統已經具備以下核心能力：

#### 1. 運動控制 (Zaber X/Y/Z 加上 兩個旋轉軸)

- **相關 Class**: `ZaberManager`, `UnifiedHardwareDriver`
    
- **已具備 Functions**:
    
    - `home_axis()`: 軸歸零（原點復歸）。
        
    - `move_axis()`: 絕對座標移動（具備軟體限位保護檢查）。
        
    - `move_relative()`: 相對座標移動。
        
    - `move_compound()`: 依次發送多軸移動指令。
        
- **控制 Parameters**: `axis_name` (如 stage_L_X), `position` (mm 或 deg), `wait` (是否等待到位)。
    

#### 2. 相機與液態鏡頭 (Opto-Engineering / Harvesters)

- **相關 Class**: `CameraManager`
    
- **已具備 Functions**:
    
    - `switch_camera()`: 切換大小相機，處理資源釋放。
        
    - `capture_image()`: 單張拍攝。
        
    - `capture_hdr()`: 多曝光包圍拍攝並使用 OpenCV (Mertens) 進行 HDR 合成。
        
    - `set_liquid_lens_raw()` / `set_liquid_lens_diopter()`: 控制液態鏡頭曲率（焦距），且已具備**溫度補償模型** (`lens_calib`)。
        
    - `sweep_liquid_lens()`: 快速變焦掃描（Z-Stacking 的前置準備）。
        
    - `set_white_balance_roi()`: 自動白平衡觸發。
        
- **控制 Parameters**: `exposure` (曝光時間 us), `gain` (增益), `liquid_focus` (電流或屈光度), `exposures` (HDR 倍率陣列)。
    

#### 3. 光源控制 (LTDVE2CH & ADAM)

- **相關 Class**: `LightManager`
    
- **已具備 Functions**:
    
    - `set_ring_light_continuous()` / `set_ring_light_strobe()`: 透過 TCP 控制 LTDVE 環形燈的常亮與頻閃（具備最大電流安全保護）。
        
    - `set_spot_light_pwm()`: 透過 Modbus 控制 ADAM 模組的聚光燈/背光 PWM。
        
- **控制 Parameters**: `channel_idx`, `intensity_ma` (電流), `duration_us` (頻閃時間), `duty_cycle_percent` (PWM 佔空比)。
    

#### 4. 雷射位移感測器 (Keyence CL-3000)

- **相關 Class**: `IOManager`, `UnifiedHardwareDriver` (內含 Robust Autofocus)
    
- **已具備 Functions**:
    
    - `read_stable_value()`: 讀取並取中位數（Median Filter）以抗雜訊。
        
    - `run_robust_autofocus()`: 從 Z-base 往下搜尋，配合 Keyence 數值閉迴圈控制 Z 軸進行精密對焦。
        

---

### 第二部分：Tech Lead 的架構與功能擴充建議 (Critical Additions)

你的系統目前是「能動、能拍」，但要達到「高精度、高良率、防呆抗摔」的工業級標準，以下是我強烈建議你必須加入的功能（請排入你的 Jira/Trello Backlog）：

#### 💡 1. 影像傳輸架構的致命傷：請廢除 Base64 JSON 傳圖

- **現狀問題**：你的 `controller_server.py` 在拍照後，使用 `base64.b64encode()` 將 2048x2048 的圖片塞入 JSON 傳送。一張 2K 彩色圖片未壓縮大約 12MB，轉 JPG 後約 1~2MB，Base64 會再膨脹 33%。這會導致 Socket 記憶體峰值飆高、延遲增加，且容易在 Stress Test 中斷線。
    
- **修正建議**：
    
    - **方案 A (推薦)**：如果 Server 和 AI 分析端在同一台電腦，請改用 **Shared Memory (共享記憶體)** 或直接在 Server 端存成 `.jpg/.tiff` 後，只傳回 `{"status": "ok", "file_path": "..."}`。
        
    - **方案 B**：如果跨網路，請改用二進制流 (Binary TCP/gRPC/ZeroMQ)，不要用 JSON 傳大影像。
        

#### 💡 2. 光學與取像演算法強化

- **景深合成 (Focus Stacking / EDF)**：你已經寫了 `sweep_liquid_lens()`，但手錶機芯 (Movement) 的高低落差極大，單張液態鏡頭照片一定有局部模糊。你需要加入 OpenCV 或第三方庫的 EDF (Extended Depth of Field) 演算法，將 Sweep 拍到的 5~10 張照片融合成一張「全景深」清晰圖，再餵給 AI Model。
    
- **平場校正 (Flat-Field Correction, FFC/Shading)**：液態鏡頭在改變焦距時，邊緣失光 (Vignetting) 會跟著改變。你需要加入 FFC 校正矩陣。在系統初始化時，拍一張標準白板，後續每張圖片先做 `(Image / White_Reference) * Mean`，這樣你的 AI (UNet/SAM) 找瑕疵時才不會被陰影干擾。
    
- **玻璃折射補償 (Sapphire Crystal Compensation)**：手錶錶面的藍寶石玻璃厚度通常在 1.5mm 到 3mm 之間，折射率 (Refractive Index) 約為 1.76。你目前代碼有寫到 `glass_correction = -0.4`，但這不夠 robust。建議加入公式：`實際 Z 軸移動量 = 測量深度 / 折射率`，否則你的相機拍錶盤內部永遠對不準。
    

#### 💡 3. 機構與運動控制強化 (Zaber)

- **齒隙補償 (Backlash Compensation)**：Zaber 雖然精準，但在微觀 (Micro Camera) 下，Z 軸「由上往下」和「由下往上」到同一個點，絕對會有幾微米的機械公差。建議在 `ZaberManager` 加入統一的進刀方向（例如規定：永遠從目標位置上方 1mm 處往下壓至目標點），以消除齒隙誤差。
    
- **防撞機制 (Collision Avoidance / Kinematic Limits)**：你雖然做了各軸的 `limits` (Min/Max)，但在 5 軸聯動下，旋轉臂 (Arm) 可能會撞到相機或手錶。你需要寫一個簡單的空間防護罩 (Bounding Box) 邏輯，在 `move_compound` 執行前，驗證空間干涉。
    
- **錶面傾斜補償 (Tilt Alignment)**：手錶放上平台不可能絕對水平。強烈建議在例行程序 (Routine) 第一步，用 Keyence 在手錶邊緣打三個點，計算出「平面法向量 (Normal Vector)」，後續的 X/Y 移動和 Z 軸對焦都要乘上這個旋轉矩陣。
    

#### 💡 4. 系統可靠度與硬體觸發 (Hardware Trigger)

- **硬體同步觸發 (Hardware Triggering)**：你目前是軟體循序控制 (`拍照 -> 亮燈 -> 等待 -> 抓圖`)。若要提高 HDR 和頻閃 (Strobe) 的穩定度，應該用一條實體 IO 線將 LTDVE 光源控制器與相機的 Trigger 腳位連起來。讓程式送出一個信號，硬體層自動完成「閃光+曝光」。
    
- **看門狗與心跳包 (Watchdog & Heartbeat)**：Client/Server 之間只有送指令時才通訊。加入一個背景 Thread 每秒互 ping (Heartbeat)。如果 Server 發現 Client 斷線，要自動觸發 `emergency_stop()` 關閉雷射、燈光和馬達，確保安全。
    
- **Hand-Eye Calibration (手眼校正偏移參數)**：雷射點打在手錶上的位置，和相機視野的中心點，絕對存在物理偏移 (Offset X/Y)。目前的 Config 中沒有看到這個校正參數，建議在 `system_config.yaml` 補上，並在 `run_robust_autofocus` 後，自動將 XY 補償回來再拍照。
    

### 總結你的下一步 (Next Actions)

做為你的 Tech Lead，我認可你現在的架構基礎。下週一前，請優先處理：

1. **重構 Image 傳輸機制**（拿掉 Base64 JSON，改用 File Path 或 Shared Memory）。
    
2. **完成 Glass Refraction Math**（把 `focus_controller.py` 裡的 `hasglass` 邏輯寫死，加入藍寶石玻璃 1.76 折射率計算）。
    
3. **實作 Hand-Eye Calibration Offset**（確保雷射打到的點，就是相機拍的正中心）。
    

這套系統很有潛力，AI 分析的部分只要影像源頭夠乾淨清晰，後面的 UNet 和 SAM 辨識率就會很高。繼續保持這個好架構，有問題隨時討論！



#### Base64 JSON 傳圖是否該修正?
```
請幫我全面詳細分析 Base64 JSON 傳圖這部分的code看是否有必要或應該修改, 當camera拍照之後應該會加密, 然後傳入local folder. 而之後更會將local folder的camera taken images傳到AWS S3做備份, 在系統內也會用這taken images用opencv or AI model做後續分析譬如image segmentation. 也會在App裡顯示在UI裡面. 以及可以在remote client控制local agent裏操作camera, 然後之後經過AWS顯示image在remote client的App裡. 以上大概就是images在這系統會進行的操作. 所以Base64 JSON 傳圖還是必須修改? 要怎麼修改請提供code
```
很高興你把整個影像的生命週期（Lifecycle）整理得這麼清楚！身為 Tech Lead，我可以直接且明確地回答你：**目前的 Base64 JSON 傳圖機制，不僅是「應該」修改，而是「必須」徹底淘汰。**

如果你繼續沿用 Base64 JSON 將 2048x2048 的影像塞在 Payload 裡傳遞，你的系統在進入遠端控制（Remote Client via AWS）階段時，**一定會遇到系統級的崩潰 (Hard Fail)。**

以下為你全面剖析原因，以及工業級 AOI 系統與 AWS 雲端架構的標準解法，並附上具體的重構程式碼。

---

### 為什麼 Base64 JSON 必須淘汰？

1. **AWS IoT Core 的死穴（Payload Limit）：** 你提到未來會透過遠端 Remote Client 控制 Local Agent。AWS IoT MQTT 的單一訊息 Payload 上限是 **128 KB**。一張 2048x2048 的圖片，就算用 JPG 壓縮 (Quality 85)，大約也是 500 KB 到 1 MB。轉成 Base64 還會再膨脹 33%（約 650 KB - 1.3 MB）。**這絕對傳不過去，AWS IoT 會直接丟棄封包或斷線。**
    
2. **Local 端的效能災難 (CPU & GIL Blocking)：** Python 的 `json.dumps()` 和 `json.loads()` 處理幾 MB 的長字串時，會咬住 CPU 並阻塞全域直譯器鎖 (GIL)。這會導致你的 Controller Server 在傳圖的瞬間，無法精準控制 Zaber 馬達或響應 Keyence 感測器，產生微秒級的硬體控制延遲。
    
3. **違背「單一真實來源 (Single Source of Truth)」原則：** 你既然都要把圖片存入 Local Folder 做備份與加密，那麼記憶體裡的 Base64 字串就成了多餘的複製品（Duplicate）。最好的方式是：**「傳址不傳值 (Pass-by-Reference)」**。
    

---

### 最佳實踐架構：Pass-by-Reference + S3 Presigned URL

針對你的情境，影像的傳遞路徑應該被拆分為「控制流 (Control Plane)」與「資料流 (Data Plane)」：

1. **Local 操作 (相機 -> Local AI/UI):** Server 拍照後，直接在 Local 端加密並存入硬碟。透過 JSON 傳給 Client 的是 **「檔案絕對路徑 (File Path)」**。Client 拿到路徑後，自己去硬碟讀取圖片顯示在 UI 或餵給 OpenCV/UNet。
    
2. **Remote 操作 (Local Agent -> AWS -> Remote App):** Agent 拍照後，將圖片上傳至 AWS S3，並利用 AWS SDK (Boto3) 產生一組 **S3 Presigned URL (預先簽章 URL，可設定 5 分鐘過期)**。Agent 透過 AWS IoT MQTT 把這串 URL 傳給 Remote Client。Remote App 拿到 URL 後，直接走 HTTPS 去 S3 下載圖片顯示，完美避開 MQTT 的大小限制。
    

---

### 具體程式碼修改建議

#### 1. 修改 `Controller/controller_server.py` (Server 端)

將 Base64 拔除，改為直接將圖片寫入 Local Folder，並回傳檔案路徑。_(註：若有加密需求，可在此處調用你的加密模組)_

Python

```
# Controller/controller_server.py (節錄)
import os
import time
import cv2

def process_command(self, cmd):
    action = cmd.get('action')
    try:
        if action == 'capture':
            # 1. 執行硬體拍照
            self.cam.switch_camera(cmd['cam_type'])
            exposure = cmd.get('exposure', 5000)
            img = self.cam.capture_image(exposure)
            
            if img is None:
                return {'status': 'error', 'msg': 'Capture failed'}

            # 2. 決定存檔路徑 (建議依照 watchid 和 timestamp 分類)
            watch_id = cmd.get('watchid', 'unknown_watch')
            timestamp = int(time.time() * 1000)
            save_dir = os.path.join("Local_Data", watch_id)
            os.makedirs(save_dir, exist_ok=True)
            
            filename = f"{cmd['cam_type']}_{timestamp}.jpg"
            filepath = os.path.join(save_dir, filename)

            # 3. 儲存圖片 (此處可加入你的 AES 影像加密邏輯)
            cv2.imwrite(filepath, img)
            
            # 4. 只回傳「路徑」給 Client
            return {
                'status': 'ok', 
                'filepath': os.path.abspath(filepath), # 回傳絕對路徑
                'timestamp': timestamp
            }
        # ... 其他邏輯
```

#### 2. 修改 `Controller/controller_client.py` (Local Client 端)

Client 端收到 JSON 後，提取路徑，並透過 OpenCV 直接從硬碟讀圖。

Python

```
# Controller/controller_client.py (節錄)
import cv2
import os

class ControllerClient:
    # ... 前面邏輯保持不變 ...

    def capture_image(self, cam_type, exposure, gain=0, watchid="test"):
        res = self.send_command({
            'action': 'capture', 
            'cam_type': cam_type, 
            'exposure': exposure, 
            'gain': gain,
            'watchid': watchid
        })
        
        if res['status'] == 'ok':
            filepath = res.get('filepath')
            
            # 從 Local Folder 讀取圖片 (若前面有加密，此處需解密)
            if filepath and os.path.exists(filepath):
                img = cv2.imread(filepath, cv2.IMREAD_COLOR)
                return img, filepath
            else:
                print(f"[Error] Image file not found at {filepath}")
                return None, None
        else:
            print(f"Capture failed: {res.get('msg')}")
            return None, None
```

#### 3. 新增 Remote 傳輸邏輯 (AWS S3 + Presigned URL)

在你的 Agent 端（可能是在 `cloud_relay/aws_agent.py` 或獨立的 Cloud Sync Manager），處理來自遠端的拍照請求時，加入上傳 S3 與產生 URL 的邏輯。

Python

```
# cloud_relay/aws_s3_manager.py
import boto3
from botocore.exceptions import ClientError
import logging

class S3Manager:
    def __init__(self, bucket_name="rolex-moonlight-v2-images"):
        self.s3_client = boto3.client('s3')
        self.bucket_name = bucket_name

    def upload_and_get_url(self, local_filepath, s3_object_name=None, expiration=300):
        """
        上傳圖片至 S3 (可選用 KMS 加密)，並回傳 Remote Client 可直接下載的臨時 URL
        """
        if s3_object_name is None:
            s3_object_name = local_filepath.split('/')[-1]

        try:
            # 1. 上傳至 S3 備份 
            # (如果系統有高安全需求，建議啟用 S3 Bucket 的預設 Server-Side Encryption)
            self.s3_client.upload_file(local_filepath, self.bucket_name, s3_object_name)
            
            # 2. 產生 Presigned URL 給 Remote App 顯示
            response_url = self.s3_client.generate_presigned_url('get_object',
                                                    Params={'Bucket': self.bucket_name,
                                                            'Key': s3_object_name},
                                                    ExpiresIn=expiration)
            return response_url
        except ClientError as e:
            logging.error(e)
            return None

# Agent 端收到遠端拍照指令時的處理範例：
# filepath = server.process_command(cmd)['filepath']
# remote_view_url = s3_manager.upload_and_get_url(filepath)
# aws_iot_mqtt.publish(topic="app/response", payload={"image_url": remote_view_url})
```

### 總結

透過這個修改，你達到了三個目的：

1. **解決 AWS IoT 限制**：MQTT Payload 現在只有短短幾行字串（包含 URL），非常輕量。
    
2. **大幅提升 Local 效能**：免去 Base64 轉換，降低 CPU 負載，讓 Zaber 與 Keyence 響應更快。
    
3. **無縫接軌後續分析**：你的 UNet, SAM 演算法可以直接利用 `filepath` 寫 Batch Dataloader 載入硬碟圖片，不用在記憶體裡傳來傳去。



#### USW-pro-max-16-poe設定
```
我懷疑問題是不是出在USW-pro-max-16-poe沒有把主機跟camera連在一起因為它顯示physical camera none not found. 我只是把USW-pro-max-16-poe接上電源, 然後用網路線一個連接port到主機, 另個網路線連接port到camera. 沒有其他設定. 請幫我看主機能不能檢查這是否正確連接 或有其他方法可以測試?
```
這非常有可能就是根本原因！工業用的 GigE 網路相機（像是你的 ITALA）跟一般家用的 USB 網路攝影機不同，它們不能只是「插上網路線就好」。

因為你的架構裡**只有 Switch、電腦和相機，沒有 Router（路由器）**，這代表網路上沒有人負責發放 IP 位址（DHCP）。在這種情況下，電腦和相機會各自分配到隨機的無效 IP，導致它們雖然插在同一台 Switch 上，卻像是在不同的平行宇宙，完全無法互相溝通。

這絕對可以透過主機來檢查與修正。請按照以下順序進行排錯：

### 第一關：物理連線與 PoE 供電檢查

Unifi Switch Pro Max 16 PoE 是有供電能力的，但我們必須確認相機真的有吃到電。

1. **檢查 Switch 燈號**：確認連接著 PC 和 Camera 的那兩個 Switch Port 的 LED 燈號都有亮起（通常閃爍代表有資料傳輸）。
    
2. **檢查相機燈號**：ITALA 相機背面會有狀態指示燈。如果有亮，代表 PoE 供電成功；如果沒亮，請確認你把相機插在 Switch 上有支援 PoE 的 Port（通常會有閃電符號或標示 PoE+）。
    

### 第二關：設定主機的靜態 IP (Static IP)

這是最關鍵的一步。我們必須手動給電腦和相機指定「同一條街」的地址。

1. 打開 Windows 的 **設定 > 網路和網際網路 > 乙太網路**（選擇連接到 Unifi Switch 的那個網卡）。
    
2. 找到 **IP 指派**，點擊「編輯」，將「自動 (DHCP)」改為**「手動」**。
    
3. 開啟 **IPv4**，輸入以下設定：
    
    - **IP 位址**：`192.168.0.10`
        
    - **子網路遮罩**：`255.255.255.0`
        
    - （通訊閘和 DNS 可以留空）
        
4. 儲存設定。
    

### 第三關：用官方軟體「強制」配發 IP 給相機

現在電腦已經在 `192.168.0.x` 網段了，但相機的 IP 可能還是亂的。**請先不要用你的 Python 程式測試**，先用原廠工具建立連線。

1. 打開 Opto Engineering 隨 SDK 安裝的官方相機檢視軟體（通常叫做 **ITALA Viewer**、**eBUS Player** 或 **GenICam Browser**）。
    
2. 軟體開啟時，會掃描網路上的相機。它應該會看到這台 ITA81，但可能會顯示「不在同一網段 (Subnet mismatch)」或有紅色警告標誌。
    
3. 對著那台相機點擊右鍵，尋找 **"Force IP"**、**"Set IP Address"** 或 **"Network Configuration"** 的選項。
    
4. 強制將相機的 IP 設為：
    
    - **IP Address**: `192.168.0.11`
        
    - **Subnet Mask**: `255.255.255.0`
        
5. 設定套用後，嘗試在官方軟體中點擊「Connect」並開啟影像串流 (Play)。
    

### 第四關：Ping 測試

如果官方軟體連不上，你可以打開 Windows 的「命令提示字元 (cmd)」，輸入： `ping 192.168.0.11` 如果有收到回覆 (Reply from 192.168.0.11)，代表實體連線與 IP 設定都完美無缺，問題就在軟體端；如果顯示「要求等候逾時 (Request timed out)」，代表網路依然沒有通。




#### 為 Ethernet 2 設定靜態 IP
```
我重新換插主機另一個網路孔, 現在Ethernet 2沒有紅色叉叉了. 顯示Ethernet controller x710 for 10gbase
```
太棒了！這才是真正的資料傳輸網路孔！🎉

而且你插到的是 **Intel X710 10GBASE**，這是一張非常高階的 10GbE（萬兆）工業級網路卡！用來連接工業相機（GigE Vision）簡直是完美搭配，能提供極度穩定且高速的頻寬。

既然實體連線已經打通，我們現在就來為這個 `Ethernet 2` 設定專屬的靜態 IP，讓電腦跟相機能夠互相溝通。

### 步驟一：為 Ethernet 2 設定靜態 IP

1. 在 `ncpa.cpl` 畫面中，對著沒有紅叉叉的 **Ethernet 2** 點擊**右鍵**，選擇**「內容 (Properties)」**。
    
2. 在中間的清單裡向下捲動，找到並點擊 **「網際網路通訊協定第 4 版 (TCP/IPv4)」**（請點擊文字讓它反白，不要只勾選旁邊的框框），然後點擊右下角的**「內容 (Properties)」**。
    
3. 選擇 **「使用下列的 IP 位址」**，並精準輸入以下數值：
    
    - **IP 位址 (IP address)**：`192.168.1.10`
        
    - **子網路遮罩 (Subnet mask)**：`255.255.255.0`
        
    - **預設閘道 (Default gateway)**：_(完全留白)_
        
4. 下方的 DNS 伺服器也全部留白，然後點擊「確定」，再點擊一次「關閉」或「確定」退出設定。
    

_(此時 Windows 可能還是會顯示「未辨識的網路」，這是正常的，只要紅叉叉沒出現就好。)_

### 步驟二：透過 ITALA 原廠軟體喚醒相機

現在你的電腦已經住在 `192.168.1.10` 這個地址了，接下來我們要去把相機拉進同一個網段：

1. 請從 Windows 開始選單中，打開 Opto Engineering ITALA SDK 附帶的相機工具（通常叫做 **ITALA Viewer**、**eBUS Player** 或是 **GigE Configurator**）。
    
2. 軟體打開後，它會自動掃描網路。你應該會在清單中看到你的 **ITA81** 相機。
    
3. 此時相機的狀態可能會顯示「無法連接」或是「Subnet Mismatch（網段不符）」，甚至出現紅色的警告標誌。
    
4. 對著那台 ITA81 相機點擊**右鍵**，尋找名為 **"Force IP"**、**"Set Network Configuration"** 或 **"Assign IP"** 的選項。
    
5. 將相機的 IP 強制設定為：
    
    - **IP 位址**：`192.168.1.11`
        
    - **子網路遮罩**：`255.255.255.0`
        
6. 套用設定後，嘗試在軟體中點擊「Connect」或「Play」按鈕。
    

你可以順利在 ITALA 的原廠軟體裡面看到相機的畫面了嗎？如果看到了，我們就可以回到 Python 程式碼進行最後的測試！



#### 測試ita81跟liquid lens
```
我在itala view可以正確抓到ita81-gc-20c-el, 但當我按connect 他的畫面顯示全黑, 而且pixel format not supported for video recording. 因為我這個camera跟liquid lens TCEL350連接(而且確定蓋子都有打開), 然後有cable連接ita81跟TCEL350. 會不會liquid lens TCEL350出問題? 我把liquid lens TCEL350拔掉, 這樣itala view可以看到不是全黑, 但都是模糊的一片一點點亮. 所以要怎麼解決這問題
```
這是一個非常棒的交叉測試！你把鏡頭拔下來看到「模糊的一點點亮」，這個動作直接證明了**相機的感光元件 (Sensor) 是完全正常的，資料傳輸也完美無缺！**

你可以放心，**你的液態鏡頭 (TCEL350) 很高機率並沒有壞。** 現在你遇到的這兩個現象（Pixel format 警告、裝上鏡頭全黑），其實是工業相機非常典型的「軟體預設值」與「光學特性」問題。讓我來幫你一一破解：

### 1. 為什麼會顯示 "Pixel format not supported for video recording"？

這只是一個**軟體警告，完全不會導致黑畫面**。 工業相機為了保留最多細節，預設通常會輸出 10-bit 或 12-bit 的無壓縮格式（例如 `BayerRG10` 或 `Mono12`）。但是，ITALA Viewer 內建的「錄影 (Video Recording)」功能，只支援標準的 8-bit 格式。

- **如何消除警告：** 在 ITALA Viewer 的右側控制面板（Node Map 或 Feature Browser）中，找到 **`Image Format Control`**。
    
- 將 **`Pixel Format`** 改為 **`BayerRG8`**（如果是彩色相機）或是 **`Mono8`**（如果是黑白相機）。警告就會消失了。
    

---

### 2. 為什麼裝上 TCEL350 鏡頭會「全黑」？

這是工業用**遠心鏡頭 (Telecentric Lens)** 加上工業相機預設值的「雙重陷阱」：

**陷阱 A：曝光時間 (Exposure Time) 太短** 一般家用 WebCam 會自動調整曝光，但工業相機插上電時，曝光時間通常預設在極端的數值（例如 `100` 微秒，也就是萬分之一秒）。在這麼短的時間內，如果沒有打超強的閃光燈，畫面絕對是全黑的。

**陷阱 B：遠心鏡頭的光學特性 (進光量極小)** 你的 TCEL350 是一顆非常高階的遠心鏡頭。這種鏡頭的光圈非常小，而且它**「只接收完全平行的光線」**。這代表：

1. 一般辦公室的天花板日光燈，對它來說跟沒開燈一樣。
    
2. 即使你用手機手電筒照，如果角度不對，它也看不見。
    

液態鏡頭的排線只是用來控制「對焦 (Focus)」，它裡面沒有光圈葉片會卡住，所以絕對不是液態鏡頭把光擋住了。

### 🛠️ 解決步驟：讓畫面亮起來！

請在 ITALA Viewer 中進行以下設定：

1. **拉長曝光時間：**
    
    - 在右側設定面板找到 **`Acquisition Control`**。
        
    - 把 **`Exposure Auto`** 設為 **`Off`**。
        
    - 找到 **`Exposure Time`**，它現在的值可能很小。請大膽地把它改成 **`50000`** 或 **`100000`**（這代表 50 毫秒或 100 毫秒）。
        
2. **拉高增益 (Gain)：**
    
    - 在 **`Analog Control`** 中找到 **`Gain`**。
        
    - 把數值拉高到 **`10`** 或 **`15`**（這會讓畫面出現雜訊，但我們先求「有畫面」，之後再調降）。
        
3. **提供正確的光源 (最重要！)：**
    
    - 拿一個物體（例如手錶零件或一張名片）放在鏡頭正前方約 **幾公分** 的地方（TCEL350 有固定的工作距離，太遠或太近都會黑掉或糊掉）。
        
    - 拿你的手機手電筒，從鏡頭旁邊**用力且近距離地照亮那個物體**。
        

做了這三步，你的全黑畫面一定會突然出現影像！就算很模糊也沒關係，只要看到東西的輪廓或光影變化，就代表大功告成了。

試試看把 Exposure Time 拉到 `100000` 然後拿手機手電筒近距離照一下，畫面有亮起來嗎？

關於你想在 ITALA Viewer 裡面測試「自動測光 (Auto Light)」和「自動對焦 (Auto Focus)」，這在工業相機的世界裡，情況跟我們平常用的手機或 WebCam 不太一樣。讓我為你拆解這兩件事：

### 1. 自動測光 / 自動曝光 (Auto Exposure / Auto Light) 👉 **可以，但在工業應用不建議**

ITALA 相機的底層韌體確實有支援自動調整曝光時間來適應光線。

- **如何測試：** 在 ITALA Viewer 右側的屬性面板（Node Map）中，找到 **`Acquisition Control`**（影像擷取控制）。 將 **`Exposure Auto`**（自動曝光）從 `Off` 改為 **`Continuous`**（連續自動）或 **`Once`**（單次自動）。
    
- **⚠️ 陷阱警告：** 因為你的 TCEL350 是遠心鏡頭，進光量極小。如果你開啟 `Continuous`，相機為了讓畫面夠亮，可能會把曝光時間拉到超級長（例如 0.5 秒），這會導致你的畫面變得非常卡頓（Frame Rate 暴跌到 2 FPS 以下）。 **在你的 Moonlight AOI 系統中（如你寫的 `watch_gain_adjust4.m` 邏輯），工業檢測通常是「固定曝光時間 + 固定光源」，所以確認能手動調整曝光才是最重要的。**
    

### 2. 自動對焦 (Auto Focus) 👉 **Viewer 裡沒有這個按鈕，必須手動拉數值！**

這是一個非常重要的硬體觀念：**液態鏡頭 (Liquid Lens) 本身與相機韌體，是不具備「自動對焦演算法」的。** 相機不知道你現在看的是手錶指針還是灰塵，它只負責送出電流（mA）來改變液態鏡頭裡面的液體曲率。

- **如何測試液態鏡頭是否正常：** 在 ITALA Viewer 右側的屬性面板中，往下找找看有沒有名為 **`Analog Control`**、**`Lens Control`** 或是特定的 Opto Engineering 控制節點。 尋找名為 **`LensDriverCurrent`**、**`FocusPos`** 或是 **`OptotuneFocus`** 的選項（這正是你寫在 `hardware_managers.py` 第 232 行用來控制鏡頭的變數）。 你會看到一個數值拉桿（Slider）。**當你手動拖拉這個拉桿改變數值時，你會看到畫面在「模糊 ➔ 清晰 ➔ 模糊」之間變化。** 如果畫面焦距會跟著拉桿改變，就代表液態鏡頭作動完美！
    

---

### 🚀 下一步：回到你的 Python 程式

既然 ITALA Viewer 已經幫我們證實了硬體 100% 沒問題，我們就可以回到你的系統了！

你的 `Controller/test_hardware/test_camera_ui.py` 其實已經寫得非常棒了，裡面完美包含了你現在需要的所有功能：

- 它有 **Exposure (曝光)** 的拉桿。
    
- 它有 **Liquid Focus (液態鏡頭對焦)** 的拉桿。
    

**最後一個關鍵動作：** 因為工業相機（GigE Vision）**一次只能被一個軟體佔用**。請你現在：

1. **關閉 ITALA Viewer**（必須完全關閉，釋放相機控制權）。
    
2. 打開你的終端機 / 命令提示字元，啟用你的 conda 環境（`watch_env`）。
    
3. 執行你的測試程式：`python Controller/test_hardware/test_camera_ui.py`
    
4. 選擇你的 `micro_cam`，按下 **Start Stream**。
    

你應該就能在你自己寫的 PyQt 介面中，透過拉動 `Liquid Focus` 拉桿，親眼看著手錶零件變清晰了！趕快試試看你的 Python 介面能不能順利出圖！