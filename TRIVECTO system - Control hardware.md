


|                                         |     |
| --------------------------------------- | --- |
| [[#### 目前有的Control hardware code跟修改建議]] |     |
| [[#### Base64 JSON 傳圖是否該修正?]]           |     |
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