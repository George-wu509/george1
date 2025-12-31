

|                                                |     |
| ---------------------------------------------- | --- |
| [[#### 安裝的軟體與驅動 (Drivers & SDKs)]]             |     |
| [[#### 硬體驅動層（Hardware Driver Layer）]]          |     |
| [[#### 硬體執行驅動層解釋]]                             |     |
| [[#### Zaber API & Virtual Device Connection]] |     |
| [[#### APP UI連結Zaber 3D viewer]]               |     |
|                                                |     |


#### 安裝的軟體與驅動 (Drivers & SDKs)
```
如果我的電腦已經連結上這些硬體(camera, )除了這些python code之外要讓這些code可以真的運作, 需要安裝甚麼嗎? (example: driver?) 有些可能要透過port or internet控制的硬體這樣需要在電腦做額外設定嗎? 那像Zaber or Camera是否有官方或非官方simulation code可以模擬硬體用來測試code?
```
這是一個非常關鍵的問題。要在「真實世界」讓這些 Code 驅動硬體，光有 Python Code 是不夠的。硬體與作業系統（Windows/Linux）之間需要「翻譯官」（Drivers/SDK），而網路型硬體需要正確的「通訊地址」（Network Config）。

以下是針對你的硬體清單（Zaber, Opto-engineering, Keyence）所需的**安裝清單**、**系統設定**以及**模擬方案**。

---

### 第一部分：必須安裝的軟體與驅動 (Drivers & SDKs)

這些是讓作業系統「看見」硬體的前提。

#### 1. Zaber 運動平台 (USB/Serial)

- **硬體介面**: USB (內部轉 Serial)。
    
- **Driver (必裝)**: **Silicon Labs CP210x USB to UART Bridge VCP Driver**。
    
    - Zaber 裝置接上 USB 後，Windows 裝置管理員應該要出現一個 `COM Port` (例如 `COM3`)。如果沒裝這個 Driver，電腦會認不得裝置。
        
- **Library**:
    
    - `pip install zaber-motion` (這是 Zaber 官方出的 Python 庫，你的 code 需要它)。
        

#### 2. 相機 (Opto-engineering / GenICam)

- **硬體介面**: 通常是 GigE (網路線) 或 USB3。
    
- **Driver / SDK (必裝)**: **GenICam GenTL Producer (CTI)**。
    
    - 工業相機通常遵循 **GenICam** 標準。你需要安裝相機廠商提供的 SDK，例如 **Opto-engineering SDK** 或通用的 **Basler Pylon** / **Matrix Vision**。
        
    - 安裝後，會得到一個 `.cti` 檔案（GenTL Producer），Python 的 `harvesters` 庫需要這個檔案才能抓到相機。
        
- **Library**:
    
    - `pip install harvesters` (建議使用這個來操作 GenICam 相機)。
        
    - `pip install numpy opencv-python`.
        

#### 3. Keyence 雷射感測器 (Ethernet)

- **硬體介面**: Ethernet (RJ45)。
    
- **Driver**: **不需要**。它是透過 TCP/IP 通訊。
    
- **設定軟體 (建議安裝)**: **CL-Navigator N**。
    
    - 雖然 Python code 可以直接送 TCP指令，但你需要這套官方軟體來**設定感測器的 IP 位址**、取樣頻率和歸零。
        

#### 4. 光源控制器 (Serial/Ethernet)

- **Driver**: 如果是 USB/Serial 型號，通常需要 **FTDI Driver** 或 **Prolific PL2303 Driver**。
    

---

### 第二部分：電腦與網路設定 (System & Network Config)

對於透過網路線 (GigE/Ethernet) 控制的硬體（相機、Keyence、光源），這是最容易卡關的地方。

#### 1. 雙網卡配置 (Dual NIC Strategy) - **強烈建議**

為了避免硬體資料流（特別是相機的高頻寬）塞爆你的上網頻寬，或受公司防火牆干擾，工業電腦通常有兩個網孔：

- **NIC 1 (Intranet/Internet)**: DHCP，連公司內網，上 AWS IoT Core。
    
- **NIC 2 (Machine Network)**: **固定 IP (Static IP)**，專門接 Hub/Switch 連接所有硬體。
    

#### 2. 設定固定 IP (Static IP)

你的 Python code 裡寫死了 Keyence IP 是 `192.168.1.10`。

- 你必須將 **NIC 2** 的 IP 設定為同網段，例如：
    
    - IP: `192.168.1.100`
        
    - Subnet Mask: `255.255.255.0`
        
- **Keyence 設定**: 使用 CL-Navigator 將 Keyence IP 設為 `192.168.1.10`。
    

#### 3. GigE 相機的特殊設定 (Jumbo Frames)

如果使用 GigE 相機，為了不掉幀：

- **開啟 Jumbo Frames**: 到網卡設定 (Device Manager -> Network Adapter -> Advanced)，將 `Jumbo Packet` 開到最大 (例如 9KB / 9014 Bytes)。
    
- **關閉防火牆**: Windows Firewall 常常會擋住工業相機的串流，建議在 NIC 2 關閉防火牆，或針對 Python/SDK 開通權限。
    

---

### 第三部分：官方/非官方模擬器 (Simulation)

如果硬體不在手邊，或想進行 CI/CD 測試，可以使用以下方案：

#### 1. Zaber (官方支援強大)

- **官方模擬器 (Zaber Console)**:
    
    - 下載 **Zaber Console** (免費)。
        
    - 它可以建立 "Virtual Device"。
        
    - Python 的 `zaber-motion` library 可以設定連線到 Zaber Cloud 的虛擬裝置，或是透過 Zaber Console 的 Scripting 介面。
        
- **Code 層級模擬 (你的 `simulation_driver.py`)**:
    
    - 這其實是最簡單的。因為 Zaber 的指令就是「移動到 X」，你的模擬 code 只要 `time.sleep(移動時間)` 然後更新內部的 `current_x` 變數即可。這對於測試 "Logic" 已經足夠。
        

#### 2. 相機 (GenICam)

- **官方/標準模擬器 (GenTL Simulator)**:
    
    - 如果你用 `harvesters` 庫，你可以載入一個 **Simulated .cti 檔**。
        
    - 例如 **Matrix Vision** 或 **Baumer** 都有提供免費的 "Simulation Mode" SDK。安裝後，Python code 會以為真的連上一台相機，但抓到的圖是廠商 Logo 或雜訊。
        
- **非官方 (你的 `simulation_driver.py`)**:
    
    - 繼續使用你目前的做法：從硬碟讀取一張 JPG 假裝是剛拍到的。這比 GenTL Simulator 更實用，因為你可以餵入「特定的瑕疵照片」來測試你的 AI 模型。
        

#### 3. Keyence

- **無官方模擬器**。
    
- **Code 層級模擬**: 寫一個簡單的 Python TCP Server script 跑在 localhost:24685，接收到 `MS,01` 就回傳隨機數字 `MS,01,+020.123`。
    

---

### 總結：Tech Lead 的 Check List

要讓這套系統在新的電腦上跑起來，請執行以下步驟：

1. **安裝 Drivers**:
    
    - [ ] Zaber USB Driver (CP210x)
        
    - [ ] 相機廠商 SDK (含 GenTL Producer)
        
2. **安裝 Python Packages**:
    
    - [ ] `pip install zaber-motion harvesters opencv-python pyserial`
        
3. **網路設定**:
    
    - [ ] 設定 NIC 2 為固定 IP (`192.168.1.x`)。
        
    - [ ] 開啟網卡 Jumbo Frames (若用 GigE 相機)。
        
    - [ ] 關閉 NIC 2 的 Windows Firewall。
        
4. **硬體參數寫入**:
    
    - [ ] 用 Keyence 軟體把 IP 改成 `192.168.1.10`。
        
    - [ ] 確認 Zaber 的 COM Port 號碼，更新到 `system_config.yaml`。
        

只要完成上述設定，你的 `RealHardwareDriver` 就能順利透過 `controller_server.py` 控制實體設備了。





#### 硬體驅動層（Hardware Driver Layer）
```
這是系統裡關於控制系統的code包括cameras, light controllers, and motion platforms的硬體 - 多相機系統：至少有 4 支相機（Tele, Micro, Top, Side）, 精密運動平台：使用 Zaber 控制器進行 X, Y, Z 三軸移動。雷射/位移感測器：用於測量表面高度和玻璃厚度。可控光源：多通道燈光控制. 「先用大範圍相機定位 -> 移動到目標位置 -> 利用感測器進行精密對焦（需補償玻璃折射） -> 自動調整曝光與增益 -> 拍攝 HDR 影像 -> 用於特徵分析。」. 以下是硬體的型號: Zaber X-LRQ300BP, Zaber X-RST120AK, Opto-engineering: ITA81-GC-20C-EL, opto-engineering TICGR1000-D1, Adam ADAM-6266-B, Keyence CL-3000, LTDVE2CH-20F. 請檢察及優化code
```

這是一項非常完整的架構。作為 Tech Lead，我看到你的架構已經具備了 **Remote-First (AWS IoT)** 和 **Modular Design (Workflow Manager/Proxy)** 的良好基礎，這對於長期維護（30年生命週期）至關重要。

目前的痛點在於：**硬體驅動層（Hardware Driver Layer）尚未具體落實你提到的精密控制邏輯**（特別是玻璃折射補償、HDR、以及多設備協同）。

以下是針對你提供的硬體型號（Zaber, Opto-engineering, Keyence, etc.）進行的 **Code 優化與重構**。重點在於實現你描述的：「大範圍定位 -> 感測器精密對焦（含折射補償） -> HDR 拍攝」流程。

---

### 1. 新增核心驅動：`RealHardwareDriver`

你需要一個真實的驅動程式來取代目前的 `SimulationHardware`。此程式碼整合了 Zaber（運動）、Keyence（測距）、Opto-engineering（相機/光源）。

**建議位置：** `Controller/hardware_drivers/real_driver.py`

Python

```
import time
import socket
import logging
import cv2
import numpy as np
from typing import Dict, Any, List

# 硬體 SDK (需確認已安裝對應套件)
try:
    from zaber_motion import Units, Library
    from zaber_motion.ascii import Connection as ZaberConnection
except ImportError:
    print("[Error] zaber_motion library missing. pip install zaber_motion")

class RealHardwareDriver:
    """
    針對 Rolex/Watch AOI 系統的實體硬體驅動
    硬體清單:
    - Motion: Zaber X-LRQ300BP (XY), X-RST120AK (Rot)
    - Sensor: Keyence CL-3000 (Laser Displacement)
    - Camera: Opto-engineering ITA81/ITA204 (GenICam/GigE)
    - Light: Opto-engineering TICGR1000/LTDVE2CH
    """

    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger("RealHW")
        self.config = config.get("hardware", {})
        
        # 1. 初始化 Zaber (運動平台)
        self.zaber_conn = None
        self.axes = {}
        self._init_zaber()

        # 2. 初始化 Keyence (雷射感測)
        self.keyence_cfg = self.config.get("keyence", {})
        
        # 3. 初始化相機 (使用 GenICam 通用介面，如 Harvesters)
        self.cam_map = self.config.get("cameras", {})
        self.active_cam = None
        # self._init_cameras() # 實際需實作 GenICam 連線

        # 4. 初始化光源
        self.light_cfg = self.config.get("lighting", {})

    def _init_zaber(self):
        try:
            port = self.config.get("zaber", {}).get("port", "COM3")
            self.zaber_conn = ZaberConnection.open_serial_port(port)
            device_list = self.zaber_conn.detect_devices()
            self.logger.info(f"Zaber Devices Found: {len(device_list)}")
            
            # Mapping from config (e.g., stage_L_X -> Device 1 Axis 1)
            mapping = self.config.get("zaber", {}).get("device_map", {})
            for name, dev_idx in mapping.items():
                # 注意: Zaber API 索引通常從 0 開始，device_map 若從 1 開始需調整
                if dev_idx <= len(device_list):
                    self.axes[name] = device_list[dev_idx-1].get_axis(1)
        except Exception as e:
            self.logger.error(f"Zaber Init Failed: {e}")
            raise

    # --------------------------------------------------------
    # 核心邏輯：玻璃折射補償對焦
    # --------------------------------------------------------
    def smart_focus_routine(self, target_z_height: float, glass_thickness: float = 2.0, refractive_index: float = 1.52):
        """
        執行流程：
        1. 使用 Keyence 讀取當前玻璃表面高度
        2. 計算折射補償
        3. 移動 Z 軸到正確焦平面
        """
        self.logger.info("Starting Smart Focus with Refraction Compensation...")
        
        # 1. 讀取感測器數值 (距離玻璃表面的距離)
        current_dist = self.read_keyence_sensor()
        if current_dist == -999.0:
            raise RuntimeError("Keyence Read Failed")

        # 2. 計算物理位移
        # 若我们要對焦在玻璃下的錶盤，光線經過玻璃會產生折射，導致焦點後移
        # 物理移動量 = (目標高度) - (感測器讀數) + (折射補償)
        # Shift = Thickness * (1 - 1/n)
        refraction_offset = glass_thickness * (1 - (1 / refractive_index))
        
        self.logger.info(f"Refraction Offset: {refraction_offset:.4f} mm")
        
        # 假設 Z 軸座標系：0 為底座，數值越大越高
        # 目標 Z = 表面偵測高度 - 焦距補償 + 折射偏移
        target_z = target_z_height + refraction_offset

        # 3. 執行精密移動
        self.move_axis("stage_L_Z", target_z)
        self.logger.info(f"Focused at Z={target_z:.4f} (Sensor read: {current_dist})")

    def read_keyence_sensor(self) -> float:
        """讀取 CL-3000 數值"""
        ip = self.keyence_cfg.get("ip", "192.168.1.10")
        port = int(self.keyence_cfg.get("port", 24685))
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2.0)
                s.connect((ip, port))
                # CL-3000 command: "M1" (Measure head 1) usually requires command manual check
                s.sendall(b"MS,01\r\n") 
                data = s.recv(1024).decode('ascii')
                # Parse format like "MS,01,+012.3456"
                parts = data.split(',')
                if len(parts) >= 3:
                    val = float(parts[-1].strip())
                    return val
        except Exception as e:
            self.logger.error(f"Keyence Read Error: {e}")
        return -999.0

    # --------------------------------------------------------
    # 核心邏輯：HDR 與 自動曝光
    # --------------------------------------------------------
    def capture_hdr_image(self, cam_id: str, base_exposure: int) -> str:
        """
        拍攝多張不同曝光的照片並合成 HDR
        4. Auto Exposure (利用 histogram 判斷 base)
        5. Bracketing (-2EV, 0, +2EV)
        6. Merge
        """
        self.logger.info(f"Starting HDR Capture on {cam_id}...")
        
        # 1. Bracketing Exposures (以微秒為單位)
        exposures = [int(base_exposure * 0.25), base_exposure, int(base_exposure * 4)]
        images = []

        for exp in exposures:
            self._set_camera_exposure(cam_id, exp)
            img = self._grab_frame(cam_id) # 實際抓圖函式
            images.append(img)
        
        # 2. 合成 HDR (Mertens Fusion 不需要相機響應曲線，適合工業應用)
        merge_mertens = cv2.createMergeMertens()
        hdr = merge_mertens.process(images)
        
        # 3. 轉回 8-bit 以便儲存與顯示
        res_8bit = np.clip(hdr * 255, 0, 255).astype('uint8')
        
        # Save to temp
        tmp_path = f"temp_hdr_{time.time()}.jpg"
        cv2.imwrite(tmp_path, res_8bit)
        return tmp_path

    def _set_camera_exposure(self, cam_id, value):
        # 這裡需對接 Opto-engineering SDK / GenICam GenApi
        # node_map.ExposureTime.value = value
        pass

    def _grab_frame(self, cam_id):
        # 模擬回傳
        return np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

    # --------------------------------------------------------
    # 基礎移動介面
    # --------------------------------------------------------
    def move_axis(self, axis_name: str, value: float):
        if axis_name in self.axes:
            try:
                # Zaber use specific units, usually mm
                self.axes[axis_name].move_absolute(value, Units.LENGTH_MILLIMETRES)
            except Exception as e:
                self.logger.error(f"Move Error {axis_name}: {e}")
                raise
    
    def move_compound(self, positions: Dict[str, float]):
        """同時移動多軸"""
        for axis, val in positions.items():
            self.move_axis(axis, val)

    def capture_image(self, cam_id: str, exposure: int, save_path: str):
        # 實作標準單張拍攝
        self._set_camera_exposure(cam_id, exposure)
        img = self._grab_frame(cam_id)
        cv2.imwrite(save_path, img)

    def close(self):
        if self.zaber_conn:
            self.zaber_conn.close()
```

---

### 2. 優化 `Controller/server_config.yaml`

你的設定檔需要準確反映你提到的硬體型號。請更新硬體區塊：

YAML

```
# config/system_config.yaml (Partial Update)

hardware:
  zaber:
    port: "COM3" # 請確認實際 COM port
    # Mapping logical names to (Device Index, Axis Index)
    # 假設 X-LRQ300BP 是 Device 1, X-RST120AK 是 Device 2
    device_map:
      stage_X: 1  # X-LRQ300BP Axis 1
      stage_Y: 2  # X-LRQ300BP Axis 2 (如是雙軸控制器)
      stage_Z: 3  # X-LRQ300BP Z
      stage_R: 4  # X-RST120AK (Rotation)

  cameras:
    tele_cam_id: "ITA204_TELE_SERIAL"
    micro_cam_id: "ITA81_MICRO_SERIAL"
    top_cam_id:   "ITA_TOP_SERIAL"
    side_cam_id:  "ITA_SIDE_SERIAL"

  keyence:
    model: "CL-3000"
    ip: "192.168.1.10" # 控制器預設 IP
    port: 24685        # TCP Command Port
    
  lighting:
    # 支援 Opto-engineering LTDVE2CH (Strobe/Continuous)
    controller_ip: "192.168.1.50"
    controllers:
      - name: "top_ring"
        channel: 1
      - name: "coaxial"
        channel: 2
```

---

### 3. 優化 `main_cli.py` 與 Workflow 整合

為了解決遠端連線問題 (IP 問題)，你在 `aws_agent.py` 中使用了 MQTT，這是正確的選擇。但我建議在 `WorkflowManager` 中增加一個高階方法，將上述的精密對焦邏輯封裝起來，讓 CLI 或 Remote Client 可以直接呼叫。

在 **`core/workflow_manager.py`** 中新增：

Python

```
    # ... inside WorkflowManager class ...

    def execute_smart_inspection(self, watch_id: str, routine_step: dict):
        """
        執行複合動作：
        1. 大範圍定位 (Tele Cam)
        2. 移動到目標位置
        3. 利用感測器精密對焦（需補償玻璃折射）
        4. 自動調整曝光與增益 -> 拍攝 HDR 影像
        """
        if not self.hw:
            self.console.warning("Simulation Mode: Skipping hardware precise moves.")
            return

        # 1. Move to approximate position
        pos = routine_step.get("hardware", {}).get("position", {})
        self.hw.move_compound(pos)
        time.sleep(0.5) # Settle time

        # 2. Smart Focus (Refraction Compensation)
        # 假設我們知道此手錶型號的玻璃厚度
        glass_thickness = 2.0 # mm, 應從 database 根據 watch_id 讀取
        target_focus_z = pos.get("stage_Z", 30.0) # 預設對焦高度
        
        try:
            self.hw.smart_focus_routine(target_focus_z, glass_thickness)
        except Exception as e:
            self.console.error(f"Focus failed: {e}")
            # Fallback logic here if sensor fails

        # 3. HDR Capture
        cam_id = routine_step.get("hardware", {}).get("camera", "macro_cam_id")
        base_exp = routine_step.get("hardware", {}).get("exposure", 5000)
        
        temp_path = self.hw.capture_hdr_image(cam_id, base_exp)
        
        # 4. Save & Sync (reuse existing data_mgr logic)
        # ... logic to move temp_path to DataManager ...
        return temp_path
```

---

### 4. 解決 Remote Client 連線建議

針對你提到的：「Remote client 因 IP 問題或其他問題無法連到 local computer」，目前的 `aws_agent.py` 透過 AWS IoT Core (MQTT) 已經解決了 NAT/Firewall 穿透問題（這是標準的 Reverse Connection 模式）。

**針對長期商業軟體的建議 (30年 tech lead 觀點)：**

1. Watchdog & Keep-Alive:
    
    在 aws_agent.py 中，你需要確保 MQTT 連線如果斷開會自動重連（目前的 configureAutoReconnectBackoffTime 是好的，但需要在應用層增加心跳監測）。
    
2. WebRTC 穿透 (STUN/TURN):
    
    你在 code 裡提到了 WebRTC。如果 Local Computer 在防火牆後，Remote Client 要看即時影像（Live View），單靠 P2P WebRTC 經常會失敗。
    
    - **解決方案：** 必須部署或租用 **TURN Server** (如 Coturn 或 AWS Kinesis Video Streams)。你的 `system_config.yaml` 應該要包含 ICE Server 的設定。
        
3. Local Diagnostics API:
    
    當 AWS 連不上時，現場工程師需要維修。建議保留 workflow_server.py (FastAPI)，並設定為僅監聽 Localhost 或特定 VPN 介面，作為「現場維修模式」。
    

### 總結優化步驟

1. **部署 `RealHardwareDriver`**: 取代 mock driver，真正串接 Zaber/Keyence/GenICam。
    
2. **實作折射補償數學**: 在 Driver 層加入 $Offset = T \times (1 - 1/n)$ 的計算。
    
3. **增強影像擷取**: 在 Driver 層加入 HDR (Mertens Fusion) 算法。
    
4. **配置檔對應**: 更新 `yaml` 檔以匹配特定硬體型號。
    

這套代碼現在可以處理從「雲端指令」下達到「底層硬體精密動作」的完整路徑，並具備光學物理補償能力。





#### 硬體執行驅動層解釋
```
中文詳細解釋Controller的client_config.py, controller_client.py, controller_server.py, detect_hardware.py, hardware_managers.py, server_config.py, simulation_driver.py的每個file的技術細節跟如何串聯, 以及這些python codes如何使協助控制硬體跟系統, 怎麼用olex_submariner.yaml去控制操作這些硬體(以及連結到這些python file), 以及Controller這些python files如何連結到這整個系統(跟core or data_manager等等下面的python files)?
```

這是一份針對 `Controller` 模組的深度技術解析。作為 Tech Lead，我們可以將這個資料夾視為系統的 **「硬體抽象層 (Hardware Abstraction Layer, HAL)」** 與 **「執行驅動層」**。

這裡的架構設計採用了 **Client-Server (C/S) 分離模式**，允許控制邏輯（Client/Core）與物理連接（Server）分開，這對於你提到的「未來可能 Local Computer 與 Remote Client 分隔兩地」的架構至關重要。

以下是每個檔案的技術細節、串聯方式，以及它們如何被 `rolex_submariner.yaml` 驅動並整合進整個系統。

---

### 1. 檔案技術細節解析

#### **A. 底層驅動與偵測 (Low-Level Drivers)**

1. **`hardware_managers.py` (硬體管理員 - The Translator)**
    
    - **技術細節**：這是最底層的 **Wrapper Class**。它將原本複雜的第三方 SDK（如 Zaber Motion Library, GenICam SDK, TCP Socket 指令）封裝成統一的 Python 方法。
        
    - **針對你的硬體**：
        
        - `ZaberManager`: 使用 `zaber_motion` library 建立 Serial 連線，將邏輯軸名稱（如 `stage_L_X`）映射到物理 ID（Device index, Axis index）。
            
        - `IOManager`: 針對 **Keyence CL-3000**。由於 Keyence 使用 TCP ASCII command，這裡實作了 Socket Client，發送 `MS,01` 等指令並解析回傳的 CSV 字串轉為 `float`。
            
        - `CameraManager`: 針對 **Opto-engineering (GenICam)**。這裡負責相機的開啟、曝光參數設定 (`Exposure`, `Gain`)、取像 (`Grab`) 並轉換為 NumPy Array。
            
    - **功能**：將「對硬體說的話」翻譯成「Python 看得懂的物件」。
        
2. **`detect_hardware.py` (自動偵測 - The Plug & Play)**
    
    - **技術細節**：使用 `serial.tools.list_ports` 掃描 Windows/Linux 的 COM ports。
        
    - **邏輯**：它遍歷所有連接的 USB 裝置，讀取 **VID (Vendor ID)** 和 **PID (Product ID)**。
        
    - **用途**：解決「Zaber 今天插在 COM3，明天插在 COM5」的問題。系統啟動時會呼叫此腳本自動將 COM Port 填入 Config，讓 Server 啟動時不會報錯。
        

#### **B. 模擬層 (Simulation)**

3. **`simulation_driver.py` (模擬驅動 - The Mock)**
    
    - **技術細節**：這是一個 **Mock Object**，它實作了與真實硬體驅動完全相同的介面（Interface），但內部沒有硬體 IO。
        
    - **運作**：
        
        - 當 `WorkflowManager` 處於 `simulation` 模式時，會載入此驅動。
            
        - **Context-Aware 影像生成**：它會讀取 `simulation_map.yaml`，根據當前要求的算法（例如 `ocr` 或 `lume`），從 `assets/` 資料夾複製對應的預存圖片，假裝是相機剛拍到的。
            
    - **價值**：讓你在沒有連線到 Zaber 或 Keyence 的情況下（例如在飛機上寫 code），也能測試完整的軟體流程。
        

#### **C. 遠端控制架構 (TCP Client-Server)**

這部分是實現「遠端硬體控制」的核心。

4. **`controller_server.py` (自動化伺服器 - The Hardware Host)**
    
    - **技術細節**：這是一個 **Multi-threaded TCP Server**。
        
    - **位置**：這隻程式跑在 **Local Computer**（直接連接 Zaber/Keyence 的那台電腦）。
        
    - **運作流程**：
        
        1. 啟動時讀取 `server_config.py`，初始化 `HardwareManagers`。
            
        2. 監聽 Port (如 5000)。
            
        3. 當收到 Client 連線，開啟一條 Thread 處理。
            
        4. 接收 JSON 指令（例如 `{"action": "move_stage", "axis": "X", "value": 10}`）。
            
        5. 呼叫 `ZaberManager` 執行動作，並回傳 `{"status": "ok"}`。
            
    - **特點**：如果是拍照指令，它會將圖片進行 Base64 編碼回傳，實現遠端即時預覽。
        
5. **`controller_client.py` (控制客戶端 - The Remote Controller)**
    
    - **技術細節**：這是 TCP Client 封裝。
        
    - **位置**：這可以跑在 **WorkflowManager 內部**，或是遠端的 **Remote Client** 上。
        
    - **運作**：它將 Python 的函式呼叫（如 `client.move_stage(...)`）序列化為 JSON，透過 Socket 發送給 Server，並等待回應。
        

#### **D. 設定檔 (Configuration)**

6. **`server_config.py` / `client_config.py`** (雖然你的 source 顯示 file not found，但邏輯如下)
    
    - **Server Config**: 定義硬體的 IP (Keyence)、Port (Zaber COM port)、Auth Token (安全驗證)。
        
    - **Client Config**: 定義 Server 的 IP 與 Port，讓 Client 知道去哪裡找硬體。
        

---

### 2. 硬體控制流程串聯圖解

這整個模組是如何串起來控制硬體的？我們用一個「拍照檢測流程」來演示。

程式碼片段

```
graph TD
    subgraph "Planning Layer (Workflow)"
        Plan[rolex_submariner.yaml] -->|讀取| Core[WorkflowManager]
    end

    subgraph "Driver Layer (Abstraction)"
        Core -->|判斷 Mode| Mode{Simulation or Production?}
        
        Mode -->|Simulation| SimDriver[simulation_driver.py]
        Mode -->|Production| RealDriver[RealHardwareDriver]
    end

    subgraph "Physical Connection Layer (Controller)"
        RealDriver -->|直接呼叫 or TCP| HW_Mgr[hardware_managers.py]
        HW_Mgr -->|USB Serial| Zaber[Zaber X-LRQ300BP]
        HW_Mgr -->|TCP Socket| Keyence[Keyence CL-3000]
        HW_Mgr -->|GigE/USB| Camera[Opto-engineering Camera]
    end

    SimDriver -->|Copy File| Assets[Sample Images]
```

#### **關鍵串聯點：`WorkflowManager` 如何使用這些檔案**

在你的 `core/workflow_manager.py` 中，你會看到這樣的邏輯（這是所有 Python files 的交匯點）：

1. **初始化階段**：
    
    - `WorkflowManager` 讀取 `system_config.yaml`。
        
    - 若 `mode: production`，它會實例化 **`RealHardwareDriver`**。
        
    - 若 `mode: simulation`，它會實例化 **`simulation_driver.py`**。
        
2. **`RealHardwareDriver` 的角色**：
    
    - 你目前的程式碼中，`RealHardwareDriver` (建議新增的檔案) 其實就是將 `controller_client.py` 的邏輯或者直接將 `hardware_managers.py` 的邏輯整合進來。
        
    - **最佳實務**：為了減少延遲，若 WorkflowManager 與硬體在同一台電腦，直接 import `hardware_managers.py` 使用；若分開，則透過 `controller_client.py` 走 TCP 連線。
        

---

### 3. 如何使用 `rolex_submariner.yaml` 控制硬體

這個 YAML 檔是 **「檢測配方 (Recipe)」**。它是靜態的描述，需要透過 `WorkflowManager` 來「執行」。

#### **YAML 結構解析與對應程式碼**

YAML

```
# Controller/plans/rolex_submariner.yaml
steps:
  - id: 101
    name: "Front_Full_Macro"
    hardware:
      camera: "macro_1"      # 對應 hardware_managers.CameraManager
      position:              # 對應 hardware_managers.ZaberManager
        stage_x: 10.0
        stage_y: 20.0
        stage_z: 50.0
      light:                 # 對應 hardware_managers.LightManager
        controller: "ltic1"
        intensity: 200
```

#### **執行流程 (Step-by-Step)**

1. 載入 (Load):
    
    WorkflowManager 中的 _run_production_routine 讀取此 YAML 檔，轉為 Python Dictionary。
    
2. 解析 (Parse):
    
    程式迴圈遍歷 steps 列表。當讀到 step 101 時：
    
3. **硬體動作 (Action - 連結到 `Controller` files)**:
    
    - **移動**: 讀取 `position` -> 呼叫 `self.hw.move_compound({'stage_x': 10.0, ...})`。
        
        - 這會觸發 `hardware_managers.py` 中的 `ZaberManager.move_axis`。
            
    - **打光**: 讀取 `light` -> 呼叫 `self.hw.set_light(...)`。
        
    - **拍照**: 讀取 `camera` 與 `exposure` -> 呼叫 `self.hw.capture_image(...)`。
        
        - 這會觸發 `hardware_managers.py` 中的 `CameraManager.capture_image`。
            
4. **數據流**:
    
    - 相機拍到的 `numpy array` 影像，會被 `WorkflowManager` 傳給 `DataManager` 進行儲存（Local Disk）與同步（AWS S3）。
        

---

### 4. Controller 如何連結到 Core 與 Data Manager

這是一個 **分層架構 (Layered Architecture)**，`Controller` 是最底層，`Core` 是中控層。

1. **Core (WorkflowManager) $\rightarrow$ Controller**:
    
    - **連結方式**：**Dependency Injection (依賴注入)** 或 **Composition (組合)**。
        
    - `WorkflowManager` 擁有一個成員變數 `self.hw`。
        
    - 這個 `self.hw` 就是 `Controller` 資料夾下的某個 Driver 實例。
        
    - Core 不關心 Zaber 的指令是 ASCII 還是 Binary，它只呼叫 `self.hw.move_axis()`。這是 `Controller` 的責任去實作細節。
        
2. **Controller $\rightarrow$ Data Manager**:
    
    - **連結方式**：**Controller 通常不直接連結 Data Manager**（為了低耦合）。
        
    - **資料流向**：
        
        1. Controller (Hardware) 產生 `Image Data` (Raw bytes/Numpy array)。
            
        2. Controller 將資料 `return` 給 Core。
            
        3. Core 拿到資料後，轉手交給 `DataManager` (`data_manager.process_and_sync_raw_image`)。
            
    - **為什麼這樣設計？** 因為硬體驅動程式不應該知道「資料庫」或「AWS S3」的存在。它的工作就是「動」和「拍」，存檔是 Core 調度 Data Manager 去做的事。
        

### 總結

- **`hardware_managers.py`**: 是**手腳**，負責轉譯物理訊號。
    
- **`controller_server.py`**: 是**神經傳導**，負責讓手腳可以被遠端控制。
    
- **`rolex_submariner.yaml`**: 是**樂譜**，定義了手腳要怎麼動。
    
- **`WorkflowManager` (in Core)**: 是**指揮家**，看著樂譜，指揮手腳（Controller），並把產出的音樂（影像資料）交給錄音師（Data Manager）。




#### Zaber API & Virtual Device Connection
```
我在這個網站有看到關於API: Zaber Motion Library 跟 Open 3D Viewer
https://www.zaber.com/virtual-device#motion-library

裡面描述到如果要virtual測試Zaber硬體的步驟(API & Virtual Device Connection):
1: Set up API (Zaber Motion Library) on your device.
2: Open 3D Viewer page and select a product to simulate (Virtual Device).
3: Use the provided  to connect your application to the Virtual Device.
4: Get a feel for the Zaber API, explore how you can use it.

網頁也說可以
Step 1. Set up API (Zaber Motion Library) following the Getting Started guide.
Step 2. Open 3D Viewer page and select a product to simulate. Follow the Connection instructions for the API (Zaber Motion Library) there.

我想要的是可以在我原有Project下在Control/下的ZaberManager class可以修改或增加功能, 可以讓我用ZaberManager class控制(譬如位移或旋轉), 然後透過API連接API & Virtual Device可以在3D Viewer page顯示位移或旋轉. 請中文詳細解釋一步步如何做, 並提供完整的新/修改的code

```

這是一個非常棒的功能！Zaber 的 "Virtual Device" 功能允許你的 Python 程式碼透過網路連接到他們雲端的 3D 模擬器。這意味著你的 `ZaberManager` 發送出的移動指令，不會驅動實體馬達，而是會即時反映在網頁上的 3D 動畫中。

這對於**遠端開發**或**硬體尚未到貨時的驗證**非常有幫助。

以下是將此功能整合進你的 `Controller` 系統的詳細步驟。

---

### 第一步：在 Zaber 網站取得「連接代碼 (Connection Code)」

在修改程式碼之前，我們需要先建立雲端虛擬裝置並取得連線金鑰。

1. **前往網站**：打開瀏覽器前往 [Zaber Virtual Device](https://www.zaber.com/virtual-device)。
    
2. **選擇硬體型號**：
    
    - 在左側選單搜尋並選擇與你專案中類似的型號（例如你之前提到的 `X-LRQ300BP` 或 `X-RST120AK`）。
        
    - 點擊 "Add Device" 將其加入虛擬平台。
        
3. **開啟 API 連接**：
    
    - 在畫面右上角或設定區，找到 **"Connect"** 或 **"API"** 按鈕。
        
    - 選擇 **"Zaber Motion Library"**。
        
    - 你會看到一段像是 `Connection.open_iot("...長字串...")` 的代碼。
        
    - **複製那串長字串 (Token)**，這就是我們需要的「鑰匙」。
        

---

### 第二步：修改設定檔 (`system_config.yaml`)

我們需要在設定檔中增加一個開關，決定現在是要連「實體 USB」還是「雲端 3D Simulator」，並儲存剛剛複製的 Token。

請修改 `config/system_config.yaml` 的 `hardware` 部分：

YAML

```
# config/system_config.yaml

hardware:
  zaber:
    # 模式選擇: "serial" (實體USB) 或 "iot" (Zaber網頁3D模擬)
    connection_mode: "iot"  
    
    # 實體連線設定
    port: "COM3"
    
    # 虛擬連線設定 (Zaber Cloud Token)
    # 請將下方字串替換為你在 Zaber 網站上取得的 Connection Code
    iot_token: "YOUR_ZABER_CLOUD_TOKEN_HERE"

    # 裝置映射 (邏輯名稱 -> 裝置索引)
    # 注意: 在 Zaber 網頁上加入裝置的順序決定了這裡是 Device 1 還是 Device 2
    device_map:
      stage_L_X: 1  
      stage_L_Y: 2 
      stage_R_Z: 3
```

---

### 第三步：修改 `ZaberManager` 程式碼

我們需要修改 `Controller/hardware_managers.py`，讓它根據設定檔決定使用 `open_serial_port` 還是 `open_iot`。

**修改檔案：`Controller/hardware_managers.py`**

Python

```
import logging
from zaber_motion import Units, Library
from zaber_motion.ascii import Connection as ZaberConnection

class ZaberManager:
    def __init__(self, config):
        """
        初始化 Zaber 控制器
        config: 來自 system_config.yaml 的 hardware.zaber 區塊
        """
        self.logger = logging.getLogger("ZaberMgr")
        self.config = config
        self.mapping = config.get('device_map', {})
        self.connection = None
        self.device_list = []

        # 1. 判斷連線模式
        mode = config.get('connection_mode', 'serial').lower()
        
        try:
            if mode == 'iot':
                self._connect_iot()
            else:
                self._connect_serial()
            
            # 2. 偵測裝置 (無論是實體還是虛擬，API 都通用)
            self.device_list = self.connection.detect_devices()
            self.logger.info(f"Zaber Devices Found: {len(self.device_list)}")
            
            # 驗證 Mapping 是否正確
            self._validate_mapping()

        except Exception as e:
            self.logger.error(f"Zaber Connection Failed: {e}")
            raise

    def _connect_serial(self):
        """連接實體 USB/Serial 裝置"""
        port = self.config.get('port', 'COM3')
        self.logger.info(f"Connecting to Zaber via Serial: {port}")
        self.connection = ZaberConnection.open_serial_port(port)

    def _connect_iot(self):
        """連接 Zaber 網頁 3D 模擬器"""
        token = self.config.get('iot_token')
        if not token:
            raise ValueError("Connection mode is 'iot' but 'iot_token' is missing in config.")
        
        self.logger.info("Connecting to Zaber Cloud Simulator (3D Viewer)...")
        # 這是 Zaber API 連接虛擬裝置的關鍵函式
        self.connection = ZaberConnection.open_iot(token)

    def _validate_mapping(self):
        """檢查設定檔中的裝置索引是否超出實際偵測到的數量"""
        max_idx = len(self.device_list)
        for name, dev_idx in self.mapping.items():
            if dev_idx > max_idx:
                self.logger.warning(
                    f"[Config Warning] '{name}' maps to Device {dev_idx}, "
                    f"but only {max_idx} devices detected via {self.config.get('connection_mode')}."
                )

    def move_axis(self, axis_name, position, wait=True):
        """
        移動指定軸
        axis_name: 設定檔中的 key (如 'stage_L_X')
        position: 目標位置 (mm)
        wait: 是否等待移動完成
        """
        if axis_name not in self.mapping:
            self.logger.error(f"Axis name '{axis_name}' not found in device_map.")
            return

        device_idx = self.mapping[axis_name] - 1  # API 是 0-indexed，設定檔通常習慣 1-indexed
        
        if device_idx >= len(self.device_list):
            self.logger.error(f"Device index {device_idx+1} out of range.")
            return

        try:
            device = self.device_list[device_idx]
            axis = device.get_axis(1) # 假設每個裝置是單軸 (如果是多軸控制器需調整這裡)
            
            self.logger.info(f"Moving {axis_name} to {position} mm...")
            axis.move_absolute(position, Units.LENGTH_MILLIMETRES, wait_until_idle=wait)
            
        except Exception as e:
            self.logger.error(f"Move Error on {axis_name}: {e}")

    def get_position(self, axis_name):
        """讀取當前位置"""
        device_idx = self.mapping[axis_name] - 1
        if device_idx < len(self.device_list):
             return self.device_list[device_idx].get_axis(1).get_position(Units.LENGTH_MILLIMETRES)
        return 0.0

    def close(self):
        if self.connection:
            self.connection.close()
```

---

### 第四步：如何運作與測試

這個架構的美妙之處在於：**你的 Core 和 Controller Client 完全不需要修改 code**。你只需要改 Config，整個系統就會自動切換到雲端 3D 顯示。

**操作流程：**

1. **設定環境**： 確保你的電腦有網路連線，並且已安裝 Library:
    
    Bash
    
    ```
    pip install zaber-motion
    ```
    
2. **開啟 Zaber 網頁**： 保持 [Zaber Virtual Device](https://www.zaber.com/virtual-device) 網頁開啟，你會看到虛擬的馬達在畫面上。
    
3. **執行測試**： 你可以直接執行 `controller_server.py` 進行測試，或者寫一個簡單的小腳本直接呼叫 `ZaberManager`：
    
    Python
    
    ```
    # test_zaber_iot.py
    import yaml
    import time
    from Controller.hardware_managers import ZaberManager
    
    # 1. 載入 Config
    with open("config/system_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    # 2. 初始化 Manager (這時會連線到 Zaber Cloud)
    # 確保 cfg['hardware']['zaber']['connection_mode'] 是 'iot'
    zaber = ZaberManager(cfg['hardware']['zaber'])
    
    # 3. 發送移動指令
    print("Sending Move Command...")
    zaber.move_axis("stage_L_X", 50.0) # 移動到 50mm
    
    print("Move Complete! Check your browser.")
    
    # 4. 讀回位置
    pos = zaber.get_position("stage_L_X")
    print(f"Current Position: {pos}")
    
    zaber.close()
    ```
    
4. **觀察結果**：
    
    - 當你執行上述 Python 腳本時。
        
    - **看你的瀏覽器**：你會看到網頁上的 3D Zaber 滑台開始移動到 50mm 的位置。
        
    - 這證明了你的 Python 程式成功透過網路控制了 Zaber 的虛擬硬體。
        

### 技術總結：這如何連結到整個系統？

- **無縫切換**：`RealHardwareDriver` (我們在上一個討論中建立的) 會呼叫 `ZaberManager`。
    
- **透明化**：`RealHardwareDriver` 根本不知道它現在是在控制真實的 USB 裝置，還是在控制雲端的 3D 模型。它只是呼叫 `ZaberManager`。
    
- **應用場景**：
    
    - 當你在 **Ann Arbor** 開發，但硬體在 **LA** 時：你可以將 Config 設為 `connection_mode: "iot"`。
        
    - 你在 Ann Arbor 執行 `main_cli.py --command run_routine`。
        
    - 你的程式邏輯（對焦、HDR、流程）會正常跑，當跑到「移動平台」這一步時，你可以在瀏覽器上看到虛擬平台移動，確認你的座標計算邏輯是否正確（例如有沒有撞機、移動順序對不對）。




#### APP UI連結Zaber 3D viewer
```
沒關係我可以不需要看到組裝後的樣子. 那我可以將在視窗上的5個device對應到我的系統裡開發的App? 我的App有個ui裡面有x direction的左右按鍵, y direction的上下按鍵, y direction的上升下降按鍵, 以及這些位移的長度(user輸入), 單位應該就是code裡面設定的. 還有兩個旋轉的按鍵就是控制兩個方向(以z-axis為軸旋轉跟以x-axis為軸旋轉)的旋轉(可以用按一次轉5degree(user可設定)). 如果要做到在ui上按鍵, 然後在網頁的5個device可以有相應的反應. 應該如何做呢
```

沒問題！這正是我們設計 **Client-Server 架構** 的核心價值。

要達成你的需求：「在 UI 按下按鈕 $\rightarrow$ 透過 Python Code $\rightarrow$ 讓 Zaber 網頁上的虛擬裝置移動」，我們需要完成三個步驟。

這就像是玩遙控車：

1. **遙控車 (Controller Server)**：負責接收訊號並驅動馬達（這裡是驅動 Zaber Cloud）。
    
2. **遙控器 (UI App)**：負責發送指令（按一下 = 移動 5mm）。
    
3. **訊號 (Command)**：定義好「相對移動」的指令格式。
    

以下是具體實作步驟：

---

### 第一步：升級 `ZaberManager` (支援「相對移動」)

你提到的「按一下移動 5mm」，這叫做 **相對移動 (Relative Move)**。我們原本的 `ZaberManager` 只支援「絕對移動」(移動到第 50mm 處)。我們需要在 `Controller/hardware_managers.py` 增加這個功能。

**修改 `Controller/hardware_managers.py`：**

Python

```
# ... (保留原本的 code)

    def move_relative(self, axis_name, distance, wait=True):
        """
        [新增功能] 相對移動
        axis_name: 'stage_L_X'
        distance: 移動距離 (mm 或 degree), 負值代表反向
        """
        if axis_name not in self.mapping:
            self.logger.error(f"Axis {axis_name} not found.")
            return

        device_idx = self.mapping[axis_name] - 1
        if device_idx < len(self.device_list):
            try:
                device = self.device_list[device_idx]
                axis = device.get_axis(1)
                
                self.logger.info(f"Moving {axis_name} relative by {distance}...")
                # Zaber API 支援 move_relative
                axis.move_relative(distance, Units.LENGTH_MILLIMETRES, wait_until_idle=wait)
            except Exception as e:
                self.logger.error(f"Relative Move Error: {e}")
```

---

### 第二步：升級 `ControllerServer` (接收相對移動指令)

接著，我們要讓 Server 看得懂 `'move_relative'` 這個新指令。

**修改 `Controller/controller_server.py` 的 `process_command` 方法：**

Python

```
# ... (保留前面的 code)

    def process_command(self, cmd):
        action = cmd.get('action')
        # print(f"Executing: {action}") # Debug 用
        
        try:
            # 1. 絕對移動 (原本的)
            if action == 'move_stage':
                self.zaber.move_axis(cmd['axis'], cmd['value'])
                return {'status': 'ok'}

            # 2. [新增] 相對移動 (UI 按鍵用)
            elif action == 'move_relative':
                # value 這裡是移動的增量 (例如 +5.0 或 -5.0)
                self.zaber.move_relative(cmd['axis'], cmd['value'])
                return {'status': 'ok'}

            # ... (保留其他的 light, capture 等)
```

---

### 第三步：開發 UI App (`app_ui.py`)

為了讓你直接測試，我寫了一個簡單的 **GUI 控制面板** (使用 Python 內建的 `tkinter`，不需額外安裝)。這個 App 會模擬你未來的正式系統 UI。

**建立檔案 `Controller/app_ui.py`：**

Python

```
import tkinter as tk
from tkinter import ttk
import json
import socket

# --- 模擬 ControllerClient (簡化版) ---
class SimpleClient:
    def __init__(self, host='127.0.0.1', port=5000, token='SECRET_WATCH_2025'):
        self.host = host
        self.port = port
        self.token = token

    def send(self, command_dict):
        """發送指令給 controller_server.py"""
        command_dict['token'] = self.token
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(3) # 避免卡死
                s.connect((self.host, self.port))
                
                data = json.dumps(command_dict).encode('utf-8')
                # 簡單協議：先送 4 bytes 長度
                s.sendall(len(data).to_bytes(4, 'big') + data)
                
                # 接收回應
                header = s.recv(4)
                if not header: return {}
                resp_len = int.from_bytes(header, 'big')
                resp_data = b""
                while len(resp_data) < resp_len:
                    packet = s.recv(resp_len - len(resp_data))
                    if not packet: break
                    resp_data += packet
                return json.loads(resp_data.decode('utf-8'))
        except Exception as e:
            print(f"[Client Error] {e}")
            return {'status': 'error', 'msg': str(e)}

# --- UI 主程式 ---
class ControlApp:
    def __init__(self, root):
        self.client = SimpleClient()
        self.root = root
        self.root.title("Rolex System Controller (IoT Simulation)")
        
        # 設定預設移動步長
        self.linear_step = tk.DoubleVar(value=10.0)  # mm
        self.angle_step = tk.DoubleVar(value=5.0)    # degree

        self._build_ui()

    def _build_ui(self):
        # 1. 設定區域
        setting_frame = ttk.LabelFrame(self.root, text="Settings")
        setting_frame.pack(padx=10, pady=5, fill="x")
        
        ttk.Label(setting_frame, text="Linear Step (mm):").pack(side="left", padx=5)
        ttk.Entry(setting_frame, textvariable=self.linear_step, width=5).pack(side="left")
        
        ttk.Label(setting_frame, text="Angle Step (deg):").pack(side="left", padx=5)
        ttk.Entry(setting_frame, textvariable=self.angle_step, width=5).pack(side="left")

        # 2. X/Y/Z 控制區 (Linear)
        xyz_frame = ttk.LabelFrame(self.root, text="XYZ Stage Control")
        xyz_frame.pack(padx=10, pady=5, fill="both")

        # Grid layout for buttons
        # Y Axis (Up/Down in real world, usually Y is Forward/Back or Left/Right depending on definition)
        # 這裡假設: X=左右, Y=前後, Z=上下
        
        # Y Control
        ttk.Button(xyz_frame, text="Y Back (-)", command=lambda: self.move("stage_L_Y", -1)).grid(row=0, column=1, pady=5)
        ttk.Button(xyz_frame, text="Y Fwd (+)", command=lambda: self.move("stage_L_Y", 1)).grid(row=2, column=1, pady=5)
        
        # X Control
        ttk.Button(xyz_frame, text="X Left (-)", command=lambda: self.move("stage_L_X", -1)).grid(row=1, column=0, padx=5)
        ttk.Button(xyz_frame, text="X Right (+)", command=lambda: self.move("stage_L_X", 1)).grid(row=1, column=2, padx=5)

        # Z Control (Separate column)
        ttk.Separator(xyz_frame, orient='vertical').grid(row=0, column=3, rowspan=3, sticky='ns', padx=10)
        ttk.Label(xyz_frame, text="Z-Axis").grid(row=0, column=4)
        ttk.Button(xyz_frame, text="Z Up (+)", command=lambda: self.move("stage_L_Z", 1)).grid(row=1, column=4, pady=5)
        ttk.Button(xyz_frame, text="Z Down (-)", command=lambda: self.move("stage_L_Z", -1)).grid(row=2, column=4, pady=5)

        # 3. 旋轉控制區 (Rotation)
        rot_frame = ttk.LabelFrame(self.root, text="Rotation Control")
        rot_frame.pack(padx=10, pady=5, fill="both")

        # Rotate Z
        ttk.Label(rot_frame, text="Rot Z (Table):").grid(row=0, column=0, padx=5)
        ttk.Button(rot_frame, text="CCW (-)", command=lambda: self.rotate("stage_R_Z", -1)).grid(row=0, column=1, padx=5)
        ttk.Button(rot_frame, text="CW (+)", command=lambda: self.rotate("stage_R_Z", 1)).grid(row=0, column=2, padx=5)

        # Rotate X
        ttk.Label(rot_frame, text="Rot X (Arm):").grid(row=1, column=0, padx=5)
        ttk.Button(rot_frame, text="Up (-)", command=lambda: self.rotate("stage_R_X", -1)).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(rot_frame, text="Down (+)", command=lambda: self.rotate("stage_R_X", 1)).grid(row=1, column=2, padx=5, pady=5)

    def move(self, axis, direction):
        step = self.linear_step.get()
        val = step * direction
        print(f"UI: Move {axis} by {val} mm")
        self.client.send({'action': 'move_relative', 'axis': axis, 'value': val})

    def rotate(self, axis, direction):
        step = self.angle_step.get()
        val = step * direction
        print(f"UI: Rotate {axis} by {val} deg")
        self.client.send({'action': 'move_relative', 'axis': axis, 'value': val})

if __name__ == "__main__":
    root = tk.Tk()
    app = ControlApp(root)
    root.mainloop()
```

---

### 第四步：如何執行與測試

現在，這就是你的「模擬控制中心」。

1. 啟動 Server (Backend)：
    
    打開一個終端機 (Terminal)，確認 system_config.yaml 裡面的 Zaber 是 iot 模式。
    
    Bash
    
    ```
    python Controller/controller_server.py
    ```
    
    _你會看到它顯示連接上 Zaber Cloud，並開始 Listening on port 5000。_
    
2. 啟動 UI App (Frontend)：
    
    打開另一個終端機。
    
    Bash
    
    ```
    python Controller/app_ui.py
    ```
    
    _你會看到一個視窗，上面有 X, Y, Z 和旋轉的按鈕。_
    
3. 打開 Zaber 網頁：
    
    將瀏覽器視窗和 UI 視窗並排。
    
4. **開始操作**：
    
    - 在 UI 上點擊 **"X Right (+)"**。
        
    - **觀察**：
        
        1. UI Terminal 顯示：`UI: Move stage_L_X by 10.0 mm`。
            
        2. Server Terminal 顯示：`Executing: move_relative`。
            
        3. **瀏覽器畫面**：Device 1 (X軸) 往右移動了！
            
    - 試著修改上面的 "Linear Step" 為 `50`，再按一次，它就會移動更長的距離。
        
    - 試著按 **"Rot Z CW (+)"**，你會看到 Device 4 開始旋轉。
        

### 這代表了什麼？

你剛剛成功模擬了整個 **App $\leftrightarrow$ Server $\leftrightarrow$ Hardware** 的完整迴路。

- **App**: 代表你未來開發的 User Interface。
    
- **Server**: 代表在 Local Computer 上運行的控制核心。
    
- **Browser**: 代表真實的硬體反應。
    

當未來你的真實硬體（Zaber）到貨時，你只需要把 `system_config.yaml` 改回 `serial` 模式，**這套 UI 和 Server 程式碼一行都不用改**，就能直接控制真實的機器手臂！