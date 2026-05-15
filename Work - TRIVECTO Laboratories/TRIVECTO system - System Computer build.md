


|                                        |     |
| -------------------------------------- | --- |
| [[#### V2 Moonlight 系統Summary跟電腦配備需求]] |     |
|                                        |     |
|                                        |     |
|                                        |     |


#### V2 Moonlight 系統Summary跟電腦配備需求
```
我現在開發這套V2 Moonlight 系統, 自動化手錶光學檢測系統 (python based imaging processing system在windows platform並支援GPU)。這是一套對每一次一個watch由自動camera system拍攝錶面不同部位約90張images (2048 x 2048)總共約1.8GB傳入, 並針對每張image有不同的imaging processing task分析上面的features並儲存結果可能包括segmentation masks及這種values並傳出某個matrics, 大部分會使用opencv等傳統影像library但其中1/3有些可能需要segmentation AI models or OCR(UNet, SAM, OCR)辨認上面文字. 最後整合90幾個這些matrics並和某些reference matrics做比對, 最後確認這watch是真的或假的.

這是系統裡硬體包括cameras, light controllers, and motion platforms - 多相機系統：至少有 4 支相機（Tele, Micro, Top, Side）, 精密運動平台：使用 Zaber 控制器進行 X, Y, Z 三軸移動。雷射/位移感測器：用於測量表面高度和玻璃厚度。可控光源：多通道燈光控制. 「先用大範圍相機定位 -> 移動到目標位置 -> 利用感測器進行精密對焦（需補償玻璃折射） -> 自動調整曝光與增益 -> 拍攝 HDR 影像 -> 用於特徵分析。」. 以下是硬體的型號: Zaber X-LRQ300BP, Zaber X-RST120AK, Opto-engineering: ITA81-GC-20C-EL, opto-engineering TICGR1000-D1, Adam ADAM-6266-B, Keyence CL-3000, LTDVE2CH-20F.

至於系統架構設計上採用了模組化設計，涵蓋了從 CLI 指令入口及APP雙入口、local system跟remote client協同控制也可以遠方完全操控硬體跟分析以及看到即時影像. AI 服務調度核心業務邏輯跟AI 服務調度image processing pipeline(Orchestrator控制30多個subworker, 採用cli work model跟api server混和模式)、資料管理(camera拍照後會加密存在local folder跟image processing pipeline的結果加密存在local experiment folder. 兩者都會由CloudSyncManager同步到AWS S3, 並有QueryEngine(SQLite)可以快速查詢local跟cloud data)、底層硬體控制(包括控制camera , light source, zaber等), 以及一個AuditLogger可以加密記錄logger, 以及後續advanced analysis跟statistics的全流程. 這套系統的核心目的是：控制硬體拍攝手錶影像 -> 進行加密與存儲 -> 呼叫 opencv跟AI 模型分析 -> 輸出檢測報告. 未來可能local computer連接會在LA, 或NY或甚至在德國. 而remote client會是在Ann Arbor或美國其他地方可以連進去local computer操控或維修. 

這個系統需要AWS Cloud 上面有WatchTemplates, WatchAnalysisResults, WatchCommandLog等DynamoDB並會需要跟Local SQLite同步, 以及將Camera taken images上傳到Cloud上面儲存, 以及Images and metadata results上傳到Cloud上面儲存, 以及AuditLogger也在Cloud上. 除此之外還有Remote client跟Local computers之間的AWS IoT (MQTT)聯繫, 傳送Local computers的camera畫面到Remote Client, 以及透過AWS cloud可以做到遠端幫local computers(可能在Germany or New York)升級軟體版本.

在系統設計跟效能要求上可能一個月系統就會scan 30000隻手錶(每天處理200~500隻watch)所以local system暫存的images跟results但系統還是必須要有足夠大的storage space. 在Scan上當然希望要夠快可以即時存到硬碟以及分析上(這應該是CPU loading重的jobs但這些tasks可以併行處理)用傳統image process analysis跟AI model analysis希望整個流程會在5分鐘之內可以完成分析. 跟Cloud上的上下傳資料以及遠端傳送影像跟遠端遙控希望可以夠通順. I would like it to be rack mounted and we'll go with Puget systems. Make sure you consider all factors, including enough network speed for both the internal system (cameras to computer) and external (moonlight to AWS). In addition to that, please select a network router, ideally a rack-mount or DIN rail mount (but not hard requirement) that will work for the internal communication. Also, please find an uninterupted power supply that will work for the system. 

請依40年tech lead深入分析這個系統並推薦這個V2 Moonlight 系統的電腦配置要如何, 並詳細解釋為何這樣選擇
```

這是一份基於您提供的 V2 Moonlight 系統架構、負載需求（30,000 隻手錶/月，每隻 1.8GB）、以及跨國部署場景（Ann Arbor 控管 Global）的深度技術分析與硬體配置建議。

作為一名有 40 年經驗的 Tech Lead，我首先要指出該系統最大的**隱形風險**不在於 AI 算力，而在於 **I/O 吞吐量（Throughput）** 與 **遠端維運的可靠性（Remote Reliability）**。

每天 900GB 的資料生成量（500 隻 $\times$ 1.8GB），意味著您的系統不只是一台檢測機，更是一個小型的**邊緣資料中心（Edge Data Center）**。

以下是針對 Puget Systems 的配置建議清單，以及網絡與電源的詳細選型。

---

### 一、 核心運算單元 (Computer Configuration)

**供應商：** Puget Systems

**機型建議：** Rackmount Workstation (4U)

**核心邏輯：** 針對 "High Concurrency Python Processing" (30+ subworkers) 與 "High Throughput I/O" 進行優化，並強制要求遠端管理功能 (IPMI)。

|**組件 (Component)**|**建議規格 (Specification)**|**40年 Tech Lead 的深度解析 (Rationale)**|
|---|---|---|
|**Platform**|**AMD Ryzen Threadripper 7000 Series**|**關鍵選擇**：您的 Orchestrator 須管理 30+ subworkers。Python 的 `multiprocessing` 在高併發下會產生大量的 Context Switching。消費級 CPU (Intel Core/AMD Ryzen) 的 PCIe 通道數不足以支撐 4x 相機、GPU、高速網卡同時滿載。Threadripper 提供足夠的 PCIe Lanes 確保相機影像傳輸不卡頓。|
|**CPU**|**Threadripper 7960X (24 Cores) 或 7970X (32 Cores)**|每個 Image Processing task 分配約 0.8~1 個實體核心是較安全的估算。32 核心可確保在執行 AI Inference (GPU loading) 的同時，CPU 仍有餘裕處理 OpenCV 運算及 AWS S3 Multipart Upload，避免系統卡頓（Stuttering）。|
|**RAM**|**256GB DDR5 ECC Registered**|**不要省錢**。每隻手錶 1.8GB raw data，若系統同時並行處理 5-10 隻手錶的 pipeline，加上 SAM/UNet 模型的 VRAM swapping 緩衝、OS cache 以及 SQLite 的 I/O 緩衝，128GB 會捉襟見肘。ECC 記憶體對於長時間運行的工業電腦是必須的，避免 bit-flip 導致 crash。|
|**GPU**|**NVIDIA GeForce RTX 4090 24GB (Blower Style)**|雖然 RTX 6000 Ada 更穩定，但在 Windows 環境下跑 PyTorch/SAM，4090 的性價比無敵。**重點：** 請指定 "Blower Style" (渦輪扇) 散熱版本，因為在 Rack 機架內，開放式散熱會導致熱氣回流，燒毀其他組件。24GB VRAM 足以跑 SAM (Segment Anything Model) 的 Large 版本。|
|**OS Drive**|**1TB Samsung 990 Pro NVMe**|系統碟與資料碟必須物理分離。|
|**Work Drive** (Scratch)|**4TB Samsung 990 Pro NVMe (或者 RAID 0 兩條)**|**效能瓶頸點**。相機寫入與 AI 讀取會同時發生。需要極高的 IOPS。建議此槽僅做 "暫存與處理"，處理完即搬移。|
|**Storage** (Buffer)|**18TB+ Enterprise HDD (Ultrastar / IronWolf Pro)**|**容災緩衝**。每天 900GB，若 AWS 上傳因網路中斷而延遲，您需要至少 10-14 天的本地緩衝空間 (Buffer)。18TB 可提供約 20 天的緩衝，讓您有時間修復網路而不停工。|
|**Network**|**Dual 10GbE NIC (Intel X550-T2)**|1 Port 接內部相機 Switch (隔離流量)，1 Port 接 Router 上傳 AWS。千萬別共用 1GbE Port，那會塞死。|
|**Management**|**IPMI / BMC Module (必選)**|**這是跨國維運的救命稻草**。當 Windows 藍屏或死機時，您需要在 Ann Arbor 透過 IPMI 進行底層重啟 (Hard Reset) 或進入 BIOS。請確保 Puget 選用的主機板支援 BMC (如 ASUS Pro WS WRX80 或 ASRock Rack)。|

---

### 二、 網絡架構與路由器 (Network Infrastructure)

**挑戰：** 需隔離 "高頻寬相機流量" 與 "對外 AWS 流量"，並確保德國/美國之間的 VPN 連線。

**路由器建議：** **Ubiquiti UniFi Dream Machine Special Edition (UDM-SE)**

- **型式：** Rack-mount (1U)。
    
- **為何選擇：**
    
    1. **Site-to-Site VPN / Magic Site-to-Site:** 您提到有 Remote Client 在 Ann Arbor，UDM 內建的 VPN 功能極強且設定簡單，可以讓您在美國像在區網內一樣直接連線到德國的 Zaber 控制器或 SQL DB。
        
    2. **IPS/IDS:** 企業級防火牆，保護 Windows 機台不被勒索病毒攻擊。
        
    3. **10G SFP+ WAN/LAN:** 支援未來的光纖升級。
        
    4. **Cloud Management:** 您可以在單一介面看到所有工廠（LA, NY, Germany）的網路狀態。
        

**內部交換器 (如果相機需要 PoE 或額外 Port)：** **USW-Pro-Max-24-PoE**

- 將相機流量鎖在 Switch 層級，不讓其廣播風暴影響 Router CPU。
    

**頻寬警示 (Critical Warning)：**

- **上傳需求：** 每天 900GB 資料上傳。
    
    $$\frac{900 \times 1024 \times 8 \text{ Mbits}}{24 \times 60 \times 60 \text{ seconds}} \approx 85 \text{ Mbps}$$
    
    這需要**恆定**的 85 Mbps 上傳速度。考慮到網路波動與日間辦公頻寬，安裝地點必須配備 **500 Mbps 以上的對稱式光纖 (Symmetric Fiber)**。若只有傳統 Cable (上傳通常僅 35Mbps)，系統的 Local Storage 會在 20 天內爆滿。
    

---

### 三、 電源保護 (Power Protection)

**挑戰：** 德國 (230V) 與美國 (110V) 電壓不同，且光學檢測對電壓波動極其敏感（光源閃爍會導致 AI 誤判）。

**UPS 建議：** **APC Smart-UPS On-Line, 3000VA (SRT3000XLA 或 SRT3000RMXLW-IEC)**

- **架構：** **Double Conversion On-Line (線上雙轉換式)**。
    
    - _原因：_ 這是唯一能提供 "零轉換時間" 且輸出純正弦波的架構。Line-interactive (在線互動式) 在切換電池時的微小斷電可能會導致相機掉線或 Zaber 控制器歸零重置，毀掉當前的檢測 Batch。
        
- **電壓兼容性：** 建議選擇 "Global" 版本或依據當地電壓採購。若要統一採購運往各地，需選購支援 208V-240V 輸入的型號，並在美國使用 L6-20 插座。
    
- **容量：** 3000VA (約 2700W)。
    
    - PC (1000W) + Lights/Motor (500W) + Buffer = 1500W 負載。3000VA 可提供約 10-15 分鐘的緩衝時間，讓您的軟體有時間寫入 `AuditLogger` 並優雅關機 (Graceful Shutdown)。
        

---

### 四、 Tech Lead 的額外架構建議 (Day 2 Operations)

1. **資料同步策略 (CloudSync Optimization)：**
    
    - **Tiered Upload:** 不要試圖將所有 2048x2048 的 Raw Images 即時上傳。這會塞爆頻寬。
        
    - **建議：**
        
        1. **Hot Path:** 僅上傳 Analysis Results (JSON/DynamoDB) 和 "Defect Crops" (有問題部位的截圖) 到雲端。
            
        2. **Cold Path:** 原始 1.8GB 數據在本地壓縮 (如 Lossless WebP 或 HEIC) 後，利用離峰時間 (半夜) 排程上傳，或僅在需要人工複驗時才上傳。
            
2. **相機與光源的觸發同步 (Hardware Triggering)：**
    
    - 確保您的相機與 Zaber/Light Controller 是透過 **Hardware Trigger (GPIO 線)** 連接，而非純依賴軟體指令 (USB/Ethernet commands)。Windows 不是即時作業系統 (RTOS)，軟體觸發會有 10-50ms 的抖動，這會導致 "Motion Blur" 或 "亮度不一致"，進而影響 AI 判讀準確率。
        
3. **遠端救援 (Out-of-Band Access)：**
    
    - 除了 TeamViewer/RDP，您需要 **KVM-over-IP** (如果主機板沒有內建 IPMI) 或者確保 Router 有設好 WireGuard VPN，以防 Windows Update 自動重開機後無法連線。
        

**總結下一步：**

您現在可以拿著這份規格 (Threadripper 7970X, 256GB RAM, RTX 4090 Blower, Dual 10G NIC) 去聯繫 Puget Systems 獲取報價，並同步確認部署地點的光纖上傳頻寬是否達標 (至少 200Mbps upload)。