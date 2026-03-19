

|                             |     |
| --------------------------- | --- |
| [[#### Support software? ]] |     |
|                             |     |
|                             |     |
|                             |     |
|                             |     |

Moonlight Server (ip: 192.168.1.10)

│

UniFi USW Pro Max switch (ip: 192.168.1.21)

│

U6+ access point (ip: 192.168.1.25)

│

Vanta XRS (ip: 169.254.x.x)

|   |   |
|---|---|
|Result|Method Description|
|X|Add a static IP address to Vanta XRS, copy ethernet.config to root|
|X|Add a static IP address to Vanta XRS, wifi.config to root|
|X|Try the python SDK|
|X|Open DHCP server function in UniFi USW Pro Max|
|X|Install DHCP Server for Windows software on Moonlight system and set ip range from 192.168.1.100-120|
|X|Use DeployLANConfig (API command)to read .config in Vanta XRS|
|X|Create new wifi (VANTA_Data2) in UniFi USW Pro Max|
|X|Use DHCP Server for Windows software and disable “Rogue DHCP Server detection” on UniFi USW Pro Max|
|X|Use DHCP Server for Windows software and close the Moonlight system  windows defender firewall|
|X|Disable the “Client Device isolation” and “Multicast and Broadcast Control” in UniFi USW Pro Max|
|X|Disable the “DHCP guard” in UniFi USW Pro Max|
|X|Try DELTA XRF Analyzer - PC Software|


#### Support software? 
```
我有一台Vanta Element Handheld XRF Analyzer, 我想要把測量的結果用wifi(透過跟主機連接的unifi u6+的wifi sihnal)傳送csv file到我主機的shared folder. XRF Analyzer已經設定成功可以用這個wifi 如果我的要存放csv file的folder在D:/Moonlight/xrf_data也已經設定shared folder, 我嘗試在folder輸入\192.168.1.10 (or PUGET-284877), 則出現xrf_data在共享資料夾所以應該資料夾設定成功. 那在XRF Analyzer我確定已偵測到wifi訊號也成功設定好. 但Vanta Element Handheld XRF Analyzer的ip一直顯示169.254.x.x, 無法和主機在同個subnet. 我在這個網頁https://ims.evidentscientific.com/en/products/xrf-analyzers/vanta-element 想找相關的軟體安裝在主機, 看能不能幫忙解決這問題或幫忙設定, 要安裝哪個軟體呢 或者哪裡有關於Vanta Element Handheld XRF Analyzer軟體能幫忙設定的文件或網頁
```
Download link: https://ims.evidentscientific.com/en/downloads/manuals


你的 Vanta Element Handheld XRF Analyzer 要透過 UniFi U6+ WiFi 直接把 CSV 結果匯出到主機的 shared folder（D:/Moonlight/xrf_data，已設成 \192.168.1.10 或主機名稱），這功能是分析儀**內建**的（export 時可選「network folder」），**不需要在主機安裝任何額外軟體**來達成資料傳輸。

你提供的產品頁面 **[https://ims.evidentscientific.com/en/products/xrf-analyzers/vanta-element](https://ims.evidentscientific.com/en/products/xrf-analyzers/vanta-element)** 並沒有提供任何可下載的 PC 軟體、WiFi 設定工具或文件連結。它只提到「optional wireless LAN」可傳到 network folder，但沒有下載區或手冊。

### 要安裝哪個軟體？

官方提供 **Vanta XRF Analyzer - PC Software**（桌面版 PC 軟體），可以用來：

- 從主機遠端控制分析儀
- 查看/下載結果
- fleet management
- 推送設定檔（configuration push）

但這**不是解決你 IP 169.254.x.x 或 shared folder 傳輸的必要工具**（你的需求是用 analyzer 直接推到 network share，不是 PC 軟體拉資料）。它主要是 USB 連線或 closed-beam 模式用，無線版也偏向 cloud 整合。

**下載方式**：

1. 前往官方下載中心：[https://ims.evidentscientific.com/en/downloads](https://ims.evidentscientific.com/en/downloads) （或 [https://evidentscientific.com/en/downloads](https://evidentscientific.com/en/downloads)）
2. 搜尋 “Vanta XRF Analyzer - PC Software”（最新版如 3.44.146）
3. 可能需要註冊 Evident 帳號才能下載（免費）。

如果你只想傳 CSV 到 shared folder，**不用裝這個**，先解決 IP 問題即可。

### 文件與設定說明在哪裡？

最關鍵的文件是：

- **Vanta Family X-Ray Fluorescence Analyzer User Interface Guide**（這本詳細說明 WiFi 連接、network folder export 步驟、IP 顯示、匯出設定等）
- **Vanta User Manual**（硬體部分提到 Wireless LAN USB adapter 插入與基本連接）

**下載位置**：

- 官方手冊頁面：[https://ims.evidentscientific.com/en/downloads/manuals](https://ims.evidentscientific.com/en/downloads/manuals) （或 [https://evidentscientific.com/en/downloads/manuals](https://evidentscientific.com/en/downloads/manuals)）
- 搜尋 “Vanta Family X-Ray Fluorescence Analyzer User Interface Guide” 或 “Vanta” 即可找到（有多語言版本）。

產品頁或你目前連結都沒這些文件，要去上面兩個下載中心找（可能需登入）。

### 額外建議（解決你的 IP 問題）

你的 analyzer 已偵測 WiFi 但拿 169.254.x.x（APIPA 自指派 IP），代表沒從 UniFi 網路拿到 DHCP lease → 無法跟主機同 subnet（192.168.1.x），即使你輸入 \192.168.1.10 能看到 folder，也會傳輸失敗。

- 在 analyzer 的設定裡確認 WiFi 是用 **DHCP**（不是 static IP）。
- 檢查 UniFi 控制器：該 SSID 的 DHCP 是否開啟？有沒有啟用 Client Isolation（常見問題，會擋 device 間通訊）？
- 重插 Wireless LAN USB adapter、重啟 analyzer + UniFi AP + 主機路由器。
- 手冊/UI Guide 裡有 WiFi 連接與 network export 的完整步驟（包括輸入 shared path、帳號密碼）。

如果你不想搞 network folder，也可改用 **Evident Connect** cloud（analyzer 內建支援），直接無線上傳結果到雲端，再從主機下載（產品頁有提到 cloud enabled）。

有其他細節（如 analyzer 顯示的 WiFi 設定畫面截圖）可以再提供，我可以幫你更精準判斷！如果需要官方 support，建議直接聯絡 Evident（產品頁有 Contact Us）。