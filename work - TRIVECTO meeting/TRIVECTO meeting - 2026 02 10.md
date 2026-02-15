

# 智慧手錶檢測系統開發會議記錄與技術規格報告

### 1. 設備交付與物流管理 (Equipment Delivery & Logistics Management)

**說明與戰略重要性：** 硬體組件（特別是相機與鏡頭感測器）的交付進度是系統整合的關鍵路徑。本專案所使用的鏡頭與配件屬借用設備 (Loaner gear)，最終必須原樣歸還予供應商 Optogineering，因此原廠包裝的完整性對於後續保固與資產管理至關重要。此外，在精密光學儀器的跨國物流中，選擇高可靠性的運輸渠道比節省運費更具戰略價值。

**對話記錄：**

"Yep. And I I got a uh look like a a packages look like a FedEx. The camera and everything." 「是的，我收到了一個看起來是 FedEx 的包裹，裡面有相機和其他東西。」

"Yeah. So So it's the the when you get it, they'll be it should get there on Thursday um right at your house. Um, so it'll be a box that that with tape that says Optogineering, which is where the lens is from. Um, keep all the boxes because this is our loaner gear. Um, So, keep the boxes because we're going to need eventually to return it to them." 「是的，所以當你收到它時，它應該會在週四直接送到你家。那是一個貼有 Optogineering 膠帶的箱子，鏡頭就是這家公司的。請保留所有盒子，因為這是我們的借用設備。請保留箱子，因為最終我們需要歸還給他們。」

"Um, so in it you'll find um the longer box in it has the the lens with the builtin um, you know, focus thing. And I just left one of the cable, one end of the cable attached to it. And then in a smaller box, you'll find uh the camera. Um but watch out. The camera is not like properly strapped down in there. Some of the stuff holding it down kind of fell. So o open that open the little camera in very carefully. Um because could roll out um you know, I mean it's it's not very secure in there. And so then there's a little cover on the lens, a little cover on the camera. You unscrew both those, you screw them together, and then that little cable that's in the lens connects to the camera." ==「在裡面你會發現一個裝有內建對焦裝置鏡頭的長盒。我留了一根電纜的一端連在上面。另外一個小盒子裡則是相機==。但要注意，相機在裡面沒有固定得很穩，原本固定的東西掉了一些。所以打開相機小盒時要非常小心，它可能會滾出來，內部並不牢靠。鏡頭和相機上各有一個蓋子。==你把這兩個蓋子都旋開，然後把它們旋在一起，鏡頭裡的那條小電纜就連接到相機上。==」

"Um, and then also in there, not in any sort of box, is a power over Ethernet injector. Um, because they forgot to give me one. Uh, that power over Ethernet is actually ours. Like, you know what I mean? We just bought it off I bought off Amazon. Uh, so and I have the two Ethernet cables and you know what I mean like already plugged into it and just wrapped around it. Um, so you do need to use that to power the camera that does needs to be plugged in." 「裡面還有一個散裝的乙太網路供電（PoE）注入器，因為他們忘了給我。這個 PoE 注入器其實是我們的，是我從 Amazon 買回來的。我已經把兩條乙太網路電纜接在上面並纏繞好。你確實需要使用它來為相機供電，它必須插上電源。」

"Yeah, I will be very careful about that. Very careful. Yeah. So just have be care and just Yeah. Save the boxes. Um, so we can ship the stuff back except so I'll probably when we do ship it back I'll probably have you ship it straight to them rather than you know I don't need to get it again. They just want it back eventually. Um but when we do the power over Ethernet and the Ethernet cables are all ours so we don't have to give it back. Yeah. I think once I finish the testing everything I I think I was just confirm with you and to see what I should uh ship to then and what is yours. So I think uh yeah, we can time." 「是的，我會非常小心，非常謹慎。是的，只要小心並保留盒子即可。這樣我們以後可以把東西寄回去，到時候我可能會讓你直接寄還給他們，我不需要再經手。他們最終只是想要拿回設備。==但 PoE 注入器和網路線是我們的==，所以不用還。是的，我想一旦我完成所有測試，我會再向你確認哪些要寄還，哪些是你的。我們到時候再處理。」

"Um so yeah, so that that's coming there. It better get there Thursday because um it Have you shipped anything recently through FedEx? It is way more expensive than it was a few years ago. Yeah, but I think we uh Yeah, but I think we we try to use the FedEx to send from Taiwan to here because I think use the the the the original I think the USPS is has a risk. I think we send many sometimes and it the package just just gone and it just disappear. Yeah. So I think uh FedEx is um expensive but uh safe we get there. Yeah. Um yeah. So that's what's coming in the mail. Um yeah. So I think once I get that I will just let you know and uh also have some um pictures about what inside and what is is okay." 「東西快到了，最好週四能抵達。你有注意到最近 FedEx 運費比幾年前貴得多嗎？是的，但我認為我們嘗試使用 FedEx 從台灣寄到這裡，是因為我覺得 USPS 有風險。我們寄過幾次，包裹就這樣消失了。所以雖然 FedEx 比較貴，但能安全送達。好的，這就是郵件裡的東西。一旦我收到，我會通知你，並拍一些內部照片確認東西都沒問題。」

**分析關鍵點：** 物流策略上，團隊一致同意排除 USPS，儘管 FedEx 成本顯著增加，但 USPS 頻繁遺失包裹的紀錄（Line 99: "package just gone"）對高價值的借用設備構成不可接受的風險。考量到設備損壞或遺失將導致開發進度全面停滯，選擇安全、可追蹤的物流方案是維護專案穩定性的必要成本。

**小結：** 設備點收流程將包含外部包裝拍照與內部感測器連接驗證。完成硬體檢查後，系統將進入高效能運算的配置評估。

--------------------------------------------------------------------------------

### 2. 高效能運算系統規格評估 (High-Performance Computing Configuration Assessment)

**說明與戰略重要性：** 運算核心效能（尤其是 RAM 容量與多核心處理能力）直接決定了 U-Net 模型在處理高解析度手錶影像時的推理延遲。在工業場景中，處理時間從 7 分鐘縮短至 5 分鐘不僅是速度的提升，更代表系統能支援更高密度的並行測試 (Parallel testing)，這對於大規模生產線整合至關重要。

**對話記錄：**

"Um I thought I was going to try to get it ordered right away. I just want to make sure we get the right one. Um it did end up being yeah a little bit pricey. Um but you know I mean looks like we need the performance we need. Um, it definitely climbed. I thought our $11,000 computer was going to be where we topped out, but, uh, apparently not, you know, because the first we bought was like 3,000. The next is six, next is 11. It's going to be like what 18." 「我本想立即下單，只是想確保我們買對了。價格確實有點貴，但看來我們確實需要這樣的效能。預算確實上升了，我原以為 11,000 美元就是頂點，但顯然不是。我們第一台買 3,000，接著是 6,000，然後是 11,000，現在看起來要 18,000 美元了。」

"So I think it should be based on you and I just to based on the those code and those AI models and we need to finish that in five minutes. So I think that is how I make a recommendation. So if it were to take six minutes or seven minutes, is there something that we could, you know, a more expensive component because RAM has gotten really expensive for... and it wants an extra three grand for an extra 128 gig is a little..." 「我認為建議應基於我們的程式碼與 AI 模型，我們需要在 5 分鐘內完成處理。如果處理時間延長到 6 或 7 分鐘，我們是否要調整某些昂貴組件？因為 RAM 現在變得非常貴，多加 128GB 居然要額外花費 3,000 美元，這有點……」

"Yeah, but that is the market right now. So maybe right now we can have less rent and maybe in the futures. So if we need more and we can add to that. So based on the price. So... I may say in right now I may say maybe 128 is enough is already enough. It's good enough. 128 should be good enough. Yep." 「是的，但這就是目前的市場行情。也許現在我們可以先用較少的 RAM，未來有需要再增加，這取決於價格。就目前而言，我認為 128GB 已經足夠了，這已經很好了。是的，128GB 應該足夠了。」

"I know they have or at least last you can pay them $500 and then your computer gets done first. You're in the front of the line for every single thing. Yeah, you jump the queue for 500. So, we need we needed the computer quickly for client. So, we did that last time. I'd rather not, you know, you know, let's see what their normal um late time is." 「我知道他們至少在上次可以讓你支付 500 美元來優先處理你的電腦。你可以插隊排到最前面。因為我們曾為客戶緊急需要電腦，所以上次這麼做過。這次我寧可先看看正常的交貨時間是多少。」

**核心分析：** 目前面臨顯著的 RAM 市場危機 (RAM crisis)，額外擴充 128GB ECC 記憶體需花費 $3,000 美元。儘管 256GB RAM 能將效能提升至目標 5 分鐘，但在目前的 U-Net 模型規模下，128GB 仍能維持 6-7 分鐘的穩定推理且成本效益較高。

- **成本規避：** 透過推遲非必要的記憶體升級，可節省 $3,000 美元的溢價。
- **緊急費用：** 上次專案曾支付 $500 的「插隊費 (Jump the queue fee)」以縮短交期，本次將優先評估 Puget Systems 的常規交期。

**小結：** 最終配置暫定為 128GB RAM，保留未來視需求擴充至 256GB 的物理空間，並專注於核心數與顯示卡顯存的平衡。

--------------------------------------------------------------------------------

### 3. 基礎設施、網路與電源解決方案 (Infrastructure, Networking, and Power Solutions)

**說明與戰略重要性：** 檢測系統在跨國部署（美國至德國）時，必須解決電壓不相容與網路帶寬受限的問題。24V/48V 的致動器與高功耗運算平台需要穩定的電源供應，而選擇「通用型不斷電系統 (Universal UPS)」能有效簡化跨國轉移時的電路改裝風險。

**對話記錄：**

"So my understanding was was that's what we only need one router not two correct. Yes. So for those um those I uh for those maybe how they talk to each other and also computers I think uh that is one and I think uh as you said maybe it uh maybe total port is maybe 7 to eight. It be good to get 16 rather than eight in case we don't want to use every last port on it. Yeah, I think as I said that so maybe speed is not enough and maybe power is not enough. So I think maybe need to confirm with that." 「我的理解是我們只需要一個路由器而非兩個，對嗎？是的，對於設備互連與電腦通信，一個就夠了，總埠數大約 7 到 8 個。最好買 16 埠而非 8 埠，以免我們把所有連接埠用光。如我所言，還需要確認速度與電力供給是否充足。」

"I'm pretty sure in New York that I don't even know where it would be hard because they want the machine where they work on the watches and where they work on the watches I don't know if there's any hardwired Ethernet jacks. So, there's a chance this is going to be Wi-Fi connected to the internet. Um, which is going to be a pain in the butt for bandwidth." 「我不確定紐約現場是否有實體網路孔。==他們想把機器放在修錶的地方，那裡可能沒有硬線。這意味著系統可能得透過 Wi-Fi 連網，這對帶寬來說會非常麻煩==。」

"The displacement sensor is 24 volts. We're buying an American converter. The biggest thing I'm worried about is the stages. Um, they run off I think 24 or 48 volts, but they have their own. We buy two chunky power supplies from the company for that. If we have an uninterrupted power supply that can take in either German or American power just switching out the power cable if it's universal to that we can plug in not only the server but also the rest of the system because the rest of the system is not that..." 「位移感測器是 24V 的。我擔心的是移動平台，它們運作需要 24V 或 48V，且有自己的大型電源供應器。如果我們有一個能同時接受德國或美國電壓的通用型不斷電系統（Universal UPS），只需更換電源線即可。這樣不僅能接伺服器，也能接系統其餘部分。」

"The actuators are 24 volts and if all of them are actuating and under load they'll pay I think like seven or eight amps at full you know full load all activating at the same time is about 8 amps 24 volts. So that's that's like 240 that's like 250 wattsish. So with above 350. So that's that's rounding it on our on our um on our power supply. So I think this might be the smartest way to go for us." 「==致動器是 24V 的，如果全部在滿載下運行，大約會消耗 8 安培。24V 乘以 8 安培大約是 240 到 250 瓦。算上餘量大約需要 350 瓦。這可能是我們最聰明的做法，直接用通用型 UPS 驅動整個系統。==」

**技術決策分析：**

- **電力規格：** 系統負載包含 24V 位移感測器與 24V/48V 致動器。經計算，滿載電流為 8A，功率需求約為 250W。考量到啟動電流與安全性，UPS 需支撐 350W 以上的總負載。
- **跨國適應性：** 採用通用型 UPS 可避免在德國重新採購變壓器。
- **網路風險：** 現場可能缺乏實體網路線（Hardwired jack），必須針對 Wi-Fi 環境下的 S3 上傳效能進行優化。

**小結：** 將採購 16 埠 PoE 交換機，並優先尋找支援 110V/220V 輸入的通用型 UPS 方案。

--------------------------------------------------------------------------------

### 4. 系統擴展需求：安全與存取控制 (System Expansion: Safety and Access Control)

**說明與戰略重要性：** 鑑於檢測設施記憶有價值數億美元的手錶庫存（Line 969: "few hundred million dollars worth of watches"），系統必須具備嚴格的門禁與安全邏輯。這不僅是為了資產保護，更是為了防止自動化致動器誤傷操作員（Line 928: "don't want to cut people's hands off"），避免潛在的法律訴訟風險。

**對話記錄：**

"I'm planning on putting some sort of proximity sensor of some sort on the door so when it's closed they can run the system and things can move if they open the door mid operation. Everything stops. You, you know, you stop, you know, we send the commands, everything stop moving." 「我計劃在門上安裝接近感測器，門關上時才能運作；若在操作中途開門，所有動作都必須停止。我們會發送指令讓一切停止移動。」

"Another would be sort of I might put like a laser grid across the front. So if you try to somehow you got past the other one when you put your hand in it it triggers it. They sell them commercially. The other thing within this we also want in engineering mode to not have that be a universal stop command but in engineering mode you could say keep running when I open the door you know I mean like we'll want to open it and see what's happening." 「我也考慮在正面安裝雷射柵欄 (Laser grid)。如果你穿過了門禁，手伸進去時也會觸發。另外，我們需要『工程模式』，在此模式下開門不會觸發全面停止，以便工程師觀察運作狀況。」

"Is there any way we can have RFID cards and a card reader for the login? So instead of a username password the the worker would just put their you know scan their... it's faster and and two it's sort of a nice way to you know prove this you know your card you're supposed to have physical possession. These facilities are secure because there's like a few hundred million dollars worth of watches sitting there." 「有沒有辦法使用 RFID 卡登入？這比輸入帳密快，也能證明操作者確實持有授權卡片。這些設施非常安全，因為那裡放著價值數億美元的手錶。」

"So I think we're thinking RFID because it's easy you know you just sort of touch your card. If you want to actually secure stuff well you do you know um you're familiar with the sec the the three parts of security something you have something you know and something you are. kind of two factor maybe." 「我們考慮 RFID 是因為方便。如果你想真正確保安全，你應該知道安全三要素：你擁有的東西（卡片）、你知悉的資訊（密碼）以及你的生物特徵。這就是雙因子認證的概念。」

**差異化評估：**

- **安全連鎖邏輯：** 引進雷射柵欄作為二級防護。工程模式將允許在門開啟時維持運作，以便進行硬體校準。
- **身分驗證：** 針對高價值庫存環境，RFID 提供「Something you have」的實體驗證，結合密碼可達成雙重保證，且大幅提升操作員登入頻率的效率。

**小結：** 將評估商用 RFID 讀卡機的 API 整合性，並設計「一般/工程」雙軌安全停止邏輯。

--------------------------------------------------------------------------------

### 5. 軟體進度演示與模板管理系統 (Software Progress & Template Management System)

**說明與戰略重要性：** 軟體架構的核心競爭力在於其雲端同步能力（AWS S3）與自動化命名規範。確保全球站點使用一致的檢測模板（如 Rolex 特定型號的圈口顏色）是維持檢測精度的關鍵。

**對話記錄：**

"I mentioned that it already can connect to the uh AWS. So I think I think once we started to run this and it will to uh to compare with the uh template because I think it will connect to the AWS and then check the template database and uh and make sure in the local database it is the latest one. I already just to copy those three um report to the the company GitHub." 「系統已經可以連接 AWS。運行時會連接雲端檢查模板資料庫，確保本地端是最新版本。我已經把相關程式碼備份到公司的 GitHub 了。」

"For the naming itself. On the other side, one feature I would like... my first priority is just getting the templates named properly because with Rolex reference and dial color, you probably some of them require bezel color. So there's the GMTs. look up like the Rolex Pepsi for example. Or Rolex root beer." 「關於命名，我首要任務是確保模板命名正確。對於勞力士，除了型號與面盤顏色，有的還需要『圈口顏色 (Bezel color)』。像是 GMT 系列的 Pepsi 或 Root Beer。」

"The Rolex root beer is a reference 126711. The problem is the Rolex Pepsi I think is also reference 126711 the but you you'd separate it by sort of it its look. It automatically so when they select information about the watch, there was a column for brand like Rolex, there's a column for reference number um and then there was a column for dial color. Um and so I autogenerated the template name. It was brand underscore reference dial color name." 「Root Beer 的型號是 126711，Pepsi 好像也是 126711，但你必須根據外觀來區分。系統現在會根據選擇的資訊自動生成名稱，格式為：品牌_型號_面盤顏色。」

**流程優化分析：**

- **自動生成命名 (Autogenerated Naming)：** 為避免人為輸入造成的數據混亂（Chaos），軟體將強制執行「品牌_型號_面盤顏色_圈口顏色」的命名建議。
- **細節辨識：** 對於型號重疊的特殊錶款（如 126711 的不同圈口），Bezel Color 將作為關鍵區分維度。

**小結：** 軟體已實現 AWS S3 的圖像與數據自動上傳，後續將針對 Rolex GMT 等特殊錶款細化模板命名邏輯。

--------------------------------------------------------------------------------

### 6. 未來部署時程與現場整合 (Future Deployment & On-site Integration)

**說明與戰略重要性：** 在正式部署至德國前，系統需在加州與洛杉磯的經銷商處進行初步掃描與校準。實地安裝不僅是接線，更涉及關鍵的「相機配准 (Camera Registration)」與空間位置校驗，這是確保 AI 模型在不同物理設備上維持一致表現的基礎。

**對話記錄：**

"Yeah, but it's going to start its life in the US. We actually probably want to use a system to do scans and watches at a Los Angeles dealer before we send it over to Germany. I don't want to pick time exact timing yet, but once the systems partly mostly put together, um it makes the most sense for you to fly out here to make sure you're sitting in front of the actual potential thing." 「系統將先在美國開始運作。在寄往德國前，我們想先在洛杉磯的經銷商處進行手錶掃描。我還不想定下確切時間，但一旦系統大致組裝完成，你飛過來親自面對設備是最合理的。」

"Because you you know I mean you need to like be in front of the system and you know what I mean and actually implement and troubleshoot everything make sure everything works flawlessly. Yep. Um here on it because I think it is need to do that because once you combine those camera things you still need to do some uh camera regations and those uh locationations. So to make sure that is okay." 「因為你需要親自操作系統進行安裝與故障排除，確保一切完美。當相機組裝完成後，還需要執行相機配准 (Camera Registration) 與位置校驗 (Locationations)，確保一切沒問題。」

**關鍵任務清單：**

1. **洛杉磯經銷商初測：** 執行首批實體手錶掃描，驗證光學參數。
2. **相機配准 (Camera Registration)：** 校對多鏡頭感測器的空間坐標一致性。
3. **位置校驗 (Locationations)：** 針對移動平台的精確停止點進行微調。

**總結：** 本專案目前硬體零件已陸續到貨（包括致動器與位移感測器），預計於 1 至 1.5 個月內啟動現場整合。透過強化安全邏輯、優化運算配置與嚴謹的命名規範，我們將確保系統在 2026 年交付時具備最高的專業度與穩定性。