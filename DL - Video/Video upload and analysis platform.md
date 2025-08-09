

|                      |     |
| -------------------- | --- |
| [[## 執行摘要]]          |     |
| [[### 手術影片分析平台實作指南]] |     |
|                      |     |

![[Pasted image 20250808152526.png]]

## 執行摘要

本報告旨在為一個先進的手術影片分析平台，提供一份全面、專家級的技術架構藍圖。此平台專為醫療技術領域設計，旨在讓使用者（如外科醫生、研究人員或醫療機構）能夠上傳大型手術影片檔案，並利用頂尖的人工智慧（AI）技術進行深入分析。本架構的核心目標是實現一個安全、合規、可擴展且高效能的系統，以應對當前及未來的醫療分析需求。

我們提出的解決方案基於一個多層次的雲端原生策略，其核心架構支柱包括：

1. **HIPAA 合規優先的設計 (HIPAA-First Design)：** 鑑於手術影片屬於受保護的健康資訊（PHI），整個平台的設計從根本上以美國《健康保險可攜性與責任法案》（<mark style="background: #BBFABBA6;">HIPAA</mark>）的嚴格要求為驅動。這不僅僅是一個後續的合規性檢查，而是影響網路、資料儲存、存取控制和稽核日誌等所有基礎設施決策的核心原則。
    
2. **事件驅動的無伺服器處理 (Event-Driven Serverless Processing)：** 平台的核心工作流程採用事件驅動的無伺服器架構。從影片上傳完成的那一刻起，一系列自動化流程便由 <mark style="background: #FF5582A6;">AWS Step Functions</mark> 進行編排，協調 <mark style="background: #FF5582A6;">AWS Lambda、Amazon Rekognition 和 Amazon SageMaker</mark> 等服務，以平行且高效的方式執行多種分析任務。這種設計不僅最大化了可擴展性，也極大地優化了成本效益，因為運算資源僅在需要時使用。
    
3. **全面的機器學習維運 (MLOps) 框架：** 為確保模型的品質、可重複性和治理，我們設計了一個成熟的 MLOps 框架。此框架整合了如 <mark style="background: #FF5582A6;">Weights & Biases (wandb) 和 ClearML</mark> 等業界領先的工具，並與 Amazon SageMaker 的原生功能（如模型註冊表和管線）緊密結合。這涵蓋了從實驗追蹤、資料與模型版本控制，到自動化訓練、評估和部署（CI/CD）的完整生命週期。
    
4. **前瞻性的視覺語言模型 (VLM) 整合：** 為了滿足未來的分析需求，本架構從設計之初就考慮了與視覺語言模型（VLM）的整合。這將使平台不僅能「看見」手術過程中的物件和事件，更能「理解」其上下文，從而生成結構化的敘述性報告、回答複雜問題，將原始資料轉化為具備臨床價值的深刻洞見。
    

本報告將深入探討從基礎設施的建立、大規模資料的擷取與管理，到 AI 模型的開發、訓練、部署與優化，乃至最終使用者介面的呈現等各個層面的技術細節。其旨在為技術長（CTO）、工程副總裁或首席架構師提供一份清晰、可執行的戰略與技術指南，以期成功建構一個世界級的手術智慧平台。

## 第 1 節：基礎架構：安全性、合規性與資料管理

本節旨在建立平台不可動搖的基礎。後續所有元件的設計與實施，都將嚴格遵循此處定義的安全性、合規性與資料治理原則。在處理高度敏感的手術影片時，這些基礎設施的建構並非選項，而是成功的先決條件。

### 1.1. HIPAA 合規的 AWS 環境：多帳戶策略

建立一個符合 HIPAA 規範的環境，是整個專案的基石。這需要深刻理解 AWS 的共擔責任模型：AWS 負責「雲本身」的安全，而客戶則必須負責「在雲中」的安全 。這意味著平台的合規性完全取決於對 AWS 服務的正確配置與管理。  

**商業夥伴附約 (Business Associate Addendum, BAA)：** 在處理任何受保護健康資訊（PHI）之前，首要且強制性的步驟是與 AWS 簽署 BAA。這是一份法律協議，確立了 AWS 在保護 PHI 方面的責任。簽署 BAA 後，平台必須專門使用 AWS 官方列出的「HIPAA 合格服務」來處理、儲存或傳輸任何 PHI 。  

**虛擬私有雲 (Virtual Private Cloud, VPC) 設計：** 我們將建構一個邏輯上與公用網路隔離的安全網路環境，作為保護 PHI 的第一道防線。

- **私有子網路 (Private Subnets)：** 所有處理 PHI 的核心資源，包括 AWS Lambda 函數、Amazon SageMaker 端點以及資料庫，都將部署在私有子網路中。這些子網路不具備直接的對外網際網路存取路徑，從根本上杜絕了來自外部的直接攻擊 。  
    
- **NAT 閘道 (NAT Gateways)：** 當私有子網路中的資源需要對外連線時（例如，下載軟體套件或呼叫已列入白名單的外部 API），所有流量都將透過位於公有子網路中的 NAT 閘道進行路由。這種設計允許向外的單向流量，同時阻止外部網路主動連入 。  
    
- **VPC 端點 (VPC Endpoints)：** 為了讓私有子網路中的資源能夠安全地存取其他 AWS 服務（如 Amazon S3、SageMaker API、KMS），我們將使用 VPC 端點。這使得內部流量可以在 AWS 的私有骨幹網路中傳輸，無需繞道公用網際網路，是強化安全性和合規性的關鍵控制措施 。  
    
- **網路 ACL 與安全群組 (Network ACLs & Security Groups)：** 採用深度防禦策略。網路存取控制清單（NACLs）作為無狀態的防火牆，在子網路層級進行粗略的流量過濾。而安全群組則提供有狀態的、更精細的防火牆規則，控制進出個別執行個體（如 EC2 或 Lambda）的流量 。  
    

**專用 AWS 帳戶策略：** 為了實現最高等級的隔離與風險控制，我們建議將開發、預備（staging）和生產環境部署在各自獨立的 AWS 帳戶中。這種做法可以有效防止非生產環境的操作意外影響到包含 PHI 的生產資料，並允許針對不同環境實施更嚴格的存取控制策略 。  

### 1.2. 大規模資料擷取：安全且具彈性的影片上傳

由於手術影片檔案通常體積龐大，標準的單次 HTTP 上傳方式在面對不穩定的網路環境時，可靠性極低。任何中斷都可能導致整個上傳過程失敗，嚴重影響使用者體驗。因此，我們必須採用專為大檔案設計的強健上傳機制。

**解決方案：S3 分段上傳 (Multipart Upload) 搭配預簽章 URL (Presigned URLs)** 這是業界公認的最佳實踐，能夠將大檔案上傳的複雜性從伺服器端轉移到用戶端，同時確保安全與高效 。  

- **伺服器端邏輯（由 API Gateway 觸發的 Lambda 函數執行）：**
    
    1. 用戶端向後端 API 發出請求，表明希望上傳一個特定檔案。
        
    2. 伺服器端的 Lambda 函數調用 S3 的 `CreateMultipartUploadCommand` API。S3 收到請求後，會初始化一個分段上傳任務，並返回一個全域唯一的 `UploadId` 。  
        
    3. 伺服器根據檔案大小計算需要分割的區塊數量。接著，針對每一個區塊，使用 `UploadPartCommand` 和前一步驟獲取的 `UploadId`，生成一個對應的預簽章 URL。這些 URL 提供了有時效性且權限極度受限的寫入權限，僅允許上傳指定的檔案區塊 。  
        
    4. 伺服器將 `UploadId` 和包含所有預簽章 URL 的陣列返回給用戶端。
        
- **用戶端邏輯（Web 應用程式）：**
    
    1. 用戶端的 JavaScript 程式碼將本地的大型影片檔案分割成多個較小的區塊（例如，每個 10 MB）。  
        
    2. 接著，它會以平行的方式，使用從伺服器獲取的預簽章 URL 陣列，透過 `PUT` 請求將每個檔案區塊獨立上傳到 S3。
        
    3. 每當一個區塊成功上傳後，S3 的回應標頭（Header）中會包含一個 `ETag` 值。用戶端必須收集並記錄每一個區塊對應的 `PartNumber` 和 `ETag` 。  
        
        **關鍵細節：** 為了讓瀏覽器能夠讀取 `ETag` 標頭，必須在 S3 儲存貯體的跨來源資源共用（CORS）設定中，明確地透過 `ExposeHeaders` 屬性暴露 `ETag` 。  
        
    4. 當所有區塊都上傳完成後，用戶端將收集到的 `PartNumber`/`ETag` 配對列表以及 `UploadId` 發送回伺服器。
        
- **伺服器端完成邏輯：**
    
    1. 伺服器接收到完成請求後，調用 S3 的 `CompleteMultipartUploadCommand` API，並附上 `UploadId` 和所有區塊的 `PartNumber`/`ETag` 資訊。
        
    2. S3 根據這些資訊，在後端將所有上傳的區塊按正確順序重新組裝成一個完整的影片檔案 。  
        

**效能增強：S3 傳輸加速 (Transfer Acceleration)** 對於地理位置較遠的使用者，啟用 S3 傳輸加速功能可以顯著提升上傳速度。此功能會將流量透過 AWS 優化的全球邊緣網路進行路由，繞過公共網際網路的擁塞路段，從而縮短資料傳輸時間 。  

### 1.3. 手術資料湖：儲存、組織與版本控制

一個健壯的資料湖是平台進行大規模分析的基礎。我們將以 Amazon S3 為核心，建立一個有組織、可追溯且成本優化的儲存系統。

- **Amazon S3 作為核心儲存庫：** S3 具備近乎無限的可擴展性、高達 99.999999999% 的持久性以及與 AWS 生態系統的深度整合，是儲存原始影片、處理後資料、模型產出物等非結構化資料的理想選擇 。  
    
- **儲存貯體策略與結構：** 清晰的目錄（前綴）結構對於資料組織、權限管理和自動化至關重要。我們建議採用以下結構：
    
    - `s3://<bucket-name>/raw-videos/{case-id}/video.mp4`
        
    - `s3://<bucket-name>/processed-metadata/{case-id}/analysis.json`
        
    - `s3://<bucket-name>/datasets/{dataset-name}/v1.2.0/`
        
    - `s3://<bucket-name>/models/{model-name}/{version-id}/model.tar.gz`
        
- **資料版本控制：** 版本控制是實現可重複性和可稽核性的關鍵 MLOps 實踐。
    
    - **S3 物件版本控制：** 在所有儲存 PHI 或關鍵 ML 產出物的 S3 儲存貯體上，應啟用原生的物件版本控制功能。這會為每一次的物件變更（包括覆寫和刪除）保留一個歷史版本，提供了一個不可變的變更記錄，對於稽核和意外刪除的災難復原至關重要。
        
    - **以目錄前綴進行明確版本控制：** 對於用於模型訓練的資料集，最佳實踐是為每個版本建立一個獨立的 S3 前綴，例如 `.../datasets/cardiac-surgery-data/v2.1/`。這種方式使得哪個模型使用了哪個版本的資料一目了然，極大地增強了實驗的可追溯性 。  
        
    - **使用 DVC 進行進階資料版本控制：** 為了達到程式碼與資料版本的完美對應，可以整合 Data Version Control (DVC)。DVC 的工作原理是將大型資料檔案的元數據（指標檔案）儲存在 Git 中，而實際的資料檔案則保留在 S3。當開發人員切換 Git 分支時，DVC 會自動同步對應版本的資料，實現了程式碼與資料的原子級同步，是實現最高等級可重複性的黃金標準 。  
        
- **成本優化：S3 生命週期策略：** 為了控制長期儲存成本，我們必須實施 S3 生命週期策略。這些策略可以設定規則，自動將較舊或不常存取的影片檔案從標準儲存層（S3 Standard）轉移到成本更低的儲存層，如 S3 標準-不常存取（S3 Standard-IA）或 S3 Glacier Flexible Retrieval，從而顯著降低資料歸檔的費用 。  
    

### 1.4. 身分、存取與稽核：合規性的三大支柱

建立強健的身分驗證、存取控制和稽核機制，是滿足 HIPAA 安全規則的技術核心。

- **IAM 最小權限原則：** 平台上的所有存取都將遵循「最小權限原則」。我們將為不同的使用者角色（如 `WebAppUserRole`, `DataScientistRole`）和服務（如 `SageMakerExecutionRole`, `LambdaExecutionRole`）建立專屬的 IAM 角色。每個角色都將附加一個精確定義的 IAM 政策，僅授予其完成任務所必需的最小權限集，嚴格限制不必要的存取 。  
    
- **無處不在的加密：**
    
    - **靜態加密 (At Rest)：** 所有儲存 PHI 的 S3 儲存貯體都將啟用伺服器端預設加密，並使用 AWS 金鑰管理服務（KMS）中由我們自己管理的客戶金鑰（CMK）。同樣，用於 SageMaker 執行個體和 RDS 資料庫的 EBS 磁碟區也將使用 KMS 進行加密。這是 HIPAA 的一項基本技術要求 。  
        
    - **傳輸中加密 (In Transit)：** 所有在服務之間以及與使用者之間傳輸的資料，都必須使用傳輸層安全性協定（TLS），即 HTTPS，進行加密 。  
        
- **全面的稽核日誌：**
    
    - **AWS CloudTrail：** 在所有 AWS 帳戶中啟用 CloudTrail，以記錄對 AWS 環境的每一次 API 調用。這會產生一個不可變的稽核軌跡，詳細記錄了「誰在何時做了什麼」，對於安全事件調查和合規性證明至關重要 。  
        
    - **AWS Config：** 使用 AWS Config 持續監控和記錄 AWS 資源的配置狀態。我們可以建立規則來自動檢測不合規的配置（例如，一個公開的 S3 儲存貯體或一個未加密的 EBS 磁碟區），並在發生違規時觸發警報或自動修復 。  
        
    - **Amazon CloudWatch Logs：** 集中收集來自應用程式、Lambda 函數和其他服務的日誌，以便進行監控、分析和警報。必須設定合理的日誌保留策略，以滿足法規要求並控制儲存成本 。  
        

為了將抽象的法規要求轉化為具體的技術實施，下表直接將 HIPAA 安全規則的關鍵控制項對應到相應的 AWS 服務和配置策略。這張表格不僅是架構設計的依據，也是未來向稽核員展示合規性姿態的有力證明。

|HIPAA 安全保障措施|法規要求|AWS 服務|實施細節|
|---|---|---|---|
|**存取控制 (Access Control)**|僅授權使用者存取 ePHI。實施基於角色的存取控制。|IAM, Amazon S3, Amazon VPC|遵循最小權限原則建立 IAM 角色。使用 S3 儲存貯體政策和 VPC 端點政策來限制僅從特定 VPC 和角色進行存取。|
|**稽核控制 (Audit Controls)**|記錄和檢查包含或使用 ePHI 的資訊系統中的活動。|AWS CloudTrail, AWS Config, Amazon CloudWatch|在所有帳戶中啟用 CloudTrail，並將日誌傳送到一個集中的、不可變的 S3 儲存貯體。使用 AWS Config 規則持續監控關鍵資源的合規性。|
|**完整性 (Integrity)**|保護 ePHI 免遭不當更改或破壞。|Amazon S3, AWS KMS|在 S3 儲存貯體上啟用物件版本控制，以保留所有資料版本的歷史記錄。使用 KMS 加密所有靜態資料，防止未經授權的修改。|
|**傳輸安全 (Transmission Security)**|保護透過電子網路傳輸的 ePHI。|Elastic Load Balancing (ELB), Amazon CloudFront, VPC|強制所有外部通訊使用 TLS 1.2 或更高版本的加密（HTTPS）。服務間通訊透過 VPC 內的私有網路進行，不暴露於公網。|

匯出到試算表

## 第 2 節：ML 分析管線：事件驅動的編排工作流程

本節將詳細闡述驅動平台核心分析功能的自動化引擎。我們將從一個簡單的「觸發-執行」模型，演進到一個強健、可觀測且可擴展的編排系統，該系統能夠高效地管理複雜的影片分析任務。

### 2.1. 編排策略：AWS Step Functions vs. 鏈式 Lambda

在設計一個涉及多個處理階段的複雜工作流程時，選擇正確的編排工具至關重要。一個看似簡單的方案是將多個 Lambda 函數鏈接起來，即一個 Lambda 完成後觸發下一個。然而，對於我們的手術影片分析場景，這種方法存在致命的缺陷。

鏈式 Lambda 的方法非常脆弱。它缺乏內建的狀態管理機制，一旦某個環節出錯，整個流程的狀態便難以追蹤和恢復。此外，Lambda 函數有最長 15 分鐘的執行時間限制，這對於可能需要數小時才能完成的影片轉碼或模型推論任務來說是遠遠不夠的。最後，這種鏈式結構難以視覺化，使得除錯和稽核變得極其困難和低效 。  

**為何 Step Functions 是此應用場景的更優選擇：** AWS Step Functions 是一個專為編排無伺服器應用和微服務而設計的視覺化工作流程服務。它完美地解決了鏈式 Lambda 的所有痛點。

- **狀態管理與持久性：** Step Functions 的標準工作流程（Standard Workflows）最長可以運行一年。這意味著它可以輕鬆地啟動一個長時間運行的 SageMaker 訓練或批次轉換任務，然後「暫停」自身，等待該任務完成後再繼續執行後續步驟。它可靠地管理著整個工作流程的狀態，即使中間步驟耗時數小時甚至數天 。  
    
- **視覺化工作流程與可稽核性：** Step Functions 提供了一個圖形化的主控台，將整個工作流程以狀態機流程圖的形式呈現出來。這使得開發人員可以直觀地理解、除錯和監控工作流程的每一步。對於合規性而言，這種視覺化的稽核軌跡極具價值，能夠清晰地展示 PHI 資料的完整處理路徑 。  
    
- **內建的錯誤處理與重試機制：** 複雜的錯誤處理邏輯，例如「如果此步驟失敗，則等待 1 分鐘後重試，最多重試 3 次，若仍然失敗，則發送通知到警報系統」，可以在 Step Functions 的定義中以宣告式的方式輕鬆設定，而無需在應用程式程式碼中編寫複雜的 try-catch-retry 邏輯 。  
    
- **直接的服務整合：** Step Functions 可以原生整合超過 220 種 AWS 服務，包括我們需要的 SageMaker、Glue 和 Lambda。這意味著 Step Functions 可以直接發起一個 SageMaker 批次轉換任務，並使用 `.sync` 整合模式等待其完成，從而大大減少了需要編寫的「黏合程式碼」(glue code)，使架構更簡潔、更易於維護 。  
    

這種將「編排邏輯」與「執行邏輯」分離的架構模式，是實現可擴展性和可維護性的基石。Step Functions 負責定義「做什麼」和「何時做」（工作流程），而 Lambda、Rekognition 和 SageMaker 則負責「如何做」（實際的運算）。這種解耦意味著我們可以獨立地更新某個分析模型（例如，改進分割模型的演算法），只需更新其對應的 SageMaker 任務的 Docker 容器和模型產出物即可，而整個 Step Functions 工作流程無需任何變動。這是微服務和強健 MLOps 實踐的核心理念，允許不同團隊（如資料管線團隊和 ML 模型團隊）獨立工作，而不會破壞整個系統。

### 2.2. 端到端的影片處理工作流程

以下是我們設計的端到端影片分析狀態機的詳細步驟：

- **觸發器 (Trigger)：** 當使用者透過前端介面成功將一個影片檔案上傳到 S3 的 `raw-videos/` 目錄後，S3 會發出一個 `ObjectCreated` 事件。這個事件會被設定為觸發我們的 Step Functions 狀態機，從而自動啟動整個分析管線。
    
- **第 1 步：影片預處理與驗證 (Lambda / Fargate 任務)**
    
    - 工作流程的第一步是一個 Lambda 函數。它的職責是驗證新上傳的影片檔案，例如檢查其格式、編碼和時長是否符合要求。
        
    - 對於需要大量運算資源或時間的預處理任務（如大型 4K 影片的轉碼），可以使用 AWS Fargate 任務來替代 Lambda，以突破其資源和時間限制。
        
    - 此步驟會調用 **AWS Elemental MediaConvert** 服務。MediaConvert 是一個廣播級的影片轉碼服務，可以將各種格式的來源影片標準化為我們 ML 模型所需的統一格式、解析度和影格率。它還可以提取關鍵元數據，如 SMPTE 時間碼 。處理完成的標準化影片將被儲存到 S3 的  
        
        `processed-videos/` 目錄中。
        
- **第 2 步：平行化分析 (Parallel State)**
    
    - 為了最大限度地縮短總處理時間，我們將使用 Step Functions 的 `Parallel` 狀態。這個狀態允許我們同時啟動多個獨立的分析分支，讓不同的 AI 模型可以並行處理同一個影片。
        
    - **分支 A：使用 Amazon Rekognition 進行基礎分析**
        
        - 其中一個分支將調用 Amazon Rekognition Video 服務，以快速獲取通用的影片分析結果。
            
        - Rekognition 可以高效地執行如偵測黑畫面、彩條、鏡頭切換等任務，這些資訊對於將影片進行邏輯分段非常有用 。它也可以進行通用的物件、人物和文字偵測 。  
            
        - 我們將使用 Rekognition 的非同步 API，這些 API 專為處理儲存在 S3 中的大型影片檔案而設計，能夠在後台處理任務並在完成後發出通知 。  
            
    - **分支 B, C, D...：使用 SageMaker 進行專業手術分析**
        
        - 其他幾個平行分支將分別觸發一個 **SageMaker 批次轉換 (Batch Transform) 任務**，每個任務對應一個我們自己訓練的專業手術分析模型（例如，器械偵測模型、手術階段分類模型、器官分割模型）。
            
        - 批次轉換是此場景的理想選擇，因為它是一種無伺服器、可自動擴展的推論方式，專門用於對儲存在 S3 中的大型資料集（在我們的案例中是單個大型影片檔）進行離線分析，而無需維護一個 24/7 運行的即時推論端點，從而極大化地節省成本 。  
            
        - Step Functions 會將處理後影片的 S3 路徑作為輸入傳遞給批次轉換任務，並設定為等待任務完成後再繼續。
            
- **第 3 步：結果彙總與儲存 (Lambda 任務)**
    
    - 當所有平行分析分支都成功完成後，一個最終的 Lambda 函數會被觸發。
        
    - 此函數的任務是從各個分支的輸出中收集分析結果（例如，來自 Rekognition 的 JSON、來自各個 SageMaker 任務的 JSON 或 CSV 檔案）。
        
    - 它會將所有這些零散的結果彙總成一個單一的、結構化的、全面的 JSON 文件。
        
    - 這個彙總後的元數據將被儲存在 **Amazon DynamoDB** 中。我們選擇 DynamoDB 是因為它提供了極低的延遲和可預測的效能，非常適合透過主鍵（`case-id`）快速查詢特定影片的分析結果 。  
        
    - 同時，我們也會在原始 S3 影片物件的元數據中，儲存一個指向 DynamoDB 中對應項目的指標，以便於交叉引用和資料溯源。
        

這種同時利用 AWS 託管 AI 服務（Rekognition）和自訂模型（SageMaker）的混合策略，形成了一種成本效益極高的分層分析方法。Rekognition 被用來處理那些「低垂的果實」，即快速、廉價地完成通用分析任務 。而那些更昂貴、更耗費運算資源的自訂 SageMaker 模型，則被保留用於執行 Rekognition 無法完成的高度專業化任務，例如識別特定型號的手術抓鉗或區分細微的手術階段差異。這種務實的設計在能力和經濟性之間取得了最佳平衡。  

## 第 3 節：開發與訓練先進的手術 AI 模型

本節將從基礎設施轉向核心的 AI/ML 開發，詳細探討平台所需的模型類型、最新的技術趨勢，以及如何在 Amazon SageMaker 上高效地進行訓練和實驗。

### 3.1. 手術場景理解的頂尖模型概覽

平台的分析能力直接取決於其所使用的 AI 模型的先進程度。因此，選擇正確的模型架構至關重要。以下是根據最新學術研究和業界實踐，針對不同手術分析任務的推薦模型類型。

- **器械/解剖結構的分割與偵測 (Segmentation & Detection)：**
    
    - **基於 CNN 的架構：** 卷積神經網路（CNN）仍然是醫學影像分割領域的基石。像 U-Net 及其變體（因其 U 型結構和跳躍連接而聞名，能有效融合深層語義特徵和淺層空間特徵）和 DeepLab 系列（以其空洞卷積 Atrous Convolution 擴大感受野而著稱）等模型，在分割較大的器官和結構時表現依然非常出色 。全卷積網路（FCN）也是一個基礎而有效的選擇 。  
        
    - **基於 Transformer 的架構：** 視覺 Transformer（ViT）正逐漸成為主流。近期最重要的進展是 **Segment Anything Model (SAM)**，這是一個由 Meta AI 發布的基礎模型。SAM 在零樣本（zero-shot）分割方面展現了驚人的能力，能夠在沒有任何額外訓練的情況下分割出圖像中的任何物體。透過在專業的手術資料集上進行微調，SAM 可以快速適應並精確分割手術器械和解剖結構，極大地降低了模型開發的門檻 。  
        
    - **實例分割 (Instance Segmentation)：** 當需要區分視野中多個相同類型的器械時（例如，兩把持針器），我們需要的是實例分割，而不僅僅是語義分割。像 **YOLOv8-seg** 這樣的模型能夠同時提供每個物件的邊界框（bounding box）和像素級的遮罩（mask），是完成此類任務的理想選擇 。  
        
- **手術階段辨識 (Video Stage Classification)：** 這是一個典型的時間序列分析問題，需要模型能夠理解影片中事件的先後順序和長期依賴關係。
    
    - **兩階段方法：** 一種常見且有效的方法是，首先使用一個強大的視覺模型（如 CNN 或 ViT）來逐幀提取高維度的特徵，然後將這些特徵序列輸入到一個專門的時間序列模型中進行學習 。  
        
    - **時間序列模型：**
        
        - **Transformers：** 由於其自註意力機制，Transformer 在捕捉序列中的長期依賴關係方面具有天然優勢，已成為影片理解領域的最新技術。像 TimeSformer、**Surgformer** 和  
            
            **MuST** 這樣的架構，專門為解決影片分析中的時間建模問題而設計，它們克服了傳統滑動窗口方法感受野有限的缺點 。  
            
        - **SlowFast 網路：** SF-TMN 模型提出了一種創新的雙路徑架構。其中，「Slow」路徑以較低的影格率處理影片，捕捉場景中變化緩慢的全局上下文（如手術階段的背景）；而「Fast」路徑則以高影格率處理，專注於捕捉變化迅速的局部動作（如器械的快速移動）。這種設計非常符合手術影片的特性 。  
            
        - **記憶體增強模型：** 「手術記憶」（Memory of Surgery, MoS）框架透過為標準的 Transformer 模型增加明確的長期和短期記憶體模組，來增強其對整個手術過程的理解。長期記憶體記錄了已經發生過的手術階段，而短期記憶體則保留了緊鄰當前時間窗口的視覺特徵，從而顯著提升了預測的時間連貫性 。  
            
- **器械追蹤 (Instrument Tracking)：** 此任務通常採用「偵測後追蹤」（tracking-by-detection）的策略。首先，在影片的每一幀上運行一個高效能的物件偵測器（如 YOLOv8）。然後，使用一個追蹤演算法（如卡爾曼濾波器 Kalman Filter 或基於交併比 IOU 的簡單追蹤器）來關聯連續幀之間的偵測結果，從而形成每個器械的運動軌跡。
    

一個重要的趨勢是，AI 領域正從為特定任務從頭開始訓練專用模型，轉向對大型、預訓練的基礎模型（Foundation Models）進行微調。這代表著一種範式轉移。例如，我們不再需要從零開始建立一個「膽囊切除術器械分割模型」，而是採用一個通用的、強大的視覺基礎模型（如 SAM），然後用一個相對較小的、有標註的手術資料集來對其進行微調，使其適應我們的特定任務 。這大大降低了開發高效能模型的資料需求和運算成本，因此我們的 MLOps 策略必須圍繞「微調」工作流程來設計，而非「從頭訓練」。  

### 3.2. 在 Amazon SageMaker 上進行訓練與實驗

Amazon SageMaker 提供了一個全託管的平台，可以簡化和加速 ML 模型的訓練過程。

- **SageMaker 訓練任務 (Training Jobs)：** 我們將利用 SageMaker 的託管訓練功能。這意味著 SageMaker 會在後台自動處理基礎設施的佈建、配置和管理（例如，啟動 GPU 執行個體、安裝驅動程式等），讓資料科學家可以專注於模型演算法的開發 。  
    
- **使用 SageMaker Python SDK：** 資料科學家將在熟悉的 Python 環境中（如 SageMaker Studio 中的 Jupyter 筆記本）使用 SageMaker Python SDK 來定義和啟動訓練任務。這個過程通常包括：
    
    1. 建立一個 `Estimator` 物件。根據所使用的框架，可以是 `PyTorch`、`TensorFlow` 或 `HuggingFace` 的 Estimator 。  
        
    2. 在 Estimator 中指定關鍵參數，包括：
        
        - `entry_point`：訓練腳本的名稱（例如 `train.py`）。
            
        - `source_dir`：包含訓練腳本和任何依賴項的目錄。
            
        - `instance_type`：用於訓練的執行個體類型（例如，`ml.g5.xlarge`，這是一種配備了 GPU 的執行個體）。
            
        - `instance_count`：用於分散式訓練的執行個體數量。
            
        - `hyperparameters`：傳遞給訓練腳本的超參數字典。
            
    3. 定義資料輸入通道，將 S3 中經過版本控制的資料集路徑與訓練任務關聯起來 。  
        
    4. 呼叫 Estimator 的 `.fit()` 方法來啟動訓練任務。SageMaker 隨後會自動完成所有後續工作：佈建執行個體、從 S3 下載資料、運行訓練腳本、並在訓練結束後將生成的模型產出物（如模型權重）上傳回 S3 。  
        
- **利用預建容器：** 對於像 PyTorch、TensorFlow 或 Hugging Face 這樣的主流框架，SageMaker 提供了預先建置和優化的 Docker 容器。使用這些容器可以免去我們自己管理 Dockerfile 和依賴項的麻煩，進一步加速開發流程 。  
    
- **自備容器 (Bring Your Own Container, BYOC)：** 如果我們的模型需要高度自訂的環境或特殊的軟體依賴，我們也可以建立自己的 Docker 容器。將這個自訂容器推送到 Amazon ECR（彈性容器註冊表）後，只需在 SageMaker Estimator 中指定其 URI 即可使用 。  
    

手術影片分析並非單一問題，而是一個由不同時間粒度的任務組成的層級體系 。因此，平台的「AI 引擎」不會是單一的巨型模型，而是一個由多個專業模型組成的集合。例如，一個為逐幀分割而優化的模型（如 SAM）其設計本身並不適用於需要長期時間推理的任務（如手術階段辨識）。因此，在第二節中討論的 Step Functions 編排層變得更加關鍵，它負責管理這個模型集合的執行順序，並將它們的輸出融合在一起，從而提供一個全面的、多粒度的手術場景理解。  

## 第 4 節：生產級的 MLOps 框架

本節將詳細闡述將 AI 模型從實驗室推向生產環境所需的自動化、治理和卓越營運實踐。一個強健的 MLOps 框架是將一個研究專案轉化為一個可靠、可維護的醫療產品的關鍵。

### 4.1. 實驗追蹤：確保可重複性與高效比較

機器學習是一個高度迭代的過程。為了確保研究的可重複性，並能夠系統性地比較不同實驗的結果，我們必須追蹤每一次實驗的所有相關資訊，包括程式碼版本、資料集版本、超參數配置以及最終的模型效能指標 。  

- **工具選擇：Weights & Biases (wandb) 和/或 ClearML** 這兩者都是業界領先的 MLOps 平台，提供強大的實驗追蹤功能，並且都能與 Amazon SageMaker 無縫整合。
    
    - **整合方式：** 整合過程非常簡單。只需在 SageMaker 的訓練腳本（例如 `train.py`）中加入幾行程式碼即可。對於 wandb，是 `import wandb; wandb.init()`；對於 ClearML，則是 `from clearml import Task; Task.init()` 。  
        
    - **身份驗證：** MLOps 工具的 API 金鑰會作為環境變數，安全地傳遞給 SageMaker 訓練任務，使其能夠向對應的平台回報資訊 。  
        
    - **自動記錄：** 這些工具能夠自動捕獲大量的實驗元數據，包括超參數、系統指標（如 CPU/GPU 使用率）、主控台日誌等。同時，它們可以輕鬆配置，以記錄關鍵的評估指標（如 `wandb.log({"accuracy": acc})`）、模型檢查點和視覺化圖表 。  
        
    - **AWS 原生方案：** Amazon SageMaker Experiments 是 AWS 提供的原生實驗追蹤解決方案。它將工作組織成「實驗 (Experiment)」、「試驗 (Trial)」和「試驗元件 (Trial Component)」，並記錄相關的產出物和指標 。雖然功能強大，但像 wandb 和 ClearML 這樣的第三方工具通常在使用者介面、視覺化和團隊協作方面提供更成熟和豐富的功能集。  
        

### 4.2. 模型與資料治理：建立單一事實來源

- **資料版本控制：** 如第 1.3 節所述，使用 S3 物件版本控制和明確的目錄前綴是基礎。而整合 DVC 則提供了程式碼和資料版本之間最強健的連結，是實現完全可重複性的最佳實踐 。  
    
- **模型版本控制與目錄：Amazon SageMaker 模型註冊表 (Model Registry)**
    
    - 模型註冊表是所有已批准模型版本的集中式、受治理的儲存庫，是 MLOps 流程的核心樞紐 。  
        
    - 當一個 SageMaker 訓練任務成功完成後，其產生的模型產出物會被註冊到一個「模型群組」（Model Group）中，作為一個新的模型版本。例如，我們可以建立一個名為「surgical-instrument-segmentation」的模型群組。
        
    - 註冊表會儲存與每個模型版本相關的豐富元數據，包括其來源的訓練任務 ARN、在驗證集上的效能指標（如準確率、F1 分數），以及模型產出物在 S3 中的路徑 。  
        
    - **審批工作流程：** 這是模型註冊表最關鍵的治理功能。每個模型版本都有一個審批狀態（例如 `Pending`、`Approved`、`Rejected`）。我們可以設定規則，只有狀態為 `Approved` 的模型才能被部署到生產環境。這個審批步驟可以由團隊負責人手動完成，也可以作為 CI/CD 管線中的一個自動化步驟，在模型通過所有測試後自動批准 。  
        

### 4.3. CI/CD for ML：自動化通往生產之路

我們將使用 GitHub Actions 作為 CI/CD 工具，因為它與程式碼儲存庫原生整合，功能強大且易於使用 。  

- **CI/CD 工作流程（在合併到 `main` 分支時觸發）：**
    
    1. **配置 AWS 憑證：** GitHub Actions 的執行器（runner）將使用 OpenID Connect (OIDC) 的方式安全地向 AWS 進行身份驗證，獲取臨時的 IAM 憑證。這是比儲存長期存取金鑰更安全的方法 。  
        
    2. **建置與測試：** 工作流程首先會檢查程式碼，運行單元測試和程式碼風格檢查（linting）。
        
    3. **建置 Docker 容器 (若使用 BYOC)：** 如果使用了自訂的訓練或推論容器，工作流程會建置 Docker 映像檔，並將其推送到 Amazon ECR 。  
        
    4. **觸發 SageMaker 管線 (Pipeline)：** 這是核心步驟。GitHub Action 將使用 AWS CLI 或 Boto3 來啟動一個 **SageMaker Pipeline** 的執行。SageMaker Pipeline 是將實驗性筆記本轉化為生產級工作流程的關鍵。它是一個由多個步驟（如資料處理、訓練、評估、註冊）組成的有向無環圖（DAG），完全由程式碼定義，確保了生產流程的標準化和可重複性 。  
        
    5. **SageMaker Pipeline 執行內容：**
        
        - **預處理步驟：** 在最新的版本化資料集上運行一個 SageMaker Processing 任務。
            
        - **訓練步驟：** 使用程式碼儲存庫中的最新程式碼來訓練模型。
            
        - **評估步驟：** 在一個保留的測試集上評估新訓練出的模型的效能。
            
        - **條件式註冊步驟：** 只有當新模型的效能（例如，準確率）超過預先定義的閾值時，才會將其註冊到 SageMaker 模型註冊表中，並將其狀態設定為 `Pending` 。  
            
    6. **手動審批閘門 (在 GitHub Actions 中)：** GitHub Actions 可以配置一個名為「環境 (environment)」的保護規則，要求在部署到生產環境之前，必須由指定的團隊負責人進行手動審批 。  
        
    7. **部署：** 獲得批准後，GitHub Actions 工作流程中的最後一個作業將從模型註冊表中獲取被標記為 `Approved` 的模型，並執行部署操作（在我們的案例中，這意味著更新批次轉換任務所使用的模型參考）。
        

### 4.4. 自動化模型再訓練：閉合 MLOps 迴圈

一個生產級的 ML 系統必須能夠適應資料的變化。

- **觸發機制：**
    
    - **資料漂移偵測：** 使用 **SageMaker Model Monitor** 持續監控輸入到模型進行推論的即時資料的統計分佈。如果偵測到資料分佈與訓練時的資料分佈發生了顯著的「漂移」（drift），Model Monitor 可以自動觸發一個 CloudWatch 事件 。  
        
    - **排程再訓練：** 使用 **Amazon EventBridge** 設定一個排程規則，定期（例如，每週或每月）在新增的、已標註的資料上觸發再訓練管線 。  
        
- **自動化工作流程：** 無論是來自漂移偵測還是排程，觸發的 CloudWatch/EventBridge 事件都將啟動一個專用於再訓練的 Step Functions 狀態機。這個狀態機執行的內容，與 CI/CD 流程中使用的 SageMaker Pipeline 完全相同。這確保了無論是程式碼變更驅動的更新，還是資料驅動的自動再訓練，都使用同一套經過測試和驗證的標準化流程 。  
    

SageMaker 模型註冊表在這個 MLOps 框架中扮演著至關重要的角色，它作為一個解耦點，有效地將資料科學團隊和維運（DevOps）團隊的職責分開。資料科學家的工作流程終點是將一個候選模型 `register_model()` 到註冊表。而維運團隊的部署管線的起點則是從註冊表中 `get_latest_approved_model()`。`ApprovalStatus` 欄位是這個交接過程中的關鍵控制閥門。這使得資料科學家可以在他們的環境中自由地進行數百次實驗，而生產環境則受到保護，因為它只會從一個經過審核和批准的模型列表中拉取模型，從而極大地降低了部署劣質模型的風險。

下表總結了我們建議的 MLOps 工具鏈及其整合策略，為工程團隊提供了一個清晰的實施指南。

|MLOps 階段|主要工具|整合點|關鍵配置 / 程式碼片段|
|---|---|---|---|
|**實驗追蹤**|Weights & Biases (wandb)|SageMaker 訓練腳本 (`train.py`)|在 Estimator 中設定環境變數 `environment={"WANDB_API_KEY": api_key}`；在腳本中加入 `import wandb; wandb.init()` 。|
|**程式碼與資料版本控制**|Git (GitHub) & DVC|開發工作流程|使用 `git` 追蹤程式碼變更，使用 `dvc` 追蹤 S3 中的資料集版本，並將 DVC 元數據檔案提交到 Git 。|
|**模型治理與目錄**|SageMaker Model Registry|SageMaker Pipeline|在管線的最後一步，使用 `RegisterModel` 步驟將通過評估的模型註冊到指定的模型群組中 。|
|**CI/CD 與管線編排**|GitHub Actions & SageMaker Pipelines|Git Repository & AWS API|GitHub Actions 工作流程在 `push` 事件時觸發，透過 AWS CLI/SDK 啟動一個預先定義好的 SageMaker Pipeline 。|
|**自動化再訓練觸發**|SageMaker Model Monitor & EventBridge|CloudWatch Events|設定 Model Monitor 偵測資料漂移，或設定 EventBridge 排程規則，兩者皆可觸發 Step Functions 來執行再訓練管線 。|

## 第 5 節：進階功能：用於手術智慧的視覺語言模型 (VLM)

本節將探討如何為平台架構未來的功能，特別是整合次世代的視覺語言模型（VLM），以實現更深層次的影片理解和智慧報告生成。

### 5.1. VLM 整合架構

視覺語言模型（VLM）是一種能夠同時處理和理解視覺資訊（如圖像或影片）和文字資訊的多模態 AI 模型 。這項技術使得如視覺問答（Visual Question Answering, VQA）和生成詳細影片描述等進階應用成為可能。  

- **VLM 架構簡介：** 大多數現代 VLM 由三個核心元件組成：一個用於提取視覺特徵的圖像編碼器（通常是 ViT）、一個用於理解文字的文字編碼器，以及一個將兩種模態的特徵表示融合起來的融合模組 。業界的知名模型包括 Google 的 Florence-2、阿里的 CogVLM 以及 Meta 的 Llama 3.2-Vision 。  
    
- **建議的整合工作流程：**
    
    1. **VLM 微調：** 我們不會從頭開始建構一個 VLM。相反，我們將選擇一個性能優越的開源 VLM，並在我們自己的手術影片資料集上對其進行微調。這個過程將透過一個 SageMaker 訓練任務來完成，與第 3 節中描述的流程類似 。訓練資料將由影片影格與對應的文字描述或問答對組成。  
        
    2. **VLM 推論路徑：** 這將成為我們主分析工作流程（Step Functions）中的一個新的平行分支，或者是一個可以按需觸發的獨立工作流程。
        
    3. **VLM 輸入：** VLM 可以將整個影片（或從中提取的關鍵影格）作為視覺輸入，同時接收一個文字提示（例如，「請總結這次手術的關鍵步驟」或「影片 15 分 30 秒時，主刀醫生使用的是哪種器械？」）。
        
    4. **VLM 輸出：** VLM 將生成豐富的文字輸出，例如手術過程的詳細描述、關鍵事件的總結，或對特定問題的回答。
        

### 5.2. 使用大型語言模型 (LLM) 生成結構化洞見與報告

平台的最終目標不僅是提取結構化數據（如邊界框座標），更是要將這些數據轉化為人類專家（如醫生或研究員）可以輕鬆理解的敘述性報告。

- **工具：OpenAI API 或 Amazon Bedrock 搭配結構化輸出**
    
    - **OpenAI 的結構化輸出：** OpenAI 最新的模型（如 GPT-4o）支援一種名為 `response_format` 的參數，可以設定為 `{"type": "json_schema"}`。這個功能可以強制模型的輸出嚴格遵守我們預先定義的 JSON 結構，確保了輸出的可靠性和可解析性，無需進行複雜的提示工程 。  
        
    - **Amazon Bedrock：** 這是 AWS 提供的全託管服務，可以讓使用者透過 API 存取來自多家頂尖 AI 公司（包括 Anthropic、Meta、Cohere 等）的基礎模型。使用 Bedrock 的主要優勢是，它在 AWS 生態系統內運行，可以更容易地滿足 HIPAA 合規性要求 。  
        
- **工作流程：**
    
    1. 當主分析管線完成後，一個 Lambda 函數會被觸發。
        
    2. 該函數從 DynamoDB 中檢索彙總後的分析結果 JSON。
        
    3. 它根據這些數據建構一個詳細的提示（prompt），例如：「你是一位專業的外科手術分析師。請根據以下 JSON 格式的手術事件數據，生成一份敘述性的手術報告。報告應包括手術摘要、關鍵事件時間軸、器械使用統計和任何可觀察到的異常情況。」
        
    4. 同時，它會定義一個目標報告的 `json_schema`，明確規定報告的結構（例如，包含 `procedure_summary`、`key_events`、`instrument_usage_statistics` 等欄位）。
        
    5. 它調用 LLM 的 API（OpenAI 或 Bedrock），並將提示和 `response_format` 結構一起傳遞。
        
    6. **HIPAA 關鍵注意事項：** 將任何包含 PHI 的資料傳送到像 OpenAI 這樣的第三方 API，必須與該供應商簽訂 BAA。如果無法獲得 BAA，則此步驟**必須**替換為在我們自己的安全 AWS 環境中部署的開源 LLM（例如，使用 SageMaker 端點託管 Llama 3）。由於 Amazon Bedrock 是 AWS 的原生服務，它更有可能被 BAA 所涵蓋，因此是首選的、更合規的替代方案 。  
        
    7. 從 LLM 接收到結構化的 JSON 報告後，該函數會將其儲存回 DynamoDB 或直接呈現給使用者。
        

這個架構設計將平台的價值鏈清晰地劃分為兩個階段：**感知 (Perception)** 和 **認知 (Cognition)**。第 2 節中的核心分析任務（偵測、分割）本質上是「感知」——識別影片中「有什麼」，其輸出是機器可讀的結構化數據。而本節中的 VLM/LLM 任務則是「認知」——理解被感知到的事物的「意義和上下文」，其輸出是人類可讀的敘述性洞見。這種模組化的兩階段流程極具擴展性。未來，我們可以輕鬆地加入新的「感知」模型（例如，一個用於估計失血量的模型），其輸出將自動成為「認知」引擎的輸入，豐富其報告內容，而無需重新訓練 LLM。

## 第 6 節：效能與成本優化

本節將直接回應使用者對平台高效能和經濟效益的非功能性需求，探討如何從模型推論和雲端資源使用兩個層面進行深度優化。

### 6.1. 推論時間優化：讓分析更快

縮短模型推論時間不僅能提升使用者體驗，也是降低成本的有效途徑。

- **使用 Amazon SageMaker Neo 進行模型編譯：**
    
    - SageMaker Neo 是一個模型編譯器，它可以將在 PyTorch、TensorFlow 等框架中訓練好的模型，轉換為針對特定目標硬體（如某款 CPU 或 GPU 執行個體）高度優化的可執行檔 。  
        
    - 這個編譯過程是在模型部署前進行的一次性操作。經過 Neo 優化後，模型在推論時的延遲可以顯著降低，吞吐量則能大幅提升，且通常不會犧牲模型的準確性。
        
- **模型量化 (Quantization)：**
    
    - 量化是一種模型壓縮技術，它將模型中的權重和活化值從高精度浮點數（如 32-bit float）降低到低精度整數（如 8-bit integer）。  
        
    - **優點：**
        
        1. **更小的模型體積：** 降低了模型的儲存空間和從 S3 下載到推論執行個體的時間。
            
        2. **更快的運算速度：** 在支援低精度運算的硬體上（特別是帶有 VNNI 指令集的現代 CPU 和最新的 GPU），整數運算比浮點數運算快得多。
            
    - **實施方式：** SageMaker 的大型模型推論（LMI）容器原生支援多種先進的量化技術，如 GPTQ、AWQ 和 SmoothQuant 。對於自訂模型，可以在 SageMaker Processing 任務中使用像 Intel Neural Compressor 這樣的函式庫來生成量化後的模型版本 。  
        
    - **效能增益：** 基準測試顯示了顯著的效能提升。對於大型語言模型，使用 TensorRT-LLM 搭配量化技術，平均可將延遲降低 33%，並將吞吐量提高 60% 。對於在 CPU 上運行的模型，INT8 量化在 C6i 執行個體上的效能相較於 FP32 可提升高達 4 倍 。  
        

效能優化和成本優化並非總是相互對立的，它們實際上是同一枚硬幣的兩面。像 Neo 編譯和量化這樣的技術，透過提升推論速度來增加單一執行個體的處理能力（吞吐量）。更高的吞吐量意味著我們可以用更少的執行個體來處理相同的工作負載，這直接導致了運算成本的降低。因此，在工程上投入時間進行推論優化，不僅是為了追求極致效能，更是一項直接的成本削減策略。

### 6.2. 雲端成本優化策略

除了透過模型優化來降低推論成本外，我們還必須從雲端資源的整體使用上進行系統性的成本管理。

- **運算成本：**
    
    - **使用 EC2 Spot 執行個體進行訓練：** 對於可以容忍中斷的 SageMaker 訓練任務（例如，大多數實驗性或非緊急的再訓練任務），使用 Spot 執行個體可以節省高達 90% 的運算成本。SageMaker 能夠管理 Spot 執行個體的中斷，並從檢查點自動恢復訓練 。  
        
    - **精準選擇執行個體 (Right-Sizing)：** 避免過度佈建是節省成本的關鍵。應使用 AWS Compute Optimizer 和 SageMaker 的監控指標來為訓練和推論任務選擇最合適的執行個體類型和大小 。  
        
    - **優化 Lambda 配置：** 對於 Lambda 函數，應使用不同的記憶體配置進行基準測試。有時，分配更多記憶體（從而獲得更多 CPU 資源）可以讓函數運行得更快，從而因執行時間縮短而降低總體的 `GB-秒` 成本 。  
        
- **儲存成本：**
    
    - **S3 智慧分層 (Intelligent-Tiering)：** 對於存取模式未知或多變的資料，S3 智慧分層儲存類別可以自動在頻繁存取和不頻繁存取層之間移動物件，以優化儲存成本。
        
    - **S3 生命週期策略：** 如前所述，明確設定生命週期規則，將舊的歸檔資料自動轉移到成本極低的 Glacier 或 Deep Archive 儲存層 。  
        
    - **清理產出物：** 定期刪除不再需要的 EBS 快照、中間資料產出物和過時的模型版本，以避免不必要的儲存費用 。  
        
- **營運成本：**
    
    - **無伺服器優先 (Serverless First)：** 我們提出的架構大量採用無伺服器元件（Lambda, Step Functions, S3, DynamoDB, SageMaker Batch Transform）。這意味著平台在沒有影片處理時，幾乎不產生運算成本，從根本上消除了閒置資源的浪費 。  
        
    - **日誌優化：** 預設情況下，CloudWatch Logs 會永久儲存日誌。必須為日誌群組設定一個合理的保留期限（例如，90 天或 1 年），以避免儲存成本的無限增長 。  
        
    - **預算與警報：** 使用 **AWS Budgets** 設定支出預算，並為實際成本和預測成本建立警報。這提供了必要的財務可見性，能夠在成本超支之前主動介入，防止意外的帳單衝擊 。  
        

最重要的成本節約來自於架構層面的選擇，而非微觀層面的優化。選擇一個無伺服器、事件驅動的架構，其節省的潛力遠大於事後清理幾個閒置資源。

## 第 7 節：使用者介面平台：視覺化手術分析

本節將勾勒出終端使用者如何與平台生成的豐富數據進行互動，將後端的複雜分析轉化為直觀、可操作的前端體驗，從而完成端到端的價值傳遞。

### 7.1. 前端架構概覽

我們建議採用現代化的單頁應用程式（Single-Page Application, SPA）架構，使用如 React、Angular 或 Vue.js 等主流框架進行開發。

- **託管與交付：** 前端應用程式的靜態檔案（HTML, CSS, JavaScript）可以託管在 AWS Amplify 或 Amazon S3 上，並搭配 Amazon CloudFront 進行全球內容分發。CloudFront 作為一個內容分發網路（CDN），可以將應用程式快取到全球各地的邊緣節點，為不同地區的使用者提供最低的載入延遲。
    
- **後端通訊：** 前端將透過一個安全的 **Amazon API Gateway** 與後端服務進行通訊。API Gateway 會將前端的請求路由到對應的 AWS Lambda 函數，以處理使用者身份驗證、影片列表查詢、以及從 DynamoDB 檢索特定影片的分析結果等操作。
    

### 7.2. 互動式手術影片播放器

一個功能強大的互動式播放器是將後端分析結果轉化為臨床洞見的核心。如果沒有有效的視覺化，後端產生的海量 JSON 數據對外科醫生來說是沒有意義的。

- **核心技術：Video.js** Video.js 是一個廣受歡迎且高度可擴展的開源 HTML5 影片播放器函式庫，擁有豐富的插件生態系統，非常適合我們的自訂需求 。  
    
- **顯示時間軸事件：**
    
    - 我們將使用 **`videojs-markers`** 插件。這個插件可以在影片播放器的時間軸上放置視覺標記 。  
        
    - 這些標記將代表從 DynamoDB 中檢索到的手術階段的開始和結束時間。當使用者將滑鼠懸停在一個標記上時，可以彈出一個提示框，顯示該階段的名稱（例如，「游離乙狀結腸」或「縫合」）。
        
- **顯示視覺化註釋：**
    
    - 我們將使用 **`videojs-overlay`** 插件或自訂的 HTML/CSS 疊加層，在影片播放時動態地顯示各種視覺註釋 。  
        
    - 後端分析產生的邊界框、分割遮罩等資訊，會以時間序列的形式儲存在 DynamoDB 的 JSON 文件中。
        
    - 前端的 JavaScript 程式碼會監聽播放器的 `timeupdate` 事件，這個事件會在播放時間改變時頻繁觸發。
        
    - 在每個時間點，前端程式碼會從已載入的分析數據中，查詢對應該影格的註釋資訊（如器械的邊界框座標、器官的分割多邊形）。
        
    - 然後，它會動態地建立或更新疊加層，將這些邊界框和半透明的分割遮罩精確地繪製在影片畫面上。透過將疊加層的 `start` 和 `end` 時間設定為極短的間隔，我們可以實現逐幀更新的視覺效果 。  
        
- **實現互動性與人機協同：**
    
    - 這些疊加層不僅僅是用於顯示，更可以設計成互動式的。例如，使用者可以點擊一個被偵測到的手術器械的邊界框，系統隨即可以彈出一個側邊欄，顯示該器械的詳細資訊，或者提供該器械在整個手術過程中的使用時長統計。
        
    - 更進一步，這個互動式播放器可以成為一個強大的「人在迴路」（human-in-the-loop）的資料標註和回饋系統。AI 的分析結果不會是 100% 完美的。當一位專家使用者（如經驗豐富的外科醫生）在觀看影片時發現一個錯誤——例如，一個錯誤的邊界框，或一個被遺漏的分割區域——我們可以提供工具讓他進行修正。
        
    - 受到像 `marker.js` 這樣的註釋函式庫的啟發 ，我們可以為播放器增加編輯功能，允許使用者拖動邊界框、重新繪製分割遮罩或更正手術階段的標籤。  
        
    - 這些由專家修正過的數據是極其寶貴的。它們可以被回傳到後端，作為高品質的訓練資料，用於下一代模型的迭代和優化。這就將平台從一個單向的分析工具，轉變為一個動態的、能夠自我完善的生態系統。使用者使用得越多，修正得越多，平台的 AI 模型就會變得越聰明。這種「飛輪效應」將構成平台長期的、難以被複製的核心競爭優勢。
        

## 結論與建議

本報告詳細闡述了一個專為手術影片分析設計的、全面的雲端原生平台架構。該架構立足於 AWS，並以 HIPAA 合規性為不可動搖的基石，整合了事件驅動的無伺服器計算、生產級的 MLOps 框架以及前瞻性的 AI 技術。透過遵循本報告提出的設計原則和技術路徑，醫療技術組織能夠建構一個不僅功能強大，而且安全、可擴展且具備成本效益的次世代手術智慧平台。

**核心建議如下：**

1. **堅持合規優先的設計原則：** 安全與合規並非事後添加的特性，而是必須在專案啟動之初就融入到每一個架構決策中的核心驅動因素。從 VPC 網路隔離到資料的端到端加密，再到全面的稽核日誌，每一項控制措施都應嚴格實施，以保護敏感的 PHI 資料。
    
2. **全面擁抱無伺服器與事件驅動架構：** 應優先採用 AWS Step Functions 進行複雜工作流程的編排，並結合 AWS Lambda、Amazon S3 和 DynamoDB。這種架構模式不僅能提供卓越的可擴展性和彈性，還能透過「按用量付費」的模式將營運成本降至最低。
    
3. **建立系統化的 MLOps 實踐：** 應將 MLOps 視為產品開發的核心流程，而非可選的附加項。利用 SageMaker Pipelines、Model Registry 以及像 wandb 或 ClearML 這樣的專業工具，建立從實驗追蹤到自動化部署和再訓練的完整閉環。這將是確保模型品質、加速迭代速度和實現長期可維護性的關鍵。
    
4. **分層實現 AI 分析能力：** 採用混合策略，結合 AWS 託管 AI 服務（如 Amazon Rekognition）進行快速、低成本的基礎分析，並將自訂訓練的專業模型（在 SageMaker 上開發）用於解決高度複雜的、特定領域的手術場景理解任務。這種分層方法在功能和成本之間取得了最佳平衡。
    
5. **前瞻性地佈局 VLM 和 LLM 應用：** 應將平台設計為一個可擴展的生態系統，將核心的視覺「感知」能力與上層的語言「認知」能力分離。這將使平台未來能夠無縫整合 VLM 和 LLM，將原始的分析數據轉化為具有深刻臨床意義的敘述性報告和互動式問答，從而創造出更高的應用價值。
    
6. **將使用者體驗置於核心地位：** 平台的最終價值體現在其前端的視覺化和互動能力上。應投入大量資源設計和開發一個直觀、高效的互動式影片播放器，將複雜的後端分析結果以清晰、可操作的方式呈現給終端使用者。考慮加入「人在迴路」的回饋機制，讓平台能夠在使用中不斷學習和進化。
    

總之，建構這樣一個平台是一項複雜的系統工程，它要求在雲端基礎設施、資料工程、機器學習和軟體開發等多個領域都具備深厚的專業知識。然而，透過採用本報告中概述的現代雲端原生架構和 MLOps 最佳實踐，可以顯著降低技術風險，加速產品上市時間，並最終打造出一個能夠真正賦能外科醫生、推動醫療研究並改善病患預後的革命性工具。




### 手術影片分析平台實作指南

本指南將逐步詳述如何建構一個功能強大、合規且可擴展的手術影片分析平台。每個章節都包含核心概念解釋以及可執行的 Python 程式碼片段，主要使用 AWS SDK for Python (Boto3) 和 Amazon SageMaker Python SDK。

### 第 1 節：基礎設施與資料擷取

此階段的目標是建立一個安全的基礎，用於儲存和處理敏感的醫療資料。

#### 1.1. 環境設定與 S3 儲存貯體

首先，我們需要一個符合 HIPAA 規範的 S3 儲存貯體來存放影片。這意味著我們必須啟用伺服器端加密和物件版本控制。

**使用 Boto3 建立 S3 儲存貯體**

```Python
import boto3
import logging

# 設定 Boto3 客戶端
s3_client = boto3.client('s3')
bucket_name = "your-hipaa-compliant-surgical-videos" # 替換為您唯一的儲存貯體名稱
region = boto3.session.Session().region_name

try:
    # 在特定區域建立儲存貯體 (如果不是 us-east-1)
    if region!= "us-east-1":
        s3_client.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration={'LocationConstraint': region}
        )
    else:
        s3_client.create_bucket(Bucket=bucket_name)

    # 1. 啟用伺服器端加密 (SSE-S3)
    s3_client.put_bucket_encryption(
        Bucket=bucket_name,
        ServerSideEncryptionConfiguration={
            'Rules':
        }
    )

    # 2. 啟用物件版本控制
    s3_client.put_bucket_versioning(
        Bucket=bucket_name,
        VersioningConfiguration={'Status': 'Enabled'}
    )

    # 3. 封鎖所有公開存取
    s3_client.put_public_access_block(
        Bucket=bucket_name,
        PublicAccessBlockConfiguration={
            'BlockPublicAcls': True,
            'IgnorePublicAcls': True,
            'BlockPublicPolicy': True,
            'RestrictPublicBuckets': True
        }
    )

    logging.info(f"儲存貯體 '{bucket_name}' 已成功建立並設定加密與版本控制。")

except Exception as e:
    logging.error(f"建立儲存貯體時發生錯誤: {e}")

```

#### 1.2. 實作大型影片檔案的安全分段上傳

為了可靠地處理大型檔案，我們將使用 S3 分段上傳搭配預簽章 URL。

**第 1 步：後端 - 建立啟動上傳的 Lambda 函數**

此 Lambda 函數由 API Gateway 觸發，負責初始化上傳並生成各個區塊的預簽章 URL。

```Python
# lambda_function.py (用於啟動上傳)
import boto3
import json
import os
from botocore.client import Config

s3_client = boto3.client('s3', config=Config(signature_version='s3v4'))
BUCKET_NAME = os.environ.get('BUCKET_NAME')
# 每個區塊的大小 (例如 10MB)
PART_SIZE_BYTES = 10 * 1024 * 1024

def lambda_handler(event, context):
    body = json.loads(event.get('body', '{}'))
    file_name = body.get('fileName')
    file_size = int(body.get('fileSize'))

    if not file_name or not file_size:
        return {'statusCode': 400, 'body': json.dumps('缺少 fileName 或 fileSize')}

    # 1. 初始化分段上傳
    multipart_upload = s3_client.create_multipart_upload(
        Bucket=BUCKET_NAME,
        Key=f"raw-videos/{file_name}" # 將影片存放在 raw-videos/ 目錄下
    )
    upload_id = multipart_upload['UploadId']

    # 2. 計算區塊數量並生成預簽章 URL
    num_parts = (file_size + PART_SIZE_BYTES - 1) // PART_SIZE_BYTES
    presigned_urls =

    for i in range(num_parts):
        part_number = i + 1
        url = s3_client.generate_presigned_url(
            'upload_part',
            Params={
                'Bucket': BUCKET_NAME,
                'Key': f"raw-videos/{file_name}",
                'UploadId': upload_id,
                'PartNumber': part_number
            },
            ExpiresIn=3600  # URL 有效期 1 小時
        )
        presigned_urls.append({'partNumber': part_number, 'url': url})

    return {
        'statusCode': 200,
        'headers': {
            "Access-Control-Allow-Origin": "*", # 生產環境中應更嚴格
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type"
        },
        'body': json.dumps({
            'uploadId': upload_id,
            'urls': presigned_urls
        })
    }
```

**第 2 步：後端 - 建立完成上傳的 Lambda 函數**

此函數負責在所有區塊上傳後，將它們組裝成一個完整的檔案。

```Python
# lambda_function.py (用於完成上傳)
import boto3
import json
import os

s3_client = boto3.client('s3')
BUCKET_NAME = os.environ.get('BUCKET_NAME')

def lambda_handler(event, context):
    body = json.loads(event.get('body', '{}'))
    file_name = body.get('fileName')
    upload_id = body.get('uploadId')
    parts = body.get('parts') # 格式:

    if not all([file_name, upload_id, parts]):
        return {'statusCode': 400, 'body': json.dumps('缺少必要參數')}

    # 驗證並排序 parts
    parts.sort(key=lambda x: x['PartNumber'])

    result = s3_client.complete_multipart_upload(
        Bucket=BUCKET_NAME,
        Key=f"raw-videos/{file_name}",
        UploadId=upload_id,
        MultipartUpload={'Parts': parts}
    )

    return {
        'statusCode': 200,
        'headers': {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type"
        },
        'body': json.dumps({'location': result['Location']})
    }
```

**第 3 步：前端 - 使用 JavaScript 處理上傳**

這段概念性程式碼展示了前端如何與後端 API 互動來上傳檔案 。  

```JavaScript
// 前端上傳邏輯 (conceptual)
async function uploadLargeFile(file) {
    const CHUNK_SIZE = 10 * 1024 * 1024; // 10MB
    const fileName = file.name;
    const fileSize = file.size;

    // 1. 呼叫後端 API 以啟動上傳
    const startResponse = await fetch('/start-upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fileName, fileSize })
    });
    const { uploadId, urls } = await startResponse.json();

    // 2. 分割檔案並平行上傳區塊
    const uploadPromises =;
    for (let i = 0; i < urls.length; i++) {
        const start = i * CHUNK_SIZE;
        const end = Math.min(start + CHUNK_SIZE, fileSize);
        const chunk = file.slice(start, end);
        
        const promise = fetch(urls[i].url, {
            method: 'PUT',
            body: chunk
        }).then(response => ({
            ETag: response.headers.get('ETag').replace(/"/g, ''), // S3 ETag 包含引號
            PartNumber: urls[i].partNumber
        }));
        uploadPromises.push(promise);
    }

    const uploadedParts = await Promise.all(uploadPromises);

    // 3. 呼叫後端 API 以完成上傳
    const completeResponse = await fetch('/complete-upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            fileName,
            uploadId,
            parts: uploadedParts
        })
    });

    const result = await completeResponse.json();
    console.log('上傳完成:', result.location);
}
```

### 第 2 節：ML 分析管線

我們使用 AWS Step Functions 來編排整個分析工作流程，確保其可靠性和可觀測性。

#### 2.1. 使用 Step Functions Data Science SDK 定義工作流程

以下程式碼展示如何使用 Python SDK 來定義一個狀態機，該狀態機包含預處理、平行分析和結果彙總等步驟 。  

```Python
from stepfunctions.steps import GlueStartJobRunStep, Chain
from stepfunctions.steps.integration import SagemakerStartBatchTransformStep
from stepfunctions.steps.states import Parallel, Pass
from stepfunctions.workflow import Workflow
import sagemaker

# 假設已設定好 IAM 角色和 S3 路徑
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()
bucket = sagemaker_session.default_bucket()

# 1. 定義 ETL 預處理步驟 (使用 AWS Glue)
etl_step = GlueStartJobRunStep(
    'Video Preprocessing',
    parameters={
        "JobName": "SurgicalVideoPreprocessingJob",
        "Arguments": {
            "--S3_SOURCE.$": "$.S3_SOURCE", # 從 Step Function 輸入中獲取
            "--S3_DEST": f"s3://{bucket}/processed-videos/"
        }
    }
)

# 2. 定義各個 SageMaker 批次轉換任務
# 假設已有 'instrument-detector', 'phase-classifier' 等模型
instrument_transformer = sagemaker.Transformer(...)
phase_transformer = sagemaker.Transformer(...)

instrument_detection_step = SagemakerStartBatchTransformStep(
    'Instrument Detection',
    transformer=instrument_transformer,
    job_name="$.jobNameInstrument",
    data="$.processedVideoPath"
)

phase_classification_step = SagemakerStartBatchTransformStep(
    'Phase Classification',
    transformer=phase_transformer,
    job_name="$.jobNamePhase",
    data="$.processedVideoPath"
)

# 3. 將分析步驟放入 Parallel 狀態中
parallel_analysis = Parallel('Parallel Analysis')
parallel_analysis.add_branch(instrument_detection_step)
parallel_analysis.add_branch(phase_classification_step)

# 4. 定義結果彙總步驟 (使用 Lambda)
aggregation_step = steps.compute.LambdaStep(
    "Aggregate Results",
    parameters={
        "FunctionName": "AggregateAnalysisResultsFunction",
        "Payload": {
            "CaseId.$": "$.CaseId",
            "InstrumentResults.$": "$.ModelOutputPath", # 平行狀態的輸出是一個陣列
            "PhaseResults.$": "$.[1]ModelOutputPath"
        }
    }
)

# 5. 將所有步驟串聯成一個工作流程
workflow_definition = Chain([
    etl_step,
    parallel_analysis,
    aggregation_step
])

# 6. 建立並部署工作流程
workflow = Workflow(
    name="Surgical-Video-Analysis-Workflow",
    definition=workflow_definition,
    role=role
)

workflow.create()
workflow.update(definition=workflow.definition, role=role)
```

### 第 3 節：模型開發與訓練

此階段專注於如何使用 SageMaker 訓練自訂模型，並整合實驗追蹤工具。

#### 3.1. 準備訓練腳本並整合 Weights & Biases (W&B)

我們將在標準的 PyTorch 訓練腳本中加入幾行程式碼來啟用 W&B 追蹤 。  

**`train.py`**

```Python
import argparse
import wandb
import torch
import os

def train(args):
    # 1. 初始化 W&B Run
    # W&B 會自動從環境變數中讀取 API 金鑰
    # SageMaker 會自動記錄超參數到 wandb.config
    wandb.init(project="surgical-video-analysis")
    
    #... (標準的 PyTorch 資料載入和模型定義程式碼)...
    
    model =...
    optimizer =...
    train_loader =...

    for epoch in range(args.epochs):
        #... (訓練迴圈)...
        loss =...

        # 2. 記錄指標
        wandb.log({"epoch": epoch, "loss": loss})

    # 3. 儲存模型
    # SageMaker 要求模型儲存在 SM_MODEL_DIR
    model_path = os.path.join(os.environ.get('SM_MODEL_DIR', '/opt/ml/model'), 'model.pth')
    torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #... (定義如 epochs, lr 等參數)...
    # SageMaker 會自動將 Estimator 中的 hyperparameters 傳遞到這裡
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    train(args)
```

#### 3.2. 使用 SageMaker Python SDK 啟動訓練任務

在 SageMaker Studio 或本地的 Jupyter 筆記本中，我們使用 `PyTorch` Estimator 來設定並啟動訓練任務 。  

```Python
import sagemaker
from sagemaker.pytorch import PyTorch

# 從 W&B 儀表板獲取您的 API 金鑰
WANDB_API_KEY = "YOUR_WANDB_API_KEY" 

sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()
bucket = sagemaker_session.default_bucket()
s3_input_path = f"s3://{bucket}/training-data/"

# 定義 Estimator
pytorch_estimator = PyTorch(
    entry_point='train.py',      # 訓練腳本
    source_dir='./source_code',  # 包含 train.py 和 requirements.txt 的目錄
    role=role,
    instance_count=1,
    instance_type='ml.g4dn.xlarge', # 使用 GPU 執行個體
    framework_version='1.13',
    py_version='py39',
    hyperparameters={
        'epochs': 20,
        'lr': 0.001
    },
    # 將 W&B API 金鑰作為環境變數傳遞給訓練容器
    environment={
        "WANDB_API_KEY": WANDB_API_KEY
    }
)

# 啟動訓練任務
pytorch_estimator.fit({'training': s3_input_path})
```

### 第 4 節：MLOps - 使用 GitHub Actions 進行 CI/CD

我們將建立一個 GitHub Actions 工作流程，在程式碼推送到 `main` 分支時，自動觸發 SageMaker Pipeline 來重新訓練和註冊模型 。  

**`.github/workflows/main.yml`**

```YAML
name: SageMaker MLOps Pipeline

on:
  push:
    branches:
      - main

jobs:
  trigger-sagemaker-pipeline:
    runs-on: ubuntu-latest
    permissions:
      id-token: write # OIDC 驗證所需
      contents: read

    steps:
    - name: Checkout repository
      uses: actions/checkout@v3

    - name: Configure AWS Credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        role-to-assume: arn:aws:iam::ACCOUNT_ID:role/GitHubActionsSageMakerRole # 替換為您的角色 ARN
        aws-region: YOUR_AWS_REGION # 替換為您的區域

    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install sagemaker boto3

    - name: Run SageMaker Pipeline
      run: python pipelines/run_pipeline.py # 執行觸發管線的 Python 腳本
      env:
        SAGEMAKER_PIPELINE_NAME: "SurgicalModelTrainPipeline"
        SAGEMAKER_ROLE_ARN: ${{ secrets.SAGEMAKER_ROLE_ARN }}
```

**`pipelines/run_pipeline.py`**

此腳本定義並執行 SageMaker Pipeline。

```Python
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep, ProcessingStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.processing import ScriptProcessor
from sagemaker.pytorch import PyTorch
#... 其他必要的 imports

# (此處省略了詳細的步驟定義，但概念與 2.1 節類似)
# 1. 定義處理步驟 (ScriptProcessor)
# 2. 定義訓練步驟 (PyTorch Estimator)
# 3. 定義評估步驟 (ScriptProcessor)
# 4. 定義模型註冊步驟 (ModelStep)

# 建立管線實例
pipeline = Pipeline(
    name="SurgicalModelTrainPipeline",
    steps=[processing_step, training_step, evaluation_step, register_step]
)

# 提交並執行管線
pipeline.upsert(role_arn=os.environ)
execution = pipeline.start()
print(f"已啟動管線執行: {execution.arn}")
```

### 第 5 節：使用 OpenAI 產生結構化報告

在分析完成後，我們使用一個 Lambda 函數來呼叫 OpenAI API，將結構化的 JSON 數據轉換為人類可讀的報告 。  

```Python
# lambda_function.py (用於報告生成)
import openai
import json
import os

# 建議從 AWS Secrets Manager 安全地獲取 API 金鑰
openai.api_key = os.environ.get("OPENAI_API_KEY")

def lambda_handler(event, context):
    # 假設 event['analysis_data'] 包含從 DynamoDB 獲取的分析結果
    analysis_data = event.get('analysis_data', {})

    # 定義我們希望 LLM 輸出的 JSON 結構
    report_schema = {
        "type": "object",
        "properties": {
            "procedure_summary": {
                "type": "string",
                "description": "對整個手術過程的簡要總結。"
            },
            "key_events": {
                "type": "array",
                "description": "按時間順序列出的關鍵手術事件列表。",
                "items": {
                    "type": "object",
                    "properties": {
                        "timestamp": {"type": "string", "description": "事件發生的時間戳 (HH:MM:SS)。"},
                        "event_description": {"type": "string", "description": "對事件的描述。"}
                    },
                    "required": ["timestamp", "event_description"]
                }
            },
            "anomalies_observed": {
                "type": "string",
                "description": "觀察到的任何異常或值得注意的事件。"
            }
        },
        "required": ["procedure_summary", "key_events"]
    }

    prompt = f"""
    你是一位專業的外科手術分析師。請根據以下 JSON 格式的手術事件數據，
    生成一份結構化的敘述性手術報告。
    數據: {json.dumps(analysis_data)}
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=,
            # 強制模型輸出符合指定的 JSON 結構
            response_format={"type": "json_schema", "json_schema": {
                "name": "surgical_report",
                "schema": report_schema
            }}
        )
        
        report_content = json.loads(response.choices.message.tool_calls.function.arguments)
        
        #... (將 report_content 儲存到 DynamoDB 或 S3)...

        return {
            'statusCode': 200,
            'body': json.dumps(report_content)
        }

    except Exception as e:
        print(f"呼叫 OpenAI 時發生錯誤: {e}")
        return {'statusCode': 500, 'body': '報告生成失敗'}

```

### 第 6 節：推論優化

訓練完成後，我們可以使用 SageMaker Neo 來編譯模型，以優化其在特定硬體上的推論效能。

```Python
# 在訓練完成後，從 Estimator 獲取模型物件
trained_model = pytorch_estimator.create_model()

# 編譯模型以用於批次轉換
# 目標硬體為 ml.g4dn.xlarge，框架為 PyTorch 1.13
compiled_model = trained_model.compile(
    target_instance_family='ml.g4dn',
    input_shape={'data': }, # 根據您的模型輸入調整
    output_path=f"s3://{bucket}/compiled-models/",
    framework='pytorch',
    framework_version='1.13'
)

# 現在，您可以在批次轉換任務中使用這個 'compiled_model'
# 而不是原始的 'trained_model'，以獲得更快的推論速度。
```

---

這份指南提供了一個從頭到尾的實作藍圖。雖然某些部分（如前端和具體的模型架構）是概念性的，但後端基礎設施、資料管線和 MLOps 流程的程式碼為建構一個生產級系統提供了堅實的基礎。