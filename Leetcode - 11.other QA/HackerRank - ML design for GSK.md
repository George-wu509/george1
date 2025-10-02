

# GSK表型組學AIML工程師機器學習系統設計面試指南

## 導論：破解生物製藥領域的機器學習系統設計面試

機器學習系統設計面試是評估候選人將複雜業務需求轉化為可擴展、可靠且高效的生產級解決方案的關鍵環節 。然而，生物製藥領域（尤其是像GSK這樣的頂級藥廠）的系統設計面試，與FAANG等科技巨頭的傳統面試有著本質上的區別。成功駕馭這類面試不僅需要深厚的技術功底，更要求對科學領域、數據特性及嚴格監管環境有著細緻入微的理解。  

傳統的系統設計面試通常圍繞推薦系統、詐欺偵測或內容審核等主題 ，其核心挑戰在於處理海量用戶流量、低延遲要求和商業指標優化。相比之下，生物製藥領域的挑戰則更為獨特且多維度。一個成功的候選人必須能夠在其設計中無縫融合三大核心支柱：  

1. **可擴展與穩健的工程實踐 (Scalable and Robust Engineering)：** 生物醫學數據，如基因組學、空間轉錄組學和高解析度病理影像，其規模和複雜性常以TB甚至PB級計算 。系統設計必須考慮到高效的數據處理、分佈式計算以及成本效益。  
    
2. **深刻的科學領域理解 (Deep Scientific and Domain Understanding)：** 候選人必須理解數據背後的生物學意義。例如，設計一個系統來校正高通量篩選中的批次效應，不僅僅是一個統計問題，更需要理解實驗流程如何引入系統性偏差 。同樣，在新藥研發中設計一個「實驗室在環」(lab-in-a-loop) 的生成式AI系統，要求對藥物發現的迭代本質有深刻認識 。  
    
3. **嚴格的監管與倫理遵循 (Rigorous Adherence to Regulatory and Ethical Standards)：** 在生命科學領域，特別是涉及臨床數據和藥物開發的後期階段，系統必須滿足嚴格的監管要求，如HIPAA、GxP（優良實踐規範）等 。這意味著數據治理、模型可追溯性、版本控制和可審計性不再是可選項，而是系統架構的基礎性約束 。  
    

本報告旨在提供20個HackerRank風格的機器學習系統設計面試問題，專為GSK表型組學（Phenomics）AIML工程師職位量身定制。每個問題都將模擬一個真實世界中的挑戰，並附有詳盡的中文解說，旨在引導候選人展示其在這三大支柱上的綜合能力。這些問題將不僅僅測試候選人是否知道某個演算法，而是評估他們是否能夠作為一名思想領袖，為加速藥物發現和改善患者生命質量的使命做出貢 F獻。

### 面試問題與考察能力概覽

下表概述了20個面試問題，並將其歸類於四大核心主題，同時標示出每個問題旨在評估的關鍵能力。這個表格可以作為候選人準備面試的戰略地圖，幫助其識別自身優勢與待加強的領域。

| 題號     | 問題標題                                 | 核心主題          | 考察的關鍵能力                                    |
| ------ | ------------------------------------ | ------------- | ------------------------------------------ |
|        |                                      |               |                                            |
|        | [[## 第一部分：基礎MLOps與可擴展的管道在受監管環境中的應用]] |               |                                            |
| **1**  | 為多模態表型數據設計一個合規的特徵商店                  | 基礎MLOps與可擴展性  | MLOps、數據治理、合規性 (HIPAA/GxP)、多模態數據、線上/線下一致性  |
| **2**  | 為TB級空間轉錄組學數據設計可擴展的數據管道               | 基礎MLOps與可擴展性  | 分佈式計算 (Spark/Ray)、工作流編排、大規模數據處理、基因組學       |
| **3**  | 為GxP環境下的模型設計CI/CD4ML系統               | 基礎MLOps與可擴展性  | MLOps、CI/CD、模型版本控制、合規性 (GxP)、可審計性          |
| **4**  | 為百人級數據科學家團隊設計實驗與模型註冊平台               | 基礎MLOps與可擴展性  | MLOps、實驗追蹤、模型註冊中心、協作與再現性                   |
| **5**  | 為藥物發現管道設計數據治理與血緣追蹤系統                 | 基礎MLOps與可擴展性  | 數據治理、數據血緣、可審計性、元數據管理                       |
|        |                                      |               |                                            |
|        | [[## 第二部分：端到端系統在藥物發現與開發中的應用]]        |               |                                            |
| **6**  | 設計一個用於識別新型激酶抑制劑的高通量虛擬篩選系統            | 端到端藥物發現       | 虛擬篩選、分佈式計算、多階段建模、藥物化學                      |
| **7**  | 使用生成式AI設計一個從零開始的藥物設計系統               | 端到端藥物發現       | 生成式AI (VAE/GAN)、強化學習、"Lab-in-a-Loop"、分子表示  |
| **8**  | 設計一個從臨床前數據預測藥物不良反應的ML系統              | 端到端藥物發現       | 多標籤分類、數據整合、不平衡數據處理、模型可解釋性                  |
| **9**  | 利用生物醫學知識圖譜設計一個藥物重定位系統                | 端到端藥物發現       | 知識圖譜、圖神經網絡 (GNN)、藥物重定位、數據整合                |
| **10** | 設計一個優化臨床試驗患者招募的系統                    | 端到端藥物發現       | 自然語言處理 (NLP)、EHR數據分析、合規性 (HIPAA)、人機迴路      |
|        |                                      |               |                                            |
|        | [[## 第三部分：先進表型組學與多模態數據整合]]           |               |                                            |
| **11** | 為腫瘤空間轉錄組學數據設計細胞類型解卷積系統               | 先進表型組學與數據融合   | 空間轉錄組學、單細胞RNA測序、數據整合、計算生物學                 |
| **12** | 設計一個多模態融合模型對腫瘤患者進行分層                 | 先進表型組學與數據融合   | 多模態融合、深度學習、病理影像分析、基因組學、模型可解釋性              |
| **13** | 為高通量篩選數據設計批次效應校正系統                   | 先進表型組學與數據融合   | 統計建模、批次效應、混淆變量、數據預處理、HTS                   |
| **14** | 為高內涵成像設計表型分析的ML系統                    | 先進表型組學與數據融合   | 計算機視覺、特徵提取、自監督學習、非監督學習、表型分析                |
| **15** | 利用大型語言模型融合非結構化與結構化數據進行靶點識別           | 先進表型組學與數據融合   | 大型語言模型 (LLM)、多模態融合、知識圖譜、靶點識別               |
|        |                                      |               |                                            |
|        | [[#### 第四部份]]                        |               |                                            |
| **16** | 為已部署的臨床診斷AI設計模型監控系統                  | 生產環境ML與生命週期管理 | 模型監控、概念漂移、數據漂移、公平性與偏見、醫療AI                 |
| **17** | 為生產環境中的ML模型設計回滾策略                    | 生產環境ML與生命週期管理 | MLOps、部署策略 (Canary/Blue-Green)、風險管理、模型版本控制 |
| **18** | 為藥物靶點推薦系統設計A/B測試框架                   | 生產環境ML與生命週期管理 | A/B測試、實驗設計、科學家工作流分析、長期指標                   |
| **19** | 為蛋白質結構預測模型設計低延遲的實時推理系統               | 生產環境ML與生命週期管理 | 實時推理、模型優化 (量化/剪枝)、GPU服務、緩存策略               |
| **20** | 為複雜生物醫學數據標註設計人機迴路系統                  | 生產環境ML與生命週期管理 | 主動學習、數據標註、人機交互、成本效益分析                      |


## 第一部分：基礎MLOps與可擴展的管道在受監管環境中的應用

本部分的問題旨在考察候選人構建製藥公司機器學習基礎設施的核心能力。這些系統是所有上層應用的基石，重點在於可靠性、可再現性、可擴展性和合規性。在生物製藥領域，監管合規性並非事後添加的功能，而是從一開始就決定每一項設計選擇的基礎架構約束 。一個優秀的系統設計必須將可審計性置於與性能同等重要的位置，這意味著模型註冊中心不僅要對模型本身進行版本控制，還必須追蹤數據、代碼、環境和評估結果的完整血緣 。  

### 問題 1：為多模態表型數據設計一個合規的特徵商店
**Question 1: Design a Compliant Feature Store for Multimodal Phenotypic Data**

- **Problem Statement:** GSK's Phenomics team requires a centralized Feature Store to support various ML applications, including real-time inference for clinical decision support tools and batch processing for cohort analysis in drug target discovery. Data sources are diverse, including genomic data (VCFs), electronic health records (EHR), and morphological feature vectors from whole-slide images (WSI).
- **System Requirements:**
    - **Compliance:** Must adhere to HIPAA regulations and be "GxP-ready" for future clinical applications.
    - **Online/Offline Consistency:** Ensure identical feature calculation logic for model training (offline) and inference (online).
    - **Scalability & Performance:** Handle terabytes of feature data, with low latency for online serving (p99 < 50ms) and high throughput for batch serving.
    - **Discoverability & Versioning:** Features and their generation logic must be easily discoverable and version-controlled.
- **Task:** Detail the high-level architecture, key components, data flow, and technology choices, focusing on how you would address the challenges of compliance, online/offline consistency, and data governance.

**問題陳述：** GSK的表型組學團隊需要一個集中式的特徵商店（Feature Store），以支持多種機器學習應用場景。該系統需要服務於兩類核心用例：1) **實時推理**，例如為臨床決策支持工具提供即時特徵；2) **批次處理**，例如為藥物靶點發現的隊列分析（cohort analysis）提供大規模特徵集。

數據源極其多樣，包括：
- **基因組數據：** 變異調用文件 (VCFs) 衍生的基因標記。
- **電子健康記錄 (EHR)：** 結構化的實驗室檢測結果和診斷碼。
- **病理影像特徵：** 從全切片影像 (WSI) 中通過深度學習模型提取的形態學特徵向量。

您需要設計一個端到端的特徵商店系統。您的設計必須滿足以下要求：
- **合規性：** 系統必須嚴格遵守HIPAA法規，保護患者隱私。此外，其設計應具備「GxP-ready」的潛力，以便未來能夠支持需要遵循GxP規範的臨床應用。
- **線上/線下一致性：** 確保在模型訓練（線下）和模型推理（線上）時使用的特徵計算邏輯完全一致，避免因此導致的性能下降。
- **可擴展性與性能：** 能夠高效地處理和存儲TB級的特徵數據。線上服務必須滿足低延遲要求（p99 < 50ms），批次服務則需要高吞吐量。
- **可發現性與版本控制：** 數據科學家應能輕鬆地發現、理解和使用特徵。所有特徵及其生成邏輯都必須進行版本控制。

請闡述您的高層架構設計，包括關鍵組件、數據流、技術選型，並重點討論您將如何解決合規性、線上/線下一致性和數據治理的挑戰。

白板圖 #1：系統高層架構 (High-Level Architecture)
這張圖展示了整個特徵儲存系統的宏觀設計，以及合規與治理層如何覆蓋所有組件。
![[Pasted image 20250928032141.png]]
這張高層架構圖展示了特徵儲存系統的核心組件，所有這些組件都被一個無處不在的 **合規與治理層 (Compliance & Governance Layer)** 所包裹，以確保安全性與可追溯性。

1. **原始數據源 (Raw Data Sources)**：系統從多種來源攝取數據，包括結構化的電子健康記錄 (EHR)、基因組數據 (VCFs) 和從全切片圖像 (WSI) 中提取的形態學特徵向量。
2. **轉換與特徵工程 (Transformation & Feature Engineering)**：這是一個計算層，使用 Apache Spark 等工具，將原始數據轉換為機器學習模型可以使用的特徵。
3. **特徵註冊表與元數據儲存 (Feature Registry & Metadata Store)**：這是系統的“大腦”或“目錄”。它存儲了每個特徵的定義、版本、所有者、生成邏輯以及數據血緣 (Data Lineage)，確保了特徵的可發現性和可管理性。
4. **離線與線上儲存 (Offline & Online Store)**：
    - **離線儲存**：用於存儲大量的歷史特徵數據，專為模型訓練所需的高吞吐量批次讀取而優化。
    - **線上儲存**：用於存儲最新的特徵值，專為需要低延遲（p99 < 50ms）的實時推理服務而優化。
5. **數據消費者 (Data Consumers)**：
    - **機器學習訓練管線** 從離線儲存中讀取大量數據來訓練模型。
    - **實時推理 API** 從線上儲存中快速讀取單個實體（如患者）的特徵，以支持臨床決策等即時應用。

白板圖 #2：數據流與線上/離線一致性 (Data Flow & Online/Offline Consistency)
此圖詳細說明了數據如何流動，以及系統如何保證用於訓練和推理的特徵計算邏輯完全一致。
![[Pasted image 20250928032223.png]]
線上/離線一致性是特徵儲存系統的核心挑戰。此設計通過確保 **單一事實來源 (Single Source of Truth)** 來解決這個問題。

1. **統一的特徵生成代碼**：所有特徵的計算邏輯（無論是 Python、SQL 還是 PySpark）都存儲在一個版本控制系統（如 Git）中。這是計算特徵的唯一權威代碼庫。
2. **編排引擎觸發**：像 Airflow 這樣的編排工具會調度不同的作業來計算和“物化”（即存儲）特徵。
3. **並行的數據流**：
    - **批次管線 (Batch Pipeline)**：定期（例如每天）運行，讀取大量的原始數據，應用特徵生成代碼，並將結果寫入**離線儲存**，供模型訓練使用。
    - **流處理管線 (Streaming Pipeline)**：實時監聽來自消息隊列（如 Kafka）的新數據事件（例如，EHR 中新增了一條記錄）。它會立即應用**完全相同的特徵生成代碼**，並將結果寫入**線上儲存**，供即時推理使用。

因為兩條路徑共享完全相同的、經過版本控制的代碼，所以可以從根本上保證在任何時間點，為任何實體計算出的特徵值都是一致的。

白板圖 #3：合規與治理框架 (Compliance & Governance Framework)
此圖詳細闡述了實現 HIPAA 和 GxP-ready 合規性的關鍵支柱。
![[Pasted image 20250928032333.png]]
合規性不是單一的功能，而是一個貫穿整個系統的框架，主要由以下四個支柱構成：

1. **身份與訪問管理 (IAM)**：
    - 實施**基於角色的訪問控制 (RBAC)**，確保用戶只能訪問其工作所需的數據（最小權限原則）。
    - 支持**細粒度權限**，可以控制到表、列甚至行的級別，這對於保護 HIPAA 定義的受保護健康信息 (PHI) 至關重要。
2. **審計追蹤與日誌記錄 (Audit Trail & Logging)**：
    - 系統中所有的操作——每一次特徵的創建、讀取、更新——都必須被記錄在一個**不可變的日誌**中。
    - 日誌必須清楚地記錄“誰 (Who)、做了什麼 (What)、什麼時候 (When)”，以便進行安全審計和事件調查。
3. **數據血緣與可追溯性 (Data Lineage & Traceability)**：
    - 必須能夠清晰地追蹤任何一個特徵的完整生命週期：從它來源的原始數據，經過哪個版本的轉換代碼，最終被哪個版本的模型所使用。
    - 這是除錯、保證可重複性以及滿足監管機構（如 FDA）對 GxP 要求的基礎。
4. **GxP 驗證與環境控制 (GxP Validation & Environment Control)**：
    - 嚴格區分**開發、測試 (QA) 和生產環境**，確保在生產環境中運行的代碼經過了充分的驗證。
    - 實施嚴格的**變更控制流程**，所有代碼變更都必須經過審查和批准。
    - 自動化測試和驗證報告是證明系統“按預期工作”的關鍵文檔，是 GxP 合規性的核心。

白板圖 #4：技術棧選擇 (Technology Stack Choices)
此圖列出了構建該系統的建議技術棧，並解釋了選擇每個技術的原因。
![[Pasted image 20250928032425.png]]
技術選擇需優先考慮安全性、可擴展性和性能。

- **特徵註冊表**：**Databricks Unity Catalog** 是一個強有力的選擇，因為它原生集成了數據治理、血緣和訪問控制功能。
- **轉換層**：**Apache Spark** 是處理 TB 級別多樣化數據（從 Parquet 到 VCF）的行業標準。
- **離線儲存**：**Snowflake** 尤其適合這個場景，因為它提供了強大的、內置的安全與治理功能（如列級安全、數據遮罩），極大地簡化了 HIPAA 合規工作。
- **線上儲存**：**Redis** 或 **DynamoDB** 都是成熟的解決方案，能夠穩定地滿足嚴格的低延遲服務要求。
- **編排**：**Airflow** 是一個成熟的工具，可以可靠地調度、監控和記錄所有數據管線的運行，這對審計至關重要。
- **合規/安全**：利用雲服務商（如 AWS, Azure）提供的原生安全工具（如 **AWS IAM** 用於權限管理，**KMS** 用於加密，**CloudTrail** 用於審計），可以為整個系統打下堅實的安全基礎。


**假設：**

- 系統需要支持100萬級別的患者數據。
- EHR數據每日批次更新，影像特徵按需生成，基因組數據相對靜態。
- 需要支持審計追蹤，記錄特徵的創建、修改和訪問歷史。
- 線上服務的QPS預計為500。

**2. 系統架構設計 (System Architecture Design)** 候選人應繪製一個高層架構圖，並解釋各個組件的功能。一個成熟的設計通常包含以下幾個層面 ：  

- **數據源層 (Data Sources)：** 原始數據存儲，如數據湖 (S3/GCS)、數據庫 (PostgreSQL)、影像檔案 (DICOM)。
    
- **數據處理/轉換層 (Processing/Transformation Layer)：**
    
    - 使用分佈式計算框架（如Apache Spark）進行批次特徵計算。Spark的結構化和SQL API非常適合處理EHR和基因組數據。
    - 對於影像特徵，可能有一個獨立的GPU集群，運行PyTorch/TensorFlow模型進行特徵提取。
    - 所有特徵計算邏輯都應被封裝成可重用的模塊，並進行版本控制（例如，存儲在Git中）。
        
- **存儲層 (Storage Layer)：** 這是特徵商店的核心。
    
    - **離線存儲 (Offline Store)：** 用於存儲大規模歷史特徵數據，供模型訓練和探索性分析使用。通常選擇基於列式存儲的格式（如Parquet、Delta Lake）存放在數據湖中，以實現高效的掃描和查詢。
    - **線上存儲 (Online Store)：** 用於存儲服務實時推理所需的最新特徵值。要求低延遲讀取。常見選擇包括Redis、DynamoDB或Cassandra。
        
- **服務層 (Serving Layer)：**
    
    - 提供一個高性能的API（例如，基於gRPC或REST），供線上模型調用以獲取特徵向量。
    - 提供一個SDK（例如，Python SDK），供數據科學家在模型訓練時方便地從離線存儲中拉取特定時間點的數據集（point-in-time correct joins），以避免數據洩漏。
        
- **註冊/元數據層 (Registry/Metadata Layer)：**
    
    - 這是實現數據治理和可發現性的關鍵。它存儲關於每個特徵的元數據，如：特徵名稱、版本、描述、所有者、數據類型、統計信息（分佈、缺失率等）、生成代碼的鏈接以及數據血緣。
    - 提供一個UI界面，讓用戶可以搜索和瀏覽可用的特徵。

**3. 關鍵挑戰與解決方案**

- **合規性與數據治理 (Compliance & Governance)：**
    
    - **訪問控制：** 實施基於角色的訪問控制（RBAC），確保只有授權的用戶和服務才能訪問特定的特徵數據。所有敏感的個人健康信息（PHI）在存儲和傳輸過程中都必須加密 。  
    - **數據脫敏：** 在特徵計算過程中，應去除所有直接的個人標識符（PII/PHI），使用假名化的患者ID。
    - **審計追蹤：** 這是「GxP-ready」的核心。系統必須記錄所有對元數據和特徵數據的CRUD（創建、讀取、更新、刪除）操作。日誌應包含操作者、時間戳、操作內容等信息，並存儲在不可變的日誌系統中。
        
- **線上/線下一致性 (Online/Offline Parity)：**
    
    - **統一計算邏輯：** 這是最根本的解決方案。特徵計算的代碼庫必須是統一的。同一個Python函數或Spark UDF應該既能用於生成離線的Parquet文件，也能被部署到流處理引擎（如Spark Streaming或Flink）或實時服務中，用於線上特徵計算 。  
    - **版本化轉換：** 所有的特徵轉換邏輯都必須與特徵本身一起進行版本控制。當模型請求特徵時，它不僅指定了特徵名稱，還指定了版本，確保訓練和推理使用完全相同的邏輯。
        
- **數據血緣 (Data Lineage)：**
    - 元數據層應自動捕獲數據血緣關係：哪個原始數據表、通過哪個版本的轉換代碼、生成了哪個版本的特徵。這對於調試問題、理解模型依賴以及滿足GxP的可追溯性要求至關重要。可以使用工具如dbt或內部開發的框架來管理和可視化這些依賴關係。

**4. 權衡分析 (Trade-off Analysis)** 一個資深候選人會討論設計中的權衡 。  

- **一致性 vs. 性能：** 線上存儲選擇強一致性的數據庫可能會增加延遲，而選擇最終一致性的數據庫（如Cassandra）性能更好，但可能存在短暫的數據不一致。需要根據業務場景（例如，臨床決策支持的容忍度）來決定。
- **構建 vs. 購買：** 是自研整個特徵商店，還是基於開源工具（如Feast）或商業解決方案（如Tecton、Vertex AI Feature Store）進行構建？購買可以加快開發速度，但可能在與內部系統（特別是合規系統）集成時缺乏靈活性。自研則可以完全定製，但需要大量的工程投入。
- **實時計算 vs. 預計算：** 對於某些特徵，是實時計算還是提前計算並存儲在線上存儲中？預計算可以降低線上服務的延遲和計算負擔，但會增加存儲成本和數據新鮮度的延遲。

通過這樣一個全面的回答，候選人不僅展示了其設計複雜數據平台的能力，還體現了對生物製藥領域特有的合規性和嚴謹性要求的深刻理解。



### 問題 2：為TB級空間轉錄組學數據設計可擴展的數據管道

**Question 2: Design a Scalable Data Pipeline for Terabyte-Scale Spatial Transcriptomics Data**
- **Problem Statement:** GSK is adopting spatial transcriptomics (ST), generating terabytes of data per experiment, including high-resolution histology images (WSI) and large gene expression matrices. Your task is to design an automated, scalable, end-to-end data processing pipeline that transforms raw data into an "Analysis-Ready Data" (ARD) object for downstream ML analysis.
- **System Requirements:**
    - **Scalability:** Must scale horizontally to process dozens of new experiments (tens of TBs) weekly.
    - **Standardization & Reproducibility:** Every step must be standardized and the entire workflow fully reproducible.
    - **Core Processing Steps:** Include image tiling, expression data QC, spatial alignment of image and expression data, and downstream feature engineering (e.g., identifying spatially variable genes, preliminary cell-type deconvolution).
    - **Cost-Effectiveness:** The design should consider and optimize for compute and storage costs.
- **Task:** Describe the high-level architecture, workflow orchestration, key technology choices, and how you would tackle the challenges of processing such large-scale and complex scientific data.

**問題陳述：** GSK正在大規模採用空間轉錄組學（Spatial Transcriptomics, ST）技術來研究疾病組織的微環境。每個ST實驗都會產生TB級別的數據，主要包括兩部分：

1. **高解析度組織學影像：** 通常是WSI（Whole Slide Image）格式，單個文件可達數十GB。
    
2. **基因表達矩陣：** 一個巨大的稀疏矩陣，記錄了數萬個基因在數千到數十萬個空間位置（spots）的表達計數。
    

您的任務是設計一個自動化、可擴展的端到端數據處理管道。該管道需要從原始數據（影像和表達矩陣）開始，最終生成一個可用於下游機器學習分析的「分析就緒數據」（Analysis-Ready Data, ARD）對象。

**系統要求：**

- **可擴展性：** 管道必須能夠橫向擴展，以處理每週數十個新實驗產生的數據量，總計數十TB。
    
- **標準化與可再現性：** 管道的每一步都必須是標準化的，以確保不同實驗結果之間的可比性。整個流程必須是完全可再現的。
    
- **核心處理步驟：** 管道至少應包括以下步驟：
    
    1. **影像處理：** 將WSI影像切片（tiling），並對每個tile進行質量控制和標準化。
        
    2. **表達數據質控：** 過濾低質量的spots和基因。
        
    3. **數據整合：** 將影像數據與基因表達數據在空間上對齊。
        
    4. **下游特徵工程：** 計算標準化的下游特徵，如空間可變基因（Spatially Variable Genes）的識別、細胞類型解卷積（cell-type deconvolution）的初步結果等。
        
- **成本效益：** 設計應考慮到計算和存儲的成本，並提出優化策略。
    

請描述您的高層架構、工作流編排、關鍵技術選型，並解釋您將如何應對處理如此大規模和複雜的科學數據所帶來的挑戰。

白板圖 #1：系統高層架-構 (High-Level Architecture)
此圖展示了整個數據處理管線的雲端原生、自動化架構，從數據上傳到最終生成「分析就緒數據」(ARD)。
![[Pasted image 20250928032804.png]]
此架構設計為一個事件驅動、可水平擴展的雲端原生系統。

1. **數據攝取與儲存 (Data Ingestion & Storage)**：當測序儀或圖像掃描儀產生新數據時，它們會被上傳到雲端儲存（如 AWS S3）的 **"Landing Zone"**。這個區域的數據是不可變的原始數據。整個管線的最終產物——分析就緒數據 (ARD)，將被儲存在 **"Curated Zone"**。
2. **工作流編排引擎 (Workflow Orchestration Engine)**：**Nextflow** 是這個系統的大腦。當新數據到達時，它會被觸發，並開始執行一個預先定義好的、由多個步驟組成的有向無環圖 (DAG)。Nextflow 負責管理任務依賴、處理錯誤重試，並確保整個流程的**可重複性**。
3. **可擴展計算層 (Scalable Compute Layer)**：Nextflow 將每個處理步驟作為一個獨立的作業提交給計算層（如 **AWS Batch**）。AWS Batch 會根據需要動態地啟動計算資源（EC2 實例）。每個作業都在一個 **Docker 容器**中運行，這保證了環境和軟體版本的一致性。為了**成本效益**，該層會優先使用 **Spot 實例**，可節省高達 70-90% 的計算成本。
4. **元數據與監控 (Metadata & Monitoring)**：管線的運行狀態、日誌和性能指標被發送到監控系統（如 CloudWatch）。每個樣本的處理結果和元數據被記錄在一個集中的資料庫中，以便於查詢和追蹤。

白板圖 #2：詳細工作流與核心處理步驟 (Detailed Workflow & Core Processing Steps)
此圖展示了由 Nextflow 編排的具體生物資訊學分析流程的 DAG。
![[Pasted image 20250928032856.png]]
這個流程圖詳細描述了從原始數據到 ARD 的每一步，每個方框代表一個可以被容器化的獨立任務。

1. **並行處理**：測序數據（左側）和影像數據（右側）的初始處理是並行進行的，以最大化效率。
    
    - **測序流**：首先將原始的 BCL 文件轉換為 FASTQ 格式，然後使用標準工具（如 10x Genomics 的 Space Ranger）進行序列比對和基因表達定量，最終生成帶有空間座標的基因表達矩陣。
        
    - **影像流**：對 TB 級的 WSI 進行預處理（如顏色校正），然後將其高效地切割成數百萬個小圖塊 (tiles)。
        
2. **空間對齊與質控 (Spatial Alignment & QC)**：這是關鍵的整合步驟。系統將基因表達點的空間座標與 WSI 上的實際物理位置對齊。在此階段，也會進行嚴格的質量控制，過濾掉低質量的數據點。
    
3. **下游特徵工程 (Downstream Feature Engineering)**：在標準化的數據上，管線會自動執行一些初步的分析，為機器學習做好準備：
    
    - **空間變異基因 (SVG) 計算**：識別那些其表達模式與空間位置顯著相關的基因。
        
    - **細胞類型解卷積 (Cell-Type Deconvolution)**：初步估計每個測量點中不同細胞類型（如癌細胞、免疫細胞）的比例。
        
4. **ARD 組裝**：最後，所有處理過的數據——包括影像圖塊、標準化的表達矩陣、QC 指標、SVG 列表和細胞類型比例——都被打包成一個標準化的、自包含的 **分析就緒數據 (ARD) 對象**。


白板圖 #3：「分析就緒數據」(ARD) 對象結構
此圖定義了管線最終產物 ARD 的內部結構，採用業界領先的 `SpatialData` 格式。
![[Pasted image 20250928032933.png]]
ARD 的目標是提供一個標準化、自包含、且易於訪問的數據容器。我們選擇 `SpatialData` 格式（基於 Zarr 儲存），因為它具備以下優勢：

1. **多模態原生**：它被設計用來同時存儲多種空間數據類型，包括光柵圖像 (images)、標籤圖像 (labels, 如組織分割)、點數據 (points, 如基因表達點) 和核心的表格數據 (table)。
    
2. **雲端友好 (Cloud-Friendly)**：**Zarr** 是一種分塊 (chunked) 陣列儲存格式。這意味著分析師可以只讀取他們需要的部分數據（例如，一個基因的表達或一小塊圖像區域），而無需下載整個 TB 級的對象。這極大地降低了數據傳輸成本和分析的啟動時間。
    
3. **標準化**：`AnnData` 對象（存儲在 `table` 中）是單細胞和空間組學分析的行業標準，擁有豐富的生態系統支持。它將表達矩陣 (`.X`)、細胞/點元數據 (`.obs`) 和基因元數據 (`.var`) 有機地組織在一起。

白板圖 #4：技術棧與成本優化策略 (Technology Stack & Cost Optimization)
此圖總結了構建該系統的技術選擇，並重點說明了如何實現可擴展性和成本效益
![[Pasted image 20250928033008.png]]
這個技術棧的選擇旨在實現最佳的性能、可擴展性和成本效益。

- **Nextflow + Docker + AWS Batch**：這個組合是現代生物資訊學可擴展計算的黃金標準。Nextflow 提供強大的工作流管理，Docker 保證可重複性，而 AWS Batch 則提供了按需、經濟的計算能力。
    
- **成本優化策略**：
    
    - **計算成本**：主要通過在 AWS Batch 中大量使用 **Spot 實例**來實現。由於生物資訊學工作流通常可以容忍中斷和重試，因此非常適合使用 Spot 實例。
        
    - **儲存成本**：
        
        1. 採用 **S3 智能分層 (Intelligent-Tiering)**，自動將不常訪問的原始數據或中間文件移動到成本更低的儲存層。
            
        2. 使用 **Zarr** 作為最終輸出格式。其高壓縮率和分塊讀取能力，不僅節省了存儲空間，更重要的是**大幅降低了數據傳輸 (egress) 成本**，因為分析人員不再需要為了分析而下載整個龐大的數據集。


---

**中文解說：**

這個問題的核心在於處理異構（影像+表格）和大規模的科學數據。一個強有力的回答會展示候選人在分佈式計算、工作流管理和雲架構方面的深厚功底，並結合對空間轉錄組學數據特性的理解 。  

**1. 問題解析與範疇界定 (Problem Clarification & Scoping)**

- **數據格式細節：** WSI的具體格式是什麼（例如，SVS, TIFF）？基因表達數據的格式（例如，HDF5, AnnData）？
    
- **處理邏輯的複雜性：** 細胞類型解卷積等步驟需要哪些參考數據（例如，單細胞RNA測序數據集）？這些算法的計算複雜度如何？
    
- **性能目標：** 從原始數據到ARD的端到端處理時間（SLA）要求是什麼？例如，一個實驗必須在24小時內處理完畢。
    
- **輸出格式：** 最終的ARD期望是什麼格式？是一個統一的數據對象（如AnnData/Seurat對象）還是存儲在數據庫/數據湖中的多個表？
    

**假設：**

- WSI為SVS格式，表達數據為10x Genomics的HDF5格式。
    
- SLA為48小時內完成一個實驗的處理。
    
- 最終輸出為存儲在S3上的Parquet文件集合和一個包含元數據的AnnData文件。
    

**2. 系統架構與工作流設計 (System Architecture & Workflow Design)** 候選人應提出一個基於雲的、由工作流引擎驅動的架構。

- **工作流編排 (Workflow Orchestration)：**
    
    - 選擇一個強大的工作流編排工具是關鍵。**AWS Step Functions** 或 **Argo Workflows** (on Kubernetes) 是很好的選擇，因為它們天然支持並行執行、錯誤處理和重試邏輯。**Nextflow** 或 **Snakemake** 也是生物資訊領域的常用工具，但可能在與通用雲服務集成方面稍遜一籌。
        
    - 整個管道應被設計成一個有向無環圖（DAG），其中每個節點代表一個處理步驟。例如，影像處理和表達數據質控可以並行執行。
        
- **數據存儲 (Data Storage)：**
    
    - **原始數據：** 存儲在成本較低的對象存儲中，如 **Amazon S3 Glacier** 或 **Standard-IA**。
        
    - **中間數據與最終ARD：** 存儲在 **Amazon S3 Standard** 中。對於表格數據（如特徵矩陣），使用 **Parquet** 格式以支持高效的列式讀取。對於影像tiles，直接存儲為JPEG/PNG。使用像 **Zarr** 這樣的格式也是一個很好的選擇，它對大規模多維數組（如影像數據）有很好的支持。
        
- **計算引擎 (Compute Engines)：**
    
    - **影像處理：** 這是計算密集型任務，且高度並行。可以使用 **AWS Batch** 或 **Kubernetes Jobs**，動態地啟動帶有GPU的計算實例（例如，`g4dn` 實例）來處理影像切片。每個tile的處理可以是一個獨立的任務。
        
    - **基因表達數據處理與特徵工程：** 這些任務通常涉及大規模矩陣操作。**Apache Spark** on **Amazon EMR** 或 **Ray** on EC2 是理想的選擇 。它們提供了強大的分佈式數據處理能力。例如，可以使用Spark來讀取巨大的表達矩陣，並行地進行質控和標準化。  
        
    - **服務集成：** 使用 **Amazon SQS** 或 **SNS** 來觸發工作流（例如，當新的原始數據上傳到S3時），並在管道的不同階段之間傳遞消息。
        

**3. 核心處理步驟的實現細節**

- **影像處理：**
    
    - 使用像 **OpenSlide** 這樣的庫來讀取WSI格式。
        
    - 將WSI切片成例如512x512像素的tiles。這個過程本身可以被高度並行化，每個worker處理WSI的一個區域。
        
    - 利用 **SageMaker Processing Jobs** 或 **AWS Batch**，可以輕鬆地將輸入的WSI數據分片到多個計算實例上 。  
        
- **數據整合：**
    
    - 這一步需要精確的空間坐標映射。通常，ST平台會提供一個坐標文件，將表達數據的spot ID映射到WSI影像上的像素坐標。
        
    - 在分佈式環境中，這一步可以通過一個 `join` 操作實現，將影像tile的元數據（包含其在原圖中的坐標範圍）與表達數據的spot元數據（包含其中心坐標）進行連接。
        
- **可再現性：**
    
    - **容器化：** 所有的處理步驟都應在 **Docker** 容器中執行。Dockerfile應詳細定義環境、依賴庫及其版本，確保任何時候執行的環境都完全一致。
        
    - **代碼與配置版本控制：** 所有的分析代碼、腳本和配置文件都必須存儲在 **Git** 中。工作流的每次運行都應記錄下所使用的Git commit hash。
        
    - **數據版本控制：** 可以使用 **DVC (Data Version Control)** 或 **Delta Lake** 的時間旅行功能來對輸入數據和生成的ARD進行版本控制。
        

**4. 成本優化策略**

- **使用Spot實例：** 對於可容忍中斷的批處理任務（如大部分影像處理和特徵工程），大量使用 **EC2 Spot實例** 可以節省高達90%的計算成本。工作流引擎需要配置好重試邏輯來應對Spot實例被回收的情況。
    
- **選擇合適的實例類型：** 根據任務的特性（CPU密集型、內存密集型或GPU密集型）選擇最優的EC2實例類型。
    
- **數據生命週期管理：** 在S3中設置生命週期策略，自動將不常用的中間數據或舊版本的ARD轉移到成本更低的存儲層。
    
- **無服務器選項：** 對於某些輕量級的協調或預處理任務，可以考慮使用 **AWS Lambda**，以避免為空閒的計算資源付費。
    

這個回答展示了候選人設計一個複雜、大規模、多階段科學計算管道的能力，並體現了其在雲原生架構、分佈式系統和成本管理方面的實踐經驗。



### 問題 3：為GxP環境下的模型設計CI/CD4ML系統

**Question 3: Design a CI/CD for ML (CI/CD4ML) System for a GxP Environment**
- **Problem Statement:** At GSK, ML models used in late-stage clinical trials or as Software as a Medical Device (SaMD) must be developed, validated, and deployed under a strict GxP regulatory framework. This places extreme demands on traditional CI/CD processes.
    
- **System Requirements:**
    
    - **Immutability & Version Control:** All assets (data, code, models, container environments) must be strictly version-controlled and immutable.
        
    - **Environment Isolation:** Strict separation of Development (DEV), Test/Validation (TEST/VAL), and Production (PROD) environments, with formal, documented approval for promotions between them.
        
    - **Automated Testing & Validation:** The pipeline must include automated tests for data validation, model performance, and fairness/bias, in addition to standard unit and integration tests.
        
    - **Audit Trail:** Automatically generate detailed audit logs for every pipeline run, including triggers, asset versions, test results, and approval records, to meet regulatory scrutiny.
        
    - **Human-in-the-Loop:** Incorporate mandatory human approval steps (e.g., electronic signatures from a QA team) at critical gates, such as deployment to production.
        
- **Task:** Detail the system architecture and describe the complete workflow from a code commit to production deployment, explaining how your toolchain would meet these stringent GxP requirements.


**問題陳述：** 在GSK，用於支持後期臨床試驗或作為「軟體即醫療設備」(SaMD) 一部分的機器學習模型，必須在嚴格的GxP（優良實踐規範，如GLP, GCP, GMP）監管框架下進行開發、驗證和部署。這對傳統的CI/CD流程提出了極高的要求。

您的任務是設計一個用於機器學習的持續集成/持續交付（CI/CD4ML）系統，該系統專為GxP環境而設計。系統需要自動化模型的訓練、驗證和部署流程，同時確保整個生命週期的完全可追溯性、可審計性和嚴格的質量控制。

**系統要求：**

- **不可變性與版本控制：** 系統中使用的所有資產——包括數據、代碼、模型、容器環境——都必須被嚴格版本控制，且歷史版本不可篡改。
    
- **環境隔離：** 必須嚴格劃分和隔離開發（DEV）、測試/驗證（TEST/VAL）和生產（PROD）環境。任何資產從一個環境到另一個環境的提升（promotion）都必須經過正式的、有記錄的審批流程。
    
- **自動化測試與驗證：** 管道必須包含自動化的測試套件，不僅包括傳統的單元測試和集成測試，還應包括數據驗證、模型性能驗證和公平性/偏見測試。
    
- **審計追蹤：** 系統必須自動生成詳細的審計日誌，記錄每一次管道運行的所有細節：觸發者、時間、使用的資產版本、測試結果、審批記錄等。這些日誌必須易於查詢，以應對監管機構的審查。
    
- **人機協同：** 儘管流程高度自動化，但在關鍵節點（如從TEST到PROD的部署）必須引入人為的審批環節（例如，質量保證QA團隊的電子簽名）。
    

請闡述您的系統架構，描述一個模型從代碼提交到生產部署的完整流程，並詳細說明您將如何利用工具鏈來滿足上述嚴格的GxP要求。

白板圖 #1：系統高層架構 (High-Level Architecture)
此圖展示了系統的整體佈局，強調了環境隔離和一個全面的審計與合規層。
![[Pasted image 20250928033211.png]]
這個架構的核心是**嚴格的環境隔離**和一個貫穿始終的**審計與合純層 (Audit & Compliance Layer)**。

1. **資產註冊庫 (Asset Registries)**：所有資產——**代碼 (Git)**、**數據 (DVC)** 和 **模型 (MLflow)**——都必須被嚴格地進行版本控制。這是可追溯性的基礎。
    
2. **CI/CD 編排器 (CI/CD Orchestrator)**：這是系統的“交通警察”，負責觸發和管理從開發到生產的所有管線。在 GxP 環境中，它的一個關鍵職責是**執行審批關卡 (Approval Gates)**。
    
3. **環境隔離 (Environment Isolation)**：
    
    - **開發 (DEV)**：供數據科學家進行實驗、模型訓練和初步測試的地方。此環境只能訪問經過匿名化處理的樣本數據。
        
    - **測試/驗證 (TEST/VAL)**：一個嚴格受控的環境，用於對模型進行正式的、可記錄的驗證。此處的模型包 (package) 應與最終部署的包完全相同。
        
    - **生產 (PROD)**：用於處理真實世界（例如患者）數據的最終部署環境。對該環境的任何變更都需要最高級別的審批。
        
4. **產物儲存庫 (Artifact Repository)**：用於存儲經過驗證的、**不可變的“GxP 包”**。這些包是部署的原子單位，包含了模型及其所有相關的元數據和文檔。
    
5. **審計與合規層**：這是一個邏輯層，確保所有操作都有日誌記錄、所有訪問都受到控制 (IAM)，並且所有關鍵決策都有電子簽名記錄。

白板圖 #2：詳細的 GxP CI/CD4ML 工作流 (Detailed GxP CI/CD4ML Workflow)
此圖詳細描述了從一次代碼提交到最終生產部署的完整、端到端的流程，重點突出了審批關卡。
![[Pasted image 20250928033252.png]]
這個工作流的核心理念是**“門控式晉升 (Gated Promotion)”**，而非傳統的“持續部署”。

1. **開發階段 (DEV)**：開發人員提交代碼後，CI/CD 系統會自動執行單元測試、代碼掃描，然後在 DEV 環境中使用樣本數據進行模型訓練和初步評估。此階段的產物是一個標記為 `-dev` 的“GxP 包”。
    
2. **審批關卡 #1 (手動)**：要將模型從 DEV 推向 TEST/VAL 環境，必須經過一個正式的審批流程，例如由數據科學主管進行審查和批准。這個批准行為本身也必須被記錄下來。
    
3. **測試/驗證階段 (TEST/VAL)**：在獲得批准後，系統將 DEV 包部署到 TEST 環境。在這裡，模型會在一個獨立的、更大的驗證數據集上運行一套更嚴格的測試，包括**公平性、偏見和穩健性分析**。此階段會生成一份**正式的驗證報告**，並創建一個新的、標記為 `-val` 的“GxP 包”。
    
4. **審批關卡 #2 (電子簽名)**：要將模型部署到生產環境，必須經過最嚴格的審批，通常由**質量保證 (QA) 團隊**執行。此審批需要一個符合法規的**電子簽名 (Electronic Signature)**。
    
5. **生產階段 (PROD)**：在獲得 QA 簽名後，系統才會將經過驗證的 “VAL GxP Package” 部署到生產環境。部署後，還會執行冒煙測試，並持續監控模型的線上性能。

白板圖 #3：不可變的“GxP 包” (The Immutable "GxP Package")
此圖定義了在環境之間傳遞的、作為部署基本單位的“GxP 包”的內容。
![[Pasted image 20250928033325.png]]
在 GxP 環境中，我們部署的不僅僅是一個模型文件，而是一個包含所有上下文的、**不可變的、經過版本控制的包**。這個包是**可審計性和可重現性**的物理體現。

- **資產 (Assets)**：包含模型本身、定義其運行環境的 Dockerfile，以及描述其預期輸入的數據模式。
    
- **可追溯性信息 (Traceability Info)**：一個“物料清單”，精確記錄了用於構建此包的代碼提交哈希值、數據版本哈希值等。
    
- **證據與報告 (Evidence & Reports)**：所有自動化測試和驗證的產物，如單元測試報告和正式的驗證報告，都與包捆綁在一起。
    
- **審計與批准 (Audit & Approvals)**：所有通過關卡的批准記錄，包括電子簽名證書，都作為包的一部分存檔。
    

這個包確保了“**在 VAL 環境中驗證的，就是最終在 PROD 環境中部署的**”，中間沒有任何改變。監管機構可以隨時審查這個包，以完全了解模型的來源和質量。


白板圖 #4：工具鏈與審計追蹤生成 (Toolchain & Audit Trail Generation)
此圖展示了具體的工具選擇以及它們如何協同工作，自動生成一個全面的審計追蹤。
![[Pasted image 20250928033503.png]]
審計追蹤不是事後添加的，而是由構成 CI/CD 系統的**每個工具在運行過程中自動生成**的。

- **版本控制工具 (Git, DVC)**：為代碼和數據的每一次變更提供了不可變的標識符（哈希值）。
    
- **編排與追蹤工具 (Jenkins, MLflow)**：記錄了每一次管線運行和模型訓練的詳細過程和結果。
    
- **驗證工具 (Great Expectations)**：生成了關於數據質量的客觀證據。
    
- **審批與簽名工具 (Jira, DocuSign)**：將人為的決策過程數字化、文檔化並賦予其法律效力。
    
- **集中式日誌系統 (Splunk, ELK Stack)**：這是審計追蹤的核心。它從所有其他工具中收集日誌和事件，並將它們存儲在一個安全的、不可變的、易於搜索的地方。當監管機構需要審查時，可以從這個單一的來源生成所有必要的報告。


---

**中文解說：**

這個問題的難點在於將敏捷的CI/CD理念與嚴格、看似瀑布式的監管要求相結合。成功的回答需要展示對MLOps和受監管軟體開發生命週期（SDLC）的雙重深刻理解 。  

**1. 問題解析與範疇界定 (Problem Clarification & Scoping)**

- **模型的類型：** 這個管道是為特定類型的模型（例如，僅限於基於TensorFlow的影像分類模型）設計，還是需要一個通用的、可配置的框架？
    
- **審批流程的具體要求：** 人為審批需要什麼級別的證據？是否需要生成一份標準化的「模型驗證報告」？
    
- **部署目標：** 生產環境是雲端的API端點，還是嵌入式醫療設備？
    
- **風險級別：** 模型的風險級別如何（例如，是提供信息參考，還是直接用於診斷）？這將決定驗證的嚴格程度。
    

**假設：**

- 設計一個通用框架，可通過配置文件適應不同類型的模型。
    
- 審批需要一份自動生成的PDF報告，包含數據摘要、模型訓練細節、所有測試結果和性能指標。
    
- 部署目標為雲端的REST API。
    
- 模型為高風險的診斷輔助工具。
    

**2. 系統架構與工具鏈 (System Architecture & Toolchain)** 一個合規的CI/CD4ML系統架構應圍繞「萬物皆代碼」(Everything as Code) 和「不可變基礎設施」的原則構建。

- **版本控制核心 (Version Control Core)：**
    
    - **Git (例如，GitHub, GitLab)：** 作為所有代碼（模型代碼、測試代碼、管道定義代碼）和配置的唯一真實來源 (Single Source of Truth)。使用Git Flow或類似的分支策略來管理開發、發布和熱修復。
        
- **CI/CD引擎 (CI/CD Engine)：**
    
    - **Jenkins, GitLab CI, 或 GitHub Actions：** 負責執行管道中定義的自動化任務。管道本身應被定義為代碼（例如，`Jenkinsfile`, `.gitlab-ci.yml`），並存儲在Git中。
        
- **資產管理與註冊 (Asset Management & Registry)：**
    
    - **DVC (Data Version Control)：** 用於對大規模的訓練/測試數據集進行版本控制。DVC將數據的元數據（指向S3中不可變數據的指針）存儲在Git中，實現了數據與代碼的同步版本化 。  
        
    - **容器註冊中心 (例如，AWS ECR, Artifactory)：** 存儲版本化的Docker鏡像，這些鏡像定義了模型的訓練和服務環境。
        
    - **模型註冊中心 (例如，MLflow Model Registry, Vertex AI Model Registry)：** 這是系統的核心。它不僅存儲訓練好的模型文件，還存儲其完整的元數據：訓練數據版本、代碼版本、性能指標、驗證報告、以及其在生命週期中的狀態（`development`, `staging`, `production`）。  
        
- **基礎設施即代碼 (Infrastructure as Code, IaC)：**
    
    - **Terraform, AWS CloudFormation：** 用於定義和管理DEV, TEST, PROD環境的基礎設施。這確保了環境的一致性和可再現性。
        

**3. GxP合規的CI/CD流程詳解** 候選人應詳細描述一次典型的端到端流程：

- **第1步：開發 (Development - DEV 環境)**
    
    1. 數據科學家在一個`feature`分支上開發模型代碼。
        
    2. 當代碼被推送到該分支時，CI觸發器啟動一個**開發管道**。
        
    3. 該管道運行**單元測試**、**代碼質量檢查 (linting)**，並在一個小規模的樣本數據集上進行快速的**模型訓練和測試**，以確保代碼能夠正常運行。此階段的目標是快速反饋。
        
- **第2步：集成與驗證 (Integration & Validation - TEST/VAL 環境)**
    
    1. 開發完成後，數據科學家提交一個**合併請求 (Merge Request)** 到`develop`或`release`分支。
        
    2. 合併請求觸發一個更為嚴格的**驗證管道**。
        
    3. **構建階段：** 管道使用IaC（Terraform）在一個隔離的TEST環境中創建所需的基礎設施。
        
    4. **測試階段：**
        
        - **數據驗證：** 使用工具（如Great Expectations）檢查完整的、版本化的驗證數據集是否符合預期的模式和分佈。
            
        - **模型訓練：** 在完整的驗證數據集上重新訓練模型。
            
        - **模型評估：** 在預留的測試集上評估模型性能（準確性、精確度、召回率等），並與預定義的閾值進行比較。
            
        - **魯棒性與公平性測試：** 針對已知子群體（例如，不同種族、性別）測試模型性能，檢查是否存在偏見。
            
        - **生成驗證報告：** 將所有測試結果、性能指標、數據集版本和代碼版本匯總成一份PDF報告。
            
    5. **註冊模型候選版：** 如果所有測試通過，管道會將模型、其元數據和驗證報告打包，註冊到模型註冊中心，並標記為`staging`狀態。
        
- **第3步：審批與部署 (Approval & Deployment - PROD 環境)**
    
    1. 模型進入`staging`狀態後，系統會自動通知指定的**QA團隊或審批人員**。
        
    2. QA團隊審查模型註冊中心中的驗證報告和所有相關資產。
        
    3. **人為審批門控：** QA審批員在CI/CD系統中（例如，Jenkins有一個「Approve」步驟）提供**電子簽名**來批准部署。這個操作會被嚴格記錄在審計日誌中。
        
    4. 審批通過後，觸發**部署管道**。
        
    5. 部署管道首先將模型在註冊中心中的狀態提升為`production`。
        
    6. 然後，它使用IaC在PROD環境中部署或更新服務（例如，更新AWS SageMaker Endpoint或Kubernetes Deployment），加載新的`production`模型。部署策略可以採用藍綠部署或金絲雀發布，以降低風險 。  
        
- **第4步：監控與退役 (Monitoring & Decommissioning)**
    
    1. 部署後，模型性能會被持續監控。
        
    2. 所有的管道運行記錄、審批記錄和部署日誌都存檔在一個長期、不可變的存儲中，以備審計。
        

這個設計的核心思想是，通過將監管要求轉化為自動化的、由代碼定義的流程，系統在確保合規性的同時，最大限度地保留了敏捷性。候選人通過這樣的回答，證明了其不僅是ML專家，也是一位理解嚴格監管環境的可靠的系統架構師。



### 問題 4：為百人級數據科學家團隊設計實驗與模型註冊平台

**Question 4: Design an Experiment Tracking and Model Registry Platform for a 100+ Person Data Science Team**

- **Problem Statement:** GSK's large data science team, spread across different therapeutic areas, faces challenges with experiment reproducibility, collaboration, and knowledge retention. Experiments are tracked manually, making it difficult to compare results, share models, and preserve project history.
    
- **System Requirements:**
    
    - **Usability:** The platform must be user-friendly for data scientists, with simple APIs (Python/R) and an intuitive web UI.
        
    - **Flexibility & Agnostic:** Support various ML frameworks (TensorFlow, PyTorch, scikit-learn) and compute environments (local, HPC, cloud).
        
    - **Comprehensive Metadata Capture:** Automatically or semi-automatically capture source code version, parameters, metrics, artifacts (model weights, plots), and dataset versions.
        
    - **Model Lifecycle Management:** Support model state transitions (e.g., Experimental, Staging, Production, Archived) with documented approvals.
        
- **Task:** Propose a platform architecture, including backend services, frontend UI, and client SDKs. Explain how this platform would foster collaboration and knowledge sharing across the team.


**問題陳述：** GSK擁有一個超過100名數據科學家和機器學習研究員的龐大團隊，他們分佈在不同的治療領域（如腫瘤學、免疫學），從事著從早期靶點發現到生物標誌物開發的各種項目。當前，團隊面臨著嚴峻的挑戰：

- **實驗混亂：** 實驗（例如，不同的特徵集、模型架構、超參數）的追蹤大多依賴於手動記錄或本地日誌，導致結果難以復現和比較。
    
- **協作困難：** 優秀的模型和發現很難在團隊和項目之間共享和重用。
    
- **知識流失：** 當成員離開團隊時，其項目的歷史記錄和關鍵決策過程往往會丟失。
    

您的任務是設計一個集中式的、可擴展的平台，以解決上述問題。該平台需要提供兩大核心功能：

1. **實驗追蹤 (Experiment Tracking)：** 允許數據科學家記錄他們每一次模型訓練運行的所有相關信息。
    
2. **模型註冊 (Model Registry)：** 作為一個中央存儲庫，管理經過驗證的、準備好被下游應用（如部署或進一步分析）使用的模型。
    

**系統要求：**

- **易用性：** 平台必須對數據科學家友好，提供簡單的API（主要是Python和R）和直觀的Web UI，以最小的學習成本集成到他們現有的工作流程中。
    
- **靈活性與語言無關性：** 平台應支持各種機器學習框架（TensorFlow, PyTorch, scikit-learn, R/caret）和計算環境（本地筆記本、HPC集群、雲實例）。
    
- **全面的元數據捕獲：** 能夠自動或半自動地捕獲包括：
    
    - 源代碼版本（Git commit hash）。
        
    - 模型參數和超參數。
        
    - 性能指標（例如，AUC, F1-score）。
        
    - 生成的模型文件（artifacts），如模型權重、可視化圖表等。
        
    - 數據集版本。
        
- **模型生命週期管理：** 支持模型從「實驗性」(Experimental) 到「預備」(Staging)，再到「生產」(Production) 或「歸檔」(Archived) 的狀態轉換，並記錄每次轉換的理由和審批人。
    
請闡述您的平台架構設計，包括後端服務、前端界面和客戶端SDK，並解釋您將如何促進團隊的協作和知識共享。

白板圖 #1：平台高層架構 (High-Level Platform Architecture)
此圖展示了平台的核心組件，以及用戶和外部系統如何與之互動，採用了靈活的微服務架構。
![[Pasted image 20250928033614.png]]
這個高層架構圖將平台分為三個主要部分：用戶訪問層、核心後端平台和底層依賴。

1. **用戶與訪問層 (Users & Access Layer)**：
    
    - 平台的用戶角色多樣，包括**數據科學家**、**機器學習工程師**和**項目經理/審核者**。
        
    - 他們可以通過三種方式與平台互動：
        
        - **客戶端 SDK (Client SDK)**：供數據科學家在其程式碼（Python/R）中以編程方式記錄實驗和模型。
            
        - **Web UI**：一個直觀的圖形介面，供所有角色搜索、比較、審查和管理模型。
            
        - **REST API**：為與其他內部工具的集成提供標準接口。
            
2. **核心後端平台 (Core Platform)**：採用微服務架構，將功能解耦，易於獨立擴展和維護。
    
    - **實驗追蹤服務 (Experiment Tracking Svc)**：處理所有與實驗運行相關的請求，如記錄參數、指標等。
        
    - **模型註冊服務 (Model Registry Service)**：管理模型的生命週期，包括版本控制、階段轉換和相關的元數據。
        
    - **產物儲存服務 (Artifact Storage Svc)**：作為一個代理，安全地處理對底層 Blob 儲存的訪問，例如為客戶端生成預簽名的上傳/下載 URL。
        
    - **認證與權限控制服務 (Auth & ACL Service)**：與企業身份提供者集成，管理用戶認證和對不同項目/模型的訪問權限。
        
3. **後端依賴 (Backend Dependencies)**：
    
    - **元數據資料庫 (Metadata DB)**：存儲所有結構化信息，如實驗參數、模型版本、階段標籤等。
        
    - **產物 Blob 儲存 (Artifact Blob Storage)**：存儲大型的非結構化文件，如模型權重、圖表、數據集快照等。
        
    - 其他依賴包括 Git 倉庫（用於關聯代碼版本）和企業身份提供者（用於單點登錄 SSO）。

白板圖 #2：數據科學家工作流與 SDK 互動 (Data Scientist Workflow & SDK Interaction)
此圖詳細展示了一位數據科學家如何使用平台的 Python SDK 來追蹤一次典型的機器學習實驗。
![[Pasted image 20250928033646.png]]
這個工作流展示了平台**易用性**的設計理念。數據科學家只需在其現有代碼中加入幾行簡單的命令，即可實現全面的追蹤。

1. **開始一個運行 (Start a Run)**：使用 `start_run` 上下文管理器，SDK 會通知後端服務創建一個新的實驗運行記錄。所有在此代碼塊內的日誌都會自動關聯到這次運行。
    
2. **記錄元數據 (Log Metadata)**：像 `log_param` 和 `log_metric` 這樣的簡單函數，允許用戶記錄超參數、評估指標、數據集版本等任何關鍵信息。
    
3. **記錄產物 (Log an Artifact)**：記錄大型文件（如模型）的過程對用戶來說是一行代碼，但後端進行了優化。SDK 首先向 `Artifact Storage Service` 請求一個安全的、一次性的上傳 URL，然後直接將文件上傳到雲端 Blob 儲存。這避免了大型文件流經後端服務器，提高了性能和可擴展性。
    
4. **註冊模型 (Register Model)**：訓練結束後，數據科學家可以選擇將某次運行中產生的模型註冊到模型註冊表中，為其指定一個有意義的名稱，這將創建該模型的第一個版本。

白板圖 #3：模型生命週期管理與 Web UI (Model Lifecycle Management & Web UI)
此圖展示了平台的 Web UI 如何支持模型的治理和生命週期管理，特別是階段轉換的審批流程。
![[Pasted image 20250928033716.png]]
Web UI 是協作與治理的中心樞紐，它為非編碼人員（如團隊主管、QA 人員）提供了一個窗口來參與模型生命週期。

1. **模型視圖 (Model View)**：UI 提供了一個清晰的儀表板來查看一個註冊模型的所有版本。用戶可以輕鬆比較不同版本的性能指標，並追溯到產生每個版本的具體實驗運行。
    
2. **階段管理 (Stage Management)**：模型具有明確的生命週期階段，如 **實驗 (Experimental)**、**待部署 (Staging)**、**生產 (Production)** 和 **歸檔 (Archived)**。這些階段標籤清晰地標識了模型的成熟度和用途。
    
3. **審批工作流 (Approval Workflow)**：從一個階段到另一個階段的轉換（特別是進入生產階段）不是自動的，而是需要一個正式的、可追溯的審批流程。
    
    - 當用戶點擊“晉升到生產”時，系統會啟動一個工作流。
        
    - 系統會自動通知指定的審核人員。
        
    - 審核人員在 UI 中審查所有相關信息（性能、參數、源代碼版本等）後，做出批准或拒絕的決定。
        
    - 只有在獲得批准後，後端服務才會更新模型的階段，這個過程會被完整地記錄下來，以備審計。

白板圖 #4：促進協作與知識共享 (Fostering Collaboration & Knowledge Sharing)
此圖是一個概念圖，展示了平台的各項功能如何直接解決大型團隊面臨的具體挑戰。
![[Pasted image 20250928033752.png]]
該平台不僅是一個工具，更是一個促進團隊合作和知識沉澱的文化催化劑。

- **解決可重複性問題**：通過自動捕獲實驗的所有上下文（代碼、數據、環境），任何人都可以一鍵重現任何歷史實驗，這徹底解決了“在我的機器上可以運行”的難題。
    
- **打破知識孤島**：一個集中、可搜索的平台意味著所有團隊的實驗都是一個共享的知識庫。不同治療領域的團隊可以相互學習，發現意想不到的聯繫和方法。
    
- **簡化結果比較**：UI 提供了強大的比較功能，讓數據科學家可以輕鬆地並排比較幾十次實驗運行的結果，從而更快地迭代和優化模型。
    
- **避免重複造輪子**：模型註冊表成為團隊的“共享大腦”。在開始一個新項目之前，團隊可以先搜索是否有現成的、高質量的模型可以作為基礎模型或用於遷移學習，從而極大地加速了開發進程。



---

**中文解說：**

這個問題考察候選人對MLOps生態系統中核心工具的理解，以及設計一個以用戶為中心（此處用戶為數據科學家）的平台的能力。一個好的回答會借鑒現有成熟工具（如MLflow）的設計理念，並根據製藥公司的特定需求進行調整 。  

**1. 問題解析與範疇界定 (Problem Clarification & Scoping)**

- **規模預估：** 100名科學家每天會運行多少次實驗？預計存儲多少個模型？這將影響數據庫和對象存儲的選型。
    
- **集成需求：** 平台需要與哪些現有的內部系統集成？例如，用戶身份驗證系統（LDAP/Active Directory）、代碼倉庫（GitHub Enterprise）、數據湖等。
    
- **安全性要求：** 不同項目之間的實驗數據和模型是否需要嚴格隔離？
    
- **搜索與比較功能：** 用戶需要哪些高級功能？例如，跨越多個實驗比較數十個模型的性能曲線。
    

**假設：**

- 每天約有500-1000次實驗運行。
    
- 需要與公司LDAP和GitHub Enterprise集成。
    
- 支持基於項目的訪問控制。
    
- UI需要提供強大的可視化和比較功能。
    

**2. 平台架構設計 (Platform Architecture Design)** 候選人應提出一個微服務架構，將平台分解為幾個獨立但協同工作的組件。

- **高層架構圖：**
    
    - **客戶端 (Clients)：** Python/R SDK，通過REST API與後端通信。
        
    - **Web前端 (Web Frontend)：** 一個單頁應用（SPA），使用React或Vue.js開發，提供儀表板、實驗比較視圖、模型詳情頁等。
        
    - **後端服務 (Backend Services)：**
        
        - **API網關 (API Gateway)：** 統一的入口點，處理認證、路由和速率限制。
            
        - **追蹤服務 (Tracking Service)：** 核心服務，負責處理來自SDK的日誌請求（記錄參數、指標、標籤等）。
            
        - **模型註冊服務 (Registry Service)：** 管理模型的生命週期、版本和狀態轉換。
            
        - **認證/授權服務 (Auth Service)：** 與公司LDAP集成，管理用戶和權限。
            
    - **數據存儲 (Data Stores)：**
        
        - **元數據數據庫 (Metadata Database)：** 存儲實驗和模型的元數據，如參數、指標、運行狀態等。**PostgreSQL** 或 **MySQL** 是不錯的選擇，因為它們支持事務和結構化查詢。
            
        - **工件存儲 (Artifact Store)：** 存儲大型文件，如模型權重、數據集樣本、可視化圖表等。**對象存儲（如Amazon S3）** 是標準選擇，因其可擴展性和成本效益。
            

**3. 核心功能實現詳解**

- **實驗追蹤 (Experiment Tracking)：**
    
    - **SDK設計：** SDK應提供極簡的API。例如，在Python中：
        
    
    Python
    
    ```
    import gsk_ml_platform as gsk_ml
    
    with gsk_ml.start_run(run_name="my_first_experiment"):
        # Log parameters
        gsk_ml.log_param("learning_rate", 0.01)
    
        # Model training code...
        model.fit(X_train, y_train)
    
        # Log metrics
        accuracy = model.score(X_test, y_test)
        gsk_ml.log_metric("accuracy", accuracy)
    
        # Log artifacts
        gsk_ml.log_artifact("model.pkl", model)
        gsk_ml.log_figure("confusion_matrix.png", fig)
    ```
    
    - **後端實現：** `Tracking Service` 接收這些請求，將參數和指標寫入元數據數據庫，並將工件上傳到S3中的一個唯一路徑（例如，`s3://artifacts/{experiment_id}/{run_id}/`）。
        
- **模型註冊 (Model Registry)：**
    
    - **概念模型：** 註冊中心的核心概念是「註冊模型」(Registered Model)，它有一個唯一的名稱（例如，`oncology_tumor_classifier`）。每個註冊模型可以有多個「版本」(Versions)。
        
    - **UI/SDK交互：** 數據科學家可以在實驗運行的UI頁面上點擊「註冊模型」按鈕，將某次成功運行的輸出工件創建為一個新的模型版本。或者通過SDK完成：
        
    
    Python
    
    ```
    # run_id from a previous experiment
    gsk_ml.register_model(
        model_uri=f"runs:/{run_id}/model.pkl",
        name="oncology_tumor_classifier"
    )
    ```
    
    - **生命週期管理：** `Registry Service` 負責處理狀態轉換的請求。例如，一個團隊負責人可以通過UI或API將模型版本`v5`的狀態從`staging`提升到`production`。這個操作需要權限檢查，並記錄下審批人和時間戳。下游的CI/CD管道可以查詢模型註冊中心，自動拉取標記為`production`的最新模型進行部署 。  
        

**4. 促進協作與知識共享**

- **可發現性：** Web UI是關鍵。它應該允許用戶按項目、標籤、指標範圍或參數值搜索和過濾實驗。一個好的UI可以讓科學家輕鬆地看到「在我們的EGFR抑制劑項目中，所有使用Graph Neural Network且AUC大於0.9的實驗」。
    
- **模型卡片 (Model Cards)：** 每個註冊的模型版本都應附帶一個「模型卡片」，這是一個結構化的文檔，描述了模型的預期用途、性能評估、訓練數據的特點、以及潛在的偏見和限制 。這極大地增強了模型的可理解性和可重用性。  
    
- **註釋與討論：** 在實驗運行和模型版本的頁面上提供評論功能，允許團隊成員討論結果、提出問題和分享見解。
    
- **模板化工作流：** 平台可以提供標準化的項目模板，包含預設的數據處理和模型訓練腳本，鼓勵團隊遵循最佳實踐。
    

**5. 權衡分析**

- **元數據捕獲的粒度：** 平台應該自動捕獲多少信息？過於詳細的自動捕獲（例如，捕獲所有局部變量）可能會產生大量噪音。設計需要在自動化便利性和用戶控制之間找到平衡。
    
- **性能：** 當實驗數量達到數百萬時，元數據數據庫的查詢性能可能會成為瓶頸。需要考慮數據庫索引策略、數據歸檔，甚至使用更專業的時間序列數據庫來存儲指標。
    
- **與現有工具的關係：** 團隊可能已經在使用TensorBoard、Weights & Biases等工具。新平台應該考慮如何與這些工具集成或提供數據導入/導出功能，而不是試圖完全取代它們，以降低採納阻力。
    

這個回答展示了候選人不僅理解MLOps工具的技術細節，還能從用戶（數據科學家）的角度思考，設計一個能夠真正提升團隊生產力和協作效率的平台。



### 問題 5：為藥物發現管道設計數據治理與血緣追蹤系統

**Question 5: Design a Data Governance and Lineage Tracking System for the Drug Discovery Pipeline**

- **Problem Statement:** A drug discovery project at GSK can span years and cost hundreds of millions, with critical decisions based on data derived from dozens of sources through complex transformations. An error in data provenance can have catastrophic consequences.
    
- **System Requirements:**
    
    - **End-to-End Lineage Capture:** Automatically capture the complete data lineage from raw instrument readings (e.g., HTS, sequencers) to final analysis results (e.g., a model's hit list).
        
    - **Fine-Grained Tracking:** Answer specific questions like, "What was the original 96-well plate and well for the activity data of compound CHEMBL123 used in this toxicity model?"
        
    - **Proactive Data Quality Monitoring:** Integrate automated data quality checks at each transformation step against predefined rules.
        
    - **Metadata Management:** Provide a central metadata catalog for all data assets (raw data, models, reports) describing their origin, owner, quality, and usage guidelines.
        
    - **Queryable & Visualizable:** Allow researchers and regulators to easily query and visualize the upstream and downstream dependencies of any data asset through a user-friendly interface.
        
- **Task:** Detail the system architecture, describe how lineage information would be captured, stored, and utilized, and discuss how the system would support scientific reproducibility and future regulatory submissions.

**問題陳述：** 在GSK，一個藥物發現項目從靶點識別到臨床前候選藥物的確定，可能歷時數年，耗資數億美元。在這個過程中，每一個關鍵決策——例如，決定推進某個化合物系列——都基於從數十個不同來源、經過多步複雜轉換的數據得出的結論。如果這些數據的來源或處理過程存在錯誤，將會導致災難性的後果。

您被要求設計一個數據治理與血緣追蹤（Data Governance and Lineage Tracking）系統，以確保藥物發現管道中數據的完整性、可追溯性和可信度。

**系統要求：**

- **端到端血緣捕獲：** 系統必須能夠自動捕獲從原始數據（例如，高通量篩選儀器的原始讀數、測序儀的FASTQ文件）到最終分析結果（例如，一個預測模型生成的hit list）的完整數據血緣。
    
- **細粒度追蹤：** 血緣追蹤需要達到細粒度級別。例如，不僅要記錄哪個腳本處理了哪個文件，還要能回答「用於訓練這個毒性預測模型的數據集中，ID為`CHEMBL123`的化合物的活性數據，其原始來源是哪個實驗的哪塊96孔板的哪個孔？」
    
- **主動數據質量監控：** 系統應集成數據質量檢查功能。在數據轉換的每一步，都應能自動驗證數據是否符合預定義的規則（例如，數值範圍、數據格式、完整性約束）。
    
- **元數據管理：** 為所有數據資產（原始數據集、處理後的數據、模型、報告）提供一個集中的元數據目錄，描述其來源、所有者、質量評級和使用指南。
    
- **可查詢與可視化：** 研究人員和監管人員必須能夠通過一個用戶友好的界面，輕鬆地查詢和可視化任何數據資產的上下游依賴關係。
    

請闡述您的系統架構，描述數據血緣信息是如何被捕獲、存儲和利用的，並討論該系統如何支持科學研究的可再現性和未來的監管申報。


白板圖 #1：系統高層架構 (High-Level Architecture)
此圖展示了系統的三大核心支柱：**數據目錄**（管什麼）、**血緣圖譜**（如何來）和**數據品質引擎**（好不好），以及它們如何協同工作。
![[Pasted image 20250928033936.png]]
這個高層架構將數據治理平台的功能分解為三個協同工作的核心服務，並由一個統一的數據處理層驅動。

1. **數據源與處理管線 (Data Sources & Pipelines)**：所有數據，無論是來自儀器的原始讀數還是來自資料庫，都通過一個**標準化的、受編排的工作流管線**（如 Nextflow）進行處理。這個管線是捕獲治理資訊的關鍵點。
    
2. **三大核心支柱 (The Three Pillars)**：
    
    - **數據目錄 (Data Catalog)**：這是系統的“圖書館目錄”。它回答“**我們有什麼數據？**”的問題。它存儲關於每個數據資產（數據集、模型、報告）的元數據，如其描述、所有者、敏感性級別和使用指南。
        
    - **血緣圖譜 (Lineage Graph)**：這是系統的“家族樹”。它回答“**這個數據從哪裡來，到哪裡去？**”的問題。它以圖的形式存儲數據資產、處理過程和用戶之間的關係，實現端到端的追蹤。
        
    - **數據品質引擎 (Data Quality Engine)**：這是系統的“質量檢驗員”。它回答“**這個數據質量好嗎？**”的問題。它根據預定義的規則，在數據處理的每一步自動執行數據品質檢查。
        
3. **用戶應用 (User-Facing Applications)**：
    
    - **治理與血緣 UI**：提供一個圖形介面，讓研究人員和監管人員可以輕鬆地搜索數據目錄、可視化和查詢數據血緣。
        
    - **警報系統**：當數據品質檢查失敗或發生重要變更時，主動通知相關的數據所有者。


白板圖 #2：血緣捕獲機制與圖模型 (Lineage Capture Mechanism & Graph Model)
此圖詳細解釋了血緣資訊是如何在處理過程中被捕獲，並如何被構建成一個圖模型的。
![[Pasted image 20250928034037.png]]
血緣追蹤的核心是將數據處理的過程轉化為結構化的圖關係。

1. **自動化血緣捕獲**：
    
    - 我們通過“**檢測 (instrumenting)**”數據處理管線中的每個步驟來實現自動化。
        
    - 每個處理步驟（例如，一個腳本或一個程序）在執行完成後，會自動向一個中央的**血緣收集服務**發送一個標準化的**血緣事件 (Lineage Event)**。
        
    - 這個事件就像一張“出生證明”，記錄了：
        
        - **輸入 (Inputs)**：這個步驟使用了哪些數據集和版本。
            
        - **輸出 (Outputs)**：這個步驟生成了哪些新的數據集和版本。
            
        - **過程信息 (Process Info)**：執行這個步驟的代碼版本、參數是什麼。
            
2. **血緣圖模型**：
    
    - 血緣收集服務接收到事件後，會將其解析並在一個**圖資料庫 (Graph Database)**（如 Neo4j）中創建或更新節點和關係。
        
    - **節點 (Nodes)** 代表實體，如 `數據集 (Dataset)`、`處理過程 (Process)`、`用戶 (User)` 等。
        
    - **關係 (Relationships)** 代表它們之間的動作，如 `使用 (USED_BY)`、`生成 (GENERATED)`、`觸發 (TRIGGERED)`。
        
    - 這種模型非常強大，可以輕鬆地回答複雜的問題，例如，“追蹤這個最終結果所依賴的所有原始 HTS 孔板數據”。

白板圖 #3：主動式數據品質 (DQ) 集成 (Proactive Data Quality Integration)
此圖展示了數據品質檢查如何作為管線中的一個強制性“門禁”，而不是事後的分析
![[Pasted image 20250928034116.png]]
這種設計將數據品質從一個被動的監控活動轉變為一個**主動的治理機制**。

1. **運行前檢查 (Pre-check)**：在處理數據之前，可以先驗證輸入數據是否符合預期的模式（schema），提前攔截明顯的錯誤。
    
2. **運行後驗證 (Post-check)**：在生成輸出數據後，立即運行一套全面的數據品質檢查。這就像是產品離開生產線後的質量檢驗。
    
3. **記錄、認證與門禁 (Record, Certify & Gate)**：
    
    - 檢查結果會被永久記錄下來，成為數據元數據的一部分。
        
    - 通過檢查的數據可以被標記為“已認證”，增加其可信度。
        
    - 最關鍵的是**門禁 (Gating)** 機制：如果數據品質檢查失敗，系統會自動將其**隔離**，防止“壞數據”污染下游的分析和模型，並立即通知數據所有者進行處理。

白板圖 #4：治理 UI 與關鍵應用場景 (The Governance UI & Key Use Cases)
此圖展示了用戶如何通過一個直觀的介面來利用這個強大的後端系統，並說明了幾個關鍵的應用場景。
![[Pasted image 20250928034153.png]]
這個系統的最終價值體現在它如何賦能研究人員和滿足合規要求。

- **統一的用戶介面**：UI 將來自**數據目錄**的元數據和來自**血緣圖譜**的關係圖整合在一個視圖中，提供了一個關於任何數據資產的 360 度全景視圖。用戶可以通過點擊圖中的節點，直觀地進行“數據考古”。
- **關鍵應用場景**：
    
    1. **科學可重複性 (Scientific Reproducibility)**：這是科學研究的基石。該系統使得任何結果的重現不再是難事。研究人員可以精確地知道一個結果是如何產生的，包括所有的代碼、數據和環境參數。
        
    2. **監管申報 (Regulatory Submissions)**：在向 FDA/EMA 等機構提交新藥申請時，提供完整的數據溯源鏈是強制性的。該系統可以自動生成一份詳盡的“數據出處報告”，極大地簡化了合規工作。
        
    3. **影響分析與除錯 (Impact Analysis & Debugging)**：當發現上游的某個原始數據有問題時，最大的挑戰是評估其影響範圍。通過血緣圖，研究人員可以立即看到所有依賴於這個“壞數據”的下游分析和模型，從而可以快速進行補救，防止基於錯誤數據做出數百萬美元的錯誤決策。



---

**中文解說：**

這個問題觸及了科學數據管理的核心，特別是在一個高風險、長週期的行業中。一個出色的回答將超越簡單的日誌記錄，提出一個主動的、集成的、元數據驅動的治理框架 。  

**1. 問題解析與範疇界定 (Problem Clarification & Scoping)**

- **追蹤的資產類型：** 除了數據，是否還需要追蹤計算環境（例如，軟件版本、操作系統）的血緣？
    
- **自動化程度：** 血緣捕獲是完全自動化的，還是需要科學家手動註釋？
    
- **現有基礎設施：** 公司現有的數據管道和工作流工具是什麼？新系統需要與它們集成。
    
- **用戶界面需求：** 誰是主要用戶？他們需要回答哪些典型的血緣相關問題？
    

**假設：**

- 需要追蹤數據、代碼和計算環境。
    
- 目標是最大程度地自動化捕獲，但允許手動補充上下文信息。
    
- 公司主要使用基於Python/Spark的工作流，由Airflow編排。
    
- 用戶需要能夠從一個最終報告反向追溯其所有原始數據來源。
    

**2. 系統架構設計 (System Architecture Design)** 候選人應提出一個基於開放標準的、與現有工作流引擎緊密集成的架構。

- **核心理念：** 將數據血緣視為一等公民，而不是事後的日誌。使用像 **OpenLineage** 這樣的開放標準來定義和傳輸血緣元數據。
    
- **架構組件：**
    
    1. **血緣收集器/代理 (Lineage Collectors/Agents)：**
        
        - 這些是集成到數據處理工具中的插件或裝飾器。
            
        - **對於Spark/Airflow：** 使用OpenLineage提供的原生集成。當一個Spark作業或Airflow任務完成時，它會自動發出一個包含輸入、輸出和轉換邏輯信息的血緣事件。
            
        - **對於Python腳本/Jupyter Notebooks：** 提供一個輕量級的SDK，科學家可以在代碼中通過簡單的函數調用來聲明數據的輸入和輸出。
            
    2. **血緣處理後端 (Lineage Processing Backend)：**
        
        - 一個中心服務（例如，基於Kafka的消息隊列和一個流處理應用），負責接收來自所有收集器的血緣事件。
            
        - 該服務解析這些事件，並將血緣關係存儲在一個專門的圖數據庫中。
            
    3. **圖數據庫 (Graph Database)：**
        
        - **Neo4j** 或 **Amazon Neptune** 是存儲血緣關係的理想選擇。數據血緣天然就是一個圖結構（節點=數據資產/處理步驟，邊=依賴關係）。圖數據庫使得複雜的、多層次的血緣查詢（例如，「找到所有受某個已撤回的原始數據集影響的下游模型」）變得高效。
            
    4. **元數據目錄與數據質量引擎 (Metadata Catalog & DQ Engine)：**
        
        - 使用像 **Amundsen** 或 **DataHub** 這樣的開源數據發現工具作為元數據目錄。
            
        - 血緣後端將解析出的元數據（例如，表、列、所有者）推送到這個目錄中。
            
        - 與數據質量工具（如 **Great Expectations**）集成。數據管道中的每個步驟都應包含一個數據質量驗證任務。驗證結果（成功/失敗，以及詳細的報告）將作為一個質量信號，附加到圖數據庫中對應的數據資產節點上。
            
    5. **API、UI與可視化層 (API, UI & Visualization Layer)：**
        
        - 提供一個REST API，用於查詢血緣圖和元數據目錄。
            
        - 一個Web UI，允許用戶搜索數據資產，並以交互式圖形的方式可視化其端到端血緣。點擊圖中的任何節點，都可以看到其詳細的元數據、質量報告和相關代碼。
            

**3. 血緣捕獲的端到端流程** 假設一個簡化的流程：`原始數據 -> 預處理 -> 特徵工程 -> 模型訓練`

1. **原始數據加載：** 一個Airflow DAG中的任務從儀器讀取數據並存入S3。任務結束時，OpenLineage代理發出一個事件：`{ "job": "load_raw_data", "outputs": ["s3://raw/dataset_A.csv"] }`。
    
2. **預處理：** 下一個任務（一個Spark作業）讀取`dataset_A.csv`，進行清洗和標準化，輸出`dataset_B.parquet`。作業結束時，代理發出事件：`{ "job": "preprocess_data", "inputs": ["s3://raw/dataset_A.csv"], "outputs":, "code_version": "git_hash_123" }`。此任務還會運行一個Great Expectations的驗證套件，並將結果附加到血緣事件中。
    
3. **血緣後端處理：** 後端接收到這些事件後，在圖數據庫中創建節點 `DatasetA`, `DatasetB`, `Job_Preprocess`，並創建邊 `(Job_Preprocess)-->(DatasetA)` 和 `(Job_Preprocess)-->(DatasetB)`。`DatasetB` 節點會被標記上數據質量的結果。
    

**4. 系統的價值與應用**

- **可再現性：** 當需要復現一個半年前的分析結果時，可以查詢血緣系統，精確地找到當時使用的數據版本、代碼版本和環境配置，從而實現精確的復現。
    
- **影響分析：** 如果發現某個原始數據集存在質量問題，可以立即通過血緣圖查詢其所有下游依賴，快速評估影響範圍，並主動通知相關的研究團隊。
    
- **簡化調試：** 當一個模型性能下降時，可以檢查其上游數據的血緣，查看是否有數據源或處理邏輯發生了變化，或者數據質量檢查開始報警。
    
- **支持監管申報：** 在向FDA等機構提交新藥申請時，需要提供詳盡的文檔，證明研究結論的數據基礎是可靠的。這個系統可以自動生成所有分析所依賴的數據血緣報告，極大地簡化了合規性工作。
    

這個回答展示了候選人對現代數據工程和治理架構的深刻理解，並能將其應用於解決科學研究中至關重要的可信度和可再現性問題。





## 第二部分：端到端系統在藥物發現與開發中的應用

本部分旨在評估候選人將基礎設施能力應用於解決藥物發現流程中具體、高影響力問題的綜合實力。這些問題要求將一個宏大的科學目標，轉化為一個具體的、可執行的機器學習系統設計。成功的關鍵在於理解藥物發現的迭代本質，特別是「實驗室在環」(lab-in-a-loop) 的概念，即ML系統不僅是靜態的預測API，而是驅動科學發現循環的動態引擎 。系統設計必須支持快速實驗、假設驗證和基於新數據的持續學習。  

### 問題 6：設計一個用於識別新型激酶抑制劑的高通量虛擬篩選系統

**Question 6: Design a High-Throughput Virtual Screening System to Identify Novel Kinase Inhibitors**

- **Problem Statement:** GSK's computational chemistry team has a virtual library of 1 billion synthesizable compounds. Your task is to design a High-Throughput Virtual Screening (HTVS) system to rapidly screen this library against a new protein kinase target and identify potential inhibitors.
    
- **System Requirements:**
    
    - **Scale & Speed:** Evaluate 1 billion compounds and produce a ranked hit list of the top 1,000 candidates within 48 hours.
        
    - **Multi-Stage Filtering:** Employ a multi-stage filtering strategy that progresses from fast, coarse-grained screening to slower, more accurate evaluation to balance cost and accuracy.
        
    - **Cost-Effectiveness:** The cloud computing cost for the entire screening campaign must be controlled within a reasonable budget.
        
    - **Configurability:** The system should be easily configurable for different kinase targets and screening criteria.
        
- **Task:** Describe your multi-stage screening architecture, the models or methods used at each stage, how data flows between stages, and how you would leverage distributed computing to achieve the required throughput.

**問題陳述：** 激酶（Kinases）是藥物發現中的一類重要靶點。GSK的計算化學團隊構建了一個包含10億個可合成化合物的虛擬化合物庫。您的任務是設計一個高通量虛擬篩選（High-Throughput Virtual Screening, HTVS）系統，用於針對一個新的蛋白激酶靶點，從這個龐大的化合物庫中快速篩選出潛在的抑制劑。

**系統要求：**

- **規模與速度：** 系統必須能夠在48小時內完成對10億個化合物的評估，並最終產出一個包含1000個最優候選化合物的排名列表（hit list）。
    
- **多階段過濾：** 考慮到計算成本和準確性的權衡，系統應採用多階段的過濾策略，從快速、粗略的篩選逐步過渡到慢速、精確的評估。
    
- **成本效益：** 整個篩選活動的雲計算成本需要被控制在一個合理的預算內。
    
- **可配置性：** 系統應易於配置，以適應不同的激酶靶點和篩選標準。
    

請闡述您的多階段篩選架構，描述每個階段使用的模型或方法，解釋數據如何在階段間流動，並討論您將如何利用分佈式計算來實現所需的吞吐量和速度。

白板圖 #1：高層架構與篩選漏斗概念 (High-Level Architecture & Funnel Concept)
此圖展示了整個系統的宏觀架構，以及其核心的、旨在平衡速度、成本與準確性的多階段篩選漏斗策略。
![[Pasted image 20250928034345.png]]
這個系統的核心是一個**篩選漏斗 (Screening Funnel)**，它將一個龐大、昂貴的計算問題分解為一系列逐步精化的階段。

1. **輸入與配置 (Input & Configuration)**：一個篩選活動 (campaign) 的啟動需要三樣東西：包含 **10 億個分子的虛擬庫**、**目標激酶的結構資訊**，以及一個定義了所有篩選參數的**配置文件**。
    
2. **工作流編排 (Workflow Orchestration)**：像 **Argo Workflows** 這樣的工具是整個系統的大腦。它讀取配置文件，並將篩選漏斗的每一步作為一個大規模並行計算任務，提交給計算集群執行。
    
3. **可擴展計算集群 (Scalable Compute Cluster)**：計算核心基於 **Kubernetes** 或 **AWS Batch**。它能夠在短時間內啟動數千個計算節點（優先使用廉價的 **Spot 實例**）來並行處理數據，並在任務完成後自動縮減，以控制成本。
    
4. **篩選漏斗 (The Screening Funnel)**：
    
    - **理念**：用計算成本最低、速度最快的方法，盡快排除掉最不可能成為候選藥物的分子。只有通過了前一階段篩選的、更有希望的分子，才會進入下一輪更精確但更昂貴的計算。
        
    - **數據流**：數據量在漏斗的每一層都急劇減少，從 **10 億**級別迅速縮減到 **1000** 個最終的候選分子，從而實現了在 48 小時內完成篩選的目標。

白板圖 #2：詳細的多階段篩選工作流與方法 (Detailed Multi-Stage Screening Workflow & Methods)
此圖詳細描述了篩選漏斗中每個階段所使用的具體計算方法和數據流。
![[Pasted image 20250928034433.png]]
- **階段 1：預處理與物化性質過濾**：此階段的目標是“清理門戶”。它會移除那些不具備基本“類藥性”（例如，分子量太大或太油膩）或已知會干擾實驗的“壞”分子。這個過程極快，可以在數十億分子上並行執行。
    
- **階段 2：2D 指紋與機器學習評分**：此階段使用機器學習模型來預測一個分子與激酶結合的可能性。它將每個分子的 2D 結構轉換為一個數字“指紋”，然後輸入到一個預先在已知激酶抑製劑數據上訓練好的模型中。這個方法比物理模擬快幾個數量級。
    
- **階段 3：3D 藥效團篩選**：此階段開始考慮分子的 3D 形狀。它會檢查一個分子是否能夠形成與已知抑製劑相似的關鍵 3D 相互作用模式（例如，氫鍵供體、受體、芳香環的空間佈局）。
    
- **階段 4：分子對接**：這是計算成本最高的階段。它模擬了將小分子（配體）放入蛋白質靶點的結合口袋中的過程，並計算一個結合親和力分數。只有最有希望的幾萬個分子會進入這一步。
    
- **階段 5：重排序與最終列表生成**：最後，系統會整合來自不同階段的分數，並對候選分子進行聚類分析，以確保最終的 1000 個命中化合物不僅得分高，而且化學結構多樣，為後續的藥物化學優化提供更多樣的起點。

白板圖 #3：分佈式計算與可擴展性策略 (Distributed Computing & Scalability Strategy)
此圖詳細說明了系統如何利用雲端的分佈式計算能力，在規定時間內完成海量計算任務
![[Pasted image 20250928034553.png]]
系統通過“分而治之”的策略來實現極高的計算吞吐量。

1. **數據分區 (Data Partitioning)**：在計算開始前，將包含 10 億個分子的單個大文件，預處理並分割成數千個較小的数据塊（例如，每個塊包含 10 萬個分子），然後存儲在雲存儲中。
    
2. **大規模並行執行 (Massively Parallel Execution)**：
    
    - 工作流編排器（如 Argo）會為**每一個數據塊**創建一個獨立的計算作業 (Job)。
        
    - 這會導致成千上萬的作業被同時提交到 Kubernetes 集群。
        
    - **集群自動擴展器 (Cluster Autoscaler)** 會檢測到大量等待運行的作業，並迅速向雲服務商請求數百甚至數千個新的虛擬機（節點）加入集群。這個過程是全自動的。
        
    - 為了最大限度地節省成本，自動擴展器會優先請求**Spot 實例**，這些實例是雲服務商的閒置資源，價格遠低於常規實例。
        
    - 每個作業都在一個獨立的容器 (Pod) 中運行，處理完自己的數據塊後就終止。
        
3. **結果聚合 (Results Aggregation)**：每個並行作業都會將其處理結果（例如，通過篩選的分子列表）寫回到雲存儲。在一個階段的所有作業都完成後，一個單獨的聚合作業會啟動，將所有這些零散的結果文件合併成一個大的結果文件，作為下一篩選階段的輸入。

白板圖 #4：系統可配置性與重用性 (Configurability & System Reusability)
此圖展示了系統如何通過配置文件將“計算邏輯”與“篩選參數”分離，從而實現高度的靈活度和可重用性。
![[Pasted image 20250928034626.png]]
為了讓這個強大的系統易於被不同的項目團隊使用，我們將**“引擎”**和**“藍圖”**分開。

- **可重用的工作流與工具 (The "Engine")**：這是系統的核心計算邏輯，包括定義了篩選漏斗步驟順序的工作流 DAG，以及包含了所有必需軟件（如 RDKit、AutoDock Vina）的 Docker 容器。這部分代碼是通用的，很少需要修改。
    
- **活動特定的配置文件 (The "Blueprint")**：對於每一次新的篩選活動，科學家不需要編寫新的代碼。他們只需要填寫一個 **YAML 格式的配置文件**。這個文件就像一份藍圖，詳細說明了這次篩選的所有具體參數：
    
    - **目標信息**：要篩選的激酶靶點是什麼，它的 PDB 結構在哪裡。
        
    - **分子庫路徑**：要使用的虛擬分子庫在哪裡。
        
    - **篩選參數**：每一階段的具體設置，例如第一階段的分子量上限、第二階段要使用的機器學習模型、第四階段對接的精度等。
        
    - **計算資源**：是否使用 Spot 實例，對接階段是否需要 GPU 實例等。
        

工作流編排器在啟動時會讀取這個 YAML 文件，並根據其中的參數來執行通用的工作流。這種設計使得平台**極其靈活和可重用**。一個團隊結束了對 KINASE_XYZ 的篩選後，另一個團隊可以立刻用同一個平台，只需提供一個新的配置文件，就可以開始對 KINASE_ABC 的篩選。


---

**中文解說：**

這個問題考察候選人設計大規模科學計算工作流的能力，特別是如何在準確性、速度和成本之間做出明智的權衡。一個優秀的回答會呈現一個漏斗式的架構，逐步縮小候選化合物的範圍 。  

**1. 問題解析與範疇界定 (Problem Clarification & Scoping)**

- **輸入數據：** 10億化合物以何種格式存儲（例如，SMILES字符串文件）？靶點信息以何種形式提供（例如，PDB結構文件、已知活性化合物列表）？
    
- **篩選標準：** 除了預測的結合親和力，是否還有其他篩選標準？例如，藥物相似性（drug-likeness）、合成可及性（synthetic accessibility）、預測的ADMET屬性（吸收、分佈、代謝、排泄、毒性）。
    
- **計算資源：** 可用的計算資源有哪些？是否有內部的高性能計算（HPC）集群，還是完全依賴公有雲？
    

**假設：**

- 化合物庫是一個包含SMILES的Parquet文件，存儲在S3上。
    
- 靶點信息包含其3D結構。
    
- 篩選標準包括結合親和力預測和符合類藥五原則（Lipinski's Rule of Five）。
    
- 完全使用AWS雲資源。
    

**2. 多階段篩選架構 (Multi-Stage Funnel Architecture)** 設計的核心是一個漏斗模型，每一層都使用計算成本和精度遞增的方法來過濾化合物。

- **階段 0：數據預處理與分片 (Data Preprocessing & Sharding)**
    
    - **目標：** 將10億SMILES的單個大文件，處理成適合分佈式計算的格式。
        
    - **實現：** 使用一個Spark作業，讀取Parquet文件，對每個SMILES進行標準化和清洗。同時，為每個化合物計算一些簡單的物理化學性質（如分子量、logP等）。將結果重新分區並寫回到S3，分成數千個較小的Parquet文件，便於後續階段並行處理。
        
- **階段 1：基於結構的快速過濾 (Fast Structural Filtering)**
    
    - **目標：** 從10億化合物中快速剔除明顯不合適的分子，保留約1億（10%）。
        
    - **方法：**
        
        1. **物理化學性質過濾：** 使用Spark並行地對所有化合物應用類藥五原則等簡單規則過濾。
            
        2. **分子指紋相似性篩選：** 如果有一些已知的該靶點的活性參照物，可以計算參照物的分子指紋（如ECFP4），然後使用Spark快速計算10億化合物與這些參照物的相似性（如Tanimoto係數），只保留相似性高於某個閾值的化合物。這屬於配體為基礎的藥物設計（LBDD）方法 。  
            
    - **計算平台：** Apache Spark on EMR。這個階段計算量相對較小，可以在數小時內完成。
        
- **階段 2：基於機器學習的親和力預測 (ML-based Affinity Prediction)**
    
    - **目標：** 從1億化合物中，預測其與靶點的結合親和力，篩選出前100萬（1%）。
        
    - **方法：** 使用一個預訓練好的、計算速度較快的機器學習模型來預測pIC50或結合親和力分數。
        
        - **模型選擇：** 可以是一個基於分子指紋的梯度提升模型（如XGBoost/LightGBM），或者是一個輕量級的圖神經網絡（GNN）。這些模型比物理對接快幾個數量級。
            
        - **部署與執行：** 將模型打包到一個Docker容器中。使用AWS Batch或SageMaker Batch Transform，啟動數千個CPU實例，每個實例處理一部分化合物數據分片。模型推理是高度並行的，可以輕鬆擴展。
            
    - **計算平台：** AWS Batch + EC2 Spot實例，以最大化成本效益。
        
- **階段 3：基於物理的分子對接 (Physics-based Molecular Docking)**
    
    - **目標：** 對100萬個最有希望的化合物進行更精確的物理模擬，篩選出前1萬（1%）。
        
    - **方法：** 使用分子對接軟件（如AutoDock Vina, Schrödinger Glide）。分子對接會模擬小分子在蛋白質靶點結合口袋中的最佳構象和結合能量。這是一個計算密集型任務。
        
    - **部署與執行：** 同樣使用AWS Batch。但這次，每個任務可能需要一個或多個CPU核心，運行數分鐘到數小時。需要一個龐大的、由Spot實例組成的計算集群來在規定時間內完成。
        
    - **計算平台：** AWS Batch + 大量C系列（計算優化型）EC2 Spot實例。
        
- **階段 4：結果匯總與再排序 (Result Aggregation & Re-ranking)**
    
    - **目標：** 匯總所有階段的結果，對前1萬個化合物進行綜合排序，生成最終的1000個hit list。
        
    - **方法：** 使用一個Spark作業，收集階段3的對接分數，並結合階段1和2的各種預測分數（如ADMET屬性、合成可及性分數等）。使用一個加權或多目標優化的方法對化合物進行最終排序。
        
    - **輸出：** 生成一份詳細的報告，包含每個hit化合物的SMILES、所有預測分數、對接構象等信息，供藥物化學家審閱。
        

**3. 系統編排與監控**

- **工作流編排：** 使用AWS Step Functions來編排這整個多階段的工作流。Step Functions可以管理各個階段（Spark作業、Batch作業）之間的依賴關係，處理錯誤和重試，並提供整個流程的可視化。
    
- **監控：** 使用Amazon CloudWatch來監控所有計算資源的使用情況、作業隊列的長度和成本。設置警報，以便在進度偏離預期或成本超出預算時及時通知團隊。
    

這個漏斗式架構有效地平衡了篩選的廣度、深度和成本。候選人通過這個設計，展示了其解決大規模、多階段、異構計算問題的系統設計能力，這在計算驅動的藥物發現中至關重要。



### 問題 7：使用生成式AI設計一個從零開始的藥物設計系統

**Question 7: Design a _De Novo_ Drug Design System Using Generative AI**

- **Problem Statement:** A revolutionary approach beyond virtual screening is _de novo_ drug design, which uses generative AI to create entirely new molecules with desired properties. Your task is to design an end-to-end _de novo_ drug design system that implements a closed-loop "Design-Predict-Test-Learn" cycle, often called a "Lab-in-the-Loop."
    
- **System Requirements:**
    
    - **Generation Module:** A generative model (e.g., VAE, GAN, Transformer) capable of producing chemically valid and novel molecular structures.
        
    - **Prediction/Scoring Module:** A module to score generated molecules on multiple properties, including binding affinity to a target, key ADMET properties, and synthetic accessibility.
        
    - **Decision/Selection Module:** A module to select a small batch of the most promising molecules for synthesis and wet-lab testing based on multi-property scores.
        
    - **Learning/Feedback Module:** A mechanism to ingest wet-lab results (e.g., measured IC50 values) and use this new data to optimize both the generative and predictive models.
        
    - **Human-in-the-Loop:** An interface for scientists to set design goals, monitor the iterative process, and review recommended molecules.
        
- **Task:** Detail the system architecture, describe how the closed-loop iterative process works, and focus on the feedback mechanism for continuous model improvement.

**問題陳述：** 傳統的虛擬篩選是在現有化合物庫中尋找潛在藥物。一個更具革命性的方法是「從零開始的藥物設計」（_De Novo_ Drug Design），即利用生成式AI模型創造出全新的、具有特定期望屬性的分子。

您的任務是設計一個端到端的_de novo_藥物設計系統。該系統的核心是一個生成式模型，但更重要的是，它必須實現一個閉環的「設計-預測-測試-學習」迭代流程，這通常被稱為「實驗室在環」(Lab-in-the-Loop)。

**系統要求：**

- **生成模塊：** 包含一個或多個生成式模型（例如，VAE, GAN, Transformer, Diffusion Model），能夠生成化學上有效且新穎的分子結構。
    
- **預測/評分模塊：** 能夠對生成的分子進行多屬性評分，包括：
    
    - 與特定靶點的結合親和力。
        
    - 關鍵的ADMET（吸收、分佈、代謝、排泄、毒性）屬性。
        
    - 合成可及性（如何輕易地在實驗室中合成）。
        
- **決策/選擇模塊：** 根據多屬性評分，選擇一小批最有希望的分子，推薦給藥物化學家進行合成和濕實驗（wet-lab）測試。
    
- **學習/反饋模塊：** 能夠接收濕實驗的結果（例如，測得的IC50值），並利用這些新的、高質量的數據來優化生成模型和預測模型。
    
- **人機交互：** 系統應提供界面，讓科學家可以設定設計目標、監控迭代過程、審查推薦的分子。
    

請闡述您的系統架構，詳細描述這個閉環迭代流程是如何運作的，並重點討論您將如何實現反饋學習機制來持續改進模型。


白板圖 #1：高層架構與「實驗室在迴路中」(Lab-in-the-Loop) 概念
此圖展示了整個 _De Novo_ 藥物設計系統的宏觀架構，以及其核心的「設計-預測-測試-學習」迭代迴路。
![[Pasted image 20250928035133.png]]
這個系統的核心是一個**閉環的迭代過程**，模擬了科學發現的循環，但速度更快、更智能。

1. **「人在迴路中」介面 (Human-in-the-Loop Interface)**：科學家通過這個介面設置藥物設計的目標（例如，針對某個激酶，希望有什麼樣的活性和安全性）。他們監控系統的進度，並最終審核要合成和測試的候選分子。
    
2. **編排與狀態管理 (Orchestration & State Management)**：像 **Kubeflow Pipelines** 這樣的工具是這個循環的大腦。它負責管理每一輪迭代的執行順序、數據流動和模型的更新。
    
3. **「實驗室在迴路中」迭代循環 (The Lab-in-the-Loop Iterative Cycle)**：
    
    - **1. 生成 (Generation)**：生成模型（例如，一個經過優化的 VAE）基於當前的知識和設計目標，生成一批全新的、化學上有效且可能具有所需性質的分子。
        
    - **2. 預測/評分 (Prediction/Scoring)**：預測模型評估每個生成分子的潛在性能，包括與靶點的結合親和力、ADMET（吸收、分佈、代謝、排泄、毒性）性質以及合成難易度。
        
    - **3. 決策/選擇 (Decision/Selection)**：一個智能算法根據多個預測分數，選出一小批最有前景的分子，供科學家審查。
        
    - **4. 濕實驗室測試與數據攝取 (Wet-Lab Testing & Data Ingestion)**：被批准的分子將在物理實驗室中被合成和測試，產生真實世界的實驗數據。
        
    - **5. 學習/反饋 (Learning/Feedback)**：最新的實驗數據被反饋給系統，用於**優化生成模型和預測模型**，使其在下一輪迭代中表現更好。

白板圖 #2：生成與預測模塊的內部結構 (Internal Structure of Generation & Prediction Modules)
此圖深入探討了生成和預測模塊的具體技術和它們如何工作。
![[Pasted image 20250928035219.png]]
- **生成模塊 (Generation Module)**：
    
    - 核心是一個**生成模型**（例如，基於 Transformer、VAE 或 GAN）。它在一個巨大的已知化學空間上進行訓練，學會了化學分子的語法和特性。
        
    - 為了引導生成過程，可能會結合**潛在空間探索器**（尋找未被充分探索的分子區域）和**強化學習 (RL)**。RL 模型從預測模塊接收“獎勵”（例如，高親和力），然後調整生成策略，使其生成更有希望的分子。
        
    - **輸出**是多個新的、獨特的分子，表示為 SMILES 字符串。
        
- **預測/評分模塊 (Prediction/Scoring Module)**：
    
    - 這是一個包含多個獨立**預測模型**的集合。每個模型專注於評估一個特定的分子性質。
        
    - 例如，一個模型可能預測與靶點的結合親和力，另一個預測分子的合成難易度，還有一個預測其潛在毒性。
        
    - 這些預測模型可以是基於圖神經網絡 (GNN)、隨機森林 (Random Forest) 或更傳統的物理化學模型。
        
    - **輸出**是每個生成分子的多維分數向量。


白板圖 #3：決策與選擇模塊 (Decision & Selection Module)
此圖展示了如何從眾多生成分子中選出最佳候選，以及這個過程如何與人類審批結合。
![[Pasted image 20250928035303.png]]
這個模塊是從數千個有潛力的分子中，挑選出最值得投入昂貴的濕實驗室資源進行測試的少量分子的關鍵。

1. **多目標優化/排序 (Multi-Objective Optimization/Ranking)**：藥物設計很少只追求單一目標（例如，高活性）。通常需要權衡多個相互衝突的性質（例如，高活性但毒性低）。此步驟使用優化算法來平衡這些目標，並生成一個綜合排名的候選列表。
    
2. **多樣性與新穎性過濾 (Diversity & Novelty Filtering)**：為了避免浪費資源測試很多相似的分子，同時也為了探索更廣闊的化學空間，這個步驟會對候選分子進行聚類，並優先選擇那些結構獨特但仍具有高預測分數的分子。目標是從大量候選中選出一個小而精、具有代表性和新穎性的子集（例如，5-10 個分子）。
    
3. **人類審核與批准 (Human Review & Approval)**：最終的決定仍然由人類科學家做出。通過 Web UI，計算化學家可以直觀地檢查系統推薦的分子，考慮其專業知識，並最終批准哪些分子進入濕實驗室進行合成和測試。這確保了 AI 輔助決策的科學嚴謹性。

白板圖 #4：學習與反饋機制 (Learning & Feedback Mechanism)
此圖詳細展示了系統如何從濕實驗室數據中學習，從而不斷優化生成和預測模型，這是「實驗室在迴路中」的精髓。
![[Pasted image 20250928035340.png]]
這是整個「實驗室在迴路中」系統的精髓所在——**持續學習和改進**。

1. **數據攝取與規範化 (Data Ingestion & Harmonization)**：當濕實驗室生成新的實驗結果時（例如，實際測量的 IC50 值），這些數據會被安全地攝取到系統中。這些結果會被精確地關聯到其所對應的、由系統生成的原始分子。
    
2. **模型重訓練觸發 (Model Retraining Trigger)**：系統會監測新數據的累積量，或者按照預定的時間表，自動觸發重新訓練模型的工作流。
    
3. **生成模型優化 (Generative Model Optimization)**：
    
    - 這一步驟使用新的實驗數據來調整生成模型。例如，如果一個生成模型經常產生在濕實驗室中表現不佳的分子，那麼它會通過強化學習等技術進行懲罰，使其在未來的迭代中減少生成此類分子。
        
    - 目標是讓生成模型根據**真實的反饋**，而非僅僅是預測，學會生成更有價值的分子。
        
4. **預測模型優化 (Predictive Model Optimization)**：
    
    - 新的實驗數據也被用來**重新訓練或微調**預測模型。
        
    - 例如，如果一個模型的結合親和力預測與實際測量值存在偏差，那麼新的數據將幫助模型修正這些偏差。
        
    - 目標是提高預測模型的準確性，使其預測更接近真實，從而減少“AI 幻覺”，提高整個系統的效率。
        

通過這個不斷循環的學習過程，系統會變得越來越智能，它生成的分子會越來越接近所需的理想藥物特性，從而加速藥物發現的進程。


---

**中文解說：**

這個問題考察候選人對前沿生成式AI技術及其在科學發現中應用的理解。一個卓越的回答不僅僅是列舉幾種生成模型，而是要闡述如何構建一個能夠自我進化的、與真實世界實驗相結合的智能系統 。  

**1. 問題解析與範疇界定 (Problem Clarification & Scoping)**

- **設計目標的輸入方式：** 科學家如何指定他們想要的分子屬性？是通過一個配置文件，還是一個交互式界面？
    
- **濕實驗反饋的週期：** 從推薦分子到獲得實驗結果需要多長時間？（數週到數月）。這意味著系統的學習循環是異步且延遲較長的。
    
- **分子表示：** 生成和預測時使用哪種分子表示？SMILES字符串、分子圖，還是3D構象？
    
- **初始數據：** 系統啟動時，有哪些可用的初始訓練數據？
    

**假設：**

- 科學家通過Web UI設定一個多目標優化問題，例如：「最大化對靶點A的親和力，最小化hERG抑制（一種心臟毒性），並確保logP在1到3之間」。
    
- 反饋週期為4週。
    
- 主要使用分子圖表示，因為它能更好地捕捉結構信息。
    
- 系統從一個包含已知藥物和性質的公共數據庫（如ChEMBL）開始訓練。
    

**2. 「實驗室在環」的系統架構** 候選人應設計一個模塊化的、事件驅動的架構，以支持長週期的異步迭代。

- **高層架構圖：**
    
    - **用戶界面 (Scientist UI)：** 一個Web應用，用於項目管理、目標設定、進度可視化和候選分子審查。
        
    - **編排引擎 (Orchestration Engine)：** 系統的核心，負責驅動整個迭代循環。可以使用Airflow或Kubeflow Pipelines來實現。
        
    - **生成服務 (Generator Service)：** 部署了生成式模型的API服務。接收請求後，生成一批新的分子。
        
    - **評分服務 (Scoring Service)：** 一個包含多個預測模型的服務。接收一個分子列表，返回一個包含多維度分數的結果。
        
    - **選擇服務 (Selection Service)：** 實現了多目標優化算法（例如，遺傳算法、帕累托最優選擇），從評分後的分子中選擇下一輪要合成的候選者。
        
    - **數據庫 (Databases)：**
        
        - **項目數據庫 (Project DB)：** 存儲每個設計項目的目標、狀態和歷史記錄。
            
        - **分子數據庫 (Molecule DB)：** 存儲所有生成過的分子及其預測和實驗數據。
            
    - **實驗數據接收器 (Lab Data Ingestor)：** 一個API端點或文件監控服務，用於接收來自LIMS（實驗室信息管理系統）的濕實驗結果。
        

**3. 閉環迭代流程詳解** 一個完整的迭代週期如下：

- **第1步：項目啟動 (Initiation)**
    
    1. 科學家在UI上創建一個新項目，定義靶點和多個優化目標函數（帶有權重或約束）。
        
    2. 編排引擎初始化項目，並觸發第一次生成-評分循環。
        
- **第2步：_在計算機中_的設計-預測循環 (In-Silico Design-Predict Loop)**
    
    1. **生成 (Generate)：** 編排引擎調用`Generator Service`，生成一批（例如，10,000個）新分子。
        
        - **模型選型：** 可以是基於圖的VAE或GAN 。這些模型在潛在空間中操作，可以生成多樣且有效的分子圖。  
            
    2. **評分 (Score)：** 生成的分子被發送到`Scoring Service`。
        
        - 該服務並行地調用多個預測模型：一個GNN模型預測結合親和力，幾個基於描述符的QSAR模型預測ADMET屬性，一個快速算法評估合成可及性。
            
    3. **選擇 (Select)：** `Selection Service`接收評分結果，並執行多目標優化，找到在所有目標上表現均衡的「帕累托前沿」上的分子。從中選擇最多樣化的20個分子作為本輪的最終候選者。
        
    4. **審查 (Review)：** 這20個候選分子及其預測屬性被展示在UI上，供藥物化學家審查。他們可以基於專業知識剔除一些結構不合理或合成困難的分子，並最終批准一個列表（例如，10個分子）進入濕實驗。
        
- **第3步：濕實驗與數據反饋 (Wet-Lab & Data Feedback)**
    
    1. 批准的分子列表被送往化學合成和生物活性測試團隊。
        
    2. （4週後）實驗結果（例如，IC50, 溶解度, 代謝穩定性）從LIMS系統發送到`Lab Data Ingestor`。
        
    3. 接收器解析數據，並將其與對應的分子一起存入`Molecule DB`，標記為高質量的「地面真實」數據。
        
- **第4步：學習與模型更新 (Learning & Model Update)**
    
    1. 新實驗數據的到達觸發一個**模型更新事件**。
        
    2. 編排引擎啟動一個**再訓練管道**。
        
    3. **強化學習 (Reinforcement Learning, RL) 優化生成器：**
        
        - 這是一個關鍵的反饋機制 。將生成模型視為RL中的  
            
            **策略 (Policy)**，生成的分子是**行動 (Action)**。將評分服務的多維度分數組合成一個**獎勵函數 (Reward Function)**。
            
        - 使用策略梯度方法（如PPO）對生成器進行微調，使其更傾向於生成能夠獲得高獎勵（即具有期望屬性）的分子。新獲得的濕實驗數據可以被用來更新獎勵函數中的預測模型，或者直接作為高獎勵樣本。
            
    4. **主動學習 (Active Learning) 優化預測器：**
        
        - 新的濕實驗數據被加入到預測模型的訓練集中。
            
        - 可以重新訓練所有的評分模型，以提高它們在感興趣的化學空間內的準確性。
            
    5. 更新後的模型被部署到服務中，準備開始下一個迭代循環。
        

這個系統設計展示了候選人將多種先進ML技術（生成模型、RL、主動學習）整合到一個複雜的、解決實際科學問題的閉環系統中的能力。它體現了從「一次性預測」到「持續學習與進化」的思維轉變。

### 問題 8：設計一個從臨床前數據預測藥物不良反應的ML系統

**Question 8: Design an ML System to Predict Adverse Drug Reactions from Preclinical Data**

- **Problem Statement:** Many promising drug candidates fail in late-stage clinical trials due to unforeseen Adverse Drug Reactions (ADRs). Predicting potential ADRs early in the preclinical stage is critical.
    
- **System Requirements:**
    
    - **Data Integration:** The system must integrate heterogeneous preclinical data: compound properties (SMILES), _in-vitro_ HTS data, _in-vivo_ animal study data (PK/PD, toxicology), and genomics data (transcriptomics).
        
    - **Prediction Target:** The output should be a multi-label classification of risk probabilities across various ADR categories (e.g., cardiotoxicity, hepatotoxicity).
        
    - **Interpretability:** The system must not be a complete "black box" and should provide insights to help toxicologists understand why a compound is flagged as high-risk.
        
    - **Handling Data Imbalance:** The design must account for the fact that many ADRs are rare events, leading to severely imbalanced training data.
        
- **Task:** Describe your system architecture, including data integration strategies, model selection, and how you would implement interpretability and address data imbalance.

**問題陳述：** 在藥物開發過程中，由於未預見的藥物不良反應（Adverse Drug Reactions, ADRs），許多有希望的候選藥物在後期臨床試驗中失敗，造成了巨大的經濟損失。在臨床前階段儘早、盡可能準確地預測潛在的ADRs是至關重要的。

您的任務是設計一個機器學習系統，該系統能夠整合多種來源的臨床前數據，來預測一個候選化合物可能在人體中引發的ADRs。

**系統要求：**

- **數據整合：** 系統必須能夠處理和整合來自以下異構數據源的數據：
    
    1. **化合物屬性：** 化學結構（SMILES）、物理化學性質。
        
    2. **_在體外_ (In-vitro) 實驗數據：** 在多種細胞系上的高通量篩選數據（例如，細胞毒性、靶點活性譜）。
        
    3. **_在體內_ (In-vivo) 動物實驗數據：** 在大鼠或小鼠等模型動物中的藥代動力學（PK/PD）和毒理學研究結果。
        
    4. **基因組學數據：** 化合物處理細胞後的基因表達變化（轉錄組學數據）。
        
- **預測目標：** 系統的輸出應該是該化合物在多個ADR類別（例如，心臟毒性、肝毒性、腎毒性等）上的風險概率。這是一個多標籤分類問題。
    
- **可解釋性：** 由於該系統的預測將影響關鍵的決策，因此它不能是一個完全的「黑箱」。系統需要提供一定的可解釋性，幫助毒理學家理解為什麼某個化合物被標記為高風險。
    
- **處理數據不平衡：** 許多ADRs是罕見事件，導致訓練數據嚴重不平衡。設計中必須考慮如何處理這個問題。
    

請闡述您的系統架構，包括數據預處理和整合策略、模型選擇以及如何實現可解釋性和應對數據不平衡。

白板圖 #1：高層架構與數據整合 (High-Level Architecture & Data Integration)
此圖展示了系統如何從多個異構數據源攝取數據，並通過一個統一的數據處理層。
![[Pasted image 20250928035523.png]]
這個高層架構是一個通用的機器學習系統，但其設計考慮了醫藥領域數據的特殊性。

1. **數據源 (Data Sources)**：多個異構數據源是其核心挑戰，包括化合物結構、體外高通量篩選 (HTS) 數據、體內動物研究數據（藥代動力學/藥效學 PK/PD，毒理學）和基因組學數據。
    
2. **數據攝取與規範化層 (Data Ingestion & Harmonization Layer)**：這是一個關鍵層。它負責：
    
    - **ETL (抽取、轉換、加載)**：將來自不同源的數據抽取出來。
        
    - **數據標準化**：將不同格式的數據轉換為統一的結構。
        
    - **數據質量檢查**：確保數據的準確性和完整性。
        
    - 最終將數據匯總到一個**中央數據湖/數據倉庫**，為下游分析做準備。
        
3. **特徵工程層 (Feature Engineering Layer)**：原始數據不能直接用於模型。這個層負責將原始數據轉換為模型可以理解的數值特徵向量：
    
    - 從 SMILES 字符串中提取**化學描述符**（例如，分子量、指紋）。
        
    - 對高維度的 HTS 或基因組學數據進行**降維**。
        
4. **ML 訓練與推斷系統 (ML Training & Inference System)**：這個層負責模型的生命週期，包括：
    
    - **模型編排**：管理模型的訓練、版本控制和部署。
        
    - **分佈式計算**：利用像 Spark 這樣的分佈式框架或 GPU 來加速模型的訓練。
        
5. **預測與可解釋性服務 (Prediction & Interpretability Service)**：這是系統對外部暴露的接口，它不僅提供預測結果，還集成**可解釋 AI (XAI) 工具**來解釋這些預測。
    
6. **用戶介面 (User Interface)**：毒理學家可以通過儀表板可視化預測結果，並查看 AI 提供的解釋。

白板圖 #2：特徵工程與數據整合策略 (Feature Engineering & Data Integration Strategy)
此圖詳細展示了如何將異構數據轉換為統一的特徵向量，以及如何處理數據不平衡。
![[Pasted image 20250928035614.png]]
將異構數據有效地轉化為統一的特徵向量，是 ML 模型成功的關鍵。

1. **多模態特徵向量 (Multi-modal Feature Vectors)**：對於每個化合物，我們會從其多個數據源中提取相應的特徵：
    
    - **化學特徵**：直接從分子結構中提取，或使用深度學習模型學習分子嵌入 (embeddings)。
        
    - **體外特徵**：來自高通量篩選實驗的數值結果，可能需要降維處理。
        
    - **體內特徵**：來自動物實驗的藥代動力學、藥效學和毒理學指標。
        
    - **基因組學特徵**：來自轉錄組數據的基因表達模式或通路活性分數。
        
2. **特徵融合 (Feature Fusion)**：這些來自不同模態的特徵最終會被**拼接 (concatenate)** 成一個單一的、高維度的特徵向量，作為機器學習模型的輸入。
    
3. **目標變量 (Target Variable)**：模型的輸出是一個**多標籤分類**任務。對於每個化合物，模型會預測它在各種 ADR 類別（如心臟毒性、肝臟毒性、腎毒性）上的**風險概率**。

白板圖 #3：模型選擇與數據不平衡處理 (Model Selection & Handling Data Imbalance)
此圖展示了模型類型選擇，以及如何克服臨床前 ADR 數據稀疏和不平衡的挑戰。
![[Pasted image 20250928035651.png]]
模型的選擇和對數據不平衡的處理，是此系統能否成功預測罕見 ADR 的關鍵。

1. **核心 ML 模型 (Core ML Model)**：
    
    - **集成方法 (Ensemble Methods)**：如 XGBoost 或 LightGBM，對於表格數據表現出色，且魯棒性高。
        
    - **深度學習 (Deep Learning)**：**多任務神經網絡 (Multi-task Neural Networks)** 特別有用。它可以在一個模型中同時預測多種 ADR，並通過共享底層的學習表示來改進對稀有 ADR 的預測。
        
    - **圖神經網絡 (GNN)**：如果我們納入化合物之間的相似性或相互作用，GNN 可能會發揮作用。
        
2. **處理數據不平衡的策略 (Strategies for Handling Data Imbalance)**：這是預測 ADR 的核心挑戰。
    
    - **數據層面技術**：
        
        - **過採樣 (Oversampling)**：例如 **SMOTE**，通過合成“少數類別”的樣本來平衡數據集。
            
        - **欠採樣 (Undersampling)**：謹慎地減少“多數類別”的樣本，但可能導致信息丟失。
            
        - **生成合成數據**：利用生成模型創建更多“陽性”ADR 樣本。
            
    - **算法層面技術**：
        
        - **成本敏感學習 (Cost-Sensitive Learning)**：在訓練模型時，對錯誤預測“少數類別”施加更高的懲罰。
            
        - **多任務學習**：通過讓模型同時學習多個相關任務，可以讓稀有 ADR 的預測從更常見 ADR 的數據中受益。
            
    - **評估指標**：不能使用傳統的準確度。應專注於**精確度 (Precision)、召回率 (Recall)、F1-分數、以及精確召回曲線下面積 (AUC-PR 或 AUPRC)**，這些指標更能反映模型在不平衡數據上的真實性能。

白板圖 #4：可解釋性與毒理學家迴路 (Interpretability & Toxicologist-in-the-Loop)
此圖展示了系統如何提供可解釋性，並讓毒理學家參與到 AI 輔助的決策過程中
![[Pasted image 20250928035735.png]]
在藥物發現中，“黑箱”模型是不可接受的。毒理學家需要理解為什麼一個化合物被標記為高風險。

1. **多標籤 ADR 預測**：首先，模型會生成一個化合物在所有 ADR 類別上的風險概率分佈。
    
2. **可解釋 AI (XAI) 集成**：這是提供洞察的關鍵：
    
    - **特徵重要性 (Feature Importance)**：使用 SHAP 或 LIME 這樣的技術，系統可以指出是哪些特定的化學特性、HTS 讀數或基因表達變化導致了高風險預測。例如，它可能指出“該化合物因其高 LogP 值和特定的分子亞結構而顯示出肝毒性風險”。
        
    - **結構警報 (Structural Alerts)**：系統可以自動識別化合物中是否存在已知的毒性基團（toxicophores）。
        
    - **相似性搜索 (Similarity Search)**：找到與當前化合物在特徵空間上相似，且已知有 ADR 的歷史化合物，提供類比證據。
        
3. **毒理學家儀表板 (Toxicologist Dashboard)**：
    
    - 儀表板是毒理學家與系統交互的主要場所。他們可以直觀地看到 ADR 風險圖譜。
        
    - 最重要的是，他們可以**審查 AI 提供的解釋**。這些解釋幫助他們將 AI 的抽象預測與他們的領域知識聯繫起來，做出最終的、明智的決策。
        
    - 此外，儀表板還應包含一個**反饋迴路**，允許毒理學家就 AI 的預測提供寶貴的反饋，這將被用於持續改進模型。
        

通過將 AI 預測與深入的解釋相結合，該系統不僅提高了預測 ADR 的效率，還增強了科學家對這些預測的信任，並加速了藥物開發過程中關鍵毒性評估。


---

**中文解說：**

這個問題考察候選人處理複雜、異構、不平衡數據的實際能力，以及在一個高風險決策場景中對模型可解釋性的重視。一個好的回答會展示一個系統性的方法來應對這些挑戰 。  

**1. 問題解析與範疇界定 (Problem Clarification & Scoping)**

- **ADR的定義與來源：** ADRs的標籤是如何定義的？它們是來自已上市藥物的標籤信息（例如，SIDER數據庫），還是來自內部的歷史臨床試驗數據？標籤的粒度是怎樣的？
    
- **數據的可用性與質量：** 歷史項目中有多少化合物同時擁有所有這些數據類型？數據缺失的情況有多嚴重？
    
- **可解釋性的要求級別：** 是需要特徵重要性級別的解釋，還是需要更深入的因果推斷？
    
- **模型的用戶：** 誰會使用這個模型的預測結果？毒理學家、藥物化學家？他們需要什麼樣的報告格式？
    

**假設：**

- ADR標籤來自公共數據庫和內部歷史數據的結合，涵蓋約50種常見的ADR類別。
    
- 數據缺失普遍存在，特別是動物實驗數據和轉錄組學數據。
    
- 可解釋性要求達到特徵重要性級別，即能夠指出哪些臨床前信號與某個ADR風險高度相關。
    

**2. 系統架構與數據管道** 候選人應設計一個包含數據整合、特徵工程、建模和報告生成的多階段管道。

- **數據整合層 (Data Ingestion & Integration)：**
    
    1. 為每種數據類型建立一個標準化的數據加載和預處理模塊。
        
    2. **化合物屬性：** 使用RDKit等化學信息學庫從SMILES計算分子指紋（如ECFP）、物理化學描述符。
        
    3. **_在體外_數據：** 將來自不同實驗（assay）的活性數據整理成一個寬表，其中行是化合物，列是每個assay的活性值（例如，IC50）。
        
    4. **動物數據：** 提取關鍵的毒理學終點（如LD50）和PK參數（如半衰期、清除率）。
        
    5. **轉錄組學數據：** 對差異表達基因進行分析，可以將其簡化為一個特徵向量，例如，使用L1000等方法將高維基因表達譜降維到約1000個「標誌性基因」的表達變化。
        
    6. 將所有這些處理過的數據以化合物ID為主鍵，合併到一個統一的特徵矩陣中。
        
- **特徵工程與預處理 (Feature Engineering & Preprocessing)：**
    
    1. **處理缺失值：** 這是關鍵一步。由於不同數據源的可用性不同，特徵矩陣會有大量缺失值。不能簡單地刪除行或列。應採用更複雜的插補策略，例如使用KNNImputer或基於矩陣分解的方法（如SoftImpute）來填充缺失值。也可以將「數據是否缺失」本身作為一個特徵。
        
    2. **特徵縮放：** 對所有數值特徵進行標準化或歸一化。
        
- **建模層 (Modeling Layer)：**
    
    1. **問題定義：** 將問題明確定義為一個多標籤分類（Multi-label Classification）任務。每個ADR是一個獨立的二元分類目標。
        
    2. **模型選擇：**
        
        - **基線模型：** 可以為每個ADR訓練一個獨立的邏輯回歸或梯度提升機（如XGBoost）模型。這是一個簡單、可解釋性強的起點。
            
        - **高級模型：**
            
            - **集成方法：** 隨機森林或XGBoost天然支持多標籤分類，並且能提供特徵重要性排序，是很好的選擇 。  
                
            - **神經網絡：** 一個多層感知機（MLP），其輸出層有N個sigmoid單元，每個單元對應一個ADR的概率。神經網絡可以學習特徵之間的非線性交互。
                
            - **考慮標籤相關性：** 如果某些ADRs傾向于同時出現（例如，肝毒性和腎毒性可能相關），可以使用能夠建模標籤相關性的算法，如Classifier Chains或基於神經網絡的架構。
                
- **報告與可解釋性層 (Reporting & Interpretability)：**
    
    1. **輸出：** 對於一個新的候選化合物，系統輸出一個列表，包含每種ADR的預測風險概率。
        
    2. **可解釋性實現：**
        
        - **SHAP (SHapley Additive exPlanations)：** 對於任何基於樹的模型或神經網絡，都可以使用SHAP來計算每個特徵對單個預測的貢獻度。
            
        - **輸出報告：** 生成的報告應包含：
            
            - 總體風險評估。
                
            - 每個高風險ADR的詳細信息。
                
            - 一個SHAP瀑布圖，可視化地展示哪些臨床前信號（例如，「高hERG抑制活性」、「低代謝穩定性」）將該ADR的風險推高或拉低。這為毒理學家提供了具體的、可操作的見解。
                

**3. 處理數據不平衡的策略** 這是一個核心挑戰，候選人必須提出具體方案 。  

- **重採樣 (Resampling)：**
    
    - **過採樣 (Oversampling)：** 對於每個罕見的ADR類別，可以使用SMOTE（Synthetic Minority Over-sampling Technique）等算法來人工合成少數類樣本。
        
    - **欠採樣 (Undersampling)：** 隨機地從多數類（無該ADR）中移除樣本。
        
- **代價敏感學習 (Cost-sensitive Learning)：**
    
    - 在模型訓練的損失函數中，為少數類的錯分類分配更高的權重（懲罰）。許多庫（如scikit-learn, XGBoost）都支持設置`class_weight`參數。
        
- **選擇合適的評估指標：**
    
    - 在數據不平衡的情況下，準確率（Accuracy）是一個誤導性的指標。
        
    - 應重點關注 **Precision-Recall AUC (PR-AUC)**、**F1-score** 或 **Matthews Correlation Coefficient (MCC)**。對於每個ADR，都應計算這些指標。
        

這個回答展示了候選人解決實際機器學習問題的端到端能力，從混亂的原始數據到可解釋的、能影響關鍵業務決策的洞見，並能熟練運用各種技術來處理數據科學中的常見難題。



### 問題 9：利用生物醫學知識圖譜設計一個藥物重定位系統

**Question 9: Design a Drug Repurposing System Using a Biomedical Knowledge Graph**

- **Problem Statement:** Drug repurposing—finding new indications for existing drugs—is a key strategy to accelerate development and reduce costs. The challenge is to systematically identify novel, scientifically-backed links between drugs and diseases.
    
- **System Requirements:**
    
    - **Knowledge Graph Construction:** Design an automated ETL pipeline to build and update a knowledge graph from public sources like DrugBank, ChEMBL, UniProt, OMIM, and STRING DB.
        
    - **Link Prediction Model:** Train a link prediction model on the graph to identify new, high-probability "treats" relationships between drugs and diseases.
        
    - **Candidate Ranking & Evidence Presentation:** Rank the predicted drug-disease pairs and provide supporting evidence for each high-scoring prediction (e.g., the drug's target is involved in the disease's key pathway).
        
    - **Scalability:** The system must efficiently handle a graph with millions of nodes and tens of millions of edges.
        
- **Task:** Detail the system architecture, including the KG construction process, the choice of link prediction model, and how you would present credible, evidence-backed repurposing suggestions to users.

**問題陳述：** 藥物重定位（Drug Repurposing），即為已批准的藥物尋找新的適應症，是加速藥物上市、降低研發成本的有效策略。其核心挑戰在於如何系統性地發現藥物與疾病之間新的、有科學依據的潛在聯繫。

您的任務是設計一個利用生物醫學知識圖譜（Biomedical Knowledge Graph）的藥物重定位系統。該系統需要整合來自多個公共數據庫的異構信息，並利用機器學習模型來預測新的「藥物-治療-疾病」關係。

**系統要求：**

- **知識圖譜構建：** 設計一個自動化的ETL（提取、轉換、加載）管道，從以下數據源構建和定期更新知識圖譜：
    
    - **藥物數據：** DrugBank, ChEMBL (化學結構、靶點、藥理學)。
        
    - **基因/蛋白質數據：** UniProt, Gene Ontology (功能、通路)。
        
    - **疾病數據：** OMIM, Mondo Disease Ontology (遺傳基礎、表型)。
        
    - **已知的藥物-疾病關係：** CTD (Comparative Toxicogenomics Database)。
        
    - **蛋白質相互作用網絡：** STRING DB。
        
- **鏈接預測模型：** 在構建的知識圖譜上，訓練一個鏈接預測（Link Prediction）模型，其目標是識別出那些目前不存在但很可能存在的「治療」(treats) 關係的邊（edge）。
    
- **候選者排序與證據呈現：** 系統需要對預測出的新藥物-疾病對進行排序，並為每個高分預測提供支持性的證據（例如，藥物的靶點參與了該疾病的關鍵通路）。
    
- **可擴展性：** 知識圖譜可能包含數百萬個節點和數千萬條邊。整個系統必須能夠高效地處理這種規模的圖數據。
    

請闡述您的系統架構，包括知識圖譜的構建流程、鏈接預測模型的選擇與訓練，以及如何向用戶呈現可信的、有證據支持的重定位建議。


白板圖 #1：高層架構與知識圖譜概念 (High-Level Architecture & Knowledge Graph Concept)
此圖展示了整個藥物再利用系統的宏觀架構，以一個**核心的生物醫學知識圖譜**為中心。
![[Pasted image 20250928040006.png]]
這個系統的核心是一個**生物醫學知識圖譜 (Knowledge Graph, KG)**，它將來自多個公共數據源的零散信息連接起來，形成一個巨大的關係網絡。

1. **數據源 (Data Sources)**：多個公共生物醫學數據庫是圖譜的原材料，它們提供了關於藥物、靶點、疾病、蛋白質相互作用等不同角度的信息。
    
2. **知識圖譜 ETL 管線 (Knowledge Graph ETL Pipeline)**：這是一個複雜的自動化過程，負責將異構的、非結構化或半結構化的原始數據轉換為統一的圖結構：
    
    - **數據獲取、清洗與標準化**：處理不同數據庫之間的格式差異和數據質量問題。
        
    - **實體識別與關係抽取 (NER/RE)**：從文本中識別出關鍵實體（如藥物、疾病名稱）和它們之間的關係。
        
    - **本體映射與統一 (Ontology Mapping)**：將不同數據庫中表示相同概念但名稱不同的實體映射到標準的本體（如 UMLS, GO），確保一致性。
        
    - **圖模型轉換**：將處理後的數據轉換為圖資料庫可以理解的節點、邊和屬性。
        
3. **核心生物醫學知識圖譜 (Central Biomedical Knowledge Graph)**：這是整個系統的“大腦”，它存儲在一個**圖資料庫**中（如 Neo4j）。
    
    - **節點 (Nodes)**：代表真實世界的實體（藥物、蛋白質、疾病等）。
        
    - **邊 (Edges)**：代表實體之間的關係（藥物治療疾病、蛋白質參與通路等）。
        
    - **屬性 (Properties)**：附加到節點和邊上的元數據（置信度分數、數據來源、時間戳等）。
        
4. **KG 驅動的 ML 與推理層 (KG-driven ML & Reasoning Layer)**：這個層利用圖譜的結構進行高級分析：
    
    - **鏈接預測模型 (Link Prediction Models)**：識別圖譜中目前不存在但很可能存在的關係（例如，某種藥物治療某種疾病）。
        
    - **圖嵌入 (Graph Embeddings)**：將圖譜中的節點和邊轉換為數值向量，以便機器學習模型處理。
        
    - **圖查詢與路徑查找引擎**：支持在圖譜中高效地查找特定模式或路徑，這對於解釋預測結果至關重要。
        
5. **候選排序與證據呈現服務 (Candidate Ranking & Evidence Presentation Service)**：對預測出的藥物-疾病對進行排序，並生成詳細的、有證據支持的說明。
    
6. **用戶介面 (User Interface)**：研究人員和藥物開發者可以通過儀表板瀏覽候選藥物再利用建議，探索支持證據，並提供反饋。

白板圖 #2：知識圖譜建構 ETL 管線詳情 (Detailed KG Construction ETL Pipeline)
此圖詳細展示了知識圖譜 ETL 管線的各個步驟，以及數據如何從原始來源轉換為圖結構
![[Pasted image 20250928040044.png]]
構建一個大規模的知識圖譜是一個多階段的工程挑戰。

1. **階段 1：數據獲取與預處理**：從不同的公共數據源獲取原始數據。由於這些數據源的格式和結構各不相同，因此需要使用定制的腳本（Python/Spark）進行解析、清洗和初步的結構化。
    
2. **階段 2：實體與關係抽取**：這是將非結構化或半結構化數據轉化為圖譜結構的關鍵。
    
    - **實體識別 (Entity Recognition)**：識別出文本中的關鍵生物醫學實體（例如，分子名稱、蛋白質名稱、疾病名稱）。
        
    - **關係抽取 (Relation Extraction)**：識別這些實體之間的語義關係（例如，Aspirin “抑制” COX-1）。
        
3. **階段 3：本體映射與規範化**：不同的數據源可能使用不同的術語來指代相同的實體（例如，“高血壓”和“Hypertension”）。
    
    - **標準化**：將所有實體映射到標準的生物醫學本體（如 PubChem CID、UniProt Accession）。
        
    - **去重**：根據標準化的 ID 合併來自不同源的相同實體。這確保了圖譜中的實體是唯一的和規範的。
        
4. **階段 4：圖模型加載**：將經過處理和規範化的三元組 (Subject-Predicate-Object) 加載到圖資料庫中。此階段需要定義圖譜的**模式 (Schema)**，即哪些是節點標籤，哪些是邊類型，然後使用圖資料庫的批量加載工具高效地構建圖譜。


白板圖 #3：鏈接預測模型與訓練流程 (Link Prediction Model & Training Flow)
此圖展示了如何使用圖譜數據訓練一個鏈接預測模型來識別新的藥物-疾病關係。
![[Pasted image 20250928040125.png]]
鏈接預測的目標是識別圖譜中缺失但真實存在的新關係，這是藥物再利用的核心。

1. **負採樣 (Negative Sampling)**：在訓練鏈接預測模型時，我們不僅需要已知的真實關係（正樣本），還需要虛假的或不存在的關係（負樣本），以教導模型什麼**不是**真實的鏈接。
    
2. **圖嵌入生成 (Graph Embedding Generation)**：這是將圖譜中的所有節點和邊轉換為稠密數值向量的過程。
    
    - **Node2Vec/DeepWalk**：基於隨機遊走學習節點的語義。
        
    - **TransE/ComplEx**：學習節點和關係的嵌入。
        
    - **圖神經網絡 (GNNs)**：利用圖的結構信息，通過聚合鄰居信息來學習節點表示。
        
    - 這些嵌入捕獲了節點和邊在圖譜中的上下文和語義信息，是鏈接預測模型的輸入。
        
3. **鏈接預測模型訓練 (Link Prediction Model Training)**：
    
    - 這本質上是一個二元分類任務：給定一對（藥物、疾病）實體的嵌入，模型預測它們之間是否存在“治療”關係。
        
    - 模型可以是簡單的分類器，也可以是專門為圖嵌入設計的模型。
        
    - **推斷 (Inference)**：訓練完成後，模型會對所有目前圖譜中沒有“治療”關係的（藥物，疾病）對進行預測，並給出它們之間存在這種關係的概率分數。

白板圖 #4：候選排名與證據呈現 (Candidate Ranking & Evidence Presentation)
此圖展示了如何將鏈接預測的結果轉化為有說服力的藥物再利用建議，並提供透明的證據
![[Pasted image 20250928040204.png]]
將原始的預測分數轉化為對科學家有用的、可信的建議，是這個系統的最終目標。

1. **初步排序與過濾 (Initial Ranking & Filtering)**：首先，將所有預測的藥物-疾病對按照其置信度分數從高到低排序，並過濾掉已知的治療關係或得分極低的預測。
    
2. **證據生成與合理化 (Evidence Generation & Justification)**：這是系統的“解釋器”。對於每一個高置信度的預測，系統會回溯知識圖譜，查找能夠**支持這個預測的生物學路徑或邏輯鏈**。
    
    - **路徑 1 (藥物-靶點-通路-疾病)**：例如，“某藥物靶向某蛋白質，該蛋白質參與某通路，該通路與某疾病相關”。這提供了直接的生物學解釋。
        
    - **路徑 2 (藥物相似性-已知治療)**：例如，“某藥物與已知治療某疾病的藥物在結構上相似”。
        
    - **路徑 3 (副作用反向關聯)**：例如，某藥物引起某副作用，而該副作用本身是某疾病的症狀。這可以提示一個反向治療的可能性。
        
    - 這些路徑提供了**可信的生物學或化學證據**，幫助研究人員理解為什麼 AI 認為這是一個有潛力的再利用機會。
        
3. **用戶介面 / 再利用儀表板 (User Interface / Repurposing Dashboard)**：
    
    - 儀表板以清晰、互動的方式呈現排名靠前的再利用建議。
        
    - 對於每一個建議，用戶可以點擊查看其**詳細信息**，包括：
        
        - 預測置信度。
            
        - **可視化的證據路徑**：以子圖的形式展示支持該預測的 KG 連接。
            
        - 鏈接到原始數據源（如 DrugBank 條目）。
            
        - 提供保存、拒絕或添加評論的功能，實現“人在迴路中”的反饋。
            

通過這種方式，系統不僅能高效地發現新的藥物-疾病鏈接，還能為這些鏈接提供透明、科學的依據，極大地加速了藥物再利用的進程。


---

**中文解說：**

這個問題考察候選人處理和建模複雜網絡數據的能力，特別是將知識圖譜和圖機器學習應用於解決實際的科學問題。一個全面的回答會覆蓋從數據工程到模型部署的全鏈路 。  

**1. 問題解析與範疇界定 (Problem Clarification & Scoping)**

- **圖譜的模式（Schema）：** 知識圖譜中包含哪些類型的節點（例如，Drug, Gene, Disease, Pathway, Phenotype）和邊（例如，`targets`, `participates_in`, `associated_with`, `treats`）？
    
- **更新頻率：** 知識圖譜需要多久更新一次？（例如，每季度）。
    
- **預測的具體目標：** 是只預測藥物和疾病之間的直接「治療」關係，還是也預測更間接的關係，如「藥物A的靶點與疾病B的致病基因相互作用」？
    
- **用戶是誰：** 系統的最終用戶是計算生物學家還是臨床研究科學家？他們對證據的需求有何不同？
    

**假設：**

- 圖譜包含上述所有節點和邊類型。
    
- 目標是預測新的 `(Drug) -[treats]-> (Disease)` 邊。
    
- 用戶是研究科學家，需要清晰的、可追溯的證據鏈。
    

**2. 系統架構設計** 候選人應提出一個包含ETL、建模和服務三個主要部分的架構。

- **1. 知識圖譜構建管道 (KG Construction Pipeline)：**
    
    - **數據提取 (Extract)：** 為每個數據源編寫適配器（adapter），定期從其FTP站點或API下載最新數據。
        
    - **數據轉換 (Transform)：** 這一步是核心。需要將來自不同數據源的、異構的數據轉換為統一的圖模型（節點和邊）。這包括：
        
        - **實體標準化/對齊：** 將不同數據庫中的同一個實體（例如，DrugBank中的「Aspirin」和ChEMBL中的「CHEMBL25」）映射到一個唯一的標識符。
            
        - **模式映射：** 將原始數據的表格結構映射到圖的 `(頭實體) -[關係]-> (尾實體)` 三元組格式。
            
    - **數據加載 (Load)：** 將生成的三元組批量加載到一個專門的圖數據庫中。
        
    - **工具鏈：** 這個ETL管道可以使用Airflow進行編排，數據轉換邏輯可以用Python或Spark實現。
        
- **2. 圖數據庫與存儲 (Graph Database & Storage)：**
    
    - **圖數據庫：** **Neo4j** 或 **Amazon Neptune** 是存儲和查詢大規模知識圖譜的理想選擇。它們提供了高效的圖遍歷查詢語言（如Cypher）。
        
    - **離線處理：** 對於大規模的模型訓練，直接在生產圖數據庫上操作可能效率低下。可以將圖數據定期導出為適合分佈式圖計算框架的格式（如邊列表）。
        
- **3. 鏈接預測建模模塊 (Link Prediction Modeling Module)：**
    
    - **問題定義：** 將藥物重定位問題形式化為知識圖譜中的鏈接預測任務。即，對於所有不存在 `treats` 邊的藥物-疾病對，預測這條邊存在的概率。
        
    - **特徵/嵌入生成：**
        
        - **方法：** 使用圖嵌入（Graph Embedding）技術為圖中的每個節點（藥物、疾病、基因等）學習一個低維的向量表示（embedding）。這些嵌入向量能夠捕捉到節點在圖中的拓撲結構和鄰域信息。
            
        - **模型選擇：**
            
            - **淺層嵌入方法：** 如 **TransE**, **ComplEx** 等，它們專門為知識圖譜設計，通過學習三元組的關係來生成嵌入。
                
            - **圖神經網絡 (GNN)：** 如 **GraphSAGE**, **GCN**。GNN可以利用節點的屬性（例如，藥物的化學結構特徵）和圖的結構信息，生成更豐富的嵌入表示。這是一個更強大和靈活的選擇。
                
    - **訓練鏈接預測器：**
        
        1. **數據準備：** 將已知的 `(Drug, treats, Disease)` 三元組作為正樣本。通過隨機替換頭實體或尾實體來生成負樣本（例如，`(Drug, treats, RandomDisease)`）。
            
        2. **模型訓練：** 訓練一個二元分類器（如邏輯回歸或小型神經網絡）。模型的輸入是藥物和疾病節點的嵌入向量（通常是它們的組合，如拼接或逐元素相乘），輸出是它們之間存在「治療」關係的概率。
            
    - **訓練管道：** 這個訓練過程可以封裝成一個Kubeflow Pipeline或Airflow DAG，定期（例如，每季度在知識圖譜更新後）自動觸發。
        
- **4. 推理與服務層 (Inference & Serving Layer)：**
    
    - **批量推理：** 訓練好模型後，對所有潛在的藥物-疾病對進行批量預測，生成一個包含數百萬預測結果的得分表。
        
    - **API與前端：**
        
        - 開發一個API，允許用戶查詢特定藥物或疾病的重定位潛力。
            
        - 一個Web UI，用戶可以輸入疾病名稱，系統返回一個按預測分數排序的潛在治療藥物列表。
            
        - **證據呈現是關鍵：** 當用戶點擊一個預測結果（例如，「Aspirin -> Alzheimer's Disease」）時，系統不能只顯示一個分數。它應該**實時查詢圖數據庫**，找到支持這個預測的最短路徑或最相關的路徑。例如，UI可以顯示：「阿司匹林 -> 靶向COX-2 -> COX-2參與炎症通路 -> 炎症通路與阿爾茨海默病相關」。這種可解釋的路徑極大地增加了預測的可信度。
            

這個設計展示了候選人構建一個從原始數據整合到複雜圖機器學習，再到可解釋的結果呈現的完整系統的能力。它強調了在科學應用中，提供證據和上下文與提供預測本身同樣重要。

### 問題 10：設計一個優化臨床試驗患者招募的系統

**Question 10: Design a System to Optimize Patient Recruitment for Clinical Trials**

- **Problem Statement:** Timely recruitment of eligible patients is a major bottleneck in clinical trials. The process relies on manual screening of medical records against complex inclusion/exclusion (I/E) criteria, which is inefficient and error-prone.
    
- **System Requirements:**
    
    - **Hybrid Data Processing:** The system must process both structured EHR data (diagnosis codes, lab results) and unstructured data (clinical notes, discharge summaries), where many key I/E criteria reside.
        
    - **Flexible Trial Criteria Definition:** Allow researchers to easily define and input I/E criteria for new trials.
        
    - **Patient-Trial Matching & Ranking:** For a given trial, the system must scan the patient database and return a ranked list of potential candidates based on their match score.
        
    - **Human-in-the-Loop & Interpretability:** The system's output is a recommendation. For each patient, it must clearly show which I/E criteria are met or not met, with links to the source evidence in the EHR.
        
    - **Compliance & Security:** The system must be fully HIPAA compliant.
        
- **Task:** Describe the system architecture, focusing on the NLP pipeline design and how you would implement an efficient and trustworthy human-in-the-loop workflow.

**問題陳述：** 臨床試驗的成功在很大程度上依賴於能否及時招募到符合複雜納入/排除（Inclusion/Exclusion, I/E）標準的患者。傳統的招募方式依賴於醫生手動篩選病歷，效率低下且容易出錯，是導致試驗延遲和成本超支的主要原因之一。

您的任務是設計一個機器學習系統，以半自動化的方式幫助臨床研究協調員（Clinical Research Coordinators, CRCs）從電子健康記錄（EHR）中快速、準確地識別符合特定臨床試驗資格的潛在患者。

**系統要求：**

- **混合數據處理：** 系統必須能同時處理EHR中的兩類數據：
    
    1. **結構化數據：** 診斷碼（ICD-10）、實驗室檢測結果、用藥記錄、人口統計學信息。
        
    2. **非結構化數據：** 臨床筆記、出院小結、病理報告等自由文本。許多關鍵的I/E標準（如疾病分期、特定症狀的描述）只存在於文本中。
        
- **靈活的試驗標準定義：** 系統應允許CRC或研究人員為新的臨床試驗方便地定義和輸入I/E標準。
    
- **患者-試驗匹配與排序：** 對於一個給定的試驗，系統需要掃描患者數據庫，返回一個按匹配度排序的潛在候選患者列表。
    
- **人機迴路與可解釋性：** 系統的輸出是給CRC的建議，而非最終決定。因此，對於每個推薦的患者，系統必須清晰地展示該患者滿足（或不滿足）了哪些具體的I/E標準，並提供原始文本或數據的證據鏈接。
    
- **合規性與安全性：** 系統處理的是高度敏感的患者數據（PHI），必須嚴格遵守HIPAA法規。
    

請闡述您的系統架構，特別是NLP管道的設計，以及如何實現一個高效且值得信賴的人機迴路工作流。

白板圖 #1：高層架構與數據流 (High-Level Architecture & Data Flow)
此圖展示了系統的整體架構，分為**離線數據處理**（構建患者畫像）和**線上匹配**（為新試驗尋找患者）兩個核心部分，所有部分都在一個 HIPAA 合規的環境中運行。
![[Pasted image 20250928040251.png]]
這個架構將複雜的數據處理和實時的匹配需求分開，以提高效率和可擴展性。

1. **離線處理：患者數據攝取與畫像構建**：
    
    - 這個過程定期（例如每晚）在後台運行。
        
    - **ETL 與特徵提取**：系統從醫院的電子健康記錄 (EHR) 數據倉庫中抽取**結構化數據**（如診斷代碼、實驗室結果）和**非結構化數據**（如臨床筆記）。非結構化數據會經過一個複雜的 **NLP 管線**進行處理。
        
    - **可搜索的患者索引 (Searchable Patient Index)**：所有處理過的數據——包括結構化數據和從筆記中提取出的實體（如疾病、藥物）——都會被存儲到一個為快速查詢而優化的數據庫中。這為每個患者創建了一個豐富、可搜索的“特徵畫像”。
        
2. **線上工作流：試驗匹配與審核**：
    
    - **試驗標準定義 UI**：研究人員通過一個用戶友好的介面，將臨床試驗的納入/排除 (I/E) 標準輸入到系統中。
        
    - **患者-試驗匹配引擎**：該引擎接收到試驗標準後，會將其轉換為對**患者索引**的查詢。它會快速掃描整個患者數據庫，並返回一個根據匹配度排序的潛在候選人列表。
        
    - **招募協調員儀表板**：這是**人在迴路中 (Human-in-the-Loop)** 的核心。協調員會在這個儀表板上審核系統推薦的候選人列表，查看支持證據，並做出最終決定。
        
3. **HIPAA 合規環境**：整個系統部署在一個安全的、符合 HIPAA 要求的環境中，確保所有患者數據的機密性和完整性。


白板圖 #2：用於非結構化數據的 NLP 管線詳情 (Detailed NLP Pipeline for Unstructured Data)
此圖詳細描述了如何從自由文本的臨床筆記中提取結構化的、有意義的臨床實體。
![[Pasted image 20250928040328.png]]
這是將混亂的文本轉化為可用數據的魔法所在。

1. **去識別化 (De-identification)**：在進行任何處理之前，第一步也是最重要的一步，是使用一個專門的模型來移除所有受保護的健康信息 (PHI)，如姓名、地址等，以確保 HIPAA 合規。
    
2. **臨床實體識別 (Clinical Entity Recognition, NER)**：這是 NLP 的核心。使用一個在大量生物醫學文本上預訓練過的深度學習模型（如 ClinicalBERT），從文本中“圈出”所有與臨床相關的實體，如疾病、藥物、檢查和手術。
    
3. **實體標準化與鏈接 (Entity Normalization & Linking)**：將識別出的非標準文本（如“心髒病發作”）鏈接到標準的醫學本體（如 ICD-10 編碼 `I21.9`）。這一步使得文本信息變得結構化，可以和結構化 EHR 數據進行統一查詢。
    
4. **上下文分析 (Contextual Analysis)**：這是最精細的一步。僅僅知道文本提到了“癌症”是不夠的，我們還需要知道上下文。
    
    - **斷言狀態 (Assertion Status)**：病人是**真的有**這個病，還是**沒有**這個病（例如，“無癌症病史”），或者是**別人的病**（例如，“家族有癌症史”）？這對於理解**排除標準**至關重要。
        
    - **時間性 (Temporality)**：這個事件是**現在的**、**過去的**還是**未來的**？
        
    - 只有結合了上下文，我們才能準確地判斷一個患者是否符合試驗標準。

白板圖 #3：試驗匹配與「人在迴路中」工作流 (Trial Matching & Human-in-the-Loop Workflow)
此圖展示了從研究人員定義試驗標準到招募協調員審核候選人的完整線上工作流。
![[Pasted image 20250928040403.png]]
這個工作流的核心是將機器的高效篩選與人類的專業判斷相結合。

1. **試驗標準定義**：研究人員使用一個結構化的介面來定義試驗的 I/E 標準，系統會將這些自然語言規則轉換為機器可讀的邏輯。
    
2. **查詢生成與執行**：匹配引擎將這些邏輯規則轉換為對後端患者索引數據庫的查詢。
    
3. **候選人排名**：系統返回一個排序列表，排名最高的患者是與試驗標準最匹配的。
    
4. **招募協調員儀表板**：這是系統最重要的部分，體現了**可解釋性**和**人在迴路中**的設計。
    
    - 協調員會看到一個清晰的候選人列表。
        
    - 當點擊某個患者時，系統會顯示一個**詳細的核對清單**，逐條列出所有 I/E 標準，並標明該患者是**滿足 (Met)**、**不滿足 (Not Met)** 還是**不確定 (Uncertain)**。
        
    - 最關鍵的是，對於每一條標準，系統都會提供一個**“證據 (Evidence)”**鏈接。點擊後，可以直接看到支持該判斷的**原始文本片段**（例如，來自臨床筆記的某句話）或具體的實驗室結果值。
        
    - 基於這些透明的證據，協調員可以自信地做出**最終決定**：確認合格、拒絕，或標記需要進一步人工審查。

白板圖 #4：HIPAA 合規與安全措施 (HIPAA Compliance & Security Measures)
此圖展示了為保護敏感患者數據而設計的多層次安全與合規框架。
![[Pasted image 20250928040443.png]]
HIPAA 合規性不是單一功能，而是一個貫穿整個系統設計和運維的綜合性框架。

1. **訪問控制 (Access Control)**：
    
    - **基於角色的訪問控制 (RBAC)** 是核心。例如，研究人員可能只能看到符合條件的患者的**聚合數量**，而招募協調員才能看到具體的患者列表。
        
    - 實施**最小權限原則**，確保每個用戶只能訪問其完成工作所必需的最少信息。
        
2. **數據保護 (Data Protection)**：
    
    - 所有**靜態數據**（存儲在資料庫或文件系統中）和**傳輸中的數據**（在網絡上流動）都必須進行**強力加密**。
        
    - 在開發和測試環境中，必須使用**數據脫敏或去識別化**技術，防止開發人員接觸到真實的患者 PHI。
        
3. **審計與監控 (Auditing & Monitoring)**：
    
    - 系統必須記錄**每一次**對 PHI 的訪問，生成詳細的**審計日誌**，說明是誰、在什麼時候、出於什麼原因訪問了什麼數據。
        
    - 部署入侵檢測系統，並對異常活動進行實時警報。
        
4. **基礎設施安全 (Infrastructure Security)**：
    
    - 整個系統應部署在一個**虛擬私有雲 (VPC)** 中，與公共網絡隔離。
        
    - 使用防火牆和網絡分段來限制內部服務之間的通信，進一步減少攻擊面。
        
    - 定期進行漏洞掃描和滲透測試。
- 
---

**中文解說：**

這個問題考察候選人將先進的NLP技術應用於解決實際醫療保健流程瓶頸的能力。一個成功的回答需要平衡自動化處理的效率和臨床決策所需的嚴謹性與可解釋性，設計一個賦能而非取代人類專家的系統 。  

**1. 問題解析與範疇界定 (Problem Clarification & Scoping)**

- **I/E標準的複雜性：** I/E標準的格式是怎樣的？是結構化的邏輯表達式，還是自然語言描述？
    
- **系統的部署環境：** 系統是部署在雲端，還是在醫院的防火牆內部（on-premise）？這對數據安全和隱私有重大影響。
    
- **實時性要求：** 系統需要實時響應新的患者數據，還是可以每日批次運行？
    
- **反饋機制：** CRC如何提供反饋（例如，確認或拒絕一個候選者）？系統如何利用這些反饋來改進？
    

**假設：**

- I/E標準以結構化和自然語言混合的形式輸入。
    
- 系統部署在符合HIPAA標準的私有雲環境中。
    
- 系統每日批次更新患者匹配列表。
    
- CRC的確認/拒絕操作會被記錄，用於未來模型的再訓練。
    

**2. 系統架構設計** 候選人應提出一個包含數據預處理、NLP信息提取、規則匹配和人機交互界面的多層架構。

- **1. 數據預處理與匿名化管道 (Data Preprocessing & De-identification Pipeline)：**
    
    - **數據源：** 連接到醫院的EHR數據庫或數據倉庫（如OMOP CDM格式）。
        
    - **ETL流程：** 一個每日運行的Airflow DAG，從EHR系統中提取過去24小時內有更新的患者數據。
        
    - **匿名化：** 在數據處理的第一步，使用專門的工具去除所有直接的PHI（姓名、地址等），用一個唯一的、假名化的ID替換患者標識符。所有後續處理都在匿名化數據上進行。
        
    - **數據轉換：** 將結構化數據（實驗室結果、診斷碼）轉換為標準化的格式。將所有非結構化文本提取出來，準備進行NLP處理。
        
- **2. 臨床試驗協議解析器 (Trial Protocol Parser)：**
    
    - **輸入：** 一份臨床試驗的I/E標準文檔。
        
    - **功能：** 這個模塊負責將複雜的I/E標準分解為機器可執行的查詢。
        
        - 對於結構化標準（例如，「年齡 > 18歲」，「最近一次的血紅蛋白 < 10 g/dL」），將其解析為對結構化數據的SQL查詢或邏輯表達式。
            
        - 對於基於文本的標準（例如，「有轉移性病灶的證據」，「沒有嚴重心力衰竭病史」），將其解析為需要從文本中提取的臨床概念（實體和關係）。
            
- **3. NLP信息提取管道 (NLP Information Extraction Pipeline)：**
    
    - **核心技術：** 這是系統的技術核心。對於每個患者的所有臨床筆記，執行一個NLP管道來提取關鍵的臨床信息。
        
    - **模型選擇：** 使用一個在生物醫學文本上預訓練或微調過的**Transformer模型**，如 **BioBERT**, **ClinicalBERT** 或 **GatorTron**。
        
    - **管道步驟：**
        
        1. **命名實體識別 (NER)：** 識別文本中的臨床實體，如`疾病`、`症狀`、`藥物`、`檢查`、`身體部位`。
            
        2. **斷言狀態檢測 (Assertion Status Detection)：** 判斷每個實體是`肯定的`、`否定的`，還是`假設的`。例如，在「患者否認胸痛」中，「胸痛」是`否定的`。
            
        3. **關係抽取 (Relation Extraction)：** 識別實體之間的關係，例如，`(藥物A) -[治療]-> (疾病B)`。
            
    - **輸出：** 將從文本中提取的結構化信息（例如，`{patient_id, concept: "轉移癌", status: "肯定", document: "path_report.txt"}`）存儲到一個專門的數據庫中（例如，Elasticsearch或一個關係型數據庫），以便快速查詢。
        
- **4. 匹配與排序引擎 (Matching & Ranking Engine)：**
    
    - 對於一個給定的臨床試驗，該引擎執行以下操作：
        
        1. 執行由`協議解析器`生成的結構化數據查詢，得到一個初步的患者候選池。
            
        2. 對於候選池中的每個患者，查詢NLP結果數據庫，檢查他們是否滿足基於文本的I/E標準。
            
        3. 為每個患者計算一個匹配分數，該分數可以基於滿足的標準數量、重要性權重等。
            
        4. 生成一個按分數排序的患者列表。
            
- **5. 人機迴路界面 (Human-in-the-Loop UI)：**
    
    - **設計：** 一個為CRC設計的Web儀表板。
        
    - **功能：**
        
        - 顯示每個臨床試驗的潛在候選患者列表。
            
        - 當CRC點擊一個患者時，界面會清晰地展示一個清單，逐項列出I/E標準，並標明該患者是「滿足」、「不滿足」還是「信息不足」。
            
        - **可追溯性是關鍵：** 對於每個判斷，系統必須提供**證據**。如果判斷來自結構化數據，就顯示該數據點（例如，實驗室結果值和日期）。如果來自文本，就高亮顯示臨床筆記中的相關句子。
            
        - CRC可以對系統的建議進行標記：「符合」、「不符合」、「需要進一步審查」。
            
- **6. 反饋學習循環 (Feedback Learning Loop)：**
    
    - CRC的標記被收集起來，作為高質量的標籤數據。
        
    - 這些數據可以用於：
        
        - 定期微調NLP模型，以提高其在特定臨床概念上的提取準確性。
            
        - 訓練一個更高級的排序模型，學習如何更好地根據CRC的偏好對患者進行排序。
            

這個設計不僅僅是一個NLP應用，而是一個完整的工作流解決方案。它通過智能地結合機器處理和人類專業知識，顯著提高了臨床試驗招募的效率和質量，同時確保了過程的透明度和合規性。

---

## 第三部分：先進表型組學與多模態數據整合

本部分的問題旨在深入探討候選人在處理表型組學團隊最核心、最複雜的數據類型方面的專業知識。這些問題不僅測試對==特定數據模態（如空間組學、高內涵成像）==的深刻理解，還考察了複雜==數據融合技術==的掌握程度。一個核心挑戰在於，高維生物數據中普遍存在的混淆變量和批次效應並非小問題，而是可能導致整個研究無效的系統性偏差 。一個穩健的系統設計必須將混淆因素的緩解視為第一優先級，而不僅僅是一個簡單的預處理步驟。  

### 問題 11：為腫瘤空間轉錄組學數據設計細胞類型解卷積系統

**Question 11: Design a Cell-Type Deconvolution System for Tumor Spatial Transcriptomics Data**

- **Problem Statement:** ==Spatial transcriptomics (ST)== reveals gene expression within a tissue's spatial context. However, current technologies often capture a mix of multiple cells at each measurement spot. To study the tumor microenvironment, it's essential to "deconvolute" each spot's signal to estimate the relative proportions of different cell types (e.g., cancer cells, immune cells).
- 空間轉錄組學（ST）技術能夠在保留組織空間位置信息的同時測量基因表達，為理解腫瘤微環境（TME）提供了前所未有的機會。然而，當前主流的ST技術（如10x Visium）其空間分辨率有限，每個測量點（spot）通常包含多個甚至數十個細胞的混合信號。為了準確研究TME中不同細胞類型（如癌細胞、免疫細胞、基質細胞）的空間分佈和相互作用，必須對每個spot的信號進行「解卷積」（deconvolution），即估算出其中每種細胞類型的相對比例。

- **System Requirements:**
    - **Data Integration Pipeline:** A pipeline to process and align ==ST data== (gene expression with spatial coordinates) and a reference ==single-cell RNA-seq== (scRNA-seq) dataset. 
	 **數據整合管道：** 系統需要一個能夠處理和對齊兩種不同數據模態的管道：
    1. **ST數據：** 包含空間坐標的基因表達矩陣。
    2. **scRNA-seq數據：** 來自相似組織類型的高質量、已註釋細胞類型的單細胞基因表達數據集，作為解卷積的「參考圖譜」

    - **Deconvolution Module:** Implement or integrate algorithms to output a set of cell-type proportions for each ST spot. 
	 **解卷積模塊：** 實現或集成一個或多個解卷積算法，能夠為ST數據中的每個spot輸出一組細胞類型比例（例如，`{T細胞: 0.3, 癌細胞: 0.6, 成纖維細胞: 0.1}`）

    - **Visualization & Validation:** Provide tools to visualize the predicted cell-type distributions on the tissue map and design strategies to validate the deconvolution accuracy. 
	 **可視化與驗證：** 系統應提供可視化工具，能夠在地圖上展示預測的細胞類型分佈。同時，需要設計策略來驗證解卷積結果的準確性

    - **Scalability:** The system must handle large ST datasets with hundreds of thousands of spots.
	 **可擴展性：** 系統應能處理大規模的ST數據集，例如包含數十萬個spots的數據

- **Task:** Detail your system architecture, discuss key data preprocessing and alignment steps (especially batch effect correction), compare the pros and cons of different deconvolution methods, and explain how you would evaluate and validate the results.
- 請闡述您的系統架構，討論數據預處理和對齊的關鍵步驟，比較不同解卷積方法的優劣，並說明您將如何評估和驗證結果的可靠性

白板圖 #1：系統高層架構 (High-Level Architecture)
![[Pasted image 20250928025519.png]]
#### **系統流程概述**
1. **資料輸入**：系統接收兩大類資料：
    - **空間轉錄組 (ST) 數據**：包含每個測量點 (spot) 的基因表達計數和其在組織切片上的 (x, y) 座標。
    - **參考單細胞 (scRNA-seq) 數據**：一個帶有精確細胞類型標註的單細胞基因表達數據集，作為「真實」細胞的參考圖譜。
2. **數據整合管線**：這是系統的基礎。此模組負責清洗、標準化兩種來源的數據，並解決最關鍵的**批次效應 (Batch Effect)** 問題，確保數據可以相互比較。最終，它會從 scRNA-seq 數據中提煉出一個「細胞類型特徵矩陣 (Signature Matrix)」。
3. **解卷積引擎**：這是系統的核心。它利用特徵矩陣和 ST 數據，執行解卷積演算法，計算出每個 ST 測量點中，各種細胞類型（如癌細胞、T細胞、巨噬細胞等）所佔的**比例**。
4. **結果儲存**：計算出的比例矩陣，連同所有中間處理過的數據，會被儲存在一個高效能的資料庫或檔案系統中，以利後續分析。
5. **視覺化與驗證**：終端使用者（通常是生物學家或病理學家）可以透過一個互動式介面，將細胞比例視覺化到原始的組織圖像上，並將結果與病理學金標準（如 H&E 染色影像）進行比對驗證。

白板圖 #2：深入探討 - 數據整合管線 (Data Integration Pipeline)
![[Pasted image 20250928025606.png]]
#### **關鍵步驟說明**
1. **品質控制 (QC)**：
    - **scRNA-seq**：過濾掉低品質的細胞（例如，基因檢測數量太少或粒線體基因比例太高）。
    - **ST**：過濾掉組織邊緣或品質差的測量點。
2. **標準化 (Normalization)**：解決測序深度不同的問題。最常用的方法是 `LogNormalize`，即將每個細胞/點的基因計數除以總計數，乘以一個比例因子，然後取對數。
3. **整合與批次效應校正 (Integration & Batch Correction)**：
    - **問題**：scRNA-seq 和 ST 數據通常來自不同的實驗、平台或技術，存在技術性差異（即批次效應），導致同一個基因在不同數據集中的表達值基線不同。直接比較會產生嚴重偏差。
    - **解決方案**：
        - **Harmony / Seurat CCA**：這些演算法旨在找到一個共享的低維度空間，將來自不同批次的細胞/點「混合」在一起，消除技術差異，同時保留生物學差異。
        - **scVI (Deep Learning)**：使用變分自編碼器 (Variational Autoencoder, VAE) 模型學習一個能代表細胞真實狀態的潛在空間 (Latent Space)，並在模型中明確地將批次作為一個變數來移除其影響。
    - **產出**：經過校正後，兩種數據的基因表達譜就具有可比性了。
4. **特徵矩陣生成 (Signature Matrix Generation)**：
    - 從校正後的 scRNA-seq 數據中，為每個已知的細胞類型，計算其標誌性基因 (marker genes) 的平均表達譜。這個矩陣的維度是 `[基因 x 細胞類型]`，它定義了「一個純粹的 T 細胞長什麼樣」、「一個純粹的癌細胞長什麼樣」。

白板圖 #3：深入探討 - 解卷積引擎與演算法比較
![[Pasted image 20250928025713.png]]
#### **演算法比較 (Pros and Cons)**

|類別|方法範例|優點 (Pros)|缺點 (Cons)|
|---|---|---|---|
|**基於回歸 (Regression-based)**|CIBERSORTx, MuSiC|速度快，計算效率高；概念直觀，易於理解。|假設基因表達是線性混合的，可能不適用於複雜的交互作用；對特徵矩陣的品質非常敏感。|
|**基於機率 (Probabilistic/Bayesian)**|cell2location, Tangram|模型更靈活，能捕捉更複雜的數據分佈；通常更穩健，能提供不確定性估計；**cell2location**能整合鄰近點的空間資訊，結果更平滑真實。|計算成本高，速度慢；模型較複雜，需要更多調參。|
|**基於深度學習 (Deep Learning-based)**|DestVI|作為 scVI 的擴展，能很好地處理批次效應和數據雜訊；可以端到端地進行模型訓練。|需要大量數據和 GPU 資源；模型的可解釋性較差。|

**決策考量**：

- 對於**初步探索**或**計算資源有限**的場景，可以先使用**基於回歸**的方法。
- 為了獲得**最高準確性**和**考慮空間關聯性**，**cell2location** 等機率模型是目前學術界和工業界的首選。
- 如果數據規模極大且批次效應非常複雜，可以考慮 **DestVI**。


白板圖 #4：深入探討 - 視覺化與驗證模組
![[Pasted image 20250928025854.png]]
##### **關鍵功能說明**
1. **視覺化工具**:
    - **空間熱圖 (Spatial Heatmap)**：在組織圖上，用顏色深淺表示**單一細胞類型**（例如 T 細胞）在各個位置的富集程度。這對於識別「免疫熱點」或「癌巢」至關重要。
    - **組成圖 (Composition Plots)**：在每個測量點或特定區域上，用**圓餅圖或堆疊長條圖**顯示所有細胞類型的相對比例。
    - **共定位分析 (Co-localization)**：分析哪些細胞類型傾向於在空間上同時出現。例如，計算 T 細胞和癌細胞比例的空間相關性，以研究免疫細胞的浸潤模式。
2. **驗證策略 (Validation)**:
    - **與金標準比對**：這是最重要的驗證方式。將解卷積預測的細胞分佈圖與同一組織切片的**免疫組織化學 (IHC) 或免疫螢光 (IF) 染色**影像進行比對。例如，預測的 T 細胞熱點區域，是否與 CD3 蛋白（T 細胞標記）的 IHC 染色陽性區域高度重合。
    - **模擬數據評估**：從 scRNA-seq 數據中，人工創建已知細胞比例的「偽測量點 (pseudo-spots)」。然後用我們的系統去解卷積這些偽點，比較預測比例與真實比例的差異（例如計算 RMSE 或 Pearson 相關係數），以此來量化演算法的內在精度。

##### **擴展性 (Scalability) 設計**
- **計算**：整個流程，特別是深度學習方法和大型數據集的批次校正，需要**GPU 加速**。系統應建立在支援 GPU 的 **HPC 叢集**或**雲端平台 (AWS/GCP)** 上。
- **數據處理**：使用如 `Dask` 或 `Spark` 的框架來平行化數據預處理步驟。
- **數據儲存**：對於數十萬甚至百萬級別的測量點，基因表達矩陣非常龐大。應使用針對大數據優化的檔案格式，如 `AnnData (HDF5)` 或 `Zarr`，它們支援高效的按塊讀取 (chunked reading)，無需將整個檔案載入記憶體。
- **架構**：將整個流程打包成 **Docker/Singularity 容器**，確保環境的可重複性，並利用 **Kubernetes** 或 **Nextflow/Snakemake** 等工作流管理工具來編排和自動化整個分析管線。



#### **系統流程概述**

**問題陳述：** 空間轉錄組學（ST）技術能夠在保留組織空間位置信息的同時測量基因表達，為理解腫瘤微環境（TME）提供了前所未有的機會。然而，當前主流的ST技術（如10x Visium）其空間分辨率有限，每個測量點（spot）通常包含多個甚至數十個細胞的混合信號。為了準確研究TME中不同細胞類型（如癌細胞、免疫細胞、基質細胞）的空間分佈和相互作用，必須對每個spot的信號進行「解卷積」（deconvolution），即估算出其中每種細胞類型的相對比例。

您的任務是設計一個系統，該系統能夠整合ST數據和一個參考的單細胞RNA測序（scRNA-seq）數據集，來實現對腫瘤組織切片的細胞類型解卷積。

**系統要求：**
- **數據整合管道：** 系統需要一個能夠處理和對齊兩種不同數據模態的管道：
    1. **ST數據：** 包含空間坐標的基因表達矩陣。
    2. **scRNA-seq數據：** 來自相似組織類型的高質量、已註釋細胞類型的單細胞基因表達數據集，作為解卷積的「參考圖譜」。
- **解卷積模塊：** 實現或集成一個或多個解卷積算法，能夠為ST數據中的每個spot輸出一組細胞類型比例（例如，`{T細胞: 0.3, 癌細胞: 0.6, 成纖維細胞: 0.1}`）。
- **可視化與驗證：** 系統應提供可視化工具，能夠在地圖上展示預測的細胞類型分佈。同時，需要設計策略來驗證解卷積結果的準確性。
- **可擴展性：** 系統應能處理大規模的ST數據集，例如包含數十萬個spots的數據。

請闡述您的系統架構，討論數據預處理和對齊的關鍵步驟，比較不同解卷積方法的優劣，並說明您將如何評估和驗證結果的可靠性。

---

**中文解說：**

這個問題考察候選人對前沿組學數據分析技術的理解，以及處理和整合多模態生物數據的實踐經驗。一個好的回答需要展示對計算生物學算法和可擴展計算架構的雙重掌握 。  

**1. 問題解析與範疇界定 (Problem Clarification & Scoping)**
- **參考數據的質量：** scRNA-seq參考圖譜的質量如何？細胞類型註釋是否準確？它是否包含了ST樣本中可能存在的所有細胞類型？
- **批次效應：** ST數據和scRNA-seq數據通常由不同的實驗產生，存在強烈的技術性批次效應。如何校正這種效應？
- **算法選擇：** 市場上有許多解卷積工具（如RCTD, cell2location, Tangram），它們基於不同的統計假設。系統應該支持多種算法以供比較嗎？
- **驗證方法：** 除了計算上的交叉驗證，是否有實驗性的方法（如免疫熒光染色）可以作為「金標準」來驗證結果？

**假設：**
- 提供一個高質量的scRNA-seq參考圖譜。
- 批次效應是主要的挑戰之一。
- 系統應設計為一個靈活的框架，可以插入不同的解卷積算法。
- 驗證將主要依賴計算指標和與組織學專家的定性比較。

**2. 系統架構設計** 候選人應設計一個模塊化的分析管道，由工作流引擎（如Airflow, Nextflow）編排。
- **1. 數據預處理與質控模塊 (Data Preprocessing & QC Module)：**
    - **ST數據處理：**
        1. 加載ST數據（基因表達矩陣和空間坐標）。
        2. 進行標準質控：過濾掉表達基因過少或線粒體基因比例過高的低質量spots。
        3. 數據標準化（例如，log-normalization）。
    - **scRNA-seq數據處理：**
        1. 加載scRNA-seq參考數據（表達矩陣和細胞類型標籤）。
        2. 進行類似的質控和標準化。
- **2. 數據整合與批次校正模塊 (Data Integration & Batch Correction Module)：**
    - **挑戰：** 這是最關鍵的步驟之一。直接比較來自兩種不同技術平台的基因表達值是不可靠的。
    - **方法：**
        1. **基因匹配：** 找到在ST和scRNA-seq數據集中都存在的共同基因集。
        2. **批次效應校正：** 使用專為整合scRNA-seq數據設計的算法來對齊兩個數據集。例如，**Seurat v4** 的 `FindTransferAnchors` 和 `IntegrateData` 功能，或者 **Harmony** 算法。這些方法旨在找到跨數據集的「錨點」（即細胞狀態相似的細胞對），並將數據投影到一個共同的、消除了批次效應的低維空間中。
- **3. 解卷積模塊 (Deconvolution Module)：**
    - **輸入：** 經過整合的ST表達數據和scRNA-seq參考圖譜。
    - **算法實現：**
        - **基於回歸的方法 (Regression-based)：** 如 **CIBERSORT**, **MuSiC**。這些方法將每個spot的表達譜建模為參考圖譜中各細胞類型平均表達譜的線性組合。它們計算速度快，但可能忽略了細胞間的變異性。
        - **基於概率模型的方法 (Probabilistic)：** 如 **cell2location**, **RCTD**。這些方法通常使用更複雜的統計模型（如負二項分佈），能夠更精確地建模計數數據的特性，並提供不確定性估計。計算成本較高。
    - **系統設計：** 該模塊應設計為可插拔的。用戶可以通過配置文件選擇使用哪種算法，並設置其參數。對於計算密集型算法，應利用分佈式計算（如Dask或Spark）來並行處理每個spot或數據子集。
- **4. 結果存儲與可視化模塊 (Result Storage & Visualization Module)：**
    - **存儲：** 將解卷積結果（每個spot的細胞類型比例矩陣）與原始ST數據一起存儲在一個統一的數據對象中（如AnnData格式）。
    - **可視化：** 開發一個交互式可視化組件（可以使用Plotly Dash或Shiny）。該組件應能：
        1. 將原始的H&E染色組織影像作為背景。
        2. 在其上疊加每個spot的解卷積結果。例如，用戶可以選擇顯示「T細胞」的豐度熱圖，或者用餅圖表示每個spot的主要細胞類型構成。
        3. 允許用戶並排比較不同解卷積算法的結果。
**3. 驗證策略** 由於缺乏絕對的「金標準」，需要從多個角度進行驗證。
- **模擬數據驗證：**
    1. 創建「偽空間」數據：從scRNA-seq數據中，人工混合不同細胞類型的表達譜來模擬ST的spots，此時我們知道每個spot的真實細胞類型比例。
    2. 在這些模擬數據上運行解卷積算法，並將預測比例與真實比例進行比較（例如，計算相關性或均方根誤差）。
- **與已知生物學知識的一致性：**
    - 將預測的細胞類型分佈與病理學家對H&E影像的判讀進行比較。例如，在被病理學家標記為「淋巴細胞浸潤」的區域，解卷積結果是否顯示出高比例的T細胞和B細胞？
- **下游分析的魯棒性：**
    - 使用解卷積結果進行下游分析，例如識別不同細胞類型之間的空間共現模式。看這些模式是否在生物學上是可解釋的，並且在不同算法之間是否一致。

這個回答展示了候選人不僅了解具體的生物資訊學算法，還能將它們置於一個可擴展、可驗證的系統工程框架中，這對於將前沿科學研究轉化為可靠的分析工具至關重要。



### 問題 12：設計一個多模態融合模型對腫瘤患者進行分層

**Question 12: Design a Multimodal Fusion Model for Tumor Patient Stratification**

- **Problem Statement:** Precision oncology requires integrating multiple data sources to predict a patient's response to therapy. A single data modality is often insufficient.
- **問題陳述：** 精準醫療的目標是根據每個患者的獨特特徵來制定個性化的治療方案。在腫瘤學中，這意味著需要整合多種數據來源來預測患者對特定療法（例如，一種新的免疫療法）的反應。單一數據模態（如僅基因組或僅影像）往往不足以捕捉到決定治療結果的複雜生物學信號。
    
- **System Requirements:**
    
    - **End-to-End Pipeline:** Design a pipeline from raw data to final prediction for three modalities: Whole-Slide Pathology Images (WSI), Bulk RNA-seq data, and structured clinical data (EHR). 
     **端到端管道：** 設計從原始數據（WSI影像、FASTQ文件、臨床數據表）到最終預測的完整管道
        
    - **Feature Extraction:** Design effective feature extraction strategies for each modality.
	 **特徵提取：** 為每個數據模態設計有效的特徵提取策略
        
    - **Multimodal Fusion Architecture:** Propose a specific deep learning architecture to fuse features from the three modalities to predict patient response to a PD-1 inhibitor (responder vs. non-responder).
	 **多模態融合架構：** 提出一個具體的深度學習模型架構，用於融合來自三個模態的特徵並做出最終預測。

    - **Interpretability:** The system must provide insights into which features (e.g., specific image patterns, gene expression levels) are most important for the prediction.
	 **可解釋性：** 系統需要提供洞見，幫助研究人員理解是哪些特徵（例如，影像中的特定形態模式、某些基因的表達水平，或臨床因素）對預測患者反應最為重要

    - **Performance Evaluation:** Describe how you would evaluate the model, especially on a potentially small and imbalanced dataset.
     **性能評估：** 描述您將如何評估模型的性能，特別是在一個可能樣本量不大且類別不平衡的數據集上。
        
- **Task:** Detail the system architecture and model design, focusing on your chosen fusion strategy and the rationale behind it.
您的任務是設計一個端到端的機器學習系統，該系統能夠融合以下三種主要的數據模態，來對非小細胞肺癌（NSCLC）患者進行分層，預測他們對一種PD-1抑制劑療法的反應（響應者 vs. 無響應者）：
1. **全切片病理影像 (WSI)：** H&E染色的腫瘤組織切片影像。
2. **批量RNA測序數據 (Bulk RNA-seq)：** 來自同一腫瘤組織的基因表達譜。
3. **結構化臨床數據：** 從EHR中提取的患者基線信息，如年齡、性別、吸煙史、腫瘤分期等。

這張圖展示了整個多模態融合預測系統的主要模組與資料流。

![[ChatGPT Image 2025年9月28日 上午02_27_31 1.png]]
白板圖 #1：系統高層架構 (High-Level Architecture)
這張圖展示了整個多模態融合預測系統的主要模組與資料流。
![[Pasted image 20250928030106.png]]
#### **系統流程概述**

1. **多模態數據輸入**：系統接收三種核心數據：
    - **全切片病理圖像 (WSI)**：高解析度的 H&E 染色組織切片圖像。
    - **批量 RNA 測序 (Bulk RNA-seq) 數據**：每個樣本的基因表達計數矩陣。
    - **結構化臨床數據 (EHR)**：包括患者人口統計學、治療史、病理診斷等。
2. **數據預處理**：對每種模態的原始數據進行清洗、標準化和轉換，為特徵提取做準備。
3. **特徵提取**：這是關鍵一步，為每種模態設計專門的深度學習模型或傳統演算法，將原始數據轉換為具有生物學意義的、可輸入融合模型的向量特徵。
4. **多模態融合模型**：這是系統的核心。一個精心設計的深度學習架構會將來自不同模態的特徵整合在一起，學習它們之間的複雜關係，最終預測患者對 PD-1 抑制劑的響應（響應者 vs. 非響應者）。
5. **預測與可解釋性**：模型輸出患者的預測結果，並提供工具來解釋模型的決策，例如哪些圖像特徵或基因表達對預測貢獻最大。

白板圖 #2：深入探討 - 各模態的特徵提取 (Feature Extraction)
![[Pasted image 20250928030217.png]]
#### **各模態特徵提取細節**

1. **圖像特徵提取 (WSI)**：
    - **挑戰**：WSI 檔案巨大（幾個 GB），直接處理困難。
    - **策略**：
        - **分塊 (Tiling)**：將 WSI 分割成大量小塊 (tiles)。
        - **深度學習**：使用在大量病理圖像上預訓練的 CNN 模型（如 ResNet, Vision Transformer）來提取每個 tile 的特徵向量。
        - **多實例學習 (MIL)**：由於一個 WSI 由許多 tiles 組成，我們需要將這些 tile 特徵**聚合 (aggregate)** 成一個單一的 WSI 級別特徵向量。注意力機制 (Attention-based MIL) 是首選，因為它可以自動識別 WSI 中對預測最重要的區域。
    - **輸出**：一個代表整個 WSI 的固定維度向量。
2. **RNA-seq 特徵提取 (Bulk RNA-seq)**：
    - **挑戰**：數萬個基因，維度高，噪音大。
    - **策略**：
        - **生物學知識導向**：計算已知生物學通路的活性得分（例如，免疫檢查點通路、腫瘤微環境相關通路）。這提供了更具可解釋性的特徵。
        - **無監督降維**：使用自編碼器 (Autoencoder) 或 PCA 來學習基因表達數據的低維度隱藏表示，減少噪音並捕捉主要變異。
    - **輸出**：一個代表基因表達模式的固定維度向量。
        
3. **臨床數據特徵提取 (Clinical Data)**：
    - **挑戰**：數據異質性（數值、類別）、缺失值。
    - **策略**：
        - **標準化預處理**：處理缺失值、對類別變量進行獨熱編碼 (One-Hot Encoding) 或標籤編碼 (Label Encoding)，對數值變量進行標準化。
        - **簡單 MLP**：一個小型多層感知器 (MLP) 可以將這些處理過的臨床特徵進一步轉換成一個更抽象的向量。
    - **輸出**：一個代表臨床資訊的固定維度向量。


白板圖 #3：多模態融合架構與可解釋性 (Fusion Architecture & Interpretability)
這裡將展示選擇的融合策略和模型架構。我們將採用一種**中間融合 (Intermediate Fusion)** 策略，這通常在生物醫學數據中表現良好。
![[Pasted image 20250928030334.png]]
#### **多模態融合架構 (Intermediate Fusion)**

1. **模態專屬投影層 (Modality-Specific Projection Heads)**：
    - 每個模態的特徵向量首先會通過一個或多個全連接層 (Dense Layer)，將其投影到一個共享的潛在空間 (shared latent space) 中，使得它們的維度一致且具有更好的語義表示。例如，都投影到 256 維。
        
2. **拼接與注意力機制 (Concatenation & Attention)**：
    - 將這些投影後的模態特徵向量**拼接 (concatenate)** 起來，形成一個更長的向量。
    - 然後，將這個拼接後的向量輸入到一個或多個**多頭自注意力 (Multi-Head Self-Attention)** 層或**Transformer 編碼器塊 (Transformer Encoder Block)**。
    - **設計理念**：Transformer 架構在處理序列數據和捕捉長距離依賴關係方面表現出色。在這裡，它可以：
        - **學習模態間的交互作用**：例如，病理圖像中的免疫細胞浸潤模式可能與某些免疫相關基因的表達水平高度相關。注意力機制可以自動發現這些跨模態的深層聯繫。
        - **動態加權模態重要性**：模型可以根據輸入數據的內容，自適應地分配不同模態的權重。例如，對於某些患者，圖像特徵可能更關鍵；對於另一些，基因表達可能更具決定性。
3. **預測頭 (Predictive Head)**：
    - Transformer 的輸出會再通過一組全連接層 (Dense Layers)，最後是一個單輸出、帶 Sigmoid 激活函數的層，輸出一個 0 到 1 之間的機率值，表示患者響應 PD-1 抑制劑的可能性。
#### **可解釋性 (Interpretability)**

- **SHAP (SHapley Additive exPlanations)**：這是一種廣泛使用的模型不可知 (model-agnostic) 的可解釋性方法。它會為每個特徵計算一個 Shapley 值，表示該特徵對模型預測的貢獻程度。
    - **圖像**：可以追溯到最重要的圖像區域或 tiles。
    - **RNA-seq**：識別哪些基因或通路對預測最重要。
    - **臨床**：揭示哪些臨床變量具有最大的影響力。
- **注意力權重可視化**：如果融合模型使用了注意力機制，我們可以視覺化注意力權重，了解模型在做決策時，是「關注」了哪些模態或模態中的哪些部分。
- **LIME (Local Interpretable Model-agnostic Explanations)**：對於單個患者的預測，LIME 可以生成一個局部可解釋的模型，說明為什麼模型會做出當前的預測。

白板圖 #4：性能評估與挑戰 (Performance Evaluation & Challenges)
![[Pasted image 20250928030436.png]]
#### **性能評估 (Performance Evaluation)**

1. **評估指標**：
    - 對於**類別不平衡 (imbalanced dataset)** 的二元分類問題（響應者通常少於非響應者），僅看準確率 (Accuracy) 會產生誤導。更合適的指標包括：
        - **ROC AUC** (Receiver Operating Characteristic Area Under the Curve)：衡量模型區分兩類的能力。
        - **PR AUC** (Precision-Recall Area Under the Curve)：在高度不平衡數據集上比 ROC AUC 更具資訊量，特別關注正類 (Positive class) 的預測表現。
        - **F1-score**：精確度 (Precision) 和召回率 (Recall) 的調和平均，兼顧兩者。
        - **敏感度 (Sensitivity/Recall)** 和 **特異度 (Specificity)**：在醫學診斷中至關重要。
            
2. **評估策略**：
    - **分層 K-折交叉驗證 (Stratified K-Fold Cross-Validation)**：確保每個交叉驗證折疊 (fold) 中，響應者和非響應者的比例與原始數據集保持一致，避免因隨機分割導致某些折疊缺少某一類樣本。
    - **外部驗證 (External Validation)**：如果有可能，在一個完全獨立、來自不同醫院或研究的數據集上進行驗證，這是衡量模型泛化能力的最強方式。

#### **小規模與不平衡數據集的挑戰與解決方案**

這是一個在現實世界腫瘤研究中常見的重大挑戰。
1. **數據增強 (Data Augmentation)**：
    - **圖像**：對 WSI tiles 進行隨機旋轉、翻轉、顏色抖動等操作，增加數據多樣性。
    - **其他模態**：對於 RNA-seq 或臨床數據，可以使用 SMOTE (Synthetic Minority Over-sampling Technique) 或基於 VAE 的生成模型來合成少數類別的樣本。
2. **損失函數修改 (Loss Function Modification)**：
    - **加權交叉熵損失 (Weighted Cross-Entropy Loss)**：在計算損失時，給予少數類別更高的權重，迫使模型更關注它們的正確分類。
    - **Focal Loss**：降低易分類樣本的損失貢獻，讓模型專注於難分類的樣本。
3. **採樣技術 (Sampling Techniques)**：
    - **過採樣 (Oversampling)**：複製少數類樣本或使用 SMOTE 生成合成樣本。
    - **欠採樣 (Undersampling)**：隨機移除多數類樣本（需謹慎，可能丟失信息）。
4. **遷移學習 (Transfer Learning)**：
    - 充分利用在大型通用數據集（如 ImageNet）或生物醫學特有數據集上預訓練的模型（例如，用於 WSI 特徵提取的 ResNet 或 ViT），然後在我們的目標任務上進行微調。
5. **集成方法 (Ensemble Methods)**：
    - 訓練多個不同的模型，然後對它們的預測進行平均或投票，這可以提高模型的魯棒性和泛化能力。
6. **正則化 (Regularization)**：
    - 使用 Dropout、L1/L2 正則化等技術，防止模型在小數據集上過度擬合。
#### **系統操作性挑戰**

- **數據隱私與安全**：必須符合 HIPAA、GDPR 等法規，對患者數據進行嚴格匿名化和加密處理。
- **數據異質性**：來自不同來源的數據格式、質量和缺失值處理需要標準化的數據預處理管線
- **計算資源**：WSI 處理和深度學習模型訓練需要強大的 GPU 計算資源。
- **臨床工作流整合**：確保系統能無縫整合到現有的醫院或診斷流程中，提供易於使用的界面和及時的結果。



#### **系統流程概述**

**標籤定義：** 「響應者」是如何定義的？是基於RECIST標準（實體瘤療效評價標準）的客觀緩解率（ORR），還是無進展生存期（PFS）？
- **計算資源：** 訓練這樣一個複雜的多模態模型需要哪些計算資源（例如，多GPU服務器）？
- **臨床應用場景：** 這個模型的預期用途是什麼？是作為一個探索性研究工具，還是一個潛在的伴隨診斷工具？

**假設：**
- 數據集包含500名患者。
- 標籤是二元的（響應者/無響應者），存在類別不平衡（例如，30%的響應者）。
- 目標是開發一個研究工具，可解釋性與準確性同等重要。

**2. 系統架構與數據管道** 候選人應設計一個包含三個並行處理分支，並在最後進行融合的端到端管道。

- **1. 數據預處理管道：**
    - **臨床數據分支：**
        1. 加載結構化臨床數據表。
        2. 進行標準的預處理：處理缺失值（例如，中位數插補）、對分類變量進行獨熱編碼（one-hot encoding）、對數值變量進行標準化。
    - **RNA-seq數據分支：**
        1. 從FASTQ文件開始，運行一個標準的生物資訊學管道（例如，STAR進行比對，RSEM進行定量）來獲得基因表達計數矩陣。
        2. 進行質控和標準化（例如，TPM或log-TPM）。
        3. 由於基因數量龐大（~20,000），需要進行降維。可以使用生物學先驗知識選擇一個基因子集（例如，免疫相關通路基因），或者使用無監督方法如PCA。
    - **WSI影像數據分支：**
        1. 這是一個計算密集型管道 。  
        2. **切片 (Tiling)：** 將巨大的WSI（例如，10萬x10萬像素）分割成數千個小的、可管理的圖塊（patches），例如256x256像素。
        3. **圖塊篩選：** 剔除背景和無組織的圖塊。
        4. **特徵提取：** 這是關鍵。與其將所有圖塊直接輸入模型，不如先提取有意義的特徵。
            
- **2. 多模態融合模型架構** 這部分是回答的核心。候選人應繪製並解釋一個深度學習架構。一個優秀的設計是基於**中間融合 (Intermediate Fusion)** 或 **混合融合 (Hybrid Fusion)** 的策略 。  
    
    - **分支一：臨床數據編碼器 (Clinical Encoder)**
        - 一個簡單的多層感知機（MLP），接收預處理後的臨床特徵向量，輸出一箇中等維度的嵌入向量（例如，64維）。
    - **分支二：基因表達編碼器 (Gene Expression Encoder)**
        - 可以是另一個MLP，或者更複雜的架構如一維卷積神經網絡（1D-CNN），用於捕捉基因之間的局部模式。它接收基因表達向量，也輸出一箇中等維度的嵌入向量（例如，256維）。
    - **分支三：影像數據編碼器 (WSI Encoder)**
        - 這是一個更複雜的層級結構：
            1. **圖塊級別編碼器 (Patch-level Encoder)：** 使用一個在大型自然圖像數據集（如ImageNet）上預訓練好的CNN（例如，ResNet-50），並可選擇在大型病理影像數據集（如TCGA）上進行微調。這個CNN為每個圖塊生成一個特徵向量（例如，2048維）。
            2. **患者級別聚合器 (Patient-level Aggregator)：** 一個患者有數千個圖塊的特徵向量。需要一個機制來將它們聚合成一個代表整個WSI的單一向量。**基於注意力機制的聚合（Attention-based Aggregation）** 是一個很好的選擇。該模型可以學習為不同的圖塊分配不同的「重要性」（attention score），例如，自動地更關注那些包含腫瘤浸潤淋巴細胞的區域。最終輸出一個患者級別的影像嵌入向量（例如，512維）。
    - **融合層 (Fusion Layer)：**
        1. 將來自三個分支的嵌入向量（64維、256維、512維）**拼接 (concatenate)** 在一起。
        2. 將拼接後的向量輸入到一個或多個全連接層（MLP）中。
        3. **高級融合策略：** 可以使用更複雜的融合機制，如**Tensor Fusion**或**跨模態注意力 (Cross-modal Attention)**，讓不同模態的特徵在融合時能夠相互交互和調節。
    - **輸出層 (Output Layer)：**
        - 一個帶有Sigmoid激活函數的單一神經元，輸出患者為「響應者」的概率。
        - **損失函數：** 使用帶有類別權重的二元交叉熵損失（weighted binary cross-entropy）來處理類別不平衡問題。
            
- **3. 可解釋性與性能評估**
    - **可解釋性：**
        - **臨床和基因特徵：** 使用SHAP或Integrated Gradients來計算每個臨床變量和基因對最終預測的貢獻度。
        - **影像特徵：** 可視化**注意力圖**。將WSI聚合器學到的高注意力分數的圖塊在原始WSI上高亮顯示出來。這可以直觀地告訴病理學家，模型主要關注了組織的哪些區域來做出判斷。
    - **性能評估：**
        - 使用**分層交叉驗證 (stratified cross-validation)** 來獲得穩健的性能估計。
        - 主要評估指標應為**AUC-ROC**和**AUC-PR**（在不平衡數據集上後者更具信息量）。同時報告精確度、召回率和F1分數。

這個回答展示了候選人設計複雜深度學習系統的專業能力，能夠為不同的數據模態選擇合適的架構，並通過先進的融合和可解釋性技術，從多維度數據中提取有臨床意義的洞見。




### 問題 13：為高通量篩選數據設計批次效應校正系統

**Question 13: Design a Batch Effect Correction System for High-Throughput Screening Data**

- **Problem Statement:** Large ==High-Throughput Screening (HTS)== campaigns are run over weeks or months, introducing strong, non-biological systemic variations known as "batch effects" (e.g., from different plates, dates, operators). If uncorrected, these effects can severely distort the data, leading to high false-positive or false-negative rates. **問題陳述：** 高通量篩選（High-Throughput Screening, HTS）是藥物發現的基石，它允許在一次實驗中測試數千至數百萬個化合物。一個大型的HTS項目通常會持續數週或數月，使用多塊微孔板（microplate）、多台不同的讀板儀，甚至由不同的操作員在不同的日期進行。這些實驗條件的變化會引入強烈的、非生物學的系統性變異，即「批次效應」（Batch Effects）。如果不加以校正，批次效應會嚴重扭曲數據，導致極高的假陽性或假陰性率，從而使後續的hit-picking決策產生嚴重偏差。您的任務是設計一個自動化的系統，用於檢測、可視化和校正HTS數據中的批次效應，同時確保不移除真實的生物學信號。
    
- **System Requirements:**
    
    - **Data & Metadata Integration:** Import HTS raw readouts along with rich metadata (Plate ID, Run Date, Plate Layout, Control Well Type).
     **數據與元數據整合：** 系統必須能夠導入HTS的原始讀數（例如，熒光強度、吸光度）以及與之相關的豐富元數據（Plate ID, Run Date, Operator ID, Plate Layout, Control Well Type等）
     
    - **Automated Detection & Visualization:** A pipeline to automatically detect and visualize the presence and strength of batch effects (e.g., using PCA plots).
     **自動化檢測與可視化：** 管道應自動執行統計檢測和生成可視化圖表（如PCA圖、箱線圖），以診斷批次效應的存在和強度。
        
    - **Robust Correction Algorithm:** Implement or integrate one or more batch effect correction algorithms (e.g., ComBat, linear mixed models).
     **穩健的校正算法：** 實現或集成一種或多種批次效應校正算法（例如，ComBat, 線性混合模型）
        
    - **Quality Control & Signal Protection:** Include validation steps to ensure the correction process does not remove the signal from known positive and negative controls.
     **質量控制與信號保護：** 系統必須包含驗證步驟，以確保校正過程沒有過度擬合或移除已知的陽性和陰性對照品（controls）的信號
        
    - **Scalable Pipeline:** The workflow must be scalable to handle projects with thousands of microplates.
     **可擴展管道：** 整個流程應被構建成一個可擴展的管道，能夠處理包含數千塊微孔板的大型項目
     
- **Task:** Describe the system architecture and workflow, from data import to corrected data generation. Focus on how you would evaluate the effectiveness of the correction and avoid over-correction, especially in the presence of confounding variables. 請闡述您的系統架構，描述從數據導入到生成校正後數據的完整工作流，並重點討論您將如何評估校正的有效性以及避免過度校正。

![[ChatGPT Image 2025年9月28日 上午02_31_25.png]]
Whiteboard #1: High-Level System Architecture & Workflow

This diagram outlines the complete end-to-end workflow of the system, from raw data input to validated, corrected data output.
![[Pasted image 20250928031511.png]]
#### **System Workflow Overview**

1. **Data Integration**: The system starts by ingesting raw numerical data from HTS instruments and combining it with crucial metadata that describes the experiment's structure (e.g., which plate a well is on, the date it was run).
    
2. **Detection & Visualization**: Before any correction, the system automatically visualizes the data's structure. If technical variables (like the plate ID) are the main drivers of variation, batch effects are present.
    
3. **Correction Engine**: A selected algorithm is applied to mathematically adjust the data, aiming to remove the variation associated with the batch variables while preserving the variation from the actual treatments.
    
4. **Validation (The Critical Step)**: This module acts as a safety check. It quantifies both the removal of batch effects and, more importantly, the preservation of the known biological signal from control wells. This prevents "over-correction."
    
5. **Output**: Once validated, the system generates the clean, corrected data and a report that documents the entire process, providing evidence of the correction's success.



Whiteboard #2: Automated Detection & Visualization
This diagram illustrates how the system provides visual proof of batch effects, which is crucial for diagnosing the problem before attempting to fix it.
![[Pasted image 20250928031604.png]]
###### **How Detection Works**

The core idea is simple: in a perfect experiment, the main reason data points differ should be the biological effect of the treatments. By plotting the data using PCA, we can see the dominant sources of variation. If the plot shows that wells from the **same plate** are closer to each other than wells with the **same treatment** from different plates, we have a batch effect. The system automates the generation of these plots for key batch variables (Plate ID, Run Date, etc.) to provide a clear, visual diagnosis.


Whiteboard #3: The Correction Engine
This diagram focuses on the algorithms that perform the actual data correction.
![[Pasted image 20250928031654.png]]
###### **Choosing an Algorithm**

- **ComBat** is often the best starting point. It's specifically designed for batch correction in genomics and has been widely adopted for HTS data due to its robustness.
    
- **Linear Mixed Models (LMMs)** are more powerful and flexible but require more computational resources and careful model specification. They are excellent when the experimental design is complex.
    
- **Simpler regression-based methods** can be used for quick corrections or when batch effects are less severe.
    

The system should allow the user to select the method or even run multiple methods and compare their results in the validation stage.


Whiteboard #4: The Quality Control & Validation Module
This is the most important module for ensuring the correction is effective and not harmful. It answers the question: "Did we fix the technical problem without creating a biological one?"
![[Pasted image 20250928031744.png]]
###### **How Validation Prevents Over-Correction**

Over-correction happens when an algorithm is too aggressive and removes not only the technical noise but also some of the true biological signal. The **Z-Factor** is our safeguard. It directly measures the separation between positive and negative controls. If the correction algorithm mistakenly makes the positive controls look more like the negative controls, the Z-Factor will decrease, immediately signaling a problem. This two-pronged approach—verifying that the _bad_ (batch) variance is gone while the _good_ (biological) variance remains—is the key to a robust system.

###### **Scalability Considerations**

- **Data Handling**: Use memory-efficient data formats like **Parquet** or **HDF5** (which backs the `AnnData` format common in biology) to handle data from thousands of plates without exhausting system memory.
    
- **Parallel Computing**: Implement data processing steps using frameworks like **Dask** or **Apache Spark**. This allows the workload (e.g., reading files, calculating metrics per plate) to be distributed across multiple CPU cores or even multiple machines in a cluster.
    
- **Workflow Orchestration**: The entire pipeline should be codified using a workflow manager like **Nextflow** or **Snakemake**. This makes the process reproducible, scalable, and easy to deploy on high-performance computing (HPC) clusters or cloud environments.




---

**中文解說：**

這個問題考察候選人對統計學和數據預處理中一個經典但極其重要問題的深刻理解。一個優秀的回答不僅僅是知道一個算法的名字（如ComBat），而是能夠設計一個包含診斷、校正和驗證的完整、穩健的工作流，並理解其背後的統計假設和潛在風險 。  

**1. 問題解析與範疇界定 (Problem Clarification & Scoping)**
- **HTS實驗設計：** 實驗設計是怎樣的？對照品（陽性/陰性/中性）是如何分佈在板上的？是否存在跨板的重複測量？
- **批次效應的來源：** 主要的批次效應來源已知嗎？（例如，板間差異、日期差異）。
- **數據分佈：** 原始數據的統計分佈是什麼？（例如，近似正態分佈，還是計數數據）。
- **下游應用：** 校正後的數據將用於什麼？（例如，計算Z-score來挑選hit，還是訓練一個QSAR模型）。
**假設：**
- 每塊板都包含標準的陽性和陰性對照。
- 主要的批次效應來自板間差異（plate-to-plate variation）。
- 下游應用是計算Z-score來識別活性化合物。

**2. 系統架構與工作流** 候選人應設計一個由多個獨立但有序的步驟組成的管道，由Airflow或類似工具編排。
- **1. 數據導入與整合模塊 (Data Ingestion & Integration Module)：**
    - 從LIMS（實驗室信息管理系統）或文件中讀取原始讀數和所有相關的元數據。
    - 將數據整合成一個單一的、整潔的數據框（tidy dataframe），每一行代表一個孔（well），列包括：`Plate_ID`, `Well_Position`, `Compound_ID`, `Raw_Value`, `Well_Type` (e.g., `positive_control`, `negative_control`, `sample`) 等。
- **2. 診斷與可視化模塊 (Diagnostics & Visualization Module)：**
    - **目標：** 在進行任何校正之前，首先要確認並理解批次效應。
    - **方法：**
        1. **主成分分析 (PCA)：** 對原始數據進行PCA，然後用批次變量（如`Plate_ID`）對PCA圖上的點進行著色。如果來自同一塊板的點聚集在一起，形成明顯的簇，這就是批次效應的強烈證據。
        2. **對照品質控：** 為每塊板繪製陽性和陰性對照的信號分佈箱線圖。理想情況下，不同板上的對照信號應該是穩定和一致的。如果存在顯著差異，則表明存在批次效應。
        3. **統計檢驗：** 使用ANOVA或線性模型來量化批次變量對觀測值的貢獻有多大。
    - **輸出：** 生成一份HTML格式的診斷報告，包含上述圖表和統計結果，供科學家審閱。
- **3. 批次效應校正模塊 (Batch Correction Module)：**
    - **輸入：** 原始數據和已識別的批次變量。
    - **算法選擇：**
        - **ComBat：** 這是一個非常流行且有效的經驗貝葉斯方法，能夠同時調整批次的均值和方差。它通過跨基因（或此處的化合物）借用信息，在小樣本批次中表現穩健 。在應用ComBat時，如果存在已知的生物學變量（例如，化合物濃度），應將其作為協變量納入模型，以保護這些生物學信號不被錯誤地當作批次效應移除。  
        - **線性混合效應模型 (LMM)：** 可以將批次變量作為隨機效應（random effect）納入模型 `Value ~ Biological_Factors + (1 | Batch)`。然後，校正後的數據可以被認為是模型的殘差加上固定效應的預測。LMM提供了更大的靈活性來處理複雜的實驗設計。
    - **實現：** 將算法封裝成一個獨立的、可配置的函數或類。
- **4. 驗證與質量控制模塊 (Validation & QC Module)：**
    - **目標：** 確保校正過程是有效的，並且沒有引入新的偏差或消除真實信號。
    - **方法：**
        1. **重複診斷分析：** 在校正後的數據上重新運行診斷模塊（例如，PCA圖）。理想情況下，之前由批次變量驅動的聚類現象應該消失了。
        2. **對照品信號保護評估：** 比較校正前後，陽性對照和陰性對照之間的分離程度。通常使用**Z'因子 (Z-prime factor)** 這個指標來衡量HTS實驗的質量。一個好的校正應該能維持甚至提高Z'因子，而不是降低它。如果Z'因子顯著下降，說明校正過程可能過度，移除了部分真實的生物學信號。
        3. **信號-噪音分析：** 檢查校正是否僅僅是將所有數據拉向全局平均值，還是保留了化合物之間的相對差異。
    - **輸出：** 生成一份驗證報告，與診斷報告一起呈現，以供最終決策。
**5. 潛在風險與應對**
- **生物學與批次的混淆 (Confounding)：** 這是最危險的陷阱。如果實驗設計不當，例如，所有A類化合物都在第一天測試，所有B類化合物都在第二天測試，那麼化合物類型和測試日期（批次）就完全混淆了。在這種情況下，任何試圖移除日期效應的算法都可能同時移除掉A類和B類化合物之間的真實生物學差異 。  
    - **應對策略：**
        1. **設計時預防：** 最好的策略是在實驗設計階段就通過**隨機化**來避免混淆。
        2. **系統性檢測：** 在校正前，系統應自動計算生物學變量（如化合物類別）和批次變量之間的關聯性（如卡方檢驗）。如果發現強關聯，應向用戶發出嚴重警告。
        3. **謹慎校正：** 在存在混淆的情況下，需要使用更高級的、能夠區分協變量的方法，或者與統計學家合作，決定是否可以進行校正，以及如何進行校正。

這個回答展示了候選人不僅僅是一個算法的使用者，而是一個能夠設計一個包含自我診斷和安全檢查的、嚴謹的科學數據處理系統的專家。這體現了在處理真實世界 messy data 時所必需的審慎和批判性思維。



### 問題 14：為高內涵成像設計表型分析的ML系統

**Question 14: Design an ML System for Phenotypic Profiling of High-Content Imaging Data**

- **Problem Statement:** High-Content Screening (HCS) captures complex morphological changes in cells after compound treatment using automated microscopy. These images contain rich information that can be used to identify a drug's Mechanism of Action (MoA).
  **問題陳述：** 高內涵篩選（High-Content Screening, HCS）或高內涵成像（HCI）是一種強大的表型藥物發現技術。在HCS實驗中，細胞被不同的化合物處理後，通過自動化顯微鏡拍攝多通道熒光圖像，以捕捉藥物引起的複雜形態學變化。這些圖像中蘊含著比單一活性讀數豐富得多的信息，可以用於識別藥物的作用機制（Mechanism of Action, MoA）、發現新的表型特徵等。您的任務是設計一個端到端的機器學習系統，用於對HCS圖像數據進行大規模的表型分析。

- **System Requirements:**
    
    - **Image Processing Pipeline:** A scalable pipeline to process millions of multi-channel microscopy images, including segmentation to identify individual cells.
     **圖像處理管道：** 設計一個可擴展的管道，用於處理數百萬張多通道顯微圖像。核心步驟包括：圖像質量控制、圖像分割（識別單個細胞）、以及單細胞特徵提取。
     
    - **Feature Extraction Strategy:** Support both: **Classical Features:** Extracting hundreds of predefined morphological features using tools like CellProfiler. **Deep Learning Features:** Using self-supervised learning models to extract high-dimensional, data-driven feature embeddings from cell images.
     **特徵提取策略：** 系統應支持兩種特徵提取方法：1. **傳統特徵：** 使用像 **CellProfiler** 這樣的工具提取數百個預定義的形態學、強度和紋理特徵。2. **深度學習特徵：** 使用深度學習模型（特別是自監督學習模型）從細胞圖像中提取高維的、數據驅動的特徵嵌入。

    - **Phenotypic Clustering & Analysis:** Cluster compounds that induce similar morphological changes to hypothesize a shared MoA.
     **表型聚類與分析：** 系統需要對化合物進行聚類，將那些引起相似細胞形態變化的化合物分在同一組。這有助於假設它們具有相似的MoA。
     
    - **Visualization & Interaction:** An interactive interface for biologists to browse clustering results, view representative cell images, and explore phenotypic features.
     **可視化與交互：** 提供一個交互式界面，讓生物學家可以瀏覽聚類結果、查看代表性的細胞圖像、以及探索不同表型特徵的意義。
     
- **Task:** Detail your system architecture, compare the pros and cons of classical vs. deep learning features, and describe how you would implement phenotypic clustering and result visualization.
  請闡述您的系統架構，比較傳統特徵與深度學習特徵的優劣，並描述您將如何實現化合物的表型聚類和結果的可視化。
![[ChatGPT Image 2025年9月28日 上午02_40_15.png]]


白板圖 #1：系統高層架構 (High-Level Architecture)
這張圖展示了整個高通量影像表型分析系統的主要模組與資料流。
![[Pasted image 20250928025142.png]]
#### **系統流程概述**

1. **原始高通量影像數據**：系統的輸入是數百萬張多通道顯微鏡圖像，這些圖像記錄了細胞在不同化合物處理後的形態變化。同時包含化合物、孔板等元數據。
2. **可擴展影像處理管線**：這個模組負責處理原始影像。核心任務是**細胞分割 (Cell Segmentation)**，精確識別出圖像中的每個獨立細胞，並從中提取單細胞圖像。
3. **特徵提取引擎**：這是系統的核心創新點。它支援兩種互補的特徵提取方法：
    - **經典特徵 (Classical Features)**：使用 CellProfiler 等工具計算數百個預定義的形態學和螢光強度特徵。
    - **深度學習特徵 (Deep Learning Features)**：使用自監督學習 (Self-supervised learning) 模型，從單細胞圖像中提取高維度、數據驅動的嵌入特徵。
4. **特徵聚合 (Feature Aggregation)**：由於我們通常對化合物的整體效應感興趣，而非單個細胞，因此需要將單細胞特徵聚合成孔板 (well) 或化合物級別的表型向量。
5. **表型聚類與分析 (Phenotypic Clustering & Analysis)**：對聚合後的化合物表型向量進行降維和聚類，以識別出具有相似形態變化的化合物組，這些組可能具有共同的作用機制 (MoA)。
6. **視覺化與互動模組**：提供一個互動式界面，讓生物學家可以瀏覽聚類結果，查看代表性的細胞圖像，並深入探索每個群集的特徵。


白板圖 #2：深入探討 - 影像處理管線 (Image Processing Pipeline)
![[Pasted image 20250928025222.png]]
#### **關鍵步驟說明**

1. **影像預處理與 QC**：
    - **平場校正 (Flat-field Correction)**：移除顯微鏡光學系統造成的不均勻照明。
    - **背景減除/去噪**：增強信噪比，使細胞目標更清晰。
    - **強度標準化**：確保不同批次或孔板間的圖像強度具有可比性。
    - **品質控制**：自動檢測並過濾掉模糊、低細胞數量或帶有顯著偽影的圖像。
2. **細胞分割 (Cell Segmentation)**：
    - **目標**：精確地識別出圖像中的每個細胞及其亞細胞結構（如細胞核、細胞質）。這是後續特徵提取的基礎。
    - **方法選擇**：
        - **深度學習方法 (例如 Cellpose, U-Net)**：這是當前最先進且魯棒的方法，尤其適用於複雜和多樣的細胞形態。Cellpose 因其廣泛的預訓練和易用性而成為一個很好的起點。
        - **傳統影像處理 (例如 Watershed 演算法)**：對於形態較簡單、對比度高的細胞，也可以使用傳統方法，但通常對參數和影像品質更敏感。
    - **挑戰**：細胞重疊、染色不均勻、細胞形態多樣性等。
3. **單細胞圖像提取**：
    - 根據分割結果，從原始多通道圖像中裁剪出每個單獨細胞的圖像塊。
    - 將這些細胞圖像調整為統一尺寸（例如 64x64 像素），並與其元數據（所屬化合物、孔板、原始圖像 ID、細胞 ID）綁定，為下一步的特徵提取做準備。
#### **可擴展性考量**

- **分散式檔案系統/對象儲存**：原始影像檔案通常非常大，需要部署在如 HDFS、AWS S3 或 Azure Blob Storage 等分散式儲存系統上。
- **平行處理框架**：整個影像處理管線需要支持平行化。可以利用 `Dask`、`Apache Spark` 或 `Ray` 等框架在多個 CPU/GPU 節點上同時處理數百萬張圖像。
- **容器化**：將影像處理工具（如 Cellpose）和自定義腳本打包成 Docker 容器，確保在不同計算環境下的一致性和可重複性。


白板圖 #3：深入探討 - 特徵提取引擎與比較
![[Pasted image 20250928025336.png]]
#### **經典特徵 vs. 深度學習特徵**

|特徵類型|優點 (Pros)|缺點 (Cons)|建議應用|
|---|---|---|---|
|**經典特徵 (CellProfiler)**|- **可解釋性強**：每個特徵都有明確的生物學意義。- **計算效率高**：對於大量數據，處理速度相對快。- **無需大量標註數據**：依賴預定義的規則。- **工業標準**：CellProfiler 是成熟且廣泛使用的工具。|- **特徵空間固定**：只能捕獲預設的形態學特徵，可能錯過潛在的、更複雜的表型。- **對圖像品質敏感**：分割或預處理的微小誤差可能導致特徵偏差。- **需要專家知識**：設計有效的 CellProfiler 管線需要了解生物學和圖像處理。|作為**基線比較**；在**對可解釋性要求極高**，或**數據量相對較小**時。|
|**深度學習特徵 (自監督學習)**|- **數據驅動，特徵更豐富**：能從像素中學習到人眼難以察覺的高級、抽象、判別性特徵。- **泛化能力強**：SSL 模型在大量無標籤數據上學習後，可以很好地泛化到新數據。- **減少人工干預**：無需手動定義特徵或複雜的規則。- **對噪音更魯棒**：能更好地處理圖像變化。|- **難以直接解釋**：特徵向量通常是高維抽象的，其各個維度沒有直接的生物學意義。- **計算成本高**：模型訓練需要大量 GPU 資源和時間。- **模型複雜度高**：需要深度學習專業知識來設計和訓練。|作為**核心特徵提取**；在**MoA 複雜、需要捕捉細微形態變化**，且**具有大量圖像數據**時。|

**決策**：在實際系統中，強烈建議**同時支持兩種特徵提取方式**，並允許用戶選擇或結合使用。深度學習特徵通常能提供更好的聚類效果，而經典特徵能提供可解釋的佐證。
##### **自監督學習 (Self-supervised Learning, SSL) 的理由**

- **HCS 數據量大但缺乏標註**：數百萬張細胞圖像，但 MoA 標籤很少。SSL 可以在無標籤數據上進行預訓練，學習有用的特徵表示。
- **捕獲細胞表型細微差異**：SSL 模型可以學習到圖像中細微的形態、紋理和強度模式，這些是定義細胞表型的關鍵。

白板圖 #4：深入探討 - 特徵聚合、表型聚類與視覺化
![[Pasted image 20250928030943.png]]
#### **特徵聚合 (Feature Aggregation)**

- **問題**：每個化合物處理的孔板中含有數百到數千個細胞。我們需要將這些單細胞特徵聚合成一個代表該化合物在該孔板中總體效應的單一向量。
- **方法**：最常見且魯棒的方法是計算每個特徵的**中位數 (Median)**。中位數對異常值（例如，分割錯誤的細胞或死亡細胞）的敏感度較低。也可以計算平均值、標準差或更穩健的統計量（如截斷均值）。
- **輸出**：一個「化合物級別」的表型向量，代表該化合物處理後的典型細胞形態特徵。

#### **表型聚類與分析 (Phenotypic Clustering & Analysis)**

1. **降維 (Dimensionality Reduction)**：
    - **UMAP** 是首選。它能有效地將高維度的表型向量映射到 2D 或 3D 空間，同時保留數據的局部和全局結構，非常適合視覺化和聚類。
    - **t-SNE** 也是一個選項，但在大型數據集上通常較慢。
    - **PCA** 可以用於初步降維或理解主要變異方向。
2. **聚類演算法 (Clustering Algorithms)**：
    - **HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)**：一個強大的選擇。它不需要預先指定聚類數量 'k'，能發現任意形狀的聚類，並且對噪音有很好的處理能力。這非常適合探索性的表型分析。
    - **K-Means**：簡單快速，但需要手動指定聚類數量。
    - **高斯混合模型 (GMM)**：提供每個樣本屬於每個聚類的機率，更具彈性。
3. **MoA 假設生成**：
    - 聚類完成後，研究人員可以檢查每個聚類中的化合物。如果一個聚類中的大多數化合物已知具有相同的 MoA，那麼該聚類中的其他未知 MoA 的化合物就可能具有類似的 MoA。
    - 將聚類結果與化合物資料庫（如 ChEMBL、DrugBank）交叉比對，找出已知作用機制，以自動化方式推斷新假設。
#### **視覺化與互動模組 (Visualization & Interaction Module)**

- **技術棧**：推薦使用 Python 的網絡應用框架，如 **Plotly Dash** 或 **Streamlit**，它們可以快速構建互動式儀表板。
- **關鍵視覺化**：
    - **2D/3D 嵌入圖**：互動式的 UMAP/t-SNE 散點圖，每個點代表一個化合物。用戶可以根據聚類結果、已知 MoA 或特定特徵值對點進行顏色編碼，並在懸停時顯示化合物詳細信息。
    - **代表性細胞圖像瀏覽器**：對於選定的聚類或化合物，顯示一系列來自該組的單細胞圖像，讓生物學家直觀地感受其形態特徵。
    - **表型特徵圖**：柱狀圖、小提琴圖或熱圖，展示不同聚類在關鍵經典特徵（例如，細胞核大小、細胞質強度）上的分佈差異，提供量化的解釋。
    - **化合物信息整合**：連接到外部化合物資料庫，為每個化合物提供額外的結構、靶點、文獻等信息。

- **用戶互動**：提供篩選、選擇、放大、導出等功能，讓生物學家能夠自由探索數據和結果。



#### **系統流程概述**

---

**中文解說：**

這個問題考察候選人在大規模計算機視覺、特徵工程（特別是自監督學習）和非監督學習方面的綜合能力。一個優秀的回答會展示如何構建一個從原始像素到可操作生物學洞見的完整分析平台 。

**1. 問題解析與範疇界定 (Problem Clarification & Scoping)**
- **圖像規模與格式：** 每個實驗產生多少圖像？圖像的分辨率和通道數是多少？
- **細胞類型：** 實驗中使用的細胞類型是什麼？這對圖像分割算法的選擇有影響。
- **對照品設計：** 實驗中是否包含已知MoA的參照化合物？這可以用於驗證聚類結果。
- **性能要求：** 圖像處理管道的吞吐量要求是多少？
**假設：**
- 一個實驗包含數千個孔，每個孔有多個視場，總計~100萬張5通道的1024x1024圖像。
- 使用常見的癌細胞系，細胞核和細胞質邊界相對清晰。
- 包含多個已知MoA的參照化合物。

**2. 系統架構設計** 候選人應設計一個基於雲的、並行化的處理和分析管道。
- **1. 圖像處理與分割管道 (Image Processing & Segmentation Pipeline)：**
    - **平台：** 使用AWS Batch或Kubernetes，每個任務處理單張或一組圖像。
    - **步驟：**
        1. **預處理：** 圖像校正（如平場校正）、去噪。
        2. **細胞核分割：** 使用一個預訓練好的深度學習模型，如 **StarDist** 或 **Cellpose**，它們在各種細胞核分割任務上表現出色。
        3. **細胞質分割：** 以細胞核為種子，向外擴張直到遇到相鄰細胞的邊界，從而確定每個細胞的輪廓。
    - **輸出：** 為每張圖像生成一個「分割掩碼」（segmentation mask），其中每個細胞被賦予一個唯一的ID。
- **2. 特徵提取模塊 (Feature Extraction Module)：**
    - 這是一個可以並行執行的模塊，輸入是原始圖像和對應的分割掩碼。
    - **分支A：傳統特徵提取**
        - **工具：** 使用 **CellProfiler**。它可以被腳本化並在無頭模式下運行。
        - **流程：** 對於每個被分割出的細胞，CellProfiler計算數百個特徵，如大小、形狀（橢圓度、緻密度）、DNA染色的強度和紋理（Gabor濾波器、Haralick紋理特徵）等。
        - **輸出：** 一個巨大的CSV或Parquet文件，每一行是一個細胞，每一列是一個特徵。
    - **分支B：深度學習特徵提取**
        - **模型：** 使用一個在大量生物圖像數據上通過**自監督學習 (Self-Supervised Learning, SSL)** 預訓練的卷積神經網絡（CNN），例如，使用對比學習（如SimCLR）或掩碼自編碼器（如MAE）的方法。SSL的優勢在於它不需要手動標註，可以從海量無標籤的細胞圖像中學習到豐富的視覺表示。
        - **流程：** 對於每個細胞，從原始圖像中裁剪出其圖像塊（patch），輸入到預訓練的SSL模型中，並提取中間層的激活值作為該細胞的特徵嵌入（embedding）。
        - **輸出：** 一個高維的嵌入矩陣，每一行是一個細胞的嵌入向量（例如，512維）。
- **3. 數據聚合與歸一化模塊 (Data Aggregation & Normalization Module)：**
    - **聚合：** HCS分析的單位通常是「孔」（well）或「化合物」。需要將單細胞級別的特徵聚合成孔級別的「表型譜」。這可以通過計算每個孔內所有細胞特徵的均值、中位數、方差等統計量來實現。
    - **歸一化：** 使用板上的陰性對照（例如，用DMSO溶劑處理的孔）來對每個特徵進行歸一化（例如，計算Robust Z-score），以消除板間差異。
- **4. 表型聚類與分析模塊 (Phenotypic Clustering & Analysis Module)：**
    - **輸入：** 每個化合物一個表型譜向量。
    - **降維：** 由於特徵維度很高，首先使用**UMAP**或**t-SNE**等非線性降維技術將數據投影到二維或三維空間以便可視化。
    - **聚類：** 在降維後的空間或原始高維空間中，使用密度聚類算法如**HDBSCAN**來識別化合物的簇。HDBSCAN的優點是它不需要預先指定簇的數量，並且可以將噪聲點識別為離群值。
    - **結果分析：**
        - 檢查已知MoA的參照化合物是否聚集在同一個簇中，以此來驗證聚類的生物學意義。
        - 對於每個簇，分析其表型譜的共同特徵（例如，「簇3的化合物都傾向於導致細胞核變大且紋理變粗糙」），從而為該簇賦予一個表型標籤。

**3. 傳統特徵 vs. 深度學習特徵** 候選人應能清晰地闡述兩者的優劣：
- **傳統特徵 (CellProfiler)：**
    - **優點：** 可解釋性強（每個特徵都有明確的物理或幾何意義）、計算成本相對較低、技術成熟。
    - **缺點：** 特徵是手動設計的，可能無法捕捉到所有細微的、非預期的表型變化。容易受到分割質量的影響。
- **深度學習特徵 (SSL Embeddings)：**
    - **優點：** 數據驅動，能夠自動學習到最具有區分度的特徵，無需人工設計。通常能捕捉到更細膩、更複雜的表型，性能更強大。
    - **缺點：** 可解釋性較差（嵌入向量的每個維度沒有直接的生物學意義）、需要大量的無標籤數據進行預訓練、計算成本高。
**結論：** 一個理想的系統應該同時提供這兩種特徵，讓生物學家可以從互補的角度來分析數據。

**4. 可視化界面**
- 一個交互式的儀表板，左側是化合物的UMAP散點圖，每個點代表一個化合物，按簇著色。
- 當用戶點擊一個點或一個簇時，右側會顯示：
    - 該化合物/簇的詳細信息。
    - 其平均表型譜（與對照組相比的特徵變化）。
    - 來自該孔的代表性細胞圖像，可以並排比較處理組和對照組的圖像。

這個回答展示了候選人具備設計一個複雜的、端到端的生物圖像分析平台的能力，涵蓋了從大規模圖像處理、現代特徵提取技術到非監督學習和交互式可視化的所有關鍵環




### **Q15: 利用大型語言模型融合非結構化與結構化數據進行靶點識別**

**Question 15: Fusing Unstructured and Structured Data for Target Identification Using Large Language Models**

- **Problem Statement:** Drug target identification relies on synthesizing evidence from disparate sources. Unstructured text from scientific literature (e.g., PubMed) contains novel hypotheses, while structured data from knowledge graphs (e.g., ChEMBL, Gene Ontology) provides established relationships. Fusing these sources is a major challenge.
  **目標**：設計一個系統，能整合來自**非結構化數據**（如科學文獻、專利、臨床試驗報告）和**結構化數據**（如基因組學數據庫、蛋白質交互網絡、化合物庫）的資訊，以自動化地識別和排序潛在的藥物靶點 (Drug Target)。

- **System Requirements:**
    
    - **Unified Representation:** Design a system that leverages a Large Language Model (LLM) to create a unified representation space for entities (genes, diseases, compounds) from both text and knowledge graphs.
        
    - **Knowledge-Infused LLM:** Propose a method to "inject" structured knowledge from the graph into the LLM to make its text processing context-aware.
        
    - **Hypothesis Generation:** Use the fused representation to identify and score novel potential drug targets for a given disease (e.g., by framing it as a link prediction or question-answering task).
        
    - **Evidence Triangulation:** For any generated hypothesis, the system must retrieve and present supporting evidence from both the original text sources and the knowledge graph paths.
        
- **Task:** Describe the architecture for this multimodal fusion system. Explain how you would represent and combine the two data types, how the LLM would be used for reasoning, and how you would ensure the generated hypotheses are explainable and backed by evidence.

![[ChatGPT Image 2025年9月28日 上午02_45_52.png]]

白板圖 #1：系統高層架構 (High-Level Architecture)
這張圖展示了整個系統的宏觀設計，分為**離線數據處理 (Offline Data Processing)** 和**線上查詢推理 (Online Query & Reasoning)** 兩個階段。

![[Pasted image 20250928024857.png]]
#### **系統流程概述**

1. **離線數據處理 (Offline)**：
    - 系統定期地從 **PubMed** 等來源攝取非結構化文本，並從 **ChEMBL**、**Gene Ontology (GO)** 等知識庫攝取結構化圖譜數據。
    - 這兩類數據經過各自的處理管線，被轉換成機器可讀的索引格式。文本被轉換為**向量嵌入 (Embeddings)** 存入**向量資料庫**；圖譜數據本身存入**圖資料庫**，同時其節點（實體）也被轉換為向量嵌入，與文本嵌入存儲在一起，形成**統一的表示空間**。
2. **線上查詢推理 (Online)**：
    - 研究人員通過介面提出一個開放式問題（例如，「阿茲海默症有哪些新的藥物靶點？」）。
    - **知識增強的 LLM 核心** 接收到查詢，首先對查詢進行分解和擴展，然後同時從向量資料庫（檢索相關文獻片段）和圖資料庫（檢索相關的實體和路徑）中**檢索 (Retrieve)** 相關信息。
    - 檢索到的「雙源」上下文被組裝成一個豐富的提示 (Prompt)，發送給 LLM 進行**推理和生成 (Reasoning & Generation)**。
    - LLM 生成一個帶有排序和評分的假設列表。
    - 最後，系統為每個假設附上來自文本和知識圖譜的**支持證據**，呈現在一個可解釋的用戶介面中，實現**證據三角測量 (Evidence Triangulation)**。

白板圖 #2：深入探討 - 統一表示與知識注入 (Unified Representation & Knowledge Infusion)
這是系統的核心創新，解釋了如何讓 LLM 同時「理解」文本和圖譜。我們主要採用**檢索增強生成 (Retrieval-Augmented Generation, RAG)** 的方法來實現知識注入。
![[Pasted image 20250928031203.png]]
#### **關鍵步驟說明**

1. **創建統一表示空間**：
    - **文本端**：使用在生物醫學文獻上預訓練的 Transformer 模型（如 **BioBERT**），將文本片段轉換為高質量的向量嵌入。
    - **圖譜端**：使用**圖神經網絡 (GNN)** 編碼器，學習圖中每個實體（基因、疾病、化合物）的向量嵌入。GNN 的優勢在於它能將實體的鄰域結構資訊編碼到嵌入中。
    - **統一**：將這兩種來源的嵌入存儲在**同一個向量資料庫**中。這使得我們可以進行跨模態的相似性搜索，例如，找到與某段描述"神經炎症"的文本最相似的基因實體。這就是**統一表示空間**的核心思想。
        
2. **知識注入 (Knowledge Infusion)**：
    - 我們**不**需要昂貴地對 LLM 進行完全的重新訓練。相反，我們採用高效的 **RAG** 框架。
    - **多模態檢索**：當用戶查詢時，系統同時從兩個數據源檢索信息：
        - **語義文本搜索**：在向量資料庫中找到與查詢語義最相關的文獻片段。
        - **圖路徑搜索**：在圖資料庫中找到與查詢中核心實體（如"阿茲海默症"）直接或間接相關的路徑或子圖。
    - **上下文提示組裝**：將檢索到的文本片段和圖路徑，格式化後一起放入 LLM 的提示 (Prompt) 中。這相當於在提問的同時，給 LLM 提供了「開卷考試」所需的所有參考資料。這一步有效地將結構化知識「注入」了 LLM 的當前對話中，使其推理更具事實依據。

白板圖 #3：深入探討 - 假設生成與推理工作流 (Hypothesis Generation Workflow)
這張圖詳細描述了從用戶查詢到生成假設的完整線上流程。
![[Pasted image 20250928031258.png]]


白板圖 #4：深入探討 - 證據三角測量與可解釋性介面 (Evidence Triangulation & Explainable UI)
這是系統的最終交付成果，確保研究人員能夠信任並驗證系統的輸出。
![[Pasted image 20250928031354.png]]
#### **介面設計理念**

- **可信度 (Trustworthiness)**：每個假設都必須有 LLM 生成的自然語言解釋和一個量化的置信度分數。
- **可追溯性 (Traceability)**：最重要的部分。用戶必須能夠一鍵點擊，追溯到生成該假設的原始文獻片段或知識圖譜中的具體關係。
- **證據三角測量 (Evidence Triangulation)**：介面並排顯示來自不同模態（文本和圖譜）的證據，讓研究人員可以像偵探一樣，從不同角度驗證一個假設。如果一個假設同時被最新的非結構化文獻和已建立的結構化知識庫所支持，那麼它的可信度就大大增加了。
- **用戶導向 (User-Centric)**：設計應符合科研人員的工作流程，讓他們能夠輕鬆地探索、驗證和導出結果，用於後續的實驗設計。


#### **系統流程概述**

1. **離線數據處理 (Offline)**：
#### 1. 問題理解與目標 (Problem Understanding & Goal)

**目標**：設計一個系統，能整合來自**非結構化數據**（如科學文獻、專利、臨床試驗報告）和**結構化數據**（如基因組學數據庫、蛋白質交互網絡、化合物庫）的資訊，以自動化地識別和排序潛在的藥物靶點 (Drug Target)。
#### 2. 系統設計 (System Design)

這是一個典型的多模態數據融合與知識發現問題，我會採用**檢索增強生成 (Retrieval-Augmented Generation, RAG)** 結合**知識圖譜 (Knowledge Graph, KG)** 的架構。
**架構圖：**

```
graph TD
    subgraph Ingestion & Preprocessing
        A[非結構化數據源 <br> PubMed, Patents, Clinical Trials] --> B{Text Processing Pipeline};
        C[結構化數據源 <br> TCGA, STRING, UniProt] --> D{KG Construction Pipeline};
    end

    subgraph Core Fusion Engine
        B --> E[Embedding Model <br> (BioBERT/Sci-LLaMA)];
        E --> F((Vector Database <br> FAISS/Pinecone));
        D --> G((Graph Database <br> Neo4j));
        E --> G;
    end

    subgraph Application Layer
        H[User Query <br> "Find novel targets for Alzheimer's"] --> I{Query Understanding & Expansion};
        I --> J{Multi-Modal Retriever};
        J -- Text Chunks --> F;
        J -- Graph Patterns --> G;
        F --> K{LLM Synthesizer & Reasoner <br> (Fine-tuned GPT/Llama)};
        G --> K;
        K --> L[Scoring & Ranking Module];
        L --> M[Output: Ranked Target List <br> with Evidence];
    end

```

**詳細流程：**
1. **數據攝取與預處理 (Data Ingestion & Preprocessing):**
    - **非結構化數據**:
        - 建立一個爬蟲和API客戶端，定期從PubMed, Google Patents, ClinicalTrials.gov等來源獲取最新文獻。
        - 對文本進行分塊 (Chunking)，例如按段落或固定大小的窗口，以便於後續的向量化。
    - **結構化數據**:
        - 解析來自TCGA (基因表達)、STRING (蛋白質交互)、UniProt (蛋白質功能) 等數據庫的資料。
        - 將這些實體（如基因、蛋白質、疾病、化合物）和它們之間的關係（如 `interacts_with`, `is_associated_with`）轉換為三元組 (Subject, Predicate, Object)。
            
2. **核心融合引擎 (Core Fusion Engine):**
    - **Embedding Model**: 使用一個在生物醫學語料上預訓練或微調的語言模型 (如BioBERT, Sci-LLaMA)，將文本區塊和結構化數據中的實體名稱統一轉換為高維向量 (Embeddings)。
    - **Vector Database**: 將所有文本區塊的向量存儲在向量數據庫中，用於快速的語義相似度搜索。
    - **Knowledge Graph (KG)**: 將結構化數據三元組構建成一個知識圖譜，並存儲在圖數據庫 (如Neo4j) 中。節點的Embedding也可以存儲為其屬性，實現圖與向量的融合。
    
3. **查詢與推理 (Query & Reasoning):**
    - **Query Understanding**: 當用戶輸入查詢（例如：“尋找治療阿茲海默症的新靶點”），LLM首先解析查詢，識別出關鍵實體（疾病：阿茲海默症）和意圖（尋找新靶點）。
    - **Multi-Modal Retriever**:
        - **文本檢索**: 將查詢轉換為向量，從向量數據庫中檢索最相關的文獻片段。
        - **圖譜檢索**: 在知識圖譜中，以「阿茲海マー症」為中心節點，執行圖查詢（如Cypher query），找出與其相關的基因、蛋白質及其交互關係。
    - **LLM Synthesizer**: 將檢索到的文本證據和圖譜子結構共同作為上下文 (Context)，餵給一個更強大的生成式LLM。LLM被指示 (Prompted) 基於這些上下文，**推理**出潛在的靶點，並**生成**支持該假設的證據摘要。
    - **Scoring & Ranking**: 設計一個評分模型，根據多種因素對LLM提出的候選靶點進行排序，例如：
        - **Novelty**: 在知識圖譜中與已知藥物的距離。
        - **Association Strength**: 文獻中提及的頻率和上下文強度。
        - **Druggability**: 根據蛋白質特性預測其是否容易被小分子藥物靶向。
#### 3. 衡量指標 (Metrics)
- **Offline**: 使用已知的藥物-靶點對進行交叉驗證，評估系統的召回率 (Recall)。
- **Online**: 追蹤科學家對推薦結果的交互率（點擊、保存）以及最終進入實驗驗證階段的靶點比例。

---

#### 第四部份

### **Q16: 為已部署的臨床診斷AI設計模型監控系統**

**Question 16: Design a Model Monitoring System for a Deployed Clinical Diagnostic AI**

- **Problem Statement:** An AI model that assists radiologists in detecting nodules in chest X-rays has been deployed in hospitals. As a Software as a Medical Device (SaMD), its performance must be continuously monitored in production to ensure it remains safe and effective, in compliance with FDA guidelines.
    
- **System Requirements:**
    
    - **Data & Concept Drift Detection:** Implement mechanisms to detect shifts in the input data distribution (e.g., new scanner models being used) and changes in the relationship between inputs and outcomes (e.g., evolving disease presentation).
        
    - **Performance Monitoring:** Continuously track key performance metrics (e.g., AUC-ROC, sensitivity, specificity) against the baseline established during clinical validation. This requires a feedback loop to get ground truth labels from radiologists.
        
    - **Fairness & Bias Monitoring:** Monitor model performance across different patient subgroups (e.g., by age, sex, race, hospital site) to detect and flag any performance degradation or bias.
        
    - **Alerting & Reporting:** An automated alerting system to notify the MLOps and clinical teams of significant performance drops or drift. The system must also generate periodic reports for regulatory compliance.
        
    - **Compliance:** The entire system must be HIPAA compliant and designed to support FDA's Total Product Lifecycle (TPLC) approach for AI/ML devices.
        
- **Task:** Design an end-to-end monitoring system. Describe what data you would log, which metrics you would track, the statistical tests you would use for drift detection, and the overall architecture for alerting and reporting.

#### 1. 問題理解與目標

**目標**：設計一個全面的監控系統，確保一個已在臨床環境中使用的AI模型（例如，用於病理影像的癌症分類模型）的性能、穩定性和可靠性，並能在發生問題時及時告警。

#### 2. 系統設計

一個強健的監控系統必須涵蓋三個層面：**數據健康度**、**模型性能**和**系統操作**。

**架構圖：**

程式碼片段

```
graph TD
    A[Live Clinical Data] --> B{AI Diagnostic Service};
    B -- Prediction & Features --> C((Logging Database <br> e.g., ELK Stack));
    
    subgraph Monitoring System (Asynchronous)
        C --> D{Data Drift Detector};
        C --> E{Model Performance Monitor};
        C --> F{Operational Monitor};
    end

    subgraph Alerting & Reporting
        D -- Drift Detected --> G{Alerting System <br> (PagerDuty, Email)};
        E -- Performance Drop --> G;
        F -- Latency Spike --> G;
        D & E & F --> H((Monitoring Dashboard <br> e.g., Grafana));
    end

    I[Reference Data <br> (Training/Validation Set)] --> D;
    J[Delayed Ground Truth <br> (Pathologist Reports)] --> E;
```

**監控細節：**

1. **數據健康度監控 (Data Health Monitoring):**
    
    - **目的**: 檢測**數據漂移 (Data Drift)**，即線上輸入數據的分佈是否與模型訓練時的數據分佈發生了顯著變化。
        
    - **監控指標**:
        
        - **特徵分佈**: 對於影像數據，監控亮度、對比度、色彩直方圖等低階特徵的分佈。對於表格數據，監控每個特徵的均值、方差、缺失率。
            
        - **漂移檢測算法**: 使用**Kolmogorov-Smirnov (K-S) 測試**或**Population Stability Index (PSI)** 來量化線上數據與參考數據（訓練集）之間的分佈差異。
            
    - **告警**: 當漂移指標超過預設閾值時，自動發出告警。
        
2. **模型性能監控 (Model Performance Monitoring):**
    
    - **目的**: 追蹤模型的預測準確性是否下降，檢測**概念漂移 (Concept Drift)**。
        
    - **監控指標**:
        
        - **輸出分佈**: 監控模型預測類別的比例。例如，如果模型突然開始預測「陽性」的比例遠高於歷史水平，這可能是一個危險信號。
            
        - **核心性能指標**:
            
            - 這需要**延遲的真實標籤 (Delayed Ground Truth)**，例如幾天後由病理學家確認的診斷結果。
                
            - 當真實標籤可用時，計算並追蹤**準確率 (Accuracy)、精確率 (Precision)、召回率 (Recall)、AUC-ROC**等指標。
                
    - **告警**: 當任何核心性能指標低於預設的服務水平協議 (SLA) 時，發出高級別告警。
        
3. **系統操作監控 (Operational Monitoring):**
    
    - **目的**: 確保模型服務本身運行穩定。
        
    - **監控指標**:
        
        - **延遲 (Latency)**: 每次預測所需的時間。
            
        - **吞吐量 (Throughput)**: 單位時間內處理的請求數量。
            
        - **錯誤率 (Error Rate)**: 服務器錯誤（如5xx）的比例。
            
        - **資源使用率**: CPU、GPU、記憶體的使用情況。
            
    - **告警**: 延遲或錯誤率的異常飆升應觸發告警。
        

#### 3. 合規性 (Compliance)

所有監控日誌、告警和應對措施都必須被詳細記錄，以符合FDA等監管機構的審計要求。

---

### **Q17: 為生產環境中的ML模型設計回滾策略**

**Question 17: Design a Rollback Strategy for ML Models in a Production Environment**

- **Problem Statement:** A new version of a predictive model has been deployed, but post-deployment monitoring reveals a critical issue (e.g., a sudden drop in performance, biased predictions for a key user segment). You need a robust, automated strategy to safely roll back to a previous, stable version of the model with minimal disruption.
    
- **System Requirements:**
    
    - **Rapid Reversion:** The system must be able to switch back to the last known good model version within minutes of detecting a critical failure.
        
    - **State Management:** The rollback strategy must handle not just the model artifact but also any associated components, such as feature engineering code or data schemas, to prevent mismatches.
        
    - **Automated Triggers:** Define clear, automated triggers for initiating a rollback (e.g., performance metric dropping below a critical threshold, data validation failure).
        
    - **Deployment Strategy Integration:** The rollback plan must be integrated with the chosen deployment strategy (e.g., Blue-Green, Canary).
        
    - **Post-Mortem Analysis:** The system should preserve the state and logs of the faulty deployment to facilitate a thorough post-mortem analysis.
        
- **Task:** Compare and contrast how you would implement a rollback strategy for a model deployed using a **Blue-Green** strategy versus a **Canary** strategy. Discuss the technical components required (e.g., model registry, traffic routing, monitoring hooks) and the trade-offs of each approach.

#### 1. 問題理解與目標

**目標**：設計一個安全機制，當新部署的ML模型版本在生產環境中表現不佳或導致系統不穩定時，能夠快速、自動地恢復到前一個已知的穩定版本，將負面影響降到最低。

#### 2. 策略設計

一個好的回滾策略不僅僅是「切換版本」，它是一個結合了**部署策略**、**自動化監控**和**版本控制**的完整流程。

**流程圖：**

程式碼片段

```
sequenceDiagram
    participant CI/CD
    participant Model_Registry as Model Registry
    participant APIGateway as API Gateway/Router
    participant Monitor as Monitoring System
    participant OnCall as On-Call Engineer

    CI/CD ->> Model_Registry: Push New Model v2 (Staged)
    CI/CD ->> APIGateway: Start Canary Release (1% traffic to v2)
    Monitor ->> APIGateway: Continuously check v1 & v2 performance
    
    alt v2 Performance OK
        CI/CD ->> APIGateway: Gradually increase traffic to v2 (10% -> 50% -> 100%)
    else v2 Performance Degrades
        Monitor -->> Monitor: Performance metrics drop below threshold
        Monitor ->> APIGateway: **Trigger Automated Rollback!**
        APIGateway ->> APIGateway: Route 100% traffic back to v1
        Monitor ->> OnCall: Send High-Severity Alert
        CI/CD ->> CI/CD: Block further deployments of v2
    end
```

**關鍵組成部分：**

1. **前提：嚴格的版本控制 (Prerequisite: Strict Versioning):**
    
    - **模型版本化**: 使用**模型註冊表 (Model Registry)**，如MLflow或Vertex AI Model Registry，來存儲和版本化模型文件、訓練參數和性能指標。每個部署的模型都必須有唯一的版本ID。
        
    - **代碼與數據版本化**: 使用Git管理代碼，DVC管理數據，確保任何模型版本都可以完全復現。
        
2. **部署策略 (Deployment Strategy):**
    
    - **金絲雀發布 (Canary Releasing)**: 這是最推薦的策略。新模型 (v2) 首先只接收一小部分（例如1%-5%）的線上流量。舊模型 (v1) 繼續處理剩餘的流量。
        
    - **藍綠部署 (Blue-Green Deployment)**: 維護兩套完全相同的生產環境（藍色和綠色）。如果藍色是當前線上環境，新模型部署到綠色環境。經過測試後，流量路由器將所有流量從藍色切換到綠色。回滾就是將流量切換回藍色。
        
3. **自動化回滾觸發器 (Automated Rollback Triggers):**
    
    - 將回滾機制與**Q16中設計的監控系統**緊密集成。
        
    - **性能指標惡化**: 如果新版本模型的關鍵業務指標（如準確率）或技術指標（如延遲）在金絲雀流量上顯著差於舊版本，立即觸發回滾。
        
    - **系統錯誤率飆升**: 如果新模型導致服務器錯誤率顯著增加，立即回滾。
        
4. **回滾執行 (Rollback Execution):**
    
    - **API網關/負載均衡器**: 回滾操作的核心是配置流量路由規則。自動化腳本會調用API網關的接口，將所有流量指向舊的、穩定的模型版本。
        
    - **鎖定與告警**: 一旦觸發自動回滾，CI/CD系統應立即**鎖定**有問題的版本，防止其被再次部署。同時，向待命工程師發出緊急警報。
        
5. **手動回滾 (Manual Override):**
    
    - 永遠要保留一個清晰、簡單的手動回滾按鈕，供工程師在自動化系統失靈或遇到未知問題時使用。
        

---

### **Q18: 為藥物靶點推薦系統設計A/B測試框架**

**Question 18: Design an A/B Testing Framework for a Drug Target Recommendation System**

- **Problem Statement:** Your team has developed a new ML model that recommends potential drug targets to research scientists. Before rolling it out to the entire R&D department, you need to prove that it provides better recommendations than the existing system (which could be an older model or a manually curated list).
    
- **System Requirements:**
    
    - **Experiment Design:** Define the key components of the A/B test: hypothesis, user groups (scientists), control (System A) vs. treatment (System B), and primary/secondary metrics.
        
    - **Metrics Selection:** The primary metrics are not simple clicks. They are long-term and complex, such as "number of targets selected for follow-up screening" or "time to identify a validated hit." How would you design metrics that are measurable within a reasonable timeframe?
        
    - **Statistical Rigor:** Ensure the experiment is statistically sound, accounting for sample size, duration, and potential biases (e.g., network effects if scientists collaborate).
        
    - **Infrastructure:** Design the technical infrastructure to randomly assign users to groups, serve recommendations from the correct system, and log user interactions and outcomes for analysis.
        
    - **User Experience:** The framework must not disrupt the scientists' workflow.
        
- **Task:** Design a complete A/B testing framework for this internal, expert-user-facing system. Pay special attention to the challenge of defining meaningful and measurable long-term success metrics and how you would handle the low-traffic, high-impact nature of the recommendations.

#### 1. 問題理解與目標

**目標**：設計一個實驗框架，用以科學地評估一個新的藥物靶點推薦算法（版本B）是否比現有算法（版本A）更有效。主要挑戰在於，靶點推薦的「成功」反饋週期非常長。

#### 2. 框架設計

**設計原則**: 分層指標，結合短期代理指標和長期真實指標。

**實驗流程:**

1. **用戶分流 (User Splitting):**
    
    - 當一位科學家登錄系統時，根據其用戶ID的哈希值，將其**隨機且持久地**分配到**A組（對照組，使用舊算法）或B組（實驗組，使用新算法）**。分流應在後端服務層完成，對用戶透明。
        
2. **假設建立 (Hypothesis Formulation):**
    
    - 一個清晰的假設是A/B測試的基礎。例如：「我們假設，新的基於知識圖譜的推薦算法（B），相比現有的基於協同過濾的算法（A），能夠將科學家『將靶點加入到實驗計劃中』的比率提高10%。」
        
3. **指標設計 (Metrics Design):**
    
    - **短期代理指標 (Short-term Proxy Metrics) - 用於快速迭代:**
        
        - **點擊率 (Click-Through Rate, CTR)**: 推薦靶點被點擊查看詳情的比例。
            
        - **交互深度 (Engagement Depth)**: 用戶在靶點詳情頁面的停留時長，或是否查看了相關文獻。
            
        - **"保存" / "收藏" 率**: 用戶將推薦靶點加入個人工作列表的比率。
            
        - **注意**: 這些指標可能具有誤導性，需要謹慎解讀。
            
    - **中期業務指標 (Mid-term Business Metrics) - 更接近真實價值:**
        
        - **實驗計劃採納率 (Lab Plan Adoption Rate)**: 這是關鍵的中期指標。追蹤有多少被推薦的靶點被正式地加入到實驗室信息管理系統 (LIMS) 的篩選計劃中。
            
        - **用戶反饋分數**: 在推薦列表旁設置一個簡單的「這些推薦相關嗎？」的問卷，收集定性反饋。
            
    - **長期目標指標 (Long-term Goal Metrics) - 最終成功標準:**
        
        - **實驗成功率 (Experimental Success Rate)**: 追蹤幾個月甚至幾年後，來自A組和B組推薦的靶點，在初步的體外 (in vitro) 實驗中的成功率。
            
        - **解決方案**: A/B測試框架需要有能力記錄下每個推薦的來源（A或B），並在未來與實驗結果數據庫進行關聯分析。這是一個**長期回顧性分析**。
            
4. **統計顯著性 (Statistical Significance):**
    
    - 在實驗開始前，計算所需的**樣本量 (Sample Size)**，以確保實驗有足夠的統計功效 (Statistical Power) 來檢測出預期的效果。
        
    - 使用**t檢驗**（用於連續指標如停留時長）或**卡方檢驗**（用於比例指標如CTR）來比較A、B兩組的差異。
        
    - 結果應報告**p-value**和**信賴區間 (Confidence Interval)**。
        
5. **基礎設施 (Infrastructure):**
    
    - **實驗配置服務**: 一個中心化的服務，用於管理所有正在運行的A/B測試的配置（例如，哪個是A，哪個是B，流量分配比例）。
        
    - **日誌系統**: 記錄每一次推薦的曝光 (Impression) 和用戶的每一次交互 (Click, Save)，並附帶用戶ID和其所屬的實驗組。
        
    - **分析儀表板**: 一個可視化儀表板，展示各項指標在A、B兩組的實時表現。
        

---

### **Q19: 為蛋白質結構預測模型設計低延遲的實時推理系統**

**Question 19: Design a Low-Latency, Real-Time Inference System for a Protein Structure Prediction Model**

- **Problem Statement:** GSK wants to provide an internal, interactive tool for chemists where they can input a novel amino acid sequence and get its predicted 3D structure in near real-time (e.g., within seconds). The underlying model is a large, computationally expensive Transformer-based model similar to AlphaFold.
    
- **System Requirements:**
    
    - **Low Latency:** The end-to-end inference time (from user request to returning the predicted structure) must be under 10 seconds for a medium-length protein.
        
    - **High Throughput:** The system should be able to handle concurrent requests from dozens of scientists.
        
    - **Cost-Effectiveness:** The GPU infrastructure required for serving the model should be managed efficiently to control costs.
        
    - **Model Optimization:** The large model must be optimized for fast inference without significant loss of accuracy.
        
- **Task:** Design the architecture for this real-time inference system. Discuss the strategies you would use to meet the latency and cost requirements, including:
    
    - Model optimization techniques (e.g., quantization, pruning, knowledge distillation).
        
    - Inference serving frameworks (e.g., NVIDIA Triton Inference Server).
        
    - Infrastructure choices (e.g., GPU types, auto-scaling, caching strategies for common sub-sequences).

#### 1. 問題理解與目標

**目標**：將一個計算密集型的蛋白質結構預測模型（如AlphaFold2或其變體）部署為一個交互式服務。用戶輸入氨基酸序列後，系統需要在幾秒到一分鐘內返回預測的3D結構，而不是傳統的數小時。

#### 2. 系統設計

低延遲的實現需要從**模型優化**、**硬件加速**和**智能緩存**三個方面進行綜合設計。

**架構圖：**

程式碼片段

```
graph TD
    User -- Protein Sequence --> A[API Gateway];
    A --> B{Cache Lookup (Redis)};
    B -- Cache Hit --> C[Return Cached PDB];
    C --> User;
    B -- Cache Miss --> D{Inference Service};

    subgraph Inference Service
        D --> E{Request Batching};
        E --> F[Optimized Model on GPU <br> (TensorRT)];
        F --> G{Result Processing};
    end

    G --> H{Update Cache};
    G --> User;
```

**設計細節：**

1. **模型優化 (Model Optimization - Offline):**
    
    - **模型編譯 (Model Compilation)**: 使用**NVIDIA TensorRT**或類似的工具，將訓練好的PyTorch/TensorFlow模型轉換為針對特定GPU（如A100, H100）高度優化的推理引擎。這個過程包括層融合 (Layer Fusion)、精度校準等操作。
        
    - **量化 (Quantization)**: 將模型的權重從32位浮點數 (FP32) 轉換為16位浮點數 (FP16) 或8位整數 (INT8)。這能顯著提升GPU上Tensor Core的計算速度，並減少內存佔用，通常只會帶來微小的精度損失。
        
    - **知識蒸餾 (Knowledge Distillation)**: 訓練一個更小、更快的「學生模型」，使其學習並模仿原始大型「教師模型」的輸出。這個學生模型用於快速提供一個近似的結果。
        
    - **流水線並行 (Pipeline Parallelism)**: 將模型的不同部分（例如，MSA搜索和結構摺疊）部署在不同的計算節點上，形成流水線，以重疊計算和I/O時間。
        
2. **推理基礎設施 (Inference Infrastructure - Online):**
    
    - **專用硬件 (Specialized Hardware)**: 部署在配備了高端GPU (NVIDIA A100/H100) 的服務器上。
        
    - **高性能服務框架 (High-Performance Serving Framework)**:
        
        - 使用**NVIDIA Triton Inference Server**。它是一個專為大規模AI推理設計的開源框架。
            
        - **動態批處理 (Dynamic Batching)**: Triton可以將在短時間內到達的多個獨立請求自動組合成一個批次 (Batch)，然後一次性送入GPU計算。這極大地提高了GPU的利用率，攤薄了單次請求的開銷。
            
        - **模型併發 (Concurrent Model Execution)**: 可以在同一GPU上同時運行多個模型實例，進一步提高吞吐量。
            
3. **智能緩存策略 (Intelligent Caching):**
    
    - **結果緩存 (Result Caching)**:
        
        - 在API網關和推理服務之間加入一個內存緩存（如**Redis**）。
            
        - 以氨基酸序列的哈希值為鍵 (Key)，以預測出的PDB結構文件為值 (Value)。
            
        - 對於常見的、已被研究的蛋白質序列，可以直接從緩存中返回結果，實現毫秒級響應。
            
    - **中間特徵緩存 (Intermediate Feature Caching)**: 蛋白質結構預測中最耗時的步驟之一是多序列比對 (MSA) 搜索。可以將計算出的MSA結果緩存起來，當有相似序列的請求時，可以重用或微調已有的MSA，而不是從頭開始搜索。
        

---

### **Q20: 為複雜生物醫學數據標註設計人機迴路系統**

**Question 20: Design a Human-in-the-Loop System for Labeling Complex Biomedical Data**

- **Problem Statement:** Training a state-of-the-art model for segmenting cell boundaries in high-content microscopy images requires a large, accurately labeled dataset. Manually labeling every cell in every image is prohibitively expensive and requires expert pathologists.
    
- **System Requirements:**
    
    - **Active Learning Integration:** Design a system where the ML model and human experts collaborate. The model should intelligently select the most informative or uncertain image patches for the experts to label, rather than labeling data randomly.
        
    - **Efficient Annotation Interface:** A user-friendly interface for pathologists to quickly review, correct, or create cell segmentation masks. The interface should be integrated with the model to provide initial "suggested" segmentations.
        
    - **Iterative Training Loop:** An automated backend that takes the newly labeled data from the experts, retrains or fine-tunes the model, and updates the model used for selecting the next batch of images.
        
    - **Quality Control & Consensus:** A mechanism to handle disagreements between annotators and to measure the quality of the labels being produced.
        
    - **Cost-Benefit Analysis:** The system should track metrics to evaluate its efficiency, such as the reduction in labeling effort compared to random sampling and the model performance improvement per labeled batch.
        
- **Task:** Describe the architecture of this human-in-the-loop system. Explain the active learning strategy you would choose, the data flow between the model and the human annotators, and how you would manage the iterative training and quality control process to maximize the value of the experts' time.

#### 1. 問題理解與目標

**目標**：設計一個高效、準確且可擴展的系統，用於標註大規模、複雜的生物醫學數據（例如，在高內涵成像中分割細胞器，或在病理切片中識別腫瘤區域）。系統應結合機器學習模型和人類專家的智慧，以最小化專家標註的負擔。

#### 2. 系統設計

核心思想是**主動學習 (Active Learning)**，讓模型主動挑選出它最「困惑」、最需要人類指導的樣本，從而將專家的寶貴時間用在刀刃上。

**人機迴路 (Human-in-the-Loop) 流程圖：**

程式碼片段

```
graph TD
    Start --> A{1. Bootstrap: Train initial model on small labeled set};
    A --> B{2. Inference: Model pre-annotates large unlabeled pool};
    B --> C{3. Active Learning: Select most uncertain samples};
    C --> D{4. Human Annotation: Experts correct/verify pre-annotations};
    D --> E{5. Quality Control: Consensus & Gold Standard};
    E --> F{6. Augment Training Set: Add new high-quality labels};
    F --> G{7. Retrain Model};
    G --> B;
    style G fill:#f9f,stroke:#333,stroke-width:2px
```

**系統組件詳解：**

1. **啟動階段 (Bootstrap):**
    
    - 由領域專家手動標註一小批「種子數據集」。
        
    - 使用這個小數據集訓練一個初始的分割模型（例如U-Net）。
        
2. **推理與預標註 (Inference & Pre-annotation):**
    
    - 將初始模型應用於大量未標註的圖像上，生成初步的分割掩碼或邊界框，作為「預標註」。
        
3. **主動學習策略 (Active Learning Strategy):**
    
    - 這是系統的核心。模型不會隨機選擇樣本給專家，而是通過**不確定性採樣 (Uncertainty Sampling)** 來挑選最有價值的樣本。
        
    - **不確定性度量**:
        
        - **熵 (Entropy)**: 對於圖像分割，計算像素級別預測概率的熵，熵越高的區域表示模型越不確定。
            
        - **最小邊界 (Least Margin)**: 選擇模型預測的最可能類別和次可能類別之間概率差最小的樣本。
            
        - **委員會查詢 (Query-by-Committee)**: 訓練多個模型，選擇它們意見分歧最大的樣本。
            
    - 系統將不確定性最高的圖像優先推送到標註隊列中。
        
4. **智能標註界面 (Intelligent Annotation UI):**
    
    - 這不是一個簡單的畫圖工具。界面會直接向專家展示圖像和模型的**預標註結果**。
        
    - 專家的工作從「從零開始畫」變成了「修正模型的錯誤」。這大大提高了效率。
        
    - 界面應提供高效的修正工具，如智能畫筆（可以吸附到邊緣）、橡皮擦等。
        
5. **質量控制 (Quality Control):**
    
    - **共識機制 (Consensus)**: 將同一個高不確定性的樣本發送給2-3位標註專家。只有當他們的標註結果高度一致時（例如，分割掩碼的IoU > 0.9），該標註才被接受進入訓練集。不一致的樣本會被提交給資深專家進行最終裁決。
        
    - **黃金標準集 (Gold Standard Set)**: 在標註隊列中悄悄插入一些已知答案的「測試樣本」，用以評估和監控每位標註專家的表現。
        
6. **模型重訓練與迭代 (Model Retraining & Iteration):**
    
    - 定期（例如，每增加1000個高質量標註）將新的專家驗證數據加入到訓練集中，重新訓練或微調模型。
        
    - 更新後的模型會變得更聰明，其預標註的質量會更高，主動學習挑選樣本的效率也會提升。這個循環不斷重複，形成一個正向反饋。