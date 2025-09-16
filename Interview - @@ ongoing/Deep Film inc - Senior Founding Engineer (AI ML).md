
Job: 
https://www.linkedin.com/jobs/view/senior-founding-engineer-ai-ml-at-deep-film-inc-4293467357/

You’re here to reshape the film industry with AI. At Deep Film Inc., you’ll take ==multimodal diffusion and transformer== models from research to production—mastering quantization and pruning to make them leaner and faster. Your CI/CD pipelines and MLflow integrations won’t just keep us running—they’ll redefine how stories are told. You’ll scale training across GPU clusters with Horovod and Optuna and build resilient data pipelines with Kafka, tf.data, or whatever it takes to get the job done. 你來這裡，是為了用人工智慧重塑電影產業。在 Deep Film Inc.，你將把多模態擴散和變換模型從研究階段推進到生產階段，掌握量化和剪枝技巧，使其更精簡、更快速。你的 CI/CD 管線和 MLflow 整合不僅能保障我們持續運營，也將重新定義故事的敘述方式。你將使用 Horovod 和 Optuna 在 GPU 叢集之間擴展訓練，並使用 Kafka、tf.data 或任何必要的工具建立彈性資料管線，以完成工作。  
  
This is no ordinary role. Expect complex challenges, daily whiteboarding, and rapid prototyping in a tight, fast moving team. You’ll join with founding-level equity and access to state-of-the-art hardware. Optional relocation to California will be available post-launch in January 2026. 這絕非普通的職位。你將在一個緊密協作、快速發展的團隊中，迎接複雜的挑戰、每日的白板練習和快速的原型設計。你將獲得創始等級的股權和最先進的硬體資源。公司成立後，2026 年 1 月將提供加州辦公地點的選配。  
  
Why You?  
You thrive on hard problems, are obsessed with deep learning, and see AI as a way to transform storytelling. Whether you’re a seasoned expert or emerging talent, show us your impact: a GitHub repo with production grade systems, optimized ML models, or a demo that leaves us speechless. 為什麼選擇你？

你擅長解決難題，沉迷於深度學習，並將人工智慧視為改變故事敘述方式的一種方式。無論您是經驗豐富的專家還是新興人才，請向我們展示您的影響力：具有生產級系統的 GitHub 儲存庫、優化的 ML 模型或讓我們無言以對的演示。





下面我把這則職缺（Job Description）拆成幾個重點，用中文詳細說明 **公司／產品（從 JD 能推測的）**、**這個職位會做哪些事**、以及 **需要哪些技能與經驗**，並補上面試 / 準備建議，方便你快速對照自己的履歷或準備資料。

# 1) 公司 & 產品（從 JD 可以推測）

- **使命與定位**：Deep Film 要「用 AI 改變電影產業」，換句話說是把生成式 / 多模態 AI（例如 diffusion model + transformer）實際用在影像/影視製作流程上，從研究做成可用的產品/服務。
    
- **產品可能的範圍（推測）**：像是劇本→分鏡／分場景視覺化（previs）、視覺效果自動化、素材/場景/角色生成、影像後製加速、或協助導演/製作在前期做視覺決策的工具。JD 強調「multimodal diffusion and transformer」，代表產品會同時處理文字、圖像、影片、可能還有音訊或時序資訊。
    
- **組織階段 & 資源**：提供「創辦人等級的 equity」、「state-of-the-art hardware」，還提到 2026 年一月後可選擇遷移到加州，顯示是早期（pre/post-launch）且拿到部分硬體資源的新創團隊 — 需要「跨領域快速落地」的能力與做事節奏。
    

# 2) 這個職位會負責的主要工作（對應 JD）

- **把研究模型 productize**：把 multimodal diffusion/transformer 從 prototype 轉成可部署、可維運、延遲/成本可控的 production model。
    
- **模型壓縮 & 加速**：做 quantization、pruning、其他蒸餾或編譯優化（例如 ONNX/TensorRT）以降低延遲與運算成本。
    
- **機器學習工程化 / MLOps**：建立 CI/CD（含模型測試、自動化部署）、整合 MLflow 做實驗及模型追蹤、以及部署/監控流程。
    
- **分散式訓練與超參數優化**：在 GPU 叢集上 scale training（JD 指 Horovod）並用 Optuna 等工具做自動化搜尋／超參數管理。
    
- **資料管線與可靠性**：建立高韌性的資料流／ETL（JD 提到 Kafka、tf.data 或其他工具），處理大量影像/影片資料、streaming 或 batch pipelines。
    
- **系統設計與快速原型**：每天白板設計（system design / algorithm sketch），快速做可 demo 的 prototype，並在小團隊內快速迭代。
    
- **跨職能合作與產品化**：和產品、視覺藝術家、導演或後製團隊合作，將技術融入實際製作流程。
    

# 3) 需要的技能與經驗（技術面＋軟實力）

## 必備的技術能力（JD 明確或強烈暗示）

- **深度學習基礎**：熟悉 transformer、diffusion models、multimodal architectures（文字↔圖像↔影片）。
    
- **模型壓縮/推理優化**：quantization（INT8/混點）、pruning、knowledge distillation、ONNX/TensorRT 等經驗。
    
- **分散式訓練**：Horovod 或 PyTorch DDP、NCCL、GPU 叢集運作經驗；能處理大型模型的訓練穩定性與效能優化。
    
- **超參數優化**：Optuna、Ray Tune 或其他自動化調參工具的實務使用經驗。
    
- **MLOps / CI-CD / 實驗追蹤**：建立 MLflow pipelines、CI/CD（GitHub Actions / Jenkins / GitLab CI）、模型版本控制與自動化部署流程。
    
- **資料工程**：Kafka（streaming）或 tf.data（高效資料讀取）、ETL pipeline 設計、處理大量影像/影片資料的經驗。
    
- **生產環境部署**：容器化（Docker）、Kubernetes、model servers（Triton / TorchServe / TF Serving）、監控（Prometheus/Grafana）等。
    
- **程式語言**：Python（主）、PyTorch / TensorFlow；若能寫 C++ / CUDA 做底層優化是加分項。
    
- **工程實務**：benchmarking、性能分析、成本優化（推理成本/訓練成本）、安全性與資料隱私考量。
    

## 軟技能與職位適應性

- **Startup / founding mindset**：願意穿多頂帽子，對不確定性與快速變動有高容忍度，能主動 ownership。
    
- **溝通與跨領域合作**：能把技術語言轉換給非技術的導演/藝術家/產品人員。
    
- **系統設計與白板思考**：JD 明確提到 daily whiteboarding，代表會考系統設計/算法思路。
    
- **結果導向與快速原型能力**：把概念變成能 demo 的成品，短週期驗證假設。
    

# 4) 面試準備 & 如何在履歷／作品集中證明你符合條件（具體建議）

- 在履歷/portfolio 強調 **end-to-end production projects**：例如把一個研究模型做到：training → MLflow tracking → quantized inference → 部署到 Triton 或 K8s，上線後的 latency/throughput 指標與成本比較。
    
- 在 GitHub/links 上放 **可 reproducible 的 demo 或 benchmark**（production-grade repo、Dockerfile、benchmark scripts、訓練/部署指令）。
    
- 準備能展示的 **模型優化案例**：例如 INT8 量化後的精度與延遲比較表、pruning 帶來的稀疏化效果、或使用 TensorRT 的 speedup。
    
- 準備 **分散式訓練實例**：展示用 Horovod/分布式 PyTorch 在多 GPU/多節點訓練的 config、log、以及如何解決梯度同步、OOM、I/O 瓶頸等問題。
    
- 練習 **系統設計白板題**：例如「設計一個可支援 4K 視頻批次推理的服務架構」或「如何設計一個可擴展的影片標註 & pipeline」。
    
- 準備案例說明你如何在 production 中使用 **MLflow/CI-CD/Kafka**：show config、pipeline diagram、錯誤處理跟回滾流程。
    

# 5) 小結 — 你適合這個職位的信號

- 你如果有 **從研究到 production 的實戰經驗**（能把 SOTA 模型工程化），以及 **模型壓縮 + 分散式訓練 + MLOps** 的組合經驗，那你就是理想人選。
    
- 若你還缺某些工具的實務經驗（比如 Horovod 或 Kafka），但在相鄰工具（PyTorch DDP、RabbitMQ、Airflow）有深度經驗且能快速上手，也仍有很大機會——因為這是 early-stage、需要能迅速做出產品的人。