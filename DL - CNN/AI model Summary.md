



# 常用物件偵測、影像分割與物件追蹤模型之比較分析報告

## I. 電腦視覺模型基準測試概論

電腦視覺領域涵蓋了多種關鍵任務，每種任務都有其特定的目標和評估標準。本報告旨在對常用的物件偵測、影像分割及物件追蹤模型進行全面的基準比較。

**A. 核心任務定義**

1. **物件偵測 (Object Detection)**： 此任務的核心目標是在影像中識別並定位多個物件，通常是透過預測圍繞物件的邊界框 (bounding box) 並對每個框內的物件進行分類。關鍵的評估指標包括平均精度均值 (mean Average Precision, mAP)，此指標通常在不同的交並比 (Intersection over Union, IoU) 閾值下計算（例如 mAP@[.5:.95]，代表 IoU 從 0.5 到 0.95，間隔 0.05 的平均 mAP）。此外，也會報告在特定 IoU 閾值下的 AP（例如 AP50、AP75），以及針對不同大小物件（小 APS、中 APM、大 APL 物件）的性能 。  
    
2. **影像分割 (Image Segmentation)**： 影像分割旨在將影像劃分為多個片段或區域。
    
    - **實例分割 (Instance Segmentation)**：此任務不僅偵測和分類每個物件實例，還在像素層級上描繪其輪廓。常用的評估指標是遮罩平均精度均值 (mask mAP, mAPmask​)，它評估預測分割遮罩的品質，通常與邊界框平均精度均值 (mAPbox​) 一同報告 。  
        
    - **語義分割 (Semantic Segmentation)**：此任務為影像中的每個像素分配一個類別標籤，提供密集的預測。主要的評估指標是平均交並比 (mean Intersection over Union, mIoU)，計算方式為所有類別的 IoU 平均值 。  
        
3. **物件追蹤 (Object Tracking) (多物件追蹤 - MOT)**： 多物件追蹤的目標是在影片序列中偵測多個物件，為其分配唯一的身份識別碼 (ID)，並在物件移動、互動和相互遮擋時跨影格保持這些 ID 的一致性 。主要的評估指標包括：  
    
    - **MOTA (Multiple Object Tracking Accuracy)**：綜合考量偽陽性 (false positives)、偽陰性 (false negatives) 和身份切換 (identity switches) 的指標 。  
        
    - **MOTP (Multiple Object Tracking Precision)**：衡量預測物件位置與真實物件位置之間不一致性的指標 。  
        
    - **IDF1 (ID F1 Score)**：正確識別偵測的 F1 分數，平衡了 ID 精確度和 ID 召回率 。  
        
    - **HOTA (Higher Order Tracking Accuracy)**：一個較新的指標，明確平衡了偵測和關聯的準確性 。  
        

**B. 關鍵性能與效率指標**

- **參數數量 (Params)**：神經網路中可學習權重和偏差的總數。此指標是模型大小及其儲存和部署時記憶體佔用的主要指示。通常以百萬 (M) 為單位報告 。  
    
- **浮點運算次數 (FLOPs)**：衡量模型計算複雜度的指標，通常指單次前向傳播的運算量，量化了乘加運算的次數。常以 GFLOPs (GigaFLOPs, 109 FLOPs) 或 BFLOPs (Billion FLOPs，等同於 GFLOPs) 為單位報告。FLOPs 通常取決於輸入影像的解析度 。  
    
- **延遲 (Latency)**：模型對單個輸入（例如一張影像或一個影格）執行一次推論所需的時間。通常以毫秒 (ms) 為單位。延遲高度依賴於硬體（CPU、GPU 型號、TPU）、批次大小 (batch size) 和軟體優化（例如 TensorRT、ONNX runtime）。  
    
- **吞吐量 (Throughput)**：模型處理輸入的速率，通常以每秒影格數 (Frames Per Second, FPS) 為單位。與延遲類似，吞吐量取決於硬體、批次大小和優化措施 。  
    
- **準確度指標 (mAP, mIoU, MOTA, IDF1, HOTA, J&F)**：這些指標量化了模型執行其指定任務的優劣程度。J&F（Jaccard Index 和 F-score）是影片物件分割的常用指標，結合了區域準確度和輪廓對齊度 。  
    

**C. 標準化基準測試的重要性**

使用如 COCO 、Pascal VOC 、Cityscapes 、MOTChallenge (MOT17, MOT20) 、ISBI 資料集 、LVIS 和 SA-1B 等通用大型資料集對於實現不同模型之間公平且可重現的比較至關重要。性能數據只有在結合評估時使用的特定硬體（例如 NVIDIA V100、A100、T4 GPU；CPU；TPU）和軟體環境（例如 PyTorch、TensorFlow，以及 Detectron2、MMDetection、Ultralytics 等專用框架）的背景下才有意義 。  

性能指標（如延遲和吞吐量）並非模型的內在屬性，而是高度依賴於評估環境。例如，YOLOv8n 在 CPU ONNX 上的延遲為 80.4ms，但在 A100 GPU 上使用 TensorRT 時則降至 0.99ms 。同樣地，EfficientDet 在 V100 GPU 上的延遲可以透過使用 TensorRT 大幅降低 。一個模型的架構定義了其在給定輸入大小下的理論 FLOPs。然而，這些 FLOPs 如何轉換為實際執行時間（延遲）取決於硬體並行性（CPU 與 GPU）、特定硬體架構（V100 與 T4）以及軟體層級的優化（編譯器、像 TensorRT 這樣的推論引擎、量化）。批次大小也扮演著重要角色；較大的批次大小可以透過更好地利用並行硬體來提高吞吐量 (FPS)，即使每個樣本的延遲可能略有增加或保持相似。因此，在比較模型時，必須對硬體和軟體設置進行標準化，或至少承認這些差異。報告通常會註明「V100 延遲」或「T4 FPS」以提供此類背景資訊。缺乏這些資訊，比較將是不可靠的。  

Papers with Code 等平台以及 MMDetection 、Ultralytics 和 Detectron2（在 MMDetection 的比較中被引用 ）等綜合框架已成為電腦視覺研究領域的核心。它們提供預訓練模型、標準化評估腳本和基準測試結果。現代電腦視覺模型的複雜性使得重新實現既容易出錯又耗時。模型庫提供了現成的、通常是預訓練的實現，可作為強大的基線。這些框架內的標準化評估（例如 MMDetection 的 `benchmark.py` ）試圖提供可比較的結果，儘管底層函式庫版本或微小設置細節的差異仍可能存在。研究人員和從業人員越來越依賴這些資源。這加速了研究進程，但也意味著這些框架內使用的特定配置和版本成為比較的 факти上標準。在報告結果時，引用這些來源及其特定設置至關重要。  

## II. 物件偵測模型

本節將深入探討各種物件偵測架構，比較它們的性能特點，主要使用 COCO 資料集作為通用基準，因為它在提供的研究摘要中出現頻率最高。

**表1：物件偵測模型比較 (主要基於 COCO 資料集)**

|模型 (變體)|骨幹網路 (Backbone)|參數 (M)|FLOPs (B/G)|輸入解析度|延遲 (ms) (硬體, 批次)|吞吐量 (FPS) (硬體, 批次)|mAP (COCO val/test-dev)|AP50|AP75|APS|APM|APL|來源|
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
|Faster R-CNN|ResNet-101-FPN (Caffe)|58.2 (total)|849.91 (G) (for sparse)|800x1280 (for sparse)|N/S|11.5 (V100, b=?) (MMDet v1.2)|38.8 (val) (MMDet v1.2) / 38.74 (val) (sparse)|N/S|N/S|N/S|N/S|N/S||
|Faster R-CNN|ResNet-101-FPN (PyTorch)|60.1 (total)|N/S|N/S|N/S|11.9 (V100, b=?) (MMDet v1.2)|38.5 (val) (MMDet v1.2)|N/S|N/S|N/S|N/S|N/S||
|Mask R-CNN (偵測部分)|ResNet-50-FPN (PyTorch)|44|262 (G) (MMDet v2.21) / 447 (B) (TorchVision)|800x1333 (MMDet)|51.02 (V100, b=1, MMDet v2.21) / 90.3 (V100, b=1, TorchVision)|19.6 (V100, b=1, MMDet v2.21) / 11.07 (V100, b=1, TorchVision)|38.2 (val) (MMDet v2.21) / 38.8 (val) (MMDet latest)|N/S|N/S|N/S|N/S|N/S||
|CenterMask2-Lite|VoVNetV2-19-FPN|11.2 (backbone)|N/S|N/S|23ms (V100, b=1)|~43.5 (V100, b=1)|35.9 (val)|N/S|N/S|N/S|N/S|N/S||
|CenterMask2|VoVNetV2-39-FPN|28.1 (backbone)|N/S|N/S|27ms (V100, b=1, maskrcnn-benchmark) / 28ms (V100, b=1, CenterMask2-Lite)|~37 (V100, b=1) / ~35.7 (V100, b=1)|40.7 (val) (maskrcnn-benchmark) / 40.9 (val) (CenterMask2-Lite)|N/S|N/S|N/S|N/S|N/S||
|CenterMask2|VoVNetV2-57-FPN|40.9 (backbone)|N/S|N/S|58ms (V100, b=1)|~17.2 (V100, b=1)|45.1 (val)|N/S|N/S|N/S|N/S|N/S||
|YOLOv8n|CSPDarknet|3.2|8.7 (B)|640x640|0.99 (A100 TensorRT, b=1)|1010 (A100 TensorRT, b=1)|37.3 (val)|N/S|N/S|N/S|N/S|N/S||
|YOLOv8l|CSPDarknet|43.7|165.2 (B)|640x640|2.39 (A100 TensorRT, b=1)|418 (A100 TensorRT, b=1)|52.9 (val)|N/S|N/S|N/S|N/S|N/S||
|YOLOv10-S|CSPDarknet-based|7.2|N/S|640x640|2.66 (T4 TensorRT, b=?)|~375 (T4 TensorRT, b=?)|46.7 (val)|N/S|N/S|N/S|N/S|N/S||
|FCOS|ResNet-50-FPN|32.27|128.21 (G)|800x1333 (default)|N/S|N/S|39.2 (val2017)|57.4|41.4|22.3|42.5|49.8||
|EfficientDet-D0|EfficientNet-B0|3.9|2.5 (B)|512x512|10.2 (V100, b=1)|97 (V100, b=1)|34.6 (test-dev)|53.0|37.1|12.4|39.0|52.7||
|EfficientDet-D1|EfficientNet-B1|6.6|6.1 (B)|640x640|13.5 (V100, b=1)|74 (V100, b=1)|40.5 (test-dev)|59.3|43.2|18.0|44.3|58.2||
|EfficientDet-D2|EfficientNet-B2|8.1|11.0 (B)|768x768|17.7 (V100, b=1)|57 (V100, b=1)|43.9 (test-dev)|62.8|47.0|22.1|47.6|60.6||
|EfficientDet-D4|EfficientNet-B4|20.7|55.2 (B)|1024x1024|42.8 (V100, b=1)|23 (V100, b=1)|49.7 (test-dev)|68.4|53.9|30.7|53.2|63.2||
|EfficientDet-D7|EfficientNet-B6 (scaled)|52 (paper) / 77 (github)|325 (B) (paper) / 410 (B) (github)|1536x1536|122 (V100, b=1)|8.2 (V100, b=1)|53.7 (test-dev) (paper) / 55.1 (test-dev) (github)|72.4|58.0|35.7|57.0|67.3||
|MobileNetV2-SSD FPNLite|MobileNetV2|4.33|0.81 (G)|320x320|22 (CPU/GPU unspecified)|~45 (CPU/GPU unspecified)|22.2 (COCO17 val)|N/S|N/S|N/S|N/S|N/S||
|MobileNet-SSD (v1)|MobileNetV1|N/S|N/S|300x300|N/S|N/S|24.8 (COCO AP) / 43.5 (COCO AP50)|N/S|N/S|N/S|N/S|N/S||
|DETR|ResNet-50|41|86 (G)|N/S|~35.7 (V100, b=1, implies 28 FPS)|28 (V100, b=1)|42.0 (val)|N/S|N/S|N/S|N/S|N/S||

 

_註：mAP 通常指 mAP@[.5:.95] IoU。Params 和 FLOPs 可能因具體實現和是否包含 FPN 或檢測頭而略有差異。延遲和吞吐量高度依賴硬體和批次大小。N/S 表示未指定。_

**A. Faster R-CNN (基於區域的卷積神經網路)**

- **架構概述**：Faster R-CNN 是一種開創性的兩階段物件偵測器。第一階段是區域提議網路 (Region Proposal Network, RPN)，用於提議候選物件區域。第二階段從這些提議中提取特徵（使用 RoIPool 或 RoIAlign），並執行分類和邊界框回歸 。它建立了一個強大的基線，並影響了許多後續設計 。  
    
- **性能指標**：
    - 使用 ResNet-101-FPN 作為主幹網路，在 MMDetection Model Zoo v1.2.0 中，PyTorch 版本的 Box AP 為 38.5%，V100 GPU 上的整體推論速度為 11.9 FPS；Caffe 版本的 Box AP 為 38.8%，速度為 11.5 FPS 。這些速度包括資料載入、網路前向傳播和後處理。  
        
    - Intel OpenVINO Model Zoo 中的一個稀疏化版本 (faster-rcnn-resnet101-coco-sparse-60-0001)，基於 TensorFlow 和 Detectron 實現，並應用了網路權重剪枝（60% 的網路參數設為零），在 COCO 驗證集上達到 38.74% mAP，參數量為 52.79 M，FLOPs 為 849.91 G，輸入尺寸為 1x800x1280x3 。  
        
    - MMDetection 在與 Detectron2 的比較中（使用 ResNet-50-FPN，1x 學習計畫，V100 GPU），Faster R-CNN 達到 38.0 mAP，純推論速度為 22.2 FPS 。  
        
- 實作細節和骨幹網路的選擇對 Faster R-CNN 的性能有重大影響。即使使用相似的 ResNet-101-FPN 骨幹網路，不同的實作（例如 MMDetection 中的 PyTorch 風格與 Caffe 風格 ）以及特定的優化（例如稀疏化 ）也會導致 mAP、速度和資源使用上的差異。核心的 Faster R-CNN 框架是一種元架構。骨幹網路的選擇（例如 ResNet-50、ResNet-101、ResNeXt）顯著改變參數數量、FLOPs 和特徵表示的品質。特徵金字塔網路 (FPN) 是一個常見的附加元件，可改善對不同尺度物件的偵測 。MMDetection 或 Detectron2 等框架內的訓練細節（學習計畫、資料增強、優化器）和微小的架構調整也會導致性能差異 。例如，Intel 的稀疏模型顯示剪枝等技術可以在保持具競爭力 mAP 的同時減少模型大小（52.79M 參數）。因此，僅僅提及「Faster R-CNN」是不夠的；必須考慮特定的骨幹網路、FPN 的使用、訓練計畫，甚至深度學習框架的版本，才能進行準確的比較。  
    

**B. YOLO (You Only Look Once) 家族**

- **架構概述**：YOLO 以其單階段偵測方法著稱，一次處理整個影像以直接預測邊界框和類別機率。這種設計優先考慮速度，使 YOLO 適用於即時應用 。後續版本整合了架構改進，如 CSPDarknet 骨幹網路、PANet 風格的頸部結構和無錨點 (anchor-free) 機制，以提高準確性和效率（例如 YOLOv8 ）。  
    
- **YOLOv8 (Ultralytics, COCO val2017, 640px 輸入)**：  
    
    - **YOLOv8n**: 37.3 mAP@50-95, 3.2M 參數, 8.7 BFLOPs, A100 TensorRT 上 0.99 ms/img, CPU ONNX 上 80.4 ms/img。
    - **YOLOv8s**: 44.9 mAP, 11.2M 參數, 28.6 BFLOPs, A100 TensorRT 上 1.20 ms/img, CPU ONNX 上 128.4 ms/img。
    - **YOLOv8m**: 50.2 mAP, 25.9M 參數, 78.9 BFLOPs, A100 TensorRT 上 1.83 ms/img, CPU ONNX 上 234.7 ms/img。
    - **YOLOv8l**: 52.9 mAP, 43.7M 參數, 165.2 BFLOPs, A100 TensorRT 上 2.39 ms/img, CPU ONNX 上 375.2 ms/img。
    - **YOLOv8x**: 53.9 mAP, 68.2M 參數, 257.8 BFLOPs, A100 TensorRT 上 3.53 ms/img, CPU ONNX 上 479.1 ms/img。
- **YOLO (通用 - 可能指早期版本如 v1-v3 )**：  
    
    - 初代 YOLO (2015)：155 FPS，52.7% mAP；後來的進階版本在 45 FPS 下達到 63.4% mAP 。這些早期數據的資料集和條件可能與目前的 COCO 基準不同。  
        
- **YOLOv10 (與 EfficientDet 比較 , T4 TensorRT 延遲)**：  
    
    - YOLOv10-S：46.7 mAP，7.2M 參數，2.66ms 延遲。
    - YOLOv10x：54.4 mAP，延遲 (12.2ms) 和 FLOPs 遠低於 EfficientDet-d7。
- **YOLOv11 (Ultralytics, COCO 資料集, 640px 輸入)**：  
    
    - YOLOv11n: 39.5 mAP@50-95, 2.6M 參數, 6.5 BFLOPs, T4 TensorRT 上 1.5 ms, CPU ONNX 上 56.1 ms。
    - YOLOv11x: 54.7 mAP@50-95, 56.9M 參數, 194.9 BFLOPs, T4 TensorRT 上 11.3 ms, CPU ONNX 上 462.8 ms。
- YOLO 系列（v8、v10、v11）提供了從 nano 到 extra-large 的廣泛模型選擇，始終在速度、準確度和模型大小之間展現出色的平衡 。在相似準確度下，它們的速度通常優於其他架構，如 EfficientDet 。YOLO 的單階段設計本質上比兩階段偵測器更快。持續的架構改進（無錨點設計、高效的骨幹/頸部網路、改進的損失函數 ）推動了準確度和速度的界限。多種模型尺寸（n、s、m、l、x）的可用性允許用戶根據應用需求（例如邊緣部署與雲端部署）進行明確的權衡。優化的實現（例如 Ultralytics 框架、TensorRT 支援）進一步增強了它們的實際性能。  
    

**C. FCOS (Fully Convolutional One-Stage Object Detection)**

- **架構概述**：FCOS 是一種全卷積單階段物件偵測器，以類似語義分割的逐像素預測方式解決物件偵測問題。與依賴預定義錨框的 RetinaNet、SSD、YOLOv3 和 Faster R-CNN 等主流偵測器不同，FCOS 是無錨框 (anchor-free) 和無提議 (proposal-free) 的 。透過消除預定義的錨框集，FCOS 完全避免了與錨框相關的複雜計算（例如訓練期間計算重疊），更重要的是，避免了所有與錨框相關的超參數，這些超參數通常對最終偵測性能非常敏感 。FCOS 的一個改進版本 FCOS-LSC，透過在網路結構中加入 LSC（層級尺度、空間、通道）注意力區塊，並使用帶有可變形卷積的 ResNet50 作為骨幹網路，在綠色水果偵測等複雜場景中取得了良好效果，參數量為 38.65M，FLOPs 為 38.72G 。  
    
- **性能指標 (FCOS ResNet-50-FPN, TorchVision, COCO val2017)**：  
    
    - Box mAP: 39.2
    - 參數: 32.27 M
    - GFLOPs: 128.21 (輸入大小未明確說明，但通常基於約 800x1333 的輸入)
    - AP50: 57.4, AP75: 41.4, APS: 22.3, APM: 42.5, APL: 49.8 (這些指標來自 PapersWithCode 上的 FCOS ResNet-50-FPN + 改進版本，COCO minival )。  
        
- FCOS 的無錨框設計代表了物件偵測領域的一個重要趨勢，旨在簡化偵測流程並減少超參數調整的複雜性。傳統基於錨框的方法需要仔細設計錨框的尺寸、長寬比和數量，這對不同資料集和任務可能不是最優的。FCOS 直接在特徵圖的每個位置預測邊界框和類別，並引入「中心度」(center-ness) 分支來抑制低品質的預測框 ，從而提高了性能。這種簡潔性使其更易於理解和實現，並且在某些情況下可以達到與基於錨框的方法相當甚至更好的性能，同時可能具有更快的推論速度，因為它減少了與錨框相關的計算開銷。FCOS-LSC 的例子表明，透過結合注意力機制和可變形卷積等先進技術，無錨框模型可以進一步提升在特定複雜場景下的魯棒性和準確性。  
    

**D. EfficientDet**

- **架構概述**：EfficientDet 是一系列物件偵測模型，其核心是 EfficientNet 骨幹網路和一種新穎的雙向特徵金字塔網路 (BiFPN) 。BiFPN 允許多尺度特徵之間進行簡單快速的融合。EfficientDet 還採用了複合縮放方法，可以同時統一縮放骨幹網路、特徵網路和預測網路的解析度、深度和寬度 。  
    
- **性能指標 (COCO test-dev, V100 GPU, batch=1)**：  
    
    - **EfficientDet-D0**: 34.6 AP, 3.9M 參數, 2.5 BFLOPs, 10.2ms 延遲 (97 FPS)。
    - **EfficientDet-D1**: 40.5 AP, 6.6M 參數, 6.1 BFLOPs, 13.5ms 延遲 (74 FPS)。
    - **EfficientDet-D2**: 43.9 AP , 8.1M 參數, 11 BFLOPs, 17.7ms 延遲 (57 FPS)。  
        
    - **EfficientDet-D3**: 47.2 AP (論文中為 47.5 AP), 12M 參數, 25 BFLOPs, 29.0ms 延遲 (36 FPS)。
    - **EfficientDet-D4**: 49.7 AP, 21M 參數, 55 BFLOPs, 42.8ms 延遲 (23 FPS)。
    - **EfficientDet-D5**: 51.5 AP, 34M 參數, 135 BFLOPs, 72.5ms 延遲 (14 FPS)。
    - **EfficientDet-D6**: 52.6 AP, 52M 參數, 226 BFLOPs, 92.8ms 延遲 (11 FPS)。
    - **EfficientDet-D7**: 53.7 AP, 52M 參數, 325 BFLOPs, 122ms 延遲 (8.2 FPS)。
    - **EfficientDet-D7x**: 55.1 AP, 77M 參數, 410 BFLOPs 。  
        
- **EfficientDet-Lite (COCO mAP, Mobile latency)**:  
    
    - **Lite0**: 26.41 mAP, 3.2M 參數, 36ms 延遲。
    - **Lite1**: 31.50 mAP, 4.2M 參數, 49ms 延遲。
    - **Lite2**: 35.06 mAP, 5.3M 參數, 69ms 延遲。
    - **Lite3**: 38.77 mAP, 8.4M 參數, 116ms 延遲。
    - **Lite4**: 43.18 mAP, 15.1M 參數, 260ms 延遲。
- EfficientDet 的設計哲學是在不同資源限制下實現高效率和高準確度的平衡。其核心創新 BiFPN 透過引入可學習的權重來決定不同輸入特徵的重要性，並結合自頂向下和自底向上的路徑進行多次特徵融合，從而比傳統的 FPN 和 PANet 更有效地利用多尺度資訊 。複合縮放策略確保了模型在擴展時，網路的各個維度（深度、寬度、解析度）能夠協調增長，避免了手動調整這些超參數的繁瑣過程，並能找到更優的準確度-效率權衡點。EfficientDet 系列模型（從 D0 到 D7/D7x 以及 Lite 版本）提供了一個廣泛的選擇範圍，使得開發者可以根據具體的硬體限制（如 GPU、CPU 或行動裝置）和性能需求（準確度 vs. 速度）來選擇最合適的模型。儘管在某些純速度比較中可能被後來的模型（如 YOLOv10 ）超越，EfficientDet 仍然是理解高效模型設計原則的一個重要里程碑。  
    

**E. MobileNet-SSD**

- **架構概述**：MobileNet-SSD 是一種為行動和嵌入式視覺應用設計的單階段物件偵測器 (Single Shot MultiBox Detector, SSD)，它使用 MobileNet 作為骨幹特徵提取網路 。MobileNet 本身以其深度可分離卷積 (depthwise separable convolutions) 而聞名，這種卷積方式能顯著減少計算量和參數數量，同時保持合理的準確度。SSD 框架則直接在骨幹網路的不同層級的特徵圖上預測物件類別和邊界框偏移。MobileNetV2-SSD FPNLite 是其一個變體，結合了 FPN-Lite 結構以改善對小物體的偵測 。  
    
- **性能指標**：
    - **MobileNetV1-SSD (COCO)**: 輸入 300x300。COCO AP 24.8%，AP50 43.5%。  
        
    - **MobileNetV2-SSD (Wolfram NeuralNet Repository, MS-COCO)**: 15.29M 參數。  
        
    - **SSD MobileNetV2 FPNLite 320x320 (TensorFlow 2 Object Detection Model Zoo, COCO 2017)**: mAP@[.5:.95] 22.2，速度 22ms (硬體未明確指定，但配置中提到 TPU-8 訓練)。  
        
    - **SSDLite with MobileNetV2 (參考比較)**: NAS-FPN 宣稱比其在行動裝置上準確度高 2 AP。  
        
- MobileNet-SSD 的核心優勢在於其計算效率，使其非常適合資源受限的環境，如行動電話或嵌入式系統。MobileNet 骨幹網路透過深度可分離卷積大幅減少了傳統卷積的計算成本。SSD 的單階段設計進一步確保了推論速度。然而，這種效率通常是以犧牲一些準確度為代價的，特別是與更大型、更複雜的兩階段偵測器相比。FPNLite 的引入 旨在部分緩解 SSD 在偵測不同尺度（尤其是小物體）方面的挑戰，透過輕量級的特徵金字塔來融合多尺度特徵。儘管其在 COCO 等大型資料集上的 mAP 可能不如頂級模型，但對於許多需要快速推論且對準確度要求不是極高的應用，MobileNet-SSD 及其變體仍然是一個有吸引力的選擇。  
    

**F. DETR (DEtection TRansformer)**

- **架構概述**：DETR (DEtection TRansformer) 引入了一種基於 Transformer 的端到端物件偵測方法，無需手動設計錨框或非極大值抑制 (NMS) 等後處理步驟 。它將物件偵測視為一個直接的集合預測問題，使用 Transformer 的編碼器-解碼器架構，並結合一個基於集合的全局損失函數，透過二分匹配 (bipartite matching) 強制進行唯一預測 。Deformable DETR 是其改進版本，透過引入可變形注意力模組，解決了 DETR 收斂緩慢和對小物件偵測性能較差的問題 。RF-DETR 是一個旨在實現即時性能的變體，宣稱在 COCO 上達到了 60+ mAP 。  
    
- **性能指標**：
    - **DETR (ResNet-50 backbone, COCO val)**: 42.0 AP, 41M 參數, 86 GFLOPs, V100 GPU 上 28 FPS。  
        
    - **Deformable DETR (ResNet-50 backbone, COCO)**: 與 DETR 相比，訓練速度快 10 倍，性能更好（尤其在小物件上）。  
        
    - **RF-DETR-large (728px input, COCO)**: 60.5 mAP，在 T4 GPU 上達到即時性能 (25+ FPS)。  
        
- DETR 的出現標誌著物件偵測領域的一個重要轉變，它首次展示了 Transformer 架構在端到端解決此類問題上的潛力。傳統的物件偵測器通常依賴於複雜的流程，包括錨框生成、區域提議、特徵池化和後處理（如 NMS）。DETR 透過其集合預測方法和 Transformer 的全局注意力機制，簡化了這一流程 。然而，最初的 DETR 模型存在訓練時間長和對小物件偵測不佳的缺點 。這主要是由於 Transformer 注意力模組在初始化時對特徵圖中的所有像素給予幾乎一致的注意力，需要長時間訓練才能學習到關注稀疏的關鍵位置，並且其編碼器中的注意力計算複雜度與像素數量成二次方關係 。Deformable DETR 透過引入可變形注意力，只關注一小部分採樣點，有效地緩解了這些問題，顯著加快了收斂速度並提高了對小物件的性能 。RF-DETR 等後續工作進一步探索了如何優化 DETR 類模型的速度和準確度平衡，使其在即時應用中更具競爭力。這些發展表明，基於 Transformer 的端到端方法是物件偵測未來的一個重要研究方向。  
    

## III. 影像分割模型

本節將討論用於影像分割的代表性模型，包括實例分割和語義分割。

**表2：影像分割模型比較**

|模型 (變體)|任務類型|骨幹網路 (Backbone)|參數 (M)|FLOPs (B/G)|輸入解析度|延遲 (ms) (硬體, 批次)|吞吐量 (FPS) (硬體, 批次)|主要指標 (數值) (資料集)|來源|
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
|Mask R-CNN|實例分割|ResNet-50-FPN (PyTorch)|44|262 (G) (MMDet v2.21) / 447 (B) (TorchVision)|800x1333 (MMDet)|51.02 (V100, b=1, MMDet v2.21) / 90.3 (V100, b=1, TorchVision)|19.6 (V100, b=1, MMDet v2.21) / 11.07 (V100, b=1, TorchVision)|mAPmask​ 34.7 (COCO val) (MMDet v2.21) / mAPmask​ 35.4 (COCO val) (MMDet latest) / mAPbox​ 38.2 (COCO val) (MMDet v2.21)||
|CenterMask2-Lite|實例分割|VoVNetV2-19-FPN|11.2 (backbone)|N/S|N/S|23ms (V100, b=1)|~43.5 (V100, b=1)|mAPmask​ 32.8 (COCO val) / mAPbox​ 35.9 (COCO val)||
|CenterMask2|實例分割|VoVNetV2-39-FPN|28.1 (backbone)|N/S|N/S|28ms (V100, b=1)|~35.7 (V100, b=1)|mAPmask​ 36.7 (COCO val) / mAPbox​ 40.9 (COCO val)||
|CenterMask2|實例分割|VoVNetV2-57-FPN|40.9 (backbone)|N/S|N/S|58ms (V100, b=1)|~17.2 (V100, b=1)|mAPmask​ 40.5 (COCO val) / mAPbox​ 45.1 (COCO val)||
|YOLOv8n-seg|實例分割|CSPDarknet|3.3 (HuggingFace) / 2.9 (Ultralytics)|9.2 (B) (HuggingFace) / 10.4 (B) (Ultralytics)|640x640|1.18 (A100 TensorRT, b=1) / 1.8 (T4 TensorRT, b=?)|~847 (A100 TensorRT, b=1) / ~555 (T4 TensorRT, b=?)|mAPmask​ 30.5 (COCO-Seg val) (HuggingFace) / mAPmask​ 32.0 (COCO-Seg val) (Ultralytics)||
|YOLOv8l-seg|實例分割|CSPDarknet|44.4 (HuggingFace) / 27.6 (Ultralytics)|168.6 (B) (HuggingFace) / 142.2 (B) (Ultralytics)|640x640|2.59 (A100 TensorRT, b=1) / 7.8 (T4 TensorRT, b=?)|~386 (A100 TensorRT, b=1) / ~128 (T4 TensorRT, b=?)|mAPmask​ 40.8 (COCO-Seg val) (HuggingFace) / mAPmask​ 42.9 (COCO-Seg val) (Ultralytics)||
|U-Net (原始)|語義/實例分割|Custom CNN|~31 (for 256x256 input)|~54 (G) (for 256x256 input)|512x512|<1000ms (NVIDIA Titan 6GB)|>1 (NVIDIA Titan 6GB)|IoU 92.03% (PhC-U373 ISBI) / IoU 77.56% (DIC-HeLa ISBI)||
|FCN-8s|語義分割|VGG16|~134 (VGG16 base)|N/S|500x500 (PASCAL VOC) / 512x512 (impl.)|~175ms (K40c, PASCAL test)|~5.7 (K40c, PASCAL test)|mIoU 67.2% (PASCAL VOC 2012 test) / mIoU 68.5% (PASCAL Plus impl.)||
|SAM (ViT-H)|任意分割|ViT-H/16|636 (Encoder) + Decoder|N/S (Encoder FLOPs high, Decoder ~50ms)|1024x1024 (typical)|~50ms (Decoder, browser CPU/GPU for prompt)|N/S|Zero-shot mIoU (16/23 datasets > RITM) / Zero-shot mask AP 44.7 (LVIS v1 val, ViTDet-H prompt)||
|SAM2 (Hiera-L)|任意分割 (影像/影片)|Hiera-L|224.4 (Total)|N/S|N/S|N/S (Video: 30.2 FPS on A100)|30.2 (Video, A100, b=1) / 61.4 (Image, A100, b=10)|J&F 90.7 (DAVIS 2017 val) / J&F 77.9 (MOSE val) / 6x faster than SAM (image)||
|SAM2 (Hiera-B+)|任意分割 (影像/影片)|Hiera-B+|N/S|N/S|N/S|N/S (Video: 43.8 FPS on A100)|43.8 (Video, A100, b=1) / 130.1 (Image, A100, b=10)|J&F 76.8 (SA-V val)||

 

_註：指標通常在特定資料集上報告，例如 COCO、Cityscapes、ISBI。參數和 FLOPs 取決於具體配置和輸入大小。延遲和吞吐量高度依賴硬體。N/S 表示未指定。SAM/SAM2 的延遲通常指解碼器部分，因其影像編碼器可預先計算。_

**A. Mask R-CNN**

- **架構概述**：Mask R-CNN 在 Faster R-CNN 的基礎上進行擴展，增加了一個平行的分支用於預測每個實例的物件遮罩，同時保留了原有的邊界框識別分支 。它引入了 RoIAlign 層以取代 RoIPool，從而實現更精確的像素級對齊，這對於生成高品質遮罩至關重要 。  
    
- **性能指標 (ResNet-50-FPN backbone, PyTorch, COCO val2017)**：
    - MMDetection (v2.21.0): mAPbox​ 38.2, mAPmask​ 34.7, 44M 參數, 262 GFLOPs (推測基於 800x1333 輸入), V100 GPU 上 19.6 FPS (純推論速度，批次大小為 1) 。  
        
    - TorchVision: mAPbox​ 37.2, mAPmask​ 33.9 (來自其模型卡，可能略有差異), 44M 參數, 447 BFLOPs, V100 GPU 上約 11 FPS (0.0903 s/im) 。  
        
    - 在 CityScapes 資料集上，改進的 Mask R-CNN 演算法達到了 62.62% mAPbox​ 和 57.58% mAPmask​，分別比原始 Mask R-CNN 提高了 4.73% 和 3.96% 。  
        
- Mask R-CNN 的設計使其能夠同時有效地偵測物件並為每個實例生成高品質的分割遮罩，這得益於其兩階段方法和 RoIAlign 的精確特徵提取 。第一個階段（RPN）生成候選物件區域，第二個階段則對這些區域進行分類、邊界框回歸和遮罩預測。這種架構的模組化特性使其易於擴展到其他任務，例如人體姿態估計 。儘管其實例分割效果良好，但在複雜交通場景等特定應用中，其泛化能力仍有提升空間 。不同框架（如 MMDetection 和 TorchVision）和骨幹網路的實現細節（如 ResNet-50 與 ResNet-101）會導致性能指標的差異。  
    

**B. CenterMask2**

- **架構概述**：CenterMask2 及其前身 CenterMask 是基於無錨框物件偵測器（如 FCOS 或 CenterNet）構建的單階段實例分割方法 。它旨在實現即時性能，特別是其輕量級版本 CenterMask-Lite。CenterMask2 通常使用 VoVNetV2 作為其骨幹網路，VoVNetV2 以其在速度和準確度之間的良好平衡而聞名 。  
    
- **性能指標 (COCO val2017, V100 GPU, batch=1)**：  
    
    - **CenterMask2-Lite (VoVNetV2-19-FPN, 4x sched)**: mAPmask​ 32.8, mAPbox​ 35.9, 11.2M (骨幹) 參數, 0.023s (23ms) 推論時間 (~43.5 FPS)。
    - **CenterMask2-Lite (VoVNetV2-19-Slim-FPN, 4x sched)**: mAPmask​ 29.8, mAPbox​ 32.5, 3.1M (骨幹) 參數, 0.021s (21ms) 推論時間 (~47.6 FPS)。
    - **CenterMask2-Lite (VoVNetV2-19Slim-DW-FPN, 4x sched)**: mAPmask​ 27.1, mAPbox​ 29.5, 1.8M (骨幹) 參數, 0.020s (20ms) 推論時間 (50 FPS)。
    - **CenterMask2 (VoVNetV2-39-FPN, 3x sched)**: mAPmask​ 39.7, mAPbox​ 44.2, 28.1M (骨幹) 參數, 0.050s (50ms) 推論時間 (20 FPS)。
    - **CenterMask2 (VoVNetV2-57-FPN, 3x sched)**: mAPmask​ 40.5, mAPbox​ 45.1, 40.9M (骨幹) 參數, 0.058s (58ms) 推論時間 (~17.2 FPS)。
    - **CenterMask2 (VoVNetV2-99-FPN, 3x sched)**: mAPmask​ 41.4, mAPbox​ 46.0, 84.0M (骨幹) 參數, 0.077s (77ms) 推論時間 (~13.0 FPS)。
    - 在 COCO 2% 標註資料的半監督實例分割基準上，CenterMask2 (ResNet50) 達到了 13.46 mAPmask​ 。  
        
- CenterMask 系列模型透過將實例分割分解為局部形狀預測和全局顯著性生成兩個並行子任務，有效地解決了單階段方法中區分重疊實例和像素級特徵對齊的挑戰 。其無錨框的特性簡化了設計並提高了效率。VoVNetV2 骨幹網路的引入，透過殘差連接和有效的 Squeeze-Excitation (eSE) 模組，進一步提升了模型的性能和速度 。CenterMask-Lite 版本特別針對即時應用進行了優化，在保持較高準確度的同時實現了高速推論，甚至在性能上超越了 YOLACT 等其他即時實例分割模型 。這使得 CenterMask2 成為需要在速度和準確度之間取得平衡的應用的有力競爭者。  
    

**C. YOLO (用於實例分割)**

- **架構概述**：YOLO 系列模型，如 YOLOv8 和 YOLOv11，也已擴展到實例分割任務，通常在其物件偵測頭的基礎上增加一個遮罩預測分支。它們繼承了 YOLO 在物件偵測方面的高效率和速度優勢。
- **性能指標 (COCO-Seg val2017, 640px input)**：  
    
    - **YOLOv8n-seg**: mAPmask​ 30.5 (HuggingFace) / 32.0 (Ultralytics), mAPbox​ 36.7 (HuggingFace) / 38.9 (Ultralytics), ~3M 參數, ~10 BFLOPs. A100 TensorRT 延遲 1.18ms (HuggingFace) / T4 TensorRT 延遲 1.8ms (Ultralytics)。
    - **YOLOv8s-seg**: mAPmask​ 36.8 (HuggingFace) / 37.8 (Ultralytics), mAPbox​ 44.6 (HuggingFace) / 46.6 (Ultralytics), ~11.5M 參數, ~32 BFLOPs. A100 TensorRT 延遲 1.45ms (HuggingFace) / T4 TensorRT 延遲 2.9ms (Ultralytics)。
    - **YOLOv8m-seg**: mAPmask​ 40.2 (HuggingFace) / 41.5 (Ultralytics), mAPbox​ 49.9 (HuggingFace) / 51.5 (Ultralytics), ~27M 參數, ~88 BFLOPs. A100 TensorRT 延遲 2.16ms (HuggingFace) / T4 TensorRT 延遲 6.3ms (Ultralytics)。
    - **YOLOv8l-seg**: mAPmask​ 40.8 (HuggingFace) / 42.9 (Ultralytics), mAPbox​ 52.3 (HuggingFace) / 53.4 (Ultralytics), ~44M 參數, ~155 BFLOPs. A100 TensorRT 延遲 2.59ms (HuggingFace) / T4 TensorRT 延遲 7.8ms (Ultralytics)。
    - **YOLOv8x-seg**: mAPmask​ 41.2 (HuggingFace) / 43.8 (Ultralytics), mAPbox​ 53.4 (HuggingFace) / 54.7 (Ultralytics), ~71M 參數, ~280 BFLOPs. A100 TensorRT 延遲 4.01ms (HuggingFace) / T4 TensorRT 延遲 15.8ms (Ultralytics)。
- YOLO 模型在實例分割任務上的應用，充分利用了其在物件偵測領域已建立的高效率和速度。透過在現有的高效偵測框架基礎上添加輕量級的遮罩預測頭，YOLO-seg 模型能夠在保持較高推論速度的同時，提供像素級的分割結果。這對於需要即時實例分割的應用場景（如機器人視覺、即時影片分析）尤其有價值。與 Mask R-CNN 等兩階段方法相比，YOLO-seg 通常在速度上有優勢，但在極高遮罩品質要求下可能略遜一籌。然而，隨著 YOLO 架構的不斷演進，其分割性能也在持續提升，使其成為一個在速度和分割準確度之間具有良好權衡的選擇。

**D. U-Net**

- **架構概述**：U-Net 是一種全卷積神經網路，最初為生物醫學影像分割而設計 。其特點是具有對稱的編碼器-解碼器結構，其中編碼器用於捕獲上下文資訊（特徵提取），解碼器則用於實現精確定位（上採樣並恢復解析度）。跳躍連接 (skip connections) 將編碼器中不同層級的特徵圖直接傳遞到解碼器的對應層級，有助於恢復細節並改善分割精度 。U-Net 及其變體（如 UNet++, Dense-UNet）已成為醫學影像分割和其他語義分割任務的流行選擇 。  
    
- **性能指標 (原始 U-Net )**：  
    
    - 參數: 約 31M (對於 256x256 輸入的標準 U-Net )。  
        
    - FLOPs: 約 54 GFLOPs (對於 256x256 輸入的標準 U-Net )。  
        
    - ISBI 細胞追蹤挑戰賽 2015:
        - PhC-U373 資料集: 平均 IoU 92.03% 。  
            
        - DIC-HeLa 資料集: 平均 IoU 77.56% 。  
            
    - 延遲: 512x512 影像分割在當時的 GPU (NVIDIA Titan 6GB) 上耗時小於 1 秒 。  
        
- U-Net 的核心優勢在於其有效的編碼器-解碼器結構和跳躍連接。編碼器路徑透過一系列卷積和池化層逐步降低空間維度並增加特徵通道數，從而捕獲圖像的上下文資訊。解碼器路徑則透過上採樣和卷積操作逐步恢復空間解析度。關鍵的跳躍連接將編碼器中較淺層的高解析度特徵圖與解碼器中對應的較深層的語義豐富特徵圖相結合 。這種設計使得網路能夠同時利用低層次的細節資訊和高層次的語義資訊，從而產生精確的分割結果，即使在訓練數據相對較少的情況下也是如此，這使其在生物醫學影像領域尤其成功 。標準 U-Net 雖然性能優越，但其參數量和計算成本對於資源受限的設備可能較高，因此催生了許多輕量化變體，旨在減少模型大小和計算複雜度，同時保持分割性能 。  
    

**E. FCN (Fully Convolutional Networks)**

- **架構概述**：FCN 是語義分割領域的開創性工作，它將傳統的分類 CNN（如 VGG, AlexNet, GoogLeNet）中的全連接層替換為全卷積層，從而使網路能夠接受任意大小的輸入並產生相應大小的像素級預測圖 。FCN 通常使用跳躍連接來融合來自不同深度層的特徵圖，以結合深層的語義資訊和淺層的外觀資訊，從而提高分割的精細度（例如 FCN-32s, FCN-16s, FCN-8s）。  
    
- **性能指標 (FCN-8s with VGG16 backbone, PASCAL VOC)**：
    - mIoU: 在 PASCAL VOC 2011/2012 上達到約 62.2% - 67.2% mIoU 。一個 TensorFlow 實現聲稱在 PASCAL VOC 2012 驗證集上達到 62.5% mIoU，在 PASCAL Plus (增強資料集) 上達到 68.5 mIoU 。  
        
    - 參數: VGG16 骨幹網路約有 134M 參數 。FCN-8s 在此基礎上增加少量參數。  
        
    - FLOPs: 未在原始論文中明確指出，但 VGG16 計算量較大 。推斷時間約為 175ms (K40c GPU) 。  
        
    - 一個基於 VGG16 的 FCN 實現，輸入 224x224，FCN-AlexNet 參數 57M，FCN-VGG16 參數 134M，FCN-GoogLeNet 參數 6M 。  
        
- FCN 的核心貢獻在於證明了深度卷積網路可以透過端到端訓練直接進行像素級的語義分割。透過將分類網路中的全連接層轉換為卷積層，FCN 能夠處理任意尺寸的輸入並輸出空間對應的熱圖 。為了克服深層網路中由於多次池化導致的空間資訊損失，FCN 引入了跳躍連接架構。例如，FCN-32s 直接對最後一個卷積層的輸出進行上採樣；FCN-16s 則結合了最後一層和 pool4 層的預測；FCN-8s 進一步結合了 pool3 層的預測 。這種多尺度特徵的融合使得 FCN 能夠生成更精細的分割結果。儘管 FCN 的感受野有限，可能無法有效捕獲全局上下文資訊 ，但它為後續的語義分割模型（如 U-Net、DeepLab 系列）奠定了重要基礎。  
    

**F. SAM (Segment Anything Model) 和 SAM2**

- **架構概述**：
    - **SAM**: 一個為影像分割設計的基礎模型，旨在透過提示工程（如點、框、文字或粗略遮罩）實現對任意物件的零樣本分割 。它由一個強大的影像編碼器（通常是 ViT-H ）、一個提示編碼器和一個輕量級的遮罩解碼器組成 。SAM 在包含超過 10 億個遮罩的 SA-1B 資料集上進行訓練 。  
        
    - **SAM2**: SAM 的升級版，專注於影像和影片中的高效可提示視覺分割 。SAM2 引入了流式記憶體機制以處理影片序列，並採用更高效的 Hiera 作為影像編碼器 。  
        
- **性能指標**：
    - **SAM (ViT-H/16 backbone )**:  
        
        - 參數: 影像編碼器 (ViT-H) 636M 。總參數更多，因為還包括提示編碼器和遮罩解碼器。  
            
        - FLOPs: ViT-H 編碼器計算量大，但具體 FLOPs 未在核心摘要中提供。遮罩解碼器推論速度快，約 50ms 。  
            
        - 性能: 在 LVIS v1.0 val 資料集上，使用 ViTDet-H 提示進行零樣本實例分割，達到 44.7 AP (mAPmask​) 。在 23 個不同分割資料集中的 16 個上，單點提示的 mIoU 超過 RITM 。  
            
    - **SAM2 (Hiera-L encoder)**:  
        
        - 參數: 224.4M 。  
            
        - FLOPs: 未明確提供 Hiera-L 的 FLOPs。
        - 性能: 在影像分割方面比 SAM 更準確且快 6 倍 。在影片分割方面，DAVIS 2017 val 上的 J&F 為 90.7，MOSE val 上的 J&F 為 77.9 。  
            
        - 吞吐量 (A100 GPU, bfloat16, torch.compile): 影像任務 (b=10) 61.4 FPS；影片任務 (b=1) 30.2 FPS 。  
            
    - **SAM2 (Hiera-B+ encoder)**:  
        
        - 參數: 未明確提供 Hiera-B+ 的參數，但 Hiera-B+ 比 Hiera-L 更小。
        - FLOPs: 未明確提供 Hiera-B+ 的 FLOPs。
        - 性能: SA-V val 上的 J&F 為 76.8 。  
            
        - 吞吐量 (A100 GPU, bfloat16, torch.compile): 影像任務 (b=10) 130.1 FPS；影片任務 (b=1) 43.8 FPS 。  
            
- SAM 和 SAM2 代表了影像分割領域向基礎模型的轉變。它們的核心思想是透過在大規模多樣化資料集 (SA-1B 包含超過 10 億個遮罩 ) 上進行預訓練，使模型能夠理解通用的分割概念，並透過簡單的提示（點、框、文字等）適應各種下游任務和新的影像分佈，而無需針對特定任務進行重新訓練（即零樣本遷移）。SAM 的架構設計，特別是將計算量大的影像編碼器與輕量級的提示編碼器和遮罩解碼器分離，使得在給定影像嵌入後能夠快速回應不同的提示，實現了互動式分割 。儘管 SAM 的分割品質在許多情況下令人印象深刻，但它在處理具有複雜結構的物件時，遮罩邊緣可能較粗糙 。HQ-SAM 等工作透過微小的架構調整（增加不到 0.5% 的參數）來提升 SAM 的分割品質，同時保持其效率和零樣本泛化能力 。SAM2 則進一步將這種能力擴展到影片領域，引入了流式記憶體機制來處理時間序列信息，並採用了更高效的 Hiera 編碼器，使其在影像分割上比 SAM 更快更準確，並在影片分割任務中展現出強大的性能 。這些基礎模型的出現極大地推動了通用視覺理解的發展，但也帶來了對模型壓縮和效率優化的新需求，以適應資源受限的部署場景，正如 TinySAM 等工作所探索的那樣。  
    

## IV. 物件追蹤模型

本節重點介紹多物件追蹤 (MOT) 模型，特別是 DeepSORT 及其相關評估指標。

**表3：物件追蹤模型比較 (主要基於 MOTChallenge)**

|模型 (變體)|偵測器 (Detector)|Re-ID 網路 (參數, FLOPs)|MOTA (%)|MOTP (%)|IDF1 (%)|HOTA (%)|FPS (Hz) (硬體)|資料集|來源|
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
|DeepSORT (原始論文)|Faster R-CNN (VGG16)|Custom CNN (2.8M, N/S)|61.4|79.1|N/S|N/S|~20 (GPU)|MOT16||
|DeepSORT + YOLOv3|YOLOv3|OSNet (或其他)|N/S|N/S|N/S|N/S|N/S|MOT17|(具體指標依賴實現)|
|LITE:DeepSORT|YOLOv8m|(整合入 YOLOv8m)|N/S|N/S|N/S|43.03|28.3 (RTX 3090)|MOT17||
|MOT_FCG++ (與 DeepSORT 比較)|N/S|N/S|76.9|N/S|78.2|63.1|N/S|MOT17|(DeepSORT 在此文中作為比較基準，其 MOTA 75.3, IDF1 77.3)|
|FeatureSORT (公開偵測)|公開|N/S|79.6|N/S|77.2|63.0|8.7 (N/S)|MOT17||
|ByteTrack (YOLOX detector)|YOLOX|N/S (關聯檢測框)|80.3|N/S|77.3|63.1|30 (V100)|MOT17||

 

_註：MOTA, MOTP, IDF1, HOTA 等指標的數值高度依賴於所使用的偵測器、Re-ID 模型以及特定的 MOTChallenge 資料集版本 (如 MOT16, MOT17, MOT20)。FPS 也與硬體和整體流程相關。N/S 表示未指定。_

**A. DeepSORT**

- **架構概述**：DeepSORT 是一種基於追蹤的偵測 (tracking-by-detection) 範式的多物件追蹤器。它擴展了 SORT (Simple Online and Realtime Tracking) 演算法，主要透過整合一個深度學習模型來提取外觀特徵 (appearance features)，從而改善在長時間遮擋情況下的身份保持能力 。其核心組件包括：卡爾曼濾波 (Kalman filter) 用於運動狀態預測，匈牙利演算法 (Hungarian algorithm) 用於數據關聯（匹配偵測與追蹤軌跡），以及一個用於計算外觀相似度的 Re-ID (Re-identification) 網路 。  
    
- **性能指標**：
    - 原始 DeepSORT 論文 在 MOT16 資料集上使用 Faster R-CNN (VGG16) 作為偵測器，其 Re-ID 網路有 2.8M 參數，報告 MOTA 為 61.4%，MOTP 為 79.1%，FPS 約為 20Hz (GPU)。IDF1 未在原始論文中報告。  
        
    - 在 MOT17 資料集上，LITE:DeepSORT (使用 YOLOv8m 作為偵測器並整合 Re-ID 特徵提取) 達到了 43.03% HOTA 和 28.3 FPS (NVIDIA RTX 3090) 。  
        
    - 一篇比較性研究 中，DeepSORT (作為基線) 在 MOT17 上的 MOTA 為 75.3%，IDF1 為 77.3%。  
        
    - MOTChallenge MOT17 排行榜上，FeatureSORT (使用公開偵測) 達到 MOTA 79.6%，IDF1 77.2%，HOTA 63.0%，FPS 8.7 。ByteTrack (使用 YOLOX 偵測器) 達到 MOTA 80.3%，IDF1 77.3%，HOTA 63.1%，FPS 30 (V100) 。  
        
- **Re-ID 網路的影響**：Re-ID 網路的選擇對 DeepSORT 的性能至關重要。OSNet 是一種流行的輕量級 Re-ID 網路，因其在準確度和效率之間的良好平衡而被廣泛應用於 DeepSORT 的各種實現中。OSNet_x1_0 版本的參數約為 2.2M，GFLOPs 約為 0.98 。  
    
- DeepSORT 的成功在很大程度上歸功於其將運動資訊（來自卡爾曼濾波）和外觀資訊（來自 Re-ID 網路）相結合的策略。運動資訊有助於短期預測和匹配，而外觀特徵則在物件被遮擋後重新出現時幫助重新識別，從而減少身份切換 。偵測器的品質對整體追蹤性能有顯著影響；高品質的偵測結果能為追蹤器提供更可靠的輸入，從而提高 MOTA 和 IDF1 等指標 。近年來，許多工作致力於改進 DeepSORT 的各個方面，包括使用更強大的偵測器（如 YOLO 系列）、更先進的 Re-ID 網路（如 OSNet），以及更有效的數據關聯策略。LITE:DeepSORT 透過將 Re-ID 特徵提取整合到偵測流程中，顯著提高了 DeepSORT 的運行速度，同時保持了相似的準確度。這表明，即使是經典的追蹤框架，在與現代高效組件結合後，仍然可以在性能和效率方面保持競爭力。  
    

**B. MOT 評估指標**

- **MOTA (Multiple Object Tracking Accuracy)**：計算公式為 1−GT(FN+FP+IDSW)​，其中 FN 是偽陰性數量，FP 是偽陽性數量，IDSW 是身份切換次數，GT 是真實物件的總數。MOTA 衡量追蹤器在保持正確軌跡方面的整體準確性，分數越高越好 。  
    
- **MOTP (Multiple Object Tracking Precision)**：衡量偵測到的物件與其對應的真實物件之間的平均不重疊度或距離。對於基於邊界框的追蹤，通常計算為 1−∑t​ct​∑t,i​dt,i​​，其中 dt,i​ 是第 t 影格中匹配對 i 的邊界框重疊度， ct​ 是第 t 影格中的匹配數量。MOTP 反映了追蹤器定位物件的精確度，分數越高越好（當定義為重疊度時）或越低越好（當定義為距離時）。  
    
- **IDF1 (ID F1 Score)**：將每個計算出的軌跡與真實軌跡進行比較，以確定身份的精確度 (IDP) 和召回率 (IDR)，然後計算它們的調和平均數：IDF1=IDP+IDR2×IDP×IDR​。IDF1 強調追蹤器在整個影片序列中正確維護物件身份的能力，是衡量長期追蹤性能的重要指標 。  
    
- **HOTA (Higher Order Tracking Accuracy)**：HOTA 旨在提供一個更平衡的指標，它將檢測準確度 (DetA) 和關聯準確度 (AssA) 分解開來，然後將它們組合成一個單一的最終分數：HOTA=DetA×AssA![](data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400em" height="1.08em" viewBox="0 0 400000 1080" preserveAspectRatio="xMinYMin slice"><path d="M95,702
    c-2.7,0,-7.17,-2.7,-13.5,-8c-5.8,-5.3,-9.5,-10,-9.5,-14
    c0,-2,0.3,-3.3,1,-4c1.3,-2.7,23.83,-20.7,67.5,-54
    c44.2,-33.3,65.8,-50.3,66.5,-51c1.3,-1.3,3,-2,5,-2c4.7,0,8.7,3.3,12,10
    s173,378,173,378c0.7,0,35.3,-71,104,-213c68.7,-142,137.5,-285,206.5,-429
    c69,-144,104.5,-217.7,106.5,-221
    l0 -0
    c5.3,-9.3,12,-14,20,-14
    H400000v40H845.2724
    s-225.272,467,-225.272,467s-235,486,-235,486c-2.7,4.7,-9,7,-19,7
    c-6,0,-10,-1,-12,-3s-194,-422,-194,-422s-65,47,-65,47z
    M834 80h400000v40h-400000z"></path></svg>)​。它能更好地捕捉檢測和關聯方面的性能 。  
    
- 其他相關指標還包括 MT (Mostly Tracked trajectories，大部分時間被追蹤的軌跡比例)、ML (Mostly Lost trajectories，大部分時間丟失的軌跡比例)、FP (False Positives)、FN (False Negatives)、IDsw (Identity Switches)、Frag (Fragmentations，軌跡斷裂次數) 等 。  
    
- 選擇合適的評估指標對於理解追蹤演算法的特定優勢和劣勢至關重要。例如，MOTA 對檢測錯誤（FP、FN）和身份切換都敏感，而 IDF1 更側重於關聯的準確性。HOTA 試圖提供一個更全面的視角，同時評估檢測和關聯的品質。在實際應用中，可能需要根據特定需求（例如，對身份保持的嚴格要求或對漏檢的容忍度）來權衡這些指標。

## V. 生成對抗網路 (GAN) 在視覺任務中的角色

生成對抗網路 (GAN) 是一種強大的生成模型，它透過兩個神經網路——生成器 (Generator) 和判別器 (Discriminator)——的對抗過程來學習數據的分佈 。生成器嘗試創建逼真的數據樣本，而判別器則努力區分真實數據和生成器生成的偽造數據。這種競爭機制使得 GAN 能夠生成高度逼真的合成數據。  

在物件偵測、影像分割和物件追蹤等電腦視覺任務中，GAN 主要扮演以下輔助角色：

1. **數據增強 (Data Augmentation)**： 高品質的標註數據對於訓練高性能的視覺模型至關重要，但獲取大規模標註數據集通常成本高昂且耗時 。GAN 可以生成逼真的合成影像及其對應的標註（例如，帶有邊界框或分割遮罩的影像），從而擴充訓練數據集 。這對於處理類別不平衡問題或罕見樣本特別有用。例如，tGAN 模型被用於生成帶有標註的逼真延時影片，以增強細胞追蹤模型的性能，減少對手動標註的依賴 。  
    
2. **領域自適應 (Domain Adaptation)**： 當訓練數據（源域）和測試數據（目標域）之間存在分佈差異時，模型的性能通常會下降。GAN 可以用於將源域的影像風格轉換為目標域的風格，或者反之，從而減少域差異，提高模型在目標域的泛化能力。例如，CycleGAN 等模型可用於無監督的影像到影像翻譯 。  
    
3. **超解析度 (Super-Resolution)**： 在某些情況下，低解析度的影像可能會影響偵測或分割的準確性。基於 GAN 的超解析度模型（如 SRGAN）可以將低解析度影像提升至高解析度，同時生成更豐富的細節，從而間接提升下游任務的性能 。  
    
4. **影像修復/補全 (Image Inpainting/Completion)**： 在物件追蹤等任務中，物件可能被部分遮擋。GAN 可以用於修復影像中的缺失部分或被遮擋的區域，從而為追蹤器提供更完整的物件外觀資訊 。  
    
5. **模擬器與合成數據生成**： GAN 可以創建高度逼真的模擬環境和合成數據，用於訓練和測試視覺模型，尤其是在難以獲取真實數據或真實數據帶有隱私問題的場景（如醫學影像、金融數據）。  
    

**評估 GAN 的指標**： 評估 GAN 生成樣本的品質是一個持續的研究課題。常用的指標包括 Inception Score (IS)、Fréchet Inception Distance (FID)、Learned Perceptual Image Patch Similarity (LPIPS) 等，它們試圖從不同角度衡量生成樣本的逼真度和多樣性 。然而，標準化評估指標的缺乏仍然是 GAN 領域的一個挑戰 。  

**挑戰與未來方向**： 儘管 GAN 取得了巨大成功，但其訓練過程可能不穩定，容易出現模式崩潰 (mode collapse) 等問題 。未來的研究方向包括提高訓練穩定性、開發更魯棒的評估基準、以及整合隱私增強技術 。  

總體而言，GAN 為電腦視覺領域提供了一種強大的工具，透過生成高品質的合成數據來輔助和增強現有的偵測、分割和追蹤模型的性能，尤其是在數據稀缺或標註成本高昂的情況下。

## VI. 結論與建議

本報告對常用的物件偵測、影像分割和物件追蹤模型進行了全面的比較分析，涵蓋了它們的架構特點、性能指標（如 mAP、mIoU、MOTA、IDF1）、資源消耗（參數數量、FLOPs）以及運行效率（延遲、吞吐量）。

**主要發現：**

1. **模型演進趨勢**：
    
    - **物件偵測**：從兩階段的 Faster R-CNN 演變為更高效的單階段偵測器（如 YOLO 系列、FCOS、EfficientDet），並且無錨框設計（如 FCOS、YOLOv8 的部分變體）和基於 Transformer 的端到端偵測器（如 DETR）成為重要趨勢。
    - **影像分割**：U-Net 及其變體在醫學影像分割領域佔據主導地位，而 Mask R-CNN 仍然是實例分割的強大基線。基礎模型如 SAM 和 SAM2 透過提示式交互和零樣本泛化能力，正在改變分割任務的範式。
    - **物件追蹤**：基於「偵測後追蹤」的範式仍然主流，DeepSORT 結合強大的偵測器和 Re-ID 模組是常用方案。SAM2 等模型也開始涉足影片分割與追蹤。
    - **骨幹網路的協同進化**：從 VGG、ResNet 到 MobileNet、EfficientNet、VoVNetV2，再到 Vision Transformer (ViT) 和 Hiera 等更高效、表徵能力更強的骨幹網路，直接推動了下游任務性能和效率的提升。
2. **性能與效率的權衡**：
    
    - 不存在適用於所有場景的「最佳」模型。模型選擇需要在準確度、速度和資源消耗之間進行權衡。
    - YOLO 系列（尤其是較小變體）、EfficientDet-Lite、MobileNet-SSD 和 CenterMask-Lite 在即時性能和資源受限場景中表現出色。
    - 對於追求最高準確度的應用，較大的 EfficientDet 變體、YOLOv8/v10 的大型版本、DETR 及其改進型，以及 Mask R-CNN 和 CenterMask2 的較大版本是主要選擇。
3. **基礎模型的影響**：
    
    - SAM/SAM2 等基礎模型展示了強大的零樣本泛化能力，減少了對特定任務訓練數據的依賴。然而，它們的計算成本較高，催生了模型壓縮、量化（如 TinySAM）和高效編碼器（如 SAM2 中的 Hiera）的研究。這表明，雖然基礎模型潛力巨大，但其實際部署往往需要進一步的優化。
4. **基準測試的重要性**：
    
    - 標準化資料集（如 COCO, MOTChallenge, Cityscapes, ISBI）和評估指標對於模型比較至關重要。
    - 延遲和吞吐量等指標高度依賴於硬體和軟體環境，在比較時必須註明上下文。
    - 模型庫（如 MMDetection, Ultralytics）和基準測試平台（如 Papers with Code）在推動研究和提供可比較結果方面發揮著核心作用。

**模型選擇建議：**

- **追求最高準確度 (COCO 基準)**：
    - **物件偵測**: EfficientDet-D7/D7x , YOLOv8x/YOLOv10x , DETR 及其進階版本 。  
        
    - **實例分割**: Mask R-CNN (配合強大骨幹) , CenterMask2 (較大 VoVNetV2 變體) , 或 SAM/SAM2 (用於零樣本/可提示任務) 。  
        
- **重視即時性能 (高 FPS)**：
    - **物件偵測/實例分割**: YOLO 系列 (尤其是 n/s/m 型號) , EfficientDet-Lite 系列 , CenterMask2-Lite , MobileNet-SSD 。  
        
- **資源受限環境 (行動/邊緣裝置)**：
    - **物件偵測/實例分割**: MobileNet-SSD , EfficientDet-Lite , TinySAM , YOLO (n/s 型號) 。  
        
- **生物醫學影像分割**：
    - U-Net 及其變體因其在小數據集上的良好性能而持續受到青睞 。SAM/SAM2 也正被積極地應用和調整於此領域 。  
        
- **影片物件追蹤**：
    - DeepSORT 配合強大的偵測器 (如 YOLOv8/v10 ) 和魯棒的 Re-ID 模組 (如 OSNet )。  
        
    - SAM2 提供了可提示的影片分割能力，為影片追蹤開闢了新途徑 。  
        

**未來展望：** 電腦視覺模型將繼續朝著更高準確度、更高效率和更強泛化能力的方向發展。基於 Transformer 的架構和基礎模型預計將發揮越來越重要的作用。影片理解、更魯棒的 Re-ID 方法以及利用 GAN 等技術生成高質量合成數據以減少對大規模手動標註數據的依賴，將是未來研究的重點。同時，標準化硬體和軟體環境下的基準測試對於推動領域的健康發展至關重要。

總之，選擇合適的視覺模型需要仔細考量特定應用的需求、可用的計算資源以及對準確度和速度的期望。本報告提供的表格化數據和分析旨在為這一決策過程提供有價值的參考。
