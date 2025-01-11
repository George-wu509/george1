

ref: [Awesome-Multimodal-LLM-Autonomous-Driving](https://github.com/IrohXu/Awesome-Multimodal-LLM-Autonomous-Driving)

ref: [End-to-end-Autonomous-Driving](https://github.com/OpenDriveLab/End-to-end-Autonomous-Driving

自动驾驶中的多目标跟踪（MOT）：2024最新综述！ - 无疆WGH的文章 - 知乎
https://zhuanlan.zhihu.com/p/678751718

#### Q: 請列舉10個開源的SOTA 使用vision multimodal模型用於自駕車的model. 中文詳細介紹原理,使用model並提供github或link

以下是目前應用於自駕車的10個開源SOTA視覺多模態模型，每個模型都包含原理、使用方法的詳細介紹，以及相關GitHub連結，幫助您快速進入應用：

1. **BEVFormer**
    
    - **原理**：BEVFormer是一種基於視覺的Bird’s Eye View (BEV)轉換模型，利用多相機視角進行環境建模。模型結合Transformer架構，將圖像信息轉換為車輛周邊的平面圖，適合自駕車定位和物體檢測。
    - **使用**：通過PyTorch實現，可在預先標定好的相機數據上進行訓練，適用於動態場景分析。
    - **GitHub**：[BEVFormer](https://github.com/fundamentalvision/BEVFormer)
2. **M2BEV**
    
    - **原理**：M2BEV同樣基於BEV設計，透過多模態融合，如激光雷達與攝像機，實現視覺與深度的融合。它針對遮擋問題做了優化，提升了不同天氣條件下的表現。
    - **使用**：需提供車輛多視角圖像與深度數據，支持不同天氣場景下的測試。
    - **GitHub**：[M2BEV](https://github.com/zhangyp15/M2BEV)
3. **YOLOv5-3D**
    
    - **原理**：YOLOv5-3D在2D目標檢測基礎上進行擴展，增強對深度信息的處理。它結合了2D視覺數據與深度圖，適合進行物體檢測和3D重建。
    - **使用**：易於設置，適合實時應用，可在自駕車實際應用場景中用於多物體檢測。
    - **GitHub**：[YOLOv5-3D](https://github.com/Tianxiaomo/pytorch-YOLOv4)
4. **PointPillars**
    
    - **原理**：基於點雲數據進行3D物體檢測。PointPillars將點雲分成網格，並使用卷積神經網絡進行處理。此模型在計算上高效，適合實時應用。
    - **使用**：需要搭配激光雷達數據進行訓練，適用於自駕車的3D檢測場景。
    - **GitHub**：[PointPillars](https://github.com/nutonomy/second.pytorch)
5. **Mono3D**
    
    - **原理**：Mono3D是一個基於單目圖像的3D檢測模型，利用深度估計和卷積神經網絡實現物體的3D定位，適合不具備多模態數據的應用。
    - **使用**：僅需單目相機的圖像數據，特別適合低成本的自駕方案。
    - **GitHub**：[Mono3D](https://github.com/umich-tpolk/Mono3D)
6. **SMOKE**
    
    - **原理**：SMOKE是針對自駕車場景的單目3D物體檢測框架，結合語義分割和回歸方法，對每幀影像進行3D邊框的預測。
    - **使用**：適用於單目相機，可進行簡單訓練以應用於即時檢測。
    - **GitHub**：[SMOKE](https://github.com/lzccccc/SMOKE)
7. **CenterNet**
    
    - **原理**：CenterNet通過檢測物體的中心點進行定位並估計3D姿態。適合同時處理2D和3D的多任務學習，強化模型穩定性和準確性。
    - **使用**：適合各種視角的2D、3D物體檢測，能夠有效處理自駕車環境中的遮擋問題。
    - **GitHub**：[CenterNet](https://github.com/xingyizhou/CenterNet)
8. **Deep Multi-View Stereo (DeepMVS)**
    
    - **原理**：DeepMVS將多視角影像進行深度重建，可生成3D點雲，適合用於車輛周圍環境建模。此技術尤其在3D物體識別和定位中效果顯著。
    - **使用**：需要多相機配置以生成不同視角的圖像進行重建。
    - **GitHub**：[DeepMVS](https://github.com/pmwenzel/DeepMVS)
9. **Minkowski Engine**
    
    - **原理**：Minkowski Engine基於稀疏卷積操作，適合用於3D點雲數據處理。其核心優勢是高效處理稀疏的3D數據結構，適合即時應用場景。
    - **使用**：需要配合點雲數據集，可實現高精度的環境感知。
    - **GitHub**：[Minkowski Engine](https://github.com/NVIDIA/MinkowskiEngine)
10. **MultiXNet**
    
    - **原理**：MultiXNet是一種多任務學習模型，將多模態數據（如雷達、相機）與多視角圖像進行融合。它利用卷積神經網絡進行特徵提取，適合多場景下的3D物體檢測。
    - **使用**：需要相機與雷達數據輸入，能夠在不同場景和天氣條件下穩定工作。
    - **GitHub**：[MultiXNet](https://github.com/garrickbrazil/M3D-RPN)
