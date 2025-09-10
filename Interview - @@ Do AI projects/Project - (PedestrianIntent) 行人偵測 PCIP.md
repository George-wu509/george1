
```
請幫我詳細規劃以下的研究計畫進行行人過街「意圖」預測（Intention）—包括video的object detection + keypoint detection + 姿態軌跡＋動作偵測融合 project分析PIE (Pedestrian Intention Estimation Dataset), 並有行人視線與注意力 (Gaze & Head Orientation) 分析, 可使用預訓練人臉關鍵點檢測模型 (如 MMPose Whole-body), 「視線落點」精確估計，可以用 ETHXGaze 或 MPIIGaze 預訓練模型，直接 inference. 也進行場景語義增強 (Scene Context, Semantic Segmentation)直接使用 SAM or SAM2,做場景分割以增強行人過街「意圖」預測. 

希望這是放在Github做成一個python library. 並要有很多及完整對每個功能的過程及結果的各式data visualization functions, 也要針對效能進行優化. 請提供每個functions的code也請提供github 英文readme, 以及安裝python environment的方便方法譬如pyproject.toml, yaml file. 並附上兩個colab範例程式碼python code. 
```

Github: https://github.com/George-wu509/PedestrianIntent

Resume:
**Enhanced Pedestrian Intention Prediction based on Dynamic Behavior Decoding**
**基於動態行為解碼的行人意圖預測增強技術 (PedestrianIntent)**

Developed a comprehensive analysis pipeline that predicts pedestrian crossing intent from video. The system leverages foundation models for real-time, zero-shot scene understanding, integrating multi-modal inputs including pose estimation, gaze direction, and trajectory to provide actionable insights for autonomous vehicle systems and urban safety analysis.
開發了一個完整的系統流程，能夠從影片中預測行人過街意圖。該系統利用基礎模型實現即時、零樣本場景理解，並整合多種模式的輸入數據，包括姿態和視線估計以及軌跡數據，從而為自動駕駛系統和城市安全分析提供有價值的洞察資訊。



|                                                           |     |
| --------------------------------------------------------- | --- |
| [[#### PIE (Pedestrian Intention Estimation) dataset 介紹]] |     |
| [[#### PedestrianIntent Project的idea]]                    |     |
| [[#### PedestrianIntent 的行人動態狀態估計：軌跡與姿態]]                 |     |
| [[#### PedestrianIntent 的行人專注與注意力：凝視與頭部姿態]]               |     |
| [[#### PedestrianIntent 透過零樣本語義分割實現場景情境化]]                |     |
| [[#### 創建場景上下文特徵 (Scene Context Features)]]               |     |
| [[#### PIE dataset 可延伸的 8 個新方向]]                          |     |

**PCIP (Pedestrian Crossing Intention Prediction) 行人過街意圖預測**

|                                                      |                                                                                                                                                                                                                                                                                                                                                                                          |
| ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Step1**: 目標檢測(行人) +tracking                        | YOLO8+SORT or Grounded SAM2                                                                                                                                                                                                                                                                                                                                                              |
| **Step2:** Keypoint detection + Pose Estimation 姿態估計 | ==MMPose==<br><br>MMPose (Whole-Body (133 keypoints) Pose Estimation) - 輸入行人的邊界框, MMPose就可以自行detect 被偵測行人的133個關鍵點及其對應的信賴度分數, 輸出是一個數據結構列表.<br><br>數據結構：輸出的數據格式通常是一個字典列表，每個字典對應一個行人，其中包含一個名為`'keypoints'`的鍵，其值為一個形狀為`(133, 3)`的NumPy陣列，每一行代表一個關鍵點的`(x, y, confidence)`                                                                                                                     |
| **Step3**: Gaze estimation 凝視估計                      | ==ETH-XGaze==<br><br>ETH-XGaze - 凝視估計模型通常需要一個裁剪好的人臉圖像作為輸入 。我們將利用由MMPose提取的臉部關鍵點(ETH-XGaze模型需要224x224像素). 載入預訓練的凝視模型，並對裁剪後的人臉圖像進行推論。模型的原始輸出通常是一個表示俯仰角（pitch）和偏航角（yaw）的二維向量<br><br>作為凝視的補充特徵，頭部的三維朝向也能提供關於注意力方向的粗略但有價值的資訊。這可以透過使用MMPose模型提供的3D臉部標記點，或利用2D臉部關鍵點和一個通用的3D人臉模型，通過解決PnP（Perspective-n-Point）問題來估計                                                                               |
| **Step4**: segmentation場景分割                          | SAM (SamAutomaticMaskGenerator) + CLIP<br><br>SAM生成的遮罩是類別無關的（class-agnostic）。我們將使用CLIP（Contrastive Language-Image Pre-training）模型為每個遮罩賦予語義標籤. 在SAM和CLIP之間引入一個「遮罩後處理」將首先過濾掉面積過小的遮罩. 最終輸出是一個場景的語義地圖<br><br>                                                                                                                                                                                |
| **Step5**: Intention Inference 意圖推斷                  | ==場景上下文 (Scene Context)== - 除了 行人本身的外觀 (bounding box, pose, keypoints, trajectory)，還需要考慮行人所處的 環境資訊. Scene Context包括靜態語義資訊 (Static Semantic Context), 動態語義資訊 (Dynamic Context), 交互資訊 (Interaction Context)<br><br>當所有必要的視覺基元（primitive features）都被提取出來後（行人軌跡、姿態、視線、與斑馬線的距離等），最後的意圖預測模組可以是一個輕量級的時間序列模型（如 LSTM, Transformer）或一個基於規則的邏輯引擎。甚至，可以將這些描述性特徵轉換為一段文字，交給一個大型視覺語言模型（VLM）來進行更高層次的因果推斷 |
|                                                      |                                                                                                                                                                                                                                                                                                                                                                                          |
**預訓練感知模型比較**

| 模態 (Modality) | 模型 (Model)      | 核心優勢 (Key Strengths) | 輸入格式 (Input Format) | 輸出格式 (Output Format)   | 相對延遲 (Relative Latency) |
| ------------- | --------------- | -------------------- | ------------------- | ---------------------- | ----------------------- |
| 姿態估計          | DWPose (MMPose) | 全身關鍵點偵測精度高，包含臉部與手部   | 圖像 + 邊界框            | 133個關鍵點 `(x, y, conf)` | 中                       |
| 凝視估計          | ETH-XGaze       | 對極端頭部姿態具有魯棒性         | 224x224 人臉裁剪圖       | 俯仰角/偏航角向量              | 低                       |
| 場景分割          | SAM + CLIP      | 零樣本能力強，無需針對特定場景進行訓練  | 原始圖像                | 帶有語義標籤的遮罩列表            | 高                       |
### 新範式下的核心理念

1. **偵測即提示 (Detection as Prompting)**: 我們不再需要在標註了「行人」、「汽車」的資料集上訓練一個 YOLO 或 Faster R-CNN 模型。取而代之，我們直接向 Grounding DINO 提供文字提示，如 `"pedestrian"`, `"car"`, `"traffic light"`, `"crosswalk"`，它就能在零樣本（zero-shot）的情況下找出這些物體的位置。
    
2. **分割即提示 (Segmentation as Prompting)**: 結合 Grounded SAM，我們可以將 Grounding DINO 找到的物體邊界框（bounding box）作為提示，進而獲得該物體像素級的精確分割遮罩（mask）。這對於分析行人與場景元素的精確空間關係（例如，行人是否踩在斑馬線上）至關重要。
    
3. **追蹤即分割 (Tracking as Segmentation)**: SAM2 的引入是關鍵。它能將影片中的物體分割任務從單一圖片擴展到時間序列。我們可以透過在第一幀提示 `"the person in the blue shirt"`，SAM2 就能在後續的影片幀中持續追蹤並分割這個特定的人，即使他/她被短暫遮擋。這為我們提供了一種非常穩健的方式來獲取**行人軌跡**（透過計算每幀遮罩的質心）。
    
4. **屬性提取的流水線 (Pipeline for Attribute Extraction)**: 對於姿態和視線這類更細微的屬性，我們仍然可以利用專用的預訓練模型（如 MMPose, ETH-XGaze）。但不同的是，我們不再需要自己訓練行人偵測器來觸發它們。Grounding DINO 可以先用 `"person"` 和 `"head"` 提示來定位，然後將這些區域裁切出來，送入對應的姿態和視線模型。整個過程無需任何自訂訓練。
    
5. **意圖推斷 (Intention Inference)**: 當所有必要的視覺基元（primitive features）都被提取出來後（行人軌跡、姿態、視線、與斑馬線的距離等），最後的意圖預測模組可以是一個輕量級的時間序列模型（如 LSTM, Transformer）或一個基於規則的邏輯引擎。甚至，可以將這些描述性特徵轉換為一段文字，交給一個大型視覺語言模型（VLM）來進行更高層次的因果推斷






#### PIE (Pedestrian Intention Estimation) dataset 介紹

```
針對PIE (Pedestrian Intention Estimation) dataset, 除了可以做object detection + keypoint detection + 姿態軌跡＋動作偵測融合之外, 還有甚麼相關的分析可以進行, 也請提供8個 new ideas可以用新技術更好的使用這個dataset
```
PIE (Pedestrian Intention Estimation) dataset 本身就是專門為 **行人過街意圖 (Pedestrian Crossing Intention)** 而設計的，它除了 **物件偵測 + 關鍵點偵測 + 姿態軌跡 + 動作偵測融合** 之外，其實還有很多延伸分析空間。下面我幫你整理 **8 個新的 research / project ideas**，結合新技術，讓你能把 PIE 用在更廣的場景。

| 資料集                                              | 特點                                            | 適合 PCIP 的理由                       | 是否有多視角/多感測器 |
| ------------------------------------------------ | --------------------------------------------- | --------------------------------- | ----------- |
| **PIE (Pedestrian Intention Estimation)**        | 車前行車紀錄器影片 (ego-view)，有 bounding box、pose、意圖標註 | ✅ 有明確的「過街意圖」標籤，是 PCIP 的黃金標準       | 單視角 (前方攝影)  |
| **JAAD (Joint Attention in Autonomous Driving)** | 行車紀錄器影片，標註行人動作 (看左右、走/停)                      | ✅ 有「head orientation、動作」標籤，可做意圖推斷 | 單視角         |
[PIE ](https://data.nvision2.eecs.yorku.ca/PIE_dataset/)   **(Pedestrian Intention Estimation) 資料集** [github](https://github.com/aras62/PIE) [download](https://drive.google.com/drive/folders/180MXX1z3aicZMwYu2pCM0TamzUKT0L16?usp=drive_link)
PIE 包含 53 個影片片段，分為 6 組。每組中的片段都是連續的

- **空間標註（Spatial Annotations）**：數據集提供了行人、與自我車輛有潛在互動的車輛，以及關鍵基礎設施（如交通號誌、交通標誌、斑馬線）的邊界框（bounding boxes）。這些數據是物體偵測、軌跡追蹤與場景理解的基礎。  
- 六種類型的物件（行人、車輛、交通號誌、標誌、行人穿越道、公車站）以邊界框標註. 每個物件都有一個唯一的 ID，格式為`<set_id>_<video_id>_<object_id>`，例如，在集合 1 的影片 2 中發現了 ID 為 的行人。`1_2_50`交通基礎設施物件的物件 ID 也以對應物件的首字母標記，例如`tl`交通號誌. 根據物件的類型，為每個邊界框提供附加資訊 譬如:
		行人: 行動(walking/standing), 手勢, 觀察(looking or not), 過馬路
		車輛: 車輛類型
		交通燈: 類型, 狀態
		符號: 標誌的類型
		行人穿越道: 有無
		交通站: 有無
    
- **行為標註（Behavioral Annotations）**：針對每個行人，數據集提供了逐幀的行為標籤，例如`walking`（行走）、`standing`（站立）、`looking`（觀看）以及至關重要的`crossing`（穿越）標籤 。這些標籤為監督式學習提供了直接的行為線索。  
    
- **意圖標註（Intention Annotations）**：此為PIE數據集的核心。行人的「穿越意圖」並非一個簡單的二元標籤，而是透過大規模人為實驗，匯總多位標註者反應後得出的機率值 。這一細微之處對於模型訓練具有深遠影響。它暗示了一個簡單的二元交叉熵損失函數可能不是最優選擇，而一個能夠處理軟標籤（soft labels）或直接進行機率回歸的損失函數可能更能捕捉人類判斷中的不確定性。  
    
- **自我車輛數據（Ego-Vehicle Data）**：與影片幀同步的車載診斷系統（OBD）感測器數據，包括車速、航向和GPS座標 。這些數據提供了關於記錄車輛動態的關鍵上下文，是許多現有模型的重要輸入





#### PedestrianIntent Project的idea

對PIE標註的分析顯示，行人的姿態、周圍環境上下文以及在短時間內（例如最後16幀，約半秒）的邊界框資訊，對於意圖預測具有高度預測性 。然而，深入檢視後，該數據集亦存在一些關鍵的局限性，
1. 沒有Gaze estimation(凝視預測)儘管數據集包含`looking`的行為標籤，但它並未提供明確的凝視向量（gaze vector）或頭部姿態角度.
2. 沒有Pose Estimation(人體姿態預測)PIE論文中提及的基準模型使用了姿態資訊 ，但公開的標註主要集中於邊界框和行為標籤。所以一個強大的全身姿態估計模型不僅是特徵增強的選項，更是必要的組成部分. 
3. Real-world vs. Annotation. 多項研究指出，在真實世界的應用中，依賴即時感知系統（如物體偵測器）的輸出，其性能通常會遜於使用數據集提供的真實標註（ground truth）進行訓練和測試的模型 。本專案計畫明確旨在彌合這一差距，透過建構一個從原始影片中提取所有特徵的完整流程
4. PIE數據集的意圖標籤是從人類標註者的反應中匯總而來的機率值 。一個直接的處理方法是設定一個閾值（例如，大於0.5即視為「意圖穿越」），並將其簡化為一個二元分類問題。然而，這種方法丟棄了關於模糊性的寶貴資訊。一個0.51的機率值與一個0.99的機率值在人類共識的確定性上存在巨大差異。這啟示我們，模型的損失函數應能處理軟目標。使用模型輸出分佈與人類標註者分佈之間的KL散度（Kullback-Leibler divergence）損失，或直接對意圖機率進行均方誤差（Mean Squared Error, MSE）回歸，可能會訓練出一個更能理解不確定性的、更細膩的模型。因此，在第七章的模型訓練部分，我們將設計實驗，比較傳統二元交叉熵方法與直接回歸意圖機率的方法，為專案增添一層研究導向的深度

**PIE數據集標註綱要及其應用價值**

|標註鍵 (Annotation Key)|數據類型 (Data Type)|關鍵欄位 (Key Fields)|描述 (Description)|預測應用價值 (Predictive Utility)|
|---|---|---|---|---|
|`ped_annotations`|JSON 物件|`bbox`, `behavior`, `intention_prob`|包含單個行人的所有逐幀標註|專案核心數據源，提供訓練標籤與基礎特徵|
|`bbox`|列表 [int]|`[x1, y1, x2, y2]`|邊界框左上角與右下角座標|軌跡與速度特徵的核心；姿態估計器的輸入|
|`behavior`|字典 {str: bool}|`walking`, `looking`, `crossing`|描述行人當前行為的布林標籤|提供高層次的行為語義，可作為模型的輔助特徵|
|`intention_prob`|浮點數|`0.0` 至 `1.0`|匯總自人類標註者的穿越意圖機率|模型訓練的核心目標標籤（Ground Truth）|
|`obd_data`|JSON 物件|`speed`, `heading`|同步的自我車輛速度與航向|提供自我車輛的動態上下文，是場景理解的重要部分|
|`traffic_light`|JSON 物件|`bbox`, `state`|交通號誌的位置與狀態（紅、黃、綠）|關鍵的場景上下文特徵，直接影響行人決策|


This project integrates several cutting-edge models and libraries to achieve its capabilities.
NEW:

| Category           | Technology                  | Purpose                                           |
| ------------------ | --------------------------- | ------------------------------------------------- |
| Foundation Models  | Grounding DINO              | Zero-Shot Object Detection via Text Prompts       |
|                    | Segment Anything (SAM/SAM2) | High-Quality Object Segmentation & Video Tracking |
| Feature Extractors | MMPose                      | Whole-Body (133 keypoints) Pose Estimation        |
|                    | ETH-XGaze (or similar)      | Gaze and Head Pose Estimation                     |
| Core Libraries     | Python 3.9+                 | Core Programming Language                         |
|                    | PyTorch                     | Deep Learning Framework                           |
|                    | Transformers (Hugging Face) | For easy access to models like Grounding DINO     |
|                    | OpenCV                      | Video/Image Processing & Visualization            |
| Project Management | Poetry                      | Dependency and Environment Management             |

**Step1**: 目標檢測(行人) +tracking -> Grounded SAM2  or YOLO8+SORT

**Step2**: segmentation -> SAM (SamAutomaticMaskGenerator) + CLIP
SAM生成的遮罩是類別無關的（class-agnostic）。我們將使用CLIP（Contrastive Language-Image Pre-training）模型為每個遮罩賦予語義標籤


**Step3:** Keypoint detection + Pose Estimation -> ==MMPose==
MMPose (Whole-Body (133 keypoints) Pose Estimation) -  輸入行人的邊界框, MMPose就可以自行detect 被偵測行人的133個關鍵點及其對應的信賴度分數, 輸出是一個數據結構列表. 

數據結構：輸出的數據格式通常是一個字典列表，每個字典對應一個行人，其中包含一個名為`'keypoints'`的鍵，其值為一個形狀為`(133, 3)`的NumPy陣列，每一行代表一個關鍵點的`(x, y, confidence)`

**Step4**: Gaze estimation -> ==ETH-XGaze==
ETH-XGaze - 凝視估計模型通常需要一個裁剪好的人臉圖像作為輸入 。我們將利用由MMPose提取的臉部關鍵點(ETH-XGaze模型需要224x224像素). 載入預訓練的凝視模型，並對裁剪後的人臉圖像進行推論。模型的原始輸出通常是一個表示俯仰角（pitch）和偏航角（yaw）的二維向量

作為凝視的補充特徵，頭部的三維朝向也能提供關於注意力方向的粗略但有價值的資訊。這可以透過使用MMPose模型提供的3D臉部標記點，或利用2D臉部關鍵點和一個通用的3D人臉模型，通過解決PnP（Perspective-n-Point）問題來估計









#### PedestrianIntent 的行人動態狀態估計：軌跡與姿態

#### 3.1 物體偵測與追蹤

感知管道的第一步是在影片序列中識別並持續追蹤每個行人。

- **物體偵測**：推薦使用一個標準且高性能的物體偵測器，例如在COCO數據集上預訓練的YOLOv8。YOLO系列模型在速度和精度之間取得了良好的平衡，非常適合即時應用。
- **物體追蹤**：為了建立行人的運動軌跡，我們將實施一個簡潔而有效的追蹤算法，如SORT（Simple Online and Realtime Tracking）或其改進版DeepSORT。SORT主要使用卡爾曼濾波器（Kalman filter）進行運動狀態預測和數據關聯。這個過程為場景中的每個行人分配一個唯一的ID，並輸出一系列連續的邊界框，從而構成其原始軌跡。
#### 3.2 基於MMPose的全身姿態估計

姿態是預測意圖的最重要線索之一，它能提供比邊界框更豐富的身體語言資訊。

- **模型選擇**：我們將採用MMPose生態系統中的頂尖模型，例如DWPose 。該模型基於MMPose開發，在COCO-WholeBody數據集上表現出色 ，能夠同時偵測身體、腳部、臉部和手部的關鍵點。值得注意的是，DWPose提供了ONNX格式的模型，這對後續的性能優化和部署非常有利 。  
- **推論流程**：我們將提供一個使用`MMPoseInferencer`接口的Python程式碼指南 ，其標準流程如下：  
    1. 初始化偵測器（例如，使用`mmdet`）和姿態估計器（`mmpose`）。
    2. 對於影片的每一幀，首先將圖像傳遞給偵測器，獲取所有行人的邊界框。
    3. 然後，將原始圖像和偵測到的邊界框一同傳遞給姿態估計器。
    4. 模型的輸出是一個數據結構列表，其中每個結構都包含了一個被偵測行人的133個關鍵點及其對應的信賴度分數 。  
        
- **數據結構**：輸出的數據格式通常是一個字典列表，每個字典對應一個行人，其中包含一個名為`'keypoints'`的鍵，其值為一個形狀為`(133, 3)`的NumPy陣列，每一行代表一個關鍵點的`(x, y, confidence)`。






#### PedestrianIntent 的行人專注與注意力：凝視與頭部姿態

#### 4.1 凝視的重要性

凝視是注意力的直接指標，是理解人類意圖的強有力線索 。一個即將穿越馬路的行人，其視線通常會反覆掃視左右來車方向。這一在PIE數據集中缺失的資訊，對於高層次的推理至關重要。  

#### 4.2 凝視估計流程

本節將詳細介紹一個從影片中提取凝視向量的實用流程。

- **模型選擇**：推薦使用在ETH-XGaze 或MPIIGaze/MPIIFaceGaze 數據集上預訓練的模型。其中，ETH-XGaze因其在極端頭部姿態下的魯棒性而特別適合本專案的應用場景。  
    
- **輸入準備**：凝視估計模型通常需要一個裁剪好的人臉圖像作為輸入 。我們將利用前一章中由MMPose提取的臉部關鍵點。具體來說，可以根據68個臉部關鍵點計算出一個包圍人臉的邊界框，從原始幀中裁剪出該區域，並將其縮放到模型所需的輸入尺寸（例如，ETH-XGaze模型需要224x224像素）。  
    
- **推論與輸出**：程式碼將演示如何載入預訓練的凝視模型，並對裁剪後的人臉圖像進行推論。模型的原始輸出通常是一個表示俯仰角（pitch）和偏航角（yaw）的二維向量 。我們將提供在ETH-XGaze代碼庫中提及的  
    `pitchyaw_to_vector`工具函數，將這個二維角度轉換為攝影機座標系下的三維單位向量。

#### 4.3 頭部姿態估計

作為凝視的補充特徵，頭部的三維朝向也能提供關於注意力方向的粗略但有價值的資訊。這可以透過使用MMPose模型提供的3D臉部標記點，或利用2D臉部關鍵點和一個通用的3D人臉模型，通過解決PnP（Perspective-n-Point）問題來估計。




#### PedestrianIntent 透過零樣本語義分割實現場景情境化

#### 5.1 語義理解的必要性

行人的意圖深受其所處環境的影響。行人是站在`人行道`上還是`馬路`上？他們的凝視向量是指向`迎面駛來的汽車`還是`建築物`？回答這些問題需要對場景進行語義分割。

#### 5.2 步驟一：使用SAM進行實例分割

我們將採用Segment Anything Model（SAM）或其後繼者SAM2，利用其強大的零樣本（zero-shot）實例分割能力 。  

- 我們將提供一個使用`SamAutomaticMaskGenerator`的程式碼範例 。這個工具接收一張圖像作為輸入，並自動為場景中偵測到的所有物體和區域生成一系列的遮罩（mask）。  
- 其輸出結構是一個字典列表，每個字典包含一個二值的`segmentation`遮罩、`bbox`、`area`等元數據 。  

#### 5.3 步驟二：使用CLIP進行語義標註

這是至關重要的融合步驟，因為SAM生成的遮罩是類別無關的（class-agnostic）。我們將使用CLIP（Contrastive Language-Image Pre-training）模型為每個遮罩賦予語義標籤。

- **零樣本分類流程** ：  
    
    1. 定義一個與交通場景相關的目標類別列表，例如：`['道路', '人行道', '汽車', '斑馬線', '交通號誌', '行人', '建築物', '天空', '植被']`。
    2. 對於SAM生成的每一個遮罩，使用該遮罩從原始圖像中裁剪出對應的區域。
    3. 將裁剪出的圖像塊輸入到CLIP的圖像編碼器，獲得一個圖像嵌入向量。
    4. 將目標類別列表（每個類別前可加上提示，如「一張...的照片」）輸入到CLIP的文本編碼器，獲得一系列文本嵌入向量。
    5. 計算圖像嵌入向量與所有文本嵌入向量之間的餘弦相似度。
    6. 相似度最高的類別即被指定為該遮罩的語義標籤。

#### 5.4 創建場景上下文特徵

此模組的最終輸出是一個場景的語義地圖。對於意圖預測模型，這些資訊可以被編碼為多種高層次特徵：

- **以行人為中心的標籤**：識別行人當前所在遮罩的語義標籤（例如，`on_sidewalk`）。
- **以凝視為中心的標籤**：計算行人的凝視向量與哪個語義物體的遮罩相交（例如，`looking_at_car`）。這是一個極具預測能力的交互特徵。

在 **行人偵測與意圖預測** 中，除了 **行人本身的外觀 (bounding box, pose, keypoints, trajectory)**，還需要考慮行人所處的 **環境資訊**，這些就是 **場景上下文 (Scene Context)**。
「上下文」包括：
- **靜態語義資訊 (Static Semantic Context)**
    - 道路 (road)、人行道 (sidewalk)、斑馬線 (crosswalk)、車道線、紅綠燈位置
- **動態語義資訊 (Dynamic Context)**
    - 附近車輛、自行車的速度與位置
    - 行人群體之間的互動
- **交互資訊 (Interaction Context)**
    - 行人是否站在斑馬線上
    - 行人是否看向車輛
    - 行人是否與其他行人結伴



#### 創建場景上下文特徵 (Scene Context Features)

**「創建場景上下文特徵 (Scene Context Features)」**，在 **行人過街意圖預測 (PCIP)** 中非常重要。我分三步詳細解釋：

##### 1. 什麼是「場景上下文 (Scene Context)」？

在 **行人偵測與意圖預測** 中，除了 **行人本身的外觀 (bounding box, pose, keypoints, trajectory)**，還需要考慮行人所處的 **環境資訊**，這些就是 **場景上下文 (Scene Context)**。

具體來說，「上下文」包括：

- **靜態語義資訊 (Static Semantic Context)**
    - 道路 (road)、人行道 (sidewalk)、斑馬線 (crosswalk)、車道線、紅綠燈位置
- **動態語義資訊 (Dynamic Context)**
    - 附近車輛、自行車的速度與位置
    - 行人群體之間的互動
- **交互資訊 (Interaction Context)**
    - 行人是否站在斑馬線上
    - 行人是否看向車輛
    - 行人是否與其他行人結伴

換句話說，**上下文就是「除了行人自身以外，影響他是否要過街的外部線索」**。


##### 2. 創建「場景上下文特徵」的流程
一般 pipeline 長這樣
1. **場景語義分割 (Semantic Segmentation)**
    - 用模型 (如 DeepLab, SAM2, BEVFormer) 把場景分成：sidewalk, crosswalk, road, car, bicycle, traffic light...
2. **生成語義地圖 (Semantic Map)**
    - 每一幀對應一張「語義遮罩 (semantic mask)」，每個 pixel 屬於某個語義類別。
    - 這就是 **語義地圖 (semantic map)**。
3. **將行人位置/凝視映射到語義地圖**
    - **以行人為中心的標籤 (Pedestrian-centered features)**
        - 看這個行人 bounding box/pose 的位置落在哪個語義區域 → 例如「on_sidewalk」、「on_crosswalk」、「on_road」。
    - **以凝視為中心的標籤 (Gaze-centered features)**
        - 用頭部朝向 + 視線向量 → 看視線與哪個語義區域相交 → 例如「looking_at_car」、「looking_at_crosswalk」。

##### 3. 解釋原文敘述與上下文關係

> **「此模組的最終輸出是一個場景的語義地圖。對於意圖預測模型，這些資訊可以被編碼為多種高層次特徵：  
> 以行人為中心的標籤：識別行人當前所在遮罩的語義標籤（例如，on_sidewalk）。  
> 以凝視為中心的標籤：計算行人的凝視向量與哪個語義物體的遮罩相交（例如，looking_at_car）。這是一個極具預測能力的交互特徵」**

1. **語義地圖 (Semantic Map)**
    - 模組的輸出不是單純影像，而是一張語義地圖，每個 pixel 都有語義標籤 (road, sidewalk, car)。
    - 這就是「上下文的結構化表徵」。
2. **行人為中心的標籤 (Pedestrian-centered label)**
    - 問題：行人「在哪裡」？
    - 答案：例如「on_sidewalk」→ 表示行人目前站在人行道上，可能還沒有要過街。
    - 這是上下文的一種，因為它描述了行人和**場景環境**的關係。
3. **凝視為中心的標籤 (Gaze-centered label)**
    - 問題：行人「看哪裡」？
    - 答案：例如「looking_at_car」→ 表示行人正注意來車，可能在評估是否安全過街。
    - 這是 **行人與場景互動的上下文**，比單純位置更強的線索。

##### 4. 舉例說明 (具體例子)

假設 PIE dataset 影片中有一幀：
- 行人 A：站在人行道邊緣 (on_sidewalk)
- 行人頭部朝向：向左 → 凝視方向交到「car」的語義遮罩 (looking_at_car)

這些上下文特徵可被編碼成：
`{  "pedestrian_location_label": "on_sidewalk",  "gaze_interaction_label": "looking_at_car" }`
**解釋**：
- 若行人只是站在 sidewalk 且看著車 → 很可能還在等待
- 若行人站在 crosswalk 且看著前方 road → 過街意圖概率就高。

##### 總結
- **上下文 (Scene Context)** = 行人所在環境的靜態 (sidewalk, crosswalk, road) + 動態 (車、其他行人) + 交互 (凝視、互動) 資訊。
- **語義地圖 (Semantic Map)** = 將場景像素轉成「語義遮罩」。
- **行人為中心標籤** = 行人當前在哪個語義區域 (on_sidewalk, on_crosswalk)。
- **凝視為中心標籤** = 行人看向哪個語義區域 (looking_at_car, looking_at_crosswalk)。
- **與上下文的關係**：這些標籤就是「場景上下文特徵」，它們能捕捉「行人 + 場景」的交互，而不只是「行人自己」。







#### PIE dataset 可延伸的 8 個新方向

### 1. **行人視線與注意力 (Gaze & Head Orientation) 分析**

- **想法**：除了走路姿態外，行人的「頭部朝向」與「視線方向」是過街意圖的重要線索。
- **技術**：MMPose Whole-body / RTMDet + 人臉關鍵點檢測；結合 Gaze Estimation (如 ETHXGaze)。
- **應用**：預測行人是否已注意到來車 → 協助自駕車判斷是否需要讓行。

### 2. **社會互動建模 (Social Interaction Modeling)**

- **想法**：行人不是單獨存在，他們會互相影響。例如：一群人一起走向斑馬線，意圖更明顯。
- **技術**：Graph Neural Network (GNN, Graph Attention Network)，將多行人軌跡建模。
- **應用**：多人群組行為預測，辨識「集體過街」vs「個體猶豫」。

### 3. **未來軌跡多模態預測 (Multi-modal Trajectory Prediction)**

- **想法**：行人未來行徑可能不是單一路徑（例如：往左繞車 or 直接過街）。
- **技術**：Trajectory Prediction 模型 (Trajectron++, Social-STGCNN)，結合 Video Transformer (VideoMAE)。
- **應用**：輸出多種可能的路徑與概率分布，而不是單一預測。

### 4. **場景語義增強 (Scene Context Augmentation)**

- **想法**：僅用 bounding box/pose 還不夠，行人意圖和「場景元素」(斑馬線、車道線、紅綠燈) 密切相關。
- **技術**：SAM2 / DeepLab / Segment Anything 做場景分割，與 PIE dataset 的行人標註融合。
- **應用**：過街意圖預測 + 場景約束，例如「行人站在斑馬線旁，紅燈 → 過街意圖較低」。

### 5. **多任務學習 (Multi-task Learning)**

- **想法**：同時訓練多個任務（過街意圖分類 + 軌跡預測 + 行人動作分類）。
- **技術**：Transformer-based multi-task head (例如 DETR-style + temporal transformer)。
- **應用**：強化模型泛化能力，讓意圖預測更穩定。

### 6. **跨天氣 / 光照條件的 Domain Adaptation**

- **想法**：PIE dataset 大多在白天好天氣拍攝，若用於夜間或下雨天準確度會下降。
- **技術**：Domain Adaptation / Domain Generalization（例如：Style Transfer GAN, CycleGAN），將原始影像轉換成「夜間版本」再訓練。
- **應用**：讓 PCIP 系統在任何環境下都能可靠運作。

### 7. **視覺語言模型 (Vision-Language Model, VLM) 解釋意圖**

- **想法**：不只是輸出「要過街/不要過街」，還能讓模型解釋「為什麼」。
- **技術**：Qwen2-VL / LLaVA / BLIP-2，輸入場景 + 行人框，輸出自然語言解釋。
- **應用**：提供可解釋 AI（例如「行人正看著前方斑馬線並開始移動 → 過街可能性高」）。

### 8. **弱監督 / 自監督學習 (Weakly/Self-Supervised Learning)**

- **想法**：PIE dataset 標註了意圖，但標註成本高，能否用少量標註 + 大量無標註影片？
- **技術**：DINOv2 / VideoMAE 做表徵學習，然後 fine-tune 在意圖預測。
- **應用**：減少人工標註，讓模型能泛化到新城市或新資料。

### 補充 Bonus Ideas

- **行人「猶豫行為」建模**：從加速度變化檢測停頓 → 代表不確定是否要過街。
- **「車-人互動」建模**：同時輸入車輛速度 + 行人動作，做交互式意圖預測。
- **跨資料集學習**：把 PIE + JAAD + NuScenes 結合，測試模型在不同 domain 下的泛化能力。

### 總結

如果你要用 **PIE dataset**：
- **基礎任務**：Detection + Pose + Intention Classification
- **進階任務**：
    1. 行人視線分析
    2. 社會互動 (多人 GNN)
    3. 多模態軌跡預測
    4. 場景語義融合
    5. 多任務學習
    6. Domain Adaptation
    7. VLM 解釋意圖
    8. 自監督表徵學習