


|                 | Detection                              | Tracking                                                                                                                                        |
| --------------- | -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| 2D 圖像，多個物體      | R-CNN, YOLO, DETR                      | no                                                                                                                                              |
| 2D 影片，多個物體      | R-CNN, YOLO, DETR                      | SORT, DeepSORT, ByteTrack                                                                                                                       |
| 多攝影機 3D 影片，多個物體 | 2D detection + <br>Multi-View Geometry | SORT, DeepSORT, ByteTrack<br>+<br>攝影機校正（Camera Calibration）<br>物件軌跡融合（Trajectory Fusion）<br>時間同步（Time Synchronization）<br>物件重識別（ReID）<br>處理重疊區域 |

|           | Tracking algorithm                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| SORT      | 先利用object detection model譬如YOLOv8 將video每一幀object detection的bounding box輸入model, 並給予一個id. 在下一幀(t+1)利用kalman filter預測在下一幀(t+1)的bounding box位置, 並與這一幀(t)的現有bounding box比對. 用hungarian algorithm建立成本矩陣考慮運動資訊, 將下一幀(t+1)的bounding box給予適當的id. , 並用ReID管理更新id                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| DeepSORT  | 先利用object detection model譬如YOLOv8 將video每一幀object detection的bounding box輸入model, 並給予一個id. 而對每個boundary box用Resnet model提取**feature map**. 在下一幀(t+1)利用kalman filter預測在下一幀(t+1)的bounding box位置 也計算feature map, 並與這一幀(t)的現有bounding box feature map比對. 用hungarian algorithm建立成本矩陣考慮運動資訊和**外觀資訊**, 將下一幀(t+1)的bounding box給予適當的id, 並用ReID管理更新id                                                                                                                                                                                                                                                                                                                                                                                        |
| ByteTrack | 先利用 object detection model 譬如 YOLOv8 將 video 每一幀 object detection 的**所有** bounding box（包含高低置信度）輸入 model，並給予一個暫時的 ID。**不對每個 boundary box 用CNN提取 feature map**。在下一幀 (t+1) 利用卡爾曼濾波器預測在下一幀 (t+1) 的 bounding box 位置。將**低置信度的檢測結果**與**高置信度追蹤軌跡**以**重疊度**和**運動一致性**進行關聯。再將**高置信度的檢測結果**與剩餘的**高置信度追蹤軌跡**計算卡爾曼濾波器的預測位置，用 Hungarian algorithm 建立成本矩陣考慮運動資訊，將下一幀 (t+1) 的 bounding box 給予適當的 ID。使用 ReID （可選）管理更新 ID。                                                                                                                                                                                                                                                                                                                          |
| 比較        | **SORT (Simple Online and Realtime Tracking):**<br>    - 它以簡單、高效的著稱，主要使用卡爾曼濾波器（Kalman Filter）來預測物體在下一幀的位置，並利用匈牙利算法（Hungarian algorithm）將預測結果與檢測結果進行關聯。<br>    - SORT 的優點是速度快，但由於僅基於物體的位置資訊，因此在物體發生遮擋或運動不規則時，追蹤效果可能會下降。<br><br>**DeepSORT (Deep Simple Online and Realtime Tracking):**<br>    - DeepSORT 是在 SORT 的基礎上進行改進，它引入了深度學習技術，利用物體的外觀特徵（Appearance Features）來輔助追蹤。<br>    - 通過使用深度關聯度量（Deep Association Metric），DeepSORT 能夠更準確地處理物體遮擋和 ID 切換（ID Switch）等問題，提高追蹤的魯棒性。<br><br>**ByteTrack (Multi-Object Tracking by Associating Every Detection Box):**<br>    - ByteTrack 的創新之處在於它充分利用了所有檢測框，而不僅僅是高置信度的檢測結果。<br>    - 它通過將低置信度的檢測框與高置信度的追蹤軌跡進行關聯，有效地減少了物體被遮擋或運動模糊時的 ID 切換問題。<br>    - ByteTrack 著重處理被遮蔽物體的追蹤，大大增加了整體追蹤的準確度。 |
|           |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |

以下表格總結了這三種算法在多目標追蹤能力上的主要差異：

|特性|SORT|DeepSORT|ByteTrack|
|:--|:--|:--|:--|
|追蹤依據|物體位置|物體位置 + 外觀特徵|所有檢測框（包括低置信度）|
|處理遮擋能力|較弱|較強|很強|
|ID 切換頻率|較高|較低|極低|
|運算速度|非常快|較快|較快|
|準確度|較低|較高|非常高|

**總結**

- SORT 適合對速度要求極高的場景，但準確度相對較低。
- DeepSORT 在準確度和速度之間取得了較好的平衡，適合大多數多目標追蹤應用。
- ByteTrack 著重準確度，在物體遮蔽嚴重的情況下，比起前兩個算法，有更好的追蹤成果。

希望這些資訊能夠幫助您更好地理解 ByteTrack、SORT 和 DeepSORT 在多目標追蹤能力上的差異。

Others:
[[PP-Tracking]]



處理多個攝影機（不同角度）的多個影片進行多目標追蹤（Multi-Object Tracking, MOT）是一項複雜的任務，單純使用 SORT、DeepSORT 或 ByteTrack 並不足以完成，需要額外的處理流程。以下是詳細的說明：

處理多攝影機 MOT 時，會遇到以下主要挑戰：

- **視角差異：**
    - 不同攝影機的視角可能導致物件外觀、大小和運動軌跡的顯著差異。
- **重疊區域：**
    - 不同攝影機的視野可能存在重疊區域，需要處理物件在這些區域的轉換和關聯。
- **時間同步：**
    - 確保不同攝影機的影片時間同步，以便正確關聯不同視角的物件。
- **物件重識別（ReID）：**
    - 由於視角變化，單一攝影機追蹤的物件可能在其他攝影機中難以識別。

**解決方案**

為了處理多攝影機 MOT，需要在基本的單攝影機追蹤算法（SORT、DeepSORT、ByteTrack）基礎上，加入以下額外流程：

1. **攝影機校正（Camera Calibration）：**
    - 使用攝影機校正技術，確定每個攝影機的內部和外部參數。
    - 這將使您能夠將不同攝影機的影像轉換到一個共同的座標系，從而實現跨攝影機的物件定位。
2. **物件軌跡融合（Trajectory Fusion）：**
    - 將不同攝影機的物件軌跡融合到一個統一的全局軌跡。
    - 這需要使用資料關聯技術，例如：
        - 基於位置的關聯：比較不同攝影機中物件在全局座標系中的位置。
        - 基於外觀的關聯：使用 ReID 技術比較不同視角下物件的外觀特徵。
    - 可以經由卡爾曼濾波器，或者其他的濾波方式來針對融合後的軌跡做優化。
3. **時間同步（Time Synchronization）：**
    - 確保所有攝影機的影片時間戳記對齊，或者使用時間同步算法來校正時間差異。
4. **物件重識別（ReID）：**
    - 在不同攝影機的視角下，同一個物件的外觀可能差異很大。
    - 可以使用進階的 ReID 技術，例如：
        - 視角不變特徵提取（Viewpoint-Invariant Feature Extraction）。
        - 跨攝影機 ReID 模型（Cross-Camera ReID Models）。
5. **處理重疊區域：**
    - 針對重疊區域做特別的處理，確定如何正確接續在不同攝影機中，同一個物件的ID。
    - 這可能涉及使用物件軌跡融合技術。
    - 並且需要針對在不同攝影機下的物件，做重複偵測的消除。

**總結**

- SORT、DeepSORT 和 ByteTrack 主要設計用於單攝影機的 MOT。
- 處理多攝影機 MOT 需要額外的攝影機校正、物件軌跡融合、時間同步和 ReID 等流程。
- 這些額外流程旨在解決多視角、重疊區域和時間差異帶來的挑戰。




**多目标跟踪MOT(multiple object tracking)或者MTT(multiple target tracking)**

**单目标跟踪SOT(single object tracking)**

**[ReID](https://zhida.zhihu.com/search?content_id=179731255&content_type=Article&match_order=1&q=ReID&zhida_source=entity)** (Re-identification): 利用算法，在图像库中找到要搜索的目标.它是属于图像检索的一个子问题。行人重识别（Person-edestrian Re-identification，ReID)

**[Trajectory](https://zhida.zhihu.com/search?content_id=179731255&content_type=Article&match_order=1&q=Trajectory&zhida_source=entity)**（轨迹）：一条轨迹对应这一个目标在一个时间段内的位置序列

**[Tracklet](https://zhida.zhihu.com/search?content_id=179731255&content_type=Article&match_order=1&q=Tracklet&zhida_source=entity)**（轨迹段）：形成Trajectory过程中的轨迹片段。完整的Trajectory是由属于同一物理目标的Tracklets构成的。

**[ID switch](https://zhida.zhihu.com/search?content_id=179731255&content_type=Article&match_order=1&q=ID+switch&zhida_source=entity)**（ID切换）：又称ID sw.。对于同一个目标，由于跟踪算法误判，导致其ID发生切换的次数称为ID sw.。跟踪算法中理想的ID switch应该为0。

### Re-identification（ReID） vs. Object Tracking

**Re-identification（ReID，目标重识别）** 和 **Object Tracking（目标跟踪）** 在计算机视觉任务中都是多目标管理的重要技术，但它们的目标和方法有所不同。

|**类别**|**Object Tracking（目标跟踪）**|**Re-identification（目标重识别）**|
|---|---|---|
|**目标**|在**同一视频流**中持续跟踪目标，并在帧与帧之间匹配目标|在**不同摄像头、不同时间或不同场景**下匹配同一对象|
|**输入**|目标检测框和历史轨迹|一组目标图片（gallery）和待识别图片（query）|
|**挑战**|遮挡、目标重叠、运动模糊|视角变化、光照变化、行人姿态变化、分辨率低|
|**输出**|目标ID的连续轨迹|目标ID的匹配分数|
|**常用方法**|运动模型（卡尔曼滤波、LSTM）、IOU匹配、深度学习匹配（DeepSORT）|深度学习特征提取（ResNet、ViT）、度量学习（Triplet Loss, Contrastive Loss）|

|**类别**|**Re-identification（目标重识别）**|**Multi-Target Multi-Camera Tracking（MTMCT）**|
|---|---|---|
|**目标**|识别同一对象（如行人、车辆）在不同摄像机中的身份|在多个摄像机之间持续跟踪多个目标的轨迹|
|**输入**|**Query 图像**（待识别目标） + **Gallery 数据库**（多个摄像头采集的候选目标）|多个摄像头的视频流（包含目标的运动信息）|
|**核心技术**|深度学习特征提取（CNN、ViT）、度量学习（Triplet Loss, Contrastive Loss）|目标检测 + ReID + 轨迹关联（Trajectory Association）|
|**输出**|Query 与 Gallery 中目标的匹配分数|各目标在多个摄像头间的完整轨迹|
|**主要挑战**|视角变化、光照变化、遮挡、行人姿态变化、衣物变化|摄像机间时间同步、摄像机视野重叠度、遮挡、目标消失与重新出现|
|**常用方法**|特征向量比对（余弦相似度、欧几里得距离）|运动模型（Kalman Filter）、外观匹配（ReID）、轨迹关联（Graph Matching, Hungarian Algorithm）|



Reference:
最新综述 | 关于单目标视觉追踪，看这一篇就够了！ - CS Conference的文章 - 知乎
https://zhuanlan.zhihu.com/p/558962899

目标跟踪算法综述 - 小茗同学的文章 - 知乎
https://zhuanlan.zhihu.com/p/433991731

目标跟踪初探（DeepSORT） - 可乐的文章 - 知乎
https://zhuanlan.zhihu.com/p/90835266

目标跟踪基础——DeepSORT - Van Helsing的文章 - 知乎
https://zhuanlan.zhihu.com/p/499240427

DeepSort：基于检测的目标跟踪的经典 - AI大道理的文章 - 知乎
https://zhuanlan.zhihu.com/p/651064644

