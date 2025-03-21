


|                 | Detection                          | Tracking                  |
| --------------- | ---------------------------------- | ------------------------- |
| 2D 圖像，單一物體      | R-CNN, YOLO, SSD, DETR             | no                        |
| 2D 圖像，多個物體      | R-CNN, YOLO, SSD, DETR             | no                        |
| 2D 影片，單一物體      | R-CNN, YOLO, SSD, DETR             | KCF, DeepSORT             |
| 2D 影片，多個物體      | R-CNN, YOLO, SSD, DETR             | DeepSORT, Tracktor, MOTDT |
| 多攝影機 3D 影片，單一物體 | 2D detection + Multi-View Geometry |                           |
| 多攝影機 3D 影片，多個物體 |                                    |                           |



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

