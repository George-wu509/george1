
以下是針對**無人機（UAV）影像資料集**，依據其適用任務分類（2D/3D影像、影片）來整理可用於：

- **Video Object Detection**
    
- **Video Segmentation**
    
- **Video Understanding**
    
- **Vision-Language Models（VLMs）**
    

---

## 🔸一、2D UAV 影像 / 影片資料集（適用於Video Detection / Segmentation / Tracking）

### 1. **VisDrone** (Visual Object Detection in Drone Imagery)

- **任務**：object detection, tracking, instance segmentation
    
- **格式**：影像與影片（JPEG + video frame sequences）
    
- **標註**：bounding boxes, class labels, occlusion/truncation flags
    
- **數量**：10K+ images, 263 video clips
    
- **類別**：車、人、自行車、卡車、巴士、摩托車等
    
- **用途**：
    
    - ✅ Video Object Detection
        
    - ✅ Video Tracking
        
    - ✅ Instance Segmentation
        
- **連結**：[https://github.com/VisDrone/VisDrone-Dataset](https://github.com/VisDrone/VisDrone-Dataset)
    

---

### 2. **UAVDT** (Unmanned Aerial Vehicle Detection and Tracking)

- **任務**：object detection, multi-object tracking
    
- **格式**：video sequences (video frames), annotation in txt/xml
    
- **數量**：100 videos (~80K frames)
    
- **類別**：車輛為主（small, medium, large vehicles）
    
- **場景**：城市、高速公路、校園等
    
- **用途**：
    
    - ✅ Video Object Detection
        
    - ✅ Video Tracking
        
- **連結**：[https://github.com/detectRecog/UAVDT](https://github.com/detectRecog/UAVDT)
    

---

### 3. **Okutama-Action**

- **任務**：human action detection in aerial videos
    
- **格式**：4K video（3840×2160 @ 30fps）
    
- **標註**：bounding box + 12 action classes
    
- **用途**：
    
    - ✅ Video Object Detection
        
    - ✅ Video Action Recognition
        
    - ✅ Video Understanding
        
- **連結**：[https://github.com/OkutamaAction/okutama-action](https://github.com/OkutamaAction/okutama-action)
    

---

### 4. **AerialMAV**

- **任務**：multi-modal aerial video dataset for video segmentation
    
- **格式**：RGB + depth video
    
- **用途**：
    
    - ✅ Video Semantic Segmentation
        
    - ✅ 3D Understanding (pseudo 3D from depth)
        
- **連結**：無公開正式下載連結，請參考論文搜尋
    

---

## 🔸二、3D UAV 資料集（適用於3D Object Detection / 3D Segmentation）

### 5. **UAVid**

- **任務**：semantic segmentation on urban scenes captured by UAVs
    
- **格式**：video frame sequences (2D), 可以生成 pseudo-3D
    
- **標註**：11 semantic classes
    
- **用途**：
    
    - ✅ Semantic Segmentation
        
    - ⚠️ 非真3D，但適合類似 3D 運算（via multiple frames）
        
- **連結**：https://uavid.nl
    

---

### 6. **Dronedeploy 3D**

- **任務**：3D reconstruction from UAV images
    
- **格式**：多角度影像 + point cloud
    
- **用途**：
    
    - ✅ Structure-from-Motion (SfM)
        
    - ✅ 3D Reconstruction
        
    - ✅ 可生成自定義Video Segmentation任務
        
- **連結**：需註冊DroneDeploy平台，非完全開源
    

---

## 🔸三、影片＋語言（VLMs）相關 UAV 資料集

### 7. **TACTIC Dataset** (Aerial Captioning)

- **任務**：video captioning of aerial videos
    
- **格式**：video + text captions
    
- **語言標註**：描述如“a car is moving on a bridge”
    
- **用途**：
    
    - ✅ Video Captioning
        
    - ✅ Video-Language Pretraining
        
    - ✅ VLM fine-tuning
        
- **連結**：[https://github.com/Tsinghua-Tencent-Prime-Lab/TACTIC](https://github.com/Tsinghua-Tencent-Prime-Lab/TACTIC)
    

---

### 8. **UAV-Caption** Dataset

- **任務**：image/video captioning from drone videos
    
- **格式**：video + natural language sentences
    
- **用途**：
    
    - ✅ VLM fine-tuning
        
    - ✅ Video Understanding
        
- **連結**：部分研究公開，請參考：
    
    - 論文：[https://arxiv.org/abs/2112.09269](https://arxiv.org/abs/2112.09269)
        

---

## 🔸四、合成與模擬 UAV 資料集（可用於任意任務）

### 9. **AirSim** by Microsoft

- **任務**：自定義 UAV 任務（支援 video / depth / segmentation）
    
- **功能**：模擬 UAV 飛行、收集 multi-modal data（RGB, Depth, Seg, IMU）
    
- **用途**：
    
    - ✅ Video Detection / Segmentation
        
    - ✅ 3D Scene Understanding
        
    - ✅ Video-Language 模擬數據生成
        
- **連結**：[https://github.com/microsoft/AirSim](https://github.com/microsoft/AirSim)
    

---

### 10. **Synthetic Aerial Dataset from CARLA / UnrealEngine**

- 自行使用 CARLA 或 Unreal Engine 模擬 UAV 飛行與錄影，標註 bounding box / segmentation / video-language
    
- ✅ 支援多鏡頭、多任務合成資料
    
- **應用**：
    
    - ✅ 可客製任意 VLM / segmentation / detection 任務
        

---

## 🔸建議搭配模型與研究方向

|任務類型|可用模型|建議資料集|
|---|---|---|
|Video Object Detection|ViT-Adapter, ViTDet, Tube DET, Deformable DETR|VisDrone, UAVDT|
|Video Instance Segmentation|VideoMask2Former, VisTR|UAVid, AerialMAV|
|Video-Language VLM|CLIP, BLIP-2, Video-ChatGPT, Flamingo|TACTIC, UAV-Caption|
|3D Reconstruction + Seg|NeRF, 3DGS, SfM+MVS|Dronedeploy 3D, AirSim|