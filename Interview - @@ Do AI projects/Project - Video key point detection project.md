
My colab project:
https://colab.research.google.com/drive/18JK_0xvzLe-XszfTzxHr_Ztgjl6ueB3L

Reference:  
![[Pasted image 20250807001320.png]]

Post: Replacing Human Referees with AI
https://www.linkedin.com/posts/skalskip92_computervision-opensource-multimodal-activity-7355920959477837825-gjVm?utm_source=share&utm_medium=member_desktop&rcm=ACoAABMNj2MBAJSP3cWd4xpiz4wB7qdx43hvW18

Basketball AI: Automatic Detection of 3-Second Violation (Colab)
https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/basketball-ai-automatic-detection-of-3-second-violations.ipynb

Supervision github
https://github.com/roboflow/supervision

```
As a basketball and computer vision fan, I've always been fascinated by the idea of replacing referees with AI. Automating complex rules like the 3-second violation is challenging, and requires a lot of moving parts. But last week, [Alexander Bodner](https://www.linkedin.com/in/alexanderbodner/) and I finally got an MVP algorithm working!  
  
The best part is that all three models we used to build this project are fully open-source and available under the Apache 2.0 license:  
  
- RF-DETR – for ball, number, player, referee, and basket detection.  
- SAM2 – for player segmentation and tracking.  
- ViTPose++ – for player pose estimation.  
  
⮑ 🔗 notebook with the code: [https://lnkd.in/d2WPAgiv](https://lnkd.in/d2WPAgiv)  
  
You'll find links to my basketball player detection, ball keypoint detection models, as well as an end-to-end blog post explaining every step of the algorithm, in the comments below
```


#### detect players skeletons

```python
def plot_skeletons(key_points: sv.KeyPoints, tracker_ids: list[int], frame: np.array):
    for tracker_id in tracker_ids:
        tracker_key_points = key_points[int(tracker_id) - 1]
        annotator = sv.EdgeAnnotator(
            color=COLOR.by_idx(tracker_id),
            thickness=3
        )
        frame = annotator.annotate(scene=frame, key_points=tracker_key_points)
    return frame
```
這個 cell 定義了一個名為 `plot_skeletons` 的輔助函式，其主要功能是在影像上繪製檢測到的骨骼。

- **`def plot_skeletons(key_points: sv.KeyPoints, tracker_ids: list[int], frame: np.array):`**
    
    - 這是一個函式定義。它接受三個參數：
        
        - `key_points`: 類型為 `sv.KeyPoints`，這是 `supervision` 函式庫中用來儲存關鍵點資訊的物件。它包含了每個人物的關節座標。
            
        - `tracker_ids`: 類型為 `list[int]`，一個整數列表，代表了每個被追蹤人物的唯一 ID。
            
        - `frame`: 類型為 `np.array`，代表單個影像幀（使用 NumPy 陣列格式）。
            
- **`for tracker_id in tracker_ids:`**
    
    - 這是一個迴圈，遍歷 `tracker_ids` 列表中的每一個追蹤 ID。這確保了我們為每一位被偵測到的人物繪製骨骼。
        
- **`tracker_key_points = key_points[int(tracker_id) - 1]`**
    
    - 從 `key_points` 物件中取出與當前 `tracker_id` 相對應的關鍵點資訊。
        
    - 這裡有一個小細節：因為追蹤 ID 通常從 1 開始，而 Python 列表或陣列的索引是從 0 開始，所以我們需要將 `tracker_id` 減 1 來獲取正確的索引。
        
- **`annotator = sv.EdgeAnnotator(...)`**
    
    - 創建一個 `supervision.EdgeAnnotator` 物件。這個物件專門用於在關鍵點之間繪製邊（也就是骨骼）。
        
    - `color=COLOR.by_idx(tracker_id)`: 根據追蹤 ID 自動選擇一個顏色，這樣每個不同的人物骨骼會有不同的顏色，方便區分。
        
    - `thickness=3`: 設定繪製骨骼線條的粗細為 3 像素。
        
- **`frame = annotator.annotate(scene=frame, key_points=tracker_key_points)`**
    
    - 使用創建好的 `annotator` 物件來在當前 `frame` 上繪製骨骼。
        
    - 它將 `tracker_key_points` 中的關鍵點連接起來，形成骨骼，並將繪製結果應用到 `frame` 上。
        
- **`return frame`**
    
    - 函式返回已經繪製了所有骨骼的影像幀。
        

總結來說，這個 cell 提供了一個可重複使用的函式，將給定的關鍵點數據視覺化為骨骼，並將其疊加到影像上。


```python
import torch
from transformers import AutoProcessor, VitPoseForPoseEstimation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

POSE_ESTIMATION_MODEL_ID = "usyd-community/vitpose-plus-large"

pose_estimation_processor = AutoProcessor.from_pretrained(POSE_ESTIMATION_MODEL_ID)
pose_estimation_model = VitPoseForPoseEstimation.from_pretrained(
    POSE_ESTIMATION_MODEL_ID, device_map=DEVICE)
```
### **Cell 2: 模型載入與初始化**

這個 cell 主要負責載入預訓練的姿勢估計（pose estimation）模型。

- **`import torch`**
    
    - 引入 PyTorch 函式庫，這是許多深度學習模型的核心。
        
- **`from transformers import AutoProcessor, VitPoseForPoseEstimation`**
    
    - 從 Hugging Face 的 `transformers` 函式庫中引入兩個重要的類別：
        
        - `AutoProcessor`: 一個通用的處理器，可以根據模型 ID 自動載入對應的資料預處理器。
            
        - `VitPoseForPoseEstimation`: 用於姿勢估計的 Vision Transformer 模型。
            
- **`DEVICE = "cuda" if torch.cuda.is_available() else "cpu"`**
    
    - 一個標準的程式碼片段，用於檢查是否有可用的 NVIDIA GPU（即 `cuda`）。如果有，模型將在 GPU 上運行，以獲得更快的運算速度；否則，它將在 CPU 上運行。
        
- **`POSE_ESTIMATION_MODEL_ID = "usyd-community/vitpose-plus-large"`**
    
    - 定義一個字串變數，儲存了我們將要使用的姿勢估計模型的 ID。這個 ID 指向 Hugging Face Hub 上的特定模型，這裡選擇的是 `vitpose-plus-large`。
        
- **`pose_estimation_processor = AutoProcessor.from_pretrained(POSE_ESTIMATION_MODEL_ID)`**
    
    - 從 Hugging Face Hub 下載並載入與模型 ID 相應的資料預處理器。這個處理器負責將原始影像（例如 NumPy 陣列）轉換成模型可以理解的格式（例如，調整大小、正規化等）。
        
- **`pose_estimation_model = VitPoseForPoseEstimation.from_pretrained(...)`**
    
    - 從 Hugging Face Hub 下載並載入指定的姿勢估計模型。
        
    - `device_map=DEVICE`: 告訴 `transformers` 函式庫將模型載入到我們之前確定的 `DEVICE`（GPU 或 CPU）上。
        

總結來說，這個 cell 載入了骨骼檢測（姿勢估計）模型，為後續的處理做準備。


```python
import cv2
import torch
from tqdm import tqdm

PLAYER_ID = 2
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.7
INTERVAL = 30

annotated_frames = []

video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
frame = next(frame_generator)

result = PLAYER_DETECTION_MODEL.infer(frame, confidence=CONFIDENCE_THRESHOLD, iou_threshold=IOU_THRESHOLD)[0]
detections = sv.Detections.from_inference(result)
detections = detections[detections.class_id == PLAYER_ID]

XYXY = detections.xyxy
CLASS_ID = detections.class_id
TRACKE_ID = list(range(1, len(CLASS_ID) + 1))

detections = sv.Detections(
    xyxy=XYXY,
    tracker_id=np.array(TRACKE_ID)
)

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.load_first_frame(frame)

    for xyxy, tracker_id in zip(XYXY, TRACKE_ID):
        xyxy = np.array([xyxy])

        _, object_ids, mask_logits = predictor.add_new_prompt(
            frame_idx=0,
            obj_id=tracker_id,
            bbox=xyxy
        )

for index, frame in tqdm(enumerate(frame_generator)):

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        tracker_ids, mask_logits = predictor.track(frame)
        tracker_ids = np.array(tracker_ids)
        masks = (mask_logits > 0.0).cpu().numpy()
        masks = np.squeeze(masks).astype(bool)

        masks = np.array([
            filter_segments_by_distance(mask, distance_threshold=300)
            for mask
            in masks
        ])

        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks=masks),
            mask=masks,
            tracker_id=tracker_ids
        )

    if index % INTERVAL == 0:

        boxes = sv.xyxy_to_xywh(detections.xyxy)
        inputs = pose_estimation_processor(frame, boxes=[boxes], return_tensors="pt").to(DEVICE)

        # This is MOE architecture, we should specify dataset indexes for each image in range 0..5
        inputs["dataset_index"] = torch.tensor([0], device=DEVICE)

        with torch.no_grad():
            outputs = pose_estimation_model(**inputs)

        results = pose_estimation_processor.post_process_pose_estimation(outputs, boxes=[boxes])
        key_points = sv.KeyPoints.from_transformers(results[0])

        annotated_frame = frame.copy()
        annotated_frame = plot_skeletons(key_points=key_points, tracker_ids=detections.tracker_id, frame=annotated_frame)
        annotated_frames.append(annotated_frame)
```
### **Cell 3: 主要處理迴圈**

這是整個專案的核心部分，負責影片的讀取、人物追蹤、骨骼估計和結果儲存。

**程式碼分為幾個主要區塊：**

**1. 初始化與第一幀處理**

- **`PLAYER_ID = 2`**: 定義人物類別的 ID。這通常是根據模型訓練時的類別對應表決定的，這裡 `2` 可能代表 'person'。
    
- **`CONFIDENCE_THRESHOLD = 0.3`, `IOU_THRESHOLD = 0.7`**: 設定物件偵測的置信度閾值和交集比（IoU）閾值，用於過濾低質量的偵測結果。
    
- **`INTERVAL = 30`**: 設定一個間隔，表示每隔 30 幀才進行一次骨骼估計。這是一個優化措施，因為骨骼估計比人物追蹤更耗費計算資源。
    
- **`annotated_frames = []`**: 創建一個空列表，用於儲存帶有骨骼註釋的影像幀。
    
- **`video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)`**: 從影片檔案路徑獲取影片的元數據（如幀率、解析度等）。
    
- **`frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)`**: 創建一個幀生成器，它允許我們逐幀讀取影片，而不是一次性將整個影片載入記憶體。
    
- **`frame = next(frame_generator)`**: 獲取影片的第一幀。
    
- **`result = PLAYER_DETECTION_MODEL.infer(...)`**: 使用一個預先定義好的 `PLAYER_DETECTION_MODEL`（這個模型在使用者提供的程式碼中沒有明確定義，但推測是一個用於人物偵測的模型，例如 YOLOv8）對第一幀進行人物偵測。
    
- **`detections = sv.Detections.from_inference(result)`**: 將偵測結果轉換為 `supervision` 函式庫的 `Detections` 物件，這是一種標準化的數據格式。
    
- **`detections = detections[detections.class_id == PLAYER_ID]`**: 過濾偵測結果，只保留類別 ID 為 `PLAYER_ID`（即人物）的偵測結果。
    
- **`XYXY = detections.xyxy`, `CLASS_ID = detections.class_id`**: 提取邊界框座標和類別 ID。
    
- **`TRACKE_ID = list(range(1, len(CLASS_ID) + 1))`**: 為第一幀中的每個人物手動分配一個唯一的追蹤 ID，從 1 開始。
    
- **`detections = sv.Detections(...)`**: 重新創建 `Detections` 物件，並加入我們手動分配的 `tracker_id`。
    

**2. 影片追蹤器初始化**

- **`with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):`**: 這是一個上下文管理器，用於優化模型推論。
    
    - `torch.inference_mode()`: 禁用梯度計算，這可以減少記憶體使用並加速運算，因為我們只進行推論而不是訓練。
        
    - `torch.autocast("cuda", dtype=torch.bfloat16)`: 使用自動混合精度（AMP）來進行運算，這可以提高在支援的 GPU 上的運算速度，同時保持模型的準確性。
        
- **`predictor.load_first_frame(frame)`**: 這是一個影片追蹤器（推測是 `segment-anything` 或類似模型的追蹤器）的方法，用於載入影片的第一幀作為追蹤的起點。
    
- **`for xyxy, tracker_id in zip(XYXY, TRACKE_ID):`**: 遍歷第一幀中所有偵測到的人物。
    
- **`predictor.add_new_prompt(...)`**: 告訴追蹤器第一幀中每個目標的位置（通過邊界框 `bbox`），這樣追蹤器就知道從哪裡開始追蹤。
    

**3. 主影片處理迴圈**

- **`for index, frame in tqdm(enumerate(frame_generator)):`**: 這是主迴圈，遍歷影片中的每一幀。`tqdm` 函式庫用於顯示進度條，讓使用者知道處理進度。
    
- **`tracker_ids, mask_logits = predictor.track(frame)`**: 在每一幀上，使用追蹤器來追蹤之前被識別的人物。它會返回每個被追蹤人物的 ID 和一個表示其分割掩碼的 logits。
    
- **`masks = (mask_logits > 0.0).cpu().numpy()`**: 將 logits 轉換為二進制掩碼（binary masks）。`> 0.0` 是一個閾值，用於將 logits 轉換為布林值，然後 `.cpu().numpy()` 將其轉換為可供 CPU 處理的 NumPy 陣列。
    
- **`masks = np.array([filter_segments_by_distance(...) for mask in masks])`**: 這部分程式碼用於優化。它可能是一個自定義函式 `filter_segments_by_distance`，用於移除一些距離較遠或無關緊要的分割區域，以提高效率或準確性。
    
- **`detections = sv.Detections(...)`**: 根據追蹤器的結果重新創建 `Detections` 物件，這次包含了分割掩碼和追蹤 ID。
    

**4. 骨骼估計和繪製區塊**

- **`if index % INTERVAL == 0:`**: 檢查當前幀的索引是否是 `INTERVAL` 的倍數。這就是之前提到的優化，只在特定間隔的幀上進行骨骼估計。
    
- **`boxes = sv.xyxy_to_xywh(detections.xyxy)`**: 將邊界框格式從 `xyxy`（左上角和右下角）轉換為 `xywh`（中心點和寬高），這是姿勢估計模型所需要的格式。
    
- **`inputs = pose_estimation_processor(...)`**: 使用 cell 2 中載入的 `pose_estimation_processor` 來預處理當前幀和邊界框，將其轉換為模型所需的張量格式。
    
- **`inputs["dataset_index"] = torch.tensor([0], device=DEVICE)`**: 這是一個特定於 `VitPose` 模型的參數，用於指定資料集索引。
    
- **`with torch.no_grad(): outputs = pose_estimation_model(**inputs)`**: 進行姿勢估計推論。`torch.no_grad()` 再次確保不計算梯度，以節省記憶體和時間。
    
- **`results = pose_estimation_processor.post_process_pose_estimation(...)`**: 對模型的輸出進行後處理，將原始輸出轉換為人類可讀的關鍵點格式。
    
- **`key_points = sv.KeyPoints.from_transformers(results[0])`**: 將後處理結果轉換為 `supervision` 的 `KeyPoints` 物件。
    
- **`annotated_frame = frame.copy()`**: 創建當前幀的副本，以避免直接修改原始幀。
    
- **`annotated_frame = plot_skeletons(...)`**: 呼叫 cell 1 中定義的函式，將關鍵點數據繪製到複製的幀上。
    
- **`annotated_frames.append(annotated_frame)`**: 將處理好的註釋幀添加到列表中，以便後續顯示或儲存。
    

總結來說，cell 3 是一個完整的影片處理流水線，它首先偵測並初始化追蹤目標，然後在每一幀上追蹤這些目標，並在特定的幀上進行更耗時的骨骼估計，最後將結果視覺化並儲存起來。


```python
images = annotated_frames[:5]
titles = [f"frame {index * INTERVAL}" for index in range(0, len(images))]

sv.plot_images_grid(
    images=images,
    grid_size=(5, 1),
    size=(5, 15),
    titles=titles
)
```
### **Cell 4: 結果視覺化**

這個 cell 負責將處理好的結果顯示給使用者。

- **`images = annotated_frames[:5]`**: 從儲存的註釋幀列表中，取出前 5 幀。
    
- **`titles = [f"frame {index * INTERVAL}" for index in range(0, len(images))]`**: 創建一個標題列表，用於標註每張影像對應的原始幀號。由於我們是每隔 `INTERVAL` 幀才處理一次，所以標題會是 `frame 0`, `frame 30`, `frame 60` 等。
    
- **`sv.plot_images_grid(...)`**: 呼叫 `supervision` 函式庫中的函式，以網格形式顯示多張影像。
    
    - `images=images`: 傳入要顯示的影像列表。
        
    - `grid_size=(5, 1)`: 指定網格的大小為 5 行 1 列，這將垂直排列影像。
        
    - `size=(5, 15)`: 設定總顯示區域的大小。
        
    - `titles=titles`: 傳入之前創建的標題列表，為每張影像添加標題。
        

總結來說，這個 cell 是一個簡單但有效的視覺化工具，用於快速檢查骨骼估計的結果，通常會用來驗證整個處理流程是否正確。


```python
def plot_ankles(key_points: sv.KeyPoints, tracker_ids: list[int], frame: np.array):
    key_points = key_points[:, [15, 16]]
    h, w = frame.shape[:2]
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=(w, h))
    for tracker_id in tracker_ids:
        tracker_key_points = key_points[int(tracker_id) - 1]
        annotator = sv.VertexLabelAnnotator(
            color=COLOR.by_idx(tracker_id),
            text_color=sv.Color.BLACK,
            text_scale=text_scale,
            border_radius=2
        )
        frame = annotator.annotate(scene=frame, key_points=tracker_key_points, labels=["L", "R"])
    return frame
```



```python
import cv2
import torch
from tqdm import tqdm

PLAYER_ID = 2
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.7
INTERVAL = 30

annotated_frames = []

video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
frame = next(frame_generator)

result = PLAYER_DETECTION_MODEL.infer(frame, confidence=CONFIDENCE_THRESHOLD, iou_threshold=IOU_THRESHOLD)[0]
detections = sv.Detections.from_inference(result)
detections = detections[detections.class_id == PLAYER_ID]

XYXY = detections.xyxy
CLASS_ID = detections.class_id
TRACKE_ID = list(range(1, len(CLASS_ID) + 1))

detections = sv.Detections(
    xyxy=XYXY,
    tracker_id=np.array(TRACKE_ID)
)

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.load_first_frame(frame)

    for xyxy, tracker_id in zip(XYXY, TRACKE_ID):
        xyxy = np.array([xyxy])

        _, object_ids, mask_logits = predictor.add_new_prompt(
            frame_idx=0,
            obj_id=tracker_id,
            bbox=xyxy
        )

for index, frame in tqdm(enumerate(frame_generator)):

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        tracker_ids, mask_logits = predictor.track(frame)
        tracker_ids = np.array(tracker_ids)
        masks = (mask_logits > 0.0).cpu().numpy()
        masks = np.squeeze(masks).astype(bool)

        masks = np.array([
            filter_segments_by_distance(mask, distance_threshold=300)
            for mask
            in masks
        ])

        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks=masks),
            mask=masks,
            tracker_id=tracker_ids
        )

    if index % INTERVAL == 0:

        boxes = sv.xyxy_to_xywh(detections.xyxy)
        inputs = pose_estimation_processor(frame, boxes=[boxes], return_tensors="pt").to(DEVICE)

        # This is MOE architecture, we should specify dataset indexes for each image in range 0..5
        inputs["dataset_index"] = torch.tensor([0], device=DEVICE)

        with torch.no_grad():
            outputs = pose_estimation_model(**inputs)

        results = pose_estimation_processor.post_process_pose_estimation(outputs, boxes=[boxes])
        key_points = sv.KeyPoints.from_transformers(results[0])

        annotated_frame = frame.copy()
        annotated_frame = plot_ankles(key_points=key_points, tracker_ids=detections.tracker_id, frame=annotated_frame)
        annotated_frames.append(annotated_frame)
```



```python
images = annotated_frames[:5]
titles = [f"frame {index * INTERVAL}" for index in range(0, len(images))]

sv.plot_images_grid(
    images=images,
    grid_size=(5, 1),
    size=(5, 15),
    titles=titles
)

```



```python

```


