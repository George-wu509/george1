
[sam2_colab](https://github.com/facebookresearch/sam2/blob/main/notebooks/image_predictor_example.ipynb)
[video_predictor_example(my colab)](https://colab.research.google.com/drive/1gviU1l1t-14aKwglsB-cg7-9ZsBxMjIs#scrollTo=3c3b1c46-9f5c-41c1-9101-85db8709ec0d)

#### 1. Import
```python
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
```

#### 2. Load Model
```python hlt:sam2
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

predictor = SAM2ImagePredictor(sam2_model)

```


#### 3. Image segmentation
```python hlt:predictor
# Load image
image = Image.open('images/truck.jpg')
image = np.array(image.convert("RGB"))

# Selecting objects with SAM2
predictor.set_image(image)

# choose point for segmentation
input_point = np.array([[500, 375]])
input_label = np.array([1])


masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)
sorted_ind = np.argsort(scores)[::-1]
masks = masks[sorted_ind]
scores = scores[sorted_ind]
logits = logits[sorted_ind]

```


#### 4. Video segmentation - using points
sem2 video segmentation [notebook](https://colab.research.google.com/drive/1gviU1l1t-14aKwglsB-cg7-9ZsBxMjIs#scrollTo=f5f3245e-b4d6-418b-a42a-a67e0b3b5aec)

```python hlt:predictor, hlredt:add_new_points_or_box,propagate_in_video
from sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Create the sam2 video predictor
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

# Import video into predictor (first time)
inference_state = predictor.init_state(video_path=video_dir)
# Reset init_state(run if not first time)
predictor.reset_state(inference_state)

# Example1 - add 2 positive click and do segmentation
ann_frame_idx = 0, ann_obj_id = 1
points = np.array([[210, 350], [250, 220]], dtype=np.float32)
labels = np.array([1, 1], np.int32)

_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,)

# show segmentation mask (1 positive click) for one referenced frame
show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

# run propagation throughout the video
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# add positive click to refine the masklet
ann_frame_idx = 150, ann_obj_id = 1
points = np.array([[82, 410]], dtype=np.float32)
labels = np.array([0], np.int32)   #1:positive click, 0:negative click

_, _, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,)
    
# run propagation throughout the video (after refine using negative click)
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

```

#### 5. Video segmentation - using bounding box
sem2 video segmentation [notebook](https://colab.research.google.com/drive/1gviU1l1t-14aKwglsB-cg7-9ZsBxMjIs#scrollTo=f5f3245e-b4d6-418b-a42a-a67e0b3b5aec)
```python hlt:predictor, hlredt:add_new_points_or_box,propagate_in_video
# if you have run any previous tracking using this `inference_state`, please reset it first via `reset_state`.
predictor.reset_state(inference_state)

ann_frame_idx = 0, ann_obj_id = 4
# Let's add a box at (x_min, y_min, x_max, y_max) = (300, 0, 500, 400) to get started
box = np.array([300, 0, 500, 400], dtype=np.float32)
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    box=box,
)

# Combine the bounding box and positive/negative click
# Let's add a positive click at (x, y) = (460, 60) to refine the mask
points = np.array([[460, 60]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1], np.int32)
# note that we also need to send the original box input along with
# the new refinement click together into `add_new_points_or_box`
box = np.array([300, 0, 500, 400], dtype=np.float32)
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
    box=box,
)

# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }


```

#### 6. Multiple objects video segmentation
sem2 video segmentation [notebook](https://colab.research.google.com/drive/1gviU1l1t-14aKwglsB-cg7-9ZsBxMjIs#scrollTo=f5f3245e-b4d6-418b-a42a-a67e0b3b5aec)
```python hlt:predictor, hlredt:add_new_points_or_box,propagate_in_video, hlyellowt=prompts
predictor.reset_state(inference_state)

prompts = {}  # hold all the clicks we add for visualization

ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 2  # give a unique id to each object we interact with (it can be any integers)

# Let's add a positive click at (x, y) = (200, 300) to get started on the first object
points = np.array([[200, 300]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1], np.int32)
prompts[ann_obj_id] = points, labels
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)

# add the first object
ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 2  # give a unique id to each object we interact with (it can be any integers)

# Let's add a 2nd negative click at (x, y) = (275, 175) to refine the first object
# sending all clicks (and their labels) to `add_new_points_or_box`
points = np.array([[200, 300], [275, 175]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 0], np.int32)
prompts[ann_obj_id] = points, labels
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
```

### **1. 架構概述**

這段代碼涉及使用 **SAM2 (Segment Anything Model 2)** 進行視頻的目標分割。主要功能包括：

1. **添加新點或框標註目標物件**。
2. **在視頻中進行分割的時間傳播**。

這些操作是基於 `SAM2` 的 `video predictor` 來完成的，`video predictor` 用於處理一系列視頻幀並對每一幀進行分割。

---

### **2. 變數及參數說明**

以下是代碼中每個變數的詳細解釋：

#### **a. `predictor`**

- **定義**： `predictor` 是基於 `build_sam2_video_predictor` 創建的物件，用於執行視頻的分割任務。
- **參數**：
```python hlyellowt:build_sam2_video_predictor
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
```

    - `model_cfg`: **模型配置文件**，包含了 SAM2 模型的結構和參數。
    - `sam2_checkpoint`: **預訓練檢查點文件**，存儲了模型的權重。
    - `device`: **運行設備**，通常是 `'cuda'`（GPU）或 `'cpu'`。
    
- **用途**： `predictor` 包含兩個主要方法：
    1. `add_new_points_or_box`: 添加點或框來標注目標。
    2. `propagate_in_video`: 從某幀開始，將分割結果傳播到視頻中的其他幀。

#### **b. `add_new_points_or_box`**

- **功能**： 在指定幀上添加新的點或框，用於初始化目標分割。
- **參數**：
```python hlbluet:predictor
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,)
```
    
    - `inference_state`: **推理狀態**，存儲了當前視頻分割的中間狀態。
    - `frame_idx`: **幀索引**，表示在哪一幀上添加標註。
    - `obj_id`: **目標 ID**，分配給新添加的目標的唯一標識符。
    - `points`: **點列表**，用於指定目標的位置。例如 `[[x1, y1], [x2, y2]]`。
    - `labels`: **標籤列表**，對應於點的類別（`1` 表示前景，`0` 表示背景）。
- **返回值**：
    - `_`: 不需要的輸出。
    - `out_obj_ids`: 添加後的目標 ID 列表。
    - `out_mask_logits`: 添加後的目標分割掩膜（logits）。

#### **c. `propagate_in_video`**

- **功能**： 從已知的分割結果開始，將分割結果傳播到視頻的其他幀。
- **參數**：
```python hlbluet:predictor
`for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):`
```
    
    - `inference_state`: **推理狀態**，維持視頻的分割信息。
- **返回值**：
    - `out_frame_idx`: 當前分割的幀索引。
    - `out_obj_ids`: 當前幀上的目標 ID。
    - `out_mask_logits`: 當前幀的分割掩膜（logits）。

#### **d. `video_segments`**

- **定義**： 字典結構，用於存儲每幀的分割結果。
- **結構**：
```python
video_segments[frame_idx] = {obj_id: mask_array}

```
    - `frame_idx`: 幀索引。
    - `obj_id`: 目標 ID。
    - `mask_array`: 分割掩膜（邏輯值）。

---

### **3. 方法詳細說明與具體例子**

#### **a. `add_new_points_or_box` 的使用**

- **完整用法**：
```python hlbluet:predictor
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=10,  # 標註的幀索引
    obj_id=1,      # 目標物件 ID
    points=[[100, 150], [200, 250]],  # 標註的兩個點
    labels=[1, 0],  # 第一點為前景 (1)，第二點為背景 (0)
)

```
    
- **示例解釋**：
    - **場景**：在第 10 幀上標注一個新目標，添加兩個點（100, 150 是前景，200, 250 是背景）。
    - **執行後**：
        - `out_obj_ids`: 返回目標 ID（例如 `[1]`）。
        - `out_mask_logits`: 返回分割掩膜（logits），形狀可能為 `(H, W)`。

---

#### **b. `propagate_in_video` 的使用**

- **完整用法**：
```python
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

```
    
- **示例解釋**：
    - **場景**：從已知的標註幀開始，將分割結果傳播到整個視頻。
    - **操作步驟**：
        1. `out_frame_idx`: 當前傳播到的幀索引。
        2. `out_obj_ids`: 當前幀中識別的目標 ID（如 `[1, 2]`）。
        3. `out_mask_logits`: 當前幀的掩膜 logits（如形狀為 `(N, H, W)`）。
        4. 通過 `(out_mask_logits[i] > 0.0)` 將 logits 轉換為邏輯掩膜，並存入字典。

---

### **4. 實例整合應用**

假設有一個 30 幀的視頻，我們希望在第 5 幀上標註一個前景點，並傳播分割到所有幀：
```python
# 初始化 predictor 和推理狀態
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device='cuda')
inference_state = predictor.init_inference(video_frames)  # 加載視頻幀

# 添加標註
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=5,  # 標註第 5 幀
    obj_id=1,  # 目標 ID
    points=[[100, 200]],  # 前景點
    labels=[1],  # 前景標籤
)

# 傳播分割到視頻
video_segments = {}
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

```

- **結果**：
    - `video_segments` 是每幀的分割結果字典。
    - 每幀結果包含分割掩膜，形狀為 `(H, W)`，值為 `True` 或 `False`。