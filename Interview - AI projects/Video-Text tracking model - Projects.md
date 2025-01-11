#### Title:
Video-Text Retrieval and tracking Multimodal model


---
##### **Resume Keyworks:
<mark style="background: #FF5582A6;">SAM</mark> ,<mark style="background: #FF5582A6;">SAM2</mark>, <mark style="background: #FF5582A6;">CLIP</mark>

##### **STEPS:**
step0. Load sam, sam2 and clip  
1. SAM1的自動mask生成器(<mark style="background: #BBFABBA6;">SamAutomaticMaskGenerator</mark>)可以生成一張圖片中所有物體和背景的masks. 先create一個 mask_generator = SamAutomaticMaskGenerator()
1. SAM2用於追蹤和傳播分割結果. 載入SAM2用 predictor = <mark style="background: #BBFABBA6;">build_sam2_video_predictor</mark>.  並用predictor.init_state(video_path)初始化sam2
2. CLIP用於將輸入的text和segmentation mask連結. 初始化CLIP用 <mark style="background: #BBFABBA6;">clip_model, clip_preprocess</mark> = clip.load(), 

step1. CLIP處理text
1. 將輸入的text (example: red shoes), 用clip.tokenize() 轉成輸入的tokenized_text
2. text_features = model.encode_text(tokenized_text) 轉成text的features
3. normalize text_features

step2. 用SAM1從第一幀開始生成full segmentation
1. masks = mask_generator.generate(image_time0)

step3. 將第一幀開始所有mask轉成CLIP image features, 並比對text_features
1. preprocessed_mask = preprocess(mask) 
2. mask_features = model.encode_image(preprocessed_mask)
3. normalize mask_features 

step4. 第一幀開始, 如果在某一幀某個mask跟text_features score高於門檻值, 則為起始mask
1. probs = 100. * mask_features @ text_features.T > Threshold

step5. 從起始mask用SAM2的predictor進行追蹤並標示每一幀的masks
1. # 設定初始遮罩信息 initial_info = { "frame_idx": M, # 初始幀索引 "mask": initial_mask, # 初始遮罩 }
2. tracked_masks = video_predictor.predict(frames, initial_info)

---

#### Resume: 
Developed a multimodal model integrating SAM2 and CLIP for text-prompt-based video object retrieval, segmentation, and tracking, enabling accurate frame matching, high-precision segmentation, and multi-object tracking and full segmentation of the video.

#### Abstract: 
**Video-Text Retrieval and Tracking Multimodal Model** is an advanced framework that integrates **SAM2 (Segment Anything Model 2)** and **CLIP (Contrastive Language-Image Pretraining)** to enable precise object retrieval, segmentation, and tracking in videos using natural language prompts. This project is designed to identify and track multiple objects across video frames based on user-defined textual queries.

Key features include:

- **Text-Based Frame Retrieval**: Utilizes CLIP to match video frames with input text prompts for accurate frame selection.
- **High-Precision Segmentation**: Leverages SAM2 to generate segmentation masks for objects of interest in selected frames.
- **Multi-Object Tracking**: Tracks segmented objects across video frames, displaying their trajectories and segmentation masks in real-time.
- **Efficient Processing**: Supports GPU acceleration and batch processing for handling large-scale video data.

Link: 
[video_predictor_example.ipynb](https://colab.research.google.com/github/facebookresearch/segment-anything-2/blob/main/notebooks/video_predictor_example.ipynb)
[CLIP-SAM github](https://github.com/maxi-w/CLIP-SAM)
[sam2_clip2_notebook](https://colab.research.google.com/drive/1JpszCuwzk92rZ9t8dQEMUagRd83CvNAq)
[Image Segmentation Using Text and Image Prompts](https://github.com/timojl/clipseg)
[AutoSeg-SAM2](https://github.com/zrporz/AutoSeg-SAM2/tree/main)

#### Technique detail:
這個多模態模型結合了 SAM2 (Segment Anything Model 2) 與 CLIP (Contrastive Language-Image Pretraining)，專注於基於文本提示的影片物體檢索與追蹤。以下是模型的核心技術細節與實現流程：

1. 整體架構
CLIP 的文本-圖像匹配
CLIP 能夠將文字與圖像轉換為共同的嵌入空間，計算兩者相似性。模型利用 CLIP 找出與文本提示最相符的影片幀。

SAM2 的分割能力
SAM2 被用於高效的物體分割，為指定幀中的目標物體生成精確遮罩 (mask)。此分割結果進一步用於物體的追蹤。

多物體追蹤
模型採用了逐幀追蹤策略，結合分割結果與軌跡推斷技術來標記物體位置。

2. 流程設計
初始化與數據處理

使用 SAM2 的 init_state 函數，處理並加載影片幀數據。
CLIP 將文本提示轉換為嵌入向量 (encode_text)。
圖像數據經 CLIP 預處理 (clip_transform) 並嵌入為圖像特徵。
檢索與相似度計算

利用 CLIP 計算每幀與文本嵌入的相似性 (retrieve_frames)。
根據相似度篩選出目標幀。
物體分割與追蹤

SAM2 使用 add_new_points_or_box 函數對幀中物體生成遮罩。
對於多幀傳播 (propagate_in_video)，追蹤物體並記錄其中心位置。
輸出可視化

疊加物體遮罩及軌跡，生成包含追蹤與分割結果的輸出影片。
#### **3. 技術細節與代碼示例**

以下是主要的關鍵類別與函數：

1. **類別 `VideoTextRetrievalTracker`**
    
    - `encode_text(text)`：將文本提示嵌入 CLIP。
    - `encode_image(image)`：將圖像嵌入 CLIP，提取特徵。
    - `retrieve_frames()`：計算影片幀與文本提示的相似性。
    - `multi_object_tracking()`：結合分割與追蹤技術，逐幀處理物體遮罩與軌跡。
2. **追蹤與輸出**
    
    - 遮罩生成：在追蹤幀中，SAM2 提供分割結果 (`add_new_points_or_box`)。
    - 軌跡疊加：通過 `cv2.line` 將累積軌跡線繪製到輸出影片。

#### **4. 優化與挑戰**

1. **計算效率**
    
    - 為減少多幀處理的計算成本，可使用模型量化技術或分批次處理。
    - 提供 GPU 加速支持以提升處理速度。
2. **模型精度**
    
    - 微調 CLIP 和 SAM2，以提高特定應用場景（如多物體重疊或快速移動目標）的效果。
    - 結合自監督學習方法改善多模態檢索準確性。
3. **可視化與輸出**
    
    - 增強輸出視頻的清晰度和直觀性，例如標記物體類別與分割遮罩的區分。

---

透過結合 CLIP 的檢索能力與 SAM2 的分割精度，該模型在文本引導的視頻物體追蹤應用中具有廣泛潛力，適用於監控分析、自動駕駛視頻檢測等場景。