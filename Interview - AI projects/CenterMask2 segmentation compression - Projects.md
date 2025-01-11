#### Title:
Commercial AI image segmentation model development, deployment and compression:


---
##### **Resume Keyworks:
<mark style="background: #BBFABBA6;">CenterMask2</mark>, <mark style="background: #BBFABBA6;">UNet</mark>, <mark style="background: #BBFABBA6;">onnxruntime</mark>, <mark style="background: #ADCCFFA6;">TensorRT</mark>, <mark style="background: #BBFABBA6;">Model compression</mark>

##### **STEPS:**
step0. Data preparation
1. Prepare cell images 

step1. Build CenterMask2 semantic and instance segmentation models

step3. Build U-Net prediction model

step2. Model training on AzureML

step3. Fine-tune segmentation models

step4. Convert Pytorch models into ONNX models

step5. Speed up Model using TendorRT

step6. Model deployment on c++ using onnxruntime

step7. Model compression using pruning, quantization, and knowledge distillation.

---


#### Resume: 
Developed an end-to-end solution for microscopy image segmentation and cell viability prediction, integrating CenterMask2 and U-Net models. Optimized deployment with ONNX Runtime and TensorRT, achieving scalable, efficient inference. Enhanced model performance using pruning, quantization, and knowledge distillation for biomedical research applications.

#### Abstract: 
This project focuses on developing advanced **microscopy image segmentation and prediction models** for biological research and analysis. It integrates **Detectron2's CenterMask2** for semantic and instance segmentation of microscopy images and a **U-Net-based model** for predicting live or dead cells using fluorescence intensity.

Key features include:

- **Data Preparation**: Supports COCO-format datasets for segmentation and customized annotations for cell viability prediction.
- **Model Training**: Utilizes Azure Machine Learning for scalable, GPU-accelerated training.
- **ONNX Deployment**: Converts PyTorch models to ONNX format for efficient inference using **ONNX Runtime** in C++.
- **Performance Optimization**: Leverages **TensorRT** for inference acceleration and implements model compression techniques such as pruning, quantization, and knowledge distillation to reduce memory usage and improve efficiency.

This end-to-end solution enhances microscopy data analysis with accurate, scalable, and deployable AI models, providing valuable tools for biomedical research and diagnostics. Explore the repository for datasets, training scripts, and deployment workflows.

#### Technique detail: 

### 建立Microscopy Image Segmentation Models項目詳細流程與原理解釋

#### **1. 項目概述**

此項目目的是建立兩種顯微影像分割模型：

1. **基於Detectron2的CenterMask2模型**：
    - 用於實現**語義分割（semantic segmentation）**和**實例分割（instance segmentation）**。
2. **基於U-Net模型的細胞存活性預測**：
    - 基於螢光濃度（fluorescent concentration），判斷細胞為「存活（live）」或「死亡（dead）」。

模型訓練後，將PyTorch格式的模型轉換為ONNX格式，並使用**ONNX Runtime**進行部署，配合**TensorRT**進行推理加速。同時，應用模型壓縮技術（如剪枝、量化、知識蒸餾）進一步優化性能和內存占用。

---

#### **2. 數據準備與預處理**

1. **數據集構建**：
    
    - 來源：螢光顯微影像，標注數據分為：
        - 語義分割標注：每個像素的類別。
        - 實例分割標注：區分不同的細胞實例。
        - 細胞存活性標注：以螢光強度區分「存活」與「死亡」。
2. **格式化為常見格式**：
    
    - **COCO格式**：適用於Detectron2模型。
    - 自定義標籤格式：適用於U-Net模型，通常以圖像-標籤對應方式存儲。
3. **數據增強（Data Augmentation）**：
    
    - 水平翻轉、旋轉、隨機裁剪、顏色抖動等增強技術，增加數據多樣性。
4. **數據加載**：
    
    - 使用PyTorch的**DataLoader**或Detectron2內建的數據加載工具，對數據進行批處理。

---

#### **3. 模型選擇與設計**

##### **(1) Detectron2 + CenterMask2**

- **Detectron2**：Facebook AI開源的目標檢測框架，支援多種分割和檢測模型。
- **CenterMask2**：在Mask R-CNN的基礎上，結合FCOS（Fully Convolutional One-Stage Object Detection）進行快速高準確率的分割。

**流程**：

1. **Backbone**：ResNet 或 ResNeXt 提取特徵。
2. **FCOS**：生成bounding box和分類信息。
3. **Mask Head**：通過分支生成每個目標的細化分割掩碼。

**適用場景**：

- **語義分割**：所有細胞的統一分割。
- **實例分割**：區分每個細胞的輪廓，便於後續分析。

##### **(2) U-Net for Live/Dead Cell Prediction**

- **U-Net架構**：
    - **Encoder**：使用卷積層提取特徵。
    - **Decoder**：進行特徵上採樣（Upsampling），生成與輸入大小相同的分割圖。
    - **跳躍連接（Skip Connections）**：保留高分辨率特徵，提升分割精度。

**設計目標**：

- 使用螢光濃度圖像作為輸入。
- 標籤類別為「存活」與「死亡」。

---

#### **4. 模型訓練**

##### **(1) 在Azure Machine Learning上訓練**

1. **設置環境**：
    - 創建訓練環境，包含PyTorch、Detectron2等依賴庫。
2. **提交訓練作業（Job）**：
    - 通過Azure SDK或Python腳本將數據集和模型腳本上傳至Azure。
    - 使用Azure ML的**GPU計算資源**進行大規模並行訓練。

##### **(2) 優化策略**：

- **損失函數（Loss Function）**：
    - 語義與實例分割：交叉熵損失（Cross Entropy Loss）、IoU Loss。
    - 細胞存活性預測：二元交叉熵（Binary Cross Entropy）。
- **學習率調度（Learning Rate Scheduler）**：使用余弦衰減或循環學習率。
- **數據平衡**：針對存活與死亡樣本數據不平衡問題，調整權重。

---

#### **5. 模型轉換與部署**

##### **(1) PyTorch to ONNX**

1. **轉換過程**：
    
    - 使用`torch.onnx.export()`將PyTorch模型轉為ONNX格式。
    - 定義動態輸入尺寸，提升部署靈活性。
2. **模型驗證**：
    
    - 使用ONNX Runtime進行模型推理，確保輸出與PyTorch模型一致。

##### **(2) 部署到C++設備**

1. **集成ONNX Runtime**：
    - 使用C++ API調用ONNX模型，實現設備端推理。
2. **性能優化（TensorRT）**：
    - 通過TensorRT進行FP16或INT8量化，加速推理。

---

#### **6. 模型壓縮與優化**

##### **(1) 剪枝（Pruning）**

- 刪除冗餘權重與結構，減少模型大小與推理計算量。

##### **(2) 量化（Quantization）**

- 將模型權重從FP32降至FP16或INT8，提高推理速度並減少內存使用。

##### **(3) 知識蒸餾（Knowledge Distillation）**

1. **設計師生模型（Teacher-Student Model）**：
    - 讓大型模型（Teacher）指導小型模型（Student）的訓練。
2. **目標**：
    - 在保持性能的同時，顯著降低模型大小。

---

#### **7. 項目亮點**

1. **高效分割**：
    - Detectron2和CenterMask2模型的結合實現高準確率的語義與實例分割。
2. **功能多樣性**：
    - U-Net模型能基於螢光濃度進行細胞存活性預測，擴展了應用場景。
3. **完整端到端流程**：
    - 涵蓋從訓練到部署的全過程，包括模型轉換、壓縮與性能優化。
4. **多技術融合**：
    - 結合了AI模型壓縮技術（剪枝、量化、知識蒸餾）與推理加速技術（ONNX Runtime和TensorRT）。