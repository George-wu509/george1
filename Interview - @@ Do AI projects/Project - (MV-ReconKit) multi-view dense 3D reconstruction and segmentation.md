
```
請幫我詳細規劃以下的研究計畫: Designed a multi-view dense 3D reconstruction and segmentation pipeline using SfM and MVS, incorporating feature matching, dense point cloud and surface reconstruction, AI based 3D segmentation, and feature-based minimal image selection. Enhanced efficiency and quality with evaluation metrics for geometric accuracy, model completeness. 

希望這是放在Github做成一個python library. 並要有很多及完整對每個功能的過程及結果的各式data visualization functions, 也要針對效能進行優化. 請提供每個functions的code也請提供github 英文readme, 以及安裝python environment的方便方法譬如pyproject.toml, yaml file. 並附上兩個colab範例程式碼python code.

並要對model做各種優化譬如knowledge distrillation, CUDA kernel optimization加速inference等等, 並用onnxruntime跟TensorRT加速inference. 如果可以的話在這個python library提供Pytorch跟JAX版本, 也提供training的functions可以用AWS或其他方式做Distributed training and Optimization.

```

MV-ReconKit Github:  https://github.com/George-wu509/MV-ReconKit

Resume:
**Multi-View 3D Reconstruction Framework with Optimized AI Segmentation:**

Developed a comprehensive Python pipeline that automatically transforms multi-camera 2D images into dense, semantically segmented 3D models. The system integrates classical SfM/MVS algorithms with a custom-trained PointNet++ model for high-accuracy segmentation featuring dual-backend support for both PyTorch and JAX.