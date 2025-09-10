

```
請幫我詳細規劃以下的研究計畫並做成python library. 這個Project是做交通車輛video或drone的object detection跟tracking. 訴求是非常快非常小的Latency的object detection而且可以針對非常多目標. object detection model有幾個選擇可以讓user選, (1)RF-DETR或(2)最新或最適合的YOLO model, 如果有更快得的models也加入. 也要有functions可以輸出segmentation masks. 並要對model做各種優化譬如knowledge distrillation, CUDA kernel optimization加速inference等等, 並用onnxruntime跟TensorRT加速inference. 如果可以的話在這個python library提供Pytorch跟JAX版本, 也提供training的functions可以用AWS或其他方式做Distributed training and Optimization. 

```

FastTrack github
https://github.com/George-wu509/FastTrack

Resume:
**FastTrack: High-Performance Multi-Object Detection and Tracking:**
**FastTrack：高效能多目標偵測與追蹤技術**

Engineered FastTrack, a high-performance Python library for real-time multi-object tracking in traffic and aerial (drone) videos. Designed a flexible dual-backend architecture (PyTorch/JAX) supporting SOTA detection and tracking models such as YOLOv10, RT-DETR and DeepSORT. Achieved substantial inference speedups and ultra-low latency through advanced optimizations, including TensorRT, CUDA kernel integration and knowledge distillation.
開發了FastTrack，這是一個高性能的Python庫，專用於即時處理交通和無人機視訊中的多目標追蹤。本函式庫採用靈活的雙後端架構（PyTorch/JAX），支援YOLOv10、RT-DETR和DeepSORT等最先進的目標偵測和追蹤模型。透過TensorRT、CUDA核心整合和知識蒸餾等高級優化技術，實現了顯著的推理速度提升和極低的延遲。
