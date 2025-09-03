

```
請幫我詳細規劃以下的研究計畫並做成python library. 這個Project是做交通車輛video或drone的object detection跟tracking. 訴求是非常快非常小的Latency的object detection而且可以針對非常多目標. object detection model有幾個選擇可以讓user選, (1)RF-DETR或(2)最新或最適合的YOLO model, 如果有更快得的models也加入. 也要有functions可以輸出segmentation masks. 並要對model做各種優化譬如knowledge distrillation, CUDA kernel optimization加速inference等等, 並用onnxruntime跟TensorRT加速inference. 如果可以的話在這個python library提供Pytorch跟JAX版本, 也提供training的functions可以用AWS或其他方式做Distributed training and Optimization. 

```

FastTrack github
https://github.com/George-wu509/FastTrack

Resume:
**FastTrack: High-Performance Multi-Object Detection and Tracking:**

Engineered FastTrack, a high-performance Python library for real-time multi-object tracking in traffic and drone videos. The system features a flexible dual-backend (PyTorch/JAX) architecture with SOTA models like YOLOv9. Achieved significant inference speedup and minimal latency by implementing advanced optimizations including TensorRT, ONNX conversion, and knowledge distillation.
