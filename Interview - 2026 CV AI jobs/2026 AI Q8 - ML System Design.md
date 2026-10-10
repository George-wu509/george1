
|                           |     |
| ------------------------- | --- |
| [[#### ML System Design]] |     |
|                           |     |
|                           |     |

#### ML System Design
```
請深入詳細回答ML System Design：Senior 以上的核心面試請以具體例子深入完整回答考題可能是：

> Design an end-to-end automated visual inspection system that can detect tiny defects from multiple cameras and continuously improve after deployment.

這類題目可能需要你在白板上設計：

Camera / Lighting / Motion Control
Acquisition / Quality Check / Image Processing
Detection / Segmentation / Feature Extraction
Decision / Confidence / Anomaly Handling
Storage / Review / Monitoring / Audit
Dataset Versioning / Retraining / Deployment

面試時還會針對每個環節深入追問：

- What if one camera fails?
- How do you handle missing images?
- How do you make the pipeline recoverable?
- How do you choose decision thresholds?
- How do you monitor false negatives?
- How do you prevent data leakage between training and testing?
- How do you safely deploy a new model across 100 machines?
```
