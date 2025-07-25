



|                                        | AI Model<br>(No Transformer)                                                           | Transformer<br>(closeset) | Grounding<br>(openset/zero-shot)                                  |
| -------------------------------------- | -------------------------------------------------------------------------------------- | ------------------------- | ----------------------------------------------------------------- |
| Image classification                   |                                                                                        |                           |                                                                   |
| Image<br>object detection              | RCNN series<br>YOLO(FCOS)<br>Lightweight<br>RetainNet<br>EfficientDet<br>MobileNet-SSD | DETR<br>DINO              | GLIP<br>Grounding DINO                                            |
| Image<br>segmentation                  | RCNN Series<br>UNet(FCN)                                                               | DETR                      | SAM<br>Grounded SAM                                               |
| Video<br>object detection<br>/Tracking | YOLO<br>+<br>SORT<br>DeepSORT<br>ByteSORT<br>(ReID)                                    | Video-DETR<br>TrackFormer | GroundingDINO + Tracker<br>Grounded-SAM + Tracker<br>+<br>Tracker |
| Video<br>segmentation                  | MaskTrack RCNN<br>Flow-warping                                                         | SAM2                      | Track Anything Model (TAM)<br>Grounded SAM + Tracker              |
