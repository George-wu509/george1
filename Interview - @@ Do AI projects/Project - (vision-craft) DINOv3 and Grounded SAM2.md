
```
請幫我詳細規劃以下的研究計畫. 這個Project是基於DINOv3跟Grounded SAM2 or SAM or SAM2跟其他需要的libraries. 輸入可以是一張image or 一段video. 這個package可以實現user輸入text譬如"穿藍色衣服的人", 就可以在image或video上顯示多個符合的object detection box跟segmentaqtion masks, 在video上即使有遮擋還是可以tracking, 在影片中也要維持跨幀一致的遮罩與 ID. 在image跟video也可以根據text顯示High-resolution dense features. 另外一個功能就是可以實現類似SAM的Automatic Mask Generation在image跟video上, 尤其在video上的Automatic Mask Generation在每個frame要一致. 在基於以上功能並加上譬如輸入prompt: "把bench換成sofa "可以實現stable diffusion的image跟video inpainting功能. 希望這是放在Github做成一個python library. 也請提供github 英文readme, 以及安裝python environment的方便方法譬如yaml file. 並附上兩個colab範例程式碼. 

```

vision-craft github: https://github.com/George-wu509/VisionCraft

My Colab: https://colab.research.google.com/drive/1Zk540B_ItqKnylZUJfDMG7vZaV595OC6

Resume:
**Open-Vocabulary Object Tracking and Segmentation in Dynamic Video Scenes using DINOv3:**

Architected and developed a unified Python library integrating multiple SOTA foundation models (DINOv3, SAM2, Grounding DINO, Video Diffusion). This end-to-end platform translates natural language prompts into complex visual tasks, including robust object tracking with occlusion handling and generative video inpainting. Delivered as a modular, installable package, showcasing advanced system design and model integration skills.
