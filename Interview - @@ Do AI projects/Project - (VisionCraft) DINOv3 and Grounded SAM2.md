
```
請幫我詳細規劃以下的研究計畫. 這個Project是基於DINOv3跟Grounded SAM2 or SAM or SAM2跟其他需要的libraries. 輸入可以是一張image or 一段video. 這個package可以實現user輸入text譬如"穿藍色衣服的人", 就可以在image或video上顯示多個符合的object detection box跟segmentaqtion masks, 在video上即使有遮擋還是可以tracking, 在影片中也要維持跨幀一致的遮罩與 ID. 在image跟video也可以根據text顯示High-resolution dense features. 另外一個功能就是可以實現類似SAM的Automatic Mask Generation在image跟video上, 尤其在video上的Automatic Mask Generation在每個frame要一致. 在基於以上功能並加上譬如輸入prompt: "把bench換成sofa "可以實現stable diffusion的image跟video inpainting功能. 希望這是放在Github做成一個python library. 也請提供github 英文readme, 以及安裝python environment的方便方法譬如yaml file. 並附上兩個colab範例程式碼. 

```

vision-craft github: https://github.com/George-wu509/VisionCraft

My Colab: https://colab.research.google.com/drive/1Zk540B_ItqKnylZUJfDMG7vZaV595OC6

Resume:
**Open-Vocabulary Object Tracking and Segmentation in Dynamic Video Scenes using DINOv3:**
**使用DINOv3在動態視訊場景中實現開放詞彙目標追蹤和分割** 

Architected and developed a unified Python library for multi-task zero-shot video understanding by integrating the self-supervised model DINOv3 with state-of-the-art foundation models (Grounding DINO, SAM2, Stable Diffusion). Built an end-to-end platform that transforms prompts into complex visual tasks, supporting robust multi-object video tracking with occlusion handling, automatic segmentation mask generation, and generative video inpainting.

設計並開發了一個統一的Python庫，用於實現多任務零樣本視訊理解，該庫整合了自監督模型DINOv3和一系列領先的基模型（Grounding DINO、SAM2、Stable Diffusion）。該平台能夠將用戶輸入的提示訊息轉化為複雜的視覺任務，支援穩健的多目標視訊追蹤（包括遮擋處理）、自動分割遮罩生成以及生成式視訊修復等功能。

-> 
使用Grounded SAM2讓我們能依據輸入prompt產生bbox, 結合DINOv3的Segmentation Tracking [colab](https://github.com/facebookresearch/dinov3/blob/main/notebooks/segmentation_tracking.ipynb)可以產生video segmentation masks,  也可像SAM可以自動產生每frame的全自動segmentation masks [colab](https://colab.research.google.com/github/facebookresearch/sam2/blob/main/notebooks/automatic_mask_generator_example.ipynb). 並可以支援stable diffusion的inpainting可以將某物體換成另外一個物體, 而且實現在全video裡



