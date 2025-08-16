
2025.08.14   DINOv3

我們推出了[DINOv3](http://ai.meta.com/dinov3)，它可以擴展影像的自監督學習，以創建通用視覺主幹，在包括網路和衛星影像在內的不同領域實現絕對最先進的性能。DINOv3 主幹網路能夠產生強大的高解析度影像特徵，從而輕鬆訓練輕量級適配器。這使得其在一系列下游視覺任務（包括影像分類、語義分割和影片中的物件追蹤）上表現出色。透過提供較小的模型來增強 DINOv3 的多功能性，這些模型在廣泛的評估套件中表現優於基於 CLIP 的同類衍生產品，以及針對資源受限用例的替代 ConvNeXt 架構

今天，我們發布了[DINOv3](http://ai.meta.com/dinov3)，這是一款通用的、先進的電腦視覺模型，採用 SSL 進行訓練，能夠產生卓越的高解析度視覺特徵。這是首次在多個長期存在的密集預測任務（包括物件偵測和語義分割）上，單一凍結視覺主幹網路的表現優於專用解決方案. 我們建立了 DINOv3，並在比其前身DINOv2大 12 倍的資料集上訓練了一個 7 倍大的模型。為了展示模型的多功能性，我們在 15 個不同的視覺任務和 60 多個基準測試中對其進行了評估。 DINOv3 主幹在所有密集預測任務中表現尤為出色，展現了對場景佈局和底層物理的卓越理解。

將 DINOv3 擴展到 7B 參數展現了 SSL 的全部潛力。然而，7B 模型對於許多下游應用而言並不實用。根據社群的回饋，我們建立了一系列涵蓋廣泛推理運算需求的模型，以賦能研究人員和開發者來應對各種用例。透過將 ViT-7B 模型提煉為更小、性能更高的變體（例如 ViT-B 和 ViT-L），DINOv3 在廣泛的評估套件中均優於基於 CLIP 的同類模型。此外，我們也引進了從 ViT-7B 提煉而來的 ConvNeXt 替代架構（T、S、B、L），以適應不同的運算限制。我們也發布了提煉流程，以便社群在此基礎上進行建置。

![[Pasted image 20250814203608.png]]

![[Pasted image 20250814203653.png]]

Reference: 
[1]
[DINOv3](https://ai.meta.com/blog/dinov3-self-supervised-vision-model/): Self-supervised learning for vision at unprecedented scale

[2] DINOv3 [github](https://github.com/facebookresearch/dinov3)

[3] DINOv3 [paper](https://ai.meta.com/research/publications/dinov3/)

[4] My DINOv3 colab [DINOv3_segmentation_tracking.ipynb](https://colab.research.google.com/drive/1IBQ4chTxowsBE_wYONRCmjcBnuSIOdkv#scrollTo=3S1MyIZucBoD)

