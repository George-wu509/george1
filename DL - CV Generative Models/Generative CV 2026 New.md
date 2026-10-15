
Generative Vision Models

|技術|需要掌握的內容|
|---|---|
|Latent Diffusion Models|Pixel Space vs Latent Space、VAE|
|Diffusion Transformer（DiT）|以 Transformer 替代傳統 U-Net Denoiser|
|Flow Matching / Rectified Flow|學習從 Noise 到 Data 的連續轉換|
|MMDiT|Image / Text Token Fusion|
|Image Editing / Inpainting|Mask Conditioning、Image-to-Image|
|Video Diffusion / Generation|Temporal Consistency、Video Latents|
|Controllable Generation|ControlNet、Depth/Edge/Pose Conditioning|
|Generative Data Augmentation|Synthetic Defects、Domain Randomization|
|Distillation / Few-step Generation|減少 Sampling Steps 與生成成本|

2024 年的 Stable Diffusion 3 已經結合 Multimodal Diffusion Transformer 與 Rectified Flow，後續的影像生成系統也持續沿著 Transformer-based Generative Modeling 發展。

![](https://www.google.com/s2/favicons?domain=https://stability.ai&sz=32)

Stability AI

+1

這些是面試時需要理解的核心差異：

- GAN：Generator 與 Discriminator 對抗訓練。
    
- Diffusion：學習反轉逐步加噪的過程。
    
- Flow Matching：學習將簡單分布傳輸到資料分布的向量場。
    
- DiT：一種以 Transformer 為核心的生成模型架構，而不是和 Diffusion、Flow Matching 並列的訓練目標。
    

常見的深入問題是：為什麼 diffusion 生成通常比較慢？Flow Matching 如何改變採樣方式？DiT 的 scaling 和傳統 U-Net 有何不同？如何建立具有可控缺陷位置的 synthetic training images？

對精密量測工作，還有一個很重要的問題：

> Can you use generative super-resolution to recover details for defect inspection?

應回答：生成模型可能補出 plausible 但未被真實感測器觀察到的高頻細節，因此不能把生成結果當作無條件可信的物理 Ground Truth。必須以真實量測、相機校正和獨立驗證評估 hallucinated detail 的風險。