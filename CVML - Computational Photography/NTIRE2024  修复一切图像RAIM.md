

NTIRE2024 | 修复一切图像RAIM: Restore All Image Model Challenge报告分析 - 老李的文章 - 知乎
https://zhuanlan.zhihu.com/p/701048076

這篇論文主要回顧了 **AIM 2020** 學習型 **ISP (Image Signal Processing) 挑戰賽**，描述了參賽隊伍所提出的解決方案及其結果。挑戰的目標是從 **Huawei P20** 手機的 **RAW** 影像轉換為與 **Canon 5D Mark IV** DSLR 相機拍攝的影像相匹配的高質量 **RGB** 圖像。這涉及一系列影像處理的子任務，如 **去馬賽克（Demosaicing）、降噪（Denoising）、白平衡（White Balancing）、色彩與對比度校正、去摩爾紋（Demoireing）等**。

1. 数据集安排

**劣化特征：**主要是虚焦模糊、少量运动模糊，没有看到降噪需求。数据对能看到镜头的对焦切换：HR是绝大部分像素正确对焦的高清图像、LR则是镜头焦点往后退一截后的虚焦图像。同时，也**留意到长焦特写的镜头素质有不同，虚焦后图像出现不同的散斑特征，这种劣化是主流图像修复赛道中鲜被提及**。也有一两对图像，表现出手工degradation处理的特点。


2. IQA（Image Quality Assessment）

|       |                                                                                                                                                                       |
| ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PSNR  | **Peak Signal-to-Noise Ratio，峰值信噪比。**PSNR数值越高，代表图像质量越好**PSNR 越高，表示移除的雜訊越多。不僅去除雜訊而且去除部分紋理的演算法將獲得良好的分數。                                                                 |
| SSIM  | **SSIM：结构相似性评价，Structural Similarity。**SSIM是衡量两幅图像相似度的指标，其取值范围为[0,1]，SSIM的值越大，表示图像失真程度越小，说明图像质量越好考慮了去雜訊影像和理想影像之間的邊緣（高頻內容）的相似性。為了獲得良好的 SSIM 測量結果，演算法需要消除噪音，同時保留物體的邊緣。。 |
| LPIPS |                                                                                                                                                                       |
| DISTS |                                                                                                                                                                       |
| NIQE  |                                                                                                                                                                       |

![[Pasted image 20250219111452.png]]



- **纹理和细节：**修复图像需要有良好和自然的纹理和细节。
- **噪声：**不能保留噪声、尤其是对色彩噪声，但是一些亮度噪声可以保留以避免平坦区域过分平滑、看起来涂抹。
- **伪纹：**蠕虫、色块、粘连、黑白边等伪纹需要尽量抑制。
- **保真度：**修复结果需要忠实于输入（作者指出，保真度评价的细则针对具体案例在赛程中与参赛者都同步到）

1. 競賽結果

|            |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| MiAlgo     | backbone= MW-UNet (Multi-level wavelet ISP network), 主要採用wavelet跟通道注意力機制(Residual channel attention blobk)提升圖像處理能力. 借助transformer强大的自注意力机制来定位整图感受野中的自相似特征. 並訓練兩個GAN model来退化(high->low)以建立更多的training dataset pair                                                                                                                                                                                                                                                                                             |
| Xhs-IAG    | backbone= SrFormer, 先用LSDIR的数据集进行训练，然后只用赛事官方提供个paired数据精调，结果在分数上达到了rank1的水平，在IR任務用SUPIR为了平衡细节效果和fidelity，又用DeSRA来辅助分类，把SUPIR放飞自我的图像区域替替换到弱生成的内容，最后用一个SRformer的backbone训练一个fusion模块融合两者结果。<br>(DeSRA的思路挺好，但其模型不过精调直接使用，笔者估计并不能在SUPIR的结果中实现很好的判定。从DeSRA论文给出的素材中，模型旨在判定GAN产生的伪纹理，对于diffusion能够产生语义不同但纹理自然的结果、则应该是无能为力的。)                                                                                                                                                                                          |
| So Elegant | Stage 1：DiffIR. 使用DiffIR去除LQ中多样化的劣化，以此获取到基础修复的图像输出. Stage 2：Stable Diffusion. 在纹理和细节的增强上，SD无疑是很优秀的。但是fidelity问题需要改善，采用Stage 1输出作为保真引导、约束diffusion的结果不要放飞自我。Stage 3：结构矫正. 这一步先利用DeSRA对图像的判别出一个mask，然后应用到前序两个stage的输出上，送入模型“矫正”。模型内部使用了可变卷积. 数据主要采用BSRGAN建立training set pair, 又在BSRGAN已有的退化基础上补充了离焦模糊                                                                                                                                                                                                            |
| IIP IR     | 采用了FTformer（Efficient frequency domain-based transformers for high-quality image deblurring）应该是分析过数据特征找到主要的劣化来自于blur，并且这种blur并不是单纯的虚焦模糊，而是有很多光学成像素质引入的模糊，因此特别需要不设限的模糊核估计. 团队放开了以往去模糊/超分工作中核权重为正且和为1的先验限制，以使得模型能够学习到更丰富的劣化表征. 而为了尽可能保留在修复过程中劣化表征的信息，团队利用SPADE结构把表征注入到每一层Unet的bridge上。stage1的处理相对只是把图像的模糊劣化给解决了，但图像缺失而有显著改善人眼主观感受的细节纹理仍有缺失。因此团队把图像经过HAT（Activating More Pixels in Image Super-Resolution Transformer）进行2倍超分，再送入StableSR@SD-Turbo. 这样做是为了让模型利用预训练的扩散模型时，能够更好的解决原本很细小的字符纹理。最后再用LANCZOS插值把图像下采样后输出。 |
| DACLIP-IR  | IR-SDE，采用stochastic differential equation（SDE）描述图像的劣化和修复过程. 对于赛事并未提供训练数据集的现实，作者先基于Real-ESRGAN的数据劣化策略，**修改了一个随机劣化顺序的流程**。然后借助自己以前的工作**DA-CLIP来提升所劣化出来的LQ图像的质量，具体的方法是最小化LQ和HQ样本embeddings之间的L1距离**。                                                                                                                                                                                                                                                                                                              |
|            | 剩余4支队伍：主要都采用了扩散模型的DiffSR和StableSR, 本次参赛队伍的提交方案中普遍采用了感知loss来提升图像细节表现                                                                                                                                                                                                                                                                                                                                                                                                                                            |
|            | 大多數提交的模型都具有多尺度編碼器-解碼器架構，並在多個尺度上處理圖像。許多參賽者使用了通道注意力 RCAN 模組、各種殘差連接以及離散小波轉換層。大多數團隊使用 MSE、L1、SSIM、基於 VGG 和基於顏色的損失函數。幾乎所有參賽者都使用 Adam 優化器和 PyTorch 框架來訓練深度學習模型。                                                                                                                                                                                                                                                                                                                                                      |
|            |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |


Reference:
一个令人惊艳的图片高清化重绘神器：SUPIR来了！ - 萤火架构的文章 - 知乎
https://zhuanlan.zhihu.com/p/692541284

图像复原的天花板在哪里？SUPIR：开创性结合文本引导先验和模型规模扩大 - AI生成未来的文章 - 知乎 (SUPIR: Scaling-UP Image Restoration)
https://zhuanlan.zhihu.com/p/680460658

ICML 2023｜两个步骤让瑕疵消失！解决GAN-SR 的伪影问题，只需配上一个DeSRA - 极市平台的文章 - 知乎(DeSRA: Detect and Delete the Artifacts of GAN-based Real-World Super-Resolution Models)
https://zhuanlan.zhihu.com/p/649117894

DiffIR：用于图像恢复的高效Diffusion模型 - Jorne的文章 - 知乎
https://zhuanlan.zhihu.com/p/708013761