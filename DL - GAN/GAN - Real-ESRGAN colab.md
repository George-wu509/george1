[Real-ESRGAN github](https://github.com/xinntao/Real-ESRGAN/blob/master/README_CN.md)

[Real-ESRGAN Inference Demo.ipynb](https://colab.research.google.com/drive/1k2Zod6kSHEvraybHl50Lys0LerhyTMCo?usp=sharing)

Real-ESRGAN 的目标是开发出**实用的图像/视频修复算法**。  
我们在 ESRGAN 的基础上使用纯合成的数据来进行训练，以使其能被应用于实际的图片修复的场景（顾名思义：Real-ESRGAN）。我们提供了一套训练好的模型（_RealESRGAN_x4plus.pth_)，可以进行4倍的超分辨率。  **现在的 Real-ESRGAN 还是有几率失败的，因为现实生活的降质过程比较复杂。** 而且，本项目对**人脸以及文字之类**的效果还不是太好，但是我们会持续进行优化的。

我们提供了五种模型：
1. realesrgan-x4plus（默认）
2. reaesrnet-x4plus
3. realesrgan-x4plus-anime（针对动漫插画图像优化，有更小的体积）
4. realesr-animevideov3 (针对动漫视频)

你可以通过`-n`参数来使用其他模型，例如`./realesrgan-ncnn-vulkan.exe -i 二次元图片.jpg -o 二刺螈图片.png -n realesrgan-x4plus-anime`