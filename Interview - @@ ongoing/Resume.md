
**Tzu-Ching Wu** **(George)** Email address: a3146654@gmail.com

Research website: [http://tzuching1.weebly.com/](http://tzuching1.weebly.com/)

**Advanced Algorithm Development, Image Processing Engineer at Sartorius**

  
**Professional overview**

- 7 imaging software development, 9 journal publications and one selected as journal cover article, 9 conference publications in medical Image processing, Computer vision, Artificial Neural network, and <mark style="background: #FFF3A3A6;">Multi-objective optimization fields</mark>.
    
- 6+ years of expertise in implementing and crafting AI/CV algorithms, specializing in AI-based image segmentation, <mark style="background: #ADCCFFA6;">classification and tracking</mark>,<mark style="background: #ADCCFFA6;"> large Vision Foundation models(DINOv2, SAM)</mark> on commercial microscopy applications, and <mark style="background: #ADCCFFA6;">AI model compression, quantization technique, and neural network optimization</mark>.
    
- 4+ years of Industrial software development experience working on microscopy <mark style="background: #BBFABBA6;">ISP algorithm and calibration</mark>, <mark style="background: #BBFABBA6;">sub-pixel image registration</mark>, <mark style="background: #BBFABBA6;">feature recognition</mark>, <mark style="background: #BBFABBA6;">image enhancement</mark>, <mark style="background: #ADCCFFA6;">AI based semantic and panoptic segmentation</mark>, <mark style="background: #ADCCFFA6;">object detection</mark> and tracking, vision foundation model, and AI model compression and <mark style="background: #ADCCFFA6;">deployment with C++</mark>.
    
- Developed novel <mark style="background: #BBFABBA6;">wavelet-based imaging segmentation</mark> algorithm and <mark style="background: #BBFABBA6;">3D cell tracking method</mark> and collaborated with four labs to develop imaging software to analysis fluorescence microscopy, two-photon calcium imaging, smFISH expression.
    
**Research expertise**

**Programming:** Python(7 years), C++(4 years), MATLAB(14 years), PyTorch, OpenCV, Git, Docker, AzureML, Linux

**Computational Modeling**: Medical Image processing, Deep learning, Computer Vision, Artificial Neural Network, Heuristic Algorithm, Fuzzy logic, Multi-objective optimization, Multivariate statistical analysis

**Experience**

**Advanced Algorithm Development, Image Process Engineer Sep 2019 to Current**

**Sartorius Stedim North America – Ann Arbor, MI**

- Maintain and optimize reliable production-quality C++ code for Live Cell Analysis Systems IncuCyte software.
    
- Develop and train the AI segmentation module in PyTorch and deploy the ONNX model to C++ code. Also working on vision foundation models(DINOv2, SAM), and AI model compression and neural network optimization.
    
- Develop and implement microscopy ISP algorithm and calibration, sub-pixel image registration, feature recognition, image enhancement, tracking and segmentation algorithms for cell microscopy image using C++.
    
- Built 2D cell tracking algorithm platform and AI-based tracking prototype model.
    
- Design and develop prototype models and tools using Python and MATLAB to assist algorithm development and algorithm testing analyze, data quality control and test design.
    

**Ph.D. Research Assistant Sep 2012 to Sep 2019**

**Umulis lab, Purdue University - West lafayette, IN**

- Developed novel wavelet based image segmentation algorithm and built automatically analysis platform: **WaveletSEG**
    
- Established comprehensive imaging analysis platform including nuclei, mRNA, membrane segmentation: **ZebEmbIM**
    
- Designed pavement cell morphogenesis analysis method and software and selected as journal cover: **Lobefinder**
    
- Built Cell-by-Cell Relative Integrated Transcript (**CCRIT**) software to identify cells clusters and mRNA expression level.
    
- Developed neuronal calcium fluorescence spatio-temporal analysis software: **Neuronal CalciumIM**
    
- Designed the first cytoneme-mediated morphogen transport mathematics model and simulation tool: **CytonemeSim**
    
- Built deep learning object detection and semantic segmentation prototype model for chest X rays lung cancer screening.

----------------------------------------------------------------------
Image processing


Feature recognition

1. SIFT（尺度不變特徵變換）
SIFT是一種局部特徵檢測算法，能在尺度、旋轉和光照變化下保持不變。SIFT能夠提取圖像中的關鍵點並生成描述符，用於匹配和分類。

2. SURF（加速穩健特徵）
SURF 是 SIFT 的加速版本，使用了Hessian矩陣進行關鍵點檢測。它比 SIFT 更快，但對尺度和旋轉的不變性稍有減弱。

3. HOG（Histogram of Oriented Gradients）
HOG 將圖像分割成小塊，並對每個塊中的梯度方向進行統計，生成梯度方向直方圖，用於特徵描述。HOG在目標檢測，特別是人體檢測中非常流行。

4. Harris 角點檢測
Harris 角點檢測是一種基於圖像梯度變化的角點檢測算法，能夠識別具有強大梯度變化的區域，這些區域通常是角點。

5. FAST（Features from Accelerated Segment Test）
FAST 是一種高效的角點檢測算法，使用一個像素環來快速檢測圖像中的角點，尤其適合時間要求高的應用。

6. Gabor Filter
Gabor過濾器是一種線性濾波器，能夠對圖像中的特定頻率和方向進行響應，常用於紋理分析和特徵提取。

7. 光流（Optical Flow）
光流是一種跟蹤技術，通過估計圖像中像素的運動向量來檢測物體的運動方向和速度，常用於運動分析和視頻處理。

8. 特徵金字塔（Feature Pyramid Networks, FPN）
FPN 是一種多層次特徵提取技術，通過構建一個特徵金字塔結構來同時處理不同尺度的物體，常用於目標檢測和分割。


Image enhancement

1. 直方圖均衡化(Histogram Equalization)
將圖像的灰度值重新分佈，使其直方圖更加均勻，從而增強圖像的對比度，適用於提升暗淡圖像的細節。

2. 對比度拉伸（Contrast Stretching）
通過拉伸圖像中的亮度範圍來增強對比度，使亮部更亮、暗部更暗，從而提高圖像的清晰度。

3. 圖像平滑處理（Smoothing）
通過使用低通濾波器（如均值濾波器或高斯濾波器）來去除圖像中的噪聲，增強圖像的平滑度。

4. 圖像銳化（Sharpening）
使用高通濾波器來增強圖像的邊緣，使圖像更加清晰。常見的方法包括拉普拉斯濾波器和非零交叉檢測等。

5. 對數變換（Log Transformation）
對圖像的像素值進行對數變換，能夠有效增強低強度區域的細節，常用於處理暗圖像。

6. 幂律變換（Gamma Correction）
通過調整圖像的伽馬值來控制亮度分佈，適合修正圖像的過亮或過暗部分，常用於顯示器調校。

7. 雙邊濾波器（Bilateral Filter）
雙邊濾波器可以在保護邊緣的同時平滑圖像，減少噪聲的同時保持邊界清晰度，適用於降噪和紋理保護。





Image registration



ISP algorithm and calibration



Object detection



Image classification



Image segmentation



2D/3D Tracking


AI model compression


AI model deployment


Vision Foundation model(DINOv2, SAM)


Fourier Transform and Wavelet


Multiobjective optimization 


Tools: Pytorch, OpenCV, Git,  Docker,  AzureML,  Linux,  ONNX
