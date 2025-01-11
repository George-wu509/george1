
## 1. Image Processing Toolbox  
[toolbox link](https://www.mathworks.com/products/image-processing.html)

| Image Analysis                                  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Basic                                           |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Geometric Transformation and Image Registration | <mark style="background: #BBFABBA6;">Common Geometric Transformations</mark>:<br>(2D/3D) Crop, Resize, Rotate, Translate<br><br><mark style="background: #BBFABBA6;">Generic Geometric Transformations</mark>:<br>(2D/3D) similarity, rigid, translation, projective, affine geometric transformation<br><br><mark style="background: #BBFABBA6;">Image Registration</mark>:<br>1. interactive registration with the Registration Estimator app, <br>2. intensity-based automatic image registration, <br>3. control point registration,<br>4. automated feature matching                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| Image Filtering and Enhancement                 | <mark style="background: #BBFABBA6;">Image Filtering</mark>:<br>Basic - <br>Gaussian, median, Filter region of interest, adaptive noise-removal, mode filtering, Local standard deviation, box filtering<br>Edge-Preserving Filtering - <br>Bilateral filtering, Anisotropic diffusion filtering<br>Texture Filtering - <br>Gabor filter, filter bank<br>Filtering By Property Characteristics, Integral Image Domain Filtering, Design Image Filters<br><br><mark style="background: #BBFABBA6;">Contrast Adjustment</mark>:<br>Adjust image intensity, Sharpen image using unsharp masking, lat-field correction, Brighten low-light image, Reduce atmospheric haze, Fast local Laplacian filtering, Render HDR image, histogram equalization, CLAHE, decorrelation stretch to multichannel image<br><br><mark style="background: #BBFABBA6;">ROI-Based Processing</mark>:<br>Line, point, circular, polygonal, elliptical, region of interest<br><br><mark style="background: #BBFABBA6;">Morphological Operations</mark>:<br>Erode, Dilate, open, close, top-hat, bottom-hat<br><br><mark style="background: #BBFABBA6;">Deblurring</mark>:<br>blind deconvolution, Lucy-Richardson method, Wiener filter<br><br><mark style="background: #BBFABBA6;">Neighborhood and Block Processing</mark>:<br>block processing, sliding-neighborhood operations<br><br><mark style="background: #BBFABBA6;">Image Arithmetic</mark>:<br>add, Subtract, multiply, divide, diff, Complement image                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| Image Segmentation and Analysis                 | <mark style="background: #BBFABBA6;">Image Segmentation</mark>:<br>SAM, threshold(Otsu, adaptive), Watershed, graph cut, K-means, superpixel oversegmentation<br><br><mark style="background: #BBFABBA6;">Object Analysis</mark>:<br>Display Boundaries - Trace object boundaries<br>Detect Circles - circular Hough transform<br>Detect Edges and Gradients - directional gradients, Find edges<br>Detect Lines - Hough transform<br>Detect Homogenous Blocks - Quadtree decomposition<br><br><mark style="background: #BBFABBA6;">Region and Image Properties</mark>:<br>Measure Properties of Image Regions - Measure properties, convex hull, Euler number, Feret properties<br>Measure Properties of Pixels - Pixel-value cross-sections<br>Measure Properties of Images - Distance transform, Histogram<br>Find, Select, and Label Objects in Binary Images - Find and count connected components, 	Extract objects from binary image<br><br><mark style="background: #BBFABBA6;">Texture Analysis</mark>:<br>Entropy of grayscale image, gray-level co-occurrence matrix, GLCM<br><br><mark style="background: #BBFABBA6;">Image Quality</mark>:<br>Full Reference Quality Metrics - Mean-squared error, PSNR, SSIM, <br>No-Reference Quality Metrics - Image Spatial Quality Evaluator (BRISQUE), Image Quality Evaluator (NIQE)<br>Test Chart Based Quality Measurements - Imatest edge spatial frequency response (eSFR)<br><br><mark style="background: #BBFABBA6;">Image Transforms</mark>:<br>Hough transform, 2-D (inverse)discrete cosine transform, Radon transform, fast (inverse) Fourier transform                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Deep Learning for Image Processing              | <mark style="background: #BBFABBA6;">Create Datastores for Image Preprocessing</mark>:<br>Datastore for image data, Denoising image datastore, Transform datastore, Transform batches to augment image data<br><br><mark style="background: #BBFABBA6;">Augment Images</mark>:<br>affine 2D/3D transformation,  randomized cuboidal cropping<br><br><mark style="background: #BBFABBA6;">Resize and Reshape Deep Learning Input</mark>:<br>2D/3D resize layer, Depth to space layer<br><br><mark style="background: #BBFABBA6;">Create Deep Learning Networks</mark>:<br>encoder-decoder, CycleGAN, PatchGAN, pix2pixHD, UNet<br><br><mark style="background: #BBFABBA6;">Denoise Images</mark>:<br>Denoise image using deep neural network                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| 3-D Volumetric Image Processing                 | <mark style="background: #BBFABBA6;">Volume Display:</mark><br>Display volume, Browse image slices, Extract oblique slice from 3-D volumetric data, Light source, Point annotation, Line annotation<br>Image Import and Conversion:<br>Read DICOM image, Create 4-D volume from set of DICOM images, Extract ROI data from DICOM-RT structure set, Binarize 2-D grayscale image or 3-D volume by thresholding, Read metadata from NIfTI file<br><br><mark style="background: #BBFABBA6;">Image Arithmetic:</mark><br>add, subtract, multiply, divide<br><br><mark style="background: #BBFABBA6;">Geometric Transformations and Image Registration</mark>:<br>3-D affine geometric transformation, 	Intensity-based image registration<br><br><mark style="background: #BBFABBA6;">Image Filtering and Enhancement</mark>:<br>3-D Gaussian filtering, Enhance contrast using histogram equalization<br><br><mark style="background: #BBFABBA6;">Morphology</mark>:<br>Erode, Dilate, open, close, top-hat, bottom-hat<br><br><mark style="background: #BBFABBA6;">Image Segmentation</mark>:<br>3-D superpixel oversegmentation, Jaccard similarity coefficient , K-means<br><br><mark style="background: #BBFABBA6;">Image Analysis</mark>:<br>properties of 3-D volumetric image regions, Find edges, Find gradient magnitude, Histogram<br><br><mark style="background: #BBFABBA6;">Image Augmentation for Deep Learning</mark>:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Hyperspectral Image Processing                  | <mark style="background: #BBFABBA6;">Explore, Analyze, and Visualize</mark>:<br>Read hyperspectral data, to ENVI file format, Read metadata from ENVI header file, Select most informative bands, Estimate color image of hyperspectral data<br><br><mark style="background: #BBFABBA6;">Filtering and Enhancement</mark>:<br>Denoise hyperspectral images using non-local meets global approach, Sharpen hyperspectral data using coupled nonnegative matrix factorization (CNMF) <br><br><mark style="background: #BBFABBA6;">Data Correction</mark>:<br>Radiometric Calibration - Convert digital number to radiance, Convert digital number to reflectance, Convert radiance to reflectance<br>Atmospheric Correction - Empirical line calibration of hyperspectral data, Apply flat field correction to hyperspectral data, Subtract dark pixel value from hyperspectral data cube, Perform atmospheric correction using satellite hypercube atmospheric rapid correction (SHARC)<br>Spectral Correction - Compute spectral smile metrics of hyperspectral data, Reduce spectral smile effect in hyperspectral data<br><br><mark style="background: #BBFABBA6;">Dimensionality Reduction</mark>:<br>PCA, Maximum noise fraction transform, Reconstruct data from PCA<br><br><mark style="background: #BBFABBA6;">Spectral Unmixing</mark>:<br>Extract endmember signatures using pixel purity index, Extract endmember signatures using fast iterative pixel purity index, Extract endmember signatures using N-FINDR<br><br><mark style="background: #BBFABBA6;">Spectral Matching and Target Detection</mark>:<br>Read data from ECOSTRESS spectral library, Resample spectral signature to required wavelengths, Measure spectral similarity, 	Measure normalized spectral similarity score, Detect target in hyperspectral image, 	Compute hyperspectral indices<br><br><mark style="background: #BBFABBA6;">Segmentation</mark>:<br>2-D superpixel oversegmentation of hyperspectral images, Segment hyperspectral images using fast spectral clustering with anchor graphs |
| Code Generation and GPU Support                 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |

-------------------------------------------------------------

## 2. Computer Vision Toolbox
[toolbox link](https://www.mathworks.com/products/computer-vision.html)

| Basic                                                                                                                             |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Feature Detection and Extraction](https://www.mathworks.com/help/vision/feature-detection-and-extraction.html?s_tid=CRUX_lftnav) | <mark style="background: #BBFABBA6;">Detect Features</mark>:<br>SIFT, SURF, ORB<br>detect corner- Harris, FAST  <br><br><mark style="background: #BBFABBA6;">Extract Features</mark>:<br>HOG, LBP<br><br><mark style="background: #BBFABBA6;">Match Features</mark>:<br>Find matching features<br><br><mark style="background: #BBFABBA6;">Image Registration</mark>:<br>Estimate 2D/3D geometric transformation from matching point pairs, Apply geometric transformation, Locate template in image, Blend two images<br><br><mark style="background: #BBFABBA6;">Visualization and Display</mark>:<br>Insert markers/shapes/text/keypoint in image or video, Downsample or upsample chrominance components of images<br><br><mark style="background: #BBFABBA6;">Store Features</mark>:<br>object for SIFT/SURF/ORB/BRISK/ interest points<br><br><mark style="background: #BBFABBA6;">Transform Objects</mark>:<br>2D/3D rigid/similarity/affine/projective geometric transformation<br><br><mark style="background: #BBFABBA6;">Retrieve Images</mark>:<br>Create Recognition Database -Bag of visual words object,Search index that maps visual words to images<br>Retrieve Images - Search image set for similar image,Evaluate image search results                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Image and Video Ground Truth Labeling                                                                                             | <mark style="background: #BBFABBA6;">Label Images and Video</mark>:<br>Create Definitions for Labels, add , edit and remove labels<br><br><mark style="background: #BBFABBA6;">Automate Labeling</mark>:<br>Action Flags, Validation, Algorithm<br><br><mark style="background: #BBFABBA6;">Create Team-Based Image Labeling Project</mark>:<br>Image Labeler, Video Labeler<br><br><mark style="background: #BBFABBA6;">Work with Ground Truth Data</mark>:<br>Select Labels, Store and Post-process Labels, Create Training Data for Object Detectors, Enumerate Attribute and Label Types<br><br><mark style="background: #BBFABBA6;">Ground Truth Data Applications</mark>:<br>Training Data for Object Detection and Semantic Segmentation, SOLOv2 for Instance Segmentation, Image Preprocessing and Augmentation for Deep Learning                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Object Detection                                                                                                                  | <mark style="background: #BBFABBA6;">Detect Objects</mark>:<br>Deep Learning Detectors - YOLO v2,3,4 YOLOX,SSD<br>Feature-based Detectors - Detect objects/people using aggregate channel features,Foreground detection using Gaussian mixture models<br>Detect Objects Using Point Features - SIFT, SURF, ORB and detect corner- Harris, FAST  <br>Select Detected Objects - <br>Select strongest bounding boxes from overlapping clusters using nonmaximal suppression (NMS)<br><br><mark style="background: #BBFABBA6;">Train Custom Object Detectors</mark>:<br>Load Training Data - Datastore for bounding box label data<br>Train Feature-Based Object Detectors - Train cascade object detector model<br>Train Deep Learning Based Object Detectors - train YOLO v2,3,4 YOLOX,SSD<br>Augment and Preprocess Training Data for Deep Learning - <br>affine 2D/3D transformation,  randomized cuboidal cropping<br><br><mark style="background: #BBFABBA6;">Design Object Detection Deep Neural Networks</mark>:<br>Mask-RCNN, YOLO, SSD, Estimate anchor boxes, focal cross-entropy loss <br><br><mark style="background: #BBFABBA6;">Visualize Detection Results</mark>:<br>Project cuboids from 3-D world coordinates to 2-D image coordinates, Annotate truecolor or grayscale image or video, Insert masks/shape in image or video<br><br><mark style="background: #BBFABBA6;">Evaluate Predicted Results:</mark><br>Mean average precision (mAP) metric, bounding box overlap ratio, precision and recall, Object detection quality metrics                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Semantic Segmentation                                                                                                             | <mark style="background: #BBFABBA6;">Augment and Preprocess Training Data</mark>:<br>affine 2D/3D transformation,  randomized cuboidal cropping<br><br><mark style="background: #BBFABBA6;">Design Semantic Segmentation Deep Learning Networks</mark>:<br>2D/3D U-Net, DeepLab v3, focal cross-entropy loss<br><br><mark style="background: #BBFABBA6;">Segment Images Using Deep Learning</mark>:<br>Segment Anything Model (SAM)<br><br><mark style="background: #BBFABBA6;">Evaluate Segmentation Results</mark>:<br>Contour matching score, Jaccard similarity coefficient, Sørensen-Dice similarity coefficient, Semantic segmentation quality metrics<br>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Instance Segmentation                                                                                                             | <mark style="background: #BBFABBA6;">Segment Objects in Images</mark>:<br>SOLOv2, MaskRCNN <br><br><mark style="background: #BBFABBA6;">Train Custom Instance Segmentation Networks</mark>:<br>Ground truth label data, Datastore for image data, train SOLOv2, MaskRCNN,  Convert region of interest (ROI) polygon to region mask, Trace object boundaries in binary image, affine 2D/3D transformation,  randomized cuboidal cropping<br><br><mark style="background: #BBFABBA6;">Evaluate Predicted Results</mark>:<br>Instance segmentation quality metrics, Evaluate instance segmentation across object mask size ranges<br><br><mark style="background: #BBFABBA6;">Perform Pose Estimation Using Instance Segmentation</mark>:<br>train/predict/estimate object pose using Pose Mask R-CNN pose estimation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Image Category Classification                                                                                                     | <mark style="background: #BBFABBA6;">ViT:</mark><br>Pretrained vision transformer (ViT) neural network, Patch embedding layer<br><br><mark style="background: #BBFABBA6;">Bag of features</mark>:<br>Train an image category classifier, Bag of visual words object, Predict image category                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Automated Visual Inspection                                                                                                       | <mark style="background: #BBFABBA6;">Load Training Data</mark>:<br>Create training data for scene classification from ground truth<br><br><mark style="background: #BBFABBA6;">Train Anomaly Detector</mark>:<br>EfficientAD, fully convolutional data description(FCDD), FastFlow, PatchCore anomaly detection network, Optimal anomaly threshold for set of anomaly scores and corresponding labels<br><br><mark style="background: #BBFABBA6;">Detect Anomalies Using Deep Learning</mark>:<br>EfficientAD, fully convolutional data description(FCDD), FastFlow, PatchCore anomaly detection network<br><br><mark style="background: #BBFABBA6;">Detect and Classify Objects</mark>:<br>YOLOX object detector<br><br><mark style="background: #BBFABBA6;">Visualize and Evaluate Results</mark>:<br>Predict per-pixel anomaly score map, Overlay heatmap on image using per-pixel anomaly scores, Anomaly detection metrics<br>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| Text Detection and Recognition                                                                                                    | <mark style="background: #BBFABBA6;">Text Recognition</mark>:<br>Recognize text using optical character recognition, Store OCR results<br><br><mark style="background: #BBFABBA6;">Training and Evaluation</mark>:<br>Train OCR model to recognize text in image, Evaluate OCR results against ground truth <br><br><mark style="background: #BBFABBA6;">Quantization</mark>:<br>Quantize OCR model<br><br><mark style="background: #BBFABBA6;">Text Detection</mark>:<br>Detect texts in images by using CRAFT deep learning model, Detect MSER features, Extract histogram of oriented gradients (HOG) features                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Keypoint Detection                                                                                                                | <mark style="background: #BBFABBA6;">Detect Object Keypoints</mark>:<br>Create object keypoint detector using HRNet deep learning network<br><br><mark style="background: #BBFABBA6;">Train Object Keypoint Detector</mark>:<br>Train HRNet object keypoint detector<br><br><mark style="background: #BBFABBA6;">Visualize Detection Results</mark>:<br>Insert object keypoints in image <br><br><mark style="background: #BBFABBA6;">Load Detector for Code Generation</mark>:<br>Load HRNet object keypoint detector model for code generation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Video Classification                                                                                                              | <mark style="background: #BBFABBA6;">Extract Video Training Data</mark>:<br>Write video sequence to video file, Time ranges of scene labels from ground truth data <br><br><mark style="background: #BBFABBA6;">Load Video Training Data</mark>:<br>Create object to read video files, Datastore with custom file reader, Combine data from multiple datastores<br><br><mark style="background: #BBFABBA6;">Design Video Classifier</mark><br>Inflated-3D (I3D) video classifier, SlowFast video classifier, R(2+1)D video classifier<br><br><mark style="background: #BBFABBA6;">Train Video Classifier</mark><br>Compute video classifier predictions, Compute video classifier outputs for training<br><br><mark style="background: #BBFABBA6;">Augment and Preprocess Training Data</mark><br>Apply geometric transformation, Crop/Resize image, 2-D affine transformation<br><br><mark style="background: #BBFABBA6;">Classify Video</mark><br>Classify a video file, Classify video sequence, Update video sequence for classification, Reset video sequence properties for streaming video classification<br><br><mark style="background: #BBFABBA6;">Visualize Classification Results</mark>:<br>Play video or display image, Display video, Insert text in image or video                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Camera Calibration                                                                                                                | <mark style="background: #BBFABBA6;">Generate and Detect Calibration Patterns</mark>:<br>Detect checkerboard pattern, Detect keypoints of AprilGrid pattern, Detect and estimate pose for AprilTag, Detect circle grid pattern, Detect and estimate pose for ArUco marker, Detect ChArUco board pattern in images<br><br><mark style="background: #BBFABBA6;">Estimate Camera Parameters</mark>:<br>Pinhole Camera - Estimate camera projection matrix from world-to-image point correspondences, alibrate a single or stereo camera<br>Fisheye Camera - Calibrate fisheye camera<br>Stereo Camera - Estimate baseline of stereo camera<br><br><mark style="background: #BBFABBA6;">Store Results</mark>:<br>Intrinsic camera parameters based on Kannala-Brandt model, Camera projection matrix, Object for storing camera parameters, intrinsic camera parameters<br><br><mark style="background: #BBFABBA6;">Remove Distortion</mark>:<br>Pinhole Camera - Correct image for lens distortion, Correct point coordinates for lens distortion<br>Fisheye Camera - Correct point coordinates for fisheye lens distortion, Correct fisheye image for lens distortion<br><br><mark style="background: #BBFABBA6;">Visualize Results</mark>:<br>Plot 3-D point cloud, Visualize extrinsic camera parameters, 	Create red-cyan anaglyph from stereo pair of images<br><br><mark style="background: #BBFABBA6;">Estimate Camera Pose</mark>:<br>Calculate location of calibrated camera, Convert extrinsics to camera pose, Convert camera pose to extrinsics, Calculate relative rotation and translation between camera poses<br><br><mark style="background: #BBFABBA6;">Conversions</mark>:<br>Convert 3-D rotation matrix to rotation vector, Convert 3-D rotation vector to rotation matrix, Convert camera intrinsic parameters and stereo camera parameters from MATLAB to OpenCV, Convert camera intrinsic parameters and stereo camera parameters from OpenCV to MATLAB                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| Structure from Motion and Visual SLAM(Stereo Vision)                                                                              | <mark style="background: #BBFABBA6;">3-D Reconstruction</mark>:<br>3-D locations of undistorted matching points in stereo images, Compute epipolar lines for stereo images, Correct image for lens distortion, Correct point coordinates for lens distortion, Compute disparity map using block matching, Uncalibrated stereo rectification, Reconstruct 3-D scene from disparity map, Rectify pair of stereo images,Object for storing stereo camera system parameters<br><br><mark style="background: #BBFABBA6;">Visualize Results</mark>:<br>Create red-cyan anaglyph from stereo pair of images, Plot 3-D point cloud, Plot a camera in 3-D coordinates<br><br><mark style="background: #BBFABBA6;">Conversions</mark>:<br>Convert 3-D rotation matrix to rotation vector, Convert 3-D rotation vector to rotation matrix                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| Structure from Motion and Visual SLAM(Structure from Motion)                                                                      | <mark style="background: #BBFABBA6;">Detect Features</mark>:<br>SIFT, SURF, MSER, BRISK. Detect corners using Harris, FAST, BRISK, minimum eigenvalue algorithm<br><br><mark style="background: #BBFABBA6;">Match Features</mark>:<br>Extract interest point descriptors, Find matching features, Find matching features within specified radius, Track points in video using Kanade-Lucas-Tomasi (KLT) algorithm<br><br><mark style="background: #BBFABBA6;">Estimate 3-D Structure</mark>:<br>Store Image and Camera Data - Manage data for structure-from-motion, visual odometry, and visual SLAM, Manage 3-D to 2-D point correspondences, Object for storing intrinsic camera parameters, 3-D rigid geometric transformation, 3-D affine geometric transformation<br>Estimate Camera Poses - Estimate essential matrix/ fundamental matrix/camera pose from corresponding points in a pair of images<br>Triangulate Image Points - Object for storing matching points from multiple views, Find matched points across multiple views, 3-D locations of undistorted matching points, 3-D locations of world points matched<br>Optimize Camera Poses and 3-D Points - Adjust collection of 3-D points and camera poses, Adjust collection of 3-D points and camera poses, Refine 3-D points using structure-only bund<br><br><mark style="background: #BBFABBA6;">Visualize Results</mark>:<br>Create red-cyan anaglyph from stereo pair of images, Plot 3-D point cloud, Plot a camera in 3-D coordinates, 	Display corresponding feature points<br><br><mark style="background: #BBFABBA6;">Conversions</mark>:<br>Convert 3-D rotation matrix to rotation vector, Convert 3-D rotation vector to rotation matrix, 	Create quaternion array                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Structure from Motion and Visual SLAM(vSLAM)                                                                                      | <mark style="background: #BBFABBA6;">Detect, Extract, and Match Features</mark>:<br>SIFT, SURF, ORB, Find matching features within specified radius <br><br><mark style="background: #BBFABBA6;">Reconstruct 3-D Structure</mark>:<br>3-D locations of undistorted matching points in stereo images, Determine world coordinates of image points, Project world points into image<br><br><mark style="background: #BBFABBA6;">Estimate Motion</mark>:<br>Estimate 2D/3D geometric transformation from matching point pairs, Estimate camera pose from 2D/3D point correspondences, Estimate fundamental matrix from corresponding points in stereo images<br><br><mark style="background: #BBFABBA6;">Optimize Motion and 3-D Structure</mark>:<br>Optimize absolute poses using relative pose constraints, Create pose graph, Adjust collection of 3-D points and camera poses<br><br><mark style="background: #BBFABBA6;">Evaluate Results</mark>:<br>Compare estimated trajectory against ground truth, Store accuracy metrics for trajectories<br><br><mark style="background: #BBFABBA6;">Visualize Results</mark>:<br>Plot a camera in 3-D coordinates, Plot 3-D point cloud, Visualize streaming 3-D point cloud data<br><br><mark style="background: #BBFABBA6;">Manage Data</mark>:<br>Bag of visual words object, Bag of visual words using DBoW2 library, Detect loop closure using visual features, 	Manage 3-D to 2-D point correspondences<br><br><mark style="background: #BBFABBA6;">Monocular Visual SLAM</mark>:<br>Visual simultaneous localization and mapping (vSLAM) with monocular camera, Add image frame to visual SLAM object, Check status of visual SLAM object, Build 3-D map of world points, Plot 3-D map points and estimated camera trajectory in visual SLAM<br><br><mark style="background: #BBFABBA6;">RGB-D Visual SLAM</mark>:<br>Feature-based visual simultaneous localization and mapping (vSLAM) with RGB-D camera, Add pair of color and depth images to RGB-D visual SLAM object, End-of-processing status for RGB-D visual SLAM object, Build 3-D map of world points from RGB-D vSLAM object<br><br><mark style="background: #BBFABBA6;">Stereo Visual SLAM</mark>:<br>Feature-based visual simultaneous localization and mapping (vSLAM) with stereo camera, End-of-processing status for stereo visual SLAM object, Build 3-D map of world points from stereo vSLAM object                                                                  |
| Point Cloud Processing                                                                                                            | <mark style="background: #BBFABBA6;">Read and Write Point Clouds</mark>:<br>Read 3-D point cloud from PLY or PCD file, Write 3-D point cloud to PLY or PCD file, Convert depth image to point cloud<br><br><mark style="background: #BBFABBA6;">Store Point Clouds</mark>:<br>Manage data for point cloud based visual odometry and SLAM, Object for storing 3-D point cloud<br><br><mark style="background: #BBFABBA6;">Visualize Point Clouds</mark>:<br>Visualize and inspect large 3-D point cloud, Plot 3-D point cloud, Visualize difference between two point clouds<br><br><mark style="background: #BBFABBA6;">Process Point Clouds</mark>:<br>Preprocess - Spatially bin point cloud points, 	Downsample a 3-D point cloud, Estimate normals for point cloud<br>Find and Remove Points - Find points within a cylindrical region in a point cloud, Find points within a region of interest in the point cloud, Find nearest neighbors of a point in point cloud, Find neighbors within a radius of a point in the point cloud<br><br><mark style="background: #BBFABBA6;">Segment Point Clouds</mark>:<br>Segment point cloud into clusters based on Euclidean distance, Segment ground points from organized lidar data, Segment organized 3-D range data into clusters<br><br><mark style="background: #BBFABBA6;">Register Point Clouds and Create Maps</mark>:<br>Register Point Clouds - Register two point clouds using phase correlation, Register two point clouds using ICP/CPD, NDT algorithm<br>Transform Point Clouds - 3-D rigid geometric transformation, Transform 3-D point cloud<br>Align or Combine Point Clouds - Align array of point clouds, Concatenate 3-D point cloud array, Merge two 3-D point clouds<br>Determine Loop Closure Candidates - Localize point cloud within map using normal distributions transform (NDT) algorithm, Distance between scan context descriptors, Extract scan context descriptor from point cloud, Detect loop closures using scan context descriptors<br>Optimize Poses - Create pose graph, Optimize absolute poses using relative pose constraints<br>Create Localization Map - Localization map based on normal distributions transform (NDT)<br><br><mark style="background: #BBFABBA6;">Fit Point Clouds to Geometric Models</mark>:<br>Fit cylinder/plane/sphere to 3-D point cloud, Fit polynomial to points using RANSAC, Object for storing parametric plane model, Object for storing a parametric sphere model |
| Tracking and Motion Estimation                                                                                                    | <mark style="background: #BBFABBA6;">Load, Save, and Display Video</mark>:<br>Read video data from binary files, Write video frames and audio samples to video file, Write binary video data to files<br><br><mark style="background: #BBFABBA6;">Object Tracking</mark>:<br>Histogram-based object tracking, Kalman filter, Kanade-Lucas-Tomasi (KLT) algorithm<br><br><mark style="background: #BBFABBA6;">Object Re-Identification</mark>:<br>Re-identification deep learning network for re-identifying and tracking objects, Re-identification (ReID) quality metrics<br><br><mark style="background: #BBFABBA6;">Motion Estimation</mark>:<br>Object for estimating optical flow using Lucas-Kanade method/RAFT deep learning algorithm<br><br><mark style="background: #BBFABBA6;">Visualization and Display</mark>:<br>Insert markers/shapes/text in image or video                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Code Generation, GPU, and Third-Party Support                                                                                     |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| Computer Vision with Simulink<br>                                                                                                 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |

## Important methods in interview

### 1. <mark style="background: #FF5582A6;">matlab 3D images</mark>

ref1: [3D Image Processing with MATLAB](https://www.mathworks.com/solutions/image-video-processing/3d-image-processing.html)

##### 1-1. **讀取和顯示 3D 醫學影像 (DICOM 格式)**

MATLAB 可以使用 `dicomread` 函數讀取 3D 醫學影像數據，例如 CT 或 MRI 圖像，並使用 `volshow` 來可視化它們。
% 讀取 DICOM 文件
volume = dicomread('CTscan.dcm');
% 顯示 3D 醫學影像
volshow(volume);

##### 1-2. **3D 醫學影像的表面重建**

從 3D 醫學影像中重建出表面模型，通常應用於骨骼或器官的表面可視化。
% 讀取 MRI 數據
mriData = load('mri.mat');
D = squeeze(mriData.D);
% 提取等值面，生成表面模型
fv = isosurface(D, 5);
% 顯示表面模型
patch(fv, 'FaceColor', 'red', 'EdgeColor', 'none');

##### 1-3. **體積渲染 (Volume Rendering)**

MATLAB 支援體積渲染技術，能夠渲染 3D 醫學影像的體積數據，適合於視覺化內部結構。
% 體積渲染
volshow(D);

##### 1-4. **3D 影像分割**

利用區域成長算法對 3D 醫學影像進行分割，例如從 CT 影像中分割肺部結構。
% 設定初始分割種子點
BW = activecontour(D, 'Chan-Vese');
% 顯示分割結果
volshow(BW);

##### 1-5. **3D 高斯濾波**

對 3D 影像數據進行平滑處理，使用高斯濾波器可以有效去除影像中的噪聲。
% 使用 3D 高斯濾波器
smoothedData = imgaussfilt3(D, 2);
% 顯示平滑後的影像
volshow(smoothedData);

##### 1-6. **3D 影像的邊緣檢測**

在 3D 影像中進行邊緣檢測，通常用於突出邊界或形狀。
% 使用 Sobel 邊緣檢測
edges = edge3(D, 'Sobel');
% 顯示邊緣
volshow(edges);

##### 1-7. **3D 影像的骨架提取**

從 3D 影像中提取骨架結構，適合用於分析脈絡、神經或血管網絡。
% 提取骨架結構
BW = imbinarize(D);
skeleton = bwskel(BW);
% 顯示骨架
volshow(skeleton);

##### 1-8. **3D 圖像的形態學運算**

應用形態學運算來處理 3D 影像，例如進行膨脹、侵蝕等操作。
% 3D 膨脹運算
se = strel('sphere', 2);
dilatedVolume = imdilate(BW, se);
% 顯示膨脹後的影像
volshow(dilatedVolume);

##### 1-9. **3D 影像配準**

通過配準將兩個或多個 3D 影像對齊，例如在不同時間點拍攝的影像對比。
% 讀取兩幅 3D 影像
fixed = load('fixedVolume.mat');
moving = load('movingVolume.mat');
% 進行影像配準
tform = imregtform(moving.volume, fixed.volume, 'rigid', optimizer, metric);
registeredVolume = imwarp(moving.volume, tform);
% 顯示配準後的影像
volshow(registeredVolume);

##### 1-10. **顯示 3D 影像切片**

MATLAB 可以顯示 3D 影像的不同切片，幫助用戶觀察各個切片的內容。
% 顯示 3D 影像的 XZ 平面切片
sliceX = D(:, :, round(end/2));
imagesc(sliceX);
colormap(gray);

##### 1-11. **點雲讀取與顯示**

MATLAB 可以讀取和顯示 3D 點雲數據，通常用於 LIDAR 或深度攝像機捕獲的 3D 資料。
% 讀取點雲數據
ptCloud = pcread('teapot.ply');
% 顯示點雲
pcshow(ptCloud);
xlabel('X'); ylabel('Y'); zlabel('Z');
title('3D 點雲顯示');

##### 1-12. **點雲對齊**

通過 ICP (Iterative Closest Point) 演算法，將兩組 3D 點雲對齊。
% 讀取兩個點雲
ptCloud1 = pcread('teapot1.ply');
ptCloud2 = pcread('teapot2.ply');
% 使用 ICP 對齊點雲
[tform, ptCloudAligned] = pcregistericp(ptCloud2, ptCloud1);
% 顯示對齊結果
pcshowpair(ptCloud1, ptCloudAligned, 'VerticalAxis','y','VerticalAxisDir','down');
title('點雲對齊');

##### 1-13. **3D 重建**

從多張 2D 圖像中進行 3D 重建，使用多視角立體視覺技術重建物體的 3D 表面。
% 載入相機參數
load('cameraParams.mat');
% 使用多視角進行 3D 重建
stereoParams = stereoParameters(cameraParams, cameraParams);
% 計算深度圖
[imagePoints1, imagePoints2] = detectSURFFeatures(img1, img2);
[reconstructedScene, disparityMap] = reconstructScene(stereoParams, imagePoints1, imagePoints2);
% 顯示重建的 3D 場景
imshow3D(reconstructedScene);

##### 1-14. **點雲裁剪**

對 3D 點雲進行裁剪，保留感興趣的區域。
% 讀取點雲
ptCloud = pcread('teapot.ply');
% 定義裁剪區域
roi = [0.5 0.8; 0.2 0.7; 0 1];
% 裁剪點雲
ptCloudCropped = crop(ptCloud, roi);
% 顯示裁剪後的點雲
pcshow(ptCloudCropped);
title('裁剪後的點雲');

##### 1-15. **點雲分割**

將點雲分割為多個部分，通常用於識別不同物體或區域。
% 讀取點雲
ptCloud = pcread('teapot.ply');
% 分割平面
maxDistance = 0.02;
[model, inlierIndices, outlierIndices] = pcfitplane(ptCloud, maxDistance);
% 顯示分割結果
pcshow(ptCloud);
hold on;
pcshow(select(ptCloud, inlierIndices), 'r');
title('點雲分割');
##### 1-16. **點雲法向量估計**

估計 3D 點雲的表面法向量，常用於進一步的形狀分析和表面重建。
% 讀取點雲
ptCloud = pcread('teapot.ply');
% 計算法向量
normals = pcnormals(ptCloud);
% 顯示法向量
pcshow(ptCloud);
hold on;
quiver3(ptCloud.Location(:,1), ptCloud.Location(:,2), ptCloud.Location(:,3), normals(:,1), normals(:,2), normals(:,3));
title('點雲法向量');

##### 1-17. **3D 場景的深度估計**

從立體相機獲取兩個視角的圖像來估計場景的深度。
% 讀取兩幅立體圖像
I1 = imread('leftImage.png');
I2 = imread('rightImage.png');
% 使用 stereoAnaglyph 查看視差
stereoParams = stereoParameters(cameraParams1, cameraParams2);
depthMap = disparityMap(I1, I2, stereoParams);
imshow(depthMap, [0, 64]);
title('深度估計');

##### 1-18. **3D 物體的姿態估計**

根據 3D 點雲數據估計物體的姿態（位置和方向）。
% 讀取點雲
ptCloud = pcread('teapot.ply');
% 使用 RANSAC 擬合平面
maxDistance = 0.02;
[model, inlierIndices] = pcfitplane(ptCloud, maxDistance);
% 計算物體姿態
pose = estimateWorldCameraPose(inlierIndices, model);
disp(pose);

##### 1-19. **3D 點雲的體素化**

將點雲數據分成體素，進行體積分析或形狀特徵提取。
% 讀取點雲
ptCloud = pcread('teapot.ply');
% 進行體素化
voxelSize = 0.1;
ptCloudVoxel = pcdownsample(ptCloud, 'gridAverage', voxelSize);
% 顯示體素化的點雲
pcshow(ptCloudVoxel);
title('體素化點雲');

#### 1-20. **相機姿態估計**

使用多張影像估計相機在場景中的姿態。
% 使用 SURF 特徵點進行匹配
imagePoints1 = detectSURFFeatures(img1);
imagePoints2 = detectSURFFeatures(img2);
[tform, inlierPoints1, inlierPoints2] = estimateGeometricTransform(imagePoints1, imagePoints2, 'affine');
% 計算相機姿態
camPose = estimateWorldCameraPose(inlierPoints1, inlierPoints2, cameraParams);
disp(camPose);



### <mark style="background: #FF5582A6;">2. matlab Video data</mark>

ref1: [Video processing with MATLAB](https://www.mathworks.com/solutions/image-video-processing/video-processing.html)

##### 2-1. **讀取影片並逐幀處理**

MATLAB 可以使用 `VideoReader` 來讀取影片，並逐幀處理。
% 讀取影片
v = VideoReader('sample.mp4');
% 顯示影片的每一幀
while hasFrame(v)
    frame = readFrame(v);
    imshow(frame);
    pause(1/v.FrameRate);  % 控制播放速度
end

##### 2-2. **影片寫入**

MATLAB 使用 `VideoWriter` 將處理後的幀重新寫入成影片。
% 創建影片寫入對象
writerObj = VideoWriter('output.avi');
open(writerObj);
% 逐幀寫入影片
for k = 1:100
    img = imread(sprintf('frame%d.jpg', k));
    writeVideo(writerObj, img);
end
close(writerObj);

##### 2-3. **背景減法**

通過減去背景來檢測前景物體。
v = VideoReader('traffic.mp4');
background = readFrame(v); % 使用第一幀作為背景
while hasFrame(v)
    frame = readFrame(v);
    diffFrame = imabsdiff(frame, background);
    grayFrame = rgb2gray(diffFrame);
    bw = imbinarize(grayFrame, 0.2);
    imshow(bw);
    pause(1/v.FrameRate);
end

##### 2-4. **影片的邊緣檢測**

使用邊緣檢測技術來處理影片中的每一幀。
v = VideoReader('sample.mp4');
while hasFrame(v)
    frame = readFrame(v);
    grayFrame = rgb2gray(frame);
    edges = edge(grayFrame, 'Canny');
    imshow(edges);
    pause(1/v.FrameRate);
end

##### 2-5. **影片的物體追蹤**

利用光流法（Optical Flow）來追蹤影片中的運動物體。
v = VideoReader('sample.mp4');
opticFlow = opticalFlowFarneback;
while hasFrame(v)
    frame = readFrame(v);
    grayFrame = rgb2gray(frame);
    flow = estimateFlow(opticFlow, grayFrame);
    imshow(frame);
    hold on;
    quiver(flow.Vx, flow.Vy, 'y');
    hold off;
    pause(1/v.FrameRate);
end

##### 2-6. **影片的色彩變換**

對影片進行色彩變換，比如轉換為灰度或其他色彩空間。
v = VideoReader('sample.mp4');
while hasFrame(v)
    frame = readFrame(v);
    hsvFrame = rgb2hsv(frame); % 將RGB轉換為HSV
    imshow(hsvFrame);
    pause(1/v.FrameRate);
end

##### 2-7. **影片的去噪**

使用濾波器來去除影片中的噪聲。
v = VideoReader('sample.mp4');
while hasFrame(v)
    frame = readFrame(v);
    filteredFrame = imgaussfilt(frame, 2); % 使用高斯濾波
    imshow(filteredFrame);
    pause(1/v.FrameRate);
end

##### 2-8. **影片的縮放**

對影片中的幀進行縮放處理。
v = VideoReader('sample.mp4');
while hasFrame(v)
    frame = readFrame(v);
    resizedFrame = imresize(frame, 0.5); % 將影像縮小一半
    imshow(resizedFrame);
    pause(1/v.FrameRate);
end

##### 2-9. **影片的幀差法**

利用前後幀的差異來檢測移動物體。
v = VideoReader('sample.mp4');
previousFrame = [];
while hasFrame(v)
    frame = readFrame(v);
    if isempty(previousFrame)
        previousFrame = frame;
        continue;
    end
    diffFrame = imabsdiff(frame, previousFrame);
    imshow(diffFrame);
    previousFrame = frame;
    pause(1/v.FrameRate);
end

##### 2-10. **影片的動態偵測**

使用 `vision.ForegroundDetector` 來動態檢測移動物體。
v = VideoReader('traffic.mp4');
detector = vision.ForegroundDetector();
while hasFrame(v)
    frame = readFrame(v);
    foreground = detector(frame);
    imshow(foreground);
    pause(1/v.FrameRate);
end

##### 2-11 Motion-based multiple object tracking

Example: Motion-based multiple object tracking
1. Detecting moving objects in each frame
2. 1. Associating the detections corresponding to the same object over time
-->detection object = background subtraction algorithm(GMM) + Morphological + blob 
-->association detections = Motion of each track(Kalman filter)
-->Track maintenance

------------------------------
v = <mark style="background: #FFB86CA6;">VideoReader</mark>('xx.mp3')
while <mark style="background: #FFB86CA6;">hasFrame</mark>(v)
    frame = <mark style="background: #FFB86CA6;">readFrame</mark>(v)
end   

obj.reader = <mark style="background: #FFB86CA6;">VideoReader</mark>('xx.mp3')
obj.maskPlayer = vision.<mark style="background: #FFB86CA6;">VideoPlayer</mark>('Position', [740, 400, 700, 400]);
obj.detector = vision.<mark style="background: #FFB86CA6;">ForegroundDetector</mark>('NumGaussians', 3)
obj.blobAnalyser = vision.<mark style="background: #FFB86CA6;">BlobAnalysis</mark>('BoundingBoxOutputPort')
opticalFlow = vision.<mark style="background: #FFB86CA6;">OpticalFlow</mark>('Method','Lucas-Kanade');

fun = @(block_struct) mean2(block_struct.data) * ones(size(block_struct.data)); 
result = <mark style="background: #FFB86CA6;">blockproc</mark>(gray_frame, [32, 32], fun);

------------------------------

### 3. <mark style="background: #FF5582A6;">get images from camera using matlab</mark>

##### 3-0. **相機初始化

MATLAB 可以使用 `videoinput` 函數來初始化相機並從中捕捉即時影像。
如果還沒有連接到webcam
<Method 1> 用<mark style="background: #D2B3FFA6;">Webcam Image Acquisition</mark>的functions
1. 安裝 Image Acquisition Toolbox
2. 安裝 Add-Ons -> MATLAB Support Package for USB Webcams
3. webcamlist
4. cam = webcam;
5. preview(cam);
<Method 2> 用<mark style="background: #D2B3FFA6;">Image Acquisition Toolbox</mark>的functions
1. 安裝 Image Acquisition Toolbox
2. 安裝 Add-Ons -> OS Generic Video interface
3. imaqhwinfo('winvideo')
4. cam = videoinput('winvideo', 1, 'RGB24_640x480'); 也可以改成'RGB24_320x240'
5. preview(cam);
如果有Unable to allocate memory for an incoming image frame due to insufficient free physical memory」錯誤: 
調整幀的解析度和格式('RGB24_320x240','Y800_640x480') % Y800 表示灰度格式

當 MATLAB 進行連續影像捕捉時，它會保留一個幀緩衝區來存儲影像幀。默認情況下，這個緩衝區可能會存儲多個幀，因此減少幀緩衝區大小可以減少內存使用
cam.FramesPerTrigger = 1;  % 每次觸發捕捉一幀
cam.TriggerRepeat = 0;  % 不重複觸發

釋放內存資源 clearvars;

##### 3-1. **相機初始化與即時影像捕捉**

% 初始化相機
vid = <mark style="background: #FFB86CA6;">videoinput</mark>('winvideo', 1, 'YUY2_640x480');
% 開啟相機即時預覽
<mark style="background: #FFB86CA6;">preview</mark>(vid);

##### 3-2. **捕捉單幀影像保存即時捕捉的影像**

使用 `getsnapshot` 函數從相機捕捉一幀影像並進行處理。
% 初始化相機
vid = videoinput('winvideo', 1);
% 捕捉單幀影像
img =<mark style="background: #FFB86CA6;"> getsnapshot</mark>(vid);
% 顯示捕捉到的影像
imshow(img);
imwrite(img, 'capturedImage.png');

##### 3-3. **保


##### 3-4. **即時影像處理**

即時從相機獲取影像並進行處理，比如轉換為灰度圖。
vid = videoinput('winvideo', 1);
preview(vid); % 開啟即時預覽
while true
    img = <mark style="background: #FFB86CA6;">getsnapshot</mark>(vid);
    grayImg = rgb2gray(img); % 轉為灰度圖
    imshow(grayImg);
    pause(0.1);
end
記得加入stop criteria !

##### 3-5. **即時物體追蹤**

使用光流法在即時捕捉的影像中追蹤移動物體。
vid = videoinput('winvideo', 1);
opticFlow = <mark style="background: #FFB86CA6;">opticalFlowFarneback</mark>;
preview(vid);
for t=1:100
    frame = getsnapshot(vid);
    grayFrame = rgb2gray(frame);
    flow = <mark style="background: #FFB86CA6;">estimateFlow</mark>(opticFlow, grayFrame);
    imshow(frame);
    hold on;
    <mark style="background: #FFB86CA6;">quiver</mark>(flow.Vx, flow.Vy, 'y');
    hold off;
    pause(0.1);
end

##### 3-6. **相機畸變校正**

使用 MATLAB 的相機校正工具箱來進行相機的校正，以糾正圖像中的鏡頭畸變。
% 載入校正影像
images = imageDatastore(fullfile(toolboxdir('vision'),'visiondata','calibration','mono'));
<span style="color:rgb(0, 200, 0)"># Detect calibration pattern</span>
[imagePoints,boardSize] = <mark style="background: #FFB86CA6;">detectCheckerboardPoints</mark>(images.Files);
<span style="color:rgb(0, 200, 0)"># Generate world coordinates of the corners of the squares</span>
squareSize = 29;
worldPoints = <mark style="background: #FFB86CA6;">patternWorldPoints</mark>('checkerboard',boardSize,squareSize);
<span style="color:rgb(0, 200, 0)"># Calibrate the camera</span>
I = readimage(images,1); imageSize = [size(I,1),size(I,2)];
cameraParams = <mark style="background: #FFB86CA6;">estimateCameraParameters </mark>(imagePoints,worldPoints,'ImageSize',imageSize);
<span style="color:rgb(0, 200, 0)"># Remove lens distortion</span>
J1 = <mark style="background: #FFB86CA6;">undistortImage</mark>(I,cameraParams);
##### 3-7. **相機畸變校正**

對

##### 3-8. **多相機同步**

使用多個相機同時捕捉影像並進行同步處理。
vid1 = videoinput('winvideo', 1);
vid2 = videoinput('winvideo', 2);
% 開啟同步捕捉
start([vid1, vid2]);
frames1 = <mark style="background: #FFB86CA6;">getdata</mark>(vid1);
frames2 = getdata(vid2);
% 顯示兩個相機捕捉的影像
<mark style="background: #FFB86CA6;">imshowpair</mark>(frames1, frames2, 'montage');

##### 3-9. **即時人臉檢測**

在即時相機影像中檢測人臉。
faceDetector = <mark style="background: #FFB86CA6;">vision.CascadeObjectDetector</mark>;
vid = videoinput('winvideo', 1);
preview(vid);
while true
    img = getsnapshot(vid);
    bbox = <mark style="background: #FFB86CA6;">step</mark>(faceDetector, img);
    detectedImg = <mark style="background: #FFB86CA6;">insertObjectAnnotation</mark>(img, 'rectangle', bbox, 'Face');
    imshow(detectedImg);
    pause(0.1);
end

##### 3-10. **自動對焦控制**

控制相機自動對焦，針對特定區域進行對焦。
vid = videoinput('winvideo', 1);
src = getselectedsource(vid);
% 設置對焦模式為自動
src.FocusMode = 'auto';
% 捕捉影像
img = getsnapshot(vid);
imshow(img);


m = mobiledev   # Acquire Images from a Mobile Device Camera
cam = <mark style="background: #FFB86CA6;">camera</mark>(m,'back')
img = <mark style="background: #FFB86CA6;">snapshot</mark>(cam,'immediate');
image(img)

### 4. <mark style="background: #FF5582A6;">matlab use c++ functions</mark>


<mark style="background: #FFB86CA6;">mex</mark> -setup c++  <span style="color:rgb(0, 200, 0)">   # 設定matlab的compiler去編譯mex file</span>

<mark style="background: #FFB86CA6;">mex</mark> yprime.c   <span style="color:rgb(0, 200, 0)"># 將c++ code轉成.mex 讓matlab可以直接調用</span>

<mark style="background: #FFB86CA6;">clibPublishInterfaceWorkflow</mark>   <span style="color:rgb(0, 200, 0)"># construct a c++ library(*.dll) from header(.h) and source files(.cpp) which matlab can use(publish a MATLAB interface for a C++ library)</span>


### 5. <mark style="background: #FF5582A6;">AI Deep learning toolbox</mark>



### 6. <mark style="background: #FF5582A6;">attach matlab on cloud </mark>

------------------------------
MATLAB 提供了許多與雲計算相關的功能，用於在雲端上執行計算、訪問數據和管理資源。這些功能可以幫助你與雲服務（如 AWS、Azure 和 MATLAB Cloud）進行交互。以下列出一些常用的 MATLAB 雲相關指令，並提供每個指令的中文解釋及示例。
1
<mark style="background: #FFB86CA6;">matlab.io.datastore.DsFileSet</mark>
這個指令可以創建一個 Datastore 文件集，方便從雲端的存儲系統讀取文件。
fs = matlab.io.datastore.DsFileSet('s3://mybucket/data/');

2.
<mark style="background: #FFB86CA6;">parcluster</mark>
用來連接到 MATLAB 雲端集群或本地集群，支持併行計算。
c = parcluster('MATLABCloud');

3.
<mark style="background: #FFB86CA6;">batch</mark>
用於將批次作業提交到 MATLAB 雲端集群上執行。
job = batch(c, @myFunction, 1, {arg1, arg2});

4.
<mark style="background: #FFB86CA6;">fetchOutputs</mark>
從雲端集群中提取已完成批次作業的輸出。
output = fetchOutputs(job);

5.
<mark style="background: #FFB86CA6;">parpool</mark>
在雲端集群或本地集群上打開並行工作池。
parpool(c, 4);

6.
<mark style="background: #FFB86CA6;">parallel.cluster.Generic</mark>
定義一個自定義的通用集群，適合不同的雲端環境。
c = parallel.cluster.Generic();

7.
<mark style="background: #FFB86CA6;">cloudStorageLocation</mark>
指定數據存儲的雲端位置，用於儲存和讀取大型數據集。
loc = cloudStorageLocation('s3://mybucket/folder/');

8.
<mark style="background: #FFB86CA6;">upload</mark>
將本地文件上傳到指定的雲端存儲位置。
upload('localfile.txt', 's3://mybucket/folder/');

9.
<mark style="background: #FFB86CA6;">download</mark>
從雲端存儲位置下載文件到本地。
download('s3://mybucket/folder/datafile.csv', 'localdatafile.csv');

10.
<mark style="background: #FFB86CA6;">deleteCloudFile</mark>
刪除存儲在雲端的文件或資料夾。
deleteCloudFile('s3://mybucket/folder/datafile.csv');

11.
<mark style="background: #FFB86CA6;">listCloudFiles</mark>
列出指定雲端存儲位置的文件和資料夾。
files = listCloudFiles('s3://mybucket/folder/');

12.
<mark style="background: #FFB86CA6;">cloudFileExists</mark>
檢查指定的雲端存儲位置是否存在某個文件。
exists = cloudFileExists('s3://mybucket/folder/datafile.csv');

13.
<mark style="background: #FFB86CA6;">cloudFileSize</mark>
獲取存儲在雲端的文件大小。
size = cloudFileSize('s3://mybucket/folder/datafile.csv');

14.
<mark style="background: #FFB86CA6;">cloudFolderExists</mark>
檢查指定的雲端資料夾是否存在。
exists = cloudFolderExists('s3://mybucket/folder/');

15.
<mark style="background: #FFB86CA6;">cloudFileModifiedTime</mark>
獲取存儲在雲端文件的上次修改時間。
modTime = cloudFileModifiedTime('s3://mybucket/folder/datafile.csv');

16.
<mark style="background: #FFB86CA6;">cloudMove</mark>
移動或重命名雲端存儲中的文件或資料夾。
cloudMove('s3://mybucket/folder/datafile.csv', 's3://mybucket/backup/datafile.csv');

17.
<mark style="background: #FFB86CA6;">createCloudFolder</mark>
在指定的雲端存儲位置創建新資料夾。
createCloudFolder('s3://mybucket/newfolder/');

18.
<mark style="background: #FFB86CA6;">datastore</mark>
創建一個數據存儲對象，用於讀取存儲在雲端的文件，支持大數據處理。
ds = datastore('s3://mybucket/datafolder/');

19.
<mark style="background: #FFB86CA6;">matlab.io.datastore.S3Datastore</mark>
讀取存儲在 Amazon S3 上的數據集。
s3ds = matlab.io.datastore.S3Datastore('s3://mybucket/data/');

20.
<mark style="background: #FFB86CA6;">cloudWrite</mark>
將數據寫入到雲端文件中。
cloudWrite('s3://mybucket/folder/data.txt', 'This is sample data');

------------------------------

### 7. <mark style="background: #FF5582A6;">export to  csv/txt</mark>

ref: [Data Import and Export](https://www.mathworks.com/help/matlab/data-import-and-export.html?s_tid=CRUX_lftnav)

------------------------------
##### CSV to table/array/cell/struct

T = <mark style="background: #FFB86CA6;">readtable</mark>('airlinesmall_subset.xlsx','Sheet','2008');
T_selected = readtable('airlinesmall_subset.xlsx','Sheet','1996','Range','A1:E11')

ds = <mark style="background: #FFB86CA6;">spreadsheetDatastore</mark>('airlinesmall_subset.xlsx');

T = readtable('csv_table.txt')
LastName = T.LastName
data_part = <mark style="background: #FFB86CA6;">table2array</mark>(T(:, 3:end))
Tstruct = <mark style="background: #FFB86CA6;">table2struct</mark>(T)
Tcell = <mark style="background: #FFB86CA6;">table2cell</mark>(T)

table_struct = <mark style="background: #FFB86CA6;">struct2table</mark>(Tstruct)
cell_struct = <mark style="background: #FFB86CA6;">cell2table</mark>(Tcell, 'VariableNames', T.Properties.VariableNames)

##### table to txt/csv

T = table(['M';'F';'M'],[45;41;36],...
	{'New York, NY';'San Diego, CA';'Boston, MA'},[true;false;false])

<mark style="background: #FFB86CA6;">writetable</mark>(T,'myData.csv')
<mark style="background: #FFB86CA6;">writetable</mark>(T,'myData.txt')

functions - [Spreadsheets — Functions](https://www.mathworks.com/help/matlab/referencelist.html?type=function&category=spreadsheets&s_tid=CRUX_topnav)

------------------------------


### 8. <mark style="background: #FF5582A6;">Image quality and calibration</mark>

Image Types in Matlab

| [Binary Images](https://www.mathworks.com/help/images/image-types-in-the-toolbox.html#f14-33397)                                                        | m × n二進位矩陣 0 值是黑色，所有非零值都是白色                                                                                                                            |
| ------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [Indexed Images](https://www.mathworks.com/help/images/image-types-in-the-toolbox.html#f14-17587)                                                       | 索引影像由影像矩陣和色彩圖組成。顏色圖是資料類型為c × 3 的矩陣 double，其值在 [0, 1] 範圍內。顏色圖的每一行指定單一顏色的RGB values<br><br>影像矩陣中的像素值是色彩圖的直接索引。索引影像中每個像素的顏色是透過將影像矩陣中的像素值對應到顏色圖中對應的顏色來確定的。 |
| [Grayscale Images](https://www.mathworks.com/help/images/image-types-in-the-toolbox.html#f14-13941) (intensity image)                                   | m × n數字矩陣，其元素指定強度值<br>對於single或 double數組，值範圍為 [0, 1]。<br>對於uint8數組，值的範圍是 [0, 255]。<br>對於uint16，值範圍為 [0, 65535]。<br>值範圍為 [-32768, 32767]。               |
| [Truecolor Images](https://www.mathworks.com/help/images/image-types-in-the-toolbox.html#f14-20224) (RGB image)                                         | m × n ×3 數值數組三個顏色通道(RGB)之一的強度值<br>對於single或 double數組，RGB 值範圍為 [0, 1]。<br>對於uint8數組，RGB 值的範圍為 [0, 255]。<br>對於uint16，RGB 值範圍為 [0, 65535]。                |
| [High Dynamic Range (HDR) Images](https://www.mathworks.com/help/images/image-types-in-the-toolbox.html#mw_44f5abc0-7a74-49d8-bce6-5bdc1625e869)        | m × n數值矩陣或 m × n ×3 數值數組類似於灰階或 RGB 影像. HDR 影像具有資料類型single或double，但資料值不限於範圍 [0, 1] 並且可以包含 Inf值                                                          |
| [Multispectral and Hyperspectral Images](https://www.mathworks.com/help/images/image-types-in-the-toolbox.html#mw_0c176e11-05b7-447b-8009-11835b396e27) | 影像資料儲存為 m × n × c 數值數組，其中c是顏色通道數。                                                                                                                      |
| [Label Images](https://www.mathworks.com/help/images/image-types-in-the-toolbox.html#mw_e2c9ca6c-2e5b-49e3-94e9-eab9275c19f5)                           | 影像資料儲存為 m × n分類矩陣或非負整數的數值矩陣。                                                                                                                           |

Image quality in Matlab

| Full reference quality metrics            |                                                                                                       |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| immse                                     | Mean-squared error                                                                                    |
| psnr                                      | Peak signal-to-noise ratio (PSNR)                                                                     |
| ssim                                      | Structural similarity (SSIM) index for measuring image quality                                        |
| **No-reference quality matrics**          |                                                                                                       |
| niqe                                      | Naturalness Image Quality Evaluator (NIQE) no-reference image quality score                           |
| brisque                                   | Blind/Referenceless Image Spatial Quality Evaluator (BRISQUE) no-reference image quality score        |
| **Test chart based quality measurements** |                                                                                                       |
| measureColor                              | Measure <mark style="background: #BBFABBA6;">color reproduction</mark> using test chart               |
| measureIlluminant                         | Measure <mark style="background: #BBFABBA6;">scene illuminant</mark> using test chart                 |
| measureNoise                              | Measure <mark style="background: #BBFABBA6;">noise</mark> of test chart                               |
| measureChromaticAberration                | Measure <mark style="background: #BBFABBA6;">chromatic aberration</mark> at slanted edges             |
| measureSharpness                          | Measure <mark style="background: #BBFABBA6;">spatial frequency response</mark> using test chart       |
| measureTexture                            | Measure <mark style="background: #BBFABBA6;">sharpness</mark> of texture using Dead Leaves test chart |
esfr test chart 可以校正

![[esfr.jpg]]

[MATLAB图像处理：106：评估 eSFR 测试图表上的质量指标](https://zhuanlan.zhihu.com/p/400938718)

I = imread("eSFRTestImage.jpg");
chart = <mark style="background: #FFB86CA6;">esfrChart</mark>(I);
<mark style="background: #FFB86CA6;">displayChart</mark>(chart,displayEdgeROIs=false, ...
    displayGrayROIs=false,displayRegistrationPoints=false)

所有 60 个倾斜边缘 ROI（以绿色标记）都是可见的，并以适当的边缘为中心。此外，20 个灰色补丁 ROI（以红色标记）和 16 个彩色补丁 ROI（以白色标记）是可见的，并且包含在每个补丁的边界内。

测量边缘锐度
[sharpnessTable,aggregateSharpnessTable] = <mark style="background: #FFB86CA6;">measureSharpness</mark>(chart);
测量色差
chTable = <mark style="background: #FFB86CA6;">measureChromaticAberration</mark>(chart);
测量噪声
noiseTable = <mark style="background: #FFB86CA6;">measureNoise</mark>(chart);
估计光源
illum = <mark style="background: #FFB86CA6;">measureIlluminant</mark>(chart)
测量颜色准确度
[colorTable,ccm] = <mark style="background: #FFB86CA6;">measureColor</mark>(chart);


[色域空间转换（color space）](https://zhuanlan.zhihu.com/p/559743079)

<mark style="background: #ADCCFFA6;">色彩模式(Colour model)</mark> - 色彩的表现方式
RGB(光的加和), 
HSB(心理学对颜色的感知，从色相（H），饱和度（S），明度（B）三个维度来描述一个颜色), CMYK(颜料对光的吸收) 
Lab(人眼对颜色的感知维度，以明度值，a为绿->红互补色对偏向值，b为蓝->黄互补色对偏向值)

<mark style="background: #ADCCFFA6;">色彩空间(Color Space) </mark>- 就是一个设备所能表现的所有颜色的集合
用数学语言表示，就是cES，其中c为任一颜色，S为某一色彩空间。 Example:sRGB，Adobe RGB

<mark style="background: #ADCCFFA6;">色域(Color gamut)</mark> - 色彩空间这个集合的范围
比如刚才说的Adobe RGB的色域比sRGB的大，而sRGB的色域又全部包含在Adobe RGB中，那么用数学语言描述，sRGB这个色彩空间就是Adobe RGB的（真）子集，即S（sRGB）C（）S（Adobe RGB）。