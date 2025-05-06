
在MATLAB、Python、C++及OpenCV（Python或C++）下，載入常用的2D或3D圖像或視頻文件，儲存成適當的數據結構並進行顯示，以下是詳細解釋和示例代碼。

|                |                                                                                                                                       |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| Matlab         | img = imread('example.jpg')<br>imshow(img)                                                                                            |
| python         | from PIL image Image<br>from matplotliv.pyplot as plt<br>img = Image.imread('example.jpg')<br>plt.imshow(img)<br>                     |
| opencv(python) | import cv2<br>img = cv2.imread('example.jpg')<br>cv2.imshow('Image', img)                                                             |
| opencv(c++)    | #include <opencv2/opencv.hpp><br><br>int main() {<br>   cv::Mat img = cv::imread("example.jpg");<br>   cv::imshow("Image", img);<br>} |

### 1. MATLAB

MATLAB支持多種圖像和視頻文件格式，主要使用`imread`載入2D圖像，`niftiread`或`dicomread`載入3D醫學圖像，`VideoReader`載入視頻文件。數據結構為`matrix`（矩陣）或`tall`（適用於大數據）。當你使用 `imread` 來載入一個32位深度的圖像時，MATLAB 默認會將圖像數據自動轉換成 `uint8` 格式。這是因為 `imread` 在不指定格式的情況下，會嘗試將圖像數據縮放到一個標準格式以便於處理，通常為 `uint8`（範圍為0到255），這是許多圖像處理任務的常見格式。

###### (1) 讀取和顯示2D圖像：
% 讀取2D圖像
img = <mark style="background: #FFB86CA6;">imread</mark>('example.jpg');
% 顯示圖像
imshow(img);
**數據結構**: `img` 是一個矩陣，大小為`[height, width, channels]`。

###### (2) 讀取和顯示3D圖像 (NIfTI格式)：
% 讀取NIfTI 3D圖像
nifti_img = <mark style="background: #FFB86CA6;">niftiread</mark>('example.nii');
% 顯示3D圖像中的某一切片
imshow(nifti_img(:,:,30), []);
**數據結構**: `nifti_img` 是一個3D矩陣，大小為`[height, width, depth]`。

###### (3) 讀取和顯示視頻：
% 創建視頻對象
videoObj = <mark style="background: #FFB86CA6;">VideoReader</mark>('example.mp4');
% 讀取第一幀圖像
frame = <mark style="background: #FFB86CA6;">readFrame</mark>(videoObj);
% 顯示幀圖像
imshow(frame);
**數據結構**: `frame` 是一個3D矩陣，大小為`[height, width, channels]`。

### 2. Python（使用PIL、NumPy、Nibabel）

在Python中，使用`Pillow`來處理2D圖像，`Nibabel`來處理3D醫學圖像，`cv2.VideoCapture`來處理視頻文件。數據結構通常為`NumPy`陣列。

###### (1) 讀取和顯示2D圖像：
from <mark style="background: #ADCCFFA6;">PIL</mark> import <mark style="background: #ADCCFFA6;">Image</mark>
import <mark style="background: #ADCCFFA6;">matplotlib</mark>.pyplot as <mark style="background: #ADCCFFA6;">plt</mark>
讀取2D圖像
img = <mark style="background: #FFB86CA6;">Image.open</mark>('example.jpg')
顯示圖像
plt.imshow(img)
plt.show()
**數據結構**: `img` 是一個`PIL`圖像對象，可以轉換成`NumPy`陣列。

###### (2) 讀取和顯示3D圖像 (NIfTI格式)：
import nibabel as nib
import matplotlib.pyplot as plt
讀取NIfTI 3D圖像
nifti_img = nib.load('example.nii')
img_data = nifti_img.get_fdata()
顯示3D圖像中的某一切片
plt.imshow(img_data[:,:,30], cmap='gray')
plt.show()
**數據結構**: `img_data` 是一個`NumPy` 3D陣列。

###### (3) 讀取和顯示視頻：
import cv2
創建視頻對象
cap = cv2.VideoCapture('example.mp4')
讀取第一幀圖像
ret, frame = cap.read()
顯示幀圖像
cv2.imshow('Frame', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
**數據結構**: `frame` 是一個`NumPy`陣列，大小為`[height, width, channels]`。

### 3. C++（使用OpenCV）

在C++中，OpenCV是一個強大的工具，既可以讀取2D圖像，也可以處理視頻。數據結構是`cv::Mat`。

#### (1) 讀取和顯示2D圖像：
#include <opencv2/opencv.hpp>
int main() {
    // 讀取2D圖像
    cv::Mat img = cv::imread("example.jpg");
    // 顯示圖像
    cv::imshow("Image", img);
    cv::waitKey(0);
    return 0;
}
**數據結構**: `img` 是一個`cv::Mat`對象，大小為`[height, width, channels]`。

###### (2) 讀取和顯示視頻：
#include <opencv2/opencv.hpp>
int main() {
    // 創建視頻對象
    cv::VideoCapture cap("example.mp4");
    // 讀取第一幀圖像
    cv::Mat frame;
    cap.read(frame);
    // 顯示幀圖像
    cv::imshow("Frame", frame);
    cv::waitKey(0);
    return 0;
}
**數據結構**: `frame` 是一個`cv::Mat`對象，大小為`[height, width, channels]`。

### 4. Python (使用OpenCV)

在Python中，OpenCV也可以用來處理2D圖像和視頻。數據結構同樣是`NumPy`陣列。

#### (1) 讀取和顯示2D圖像：
import <mark style="background: #ADCCFFA6;">cv2</mark>
讀取2D圖像
img = <mark style="background: #FFB86CA6;">cv2.imread</mark>('example.jpg')
顯示圖像
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
**數據結構**: `img` 是一個`NumPy`陣列。

###### (2) 讀取和顯示視頻：
import cv2
創建視頻對象
cap = cv2.VideoCapture('example.mp4')
讀取第一幀圖像
ret, frame = cap.read()
顯示幀圖像
cv2.imshow('Frame', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
**數據結構**: `frame` 是一個`NumPy`陣列。




