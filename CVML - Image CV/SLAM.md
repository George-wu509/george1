
SLAM（Simultaneous Localization and Mapping）是指機器人在未知環境中一邊定位自身位置，一邊構建周圍環境的地圖。這是自主機器人和無人駕駛技術中的關鍵技術。

#### 常用的SLAM算法：

1. **<mark style="background: #FFB8EBA6;">EKF-SLAM</mark>（Extended Kalman Filter SLAM）**
    
    - **方法概述**: EKF-SLAM 利用<mark style="background: #BBFABBA6;">擴展卡爾曼濾波器</mark>進行機器人狀態估計和地圖構建。這種方法基於線性化的卡爾曼濾波器，適合處理小噪聲的情況。
    - **優點**: 算法簡單，理論完善，適合小規模場景。
    - **缺點**: 當環境變得複雜或存在較大噪聲時，EKF-SLAM 的性能會明顯下降。
2. **<mark style="background: #FFB8EBA6;">FastSLAM</mark>**
    
    - **方法概述**: FastSLAM 將機器人的<mark style="background: #BBFABBA6;">狀態估計和地圖的特徵估計解耦</mark>，使用<mark style="background: #BBFABBA6;">粒子濾波器</mark>來進行機器人狀態的跟蹤，並使用EKF來估計每個地標的位置。
    - **優點**: 具有較好的擴展性，能夠處理大型地圖並且在多個特徵點的環境中表現良好。
    - **缺點**: 計算量較大，粒子數過少會影響效果，過多則增加運算負擔。
3. **<mark style="background: #FFB8EBA6;">ORB-SLAM</mark>（Oriented FAST and Rotated BRIEF SLAM）**
    
    - **方法概述**: ORB-SLAM 是基於特徵點的視覺SLAM算法，使用ORB特徵來檢測和描述圖像中的關鍵點，從而進行位姿估計和地圖構建。
    - **優點**: 精度高，適合室內外場景，且對運算資源的要求較低。
    - **缺點**: 對光照變化敏感，且對於無明顯特徵的環境效果不佳。
4. **<mark style="background: #FFB8EBA6;">LSD-SLAM</mark>（Large-Scale Direct Monocular SLAM）**
    
    - **方法概述**: LSD-SLAM 是一種直接法的單目SLAM，使用每個像素的光度信息來估計深度和位姿，而不依賴特徵點匹配。
    - **優點**: 能夠處理大規模的環境，且在低特徵環境中也能工作。
    - **缺點**: 對噪聲敏感，對運算能力要求高。
5. **<mark style="background: #FFB8EBA6;">DSO</mark>（Direct Sparse Odometry）**
    
    - **方法概述**: DSO 是一種稀疏的直接法SLAM算法，通過使用圖像中的少量像素來估計位姿和重建場景。
    - **優點**: 速度快，對計算資源的要求較低，且能處理複雜場景。
    - **缺點**: 對光照變化敏感，對影像質量要求高。

### 總結

- **影像修復** 常用的算法包括基於紋理合成、樣本傳播、偏微分方程、低秩矩陣分解和GAN等方法。
- **三維重建** 常用的算法包括結構從運動（SfM）、多視角立體（MVS）、單視圖重建、深度學習三維重建和體積重建。
- **SLAM** 常用的算法包括EKF-SLAM、FastSLAM、ORB-SLAM、LSD-SLAM和DSO。