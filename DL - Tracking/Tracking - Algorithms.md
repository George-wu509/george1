


|                           |                                                                    |
| ------------------------- | ------------------------------------------------------------------ |
| [[###EuclideanDistTrack]] |                                                                    |
| [[###SORT]]               | Simple Online And RealtimeTracking                                 |
| [[###DeepSORT]]           | Simple Online and Realtime Tracking with a Deep Association Metric |
| ByteTrack                 | Multi-Object Tracking by Associating Every Detection Box           |


目標跟蹤的主要流程包括三步，下文的描述主要也是圍繞下面三點展開：

- 新的檢測目標進入後，需要初始化，分配新ID
- 當前幀與前面若干幀的物體如何匹配
- 當某個目標多次不出現（連續多幀不出現），如何剔除

#### 相關難點

本文所提及的所有目標跟蹤算法需要在良好的目標檢測基礎上進行，故第一個難點在於“精確的目標檢測”；當然，這不是本文描述的重點，可以假設我們已經擁有了一個高精度的目標檢測器。其次，在一整段影片中，目標是在不斷變化的，包括其位置、外觀、大小，如何匹配當前幀和上一幀的物體是一個富有技巧性的問題。再者，每一幀中新進入的目標、退出的目標應該如何處理？最後，由於遮擋等原因導致的目標短暫離開又應該如何解決？

#### 基本名詞

- Tracks: 是指要開始追蹤的、已經初始化的所有目標狀態量。
- Detections: 是通过目標檢測器获取的当前帧的檢測框。



### EuclideanDistTrack

#### 主要思路及流程

EuclideanDistTrack可以説是最簡單、直觀的目標跟蹤算法。我們可以直觀地想，影片相鄰兩幀中的同一個目標距離必然不會“太遠”，我們可以通過計算相鄰兩幀中所有物體的距離從而進行判斷；距離小於一定值的很大機率是同一個目標，從而實現目標跟蹤的目的。

上面所講的“距離”我們可以使用兩物體中心點的EuclideanDist（二范數）進行計算。

我們可以結合以下流程圖，講解EuclideanDistTrack的總體思路。

當最開始，目標檢測器檢測到新物體時，分配新ID，並把ID和當前物體的中心座標同時放入容器存儲；在下一幀影片中，需要計算檢測到的物體與存儲的物體的距離，當距離小於一定閾值，即認為兩者是同一個目標，此時更新原始容器中該物體的中心座標，並將它的所有訊息再次存儲起來；當距離不小於一定閾值，將當前檢測到的目標當作新物體，操作同上。最後，之前存儲而在當前幀中沒有匹配到的物體會被直接刪除。

為了方便，我們直接定義一個class，其中id_count會記錄id的更新，center_points會記錄各個物體的ID以及當前中心點的座標。在每一幀對Tracks進行更新時，都會調用update方法（下文會解釋）。而在update方法中，目標檢測器檢測到新物體時，分配新ID，並把ID和當前物體的中心座標同時放入容器center_points和臨時容器1objects_bbs_ids存儲。在下一幀影片中，需要計算當前幀檢測到的物體與center_points存儲的物體的距離；當距離小於一定閾值（這裡設定為25，可改變），即認為兩者是同一個目標，此時更新原始容器中該物體的中心座標，並將它的所有訊息再次存入容器objects_bbs_ids。遍歷完當前幀所有的檢測後，只在容器center_points而不在容器objects_bbs_ids中的物體會視為從當前幀開始已經消失的物體，予以刪除（在代碼上表示的是沒有從容器center_points搬到容器new_center_points的物體）。



### SORT

#### SORT：基於Kalman filter的IOU匹配

SORT算法引入了Kalman filter，利用前面的幀對後面的幀的狀態（主要是位置）進行預測，利用預測的位置與檢測器檢測到的位置計算IOU，以IOU作為成本矩陣，利用hungarian algorithm(匈牙利算法)進行最優匹配。同時對於前面幀中的Tracks，如果當前幀未檢測到，不會直接刪除，會具有一定容忍度；只有連續若干幀都無法匹配，才會將其刪除。

卡爾曼濾波器在預測的同時也要利用檢測器觀測的值（預測與預測的誤差值）進行更新，不斷提升往後的預測性能，即卡爾曼濾波器是觀測值與預測值的結合與取捨。總體來看，SORT算法在匹配與刪除兩個步驟中合理了不少。

![[Pasted image 20250326115854.png]]

您對 SORT（Simple Online and Realtime Tracking）演算法流程的描述基本正確。以下我將更詳細地說明，並回答您關於多目標追蹤的問題：

**SORT 演算法流程：**

1. **目標檢測：**
    - 首先，使用目標檢測模型（例如 YOLOv8）對影片的每一幀進行物件檢測。
    - 目標檢測模型提供每個檢測到物件的邊界框（bounding box）。
2. **狀態估計（卡爾曼濾波器）：**
    - 對於每一幀的每個已追蹤物件，使用卡爾曼濾波器（Kalman filter）預測其在下一幀（t+1）的位置。
    - 卡爾曼濾波器用於平滑物件的運動軌跡，並預測其未來位置，這有助於處理檢測中的噪聲和不確定性。
3. **資料關聯（匈牙利演算法）：**
    - 在下一幀（t+1），將檢測到的邊界框與卡爾曼濾波器預測的邊界框進行比較。
    - 使用匈牙利演算法建立成本矩陣，該矩陣表示檢測到的邊界框和預測的邊界框之間的「距離」（例如，IoU 的倒數）。
    - 匈牙利演算法用於找到一種最佳的配對方式，使得總「距離」最小。
    - 將適當的 ID 分配給下一幀（t+1）中的檢測到的邊界框，以保持物件的追蹤。
4. **ID 管理：**
    - 追蹤演算法還需要管理物件的 ID。例如：
        - 當一個新物件進入場景時，分配一個新的 ID。
        - 當一個物件離開場景時，刪除其 ID。

**SORT 與多目標追蹤（MOT）：**

- 是的，SORT 專為多目標追蹤而設計。
- 它的主要目的是在影片序列中追蹤多個物件的運動。
- SORT通過卡爾曼濾波器對於物件位置的預測，結合匈牙利算法對於物件在不同影格中的匹配，有效的完成了多目標追蹤。
- 但是SORT演算法單純的使用物件的運動訊息進行多物件追蹤，對於物件發生遮擋，或者外觀類似的情況，容易發生ID switch的現象。為了改善這個情況，DeepSORT在SORT的基礎之上增加了物件外觀特徵的判別，有效的降低了ID switch的機率。

**總結：**

- 您的描述準確地反映了 SORT 演算法的基本流程。
- SORT 是一種有效且快速的多目標追蹤演算法。
- 為了更加有效率的完成多物件追蹤，可以考慮使用DeepSORT這類的追蹤演算法。


首先需要定義一個Tracker類，當檢測器檢測到新物體時，都需要初始化（實例化）一個Tracker
```python
class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=7, dim_z=4) 
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    self.kf.R[2:,2:] *= 10.  #measurement uncertainty/noise
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01  #Process uncertainty/noise
    self.kf.Q[4:,4:] *= 0.01

    self.kf.x[:4] = convert_bbox_to_z(bbox)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0  # the continue time object tracked
    self.age = 0
```

初始化過程中，需要初始化卡爾曼濾波器的所有相關矩陣（按照上文的數值進行初始化）。還有幾個參數值需要特別留意，包括：time_since_update 該Tracker上次更新後總共未更新的次數（幀數），KalmanBoxTracker.count用於產生新id，hits物體被追蹤的次數（幀數），hit_streak物體被連續追蹤的次數（幀數），age Tracker被預測的次數，history 用於保存被追蹤的Tracker最新的狀態。

根據上文，卡爾曼濾波器需要有預測和更新兩步，因此類中需要定義兩個方法：
```python
  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0 #一旦更新表示Tracker已被追蹤
    self.history = []
    self.hits += 1  #物體被追蹤的次數（幀數）+1
    self.hit_streak += 1  #物體被連續追蹤的次數（幀數）+1
    self.kf.update(convert_bbox_to_z(bbox)) #更新狀態

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):  #防止邊界溢出
      self.kf.x[6] *= 0.0
    self.kf.predict()  #預測
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]
```
接着定義SORT類：
```python
class Sort(object):
  def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []
    self.frame_count = 0
```
min_hits表示最少需要被追蹤的幀數，超過該數量，追蹤結果才會顯示在影片中；max_age表示Tacker被容忍丟失的幀數，一旦超過，將被刪除；iou_threshold表示成本矩陣的匹配閾值。

SORT類中需要一個update方法，按照流程圖對每一幀分情況進行處理：
```python
def update(self, dets=np.empty((0, 5))):
    self.frame_count += 1
    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 5))
    to_del = []
    ret = []
#預測結果超出範圍的Trackers直接刪除
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)

    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks, self.iou_threshold)

    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :])

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:])
        self.trackers.append(trk)
    i = len(self.trackers)

    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        #print('ss', trk)
#當前幀更新的Trackers同時 （被連續追蹤超過次數）的保留
        if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        # remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,5))
```
上面的代碼中，會對現有的Trackers進行預測，並對預測結果進行處理（結果超出範圍的Trackers直接刪除）。預測的結果會與檢測器的結果進行IOU匹配。根據前文提及的三個步驟， matched的Tackers直接更新，unmatched detections進行卡爾曼濾波器初始化，unmatched_trks中連續未被追蹤超過max_ag次的刪除。

結合YOLOV8檢測器，SORT跟蹤器的使用如下：
```python
from ultralytics import YOLO
import cv2,cvzone,math
from sort import *

cap=cv2.VideoCapture("E:\\opencv\\yolo8Sort\\Videos\\cars.mp4")
model=YOLO("../weights/yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

tracker=Sort(max_age=20,min_hits=2,iou_threshold=0.3)

limits=[400,297,673,297]
totalCount=set()

while True:
    success,img=cap.read()
    results=model(img,stream=True)
    detections=np.empty((0,6))
    imgGraphics=cv2.imread('E:\\opencv\\yolo8Sort\\graphics.png',cv2.IMREAD_UNCHANGED)
    img=cvzone.overlayPNG(img,imgGraphics,(0,0))

    for r in results:
        boxes=r.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]
            x1, y1, x2, y2=int(x1),int(y1),int(x2),int(y2)
            w,h=x2-x1,y2-y1
            cvzone.cornerRect(img,(x1,y1,w,h),l=9)

            conf=math.ceil((box.conf[0]*100))/100
            cls=int(box.cls[0])
            currentClass=classNames[cls]
            if currentClass=="car" or currentClass=="truck" or currentClass=="bus"\
                or currentClass=="motorbike" and conf>0.3:

                cvzone.putTextRect(img,f'{classNames[cls]}',(max(0,x1),max(35,y1)),
                               scale=1,offset=3,thickness=1)
                currentArray=np.array([x1,y1,x2,y2,conf,cls])
                detections=np.vstack((detections,currentArray))
    resultTracker=tracker.update(detections)
    cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5)
    for result in resultTracker:
        x1, y1, x2, y2 ,id=result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.putTextRect(img, f'ID:{int(id)}', (max(0, x1)+50, max(35, y1)),
                           scale=1, offset=3 , thickness=1)
        cx,cy=x1+w//2,y1+h//2

        if limits[0]<cx <limits[2] and limits[1] -20<cy <limits[1]+20:
            totalCount.add(int(id))
        cv2.putText(img,str(len(totalCount)),(255,100),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8)

    cv2.imshow("image",img)
    key=cv2.waitKey(10)
    if key==27:
        break
```



### DeepSORT
#### DeepSORT:加入神經網絡特徵提取器生成的embedding特徵進行匹配

DeepSORT可以算是SORT算法的升級版，主要針對如何“更好地匹配”進行改進.。為了更好地進行匹配，DeepSORT引入了級聯匹配和深度特徵（ReID）匹配，極大優化了匹配的過程。

### 整體流程

先看整體流程，以對算法有一個大致的瞭解。
![[Pasted image 20250326132826.png]]

所有Tracker被分成兩個類別（對應兩種狀態），分別是confirmed(被連續追蹤三次以上)以及unconfirmed（被連續追蹤少於三次）。

與SORT相比，在每一次中，DeepSORT先後進行了兩次匹配。第一次匹配是級聯匹配，針對的是confirmed狀態的Tracker。每個Tracker都保存了被丟失的次數，級聯匹配會優先匹配丟失次數少的Tracker。匹配的Tracker會利用卡爾曼濾波更新狀態。未匹配的Tracker和Detection會進行第二次IOU匹配，匹配的Tracker會利用卡爾曼濾波更新狀態，未匹配的Detection會初始化為unconfirmed的Tracker；而未匹配的Tracker中，unconfirmed的會直接刪除，confirmed的會查看其“連續丟失次數”，超過max_age次直接刪除，否則繼續保留到下一次。IOU匹配與SORT算法類似，下面主要研究級聯匹配流程。

各狀態變量與SORT算法比較相似。




Reference: 
目標跟蹤(object tracking)基礎算法總結 - 霁澂的文章 - 知乎
https://zhuanlan.zhihu.com/p/628015159
