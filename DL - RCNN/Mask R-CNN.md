
![[Pasted image 20240912133211.png]]
![[Pasted image 20240912133214.png]]
Reference:

[1] Mask R-CNN 网络结构详解
https://blog.csdn.net/qq_37541097/article/details/123754766

[2] Mask R-CNN网络结构理解all
https://blog.csdn.net/WangNning2000/article/details/109629665

[3] Mask R-CNN 网络结构
https://www.cnblogs.com/wujianming-110117/p/15183270.html




### MaskRCNN**网络结构**

MaskRCNN作为FasterRCNN的扩展，产生RoI的RPN网络和FasterRCNN网络。

结构：ResNet101+FPN

代码：TensorFlow+ Keras（Python）

代码中将Resnet101网络，分成5个stage，记为[C1,C2,C3,C4,C5]；这里的5个阶段分别对应着5中不同尺度的feature map输出，用来建立FPN网络的特征金字塔（feature pyramid）.

先通过两张MaskRCNN整体网络结构图，再附带一张绘制了stage1和stage2的层次结构图（stage3到stage5的结构层次比较多，未绘制），整体了解下MaskRCNN网络。

 MaskRCNN网络结构泛化图：

 ![](https://img2020.cnblogs.com/blog/1251718/202108/1251718-20210825061506151-1863747491.png)

 MaskRCNN网络结构细化图：

 ![](https://img2020.cnblogs.com/blog/1251718/202108/1251718-20210825061441851-1481605534.png)

 stage1和stage2层次结构图：

 ![](https://img2020.cnblogs.com/blog/1251718/202108/1251718-20210825061400101-1079199895.png)

**Mask-RCNN** **技术要点**

● 技术要点1 - 强化的基础网络

     通过 ResNeXt-101+FPN 用作特征提取网络，达到 state-of-the-art 的效果。

● 技术要点2 - ROIAlign

     采用 ROIAlign 替代 RoiPooling（改进池化操作）。引入了一个插值过程，先通过双线性插值到14*14，再 pooling到7*7，很大程度上解决了仅通过 Pooling 直接采样带来的 Misalignment 对齐问题。

     PS： 虽然 Misalignment 在分类问题上影响并不大，但在 Pixel 级别的 Mask 上会存在较大误差。

     后面我们把结果对比贴出来（Table2 c & d），能够看到 ROIAlign 带来较大的改进，可以看到，Stride 越大改进越明显。

● 技术要点3 - Loss Function

     每个 ROIAlign 对应 K * m^2 维度的输出。K 对应类别个数，即输出 K 个mask，m对应池化分辨率（7*7）。Loss函数定义：

    Lmask(Cls_k) = Sigmoid (Cls_k)，平均二值交叉熵 （average binary cross-entropy）Loss，通过逐像素的 Sigmoid 计算得到。

     Why K个mask？通过对每个 Class 对应一个 Mask 可以有效避免类间竞争（其它Class 不贡献 Loss ）。

**ROIpooling**

ROI pooling就不多解释了，直接通过一个例子来形象理解。假设有一个8x8大小的feature map，要在这个feature map上得到ROI，并且进行ROI pooling到2x2大小的输出。

 ![](https://img2020.cnblogs.com/blog/1251718/202108/1251718-20210825061234715-14071067.png)

 假设ROI的bounding box为 ![](https://img2020.cnblogs.com/blog/1251718/202108/1251718-20210825061323522-652746658.png)

如图：

 ![](https://img2020.cnblogs.com/blog/1251718/202108/1251718-20210825061219344-785115287.png)

 划分为2x2的网格，因为ROI的长宽除以2是不能整除的，所以会出现每个格子大小不一样的情况。

 ![](https://img2020.cnblogs.com/blog/1251718/202108/1251718-20210825061145903-1719158983.png)

 进行max pooling的最终2x2的输出为：

 ![](https://img2020.cnblogs.com/blog/1251718/202108/1251718-20210825061105357-1232715566.png)

 ROI Align

在Faster RCNN中，有两次整数化的过程：

1. region proposal的xywh通常是小数，为了方便操作会把它整数化。
2. 将整数化后的边界区域平均分割成 k x k 个单元，对每一个单元的边界进行整数化。

两次整数化的过程如下图所示：

 ![](https://img2020.cnblogs.com/blog/1251718/202108/1251718-20210825061043224-1607508415.png)

 事实上，经过上述两次整数化，此时的候选框已经和最开始回归出来的位置有一定的偏差，这个偏差会影响检测或者分割的准确度。在论文里，总结为“不匹配问题”（misalignment）。

为了解决这个问题，ROI Align方法取消整数化操作，保留了小数，使用以上介绍的双线性插值的方法获得坐标为浮点数的像素点上的图像数值。但在实际操作中，ROI Align并不是简单地补充出候选区域边界上的坐标点，然后进行池化，而是重新进行设计。

下面通过一个例子来讲解ROI Align操作。如下图所示，虚线部分表示feature map，实线表示ROI，这里将ROI切分成2x2的单元格。如果采样点数是4，首先将每个单元格子均分成四个小方格（如红色线所示），每个小方格中心就是采样点。这些采样点的坐标通常是浮点数，所以需要对采样点像素进行双线性插值（如四个箭头所示），就可以得到该像素点的值了。对每个单元格内的四个采样点进行maxpooling，就可以得到最终的ROIAlign的结果。

 ![](https://img2020.cnblogs.com/blog/1251718/202108/1251718-20210825061006484-49145501.png)




### Mask R-CNN 中Anchor 的產生流程、控制參數、與 RPN、ROIAlign、Region Proposal 的關係

# 一、概覽：Mask R-CNN 是什麼？

**Mask R-CNN 是 Faster R-CNN 的擴展版**，在物件偵測（bounding box）與分類之外，**額外預測每個物件的像素級 segmentation mask（語義遮罩）**。

主要組件包含：

1. **Backbone（ResNet + FPN）**
    
2. **RPN（Region Proposal Network）**
    
3. **RoIAlign（Region Feature Extraction）**
    
4. **Detection Head（分類與框）+ Mask Head（產生 segmentation mask）**
    

---

# ✅ 二、Anchor 的產生流程（在 RPN 階段）

## 🔹 Anchor 是在 RPN 裡產生的，用來尋找候選物件區域（Region Proposals）

### 🔧 Anchor 的控制參數（以 PyTorch `torchvision.models.detection` 為例）：

|參數名稱|功能描述|
|---|---|
|`anchor_sizes`|每層 feature map 所產生 anchor 的基礎大小（例如 `[32], [64], [128], [256], [512]`）|
|`aspect_ratios`|每個 anchor 的長寬比（例如 `[0.5, 1.0, 2.0]`）|
|`strides`|每層 feature map 的 stride，例如特徵圖是原圖的 1/8、1/16|
|`pre_nms_top_n`|預選取的 proposal 數量（NMS 前）|
|`post_nms_top_n`|NMS 後保留的 proposal 數量|
|`nms_thresh`|NMS 的 IoU 門檻，用來過濾重疊的 proposal|
|`fg_iou_thresh / bg_iou_thresh`|Anchor 被視為正樣本或負樣本的 IoU 門檻（常見為 0.7 / 0.3）|

---

## ✅ Anchor 產生步驟：

以 ResNet + FPN 的一個輸入影像為例：

1. **FPN 會輸出多個層級的特徵圖**（例如 P2, P3, P4, P5, P6），尺寸分別是原圖的 1/4、1/8、1/16、1/32、1/64。
    
2. 在每個 pixel 的位置，**根據 `anchor_sizes` 和 `aspect_ratios` 建立多個 anchor**。
    

例如：

- 在特徵圖 P4（stride=16）上，每個 pixel 對應到原圖一塊 16x16 區域。
    
- 假設設定 `anchor_sizes = [128]`，`aspect_ratios = [0.5, 1.0, 2.0]`，則每個 pixel 上會有 3 個 anchor：
    
    - 1282/0.5×1282×0.5=181×91\sqrt{128^2 / 0.5} \times \sqrt{128^2 \times 0.5} = 181 \times 911282/0.5​×1282×0.5​=181×91
        
    - 128×128128 \times 128128×128
        
    - 91×18191 \times 18191×181
        

每個位置都會產生這些 anchor，最後總共會產生數萬個 anchor。

---

# ✅ 三、Anchor 跟 RPN、Region Proposal 的關係

### 📌 **RPN（Region Proposal Network）** 是一個小型 CNN，用來評估每個 anchor 的物件性：

|功能|輸出|
|---|---|
|分類（cls）|每個 anchor 是不是物件（二元分類）|
|回歸（reg）|Anchor 框微調修正值（dx, dy, dw, dh）|

> ✅ 最後由 RPN 將 anchor 調整為更接近的 **Proposal（候選框）**，然後進行 **NMS（非極大抑制）**，保留最可能是物件的 Top-N 框。

這些 Proposal 就是給後續 RoIAlign + Detection Head 的輸入。

---

# ✅ 四、Anchor 與 RoIAlign / RoIPooling 的關係

### 🔁 流程關係圖：

text

複製編輯

`輸入圖像 → Backbone + FPN → 特徵圖 → Anchor + RPN → Proposal → RoIAlign → Head 分支`

|模組|作用|是否與 Anchor 有關|
|---|---|---|
|**Anchor**|為每個位置生成固定框|✅ 產生位置與比例|
|**RPN**|根據 anchor 評估是不是物件|✅ 依賴 anchor|
|**Proposal**|經過 NMS 的候選框|✅ 修正 anchor|
|**RoIAlign**|將 Proposal 區域的特徵對齊並抽取|✅ 用 Proposal 作輸入|
|**Detection Head / Mask Head**|最終分類與分割|❌ 與 anchor 無直接關係（輸入來自 RoIAlign）|

📝 **注意：RoIAlign 使用的是 Proposal，不是 anchor 本身。anchor 只是 RPN 的中介工具。**

---

# ✅ 五、Mask R-CNN 的訓練流程（含 Anchor 流程）

### 🔹 Step 1：Anchor Matching

- 每張圖片根據設定產生 10k~100k anchor。
    
- 與 Ground Truth 做 IoU 比較（通常 Positive ≥ 0.7，Negative ≤ 0.3）。
    

### 🔹 Step 2：RPN 訓練 Loss

- **分類損失（Object / Non-object）**
    
- **框回歸損失（對正樣本 anchor 回歸 dx, dy, dw, dh）**
    

### 🔹 Step 3：Proposal 過 NMS

- 將高 score anchor 微調後，選出前 N 個作為 Proposal。
    
- 這些 Proposal 提供給 RoIAlign。
    

### 🔹 Step 4：RoIAlign 特徵提取

- 每個 Proposal 映射回特徵圖，透過 RoIAlign 抽取對齊後的固定大小特徵（如 7x7）。
    

### 🔹 Step 5：Head 分支訓練

- **分類頭（分類物件類別）**
    
- **框回歸頭（再次 refine bbox）**
    
- **Mask head（像素級預測物件遮罩）**
    

### 🔹 Step 6：總 Loss

text

複製編輯

`總 Loss = RPN Loss + Classification Loss + BBox Regression Loss + Mask Loss`

---

## ✅ 補充：常見 anchor 相關參數（PyTorch 預設）

python

複製編輯

`anchor_generator = AnchorGenerator(     sizes=((32,), (64,), (128,), (256,), (512,)),     aspect_ratios=((0.5, 1.0, 2.0),) * 5 )`

這表示：

- 在每層特徵圖上產生不同 base size 的 anchor
    
- 每個位置上有 3 種比例的 anchor → 每層 anchor 種類 = 3 × 特徵圖 pixel 數
    

---

# ✅ 總結表

|名稱|是否依賴 Anchor|功能|
|---|---|---|
|**Anchor**|✅|RPN 使用的基礎參考框，固定大小與比例|
|**RPN**|✅|用 anchor 產生 region proposals|
|**Proposal**|✅|anchor 回歸修正 + NMS 結果|
|**RoIAlign**|✅|從 Proposal 中抽取對齊特徵|
|**Detection Head**|❌|接收 RoI 特徵進行分類與回歸|
|**Mask Head**|❌|預測 segmentation mask|