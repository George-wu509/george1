

FCOS是'**F**ully **C**onvolutional **O**ne-**S**tage'的缩写。顾名思义，它是一种**全卷积**、**一阶段**目标检测算法。嗯？[全卷积](https://zhida.zhihu.com/search?content_id=173760074&content_type=Article&match_order=2&q=%E5%85%A8%E5%8D%B7%E7%A7%AF&zhida_source=entity)？是否不禁联想到隔壁语义分割的大咖——FCN？对的！兄弟，不用怀疑，FCOS就是借鉴了FCN**逐像素预测**的范式，是一种**'[per-pixel](https://zhida.zhihu.com/search?content_id=173760074&content_type=Article&match_order=1&q=per-pixel&zhida_source=entity) prediction fashion'**。

“一阶段”代表着它没有RPN，无需像[两阶段算法](https://zhida.zhihu.com/search?content_id=173760074&content_type=Article&match_order=1&q=%E4%B8%A4%E9%98%B6%E6%AE%B5%E7%AE%97%E6%B3%95&zhida_source=entity)那样先产生目标候选区域，再基于这些区域去预测。因此简单许多、计算量也少。

另外，它还是**anchor-free**哦！这玩起来可轻松太多了！为何？因为没有一系列和anchor相关的超参(anchor数量、大小、长宽比等)呀，而这些超参却极大地影响着检测器的性能(可怜anchor-based的宝宝们)..

当然，在FCOS诞生的那个年代，后处理中还是会存在NMS的身影，不像近来的 _DETR_、_Sparse R-CNN_ 这些年轻人在label assignments上下了番功夫(one ground truth maps only one positive sample)，于是干掉了NMS。


FCOS

![[Pasted image 20240912132920.png]]

![[Pasted image 20250323121406.png]]

FCOS([https://arxiv.org/pdf/1904.01355.pdf](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1904.01355.pdf))可以说是2019年最火的Anchor free目标检测器了。相比于之前的Anchor free目标检测器，FCOS的结构十分简洁，且超过了RetinaNet等Anchor based目标检测器的表现。有了之前复现RetinaNet的经验，现在我只需要复现网络结构、loss和后处理部分就可以构建一个完整的FCOS目标检测器。

COCO和VOC数据集处理部分直接沿用RetinaNet中的实现。

**FCOS (Fully Convolutional One-Stage Object Detection)**

- **Backbone Model:**
    - FCOS 的 backbone model 與 Faster R-CNN 類似，通常採用 ResNet 等常見的 CNN 架構來提取特徵。
    - 因此，您說的 backbone model 類似是正確的。
- **Neck Model:**
    - FCOS 也廣泛使用 FPN（Feature Pyramid Network）作為 neck model。
    - FPN 能夠融合不同層級的特徵，提高模型對不同尺度目標的檢測能力。這點您的敘述是正確的。
- **Head Model:**
    - 這裡是最關鍵的差異點。FCOS 的 head model 確實也有分類和回歸分支，但它與 Faster R-CNN 的根本區別在於：
        - **Anchor-free:** FCOS 是 anchor-free 的目標檢測器，這意味著它不使用預定義的 anchor boxes。
        - **像素級預測:** FCOS 直接預測圖像中每個像素屬於哪個目標類別，以及該像素到目標邊界框的距離。
        - 因此，它不是在anchor上面做分類與回歸，他是在影像的每個像素點上面作分類與回歸的。
    - 所以雖然都是有分類與回歸head，但是運作的模式與faster rcnn有很大的不同。
- **總結:**
    - FCOS 與 Faster R-CNN 在 backbone 和 neck model 上有相似之處，但它們在 head model 的設計上有根本性的差異。
    - FCOS 的 anchor-free 設計使其更加簡潔高效，並且在某些情況下能夠取得更好的性能。

**主要區別：Anchor-free vs. Anchor-based**

- Faster R-CNN 是 anchor-based 的，它依賴於預定義的 anchor boxes 來生成候選區域。
- FCOS則是直接對圖像中每個像素進行預測，避免了 anchor boxes 的複雜性。

因此，總結來說，您的敘述大方向正確，需要強調FCOS是Anchor free,並且在影像的pixel上面進行分類與回歸。



## FPN

FCOS的FPN结构和RetinaNet基本一致，唯一的区别在于产生P6特征图的输入由C5变为P5。这减少了大约0.5GFLOPS的计算量，不过和总FLOPS比起来只是很小的一部分。下面的实现中，为了复用代码，我直接对RetinaNet的FPN类进行了修改，代码实现如下：

```python
class RetinaFPN(nn.Module):
    def __init__(self, C3_inplanes, C4_inplanes, C5_inplanes, planes,use_p5=False):
        super(RetinaFPN, self).__init__()
        self.use_p5=use_p5
        self.P3_1 = nn.Conv2d(C3_inplanes,
                              planes,
                              kernel_size=1,
                              stride=1,
                              padding=0)
        self.P3_2 = nn.Conv2d(planes,
                              planes,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.P4_1 = nn.Conv2d(C4_inplanes,
                              planes,
                              kernel_size=1,
                              stride=1,
                              padding=0)
        self.P4_2 = nn.Conv2d(planes,
                              planes,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.P5_1 = nn.Conv2d(C5_inplanes,
                              planes,
                              kernel_size=1,
                              stride=1,
                              padding=0)
        self.P5_2 = nn.Conv2d(planes,
                              planes,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        if self.use_p5:
            self.P6 = nn.Conv2d(planes,
                                planes,
                                kernel_size=3,
                                stride=2,
                                padding=1)
        else:
            self.P6 = nn.Conv2d(C5_inplanes,
                                planes,
                                kernel_size=3,
                                stride=2,
                                padding=1)            

        self.P7 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1))

    def forward(self, inputs):
        [C3, C4, C5] = inputs

        P5 = self.P5_1(C5)
        P4 = self.P4_1(C4)
        P4 = F.interpolate(P5, size=(P4.shape[2], P4.shape[3]),
                           mode='nearest') + P4
        P3 = self.P3_1(C3)
        P3 = F.interpolate(P4, size=(P3.shape[2], P3.shape[3]),
                           mode='nearest') + P3

        P5 = self.P5_2(P5)
        P4 = self.P4_2(P4)
        P3 = self.P3_2(P3)

        if self.use_p5:        
            P6 = self.P6(P5)
        else:
            P6 = self.P6(C5)    

        del C3, C4, C5

        P7 = self.P7(P6)

        return [P3, P4, P5, P6, P7]
```

如上代码所示，当use_p5=False时使用的就是RetinaNet的FPN；当use_p5=True时使用的就是FCOS的FPN。

## heads

FPN也包括两个heads。即分类heads和回归heads。对于centerness heads，我参照FCOS论文中后续的改进方案，将centerness heads与回归heads共用。

分类heads代码实现如下：

```python
class FCOSClsHead(nn.Module):
    def __init__(self, inplanes, num_classes, num_layers=4, prior=0.01):
        super(FCOSClsHead, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.Conv2d(inplanes,
                          inplanes,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(inplanes,
                      num_classes,
                      kernel_size=3,
                      stride=1,
                      padding=1))
        self.cls_head = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

        prior = prior
        b = -math.log((1 - prior) / prior)
        self.cls_head[-1].bias.data.fill_(b)

    def forward(self, x):
        x = self.cls_head(x)

        return x
```

回归heads代码实现如下：

```python
class FCOSRegCenterHead(nn.Module):
    def __init__(self, inplanes, num_layers=4):
        super(FCOSRegCenterHead, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.Conv2d(inplanes,
                          inplanes,
                          kernel_size=3,
                          stride=1,
                          padding=1))
            layers.append(nn.ReLU(inplace=True))
        self.reg_head = nn.Sequential(*layers)
        self.reg_out = nn.Conv2d(inplanes,
                                 4,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.center_out = nn.Conv2d(inplanes,
                                    1,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

    def forward(self, x):
        x = self.reg_head(x)
        reg_output = self.reg_out(x)
        center_output = self.center_out(x)

        return reg_output, center_output
```

## points

FCOS并没有显式的Anchor，但是在生成loss ground truth和inference得到预测坐标时仍然需要知道feature map上各点映射在原图上的坐标。  
生成points代码实现如下：

```python
class FCOSPositions(nn.Module):
    def __init__(self, strides):
        super(FCOSPositions, self).__init__()
        self.strides = strides

    def forward(self, batch_size, fpn_feature_sizes):
        """
        generate batch positions
        """
        device = fpn_feature_sizes.device

        one_sample_positions = []
        for stride, fpn_feature_size in zip(self.strides, fpn_feature_sizes):
            featrue_positions = self.generate_positions_on_feature_map(
                fpn_feature_size, stride)
            featrue_positions = featrue_positions.to(device)
            one_sample_positions.append(featrue_positions)

        batch_positions = []
        for per_level_featrue_positions in one_sample_positions:
            per_level_featrue_positions = per_level_featrue_positions.unsqueeze(
                0).repeat(batch_size, 1, 1, 1)
            batch_positions.append(per_level_featrue_positions)

        # if input size:[B,3,640,640]
        # batch_positions shape:[[B, 80, 80, 2],[B, 40, 40, 2],[B, 20, 20, 2],[B, 10, 10, 2],[B, 5, 5, 2]]
        # per position format:[x_center,y_center]
        return batch_positions

    def generate_positions_on_feature_map(self, feature_map_size, stride):
        """
        generate all positions on a feature map
        """

        # shifts_x shape:[w],shifts_x shape:[h]
        shifts_x = (torch.arange(0, feature_map_size[0]) + 0.5) * stride
        shifts_y = (torch.arange(0, feature_map_size[1]) + 0.5) * stride

        # feature_map_positions shape:[w,h,2] -> [h,w,2] -> [h*w,2]
        feature_map_positions = torch.tensor([[[shift_x, shift_y]
                                               for shift_y in shifts_y]
                                              for shift_x in shifts_x
                                              ]).permute(1, 0, 2).contiguous()

        # feature_map_positions format: [point_nums,2],2:[x_center,y_center]
        return feature_map_positions
```

## 网络结构

网络结构代码实现如下：

```python
import os
import sys
import math
import numpy as np

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from public.path import pretrained_models_path

from public.detection.models.backbone import ResNetBackbone
from public.detection.models.fpn import RetinaFPN
from public.detection.models.head import FCOSClsHead, FCOSRegCenterHead
from public.detection.models.anchor import FCOSPositions

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'resnet18_fcos',
    'resnet34_fcos',
    'resnet50_fcos',
    'resnet101_fcos',
    'resnet152_fcos',
]

model_urls = {
    'resnet18_fcos':
    'empty',
    'resnet34_fcos':
    'empty',
    'resnet50_fcos':
    '{}/detection_models/resnet50_fcos_coco_resize667_mAP0.286.pth'.format(
        pretrained_models_path),
    'resnet101_fcos':
    'empty',
    'resnet152_fcos':
    'empty',
}


# assert input annotations are[x_min,y_min,x_max,y_max]
class FCOS(nn.Module):
    def __init__(self, resnet_type, num_classes=80, planes=256):
        super(FCOS, self).__init__()
        self.backbone = ResNetBackbone(resnet_type=resnet_type)
        expand_ratio = {
            "resnet18": 1,
            "resnet34": 1,
            "resnet50": 4,
            "resnet101": 4,
            "resnet152": 4
        }
        C3_inplanes, C4_inplanes, C5_inplanes = int(
            128 * expand_ratio[resnet_type]), int(
                256 * expand_ratio[resnet_type]), int(
                    512 * expand_ratio[resnet_type])
        self.fpn = RetinaFPN(C3_inplanes,
                             C4_inplanes,
                             C5_inplanes,
                             planes,
                             use_p5=True)

        self.num_classes = num_classes
        self.planes = planes

        self.cls_head = FCOSClsHead(self.planes,
                                    self.num_classes,
                                    num_layers=4,
                                    prior=0.01)
        self.regcenter_head = FCOSRegCenterHead(self.planes, num_layers=4)

        self.strides = torch.tensor([8, 16, 32, 64, 128], dtype=torch.float)
        self.positions = FCOSPositions(self.strides)

        self.scales = nn.Parameter(torch.FloatTensor([1., 1., 1., 1., 1.]))

    def forward(self, inputs):
        self.batch_size, _, _, _ = inputs.shape
        device = inputs.device
        [C3, C4, C5] = self.backbone(inputs)

        del inputs

        features = self.fpn([C3, C4, C5])

        del C3, C4, C5

        self.fpn_feature_sizes = []
        cls_heads, reg_heads, center_heads = [], [], []
        for feature, scale in zip(features, self.scales):
            self.fpn_feature_sizes.append([feature.shape[3], feature.shape[2]])
            cls_outs = self.cls_head(feature)
            # [N,num_classes,H,W] -> [N,H,W,num_classes]
            cls_outs = cls_outs.permute(0, 2, 3, 1).contiguous()
            cls_heads.append(cls_outs)

            reg_outs, center_outs = self.regcenter_head(feature)
            # [N,4,H,W] -> [N,H,W,4]
            reg_outs = reg_outs.permute(0, 2, 3, 1).contiguous()
            reg_outs = reg_outs * scale
            reg_heads.append(reg_outs)
            # [N,1,H,W] -> [N,H,W,1]
            center_outs = center_outs.permute(0, 2, 3, 1).contiguous()
            center_heads.append(center_outs)

        del features

        self.fpn_feature_sizes = torch.tensor(
            self.fpn_feature_sizes).to(device)

        batch_positions = self.positions(self.batch_size,
                                         self.fpn_feature_sizes)

        # if input size:[B,3,640,640]
        # features shape:[[B, 256, 80, 80],[B, 256, 40, 40],[B, 256, 20, 20],[B, 256, 10, 10],[B, 256, 5, 5]]
        # cls_heads shape:[[B, 80, 80, 80],[B, 40, 40, 80],[B, 20, 20, 80],[B, 10, 10, 80],[B, 5, 5, 80]]
        # reg_heads shape:[[B, 80, 80, 4],[B, 40, 40, 4],[B, 20, 20, 4],[B, 10, 10, 4],[B, 5, 5, 4]]
        # center_heads shape:[[B, 80, 80, 1],[B, 40, 40, 1],[B, 20, 20, 1],[B, 10, 10, 1],[B, 5, 5, 1]]
        # batch_positions shape:[[B, 80, 80, 2],[B, 40, 40, 2],[B, 20, 20, 2],[B, 10, 10, 2],[B, 5, 5, 2]]

        return cls_heads, reg_heads, center_heads, batch_positions


def _fcos(arch, pretrained, progress, **kwargs):
    model = FCOS(arch, **kwargs)
    # only load state_dict()
    if pretrained:
        pretrained_models = torch.load(model_urls[arch + "_fcos"],
                                       map_location=torch.device('cpu'))
        # del pretrained_models['cls_head.cls_head.8.weight']
        # del pretrained_models['cls_head.cls_head.8.bias']

        # only load state_dict()
        model.load_state_dict(pretrained_models, strict=False)

    return model


def resnet18_fcos(pretrained=False, progress=True, **kwargs):
    return _fcos('resnet18', pretrained, progress, **kwargs)


def resnet34_fcos(pretrained=False, progress=True, **kwargs):
    return _fcos('resnet34', pretrained, progress, **kwargs)


def resnet50_fcos(pretrained=False, progress=True, **kwargs):
    return _fcos('resnet50', pretrained, progress, **kwargs)


def resnet101_fcos(pretrained=False, progress=True, **kwargs):
    return _fcos('resnet101', pretrained, progress, **kwargs)


def resnet152_fcos(pretrained=False, progress=True, **kwargs):
    return _fcos('resnet152', pretrained, progress, **kwargs)


if __name__ == '__main__':
    net = FCOS(resnet_type="resnet50")
    image_h, image_w = 600, 600
    cls_heads, reg_heads, center_heads, batch_positions = net(
        torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    annotations = torch.FloatTensor([[[113, 120, 183, 255, 5],
                                      [13, 45, 175, 210, 2]],
                                     [[11, 18, 223, 225, 1],
                                      [-1, -1, -1, -1, -1]],
                                     [[-1, -1, -1, -1, -1],
                                      [-1, -1, -1, -1, -1]]])

    print("1111", cls_heads[0].shape, reg_heads[0].shape,
          center_heads[0].shape, batch_positions[0].shape)
```

这里要特别说明一下self.scales的作用。FCOS的回归标签是某个在目标框内的点到左、上、右、下的距离再用log平滑后的值，且在分配标签时不同大小的目标会被分配到不同FPN level的分支上，大尺度的目标会被分配到更高尺度FPN level分支上。尽管标签已经用log平滑了，但是在更高尺度的FPN level分支上分配的标签数值仍然要比低尺度FPN level分配的标签数值要大。如果不创建一个可学习的变量self.scales乘以该尺度FPN level的回归分支，那么会造成不同FPN level的回归分支学习不同步(因为更高尺度的FPN level回归分支回归的差值更大)，最终模型的性能表现会明显下降。

同RetinaNet相比，由于FCOS的head不那么厚重(FCOS可以看成每个位置上Anchor数为1的特殊形式，这个后面再详细分析)，在相同输入分辨率的情况下，FCOS的FLOPS要比RetinaNet减少10%左右。




### FCOS的基本結構與特點是否正確？

你的描述提到：

1. FCOS是一個目標檢測（object detection）模型。
2. 結構上與Faster R-CNN類似，具有相同的backbone和neck，且使用FPN（Feature Pyramid Network）。
3. FCOS的head model是anchor-free，直接對每個像素點進行分類與回歸。

這些描述基本上是正確的，但讓我們更精確地分析一下：

#### 1. **FCOS是目標檢測模型**

是的，FCOS（Fully Convolutional One-Stage Object Detection）是一個單階段（one-stage）的目標檢測模型，於2019年提出，旨在解決傳統基於anchor的方法（如Faster R-CNN）的複雜性和超參數調整問題。

#### 2. **與Faster R-CNN的結構相似性**

- **Backbone**：FCOS和Faster R-CNN都可以使用相同的backbone（如ResNet或ResNeXt）來提取特徵。
- **Neck**：兩者都使用FPN來生成多尺度特徵圖，以處理不同大小的目標。這一點是相同的。
- **差異點**：Faster R-CNN是兩階段（two-stage）模型，先通過RPN（Region Proposal Network）生成候選區域（proposals），再進行分類和邊框回歸；而FCOS是單階段模型，直接在特徵圖的每個像素點上進行預測，沒有候選區域生成步驟。

#### 3. **FCOS的head model是anchor-free，直接預測每個像素點**

- **Anchor-free**：這是FCOS的核心特徵。傳統方法（如Faster R-CNN）依賴於預定義的anchor box來進行目標檢測，而FCOS完全拋棄了anchor的概念，直接將特徵圖上的每個像素點視為一個潛在的檢測點。
- **像素級預測**：FCOS的head model會對每個像素點預測：
    - **分類（Classification）**：該點是否屬於某個類別（前景類別或背景）。
    - **回歸（Regression）**：從該點到目標邊框四個邊的距離（通常表示為左、上、右、下四個值）。
    - **Centerness**：額外引入一個分支，預測該點是否位於目標的中心區域，用來抑制邊緣點的低質量預測。

因此，你的描述是正確的，但需要補充的是，FCOS不僅僅是分類與回歸，還包括centerness分支，這是它與傳統模型的一個關鍵區別。

---

### FCOS與Faster R-CNN的Head Model詳細比較

#### 1. **流程**

- **Faster R-CNN的Head Model**：
    1. **RPN階段**：首先通過RPN生成候選區域（proposals），每個proposal基於預定義的anchor box生成。
    2. **ROI Pooling/Align**：從特徵圖中提取每個proposal的固定大小特徵（例如7x7）。
    3. **Head分支**：
        - **分類分支**：預測每個proposal屬於某個類別的概率（包括背景類）。
        - **回歸分支**：對每個proposal的bounding box進行微調，預測四個偏移量（Δx, Δy, Δw, Δh）。
    4. **後處理**：應用NMS（Non-Maximum Suppression）來去除重疊框。
- **FCOS的Head Model**：
    1. **無RPN**：直接在FPN生成的特徵圖上進行預測，沒有生成proposal的步驟。
    2. **像素級預測**：
        - **分類分支**：對特徵圖的每個像素點預測類別概率。
        - **回歸分支**：直接預測該點到目標邊框四邊的距離（l, t, r, b）。
        - **Centerness分支**：預測該點是否位於目標中心，用於過濾低質量預測。
    3. **後處理**：同樣使用NMS，但基於像素級預測結果。

#### 2. **輸出格式**

- **Faster R-CNN的輸出**：
    - 每個proposal生成一個bounding box，格式為：
        - **四個座標偏移量**：(Δx, Δy, Δw, Δh)，相對於anchor box的調整。
        - **類別信心分數**：每個類別的confidence score（包括背景），通常是一個softmax概率。
    - 最終輸出是NMS後的bounding box列表，每個框包含：
        - 座標：(x_min, y_min, x_max, y_max)。
        - 類別標籤和對應的confidence score。
- **FCOS的輸出**：
    - 每個像素點生成一個bounding box，格式為：
        - **四個距離值**：(l, t, r, b)，表示從該點到目標邊框左、上、右、下的距離。
        - **類別信心分數**：每個類別的得分（通常是sigmoid激活，因為是像素級獨立預測）。
        - **Centerness得分**：一個0到1之間的值，表示該點的中心性。
    - 最終輸出是NMS後的bounding box列表，每個框包含：
        - 座標：通過(l, t, r, b)轉換為(x_min, y_min, x_max, y_max)。
        - 類別標籤和對應的confidence score（通常乘以centerness作為最終得分）。

#### 3. **主要區別**

|特性|Faster R-CNN|FCOS|
|---|---|---|
|**Anchor使用**|依賴anchor box|Anchor-free|
|**檢測方式**|兩階段：RPN + Head|單階段：直接像素級預測|
|**回歸目標**|偏移量(Δx, Δy, Δw, Δh)|距離值(l, t, r, b)|
|**額外分支**|無|Centerness分支|
|**計算複雜度**|較高（因RPN和ROI操作）|較低（全卷積結構）|
|**靈活性**|對anchor設計敏感|無需調整anchor參數|

---

### 結論

你的敘述是正確的：FCOS和Faster R-CNN在backbone和neck（FPN）上相似，但FCOS的head model是anchor-free，直接對每個像素點進行分類和回歸，並引入centerness來提升檢測質量。相比之下，Faster R-CNN依賴anchor和兩階段流程，輸出格式和預測目標也有所不同。FCOS的設計更簡單高效，而Faster R-CNN則在某些場景下可能更精確，但需要更多超參數調優。











Github:
https://github.com/zgcr/SimpleAICV_pytorch_training_examples

【庖丁解牛】从零实现FCOS（一）：FPN、heads、points、网络结构 - 每天进步一点点的文章 - 知乎
https://zhuanlan.zhihu.com/p/159710460