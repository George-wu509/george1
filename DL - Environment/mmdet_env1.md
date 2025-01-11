
conda create -n mmdet_env1 python=3.9

$ conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia

$ pip install -U openmim

$ mim install mmengine

$ mim install "mmcv>=2.0.0"

<CASE A>

$ git clone [https://github.com/open-mmlab/mmdetection.git](https://github.com/open-mmlab/mmdetection.git)

$ cd mmdetection

$ pip install -e .

<CASE B>

$ mim install mmdet

Ref: [https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/get_started.md#installation](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/get_started.md#installation)

@ 下載MMDet 的instance segmentation model:

Mask RCNN model:

$ wget --no-check-certificate -c [https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco_20210526_120447-c376f129.pth](https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco_20210526_120447-c376f129.pth)

YOLO model:

$ wget --no-check-certificate -c [https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_x101_dcn_fpn_3x_coco/solov2_x101_dcn_fpn_3x_coco_20220513_214337-aef41095.pth](https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_x101_dcn_fpn_3x_coco/solov2_x101_dcn_fpn_3x_coco_20220513_214337-aef41095.pth)