
$ conda create -n mmseg_env1 python=3.9

$ conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia

$ pip install -U openmim

$ mim install mmengine

$ mim install "mmcv>=2.0.0"

<CASE A>

$ git clone -b main [https://github.com/open-mmlab/mmsegmentation.git](https://github.com/open-mmlab/mmsegmentation.git)

$ cd mmsegmentation

$ pip install -e .

<CASE B>

pip install mmsegmentation

Ref: [https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/get_started.md#installation](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/get_started.md#installation)