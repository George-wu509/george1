ref: 


@@ SAM2

$ conda create -n sam2_env1 python=3.10

$ conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia

到目錄segment-anything-2

$ pip install .

@@ SAM

$ conda create -n sam_env1 python=3.10

$ conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install opencv-python pycocotools matplotlib onnxruntime onnx

到目錄segment-anything

$ pip install .