
檢查 

---------------------------------------------------

./ngc-cli/ngc --version                   檢查 NGC CLI

docker --version                              檢查 docker

sudo docker run hello-world         檢查 docker運行

sudo nvidia-container-cli info       檢查 NVIDIA Container Cli

 ping [nvcr.io](https://www.google.com/url?q=http://nvcr.io&sa=D&source=calendar&usd=2&usg=AOvVaw1PJuq35CWXHXjH6TZi1uch)                                      檢查nvcr                                                           

檢查NVIDIA Container Toolkit

sudo docker pull nvidia/cuda:12.5.0-base-ubuntu20.04

sudo docker run --rm --gpus all nvidia/cuda:12.5.0-base-ubuntu20.04 nvidia-smi

sudo systemctl status docker 檢查 Docker 守護程序狀態

From <[https://calendar.google.com/calendar/u/0/r](https://calendar.google.com/calendar/u/0/r)>

Step1: 使用DINOv2 docker file based on Nvidia Docker iamge

===================================================================

# Use the NVIDIA CUDA base image

FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# Set environment variables to avoid interactive installations

ENV DEBIAN_FRONTEND=noninteractive

# Update and install basic utilities

RUN apt-get update && apt-get install -y --no-install-recommends \

    build-essential \

    cmake \

    git \

    curl \

    vim \

    wget \

    ca-certificates \

    libjpeg-dev \

    libpng-dev \

    && apt-get clean \

    && rm -rf /var/lib/apt/lists/*

# Install Miniconda

ENV CONDA_DIR=/opt/conda

RUN wget --quiet [https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh) -O /tmp/miniconda.sh && \

    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \

    rm /tmp/miniconda.sh

ENV PATH=$CONDA_DIR/bin:$PATH

# Create a new conda environment and install Python dependencies

RUN conda create -n dinov2 python=3.9 -y

RUN /bin/bash -c "source activate dinov2 && \

conda install -y pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 xformers::xformers conda-forge::omegaconf iopath fvcore pip -c pytorch -c nvidia -c xformers -c conda-forge"

RUN pip install git+https://github.com/facebookincubator/submitit ftfy regex -U openmim mmcv==1.5.0 mmsegmentation==0.27.0

# Set up working directory

WORKDIR /workspace

# Copy the project files

COPY . /workspace

# Expose ports if necessary

EXPOSE 8888

# Command to run your application or start a Jupyter Notebook, etc.

CMD ["bash"]

優化版本

===================================================================

# Use the NVIDIA CUDA base image

FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

# Set environment variables to avoid interactive installations

ENV DEBIAN_FRONTEND=noninteractive

# Update and install basic utilities

RUN apt-get update && apt-get install -y --no-install-recommends \

    build-essential \

    cmake \

    git \

    curl \

    wget \

    ca-certificates \

    libjpeg-dev \

    libpng-dev \

    && apt-get clean \

    && rm -rf /var/lib/apt/lists/*

# Install Miniconda

ENV CONDA_DIR=/opt/conda

RUN wget --quiet [https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh) -O /tmp/miniconda.sh && \

    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \

    rm /tmp/miniconda.sh

ENV PATH=$CONDA_DIR/bin:$PATH

# Create a new conda environment and install Python dependencies

RUN conda install -n base -c conda-forge mamba && \

    mamba create -n dinov2 python=3.9 -y && \

    /bin/bash -c "source activate dinov2 && \

    mamba install -y pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 xformers::xformers conda-forge::omegaconf iopath fvcore pip -c pytorch -c nvidia -c xformers -c conda-forge" && \

    pip install git+https://github.com/facebookincubator/submitit ftfy regex -U openmim mmcv==1.5.0 mmsegmentation==0.27.0"

WORKDIR /workspace

# Copy the project files

COPY . /workspace

# Expose ports if necessary

EXPOSE 8888

# Command to run your application or start a Jupyter Notebook, etc.

CMD ["bash"]

===================================================================

Step2: 在build Docker container之前必須安裝NVIDIA Container Toolkit才能使用GPU:

$ distribution=$(. /etc/os-release;echo$ID$VERSION_ID)

$ curl -s -L [https://nvidia.github.io/nvidia-docker/gpgkey](https://nvidia.github.io/nvidia-docker/gpgkey) | sudo apt-key add -

$ curl -s -L [https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list](https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list) | sudo tee/etc/apt/sources.list.d/nvidia-docker.list

$ sudo apt-get update

$ sudo apt-get install -y nvidia-docker2

$ sudo systemctl restart docker

===================================================================

Step3: 建立Docker image, 從Docker image建立Docker container

sudo docker build -t dinov2_denv .

sudo docker run --gpus all -itd --name dinov2_denv2_container -p 8080:80 -v /home/a3146654/dinov2:/dinov2 dinov2_denv2 /bin/bash

===================================================================

測試在Docker container是否可用GPU (optional)

$ docker exec -it dinov2_denv2_container /bin/bash

$ nvidia-smi

===================================================================

下載DINOv2到資料夾

step1: 如果在dinov2_docker folder 換到dinov2 folder then git clone

Step2: git clone [https://github.com/facebookresearch/dinov2.git](https://github.com/facebookresearch/dinov2.git)

===================================================================

從Docker image build docker container

sudo docker run --gpus all -itd --name dinov2_denv2_container -p 8080:80 -v /home/a3146654/dinov2:/dinov2 dinov2_denv2 /bin/bash