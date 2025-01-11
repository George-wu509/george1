
Checking List

(0) install Nvidia GPU driver on windows, install CUDA toolkit in wsl2

(1) WSL2 (ubuntu20.04) to D:   <-- WSL1 not support GPU

(2) Miniconda3

(3) Docker

(4) Nvidia container toolkit

(5) net-tools

(6) gcc

(7) cuda11.7   --> check $nvidia-smi, $nvcc -V

(8) python environment

(9) dinov2 package

移除原本安裝的 WSL2

step1. 打開 PowerShell（以系統管理員身份運行），輸入以下命令列出已安裝的 WSL 發行版：

> wsl --list --verbose

step2. 找到你需要移除的發行版名稱，然後運行以下命令卸載該發行版

> wsl --unregister Ubuntu-20.04

step3. 確保所有相關資料夾已被刪除。通常在 C:\Users\你的用戶名

> \AppData\Local\Packages\ 中。

在 D重新安裝 WSL 和 Ubuntu 20.04

step1. 啟用 WSL 功能, 打開 PowerShell（以系統管理員身份運行）

> dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart

> dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

step2. 安裝 WSL 2 內核更新包. 下載並安裝最新的 WSL 2 內核包及安裝Ubuntu 20.04

> wsl --install -d Ubuntu-20.04

Please create a default UNIX user account.

UNIX username/password: a3146654 / rain0601

關閉PowerShell再重開.

step3. 設置 WSL 2 為預設版本

在 PowerShell 中運行以下命令

> wsl --set-default-version 2

step4. 升級WSL kernel

> wsl --update

step5. 如果要安裝其他版本可以檢視

> wsl --list --online

step6. 將Ubuntu 20.04移到D drive

安裝完成後，不要啟動 Linux 發行版。首先需要將其移動到 D drive

在 C drive的用戶資料夾中找到安裝的 Linux 發行版的 AppData 資料夾。路徑類似於 C:\Users\你的用戶名\AppData\Local\Packages\CanonicalGroupLimited...

將該資料移動到D driver的目標位置(D:\WSL\Ubuntu-20.04)

並移除原路徑資料夾(C:\Users\....) 如果無法移除或刪除原資料夾, 重新開powershell再關閉

step7. 創建符號鏈接（符號鏈接讓系統仍然認為 WSL 安裝在原位置）在cmd下執行

> mklink /J "C:\Users\你的用戶名\AppData\Local\Packages\CanonicalGroupLimited..." "D:\WSL\Ubuntu-20.04"

完成移動和鏈接後，從開始菜單啟動你所安裝的 Linux 發行版。此時 WSL 應該已經安裝在 D drive上. 也

可以在powershell執行cmd.exe /c "mklink /J 'CanonicalGroupLimited.Ubuntu20.04LTS_79rhkp1fndgsc' 'D:\WSL\Ubuntu-20.04'"

step8. 在開始菜單啟動你所安裝的 Linux 發行版(Ubuntu-20.04)

step8. 設置DNS解決網路問題

編輯/etc/resolv.conf文件

$ sudo nano /etc/resolv.conf

在文件內應該只有nameserver 172.23.160.1

添加以下行

nameserver 8.8.8.8

nameserver 8.8.4.4

保存文件並退出

wsl2下的ubuntu20.04用conda安裝dinov2的步驟

step1 安裝Miniconda3

$  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 

$ bash Miniconda3-latest-Linux-x86_64.sh

按照提示完成 Miniconda 的安装，然后重新打开终端或运行以下命令以激活 conda

Do you wish to update your shell profile to automatically initialize conda?

This will activate conda on startup and change the command prompt when activated.

If you'd prefer that conda's base environment not be activated on startup,

   run the following command when conda is activated:

conda config --set auto_activate_base false

You can undo this by running `conda init --reverse $SHELL`? [yes|no]

[no] >>> no

$ conda init

step2 創建python environment

$ source ~/.bashrc

$ conda create -n dinov2_env1 python=3.9

$ conda activate dinov2_env1

如果顯示CondaError: Run 'conda init' before 'conda activate'執行

$ conda init

step3 添加 Miniconda 到 PATH(如果找不到conda)

$ nano ~/.bashrc

在文件末尾添加以下行

export PATH="$HOME/miniconda3/bin:$PATH"

保存并关闭文件，然后重新加载 .bashrc

$ source ~/.bashrc

重新開啟ubuntu20.04

验证 conda 是否可用

$ conda --version

step4 安装和配置 conda 环境for dinov2

$ conda create -n dinov2 python=3.9

$ conda activate dinov2

step5 安装 pytorch and xformers

$ conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

$ pip install -U xformers

(conda install xformers::xformers=0.0.18)

$ conda install conda-forge::omegaconf

$ conda install torchmetrics

$ conda install fvcore

$ conda install iopath

$ pip install git+https://github.com/facebookincubator/submitit

(移除package)

$ conda remove -n dinov2 --all

step5 安装 cuml-cu11

$ sudo apt-get update

$ sudo apt-get install --reinstall ca-certificates

$ sudo update-ca-certificates

通过在 pip 命令中添加 --trusted-host 选项来绕过 SSL 证书验证

$ pip install --trusted-host pypi.nvidia.com --extra-index-url [https://pypi.nvidia.com](https://pypi.nvidia.com) cuml-cu11

Step6 安裝mmcv-full==1.5.0

$ nvcc --version

Cuda compilation tools, release 11.3  ---> cuda版本=11.3

(nvidia-smi 的cuda版本是cuda driver版本)

檢查setuptools版本

$ pip list | grep seruptools

$ python -m pip install setuptools==69.5.1

$ sudo apt update

$ sudo apt install g++

$ pip install mmcv-full==1.5.0

在 WSL2 的 Ubuntu 20.04 中安装 Docker

step1 更新包管理器添加 Docker 的官方 GPG 密钥

$ sudo apt update

$ sudo apt install apt-transport-https ca-certificates curl software-properties-common

$ curl -fsSL [https://download.docker.com/linux/ubuntu/gpg](https://download.docker.com/linux/ubuntu/gpg) | sudo apt-key add -

step2 安装 Docker及添加 Docker 仓库並启动 Docker 服务

$ sudo add-apt-repository "deb [arch=amd64] [https://download.docker.com/linux/ubuntu](https://download.docker.com/linux/ubuntu) $(lsb_release -cs) stable"

$ sudo apt update

$ sudo apt install docker-ce

$ sudo service docker start

step3 拉取 Docker 镜像并运行容器並运行 Docker 容器

$ sudo docker pull chjkusters4/dinov2:V4       (this is BIG)

$ sudo docker run -it --name dinov2_container chjkusters4/dinov2:V4   (this is BIG)

如果要設置接口以及掛載路徑

$ sudo docker run -itd --name dinov2_container -p 8080:80 -v /home/a3146654/dinov2:/dinov2 chjkusters4/dinov2:V4 /bin/bash

step4 将用户添加到 Docker

$ sudo usermod -aG docker $USER

重新启动 WSL2 终端

$ groups 验证是否生效

在 WSL2 的 Ubuntu 20.04 中安装Nvidia container toolkit

step1 Set up the package repository and the GPG key

$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

$ curl -s -L [https://nvidia.github.io/nvidia-docker/gpgkey](https://nvidia.github.io/nvidia-docker/gpgkey) | sudo apt-key add -

$ curl -s -L [https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list](https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list) | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

Step2 Install the NVIDIA Container Toolkit:

$ sudo apt-get update

$ sudo apt-get install -y nvidia-docker2

Step3 Restart Docker to apply changes:

$ sudo service docker restart

Step4 Test the NVIDIA runtime with a simple GPU container:

$ sudo docker run --rm --gpus all nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04 nvidia-smi

在 WSL2 的 Ubuntu 20.04 中安装CUDA toolkit

step1 Install  CUDA Toolkit:

sudo apt-get update

sudo apt-get upgrade

wget --no-check-certificate [https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin)

sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600

sudo apt-key adv --fetch-keys [https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub)

sudo add-apt-repository "deb [https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/) /"

sudo apt-get update

sudo apt-get -y install cuda-11-7

step2 setup path

$ nano ~/.bashrc

export PATH=/usr/local/cuda-11.7/bin${PATH:+:${PATH}}

export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

source ~/.bashrc

Step3: test

nvcc --version

Explanation 說明

sudo docker run -itd --name dinov2_container -p 8080:80 -v /home/a3146654/dinov2:/dinov2 chjkusters4/dinov2:V4 /bin/bash

docker run

运行一个新的容器

-itd

（interactive）使容器保持“交互模式”,（tty）：分配一个伪终端，这使你可以与容器交互.（detached）：使容器在后台运行，并返回容器 ID

--name dinov2_container

用于给容器指定一个名称(dinov2_container)

-p 8080:80 参数用于端口映射. 格式是 宿主机端口:容器端口

将宿主机的 8080 端口映射到容器的 80 端口。换句话说，访问宿主机的 [http://localhost:8080](http://localhost:8080) 实际上是在访问容器的 80 端口。

-v /home/a3146654/dinov2:/dinov2

参数用于挂载卷，格式是 宿主机路径:容器路径. 这里表示将宿主机的 /home/a3146654/dinov2 目录挂载到容器的 /dinov2 目录。这样你可以在容器内访问和操作宿主机的 /home/a3146654/dinov2 目录中的文件。

chjkusters4/dinov2:V4

这是要使用的 Docker 镜像的名称和标签。 Docker 会从 Docker Hub 上拉取这个镜像，如果本地没有这个镜像，它会自动下载。

/bin/bash

这是在容器启动后要运行的命令。/bin/bash 启动一个 Bash shell

解決SSL: CERTIFICATE_VERIFY_FAILED]

Solution1:

遇到了此错误SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed

<python code>

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

Solution2:

$ sudo apt-get update

$ sudo apt-get install --reinstall ca-certificates

$ sudo update-ca-certificates

通过在 pip 命令中添加 --trusted-host 选项来绕过 SSL 证书验证

$ pip install --trusted-host pypi.nvidia.com --extra-index-url [https://pypi.nvidia.com](https://pypi.nvidia.com) cuml-cu11

安裝spped2000/dinov2manyproblem docker

[https://hub.docker.com/r/spped2000/dinov2manyproblem/tags](https://hub.docker.com/r/spped2000/dinov2manyproblem/tags)

step1 拉取 Docker 镜像并运行容器並运行 Docker 容器

$ sudo docker pull spped2000/dinov2manyproblem:lastest

step2

sudo docker run -itd --name dinov2many -p 8080:80 -v /home/a3146654/dinov2a:/dinov2 spped2000/dinov2manyproblem:lastest /bin/bash

step3 (use gpu?)

sudo docker run -itd --name dinov2many --gpus all -p 8080:80 -v /home/a3146654/dinov2:/dinov2 spped2000/dinov2manyproblem:lastest /bin/bash

安裝WSL2 網路問題

[https://www.cnblogs.com/SocialistYouth/p/16691035.html](https://www.cnblogs.com/SocialistYouth/p/16691035.html)

Step1: wsl2內安裝網路包

$ sudo apt install net-tools

$ ifconfig eth0

Step2: get local ip address and mask (use cmd)

$ ipconfig

--->   ip address=(172.31.144.1)  and mask=(255.255.240.0)

Step3: get local WSL address and mask

$  ifconfig eth0

--->   ip address=(172.23.174.157)  and mask=(255.255.240.0)

Step4: 如果兩個mask不相同, 將wsl2 mask改成跟local 一樣

$ sudo ifconfig eth0 172.23.174.157 netmask 255.255.240.0

$ sudo ifconfig eth0 WSL IP ADDRESS netmask LOCAL MASK IP

Step5: 如果不行則增加local firewall rule

[https://www.cnblogs.com/SocialistYouth/p/16691035.html](https://www.cnblogs.com/SocialistYouth/p/16691035.html)

在WSL2 內正確安裝cuda11.7

Step0. 確保windows可以使用nvidia-smi  --> GPU driver OK

            確保windows可以使用nvcc -V  --> cuda OK (inside windows)

            確保WSL2可以使用nvidia-smi

如果沒有g++

ste1: install g++

$ sudo apt update

$ sudo apt install g++

Step2.  update apt

$ sudo apt update

$ sudo apt install build-essential dkms

Step3. add Nvidia cuda toolbox register

$ sudo apt-key adv --fetch-keys [https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub)

$ sudo sh -c 'echo "deb [https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/) /" > /etc/apt/sources.list.d/cuda.list'

Step4. Install cuda 11.7

$ sudo apt update

$ sudo apt install cuda-11-7

Step5  setup Path

$ nano ~/.bashrc  在最後加上兩行

exportPATH=/usr/local/cuda-11.7/bin${PATH:+:${PATH}}exportLD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}