
在cmd or powershell執行:

1. Install Docker 用docker image “hello-world” 建立container

$ sudo apt-get install docker

$ sudo apt-get install docker.io

$ docker run hello-world   <-check docker install correctly

2a  用docker image “ubuntu:15.10” 建立container並在容器里执行 bin/echo "Hello world"，然后输出结果

$ docker run ubuntu:15.10 /bin/echo “Hello world“

2b  Explanation 說明

sudo docker run -itd --name dinov2_container -p 8080:80 -v /home/a3146654/dinov2:/dinov2 chjkusters4/dinov2:V4 /bin/bash

-t: 在新容器内指定一个伪终端或终端。(终端)

-i: 允许你对容器内的标准输入 (STDIN) 进行交互。(交互式操作)

-d:使容器在后台运行，并返回容器 ID  （detached）

-p 8080:80 参数用于端口映射. 格式是 宿主机端口:容器端口

将宿主机的 8080 端口映射到容器的 80 端口。换句话说，访问宿主机的 [http://localhost:8080](http://localhost:8080) 实际上是在访问容器的 80 端口

-v /home/a3146654/dinov2:/dinov2

参数用于挂载卷，格式是 宿主机路径:容器路径. 这里表示将宿主机的 /home/a3146654/dinov2 目录挂载到容器的 /dinov2 目录。这样你可以在容器内访问和操作宿主机的 /home/a3146654/dinov2 目录中的文件。

--name dinov2_container

用于给容器指定一个名称(dinov2_container)

chjkusters4/dinov2:V4

这是要使用的 Docker 镜像的名称和标签。 Docker 会从 Docker Hub 上拉取这个镜像，如果本地没有这个镜像，它会自动下载。

/bin/bash

这是在容器启动后要运行的命令。/bin/bash 启动一个 Bash shell

$ docker run -i -t ubuntu:15.10 /bin/bash

2c. 创建一个以进程方式运行的容器. -d 指定容器的运行模式。

加了 -d 参数默认不会进入容器，想要进入容器需要使用指令 docker exec（下面会介绍到）。

$ docker run -d ubuntu:15.10 /bin/sh -c "while true; do echo hello world; sleep 1; done"

輸出得到2b1b7a428627c51ab8810d541d759f072b4fc75487eed05812646b8534a2fe63

这个长字符串叫做容器 ID

容器(Docker container)

$ docker ps  列出本地主机上正在run的docker container

$ docker ps  -a 列出本地主机上所有的docker container(running, stop, paused)

$ docker stop    docker stop 命令来停止容器:

$ docker  直接输入 docker 命令来查看到 Docker 客户端的所有命令选项。

$ docker start <container ID>   使用 docker start 启动一个已停止的容器：

$ docker restart <container ID>   停止的容器可以通过 docker restart 重启：

$ exit    or   ctrl+D  

$ ls     当前目录下的文件列表

$ cat /proc/version  查看当前系统的版本信息

镜像(Docker Image)

$ docker images  列出本地主机上的镜像。

$ docker search <image name>使用 docker search 命令来搜索镜像

$ docker pull <image name>從docker hub获取一个新的镜像到local

$ docker run -t -i <image name> /bin/bash  使用镜像来运行容器

$ docker rmi <image name> 镜像删除使用 docker rmi 命令

创建镜像 (更新镜像，并且提交这个镜像)

更新镜像之前，我们需要使用镜像来创建一个容器

$ docker run -t -i <image name> /bin/bash

在运行的容器内使用 $ apt-get update 命令进行更新。

在完成操作之后，输入 $ exit 命令来退出这个容器。

 ID 为 e218edb10161 的容器，是按我们的需求更改的容器。我们可以通过命令 docker commit 来提交容器副本。

$ docker commit -m="has update" -a="runoob" e218edb10161 runoob/ubuntu:v2

-m: 提交的描述信息,  -a: 指定镜像作者, e218edb10161：容器 ID. runoob/ubuntu:v2: 指定要创建的目标镜像名

$ docker run -t -i runoob/ubuntu:v2 /bin/bash   

创建镜像  (使用 Dockerfile 指令来创建一个新的镜像)

我们使用命令 $ docker build  从零开始来创建一个新的镜像

我们需要创建一个 Dockerfile 文件，其中包含一组指令来告诉 Docker 如何构建我们的镜像。

然后，我们使用 Dockerfile 文件，通过 $ docker build 命令来构建一个镜像。

$ docker build -t nginx:v3 .

镜像(Docker Compose)

從映像檔產生 Dockerfile [https://philipzheng.gitbook.io/docker_practice/dockerfile/file_from_image](https://philipzheng.gitbook.io/docker_practice/dockerfile/file_from_image)

$ docker run -v /var/run/docker.sock:/var/run/docker.sock \ centurylink/dockerfile-from-image <IMAGE_TAG_OR_ID> > Dockerfile.txt

Reinstall Docker

sudo apt-get remove docker docker-engine docker.io containerd runc    卸載現有的 Docker