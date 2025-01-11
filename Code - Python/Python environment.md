
conda --version 檢視 conda 版本

conda update PACKAGE_NAME更新指定套件

conda --help 檢視 conda 指令說明文件

conda list --ENVIRONMENT 檢視指定工作環境安裝的套件清單

conda install PACAKGE_NAME=MAJOR.MINOR.PATCH 在目前的工作環境安裝指定套件

conda remove PACKAGE_NAME 在目前的工作環境移除指定套件

conda create --name ENVIRONMENT python=MAIN.MI 建立新的工作環境且安裝指定 Python 版本

conda activate ENVIRONMENT 切換至指定工作環境

conda deactivate 回到 base 工作環境

conda env export --name ENVIRONMENT --file ENVIRONMENT.yml 將指定工作環境之設定匯出為 .yml 檔藉此複製且重現工作環境

conda remove --name ENVIRONMENT --all 移除指定工作環境

Export/create conda env

1. (export conda env) conda env export  > xxxxxx.yml

2. (create env using yml file) conda env create -f xxxxx.yml

3. (remove env from conda) conda env remove - -name xxxxx

4. (list all conda packages) conda env list

5. (list packages in xxxx env) conda list - -name xxxxx

6. (install yyyy package to some env) conda install yyyy - -name xxxx

7. (install yyyy package in conda-firge chanel to some env) conda install -c conda-forge yyyy - -name xxxx

8. (clone one env from env) conda create - -name CLONEENV - -clone OLDENV

如果遇到certicate error:

WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1006)'))': /compute/redist/opencv-python/

---->  可以用system-certs代替去解決問題  pip install pip-system-certs