
nvcc屬於時CUDA的編譯器，將程序編譯成可執行的二進製文件

nvidia-smi全稱是NVIDIA System Management Interface，是一種命令行實用工具，用來幫助管理和監控NVIDIA GPU設備的。

當我們安裝一個版本的cuda時，實際上會同時安裝runtime api和driver api，前者對應nvcc後者對應nvidia-smi查看到的。個人理解是，第一次安裝cuda時，nvcc關聯了第一次安裝的版本，並放在了環境變量中的/usr/bin目錄下。

應該選擇與nvcc --version對應的CUDA版本匹配的pytorch

Reference:

[1] 【CUDA】nvcc和nvidia-smi显示的版本不一致？

[https://www.jianshu.com/p/eb5335708f2a](https://www.jianshu.com/p/eb5335708f2a)