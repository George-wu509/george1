
1  How to install nvidia-smi

The nvidia-smi utility normally gets installed in the driver install step. It cannot/does not get installed in any other installation step. f you install a NVIDIA GPU driver using a repository that is maintained by NVIDIA, you will always get the nvidia-smi utility with any recent driver install. The nvidia-smi.exe is located at C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe . If you don't see the file, it's possible that the driver installation is still running in the background.

2. How to use nvidia-smi to monitor GPU usage

Type " nvidia-smi " on cmd or windows power shell will display GPU informationa nd GPU usage

To monitor and update the GPU usage you can use " nvidia-smi -l 1 " command. The -l options performs polling on nvidia-smi every given seconds ( -lms if you want to perform every given milliseconds).

Other useful nvidia-smi queries:

$ ==nvidia-smi --query-gpu=gpu_name,gpu_bus_id,vbios_version==

It will show the official product name of the GPU. This is an alphanumeric string. For all products, PCI bus id as "domain:bus:device.function", in hex, and the BIOS of the GPU board.

3. How to use nvidia-smi to monitor GPU usage and redirect the result metric to some file

You may need to run the following command on windows power shell (cannot use cmd) because we need to use tee command to create csv file. The following command means monitor the time, gpu information, gpu temp, gpu usage, gpu memory used, total gpu memory, and free gpu memory every seconds and export the result to test1.csv

==nvidia-smi -l 1 --query-gpu=timestamp,pstate,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv | tee test1.csv==

![[Pasted image 20240917121646.png]]

It will create csv file under your user account root dir

![[Pasted image 20240917121705.png]]

This is the matlab script to import the csv file and build mat data file with variables: gpu_memory, gpu_usage, mem_free, mem_total, and mem_used.  Use the gpu_usage and mem_used can be used to draw figure just like this.

![[Pasted image 20240917121742.png]]
![[Pasted image 20240917121803.png]]![[Pasted image 20240917121818.png]]