
Ref:  [https://blog.csdn.net/weixin_41977938/article/details/117021480?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-3-117021480-blog-115324328.pc_relevant_multi_platform_whitelistv1&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-3-117021480-blog-115324328.pc_relevant_multi_platform_whitelistv1&utm_relevant_index=6](https://blog.csdn.net/weixin_41977938/article/details/117021480?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-3-117021480-blog-115324328.pc_relevant_multi_platform_whitelistv1&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-3-117021480-blog-115324328.pc_relevant_multi_platform_whitelistv1&utm_relevant_index=6)

![[Pasted image 20240912135100.png]]
![[Pasted image 20240912135125.png]]

其中寄存器(Registers)是GPU上运行速度最快的内存空间，通常其带宽为8TB/s左右，延迟为1个时钟周期。核函数中声明的一个没有其他修饰符的自变量，通常就存储在寄存器中。最快速也最受偏爱的存储器就是设备中的寄存器，属于具有重要价值有极度缺乏的资源。

接下来是共享内存(shared memory)，共享内存是GPU上可受用户控制的一级缓存。共享内存类似于CPU的缓存，不过与CPU的缓存不同，GPU的共享内存可以有CUDA内核直接编程控制。由于共享内存是片上内存，所以与全局内存相比，它具有更高的带宽与更低的延迟，通常其带宽为1.5TB/s左右，延迟为1～32个时钟周期。对于共享内存的使用，主要考虑数据的重用性。当存在着数据的重复利用时，使用共享内存是比较合适的。如果数据不被重用，则直接将数据从全局内存或常量内存读入寄存器即可。

全局内存(global memory)是GPU中最大、延迟最高并且最常使用的内存。全局内存类似于CPU的系统内存。在编程中对全局内存访问的优化以最大化程度提高全局内存的数据吞吐量是十分重要的。

![[Pasted image 20240912135138.png]]
Reference:  Nvidia CUDA c++ programming guide

[https://docs.nvidia.com/cuda/cuda-c-programming-guide/#axzz47OK4xWZP](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#axzz47OK4xWZP)

![[Pasted image 20240912135149.png]]

![[Pasted image 20240912135154.png]]

![[Pasted image 20240912135202.png]]

【CUDA编程】CUDA入门笔记

[https://blog.csdn.net/QLeelq/article/details/122359945?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-3-122359945-blog-117021480.pc_relevant_antiscanv2&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-3-122359945-blog-117021480.pc_relevant_antiscanv2&utm_relevant_index=6](https://blog.csdn.net/QLeelq/article/details/122359945?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-3-122359945-blog-117021480.pc_relevant_antiscanv2&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-3-122359945-blog-117021480.pc_relevant_antiscanv2&utm_relevant_index=6)