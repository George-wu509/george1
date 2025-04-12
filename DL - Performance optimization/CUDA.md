
Ref:  [https://blog.csdn.net/weixin_41977938/article/details/117021480?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-3-117021480-blog-115324328.pc_relevant_multi_platform_whitelistv1&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-3-117021480-blog-115324328.pc_relevant_multi_platform_whitelistv1&utm_relevant_index=6](https://blog.csdn.net/weixin_41977938/article/details/117021480?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-3-117021480-blog-115324328.pc_relevant_multi_platform_whitelistv1&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-3-117021480-blog-115324328.pc_relevant_multi_platform_whitelistv1&utm_relevant_index=6)

一个典型的CUDA程序是按这样的步骤执行的：

把数据从CPU内存拷贝到GPU内存。
调用核函数对存储在GPU内存中的数据进行操作的。
将数据从GPU内存传送回CPU内存。
   一般CPU一个核只支持一到两个硬件线程，而GPU往往在硬件层面上就支持同时成百上千个并发线程。不过这也要求我们在GPU编程中更加高效地管理这些线程，以达到更高的运行效率。在CUDA编程中，线程是通过线程网格（Grid）、线程块（Block）、线程束（Warp）、线程（Thread）这几个层次进行管理的.

  第二点，为了达到更高的效率，在CUDA编程中我们需要格外关注内存的使用。与CPU编程不同，<mark style="background: #FFF3A3A6;">GPU中的各级缓存以及各种内存是可以软件控制的，在编程时我们可以手动指定变量存储的位置。</mark>具体而言，这些内存包括**寄存器、共享内存、常量内存、全局内存**等。这就造成了CUDA编程中有很多内存使用的小技巧，<mark style="background: #FF5582A6;">比如我们要尽量使用寄存器，尽量将数据声明为局部变量。而当存在着数据的重复利用时，可以把数据存放在共享内存里。而对于全局内存，我们需要注意用一种合理的方式来进行数据的合并访问，以尽量减少设备对内存子系统再次发出访问操作的次数。</mark>

首先我們需要了解線程thread是如何組織的, 下面這張圖清晰的表示出了線程的組織結構. 當核函數在主機端啟動時, 其執行會移動到設備上. 此時設備中會產生大量的線程並且每個線程都執行由核函數指定的語句

![[Pasted image 20240912135100.png]]
內存管理. CUDA編程另一個顯著的特點就是解釋了內存層次結構. 每一個GPU設備都會有用於不同用途的存儲類型.

![[Pasted image 20240912135125.png]]

其中<mark style="background: #FF5582A6;">寄存器(Registers)</mark>是GPU上运行速度最快的内存空间，通常其带宽为8TB/s左右，延迟为1个时钟周期。核函数中声明的一个没有其他修饰符的自变量，通常就存储在寄存器中。最快速也最受偏爱的存储器就是设备中的寄存器，属于具有重要价值有极度缺乏的资源。暫存器 (Registers) 是位於 GPU 運算單元（如 CUDA 核心/SP）內部或極其接近的儲存空間。它們是執行緒thread私有的，用於儲存執行緒的局部變數和計算的中間結果。訪問延遲極低（幾乎為零，通常在單個時脈週期內完成），是 GPU 上最快的記憶體。- 

**暫存器是分配給執行緒 (Thread) 的，而不是 SP (Streaming Processor / CUDA Core)。** 一個執行緒會使用_多個_暫存器來儲存其狀態和計算所需的值。
- **暫存器與 SP 沒有一對一的固定對應關係。** 一個 SM (Streaming Multiprocessor) 包含多個 SP。SM 內有一個大型的暫存器檔案 (Register File)，這個檔案由所有在該 SM 上運行的執行緒共享。當一個執行緒被排程到某個 SP 上執行時，它會從這個共享的暫存器檔案中讀取/寫入分配給_該執行緒_的暫存器。

接下来是<mark style="background: #FFB86CA6;">共享内存(shared memory)</mark>，共享内存是GPU上可受用户控制的一级缓存。共享内存类似于CPU的缓存，不过与CPU的缓存不同，GPU的共享内存可以有CUDA内核直接编程控制。由于共享内存是片上内存，所以与全局内存相比，它具有更高的带宽与更低的延迟，通常其带宽为1.5TB/s左右，延迟为1～32个时钟周期。对于共享内存的使用，主要考虑数据的重用性。当存在着数据的重复利用时，使用共享内存是比较合适的。如果数据不被重用，则直接将数据从全局内存或常量内存读入寄存器即可。每個 SM (Streaming Multiprocessor) 都有自己獨立的一塊片上記憶體，這塊記憶體可以被配置為 L1 快取和共享記憶體。運行在某個 SM 上的執行緒區塊只能訪問_該 SM_ 的共享記憶體。不同 SM 之間的共享記憶體是互相隔離的。
- **速度快、延遲低：** 共享記憶體 (Shared Memory) 和 CPU 的 L1 快取一樣，都是位於處理器晶片上（On-chip），比主記憶體（GPU 的 Global Memory 或 CPU 的 RAM）快得多。
- **資料共享/重用：** 它們都用於在處理單元（GPU 的執行緒區塊或 CPU 核心）之間快速共享資料，或者作為一個可控的快取來減少對較慢記憶體的訪問。
- - **管理方式：** CPU 快取通常由硬體自動管理，對程式設計師是透明的；而 GPU 的共享記憶體需要**程式設計師在 CUDA 核心中顯式聲明 (`__shared__`) 和管理**（手動載入資料、確保同步 `__syncthreads()`）。
- **可見性：** 共享記憶體只對同一個執行緒區塊 (Thread Block) 內的執行緒可見和共享。CPU L1 快取通常是單個 CPU 核心私有。

<mark style="background: #ABF7F7A6;">全局内存(global memory)</mark>是GPU中最大、延迟最高并且最常使用的内存。全局内存类似于CPU的系统内存。在编程中对全局内存访问的优化以最大化程度提高全局内存的数据吞吐量是十分重要的。全域記憶體 (Global Memory) 通常是指 GPU 顯示卡上的主記憶體（DRAM）。相較於片上的暫存器和共享記憶體/L1 快取，它的訪問延遲要高得多，頻寬雖然總量很大，但單次訪問效率較低。它是 GPU 上容量最大但速度最慢的記憶體層次。從 CUDA 程式設計模型的角度看，整個 GPU 共享一個統一的邏輯全域記憶體地址空間。所有 SM 都可以訪問這個地址空間。物理上，這對應於 GPU 卡上的所有 DRAM 晶片組成的記憶體池。

![[Pasted image 20240912135138.png]]
Reference:  Nvidia CUDA c++ programming guide

[https://docs.nvidia.com/cuda/cuda-c-programming-guide/#axzz47OK4xWZP](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#axzz47OK4xWZP)

![[Pasted image 20240912135149.png]]

![[Pasted image 20240912135154.png]]

![[Pasted image 20240912135202.png]]

【CUDA编程】CUDA入门笔记

[https://blog.csdn.net/QLeelq/article/details/122359945?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-3-122359945-blog-117021480.pc_relevant_antiscanv2&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-3-122359945-blog-117021480.pc_relevant_antiscanv2&utm_relevant_index=6](https://blog.csdn.net/QLeelq/article/details/122359945?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-3-122359945-blog-117021480.pc_relevant_antiscanv2&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-3-122359945-blog-117021480.pc_relevant_antiscanv2&utm_relevant_index=6)