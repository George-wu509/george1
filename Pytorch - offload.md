
AI Infra: PyTorch Offload技术优化实践 - RuYy的文章 - 知乎
https://zhuanlan.zhihu.com/p/1949075068694013367


> 本文很大程度上参考[PyTorch](https://zhida.zhihu.com/search?content_id=262882102&content_type=Article&match_order=1&q=PyTorch&zhida_source=entity) Tutorial：[https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html](https://link.zhihu.com/?target=https%3A//docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html) 以及 [https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/](https://link.zhihu.com/?target=https%3A//developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/)

在大模型推理过程中，有很多内存换计算的技术，如LLM的[KVCache](https://zhida.zhihu.com/search?content_id=262882102&content_type=Article&match_order=1&q=KVCache&zhida_source=entity)、Diffusion的[Feature Cache](https://zhida.zhihu.com/search?content_id=262882102&content_type=Article&match_order=1&q=Feature+Cache&zhida_source=entity)等，然而GPU的显存往往是不够的，我们会希望借助更充足的CPU内存进行“扩容”，这就是[offload技术](https://zhida.zhihu.com/search?content_id=262882102&content_type=Article&match_order=1&q=offload%E6%8A%80%E6%9C%AF&zhida_source=entity)——前提是offload的开销（主要为数据搬移）小于直接计算开销。

**将数据在CPU和GPU之间传输**是非常基本的PyTorch操作，我们需要掌握如何高效地进行传输。

通过本文你将学习到：

1. CPU上普通内存和锁页内存对数据传输效率的影响。
2. PyTorch提供的`to()`和`pin_memory()`方法对数据传输效率的影响。
3. 如何高效而正确地进行异步数据传输。
4. 高效offload：通过双流流水线实现存、算、取的重叠。