
================== Data ==================

Create Tensor

------------------------

x1 = torch.zeros(2,3), or torch.rand(2,3)

x2 = torch.tensor([[1.0, 3.8, 2.1], [8.6, 4.0, 2.4]])

x3 = torch.from_numpy(np.array([[1.0, 3.8, 2.1], [8.6, 4.0, 2.4]]))

Send Tensor to GPU/CPU

------------------------

torch.rand(2, 3).cuda()

torch.rand(2, 3, device="cuda")

torch.rand(2, 3).to("cuda")

Backward

------------------------

x = torch.tensor([2.], requires_grad=True)

Reference:  

(1)  用例子学习 PyTorch  

[https://github.com/apachecn/pytorch-doc-zh/blob/master/docs/1.0/pytorch_with_examples.md](https://github.com/apachecn/pytorch-doc-zh/blob/master/docs/1.0/pytorch_with_examples.md)

(2)  pytorch的计算图

[https://zhuanlan.zhihu.com/p/33378444](https://zhuanlan.zhihu.com/p/33378444)

(3) PyTorch 中文手册（pytorch handbook）

[https://github.com/zergtant/pytorch-handbook](https://github.com/zergtant/pytorch-handbook)

(4) 从基础概念到实现，小白如何快速入门PyTorch

[https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650738301&idx=1&sn=d30f1f85b202c32f5b33444bec71f3c7&chksm=871aca03b06d4315bf7fb51e512d30de1da95ab056eee34e30e7e6a9a933dcad644028ea523c&mpshare=1&scene=21&srcid=070970WM2VGKFhvfHykC4nUJ&pass_ticket=Ard1NNQxVY96CuPn2ht/pJM7QzZ1Wb0d/CljJNUG9Jqd2NdVb7AHg7Igsxen082O#wechat_redirect](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650738301&idx=1&sn=d30f1f85b202c32f5b33444bec71f3c7&chksm=871aca03b06d4315bf7fb51e512d30de1da95ab056eee34e30e7e6a9a933dcad644028ea523c&mpshare=1&scene=21&srcid=070970WM2VGKFhvfHykC4nUJ&pass_ticket=Ard1NNQxVY96CuPn2ht/pJM7QzZ1Wb0d/CljJNUG9Jqd2NdVb7AHg7Igsxen082O#wechat_redirect)

(5) Pytorch——一个简单但强大的深度学习库

[https://zhuanlan.zhihu.com/p/70132718](https://zhuanlan.zhihu.com/p/70132718)

[6] PyTorch 中構建模型和輸入数據的方法  [https://tangh.github.io/articles/build-model-and-dataset-in-pytorch/](https://tangh.github.io/articles/build-model-and-dataset-in-pytorch/)

[7] PyTorch之前向传播函数forward

[https://blog.csdn.net/u011501388/article/details/84062483](https://blog.csdn.net/u011501388/article/details/84062483)

[8] PyTorch JIT

[https://zhuanlan.zhihu.com/p/410507557](https://zhuanlan.zhihu.com/p/410507557)

[9] PyTorch 学习笔记（八）：PyTorch的六个学习率调整方法

[https://zhuanlan.zhihu.com/p/69411064](https://zhuanlan.zhihu.com/p/69411064)

[10] PyTorch官方教程中文版

[https://pytorch123.com/ThirdSection/SaveModel/](https://pytorch123.com/ThirdSection/SaveModel/)

[11] pytorch中的钩子（Hook）有何作用？

[https://www.zhihu.com/question/61044004](https://www.zhihu.com/question/61044004)

[12] PyTorch中文文档

[https://pytorch-cn.readthedocs.io/zh/latest/](https://pytorch-cn.readthedocs.io/zh/latest/)

[13] 理解optimizer.zero_grad(), loss.backward(), optimizer.step()的作用及原理

[https://flyswiftai.com/li-jieoptimizerzerograd-lossbackward-opt/](https://flyswiftai.com/li-jieoptimizerzerograd-lossbackward-opt/)