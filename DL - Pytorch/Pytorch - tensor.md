
創建torch.tensor()

(1) torch.tensor([1.0, 2.0]).type()

(2) torch.Tensor([1.0, 2.0]).type()

(3) torch.ones(shape), torch.zeros(shape), torch.full(shape, fill_value), torch.empty(shape)

(4) torch.ones_like(X), torch.zeros_like(X), torch.rand_like(X)       Create 與X相同shape的torch tensor

(5) torch.normal(mean, std)

(6) torch.rand(shape),  torch.randn(shape)

(7) torch.arange(start, end, step),  torch.linspace(start, end, step),  torch.logspace(start, end, step)

(8) torch.as_tensor(ndarray),  torch.from_numpy(ndarray)    将Numpy数组转化为PyTorch张量

(9) tensor.numpy()     将PyTorch张量转化为Numpy数组

A = torch.randn(1, 2, 3, 4)

--->

A =  [ [    [ [a1, a2,  a3, a4],

                  [b1, b2, b3, b4],

                  [c1,  c2, c3,  c4] ],

                [ [d1, d2,  d3, d4],

                  [e1, e2,   e3, e4],

                  [ f1,  f2,   f3,  f4] ]    ] ]

B = A.permute(0, 1, 3, 2)

--->

B=  [ [     [ [a1, b1, c1],

                  [a2, b2, c2],

                  [a3, b3, c3],

                  [a4, b4, c4] ],

               [ [d1, e1, f1],

                  [d2, e2, f2],

                  [d3, e3, f3],

                  [d4, e4, f4] ]       ] ]

C = A.view(1, 2, 4, 3)

--->

C=  [ [     [ [a1, a2, a3],

                  [a4, b1, b2],

                  [b3, b4, c1],

                  [c2, c3, c4] ],

               [ [d1, d2, d3],

                  [d4, e1, e2],

                  [e3, e4, f1],

                  [f2, f3, f4] ]       ] ]

D = A.reshape(1, 2, 4, 3)

--->

C=  [ [     [ [a1, a2, a3],

                  [a4, b1, b2],

                  [b3, b4, c1],

                  [c2, c3, c4] ],

               [ [d1, d2, d3],

                  [d4, e1, e2],

                  [e3, e4, f1],

                  [f2, f3, f4] ]       ] ]

reshape 和view方法的结果是一致的，但是view没有开辟新的内存空间，而reshape开辟了新的内存空间。尽管reshape开辟了新的内存空间，但是指向的底层元素地址依旧没有变换，也就是说，对D的操作会影响到A

Reference:

[1] PyTorch 常用方法总结4：张量维度操作（拼接、维度扩展、压缩、转置、重复……）

[https://zhuanlan.zhihu.com/p/31495102](https://zhuanlan.zhihu.com/p/31495102)

[2] 1. PyTorch中的基本数据类型——张量

[https://zhuanlan.zhihu.com/p/326340425](https://zhuanlan.zhihu.com/p/326340425)

[3] pytorch常用总结 之 tensor维度变换

[https://zhuanlan.zhihu.com/p/206444428](https://zhuanlan.zhihu.com/p/206444428)

[4] PyTorch 常用方法总结1：生成随机数Tensor的方法汇总（标准分布、正态分布……

[https://zhuanlan.zhihu.com/p/31231210](https://zhuanlan.zhihu.com/p/31231210)