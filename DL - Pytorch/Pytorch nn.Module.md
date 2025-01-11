
BackboneWithFPN

類BackboneWithFPN的在backbon_utils.py中，功能是給骨幹網絡加上FPN

IntermediateLayerGetter

返回特定層的feature，這個feature通過輸入return_layers參數來控制

FeaturePyramidNetwork

FeaturePyramidNetwork的定義：

Module that adds a FPN from on top of a set of feature maps.

其實這個纔是BackboneWithFPN的核心組件，功能也很明確，將FPN加到給定的feature maps中。

參數定義：

in_channels_list : 輸入feature的channels

out_channels : 輸出feature的channels（統一值）

extra_blocks : 如果你想對某些層做額外的操作需要傳入這個參數，注意extra_blocks一定要繼承自ExtraFPNBlock類。

Reference;

[1] PyTorch 源码解读之 nn.Module：核心网络模块接口详解

[https://zhuanlan.zhihu.com/p/340453841](https://zhuanlan.zhihu.com/p/340453841)

[2] Pytorch torchvision構建Faster-rcnn（二）----基礎網絡

[https://www.twblogs.net/a/5d5eb8c5bd9eee5327fdc070](https://www.twblogs.net/a/5d5eb8c5bd9eee5327fdc070)