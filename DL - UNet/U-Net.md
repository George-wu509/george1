
ref:  [UNet理解，pytorch实现，源码解读](https://zhuanlan.zhihu.com/p/571760241)
ref: [Unet论文超级详解（附图文：超细节超容易理解）](https://zhuanlan.zhihu.com/p/716339396)
ref: [U-Net原理分析与代码解读](https://zhuanlan.zhihu.com/p/150579454)
![[unet.png]]

如上图，Unet 网络结构是对称的，形似英文字母 U 所以被称为 Unet。整张图都是由蓝/白色框与各种颜色的箭头组成，其中，**蓝/白色框表示 feature map；蓝色箭头表示 3x3 卷积，用于特征提取；灰色箭头表示 skip-connection，用于[特征融合](https://zhida.zhihu.com/search?content_id=121594236&content_type=Article&match_order=1&q=%E7%89%B9%E5%BE%81%E8%9E%8D%E5%90%88&zhida_source=entity)；红色箭头表示池化 pooling，用于降低维度；绿色箭头表示上采样 upsample，用于恢复维度；青色箭头表示 1x1 卷积，用于输出结果**。其中灰色箭头`copy and crop`中的`copy`就是`concatenate`而`crop`是为了让两者的长宽一致


ref: [nn.ConvTranspose2d原理，深度网络如何进行上采样？](https://blog.51cto.com/u_15274944/5244229)

