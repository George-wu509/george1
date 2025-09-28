
影像新势力 | ISP 色彩调试基本原理 - 大大通的文章 - 知乎
https://zhuanlan.zhihu.com/p/2523071522

Color Correction Tuning Guide - 烫手的洋芋的文章 - 知乎
https://zhuanlan.zhihu.com/p/31911875


[P 色彩调试基本原理](https://link.zhihu.com/?target=https%3A//www.wpgdadatong.com.cn/blog/detail/75831%3Futm_source%3Dwpg_ddt%26utm_medium%3Dzhihu_ddt%26utm_campaign%3D2024_oct_25%26utm_content%3Dblog_75831)

**Camera硬件基本组成：**

![](https://pic4.zhimg.com/v2-06cb9dfc9759d82a3b89708abaee506b_1440w.jpg)

Camera硬件基本组成

  

**[BLC](https://zhida.zhihu.com/search?content_id=249527258&content_type=Article&match_order=1&q=BLC&zhida_source=entity)（黑电平校正-black level Correction）**

**1. 产生原因**

因为sensor电路本身存在暗电流，导致在没有光线照射的时候像素单位也会有一定的输出电压，BLC易受到Again和温度的影响，电路的增益增大，暗电流也会增强。


**2. 校正原理**

在Sensor上预留一些没有曝光的像素，通过读其像素值的大小得到OB（Optical Black Level）此时sensor输出的RAW=input-ob,考虑到信噪比一般还会进行基底矫正（pedestal），此时sensor输出的RAW=input-ob-pedestal。


**3. 校正步骤**

（1）全黑采集RAW图，分为R、Gr、Gb、B 四个通道；

（2）对四个通道求平均值（或中值）作为校正值；

（3）对图像的每个通道都减去这个校正值。

![](https://pica.zhimg.com/v2-f8e08dd94a7b23527f468507ceccf6d0_1440w.jpg)

**4 影响范围**

色彩：偏色，RGB分量、对比度、破坏噪声形态降低信噪比。

![](https://pic4.zhimg.com/v2-f4f747c3efcbd85d710c0c33f5e0ecbb_1440w.jpg)

注：当Blc扣除过多时，会导致整体画面亮度过低，动态范围变低，细节损失多，黑色部分偏色无法通过白平衡纠正，画面偏绿。当BLC扣除过少时，扣除的黑电平值不足以抵消红色通道的信号，导致红色部分的亮度过高，从而使整个图像偏红。

  

**LSC（镜头阴影抑制-Lens Shading Correction）**

**1. 产生原因**

由于镜头的光学特性会导致sensor的影像边缘区域接收到的光强比中心小，造成中心和四角亮度不一致的现象，并且镜头本身是一个凸透镜，由于凸透镜的原理，中心的感光必然比周边多。需要根据像素的位置做增益补偿

![](https://pica.zhimg.com/v2-09eacdda80a7c659bc96acbc7fecc0ca_1440w.jpg)

LSC（镜头阴影抑制-Lens Shading Correction）

  

**Luma shading：**会造成图像边角偏暗

![](https://pic3.zhimg.com/v2-525030c2bd506ec9bc0dd9f4e164918a_1440w.jpg)

Luma shading：会造成图像边角偏暗

  

**color shading：**中心和四周颜色不一致，偏色

![](https://pica.zhimg.com/v2-9434c685d552fa9cbb9ab68ca9b999c4_1440w.jpg)

color shading：中心和四周颜色不一致，偏色

  

**2. 校正方法**

**网格法：**把整幅图像分成m*n个网格，然后针对网格顶点求出校正的增益，然后把这些顶点的增益储存到内存中，其它点的增益通过插值的方式求出 。

![](https://pic2.zhimg.com/v2-33ba5e099cdbe0c5c7d410287a1e56f3_1440w.jpg)

网格法

![](https://picx.zhimg.com/v2-e1edbc60178877ee4c80705094e74753_1440w.jpg)

网格法

  

**[AWB](https://zhida.zhihu.com/search?content_id=249527258&content_type=Article&match_order=1&q=AWB&zhida_source=entity)（自动白平衡-Automatic White Balance）**

**1. 基本原理**

白平衡就是：不管在任何光源下，都能将拍摄的白色物体的图像还原为白色，人眼在早晨、中午、晚上 不同色温下都能准确看到白色。CMOS 要获得这种能力，必须对每种光源做相应的色彩校准，才能完成人眼一样的功能。这个功能叫做白平衡。

![](https://pic2.zhimg.com/v2-4be73f136d677b82b285ab0b2667c609_1440w.jpg)

**2. 校准原理**

人眼中的白色总是R=G=B，那么白平衡所做的事情就是在不同色温条件下将图像做归一化，也就是如下将Sensor响应的RGB分别乘上一个系数，使得物体在不同光源条件下所呈现出来的颜色，恢复到物体的固有色。

R'=R×R_Gain，

G'=G×G_Gain，

B'=B×B_Gain，

![](https://pic3.zhimg.com/v2-828c58c44684103ff19ba547a1e0d190_1440w.jpg)

使得R'=G'=B'。  

  

**3. 算法原理**

**（1）灰世界算法**

原理：灰度世界算法以灰度世界假设为基础，该假设认为：对于一幅有着大量色彩变化的图像,其R,G,B 三个色彩分量的平均值趋于同一灰度值 K。从物理意义上讲，灰色世界法假设自然界景物对于光线的平均反射的均值在总体上是个定值，这个定值近似地为“灰色”。颜色平衡算法将这一假设强制应用于待处理图像，可以从图像中消除环境光的影响，获得原始图像。

**步骤：**

1. 确定K值.一般有2种方法确定K值，

1).K取固定值，如最亮灰度值的一半，针对0-255图像，可以取128

2).计算图像R,G,B三个通道的平均值 R ˉ , G ˉ , B ˉ ，K=（Rˉ+Gˉ+Bˉ）/3

  

2. 计算R,G,B三个通道的增益系数:

![](https://pic1.zhimg.com/v2-1ee8c4eb8fc96555bb7aeddebf757f3c_1440w.jpg)

  

3. 对于图像中的每个像素的像素值R,G,B,计算其调整后的值:

![](https://pic4.zhimg.com/v2-094c92c6784627653968d24b9544e96f_1440w.jpg)

  

注：这种算法简单快速，但是当图像场景颜色并不丰富时，尤其出现大块单色物体时，该算法常会失效。当图片中没有足够丰富的色彩来近似理想情况时，灰度世界算法的白平衡效果就差强人意。

  

（2）白点统计法

1）将原图转换为YCrCb空间：

转换关系如下：

![](https://pic3.zhimg.com/v2-d6b57fc698c0e3b15f852697046bfa6e_1440w.jpg)

  

**（2）白点统计法**

在YCrCb空间定义一个三维结构，落于此区域内的像素统计为白点，计算[色差](https://zhida.zhihu.com/search?content_id=249527258&content_type=Article&match_order=1&q=%E8%89%B2%E5%B7%AE&zhida_source=entity)时只需计算白色像素的平均色差来取代整个图像的色差，从而提高色温计算的准确度，限定YCrCb的约束条件来判断是否为白点：

![](https://pic1.zhimg.com/v2-5051338a9ecae0e8de23302c6bf9bf40_1440w.jpg)

  

![](https://pic4.zhimg.com/v2-cb149a4fec78c8f1e37983bb8b53304b_1440w.jpg)

Ymin:CbCr较小的像素表现为接近白色的灰度值；Ymax:CbCr较大的像素可以认为是白色物体受到光源干扰得到的，当图像中出现灯光时灯光周围区域会出现过曝的现象色彩成分被抑制。

  

2) 统计白点（有效像素点）数量、累加对应的R、G、B、分量值，得到R、G、B的平均值：

![](https://pica.zhimg.com/v2-e8e0e62d50491cc25a566bebfc0cb9c8_1440w.jpg)

  

3）得到平均亮度、确定各分量的增益系数：

![](https://pic3.zhimg.com/v2-a2e3077cc8a271709e6290414b7cd940_1440w.jpg)

  

4）最后进行白平衡校正：

![](https://pic1.zhimg.com/v2-88e243dfd924ac27ae1f3c704c059fb8_1440w.jpg)

  

**（3）白点统计法的实现**

1）标定参考点：提取Sensor在不同标准光源下的白点特征（R/G、B/G）；

2）计算色温拟合曲线；

3）根据色温曲线确定灰区；

4）调整各通道增益实现收白或者喜好色调整。

![](https://pic3.zhimg.com/v2-e00ca771bd0bef9abca0c680039f2474_1440w.jpg)

  

**（4）色温计算原理**

1）取的图像数据，并划分MxN块，如果是25x25，并统计每一块的基本信息(白色像素的数量及R/G/B通道的分量的均值)。

![](https://pic1.zhimg.com/v2-9385548c1e0602019c8653a3a93ba56a_1440w.jpg)

  

2）根据第1步中的统计值，找出图像中所有的白色块，并根据色温曲线判断色温。

比如25x25=625 个块中，一共找出了100个有效白色块，里面又有80个白色块代表了色温4500左右，那当前色温基本就是4500。根据4500色温得出的Rgain，Bgain来调整当前图像的白平衡。

![](https://pic1.zhimg.com/v2-a6ad03faa90095c5040219bea6cbccfa_1440w.jpg)

  

**[CCM](https://zhida.zhihu.com/search?content_id=249527258&content_type=Article&match_order=1&q=CCM&zhida_source=entity)（色彩矩阵校正-Color Correction Matrix）**

**1. 基本原理**

sensor 的 RGB 分量对光谱的响应与人眼的不同，为了使图像显示的颜色与人眼接近，则需要色彩校正模块对各种颜色进行还原，将颜色从sensor RGB空间变换到人眼的RGB空间，使图像的效果符合人的主观感受。

![](https://picx.zhimg.com/v2-6d7f56180060fddf2bbcedcec614a3b1_1440w.jpg)

  

**2. 校正原理**

使用 sensor 抓拍到的 24 色卡场景下前 18 个色块的实际颜色信息和其期望值，计算 3x3 的 CCM 矩阵。输入颜色经 CCM 矩阵处理得到的颜色与其期望值差距越小，则 CCM 矩阵就越理想。

![](https://pica.zhimg.com/v2-be1a5c9ed500c3b04b3828f6c53b130a_1440w.jpg)

**3. 调试方法**

（1）调试CCM矩阵时，先将矩阵配置成单位阵（ rr=gg=bb=1），再将饱和度提高到适当值；

![](https://pica.zhimg.com/v2-a69b7424bf2c44a24f5a7481a3bf7c60_1440w.jpg)

  

（2）对着24色卡对颜色进行调整至用户喜好颜色。

以纵轴为基色，横轴为分量，找到对应的色彩在色相环上的位置，再根据色彩偏向调整CCM数值。

![](https://pic4.zhimg.com/v2-bfe1c87297555b1eec4c7e18d8a3fcd5_1440w.jpg)

![](https://pic2.zhimg.com/v2-49cffeddc862a584137bc6691c196d6f_1440w.jpg)

备注：此图参考此博文所总结整理：Hisi平台CCM调试-细调_Camera Man的博客-CSDN博客

  

**[3DLUT](https://zhida.zhihu.com/search?content_id=249527258&content_type=Article&match_order=1&q=3DLUT&zhida_source=entity)(Look up ttable)**

**基本原理**

颜色查找表，用于颜色校正的技术，它可以将输入的颜色值映射到输出的颜色值。它的原理是将输入颜色空间中的每个颜色值映射到输出颜色空间中的一个对应颜色值，这个映射关系可以通过一个三维数组来表示，这个数组就是3D LUT。

3D LUT的作用是对图像进行颜色校正，使得图像在不同设备上显示时颜色表现一致。它可以校正图像的亮度、对比度、饱和度等参数，从而达到更加真实、自然的效果。

![](https://pic3.zhimg.com/v2-f5f0a2a13d3c2e7e8b385cc1519d3ce2_1440w.jpg)

  

**色差（chromatic aberration）**

**1. 产生原因**

透镜对不同波长的色光有不同的折射率，波长越长折射率越高。

![](https://pic3.zhimg.com/v2-2d0f9530ece58e58b19bb095e3e8be44_1440w.jpg)

**2. 分类**

（1）横向色差

不同波长的光按照一定的角度进入透镜，并且聚焦到沿着相同的平面上不同的位置时产生的色差。一般出现于图像边缘高对比度的地方。图像的放大倍数随波长的不同而产生的颜色条纹。

![](https://picx.zhimg.com/v2-033dbc8902f936d354c94e2460cbe505_1440w.jpg)

  

（2）纵向色差

不同波长的光沿着水平光轴进入透镜后不能聚焦同一平面的同一点而产生的色差。一般出现于物体边缘呈现红绿蓝或者这些颜色的结合。可以通过缩小镜头光圈来大幅度减少色差。

![](https://pic2.zhimg.com/v2-9307c987f051383c524a3b4f6a4604ab_1440w.jpg)

![](https://pic4.zhimg.com/v2-06a1d3e5dd4c877f244f46779bfbf0a9_1440w.jpg)

**紫边（purple boundary）**

**产生原因：**

通常认为紫边的成因是镜头色差，即镜头对不同光谱光线的折射程度不同，导致不同光谱的光线不能成像到一点上。成像系统一般将绿色通道准确对焦，然而由于镜头色差，蓝色和红色通道不能完全准确对焦，从而使物体边缘出现紫红色的色边。实际就是色差导致了R≠G≠B，所以才会有了颜色；紫色就是R、B通道的分量 > G通道的分量。

  

**步骤：**

![](https://pica.zhimg.com/v2-68df9fb0ef0aa51eb4e88c637df1bbf2_1440w.jpg)

  

**现象：**

![](https://pic1.zhimg.com/v2-7523954a8416b107f585804b698fd044_1440w.jpg)