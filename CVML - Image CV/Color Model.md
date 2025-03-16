
[色域空间转换（color space）](https://zhuanlan.zhihu.com/p/559743079)

<mark style="background: #ADCCFFA6;">色彩模式(Colour model)</mark> - 色彩的表现方式
RGB(光的加和), 
HSB(心理学对颜色的感知，从色相（H），饱和度（S），明度（B）三个维度来描述一个颜色), CMYK(颜料对光的吸收) 
Lab(人眼对颜色的感知维度，以明度值，a为绿->红互补色对偏向值，b为蓝->黄互补色对偏向值)

<mark style="background: #ADCCFFA6;">色彩空间(Color Space) </mark>- 就是一个设备所能表现的所有颜色的集合
用数学语言表示，就是cES，其中c为任一颜色，S为某一色彩空间。 Example:sRGB，Adobe RGB

<mark style="background: #ADCCFFA6;">色域(Color gamut)</mark> - 色彩空间这个集合的范围
比如刚才说的Adobe RGB的色域比sRGB的大，而sRGB的色域又全部包含在Adobe RGB中，那么用数学语言描述，sRGB这个色彩空间就是Adobe RGB的（真）子集，即S（sRGB）C（）S（Adobe RGB）。