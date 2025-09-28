
图像处理基础（十五）基于双边滤波的 tone-mapping - 山与水你和我的文章 - 知乎
https://zhuanlan.zhihu.com/p/496261579


HDR（High Dynamic Range），首先动态范围指的是图中最亮和最暗的比值，而高动态范围是个有点复杂的概念，本人对它的认识局限于三个方面

- HDR 成像(imaging)
- tone-mapping
- 多曝光融合

第一项，HDR imaging，[CMU](https://link.zhihu.com/?target=http%3A//graphics.cs.cmu.edu/courses/15-463/lectures/lecture6.pdf) 课程上写的很清楚，解决的是从场景到传感器的限制

> HDR imaging compensates for sensor limitations

拍摄设备所能捕捉的动态范围 ＜ 现实场景的动态范围，一次拍摄难以完整呈现场景亮度和细节，一般会通过更改快门速度、ISO 等设置拍摄不同曝光水平的图像，做融合，分两步

> 1. [Exposure bracketing](https://zhida.zhihu.com/search?content_id=198412077&content_type=Article&match_order=1&q=Exposure+bracketing&zhida_source=entity): Capture multiple LDR images at different exposures  
> 2. [Merging](https://zhida.zhihu.com/search?content_id=198412077&content_type=Article&match_order=1&q=Merging&zhida_source=entity): Combine them into a single HDR image

从而得到具有接近真实场景的高动态范围 HDR 图像。

第二项，tone-mapping，解决的是从 HDR 图像到显示设备的限制

> Tonemapping compensates for display limitations

显示设备的动态范围 ＜ HDR 成像结果的动态范围，对于一般的显示设备来说，HDR 成像结果仍然是高动态范围，比如说 4000：1，但一般显示设备达不到这么高的动态显示范围，可能只有  的动态范围，所以，需要压缩动态范围（粗略理解成对比度吧），同时要保留细节。

常用的解决办法有两类——第一类是全局的动态范围压缩方法，使用类 S 曲线，如 [Reinhard](https://zhida.zhihu.com/search?content_id=198412077&content_type=Article&match_order=1&q=Reinhard&zhida_source=entity)、[ACES](https://zhida.zhihu.com/search?content_id=198412077&content_type=Article&match_order=1&q=ACES&zhida_source=entity) 等，像素的调整只和像素本身的灰度值有关系，这种方法速度快、直观、有效避免光晕和色调逆转，但是容易破坏图像的白平衡、局部丢失细节；第二类是局部的动态范围压缩方法，像素的调整和邻域的亮度分布有关系，局部方法可以呈现保留更多的细节，但是计算十分耗时、容易引入噪声且可能生成光晕。

第三项，多曝光融合，和第一项 HDR imaging 的一般解决策略好像重复了。思路相似，但这个曝光融合不是根据传感器的输入生成 HDR 图像，而是处理经过 [Camera pipeline](https://zhida.zhihu.com/search?content_id=198412077&content_type=Article&match_order=1&q=Camera+pipeline&zhida_source=entity) 处理之后的png格式等图像文件，也是根据不同曝光水平的图像，做融合，之前写过一个 [exposure fusion 的笔记](https://zhuanlan.zhihu.com/p/455674916) 。

---

最近在学习一些 HDR 在 tone-mapping 方面的传统算法。其中经典算法之一就是源于 2002年的 《Fast [Bilateral Filtering](https://zhida.zhihu.com/search?content_id=198412077&content_type=Article&match_order=1&q=+Bilateral+Filtering&zhida_source=entity) for the Display of High-Dynamic-Range Images》，基于快速双边滤波的高动态范围压缩方法，在此做个记录。

## 原理

基本思路是，使用滤波等手段将 HDR 图像分解成基础层（base）和细节层（detail），对基础层做对比度压缩，压缩之后的基础层再和原来的细节层相加，得到保留了细节信息的低动态范围图像。如下图

![](https://pic3.zhimg.com/v2-54679eb9a3a147ae61a92bddc4ab07da_1440w.jpg)

图来自 CMU 课程

两个问题：

1. 怎么分解成基础层和细节层
2. 怎么对基础层做压缩

### 分离 base 和 detail

思路比较多，比如分离低频和高频，认为低频是平滑过渡的部分，高频是灰度突变、纹理等信息，简单的比如使用高斯滤波，平滑过后的图像丢失了很多细节（细节是高频），平滑后的图像可以看成基础层；原图减去平滑图像，就是丢失的细节，可以看成细节层。

但这种方法得到的结果容易产生光晕，如下

![](https://picx.zhimg.com/v2-100b87748c0defc70d1d38ef02884a07_1440w.jpg)

高斯滤波分解的 tone-mapping 方法

出现这个的原因是高斯模糊会模糊掉边缘，原图减去这个平滑结果，得到的也是更平滑的细节（原来的细节处灰度过渡也更平滑），整个变化流程如下：

![](https://pic2.zhimg.com/v2-b9175ea48fb55b96f2126b0e65669b0b_1440w.jpg)

HDR 图

![](https://pica.zhimg.com/v2-abb114bf8636a03168b3fc95a540a89a_1440w.jpg)

HDR 图像的亮度图

![](https://pic4.zhimg.com/v2-0d509e68c2e94710376284952fa8d2c7_1440w.jpg)

亮度图取 log, 在 log 域做运算

![](https://picx.zhimg.com/v2-2716ea27c979bfb80bcca9fe2d2ce92d_1440w.jpg)

log 域的亮度图做高斯模糊的结果

![](https://pic3.zhimg.com/v2-fa6c4b3376cdbbc72ba8527481ff5f7e_1440w.jpg)

log 域的细节 = 原图 - 高斯模糊结果

在这里就可以看出，使用高斯模糊分离出的细节，本身就是有一些光晕了，后面只压缩基础层，再和细节层相加的结果自然会包括细节图的光晕。

进一步，可以通过减小高斯模糊的模糊程度（标准差），获得更少的光晕，如标准差为 1.0 时

![](https://pic2.zhimg.com/v2-166bef45f1dbb2ea14b0d254205fedcd_1440w.jpg)

高斯模糊标准差 1.0 的最终结果

可以发现，光晕消失了，但是细节被压缩的太厉害了，高动态的优势没体现出来，具体可以看下面细节层

![](https://pic2.zhimg.com/v2-b691c839f7c703b2695ad74cc9612199_1440w.jpg)

保留的细节层

可以看出，保留的细节很少很少，细节基本都被分到基础层，被压缩了，所以最后得到的结果虽然没有光晕，但细节其实丢失很严重！

所以目标就是，**平滑力度可以大一些，但是边缘不能被平滑了，所以可以考虑一系列保边平滑算法**，如双边滤波、引导滤波、[WLS 最小二乘滤波](https://zhida.zhihu.com/search?content_id=198412077&content_type=Article&match_order=1&q=WLS+%E6%9C%80%E5%B0%8F%E4%BA%8C%E4%B9%98%E6%BB%A4%E6%B3%A2&zhida_source=entity)等。

原论文使用的是双边滤波，使用了一些手段加速，大大降低了时间复杂度（双边滤波看起来不难，但是一系列优化算法还有很多研究，头大，日后整理），本次笔记对如何加速不做分析，还不会。

更改成双边滤波之后的结果如下：

![](https://pic3.zhimg.com/v2-bc99652fff3da6c6795dad09b83de51e_1440w.jpg)

双边滤波分离出的细节层

![](https://pic2.zhimg.com/v2-1291f97d841fd2c5c07b0c9c446726fd_1440w.jpg)

最终结果

和之前使用高斯滤波对比

![](https://pic3.zhimg.com/v2-3d730a93157c276fff243228c7087d16_1440w.jpg)

高斯模糊的最终结果

去除光晕还是很有效果的。

双边滤波的时间复杂度还很高  ，而且在这里处理 HDR 的 float 数据没法做查表优化，老老实实算 exp；但引导滤波可以做到  的时间复杂度，而且可以保留边缘，具体效果如下：

![](https://pic1.zhimg.com/v2-f114925a9a52455a0bf71fdbcbabb1b2_1440w.jpg)

引导滤波分离的细节层

![](https://picx.zhimg.com/v2-f6c837f57ff20817943f1113998af7f9_1440w.jpg)

引导滤波的最终结果

和双边滤波的效果差不多。换其他的保边平滑算法应该也有不错的效果。

### 压缩基础层

上面讲了分离基础层和细节层的一个滤波思路，接下来就是压缩基础层的动态范围。

具体压缩方法，原来的 HDR 图像

**HDR 图像 = 1.0 * 基础层 + 1.0 * 细节层**

压缩基础层，举个例子

**压缩的 HDR 图像 = 0.2 * 基础层 + 1.0 * 细节层**

压缩程度是一个人为定义的参数，上面假如是 0.2，在论文里的计算方法是给一个 contrast 参数，然后压缩参数由下式得到

其中  是基础层。

这里有个至关重要的细节，那就是在  域操作，一般是  。为什么？

[CMU HDR](https://link.zhihu.com/?target=http%3A//graphics.cs.cmu.edu/courses/15-463/lectures/lecture6.pdf) 给出的原因是

> Recall: humans are sensitive to multiplicative contrast  
> With log domain, our notion of “strong edge” always corresponds to the same contrast

大致意思，人对于同乘法比例的动态范围感知是一样的，比如动态范围 50: 10 和 50000: 10000，人眼对于这两种动态范围的感知是一样的，不会因为 10000：50000 很亮就认为这个动态范围更高。因此可以使用  函数将这种“量纲”变成常数。

我暂时不是很理解，我自己想的原因是，高动态范围很大，但是取  就能把灰度范围大大缩小，这样在计算双边滤波的值域之差时，避免出现  甚至更高的这种计算，在局部区域一旦出现这种权重”一遍倒“的现象，很容易产生噪声，就失去了双边滤波的效果了。

因此，压缩动态范围就变成了

**log10(压缩的 HDR 图像) = factor * log10(基础层) + 1.0 * log10(细节层)**

压缩结果取 pow(10, x) 就可得到压缩结果。

## 步骤

虽然上面的步骤看起来简单，但实现起来还是有很多不同。官方给出的步骤（我修改之后）如下

1. input intensity= 1/61*(R*20+G*40+B)
2. log(base) = Bilateral(log(input intensity))
3. log(detail) = log(input intensity)-log(base)
4. log (output intensity) = log(base) * compressionfactor + log(detail)
5. R output = R * **10^(log(output intensity)) / intensity**

第 1 步先根据 RGB 三通道的加权，得到一个亮度图，为什么是  我不知道；

第 2-3 步根据双边滤波，分离出基础层和线性层，都是在  域运算的，双边滤波有两个参数，值域标准差  取 0.4，空域标准差  取 0.02 * min(H, W)，倍数取 0.02 ~ 0.05 之间；

第 4 步压缩基础层，其中 compressionfactor 通过  得到，contrast 人为给定；

第 5 步，根据压缩后的 HDR 输出 / 亮度图的比例，r, g, b 三通道每个点对应等比例缩放——这么做的原因是，之前压缩的基础层都是在亮度图上操作的，求出每个点的压缩比例即可对 r, g, b 通道做统一变换，避免 r,g,b 不同步变换破坏白平衡；

上面压缩了 HDR 图像得到了一个相对动态范围更小的图像，但动态范围还不是 0-255，不方便显示设备显示，于是可以把数据标准化到 0-1，乘以 255 将数据转成 0-255 范围内，标准化直接除以一个最大值即可令值 ＜ 1，论文给的方式是

  
（疑问：除以 “基础层在  层的最大值和 factor 的乘积”，能保证小于 1 ？）

## 代码

C++ 代码主体，双边滤波使用的是最朴实的写法，没考虑官方的加速（日后补上）。

```cpp
// 一些辅助函数
namespace {

    void cv_show(const cv::Mat& one_image, const char* info="") {
        cv::imshow(info, one_image);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    bool cv_write(const cv::Mat& source, const std::string save_path) {
        return cv::imwrite(save_path, source, std::vector<int>({cv::IMWRITE_PNG_COMPRESSION, 0}));
    }

    cv::Mat make_pad(const cv::Mat& one_image, const int pad_H, const int pad_W) {
        cv::Mat padded_image;
        cv::copyMakeBorder(one_image, padded_image, pad_H, pad_H, pad_W, pad_W, cv::BORDER_REFLECT);
        return padded_image;
    }

    inline float _min(const float* data, const int length) {
        float min_value = data[0];
        for(int i = 1;i < length; ++i)
            if(data[i] < min_value) min_value = data[i];
        return min_value;
    }

    inline float _max(const float* data, const int length) {
        float max_value = data[0];
        for(int i = 1;i < length; ++i)
            if(data[i] > max_value) max_value = data[i];
        return max_value;
    }

    inline float square(const float x) {
        return x * x;
    }

    inline float clip(float x, const float low, const float high) {
        if(x < low) x = low;
        else if(x > high) x = high;
        return x;
    }

    inline double fast_exp(const double y) {
        double d;
        *(reinterpret_cast<int*>(&d) + 0) = 0;
        *(reinterpret_cast<int*>(&d) + 1) = static_cast<int>(1512775 * y + 1072632447);
        return d;
    }

}


// 高斯滤波
cv::Mat gaussi_filtering(const cv::Mat& origin, const float spatial_sigma=18) {
    // 收集图像信息
    const int H = origin.rows;
    const int W = origin.cols;
    const int C = origin.channels();
    assert(C == 1 and "only images of single channel is supported !");
    // 计算窗口半径等
    const int radius = int(3 * spatial_sigma);
    const int window_size = square(2 * radius + 1);
    // 对图像做 padding
    const auto padded_image = make_pad(origin, radius, radius);
    const int W2 = W + 2 * radius;
    // 准备一个空域模板
    int max_k = 0;
    std::vector<double> spatial_table(window_size);
    std::vector<int> offset(window_size, 0);
    const float sigma_inv = -0.5 / square(spatial_sigma);
    for(int i = -radius;i <= radius; ++i) {
        for(int j = -radius; j <= radius; ++j) {
            spatial_table[max_k] = fast_exp(double(sigma_inv * (i * i + j * j)));
            offset[max_k++] = i * W2 + j;
        }
    }
    // 准备一个结果
    cv::Mat result(H, W, CV_32F);
    float* const res_ptr = result.ptr<float>();
    int cnt = 0;
    // 求解每一个点
    for(int i = 0;i < H; ++i) {
        const float* const row_ptr = padded_image.ptr<float>() + (radius + i) * W2 + radius;
        for(int j = 0;j < W; ++j) {
            float sum_value = 0;
            float weight_sum = 0;
            for(int k = 0;k < max_k; ++k) {
                const float w = spatial_table[k];
                sum_value += w * row_ptr[j + offset[k]];
                weight_sum += w;
            }
            res_ptr[cnt++] = sum_value / weight_sum;
        }
    }
    return result;
}


// 双边滤波
cv::Mat bilateral_filtering(const cv::Mat& origin, const float range_sigma=0.4, const float spatial_sigma=18) {
    // 收集信息
    const int H = origin.rows;
    const int W = origin.cols;
    assert(origin.channels() == 1);
    // 求窗口大小
    const int radius = int(3 * spatial_sigma);
    const int window_size = radius * 2 + 1;
    // 对图像做 padding
    const auto padded_image = make_pad(origin, radius, radius);
    const int W2 = padded_image.cols;
    // 准备一个空域的模板(double 是因为 fast_exp 可以大大加快速度)
    std::vector<double> space_table(window_size * window_size);
    std::vector<int> space_offset(window_size * window_size);
    int max_k = 0;
    const double space_variance_2 = - 0.5 / (spatial_sigma * spatial_sigma);
    for(int i = -radius;i <= radius; ++i) {
	for(int j = -radius;j <= radius; ++j) {
	    space_table[max_k] = fast_exp(double(space_variance_2 * (i * i + j * j)));
	    space_offset[max_k] = i * W2 + j;
	    ++max_k;
	}
    }
    // 准备值域的
    const float sigma_inv = 0.5f / (range_sigma * range_sigma);
    // 准备一个结果
    cv::Mat result(H, W, CV_32F);
    float* const res_ptr = result.ptr<float>();
	// 开始滤波
	int cnt = 0; // 存放每次的加权结果
	for(int i = 0;i < H; ++i) {
	    // 取出当前滤波的这一行, 在 pad 图像中的行指针, 第 radiu + i 行, 偏移 radius 个像素
	    const float* const pad_ptr = padded_image.ptr<float>() + (radius + i) * W2 + radius;
	    for(int j = 0;j < W; ++j) {
	        const float center = pad_ptr[j];
	        // 遍历窗口
	        float intensity_sum = 0;
	        float weight_sum = 0;
	        for(int k = 0;k < max_k; ++k) {
	            const float neighbor = pad_ptr[j + space_offset[k]];
	            const float w = space_table[k] * fast_exp(double(-sigma_inv * square(neighbor - center)));;
	            intensity_sum += neighbor * w;
	            weight_sum += w;
	        }
            res_ptr[cnt++] = intensity_sum / weight_sum;
	    }
	}
    return result;
}


std::list<std::pair<std::string, cv::Mat> >
        bilateral_local_tonemapping(const cv::Mat& hdr_image, const float contrast_value=10) {
    // 收集中间结果
    std::list<std::pair<std::string, cv::Mat> > collections;
    // 获取图像信息
    const int H = hdr_image.rows;
    const int W = hdr_image.cols;
    const int C = hdr_image.channels();
    assert(C == 3 and "only BGR channels are supported!");
    const float hdr_min = std::max(_min(hdr_image.ptr<float>(), H * W * C), 1e-6f);
    const float hdr_max = _max(hdr_image.ptr<float>(), H * W * C);

    std::cout << "输入的高动态范围图像信息如下 : \n";
    std::cout << "\theight = " << hdr_image.rows << "\n\twidth = " << hdr_image.cols << "\n";
    std::cout << "\tdepth =  " << hdr_image.type() << std::endl;
    std::cout << "\tMax = " << hdr_max << "\n\tMin = " << hdr_min << std::endl;
    std::cout << "\t动态范围 = " << hdr_max / hdr_min << std::endl;

    // 获取 hdr 图像指针
    const float* const hdr_ptr = hdr_image.ptr<float>();

    // 先求亮度图 intensity = (20 * R + 40 * G + 1 * B) / 61;
    const int length = H * W;
    cv::Mat intensity(H, W, CV_32F);
    float* const intensity_ptr = intensity.ptr<float>();
    for(int i = 0;i < length; ++i) {
        const int p = 3 * i;
        intensity_ptr[i] = (20 * hdr_ptr[p + 2] + 40 * hdr_ptr[p + 1] + hdr_ptr[p]) / 61.f;
    }
    collections.emplace_back("intensity", intensity);

    // 计算 log10(intensity), 在 log 域很重要,
    cv::Mat log_intensity(H, W, CV_32F);
    float* const log_intensity_ptr = log_intensity.ptr<float>();
    for(int i = 0;i < length; ++i)
        log_intensity_ptr[i] = std::log10(intensity_ptr[i]);
    collections.emplace_back("intensity_log", log_intensity);

    // 对 log_intensity 做双边滤波, 得到更平滑的亮度图(base 层)
    const float range_sigma = 0.4;
    const float spatial_sigma = 0.02f * std::min(H, W);
    std::cout << "值域标准差 = " << range_sigma << "\n空域标准差 = " << spatial_sigma << std::endl;
    auto log_base = bilateral_filtering(log_intensity, range_sigma, spatial_sigma);
//    auto log_base = gaussi_filtering(log_intensity, spatial_sigma);
//    auto log_base = guided_filter_with_gray(log_intensity, log_intensity, 3 * spatial_sigma, 3 * spatial_sigma, 0.1);

    // 求 log_detail, 原 log 亮度图 - 平滑过后的亮度图(base 层) = 细节层(log)
    cv::Mat log_detail = log_intensity - log_base;
    collections.emplace_back("base", log_base);
    collections.emplace_back("detail", log_detail);

    // 压缩 base 层的对比度
    // 原来是 1.0 * base + 1.0 * detail
    // 现在假设变成 0.2 * base + 1.0 * detail
    const float log_base_max = _max(log_base.ptr<float>(), length);
    const float log_base_min = _min(log_base.ptr<float>(), length);
    const float factor = std::log10(contrast_value) / (log_base_max - log_base_min);
    std::cout << "Base 层的对比度缩放因子 = " << factor << std::endl;

    cv::Mat log_fusion = factor * log_base + log_detail;
    float* const fusion_ptr = log_fusion.ptr<float>();
    for(int i = 0;i < length; ++i)
        fusion_ptr[i] = std::pow(10.0, fusion_ptr[i]);

    // 准备一个结果, 三通道, 存放 float 数据
    cv::Mat result(H, W, CV_32FC3);
    float* const result_ptr = result.ptr<float>();

    // 计算每个点在亮度通道 intensity 压缩之后的比例大小, 然后 R, G, B 等比例缩放
    for(int i = 0;i < length; ++i) {
        const float ratio = fusion_ptr[i] / intensity_ptr[i];
        const int pos = 3 * i;  // 找到每个点的坐标
        result_ptr[pos + 2] = hdr_ptr[pos + 2] * ratio;
        result_ptr[pos + 1] = hdr_ptr[pos + 1] * ratio;
        result_ptr[pos] = hdr_ptr[pos] * ratio;
    }
    collections.emplace_back("compressed", result.clone());

    // 现在已经压缩了对比度, 尽可能保留了细节

    // 计算新的动态范围
    const float new_hdr_max = _max(result_ptr, length * 3);
    const float new_hdr_min = std::max(1e-6f, _min(result_ptr, length * 3));
    std::cout << "压缩之后的动态范围 = " << new_hdr_max / new_hdr_min << std::endl;

    // 在动态范围内, 将数据标准化到 0-1 或者 0-255, 方便显示器显示
    const float max_scale = std::pow(10.f, log_base_max * factor);
    // 截断函数
    auto normalize = [max_scale](float x) -> float {
        return clip(255 * x / max_scale, 0, 255);
    };
    for(int i = 0;i < length; ++i) {
        const int p = 3 * i;
        result_ptr[p + 2] = normalize(result_ptr[p + 2]);
        result_ptr[p + 1] = normalize(result_ptr[p + 1]);
        result_ptr[p] = normalize(result_ptr[p]);
    }
    // 数据从 float -> uchar,
    result.convertTo(result, CV_8UC3);
    collections.emplace_back("result", result);
    return collections;
}


int main() {
    // 读取图像
    cv::Mat hdr_image = cv::imread("./images/input/vinesunset_2.hdr", cv::IMREAD_ANYDEPTH);

    cv_show(hdr_image);

    // 直接处理
    auto collections = bilateral_local_tonemapping(hdr_image, 10);

    // 保存
    std::string save_dir("./images/output/memorial_");
    for(const auto& item : collections) {
        cv_show(item.second, item.first.c_str());
        cv_write(item.second, save_dir + item.first + ".png");
    }
    return 0;
}
```

以下图为例

![](https://pic1.zhimg.com/v2-8f368f3768f26a14ee3c34b1a5589220_1440w.jpg)

HDR 原图

![](https://pica.zhimg.com/v2-299066d8804fedfd6a601102664e3002_1440w.jpg)

中间输出

可以看出，动态范围降低了不少，

![](https://pic2.zhimg.com/v2-c546e5069de2521fd41e81349661e463_1440w.jpg)

最终结果

![](https://pic2.zhimg.com/v2-d3391608a401e7ff8e2bf1c464556f05_1440w.jpg)

Base 层

![](https://pic4.zhimg.com/v2-f9671ca577e80ef464b81824f74fbde7_1440w.jpg)

细节层

![](https://pica.zhimg.com/v2-b6366435623a3283ec49d190048f8b18_1440w.jpg)

标准化到 0-255 之前

## 实验

该算法主要的几个参数是对比度、双边滤波的值域标准差和空域标准差。

### 对比度

对比度参数的主要应用是下面三个公式

**log (output intensity) = log(base) * factor + log(detail)**

**R output = R * 10^(log(output intensity)) / intensity**

随着 contrast 对比度的增大，factor 越大，基础层被压缩的程度减小，因此图像标准化之前应该是动态范围越大

![](https://pica.zhimg.com/v2-fb90f71a337a8661304ae7eb7603da5c_1440w.jpg)

对比度 3，标准化之前的动态范围 76

![](https://picx.zhimg.com/v2-aaf6fee465f829b944def441ef4fa26b_1440w.jpg)

对比度 10，标准化之前的动态范围 134

![](https://pic3.zhimg.com/v2-70bbfd0c3c7cfe25856368699fe09a58_1440w.jpg)

对比度 50，标准化之前的动态范围 464

### 值域标准差

保持其它参数不变，更改双边滤波的值域的标准差，可以预见，值域标准差越大，平滑地越狠，越有可能产生光晕；相反，值域标准差越小，平滑程度越小，细节层保留地越少（都分到基础层去了）。

![](https://pic4.zhimg.com/v2-49f5a29ac9a900683edf245a6b1ee269_1440w.jpg)

值域标准差 0.1

![](https://picx.zhimg.com/v2-0712ae9ed348827b2d704a04a111e76d_1440w.jpg)

值域标准差 0.4

![](https://pic3.zhimg.com/v2-887fa02b2dbf74e4b86ebadaa27ea3b6_1440w.jpg)

值域标准差 1.0

### 空域标准差

和之前值域标准差基本一致。但等倍数产生的效果没有值域标准差那么明显，算法对这个参数不是很敏感。

![](https://pic3.zhimg.com/v2-f33e85bfbdfa47b778180e3025480daa_1440w.jpg)

空域标准差 0.005 * min*(H, W)

![](https://picx.zhimg.com/v2-0712ae9ed348827b2d704a04a111e76d_1440w.jpg)

空域标准差 0.02 * min*(H, W)

![](https://pic1.zhimg.com/v2-7558890d401245d051dfca35067ecf2e_1440w.jpg)

空域标准差 0.10 * min*(H, W)

## 问题

速度慢，噪声。

速度慢指的是我写的双边滤波垃圾，复杂度高，官方用快速傅里叶变换写的其实很快，秒出，我的开了 O2 都得跑个七八秒。

噪声的话，因为双边滤波在平滑的过程中，把噪声也给平滑了，然后原图 - 平滑结果得到细节层，被平滑的噪声也被分到了细节层，细节层不做去噪的话，就很容易得到噪声，如下：

![](https://pic1.zhimg.com/v2-7f29b3dbd36762b1389fea936a537914_1440w.jpg)

结果

![](https://pica.zhimg.com/v2-d1d4ab73b4b3eae3fbef594cd51d0eac_1440w.jpg)

噪声

![](https://pic1.zhimg.com/v2-82df640cdffe0536a140711ae6199552_1440w.jpg)

细节层放大 20 倍

可以看见，放大之后，细节层存在不少的噪声。后续可以考虑用 BM3D 等手段做进一步的优化。

## 改进

waiting

## 图片

测试图像来源于 [https://ict.usc.edugraphics/HDRShop/](https://link.zhihu.com/?target=https%3A//ict.usc.edugraphics/HDRShop/) ，但我访问不了，之前还好好的。

「vinesunset_2.hdr」[https://www.aliyundrive.com/s/WquGv](https://link.zhihu.com/?target=https%3A//www.aliyundrive.com/s/WquGv41qKSG)