
<mark style="background: #FFB86CA6;">1D Gaussian Function </mark>

$\Huge g(x) = \frac{1}{\sigma\sqrt{2 \pi}}e^ \frac{-(x-\mu)^2}{2\sigma^2}$

$\sigma$ : standard  deviation, $\mu$ : mean


<mark style="background: #FFB86CA6;">2D Gaussian Function</mark>

$\Huge g(x,y) = (\frac{1}{\sigma^2 2\pi}) e^-(\frac{(x-\mu_x)^2}{2\sigma_x^2}+\frac{(x-\mu_y)^2}{2\sigma_y^2})$


<mark style="background: #FFB86CA6;">2D Gaussian Filtering</mark> 

$\Huge g(x,y) = (\frac{1}{\sigma^2 2\pi}) e^-(\frac{x^2 + y^2}{2\sigma^2})$



<mark style="background: #BBFABBA6;">1D Fourier transform</mark>

$\large X(\omega)=\int_{-\infty}^\infty x(t) e^{-j \omega t}dt$


<mark style="background: #BBFABBA6;">2D Fourier transform</mark>

$\large F(u,v)=\int_{-\infty}^\infty\int_{-\infty}^\infty f(x,y) e^{-j2\pi (ux+vy)}dxdy$


<mark style="background: #ADCCFFA6;">1D continuous wavelet transform</mark>

$\large F_\psi(a,b)=\frac{1}{\sqrt{a}}\int_{-\infty}^\infty f(t) \psi(\frac{t-b}{a}) dt$

$F_\psi(a,b)$ :  在尺度 a 和位移 b 下的 1D CWT 系數
 f(t) 是原始信號。
 $\psi(t)$ 是母小波 (mother wavelet)，它是一個有限支持的或快速衰減的波形。
 a是尺度參數，控制小波的擴展或壓縮 (即頻率)。
 b 是位移參數，控制小波在信號上移動的時間點。
$\frac{1}{\sqrt{a}}$是歸一化因子，確保小波在不同尺度上的能量保持一致。


<mark style="background: #ADCCFFA6;">2D continuous wavelet transform</mark>

$\large F(u,v)=\frac{1}{a} \int_{-\infty}^\infty\int_{-\infty}^\infty f(x,y) \psi(\frac{x-b_1}{a}, \frac{y-b_2}{a}) dxdy$

$F_\psi(a,b)$ :  在尺度 a 和位移 b 下的 1D CWT 系數
 f(t) 是原始信號。
 $\psi(t)$ 是母小波 (mother wavelet)，它是一個有限支持的或快速衰減的波形。
 a是尺度參數，控制小波的擴展或壓縮 (即頻率)。
 b 是位移參數，控制小波在信號上移動的時間點。
$\frac{1}{\sqrt{a}}$是歸一化因子，確保小波在不同尺度上的能量保持一致。