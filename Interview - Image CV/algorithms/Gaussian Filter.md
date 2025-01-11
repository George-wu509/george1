



Ref1: [[Python]Gaussian Filter-概念與實作](https://medium.com/@bob800530/python-gaussian-filter-%E6%A6%82%E5%BF%B5%E8%88%87%E5%AF%A6%E4%BD%9C-676aac52ea17)

## 3. Gaussian Filter數學函式

實際的**Gaussian Filter**由此方程式產生

![](https://miro.medium.com/v2/resize:fit:331/1*2xnfNXq_unnHzOQHJgpP3A.png)

Gassian Function

簡化計算，假設sigma = 0.707(根號0.5)

![](https://miro.medium.com/v2/resize:fit:305/1*RJpQtrtRrOy2C1bvnZeYZA.png)

Gassian Function, sigma = 1

Gaussian Filter的中心點(x,y)須為(0,0)，下例為一個3*3的(x,y)值矩陣

![](https://miro.medium.com/v2/resize:fit:324/1*xS_sqenLZFDuGAdST2mIpw.png)

Gassian Function, sigma = 1

將此矩陣的x,y值套入**Gaussian Function**並**正規化**後就可以得到3*3的**Gaussian filter**了!
計算過程請點此[**連結**](https://docs.google.com/spreadsheets/d/1HDosoi3RrPLLogbDzuii-rSVCQZs1XLkuVzP_8bR-J0/edit?usp=sharing)

![](https://miro.medium.com/v2/resize:fit:321/1*jT161zGZgYUg-hp-Nk88iQ.png)


1D Gaussian Function 

$\Huge g(x) = \frac{1}{\sigma\sqrt{2 \pi}}e^ \frac{-(x-\mu)^2}{2\sigma^2}$

$\sigma$ : standard  deviation, $\mu$ : mean

2D Gaussian Function 

$\Huge g(x,y) = (\frac{1}{\sigma\sqrt{2 \pi}}) e^-((\frac{(x-\mu_x)^2}{2\sigma_x^2}+\frac{(x-\mu_y)^2}{2\sigma_y^2}))$




