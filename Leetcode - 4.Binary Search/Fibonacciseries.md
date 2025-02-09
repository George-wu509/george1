Lintcode 366

查找斐波纳契数列中第 N 个数。（N 从 1 开始）
所谓的斐波纳契数列是指：
- 前2个数是 0 和 1 。
- 第 _i_ 个数是第 _i_-1 个数和第_i_-2 个数的和。
斐波纳契数列的前10个数字是：
`0, 1, 1, 2, 3, 5, 8, 13, 21, 34 ...`

所以在function裡create的result = [0, 0, 1, 1, 2, 3, 5....]
fib[1]=0, fib[2]=1, fib[3]=1, fib[4]=2, ..

```python
"""
样例  1:
	输入:  1
	输出: 0
	
	样例解释: 
	返回斐波那契的第一个数字，是0.

样例 2:
	输入:  2
	输出: 1
	
	样例解释: 
	返回斐波那契的第二个数字是1.
```


```python
"""
class Solution:
    def dfs(self, n, fib):
        if fib[n] != -1:
            return fib[n]
        if n <= 2:
            fib[n] = n - 1
            return fib[n]
        fib[n] = self.dfs(n - 1, fib) + self.dfs(n - 2, fib)
        return fib[n]
    
    def fibonacci(self, n):
        result = [-1] * (n + 1)
        self.dfs(n, result)
        return result[n]
```
pass



# Solutions

[https://www.lintcode.com/problem/366/solution/17953](https://www.lintcode.com/problem/366/solution/17953)

纯用递归会超时，如果用带有记忆化的递归就可以，使用循环和记忆化递归的时间复杂度一样，都是�(�)_O_(_n_)。

不超出Integer的斐波那契数很少，仅有50个左右。但是使用纯的递归，复杂度会是�(2�)_O_(2_n_)。因此会超时。

class Solution: def fibonacci(self, n): a = 0 b = 1 for i in range(n - 1): a, b = b, a + b return a

[https://www.lintcode.com/problem/366/solution/18282](https://www.lintcode.com/problem/366/solution/18282)

```python
class Solution:
2    def fibonacci(self, n):
3        fib = [0, 0, 1]
4        for i in range(3, n + 1, 1):
5            fib.append(fib[i - 1] + fib[i - 2])
6        return fib[n]
```

这道题可以说是十分经典的入门题目，但是却有很多种做法。

刚开始学习编程语言的同学应该会用一个for循环来解决这个问题。

逐渐地学会递归后，这道题也可以用递归来解，但是效率却太差了，所以需要使用记忆化搜索来优化。

最后我们可能会遇到一种求下标很大的fibonacci数并取模数，例如下面这道题就让我们求得fibonacci数列的第1,000,000,000项的后四位。

[https://www.lintcode.com/problem/fibonacci-ii/description](https://www.lintcode.com/problem/fibonacci-ii/description)

在看完了本篇题解之后，我相信这道进阶版的题目也不会难住你。

### **算法一：递推法**

这种算法的朴素写法是所有人都可以写出来的，因此不做赘述。

它的时间复杂度和空间复杂度均为O(n)。

下面的三份代码依次为Java、C++、Python。

```python

class Solution:
    def fibonacci(self, n):
        fib = [0, 0, 1]
        for i in range(3, n + 1, 1):
            fib.append(fib[i - 1] + fib[i - 2])
        return fib[n

```

但是我们发现，这道题并不需要存储那么多的fibonacci数，因为是返回第n项，并且第n项只和前面的两个数字有关，所以利用一个长度为2的空间记录前两个数字即可。

此时时间复杂度不变，但是空间复杂度降为O(1)。

这种节省空间的方法其实就是动态规划的滚动数组思想。

下面的三份代码依次为Java、C++、Python。

```python
class Solution:
    def fibonacci(self, n):
        fib = [0, 1]
        for i in range(2, n + 1, 1):
            fib[i % 2] = fib[0] + fib[1]
        return fib[(n + 1) % 2]
```

### **算法二：递归法**

这道题是我们学递归时一定会学到的例题，因为`fibonacci[i] = fibonacci [i - 1] + fibonacci[i - 2]`，故递归式为：`thisFibonbacci = dfs(i) + dfs(i - 1)`。

但是这么做的时间复杂度难以接受，因为有很多被重复计算的数字，比如我们在求解fib(10)的时候，会找到fib(9)和fib(8)共两个，然后下一层会是fib(8)和fib(7)，fib(7)和fib(6)共四个。这是一个呈指数增长的曲线，其底数为2，是稳定超时的代码。

时间复杂度为O(2^n)，空间复杂度为O(n)（不考虑递归的栈空间占用则为O(1)）。

下面的三份代码依次为Java、C++、Python。

```python
class Solution:
    def dfs(self, n):
        if n <= 2:
            return n - 1
        return self.dfs(n - 1) + self.dfs(n - 2)
    
    def fibonacci(self, n):
        return self.dfs(n)

```

这时候就要用到一种经常被用来以空间换时间的优化算法——记忆化搜索。

顾名思义，它将计算出的结果存储下来，在计算到指定值的时候，先判断这个值是否已经计算过，若没有，才进行计算，否则读取已经存储下来的值。这样就把一个指数级复杂度变成了线性复杂度，代价是空间复杂度从常数级上升至线性级。

时间复杂度为O(n)，空间复杂度为O(n)。

下面的三份代码依次为Java、C++、Python。

```python
class Solution:
    def dfs(self, n, fib):
        if fib[n] != -1:
            return fib[n]
        if n <= 2:
            fib[n] = n - 1
            return fib[n]
        fib[n] = self.dfs(n - 1, fib) + self.dfs(n - 2, fib)
        return fib[n]
    
    def fibonacci(self, n):
        result = [-1] * (n + 1)
        self.dfs(n, result)
        return result[n]

```

### **算法三：矩阵快速幂**

先修知识1：快速幂：[https://www.lintcode.com/problem/fast-power/description](https://www.lintcode.com/problem/fast-power/description)

先修知识2：矩阵的乘法运算原理。

在先修知识掌握之后，我们不禁要问：

为什么求一个fibonacci还能和矩阵、求幂扯上关系？

我们首先来看一个例子：

假设我们有一个2_2的矩阵[[1,1],[1,0]]和一个2_1的矩阵[[2],[1]]，将上面两个矩阵相乘会变成[[3],[2]]对吧，再用[[1,1][1,0]]和新的矩阵[[3],[2]]继续相乘又会变成[[5],[3]]，继续运算就是[[8],[5]]，[[13],[8]]......神奇的事情出现了，当我们不断地用这个[[1,1],[1,0]]乘上初始矩阵，得到的新矩阵的上面一个元素就会变成的fibonacci数列中的一个数字，下面的元素则是上面元素的前一项，而且每多乘一次，这个数字的下标就增加一。

那么这个矩阵是怎么来的呢？

从刚才的推理中我们发现：某个矩阵A乘上[[fib(n+1)],[fib(n)]]会变成[[fib(n+2)],[fib(n+1)]]。现在设矩阵A为[[a,b],[c,d]]，（为什么矩阵A是一个2_2的矩阵？因为只有2_2的矩阵乘一个2_1的矩阵才会得到一个2_1的矩阵。）那么可以列出下面的等式：

a * fib(n + 1) + b * fib(n) = fib(n + 2)

c * fib(n + 1) + d * fib(n) = fib(n + 1)

很容易地，我们得到：

1 * fib(n + 1) + 1 * fib(n) = fib(n + 2)

1 * fib(n + 1) + 0 * fib(n) = fib(n + 1)

也就是说矩阵A就是[[1,1],[1,0]]。

现在我们知道了原矩阵M连续多次乘上某个矩阵A会得到新的矩阵M'，并且M'的第一个元素就是我们想要的值。

根据矩阵的运算法则，中间的若干次相乘可以先乘起来，但是矩阵乘法的复杂度是O(n^3)，是不是一次一次的乘有点慢呢？这时候就是快速幂出场的时候了，我们可以使用快速幂来优化矩阵乘法的速度，这就是矩阵快速幂算法。

值得注意的是，在快速幂中，我们有一步操作是：int result = 1。那么如何使用矩阵来实现这个单位1呢，我们要借助单位矩阵。所谓的单位矩阵是一个从左上角到右下角对角线上都是1，其余位置都是0的边长相等的矩阵（方阵）。比如[[1,0,0],[0,1,0],[0,0,1]]。单位矩阵E的特性在于满足矩阵乘法的任意矩阵A_E一定等于A，E_A一定等于A。

所以本题需要将初始矩阵设置为[[1],[0]]，这样我们只需要将中间矩阵[[1,1],[1,0]]使用快速幂连乘n-2次，再和[[1],[0]]相乘，矩阵就变成了[[fib(n)],[fib(n-1)]]。

矩阵快速幂算法常常被应用在递推式的加速中，可以很轻松的递推至下标相当大的位置，而不用担心超时的问题。

但是要注意以下两点：

第一，矩阵快速幂使用的过程中要注意是否应该取模，因为C++和Java会有数值溢出，如果题目要求递推式取模，那么有很大概率是一道矩阵快速幂题目。

第二，矩阵乘法是没有交换律的（A_B ≠ B_A），因此我们一定要注意乘法顺序。

因为矩阵乘法的复杂度是矩阵长度 L 的三次方，需要乘logn次。所占的空间一般只有矩阵的空间，为L的平方。

因此时间复杂度为O(L^3*logn)，空间复杂度为O(L^2)。