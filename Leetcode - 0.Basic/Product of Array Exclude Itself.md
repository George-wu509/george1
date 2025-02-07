Lintcode 50
给定一个整数数组`A`。  
定义$∗A[i+1]∗...∗A[n−1]$， 计算`B`的时候请不要使用除法。请输出`B`。

**样例 1：**
输入：
```python
#A = [1,2,3]
```
输出：
```python
#[6,3,2]
```
解释：
B[0] = A[1] * A[2] = 6; B[1] = A[0] * A[2] = 3; B[2] = A[0] * A[1] = 2

**样例 2：**
输入：
```python
#A = [2,4,6]
```
输出：
```python
#[24,12,8]
```
解释：
B[0] = A[1] * A[2] = 24; B[1] = A[0] * A[2] = 12; B[2] = A[0] * A[1] = 8



```python
class Solution:
    def productExcludeItself(self, nums):
        length ,B  = len(nums) ,[]
        f = [ 0 for i in range(length + 1)]
        f[ length ] = 1
        for i in range(length - 1 , 0 , -1):
            f[ i ] = f[ i + 1 ] * nums[ i ]
        tmp = 1
        for i in range(length):
            B.append(tmp * f[ i + 1 ])
            tmp *= nums[ i ]
        return B
```
pass

解釋:
step1