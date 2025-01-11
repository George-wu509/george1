
**示例 1:**
```
输入:
A = [6, 1, 4, 6, 3, 2, 7, 4]
K = 3
L = 2
输出: 
24
解释：
因为Alice 可以选择3-5颗树然后摘到4 + 6 + 3 = 13 个苹果， Bob可以选择7-8棵树然后摘到7 + 4 = 11个苹果，因此他们可以收集到13 + 11 = 24。
```
**示例 2:**
```
输入:
A = [10, 19, 15]
K = 2
L = 2
输出: 
-1
解释：
因为对于 Alice 和 Bob 不能选择两个互不重合的区间。
```


```python
class Solution:
    def pick_apples(self, a: List[int], k: int, l: int) -> int:
        n = len(a)
        if n < k + l:
            return -1
        
        prefix_sum = a[:]
        # 计算前缀和
        for i in range(1, n):
            prefix_sum[i] += prefix_sum[i - 1]
            
        # prefixK 代表前 i 个数中，长度为 K 的子区间和的最大值
        prefix_k = [0] * n
        prefix_l = [0] * n
        
        # postfixK 代表后面 [i, n - 1] 中，长度为 k 的子区间和的最大值
        postfix_k = [0] * n
        postfix_l = [0] * n
        
        # 计算前缀值
        for i in range(n):
            if i == k - 1:
                prefix_k[i] = self.range_sum(prefix_sum, i - k + 1, i)
            elif i > k - 1:
                prefix_k[i] = max(self.range_sum(prefix_sum, i - k + 1, i), prefix_k[i - 1])
            if i == l - 1:
                prefix_l[i] = self.range_sum(prefix_sum, i - l + 1, i)
            elif i > l - 1:
                prefix_l[i] = max(self.range_sum(prefix_sum, i - l + 1, i), prefix_l[i - 1])
        
        # 计算后缀值
        for i in range(n - 1, -1, -1):
            if i + k - 1 == n - 1:
                postfix_k[i] = self.range_sum(prefix_sum, i, i + k - 1)
            elif i + k - 1 < n - 1:
                postfix_k[i] = max(self.range_sum(prefix_sum, i, i + k - 1), postfix_k[i + 1])
            if i + l - 1 == n - 1:
                postfix_l[i] = self.range_sum(prefix_sum, i, i + l - 1)
            elif i + l - 1 < n - 1:
                postfix_l[i] = max(self.range_sum(prefix_sum, i, i + l - 1), postfix_l[i + 1])
        
        result = 0
        # 枚举分界点，计算答案
        for i in range(i, n - 1):
            result = max(result, prefix_k[i] + postfix_l[i + 1])
            result = max(result, prefix_l[i] + postfix_k[i + 1])
        
        return result
        
        
    def range_sum(self, prefix_sum, l, r):
        if l == 0:
            return prefix_sum[r]
        return prefix_sum[r] - prefix_sum[l - 1]
```
pass