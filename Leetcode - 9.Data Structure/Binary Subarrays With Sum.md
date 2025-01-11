
**样例 1:**
```
输入：A = [1,0,1,0,1], S = 2
输出：4
解释：
如下面黑体所示，有 4 个满足题目要求的子数组：
[1,0,1]
[1,0,1]
[1,0,1,0]
[0,1,0,1]
```
**样例 2:**
```
输入：A = [0,0,0,0,0,0,1,0,0,0], S = 0
输出：27
解释：
和为 S 的子数组有 27 个
```


```python
    def num_subarrays_with_sum(self, a: List[int], s: int) -> int:
        n = len(a)
        left1, left2, right = 0, 0, 0
        sum1, sum2 = 0, 0
        ret = 0
        while right < n:
            sum1 += a[right]
            while left1 <= right and sum1 > s:
                sum1 -= a[left1]
                left1 += 1
            sum2 += a[right]
            while left2 <= right and sum2 >= s:
                sum2 -= a[left2]
                left2 += 1
            ret += left2 - left1
            right += 1
        return ret
```
pass