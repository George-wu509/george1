
**样例 1:**
```
输入: [-3, 1, 3, -3, 4]
输出: [1, 4]
```
**样例 2:**
```
输入: [0, 1, 0, 1]
输出: [0, 3]
解释: 字典序最小.
```


```python
    def continuous_subarray_sum(self, a):
        ans = -0x7fffffff
        sum = 0
        start, end = 0, -1
        result = [-1, -1]
        for x in a:
            if sum < 0:
                sum = x
                start = end + 1
                end = start
            else:
                sum += x
                end += 1
            if sum > ans:
                ans = sum
                result = [start, end]

        return result
```
pass