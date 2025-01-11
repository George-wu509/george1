
**样例 1:**
```
输入 : n = 10, 我选择了 4 (但是你不知道)
输出 : 4
```


```python
class Solution:
    def guessNumber(self, n):
        l = 1
        r = n
        while l <= r:
            mid = l + (r - l) // 2
            res = Guess.guess(mid)
            if res == 0:
                return mid
            if res == -1:
                r = mid - 1
            if res == 1:
                l = mid + 1
        return -1
```
pass