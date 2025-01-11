
```python
输入: [3, 4, 6, 7]
输出: 3
解释:
可以组成的是 (3, 4, 6), 
           (3, 6, 7),
           (4, 6, 7)
```

```python
输入: [4, 4, 4, 4]
输出: 4
解释:
任何三个数都可以构成三角形
所以答案为 C(3, 4) = 4
```



```python
    def triangleCount(self, S):
        S.sort()
        
        ans = 0
        for i in range(len(S)):
            left, right = 0, i - 1
            while left < right:
                if S[left] + S[right] > S[i]:
                    ans += right - left
                    right -= 1
                else:
                    left += 1
        return ans

```
pass