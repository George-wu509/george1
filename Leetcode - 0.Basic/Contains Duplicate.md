
**样例 1：**
```
输入：nums = [1, 1]
输出：True
```
**样例 2：**
```
输入：nums = [1, 2, 3]
输出：False
```


```python
    def contains_duplicate(self, nums: List[int]) -> bool:
        s = set()
        for x in nums:
            if x in s: return True
            s.add(x)
        return False
```
pass