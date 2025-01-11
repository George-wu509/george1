
_**样例 1**_
```
输入 : [-1, -2, -3, 4, 5, 6]
输出 : [-1, 5, -2, 4, -3, 6]
解释 : 或者任何满足条件的答案 
```


```python
class Solution:
    """
    @param: a: an integer array.
    @return: nothing
    """
    def rerange(self, a):
        pos, neg = 0, 0
        for num in a:
            if num > 0:
                pos += 1
            else:
                neg += 1
        
        self.partition(a, pos > neg)
        self.interleave(a, pos == neg)
            
    def partition(self, a, start_positive):
        flag = 1 if start_positive else -1
        left, right = 0, len(a) - 1
        while left <= right:
            while left <= right and a[left] * flag > 0:
                left += 1
            while left <= right and a[right] * flag < 0:
                right -= 1
            if left <= right:
                a[left], a[right] = a[right], a[left]
                left += 1
                right -= 1
    
    def interleave(self, a, has_same_length):
        left, right = 1, len(a) - 1
        if has_same_length:
            right = len(a) - 2
            
        while left < right:
            a[left], a[right] = a[right], a[left]
            left, right = left + 2, right - 2
```
pass