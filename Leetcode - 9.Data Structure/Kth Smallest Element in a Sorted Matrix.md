

**样例1**
```
输入：
matrix = [
  [1, 5, 9],
  [10, 11, 13],
  [12, 13, 15]
]
k = 8
输出：
13
```
**样例2**
```
输入：
matrix = [
  [-5]
]
k = 1
输出：
-5
```



```python
class Solution:
    
    def kthSmallest(self, matrix, k) -> int:
        
        m, n = len(matrix), len(matrix[0])
        
        lo, hi = matrix[0][0], matrix[m-1][n-1]
        while lo + 1 < hi:
            mid = lo + (hi - lo) // 2
            count = self.num_of_less_or_equal(matrix, mid)
            if count < k:
                lo = mid 
            else:
                hi = mid 
        
        if self.num_of_less_or_equal(matrix, lo) >= k:
            return lo 
        
        return hi
            
            
    def num_of_less_or_equal(self, matrix, target):
        
        m, n = len(matrix), len(matrix[0])
        i, j = m-1, 0
        count = 0
        while i >=0 and j < n:
            if matrix[i][j] <= target:
                count += i+1 
                j += 1 
            else:
                i -= 1 
        return count
```
pass