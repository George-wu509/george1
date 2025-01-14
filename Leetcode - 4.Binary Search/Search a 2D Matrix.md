
**样例 1：**
输入：
```
矩阵 = [[5]]
target = 2
```
输出：
```
false
```
解释：
矩阵中没有包含2，返回false。

**样例 2：**
输入：
```
矩阵 = [
  [1, 3, 5, 7],
  [10, 11, 16, 20],
  [23, 30, 34, 50]
]
target = 3
```
输出：
```
true
```
解释：
矩阵中包含3，返回true。


```python
    def search_matrix(self, matrix: List[List[int]], target: int) -> bool:
        if not matrix or not target:
            return False
        col = len(matrix[0])
        start, end = 0, len(matrix)*col-1

        while start+1 < end:
            mid = start + (end-start)//2
            value = matrix[mid//col][mid%col]
            if value > target:
                end = mid
            elif value < target:
                start = mid
            else:
                start = mid
        
        if matrix[start//col][start%col] == target:
            return True
        if matrix[end//col][end%col] == target:
            return True
        return False
```
pass