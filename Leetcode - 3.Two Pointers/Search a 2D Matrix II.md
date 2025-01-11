
**样例 1：**
输入：
```
矩阵 = [[3,4]]
target = 3
```
输出：
```
1
```
解释：
矩阵中只有1个3。

**样例 2：**
输入：
```
矩阵 = [
      [1, 3, 5, 7],
      [2, 4, 7, 8],
      [3, 5, 9, 10]
    ]
target = 3
```
输出：
```
2
```
解释：
矩阵中有2个3。


```python
    def search_matrix(self, matrix, target):
        if not matrix or not matrix[0]:
            return 0
            
        row, column = len(matrix), len(matrix[0])
        i, j = row - 1, 0
        count = 0
        while i >= 0 and j < column:
            if matrix[i][j] == target:
                count += 1
                i -= 1
                j += 1
            elif matrix[i][j] < target:
                j += 1
            elif matrix[i][j] > target:
                i -= 1
        return count
```
pass