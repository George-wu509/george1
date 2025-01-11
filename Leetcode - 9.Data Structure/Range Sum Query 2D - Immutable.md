
**样例1**
```
输入：
[[3,0,1,4,2],[5,6,3,2,1],[1,2,0,1,5],[4,1,0,1,7],[1,0,3,0,5]]
sumRegion(2, 1, 4, 3)
sumRegion(1, 1, 2, 2)
sumRegion(1, 2, 2, 4)
输出：
8
11
12
解释：
给出矩阵
[
  [3, 0, 1, 4, 2],
  [5, 6, 3, 2, 1],
  [1, 2, 0, 1, 5],
  [4, 1, 0, 1, 7],
  [1, 0, 3, 0, 5]
]
sumRegion(2, 1, 4, 3) = 2 + 0 + 1 + 1 + 0 + 1 + 0 + 3 + 0 = 8
sumRegion(1, 1, 2, 2) = 6 + 3 + 2 + 0 = 11
sumRegion(1, 2, 2, 4) = 3 + 2 + 1 + 0 + 1 + 5 = 12
```
**样例2**
```
输入：
[[3,0],[5,6]]
sumRegion(0, 0, 0, 1)
sumRegion(0, 0, 1, 1)
输出：
3
14
解释：
给出矩阵
[
  [3, 0],
  [5, 6]
]
sumRegion(0, 0, 0, 1) = 3 + 0 = 3
sumRegion(0, 0, 1, 1) = 3 + 0 + 5 + 6 = 14
```


```python
class NumMatrix:
    """
    @param: matrix: a 2D matrix
    """
    def __init__(self, matrix):
        self.prefix = [[0] * (len(matrix[0]) + 1) for _ in range(len(matrix) + 1)]
        for i in range(1, len(matrix) + 1):
            for j in range(1, len(matrix[0]) + 1):
                self.prefix[i][j] = self.prefix[i - 1][j] + self.prefix[i][j - 1] - self.prefix[i - 1][j - 1] + matrix[i - 1][j - 1]

    """
    @param: row1: An integer
    @param: col1: An integer
    @param: row2: An integer
    @param: col2: An integer
    @return: An integer
    """
    def sumRegion(self, row1, col1, row2, col2):
        return self.prefix[row2 + 1][col2 + 1] - self.prefix[row1][col2 + 1] - self.prefix[row2 + 1][col1] + self.prefix[row1][col1]
```
pass