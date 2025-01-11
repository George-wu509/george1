
**样例 1:**
```
输入:
  NumMatrix(
    [[3,0,1,4,2],
     [5,6,3,2,1],
     [1,2,0,1,5],
     [4,1,0,1,7],
     [1,0,3,0,5]]
  )
  sumRegion(2,1,4,3)
  update(3,2,2)
  sumRegion(2,1,4,3)
输出: 
  8
  10
```
**样例 2:**
```
输入: 
  NumMatrix([[1]])
  sumRegion(0, 0, 0, 0)
  update(0, 0, -1)
  sumRegion(0, 0, 0, 0)
输出: 
  1
  -1
```



```python
class NumMatrix(object):

    def __init__(self, matrix):
        """
        :type matrix: List[List[int]]
        """
        self.matrix = matrix
        self.prefix_sum = [[0 for i in range(len(matrix[0]))] for j in range(len(matrix))]
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if i == 0:
                    self.prefix_sum[i][j] = matrix[i][j]
                else:
                    self.prefix_sum[i][j] = matrix[i][j] + self.prefix_sum[i-1][j]
            
    def update(self, row, col, val):
        """
        :type row: int
        :type col: int
        :type val: int
        :rtype: void
        """
        diff = val - self.matrix[row][col]
        self.matrix[row][col] = val 
        for i in range(row, len(self.matrix)):
            self.prefix_sum[i][col] += diff

    def sumRegion(self, row1, col1, row2, col2):
        """
        :type row1: int
        :type col1: int
        :type row2: int
        :type col2: int
        :rtype: int
        """
        ans = 0
        for j in range(col1, col2+1):
            if row1 > 0:
                ans += (self.prefix_sum[row2][j] - self.prefix_sum[row1-1][j])
            else:
                ans += self.prefix_sum[row2][j]

        return ans
```
pass