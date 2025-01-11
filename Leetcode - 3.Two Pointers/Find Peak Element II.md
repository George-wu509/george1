

**样例 1:**
```
输入: 
    [
      [1, 2, 3, 6,  5],
      [16,41,23,22, 6],
      [15,17,24,21, 7],
      [14,18,19,20,10],
      [13,14,11,10, 9]
    ]
输出: [1,1]
解释: [2,2] 也是可以的. [1,1] 的元素是 41, 大于它四周的每一个元素 (2, 16, 23, 17).
```
**样例 2:**
```
输入: 
    [
      [1, 5, 3],
      [4,10, 9],
      [2, 8, 7]
    ]
输出: [1,1]
解释: 只有这一个峰值
```


```python
class Solution:
   
    def find_peak_i_i(self, A):
        left, right = 0, len(A[0]) - 1
        up, down = 0, len(A) - 1
        
        while left + 1 < right or up + 1 < down:
            if up - down >= right - left:
                row = (up + down) // 2
                max_col = self.findMaxCol(row, left, right, A)
                if A[row][max_col] < A[row - 1][max_col]:
                    down = row - 1
                elif A[row][max_col] < A[row + 1][max_col]: 
                    up = row + 1
                else:
                    return [row, max_col]
            else:
                col = (left + right) // 2
                max_row = self.findMaxRow(up, down, col, A)
                if A[max_row][col] < A[max_row][col + 1]:
                    left = col + 1
                elif A[max_row][col] < A[max_row][col - 1]:
                    right = col - 1
                else:
                    return [max_row, col]
        
        return [up, left]
    
    def findMaxRow(self, up, down, col, A):
        max_val = -sys.maxsize
        row = -1
        for i in range(up, down + 1):
            if A[i][col] > max_val:
                max_val = A[i][col]
                row = i
        return row
    
    def findMaxCol(self, row, left, right, A):
        max_val = -sys.maxsize
        col = -1
        for i in range(left, right + 1):
            if A[row][i] > max_val:
                max_val = A[row][i]
                col = i
        return col
```
pass