
**样例 1：**
```
给定的数独谜题:
[
[0,0,9,7,4,8,0,0,0],
[7,0,0,0,0,0,0,0,0],
[0,2,0,1,0,9,0,0,0],
[0,0,7,0,0,0,2,4,0],
[0,6,4,0,1,0,5,9,0],
[0,9,8,0,0,0,3,0,0],
[0,0,0,8,0,3,0,2,0],
[0,0,0,0,0,0,0,0,6],
[0,0,0,2,7,5,9,0,0]
]
```
![](http://lintcode-media.s3.amazonaws.com/problem/250px-Sudoku-by-L2G-20050714.svg.png)
```
返回结果：
[
[5,1,9,7,4,8,6,3,2],
[7,8,3,6,5,2,4,1,9],
[4,2,6,1,3,9,8,7,5],
[3,5,7,9,8,6,2,4,1],
[2,6,4,3,1,7,5,9,8],
[1,9,8,5,2,4,3,6,7],
[9,7,5,8,6,3,1,2,4],
[8,3,2,4,9,1,7,5,6],
[6,4,1,2,7,5,9,8,3]
]
```


```python
class Solution:

    def solve_sudoku(self, board):
        self.backtrack(board, 0, 0)
        return board
    def backtrack(self, board, i, j):
        m, n = 9, 9
        # 到达第n列，越界，换到下一行第0列重新开始
        if j == n:
            return self.backtrack(board, i + 1, 0)
        # 到达第m行，说明找到可行解，触发 base case
        if i == m:
            return True
        # 如果有预设数字，不用我们穷举
        if board[i][j] != 0:
            return self.backtrack(board, i, j + 1)
        for val in range(1, 10):
            # 如果遇到不合法的数字，就跳过
            if not self.isValid(board, i, j, val):
                continue
            # 添加选择
            board[i][j] = val
            # 如果找到一个可行解，立即结束
            if self.backtrack(board, i, j + 1):
                return True
            # 撤回选择
            board[i][j] = 0
        # 穷举完1~9，依然没有找到可行解，此路不通
        return False
    def isValid(self, board, row, col, val):
        for i in range(9):
            # 判断行是否存在重复
            if board[row][i] == val:
                return False
            # 判断列是否存在重复
            if board[i][col] == val:
                return False
            # 判断 3 x 3 方框是否存在重复
            if board[row // 3 * 3 + i // 3][col // 3 * 3 + i % 3] == val:
                return False
        return True
```
pass