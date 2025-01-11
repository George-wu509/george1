
**样例 1:**

```
输入:[[1,2],[0,3]]
输出:[[0,2],[0,0]]
```

**样例 2:**

```
输入:[[1,2,3],[4,0,6],[7,8,9]]
输出:[[1,0,3],[0,0,0],[7,0,9]]
```


```python
    def set_zeroes(self, matrix):
        # write your code here
        if len(matrix)==0:
            return
        rownum = len(matrix)
        colnum = len(matrix[0])
        row = [False for i in range(rownum)]
        col = [False for i in range(colnum)]
        for i in range(rownum):
            for j in range(colnum):
                if matrix[i][j] == 0:
                    row[i] = True
                    col[j] = True
        for i in range(rownum):
            for j in range(colnum):
                if row[i] or col[j]:
                    matrix[i][j] = 0
```
pass