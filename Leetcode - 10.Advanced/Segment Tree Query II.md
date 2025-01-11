

**样例 1:**
```
输入："[0,3,count=3][0,1,count=1][2,3,count=2][0,0,count=1][1,1,count=0][2,2,count=1][3,3,count=1]",[[1, 1], [1, 2], [2, 3], [0, 2]]
输出：[0,1,2,2]
解释：
对应的线段树为：

	                     [0, 3, count=3]
	                     /             \
	          [0,1,count=1]             [2,3,count=2]
	          /         \               /            \
	   [0,0,count=1] [1,1,count=0] [2,2,count=1], [3,3,count=1]

Input : query(1,1), Output: 0

Input : query(1,2), Output: 1

Input : query(2,3), Output: 2

Input : query(0,2), Output: 2
```
**样例 2:**
```
输入："[0,3,count=3][0,1,count=1][2,3,count=2][0,0,count=1][1,1,count=0][2,2,count=0][3,3,count=1]",[[1, 1], [1, 2], [2, 3], [0, 2]]
输出：[0,0,1,1]
解释：
对应的线段树为：

	                     [0, 3, count=2]
	                     /             \
	          [0,1,count=1]             [2,3,count=1]
	          /         \               /            \
	   [0,0,count=1] [1,1,count=0] [2,2,count=0], [3,3,count=1]

Input : query(1,1), Output: 0

Input : query(1,2), Output: 0

Input : query(2,3), Output: 1

Input : query(0,2), Output: 1
```



```python
class Solution:	
    def query(self, root, start, end):
        if root is None:
            return 0

        if root.start > end or root.end < start:
            return 0
    
        if start <= root.start and root.end <= end:
            return root.count
        
        return self.query(root.left, start, end) + \
                   self.query(root.right, start, end)
```
pass