
**样例 1:**
```
输入："[0,3,max=4][0,1,max=4][2,3,max=3][0,0,max=1][1,1,max=4][2,2,max=2][3,3,max=3]",1,2
输出：4
解释：
对于数组 [1, 4, 2, 3], 对应的线段树为 :

	                  [0, 3, max=4]
	                 /             \
	          [0,1,max=4]        [2,3,max=3]
	          /         \        /         \
	   [0,0,max=1] [1,1,max=4] [2,2,max=2], [3,3,max=3]
[1,2]区间最大值为4
```
**样例 2:**

```
输入："[0,3,max=4][0,1,max=4][2,3,max=3][0,0,max=1][1,1,max=4][2,2,max=2][3,3,max=3]",2,3
输出：3
解释：
对于数组 [1, 4, 2, 3], 对应的线段树为 :

	                  [0, 3, max=4]
	                 /             \
	          [0,1,max=4]        [2,3,max=3]
	          /         \        /         \
	   [0,0,max=1] [1,1,max=4] [2,2,max=2], [3,3,max=3]
[2,3]区间最大值为3
```



```python
class Solution:	
    # @param root, start, end: The root of segment tree and 
    #                          an segment / interval
    # @return: The maximum number in the interval [start, end]
    def query(self, root, start, end):
        # write your code here
        if root.start > end or root.end < start:
            return -0x7fffff
    
        if start <= root.start and root.end <= end:
            return root.max
        
        return max(self.query(root.left, start, end), \
                   self.query(root.right, start, end))
```
pass