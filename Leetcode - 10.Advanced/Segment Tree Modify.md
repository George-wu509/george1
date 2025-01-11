

**样例 1:**

```
输入："[1,4,max=3][1,2,max=2][3,4,max=3][1,1,max=2][2,2,max=1][3,3,max=0][4,4,max=3]",2,4
输出："[1,4,max=4][1,2,max=4][3,4,max=3][1,1,max=2][2,2,max=4][3,3,max=0][4,4,max=3]"
解释：
线段树:

	                      [1, 4, max=3]
	                    /                \
	        [1, 2, max=2]                [3, 4, max=3]
	       /              \             /             \
	[1, 1, max=2], [2, 2, max=1], [3, 3, max=0], [4, 4, max=3]

如何调用modify(root, 2, 4), 可以得到:

	                      [1, 4, max=4]
	                    /                \
	        [1, 2, max=4]                [3, 4, max=3]
	       /              \             /             \
	[1, 1, max=2], [2, 2, max=4], [3, 3, max=0], [4, 4, max=3]
```

**样例 2:**

```
输入："[1,4,max=3][1,2,max=2][3,4,max=3][1,1,max=2][2,2,max=1][3,3,max=0][4,4,max=3]",4,0
输出："[1,4,max=4][1,2,max=4][3,4,max=0][1,1,max=2][2,2,max=4][3,3,max=0][4,4,max=0]"
解释：
线段树:

	                      [1, 4, max=3]
	                    /                \
	        [1, 2, max=2]                [3, 4, max=3]
	       /              \             /             \
	[1, 1, max=2], [2, 2, max=1], [3, 3, max=0], [4, 4, max=3]
如果调用modify(root, 4, 0), 可以得到:
	
	                      [1, 4, max=2]
	                    /                \
	        [1, 2, max=2]                [3, 4, max=0]
	       /              \             /             \
	[1, 1, max=2], [2, 2, max=1], [3, 3, max=0], [4, 4, max=0]
```



```python
```python
class Solution:	
10    """
11    @param root, index, value: The root of segment tree and 
12    @ change the node's value with [index, index] to the new given value
13    @return: nothing
14    """
15    def modify(self, root, index, value):
16        # write your code here
17        if root is None:
18            return
19
20        if root.start == root.end:
21            root.max = value
22            return
23    
24        if root.left.end >= index:
25            self.modify(root.left, index, value)
26        else:
27            self.modify(root.right, index, value)
28        
29        root.max = max(root.left.max, root.right.max)
```
```
pass