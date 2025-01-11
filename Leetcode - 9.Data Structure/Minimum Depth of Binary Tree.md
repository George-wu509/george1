
**样例 1:**
```
输入: {}
输出: 0
```
**样例 2:**
```
输入:  {1,#,2,3}
输出: 3	
解释:
	1
	 \ 
	  2
	 /
	3    
它将被序列化为 {1,#,2,3}
```
**样例 3:**
```
输入:  {1,2,3,#,#,4,5}
输出: 2	
解释: 
      1
     / \ 
    2   3
       / \
      4   5  
它将被序列化为 {1,2,3,#,#,4,5}
```


```python
def min_depth(self, root):
	if root is None:
		return 0
	leftDepth = self.min_depth(root.left)
	rightDepth = self.min_depth(root.right)
	# 当左子树或右子树为空时，最小深度取决于另一颗子树
	if leftDepth == 0 or rightDepth == 0:
		return leftDepth + rightDepth + 1
	return min(leftDepth, rightDepth) + 1
```
pass