Lintcode 155
给定一个二叉树，找出其最小深度。

二叉树的最小深度为根节点到最近叶子节点的最短路径上的节点数量。

**样例 1:**
```python
"""
输入: {}
输出: 0
```
**样例 2:**
```python
"""
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
```python
"""
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