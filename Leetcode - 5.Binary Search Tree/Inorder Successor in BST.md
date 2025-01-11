
**样例 1:**
```
输入: {1,#,2}, node with value 1
输出: 2
解释: 
  1
   \
    2
```
**样例 2:**

```
输入: {2,1,3}, node with value 1
输出: 2
解释: 
    2
   / \
  1   3
```


```python
def inorderSuccessor(self, root, p):
	if root == None:
		return None
	if root.val <= p.val:
		return self.inorderSuccessor(root.right, p)
	
	left = self.inorderSuccessor(root.left, p)
	if left != None:
		return left
	else:
		return root
```
pass