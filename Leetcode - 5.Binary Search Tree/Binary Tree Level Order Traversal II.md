lintcode 70
给出一棵二叉树，返回其节点值从底向上的层次序遍历（按从叶节点所在层到根节点所在的层遍历，然后逐层从左往右遍历）


**样例 1：**
输入：
```
tree = {1,2,3}
```
输出：
```
[[2,3],[1]]
```
解释：
```
    1
   / \
  2   3
```
它将被序列化为 {1,2,3}  

**样例 2：**
输入：
```
tree = {3,9,20,#,#,15,7}
```
输出：
```
[[15,7],[9,20],[3]]
```
解释：
```
    3
   / \
  9  20
    /  \
   15   7
```
它将被序列化为 {3,9,20,#,#,15,7}



```python
def level_order_bottom(self, root):
	self.results = []
	if not root:
		return self.results
	q = [root]
	while q:
		new_q = []
		self.results.append([n.val for n in q])
		for node in q:
			if node.left:
				new_q.append(node.left)
			if node.right:
				new_q.append(node.right)
		q = new_q
	return list(reversed(self.results))
```
pass