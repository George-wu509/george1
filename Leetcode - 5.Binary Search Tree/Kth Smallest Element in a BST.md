
**样例 1:**
```
输入：{1,#,2},2
输出：2
解释：
	1
	 \
	  2
第二小的元素是2。
```
**样例 2:**
```
输入：{2,1,3},1
输出：1
解释：
  2
 / \
1   3
第一小的元素是1。
```


```python
def kth_smallest(self, root: TreeNode, k: int) -> int:
	stack = []
	while root:
		stack.append(root)
		root = root.left

	for i in range(k-1):
		node = stack.pop()

		if node.right:
			node = node.right
			while node:
				stack.append(node)
				node = node.left
	
	return stack[-1].val
```
pass