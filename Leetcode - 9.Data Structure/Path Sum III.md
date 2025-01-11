
**样例 1:**
```
输入：root = [10,5,-3,3,2,null,11,3,-2,null,1], sum = 8
输出：3
解释：
      10
     /  \
    5   -3
   / \    \
  3   2   11
 / \   \
3  -2   1

返回 3。 和为8的路径为:

1.  5 -> 3
2.  5 -> 2 -> 1
3. -3 -> 11
```
**样例 2:**
```
输入：root = [11,6,-3], sum = 17
输出：1
解释：
      11
     /  \
    6   -3
返回 1。 和为17的路径为:

1.  11 -> 6
```


```python
def path_sum(self, root: TreeNode, sum: int) -> int:
	prefix = collections.defaultdict(int)
	prefix[0] = 1

	def dfs(root, curr):
		if not root:
			return 0
		
		ret = 0
		curr += root.val
		ret += prefix[curr - sum]
		prefix[curr] += 1
		ret += dfs(root.left, curr)
		ret += dfs(root.right, curr)
		prefix[curr] -= 1

		return ret

	return dfs(root, 0)
```
pass