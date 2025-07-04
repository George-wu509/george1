### **LintCode 480：Binary Tree Paths**

给一棵二叉树，找出从根节点到叶子节点的所有路径。

Example:
**样例 1:**
```python
输入：{1,2,3,#,5}
输出：["1->2->5","1->3"]
解释：
   1
 /   \
2     3
 \
  5
```
**样例 2:**
```python
输入：{1,2}
输出：["1->2"]
解释：
   1
 /   
2     
```



```python
class Solution2: # passed in leetcode
    def binary_tree_paths(self, root: TreeNode) -> List[str]:
        if not root:
            return []
        
        paths = []
        self.dfs(root, [root], paths)
        return paths

    def dfs(self, node, path, paths):
        if not node:
            return

        if not node.left and not node.right:
            paths.append('->'.join([str(n.val) for n in path]))
            return

        path.append(node.left)
        self.dfs(node.left, path, paths)
        path.pop()

        path.append(node.right)
        self.dfs(node.right, path, paths)
        path.pop()

```
pass


比較dfs的code 
LintCode 15 全排列（Permutations） [[Permutations]]
```python
class Solution:
    def permute(self, nums):
        results = []
        visited = [False]*len(nums)
        self.dfs(nums, visited, [], results)
        return results
        
    def dfs(self, nums, visited, num, results):
        if len(num) == len(nums):
            results.append(list(num))
            return
        
        for i in range(len(nums)):
            if not visited[i]:
                visited[i] = True
                num.append(nums[i])
                self.dfs(nums, visited, num, results)
                num.pop()
                visited[i] = False
                
sol = Solution()
nums = [1,2,3]
result = sol.permute(nums)
print(result)
```