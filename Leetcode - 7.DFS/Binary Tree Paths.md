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
        self.find_paths(root, [root], paths)
        return paths

    def find_paths(self, node, path, paths):
        if not node:
            return

        if not node.left and not node.right:
            paths.append('->'.join([str(n.val) for n in path]))
            return

        path.append(node.left)
        self.find_paths(node.left, path, paths)
        path.pop()

        path.append(node.right)
        self.find_paths(node.right, path, paths)
        path.pop()

if __name__ == "__main__":
    sol = Solution2()
    lst = ['1','2','3','4','5']
    tree = create_tree(lst)

    paths = sol.binary_tree_paths(tree)
    print(paths)
```
pass