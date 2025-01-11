
**样例 1：**
输入：
```
tree = {}
node= 1
```
输出：
```
{1}
```
解释：
在空树中插入一个点，应该插入为根节点。

**样例 2：**
输入：
```
tree = {2,1,4,#,#,3}
node = 6
```
输出：
```
{2,1,4,#,#,3,6}
```
解释：

![85_1.png](https://media-cn.lintcode.com/new_storage_v2/public/202404/69f3e93219f74784aeddbefdf86e8b4a/85_1.png)


```python
class Solution:

    def insertNode(self, root, node):
        if root is None:
            return node
            
        curt = root
        while curt != node:
            if node.val < curt.val:
                if curt.left is None:
                    curt.left = node
                curt = curt.left
            else:
                if curt.right is None:
                    curt.right = node
                curt = curt.right
        return root
```
pass