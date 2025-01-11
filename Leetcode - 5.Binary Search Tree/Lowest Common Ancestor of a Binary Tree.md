

**样例 1：**
输入：
```
tree = {1}
A = 1
B = 1
```
输出：
```
1
```
解释：
```
二叉树如下（只有一个节点）:
        1
LCA(1,1) = 1
```

**样例 2：**
输入：
```
tree = {4,3,7,#,#,5,6}
A = 3
B = 5
```
输出：
```
4
```
解释：
```
二叉树如下:

    4
   / \
  3   7
     / \
    5   6
                        
LCA(3, 5) = 4
```


```python
    def lowestCommonAncestor(self, root, A, B):
        if root is None:
            return None
        
        if root is A or root is B:
            return root

        left_result = self.lowestCommonAncestor(root.left, A, B)
        right_result = self.lowestCommonAncestor(root.right, A, B)

        if left_result and right_result:
            return root

        if left_result:
            return left_result

        if right_result:
            return right_result

        return None
```
pass