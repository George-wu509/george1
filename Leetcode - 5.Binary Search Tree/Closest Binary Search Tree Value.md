
**样例1**
```
输入: root = {5,4,9,2,#,8,10} and target = 6.124780
输出: 5
解释：
二叉树 {5,4,9,2,#,8,10}，表示如下的树结构：
        5
       / \
     4    9
    /    / \
   2    8  10
```
**样例2**
```
输入: root = {3,2,4,1} and target = 4.142857
输出: 4
解释：
二叉树 {3,2,4,1}，表示如下的树结构：
     3
    / \
  2    4
 /
1
```



```python
```def closest_value(self, root: TreeNode, target: float) -> int:
        upper = root
17        lower = root
18        while root:
19            if target > root.val:
20                lower = root
21                root = root.right
22            elif target < root.val:
23                upper = root
24                root = root.left
25            else:
26                return root.val
27        if abs(upper.val - target) <= abs(lower.val - target):
28            return upper.val
29        return lower.val
```
```
pass