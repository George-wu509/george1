
**样例 1:**
```
输入:
{1}
0.000000
1
输出:
[1]
解释：
二叉树 {1}，表示如下的树结构：
 1
```
**样例 2:**
```
输入:
{3,1,4,#,2}
0.275000
2
输出:
[1,2]
解释：
二叉树 {3,1,4,#,2}，表示如下的树结构：
  3
 /  \
1    4
 \
  2
```



```python
```python
class Solution:
10    """
11    @param root: the given BST
12    @param target: the given target
13    @param k: the given k
14    @return: k values in the BST that are closest to the target
15    """
16    def closestKValues(self, root, target, k):
17        if root is None or k == 0:
18            return []
19            
20        lower_stack = self.get_stack(root, target)
21        upper_stack = list(lower_stack)
22        if lower_stack[-1].val < target:
23            self.move_upper(upper_stack)
24        else:
25            self.move_lower(lower_stack)
26        
27        result = []
28        for i in range(k):
29            if self.is_lower_closer(lower_stack, upper_stack, target):
30                result.append(lower_stack[-1].val)
31                self.move_lower(lower_stack)
32            else:
33                result.append(upper_stack[-1].val)
34                self.move_upper(upper_stack)
35                
36        return result
37        
38    def get_stack(self, root, target):
39        stack = []
40        while root:
41            stack.append(root)
42            if target < root.val:
43                root = root.left
44            else:
45                root = root.right
46                
47        return stack
48        
49    def move_upper(self, stack):
50        if stack[-1].right:
51            node = stack[-1].right
52            while node:
53                stack.append(node)
54                node = node.left
55        else:
56            node = stack.pop()
57            while stack and stack[-1].right == node:
58                node = stack.pop()
59                
60    def move_lower(self, stack):
61        if stack[-1].left:
62            node = stack[-1].left
63            while node:
64                stack.append(node)
65                node = node.right
66        else:
67            node = stack.pop()
68            while stack and stack[-1].left == node:
69                node = stack.pop()
70                
71    def is_lower_closer(self, lower_stack, upper_stack, target):
72        if not lower_stack:
73            return False
74            
75        if not upper_stack:
76            return True
77            
78        return target - lower_stack[-1].val < upper_stack[-1].val - target
```
```
pass