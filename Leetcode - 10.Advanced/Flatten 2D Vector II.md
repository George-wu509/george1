
**样例1**
```
输入： {3,9,20,#,#,15,7}
输出： [[9],[3,15],[20],[7]]
解释：
   3
  /\
 /  \
 9  20
    /\
   /  \
  15   7
```
**样例2**
```
输入： {3,9,8,4,0,1,7}
输出：[[4],[9],[3,0,1],[8],[7]]
解释：
     3
    /\
   /  \
   9   8
  /\  /\
 /  \/  \
 4  01   7
```



```python
import collections
class Solution:
    """
    @param root: the root of tree
    @return: the vertical order traversal
    """
    def verticalOrder(self, root):
        results = collections.defaultdict(list)
        queue = collections.deque()
        
        queue.append((root, 0))
        # 宽度优先遍历，同时记录列编号
        while queue:
            node, col_idx = queue.popleft()
            if node:
                results[col_idx].append(node.val)
                queue.append((node.left, col_idx - 1))
                queue.append((node.right, col_idx + 1))
        
        return [results[i] for i in sorted(results)]
```
pass