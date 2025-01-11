
```
输入: [3,2,1,4]
解释: 
这颗线段树将会是
          [0,3](max=4)
          /          \
       [0,1]         [2,3]    
      (max=3)       (max=4)
      /   \          /    \    
   [0,0]  [1,1]    [2,2]  [3,3]
  (max=3)(max=2)  (max=1)(max=4)
```




```python
class Solution:	
    # @oaram a: a list of integer
    # @return: The root of Segment Tree
    def build(self, a):
        return self.buildTree(0, len(a)-1, a)

    def buildTree(self, start, end, a):
        if start > end:
            return None

        node = SegmentTreeNode(start, end, a[start])
        if start == end:
            return node

        mid = (start + end) // 2
        node.left = self.buildTree(start, mid, a)
        node.right = self.buildTree(mid + 1, end, a)
        if node.left is not None and node.left.max > node.max:
            node.max = node.left.max
        if node.right is not None and node.right.max > node.max:
            node.max = node.right.max
        return node
```
pass