
**样例1：**  
`输入：数组 ：[1,2,7,8,5] 查询 ：[(1,2),(0,4),(2,4)]。输出：[2,1,5]`  
**样例2：**  
`输入：数组 ：[4,5,7,1] 查询 ：[(1,2),(1,3)]。输出：[5,1]`


```python
class SegmentNode:
    def __init__(self, start, end, min_num = sys.maxsize):
        self.start = start
        self.end = end
        self.min_num = min_num
        self.left = None
        self.right = None
    
class Solution:
    """
    @param A: An integer array
    @param queries: An query list
    @return: The result list
    """
    def build(self, A, start, end):
        if start > end:
            return None
        
        if start == end:
            return SegmentNode(start, end, A[start])
            
        root = SegmentNode(start, end, A[start])
        root.left = self.build(A, start, (start + end) // 2)
        root.right = self.build(A, (start + end) // 2 + 1, end)
        
        root.min_num = min(root.left.min_num, root.right.min_num)
        
        return root
    
    def query(self, root, start, end):
        if root is None:
            return
        
        if root.start > end or root.end < start:
            return sys.maxsize
        
        if root.start >= start and root.end <= end:
            return root.min_num
        
        return min(self.query(root.left, start, end), self.query(root.right, start, end))
    
    
    def interval_min_number(self, a, queries):
        # write your code here
        root = self.build(a, 0, len(a) - 1)
        
        result = []
        
        for q in queries:
            result.append(self.query(root, q.start, q.end))
            
        return result
```
pass