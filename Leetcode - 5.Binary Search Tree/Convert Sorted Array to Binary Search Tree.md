
**样例 1:**
```
输入: [-10,-3,0,5,9],
输出: [0,-3,9,-10,#,5],
解释:
针对该数组的其中一个解为 [0,-3,9,-10,null,5], 其对应的平衡BST树如下:
      0
     / \
   -3   9
   /   /
 -10  5
 对于这棵树，你应该返回值为0的根节点。

针对该数组的另一个解为 [0,-10,5,null,-3,null,9]，其对应的平衡BST树如下：
      0
     / \
   -10  5
     \   \
     -3   9
 对于这棵树，你应该返回值为0的根节点。
```
**样例 2:**
```
输入: [1,3]
输出: [3,1]
解释:
针对该数组的其中一个解为 [3,1], 其对应的平衡BST树如下:
  3
 / 
1   
对于这棵树，你应该返回值为3的根节点。
```



```python
class Solution:

    def convert_sorted_arrayto_binary_search_tree(self, nums: List[int]) -> TreeNode:
        return self.binarySearch(nums, 0, len(nums)-1)

    def binarySearch(self, nums, start, end):
        if start > end:
            return None 
        if start == end:
            return TreeNode(nums[start])
        mid = start + (end-start)//2
        node = TreeNode(nums[mid])
        node.left = self.binarySearch(nums, start, mid-1)
        node.right = self.binarySearch(nums, mid+1, end)

        return node
```
pass