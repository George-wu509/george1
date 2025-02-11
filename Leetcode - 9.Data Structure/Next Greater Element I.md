Lintcode 1206
你有两个数组 `nums1`和`nums2`**（互不重复）**，其中`nums1`是`nums2`的子集。 在`nums2`的相应位置找到`nums1`所有元素的下一个更大数字。

`nums1`中的数字x的下一个更大数字是`nums2`中x右边第一个更大的数字。 如果它不存在，则为此数字输出-1。


**例子 1:**
```python
"""
输入: nums1 = [4,1,2], nums2 = [1,3,4,2].
输出: [-1,3,-1]
解释:
     对于第一个数组中的数字4，在第二个数组中找不到下一个更大的数字，因此输出-1。
     对于第一个数组中的数字1，第二个数组中的下一个更大数字是3。
     对于第一个数组中的数字2，第二个数组中没有下一个更大的数字，因此输出-1。
```
**例子 2:**
```python
"""
输入: nums1 = [2,4], nums2 = [1,2,3,4].
输出: [3,-1]
解释:
     对于第一个数组中的数字2，第二个数组中的下一个更大数字是3。
     对于第一个数组中的数字4，第二个数组中没有下一个更大的数字，因此输出-1。
```



```python
    def next_greater_element(self, nums1: List[int], nums2: List[int]) -> List[int]:
        res = {}
        stack = []
        for num in reversed(nums2):
            while stack and num >= stack[-1]:
                stack.pop()
            res[num] = stack[-1] if stack else -1
            stack.append(num)
        return [res[num] for num in nums1]
```
pass