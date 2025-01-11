
**例子 1:**
```
输入: nums1 = [4,1,2], nums2 = [1,3,4,2].
输出: [-1,3,-1]
解释:
     对于第一个数组中的数字4，在第二个数组中找不到下一个更大的数字，因此输出-1。
     对于第一个数组中的数字1，第二个数组中的下一个更大数字是3。
     对于第一个数组中的数字2，第二个数组中没有下一个更大的数字，因此输出-1。
```
**例子 2:**
```
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