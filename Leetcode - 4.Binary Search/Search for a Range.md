
**样例 1：**
输入：
```
数组 = []
target = 9
```
输出：
```
[-1,-1]
```
解释：
9不在数组中。

**样例 2：**
输入：
```
数组 = [5, 7, 7, 8, 8, 10]
target = 8
```
输出：
```
[3,4]
```
解释：
数组的[3,4]子区间值为8。


```python
class Solution:
    """
    @param a: an integer sorted array
    @param target: an integer to be inserted
    @return: a list of length 2, [index1, index2]
    """
    def search_range(self, a: List[int], target: int) -> List[int]:
        if len(a) == 0:
            return [-1, -1]
        
        start, end = 0, len(a) - 1
        while start + 1 < end:
            mid = (start + end) // 2
            if a[mid] < target:
                start = mid
            else:
                end = mid
        
        if a[start] == target:
            leftBound = start
        elif a[end] == target:
            leftBound = end
        else:
            return [-1, -1]
        
        start, end = leftBound, len(a) - 1
        while start + 1 < end:
            mid = (start + end) // 2
            if a[mid] <= target:
                start = mid
            else:
                end = mid
        if a[end] == target:
            rightBound = end
        else:
            rightBound = start
        return [leftBound, rightBound]
```
pass