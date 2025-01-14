
**样例 1：**
输入：
```
数组 = [1,4,4,5,7,7,8,9,9,10]
target = 1
```
输出：
```
0
```
解释：
第一次出现在第0个位置。

**样例 2：**
输入：
```
数组 = [1, 2, 3, 3, 4, 5, 10]
target = 3
```
输出：
```
2
```
解释：
第一次出现在第2个位置

**样例 3：**
输入：
```
数组 = [1, 2, 3, 3, 4, 5, 10]
target = 6
```
输出：
```
-1
```
解释：
没有出现过6， 返回-1


```python
    def binary_search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums)-1
        while left + 1 < right :
            mid = (left + right)//2
            if nums[mid] < target :
                left = mid
            else :
                right = mid
        if nums[left] == target :
            return left
        elif nums[right] == target :
            return right
        return -1;
```
pass