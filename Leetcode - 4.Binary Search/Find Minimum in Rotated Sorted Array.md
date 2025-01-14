
**样例 1:**
```python
输入：[4, 5, 6, 7, 0, 1, 2]
输出：0
解释：
数组中的最小值为0
```
**样例 2:**
```python
输入：[2,1]
输出：1
解释：
数组中的最小值为1
```


```python
def find_min(self, nums):
	if not nums:
		return -1
		
	start, end = 0, len(nums) - 1
	while start + 1 < end:
		mid = (start + end) // 2
		if nums[mid] > nums[end]:
			start = mid
		else:
			end = mid
			
	return min(nums[start], nums[end])
```
pass