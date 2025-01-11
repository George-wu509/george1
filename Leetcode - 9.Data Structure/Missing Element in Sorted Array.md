
样例 1：
```
输入：
nums = [1,3,5,7,9]
k = 1
输出：
2
解释：
第一个缺失的数字为 2
```
样例 2：
```
输入：
nums = [1,3,5,7,9]
k = 4
输出：
8
解释：
缺失的数字有 [2,4,6,8,...]，第 4 个缺失的数字为 8
```
样例 3：
```
输入：
nums = [2,3,4,5,7]
k = 4
输出：
10
解释：
缺失的数字有 [6,8,9,10,...]，第 4 个缺失的数字为 10
```



```python
def missing_element(self, nums: List[int], k: int) -> int:
	def missing(i: int):
		return nums[i] - nums[0] - i

	n = len(nums)
	if k > missing(n - 1):
		return nums[n - 1] + k - missing(n - 1)
	left, right = 0, n - 1
	while left < right:
		mid = (left + right) // 2
		if missing(mid) >= k:
			right = mid
		else:
			left = mid + 1
	return nums[left - 1] + k - missing(left - 1)
```
pass