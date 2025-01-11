
**样例 1:**
```
输入: [2, 2]
输出: 2
```
**样例 2：**
```
输入: [1, 2, 2, 3, 1]
输出: 2
解释: 
输入数组的度是2，1和2都出现了两次。
具有相同度的子串包括：
[1, 2, 2, 3, 1], [1, 2, 2, 3], [2, 2, 3, 1], [1, 2, 2], [2, 2, 3], [2, 2]
其中长度最短为2。所以返回2。
```




```python
def find_shortest_sub_array(self, nums: List[int]) -> int:
	mp = dict()

	for i, num in enumerate(nums):
		if num in mp:
			mp[num][0] += 1
			mp[num][2] = i
		else:
			mp[num] = [1, i, i]
	
	maxNum = minLen = 0
	for count, left, right in mp.values():
		if maxNum < count:
			maxNum = count
			minLen = right - left + 1
		elif maxNum == count:
			if minLen > (span := right - left + 1):
				minLen = span
	
	return minLen
```
pass