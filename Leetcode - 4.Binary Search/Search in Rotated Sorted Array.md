Lintcode 62
给定一个有序数组，但是数组以某个元素作为支点进行了旋转(比如，`0 1 2 4 5 6 7` 可能成为`4 5 6 7 0 1 2`)。给定一个目标值`target`进行搜索，如果在数组中找到目标值返回数组中的索引位置，否则返回`-1`。你可以假设数组中不存在重复的元素。

**样例 1：**
输入：
```python
数组 = [4, 5, 1, 2, 3]
target = 1
```
输出：
```python
2
```
解释：
1在数组中对应索引位置为2。

**样例 2：**
输入：
```python
数组 = [4, 5, 1, 2, 3]
target = 0
```
输出：
```python
-1
```
解释：
0不在数组中，返回-1。


```python
def search(self, A, target):
	if not A:
		return -1
		
	start, end = 0, len(A) - 1
	while start + 1 < end:
		mid = (start + end) // 2
		if A[mid] >= A[start]:
			if A[start] <= target <= A[mid]:
				end = mid
			else:
				start = mid
		else:
			if A[mid] <= target <= A[end]:
				start = mid
			else:
				end = mid
				
	if A[start] == target:
		return start
	if A[end] == target:
		return end
	return -1
```
pass