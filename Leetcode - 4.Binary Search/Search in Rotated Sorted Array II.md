Lintcode 63
跟进“搜索旋转排序数组”，假如有重复元素又将如何？  
是否会影响运行时间复杂度？  
如何影响？  
为何会影响？  
写出一个函数判断给定的目标值是否出现在数组中。

**样例 1：**
输入：
```python
A = []
target = 1
```
输出：
```python
false
```
解释：
数组为空，1不在数组中。  

**样例 2：**
输入：
```python
A = [3,4,4,5,7,0,1,2]
target = 4
```
输出：
```python
true
```
解释：
4在数组中。


```python
def search(self, a, target):
	if not len(a):
		return False
	# step1: find pivot
	left, right = 0, len(a)-1
	while(left < right):
		mid = left + (right-left)//2
		if a[mid] > a[right]:
			left = mid + 1
		elif a[mid] < a[right]:
			right = mid
		else:
			if (right > 0 and a[right - 1] > a[right]):
				left = right
			else:
				right -= 1
	pivot = left
	# step2: split
	if pivot == 0:
		left, right = 0, len(a)-1
	elif target >= a[0]:
		left, right = 0, pivot - 1
	else:
		left, right = pivot, len(a) - 1
	# step3: find target
	while left + 1 < right:
		mid = left + (right - left) // 2
		if a[mid] < target:
			left = mid
		else:
			right = mid
	if a[left] == target:
		return True
	if a[right] == target:
		return True
	return False
```
pass