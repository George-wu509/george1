
**样例 1：**
输入：
```
A = [1, 2, 1, 3, 4, 5, 7, 6]
```
输出：
```
1
```
解释：
返回任意一个峰顶元素的下标，6也同样正确。  

**样例 2：**
输入：
```
A = [1,2,3,4,1]
```
输出：
```
3
```
解释：
返回峰顶元素的下标。


```python
def find_peak(self, a: List[int]) -> int:
	start, end = 1, len(a) - 2
	while start + 1 <  end:
		mid = (start + end) // 2
		if a[mid] < a[mid - 1]:
			end = mid
		elif a[mid] < a[mid + 1]:
			start = mid
		else:
			return mid

	if a[start] < a[end]:
		return end
	else:
		return start
```
pass