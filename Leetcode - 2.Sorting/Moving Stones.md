
**样例1**
```
输入: arr = [5,4,1]
输出: 1
解释: 
你只需要把位置4的石子往左移动到3，
[1,3,5],符合要求。
```
**样例2**
```
输入:arr = [1,6,7,8,9]
输出: 5
解释: 
最优的移动方案为把1移动到2，把6移动到4，把7移动到6，把9移动到10。
花费为1+2+1+1=5。
```


```python
def moving_stones(self, arr):
	ans1 = 0
	ans2 = 0
	n = len(arr);
	arr.sort()
	for i in range(n):
		ans1 += abs(arr[i] - i * 2 - 1)
		ans2 += abs(arr[i] - i * 2 - 2)
	return min(ans1, ans2)
```
pass