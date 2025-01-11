
**样例 1:**
```
输入: 5
输出:
[
     [1],
    [1,1],
   [1,2,1],
  [1,3,3,1],
 [1,4,6,4,1]
]
```
**样例 2:**
```
输入: 3
输出:
[
     [1],
    [1,1],
   [1,2,1]
]
```


```python
def generate(self, num_rows: int) -> List[List[int]]:
	ret = list()
	for i in range(num_rows):
		row = list()
		for j in range(0, i + 1):
			if j == 0 or j == i:
				row.append(1)
			else:
				row.append(ret[i - 1][j] + ret[i - 1][j - 1])
		ret.append(row)
	return ret
```
pass