Lintcode 1354
给定非负索引k，其中k≤33，返回杨辉三角形的第k个索引行。

**样例1**
```python
"""
输入: 3
输出: [1,3,3,1]
```
**样例2**
```python
"""
输入: 4
输出: [1,4,6,4,1]
```

```python

def get_row(self, row_index: int) -> List[int]:
	row = [0 for _ in range(row_index + 1)]
	row[0] = 1
	for i in range(1, row_index + 1):
		row[i] = row[i - 1] * (row_index - i + 1) // i
	return row
```
pass