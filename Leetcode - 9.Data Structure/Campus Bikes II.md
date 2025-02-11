Lintcode


样例 1：
```python
"""
输入：
workers = [[0,0],[2,1]]
bikes = [[1,2],[3,3]]
输出：
6
解释：
工人 0 分配到自行车 0，工人 1 分配到自行车 1 。
工人 0 与自行车的曼哈顿距离为：|1 - 0| + |2 - 0| = 3
工人 1 与自行车的曼哈顿距离为：|3 - 2| + |3 - 1| = 3
此方案的曼哈顿距离总和最小，返回 6
```
![图片](https://media-test.jiuzhang.com/media/markdown/images/4/7/fa49c968-78ca-11ea-b383-0242ac1e0004.jpg)

样例 2：
```
输入：
workers = [[0,0],[1,1],[2,0]]
bikes = [[1,0],[2,2],[2,1]]
输出：
4
解释：
工人 0 首先分配到自行车 0 ，工人 1 分配到自行车 1（或自行车 2），工人 2 将分配到自行车 2（或自行车 1）
工人 0 与自行车的曼哈顿距离为：|1 - 0| + |0 - 0| = 1
工人 1 与自行车 1 的曼哈顿距离为：|2 - 1| + |2 - 1| = 2
工人 2 与自行车 2 的曼哈顿距离为：|2 - 2| + |1 - 0| = 1
此方案的曼哈顿距离总和最小，返回 4
```

![图片](https://media-test.jiuzhang.com/media/markdown/images/4/7/0391be68-78cb-11ea-b19f-0242ac1e0004.jpg)



```python
def assign_bikes_i_i(self, workers: List[List[int]], bikes: List[List[int]]) -> int:
	n, m = len(workers), len(bikes)
	dp = [[float("inf")] * (1 << m) for _ in range(n + 1)]
	dp[0][0] = 0
	for i, (x1, y1) in enumerate(workers, 1):
		for j in range(1 << m):
			for k, (x2, y2) in enumerate(bikes):
				if j >> k & 1:
					dp[i][j] = min(dp[i][j], dp[i - 1][j ^ (1 << k)] + abs(x1 - x2) + abs(y1 - y2))
	return min(dp[n])
```
pass