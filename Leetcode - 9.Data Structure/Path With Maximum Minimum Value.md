Lintcode


样例

![图片](https://media.jiuzhang.com/media/markdown/images/11/19/fd09b8a4-0a5f-11ea-8a23-0242ac110002.jpg)
```python
"""
示例1:
输入: [[5,4,5],[1,2,6],[7,4,6]]
输出: 4
解释: 
黄色高亮的路径是最大分数
```

![图片](https://media.jiuzhang.com/media/markdown/images/11/19/2aa2c08a-0a60-11ea-8a23-0242ac110002.jpg)

```python
"""
示例2:
输入: [[2,2,1,2,2,2],[1,2,2,2,1,2]]
输出: 2
```

![图片](https://media.jiuzhang.com/media/markdown/images/11/19/3c9095b0-0a60-11ea-8a23-0242ac110002.jpg)

```python
"""
示例3:
输入: [[3,4,6,3,4],[0,2,1,1,7],[8,8,3,2,7],[3,2,4,9,8],[4,1,2,0,0],[4,6,5,4,3]]
输出: 3
```




```python
def maximum_minimum_path(self, a: List[List[int]]) -> int:
	m, n = len(a), len(a[0])
	self.p = [0] * (m * n)
	flatten = []
	for i in range(m):
		for j in range(n):
			k = i * n + j
			self.p[k] = k
			flatten.append([i, j, a[i][j]])
	flatten.sort(key=lambda x: -x[2])
	ans = float('inf')
	visited = [[False for _ in range(n)] for _ in range(m)]
	directions = [1, 0, -1, 0, 1]
	for i in range(len(flatten)):
		arr = flatten[i]
		y, x, v = arr[0], arr[1], arr[2]
		visited[y][x] = True
		ans = min(v, ans)
		for j in range(4):
			for k in range(4):
				ny, nx = y + directions[k], x + directions[k + 1]
				if 0 <= ny < m and 0 <= nx < n and visited[ny][nx]:
					self.p[self.find(y * n + x)] = self.find(ny * n + nx)
		if self.find(0) == self.find(m * n - 1):
			break
	return ans

def find(self, x: int) -> int:
	if self.p[x] != x:
		self.p[x] = self.find(self.p[x])
	return self.p[x]
```
pass