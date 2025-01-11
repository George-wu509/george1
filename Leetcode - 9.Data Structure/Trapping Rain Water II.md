
**样例 1:**
```
例如，给定一个 5*4 的矩阵： 
输入:
[[12,13,0,12],[13,4,13,12],[13,8,10,12],[12,13,12,12],[13,13,13,13]]
输出:
14
```
**样例 2:**
```
输入:
[[2,2,2,2],[2,2,3,4],[3,3,3,1],[2,3,4,5]]
输出:
0
```



```python
class Solution:
    def trap_rain_water(self, heights: List[List[int]]) -> int:
        if len(heights) <= 2 or len(heights[0]) <= 2:
            return 0

        m, n = len(heights), len(heights[0])
        visited = [[0 for _ in range(n)] for _ in range(m)]
        pq = []
        for i in range(m):
            for j in range(n):
                if i == 0 or i == m - 1 or j == 0 or j == n - 1:
                    visited[i][j] = 1
                    heapq.heappush(pq, (heights[i][j], i * n + j))
        
        res = 0
        dirs = [-1, 0, 1, 0, -1]
        while pq:
            height, position = heapq.heappop(pq)
            for k in range(4):
                nx, ny = position // n + dirs[k], position % n + dirs[k + 1]
                if nx >= 0 and nx < m and ny >= 0 and ny < n and visited[nx][ny] == 0:
                    if height > heights[nx][ny]:
                        res += height - heights[nx][ny]
                    visited[nx][ny] = 1    
                    heapq.heappush(pq, (max(height, heights[nx][ny]), nx * n + ny))
        return res
```
pass