LintCode 1080: 最大岛屿面积


**样例 1:**
```python
输入：
[[0,0,1,0,0,0,0,1,0,0,0,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,1,1,0,1,0,0,0,0,0,0,0,0],
 [0,1,0,0,1,1,0,0,1,0,1,0,0],
 [0,1,0,0,1,1,0,0,1,1,1,0,0],
 [0,0,0,0,0,0,0,0,0,0,1,0,0],
 [0,0,0,0,0,0,0,1,1,1,0,0,0],
 [0,0,0,0,0,0,0,1,1,0,0,0,0]]
输出：6。
解释：注意不是11！因为是4方向联通。
```
**样例 2:**
```python
输入：[[0,0,0,0,0,0,0,0]]
输出：0
```


```python
from collections import deque
DIRECTIONS = [(1, 0), (0, -1), (-1, 0), (0, 1)]

class Solution:
    def max_area_of_island(self, grid: List[List[int]]) -> int:
        if not grid or not grid[0]:
            return 0
            
        max_area = 0
        visited = set()
        
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] and (i, j) not in visited:
                    area = self.bfs(grid, i, j, visited)
                    max_area = max(max_area, area)
                    
        return max_area                   

    def bfs(self, grid, x, y, visited):
        queue = deque([(x, y)])
        visited.add((x, y))
        area = 1
        
        while queue:
            x, y = queue.popleft()
            for delta_x, delta_y in DIRECTIONS:
                next_x = x + delta_x
                next_y = y + delta_y
                if not self.is_valid(grid, next_x, next_y, visited):
                    continue
                queue.append((next_x, next_y))
                visited.add((next_x, next_y))
                area += 1
        return area

    def is_valid(self, grid, x, y, visited):
        n, m = len(grid), len(grid[0])
        if not (0 <= x < n and 0 <= y < m):
            return False
        if (x, y) in visited:
            return False
        return grid[x][y]

```
pass

解釋:
step1  遍歷grid, 找到value=1的格子然後用BFS並計算這格BFS的面積.  
step2  ans = max(ans, area)並輸出最大面積