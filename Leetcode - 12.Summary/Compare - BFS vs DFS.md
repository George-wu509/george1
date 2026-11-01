
|                                 |     |
| ------------------------------- | --- |
| [[#### BFS vs DFS pseudo code]] |     |
| [[#### BFS vs DFS何時使用]]         |     |
|                                 |     |
#### BFS vs DFS pseudo code
```
BFS vs DFS pseudo code
```

BFS 廣度優先搜索(Breadth-First Search)
```python
visited = set()
for i
for j
   if grid[i][j] not in visited:
      BFS(grid, i, j, visited)

BFS(grid, x, y, visited):
   qu = deque([x,y])
   visited.add((x,y))
   while qu:
      x,y = queue.popleft()
      for delta_x, delta_y in DIRECTIONS:
         next_x = x + delta_x
         next_y = y + delta_y
            
         queue.append((next_x, next_y))
         visited.add((next_x, next_y))
```
ref: [[Number of Islands]]
explain:
step1: Create visited=set(), DIRECTIONS
step2: 先遍歷Array(2D array)在每個沒有visited點執行BFS()
step3: 在BFS裡面(沒有遞迴), 先create deque([x,y]), 並標示visited
step4: 接著 while queue, 並把這個點(x,y)從queue左邊移除
step5: 用for迴圈, deque加入這個點周圍(DIRECTIONS)(x+-dx,y+-y)
step6: 在for迴圈裏面也把這些點(x+-dx,y+-y)標示成visited.
step7: 接著 while queue=(x+-dx,y+-y), 也依序移除這些點
step8. 並在每個這些點都用for迴圈加入周圍的點到deque以及visited

![[Pasted image 20261101105315.png]]
![[Pasted image 20261101105502.png|401]]

DFS 深度優先搜索(Depth-First Search)
```python
results = []
visited = [False]*len(nums)

DFS(nums, visited, num, results):
   到tree底則append加入results
   
   for i in nums全部
      visited[i] = True
      nums.append(num[i])
      DFS()
      nums.pop()
```
ref: [[Permutations]]
Explain:
step1: Create visited=[False]* len(nums)
step2: 在DFS(nums, visited, num, results)裡面(有遞迴)
step3: DFS()開頭如果走到底則num=[1,2,3]加入results成= [ [1,2,3], ]
step4: 在DFS()之後用for迴圈i in nums, 如果nums[i]沒有在visited, 則加
      入到visited裡面
step5: 則加入到visited裡面 -> num用append加入nums[i] -> 遞迴DFS           -> num用pop()移除 -> 從visited移除nums[i] 
```python
    visited[i] = True         #i加入到visited
    num.append(nums[i])       #數值 append加入到num
    self.dfs(nums, visited, num, results)
    num.pop()                 #數值 移除 pop from num
    visited[i] = False        #從visited移除i
```


![[Pasted image 20261101110143.png|397]]

![[Pasted image 20261101111953.png|382]]




#### BFS vs DFS何時使用
```
BFS vs DFS何時使用
```

能用BFS (沒有遞迴)的一定不要用DFS(有遞迴)（除非面试官特别要求）

![[Pasted image 20261101115147.png]]

![[Pasted image 20261101115211.png]]

