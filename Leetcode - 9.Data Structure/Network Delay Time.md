Lintcode


```python
"""
样例 1:
	输入:  times = [[2,1,1],[2,3,1],[3,4,1]], N = 4, K = 2
	输出:  2
	
样例 2:
	输入: times = [[1,2,1]], N = 2, K = 1
	输出:  1
	
	解释:
	两条路选择最短的。
```



```python
def network_delay_time(self, times, n, k):      
	INF = 0x3f3f3f3f
	G = [[INF for i in range(n+1)] for j in range(n+1)]
	for i in range(1, n+1):
		G[i][i] = 0
	for i in range(0, len(times)): 
		G[times[i][0]][times[i][1]] = times[i][2]
	
	dis = G[k][:]
	vis = [0]*(n + 1)
	
	for i in range(0,  n-1):
		Mini = INF
		p = k
		for j in range(1, n+1):
			if vis[j] == 0 and dis[j] < Mini:
				Mini = dis[j]
				p = j
		vis[p] = 1
		for j in range(1, n+1):
			if vis[j] == 0 and dis[j] > dis[p] + G[p][j]:
				dis[j] = dis[p] + G[p][j]
	ans = 0
	for i in range(1, n+1):
		ans = max(ans, dis[i])
	if ans == INF:
		return -1
	
	return ans
```
pass