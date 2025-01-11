
如下图:

```
	A----->B----->C
	 \     |
	  \    |
	   \   |
	    \  v
	     ->D----->E
			 
样例 1:
输入:s = B and t = E,
输出:true

样例 2:
输入:s = D and t = C,
输出:false
```



```python
def hasRoute(self, graph, s, t):
	queue = collections.deque([s])
	visited = set()
	visited.add(s)
	
	while queue:
		node = queue.popleft()
		if node == t:
			return True
		for neighbor in node.neighbors:
			if neighbor in visited:
				continue
			queue.append(neighbor)
			visited.add(neighbor)
	return False
```
pass