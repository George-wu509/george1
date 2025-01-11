
例1:
```
输入:
{1,2,3,4#2,1,3#3,1,2#4,1,5#5,4}
[3,4,10,50,50]
1
50
输出:
4
解释:
2------3  5
 \     |  | 
  \    |  |
   \   |  |
    \  |  |
      1 --4
Give a node 1, target is 50

there a hash named values which is [3,4,10,50,50], represent:
Value of node 1 is 3
Value of node 2 is 4
Value of node 3 is 10
Value of node 4 is 50
Value of node 5 is 50

Return node 4
```
例2:

```
输入:
{1,2#2,1}
[0,1]
1
1
输出:
2
```



```python
def searchNode(self, graph, values, node, target):
	if len(graph)==0 or len(values)==0:
		return None
	q=collections.deque()
	Set=set()
	q.append(node)
	Set.add(node)
	while len(q):
		if values[q[0]] == target:#找到结果
			return q[0];
		for neighbor in q[0].neighbors:#遍历每个点的所有边
			#判断此点是否出现过
			if neighbor not in Set:
				q.append(neighbor)
				Set.add(neighbor)
		q.popleft()
	return None
```
pass