
**样例 1:**
```
输入:
8 8
输出:
yes

解释:
给出`L=8`，`W=8`，障碍物坐标为`[[1,1],[6,6]]`。返回`yes`。
车的圆心可以从（0,5)到(2,5)到(5,2)到(8,2)，所以返回yes。
```
**样例 2:**
```
输入:
8 6
输出:
no

解释:
给出`L=8`，`W=6`，障碍物坐标为`[[1,1]]`,返回`no`。
不管如何驾驶，车总会与障碍物相切或者相交，这都是不被允许的。
```


```python
OBSTAClE_MIN_DISTANCE = 6
BOUND_MIN_DISTANCE = 5

class Solution:
    def driving_problem(self, l, w, obstacles):
        from collections import deque
        
        # consider the upper & bottom line y=w, y=0, all obstacles
        # as nodes in a graph, if the car can not pass between two
        # nodes, we connect the two nodes with an edge in the graph.
        # the car can pass the road only if we CANNOT find a path
        # from start node y=w to the end node y=0
        
        queue = deque([(None, w)])
        visited = set([(None, w)])
        while queue:
            x, y = queue.popleft()
            # y <= 5 means (x, y) can connect the end node y=0
            if y <= BOUND_MIN_DISTANCE:
                return "no"
            for obstacle in obstacles:
                if (obstacle[0], obstacle[1]) in visited:
                    continue
                if not self.is_connected(x, y, obstacle[0], obstacle[1]):
                    continue
                queue.append((obstacle[0], obstacle[1]))
                visited.add((obstacle[0], obstacle[1]))
        return "yes"
    
    def is_connected(self, x1, y1, x2, y2):
        if x1 is None:
            return abs(y1 - y2) <= BOUND_MIN_DISTANCE
        # check the distance between (x1, y1) and (x2, y2) <= 6
        # 6 = 2 x (car radius + obstacle radius)
        return abs(x1 - x2) ** 2 + abs(y1 - y2) ** 2 <= OBSTAClE_MIN_DISTANCE ** 2
```
pass