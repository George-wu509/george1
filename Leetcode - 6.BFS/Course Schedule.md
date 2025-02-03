
lintcode 615

### LeetCode 207: **Course Schedule（课程表）**

---

#### **问题描述**

给定 `numCourses` 门课程，编号从 `0` 到 `numCourses-1`。另有一个先修课程的需求列表 `prerequisites`，其中每个元素 `[a, b]` 表示：

- 学习课程 `a` 之前必须先学习课程 `b`。

请判断是否可以完成所有课程。

---

#### **输入输出示例**

**输入：**

`numCourses = 2 prerequisites = [[1, 0]]`

**输出：**

`True`

**解释：**  
可以学习课程 0，然后学习课程 1。

---

**输入：**

`numCourses = 2 prerequisites = [[1, 0], [0, 1]]`

**输出：**

`False`

**解释：**  
无法完成课程学习，因为存在课程之间的循环依赖。

---

### **解题思路**

解决问题的关键是：**判断有向图是否有环**。

- 如果图中有环，则课程之间存在循环依赖，无法完成所有课程。
- 如果图中无环，则可以完成课程学习。

---

#### **方法 1：Kahn 拓扑排序算法**

1. **构建图和入度表：**
    
    - 使用邻接表 `graph` 表示课程之间的先修关系。
    - 使用数组 `in_degree` 记录每门课程的入度（依赖于其他课程的数量）。
2. **初始化队列：**
    
    - 将所有入度为 `0` 的课程加入队列 `queue`，它们可以直接学习。
3. **遍历图：**
    
    - 从队列中取出课程，表示当前课程可以学习，减少它指向的课程的入度。
    - 如果某门课程的入度变为 `0`，将其加入队列。
4. **检查结果：**
    
    - 如果遍历过程中可以学习的课程数量等于 `numCourses`，则可以完成所有课程。
    - 如果不能学习所有课程，则存在环。

---

#### **方法 2：DFS 检测环**

1. **构建图：**
    
    - 使用邻接表 `graph` 表示课程之间的依赖关系。
2. **DFS 遍历：**
    
    - 使用数组 `visited` 标记课程状态：
        - `0`：未访问。
        - `1`：访问中。
        - `2`：访问完成。
    - 当访问到某门课程时，将其标记为 `访问中`，并递归访问其所有依赖课程。
    - 如果递归过程中再次访问到某门 `访问中` 的课程，说明存在环。
3. **返回结果：**
    
    - 如果所有课程都能完成递归，说明无环，可以完成课程学习。
    - 如果检测到环，返回 `False`。

---
Example:
例1:
```python
输入: n = 2, prerequisites = [[1,0]] 
输出: true
```
例2:
```python
输入: n = 2, prerequisites = [[1,0],[0,1]] 
输出: false
```



### **方法 1：Kahn 算法实现**

#### **代码实现**

```python
from collections import deque, defaultdict
class Solution:
    """
    @param num_courses: a total of n courses
    @param prerequisites: a list of prerequisite pairs
    @return: true if can finish all courses or false
    """
    def can_finish(self, num_courses: int, prerequisites: List[List[int]]) -> bool:
        graph = defaultdict(list)
        in_degree = [0] * num_courses
        for course, pre in prerequisites:
            graph[pre].append(course)
            in_degree[course] += 1

        # 初始化队列
        queue = deque([i for i in range(num_courses) if in_degree[i] == 0])

        # 记录已完成的课程数量
        completed_courses = 0

        while queue:
            course = queue.popleft()
            completed_courses += 1

            # 减少依赖课程的入度
            for next_course in graph[course]:
                in_degree[next_course] -= 1
                if in_degree[next_course] == 0:
                    queue.append(next_course)

        # 如果已完成课程数等于总课程数，返回 True
        return completed_courses == num_courses
```
pass


#### **示例运行**

##### 示例 1：

输入：

`numCourses = 2 prerequisites = [[1, 0]]`

运行步骤：

1. **构建图和入度表：**

    `graph = {0: [1]} in_degree = [0, 1]`
    
2. **初始化队列：**

    `queue = deque([0])`
    
3. **处理课程 0：**
    - 从队列取出课程 0，完成课程数量 `completed_courses = 1`。
    - 更新课程 1 的入度：`in_degree = [0, 0]`。
    - 将课程 1 加入队列：`queue = deque([1])`。
4. **处理课程 1：**
    - 从队列取出课程 1，完成课程数量 `completed_courses = 2`。
    - 队列为空。

**输出：**

`True`

---

##### 示例 2：

输入：

`numCourses = 2 prerequisites = [[1, 0], [0, 1]]`

运行步骤：

1. **构建图和入度表：**

    `graph = {0: [1], 1: [0]} in_degree = [1, 1]`
    
2. **初始化队列：**

    `queue = deque([])`
    
3. **队列为空，无法完成课程。**

**输出：**

`False`

---

### **方法 2：DFS 实现**

#### **代码实现**

```python
def canFinish(numCourses, prerequisites):
    # 构建图
    graph = defaultdict(list)
    for course, pre in prerequisites:
        graph[pre].append(course)

    # 记录访问状态
    visited = [0] * numCourses

    def dfs(course):
        if visited[course] == 1:  # 访问中，说明存在环
            return False
        if visited[course] == 2:  # 已访问过，无需重复处理
            return True

        # 标记为访问中
        visited[course] = 1
        for next_course in graph[course]:
            if not dfs(next_course):
                return False

        # 标记为访问完成
        visited[course] = 2
        return True

    # 遍历每门课程
    for course in range(numCourses):
        if not dfs(course):
            return False

    return True

```

---

#### **示例运行**

##### 示例 1：

输入：

`numCourses = 2 prerequisites = [[1, 0]]`

运行步骤：

1. **构建图：**

    `graph = {0: [1]}`
    
2. **遍历课程 0：**
    - 访问课程 0：`visited = [1, 0]`。
    - 递归访问课程 1：`visited = [1, 1]`。
    - 完成课程 1：`visited = [1, 2]`。
    - 完成课程 0：`visited = [2, 2]`。
3. **所有课程均完成。**

**输出：**

`True`

##### 示例 2：

`numCourses = 2 prerequisites = [[1, 0], [0, 1]]`

运行步骤：

1. **构建图：**

    `graph = {0: [1], 1: [0]}`
    
2. **遍历课程 0：**
    - 访问课程 0：`visited = [1, 0]`。
    - 递归访问课程 1：`visited = [1, 1]`。
    - 再次访问课程 0，检测到环。

**输出：**

`False`

---

### **总结**

1. **两种解法：**
    - Kahn 算法适用于实时维护依赖关系的问题。
    - DFS 更适合图的递归遍历和检测环的问题。
2. **复杂度：**
    - 时间复杂度：均为 O(V+E)O(V + E)O(V+E)，其中 VVV 是课程数，EEE 是依赖关系数。
    - 空间复杂度：Kahn 算法为 O(V)O(V)O(V)，DFS 由于递归栈的存在为 O(V)O(V)O(V)。
3. **应用场景：**
    - Kahn 算法更适合返回拓扑排序。
    - DFS 更适合仅检测是否有环。