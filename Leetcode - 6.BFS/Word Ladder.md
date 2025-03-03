
LintCode 120「单词接龙」（Word Ladder）问题要求找出从起始单词到目标单词的最短转换序列长度，每次只能改变一个字母，且每个中间单词必须存在于给定的字典中。
#### **题目描述**

给定两个单词 _start_ 和 _end_，以及一个字典 `dict`，找到从 `start` 到 `end` 的最短转换序列的长度，其中：
1. 每次只能更改一个字符。
2. 更改后的单词必须存在于字典 `dict` 中。
#### **示例**
```python
start = "hit"
end = "cog"
dict = ["hot", "dot", "dog", "lot", "log", "cog"]
```
##### **转换路径**

`hit -> hot -> dot -> dog -> cog`

最短路径长度为 `5`。

---
## **三种解法**

本题可以用 **DFS 递归**、**DFS 非递归（显式栈）** 和 **BFS** 来解决。

---

## **方法 1: DFS 递归**

**解题思路**

- **深度优先搜索 (DFS)** 适用于搜索路径，但由于要找**最短路径**，普通 DFS 可能会导致无效搜索较多，需要**剪枝**。
- 在递归过程中，我们维护一个 `visited` 集合，防止重复访问。
- 若当前单词等于 `end`，更新全局 `min_length`。

### **步骤**

1. **构建单词字典集合 (`word_set`)**，便于 O(1) 查找是否存在某个单词。
2. **DFS 递归搜索**：
    - 递归遍历所有可能的单词（改变一个字符形成新单词）。
    - 如果新单词在 `word_set` 且未访问，则递归继续。
    - 记录当前路径的长度，若找到 `end`，更新 `min_length`。
3. **返回 `min_length`**，若不可达返回 `0`。

### **时间复杂度**

- **最坏情况下**，遍历所有可能的路径，时间复杂度为 `O(N!)` (N 为单词数量)。
- **优化后**（剪枝），接近 `O(26L * N)`，其中 `L` 是单词长度。

---

## **方法 2: DFS 非递归（使用栈）**

**解题思路**

- 与递归 DFS 相同，但改为 **显式栈** 进行深度优先搜索。
- 用 **显式栈** 维护搜索路径，避免 Python 递归深度限制。

### **步骤**

1. **构建 `word_set` 以便快速查询单词是否存在。**
2. **初始化 `stack` 作为 DFS 栈**，存放 `(当前单词, 当前路径长度)`。
3. **深度优先搜索 (使用 `stack`)**：
    - 每次取出栈顶元素，遍历所有可能的一步变换。
    - 若新单词在 `word_set` 且未访问，则入栈。
    - 记录最短路径长度。

### **时间复杂度**

- 由于仍然是深度优先搜索，其复杂度与递归版本类似，`O(26L * N)`。

---

## **方法 3: BFS（最优解）**

**解题思路**

- **广度优先搜索 (BFS) 是求最短路径的最佳方法**。
- BFS 按层遍历，每次转换一步，保证第一次到达 `end` 的路径是最短的。

### **步骤**

1. **构建 `word_set` 以便快速查询。**
2. **初始化 `queue`，存放 `(当前单词, 当前路径长度)`。**
3. **BFS 遍历**：
    - 每次取出队首元素，尝试所有可能的一步变换。
    - 若找到 `end`，返回路径长度。
    - 若新单词在 `word_set` 且未访问，则入队。
4. **如果 BFS 结束仍未找到 `end`，返回 `0`。**

### **时间复杂度**

- **每个单词最多尝试 `26L` 种变化**，最多访问 `N` 个单词，时间复杂度为： O(26L×N)O(26L \times N)O(26L×N)
- **空间复杂度**：`O(N)`（用于 `queue`）。

## **总结**

|方法|适用情况|时间复杂度|额外空间|
|---|---|---|---|
|**DFS 递归**|适用于所有路径搜索，容易超时|`O(26L * N)`|`O(N)`|
|**DFS 非递归 (栈)**|适用于路径搜索，避免递归栈溢出|`O(26L * N)`|`O(N)`|
|**BFS (最优解)**|适用于最短路径，保证最优解|`O(26L * N)`|`O(N)`|

**推荐解法：BFS**（能最快找到最短路径）。
如果问题要求 **最短路径**，BFS 是最佳解法，因为它按层遍历，保证第一次找到 `end` 的路径是最短的。

方法 1: DFS 递归
```python
class Solution:
    def ladder_length(self, start, end, word_list):
        word_set = set(word_list)
        if end not in word_set:
            return 0
        
        self.min_length = float('inf')
        self.dfs(start, end, word_set, set(), 1)
        return self.min_length if self.min_length != float('inf') else 0
    
    def dfs(self, word, end, word_set, visited, depth):
        if word == end:
            self.min_length = min(self.min_length, depth)
            return
        
        visited.add(word)
        
        for next_word in self.get_neighbors(word, word_set):
            if next_word not in visited:
                self.dfs(next_word, end, word_set, visited, depth + 1)
        
        visited.remove(word)  # 回溯
    
    def get_neighbors(self, word, word_set):
        neighbors = []
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                if c != word[i]:
                    new_word = word[:i] + c + word[i+1:]
                    if new_word in word_set:
                        neighbors.append(new_word)
        return neighbors

```
DFS 递归版特点
- 适合所有路径搜索，但会超时，适用于找所有路径（不是最短路径）。
- **时间复杂度**: `O(26L * N)`
- **空间复杂度**: `O(N)`

方法 2: DFS 非递归（显式栈）
```python
class Solution:
    def ladder_length(self, start, end, word_list):
        word_set = set(word_list)
        if end not in word_set:
            return 0
        
        stack = [(start, 1)]
        visited = set()

        while stack:
            word, depth = stack.pop()
            if word == end:
                return depth

            visited.add(word)
            for next_word in self.get_neighbors(word, word_set):
                if next_word not in visited:
                    stack.append((next_word, depth + 1))

        return 0
    
    def get_neighbors(self, word, word_set):
        neighbors = []
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                if c != word[i]:
                    new_word = word[:i] + c + word[i+1:]
                    if new_word in word_set:
                        neighbors.append(new_word)
        return neighbors

```
DFS 非递归版特点
- 适用于路径搜索，但容易走较长的路径（不保证最短路径）。
- **时间复杂度**: `O(26L * N)`
- **空间复杂度**: `O(N)`

方法 3: BFS（最优解）
```python
from collections import deque

class Solution:
    def ladder_length(self, start, end, word_list):
        word_set = set(word_list)
        if end not in word_set:
            return 0

        queue = deque([(start, 1)])  # (当前单词, 当前路径长度)

        while queue:
            word, depth = queue.popleft()
            if word == end:
                return depth

            for next_word in self.get_neighbors(word, word_set):
                word_set.remove(next_word)  # 标记访问
                queue.append((next_word, depth + 1))

        return 0
    
    def get_neighbors(self, word, word_set):
        neighbors = []
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                if c != word[i]:
                    new_word = word[:i] + c + word[i+1:]
                    if new_word in word_set:
                        neighbors.append(new_word)
        return neighbors

```

BFS 版特点
- **保证找到的路径是最短路径**。
- **时间复杂度**: `O(26L * N)`