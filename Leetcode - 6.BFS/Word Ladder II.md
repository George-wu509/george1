
### LintCode 121: 单词接龙 II

---

### 问题描述

给定一个起始单词 `start` 和结束单词 `end`，以及一个单词字典 `dict`，找到所有从 `start` 到 `end` 的最短转换序列。

转换规则：

1. 每次只能改变一个字母。
2. 每次转换后的单词必须在字典中。

---

### 解法：BFS + DFS

此解法分为两个阶段：

1. 使用 **BFS** 构建单词图，找到从 `end` 到 `start` 的最短路径，并记录每个单词的距离。
2. 使用 **DFS** 从 `start` 出发，根据 BFS 的距离信息，构造所有可能的路径。

#### 详细步骤

1. **BFS 阶段**：
    
    - 从目标单词 `end` 开始，逐层向外扩展，计算每个单词到 `end` 的最短距离。
    - 使用队列和哈希表 `distance` 记录每个单词的最短距离。
2. **DFS 阶段**：
    
    - 从 `start` 开始，根据 BFS 记录的最短距离信息递归构建路径。
    - 只沿着距离递减的方向搜索，避免重复或非最短路径。
3. **结果**：
    
    - 所有从 `start` 到 `end` 的最短路径都保存在 `results` 中。

---
Example:
**样例 1：**
输入：
```
start = "a"
end = "c"
dict =["a","b","c"]
```
输出：
```
[["a","c"]]
```
解释：
"a"->"c"

**样例 2：**
输入：
```
start ="hit"
end = "cog"
dict =["hot","dot","dog","lot","log"]
```
输出：
```
[["hit","hot","dot","dog","cog"],["hit","hot","lot","log","cog"]]
```
解释：
1."hit"->"hot"->"dot"->"dog"->"cog"  
2."hit"->"hot"->"lot"->"log"->"cog"


### 代码实现
```python
from collections import deque

class Solution:
    def find_ladders(self, start, end, dict):
        # 将 start 和 end 加入字典
        dict = set(dict)
        dict.add(start)
        dict.add(end)
        
        # BFS 阶段：计算每个单词到 end 的最短距离
        distance = {}
        self.bfs(end, distance, dict)
        
        # DFS 阶段：递归构建路径
        results = []
        self.dfs(start, end, distance, dict, [start], results)
        
        return results

    def bfs(self, start, distance, dict):
        # BFS 初始化
        distance[start] = 0
        queue = deque([start])
        
        # BFS 遍历
        while queue:
            word = queue.popleft()
            for next_word in self.get_next_words(word, dict):
                if next_word not in distance:  # 如果 next_word 未访问
                    distance[next_word] = distance[word] + 1
                    queue.append(next_word)
    
    def get_next_words(self, word, dict):
        # 找到所有与 word 相邻的单词（改变一个字母后在 dict 中的单词）
        words = []
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                next_word = word[:i] + c + word[i + 1:]
                if next_word != word and next_word in dict:
                    words.append(next_word)
        return words
                        
    def dfs(self, curt, target, distance, dict, path, results):
        # 如果当前单词等于目标单词，记录路径
        if curt == target:
            results.append(list(path))
            return
        
        # 遍历所有邻接单词
        for word in self.get_next_words(curt, dict):
            if distance[word] != distance[curt] - 1:  # 必须沿着最短路径递减方向
                continue
            path.append(word)  # 选择当前单词
            self.dfs(word, target, distance, dict, path, results)  # 递归
            path.pop()  # 回溯


```
pass

---

### 示例输入输出

#### 输入

python

複製程式碼

`start = "hit" end = "cog" dict = ["hot", "dot", "dog", "lot", "log"]`

#### 运行过程

1. **BFS 阶段**：
    
    - 初始状态：`queue = deque(["cog"])`, `distance = {"cog": 0}`。
    - 第一轮：
        - 当前单词：`cog`。
        - 相邻单词：`dog`, `log`。
        - 更新：`distance = {"cog": 0, "dog": 1, "log": 1}`。
        - 队列：`queue = deque(["dog", "log"])`。
    - 第二轮：
        - 当前单词：`dog`。
        - 相邻单词：`dot`。
        - 更新：`distance = {"cog": 0, "dog": 1, "log": 1, "dot": 2}`。
        - 队列：`queue = deque(["log", "dot"])`。
    - 第三轮：
        - 当前单词：`log`。
        - 相邻单词：`lot`。
        - 更新：`distance = {"cog": 0, "dog": 1, "log": 1, "dot": 2, "lot": 2}`。
        - 队列：`queue = deque(["dot", "lot"])`。
    - 第四轮：
        - 当前单词：`dot`。
        - 相邻单词：`hot`。
        - 更新：`distance = {"cog": 0, "dog": 1, "log": 1, "dot": 2, "lot": 2, "hot": 3}`。
        - 队列：`queue = deque(["lot", "hot"])`。
    - 第五轮：
        - 当前单词：`lot`。
        - 相邻单词：无更新。
    - 最终 `distance`：
        
        python
        
        複製程式碼
        
        `distance = {"cog": 0, "dog": 1, "log": 1, "dot": 2, "lot": 2, "hot": 3, "hit": 4}`
        
2. **DFS 阶段**：
    
    - 从 `hit` 出发递归搜索。
    - 搜索路径：
        - `hit -> hot -> dot -> dog -> cog`
        - `hit -> hot -> lot -> log -> cog`
    - 最终结果：
        
        python
        
        複製程式碼
        
        `[["hit", "hot", "dot", "dog", "cog"], ["hit", "hot", "lot", "log", "cog"]]`
        

---

### 时间和空间复杂度分析

#### 时间复杂度

1. **BFS 阶段**：
    
    - 每个单词最多被访问一次，计算其相邻单词需遍历所有可能的字母替换，复杂度为 `O(L * 26 * N)`。
        - `L` 是单词长度，`N` 是字典中单词数量。
2. **DFS 阶段**：
    
    - 每条路径最多包含 `L` 个单词，路径数量为结果数量 `R`。
    - 复杂度为 `O(R * L)`。

总时间复杂度为 **`O(L * 26 * N + R * L)`**。

#### 空间复杂度

1. **BFS 阶段**：
    
    - 需要存储 `distance` 和队列，复杂度为 `O(N)`。
2. **DFS 阶段**：
    
    - 递归深度最多为 `L`，路径存储复杂度为 `O(R * L)`。

总空间复杂度为 **`O(N + R * L)`**。

---

### 其他解法简述

#### 1. 双向 BFS

- 从 `start` 和 `end` 同时进行 BFS，搜索空间减半。
- 时间复杂度更低，但实现复杂度较高。

#### 2. 动态规划

- 使用 DP 记录从 `start` 到 `end` 的所有路径。
- 适用于路径数量较少的情况。

---

### 总结

- 本解法结合 **BFS** 和 **DFS**，先计算最短路径再构造路径，适用于路径数量较多的情况。
- 时间复杂度为 **`O(L * 26 * N + R * L)`**，适合中等规模数据。
- **双向 BFS** 提供了更高效的替代方案，但实现复杂度较高。

