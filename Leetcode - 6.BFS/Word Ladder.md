
LintCode 120「单词接龙」（Word Ladder）问题要求找出从起始单词到目标单词的最短转换序列长度，每次只能改变一个字母，且每个中间单词必须存在于给定的字典中。

以下是详细的中文解法，包括 BFS 解法的完整逐步解释、复杂度分析、以及其他解法的简述。

---

### 问题描述

输入：

1. `start`：起始单词。
2. `end`：目标单词。
3. `dict`：包含所有可能中间单词的集合。

要求：

- 每次改变一个字母，结果单词必须存在于 `dict` 中。
- 找到从 `start` 到 `end` 的最短路径长度。如果无法到达 `end`，返回 `0`。

---

### BFS 解法

#### 思路

使用广度优先搜索（BFS）来解决最短路径问题：

1. 将每个单词视为图中的一个节点。
2. 如果两个单词之间可以通过改变一个字母相连，则它们之间有一条边。
3. 使用队列实现 BFS，每次从当前单词扩展到所有相邻的单词，直到找到目标单词。

---

#### 步骤

1. **初始化队列和集合**：
    
    - 使用队列 `queue` 存储当前单词和路径长度，初始化为 `queue = deque([(start, 1)])`。
    - 将 `dict` 转为集合，方便快速查找。
2. **BFS 遍历**：
    
    - 从队列中取出当前单词和路径长度。
    - 遍历当前单词的每一位字母，用 `'a'` 到 `'z'` 替换这一位，生成所有可能的新单词。
    - 如果生成的新单词是目标单词 `end`，返回当前路径长度 +1。
    - 如果生成的新单词在 `dict` 中，则将其加入队列，并从 `dict` 中移除以避免重复访问。
3. **结束条件**：
    
    - 如果队列为空且未找到目标单词，则返回 `0`。

Example:
样例 1：
输入：
start = "a"
end = "c"
dict =["a","b","c"]
输出：
2
解释：
"a"->"c"

样例 2：
输入：
start ="hit"
end = "cog"
dict =["hot","dot","dog","lot","log"]
输出：
5
解释：
"hit"->"hot"->"dot"->"dog"->"cog"

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
2
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
5
```
解释：
"hit"->"hot"->"dot"->"dog"->"cog"

#### 代码实现

```python
from collections import deque

def ladder_length(start, end, dict):
    if not start or not end or not dict:
        return 0

    # 将目标单词加入字典
    dict.add(end)
    queue = deque([(start, 1)])  # (当前单词, 当前路径长度)

    while queue:
        word, length = queue.popleft()

        # 遍历当前单词的每一位字母
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                # 替换第 i 个字母生成新单词
                new_word = word[:i] + c + word[i+1:]

                # 如果是目标单词，返回路径长度
                if new_word == end:
                    return length + 1

                # 如果新单词在字典中，加入队列并移除
                if new_word in dict:
                    queue.append((new_word, length + 1))
                    dict.remove(new_word)

    return 0  # 无法到达目标单词

```
pass

---

#### 示例输入输出

##### 示例 1

**输入**：

`start = "hit" end = "cog" dict = {"hot", "dot", "dog", "lot", "log"}`

**运行过程**：

1. 初始化：
    
    - `queue = deque([("hit", 1)])`。
    - `dict = {"hot", "dot", "dog", "lot", "log", "cog"}`。
2. 第 1 步：
    
    - 当前单词 `hit`，路径长度 `1`。
    - 替换字母生成可能的新单词：`hot`。
    - 将 `hot` 加入队列：`queue = deque([("hot", 2)])`。
    - 更新字典：`dict = {"dot", "dog", "lot", "log", "cog"}`。
3. 第 2 步：
    
    - 当前单词 `hot`，路径长度 `2`。
    - 替换字母生成可能的新单词：`dot`、`lot`。
    - 将 `dot` 和 `lot` 加入队列：`queue = deque([("dot", 3), ("lot", 3)])`。
    - 更新字典：`dict = {"dog", "log", "cog"}`。
4. 第 3 步：
    
    - 当前单词 `dot`，路径长度 `3`。
    - 替换字母生成可能的新单词：`dog`。
    - 将 `dog` 加入队列：`queue = deque([("lot", 3), ("dog", 4)])`。
    - 更新字典：`dict = {"log", "cog"}`。
5. 第 4 步：
    
    - 当前单词 `dog`，路径长度 `4`。
    - 替换字母生成可能的新单词：`cog`。
    - 找到目标单词，返回路径长度 `5`。

**输出**：

`5`

---

#### 时间复杂度分析

1. **单词替换复杂度**： 每个单词长度为 `L`，每次替换 26 个字母，共有 `26 * L` 种可能。
    
2. **遍历所有单词**： 字典大小为 `N`，最坏情况下需要遍历所有单词。
    
3. **总复杂度**： 时间复杂度为 `O(N * L * 26)`，即 `O(N * L)`，其中：
    
    - `N` 是字典中单词的数量。
    - `L` 是单词的平均长度。
4. **空间复杂度**： 队列中最多存储 `N` 个单词，空间复杂度为 `O(N)`。
    

---

### 其他解法简述

1. **双向 BFS**：
    
    - 同时从 `start` 和 `end` 开始进行 BFS，每次扩展较小的一端。
    - 时间复杂度较普通 BFS 更低，约为 `O((N * L) / 2)`。
2. **图的预构建**：
    
    - 预先将字典中所有单词构造成图结构，每个节点的邻居是只差一个字母的单词。
    - 通过 BFS 或 DFS 进行搜索。
    - 构建图的复杂度为 `O(N * L^2)`。
3. __A_ 算法_*：
    
    - 使用启发式搜索，通过估计当前单词到目标单词的距离（如字符差异）进行路径搜索。
    - 在某些情况下可以减少实际搜索的节点数量。

---

### 总结

BFS 是最直观且易于实现的方法，适合解决最短路径问题。双向 BFS 是更高效的变种。对于更复杂的场景，可以考虑使用图结构或 A* 算法。希望以上解释和例子能帮助你理解该问题！



在 `LintCode 120 单词接龙` 的传统 BFS 解法中，每次都对单词的每个字符逐一尝试从 `'a'` 到 `'z'` 的替换，目的是模拟所有可能的路径。然而，这种方法可能效率较低，因为会生成许多无效单词（它们根本不在字典中）。你提出的改进方法非常合理——只尝试 `dict` 中实际存在的单词，可以显著优化效率。

以下是详细说明为什么传统方法需要遍历字母，并基于你的建议提供新代码和两种方法的对比分析。

---

### 原因：传统解法为什么遍历所有字母？

1. **广度优先搜索的特点**：
    
    - BFS 需要探索所有可能的路径，以确保找到最短路径。
    - 对于每个单词，我们必须尝试所有可能的替换，因为不知道哪些替换能生成有效单词。
2. **避免无效路径**：
    
    - 字符替换后需要验证新单词是否在 `dict` 中。如果直接跳过字母替换的过程，可能遗漏潜在的邻居节点。
3. **效率问题**：
    
    - 字符替换会生成大量无效单词，这种暴力方法的效率取决于 `L`（单词长度）和 `26` 个字母的枚举。

---

### 改进：直接针对字典中的单词

如果我们可以提前计算出每个单词的邻居（即字典中与当前单词只相差一个字母的单词），可以直接在 BFS 中只扩展实际有效的邻居节点。这样避免了无效单词的生成。

#### 思路

1. **预处理构造邻居关系**：
    
    - 遍历字典中的所有单词。
    - 判断字典中哪些单词可以通过改变一个字符相连，构造一个图。
    - 这个图的每个节点是字典中的单词，边表示两者可以通过改变一个字母相连。
2. **在 BFS 中直接使用邻居**：
    
    - 在 BFS 阶段，只扩展字典中预先计算的邻居节点。

---

### 新代码

```python
from collections import defaultdict, deque

def ladderLength_optimized(start, end, dict):
    if not start or not end or not dict:
        return 0

    # 将目标单词加入字典
    dict.add(end)

    # 构建图：字典中每个单词的邻居
    neighbors = defaultdict(list)
    word_list = list(dict)
    n = len(word_list)
    
    for i in range(n):
        for j in range(i + 1, n):
            if is_one_letter_diff(word_list[i], word_list[j]):
                neighbors[word_list[i]].append(word_list[j])
                neighbors[word_list[j]].append(word_list[i])

    # BFS 搜索
    queue = deque([(start, 1)])  # (当前单词, 当前路径长度)
    visited = set()
    visited.add(start)

    while queue:
        word, length = queue.popleft()

        for neighbor in neighbors[word]:
            if neighbor == end:
                return length + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, length + 1))

    return 0

# 辅助函数：判断两个单词是否只差一个字母
def is_one_letter_diff(word1, word2):
    if len(word1) != len(word2):
        return False
    diff = sum(c1 != c2 for c1, c2 in zip(word1, word2))
    return diff == 1

```

---

### 示例输入输出

#### 示例 1

输入：
`start = "hit" end = "cog" dict = {"hot", "dot", "dog", "lot", "log"}`

输出：
`5`

#### 运行过程

1. **构造邻居关系**：

    `neighbors = {     "hot": ["dot", "lot"],     "dot": ["hot", "dog"],     "dog": ["dot", "log", "cog"],     "lot": ["hot", "log"],     "log": ["lot", "dog", "cog"],     "cog": ["dog", "log"] }`
    
2. **BFS 搜索**：
    
    - 起始队列：`queue = deque([("hit", 1)])`。
    - 第 1 步：`hit -> hot`，`queue = deque([("hot", 2)])`。
    - 第 2 步：`hot -> dot, lot`，`queue = deque([("dot", 3), ("lot", 3)])`。
    - 第 3 步：`dot -> dog`，`queue = deque([("lot", 3), ("dog", 4)])`。
    - 第 4 步：`dog -> cog`，找到目标单词，返回 `5`。

---

### 方法对比

|比较点|原始 BFS 方法|优化 BFS 方法|
|---|---|---|
|**时间复杂度**|`O(N * L * 26)`|`O(N^2 * L + N * L)`|
|**空间复杂度**|`O(N + L)`|`O(N^2 + N)`|
|**生成无效单词**|会生成大量无效单词|仅针对字典中的有效单词|
|**预处理时间**|无预处理|构造邻居图需要额外时间|
|**代码实现难度**|较简单|略复杂，需要构造邻居图|
|**适用场景**|适合字典较小，单词较短的情况|适合字典较大，需要频繁查询的情况|

#### 时间复杂度说明

1. **原始 BFS**：
    
    - 每个单词需要尝试替换 `L` 个字符，每次替换 `26` 种字母，共计 `O(N * L * 26)`。
    - `N` 是字典大小，`L` 是单词长度。
2. **优化 BFS**：
    
    - 预处理构造邻居需要比较 `N^2` 对单词，每次比较耗时 `O(L)`，总计 `O(N^2 * L)`。
    - BFS 阶段只需要遍历每个单词的邻居，总计 `O(N * L)`。

---

### 总结

- **原始 BFS 方法**：实现简单，适合小规模数据，但性能较差。
- **优化 BFS 方法**：通过预处理邻居关系，大幅减少无效单词的生成，适合大规模数据。
- 如果字典中单词较多且需要多次查询，建议使用优化方法，否则原始方法更易于实现。

希望这两种方法的对比和代码实现能帮助你理解问题！