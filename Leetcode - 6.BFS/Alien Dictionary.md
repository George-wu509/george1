
### LintCode 892: 外星人字典

---

### 问题描述

给定一组由外星语单词组成的字典 `words`，单词按外星语的字典序排序。通过比较相邻单词，推导出外星语的字母顺序。如果顺序无法确定，返回空字符串。

**输入**：

python

複製程式碼

`["wrt", "wrf", "er", "ett", "rftt"]`

**输出**：

python

複製程式碼

`"wertf"`

---

### 解法：BFS（拓扑排序）

#### 思路

1. **构建图**：
    
    - 用有向图表示字母之间的顺序关系，节点是字母，边是字母之间的先后关系。
    - 每个字母的 "入度" 表示有多少其他字母需要在它之前。
2. **比较相邻单词**：
    
    - 遍历每对相邻单词，比较它们第一个不同的字符，确定顺序。
    - 如果较短的单词是较长单词的前缀，则不可能有有效的顺序，直接返回空字符串。
3. **拓扑排序**：
    
    - 将入度为 0 的字母加入优先队列（最小堆），按字典序顺序取出。
    - 遍历其邻居节点，减少邻居节点的入度；如果某邻居节点入度为 0，加入优先队列。
4. **结果验证**：
    
    - 如果最终排序结果的长度小于字母总数，则图中有环，返回空字符串。

---
Example:
**样例 1:**
```
输入：["wrt","wrf","er","ett","rftt"]
输出："wertf"
解释：
从 "wrt"和"wrf" ,我们可以得到 't'<'f'
从 "wrt"和"er" ,我们可以得到 'w'<'e'
从 "er"和"ett" ,我们可以得到 'r'<'t'
从 "ett"和"rftt" ,我们可以得到 'e'<'r'
所以返回 "wertf"
```
**样例 2:**
```
输入：["z","x"]
输出："zx"
解释：
从 "z" 和 "x"，我们可以得到 'z' < 'x'
所以返回"zx"
```



### 代码实现

```python
import heapq
from typing import List

class Solution:
    def alien_order(self, words: List[str]) -> str:
        # 初始化图结构
        neighbors = {ch: set() for word in words for ch in word}
        in_degrees = {ch: 0 for ch in neighbors.keys()}

        # 比较相邻单词，构建图
        for i in range(len(words) - 1):
            first_word = words[i]
            second_word = words[i + 1]
            for j in range(min(len(first_word), len(second_word))):
                if first_word[j] != second_word[j]:
                    if second_word[j] not in neighbors[first_word[j]]:
                        neighbors[first_word[j]].add(second_word[j])
                        in_degrees[second_word[j]] += 1
                    break
            else:
                # 检查前缀关系是否合法
                if len(first_word) > len(second_word):
                    return ""

        # 拓扑排序（使用优先队列）
        heap = [ch for ch in in_degrees if in_degrees[ch] == 0]
        heapq.heapify(heap)
        result = []

        while heap:
            ch = heapq.heappop(heap)
            result.append(ch)
            for neighbor in neighbors[ch]:
                in_degrees[neighbor] -= 1
                if in_degrees[neighbor] == 0:
                    heapq.heappush(heap, neighbor)

        # 如果排序结果长度小于节点总数，说明有环
        return "".join(result) if len(result) == len(neighbors) else ""

```
pass


### 示例输入输出

#### 输入

python

複製程式碼

`words = ["wrt", "wrf", "er", "ett", "rftt"]`

#### 输出

python

複製程式碼

`"wertf"`

---

### 运行过程

1. **初始化图结构**：
    
    - `neighbors`：
        
        python
        
        複製程式碼
        
        `{'w': set(), 'r': set(), 't': set(), 'f': set(), 'e': set()}`
        
    - `in_degrees`：
        
        python
        
        複製程式碼
        
        `{'w': 0, 'r': 0, 't': 0, 'f': 0, 'e': 0}`
        
2. **构建图**：
    
    - 比较 `"wrt"` 和 `"wrf"`：
        - `'t' < 'f'`，加入边 `t -> f`。
    - 比较 `"wrt"` 和 `"er"`：
        - `'w' < 'e'`，加入边 `w -> e`。
    - 比较 `"er"` 和 `"ett"`：
        - `'r' < 't'`，加入边 `r -> t`。
    - 比较 `"ett"` 和 `"rftt"`：
        - `'e' < 'r'`，加入边 `e -> r`。
    
    **最终图**：
    
    - `neighbors`：
        
        python
        
        複製程式碼
        
        `{'w': {'e'}, 'r': {'t'}, 't': {'f'}, 'f': set(), 'e': {'r'}}`
        
    - `in_degrees`：
        
        python
        
        複製程式碼
        
        `{'w': 0, 'r': 1, 't': 1, 'f': 1, 'e': 1}`
        
3. **拓扑排序**：
    
    - 初始入度为 0 的节点：`['w']`。
    - 排序过程：
        - 取出 `'w'`，加入结果：`result = ['w']`。
            - 更新 `'e'` 的入度：`in_degrees['e'] -= 1`。
        - 取出 `'e'`，加入结果：`result = ['w', 'e']`。
            - 更新 `'r'` 的入度：`in_degrees['r'] -= 1`。
        - 取出 `'r'`，加入结果：`result = ['w', 'e', 'r']`。
            - 更新 `'t'` 的入度：`in_degrees['t'] -= 1`。
        - 取出 `'t'`，加入结果：`result = ['w', 'e', 'r', 't']`。
            - 更新 `'f'` 的入度：`in_degrees['f'] -= 1`。
        - 取出 `'f'`，加入结果：`result = ['w', 'e', 'r', 't', 'f']`。
4. **验证结果**：
    
    - `result` 长度等于节点数，返回结果：`"wertf"`。

---

### 时间和空间复杂度分析

#### 时间复杂度

1. **构建图**：
    
    - 遍历每对相邻单词，复杂度为 `O(L)`，其中 `L` 是所有单词字符总数。
2. **拓扑排序**：
    
    - 遍历每个节点及其邻居，复杂度为 `O(V + E)`，其中 `V` 是字母数，`E` 是边数。

总时间复杂度为 **`O(L + V + E)`**。

#### 空间复杂度

1. **图存储**：
    - `neighbors` 和 `in_degrees`，复杂度为 `O(V + E)`。
2. **优先队列**：
    - 最多存储 `V` 个节点，复杂度为 `O(V)`。

总空间复杂度为 **`O(V + E)`**。

---

### 其他解法简述

#### 1. DFS（深度优先搜索）

- 使用递归对图进行深度优先遍历，按完成时间倒序排列即为拓扑排序。
- 时间复杂度和空间复杂度均为 `O(V + E)`。

#### 2. 普通 BFS（无需优先队列）

- 使用普通队列代替优先队列，拓扑排序不保证字典序。
- 时间复杂度和空间复杂度与当前解法相同。

---

### 总结

- **BFS 解法**通过拓扑排序解决问题，使用优先队列保证字典序，适合生成唯一排序。
- 时间复杂度为 **`O(L + V + E)`**，空间复杂度为 **`O(V + E)`**，适用于中等规模问题。
- **DFS 解法**提供了另一种实现方式，适合任务处理顺序问题。