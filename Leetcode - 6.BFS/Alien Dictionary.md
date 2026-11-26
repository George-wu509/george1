
### LintCode 892: 外星人字典

---
比較: [[Compare - Topological Sort]]


|                    |     |
| ------------------ | --- |
| [[#### 外星人字典问题描述]] |     |
| [[#### 拓撲排序之前處理]]  |     |

#### 外星人字典问题描述

给定一组由外星语单词组成的字典 `words`，单词按外星语的字典序排序。通过比较相邻单词，推导出外星语的字母顺序。如果顺序无法确定，返回空字符串。

**输入**：
`["wrt", "wrf", "er", "ett", "rftt"]`

**输出**：
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
```python
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
```python
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
        neighbors = {ch: set() for word in words for ch in word}
        in_degrees = {ch: 0 for ch in neighbors.keys()}

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
                if len(first_word) > len(second_word):
                    return ""

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

        return "".join(result) if len(result) == len(neighbors) else ""
```
解釋版
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


topology sort之前的這段程式碼的唯一目的，就是「當偵探」**：從已經排好序的單字清單中，找出隱藏的「字母先後線索」，並把這些線索畫成一張**有向圖（`neighbors`）**與**前置條件計數表（`in_degrees`），交給後面的拓撲排序處理。

整個建圖流程可以拆成 **3 個步驟**：

### 步驟 1：把所有出現過的字母變成「節點」

```Python
neighbors = {ch: set() for word in words for ch in word}
in_degrees = {ch: 0 for ch in neighbors.keys()}
```

- **想法**：不管字母順序為何，先把清單裡所有不重複的字母抓出來建檔。
    
- **圖解狀態**（以 `["wrt", "wrf", "er", "ett", "rftt"]` 為例）：
    
    ```
    建立 5 個孤立節點，初始入度（被幾個字母擋在前面）全部為 0：
    
      [w] (in: 0)    [r] (in: 0)    [t] (in: 0)    [f] (in: 0)    [e] (in: 0)
    ```
    
    _(註：`neighbors` 用 `set()` 而不是 `list`，是為了防止重複加入同一條邊)_
    

### 步驟 2：兩兩比較「相鄰單字」，抓出第一個不同的字母

```Python
for i in range(len(words) - 1):
    first_word = words[i]
    second_word = words[i + 1]
    for j in range(min(len(first_word), len(second_word))):
        if first_word[j] != second_word[j]:
            if second_word[j] not in neighbors[first_word[j]]:
                neighbors[first_word[j]].add(second_word[j])
                in_degrees[second_word[j]] += 1
            break
```

- **為什麼只比相鄰單字？** 因為字典序有遞移性（如果 `A < B` 且 `B < C`，自然 `A < C`），不需要兩兩全部配對。
    
- **為什麼找到第一個不同字母就 `break`？** 就像英文字典裡 `"apple"` 排在 `"banana"` 前面，是因為第 1 個字母 `'a' < 'b'` 就分出勝負了；後面的 `'p'` 和 `'a'` 誰大誰小完全不影響排序，硬比較反而會得到錯誤線索！
    
- **圖解 4 組比對過程**：
    
    ```
    ① "wrt" vs "wrf"         ② "wrf" vs "er"          ③ "er" vs "ett"          ④ "ett" vs "rftt"
       w  r  [t]                [w] r  f                 e  [r]                   [e] t  t
       w  r  [f]                [e] r                    e  [t] t                 [r] f  t  t
       ▲  ▲   │                  │                       ▲   │                     │
      相同相同 └─▶ t ──▶ f       └─▶ w ──▶ e            相同  └─▶ r ──▶ t           └─▶ e ──▶ r
              (f 入度 +1)         (e 入度 +1)                 (t 入度 +1)             (r 入度 +1)
    ```
    
- **為什麼要檢查 `if second_word[j] not in neighbors[first_word[j]]`？** 假設後面又出現一組單字也推導出 `t ──▶ f`，因為 `neighbors['t']` 是 `set` 不會重複存，若沒加這個判斷，`in_degrees['f']` 就會被多加 1 次，導致後面拓撲排序時 `f` 的入度永遠無法降為 0。
    

### 步驟 3：`for ... else` 抓出「違反字典常識」的無效輸入

```Python

        else:
            if len(first_word) > len(second_word):
                return ""
```

- **Python 語法小提醒**：`for ... else` 的 `else` 只有在「內層 `for` 迴圈完全沒有觸發 `break`（也就是較短的單字完全是較長單字的前綴）」時才會執行。
    
- **為什麼這樣不合法？** 在任何語言的字典中，空白永遠排在字母前面，所以 `"app"` 一定排在 `"apple"` 前面。如果出現長字在短字前面，代表這份字典根本是錯的，直接回傳 `""` 結束程式：
    
    ```
    情境 A（合法，不觸發 return）：      情境 B（不合法，直接 return ""）：
       first : "ab"  (長度 2)              first : "abc" (長度 3)
       second: "abc" (長度 3)              second: "ab"  (長度 2)
                ▲▲                                  ▲▲
           前兩字完全相同，短字在前             前兩字完全相同，長字卻排在短字前面 ❌
    ```
    

### 建圖完成！準備交棒給拓撲排序

跑完上述迴圈後，程式拿到了兩份整理好的資料，準備進入下一階段的 `heap` 拓撲排序：

```
1. 有向圖結構 (neighbors)：
   [w] ──▶ [e] ──▶ [r] ──▶ [t] ──▶ [f]

2. 入度計數表 (in_degrees)：
   w: 0  (唯一沒有前置條件的字母，馬上進入 heap！)
   e: 1
   r: 1
   t: 1
   f: 1
```

---


#### 拓撲排序之前處理
```
以 ["wrt", "wrf", "er", "ett", "rftt"] 為例解釋以下code:
neighbors = {ch: set() for word in words for ch in word}
in_degrees = {ch: 0 for ch in neighbors.keys()}
neighbors跟in_degrees各是甚麼?

並解釋以下for loop每個迴圈 first_word , second_word是甚麼, 以及每個流程
```
### 一、`neighbors` 與 `in_degrees` 各是什麼？

這兩行程式碼透過字典推導式（Dictionary Comprehension），把 `words` 裡出現過的所有不重複字元（`'w', 'r', 't', 'f', 'e'`）抽出來初始化：

```Python
neighbors = {ch: set() for word in words for ch in word}
in_degrees = {ch: 0 for ch in neighbors.keys()}
```

- **`neighbors`（鄰接表 / 有向邊集合）**：
    - **意義**：記錄每個字母「**後面緊接著哪些字母**」（即 `key -> value` 的有向邊）。使用 `set()` 是為了避免重複加入同一條規則。
        
    - **初始值**：
        ```Python
        {'w': set(), 'r': set(), 't': set(), 'f': set(), 'e': set()}
        ```
        
- **`in_degrees`（入度計數表）**：
    
    - **意義**：記錄「**有多少個字母必須排在該字母前面**」（有幾個箭頭指向它）。`0` 代表目前沒有任何前置限制。
        
    - **初始值**：

        ```Python
        {'w': 0, 'r': 0, 't': 0, 'f': 0, 'e': 0}
        ```
        

### 二、`for` 迴圈逐輪拆解

`words` 長度為 `5`，所以外層 `for i in range(len(words) - 1)` 會執行 `i = 0, 1, 2, 3` 共 **4 輪**，每次取出相鄰的兩個單字進行逐字比對：

#### 第 1 輪：`i = 0`

- **`first_word`**：`"wrt"`（長度 3）
    
- **`second_word`**：`"wrf"`（長度 3）
    
- **內層迴圈範圍**：`range(min(3, 3))` $\rightarrow$ `j = 0, 1, 2`
    
- **執行流程**：
    
    1. `j = 0`：比較 `'w'` 與 `'w'` $\rightarrow$ 相同，繼續下一輪 `j`。
        
    2. `j = 1`：比較 `'r'` 與 `'r'` $\rightarrow$ 相同，繼續下一輪 `j`。
        
    3. `j = 2`：比較 `'t'` 與 `'f'` $\rightarrow$ **不相同**（`'t' != 'f'`），找到線索 `'t' -> 'f'`：
        
        - 檢查 `'f' not in neighbors['t']` 為 `True`。
            
        - `neighbors['t'].add('f')` $\rightarrow$ `neighbors['t']` 變成 `{'f'}`。
            
        - `in_degrees['f'] += 1` $\rightarrow$ `in_degrees['f']` 變成 `1`。
            
        - 執行 **`break`** 跳出內層迴圈（後續字元無須再比）。
            

#### 第 2 輪：`i = 1`

- **`first_word`**：`"wrf"`（長度 3）
    
- **`second_word`**：`"er"`（長度 2）
    
- **內層迴圈範圍**：`range(min(3, 2))` $\rightarrow$ `j = 0, 1`
    
- **執行流程**：
    
    1. `j = 0`：比較 `'w'` 與 `'e'` $\rightarrow$ **不相同**（`'w' != 'e'`），找到線索 `'w' -> 'e'`：
        
        - 檢查 `'e' not in neighbors['w']` 為 `True`。
            
        - `neighbors['w'].add('e')` $\rightarrow$ `neighbors['w']` 變成 `{'e'}`。
            
        - `in_degrees['e'] += 1` $\rightarrow$ `in_degrees['e']` 變成 `1`。
            
        - 執行 **`break`** 跳出內層迴圈（`j = 1` 不會執行）。
            

#### 第 3 輪：`i = 2`

- **`first_word`**：`"er"`（長度 2）
    
- **`second_word`**：`"ett"`（長度 3）
    
- **內層迴圈範圍**：`range(min(2, 3))` $\rightarrow$ `j = 0, 1`
    
- **執行流程**：
    
    1. `j = 0`：比較 `'e'` 與 `'e'` $\rightarrow$ 相同，繼續下一輪 `j`。
        
    2. `j = 1`：比較 `'r'` 與 `'t'` $\rightarrow$ **不相同**（`'r' != 't'`），找到線索 `'r' -> 't'`：
        
        - 檢查 `'t' not in neighbors['r']` 為 `True`。
            
        - `neighbors['r'].add('t')` $\rightarrow$ `neighbors['r']` 變成 `{'t'}`。
            
        - `in_degrees['t'] += 1` $\rightarrow$ `in_degrees['t']` 變成 `1`。
            
        - 執行 **`break`** 跳出內層迴圈。
            

#### 第 4 輪：`i = 3`

- **`first_word`**：`"ett"`（長度 3）
    
- **`second_word`**：`"rftt"`（長度 4）
    
- **內層迴圈範圍**：`range(min(3, 4))` $\rightarrow$ `j = 0, 1, 2`
    
- **執行流程**：
    
    1. `j = 0`：比較 `'e'` 與 `'r'` $\rightarrow$ **不相同**（`'e' != 'r'`），找到線索 `'e' -> 'r'`：
        
        - 檢查 `'r' not in neighbors['e']` 為 `True`。
            
        - `neighbors['e'].add('r')` $\rightarrow$ `neighbors['e']` 變成 `{'r'}`。
            
        - `in_degrees['r'] += 1` $\rightarrow$ `in_degrees['r']` 變成 `1`。
            
        - 執行 **`break`** 跳出內層迴圈（`j = 1, 2` 不會執行）。
            

### 三、迴圈結束後的最終資料狀態

4 輪外層迴圈全部跑完後，兩個字典的內容如下：

Python

```
neighbors = {
    'w': {'e'},   # w 指向 e
    'r': {'t'},   # r 指向 t
    't': {'f'},   # t 指向 f
    'f': set(),   # f 沒有指向任何人
    'e': {'r'}    # e 指向 r
}

in_degrees = {
    'w': 0,       # 唯一沒有前置條件的起點
    'r': 1,       # 被 e 指向
    't': 1,       # 被 r 指向
    'f': 1,       # 被 t 指向
    'e': 1        # 被 w 指向
}
```






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