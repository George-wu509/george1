Lintcode 1413
给出两个list x，y，代表x[i]与y[i]之间有一条边，整个边集构成一棵树，`1`为根，现在有个list a,b，表示询问节点a[i]与b[i]是什么关系，如果a[i]与b[i]是兄弟，即有同一个父节点，输出`1`，如果a[i]与b[i]是父子关系，输出`2`，否则输出`0`。


**样例 1:**
```python
"""
输入； x = [1,1], y = [2,3], a =[1,2], b = [2,3]
输出：[2,1]
解释：1与2是父子关系，2与3是兄弟关系，它们的共同父节点为1。
```
解釋:
2 - 1 - 3    

**样例 2：**
```python
"""
输入：x = [1,1,2], y =[2,3,4], a = [1,2,1], b = [2,3,4]
输出：[2,1,0]。
解释：1与2是父子关系，2与3是兄弟关系，它们的共同父节点为1，1与4不是兄弟关系也不是父子关系。
```
解釋:
4 - 2 - 1 - 3    
graph = {1:{2,3}, 2:{1,4}, 3:{1}, 4:{2}}
parent = {1: None, 2: 1, 3:1, 4:2}

```python
class Solution:
    """
    @param x: The x
    @param y: The y
    @param a: The a
    @param b: The b
    @return: The Answer
    """
    def tree(self, x, y, a, b):
        graph = self.build_graph(x, y)
        parent = self.build_tree(graph)
        
        results = []
        for u, v in zip(a, b):
            if parent[u] == parent[v]:
                results.append(1)
            elif parent[u] == v or parent[v] == u:
                results.append(2)
            else:
                results.append(0)
        return results
    
    def build_graph(self, x, y):
        graph = {}
        for u, v in zip(x, y):
            if u not in graph:
                graph[u] = set()
            if v not in graph:
                graph[v] = set()
            graph[u].add(v)
            graph[v].add(u)
        return graph
        
    def build_tree(self, graph):
        from collections import deque
        visited = set([1])
        queue = deque([1])
        parent = {1: None}
        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)
                parent[neighbor] = node
        return parent
```
pass


# **LintCode 1413: O(1) Tree 解法分析**

## **解題目標**

給定一棵 **無向樹**，其邊由兩個陣列 `x` 和 `y` 表示，分別存儲樹中的邊 `(x[i], y[i])`，我們需要回答 **多組查詢 `(a[i], b[i])`**，判斷兩個節點 `a[i]` 和 `b[i]` 之間的關係：

1. 若 `a[i]` 和 `b[i]` 屬於**同一個子樹**，則回傳 `1`。
2. 若 `a[i]` 或 `b[i]` **是對方的直接父節點**，則回傳 `2`。
3. 否則回傳 `0`（兩者不在同一個子樹）。

---

## **解法核心**

這是一個 **樹的圖論問題**，我們可以拆解成兩個主要步驟：

1. **構建樹 (Graph Construction)**：
    - 透過 `x[i]` 和 `y[i]` 建立鄰接表 `graph` 來表示樹。
2. **標記父節點 (Tree Construction & Parent Mapping)**：
    - 使用 **BFS 遍歷樹**，確定每個節點的**父節點**，建立 `parent` 映射表。
3. **處理查詢**：
    - 根據 `parent` 映射表，判斷 `a[i]` 和 `b[i]` 的關係：
        - **同一個子樹 (`parent[u] == parent[v]`)** → 回傳 `1`
        - **父子關係 (`parent[u] == v or parent[v] == u`)** → 回傳 `2`
        - **否則 (`0`)**。

---

## **為何這樣解？**

1. **O(n) 建樹，O(1) 查詢**
    
    - **BFS 建樹 `O(n)`**：因為每個節點只有一個父節點，遍歷 `n-1` 條邊即可完成建樹。
    - **查詢 `O(1)`**：只需透過 `parent` 表來查找 `O(1)` 時間判斷關係。
2. **適用於較大數據範圍**
    
    - 若直接遍歷樹來檢查節點關係，每次 `O(n)`，查詢 `m` 次則 `O(nm)`，無法接受。
    - **透過 `parent` 表，所有查詢時間均為 `O(1)`**，非常高效。

---

## **解法步驟**

### **Step 1: 構建樹 (Graph Construction)**

使用 `x, y` 來建構 **鄰接表 (Adjacency List)** `graph`：

- **無向樹** 表示法：
    
    python
    
    複製編輯
    
    `graph[u].add(v) graph[v].add(u)`
    

**範例輸入**

text

複製編輯

`x = [1, 1, 2, 2, 3] y = [2, 3, 4, 5, 6]`

轉換成鄰接表：

text

複製編輯

`{   1: {2, 3},   2: {1, 4, 5},   3: {1, 6},   4: {2},   5: {2},   6: {3} }`

---

### **Step 2: BFS 建立 `parent` 表**

我們從 `1` **作為根節點 (Root)**，使用 **BFS (廣度優先搜索)** 建立 **父節點表 `parent`**：

text

複製編輯

`父節點表： {   1: None,  # 根節點沒有父節點   2: 1,   3: 1,   4: 2,   5: 2,   6: 3 }`

**BFS 建樹步驟**

|訪問節點|佇列狀態|記錄 `parent`|
|---|---|---|
|`1` (起點)|`[2, 3]`|`{1: None}`|
|`2`|`[3, 4, 5]`|`{2: 1}`|
|`3`|`[4, 5, 6]`|`{3: 1}`|
|`4`|`[5, 6]`|`{4: 2}`|
|`5`|`[6]`|`{5: 2}`|
|`6`|`[]`|`{6: 3}`|

---

### **Step 3: 查詢 `a[i], b[i]` 關係**

對於每個 `(a[i], b[i])`：

1. 若 `parent[u] == parent[v]`，回傳 `1`（**同一個子樹**）。
2. 若 `parent[u] == v or parent[v] == u`，回傳 `2`（**父子關係**）。
3. 否則回傳 `0`（**不同子樹**）。

#### **範例查詢**

text

複製編輯

`a = [4, 5, 3] b = [5, 6, 2]`

對應 `parent` 表查詢：

|`(a, b)`|`parent[a]`|`parent[b]`|結果|
|---|---|---|---|
|`(4, 5)`|`2`|`2`|`1` (同子樹)|
|`(5, 6)`|`2`|`3`|`0` (不同子樹)|
|`(3, 2)`|`1`|`1`|`1` (同子樹)|

最終輸出：

text

複製編輯

`[1, 0, 1]`

---

## **時間與空間複雜度分析**

### **時間複雜度**

|操作|複雜度|說明|
|---|---|---|
|**構建 `graph`**|`O(n)`|需要遍歷 `n-1` 條邊|
|**BFS 建立 `parent` 表**|`O(n)`|每個節點最多被訪問一次|
|**查詢 `m` 次**|`O(m)`|每次 `O(1)` 查詢|
|**總計**|`O(n + m)`|查詢非常高效|

### **空間複雜度**

|操作|空間複雜度|說明|
|---|---|---|
|**鄰接表 `graph`**|`O(n)`|存儲 `n-1` 條邊|
|**`parent` 表**|`O(n)`|每個節點存儲 1 個值|
|**佇列 `queue`**|`O(n)`|BFS 需要存儲 `n` 個節點|
|**總計**|`O(n)`|所有操作均為線性|

---

## **其他解法 (不寫 Code)**

1. **DFS 建立 `parent` 表 (`O(n)`)**
    
    - 使用 DFS 來建立 `parent` 表，時間複雜度相同 (`O(n)`)。
    - **適合遞歸操作**，但需要考慮**遞歸深度 (`O(n)`)**。
2. **並查集 (Union-Find, `O(n log n)`)**
    
    - 使用 **並查集** 來查找 **公共祖先**，判斷兩者是否在**同一個子樹**。
    - 若 `m` 很大，並查集會比 `O(1)` 查詢慢。
3. **暴力 DFS 查詢 (`O(nm)`)**
    
    - **暴力解法**：對於每個 `(a[i], b[i])`，使用 DFS 來判斷是否在同一子樹。
    - **缺點**：當 `m` 很大時，時間複雜度可能達到 `O(nm)`，不可接受。

---

## **最佳解法**

✅ **BFS 建 `parent` 表 + O(1) 查詢 (`O(n + m)`)**

- **高效建樹** (`O(n)`)
- **查詢僅需 `O(1)`**，適合 **大量查詢** (`m` 很大時) 🚀

---

**總結**： 這題的 **核心是建樹 (`O(n)`) + 查詢 (`O(1)`)**，利用 **BFS 建 `parent` 表** 可以讓查詢變得超快！💡