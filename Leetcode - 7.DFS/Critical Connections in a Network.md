
### **LintCode 1271：查找集群內的「關鍵連接」**

#### **題目描述**

給定一個無向圖，該圖由 `n` 個節點和一些邊組成，找出所有的「關鍵連接」。關鍵連接是指，如果移除這條邊，則圖將變得不連通。

---

### **DFS 解法**

#### **算法思路**

此問題可以用 Tarjan 算法來解決，它是一種基於深度優先搜索（DFS）的技術，用於發現圖中的橋（關鍵連接）。主要步驟如下：

1. **構建鄰接表**：
    
    - 將邊 `connections` 轉換為鄰接表，方便訪問每個節點的鄰居。
2. **使用 DFS 查找橋**：
    
    - 定義 `rank[node]` 為節點在 DFS 過程中的深度（訪問順序）。
    - 在遍歷鄰居節點時，通過比較 `rank` 值判斷是否存在橋。
3. **判斷關鍵連接**：
    
    - 若 `rank[neighbor] > rank[node]`，則 `(node, neighbor)` 是一條橋。
4. **結果輸出**：
    
    - 將所有的橋存入結果集合 `result`，最終返回。

---
Example:
```plain
Input:
4
[[0,1],[1,2],[2,0],[1,3]]
Output: [[1,3]]
```


#### **代碼詳解**
```python
class Solution:
    def critical_connectionsina_network(self, n, connections):
        # 構建鄰接表
        graph = self.build_graph(n, connections)
        print("Graph:", graph)

        # 初始化 rank 數組，-float('inf') 表示未訪問
        rank = [-float('inf')] * n
        result = set()
        
        # 將邊存入集合，方便後續移除
        for connection in connections:
            result.add(tuple(sorted(connection)))

        # 從節點 0 開始 DFS
        self.dfs(graph, 0, 0, rank, result, n)
        return list(result)
    
    def build_graph(self, n, connections):
        # 構建鄰接表
        graph = [[] for i in range(n)]
        for connection in connections:
            graph[connection[0]].append(connection[1])
            graph[connection[1]].append(connection[0])
        return graph
        
    def dfs(self, graph, node, depth, rank, result, n):
        if rank[node] >= 0:
            return rank[node]  # 節點已被訪問，返回其深度
        
        rank[node] = depth  # 設置當前節點的深度
        min_depth = n  # 初始化最小深度
        
        for neighbor in graph[node]:
            if rank[neighbor] == rank[node] - 1:  # 忽略回到父節點的邊
                continue

            # 遞歸訪問鄰居節點
            dfs_depth = self.dfs(graph, neighbor, depth + 1, rank, result, n)
            
            # 如果鄰居的深度小於等於當前節點，則不是關鍵連接
            if dfs_depth <= rank[node]:
                result.discard(tuple(sorted((neighbor, node))))
            
            # 更新當前節點的最小深度
            min_depth = min(min_depth, dfs_depth)
        
        return min_depth  # 返回最小深度

```
pass

---

#### **執行過程舉例**

輸入：

python

複製程式碼

`n = 4 connections = [[0,1],[1,2],[2,0],[1,3]]`

1. **構建鄰接表**：
    
    python
    
    複製程式碼
    
    `graph = [     [1, 2],  # 節點 0 的鄰居     [0, 2, 3],  # 節點 1 的鄰居     [1, 0],  # 節點 2 的鄰居     [1]  # 節點 3 的鄰居 ]`
    
2. **初始化**：
    
    - `rank = [-inf, -inf, -inf, -inf]`。
    - `result = {(0, 1), (1, 2), (2, 0), (1, 3)}`。
3. **DFS 遍歷**：
    
    - 從節點 0 開始：
        - 設置 `rank[0] = 0`。
        - 遍歷鄰居 `1`：
            - 遞歸進入節點 1，設置 `rank[1] = 1`。
            - 遍歷鄰居 `2`：
                - 遞歸進入節點 2，設置 `rank[2] = 2`。
                - 遍歷鄰居 `0`，返回深度 `0`。
                - 更新 `min_depth = 0`。
            - 返回到節點 1，發現 `(1, 2)` 不是關鍵連接，從結果中移除。
            - 遍歷鄰居 `3`：
                - 遞歸進入節點 3，設置 `rank[3] = 2`。
                - 返回到節點 1，發現 `(1, 3)` 是關鍵連接。
        - 返回到節點 0，發現 `(0, 1)` 不是關鍵連接，從結果中移除。
4. **最終結果**：
    
    python
    
    複製程式碼
    
    `result = [(1, 3)]`
    

---

### **複雜度分析**

1. **時間複雜度**：
    
    - 建立鄰接表需要 O(E)O(E)O(E)，其中 EEE 是邊的數量。
    - 每個節點和每條邊在 DFS 中被訪問一次，因此 DFS 的時間複雜度為 O(V+E)O(V + E)O(V+E)。
    - 總時間複雜度為 **O(V+E)O(V + E)O(V+E)**。
2. **空間複雜度**：
    
    - 鄰接表需要 O(V+E)O(V + E)O(V+E) 的空間。
    - 遞歸深度最多為 O(V)O(V)O(V)（最壞情況下的遞歸棧）。
    - 總空間複雜度為 **O(V+E)O(V + E)O(V+E)**。

---

### **其他解法簡述**

#### 1. **暴力法**

- 刪除每條邊，使用 BFS 或 DFS 檢查圖的連通性。
- 時間複雜度為 O(E⋅(V+E))O(E \cdot (V + E))O(E⋅(V+E))，不適合大圖。

#### 2. **分支限界法**

- 結合啟發式方法和 DFS，提前剪枝。

#### 3. **動態規劃結合圖論**

- 記錄每個節點的「最小到達深度」，類似於 Tarjan 算法，但更依賴動態狀態的緩存。

如果需要進一步解釋或其他方法實現，請隨時告訴我！

4o