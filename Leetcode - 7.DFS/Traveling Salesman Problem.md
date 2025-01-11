
### **LintCode 816：Traveling Salesman Problem (TSP)**

#### **題目描述**

給定 `n` 個城市和一些連接它們的道路，每條道路以三元組 `(a, b, c)` 表示，意味著城市 `a` 和城市 `b` 之間有一條成本為 `c` 的道路。尋找一條從某個城市開始並訪問所有城市一次且僅一次的最小成本路徑。

---

### **DFS 解法**

#### **算法思路**

1. **構建圖結構**：
    
    - 將道路信息 `roads` 轉換為鄰接表表示的圖結構，方便訪問每個城市的相鄰城市及其對應的成本。
2. **深度優先搜索（DFS）**：
    
    - 從某個城市開始遞歸遍歷，記錄當前的訪問城市數量和路徑成本。
    - 使用 `visited` 集合跟蹤已訪問的城市，避免重複訪問。
    - 當訪問的城市數量等於總城市數時，更新最小成本。
3. **剪枝**：
    
    - 如果當前路徑的成本已經超過已知的最小成本，則提前返回。
4. **結果輸出**：
    
    - 遞歸結束後，返回最小成本。

---
Example:
```plain
输入: 
n = 3
tuple = [[1,2,1],[2,3,2],[1,3,3]]
输出: 3
说明：最短路是1->2->3
```
**样例2**
```plain
输入:
n = 1
tuple = []
输出: 0
```


#### **代碼詳解**
```python
class Result:
    def __init__(self):
        self.min_cost = float('inf')  # 初始化最小成本為無窮大

class Solution:
    def min_cost(self, n, roads):
        # 構建圖
        graph = self.construct_graph(roads, n)
        result = Result()
        
        # 開始 DFS 從城市 1 出發
        self.dfs(1, n, set([1]), 0, graph, result)
        return result.min_cost

    def dfs(self, city, n, visited, cost, graph, result):
        # 如果已訪問所有城市，更新最小成本
        if len(visited) == n:
            result.min_cost = min(result.min_cost, cost)
            return

        # 遍歷所有鄰居城市
        for next_city in graph[city]:
            if next_city in visited:
                continue
            visited.add(next_city)  # 標記城市為已訪問
            self.dfs(next_city, n, visited, cost + graph[city][next_city], graph, result)
            visited.remove(next_city)  # 回溯，撤銷訪問

    def construct_graph(self, roads, n):
        # 使用鄰接表構建圖
        graph = {i: {} for i in range(1, n + 1)}
        for a, b, c in roads:
            if b not in graph[a]:
                graph[a][b] = c
            else:
                graph[a][b] = min(graph[a][b], c)
            if a not in graph[b]:
                graph[b][a] = c
            else:
                graph[b][a] = min(graph[b][a], c)
        return graph

```
pass

---

### **執行過程舉例**

輸入：

`n = 4 roads = [     (1, 2, 10),     (2, 3, 20),     (3, 4, 30),     (4, 1, 40),     (1, 3, 50) ]`

**步驟 1：構建圖**

python

複製程式碼

`graph = {     1: {2: 10, 3: 50, 4: 40},     2: {1: 10, 3: 20},     3: {2: 20, 4: 30, 1: 50},     4: {3: 30, 1: 40} }`

**步驟 2：開始 DFS**

- 初始狀態：`city = 1, visited = {1}, cost = 0`。
- 遍歷鄰居城市：
    1. 選擇 `2`：進入 `dfs(2, 4, {1, 2}, 10)`。
    2. 選擇 `3`：進入 `dfs(3, 4, {1, 3}, 50)`。
    3. 選擇 `4`：進入 `dfs(4, 4, {1, 4}, 40)`。

**步驟 3：遞歸和回溯**

- 每次遞歸處理當前城市的所有鄰居，直至訪問所有城市。
- 當訪問所有城市時嘗試返回起點，並更新最小成本。

**結果計算**：

- 最小路徑為 `1 → 2 → 3 → 4 → 1`，總成本為 `10 + 20 + 30 + 40 = 100`。

---

### **複雜度分析**

1. **時間複雜度**：
    
    - 理論上，每條完整路徑都會被枚舉，總共有 n!n!n! 條路徑。
    - 每次遞歸需要 O(n)O(n)O(n) 的時間處理鄰居。
    - 總時間複雜度為 **O(n×n!)O(n \times n!)O(n×n!)**。
2. **空間複雜度**：
    
    - 遞歸深度最多為 nnn，需要 O(n)O(n)O(n) 的空間。
    - 額外的 `visited` 集合也需要 O(n)O(n)O(n) 空間。
    - 總空間複雜度為 **O(n)O(n)O(n)**。

---

### **其他解法簡述**

#### 1. **動態規劃 + 位運算 (DP + Bitmask)**

- 狀態表示：`dp[mask][i]` 表示當前訪問過的城市集合為 `mask`，最後停留在城市 iii 的最小成本。
- 狀態轉移： dp[mask][i]=min⁡(dp[mask][i],dp[mask∖{i}][j]+graph[j][i])dp[mask][i] = \min(dp[mask][i], dp[mask \setminus \{i\}][j] + graph[j][i])dp[mask][i]=min(dp[mask][i],dp[mask∖{i}][j]+graph[j][i])
- 時間複雜度：**O(n2⋅2n)O(n^2 \cdot 2^n)O(n2⋅2n)**。
- 適合中等規模問題。

#### 2. **貪心法**

- 從起點出發，每次選擇最短的邊進行遍歷。
- 無法保證全局最優，但適合快速近似解。

#### 3. **分支限界法 (Branch and Bound)**

- 在 DFS 基礎上引入下界估計，用於剪枝。

#### 4. **遺傳算法**

- 使用啟發式算法求解近似結果，適合大規模問題。

---

### **總結**

- **DFS** 適合理解問題及解決小規模圖問題。
- **DP + Bitmask** 是解決 TSP 的經典方法，效率高於 DFS。
- **分支限界法和啟發式算法** 適用於大規模圖的近似解問題。

如果需要更詳細的解釋或其他算法實現，請隨時告訴我！

4o

  

O

ChatGPT 可能會發生錯