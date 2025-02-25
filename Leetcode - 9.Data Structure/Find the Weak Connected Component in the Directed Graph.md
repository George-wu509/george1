Lintcode 432
请找出有向图中弱连通分量。图中的每个节点包含 1 个标签和1 个相邻节点列表。（有向图的弱连通分量是任意两点均有有向边相连的极大子图）

**样例 1:**
```python
"""
输入: {1,2,4#2,4#3,5#4#5#6,5}
输出: [[1,2,4],[3,5,6]]
解释: 
  1----->2    3-->5
   \     |        ^
    \    |        |
     \   |        6
      \  v
       ->4
```
### **解釋輸入格式 `{1,2,4#2,4#3,5#4#5#6,5}`**

這個輸入格式是一種**圖的鄰接表（Adjacency List）表示法**，其中：

- **每個 `#` 分隔不同的節點與其鄰居**
- **每個逗號 `,` 代表這個節點的鄰接點**

#### **如何解析這個輸入**

- `{1,2,4#2,4#3,5#4#5#6,5}`
    - **`1,2,4`**：節點 `1` 連接到 `2` 和 `4`
    - **`2,4`**：節點 `2` 連接到 `4`
    - **`3,5`**：節點 `3` 連接到 `5`
    - **`4`**：節點 `4` 沒有額外的鄰居
    - **`5`**：節點 `5` 沒有額外的鄰居
    - **`6,5`**：節點 `6` 連接到 `5`


**样例 2:**
```python
"""
输入: {1,2#2,3#3,1}
输出: [[1,2,3]]
```


```python
class Solution:
    def __init__(self):
        self.f = {}

    def merge(self, x, y):
        x = self.find(x)
        y = self.find(y)
        if x != y:
            self.f[x] = y

    def find(self, x):
        if self.f[x] == x:
            return x
        
        self.f[x] = self.find(self.f[x])
        return self.f[x]

    # @param {DirectedGraphNode[]} nodes a array of directed graph node
    # @return {int[][]} a connected set of a directed graph
    def connectedSet2(self, nodes):
        for node in nodes:
            self.f[node.label] = node.label

        for node in nodes:
            for nei in node.neighbors:
                self.merge(node.label, nei.label)

        result = []
        g = {}
        cnt = 0

        for node in nodes:
            x = self.find(node.label)
            if not x in g:
                cnt += 1
                g[x] = cnt
            
            if len(result) < cnt:
                result.append([])
        
            result[g[x] - 1].append(node.label)

        return result
```
pass


## **LintCode 432: Find the Weak Connected Component in the Directed Graph**

這題的目標是找到**弱連通分量（Weak Connected Component, WCC）**，也就是在無向視角下能夠連通的子圖。

---

## **解法分析**

本題的核心思想是使用 **并查集（Union-Find）** 來高效地合併節點，並找出所有的連通分量。我們的並查集會針對有向圖，但它只考慮連接關係（無視方向），將其當作無向圖來處理。

### **為何要這樣解？**

1. **使用并查集來合併節點**
    
    - 這樣可以高效處理「哪些節點屬於同一個連通分量」的問題。
    - 并查集的**合併（Union）**與**查找（Find）**操作在帶有路徑壓縮的情況下幾乎是 O(1) 時間複雜度。
2. **忽略方向視作無向圖**
    
    - 本題要找的是**弱連通分量**，即便 `A → B`，只要 A 能到 B，那麼 B 也應該能到 A（無向視角）。
    - 因此，我們在遍歷有向圖時，將邊視為**無向邊來合併**。
3. **用哈希表（字典）來追蹤連通分量**
    
    - 通過 **`find(x)`** 找到每個節點的代表元素，然後將相同代表元素的節點收集在一起。
    - 這樣可以方便地輸出每個連通分量的結果。

---

## **解法步驟**

### **Step 1: 初始化并查集**

我們需要一個 **字典 `f` 來記錄父節點**，初始時，每個節點的父節點指向自己。

`for node in nodes:     self.f[node.label] = node.label`

- **示例**：

    `節點: [1, 2, 3, 4, 5] 初始 self.f: { 1: 1, 2: 2, 3: 3, 4: 4, 5: 5 }`
    

### **Step 2: 合併所有鄰居**

遍歷所有節點，對於 `node.label → neighbor.label`，視為無向邊，進行合併：

`for node in nodes:     for nei in node.neighbors:         self.merge(node.label, nei.label)`

- **示例**：

    `給定有向圖:    1 → 2    2 → 3    4 → 5     轉換為無向連接:    1 - 2 - 3  (變成一個連通分量)    4 - 5      (變成另一個連通分量)     合併後的 self.f: { 1: 2, 2: 3, 3: 3, 4: 5, 5: 5 }`
    

### **Step 3: 收集連通分量**

我們建立一個字典 `g` 來存儲「根節點 → 該集合的索引」，並用 `result` 存儲最終的 WCC：

`for node in nodes:     x = self.find(node.label)  # 找到該節點的根     if x not in g:         cnt += 1         g[x] = cnt  # 紀錄該根節點對應的索引          if len(result) < cnt:         result.append([])      result[g[x] - 1].append(node.label)`

- **示例**：

    `self.f 經過 find() 壓縮後: { 1: 3, 2: 3, 3: 3, 4: 5, 5: 5 }  g 映射: { 3: 1, 5: 2 }  # 3 對應 WCC-1，5 對應 WCC-2  result 最終輸出: [[1, 2, 3], [4, 5]]`
    

### **Step 4: 輸出結果**

最後，我們返回 `result`，這是一個**排序好的弱連通分量列表**。

---

## **時間複雜度分析**

假設圖中有 `n` 個節點，`m` 條邊：

1. **初始化 `self.f`** 需要 `O(n)`。
2. **合併操作（`merge`）** 遍歷所有 `m` 條邊，每次 `find()` 使用路徑壓縮，均攤時間複雜度約為 `O(α(n))`（α 為阿克曼函數的反函數，幾乎是常數）。
    - 合併所有邊總計 `O(m α(n))` ≈ `O(m)`。
3. **收集連通分量** 需要 `O(n)`。
4. **總體複雜度為 `O(n + m)`**。

這是一個**非常高效**的算法，適合用於大規模圖的連通性問題。

---

## **其他解法思路（不需要代碼）**

### **1. 深度優先搜索（DFS）**

- 遍歷每個節點，使用 DFS 找出所有能到達的節點，把它們歸類到同一個連通分量中。
- 時間複雜度：`O(n + m)`。
- 缺點：遞歸版本可能導致深度過大，需改為非遞歸 DFS。

### **2. 廣度優先搜索（BFS）**

- 使用隊列，類似 DFS，但逐層擴展，同樣能夠找到所有連通的節點。
- 時間複雜度：`O(n + m)`。
- 適用場景：較適合處理大規模數據。

### **3. Floyd-Warshall（適用於密集圖）**

- 使用動態規劃的方法，透過矩陣計算所有節點對之間的可達性。
- 時間複雜度：`O(n^3)`。
- 缺點：適合小型圖，大型圖無法處理。

---

## **總結**

- **最佳解法**：**并查集**，時間複雜度 `O(n + m)`，適合大規模圖。
- **可選解法**：
    - **DFS/BFS**：適用於較小規模的圖。
    - **Floyd-Warshall**：適用於密集圖但效率較低。

並查集方法充分利用了「**動態合併集合**」的特性，使得查詢和合併操作**幾乎是 O(1)**，這也是為什麼我們選擇這種方法來解這道題的原因。