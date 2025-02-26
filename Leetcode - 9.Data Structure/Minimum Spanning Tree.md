Lintcode 629
给定一个Connections，即Connection类（边缘两端的城市名称和它们之间的开销），找到可以连接所有城市并且花费最少的边缘。  
如果可以连接所有城市，则返回连接方法。  
如果城市无法全部连通，则返回空列表。


**样例 1:**
```python
"""
输入:
["Acity","Bcity",1]
["Acity","Ccity",2]
["Bcity","Ccity",3]
输出:
["Acity","Bcity",1]
["Acity","Ccity",2]
```
**样例 2:**
```python
"""
输入:
["Acity","Bcity",2]
["Bcity","Dcity",5]
["Acity","Dcity",4]
["Ccity","Ecity",1]
输出:
[]

解释:
没有办法连通
```


```python
import functools

def comp(a, b):
    if a.cost != b.cost:
        return a.cost - b.cost
    
    if a.city1 != b.city1:
        if a.city1 < b.city1:
            return -1
        else:
            return 1

    if a.city2 == b.city2:
        return 0
    elif a.city2 < b.city2:
        return -1
    else:
        return 1

class Solution:
    # @param {Connection[]} connections given a list of connections include two cities and cost
    # @return {Connection[]} a list of connections from results
    def lowestCost(self, connections):
        # Write your code here
        connections.sort(key=functools.cmp_to_key(comp))
        hash = {}   
        n = 0
        for connection in connections:
            if connection.city1 not in hash:
                n += 1
                hash[connection.city1] = n
            
            if connection.city2 not in hash:
                n += 1
                hash[connection.city2] = n

        father = [0 for _ in range(n + 1)] 

        results = []
        for connection in connections:
            num1 = hash[connection.city1]
            num2 = hash[connection.city2]

            root1 = self.find(num1, father)
            root2 = self.find(num2, father)
            if root1 != root2:
                father[root1] = root2
                results.append(connection)

        if len(results)!= n - 1:
            return []
        return results
    
    def find(self, num, father):
        if father[num] == 0:
            return num
        father[num] = self.find(father[num], father)
        return father[num]
```
pass


## **LintCode 629: Minimum Spanning Tree（最小生成樹）**

本題的目標是找到**最小生成樹（Minimum Spanning Tree, MST）**，即在一個連通圖中選擇一組邊，使得所有節點保持連通，並且邊的總權重最小。

---

## **解法分析**

本題使用的是 **Kruskal 演算法** + **並查集（Union-Find）** 來構造最小生成樹（MST）。這樣做的原因是：

1. **Kruskal 演算法的特性**
    
    - Kruskal 是基於 **貪心法（Greedy）** 的演算法，通過逐步選擇**權重最小的邊**來構造最小生成樹。
    - 每次選邊時確保不產生環（使用並查集來檢查連通性）。
2. **並查集的應用**
    
    - 當我們選擇一條邊 `(A, B)` 時，使用並查集判斷 `A` 和 `B` 是否已經在同一集合中。
    - 如果 `A` 和 `B` **不在同一集合**，則合併它們，並加入這條邊到最小生成樹的邊集中。
    - 如果 `A` 和 `B` **已在同一集合**，則跳過這條邊，避免形成環。
3. **為何 Kruskal + Union-Find 是最佳解法？**
    
    - Kruskal 的時間複雜度是 `O(E log E)`，其中 `E` 是邊的數量。
    - **並查集**（使用路徑壓縮和按秩合併）可以使 `Find` 和 `Union` 操作的均攤時間接近 `O(1)`。
    - 因此，總體時間複雜度為 `O(E log E)`，比 Prim 演算法 `O(V^2)` 在稀疏圖（E ≪ V^2）時更優。

---

## **解法步驟**

### **Step 1: 解析輸入**

題目提供的 `connections` 是一個**無向加權圖**，每條邊表示兩個城市之間的連接及其成本：

```python
connections = [
    Connection("A", "B", 1),
    Connection("B", "C", 2),
    Connection("A", "C", 3)
]
```

這裡的 `Connection` 類包含：

- `city1`: 邊的起點城市
- `city2`: 邊的終點城市
- `cost`: 這條邊的權重

---

### **Step 2: 對邊按成本排序**

根據 Kruskal 演算法，我們需要**按照 `cost` 升序排序**邊，以便從最小的權重開始選擇。

`connections.sort(key=functools.cmp_to_key(comp))`

這裡 `comp(a, b)` 是自定義的比較函數：

1. **先比較 `cost`，優先選擇權重較小的邊**。
2. **如果 `cost` 相同，按 `city1` 字母序排序**（確保結果的字典序最小）。
3. **如果 `city1` 也相同，按 `city2` 字母序排序**。

舉例：
```python
原始邊: [("A", "C", 3), ("A", "B", 1), ("B", "C", 2)]
排序後: [("A", "B", 1), ("B", "C", 2), ("A", "C", 3)]
```

---

### **Step 3: 建立城市對應的數字編號**

我們需要對城市進行**數字映射**，這樣並查集可以用數組表示：
```python
hash = {}   
n = 0
for connection in connections:
    if connection.city1 not in hash:
        n += 1
        hash[connection.city1] = n

    if connection.city2 not in hash:
        n += 1
        hash[connection.city2] = n
```

- `hash["A"] = 1`
- `hash["B"] = 2`
- `hash["C"] = 3` 這樣 `A, B, C` 就被映射為 `1, 2, 3`，方便用陣列 `father[]` 來管理 Union-Find。

---

### **Step 4: 初始化並查集**

`father = [0 for _ in range(n + 1)]`

這裡 `father[i]` 表示編號 `i` 所屬的集合代表元素（初始時 `i` 指向自己）。

---

### **Step 5: 遍歷邊並執行 Kruskal 演算法**

```python
results = []
for connection in connections:
    num1 = hash[connection.city1]
    num2 = hash[connection.city2]

    root1 = self.find(num1, father)
    root2 = self.find(num2, father)

    if root1 != root2:  # 不在同一個集合，則選擇這條邊
        father[root1] = root2
        results.append(connection)

```

這裡：

1. 透過 `find()` 找到 `num1` 和 `num2` 所屬的集合。
2. **如果 `root1 ≠ root2`，代表它們是不同的集合**：
    - 選擇這條邊，加入結果 `results`。
    - 使用 **Union** 操作，將兩個集合合併。

---

### **Step 6: 檢查最小生成樹是否構成**

```python
if len(results) != n - 1:
    return []
```
- **MST 必須有 `n-1` 條邊**，否則無法構成連通圖。
- 如果邊數少於 `n-1`，表示有節點未連接，返回 `[]`。

---

## **時間複雜度分析**

假設：

- `V` 為城市數（節點數）
- `E` 為道路數（邊數）

|步驟|複雜度|
|---|---|
|**Step 2: 排序邊**|`O(E log E)`|
|**Step 3: 建立城市哈希表**|`O(V)`|
|**Step 4: 初始化並查集**|`O(V)`|
|**Step 5: 遍歷邊並執行 Union-Find**|`O(E α(V))`|
|**Step 6: 檢查結果**|`O(1)`|
|**總計**|`O(E log E)`|

其中 `α(V)` 是阿克曼函數的反函數，幾乎是常數，所以 `O(E α(V))` 近似 `O(E)`。

---

## **其他解法（不需要代碼）**

### **1. Prim 演算法**

- **思路**：從一個節點開始，不斷選擇與已選節點相鄰的最小權重邊，直到涵蓋所有節點。
- **時間複雜度**：
    - 使用**暴力選擇最小邊**：`O(V^2)`
    - 使用**優先隊列（Heap）**：`O(E log V)`
- **適用場景**：當**邊密集（E ≈ V^2）**時，Prim 可能比 Kruskal 更快。

### **2. Borůvka 演算法**

- **思路**：每個連通分量各自選擇一條最小邊，合併，重複直到所有節點連接。
- **時間複雜度**：`O(E log V)`
- **適用場景**：適用於大規模並行運算。

---

## **總結**

- **最佳解法**：使用 **Kruskal + 并查集**，時間複雜度 `O(E log E)`，適用於**稀疏圖**。
- **其他解法**：
    - **Prim（O(E log V)）**：適用於**稠密圖**。
    - **Borůvka（O(E log V)）**：適用於**大規模並行運算**。

Kruskal + 并查集是一種**簡單、高效且易於實現**的方法，因此是解這類最小生成樹問題的最佳選擇！

  



## **詳細解析 Kruskal 算法求解最小生成樹（MST）**

Kruskal 算法是一種**貪心算法（Greedy Algorithm）**，用來計算**無向加權圖**的**最小生成樹（MST, Minimum Spanning Tree）**。它的基本思路是：

1. **將所有邊按照權重排序**（從小到大）。
2. **逐步加入邊，確保不形成環**（用**並查集**來檢查是否連通）。
3. **直到 MST 包含 `n-1` 條邊**（其中 `n` 是節點數）。

---

## **問題描述**

給定一個無向圖，其中的邊格式為 `(u, v, w)`，表示從節點 `u` 到節點 `v` 的邊，權重為 `w`：
```python
"""
edges = [(0, 1, 1),  # 節點 0 和 1 之間的邊，權重 1
         (1, 2, 2),  # 節點 1 和 2 之間的邊，權重 2
         (0, 2, 3),  # 節點 0 和 2 之間的邊，權重 3
         (1, 3, 4)]  # 節點 1 和 3 之間的邊，權重 4
```

![[Pasted image 20250225100709.png]]

我們要構造一棵**最小生成樹（MST）**，使得：

- **所有節點連通**
- **權重總和最小**
- **不能形成環**

解釋:
step1:
parent = [0, 1, 2, 3] rank = [1, 1, 1, 1]

step2: 加入(0,1,1)
parent = [1, 1, 2, 3], weight=1

step3: 加入(1,2,2)
parent = [1, 2, 2, 3], weight=3

step4: 加入(0,2,3)
**跳過這條邊（避免形成環）**。

---

## **Kruskal 算法的解法步驟**

### **Step 1: 將所有邊按權重排序**

`edges.sort(key=lambda x: x[2])  # 按照邊權重升序排序`

排序後的 `edges`：

`[(0, 1, 1), (1, 2, 2), (0, 2, 3), (1, 3, 4)]`

這樣我們可以確保**優先選擇最小權重的邊**來構造 MST。

---

### **Step 2: 初始化並查集**

我們使用**並查集（Union-Find）**來檢查邊是否會形成環：

`uf = UnionFind(4)  # 4 個節點 (0, 1, 2, 3)`

初始化的 `UnionFind` 結構：
```python
"""
parent = [0, 1, 2, 3]  # 每個節點的父節點先指向自己
rank = [1, 1, 1, 1]  # 初始化秩為 1
```

---

### **Step 3: 遍歷邊並執行 Kruskal 算法**

遍歷排序後的邊，每次嘗試將邊 `(u, v, w)` 加入最小生成樹：
```python
"""
mst_weight = 0
mst_edges = []

for u, v, w in edges:
    if not uf.connected(u, v):  # 如果 u 和 v 不在同一個集合
        uf.union(u, v)  # 合併兩個節點
        mst_weight += w  # 累加 MST 的總權重
        mst_edges.append((u, v, w))  # 記錄選中的邊

```

---

## **具體步驟拆解**

### **初始化**
```python
"""
parent = [0, 1, 2, 3]
rank = [1, 1, 1, 1]

```

### **遍歷排序後的邊**

#### **1. 選擇 (0,1,1)**

- `find(0) = 0`，`find(1) = 1`（不同集合）。
- **Union(0,1)**：將 `0` 合併到 `1`。
- **更新 `parent`**：

    `parent = [1, 1, 2, 3]`
    
- **MST 邊集合**：

    `[(0, 1, 1)]`
    
- **當前權重和**：

    `mst_weight = 1`
    

---

#### **2. 選擇 (1,2,2)**

- `find(1) = 1`，`find(2) = 2`（不同集合）。
- **Union(1,2)**：將 `1` 合併到 `2`。
- **更新 `parent`**：

    `parent = [1, 2, 2, 3]`
    
- **MST 邊集合**：

    `[(0, 1, 1), (1, 2, 2)]`
    
- **當前權重和**：

    `mst_weight = 3`
    

---

#### **3. 選擇 (0,2,3)**

- `find(0) = 2`，`find(2) = 2`（同一集合）。
- **跳過這條邊（避免形成環）**。

---

#### **4. 選擇 (1,3,4)**

- `find(1) = 2`，`find(3) = 3`（不同集合）。
- **Union(1,3)**：將 `2` 合併到 `3`。
- **更新 `parent`**：

    `parent = [1, 2, 3, 3]`
    
- **MST 邊集合**：

    `[(0, 1, 1), (1, 2, 2), (1, 3, 4)]`
    
- **當前權重和**：

    `mst_weight = 7`
    

---

### **Step 4: 返回最小生成樹結果**

`print("MST weight:", mst_weight)  # 輸出 7 print("MST edges:", mst_edges)  # 輸出 [(0,1,1), (1,2,2), (1,3,4)]`

最終的 **MST**：

  `1  / \ 0   2     |     3`

這是一棵 **連通樹**，包含所有節點，且權重最小。

---

## **時間複雜度分析**

假設：

- `V` 是節點數
- `E` 是邊數

|步驟|複雜度|
|---|---|
|**邊排序** `edges.sort()`|`O(E log E)`|
|**並查集初始化**|`O(V)`|
|**遍歷邊並執行 Union-Find**|`O(E α(V))`（幾乎是 `O(E)`)|
|**總計**|`O(E log E)`|

**為何 `O(E log E)` 是最佳解法？**

- **排序是 Kruskal 的主要成本**，使用快排或合併排序，時間複雜度 `O(E log E)`。
- **Union-Find 的 `Find` 和 `Union` 操作幾乎是 `O(1)`（阿克曼函數的反函數 `α(V) ≈ O(1)`)。
- **對於稀疏圖（E ≪ V^2），比 Prim 更優**。

---

## **其他解法（不需代碼）**

### **1. Prim 算法**

- **適用於密集圖（E ≈ V^2）**
- **時間複雜度 `O(E log V)`**
- **使用優先隊列（Heap）每次選最小邊**
- **比 Kruskal 更適合邊數很多的情況**

### **2. Borůvka 算法**

- **適用於並行計算**
- **每輪找所有連通分量的最小邊，合併直到形成一棵樹**
- **時間複雜度 `O(E log V)`**
- **適用於超大規模分布式計算**

---

## **總結**

- **Kruskal + Union-Find 是求 MST 最優解法之一，時間複雜度 `O(E log E)`，適用於稀疏圖（E ≪ V^2）。**
- **Prim 更適合稠密圖（E ≈ V^2）。**
- **Borůvka 適用於並行計算。**
- **Kruskal 方法簡單易懂，適用範圍廣泛。**