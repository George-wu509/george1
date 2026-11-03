Lintcode 178
给出 `n` 个节点，标号分别从 `0` 到 `n - 1` 并且给出一个 `无向` 边的列表 (给出每条边的两个顶点), 写一个函数去判断这张｀无向｀图是否是一棵树


**样例 1:**
```python
"""
输入: n = 5 edges = [[0, 1], [0, 2], [0, 3], [1, 4]]
输出: true.
```
**样例 2:**
```python
"""
输入: n = 5 edges = [[0, 1], [1, 2], [2, 3], [1, 3], [1, 4]]
输出: false.
```



```python
class Solution:
    """
    @param n: An integer
    @param edges: a list of undirected edges
    @return: true if it's a valid tree, or false
    """
    def validTree(self, n, edges):
        if n - 1 != len(edges):
            return False
        
        self.father = {i: i for i in range(n)}
        self.size = n
        
        for a, b in edges:
            self.union(a, b)
            
        return self.size == 1
        
    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.size -= 1
            self.father[root_a] = root_b
        
    def find(self, node):
        path = []
        while node != self.father[node]:
            path.append(node)
            node = self.father[node]
            
        for n in path:
            self.father[n] = node
            
        return node
```
pass



##### LeetCode: Graph Valid Tree 代碼詳解

在無向圖中，**$n$ 個節點要構成一棵「合法的樹」，必須同時滿足兩個條件**
1. **邊數條件**：無向邊的數量必須恰好等於 **$n - 1$**。
2. **連通性條件**：所有節點必須全部連通（即整張圖只有 **1 個連通分量**）。
###### 代碼逐段剖析與邏輯分析

```python
class Solution:
    def validTree(self, n: int, edges: list[list[int]]) -> bool:
        # 1. 邊數定理過濾：
        # N 個節點的樹必有 N-1 條邊。
        # 若邊數 < n - 1，則圖必然不連通；若邊數 > n - 1，則圖中必然存在環（Cycle）。
        if n - 1 != len(edges):
            return False
        
        # 2. 初始化父節點映射與連通分量計數
        self.father = {i: i for i in range(n)}  # 用 dict 記錄 father（效果等同於 list）
        self.size = n                           # 初始時有 n 個獨立節點（n 個連通分量）
        
        # 3. 遍歷每條無向邊進行合併
        for a, b in edges:
            self.union(a, b)
            
        # 4. 若所有邊處理完成後連通分量縮減為 1，代表圖完全連通且無環，即為合法樹
        return self.size == 1
        
    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        
        # 若根節點不同，說明兩節點原先處於兩個獨立的連通塊
        if root_a != root_b:
            self.size -= 1                # 連通塊數量減 1
            self.father[root_a] = root_b  # 將 root_a 掛載到 root_b 下
        
    def find(self, node):
        path = []
        # 顯式迴圈尋找根節點，同時收集沿途節點
        while node != self.father[node]:
            path.append(node)
            node = self.father[node]
            
        # 顯式路徑壓縮：將沿途所有節點直接指向根節點 node
        for n in path:
            self.father[n] = node
            
        return node
```