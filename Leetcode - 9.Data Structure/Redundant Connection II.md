Lintcode 1087
在这个问题中，有根树是一种这样的有向图，它只有一个根节点，所有其他节点都是该节点的后代，加上每个节点只有一个父母节点，（除了根节点没有父母）。

给定的输入是一个有向图，它以具有N个节点（具有不同的值1,2，...，N）的有根树开始，并添加了一个额外的有向边。 添加的边有两个不同的顶点，从1到N中选择，并且不是已经存在的边。

得到的图形以边的2D数组给出。 边的每个元素是一对[u，v]，表示连接节点u和v的有向边，其中u是子v的父节点。

返回可以删除的边，以便生成的图是N个节点的有根树。 如果有多个答案，则返回给定2D数组中最后出现的答案。


样例 1:
```python
"""
输入: [[1,2], [1,3], [2,3]]
输出: [2,3]
解释: 给定的有向图将是这样的:
  1
 / \
v   v
2-->3
```
样例 2:
```python
"""
输入: [[1,2], [2,3], [3,4], [4,1], [1,5]]
输出: [4,1]
解释: 给定的有向图将是这样的:
5 <- 1 -> 2
     ^    |
     |    v
     4 <- 3
```


```python
from typing import (
    List,
)
class UnionFind:
    def __init__(self, n):
        self.ancestor = list(range(n))
    
    def union(self, index1: int, index2: int):
        self.ancestor[self.find(index1)] = self.find(index2)
    
    def find(self, index: int) -> int:
        if self.ancestor[index] != index:
            self.ancestor[index] = self.find(self.ancestor[index])
        return self.ancestor[index]

class Solution:
    """
    @param edges: List[List[int]]
    @return: return List[int]
    """
    def find_redundant_directed_connection(self, edges: List[List[int]]) -> List[int]:
        # write your code here
        n = len(edges)
        uf = UnionFind(n + 1)
        parent = list(range(n + 1))
        conflict = -1
        cycle = -1
        for i, (node1, node2) in enumerate(edges):
            if parent[node2] != node2:
                conflict = i
            else:
                parent[node2] = node1
                if uf.find(node1) == uf.find(node2):
                    cycle = i
                else:
                    uf.union(node1, node2)

        if conflict < 0:
            return [edges[cycle][0], edges[cycle][1]]
        else:
            conflictEdge = edges[conflict]
            if cycle >= 0:
                return [parent[conflictEdge[1]], conflictEdge[1]]
            else:
                return [conflictEdge[0], conflictEdge[1]]
```
pass


# **LintCode 1087: Redundant Connection II 解法詳細解析**

## **問題描述**

給定一個 **有向圖（directed graph）**，其中的 `edges[i] = [u, v]` 代表從節點 `u` 到 `v` 的一條邊。

這個圖有 **`n` 個節點和 `n` 條邊**（這意味著必然存在一條冗餘邊，因為在 `n` 個節點的樹中應該只有 `n-1` 條邊）。

我們需要**刪除**恰好 **一條邊**，使得圖變成**有向樹（directed tree）**：

- **有向樹** 的特性：
    1. **沒有環（Cycle）**
    2. **每個節點最多有一個父節點（No Node Has Two Parents）**
    3. **所有節點都連接到一個根節點（Connected）**

我們的目標是找到並刪除**最後一條加入的冗餘邊**。

---

## **解法分析**

這道題的難點在於：

- 圖可能形成 **環（Cycle）**
- 圖可能出現 **一個節點有兩個父節點**
- 這兩種情況可能**同時發生**

為了解決這個問題，我們**分三步處理**：

### **Step 1: 先檢查是否有節點擁有** **兩個父節點**

1. 我們用 `parent` 陣列記錄每個節點的父節點：
    
    - **如果某個節點已經有父節點，則標記這條邊為 `conflict`（衝突邊）**
    - 如果沒有，則繼續正常記錄。
2. 如果發現一個節點有 **兩個父節點**，這表示：
    
    - 圖中存在 **一條邊必須被刪除**。

---

### **Step 2: 檢查是否存在環（Cycle）**

我們使用 **並查集（Union-Find）** 來檢查**是否形成環**：

- **如果 `find(node1) == find(node2)`，表示 `node1 → node2` 形成了環，記錄 `cycle = i`**
- 否則，將 `node1` 和 `node2` 進行合併。

---

### **Step 3: 根據 `conflict` 和 `cycle` 來決定要刪除哪條邊**

- **Case 1: 只有環（沒有 `conflict`）**
    
    - 直接刪除 `cycle` 這條邊即可。
- **Case 2: 只有 `conflict`（沒有環）**
    
    - 直接刪除 `conflict` 這條邊即可。
- **Case 3: 同時有 `conflict` 和 `cycle`**
    
    - 此時 `conflict` 的其中一條邊**一定在環內**。
    - 刪除 `conflict` 中**較早加入的那條邊**（`parent[conflictEdge[1]]`）。

---

## **解法步驟詳細舉例**

### **Example 1**

python

複製編輯

`edges = [[1,2], [1,3], [2,3]]`

#### **Step 1: 建立 `parent` 陣列**

- `parent[2] = 1`
- `parent[3] = 1`
- **發現 `parent[3]` 已經存在，發生 `conflict = 2`**
    - `conflictEdge = [2,3]`，`3` 出現了兩次。

#### **Step 2: 使用並查集檢查環**

- 加入 `edges[0] = [1,2]`，不形成環。
- 加入 `edges[1] = [1,3]`，不形成環。
- **跳過 `edges[2] = [2,3]`，因為它是 `conflictEdge`。**

圖現在是：

複製編輯

  `1 → 2   1 → 3`

是**有效的有向樹**，所以刪除 `conflictEdge = [2,3]`。

#### **結果**

python

複製編輯

`[2,3]`

---

### **Example 2**

python

複製編輯

`edges = [[1,2], [2,3], [3,4], [4,1], [1,5]]`

#### **Step 1: `parent` 陣列**

所有 `parent` 均正常，無 `conflict`。

#### **Step 2: 使用並查集檢查環**

- 加入 `[1,2]`，不形成環。
- 加入 `[2,3]`，不形成環。
- 加入 `[3,4]`，不形成環。
- 加入 `[4,1]`，**發現形成環 `cycle = 3`**。

#### **Step 3: 只有 `cycle`，刪除 `cycle`**

刪除 `[4,1]`，得到：

複製編輯

  `1 → 2 → 3 → 4   1 → 5`

是一棵**有效的有向樹**。

#### **結果**

python

複製編輯

`[4,1]`

---

## **時間複雜度分析**

|步驟|複雜度|
|---|---|
|**Step 1: 建立 `parent` 陣列**|`O(n)`|
|**Step 2: 使用並查集檢查環**|`O(n α(n)) ≈ O(n)`|
|**Step 3: 刪除對應的邊**|`O(1)`|
|**總計**|`O(n)`|

**為什麼 `O(n)`？**

- **遍歷 `edges` 是 `O(n)`**。
- **並查集（Union-Find）的 `find()` 操作是均攤 `O(1)`，整體 `O(n α(n)) ≈ O(n)`**。
- **刪除邊的操作是 `O(1)`**。

---

## **其他解法**

### **1. 拓撲排序（Kahn's Algorithm）**

- **思路**：
    - 建立**入度表（in-degree table）**，找出**入度大於 1** 的節點。
    - 使用 **BFS 拓撲排序** 找到環。
- **時間複雜度**：
    - **建立入度表 `O(n)`**
    - **BFS 檢測環 `O(n)`**
    - **總計 `O(n)`**
- **缺點**：
    - 需要額外的佇列結構（Queue），並查集在處理這類問題時更直覺。

---

## **總結**

### **最佳解法：並查集（Union-Find）**

- **時間複雜度**：`O(n)`
- **思路**：
    1. 先找 `conflict`（是否有節點有兩個父節點）
    2. 再用並查集檢查 `cycle`
    3. 根據 `conflict` 和 `cycle` 來決定要刪除哪條邊

### **其他解法**

- **拓撲排序（BFS）**
    - 透過**入度表**與**拓撲排序**來檢測環與重複父節點。
    - **時間 `O(n)`，但並查集更直覺**。

這道題的核心是 **環檢測** 和 **多個父節點的處理**，**並查集（Union-Find）是最佳解法**，因為它可以**快速合併並檢查是否形成環**，且時間複雜度是 `O(n)`，適合大規模的圖處理問題。

  

O

搜尋

ChatGPT 可能