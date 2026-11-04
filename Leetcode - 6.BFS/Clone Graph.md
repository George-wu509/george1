
### LintCode 137: 克隆图 (Clone Graph)

---

### 问题描述

克隆一张无向图. 无向图的每个节点包含一个 `label` 和一个列表 `neighbors`. 保证每个节点的 `label` 互不相同.

你的程序需要返回一个经过深度拷贝的新图. 新图和原图具有同样的结构, 并且对新图的任何改动不会对原图造成任何影响.

**输入**：
- `node` 是图的一个节点。
**输出**：
- 返回图的克隆。

Example:
```python
输入:
{1,2,4#2,1,4#4,1,2}
输出: 
{1,2,4#2,1,4#4,1,2}
解释:
1------2  
 \     |  
  \    |  
   \   |  
    \  |  
      4   
节点之间使用 '#' 分隔
1,2,4 表示某个节点 label = 1, neighbors = [2,4]
2,1,4 表示某个节点 label = 2, neighbors = [1,4]
4,1,2 表示某个节点 label = 4, neighbors = [1,2]
```


---

### 解法：BFS（广度优先搜索 + 队列）

#### 思路

1. **克隆节点和边**：
    
    - 遍历整个图，逐一复制每个节点和它的边（邻居关系）。
    - 使用一个哈希表 `node_map` 记录原节点和克隆节点的映射关系，避免重复克隆。
2. **广度优先搜索**：
    
    - 从起始节点开始，使用队列依次克隆节点及其邻居。
    - 遍历当前节点的邻居，如果邻居未克隆，则创建克隆节点并加入队列；如果已克隆，则直接添加到当前克隆节点的邻居列表。


### 代码实现

```python
from collections import deque

class Solution:
    def clone_graph(self, node):
        root = node
        if node is None:
            return node
            
        # use bfs algorithm to traverse the graph and get all nodes.
        nodes = self.getNodes(node)
        
        # copy nodes, store the old->new mapping information in a hash map
        mapping = {}
        for node in nodes:
            mapping[node] = UndirectedGraphNode(node.label)
        
        # copy neighbors(edges)
        for node in nodes:
            new_node = mapping[node]
            for neighbor in node.neighbors:
                new_neighbor = mapping[neighbor]
                new_node.neighbors.append(new_neighbor)
        
        return mapping[root]
        
    def getNodes(self, node):
        q = collections.deque([node])
        result = set([node])
        while q:
            head = q.popleft()
            for neighbor in head.neighbors:
                if neighbor not in result:
                    result.add(neighbor)
                    q.append(neighbor)
        return result
```
pass
解釋:
step1  先用getNodes() functionm用BFS將graph存到nodes是一個set 譬如 {1,2,3}  
step2  創建新node mapping = {1': UndirectedGraphNode(1'), 2': UndirectedGraphNode(2'), 3': UndirectedGraphNode(3')}  
step3  複製neighbors (edges)  最後新的node應該為
node1 = UndirectedGraphNode(1)
node2 = UndirectedGraphNode(2)
node3 = UndirectedGraphNode(3)
node1.neighbors = [node2, node3]
node2.neighbors = [node1, node3]
node3.neighbors = [node1, node2]


這段程式碼採用了清晰的 **三步驟 BFS（廣度優先搜尋）算法** 來複製無向圖（Clone Graph）。這種拆分式的寫法非常結構化，邏輯明確。

##### 一、 核心解法邏輯拆解

複製圖的關鍵問題在於：**圖可能包含環（Cycle）**，且點與點之間存在雙向關係。直接遞迴或隨意複製容易陷入死循環或重複創建節點。

這段程式碼將難題拆解為三個步驟：

1. **`getNodes(node)`（尋找所有點）**：從入口節點開始，用 BFS 遍歷整張圖，找出原始圖中的所有點，並收集成集合 `result`
2. **`mapping`（複製所有點）**：遍歷第一步找到的所有舊節點，為每一個舊節點建立一個對應的新節點（複製 `val` / `label`），並用 Hash Map 記錄對應關係：`{舊節點: 新節點}`。
3. **`copy neighbors`（複製所有邊）**：遍歷所有舊節點，找出其原本連接的鄰居，透過 `mapping` 找到對應的新鄰居，並把新鄰居加入新節點的 `neighbors` 串列中。

##### 二、 具體範例圖解分析

###### 範例輸入結構：`graph = {1,2,4#2,1,4#4,1,2}`
- `1,2,4`：節點 **1** 的鄰居是 **2** 和 **4**
- `2,1,4`：節點 **2** 的鄰居是 **1** 和 **4**
- `4,1,2`：節點 **4** 的鄰居是 **1** 和 **2**

這是一個包含三個點（1, 2, 4）且兩兩相連（三角形）的無向圖：
```
 (1) -------- (2)
   \        /
    \      /
     \    /
      (4)
```
##### 三、 執行步驟與變數變化詳解

假設輸入的起點為 **節點 1**（記作 `N1`）。
###### Step 1: 執行 `getNodes(N1)` —— 遍歷圖並蒐集所有節點

- **初始化**：`q = [N1]`, `result = {N1}`
- **Round 1**：
    - 彈出 `head = N1`
    - 檢查 `N1.neighbors` 包含 `[N2, N4]`：
        - `N2` 不在 `result` 中 $\rightarrow$ `result.add(N2)`, `q.append(N2)`
        - `N4` 不在 `result` 中 $\rightarrow$ `result.add(N4)`, `q.append(N4)`
    - 目前 `q = [N2, N4]`, `result = {N1, N2, N4}`
        
- **Round 2**：
    - 彈出 `head = N2`
    - 檢查 `N2.neighbors` 包含 `[N1, N4]`：
        - `N1` 與 `N4` 均已在 `result` 中，跳過
            
- **Round 3**：
    - 彈出 `head = N4`
    - 檢查 `N4.neighbors` 包含 `[N1, N2]`：
        - `N1` 與 `N2` 均已在 `result` 中，跳過
- **`q` 清空，回傳**：`nodes = {N1, N2, N4}`

#### Step 2: 建立點的對應映射表 (`mapping`)

遍歷 `nodes = {N1, N2, N4}`，為每個舊節點建立一個新的無向圖節點物件（記作 `N1'`, `N2'`, `N4'`）：

Plaintext

```
mapping = {
    N1: N1',  (標籤為 1, neighbors 為 [])
    N2: N2',  (標籤為 2, neighbors 為 [])
    N4: N4'   (標籤為 4, neighbors 為 [])
}
```

#### Step 3: 連接邊（複製鄰居關係）

遍歷每一個舊節點，並建立對應新節點的 `neighbors`：

1. **處理 `N1`**：
    
    - `new_node` = `mapping[N1]` $\rightarrow$ `N1'`
        
    - `N1.neighbors` 為 `[N2, N4]`
        
    - 查表找到新鄰居：`mapping[N2]` 是 `N2'`，`mapping[N4]` 是 `N4'`
        
    - 執行：`N1'.neighbors.append(N2')`、`N1'.neighbors.append(N4')`
        
    - 結果：`N1'.neighbors = [N2', N4']`
        
2. **處理 `N2`**：
    
    - `new_node` = `mapping[N2]` $\rightarrow$ `N2'`
        
    - `N2.neighbors` 為 `[N1, N4]`
        
    - 查表找到新鄰居：`mapping[N1]` 是 `N1'`，`mapping[N4]` 是 `N4'`
        
    - 執行：`N2'.neighbors.append(N1')`、`N2'.neighbors.append(N4')`
        
    - 結果：`N2'.neighbors = [N1', N4']`
        
3. **處理 `N4`**：
    
    - `new_node` = `mapping[N4]` $\rightarrow$ `N4'`
        
    - `N4.neighbors` 為 `[N1, N2]`
        
    - 查表找到新鄰居：`mapping[N1]` 是 `N1'`，`mapping[N2]` 是 `N2'`
        
    - 執行：`N4'.neighbors.append(N1')`、`N4'.neighbors.append(N2')`
        
    - 結果：`N4'.neighbors = [N1', N2']`
        

#### Step 4: 回傳結果

- 回傳 `mapping[root]`，即 `mapping[N1]`，為全新的複製圖入口節點 `N1'`。
    

### 四、 複雜度分析

- **時間複雜度**：$O(V + E)$
    
    - $V$ 為節點數（Vertices），$E$ 為邊數（Edges）。
        
    - `getNodes` 遍歷所有點與邊一次：$O(V + E)$。
        
    - 複製點與邊的迴圈總共執行 $V$ 次點的建立，與 $2E$ 次邊的連接：$O(V + E)$。
        
- **空間複雜度**：$O(V)$
    
    - `q`、`result` 以及 `mapping` 最多儲存 $V$ 個節點。