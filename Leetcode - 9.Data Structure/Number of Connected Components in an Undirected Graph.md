
LeetCode 323 / LintCode 431 你有一个包含 `n` 个节点的图。给定一个整数 `n` 和一个数组 `edges` ，其中 `edges[i] = [ai, bi]` 表示图中 `ai` 和 `bi` 之间有一条无向边。

返回 **图中连通分量的数目** 。

**注意事項：**

- 節點數量 `n` 滿足 `1 小於等於 n 小於等於 2000`
- 邊的數量滿足 `0 小於等於 edges.length 小於等於 5000`
- `edges[i].length == 2`
- `0 小於等於 ai 小於 bi 小於 n`
- `ai != bi`
- 不存在重複的邊（無自環與重邊）。

**样例 1：** 输入：
```Python
n = 5
edges = [[0, 1], [1, 2], [3, 4]]
```

输出：
```Python
2
```

解释： 节点 `0, 1, 2` 互相连通构成第 1 个连通分量；节点 `3, 4` 互相连通构成第 2 个连通分量。因此共有 2 个连通分量。

**样例 2：** 输入：
```Python
n = 5
edges = [[0, 1], [1, 2], [2, 3], [3, 4]]
```

输出：
```Python
1
```

解释： 所有 5 个节点通过边连接成一条链，全部属于同一个连通分量。

**样例 3：** 输入：
```Python
n = 4
edges = []
```

输出：
```Python
4
```

解释： 没有任何边连接节点，4 个节点各自独立，因此有 4 个连通分量。

```Python
# 解法一：並查集（Union-Find / Disjoint Set Union，含路徑壓縮與按秩合併）
def countComponents(self, n: int, edges: list[list[int]]) -> int:
    parent = list(range(n))  # 初始時每個節點的父節點都是自己
    rank = [1] * n           # 記錄每個連通分量的樹高（或大小），用於按秩合併

    def find(x: int) -> int:
        # 路徑壓縮（Path Compression）：查找根節點時順便將沿途節點直接掛到根節點下
        while x != parent[x]:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> int:
        root_x, root_y = find(x), find(y)
        if root_x == root_y:
            return 0  # 已經在同一個連通分量中，無需合併，連通分量數不變

        # 按秩合併（Union by Rank）：將較小的樹掛到較大的樹底下，維持樹平衡
        if rank[root_x] >= rank[root_y]:
            parent[root_y] = root_x
            rank[root_x] += rank[root_y]
        else:
            parent[root_x] = root_y
            rank[root_y] += rank[root_x]
        return 1  # 成功合併兩個獨立分量，總連通分量數減少 1

    components = n
    for u, v in edges:
        components -= union(u, v)

    return components
```


```Python
# 解法二：鄰接表建圖 + 深度優先搜尋（Adjacency List + DFS）
from collections import defaultdict

def countComponents(self, n: int, edges: list[list[int]]) -> int:
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)  # 無向圖需雙向建邊

    visited = set()

    def dfs(node: int) -> None:
        visited.add(node)
        for neighbor in adj[node]:
            if neighbor not in visited:
                dfs(neighbor)

    components = 0
    for i in range(n):
        if i not in visited:
            components += 1  # 發現尚未造訪過的新連通塊起點
            dfs(i)           # 將該連通塊內的所有節點一次標記完畢

    return components
```

解釋: step1: 最初沒有任何邊時，`n` 個節點各自獨立，連通分量數量為 `n`。每當一條邊連接了兩個原本「互不相連」的集合時，總連通分量數量就會減 `1`。 step2: 以 `n = 5, edges = [[0, 1], [1, 2], [3, 4]]` 為例（採用解法一的並查集）：

- 初始化：`parent = [0, 1, 2, 3, 4]`，`components = 5` step3: 讀取邊 `[0, 1]`：`find(0)=0`、`find(1)=1`（不同根），將 `1` 合併到 `0` 底下 → `parent = [0, 0, 2, 3, 4]`，`components = 5 - 1 = 4` step4: 讀取邊 `[1, 2]`：`find(1)=0`、`find(2)=2`（不同根），將 `2` 合併到 `0` 底下 → `parent = [0, 0, 0, 3, 4]`，`components = 4 - 1 = 3` step5: 讀取邊 `[3, 4]`：`find(3)=3`、`find(4)=4`（不同根），將 `4` 合併到 `3` 底下 → `parent = [0, 0, 0, 3, 3]`，`components = 3 - 1 = 2` step6: 所有邊處理完畢，最終回傳 `components = 2`。
    

解釋: 核心規則：求無向圖連通塊有兩大標準範式——**並查集（Union-Find）** 與 **圖遍歷（DFS / BFS）**。

並查集透過「動態合併邊端點」並統計成功合併的次數（`初始節點數 - 成功合併次數`）直接得出答案，特別適合處理「動態加邊」的場景；DFS/BFS 則先建立鄰接表，每次從未訪問的節點出發把整塊連通區域染色，統計啟動遍歷的次數。

### **LeetCode 323 / LintCode 431 - Number of Connected Components in an Undirected Graph（無向圖中連通分量的數量）**

這道題目是 **圖論連通性（Graph Connectivity）** 與 **並查集（Union-Find / Disjoint Set Union）** 的最純粹模板題（Blind 75 & NeetCode 150 必考題）。掌握這題的兩種寫法，幾乎就能直接套用解決絕大多數「朋友圈 / 省份數量 / 島嶼連通 / 冗餘連接」等圖論問題。

## **解法解析**

這道題可以轉化為 **集合動態合併（Union-Find）** 或 **連通塊染色遍歷（DFS / BFS）** 問題來求解。

### **思路**

1. **並查集（Union-Find）的核心思想**
    
    - 把每個連通分量看作一棵樹，樹的根節點（Root）代表整個集合的「老大」。
        
    - **查詢（`find`）**：沿著 `parent` 陣列往上找，直到 `x == parent[x]`。加上 **路徑壓縮（Path Compression）** 後，查詢過的節點會直接指向根節點，讓樹高度趨近於常數。
        
    - **合併（`union`）**：對於每條邊 `[u, v]`，先找出兩者的老大 `root_u` 和 `root_v`。
        
        - 若 `root_u == root_v`：代表 `u` 和 `v` 早在同一個連通分量裡（這條邊是多餘的，會形成環），連通分量總數 **不變**。
            
        - 若 `root_u != root_v`：將其中一棵樹掛到另一棵樹底下（配合 **按秩合併 Union by Rank** 讓小樹掛大樹），兩個集合合二為一，連通分量總數 **減 1**。
            
2. **DFS / BFS 圖遍歷的核心思想**
    
    - 必須先將邊列表 `edges` 轉換為 **鄰接表（Adjacency List）** `adj`，且因為是 **無向圖**，每條邊 `[u, v]` 都要雙向加入：`adj[u].append(v)` 與 `adj[v].append(u)`。
        
    - 遍歷 `0` 到 `n - 1` 的每個節點：只要發現某個節點 `i` 還沒被訪問過（`i not in visited`），就代表發現了一個全新的連通分量（`components += 1`），立刻啟動 DFS 或 BFS 把與 `i` 相連的所有節點全部加入 `visited`。
        
3. **代碼執行流程（以並查集為例）**
    
    - 初始化 `parent = [0, 1, ..., n - 1]`，`rank = [1] * n`，`components = n`。
        
    - 遍歷 `edges` 中的每條邊 `[u, v]`，呼叫 `union(u, v)`：
        
        - 若兩者原本不連通，合併並回傳 `1`，讓 `components` 減 `1`。
            
        - 若兩者已連通，回傳 `0`，`components` 保持不變。
            
    - 遍歷結束後回傳 `components`。
        

## **具體舉例**

假設 `n = 5, edges = [[0, 1], [1, 2], [3, 4]]`，以 **解法一（並查集）** 為例，變數變化如下：

|**步驟**|**當前邊 [u, v]**|**find(u) 與 find(v)**|**合併動作**|**parent 陣列狀態**|**components**|
|---|---|---|---|---|---|
|初始化|-|-|各自獨立為根|`[0, 1, 2, 3, 4]`|`5`|
|1|`[0, 1]`|`0` != `1`|`1` 併入 `0`|`[0, 0, 2, 3, 4]`|`4`|
|2|`[1, 2]`|`0` != `2`|`2` 併入 `0`|`[0, 0, 0, 3, 4]`|`3`|
|3|`[3, 4]`|`3` != `4`|`4` 併入 `3`|`[0, 0, 0, 3, 3]`|`2`|

最後返回 **`2`**。

## **時間與空間複雜度分析**

設 `V = n` 為頂點數量，`E = len(edges)` 為邊的數量：

- **解法一（並查集 Union-Find）**：
    
    - **時間複雜度：O(V + E · α(V))**
        
        - 初始化 `parent` 和 `rank` 陣列需要 $O(V)$ 時間；處理 `E` 條邊時，在同時使用路徑壓縮與按秩合併的情況下，單次 `find` 和 `union` 的攤銷時間複雜度為反阿克曼函數 $O(\alpha(V))$（對於宇宙中所有實際規模的輸入，$\alpha(V)$ 皆小於 `5`，可視為常數 $O(1)$），因此總時間複雜度近似線性 **O(V + E)**。
            
    - **空間複雜度：O(V)**
        
        - 只需要維護兩個長度為 `V` 的一維陣列 `parent` 和 `rank`，無需建立鄰接表，空間複雜度為 **O(V)**。
            
- **解法二（鄰接表 + DFS / BFS）**：
    
    - **時間複雜度：O(V + E)**
        
        - 建立鄰接表需要遍歷所有 `E` 條邊，耗時 $O(E)$；DFS/BFS 遍歷過程中，每個頂點最多被訪問一次，每條無向邊最多被檢查兩次，耗時 $O(V + E)$，總時間複雜度為 **O(V + E)**。
            
    - **空間複雜度：O(V + E)**
        
        - 鄰接表 `adj` 需要儲存 `V` 個節點與 `2E` 條有向邊（佔用 $O(V + E)$ 空間），`visited` 集合與 DFS 遞迴呼叫棧（或 BFS 佇列）最多佔用 $O(V)$ 空間，總空間複雜度為 **O(V + E)**。
            

## **其他解法**

### **1. 鄰接表 + 廣度優先搜尋（BFS）**

- **思路**：
    
    - 與解法二完全相同的鄰接表建圖方式，但將遞迴 `dfs` 改為使用 `collections.deque` 的迭代式 **BFS**，逐層向外擴散標記 `visited`。
        
    - **優點**：當圖是一條極長的單鏈（例如 `n = 10^5`）時，不會像遞迴 DFS 那樣觸發 Python 的預設遞迴深度限制（`RecursionError: maximum recursion depth exceeded`）。
        
- **時間複雜度**：**O(V + E)**
    
- **空間複雜度**：**O(V + E)**
    

## **總結**

- **最佳解法（空間最省 & 動態連通首選）**：**並查集（Union-Find）**，時間複雜度 $O(V + E \cdot \alpha(V))$，空間僅需 $O(V)$（連鄰接表都不用建），且天然支援邊的串流動態輸入。
    
- **最直觀解法（圖論通用模板）**：**鄰接表 + BFS / DFS**，時間 $O(V + E)$，空間 $O(V + E)$，程式碼簡短，若題目規模較大建議優先選用 **BFS** 以避開遞迴棧溢位。
    
- **避坑提醒**：使用 DFS/BFS 解無向圖問題時，建立鄰接表切記要 **雙向加邊**（`adj[u].append(v)` 與 `adj[v].append(u)`），否則會漏算反向連通路徑。
    

## **LeetCode / LintCode 圖論連通性與並查集（Union-Find）類問題整理**

|**題號**|**題目名稱**|**問題類型**|**解法**|
|---|---|---|---|
|**LeetCode 323**|Number of Connected Components in an Undirected Graph（無向圖連通分量數）|**基礎無向圖連通塊計數**|**Union-Find**（計數合併次數）或 **鄰接表 + DFS / BFS**|
|**LeetCode 547**|Number of Provinces（省份數量）|**鄰接矩陣（Adjacency Matrix）連通塊計數**|與 LC 323 本質完全相同，差別在於輸入格式為 `n x n` 矩陣而非邊列表|
|**LeetCode 200**|Number of Islands（島嶼數量）|**2D 網格（Grid）連通塊計數**|**網格 DFS / BFS**（將訪問過的 `'1'` 原地改為 `'0'`）或 **2D 轉 1D 並查集**|
|**LeetCode 261**|Graph Valid Tree（以圖判樹）|**無向圖成樹條件判斷**|**樹的充要條件**：1. 邊數恰好等於 `n - 1`；2. 所有節點連通（連通分量數為 `1`）|
|**LeetCode 684**|Redundant Connection（冗餘連接）|**無向圖找環（導致成環的最後一條邊）**|**Union-Find**：第一條發現 `find(u) == find(v)` 的邊即為冗餘邊|
|**LeetCode 721**|Accounts Merge（帳戶合併）|**字串 / Email 集合連通合併**|**HashMap（Email 映射索引）+ Union-Find**，最後對每個集合內 Email 排序|
|**LeetCode 1319**|Number of Operations to Make Network Connected（連通網路的操作次數）|**連通分量統計 + 冗餘邊判斷**|若邊數小於 `n - 1` 回傳 `-1`；否則所需最少操作次數即為 `連通分量數 - 1`|
|**LeetCode 990**|Satisfiability of Equality Equations（等式方程的可滿足性）|**等價關係傳遞性判斷**|**兩階段處理 + 並查集**：先用 `==` 合併所有相等變數，再檢查 `!=` 兩端是否同根|

### **進階變體練習：LeetCode 261 - Graph Valid Tree（以圖判樹）**

如果題目在「無向圖連通分量」的基礎上，進一步問你這張無向圖 **能否構成一棵合法的樹（Valid Tree）**（樹的定義：沒有環，且所有節點全部連通）：

Python

```
class Solution:
    def validTree(self, n: int, edges: list[list[int]]) -> bool:
        # 樹的必要條件：n 個節點的無向樹必須恰好有 n - 1 條邊
        # 邊數小於 n - 1 必定不連通；邊數大於 n - 1 必定有環
        if len(edges) != n - 1:
            return False

        parent = list(range(n))

        def find(x: int) -> int:
            while x != parent[x]:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for u, v in edges:
            root_u, root_v = find(u), find(v)
            # 若兩個端點已經在同一個集合中，加入這條邊會形成環（Cycle）
            if root_u == root_v:
                return False
            parent[root_u] = root_v

        # 邊數為 n - 1 且無環，必定只有 1 個連通分量，即為合法的樹
        return True
```