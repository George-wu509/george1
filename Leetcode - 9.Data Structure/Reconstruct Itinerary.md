
重新安排行程 Reconstruct Itinerary

Leetcode 332 / Lintcode 1288 给你一份航线列表 `tickets` ，其中 `tickets[i] = [from_i, to_i]` 表示飞机出发和降落的机场地点。请你对该行程进行重新规划排序。

所有这些机票都属于一个从 `JFK`（肯尼迪国际机场）出发的先生，所以该行程必须从 `JFK` 开始。如果存在多种有效的行程，请你按字典序返回最小的行程组合。

- 例如，行程 `["JFK", "LGA"]` 与 `["JFK", "LGB"]` 相比就更小，排序更靠前。

假定所有机票至少存在一种合理的行程。且所有的机票 **必须都用一次 且 只能用一次**。

**样例 1：** 输入：
```Python
tickets = [["MUC","LHR"],["JFK","MUC"],["SFO","SJC"],["LHR","SFO"]]
```

输出：
```Python
["JFK","MUC","LHR","SFO","SJC"]
```

解释： 从 JFK 出发，依次经过 MUC -> LHR -> SFO -> SJC，所有机票刚好用完。

**样例 2：** 输入：
```Python
tickets = [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]
```

输出：
```Python
["JFK","ATL","JFK","SFO","ATL","SFO"]
```

解释： 另一种有效的行程是 `["JFK","SFO","ATL","JFK","ATL","SFO"]` ，但是它字典序更大（第二个机场 ATL < SFO）。

**样例 3（死胡同陷阱）：** 输入：
```Python
tickets = [["JFK","KUL"],["JFK","NRT"],["NRT","JFK"]]
```

输出：
```Python
["JFK","NRT","JFK","KUL"]
```

解释： 如果先贪心去字典序更小的 `KUL`，就会卡在 `KUL` 无法回到 `JFK` 使用剩下的机票。必须先走 `JFK -> NRT -> JFK`，最后再去 `KUL`。

```Python
from collections import defaultdict
import heapq
from typing import List

class Solution:
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        graph = defaultdict(list)
        for src, dst in tickets:
            heapq.heappush(graph[src], dst)
            
        route = []
        
        def dfs(airport: str):
            while graph[airport]:
                next_airport = heapq.heappop(graph[airport])
                dfs(next_airport)
            route.append(airport)
            
        dfs("JFK")
        return route[::-1]
```


```Python
from collections import defaultdict
from typing import List

class Solution:
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        graph = defaultdict(list)
        # 逆序排序，方便后续用 O(1) 的 pop() 取出字典序最小的终点
        for src, dst in sorted(tickets, reverse=True):
            graph[src].append(dst)
            
        route = []
        stack = ["JFK"]
        
        while stack:
            while graph[stack[-1]]:
                stack.append(graph[stack[-1]].pop())
            route.append(stack.pop())
            
        return route[::-1]
```

解釋: step1: 建立鄰接表 `graph`，將每個出發機場的目的地按字典序排列（使用最小堆 `heapq` 或逆序排序後 `pop()`）。 step2: 以 `tickets = [["JFK","KUL"],["JFK","NRT"],["NRT","JFK"]]` 為例，初始 `graph = {"JFK": ["KUL", "NRT"], "NRT": ["JFK"]}`，`stack = ["JFK"]`。 step3: 棧頂為 `"JFK"`，彈出字典序最小的目的地 `"KUL"` 入棧 -> `stack = ["JFK", "KUL"]`。 step4: 棧頂為 `"KUL"`，發現 `graph["KUL"]` 為空（遇到死胡同！），將 `"KUL"` 彈出並加入 `route` -> `route = ["KUL"]`，`stack = ["JFK"]`。 step5: 回到 `"JFK"`，彈出下一個目的地 `"NRT"` 入棧 -> `stack = ["JFK", "NRT"]`；接著從 `"NRT"` 彈出 `"JFK"` 入棧 -> `stack = ["JFK", "NRT", "JFK"]`。 step6: 此時所有機票用完，依序退棧加入 `route` -> `route = ["KUL", "JFK", "NRT", "JFK"]`。 step7: 最後將 `route` 反轉 `route[::-1]`，得到正確行程 `["JFK", "NRT", "JFK", "KUL"]`。

解釋: 這題本質上是尋找有向圖中的**歐拉路徑（Eulerian Path）**——即「一筆畫」走完圖中所有的邊（機票），且起點固定為 `JFK`。 核心算法為 **Hierholzer 算法**：每次貪心選擇字典序最小的邊前進；當某個節點無路可走（死胡同）時，代表它一定是當前子路徑的終點，將其逆序（後序遍歷）加入結果中，最後再整體反轉即可。

### **LeetCode 332 - Reconstruct Itinerary（重新安排行程）**

這道題目是圖論中非常經典的 **歐拉路徑（Eulerian Path）** 問題，標準解法為 **Hierholzer 算法 + 貪心（最小堆 / 排序）**。題目要求從 `"JFK"` 出發，用完所有機票且不重複，並在多組解中返回字典序最小的路徑。

## **解法解析**

這道題不能單純使用「普通 DFS + 貪心」，因為直接選字典序最小的機場可能會走進**死胡同**，導致剩下的機票無法走完。

### **思路**

1. **問題轉化（圖論建模）**
    
    - **節點（Vertex）**：每個機場（如 `"JFK"`, `"SFO"`）。
        
    - **有向邊（Directed Edge）**：每張機票 `[from, to]`。
        
    - **目標**：從 `"JFK"` 出發，經過圖中**每一條邊恰好一次**，且路徑字典序最小。這正是圖論中的 **歐拉路徑（一筆畫問題）**。
        
2. **為何會遇到「死胡同」與 Hierholzer 算法**
    
    - 在存在歐拉路徑的圖中，除了起點之外，**最多只有一個節點**的「入度比出度大 1」（這個節點就是整趟行程的最終終點）。
        
    - 如果我們在遍歷過程中提早走進了這個終點（死胡同），此時手上還有其他環路（Cycle）的機票沒用完。
        
    - **Hierholzer 算法的核心**：當我們在某個節點卡住（出度為 0）時，直接把這個節點放進結果列表的**最前面**（實作上先 `append`，最後再 `reverse`），然後回溯去走完其他環路。這樣一來，死胡同節點自然會被排在整個行程的最後面！
        
3. **代碼執行流程**
    
    - **建圖**：使用 `defaultdict(list)` 儲存每個機場能飛往的目的地，並透過 `heapq`（最小堆）或預先排序，確保每次都能在 $O(1)$ 或 $O(\log E)$ 時間內拿到字典序最小的下一站。
        
    - **後序 DFS 遍歷**：不斷從當前機場取出字典序最小的未訪問機票並遞迴；當該機場沒有剩餘出邊時，將該機場 `append` 到 `route`。
        
    - **反轉結果**：由於最先卡住的終點最早被加入 `route`，最後將 `route[::-1]` 反轉即為從 `"JFK"` 出發的正確順序。
        

## **具體舉例**

假設 `tickets = [["JFK","KUL"],["JFK","NRT"],["NRT","JFK"]]`，初始 `graph = {"JFK": ["KUL", "NRT"], "NRT": ["JFK"]}`，使用迭代棧（Stack）模擬變化如下：

|**步驟**|**當前 stack**|**動作說明**|**graph 剩餘邊**|**route（逆序收集）**|
|---|---|---|---|---|
|初始化|`["JFK"]`|起點入棧|`JFK:[KUL, NRT], NRT:[JFK]`|`[]`|
|Step 1|`["JFK", "KUL"]`|`JFK` 取出最小邊 `KUL`|`JFK:[NRT], NRT:[JFK]`|`[]`|
|Step 2|`["JFK"]`|`KUL` 無出邊（死胡同），彈出加入 `route`|`JFK:[NRT], NRT:[JFK]`|`["KUL"]`|
|Step 3|`["JFK", "NRT"]`|`JFK` 取出下一條邊 `NRT`|`JFK:[], NRT:[JFK]`|`["KUL"]`|
|Step 4|`["JFK", "NRT", "JFK"]`|`NRT` 取出唯一邊 `JFK`|`JFK:[], NRT:[]`|`["KUL"]`|
|Step 5|`["JFK", "NRT"]`|棧頂 `JFK` 無出邊，彈出加入 `route`|空|`["KUL", "JFK"]`|
|Step 6|`["JFK"]`|棧頂 `NRT` 無出邊，彈出加入 `route`|空|`["KUL", "JFK", "NRT"]`|
|Step 7|`[]`|棧頂 `JFK` 無出邊，彈出加入 `route`|空|`["KUL", "JFK", "NRT", "JFK"]`|

最後執行 `route[::-1]`，返回 `["JFK", "NRT", "JFK", "KUL"]`。

## **時間與空間複雜度分析**

設 $E$ 為機票總數（邊數），$V$ 為機場總數（節點數）。

- **時間複雜度：O(E log E)**
    
    - 建圖時需要對所有邊進行排序（或將 $E$ 條邊推入最小堆），耗時 $O(E \log E)$。
        
    - DFS / 棧遍歷每條邊恰好一次，共 $E$ 次操作，每次彈出邊耗時 $O(1)$（預排序列表）或 $O(\log E)$（最小堆）。
        
    - 總時間複雜度為 **$O(E \log E)$**。
        
- **空間複雜度：O(E)**
    
    - 鄰接表 `graph` 需要儲存全部 $E$ 條邊，佔用 $O(E)$ 空間。
        
    - 遞迴調用棧或顯式 `stack` 的最大深度為 $E + 1$，佔用 $O(E)$ 空間。
        

## **其他解法**

### **1. Hierholzer 算法 - 預排序 + 迭代棧（推薦面試寫法）**

- **思路**：
    
    - 先用 `sorted(tickets, reverse=True)` 逆序排列所有機票，放入鄰接表後，每次直接 `list.pop()` 即可在 $O(1)$ 拿到字典序最小的機票。
        
    - 使用顯式 `while stack` 取代遞迴，避免在極端長鏈數據（如 $E = 10^5$）下觸發 Python 的 `RecursionError`。
        
- **時間複雜度**：O(E log E)
    
- **空間複雜度**：O(E)
    

### **2. 普通回溯法（Backtracking DFS）**

- **思路**：
    
    - 將所有機票按目的地排序後，嘗試每一條路徑；若走到死胡同且機票尚未用完，則撤銷選擇（回溯）並嘗試下一個目的地，直到路徑長度等於 `len(tickets) + 1`。
        
- **缺點**：
    
    - 在最壞情況下（存在大量分支與環路），會產生重複搜索，時間複雜度可能退化至指數級 $O(E^d)$，容易超時（TLE）。
        

## **總結**

- **最直觀但易超時**：普通回溯法（Backtracking），遇到死胡同就退回重試，未利用歐拉圖的性質。
    
- **最佳解法（遞迴版）**：**Hierholzer 算法 + `heapq` 最小堆**，代碼簡潔好寫，時間 $O(E \log E)$，空間 $O(E)$。
    
- **最佳解法（工程/防爆棧版）**：**Hierholzer 算法 + 逆序排序 + 顯式 `stack`**，完全避免遞迴深度限制，常數時間更優。
    

只要題目要求「**一筆畫走完所有邊（每條邊用一次）**」，無條件優先想到 **Hierholzer 算法（後序加入結果 + 最後反轉）**。

## **圖論 / 歐拉路徑與行程規劃類問題整理**

|**題號**|**題目名稱**|**問題類型**|**解法**|
|---|---|---|---|
|**LeetCode 332**|Reconstruct Itinerary（重新安排行程）|**有向圖歐拉路徑 + 字典序最小**|**Hierholzer 算法 + 最小堆 / 排序（後序反轉）**|
|**LeetCode 2097**|Valid Arrangement of Pairs（合法重新排列數對）|**有向圖歐拉路徑（無字典序限制）**|**計算入度出度找起點 + Hierholzer 算法**|
|**LeetCode 753**|Cracking the Safe（破解保險箱）|**De Bruijn 序列 / 有向圖歐拉迴路**|**Hierholzer 算法（貪心優先嘗試最大數字）**|
|**LeetCode 207**|Course Schedule（課程表）|**有向圖判環 / 依賴關係**|**拓撲排序（Kahn 算法 BFS / DFS 三色標記）**|
|**LeetCode 210**|Course Schedule II（課程表 II）|**有向無環圖（DAG）求合法順序**|**拓撲排序（入度表 + Queue）**|
|**LeetCode 787**|Cheapest Flights Within K Stops（K 站中轉內最便宜的航班）|**帶邊數限制的最短路徑**|**Bellman-Ford / 限制層數的 BFS / Dijkstra**|
|**LeetCode 743**|Network Delay Time（網絡延遲時間）|**有向加權圖單源最短路徑**|**Dijkstra 算法（最小堆）**|
|**LeetCode 1584**|Min Cost to Connect All Points（連接所有點的最小費用）|**無向圖最小生成樹（MST）**|**Prim 算法 / Kruskal 算法 + 並查集（Union-Find）**|

**進階變形題（LeetCode 2097 - Valid Arrangement of Pairs）：** 給你一個下標從 `0` 開始的二維整數數組 `pairs` ，其中 `pairs[i] = [start_i, end_i]` 。如果對於每一個 `1 <= i < pairs.length` 都有 `end_{i-1} == start_i` ，那麼我們稱這個數組是合法的。請你返回任意一個合法排列（不固定起點為 JFK，需自行尋找歐拉路徑起點）。

Python

```
from collections import defaultdict
from typing import List

class Solution:
    def validArrangement(self, pairs: List[List[int]]) -> List[List[int]]:
        graph = defaultdict(list)
        in_deg = defaultdict(int)
        out_deg = defaultdict(int)
        
        for u, v in pairs:
            graph[u].append(v)
            out_deg[u] += 1
            in_deg[v] += 1
            
        # 尋找歐拉路徑起點：出度比入度大 1 的節點；若無（代表是歐拉迴路），則任選一個有出邊的點
        start_node = pairs[0][0]
        for node in out_deg:
            if out_deg[node] - in_deg[node] == 1:
                start_node = node
                break
                
        route = []
        stack = [start_node]
        
        while stack:
            while graph[stack[-1]]:
                stack.append(graph[stack[-1]].pop())
            route.append(stack.pop())
            
        route.reverse()
        # 將點序列轉換回邊序列 [[u0, u1], [u1, u2], ...]
        return [[route[i], route[i + 1]] for i in range(len(route) -
```