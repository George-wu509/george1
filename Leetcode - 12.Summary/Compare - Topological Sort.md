

|                           |     |
| ------------------------- | --- |
| [[#### General拓撲排序的想法流程]] |     |
|                           |     |
|                           |     |

#### General拓撲排序的想法流程
```
請用圖輔助並詳細解釋拓撲排序(Topological Sort)的想法流程跟模板. 並請整理leetcode中最常出的幾題拓撲排序（Topological Sort）問題, 是不是都跟拓撲排序（Topological Sort）有相似的解法? 或者差別在哪裡
```
**拓撲排序（Topological Sort）**的核心任務，是將一張存在「先後依賴關係」的**有向無環圖（DAG）**，攤平成一條「所有箭頭都由左指向右」的線性序列。

最直覺的生活場景就是**大學修課的擋修規定**：必須先修完前置課程，才能修進階課程。

### 一、 核心想法與圖解流程（Kahn's Algorithm）

演算法只依賴一個關鍵指標——**入度（In-degree）**，也就是「有多少個箭頭指向這個節點」，代表它還有多少個前置條件尚未完成。
- **規則 1**：只有 `in-degree == 0` 的節點，代表所有前置條件都已滿足，可以立刻執行。
- **規則 2**：每完成一個節點，就把它射出的箭頭全部剪斷，讓它指向的鄰居 `in-degree` 減 `1`。

以修課依賴圖為例（`A` 擋修 `B` 與 `C`，且 `B`、`C` 同時擋修 `D`）：

1. **初始狀態（計算入度）**
    
    > **`[A (in:0)]`** ──▶ `[B (in:1)]` ──▶ `[D (in:2)]`
    > 
    > └──▶ `[C (in:1)]` ──────┘
    
    - 只有 `A` 的入度為 `0`，放入佇列：`queue = [A]`，`result = []`
        
2. **第 1 步：取出 `A`，剪斷 `A` 的所有出邊**
    
    > `~~[A]~~` ✂️ **`[B (in:0)]`** ──▶ `[D (in:2)]`
    > 
    > ✂️ **`[C (in:0)]`** ──────┘
    
    - `B` 與 `C` 的入度都從 `1` 降為 `0`，同時取得執行資格：`queue = [B, C]`，`result = [A]`
        
3. **第 2 步：取出 `B`，剪斷 `B ──▶ D`**
    
    > `~~[A]~~` `~~[B]~~` ✂️ `[D (in:1)]`
    > 
    > `[C (in:0)]` ──────┘
    
    - `D` 的入度從 `2` 降為 `1`（因為 `C` 還沒修完），**還不能入隊**：`queue = [C]`，`result = [A, B]`
        
4. **第 3 步：取出 `C`，剪斷 `C ──▶ D`**
    
    > `~~[A]~~` `~~[B]~~` `~~[C]~~` ✂️ **`[D (in:0)]`**
    
    - `D` 的入度終於降為 `0`，放入佇列：`queue = [D]`，`result = [A, B, C]`
        
5. **第 4 步：取出 `D`，驗證是否無環**
    
    > `~~[A]~~` `~~[B]~~` `~~[C]~~` `~~[D]~~`
    
    - `queue` 清空，檢查 `len(result) == 4`（等於總節點數）。若圖中存在互相擋修的**環（Cycle）**，環內節點的入度永遠不會降為 `0`，最終 `len(result)` 就會小於總節點數。
        
![[Pasted image 20261126103319.png]]
### 二、 Python 通用標準模板（Kahn's Algorithm）

絕大多數拓撲排序題目都可以直接套用以下骨架，只需根據題意微調標註的擴充點：


```Python
from collections import deque, defaultdict
from typing import List

def topological_sort(num_nodes: int, edges: List[List[int]]) -> List[int]:
    # 1. 建圖與統計入度（擴充點 A：建圖方向、節點型別）
    graph = defaultdict(list)
    in_degree = [0] * num_nodes  # 若節點為字串，改用 {node: 0 for node in nodes}
    
    for pre, nxt in edges:
        graph[pre].append(nxt)
        in_degree[nxt] += 1

    # 2. 初始化佇列：將所有入度為 0 的節點放入（擴充點 B：需字典序最小時改用 heapq）
    queue = deque([i for i in range(num_nodes) if in_degree[i] == 0])
    result = []

    # 3. 剝洋蔥式 BFS（擴充點 C：若需計算層數/最少時間，在此加一層 for _ in range(len(queue))）
    while queue:
        curr = queue.popleft()
        result.append(curr)
        
        for neighbor in graph[curr]:
            # （擴充點 D：若為 DAG 上的動態規劃，在此更新 dp[neighbor]）
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # 4. 環偵測：排出的節點數必須等於總節點數
    return result if len(result) == num_nodes else []
```

### 三、 LeetCode 最常考的拓撲排序題目與差異分析

所有拓撲排序題目都共用「計算度數 $\rightarrow$ 度數歸零入隊 $\rightarrow$ 削弱鄰居度數」**的核心骨架，但考官會透過**4 個維度製造變化：

|**題號與題目**|**核心任務**|**與標準模板的關鍵差異**|
|---|---|---|
|**207. Course Schedule**<br><br>  <br><br>（課程表 I）|判斷能否修完所有課程|**完全符合標準模板**，最後只需回傳 `len(result) == numCourses`（布林值）。|
|**210. Course Schedule II**<br><br>  <br><br>（課程表 II）|回傳任意一種合法修課順序|**完全符合標準模板**，直接回傳 `result` 陣列即可。|
|**269. Alien Dictionary**<br><br>  <br><br>（外星字典）|從已排序的外星單字推導字母表|1. **建圖邏輯**：需比較相鄰單字的第一個不同字元來抽邊，且要攔截前綴不合法（如 `["abc", "ab"]`）。<br><br>  <br><br>2. **容器差異**：若要求多解時字典序最小，需將 `deque` 換成 `heapq`（Min-Heap）。|
|**1136. Parallel Courses**<br><br>  <br><br>（平行修課）|每學期可同時修無擋修的課，求最少幾學期修完|**遍歷節奏（逐層 BFS）**：`while queue:` 內部必須加上 `for _ in range(len(queue)):` 一次清空當學期所有入度為 `0` 的課，每輪 `semesters += 1`。|
|**310. Minimum Height Trees**<br><br>  <br><br>（最小高度樹）|在無向樹中找能讓樹高最小的根節點|**無向圖由外向內剝離**：邊沒有方向，改統計每個點的連接數（`degree`）。將所有 `degree == 1` 的最外圍葉子節點入隊，一層層往內剝，直到圖中只剩下 `<= 2` 個中心節點。|
|**802. Find Eventual Safe States**<br><br>  <br><br>（最終的安全狀態）|找出所有「無論怎麼走都不會掉進環裡」的節點|**反向建圖（Reverse Graph）**：原圖中「出度為 `0` 的終點」是安全點。將所有邊反轉（`u -> v` 變成 `v -> u`）後，終點就變成入度為 `0` 的起點，直接套用標準模板即可。|
|**1857. Largest Color Value in a Directed Graph**<br><br>  <br><br>（有向圖中最大顏色值）|找出一條路徑，使路徑上出現最多次的顏色數量最大化|**拓撲排序 + 動態規劃（DAG DP）**：拓撲順序天然保證無後效性。在剪斷 `curr -> neighbor` 時，順便用 `dp[curr]` 更新 `dp[neighbor]` 的各顏色最大值。|

#### 總結：什麼時候該想到拓撲排序？

只要題目出現以下任一特徵，就可以直接套用這套模板思考：

1. **隱含先後順序**：出現「先修、依賴、解鎖、排序規則、誰在誰前面」等字眼。
    
2. **需要偵測有向圖是否有環**：或要求找出「不在環上的安全節點」（配合反向建圖）。
    
3. **在有向無環圖（DAG）上求最長路徑 / 最大權重 / 最少完成時間**：利用拓撲序由上游往下游遞推（DP）。