Lintcode 1628
有一条路长为`L`，宽为`W`，在路中有一些圆形障碍物，半径1，有一辆圆形的车，半径2，问车是否能通过这条路。你可以把路面当成二维坐标上的一个矩形，四个点为`(0,0),(0,W),(L,0),(L,W)`,现在你需要从`x=0`出发，到`x=L`，不允许与障碍物接触,且车的所有部分都在`y=0`到`y=W`之间，不允许接触。


**样例 1:**
```python
"""
输入:
8 8
输出:
yes

解释:
给出`L=8`，`W=8`，障碍物坐标为`[[1,1],[6,6]]`。返回`yes`。
车的圆心可以从（0,5)到(2,5)到(5,2)到(8,2)，所以返回yes。
```
**样例 2:**
```python
"""
输入:
8 6
输出:
no

解释:
给出`L=8`，`W=6`，障碍物坐标为`[[1,1]]`,返回`no`。
不管如何驾驶，车总会与障碍物相切或者相交，这都是不被允许的。
```


```python
OBSTAClE_MIN_DISTANCE = 6
BOUND_MIN_DISTANCE = 5

class Solution:
    def driving_problem(self, l, w, obstacles):
        from collections import deque
        
        # consider the upper & bottom line y=w, y=0, all obstacles
        # as nodes in a graph, if the car can not pass between two
        # nodes, we connect the two nodes with an edge in the graph.
        # the car can pass the road only if we CANNOT find a path
        # from start node y=w to the end node y=0
        
        queue = deque([(None, w)])
        visited = set([(None, w)])
        while queue:
            x, y = queue.popleft()
            # y <= 5 means (x, y) can connect the end node y=0
            if y <= BOUND_MIN_DISTANCE:
                return "no"
            for obstacle in obstacles:
                if (obstacle[0], obstacle[1]) in visited:
                    continue
                if not self.is_connected(x, y, obstacle[0], obstacle[1]):
                    continue
                queue.append((obstacle[0], obstacle[1]))
                visited.add((obstacle[0], obstacle[1]))
        return "yes"
    
    def is_connected(self, x1, y1, x2, y2):
        if x1 is None:
            return abs(y1 - y2) <= BOUND_MIN_DISTANCE
        # check the distance between (x1, y1) and (x2, y2) <= 6
        # 6 = 2 x (car radius + obstacle radius)
        return abs(x1 - x2) ** 2 + abs(y1 - y2) ** 2 <= OBSTAClE_MIN_DISTANCE ** 2
```
pass


# **LintCode 1628: Driving Problem 解法詳細解析**

## **問題描述**

有一條**寬度為 `w`、長度無窮大**的道路，車子需要從**起點 `y = w` 行駛到 `y = 0`**。道路上有一些障礙物 `obstacles`，我們要判斷車子**是否能夠安全通過**。

**條件限制**

1. 車子與障礙物都是**圓形**的，**半徑未知，但與障礙物的最小安全距離是 `6`**。
2. 車子無法穿越障礙物，且車子與邊界 (`y=0` 和 `y=w` 兩條線) **必須保持至少 `5` 的距離**。
3. 如果障礙物之間的距離小於 `6`，則它們是**相連的**。
4. 如果障礙物形成一條**從 `y=w` 到 `y=0` 的連通障礙物鏈**，則車子無法通過。

**目標** 我們需要判斷**車子是否能夠從 `y = w` 成功駛入 `y = 0`**，輸出 `"yes"` 或 `"no"`。

---

## **解法分析**

這個問題的**本質**是**判斷障礙物是否形成一條從 `y = w` 到 `y = 0` 的封閉路障**，即：**這些障礙物是否形成了一個連通塊**，使得車子無法穿過。

### **關鍵點**

1. **建模為無向圖**
    
    - 每個障礙物是一個**節點**，如果兩個障礙物的距離 ≤ `6`，則它們之間有一條邊。
    - `y=w` 視為一個起點 `start_node`，`y=0` 視為終點 `end_node`。
    - **如果存在一條連接 `start_node` 和 `end_node` 的路徑，那麼車子無法通過**，返回 `"no"`。
    - **如果 `start_node` 無法與 `end_node` 連接，那麼車子可以通過**，返回 `"yes"`。
2. **使用 BFS（Breadth-First Search） 檢查可達性**
    
    - 從 `y=w`（起點）開始，嘗試遍歷所有障礙物，構造障礙物之間的圖。
    - 如果能夠連接到 `y=0`（終點），則返回 `"no"`。
    - 否則，返回 `"yes"`。

---

## **解法步驟**

### **Step 1: 建立 BFS 搜索隊列**

python

複製編輯

`from collections import deque queue = deque([(None, w)])  # 從 y=w 的無窮遠點開始 visited = set([(None, w)])`

- `queue`：用來進行 **BFS 搜索**，初始包含 `(None, w)`，代表起點 `y=w`。
- `visited`：記錄**已經訪問過的節點**，避免重複搜索。

---

### **Step 2: 執行 BFS，檢查是否能夠到達 `y=0`**

python

複製編輯

`while queue:     x, y = queue.popleft()     # 如果 y <= 5，表示可以與 y=0 連接，則返回 "no"     if y <= BOUND_MIN_DISTANCE:         return "no"`

- 如果當前節點 `y ≤ 5`，表示它可以**直接連接到 `y=0`**，車子無法通過，返回 `"no"`。

---

### **Step 3: 檢查所有障礙物，判斷是否連通**

python

複製編輯

``for obstacle in obstacles:     if (obstacle[0], obstacle[1]) in visited:         continue  # 避免重複訪問      if not self.is_connected(x, y, obstacle[0], obstacle[1]):         continue  # 如果當前障礙物與 `x, y` 不相連，跳過      queue.append((obstacle[0], obstacle[1]))  # 加入隊列     visited.add((obstacle[0], obstacle[1]))  # 標記為已訪問``

- 遍歷 `obstacles`，如果**當前障礙物與 `(x, y)` 相連**，則加入 `queue`，表示可以擴展搜索。
- **相連的條件**由 `self.is_connected(x1, y1, x2, y2)` 決定。

---

### **Step 4: 判斷障礙物是否相連**

python

複製編輯

`def is_connected(self, x1, y1, x2, y2):     if x1 is None:         return abs(y1 - y2) <= BOUND_MIN_DISTANCE  # y=w 是否與障礙物連接     return abs(x1 - x2) ** 2 + abs(y1 - y2) ** 2 <= OBSTAClE_MIN_DISTANCE ** 2  # 判斷兩個障礙物距離`

- **如果 `x1=None`**，表示 `y=w` 的起點，檢查 `y2` 是否與邊界距離 ≤ `5`。
- **一般情況**：如果兩個障礙物的歐幾里得距離 `≤ 6`，則它們相連，返回 `True`。

---

### **Step 5: 如果 BFS 完成後仍無法連接到 `y=0`，返回 `"yes"`**

python

複製編輯

`return "yes"`

- 如果 BFS 遍歷完所有可達的障礙物後 **沒有找到 `y ≤ 5` 的障礙物**，說明車子可以通過，返回 `"yes"`。

---

## **舉例分析**

### **Example 1**

#### **輸入**

python

複製編輯

`l = 10 w = 10 obstacles = [(1, 6), (2, 6), (3, 6), (4, 5)]`

#### **Step 1: 初始化**

ini

複製編輯

`queue = [(None, 10)] visited = {(None, 10)}`

#### **Step 2: BFS 搜索**

|當前節點 `(x, y)`|能否與障礙物相連？|加入 `queue`|
|---|---|---|
|`(None, 10)`|`(1,6), (2,6), (3,6)`|`(1,6), (2,6), (3,6)`|
|`(1,6)`|`(2,6), (3,6), (4,5)`|`(4,5)`|
|`(4,5)`|`y=5`，可連接到 `y=0`|**返回 `"no"`**|

**結果：`"no"`**（障礙物形成封鎖區）

---

### **Example 2**

#### **輸入**

python

複製編輯

`l = 10 w = 10 obstacles = [(1, 8), (2, 7)]`

#### **BFS 搜索**

- `(None, 10)` → `(1,8)`, `(2,7)`
- **都無法到達 `y ≤ 5`**
- **返回 `"yes"`**

---

## **時間與空間複雜度分析**

### **時間複雜度**

|步驟|複雜度|
|---|---|
|**Step 1: 初始化 BFS**|`O(1)`|
|**Step 2: 遍歷所有 `obstacles`**|`O(N)`|
|**Step 3: 每個障礙物最多訪問一次**|`O(N)`|
|**Step 4: 判斷 `is_connected()`**|`O(1)`|
|**總計**|`O(N)`|

### **空間複雜度**

|結構|空間|
|---|---|
|**訪問記錄 `visited`**|`O(N)`|
|**BFS `queue`**|`O(N)`|
|**總計**|`O(N)`|

---

## **其他解法（不需代碼）**

### **1. 使用 DFS（深度優先搜尋）**

- **時間複雜度 `O(N)`**
- **與 BFS 類似，但可能會有較深的遞歸開銷**

### **2. 使用圖論的連通性檢查（並查集）**

- **構建無向圖，檢查 `y=w` 是否與 `y=0` 連接**
- **時間 `O(N α(N))`**

---

## **總結**

- **最佳解法：BFS**
- **時間 `O(N)`，空間 `O(N)`**
- **適合處理障礙物數量較大的情況**
- **比 DFS 更適合大規模搜索**

這道題的本質是**無向圖的連通性問題**，BFS 是最佳選擇，能夠高效地判斷車輛是否可以通過 🚗

  

O

搜尋