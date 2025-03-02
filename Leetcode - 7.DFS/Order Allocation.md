### **LintCode 1909：Order Allocation**

#### **題目描述**

給定一個 `n x n` 的分數矩陣 `score`，每行表示某位工人執行某項工作的得分。要求將每個工作分配給唯一的一位工人，使總得分最大。

Example:
```python
输入：
[[1,2,4],[7,11,16],[37,29,22]]
输出：
[1,2,0]
解释：
标号为0的订单给标号为1的司机，获得 score[0][1] = 2 分，
标号为1的订单给标号为2的司机，获得 score[1][2] = 16 分，
标号为2的订单给标号为0的司机，获得 score[2][0] = 37 分，
所以一共获得了 2 + 16 + 37 = 55 分。
```
解釋:
[1,2,4] 代表標號0的訂單, 司機1得score 1, 司機2得score 2, 司機3得score 4
[37,29,22] 代表標號2的訂單, 司機1得score 37, 司機2得score 29, 司機3得score 22
所以要score最大就是標號0->司機2, 標號1->司機3, 標號2->司機0

---

### **DFS 解法**

#### **算法思路**

我們使用深度優先搜索（DFS）來枚舉所有可能的分配方案，並計算得分，最終找到最大得分的分配方式。

1. **狀態表示**：
    
    - 每個工作只能分配給唯一的工人。
    - 我們需要追蹤哪些工人已被分配，哪些工作已被分配。
2. **遞歸終止條件**：
    
    - 當所有工作都已被分配（即分配的工作數等於總數 `n`）時，計算當前分配方案的總得分，並更新最大得分。
3. **遞歸過程**：
    
    - 遍歷所有未被分配的工作，嘗試將該工作分配給當前工人，然後進一步遞歸處理。
    - 在回溯過程中，撤銷當前的分配，恢復狀態，探索其他分支。
4. **優化（剪枝）**：
    
    - 如果當前的部分得分已經不可能超過當前最大得分，可以提前終止。

---
Example:
```python
输入：
[[1,2,4],[7,11,16],[37,29,22]]
输出：
[1,2,0]
解释：
标号为0的订单给标号为1的司机，获得 score[0][1] = 2 分，
标号为1的订单给标号为2的司机，获得 score[1][2] = 16 分，
标号为2的订单给标号为0的司机，获得 score[2][0] = 37 分，
所以一共获得了 2 + 16 + 37 = 55 分。
```


#### **代碼詳解**

```python
class Solution:
    def order_allocation(self, score):
        if not score:
            return None

        n = len(score)  # 工人和工作的數量
        visited = set()  # 已被分配的工作
        answer = [float("-inf"), None]  # 最大得分和對應的分配方案

        self.dfs([], visited, score, answer)
        return answer[1]

    def dfs(self, orders, visited, score, answer):
        n = len(score)  # 獲取工人數

        # 遞歸終止條件：所有工作都已被分配
        if len(orders) == n:
            current_score = sum(score[i][orders[i]] for i in range(n))  # 計算得分
            if current_score > answer[0]:
                answer[0], answer[1] = current_score, list(orders)
            return

        # 遍歷所有工作
        for i in range(n):
            if i in visited:  # 跳過已分配的工作
                continue

            # 選擇當前工作
            visited.add(i)
            orders.append(i)
            self.dfs(orders, visited, score, answer)  # 遞歸分配下一個工人
            orders.pop()  # 回溯：撤銷當前分配
            visited.remove(i)

```
pass

#### **執行過程舉例**

輸入：

`score = [[1, 2, 4], [7, 11, 16], [37, 29, 22]]`

**步驟 1：初始化**

- `n = 3`，`visited = set()`，`orders = []`，`answer = [-inf, None]`。

**步驟 2：遞歸搜索**

- 開始分配工作給工人：
    1. **工人 0**：
        - 選擇工作 0：`orders = [0]`，`visited = {0}`。
            - 遞歸到 **工人 1**：
                - 選擇工作 1：`orders = [0, 1]`，`visited = {0, 1}`。
                    - 遞歸到 **工人 2**：
                        - 選擇工作 2：`orders = [0, 1, 2]`，計算得分：

                            `score = 1 + 11 + 22 = 34`
                            
                            更新 `answer = [34, [0, 1, 2]]`。
                        - 回溯到 `orders = [0, 1]`。
                    - 選擇工作 2，重複過程。
                - 回溯到 `orders = [0]`。
        - 選擇工作 1 或工作 2，重複過程。

**步驟 3：探索所有分支**

- 重複上述過程，最終發現最大得分為 `score = 1 + 16 + 37 = 54`，對應分配方案 `[2, 1, 0]`。

---

### **複雜度分析**

1. **時間複雜度**：
    
    - 理論上，DFS 遍歷所有分配方案，共有 n!n!n! 種可能性。
    - 每種方案需要計算得分，時間為 O(n)O(n)O(n)。
    - 總時間複雜度為 **O(n×n!)O(n \times n!)O(n×n!)**。
2. **空間複雜度**：
    
    - 遞歸深度最多為 nnn，每層需要存儲當前的 `orders` 和 `visited`。
    - 空間複雜度為 **O(n)O(n)O(n)**。

---

### **其他解法簡述**

#### 1. **動態規劃 (Bitmask DP)**

- 狀態表示：`dp[mask]` 表示已分配 `mask` 所對應工作時的最大得分。
- 狀態轉移：

    `dp[new_mask] = max(dp[new_mask], dp[mask] + score[i][j])`
    
- 時間複雜度：**O(n2⋅2n)O(n^2 \cdot 2^n)O(n2⋅2n)**。

#### 2. **啟發式搜索 (Branch and Bound)**

- 在搜索過程中使用啟發式方法剪枝，減少不必要的分支。
- 適合處理較大的數據集，但需要設計有效的剪枝策略。

#### 3. **暴力法**

- 遍歷所有排列，計算每種排列的得分。
- 時間複雜度為 **O(n×n!)O(n \times n!)O(n×n!)**，僅適合小規模數據。