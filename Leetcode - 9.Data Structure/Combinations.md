Lintcode 152
给定两个整数 `n` 和 `k`. 返回从 `1, 2, ... , n` 中选出 `k` 个数的所有可能的组合.


**样例 1:**
```python
"""
输入: n = 4, k = 2
输出: [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
```
**样例 2:**
```python
"""
输入: n = 3, k = 2
输出: [ [1,2],[1,3],[2,3] ]
```


For compare
```python
class Solution:
    def run(self, n, k):
        results = []
        self.dfs(1, n, k, [], results) 
        return results
      
    def dfs(self, start, n, k, num, results):
        if len(num) == k:
            results.append(num[:])
            return
        
        if len(num) + (n - start + 1) < k:
            return

        for i in range(start, n + 1):
            num.append(i)
            self.dfs(i + 1, n, k, num, results)           
            num.pop()
```
為什麼第一個dfs要從start=1開始? 因為這裡沒有nums, 所以1 = nums[0]. 同理所以for loop也是從start到n+1

關於這一行if len(num) + (n - start + 1) < k 是檢查是否有可能湊成長度為k的num. len(num)是目前已經填入的num, (n-start+1)是有幾個數字選擇可以填進去. 如果兩者加起來還不到k, 則決定無法湊成長度k的num. 直接停止

在dfs裡面的results.append(num) 是會出現錯誤, 因為使用 `results.append(num)` 時，你添加到 `results` 列表中的並不是 `num` 列表的內容副本，而是 `num` 列表本身的**引用 (reference)**。這意味著 `results` 列表中所有元素都指向同一個 `num` 列表對象。由於 `num` 列表在遞歸過程中會不斷地被修改（通過 `num.append()` 和 `del num[-1]`），所有在 `results` 中引用的子集都會隨著 `num` 的變化而變化。最終，當所有遞歸結束時，`num` 列表將會被清空（因為所有的 `del num[-1]` 操作），導致 `results` 列表中所有的子集都變成了空列表 `[]`。

所以只要使用results.append(num[:]) 或 results.append(list(num)), 它會創建 `num` 列表的一個淺拷貝 (shallow copy). 就不會出現錯誤

如果沒有這行 `del num[-1]` 會怎麼樣？無限增長的 `num` 列表：** `num` 列表會持續增長，從不移除元素。





```python
class Solution:
    def run(self, n, k):
        results = []
        # nums 在組合問題中是從 1 到 n 的數字序列
        # 我們不需要顯式地創建 nums 列表，可以直接使用範圍
        # self.dfs(nums, 0, [], results) 
        
        # 這裡的 dfs 參數需要調整，因為我們現在有 n 和 k
        # 從數字 1 開始考慮，當前組合為空列表
        self.dfs(1, n, k, [], results) 
        return results
      
    def dfs(self, start, n, k, num, results):
        # 1. 達到目標長度 k 的基線條件
        if len(num) == k:
            results.append(num[:])
            return
        
        # 2. 剪枝優化：如果即使選上所有剩餘的數字也無法達到 k 的長度，就沒必要繼續了
        # num.len: 當前組合的長度
        # (n - start + 1): 從 start 到 n 還有多少個數字可以選擇
        if len(num) + (n - start + 1) < k:
            return

        # 遍歷從 start_num 到 n 的所有數字
        # 這裡的 for 迴圈結構和 subsets 的核心思想保持一致
        for i in range(start, n + 1): # 注意這裡的範圍是到 n (包含 n)
            # 選擇當前數字 i
            num.append(i)
            
            # 遞歸呼叫：從下一個數字 i+1 開始繼續尋找
            # 這是關鍵，確保組合中的數字是遞增的，且不重複
            self.dfs(i + 1, n, k, num, results)  
            
            # 回溯：撤銷選擇，以便探索不包含當前數字 i 的其他路徑
            num.pop() # 或者 del current_combination[-1]
```
pass


當你想用求解 **子集 (Subsets)** 的程式碼結構來解決 **組合 (Combinations)** 問題時，主要的改變在於你需要引入一個目標長度 `k`，並且只在當前組合的長度達到 `k` 時才將其加入結果。

以下是基於您提供的子集程式碼進行修改，以解決組合問題的方案：

1. **`run` 方法更名為 `combine` 並調整參數：**
    
    - `def combine(self, n: int, k: int) -> list[list[int]]:`：將方法名改為 `combine` 更符合問題語義，並明確參數 `n` (數字範圍上限) 和 `k` (組合長度)。
    - `results = []`：保持不變，用於儲存所有結果。
    - `self.dfs(1, n, k, [], results)`：這是主要改變。
        - 不再需要 `nums` 列表，因為數字範圍是隱含的 `1` 到 `n`。
        - `1`：表示我們從數字 `1` 開始考慮選擇。
        - `n`：傳遞 `n` 給 `dfs`，作為數字範圍的上限。
        - `k`：傳遞 `k` 給 `dfs`，作為目標組合的長度。
        - `[]`：依然是空的 `current_combination` 列表，表示當前正在構建的組合。
        - `results`：用於收集最終結果。
2. **`dfs` 方法的參數調整：**
    
    - `def dfs(self, start_num: int, n: int, k: int, current_combination: list[int], results: list[list[int]]):`：
        - `start_num`：替代了原先 `k` 參數的作用，表示當前遞歸層次中，我們應該從哪個數字開始選擇。這確保了組合中數字的遞增性和不重複性。
        - `n` 和 `k`：從 `run` 方法傳遞下來，用於判斷數字範圍和目標長度。
        - `current_combination`：替代了原先的 `num`，名稱更具描述性，表示目前正在建構的組合。
        - `results`：儲存所有組合的總列表。
3. **核心邏輯的調整（在 `dfs` 內部）：**
    
    - **基線條件（完成一個組合）**：
        
        Python
        
        ```
        if len(current_combination) == k:
            results.append(current_combination[:])
            return
        ```
        
        這是最關鍵的改變。子集問題是將**所有**中間狀態的 `num` 都加入 `results`。但對於組合問題，我們只關心長度恰好為 `k` 的組合。當 `current_combination` 的長度達到 `k` 時，我們就找到了 P 一個有效組合，將其副本添加到 `results` 並返回。
        
    - **剪枝優化 (Pruning)**：
        
        Python
        
        ```
        if len(current_combination) + (n - start_num + 1) < k:
            return
        ```
        
        這行程式碼與您之前提供的組合程式碼中的剪枝邏輯相同，用於提高效率。它檢查即使把從 `start_num` 到 `n` 的所有剩餘數字都選上，當前組合的長度是否仍然達不到 `k`。如果達不到，就沒必要繼續探索這個分支了，直接返回。這避免了大量不必要的遞歸調用。
        
    - **迴圈範圍調整：**
        
        Python
        
        ```
        for i in range(start_num, n + 1):
        ```
        
        - 原先的子集程式碼是 `range(k, len(nums))`，這裡的 `k` 是當前考慮的起始索引。
        - 在組合問題中，`i` 代表我們**要選擇的數字本身**。因此，迴圈的起點是 `start_num`（表示從這個數字開始往後選），終點是 `n + 1` (不包含 `n + 1`，所以實際會遍歷到 `n`)。
        - **`start_num` 的作用**：它確保了組合中數字的遞增性。例如，如果 `start_num` 是 `3`，那麼 `i` 會從 `3` 開始，保證選到的數字都大於或等於 `3`，這樣就不會重複生成像 `[2, 1]` 這樣的組合（因為 `[1, 2]` 已經由 `start_num=1` 時生成了）。
    - **選擇、遞歸與回溯：**
        
        Python
        
        ```
        current_combination.append(i) # 選擇當前數字 i
        self.dfs(i + 1, n, k, current_combination, results) # 遞歸，從 i 的下一個數字開始探索
        current_combination.pop() # 回溯，撤銷對 i 的選擇
        ```
        
        這部分與子集程式碼的邏輯基本相同，是回溯法的核心。
        
        - `append(i)`：將當前數字 `i` 加入 `current_combination`。
        - `self.dfs(i + 1, ...)`：遞歸地呼叫 `dfs`，傳遞 `i + 1` 作為新的 `start_num`。這意味著在下一個層級，我們將從 `i` 的下一個數字開始考慮，以保證組合的唯一性和遞增性。
        - `pop()`：回溯操作。當從這個遞歸分支返回時，將 `i` 從 `current_combination` 中移除，以便 `for` 迴圈可以探索其他不包含 `i` 的組合路徑。

---

通過這些修改，你就可以用類似子集解決方案的結構來高效地解決組合問題了。