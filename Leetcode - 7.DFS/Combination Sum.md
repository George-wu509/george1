
LeetCode 39 / LintCode 135 (Blind 75 + NeetCode 150)
給你一個 **無重複元素** 的整數陣列 `candidates` 和一個目標整數 `target` ，找出 `candidates` 中可以使數字和為目標數 `target` 的 **所有不同組合** ，並以列表形式返回。你可以按 **任意順序** 返回這些組合。
`candidates` 中的 **同一個** 數字可以 **無限制重複被選取** 。如果至少一個數字的被選數量不同，則兩種組合是不同的。

**樣例 1：**
輸入：
```python
candidates = [2, 3, 6, 7], target = 7
```
輸出：
```python
[[2, 2, 3], [7]]
```
解釋：
`2` 和 `3` 可以形成一組候選，`2 + 2 + 3 = 7` 。注意 `2` 可以使用多次。
`7` 僅僅是一個候選，`7 = 7` 。
僅有這兩種組合。

**樣例 2：**
輸入：
```python
candidates = [2, 3, 5], target = 8
```
輸出：
```python
[[2, 2, 2, 2], [2, 3, 3], [3, 5]]
```
解釋：
共有 `2+2+2+2`、`2+3+3`、`3+5` 三種組合總和為 `8`

**樣例 3：**
輸入：
```python
candidates = [2], target = 1
```
輸出：
```python
[]
```
解釋：
無法湊出總和 `1`，回傳空陣列

---

### 解法一：決策樹二元選擇回溯法（NeetCode 經典寫法，選或不選）
```python
class Solution:
    def combinationSum(self, candidates, target):
        res = []

        def dfs(i, cur, total):
            if total == target:
                res.append(cur.copy())
                return
            if i >= len(candidates) or total > target:
                return

            cur.append(candidates[i])
            dfs(i, cur, total + candidates[i])
            cur.pop()
            dfs(i + 1, cur, total)

        dfs(0, [], 0)
        return res
```
解釋版
```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []

        def dfs(i: int, cur: List[int], total: int) -> None:
            # 1. 找到一組總和等於 target 的合法組合
            if total == target:
                res.append(cur.copy())
                return
            # 2. 索引越界或總和已經超過 target，直接返回
            if i >= len(candidates) or total > target:
                return

            # 決策 A：選擇當前數字 candidates[i]（索引維持 i，允許下次繼續重複選取）
            cur.append(candidates[i])
            dfs(i, cur, total + candidates[i])
            cur.pop()

            # 決策 B：不選當前數字，直接跳往下一種數字 candidates[i + 1]
            dfs(i + 1, cur, total)

        dfs(0, [], 0)
        return res
```
pass

### 解法二：排序 + 剪枝多叉樹回溯法（面試最佳解，實務執行最快）

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        # 先將陣列由小到大排序，方便後續進行「提前剪枝 (Pruning)」
        candidates.sort()

        def backtrack(start: int, path: List[int], remain: int) -> None:
            # 剩餘目標值剛好歸零，將當前路徑複製一份存入結果
            if remain == 0:
                res.append(path.copy())
                return

            for i in range(start, len(candidates)):
                # 剪枝優化：因為陣列已排序，若當前數字已大於剩餘值，後面的數字只會更大，直接 break！
                if candidates[i] > remain:
                    break

                # 1. 做選擇
                path.append(candidates[i])
                # 2. 遞迴探索：傳入 i（而非 i + 1），允許重複使用同一個數字，但禁止回頭選前面的數字避免重複組合
                backtrack(i, path, remain - candidates[i])
                # 3. 撤銷選擇（回溯）
                path.pop()

        backtrack(0, [], target)
        return res
```

解釋:
step1: 先將 `candidates` 排序（若原本為 `[2, 3, 6, 7]`，排序後不變），並呼叫 `backtrack(start=0, path=[], remain=7)`。
step2: 第一層迴圈（`start = 0, remain = 7`）：
  - `i = 0`（數字 `2`）：`2 <= 7`，加入 `path = [2]`，遞迴呼叫 `backtrack(0, [2], 5)`。
step3: 進入第二層（`start = 0, path = [2], remain = 5`）：
  - `i = 0`（數字 `2`）：`2 <= 5`，加入 `path = [2, 2]`，遞迴呼叫 `backtrack(0, [2, 2], 3)`。
step4: 進入第三層（`start = 0, path = [2, 2], remain = 3`）：
  - `i = 0`（數字 `2`）：`2 <= 3`，加入 `path = [2, 2, 2]`，遞迴呼叫 `backtrack(0, [2, 2, 2], 1)`。
  - 在第四層（`remain = 1`）中，`i = 0` 時 `candidates[0] = 2 > 1`，直接觸發 `break` 剪枝返回！撤銷選擇後 `path` 變回 `[2, 2]`。
  - 回到第三層繼續迴圈，`i = 1`（數字 `3`）：`3 <= 3`，加入 `path = [2, 2, 3]`，遞迴呼叫 `backtrack(1, [2, 2, 3], 0)`。
  - 發現 `remain == 0`，將 `[2, 2, 3]` 存入 `res`！隨後回溯彈出 `3`。
  - 第三層 `i = 2`（數字 `6`）：`6 > 3`，觸發 `break` 結束第三層。
step5: 回到第二層（`path = [2], remain = 5`），嘗試 `i = 1`（數字 `3`）形成 `[2, 3]`（`remain = 2`），下一層因 `start = 1` 最小數字為 `3 > 2` 直接 `break`；繼續嘗試 `i = 2`（數字 `6 > 5`）直接 `break`。
step6: 回到第一層（`path = [], remain = 7`），依序嘗試 `i = 1`（數字 `3`，因不回頭選 `2`，不會產生重複的 `[3, 2, 2]`）、`i = 2`（數字 `6`）、`i = 3`（數字 `7`，湊出 `remain == 0`，存入 `[7]`）。最終回傳 `[[2, 2, 3], [7]]`。

---

### **LeetCode 39 - Combination Sum（組合總和）**

這道題目是 **回溯演算法（Backtracking）** 最核心的經典代表題。本題要求我們在一個不含重複元素的正整數集合中，找出所有和為 `target` 的組合，且每個數字可以 **無限次重複使用**。

---

## **解法解析**

寫好回溯法（DFS）的關鍵在於解決兩個問題：**「如何允許重複選取同一個數字？」** 以及 **「如何避免產生順序不同但內容相同的重複組合（如 `[2, 2, 3]` 與 `[2, 3, 2]`）？」**

### **思路**

1. **避免重複組合的關鍵：`start` 索引限制（不走回頭路）**
   - 組合（Combination）不講究順序，排列（Permutation）才講究順序。
   - 為了避免先選 `2` 再選 `3`，之後又在另一條分支先選 `3` 再選 `2`，我們在每次遞迴時傳入一個起始位置 `start`：規定每一層 `for` 迴圈 **只能從 `start` 往後挑數字**，絕對不回頭選 `start` 之前的數字。
2. **允許重複選取的關鍵：遞迴時傳入 `i` 而非 `i + 1`**
   - 在一般的組合題（每個元素只能用一次）中，選完 `candidates[i]` 後下一層要從 `i + 1` 開始；
   - 在本題中，因為 `candidates[i]` 可以無限次使用，所以遞迴呼叫時依然傳入 `i`（`backtrack(i, path, remain - candidates[i])`），直到該數字加到超過 `remain` 為止，迴圈才會推進到 `i + 1`。
3. **排序 + 剪枝優化（Sorting & Pruning）**
   - 事先執行 `candidates.sort()`。在 `for` 迴圈檢查數字時，只要發現 `candidates[i] > remain`，代表連當前數字都塞不下了，後面的數字只會更大，直接用 `break` 砍掉整棵右半邊子樹，大幅減少無效遞迴。
4. **別忘了深拷貝 `path.copy()`**
   - 在 Python 中，`path` 是一個全程共用的 List 參考（Reference），當 `remain == 0` 時必須寫 `res.append(path.copy())`（或 `path[:]`），否則後續 `path.pop()` 會把已經存進 `res` 的答案也清空！

---

## **具體舉例**

以 `candidates = [2, 3, 6, 7], target = 7` 為例，展示排序剪枝回溯樹的執行軌跡：

| 遞迴深度 | `start` 索引 | 當前路徑 `path` | 剩餘目標 `remain` | 執行動作與剪枝說明 |
|---|---|---|---|---|
| 0 | `0` | `[]` | `7` | 根節點出發，依序嘗試 `2, 3, 6, 7` |
| 1 | `0` | `[2]` | `5` | 選 `2`，下一層仍從 `start = 0` 開始 |
| 2 | `0` | `[2, 2]` | `3` | 再選 `2`，下一層仍從 `start = 0` 開始 |
| 3 | `0` | `[2, 2, 2]` | `1` | 再選 `2`，下一層 `candidates[0] = 2 > 1` 觸發 **`break` 剪枝** |
| 3 | `1` | `[2, 2, 3]` | `0` | 回溯改選 `3`，**`remain == 0` 收集答案 `[2, 2, 3]`**；下個 `6 > 3` 觸發 **`break`** |
| 2 | `1` | `[2, 3]` | `2` | 回溯改選 `3`，下一層從 `start = 1`（最小為 `3 > 2`）觸發 **`break`** |
| 1 | `1` | `[3]` | `4` | 只能往後選 `3, 6, 7`，最終湊不出 `4` 全部剪枝返回 |
| 1 | `2` | `[6]` | `1` | 只能往後選 `6, 7`，湊不出 `1` 返回 |
| 1 | `3` | `[7]` | `0` | 選 `7`，**`remain == 0` 收集答案 `[7]`**，結束搜尋 |

---

## **時間與空間複雜度分析**

設 `N` 為 `candidates` 的長度，`T` 為 `target` 的值，`M` 為 `candidates` 中的最小值。

- **時間複雜度：O(N^(T/M + 1))**
  - 這是一棵多叉樹的搜尋過程，每個節點最多有 `N` 個分支，而樹的最大深度（最多能選幾個數字）取決於不斷選取最小數字 `M` 直到超過 `T`，即 `T / M` 層。
  - 因此最壞情況下的總節點數為 **O(N^(T/M + 1))**（實際上由於排序與剪枝，執行次數會遠小於這個理論上界）。
- **空間複雜度：O(T / M)**
  - 不計算儲存最終答案 `res` 的輸出空間，額外空間主要來自遞迴呼叫堆疊（Recursion Call Stack）以及暫存路徑 `path` 的長度。
  - 遞迴最深只會走到 `T / M` 層，因此空間複雜度為 **O(T / M)**。

---

## **其他解法**

### **1. 完全背包動態規劃（Unbounded Knapsack DP）**
- **思路**：
  - 本題其實等同於「物品可無限次選取、背包容量為 `target`，求所有恰好裝滿背包的方案」。
  - 定義 `dp[t]` 為總和等於 `t` 的所有組合列表，初始化 `dp[0] = [[]]`。外層遍歷每個候選數字 `num`，內層從 `t = num` 遍歷到 `target`，將 `dp[t - num]` 中的每個組合加上 `num` 存入 `dp[t]`。
- **時間複雜度**：O(target × N × 組合平均長度)（需要頻繁複製並建立新的子陣列，常數開銷較回溯法大）
- **空間複雜度**：O(target × 組合數量)（需儲存從 `1` 到 `target` 所有中間狀態的組合列表，記憶體消耗遠高於回溯法）

---

## **總結**

- **面試最佳解**：**排序 + 剪枝多叉樹回溯法**（或 NeetCode 二元決策樹寫法），時間複雜度為指數級搜索樹上界 `O(N^(T/M + 1))`，輔助空間僅需遞迴深度 `O(T / M)`。
- **為什麼不用 DP 而用回溯？**：當題目問「有 **幾種** 組合方法（求數量）」時（如 LeetCode 518 Coin Change II），用動態規劃 `O(N × target)` 最快；但當題目要求「列出 **所有具體組合**（求路徑）」時，回溯法透過原地修改 `path` 與 `pop()` 撤銷選擇，能把空間壓到最小，是唯一首選。

---

## **回溯法（組合 / 子集 / 排列）類問題整理**

| 題號 | 題目名稱 | 問題類型 | 解法 |
|---|---|---|---|
| **LeetCode 39** | Combination Sum（組合總和） | 元素 **無重複**，可 **無限次重複選取**，求總和為 `target` 的組合 | 排序剪枝 + 回溯，遞迴傳入 **`i`** 允許重複選取 |
| **LeetCode 40** | Combination Sum II（組合總和 II） | 元素 **有重複**，每個元素 **只能選一次**，求總和為 `target` 的組合 | 排序 + 同層去重（`if i > start and nums[i] == nums[i-1]: continue`）+ 遞迴傳 **`i + 1`** |
| **LeetCode 216** | Combination Sum III（組合總和 III） | 只用數字 `1 ~ 9`，挑出 **剛好 `k` 個不重複數字** 使其和為 `n` | 雙重終止條件（`len(path) == k` 且 `remain == 0`）+ 剪枝 |
| **LeetCode 77** | Combinations（組合） | 從 `1 ~ n` 中選出所有長度為 `k` 的組合 | 標準組合回溯模板，剩餘元素不足 `k - len(path)` 時可提前剪枝 |
| **LeetCode 78** | Subsets（子集） | 元素 **無重複**，回傳陣列所有可能的子集（冪集） | 回溯模板，在進入遞迴的每個節點都執行 `res.append(path.copy())` |
| **LeetCode 90** | Subsets II（子集 II） | 元素 **有重複**，回傳所有不重複的子集 | 先排序 + 同層跳過相同元素（`i > start and nums[i] == nums[i-1]`） |
| **LeetCode 46** | Permutations（全排列） | 元素 **無重複**，回傳所有可能的排列順序 | 不使用 `start`（每次從 `0` 遍歷），改用 `used` 布林陣列記錄已選元素 |
| **LeetCode 518** | Coin Change II（零錢兌換 II） | 與 LeetCode 39 條件相同，但只求 **組合總數**（不需列出路徑） | **完全背包動態規劃** `dp[j] += dp[j - coin]`（時間 O(N × target)，空間 O(target)） |

---

### **進階變形題：如果 `candidates` 中包含重複數字，且每個數字只能使用一次（LeetCode 40 Combination Sum II）？**

只需在 **解法二（排序 + 剪枝）** 的基礎上改動兩個細節：
1. 在 `for` 迴圈中加入 **同層去重** 判斷：`if i > start and candidates[i] == candidates[i - 1]: continue`（避免同一層選出相同的數字導致重複組合，但允許不同層選到相同的數字）。
2. 遞迴呼叫時改傳 **`i + 1`**（每個元素只能用一次）：

```python
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        candidates.sort()

        def backtrack(start: int, path: List[int], remain: int) -> None:
            if remain == 0:
                res.append(path.copy())
                return

            for i in range(start, len(candidates)):
                # 1. 剪枝：後面數字更大，直接中斷
                if candidates[i] > remain:
                    break
                # 2. 同層去重：同一層樹枝不重複選取相同數值的元素
                if i > start and candidates[i] == candidates[i - 1]:
                    continue

                path.append(candidates[i])
                # 3. 每個元素只能選一次，下一層從 i + 1 開始
                backtrack(i + 1, path, remain - candidates[i])
                path.pop()

        backtrack(0, [], target)
        return res
```
````