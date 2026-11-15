
單字拆分 Word Break

Leetcode 139 / Lintcode 107

给你一个字符串 `s` 和一个字符串列表 `wordDict` 作为字典。如果可以利用字典中出现的一个或多个单词拼接出 `s` 则返回 `true`。

注意：不要求字典中出现的单词全部都使用，并且字典中的单词可以重复使用。

**样例 1：**

输入：
```Python
s = "leetcode"
wordDict = ["leet", "code"]
```

输出：
```Python
True
```

解释：

返回 True 因为 `"leetcode"` 可以由 `"leet"` 和 `"code"` 拼接成。

**样例 2：**

输入：
```Python
s = "applepenapple"
wordDict = ["apple", "pen"]
```

输出：
```Python
True
```

解释：

返回 True 因为 `"applepenapple"` 可以由 `"apple"` `"pen"` `"apple"` 拼接成。

注意，你可以重复使用字典中的单词。

**样例 3：**

输入：
```Python
s = "catsandog"
wordDict = ["cats", "dog", "sand", "and", "cat"]
```

输出：
```Python
False
```

解释：

虽然可以拆出 `"cats"` + `"and"` 或 `"cat"` + `"sand"`，但剩下的 `"og"` 或 `"dog"` 无法完整匹配字典，因此无法拼接出完整字符串。

```Python
def wordBreak(self, s: str, wordDict: list[str]) -> bool:
	word_set = set(wordDict)
	n = len(s)
	dp = [False] * (n + 1)
	dp[0] = True
	for i in range(1, n + 1):
		for j in range(i):
			if dp[j] and s[j:i] in word_set:
				dp[i] = True
				break
	return dp[n]
```


```Python
def wordBreak(self, s: str, wordDict: list[str]) -> bool:
	n = len(s)
	dp = [False] * (n + 1)
	dp[0] = True
	for i in range(1, n + 1):
		for word in wordDict:
			w_len = len(word)
			if i >= w_len and dp[i - w_len] and s[i - w_len:i] == word:
				dp[i] = True
				break
	return dp[n]
```

解釋:

step1: 本質是「帶字典驗證的爬樓梯」，每次可以走 `len(word)` 步（前提是該段子字串剛好等於字典裡的單字）

step2: 建立長度為 `n + 1` 的布林陣列 `dp`，`dp[i]` 代表前 `i` 個字元能否被成功拆分。初始化 `dp[0] = True`（空字串視為可拆分）

step3: 以 `s = "leetcode"`, `wordDict = ["leet", "code"]` 為例（`n = 8`），外層 `i` 從 1 遍歷到 8：

step4: `i = 1 ~ 3`：長度皆小於 `"leet"`(4) 和 `"code"`(4)，條件不成立，`dp[1..3]` 維持 `False`

step5: `i = 4`（看前 4 碼 `"leet"`）：

- 檢查 `"leet"`（長度 4）：`i >= 4` 且 `dp[4 - 4]` (`dp[0]`) 為 `True`，且 `s[0:4] == "leet"`
    
- 條件全部成立 -> `dp[4] = True`，直接 `break`
    
    step6: `i = 5 ~ 7`：`dp[i - 4]` 皆為 `False`，無法從合法斷點接續，`dp[5..7]` 維持 `False`
    
    step7: `i = 8`（看前 8 碼 `"leetcode"`）：
    
- 檢查 `"code"`（長度 4）：`i >= 4` 且 `dp[8 - 4]` (`dp[4]`) 為 `True`，且 `s[4:8] == "code"`
    
- 條件全部成立 -> `dp[8] = True`，直接 `break`
    
    step8: 迴圈結束，回傳 `dp[8] = True`
    

s = "leetcode", wordDict = ["leet", "code"] -> result = True

1. dp[0] = True ("")
    
2. dp[4] = True ("leet" 在字典中，且 dp[0] 為 True)
    
3. dp[8] = True ("code" 在字典中，且 dp[4] 為 True) 共 1 种完整拆分
    

解釋:

考慮前 `i` 個字元 `s[0..i-1]` 的最後一段切分點 `j`（`0 ≤ j < i`）。

只要存在至少一個切分點 `j`，同時滿足以下兩個條件，`dp[i]` 就為 `True`：

1. **前半段可拆分**：`dp[j] == True`（代表 `s[0..j-1]` 已經能被字典完美拼出）。
    
2. **後半段在字典中**：`s[j:i] in word_set`（代表切出來的最後一個單字是合法的）。
    

- 狀態轉移方程：`dp[i] = any(dp[j] and s[j:i] in word_set for j in range(i))`。
    

### **LeetCode 139 / LintCode 107 - Word Break（單字拆分）**

這道題目是 **1D 動態規劃（字串切分類 DP）** 的核心必考題。不同於「爬樓梯」每次固定走 1 步或 2 步，這題的「步長」是由字典 `wordDict` 裡的單字長度決定的，目標是判斷能否從起點 `0` 一路合法跳躍到終點 `n`。

## **解法解析**

這道題可以轉化為 **前綴可達性（Reachability）動態規劃** 來求解。

### **思路**

1. **狀態定義**
    
    - 定義 `dp[i]` 為布林值（`True` / `False`），表示 **字串 `s` 的前 `i` 個字元（`s[0..i-1]`）能否被拆分成字典中的單字**。
        
    - 最終目標即為求 `dp[n]`（`n = len(s)`）。
        
2. **邊界條件（Base Case）**
    
    - **`dp[0] = True`**：空字串 `""` 不需要任何單字即可構成，作為整個字串第一個單字匹配成功時的起跳基準（例如 `s[0:4]` 在字典中時，需要依賴 `dp[0]` 為 `True`）。
        
3. **兩種常見的內層轉移寫法**
    
    - **寫法 A（枚舉切分點 `j`）**：先將 `wordDict` 轉為 `set` 實現 `O(1)` 查詢，內層枚舉 `0 ≤ j < i`，檢查 `dp[j]` 且 `s[j:i] in word_set`。適合單字很多、但字串 `s` 較短的情況。
        
    - **寫法 B（枚舉字典單字 `word`）**：內層直接遍歷 `wordDict`，檢查 `dp[i - len(word)]` 且 `s[i - len(word):i] == word`。適合單字數量少、但字串 `s` 很長的情況（可避免無效的長子字串切片）。
        
4. **剪枝技巧**
    
    - 一旦找到任何一個合法的 `j`（或 `word`）使 `dp[i] = True`，即可立刻 `break` 跳出內層迴圈，因為我們只需要知道「能不能拼出來」，不需要統計總共有幾種拼法。
        

## **具體舉例**

假設 `s = "catsandog"`, `wordDict = ["cats", "dog", "sand", "and", "cat"]`（`n = 9`），`dp` 陣列變化如下：

|**i**|**當前前綴 s[0:i]**|**檢查合法斷點 j（滿足 dp[j] == True）**|**切出的後綴 s[j:i] 是否在字典**|**dp[i]**|
|---|---|---|---|---|
|`0`|`""`|Base Case|-|**`True`**|
|`1`|`"c"`|`j=0` (`"c"`)|否|`False`|
|`2`|`"ca"`|`j=0` (`"ca"`)|否|`False`|
|`3`|`"cat"`|`j=0` (`"cat"`)|**是**（`"cat"`）|**`True`**|
|`4`|`"cats"`|`j=0` (`"cats"`)|**是**（`"cats"`）|**`True`**|
|`5`|`"catsa"`|`j=0, 3, 4`|否|`False`|
|`6`|`"catsan"`|`j=0, 3, 4`|否|`False`|
|`7`|`"catsand"`|`j=3` (`"sand"`) / `j=4` (`"and"`)|**是**（`"sand"` / `"and"`）|**`True`**|
|`8`|`"catsando"`|`j=0, 3, 4, 7`|否|`False`|
|`9`|`"catsandog"`|`j=0, 3, 4, 7`（`"dog"` 需 `j=6`，但 `dp[6]=False`）|否|**`False`**|

最後返回 `dp[9] = False`。可以看到雖然 `"dog"` 在字典裡，但它的起點 `j = 6`（`"catsan"`）無法被合法拆分，因此無法接續。

## **時間與空間複雜度分析**

假設 `n` 為字串 `s` 的長度，`m` 為 `wordDict` 的單字數量，`L` 為字典中最長單字的長度：

- **時間複雜度：O(n² + m·L) 或 O(n · m · L)**
    
    - **枚舉切分點 `j`（寫法 A）**：建 `set` 需 `O(m·L)`；雙層迴圈 `O(n²)`，每次切片 `s[j:i]` 耗時 `O(n)`，嚴格上界為 **O(n³ + m·L)**（若限制 `i - j ≤ L` 則優化為 **O(n·L² + m·L)**）。
        
    - **枚舉字典單字（寫法 B）**：外層 `n` 次，內層遍歷 `m` 個單字，每次字串比對耗時 `O(L)`，總時間為 **O(n · m · L)**。
        
- **空間複雜度：O(n + m·L)**
    
    - `dp` 陣列需要 `O(n)` 空間，儲存 `word_set` 需要 `O(m·L)` 空間。
        

## **其他解法**

### **1. 記憶化搜索（DFS + Memoization / Top-Down DP）**

- **思路**：
    
    - 定義 `dfs(start)` 表示從索引 `start` 開始的後綴 `s[start:]` 能否被成功拆分。
        
    - 枚舉 `end` 從 `start + 1` 到 `n`，若 `s[start:end] in word_set` 且 `dfs(end)` 為 `True`，則回傳 `True`。
        
    - 使用 `@cache` 或 `memo` 陣列記錄失敗過的 `start` 位置，避免在 `"aaaa...ab"` 這類測試案例中發生指數級退化（TLE）。
        
- **時間複雜度**：O(n³)（加最大單字長度限制可降至 O(n·L²)）
    
- **空間複雜度**：O(n + m·L)
    

### **2. BFS（廣度優先搜索 / 最短路徑視角）**

- **思路**：
    
    - 將索引 `0` 視為起點，`n` 視為終點。若 `s[u:v] in word_set`，則代表有一條從節點 `u` 連向節點 `v` 的有向邊。
        
    - 利用 Queue 從 `0` 開始做 BFS，搭配 `visited` 集合避免重複拜訪同一個索引 `u`，只要能走到 `n` 就回傳 `True`。
        
- **時間複雜度**：O(n³)
    
- **空間複雜度**：O(n + m·L)
    

## **總結**

- **最推薦解法**：**Bottom-Up 1D DP（寫法 A 搭配 `max_len` 剪枝，或寫法 B 遍歷 `wordDict`）**，沒有遞迴深度限制，邏輯直觀且執行效率極高。
    
- **核心思維轉換**：這題本質是「圖的連通性問題」或「變長步數的爬樓梯」，`dp[i]` 代表能否從索引 `0` 透過字典單字作為橋樑跳到索引 `i`。
    
- **避坑重點**：
    
    1. 務必將 `wordDict` 轉成 `set(wordDict)`，否則每次 `in wordDict` 都是 `O(m)` 線性搜尋，會大幅拖慢速度。
        
    2. 寫 DFS 或 BFS 時 **必須加 `memo` 或 `visited`**，否則遇到 `s = "aaaaaaaaab"`, `wordDict = ["a", "aa", "aaa"]` 時會產生 $O(2^n)$ 級別的重複分支而超時。
        

## **Blind 75 / NeetCode 150 1D 動態規劃類問題整理**

|**題號 (LC / Lint)**|**題目名稱**|**問題類型**|**核心狀態轉移方程 / 解法**|
|---|---|---|---|
|**LC 70 / Lint 111**|Climbing Stairs（爬樓梯）|**1D 基礎 DP，每次走 1 或 2 步**|`dp[i] = dp[i-1] + dp[i-2]`|
|**LC 746 / Lint 1020**|Min Cost Climbing Stairs（最小花費爬樓梯）|**1D 帶權重爬樓梯，求最小花費**|`dp[i] = min(dp[i-1] + cost[i-1], dp[i-2] + cost[i-2])`|
|**LC 91 / Lint 512**|Decode Ways（解碼方法）|**1D 字串分段，帶條件走 1 或 2 步**|`dp[i] = (dp[i-1] if 1<=s[i]<=9) + (dp[i-2] if 10<=s[i-1..i]<=26)`|
|**LC 139 / Lint 107**|Word Break（單字拆分）|**1D 字串切分，判斷能否由字典組成**|`dp[i] = any(dp[j] and s[j:i] in wordDict)`|
|**LC 198 / Lint 392**|House Robber（打家劫舍）|**1D 線性陣列，不能選相鄰元素**|`dp[i] = max(dp[i-1], dp[i-2] + nums[i])`|
|**LC 213 / Lint 534**|House Robber II（打家劫舍 II）|**1D 環形陣列，首尾不能同時選**|拆成 `nums[0..n-2]` 與 `nums[1..n-1]` 兩次 House Robber 取最大值|
|**LC 322 / Lint 669**|Coin Change（零錢兌換）|**1D 完全背包，求湊出金額的最少硬幣數**|`dp[i] = min(dp[i], dp[i - coin] + 1)`|
|**LC 152 / Lint 191**|Maximum Product Subarray（乘積最大子陣列）|**1D 子陣列極值，需同時維護最大與最小值**|`max_dp = max(n, max_dp*n, min_dp*n)`（負負得正）|
|**LC 300 / Lint 76**|Longest Increasing Subsequence（最長遞增子序列）|**1D 子序列，求嚴格遞增最大長度**|DP `O(n²)` 或 貪心 + 二分查找 `O(n log n)`|
|**LC 5 / Lint 200**|Longest Palindromic Substring（最長回文子串）|**字串回文判斷**|中心擴展法 `O(n²)` 時間 / `O(1)` 空間|

### **進階延伸題：LeetCode 140 - Word Break II（單字拆分 II）**

給定字串 `s` 和字典 `wordDict`，不再只回傳 `True/False`，而是要求在 `s` 中增加空格來構建一個句子，使句子中的所有單字都在詞典中，**回傳所有這些可能的句子**（任意順序）。

Python

```
class Solution:
    def wordBreak(self, s: str, wordDict: list[str]) -> list[str]:
        word_set = set(wordDict)
        memo = {}  # 記錄從索引 start 開始的所有合法句子組合
        
        def dfs(start: int) -> list[str]:
            if start in memo:
                return memo[start]
            if start == len(s):
                return [""]
            
            res = []
            for end in range(start + 1, len(s) + 1):
                word = s[start:end]
                if word in word_set:
                    # 遞迴取得後半段的所有合法拆分結果
                    for sub_sentence in dfs(end):
                        if sub_sentence == "":
                            res.append(word)
                        else:
                            res.append(word + " " + sub_sentence)
                            
            memo[start] = res
            return res
            
        return dfs(0)
```