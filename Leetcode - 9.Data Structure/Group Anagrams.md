
LeetCode 49 / LintCode 772 給你一個字串陣列 `strs`，請你將 **字母異位詞（Anagrams）** 組合在一起。可以按 **任意順序** 返回結果列表。

**字母異位詞** 是由重新排列源單字的所有字母得到的一個新單字（即兩個字串包含的字母種類與數量完全相同，只是排列順序不同）。

**注意事項：**
- `1 <= strs.length <= 10^4`
- `0 <= strs[i].length <= 100`
- `strs[i]` 僅包含小寫英文字母。

**样例 1：** 输入：
```Python
strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
```

输出：
```Python
[["bat"], ["nat", "tan"], ["ate", "eat", "tea"]]
```

解释：
- 在 `strs` 中沒有字串可以通過重新排列來形成 `"bat"`。
- 字串 `"nat"` 和 `"tan"` 是字母異位詞，因為它們可以重新排列以形成彼此（都包含 1 個 `a`、1 個 `n`、1 個 `t`）。
- 字串 `"ate"`、`"eat"` 和 `"tea"` 是字母異位詞，因為它們可以重新排列以形成彼此（都包含 1 個 `a`、1 個 `e`、1 個 `t`）。

**样例 2：** 输入：
```Python
strs = [""]
```

输出：
```Python
[[""]]
```
解释： 空字串自成一組。

**样例 3：** 输入：
```Python
strs = ["a"]
```

输出：
```Python
[["a"]]
```

解释： 單一字元字串自成一組。
```Python
# 解法一：排序字串作為 HashMap 的 Key（最直觀）
from collections import defaultdict

def groupAnagrams(self, strs: list[str]) -> list[list[str]]:
    anagram_map = defaultdict(list)

    for s in strs:
        # 將字串排序後轉為 tuple（或字串）作為唯一特徵鍵
        key = tuple(sorted(s))
        anagram_map[key].append(s)

    return list(anagram_map.values())
```


```Python
# 解法二：字元頻率計數陣列轉 Tuple 作為 Key（時間複雜度最佳）
from collections import defaultdict

def groupAnagrams(self, strs: list[str]) -> list[list[str]]:
    anagram_map = defaultdict(list)

    for s in strs:
        count = [0] * 26  # 記錄 'a' 到 'z' 出現的次數
        for char in s:
            count[ord(char) - ord('a')] += 1
        # list 不可雜湊（unhashable），需轉為 tuple 才能當作 dict 的 key
        anagram_map[tuple(count)].append(s)

    return list(anagram_map.values())
```

解釋: step1: 字母異位詞的核心特徵是「**每個字母出現的次數完全一樣**」，因此只要能為每組異位詞設計出一個「標準化簽名（Canonical Key）」，就能用 HashMap（字典）將它們歸類到同一個桶子裡。 step2: 以 `strs = ["eat", "tea", "tan", "ate", "nat", "bat"]` 為例（採用解法一的排序簽名）：

- 讀取 `"eat"`：排序後為 `('a', 'e', 't')` -> 存入 `map[('a', 'e', 't')] = ["eat"]`
    
- 讀取 `"tea"`：排序後為 `('a', 'e', 't')` -> 命中同一個 Key -> `["eat", "tea"]`
    
- 讀取 `"tan"`：排序後為 `('a', 'n', 't')` -> 存入 `map[('a', 'n', 't')] = ["tan"]`
    
- 讀取 `"ate"`：排序後為 `('a', 'e', 't')` -> 命中同一個 Key -> `["eat", "tea", "ate"]`
    
- 讀取 `"nat"`：排序後為 `('a', 'n', 't')` -> 命中同一個 Key -> `["tan", "nat"]`
    
- 讀取 `"bat"`：排序後為 `('a', 'b', 't')` -> 存入 `map[('a', 'b', 't')] = ["bat"]` step3: 遍歷結束後，直接回傳 `list(anagram_map.values())`，即為分組後的所有陣列。
    

解釋: 核心規則：為每個字串生成一個「無視排列順序的唯一特徵值（Key）」並存入 `defaultdict(list)`。

生成 Key 有兩種主流方式：**將字串排序**（寫法最簡潔），或是 **統計 26 個英文字母的出現次數**（免去排序開銷，理論時間複雜度更優）。

### **LeetCode 49 / LintCode 772 - Group Anagrams（字母異位詞分組）**

這道題目是 **雜湊表（HashMap）** 與 **字串特徵編碼（Canonical Representation）** 的經典代表題（Blind 75 & NeetCode 150 必考題）。題目的本質在於「分組（Grouping）」，而分組的關鍵是找出一個不隨排列順序改變的 **不變量（Invariant）** 作為雜湊表的鍵值（Key）。

## **解法解析**

這道題可以轉化為 **設計雜湊鍵值（Hash Key Design）+ 桶歸類** 問題來求解。

### **思路**

1. **觀察規律（尋找不變量）**
    
    - 兩個字串互為字母異位詞，若且唯若：
        
        1. **按字典序排序後完全相同**：例如 `"eat"`、`"tea"`、`"ate"` 排序後都是 `"aet"`。
            
        2. **每個字元的頻率分佈完全相同**：例如 `"eat"`、`"tea"`、`"ate"` 都是 `a:1, e:1, t:1`，其餘字母為 `0`。
            
2. **Python 語法細節（為何必須轉成 `tuple`？）**
    
    - 在 Python 中，`dict` 的 Key 必須是 **可雜湊（Hashable / Immutable）** 的物件。
        
    - `sorted(s)` 回傳的是 `list`，而計數陣列 `count = [0] * 26` 也是 `list`，兩者都是可變物件（Mutable），直接當 Key 會觸發 `TypeError: unhashable type: 'list'`。
        
    - 因此必須使用 `tuple(sorted(s))` 或 `tuple(count)`（或 `"".join(sorted(s))`）將其轉為不可變物件。
        
3. **代碼執行流程（以字元頻率計數法為例）**
    
    - 初始化 `anagram_map = defaultdict(list)`。
        
    - 遍歷輸入陣列 `strs` 中的每個字串 `s`：
        
        - 建立長度為 26 的零陣列 `count = [0] * 26`。
            
        - 遍歷字串 `s` 的每個字元 `char`，透過 ASCII 差值 `ord(char) - ord('a')` 將對應位置加 `1`。
            
        - 將 `tuple(count)` 作為 Key，把原字串 `s` 加入 `anagram_map[tuple(count)]` 列表中。
            
    - 最後回傳 `list(anagram_map.values())`。
        

## **具體舉例**

假設 `strs = ["eat", "tea", "tan", "ate", "nat", "bat"]`，以 **解法二（26 字母頻率統計）** 為例，變數變化如下：

|**步驟**|**當前字串 s**|**非零字元頻率統計（轉為 26 維 Tuple Key）**|**anagram_map 對應群組狀態**|
|---|---|---|---|
|1|`"eat"`|`(a:1, e:1, t:1)`|`["eat"]`|
|2|`"tea"`|`(a:1, e:1, t:1)`|`["eat", "tea"]`|
|3|`"tan"`|`(a:1, n:1, t:1)`|`["tan"]`|
|4|`"ate"`|`(a:1, e:1, t:1)`|`["eat", "tea", "ate"]`|
|5|`"nat"`|`(a:1, n:1, t:1)`|`["tan", "nat"]`|
|6|`"bat"`|`(a:1, b:1, t:1)`|`["bat"]`|

最後返回 `list(anagram_map.values())`，即 **`[["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]`**。

## **時間與空間複雜度分析**

設 `n` 為字串陣列 `strs` 的長度，`k` 為陣列中最長字串的長度：

- **解法一（排序作為 Key）**：
    
    - **時間複雜度：O(n · k log k)**
        
        - 需要遍歷 `n` 個字串，對每個長度為 `k` 的字串進行排序需要 $O(k \log k)$ 時間，計算雜湊值需要 $O(k)$ 時間，總時間為 **O(n · k log k)**。
            
    - **空間複雜度：O(n · k)**
        
        - 雜湊表需要儲存所有 `n` 個長度為 `k` 的字串以及對應的 Key，總空間為 **O(n · k)**。
            
- **解法二（26 字母計數陣列作為 Key）**：
    
    - **時間複雜度：O(n · k)**
        
        - 遍歷 `n` 個字串，統計每個字串的字元頻率需要 $O(k)$ 時間，將長度 26 的陣列轉為 `tuple` 並雜湊需要 $O(26)$ 時間，總時間為 $O(n \cdot (k + 26))$，即 **O(n · k)**。
            
    - **空間複雜度：O(n · k)**
        
        - 雜湊表儲存所有字串及最多 `n` 個長度為 26 的 Tuple Key，總空間為 $O(n \cdot (k + 26))$，即 **O(n · k)**。
            

## **其他解法**

### **1. 質數乘積雜湊法（Prime Number Multiplication）**

- **思路**：
    
    - 將 26 個小寫英文字母分別對應到前 26 個質數（`a=2, b=3, c=5, ..., z=101`）。
        
    - 根據算術基本定理（唯一分解定理），任何兩個字串的字母組成完全相同，若且唯若它們對應質數的 **乘積** 完全相等。
        
    - Key 直接是一個整數（Integer），無需建立 Tuple 或排序。
        
- **缺點**：當字串很長時（例如 `k = 100`），乘積會非常龐大（在 C++/Java 中會造成整數溢位，雖然 Python 支援大整數，但大數乘法仍有額外開銷），因此面試中通常僅作為數學趣味補充。
    
- **時間複雜度**：O(n · k)
    
- **空間複雜度**：O(n · k)
    

### **2. 支援 Unicode 任意字元的通用解法（`collections.Counter` + `frozenset`）**

- **思路**：
    
    - 若面試官追問：「**如果輸入包含 Unicode 字元（如中文、Emoji、大小寫混合），不能開固定長度 26 的陣列怎麼辦？**」
        
    - 可使用 `frozenset(Counter(s).items())` 或 `tuple(sorted(Counter(s).items()))` 作為 Key，既能支援任意字元集，又能在字元種類遠小於字串長度時保持高效。
        
- **時間複雜度**：O(n · (k + U log U))，其中 `U` 為單一字串中的不重複字元數
    
- **空間複雜度**：O(n · k)
    

## **總結**

- **面試首選寫法（兼顧簡潔與效率）**：
    
    - 若字串平均長度 `k` 較短（如小於 20），**排序法 `tuple(sorted(s))`** 在 Python 底層由 C 語言 Timsort 實作，實際執行速度往往比純 Python 迴圈計數更快，且程式碼只有 5 行，最不容易寫錯。
        
    - 若 `k` 很大，或面試官明確要求優化掉 $\log k$ 因子，則寫出 **26 字母計數陣列轉 `tuple(count)`**，將時間複雜度降至嚴格線性的 $O(n \cdot k)$。
        
- **面試加分追問點**：主動提及若字元集擴展為 **Unicode** 時，應將固定大小陣列 `[0] * 26` 替換為 `tuple(sorted(Counter(s).items()))` 或 `frozenset(Counter(s).items())`。
    

## **LeetCode / LintCode 雜湊表（HashMap）與字串異位詞類問題整理**

|**題號**|**題目名稱**|**問題類型**|**解法**|
|---|---|---|---|
|**LeetCode 49**|Group Anagrams（字母異位詞分組）|**字串分組 / 特徵雜湊**|**HashMap** + `tuple(sorted(s))` 或 `tuple(count[26])`|
|**LeetCode 242**|Valid Anagram（有效的字母異位詞）|**基礎異位詞判斷**|**長度比對 + 26 字母頻率陣列**（或 `Counter(s) == Counter(t)`）|
|**LeetCode 438**|Find All Anagrams in a String（找到字串中所有字母異位詞）|**固定長度滑動視窗（Sliding Window）**|**滑動視窗 + 頻率陣列比對**，時間複雜度 $O(n)$|
|**LeetCode 567**|Permutation in String（字串的排列）|**固定長度滑動視窗（判斷是否包含異位詞）**|與 LC 438 完全相同，維護長度為 `len(s1)` 的視窗與字元差值計數|
|**LeetCode 76**|Minimum Window Substring（最小覆蓋子串）|**可變長度滑動視窗（進階版）**|**雙指針滑動視窗 + `need` 雜湊表**，維護 `valid` 達標字元數|
|**LeetCode 1**|Two Sum（兩數之和）|**基礎雜湊表查找**|**單次遍歷 HashMap**，存儲 `{數值: 索引}` 尋找 `target - num`|
|**LeetCode 128**|Longest Consecutive Sequence（最長連續序列）|**無序陣列集合查詢**|**HashSet**，僅從 `num - 1` 不在集合中的「序列起點」開始向後計數|
|**LeetCode 347**|Top K Frequent Elements（前 K 個高頻元素）|**頻率統計 + 排序 / 桶排**|**Counter + 桶排序（Bucket Sort）** 達到 $O(n)$ 時間複雜度|
|**LeetCode 271**|Encode and Decode Strings（字串的編碼與解碼）|**字串序列化設計**|**長度前綴編碼（`長度#字串`）**，避免分隔符衝突|

### **進階變體練習：LeetCode 438 - Find All Anagrams in a String（滑動視窗找所有字母異位詞）**

如果題目改為給你一個長字串 `s` 和一個目標字串 `p`，要求找出 `s` 中所有是 `p` 的 **字母異位詞** 的子串起始索引（利用「固定長度滑動視窗 + 26 字母頻率陣列」在 $O(n)$ 內完成）：

Python

```
class Solution:
    def findAnagrams(self, s: str, p: str) -> list[int]:
        n, m = len(s), len(p)
        if n < m:
            return []

        ans = []
        p_count = [0] * 26
        s_count = [0] * 26

        # 初始化目標字串 p 與第一個視窗的字元頻率
        for i in range(m):
            p_count[ord(p[i]) - ord('a')] += 1
            s_count[ord(s[i]) - ord('a')] += 1

        if s_count == p_count:
            ans.append(0)

        # 滑動視窗：每次右邊納入一個新字元，左邊移出一個舊字元
        for i in range(m, n):
            s_count[ord(s[i]) - ord('a')] += 1          # 右端點進入視窗
            s_count[ord(s[i - m]) - ord('a')] -= 1      # 左端點離開視窗

            if s_count == p_count:
                ans.append(i - m + 1)

        return ans
```