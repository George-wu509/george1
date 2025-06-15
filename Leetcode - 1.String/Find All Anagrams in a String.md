Lintcode 647
给定一个字符串 `s` 和一个 **非空字符串** `p` ，找到在 `s` 中所有关于 `p` 的字谜的起始索引。  
如果`s`是`p`的一个字谜，则`s`是`p`的一个排列。  
字符串仅由小写英文字母组成，字符串 **s** 和 **p** 的长度不得大于 40,000。  
输出顺序无关紧要。

**样例 1:**

```python
"""
输入 : s = "cbaebabacd", p = "abc"
输出 : [0, 6]
说明 : 
子串起始索引 index = 0 是 "cba"，是"abc"的字谜。
子串起始索引 index = 6 是 "bac"，是"abc"的字谜。
```


```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> list[int]:
        result = []
        char_counts = {}
        for char in p:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        left = 0
        match_count = 0 
        
        for right in range(len(s)):
            current_char = s[right]
            
            if current_char in char_counts:
                char_counts[current_char] -= 1
                if char_counts[current_char] >= 0:
                    match_count += 1
            
            if right - left + 1 == len(p):
                if match_count == len(p):
                    result.append(left)
                
                char_to_remove = s[left]
                if char_to_remove in char_counts:
                    if char_counts[char_to_remove] >= 0:
                        match_count -= 1
                    char_counts[char_to_remove] += 1
                
                left += 1
                
        return result
```


加入解釋的版本
```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> list[int]:
        result = []
        # 1. 構建 p 的字符頻率表
        # 這個字典將用於追蹤在當前窗口中，p 還有哪些字符需要被匹配，以及它們需要的數量。
        char_counts = {}
        for char in p:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # 2. 初始化滑動窗口的左右指針和匹配計數器
        left = 0
        # `match_count` 追蹤當前窗口中，有多少個字符已經成功匹配了 p 的需求。
        # 當 `match_count` 等於 `len(p)` 時，表示找到了異位詞。
        match_count = 0 
        
        # 3. 遍歷主字串 s，移動右指針擴展窗口
        for right in range(len(s)):
            current_char = s[right]
            
            # 如果當前字符在 p 的頻率表中，表示它是我們關心的字符
            if current_char in char_counts:
                char_counts[current_char] -= 1  # 減少需求量
                # 如果減少後，該字符的需求量 >= 0，說明它成功匹配了 p 中的一個字符
                if char_counts[current_char] >= 0:
                    match_count += 1
            
            # 4. 檢查窗口大小並處理左指針移動（收縮窗口）
            # 當窗口大小達到 p 的長度時，我們需要檢查它是否是異位詞
            # 並且無論是否是異位詞，都需要將左指針向右移動，以保持窗口大小或為下一輪準備
            if right - left + 1 == len(p):
                # 如果匹配計數等於 p 的長度，說明當前窗口是一個異位詞
                if match_count == len(p):
                    result.append(left)
                
                # 移動左指針之前，將被移出窗口的字符的計數「歸還」
                char_to_remove = s[left]
                if char_to_remove in char_counts:
                    # 如果該字符在被移出窗口前，其需求量是 >= 0 的，說明它之前是匹配的
                    # 所以在移出後，`match_count` 應該減 1
                    if char_counts[char_to_remove] >= 0:
                        match_count -= 1
                    char_counts[char_to_remove] += 1 # 增加需求量
                
                left += 1 # 左指針向右移動
                
        return result
```
pass

s = "cbaebabacd", 
p = "abc"
解釋: 
step1: 在char_counts的dict儲存p中每个字母出现的次数 {"a":1, "b":1, "c":1}
step2: create雙指針, 左指針left = 0 跟match_count = 0
step3: 從for loop右指針right = 0開始, 檢查值(current_char='c')是否在char_counts的dict裡面, 如果是則將dict中的'c'次數減1. 然後如果次數=>0, match_count+1
step4: 如果滑動窗口(right-left+1)的長度跟p的長度相同, 而且match_count跟p的長度相同, 加入result
step5: 當滑動窗口跟p的長度相同, 並檢查完match_count, 試著將左指針left移除


### 程式碼解釋

這個改進後的程式碼基於標準的**滑動窗口**模式，並使用了以下關鍵元素來確保正確性和簡潔性：

1. **`char_counts` 字典 (頻率需求表)**
    
    - 在函式開始時，它會遍歷字串 `p`，建立一個字典 `char_counts`，儲存 `p` 中每個字符所需的數量。例如，如果 `p = "aba"`，那麼 `char_counts` 會是 `{'a': 2, 'b': 1}`。
    - 這個字典在整個滑動過程中會被動態更新。當一個字符進入窗口時，它的計數會減 1（表示我們“消耗”了一個這種字符）；當一個字符離開窗口時，它的計數會加 1（表示我們“歸還”了一個這種字符）。
    - **核心思想：** 當 `char_counts` 中所有 `p` 的字符計數都變成 0（或負數，代表過多），且窗口長度也正確時，就找到了異位詞。
2. **`left` 和 `right` 指針**
    
    - `left`：滑動窗口的左邊界。
    - `right`：滑動窗口的右邊界。
    - 通過遍歷 `s` 的索引作為 `right` 指針，窗口不斷向右擴展。
3. **`match_count` 計數器**
    
    - 這是判斷是否找到異位詞的**關鍵**。它追蹤的是：**在當前窗口中，有多少個字符是 `p` 所需的，並且其數量已經滿足了 `p` 的要求（或者說，它們在 `char_counts` 中的計數變成了 0 或負數）**。
    - 當 `current_char` 進入窗口，並且 `char_counts[current_char]` 從正數變為 0 或負數時，說明我們成功匹配了一個 `p` 所需的字符，`match_count` 就會增加。
    - 當 `char_to_remove` 離開窗口，並且 `char_counts[char_to_remove]` 從 0 或負數變回正數時，說明我們失去了一個已匹配的字符，`match_count` 就會減少。
    - **最終判斷：** 當 `match_count` 等於 `len(p)` （即 `p` 的總長度）時，表示窗口內的所有字符都已經精確地匹配了 `p` 的所有字符，此時窗口就是一個異位詞。

### 執行流程概括

1. **初始化** `char_counts` (根據 `p`)，`result` 列表，`left` 指針為 0，`match_count` 為 0。
2. **右指針 `right` 遍歷 `s`**，每次循環都將 `s[right]` 納入考慮：
    - 如果 `s[right]` 是 `p` 中的字符，則將 `char_counts[s[right]]` 減 1。
    - 如果 `char_counts[s[right]]` 減 1 後仍然大於或等於 0，表示這個字符是 `p` 所需的且被有效匹配，`match_count` 加 1。
3. **檢查窗口大小**：當 `right - left + 1` 等於 `len(p)` 時（即窗口長度等於 `p` 的長度）：
    - **判斷是否為異位詞**：如果 `match_count` 等於 `len(p)`，則將 `left` 加入 `result` 列表。
    - **左指針移動**：準備將 `s[left]` 移出窗口。
        - 如果 `s[left]` 是 `p` 中的字符，則將 `char_counts[s[left]]` 加 1（歸還）。
        - 如果 `char_counts[s[left]]` 加 1 後，它的值變為大於 0（表示這個字符的需求又恢復了），則 `match_count` 減 1。
        - `left` 指針向右移動 1。
4. 重複步驟 2 和 3 直到 `right` 遍歷完 `s`。
5. 返回 `result` 列表。

這個方法比原始程式碼更健壯，因為它精確地追蹤了 `p` 中每個字符的需求，並在 `match_count` 達到 `p` 的長度時才判斷為異位詞，避免了只檢查窗口長度導致的錯誤。