1246
给定一个仅包含大写英文字母的字符串，您可以将字符串中的任何一个字母替换为的另一个字母，最多替换k次。 执行上述操作后，找到最长的，只含有同一字母的子字符串的长度。

**样例1**
```python
输入:
"ABAB"
2
输出:
4
解释:
将两个'A’替换成两个’B’，反之亦然。
```
**样例2**
```python
输入:
"AABABBA"
1
输出:
4
解释:
将中间的 'A’ 替换为 'B' 后得到 “AABBBBA"。
子字符串"BBBB" 含有最长的重复字符, 长度为4。
```

```python
def characterReplacement(self, s: str, k: int) -> int:
    counter = {}
    left = 0
    max_freq = 0
    answer = 0
    
    for right in range(len(s)):
        counter[s[right]] = counter.get(s[right], 0) + 1
        max_freq = max(max_freq, counter[s[right]])
        
        while (right - left + 1) - max_freq > k:
            counter[s[left]] -= 1
            left += 1
            
        answer = max(answer, right - left + 1)
        
    return answer
```
解釋版
```python
def characterReplacement(self, s: str, k: int) -> int:
    counter = {}
    left = 0
    max_freq = 0
    answer = 0
    
    # 1. 右指針主動向右探索，每次納入一個新字元
    for right in range(len(s)):
        counter[s[right]] = counter.get(s[right], 0) + 1
        max_freq = max(max_freq, counter[s[right]])
        
        # 2. 當視窗「不合法」時（需要替換的字元數 > k），左指針被迫收縮
        # 當前視窗總長度為 right - left + 1
        while (right - left + 1) - max_freq > k:
            counter[s[left]] -= 1
            left += 1
            
        # 3. 走到這裡時，視窗 [left, right] 絕對是合法的，直接更新最大長度
        answer = max(answer, right - left + 1)
        
    return answer
```
pass

解釋:   s="AABABBA", k=1
**解法（`right` 右指針在 `for` 迴圈外、`left` 左指針在 `while` 迴圈內）**，以 `s = "AABABBA", k = 1` 為例的逐步運作解釋如下：

1. **初始化變數：** 創立 `counter` 儲存字元頻率，左指針 `left = 0`，紀錄歷史最高單一字元頻率 `max_freq = 0`，以及最終答案 `answer = 0`。
    
2. **右指針 `right` 向右擴張直到不合法（`right = 0` 到 `4`）：**
    
    - 右指針 `right` 透過 `for` 迴圈從 `index = 0` 開始，一步步往右移動，每次都將 `s[right]` 加入 `counter` 並更新 `max_freq`。
        
    - 當 `right` 走到 `0, 1, 2, 3`（視窗為 `"AABA"`，長度 4，`max_freq = 3`）時，需要替換的非主力字元數為 `4 - 3 = 1`（沒有超過 `k = 1`），視窗皆合法，持續更新 `answer = 4`。
        
    - 接著 `right` 繼續往右移到 **`index = 4`**（納入 `'B'`，視窗變成 `"AABAB"`，長度 5，此時 `counter = {'A': 3, 'B': 2}`，`max_freq = 3`）。此時需要替換的字元數為 `5 - 3 = 2`，**超過了 `k = 1`（視窗不合法）**。
        
3. **觸發內層 `while`，左指針 `left` 右移收縮：**
    
    - 因為視窗不合法，進入 `while` 迴圈：將左指針 `left`（目前在 `index = 0` 的 `'A'`）從 `counter` 中減 1，並把 `left` 往右移到 **`index = 1`**。
        
    - 收縮後視窗變成 `"ABAB"`（`left = 1, right = 4`，長度 4，`max_freq` 維持 3），替換數變回 `4 - 3 = 1 <= k`，視窗重新合法，跳出 `while` 並更新 `answer = max(4, 4) = 4`。
        
4. **重複「右擴張、左收縮」直到 `right` 走完整個字串（`right = 5` 到 `6`）：**
    
    - `right` 繼續往右移到 **`index = 5`**（納入 `'B'`，視窗 `"ABABB"`，長度 5，`counter = {'A': 2, 'B': 3}`，`max_freq = 3`），替換數為 `5 - 3 = 2 > k`（不合法），觸發 `while` 將 `left` 移至 **`index = 2`**（吐出 `'A'`，視窗變 `"BABB"`，長度 4），更新 `answer = 4`。
        
    - `right` 移到最後 **`index = 6`**（納入 `'A'`，視窗 `"BABBA"`，長度 5），不合法，觸發 `while` 將 `left` 移至 **`index = 3`**（吐出 `'B'`，視窗變 `"ABBA"`，長度 4），最終回傳 `answer = 4`。

### **問題描述**

給定一個由大寫英文字母組成的字符串 `s` 和一個整數 `k`，可以將字符串中的最多 `k` 個字符替換為任何其他字符，使替換後的字符串中包含**連續重複字符的最長子串的長度**。

---

### **雙指針解法：逐步解釋**

#### **1. 思路**

雙指針（滑動窗口）方法：

1. **窗口表示有效區間**：
    
    - 使用兩個指針 `i` 和 `j` 表示當前窗口的左右邊界，窗口內的子串為 `s[i:j]`。
    - 在窗口內，我們統計每個字母的出現次數，並記錄窗口中出現次數最多的字符的頻率 `max_freq`。
2. **條件檢查**：
    
    - 如果窗口長度減去 `max_freq`（即窗口內需要替換的字符數）小於等於 `k`，則該窗口是有效的。
    - 如果超出 `k`，則需要縮小窗口，讓左指針 `i` 右移一格。
3. **更新結果**：
    
    - 每次擴大或調整窗口時，計算當前有效窗口的長度，並更新答案 `answer`。

#### **2. 實現步驟**

以下是代碼的邏輯詳細分解：

1. **初始化變量**：
    - `counter`：用於記錄窗口內每個字符的出現次數。
    - `answer`：存儲目前找到的最大有效子串長度。
    - `j`：右指針，用於擴展窗口。
    - `max_freq`：記錄窗口內出現次數最多的字符的頻率。
2. **滑動窗口擴展與調整**：
    
    - **右指針移動**：
        - 將 `s[j]` 加入窗口，更新該字符的出現次數。
        - 更新 `max_freq`。
        - 如果窗口內需要替換的字符數 `j - i - max_freq` 超過 `k`，停止擴展。
    - **結果更新**：
        - 如果替換次數超過 `k`，有效窗口長度為 `j - 1 - i`。
        - 如果替換次數不超過 `k`，有效窗口長度為 `j - i`。
    - **左指針移動**：
        - 當窗口需要收縮時，將 `s[i]` 從窗口移出，並減少其計數。
3. **返回結果**： 最後返回找到的最大有效子串長度。
    

---

#### **3. 詳細代碼執行解釋**
```python
def character_replacement(self, s, k):
    counter = {}  # 初始化窗口內字符頻率的計數字典
    answer = 0    # 初始化答案
    j = 0         # 初始化右指針
    max_freq = 0  # 窗口內字符的最大頻率

    for i in range(len(s)):  # 遍歷每個可能的起點 i
        # 擴展窗口
        while j < len(s) and j - i - max_freq <= k:
            counter[s[j]] = counter.get(s[j], 0) + 1  # 更新當前字符頻率
            max_freq = max(max_freq, counter[s[j]])   # 更新最大頻率
            j += 1  # 右指針右移擴展窗口
        
        # 根據條件更新答案
        if j - i - max_freq > k:  # 替換字符數超過 k
            answer = max(answer, j - 1 - i)
        else:  # 替換字符數不超過 k
            answer = max(answer, j - i)
        
        # 縮小窗口：移除起點字符的計數
        counter[s[i]] -= 1

    return answer

```

---

#### **4. 例子：逐步運行**

##### 輸入：

`s = "AABABBA", k = 1`

##### 運行過程：

1. **初始化**：  
    `counter = {}`, `answer = 0`, `j = 0`, `max_freq = 0`
    
2. **第一輪 (i = 0)**：
    
    - `j = 0, s[j] = 'A'`：
        - 更新：`counter = {'A': 1}`, `max_freq = 1`
        - 窗口長度：`j - i = 1`
    - `j = 1, s[j] = 'A'`：
        - 更新：`counter = {'A': 2}`, `max_freq = 2`
        - 窗口長度：`j - i = 2`
    - `j = 2, s[j] = 'B'`：
        - 更新：`counter = {'A': 2, 'B': 1}`, `max_freq = 2`
        - 窗口長度：`j - i = 3`
    - `j = 3, s[j] = 'A'`：
        - 更新：`counter = {'A': 3, 'B': 1}`, `max_freq = 3`
        - 窗口長度：`j - i = 4`
    - `j = 4, s[j] = 'B'`：
        - 更新：`counter = {'A': 3, 'B': 2}`, `max_freq = 3`
        - 替換字符數超過 `k`，停止擴展。
    - 更新答案：`answer = max(0, 4) = 4`
    - 左指針移動：`counter = {'A': 2, 'B': 2}`
3. **第二輪 (i = 1)**：
    
    - `j = 4, s[j] = 'B'`：
        - 更新：`counter = {'A': 2, 'B': 3}`, `max_freq = 3`
    - 更新答案：`answer = max(4, 4) = 4`
    - 左指針移動：`counter = {'A': 2, 'B': 2}`
4. **最終結果**：`answer = 4`
    

---

#### **5. 時間與空間複雜度分析**

1. **時間複雜度**：
    
    - 右指針 `j` 在整個過程中只向右移動，每個字符最多訪問一次，因此時間複雜度為 **`O(n)`**。
    - 左指針 `i` 也只移動一次，因此總體時間複雜度為 **`O(n)`**。
2. **空間複雜度**：
    
    - 使用了一個字典 `counter`，大小與字符集的數量有關（最多為 26 個字母），因此空間複雜度為 **`O(1)`**。

---

### **其他解法簡述**

1. **暴力法**：
    
    - 對每個可能的起點 `i` 和結尾 `j` 計算子串內需要替換的字符數，檢查是否小於等於 `k`。
    - 時間複雜度：`O(n^2)`。
2. **前綴和法**：
    
    - 使用前綴和計算子串內每個字符的出現次數，快速檢查替換條件。
    - 時間複雜度：`O(n)`，但實現稍複雜。