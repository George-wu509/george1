Lintcode 1270
给定一个任意的表示勒索信内容的字符串，和另一个字符串表示杂志的内容，写一个方法判断能否通过剪下杂志中的内容来构造出这封勒索信，若可以，返回 true；否则返回 false。

杂志字符串中的每一个字符仅能在勒索信中使用一次。

**样例 1**
```python
"""
输入 : ransomNote = "aa", magazine = "aab"
输出 : true
解析 : 勒索信的内容可以有杂志内容剪辑而来
```
**样例 2**
```python
"""
输入 : ransomNote = "aaa", magazine = "aab"
输出 : false
解析 : 勒索信的内容无法从杂志内容中剪辑而来
```


```python
class Solution:
    def can_construct(self, ransom_note: str, magazine: str) -> bool:
        if len(ransom_note) > len(magazine):
            return False
        return not collections.Counter(ransom_note) - collections.Counter(magazine)
```
pass
解釋:
step1: 將ransomNote跟magazine都各自用dict紀錄各自的字元頻率 
s = "anagram" ->  {'a': 3, 'n': 1, 'g': 1, 'r': 1, 'm': 1}
step2: 比較字元頻率, 如果magazine的每個字元頻率都大於ransomNote的每個字元頻率, True


本題 **`can_construct(self, ransom_note: str, magazine: str) -> bool`** 的目標是判斷 `ransom_note` 是否可以由 `magazine` 中的字母組成，每個字母只能被使用一次。

**核心概念**：

- `ransom_note` 需要的字符數量不能超過 `magazine` 提供的字符數量。
- 若 `ransom_note` 中的所有字符**在 `magazine` 中的出現次數都足夠**，則返回 `True`，否則返回 `False`。

此解法利用 **`collections.Counter`** 計算每個字母的出現次數，然後使用 **字典減法 (`-`)** 判斷是否 `ransom_note` 需要的字符數量**超過** `magazine` 所擁有的字符數量。

---

## **變數表**

|變數名稱|含義|
|---|---|
|`ransom_note`|需要構造的字串|
|`magazine`|可用來構造 `ransom_note` 的來源字串|
|`Counter(ransom_note)`|記錄 `ransom_note` 中各字母的出現次數|
|`Counter(magazine)`|記錄 `magazine` 中各字母的出現次數|

---

## **具體步驟**

1. **長度檢查**
    
    - 若 `ransom_note` 長度**大於** `magazine`，則一定無法構造，直接返回 `False`。
2. **計算字母頻率**
    
    - `collections.Counter(ransom_note)` 計算 `ransom_note` 中**每個字母的數量**。
    - `collections.Counter(magazine)` 計算 `magazine` 中**每個字母的數量**。
3. **檢查是否可構造**
    
    - 直接使用 **字典減法 (`-`)**：
        - `collections.Counter(ransom_note) - collections.Counter(magazine)` **若結果為空**，則 `magazine` 中的所有字母數量足夠，返回 `True`。
        - **若結果不為空**，則 `magazine` 提供的字母不足，返回 `False`。

---

## **範例解析**

### **範例 1**

python

複製編輯

`ransom_note = "aa" magazine = "aab"`

#### **步驟**

|字母|`Counter(ransom_note)`|`Counter(magazine)`|
|---|---|---|
|`'a'`|`2`|`2`|
|`'b'`|`0`|`1`|

- `Counter(ransom_note) - Counter(magazine) = {}` (結果為空)
- **返回 `True`**，因為 `magazine` 中的 `'a'` 足夠使用。

---

### **範例 2**

python

複製編輯

`ransom_note = "aa" magazine = "ab"`

#### **步驟**

|字母|`Counter(ransom_note)`|`Counter(magazine)`|
|---|---|---|
|`'a'`|`2`|`1`|
|`'b'`|`0`|`1`|

- `Counter(ransom_note) - Counter(magazine) = {'a': 1}` (結果非空)
- **返回 `False`**，因為 `magazine` 中 `'a'` 的數量不夠。

---

### **範例 3**

python

複製編輯

`ransom_note = "abc" magazine = "cba"`

#### **步驟**

|字母|`Counter(ransom_note)`|`Counter(magazine)`|
|---|---|---|
|`'a'`|`1`|`1`|
|`'b'`|`1`|`1`|
|`'c'`|`1`|`1`|

- `Counter(ransom_note) - Counter(magazine) = {}` (結果為空)
- **返回 `True`**，因為 `magazine` 提供的字符數量足夠。

---

## **時間與空間複雜度分析**

- **時間複雜度**
    
    - `collections.Counter(ransom_note)` **遍歷 `ransom_note`**，時間為 `O(n)`
    - `collections.Counter(magazine)` **遍歷 `magazine`**，時間為 `O(m)`
    - `Counter(ransom_note) - Counter(magazine)` 進行字典減法，最多檢查 `O(26)` 個字母，視為 **常數時間 `O(1)`**
    - **總體時間複雜度**：`O(n + m)`
- **空間複雜度**
    
    - `collections.Counter(ransom_note)` 和 `collections.Counter(magazine)` 各存 `O(26)` 個字母的計數，視為 **常數空間 `O(1)`**
    - **總體空間複雜度**：`O(1)`

---

## **其他解法想法（不含代碼）**

4. **暴力檢查 (`O(n * m)`)**
    
    - 遍歷 `ransom_note`，對每個字符在 `magazine` **尋找並刪除第一個匹配項**
    - 若無法找到匹配項則返回 `False`
    - 時間複雜度為 `O(n * m)`，不適合長 `magazine`
5. **字典計數 (`O(n + m)`)**
    
    - 手動建立 `dict` 計數 `ransom_note` 和 `magazine`
    - 比較兩個字典是否 `magazine` 能滿足 `ransom_note`
    - 時間複雜度 `O(n + m)`，空間 `O(1)`
6. **陣列計數 (`O(n + m)`)**
    
    - 若只包含小寫字母 (`a-z`)，可用 **固定大小的陣列 (`size=26`)** 來計數，而非 `Counter`
    - 時間 `O(n + m)`，空間 `O(1)`，比 `Counter` 更快

---

## **結論**

- **最佳方法**：`collections.Counter`，時間 `O(n + m)`，空間 `O(1)`
- **可替代方法**：
    - **字典計數 (`O(n + m)`)**：適合 Python 內建 `dict`
    - **陣列計數 (`O(n + m)`)**：更快，但僅適用於小寫字母 `a-z`
    - **暴力搜尋 (`O(n * m)`)**：不適用於長 `magazine`

使用 `collections.Counter` **簡潔且高效**，適合大部分情境。





### **字典減法 (`collections.Counter` 差集運算 `-`) 詳細解釋**

在 Python 的 `collections.Counter` 中，**字典減法 (`-`)** 是一種 **非對稱減法 (asymmetric subtraction)**，其行為如下：

1. **當 `A - B` 時，結果只會包含 `A` **中仍然多出的字母**，但不會包含 `B` 中有而 `A` 沒有的字母。
2. **不會產生負數**，如果 `A` 的某個字母數量小於 `B`，則該字母在結果中不會出現。

這意味著：

- 若 `A - B` **結果為空** (`{}`)，則表示 `A` 需要的字母**都能在 `B` 中找到，且數量足夠**，可以構造 `ransom_note`。
- 若 `A - B` **結果非空**，則表示 `A` 需要的某些字母在 `B` 中數量不夠，因此無法構造 `ransom_note`。

---

### **具體舉例**

#### **範例 1：字母數量完全足夠**

python

複製編輯

`from collections import Counter  ransom_note = "aa" magazine = "aab"  cnt_ransom = Counter(ransom_note) cnt_magazine = Counter(magazine)  result = cnt_ransom - cnt_magazine print(result)  # Output: Counter() (空字典)`

|字母|`Counter(ransom_note)`|`Counter(magazine)`|`Counter(ransom_note) - Counter(magazine)`|
|---|---|---|---|
|`'a'`|`2`|`2`|`0` (被消去)|
|`'b'`|`0`|`1`|不存在於 `ransom_note`，不影響結果|

**結果為空 (`Counter()`)**，表示 `ransom_note` **所有需求** 都能在 `magazine` 中找到，返回 `True`。

---

#### **範例 2：字母數量不足**

python

複製編輯

`ransom_note = "aa" magazine = "ab"  cnt_ransom = Counter(ransom_note) cnt_magazine = Counter(magazine)  result = cnt_ransom - cnt_magazine print(result)  # Output: Counter({'a': 1})`

|字母|`Counter(ransom_note)`|`Counter(magazine)`|`Counter(ransom_note) - Counter(magazine)`|
|---|---|---|---|
|`'a'`|`2`|`1`|`1` (還需要一個 `'a'`)|
|`'b'`|`0`|`1`|不存在於 `ransom_note`，不影響結果|

**結果為 `Counter({'a': 1})`，表示 `'a'` 仍需要 1 個，返回 `False`**。

---

#### **範例 3：magazine 有多餘的字母，但 ransom_note 需求已滿足**

python

複製編輯

`ransom_note = "abc" magazine = "cbaaa"  cnt_ransom = Counter(ransom_note) cnt_magazine = Counter(magazine)  result = cnt_ransom - cnt_magazine print(result)  # Output: Counter() (空字典)`

|字母|`Counter(ransom_note)`|`Counter(magazine)`|`Counter(ransom_note) - Counter(magazine)`|
|---|---|---|---|
|`'a'`|`1`|`3`|`0` (被消去)|
|`'b'`|`1`|`1`|`0` (被消去)|
|`'c'`|`1`|`1`|`0` (被消去)|

- 雖然 `magazine` 有額外的 `'a'`，但 `ransom_note` 需要的數量已經足夠。
- **結果為空 (`Counter()`)，返回 `True`**。

---

#### **範例 4：ransom_note 需要 `magazine` 沒有的字母**

python

複製編輯

`ransom_note = "xyz" magazine = "aabbcc"  cnt_ransom = Counter(ransom_note) cnt_magazine = Counter(magazine)  result = cnt_ransom - cnt_magazine print(result)  # Output: Counter({'x': 1, 'y': 1, 'z': 1})`

|字母|`Counter(ransom_note)`|`Counter(magazine)`|`Counter(ransom_note) - Counter(magazine)`|
|---|---|---|---|
|`'x'`|`1`|`0`|`1` (magazine 沒有 `'x'`)|
|`'y'`|`1`|`0`|`1` (magazine 沒有 `'y'`)|
|`'z'`|`1`|`0`|`1` (magazine 沒有 `'z'`)|

- **結果為 `Counter({'x': 1, 'y': 1, 'z': 1})`，表示 `'x'`, `'y'`, `'z'` 在 `magazine` 中根本不存在，返回 `False`**。

---

### **總結**

1. **`Counter(A) - Counter(B)`** 會**減去 `A` 和 `B` 共有的部分，但不會產生負數**。
2. **只有 `ransom_note` 需要的所有字母數量** 都足夠時，結果才會為**空字典 (`Counter()`)**，才能返回 `True`。
3. **`magazine` 多出的字母不影響結果**，只要 `ransom_note` 需要的部分足夠即可。
4. **`ransom_note` 需要 `magazine` 沒有的字母時，結果一定非空，返回 `False`**。

這個**字典減法**操作讓我們可以 **O(n + m)** **時間內快速判斷** `ransom_note` 是否可以從 `magazine` 構造出來，而不用遍歷 `ransom_note` 一個個去匹配 `magazine`，大大提升效能。