Lintcode 171
给出一个字符串数组S，找到其中所有的乱序字符串(Anagram)。如果一个字符串是乱序字符串，那么他存在一个字母集合相同，但顺序不同的字符串也在S中。


```python
"""
输入:["lint", "intl", "inlt", "code"]
输出:["lint", "inlt", "intl"]
```

**样例 2:**

```python
"""
输入:["ab", "ba", "cd", "dc", "e"]
输出: ["ab", "ba", "cd", "dc"]
```


```python
def anagrams(self, strs):
	dict = {}
	for word in strs:
		sortedword = ''.join(sorted(word))
		dict[sortedword] = [word] if sortedword not in dict else dict[sortedword] + [word]
	res = []
	for item in dict:
		if len(dict[item]) >= 2:
			res += dict[item]
	return res
```
pass

本題 **`anagrams(self, strs) -> List[str]`** 的目標是：

- **找出 `strs` 中所有的字母異位詞 (Anagrams)**，並返回這些異位詞的列表。

**異位詞 (Anagram) 的特性：**

- 由相同字母組成，但排列順序不同。
- 例如：`["bat", "tab", "cat"]` 中，`"bat"` 和 `"tab"` 是異位詞。

---

## **解法分析**

**核心想法**：

- **利用排序後的字串作為哈希鍵 (Hash Key)**
- **使用字典 (`dict`) 分組存儲異位詞**
- **最後篩選出包含至少兩個單詞的異位詞組**

---

## **變數表**

|變數名稱|含義|
|---|---|
|`strs`|輸入的字串列表|
|`dict`|存儲 **排序後的字串 -> 原始字串列表** 的字典|
|`word`|當前遍歷的單詞|
|`sortedword`|`word` 按字母排序後的結果，用於作為 `dict` 的鍵|
|`res`|最終結果列表，存儲所有異位詞|

---

## **具體步驟**

### **Step 1: 建立字典 `dict`**

- 遍歷 `strs`，將 **每個字串排序後作為 key**，並將原始字串存入對應的 `list`：

python

複製編輯

`for word in strs:     sortedword = ''.join(sorted(word))     dict[sortedword] = [word] if sortedword not in dict else dict[sortedword] + [word]`

這樣相同字母異位的單詞會被分到同一個 `list` 裡。

### **Step 2: 過濾並收集異位詞**

- 遍歷 `dict`，只保留**出現次數 >= 2** 的異位詞：

python

複製編輯

`for item in dict:     if len(dict[item]) >= 2:         res += dict[item]`

- `res` 存儲最終答案。

---

## **範例解析**

### **範例 1**

python

複製編輯

`strs = ["eat", "tea", "tan", "ate", "nat", "bat"]`

#### **Step 1: 建立 `dict`**

|`word`|`sortedword`|`dict` 更新狀態|
|---|---|---|
|`"eat"`|`"aet"`|`{"aet": ["eat"]}`|
|`"tea"`|`"aet"`|`{"aet": ["eat", "tea"]}`|
|`"tan"`|`"ant"`|`{"aet": ["eat", "tea"], "ant": ["tan"]}`|
|`"ate"`|`"aet"`|`{"aet": ["eat", "tea", "ate"], "ant": ["tan"]}`|
|`"nat"`|`"ant"`|`{"aet": ["eat", "tea", "ate"], "ant": ["tan", "nat"]}`|
|`"bat"`|`"abt"`|`{"aet": ["eat", "tea", "ate"], "ant": ["tan", "nat"], "abt": ["bat"]}`|

#### **Step 2: 過濾異位詞**

- `"aet"` → `["eat", "tea", "ate"]` (✅)
- `"ant"` → `["tan", "nat"]` (✅)
- `"abt"` → `["bat"]` (❌ 過濾掉)

**結果**：

python

複製編輯

`["eat", "tea", "ate", "tan", "nat"]`

---

### **範例 2**

python

複製編輯

`strs = ["abc", "bca", "xyz", "zyx", "aaa"]`

#### **Step 1: 建立 `dict`**

|`word`|`sortedword`|`dict` 更新狀態|
|---|---|---|
|`"abc"`|`"abc"`|`{"abc": ["abc"]}`|
|`"bca"`|`"abc"`|`{"abc": ["abc", "bca"]}`|
|`"xyz"`|`"xyz"`|`{"abc": ["abc", "bca"], "xyz": ["xyz"]}`|
|`"zyx"`|`"xyz"`|`{"abc": ["abc", "bca"], "xyz": ["xyz", "zyx"]}`|
|`"aaa"`|`"aaa"`|`{"abc": ["abc", "bca"], "xyz": ["xyz", "zyx"], "aaa": ["aaa"]}`|

#### **Step 2: 過濾異位詞**

- `"abc"` → `["abc", "bca"]` (✅)
- `"xyz"` → `["xyz", "zyx"]` (✅)
- `"aaa"` → `["aaa"]` (❌ 過濾掉)

**結果**：

python

複製編輯

`["abc", "bca", "xyz", "zyx"]`

---

## **時間與空間複雜度分析**

- **時間複雜度**
    - 遍歷 `strs`：`O(N)`
    - **對每個字串排序**：`O(K log K)` (`K` 是單詞的長度)
    - **總體時間複雜度**： O(N⋅Klog⁡K)O(N \cdot K \log K)O(N⋅KlogK)
- **空間複雜度**
    - `dict` 需要儲存 `N` 個 key-value，最壞情況為 `O(NK)`。
    - `res` 需要存 `N` 個單詞，最壞情況 `O(NK)`。
    - **總體 `O(NK)`**

---

## **其他解法想法（不含代碼）**

1. **計數哈希 (`O(NK)`)**
    
    - 用 **26 個長度的陣列** (`[0]*26`) 來計算字母頻率，而非 `sorted()`
    - 比 `sorted()` 更快，但只適用於**純字母單詞**
2. **使用 `collections.defaultdict` (`O(NK log K)`)**
    
    - `defaultdict(list)` 取代 `dict`，減少 `if sortedword not in dict` 檢查
    - 可減少 `dict` 操作時間
3. **暴力比較 (`O(N^2 K)`)**
    
    - 逐一檢查 `strs[i]` 是否為 `strs[j]` 的異位詞 (`sorted(s1) == sorted(s2)`)
    - **時間複雜度過高，不適用**

---

## **結論**

- **最佳解法**：使用 `sorted()` 作為哈希鍵 (`O(N K log K)`)
- **可替代方法**：
    - **計數哈希 (`O(NK)`)**：更快，但只適用純字母
    - **`defaultdict(list)` (`O(NK log K)`)**：簡潔但相同效率
    - **暴力解法 (`O(N^2 K)`)**：不可行