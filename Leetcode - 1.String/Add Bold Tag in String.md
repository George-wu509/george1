Lintcode 1127
给定一个字符串s和一个字符串列表dict，你需要添加一对封闭的粗体标记 `<b>` 和 `</b>` 来包装dict中存在的s中的子串。如果两个这样的子串重叠，则需要通过一对封闭的粗体标记将它们包装在一起。此外，如果由粗体标记包装的两个子字符串是连续的，则需要将它们组合在一起

```python
"""
输入: 
s = "abcxyz123"
dict = ["abc","123"]
输出:
"<b>abc</b>xyz<b>123</b>"
```

```python
"""
输入: 
s = "aaabbcc"
dict = ["aaa","aab","bc"]
输出:
"<b>aaabbc</b>c"
```


```python
def add_bold_tag(self, s: str, dict: List[str]) -> str:
	import itertools
	N = len(s)
	mask = [False] * N
	for i in range(N):
		prefix = s[i:]
		for word in dict:
			if prefix.startswith(word):
				for j in range(i, min(i+len(word), N)):
					mask[j] = True

	ans = []
	for incl, grp in itertools.groupby(zip(s, mask), lambda z: z[1]):
		if incl: ans.append("<b>")
		ans.append("".join(z[0] for z in grp))
		if incl: ans.append("</b>")
	return "".join(ans)
```
pass


## **解法思路**

本題 **`add_bold_tag(self, s: str, dict: List[str]) -> str`** 目標是：

- **在 `s` 中所有 `dict` 內的單詞加上 `<b>` 標籤**，並且 **重疊的部分只加一次 `<b>` 標籤**。

例如：

python

複製編輯

`s = "abcxyz123" dict = ["abc", "123"]`

**結果**：

python

複製編輯

`"<b>abc</b>xyz<b>123</b>"`

---

### **解法分析**

1. **使用 `mask` 陣列標記需要加粗的位置**
    
    - `mask[i] = True` 表示 `s[i]` 這個字符應該被 `<b>` 包圍。
    - 遍歷 `s`，如果某個索引 `i` 開始的字串以 `dict` 內的某個單詞開頭，則將該單詞範圍內的 `mask[j]` 設為 `True`。
2. **使用 `itertools.groupby()` 合併連續的 `True`**
    
    - **遍歷 `mask` 來生成最終的加粗字串**
    - **合併連續的 `True` 部分**，確保所有重疊區域只加一次 `<b>`。

---

## **變數表**

|變數名稱|含義|
|---|---|
|`N`|`s` 的長度|
|`mask`|`mask[i] = True` 表示 `s[i]` 需要加粗|
|`i`|當前遍歷 `s` 的索引|
|`prefix`|`s[i:]`，用來檢查是否匹配 `dict` 中的單詞|
|`word`|`dict` 內的單詞|
|`j`|標記 `word` 的範圍內的索引|
|`ans`|存儲最終帶有 `<b>` 的輸出字串|
|`grp`|`groupby()` 產生的字符群組|
|`incl`|`True` 表示該群組需要加 `<b>`|

---

## **具體步驟**

### **Step 1：建立 `mask` 陣列**

- 初始化 `mask = [False] * N`
- **遍歷 `s`，對 `dict` 內的單詞進行 `startswith()` 檢查**
- **若 `s[i:]` 以 `word` 開頭，則將 `mask[i:i+len(word)]` 設為 `True`**

### **Step 2：合併連續的 `True`，加 `<b>` 標籤**

- 使用 **`itertools.groupby()`**，當 `mask` 變化時切割 `s`
- **連續的 `True` 只包裹一次 `<b>`**

---

## **範例解析**

### **範例 1**

python

複製編輯

`s = "abcxyz123" dict = ["abc", "123"]`

**Step 1：標記 `mask`**

|`i`|`s[i:]`|匹配的單詞|`mask` 變化|
|---|---|---|---|
|0|`"abcxyz123"`|`"abc"`|`[T, T, T, F, F, F, F, F, F]`|
|6|`"123"`|`"123"`|`[T, T, T, F, F, F, T, T, T]`|

**Step 2：合併 `True`，加 `<b>`**

- `mask = [T, T, T, F, F, F, T, T, T]`
- **結果**：

python

複製編輯

`"<b>abc</b>xyz<b>123</b>"`

---

### **範例 2**

python

複製編輯

`s = "abcabcabc" dict = ["abc", "bca"]`

**Step 1：標記 `mask`**

|`i`|`s[i:]`|匹配的單詞|`mask` 變化|
|---|---|---|---|
|0|`"abcabcabc"`|`"abc"`|`[T, T, T, F, F, F, F, F, F]`|
|1|`"bcabcabc"`|`"bca"`|`[T, T, T, T, T, T, F, F, F]`|
|3|`"abcabc"`|`"abc"`|`[T, T, T, T, T, T, T, T, T]`|

**Step 2：合併 `True`，加 `<b>`**

- `mask = [T, T, T, T, T, T, T, T, T]`
- **結果**：

python

複製編輯

`"<b>abcabcabc</b>"`

> 因為所有字符都被 `mask` 覆蓋，所以只需要加一次 `<b>`。

---

## **時間與空間複雜度分析**

- **時間複雜度**
    
    - **標記 `mask` (`O(N * M)`)**：
        - 對 `s` 的每個索引 `i`，遍歷 `dict` (`M` 個單詞)
        - **最壞情況**：`O(N * M)`
    - **合併標籤 (`O(N)`)**：
        - 遍歷 `s` 並處理 `groupby()`，為 `O(N)`
    - **總體時間複雜度**：
        - `O(N * M + N) ≈ O(N * M)`
- **空間複雜度**
    
    - `mask` 佔用 `O(N)`
    - `ans` 最壞情況下 `O(N)`
    - **總體 `O(N)`**

---

## **其他解法想法（不含代碼）**

1. **暴力法 (`O(N^2)`)**
    
    - 遍歷 `s` 的所有子串，檢查是否在 `dict` 裡，然後標記 `mask`
    - 時間複雜度太高，對長字串效率差。
2. **Trie + 文字匹配 (`O(N + ΣL)`)**
    
    - 構建 `dict` 的 Trie 樹 (`O(ΣL)`)
    - 使用 **Trie 樹進行匹配**，加快 `s` 的查詢 (`O(N)`)
    - 適合大字典 (`dict` 很大時更快)
3. **KMP / Rabin-Karp (`O(N + M)`)**
    
    - 用 **KMP / Rabin-Karp** 進行 `dict` 內單詞匹配
    - 更適合較長 `s` + `dict` 较短的情況

---

## **結論**

- **最佳方法**：`mask + groupby()` (`O(N * M)`)，簡單且高效。
- **可替代方法**：
    - **Trie (`O(N + ΣL)`)**：適合 `dict` 很大但 `s` 很短的情況。
    - **KMP / Rabin-Karp (`O(N + M)`)**：適合 `s` 很長但 `dict` 較小的情況。
    - **暴力 (`O(N^2)`)**：不可行。