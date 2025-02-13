Lintcode 1243
计算字符串中的单词数，其中一个单词定义为不含空格的连续字符串。


```python
"""
输入: "Hello, my name is John"
输出: 5
解释：有五个字符串段落："Hello"、"my"、"name"、"is"、"John"
```


```python
    def count_segments(self, s: str) -> int:
        segment_count = 0

        for i in range(len(s)):
            if (i == 0 or s[i - 1] == ' ') and s[i] != ' ':
                segment_count += 1

        return segment_count
```
pass


```python
    def count_segments(self, s: str) -> int:
        segment = s.strip().split(" ")

        return len(segment)
```
not pass

```python
    def count_segments(self, s: str) -> int:
        segment = s.strip().split()

        return len(segment)
```
pass

本題 **`count_segments(self, s: str) -> int`** 目標是 **計算字串 `s` 中的單詞數量**。  
「單詞」是由**非空格字符組成的連續序列**，例如：

- `"Hello, my friend!"` → `3`
- `" "` → `0`
- `"a b c"` → `3`（空格不應影響計數）

---

### **為何 `s.strip().split(" ")` 會失敗？**

#### **錯誤點：`split(" ")` 會保留多餘的空字串**

- `s.split(" ")` 會**按照單一空格切割**，但 **連續空格會產生空字串 (`""`)**。
- 這導致 `len(segment)` 的結果 **錯誤**。

---

### **錯誤範例解析**

#### **範例 1**

python

複製編輯

`s = "Hello,   world!" segment = s.strip().split(" ") print(segment)  # ['Hello,', '', '', 'world!'] print(len(segment))  # 4 (錯誤，應該是 2)`

#### **範例 2**

python

複製編輯

`s = "    "  # 只有空格，應該回傳 0 segment = s.strip().split(" ") print(segment)  # [''] print(len(segment))  # 1 (錯誤，應該是 0)`

---

### **如何修正？**

#### **✅ 修正方法：使用 `split()`（不帶參數）**

python

複製編輯

``def count_segments(self, s: str) -> int:     return len(s.split())  # `split()` 會自動處理連續空格``

#### **修正後行為**

|測試輸入|`s.split(" ")` (錯誤)|`s.split()` (正確)|
|---|---|---|
|`"Hello, world!"`|`['Hello,', '', '', 'world!']`（錯誤計算 4）|`['Hello,', 'world!']`（正確計算 2）|
|`" "`|`['']`（錯誤計算 1）|`[]`（正確計算 0）|
|`"a b c"`|`['a', 'b', '', 'c']`（錯誤計算 4）|`['a', 'b', 'c']`（正確計算 3）|

---

### **時間與空間複雜度**

- **時間複雜度：`O(n)`**（`split()` 遍歷 `s` 並分割）
- **空間複雜度：`O(n)`**（`split()` 會建立一個 `list`）

---

### **結論**

- `split(" ")` **無法正確處理連續空格**，會產生額外的 `""`，導致 `len(segment)` 錯誤。
- `split()`（不帶參數）**自動忽略連續空格**，可正確計算段落數量，應使用此方法。

