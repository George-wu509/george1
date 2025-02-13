32
给定两个字符串 `source` 和 `target`. 求 `source` 中最短的包含 `target` 中每一个字符的子串.

Example
样例 1：
输入：
source = "abc"
target = "ac"
输出：
"abc"
解释：
"abc" 是 source 的包含 target 的每一个字符的最短的子串。

样例 2：
输入：
source = "adobecodebanc"
target = "abc"
输出：
"banc"
解释：
"banc" 是 source 的包含 target 的每一个字符的最短的子串。

样例 3：
输入：
source = "abc"
target = "aa"
输出：
""
解释：
没有子串包含两个 'a'。


```python
from collections import defaultdict
class Solution:
    """
    @param source : A string
    @param target: A string
    @return: A string denote the minimum window, return "" if there is no such a string
    """
    def minWindow(self, source , target):
        # 初始化counter_s和counter_t
        counter_s = defaultdict(int)
        counter_t = defaultdict(int)
        for ch in target:
            counter_t[ch] += 1
        left = 0
        valid = 0
        # 记录最小覆盖子串的起始索引及长度
        start = -1
        minlen = float('inf')
        for right in range(len(source)):
            # 移动右边界, ch 是将移入窗口的字符
            ch = source[right]
            if ch in counter_t:
                counter_s[ch] += 1
                if counter_s[ch] == counter_t[ch]:
                    valid += 1
            
            # 判断左侧窗口是否要收缩
            while valid == len(counter_t):
                # 更新最小覆盖子串
                if right - left < minlen:
                    minlen = right - left
                    start = left
                # left_ch 是将移出窗口的字符
                left_ch = source[left]
                # 左移窗口
                left += 1
                # 进行窗口内数据的一系列更新
                if left_ch in counter_s:
                    counter_s[left_ch] -= 1
                    if counter_s[left_ch] < counter_t[left_ch]:
                        valid -= 1
        # 返回最小覆盖子串
        if start == -1:
            return ""
        return source[start: start + minlen + 1]
```
pass

解說
source = "adobecodebanc"
target = "abc"

Step1
建立兩個dict,  counter_s一個準備儲存source內子字串, 另一個counter_t則是將target轉成dict = {"a":1, "b":1, "c":1}

Step2
雙指針Left=0 and Right=0. Right一步步往右移, 並將字元加入counter_s. 並跟counter_t 比較. 如果Right所在的字元也在counter_t內, 則valid +1. 當valid = 3時代表目前Left到Right的子字串有target所有字元. 
"adobecodebanc"
 L=0    R=5

Step3
接下來開始移動Left指針往右移並跟counter_t比較試圖找到更短的子字串. 如果沒有則換成Right指針往右移, 回到step2


## **一般 `dict` 差異比較**

在 Python 中，`collections.Counter`、`collections.defaultdict` 和一般的 `dict` 都是字典類型，但它們有不同的用途與特性。下面將詳細分析三者的區別，並提供具體範例。

---

## **功能比較**

|特性|`Counter`|`defaultdict`|一般 `dict`|
|---|---|---|---|
|**主要用途**|計算元素出現次數|設定默認值的字典|一般鍵值對存儲|
|**鍵不存在時行為**|返回 `0`（適用於計數）|自動創建並初始化鍵|觸發 `KeyError`|
|**自動初始化值**|`int`（默認為 0）|由 `default_factory` 指定|無默認值|
|**支持運算**|支持加法、減法、交集、差集等運算|不支持|不支持|
|**適用場景**|統計、頻率分析|分組、累加、計數等需要默認值的應用|普通鍵值存儲|

---

## **詳細比較與舉例**

### **1️⃣ `collections.Counter`**

- `Counter` 是 **專門用來計算元素出現次數** 的字典。
- **鍵是元素，值是該元素的計數**，當鍵不存在時，預設值為 `0`，不會報 `KeyError`。
- **支持數學運算**（加法、減法、交集、差集）。
- 適用於 **字母頻率統計、單詞出現次數統計、數據去重分析** 等場景。

#### **範例**
```python
"""
from collections import Counter

# 計算字母出現次數
word = "banana"
cnt = Counter(word)

print(cnt)  # {'b': 1, 'a': 3, 'n': 2}

# 取得某個字符的計數
print(cnt['a'])  # 3
print(cnt['z'])  # 0（不存在的鍵預設為 0）

# 支持減法（減去另一個 Counter）
cnt2 = Counter("band")
print(cnt - cnt2)  # {'a': 2, 'n': 1}

```

**特性** ✅ 自動初始化鍵的值為 `0`  
✅ **支持數學運算** (`+`, `-`, `&`, `|`)  
✅ 適合 **頻率計算、文本分析**

---

### **2️⃣ `collections.defaultdict`**

- `defaultdict` 是 **普通字典的增強版**，允許**為不存在的鍵提供默認值**。
- 需要提供一個 **`default_factory`**（例如 `int`, `list`, `set`），當鍵不存在時，自動創建該鍵並賦予默認值，而不會拋出 `KeyError`。
- 適用於 **列表分組、累加計算、動態創建鍵值對** 等場景。

#### **範例**
```python
"""
from collections import defaultdict

# 默認值為 0 的 defaultdict
dd = defaultdict(int)
dd['a'] += 1  # 'a' 不存在時，自動設為 0，然後 +1
print(dd)  # {'a': 1}

# 默認值為 list 的 defaultdict
dd_list = defaultdict(list)
dd_list['a'].append(10)  # 'a' 不存在時，自動創建空列表，然後加入 10
print(dd_list)  # {'a': [10]}

# 默認值為 set
dd_set = defaultdict(set)
dd_set['a'].add(10)  # 'a' 不存在時，自動創建空集合
print(dd_set)  # {'a': {10}}

```

**特性** ✅ **當鍵不存在時，自動創建值**（不會 `KeyError`）  
✅ **適用於分組、累加、關聯數據存儲**（例如 `list`、`set`）  
✅ 比一般 `dict` 更靈活

---

### **3️⃣ 一般 `dict`**

- `dict` 是 Python 中最基本的鍵值對存儲結構。
- 當嘗試訪問 **不存在的鍵** 時，會拋出 `KeyError`，不像 `Counter` 和 `defaultdict` 能提供默認值。
- 適用於 **一般的鍵值對存儲，不需要特殊默認行為**。

#### **範例**

python

複製編輯

`d = {}  # 嘗試訪問不存在的鍵會報錯 print(d.get('a', 0))  # 0 （用 get() 方法可以提供默認值） print(d['a'])  # KeyError: 'a'`

**特性** ✅ **標準的鍵值對存儲**  
❌ **不存在的鍵會拋出 `KeyError`**  
✅ **適用於一般用途，但不適合計數或自動初始化值的場景**

---

## **綜合比較表**

||`Counter`|`defaultdict`|`dict`|
|---|---|---|---|
|**用途**|計數、頻率統計|自動初始化鍵值|一般鍵值存儲|
|**鍵不存在時行為**|返回 `0`|創建並初始化|拋出 `KeyError`|
|**是否需要 `default_factory`**|否|是|否|
|**支持數學運算 (`+`, `-`, `&`, `|`)**|是|否|
|**適用場景**|統計字母數量、詞頻分析|分組存儲、避免 `KeyError`|通用存儲|

---

## **選擇哪種結構？**

|使用場景|推薦數據結構|
|---|---|
|**計算字母、單詞頻率**|`Counter`|
|**需要自動創建 `list`、`set` 來分組**|`defaultdict(list/set)`|
|**普通鍵值存儲（沒有特殊需求）**|`dict`|
|**減法、交集、加法等運算**|`Counter`|

---

## **結論**

- **`Counter`** 適用於 **計數場景**，如字母頻率、詞頻分析，並支持 **數學運算 (`+`, `-`, `&`, `|`)**。
- **`defaultdict`** 適用於 **動態創建鍵值對**，如 **分組存儲、列表累加**，可以避免 `KeyError`。
- **一般 `dict`** 適合 **普通的鍵值存儲**，但當鍵不存在時會拋 `KeyError`，不像 `Counter` 和 `defaultdict` 有自動初始化機制。