
|     |     |
| --- | --- |
|     |     |

```
在這裡為什麼要使用defaultdict? 這跟python的dict的差別在哪裡? 如果prerequisites = [[1,0],[2,0],[3,1],[3,2]], 還是可以這樣寫嗎? for a, b in prerequisites
```
這兩個問題都問到了 Python 語法與資料結構的關鍵細節，以下為你逐一解答：

### 一、 為什麼要使用 `defaultdict`？它跟普通 `dict` 差在哪？

主要差別在於**當存取一個不存在的鍵（Key）時的處理方式**。

#### 1. 普通 `dict` 的行為：

如果你在普通字典中存取或 append 一個尚未存在的 Key，Python 會直接拋出 `KeyError` 錯誤。

Python

```
# 普通 dict
graph = {}

# 如果寫這行會報錯！ KeyError: 0
# 因為 graph[0] 還不存在，不能直接 append
graph[0].append(1)

# 普通 dict 必須這樣寫（先檢查是否存在）：
if 0 not in graph:
    graph[0] = []
graph[0].append(1)

# 或者用 setdefault：
graph.setdefault(0, []).append(1)
```

#### 2. `defaultdict(list)` 的行為：

`defaultdict` 來自 Python 標準庫的 `collections` 模組。當你試圖存取一個不存在的 Key 時，它會**自動呼叫帶入的型別（這裡指定為 `list`），自動建立一個預設值（空列表 `[]`）**，然後再進行操作。

Python

```
from collections import defaultdict

# defaultdict
graph = defaultdict(list)

# 即使 graph[0] 不存在，也會自動建立 0: []，然後順利 append
graph[0].append(1)  # 結果為 {0: [1]}，不會報錯！
```

#### 💡 總結：為什麼這裡用 `defaultdict`？

用 `defaultdict(list)` 可以**讓程式碼極度簡潔**，省去每次建圖時都要用 `if b not in graph:` 來判斷字典裡有沒有這個節點的贅語。

### 二、 如果 `prerequisites = [[1,0],[2,0],[3,1],[3,2]]`，還是可以寫 `for a, b in prerequisites` 嗎？

**完全可以！而且這是 Python 處理這種資料型態最標準、最優雅的寫法。**

#### 運作原理（解包 Unpacking）：

在 Python 中，當一個列表裡的元素也是列表或元組（且元素個數固定為 2 個）時，`for a, b in prerequisites` 會自動執行**解包（Unpacking）**：

在迴圈的每一輪中：

- 第 1 輪：`[1, 0]` $\rightarrow$ `a = 1`, `b = 0`
    
- 第 2 輪：`[2, 0]` $\rightarrow$ `a = 2`, `b = 0`
    
- 第 3 輪：`[3, 1]` $\rightarrow$ `a = 3`, `b = 1`
    
- 第 4 輪：`[4, 2]` $\rightarrow$ `a = 3`, `b = 2`
    

這等同於以下寫法，但更簡潔：

Python

```
# 效果完全相同的普通寫法
for pair in prerequisites:
    a = pair[0]
    b = pair[1]
    graph[b].append(a)
    indegree[a] += 1
```

所以直接寫 `for a, b in prerequisites` 是完全沒問題且非常推薦的寫法！