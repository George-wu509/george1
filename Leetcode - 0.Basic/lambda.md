
|                          |     |
| ------------------------ | --- |
| [[#### lambda用在sort的用法]] |     |
|                          |     |
|                          |     |
#### lambda用在sort的用法
```
能否詳細介紹lambda的用法並用具體例子舉例, 所以如果intervals = [[1, 2], [2, 3], [3, 4], [1, 3]], intervals.sort(key=lambda x:x[1])裡面的x是否就是[1,2], [2,3]這類的在list裡面的子list? 有沒有可能是其他的或是怎麼決定的? 所以也可以寫成 intervals.sort(key=lambda s:s[1])? 那可以這樣寫嗎 intervals.sort(key=lambda x^2:x[1])還是lambda之後變數只能放變數不能類似這樣x^2, 使用lambda還有甚麼規則?
```
##### 1. 直接回答你的三個核心問題

#### Q1：`intervals.sort(key=lambda x: x[1])` 裡面的 `x` 是子 list 嗎？怎麼決定的？

**是的！`x` 就是 `intervals` 裡面的每一個子 list（例如 `[1, 2]`、`[2, 3]`）。**

`x` 是什麼，完全由「外層容器（`intervals`）裡面裝了什麼元素」決定。`sort()` 在排序時，會把 `intervals` 最外層的元素一個一個拿出來，依序塞進 `lambda` 的變數裡：

- 如果 `intervals = [[1, 2], [2, 3]]`（list 裡面裝 list） $\rightarrow$ 拿出來的一筆元素就是 `[1, 2]`，所以 `x` 是 `list`，`x[1]` 就是 `2`。
    
- 如果 `words = ["apple", "banana", "kiwi"]`（list 裡面裝字串） $\rightarrow$ 拿出來的一筆元素就是 `"apple"`，所以 `x` 是 `str`，此時寫 `lambda x: len(x)` 就是依字串長度排序。
    
- 如果 `users = [{"age": 30}, {"age": 18}]`（list 裡面裝字典） $\rightarrow$ 拿出來的一筆元素就是 `{"age": 30}`，所以 `x` 是 `dict`，此時寫 `lambda x: x["age"]` 就是依年齡排序。
    

#### Q2：可以寫成 `intervals.sort(key=lambda s: s[1])` 嗎？

**完全可以！** `x` 或 `s` 只是一個「參數名稱（代號）」，就像你在寫一般函式 `def func(s): return s[1]` 一樣，只要冒號左邊跟右邊用的是同一個名字即可。寫 `lambda interval: interval[1]` 甚至可讀性更高。

#### Q3：可以寫成 `intervals.sort(key=lambda x^2: x[1])` 嗎？

**絕對不行，會直接語法錯誤（`SyntaxError`）！**

原因有兩個：

1. **冒號左邊只能放「合法的變數名稱」（參數清單）**：左邊是「接收資料的盒子」，不能放計算公式（`x^2` 是運算，不是變數名字）。計算公式必須寫在**冒號右邊**（例如 `lambda x: x ** 2`，另外提醒在 Python 中次方是 `**`，`^` 是位元互斥或 XOR 運算）。
    
2. **冒號左右兩邊名稱必須對應**：就算假設左邊能寫 `x^2`，右邊寫 `x[1]` 時 Python 也會找不到 `x` 是誰。
    

### 2. `lambda` 的完整語法與 5 大核心規則

`lambda` 的設計初衷是「用完即丟的微型函式」。它的標準長相只有一種：

Python

```
lambda 參數1, 參數2, ... : 單一運算式（自動回傳這個結果）
```

使用時必須遵守以下 **5 條規則**：

1. **不能寫 `return`**：冒號右邊算出來的值會**自動回傳**，寫了 `return` 反而會語法錯誤。
    
2. **只能寫「單一運算式（Expression）」**：不能換行，也不能寫 `for` 迴圈、`while` 迴圈、`print()` 賦值語句（如 `x = 1`）。
    
3. **可以支援多參數或無參數**：
    
    - 零個參數：`say_hi = lambda: "Hello!"`（呼叫 `say_hi()` 會得到 `"Hello!"`）
        
    - 多個參數：`add = lambda a, b: a + b`（呼叫 `add(3, 5)` 會得到 `8`）
        
4. **條件判斷必須寫成「單行三元運算式」**： 雖然不能寫多行的 `if...else:` 區塊，但可以寫成一行 `值A if 條件 else 值B`：
    
    Python
    
    ```
    # 如果是偶數就保留原值，奇數就乘以 2
    calc = lambda x: x if x % 2 == 0 else x * 2
    ```
    
5. **通常不建議把 `lambda` 存進變數**：像上面 `add = lambda a, b: a + b` 雖然語法合法，但實務上如果需要重複呼叫一個有名字的函式，直接寫 `def add(a, b): return a + b` 比較符合 Python 規範（PEP 8）。`lambda` 最適合直接塞進其他函式的參數裡。
    

### 3. `lambda` 的 4 個具體實戰例子

除了演算法刷題，日常寫 Python 處理資料時，以下四種場景最常出現 `lambda`：

#### 例子 1：對複雜資料（字典或物件）進行排序

假設有一份商品清單，你想依照「價格由低到高」排序：

Python

```
products = [
    {"name": "滑鼠", "price": 800},
    {"name": "鍵盤", "price": 2500},
    {"name": "螢幕擦拭布", "price": 150}
]

# p 代表列表裡的每一個字典元素
products.sort(key=lambda p: p["price"])
# 結果順序：螢幕擦拭布(150) -> 滑鼠(800) -> 鍵盤(2500)
```

#### 例子 2：多條件排序（回傳 Tuple）

假設你在做成績單排名，規則是「總分高到低排；若總分相同，則依座號小到大排」：

Python

```
# 格式：(座號, 總分)
scores = [(3, 90), (1, 95), (2, 90)]

# 用負號 -s[1] 讓分數變成「由大到小」，s[0] 維持座號「由小到大」
scores.sort(key=lambda s: (-s[1], s[0]))
# 結果：[(1, 95), (2, 90), (3, 90)]
```

#### 例子 3：在 `defaultdict` 中設定預設結構（刷題與計數神器）

在建立圖論（Graph）或分組資料時，`defaultdict` 需要傳入一個「工廠函式」來決定預設值：

Python

```
from collections import defaultdict

# 當查詢一個不存在的 key 時，自動建立預設值 [0, "未命名"]
records = defaultdict(lambda: [0, "未命名"])

records["A01"][0] += 10
print(records["A01"])  # 輸出：[10, '未命名']
```

#### 例子 4：搭配 `max()` / `min()` 找特定條件的極值

不一定要整份列表排序（排序要花 $O(N \log N)$ 時間），如果只是要找「最長的名字」，直接用 `max()` 搭配 `lambda` 只要 $O(N)$ 時間：

Python

```
names = ["Bob", "Alexander", "Mia"]

longest_name = max(names, key=lambda name: len(name))
print(longest_name)  # 輸出：Alexander
```