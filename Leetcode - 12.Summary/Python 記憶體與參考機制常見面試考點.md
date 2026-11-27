
|                                 |     |
| ------------------------------- | --- |
| [[#### Python 記憶體與參考機制的常見面試考點]] |     |
| [[#### python跟c++的差異記憶體與參考機制]]  |     |
|                                 |     |
#### Python 記憶體與參考機制的常見面試考點
```
請完整整理Python 記憶體與參考機制的常見面試考點, 並詳細解釋並具體舉例說明, 最後整理成表格
```
Python 的記憶體模型遵循「**一切皆物件（All objects on Heap）、變數皆參考（Variables are references）**」。所有 Python 面試中的記憶體考題，本質上都在測試兩個核心動作的區別：**「重新綁定指標（Rebinding）」**（讓變數指向新的 Heap 位址）與「原地修改狀態（Mutation）」（順著指標修改原本 Heap 物件的內部資料）。

以下整理 Python 面試最常考的 7 大記憶體與參考機制考點：

### 1. 函式傳參機制：Call by Object Reference（Pass by Assignment）

Python 既不是純粹的 Call by Value，也不是 Call by Reference，而是 **Call by Object Reference**。呼叫函式時，系統會複製一份「指向 Heap 物件的參考指標」傳給區域參數：

- 若參數指向**不可變物件**（`int`、`str`、`tuple`），函式內重新賦值只會讓區域變數改指新物件，外部不受影響。
    
- 若參數指向**可變物件**（`list`、`dict`、`set`），函式內執行原地修改方法（如 `.append()`）會直接改動 Heap 上的共用物件；但若用 `=` 重新賦值（如 `arr = [9, 9]`），則只會斷開區域變數的指標，外部依然指著舊物件。
    
```Python
def modify(num: int, arr: list) -> None:
    num += 100          # Rebinding：區域變數 num 改指新的整數物件 105
    arr.append(4)       # Mutation：順著指標修改 Heap 上的同一個 list
    arr = [9, 9, 9]     # Rebinding：區域變數 arr 改指新 list，外部變數不受影響

x = 5
items = [1, 2, 3]
modify(x, items)
print(x)      # 5
print(items)  # [1, 2, 3, 4]
```

### 2. 可變預設參數陷阱（Mutable Default Arguments）

Python 函式在 **`def` 定義被執行的那一刻**（而非每次呼叫時）就會計算預設參數的表達式，並把建立好的物件參考儲存在函式物件的 `__defaults__` 屬性中。若預設值是可變物件（如 `[]` 或 `{}`），所有未傳入該參數的呼叫都會共用同一個 Heap 物件。

```Python
# 錯誤寫法：預設 list 在定義時只建立一次，掛在 add_item.__defaults__ 上
def add_item_buggy(val, container=[]):
    container.append(val)
    return container

print(add_item_buggy(1))  # [1]
print(add_item_buggy(2))  # [1, 2]（前一次的資料殘留！）

# 正確寫法：使用不可變的 None 作為哨兵值（Sentinel）
def add_item_safe(val, container=None):
    if container is None:
        container = []    # 每次呼叫都在 Heap 配置全新的 list
    container.append(val)
    return container
```

### 3. 賦值、淺拷貝（Shallow Copy）與深拷貝（Deep Copy）

在處理巢狀結構（如二維陣列、圖的鄰接表）時，三者在 Heap 上的物件配置完全不同：

- **直接賦值（`b = a`）**：完全不建立新物件，只是多一個變數指向同一個 Heap 位址。
    
- **淺拷貝（`a.copy()`、`a[:]`、`copy.copy(a)`）**：在 Heap 配置一個**新的外層容器**，但容器裡填入的依然是**原本內層子物件的參考指標**。
    
- **深拷貝（`copy.deepcopy(a)`）**：遞迴複製外層容器與所有層級的內層子物件，兩者記憶體完全隔離（內部透過 `memo` 字典記錄已複製的 `id`，防止循環參考導致無窮遞迴）。
    
```Python
import copy

origin = [[1, 2], [3, 4]]
alias = origin
shallow = copy.copy(origin)       # 或 origin[:]
deep = copy.deepcopy(origin)

origin[0][0] = 99
origin.append([5, 6])

print(alias)    # [[99, 2], [3, 4], [5, 6]] -> 外層與內層全同步
print(shallow)  # [[99, 2], [3, 4]]         -> 外層獨立（無 [5, 6]），但內層仍共用（99）
print(deep)     # [[1, 2], [3, 4]]          -> 內外層完全獨立
```

### 4. 身份比較 `is` vs. 值比較 `==`（含小整數快取池與字串駐留）

- **`==`（Equality）**：呼叫物件的 `__eq__()` 方法，比較兩者的**邏輯數值**是否相等。
    
- **`is`（Identity）**：直接比較兩者的 **Heap 記憶體位址**（等同於 `id(a) == id(b)`）。
    

CPython 為了節省頻繁建立小物件的效能開銷，內建了兩種快取機制：

1. **小整數快取池（Small Integer Caching）**：直譯器啟動時預先在 Heap 建立 **`-5` 到 `256`** 的整數物件，所有該範圍內的變數都指向同一個快取位址。
    
2. **字串駐留（String Interning）**：符合識別字規則（僅含英數字與底線）的短字串會被放入駐留池共用記憶體位址；包含空白或特殊符號的字串通常不會自動駐留。
    
```Python
a = 256
b = 256
print(a is b)  # True（命中 [-5, 256] 小整數快取池）

x = int("257")
y = int("257")
print(x == y)  # True（數值相等）
print(x is y)  # False（超出快取範圍，各自在 Heap 配置不同物件）

# 面試鐵律：檢查 None 永遠用 `is None`，因為 None 是全局唯一單例物件
```

_(註：若在同一函式或同一行寫 `x = 257; y = 257`，Python 編譯器的常數折疊優化會讓兩者共用同一個 `co_consts` 參考而回傳 `True`，因此測試動態配置需用 `int("257")` 或執行期運算。)_

### 5. 增強賦值運算子 `+=` 與 Tuple 內含可變物件陷阱

`x += y` 的底層執行邏輯是優先呼叫 `__iadd__()`（原地修改）；若物件沒有實作 `__iadd__`（例如不可變物件），才退而求其次呼叫 `__add__()`（建立新物件並重新綁定指標）。

這會引發一道經典面試陷阱——**在 `tuple` 裡對 `list` 使用 `+=`**：

```Python
t = (1, [2, 3])
try:
    t[1] += [4]
except TypeError as e:
    print(e)  # 'tuple' object does not support item assignment

print(t)      # (1, [2, 3, 4]) -> 雖然報錯了，但 list 已經被修改成功！
```

**底層拆解**：`t[1] += [4]` 等價於以下兩步：

1. 先執行 `temp = t[1].__iadd__([4])`，這一步直接**原地修改**了 Heap 上的串列，使其變成 `[2, 3, 4]`。
    
2. 接著執行賦值 `t[1] = temp`，試圖改寫 `tuple` 的指標槽位，此時才觸發 `TypeError`；但第一步的記憶體修改已經發生且無法自動回滾。
    

### 6. 閉包延遲綁定（Closure Late Binding）

Python 的 `for` 迴圈不會建立獨立的區塊作用域（Block Scope），迴圈變數 `i` 始終是外層作用域中的同一個變數槽位（`Cell` 物件）。在迴圈內建立的閉包（如 `lambda`）並不會在定義時拷貝 `i` 的數值，而是保存對該 `Cell` 物件的參考，等到**真正被呼叫時**才去讀取 `i` 的最終值。

```Python
# 錯誤寫法：三個 lambda 都參考同一個變數 i，迴圈結束後 i 為 2
funcs_buggy = [lambda: i for i in range(3)]
print([f() for f in funcs_buggy])  # [2, 2, 2]

# 正確寫法：利用預設參數在每次迴圈定義函式時，立刻綁定當下的整數物件參考
funcs_safe = [lambda i=i: i for i in range(3)]
print([f() for f in funcs_safe])   # [0, 1, 2]
```

### 7. 垃圾回收機制（GC）與循環參考（Circular Reference）

CPython 採用「**引用計數為主、分代回收為輔**」的雙層記憶體回收架構：

1. **引用計數（Reference Counting）**：每個 Heap 物件標頭都維護一個 `ob_refcnt` 計數器。每多一個變數指向它就 `+1`，參考離開作用域或 `del` 就 `-1`。一旦歸零，記憶體**立刻釋放**（無需等待 GC 週期）。
    
2. **分代垃圾回收（Generational GC，Gen 0 / 1 / 2）**：引用計數無法處理**循環參考**（例如雙向鏈結串列或父子節點互相指向，導致兩者 `ob_refcnt` 永遠 $\ge 1$）。分代回收器會定期掃描容器物件（`list`、`dict`、自定義類別實例），找出與外部斷開連結的循環群組並強制回收。
    

Python

```
import gc

class Node:
    def __init__(self):
        self.peer = None

a = Node()
b = Node()
a.peer = b
b.peer = a  # 形成循環參考

del a, b    # 外部變數已刪除，但兩個 Node 互相指著對方，引用計數仍為 1
gc.collect()  # 觸發分代回收器偵測循環參考並釋放 Heap 記憶體
```

### Python 記憶體與參考機制考點總表

| **面試考點**           | **底層記憶體與參考原理**                                | **常見踩坑寫法**                                           | **正確做法與面試結論**                                       |
| ------------------ | --------------------------------------------- | ---------------------------------------------------- | --------------------------------------------------- |
| **函式參數傳遞**         | **Call by Object Reference**，傳遞 Heap 物件指標的副本  | 以為傳入 `int` 可在函式內修改外部值，或傳入 `list` 後用 `=` 重新賦值         | 原地修改（`.append`）會影響外部；重新綁定（`=`）只影響區域變數               |
| **可變預設參數**         | 預設值在 `def` 定義時評估一次並存於 `func.__defaults__`     | `def f(x, arr=[]):` 導致多次呼叫共用同一個 Heap 串列              | 改用 `arr=None`，在函式內部判斷 `if arr is None: arr = []`    |
| **二維陣列初始化**        | `list * N` 僅複製內層物件的記憶體位址（Reference）           | `grid = [[0] * 3] * 3` 導致所有 Row 指向同一個內層 `list`       | 使用 List Comprehension：`[[0] * 3 for _ in range(3)]` |
| **淺拷貝 vs. 深拷貝**    | 淺拷貝僅複製外層容器並共用內層指標；深拷貝遞迴複製全層級                  | 用 `b = a[:]` 複製巢狀 `list`/`dict` 後修改內層元素              | 單層容器用 `.copy()`；多層巢狀可變結構使用 `copy.deepcopy()`        |
| **`is` 與 `==` 比較** | `is` 比較 Heap 記憶體位址（`id`）；`==` 比較邏輯值（`__eq__`） | 用 `x is 1000` 判斷數值相等（超出 `-5 ~ 256` 快取池會失敗）           | 比較數值/字串一律用 `==`；僅在比對單例（如 `is None`）時用 `is`          |
| **增強賦值 `+=`**      | 可變物件呼叫 `__iadd__` 原地修改（`id` 不變）；不可變物件建新物件     | `t = (1, [2]); t[1] += [3]` 拋出 `TypeError` 但內部串列已被修改 | 避免在 `tuple` 內存放可變物件並進行原地修改操作                        |
| **閉包延遲綁定**         | 閉包保存外層變數的 `Cell` 參考，呼叫時才解析最終數值                | `[lambda: i for i in range(3)]` 全部回傳最後一圈的 `2`        | 透過預設參數捕獲當下值：`[lambda i=i: i for i in range(3)]`     |
| **記憶體回收機制**        | **引用計數**（即時釋放）搭配 **分代 GC**（專治循環參考）            | 在高頻建立節點的類別中產生循環參考且覆寫 `__del__` 導致洩漏                  | 長期快取或父子雙向指標改用 `weakref`（弱參考）打破循環                    |




#### python跟c++的差異記憶體與參考機制
```
請幫我再做一個表格比對Python 記憶體與參考機制考點總表的8個面試考點, 比對python跟c++的差異. 並之後詳細解釋
```

|**面試考點**|**Python 機制（一切皆 Heap 物件參考）**|**C++ 機制（值語義為主、顯式指標與參考）**|
|---|---|---|
|**1. 函式參數傳遞**|**Call by Object Reference**：傳遞 Heap 指標副本；不可變物件無法改外部，可變物件可原地修改。|預設 **Call by Value**（完整拷貝 Stack/Heap 資料）；需顯式宣告 **Reference (`&`)** 或 **Pointer (`*`)** 才能修改外部變數。|
|**2. 可變預設參數**|預設參數在 `def` **定義時評估一次**並存於 `__defaults__`，多次呼叫共用同一個 Heap 物件。|預設參數在**每次函式呼叫時**於該次 Call Stack 重新建構，每次呼叫皆為全新獨立物件。|
|**3. 二維陣列初始化**|`[[0] * 3] * 3` 僅複製內層 `list` 的 Heap 位址指標，導致整欄連動；需用 List Comprehension。|`int grid[3][3] = {0}` 或 `vector<vector<int>>` 皆採取**值拷貝**，每一列配置各自獨立的記憶體空間。|
|**4. 淺拷貝 vs. 深拷貝**|`=` 為別名共用；`.copy()` 複製外層並共用內層指標；需用 `copy.deepcopy()` 遞迴複製全層級。|標準容器（如 `vector`、`string`）的複製建構子與賦值預設即為**深拷貝**；僅在容器儲存裸指標（`Node*`）時才是淺拷貝。|
|**5. 身份 vs. 值比較**|`is` 比對 Heap 位址（內建 `-5 ~ 256` 小整數快取池）；`==` 呼叫 `__eq__` 比對邏輯值。|`==` 預設比對數值（自定義類別需重載 `operator==`）；比對位址需顯式取址 `&a == &b`，無小整數快取池。|
|**6. 增強賦值 `+=` 與不可變性**|`tuple` 僅在執行期保護頂層指標，內部 `list` 執行 `+=` 會先原地修改成功才拋出 `TypeError`。|不可變性由**編譯期 `const`** 嚴格把關，`const` 容器內的元素完全無法呼叫任何修改方法（直接編譯失敗）。|
|**7. 閉包與變數捕獲**|**延遲綁定（Late Binding）**，閉包保存外層變數的 `Cell` 參考，呼叫時才讀取迴圈最終值。|Lambda 必須**顯式宣告捕獲清單**：按值 `[i]` 建立當下即拷貝；按參考 `[&i]` 共享變數（離開作用域會變懸空參考）。|
|**8. 記憶體回收與循環參考**|**引用計數（即時釋放）＋ 分代 GC（自動掃描回收循環參考）**。|**RAII 機制**（Stack 離開作用域瞬間解構）；Heap 使用 `shared_ptr` 發生循環參考**不會自動回收**，需用 `weak_ptr` 打破。|

### 1. 函式參數傳遞：隱式指標副本 vs. 顯式值／參考語義

- **Python**：所有參數傳遞都是複製「指向 Heap 物件的參考指標」。傳入 `list` 時，呼叫 `.append()` 會改動外部物件，但用 `=` 重新賦值只會斷開區域變數的指標。
    
- **C++**：預設採取 **Call by Value**，傳入 `std::vector<int>` 會把整個陣列在 Heap 上完整拷貝一份（時間與空間成本極高）。因此 C++ 面試中寫 DFS/BFS 時，圖或網格必須顯式加上 `&` 改為 **Call by Reference**（如 `vector<vector<int>>& grid`），才能避免複製開銷並直接修改原資料。
    

C++

```
// C++：必須顯式加上 & 才能避免整份拷貝並修改外部變數
void modify(int by_val, int& by_ref, std::vector<int>& arr) {
    by_val += 100;      // 只改到 Stack 上的副本，外部不變
    by_ref += 100;      // 直接修改外部變數的記憶體位址
    arr.push_back(4);   // 直接修改外部 vector
}
```

### 2. 可變預設參數：定義期求值 vs. 呼叫期求值

- **Python**：`def func(arr=[]):` 在解析函式定義時就建立好唯一的 `list` 物件並綁定在函式屬性上，導致後續每次呼叫都累積前一次的殘留資料。
    
- **C++**：預設參數的表達式是在每次呼叫函式當下（Call Site）才進行求值與建構。每次呼叫都會在當次 Call Stack 上重新建立一個全新的區域物件，函式結束即銷毀，完全不會殘留。
    

C++

```
// C++：每次呼叫若未傳入 container，都會重新建構一個全新的空 vector
std::vector<int> add_item(int val, std::vector<int> container = {}) {
    container.push_back(val);
    return container;  // 永遠只回傳 [val]，不會累積歷史資料
}
```

### 3. 二維陣列初始化：指標複製 vs. 獨立值配置

- **Python**：`[[0] * 3] * 3` 複製的是指向同一個內層 `list` 的記憶體位址，修改 `grid[0][1]` 會讓三列同時改變。
    
- **C++**：無論是配置在 Stack 上的原生二維陣列 `int grid[3][3] = {0};`，或是配置在 Heap 上的 `std::vector<std::vector<int>> grid(3, std::vector<int>(3, 0));`，底層都是透過**值拷貝**獨立配置 3 塊互不重疊的記憶體區塊，修改一格絕不影響其他列。
    

### 4. 淺拷貝 vs. 深拷貝：指標複製 vs. 預設深拷貝

- **Python**：`b = a` 只是建立別名（Aliasing）；`b = a.copy()` 只複製最外層容器，內層子串列依然共用位址，必須主動呼叫 `copy.deepcopy(a)` 才能徹底隔離。
    
- **C++**：C++ 的標準容器遵循「值語義」，執行 `std::vector<std::vector<int>> b = a;` 時，會自動觸發 `std::vector` 的**複製建構子（Copy Constructor）**，連同內部每一層 `vector<int>` 全部配置新記憶體並複製內容，**天生就是深拷貝**。只有在容器裡刻意存放裸指標（如 `std::vector<Node*> b = a;`）時，才會出現指標位址共用的淺拷貝現象。
    

### 5. 身份比較 vs. 值比較：快取單例 vs. 獨立記憶體槽位

- **Python**：`a = 256; b = 256` 會命中 CPython 的小整數快取池（`-5` 到 `256`），導致 `a is b` 為 `True`。
    
- **C++**：基本型別直接存放在 Call Stack 的獨立記憶體槽位中，沒有小整數快取池的概念。
    

C++

```
int a = 256;
int b = 256;
// a == b 為 true（數值相等）
// &a == &b 永遠為 false（兩者在 Call Stack 上擁有各自獨立的記憶體位址）
```

### 6. 增強賦值 `+=` 與不可變性：執行期半途報錯 vs. 編譯期嚴格攔截

- **Python**：`t = (1, [2, 3]); t[1] += [4]` 會先透過 `__iadd__` 成功修改 Heap 上的 `list`，接著嘗試寫回 `tuple` 時才在執行期拋出 `TypeError`，造成資料處於「已修改卻報錯」的不一致狀態。
    
- **C++**：C++ 的不可變性由編譯器透過 `const` 關鍵字進行**傳遞性檢查（Transitive Constness）**。一旦外層容器被宣告為 `const`，其內部所有成員在編譯期也自動視為 `const`，任何試圖原地修改的寫法都會被編譯器直接拒絕，絕對不會發生「改一半才報錯」的情況。
    

C++

```
const std::pair<int, std::vector<int>> t = {1, {2, 3}};
// t.second.push_back(4);  // 編譯錯誤 (Compile Error)：無法對 const 物件呼叫非 const 方法
```

### 7. 閉包延遲綁定 vs. C++ Lambda 捕獲清單

- **Python**：在迴圈中建立的 `lambda` 會保存外層變數的 `Cell` 參考，迴圈結束後所有函式讀到的都是最後一圈的數值（延遲綁定）。
    
- **C++**：Lambda 必須在 `[]` 中明確指定捕獲方式，將「按值拷貝」與「按參考共享」徹底分開：
    
    - **按值捕獲 `[i]`**：在建立 Lambda 的當下立刻將 `i` 的數值拷貝進 Lambda 物件內部，迴圈後續變化完全不影響它。
        
    - **按參考捕獲 `[&i]`**：與 Python 類似共享同一個變數，但若該 Lambda 被回傳到迴圈外部執行，由於原本的 Stack 變數 `i` 已被銷毀，會直接引發嚴重的 **Dangling Reference（懸空參考／未定義行為）**，而 Python 則會因為 Heap 上的 `Cell` 仍被參考而安全存活。
        

C++

```
std::vector<std::function<int()>> funcs;
for (int i = 0; i < 3; ++i) {
    funcs.push_back([i]() { return i; });  // [i] 按值捕獲：依序回傳 0, 1, 2
}
```

### 8. 記憶體回收與循環參考：分代 GC 兜底 vs. `weak_ptr` 手動破環

- **Python**：引用計數（Reference Counting）在循環參考時會失效，但 Python 內建**分代垃圾回收器（Generational GC）**，會在背景定期掃描容器物件並自動回收孤立的循環參考群組。
    
- **C++**：C++ **沒有垃圾回收器（No GC）**。Heap 記憶體通常使用智慧指標 `std::shared_ptr`（底層同樣是引用計數）管理。如果兩個節點用 `std::shared_ptr` 互相指向對方，即使外部指標全部離開作用域，兩者的引用計數依然維持為 `1`，導致**永久性記憶體洩漏（Memory Leak）**。工程上必須將其中一方改為不增加引用計數的 **`std::weak_ptr`** 來打破循環。
    

C++

```
struct Node {
    std::shared_ptr<Node> next;
    std::weak_ptr<Node> prev;  // 必須使用 weak_ptr 指向父節點／前驅節點，否則會發生 Memory Leak
};
```