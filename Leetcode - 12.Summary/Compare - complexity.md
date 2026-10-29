

|                                         |     |
| --------------------------------------- | --- |
| [[#### data structure list的complexity]] |     |
|                                         |     |
|                                         |     |
#### data structure list的complexity
```
以下是我在lintcode會用到解題的data structure list, 我想要列表整理個別的時間複雜度跟 空間複雜度. 所以這些複雜度有沒有固定的complexity, 還是在個別的data structure用不同的算法會有不同的complexity? 如果沒有請直接寫出對應的complexity, 如果按照不同的算法(譬如BFS,DFS...)有不同, 則把資料結構對應的經典算法列出並寫complexity. 之後針對每個complexity為什麼是這樣做詳細的解釋
```

|**資料結構 / 演算法**|**常見操作 / 應用場景**|**時間複雜度 (平均 / 最差)**|**空間複雜度**|
|---|---|---|---|
|**1. Array (陣列)**|索引存取<br><br>  <br><br>插入 / 刪除|O(1) / O(1)<br><br>  <br><br>O(n) / O(n)|O(n)|
|**1. PreSum (前綴和)**|陣列預處理 (Build)<br><br>  <br><br>區間求和查詢 (Query)|O(n)<br><br>  <br><br>O(1)|O(n)|
|**2. Linked / Doubly Linked List**|索引存取 / 搜尋<br><br>  <br><br>已知節點插入 / 刪除|O(n) / O(n)<br><br>  <br><br>O(1) / O(1)|O(n)|
|**3. Queue / Deque**|頭尾加入 / 移除 (Push/Pop)|O(1) / O(1)|O(n)|
|**3. Monotone Queue (單調佇列)**|維護滑動區間極值|均攤 O(1)|O(n)|
|**4. Stack (堆疊)**|頂部加入 / 移除|O(1) / O(1)|O(n)|
|**4. Monotone Stack (單調堆疊)**|尋找下一個更大/更小元素|均攤 O(1)|O(n)|
|**5. Heap (堆積)**|建堆 (Heapify)<br><br>  <br><br>推入 (Push) / 彈出 (Pop)|O(n)<br><br>  <br><br>O(log n)|O(n)|
|**6. Binary Tree (BT)**|DFS / BFS 遍歷全樹|O(n)|O(h) / O(w)|
|**6. Binary Search Tree (BST)**|搜尋 / 插入 / 刪除|O(log n) / O(n)|O(h)|
|**7. Hash Table / Map**|搜尋 / 插入 / 刪除|O(1) / O(n)|O(n)|
|**8. Trie (字典樹)**|字串搜尋 / 插入|O(L) (L為字串長度)|O(N × L)|
|**9. Union Find (並查集)**|Find / Union (含路徑壓縮)|均攤 O(1)|O(n)|
|**10. Segment Tree (線段樹)**|建樹 (Build)<br><br>  <br><br>區間查詢 / 單點修改|O(n)<br><br>  <br><br>O(log n)|O(n)|
|**11. Balanced Tree (平衡樹)**|搜尋 / 插入 / 刪除|嚴格 O(log n)|O(n)|
|**12. BIT (樹狀陣列)**|建樹 (Build)<br><br>  <br><br>前綴查詢 / 單點修改|O(n log n) 或 O(n)<br><br>  <br><br>O(log n)|O(n)|

## 複雜度詳細原理解析

### 1. Array 與 PreSum (前綴和)

- **Array**：
    
    - **時間 O(1) 存取**：陣列記憶體連續，透過「起始位址 + 索引 × 大小」可直接計算位址。
        
    - **時間 O(n) 插入/刪除**：中間插入會迫使後續所有元素搬移，最差需搬移 n 個元素。
        
- **PreSum**：
    
    - **時間 O(n) 建構**：遍歷一次原陣列，`prefix[i] = prefix[i-1] + nums[i]`。
        
    - **時間 O(1) 查詢**：求區間 `[i, j]` 總和只需計算 `prefix[j] - prefix[i-1]`，將原本 O(n) 的迴圈加總降維成一次減法。
        

### 2. Linked List 與 Doubly Linked List

- **時間 O(n) 搜尋**：記憶體不連續，必須從 Head 循著指標逐一尋找。
    
- **時間 O(1) 插入/刪除**：只需切斷並重新連結相鄰節點的指標，無需搬移記憶體。Doubly Linked List 額外維護 `prev` 指標，使得「刪除當前節點」時不需要從頭尋找它的前一個節點，操作更加純粹的 O(1)。
    

### 3. Queue, Deque 與 Monotone Queue

- **Queue / Deque 時間 O(1)**：僅在頭尾兩端操作指標，無論資料量多大都是常數時間。
    
- **Monotone Queue (單調佇列) 均攤 O(1)**：
    
    - 用於「滑動窗口最大/小值」。每次元素進入時，會將佇列尾端破壞單調性的元素全部踢出。
        
    - **為什麼是均攤 O(1)？** 雖然某一次 push 可能會觸發一個 `while` 迴圈踢出多個元素，但每個元素在整個生命週期中「最多進入佇列 1 次、離開佇列 1 次」。總操作次數為 2n，平攤給 n 個元素，單次操作為 O(1)。
        

### 4. Stack 與 Monotone Stack

- **Stack 時間 O(1)**：永遠只在頂端一側進行 push/pop，操作局部化。
    
- **Monotone Stack (單調堆疊) 均攤 O(1)**：
    
    - 用於解決「下一個更大/更小元素」問題。原理與單調佇列相同，每個元素最多 push 一次、pop 一次。即便某些特定步驟的迴圈執行較多次，整體均攤下來依然是 O(1)。
        

### 5. Heap (堆積 / 優先佇列)

- **時間 O(n) 建堆 (Heapify)**：由下而上執行 Sift-Down，底層節點多但下沉距離短，數學級數推導總和為 O(n)，比一個個 Push O(n log n) 快。
    
- **時間 O(log n) Push/Pop**：Heap 是一棵完全二元樹。插入或刪除最值時，需沿著樹高進行上浮 (Bubble-Up) 或下沉 (Sift-Down) 調整，樹高嚴格受限於 log n。
    

### 6. Binary Tree (BT) 與 Binary Search Tree (BST)

- **BT 遍歷時間 O(n)**：DFS (深度優先) 或 BFS (廣度優先) 都保證每個節點剛好被訪問 1 次。
    
- **BT 空間 O(h) 或 O(w)**：DFS 的 Call Stack 深度等於樹高 h；BFS 的 Queue 需容納最寬的一層，大小為最大寬度 w。
    
- **BST 搜尋/插入 O(log n)**：每次比較能淘汰一半子樹。但若資料有序插入，樹會退化成單鏈，時間退化為 O(n)。
    

### 7. Hash Table / Hash Map

- **時間 O(1) 平均**：透過雜湊函數 (Hash Function) 算出索引，直接寫入或讀取陣列位置。
    
- **時間 O(n) 最差**：當大量相異 Key 發生雜湊碰撞 (Hash Collision)，全擠在同一個位置形成鏈結串列時，搜尋退化為線性遍歷。
    

### 8. Trie (字典樹 / 前綴樹)

- **時間 O(L) 搜尋/插入**：L 為目標字串長度。找一個單字只需要往下走 L 層，與樹中存放了十萬還是一百萬個字無關。
    
- **空間 O(N × L)**：每個字元都是一個節點，且每個節點需維護一個大小為 26 (字母表) 的指標陣列。極度耗費記憶體，典型的空間換取時間。
    

### 9. Union Find (並查集)

- **時間均攤 O(1)**：
    
    - 使用「路徑壓縮 (Path Compression)」：尋找根節點的同時，將沿途所有節點直接連到根節點。
        
    - 使用「按秩合併 (Union by Rank)」：將較矮的樹掛到較高的樹下。
        
    - 兩者結合可將樹高壓平，複雜度降至反阿克曼函數 O(α(n))，在實際應用中不大於 4，視為 O(1)。
        

### 10. Segment Tree (線段樹)

- **時間 O(n) 建樹**：每個節點代表一個區間，總節點數約為 2n-1，遞迴建樹訪問每個節點一次。
    
- **時間 O(log n) 區間查詢 / 單點修改**：線段樹是完美/接近完美的二元樹。每次查詢或修改，頂多在每一層訪問 4 個節點，總訪問量正比於樹高 log n。
    
- **空間 O(n)**：實作上通常開 4n 大小的陣列來保證足夠裝下所有節點。
    

### 11. Balanced Tree (平衡二元搜尋樹：AVL / 紅黑樹)

- **時間嚴格 O(log n)**：與普通 BST 不同的地方在於，每次插入或刪除後，會透過「旋轉 (Rotation)」強制維持左右子樹的高度差在特定範圍內。這保證了樹不會退化成鏈結串列，最差情況依然是 O(log n)。在 LintCode 常用語言如 Java 的 `TreeMap` 或 C++ 的 `std::set` 底層即為紅黑樹。
    

### 12. BIT (Binary Indexed Tree / 樹狀陣列 / Fenwick Tree)

- **時間 O(log n) 前綴查詢 / 單點修改**：利用整數的二進位特徵 (透過 `lowbit(x) = x & -x`) 來跳躍區間。每次跳躍都會消去二進位中最右邊的 1，或加上最右邊的 1，最多跳躍次數等同於數字的位元數，即 O(log n)。
    
- **相較於線段樹的優勢**：空間極度節省，僅需長度為 n+1 的陣列（空間 O(n) 且常數極小），且程式碼極短；但缺點是僅能處理具備可減性 (如區間和) 的問題，無法像線段樹一樣輕易處理區間最大/最小值。
