
|                                                   |                                                                                        |
| ------------------------------------------------- | -------------------------------------------------------------------------------------- |
| [[#### 一、 基礎線性與指標操作 (Linear & Pointers)]]         | Array<br>Linked list<br>Doubly Linked List<br>Two Pointers                             |
| [[#### 二、 堆疊、佇列與雜湊映射 (Stacks, Queues & Hashing)]] | Stack<br>Monotonic Stack<br>Queue / Deque<br>Monotonic Queue<br>Hash Table / Map / Set |
| [[#### 三、 區間計算與動態查詢 (Ranges & Dynamic Queries)]]  | Prefix Sum<br>Binary Indexed Tree(BIT)<br>Segment Tree                                 |
| [[#### 四、 樹狀與圖論高階結構 (Trees & Graphs)]]            | Heap<br>Binary Tree<br>BST<br>Trie<br>Union Find<br>Balanced Tree                      |
| [[#### 五、 核心決策演算法 (Core Algorithmic Paradigms)]]  | Sorting<br>Binary Search<br>Dynamic Programming                                        |
| [[#### 高階選擇心法與判定規則 (How to Choose)]]              |                                                                                        |
|                                                   |                                                                                        |

#### 一、 基礎線性與指標操作 (Linear & Pointers)
```
基礎線性與指標操作 (Linear & Pointers)
```
Array
Linked list
Doubly Linked List
Two Pointers

| **資料結構 / 算法**                | **核心特性與運作原理**                                            | **判定場景：當你看到題目出現以下特徵時使用**                                                                                                                                        | **經典 LeetCode 題目代表**                                                                                                                                           |
| ---------------------------- | -------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Array<br>(陣列)                | 記憶體連續，提供 $O(1)$ 的極速索引。缺點是插入與刪除需要搬動大量元素。                  | 1. 題目要求空間複雜度必須是 $O(1)$ (原地修改，In-place)。<br><br>2. 題目需要頻繁透過 Index 隨機存取資料。<br><br>3. 經常作為雙指針 (Two Pointers) 或滑動窗口的底層載體。                                           | • LC 27. Remove Element<br><br>• LC 26. Remove Duplicates from Sorted Array<br><br>• LC 189. Rotate Array<br><br>• LC 54. Spiral Matrix                        |
| Linked List<br>(單向鏈結)        | 節點散落於記憶體，靠指標相連。只能從頭開始 $O(N)$ 循序走訪，但已知位置的插入/刪除是 $O(1)$。   | 1. 題目明示輸入結構為 ListNode。<br><br>2. 頻繁要求將某段資料剪下、貼到另一個位置。<br><br>3. 題目禁止使用額外陣列儲存，必須靠改變指標方向來重組資料。                                                                    | • LC 206. Reverse Linked List<br><br>• LC 21. Merge Two Sorted Lists<br><br>• LC 141. Linked List Cycle (搭配快慢指針)<br><br>• LC 25. Reverse Nodes in k-Group      |
| Doubly Linked List<br>(雙向鏈結) | 每個節點同時記住「上一個」與「下一個」是誰，可以雙向來回走訪。                          | 1. 題目需要在 $O(1)$ 時間內刪除「陣列中間」的特定元素（通常會搭配 Hash Map 來定位該節點）。<br><br>2. 需要實作某種具備「淘汰機制」的快取系統 (Cache)。                                                                 | • LC 146. LRU Cache<br><br>• LC 460. LFU Cache<br><br>• LC 432. All O`one Data Structure<br><br>• LC 1472. Design Browser History                              |
| Two Pointers<br>(雙指針)        | 透過兩個變數模擬位置，**降維打擊**：將暴力解的 $O(N^2)$ 雙層迴圈優化為 $O(N)$ 的單層遍歷。 | 1. **碰撞指針 (左右向內)：** 陣列**已排序**，尋找兩數之和，或是在兩端夾擠求最大容量 (如裝水)。<br><br>2. **快慢指針 (同向)：** 偵測鏈表是否有環、尋找鏈表中點，或是覆寫陣列中的無效元素。<br><br>3. **滑動窗口 (同向)：** 尋找滿足特定條件的「連續」最長或最短子陣列。 | • LC 15. 3Sum<br><br>• LC 11. Container With Most Water<br><br>• LC 3. Longest Substring Without Repeating Characters<br><br>• LC 76. Minimum Window Substring |
|                              |                                                          |                                                                                                                                                                 |                                                                                                                                                                |


#### 二、 堆疊、佇列與雜湊映射 (Stacks, Queues & Hashing)
```
二、 堆疊、佇列與雜湊映射 (Stacks, Queues & Hashing)
```
Stack
Monotonic Stack
Queue / Deque
Monotonic Queue
Hash Table / Map / Set

| **資料結構 / 算法**              | **核心特性與運作原理**                                    | **判定場景：當你看到題目出現以下特徵時使用**                                                                                                            | **經典 LeetCode 題目代表**                                                                                                                                                                                                               |
| -------------------------- | ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Stack<br>(一般棧)             | LIFO。只能對最後放入的元素進行操作。具有「記憶與回溯」的能力。                | 1. 出現**巢狀結構**或**成對出現**的符號（如括號、XML標籤），需要檢查是否合法匹配。<br><br>2. 運算式解析：遇到運算符號需要「回頭」拿出前兩個數字計算。<br><br>3. 類似檔案系統或瀏覽器的「上一頁 / 返回」操作。          | • LC 20. Valid Parentheses<br><br>• LC 150. Evaluate Reverse Polish Notation<br><br>• LC 71. Simplify Path<br><br>• LC 227. Basic Calculator II                                                                                    |
| Monotonic Stack<br>(單調棧)   | 強制維持棧內元素嚴格遞增或遞減。遇到破壞規則的新元素，必須觸發「結算」並剔除舊元素。       | 1. 關鍵字：尋找陣列中每個元素「左邊或右邊」的**第一個**比它大（或小）的元素。<br><br>2. 題目涉及圖形邊界控制：例如接雨水能積多高、直方圖能框出多大的矩形。                                              | • LC 739. Daily Temperatures<br><br>• LC 496. Next Greater Element I<br><br>• LC 84. Largest Rectangle in Histogram<br><br>• LC 42. Trapping Rain Water                                                                            |
| Queue / Deque<br>(一般 / 雙端) | FIFO。確保處理的順序公平。Deque 則是頭尾皆可 $O(1)$ 進出。           | 1. **廣度優先搜尋 (BFS)：** 題目要求「最短步數」、「最少層數」或「按層級印出結果」時，必定使用 Queue。<br><br>2. 資源排程問題或滑動窗口的底層。                                             | • LC 102. Binary Tree Level Order Traversal<br><br>• LC 279. Perfect Squares<br><br>• LC 994. Rotting Oranges<br><br>• LC 207. Course Schedule (拓撲排序)                                                                              |
| Monotonic Queue<br>(單調隊列)  | 雙端操作，但隊列內強制維持單調性。用來維護一個會移動的區間內的最優解。              | 1. 題目明確給定一個**固定大小為 K 的滑動窗口**，且要求每個窗口的極值。<br><br>2. 題目要求在一個連續區間內尋找極大/極小值，且區間的左右邊界都在單向移動中。                                            | • LC 239. Sliding Window Maximum<br><br>• LC 862. Shortest Subarray with Sum at Least K<br><br>• LC 1425. Constrained Subsequence Sum<br><br>• LC 1438. Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit |
| **Hash Table / Map / Set** | 透過 Hash Function 直接計算記憶體位置，達成 $O(1)$ 極速查找、更新與去重。 | 1. 題目需要判斷某個元素**「是否存在」，且無法承受 $O(N)$ 的搜尋時間。<br><br>2. 需要統計元素出現的「頻率 (Count)」**。<br><br>3. 需要建立一種「查表」機制（Key 映射到 Value），例如利用字串特徵對資料進行分組。 | • LC 1. Two Sum<br><br>• LC 49. Group Anagrams<br><br>• LC 128. Longest Consecutive Sequence<br><br>• LC 560. Subarray Sum Equals K                                                                                                |




#### 三、 區間計算與動態查詢 (Ranges & Dynamic Queries)
```
三、 區間計算與動態查詢 (Ranges & Dynamic Queries)
```
Prefix Sum
Binary Indexed Tree(BIT)
Segment Tree

| **資料結構 / 算法**                 | **核心特性與運作原理**                                                           | **判定場景：當你看到題目出現以下特徵時使用**                                                                                                  | **經典 LeetCode 題目代表**                                                                                                                                                               |
| ----------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Prefix Sum<br>(前綴和)           | 預先花 $O(N)$ 計算累加陣列。之後每一次查詢任意區間 `[i, j]` 的總和，只需 $O(1)$ 即 `P[j+1] - P[i]`。 | 1. 題目要求頻繁查詢**連續子陣列 (Subarray)** 的總和，且該陣列**不會被中途修改 (Immutable)**。<br><br>2. 常搭配 Hash Map 使用，用來尋找「總和等於 K」或是「總和能被 K 整除」的子陣列。 | • LC 303. Range Sum Query - Immutable<br><br>• LC 560. Subarray Sum Equals K<br><br>• LC 974. Subarray Sums Divisible by K<br><br>• LC 525. Contiguous Array                       |
| **Binary Indexed Tree (BIT)** | 又稱樹狀數組。利用二進位特徵，在 $O(\log N)$ 內完成「單點更新」與「前綴和查詢」。空間極小。                    | 1. 陣列內容**會被頻繁修改 (Mutable)**，同時又需要頻繁查詢區間總和。<br><br>2. 經典應用：統計陣列中的「逆序對」數量，或是「計算右側小於當前元素的個數」。                                | • LC 307. Range Sum Query - Mutable<br><br>• LC 315. Count of Smaller Numbers After Self<br><br>• LC 493. Reverse Pairs<br><br>• LC 1649. Create Sorted Array through Instructions |
| Segment Tree<br>(線段樹)         | 完美平衡的二元樹，將陣列切分為無數個小區間。在 $O(\log N)$ 內完成「區間更新」與「區間極值/總和查詢」。              | 1. 題目不只要求單點更新，還要求**「整個區間同時加上某個值 (區間更新)」。<br><br>2. BIT 只能求和，如果題目要求動態查詢區間的最大值 / 最小值**，通常必須使用線段樹。                           | • LC 307. Range Sum Query - Mutable (亦可用線段樹)<br><br>• LC 218. The Skyline Problem<br><br>• LC 715. Range Module<br><br>• LC 699. Falling Squares                                   |



#### 四、 樹狀與圖論高階結構 (Trees & Graphs)
```
### 四、 樹狀與圖論高階結構 (Trees & Graphs)
```
Heap
Binary Tree
BST
Trie
Union Find
Balanced Tree

| **資料結構 / 算法**              | **核心特性與運作原理**                                                        | **判定場景：當你看到題目出現以下特徵時使用**                                                                                                                            | **經典 LeetCode 題目代表**                                                                                                                                                                       |
| -------------------------- | -------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Heap<br>(優先佇列)             | 一棵完全二元樹，確保最大的（或最小的）元素永遠在根節點。插入與提取極值為 $O(\log N)$。                    | 1. 關鍵字：**Top K 問題**（求第 K 大、前 K 高頻的元素）。<br><br>2. 在動態新增資料的過程中，隨時需要取得當前所有資料的最大值或最小值。<br><br>3. 多路歸併問題（將 K 個已排序的陣列合併為一個）。                              | • LC 215. Kth Largest Element in an Array<br><br>• LC 347. Top K Frequent Elements<br><br>• LC 295. Find Median from Data Stream<br><br>• LC 23. Merge k Sorted Lists                      |
| Binary Tree & BST<br>(二元樹) | 遞迴結構。BST 追加了「左小右大」的限制，其中序走訪 (Inorder) 必定是遞增陣列。                       | 1. 題目給定 `TreeNode`，要求計算深度、反轉結構、或是尋找兩個節點的最低公共祖先 (LCA)。<br><br>2. 題目要求設計一個能在 $O(\log N)$ 內支援插入與搜尋，且能維持資料順序的結構 (BST 應用)。                               | • LC 104. Maximum Depth of Binary Tree<br><br>• LC 236. Lowest Common Ancestor of a Binary Tree<br><br>• LC 98. Validate Binary Search Tree<br><br>• LC 230. Kth Smallest Element in a BST |
| Trie<br>(字典樹)              | 以字串的「共同前綴」建構的多元樹。插入與搜尋的速度只跟單字長度 $L$ 有關，$O(L)$。                       | 1. 題目涉及大量的字串搜尋，且明確提到**「前綴 (Prefix)」**或「自動補齊」。<br><br>2. 在一個充滿字母的 2D 矩陣中，尋找多個單字是否存在（搭配 DFS）。<br><br>3. 尋找數字陣列中的最大 XOR 值 (Bitwise Trie)。              | • LC 208. Implement Trie (Prefix Tree)<br><br>• LC 212. Word Search II<br><br>• LC 648. Replace Words<br><br>• LC 421. Maximum XOR of Two Numbers in an Array                              |
| Union Find<br>(並查集)        | 專門用來管理集合的合併 (`union`) 與查詢 (`find`)，搭配路徑壓縮後，操作幾近 $O(1)$。              | 1. 題目探討元素的**「連通性 (Connectivity)」：A 和 B 是否在同一個群組（朋友圈、島嶼）？<br><br>2. 題目給定多條邊，要求判斷無向圖中「是否產生了環 (Cycle)」**。<br><br>3. 最小生成樹 (Kruskal's Algorithm) 的核心組件。 | • LC 200. Number of Islands<br><br>• LC 547. Number of Provinces<br><br>• LC 684. Redundant Connection<br><br>• LC 990. Satisfiability of Equality Equations                               |
| Balanced Tree<br>(平衡樹)     | 自動維持樹高的 BST (如紅黑樹)。支援在 $O(\log N)$ 內找到某個數的地板 (Floor) 或天花板 (Ceiling)。 | _(註：Python 需藉助 `bisect` 模組結合 List 模擬，或匯入第三方庫 `sortedcontainers`)_<br><br>1. 題目需要動態插入資料，且每次插入後都要找「最接近」某個數的值（如日程表是否重疊）。                               | • LC 220. Contains Duplicate III<br><br>• LC 729. My Calendar I<br><br>• LC 731. My Calendar II<br><br>• LC 480. Sliding Window Median                                                     |


#### 五、 核心決策演算法 (Core Algorithmic Paradigms)
```
五、 核心決策演算法 (Core Algorithmic Paradigms)
```
Sorting
Binary Search
Dynamic Programming

| **資料結構 / 算法**                 | **核心特性與運作原理**                                            | **判定場景：當你看到題目出現以下特徵時使用**                                                                                                                               | **經典 LeetCode 題目代表**                                                                                                                                                  |
| ----------------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Sorting<br>(排序)               | 破壞資料原本的位置，但賦予資料大小的單調性，使其具備規律。時間複雜度 $O(N \log N)$。        | 1. 題目要求合併或判斷多個**「區間 (Intervals)」**是否重疊（必須先按起點或終點排序）。<br><br>2. 在使用雙指針（如 3Sum）或二分搜尋之前，資料必須先排序。<br><br>3. 貪婪算法 (Greedy) 的前置步驟，先按權重或價值排序。                 | • LC 56. Merge Intervals<br><br>• LC 435. Non-overlapping Intervals<br><br>• LC 252. Meeting Rooms<br><br>• LC 179. Largest Number                                    |
| Binary Search<br>(二分搜尋)       | 每次將搜尋範圍砍半，極度高效 $O(\log N)$。不僅能搜數字，還能搜「答案」。               | 1. 題目明確說明陣列**「已排序 (Sorted)」**，且要求 $O(\log N)$ 時間內找到目標。<br><br>2. **對答案進行二分搜 (Binary Search on Answer)：** 題目要求尋找滿足某條件的「最大化的最小值」或「最小化的最大值」（例如：吃香蕉的最慢速度）。 | • LC 33. Search in Rotated Sorted Array<br><br>• LC 74. Search a 2D Matrix<br><br>• LC 875. Koko Eating Bananas<br><br>• LC 410. Split Array Largest Sum              |
| Dynamic Programming<br>(動態規劃) | 將大問題拆解為重疊的子問題，並記錄子問題的答案 (Memoization/Tabulation) 避免重複計算。 | 1. 題目要求求出某種**極值**（最長、最大收益、最小成本），且每一步的選擇會影響後續。<br><br>2. 題目要求求出**方案總數**（有幾種方法走到終點）。<br><br>3. 決策具備「無後效性」（未來只看當下狀態，不管過去怎麼走的）。                            | • LC 70. Climbing Stairs<br><br>• LC 322. Coin Change (背包問題)<br><br>• LC 300. Longest Increasing Subsequence (LIS)<br><br>• LC 1143. Longest Common Subsequence (LCS) |



#### 高階選擇心法與判定規則 (How to Choose)
```
高階選擇心法與判定規則 (How to Choose)
```
### 高階選擇心法與判定規則 (How to Choose)

面對一道未知的題目，不要盲目套用資料結構，請先透過「題目的核心意圖 (Intent)」來進行決策樹過濾：


|                                                                                                              |                                                                                                                                                                                              |
| ------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 規則 1：當題目要求「搜尋 (Search) / 尋找 (Find)<br>  -> Hash Map / Hash Set<br>  -> Binary Search<br>  -> Trie            | 資料未排序，且只需知道「有沒有」或「出現幾次」 $\rightarrow$ Hash Map / Hash Set<br><br>資料已排序，且要求 $O(\log N)$ 時間複雜度 $\rightarrow$ Binary Search (二分搜尋)<br><br>搜尋的對象是「字串的開頭 (Prefix)」或多字串比對 $\rightarrow$ Trie (字典樹) |
| 規則 2：當題目要求求出「極值 (Max / Min / 最優解)」時<br>  -> Heap<br>  -> Monotonic Queue<br>  -> * DP                        | 動態變化的資料流，隨時要拿最大或最小值 $\rightarrow$ Heap (優先佇列)<br><br>在平移的固定長度視窗（或動態視窗）中求極值 $\rightarrow$ Monotonic Queue (單調隊列)<br><br>求所有可能組合中的最大收益/最短路徑，且選擇互相影響 $\rightarrow$ DP (動態規劃)                    |
| 規則 3：當題目與「子數組 (Subarray) / 區間 (Range)」有關時<br>  -> Prefix Sum<br>  -> Two pointers<br>  -> * Segment Tree/BIT | 求連續子數組的和，且陣列中途不會被修改 $\rightarrow$ Prefix Sum (前綴和)<br><br>求連續子數組的和/極值，且陣列中途會被修改 $\rightarrow$ Segment Tree / BIT<br><br>求滿足條件的最長/最短子數組（正數陣列） $\rightarrow$ Two Pointers (同向滑動窗口)             |
| 規則 4：當題目與「相對位置 / 距離」有關時<br>  -> Monotonic Stack <br>  -> * Merge Sort                                        | 找每個數字左/右邊，第一個比它大/小的數字 $\rightarrow$ Monotonic Stack (單調棧)<br><br> 範圍內有幾個數字比當前數字小 (逆序對) $\rightarrow$ Merge Sort (歸併排序) / BIT                                                                 |
| 規則 5：當題目探討「連通性 / 群組關係」時<br>  -> DFS / BFS<br>  -> Union Find                                                 | 一次性給定圖，問能走多深、有幾個區塊 $\rightarrow$ DFS / BFS<br><br>動態連通，隨時查詢 A 和 B 是否同群組，或找環 $\rightarrow$ Union Find (並查集)                                                                                   |




#### 規則 1：當題目要求「搜尋 (Search) / 尋找 (Find)」時

|                                                                                    |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 如果資料**沒有排序**，且只需知道「有沒有」或「出現幾次」 $\rightarrow$ 絕對是 **Hash Map / Hash Set** ($O(1)$)。 | - **56. [[Two Sum]] (兩數之和)：** 給定目標值，尋找陣列中是否存在 `target - nums[i]`。（用 Hash Map 記錄看過的數字與索引）。通常解法Two pointers, 最優解法=Hash Map<br>nums = [2,7,11,15], target=9<br>-> [0,1]<br>    <br>- **124. [[Longest Consecutive Sequence]] (最長連續序列)：** 要求 $O(N)$ 找到最長的連續數字。（用 Hash Set 達成 $O(1)$ 查找相鄰數字是否存在）。<br>nums = [100,4,200,1,3,2] <br>-> 4 (1,2,3,4)<br>    <br>- **838. [[Subarray Sum Equals K]]  (和為 K 的子數組)：** Q: 给定一个整数数组和一个整数k，你需要找到连续子数列的和为k的总个数. -> 尋找前綴和是否出現過特定數值。（用 Hash Map 記錄前綴和出現的次數）。<br>nums = [2,1,-1,1,2], k=3<br>-> [0,1], [1,4], [0,3] and [3,4]                                                                                                               |
| 如果資料**已經排序**，且要求 $O(\log N)$ 時間複雜度 $\rightarrow$ 絕對是 **Binary Search**。            | - **457. [[Classical Binary Search]] (二分搜尋)：** Q:在一个排序数组中找一个数，返回该数出现的任意位置，如果不存在，返回 `-1`。<br>nums = [1,2,2,4,5,5], target=2<br>-> 1 or 2<br>    <br>- **62. [[Search in Rotated Sorted Array]] (搜尋旋轉排序陣列)：** 陣列局部有序，透過判斷哪一半是有序的來決定二分縮減的方向。<br>nums = [4,5,1,2,3], target=1<br>-> 2<br>    <br>- **28. [[Search a 2D Matrix]] (搜尋二維矩陣)：** 將二維矩陣映射為一維陣列後，直接套用二分搜尋。<br>2D matrix = [<br>  [1, 3, 5, 7],<br>  [10, 11, 16, 20],<br>  [23, 30, 34, 50]<br>]<br>target = 3<br>-> true                                                                                                                                                                                           |
| 如果搜尋的對象是**字串的開頭 (Prefix)** $\rightarrow$ 絕對是 **Trie (字典樹)**。                       | - **442. [[Implement Trie (Prefix Tree)]] (實作字典樹)：** 建立 Trie 來支援 `insert`, `search`, 與 `startsWith`。<br>   insert("lintcode")<br>   search("lint")<br>   startsWith("lint")<br>   -> false, true<br>    <br>- **1110. [[Replace Words]] (替換單詞)：** 給定詞根，找出句子中單字的最短前綴並替換。<br>    dict = ["cat", "bat", "rat"]<br>  sentence = "the cattle was rattled by the battery"<br>  -> "the cat was rat by the bat"<br>  <br>- **132. [[Word Search II]]  (單詞搜尋 II)：** Q:给出一个由小写字母组成的2D矩阵和一个字典。找出所有同时在字典和矩阵中出现的单词 ->在矩陣中找多個單字，將單字存入 Trie，用 DFS 配合 Trie 進行快速剪枝。<br>   2D board=["doaf","agai","dcan"]，words =["dog","dad","dgdg","can","again"]<br>   输出：["again","can","dad","dog"] |


#### 規則 2：當題目要求求出「極值 (Max / Min / 最優解)」時

|                                                            |                                                                                                                                                                                                                                                                                                                        |
| ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 動態變化的資料流，隨時要拿最大或最小值 $\rightarrow$ Heap (優先佇列)              | - **LC 215. Kth Largest Element in an Array (陣列中的第 K 個最大元素)：** 維護一個大小為 K 的 Min-Heap，最後留在堆頂的就是答案。<br>    <br>- **LC 295. Find Median from Data Stream (資料流的中位數)：** 新增資料的同時隨時求中位數。用一個 Max-Heap 和一個 Min-Heap 各管一半資料。<br>    <br>- **LC 23. Merge k Sorted Lists (合併 K 個排序鏈表)：** 將 K 個鏈表的頭節點丟進 Min-Heap，每次彈出最小的，再把下一個節點推入。 |
| 在平移的固定長度視窗（或動態視窗）中求極值 $\rightarrow$ Monotonic Queue (單調隊列) | - **LC 239. Sliding Window Maximum (滑動窗口最大值)：** 最經典模板題，維護單調遞減隊列來拿窗口最大值。<br>    <br>- **LC 862. Shortest Subarray with Sum at Least K (和至少為 K 的最短子數組)：** 動態視窗。維護前綴和的單調遞增隊列。<br>    <br>- **LC 1438. Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit：** 同時維護一個單調遞增和一個單調遞減隊列，隨時掌握動態窗口內的最大差值。   |
| 求所有可能組合中的最大收益/最短路徑，且選擇互相影響 $\rightarrow$ DP (動態規劃)         | - **LC 322. Coin Change (零錢兌換)：** 給定硬幣面額求湊出金額的最少硬幣數。（完全背包問題）。<br>    <br>- **LC 300. Longest Increasing Subsequence (最長遞增子序列)：** 每個數字選或不選會影響後面的長度，典型的狀態轉移。<br>    <br>- **LC 1143. Longest Common Subsequence (最長公共子序列)：** 比較兩個字串，二維 DP 的經典題。                                                                          |



#### 規則 3：當題目與「子數組 (Subarray) / 區間 (Range)」有關時

|                                                                                            |                                                                                                                                                                                                                                                                                                        |
| ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 如果是求**連續子數組的和**，且陣列中途**不會**被修改 $\rightarrow$ 使用 **Prefix Sum (前綴和)**。                      | - **LC 303. Range Sum Query - Immutable (區域和檢索 - 陣列不可變)：** 初始化後，多次查詢 `[i, j]` 的總和。<br>    <br>- **LC 974. Subarray Sums Divisible by K (和可被 K 整除的子數組)：** 尋找前綴和除以 K 餘數相同的兩個點。<br>    <br>- **LC 525. Contiguous Array (連續陣列)：** 將 0 視為 -1，尋找前綴和相等的兩個點（代表區間內和為 0，即 0 和 1 數量相等）。                          |
| 如果是求**連續子數組的和/極值**，且陣列中途**會**被修改 $\rightarrow$ 使用 **Segment Tree (線段樹)** 或 **BIT (樹狀數組)**。 | - **LC 307. Range Sum Query - Mutable (區域和檢索 - 陣列可修改)：** 支援單點更新與區間和查詢（可用 BIT 或線段樹）。<br>    <br>- **LC 315. Count of Smaller Numbers After Self (計算右側小於當前元素的個數)：** 將數字映射為索引，由後往前遍歷，使用 BIT 動態更新並查詢前綴和（逆序對）。<br>    <br>- **LC 218. The Skyline Problem (天際線問題)：** 建築物區間重疊，需動態求區間最大高度變化。（可用線段樹或掃描線+Heap）。 |
| 如果是求「滿足某條件的**最長/最短**子數組」，且所有數字皆為正數 $\rightarrow$ 使用 **Two Pointers (同向滑動窗口)**。             | - **LC 209. Minimum Size Subarray Sum (長度最小的子數組)：** 尋找和 $\ge target$ 的最短連續子陣列。<br>    <br>- **LC 3. Longest Substring Without Repeating Characters (無重複字元的最長子字串)：** 用 Hash Set 配合雙指針維護不重複的視窗。<br>    <br>- **LC 1004. Max Consecutive Ones III (最大連續 1 的個數 III)：** 允許將 K 個 0 變成 1，求最長的 1 子陣列。          |


#### 規則 4：當題目與「相對位置 / 距離」有關時

|                                                                                         |                                                                                                                                                                                                                                                                                   |
| --------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 如果要找「每個數字左/右邊，第一個比它大/小的數字」 $\rightarrow$ 絕對是 **Monotonic Stack (單調棧)**。                 | - **LC 739. Daily Temperatures (每日溫度)：** 找右邊第一個比自己大的數字（存索引求距離）。<br>    <br>- **LC 84. Largest Rectangle in Histogram (直方圖最大矩形面積)：** 找左右兩邊第一個比自己小的數字來確立寬度邊界。<br>    <br>- **LC 42. Trapping Rain Water (接雨水)：** 找左右兩邊的更高柱子來形成凹槽積水。                                                 |
| 如果要求出「某個範圍內有幾個數字比當前數字小」(例如逆序對) $\rightarrow$ 使用 **BIT (樹狀數組)** 或 **Merge Sort (歸併排序)**。 | - **LC 315. Count of Smaller Numbers After Self：**（這題亦可歸類於此，是求相對位置的經典題）。<br>    <br>- **LC 493. Reverse Pairs (翻轉對)：** 尋找 `i < j` 且 `nums[i] > 2 * nums[j]` 的數量。（利用 Merge Sort 過程順便統計）。<br>    <br>- **LC 327. Count of Range Sum (區間和的個數)：** 利用 Merge Sort 來統計兩個前綴和之差落在特定範圍內的數量。 |
|                                                                                         |                                                                                                                                                                                                                                                                                   |



#### 規則 5：當題目探討「連通性 / 群組關係」時

|                                                                               |                                                                                                                                                                                                                                                                                                         |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 如果是一次性給定一張圖，問能走多深、有幾個區塊 $\rightarrow$ 使用 **DFS / BFS**。                       | - **LC 200. Number of Islands (島嶼數量)：** 遍歷二維陣列，遇到 1 就發動 DFS 將整座島沉沒（標記為 0）。<br>    <br>- **LC 994. Rotting Oranges (腐爛的橘子)：** 多源 BFS，從所有爛橘子同時開始向外擴散，求感染所需時間。<br>    <br>- **LC 133. Clone Graph (複製圖)：** 利用 BFS 或 DFS 走訪圖，並用 Hash Map 記錄原節點與新節點的對應關係。                                                    |
| 如果是**動態**將兩個元素連在一起，隨時問你 A 和 B 是不是同一國的 $\rightarrow$ 絕對是 **Union Find (並查集)**。 | - **LC 547. Number of Provinces (省份數量)：** 給定相鄰矩陣，將連通的城市合併，最後計算有幾個獨立的 Root（群組數）。<br>    <br>- **LC 684. Redundant Connection (冗餘連接)：** 在無向圖的建樹過程中，如果發現兩個端點已經在同一個集合，代表這條邊會產生環，就是冗餘的邊。<br>    <br>- **LC 323. Number of Connected Components in an Undirected Graph (無向圖中連通分量的數目)：** 標準的並查集題，統計合併後剩下的集合數量。 |
|                                                                               |                                                                                                                                                                                                                                                                                                         |
