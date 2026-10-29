
|                                                   |                                                                                        |
| ------------------------------------------------- | -------------------------------------------------------------------------------------- |
| [[#### 一、 基礎線性與指標操作 (Linear & Pointers)]]         | Array<br>Linked list<br>Doubly Linked List<br>Two Pointers                             |
| [[#### 二、 堆疊、佇列與雜湊映射 (Stacks, Queues & Hashing)]] | Stack<br>Monotonic Stack<br>Queue / Deque<br>Monotonic Queue<br>Hash Table / Map / Set |
| [[#### 三、 區間計算與動態查詢 (Ranges & Dynamic Queries)]]  | Prefix Sum<br>Binary Indexed Tree(BIT)<br>Segment Tree                                 |
| [[#### 四、 樹狀與圖論高階結構 (Trees & Graphs)]]            | Heap<br>Binary Tree<br>BST<br>Trie<br>Union Find<br>Balanced Tree                      |
| [[#### 五、 核心決策演算法 (Core Algorithmic Paradigms)]]  | Sorting<br>Binary Search<br>Dynamic Programming                                        |
| [[#### 高階選擇心法與判定規則 (How to Choose)]]              |                                                                                        |
| [[#### 更完整的高階選擇心法與判定規則(8個Rules)]]                 |                                                                                        |
|                                                   |                                                                                        |
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


|                                                                                                              |                                                                                                                                                                                                  |
| ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 規則 1：當題目要求「搜尋 (Search) / 尋找 (Find)<br>  -> Hash Map / Hash Set<br>  -> Binary Search<br>  -> Trie            | 資料未排序，且只需知道「有沒有」或「出現幾次」 $\rightarrow$ Hash Map / Hash Set<br><br>資料已排序，且要求 $O(\log N)$ 時間複雜度 $\rightarrow$ Binary Search (二分搜尋)<br><br>搜尋的對象是「字串的開頭 (Prefix)」或多字串比對 $\rightarrow$ Trie (字典樹)     |
| 規則 2：當題目要求求出「極值 (Max / Min / 最優解)」時<br>  -> Heap<br>  -> Monotonic Queue<br>  -> * Segment Tree<br>  -> * DP | 在可動態變化的資料，隨時要拿全域最大或最小值 -> Heap (優先佇列)<br><br>在規律滑動的固定長度視窗中求區間極值 -> Monotonic Queue (單調隊列)<br><br>在可隨機修改某筆資料或整段資料的隨機任意區間求區間極值 -> Segment Tree(線段樹)<br><br>求所有可能組合中的最大收益/最短路徑，且選擇互相影響 -> DP (動態規劃) |
| 規則 3：當題目與「子數組 (Subarray) / 區間 (Range)」有關時<br>  -> Prefix Sum<br>  -> Two pointers<br>  -> * Segment Tree/BIT | 求連續子數組的和，且陣列中途不會被修改 $\rightarrow$ Prefix Sum (前綴和)<br><br>求連續子數組的和/極值，且陣列中途會被修改 $\rightarrow$ Segment Tree / BIT<br><br>求滿足條件的最長/最短子數組（正數陣列） $\rightarrow$ Two Pointers (同向滑動窗口)                 |
| 規則 4：當題目與「相對位置 / 距離」有關時<br>  -> Monotonic Stack <br>  -> * Merge Sort                                        | 找每個數字左/右邊，第一個比它大/小的數字 $\rightarrow$ Monotonic Stack (單調棧)<br><br> 範圍內有幾個數字比當前數字小 (逆序對) $\rightarrow$ Merge Sort (歸併排序) / BIT                                                                     |
| 規則 5：當題目探討「連通性 / 群組關係」時<br>  -> DFS / BFS<br>  -> Union Find                                                 | 一次性給定圖，問能走多深、有幾個區塊 $\rightarrow$ DFS / BFS<br><br>動態連通，隨時查詢 A 和 B 是否同群組，或找環 $\rightarrow$ Union Find (並查集)                                                                                       |




#### 規則 1：當題目要求「搜尋 (Search) / 尋找 (Find)」時

|                                                                                    |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 如果資料**沒有排序**，且只需知道「有沒有」或「出現幾次」 $\rightarrow$ 絕對是 **Hash Map / Hash Set** ($O(1)$)。 | - **56. [[Two Sum]] (兩數之和)：** 給定目標值，尋找陣列中是否存在 `target - nums[i]`。（用 Hash Map 記錄看過的數字與索引）。通常解法Two pointers, 最優解法=Hash Map<br>nums = [2,7,11,15], target=9<br>-> [0,1]<br>    <br>- **124. [[Longest Consecutive Sequence]] (最長連續序列)：** 要求 $O(N)$ 找到最長的連續數字。（用 Hash Set 達成 $O(1)$ 查找相鄰數字是否存在）。<br>nums = [100,4,200,1,3,2] <br>-> 4 (1,2,3,4)<br>    <br>- **838. [[Subarray Sum Equals K]]  (和為 K 的子數組)：** Q: 给定一个整数数组和一个整数k，你需要找到连续子数列的和为k的总个数. -> 尋找前綴和是否出現過特定數值。（用 Hash Map 記錄前綴和出現的次數）。<br>nums = [2,1,-1,1,2], k=3<br>-> [0,1], [1,4], [0,3] and [3,4]                                                                                                               |
| 如果資料**已經排序**，且要求 $O(\log N)$ 時間複雜度 $\rightarrow$ 絕對是 **Binary Search**。            | - **457. [[Classical Binary Search]] (二分搜尋)：** Q:在一个排序数组中找一个数，返回该数出现的任意位置，如果不存在，返回 `-1`。<br>nums = [1,2,2,4,5,5], target=2<br>-> 1 or 2<br>    <br>- **62. [[Search in Rotated Sorted Array]] (搜尋旋轉排序陣列)：** 陣列局部有序，透過判斷哪一半是有序的來決定二分縮減的方向。<br>nums = [4,5,1,2,3], target=1<br>-> 2<br>    <br>- **28. [[Search a 2D Matrix]] (搜尋二維矩陣)：** 將二維矩陣映射為一維陣列後，直接套用二分搜尋。<br>2D matrix = [<br>  [1, 3, 5, 7],<br>  [10, 11, 16, 20],<br>  [23, 30, 34, 50]<br>]<br>target = 3<br>-> true                                                                                                                                                                                           |
| 如果搜尋的對象是**字串的開頭 (Prefix)** $\rightarrow$ 絕對是 **Trie (字典樹)**。                       | - **442. [[Implement Trie (Prefix Tree)]] (實作字典樹)：** 建立 Trie 來支援 `insert`, `search`, 與 `startsWith`。<br>   insert("lintcode")<br>   search("lint")<br>   startsWith("lint")<br>   -> false, true<br>    <br>- **1110. [[Replace Words]] (替換單詞)：** 給定詞根，找出句子中單字的最短前綴並替換。<br>    dict = ["cat", "bat", "rat"]<br>  sentence = "the cattle was rattled by the battery"<br>  -> "the cat was rat by the bat"<br>  <br>- **132. [[Word Search II]]  (單詞搜尋 II)：** Q:给出一个由小写字母组成的2D矩阵和一个字典。找出所有同时在字典和矩阵中出现的单词 ->在矩陣中找多個單字，將單字存入 Trie，用 DFS 配合 Trie 進行快速剪枝。<br>   2D board=["doaf","agai","dcan"]，words =["dog","dad","dgdg","can","again"]<br>   输出：["again","can","dad","dog"] |


#### 規則 2：當題目要求求出「極值 (Max / Min / 最優解)」時

|                                                    |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 在可動態變化的資料，隨時要拿全域最大或最小值 -> Heap (優先佇列)              | - - **606. [[Kth Largest Element II]](陣列中的第 K 個最大元素)：** Q: 找到数组中第K大的元素 ->維護一個大小為 K 的 Min-Heap，最後留在堆頂的就是答案。<br>nums = [9,3,2,4,8], k=3<br>-> 4<br>    <br>- **81. [[Find Median from Data Stream]] (資料流的中位數)：** 新增資料的同時隨時求中位數。用一個 Max-Heap 和一個 Min-Heap 各管一半資料。<br>add(1)<br>getMedian()<br>add(2)<br>getMedian()<br>add(3)<br>-> 1, 1, 2<br>    <br>- **104. [[Merge K Sorted Lists]] (合併 K 個排序鏈表)：** Q: 合并 `k` 个排序链表（序列为升序序列），并且返回合并后的排序链表（序列为升序序列） -> 將 K 個鏈表的頭節點丟進 Min-Heap，每次彈出最小的，再把下一個節點推入。<br>lists = [2->4->null,null,-1->null]<br>-1->2->4->null |
| 在規律滑動的固定長度視窗中求區間極值 -> Monotonic Queue (單調隊列)       | - **362. [[Sliding Window Maximum]] (滑動窗口最大值)：** 最經典模板題，維護單調遞減隊列來拿窗口最大值。<br>nums = 1,3,-1,-3,5,3,6,7], k=3<br>-> [3,3,5,5,6,7]<br>    <br>- **1507. [[Shortest Subarray with Sum at Least K]] (和至少為 K 的最短子數組)：** Q: 给定一个整数数组，找到和至少为K的最短子数组长度 -> 動態視窗。維護前綴和的單調遞增隊列。<br>nums = [2,-1,2], k=3<br>-> 3<br>    <br>- **397. [[Longest Continuous Increasing Subsequence]] (最长上升连续子序列)：** Q: 找到一個數組中，最长的连续严格上升子序列的长度。<br>nums = [5, 4, 2, 1, 3]<br>-> 4                                                                                                         |
| 在可隨機修改某筆資料或整段資料的隨機任意區間求區間極值 -> Segment Tree(線段樹)   | - **205. [[Interval Minimum Number]] (区间最小数)：** Q:找到数组中多个区间的最小值。<br>nums = [4,3,2,1], queries = [[1,2],[2,4]]  <br>-> [3,1]<br>    <br>- **206. [[Interval Sum]] (区间求和 I)：** Q: 计算数组中多个区间的和。<br>nums = [1,2,3,4]  , queries = [[1,2],[2,4]]  <br>-> [3,9]<br>    <br>- **212. [[Space Replacement ]] (空格替换)：** Q: 将字符串中的空格替换为 `%20`<br>s = "Mr John Smith" <br>-> "Mr%20John%20Smith"                                                                                                                                                                     |
| 求所有可能組合中的最大收益/最短路徑，且選擇互相影響 $\rightarrow$ DP (動態規劃) | - **92. [[Best Time to Buy and Sell Stock]] (背包问题 )：** 给定一个背包的最大容量和一组物品的体积，找到可以装下的最大体积。<br>nums = [3,4,8,5], m=10<br>-> 9<br>    <br>- **76. [[Longest Increasing Subsequence]](最長遞增子序列)：** 每個數字選或不選會影響後面的長度，典型的狀態轉移。<br>nums = [10,9,2,5,3,7,101,18]  <br>-> 4<br><br>- **77. [[Longest Common Subsequence]] (最長公共子序列)：** 比較兩個字串，二維 DP 的經典題。<br>text1 = "abcde", text2 = "ace"  <br>-> 3                                                                                                                                                                 |



#### 規則 3：當題目與「子數組 (Subarray) / 區間 (Range)」有關時

|                                                                                            |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 如果是求**連續子數組的和**，且陣列中途**不會**被修改 $\rightarrow$ 使用 **Prefix Sum (前綴和)**。                      | - **943. [[Range Sum Query - Immutable]] - Immutable (區域和檢索 - 陣列不可變)：** Q: 给定一个不可变的整数数组，计算任意区间 `[i, j]` 的和 -> 初始化後，多次查詢 `[i, j]` 的總和。<br>nums = [-2, 0, 3, -5, 2, -1]<br>sumRange(0, 2)<br>sumRange(2, 5)<br>sumRange(0, 5)<br>-> 1, -1, -3<br><br>- **138. [[Subarray Sum]] (子数组之和为零)：** Q:定一个整数数组，找到和为零的子数组，返回第一个出现的子数组的起始和结束下标。<br>nums = [-3, 1, 2, -3, 4]<br>-> [0,2] or [1,3]<br>    <br>- **41. [[Maximum Subarray]] (子数组和最大的和)：** Q:找到数组中和最大的连续子数组。<br>nums = [−2,2,−3,4,−1,2,1,−5,3]<br>-> 6                                                                                                                                                                                   |
| 如果是求**連續子數組的和/極值**，且陣列中途**會**被修改 $\rightarrow$ 使用 **Segment Tree (線段樹)** 或 **BIT (樹狀數組)**。 | - **840. [[Range Sum Query - Mutable]] (區域和檢索 - 陣列可修改)：** Q: 给定一个数组，支持更新单个元素值和查询任意子数组的元素和 -> 支援單點更新與區間和查詢（可用 BIT 或線段樹）。<br>  nums = [1, 3, 5]<br>  sumRange(0, 2)<br>  update(1, 2)<br>  sumRange(0, 2)<br>-> 9, 8<br>    <br>- **817. [[Range Sum Query 2D - Mutable]] (范围矩阵元素和 - 可变的 )：** 给定一个二维矩阵，支持更新单个元素值和查询任意子矩阵的元素和。<br>  NumMatrix(<br>    [[3,0,1,4,2],<br>     [5,6,3,2,1],<br>     [1,2,0,1,5],<br>     [4,1,0,1,7],<br>     [1,0,3,0,5]]<br>  )<br>  sumRegion(2,1,4,3)<br>  update(3,2,2)<br>  sumRegion(2,1,4,3)<br>-> 8, 10<br>    <br>- **1645. [[Minimum Subsequence in Non-Increasing Order]] (最少子序列)：** 给定一个数组，找到总和大于剩余数字总和的最小递减子序列。<br>nums = [5,2,4,3,1,6]<br>-> 3 |
| 如果是求「滿足某條件的**最長/最短**子數組」，且所有數字皆為正數 $\rightarrow$ 使用 **Two Pointers (同向滑動窗口)**。             | - **56. [[Two Sum]] (两数之和)：** Q:在数组中找到两个数，使得它们的和等于目标值，返回它们的下标。<br>nums = [2, 7, 11, 15]<br>target = 9 <br>-> [0,1]  <br>  <br>- **539. [[Move Zeroes]](移动零)：** Q:将数组中的所有零移动到末尾，同时保持非零元素的相对顺序<br>nums = [0,1,0,3,12]  <br>-> [1,3,12,0,0]<br>    <br>- **604. [[Window Sum]] (滑动窗口内数的和 )：** Q:给定一个数组和窗口大小 k，返回每个窗口的元素和<br>nums = [1,2,3,4,5], k = 3  <br>-> [6,9,12]                                                                                                                                                                                                                                                                                                                     |


#### 規則 4：當題目與「相對位置 / 距離」有關時

|                                                                         |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 如果要找「每個數字左/右邊，第一個比它大/小的數字」 $\rightarrow$ 絕對是 **Monotonic Stack (單調棧)**。 | - **1060. [[Daily Temperatures]]  (每日溫度)：** Q: 给定一个温度数组，返回每一天之后多少天会出现更高的温度 -> 找右邊第一個比自己大的數字（存索引求距離）。<br>temperatures = [73, 74, 75, 71, 69, 72, 76, 73]<br>-> [1, 1, 4, 2, 1, 1, 0, 0]<br>    <br>- **122. [[Largest Rectangle in Histogram]](直方圖最大矩形面積)：** Q: 给定一个直方图，求直方图中最大的矩形面积 -> 找左右兩邊第一個比自己小的數字來確立寬度邊界。<br>height = [2,1,5,6,2,3]<br>-> 10<br>    <br>- **363. [[Trapping Rain Water]] (接雨水)：** Q: 计算柱状图中能够存储的雨水总量 -> 找左右兩邊的更高柱子來形成凹槽積水。<br>height = <br>[0,1,0,2,<br>1,0,1,3,  <br>2,1,2,1] <br>-> 6 |
| 如果要求出「某個範圍內有幾個數字比當前數字小」(例如逆序對) $\rightarrow$ 使用 **BIT (樹狀數組)**          | - **1297. [[Count of Smaller Numbers After Self]]  (比自己小的元素个数)：** Q: 给定一个整数数组，返回一个新数组，新数组中第 i 个元素是 nums[i] 右侧小于 nums[i] 的元素个数。<br>nums= [5, 2, 6, 1]<br>-> [2, 1, 1, 0]<br>                                                                                                                                                                                                                                                                                                                                  |
|                                                                         |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |



#### 規則 5：當題目探討「連通性 / 群組關係」時

|                                                                               |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 如果是一次性給定一張圖，問能走多深、有幾個區塊 $\rightarrow$ 使用 **DFS / BFS**。                       | - **433. [[Number of Islands]] (島嶼數量)：** Q: 给定一个二维网格，计算其中岛屿的数量 -> 遍歷二維陣列，遇到 1 就發動 DFS 將整座島沉沒（標記為 0）。<br>[<br>  [1,1,0,0,0],<br>  [0,1,0,0,1],<br>  [0,0,0,1,1],<br>  [0,0,0,0,0],<br>  [0,0,0,0,1]<br>]<br>-> 3<br><br>- **611. [[Knight Shortest Path]] (骑士的最短路线)：** Q: 在棋盘上从起点移动到终点，返回骑士所需的最小步数<br>[ <br>[0,0,0],<br>[0,0,0],<br>[0,0,0]]<br>source<br>= [2, 2] <br>destination <br>= [2, 2] <br>-> 2<br><br>- **137. [[Clone Graph]] (複製圖)：** Q: 给定一个无向图，返回其深度克隆后的新图 -> 利用 BFS 或 DFS 走訪圖，並用 Hash Map 記錄原節點與新節點的對應關係。<br>{1,2,4#2,1<br>,4#4,1,2}<br>-><br>{1,2,4#2,1 |
| 如果是**動態**將兩個元素連在一起，隨時問你 A 和 B 是不是同一國的 $\rightarrow$ 絕對是 **Union Find (並查集)**。 | - **178. [[Graph Valid Tree]] (图是否是树 )：** 判断一个无向图是否是一棵树<br>n = 5 <br>edges = [<br>[0, 1], [0, 2], <br>[0, 3], [1, 4]]<br>-> true<br>    <br>- **684. [[Redundant Connection]] (缺少的字符串)：** Q:在图中找到冗余的边，使得移除这条边后图仍然是连通树 ->在無向圖的建樹過程中，如果發現兩個端點已經在同一個集合，代表這條邊會產生環，就是冗餘的邊<br>str1 = <br>"This is an example"<br>str2 = <br>"is example"<br>-> ["This", "an"]<br>    <br>- **805. [[Maximum Association Set]] (最大关联集合 )：** Q: 找到多个字符串集合中关联度最大的集合。<br>ListA = <br>["abc","abc","abc"], <br>ListB = <br>["bcd","acd","def"] <br>-> ["abc","acd","bcd","def"]                   |
|                                                                               |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |




#### 更完整的高階選擇心法與判定規則(8個Rules)
```
以下是我lintcode/lintcode解題時整理的12個data structure. 而在下面的表格則是高階選擇心法與判定規則 (How to Choose)的五個Rules, 請幫我檢查是否合理以及是否能將12個data structure融合進去高階選擇心法與判定規則 (How to Choose)以及更優化這些Rules. 1. Array / PreSum 2. Linked list / Doubly linked list 3. queue / deque / Monotone queue 4. Stack / Monotone stack 5. Heap(堆) 6. Binary Tree(BT) / Binary Search tree(BST) 7. Hash table / Hash map 8. Trie 9. Union find 10. segment tree 11. balanced tree 12 BIT
```

# 二、建議採用的完整 8 Rules（How to Choose）

這是我建議你整理進 LeetCode / LintCode 筆記的正式版本。它涵蓋你列出的全部 12 種 Data Structure，也保留必要的演算法選擇。

|   |   |
|---|---|
|Rule 1|Search / Find / Lookup|
|Rule 2|Sequence / Order / Processing|
|Rule 3|Max / Min / Priority / Top K|
|Rule 4|Subarray / Range Query / Update|
|Rule 5|Relative Position / Rank / Order Statistics|
|Rule 6|Tree / Hierarchy / Traversal|
|Rule 7|Graph / Connectivity / Components|
|Rule 8|Optimization / Shortest Path / Feasibility|

| RULES choices                                                                            | Methods                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Rule 1<br>當題目要求搜<br>尋、查找或有<br>序查詢<br>Search<br>Find<br>Lookup                            | 未排序資料：判斷 Key 是否存在、出現次數或 Key→Value<br>-> Hash Map / Hash Set<br><br>已排序 Array：查特定值、Lower Bound、Upper Bound<br>-> Array + Binary Search<br><br>答案空間具有單調可行性<br>-> Binary Search on Answer<br><br>動態插入／刪除，需保持排序並尋找前驅／後繼<br>-> Balanced BST / Ordered Set<br><br>題目提供 BST，利用 Key 大小關係搜尋<br>-> Binary Search Tree (BST)<br><br>搜尋字串 Prefix、字典或自動補全<br>-> Trie<br><br>                                                                                  |
| Rule 2：<br>當題目涉及資<br>料順序、節點<br>操作或處理流<br>程<br>Sequence<br>Order<br>Processing            | 依 Index 存取、原地修改或分割陣列<br>-> Array / Two Pointers<br><br>已知節點位置，需調整鏈結、插入或刪除<br>-> Linked List / Doubly Linked List<br><br>先進先出 FIFO，依加入順序處理<br>-> Queue<br><br>後進先出 LIFO、括號或巢狀運算<br>-> Stack<br><br>頭尾兩側插入、刪除或維護雙端候選<br>-> Deque<br><br>Linked List 中點、環、倒數節點<br>-> Fast / Slow Pointers                                                                                                                                                           |
| Rule 3<br>當題目要求極<br>值、優先順序<br>或 <br>Top K<br>Max/Min<br>Priority                         | 動態插入，反覆取出全域最高優先級元素<br>-> Heap / Priority Queue<br><br>找 Top K 最大／最小或動態維護 K 個候選<br>-> Size-K Heap<br><br>需要有序 Min/Max，且支援任意元素刪除<br>-> Balanced BST / Multiset<br><br>固定／規律移動視窗求最大值或最小值<br>-> Monotonic Deque<br><br>資料動態修改，查任意區間 Min/Max/Sum<br>-> Segment Tree<br><br>靜態資料，只需要一次極值統計或單次掃描最佳值<br>-> Array Linear Scan                                                                                                                             |
| Rule 4<br>當題目要求 <br>Subarray、<br>Range Query <br>或區間修改                                   | 靜態 1D Array，多次查任意區間 Sum<br>-> Prefix Sum<br><br>靜態 2D Matrix，查子矩形總和<br>-> 2D Prefix Sum<br><br>最長／最短連續區間，視窗條件具有單調性<br>-> Two Pointers / Sliding Window<br><br>子陣列 Sum = K，包含負數也要處理<br>-> Prefix Sum + Hash Map<br><br>單點修改，反覆查 Prefix Sum / Range Sum<br>-> BIT / Fenwick Tree<br><br>單點修改，任意 Range Sum/Min/Max/GCD<br>-> Segment Tree<br><br>整段 Range Update，且需要區間聚合；或僅要求離線最終結果<br>-> Lazy Segment Tree / Difference Array                      |
| Rule 5<br>當題目涉及相<br>對位置、排名<br>、順序統計<br>Relative Position<br>Rank<br>Order                | 找每個元素左／右第一個 Greater / Smaller<br>-> Monotonic Stack<br><br>Histogram 或 Matrix：尋找邊界、計算矩形<br>-> Monotonic Stack<br><br>逆序對、右側較小值的離線統計<br>-> Merge Sort Counting<br><br>由左到右動態統計比當前值小／大的元素個數<br>-> BIT + Coordinate Compression<br><br>動態有序資料：Floor / Ceiling / Predecessor / Successor<br>-> Balanced BST / Ordered Set<br><br>動態查第 K 小、Rank，維護子樹大小<br>-> Order-Statistic Tree / Augmented BST                                                     |
| Rule 6<br>當題目有階層<br>父子節點或樹<br>狀結構<br>Tree<br>Hierarchy<br>Traversal                      | Binary Tree：Preorder / Inorder / Postorder<br>-> DFS + Recursion / Stack<br><br>Binary Tree：Level Order / Zigzag / 按層分組<br>-> BFS + Queue / Deque<br><br>BST：利用鍵值大小關係搜尋、插入、刪除<br>-> Binary Search Tree (BST)<br><br>BST 需維持高度平衡，或從已排序資料建立平衡樹<br>-> Balanced BST / Balanced BST Construction<br><br>Tree Height / Diameter / Path Sum / Subtree Result<br>-> Tree DFS / Tree DP<br><br>按邊數尋找最近葉節點或逐層探索<br>-> BFS + Queue<br>                      |
| Rule 7<br>當題目探討連<br>通性、群組、<br>朋友圈或Graph<br>Connectivity<br>Components                    | 靜態 Graph：Reachability / Connected Components<br>-> DFS / BFS<br><br>無向圖持續加邊，動態查兩點連通或群組大小<br>-> Union Find / DSU<br><br>無向圖新增 Edge，判斷是否形成 Cycle<br>-> Union Find / DSU<br><br>Unweighted Graph：最短路徑／最少步數<br>-> BFS + Queue<br><br>Directed Graph：Strongly Connected Components<br>-> Tarjan / Kosaraju (DFS)<br><br>存在刪邊或時間批次變化，需離線處理連通性<br>-> Reverse Processing / Rollback DSU                                                                  |
| Rule 8<br>當題目要求最<br>佳化、最短<br>路徑、最大收<br>益或可行性<br>Optimization<br>Optimal <br>Solution<br> | 具 Optimal Substructure + Overlapping Subproblems<br>-> Dynamic Programming (DP)<br><br>能證明局部最佳選擇可達到全域最佳<br>-> Greedy<br><br>無權 Graph 最短路徑／最少步數<br>-> BFS + Queue<br><br>Non-negative Weighted Graph 最短路徑或相關變體<br>-> Dijkstra + Min Heap（瓶頸路徑可用 Max Heap）<br><br>存在負權邊、負權環，或限制邊數的最短路徑<br>-> Bellman-Ford / K-round Relaxation<br><br>答案存在單調可行性，可用二分搜尋最佳界線<br>-> Binary Search on Answer<br><br>DAG：依依賴順序求最長／最短路徑或狀態累積<br>-> Topological Sort + DP |
|                                                                                          |                                                                                                                                                                                                                                                                                                                                                                                                                                                  |

|#|Data Structure|核心用途|對應 Rules|
|---|---|---|---|
|1|Array / Prefix Sum|Random Access、靜態區間總和|2、4|
|2|Linked List / Doubly Linked List|已知節點位置的快速插入／刪除、節點指標操作|2|
|3|Queue / Deque / Monotonic Queue|FIFO、BFS、Sliding Window Extreme|2、3、4、6、7、8|
|4|Stack / Monotonic Stack|LIFO、DFS、Next Greater / Smaller|2、5、6|
|5|Heap|Global Min / Max、Top K、Priority|3、8|
|6|Binary Tree / BST|Hierarchical Traversal、Ordered Search|1、6|
|7|Hash Map / Hash Set|Exact Lookup、Frequency、Prefix Sum Lookup|1、4|
|8|Trie|Prefix Search、Dictionary|1|
|9|Union Find|Dynamic Connectivity、Components|7|
|10|Segment Tree|Dynamic Range Query、Range Update|3、4|
|11|Balanced Tree|Dynamic Ordered Data、Floor / Ceiling、Rank（需擴充）|1、3、5、6|
|12|BIT / Fenwick Tree|Dynamic Prefix / Range Sum、Frequency Counting|4、5|




## Rule 1：當題目要求搜尋、查找或有序查詢（Search / Find / Lookup）

核心判斷：搜尋的是精確 Key、排序位置、相鄰值，還是字串 Prefix？

| 判定條件（When to Choose）                   | 選擇（Data Structure + Big O）                                                    | 例題（每列 3 題）                                                                                                                                                         |
| -------------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 未排序資料：判斷 Key 是否存在、出現次數或 Key→Value      | Hash Map / Hash Set<br><br>Big O：平均查詢／插入 O(1)；最壞 O(N)                         | 1. #1320 Contains Duplicate — 判斷陣列有無重複元素<br><br>2. #547 Intersection of Two Arrays — 找出兩陣列交集<br><br>3. #171 Anagrams — 將字母異位詞分組                                    |
| 已排序 Array：查特定值、Lower Bound、Upper Bound | Array + Binary Search<br><br>Big O：單次查詢 O(log N)                              | 1. #457 Classical Binary Search — 在排序陣列找 target<br><br>2. #14 Binary Search — 找 target 首次出現位置<br><br>3. #62 Search in Rotated Sorted Array — 搜尋旋轉排序陣列              |
| 答案空間具有單調可行性                            | Binary Search on Answer<br><br>Big O：O(T_check × log R)，其中 T_check 為單次可行性檢查成本 | 1. #183 Wood Cut — 求可切出 K 段的最大木段長度<br><br>2. #437 Copy Books — 最小化分配書本的完成時間<br><br>3. #1046 Minimize Max Distance to Gas Station — 最小化加油站間最大距離                     |
| 動態插入／刪除，需保持排序並尋找前驅／後繼                  | Balanced BST / Ordered Set<br><br>Big O：查詢／插入／刪除 O(log N)                     | 1. #1280 Data Stream as Disjoint Intervals — 動態維護不相交區間<br><br>2. #1513 Exam Room — 動態安排與釋放座位<br><br>3. #360 Sliding Window Median — 動態維護視窗中位數                      |
| 題目提供 BST，利用 Key 大小關係搜尋                 | Binary Search Tree (BST)<br><br>Big O：O(h)；平衡時 O(log N)，最壞 O(N)               | 1. #1524 Search in a Binary Search Tree — 搜尋 BST 指定節點<br><br>2. #448 Inorder Successor in BST — 找中序後繼<br><br>3. #900 Closest Binary Search Tree Value — 找最接近目標值的節點 |
| 搜尋字串 Prefix、字典或自動補全                    | Trie<br><br>Big O：插入／查詢 O(L)，L 為字串長度                                          | 1. #442 Implement Trie — 實作單字插入與前綴搜尋<br><br>2. #473 Add and Search Word — 支援萬用字元查詢<br><br>3. #1110 Replace Words — 用最短字根取代句子單字                                     |

其中 \(h\) 是 BST 的高度，\(L\) 是字串長度，\(R\) 是答案搜尋範圍。

關鍵選擇
- Hash Map vs Balanced Tree：只問 Key 是否存在，通常用 Hash；需要大小順序或前驅／後繼，用 Balanced Tree。
- Binary Search vs Balanced Tree：資料已排序且不太修改，用 Array + Binary Search；資料頻繁變動且需要有序查詢，用 Balanced Tree。
- BST vs Balanced Tree：一般 BST 最壞可能退化成 \(O(N)\)；Balanced BST 才能維持 \(O(\log N)\) 的高度界限。
- Trie vs Hash Set：完整單字查詢通常 Hash Set 足夠；大量共享 Prefix 的查詢，Trie 更有優勢。

## Rule 2：當題目涉及資料順序、節點操作或處理流程（Sequence / Order / Processing）

核心判斷：資料需要 Random Access、頻繁修改相鄰節點，還是依照特定順序取出？

| 判定條件（When to Choose）  | 選擇（Data Structure + Big O）                                           | 例題（每列 3 題）                                                                                                                                                       |
| --------------------- | -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 依 Index 存取、原地修改或分割陣列  | Array / Two Pointers<br><br>Big O：Index O(1)；一般遍歷 O(N)               | 1. #100 Remove Duplicates from Sorted Array — 原地移除重複數字<br><br>2. #148 Sort Colors — 原地排列三種顏色<br><br>3. #162 Set Matrix Zeroes — 將含零位置的整列整行設零                     |
| 已知節點位置，需調整鏈結、插入或刪除    | Linked List / Doubly Linked List<br><br>Big O：已知適當節點時修改 O(1)；搜尋 O(N) | 1. #35 Reverse Linked List — 反轉鏈表<br><br>2. #36 Reverse Linked List II — 反轉指定鏈表區段<br><br>3. #372 Delete Node in a Linked List — 刪除指定非尾節點                         |
| 先進先出 FIFO，依加入順序處理     | Queue<br><br>Big O：入隊／出隊均攤 O(1)                                      | 1. #642 Moving Average from Data Stream — 求資料流滑動平均<br><br>2. #540 Zigzag Iterator — 輪流讀取兩個列表<br><br>3. #494 Implement Stack using Queues — 用 Queue 實作 Stack      |
| 後進先出 LIFO、括號或巢狀運算     | Stack<br><br>Big O：Push / Pop / Top O(1)                             | 1. #423 Valid Parentheses — 判斷括號是否合法<br><br>2. #575 Decode String — 解碼巢狀重複字串<br><br>3. #978 Basic Calculator — 計算含括號的算式                                          |
| 頭尾兩側插入、刪除或維護雙端候選      | Deque<br><br>Big O：兩端操作均攤 O(1)                                       | 1. #999 Design Circular Deque — 設計循環雙端佇列<br><br>2. #71 Binary Tree Zigzag Level Order Traversal — 交替方向層次遍歷<br><br>3. #362 Sliding Window Maximum — 以單調雙端佇列求視窗最大值 |
| Linked List 中點、環、倒數節點 | Fast / Slow Pointers<br><br>Big O：O(N) 時間；O(1) 額外空間                  | 1. #1609 Middle of the Linked List — 找鏈表中點<br><br>2. #102 Linked List Cycle — 判斷有無環<br><br>3. #103 Linked List Cycle II — 找環的入口                                  |

特別注意 Linked List 的複雜度：

> 已知適當節點位置時，插入／刪除可以是 \(O(1)\)；但如果需要先搜尋節點，搜尋仍然是 \(O(N)\)。

另外，Singly Linked List 刪除節點通常需要知道前一個節點；Doubly Linked List 如果已知待刪除節點，可以直接調整前後指標。

這條規則可以補足你原本五條中完全缺少的 Array、Linked List、一般 Stack、Queue 和 Deque。

## Rule 3：當題目要求極值、優先順序或 Top K（Max / Min / Priority）

核心判斷：要找全域極值、滑動視窗極值，還是任意區間極值？

| 判定條件（When to Choose）     | 選擇（Data Structure + Big O）                                                    | 例題（每列 3 題）                                                                                                                                                                           |
| ------------------------ | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 動態插入，反覆取出全域最高優先級元素       | Heap / Priority Queue<br><br>Big O：Peek O(1)；Push / Pop O(log N)              | 1. #1872 Minimum Cost to Connect Sticks — 最低成本合併木棍<br><br>2. #919 Meeting Rooms II — 最少會議室數量<br><br>3. #81 Find Median from Data Stream — 以雙 Heap 維護中位數                              |
| 找 Top K 最大／最小或動態維護 K 個候選 | Size-K Heap<br><br>Big O：O(N log K) 時間；O(K) 空間                                | 1. #544 Top k Largest Numbers — 找陣列前 K 大<br><br>2. #545 Top k Largest Numbers II — 動態維護前 K 大<br><br>3. #612 K Closest Points — 找最近的 K 個點                                             |
| 需要有序 Min/Max，且支援任意元素刪除   | Balanced BST / Multiset<br><br>Big O：增刪 O(log N)；取極值 O(1) 或 O(log N)，視 API 而定 | 1. #360 Sliding Window Median — 滑動視窗刪除舊數字<br><br>2. #1513 Exam Room — 動態管理座位區間<br><br>3. #1280 Data Stream as Disjoint Intervals — 維護有序區間                                            |
| 固定／規律移動視窗求最大值或最小值        | Monotonic Deque<br><br>Big O：整趟 O(N)                                          | 1. #362 Sliding Window Maximum — 各視窗最大值<br><br>2. #347 Maximum Number Expectation — 視窗極值統計<br><br>3. #3493 Longest Subarray — 用 Deque 輔助最長區間查找                                       |
| 資料動態修改，查任意區間 Min/Max/Sum | Segment Tree<br><br>Big O：建樹 O(N)；Query / Update O(log N)                     | 1. #202 Segment Tree Query — 區間最小值<br><br>2. #247 Segment Tree Query II — 區間最大值<br><br>3. #207 Interval Sum II — 動態修改與區間求和                                                           |
| 靜態資料，只需要一次極值統計或單次掃描最佳值   | Array Linear Scan<br><br>Big O：O(N) 時間；O(1) 額外空間                              | 1. #397 Longest Continuous Increasing Subsequence — 最長連續遞增長度<br><br>2. #149 Best Time to Buy and Sell Stock — 單次買賣最大收益<br><br>3. #585 Maximum Number in Mountain Sequence — 找山形陣列最大值 |

這裡最重要的是區分：
![[Pasted image 20261029110117.png]]

例如 LeetCode 239（Sliding Window Maximum）通常使用 Monotonic Deque，並不需要 Segment Tree。

## Rule 4：當題目要求 Subarray、Range Query 或區間修改

核心判斷：資料是否修改？區間是固定還是任意？查詢的是 Sum、Min/Max 還是其他統計？

| 判定條件（When to Choose）               | 選擇（Data Structure + Big O）                                                                   | 例題（每列 3 題）                                                                                                                                                                                             |
| ---------------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 靜態 1D Array，多次查任意區間 Sum            | Prefix Sum<br><br>Big O：建立 O(N)；每次 Query O(1)                                                | 1. #943 Range Sum Query - Immutable — 查靜態區間和<br><br>2. #206 Interval Sum — 查多個指定區間的和<br><br>3. #604 Window Sum — 計算各固定視窗總和                                                                             |
| 靜態 2D Matrix，查子矩形總和                | 2D Prefix Sum<br><br>Big O：建立 O(MN)；每次 Query O(1)                                            | 1. #665 Range Sum Query 2D - Immutable — 查矩形區域總和<br><br>2. 補充 LC #1314 Matrix Block Sum — 計算各格子周圍矩形和<br><br>3. 補充 LC #1074 Number of Submatrices That Sum to Target — 統計指定總和的子矩形                       |
| 最長／最短連續區間，視窗條件具有單調性                | Two Pointers / Sliding Window<br><br>Big O：通常 O(N)                                           | 1. #406 Minimum Size Subarray Sum — 最短達標子陣列<br><br>2. #384 Longest Substring Without Repeating Characters — 最長無重複子字串<br><br>3. #386 Longest Substring with At Most K Distinct Characters — 最長 K 種字元子字串 |
| 子陣列 Sum = K，包含負數也要處理               | Prefix Sum + Hash Map<br><br>Big O：平均 O(N) 時間；O(N) 空間                                        | 1. #838 Subarray Sum Equals K — 統計總和為 K 的子陣列<br><br>2. #911 Maximum Size Subarray Sum Equals K — 最長總和為 K 的子陣列<br><br>3. #994 Contiguous Array — 最長 0/1 數量相同區間                                          |
| 單點修改，反覆查 Prefix Sum / Range Sum    | BIT / Fenwick Tree<br><br>Big O：Update / Query O(log N)                                      | 1. #840 Range Sum Query - Mutable — 單點更新與區間和<br><br>2. #817 Range Sum Query 2D - Mutable — 二維更新與矩形和<br><br>3. #249 Count of Smaller Numbers Before Self — 前面較小數量統計                                     |
| 單點修改，任意 Range Sum/Min/Max/GCD      | Segment Tree<br><br>Big O：建樹 O(N)；Update / Query O(log N)                                    | 1. #202 Segment Tree Query — 查區間最小值<br><br>2. #247 Segment Tree Query II — 查區間最大值<br><br>3. #207 Interval Sum II — 更新元素並查區間總和                                                                          |
| 整段 Range Update，且需要區間聚合；或僅要求離線最終結果 | Lazy Segment Tree / Difference Array<br><br>Big O：Lazy Update / Query O(log N)；離線差分總計 O(N+Q) | 1. #903 Range Addition — 批次區間加值後輸出最終陣列<br><br>2. 補充 LC #2569 Handling Sum Queries After Update — 區間位元反轉與求和<br><br>3. 補充 LC #732 My Calendar III — 動態求最大重疊區間數                                           |

BIT vs Segment Tree 的判斷

這是演算法面試最容易混淆的選擇之一。

- BIT：特別適合頻繁的 Point Update + Prefix Sum / Range Sum，程式碼較短，空間通常 \(O(N)\)。
- Segment Tree：支援更廣泛的可合併區間資訊，例如 Sum、Min、Max、GCD；也可以使用 Lazy Propagation 處理區間更新。
- BIT 不是只能做 Point Update：透過 Difference Array 的觀念及一棵或兩棵 BIT，也能支援特定的 Range Update / Range Query 組合。
- 一般 BIT 不適合直接處理任意 Point Assignment + Range Minimum Query，這種情況 Segment Tree 通常更直觀。

所以不能只根據「資料有沒有修改」決定 BIT 或 Segment Tree，還必須看 Aggregate Operation 是什麼。

## Rule 5：當題目涉及相對位置、排名、順序統計（Relative Position / Rank / Order）

核心判斷：要找最近的比較對象、計算多少個元素符合大小關係，還是維護動態排名？

| 判定條件（When to Choose）                             | 選擇（Data Structure + Big O）                                                         | 例題（每列 3 題）                                                                                                                                                                                       |
| ------------------------------------------------ | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 找每個元素左／右第一個 Greater / Smaller                    | Monotonic Stack<br><br>Big O：整趟 O(N)                                               | 1. #1206 Next Greater Element I — 右邊第一個更大值<br><br>2. #1201 Next Greater Element II — 循環陣列下一個更大值<br><br>3. #1060 Daily Temperatures — 下一個更暖日的距離                                                   |
| Histogram 或 Matrix：尋找邊界、計算矩形                     | Monotonic Stack<br><br>Big O：1D O(N)；M×N 矩陣 O(MN)                                  | 1. #122 Largest Rectangle in Histogram — 最大柱狀圖矩形<br><br>2. #510 Maximal Rectangle — 最大全 1 矩形<br><br>3. #1860 The Number of 0-submatrix — 統計全 0 子矩形                                               |
| 逆序對、右側較小值的離線統計                                   | Merge Sort Counting<br><br>Big O：O(N log N) 時間；O(N) 空間                             | 1. #1297 Count of Smaller Numbers After Self — 統計右側較小數量<br><br>2. #249 Count of Smaller Numbers Before Self — 統計左側較小數量<br><br>3. 補充 LC #493 Reverse Pairs — 計算滿足倍數大小關係的逆序數對                      |
| 由左到右動態統計比當前值小／大的元素個數                             | BIT + Coordinate Compression<br><br>Big O：O(N log N) 時間；O(N) 空間                    | 1. #249 Count of Smaller Numbers Before Self — 統計前方較小值<br><br>2. #1297 Count of Smaller Numbers After Self — 統計後方較小值<br><br>3. 補充 LC #1649 Create Sorted Array through Instructions — 插入時查左右大小分布 |
| 動態有序資料：Floor / Ceiling / Predecessor / Successor | Balanced BST / Ordered Set<br><br>Big O：增刪／有序查詢 O(log N)                           | 1. #448 Inorder Successor in BST — BST 中序後繼<br><br>2. #900 Closest Binary Search Tree Value — 找最近鍵值<br><br>3. #1513 Exam Room — 依排序座位區間安排位置                                                      |
| 動態查第 K 小、Rank，維護子樹大小                             | Order-Statistic Tree / Augmented BST<br><br>Big O：Kth / Rank / Update O(log N)，須平衡 | 1. #902 Kth Smallest Element in a BST — 查第 K 小節點<br><br>2. #360 Sliding Window Median — 用有序排名維護中位數<br><br>3. #1297 Count of Smaller Numbers After Self — 用排名查較小數量                                |

需要注意：一般 Balanced BST 不一定直接提供 `kth` 或 `rank`，通常必須在節點額外維護 Subtree Size，或者使用支援這些操作的資料結構。

另外，Monotonic Stack 和 Monotonic Queue 的核心差異是：

|        | Monotonic Stack                 | Monotonic Queue        |
| ------ | ------------------------------- | ---------------------- |
| 核心用途   | 找相對位置、Nearest Greater / Smaller | 取得移動區間內的極值             |
| 元素移除原因 | 新元素破壞單調性                        | 新元素破壞單調性，或舊元素離開視窗      |
| 典型題目   | Next Greater Element            | Sliding Window Maximum |
| 整體複雜度  | O(N)                            | O(N)                   |


## Rule 6當題目有階層、父子節點或樹狀結構（Tree / Hierarchy / Traversal）

核心判斷：題目是一般 Tree Traversal、BST 的有序搜尋，還是需要不同的遍歷順序？

| 判定條件（When to Choose）                               | 選擇（Data Structure + Big O）                                                      | 例題（每列 3 題）                                                                                                                                                                           |
| -------------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Binary Tree：Preorder / Inorder / Postorder         | DFS + Recursion / Stack<br><br>Big O：O(N) 時間；O(h) 輔助空間                          | 1. #66 Binary Tree Preorder Traversal — 先序遍歷<br><br>2. #67 Binary Tree Inorder Traversal — 中序遍歷<br><br>3. #68 Binary Tree Postorder Traversal — 後序遍歷                                 |
| Binary Tree：Level Order / Zigzag / 按層分組            | BFS + Queue / Deque<br><br>Big O：O(N) 時間；最多 O(W) Queue 空間                       | 1. #70 Binary Tree Level Order Traversal II — 自底向上層次遍歷<br><br>2. #71 Binary Tree Zigzag Level Order Traversal — 鋸齒層次遍歷<br><br>3. #649 Binary Tree Vertical Order Traversal — 依垂直欄位分組 |
| BST：利用鍵值大小關係搜尋、插入、刪除                               | Binary Search Tree (BST)<br><br>Big O：O(h)；最壞 O(N)                              | 1. #1524 Search in a Binary Search Tree — 搜尋節點<br><br>2. #85 Insert Node in a Binary Search Tree — 插入節點<br><br>3. #87 Remove Node in Binary Search Tree — 刪除節點                       |
| BST 需維持高度平衡，或從已排序資料建立平衡樹                           | Balanced BST / Balanced BST Construction<br><br>Big O：動態操作 O(log N)；有序陣列建樹 O(N) | 1. #1359 Convert Sorted Array to Binary Search Tree — 有序陣列建平衡 BST<br><br>2. #106 Convert Sorted List to Binary Search Tree — 有序鏈表建平衡 BST<br><br>3. #1513 Exam Room — 以有序集合動態管理座位     |
| Tree Height / Diameter / Path Sum / Subtree Result | Tree DFS / Tree DP<br><br>Big O：通常 O(N) 時間；O(h) 遞迴空間                            | 1. #97 Maximum Depth of Binary Tree — 最大深度<br><br>2. #1181 Diameter of Binary Tree — 最長節點距離<br><br>3. #94 Binary Tree Maximum Path Sum — 最大路徑和                                       |
| 按邊數尋找最近葉節點或逐層探索                                    | BFS + Queue<br><br>Big O：O(N) 時間；最多 O(W) 空間                                     | 1. #155 Minimum Depth of Binary Tree — 根到最近葉節點的深度<br><br>2. #70 Binary Tree Level Order Traversal II — 分層處理樹節點<br><br>3. #71 Binary Tree Zigzag Level Order Traversal — 逐層交替遍歷       |

這一條補足你原本缺少的 Binary Tree / BST 判斷規則。

特別是 Binary Tree、BST、Balanced BST 不應當作三個完全無關的資料結構：

- Binary Tree：每個節點最多兩個子節點，沒有必然的 Key 大小關係。
- BST：額外滿足左子樹 Key 小於節點、右子樹 Key 大於節點的有序性（重複值依實作規則處理）。
- Balanced BST：進一步維持高度平衡，例如 AVL Tree、Red-Black Tree，防止嚴重退化。

## Rule 7：當題目探討連通性、群組、朋友圈或 Graph（Connectivity / Components）

核心判斷：只需要知道是否連通，還是要找到實際路徑、遍歷順序或最短距離？

| 判定條件（When to Choose）                         | 選擇（Data Structure + Big O）                                      | 例題（每列 3 題）                                                                                                                                                                                   |
| -------------------------------------------- | --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 靜態 Graph：Reachability / Connected Components | DFS / BFS<br><br>Big O：完整遍歷 O(V+E)                              | 1. #433 Number of Islands — 計算島嶼數量<br><br>2. #431 Connected Component in Undirected Graph — 求連通分量<br><br>3. #176 Route Between Two Nodes in Graph — 判斷兩節點是否可達                                |
| 無向圖持續加邊，動態查兩點連通或群組大小                         | Union Find / DSU<br><br>Big O：每次均攤 O(α(N))                      | 1. #589 Connecting Graph — 動態加邊與連通查詢<br><br>2. #590 Connecting Graph II — 查所在群組大小<br><br>3. #591 Connecting Graph III — 動態查連通分量數量                                                            |
| 無向圖新增 Edge，判斷是否形成 Cycle                      | Union Find / DSU<br><br>Big O：每次均攤 O(α(N))                      | 1. #684 Redundant Connection — 找出造成環的邊<br><br>2. #178 Graph Valid Tree — 判斷無向圖是否為樹<br><br>3. #444 Graph Valid Tree II — 動態檢查新增邊後是否仍為樹                                                        |
| Unweighted Graph：最短路徑／最少步數                   | BFS + Queue<br><br>Big O：O(V+E)                                 | 1. #611 Knight Shortest Path — 騎士最短步數<br><br>2. #630 Knight Shortest Path II — 限制移動方向的最短步數<br><br>3. #120 Word Ladder — 最短單字轉換序列                                                             |
| Directed Graph：Strongly Connected Components | Tarjan / Kosaraju (DFS)<br><br>Big O：O(V+E)                     | 1. 補充 LC #802 Find Eventual Safe States — 找出不會進入環的安全節點<br><br>2. 補充 LC #2360 Longest Cycle in a Graph — 找有向圖最長環<br><br>3. 補充 LC #2127 Maximum Employees to Be Invited to a Meeting — 分析有向環與鏈 |
| 存在刪邊或時間批次變化，需離線處理連通性                         | Reverse Processing / Rollback DSU<br><br>Big O：依演算法而異；逆序加邊通常近線性 | 1. #1014 Bricks Falling When Hit — 逆序恢復磚塊連通性<br><br>2. 補充 LC #1970 Last Day Where You Can Still Cross — 逆序恢復可通行格子<br><br>3. 補充 LC #2092 Find All People With Secret — 依時間批次維護連通群組            |

\(\alpha(N)\) 是 Inverse Ackermann Function，在實際資料規模下非常接近常數。

這條規則最重要的修正是：

> Union Find 維護的是群組連通關係，不是路徑本身。

因此，即使 Union Find 能判斷 A 和 B 是否連通，也不能直接回答「A 到 B 最短路徑經過哪些節點」。

## Rule 8：當題目要求最佳化、最短路徑、最大收益或可行性（Optimization / Optimal Solution）

核心判斷：問題是否存在可重複利用的子問題、貪心選擇性質、最短路徑特性，或單調的可行性條件？

| 判定條件（When to Choose）                             | 選擇（Data Structure + Big O）                                          | 例題（每列 3 題）                                                                                                                                                                                      |
| ------------------------------------------------ | ------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 具 Optimal Substructure + Overlapping Subproblems | Dynamic Programming (DP)<br><br>Big O：O(S×T)，S 為狀態數、T 為每狀態轉移成本      | 1. #77 Longest Common Subsequence — 最長公共子序列<br><br>2. #119 Edit Distance — 最少字串編輯步數<br><br>3. #92 Backpack — 背包容量內最大可裝體積                                                                        |
| 能證明局部最佳選擇可達到全域最佳                                 | Greedy<br><br>Big O：常見 O(N) 或 O(N log N)，視問題而定                      | 1. #116 Jump Game — 判斷能否跳至終點<br><br>2. #150 Best Time to Buy and Sell Stock II — 多次買賣最大利潤<br><br>3. #1872 Minimum Cost to Connect Sticks — 每次合併最短兩根木棍                                           |
| 無權 Graph 最短路徑／最少步數                               | BFS + Queue<br><br>Big O：O(V+E)                                     | 1. #611 Knight Shortest Path — 最少騎士移動次數<br><br>2. #120 Word Ladder — 最少單字轉換步數<br><br>3. #513 Perfect Squares — 最少完全平方數組合                                                                        |
| Non-negative Weighted Graph 最短路徑或相關變體            | Dijkstra + Min Heap（瓶頸路徑可用 Max Heap）<br><br>Big O：通常 O((V+E) log V) | 1. #1057 Network Delay Time — 計算訊號傳到所有節點時間<br><br>2. #1418 Path With Maximum Minimum Value — 最大化路徑最小值<br><br>3. #789 The Maze III — 以滾動距離求迷宮最短路徑                                                |
| 存在負權邊、負權環，或限制邊數的最短路徑                             | Bellman-Ford / K-round Relaxation<br><br>Big O：標準 O(VE)；K 輪鬆弛 O(KE) | 1. 補充 LC #787 Cheapest Flights Within K Stops — 限制轉機數求最低票價<br><br>2. 補充 CSES #1673 High Score — 處理圖中可能無界的最大得分<br><br>3. 補充 CSES #1197 Cycle Finding — 偵測負權環                                     |
| 答案存在單調可行性，可用二分搜尋最佳界線                             | Binary Search on Answer<br><br>Big O：O(T_check log R)               | 1. #183 Wood Cut — 求可行最大木段長度<br><br>2. #437 Copy Books — 最小化最大抄寫時間<br><br>3. #1046 Minimize Max Distance to Gas Station — 最小化最大間距                                                               |
| DAG：依依賴順序求最長／最短路徑或狀態累積                           | Topological Sort + DP<br><br>Big O：通常 O(V+E)                        | 1. 補充 LC #2050 Parallel Courses III — 課程最早完成時間<br><br>2. 補充 LC #1857 Largest Color Value in a Directed Graph — 路徑中的最大顏色次數<br><br>3. 補充 LC #329 Longest Increasing Path in a Matrix — DAG 最長遞增路徑 |

這條規則與前面七條不同：它是演算法策略選擇規則，不是純資料結構選擇規則。

例如：
- Dijkstra 需要 Heap 提升取出最小距離節點的效率。
- BFS 需要 Queue 維持逐層探索順序。
- DP 可以使用 Array、Hash Map 或 Tree 等不同結構儲存狀態。
- Binary Search on Answer 往往需要搭配 Greedy、DFS 或其他 Feasibility Check。

也就是說，先選 Algorithm，再選支援這個 Algorithm 的 Data Structure，通常比只看到「最大值」就想到 Heap 或 DP 更可靠。

# 三、12 種 Data Structure 對應到 8 Rules 的完整整理

這個表可以當作你原本 12 種資料結構的總索引。每種資料結構都能對應到至少一條 Rule。

|#|Data Structure|核心用途|對應 Rules|
|---|---|---|---|
|1|Array / Prefix Sum|Random Access、靜態區間總和|2、4|
|2|Linked List / Doubly Linked List|已知節點位置的快速插入／刪除、節點指標操作|2|
|3|Queue / Deque / Monotonic Queue|FIFO、BFS、Sliding Window Extreme|2、3、4、6、7、8|
|4|Stack / Monotonic Stack|LIFO、DFS、Next Greater / Smaller|2、5、6|
|5|Heap|Global Min / Max、Top K、Priority|3、8|
|6|Binary Tree / BST|Hierarchical Traversal、Ordered Search|1、6|
|7|Hash Map / Hash Set|Exact Lookup、Frequency、Prefix Sum Lookup|1、4|
|8|Trie|Prefix Search、Dictionary|1|
|9|Union Find|Dynamic Connectivity、Components|7|
|10|Segment Tree|Dynamic Range Query、Range Update|3、4|
|11|Balanced Tree|Dynamic Ordered Data、Floor / Ceiling、Rank（需擴充）|1、3、5、6|
|12|BIT / Fenwick Tree|Dynamic Prefix / Range Sum、Frequency Counting|4、5|

這樣你的 12 種資料結構就全部整合進來，而且不會因為某個資料結構跨越多種題型而被錯誤地限制在單一 Rule。

# 四、比題目關鍵字更重要：四個決策問題

我建議在看到一道新題目時，先問自己以下四個問題，再套用八條 Rules。

STEP 1
What to Query?
Membership / Ordered Search / Sum / Max / Rank / Connectivity / Path

STEP 2
How Does Data Change?
Static / Append Only / Sliding Window / Point Update / Range Update / Delete

STEP 3
What Structure or Property Exists?
Sorted / Monotonic / Tree / Graph / Prefix / Contiguous

STEP 4
Choose Algorithm + Data Structure
確認目標時間複雜度、空間限制與正確性
例如同樣是問 Maximum，因為 Update 與 Query 的方式不同，解法就會不同：

|題目要求|選擇|
|---|---|
|靜態 Array 找一次最大值|Linear Scan|
|動態插入資料，隨時取最大值|Max Heap|
|動態插入／任意刪除，隨時查最大值|Balanced Tree / Multiset|
|找每個固定 Sliding Window 最大值|Monotonic Deque|
|任意 Range Maximum + Point Update|Segment Tree|
|找每個元素右邊第一個 Greater Element|Monotonic Stack|
|找可行解中的最大收益|視問題性質選擇 DP / Greedy / 其他最佳化演算法|

這也是為什麼不能把「Max / Min」直接等同於 Heap。

# 五、用實際 LeetCode 題目驗證這套規則

以下是精選的 12 個典型情境，可以用來練習「由需求反推資料結構」。

|LeetCode 題目|主要判斷條件|最典型解法|
|---|---|---|
|1. Two Sum|未排序、Exact Lookup|Hash Map|
|206. Reverse Linked List|Linked Node Pointer Manipulation|Linked List + Pointers|
|20. Valid Parentheses|LIFO、Nested Structure|Stack|
|239. Sliding Window Maximum|固定滑動視窗極值|Monotonic Deque|
|215. Kth Largest Element|Top K / Priority|Min Heap（或 Quickselect）|
|98. Validate Binary Search Tree|Tree + Ordered Invariant|DFS|
|208. Implement Trie|Prefix Search|Trie|
|684. Redundant Connection|Undirected Cycle Detection|Union Find|
|307. Range Sum Query – Mutable|Point Update + Range Sum|BIT / Segment Tree|
|315. Count of Smaller Numbers After Self|Relative Order + Counting|BIT / Merge Sort|
|450. Delete Node in a BST|Ordered Tree + Node Deletion|BST|
|23. Merge k Sorted Lists|每次取 K 個候選中的最小值|Min Heap|

這些例子還能看出一件事：同一題往往不只有一種正確解法。選擇時應比較時間、空間、實作成本，以及輸入規模。

# 六、建議記下來的快速判斷表

這張是我最推薦在面試準備時反覆練習的版本。

|看到的關鍵需求|第一個應考慮的解法|
|---|---|
|Exact Key / Frequency|Hash Map|
|Sorted Array Search|Binary Search|
|Dynamic Ordered Search|Balanced BST|
|String Prefix|Trie|
|Random Access|Array|
|Node Insertion / Deletion|Linked List|
|FIFO / Level Order|Queue|
|LIFO / Nested Structure|Stack|
|Nearest Greater / Smaller|Monotonic Stack|
|Sliding Window Max / Min|Monotonic Deque|
|Global Min / Max、Top K|Heap|
|Static Range Sum|Prefix Sum|
|Dynamic Range Sum|BIT|
|Dynamic Range Min / Max|Segment Tree|
|Range Update + Range Query|Lazy Segment Tree / 特定 BIT 技巧|
|Inversion Count|Merge Sort / BIT|
|Dynamic Rank / Predecessor / Successor|Augmented Balanced BST|
|Tree Traversal|DFS / BFS|
|Dynamic Undirected Connectivity（只加邊）|Union Find|
|Unweighted Shortest Path|BFS|
|Non-negative Weighted Shortest Path|Dijkstra + Heap|
|Overlapping Subproblems + Optimal Substructure|DP|
|Monotonic Feasibility|Binary Search on Answer|

# 七、最終建議：你的筆記應該如何組織？

我建議保留你原本的 12 個 Data Structure 章節，因為這些章節適合學習每種結構的 Implementation、Operations、Time / Space Complexity。

但在最前面增加一章 How to Choose — 8 Decision Rules，作為所有題目的入口。

最終版 8 Rules

|   |   |
|---|---|
|Rule 1|Search / Find / Lookup|
|Rule 2|Sequence / Order / Processing|
|Rule 3|Max / Min / Priority / Top K|
|Rule 4|Subarray / Range Query / Update|
|Rule 5|Relative Position / Rank / Order Statistics|
|Rule 6|Tree / Hierarchy / Traversal|
|Rule 7|Graph / Connectivity / Components|
|Rule 8|Optimization / Shortest Path / Feasibility|

如果只希望保留原本五條，也能把 Rule 2、6、8 併入其他規則，但會犧牲分類清晰度。我認為八條是更合理的平衡：既能完整涵蓋 12 種資料結構，又不會拆成過多零碎的規則。

最重要的解題心法是：

Problem Requirements → Query Type → Update Pattern → Structural Properties → Algorithm → Data Structure → Complexity

比起背誦「看到 Max 就用 Heap」、「看到 Subarray 就用 Sliding Window」，這套流程更適合 Medium / Hard 題目，也更符合 Senior Engineer 面試時對演算法選擇與技術取捨的期待。