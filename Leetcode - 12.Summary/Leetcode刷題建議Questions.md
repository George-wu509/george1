
|                                     |     |
| ----------------------------------- | --- |
| [[#### 2026科技大廠刷題]]                 |     |
| [[#### Blind 75**加上**NeetCode 150]] |     |
|                                     |     |
#### 2026科技大廠刷題
```
我已經安裝了conjudge, 我在裡面有看到basic50, blind 75, neetcode 150, neetcode250, leetcode1000的題庫. 如果我目標是2026/2027三個月準備完要通過interview美國科技大廠(google, meta, apple, nvidia..), 至少要刷過哪些題目
```
### **2026–2027 美國科技大廠面試：3 個月 LeetCode 刷題計畫**

針對你想應徵的 Google、Meta、Apple、NVIDIA，以及 Senior Software Engineer、Computer Vision Engineer、AI Engineer 等職位，我建議：

**以 Cojudge 的 NeetCode 150 為主要題庫，Blind 75 為第一階段核心，再利用 NeetCode 250 和 LeetCode 1000 補強弱點。**

你不需要在三個月內把五個題庫全部刷完。最重要的是能在面試時獨立辨識演算法、寫出正確的程式、分析複雜度，並清楚解釋最佳化的過程。

## **一、究竟需要刷多少題？**

最低目標

# **75**

Blind 75

建議目標

# **150**

NeetCode 150

加強目標

# **180–200**

公司與弱點專項

**我的建議：三個月掌握約 120–150 道不同題目，另外安排 80–120 次舊題重做，並進行 8–12 場模擬 Coding Interview。**

以上是依三個月準備期制定的學習目標，不是任何公司官方公布的通過門檻。

如果你每天可以安排 2–3 小時、每週六天練習，三個月完成這個目標是有機會的。如果你每天只有一小時，則應優先完成 Blind 75，再補最重要的 30–40 題。

## **二、Cojudge 五個題庫怎麼選？**

|題庫|我的建議|使用方式|
|---|---|---|
|Basic 50|選用|如果基本 Array、Hash Map、Tree 不熟，先暖身|
|**Blind 75**|**必刷**|第一階段，建立最核心的演算法能力|
|**NeetCode 150**|**主要目標**|第二階段，完成剩下的題型與進階題|
|NeetCode 250|選擇性|補容易失分的基礎題或特定演算法|
|LeetCode 1000|選擇性|最後針對公司、主題與難度加練|

這些題庫並非互不重複。

根據 NeetCode 官方，NeetCode 150 是在 Blind 75 基礎上增加 75 題；NeetCode 250 又在 NeetCode 150 基礎上增加 100 題。因此，不應把刷完三個題庫理解成完成 475 道不同題目。

Coding Interview Questions+1

NeetCode 250

250 題

NeetCode 150

150 題

Blind 75

75 題

第一個月優先完成的核心題型

額外 75 題：擴充演算法涵蓋範圍

額外 100 題：進一步練習與強化

以上表示 NeetCode 官方題單的包含關係；Basic 50 與 LeetCode 1000 則另外依題號去重。

## **三、四家公司的準備方向有什麼差別？**

|公司|Coding 準備重點|我建議的調整|
|---|---|---|
|Google|DSA 廣度、Graph、Tree、DP、複雜度及 Follow-up|NeetCode 150 全面覆蓋|
|Meta|Array、Hash Map、Tree、Graph、速度、程式碼正確性|Blind 75 + 額外 Medium 計時訓練|
|Apple|視團隊而定，演算法、程式能力、系統及專業領域|NeetCode 150 + C++ / Imaging 專項|
|NVIDIA|Coding + C++、演算法效能、平行處理或 GPU 等職務知識|NeetCode 150 + C++ / Performance 專項|

這是準備策略，不是各家公司固定的出題頻率排名。

有兩個值得注意的官方資訊：

- **Meta** 的 Software Engineer 面試指南指出，Coding Interview 通常應準備約 40 分鐘解兩題，而且明確建議廣度優先。該指南甚至表示一般 SWE Coding Interview 不問 Dynamic Programming；不過如果同時準備 Google，就不能完全跳過 DP。
    
    CloudFront
    
- **NVIDIA** 官方招聘說明指出，技術職位可能有 Coding Exercise，並依職務安排不同的技術面談，因此不能只依靠 LeetCode。
    
    NVIDIA
    

接下來我把 NeetCode 150 中最值得優先刷的題目，依演算法分類、LeetCode 題號及十二週進度整理出來。

## **四、我建議優先完成的 120 道經典題目**

以下按照 NeetCode 150 的分類，選出我建議優先掌握的 **120 道核心題目**。

這不是任何公司的官方必考題，而是為三個月準備期挑選的跨公司高價值練習清單。題號均為 LeetCode 題號，你可以直接在 Cojudge 搜尋英文題名。題型分類參考 NeetCode 官方題庫。

GitHub+1

### **A. Array、Two Pointers、Sliding Window（20 題）**

|題型|必刷題目|核心演算法|
|---|---|---|
|Array & Hashing（9）|217. Contains Duplicate|Hash Set|
||242. Valid Anagram|Frequency Counter|
||1. Two Sum|Hash Map|
||49. Group Anagrams|Hash Map|
||347. Top K Frequent Elements|Bucket Sort / Heap|
||238. Product of Array Except Self|Prefix / Suffix Product|
||36. Valid Sudoku|Hash Set|
||128. Longest Consecutive Sequence|Hash Set|
||271. Encode and Decode Strings|String Encoding|
|Two Pointers（5）|125. Valid Palindrome|Two Pointers|
||167. Two Sum II|Two Pointers|
||15. 3Sum|Sorting + Two Pointers|
||11. Container With Most Water|Two Pointers|
||42. Trapping Rain Water|Two Pointers / Prefix Max|
|Sliding Window（6）|121. Best Time to Buy and Sell Stock|One-pass|
||3. Longest Substring Without Repeating Characters|Sliding Window + Set|
||424. Longest Repeating Character Replacement|Sliding Window|
||567. Permutation in String|Frequency Window|
||76. Minimum Window Substring|Sliding Window + Counter|
||239. Sliding Window Maximum|Monotonic Deque|

這一組非常重要，尤其是 Array、Hash Map、Two Pointers 和 Sliding Window。你應該熟悉從 O(n²) Brute Force 改善為 O(n) 的常見方法。

### **B. Stack、Binary Search、Linked List（23 題）**

|題型|必刷題目|核心演算法|
|---|---|---|
|Stack（6）|20. Valid Parentheses|Stack|
||155. Min Stack|Stack Design|
||150. Evaluate Reverse Polish Notation|Stack|
||22. Generate Parentheses|Backtracking|
||739. Daily Temperatures|Monotonic Stack|
||853. Car Fleet|Sorting + Stack|
|Binary Search（7）|704. Binary Search|Binary Search|
||74. Search a 2D Matrix|Binary Search|
||875. Koko Eating Bananas|Binary Search on Answer|
||153. Find Minimum in Rotated Sorted Array|Modified Binary Search|
||33. Search in Rotated Sorted Array|Modified Binary Search|
||981. Time Based Key-Value Store|Hash Map + Binary Search|
||4. Median of Two Sorted Arrays|Binary Search Partition|
|Linked List（10）|206. Reverse Linked List|Pointer Manipulation|
||21. Merge Two Sorted Lists|Two Pointers|
||141. Linked List Cycle|Fast / Slow Pointers|
||143. Reorder List|Reverse + Merge|
||19. Remove Nth Node From End|Two Pointers|
||138. Copy List with Random Pointer|Hash Map / Node Linking|
||2. Add Two Numbers|Linked List Simulation|
||146. LRU Cache|Hash Map + Doubly Linked List|
||23. Merge k Sorted Lists|Heap / Divide and Conquer|
||287. Find the Duplicate Number|Floyd's Cycle Detection|

特別推薦你熟練：

- **LC 739**：辨識 Monotonic Stack。
    
- **LC 875**：辨識 Binary Search on Answer。
    
- **LC 146**：Data Structure Design，在 Senior SWE 面試也很有練習價值。
    

### **C. Tree、Trie、Backtracking（23 題）**

|題型|必刷題目|核心演算法|
|---|---|---|
|Tree（13）|226. Invert Binary Tree|DFS|
||104. Maximum Depth of Binary Tree|DFS / BFS|
||543. Diameter of Binary Tree|Postorder DFS|
||110. Balanced Binary Tree|DFS|
||100. Same Tree|DFS|
||572. Subtree of Another Tree|DFS|
||102. Binary Tree Level Order Traversal|BFS|
||199. Binary Tree Right Side View|BFS|
||98. Validate Binary Search Tree|DFS / Bounds|
||230. Kth Smallest Element in a BST|Inorder Traversal|
||105. Construct Binary Tree from Preorder and Inorder|Recursion + Hash Map|
||124. Binary Tree Maximum Path Sum|Tree DP|
||297. Serialize and Deserialize Binary Tree|BFS / DFS|
|Trie（2）|208. Implement Trie|Prefix Tree|
||211. Design Add and Search Words|Trie + DFS|
|Backtracking（8）|78. Subsets|Backtracking|
||39. Combination Sum|Backtracking|
||46. Permutations|Backtracking|
||90. Subsets II|Backtracking + Deduplication|
||40. Combination Sum II|Backtracking|
||79. Word Search|Grid DFS / Backtracking|
||131. Palindrome Partitioning|Backtracking|
||17. Letter Combinations of a Phone Number|Backtracking|

這一組的核心是理解 DFS、BFS、Recursion 和 Backtracking 之間的差別。

例如 LC 124，你必須知道 DFS 回傳給父節點的值，和用來更新全域最佳答案的值，並不一定相同。

### **D. Heap、Graph、Advanced Graph（20 題）**

|題型|必刷題目|核心演算法|
|---|---|---|
|Heap（6）|703. Kth Largest Element in a Stream|Min Heap|
||1046. Last Stone Weight|Max Heap|
||973. K Closest Points to Origin|Heap / Quickselect|
||215. Kth Largest Element in an Array|Heap / Quickselect|
||621. Task Scheduler|Greedy / Heap|
||295. Find Median from Data Stream|Two Heaps|
|Graph（10）|200. Number of Islands|Grid DFS / BFS|
||695. Max Area of Island|DFS|
||133. Clone Graph|DFS + Hash Map|
||994. Rotting Oranges|Multi-source BFS|
||417. Pacific Atlantic Water Flow|Reverse DFS / BFS|
||130. Surrounded Regions|Boundary DFS|
||207. Course Schedule|Topological Sort|
||210. Course Schedule II|Topological Sort|
||684. Redundant Connection|Union Find|
||127. Word Ladder|BFS|
|Advanced Graph（4）|332. Reconstruct Itinerary|Eulerian Path|
||1584. Min Cost to Connect All Points|MST|
||743. Network Delay Time|Dijkstra|
||787. Cheapest Flights Within K Stops|Bellman-Ford / State-aware Shortest Path|

這組尤其適合 Google，以及各種需要影像、圖形、資料處理能力的 Engineer 職位。

你至少要能不看答案，直接寫出：

- BFS：`collections.deque`
    
- DFS：遞迴與迭代版本
    
- Topological Sort：Kahn's Algorithm
    
- Union Find：Path Compression + Union by Rank/Size
    
- Dijkstra：`heapq`
    

### **E. Dynamic Programming、Greedy（22 題）**

|題型|必刷題目|核心演算法|
|---|---|---|
|1D DP（10）|70. Climbing Stairs|DP|
||198. House Robber|1D DP|
||213. House Robber II|Circular DP|
||5. Longest Palindromic Substring|Expand Around Center / DP|
||91. Decode Ways|1D DP|
||322. Coin Change|Unbounded Knapsack DP|
||152. Maximum Product Subarray|Max / Min DP|
||139. Word Break|DP|
||300. Longest Increasing Subsequence|DP / Binary Search|
||416. Partition Equal Subset Sum|0/1 Knapsack|
|2D DP（6）|62. Unique Paths|Grid DP|
||1143. Longest Common Subsequence|2D DP|
||309. Best Time to Buy and Sell Stock with Cooldown|State DP|
||518. Coin Change II|Knapsack DP|
||494. Target Sum|DP|
||72. Edit Distance|2D DP|
|Greedy（6）|53. Maximum Subarray|Kadane's Algorithm|
||55. Jump Game|Greedy|
||45. Jump Game II|Greedy|
||134. Gas Station|Greedy|
||763. Partition Labels|Greedy|
||678. Valid Parenthesis String|Greedy / Range Tracking|

建議學習 DP 時，不要直接背答案，而是對每一題回答：

1. `dp[i]` 或 `dp[i][j]` 代表什麼？
    
2. State Transition 怎麼推導？
    
3. Base Case 是什麼？
    
4. 能否把空間從 O(n) 或 O(mn) 改善到 O(1) 或 O(n)？
    

### **F. Intervals、Bit Manipulation、Math（12 題）**

|題型|必刷題目|核心演算法|
|---|---|---|
|Intervals（5）|57. Insert Interval|Interval Merge|
||56. Merge Intervals|Sorting|
||435. Non-overlapping Intervals|Greedy|
||252. Meeting Rooms|Sorting|
||253. Meeting Rooms II|Heap / Sweep Line|
|Bit Manipulation（3）|136. Single Number|XOR|
||191. Number of 1 Bits|Bit Counting|
||338. Counting Bits|Bit DP|
|Math / Matrix（4）|48. Rotate Image|Matrix Transpose + Reverse|
||54. Spiral Matrix|Matrix Traversal|
||73. Set Matrix Zeroes|In-place Matrix|
||50. Pow(x, n)|Fast Exponentiation|

這些題目雖然單一分類題數較少，但能補齊演算法知識。部分例如 LC 252、253、271 在 LeetCode 屬於 Premium 題目；Cojudge 是否提供，請以你安裝版本的本地題庫為準。

以上共 **120 道不同題目**。

## **五、你之前練習過的 Prefix Sum 與 Monotonic Queue，還應加刷哪些？**

NeetCode 150 很完整，但 **Prefix Sum 並不是它的獨立分類**。如果你已經在練習 LintCode 的 Prefix Sum、Non-negative Prefix Sum 和 Shortest Subarray，我特別建議從 Cojudge 的 LeetCode 1000 題庫額外找這幾題。

|題目|核心演算法|Time|
|---|---|---|
|LC 303. Range Sum Query – Immutable|Prefix Sum|建立 O(n)，查詢 O(1)|
|LC 560. Subarray Sum Equals K|Prefix Sum + Hash Map|O(n)|
|LC 209. Minimum Size Subarray Sum|Sliding Window（正整數）|O(n)|
|LC 862. Shortest Subarray with Sum at Least K|Prefix Sum + Monotonic Deque|O(n)|
|LC 239. Sliding Window Maximum|Monotonic Deque|O(n)|

其中 LC 239 已計入上面 120 題，所以如果五題都完成，實際只增加四道不同題目。

建議把 LC 209、560、862 放在同一週比較。你會更清楚為什麼有負數時，一般 Sliding Window 不再適用，以及 Monotonic Deque 如何維持候選 Prefix Sum 的有效性。

## **六、三個月（12 週）具體安排**

以 2026 年 11 月開始準備，目標約 2027 年 2 月開始進行面試為例：

12 週刷題目標

第 1–10 週完成核心 120 題

10 週核心 + 2 週衝刺

核心 DSA

DP / Greedy

Intervals / Math

|週次|主題|新題目標|
|---|---|---|
|Week 1|Array & Hashing + Two Pointers|12|
|Week 2|Two Pointers + Sliding Window + Stack|12|
|Week 3|Stack + Binary Search + Linked List|12|
|Week 4|Linked List + Tree|12|
|Week 5|Tree + Trie + Backtracking|12|
|Week 6|Backtracking + Heap|12|
|Week 7|Graph + Advanced Graph|12|
|Week 8|Advanced Graph + 1D DP|12|
|Week 9|2D DP + Greedy|12|
|Week 10|Intervals + Bit Manipulation + Math|12|
|Week 11|強化弱點、額外 10–15 題、Mock Interview|視進度|
|Week 12|公司專項、額外題目、Mock Interview|視進度|

第 1–10 週的 120 題就是上面分類的題目，並按表格順序往下做。第 11–12 週的新增題目則優先從 NeetCode 150 尚未完成的部分選擇，而不是隨機打開 LeetCode 1000。

**重要：複習從第一週就要開始，不是第十一週才重做舊題。**

### **每天 2–3 小時該怎麼分配？**

依每日可用時間調整練習

1.5 小時2.5 小時3 小時

新題 2 題

60 分鐘

複習舊題 1–2 題

30 分鐘

解法比較與 Big-O

25 分鐘

面試口頭表達

15 分鐘

系統設計／專業題

20 分鐘

每週練習六天，第七天用來檢查進度、補落後題目或休息。

### **如何判定一道題「真的學會」？**

我會建立比 Cojudge Pass / Fail 更嚴格的標準：

|評估項目|達標條件|
|---|---|
|Correctness|Cojudge Submit 通過|
|Algorithm Recognition|能說出為什麼選這個演算法|
|Time / Space Complexity|不查資料就能正確分析|
|Optimization|能比較 Brute Force 與最佳解|
|Edge Cases|自己提出並驗證至少 2–3 個案例|
|Speed|多數未看過的 Medium 題在 20–25 分鐘完成|
|Retention|7 天、21 天後仍能重新獨立解出|

NeetCode 作者在 2026 年更新的準備指南中，也把「多數陌生 Medium 題在 20–25 分鐘內解出」視為實用的面試準備指標。

NeetCode

## **七、第 11–12 週：Google 與 Meta 的訓練方式不同**

對 Google，我建議保留完整的 DP、Graph、Advanced Graph 練習，並做 Follow-up，例如把輸入規模從一萬增加到一億、允許負數、要求使用更少記憶體等。

對 Meta，我會把後期更多時間分配給 **40 分鐘內解兩題** 的計時練習。

例如：

**Meta-style Mock Coding Round**

40 分鐘

LC 49. Group Anagrams

18 分鐘

LC 102. Binary Tree Level Order Traversal

18 分鐘

最後檢查與 Edge Cases

4 分鐘

這是我設計的模擬題組和時間分配，不代表 Meta 真的使用這兩題。

同時要練習不依靠 Cojudge 自動測試。在真正的面試中，應能自己 Dry Run，並且一邊寫程式一邊解釋思路。這也符合 Meta 官方面試指南的準備建議。

CloudFront

## **八、如果目標是 Senior Software Engineer / AI / Computer Vision，LeetCode 以外還要準備什麼？**

這部分非常重要。

即使你能完成全部 NeetCode 150，也不代表已經準備好 Google L5、Meta E5，或 Apple、NVIDIA 的 Senior 等級完整面試。

我建議把三個月的總準備時間大致分成：

Senior 技術面試準備時間配置（建議）

除了刷題，應該同步準備：

|項目|典型問題|
|---|---|
|C++ / Python|Memory Management、Multithreading、GIL、STL|
|Concurrency|Mutex、Thread Safety、Producer–Consumer、Race Condition|
|Computer Vision|Image Processing、Detection、Segmentation、Camera Calibration|
|AI / ML|PyTorch、Training、Inference、Quantization、Evaluation|
|System Design|Scalable Image Processing Pipeline、Distributed Job Queue|
|ML System Design|Model Serving、Data Pipeline、Monitoring、Retraining|
|Behavioral|Technical Leadership、Ownership、Conflict、Technical Decisions|

尤其如果你主攻 Imaging / Computer Vision 和 AI 系統職位，實際影像處理管線與模型部署經驗，可以成為 System Design 和 Technical Deep Dive 的核心素材。Apple 的招聘經驗整理也顯示，面試內容會隨團隊領域明顯不同。

Career Management Center UCSD Rady

## **九、最後：三個月完成到什麼程度才值得開始面試？**

我的建議是使用以下檢查表作為目標：

已完成 Blind 75

已完成核心 120 題

主要題型都能獨立辨識

至少 80% 的陌生 Medium 題能在 25 分鐘內完成

能正確解釋 Time / Space Complexity

已重做至少 80 次舊題

完成至少 8 場計時 Mock Interview

準備好 5–8 個 System Design / ML Design 案例

準備好 Leadership、Ownership、Conflict 的 STAR 故事

準備檢查進度

0 / 9

這不是保證通過面試的標準，但比單純統計解過幾題更有意義。

**最終我會選擇：Cojudge NeetCode 150 作為唯一主要刷題清單，先完成上述 120 題，剩下的時間反覆練習弱點和陌生 Medium 題。** 不需要為了數量而急著開始 LeetCode 1000。

另外，Cojudge 官方也提醒，它是離線練習用的次級 Judge，題目和測資仍可能有錯漏。有網路時，可以把重要題目再拿到 LeetCode 線上提交驗證。

GitHub

如果三個月後你能在沒有提示的情況下，快速而清楚地解出陌生題目，並把省下的時間投入 System Design、AI / CV Domain Knowledge 和 Behavioral Preparation，會比只追求「刷滿 300 題」更符合你的求職目標。



#### Blind 75**加上**NeetCode 150
```
請將全部的**Blind 75**加上**NeetCode 150地所以題目的中文英文跟號碼**, 按照以下的分類歸類:

|1. Array|
|---|
|2. String|
|3. Prefix Sum Array|
|4. Sorting|
|5. Binary Search|
|6. Two Pointers|
|7. Linked List|
|8. Queue/deque|
|9. Monotone queue|
|10. BFS|
|11. DFS|
|12. Stack|
|13. Monotonic stack|
|14. Hash Map|
|15. Union find|
|16. Iterator|
|17. Binary Tree|
|18. Binary Search Tree|
|19. Heap|
|20. Trie|
|21. Segment Tree|
|22. Balanced Binary Tree|
|23. Binary Indexed Tree|
|24. Graph|
|25. Dp|
|26. Math|
```

### **Blind 75 + NeetCode 150 完整題目分類表（26 類）**

我已核對 [Blind 75](https://neetcode.io/practice/practice/blind75?utm_source=chatgpt.com)、[NeetCode 150](https://neetcode.io/practice/practice/neetcode150?utm_source=chatgpt.com) 的題庫範圍，以及包含逐題清單的對照資料。這裡採用 **NeetCode 官網版本**，它包含全部 Blind 75，再增加 75 題。

Coding Interview Questions+1

Blind 75

# **75**

NeetCode 150

# **150**

去重後

# **150**

整理規則如下：

- 所有題目都有 **LeetCode 題號、繁體中文名稱、英文原名**。
    
- 有 **★** 的代表屬於 Blind 75，同時也屬於 NeetCode 150；沒有 ★ 則是額外 75 題。
    
- 每題歸入一個**主要分類**，確保 150 題不重複計數。跨類別的題目會在最後補充 Secondary Tags。
    
- 原始 NeetCode 使用 18 個題型分類；以下依你指定的 26 類重新編排。因此這不是 NeetCode 官網原始的分類順序。
    

有幾項分類要先區分：Sliding Window、Backtracking、Greedy、Bit Manipulation 不在你指定的 26 個分類名稱中，我會分別按照主要資料結構或演算法歸入其他類別。另外，Prefix Sum 和 Prefix Product 並不相同，不能直接混為一談。

## **1. Array（陣列）— 13 題**

包含陣列走訪、Matrix、Greedy 與區間操作。

|題號|中文題名|English Title|
|---|---|---|
|36|有效的數獨|Valid Sudoku|
|45|跳躍遊戲 II|Jump Game II|
|48 ★|旋轉影像|Rotate Image|
|53 ★|最大子陣列和|Maximum Subarray|
|54 ★|螺旋矩陣|Spiral Matrix|
|55 ★|跳躍遊戲|Jump Game|
|57 ★|插入區間|Insert Interval|
|66|加一|Plus One|
|73 ★|矩陣置零|Set Matrix Zeroes|
|121 ★|買賣股票的最佳時機|Best Time to Buy and Sell Stock|
|134|加油站|Gas Station|
|238 ★|除自身以外陣列的乘積|Product of Array Except Self|
|1899|合併三元組以形成目標三元組|Merge Triplets to Form Target Triplet|

LC 238 是 Prefix/Suffix Product，不是 Prefix Sum；LC 53、55、45、134、1899 等也可以進一步歸類為 Greedy 或 Dynamic Programming 的相關練習。

## **2. String（字串）— 9 題**

|題號|中文題名|English Title|
|---|---|---|
|3 ★|無重複字元的最長子字串|Longest Substring Without Repeating Characters|
|76 ★|最小涵蓋子字串|Minimum Window Substring|
|125 ★|驗證回文串|Valid Palindrome|
|242 ★|有效的字母異位詞|Valid Anagram|
|271 ★|字串的編碼與解碼|Encode and Decode Strings|
|424 ★|替換後最長重複字元|Longest Repeating Character Replacement|
|567|字串的排列|Permutation in String|
|678|有效的括號字串|Valid Parenthesis String|
|763|劃分字母區間|Partition Labels|

這組還可以分成 Sliding Window（3、76、424、567）、Two Pointers（125）、Hash Map（242）和 Greedy（678、763）。

## **3. Prefix Sum Array（前綴和陣列）— 0 題**

在這份 NeetCode 150 清單中，沒有以標準 Prefix Sum Array 為核心的獨立題目。

但考慮到這是很重要的面試技巧，我建議額外從 Cojudge LeetCode 1000 尋找以下題目：

|題號|中文題名|English Title|
|---|---|---|
|303|區域和檢索－陣列不可變|Range Sum Query – Immutable|
|560|和為 K 的子陣列|Subarray Sum Equals K|
|525|連續陣列|Contiguous Array|
|523|連續的子陣列和|Continuous Subarray Sum|
|862|和至少為 K 的最短子陣列|Shortest Subarray with Sum at Least K|

這 5 題是額外推薦，不計入原始 150 題，也不保證全部存在於你的 Cojudge 本地版本中。

## **4. Sorting（排序）— 4 題**

|題號|中文題名|English Title|
|---|---|---|
|56 ★|合併區間|Merge Intervals|
|252 ★|會議室|Meeting Rooms|
|435 ★|無重疊區間|Non-overlapping Intervals|
|846|一手順子|Hand of Straights|

這四題主要訓練排序後的掃描、Greedy，以及區間處理。

## **5. Binary Search（二分搜尋）— 7 題**

|題號|中文題名|English Title|
|---|---|---|
|4|尋找兩個正序陣列的中位數|Median of Two Sorted Arrays|
|33 ★|搜尋旋轉排序陣列|Search in Rotated Sorted Array|
|74|搜尋二維矩陣|Search a 2D Matrix|
|153 ★|尋找旋轉排序陣列中的最小值|Find Minimum in Rotated Sorted Array|
|704|二分搜尋|Binary Search|
|875|珂珂吃香蕉|Koko Eating Bananas|
|981|基於時間的鍵值儲存|Time Based Key-Value Store|

必須掌握一般 Binary Search、Rotated Sorted Array、Binary Search on Answer。

## **6. Two Pointers（雙指標）— 4 題**

|題號|中文題名|English Title|
|---|---|---|
|11 ★|盛最多水的容器|Container With Most Water|
|15 ★|三數之和|3Sum|
|42|接雨水|Trapping Rain Water|
|167|兩數之和 II－輸入有序陣列|Two Sum II – Input Array Is Sorted|

Two Pointers 的其他重要題目還包括 String 分類的 LC 125，以及 Linked List 分類的 LC 141、143、287。

## **7. Linked List（鏈結串列）— 11 題**

|題號|中文題名|English Title|
|---|---|---|
|2|兩數相加|Add Two Numbers|
|19 ★|刪除鏈結串列的倒數第 N 個節點|Remove Nth Node From End of List|
|21 ★|合併兩個排序鏈結串列|Merge Two Sorted Lists|
|23 ★|合併 K 個排序鏈結串列|Merge k Sorted Lists|
|25|K 個一組翻轉鏈結串列|Reverse Nodes in k-Group|
|138|複製帶隨機指標的鏈結串列|Copy List with Random Pointer|
|141 ★|環狀鏈結串列|Linked List Cycle|
|143 ★|重排鏈結串列|Reorder List|
|146|LRU 快取機制|LRU Cache|
|206 ★|反轉鏈結串列|Reverse Linked List|
|287|尋找重複數|Find the Duplicate Number|

LC 287 雖然輸入是陣列，但可以把數值視為下一個節點的索引，使用 Floyd's Cycle Detection，所以這裡歸入 Linked List 思維。

## **8. Queue/deque（佇列／雙端佇列）— 2 題**

|題號|中文題名|English Title|
|---|---|---|
|102 ★|二元樹的層序遍歷|Binary Tree Level Order Traversal|
|199|二元樹的右視圖|Binary Tree Right Side View|

這兩題都使用 Queue 執行 BFS；Python 可使用 `collections.deque`。

## **9. Monotone Queue（單調佇列）— 1 題**

|題號|中文題名|English Title|
|---|---|---|
|239|滑動視窗最大值|Sliding Window Maximum|

這是經典的 Monotonic Deque 題目。

它的核心是維護索引的單調性，讓整個陣列的處理時間達到 O(n)。

## **10. BFS（廣度優先搜尋）— 3 題**

|題號|中文題名|English Title|
|---|---|---|
|127|單字接龍|Word Ladder|
|286|牆與門|Walls and Gates|
|994|腐爛的橘子|Rotting Oranges|

這三題分別涵蓋：

- 最短路徑 BFS
    
- Multi-source BFS
    
- BFS Level Traversal
    

另外，LC 102、199、207、210 也非常適合一起練習 BFS。

## **11. DFS（深度優先搜尋）— 14 題**

這裡包含 Recursive DFS、Grid DFS 與 Backtracking。

|題號|中文題名|English Title|
|---|---|---|
|17|電話號碼的字母組合|Letter Combinations of a Phone Number|
|22|括號生成|Generate Parentheses|
|39 ★|組合總和|Combination Sum|
|40|組合總和 II|Combination Sum II|
|46|全排列|Permutations|
|51|N 皇后|N-Queens|
|78|子集|Subsets|
|79 ★|單字搜尋|Word Search|
|90|子集 II|Subsets II|
|130|被圍繞的區域|Surrounded Regions|
|131|分割回文串|Palindrome Partitioning|
|200 ★|島嶼數量|Number of Islands|
|417 ★|太平洋大西洋水流問題|Pacific Atlantic Water Flow|
|695|島嶼的最大面積|Max Area of Island|

建議把這類分成兩組來練：

**Grid DFS / Flood Fill：** 130、200、417、695、79。

**Backtracking：** 17、22、39、40、46、51、78、90、131，以及 79。

## **12. Stack（堆疊）— 3 題**

|題號|中文題名|English Title|
|---|---|---|
|20 ★|有效的括號|Valid Parentheses|
|150|逆波蘭表示式求值|Evaluate Reverse Polish Notation|
|155|最小堆疊|Min Stack|

## **13. Monotonic Stack（單調堆疊）— 3 題**

|題號|中文題名|English Title|
|---|---|---|
|84|柱狀圖中最大的矩形|Largest Rectangle in Histogram|
|739|每日溫度|Daily Temperatures|
|853|車隊|Car Fleet|

其中 LC 739 與 84 是非常典型的 Monotonic Stack 題目；LC 853 則可用排序搭配到達時間的單調堆疊思路。

## **14. Hash Map（雜湊表）— 5 題**

|題號|中文題名|English Title|
|---|---|---|
|1 ★|兩數之和|Two Sum|
|49 ★|字母異位詞分組|Group Anagrams|
|128 ★|最長連續序列|Longest Consecutive Sequence|
|217 ★|存在重複元素|Contains Duplicate|
|347 ★|前 K 個高頻元素|Top K Frequent Elements|

這裡包括 Hash Map、Hash Set、Frequency Counter 與 Bucket Sort。

## **15. Union Find（並查集）— 3 題**

|題號|中文題名|English Title|
|---|---|---|
|261 ★|有效樹圖|Graph Valid Tree|
|323 ★|無向圖中連通分量的數量|Number of Connected Components in an Undirected Graph|
|684|冗餘連線|Redundant Connection|

核心必須熟悉 Disjoint Set Union（DSU）的 `find()`、`union()`、Path Compression 與 Union by Size/Rank。

## **16. Iterator（迭代器）— 0 題**

Blind 75 + NeetCode 150 沒有以 Iterator Design 為主要題型的題目。

建議額外練習：

|題號|中文題名|English Title|
|---|---|---|
|173|二元搜尋樹迭代器|Binary Search Tree Iterator|
|284|窺視迭代器|Peeking Iterator|
|341|扁平化巢狀串列迭代器|Flatten Nested List Iterator|

額外推薦，不計入 150 題。

## **17. Binary Tree（二元樹）— 9 題**

|題號|中文題名|English Title|
|---|---|---|
|100 ★|相同的樹|Same Tree|
|104 ★|二元樹的最大深度|Maximum Depth of Binary Tree|
|105 ★|從前序與中序遍歷序列構造二元樹|Construct Binary Tree from Preorder and Inorder Traversal|
|124 ★|二元樹中的最大路徑和|Binary Tree Maximum Path Sum|
|226 ★|翻轉二元樹|Invert Binary Tree|
|297 ★|二元樹的序列化與反序列化|Serialize and Deserialize Binary Tree|
|543|二元樹的直徑|Diameter of Binary Tree|
|572 ★|另一棵樹的子樹|Subtree of Another Tree|
|1448|統計二元樹中的好節點|Count Good Nodes in Binary Tree|

這組主要使用 Recursive DFS、Postorder Traversal 和 Tree Construction。

## **18. Binary Search Tree（二元搜尋樹）— 3 題**

|題號|中文題名|English Title|
|---|---|---|
|98 ★|驗證二元搜尋樹|Validate Binary Search Tree|
|230 ★|二元搜尋樹中第 K 小的元素|Kth Smallest Element in a BST|
|235 ★|二元搜尋樹的最近共同祖先|Lowest Common Ancestor of a Binary Search Tree|

## **19. Heap（堆積／優先佇列）— 9 題**

|題號|中文題名|English Title|
|---|---|---|
|215|陣列中的第 K 大元素|Kth Largest Element in an Array|
|253 ★|會議室 II|Meeting Rooms II|
|295 ★|資料串流的中位數|Find Median from Data Stream|
|355|設計 Twitter|Design Twitter|
|621|任務排程器|Task Scheduler|
|703|資料串流中的第 K 大元素|Kth Largest Element in a Stream|
|973|最接近原點的 K 個點|K Closest Points to Origin|
|1046|最後一塊石頭的重量|Last Stone Weight|
|1851|包含每個查詢的最小區間|Minimum Interval to Include Each Query|

建議特別熟練 Python `heapq`，以及 Min Heap、Max Heap、Two Heaps 和 Top K Problems。

## **20. Trie（字首樹）— 3 題**

|題號|中文題名|English Title|
|---|---|---|
|208 ★|實作 Trie（字首樹）|Implement Trie (Prefix Tree)|
|211 ★|新增與搜尋單字－資料結構設計|Design Add and Search Words Data Structure|
|212 ★|單字搜尋 II|Word Search II|

LC 212 是 Trie + DFS / Backtracking 的重要綜合題。

## **21. Segment Tree（線段樹）— 0 題**

Blind 75 + NeetCode 150 沒有以 Segment Tree 為核心的題目。

建議額外練習：

|題號|中文題名|English Title|
|---|---|---|
|307|區域和檢索－陣列可修改|Range Sum Query – Mutable|
|729|我的行事曆 I|My Calendar I|
|715|區間模組|Range Module|

LC 307 可使用 Segment Tree 或 Binary Indexed Tree。LC 729、715 也有其他解法，不是一定要使用 Segment Tree。

## **22. Balanced Binary Tree（平衡二元樹）— 1 題**

|題號|中文題名|English Title|
|---|---|---|
|110|平衡二元樹|Balanced Binary Tree|

注意：LC 110 是判斷一般二元樹是否符合高度平衡條件，不是實作 AVL Tree 或 Red-Black Tree。

如果想練習真正的 Self-balancing BST，需要另外研究 AVL Tree 的旋轉、插入和刪除。

## **23. Binary Indexed Tree（樹狀陣列／Fenwick Tree）— 0 題**

這份題單沒有以 Binary Indexed Tree 為主要解法的題目。

建議額外練習：

|題號|中文題名|English Title|
|---|---|---|
|307|區域和檢索－陣列可修改|Range Sum Query – Mutable|
|315|計算右側小於目前元素的個數|Count of Smaller Numbers After Self|
|493|翻轉對|Reverse Pairs|

這些是額外推薦題目，不計入 150 題。

## **24. Graph（圖論）— 9 題**

|題號|中文題名|English Title|
|---|---|---|
|133 ★|複製圖|Clone Graph|
|207 ★|課程表|Course Schedule|
|210|課程表 II|Course Schedule II|
|269 ★|外星文字典|Alien Dictionary|
|332|重新安排行程|Reconstruct Itinerary|
|743|網路延遲時間|Network Delay Time|
|778|水位上升的泳池中游泳|Swim in Rising Water|
|787|K 站中轉內最便宜的航班|Cheapest Flights Within K Stops|
|1584|連接所有點的最小費用|Min Cost to Connect All Points|

這一組涵蓋 Topological Sort、Shortest Path、Minimum Spanning Tree、Eulerian Path 等進階圖論。

## **25. DP（動態規劃）— 23 題**

這是你指定的 26 類中題目數最多的一組。

|題號|中文題名|English Title|
|---|---|---|
|5 ★|最長回文子字串|Longest Palindromic Substring|
|10|正規表示式匹配|Regular Expression Matching|
|62 ★|不同路徑|Unique Paths|
|70 ★|爬樓梯|Climbing Stairs|
|72|編輯距離|Edit Distance|
|91 ★|解碼方法|Decode Ways|
|97|交錯字串|Interleaving String|
|115|不同的子序列|Distinct Subsequences|
|139 ★|單字拆分|Word Break|
|152 ★|乘積最大子陣列|Maximum Product Subarray|
|198 ★|打家劫舍|House Robber|
|213 ★|打家劫舍 II|House Robber II|
|300 ★|最長遞增子序列|Longest Increasing Subsequence|
|309|買賣股票的最佳時機（含冷凍期）|Best Time to Buy and Sell Stock with Cooldown|
|312|戳氣球|Burst Balloons|
|322 ★|零錢兌換|Coin Change|
|329|矩陣中的最長遞增路徑|Longest Increasing Path in a Matrix|
|416|分割等和子集|Partition Equal Subset Sum|
|494|目標和|Target Sum|
|518|零錢兌換 II|Coin Change II|
|647 ★|回文子字串|Palindromic Substrings|
|746|使用最小花費爬樓梯|Min Cost Climbing Stairs|
|1143 ★|最長共同子序列|Longest Common Subsequence|

建議依 DP 的五個子類練習：

|DP 分類|題號|
|---|---|
|1D DP|70、746、198、213、91、322|
|String DP|5、10、72、97、115、139、647、1143|
|Knapsack / Subset DP|416、494、518|
|State / Interval DP|152、309、312|
|Grid / Sequence DP|62、300、329|

部分題目可以歸到不只一種 DP 類型，例如 LC 329 是 DFS + Memoization，LC 5、647 也常使用中心擴展法。

## **26. Math（數學／位元運算）— 11 題**

|題號|中文題名|English Title|
|---|---|---|
|7|整數反轉|Reverse Integer|
|43|字串相乘|Multiply Strings|
|50|x 的 n 次方|Pow(x, n)|
|136|只出現一次的數字|Single Number|
|190 ★|反轉位元|Reverse Bits|
|191 ★|位元 1 的數量|Number of 1 Bits|
|202|快樂數|Happy Number|
|268 ★|缺失的數字|Missing Number|
|338 ★|位元計數|Counting Bits|
|371 ★|兩整數之和|Sum of Two Integers|
|2013|偵測正方形|Detect Squares|

其中 LC 136、190、191、268、338、371 建議再加上 Secondary Tag：**Bit Manipulation**。LC 7、43、50、202 偏向數學運算，LC 2013 則偏向幾何與 Hash Map。

## **完整性檢查：26 個分類共 150 題**

|#|分類|題數|其中 Blind 75|
|---|---|---|---|
|1|Array|13|7|
|2|String|9|6|
|3|Prefix Sum Array|0|0|
|4|Sorting|4|3|
|5|Binary Search|7|2|
|6|Two Pointers|4|2|
|7|Linked List|11|6|
|8|Queue/deque|2|1|
|9|Monotone Queue|1|0|
|10|BFS|3|0|
|11|DFS|14|4|
|12|Stack|3|1|
|13|Monotonic Stack|3|0|
|14|Hash Map|5|5|
|15|Union Find|3|2|
|16|Iterator|0|0|
|17|Binary Tree|9|7|
|18|Binary Search Tree|3|3|
|19|Heap|9|2|
|20|Trie|3|3|
|21|Segment Tree|0|0|
|22|Balanced Binary Tree|1|0|
|23|Binary Indexed Tree|0|0|
|24|Graph|9|3|
|25|DP|23|12|
|26|Math|11|9|
||**總計**|**150**|**75**|

計數以每題唯一主要分類為準，額外推薦題不計入。

## **最後建議：不要只用單一分類管理題庫**

對你準備 Google、Meta、Apple、NVIDIA 的面試而言，實際刷題時最好是 **Primary Category + Secondary Tags**，而不是每題只能屬於一個演算法。

例如：

|題號|Primary Category|Secondary Tags|
|---|---|---|
|3|String|Sliding Window、Hash Set|
|15|Two Pointers|Sorting、Array|
|22|DFS|Backtracking、Stack|
|53|Array|Greedy、Kadane、DP|
|76|String|Sliding Window、Hash Map|
|102|Queue/deque|BFS、Binary Tree|
|146|Linked List|Hash Map、System Design|
|207|Graph|Topological Sort、BFS、DFS|
|212|Trie|DFS、Backtracking|
|238|Array|Prefix Product、Suffix Product|
|239|Monotone Queue|Sliding Window、Deque|
|253|Heap|Sorting、Intervals|
|287|Linked List|Fast/Slow Pointers、Cycle Detection|
|329|DP|DFS、Memoization、Matrix|
|862（額外）|Prefix Sum Array|Monotonic Deque|

這樣你之後想在 Cojudge 練習「所有 Prefix Sum 題目」、「所有 BFS 題目」或「所有使用 Monotonic Stack 的題目」時，才不會因為單一主分類而漏掉重要的相關題目。

尤其對三個月的面試準備，**建議保留這份完整 150 題作為 Master List，並額外建立 Algorithm Tags、難度、是否 Blind 75、Pass/Fail、重做次數、Time Complexity 和 Space Complexity 欄位**，將它當成自己的長期刷題追蹤表。