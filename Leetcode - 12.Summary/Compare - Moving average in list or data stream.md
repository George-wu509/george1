
|                                    |     |
| ---------------------------------- | --- |
| [[#### 移動窗口和在list vs data stream]] |     |
| [[#### 常見的移動窗口問題整理]]               |     |
|                                    |     |

#### 移動窗口和在list vs data stream
```
請比較這兩個lintcode的問題 A. Window Sum vs B. Moving Average from Data Stream

這兩題應該是非常類似都是移動窗口, 而且窗口長度固定所以差別只在和除以長度就是moving average. 但A是用Two pointers, B是用deque. 但我是不是可以理解成A也可以用deque也應該是更好的解法? 
```

604 A. 滑动窗口内数的和 [[Window Sum]]
642 B. 数据流滑动窗口平均值 [[Moving Average from Data Stream]] 

**A 確實可以用 `deque` 寫，但對於 A 來說，原本的索引滑動（Two Pointers 概念）才是更好的解法，改用 `deque` 反而會造成多餘的空間與時間開銷。**

這兩題雖然核心邏輯都是「固定長度的滑動視窗」，但它們面臨的**資料輸入模型（Data Model）**完全不同：一個是**靜態陣列**，一個是**資料流（Data Stream）**。

### 為什麼 B 需要 `deque`，而 A 不需要？

滑動視窗在每次右移時，都需要做兩件事：**加上新進來的右邊界值**，以及**減去被踢出去的左邊界值**。關鍵就在於「你怎麼拿到那個要被踢出去的左邊界值」：

1. **A（靜態陣列）已經擁有完整歷史資料** 在 `win_Sum` 中，整個 `nums` 陣列都已經在記憶體裡了，並且支援 O(1)的隨機存取（Random Access）。當視窗往右滑動時，你可以直接透過索引 `nums[i - 1]` 讀取要被踢掉的舊數字，**完全不需要額外開一個資料結構把看過的數字再存一遍**。如果硬在 A 裡面用 `deque`，等於是把 `nums` 裡的資料複製一份到 `deque` 中，多花了 O(k)的記憶體，且 `append` 與 `popleft` 的操作常數時間也比直接陣列取值慢。
    
2. **B（資料流）沒有完整陣列可以回頭查** 在 `MovingAverage` 中，每次呼叫 `next(val)` 只會給你當下這一個數字，系統並不會幫你保留過去傳進來的歷史紀錄。為了在視窗滿了的時候知道「最早進來的數字是多少」並將它從 `self.sum` 扣除，類別內部**必須自己維護一個容量為 `size` 的緩衝區**，這正是 `deque` 派上用場的原因。
    

### 兩題核心差異對照

|比較維度|A. Window Sum|B. Moving Average from Data Stream|
|---|---|---|
|**輸入模式**|一次性給定完整陣列（Offline / Static）|每次 `next()` 給一個元素（Online / Stream）|
|**取得左邊界方式**|直接透過索引 `nums[i - 1]` 讀取|透過 `queue.popleft()` 彈出最舊元素|
|**額外空間複雜度**|O(1)（不計入回傳的結果陣列）|O(size)（必須儲存視窗內的元素）|
|**視窗未滿時的處理**|若陣列長度小於 `k` 直接回傳 `[]`|元素數量小於 `size` 時仍要計算當下平均（除以 `len(self.queue)`）|

### 補充：什麼時候「靜態陣列」的滑動視窗也會用 `deque`？

在處理靜態陣列的滑動視窗時，如果題目要求的不是「總和」或「平均值」（這類具備可逆運算、減掉舊值就能更新的數值），而是求「視窗內的最大值 / 最小值」**（例如 LintCode 362 / LeetCode 239 Sliding Window Maximum），即使有完整陣列，我們也無法只靠減法更新狀態，這時就會使用**單調佇列（Monotonic Deque）來維護視窗內的極值候選人。




#### 常見的移動窗口問題整理
```
那也請整理leetcode/lintcode常見的移動窗口問題, 以及所推薦的解法, 以及解釋為何要用這解法. 並最後整理成table
```
在 LeetCode 與 LintCode 中，滑動視窗（Sliding Window）題目之所以會搭配不同的資料結構，核心取決於兩件事：**視窗長度是否固定**，以及**視窗滑動時「狀態（State）是否具備可逆性或單調性」**。

以下將常見的移動窗口題型分為 6 大類，說明推薦解法與背後原因，並在最後整理成總表。

### 1. 固定長度 ＋ 可逆數值 / 字元頻率統計

- **代表題目**：
    - 滑动窗口内数的和 Window Sum (LintCode 604)
    - Maximum Average Subarray I (LC 643)
    - Permutation in String (LC 567) / Find All Anagrams in a String (LC 438)
    
- **推薦解法**：**固定間距雙指標（或單一索引 `i` 與 `i - k`）＋ 變數 / 長度為 26 的陣列**
- **為何用這解法**：加法與字元計數都具備「可逆性」——右端進來一個元素就 `+1`，左端出去一個元素就 `-1`，狀態就能精準還原。因為靜態陣列支援 $O(1)$ 隨機存取，直接透過索引讀取被踢出的元素即可，不需要額外配置 `deque` 複製資料。

### 2. 固定長度 ＋ 資料流（Data Stream）

- **代表題目**：
    - Moving Average from Data Stream (LC 346 / LintCode 642)
        
- **推薦解法**：**`deque`（雙端佇列）或 固定長度環形陣列（Circular Array）**
- **為何用這解法**：在資料流情境下，系統不會幫你保存完整的歷史陣列，每次只會傳入一個新數值。為了在視窗滿載時能以 $O(1)$ 取出並扣除「最早進入的元素」，必須自己維護一個容量上限為 `size` 的 FIFO（先進先出）緩衝區。
    

### 3. 固定長度 ＋ 視窗內極值（最大值 / 最小值）

- **代表題目**：
    - Sliding Window Maximum (LC 239 / LintCode 362)
        
- **推薦解法**：**單調佇列（Monotonic Deque）**
- **為何用這解法**：`max()` 與 `min()` **不具備可逆性**——當目前的最大值滑出視窗時，無法靠減法推算出「第二大是誰」。單調佇列會在內部維持嚴格遞減（或遞增）的索引序列：新元素進來時，把前面比它小的「過期且無競爭力」的元素從尾端踢掉；隊首永遠是當前視窗的最大值，讓每次查詢與更新均攤為 $O(1)$。

### 4. 可變長度 ＋ 求「最長」合法子陣列 / 子字串

- **代表題目**：
    - Longest Substring Without Repeating Characters (LC 3 / LintCode 384)
    - Longest Repeating Character Replacement (LC 424 / LintCode 1246)
    - Max Consecutive Ones III (LC 1004)
        
- **推薦解法**：**同向雙指標（`left`, `right`）＋ Hash Map / 頻率陣列**
- **為何用這解法**：這類題目具有**單調性**——如果一個視窗已經「不合法」（例如出現重複字元），繼續往右擴張只會更不合法。因此讓 `right` 持續往右探索更長的解；一旦觸發不合法條件，就移動 `left` 縮小視窗直到重新合法，並在每次「合法狀態」時更新最大長度。

### 5. 可變長度 ＋ 求「最短」合法子陣列 / 子字串

- **代表題目**：
    - Minimum Window Substring (LC 76 / LintCode 32)
    - Minimum Size Subarray Sum (LC 209 / LintCode 406)
        
- **推薦解法**：**同向雙指標 ＋ 欠缺計數器（`missing` / `valid` counter）**
- **為何用這解法**：與求「最長」相反，這裡的單調性在於：一旦視窗「已經滿足條件」，繼續往右擴張雖然仍合法，但長度只會變差。因此 `right` 負責擴張直到條件達成，接著進入 `while (條件滿足)` 迴圈，不斷右移 `left` 榨乾多餘元素、更新最短長度，直到視窗剛好退回不合法為止。

### 6. 可變長度 ＋ 「恰好 $K$ 個（Exact K）」條件

- **代表題目**：
    - Subarrays with K Different Integers (LC 992)
    - Binary Subarrays With Sum (LC 930)
        
- **推薦解法**：**雙指標轉化公式 `atMost(K) - atMost(K - 1)`**
- **為何用這解法**：標準雙指標依賴「單調性」，但「恰好 $K$ 個不同元素」並不單調——當視窗已有 $K$ 個元素時，右移 `right` 可能遇到重複元素（依然合法），也可能遇到新元素（變成 $K+1$ 而違法），導致無法決定何時該停下。將問題拆解為「最多 $K$ 個」減去「最多 $K-1$ 個」，就能恢復單調性並直接套用第 4 類的標準模板。

### 常見移動窗口題型與解法對照表

|**題型分類**|**代表題目 (LC / LintCode)**|**推薦解法與資料結構**|**選擇該解法的核心原因**|**時間 / 空間複雜度**|
|---|---|---|---|---|
|**固定長度 ＋ 可逆統計**|Window Sum (LintCode 604)<br><br>  <br><br>Find All Anagrams (LC 438)|雙指標（索引差 $k$）<br><br>  <br><br>＋ 變數 / 長度 26 陣列|加減與頻率計數可逆，且靜態陣列支援 $O(1)$ 隨機存取舊元素，無需額外佇列|時間 $O(n)$<br><br>  <br><br>空間 $O(1)$|
|**固定長度 ＋ 資料流**|Moving Average from Data Stream (LC 346)|`deque` 或<br><br>  <br><br>環形陣列 (Circular Array)|無法回頭讀取歷史陣列，需自行維護大小為 $k$ 的緩衝區以取得最早元素|時間 $O(1)$ / 次<br><br>  <br><br>空間 $O(k)$|
|**固定長度 ＋ 視窗極值**|Sliding Window Maximum (LC 239)|單調佇列<br><br>  <br><br>(Monotonic Deque)|極值不具可逆性，需透過單調佇列淘汰無用元素，確保隊首永遠是當前極值|時間 $O(n)$<br><br>  <br><br>空間 $O(k)$|
|**可變長度 ＋ 求「最長」**|Longest Substring Without Repeating Chars (LC 3)|同向雙指標<br><br>  <br><br>＋ Hash Map / Set|視窗違法後繼續擴張必仍違法，故違法時收縮左指標、合法時更新最大長度|時間 $O(n)$<br><br>  <br><br>空間 $O(\vert{}\Sigma\vert{})$|
|**可變長度 ＋ 求「最短」**|Minimum Window Substring (LC 76)<br><br>  <br><br>Min Size Subarray Sum (LC 209)|同向雙指標<br><br>  <br><br>＋ 欠缺計數器|視窗合法後繼續擴張只會更長，故達成條件後立刻在 `while` 中收縮左指標逼出最短解|時間 $O(n)$<br><br>  <br><br>空間 $O(\vert{}\Sigma\vert{})$|
|**可變長度 ＋ 恰好 $K$ 個**|Subarrays with K Different Integers (LC 992)|雙指標轉化：<br><br>  <br><br>`atMost(K) - atMost(K-1)`|「恰好 $K$ 個」缺乏單調性，轉成「最多 $K$ 個」後即可恢復單調性並用雙指標求解|時間 $O(n)$<br><br>  <br><br>空間 $O(k)$|
