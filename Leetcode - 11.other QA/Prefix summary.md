
在數組處理中，「**前綴和 (Prefix Sum)**」、「**後綴和 (Suffix Sum)**」、「**前綴積 (Prefix Product)**」和「**後綴積 (Suffix Product)**」是常見的計算方式，它們的計算方式如下：

---

### 1. **前綴和 (Prefix Sum)**

**定義**：前綴和是指從數組的第一個元素開始，累積到當前元素的總和。

**計算方式**：

- `prefix_sum[i] = nums[0] + nums[1] + ... + nums[i]`
- 以 `nums = [1, 2, 3, 4]` 為例：


`prefix_sum[0] = 1`
`prefix_sum[1] = 1 + 2 = 3`
`prefix_sum[2] = 1 + 2 + 3 = 6`
`prefix_sum[3] = 1 + 2 + 3 + 4 = 10`

**結果**：

`prefix_sum = [1, 3, 6, 10]`

**應用**：
	`Sum[1,2,3] = nums[1]+ nums[2]+ nums[3]`
	`= prefix-sum[3] - prefix-sum[0] =10-1 = 9` 

- **區間和查詢 (Range Sum Query, RSQ)**，用來快速計算某段範圍 `[l, r]` 的總和 ，時間複雜度從 `O(n)` 降至 `O(1)`。總和=prefix[r] - prefix[l-1]
- **動態規劃 (Dynamic Programming)** 的狀態轉移時，常用來快速查詢累計和。

---

### 2. **後綴和 (Suffix Sum)**

**定義**：後綴和是指從數組的最後一個元素開始，累積到當前元素的總和。

**計算方式**：

- `suffix_sum[i] = nums[i] + nums[i+1] + ... + nums[n-1]`
- 以 `nums = [1, 2, 3, 4]` 為例：

`suffix_sum[3] = 4`
`suffix_sum[2] = 3 + 4 = 7`
`suffix_sum[1] = 2 + 3 + 4 = 9`
`suffix_sum[0] = 1 + 2 + 3 + 4 = 10`

**結果**：

`suffix_sum = [10, 9, 7, 4]`

**應用**：

- 在某些需要從後往前處理數組的問題，例如 **右側區間和查詢** 或 **滑動窗口問題**。

---

### 3. **前綴積 (Prefix Product)**

**定義**：前綴積是指從數組的第一個元素開始，累積到當前元素的乘積。

**計算方式**：

- `prefix_product[i] = nums[0] * nums[1] * ... * nums[i]`
- 以 `nums = [1, 2, 3, 4]` 為例：

`prefix_product[0] = 1`
`prefix_product[1] = 1 * 2 = 2`
`prefix_product[2] = 1 * 2 * 3 = 6`
`prefix_product[3] = 1 * 2 * 3 * 4 = 24`


**結果**：

`prefix_product = [1, 2, 6, 24]`

**應用**：
    Product[1,2,3] = nums[1]  * nums[2]  * nums[3] 

- 用於 **乘積查詢問題** 或 **求某元素除自身外的乘積 (Product of Array Except Self)**，這在 Leetcode/Lintcode 上是一道經典題。
- **解法（使用前綴積與後綴積）**：

1. 計算 `prefix_product`，存儲從 `0` 到 `i-1` 的乘積。prefix_product[i-1] 
2. 計算 `suffix_product`，存儲從 `i+1` 到 `n-1` 的乘積。prefix_product[i+1] 
3. `output[i] = prefix_product[i-1] × suffix_product[i+1]`

---

### 4. **後綴積 (Suffix Product)**

**定義**：後綴積是指從數組的最後一個元素開始，累積到當前元素的乘積。

**計算方式**：

- `suffix_product[i] = nums[i] * nums[i+1] * ... * nums[n-1]`
- 以 `nums = [1, 2, 3, 4]` 為例：

`suffix_product[3] = 4`
`suffix_product[2] = 3 * 4 = 12`
`suffix_product[1] = 2 * 3 * 4 = 24`
`suffix_product[0] = 1 * 2 * 3 * 4 = 24`


**結果**：

`suffix_product = [24, 24, 12, 4]`

**應用**：

- 用於 **求某元素除自身外的乘積 (Product of Array Except Self)** 問題，避免直接計算所有元素相乘再除以當前元素，從而提高效率。

---

### 📌 **應用於 LintCode / LeetCode**

這些前綴和、後綴和、前綴積、後綴積的概念在競程 (competitive programming) 和演算法問題中有很多應用，例如：

1. **Leetcode 303. Range Sum Query - Immutable**
    
    - 使用前綴和來高效計算區間和。
2. **Leetcode 238. Product of Array Except Self**
    
    - 透過前綴積和後綴積來計算 **不使用除法** 的解法。
3. **LintCode 206. Interval Sum**
    
    - 使用前綴和來優化區間查詢。
4. **LintCode 665. Range Sum Query II**
    
    - 動態更新數組時的區間和查詢，結合前綴和與樹狀數組 (Fenwick Tree)。
5. **Leetcode 560. Subarray Sum Equals K**
    
    - 利用前綴和搭配哈希表來高效計算連續子數組總和為 `k` 的次數。

這些技巧能大幅提升演算法的效率，避免暴力 `O(n^2)` 的計算方式，讓某些問題可以在 `O(n)` 或 `O(log n)` 時間內解決。

[^1]: 

[^2]: prefix_sum[0] = 1
	prefix_sum[1] = 1 + 2 = 3
	prefix_sum[2] = 1 + 2 + 3 = 6
	prefix_sum[3] = 1 + 2 + 3 + 4 = 10
