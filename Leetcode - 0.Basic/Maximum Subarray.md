Lintcode 41
给定一个整数数组，找到一个具有最大和的子数组，返回其最大和。  
每个子数组的数字在数组中的位置应该是连续的

**样例 1：**
输入：
```python
nums = [−2,2,−3,4,−1,2,1,−5,3]
```
输：
```python
6
```
解释：
符合要求的子数组为[4,−1,2,1]，其最大和为 6。

**样例 2：**
输入：
```python
nums = [1,2,3,4]
```
输出：
```python
10
```
解释：
符合要求的子数组为[1,2,3,4]，其最大和为 10。



```python
class Solution:
    """
    @param nums: A list of integers
    @return: A integer indicate the sum of max subarray
    """
    def max_sub_array(self, nums: List[int]) -> int:
        #prefix_sum记录前i个数的和，max_Sum记录全局最大值，min_Sum记录前i个数中0-k的最小值
        min_sum, max_sum = 0, -sys.maxsize
        prefix_sum = 0
        
        for num in nums:
            prefix_sum += num
            max_sum = max(max_sum, prefix_sum - min_sum)
            min_sum = min(min_sum, prefix_sum)
            
        return max_sum
```
pass
解釋:
step1: 新變數: prefix_sum(目前前綴和值), min_sum(前綴值和紀錄中的最小值), 
max_sum(前綴和值紀錄中的(前綴和值-min_sum)的最大值)
step2: for loop遍歷nums中的每個值, 並更新prefix_sum, min_sum, max_sum
step3: output max_sum

idea: nums中最大和的子数组應該就是前綴和值list中最大值-最小值

### 解析 `Maximum Subarray` 問題：

這個問題要求找出一個連續子數組（subarray），使得其總和最大，並返回該最大總和。

這是一個經典的 **最大子陣列和（Maximum Subarray Sum）** 問題，最常見的解法是 **Kadane’s Algorithm**，其核心思想是透過動態規劃（Dynamic Programming）或前綴和（Prefix Sum）來高效計算最大子數組的和。

---

## **解法解析**

此程式碼的主要邏輯是透過 **前綴和（Prefix Sum）+ 最小前綴和（Min Prefix Sum）** 來計算 **最大子數組和**，具體步驟如下：

1. **定義變數**
    
    - `prefix_sum`：用於累積當前數組的總和。
    - `min_sum`：用於記錄 **前i個數** 中的最小前綴和，初始值為 `0`。
    - `max_sum`：記錄全局的最大子數組和，初始值為負無窮大（`-sys.maxsize`）。
2. **遍歷 `nums`，累積前綴和**
    
    - `prefix_sum += num`：計算當前總和。
    - `max_sum = max(max_sum, prefix_sum - min_sum)`：
        - `prefix_sum - min_sum`：代表「當前前綴和 `prefix_sum` 減去最小前綴和 `min_sum`」，即得到某一個區間 `[j, i]` 的最大子數組和。
        - `max()` 確保 `max_sum` 永遠儲存全局最大子數組和。
    - `min_sum = min(min_sum, prefix_sum)`：
        - 不斷更新最小的 `prefix_sum`，確保可以找到一個最大的 `prefix_sum - min_sum`。

---

## **具體舉例**

假設 `nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]`，我們的變數變化如下：

|`num`|`prefix_sum`|`min_sum`|`max_sum`|`prefix_sum - min_sum`|
|---|---|---|---|---|
|-2|-2|-2|-2|-2|
|1|-1|-2|1|(-1 - -2) = 1|
|-3|-4|-4|1|(-4 - -4) = 0|
|4|0|-4|4|(0 - -4) = 4|
|-1|-1|-4|4|(-1 - -4) = 3|
|2|1|-4|5|(1 - -4) = 5|
|1|2|-4|6|(2 - -4) = 6|
|-5|-3|-4|6|(-3 - -4) = 1|
|4|1|-4|6|(1 - -4) = 5|

最終 `max_sum = 6`，對應的最大子數組為 `[4, -1, 2, 1]`。

---

## **時間與空間複雜度分析**

- **時間複雜度：O(n)**  
    這個方法僅遍歷數組一次，因此時間複雜度為 **O(n)**，對於大數據量非常高效。
    
- **空間複雜度：O(1)**  
    這個方法只使用了固定數量的變數 (`prefix_sum`, `min_sum`, `max_sum`)，因此空間複雜度為 **O(1)**。
    

---

## **其他解法**

### **1. Kadane’s Algorithm（貪心法）**

- 核心思路：維護當前最大子數組和 `current_sum`，如果 `current_sum` 變成負數就重置為 0。
- 時間複雜度：O(n)
- 空間複雜度：O(1)

### **2. 分治法（Divide and Conquer）**

- 使用遞迴將數組拆分成左右兩半，分別計算最大子數組和，再考慮跨越中間的情況。
- 時間複雜度：O(n log n)
- 空間複雜度：O(log n)（遞歸棧）

### **3. 動態規劃（DP）**

- `dp[i] = max(dp[i-1] + nums[i], nums[i])`，即當前元素 `nums[i]` 要么加入前面的子數組，要么自己成為新子數組。
- 時間複雜度：O(n)
- 空間複雜度：O(1)

---

## **總結**

這題的最佳解法為 **前綴和 + 最小前綴和**，與 **Kadane’s Algorithm**，皆能在 **O(n) 時間與 O(1) 空間**內完成計算，非常適合大數據處理。