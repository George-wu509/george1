Lintcode 50 = Lintcode 1310
给定一个整数数组`A`。  
定义$∗A[i+1]∗...∗A[n−1]$， 计算`B`的时候请不要使用除法。请输出`B`。

**样例 1：**
输入：
```python
#A = [1,2,3]
```
输出：
```python
#[6,3,2]
```
解释：
B[0] = A[1] * A[2] = 6; B[1] = A[0] * A[2] = 3; B[2] = A[0] * A[1] = 2

**样例 2：**
输入：
```python
#A = [2,4,6]
```
输出：
```python
#[24,12,8]
```
解释：
B[0] = A[1] * A[2] = 24; B[1] = A[0] * A[2] = 12; B[2] = A[0] * A[1] = 8



```python
class Solution:
    def productExcludeItself(self, nums):
        length ,B  = len(nums) ,[]
        f = [ 0 for i in range(length + 1)]
        f[ length ] = 1
        for i in range(length - 1 , 0 , -1):
            f[ i ] = f[ i + 1 ] * nums[ i ]
        tmp = 1
        for i in range(length):
            B.append(tmp * f[ i + 1 ])
            tmp *= nums[ i ]
        return B
```
pass

[1, 2, 3, 4]
解釋:
step1 計算後綴積 suffix = [0,24,12,4,1]
step2 計算前綴積prefix = [1,1,2,6,24]
result= [24,12,8,6]

### **LintCode 50 - Product of Array Exclude Itself**

這道題目要求 **對於數組 `nums` 中的每個元素 `nums[i]`，計算不包含 `nums[i]` 的所有元素的乘積**，並返回一個新的數組。

這是一道典型的 **前綴積（Prefix Product）+ 後綴積（Suffix Product）** 問題，關鍵在於 **如何不使用除法** 來高效計算結果。

---

## **解法解析**

這裡使用 **前綴乘積（Prefix Product）+ 後綴乘積（Suffix Product）** 來解決問題。

### **思路**

1. **定義 `B[i]` 的計算方式**
    
    - `B[i]` 需要計算 **不包含 `nums[i]` 的所有元素乘積**。
    - 若我們直接遍歷數組，對每個 `i` 計算所有不包含 `nums[i]` 的乘積，則時間複雜度為 **O(n²)**，這是不可接受的。
2. **使用 `前綴積`（Prefix Product）**
    
    - `prefix[i]`：表示 **從索引 `0` 到 `i-1` 的乘積**（不包含 `nums[i]`）。
    - `prefix[0] = 1`（因為 `nums[0]` 左邊沒有數字）。
    - 透過 `prefix[i] = prefix[i-1] * nums[i-1]` 計算前綴積。
3. **使用 `後綴積`（Suffix Product）**
    
    - `suffix[i]`：表示 **從索引 `i+1` 到 `n-1` 的乘積**（不包含 `nums[i]`）。
    - `suffix[n-1] = 1`（因為 `nums[n-1]` 右邊沒有數字）。
    - 透過 `suffix[i] = suffix[i+1] * nums[i+1]` 計算後綴積。
4. **計算 `B[i]`**
    
    - `B[i] = prefix[i] * suffix[i]`。

---

## **具體舉例**

假設 `nums = [1, 2, 3, 4]`，我們的變數變化如下：

### **Step 1: 計算 `suffix`（從右往左計算後綴積）**

|`i`|`nums[i]`|`suffix[i]`|
|---|---|---|
|3|4|1|
|2|3|`4 * 1 = 4`|
|1|2|`3 * 4 = 12`|
|0|1|`2 * 12 = 24`|

`suffix = [24, 12, 4, 1]`

---

### **Step 2: 計算 `B[i]`**

|`i`|`nums[i]`|`tmp`（prefix）|`B[i] = prefix[i] * suffix[i]`|
|---|---|---|---|
|0|1|1|`1 * 12 = 12`|
|1|2|`1 * 1 = 1`|`1 * 4 = 8`|
|2|3|`1 * 2 = 2`|`2 * 1 = 6`|
|3|4|`2 * 3 = 6`|`6 * 1 = 6`|

最終結果 `B = [12, 8, 6, 6]`。

---

## **時間與空間複雜度分析**

- **時間複雜度：O(n)**
    
    - **後綴積計算**：O(n)
    - **前綴積計算**：O(n)
    - **合併計算 `B[i]`**：O(n)
    - **總計 O(n) + O(n) + O(n) = O(n)**。
- **空間複雜度：O(n)**
    
    - 需要額外的 `suffix` 陣列儲存後綴乘積，空間為 **O(n)**。

---

## **其他解法**

### **1. 雙陣列方式（Prefix Product + Suffix Product）**

- **思路**：
    - 先計算 `prefix` 陣列（左側累積乘積）。
    - 再計算 `suffix` 陣列（右側累積乘積）。
    - 最後 `B[i] = prefix[i] * suffix[i]`。
- **時間複雜度**：O(n)
- **空間複雜度**：O(n)

### **2. 優化空間至 O(1)**

- **思路**：
    - 用一個變數 `suffix_product` 來動態計算後綴乘積，避免使用額外陣列。
- **時間複雜度**：O(n)
- **空間複雜度**：O(1)

### **3. 直接暴力法**

- **思路**：
    - 對每個 `i`，計算 `nums[0] * nums[1] * ... * nums[i-1] * nums[i+1] * ... * nums[n-1]`。
- **時間複雜度**：O(n²)
- **空間複雜度**：O(1)（但計算速度太慢，不適合大數據）。

---

## **總結**

- **最佳解法**：使用 **前綴積 + 後綴積**，時間 O(n)，空間 O(n)（或 O(1) 優化）。
- **若需要最優空間**：使用變數 `suffix_product` 動態計算後綴乘積，空間 O(1)。
- **暴力解法不可取**，時間複雜度 O(n²) 太慢。

## **總結**

|題號|題目|难度|主要技術|
|---|---|---|---|
|**LintCode 50**|Product of Array Exclude Itself|⭐⭐⭐|**前綴積 + 後綴積**|
|**LintCode 515**|Paint House|⭐⭐⭐|**變形前綴積**|
|**LintCode 883**|Three Sum Closest|⭐⭐⭐|**前綴積加速雙指針**|
|**LintCode 189**|First Missing Positive|⭐⭐⭐|**標記缺失數字**|
|**LintCode 207**|Interval Sum|⭐⭐⭐|**前綴積變形**|
|**LintCode 665**|Range Sum Query - Mutable|⭐⭐⭐|**動態前綴積**|
|**LintCode 534**|House Robber II|⭐⭐⭐|**變形前綴積**|
|**LintCode 838**|Subarray Sum Equals K|⭐⭐⭐|**前綴積變形**|
|**LintCode 1854**|Product of Array Except Self|⭐⭐⭐|**標準前綴積 + 後綴積**|
|**LintCode 192**|Wildcard Matching|⭐⭐⭐|**前綴匹配 + 後綴匹配**|

這些題目大多數都涉及 **陣列前綴積 / 後綴積運算、區間查詢、動態規劃（DP）**，希望這個整理能幫助你更有效率地應對相關題目！

### **使用「前綴積（Prefix Product）+ 後綴積（Suffix Product）」解法的 LintCode Easy & Medium 題目整理**

這類題目通常涉及 **陣列運算**，需要快速計算「不包含某個元素的乘積」、「區間乘積」或「某個範圍內的乘積變化」。以下是一些適合此解法的 **LintCode Easy & Medium** 題目：

---

## **1. LintCode 50 - Product of Array Exclude Itself（中等）**

**👉 題目描述：**

- 給定一個整數數組 `nums`，返回一個新數組 `B`，其中 `B[i]` 為 **不包含 `nums[i]` 的所有元素的乘積**。
- **不允許使用除法**。

**🔹 解法：**

- 使用 **前綴積（Prefix Product）+ 後綴積（Suffix Product）** 來計算 `B[i] = prefix[i] * suffix[i]`。

---

## **2. LintCode 515 - Paint House（中等）**

**👉 題目描述：**

- 給定 `n` 棟房子，每棟房子可以塗 `k` 種顏色，每種顏色都有成本 `costs[i][j]`，求最小的總成本，使得相鄰的房子顏色不同。

**🔹 解法：**

- **前綴積 + 後綴積變形**：對於每個房子，計算它能選擇的最小成本，但不能選擇相鄰的房子相同的顏色。
- **狀態轉移**：計算 `prefix_min[i]` 和 `suffix_min[i]` 來找出當前房子的最低塗色成本。

---

## **3. LintCode 883 - Three Sum Closest（中等）**

**👉 題目描述：**

- 給定一個數組 `nums` 和一個目標數 `target`，找到三個數的和，使其最接近 `target`，並返回該和。

**🔹 解法：**

- 在進行 **三數和（Three Sum）** 過程中，使用 **前綴積** 來優化 **雙指針遍歷的查找過程**，加速查找接近 `target` 的三元組。

---

## **4. LintCode 189 - First Missing Positive（中等）**

**👉 題目描述：**

- 找到未出現在陣列 `nums` 中的最小正整數（`> 0`）。

**🔹 解法：**

- 可以使用 **前綴積** 來加速標記出現在陣列中的數字，確保在 `O(n)` 時間內找到最小的缺失正整數。

---

## **5. LintCode 207 - Interval Sum（區間和查詢，中等）**

**👉 題目描述：**

- 給定一個數組 `nums` 和多個查詢 `(start, end)`，返回每個區間的總和。

**🔹 解法：**

- **前綴積變形（Prefix Sum Product）**：使用 `prefix[i]` 來存儲從 `0` 到 `i` 的和，然後用 `prefix[end] - prefix[start - 1]` 來快速查詢。

---

## **6. LintCode 665 - Range Sum Query - Mutable（區間和可變，中等）**

**👉 題目描述：**

- `nums` 是一個可變數組，支援兩種操作：
    1. `update(index, value)`：更新 `nums[index]` 為 `value`。
    2. `sumRange(start, end)`：返回 `nums[start]` 到 `nums[end]` 的總和。

**🔹 解法：**

- 使用 **前綴積變形（Prefix Sum Product）** 來計算區間和，類似於「區間和查詢」問題。

---

## **7. LintCode 534 - House Robber II（中等）**

**👉 題目描述：**

- 與 **House Robber** 類似，但房子是環形排列的，不能搶劫相鄰房子。

**🔹 解法：**

- 使用 **前綴積與後綴積變形**，將環形陣列拆成兩個線性陣列，計算不包含第一個元素與不包含最後一個元素的最大收益。

---

## **8. LintCode 838 - Subarray Sum Equals K（中等）**

**👉 題目描述：**

- 找出數組 `nums` 中所有子數組，使其和等於 `k`。

**🔹 解法：**

- 使用 **前綴積變形** 來優化 `O(n²)` 暴力解法，使其達到 `O(n)`。

---

## **9. LintCode 1854 - Product of Array Except Self（中等）**

**👉 題目描述：**

- 與 LintCode 50 類似，計算陣列 `nums` 除了 `nums[i]` 以外所有數的乘積，**不允許使用除法**。

**🔹 解法：**

- **標準的前綴積 + 後綴積**，時間 `O(n)`，空間 `O(1)`（如果使用變數來存後綴積）。

---

## **10. LintCode 192 - Wildcard Matching（中等）**

**👉 題目描述：**

- 給定字串 `s` 和匹配模式 `p`，其中 `?` 可匹配任何單個字母，`*` 可匹配零個或多個字母，判斷 `s` 是否匹配 `p`。

**🔹 解法：**

- 可以使用 **前綴匹配（Prefix Matching）+ 後綴匹配（Suffix Matching）** 來加速 `O(n²)` DP 解法。