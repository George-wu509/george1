Lintcode 346
给你一个数组，请计算出所有子区间的最大值异或最小值的异或

**样例 1:**
```python
"""
输入:[1, 2, 3]
输出:0
说明: 
这个数组有6个子区间: [1], [2], [3], [1, 2], [2, 3], [1, 2, 3]
分别对应的异或和是: 0, 0, 0, 3, 1, 2
最后的答案是: 0
```
**样例 2:**
```python
"""
输入:[1, 3, 2]
输出:1
说明: 
这个数组有6个子区间: [1], [3], [2], [1, 3], [3, 2], [1, 3, 2]
分别对应的异或和是: 0, 0, 0, 2, 1, 2
最后的答案是: 1
```
解釋:
XOR 總和是指 對一組數據進行「位元異或運算（XOR）
Example1(解釋XOR sum):  nums=[3,1,2]  --> XOR sum = 3 ^ 1 ^ 2 = 3 ⊕ 1 ⊕ 2
step1: 3=011, 1=001, 2=010
step2: 3^1 = 011 ⊕ 001 = 010   (所以=2)
step3: 2^2 = 010 ⊕ 010 = 000   (所以=0)
step4: 所以XOR sum = 3 ^ 1 ^ 2 = 0

Example2: [1, 2, 3] 最大值和最小值的 XOR 總和
step1: 最大值=3, 最小值=1
step2: 3^1 = 011 ⊕ 001 = 010   (所以=2)


```python
def xor_sum(self, nums):
	n = len(nums)
	left_min, left_max = [0] * n, [0] * n
	right_min, right_max = [0] * n, [0] * n
	
	stack = []
	for i in range(n):
		while stack and nums[i] > nums[stack[-1]]:
			stack.pop(-1)
		left_min[i] = i - stack[-1] - 1 if stack else i
		stack.append(i)
	
	stack = []
	for i in range(n):
		while stack and nums[i] < nums[stack[-1]]:
			stack.pop(-1)
		left_max[i] = i - stack[-1] - 1 if stack else i
		stack.append(i)
	
	stack = []
	for i in range(n - 1, -1, -1):
		while stack and nums[i] >= nums[stack[-1]]:
			stack.pop(-1)
		right_min[i] = stack[-1] - i - 1 if stack else n - 1 - i
		stack.append(i)
	
	stack = []
	for i in range(n - 1, -1, -1):
		while stack and nums[i] <= nums[stack[-1]]:
			stack.pop(-1)
		right_max[i] = stack[-1] - i - 1 if stack else n - 1 - i
		stack.append(i)
	
	result = 0
	for i in range(n):
		times = left_min[i] + right_min[i] + 1
		times += left_min[i] * right_min[i]
		times += left_max[i] + right_max[i] + 1
		times += left_max[i] * right_max[i]
		result ^= (times % 2) * nums[i]
	
	return result
```
pass


### **LintCode 346 - XOR Sum of Interval Extremum**

#### **解法分析**

本題的目標是計算數組 `nums` 中所有區間內的最大值和最小值的 XOR 總和。  
具體來說：

1. 對於 **每個元素 `nums[i]`**，找出它作為 **區間最大值或最小值** 時的貢獻次數。
2. 如果 `nums[i]` 在奇數次數的區間內是最大或最小值，則它會影響最終的 XOR 值。
3. 使用 **單調棧 (Monotonic Stack)** 來有效計算 `nums[i]` 作為最大值和最小值的影響範圍。

---

### **解法思路**

#### **1. 計算 `nums[i]` 作為最小值的影響區間**

- `left_min[i]`：`nums[i]` 能夠影響多少個區間 **向左延伸** 作為最小值
- `right_min[i]`：`nums[i]` 能夠影響多少個區間 **向右延伸** 作為最小值

✅ **單調遞增棧 (Monotonic Increasing Stack)** 幫助我們找到 `nums[i]` 在最小值時的影響範圍。

#### **2. 計算 `nums[i]` 作為最大值的影響區間**

- `left_max[i]`：`nums[i]` 能夠影響多少個區間 **向左延伸** 作為最大值
- `right_max[i]`：`nums[i]` 能夠影響多少個區間 **向右延伸** 作為最大值

✅ **單調遞減棧 (Monotonic Decreasing Stack)** 幫助我們找到 `nums[i]` 在最大值時的影響範圍。

#### **3. 計算 XOR**

- `times`：`nums[i]` 作為最大值或最小值的 **影響區間數量**。
- `times % 2`：
    - 若 `times` 為 **奇數**，則 `nums[i]` 會影響最終的 XOR 值。
    - 若 `times` 為 **偶數**，則 `nums[i]` 貢獻會被抵消，不影響 XOR。

✅ **XOR 計算**：

`result ^= (times % 2) * nums[i]`

這樣就能高效計算 XOR。

---

### **變數說明**

|變數名稱|說明|
|---|---|
|`nums`|輸入數組|
|`n`|數組長度|
|`left_min[i]`|`nums[i]` 作為最小值時，向左能影響的區間長度|
|`right_min[i]`|`nums[i]` 作為最小值時，向右能影響的區間長度|
|`left_max[i]`|`nums[i]` 作為最大值時，向左能影響的區間長度|
|`right_max[i]`|`nums[i]` 作為最大值時，向右能影響的區間長度|
|`stack`|單調棧，用來快速確定 `nums[i]` 的影響區間|
|`times`|`nums[i]` 影響 XOR 的總區間數|
|`result`|最終的 XOR 值|

---

### **範例**

#### **輸入**

python

複製編輯

`nums = [3, 1, 2]`

#### **處理流程**

4. **計算 `left_min[i]`**
    
    ini
    
    複製編輯
    
    `left_min = [0, 1, 1]  # 代表 nums[i] 作為最小值的左影響範圍`
    
5. **計算 `right_min[i]`**
    
    ini
    
    複製編輯
    
    `right_min = [0, 1, 0]  # 代表 nums[i] 作為最小值的右影響範圍`
    
6. **計算 `left_max[i]`**
    
    ini
    
    複製編輯
    
    `left_max = [0, 0, 1]  # 代表 nums[i] 作為最大值的左影響範圍`
    
7. **計算 `right_max[i]`**
    
    ini
    
    複製編輯
    
    `right_max = [2, 0, 0]  # 代表 nums[i] 作為最大值的右影響範圍`
    
8. **計算 XOR**
    - `nums[0] = 3, times = 3 (奇數) → XOR 影響`
    - `nums[1] = 1, times = 1 (奇數) → XOR 影響`
    - `nums[2] = 2, times = 3 (奇數) → XOR 影響`
    - **XOR 結果：`3 ^ 1 ^ 2 = 0`**

#### **輸出**

python

複製編輯

`0`

---

### **時間與空間複雜度分析**

#### **時間複雜度**

- **單調棧計算 `left_min`、`right_min`、`left_max`、`right_max`**：
    - 每個元素最多進棧 **一次**，出棧 **一次**，總共 **O(N)**
- **計算 XOR**：
    - 遍歷 `nums` **O(N)**。
- **總時間複雜度：O(N)**。

#### **空間複雜度**

- `left_min`、`right_min`、`left_max`、`right_max` 皆為 **O(N)**
- `stack` 最壞情況下存 **O(N)**
- **總空間複雜度：O(N)**。

---

### **其他解法想法**

9. **暴力解法 (O(N³))**
    
    - 枚舉所有子數組，計算最大值與最小值並 XOR。
    - **時間複雜度 O(N³)**，適用於 `N` 很小的情況。
10. **線段樹 (Segment Tree) (O(N log N))**
    
    - 使用 **線段樹 (Segment Tree)** 來計算區間最大值與最小值，並統計 XOR 貢獻。
    - **時間複雜度 O(N log N)**，適用於大量查詢的情境。
11. **稀疏表 (Sparse Table) (O(N log N))**
    
    - 適用於 **靜態數組**，使用 **預處理最小值與最大值** 來加速查詢。
    - **時間複雜度 O(N log N)**，但需要 **O(N log N) 空間**。

---

### **總結**

- **最優解法：單調棧 + 貢獻計算 (O(N) 時間, O(N) 空間)**
- **若 `N` 很小，可使用暴力解法 (O(N³))**
- **若需要頻繁查詢，可以使用線段樹 (O(N log N))**



### **XOR 總和（XOR Sum）**

XOR 總和是指 **對一組數據進行「位元異或運算（XOR）」的結果**，也就是將數據逐一應用 `^` 運算符的結果。

---

### **XOR 運算的特性**

1. **自反性 (Identity Property)**
    
    - `a ^ 0 = a`（任何數與 `0` XOR，結果仍是它本身）
2. **交換律 (Commutative Property)**
    
    - `a ^ b = b ^ a`（順序不影響結果）
3. **結合律 (Associative Property)**
    
    - `(a ^ b) ^ c = a ^ (b ^ c)`（可依序分組運算）
4. **自消性 (Self-Canceling Property)**
    
    - `a ^ a = 0`（同一個數 XOR 兩次會變成 `0`）

---

### **XOR 總和的計算方式**

#### **範例 1：簡單 XOR 總和**

假設有數組：

python

複製編輯

`nums = [3, 1, 2]`

計算 XOR 總和：

複製編輯

`3 ^ 1 ^ 2 = 3 ⊕ 1 ⊕ 2`

我們逐步計算：

1. `3 = 011`
2. `1 = 001`
3. `2 = 010`

異或計算：

markdown

複製編輯

`011 ⊕ 001 ------   010  (結果 2)  010 ⊕ 010 ------   000  (結果 0)`

所以 `3 ^ 1 ^ 2 = 0`。

---

### **如何在問題中應用 XOR 總和**

在 **LintCode 346 - XOR Sum of Interval Extremum** 問題中，我們透過計算每個元素 `nums[i]` 在所有區間內作為「最大值」或「最小值」的次數來影響 XOR 結果：

- 若 `nums[i]` 在 **奇數個區間內** 作為最大或最小值，則它會影響 XOR 值。
- 若 `nums[i]` 在 **偶數個區間內** 作為最大或最小值，則它的影響會被抵消。

### **XOR 總和的應用場景**

4. **計算某範圍內的異或總和**（例如前綴 XOR）
5. **尋找唯一數字（Single Number 問題）**
    - 例如，在一個所有數字都出現兩次，只有一個數字出現一次的數組中，我們可以透過 XOR 來找出該數：
    - `nums = [2, 3, 2, 4, 4]`
    - `2 ^ 3 ^ 2 ^ 4 ^ 4 = 3`（相同數字 XOR 會變 `0`）

---

### **總結**

- **XOR 總和** 是一種 **透過 XOR 計算一組數據的總和** 的方式，常用於數組處理和某些特定問題（例如找出唯一元素）。
- **在 LintCode 346 問題中，我們計算每個元素在所有子數組內出現為最大值或最小值的次數，若該次數為奇數，則該元素會影響 XOR 總和。**
- **XOR 具有自消性 (`a ^ a = 0`)，因此當某個數在偶數個區間內出現時，其影響會被抵消，不會影響最終的 XOR 總和。**