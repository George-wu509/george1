1850
Alice 和 Bob 在一个漂亮的果园里面工作，果园里面有N棵苹果树排成了一排，这些苹果树被标记成1 - N号。  
Alice 计划收集连续的K棵苹果树上面的所有苹果，Bob 计划收集连续的L棵苹果树上面的所有苹果。  
他们希望选择不相交的部分（一个由 Alice 的K树组成，另一个由鲍勃 Bob 的L树组成），以免相互干扰。你应该返回他们可以收集的最大数量的苹果。
**示例 1:**
```python
输入:
A = [6, 1, 4, 6, 3, 2, 7, 4]
K = 3
L = 2
输出: 
24
解释：
因为Alice 可以选择3-5颗树然后摘到4 + 6 + 3 = 13 个苹果， Bob可以选择7-8棵树然后摘到7 + 4 = 11个苹果，因此他们可以收集到13 + 11 = 24。
```
**示例 2:**
```python
输入:
A = [10, 19, 15]
K = 2
L = 2
输出: 
-1
解释：
因为对于 Alice 和 Bob 不能选择两个互不重合的区间。
```


```python
class Solution:
    def pick_apples(self, a: List[int], k: int, l: int) -> int:
        n = len(a)
        if n < k + l:
            return -1
        
        prefix_sum = a[:]
        # 计算前缀和
        for i in range(1, n):
            prefix_sum[i] += prefix_sum[i - 1]
            
        # prefixK 代表前 i 个数中，长度为 K 的子区间和的最大值
        prefix_k = [0] * n
        prefix_l = [0] * n
        
        # postfixK 代表后面 [i, n - 1] 中，长度为 k 的子区间和的最大值
        postfix_k = [0] * n
        postfix_l = [0] * n
        
        # 计算前缀值
        for i in range(n):
            if i == k - 1:
                prefix_k[i] = self.range_sum(prefix_sum, i - k + 1, i)
            elif i > k - 1:
                prefix_k[i] = max(self.range_sum(prefix_sum, i - k + 1, i), prefix_k[i - 1])
            if i == l - 1:
                prefix_l[i] = self.range_sum(prefix_sum, i - l + 1, i)
            elif i > l - 1:
                prefix_l[i] = max(self.range_sum(prefix_sum, i - l + 1, i), prefix_l[i - 1])
        
        # 计算后缀值
        for i in range(n - 1, -1, -1):
            if i + k - 1 == n - 1:
                postfix_k[i] = self.range_sum(prefix_sum, i, i + k - 1)
            elif i + k - 1 < n - 1:
                postfix_k[i] = max(self.range_sum(prefix_sum, i, i + k - 1), postfix_k[i + 1])
            if i + l - 1 == n - 1:
                postfix_l[i] = self.range_sum(prefix_sum, i, i + l - 1)
            elif i + l - 1 < n - 1:
                postfix_l[i] = max(self.range_sum(prefix_sum, i, i + l - 1), postfix_l[i + 1])
        
        result = 0
        # 枚举分界点，计算答案
        for i in range(i, n - 1):
            result = max(result, prefix_k[i] + postfix_l[i + 1])
            result = max(result, prefix_l[i] + postfix_k[i + 1])
        
        return result
        
        
    def range_sum(self, prefix_sum, l, r):
        if l == 0:
            return prefix_sum[r]
        return prefix_sum[r] - prefix_sum[l - 1]
```
pass


## **解法思路**

### **核心想法**

1. **定義區間問題**：
    
    - 我們需要選擇兩個不重疊的區間，分別長度為 `k` 和 `l`，使得兩個區間的和最大。
    - 第一個區間可以在左，第二個區間可以在右；或者第一個區間在右，第二個區間在左。
2. **前綴與後綴最大值**：
    
    - 利用 **前綴最大值（prefix）** 和 **後綴最大值（postfix）**：
        - **`prefix_k[i]`**：從索引 `0` 到 `i` 的長度為 `k` 的子區間最大值。
        - **`postfix_l[i]`**：從索引 `i` 到結尾的長度為 `l` 的子區間最大值。
    - 這些數組記錄了當前區間中能取到的最大值，便於快速計算跨區間的總和。
3. **分界點枚舉**：
    
    - 對於每個分界點 `i`，將數組分為左區間 `[0, i]` 和右區間 `[i+1, n-1]`。
    - 將左區間的 `prefix_k` 與右區間的 `postfix_l` 相加，或將左區間的 `prefix_l` 與右區間的 `postfix_k` 相加。
4. **最終結果**：
    
    - 遍歷所有可能的分界點，取最大總和。

---

### **公式**

- 計算公式： result=max⁡(prefix_k[i]+postfix_l[i+1],prefix_l[i]+postfix_k[i+1])\text{result} = \max(\text{prefix\_k}[i] + \text{postfix\_l}[i+1], \text{prefix\_l}[i] + \text{postfix\_k}[i+1])result=max(prefix_k[i]+postfix_l[i+1],prefix_l[i]+postfix_k[i+1])
    - 第一項：左區間長度為 `k`，右區間長度為 `l`。
    - 第二項：左區間長度為 `l`，右區間長度為 `k`。

---

## **逐步計算**

### **輸入**

- `a = [6, 1, 4, 6, 3, 2, 7, 4]`
- `k = 3`
- `l = 2`

---

### **步驟 1：計算前綴和**

計算前綴和：

prefix_sum=[6,7,11,17,20,22,29,33]\text{prefix\_sum} = [6, 7, 11, 17, 20, 22, 29, 33]prefix_sum=[6,7,11,17,20,22,29,33]

---

### **步驟 2：計算前綴最大值（`prefix_k` 和 `prefix_l`）**

#### **計算 `prefix_k`**

1. 初始化 `prefix_k = [0, 0, 0, 0, 0, 0, 0, 0]`
2. 遍歷每個 `i`，計算長度為 `k` 的區間最大值：
    - **`i = 2`**：區間 `[0, 2]`，和為 `11`
    - **`i = 3`**：區間 `[1, 3]`，和為 `11`
    - **`i = 4`**：區間 `[2, 4]`，和為 `13`
    - **`i = 5`**：區間 `[3, 5]`，和為 `11`
    - **`i = 6`**：區間 `[4, 6]`，和為 `12`
    - **`i = 7`**：區間 `[5, 7]`，和為 `13`
    - 更新最大值： prefix_k=[0,0,11,11,13,13,13,13]\text{prefix\_k} = [0, 0, 11, 11, 13, 13, 13, 13]prefix_k=[0,0,11,11,13,13,13,13]

#### **計算 `prefix_l`**

1. 初始化 `prefix_l = [0, 0, 0, 0, 0, 0, 0, 0]`
2. 遍歷每個 `i`，計算長度為 `l` 的區間最大值：
    - **`i = 1`**：區間 `[0, 1]`，和為 `7`
    - **`i = 2`**：區間 `[1, 2]`，和為 `5`
    - **`i = 3`**：區間 `[2, 3]`，和為 `10`
    - **`i = 4`**：區間 `[3, 4]`，和為 `9`
    - **`i = 5`**：區間 `[4, 5]`，和為 `5`
    - **`i = 6`**：區間 `[5, 6]`，和為 `9`
    - **`i = 7`**：區間 `[6, 7]`，和為 `11`
    - 更新最大值： prefix_l=[0,7,7,10,10,10,10,11]\text{prefix\_l} = [0, 7, 7, 10, 10, 10, 10, 11]prefix_l=[0,7,7,10,10,10,10,11]

---

### **步驟 3：計算後綴最大值（`postfix_k` 和 `postfix_l`）**

#### **計算 `postfix_k`**

1. 初始化 `postfix_k = [0, 0, 0, 0, 0, 0, 0, 0]`
2. 遍歷每個 `i`，從右往左計算長度為 `k` 的區間最大值：
    - **`i = 5`**：區間 `[5, 7]`，和為 `13`
    - **`i = 4`**：區間 `[4, 6]`，和為 `12`
    - **`i = 3`**：區間 `[3, 5]`，和為 `11`
    - **`i = 2`**：區間 `[2, 4]`，和為 `13`
    - **`i = 1`**：區間 `[1, 3]`，和為 `11`
    - **`i = 0`**：區間 `[0, 2]`，和為 `11`
    - 更新最大值： postfix_k=[13,13,13,13,13,13,0,0]\text{postfix\_k} = [13, 13, 13, 13, 13, 13, 0, 0]postfix_k=[13,13,13,13,13,13,0,0]

#### **計算 `postfix_l`**

1. 初始化 `postfix_l = [0, 0, 0, 0, 0, 0, 0, 0]`
2. 遍歷每個 `i`，從右往左計算長度為 `l` 的區間最大值：
    - **`i = 6`**：區間 `[6, 7]`，和為 `11`
    - **`i = 5`**：區間 `[5, 6]`，和為 `9`
    - **`i = 4`**：區間 `[4, 5]`，和為 `5`
    - **`i = 3`**：區間 `[3, 4]`，和為 `9`
    - **`i = 2`**：區間 `[2, 3]`，和為 `10`
    - **`i = 1`**：區間 `[1, 2]`，和為 `5`
    - **`i = 0`**：區間 `[0, 1]`，和為 `7`
    - 更新最大值： postfix_l=[11,11,11,9,9,9,11,0]\text{postfix\_l} = [11, 11, 11, 9, 9, 9, 11, 0]postfix_l=[11,11,11,9,9,9,11,0]

---

### **步驟 4：計算最大總和**

使用公式枚舉分界點 `i`，計算結果：

result=max⁡(prefix_k[i]+postfix_l[i+1],prefix_l[i]+postfix_k[i+1])\text{result} = \max(\text{prefix\_k}[i] + \text{postfix\_l}[i+1], \text{prefix\_l}[i] + \text{postfix\_k}[i+1])result=max(prefix_k[i]+postfix_l[i+1],prefix_l[i]+postfix_k[i+1])

- 對每個分界點計算後，結果為 **37**，對應區間為 `[0, 2]`（長度為 `k`）和 `[6, 7]`（長度為 `l`）。

---

### **最終答案**

最大總和=37\text{最大總和} = 37最大總和=37