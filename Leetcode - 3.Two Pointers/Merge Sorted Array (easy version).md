Lintcode 64
提供兩個int array num1和num2，長度分別為m和n, 兩個陣列都是non-decreaing order，需要在num1合併陣列. 原地修改数组 `A` 把数组 `B` 合并到数组 `A` 的后面。
**样例 1：**
输入：
```python
"""
A = [1,2,3]
m = 3
B = [4,5]
n = 2
```
输出：
```python
[1,2,3,4,5]
```
解释：
经过合并新的数组为[1,2,3,4,5]  

**样例 2：**
输入：
```python
"""
A = [1,2,3,0,0,0], m = 3
B = [2,5,6], n = 2
```
输出：
```python
[1,2,2,3,5,6]
```
解释：
经过合并新的数组为[1,2,2,3,5,6]

```python

    def mergeSortedArray(self, A, m, B, n):
        sorted = []
        p1, p2 = 0, 0
        while p1 < m or p2 < n:
            if p1 == m:
                sorted.append(B[p2])
                p2 += 1
            elif p2 == n:
                sorted.append(A[p1])
                p1 += 1
            elif A[p1] < B[p2]:
                sorted.append(A[p1])
                p1 += 1
            else:
                sorted.append(B[p2])
                p2 += 1
        A[:] = sorted
```
pass

![[Screenshot 2026-10-23 060816.png]]
在 LintCode 64 / LeetCode 88（Merge Sorted Array）中，**`m`** 代表陣列 `A` 中原本存在的**有效元素數量**，而 **`n`** 代表陣列 `B` 的**總元素數量**。陣列 `A` 的真實記憶體容量大小為 $m + n$，其中後半段的 $n$ 個位置是預留給合併使用的空白空間。

直覺解法「看起來不像原地修改（In-place）」的原因在於：如果從陣列開頭（由小到大、由左至右）進行比較並寫入 `A`，會把 `A` 還沒被比較到的元素覆蓋掉，進而被迫開立一個額外陣列來暫存資料，空間複雜度變成 $O(m + n)$。

### 真正的原地修改原理：雙指標由後往前（From Right to Left）

由於陣列 `A` 的末端（索引 $m$ 至 $m + n - 1$）本來就是留空的，從右向左（由大到小）填充不會覆蓋任何尚未處理的有效資料：

1. **設定三個指標**：
    
    - 指標 `i = m - 1`：指向 `A` 的最後一個有效元素。
        
    - 指標 `j = n - 1`：指向 `B` 的最後一個元素。
        
    - 指標 `k = m + n - 1`：指向 `A` 的最後一個可用空間位置（最右端）。
        
2. **比較與填入**：
    
    - 每次比較 `A[i]` 與 `B[j]`，將較大者放入 `A[k]`，並將對應指標與 `k` 同步向左移動一位。
        
3. **邊界處理**：
    
    - 若 `B` 先處理完（`j < 0`），合併即結束（`A` 剩餘的前半段元素原本就已在正確位置）。
        
    - 若 `A` 的有效元素先處理完（`i < 0`），則將 `B` 剩餘的元素依序填入 `A` 前半段。

```
class Solution:

    def merge(self, A: list[int], m: int, B: list[int], n: int) -> None:
        i = m - 1
        j = n - 1
        k = m + n - 1

        # 由後往前比較，將較大者放入 A[k]
        while i >= 0 and j >= 0:
            if A[i] > B[j]:
                A[k] = A[i]
                i -= 1
            else:
                A[k] = B[j]
                j -= 1
            k -= 1

        # 若 B 還有剩餘元素，填入 A 的剩餘空間
        while j >= 0:
            A[k] = B[j]
            j -= 1
            k -= 1
```
