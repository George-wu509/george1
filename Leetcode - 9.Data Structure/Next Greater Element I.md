Lintcode 1206
你有两个数组 `nums1`和`nums2`**（互不重复）**，其中`nums1`是`nums2`的子集。 在`nums2`的相应位置找到`nums1`所有元素的下一个更大数字。

`nums1`中的数字x的下一个更大数字是`nums2`中x右边第一个更大的数字。 如果它不存在，则为此数字输出-1。


**例子 1:**
```python
"""
输入: nums1 = [4,1,2], nums2 = [1,3,4,2].
输出: [-1,3,-1]
解释:
     对于第一个数组中的数字4，在第二个数组中找不到下一个更大的数字，因此输出-1。
     对于第一个数组中的数字1，第二个数组中的下一个更大数字是3。
     对于第一个数组中的数字2，第二个数组中没有下一个更大的数字，因此输出-1。
```
nums2中的4 在nums2=[1,3,4,2]後沒有更大的數字(後面只有2),  nums2中的1 在nums2=[1,3,4,2]的下一個數字更大數字是3 (後面數字是3,4,2), nums2中的2 在nums2=[1,3,4,2]後沒有更大的數字(後面沒有數字因為是最後一個)


**例子 2:**
```python
"""
输入: nums1 = [2,4], nums2 = [1,2,3,4].
输出: [3,-1]
解释:
     对于第一个数组中的数字2，第二个数组中的下一个更大数字是3。
     对于第一个数组中的数字4，第二个数组中没有下一个更大的数字，因此输出-1。
```



```python
    def next_greater_element(self, nums1: List[int], nums2: List[int]) -> List[int]:
        result = {}
        stack = []
        for num in reversed(nums2):
            while stack and num >= stack[-1]:
                stack.pop()
            result[num] = stack[-1] if stack else -1
            stack.append(num)
        return [result[num] for num in nums1]
```
pass

nums = [ 1,3,4,5,2 ]
解釋: 
step1. create stack跟res(紀錄往後最大的值)
step2. 反向的for loop從後面開始, 比較每個num跟棧頭. while loop如果num大於棧頭, 則pop()
step3. res紀錄stack最大的(應該是棧頭 stack[-1]), 以dict格式存入 res[num]: 棧頭, 代表num右邊有更大的(=棧頭)
step4. 將num壓入stack

2-> stack=[2], res = {2:-1}
4-> stack=[4], res = {2:-1, 4:-1}
3-> stack=[4,3], res = {2:-1, 4:-1, 3:4}
5-> stack=[5], res = {2:-1, 4:-1, 3:4, 5:-1}
1-> stack=[5,1], res = {2:-1, 4:-1, 3:4, 5:-1, 1:5}

### **LintCode 1206 - Next Greater Element I**

#### **解法分析**

本題的目標是對於 `nums1` 中的每個數字 `num`，找到它在 `nums2` 中的 **Next Greater Element**（下一個更大數），如果沒有更大的數，則返回 `-1`。其中，`nums1` 是 `nums2` 的子集。

解法的主要策略是使用 **單調遞減棧 (Monotonic Stack)** 來有效地找到 `nums2` 中每個元素的 Next Greater Element，然後用哈希表來快速查找 `nums1` 中元素的對應結果。

---

#### **解法步驟**

1. **反向遍歷 `nums2`，利用單調棧求 Next Greater Element**
    
    - 從 `nums2` 的最後一個元素開始向前遍歷。
    - 使用單調棧 `stack`，確保棧內元素是 **單調遞減** 的（棧底到棧頂遞減)。
    - 當當前元素 `num` **大於等於** `stack` 的棧頂元素時，則棧頂元素不可能是 `num` 的 Next Greater Element，因此將其彈出。
    - 這樣，當 `stack` 不為空時，棧頂元素即為 `num` 的 Next Greater Element，否則為 `-1`。
    - 記錄 `num` 對應的 Next Greater Element 到 `res` 字典中，然後將 `num` 入棧。
2. **對 `nums1` 中的元素查找結果**
    
    - 遍歷 `nums1`，對於每個 `num`，直接從 `res` 字典中取出其 Next Greater Element。

---

### **變數說明**

|變數名稱|說明|
|---|---|
|`nums1`|查詢數組，每個元素需在 `nums2` 找到 Next Greater Element|
|`nums2`|目標數組，`nums1` 是 `nums2` 的子集|
|`res`|字典，存儲 `nums2` 中每個元素對應的 Next Greater Element|
|`stack`|單調遞減棧，幫助快速找到 Next Greater Element|

---

### **範例**

#### **輸入**

`nums1 = [4,1,2] nums2 = [1,3,4,2]`

#### **處理流程**

1. **遍歷 `nums2` 並計算 Next Greater Element**
    
    - `nums2` 逆序遍歷：
        
        - `num = 2`，棧為空 → `res[2] = -1`，入棧 `stack = [2]`
        - `num = 4`，棧內 `2` 小於 `4`，彈出 `2`，棧為空 → `res[4] = -1`，入棧 `stack = [4]`
        - `num = 3`，棧內 `4` 大於 `3` → `res[3] = 4`，入棧 `stack = [4, 3]`
        - `num = 1`，棧內 `3` 大於 `1` → `res[1] = 3`，入棧 `stack = [4, 3, 1]`
    - 最終 `res = {1: 3, 3: 4, 4: -1, 2: -1}`
        
2. **查詢 `nums1`**
    
    - `res[4] = -1`
    - `res[1] = 3`
    - `res[2] = -1`

#### **輸出**

python

複製編輯

`[-1, 3, -1]`

---

### **時間與空間複雜度分析**

#### **時間複雜度**

- 逆序遍歷 `nums2` 時，每個元素最多 **進入和彈出棧一次**，總共 **O(N)**。
- 查詢 `nums1` 需要 **O(M)**，其中 `M = len(nums1)`。
- 總體時間複雜度為 **O(N + M)**。

#### **空間複雜度**

- `res` 字典存儲 `nums2` 的元素對應的 Next Greater Element，需要 **O(N)** 空間。
- 單調棧 `stack` 在最壞情況下存儲 `nums2` 的所有元素，最多佔用 **O(N)** 空間。
- 結果數組 **O(M)**。
- 總體空間複雜度為 **O(N + M)**。

---

### **其他解法想法**

1. **暴力解法 (O(NM))**
    
    - 對於 `nums1` 中的每個數 `num`，在 `nums2` 中線性掃描找到 `num`，然後再往右找第一個比 `num` 大的數。
    - 時間複雜度為 **O(NM)**，適用於 `nums2` 很小的情況。
2. **HashMap + Stack (O(N + M))**
    
    - 本題解法即是此法，利用 **單調遞減棧 + 哈希表** 來加速查詢。
3. **雙指針法 (O(N + M))**
    
    - 若 `nums2` **是遞增數列**，可以用雙指針來找到 Next Greater Element，否則無法應用。
4. **Segment Tree (O(N log N))**
    
    - 使用 **線段樹 (Segment Tree)** 來查詢範圍內的最大值，可以加速查詢過程。

---

### **總結**

- **最優解法：單調遞減棧 + 哈希表，時間複雜度 O(N + M)，空間複雜度 O(N + M)**。
- **若 `nums2` 很小，可以用暴力解法，時間複雜度 O(NM)**。
- **如果 `nums2` 是有序數組，可以用雙指針加速查詢**。