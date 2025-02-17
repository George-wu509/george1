
|                                                                                                                                                                                                                                                       |                                   |                                                           |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | --------------------------------------------------------- |
| Array  [[###### 數組]]<br>String  [[###### 字符串]]<br>Prefix Sum Array  [[###### 前綴和數組]]<br>Sorting* [[排序]]<br>line sweep [[掃描線]]                                                                                                                         |                                   | (18)<br>(36)<br>(16)<br>(2)<br>(0)                        |
| Bindary Search* [[二分法]]                                                                                                                                                                                                                               | Array二分查找, 搜索二维矩阵, 斐波纳契数列         | (19)                                                      |
| Two Pointers* [[雙指針]]                                                                                                                                                                                                                                 | 回文串, K数之和, 分割数组, 去除重复             | (30)                                                      |
| Linked List  [[###### 链表]]<br>Doubly linked list  [[###### 雙向链表]]                                                                                                                                                                                     | LRU                               | (24)<br>(2)                                               |
| Queue  [[###### 隊列]]<br>deque  雙向隊列<br>Monotone queue  [[###### 單調隊列]]                                                                                                                                                                                | [[BFS]], 层次遍历, 最短路径, 滑动窗口         | (8)<br>(6)<br>(25)                                        |
| Stack [[###### 棧]]<br>Monotonic stack  [[###### 單調棧]]                                                                                                                                                                                                 | [[DFS]], 括号匹配, 表达式求值, 递归<br>下一個最大 | (13)<br>(18)<br>(12)                                      |
| Hash Map [[###### 哈希表]]<br>Union find  [[###### 併查集]]<br>Iterator  [[###### 迭代器]]                                                                                                                                                                     |                                   | (27)<br>(22)<br>(12)                                      |
| Binary Tree [[###### 二叉樹]]<br>Binary Search Tree  [[###### 二叉搜索樹]]<br>Heap [[###### 堆]]<br>Trie [[###### 字典樹]]<br>Segmetn Tree  [[###### 線段樹]]<br>Balanced Binary Tree  [[###### 平衡樹]]<br>Binary Indexed Tree  [[###### 樹狀數組]]<br>Graph  [[###### 圖]] |                                   | (14)<br>(24)<br>(19)<br>(15)<br>(10)<br>(2)<br>(4)<br>(5) |
| DP*  [[動態規劃]]                                                                                                                                                                                                                                         |                                   | (8)                                                       |
| Math 數學  [[###### 數學]]                                                                                                                                                                                                                                |                                   | (1)                                                       |


# **<mark style="background: #FF5582A6;">數組（Array）</mark>的詳細介紹**
###### 數組
---

## **一、數組的原理與特點**

### **1. 原理**

- **數組（Array）** 是一種線性數據結構，它將元素以**連續的內存空間**存儲，並且每個元素都能通過索引訪問。
- **索引（Index）**：從0開始，每個索引對應數組中的一個元素。

### **2. 特點**

- **存儲連續性**：元素存儲在連續的內存地址中，支持隨機訪問，時間複雜度為 O(1)。
- **固定長度**：數組的大小通常是固定的，創建後無法動態擴展（Python 中的 `list` 是動態數組）。
- **高效查詢**：根據索引快速訪問元素。
- **低效插入/刪除**：當插入/刪除元素時，可能需要移動大量元素，時間複雜度為 O(n)。

---

## **二、具體例子**

假設有數組 `Array = [1, 4, 6, 8]`：

- 訪問第2個元素：`Array[1] = 4`
- 修改第4個元素：`Array[3] = 10` → 結果 `Array = [1, 4, 6, 10]`
- 插入元素 `5` 到索引 `2`：
    - 新數組：`[1, 4, 5, 6, 10]`（需移動元素）。

---

## **三、Python 實作**

```python
# 創建一個數組
array = [1, 4, 6, 8]

# 訪問元素
print("第2個元素:", array[1])  # 輸出: 4

# 修改元素
array[3] = 10
print("修改後的數組:", array)  # 輸出: [1, 4, 6, 10]

# 插入元素
array.insert(2, 5)  # 在索引2插入5
print("插入後的數組:", array)  # 輸出: [1, 4, 5, 6, 10]

# 刪除元素
array.remove(6)  # 刪除值為6的元素
print("刪除後的數組:", array)  # 輸出: [1, 4, 5, 10]

# 遍歷數組
for i in array:
    print(i, end=" ")

```

---

## **四、LeetCode Array 題目簡單描述及解法**

以下是 LeetCode 上幾個經典的 **Easy** 到 **Medium** 題目，並附上簡單描述及解法：

以下是整理的 LintCode 中涉及数组的入门、简单到中等难度的 50 道题目，包括题目编号、题目名称（英文）、题目简述（中文）、样例及解法：

| 题目编号                 | 题目名称（英文）                                                                                  | 题目简述（中文）                                      | 样例                                                                                                 | 解法                                          |
| -------------------- | ----------------------------------------------------------------------------------------- | --------------------------------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| 100<br><br>(e)       | [[Remove Duplicates]]删除排序数组中的重复数字                                                         | 給定一個排序數組，刪除重複元素，並返回數組的新長度                     | 输入：<br>nums = [1,1,2]<br>输出：<br>[1,2]                                                              | 使用雙指針：一個指針遍歷數組，另一個指針記錄非重複元素的位置。             |
| 30<br>*<br>(m)<br>   | [[Insert Interval]]插入区间                                                                   | 给定一组非重叠的区间和一个新的区间，将新区间合并到已有的区间中（可能需要合并重叠的部分）。 | 输入: intervals = [(1,2),(3,5),(6,7),(8,10),(12,16)], newInterval = (4,9) 输出: [(1,2),(3,10),(12,16)] | 遍历已有区间，找到需要合并的位置，进行合并并插入新的区间。               |
| 8<br>*<br>(e)        | [[Rotate String]]旋转字符数组                                                                   | 给定一个字符串和一个偏移量，根据偏移量旋转字符串。                     | 输入: "abcdefg", offset = 3 输出: "efgabcd"<br><br>O(n),O(n)                                           | 将字符串分为两部分，分别反转，然后再整体反转。                     |
| 39<br><br>(e)        | [[Recover Rotated Sorted Array]]恢复旋转排序数组                                                  | 恢复被旋转过的排序数组，使其重新有序。                           | 输入: [4, 5, 1, 2, 3] 输出: [1, 2, 3, 4, 5]<br><br>O(n),O(1)                                           | 找到旋转点，将数组分为两部分，分别反转，然后再整体反转。                |
| 62<br>*<br>(m)<br>   | [[Search in Rotated Sorted Array\|Search in Rotated Sorted Array]]搜索旋转排序数组<br><br>(也在二分法) | 在旋转过的排序数组中搜索目标值，返回其下标。                        | 输入: [4, 5, 6, 7, 0, 1, 2], target = 0 输出: 4<br><br>O(nlogn)?O(n)?                                  | 使用二分查找，判断中间元素与左右边界的关系，确定搜索范围。               |
| 63<br><br>(m)<br>    | [[Search in Rotated Sorted Array II]]搜索旋转排序数组 II                                          | 在旋转过的排序数组中搜索目标值，数组中可能包含重复元素。                  | 输入: [2,5,6,0,0,1,2], target = 0 输出: true                                                           | 使用二分查找，判断中间元素与目标值的关系，并处理可能的重复元素情况。          |
| 46<br>*<br>(e)       | [[Majority Number]]主元素                                                                    | 找到数组中的主元素，出现次数超过数组长度的一半。                      | 输入: [1, 1, 1, 2, 2] 输出: 1<br><br>O(n),O(1)                                                         | 使用摩尔投票算法，维护一个候选元素和计数器，遍历数组更新候选元素。           |
| 50<br><br>(e)        | [[Product of Array Exclude Itself]]数组剔除元素后的乘积                                             | 给定一个整数数组，返回一个新数组，其中每个元素是原数组中除自身外其他元素的乘积。      | 输入: [1, 2, 3] 输出: [6, 3, 2]<br><br>O(n),O(n)                                                       | 使用前缀积和后缀积，分别计算每个位置左侧和右侧的乘积，然后相乘得到结果。        |
| 1310<br>*<br>(m)     | [[Product of Array Except Self]]数组除了自身的乘积                                                 | 给定一个整数数组，返回每个元素除自身外其他元素的乘积，要求不能使用除法运算。        | 输入: [1,2,3,4] 输出: [24,12,8,6]<br><br>O(n),O(n)                                                     | 使用前缀乘积和后缀乘积，逐个计算每个位置的结果。                    |
| 111<br>*<br>(e)      | [[Climbing Stairs]]爬楼梯                                                                    | 一次可以爬 1 步或 2 步，求爬到第 n阶台阶的总方法数。                | 输入: n = 3 输出: 3<br><br>O(n),O(n)                                                                   | 使用动态规划或斐波那契数列公式递推计算方法数。                     |
| 162<br>*<br>(m)<br>  | [[Set Matrix Zeroes]]矩阵归零                                                                 | 给定一个二维矩阵，若某个元素为 0，则将该元素所在行和列的所有元素设为 0。        | 输入: [ [1,2,3],[4,0,6],[7,8,9] ] 输出: [ [1,0,3],[0,0,0],[7,0,9] ]<br><br>O(m*n),O(m+n)               | 使用额外空间记录 0 的位置，或利用矩阵的第一行和第一列作为标记。           |
| 101<br><br>(e)       | [[Remove Duplicates from Sorted Array II]]删除排序数组中的重复数字（二)                                 | 删除排序数组中出现超过两次的重复项，使每个元素最多出现两次，并返回新的数组长度。      | 输入: [1,1,1,2,2,3] 输出: 5 ([1,1,2,2,3])                                                              | 使用双指针，一个指向当前遍历位置，一个指向更新位置，控制每个元素的出现次数不超过两次。 |
| 156<br>**<br>(e)<br> | [[Merge Intervals]]合并区间                                                                   | 给定一组区间，合并所有重叠的区间。                             | 输入: [(1,3),(2,6),(8,10),(15,18)] 输出: [(1,6),(8,10),(15,18)]                                        | 按区间起点排序，遍历并合并重叠区间。                          |
| 1355<br>*<br>(e)     | [[Pascal's Triangle]]杨辉三角                                                                 | 给定一个整数 n，生成帕斯卡三角形的前 n 行。                      | 输入: 5 输出: [ [1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1] ]<br><br>O(n²),O(n²)                           | 使用循环逐行生成，每行的值由上一行计算得到。                      |
| 1354<br><br>(e)      | [[Pascal's Triangle II]]杨辉三角形II                                                           | 给定一个索引 k，返回帕斯卡三角形的第 k 行。(從0開始)                | 输入: 0 输出: [1]<br>输入: 3 输出: [1,3,3,1]<br>從0行開始,第n行n+1數字                                             | 使用动态规划，仅存储上一行值，逐步生成下一行。                     |
| 82<br><br>(e)        | [[Single Number]]落单的数                                                                     | 给定一个非空数组，其中每个元素出现两次，只有一个元素出现一次，找出该元素。         | 输入: [4,1,2,1,2] 输出: 4<br><br>O(n),O(1)                                                             | 使用XOR异或操作，所有出现两次的元素会抵消为 0，剩下的即为单独的元素。       |
| 1320<br>*<br>(e)     | [[Contains Duplicate]]包含重复值                                                               | 判断数组中是否存在重复元素。                                | 输入: [1,2,3,1] 输出: true                                                                             | 使用哈希表记录出现的元素，若遇到重复则返回 true。                 |
| 397<br>*<br>(e)<br>G | 最长上升连续子序列 [[Longest Continuous Increasing Subsequence]]                                   | 找到一個數組中，最长的连续严格上升子序列的长度。                      | 输入: nums = [ 1,5,2,3,4 ]  <br>输出: 3   ([2,3,4])<br><br>O(n),O(1)                                   | 使用遍历一次数组的方法，记录当前的连续上升子序列长度，动态更新最大值。         |



---

## **五、選取三題詳細解釋**

### **1. LeetCode 1: Two Sum**(lintcode 56)

**題目描述**：  
給定一個整數數組 `nums` 和一個目標值 `target`，找出兩個數字，使得它們的和等於目標值，並返回它們的索引。
example:
nums = [2,7,11,15], target = 9
output = [0,1]

**解法思路**：

- 使用 **哈希表** 記錄已經訪問過的數字及其索引。
- 遍歷數組時，檢查當前數字與目標值的差是否已在哈希表中。
```python
def twoSum(nums, target):
    hashmap = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in hashmap:
            return [hashmap[complement], i]
        hashmap[num] = i

```

**為什麼使用 Array**：

- 數組支持索引訪問，可以快速找到元素。
- 哈希表可以加速查找，將時間複雜度降到 O(n)。

---

### **2. LeetCode 26: Remove Duplicates**(lintcode 100)

**題目描述**：  
給定一個排序數組，刪除重複元素，並返回數組的新長度。要求inplace(就地), 而且空間複雜度O(1)
Example:
nums = [1,1,2]
output = [1,2]

**解法思路**：

- 使用 **雙指針**：一個指針遍歷數組，另一個指針記錄非重複元素的位置。

**Python 代碼**：
```python
def removeDuplicates(nums):
    if not nums:
        return 0
    slow = 0
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]
    return slow + 1

```

**為什麼使用 Array**：

- 數組的連續性允許使用雙指針法高效遍歷和修改元素。

比較其他兩種算法:
## **使用 `set()` 方法**

### 思路：

- 使用集合 `set` 來存儲唯一元素，遍歷整个数组時將元素放入集合。
- 最後將集合內的元素複製回 `nums` 数组的前部分。

### 時間與空間複雜度：

- 時間複雜度: **O(n)**。
- 空間複雜度: **O(n)**，因為 `set` 需要额外空間來存儲唯一元素。
```python
def remove_duplicates(nums):
    if not nums:
        return 0

    unique_nums = list(set(nums))  # 使用set去重并转换为list
    unique_nums.sort()  # 排序回原数组顺序（因为数组已排序）

    for i in range(len(unique_nums)):
        nums[i] = unique_nums[i]  # 覆蓋原始数组
    
    return len(unique_nums)

```
## **3. 使用 `hash map` 方法**

### 思路：

- 使用 `hash map`（字典）來跟蹤唯一元素並計數。
- 然後將字典的鍵值複製回数组的前部分。

### 時間與空間複雜度：

- 時間複雜度: **O(n)**。
- 空間複雜度: **O(n)**，因為字典需要额外空間來存儲唯一元素。
### 代码：
```python
def remove_duplicates(nums):
    if not nums:
        return 0

    unique_map = {}  # 使用hash map記錄唯一元素
    for num in nums:
        unique_map[num] = True
    
    unique_nums = list(unique_map.keys())  # 取出所有唯一元素
    unique_nums.sort()  # 确保順序符合排序数组要求

    for i in range(len(unique_nums)):
        nums[i] = unique_nums[i]  # 覆蓋原始数组
    
    return len(unique_nums)

```

## **三种方法比较：**

| 方法                 | 時間複雜度 | 空間複雜度 | 備註                     |
| ------------------ | ----- | ----- | ---------------------- |
| 雙指針 (Two Pointers) | O(n)  | O(1)  | 最优解，符合 **in-place** 要求 |
| `set()` 方法         | O(n)  | O(n)  | 額外空間消耗，排序陣列可保序         |
| `hash map` 方法      | O(n)  | O(n)  | 類似 `set()`，但略顯冗餘       |

---

### **3. LeetCode 88: Merge Sorted Array**(lintcode 64)

**題目描述**：  
合併兩個有序數組，將結果存儲到第一個數組中，並保證有序。
Example:
A = [1,2,3], B = [4,5]
output = [1,2,3,4,5]

**解法思路**：

- 使用 **雙指針法** 從後向前遍歷兩個數組，將較大的元素放入目標數組末尾。

**Python 代碼**：
```python
def merge(nums1, m, nums2, n):
    i, j, k = m - 1, n - 1, m + n - 1
    while i >= 0 and j >= 0:
        if nums1[i] > nums2[j]:
            nums1[k] = nums1[i]
            i -= 1
        else:
            nums1[k] = nums2[j]
            j -= 1
        k -= 1
    while j >= 0:  # 若 nums2 仍有剩餘
        nums1[k] = nums2[j]
        j -= 1
        k -= 1

```

**為什麼使用 Array**：

- 使用雙指針法可高效地將兩個數組合併，節省空間（直接修改 nums1）。

---

### **總結**

- **數組（Array）** 是一種基礎數據結構，支援快速訪問與遍歷。
- 常用技巧包括**雙指針**、**哈希表** 和 **動態規劃**，適合求解元素查找、數組合併及子數組問題。
- 通過掌握 LeetCode 的這些題目，可以熟練運用數組解決實際問題。


# **<mark style="background: #FF5582A6;">字符串（String）</mark>的詳細介紹**
###### 字符串
---
## **一、字符串（String）的原理與特點**

### **1. 原理**

- **字符串（String）** 是字符的序列，是一種線性數據結構，用於表示文本數據。
- 在 Python 中，字符串是**不可變**的，一旦創建就無法修改。所有的字符串操作（例如拼接、切片）會返回一個新的字符串。

### **2. 特點**

- **索引訪問**：字符串中的字符可以通過索引快速訪問（時間複雜度為 O(1)）。
- **不可變性**：字符串的內容無法修改，只能創建新的字符串。
- **遍歷高效**：支持逐個字符進行遍歷操作。
- **常用操作**：
    - 查找、拼接、切片、反轉、大小寫轉換等。

### **3. 時間複雜度**

| 操作          | 時間複雜度 |
| ----------- | ----- |
| 訪問第 iii 個字符 | O(1)  |
| 拼接字符串       | O(n)  |
| 查找子串        | O(n)  |
| 遍歷字符串       | O(n)  |

---

## **二、具體例子**

假設有字符串 `String = "hello"`：

- **訪問第2個字符**：`String[1] = 'e'`
- **字符串拼接**：`"hello" + " world" = "hello world"`
- **字符串切片**：`String[1:4] = "ell"`
- **反轉字符串**：`String[::-1] = "olleh"`

---

## **三、Python 實作**

```python
# 創建字符串
string = "hello"

# 訪問字符串中的字符
print(string[1])  # 輸出: e

# 字符串拼接
new_string = string + " world"
print(new_string)  # 輸出: hello world

# 字符串切片
print(string[1:4])  # 輸出: ell

# 字符串反轉
reversed_string = string[::-1]
print(reversed_string)  # 輸出: olleh

# 遍歷字符串
for char in string:
    print(char, end=" ")  # 輸出: h e l l o

# 查找子串
print("ll" in string)  # 輸出: True

# 字符串長度
print(len(string))  # 輸出: 5

```


---

## **四、LeetCode 字符串題目描述及解法**

以下是整理的 LintCode 中涉及字符串的入门、简单到中等难度的 50 道题目，包括题目编号、题目名称（英文）、题目简述（中文）、样例及解法：

| 题目编号                      | 题目名称（英文）                                                     | 题目简述（中文）                                                            | 样例                                                                                                                | 解法                                                          |
| ------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| 13<br>*<br>(e)            | [[Implement strStr]]字符串查找                                    | 实现 `strStr()` 函数，返回子字符串在母字符串中首次出现的索引，若不存在则返回 -1。                    | 输入: <br>source = "abcdabcdefg", <br>target = "bcd" <br>输出: 1<br>                                                  | 使用双指针從0,0出發遍历源字符串，检查目标字符串是否匹配                               |
| 594<br>**<br>(h)          | 字符串查找 II [[strStr II]]                                       | 在字符串中查找子字符串，返回第一次出现的起始索引，若未找到返回-1。                                  | 输入: <br>source = "abcdabcdefg", <br>target = "bcd" <br>输出: 1<br>O(m+n),O(1)                                       | 用Rabin-Karp (滾動哈希) 字符串匹配演算法匹配target, source的hash value      |
| 384<br>*<br>(m)           | [[Longest Substring Without Repeating Characters]]無重複字符的最長子串 | 給定一個字符串，找出其中不包含重複字符的最長子串的長度。                                        | 输入：<br>s = "abcabcbb"<br>输出：<br>output = 3  ("abc")                                                               | 使用滑動窗口來維護一個當前有效的無重複字符子串。當set()出現重複字符時，窗口左邊界右移以維持無重複狀態。      |
| 422<br><br>(e)            | [[Length of Last Word]]最后一个单词的长度                             | 返回字符串中最后一个单词的长度，单词之间以空格分隔。                                          | 输入: "Hello World" 输出: 5                                                                                           | 从字符串末尾向前遍历，跳过末尾空格，计算最后一个单词的长度。                              |
| 408<br><br>(e)            | [[Add Binary]]二进制求和                                          | 给定两个二进制字符串，返回它们的和（用二进制字符串表示）。                                       | 输入: "11", "1" 输出: "100"                                                                                           | 从后向前遍历两个字符串，逐位相加并处理进位问题。                                    |
| 415<br>*<br>(m)           | [[Valid Palindrome]]有效回文串                                    | 判断一个字符串是否为回文，只考虑字母和数字字符，忽略大小写。                                      | 输入: "A man, a plan, a canal: Panama" 输出: true                                                                     | 使用双指针从字符串两端向中间移动，跳过非字母数字字符，比较字符是否相同。                        |
| 837<br>*<br>(m)           | [[Palindromic Substrings]]回文子串                               | 给定一个字符串，计算字符串中回文子串的个数。                                              | 输入: "aaa" 输出: 6                                                                                                   | 使用中心扩展法或动态规划，逐个检查每个子串是否为回文并计数。                              |
| 891<br><br>(m)<br>        | [[Valid Palindrome II]]有效回文串（二）                              | 判断一个字符串是否可以通过删除最多一个字符变成回文。                                          | 输入: "abca" 输出: true                                                                                               | 使用双指针从两端向中间移动，遇到不匹配时尝试跳过左指针或右指针的字符继续判断是否为回文。                |
| 627<br>*<br>(e)           | [[Longest Palindrome]]最长回文串                                  | 给定一个字符串，找到能够通过重新排列组成的最长回文串的长度。                                      | 输入: "abccccdd" 输出: 7                                                                                              | 使用hash table统计各字符的出现次数，计算偶數並刪去 or用Counter                   |
| 53<br><br>(e)             | [[Reverse Words in a String]]翻转字符串                           | 给定一个字符串，逐个翻转字符串中的每个单词。                                              | 输入: "the sky is blue" 输出: "blue is sky the"                                                                       | 先去除多余空格，将字符串拆分为单词列表，反转列表后再拼接成字符串。                           |
| 927<br>*<br>(m)<br>       | [[Reverse Words in a String II]]I翻转字符串II                     | 给定一个字符数组，逐个翻转字符串中的每个单词。                                             | 输入: "the sky is blue" 输出: "blue is sky the"                                                                       | 先整体反转字符数组，然后逐个反转每个单词。                                       |
| 1173<br><br>(e)<br>       | [[Reverse Words in a String III]]翻转字符串 III                   | 给定一个字符串，逐个反转字符串中的每个单词，但保持单词的顺序不变。                                   | 输入: "Let's take LeetCode contest" 输出: "s'teL ekat edoCteeL tsetnoc"                                               | 拆分字符串为单词列表，逐个反转每个单词后拼接成字符串。                                 |
| 1282<br><br>(e)           | [[Reverse Vowels of a String]]翻转字符串中的元音字母                    | 反转字符串中的元音字母。                                                        | 输入: "hello" 输出: "holle"                                                                                           | 使用双指针从(0, N)向中间移动，如果兩個指針都是元音, 交换元音字母的位置。                    |
| 773<br><br>(e)            | [[Valid Anagram]]有效的字母异位词                                    | 判断两个字符串是否是字母异位词，即两个字符串包含的字符相同，顺序可以不同。                               | 输入: s = "anagram", t = "nagaram" 输出: true                                                                         | 使用两个dict统计两个字符串中各字符的出现次数，比较是否相同。                            |
| 1270<br><br>(e)<br>       | [[Ransom Note]]勒索信                                           | 判断一个字符串能否由另一个字符串中的字符构成。                                             | 输入: ransomNote = "aa", magazine = "aab" 输出: true                                                                  | 使用兩個Counter统计杂志字符串中各字符的出现次数，检查赎金信字符串中的字符是否都能由杂志提供。          |
| 209<br>*<br>(e)           | [[First Unique Character in a String]]第一个只出现一次的字符            | 找到字符串中第一个不重复的字符，返回其索引，若不存在则返回 -1。                                   | 输入: "leetcode" 输出: 0                                                                                              | 使用哈希表统计各字符的出现次数，然后遍历字符串找到第一个出现次数为 1 的字符。                    |
| 655<br><br>(e)            | [[Add Strings]]大整数加法                                         | 给定两个非负整数的字符串表示，计算它们的和，并以字符串形式返回。                                    | 输入: num1 = "11", num2 = "123" 输出: "134"                                                                           | 从后向前遍历两个字符串，逐位相加并处理进位问题。                                    |
| 1243<br><br>(e)           | [[Number of Segments in a String]]字符串中的单词数                   | 统计字符串中的单词数量，单词由非空格字符组成。                                             | 输入: "Hello, my name is John" 输出: 5                                                                                | 遍历字符串，统计相邻非空格字符段的数量。                                        |
| 213<br><br>(e)<br>        | [[String Compression]]字符串压缩                                  | 对字符串进行基本的压缩，使用字符计数的方式，将压缩后的字符数组长度返回。                                | 输入: ['a','a','b','b','c','c','c'] 输出: 返回新长度 6，字符数组变为 ['a','2','b','2','c','3']                                    | 使用双指针遍历字符数组，记录每个字符的出现次数，进行原地修改。                             |
| 1227<br><br>(e)<br>!!<br> | [[Repeated Substring Pattern]]重复的子串模式                        | 判断一个非空字符串是否可以由它的一个子串重复多次构成。                                         | 输入: "abab" 输出: true                                                                                               | 将字符串自身拼接一次，去掉头尾字符后检查是否包含原字符串。                               |
| 1193<br><br>(e)<br>       | [[Detect Capital]]检测大写的正确性                                   | 判断一个单词中的大写字母使用是否正确，正确的情况包括全部大写、全部小写、只有首字母大写。                        | 输入: "USA" 输出: true                                                                                                | 检查字符串是否全部大写、全部小写或只有首字母大写。                                   |
| 1178<br><br>(e)<br>       | [[Student Attendance Record I]]学生出勤记录 I                      | 判断一个学生的出勤记录是否符合奖励条件：连续迟到不超过两次，总缺席次数不超过一次。                           | 输入: "PPALLP" 输出: true                                                                                             | 遍历字符串，统计 'A' 的数量，并检查是否存在连续三个 'L'。                           |
| 1169<br>*<br>(m)          | [[Permutation in String]]字符串的排列                              | 给定两个字符串，判断 s1 的某个排列是否是 s2 的子串。                                      | 输入: s1 = "ab", s2 = "eidbaooo" 输出: true                                                                           | 使用滑动窗口维护 s2 中与 s1 等长的子串，统计字符频次并比较是否相等。                      |
| 1086<br><br>(e)           | [[Repeated String Match]]重复字符串匹配                             | 判断一个字符串需要重复多少次，才能使另一个字符串成为其子串。                                      | 输入: a = "abcd", b = "cdabcdab" 输出: 3                                                                              | 重复字符串 a，直到其长度大于等于 b，检查是否包含 b，若不包含则再重复一次，若仍不包含则返回 -1。        |
| 1079<br><br>(e)<br>       | [[Count Binary Substrings]]连续子串计数                            | 计算二进制字符串中连续子字符串的数量，这些子字符串中 0 和 1 的个数相等且相邻。                          | 输入: "00110011" 输出: 6                                                                                              | 统计连续相同字符的数量，逐对比较相邻组的大小，取较小值累加到结果中。                          |
| 1041<br>*<br>(m)<br>      | [[Reorganize String]]重构字符串                                   | 给定一个字符串，重新排列使得相邻字符不相同，若无法实现则返回空字符串。                                 | 输入: "aab" 输出: "aba"                                                                                               | 使用优先队列存储字符及其频次，按照频次从高到低排列，逐个取出字符并重新排列。                      |
| 1025<br><br>(m)           | [[Custom Sort String]]自定义字符串排序                               | 给定两个字符串 order 和 s，根据 order 的顺序对 s 中的字符重新排序。                         | 输入: order = "cba", s = "abcd" 输出: "cbad"                                                                          | 使用哈希表记录 order 中字符的优先级，按照优先级对 s 进行排序。                        |
| 1013<br>*<br>(e)<br>      | [[Unique Morse Code Words]]独特的摩尔斯编码                          | 给定一个字符串数组，将每个单词翻译为摩斯密码，返回不同翻译后的单词数量。                                | 输入: words = ["gin", "zen", "gig", "msg"] 输出: 2                                                                    | 使用哈希表存储摩斯密码翻译后的单词，统计唯一的单词数量。                                |
| 1394<br><br>(e)           | [[Goat Latin]]山羊拉丁文                                          | 按照规则将一个句子中的单词转换为 Goat Latin：若单词以元音开头，添加 "ma"；若以辅音开头，移到单词末尾并添加 "ma"。 | 输入: "I speak Goat Latin" 输出: "Imaa peaksmaaa oatGmaaaa atinLmaaaaa"                                               | 遍历单词，判断首字母是元音还是辅音，进行相应转换并添加 'ma' 和重复的 'a'。                  |
| 1435<br><br>(m)<br>!!     | [[Find And Replace in String]]字符串中的查找与替换                     | 给定一个字符串、索引数组、源数组和目标数组，将字符串中匹配源数组的子串替换为目标数组对应的值。                     | 输入: s = "abcd", indexes = [0, 2], sources = ["a", "cd"], targets = ["eee", "ffff"] 输出: "eeebffff"                 | 按照索引逆序遍历，逐个检查子串是否匹配，若匹配则进行替换。                               |
| 1425<br><br>(e)<br>       | [[Backspace String Compare]]比较含退格的字符串                        | 判断两个字符串在模拟退格操作后是否相等，'#' 表示退格符。                                      | 输入: s = "ab#c", t = "ad#c" 输出: true                                                                               | 從後往前遍歷 s 和 t, 可以直接跳過被刪除的字母 ，最后比较栈中的字符串是否相等。                 |
| 1510<br><br>(e)           | [[Buddy Strings]]亲密字符串                                       | 判断两个字符串是否可以通过交换其中的两个字符变成相等。                                         | 输入: A = "aaaaaaabc", B = "aaaaaaacb"<br>输出: true                                                                  | 若字符串长度不同则直接返回 false；若相等则检查是否有重复字符；否则记录不同位置的字符，判断是否可以通过交换相等。 |
| 171<br>**<br>(m)          | 乱序字符串 [[Anagrams]]                                           | 给定一个字符串列表，找出其中所有乱序的字符串组。                                            | 输入:  <br>strs = ["eat", "tea", "tan", "ate", "nat", "bat"]  <br>输出: [ ["eat","tea","ate"],["tan","nat"],["bat"] ] | 将字符串排序后作为key存入字典，按key分组。                                    |
| 1127<br><br>(m)           | 在字符串中添加粗体标签 [[Add Bold Tag in String]]                       | 在字符串中所有出现在给定单词列表中的子串周围添加粗体标签，结果字符串用最少的标签包裹所有符合条件的部分。                | 输入:  <br>s = "abcxyz123", words = ["abc","123"]  <br>输出: "<b>abc</b>xyz<b>123</b>"                                | 使用布尔数组标记哪些位置需要加粗体标签，再按范围合并标签。                               |
| 10<br><br>(m)             | 字符串的不同排列 [[String Permutation II]]                           | 给定一个字符串，返回其所有不同的排列（考虑字符重复）。                                         | 输入:  <br>s = "aab"  <br>输出: ["aab", "aba", "baa"]                                                                 | 使用DFS生成排列，并用集合去重或在递归中跳过重复字符。                                |
| 1169<br>*<br>(m)          | [[String Permutation]] 字符串的排列                                | 判断一个字符串是否是另一个字符串的排列。                                                | 输入:  <br>s1 = "abc", s2 = "bca"  <br>输出:  <br>true                                                                | 使用哈希表记录字符频率，检查两个字符串是否匹配。                                    |

## **五、選取三題詳細解釋**

### **1. LeetCode 125: 有效的回文串（Valid Palindrome）**(lintcode 415)

**題目描述**：  
給定一個字符串，忽略非字母和數字字符，並忽略大小寫，判斷該字符串是否為回文串。
Example:
s = "A man, a plan, a canal: Panama"
output = True

**解法思路**：

- 使用 **雙指針**，從字符串的兩端向中間遍歷，跳過非字母數字字符。
- 比較字符是否相等。

**Python 代碼**：
```python
def isPalindrome(s: str) -> bool:
    left, right = 0, len(s) - 1
    while left < right:
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        if s[left].lower() != s[right].lower():
            return False
        left += 1
        right -= 1
    return True

```

**為什麼使用 String**：

- 字符串本質是字符的序列，雙指針適合遍歷和比較字符。

---

### **2. LeetCode 242: 有效的字母異位詞（Valid Anagram）**(lintcode 773)

**題目描述**：  
給定兩個字符串 `s` 和 `t`，判斷它們是否為字母異位詞（字符重排後相同）。
Example:
s = "anagram", t = "nagaram"
output = True

**解法思路**：

- 使用 **哈希表** 統計兩個字符串中每個字符的出現次數，並進行比較。
- 也可以排序兩個字符串，若排序後相等則為異位詞。

**Python 代碼**：
```python
# method1 - sorted
def isAnagram(s: str, t: str) -> bool:
    return sorted(s) == sorted(t)

# method2 - Hash table
def isAnagram(s: str, t: str) -> bool:
    if len(s) != len(t):
        return False
    
    # 創建哈希表
    count = {}
    
    # 計數字符串 s 中的字母
    for char in s:
        count[char] = count.get(char, 0) + 1
    
    # 遍歷字符串 t，減少對應字母的計數
    for char in t:
        if char not in count:
            return False
        count[char] -= 1
        if count[char] < 0:
            return False
    
    # 確保哈希表中的所有值都為零
    return all(value == 0 for value in count.values())

```

**為什麼使用 String**：

- 字符串操作如排序和哈希計數，適合檢查字符的頻率和順序。

---

### **3. LeetCode 3: 無重複字符的最長子串（Longest Substring Without Repeating Characters）**(lintcode 384)

**題目描述**：  
給定一個字符串，找出其中不包含重複字符的最長子串的長度。
Example:
s = "abcabcbb"
output = 3

**解法思路**：

- 使用 **滑動窗口** 來維護一個當前有效的無重複字符子串。
- 當出現重複字符時，窗口左邊界右移以維持無重複狀態。
- 從L(左指針)=0,R(右指針)=0開始, R在迴圈每次往右移, 並check新加的char是否存在set裡, 如果沒有就加入set, 然R再往右移. 如果已經存在set裡, 則L往右移並remove set裡原本L對應的char, 一直到沒有重複. 

**Python 代碼**：
```python
def lengthOfLongestSubstring(s: str) -> int:
    char_set = set()
    left = 0
    max_len = 0
    for right in range(len(s)):
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        char_set.add(s[right])
        max_len = max(max_len, right - left + 1)
    return max_len

```

**為什麼使用 String**：

- 字符串可以通過滑動窗口逐個字符遍歷並維護當前子串狀態。

---

## **總結**

- **字符串（String）** 適合處理文本數據，包括字符比較、拼接、反轉等操作。
- 常用算法包括**雙指針**、**滑動窗口** 和 **哈希表**，適合解決子串、字符頻率和字符串匹配等問題。
- 通過 LeetCode 的這些題目，能夠熟練掌握字符串處理技巧。


# **<mark style="background: #BBFABBA6;">前綴和數組（Prefix Sum Array）</mark>的詳細介紹**
###### 前綴和數組
---

## **一、前綴和數組的原理與特點**

### **1. 原理**

- **前綴和數組（Prefix Sum Array）** 是一種預處理技術，用於快速計算數組中任意區間的和。
    
- **核心思想**：通過構建一個前綴和數組 `prefix_sum`，使得數組 nums[i]的前 i個元素的和可以快速獲取。
    
- 定義：
    
    $prefix\_sum[i] = nums[0] + nums[1] + \dots + nums[i-1]$
    - **prefix_sum** 的第 iii 個位置表示原數組前 i−1i-1i−1 個元素的總和。
    
- **區間和計算公式**：  
    對於原數組的任意區間 [i,j][i, j][i,j]：
    
    $sum(i, j) = prefix\_sum[j+1] - prefix\_sum[i]$
    
    這樣，區間和的計算時間複雜度可從 O(n)優化為 O(1)。
    

---

### **2. 特點**

- **高效查詢**：查詢區間和的時間複雜度為 O(1)O(1)O(1)。
- **預處理代價**：構建前綴和數組需要一次遍歷，時間複雜度為 O(n)O(n)O(n)。
- **不可修改**：適合靜態數組，若原數組有頻繁的修改操作，前綴和數組需要重新計算。

---

## **二、具體例子**

假設原數組 `nums = [1, 4, 6, 8]`：

1. **構建前綴和數組**：
    
    - `prefix_sum[0] = 0`
    - `prefix_sum[1] = 1`
    - `prefix_sum[2] = 1 + 4 = 5`
    - `prefix_sum[3] = 1 + 4 + 6 = 11`
    - `prefix_sum[4] = 1 + 4 + 6 + 8 = 19`  
        結果：`prefix_sum = [0, 1, 5, 11, 19]`
2. **查詢區間和**：
    
    - 求區間 `[1, 3]` 的和： =nums[1]+nums[2]+nums[3] sum(1,3)=prefix_sum[4]−prefix_sum[1]=19−1=18sum(1, 3) = prefix\_sum[4] - prefix\_sum[1] = 19 - 1 = 18sum(1,3)=prefix_sum[4]−prefix_sum[1]=19−1=18

---

## **三、Python 實作**

```python
class PrefixSum:
    def __init__(self, nums):
        self.prefix_sum = [0] * (len(nums) + 1)
        for i in range(1, len(self.prefix_sum)):
            self.prefix_sum[i] = self.prefix_sum[i - 1] + nums[i - 1]

    def range_sum(self, left, right):
        # 返回區間 [left, right] 的總和
        return self.prefix_sum[right + 1] - self.prefix_sum[left]

# 使用示例
nums = [1, 4, 6, 8]
prefix = PrefixSum(nums)

# 查詢區間和 [1, 3]
print(prefix.range_sum(1, 3))  # 輸出 18

```

---

## **四、LeetCode 前綴和題目描述及解法**

以下是整理的 LintCode 中涉及前缀和数组的入门、简单到中等难度的 50 道题目，包括题目编号、题目名称（英文）、题目简述（中文）、样例及解法：

| 题目编号                 | 题目名称（英文）                                           | 题目简述（中文）                                 | 样例                                                                                 | 解法                                                                                    |
| -------------------- | -------------------------------------------------- | ---------------------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| 138<br>*<br>(e)<br>  | [[Subarray Sum]]子数组之和为零                            | 给定一个整数数组，找到和为零的子数组，返回第一个出现的子数组的起始和结束下标。  | 输入: [-3, 1, 2, -3, 4] 输出: [0, 2]<br><br>O(n),O(n)                                  | 使用前缀和数组，记录每个前缀和第一次出现的位置，若再次出现相同的前缀和，则表示子数组和为零。                                        |
| 139<br><br>(m)       | [[Subarray Sum Closest]]最接近零的子数组和                  | 给定一个整数数组，找到和最接近零的子数组，返回其起始和结束下标。         | 输入: [-3, 1, 1, -3, 5] 输出: [0, 2]<br><br>O(n),O(n)                                  | 计算前缀和数组，对其排序，寻找相邻前缀和差值最小的两个元素，其对应的子数组和最接近零。                                           |
| 838<br><br>(e)<br>   | [[Subarray Sum Equals K]]子数组和为K的個數                 | 给定一个整数数组和一个整数 k，找到和等于 k 的连续子数组的个数。       | 输入: nums = [1,1,1], k = 2 <br>输出: 2<br>([0,1],[1,2])<br><br>O(n),O(n)              | 使用前缀和数组，利用哈希表记录前缀和出现的次数，遍历数组时检查当前前缀和减去 k 是否在哈希表中，若存在则表示找到一个符合条件的子数组。                  |
| 1712<br>*<br>(m)<br> | [[Binary Subarrays With Sum]]子数组和为S的二元子数组数量        | 给定一个二进制数组和一个整数 `S`，返回和为 `S` 的子数组数量。      | 输入: nums = [1,0,1,0,1], S = 2 输出: 4<br><br>O(n),O(n)                               | 使用前缀和数组，利用哈希表记录前缀和出现的次数，遍历数组时检查当前前缀和减去 `S` 是否在哈希表中，若存在则累加计数。                          |
| 1844<br>*<br>(m)     | [[Subarray Sum Equals to K II]] 子数组和为K的最短子数组       | 给定一个整数数组和一个整数k，你需要找到和为k的最短非空子数组，并返回它的长度。 | 输入:  <br>nums = [1, 1, 1, 2], k = 3  <br>输出: 2  ([1,2])<br><br>O(n),O(n)           | 使用前缀和与哈希表记录出现的和，通过快速查找加速匹配和为K的子数组数量。                                                  |
| 911<br>*<br>(m)      | [[Maximum Size Subarray Sum Equals k]]子数组和为K的最長子数组 | 给定一个整数数组和一个整数 k，找到和等于 k 的最长子数组，返回其长度。    | 输入: nums = [1, -1, 5, -2, 3], k = 3 输出: 4  ([1,-1,5,-2])<br><br>O(n),O(n)<br><br>  | 使用前缀和数组，利用哈希表记录前缀和第一次出现的位置，遍历数组时检查当前前缀和减去 k 是否在哈希表中，若存在则计算子数组长度并更新最大长度。               |
| 406<br>*<br>(m)      | [[Minimum Size Subarray Sum]] 子数组和大於K的最短子数组        | 找出数组中和大于或等于 S 的最小连续子数组长度。                | 输入:  <br>nums = [2,3,1,2,4,3], <br>s = 7  <br>输出: 2  ([4,3])                       | 使用滑动窗口动态调整窗口范围，记录最短长度。                                                                |
| 41<br>*<br>(e)       | [[Maximum Subarray]]子数组和最大的和                       | 找到数组中和最大的连续子数组。                          | 输入: [-2,2,-3,4,-1,2,1,-5,3] 输出: 6 ([4,-1,2,1,-5,3])                                | 使用前綴和，维护当前子数组的最大和，更新全局最大和。                                                            |
| 402<br><br>(m)<br>   | [[Continuous Subarray Sum]]子数组和最大的子数组              | 给定一个整数数组，请找出一个连续子数组，使得该子数组的和最大           | 输入: [ -3, 1, 3, -3, 4 ]<br>输出: [ 1,3,-3,4 ]<br><br>O(n),O(1)                       | 使用前缀和数组，並計算minsum 记录最小前缀和. 用雙指針紀錄子數組啟始結束索引                                            |
| 42<br><br>(m)<br>!!  | [[Maximum Two Subarrays]]和最大的两个不重叠子数组              | 找到数组中和最大的两个不重叠子数组。                       | 输入: [1, 3, -1, 2, -1, 2] 输出: 7<br><br>O(n),O(n)                                    | 使用动态规划，分别从左到右和从右到左计算最大子数组和，然后寻找两个不重叠部分的最大和。                                           |
| 1850<br>*<br>(m)     | 捡苹果 [[Pick Apples]]                                | 给定每棵树上的苹果数量和两个篮子容量，找到可以收集的最大苹果数量。        | 输入:  <br>apples = [1,2,3,2,1], basket1 = 2, basket2 = 2  <br>输出: 4                 | 使用滑动窗口记录当前窗口的苹果数量，动态调整窗口范围，找到最大值。                                                     |
| 1075<br><br>(m)<br>  | [[Subarray Product Less Than K]]乘积小于K的子数组          | 给定一个正整数数组和一个整数 k，计算乘积小于 k 的连续子数组的个数。     | 输入: nums = [10, 5, 2, 6], k = 100 输出: 8<br><br>O(n), O(1)                          | 使用滑动窗口方法，维护一个窗口使其乘积小于 k，计算以当前元素为结尾的符合条件的子数组个数。                                        |
| 994<br><br>(m)<br>   | [[Contiguous Array]] 0 和 1 数量相等的最长子数组              | 给一个二进制数组，找到 0 和 1 数量相等的子数组的最大长度          | 输入: [0,1,0] 输出: 2<br><br>O(n),O(n)<br>                                             | 将数组中的 0 转换为 -1，然后计算前缀和数组，利用哈希表记录前缀和第一次出现的位置，若再次出现相同的前缀和，则表示找到一个符合条件的子数组，计算其长度并更新最大长度。 |
| 943<br><br>(e)       | [[Range Sum Query - Immutable]]区间和查询 - 不可变的        | 给定一个不可变的整数数组，计算任意区间 `[i, j]` 的和。         | 输入: nums = [-2, 0, 3, -5, 2, -1], sumRange(0, 2) 输出: 1<br><br>O(n),O(n)            | 预处理前缀和数组，计算每个位置的前缀和，区间和为两个前缀和之差。                                                      |
| 665<br><br>(m)<br>!! | [[Range Sum Query 2D - Immutable]]平面范围求和 -不可变矩阵    | 给定一个不可变的二维矩阵，计算任意子矩形的和。                  | 输入: matrix = [[1,2,3],[4,5,6],[7,8,9]]，sumRegion(1,1,2,2) 输出: 28<br><br>O(mn),O(1) | 构建二维前缀和数组，计算子矩形和为多个前缀和之差。                                                             |
| 1068<br><br>(e)      | [[Find Pivot Index]]寻找数组的中心索引                      | 找到數組中一個索引，使得左側所有元素的和等於右側所有元素的和           | 输入：<br>nums = [1,7,3,6,5,6]<br>输出：<br>output = 3<br><br>O(n),O(1)                  | 使用前綴和計算數組的總和，再通過遍歷計算每個索引的左側和，判斷是否滿足條件                                                 |




---

## **五、選取三題詳細解釋**

### **1. LeetCode 303: Range Sum Query - Immutable**(lintcode 943)

**題目描述**：  
給定一個數組，實現一個類來預處理數組，快速查詢任意區間 $[i, j]$ 的和。
Example:
nums = [-2, 0, 3, -5, 2, -1]
SumRange(0, 2) = 1

**解法思路**：

- 構建前綴和數組 `prefix_sum`，使查詢區間和只需 O(1)時間。

**Python 代碼**：
```python
class NumArray:
    def __init__(self, nums):
        self.prefix_sum = [0] * (len(nums) + 1)
        for i in range(1, len(self.prefix_sum)):
            self.prefix_sum[i] = self.prefix_sum[i - 1] + nums[i - 1]

    def sumRange(self, left: int, right: int) -> int:
        return self.prefix_sum[right + 1] - self.prefix_sum[left]

```

**時間複雜度**：

- **初始化**：O(n)
- **查詢區間和**：O(1)

**為什麼使用前綴和數組**：

- 提高查詢區間和的效率，將每次查詢的時間降到 O(1)。

---

### **2. LeetCode 560: Subarray Sum Equals K** (lintcode 838)

**題目描述**：  
給定一個數組和目標值 k，找出和等於 k 的連續子數組個數。
Example: 
nums = [2,1,-1,1,2] and k=3  
output = 4

**解法思路**：

- 使用前綴和數組與哈希表來加速查找。
- 記錄前綴和出現的次數，檢查當前前綴和是否存在一個差值等於 kkk。

**Python 代碼**：
```python
def subarraySum(nums, k):
    count = 0
    prefix_sum = 0
    prefix_count = {0: 1}
    
    for num in nums:
        prefix_sum += num
        if prefix_sum - k in prefix_count:
            count += prefix_count[prefix_sum - k]
        prefix_count[prefix_sum] = prefix_count.get(prefix_sum, 0) + 1
        
    return count

```

**時間複雜度**：

- O(n)：只需一次遍歷。

**為什麼使用前綴和數組**：

- 快速查找前綴和之間的差值，並利用哈希表記錄中間結果。
- prefix_count 是一個哈希表，鍵（key）是某個「前綴和」，值（value）是該「前綴和」出現的次數。它用來記錄每個前綴和出現的次數，方便在後續查找中快速獲得結果。
- prefix_count.get(prefix_sum, 0) 表示從哈希表 prefix_count 中查詢鍵 prefix_sum 的值。如果該鍵存在，返回其對應的值（即該前綴和出現的次數）。如果該鍵不存在，返回默認值 0

---

### **3. LeetCode 724: Find Pivot Index** (lintcode 1068)

**題目描述**：  
找到數組中一個索引，使得左側所有元素的和等於右側所有元素的和。
Example:
nums = [1,7,3,6,5,6]
output = 3

**解法思路**：

- 使用前綴和計算數組的總和，再通過遍歷計算每個索引的左側和，判斷是否滿足條件。

**Python 代碼**：
```python
def pivotIndex(nums):
    total_sum = sum(nums)
    left_sum = 0
    
    for i, num in enumerate(nums):
        if left_sum == total_sum - left_sum - num:
            return i
        left_sum += num
    
    return -1

```

**時間複雜度**：

- O(n)：遍歷數組一次。

**為什麼使用前綴和數組**：

- 快速計算總和，並利用累加求解左側和，減少重複計算。

---

## **總結**

- **前綴和數組** 適合快速計算區間和及子數組和等問題。
- 它將查詢操作優化到 O(1)，但初始化代價為 O(n)。
- 在 LeetCode 上常用於靜態數組問題，如區間和、子數組和等問題，結合哈希表可以進一步解決和的查找問題。


# **<mark style="background: #FF5582A6;">链表（Linked List）</mark>的详细介绍**
###### 链表
---

## **一、链表的原理与特点**

### **1. 原理**

- **链表（Linked List）** 是一种**非连续存储**的线性数据结构，由**节点（Node）** 组成，每个节点包含两部分：
    - **数据域**：存储数据。
    - **指针域**：存储指向下一个节点的引用（指针）。
- 链表不像数组一样存储在连续的内存空间中，而是通过指针连接节点。

### **2. 链表的类型**

1. **单向链表**：每个节点只存储指向下一个节点的指针。
2. **双向链表**：每个节点存储指向前后两个节点的指针。
3. **循环链表**：尾节点的指针指向头节点，形成一个循环。

### **3. 特点**

- **动态内存分配**：链表的大小可以动态调整，无需提前分配固定空间。
- **插入/删除高效**：链表在插入或删除元素时，只需修改指针，时间复杂度为 O(1)。
- **访问效率较低**：无法通过索引直接访问元素，必须从头节点开始逐个遍历，时间复杂度为 O(n)。
- **灵活性**：适合需要频繁插入/删除的场景。

---

## **二、具体例子**

### **单向链表示例**

假设有一个链表 `1 -> 4 -> 6 -> 8 -> None`：

- 头节点（Head）：存储值 `1`，指向下一个节点。
- 中间节点：存储值 `4`，指向下一个节点。
- 尾节点：存储值 `8`，指向 `None`（链表结束）。

---

## **三、Python 实作**

### **单向链表**
```python
# 定义链表的节点类
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# 定义链表的操作类
class LinkedList:
    def __init__(self):
        self.head = None

    # 插入节点（在链表末尾）
    def append(self, val):
        new_node = ListNode(val)
        if not self.head:
            self.head = new_node
            return
        cur = self.head
        while cur.next:
            cur = cur.next
        cur.next = new_node

    # 打印链表
    def print_list(self):
        cur = self.head
        while cur:
            print(cur.val, end=" -> ")
            cur = cur.next
        print("None")

# 使用示例
ll = LinkedList()
ll.append(1)
ll.append(4)
ll.append(6)
ll.append(8)
ll.print_list()  # 输出: 1 -> 4 -> 6 -> 8 -> None

```

---

## **四、LeetCode 链表题目描述及解法**

以下是整理的 LintCode 中涉及链表的入门、简单到中等难度的 50 道题目，包括题目编号、题目名称（英文）、题目简述（中文）、样例及解法：

| 题目编号                 | 题目名称（英文）                                                  | 题目简述（中文）                                                       | 样例                                                                                                                                                                       | 解法                                                                 |
| -------------------- | --------------------------------------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------ |
| 35<br>*<br>(e)       | [[Reverse Linked List]]翻转链表                               | 反转一个链表。                                                        | 输入: 1->2->3->null 输出: 3->2->1->null                                                                                                                                      | 使用迭代或递归方法反转链表。迭代方法中，遍历链表，将当前节点的 next 指向前一个节点。递归方法中，递归反转子链表，然后调整指针。 |
| 36<br><br>(m)<br>    | [[Reverse Linked List II]]翻转链表II                          | 反转链表中第 m 个节点到第 n 个节点之间的部分。1 ≤ m ≤ n ≤ 链表长度。                    | 输入: 1->2->3->4->5->null, m = 2, n = 4 输出: 1->4->3->2->5->null                                                                                                            | 使用迭代方法，找到第 m-1 个节点，反转第 m 到第 n 个节点，然后重新连接链表。                        |
| 96<br><br>(e)<br>    | [[Partition List]]链表划分                                    | 给定一个链表和一个值 x，分隔链表，使得所有小于 x 的节点在大于或等于 x 的节点之前。保留两个部分内节点的初始相对顺序。 | 输入: 1->4->3->2->5->2->null, x = 3 输出: 1->2->2->4->3->5->null                                                                                                             | 使用两个指针分别处理小于 x 和大于等于 x 的节点，最后将两个链表连接。<br><br>                      |
| 112<br><br>(e)<br>   | [[Remove Duplicates from Sorted List]]删除排序链表中的重复元素        | 删除排序链表中的重复元素，使每个元素只出现一次。                                       | 输入: 1->1->2->3->3->null 输出: 1->2->3->null                                                                                                                                | 遍历链表，删除重复的节点。                                                      |
| 113<br><br>(m)       | [[Remove Duplicates from Sorted List II]]删除排序链表中的重复元素II   | 删除排序链表中的重复元素，重复的节点全部删除，只保留没有重复的数字。                             | 输入: 1->2->3->3->4->4->5->null 输出: 1->2->5->null                                                                                                                          | 使用双指针遍历链表，跳过重复的节点。                                                 |
| 165<br>*<br>(e)      | [[Merge Two Sorted Lists]]合并两个排序链表                        | 将两个排序链表合并为一个新的排序链表。                                            | 输入: 1->2->4->null, 1->3->4->null 输出: 1->1->2->3->4->4->null                                                                                                              | 使用迭代或递归方法合并两个链表。迭代方法中，使用两个指针遍历两个链表，按顺序连接节点。递归方法中，递归合并子链表。          |
| 166<br><br>(e)       | [[Nth to Last Node in List]]链表倒数第n个节点                     | 找到链表中倒数第 n 个节点。                                                | 输入: 1->2->3->4->5->null, n = 2 输出: 4                                                                                                                                     | 使用双指针，先让第一个指针移动 n 步，然后两个指针一起移动，直到第一个指针到达末尾，第二个指针即为倒数第 n 个节点。       |
| 167<br><br>(e)<br>   | [[Add Two Numbers]]链表求和                                   | 给定两个非空链表，表示两个非负整数。数字以逆序存储，每个节点包含一个数字。将两个数相加，并以相同形式返回一个链表。      | 输入: (2 -> 4 -> 3) + (5 -> 6 -> 4) 输出: 7 -> 0 -> 8                                                                                                                        | 使用指针遍历两个链表，逐位相加，处理进位，生成新的链表。                                       |
| 170<br><br>(m)<br>   | [[Rotate List]]旋转链表                                       | 给定一个链表，向右旋转链表，使每个节点向右移动 k 个位置，其中 k 是非负数。                       | 输入: 1->2->3->4->5->null, k = 2 输出: 4->5->1->2->3->null                                                                                                                   | 计算链表长度，找到新的头节点位置，重新连接链表。<br><br>                                   |
| 174<br>*<br>(e)<br>  | [[Remove Nth Node From End of List]]删除链表中倒数第 n 个节点        | 删除链表的倒数第 n 个节点，并返回链表的头节点。                                      | 输入: 1->2->3->4->5->null, n = 2 输出: 1->2->3->5->null                                                                                                                      | 使用双指针，先让第一个指针移动 n 步，然后两个指针一起移动，直到第一个指针到达末尾，删除第二个指针指向的节点。           |
| 452<br><br>(e)       | [[Remove Linked List Elements]]删除链表中的元素                   | 删除链表中等于给定值 val 的所有节点。                                          | 输入: 1->2->6->3->4->5->6->null, val = 6 输出: 1->2->3->4->5->null                                                                                                           | 遍历链表，删除值为 val 的节点。                                                 |
| 223<br><br>(m)<br>   | [[Palindrome Linked List]]回文链表                            | 判断链表是否为回文链表。                                                   | 输入: 1->2->2->1->null 输出: true                                                                                                                                            | 使用快慢指针找到链表中点，反转后半部分链表，然后比较前半部分和反转后的后半部分是否相同。                       |
| 372<br><br>(e)<br>   | [[Delete Node in a Linked List]]删除链表节点                    | 删除链表中的一个非末尾节点，给定该节点的指针。                                        | 输入: 1->2->3->4->5->null, 删除节点 3 输出: 1->2->4->5->null                                                                                                                     | 将要删除的节点的值替换为下一个节点的值，然后删除下一个节点。                                     |
| 1292<br><br>(m)      | [[Odd Even Linked List]]奇偶链表                              | 给定一个单链表，将所有奇数节点和偶数节点分组在一起，保持它们的相对顺序，并返回重新排列后的链表。               | 输入: 1->2->3->4->5->null 输出: 1->3->5->2->4->null                                                                                                                          | 使用两个指针分别指向奇数节点和偶数节点，将它们分离并重新连接。                                    |
| 904<br><br>(m)       | [[Plus One Linked List]]加一链表                              | 给定一个链表，表示一个非负整数，每个节点包含一位数字，返回加 1 后的链表。                         | 输入: 1->2->3->null 输出: 1->2->4->null                                                                                                                                      | 反转链表，从头开始加 1，处理进位问题，最后再反转链表。                                       |
| 221<br><br>(m)       | [[Add Two Numbers II]]链表求和 II                             | 给定两个链表，表示两个非负整数，数字按正常顺序存储，返回它们的和（以链表形式）。                       | 输入: (7 -> 2 -> 4 -> 3) + (5 -> 6 -> 4) 输出: 7 -> 8 -> 0 -> 7                                                                                                              | 反转链表，逐位相加处理进位，然后再反转链表得到结果。                                         |
| 1609<br>*<br>(e)<br> | [[Middle of the Linked List]]链表的中间结点                      | 找到链表的中间节点，若有两个中间节点，返回第二个中间节点。                                  | 输入: 1->2->3->4->5->null 输出: 3->4->5->null                                                                                                                                | 使用快慢指针，快指针移动两步，慢指针移动一步，当快指针到达末尾时，慢指针即为中间节点。                        |
| 105<br><br>(m)       | [[Copy List with Random Pointer]]复制带随机指针的链表               | 给定一个链表，其中每个节点包含一个额外的随机指针，指向链表中的任意节点或 null。返回其深拷贝。              | 输入: head = [[7,null],[13,0],[11,4],[10,2],[1,0]] 输出: 深拷贝链表                                                                                                               | 使用哈希表存储旧节点与新节点的映射，遍历链表创建新节点并建立连接。                                  |
| 106<br><br>(m)       | [[Convert Sorted List to Binary Search Tree]]有序链表转换为二叉搜索树 | 将一个升序排列的链表转换为高度平衡的二叉搜索树。                                       | 输入: head = [-10,-3,0,5,9] 输出: 树根节点为 0                                                                                                                                    | 使用快慢指针找到链表中间节点作为根节点，递归构建左右子树。                                      |
| 102<br>*<br>(m)<br>  | [[Linked List Cycle]]带环链表                                 | 判断链表中是否有环。                                                     | 输入: head = [3,2,0,-4], pos = 1 输出: true                                                                                                                                  | 使用快慢指针，若快指针和慢指针相遇，则链表中存在环。                                         |
| 103<br><br>(h)       | [[Linked List Cycle II]]带环链表 II                           | 找到链表环的起始节点，若无环则返回 null。                                        | 输入: head = [3,2,0,-4], pos = 1 输出: 返回指向节点 2 的指针                                                                                                                          | 使用快慢指针找到相遇点，然后一个指针从头开始，另一个指针从相遇点开始，每次移动一步，直到两个指针相遇即为环的起始节点。        |
| 99<br>*<br>(m)       | [[Reorder List]]重排链表                                      | 给定一个链表，将其重新排序为 L0 → Ln → L1 → Ln-1 → L2 → Ln-2 → ……            | 输入: 1->2->3->4->5->null 输出: 1->5->2->4->3->null                                                                                                                          | 使用快慢指针找到链表中点，反转后半部分链表，然后将两部分交替连接。                                  |
| 134<br>**<br>(h)     | LRU缓存策略 [[LRU Cache]]                                     | 实现一个数据结构支持LRU缓存机制，包含获取和放入两个操作，并保持最近使用的数据优先。                    | 输入:  <br>LRUCache cache = new LRUCache(2);  <br>cache.put(1, 1);  <br>cache.put(2, 2);  <br>cache.get(1);  <br>cache.put(3, 3);  <br>cache.get(2);  <br>输出:  <br>[1, -1] | 使用哈希表和双向链表，哈希表用于快速定位，链表维护使用顺序，将最新访问的数据移到链表头部。                      |
| 104<br>**<br>(m)     | 合并k个排序链表 [[Merge K Sorted Lists]]                         | 合并k个排序链表，返回一个升序合并后的链表。                                         | 输入:  <br>lists = [[1,4,5],[1,3,4],[2,6]]  <br>输出: [1,1,2,3,4,4,5,6]                                                                                                      | 使用优先队列（最小堆）动态获取每个链表当前最小值，将其连接到结果链表中，重复直到所有链表为空。                    |
| 453<br><br>(e)<br>   | [[Flatten Binary Tree to Linked List]]将二叉树拆成链表            | 将二叉树展开为单链表，按先序遍历顺序排列。                                          | 输入:  <br>root = [1,2,5,3,4,null,6]  <br>输出:  <br>[1,null,2,null,3,null,4,null,5,null,6]                                                                                  | 使用递归或栈模拟先序遍历，将左右子树依次连接到当前节点后面。                                     |
| 116<br>*<br>(m)      | [[Jump Game]]跳跃游戏                                         | 给定一个非负整数数组，判断是否能跳到最后一个位置。                                      | 输入:  <br>nums = [2,3,1,1,4]  <br>输出:  <br>true                                                                                                                           | 从后向前遍历，记录最远可跳位置，若当前位置能到达最远可跳位置，则继续检查下一个位置。                         |

## **五、选取三题详细解释**

### **1. LeetCode 206: 反转链表（Reverse Linked List）**(lintcode 35)

**题目描述**：  
给定一个单向链表，反转链表并返回新的头节点。
Example:
1->2->3->null
output: 3->2->1->null

**解法思路**：

- 使用 **双指针法** 迭代遍历链表，将每个节点的指针反转。
- 通过一个指针记录当前节点的前一个节点。

**Python 代码**：
```python
def reverseList(head):
    prev = None
    cur = head
    while cur:
        next_node = cur.next  # 保存下一个节点
        cur.next = prev       # 反转当前节点的指针
        prev = cur            # 前移 prev
        cur = next_node       # 前移 cur
    return prev  # 新的头节点

```

**时间复杂度**：

- **O(n)O(n)O(n)**：需要遍历整个链表。

**为什么使用链表**：

- 反转链表是一种常见的链表操作，利用链表的**指针操作**特点，轻松完成反转。

---

### **2. LeetCode 21: 合并两个有序链表（Merge Two Sorted Lists）**(lintcode 165)

**题目描述**：  
合并两个有序链表，返回合并后的链表，保证链表有序。
Example: 
list1 =  1->3->8->11->15->null, list2 = 2
output: 1->2->3->8->11->15->null

**解法思路**：

- 使用 **递归** 或 **迭代** 比较两个链表当前节点的值，将较小值节点连接到结果链表。

**Python 代码（迭代解法）**：
```python
def mergeTwoLists(list1, list2):
    dummy = ListNode(0)  # 虚拟头节点
    cur = dummy

    while list1 and list2:
        if list1.val < list2.val:
            cur.next = list1
            list1 = list1.next
        else:
            cur.next = list2
            list2 = list2.next
        cur = cur.next

    cur.next = list1 if list1 else list2
    return dummy.next

```

**时间复杂度**：

- **O(n+m)O(n + m)O(n+m)**：n 和 m 分别是两个链表的长度。

**为什么使用链表**：

- 链表适合动态操作，节点之间的连接可以轻松重组，无需额外空间。

---

### **3. LeetCode 141: 判断链表是否有环（Linked List Cycle）**(lintcode 102)

**题目描述**：  
给定一个链表，判断链表中是否存在环。
Example:
linked list = 21->10->4->5
output: False


**解法思路**：

- 使用 **快慢指针法**（Floyd 循环检测算法）：
    - 快指针每次移动两步，慢指针每次移动一步。
    - 若链表中存在环，则快慢指针必定相遇。

**Python 代码**：
```python
def hasCycle(head):
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False

```

**时间复杂度**：

- **O(n)O(n)O(n)**：快慢指针最多遍历链表一次。

**为什么使用链表**：

- 链表中的节点可以通过指针形成环，利用快慢指针的特性高效检测环。

---

## **总结**

- **链表（Linked List）** 适合频繁插入和删除操作的场景，避免数组的元素移动开销。
- 链表中的**指针操作**使得问题如反转链表、合并链表、环检测等变得自然高效。
- 通过 LeetCode 的这些经典题目，可以深入理解链表的特点与常用技巧。


# **<mark style="background: #ADCCFFA6;">雙向鏈表（Doubly Linked List）</mark>的詳細介紹**

###### 雙向链表
---

## **一、雙向鏈表的原理與特點**

### **1. 原理**

- **雙向鏈表（Doubly Linked List）** 是一種非連續存儲的線性數據結構，每個節點包含三個部分：
    - **數據域（Data）**：存儲節點中的數據。
    - **前向指針（prev）**：指向前一個節點。
    - **後向指針（next）**：指向下一個節點。
- 與單向鏈表相比，雙向鏈表的節點可以**雙向遍歷**，即可以從頭節點向後，也可以從尾節點向前遍歷。

### **2. 特點**

- **雙向訪問**：支援前向和後向兩個方向的遍歷。
- **靈活的插入/刪除**：插入和刪除節點時，不需要像數組那樣移動元素，只需修改指針。
- **額外空間**：由於每個節點需要額外的指針來指向前一個節點，存儲空間開銷較大。
- **時間複雜度**：
    - 查找節點：O(n)O(n)O(n)
    - 插入/刪除節點：O(1)O(1)O(1)

---

## **二、具體例子**

假設有一個雙向鏈表：  
`1 <-> 4 <-> 6 <-> 8`

- **頭節點（Head）**：存儲值 `1`，`prev = None`，`next` 指向 `4`。
- **中間節點**：存儲值 `4`，`prev` 指向 `1`，`next` 指向 `6`。
- **尾節點（Tail）**：存儲值 `8`，`next = None`，`prev` 指向 `6`。

---

## **三、Python 實作**

### **雙向鏈表實現**
```python
# 定義雙向鏈表的節點類
class DoublyListNode:
    def __init__(self, val=0):
        self.val = val
        self.prev = None
        self.next = None

# 定義雙向鏈表的操作類
class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    # 插入節點（在尾部）
    def append(self, val):
        new_node = DoublyListNode(val)
        if not self.head:  # 若鏈表為空
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            new_node.prev = self.tail
            self.tail = new_node

    # 遍歷鏈表（正向）
    def print_forward(self):
        cur = self.head
        while cur:
            print(cur.val, end=" <-> ")
            cur = cur.next
        print("None")

    # 遍歷鏈表（反向）
    def print_backward(self):
        cur = self.tail
        while cur:
            print(cur.val, end=" <-> ")
            cur = cur.prev
        print("None")

# 使用示例
dll = DoublyLinkedList()
dll.append(1)
dll.append(4)
dll.append(6)
dll.append(8)
dll.print_forward()   # 輸出: 1 <-> 4 <-> 6 <-> 8 <-> None
dll.print_backward()  # 輸出: 8 <-> 6 <-> 4 <-> 1 <-> None

```



## **五、選取三題詳細解釋**

---

### **1. LeetCode 146: LRU Cache** (lintcode 134)

**題目描述**：  
設計一個數據結構來實現 LRU（最近最少使用）緩存，要求支持以下操作：

- `get(key)`：獲取指定鍵的值，如果不存在返回 `-1`。
- `put(key, value)`：插入或更新鍵值對，如果緩存滿了，刪除最近最少使用的鍵值對。

Example:
LRUCache(2)
set(2, 1)
set(1, 1)
get(2)
set(4, 1)
get(1)
get(2)
output: [1,-1,1]
解释：
cache上限为2，set(2,1)，set(1, 1)，get(2) 然后返回 1，set(4,1) 然后 delete (1,1)，因为 （1,1）最少使用，get(1) 然后返回 -1，get(2) 然后返回 1。

**解法思路**：

- 使用 **雙向鏈表** 存儲鍵值對，維護最近使用的順序。
- 使用 **哈希表** 來快速查找節點位置。
- 插入/刪除操作在雙向鏈表中時間複雜度為 O(1)O(1)O(1)。

**Python 代碼**：
```python
class ListNode:
    def __init__(self, key=0, val=0):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # 哈希表存儲鍵值對
        self.head = ListNode()
        self.tail = ListNode()
        self.head.next = self.tail
        self.tail.prev = self.head

    # 將節點移到頭部
    def _move_to_head(self, node):
        self._remove(node)
        self._add(node)

    # 添加節點到頭部
    def _add(self, node):
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    # 刪除節點
    def _remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    # 獲取鍵的值
    def get(self, key: int) -> int:
        if key in self.cache:
            node = self.cache[key]
            self._move_to_head(node)
            return node.val
        return -1

    # 插入或更新鍵值對
    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            node = self.cache[key]
            node.val = value
            self._move_to_head(node)
        else:
            node = ListNode(key, value)
            self.cache[key] = node
            self._add(node)
            if len(self.cache) > self.capacity:
                tail = self.tail.prev
                self._remove(tail)
                del self.cache[tail.key]

```

**時間複雜度**：

- **get 操作**：O(1)O(1)O(1)。
- **put 操作**：O(1)O(1)O(1)。

**為什麼使用雙向鏈表**：

- 雙向鏈表支援快速的插入和刪除節點，可以高效地維護最近使用的順序。

---

### **2. LeetCode 430: Flatten a Multilevel Doubly Linked List**

**題目描述**：  
將一個多層雙向鏈表展平成單層雙向鏈表。

**解法思路**：

- 使用遞歸或迭代法展開多層鏈表，維護前後指針關係。

**Python 代碼**：
```python
def flatten(head):
    if not head:
        return head

    stack = []
    cur = head

    while cur:
        if cur.child:
            if cur.next:
                stack.append(cur.next)
            cur.next = cur.child
            cur.child.prev = cur
            cur.child = None
        if not cur.next and stack:
            temp = stack.pop()
            cur.next = temp
            temp.prev = cur
        cur = cur.next

    return head

```

**時間複雜度**：

- O(n)O(n)O(n)：遍歷所有節點。

**為什麼使用雙向鏈表**：

- 雙向鏈表的靈活性支援多層結構的展開。

---

### **3. LeetCode 707: Design Linked List**

**題目描述**：  
設計鏈表，支援在頭部、尾部插入節點，並支援刪除節點等操作。

**解法思路**：

- 使用雙向鏈表實現鏈表的操作。

**Python 代碼**：
```python
class MyLinkedList:
    def __init__(self):
        self.head = ListNode(0)
        self.tail = ListNode(0)
        self.head.next = self.tail
        self.tail.prev = self.head

```

---

## **總結**

- **雙向鏈表** 適合需要**雙向遍歷**和高效插入/刪除的場景。
- 在設計緩存、合併多層結構等應用中，雙向鏈表能有效解決問題並優化操作效率。


# **<mark style="background: #FF5582A6;">队列（Queue）</mark>的详细介绍**

 ###### 隊列
---

## **一、队列的原理与特点**

### **1. 原理**

- **队列（Queue）** 是一种**先进先出（FIFO，First In First Out）** 的线性数据结构。
- 元素的**插入（enqueue）** 只能在队尾进行，而**删除（dequeue）** 只能在队头进行。
- 生活中常见的例子：排队买票，先到先服务。

### **2. 特点**

- **先进先出（FIFO）**：第一个进入队列的元素最先被处理。
- **操作受限**：只能在两端操作：
    - **入队**（enqueue）：在队尾插入元素。
    - **出队**（dequeue）：从队头移除元素。
- **时间复杂度**：
    - 入队和出队操作的时间复杂度均为 O(1)。
    - 遍历队列的时间复杂度为 O(n)。

### **3. 常见类型**

- **普通队列**：最基础的先进先出队列。
- **双端队列（Deque）**：允许在队头和队尾进行插入和删除操作。
- **优先队列（Priority Queue）**：元素按优先级出队，而非按插入顺序。
- **寬度優先搜索 [BFS ](https://www.notion.so/6-BFS-32-32-a3f57279762b4b26b72e6c2d04d3bbc8)**

---

## **二、具体例子**

假设有一个队列 `Queue = [1, 4, 6, 8]`：

- **入队操作**（插入元素 `10` 到队尾）：
    - 新队列：`[1, 4, 6, 8, 10]`。
- **出队操作**（移除队头的元素）：
    - 新队列：`[4, 6, 8, 10]`。

---

## **三、Python 实作**

### **普通队列的实现**
```python
from collections import deque

# 使用 deque 实现队列
class Queue:
    def __init__(self):
        self.queue = deque()

    # 入队操作
    def enqueue(self, val):
        self.queue.append(val)

    # 出队操作
    def dequeue(self):
        if not self.is_empty():
            return self.queue.popleft()
        else:
            print("队列为空")
            return None

    # 获取队头元素
    def front(self):
        if not self.is_empty():
            return self.queue[0]
        else:
            print("队列为空")
            return None

    # 判断队列是否为空
    def is_empty(self):
        return len(self.queue) == 0

    # 打印队列
    def print_queue(self):
        print("队列内容:", list(self.queue))

# 测试队列
q = Queue()
q.enqueue(1)
q.enqueue(4)
q.enqueue(6)
q.enqueue(8)
q.print_queue()  # 输出: [1, 4, 6, 8]

q.dequeue()
q.print_queue()  # 输出: [4, 6, 8]
print("队头元素:", q.front())  # 输出: 4

```
deque指令:  append(), appendleft(), pop(), popleft()

---

## **四、LeetCode 队列题目描述及解法**

以下是整理的 LintCode 中涉及队列的入门级别、难度为 Easy 到 Medium 的题目列表，包括题目编号、题目名称（英文）、题目简述（中文）、样例及解法：

| 题目编号                | 题目名称（英文）                                               | 题目简述（中文）                                             | 样例                                                                                                                        | 解法                                                                                                              |
| ------------------- | ------------------------------------------------------ | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| 494<br>*<br>(e)<br> | [[Implement Stack using Queues]]双队列实现栈                 | 使用队列实现栈，支持 `push(x)`、`pop()`、`top()` 和 `empty()` 操作。 | 输入：<br>push(1)<br>pop()<br>push(2)<br>isEmpty() // return false<br>top() // return 2<br>pop()<br>isEmpty() // return true | 使用两个队列实现栈功能：一个队列用于存储元素，另一个队列用于辅助反转元素顺序。每次插入元素时，将新元素添加到辅助队列，然后将主队列的所有元素依次移动到辅助队列，交换主辅队列。                         |
| 40<br><br>(m)       | [[Implement Queue by Two Stacks]]用栈实现队列                | 使用两个栈实现队列，支持 `push(element)`、`pop()` 和 `top()` 操作。   | 队列操作 = <br>    push(1)<br>    pop()    <br>    push(2)<br>    push(3)<br>    top()    <br>    pop()                       | 使用两个栈实现队列功能：一个栈用于入队操作，另一个栈用于出队操作。当出队栈为空时，将入队栈的所有元素依次弹出并压入出队栈，然后从出队栈弹出元素以实现队列的先进先出特性。                            |
| 999<br><br>(m)<br>  | [[Design Circular Deque]]用循环数组实现双向队列                   | 设计实现双端循环队列，支持插入、删除、获取队首和队尾元素、检查队列是否为空或已满等操作。         |                                                                                                                           | 使用固定大小的数组实现循环双端队列，维护头尾指针和当前元素数量，实现各项操作。                                                                         |
| 642<br><br>(e)      | [[Moving Average from Data Stream]]数据流滑动窗口平均值          | 给定一个整数流和窗口大小，计算滑动窗口的平均值。                             |                                                                                                                           | 使用一个队列存储滑动窗口中的元素，并维护窗口元素总和。每次插入新元素时，若队列长度超过窗口大小，则移除队列头部元素，并更新总和，然后计算平均值。                                        |
| 362<br>**<br>(h)    | [[Sliding Window Maximum]]滑动窗口的最大值                     | 给定一个整数数组和滑动窗口大小，找出每个窗口中的最大值。                         | 输入: nums = <br>[1,3,-1,-3,5,3,6,7], <br>k = 3 <br>输出: [3,3,5,5,6,7]                                                       | 使用双端队列（deque）维护当前窗口的最大值索引。遍历数组时，移除队列中不在当前窗口范围内的元素，并移除队列中小于当前元素的所有元素，然后将当前元素索引添加到队列。当前元素索引大于等于窗口大小时，队首即为当前窗口最大值。 |
| 360<br><br>(h)<br>  | [[Sliding Window Median]]滑动窗口的中位数                      | 给定一个整数数组和滑动窗口大小，计算滑动窗口中位数。                           | 输入:<br>[1,2,7,8,5]<br>3<br>输出:<br>[2,7,7]                                                                                 | 使用两个优先队列（最大堆和最小堆）来维护当前窗口的元素，以便快速获取中位数。每次滑动窗口时，移除离开窗口的元素，添加进入窗口的元素，调整两个堆的平衡，然后根据窗口大小确定中位数。                       |
| 1031<br><br>(m)<br> | [[Is Graph Bipartite]]?图可以被二分么                         | 判断给定的无向图是否是二分图。                                      | 输入: `graph = [[1,3],[0,2],[1,3],[0,2]]` 输出: `true`                                                                        | 使用队列进行广度优先搜索（BFS），尝试将图中的节点染色为两种颜色。若某个节点的相邻节点已经被染成相同颜色，则图不是二分图；否则继续染色，直到遍历所有节点。                                  |
| 71<br><br>(m)<br>   | [[Binary Tree Zigzag Level Order]]Traversal二叉树的锯齿形层次遍历 | 给定一个二叉树，返回其节点值的锯齿形层次遍历（即第一层从左到右，第二层从右到左，依此类推）。       | 输入: <br>{3,9,20,#,#,15,7} <br>输出: [<br>[3],<br>[20, 9], <br>[15, 7]]                                                      | 使用双端队列进行广度优先搜索（BFS），根据当前层的遍历方向决定节点值的添加顺序，实现锯齿形遍历。                                                               |





## **五、选取三题详细解释**

---

### **1. LeetCode 225: 用队列实现栈（Implement Stack using Queues）**

**题目描述**：  
使用两个队列实现一个栈的功能，包括 `push`、`pop`、`top` 和 `empty` 操作。

**解法思路**：

- 使用 **两个队列**：`q1` 负责存储元素，`q2` 辅助操作。
- 栈的特点是 **后进先出（LIFO）**，而队列是先进先出。
- 每次插入新元素时，将旧元素逐个出队并入队，保证新元素位于队首。

**Python 代码**：
```python
from collections import deque

class MyStack:
    def __init__(self):
        self.queue = deque()

    # 入栈操作
    def push(self, x: int) -> None:
        self.queue.append(x)
        # 将前面的元素重新加入到队尾，保证新元素在队首
        for _ in range(len(self.queue) - 1):
            self.queue.append(self.queue.popleft())

    # 出栈操作
    def pop(self) -> int:
        return self.queue.popleft()

    # 获取栈顶元素
    def top(self) -> int:
        return self.queue[0]

    # 判断栈是否为空
    def empty(self) -> bool:
        return not self.queue

```

**时间复杂度**：

- **Push 操作**：O(n)O(n)O(n)。
- **Pop 和 Top 操作**：O(1)O(1)O(1)。

**为什么使用队列**：

- 使用队列模拟栈的后进先出特性，通过元素重新排列实现栈的功能。

---

### **2. LeetCode 102: 二叉树的层序遍历（[[Leetcode - 5.Binary Search Tree/Binary Tree Level Order Traversal]]）**

**题目描述**：  
给定一个二叉树，返回按层遍历的节点值（即每一层从左到右的节点值）。

**解法思路**：

- 使用 **队列** 存储当前层的节点。
- 每次从队列中取出一个节点，访问其值，并将其子节点加入队列。

**Python 代码**：
```python
from collections import deque

def levelOrder(root):
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)

    return result

```

**时间复杂度**：

- O(n)O(n)O(n)：遍历所有节点。

**为什么使用队列**：

- 队列的先进先出特性非常适合按层遍历二叉树。

---

### **3. LeetCode 933: 最近的请求次数（Number of Recent Calls）**

**题目描述**：  
设计一个类，统计最近 3000 毫秒内的请求次数。

**解法思路**：

- 使用 **队列** 存储每次请求的时间戳。
- 每次请求时，将超过 3000 毫秒的时间戳从队头移除，队列中剩下的即为有效请求。

**Python 代码**：
```python
from collections import deque

class RecentCounter:
    def __init__(self):
        self.queue = deque()

    def ping(self, t: int) -> int:
        self.queue.append(t)
        while self.queue[0] < t - 3000:
            self.queue.popleft()
        return len(self.queue)

```

**时间复杂度**：

- O(1)O(1)O(1)（均摊）：每个请求时间戳最多进出队列一次。

**为什么使用队列**：

- 队列可以有效地维护请求的时间窗口，且出队和入队操作的时间复杂度为 O(1)O(1)O(1)。

---

## **总结**

- **队列（Queue）** 适合处理**先进先出**场景，常用于数据流管理、广度优先搜索（BFS）和时间窗口问题。
- 在 LeetCode 的相关题目中，队列通过其特性高效解决了数据流维护和层级遍历等问题。


# **<mark style="background: #BBFABBA6;">双向队列（Deque）</mark>的详细介绍**

###### 雙向隊列
---

## **一、双向队列的原理与特点**

### **1. 原理**

- **双向队列（Deque, Double-Ended Queue）** 是一种**可以在两端进行插入和删除操作**的数据结构。
- 与普通队列（Queue）的**先进先出（FIFO）** 不同，双向队列提供了更灵活的操作：
    - **在队头插入/删除元素**。
    - **在队尾插入/删除元素**。

### **2. 特点**

- **两端操作灵活**：支持在队头和队尾进行元素的插入和删除。
- **动态调整大小**：可以动态扩展以适应数据量。
- **时间复杂度**：
    - 队头/队尾插入和删除：O(1)O(1)O(1)。
    - 访问队列中的元素：O(n)O(n)O(n)。

### **3. 常见用途**

- **滑动窗口问题**：维护当前窗口的最大/最小值。
- **广度优先搜索（BFS）**：双向队列可以优化节点的遍历操作。
- **回文字符串检测**：在两端检查字符是否相等。

---

## **二、具体例子**

假设有一个双向队列 `Deque = [1, 4, 6, 8]`：

1. **在队头插入元素** `10`：
    - 新队列：`[10, 1, 4, 6, 8]`。
2. **在队尾插入元素** `12`：
    - 新队列：`[10, 1, 4, 6, 8, 12]`。
3. **从队头删除元素**：
    - 删除 `10`，新队列：`[1, 4, 6, 8, 12]`。
4. **从队尾删除元素**：
    - 删除 `12`，新队列：`[1, 4, 6, 8]`。

---

## **三、Python 实作**

### **使用 `collections.deque` 实现双向队列**

Python 提供了 `collections` 模块中的 `deque` 类，它是一个高效的双向队列实现。
```python
from collections import deque

# 创建双向队列
dq = deque([1, 4, 6, 8])

# 在队头插入元素
dq.appendleft(10)  
print("队头插入 10:", dq)  # 输出: deque([10, 1, 4, 6, 8])

# 在队尾插入元素
dq.append(12)  
print("队尾插入 12:", dq)  # 输出: deque([10, 1, 4, 6, 8, 12])

# 从队头删除元素
dq.popleft()
print("队头删除:", dq)  # 输出: deque([1, 4, 6, 8, 12])

# 从队尾删除元素
dq.pop()
print("队尾删除:", dq)  # 输出: deque([1, 4, 6, 8])

```


## **五、选取三题详细解释**

---

### **1. LeetCode 239: 滑动窗口最大值（Sliding Window Maximum）**

**题目描述**：  
给定一个数组 `nums` 和一个整数 `k`，每次滑动一个窗口大小为 `k` 的子数组，返回每个窗口中的最大值。

**解法思路**：

- 使用 **双向队列** 维护当前窗口的索引。
- 确保队列中：
    - 元素单调递减（索引对应的值从大到小）。
    - 队头始终为当前窗口的最大值。
- 每次滑动窗口时：
    - 移除不在窗口内的元素。
    - 将新元素的索引插入队列，保持单调递减。

**Python 代码**：
```python
from collections import deque

def maxSlidingWindow(nums, k):
    result = []
    dq = deque()

    for i in range(len(nums)):
        # 移除窗口外的索引
        if dq and dq[0] < i - k + 1:
            dq.popleft()

        # 保持单调递减，移除小于当前值的索引
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()

        # 加入当前索引
        dq.append(i)

        # 当前窗口的最大值
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result

```

**时间复杂度**：

- **O(n)O(n)O(n)**：每个元素最多入队和出队一次。

**为什么使用双向队列**：

- 双向队列高效维护窗口中元素的顺序，确保最大值快速查找。

---

### **2. LeetCode 641: 设计循环双端队列（Design Circular Deque）**

**题目描述**：  
设计一个支持以下操作的循环双端队列：

- 插入/删除元素到队头或队尾。
- 检查队列是否为空或已满。

**解法思路**：

- 使用 `deque` 实现队头和队尾的插入与删除操作。

**Python 代码**：
```python
from collections import deque

class MyCircularDeque:
    def __init__(self, k: int):
        self.k = k
        self.dq = deque()

    def insertFront(self, value: int) -> bool:
        if len(self.dq) < self.k:
            self.dq.appendleft(value)
            return True
        return False

    def insertLast(self, value: int) -> bool:
        if len(self.dq) < self.k:
            self.dq.append(value)
            return True
        return False

    def deleteFront(self) -> bool:
        if self.dq:
            self.dq.popleft()
            return True
        return False

    def deleteLast(self) -> bool:
        if self.dq:
            self.dq.pop()
            return True
        return False

    def isEmpty(self) -> bool:
        return len(self.dq) == 0

    def isFull(self) -> bool:
        return len(self.dq) == self.k

```

**时间复杂度**：

- 所有操作的时间复杂度为 O(1)O(1)O(1)。

**为什么使用双向队列**：

- 支持队头和队尾的高效操作，满足循环队列的需求。

---

### **3. LeetCode 225: 使用队列实现栈（Implement Stack using Queues）**

**题目描述**：  
使用队列实现一个栈，包含 `push`、`pop` 和 `top` 操作。

**解法思路**：

- 使用 `deque` 反转元素顺序，使队尾元素成为栈顶。

**Python 代码**：
```python
from collections import deque

class MyStack:
    def __init__(self):
        self.queue = deque()

    def push(self, x: int) -> None:
        self.queue.append(x)
        for _ in range(len(self.queue) - 1):
            self.queue.append(self.queue.popleft())

    def pop(self) -> int:
        return self.queue.popleft()

    def top(self) -> int:
        return self.queue[0]

    def empty(self) -> bool:
        return not self.queue

```

**时间复杂度**：

- **Push 操作**：O(n)O(n)O(n)。
- **Pop 和 Top 操作**：O(1)O(1)O(1)。

**为什么使用双向队列**：

- 利用队列的双端操作模拟栈的后进先出特性。

---

## **总结**

- **双向队列（Deque）** 提供队头和队尾的高效插入与删除操作。
- 在解决滑动窗口、任务调度和循环队列问题中，双向队列表现出色。
- 在 LeetCode 的题目中，双向队列常用于**滑动窗口**和**双端数据处理**的场景。


# **<mark style="background: #ADCCFFA6;">单调队列（Monotone Queue）</mark>的详细介绍**

###### 單調隊列
---

## **一、单调队列的原理与特点**

### **1. 原理**

- **单调队列（Monotone Queue）** 是一种特殊的队列，它维护队列中元素的单调性（递增或递减）。
- **单调递减队列**：队列中的元素从前到后严格递减。
- **单调递增队列**：队列中的元素从前到后严格递增。
- 单调队列通常用于**滑动窗口问题**，可以在窗口滑动时高效维护最大值或最小值。

---

### **2. 特点**

- **双向队列实现**：单调队列通常使用**双向队列（deque）** 实现。
- **维护单调性**：
    - 插入新元素时，移除不满足单调性的队尾元素，保证队列的单调性。
- **时间复杂度**：
    - 每个元素最多入队和出队一次，时间复杂度为 O(n)。
    - 维护队列单调性的过程时间复杂度为 O(1)。

---

### **3. 适用场景**

- **滑动窗口最大值/最小值**：在一个窗口范围内找到最大值或最小值。
- **动态维护最优解**：在遍历过程中，动态维护当前区间的最优解。

---

## **二、具体例子**

假设有数组 `nums = [1, 3, -1, -3, 5, 3, 6, 7]`，窗口大小 k=3。

- 使用**单调递减队列**维护当前窗口内的最大值。
- 滑动窗口从左到右移动：
    - 初始：`[1, 3, -1]`，最大值为 `3`。
    - 滑动：`[3, -1, -3]`，最大值为 `3`。
    - 滑动：`[-1, -3, 5]`，最大值为 `5`。
    - 依此类推。

---

## **三、Python 实作**

### **单调递减队列实现滑动窗口最大值**
```python
from collections import deque

def maxSlidingWindow(nums, k):
    if not nums or k == 0:
        return []

    result = []
    dq = deque()  # 存储元素索引，保持单调递减

    for i in range(len(nums)):
        # 移除窗口外的元素
        while dq and dq[0] < i - k + 1:
            dq.popleft()

        # 维护单调性，移除小于当前元素的队尾元素
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()

        # 将当前索引加入队列
        dq.append(i)

        # 当窗口满足大小时，记录窗口的最大值
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result

# 示例
nums = [1, 3, -1, -3, 5, 3, 6, 7]
k = 3
print(maxSlidingWindow(nums, k))  # 输出: [3, 3, 5, 5, 6, 7]

```

---

## **四、LeetCode 单调队列题目描述及解法**

以下是整理的 LintCode 中涉及单调队列的题目表格，包括题目编号、题目名称（英文）、题目简述（中文）、样例及解法：

| 题目编号                 | 题目名称（英文）                                                | 题目简述（中文）                                              | 样例                                                                                           | 解法                                   |
| -------------------- | ------------------------------------------------------- | ----------------------------------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------ |
| 653<br><br>(h)       | [[Expression Add Operators]]添加运算符                       | 给定一个仅包含数字的字符串和一个目标值，为该字符串插入操作符，使得计算结果等于目标值，返回所有可能的结果。 | 输入:  <br>num = "123"  <br>target = 6  <br>输出:  <br>["1+2+3", "1_2_3"]                        | 使用回溯法或单调队列，逐步插入操作符计算结果。              |
| 603<br><br>(m)       | [[Largest Divisible Subset]]最大整除子集                      | 找到给定数组中最长的可整除子集。                                      | 输入:  <br>nums = [1,2,3]  <br>输出:  <br>[1,2]                                                  | 使用动态规划和单调队列记录每个子集的最大长度及路径。           |
| 903<br><br>(m)       | [[Range Addition]]范围加法                                  | 给定一个长度为 n 的数组和一组更新操作，返回更新后的数组。                        | 输入:  <br>length = 5  <br>updates = [ [1,3,2],[2,4,3],[0,2,-2] ]  <br>输出:  <br>[ -2,0,3,5,3 ] | 使用差分数组记录区间增量，最后计算前缀和得到结果。            |
| 1276<br>*<br>(e)<br> | [[Sum of Two Integers]]两整数之和                            | 使用位运算计算两个整数的和，而不能使用 `+` 和 `-` 操作符。                    | 输入:  <br>a = 1, b = 2  <br>输出:  <br>3                                                        | 使用位运算模拟加法过程，按位计算进位和当前位值。             |
| 1275<br><br>(m)      | [[Super Pow]]超级幂次                                       | 计算 `a^b % 1337`，其中 `b` 是一个非常大的整数以数组形式给出。              | 输入:  <br>a = 2, b = [3]  <br>输出:  <br>8                                                      | 使用快速幂和单调队列，动态处理大整数。                  |
| 1507<br>*<br>(h)     | 和至少为 K 的最短子数组 [[Shortest Subarray with Sum at Least K]] | 给定一个整数数组，找到和至少为K的最短子数组长度，若不存在返回-1。                    | 输入:  <br>nums = [2, -1, 2], k = 3  <br>输出: 3                                                 | 使用单调队列记录前缀和，动态检查当前子数组和是否满足条件并更新最短长度。 |


## **五、选取三题详细解释**

---

### **1. LeetCode 239: 滑动窗口最大值（Sliding Window Maximum）**

**题目描述**：  
给定一个数组 `nums` 和一个整数 kkk，找到所有大小为 kkk 的滑动窗口的最大值。

**解法思路**：

- 使用 **单调递减队列** 来维护当前窗口的最大值。
- 每次滑动时：
    - 移除窗口外的元素（队头）。
    - 移除队列中小于当前元素的队尾元素，保证单调递减。
    - 将当前元素的索引加入队列。
    - 当窗口形成时，队头元素即为当前窗口的最大值。

**Python 代码**：
```python
from collections import deque

def maxSlidingWindow(nums, k):
    result = []
    dq = deque()  # 单调递减队列

    for i in range(len(nums)):
        # 移除不在窗口范围内的元素
        while dq and dq[0] < i - k + 1:
            dq.popleft()

        # 移除队尾小于当前元素的索引
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()

        # 加入当前元素索引
        dq.append(i)

        # 记录窗口最大值
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result

```

**时间复杂度**：

- **O(n)O(n)O(n)**：每个元素最多入队和出队一次。

**为什么使用单调队列**：

- 单调队列可以高效维护窗口内的最大值，避免重复比较。

---

### **2. LeetCode 862: 和至少为 K 的最短子数组（Shortest Subarray with Sum at Least K）**

**题目描述**：  
给定一个整数数组 `nums` 和一个整数 KKK，找到一个和大于等于 KKK 的最短连续子数组的长度。

**解法思路**：

- 使用 **前缀和** 和 **单调队列**。
- 通过前缀和计算累积和，单调队列维护递增的前缀和索引。

**Python 代码**：
```python
from collections import deque

def shortestSubarray(nums, K):
    n = len(nums)
    prefix_sum = [0] * (n + 1)

    for i in range(n):
        prefix_sum[i + 1] = prefix_sum[i] + nums[i]

    dq = deque()
    min_len = float('inf')

    for i in range(len(prefix_sum)):
        while dq and prefix_sum[i] - prefix_sum[dq[0]] >= K:
            min_len = min(min_len, i - dq.popleft())

        while dq and prefix_sum[i] <= prefix_sum[dq[-1]]:
            dq.pop()

        dq.append(i)

    return min_len if min_len != float('inf') else -1

```

**时间复杂度**：

- **O(n)O(n)O(n)**：每个元素最多入队和出队一次。

**为什么使用单调队列**：

- 单调队列帮助快速找到满足条件的最短子数组。

---

### **3. LeetCode 1696: 跳跃游戏 VI（Jump Game VI）**

**题目描述**：  
给定一个数组，玩家可以跳跃 kkk 步，求跳到最后位置的最大得分。

**解法思路**：

- 使用单调递减队列维护前 kkk 步内的最大得分。

**Python 代码**：
```python
from collections import deque

def maxResult(nums, k):
    dq = deque([0])
    dp = [0] * len(nums)
    dp[0] = nums[0]

    for i in range(1, len(nums)):
        while dq and dq[0] < i - k:
            dq.popleft()

        dp[i] = nums[i] + dp[dq[0]]

        while dq and dp[i] >= dp[dq[-1]]:
            dq.pop()

        dq.append(i)

    return dp[-1]

```

**时间复杂度**：

- **O(n)O(n)O(n)**：遍历一次数组。

**为什么使用单调队列**：

- 单调队列可以高效维护滑动窗口内的最大得分，避免重复计算。

---

## **总结**

- **单调队列（Monotone Queue）** 通过维护单调性，在滑动窗口和动态最优解问题中提供高效的解决方案。
- 通过 LeetCode 的经典题目，可以熟练掌握单调队列的核心技巧。


# **<mark style="background: #FF5582A6;">栈（Stack）</mark>的详细介绍**

###### 棧
---

## **一、栈的原理与特点**

### **1. 原理**

- **栈（Stack）** 是一种**后进先出（LIFO, Last In First Out）** 的线性数据结构。
- 元素的**插入（push）** 和**删除（pop）** 操作只能在栈顶进行。.top
- 栈的操作受限于：
    - **push**：将元素压入栈顶。
    - **pop**：将栈顶元素弹出。
    - **peek（top）**：获取栈顶元素但不删除。

---

### **2. 特点**

- **后进先出（LIFO）**：最后插入的元素最先被移除。
- **操作受限**：只能操作栈顶，无法随机访问中间元素。
- **时间复杂度**：
    - **push** 和 **pop** 操作：O(1)。
    - **查找元素**：O(n)。

---

### **3. 常见用途**

- **括号匹配**：验证括号是否匹配，如 `({[]})`。
- **表达式求值**：中缀表达式、前缀表达式、后缀表达式计算。
- **函数调用栈**：维护递归调用顺序。
- **单调栈**：维护元素单调性，常用于求解下一个更大/更小元素。
- **深度優先搜索 [DFS](https://www.notion.so/7-DFS-26-26-e38b5329508d4a29bd21591a14439077)** 

---

## **二、具体例子**

假设有栈 `Stack = [1, 4, 6]`：

- **压入元素** `8`：
    - 新栈：`[1, 4, 6, 8]`。
- **弹出栈顶元素**：
    - 弹出 `8`，新栈：`[1, 4, 6]`。
- **查看栈顶元素**：
    - 栈顶元素：`6`。

---

## **三、Python 实作**

```python
# 使用列表实现栈
class Stack:
    def __init__(self):
        self.stack = []

    # 压入元素
    def push(self, val):
        self.stack.append(val)

    # 弹出栈顶元素
    def pop(self):
        if not self.is_empty():
            return self.stack.pop()
        else:
            print("栈为空")
            return None

    # 查看栈顶元素
    def top(self):
        if not self.is_empty():
            return self.stack[-1]
        else:
            print("栈为空")
            return None

    # 判断栈是否为空
    def is_empty(self):
        return len(self.stack) == 0

    # 打印栈
    def print_stack(self):
        print("栈内容:", self.stack)

# 示例
s = Stack()
s.push(1)
s.push(4)
s.push(6)
s.print_stack()  # 输出: [1, 4, 6]

print("栈顶元素:", s.top())  # 输出: 6

s.pop()
s.print_stack()  # 输出: [1, 4]

```

---

## **四、LeetCode 栈题目描述及解法**

以下是整理的 LintCode 中涉及栈（stack）的题目表格，包括题目编号、题目名称（英文）、题目简述（中文）、样例及解法：

| 题目编号                | 题目名称（英文）                                  | 题目简述（中文）                                          | 样例                                                                                                                     | 解法                                                                                   |
| ------------------- | ----------------------------------------- | ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| 12<br>*<br>(m)      | [[Min Stack]]带最小值操作的栈                     | 设计一个支持常数时间内获取最小值的栈，支持 `push`、`pop` 和 `getMin` 操作。 | 输入:  <br>stack.push(-2)  <br>stack.push(0)  <br>stack.push(-3)  <br>stack.getMin()  <br>输出: -3                         | 使用两个栈，一个存储元素，一个存储最小值，更新最小值时同步操作。                                                     |
| 859<br>*<br>(h)<br> | 最大栈 [[Max Stack]]                         | 设计一个支持 `push`、`pop` 和 `getMax` 操作的栈，能够返回栈中最大值。    | 输入:  <br>push(5), push(1), push(5), getMax(), pop(), getMax()  <br>输出: [5, 5]                                          | 使用两个栈，一个存储元素，另一个维护当前最大值，确保 `getMax` 操作的高效性。                                          |
| 423<br>*<br>(e)<br> | [[Valid Parentheses]] 有效的括号               | 给定一个只包含 ()、{} 和 [] 的字符串，判断字符串中的括号是否匹配             | 输入:  <br>Example: ({[()]}) <br>输出: True                                                                                | 使用 栈 逐个遍历字符串中的字符。遇到左括号时，将其压入栈中；遇到右括号时，检查栈顶是否为匹配的左括号，                                 |
| 575<br><br>(m)      | [[Decode String]]字符串解码                    | 给定一个编码字符串，解码并返回原字符串。                              | 输入:  <br>s = "3[a]2[bc]"  <br>输出: "aaabcbc"                                                                            | 使用栈存储当前字符和数字，遇到 `]` 时解码子字符串。                                                         |
| 978<br>*<br>(m)     | 基础计算器 [[Basic Calculator]]                | 实现一个简单的计算器，支持加减法和括号。                              | 输入:  <br>"(1+(4+5+2)-3)+(6+8)"  <br>输出: 23                                                                             | 使用栈存储运算符和操作数，遇到括号时计算其内部表达式的值。                                                        |
| 980<br>**<br>(m)    | 基础计算器II [[Basic CalculatorII]]            | 实现一个基础计算器来计算一个简单表达式字符串。                           | 输入: "3+2*2" 输出: 7                                                                                                      | 我们可以用一个栈，保存这些（进行乘除运算后的）整数的值。对于加减号后的数字，将其直接压入栈中；对于乘除号后的数字，可以直接与栈顶元素计算，并替换栈顶元素为计算后的结果。 |
| 849<br><br>(h)<br>  | 基础计算器III [[Basic Calculator III]]         | 实现一个支持加减乘除和括号的高级计算器。                              | 输入:  <br>"2*(5+5*2)/3+(6/2+8)"  <br>输出: 21                                                                             | 使用栈处理括号和优先级，先解析出括号内表达式，再计算最终结果。                                                      |
| 367<br>*<br>(h)     | 表达树构造 [[Expression Tree Build]]           | 从后缀表达式构造表达式树，支持加减乘除。                              | 输入:  <br>postfix = ["2", "3", "+", "4", "*"]  <br>输出: 表达式树                                                             | 使用栈存储中间节点，遇到运算符时弹出栈顶的两个节点构建子树并压栈。                                                    |
| 1908<br>*<br>(h)    | 布尔表达式求值 [[Boolean Expression Evaluation]] | 计算布尔表达式的结果，支持 `AND`、`OR` 和 `NOT` 运算。              | 输入:  <br>expression = "true AND false OR true"  <br>输出: true                                                           | 使用栈解析表达式，依次处理运算符和操作数，并按优先级进行布尔运算。                                                    |
| 510<br>*<br>(h)     | 最大矩形 [[Maximal Rectangle]]                | 给定一个二维二进制矩阵，找出其中包含最多1的矩形面积。                       | 输入:  <br>matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]  <br>输出: 6 | 将每行看作直方图高度数组，逐行调用 `Largest Rectangle in Histogram` 解法。                               |
| 126<br>*<br>(h)     | 最大树 [[Max Tree]]                          | 给定一个数组，构造一棵最大二叉树，规则是父节点大于其子节点且树的根节点为数组中最大值。       | 输入:  <br>nums = [3,2,1,6,0,5]  <br>输出: 最大树                                                                             | 使用单调栈构建树，维护每个节点的左右子树，遇到较大值时构造父节点并更新子树。                                               |
| 1860<br>*<br>(h)    | 0子矩阵的数量 [[The Number of 0-submatrix]]     | 计算矩阵中包含全0的子矩阵数量。                                  | 输入:  <br>matrix = [[0,0,1],[0,0,0],[1,0,0]]  <br>输出: 9                                                                 | 使用栈按列计算每一行的直方图高度，调用 `Largest Rectangle in Histogram` 解法统计。                           |
| 86<br><br>(h)<br>   | [[Binary Search Tree Iterator]]二叉查找树迭代器   | 实现二叉搜索树的迭代器，支持按升序遍历树中节点值的 `next` 操作。              | 输入:  <br>root = [7,3,15,null,null,9,20]  <br>输出: 3                                                                     | 使用栈记录路径，每次访问栈顶元素的右子树。                                                                |


## **五、选取三题详细解释**

---

### **1. LeetCode 20: 有效的括号（Valid Parentheses）**(lintcode 423)

**题目描述**：  
给定一个只包含 `()`、`{}` 和 `[]` 的字符串，判断字符串中的括号是否匹配。
Example:
({[()]})
output = True

**解法思路**：

- 使用 **栈** 逐个遍历字符串中的字符。
- 遇到左括号时，将其压入栈中；
- 遇到右括号时，检查栈顶是否为匹配的左括号，若不匹配或栈为空则返回 `False`；
- 遍历结束后，栈必须为空才为有效字符串。

**Python 代码**：
```python
def isValid(s: str) -> bool:
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}

    for char in s:
        if char in mapping:  # 遇到右括号
            top_element = stack.pop() if stack else '#'
            if mapping[char] != top_element:
                return False
        else:
            stack.append(char)  # 左括号入栈

    return not stack

```

**时间复杂度**：

- O(n)O(n)O(n)：遍历字符串一次。

**为什么使用栈**：

- 栈的后进先出特性非常适合匹配括号这种成对的结构。

---

### **2. LeetCode 155: 最小栈（Min Stack）**(lintcode 12)

**题目描述**：  
设计一个栈，支持 `push`、`pop`、`top` 操作，并能在常数时间内返回最小值。
Example:
push(1)
min()
push(2)
min()
push(3)
min()

**解法思路**：

- 使用 **辅助栈** 保存当前栈的最小值。
- 每次压入元素时，若元素小于等于辅助栈栈顶，将其也压入辅助栈。

**Python 代码**：
```python
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val: int) -> None:
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self) -> None:
        if self.stack.pop() == self.min_stack[-1]:
            self.min_stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]

```

**时间复杂度**：

- O(1)O(1)O(1)：所有操作的时间复杂度均为常数。

**为什么使用栈**：

- 栈与辅助栈的结合可以高效维护栈中的最小值。

---

## **总结**

- **栈（Stack）** 适用于**后进先出**的场景，如括号匹配、表达式求值、最小栈和单调栈问题。
- 通过 LeetCode 的经典题目，可以熟练掌握栈的基础操作和高级应用，如单调栈的使用技巧。


# **<mark style="background: #ADCCFFA6;">单调栈（Monotonic Stack）</mark>的详细介绍**

###### 單調棧
---

## **一、单调栈的原理与特点**

### **1. 原理**

- **单调栈（Monotonic Stack）** 是一种特殊的栈，其元素满足**单调性**：
    - **单调递增栈**：栈中的元素从栈底到栈顶**递增**。
    - **单调递减栈**：栈中的元素从栈底到栈顶**递减**。
- 当新元素进入栈时，栈会**弹出**不满足单调性（多余）的元素，保持栈的有序性。
- 单调栈适合用来解决**下一个更大元素**或**下一个更小元素**等问题。

---

### **2. 特点**

- **栈的单调性**：栈中的元素按一定顺序排列，维护单调性。
- **时间复杂度**：
    - 每个元素最多入栈和出栈一次，因此总时间复杂度为 O(n)。
- **空间复杂度**：栈最多存储 O(n)个元素。

---

### **3. 适用场景**

- **找下一个更大/更小的元素**：例如下一个更大的温度、下一个更大的高度等。
- **维护动态区间内的最值**：用于解决滑动窗口、直方图最大矩形面积等问题。

---

## **二、具体例子**

假设有数组 `nums = [2, 1, 2, 4, 3]`，要求找出每个元素**下一个更大元素**：

1. 从左到右遍历数组。
2. 使用**单调递减栈**：栈中保存未找到下一个更大元素的索引，栈顶元素最小。
3. 当新元素大于栈顶元素时，弹出栈顶元素，说明找到了栈顶元素的下一个更大元素。

**结果**：

- `nums[0] = 2`，下一个更大元素是 `4`。
- `nums[1] = 1`，下一个更大元素是 `2`。
- `nums[2] = 2`，下一个更大元素是 `4`。
- `nums[3] = 4`，下一个更大元素不存在（-1）。
- `nums[4] = 3`，下一个更大元素不存在（-1）。

---

## **三、Python 实作**

### **单调栈实现下一个更大元素**
```python
def nextGreaterElement(nums):
    n = len(nums)
    result = [-1] * n  # 初始化结果数组
    stack = []  # 单调递减栈，存储元素索引

    for i in range(n):
        # 栈非空且当前元素大于栈顶元素
        while stack and nums[i] > nums[stack[-1]]:
            index = stack.pop()
            result[index] = nums[i]
        stack.append(i)  # 当前元素索引入栈

    return result

# 示例
nums = [2, 1, 2, 4, 3]
print(nextGreaterElement(nums))  # 输出: [4, 2, 4, -1, -1]

```

---

## **四、LeetCode 单调栈题目描述及解法**


以下是整理的 LintCode 中涉及单调栈的题目表格，包括题目编号、题目名称（英文）、题目简述（中文）、样例及解法：

| 题目编号                 | 题目名称（英文）                                               | 题目简述（中文）                                                     | 样例                                                                                                                   | 解法                                                               |
| -------------------- | ------------------------------------------------------ | ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| 1206<br><br>(e)<br>  | [[Next Greater Element I]]<br>下一个更大元素 I                | 给定两个数组 `nums1` 和 `nums2`，找出 `nums1` 中每个元素在 `nums2` 中的下一个更大元素 | 输入:  <br>nums1 = [4,1,2], nums2 = [1,3,4,2].<br>输出: [-1,3,-1]                                                        | 使用 单调递减栈 来找到 nums2 中每个元素的下一个更大值, 使用 哈希表记录每个元素的下一个更大值，方便在nums1中查找 |
| 1201<br><br>(m)      | [[Next Greater Element II]]下一个更大的数 II                  | 给定一个循环数组，返回每个元素的下一个更大元素。                                     | 输入:  <br>nums = [1,2,1]  <br>输出: [2,-1,2]                                                                            | 使用单调栈处理数组的循环访问，维护索引映射。                                           |
| 1060<br><br>(m)      | [[Daily Temperatures]] 每日温度                            | 给定一个温度数组，返回每一天之后多少天会出现更高的温度。                                 | 输入<br>temperatures = [73, 74, 75, 71, 69, 72, 76, 73]<br>输出<br>[1, 1, 4, 2, 1, 1, 0, 0]                              | 使用 单调栈，栈中保存温度的索引，保持递减顺序。遍历温度数组时，若当前温度高于栈顶元素对应的温度，计算间隔天数。         |
| 122<br>*<br>(h)      | [[Largest Rectangle in Histogram]]<br>直方图中最大的矩形面积      | 给定一个直方图，求直方图中最大的矩形面积                                         | 输入:  <br>height = [2,1,5,6,2,3]<br>输出: 10                                                                            | 使用单调递增栈维护柱子的索引,每次遇到高度较低的柱子时，弹出栈顶柱子，计算以该柱子为高度的最大矩形面积              |
| 363<br>**<br>(m)     | [[Trapping Rain Water]]接雨水                             | 计算柱状图中能够存储的雨水总量。                                             | 输入:  <br>height = [0,1,0,2,1,0,1,3,  <br>2,1,2,1]  <br>输出:  <br>6                                                    | 使用双指针和单调栈记录左右最大高度，计算每个位置的雨水。                                     |
| 364<br>*<br>(h)      | [[Trapping Rain Water II]]接雨水II                        | 计算二维地形能够存储的雨水总量。                                             | 输入:<br>heightMap = <br>[[1,4,3,1,3,2],<br>[3,2,1,3,2,4],<br>[2,3,3,2,3,1]]<br><br>输出:<br>4<br>                       | 使用最小堆模拟单调栈，动态更新边界最低高度。                                           |
| 1297<br><br>(h)<br>  | [[Count of Smaller Numbers After Self]]比自己小的元素个数       | 给定一个整数数组，返回一个新数组，新数组中第 i 个元素是 nums[i] 右侧小于 nums[i] 的元素个数。    | 输入:  <br>nums = [5,2,6,1]  <br>输出:  <br>[2,1,1,0]                                                                    | 使用单调栈维护每个元素右侧的较小值统计。                                             |
| 1274<br><br>(m)<br>  | [[Find K Pairs with Smallest Sum]]查找和最小的K对数字           | 找到两个已排序数组中和最小的 k 对数对。                                        | 输入:  <br>nums1 = [1,7,11]  <br>nums2 = [2,4,6]  <br>k = 3  <br>输出:  <br>[[1,2],[1,4],[1,6]]                          | 使用最小堆或单调栈动态生成数对并维护大小。                                            |
| 564<br><br>(m)<br>   | [[Combination Sum IV]]组合总和 IV                          | 给定一个正整数数组，计算能够组成目标值的所有组合数目。                                  | 输入:  <br>nums = [1,2,3]  <br>target = 4  <br>输出:  <br>7                                                              | 使用动态规划和单调栈统计每种和的组合数。                                             |
| 1272<br><br>(m)<br>  | [[Kth Smallest Element in a Sorted Matrix]]有序矩阵中的第K小元素 | 在一个行列均递增的矩阵中找到第 k 小的元素。                                      | 输入:  <br>matrix = <br>[ [1,5,9],<br> [10,11,13],<br> [12,13,15] ]  <br>k = 8  <br><br>输出:  <br>13                    | 使用最小堆或单调栈动态维护当前最小值。                                              |
| 3659<br><br>(e)<br>  | [[Design Phone Directory]]电话目录管理系统                     | 设计一个电话目录，支持分配、回收电话号码和检查号码是否可用。                               | 输入:  <br>["PhoneDirectory",<br>"get","check",<br>"release"],<br>[ [3],[],[2],[] ]  <br>输出:  <br>[ null,0,true,null ] | 使用单调栈模拟分配和回收过程。                                                  |
| 1852<br>*<br>(m)<br> | [[Final Discounted Price]]最终优惠价                        | 给定一个商品价格列表，计算每个商品的最终价格，最终价格为原价减去右侧第一个小于等于当前商品价格的值，如果没有则保留原价。 | 输入: prices = [8, 4, 6, 2, 3]  <br>输出: [4, 2, 4, 2, 3]                                                                | 使用单调递增栈，记录索引以查找右侧第一个符合条件的价格。                                     |
| 285<br>*<br>(m)<br>  | [[Tall Building]]高楼大厦                                  | 给定每栋建筑物的高度数组，计算每栋建筑物能看到右侧至少一栋更高建筑物的最远距离。                     | 输入: heights = [5, 3, 8, 3, 2]  <br>输出: [2, 1, 0, 0, 0]                                                               | 使用单调递减栈，记录右侧第一个比当前建筑高的索引。                                        |
| 1740<br>*<br>(m)<br> | [[Online Stock Span]]股票价格跨度                            | 给定一个股票价格流，计算每一天股票价格不小于今天价格的连续天数。                             | 输入: prices = [100, 80, 60, 70, 60, 75, 85]  <br>输出: [1, 1, 1, 2, 1, 4, 6]                                            | 使用单调递减栈，存储价格和天数以计算跨度。                                            |
| 346<br>*<br>(h)<br>  | [[xorsum of Interval extremum]]区间极值异或                  | 给定一个数组，计算所有子区间的最小值与最大值的异或和。                                  | 输入: nums = [3, 1, 2, 4]  <br>输出: 11                                                                                  | 使用单调栈分别找到每个元素作为最小值和最大值时的贡献，计算结果的异或和。                             |
| 1778<br>*<br>(h)     | 奇偶跳 [[Odd Even Jump]]                                  | 给定一个数组，判断从数组每个位置出发的奇偶跳能否跳到数组末尾，返回能到达的起始位置数量。                 | 输入:  <br>arr = [10,13,12,14,15]  <br>输出: 2                                                                           | 使用单调栈分别计算奇跳和偶跳的下一个位置，动态规划记录能到达末尾的位置。                             |
| 347<br>*<br>(h)      | 最大值期望 [[Maximum Number Expectation]]                   | 给定一个数组和一个窗口大小，找到窗口内的最大值并计算期望值。                               | 输入:  <br>nums = [1,3,-1,-3,5,3,6,7], k = 3  <br>输出: [3,3,5,5,6,7]                                                    | 使用单调递减栈计算每个窗口的最大值，求平均作为期望值。                                      |
| 368<br>*<br>(h)      | 表达式求值 [[Expression Evaluation]]                        | 给一个用字符串表示的表达式数组，求出这个表达式的值。                                   |                                                                                                                      | 使用递归的方法，先处理 +-<br>再处理 */最后再处理括号。                                 |


### 說明

- **样例部分** 按題目輸入輸出的長度進行了換行處理，保持所有列寬一致。
- **解法部分** 使用簡明的文字描述解法，突出了單調棧的應用場景。


---

## **五、选取三题详细解释**

---

### **1. LeetCode 496: 下一个更大元素 I（Next Greater Element I）** (lintcode 1206)

**题目描述**：  
给定两个数组 `nums1` 和 `nums2`，找出 `nums1` 中每个元素在 `nums2` 中的下一个更大元素。
Example:
nums1 = [4,1,2], nums2 = [1,3,4,2] 
在nums2裡每個元素的下個更大元素分別是 1->3, 3->4, 而4,2沒有更大元素=-1
所以nums1=[4,1,2] -> output=[-1,3,-1]

**解法思路**：

- 使用 **单调递减栈** 来找到 `nums2` 中每个元素的下一个更大值。
- 使用 **哈希表** 记录每个元素的下一个更大值，方便在 `nums1` 中查找。

**Python 代码**：
```python
def nextGreaterElement(nums1, nums2):
    stack = []
    greater_map = {}

    # 遍历 nums2，使用单调栈找出每个元素的下一个更大元素
    for num in nums2:
        while stack and stack[-1] < num:
            greater_map[stack.pop()] = num
        stack.append(num)

    # nums1 中的元素通过哈希表查询
    return [greater_map.get(num, -1) for num in nums1]

# 示例
nums1 = [4, 1, 2]
nums2 = [1, 3, 4, 2]
print(nextGreaterElement(nums1, nums2))  # 输出: [-1, 3, -1]

```

**时间复杂度**：

- O(n)：每个元素最多入栈和出栈一次。

**为什么使用单调栈**：

- 单调栈可以高效找到每个元素的下一个更大值，避免重复遍历数组。

---

### **2. LeetCode 739: 每日温度（Daily Temperatures）**(lintcode 1060)

**题目描述**：  
给定一个温度数组，返回每一天需要等待多少天，才能等到更高的温度。

**解法思路**：

- 使用 **单调递减栈**：栈中存储温度的索引，保持递减顺序。
- 遍历温度数组，若当前温度大于栈顶索引对应的温度，说明找到了栈顶的下一个更高温度。

**Python 代码**：
```python
def dailyTemperatures(T):
    n = len(T)
    result = [0] * n
    stack = []

    for i, temp in enumerate(T):
        while stack and temp > T[stack[-1]]:
            index = stack.pop()
            result[index] = i - index
        stack.append(i)

    return result

# 示例
T = [73, 74, 75, 71, 69, 72, 76, 73]
print(dailyTemperatures(T))  # 输出: [1, 1, 4, 2, 1, 1, 0, 0]

```

**时间复杂度**：

- O(n)O(n)O(n)：每个元素最多入栈和出栈一次。

**为什么使用单调栈**：

- 单调栈能够高效维护温度的索引，快速找到下一个更高温度。

---

### **3. LeetCode 84: 直方图中最大的矩形面积（Largest Rectangle in Histogram）** (lintcode122)

**题目描述**：  
给定一个直方图，求直方图中最大的矩形面积。

**解法思路**：

- 使用 **单调递增栈** 维护柱子的索引。
- 每次遇到高度较低的柱子时，弹出栈顶柱子，计算以该柱子为高度的最大矩形面积。

**Python 代码**：
```python
def largestRectangleArea(heights):
    stack = []
    max_area = 0
    heights.append(0)  # 添加一个 0，保证栈中所有元素弹出

    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)

    return max_area

# 示例
heights = [2, 1, 5, 6, 2, 3]
print(largestRectangleArea(heights))  # 输出: 10

```

**时间复杂度**：

- O(n)O(n)O(n)：每个柱子最多入栈和出栈一次。

**为什么使用单调栈**：

- 单调栈可以快速找到左边和右边第一个小于当前柱子的高度。

---

## **总结**

- **单调栈（Monotonic Stack）** 通过维护栈的单调性，可以高效解决**下一个更大/更小元素**和**动态区间问题**。
- 在 LeetCode 中，单调栈常用于**滑动窗口**、**直方图问题** 和 **动态查询最优解**，时间复杂度通常为 O(n)O(n)O(n)。


# **<mark style="background: #FF5582A6;">哈希表（Hash Map）</mark>的详细介绍**

###### 哈希表
---

## **一、哈希表的原理与特点**

### **1. 原理**

- **哈希表（Hash Map）** 是一种基于 **键值对（key-value）** 的数据结构，支持快速查找、插入和删除操作。
- 通过**哈希函数**将**键（key）** 映射到数组中的索引位置，将数据存储在对应位置。
- **冲突解决**：当两个键经过哈希函数映射到同一个索引时，称为哈希冲突，常用的解决方法有：
    - **链式法**：将冲突的元素存储在一个链表中。
    - **开放寻址法**：寻找数组中的下一个空位置存储元素。

---

### **2. 特点**

- **时间复杂度**：
    - 插入、删除、查找的平均时间复杂度为 O(1)O(1)O(1)。
    - 最坏情况下（哈希冲突严重），时间复杂度为 O(n)O(n)O(n)。
- **数据存储**：非连续存储，通过哈希函数映射。
- **无序性**：哈希表中的数据是无序的。
- **唯一键**：每个键在哈希表中是唯一的，若重复插入相同的键，会覆盖之前的值。

---

### **3. 常见用途**

- **快速查找**：通过键快速定位值，例如统计字符频率。
- **去重**：使用哈希表存储唯一元素。
- **缓存**：实现 LRU 缓存机制等。
- **关联映射**：将一个数据映射到另一个数据，例如两数之和问题。

---

## **二、具体例子**

假设有一组数据 `Array = [1, 4, 6, 8]`，使用哈希表存储元素与其索引：

- **键（Key）**：元素值。
- **值（Value）**：元素的索引。

**结果**：  
哈希表：`{1: 0, 4: 1, 6: 2, 8: 3}`

- 通过 `hash_map[6]` 可以快速得到索引 `2`。

---

## **三、Python 实作**

### **基本哈希表操作**

```python
# 使用 Python 的字典（dict）实现哈希表
hash_map = {}

# 插入键值对
hash_map[1] = "A"
hash_map[2] = "B"
hash_map[3] = "C"

# 查找元素
print("键为 2 的值:", hash_map[2])  # 输出: B

# 删除元素
del hash_map[2]

# 判断键是否存在
if 2 in hash_map:
    print("存在键 2")
else:
    print("键 2 不存在")  # 输出: 键 2 不存在

# 遍历哈希表
for key, value in hash_map.items():
    print(f"键: {key}, 值: {value}")

```

---

## **四、LeetCode 使用哈希表的题目描述及解法**

以下是整理的 LintCode 中涉及哈希表的题目表格，包括题目编号、题目名称（英文）、题目简述（中文）、样例及解法：

| 题目编号               | 题目名称（英文）                                                 | 题目简述（中文）                             | 样例                                                                                                                                                      | 解法                                    |
| ------------------ | -------------------------------------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| 124<br>*<br>(m)    | [[Longest Consecutive Sequence]]最长连续序列                   | 给定一个未排序的整数数组，找出最长连续序列的长度。            | 输入:  <br>nums = [100,4,200,1,3,2]  <br>输出:  <br>4                                                                                                       | 使用哈希表记录数组中的数字，动态扩展连续序列的长度。            |
| 140<br><br>(m)     | [[Fast Power]]快速幂                                        | 计算 `a^b % n`。                        | 输入:  <br>a = 2  <br>b = 3  <br>n = 5  <br>输出:  <br>3                                                                                                    | 使用快速幂和哈希表记录中间结果。                      |
| 141<br><br>(e)     | [[Sqrt(x)]]对x开根                                          | 计算并返回非负整数 xxx 的平方根（只保留整数部分）。         | 输入: 8 输出: 2                                                                                                                                             | 使用二分查找法确定平方根的整数部分。                    |
| 142<br><br>(e)     | [[O(1) Check Power of 2]]   O(1)时间检测2的幂次                 | 判断一个整数是否是 2 的幂次。                     | 输入:  <br>n = 8  <br>输出:  <br>true                                                                                                                       | 使用哈希表记录所有 2 的幂次进行查找。                  |
| 143<br><br>(m)     | [[Sort Colors II]]颜色分类 II                                | 对包含 k 种颜色的数组进行排序，颜色从 1 到 k。          | 输入:  <br>colors = [3,2,2,1,4]  <br>k = 4  <br>输出:  <br>[1,2,2,3,4]                                                                                      | 使用哈希表记录每种颜色出现的频率后重建数组。                |
| 144<br><br>(m)<br> | [[Interleaving Positive and Negative Numbers]]交错正负数      | 重新排列数组使得正负数交替出现。                     | 输入:  <br>nums = [-1,-2,1,2]  <br>输出:  <br>[1,-1,2,-2]                                                                                                   | 使用哈希表和双指针分别记录正数和负数并交替排列。              |
| 145<br><br>(e)<br> | [[Lowercase to Uppercase]]大小写转换                          | 将给定的字符串中的小写字母转换为大写字母。                | 输入:  <br>char = "a"  <br>输出:  <br>"A"                                                                                                                   | 使用哈希表记录字母映射关系，动态转换。                   |
| 146<br><br>(e)     | [[Lowercase to Uppercase II]]大小写转换II                     | 将字符串中的所有小写字母转换为大写字母。                 | 输入:  <br>s = "abc"  <br>输出:  <br>"ABC"                                                                                                                  | 遍历字符串，用哈希表映射小写字母到大写字母。                |
| 147<br><br>(e)<br> | [[Narcissistic Number]]水仙花数                              | 判断一个数是否是自恋数（阿姆斯特朗数）。                 | 输入:  <br>num = 153  <br>输出:  <br>true                                                                                                                   | 使用哈希表记录每个位的次方计算结果。                    |
| 148<br><br>(m)<br> | [[Sort Colors]]颜色分类                                      | 对数组中的三种颜色（0、1、2）进行排序。                | 输入:  <br>nums = [2,0,2,1,1,0]  <br>输出:  <br>[0,0,1,1,2,2]                                                                                               | 使用哈希表记录每种颜色的频率后重建数组。                  |
| 152<br><br>(m)     | [[Combinations]]组合                                       | 找到从 n 个数字中选择 k 个数字的所有组合。             | 输入:  <br>n = 4  <br>k = 2  <br>输出:  <br>[[2,4],[3,4],[2,3],[1,2],[1,3],[1,4]]                                                                           | 使用回溯法和哈希表记录每次选择的数字。                   |
| 153<br><br>(m)<br> | [[Combination Sum II]]组合II                               | 找到数组中和等于目标值的所有不重复组合。                 | 输入:  <br>candidates = [10,1,2,7,6,1,5]  <br>target = 8  <br>输出:  <br>[[1,1,6],[1,2,5],[1,7],[2,6]]                                                      | 使用回溯法结合哈希表记录已访问的数字避免重复。               |
| 154<br>*<br>(h)    | [[Regular Expression Matching]]正则表达式匹配                   | 实现一个支持 `.` 和 `*` 的正则表达式匹配器。          | 输入:  <br>s = "aa"  <br>p = "a*"  <br>输出:  <br>true                                                                                                      | 使用动态规划和哈希表记录子问题状态。                    |
| 155<br><br>(e)     | [[Minimum Depth of Binary Tree]]二叉树的最小深度                 | 找到二叉树的最小深度。                          | 输入:  <br>root = [3,9,20,null,null,15,7]  <br>输出:  <br>2                                                                                                 | 使用哈希表记录节点的深度，动态更新最小深度。                |
| 685<br>*<br>(m)    | 数据流中第一个唯一的数字[[First Unique Number in Data Stream]]       | 设计一个数据结构支持数据流中的第一个唯一数字的查询和插入。        | 输入:  <br>add(2), add(3), add(5), showFirstUnique()  <br>输出: 2                                                                                           | 使用哈希表记录数字出现次数，并维护一个队列按顺序存储候选唯一数字。     |
| 960<br>*<br>(m)    | 数据流中第一个独特的数 II [[First Unique Number in Data Stream II]] | 设计一个数据结构支持高效查询数据流中第一个独特数字，并允许删除任意数字。 | 输入:  <br>add(2), add(2), add(3), remove(2), showFirstUnique()  <br>输出: 3                                                                                | 哈希表记录每个数字的状态（唯一或重复），结合链表维护唯一数字的顺序。    |
| 657<br>*<br>(m)    | O(1)实现数组插入/删除/随机访问 [[Insert Delete GetRandom O(1)]]      | 设计一个数据结构支持O(1)时间的插入、删除和随机访问操作。       | 输入:  <br>insert(1), remove(1), getRandom()  <br>输出: 1                                                                                                   | 使用数组存储元素，哈希表记录元素索引，实现插入和删除的快速访问。      |
| 613<br>*<br>(m)    | 优秀成绩 [[High Five]]                                       | 给定一个学生成绩的记录，返回每个学生的前五门成绩的平均值。        | 输入:  <br>records = [[1,91],[1,92],[2,93],[2,99],[1,60],[2,77],[1,65],[1,87],[1,100],[2,100],[2,76]]  <br>输出: [[1,87],[2,88]]                            | 使用哈希表记录学生ID及其所有成绩，排序后取前五个成绩计算平均值。     |
| 128<br>**<br>(m)   | 哈希函数 [[Hash Function]]                                   | 设计一个简单的哈希函数，将字符串映射到指定大小的哈希表。         | 输入:  <br>key = "hello", HASH_SIZE = 10  <br>输出: 2                                                                                                       | 使用字符串的ASCII值结合哈希表大小进行取模运算。            |
| 129<br>**<br>(m)   | 重哈希 [[Rehashing]]                                        | 给定一个哈希表和一个新的大小，对其进行重新哈希，使得数据分布更加均匀。  | 输入:  <br>hash_table = [null,21,null,null,14,null]  <br>输出: [null,null,null,null,14,21]                                                                  | 遍历原哈希表，将非空元素根据新大小重新计算哈希值并插入新哈希表。      |
| 1280<br>*<br>(h)   | 将数据流变为多个不相交区间 [[Data Stream as Disjoint Intervals]]      | 给定一个整数流，实现一个数据结构支持动态维护不相交区间。         | 输入:  <br>addNum(1), addNum(3), getIntervals()  <br>输出: [[1, 1], [3, 3]]                                                                                 | 使用有序字典记录区间的起始和结束，插入时动态合并重叠区间。         |
| 547<br>**<br>(e)   | 两数组的交集 [[Intersection of Two Arrays]]                    | 给定两个数组，返回它们的交集，结果中不包含重复元素。           | 输入:  <br>nums1 = [1,2,2,1], nums2 = [2,2]  <br>输出: [2]                                                                                                  | 使用哈希表记录第一个数组的元素，遍历第二个数组时检查是否存在于哈希表中。  |
| 548<br>*<br>(e)    | 两数组的交集 II [[Intersection of Two Arrays II]]              | 给定两个数组，返回它们的交集，结果中可以包含重复元素。          | 输入:  <br>nums1 = [1,2,2,1], nums2 = [2,2]  <br>输出: [2,2]                                                                                                | 使用哈希表记录第一个数组中每个元素的次数，遍历第二个数组时按次数匹配交集。 |
| 793<br>*<br>(e)    | 多个数组的交集 [[Intersection of Arrays]]                       | 给定多个数组，返回它们的交集。                      | 输入:  <br>nums = [[1,2,2,1],[2,2,3],[2,2,4]]  <br>输出: [2]                                                                                                | 使用哈希表记录每个元素在所有数组中的出现次数，检查次数是否等于数组总数。  |
| 1848<br>*<br>(h)   | 单词搜索 III [[Word Search III]]                             | 给定一个单词列表和一个二维字符网格，找出所有单词在网格中的位置。     | 输入:  <br>board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], words = ["oath","pea","eat","rain"]  <br>输出: ["oath","eat"] | 使用字典树构建单词索引，结合DFS遍历网格查找所有单词。          |
| 1413<br><br>(m)    | 树 [[Tree]]                                               | 构建哈希表存储树的父节点和子节点关系，支持树的增删查操作。        | 输入:  <br>添加节点，删除节点，查找节点  <br>输出: 树结构                                                                                                                    | 使用嵌套哈希表存储节点关系，动态支持树的维护。               |
| 132<br>*<br>(h)    | 单词搜索 II [[Word Search II]]                               | 给定一个二维字符网格和单词字典，找出所有字典中的单词在网格中的位置。   | 输入:  <br>board = [["o","a","b","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], words = ["oath","pea","eat","rain"]  <br>输出: ["oath","eat"] | 使用字典树存储单词，结合DFS遍历网格查找。                |


## **五、选取三题详细解释**

---

### **1. LeetCode 1: 两数之和（Two Sum）**

**题目描述**：  
给定一个整数数组 `nums` 和一个目标值 `target`，返回数组中两个数的索引，使它们的和等于目标值。

**解法思路**：

- 使用 **哈希表** 存储遍历过的元素值和索引。
- 每次遍历时，检查 `target - nums[i]` 是否存在于哈希表中。

**Python 代码**：
```python
def twoSum(nums, target):
    hash_map = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in hash_map:
            return [hash_map[complement], i]
        hash_map[num] = i

```

**时间复杂度**：

- O(n)O(n)O(n)：只需遍历数组一次。

**为什么使用哈希表**：

- 哈希表可以快速查找 `target - num` 是否存在，时间复杂度为 O(1)O(1)O(1)。

---

### **2. LeetCode 128: 最长连续序列（Longest Consecutive Sequence）**

**题目描述**：  
给定一个未排序的整数数组，找出最长连续元素序列的长度。

**解法思路**：

- 使用 **哈希表** 存储数组中的所有元素。
- 遍历数组，查找当前元素的连续前驱（`num - 1`）是否存在，若不存在则开始向后查找连续序列长度。

**Python 代码**：
```python
def longestConsecutive(nums):
    num_set = set(nums)  # 使用哈希表去重
    longest = 0

    for num in num_set:
        if num - 1 not in num_set:  # 当前元素是序列的起点
            current_num = num
            current_length = 1

            while current_num + 1 in num_set:
                current_num += 1
                current_length += 1

            longest = max(longest, current_length)

    return longest

```

**时间复杂度**：

- O(n)O(n)O(n)：每个元素最多访问两次（查找和遍历）。

**为什么使用哈希表**：

- 哈希表用于快速查找元素是否存在，时间复杂度为 O(1)O(1)O(1)。

---

### **3. LeetCode 242: 有效的字母异位词（Valid Anagram）**

**题目描述**：  
判断两个字符串 `s` 和 `t` 是否为字母异位词（包含相同字符且字符出现次数相等）。

**解法思路**：

- 使用 **哈希表** 统计字符串 `s` 和 `t` 中每个字符的出现次数。
- 比较两个哈希表是否相等。

**Python 代码**：
```python
def isAnagram(s: str, t: str) -> bool:
    if len(s) != len(t):
        return False

    count = {}
    for char in s:
        count[char] = count.get(char, 0) + 1
    
    for char in t:
        if char not in count or count[char] == 0:
            return False
        count[char] -= 1
    
    return all(val == 0 for val in count.values())

```

**时间复杂度**：

- O(n)O(n)O(n)：遍历字符串一次。

**为什么使用哈希表**：

- 哈希表可以高效地统计字符出现次数，并快速比较两个字符串。

---

## **总结**

- **哈希表（Hash Map）** 适合快速查找、插入和删除操作，平均时间复杂度为 O(1)O(1)O(1)。
- 在解决**去重**、**映射关系** 和 **统计频率** 等问题时，哈希表是最常用的数据结构。
- 在 LeetCode 中，哈希表的应用广泛，经典题目如**两数之和**、**最长连续序列** 和 **有效字母异位词** 等均利用了哈希表的高效查找特性。

# **<mark style="background: #ADCCFFA6;">并查集（Union-Find）</mark>的详细介绍**

###### 併查集
---

## **一、并查集的原理与特点**

### **1. 原理**

- **并查集（Union-Find）** 是一种用于管理**不相交集合（Disjoint Sets）** 的数据结构。
- 它支持两种核心操作：
    - **合并（Union）**：将两个元素所属的集合合并为一个集合。
    - **查找（Find）**：确定某个元素属于哪个集合，通常返回该集合的代表元素（根节点）。

### **2. 特点**

- **树结构表示集合**：并查集使用**树**的形式来表示集合，每个集合的元素指向一个**根节点**，根节点代表集合的标识。
- **路径压缩**：在查找操作中，将路径上的节点直接连接到根节点，从而加速查找操作。
- **按秩合并**：合并时将较小的树挂到较大的树上，保证树的高度尽量小。
- **时间复杂度**：
    - **查找（Find）** 和 **合并（Union）** 的平均时间复杂度为 O(α(n))O(\alpha(n))O(α(n))，其中 α(n)\alpha(n)α(n) 是**阿克曼函数的反函数**，接近常数时间。
- **适用场景**：
    - 解决**连通性**问题。
    - 最小生成树算法（如 Kruskal 算法）。
    - 群组分类或动态连接问题。

---

## **二、具体例子**

假设有以下元素与集合：

- 初始状态：`{1} {2} {3} {4}`（每个元素是一个独立的集合）。
- 执行操作：
    - **Union(1, 2)**：将 `1` 和 `2` 合并，结果：`{1, 2} {3} {4}`。
    - **Union(2, 3)**：将 `2` 和 `3` 合并，结果：`{1, 2, 3} {4}`。
    - **Find(3)**：返回 `3` 所在集合的代表元素（根节点 `1`）。

---

## **三、Python 实作**

### **并查集基本实现**
```python
class UnionFind:
    def __init__(self, n):
        self.parent = [i for i in range(n)]  # 初始化每个元素的父节点为自己
        self.rank = [1] * n  # 记录树的高度（秩）

    # 查找操作，路径压缩
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 路径压缩
        return self.parent[x]

    # 合并操作，按秩合并
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x != root_y:  # 只有根节点不同才合并
            if self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            elif self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1

    # 判断两个元素是否在同一集合
    def connected(self, x, y):
        return self.find(x) == self.find(y)

# 示例
uf = UnionFind(5)  # 初始化 5 个元素
uf.union(0, 1)
uf.union(1, 2)
print(uf.connected(0, 2))  # 输出: True
print(uf.connected(0, 3))  # 输出: False

```

---

## **四、LeetCode 使用并查集的题目描述及解法**

以下是整理的 LintCode 中涉及并查集（Union-Find）的题目表格，包括题目编号、题目名称（英文）、题目简述（中文）、样例及解法：

| 题目编号                | 题目名称（英文）                                                                | 题目简述（中文）                           | 样例                                                                                                                                                                                        | 解法                                 |
| ------------------- | ----------------------------------------------------------------------- | ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| 432<br><br>(m)      | [[Find the Weak Connected Component in the Directed Graph]]找出有向图中的弱连通分量 | 找到有向图中所有的弱连通分量。                    | 输入:  <br>nodes = <br>[ [1,2],[2,3],[4,5] ]  <br><br>输出:  <br>[ [1,2,3],[4,5] ]                                                                                                            | 使用并查集维护节点的连通性，合并有边的节点。             |
| 629<br>*<br>(h)     | [[Minimum Spanning Tree]]最小生成树                                          | 在加权无向图中找到最小生成树。                    | 输入:  <br>edges = [[1,2,1],[2,3,2],[1,3,2]]  <br>输出:  <br>3                                                                                                                                | 使用并查集和 Kruskal 算法找到最小生成树。          |
| 684<br><br>(e)<br>  | [[Redundant Connection]]缺少的字符串                                          | 在图中找到冗余的边，使得移除这条边后图仍然是连通树。         | 输入:<br>edges = <br>[ [1,2],[1,3],<br>[2,3] ] <br>输出:<br>[2,3]                                                                                                                             | 使用并查集判断新边是否会形成环，若形成则为冗余边。          |
| 1087<br>*<br>(m)    | [[Redundant Connection II]]缺少的字符串II                                     | 在图中找到一条冗余的边，使得移除这条边后图仍然是连通的有向树。    | 输入:<br>edges = <br>[ [1,2],[1,3],<br>[2,3] ]<br><br>输出:<br>[2,3]                                                                                                                          | 使用并查集和入度检查判断冗余边。                   |
| 1070<br>*<br>(m)    | [[Accounts Merge]]账户合并                                                  | 合并具有相同邮件地址的账户。                     |                                                                                                                                                                                           | 使用并查集将邮箱地址合并为一个集合。                 |
| 855<br><br>(m)      | [[Sentence Similarity II]]句子相似性II                                       | 判断两个句子是否相似，每对相似单词通过列表给出。           | 输入:<br>[ "great","acting"<br>,"skills" ]<br>[ "fine","drama",<br>"talent" ]<br>输出:<br>true                                                                                                | 使用并查集合并相似单词，判断对应位置单词是否属于同一集合。      |
| 1043<br><br>(h)<br> | [[Couples Holding Hands]]夫妻手牵手                                          | 计算最少的交换次数，使得情侣在一起。                 | 输入:  <br>row = [0,2,1,3]  <br>输出:  <br>1                                                                                                                                                  | 使用并查集维护每对情侣的关系，合并需要交换的座位。          |
| 1718<br><br>(h)<br> | [[Minimize Malware Spread]]尽量减少恶意软件的传播                                  | 移除一个节点，尽量减少恶意软件的传播范围。              | 输入:<br>graph = <br>[ [1,1,0]<br>,[1,1,0],<br>[0,0,1] ]<br>initial = <br>[ 0,1 ]<br>输出:<br>0                                                                                               | 使用并查集统计每个连通分量的大小和恶意节点的分布情况。        |
| 3670<br><br>(m)<br> | [[The Earliest Moment When Everyone Become Friends]]彼此认识的最早时间           | 找到所有人成为朋友的最早时间。                    | 输入:<br>logs = <br>[ 20190101,0,1],<br>[20190104,3,4],<br>[20190107,2,3],<br>[20190109,1,2],<br>[20190110,1,3],<br>[20190113,4,5] ]<br><br>输出:<br>20190109                                 | 使用并查集动态合并好友关系，记录合并时间。              |
| 3672<br><br>(m)<br> | [[Connecting Cities With Minimum Cost]]最低成本联通所有城市                       | 使用最小成本连接所有城市。                      | 输入:<br>n = 3<br>connections = <br>[ [1,2,5],<br>[1,3,6],<br>[2,3,1] ]<br>输出:<br>6                                                                                                         | 使用并查集结合 Kruskal 算法计算最小生成树成本。       |
| 178<br>**<br>(m)    | 图是否是树 [[Graph Valid Tree]]                                              | 判断一个无向图是否是一棵树。                     | 输入: n = 5 <br>edges = [<br>[0, 1], [0, 2], <br>[0, 3], [1, 4]]<br>输出: true                                                                                                                | 使用并查集判断图是否连通且无环。                   |
| 444<br>*<br>(m)     | 图是否是树 II [[Graph Valid Tree II]]                                        | 判断一个加权图是否是一棵树。                     | 输入:<br>addEdge(1, 2)<br>isValidTree()<br>addEdge(1, 3)<br>isValidTree()<br>addEdge(1, 5)<br>isValidTree()<br>addEdge(3, 5)<br>isValidTree()<br>输出: <br>["true","true",<br>"true","false"] | 使用并查集结合最小生成树算法（Kruskal）检查连通性和边的数量。 |
| 589<br>*<br>(m)     | 连接图 [[Connecting Graph]]                                                | 判断图中两个节点是否连通，并支持动态合并节点。            | 输入:<br>ConnectingGraph(5)<br>query(1, 2)<br>connect(1, 2)<br>query(1, 3) <br>connect(2, 4)<br>query(1, 4) <br>输出:<br>[false,false,true]                                                   | 使用并查集维护节点连通关系，查询时检查根节点是否相同。        |
| 590<br>*<br>(m)     | 连接图II [[Connecting Graph II]]                                           | 在图的基础上增加边权，并支持动态合并和连通性查询。          | 输入:<br>ConnectingGraph2(5)<br>query(1)<br>connect(1, 2)<br>query(1)<br>connect(2, 4)<br>query(1)<br>connect(1, 4)<br>query(1)<br><br>输出:<br>[1,2,3,3]                                     | 并查集基础上记录每个集合的边权和，查询时检查连通性并计算边权和。   |
| 591<br>*<br>(m)     | 连接图III [[Connecting Graph III]]                                         | 支持动态删除和添加边，并判断图的连通性。               | 输入:<br>ConnectingGraph3(5)<br>query()<br>connect(1, 2)<br>query()<br>connect(2, 4)<br>query()<br>connect(1, 4)<br>query()<br><br>输出:[5,4,3,3]                                             | 使用启发式合并和路径压缩的并查集动态维护连通关系。          |
| 1014<br>*<br>(h)    | 打砖块 [[Bricks Falling When Hit]]                                         | 给定一个二维砖块网格和一系列敲砖操作，计算每次操作后剩余的砖块数量。 | 输入: <br>grid = [<br>[1,0,0,0],<br>[1,1,1,0]], <br>hits = [[1,0]]<br>输出: [2]<br>                                                                                                           | 逆序执行敲砖操作，使用并查集维护砖块连通性，动态更新剩余数量。    |
| 805<br>*<br>(m)     | 最大关联集合 [[Maximum Association Set]]                                      | 找到多个字符串集合中关联度最大的集合。                | 输入:  <br>ListA = <br>["abc","abc","abc"], <br>ListB = <br>["bcd","acd","def"]<br>输出:  <br>["abc","acd","bcd","def"]<br>                                                                   | 使用并查集合并关联集合，找到最大集合并返回。             |
| 1463<br>*<br>(m)    | 论文查重 [[Paper Review]]                                                   | 检查论文引用网络中是否存在循环引用。                 |                                                                                                                                                                                           | 使用并查集判断是否存在环路结构。                   |
| 1179<br>*<br>(m)    | 朋友圈 [[Friend Circle]]                                                   | 计算社交网络中的朋友圈数量。                     | 输入：<br>[ [1,1,0],<br>[1,1,0],<br>[0,0,1] ]<br>输出：2<br>                                                                                                                                    | 使用并查集合并相互认识的好友，最终统计独立朋友圈的数量。       |
| 1396<br>*<br>(m)    | 集合合并 [[Set Union]]                                                      | 支持集合的动态合并和查询操作。                    | 输入：<br>list = [<br>[1,2,3],<br>[3,9,7],<br>[4,5,10] ]<br>输出：2                                                                                                                             | 使用路径压缩优化并查集，实现高效的集合操作。             |
| 1628<br><br>(m)<br> | 开车问题 [[Driving Problem]]                                                | 判断多个城市之间的道路是否连通。                   | 输入:  <br>roads = [[1,2],[2,3],[4,5]], queries = [[1,3],[2,4]]  <br>输出: [true, false]                                                                                                      | 使用并查集动态维护城市连通性，查询时判断两个城市是否属于同一集合。  |
| 1430<br><br>(h)<br> | 相似字符串组 [[Similar String Groups]]                                        | 给定一组字符串，计算相似字符串的组数。                | 输入:  <br>strings = ["tars","rats","arts","star"]  <br>输出: 2                                                                                                                               | 使用并查集将相似的字符串合并为一个组，统计独立组的数量。       |


## **五、选取三题详细解释**

---

### **1. LeetCode 200: 岛屿数量（Number of Islands）**

**题目描述**：  
给定一个二维网格，其中 `'1'` 表示陆地，`'0'` 表示水，计算网格中**岛屿的数量**。

**解法思路**：

- 使用 **并查集** 将相邻的陆地格子合并到同一个集合中。
- 最终计算并查集中的连通分量个数，即岛屿数量。

**Python 代码**：
```python
class UnionFind:
    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.rank = [1] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            if self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            elif self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1

def numIslands(grid):
    rows, cols = len(grid), len(grid[0])
    uf = UnionFind(rows * cols)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == '1':
                        uf.union(r * cols + c, nr * cols + nc)

    # 统计根节点的数量
    unique_islands = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                unique_islands.add(uf.find(r * cols + c))
    return len(unique_islands)

```

**时间复杂度**：

- O(m×n)O(m \times n)O(m×n)：遍历整个网格一次。

**为什么使用并查集**：

- 并查集可以高效合并相邻的陆地格子，快速计算连通分量的数量。

---

### **2. LeetCode 547: 省份数量（Number of Provinces）**

**题目描述**：  
给定一个城市的邻接矩阵，计算有多少个省份（连通分量）。

**解法思路**：

- 使用 **并查集** 合并相连的城市，最后统计集合的数量。

**Python 代码**：
```python
def findCircleNum(isConnected):
    n = len(isConnected)
    uf = UnionFind(n)

    for i in range(n):
        for j in range(n):
            if isConnected[i][j] == 1:
                uf.union(i, j)

    return len(set(uf.find(i) for i in range(n)))

```

---

### **3. LeetCode 684: 冗余连接（Redundant Connection）**

**题目描述**：  
给定一棵树添加了一条多余的边，找出这条边。

**解法思路**：

- 使用 **并查集** 检测环，当合并过程中发现两个节点已经在同一个集合中，则该边为冗余边。

**Python 代码**：
```python
def findRedundantConnection(edges):
    uf = UnionFind(len(edges))

    for u, v in edges:
        if uf.connected(u - 1, v - 1):
            return [u, v]
        uf.union(u - 1, v - 1)

```

---

## **总结**

- **并查集（Union-Find）** 适用于**动态连通性**问题，如连通分量、岛屿数量、冗余边检测等。
- 核心操作为**路径压缩**和**按秩合并**，保证了高效的时间复杂度 O(α(n))O(\alpha(n))O(α(n))。
- 并查集是解决图论和动态连通问题的重要工具。


# **<mark style="background: #ADCCFFA6;">迭代器（Iterator）</mark>的详细介绍**
###### 迭代器
---

## **一、迭代器的原理与特点**

### **1. 原理**

- **迭代器（Iterator）** 是一种用于逐个访问容器（如数组、链表、集合等）中的元素的数据结构。
- 迭代器提供了一种通用的方式来遍历数据结构，而无需了解数据结构的底层实现。
- 迭代器通过两个核心方法实现：
    - **`__iter__()`**：返回迭代器本身。
    - **`__next__()`**：返回容器中的下一个元素，若没有元素则抛出 `StopIteration` 异常。

### **2. 特点**

- **线性遍历**：迭代器适合逐个访问容器中的元素。
- **懒加载**：迭代器不会一次性加载所有数据，适合处理大数据集或无限数据流。
- **不可逆**：大多数迭代器是单向的，不能回退到上一个元素。
- **通用性**：支持多种数据结构（如数组、链表、字典、集合等）的遍历。

---

### **3. 适用场景**

- **遍历数据结构**：迭代器提供统一的遍历接口，简化数据结构的遍历操作。
- **生成大数据集**：通过懒加载实现对大数据流的逐步访问。
- **自定义迭代器**：允许用户定义如何访问数据。

---

## **二、具体例子**

假设有一个数组 `Array = [1, 4, 6, 8]`，迭代器将依次访问数组中的每个元素。

1. 调用 `__iter__()` 获取迭代器对象。
2. 调用 `__next__()` 返回数组中的下一个元素。

**示例访问过程**：

- 调用 `next()` 第一次：返回 `1`。
- 调用 `next()` 第二次：返回 `4`。
- 调用 `next()` 第三次：返回 `6`。
- 调用 `next()` 第四次：返回 `8`。
- 调用 `next()` 第五次：抛出 `StopIteration` 异常。

---

## **三、Python 实作**

### **Python 自带迭代器**
```python
# 使用内置迭代器遍历列表
my_list = [1, 4, 6, 8]
iterator = iter(my_list)  # 获取迭代器对象

print(next(iterator))  # 输出: 1
print(next(iterator))  # 输出: 4
print(next(iterator))  # 输出: 6
print(next(iterator))  # 输出: 8

# 使用 for 循环遍历迭代器（自动调用 __iter__ 和 __next__）
for item in my_list:
    print(item)

```

### **自定义迭代器**
```python
class MyIterator:
    def __init__(self, data):
        self.data = data
        self.index = 0

    def __iter__(self):
        return self  # 返回迭代器对象本身

    def __next__(self):
        if self.index < len(self.data):
            value = self.data[self.index]
            self.index += 1
            return value
        else:
            raise StopIteration  # 没有更多元素时抛出异常

# 使用自定义迭代器
my_data = [1, 4, 6, 8]
my_iter = MyIterator(my_data)

for value in my_iter:
    print(value)  # 输出: 1 4 6 8

```

---

## **四、LeetCode 使用迭代器的题目描述及解法**


以下是整理的 LintCode 中涉及迭代器（Iterator）的题目表格，包括题目编号、题目名称（英文）、题目简述（中文）、样例及解法：

| 题目编号                | 题目名称（英文）                                          | 题目简述（中文）                        | 样例                                                                                            | 解法                       |
| ------------------- | ------------------------------------------------- | ------------------------------- | --------------------------------------------------------------------------------------------- | ------------------------ |
| 540<br><br>(m)<br>  | [[Zigzag Iterator]]左旋右旋迭代器                        | 给定两个列表，实现一个迭代器，以之字形顺序输出两个列表的元素。 | 输入:  <br>v1 = [1,2]  <br>v2 = [3,4,5,6]  <br>输出:  <br>[1,3,2,4,5,6]                           | 使用队列存储非空列表的迭代器，按顺序依次访问。  |
| 528<br><br>(m)      | [[Flatten Nested List Iterator]]摊平嵌套的列表           | 将嵌套列表展开为一个扁平化的迭代器。              | 输入:  <br>nestedList = [[1,1],2,[1,1]]  <br>输出:  <br>[1,1,2,1,1]                               | 使用栈存储嵌套列表，递归展开当前列表项。     |
| 606<br><br>(m)      | [[Kth Largest Element II]]第K大的元素 II               | 实现一个迭代器以动态计算数据流中的第 k 大元素。       | 输入:  <br>nums = [4,5,8,2]  <br>k = 3  <br>输出:  <br>[4]                                        | 使用最小堆动态维护数据流中的第 k 大元素。   |
| 486<br><br>(m)      | [[Merge K Sorted Arrays]]合并k个排序数组                 | 合并 k 个已排序的数组，返回一个排序后的列表。        | 输入: <br>  [<br>    [1,2,3],<br>    [1,2]<br>  ]<br>输出: <br>[ 1,1,2,2,3 ]                      | 使用最小堆存储每个数组当前最小值的迭代器。    |
| 24<br>*<br>(h)      | [[LFU Cache]]   LFU缓存策略                           | 实现 LFU 缓存，支持 `get` 和 `put` 操作。  |                                                                                               | 使用哈希表和双向链表动态维护最少使用的元素。   |
| 601<br>*<br>(m)     | [[Flatten 2D Vector]]摊平二维向量                       | 将二维数组展开为一个扁平化的迭代器。              | 输入:<br>vec2d = <br>[ [1,2],[3],[4,5,6] ]<br>输出:<br>[ 1,2,3,4,5,6 ]                            | 使用索引记录当前行和列，动态访问元素。      |
| 59<br><br>(m)<br>   | [[3 Sum Closest]]最接近的三数之和                         | 找到数组中三个数的和最接近目标值。               | 输入:  <br>nums = [-1,2,1,-4]  <br>target = 1  <br>输出:  <br>2                                   | 排序后使用双指针遍历，动态更新最接近的和。    |
| 647<br>*<br>(m)<br> | [[Find All Anagrams in a String]]子串字谜             | 在字符串中找到目标字符串所有变位词的起始索引。         | 输入:  <br>s = "cbaebabacd"  <br>p = "abc"  <br>输出:  <br>[0,6]                                  | 使用滑动窗口和哈希表记录字符频率。        |
| 648<br><br>(m)      | [[Unique Word Abbreviation]]单词缩写集                 | 实现一个单词缩写系统，检查一个缩写是否唯一。          | 输入:  <br>dictionary = ["deer","door","cake","card"]  <br>isUnique("dear")  <br>输出:  <br>false | 使用哈希表记录单词及其缩写关系，动态检查唯一性。 |
| 649<br><br>(m)      | [[Binary Tree Vertical Order Traversal]]<br>二叉树翻转 | 按垂直顺序遍历二叉树节点。                   | 输入:<br>root = <br>[ 3,9,20,null,<br>null,15,7 ]<br>输出:<br>[ [9],[3,15],<br>[20],[7] ]]]       | 使用哈希表记录每列节点，按列编号排序输出。    |
| 650<br><br>(m)      | [[Find Leaves of Binary Tree]]二叉树叶子顺序遍历           | 找到二叉树中所有叶子节点并移除，返回结果。           | 输入:<br>root = <br>[ 1,2,3,4,5 ]<br>输出:<br>[ [4,5,3],<br>[2],[1] ]                             | 使用递归记录节点深度，将叶子节点按深度分组。   |
| 651<br><br>(m)<br>  | [[Flatten 2D Vector II]]二叉树垂直遍历                   | 将二维数组展开为一个扁平化的迭代器（支持动态删除）。      | 输入:<br>vec2d = <br>[ [1,2],[3],<br>[4,5,6] ]<br>输出:<br>[ 1,2,3,4,5,6 ]                        | 使用索引和标记动态管理行列访问。         |


## **五、选取三题详细解释**

---

### **1. LeetCode 284: 预览迭代器（Peeking Iterator）**

**题目描述**：  
实现一个迭代器，使得可以在不调用 `next()` 的情况下**预览**下一个元素。

**解法思路**：

- 使用封装原始迭代器的方式，添加一个额外的变量来存储下一个元素。
- 提供 `peek()` 方法返回下一个元素，但不移动迭代器指针。

**Python 代码**：
```python
class PeekingIterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self._next = next(self.iterator, None)

    def peek(self):
        return self._next

    def next(self):
        value = self._next
        self._next = next(self.iterator, None)
        return value

    def hasNext(self):
        return self._next is not None

# 示例
nums = iter([1, 2, 3])
peek_iter = PeekingIterator(nums)
print(peek_iter.peek())  # 输出: 1
print(peek_iter.next())  # 输出: 1
print(peek_iter.next())  # 输出: 2

```

**时间复杂度**：

- **`next()` 和 `peek()`**：O(1)O(1)O(1)。

**为什么使用迭代器**：

- 在标准迭代器基础上封装，满足预览需求而不改变迭代逻辑。

---

### **2. LeetCode 173: 二叉搜索树迭代器（Binary Search Tree Iterator）**

**题目描述**：  
实现一个二叉搜索树（BST）的迭代器，按中序遍历的顺序访问树中的元素。

**解法思路**：

- 使用 **栈** 来模拟中序遍历，懒加载访问节点。

**Python 代码**：
```python
class BSTIterator:
    def __init__(self, root):
        self.stack = []
        self._leftmost_inorder(root)

    def _leftmost_inorder(self, root):
        while root:
            self.stack.append(root)
            root = root.left

    def next(self) -> int:
        top_node = self.stack.pop()
        if top_node.right:
            self._leftmost_inorder(top_node.right)
        return top_node.val

    def hasNext(self) -> bool:
        return len(self.stack) > 0

```

**时间复杂度**：

- **`next()` 和 `hasNext()`**：均为 O(1)O(1)O(1)（均摊）。

**为什么使用迭代器**：

- 提供了按需访问树中节点的机制，避免一次性加载所有节点。

---

### **3. LeetCode 341: 扁平化嵌套列表迭代器（Flatten Nested List Iterator）**

**题目描述**：  
给定一个嵌套列表，实现一个迭代器，逐个返回列表中的元素。

**解法思路**：

- 使用栈或递归将嵌套列表**扁平化**。
- 每次调用 `next()` 时，返回下一个元素。

**Python 代码**：
```python
class NestedIterator:
    def __init__(self, nestedList):
        self.stack = []
        self.flatten(nestedList)

    def flatten(self, nestedList):
        for item in reversed(nestedList):
            if item.isInteger():
                self.stack.append(item.getInteger())
            else:
                self.flatten(item.getList())

    def next(self) -> int:
        return self.stack.pop()

    def hasNext(self) -> bool:
        return len(self.stack) > 0

```

**时间复杂度**：

- O(n)O(n)O(n)：访问所有元素的时间复杂度。

**为什么使用迭代器**：

- 提供逐个访问嵌套结构的元素，避免一次性展开整个列表，节省空间。

---

## **总结**

- **迭代器（Iterator）** 适用于逐个访问数据结构中的元素，且无需关心底层实现。
- 它具有通用性，广泛应用于数据流处理、自定义数据结构遍历等场景。
- 在 LeetCode 上，迭代器常用于解决**树遍历**、**扁平化嵌套结构** 和 **数据流管理** 等问题。


# **<mark style="background: #FF5582A6;">二叉树（Binary Tree）</mark>的详细介绍**
###### 二叉树
---

## **一、二叉树的原理与特点**

### **1. 原理**

- **二叉树（Binary Tree）** 是一种**树形结构**的数据结构，其中每个节点最多有两个子节点，分别是**左子节点（Left Child）** 和 **右子节点（Right Child）**。
- 二叉树有多种形式，常见类型包括：
    - **普通二叉树**：每个节点最多两个子节点。
    - **满二叉树**：所有节点的子节点都存在且每层节点数达到最大值。
    - **完全二叉树**：除了最后一层，所有层的节点都被填满，且最后一层的节点从左到右排列。
    - **二叉搜索树（BST）**：左子节点的值小于根节点，右子节点的值大于根节点。
    - **平衡二叉树**：所有节点的左右子树高度差不超过 1。

### **2. 特点**

- **层次结构**：根节点在最上层，叶节点在最下层。
- **递归结构**：二叉树的每个子树本身也是一棵二叉树。
- **遍历方式**：
    - **前序遍历（Preorder）**：根 → 左 → 右。
    - **中序遍历（Inorder）**：左 → 根 → 右。
    - **后序遍历（Postorder）**：左 → 右 → 根。
    - **层序遍历（Level Order）**：按层次逐层访问。

### **3. 时间复杂度**

- **查找/插入/删除**（二叉搜索树）：平均 O(log⁡n)O(\log n)O(logn)，最坏 O(n)O(n)O(n)。
- **遍历整个树**：O(n)O(n)O(n)，其中 nnn 为节点数。

---

## **二、具体例子**

假设有一棵二叉树：
```markdown
       5
      / \
     3   8
    / \   \
   2   4   9

```

- **前序遍历**：`5 → 3 → 2 → 4 → 8 → 9`
- **中序遍历**：`2 → 3 → 4 → 5 → 8 → 9`
- **后序遍历**：`2 → 4 → 3 → 9 → 8 → 5`
- **层序遍历**：`5 → 3 → 8 → 2 → 4 → 9`

---

## **三、Python 实作**

### **定义二叉树节点类**
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

```

### **前序、中序、后序遍历的递归实现**
```python
# 前序遍历
def preorderTraversal(root):
    if not root:
        return []
    return [root.val] + preorderTraversal(root.left) + preorderTraversal(root.right)

# 中序遍历
def inorderTraversal(root):
    if not root:
        return []
    return inorderTraversal(root.left) + [root.val] + inorderTraversal(root.right)

# 后序遍历
def postorderTraversal(root):
    if not root:
        return []
    return postorderTraversal(root.left) + postorderTraversal(root.right) + [root.val]

```

### **层序遍历（BFS）**
```python
from collections import deque

def levelOrderTraversal(root):
    if not root:
        return []

    result = []
    queue = deque([root])
    
    while queue:
        node = queue.popleft()
        result.append(node.val)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    
    return result

```

---

## **四、LeetCode 使用二叉树的题目描述及解法**

以下是整理的 LintCode 中涉及二叉树的题目表格，包括题目编号、题目名称（英文）、题目简述（中文）、最简单样例及解法：

| 题目编号                      | 题目名称（英文）                                                       | 题目简述（中文）                   | 样例                                                                                  | 解法                                              |
| ------------------------- | -------------------------------------------------------------- | -------------------------- | ----------------------------------------------------------------------------------- | ----------------------------------------------- |
| 66<br><br>(e)             | [[Binary Tree Preorder Traversal]]二叉树的前序遍历                     | 返回二叉树的前序遍历结果。              | 输入:  <br>root = [1,null,2,3]  <br>输出:  <br>[1,2,3]                                  | 使用递归或栈完成前序遍历操作。                                 |
| 67<br>*<br>(e)            | [[Binary Tree Inorder Traversal]]- 二叉树的中序遍历                    | 返回二叉树的中序遍历结果。              | 输入:  <br>root = [1,null,2,3]  <br>输出:  <br>[1,3,2]                                  | 使用递归或栈完成中序遍历操作。                                 |
| 68<br><br>(e)<br>         | [[Binary Tree Postorder Traversal]]- 二叉树的后序遍历                  | 返回二叉树的后序遍历结果。              | 输入:  <br>root = [1,null,2,3]  <br>输出:  <br>[3,2,1]                                  | 使用递归或栈完成后序遍历操作。                                 |
| 70<br>*<br>(m)<br>        | [[Binary Tree Level Order Traversal II]]二叉树的层次遍历 II            | 返回二叉树的层次遍历结果（从底层到顶层）。      | 输入:  <br>root = [3,9,20,null,null,15,7]  <br>输出:  <br>[[15,7],[9,20],[3]]           | 使用队列按层遍历节点，结果反转。                                |
| 88<br><br>(m)             | [[Lowest Common Ancestor of a Binary Tree]]最近公共祖先              | 找到二叉树中两个节点的最近公共祖先。         | 输入:  <br>root = [3,5,1,6,2,0,8,null,null,7,4]  <br>p = 5  <br>q = 1  <br>输出:  <br>3 | 使用递归同时检查左子树和右子树，找到公共祖先。                         |
| 94<br>*<br>(m)            | [[Binary Tree Maximum Path Sum]]二叉树中的最大路径和                     | 计算二叉树中从任意节点到任意节点的路径的最大和。   | 输入:  <br>root = [1,2,3]  <br>输出:  <br>6                                             | 使用递归计算每个节点的最大路径和，动态更新全局最大值。                     |
| 97<br>*<br>(e)            | [[Maximum Depth of Binary Tree]]二叉树的最大深度                       | 计算二叉树的最大深度。                | 输入:  <br>root = [3,9,20,null,null,15,7]  <br>输出:  <br>3                             | 使用递归计算每个节点的最大深度。                                |
| 175<br>*<br>(e)           | [[Invert Binary Tree]]翻转二叉树                                    | 翻转二叉树的左右子树。                | 输入:  <br>root = [4,2,7,1,3,6,9]  <br>输出:  <br>[4,7,2,9,6,3,1]                       | 使用递归交换每个节点的左右子树。                                |
| 614<br><br>(m)            | [[Binary Tree Longest Consecutive Sequence II]] 二叉树的最长连续子序列 II | 找到二叉树中的最长连续序列路径，可以从任意节点开始。 | 输入:  <br>root = [1,2,3,4]  <br>输出:  <br>3                                           | 使用递归计算以当前节点为起点的最长递增和递减序列长度。                     |
| 632<br><br>(e)            | [[Binary Tree Maximum Node]]二叉树的最大节点                           | 找到二叉树中值最大的节点。              | 输入:  <br>root = [1,-5,3,1,2,null,4]  <br>输出:  <br>4                                 | 使用递归遍历树，动态更新最大值节点。                              |
| 7<br>*<br>(m)             | 二叉树的序列化和反序列化 [[Serialize and Deserialize Binary Tree]]         | 将二叉树转换为字符串并能够从字符串恢复二叉树。    | 输入:  <br>root = [1,2,3,null,null,4,5]  <br>输出: "1,2,3,null,null,4,5"                | 使用前序遍历序列化二叉树，空节点用标记符占位，反序列化时递归构建树。              |
| 864<br><br>(m)            | 相等树划分 [[Equal Tree Partition]]                                 | 判断是否能将二叉树分为两个和相等的部分。       | 输入:  <br>root = [5,10,10,null,null,2,3]  <br>输出: true                               | 使用后序遍历计算子树和并记录所有子树和，判断是否存在一半总和的子树。              |
| 689<br>*<br>(m)           | [[Two Sum IV]] - Input is a BST两数之和 - BST版本                    | 在二叉搜索树中找到两个节点，使它们的和等于目标值。  | 输入:  <br>root = [5,3,6,2,4,null,7]  <br>target = 9  <br>输出:  <br>true<br>O(n),O(n)  | 使用中序遍历获取节点值并查找目标值，或使用哈希表记录访问的节点值。               |
| 1240<br><br>(e)<br>!!<br> | [[Path Sum III]]路径总和III                                        | 找到二叉树中和为目标值的所有路径的数量。       | 输入: root = [10,5,-3,3,2,null,11,3,-2,null,1], sum = 8 输出: 3<br><br>O(n),O(n)        | 使用递归计算每个节点的路径和，利用前缀和的思想存储路径和的累计值，检查是否存在符合条件的路径。 |


## **五、选取三题详细解释**

---

### **1. LeetCode 94: 二叉树的中序遍历（Binary Tree Inorder Traversal）**

**题目描述**：  
返回给定二叉树的**中序遍历**结果。

**解法思路**：

- 使用 **递归** 访问左子树 → 根节点 → 右子树。
- 也可以使用 **栈** 进行迭代实现。

**Python 代码（递归实现）**：
```python
def inorderTraversal(root):
    if not root:
        return []
    return inorderTraversal(root.left) + [root.val] + inorderTraversal(root.right)

```

**Python 代码（迭代实现）**：
```python
def inorderTraversal(root):
    stack, result = [], []
    current = root

    while current or stack:
        while current:
            stack.append(current)
            current = current.left
        current = stack.pop()
        result.append(current.val)
        current = current.right

    return result

```

**时间复杂度**：

- O(n)O(n)O(n)：访问树中的所有节点。

**为什么使用二叉树**：

- 二叉树的中序遍历可以用于**有序数据**的访问，尤其在二叉搜索树中。

---

### **2. LeetCode 102: 二叉树的层序遍历（Binary Tree Level Order Traversal）**

**题目描述**：  
返回二叉树的**层序遍历**结果（从上到下逐层访问）。

**解法思路**：

- 使用 **队列**（BFS）按层访问节点，每次访问时，将当前节点的左右子节点加入队列。

**Python 代码**：
```python
from collections import deque

def levelOrderTraversal(root):
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)

    return result

```

**时间复杂度**：

- O(n)O(n)O(n)：访问树中的所有节点。

**为什么使用二叉树**：

- 层序遍历广泛用于搜索树中的节点，适合按层次访问数据。

---

### **3. LeetCode 226: 翻转二叉树（Invert Binary Tree）**

**题目描述**：  
翻转一棵二叉树，即交换所有节点的左右子节点。

**解法思路**：

- 使用 **递归** 交换每个节点的左右子树。

**Python 代码**：
```python
def invertTree(root):
    if not root:
        return None
    root.left, root.right = invertTree(root.right), invertTree(root.left)
    return root

```

**时间复杂度**：

- O(n)O(n)O(n)：遍历树中的所有节点。

**为什么使用二叉树**：

- 翻转二叉树是对树结构操作的经典示例。

---

## **总结**

- **二叉树（Binary Tree）** 是一种基本且广泛使用的数据结构，适用于搜索、排序和层次结构表示。
- 通过遍历方式（前序、中序、后序、层序），二叉树能够高效地处理树形数据。
- 在 LeetCode 中，二叉树的应用场景丰富，常用于解决树的遍历、路径和查找问题。


# **<mark style="background: #BBFABBA6;">二叉搜索树（Binary Search Tree）</mark>的详细介绍**
###### 二叉搜索树
---

## **一、二叉搜索树的原理与特点**

### **1. 原理**

- **二叉搜索树（Binary Search Tree, BST）** 是一种特殊的二叉树，具有以下性质：
    1. **左子树** 的所有节点的值均**小于**根节点的值。
    2. **右子树** 的所有节点的值均**大于**根节点的值。
    3. 左右子树本身也是一棵二叉搜索树。
- 这种结构保证了树中的元素是**有序**的，便于查找、插入和删除操作。

---

### **2. 特点**

- **查找操作**：
    - 从根节点开始，若目标值小于当前节点，继续向左子树查找；若大于当前节点，向右子树查找。
- **插入操作**：
    - 类似于查找，找到合适的位置插入新节点，确保二叉搜索树的性质不变。
- **删除操作**：
    - 删除节点分三种情况：
        1. **节点无子节点**：直接删除。
        2. **节点有一个子节点**：将子节点替代被删除节点。
        3. **节点有两个子节点**：找到右子树的最小节点替代当前节点。
- **时间复杂度**：
    - **平均情况下**：查找、插入、删除的时间复杂度均为 O(log⁡n)O(\log n)O(logn)。
    - **最坏情况下**（退化为链表）：时间复杂度为 O(n)O(n)O(n)。

---

## **二、具体例子**

假设有以下元素：`[5, 3, 8, 2, 4, 7, 9]`  
构建一棵二叉搜索树：
```markdown
       5
      / \
     3   8
    / \  / \
   2  4 7  9

```

- **查找** `4`：
    - `4 < 5`，向左子树移动。
    - `4 > 3`，向右子树移动，找到节点 `4`。
- **插入** `6`：
    - `6 > 5`，向右子树移动。
    - `6 < 8`，向左子树移动，插入到节点 `7` 的左子树。
- **删除** `3`：
    - 节点 `3` 有两个子节点，找到右子树的最小值 `4` 替代 `3`。

---

## **三、Python 实作**

### **定义二叉搜索树的节点类和基本操作**
```python
class TreeNode:
    def __init__(self, val=0):
        self.val = val
        self.left = None
        self.right = None

class BST:
    def __init__(self):
        self.root = None

    # 插入操作
    def insert(self, val):
        if not self.root:
            self.root = TreeNode(val)
        else:
            self._insert(self.root, val)

    def _insert(self, node, val):
        if val < node.val:
            if node.left:
                self._insert(node.left, val)
            else:
                node.left = TreeNode(val)
        else:
            if node.right:
                self._insert(node.right, val)
            else:
                node.right = TreeNode(val)

    # 查找操作
    def search(self, val):
        return self._search(self.root, val)

    def _search(self, node, val):
        if not node or node.val == val:
            return node
        if val < node.val:
            return self._search(node.left, val)
        return self._search(node.right, val)

    # 中序遍历（返回有序元素）
    def inorderTraversal(self):
        result = []
        self._inorder(self.root, result)
        return result

    def _inorder(self, node, result):
        if node:
            self._inorder(node.left, result)
            result.append(node.val)
            self._inorder(node.right, result)

# 示例
bst = BST()
bst.insert(5)
bst.insert(3)
bst.insert(8)
bst.insert(2)
bst.insert(4)
bst.insert(7)
bst.insert(9)
print("中序遍历结果:", bst.inorderTraversal())  # 输出: [2, 3, 4, 5, 7, 8, 9]
print("查找 4:", bst.search(4) is not None)  # 输出: True

```

---

## **四、LeetCode 使用二叉搜索树的题目描述及解法**

以下是整理的 LintCode 中涉及二叉搜索树（BST）的题目表格，包括题目编号、题目名称（英文）、题目简述（中文）、最简单样例及解法：

| 题目编号             | 题目名称（英文）                                                               | 题目简述（中文）                              | 最简单样例                                                                                 | 解法                               |
| ---------------- | ---------------------------------------------------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------- | -------------------------------- |
| 87<br><br>(h)    | [[Remove Node in Binary Search Tree]]删除二叉查找树的节点                        | 删除二叉搜索树中的指定节点。                        | 输入:  <br>root = [5,3,6,2,4,null,7]  <br>key = 3  <br>输出:  <br>[5,4,6,2,null,null,7]   | 找到目标节点，调整子树结构以保持二叉搜索树性质。         |
| 95<br>*<br>(m)   | [[Validate Binary Search Tree]]验证二叉查找树                                 | 验证一个二叉树是否为有效的二叉搜索树。                   | 输入:  <br>root = [2,1,3]  <br>输出:  <br>true                                            | 使用递归检查节点值是否满足二叉搜索树的性质。           |
| 98<br><br>(m)    | [[Sort List]]链表排序                                                      | 将升序链表转换为高度平衡的二叉搜索树。                   | 输入:  <br>head = [-10,-3,0,5,9]  <br>输出:  <br>[0,-3,9,-10,null,5]                      | 使用快慢指针找到链表中点作为根节点，递归构造左右子树。      |
| 448<br><br>(m)   | [[Inorder Successor in BST]]二叉查找树的中序后继                                 | 找到二叉搜索树中指定节点的中序后继。                    | 输入:  <br>root = [2,1,3]  <br>p = 1  <br>输出:  <br>2                                    | 使用递归或迭代寻找目标节点，并根据二叉搜索树性质找到后继节点。  |
| 900<br><br>(e)   | [[Closest Binary Search Tree Value]]二叉搜索树中最接近的值                        | 找到二叉搜索树中与目标值最接近的节点值。                  | 输入:  <br>root = [4,2,5,1,3]  <br>target = 3.714286  <br>输出:  <br>4                    | 使用递归或迭代遍历树节点，动态更新最接近的节点值。        |
| 901<br><br>(h)   | [[Closest Binary Search Tree Value II]]二叉搜索树中最接近的值 II                  | 找到二叉搜索树中与目标值最接近的 k 个节点值。              | 输入:  <br>root = [4,2,5,1,3]  <br>target = 3.714286  <br>k = 2  <br>输出:  <br>[4,3]     | 使用中序遍历记录节点值，并用滑动窗口动态选择最近的 k 个值。  |
| 903<br><br>(m)   | [[Range Sum of BST]]范围加法                                               | 计算二叉搜索树中值在指定范围内的所有节点值的和。              | 输入:  <br>root = [10,5,15,3,7,null,18]  <br>L = 7  <br>R = 15  <br>输出:  <br>32         | 使用递归只访问在范围内的子树节点，动态累加节点值。        |
| 701<br><br>(m)   | [[Trim a Binary Search Tree]]修剪二叉搜索树                                   | 修剪二叉搜索树，使得所有节点值在给定范围内。                | 输入:  <br>root = [1,0,2]  <br>L = 1  <br>R = 2  <br>输出:  <br>[1,null,2]                | 使用递归修剪节点，不在范围内的子树替换为空。           |
| 661<br><br>(e)   | [[Binary Search Tree to Greater Sum Tree]]把二叉搜索树转化成更大的树                | 将二叉搜索树转换为一个新的树，每个节点值为原值加上所有大于它的节点值的和。 | 输入 : {5,2,13}<br>输出 : {18,20,13}<br>                                                  | 使用反向中序遍历动态更新节点值。                 |
| 1744<br>*<br>(e) | [[Increasing Order Search Tree]]递增顺序查找树                                | 将二叉搜索树转换为一个递增顺序的单链表。                  |                                                                                       | 使用中序遍历重组树的结构，按递增顺序连接节点。          |
| 1359<br><br>(e)  | [[Convert Sorted Array to Binary Search Tree]]有序数组转换为二叉搜索树             | 将升序数组转换为高度平衡的二叉搜索树。                   | 输入:  <br>nums = [-10,-3,0,5,9]  <br>输出:  <br>[0,-3,9,-10,null,5]                      | 使用递归分治将数组中间元素作为根节点，构造左右子树。       |
| 1746<br><br>(e)  | [[Minimum Distance Between BST Nodes]]二叉搜索树结点最小距离                      | 找到二叉搜索树中任意两个节点值之间的最小差值。               | 输入:  <br>root = [4,2,6,1,3]  <br>输出:  <br>1                                           | 使用中序遍历动态记录上一个节点值和当前节点值的差值。       |
| 1593<br><br>(m)  | [[Construct Binary Search Tree from Preorder Traversal]]根据前序和后序遍历构造二叉树 | 从先序遍历的结果构造二叉搜索树。                      | 输入:  <br>preorder = [8,5,1,7,10,12]  <br>输出:  <br>[8,5,10,1,7,null,12]                | 使用递归或栈根据先序遍历构造二叉搜索树。             |
| 597<br>**<br>(e) | 具有最大平均数的子树 [[Subtree with Maximum Average]]                            | 找出二叉树中具有最大平均数的子树。                     | 输入:  <br>root = [1,-5,11,1,2,4,-2]  <br>输出: [11,4,-2]                                 | 使用递归后序遍历计算子树的节点和和节点数，动态更新最大平均值。  |
| 596<br>*<br>(e)  | 最小子树 [[Minimum Subtree]]                                               | 找出二叉树中和最小的子树。                         | 输入:  <br>root = [1,-5,2,1,2,3,-4]  <br>输出: [-5,1,2]                                   | 使用后序遍历计算子树的和，动态记录最小值及其根节点。       |
| 474<br>**<br>(e) | 最近公共祖先II [[Lowest Common AncestorII (LCA)]]                            | 找出二叉树中两个节点的最近公共祖先，节点可能不存在树中。          | 输入:  <br>root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4  <br>输出: 5                | 递归检查左右子树是否包含目标节点，根节点是最近公共祖先。     |
| 88<br>**<br>(m)  | 最近公共祖先 [[Lowest Common Ancestor (LCA)]]                                | 找出二叉树中两个节点的最近公共祖先，节点一定存在树中。           | 输入:  <br>root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1  <br>输出: 3                | 递归检查左右子树是否包含目标节点，结合二叉树特性优化查找。    |
| 578<br>*<br>(m)  | 最近公共祖先III [[Lowest Common Ancestor (LCAIII)]]                          | 找出二叉树中多个节点的最近公共祖先。                    | 输入:  <br>root = [3,5,1,6,2,0,8,null,null,7,4], nodes = [5,4]  <br>输出: 5               | 递归检查子树是否包含目标节点集合，根节点是最近公共祖先。     |
| 902<br>**<br>(m) | BST中第K小的元素 [[Kth Smallest Element in a BST]]                           | 找出二叉搜索树中第K小的元素。                       | 输入:  <br>root = [3,1,4,null,2], k = 1  <br>输出: 1                                      | 使用中序遍历计数到第K个节点返回结果，或记录子树节点数加速查找。 |
| 11<br>*<br>(m)   | 二叉查找树中搜索区间 [[Search Range in Binary Search Tree]]                      | 找出二叉搜索树中在指定区间内的所有节点值。                 | 输入:  <br>root = [10,5,15,3,7,null,18], range = [7,15]  <br>输出: [7,10,15]              | 使用递归结合区间条件过滤节点，返回符合条件的值。         |
| 85<br>*<br>(e)   | 在二叉查找树中插入节点 [[Insert Node in a Binary Search Tree]]                    | 在二叉搜索树中插入一个新节点。                       | 输入:  <br>root = [4,2,7,1,3], val = 5  <br>输出: [4,2,7,1,3,5]                           | 递归定位插入位置，构造新节点插入树中。              |
| 1524<br><br>(e)  | 在二叉搜索树中查找 [[Search in a Binary Search Tree]]                           | 在二叉搜索树中查找一个值是否存在，并返回该节点。              | 输入:  <br>root = [4,2,7,1,3], val = 2  <br>输出: [2,1,3]                                 | 使用递归或迭代根据二叉搜索树特性快速定位目标节点。        |
| 33<br>*<br>(m)   | N皇后问题（一） [[N-Queens]]                                                  | 在N×N的棋盘上放置N个皇后，确保它们互相不攻击，返回所有可能的解法。   | 输入:  <br>n = 4  <br>输出: [[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]] | 使用回溯算法逐行放置皇后，验证安全性并记录解法。         |
| 1181<br>*<br>(e) | 二叉树的直径 [[Diameter of Binary Tree]]                                     | 找出二叉树中两个节点之间的最长路径长度。                  | 输入:  <br>root = [1,2,3,4,5]  <br>输出: 3                                                | 使用后序遍历计算每个节点的左右子树深度，动态更新最长路径长度。  |




## **五、选取三题详细解释**

---

### **1. LeetCode 700: 在二叉搜索树中查找节点（Search in a Binary Search Tree）**

**题目描述**：  
在二叉搜索树中查找值等于 `val` 的节点，返回该节点的子树。

**解法思路**：

- 根据 **BST 的性质**，若 `val` 小于当前节点值，向左子树查找；若大于，向右子树查找。

**Python 代码**：
```python
def searchBST(root, val):
    if not root or root.val == val:
        return root
    if val < root.val:
        return searchBST(root.left, val)
    return searchBST(root.right, val)

```

**时间复杂度**：

- O(log⁡n)O(\log n)O(logn)（平均情况）；O(n)O(n)O(n)（最坏情况）。

**为什么使用 BST**：

- BST 可以高效地进行查找操作。

---

### **2. LeetCode 701: 向二叉搜索树插入节点（Insert into a Binary Search Tree）**

**题目描述**：  
向二叉搜索树中插入一个新节点，返回插入后的树。

**解法思路**：

- 递归找到合适的位置插入新节点，保持 BST 的性质。

**Python 代码**：
```python
def insertIntoBST(root, val):
    if not root:
        return TreeNode(val)
    if val < root.val:
        root.left = insertIntoBST(root.left, val)
    else:
        root.right = insertIntoBST(root.right, val)
    return root

```

**时间复杂度**：

- O(log⁡n)O(\log n)O(logn)（平均情况）；O(n)O(n)O(n)（最坏情况）。

**为什么使用 BST**：

- BST 保证了插入操作的高效性。

---

### **3. LeetCode 450: 删除二叉搜索树中的节点（Delete Node in a BST）**

**题目描述**：  
删除 BST 中的一个节点，并保证树仍然满足二叉搜索树的性质。

**解法思路**：

- 根据要删除节点的子树情况分三种情况处理：
    1. **无子节点**：直接删除。
    2. **有一个子节点**：将子节点替代被删除节点。
    3. **有两个子节点**：用右子树的最小值替代当前节点。

**Python 代码**：
```python
def deleteNode(root, key):
    if not root:
        return None

    if key < root.val:
        root.left = deleteNode(root.left, key)
    elif key > root.val:
        root.right = deleteNode(root.right, key)
    else:
        if not root.left:
            return root.right
        if not root.right:
            return root.left

        min_larger_node = findMin(root.right)
        root.val = min_larger_node.val
        root.right = deleteNode(root.right, min_larger_node.val)

    return root

def findMin(node):
    while node.left:
        node = node.left
    return node

```

**时间复杂度**：

- O(log⁡n)O(\log n)O(logn)（平均情况）；O(n)O(n)O(n)（最坏情况）。

**为什么使用 BST**：

- BST 提供高效的节点删除操作，符合有序性。

---

## **总结**

- **二叉搜索树（BST）** 是一种有序数据结构，适用于高效查找、插入和删除操作。
- 通过递归实现，可以完成多种操作，如节点查找、插入、删除和查找第 kkk 小的元素。
- 在 LeetCode 中，BST 相关题目常考察其有序性与递归的特性，掌握 BST 的核心性质非常重要。

# **<mark style="background: #FF5582A6;">堆（Heap）</mark>的详细介绍**
###### 堆
---

## **一、堆的原理与特点**

### **1. 原理**

- **堆（Heap）** 是一种特殊的**完全二叉树**，满足以下性质：
    - **大根堆（Max-Heap）**：每个节点的值都**大于或等于**其子节点的值，根节点是最大值。
    - **小根堆（Min-Heap）**：每个节点的值都**小于或等于**其子节点的值，根节点是最小值。
- 堆的实现通常基于**数组**，通过索引关系维护父节点和子节点：
    - **父节点索引**：`i`
    - **左子节点索引**：`2 * i + 1`
    - **右子节点索引**：`2 * i + 2`

---

### **2. 特点**

- **插入操作**：新元素插入堆的末尾，然后通过**上浮操作**恢复堆的性质。
- **删除最大/最小值**：移除根节点，将堆尾元素移动到根节点，通过**下沉操作**恢复堆的性质。
- **时间复杂度**：
    - **插入操作**：O(log⁡n)O(\log n)O(logn)。
    - **删除操作**：O(log⁡n)O(\log n)O(logn)。
    - **查找最大/最小值**：O(1)O(1)O(1)。
- **存储方式**：堆的实现一般基于数组，而不是链式结构。

---

### **3. 适用场景**

- **优先队列**：根据优先级快速获取最大值或最小值。
- **排序算法**：堆排序是一种 O(nlog⁡n)O(n \log n)O(nlogn) 的排序算法。
- **Top-K 问题**：找出第 kkk 大/小的元素。
- **调度问题**：任务优先级管理。

---

## **二、具体例子**

假设有以下数组 `Array = [1, 4, 6, 8]`，构建**大根堆**：

- 初始数组：`[1, 4, 6, 8]`。
- 构建大根堆的步骤：
    1. **上浮操作**：`8` 上浮到根节点。
    2. **结果**：`[8, 4, 6, 1]`。

---

## **三、Python 实作**

### **使用 `heapq` 实现小根堆**

Python 提供了标准库 `heapq`，实现了**小根堆**的功能。
```python
import heapq

# 初始化一个小根堆
heap = []
heapq.heappush(heap, 4)
heapq.heappush(heap, 1)
heapq.heappush(heap, 6)
heapq.heappush(heap, 8)

print("堆的内容:", heap)  # 输出: [1, 4, 6, 8]（自动维护小根堆）

# 弹出最小值
print("最小值:", heapq.heappop(heap))  # 输出: 1

# 查看堆顶（最小值）
print("堆顶元素:", heap[0])  # 输出: 4

```

---

### **实现大根堆（手动反转元素值）**

Python 默认是小根堆，可以通过将元素取反来实现大根堆。
```python
import heapq

# 大根堆：取负值存入堆
max_heap = []
heapq.heappush(max_heap, -4)
heapq.heappush(max_heap, -1)
heapq.heappush(max_heap, -6)
heapq.heappush(max_heap, -8)

print("大根堆内容:", [-x for x in max_heap])  # 输出: [8, 4, 6, 1]

# 弹出最大值
print("最大值:", -heapq.heappop(max_heap))  # 输出: 8

```

---

## **四、LeetCode 使用堆的题目描述及解法**

以下是整理的 LintCode 中涉及堆（Heap）的题目表格，包括题目编号、题目名称（英文）、题目简述（中文）、最简单样例及解法：

| 题目编号                | 题目名称（英文）                                                | 题目简述（中文）                                                                                                        | 最简单样例                                                                                      | 解法                                         |
| ------------------- | ------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | ------------------------------------------ |
| 4<br>*<br>(m)       | [[Ugly Number II]] 丑数 II                                | 找到第 n 个丑数，丑数是仅由 2, 3 和 5 的质因子构成的正整数。                                                                            | 输入:  <br>n = 10  <br>输出:  <br>12                                                           | 使用最小堆生成丑数，动态记录已生成的数以避免重复计算。                |
| 130<br><br>(m)      | [[Heapify]] 堆化                                          | 给定一个未排序的数组，将其调整为堆（最小堆或最大堆）。                                                                                     | 输入:  <br>nums = [3,2,1,4,5]  <br>输出:  <br>[1,2,3,4,5]                                      | 使用下沉操作构建堆，从最后一个非叶节点向上调整堆。                  |
| 401<br><br>(m)      | [[Kth Smallest Number in Sorted Matrix]] 排序矩阵中的从小到大第k个数 | 找到排序矩阵中第 k 小的数。                                                                                                 | 输入:  <br>matrix = [[1,5,9],[10,11,13],[12,13,15]]  <br>k = 8  <br>输出:  <br>13              | 使用最小堆存储矩阵每行的首元素，动态更新并查找第 k 小值。             |
| 612<br>*<br>(m)     | [[K Closest Points]] K个最近的点                             | 找到离原点最近的 k 个点。                                                                                                  | 输入:  <br>points = [[1,3],[-2,2],[2,-2]]  <br>k = 2  <br>输出:  <br>[[-2,2],[2,-2]]           | 使用最大堆存储 k 个最近的点，动态比较并维护堆的大小。               |
| 839<br><br>(e)      | [[Merge Two Sorted Interval Lists]] 合并两个排序的间隔列表         | 合并两个排序的区间列表，返回一个合并后的列表。                                                                                         | 输入:  <br>list1 = [[1,2],[3,4]]  <br>list2 = [[2,3],[5,6]]  <br>输出:  <br>[[1,4],[5,6]]      | 使用最小堆维护区间的起点，逐步合并重叠区间。                     |
| 857<br><br>(h)      | [[Minimum Window Subsequence]]最小的窗口子序列                  | 找到目标字符串中覆盖源字符串的最短子序列。                                                                                           | 输入:  <br>S = "abcdebdde"  <br>T = "bde"  <br>输出:  <br>"bcde"                               | 使用堆记录窗口的起始索引，动态更新最短窗口。                     |
| 1046<br><br>(e)     | [[Minimize Max Distance to Gas Station]] 二进制表示中质数个计算置位  | 在道路上新增加油站以最小化最大距离，返回最小的最大距离。                                                                                    | 输入:  <br>stations = [1,2,3,4,5]  <br>k = 4  <br>输出:  <br>0.5                               | 使用最大堆模拟新增加油站的过程，逐步缩小最大距离。                  |
| 1057<br><br>(m)     | [[Network Delay Time]] 网络延迟时间                           | 有 N个网络节点，一个旅行时间和有向边列表 times[i] = (u, v, w)，其中u 是起始点， v是目标点， w是一个信号从起始到目标点花费的时间。 从一个特定节点 K发出信号，所有节点收到信号需要花费多长时间? | 输入: times = <br>[ [2,1,1],[2,3,1],<br>[3,4,1] ], <br>N = 4, K = 2<br>输出:  2                | Dijkstra 求最短路                              |
| 3661<br><br>(m)     | [[Missing Element in Sorted Array]] 有序数组中的缺失元素          | 找到排序数组中缺失的第 k 个数。                                                                                               | 输入:  <br>nums = [4,7,9,10]  <br>k = 3  <br>输出:  <br>8                                      | 使用堆存储缺失的元素数目，动态计算目标值。                      |
| 3666<br><br>(h)     | [[Campus Bikes II ]]校园自行车分配（二）                          | 将工人和自行车分配，返回分配的最小总距离。                                                                                           | 输入:  <br>workers = [[0,0],[1,1],[2,0]]  <br>bikes = [[1,0],[2,2],[2,1]]  <br>输出:  <br>4    | 使用最小堆枚举所有分配方案，按最小距离进行选择。                   |
| 1418<br><br>(m)     | [[Path With Maximum Minimum Value]]具有最大最小值的路径           | 找到二维网格中最大最小路径值，从左上角到右下角。                                                                                        | 输入:  <br>grid = [[5,4,5],[1,2,6],[7,4,6]]  <br>输出:  <br>4                                  | 使用最大堆记录路径中的最小值，动态更新网格状态。                   |
| 3707<br><br>(m)     | [[Corporate Flight Bookings]] 统计航班预订信息                  | 给定航班预订记录，计算每个航班的预订总数。                                                                                           | 输入:  <br>bookings = [[1,2,10],[2,3,20],[2,5,25]]  <br>n = 5  <br>输出:  <br>[10,55,45,25,25] | 使用差分数组和堆动态更新每个航班的预订数目。                     |
| 1872<br><br>(m)     | [[Minimum Cost to Connect Sticks]] 连接棒材的最低费用            | 计算将所有木棍连接成一根木棍的最小成本。                                                                                            | 输入:  <br>sticks = [2,4,3]  <br>输出:  <br>14                                                 | 使用最小堆合并长度最小的木棍，动态计算总成本。                    |
| 81<br>**<br>(h)     | 寻找数据流的中位数 [[Find Median from Data Stream]]              | 实现一个数据结构，支持从数据流中高效找到中位数。                                                                                        | 输入:  <br>addNum(1), addNum(2), findMedian(), addNum(3), findMedian()  <br>输出: [1.5, 2]     | 使用两个堆（最大堆和最小堆）分别存储较小和较大的元素，中位数取决于堆的大小或堆顶值。 |
| 544<br>*<br>(m)<br> | 前K大数 [[Top k Largest Numbers]]                          | 给定一个数组和整数k，找出数组中前k大的数。                                                                                          | 输入:  <br>nums = [3,2,1,5,6,4], k = 2  <br>输出: [5,6]                                        | 使用最小堆存储前k个数，遍历数组动态维护堆中最大的k个数。              |
| 545<br>*<br>(m)     | 前K大数 II [[Top k Largest Numbers II]]                    | 实现一个数据结构支持动态插入数字并能高效返回前k大的数。                                                                                    | 输入:  <br>add(3), add(10), topk()  <br>输出: [10,3]                                           | 使用最小堆维护前k大的数，每次插入新数字时更新堆中的值。               |
| 1512<br>*<br>(h)    | 雇佣K个人的最低费用 [[Minimum Cost to Hire K Workers]]           | 给定一组工人的工资和质量，找到雇佣k个工人的最低费用。                                                                                     | 输入:  <br>quality = [10,20,5], wage = [70,50,30], k = 2  <br>输出: 105                        | 使用最小堆存储工人的质量比，动态维护前k个工人组合的最低工资。            |
| 577<br>*<br>(m)     | 合并K个排序间隔列表 [[Merge K Sorted Interval Lists]]            | 合并k个排序的区间列表，返回一个合并后的区间列表。                                                                                       | 输入:  <br>intervals = [[[1,3],[5,7]],[[2,4],[6,8]],[[9,10]]]  <br>输出: [[1,4],[5,8],[9,10]]  | 使用最小堆按区间起点排序，动态合并重叠的区间。                    |
| 919<br>*<br>(m)     | 会议室 II [[Meeting Rooms II]]                             | 给定一组会议时间，找出需要的最小会议室数量。                                                                                          | 输入:  <br>intervals = [[0,30],[5,10],[15,20]]  <br>输出: 2                                    | 使用最小堆存储当前会议的结束时间，动态分配会议室数量。                |

## **五、选取三题详细解释**

---

### **1. LeetCode 215: 数组中的第 K 大元素（Kth Largest Element in an Array）**

**题目描述**：  
给定一个数组，找出其中第 kkk 大的元素。

**解法思路**：

- 使用**小根堆**，维护前 kkk 大的元素。
- 遍历数组，每次将元素加入堆中，当堆的大小超过 kkk，弹出堆顶（最小值）。

**Python 代码**：
```python
import heapq

def findKthLargest(nums, k):
    min_heap = []
    for num in nums:
        heapq.heappush(min_heap, num)
        if len(min_heap) > k:
            heapq.heappop(min_heap)
    return min_heap[0]

# 示例
nums = [3, 2, 1, 5, 6, 4]
k = 2
print(findKthLargest(nums, k))  # 输出: 5

```

**时间复杂度**：

- O(nlog⁡k)O(n \log k)O(nlogk)：遍历所有元素，每次操作堆的时间复杂度为 log⁡k\log klogk。

**为什么使用堆**：

- 堆能够高效维护前 kkk 大的元素，时间复杂度比排序 O(nlog⁡n)O(n \log n)O(nlogn) 更优。

---

### **2. LeetCode 347: 前 K 个高频元素（Top K Frequent Elements）**

**题目描述**：  
返回数组中出现频率最高的 kkk 个元素。

**解法思路**：

- 使用 **哈希表** 统计每个元素的出现次数。
- 使用**小根堆**，维护频率最高的 kkk 个元素。

**Python 代码**：
```python
import heapq
from collections import Counter

def topKFrequent(nums, k):
    count = Counter(nums)
    min_heap = []

    for num, freq in count.items():
        heapq.heappush(min_heap, (freq, num))
        if len(min_heap) > k:
            heapq.heappop(min_heap)
    
    return [num for freq, num in min_heap]

# 示例
nums = [1, 1, 1, 2, 2, 3]
k = 2
print(topKFrequent(nums, k))  # 输出: [1, 2]

```

**时间复杂度**：

- O(nlog⁡k)O(n \log k)O(nlogk)：遍历元素并维护堆的大小为 kkk。

**为什么使用堆**：

- 堆可以高效维护频率最高的 kkk 个元素。

---

### **3. LeetCode 295: 数据流的中位数（Find Median from Data Stream）**

**题目描述**：  
设计一个数据结构，能够动态维护数据流的中位数。

**解法思路**：

- 使用 **大根堆** 和 **小根堆**：
    - 大根堆存储较小的一半元素。
    - 小根堆存储较大的一半元素。
    - 两堆大小平衡，保证中位数位于堆顶。

**Python 代码**：
```python
import heapq

class MedianFinder:
    def __init__(self):
        self.small = []  # 大根堆（存负值）
        self.large = []  # 小根堆

    def addNum(self, num: int) -> None:
        heapq.heappush(self.small, -num)
        heapq.heappush(self.large, -heapq.heappop(self.small))
        
        if len(self.small) < len(self.large):
            heapq.heappush(self.small, -heapq.heappop(self.large))

    def findMedian(self) -> float:
        if len(self.small) == len(self.large):
            return (-self.small[0] + self.large[0]) / 2.0
        return -self.small[0]

# 示例
mf = MedianFinder()
mf.addNum(1)
mf.addNum(2)
print(mf.findMedian())  # 输出: 1.5
mf.addNum(3)
print(mf.findMedian())  # 输出: 2

```

**时间复杂度**：

- O(log⁡n)O(\log n)O(logn)：插入元素的时间复杂度。

**为什么使用堆**：

- 堆能够动态维护数据流中较小和较大的部分，快速找到中位数。

---

## **总结**

- **堆（Heap）** 是一种适合处理**动态最值**、**Top-K 问题** 和 **优先级调度** 的数据结构。
- Python 的 `heapq` 提供了高效的最小堆实现，大根堆可以通过取负值来模拟。
- 在 LeetCode 中，堆的应用广泛，能够高效解决排序、优先级管理等问题。


# **<mark style="background: #ADCCFFA6;">字典树（Trie）</mark>的详细介绍**
###### 字典树
---

## **一、字典树的原理与特点**

### **1. 原理**

- **字典树（Trie）** 是一种用于存储**字符串集合**的树形数据结构。
- 每个节点表示字符串中的一个字符，从根节点到某个节点的路径可以表示一个前缀或完整的单词。
- 常用于处理**字符串前缀匹配**问题，如搜索引擎自动补全、单词检索等。

---

### **2. 特点**

1. **节点结构**：
    
    - 每个节点包含：
        - 子节点映射（通常是字典，用来指向下一个字符节点）。
        - 是否为单词结束的标志（`is_end`）。
2. **核心操作**：
    
    - **插入（Insert）**：将字符串逐字符插入树中。
    - **搜索（Search）**：检查树中是否存在指定字符串。
    - **前缀匹配（Prefix Search）**：判断是否存在指定前缀的字符串。
3. **时间复杂度**：
    
    - **插入**：O(m)O(m)O(m)，其中 mmm 是字符串的长度。
    - **搜索**：O(m)O(m)O(m)。
    - **前缀匹配**：O(m)O(m)O(m)。
4. **空间复杂度**：
    
    - 最坏情况下，存储 nnn 个长度为 mmm 的字符串需要 O(n×m)O(n \times m)O(n×m) 空间。

---

### **3. 适用场景**

- **自动补全**：快速查找以某个前缀开头的单词集合。
- **拼写检查**：检查字符串是否存在于字典中。
- **字符串集合**：存储和搜索大量字符串。
- **单词搜索和词频统计**。

---

## **二、具体例子**

假设有单词集合：`["apple", "app", "bat"]`，使用 Trie 构建如下图：
```css
        root
       /    \
      a      b
      |       \
      p        a
      |         \
      p          t
     / \
    l   (is_end)
    |
    e
   (is_end)

```

- 插入单词 "apple" 和 "app"：
    - **"app"** 标记为结束。
    - **"apple"** 继续向下插入。
- 插入单词 "bat"：从根节点开始，沿路径插入。

---

## **三、Python 实作**

### **Trie 的实现**
```python
class TrieNode:
    def __init__(self):
        self.children = {}  # 子节点字典
        self.is_end = False  # 是否是单词结束标志

class Trie:
    def __init__(self):
        self.root = TrieNode()

    # 插入单词
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    # 查找单词
    def search(self, word):
        node = self._searchPrefix(word)
        return node is not None and node.is_end

    # 查找前缀
    def startsWith(self, prefix):
        return self._searchPrefix(prefix) is not None

    # 辅助函数：查找前缀
    def _searchPrefix(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node

# 示例
trie = Trie()
trie.insert("apple")
print(trie.search("apple"))  # 输出: True
print(trie.search("app"))    # 输出: False
print(trie.startsWith("app"))  # 输出: True
trie.insert("app")
print(trie.search("app"))    # 输出: True

```

---

## **四、LeetCode 使用字典树的题目描述及解法**

以下是整理的 LintCode 中涉及字典树（Trie）的题目表格，包括题目编号、题目名称（英文）、题目简述（中文）、最简单样例及解法：

| 题目编号             | 题目名称（英文）                                                 | 题目简述（中文）                           | 最简单样例                                                                                                                                         | 解法                            |
| ---------------- | -------------------------------------------------------- | ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- |
| 442<br>**<br>(m) | [[Implement Trie (Prefix Tree)]] 实现前缀树                   | 实现一个字典树（前缀树），支持插入、查找和判断前缀操作。       | 输入: <br>  insert("lintcode")<br>  search("lint")<br>  startsWith("lint")<br>输出: <br>  false<br>  true                                         | 使用嵌套字典构造前缀树，递归或迭代实现插入和查询操作。   |
| 473<br>**<br>(m) | [[Add and Search Word]] 单词的添加与查找                         | 实现一个单词搜索数据结构，支持通配符查询（“.”表示任意字符）。   | 输入:<br>  addWord("a")<br>  search(".")<br>输出: <br>  true                                                                                      | 使用字典树构建单词结构，递归处理通配符匹配。        |
| 634<br>*<br>(h)  | [[Word Squares]] 单词矩阵                                    | 找到所有可以构成单词方阵的单词集合。                 | 输入:<br>["abat","baba",<br>"atan","atal"]<br>输出:<br> [ ["baba","abat",<br>"baba","atan"],<br>["baba","abat",<br>"baba","atal"] ]]]             | 使用字典树和回溯法生成所有可能的单词组合。         |
| 1090<br><br>(m)  | [[Map Sum Pairs]] 映射配对之和                                 | 实现一个键值映射数据结构，支持以任意前缀开头的键值和查询。      | 输入: insert("apple", 3), <br>输出: Null<br>输入: sum("ap"), <br>输出: 3                                                                              | 使用字典树记录每个节点的累积和，动态更新结果。       |
| 1110<br><br>(m)  | [[Replace Words]] 单词替换                                   | 使用词典中的单词替换输入句子中的单词，替换为最短前缀。        | 输入:  <br>dictionary = ["cat","bat","rat"],  <br>sentence = "the cattle was rattled by the battery"  <br>输出:  <br>"the cat was rat by the bat" | 使用字典树存储词典，逐词匹配替换句子中的单词。       |
| 1071<br><br>(e)  | [[Longest Word in Dictionary ]]词典中最长的单词                  | 找到字典中可以逐步构建的最长单词，多个答案时返回字典序最小的单词。  |                                                                                                                                               | 使用字典树构建词典，按长度和字典序排序返回结果。      |
| 1248<br>*<br>(m) | [[Maximum XOR of Two Numbers in an Array]] 数组中两个数字的最大异或  | 在数组中找到两个数的最大异或值。                   | 输入:  <br>nums = [3,10,5,25,2,8]  <br>输出:  <br>28                                                                                              | 使用字典树存储二进制位，逐位匹配最大异或结果。       |
| 1221<br>*<br>(h) | [[Concatenated Words ]]连接词                               | 找到所有由其他单词拼接而成的单词。                  | 输入: <br>words = <br>["a","b","ab","abc"]<br>输出: ["ab"]                                                                                        | 使用字典树存储单词，结合回溯判断是否能由其他单词拼接而成。 |
| 775<br>*<br>(h)  | [[Palindrome Pairs]] 回文对                                 | 找到数组中所有回文数对。                       | 输入:<br>["bat", "tab", "cat"]<br>输出:<br>[ [0, 1], [1, 0] ]<br>                                                                                 | 使用字典树存储单词及其反转，检查是否能形成回文对。     |
| 333<br>*<br>(m)  | 识别字符串 [[Identifying Strings]]                            | 判断一个字符串是否存在于字典中，并支持动态添加字符串到字典中。    | 输入:  <br>add("apple"), <br>search("apple"), <br>search("app")  <br>输出: true, false                                                            | 使用字典树存储所有字符串，遍历节点检查字符串是否存在。   |
| 1624<br>*<br>(h) | 最大距离 [[Max Distance]]                                    | 给定一个单词列表，找到两个单词之间的最大距离。            | 输入:  <br>words = <br>["practice", <br>"makes", "perfect", <br>"coding", "makes"]  <br>输出: 3                                                   | 使用字典树记录每个单词出现的索引，计算索引差的最大值。   |
| 623<br>*<br>(h)  | K步编辑 [[K Edit Distance]]                                 | 找出所有与给定单词距离不超过K的字典单词。              | 输入:  <br>words = <br>["abc", "abd", "abcd"], <br>target = "abc", k = 1  <br>输出: ["abc", "abd"]                                                | 使用字典树结合动态规划计算编辑距离，返回符合条件的单词。  |
| 635<br>*<br>(h)  | 拼字游戏 [[Boggle Game]]                                     | 给定一个字母矩阵和单词列表，找出矩阵中所有可以由相邻字母组成的单词。 | 输入:  <br>board = [<br>["b","o","g"],<br>["l","e","o"],<br>["g","o","d"]], <br>words = <br>["bog","dog"]  <br>输出: ["bog"]                      | 使用字典树存储单词，结合深度优先搜索查找矩阵中的有效路径。 |
| 270<br>*<br>(m)  | 电话号码的字母组合II [[Letter Combinations of a Phone Number II]] | 给定一个数字字符串，返回所有可能的字母组合。             | 输入:  <br>digits = "23"  <br>输出: <br>["ad","ae","af",<br>"bd","be","bf",<br>"cd","ce","cf"]                                                    | 使用字典树存储数字到字母的映射，结合回溯生成所有可能组合。 |
| 722<br>*<br>(h)  | 最大子数组VI [[Maximum Subarray VI]]                          | 找出数组中和最大的子数组，同时满足每个元素在某个字典的子集范围内。  | 输入:  <br>nums = [1,2,3], dict = [1,2]  <br>输出: [1,2]                                                                                          | 使用字典树存储字典范围，动态查找符合条件的子数组。     |


## **五、选取三题详细解释**

---

### **1. LeetCode 208: 实现 Trie（Implement Trie）**

**题目描述**：  
实现一个 Trie，支持以下操作：

1. **insert(word)**：插入单词。
2. **search(word)**：查找单词是否存在。
3. **startsWith(prefix)**：判断是否存在以指定前缀开头的单词。

**解法思路**：

- 使用 TrieNode 类表示节点，包含 `children` 和 `is_end` 属性。
- 在 Trie 类中实现插入、查找和前缀查找操作。

**Python 代码**：
```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def search(self, word):
        node = self._searchPrefix(word)
        return node is not None and node.is_end

    def startsWith(self, prefix):
        return self._searchPrefix(prefix) is not None

    def _searchPrefix(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node

```

**时间复杂度**：

- O(m)O(m)O(m)：mmm 为字符串长度。

**为什么使用 Trie**：

- Trie 可以高效地实现字符串的查找和前缀匹配，适合存储和搜索大量字符串。

---

### **2. LeetCode 212: 单词搜索 II（Word Search II）**

**题目描述**：  
给定一个字符网格和单词列表，找出所有存在于网格中的单词。

**解法思路**：

- 使用 Trie 存储单词列表。
- 使用 DFS 遍历网格，同时在 Trie 中查找前缀是否存在。

**Python 代码**：
```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.word = None

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.word = word

def findWords(board, words):
    def dfs(node, r, c):
        char = board[r][c]
        if char not in node.children:
            return
        node = node.children[char]
        if node.word:
            result.add(node.word)
        
        board[r][c] = "#"
        for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(board) and 0 <= nc < len(board[0]):
                dfs(node, nr, nc)
        board[r][c] = char

    trie = Trie()
    for word in words:
        trie.insert(word)

    result = set()
    for r in range(len(board)):
        for c in range(len(board[0])):
            dfs(trie.root, r, c)
    
    return list(result)

```

**时间复杂度**：

- O(m⋅n⋅l)O(m \cdot n \cdot l)O(m⋅n⋅l)：网格大小为 m×nm \times nm×n，单词平均长度为 lll。

---

### **3. LeetCode 648: 替换单词（Replace Words）**

**题目描述**：  
使用字典中的词根替换句子中的单词，如果存在多个词根，以最短的词根为准。

**解法思路**：

- 使用 Trie 存储词根。
- 遍历句子中的每个单词，在 Trie 中查找最短的前缀。

**Python 代码**：
```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def findRoot(self, word):
        node = self.root
        prefix = ""
        for char in word:
            if char not in node.children:
                break
            node = node.children[char]
            prefix += char
            if node.is_end:
                return prefix
        return word

def replaceWords(dictionary, sentence):
    trie = Trie()
    for root in dictionary:
        trie.insert(root)
    
    words = sentence.split()
    return " ".join([trie.findRoot(word) for word in words])

# 示例
dictionary = ["cat", "bat", "rat"]
sentence = "the cattle was rattled by the battery"
print(replaceWords(dictionary, sentence))  # 输出: "the cat was rat by the bat"

```

---

## **总结**

- **字典树（Trie）** 适用于解决**字符串前缀**和**字符串集合**相关的问题。
- Trie 提供了高效的插入、查找和前缀匹配操作，时间复杂度为 O(m)O(m)O(m)。
- 在 LeetCode 中，Trie 的应用场景丰富，如自动补全、单词搜索、字符串替换等问题。


# **<mark style="background: #ADCCFFA6;">线段树（Segment Tree）</mark>的详细介绍**
###### 线段树
---

## **一、线段树的原理与特点**

### **1. 原理**

- **线段树（Segment Tree）** 是一种**二叉树结构**，主要用于高效处理**区间查询**和**区间更新**的问题。
- 通过将数组分为若干个子区间，并将子区间的结果存储在树的节点中，线段树能够快速地对区间进行求和、最小值、最大值等操作。

### **2. 线段树的性质**

- **结构**：
    - 每个节点表示一个区间 `[l, r]`，根节点表示整个数组的区间 `[0, n-1]`。
    - 叶子节点表示数组中的单个元素。
    - 父节点存储的值是其两个子节点值的聚合（如求和或最小值）。
- **区间划分**：
    - 左子节点表示区间 `[l, mid]`，右子节点表示区间 `[mid+1, r]`，其中 mid=(l+r)//2mid = (l + r) // 2mid=(l+r)//2。
- **操作**：
    - **单点更新**：修改数组中的某个元素，并更新相关区间的结果。
    - **区间查询**：查询数组中某个区间的聚合结果。

---

### **3. 复杂度分析**

- **构建线段树**：O(n)O(n)O(n)。
- **区间查询**：O(log⁡n)O(\log n)O(logn)。
- **单点更新**：O(log⁡n)O(\log n)O(logn)。

---

### **4. 适用场景**

- **区间求和**、**区间最值**、**区间乘积**等聚合操作。
- **动态更新**：支持数组元素的修改和查询操作。
- 适合用于处理大量区间查询和更新的问题。

---

## **二、具体例子**

假设有一个数组 `Array = [1, 4, 6, 8]`，我们使用线段树进行以下操作：

1. **构建线段树**：
    - 根节点表示整个数组 `[0, 3]`。
    - 左子树表示 `[0, 1]`，右子树表示 `[2, 3]`。
    - 叶子节点分别表示 `1, 4, 6, 8`。
2. **区间求和**：
    - 查询 `[1, 3]` 的和，即 `4 + 6 + 8 = 18`。
3. **单点更新**：
    - 修改 `Array[2] = 10`，更新线段树。

---

## **三、Python 实作**

### **线段树实现**

以下代码实现线段树的构建、单点更新和区间求和：
```python
class SegmentTree:
    def __init__(self, nums):
        self.n = len(nums)
        self.tree = [0] * (4 * self.n)  # 线段树存储数组
        self.build(0, 0, self.n - 1, nums)

    # 构建线段树
    def build(self, node, start, end, nums):
        if start == end:  # 叶子节点
            self.tree[node] = nums[start]
        else:
            mid = (start + end) // 2
            left_child = 2 * node + 1
            right_child = 2 * node + 2

            self.build(left_child, start, mid, nums)
            self.build(right_child, mid + 1, end, nums)

            self.tree[node] = self.tree[left_child] + self.tree[right_child]

    # 区间求和
    def query(self, node, start, end, l, r):
        if r < start or l > end:  # 查询区间不在当前节点区间内
            return 0
        if l <= start and r >= end:  # 当前节点区间完全包含查询区间
            return self.tree[node]

        mid = (start + end) // 2
        left_sum = self.query(2 * node + 1, start, mid, l, r)
        right_sum = self.query(2 * node + 2, mid + 1, end, l, r)
        return left_sum + right_sum

    # 单点更新
    def update(self, node, start, end, idx, val):
        if start == end:  # 找到要更新的叶子节点
            self.tree[node] = val
        else:
            mid = (start + end) // 2
            if idx <= mid:
                self.update(2 * node + 1, start, mid, idx, val)
            else:
                self.update(2 * node + 2, mid + 1, end, idx, val)
            self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]

# 示例
nums = [1, 4, 6, 8]
st = SegmentTree(nums)

# 查询区间和 [1, 3]
print(st.query(0, 0, len(nums) - 1, 1, 3))  # 输出: 18

# 更新 nums[2] = 10
st.update(0, 0, len(nums) - 1, 2, 10)
print(st.query(0, 0, len(nums) - 1, 1, 3))  # 输出: 22

```

---

## **四、LeetCode 使用线段树的题目描述及解法**

以下是整理的 LintCode 中涉及线段树（Segment Tree）的题目表格，包括题目编号、题目名称（英文）、题目简述（中文）、最简单样例及解法：

| 题目编号            | 题目名称（英文）                                             | 题目简述（中文）                      | 最简单样例                                                                      | 解法                               |
| --------------- | ---------------------------------------------------- | ----------------------------- | -------------------------------------------------------------------------- | -------------------------------- |
| 201<br><br>(m)  | [[Leetcode - 10.Advanced/Segment Tree Build]] 线段树的构造 | 构建一个线段树，支持范围查询和修改操作。          | 输入:  <br>nums = [3,2,1,4]  <br>输出:  <br>线段树的根节点                            | 使用递归分治构建线段树，每个节点存储区间值。           |
| 202<br><br>(m)  | [[Segment Tree Query]] 线段树的查询                        | 实现线段树的查询操作，返回指定范围内的最小值。       | 输入:  <br>nums = [3,2,1,4]  <br>query(1,3)  <br>输出:  <br>1                  | 使用线段树的递归方法，逐层访问区间并返回最小值。         |
| 203<br><br>(m)  | [[Segment Tree Modify]]线段树的修改                        | 修改线段树中一个元素的值，并动态更新相关节点。       | 输入:  <br>nums = [3,2,1,4]  <br>modify(2,5)  <br>输出:  <br>线段树更新后的状态         | 修改节点值后递归更新父节点的区间值。               |
| 247<br><br>(m)  | [[Segment Tree Query II]] 线段树查询 II                   | 查询线段树中指定范围内的最大值。              | 输入:  <br>nums = [1,3,5,7]  <br>query(1,3)  <br>输出:  <br>7                  | 使用线段树的递归方法逐层检查并返回最大值。            |
| 206<br>*<br>(m) | [[Interval Sum]] 区间求和 I                              | 计算数组中多个区间的和。                  | 输入:  <br>nums = [1,2,3,4]  <br>queries = [[1,2],[2,4]]  <br>输出:  <br>[3,9] | 使用线段树构建区间和模型，支持高效查询。             |
| 205<br><br>(m)  | [[Interval Minimum Number]] 区间最小数                    | 找到数组中多个区间的最小值。                | 输入:  <br>nums = [4,3,2,1]  <br>queries = [[1,2],[2,4]]  <br>输出:  <br>[3,1] | 使用线段树构建区间最小值模型，支持高效查询。           |
| 212<br>*<br>(e) | [[Space Replacement ]]空格替换                           | 将字符串中的空格替换为 `%20`。            | 输入:  <br>s = "Mr John Smith"  <br>输出:  <br>"Mr%20John%20Smith"             | 使用双指针从后向前遍历字符串，替换空格。             |
| 439<br><br>(m)  | [[Segment Tree Build II ]]线段树的构造 II                  | 构建一个线段树，每个节点存储区间内的最小值和最大值。    | 输入:  <br>nums = [1,4,2,3]  <br>输出:  <br>线段树根节点                             | 使用递归构建节点，动态存储区间值。                |
| 217<br><br>(e)  | [[Remove Duplicates from Unsorted List]] 无序链表的重复项删除  | 删除未排序链表中的重复节点。                | 输入:  <br>head = [1,2,2,3]  <br>输出:  <br>[1,2,3]                            | 使用哈希表记录已访问节点值，过滤重复节点。            |
| 207<br>*<br>(h) | 区间求和II [[Interval Sum II]]                           | 在区间求和的基础上，增加更新操作，支持修改数组中的元素值。 | 输入:  <br>nums = [1,2,3], update(1,5), sum(0,2)  <br>输出: [9]                | 构建线段树存储区间和，更新时动态调整树结构，查询时递归分段求和。 |


## **五、选取三题详细解释**

---

### **1. LeetCode 307: 可变区间和（Range Sum Query - Mutable）**

**题目描述**：  
给定一个数组，支持以下操作：

1. **update(index, val)**：将指定位置的元素更新为 `val`。
2. **sumRange(left, right)**：返回区间 `[left, right]` 的和。

**解法思路**：

- 使用线段树存储区间和。
- **更新**：修改单个节点，并向上传播更新区间和。
- **查询**：通过线段树快速计算指定区间的和。

**Python 代码**：
```python
class NumArray:
    def __init__(self, nums):
        self.st = SegmentTree(nums)

    def update(self, index: int, val: int) -> None:
        self.st.update(0, 0, len(nums) - 1, index, val)

    def sumRange(self, left: int, right: int) -> int:
        return self.st.query(0, 0, len(nums) - 1, left, right)

```

**时间复杂度**：

- **更新**：O(log⁡n)O(\log n)O(logn)。
- **查询**：O(log⁡n)O(\log n)O(logn)。

**为什么使用线段树**：

- 线段树支持快速的区间和查询和动态更新操作，性能优于暴力方法。

---

### **2. LeetCode 303: 不可变区间和（Range Sum Query - Immutable）**

**题目描述**：  
查询数组中指定区间的和，数组不可修改。

**解法思路**：

- 使用前缀和数组，避免重复计算区间和。

**Python 代码**：
```python
class NumArray:
    def __init__(self, nums):
        self.prefix_sum = [0]
        for num in nums:
            self.prefix_sum.append(self.prefix_sum[-1] + num)

    def sumRange(self, left: int, right: int) -> int:
        return self.prefix_sum[right + 1] - self.prefix_sum[left]

```

---

### **3. LeetCode 315: 计算右侧小于当前元素的数量**

**题目描述**：  
给定一个数组，返回每个元素右侧小于当前元素的数量。

**解法思路**：

- 使用线段树维护元素出现次数，动态计算右侧小于当前元素的数量。

---

## **总结**

- **线段树（Segment Tree）** 是处理**区间查询**和**动态更新**问题的高效数据结构，时间复杂度为 O(log⁡n)O(\log n)O(logn)。
- 在 LeetCode 中，线段树常用于解决区间和、区间最值和二维矩阵问题。
- 学习线段树的关键在于理解其**树形结构**和**分治思想**。


# **<mark style="background: #ADCCFFA6;">平衡二叉树（Balanced Binary Tree）</mark>的详细介绍**
###### 平衡二叉树
---

## **一、平衡二叉树的原理与特点**

### **1. 原理**

- **平衡二叉树**是一种二叉树，其特点是：**任意节点的左右子树高度差不超过1**。
- 平衡二叉树的结构能够有效避免二叉树退化成链表，从而保证较低的高度，使查找、插入和删除操作维持较高的效率。
- 常见的平衡二叉树类型包括：
    1. **AVL树**：严格的高度平衡二叉搜索树，每次插入或删除节点后，通过旋转操作维持平衡。
    2. **红黑树**：一种近似平衡的二叉搜索树，允许局部不平衡，通过颜色标记和平衡规则实现高效操作。

---

### **2. 平衡二叉树的性质**

- **平衡性**：左右子树高度差绝对值不超过1。
- **时间复杂度**：
    - 查找操作：O(log⁡n)O(\log n)O(logn)。
    - 插入/删除操作：O(log⁡n)O(\log n)O(logn)。
- **空间复杂度**：O(n)O(n)O(n)。

---

### **3. 平衡二叉树的优势**

- 平衡二叉树可以避免普通二叉搜索树退化成链表，保证所有操作的时间复杂度接近 O(log⁡n)O(\log n)O(logn)。
- 在搜索、插入和删除操作中性能稳定。

---

## **二、具体例子**

给定以下元素：`[1, 4, 6, 8, 10, 12]`  
构建一棵平衡二叉树如下：
```markdown
        6
       / \
      4   10
     /    / \
    1    8  12

```

- **平衡性**：任意节点的左右子树高度差不超过1。
- **搜索**：查找 `8`，从根节点 `6` 开始，向右移动即可找到。
- **插入**：插入新节点时，通过旋转操作（如左旋、右旋）维持平衡。

---

## **三、Python 实作**

### **AVL 树的实现**

以下是实现平衡二叉树（AVL 树）的基本操作，包括插入节点和保持平衡：

```python
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.height = 1  # 初始高度为1

class AVLTree:
    # 获取节点的高度
    def getHeight(self, node):
        return node.height if node else 0

    # 计算平衡因子
    def getBalance(self, node):
        return self.getHeight(node.left) - self.getHeight(node.right) if node else 0

    # 左旋
    def leftRotate(self, z):
        y = z.right
        T2 = y.left

        y.left = z
        z.right = T2

        z.height = 1 + max(self.getHeight(z.left), self.getHeight(z.right))
        y.height = 1 + max(self.getHeight(y.left), self.getHeight(y.right))

        return y

    # 右旋
    def rightRotate(self, z):
        y = z.left
        T3 = y.right

        y.right = z
        z.left = T3

        z.height = 1 + max(self.getHeight(z.left), self.getHeight(z.right))
        y.height = 1 + max(self.getHeight(y.left), self.getHeight(y.right))

        return y

    # 插入节点并保持平衡
    def insert(self, root, val):
        if not root:
            return TreeNode(val)

        if val < root.val:
            root.left = self.insert(root.left, val)
        elif val > root.val:
            root.right = self.insert(root.right, val)
        else:
            return root  # 重复值不插入

        root.height = 1 + max(self.getHeight(root.left), self.getHeight(root.right))
        balance = self.getBalance(root)

        # 左左情况
        if balance > 1 and val < root.left.val:
            return self.rightRotate(root)

        # 右右情况
        if balance < -1 and val > root.right.val:
            return self.leftRotate(root)

        # 左右情况
        if balance > 1 and val > root.left.val:
            root.left = self.leftRotate(root.left)
            return self.rightRotate(root)

        # 右左情况
        if balance < -1 and val < root.right.val:
            root.right = self.rightRotate(root.right)
            return self.leftRotate(root)

        return root

# 示例
avl = AVLTree()
root = None
values = [10, 20, 30, 25, 28, 27, 5]
for val in values:
    root = avl.insert(root, val)

print("根节点:", root.val)  # 输出平衡后的根节点

```

---

## **四、LeetCode 使用平衡二叉树的题目描述及解法**

以下是整理的平衡二叉树相关题目表格，包含题目编号、题目名称、题目简述、样例和解法五个栏位：

| **题目编号**        | **题目名称 (英文/中文)**               | **题目简述 (中文)**                        | **样例**                                                     | **解法**                                     |
| --------------- | ------------------------------ | ------------------------------------ | ---------------------------------------------------------- | ------------------------------------------ |
| 93<br>*<br>(e)  | 平衡二叉树 [[Balanced Binary Tree]] | 判断一棵二叉树是否为平衡二叉树（每个节点的左右子树高度差不超过 1）。  | 输入:  <br>root = [3,9,20,null,null,15,7]  <br>输出: true      | 使用递归计算每个节点的高度，检查左右子树的高度差，若所有节点满足条件则为平衡二叉树。 |
| 1513<br><br>(m) | 考场就座 [[Exam Room]]             | 实现一个考场就座系统，支持学生尽可能坐到离他人最远的位置，支持离开座位。 | 输入:  <br>seat(), seat(), leave(0), seat()  <br>输出: [0,9,4] | 使用平衡二叉树存储座位区间，动态更新座位状态，通过区间长度计算最优座位。       |

## **五、选取三题详细解释**

---

### **1. LeetCode 110: 判断平衡二叉树**

**题目描述**：  
给定一个二叉树，判断它是否是一棵平衡二叉树。

**解法思路**：

- 递归计算左右子树的高度，并判断高度差是否满足条件。

**Python 代码**：
```python
def isBalanced(root):
    def height(node):
        if not node:
            return 0
        left = height(node.left)
        right = height(node.right)
        if abs(left - right) > 1:
            raise ValueError  # 提前结束
        return 1 + max(left, right)
    
    try:
        height(root)
        return True
    except ValueError:
        return False

```

**时间复杂度**：

- O(n)O(n)O(n)：遍历所有节点。

---

### **2. LeetCode 1382: 平衡二叉搜索树**

**题目描述**：  
将一棵二叉搜索树重新平衡，使其成为平衡二叉树。

**解法思路**：

- **中序遍历**获取有序元素列表。
- 递归构建平衡二叉搜索树。

**Python 代码**：
```python
def balanceBST(root):
    def inorder(node):
        if not node:
            return []
        return inorder(node.left) + [node.val] + inorder(node.right)

    def buildTree(nums, left, right):
        if left > right:
            return None
        mid = (left + right) // 2
        node = TreeNode(nums[mid])
        node.left = buildTree(nums, left, mid - 1)
        node.right = buildTree(nums, mid + 1, right)
        return node
    
    nums = inorder(root)
    return buildTree(nums, 0, len(nums) - 1)

```

**时间复杂度**：

- O(n)O(n)O(n)：中序遍历 + 重建树。

---

### **3. LeetCode 108: 将有序数组转换为平衡 BST**

**题目描述**：  
将一个有序数组转换为高度平衡的二叉搜索树。

**解法思路**：

- 分治法：选择数组中间的元素作为根节点，递归构建左右子树。

**Python 代码**：
```python
def sortedArrayToBST(nums):
    if not nums:
        return None
    mid = len(nums) // 2
    root = TreeNode(nums[mid])
    root.left = sortedArrayToBST(nums[:mid])
    root.right = sortedArrayToBST(nums[mid+1:])
    return root

```

**时间复杂度**：

- O(n)O(n)O(n)：每个元素访问一次。

---

## **总结**

- **平衡二叉树（Balanced Binary Tree）** 是一种保证高度平衡的二叉树，能够实现高效的查找、插入和删除操作，时间复杂度为 O(log⁡n)O(\log n)O(logn)。
- 在 LeetCode 中，平衡二叉树常用于处理**排序数组构建**、**树的平衡判断** 和 **重建树结构** 等问题。
- 学习平衡二叉树的关键在于理解**高度平衡**的性质及相关的旋转操作（如 AVL 树）。


# **<mark style="background: #ADCCFFA6;">树状数组（Binary Indexed Tree）</mark>的详细介绍**
###### 树状数组
---

## **一、树状数组的原理与特点**

### **1. 原理**

- **树状数组（Binary Indexed Tree，简称 BIT）** 是一种高效的数据结构，用于处理数组的**前缀和**和**单点更新**问题。
- 它通过维护一个额外的数组，将区间和问题转化为多个单点操作，从而将时间复杂度优化为 O(log⁡n)O(\log n)O(logn)。

### **2. 基本概念**

- 树状数组利用**二进制**特性，将数组索引拆分为多个**低位二进制区间**，从而加速前缀和的查询和更新操作。
- 每个节点 `i` 存储的值表示一个子区间的和，这个区间的大小由 `i` 的二进制最低位的 1 决定。
    - 如 i=6i = 6i=6（二进制 110110110）：最低位的 1 表示范围是 2，节点 6 存储的是 `[5, 6]` 的和。

---

### **3. 特点**

- **前缀和查询**：快速计算数组前 iii 项的和。
- **单点更新**：快速更新数组中某个位置的值，并维护相关区间的和。
- **空间复杂度**：O(n)O(n)O(n)。
- **时间复杂度**：
    - **单点更新**：O(log⁡n)O(\log n)O(logn)。
    - **前缀和查询**：O(log⁡n)O(\log n)O(logn)。

### **4. 适用场景**

- 动态处理区间和问题（频繁的查询与更新操作）。
- 排名统计、逆序对问题、区间更新与查询等。

---

## **二、具体例子**

假设有数组 `Array = [1, 4, 6, 8]`，索引从 `1` 开始：

- **原始数组**：`[1, 4, 6, 8]`
- 构建树状数组后的结构：
    - 树状数组 `BIT`：`[0, 1, 5, 6, 19]`
        - BIT[1] = Array[1] = 1
        - BIT[2] = Array[1] + Array[2] = 1 + 4 = 5
        - BIT[3] = Array[3] = 6
        - BIT[4] = Array[1] + Array[2] + Array[3] + Array[4] = 1 + 4 + 6 + 8 = 19

---

## **三、Python 实作**

### **树状数组的实现**

```python
class BinaryIndexedTree:
    def __init__(self, n):
        self.n = n
        self.bit = [0] * (n + 1)  # 树状数组初始化

    # 单点更新：将 idx 位置加上 delta
    def update(self, idx, delta):
        while idx <= self.n:
            self.bit[idx] += delta
            idx += idx & -idx  # 向上更新父节点

    # 前缀和查询：返回前 idx 项的和
    def query(self, idx):
        result = 0
        while idx > 0:
            result += self.bit[idx]
            idx -= idx & -idx  # 向上移动到前一个区间
        return result

    # 查询区间和 [left, right]
    def range_query(self, left, right):
        return self.query(right) - self.query(left - 1)

# 示例
nums = [1, 4, 6, 8]
bit = BinaryIndexedTree(len(nums))

# 初始化树状数组
for i, num in enumerate(nums):
    bit.update(i + 1, num)

# 查询前缀和
print("前 3 项和:", bit.query(3))  # 输出: 11

# 查询区间和 [2, 4]
print("区间 [2, 4] 的和:", bit.range_query(2, 4))  # 输出: 18

# 更新单点：将索引 3 的值加 2
bit.update(3, 2)
print("更新后区间 [2, 4] 的和:", bit.range_query(2, 4))  # 输出: 20

```

---

## **四、LeetCode 使用树状数组的题目描述及解法**

以下是整理的 LintCode 中涉及树状数组（Binary Indexed Tree）的题目表格，包括题目编号、题目名称（英文和中文）、题目简述（中文）、最简单样例及解法：

| **题目编号**             | **题目名称 (英文/中文)**                                       | **题目简述 (中文)**                   | **样例**                                                                                                                          | **解法**                                   |
| -------------------- | ------------------------------------------------------ | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| 249<br>*<br>(h)      | 统计前面比自己小的数的个数 [[Count of Smaller Numbers Before Self]] | 给定一个数组，统计每个数字前面比它小的数字个数，并返回结果。  | 输入:  <br>nums = [5,2,6,1]  <br>输出:  <br>[2,1,1,0]                                                                               | 使用树状数组维护前缀和，动态更新数字的出现次数，通过查询累计小于当前数字的计数。 |
| 817<br>*<br>(m)      | 范围矩阵元素和 - 可变的 [[Range Sum Query 2D - Mutable]]         | 给定一个二维矩阵，支持更新单个元素值和查询任意子矩阵的元素和。 | 输入:  <br>matrix = [[3,0,1,4,2],[5,6,3,2,1],[1,2,0,1,5],[4,1,0,1,7],[1,0,3,0,5]], update(3,2,2), sumRegion(2,1,4,3)  <br>输出: [8] | 使用二维树状数组，动态更新矩阵元素，通过查询累加子矩阵的和。           |
| 840<br>*<br>(m)      | 可变范围求和 [[Range Sum Query - Mutable]]                   | 给定一个数组，支持更新单个元素值和查询任意子数组的元素和。   | 输入:  <br>nums = [1,3,5]  <br>update(1,2)  <br>sumRange(0,2) <br>输出:  <br>8                                                      | 使用树状数组维护前缀和，动态更新元素值，通过查询累加子数组的和。         |
| 1645<br>*<br>(m)<br> | 最少子序列 [[Minimum Subsequence in Non-Increasing Order]]  | 给定一个数组，找到总和大于剩余数字总和的最小递减子序列。    | 输入:  <br>nums = [4,3,10,9,8]  <br>输出: [10,9]                                                                                    | 先对数组排序，使用树状数组维护前缀和，动态查找满足条件的最小子序列。       |


## **五、选取三题详细解释**

---

### **1. LeetCode 307: 可变区间和（Range Sum Query - Mutable）**

**题目描述**：  
支持两种操作：

1. **update(index, val)**：更新数组中某个索引的值。
2. **sumRange(left, right)**：查询区间 `[left, right]` 的和。

**解法思路**：

- 使用树状数组高效维护数组的前缀和。

**Python 代码**：
```python
class NumArray:
    def __init__(self, nums):
        self.n = len(nums)
        self.bit = BinaryIndexedTree(self.n)
        self.nums = nums
        for i, num in enumerate(nums):
            self.bit.update(i + 1, num)

    def update(self, index: int, val: int) -> None:
        delta = val - self.nums[index]
        self.nums[index] = val
        self.bit.update(index + 1, delta)

    def sumRange(self, left: int, right: int) -> int:
        return self.bit.range_query(left + 1, right + 1)

```

**时间复杂度**：

- **更新**：O(log⁡n)O(\log n)O(logn)。
- **查询**：O(log⁡n)O(\log n)O(logn)。

**为什么使用树状数组**：

- 树状数组能够高效处理动态更新和区间和查询。

---

### **2. LeetCode 315: 右侧小于当前元素的数量**

**题目描述**：  
给定一个数组，返回每个元素右侧小于当前元素的数量。

**解法思路**：

- 使用树状数组维护已遍历元素的出现次数，动态查询当前元素右侧小于它的数量。

**Python 代码**：
```python
def countSmaller(nums):
    offset = 10**4  # 偏移量，确保索引为正数
    size = 2 * 10**4 + 1
    bit = BinaryIndexedTree(size)
    result = []
    for num in reversed(nums):
        count = bit.query(num + offset)
        result.append(count)
        bit.update(num + offset + 1, 1)
    return result[::-1]

```

**时间复杂度**：

- O(nlog⁡n)O(n \log n)O(nlogn)。

---

### **3. LeetCode 493: 逆序对（Reverse Pairs）**

**题目描述**：  
统计数组中满足 `nums[i] > 2 * nums[j]`（i<ji < ji<j）的逆序对数量。

**解法思路**：

- 使用树状数组维护已遍历元素，并查询满足条件的元素数量。

---

## **总结**

- **树状数组（BIT）** 是一种用于处理**动态区间和**和**单点更新**的高效数据结构，时间复杂度为 O(log⁡n)O(\log n)O(logn)。
- 在 LeetCode 中，树状数组常用于解决**区间查询**、**逆序对** 和 **动态更新** 等问题，性能优越且实现简单。
- 学习树状数组的关键在于理解索引的二进制表示及其更新规则。


# **<mark style="background: #ADCCFFA6;">圖（Graph）</mark>的詳細介紹**
###### 圖
---

## **一、圖的原理與特點**

### **1. 原理**

- **圖（Graph）** 是由**節點（Node, 或稱頂點 Vertex）** 和**邊（Edge）** 組成的數據結構。
- **圖的分類**：
    - **無向圖**：邊沒有方向，邊 (u,v)(u, v)(u,v) 表示 uuu 和 vvv 相互連接。
    - **有向圖**：邊有方向，邊 (u,v)(u, v)(u,v) 表示從 uuu 到 vvv 的連接。
    - **帶權圖**：每條邊上帶有一個權重（數值），常用於最短路徑等問題。
- **圖的表示方法**：
    1. **鄰接矩陣（Adjacency Matrix）**：使用 n×nn \times nn×n 的二維矩陣存儲邊的連接關係。
    2. **鄰接表（Adjacency List）**：使用一個列表存儲與每個節點相鄰的其他節點。

---

### **2. 特點**

- **靈活性**：圖可以表示任何形式的網絡，如社交網絡、地圖、網絡拓撲等。
- **圖的遍歷**：
    - **深度優先搜索（DFS）**：優先訪問當前節點的鄰居，使用遞歸或棧實現。
    - **廣度優先搜索（BFS）**：逐層訪問節點，使用隊列實現。
- **常見問題**：
    - 最短路徑問題（如 Dijkstra 算法）。
    - 拓撲排序（有向無環圖）。
    - 圖的連通性檢查（聯通分量）。
    - 最小生成樹（如 Kruskal 或 Prim 算法）。

---

### **3. 複雜度**

- **鄰接矩陣**：
    - 空間複雜度：O(n2)O(n^2)O(n2)
    - 添加邊：O(1)O(1)O(1)
    - 查詢邊：O(1)O(1)O(1)
- **鄰接表**：
    - 空間複雜度：O(n+m)O(n + m)O(n+m)（nnn 是節點數，mmm 是邊數）
    - 添加邊：O(1)O(1)O(1)
    - 查詢邊：O(degree of node)O(\text{degree of node})O(degree of node)

---

## **二、具體例子**

給定一個無向圖（鄰接表表示）：
```scss
節點集合：{0, 1, 2, 3}  
邊集合：{(0, 1), (0, 2), (1, 2), (2, 3)}

```

圖的結構如下：
```scss
    0
   / \
  1---2
       \
        3

```

- **鄰接矩陣表示**：
```csharp
[
 [0, 1, 1, 0],  # 節點 0 連接到節點 1 和 2
 [1, 0, 1, 0],  # 節點 1 連接到節點 0 和 2
 [1, 1, 0, 1],  # 節點 2 連接到節點 0, 1 和 3
 [0, 0, 1, 0]   # 節點 3 連接到節點 2
]

```

- **鄰接表表示**：
```yaml
{
  0: [1, 2],
  1: [0, 2],
  2: [0, 1, 3],
  3: [2]
}

```

---

## **三、Python 實作**

### **鄰接表表示圖**

```python
# 圖的鄰接表表示
class Graph:
    def __init__(self):
        self.adj_list = {}

    # 添加邊
    def add_edge(self, u, v):
        if u not in self.adj_list:
            self.adj_list[u] = []
        if v not in self.adj_list:
            self.adj_list[v] = []
        self.adj_list[u].append(v)
        self.adj_list[v].append(u)  # 無向圖

    # 深度優先搜索 (DFS)
    def dfs(self, node, visited):
        if node not in visited:
            print(node, end=" ")
            visited.add(node)
            for neighbor in self.adj_list[node]:
                self.dfs(neighbor, visited)

    # 廣度優先搜索 (BFS)
    def bfs(self, start):
        visited = set()
        queue = [start]
        visited.add(start)

        while queue:
            node = queue.pop(0)
            print(node, end=" ")
            for neighbor in self.adj_list[node]:
                if neighbor not in visited:
                    queue.append(neighbor)
                    visited.add(neighbor)

# 示例
g = Graph()
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 2)
g.add_edge(2, 3)

print("深度優先搜索:")
g.dfs(0, set())  # 輸出: 0 1 2 3

print("\n廣度優先搜索:")
g.bfs(0)         # 輸出: 0 1 2 3

```

---

## **四、LeetCode 使用圖的題目描述及解法**


以下是整理的 LintCode 中涉及图（Graph）的题目表格，包括题目编号、题目名称（英文和中文）、题目简述（中文）、最简单样例及解法：

| 题目编号            | 题目名称（英文和中文）                                           | 题目简述（中文）                       | 最简单样例                                                                                    | 解法                             |
| --------------- | ----------------------------------------------------- | ------------------------------ | ---------------------------------------------------------------------------------------- | ------------------------------ |
| 176<br><br>(m)  | [[Route Between Two Nodes in Graph]] (两点间路径)          | 判断有向图中是否存在两个节点之间的路径。           | 输入:  <br>graph = [ [1, 2], [3], [3], [] ]  <br>start = 0  <br>end = 3  <br>输出:  <br>true | 使用 BFS 或 DFS 遍历图，检查是否能到达目标节点。  |
| 431<br>*<br>(m) | [[Connected Component in Undirected Graph]] (无向图连通分量) | 找到无向图中的所有连通分量。                 | 输入:  <br>graph = [ [1, 2], [], [3], [] ]  <br>输出:  <br>[[0, 1, 2, 3]]                    | 使用 BFS 或 DFS 遍历图，记录每个连通分量中的节点。 |
| 618<br><br>(m)  | [[Search Graph Nodes ]](搜索图节点)                        | 在图中查找与目标节点颜色相同的所有节点，返回距离最近的节点。 | 输入:  <br>graph = [ [1, 2], [0, 3], [0, 3], [1, 2] ]  <br>target = 2  <br>输出:  <br>[2, 3] | 使用 BFS 遍历图，同时检查节点颜色。           |
| 1078<br><br>(e) | [[Degree of an Array]] (数组的度)                         | 找到具有数组中相同度的最短连续子数组长度。          | 输入:  <br>nums = [1, 2, 2, 3, 1]  <br>输出:  <br>2                                          | 使用哈希表记录元素的首次和最后出现位置以及频率，计算结果。  |
| 836<br><br>(h)  | [[Partition to K Equal Sum Subsets]] (分割为K个等和子集)      | 判断是否可以将数组分成 k 个子集，使每个子集的和相等。   | 输入:  <br>nums = [4, 3, 2, 3, 5, 2, 1]  <br>k = 4  <br>输出:  <br>true                      | 使用回溯和动态规划检查可能的分区方案。            |




## **五、選取三題詳細解釋**

---

### **1. LeetCode 133: 克隆圖（Clone Graph）**

**題目描述**：  
克隆一個無向連通圖，使得克隆後的圖與原圖結構相同。

**解法思路**：

- 使用 DFS 或 BFS 遍歷圖，複製節點並記錄克隆關係。

**Python 代碼**（DFS 實現）：

```python
class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []

def cloneGraph(node):
    old_to_new = {}

    def dfs(node):
        if node in old_to_new:
            return old_to_new[node]
        
        clone = Node(node.val)
        old_to_new[node] = clone
        for neighbor in node.neighbors:
            clone.neighbors.append(dfs(neighbor))
        return clone

    return dfs(node) if node else None

```
**時間複雜度**：

- O(N+M)O(N + M)O(N+M)：NNN 是節點數，MMM 是邊數。

**為何使用圖**：

- 使用圖結構適合表示節點之間的關聯，克隆圖要求遍歷整個結構。

---

### **2. LeetCode 200: 岛屿数量（Number of Islands）**

**題目描述**：  
給定一個二維網格，計算其中島嶼的數量（連通的1）。

**解法思路**：

- 使用 DFS 或 BFS 遍歷連通的陸地，將其標記為已訪問。

**Python 代碼**：
```python
def numIslands(grid):
    def dfs(r, c):
        if r < 0 or c < 0 or r >= rows or c >= cols or grid[r][c] == "0":
            return
        grid[r][c] = "0"
        for dr, dc in directions:
            dfs(r + dr, c + dc)

    rows, cols = len(grid), len(grid[0])
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    count = 0
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1":
                dfs(r, c)
                count += 1
    return count

```

**時間複雜度**：

- O(M×N)O(M \times N)O(M×N)：網格的總大小。

---

### **3. LeetCode 207: 課程表（Course Schedule）**

**題目描述**：  
給定一個課程的先修順序，判斷是否可以完成所有課程（有向圖環檢測）。

**解法思路**：

- 使用拓撲排序（Kahn 算法）判斷有向圖中是否存在環。

**Python 代碼**：
```python
from collections import deque, defaultdict

def canFinish(numCourses, prerequisites):
    graph = defaultdict(list)
    in_degree = [0] * numCourses

    for course, pre in prerequisites:
        graph[pre].append(course)
        in_degree[course] += 1

    queue = deque([i for i in range(numCourses) if in_degree[i] == 0])
    visited = 0

    while queue:
        node = queue.popleft()
        visited += 1
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return visited == numCourses

```

**時間複雜度**：

- O(N+M)O(N + M)O(N+M)：NNN 是課程數，MMM 是先修要求數。

---

## **總結**

- **圖（Graph）** 是一種靈活的數據結構，可用於表示各種實際場景，如網絡、地圖等。
- 在 LeetCode 中，圖問題主要通過 BFS、DFS 和拓撲排序等算法解決。
- 掌握圖的鄰接表表示及遍歷方法是解決圖相關問題的關鍵。




###### 數學

| **题目编号**       | **题目名称 (英文/中文)**        | **题目简述 (中文)**                                   | **样例**               | **解法**                                                                     |
| -------------- | ----------------------- | ----------------------------------------------- | -------------------- | -------------------------------------------------------------------------- |
| 513<br><br>(m) | [[Perfect Squares]]完美平方 | 给定一个正整数 n，找到至少需要多少个完全平方数（比如 1、4、9 等）使得它们的和等于 n。 | 输入: `n = 12` 输出: `3` | 使用队列进行广度优先搜索（BFS）。从 n 开始，减去每个小于等于 n 的完全平方数，将结果作为下一层的节点，直到找到 0 为止，记录层数作为结果。 |