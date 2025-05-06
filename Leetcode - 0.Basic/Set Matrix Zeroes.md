Lintcode 162
给定一个m×n矩阵，如果一个元素是0，则将其所在行和列全部元素变成0。
需要在原矩阵上完成操作。

**样例 1:**

```python
#输入:[[1,2],[0,3]]
#输出:[[0,2],[0,0]]
```

**样例 2:**

```python
#输入:[[1,2,3],[4,0,6],[7,8,9]]
#输出:[[1,0,3],[0,0,0],[7,0,9]]
```


```python
    def set_zeroes(self, matrix):
        # write your code here
        if len(matrix)==0:
            return
        rownum = len(matrix)
        colnum = len(matrix[0])
        row = [False for i in range(rownum)]
        col = [False for i in range(colnum)]
        for i in range(rownum):
            for j in range(colnum):
                if matrix[i][j] == 0:
                    row[i] = True
                    col[j] = True
        for i in range(rownum):
            for j in range(colnum):
                if row[i] or col[j]:
                    matrix[i][j] = 0
```
pass
解釋:
step1: create兩個list. row長度等於 [1, row number ],  col長度等於 [1, col number ]
step2: 遍歷矩陣, 如果遇到value=0, 則這個value的row跟col 更新為True
step3: 再次遍歷矩陣, 依照row跟col紀錄的True把對應的row跟col的元素都更新為0

# **LintCode 162 - Set Matrix Zeroes（矩陣置零）**

## **題目解析**

給定一個 `m x n` 的矩陣 `matrix`，如果某個元素 `matrix[i][j]` 為 `0`，則要求將該元素所在的**整行與整列**全部置 `0`。

---

## **解法解析**

### **思路**

這題的關鍵點在於：

1. **如何記錄哪些行和列需要置零**
2. **如何有效率地修改矩陣**

最直觀的做法是：

1. **第一輪遍歷** 找出矩陣中所有的 `0`，並記錄它們所在的行與列。
2. **第二輪遍歷** 根據記錄來更新矩陣，將對應的行與列全部設為 `0`。

---

### **解法步驟**

1. **使用兩個陣列 `row[]` 和 `col[]` 來記錄哪些行與列需要置 `0`**
    
    - `row[i] = True` 表示 **第 `i` 行要全設為 `0`**
    - `col[j] = True` 表示 **第 `j` 列要全設為 `0`**
    - **時間複雜度：O(m * n)**
2. **第二次遍歷矩陣，根據 `row[]` 和 `col[]` 修改矩陣**
    
    - 如果 `row[i] == True` 或 `col[j] == True`，則 `matrix[i][j] = 0`
    - **時間複雜度：O(m * n)**
3. **最終矩陣即為答案**
    

---

## **具體舉例**

假設輸入：

python

複製編輯

`matrix = [     [1, 2, 3],     [4, 0, 6],     [7, 8, 9] ]`

### **Step 1: 記錄 `0` 的行與列**

初始化 `row` 和 `col` 陣列：

ini

複製編輯

`row = [False, False, False] col = [False, False, False]`

遍歷 `matrix`：

sql

複製編輯

`(1, 1) 是 0，標記 row[1] = True, col[1] = True`

更新後：

ini

複製編輯

`row = [False, True, False] col = [False, True, False]`

---

### **Step 2: 更新矩陣**

根據 `row[]` 和 `col[]`，修改 `matrix`：

sql

複製編輯

`row[1] = True，所以第 1 行全部變為 0 col[1] = True，所以第 1 列全部變為 0`

**最終輸出：**

python

複製編輯

`[     [1, 0, 3],     [0, 0, 0],     [7, 0, 9] ]`

---

## **時間與空間複雜度分析**

- **時間複雜度：O(m * n)**
    - **第一輪遍歷** 找出 `0` 位置，O(m * n)
    - **第二輪遍歷** 設置 `0`，O(m * n)
    - **總計：O(m * n)**
- **空間複雜度：O(m + n)**
    - 需要兩個陣列 `row[]` 和 `col[]`，各佔 `O(m)` 和 `O(n)`
    - **總計：O(m + n)**

---

## **最佳解法：使用 O(1) 空間**

**問題**：目前的解法使用了 `O(m + n)` 額外空間，能否優化？

### **改進方案**

- **利用矩陣的第一行和第一列來記錄是否需要置零**
- **用 `matrix[0][j]` 記錄第 `j` 列是否要設為 0**
- **用 `matrix[i][0]` 記錄第 `i` 行是否要設為 0**
- **使用一個變數 `first_col` 來記錄第一列是否要變 0**

這樣就可以將空間複雜度降低至 `O(1)`。

### **時間與空間複雜度**

- **時間複雜度：O(m * n)**
- **空間複雜度：O(1)**（不再使用額外陣列）

---

## **其他解法**

### **1. 使用集合 `set()` 儲存 0 的行和列**

- **時間複雜度**：O(m * n)
- **空間複雜度**：O(min(m, n))

### **2. 直接修改矩陣（O(1) 空間最佳解）**

- **時間複雜度**：O(m * n)
- **空間複雜度**：O(1)

---

## **類似題目（LintCode / LeetCode 類似題）**

這類題目通常涉及 **矩陣操作 + 動態更新**，有以下幾個類似的題目：

|**題號**|**題目名稱**|**難度**|**說明**|
|---|---|---|---|
|**LintCode 162**|Set Matrix Zeroes|⭐⭐⭐|**矩陣置零**|
|**LintCode 903**|Range Addition|⭐⭐⭐|**區間加法，類似矩陣操作**|
|**LintCode 431**|Connected Components in Undirected Graph|⭐⭐⭐|**矩陣操作，DFS / BFS 找聯通區塊**|
|**LintCode 655**|Add Strings|⭐⭐|**數字處理，類似矩陣遍歷**|
|**LintCode 829**|Word Pattern II|⭐⭐⭐|**字串匹配問題，類似矩陣標記**|
|**LeetCode 73**|Set Matrix Zeroes|⭐⭐⭐|**相同問題**|
|**LeetCode 289**|Game of Life|⭐⭐⭐|**使用額外空間來標記矩陣的變化**|

---

## **總結**

4. **直觀解法**：
    
    - 使用 `O(m + n)` 陣列標記要置零的行和列，時間 `O(m * n)`，空間 `O(m + n)`。
5. **最佳解法（O(1) 空間）**：
    
    - 直接利用 **第一行和第一列** 來標記變化，空間 `O(1)`。
6. **這類問題的關鍵技巧**：
    
    - **遍歷矩陣並標記變化**
    - **避免額外空間使用**
    - **優化矩陣修改步驟**