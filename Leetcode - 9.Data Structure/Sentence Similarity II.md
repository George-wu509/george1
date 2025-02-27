Lintcode 855
给出两个句子“words1”、“words2”(每个单词都表示为字符串数组) 和一组相似的单词对“pair”，判断两个句子是否相似。  
例如，' words1 = ["great"， "acting"，" skills"]和' words2 = ["fine"， "drama"， "talent"]是相似的，如果相似的单词对是' pair = [[great"， "good"]， ["fine"， "good"]， ["acting"，"drama"]， ["skills"，"talent"]]。  
注意，相似性关系是可传递的。例如，如果“great”和“good”相似，“fine”和“good”相似，那么“great”和“fine”**相似。**  
相似性也是对称的。例如，“great”和“fine”相似等同于“fine”和“great”相似。  
而且，一个单词总是和它自己相似。例如，' words1 = ["great"] '、' words2 = ["great"] '、' pair =[] '这几个句子是相似的，即使没有指定相似的单词对。  
最后，句子只有在单词数量相同的情况下才能相似。所以像words1 = ["great"]这样的句子永远不可能和words2 = ["doubleplus"，"good"]相似。


**样例 1:**
```python
"""
输入:
["7", "5", "4", "11", "13", "15", "19", "12", "0", "10"]
["16", "1", "7", "3", "15", "10", "13", "2", "19", "8"]
[["6", "18"], ["8", "17"], ["1", "13"], ["0", "8"], ["9", "14"], ["11", "17"], ["11", "19"], ["13", "16"], ["0", "18"], ["3", "11"], ["1", "9"], ["2", "11"], ["2", "4"], ["0", "19"], ["8", "12"], ["8", "19"], ["16", "19"], ["1", "11"], ["2", "18"], ["0", "16"], ["7", "11"], ["6", "8"], ["9", "17"], ["8", "16"], ["3", "13"], ["7", "9"], ["7", "10"], ["3", "6"], ["15", "19"], ["1", "5"], ["2", "14"], ["1", "18"], ["8", "15"], ["14", "19"], ["3", "17"], ["6", "10"], ["5", "17"], ["10", "15"], ["1", "10"], ["4", "6"]]
输出:
true
```
**样例 2:**
```python
"""
输入:
["great","acting","skills"]
["fine","drama","talent"]
[["great","good"],["fine","good"],["drama","acting"],["skills","talent"]]
输出:
true
```



```python
"""
class Solution:
    def are_sentences_similar_two(self, words1: List[str], words2: List[str], pairs: List[List[str]]) -> bool:
        if len(words1) != len(words2): return False
        import itertools
        index = {}
        count = itertools.count()
        dsu = DSU(2 * len(pairs))
        for pair in pairs:
            for p in pair:
                if p not in index:
                    index[p] = next(count)
            dsu.union(index[pair[0]], index[pair[1]])

        return all(w1 == w2 or
                   w1 in index and w2 in index and
                   dsu.find(index[w1]) == dsu.find(index[w2])
                   for w1, w2 in zip(words1, words2))
class DSU:
    def __init__(self, N):
        self.par = list(range(N))
    def find(self, x):
        if self.par[x] != x:
            self.par[x] = self.find(self.par[x])
        return self.par[x]
    def union(self, x, y):
        self.par[self.find(x)] = self.find(y)
```
pass

# **LintCode 855: Sentence Similarity II 解法詳細解析**

## **問題描述**

給定兩個句子 `words1` 和 `words2`，它們的長度相同。我們還有一組**相似單詞對 `pairs`**，其中每個 `pairs[i] = [a, b]` 表示 `a` 和 `b` 是**相似的單詞**。

我們要判斷 `words1` 和 `words2` 是否可以視為**相似的句子**：

- `words1[i]` 和 `words2[i]` **要麼相同，要麼在相似單詞對中可以連通**。

---

## **解法分析**

這題的核心在於：

1. **如何判斷單詞 `w1` 和 `w2` 是否相似？**
    
    - 直接查找 `pairs` 效率太低，因此我們要**建立單詞之間的連接關係**。
    - **如果 `w1` 和 `w2` 通過若干步相似關係連接，那麼它們仍然視為相似單詞**。
2. **如何高效地查找單詞之間的關聯？**
    
    - **使用並查集（Union-Find / DSU, Disjoint Set Union）來構建連通分量**。
    - 這樣**可以快速查找 `w1` 和 `w2` 是否屬於同一組**。
3. **為什麼使用並查集（DSU）？**
    
    - **並查集能夠高效合併關聯單詞**。
    - **合併與查找的均攤時間複雜度為 `O(α(n)) ≈ O(1)`**（阿克曼函數的反函數）。
    - **比 BFS 或 DFS 更快**，因為我們不需要遍歷整個圖。

---

## **解法步驟**

### **Step 1: 檢查 `words1` 和 `words2` 長度**

`if len(words1) != len(words2): return False`

- 如果長度不同，則**必然無法相似**，直接返回 `False`。

---

### **Step 2: 使用並查集建立單詞之間的關聯**

`import itertools index = {}  # 單詞映射到唯一的數字索引 count = itertools.count()  # 自動生成唯一索引 dsu = DSU(2 * len(pairs))  # 並查集初始化`

- `index`：用來存儲**單詞到索引的映射**（因為 `DSU` 只處理數字）。
- `count`：用來分配唯一的索引給每個單詞。
- `dsu`：並查集數據結構。

---

### **Step 3: 將 `pairs` 中的單詞加入 `index`，並用並查集合併它們**

`for pair in pairs:     for p in pair:         if p not in index:             index[p] = next(count)  # 分配唯一的數字索引     dsu.union(index[pair[0]], index[pair[1]])  # 合併相似的單詞`

- **將 `pairs[i] = [a, b]` 轉換為 `index[a]` 和 `index[b]`**。
- **合併 `a` 和 `b`，使它們屬於同一組**。
- **並查集保證相似單詞會形成連通分量**。

---

### **Step 4: 遍歷 `words1` 和 `words2`，判斷它們是否屬於相同的集合**

`return all(w1 == w2 or            w1 in index and w2 in index and            dsu.find(index[w1]) == dsu.find(index[w2])            for w1, w2 in zip(words1, words2))`

- `w1 == w2`：如果 `w1` 和 `w2` 本來就相同，那麼它們顯然是相似的。
- `w1 in index and w2 in index`：確保 `w1` 和 `w2` **都在 `pairs` 中**，否則它們沒有相似關係。
- `dsu.find(index[w1]) == dsu.find(index[w2])`：
    - **如果 `w1` 和 `w2` 屬於同一個連通分量，那麼它們是相似的**。
    - **否則返回 `False`**。

---

## **舉例解析**

### **Example 1**

`words1 = ["great", "acting", "skills"] words2 = ["fine", "drama", "talent"] pairs = [["great", "good"], ["good", "fine"], ["acting", "drama"], ["skills", "talent"]]`

### **Step 1: 初始化**

- **長度相同**，繼續執行。

---

### **Step 2: 建立 `index` 映射**

`index = {     "great": 0, "good": 1, "fine": 2,     "acting": 3, "drama": 4, "skills": 5, "talent": 6 }`

---

### **Step 3: 並查集合併**

合併 `pairs`：

`great - good - fine acting - drama skills - talent`

對應的並查集 `par[]`：

`[1, 1, 1, 4, 4, 6, 6]`

---

### **Step 4: 判斷句子是否相似**

`find("great") == find("fine") → True find("acting") == find("drama") → True find("skills") == find("talent") → True`

**結果：`True`**

---

## **時間複雜度分析**

|步驟|複雜度|
|---|---|
|**Step 1: 長度檢查**|`O(1)`|
|**Step 2: 建立 `index`（單詞映射到索引）**|`O(P)`（`P` 為 `pairs` 數量）|
|**Step 3: 並查集合併 `pairs`**|`O(P α(P)) ≈ O(P)`|
|**Step 4: 判斷 `words1[i]` 和 `words2[i]` 是否相似**|`O(W α(W)) ≈ O(W)`（`W` 為 `words1` 長度）|
|**總計**|`O(P + W)`|

由於 `α(n) ≈ O(1)`，所以這是一個接近線性的解法。

---

## **其他解法（不需代碼）**

### **1. DFS 或 BFS**

- **構建圖**，`pairs[i] = [a, b]` 代表一條**無向邊**。
- **對 `words1[i]` 和 `words2[i]` 進行 BFS 或 DFS 搜索，檢查是否可達**。
- **時間複雜度 `O(W * P)`**，比並查集稍慢。

### **2. Floyd-Warshall（動態規劃）**

- **使用鄰接矩陣存儲單詞之間的相似度**。
- **使用 Floyd-Warshall 求任意兩個單詞的最短路徑（可達性）**。
- **時間複雜度 `O(N^3)`（太慢，不適合大數據）**。

---

## **總結**

### **最佳解法：並查集（Union-Find）**

- **時間 `O(P + W)`，空間 `O(P)`**
- **利用並查集快速合併與查找**
- **適用於大數據量的查詢**

### **其他解法**

1. **DFS / BFS**（`O(W * P)`）：適用於小規模數據。
2. **Floyd-Warshall**（`O(N^3)`）：適用於極小數據，但不適合大規模圖。

### **為何選擇並查集？**

- **查找是否連通最快，幾乎 `O(1)`**
- **比 DFS/BFS 更適合大數據**
- **比 Floyd-Warshall 節省空間**
- **適合動態處理單詞關係的場景**

這就是為什麼 **並查集** 是解決這類 **連通性查詢問題** 的**最佳選擇**！ 🚀

  

O

搜尋
