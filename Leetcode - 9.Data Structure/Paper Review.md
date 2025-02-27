Lintcode 1463
我们定义，两个论文的相似度为最长的相似单词**子序列**长度 * 2 除以两篇论文的**总长度**。  
给定两篇论文`words1`，`words2`（每个表示为字符串数组），和相似单词对列表`pairs`，求两篇论文的相似度。

注意：相似关系是可传递的。例如，如果“great”和“good”类似，而“fine”和“good”类似，那么“great”和“fine”类似。  
相似性也是对称的。 例如，“great”和“good”相似，则“good”和“great”相似。  
另外，一个词总是与其本身相似。


**样例 1:**
```python
"""
输入：words1= ["great","acting","skills","life"]，words2= ["fine","drama","talent","health"]，pairs=  [["great", "good"], ["fine", "good"], ["acting","drama"], ["skills","talent"]]
输出：：0.75
解释：
两篇单词相似的子单词序列为
"great","acting","skills"
"fine","drama","talent"
总长度为8
相似度为6/8=0.75`
```
**样例 2:**
```python
"""
输入：words1= ["I","love","you"]，words2= ["you","love","me"]，pairs=  [["I", "me"]]
输出：0.33
解释：
两篇单词相似的子单词序列为
"I"
"me"
或
"love"
"love"
或
"you"
"you"
总长度为6
相似度为2/6=0.33
```



```python
class Node :
    def __init__(self) :
        self.fa=[]
        for i in range(6200) :
             self.fa.append(i)
    def find(self,x) :
        if self.fa[x]==x :
            return x
        else :
            self.fa[x]=self.find(self.fa[x])
            return self.fa[x]
    def unity(self,x,y) :
        x=self.find(x)
        y=self.find(y)
        self.fa[x]=y
class Solution:
    """
    @param words1: the words in paper1
    @param words2: the words in paper2
    @param pairs: the similar words pair
    @return: the similarity of the two papers
    """
    def get_similarity(self, words1, words2, pairs):
        # Write your code here
        ans=Node()
        cnt=0
        a=[]
        b=[]
        s={}
        a.append(0)
        b.append(0)
        for i in pairs:
            if s.__contains__(i[0])==0 :
                cnt+=1
                s[i[0]]=cnt
            if s.__contains__(i[1])==0 :
                cnt+=1
                s[i[1]]=cnt
            ans.unity(s[i[0]],s[i[1]])
        for i in words1 :
            if s.__contains__(i)==0: 
                cnt+=1
                s[i]=cnt
            a.append(s[i])
        for i in words2 :
            if s.__contains__(i)==0 : 
                cnt+=1
                s[i]=cnt
            b.append(s[i])
        dp=[[0]*1000 for i in range(1000)]
        for i in range(1,len(words1)+1) :
            for j in range(1,len(words2)+1) :
                x=ans.find(a[i])
                y=ans.find(b[j])
                if x==y :
                    dp[i][j]=dp[i-1][j-1]+1
                else :
                    dp[i][j]=max(dp[i-1][j],dp[i][j-1])
        res=dp[len(words1)][len(words2)]*2.0/(len(words1)+len(words2))
        return res
```
pass

- dp
- 并查集

题解：  
本题强调相似的定义和性质，若相似具有传递性，就可以利用并查集维护相似的字符串。寻找最长公共子序列即为经典的dp问题。  
此处选择使用map将字符串与数字映射，方便判断以及并查集维护。

-  最长公共子序列状态转移方程：

   dp[i][j]=0  (i=0||j=0)  
dp[i][j]=max(dp[i-1][j],dp[i][j-1])  (a[i]!=b[j])  
dp[i][j]=dp[i-1][j-1]+1 (a[i]=b[j])

# **LintCode 1463: Paper Review 解法詳細解析**

## **問題描述**

我們有兩篇論文 `words1` 和 `words2`，它們分別由單詞組成。我們還有一組**相似單詞對 `pairs`**，其中 `pairs[i] = [a, b]` 表示 `a` 和 `b` 是**相似的單詞**。

**我們的目標是計算 `words1` 和 `words2` 之間的相似度**：

$\Huge \text{similarity} = \frac{2 \times \text{LCS}(words1, words2)}{\text{len(words1)} + \text{len(words2)}}$​

其中 `LCS` 是**最長公共子序列（Longest Common Subsequence, LCS）**，並且**相似單詞也算是匹配的**。

---

## **解法分析**

這題的核心是：

1. **如何判斷 `words1[i]` 和 `words2[j]` 是否相似？**
    
    - 除了 `words1[i] == words2[j]`，還需要考慮**是否在 `pairs` 中間接相連**。
    - 使用 **並查集（Union-Find）** 來合併**所有相似單詞組**，這樣我們可以快速查詢 `words1[i]` 和 `words2[j]` 是否在同一個集合中。
2. **如何計算 `words1` 和 `words2` 的 LCS？**
    
    - 使用 **動態規劃（DP）**： $dp[i][j] = \begin{cases} dp[i-1][j-1] + 1, & \text{if words1[i] 和 words2[j] 相似} \\ \max(dp[i-1][j], dp[i][j-1]), & \text{otherwise} \end{cases}$
    - 這個方法可以在 `O(mn)` 的時間內計算 LCS，其中 `m = len(words1)`，`n = len(words2)`。

---

## **解法步驟**

### **Step 1: 使用並查集合併所有相似單詞**

`ans = Node()  # 並查集 cnt = 0 s = {}`

- **`s` 用來映射單詞到唯一的數字編號**，因為 `並查集（Union-Find）` 只能處理數字，而不是字串。
- **遍歷 `pairs`，合併所有相似單詞**：
```python
 for i in pairs:
    if i[0] not in s:
        cnt += 1
        s[i[0]] = cnt
    if i[1] not in s:
        cnt += 1
        s[i[1]] = cnt
    ans.unity(s[i[0]], s[i[1]])
```
    
- **合併 `pairs` 內的單詞，使得所有相似的單詞組成一個集合**。

---

### **Step 2: 將 `words1` 和 `words2` 轉換為數字索引**

`a = [] b = []`

- **遍歷 `words1` 和 `words2`，將單詞轉換為數字編號**：
   
    `for i in words1:     if i not in s:         cnt += 1         s[i] = cnt     a.append(s[i])`

    `for i in words2:     if i not in s:         cnt += 1         s[i] = cnt     b.append(s[i])`
    
- **這樣 `words1[i]` 和 `words2[j]` 可以透過 `find()` 來判斷是否相似**。

---

### **Step 3: 計算 LCS**

`dp = [[0] * (len(words2) + 1) for _ in range(len(words1) + 1)]`

- **`dp[i][j]` 表示 `words1[0:i]` 和 `words2[0:j]` 的最長公共子序列長度**。
- **轉移方程**：

    `for i in range(1, len(words1) + 1):     for j in range(1, len(words2) + 1):         x = ans.find(a[i-1])         y = ans.find(b[j-1])         if x == y:             dp[i][j] = dp[i-1][j-1] + 1         else:             dp[i][j] = max(dp[i-1][j], dp[i][j-1])`
    
- **如果 `words1[i]` 和 `words2[j]` 在同一個集合，則 `LCS +1`**。
- **否則，取 `dp[i-1][j]` 和 `dp[i][j-1]` 的較大值**。

---

### **Step 4: 計算相似度**

`res = dp[len(words1)][len(words2)] * 2.0 / (len(words1) + len(words2)) return res`

- **`LCS * 2 / (words1 長度 + words2 長度)`** 計算出相似度。

---

## **舉例分析**

### **Example 1**

words1 = ["apple", "banana", "cherry"] 
words2 = ["apple", "pear", "cherry"] 
pairs = [ ["banana", "pear"], ["cherry", "cherry"] ]

#### **Step 1: 使用並查集合併**

`s = {"banana": 1, "pear": 2, "cherry": 3} 
ans: {1 → 2, 3 → 3}`

- `banana` 和 `pear` 被合併。
- `cherry` 自己是相似的。

#### **Step 2: 將單詞轉換為索引**

words1 = ["apple", "banana", "cherry"] → [4, 1, 3] 
words2 = ["apple", "pear", "cherry"] → [4, 2, 3]`

#### **Step 3: 計算 `LCS`**

|`dp` 表|`"apple"`|`"pear"`|`"cherry"`|
|---|---|---|---|
|`"apple"`|1|1|1|
|`"banana"`|1|**2**|2|
|`"cherry"`|1|2|**3**|

- **`LCS = 2`**
- **相似度 = `(2 * 2) / (3 + 3) = 0.6667`**

---

## **時間與空間複雜度分析**

### **時間複雜度**

|步驟|時間複雜度|
|---|---|
|**Step 1: 構建並查集**|`O(P α(P))`（`P` 是 `pairs` 數量）|
|**Step 2: 轉換 `words1` 和 `words2`**|`O(M + N)`|
|**Step 3: 計算 LCS**|`O(MN)`|
|**Step 4: 計算相似度**|`O(1)`|
|**總計**|`O(MN + P α(P))`|

### **空間複雜度**

|內容|空間|
|---|---|
|**並查集 `fa[]`**|`O(P)`|
|**哈希表 `s`**|`O(P + M + N)`|
|**DP 表 `dp[][]`**|`O(MN)`|
|**總計**|`O(P + MN)`|

---

## **其他解法（不需代碼）**

### **1. 直接 LCS + 哈希表**

- **先計算 LCS，然後用 `dict` 查詢相似單詞**。
- **時間複雜度 `O(MN + P)`**。

### **2. DFS/BFS 遍歷圖**

- **將 `pairs` 視為圖，使用 DFS/BFS 查找 `words1[i]` 和 `words2[j]` 是否相似**。
- **時間複雜度 `O(P + MN)`**。

---

## **總結**

- **最佳解法：並查集 + LCS**
- **時間 `O(MN + P α(P))`，空間 `O(MN + P)`**
- **比 DFS/BFS 更快，適合大數據場景**
- **適用於所有需要比較兩個文本相似度的問題** 🚀

  

O

搜尋