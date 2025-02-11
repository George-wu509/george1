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