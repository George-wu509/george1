
**样例 1:**
```
输入: words = ["cat","cats","catsdogcats","dog","dogcatsdog","hippopotamuses","rat","ratcatdogcat"]
输出: ["catsdogcats","dogcatsdog","ratcatdogcat"]
解释: 
"catsdogcats"由"cats", "dog" 和 "cats"组成; 
"dogcatsdog"由"dog", "cats"和"dog"组成; 
"ratcatdogcat"由"rat", "cat", "dog"和"cat"组成。
```
**样例 2:**
```
输入: words = ["a","b","ab","abc"]
输出: ["ab"]
解释: 
"ab"由"a"和 "b"组成。
```




```python
class Solution:
    """
    @param words: List[str]
    @return: return List[str]
    """
    def wordBreak(self, word, cands):
        if not cands:
            return False
        dp = [False] * (len(word) + 1) #store whether w.substr(0, i) can be formed by existing words
        dp[0] = True #empty string is always valid
        for i in range(1, len(word) + 1):
            for j in reversed(range(0, i)):
                if not dp[j]:
                    continue
                if word[j:i] in cands:
                    dp[i] = True
                    break
        return dp[-1]
    
    def find_all_concatenated_words_in_a_dict(self, words):
        # write your code here
        words.sort(key=lambda x: -len(x))
        cands = set(words) # using hash for acceleration
        ans = []
        for i in range(0, len(words)):
            cands -= {words[i]}
            if self.wordBreak(words[i], cands):
                ans += words[i],
        return ans
```
pass