

**样例 1:**
```
输入:
["bat", "tab", "cat"]
输出:
[[0, 1], [1, 0]]

解释:
回文串为 `["battab", "tabbat"]`
```

**样例 2:**
```
输入:
["abcd", "dcba", "lls", "s", "sssll"]
输出:
[[0, 1], [1, 0], [3, 2], [2, 4]]

解释:
回文串为 `["dcbaabcd", "abcddcba", "slls", "llssssll"]
```


```python
class TrieNode:  
    def __init__(self):  
        # 对应单词的下标
        self.index = -1
        # 儿子结点
        self.next = [None] * 26 
        
class Solution:
    """
    @param words: a list of unique words
    @return: all pairs of distinct indices
    """
    # 判断是否为回文串
    def is_palindrome(self, str):
        l = len(str)
        for i in range(l // 2):
            if str[i] != str[l-1-i]:
                return False
        return True
    
    # 寻找单词
    def search(self, s, root):
        for ch in s:
            if root.next[ord(ch) - ord('a')]:
                root = root.next[ord(ch) - ord('a')]
            else:
                return -1
        return root.index
    
    def palindrome_pairs(self, words):
        root = TrieNode()
        ans = []
        n = len(words)
        for i in range(n):
            # 单词逆序加入字典树，最后一个结点index为单词下标
            tmp = words[i][::-1]
            # pos表示当前结点
            pos = root
            for j in tmp:
                if pos.next[ord(j) - ord('a')] is None:
                    pos.next[ord(j) - ord('a')] = TrieNode()
                pos = pos.next[ord(j) - ord('a')]
            pos.index = i;
        for i in range(n):
            l = len(words[i])
            for j in range(l + 1):
                left = words[i][:j]
                right = words[i][j:]
                # 如果左边前缀部分可以找到对应部分，且后边后缀部分为回文串
                # 那么可以组成回文串，加入答案
                index = self.search(left, root)
                if index != -1 and index != i and self.is_palindrome(right):
                    ans.append([i, index])
                # 如果右边后缀部分可以找到对应部分，且左边前缀部分为回文串
                # 那么可以组成回文串，如果前缀部分不为""，则加入答案
                index = self.search(right , root)
                if index != -1 and index != i and self.is_palindrome(left) and left != "":
                    ans.append([index, i])
        return ans
```
pass