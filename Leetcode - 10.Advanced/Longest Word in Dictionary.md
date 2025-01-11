
**样例1:**
```
输入: 
words = ["w","wo","wor","worl", "world"]
输出: "world"
解释: 
单词 "wo" 可以通过 "w" 增加一个字母构成
单词 "wor" 可以通过 "wo" 增加一个字母构成
单词 "worl" 可以通过 "wor" 增加一个字母构成
单词 "world" 可以通过 "worl" 增加一个字母构成
单词 "world" 是所有情况中的最长的一个，因此答案为 "world"
```
**样例2:**
```
输入: 
words = ["a", "banana", "app", "appl", "ap", "apply", "apple"]
输出: "apple"
解释: 
单词 "apple" 可以通过 "appl" 增加一个字母构成
单词 "apply" 可以通过 "appl" 增加一个字母构成
单词 "apple" 和 "apply" 都是最长的单词, 但是 "apple" 的字典序比 "apply" 小
```



```python
class Trie:
    def __init__(self):
        self.children = [None] * 26
        self.isEnd = False

    def insert(self, word):
        node = self
        for ch in word:
            ch = ord(ch) - ord('a')
            if not node.children[ch]:
                node.children[ch] = Trie()
            node = node.children[ch]
        node.isEnd = True

    def search(self, word):
        node = self
        for ch in word:
            ch = ord(ch) - ord('a')
            if node.children[ch] is None or not node.children[ch].isEnd:
                return False
            node = node.children[ch]
        return True

class Solution:
    def longest_word(self, words):
        t = Trie()
        for word in words:
            t.insert(word)
        longest = ""
        for word in words:
            if t.search(word) and (len(word) > len(longest) or len(word) == len(longest) and word < longest):
                longest = word
        return longest
```
pass