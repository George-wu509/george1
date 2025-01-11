

输入:["aaa","bbc","bcd"]  
输出:["a","bb","bc"]  
解释:"a"仅是"aaa" 的前缀  
"bb"仅是"bbc"的前缀  
"bc"仅是"bcd"的前缀


```python
from typing import (
    List,
)

class Trie:
    def __init__(self):
        self.words = []
        self.sons = dict()
    
    def insert(self, word):
        root = self
        root.words.append(word)
        for ch in word:
            if ch not in root.sons:
                root.sons[ch] = Trie()
            root = root.sons[ch]
            root.words.append(word)
    
    def searchPrefix(self, word):
        root = self
        prefix = ''
        for ch in word:
            if len(root.words) == 1 and root.words[0]==word: return prefix
            if ch not in root.sons: return ''
            prefix+=ch
            root = root.sons[ch]
        return prefix

class Solution:
    def short_perfix(self, string_array: List[str]) -> List[str]:
        res = []
        root = Trie()
        for word in string_array:
            root.insert(word)
        for word in string_array:
            res.append(root.searchPrefix(word))
        return res
```
pass