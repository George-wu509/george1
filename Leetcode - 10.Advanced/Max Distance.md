
**样例 1:**
```
输入：["011000","0111010","01101010"]
输出：9
解释: "0111010" 和 "01101010" 的最长前缀是 "011", 距离为 len("1010")+len("01010")=9
```
**样例 2:**

```
输入：["011000","0111011","01001010"]
输出：11
解释："0111011" 和 "01001010" 的最长前缀 "01", 距离是 len("11011")+len("001010")=11
```




```python
from typing import (
    List,
)

class TrieNode:
    def __init__(self):
        self.is_end = False
        self.height = 1
        self.children = {}

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, s):
        node = self.root
        for i, c in enumerate(s):
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
            node.height = max(node.height, len(s) - i)
        node.is_end = True


class Solution:

    def getAns(self, str_list):
        trie = Trie()
        for s in str_list:
            trie.insert(s)

        return self.calc_diff(trie.root)

    def calc_diff(self, node):
        diff = 0
        if len(node.children) == 2:
            diff = max(diff, node.children['0'].height + node.children['1'].height)
        for c in node.children:
            diff = max(diff, self.calc_diff(node.children[c]))
        if len(node.children) == 1 and node.is_end:
            diff = max(diff, node.height - 1)
        return diff
```
pass