
**样例 1:**
```
给出字符串 `["abc","abd","abcd","adc"]`，目标字符串为 `"ac"` ，k = `1`
返回 `["abc","adc"]`
输入:
["abc", "abd", "abcd", "adc"]
"ac"
1
输出:
["abc","adc"]

解释:
"abc" 去掉 "b"
"adc" 去掉 "d"
```
**样例 2:**
```
输入:
["acc","abcd","ade","abbcd"]
"abc"
2
输出:
["acc","abcd","ade","abbcd"]

解释:
"acc" 把 "c" 变成 "b"
"abcd" 去掉 "d"
"ade" 把 "d" 变成 "b"把 "e" 变成 "c"
"abbcd" 去掉 "b" 和 "d"
```



```python
from typing import (
    List,
)

class TrieNode:
    def __init__(self):
        self.is_word = False
        self.children = {}


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.is_word = True

class Solution:

    def kDistance(self, words, target, k):
        trie = Trie()
        for word in words:
            trie.insert(word)

        n = len(target)
        dp = list(range(n + 1))  # [0, 1, 2 ... ]
        results = []
        self.traverse(trie.root, '', 0, dp, target, k, results)
        return results

    def traverse(self, node, word, i, dp, target, k, results):
        n = len(target)
        if node.is_word and dp[n] <= k:
            results.append(word)

        for c in node.children:
            dp_next = [i + 1] * (n + 1)
            for j in range(1, n + 1):
                # dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
                dp_next[j] = min(dp[j], dp_next[j - 1], dp[j - 1]) + 1
                if c == target[j - 1]:
                    # dp[i][j] = min(dp[i][j], dp[i - 1][j - 1])
                    dp_next[j] = min(dp_next[j], dp[j - 1])
            self.traverse(node.children[c], word + c, i + 1, dp_next, target, k, results)
```
pass