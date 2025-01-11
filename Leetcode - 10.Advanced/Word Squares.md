

**样例 1:**
```
输入:
["area","lead","wall","lady","ball"]
输出:
[["wall","area","lead","lady"],["ball","area","lead","lady"]]

解释:
输出包含 两个单词矩阵，这两个矩阵的输出的顺序没有影响(只要求矩阵内部有序)。
```
**样例 2:**
```
输入:
["abat","baba","atan","atal"]
输出:
 [["baba","abat","baba","atan"],["baba","abat","baba","atal"]]
```



```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.word_list = []


class Trie:
    def __init__(self):
        self.root = TrieNode()
        
    def add(self, word):
        node = self.root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
            node.word_list.append(word)
        node.is_word = True

    def find(self, word):
        node = self.root
        for c in word:
            node = node.children.get(c)
            if node is None:
                return None
        return node
        
    def get_words_with_prefix(self, prefix):
        node = self.find(prefix)
        return [] if node is None else node.word_list
        
    def contains(self, word):
        node = self.find(word)
        return node is not None and node.is_word
        
        
class Solution:
    def wordSquares(self, words):
        trie = Trie()
        for word in words:
            trie.add(word)
            
        squares = []
        for word in words:
            self.search(trie, [word], squares)
        
        return squares
        
    def search(self, trie, square, squares):
        n = len(square[0])
        curt_index = len(square)
        if curt_index == n:
            squares.append(list(square))
            return
        
        # Pruning, it's ok to remove it, but will be slower
        for row_index in range(curt_index, n):
            prefix = ''.join([square[i][row_index] for i in range(curt_index)])
            if trie.find(prefix) is None:
                return
        
        prefix = ''.join([square[i][curt_index] for i in range(curt_index)])
        for word in trie.get_words_with_prefix(prefix):
            square.append(word)
            self.search(trie, square, squares)
            square.pop() # remove the last word
```
pass