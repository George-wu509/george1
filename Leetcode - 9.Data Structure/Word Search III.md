
**样例 1:**
```
输入：
["doaf","agai","dcan"]，["dog","dad","dgdg","can","again"]
输出：
2
解释：
  d o a f
  a g a i
  d c a n
矩阵中查找，你可以同时找到dad和can。
```
**样例 2:**
```
输入：
["a"]，["b"]
输出：
0
解释：
 a
矩阵中查找，返回0。
```


```python
class Trie:
    def __init__(self):
        self.children = {}
        self.flag = False
        self.hasWord = False
    
    def put(self, key):
        if key == '':
            self.flag = True
            self.hasWord = True
            return
        
        if key[0] not in self.children:
            self.children[key[0]] = Trie()
        self.children[key[0]].put(key[1:])
        self.hasWord = True
    
    def pop(self, key):
        if key == '':
            self.flag = False
            self.hasWord = False
            return
        if key[0] not in self.children:
            return
        self.children[key[0]].pop(key[1:])
        self.hasWord = any([child.hasWord for child in self.children.values()])
        
    def has(self, key):
        if key == '':
            return self.flag
        
        if not self.hasWord:
            return False
        if key[0] not in self.children:
            return False
        return self.children[key[0]].has(key[1:])


class Solution:
    DIRECT_X = [1, 0, 0, -1]
    DIRECT_Y = [0, 1, -1, 0]
    def word_search_i_i_i(self, board, words):
        trie = Trie()
        for word in words:
            trie.put(word)
        
        self.results = 0
        self.ans = 0
        for r in range(len(board)):
            for c in range(len(board[0])):
                self.search(trie, trie, board, r, c, r, c)
        return self.ans
                
    def search(self, root, trie, board, x, y, start_x, start_y):
        char = board[x][y]
        if char not in trie.children:
            return
        trie = trie.children[char]
        board[x][y] = '.'
        if trie.flag:
            self.results += 1
            trie.flag = False
            self.ans = max(self.ans, self.results)
            for i in range(start_x, len(board)):
                if i == start_x:
                    range_j = range(start_y + 1, len(board[0]))
                else:
                    range_j = range(len(board[0]))
                for j in range_j:
                    if board[i][j] != '.':
                        self.search(root, root, board, i, j, i, j)
            trie.flag = True
            self.results -= 1
        
        for i in range(4):
            r = x + self.DIRECT_X[i]
            c = y + self.DIRECT_Y[i]
            if r < 0 or r == len(board) or c < 0 or c == len(board[0]):
                continue
            self.search(root, trie, board, r, c, start_x, start_y)
        board[x][y] = char
```
pass