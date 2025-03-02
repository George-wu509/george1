Lintcode 132
给出一个由小写字母组成的矩阵和一个字典。找出所有同时在字典和矩阵中出现的单词。一个单词可以从矩阵中的任意位置开始，可以向左/右/上/下四个相邻方向移动。一个字母在一个单词中只能被使用一次。且字典中不存在重复单词

|题目|单词匹配方式|字母使用规则|返回值|
|---|---|---|---|
|**Lintcode 132 单词搜索 II**|每个单词独立查找|**每个单词的搜索过程中**，字母只能使用一次，但不同单词搜索时可以重复使用同一字母|返回所有可以找到的单词|
|**Lintcode 1848 单词搜索 III**|需要在矩阵中找到最多数量的单词|**整个矩阵的搜索过程中**，字母只能使用一次（即，一个字母被用在某个单词后，就不能再用于其他单词）|返回找到的最多单词数|

**样例 1:**
```python
"""
输入：["doaf","agai","dcan"]，["dog","dad","dgdg","can","again"]
输出：["again","can","dad","dog"]
解释：
  d o a f
  a g a i
  d c a n
矩阵中查找，返回 ["again","can","dad","dog"]。
```
解釋:
["doaf","agai","dcan"]是二D矩陣. 要尋找字典裏面可以在2D矩陣裡找的到的單詞
譬如 dog: 從d在(0,0)開始->往右o在(0,1)->往下g在(1,1). 所以result裡面有"dog"
譬如 dgdg: 從d在(0,0)or (2,0)開始->往右跟往下(往上)都沒有g , 所以不行

**样例 2:**
```python
"""
输入：["a"],["b"]
输出：[]
解释：
 a
矩阵中查找，返回 []。
```


```python
DIRECTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
class Solution:
    """
    @param board: A list of lists of character
    @param words: A list of string
    @return: A list of string
    """
    def word_search_i_i(self, board, words):
        if board is None or len(board) == 0:
            return []
        
        # pre-process
        # 预处理
        word_set = set(words)
        prefix_set = set()
        for word in words:
            for i in range(len(word)):
                prefix_set.add(word[:i + 1])
        
        result = set()
        for i in range(len(board)):
            for j in range(len(board[0])):
                c = board[i][j]
                self.search(
                    board,
                    i,
                    j,
                    board[i][j],
                    word_set,
                    prefix_set,
                    set([(i, j)]),
                    result,
                )
                
        return list(result)
        
    def search(self, board, x, y, word, word_set, prefix_set, visited, result):
        if word not in prefix_set:
            return
        
        if word in word_set:
            result.add(word)
        
        for delta_x, delta_y in DIRECTIONS:
            x_ = x + delta_x
            y_ = y + delta_y
            
            if not self.inside(board, x_, y_):
                continue
            if (x_, y_) in visited:
                continue
            
            visited.add((x_, y_))
            self.search(
                board,
                x_,
                y_,
                word + board[x_][y_],
                word_set,
                prefix_set,
                visited,
                result,
            )
            visited.remove((x_, y_))
            
    def inside(self, board, x, y):
        return 0 <= x < len(board) and 0 <= y < len(board[0])
```
pass


### Lintcode 132: Word Search II 解法解析

#### 解題思路

這道題目的目標是給定一個字母棋盤 (`board`) 和一組單詞 (`words`)，我們需要在棋盤上找到所有能夠被組成的單詞。每個單詞可以從棋盤上的任何位置開始，並且可以沿著四個方向 (上、下、左、右) 移動，但不能走回已經訪問過的格子。

基於這個問題的特性，直接用暴力搜尋所有單詞的方式效率非常低，因此我們需要更有效率的方法來解決它。本解法的核心思想是 **字典樹 (Trie) + 深度優先搜尋 (DFS)**，我們詳細分析這種方法的思路和時間複雜度。

---

### 解法解析 (字典樹 + DFS)

本解法不是直接暴力檢查 `words` 中的每個單詞，而是先進行 **前綴優化 (prefix pruning)**，利用兩個集合：

1. `word_set`：存放所有的完整單詞，讓我們快速判斷某個組成的字串是否是一個有效單詞。
2. `prefix_set`：存放 `words` 中所有可能的前綴，讓我們在 DFS 的過程中，如果某個字串不是 `prefix_set` 的一部分，我們就可以立即剪枝 (prune)，不用繼續探索這條路徑。

這樣做的好處：

- **減少無效的搜尋空間**：例如，如果我們的 `words` 裡沒有任何單詞是 `xy` 開頭的，那麼當我們 DFS 走到 `xy` 這個字串時，我們就可以立即剪枝，避免無謂的搜索。
- **加快判斷是否為目標單詞**：直接使用 `word_set` 來查找，時間複雜度是 O(1)O(1)O(1)。

#### 具體步驟

1. **預處理 (`preprocess`)**：
    
    - 使用 `word_set` 記錄 `words` 中所有完整單詞，方便後續檢查。
    - 使用 `prefix_set` 記錄 `words` 中所有可能的前綴，避免搜尋無效的分支。
2. **遍歷棋盤的每一個字母，作為 DFS 搜索的起點**：
    
    - 每次從某個字母開始，進行 DFS 搜索。
    - 透過 `visited` 記錄已經訪問過的座標，避免重複使用同一個字母。
3. **深度優先搜索 (`DFS`)**：
    
    - 若目前拼接的 `word` 不在 `prefix_set`，直接剪枝返回。
    - 若目前拼接的 `word` 在 `word_set`，則記錄該單詞。
    - 遍歷四個方向 (上、下、左、右) 繼續搜索。
4. **返回結果**：
    
    - 用 `set` 去重，然後轉成列表返回。

---

### 具體範例解析

#### **範例輸入**

python

複製編輯

`board = [     ['o', 'a', 'a', 'n'],     ['e', 't', 'a', 'e'],     ['i', 'h', 'k', 'r'],     ['i', 'f', 'l', 'v'] ] words = ["oath", "pea", "eat", "rain"]`

#### **預處理**

- `word_set = {"oath", "pea", "eat", "rain"}`
- `prefix_set = {"o", "oa", "oat", "oath", "p", "pe", "pea", "e", "ea", "eat", "r", "ra", "rai", "rain"}`

#### **遍歷棋盤起點**

- `board[0][0] = 'o'`，符合 `prefix_set`，開始 DFS 搜索。
- `board[0][1] = 'a'`，符合 `prefix_set`，繼續搜索…
- `board[0][2] = 'a'`，剪枝 (因為 "oaa" 不在 `prefix_set`)。

最後找到的單詞是：

python

複製編輯

`["oath", "eat"]`

---

### **時間與空間複雜度分析**

假設 `m` 為 `board` 的行數，`n` 為 `board` 的列數，`k` 為 `words` 中的單詞數，`L` 為單詞的最長長度。

1. **構建 `prefix_set` 與 `word_set`**
    
    - `prefix_set` 需要處理 `k` 個單詞，每個單詞最多 `L` 個前綴，所以複雜度是： O(kL)O(kL)O(kL)
2. **遍歷棋盤的所有起點進行 DFS**
    
    - 最差情況下，每個點都可能展開四個方向的 DFS，並且最長到 `L` 個深度： O(mn⋅4L)O(mn \cdot 4^L)O(mn⋅4L)
    - 但因為 `prefix_set` 提供了剪枝，實際上 DFS 不會展開到最差情況，因此實際運行時間通常遠低於這個理論上限。
3. **空間複雜度**
    
    - `prefix_set` 和 `word_set` 需要存 `kL` 個字串，最差情況是： O(kL)O(kL)O(kL)
    - `visited` 需要存儲當前搜尋路徑，最差需要 `O(L)` 空間。
    - `result` 需要 `O(k)` 存放找到的單詞。
    
    總體來說，空間複雜度約為：
    
    O(kL+L)O(kL + L)O(kL+L)

---

### **其他可能的解法**

1. **Trie (字典樹) + DFS**
    
    - 直接用 Trie 來存 `words`，DFS 時查找 Trie 節點來剪枝。
    - Trie 能夠比 `prefix_set` 提供更快的查找，但額外的 Trie 結構會增加空間需求。
    - **時間複雜度：** O(mn⋅4L)O(mn \cdot 4^L)O(mn⋅4L) (仍然取決於 `L`)
    - **空間複雜度：** O(kL)O(kL)O(kL)
2. **Trie (字典樹) + BFS**
    
    - 與 DFS 類似，但改用 BFS 搜索。
    - DFS 會先深入，而 BFS 會逐層展開，適合尋找較短的單詞。
    - **時間複雜度：** O(mn⋅4L)O(mn \cdot 4^L)O(mn⋅4L) (仍然取決於 `L`)
    - **空間複雜度：** O(kL)O(kL)O(kL)
3. **暴力解法**
    
    - 直接遍歷 `words` 的每個單詞，在 `board` 上嘗試所有可能的起點搜索。
    - **時間複雜度：** O(k⋅mn⋅4L)O(k \cdot mn \cdot 4^L)O(k⋅mn⋅4L)，非常慢，對於 `k` 大時不可行。

---

### **結論**

本解法透過 **前綴剪枝 (prefix pruning) + DFS**，成功減少不必要的搜索，使得搜尋過程更高效。對比其他方法：

- **Trie + DFS** 是最優解，能夠加快搜索但空間需求更高。
- **Trie + BFS** 適合更短的單詞搜索。
- **暴力解法** 適用於 `words` 很少時，但對於大數據量不可行。

**推薦使用：Trie + DFS** 或 **前綴剪枝 + DFS**。