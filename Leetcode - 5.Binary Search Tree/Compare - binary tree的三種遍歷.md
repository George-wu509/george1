
|                             |     |
| --------------------------- | --- |
| [[#### 三種 DFS 走訪的程式碼與原理比較]] |     |
|                             |     |
|                             |     |

#### 三種 DFS 走訪的程式碼與原理比較

![[Pasted image 20261105200007.png]]
##### 1. Pre-order Traversal（前序走訪）：根 $\rightarrow$ 左 $\rightarrow$ 右
```python
def dfs(self, node, result):
    if not node:
        return
    result.append(node.val)      # 1. 根：第一時間記錄自己
    self.dfs(node.left, result)  # 2. 左：深入左子樹
    self.dfs(node.right, result) # 3. 右：深入右子樹
```
為什麼這樣寫：
- 每到達一個新節點，立刻將其存入 `result`，隨後才遞迴處理左右兩側。
- **典型用途：** 複製樹（Tree Cloning）、序列化樹（Tree Serialization）、資料夾目錄樹結構輸出。
##### 2. In-order Traversal（中序走訪）：左 $\rightarrow$ 根 $\rightarrow$ 右
```python
def dfs(self, node, result):
    if not node:
        return
    self.dfs(node.left, result)  # 1. 左：先鑽完左子樹
    result.append(node.val)      # 2. 根：記錄當前節點
    self.dfs(node.right, result) # 3. 右：深入右子樹
```
為什麼這樣寫：
- 先把左子樹全部記錄完，才會回溯到當前節點記錄 `node.val`，最後才往右子樹走。
##### Post-order Traversal（後序走訪）：左 $\rightarrow$ 右 $\rightarrow$ 根
```python
def dfs(self, node, result):
    if not node:
        return
    self.dfs(node.left, result)  # 1. 左：深入左子樹
    self.dfs(node.right, result) # 2. 右：深入右子樹
    result.append(node.val)      # 3. 根：最後記錄自己
```
為什麼這樣寫：
- 必須等到左子樹與右子樹的所有子節點都處理完畢後，才會在最後一行記錄當前節點。
- **典型用途：** 由下而上（Bottom-up）計算問題，例如刪除整個二元樹（先刪子節點再刪父節點）、計算二元樹的高度/深度、計算節點的子樹大小。


##### Binary Search Tree（BST）的遍歷特別規則
**二元搜尋樹（BST）的中序遍歷（In-order Traversal）結果，必定是一個「嚴格遞增的排序數列（Sorted Array）」！**
###### 原因分析
在 BST 的定義中：
- 左子樹中的所有節點值 $<$ 當前節點值
- 右子樹中的所有節點值 $>$ 當前節點值