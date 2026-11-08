lintcode 70
给出一棵二叉树，返回其节点值从底向上的层次序遍历（按从叶节点所在层到根节点所在的层遍历，然后逐层从左往右遍历）


**样例 1：**
输入：
```
tree = {1,2,3}
```
输出：
```
[[2,3],[1]]
```
解释：
```
    1
   / \
  2   3
```
它将被序列化为 {1,2,3}  

**样例 2：**
输入：
```
tree = {3,9,20,#,#,15,7}
```
输出：
```
[[15,7],[9,20],[3]]
```
解释：
```
    3
   / \
  9  20
    /  \
   15   7
```
它将被序列化为 {3,9,20,#,#,15,7}



```python
def level_order_bottom(self, root):
	self.results = []
	if not root:
		return self.results
	q = [root]
	while q:
		new_q = []
		self.results.append([n.val for n in q])
		for node in q:
			if node.left:
				new_q.append(node.left)
			if node.right:
				new_q.append(node.right)
		q = new_q
	return list(reversed(self.results))
```
pass
解釋: 



這段程式碼**就是典型的 BFS（廣度優先搜尋）**，它採用了「按層批次處理（Batch Level Processing）」的技巧，實現了一層一層（Level by Level）自頂向下的掃描，最後再翻轉結果。

### 這是不是 BFS？與傳統寫法差別在哪裡？

1. **這就是標準的 BFS：** BFS 的核心概念是「先被發現的層級先處理（FIFO）」。這段程式碼用 `q` 儲存「當前層的所有節點」，並用 `new_q` 搜集「下一層的所有節點」，完全符合 BFS 層序走訪的本質。
    
2. **與傳統 Queue 寫法的差異：**
    
    - **傳統單佇列寫法**：使用 `collections.deque()`，搭配 `for _ in range(len(queue))` 一個個取出並放入子節點。
        
    - **這段程式碼的寫法**：直接使用 Python 的清單替換（`q = new_q`）。每一輪迴圈處理**整整一整層**節點，寫法極其簡潔。
        
3. **Bottom-Up（自底向上）特徵：** 題目是 LeetCode 107 (_Binary Tree Level Order Traversal II_)，因此最後使用 `reversed(self.results)` 將原本自頂向下的層序結果反轉過來。
    

### 二元樹範例：`tree = [3, 9, 20, null, null, 15, 7]`

依據題目輸入，二元樹的結構如下：

Plaintext

```
       3         <-- Layer 0 (Depth 0)
      / \
     9   20      <-- Layer 1 (Depth 1)
        /  \
       15   7    <-- Layer 2 (Depth 2)
```

### 具體執行過程一步步追蹤

#### 初始化

- `self.results = []`
    
- `q = [Node(3)]`
    

#### 第 1 輪迴圈 (`while q` 成立，此時 `q` 有 1 個節點)

1. `new_q = []`
    
2. `self.results.append([n.val for n in q])` $\rightarrow$ 記錄 Layer 0：**`self.results = [[3]]`**
    
3. 遍歷 `q`（處理 `Node(3)`）：
    
    - `3.left` 為 `Node(9)` $\rightarrow$ `new_q.append(Node(9))`
        
    - `3.right` 為 `Node(20)` $\rightarrow$ `new_q.append(Node(20))`
        
4. **換層**：`q = new_q` $\rightarrow$ `q = [Node(9), Node(20)]`
    

#### 第 2 輪迴圈 (`q` 有 2 個節點)

1. `new_q = []`
    
2. `self.results.append([n.val for n in q])` $\rightarrow$ 記錄 Layer 1：**`self.results = [[3], [9, 20]]`**
    
3. 遍歷 `q`：
    
    - 處理 `Node(9)`：無左右子節點（`#`, `#`）。
        
    - 處理 `Node(20)`：
        
        - `20.left` 為 `Node(15)` $\rightarrow$ `new_q.append(Node(15))`
            
        - `20.right` 為 `Node(7)` $\rightarrow$ `new_q.append(Node(7))`
            
4. **換層**：`q = new_q` $\rightarrow$ `q = [Node(15), Node(7)]`
    

#### 第 3 輪迴圈 (`q` 有 2 個節點)

1. `new_q = []`
    
2. `self.results.append([n.val for n in q])` $\rightarrow$ 記錄 Layer 2：**`self.results = [[3], [9, 20], [15, 7]]`**
    
3. 遍歷 `q`：
    
    - 處理 `Node(15)`：無左右子節點。
        
    - 處理 `Node(7)`：無左右子節點。
        
4. **換層**：`q = new_q` $\rightarrow$ `q = []`
    

#### 迴圈結束與結果翻轉

- `q` 為空清單 `[]`，退出 `while` 迴圈。
    
- 執行 `list(reversed(self.results))`：
    
    - 原本自頂向下：`[[3], [9, 20], [15, 7]]`
        
    - 翻轉後輸出（自底向上）：**`[[15, 7], [9, 20], [3]]`**