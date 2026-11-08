

**样例 1：**
输入：
```
tree = {1}
A = 1
B = 1
```
输出：
```
1
```
解释：
```
二叉树如下（只有一个节点）:
        1
LCA(1,1) = 1
```

**样例 2：**
输入：
```
tree = {4,3,7,#,#,5,6}
A = 3
B = 5
```
输出：
```
4
```
解释：
```
二叉树如下:

    4
   / \
  3   7
     / \
    5   6
                        
LCA(3, 5) = 4
```


```python
def lowestCommonAncestor(self, root, A, B):
	if root is None:
		return None
	
	if root is A or root is B:
		return root

	left_result = self.lowestCommonAncestor(root.left, A, B)
	right_result = self.lowestCommonAncestor(root.right, A, B)

	if left_result and right_result:
		return root

	if left_result:
		return left_result

	if right_result:
		return right_result

	return None
```
pass
解釋:
step1: create lowestCommonAncestor(LCA) function並從根開始走
step2:  LCA function裡如果root是A or B直接output node
step3: 在LCA function裡不是root, 則接著對root.left跑LCA function
step3: 在LCA function裡不是root, 則接著對root.right跑LCA function
step3:如果left_result跟right_result都存在, 則return root <-答案
step4: 如果只有left_result存在, 則return left_result <-答案
step4: 如果只有right_result存在, 則return right_result <-答案


二元樹最近公共祖先（Lowest Common Ancestor, LCA）的求解核心在於「自底向上的後序遍歷（Post-order Traversal: 左 $\rightarrow$ 右 $\rightarrow$ 根）」。透過深度優先搜尋（DFS）一路探底至葉子節點，再利用「回溯」將尋找結果向上傳遞給父節點。
![[Pasted image 20261108122205.png|337]]

## 演算法核心思想與邏輯

遞迴函式 `lowestCommonAncestor(root, A, B)` 的回傳值具有以下三種明確含義：

1. **傳回 $A$ 或 $B$**：代表在當前子樹中**只找到了目標節點 $A$ 或 $B$ 的其中一個**。
2. **傳回當前 `root`**：代表 $A$ 與 $B$ **分別落在當前 `root` 的左右兩棵子樹中**，因此當前 `root` 就是兩者的最近公共祖先。
3. **傳回 `None`**：代表當前子樹中**完全沒有發現 $A$ 或 $B$**。

##### 實例演練：`tree = {4,3,7,#,#,5,6}`

陣列格式 `{4,3,7,#,#,5,6}` 對應的二元樹結構如下：
```
       4
     /   \
    3     7
         / \
        5   6
```

##### 範例 1：尋找 $A = 5, B = 6$ 的 LCA

1. **從根節點 `4` 開始 DFS**：
    - 先遞迴左子樹 `3`：`3` 不是 $5$ 或 $6$，且無子節點，回傳 `None`（`left_result = None`）。
    - 再遞迴右子樹 `7`：
        - 節點 `7` 的左子樹 `5`：觸發 `root is A`，回傳 `5`。
        - 節點 `7` 的右子樹 `6`：觸發 `root is B`，回傳 `6`。
        - 節點 `7` 收到 `left_result = 5` 且 `right_result = 6`。
        - 觸發 `if left_result and right_result`，**節點 `7` 回傳自身 `7` 向上給根節點 `4`**。
2. **根節點 `4` 進行最終彙整**：
    - `left_result`（來自 `3`）$= None$
    - `right_result`（來自 `7`）$= 7$
    - 觸發 `if right_result`，回傳 `7`。

**最終結果**：LCA 為 **`7`**。

##### 範例 2：尋找 $A = 3, B = 5$ 的 LCA
```
       4 (left_result=3, right_result=5 -> 回傳 4)
     /   \
(找到3)3   7 (left_result=5, right_result=None -> 回傳 5)
         / \
  (找到5)5   6 (回傳 None)
```

1. **根節點 `4` 開始**：
    - 遞迴左子樹 `3`：觸發 `root is A`，直接回傳 `3`（`left_result = 3`）。
    - 遞迴右子樹 `7`：
        - 左子樹 `5` 回傳 `5`。
        - 右子樹 `6` 回傳 `None`。
        - 節點 `7` 觸發 `if left_result`，將 `5` 向上傳遞給 `4`（`right_result = 5`）。
2. **根節點 `4` 進行最終彙整**：
    - `left_result` $= 3$
    - `right_result` $= 5$
    - 左右皆非 `None`，觸發 `if left_result and right_result`，**回傳根節點 `4`**。

**最終結果**：LCA 為 **`4`**。
```python
def lowestCommonAncestor(self, root, A, B):
    # 1. 終止條件：到達空節點，代表沒找到，回傳 None
    if root is None:
        return None
    
    # 2. 終止條件：若當前節點就是 A 或 B，直接回傳當前節點
    # （不需要再深入該節點的子樹搜尋）
    if root is A or root is B:
        return root

    # 3. 遞迴探尋：分別在左子樹與右子樹尋找 A 與 B
    left_result = self.lowestCommonAncestor(root.left, A, B)
    right_result = self.lowestCommonAncestor(root.right, A, B)

    # 4. 回溯判斷（向上彙整結果）：
    
    # 情況 A：左右子樹各自找到了一個目標節點
    # 代表 A、B 分居兩側，當前 root 就是兩者的最近公共祖先
    if left_result and right_result:
        return root

    # 情況 B：只有左子樹回傳結果，將左子樹的結果繼續向上傳遞
    if left_result:
        return left_result

    # 情況 C：只有右子樹回傳結果，將右子樹的結果繼續向上傳遞
    if right_result:
        return right_result

    # 情況 D：左右子樹都沒有找到目標
    return None
```