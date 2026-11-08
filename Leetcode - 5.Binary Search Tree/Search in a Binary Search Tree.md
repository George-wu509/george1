
**样例 1:**
```
输入: value = 2
        4
       / \
      2   7
     / \
    1   3
输出: 节点 2
```
**样例 2:**
```
输入: value = 5
        4
       / \
      2   7
     / \
    1   3
输出: null
```


```python
class Solution:

    def search_b_s_t(self, root: TreeNode, val: int) -> TreeNode:
        while root:
            if root.val == val:
                return root
            root = root.left if root.val > val else root.right
        
        return None
```
pass
解釋:
step1: 在search_bst function裡面, while迴圈內檢查root存在
step2: 在while root迴圈, val是目標則return root <-答
step3: 在while root迴圈, 如果root.val大於目標root=root.left
step4: 在while root迴圈, 如果root.val小於目標root=root.right


二叉搜尋樹（Binary Search Tree, BST）搜尋的核心思想是**利用有序性進行方向選擇**：對於任意節點，其左子樹的所有節點值都嚴格小於該節點值，右子樹的所有節點值都嚴格大於該節點值。因此每次比較都能直接排除半邊子樹，將搜尋規模縮減一半。
![[Pasted image 20261108130113.png|370]]

##### 核心 Idea 與時間複雜度

1. **比較與剪枝**：當前節點值如果比目標值 `val` 大，目標必定落在**左子樹**；反之則在**右子樹**。
2. **終止條件**：找到節點值等於 `val` 即返回；若走到空節點 `None`，說明樹中不存在該值。
3. **複雜度分析**：
    - **時間複雜度**：平均為 $O(\log N)$（樹的高度的度數），最壞情況樹退化成單鏈表時為 $O(N)$。
    - **空間複雜度**：迭代解法僅維護一個指針，為 $O(1)$。
##### 程式碼逐行拆解
```Python
class Solution:
    def search_b_s_t(self, root, val):
        while root:
# 1. 若當前節點值等於目標值，直接返回該節點（即該節點為根的子樹）
            if root.val == val:
                return root
            
# 2. 三元運算子判斷搜尋方向：
# 若 root.val > val，說明目標在左邊，指針指向 root.left
# 否則說明 root.val < val，目標在右邊，指針指向 root.right
            root = root.left if root.val > val else root.right
        
# 3. while 結束代表走到了 None，說明目標不存在
        return None
```
##### 具體範例追蹤：`root = [4, 2, 7, 1, 3]`, `val = 2`
輸入的二叉搜尋樹結構如下：
```
       4 (root)
      / \
     2   7
    / \
   1   3
```
###### 執行步驟推導：

| **步驟**      | **當前 root.val** | **比較邏輯 (val = 2)** | **執行動作**                       | **指針下一步位置** |
| ----------- | --------------- | ------------------ | ------------------------------ | ----------- |
| **第 1 次迴圈** | `4`             | `4 != 2` 且 `4 > 2` | 執行 `root = root.left`          | 移動到節點 `2`   |
| **第 2 次迴圈** | `2`             | `2 == 2`（命中目標）     | 觸發 `if root.val == val` 返回當前節點 | 回傳節點 `2`    |
###### 執行路徑與結果圖解：
```
Step 1: 檢查根節點 4 (4 > 2，向左走)
       [4]  <-- 檢查此節點
      /   \
    (2)    7
    / \
   1   3

Step 2: 檢查節點 2 (2 == 2，找到目標！回傳該節點及其子樹)
        4
       / \
     [2]  7  <-- 命中並返回此子樹
     / \
    1   3
```
最終回傳的結果是以節點 `2` 為根的完整子樹：
```
    2
   / \
  1   3
```
