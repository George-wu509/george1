
從前序與中序遍歷序列構造二元樹 Construct Binary Tree from Preorder and Inorder Traversal

LeetCode 105 / LintCode 73 給定兩個整數陣列 `preorder` 和 `inorder`，其中 `preorder` 是二元樹的**前序遍歷**，`inorder` 是同一棵樹的**中序遍歷**，請構造並返回這棵二元樹。 （假設樹中沒有重複的元素。）

**样例 1：** 输入：
```Python
"""
preorder = [3, 9, 20, 15, 7]
inorder  = [9, 3, 15, 20, 7]
"""
```

输出：
```Python
"""
[3, 9, 20, None, None, 15, 7]
    3
   / \
  9  20
    /  \
   15   7
"""
```

解释： 前序第一個元素 `3` 為根節點；在中序中找到 `3`，其左側 `[9]` 為左子樹，右側 `[15, 20, 7]` 為右子樹，再遞迴構造左右子樹即可。

**样例 2：** 输入：
```Python
"""
preorder = [-1]
inorder  = [-1]
"""
```

输出：
```Python
"""
[-1]
"""
```

解释： 只有一個根節點 `-1`，沒有左右子樹。

**样例 3：** 输入：
```Python
"""
preorder = [1, 2, 3]
inorder  = [3, 2, 1]
"""
```

输出：
```Python
"""
[1, 2, None, 3]
    1
   /
  2
 /
3
"""
```

解释： 根節點 `1` 在中序的最右側，代表沒有右子樹，所有剩餘節點都在左子樹中，形成向左傾斜的單邊樹。

```Python
from typing import List, Optional

class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        # 建立雜湊表，O(1) 查詢節點值在 inorder 中的索引位置
        inorder_index_map = {val: idx for idx, val in enumerate(inorder)}
        self.pre_idx = 0  # 全局指標，依序指向 preorder 中的當前根節點

        def helper(left: int, right: int) -> Optional[TreeNode]:
            # 區間無效，代表該子樹為空
            if left > right:
                return None

            # 從 preorder 取出當前子樹的根節點，並將指標後移
            root_val = preorder[self.pre_idx]
            self.pre_idx += 1
            root = TreeNode(root_val)

            # 在 inorder 中找到根節點的位置，切分左右子樹區間
            mid = inorder_index_map[root_val]

            # 注意：因為 preorder 是「根 -> 左 -> 右」，必須先構造左子樹，再構造右子樹
            root.left = helper(left, mid - 1)
            root.right = helper(mid + 1, right)

            return root

        return helper(0, len(inorder) - 1)
```


```Python
from typing import List, Optional

class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        if not preorder or not inorder:
            return None

        # 前序遍歷的第一個元素永遠是當前子樹的根節點
        root_val = preorder[0]
        root = TreeNode(root_val)

        # 找出根節點在中序遍歷中的索引，該索引同時也等於「左子樹的節點數量」
        mid = inorder.index(root_val)

        # 利用陣列切片遞迴構造左右子樹
        root.left = self.buildTree(preorder[1 : mid + 1], inorder[:mid])
        root.right = self.buildTree(preorder[mid + 1 :], inorder[mid + 1 :])

        return root
```

解釋 (雜湊表 + 雙指標遞迴解法): step1: 預先將 `inorder` 的 `{節點值: 索引}` 存入雜湊表 `inorder_index_map`，避免每次遞迴都要線性搜尋 step2: 維護一個遞增指標 `self.pre_idx = 0`，按照 `preorder` 的順序逐一建立根節點 step3: 假設 `preorder = [3, 9, 20, 15, 7]`, `inorder = [9, 3, 15, 20, 7]`，初始呼叫 `helper(0, 4)` step4: `pre_idx = 0` 取出 `3`，在 `inorder` 中索引 `mid = 1`，建立 `TreeNode(3)`，切出左區間 `[0, 0]` 與右區間 `[2, 4]` step5: 遞迴左子樹 `helper(0, 0)`：`pre_idx = 1` 取出 `9` (`mid = 0`)，其左右區間皆無效 (`left > right`)，回傳 `TreeNode(9)` step6: 遞迴右子樹 `helper(2, 4)`：`pre_idx = 2` 取出 `20` (`mid = 3`)，依序構造出左子節點 `15` (`[2, 2]`) 與右子節點 `7` (`[4, 4]`)，回傳 `TreeNode(20)` step7: 將 `9` 與 `20` 接回 `3` 的左右子節點，最終回傳根節點 `3`

解釋 (核心觀念):**前序遍歷負責「定位根節點」，中序遍歷負責「切分左右子樹」**：

- **前序 (`根 -> 左 -> 右`)**：第一個元素必定是當前整棵樹（或子樹）的根節點。
    
- **中序 (`左 -> 根 -> 右`)**：一旦知道根節點的位置 `mid`，`mid` 左邊的所有元素就屬於左子樹，`mid` 右邊的所有元素就屬於右子樹，且左子樹的節點總數剛好為 `mid - left`。
    

### **LeetCode 105 - Construct Binary Tree from Preorder and Inorder Traversal（從前序與中序遍歷序列構造二元樹）**

這道題目是 Blind 75 與 NeetCode 150 中最具代表性的 **分治法（Divide and Conquer）** 與 **樹構造** 經典題。題目考驗對二元樹遍歷順序特性的深刻理解，以及如何利用索引邊界將時間複雜度從 $O(N^2)$ 優化至 $O(N)$。

## **解法解析**

這道題可以轉化為 **利用前序確定根節點、利用中序計算子樹大小並切分區間** 的遞迴分治問題。

### **思路**

1. **觀察兩種遍歷的結構特性**
    
    - **前序遍歷 (`preorder`)**：`[ 根節點 | 左子樹所有節點 | 右子樹所有節點 ]`
        
    - **中序遍歷 (`inorder`)**：`[ 左子樹所有節點 | 根節點 | 右子樹所有節點 ]`
        
    - 由於題目保證樹中**沒有重複元素**，我們可以精確地透過值在 `inorder` 中找到唯一的根節點位置 `mid`。
        
2. **為什麼直接切片 (`preorder[1:mid+1]`) 較慢？**
    
    - 直觀寫法每次遞迴都呼叫 `inorder.index(root_val)` 尋找位置，單次耗時 $O(N)$；同時陣列切片 `preorder[...]` 也會產生額外的空間複製開銷，導致整體時間與空間複雜度退化為 $O(N^2)$。
        
    - **優化策略**：使用 `HashMap` 預存 `inorder` 的值與索引映射，並用左右邊界指標 `(left, right)` 原地限定 `inorder` 的有效範圍。
        
3. **代碼執行流程（全局 `pre_idx` 技巧）**
    
    - 因為前序遍歷的訪問順序本來就是「根 → 左 → 右」，只要我們在遞迴時**嚴格先建構左子樹 `helper(left, mid - 1)`，再建構右子樹 `helper(mid + 1, right)`**，那麼每次遞迴進入時，`preorder[self.pre_idx]` 就剛好會是當前子樹的根節點。
        
    - 當區間縮減到 `left > right` 時，代表該側已經沒有節點，直接回傳 `None`。
        

## **具體舉例**

假設輸入為範例 1：`preorder = [3, 9, 20, 15, 7]`，`inorder = [9, 3, 15, 20, 7]`，遞迴呼叫過程如下：

|**遞迴呼叫**|**inorder 區間 [left, right]**|**當前 pre_idx**|**取出根節點 root_val**|**inorder 中位置 mid**|**切分出的左 / 右子樹區間**|
|---|---|---|---|---|---|
|第 1 層|`[0, 4]` (`[9, 3, 15, 20, 7]`)|`0`|`3`|`1`|左 `[0, 0]`，右 `[2, 4]`|
|第 2 層（左）|`[0, 0]` (`[9]`)|`1`|`9`|`0`|左 `[0, -1]` (空)，右 `[1, 0]` (空)|
|第 2 層（右）|`[2, 4]` (`[15, 20, 7]`)|`2`|`20`|`3`|左 `[2, 2]`，右 `[4, 4]`|
|第 3 層（右-左）|`[2, 2]` (`[15]`)|`3`|`15`|`2`|左 `[2, 1]` (空)，右 `[3, 2]` (空)|
|第 3 層（右-右）|`[4, 4]` (`[7]`)|`4`|`7`|`4`|左 `[4, 3]` (空)，右 `[5, 4]` (空)|

最終組裝完成的樹結構為 `[3, 9, 20, None, None, 15, 7]`。

## **時間與空間複雜度分析**

- **時間複雜度：O(N)**
    
    - 其中 `N` 為二元樹的節點總數。建立 `inorder` 雜湊表需要 $O(N)$ 時間；後續遞迴每個節點只會被建立並訪問一次，雜湊表查詢與指標移動皆為 $O(1)$，因此總時間複雜度為 **O(N)**。（若使用陣列切片 + `.index()` 寫法則為 **O(N²)**）。
        
- **空間複雜度：O(N)**
    
    - 雜湊表 `inorder_index_map` 需要儲存 `N` 個鍵值對，佔用 $O(N)$ 空間；遞迴呼叫棧深度取決於樹高 `H`（平衡樹為 $O(\log N)$，最壞斜樹為 $O(N)$）。兩者相加總空間複雜度為 **O(N)**。
        

## **其他解法**

### **1. 迭代法（單調棧 Stack 模擬前序遍歷）**

- **思路**：
    
    - 觀察 `preorder` 中相鄰兩個元素 `preorder[i]` 和 `preorder[i+1]` 的關係：只有兩種可能——要麼 `preorder[i+1]` 是 `preorder[i]` 的**左子節點**；要麼 `preorder[i]` 沒有左子樹，`preorder[i+1]` 是 `preorder[i]` 或其某個祖先節點的**右子節點**。
        
    - 維護一個棧 `stack` 存放尚未找到右子樹的節點，並用指標 `inorder_idx` 指向 `inorder` 當前最左下角的節點。當 `stack[-1].val == inorder[inorder_idx]` 時，不斷彈棧並移動 `inorder_idx`，最後彈出的節點即為下一個新節點的父節點（接在右側）。
        
- **時間複雜度**：O(N)
    
- **空間複雜度**：O(H)，最壞 O(N)（省去了雜湊表的額外常數空間）。
    

### **2. 傳遞完整四邊界指標的遞迴（無全局變數）**

- **思路**：
    
    - 不使用 `self.pre_idx`，而是在遞迴函數中同時傳入 `pre_left, pre_right, in_left, in_right` 四個參數。
        
    - 透過計算左子樹長度 `left_size = mid - in_left`，精確推算出前序陣列的左子樹範圍 `[pre_left + 1, pre_left + left_size]` 與右子樹範圍 `[pre_left + left_size + 1, pre_right]`。這樣寫為純函數（Pure Function），更利於平行化或無副作用設計。
        
- **時間複雜度**：O(N)
    
- **空間複雜度**：O(N)
    

## **總結**

- **面試最佳實踐**：**HashMap + 全局 `pre_idx` + `inorder` 左右雙指標**，兼具 $O(N)$ 最優時間複雜度與極高的代碼可讀性，不容易算錯前序陣列的偏移量。
    
- **核心前提提醒**：唯有**包含中序遍歷**（前序+中序 或 後序+中序）才能唯一確定一棵普通二元樹；若只有「前序 + 後序」，當某個節點只有單一子節點時，無法區分它是左子樹還是右子樹（除非題目限定為滿二元樹 Full Binary Tree）。
    

## **二元樹構造 / 序列化類問題整理**

|**題號**|**題目名稱**|**問題類型**|**解法**|
|---|---|---|---|
|**LeetCode 105**|Construct Binary Tree from Preorder and Inorder Traversal|**由前序 + 中序還原二元樹**|**HashMap + 分治遞迴**：前序首元素定根，中序切分左右區間|
|**LeetCode 106**|Construct Binary Tree from Inorder and Postorder Traversal|**由中序 + 後序還原二元樹**|**HashMap + 分治遞迴**：後序末元素定根，注意**必須先建右子樹再建左子樹**|
|**LeetCode 889**|Construct Binary Tree from Preorder and Postorder Traversal|**由前序 + 後序還原二元樹（多解回傳其一）**|**分治遞迴**：用 `preorder[1]` 作為左子樹的根，在 `postorder` 中尋找其位置來切分大小|
|**LeetCode 1008**|Construct Binary Search Tree from Preorder Traversal|**僅由前序遍歷還原二元搜尋樹（BST）**|**上下界限制遞迴 $O(N)$**：利用 BST 大小性質取代中序遍歷，傳遞 `(lower, upper)` 邊界|
|**LeetCode 108**|Convert Sorted Array to Binary Search Tree|**將升序陣列轉為高度平衡的 BST**|**二分分治遞迴**：每次取區間中點 `mid` 作為根節點，左右半邊遞迴建樹|
|**LeetCode 297**|Serialize and Deserialize Binary Tree（二元樹的序列化與反序列化）|**將二元樹編碼為字串並還原**|**前序 DFS（帶空節點 `#` 標記）** 或 **BFS 層序遍歷**，使用 `deque` 逐一消費節點|
|**LeetCode 654**|Maximum Binary Tree（最大二元樹）|**陣列最大值為根，左右子陣列遞迴建樹**|**單調遞減棧 $O(N)$** 或 **分治遞迴 $O(N^2)$**|

### **延伸變形題：從中序與後序遍歷序列構造二元樹（Construct Binary Tree from Inorder and Postorder Traversal - LeetCode 106）**

給定兩個整數陣列 `inorder` 和 `postorder`，其中 `inorder` 是二元樹的中序遍歷，`postorder` 是同一棵樹的後序遍歷，請構造並返回這棵二元樹。

Python

```
from typing import List, Optional

class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        inorder_index_map = {val: idx for idx, val in enumerate(inorder)}
        # 後序遍歷順序為「左 -> 右 -> 根」，因此從最後一個元素開始往前取根節點
        self.post_idx = len(postorder) - 1

        def helper(left: int, right: int) -> Optional[TreeNode]:
            if left > right:
                return None

            root_val = postorder[self.post_idx]
            self.post_idx -= 1
            root = TreeNode(root_val)

            mid = inorder_index_map[root_val]

            # 關鍵差異：後序倒著看是「根 -> 右 -> 左」，因此必須先構造右子樹，再構造左子樹！
            root.right = helper(mid + 1, right)
            root.left = helper(left, mid - 1)

            return root

        return helper(0, len(inorder) - 1)
```