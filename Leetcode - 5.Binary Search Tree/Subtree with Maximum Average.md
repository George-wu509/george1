
**样例1**
```
输入：
{1,-5,11,1,2,4,-2}
输出：11
说明:
这棵树如下所示：
     1
   /   \
 -5     11
 / \   /  \
1   2 4    -2 
11子树的平均值是4.333，为最大的。
```
**样例2**
```
输入：
{1,-5,11}
输出：11
说明:
     1
   /   \
 -5     11
1,-5,11 三棵子树的平均值分别是 2.333,-5,11。因此11才是最大的
```


```python
class Solution:
    ans_average, ans_node = float('-inf'), None 
    def find_subtree2(self, root: TreeNode) -> TreeNode:
        self.dfs(root)
        return self.ans_node
        
    def dfs(self, node):
        if not node:
            return 0, 0
        left_sum, left_n = self.dfs(node.left)
        right_sum, right_n = self.dfs(node.right)

        total_n = left_n + right_n + 1
        total_sum = left_sum + right_sum + node.val
        if total_sum / total_n > self.ans_average:
            self.ans_average = total_sum / total_n
            self.ans_node = node

        return total_sum, total_n
```
pass
```
     1
   /   \
 -5     11
 / \   /  \
1   2 4    -2 
```
解釋: root = [1,-5,11,1,2,4,-2]為例
step1: 使用Post-order traversal, 因為要計算和/平均要從下往上加
step2: 從左下1子樹開始計算sum(1)跟個數(1)
step3: 接著左下2子樹開始計算sum(2)跟個數(1)
step4: 然後就可以計算到左下-5子樹sum = -5+1+2, 個數=3




這道 LeetCode 題目「**Subtree with Maximum Average**」（找出二叉樹中具有最大平均數的子樹）的核心精神是：**如何有效率地計算出每一個子樹的總和與節點個數，並找出平均值最大者。**
##### 一、 解題核心概念

1. **為什麼使用後序走訪 (Post-order Traversal / Bottom-Up)？**
    - 如果從根節點由上往下計算，我們需要重複走訪子樹多次來加總數值，時間複雜度會劣化到 $O(N^2)$。
    - 透過**遞迴（DFS）由下往上**處理，每個節點只要向它的左右子樹各要兩個資訊：
        - **`sum` (子樹節點總和)**
        - **`n` (子樹節點個數)**
    - 拿到左右子樹的數據後，再加上當前節點的值，就能以 $O(1)$ 時間計算出當前子樹的平均值（`total_sum / total_n`），並與全域最大值比較。
        
2. **時間與空間複雜度**
    - **時間複雜度**：$O(N)$，每個節點只被走訪一次。
    - **空間複雜度**：$O(H)$，其中 $H$ 為樹的高度，主要由遞迴呼叫的系統堆疊（Call Stack）所佔用。

##### 二、 範例圖解 `root = [1, -5, 11, 1, 2, 4, -2]`

讓我們先將陣列還原成二叉樹的結構圖：
```
       ( 1 )        <-- 根節點
      /     \
   (-5)     (11)
   /   \    /   \
 (1)   (2) (4)  (-2)
```
#### 逐步追蹤 DFS 計算過程：
1. **左子樹底部的葉子節點 (`1` 與 `2`)**
    - 節點 `1`：回傳 `sum = 1`, `n = 1`。平均值 = $1 / 1 = 1.0
    - 節點 `2`：回傳 `sum = 2`, `n = 1`。平均值 = $2 / 1 = 2.0$
        
2. **子樹根節點 `-5`**
    - 接收左子樹 `(1, 1)` 與右子樹 `(2, 1)`。
    - `total_sum = 1 + 2 + (-5) = -2`
    - `total_n = 1 + 1 + 1 = 3`
    - 平均值 = $-2 / 3 \approx -0.67$（更新全域最大值：目前最大為 `-0.67`，對應節點 `-5`）
    - 回傳 `-2, 3` 給上層。
3. **右子樹底部的葉子節點 (`4` 與 `-2`)**
    - 節點 `4`：回傳 `sum = 4`, `n = 1`。平均值 = $4.0$
    - 節點 `-2`：回傳 `sum = -2`, `n = 1`。平均值 = $-2.0$
4. **子樹根節點 `11`**
    - 接收左子樹 `(4, 1)` 與右子樹 `(-2, 1)`。
    - `total_sum = 4 + (-2) + 11 = 13`
    - `total_n = 1 + 1 + 1 = 3`
    - 平均值 = $13 / 3 \approx \mathbf{4.33}$ **（大於目前的 `-0.67`！更新全域最大值：平均值 `4.33`，對應節點 `11`）**
    - 回傳 `13, 3` 給上層。
5. **根節點 `1`**
    - 接收左子樹 `-5` 的數據 `(-2, 3)` 與右子樹 `11` 的數據 `(13, 3)`。
    - `total_sum = (-2) + 13 + 1 = 12`
    - `total_n = 3 + 3 + 1 = 7`
    - 平均值 = $12 / 7 \approx 1.71$（小於現行的 `4.33`，不更新）。
    - 回傳 `12, 7`。
**最終結果**：返回節點 `11`，其子樹平均值最高（約 `4.33`）。

##### 三、 程式碼關鍵邏輯解析

```Python
def dfs(self, node):
    if not node:
        return 0, 0
    # 1. 取得左右子樹的總和與節點數
    left_sum, left_n = self.dfs(node.left)
    right_sum, right_n = self.dfs(node.right)

    # 2. 計算包含當前節點在內的子樹總和與總個數
    total_n = left_n + right_n + 1
    total_sum = left_sum + right_sum + node.val
    
    # 3. 檢查並更新最大平均數與對應的節點
    if total_sum / total_n > self.ans_average:
        self.ans_average = total_sum / total_n
        self.ans_node = node

    # 4. 將當前子樹的總和與個數往上回傳給父節點
    return total_sum, total_n
```
