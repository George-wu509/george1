
LintCode 367 
表达树是一个二叉树的结构，用于衡量特定的表达。所有表达树的叶子都有一个数字字符串值。而所有表达树的非叶子都有一个操作字符串值。

给定一个表达数组，请构造该表达的表达树，并返回该表达树的根。


---

### **題目分析**

表達樹是一種二叉樹，其中每個葉子節點表示一個操作數，每個內部節點表示一個操作符。輸入是一個中序表達式（中序：操作數和操作符按照計算順序排列），我們需要構造對應的表達樹。

Example:
```python
输入: ["2","*","6","-","(","23","+","7",")","/","(","1","+","2",")"]
输出: {[-],[*],[/],[2],[6],[+],[+],#,#,#,#,[23],[7],[1],[2]} 
解释:
其表达树如下：

	                 [ - ]
	             /          \
	        [ * ]              [ / ]
	      /     \           /         \
	    [ 2 ]  [ 6 ]      [ + ]        [ + ]
	                     /    \       /      \
	                   [ 23 ][ 7 ] [ 1 ]   [ 2 ] .

在构造该表达树后，你只需返回根节点`[-]`。
```

```python
输入: ["10","+","(","3","-","5",")","+","7","*","7","-","2"]
输出: {[-],[+],[2],[+],[*],#,#,[10],[-],[7],[7],#,#,[3],[5]}
解释:
其表达树如下：
 	                       [ - ]
	                   /          \
	               [ + ]          [ 2 ]
	           /          \      
	       [ + ]          [ * ]
              /     \         /     \
	  [ 10 ]  [ - ]    [ 7 ]   [ 7 ]
	          /    \ 
                [3]    [5]
在构造该表达树后，你只需返回根节点`[-]`。
```



```python
from typing import (
    List,
)
from lintcode import (
    ExpressionTreeNode,
)
import sys
class MyTreeNode:
    def __init__(self, val, s):
        self.left = None
        self.right = None
        self.val = val
        self.exp_node = ExpressionTreeNode(s)

class Solution:
    # @param expression: A string list
    # @return: The root of expression tree
    def get_val(self, a, base):
        if a == '+' or a == '-':
            if base == sys.maxsize:
                return base
            return 1 + base
        if a == '*' or a == '/':
            if base == sys.maxsize:
                return base
            return 2 + base
        return sys.maxsize

    def build(self, expression):
        # write your code here
        root = self.create_tree(expression)
        return self.copy_tree(root)
    
    def copy_tree(self, root):
        if not root:
            return None
        root.exp_node.left = self.copy_tree(root.left)
        root.exp_node.right = self.copy_tree(root.right)
        return root.exp_node
        
    
    def create_tree(self, expression):
        stack = []
        base = 0
        for i in range(len(expression)):
            if i != len(expression):
                if expression[i] == '(':
                    if base != sys.maxsize:
                        base += 10
                    continue
                elif expression[i] == ')':
                    if base != sys.maxsize:
                        base -= 10
                    continue
                val = self.get_val(expression[i], base)
    
            node = MyTreeNode(val, expression[i])
            while stack and val <= stack[-1].val:
                node.left = stack.pop()
            if stack:
                stack[-1].right = node
            stack.append(node)
        if not stack:
            return None
        return stack[0]
```
pass
### **解法步驟**

#### **1. 優先級計算（`get_val` 函數）**

- 操作符的優先級定義：
    - `+` 和 `-` 的優先級最低，為 `1`。
    - `*` 和 `/` 的優先級較高，為 `2`。
    - 使用 `base` 來處理括號優先級，每遇到一個左括號 `(`，`base` 增加 `10`，每遇到一個右括號 `)`，`base` 減少 `10`。
    - 優先級公式：`優先級 = 基本優先級 + base`。
    - 其他字符（如數字）被視為最大優先級（不影響運算）。

---

#### **2. 構造表達樹（`create_tree` 函數）**

- 遍歷輸入表達式 `expression`，對於每個字符：
    - 如果是括號，更新 `base` 值後繼續。
    - 計算當前字符的優先級，創建對應的樹節點。
    - 使用單調棧（`stack`）來構造樹：
        - **棧頂元素優先級大於等於當前節點：**
            - 從棧中彈出節點，設置為當前節點的左子節點。
        - **棧非空：**
            - 設置棧頂節點的右子節點為當前節點。
        - 將當前節點壓入棧中。
- 最終棧底部的節點即為表達樹的根。

---

#### **3. 拷貝樹結構（`copy_tree` 函數）**

- 將構造好的樹轉換為 `ExpressionTreeNode` 格式（題目要求的返回類型）。

---

### **具體舉例**

#### 輸入：

`["2", "*", "6", "-", "(", "23", "+", "7", ")", "/", "(", "1", "+", "2", ")"]`

#### 步驟詳解：

1. 初始化 `stack = []`，`base = 0`。
2. 遍歷表達式：
    - 遇到數字或操作符，計算優先級，創建節點並更新棧。
    - 遇到括號，調整 `base` 值。

逐步棧操作如下：

|當前字符|優先級|操作|棧中節點（從棧底到棧頂）|
|---|---|---|---|
|`"2"`|`sys.maxsize`|壓入棧中|`[2]`|
|`"*"`|`2`|設置為 `2` 的右子節點|`[2 -> *]`|
|`"6"`|`sys.maxsize`|壓入棧中|`[2 -> * -> 6]`|
|`"-"`|`1`|彈出棧節點，設置 `6` 為左子|`[2 -> -]`|
|`"("`|-|增加 `base`，繼續|`[2 -> -]`|
|`"23"`|`sys.maxsize`|壓入棧中|`[2 -> - -> 23]`|
|`"+"`|`11`|設置 `23` 的右子節點|`[2 -> - -> 23 -> +]`|
|`"7"`|`sys.maxsize`|壓入棧中|`[2 -> - -> 23 -> + -> 7]`|
|`")"`|-|減少 `base`，繼續|`[2 -> - -> 23 -> + -> 7]`|
|`"/"`|`2`|彈出節點，設置 `+` 為左子|`[2 -> - -> /]`|
|`"("`|-|增加 `base`，繼續|`[2 -> - -> /]`|
|`"1"`|`sys.maxsize`|壓入棧中|`[2 -> - -> / -> 1]`|
|`"+"`|`11`|設置 `1` 的右子節點|`[2 -> - -> / -> 1 -> +]`|
|`"2"`|`sys.maxsize`|壓入棧中|`[2 -> - -> / -> 1 -> + -> 2]`|
|`")"`|-|減少 `base`，結束|`[2 -> - -> / -> 1 -> + -> 2]`|

---

#### 最終構造出的樹：

```javascript
          -
         / \
        *   /
       / \ / \
      2  6+   +
           / \ / \
          23 71   2

```

---

### **代碼複雜度分析**

1. **時間複雜度**：
    
    - 遍歷輸入表達式一次，為 O(n)O(n)O(n)。
    - 單調棧的壓入和彈出操作總共最多 O(n)O(n)O(n)。
    - 總時間複雜度為 O(n)O(n)O(n)。
2. **空間複雜度**：
    
    - 使用單調棧儲存節點，最多儲存 O(n)O(n)O(n) 節點。
    - 總空間複雜度為 O(n)O(n)O(n)。

---

### **其他解法簡述**

1. **後序遍歷轉換法**：
    
    - 將中序表達式轉換為後序表達式，再構造表達樹。
    - 時間複雜度：O(n)O(n)O(n)。
2. **遞歸劃分法**：
    
    - 遇到括號時劃分子表達式，對每個子表達式遞歸處理。
    - 時間複雜度：最壞情況 O(n2)O(n^2)O(n2)。
3. **帶有索引的堆棧法**：
    
    - 使用索引追蹤當前節點的父節點，減少棧操作。
    - 時間複雜度：O(n)O(n)O(n)。