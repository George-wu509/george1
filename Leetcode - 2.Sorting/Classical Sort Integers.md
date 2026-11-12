Lintcode 463
给一组整数，按照升序排序，使用选择排序，冒泡排序，插入排序或者任何 O(n2) 的排序算法。

|                           |     |
| ------------------------- | --- |
| [[排序問題跟quicksort code]]   |     |
| [[#### 幾個關於quicksort的問題]] |     |
#### 排序問題跟quicksort code
```python
"""
样例  1:
	输入:  [3, 2, 1, 4, 5]
	输出:  [1, 2, 3, 4, 5]
	
	样例解释: 
	返回排序后的数组。

样例 2:
	输入:  [1, 1, 2, 1, 1]
	输出:  [1, 1, 1, 1, 2]
	
	样例解释: 
	返回排好序的数组。
```


```python
class Solution:
    def sort_integers(self, a):
        if a == None or len(a) == 0:
            return
        self.quickSort(a, 0, len(a) - 1)
        
    def quickSort(self, a, start, end):
        if start >= end:
            return
        
        left, right = start, end
        pivot = a[(start+end)//2]
        
        while left <= right:
            while left <= right and a[left] < pivot:
                left += 1
            while left <= right and a[right] > pivot:
                right -= 1
                
            if left <= right:
                a[left], a[right] = a[right], a[left]
                left += 1
                right -= 1
                
        self.quickSort(a, start, right)
        self.quickSort(a, left, end)
```
pass

Quick Sort:
	Step1:  **left, right** = start, end  以及  **pivot** = A[start + (end-start)/2]
	
	Step2:  left index往右走, 遇到A[left]比pivot大則停下來
	
	Right index 往左走, 遇到A[right]比pivot小則停下來
	
	Step3:  把停下來的A[left]  跟 A[right] 交換, 然後 left index +1, right index –1
	
	Step4:  **quick_sort**(A, start, right)  recursive: 從start到新的right
	
	**quick_sort**(A, left, end)     recursive: 從新的left到end



#### 幾個關於quicksort的問題
```
這是quicksort排序a(example: a=[ 3, 2, 1, 4, 5 ])的解法, 請幫我分析回答以下問題:

1. pivot = a[(start+end)//2] 跟 pivot = a[left + (right-left)//2]是否都正確, 哪種寫法比較好

2. 在while left <= right裡面的迴圈, 為何兩個while迴圈還要再加上while left <= right? 譬如while left <= right and a[left] < pivot>

3. 兩個while迴圈要使用a[left] < pivot不是a[left] <= pivot

3.  為什麼藥用while left <= right而不是while left < right

4. 怎麼從這個流程計算complexity, 是從哪一段code? 請詳細說明

```
這段程式碼是經典的 **Hoare Partition（雙向指標分區）** 寫法。以下針對你的五個問題逐一解析：

### 1. `pivot = a[(start+end)//2]` 與 `pivot = a[left + (right-left)//2]` 是否都正確？哪種較好？

**兩者在此處完全等價且都正確**，但 **`start + (end - start) // 2`（或 `left + (right - left) // 2`）是更好的工程習慣**。

- **為何等價**：在宣告 `pivot` 的那一行，`left` 剛被賦值為 `start`，`right` 剛被賦值為 `end`，兩者在數學上完全相同（$start + \frac{end - start}{2} = \frac{start + end}{2}$）。
    
- **為何後者較好**：在 C++ 或 Java 等固定整數長度的語言中，若陣列極大，`start + end` 可能會超過整數上限導致 **溢位（Integer Overflow）**，而 `start + (end - start) // 2` 先減後加可完全避免溢位。雖然 Python 的整數支援無限精度不會溢位，但養成防溢位寫法在跨語言面試時更受青睞。
    

### 2. 內層的兩個 `while` 迴圈為何還要加上 `left <= right`？

主要有兩個原因：**防止陣列越界**，以及**確保指標在交錯瞬間立刻停下**。

1. **防止指標越界（Index Out of Bounds）**：如果在某次交換後，剩餘元素全部小於或大於 `pivot`，沒有 `left <= right` 限制的話，`left` 可能會一路加到超過 `end`（或 `right` 減到小於 `start`），引發索引錯誤。
    
2. **維持遞迴邊界的正確性**：外層迴圈的目標是讓 `left` 與 `right` **剛好交錯**（結束時 `right < left`）。若內層不加 `left <= right`，即使兩指標已經交錯，`left` 仍可能繼續往右衝過頭，導致最後 `self.quickSort(a, start, right)` 和 `self.quickSort(a, left, end)` 切分出的區間遺漏元素。
    

### 3. 內層為何要用 `a[left] < pivot` 而不是 `a[left] <= pivot`？

核心目的是**將「等於 `pivot` 的重複元素」平均分配到左右兩側，避免時間複雜度退化成 $O(N^2)$**。

- **假設陣列全為重複數字**：例如 `a = [2, 2, 2, 2, 2]`，`pivot = 2`。
    
    - **若用 `<=` 與 `>=`**：`left` 會一路向右衝到底（`left = 5`），`right` 停在 `4`。這會導致每次遞迴只排好 1 個元素，剩下 $N-1$ 個元素全部丟進左半邊遞迴，分區極度不平衡，複雜度退化為 **$O(N^2)$**。
        
    - **若用嚴格 `<` 與 `>`**：遇到等於 `pivot` 的值時，左右指標都會停下來並進行交換。雖然看似多做了交換，但這能讓 `left` 與 `right` 穩穩地在陣列正中央交錯，確保每次都對半切分，維持最佳的 **$O(N \log N)$** 複雜度。
        

### 4. 外層與判斷為何要用 `left <= right` 而不是 `left < right`？

這是為了**強制讓迴圈結束時 `left` 與 `right` 徹底交錯（即 `right < left`）**，避免無限遞迴（RecursionError）。

- **若改用 `while left < right`**：迴圈可能在 `left == right` 時就提早結束。此時會發生兩個問題：
    
    1. **元素重複進入兩邊遞迴**：`a[left]`（同時也是 `a[right]`）會同時被包含在 `quickSort(a, start, right)` 和 `quickSort(a, left, end)` 中。
        
    2. **無限遞迴（死循環）**：以 `a = [1, 2]` 為例，`start = 0, end = 1`，`pivot = 1`。若用 `left < right`，迴圈結束時 `left = 0, right = 0`。接著呼叫 `self.quickSort(a, 0, 0)` 和 `self.quickSort(a, 0, 1)`——後者與原本的呼叫一模一樣，程式將陷入無窮迴圈直到堆疊溢位。
        
- **使用 `left <= right`**：當 `left == right` 時還會再進入迴圈一次，執行 `left += 1` 與 `right -= 1`，保證離開迴圈時必定滿足 `right < left`，左右子區間絕對不會重疊，且長度必定嚴格小於原本的區間。
    

### 5. 如何從程式碼計算時間與空間複雜度？

複雜度由「單層分區的掃描成本」**與**「遞迴樹的深度」共同決定，對應到程式碼中的兩個區塊：

#### 時間複雜度（Time Complexity）

1. **單層工作量：$O(N)$**
    
    - **對應程式碼**：`while left <= right:` 整個分區迴圈。
        
    - **計算方式**：`left` 從 `start` 往右走、`right` 從 `end` 往左走，兩者相向而行直到交錯。無論內層 `while` 怎麼跳，整個區間長度為 $N$ 的陣列，每個元素最多只會被 `left` 或 `right` 存取一次，因此單層呼叫的時間為線性 $O(N)$。
        
2. **遞迴深度（樹高）：$\log N$ 層（平均）至 $N$ 層（最差）**
    
    - **對應程式碼**：最後兩行的 `self.quickSort(a, start, right)` 與 `self.quickSort(a, left, end)`。
        
    - **平均 / 最佳情況 $O(N \log N)$**：每次選中的 `pivot` 都能將陣列大致對半切分（長度變為 $N/2$），需要切 $\log_2 N$ 次才會觸發終止條件 `if start >= end: return`。每層總工作量皆為 $O(N)$，共 $\log N$ 層，相乘得 **$O(N \log N)$**。
        
    - **最差情況 $O(N^2)$**：若每次選到的 `pivot` 剛好都是當前區間的最小值或最大值（例如陣列呈特定山峰狀排列，每次切出 $1$ 個與 $N-1$ 個元素），遞迴樹深度會變成 $N$ 層。總運算次數為 $N + (N-1) + \dots + 1$，退化為 **$O(N^2)$**。
        

#### 空間複雜度（Space Complexity）

- **對應程式碼**：函式開頭的終止條件與末尾的兩行遞迴呼叫（因為陣列交換是原地 In-place 進行，沒有額外開新陣列）。
    
- **計算方式**：空間消耗完全來自系統的 **Call Stack（遞迴呼叫堆疊）**，取決於遞迴樹的最大深度：
    
    - **平均 / 最佳空間複雜度**：**$O(\log N)$**
        
    - **最差空間複雜度**：**$O(N)$**
