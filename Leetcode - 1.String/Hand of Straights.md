
LeetCode 846 / LintCode 1166 Alice 手上有一副牌，裡面有若干張牌，每張牌上都寫著一個整數，紀錄在整數陣列 `hand` 中，其中 `hand[i]` 是第 `i` 張牌上的數字。

現在她想把這些牌重新排列成若干組，使得**每一組的大小都剛好是 `groupSize`**，並且**每一組都由 `groupSize` 張連續的牌組成**（例如 `[1, 2, 3]`）。

如果她可以完成這樣的分組，請回傳 `True`；否則，請回傳 `False`。

**样例 1：** 输入：
```Python
hand = [1, 2, 3, 6, 2, 3, 4, 7, 8]
groupSize = 3
```

输出：
```Python
True
```

解释： Alice 手上的牌可以被重新排列為 `[1, 2, 3]`、`[2, 3, 4]`、`[6, 7, 8]`，共 3 組，每組都有 3 張連續的數字。

**样例 2：** 输入：
```Python
hand = [1, 2, 3, 4, 5]
groupSize = 4
```

输出：
```Python
False
```

解释： 總共有 5 張牌，無法被 `groupSize = 4` 整除，不可能分成每組 4 張的順子。

**样例 3：** 输入：
```Python
hand = [1, 1, 2, 2, 3, 3]
groupSize = 2
```

输出：
```Python
True
```

解释： 可以分成 `[1, 2]`、`[1, 2]`、`[2, 3]` 嗎？注意每張牌只能用一次，正確分組為 `[1, 2]`、`[1, 2]`、`[3, 3]`（不合法），但 `[1, 2]`、`[2, 3]`、`[1, 2]` 也不行；實際上這裡 `[1, 2]`、`[1, 2]` 剩 `[3, 3]` 會失敗，若為 `groupSize = 3` 則可分成 `[1, 2, 3]`、`[1, 2, 3]`。 （修正範例輸入為 `hand = [1, 1, 2, 2, 3, 3], groupSize = 3`，輸出 `True`，可分成兩組 `[1, 2, 3]`。）


```Python
from collections import Counter
import heapq

class Solution:
    def isNStraightHand(self, hand: list[int], groupSize: int) -> bool:
        # 1. 若總牌數無法被 groupSize 整除，必定無法完全分組
        if len(hand) % groupSize != 0:
            return False

        # 2. 統計每張牌出現的次數，並將不重複的牌面放入最小堆（Min-Heap）
        count = Counter(hand)
        min_heap = list(count.keys())
        heapq.heapify(min_heap)

        # 3. 每次從目前最小的牌面出發，嘗試湊出長度為 groupSize 的連續順子
        while min_heap:
            first = min_heap[0]  # 當前剩餘牌中最小的數字，必定是某組順子的起點

            for num in range(first, first + groupSize):
                if count[num] == 0:
                    return False
                count[num] -= 1

                # 若某張牌的剩餘數量歸零，它必須剛好是當前堆頂元素，否則代表中間出現斷層
                if count[num] == 0:
                    if num != min_heap[0]:
                        return False
                    heapq.heappop(min_heap)

        return True
```


```Python
# 另一種寫法：直接對牌面排序後貪心消除（代碼最簡潔）
from collections import Counter

class Solution:
    def isNStraightHand(self, hand: list[int], groupSize: int) -> bool:
        if len(hand) % groupSize != 0:
            return False

        count = Counter(hand)

        # 依照牌面數字由小到大遍歷
        for num in sorted(count):
            need = count[num]
            if need > 0:
                # 當前還有 need 張 num，就必須同時開啟 need 組以 num 為起點的順子
                for i in range(num, num + groupSize):
                    if count[i] < need:
                        return False
                    count[i] -= need

        return True
```

解釋: step1: 先檢查 `len(hand) % groupSize == 0`，若不能整除直接回傳 `False` step2: 以 `hand = [1, 2, 3, 6, 2, 3, 4, 7, 8], groupSize = 3` 為例，統計每種牌面張數：`count = {1: 1, 2: 2, 3: 2, 4: 1, 6: 1, 7: 1, 8: 1}`，並將所有鍵值放入最小堆 `min_heap = [1, 2, 3, 4, 6, 7, 8]` step3: 第一輪：堆頂最小牌 `first = 1`，需要湊 `[1, 2, 3]`。檢查三者數量皆大于 `0`，各減 `1` 後變成 `{1: 0, 2: 1, 3: 1, 4: 1, 6: 1, 7: 1, 8: 1}`；其中 `1` 歸零且剛好是堆頂，將 `1` 彈出堆，`min_heap` 頂端變為 `2` step4: 第二輪：堆頂最小牌 `first = 2`，需要湊 `[2, 3, 4]`。三者數量皆為 `1`，各減 `1` 後全數歸零，依序從堆頂彈出 `2, 3, 4`，`min_heap` 頂端變為 `6` step5: 第三輪：堆頂最小牌 `first = 6`，需要湊 `[6, 7, 8]`。三者數量皆為 `1`，各減 `1` 後全數歸零並彈出堆，此時 `min_heap` 為空 step6: 所有牌都成功分組，所以答案 = `True`

解釋: 為什麼「目前剩下的最小牌」一定可以當作某組順子的起點？

因為不存在任何比它更小的牌可以排在它前面！既然它不能當順子的中間或結尾，它就**唯一只能作為某組順子的開頭**。依此貪心原則，只要鎖定當前最小牌 `first`，就強制向後消耗 `first, first+1, ..., first+groupSize-1` 各一張（或一次性消耗 `count[first]` 張）；如果在消耗過程中發現某個數字張數不夠，或者中間某張牌提早歸零（導致後續更小的牌失去連續橋樑），就代表無解。

### **LeetCode 846 / LintCode 1166 - Hand of Straights（一手順子）**

這道題目是 **貪心演算法（Greedy）與雜湊表 + 最小堆（Hash Map + Min-Heap）** 的經典代表題（NeetCode 150 必考題，與 LeetCode 1296 完全相同）。問題核心在於利用「全局最小元素無前驅」的性質，將看似需要回溯組合的撲克牌分組問題，轉化為確定性的單向貪心消除過程。

## **解法解析**

這道題可以轉化為 **頻率統計 + 最小起點貪心消除（Frequency Map + Greedy）** 來求解。

### **思路**

1. **觀察關鍵約束（最小值的唯一角色）**
    
    - 假設手牌中目前還剩下的最小數字是 `x`。
        
    - 因為沒有任何剩餘的牌比 `x` 小，所以 `x` 絕對不可能成為 `[x-1, x, x+1]` 這種順子的中間牌，它**唯一只能是某一組順子的第一張牌**。
        
    - 因此，一旦存在 `x`，就**必須**同時存在 `x+1, x+2, ..., x + groupSize - 1` 來與它配對成組，這一步沒有任何其他選擇空間（完全滿足貪心選擇性質）。
        
2. **發現斷層檢測機制（Min-Heap 關鍵細節）**
    
    - 在用最小堆維護剩餘可用數字時，每次從堆頂取出 `first = min_heap[0]`，並依序將 `num`（從 `first` 到 `first + groupSize - 1`）的計數減 `1`。
        
    - 當某個 `num` 的計數變成 `0` 時，它必須剛好等於當前的 `min_heap[0]`。
        
    - **為什麼？** 假設 `first = 1, groupSize = 3`，牌組為 `[1, 2, 2, 3]`。消耗一組 `[1, 2, 3]` 後，`1` 變成 `0`（在堆頂，合法彈出），`2` 剩 `1` 張，但 `3` 提前變成 `0`（此時堆頂是 `2` 而不是 `3`）！這代表還有沒用完的 `2`，但通往 `3` 的路已經斷了，後續 `2` 永遠無法湊出順子，因此可直接判定 `False`。
        
3. **代碼執行流程**
    
    - 檢查 `len(hand) % groupSize != 0`，若無法整除直接回傳 `False`。
        
    - 使用 `Counter(hand)` 統計每種牌面的數量，將所有不重複的鍵值建立為最小堆 `min_heap`。
        
    - 當 `min_heap` 不為空時，取得堆頂 `first = min_heap[0]`，遍歷 `num` 從 `first` 到 `first + groupSize - 1`：
        
        - 若 `count[num] == 0`，回傳 `False`。
            
        - 將 `count[num]` 減 `1`；若減完後 `count[num] == 0`，檢查 `num == min_heap[0]`，不相等則回傳 `False`，相等則 `heappop(min_heap)`。
            
    - 迴圈順利結束則回傳 `True`。
        

## **具體舉例**

假設 `hand = [1, 2, 3, 6, 2, 3, 4, 7, 8]`，`groupSize = 3`，初始 `count = {1:1, 2:2, 3:2, 4:1, 6:1, 7:1, 8:1}`：

|**輪次**|**當前堆頂 first**|**目標順子區間**|**扣除後 count 變化**|**堆頂彈出動作**|**剩餘 min_heap**|
|---|---|---|---|---|---|
|初始|-|-|`{1:1, 2:2, 3:2, 4:1, 6:1, 7:1, 8:1}`|-|`[1, 2, 3, 4, 6, 7, 8]`|
|第 1 輪|`1`|`[1, 2, 3]`|`1->0`, `2->1`, `3->1`|`1` 歸零且在堆頂，彈出 `1`|`[2, 3, 4, 6, 7, 8]`|
|第 2 輪|`2`|`[2, 3, 4]`|`2->0`, `3->0`, `4->0`|`2, 3, 4` 依序歸零並從堆頂彈出|`[6, 7, 8]`|
|第 3 輪|`6`|`[6, 7, 8]`|`6->0`, `7->0`, `8->0`|`6, 7, 8` 依序歸零並從堆頂彈出|`[]`（空）|

所有牌均成功消除，最終返回 `True`。

## **時間與空間複雜度分析**

假設陣列 `hand` 的總長度為 $N$，不重複的牌面種類數為 $K$（$K \le N$）：

- **時間複雜度：$O(N + K \log K)$（最壞情況 $O(N \log N)$）**
    
    - 統計頻率 `Counter(hand)` 需要 $O(N)$ 時間。
        
    - 建立最小堆 `heapify` 需要 $O(K)$ 時間。
        
    - 每個不重複數字最多只會被 `heappop` 一次（每次 $O(\log K)$，共 $O(K \log K)$），而內層迴圈總共只會扣減 $N$ 張牌的計數（每次 $O(1)$，共 $O(N)$）。
        
- **空間複雜度：$O(N)$**
    
    - 雜湊表 `count` 與最小堆 `min_heap` 最多儲存 $K$ 個元素，在所有牌皆不重複的最壞情況下（$K = N$），空間複雜度為 **$O(N)$**。
        

## **其他解法**

### **1. 排序鍵值 + 批量貪心扣除（Sorted Keys + Batch Subtraction）**

- **思路**：
    
    - 不使用最小堆，而是直接對 `Counter(hand)` 的鍵值進行升序排序 `sorted(count)`。
        
    - 從小到大遍歷每個數字 `num`，若 `need = count[num] > 0`，代表還有 `need` 張 `num` 尚未分組，直接一次性將 `num` 到 `num + groupSize - 1` 的計數全部減去 `need`（若其中任何一張牌數量小於 `need`，直接回傳 `False`）。
        
- **時間複雜度**：$O(N + K \log K)$（當有大量重複牌面時，批量扣除比逐張扣除更快）
    
- **空間複雜度**：$O(N)$
    

### **2. $O(N)$ 線性時間解法（無需排序與最小堆）**

- **思路**：
    
    - 利用 **LeetCode 128（最長連續序列）** 的技巧：不對牌面排序，而是針對每個牌面 `num`，先不斷往左找 `start = num` 直到 `count[start - 1] == 0`（找到該連續段的最左起點）。
        
    - 從起點 `start` 開始往右逐一消除順子，直到原本的 `num` 被消耗完為止。因為每個數字最多被訪問常數次，可將時間複雜度從 $O(N \log N)$ 降至嚴格線性。
        
- **時間複雜度**：$O(N)$
    
- **空間複雜度**：$O(N)$
    

## **總結**

- **面試標準最佳解（Counter + Min-Heap / Sorted Keys）**：時間 **$O(N \log N)$**、空間 **$O(N)$**，邏輯直觀、容易在面試時無 Bug 寫出，且「批量扣除法（`count[i] -= need`）」代碼極為簡潔。
    
- **極致優化進階解（逆向找連續段起點）**：時間 **$O(N)$**、空間 **$O(N)$**，適合在面試官追問「如果牌面數字範圍極大，能否省去排序的 $\log N$ 開銷？」時提出。
    

## **LeetCode / LintCode 順子與分組貪心類問題整理**

|**題號**|**題目名稱**|**問題類型**|**解法**|
|---|---|---|---|
|**LeetCode 846 / LintCode 1166**|Hand of Straights（一手順子）|**將陣列分成若干個大小固定為 `k` 的連續整數子序列**|**雜湊表計數 + 最小堆（或排序鍵值批量扣除）**，$O(N \log N)$|
|**LeetCode 1296**|Divide Array in Sets of K Consecutive Numbers|**與 LeetCode 846 完全相同的雙胞胎題**|**雜湊表計數 + 最小堆 / 排序貪心消除**|
|**LeetCode 659 / LintCode 1103**|Split Array into Consecutive Subsequences|**將已排序陣列分割為若干個「長度至少為 3」的連續子序列**|**雙雜湊表貪心法（優先接在現有順子後方，其次才開新順子）**，$O(N)$|
|**LeetCode 128 / LintCode 124**|Longest Consecutive Sequence（最長連續序列）|**在未排序陣列中找出最長連續整數序列的長度**|**雜湊集合（Hash Set）+ 只從連續段起點（`num - 1 not in set`）開始往右枚舉**，$O(N)$|
|**LeetCode 2405**|Optimal Partition of String（子字串的最優劃分）|**將字串切成最少片段，使每個片段內無重複字元**|**雜湊集合 / 位元遮罩貪心切割**，$O(N)$|
|**LeetCode 621 / LintCode 945**|Task Scheduler（任務調度器）|**相同任務需間隔 `n` 個冷卻時間，求完成所有任務的最短時間**|**最大堆貪心排程** 或 **數學桶構造公式**|

### **進階變形題：如果順子的長度不再固定為 `groupSize`，而是要求「每組長度至少為 3」且陣列已升序排列，該如何處理？**

這對應經典高頻題 **LeetCode 659 - Split Array into Consecutive Subsequences**。因為長度不固定，我們多了一個選擇：**「把當前數字接在已存在的順子後面」** vs. **「用當前數字開一組長度為 3 的新順子」**。貪心策略是**能接在舊順子後面就優先接（避免浪費後面的牌）**，接不上才嘗試開啟新順子：

Python

```
from collections import Counter, defaultdict

class Solution:
    def isPossible(self, nums: list[int]) -> bool:
        left = Counter(nums)       # 紀錄每張牌還剩幾張可用
        end = defaultdict(int)     # 紀錄以數字 x 結尾的合法順子目前有幾條

        for x in nums:
            if left[x] == 0:
                continue

            # 策略 1：優先接在以 x - 1 結尾的現有順子後面
            if end[x - 1] > 0:
                left[x] -= 1
                end[x - 1] -= 1
                end[x] += 1
            # 策略 2：無法接續，則嘗試以 x 為起點開啟一條長度為 3 的新順子 [x, x+1, x+2]
            elif left[x + 1] > 0 and left[x + 2] > 0:
                left[x] -= 1
                left[x + 1] -= 1
                left[x + 2] -= 1
                end[x + 2] += 1
            # 兩種策略都行不通，代表 x 落單無法成組
            else:
                return False

        return True
```