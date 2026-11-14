
LeetCode 347 / LintCode 1281 給你一個整數陣列 `nums` 和一個整數 `k`，請你返回其中出現頻率前 `k` 高的元素。你可以按 **任意順序** 返回答案。

**注意事項：**

- 陣列長度 `n` 滿足 `1 <= n <= 10^5`
- 陣列元素範圍為 `-10^4 <= nums[i] <= 10^4`
- `k` 的取值範圍是 `[1, 陣列中不相同元素的個數]`
- 題目資料保證答案唯一，換句話說，陣列中前 `k` 個高頻元素的集合是唯一的。
- **進階要求**：你所設計演算法的時間複雜度 **必須** 優於 $O(n \log n)$，其中 `n` 是陣列大小。

**样例 1：** 输入：
```Python
nums = [1, 1, 1, 2, 2, 3]
k = 2
```

输出：
```Python
[1, 2]
```

解释： 元素 `1` 出現了 3 次，元素 `2` 出現了 2 次，元素 `3` 出現了 1 次。出現頻率前 2 高的元素是 `1` 和 `2`。

**样例 2：** 输入：
```Python
nums = [1]
k = 1
```

输出：
```Python
[1]
```

解释： 陣列只有一個元素 `1`，出現頻率最高的就是 `[1]`。

**样例 3：** 输入：
```Python
nums = [4, 1, -1, 2, -1, 2, 3]
k = 2
```

输出：
```Python
[-1, 2]
```

解释： 元素 `-1` 出現 2 次，元素 `2` 出現 2 次，其餘元素各出現 1 次，因此前 2 高頻元素為 `[-1, 2]`。


```Python
# 解法一：雜湊表計數 + 桶排序（Bucket Sort，時間複雜度最佳 O(n)）
from collections import Counter

def topKFrequent(self, nums: list[int], k: int) -> list[int]:
    count = Counter(nums)  # 統計每個數字出現的頻率：{數字: 頻率}
    # 建立 n + 1 個桶子，索引 i 代表「出現頻率為 i」的數字列表
    buckets = [[] for _ in range(len(nums) + 1)]

    for num, freq in count.items():
        buckets[freq].append(num)

    res = []
    # 從最高頻率（陣列尾端）往低頻率倒序遍歷
    for freq in range(len(buckets) - 1, 0, -1):
        for num in buckets[freq]:
            res.append(num)
            if len(res) == k:
                return res
    return res
```


```Python
# 解法二：雜湊表計數 + 大小為 k 的最小堆（Min-Heap，適合海量串流資料）
import heapq
from collections import Counter

def topKFrequent(self, nums: list[int], k: int) -> list[int]:
    count = Counter(nums)
    min_heap = []  # 維護大小不超過 k 的最小堆，元素格式為 (頻率, 數字)

    for num, freq in count.items():
        heapq.heappush(min_heap, (freq, num))
        # 一旦堆內元素超過 k 個，就彈出「頻率最小」的門檻元素
        if len(min_heap) > k:
            heapq.heappop(min_heap)

    return [num for freq, num in min_heap]
```

解釋: step1: 任何一個數字在長度為 `n` 的陣列中，出現的次數最多就是 `n` 次（頻率範圍被嚴格限制在 `1 ~ n` 之間），這讓我們可以使用 **桶排序（Bucket Sort）** 免去比較排序。 step2: 以 `nums = [1, 1, 1, 2, 2, 3], k = 2` 為例，先用 `Counter(nums)` 統計頻率：`{1: 3, 2: 2, 3: 1}`。 step3: 建立長度為 `6 + 1 = 7` 的二維陣列 `buckets`，將每個數字放進「索引等於其出現次數」的桶子裡：

- `buckets[1] = [3]`（出現 1 次的數字）
    
- `buckets[2] = [2]`（出現 2 次的數字）
    
- `buckets[3] = [1]`（出現 3 次的數字）
    
- 其餘桶子為空 `[]` step4: 從最大索引 `freq = 6` 開始往 `1` 倒序掃描 `buckets`：
    
- 掃到 `freq = 3` 時，取出 `1` -> `res = [1]`
    
- 掃到 `freq = 2` 時，取出 `2` -> `res = [1, 2]`，此時 `len(res) == k`（達到 2 個），直接提前結束並回傳 `[1, 2]`。
    

解釋: 核心規則：先用 `HashMap` 統計頻率，再決定用「空間換時間」還是「堆維護門檻」。

**桶排序法** 利用「陣列索引天然有序」的特性，把頻率當作索引，倒序收集前 `k` 個元素即可達到 $O(n)$；**最小堆法** 則維護一個容量為 `k` 的淘汰賽機制，隨時踢掉頻率最低的候選人，適合資料量大到無法一次全塞進桶子的場景。

### **LeetCode 347 / LintCode 1281 - Top K Frequent Elements（前 K 個高頻元素）**

這道題目是 **雜湊表（HashMap）** 結合 **桶排序（Bucket Sort）** 與 **優先佇列（Heap）** 的經典高頻面試題（Blind 75 & NeetCode 150 必考題）。它直接考察面試者如何在「全量排序 $O(n \log n)$」的基礎上，利用題目特性將複雜度優化到 $O(n \log k)$ 甚至嚴格線性的 $O(n)$。

## **解法解析**

這道題可以轉化為 **頻率統計 + Top-K 篩選** 問題來求解。

### **思路**

1. **為何桶排序能突破 $O(n \log n)$ 的下限？**
    
    - 傳統基於比較的排序演算法（如 QuickSort、MergeSort）時間複雜度下限為 $O(n \log n)$。
        
    - 但本題我們排序的對象是「**出現次數（Frequency）**」，而一個長度為 `n` 的陣列中，所有數字的出現次數必定落在 `[1, n]` 的整數區間內。
        
    - 既然數值範圍有明確上下界，我們就可以直接開一個長度為 `n + 1` 的陣列 `buckets`，用 **陣列下標代表頻率**，實現 $O(n)$ 的線性分桶。
        
2. **為何「前 K 大」要用「最小堆（Min-Heap）」而不是最大堆？**
    
    - 這是很多初學者容易搞反的直覺：求前 `k` 高頻，若用最大堆，必須把所有 `m` 個不重複元素都塞進堆裡（建堆或入堆耗時 $O(m \log m)$），再彈出 `k` 次。
        
    - 反之，若維護一個大小僅為 `k` 的 **最小堆（Min-Heap）**，堆頂永遠存放著「目前前 `k` 名中的最後一名（守門員）」。每當新元素進來時，只要比守門員強就替換掉它，每次操作僅需 $O(\log k)$，總時間降至 $O(n \log k)$。
        
3. **代碼執行流程（以桶排序法為例）**
    
    - 使用 `Counter(nums)` 遍歷陣列一次，得到每個數字的出現頻率表 `count`。
        
    - 初始化 `buckets = [[] for _ in range(len(nums) + 1)]`。
        
    - 遍歷 `count.items()`，將數字 `num` 附加到 `buckets[freq]` 中。
        
    - 從 `len(buckets) - 1` 倒序遍歷到 `1`，依序將非空桶子裡的元素加入 `res`，一旦 `len(res) == k` 立即回傳。
        

## **具體舉例**

假設 `nums = [1, 1, 1, 2, 2, 3], k = 2`，頻率統計為 `{1: 3, 2: 2, 3: 1}`，桶排序建構與倒序收集過程如下：

|**桶子索引 freq**|**buckets[freq] 內容**|**倒序掃描動作**|**res 狀態**|
|---|---|---|---|
|`6`|`[]`|空桶，跳過|`[]`|
|`5`|`[]`|空桶，跳過|`[]`|
|`4`|`[]`|空桶，跳過|`[]`|
|`3`|`[1]`|取出 `1` 加入結果|`[1]`|
|`2`|`[2]`|取出 `2` 加入結果，達到 `k=2` 提前結束|`[1, 2]`|
|`1`|`[3]`|（已提前返回，無需掃描）|`[1, 2]`|

最後返回 **`[1, 2]`**。

## **時間與空間複雜度分析**

設 `n` 為輸入陣列 `nums` 的長度，`m` 為不重複元素的個數（`m` 小於等於 `n`）：

- **解法一（雜湊表 + 桶排序）**：
    
    - **時間複雜度：O(n)**
        
        - `Counter(nums)` 統計頻率需要 $O(n)$ 時間；將 `m` 個不重複元素放入桶中需要 $O(m)$ 時間；倒序掃描桶子時，外層迴圈最多跑 `n` 次，內層迴圈總共只會取出 `k` 個元素，因此總時間複雜度為 **O(n)**。
            
    - **空間複雜度：O(n)**
        
        - `count` 雜湊表最多佔用 $O(n)$ 空間，二維陣列 `buckets` 包含 `n + 1` 個子列表且總共存放 `m` 個元素，佔用 $O(n)$ 空間，總空間複雜度為 **O(n)**。
            
- **解法二（雜湊表 + 大小為 `k` 的最小堆）**：
    
    - **時間複雜度：O(n log k)**
        
        - 統計頻率需要 $O(n)$ 時間；遍歷 `m` 個不重複元素並維護大小為 `k` 的最小堆，每次 `heappush` 與 `heappop` 需要 $O(\log k)$ 時間，總時間複雜度為 $O(n + m \log k)$，最壞情況（所有元素皆不相同，`m = n`）下為 **O(n log k)**（當 `k == n` 時退化為 $O(n \log n)$）。
            
    - **空間複雜度：O(n)**
        
        - 雜湊表 `count` 佔用 $O(m)$ 空間，最小堆 `min_heap` 佔用 $O(k)$ 空間，最壞情況下總空間複雜度為 **O(n)**。
            

## **其他解法**

### **1. 快速選擇演算法（Quickselect / Hoare's Selection Algorithm）**

- **思路**：
    
    - 先用 `Counter(nums)` 取得 `(num, freq)` 列表 `unique_items`（長度為 `m`）。
        
    - 借鑑快速排序（QuickSort）的 Partition 思想，隨機挑選一個樞紐（Pivot）頻率，將頻率大於 Pivot 的元素移到左邊，小於的移到右邊。
        
    - 根據 Pivot 最終落點與 `k` 的比較，只需往其中一邊繼續遞迴切分，直到左半邊剛好包含前 `k` 個高頻元素。
        
- **時間複雜度**：平均 **O(n)**，最壞情況下（每次都選到極值 Pivot）退化為 **O(n²)**。
    
- **空間複雜度**：**O(n)**（儲存頻率列表及遞迴呼叫棧）。
    

### **2. Python 內建工具一行解（`Counter.most_common(k)`）**

- **思路**：
    
    - Python 標準庫 `collections.Counter` 內建了 `.most_common(k)` 方法，底層正是呼叫 `heapq.nlargest(k, self.items(), key=itemgetter(1))` 實作的。
        
    - 寫法極簡：`return [num for num, _ in Counter(nums).most_common(k)]`。
        
- **注意**：在面試中通常建議先展示手寫的 **桶排序** 或 **最小堆**，再補充說明在實際工程開發中會直接呼叫 `Counter.most_common(k)`。
    
- **時間複雜度**：**O(n log k)**
    
- **空間複雜度**：**O(n)**
    

## **總結**

- **理論複雜度最優解**：**桶排序（Bucket Sort）**，時間複雜度嚴格為 $O(n)$，完全滿足題目「優於 $O(n \log n)$」的進階要求，也是面試中最受青睞的標準答案。
    
- **海量資料 / 串流場景首選**：**最小堆（Min-Heap）**，當資料量極大（例如從硬碟或網路串流讀取數十億筆紀錄，無法開出長度 `n` 的桶子陣列，但 `k` 很小如 `k = 100`）時，只需維護大小為 `k` 的堆即可。
    
- **面試加分亮點**：能清楚對比 **桶排序（$O(n)$ 時間、需 $O(n)$ 桶空間）** 與 **最小堆（$O(n \log k)$ 時間、僅需 $O(k)$ 堆空間）** 各自的適用邊界。
    

## **LeetCode / LintCode Top-K 與頻率統計類問題整理**

|**題號**|**題目名稱**|**問題類型**|**解法**|
|---|---|---|---|
|**LeetCode 347**|Top K Frequent Elements（前 K 個高頻元素）|**頻率統計 + Top-K 篩選**|**Counter + 桶排序**（$O(n)$）或 **Counter + 大小為 `k` 的最小堆**（$O(n \log k)$）|
|**LeetCode 215**|Kth Largest Element in an Array（陣列中的第 K 個最大元素）|**無序陣列 Top-K**|**Quickselect 快速選擇**（平均 $O(n)$）或 **大小為 `k` 的最小堆**（$O(n \log k)$）|
|**LeetCode 692**|Top K Frequent Words（前 K 個高頻單字）|**頻率統計 + 多條件排序（頻率降序 + 字典序升序）**|**Counter + 自訂比較規則最小堆**（`(-freq, word)` 或自訂 `__lt__`）|
|**LeetCode 451**|Sort Characters By Frequency（根據字元出現頻率排序）|**頻率統計 + 全量重排**|**Counter + 桶排序**（按頻率桶倒序拼接字元），時間複雜度 $O(n)$|
|**LeetCode 973**|K Closest Points to Origin（最接近原點的 K 個點）|**幾何距離 Top-K**|**大小為 `k` 的最大堆**（存負距離）或 **Quickselect**|
|**LeetCode 703**|Kth Largest Element in a Stream（資料流中的第 K 大元素）|**動態資料流 Top-K**|**大小為 `k` 的最小堆**：每次新增元素後若超過 `k` 就彈出堆頂|
|**LeetCode 295**|Find Median from Data Stream（資料流的中位數）|**動態資料流雙端維護**|**對頂堆（最大堆存較小一半 + 最小堆存較大一半）**，$O(\log n)$ 插入、$O(1)$ 查詢|
|**LeetCode 355**|Design Twitter（設計推特）|**多路歸併 Top-K**|**HashMap + 多路合併最大堆（Merge K Sorted Lists）** 取前 10 則貼文|

### **進階變體練習：LeetCode 692 - Top K Frequent Words（頻率相同時按字典序升序排列）**

如果題目改為給你一個單字陣列 `words`（例如 `["i", "love", "leetcode", "i", "love", "coding"]`），要求返回前 `k` 個出現次數最多的單字，且 **不同單字有相同出現頻率時，按字典順序（Alphabetical Order）較小的排在前面**：

Python

```
import heapq
from collections import Counter

class WordItem:
    def __init__(self, word: str, freq: int):
        self.word = word
        self.freq = freq

    # 自訂小於比較規則，讓最小堆把「頻率較小」或「頻率相同但字典序較大」的元素放在堆頂（優先淘汰）
    def __lt__(self, other: "WordItem") -> bool:
        if self.freq != other.freq:
            return self.freq < other.freq
        return self.word > other.word

class Solution:
    def topKFrequent(self, words: list[str], k: int) -> list[str]:
        count = Counter(words)
        min_heap = []

        for word, freq in count.items():
            heapq.heappush(min_heap, WordItem(word, freq))
            if len(min_heap) > k:
                heapq.heappop(min_heap)

        # 從最小堆依序彈出是由小到大，因此最後需反轉陣列
        res = []
        while min_heap:
            res.append(heapq.heappop(min_heap).word)
        return res[::-1]
```