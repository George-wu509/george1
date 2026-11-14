
設計 Twitter Design Twitter

LeetCode 355 (LintCode 501) 設計一個簡化版的 Twitter（推特），可以讓用戶發送推文（Tweet）、關注（Follow）或取消關注（Unfollow）其他用戶，並且能夠看見關注人（包括自己）最近的 **10 條**推文。

實作 `Twitter` 類別：

- `Twitter()` 初始化簡易版 Twitter 物件。
- `void postTweet(int userId, int tweetId)` 根據給定的 `tweetId` 和 `userId` 建立一則新推文。每次呼叫此函式都會使用一個不同的 `tweetId`。
- `List<Integer> getNewsFeed(int userId)` 檢索當前用戶新聞推播（News Feed）中最近 **10 則**推文的 ID。新聞推播中的每一則項目都必須是由**當前用戶自己**或是**他所關注的用戶**所發佈的。推文必須**由最近到最舊**排序。
- `void follow(int followerId, int followeeId)` ID 為 `followerId` 的用戶開始關注 ID 為 `followeeId` 的用戶。
- `void unfollow(int followerId, int followeeId)` ID 為 `followerId` 的用戶不再關注 ID 為 `followeeId` 的用戶。

**样例 1：** 输入：

```Python
["Twitter", "postTweet", "getNewsFeed", "follow", "postTweet", "getNewsFeed", "unfollow", "getNewsFeed"]
[[], [1, 5], [1], [1, 2], [2, 6], [1], [1, 2], [1]]
```

输出：
```Python
[null, null, [5], null, null, [6, 5], null, [5]]
```

解释：
```Python
twitter = Twitter()
twitter.postTweet(1, 5)  # 用戶 1 發佈了一則新推文 (id = 5)
twitter.getNewsFeed(1)   # 用戶 1 的新聞推播應該只包含他自己發佈的推文 -> [5]
twitter.follow(1, 2)     # 用戶 1 關注了用戶 2
twitter.postTweet(2, 6)  # 用戶 2 發佈了一則新推文 (id = 6)
twitter.getNewsFeed(1)   # 用戶 1 的新聞推播應該包含自己和用戶 2 的推文，按時間由新到舊排序 -> [6, 5]
twitter.unfollow(1, 2)   # 用戶 1 取消關注了用戶 2
twitter.getNewsFeed(1)   # 用戶 1 的新聞推播只剩自己的推文 -> [5]
```

**样例 2：** 输入：
```Python
["Twitter", "postTweet", "postTweet", "getNewsFeed"]
[[], [1, 101], [1, 102], [1]]
```

输出：
```Python
[null, null, null, [102, 101]]
```

解释： 用戶 1 連續發佈推文 101 與 102，因為 102 較晚發佈（時間戳較大），所以 `getNewsFeed(1)` 依序回傳 `[102, 101]`。

**样例 3：** 输入：
```Python
["Twitter", "follow", "unfollow", "getNewsFeed"]
[[], [1, 1], [1, 1], [1]]
```

输出：
```Python
[null, null, null, []]
```

解释： 邊界條件測試：用戶自己關注自己、又自己取消關注自己，系統不應報錯；且當沒有任何推文時，回傳空陣列 `[]`。

```Python
# 方法一：HashMap + 最大堆（K-路歸併指標法）- 面試最佳解
import heapq
from collections import defaultdict
from typing import List

class Twitter:
    def __init__(self):
        self.time = 0  # 全域遞增時間戳（注意：tweetId 不代表時間順序！）
        self.tweetMap = defaultdict(list)  # userId -> list of [time, tweetId]
        self.followMap = defaultdict(set)  # followerId -> set of followeeId

    def postTweet(self, userId: int, tweetId: int) -> None:
        self.time += 1
        self.tweetMap[userId].append([self.time, tweetId])

    def getNewsFeed(self, userId: int) -> List[int]:
        res = []
        maxHeap = []

        # 自己也必須看到自己的推文，使用聯集避免破壞原本的關注名單
        followees = self.followMap[userId] | {userId}

        # 1. 將每位關注者「最新的一篇推文」放入最大堆
        for followeeId in followees:
            if followeeId in self.tweetMap and len(self.tweetMap[followeeId]) > 0:
                index = len(self.tweetMap[followeeId]) - 1
                time, tweetId = self.tweetMap[followeeId][index]
                # Python heapq 預設為最小堆，因此對 time 加負號模擬最大堆
                heapq.heappush(maxHeap, [-time, tweetId, followeeId, index - 1])

        # 2. 每次彈出全域最新的推文，並補進該作者的上一篇推文，直到收集滿 10 篇
        while maxHeap and len(res) < 10:
            neg_time, tweetId, followeeId, next_index = heapq.heappop(maxHeap)
            res.append(tweetId)
            if next_index >= 0:
                time, next_tweetId = self.tweetMap[followeeId][next_index]
                heapq.heappush(maxHeap, [-time, next_tweetId, followeeId, next_index - 1])

        return res

    def follow(self, followerId: int, followeeId: int) -> None:
        if followerId != followeeId:
            self.followMap[followerId].add(followeeId)

    def unfollow(self, followerId: int, followeeId: int) -> None:
        self.followMap[followerId].discard(followeeId)
```


```Python
# 方法二：Deque 限制長度 + heapq.nlargest（工程簡潔版）
import heapq
from collections import defaultdict, deque
from typing import List

class Twitter:
    def __init__(self):
        self.time = 0
        # 因為最多只取前 10 篇，每位用戶本地最多只需保留最新 10 篇推文
        self.tweetMap = defaultdict(lambda: deque(maxlen=10))
        self.followMap = defaultdict(set)

    def postTweet(self, userId: int, tweetId: int) -> None:
        self.time += 1
        self.tweetMap[userId].append((self.time, tweetId))

    def getNewsFeed(self, userId: int) -> List[int]:
        followees = self.followMap[userId] | {userId}
        # 收集所有關注者各自的最近推文（每人最多 10 篇）
        candidates = []
        for f_id in followees:
            candidates.extend(self.tweetMap[f_id])
        # 直接取時間戳最大的前 10 筆
        top_10 = heapq.nlargest(10, candidates, key=lambda x: x[0])
        return [tweetId for _, tweetId in top_10]

    def follow(self, followerId: int, followeeId: int) -> None:
        if followerId != followeeId:
            self.followMap[followerId].add(followeeId)

    def unfollow(self, followerId: int, followeeId: int) -> None:
        self.followMap[followerId].discard(followeeId)
```

解釋: step1: 設計兩個核心雜湊表：`followMap`（用 `set` 記錄關注名單，達到 `O(1)` 新增與刪除）和 `tweetMap`（用 `list` 依序記錄每個人的推文）。 step2: 維護一個全域計數器 `self.time`，每次 `postTweet` 時遞增並連同 `tweetId` 一起存入，因為題目給的 `tweetId` **不保證按時間遞增**。 step3: 執行 `getNewsFeed(userId)` 時，把問題轉化為「合併 K 個已排序陣列（Merge K Sorted Lists），並取前 10 大」**。 step4: 先把所有關注者（包含 `userId` 自己）陣列**最尾端（最新）的那一則推文推入 `maxHeap`。 step5: 每次從 `maxHeap` 彈出時間戳最大的推文加入 `res`，同時看該作者是否還有更早一則推文（`next_index >= 0`），有的話補進堆中，重複最多 10 次即完成。

`followMap = {1: {2}}`, `tweetMap = {1: [[1, 5], [3, 7]], 2: [[2, 6]]}`, 呼叫 `getNewsFeed(1)` -> `result = [7, 6, 5]`

1. 初始化 `maxHeap` 放入各關注者最新推文：`[[-3, 7, user=1, next=0], [-2, 6, user=2, next=-1]]`
    
2. 第 1 次 pop：彈出 `tweetId = 7`（時間 3 最新），補入 user 1 的下一篇 `[-1, 5, user=1, next=-1]`
    
3. 第 2 次 pop：彈出 `tweetId = 6`（時間 2），user 2 的 `next=-1` 無推文可補
    
4. 第 3 次 pop：彈出 `tweetId = 5`（時間 1），堆空結束，回傳 `[7, 6, 5]`
    

解釋: 不需要把所有關注者的全部推文都倒出來重新排序。

因為每個用戶自己的推文列表天生就是按時間排好序的，只要用**最大堆（Max-Heap）維護每個列表當前的最前端指標**，就能像多路歸併一樣精準撈出前 10 篇最新動態。

### **LeetCode 355 - Design Twitter（設計 Twitter）**

這道題目是 **雜湊表（Hash Map）** 與 **優先佇列（Heap / Priority Queue）** 結合的經典系統設計題（Blind 75 & NeetCode 150 必考題）。核心考點在於如何高效率支援 `follow` / `unfollow` 的動態關係變更，以及在 `getNewsFeed` 時快速合併多位用戶的時間軸（Timeline）。

## **解法解析**

這道題是系統設計中經典的 **「推模式（Fan-out on Write） vs 拉模式（Fan-out on Read）」** 權衡，在演算法面試中標準採用**拉模式 + K-路歸併（K-way Merge）**。

### **思路**

1. **資料結構選型**
    
    - **`followMap`（關注關係）**：使用 `defaultdict(set)`。因為需要頻繁執行 `follow`（新增）、`unfollow`（移除）以及避免重複關注，`set` 提供均攤 **O(1)** 的操作效率（且刪除不存在的對象時用 `.discard()` 不會拋出 `KeyError`）。
        
    - **`tweetMap`（用戶推文）**：使用 `defaultdict(list)`。因為每次發推都是按時間順序追加在最後面，陣列尾端自然就是該用戶最新的推文。
        
    - **全域時間戳 `self.time`**：千萬不能直接用 `tweetId` 大小來比較新舊，必須自己維護遞增時間戳。
        
2. **將 `getNewsFeed` 轉化為「合併 K 個有序串列」**
    
    - 假設用戶 `u` 關注了 `K - 1` 個人（加上自己共 `K` 個人）。
        
    - 每個人都有一條按時間升序排列的推文陣列，我們要從這 `K` 條陣列中找出全域最新的 **10 則**。
        
    - 如果把所有人的所有推文全部混在一起排序，當某人發了 10 萬篇推文時會極度浪費時間；使用**大小最多為 `K` 的最大堆**，每次只追蹤每條陣列當前的「候選最新推文」，就能把排序規模壓到最小。
        
3. **代碼執行流程**
    
    - 透過 `followees = self.followMap[userId] | {userId}` 取得所有要看的名單（包含自己）。
        
    - 遍歷 `followees`，將每人 `tweetMap` 的最後一筆（索引 `len - 1`）壓入 `maxHeap`。
        
    - 進入 `while maxHeap and len(res) < 10:` 迴圈，彈出時間最新的一筆推文，並將該作者的前一筆（索引 `index - 1`）壓入堆中，直到集滿 10 筆或堆為空。
        

## **具體舉例**

假設 `user 1` 關注了 `user 2` 與 `user 3`，三人各自發佈的推文 `[time, tweetId]` 如下：

- `user 1`：`[[1, 10], [4, 40]]`
    
- `user 2`：`[[2, 20], [5, 50]]`
    
- `user 3`：`[[3, 30]]`
    

呼叫 `getNewsFeed(1)` 時（假設取前 3 篇示範），`maxHeap` 的變化如下：

|**步驟**|**操作動作**|**maxHeap 內部狀態 [-time, tweetId, user, next_idx]**|**res 結果**|
|---|---|---|---|
|初始化|放入 3 人各自最新推文|`[[-5, 50, u2, 0], [-4, 40, u1, 0], [-3, 30, u3, -1]]`|`[]`|
|第 1 步|Pop `50` (u2)，補入 u2 的 `[2, 20]`|`[[-4, 40, u1, 0], [-3, 30, u3, -1], [-2, 20, u2, -1]]`|`[50]`|
|第 2 步|Pop `40` (u1)，補入 u1 的 `[1, 10]`|`[[-3, 30, u3, -1], [-2, 20, u2, -1], [-1, 10, u1, -1]]`|`[50, 40]`|
|第 3 步|Pop `30` (u3)，u3 無更早推文|`[[-2, 20, u2, -1], [-1, 10, u1, -1]]`|`[50, 40, 30]`|

最終依序取得 `[50, 40, 30]`，完全符合由新到舊的順序，且無需掃描到底部的 `10` 與 `20`。

## **時間與空間複雜度分析**

設 `U` 為總用戶數，`T` 為總推文數，`K` 為單一用戶關注的人數（含自己）。

- **`postTweet`**：
    
    - **時間複雜度：O(1)**（陣列尾端 `append`）。
        
- **`follow` / `unfollow`**：
    
    - **時間複雜度：O(1)**（雜湊集合 `set` 新增與刪除）。
        
- **`getNewsFeed`**：
    
    - **時間複雜度：O(K log K)**（若不使用方法二的 `deque(maxlen=10)` 限制，最多需將 `K` 個關注者的最新推文建堆，接著進行最多 10 次 `heappop` 與 `heappush`，每次堆操作為 `O(log K)`，故總時間為 `O(K log K + 10 log K) = O(K log K)`）。
        
- **空間複雜度：O(U + T + K)**
    
    - 系統儲存所有關注關係 `O(U + E)` 與所有推文 `O(T)`；單次 `getNewsFeed` 的堆空間為 **O(K)**。
        

## **其他解法**

### **1. 全量收集排序法（Brute Force）**

- **思路**：
    
    - 執行 `getNewsFeed(userId)` 時，直接把 `userId` 及其所有關注者的全部推文複製到一個大陣列中，呼叫內建 `sort()` 按 `time` 降序排列後切片 `[:10]`。
        
- **時間複雜度**：`O(M log M)`（`M` 為所有關注者的推文總數，當推文極多時效能遠差於 K-路歸併）。
    
- **空間複雜度**：`O(M)`
    

### **2. 寫入擴散法（Fan-out on Write / Push Model）**

- **思路**：
    
    - 類似真實系統中對普通用戶的優化：每次 `postTweet` 時，直接把該推文推送到所有粉絲的 `newsFeed` 快取清單中，這樣 `getNewsFeed` 只要 `O(1)` 讀取。
        
    - **缺點**：在本題中因為有頻繁的 `follow` 和 `unfollow` 操作，一旦關注或取關，就必須動態重算整個歷史推播清單，維護成本極高；且遇到擁有千萬粉絲的「名人（Celebrity）」發推會造成寫入風暴。
        

## **總結**

- **面試最佳解**：**HashMap + Max-Heap（K-路歸併指標法）**，將時間複雜度控制在與歷史推文總數無關的 `O(K log K)`。
    
- **三大常見陷阱（Edge Cases）**：
    
    1. **誤用 `tweetId` 當時間**：`tweetId` 只是唯一識別碼，不保證遞增，務必自己維護 `self.time`。
        
    2. **忘記包含自己**：`getNewsFeed(userId)` 必須同時顯示 `userId` 自己發的推文。
        
    3. **自己關注/取關自己**：若直接把 `userId` 寫死進 `followMap[userId]`，當測試案例呼叫 `unfollow(1, 1)` 時會把自己的推文弄丟；因此建議在 `follow()` 過濾 `followerId != followeeId`，並在 `getNewsFeed()` 用聯集 `self.followMap[userId] | {userId}` 讀取。
        

## **系統設計與堆積（Heap / Design）類問題整理**

|**題號**|**題目名稱**|**問題類型**|**解法**|
|---|---|---|---|
|**LeetCode 355**|Design Twitter（設計 Twitter）|**物件導向設計 + 多路時間軸合併**|**HashMap + Set + Max-Heap（K-路歸併）**，按時間戳取最新前 10 則|
|**LeetCode 23**|Merge k Sorted Lists（合併 K 個升序鏈結串列）|**K-路歸併（K-way Merge）基礎原型**|**Min-Heap 維護 K 條串列當前頭節點**，時間 `O(N log K)` 空間 `O(K)`|
|**LeetCode 295**|Find Median from Data Stream（數據流的中位數）|**動態資料流維護中位數**|**對頂堆（最大堆存較小一半 + 最小堆存較大一半）**，查詢 `O(1)` 插入 `O(log N)`|
|**LeetCode 703**|Kth Largest Element in a Stream（數據流中的第 K 大元素）|**動態資料流維護 Top-K**|**固定大小為 K 的 Min-Heap**，堆頂即為第 K 大元素|
|**LeetCode 621**|Task Scheduler（任務排程器）|**帶冷卻時間 `n` 的高頻任務排程**|**Max-Heap + 佇列（Queue）** 模擬冷卻時間，或貪心公式計算總間隔|
|**LeetCode 146**|LRU Cache（LRU 快取機制）|**O(1) 查詢與淘汰最久未使用快取**|**HashMap + 雙向鏈結串列（Doubly Linked List）**，存取與更新皆為 `O(1)`|
|**LeetCode 460**|LFU Cache（LFU 最不經常使用快取）|**O(1) 按頻率與時間淘汰快取**|**雙雜湊表（Key 表 + 頻率對應雙向鏈結串列）**，維護 `min_freq` 指標|
|**LeetCode 380**|Insert Delete GetRandom O(1)|**O(1) 新增、刪除與隨機抽取元素**|**動態陣列（Array）+ HashMap 記錄索引**，刪除時與陣列尾端元素交換|

### **進階延伸題：合併 K 個升序鏈結串列（LeetCode 23）**

`Design Twitter` 中 `getNewsFeed` 的核心引擎，其實就是 LeetCode 23 的陣列版！掌握這題就能徹底打通所有「K-路歸併」題型：

Python

```
import heapq
from typing import List, Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        minHeap = []
        # 1. 將每條鏈結串列的頭節點放入最小堆（加入索引 i 避免 node.val 相同時比較 ListNode 報錯）
        for i, node in enumerate(lists):
            if node:
                heapq.heappush(minHeap, (node.val, i, node))
        
        dummy = ListNode(0)
        curr = dummy
        
        # 2. 每次彈出全域最小節點，並將該串列的下一個節點推入堆中
        while minHeap:
            val, i, node = heapq.heappop(minHeap)
            curr.next = node
            curr = curr.next
            if node.next:
                heapq.heappush(minHeap, (node.next.val, i, node.next))
                
        return dummy.next
```