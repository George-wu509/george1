

|                                      |     |
| ------------------------------------ | --- |
| [[#### Python 效能之鬼！你應該學會使用的 deque]]  |     |
| [[#### Monotone Queue 是什麼？用來解什麼題目？]] |     |
|                                      |     |
|                                      |     |


#### Python 效能之鬼！你應該學會使用的 deque
```
Python 效能之鬼！你應該學會使用的 deque
```

https://myapollo.com.tw/blog/python-deque/
deque 是堆疊(stack)與佇列(queue)的一般化(generalization)，也就是說你可以把 deque 當作 stack 或者 queue 使用都沒問題。

可能有人會好奇—「 list 也可以當 stack 或者 queue 使用，為什麼 Python 還要特別提供 1 個 deque 呢？」

![[deque.png]]



#### Monotone Queue 是什麼？用來解什麼題目？
```
Monotone Queue 是什麼？用來解什麼題目？
```

**定義**：單調隊列是一種特殊的雙端隊列 (Double-ended Queue)，它不僅支援從兩端進出元素，更重要的是**隊列內部的元素保持著單調遞增或單調遞減的性質**。

**核心思想**：及早淘汰那些「既不優秀，又比別人老」的元素（例如在找最大值時，數值比別人小，且位置比別人前面的元素）。

**常用來解決的題型**： 單調隊列專門用來解決 **「區間最值問題」**，特別是當區間大小固定或有一定限制的滑動區間，並且要求整體時間複雜度為 $O(N)$ 時。常見應用包括：

1. **Sliding Window Maximum / Minimum** (本題，LeetCode 239)。
    
2. **動態規劃 (Dynamic Programming) 的狀態轉移優化**：當 DP 狀態轉移方程式需要在一個連續區間內找最大/最小值時，可以使用單調隊列將找極值的時間複雜度從 $O(K)$ 降到 $O(1)$。
    
3. **最短連續子陣列之和 / 條件限制下的區間問題** (例如 LeetCode 862. Shortest Subarray with Sum at Least K)。