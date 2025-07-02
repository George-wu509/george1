
Lintcode 3493

给定一个非空非负整数数组 `nums`，其长度为 `n`，再给定一个正整数 `d`，其中 `d <= n`，在 `nums` 中选择一段长度不超过 `d` 的连续子数组，把其中的元素都变为 `0`。

最后给定一个非负整数 `p`，在**变化后的数组**中找到一个**尽可能长的连续子数组**，使得该子数组的和不超过 `p`，并返回这个子数组的长度。

```python
输入：
nums = [3,4,1,9,4,1,7,1,3]
d = 2
p = 7
输出：
5
解释：
将 nums 的子数组 [9,4] 元素变为 0
此时 nums 为 [3,4,1,0,0,1,7,1,3]
最长且总和不超过 7 的子数组是 [4,1,0,0,1]，长度为 5
```


```python
输入：
nums = [1,1,1,1,1,1]
d = 3
p = 7
输出：
6
解释：
nums 总和不超过 7，最长子数组为 nums 本身，即返回 nums 的长度 6
```



```python
def longest_subarray(self, nums, d, p):
	left, right = 0, d
	res = d
	n = len(nums)
	queue = collections.deque()
	max_queue = collections.deque()
	prefix = [0] * (n + 1)
	for i in range(1, n + 1):
		prefix[i] = prefix[i - 1] + nums[i - 1]
	s = []
	for i in range(n - d + 1):
		s.append(prefix[i + d] - prefix[i])
	queue.append(s[0])
	max_queue.append(s[0])
	while right <= n - 1:
		while max_queue and max_queue[-1] < s[right - d + 1]:
			max_queue.pop()
		queue.append(s[right - d + 1])
		max_queue.append(s[right - d + 1])
		if prefix[right + 1] - prefix[left] - max_queue[0] <= p:
			res = max(res, right - left + 1)
		else:
			left += 1
			if max_queue[0] == queue[0]:
				max_queue.popleft()
			queue.popleft()
		right += 1
	return res
```


#### 方法：单调队列

我们先明确以下几点：

1. 题目要求选择一段**长度不超过 `d` **的连续子数组并置 `0`，但是**选择长度为 `d` 的子数组并置 `0` 为最优选择**
2. 如果确定选择一个子数组作为最终答案，应该把其中和最大的长度为 `d` 的子数组全部置 `0`
3. **`d` 一定是一个可行解**，需要从每个 `j >= d` 的右端点，查找最优的最终答案数组的左端点 `i`
4. 需要通过预处理前缀和的方式来获取 `nums[i]` 到 `nums[j]` 区间的子数组和

#### 过程

1. 由上述结论 3，我们设置初始值：`i = 0`，`j = d - 1`，`res = d`
2. 设 `s[i]` 为**以 `i` 为左端点，长度为 `d` 的 `nums` 子数组和**，表达式为：

s[i]=nums[i]+nums[i+1]+...+nums[i+d−1](i=0,1,...,n−d)s[i]=nums[i]+nums[i+1]+...+nums[i+d−1](i=0,1,...,n−d)

3. 对于每一个右端点 `j`，用一个单调队列 queuequeue 维护 `s[i], s[i + 1], ..., s[j - d + 1]`
4. 计算 nums(i,j)nums(i,j) 的区间和与单调队列 queuequeue 和题目给定的 pp 的关系

- 如果：

(nums[i]+nums[i+1]+...+nums[j])−max(queue)>p(nums[i]+nums[i+1]+...+nums[j])−max(queue)>p

  则说明当前 `nums[i]` 到 `nums[j]` 区间的子数组和不符合要求，需要移动区间左端点，即 `i++`，并在单调队列中弹出 `s[i]`。

- 如果：

(nums[i]+nums[i+1]+...+nums[j])−max(queue)≤p(nums[i]+nums[i+1]+...+nums[j])−max(queue)≤p

  则说明当前 `nums[i]` 到 `nums[j]` 区间的子数组和符合要求，更新 resres 的值：

res=max(res,j−i+1)res=max(res,j−i+1)

#### 注意

s[i]s[i] 与前缀和 prefix[i]prefix[i] 并不等价，对于样例一给出的数据，`d = 2`，`nums` 与 `prefix` 和 `s` 数组如下图所示：  

![8.jpg](https://media-lc.lintcode.com/u_394541/202304/8a01366e16094f7ca184e661ef029cc8/8.jpg)

可以看出，`s` 数组可以由前缀和数组元素相减得出，比如：

 d=2 prefix[2]=nums[0]+nums[1] prefix[4]=nums[0]+nums[1]+nums[2]+nums[3] s[2]=nums[2]+nums[3]=prefix[4]−prefix[2] d=2 prefix[2]=nums[0]+nums[1] prefix[4]=nums[0]+nums[1]+nums[2]+nums[3] s[2]=nums[2]+nums[3]=prefix[4]−prefix[2]

即：

s[i]=prefix[i+d]−prefix[i]s[i]=prefix[i+d]−prefix[i]

最后我们通过处理滑动窗口方式去寻找 `j - i + 1` 的最大值  

![10.jpg](https://media-lc.lintcode.com/u_394541/202304/66db09d5f9fb441ab43b47aa77662693/10.jpg)

![11.jpg](https://media-lc.lintcode.com/u_394541/202304/c9c772c8c82441ebb95fa0c31f619d5d/11.jpg)

![12.jpg](https://media-lc.lintcode.com/u_394541/202304/4d8f0e28f6b24e9285b20f6ceb0cae39/12.jpg)

### 题目理解

**目标：** 在一个非空非负整数数组 `nums` 中，通过一次操作，找到一个最长的连续子数组。

**操作步骤：**

1. **选择并清零：** 在 `nums` 中选择一段**长度不超过 `d`** 的连续子数组，将其所有元素变为 0。
    
2. **查找最长子数组：** 在经过上述变化后的数组中，找到一个尽可能长的连续子数组，使得该子数组的**和不超过 `p`**。
    
3. **返回长度：** 返回这个最长子数组的长度。
    

### 代码解释

您提供的代码使用了滑动窗口（`left`, `right`）、前缀和（`prefix`）和双端队列（`queue`, `max_queue`）来解决这个问题。

**核心思想：** 这段代码的核心思想是遍历所有可能的“变化后的数组中的连续子数组”（即 `nums[left:right+1]` 这个窗口），并对于每个这样的窗口，尝试找到一个长度为 `d` 的子数组（**注意：这里代码只考虑了长度为 `d` 的子数组进行清零操作，并没有考虑长度小于 `d` 的情况，这是与题目描述中“不超过 d”的一个潜在差异点**），将其清零后，判断这个窗口的剩余和是否满足 `<= p` 的条件。如果满足，就更新最长子数组的长度。

**变量解释：**

- `nums`: 输入的非空非负整数数组。
    
- `d`: 允许清零的子数组的最大长度。
    
- `p`: 变化后子数组的和不能超过的阈值。
    
- `left`, `right`: 滑动窗口的左右边界（在 `nums` 数组中的索引）。
    
- `res`: 存储找到的最长满足条件的子数组的长度，初始化为 `d`（因为至少可以清零一个长度为 `d` 的子数组，使其和为0，满足条件）。
    
- `n`: `nums` 数组的长度。
    
- `prefix`: 前缀和数组。`prefix[i]` 存储 `nums[0]` 到 `nums[i-1]` 的和。`prefix[0]` 为 0。
    
    - 作用：快速计算任意子数组的和。例如，`nums[i:j]` 的和为 `prefix[j] - prefix[i]`。
        
- `s`: 存储所有长度为 `d` 的连续子数组的和。`s[k]` 对应 `nums[k:k+d]` 的和。
    
    - 作用：预先计算所有可能被清零的长度为 `d` 的子数组的和。
        
- `queue`: 一个普通双端队列，用于存储当前滑动窗口 `[left, right]` 中包含的 `s` 数组中的元素（即所有落在 `[left, right]` 范围内的长度为 `d` 的子数组的和）。
    
- `max_queue`: 一个单调递减的双端队列，用于存储当前滑动窗口 `[left, right]` 中所有长度为 `d` 的子数组和中的最大值。
    
    - 作用：高效地找到当前窗口内，通过清零操作可以使和减少最多的那个长度为 `d` 的子数组的和（即最大的那个和）。
        

**代码流程：**

1. **初始化：**
    
    - `left`, `right` 初始化为 `0` 和 `d`。`res` 初始化为 `d`。
        
    - 计算 `nums` 的前缀和数组 `prefix`。
        
    - 计算所有长度为 `d` 的子数组的和，并存储在列表 `s` 中。
        
2. **滑动窗口初始化：**
    
    - 将 `s` 中的第一个元素（即 `nums[0:d]` 的和）加入 `queue` 和 `max_queue`。
        
3. **滑动窗口遍历：**
    
    - `while right <= n - 1`: 循环直到 `right` 遍历到 `nums` 数组的末尾。
        
    - **维护 `max_queue`：**
        
        - 当 `max_queue` 不为空且其尾部元素小于新的 `d` 长度子数组和 `s[right - d + 1]` 时，弹出 `max_queue` 的尾部元素，直到 `max_queue` 保持单调递减。
            
        - 将 `s[right - d + 1]`（对应 `nums[right - d + 1 : right + 1]` 的和）加入 `queue` 和 `max_queue`。
            
    - **判断条件：**
        
        - `current_window_sum = prefix[right + 1] - prefix[left]`：计算当前滑动窗口 `nums[left:right+1]` 的总和。
            
        - `max_d_subarray_sum = max_queue[0]`：获取当前窗口内所有长度为 `d` 的子数组和中的最大值。
            
        - `if current_window_sum - max_d_subarray_sum <= p:`
            
            - 这个条件模拟了清零操作：如果将当前窗口内和最大的长度为 `d` 的子数组清零，剩余的和是否不超过 `p`。
                
            - 如果满足，则更新 `res = max(res, right - left + 1)`，因为找到了一个更长的满足条件的子数组。
                
    - **窗口收缩或扩展：**
        
        - 如果条件不满足 (`else` 分支)：说明当前窗口 `[left, right]` 即使清零了和最大的 `d` 长度子数组，也无法满足条件。此时需要收缩窗口，将 `left` 向右移动一位。
            
            - 如果 `max_queue` 的头部元素（当前窗口最大 `d` 长度子数组和）恰好是 `queue` 的头部元素（即将移出窗口的 `d` 长度子数组和），则同时从 `max_queue` 头部弹出。
                
            - 从 `queue` 头部弹出元素。
                
        - `right += 1`：无论条件是否满足，窗口的右边界 `right` 都会向右移动一位，扩展窗口。
            
4. **返回结果：** 循环结束后，`res` 即为所求的最长子数组的长度。
    

### 示例分析：`nums = [3,4,1,9,4,1,7,1,3]`, `d=2`, `p=7`

**1. 初始化**

- `nums = [3,4,1,9,4,1,7,1,3]`
    
- `d = 2`, `p = 7`, `n = 9`
    
- `left = 0`, `right = 2`, `res = 2`
    
- `queue = deque()`, `max_queue = deque()`
    

**2. 计算前缀和 `prefix`**

- `prefix = [0, 3, 7, 8, 17, 21, 22, 29, 30, 33]`
    

**3. 计算所有长度为 `d=2` 的子数组的和 `s`**

- `s[0]` (对应 `[3,4]`) = `prefix[2] - prefix[0] = 7 - 0 = 7`
    
- `s[1]` (对应 `[4,1]`) = `prefix[3] - prefix[1] = 8 - 3 = 5`
    
- `s[2]` (对应 `[1,9]`) = `prefix[4] - prefix[2] = 17 - 7 = 10`
    
- `s[3]` (对应 `[9,4]`) = `prefix[5] - prefix[3] = 21 - 8 = 13`
    
- `s[4]` (对应 `[4,1]`) = `prefix[6] - prefix[4] = 22 - 17 = 5`
    
- `s[5]` (对应 `[1,7]`) = `prefix[7] - prefix[5] = 29 - 21 = 8`
    
- `s[6]` (对应 `[7,1]`) = `prefix[8] - prefix[6] = 30 - 22 = 8`
    
- `s[7]` (对应 `[1,3]`) = `prefix[9] - prefix[7] = 33 - 29 = 4`
    
- 所以 `s = [7, 5, 10, 13, 5, 8, 8, 4]`
    

**4. 滑动窗口初始化**

- `queue.append(s[0])` -> `queue = [7]`
    
- `max_queue.append(s[0])` -> `max_queue = [7]`
    

**5. 滑动窗口遍历 (`while right <= 8`)**

- **Iteration 1: `right = 2`**
    
    - 当前窗口 `nums[0:3]` = `[3,4,1]`
        
    - 新加入的 `d` 长度子数组和是 `s[1]` (`nums[1:3]`) = `5`。
        
    - `max_queue` 维护：`max_queue = [7]`，因为 `5` 小于 `7`，直接加入 `max_queue = [7, 5]`。
        
    - `queue.append(5)` -> `queue = [7, 5]`
        
    - **判断：** `(prefix[3] - prefix[0]) - max_queue[0]` = `(8 - 0) - 7` = `1`
        
    - `1 <= 7` (True)。满足条件。
        
    - `res = max(2, 2 - 0 + 1)` = `max(2, 3)` = `3`。
        
    - `right = 3`。
        
- **Iteration 2: `right = 3`**
    
    - 当前窗口 `nums[0:4]` = `[3,4,1,9]`
        
    - 新加入的 `d` 长度子数组和是 `s[2]` (`nums[2:4]`) = `10`。
        
    - `max_queue` 维护：`max_queue = [7, 5]`。`5 < 10`，弹出 `5`。`max_queue = [7]`。`7 < 10`，弹出 `7`。`max_queue = []`。
        
    - `max_queue.append(10)` -> `max_queue = [10]`。
        
    - `queue.append(10)` -> `queue = [7, 5, 10]`
        
    - **判断：** `(prefix[4] - prefix[0]) - max_queue[0]` = `(17 - 0) - 10` = `7`
        
    - `7 <= 7` (True)。满足条件。
        
    - `res = max(3, 3 - 0 + 1)` = `max(3, 4)` = `4`。
        
    - `right = 4`。
        
- **Iteration 3: `right = 4`**
    
    - 当前窗口 `nums[0:5]` = `[3,4,1,9,4]`
        
    - 新加入的 `d` 长度子数组和是 `s[3]` (`nums[3:5]`) = `13`。
        
    - `max_queue` 维护：`max_queue = [10]`。`10 < 13`，弹出 `10`。`max_queue = []`。
        
    - `max_queue.append(13)` -> `max_queue = [13]`。
        
    - `queue.append(13)` -> `queue = [7, 5, 10, 13]`
        
    - **判断：** `(prefix[5] - prefix[0]) - max_queue[0]` = `(21 - 0) - 13` = `8`
        
    - `8 <= 7` (False)。不满足条件。
        
    - **收缩窗口：** `left = 1`。
        
        - `max_queue[0]` (13) 不等于 `queue[0]` (7)，所以 `max_queue` 不弹出。
            
        - `queue.popleft()` -> `queue = [5, 10, 13]`
            
    - `right = 5`。
        
- **Iteration 4: `right = 5`**
    
    - 当前窗口 `nums[1:6]` = `[4,1,9,4,1]`
        
    - 新加入的 `d` 长度子数组和是 `s[4]` (`nums[4:6]`) = `5`。
        
    - `max_queue` 维护：`max_queue = [13]`。`5` 小于 `13`，直接加入 `max_queue = [13, 5]`。
        
    - `queue.append(5)` -> `queue = [5, 10, 13, 5]`
        
    - **判断：** `(prefix[6] - prefix[1]) - max_queue[0]` = `(22 - 3) - 13` = `19 - 13` = `6`
        
    - `6 <= 7` (True)。满足条件。
        
    - `res = max(4, 5 - 1 + 1)` = `max(4, 5)` = `5`。
        
    - `right = 6`。
        
- **Iteration 5: `right = 6`**
    
    - 当前窗口 `nums[1:7]` = `[4,1,9,4,1,7]`
        
    - 新加入的 `d` 长度子数组和是 `s[5]` (`nums[5:7]`) = `8`。
        
    - `max_queue` 维护：`max_queue = [13, 5]`。`5 < 8`，弹出 `5`。`max_queue = [13]`。`8` 小于 `13`，直接加入 `max_queue = [13, 8]`。
        
    - `queue.append(8)` -> `queue = [5, 10, 13, 5, 8]`
        
    - **判断：** `(prefix[7] - prefix[1]) - max_queue[0]` = `(29 - 3) - 13` = `26 - 13` = `13`
        
    - `13 <= 7` (False)。不满足条件。
        
    - **收缩窗口：** `left = 2`。
        
        - `max_queue[0]` (13) 不等于 `queue[0]` (5)，所以 `max_queue` 不弹出。
            
        - `queue.popleft()` -> `queue = [10, 13, 5, 8]`
            
    - `right = 7`。
        
- **Iteration 6: `right = 7`**
    
    - 当前窗口 `nums[2:8]` = `[1,9,4,1,7,1]`
        
    - 新加入的 `d` 长度子数组和是 `s[6]` (`nums[6:8]`) = `8`。
        
    - `max_queue` 维护：`max_queue = [13, 8]`。`8` 不小于 `8`，直接加入 `max_queue = [13, 8, 8]`。
        
    - `queue.append(8)` -> `queue = [10, 13, 5, 8, 8]`
        
    - **判断：** `(prefix[8] - prefix[2]) - max_queue[0]` = `(30 - 7) - 13` = `23 - 13` = `10`
        
    - `10 <= 7` (False)。不满足条件。
        
    - **收缩窗口：** `left = 3`。
        
        - `max_queue[0]` (13) 不等于 `queue[0]` (10)，所以 `max_queue` 不弹出。
            
        - `queue.popleft()` -> `queue = [13, 5, 8, 8]`
            
    - `right = 8`。
        
- **Iteration 7: `right = 8`**
    
    - 当前窗口 `nums[3:9]` = `[9,4,1,7,1,3]`
        
    - 新加入的 `d` 长度子数组和是 `s[7]` (`nums[7:9]`) = `4`。
        
    - `max_queue` 维护：`max_queue = [13, 8, 8]`。`4` 小于 `8`，直接加入 `max_queue = [13, 8, 8, 4]`。
        
    - `queue.append(4)` -> `queue = [13, 5, 8, 8, 4]`
        
    - **判断：** `(prefix[9] - prefix[3]) - max_queue[0]` = `(33 - 8) - 13` = `25 - 13` = `12`
        
    - `12 <= 7` (False)。不满足条件。
        
    - **收缩窗口：** `left = 4`。
        
        - `max_queue[0]` (13) 等于 `queue[0]` (13)，所以 `max_queue.popleft()` -> `max_queue = [8, 8, 4]`。
            
        - `queue.popleft()` -> `queue = [5, 8, 8, 4]`
            
    - `right = 9`。
        

**循环结束，因为 `right` (9) 不再 `<= n - 1` (8)。**

**最终结果：`res = 5`**

### 代码与题目描述的潜在差异

如上所述，这段代码在计算 `s` 数组时，只考虑了长度**等于 `d`** 的子数组的和。而在 `max_queue` 中维护的也是这些长度为 `d` 的子数组的最大和。

题目描述中提到的是“选择一段长度**不超过 `d`** 的连续子数组”。这意味着，除了长度为 `d` 的子数组，我们还可以选择长度为 `1, 2, ..., d-1` 的子数组进行清零。为了使变化后的数组和最小，我们应该清零当前窗口内所有长度不超过 `d` 的子数组中和最大的那个。

如果题目严格要求考虑所有长度不超过 `d` 的子数组，那么 `s` 数组和 `max_queue` 的逻辑需要进行修改，以能够找到并跟踪所有长度 `1` 到 `d` 的子数组的最大和。这通常会使问题变得更复杂，可能需要一个更通用的数据结构（如线段树或更复杂的滑动窗口最大值结构）来维护所有长度在 `1` 到 `d` 之间子数组的最大和。

然而，对于您提供的这段代码，它解决的是一个简化版本的问题：即假设我们总是清零一个长度**等于 `d`** 的子数组。在许多编程竞赛问题中，为了简化，"不超过 d" 可能会在实现时被简化为 "等于 d"，特别是当 `d` 是一个关键参数时。