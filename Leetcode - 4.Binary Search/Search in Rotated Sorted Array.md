Lintcode 62
给定一个有序数组，但是数组以某个元素作为支点进行了旋转(比如，`0 1 2 4 5 6 7` 可能成为`4 5 6 7 0 1 2`)。给定一个目标值`target`进行搜索，如果在数组中找到目标值返回数组中的索引位置，否则返回`-1`。你可以假设数组中不存在重复的元素。

**样例 1：**
输入：
```python
"""
数组 = [4, 5, 1, 2, 3]
target = 1
```
输出：
```python
"""
2
```
解释：
1在数组中对应索引位置为2。

**样例 2：**
输入：
```python
"""
数组 = [4, 5, 1, 2, 3]
target = 0
```
输出：
```python
"""
-1
```
解释：
0不在数组中，返回-1。


```python
def search(self, A, target):
	if not A:
		return -1
		
	start, end = 0, len(A) - 1
	while start + 1 < end:
		mid = (start + end) // 2
		if A[mid] >= A[start]:
			if A[start] <= target <= A[mid]:
				end = mid
			else:
				start = mid
		else:
			if A[mid] <= target <= A[end]:
				start = mid
			else:
				end = mid
				
	if A[start] == target:
		return start
	if A[end] == target:
		return end
	return -1
```
pass




請幫我比較兩種解法哪種更優化
lintcode 62
在旋转过的排序数组中搜索目标值，返回其下标。

method1: 
直接在旋转过的排序数组中用nums[k] > nums[k+1] 找旋转点, 找到旋轉點之後分成兩部分已排序的list也很容易就可以判斷目標值是位在哪個list, 再用二分法找target.

method2:
```python
method2:
def search(self, A, target):
	if not A:
		return -1
		
	start, end = 0, len(A) - 1
	while start + 1 < end:
		mid = (start + end) // 2
		if A[mid] >= A[start]:
			if A[start] <= target <= A[mid]:
				end = mid
			else:
				start = mid
		else:
			if A[mid] <= target <= A[end]:
				start = mid
			else:
				end = mid
				
	if A[start] == target:
		return start
	if A[end] == target:
		return end
	return -1
```

這兩種解法都是在旋轉過的排序數組中搜索目標值的有效方法，但它們在某些方面有所不同，並且 **`method2` 通常被認為是更優化的解法**，原因如下：

**Method 1 的分析：**

1. **尋找旋轉點：** 需要遍歷數組（在最壞情況下）來找到旋轉點，即 `nums[k] > nums[k+1]` 的位置。這一步的時間複雜度是 O(n)，其中 n 是數組的長度。
2. **劃分數組：** 找到旋轉點後，將數組劃分為兩個已排序的部分。這一步是 O(1) 的操作。
3. **判斷目標值在哪個部分：** 需要進行一次比較來確定目標值可能在哪個已排序的部分。這一步是 O(1) 的操作。
4. **二分查找：** 最後在確定的已排序部分執行二分查找，時間複雜度是 O(log n)。

**Method 1 的總體時間複雜度在最壞情況下是 O(n) + O(log n) = O(n)。**

**Method 2 的分析：**

這個方法直接在旋轉過的排序數組上進行修改過的二分查找，**不需要顯式地尋找旋轉點並劃分數組。**

1. **初始化：** 設定 `start` 和 `end` 指針。
2. **循環條件：** `while start + 1 < end:` 確保循環在 `start` 和 `end` 相鄰或重合時停止。
3. **計算中間點：** `mid = (start + end) // 2`。
4. **判斷中間點在哪個已排序的部分：**
    - `if A[mid] >= A[start]:` 表示中間點位於左邊已排序的部分（或整個數組未旋轉）。
        - `if A[start] <= target <= A[mid]:` 如果目標值在左邊已排序部分的範圍內，則將 `end` 移到 `mid`。
        - `else:` 否則，目標值在右邊未排序的部分，將 `start` 移到 `mid`。
    - `else:` 表示中間點位於右邊已排序的部分。
        - `if A[mid] <= target <= A[end]:` 如果目標值在右邊已排序部分的範圍內，則將 `start` 移到 `mid`。
        - `else:` 否則，目標值在左邊未排序的部分，將 `end` 移到 `mid`。
5. **檢查剩餘的兩個元素：** 循環結束後，`start` 和 `end` 要么指向目標值，要么目標值不在數組中。

**Method 2 的總體時間複雜度是 O(log n)。** 這是因為在每次循環中，搜索範圍都會減半。

**為什麼 Method 2 更優化：**

- **更好的時間複雜度：** Method 2 的時間複雜度是 O(log n)，而 Method 1 在最壞情況下是 O(n)。對於大型數組，這種差異會非常明顯。
- **更直接的解決方案：** Method 2 直接在旋轉後的數組上進行搜索，避免了尋找旋轉點的額外步驟，使得邏輯更加緊湊和高效。
- **更少的遍歷：** Method 1 可能需要一次遍歷找到旋轉點，然後再進行二分查找。Method 2 只需要一次對數級別的遍歷。

**總結：**

**`method2` 是更優化的解法**，因为它直接利用了旋转排序数组的特性，通过修改过的二分查找在 O(log n) 的时间复杂度内找到目标值，而不需要像 `method1` 那样先进行 O(n) 的操作来寻找旋转点。在实际应用中，尤其对于大型数组，`method2` 的性能会显著优于 `method1`。