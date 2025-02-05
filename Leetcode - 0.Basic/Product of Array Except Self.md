
**样例1**
```python
输入: [1,2,3,4]
输出: [24,12,8,6]
解释:
2*3*4=24
1*3*4=12
1*2*4=8
1*2*3=6
```
**样例2**
```python
输入: [2,3,8]
输出: [24,16,6]
解释:
3*8=24
2*8=16
2*3=6
```


```python
def product_except_self(self, nums: List[int]) -> List[int]:
	# write your code here
	length = len(nums)
	answer = [0]*length
	
	# answer[i] 表示索引 i 左侧所有元素的乘积
	# 因为索引为 '0' 的元素左侧没有元素， 所以 answer[0] = 1
	answer[0] = 1
	for i in range(1, length):
		answer[i] = nums[i - 1] * answer[i - 1]
	
	# R 为右侧所有元素的乘积
	# 刚开始右边没有元素，所以 R = 1
	R = 1;
	for i in reversed(range(length)):
		# 对于索引 i，左边的乘积为 answer[i]，右边的乘积为 R
		answer[i] = answer[i] * R
		# R 需要包含右边所有的乘积，所以计算下一个结果时需要将当前值乘到 R 上
		R *= nums[i]
	
	return answer
```
pass


## **LintCode 1310：数组除了自身的乘积**

---

### **题目描述**

给定一个整数数组 `nums`，返回一个数组 `answer`，其中 `answer[i]` 等于 **`nums` 除了 `nums[i]` 之外所有元素的乘积**。

- 不能使用除法 `"/"`。
- **示例**

`输入: [1,2,3,4]  输出: [24,12,8,6]`

**解释**

複製編輯

`answer[0] = 2 × 3 × 4 = 24 answer[1] = 1 × 3 × 4 = 12 answer[2] = 1 × 2 × 4 = 8 answer[3] = 1 × 2 × 3 = 6`

---

## **解法：前缀乘积 + 后缀乘积（双指针）**

### **核心思路**

1. **前缀乘积 `answer[i]`**
    
    - `answer[i]` 存储的是 **`nums[0]` 到 `nums[i-1]`** 的乘积（即 `nums[i]` 左侧的乘积）。
    - 初始化 `answer[0] = 1`（因为 `nums[0]` 左侧没有元素）。
2. **后缀乘积 `R`**
    
    - `R` 代表 `nums[i]` **右侧所有元素的乘积**。
    - 逆序遍历 `nums`，每次更新 `answer[i] = answer[i] * R`，再更新 `R`。

---

### **代码解析**
```python
def product_except_self(self, nums: List[int]) -> List[int]:
    length = len(nums)
    answer = [0] * length  # 初始化结果数组

    # 计算左侧前缀乘积
    answer[0] = 1
    for i in range(1, length):
        answer[i] = nums[i - 1] * answer[i - 1]

    # 计算右侧乘积并更新 answer
    R = 1  # 右侧乘积初始化为 1
    for i in reversed(range(length)):
        answer[i] = answer[i] * R  # 乘上右侧乘积
        R *= nums[i]  # 更新右侧乘积

    return answer

```

---

## **逐步执行分析**

### **输入**

`nums = [1,2,3,4]`

---

### **步骤 1：计算左侧前缀乘积**

|`i`|`nums[i]`|`answer[i]` (左侧乘积)|
|---|---|---|
|0|1|1|
|1|2|`1 × 1 = 1`|
|2|3|`1 × 2 = 2`|
|3|4|`1 × 2 × 3 = 6`|

此时：

`answer = [1, 1, 2, 6]`

---

### **步骤 2：计算右侧乘积 `R`**

从右向左遍历 `nums`：

|`i`|`nums[i]`|`R` (右侧乘积)|`answer[i]` (乘上 `R`)|
|---|---|---|---|
|3|4|1|`6 × 1 = 6`|
|2|3|`4 × 1 = 4`|`2 × 4 = 8`|
|1|2|`3 × 4 = 12`|`1 × 12 = 12`|
|0|1|`2 × 12 = 24`|`1 × 24 = 24`|

最终 `answer = [24, 12, 8, 6]`。

---

## **时间与空间复杂度分析**

1. **时间复杂度**
    
    - **`O(n)`**：遍历 `nums` 两次（计算左侧 + 计算右侧），因此是 `O(n)`。
2. **空间复杂度**
    
    - **`O(1)`**（若不计 `answer` 作为输出数组）。
    - 只用了一个额外变量 `R`，所以是 **常数空间 `O(1)`**。

---

## **其他解法**

### **1. 暴力法 `O(n^2)`**

- 遍历 `nums`，每次计算 `nums[i]` 除了自身的乘积。
- **缺点**：时间复杂度 **`O(n^2)`**，对于大数组不可行。

### **2. 直接前缀 + 后缀数组**

- 维护 `left` 和 `right` 两个数组，分别存左侧和右侧乘积。
- **空间复杂度 `O(n)`**，可优化为 `O(1)`。
```python
def product_except_self(nums: List[int]) -> List[int]:
    length = len(nums)
    left = [1] * length
    right = [1] * length
    answer = [0] * length

    # 计算左侧乘积
    for i in range(1, length):
        left[i] = left[i - 1] * nums[i - 1]

    # 计算右侧乘积
    for i in range(length - 2, -1, -1):
        right[i] = right[i + 1] * nums[i + 1]

    # 计算最终结果
    for i in range(length):
        answer[i] = left[i] * right[i]

    return answer

```

**时间复杂度 `O(n)`，空间复杂度 `O(n)`**。

---

## **方法比较**

|方法|时间复杂度|空间复杂度|适用情况|
|---|---|---|---|
|**双指针（最佳解法）**|`O(n)`|`O(1)`|**最优解**|
|暴力法|`O(n^2)`|`O(1)`|**数据小**|
|前缀 & 后缀数组|`O(n)`|`O(n)`|**适用于不能修改输入数据的情况**|

🚀 **双指针方法是最优解，时间 `O(n)`，空间 `O(1)`，适合大规模数据！**