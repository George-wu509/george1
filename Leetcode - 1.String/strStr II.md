Lintcode 594
实现时间复杂度为 O(n + m)的方法 `strStr`。  
`strStr` 返回目标字符串在源字符串中第一次出现的第一个字符的位置. 目标字串的长度为 _m_ , 源字串的长度为 _n_ . 如果目标字串不在源字串中则返回 -1。
**样例 1:**
```python
"""
输入：source = "abcdef"， target = "bcd"
输出：1
解释：
字符串第一次出现的位置为1。
```
**样例 2:**
```python
"""
输入：source = "abcde"， target = "e"
输出：4
解释：
字符串第一次出现的位置为4。
```


```python
def str_str2(self, source: str, target: str) -> int:
	if source is None or target is None:
		return -1
	m = len(target)
	n = len(source)

	if m == 0:
		return 0

	import random
	mod = random.randint(1000000, 2000000)
	hash_target = 0
	m26 = 1

	for i in range(m):
		hash_target = (hash_target * 26 + ord(target[i]) - ord('a')) % mod
		if hash_target < 0:
			hash_target += mod

	for i in range(m - 1):
		m26 = m26 * 26 % mod

	value = 0
	for i in range(n):
		if i >= m:
			value = (value - m26 * (ord(source[i - m]) - ord('a'))) % mod

		value = (value * 26 + ord(source[i]) - ord('a')) % mod
		if value < 0:
			value += mod

		if i >= m - 1 and value == hash_target:
			return i - m + 1

	return -1
```
pass


## 解法思路

本題 **`str_str2(self, source: str, target: str) -> int`** 的目標是找出 `target` 在 `source` 中的第一次出現的索引位置，如果 `target` 沒有出現在 `source` 中則返回 `-1`。

這題使用的是 **Rabin-Karp (滾動哈希) 字符串匹配演算法**，其核心思路是：

1. **計算 `target` 的哈希值 (`hash_target`)**
2. **使用滑動窗口計算 `source` 中長度為 `m` 的子字串的哈希值 (`value`)**，並與 `hash_target` 進行比較
3. **如果 `value == hash_target` 則進一步確認是否是相同的字串**
4. **使用滾動哈希公式來高效地更新 `value`**

---

## 變數表

|變數名稱|含義|
|---|---|
|`source`|原始字符串|
|`target`|需要查找的子字符串|
|`m`|`target` 字符串長度|
|`n`|`source` 字符串長度|
|`mod`|隨機選擇的大質數，用於減少哈希衝突|
|`hash_target`|`target` 的哈希值|
|`m26`|26 的 `m-1` 次方，作用是移除最左邊字符影響|
|`value`|當前滑動窗口子串的哈希值|

---

## 具體步驟

1. **邊界條件檢查**
    
    - 如果 `source` 或 `target` 為 `None`，則返回 `-1`
    - 如果 `target` 長度 `m` 為 `0`，根據題意直接返回 `0`（空字串應該匹配索引 `0`）
2. **哈希值初始化**
    
    - 隨機選擇一個 `mod`（大質數）來減少哈希衝突
    - 計算 `target` 的哈希值 `hash_target`
    - 計算 `m26 = 26^(m-1) % mod`，用於後續更新滑動窗口的哈希值
3. **使用滾動哈希法掃描 `source`**
    
    - 用變數 `value` 計算 `source` 中前 `m` 個字符的哈希值
    - 在遍歷 `source` 的過程中，使用 `m26` 來減去最左邊字符的影響，並加入新字符，維持一個長度為 `m` 的滑動窗口
    - 若 `value == hash_target` 則返回起始索引
4. **返回 `-1` 表示 `target` 沒有出現在 `source` 中**
    

---

## 時間與空間複雜度分析

- **時間複雜度**
    
    - 計算 `target` 的哈希值 `O(m)`
    - 初始化 `source` 前 `m` 個字符的哈希值 `O(m)`
    - 使用滾動哈希掃描 `source`，時間複雜度為 `O(n)`
    - 總體時間複雜度為 **`O(m + n)`**
- **空間複雜度**
    
    - 只使用了常數額外空間 `O(1)`

這比直接使用 **暴力法 `O(m * n)`** 的時間複雜度優秀得多。

---

## 其他解法想法（不含代碼）

1. **暴力解法 (`O(m * n)`)**
    
    - 直接在 `source` 中檢查每個長度為 `m` 的子字串是否等於 `target`
    - 時間複雜度高，適用於 `m` 很小的情況
2. **KMP（Knuth-Morris-Pratt 演算法，`O(m + n)`)**
    
    - 預處理 `target` 生成部分匹配表 (`prefix function`)
    - 在 `source` 上進行匹配時利用前綴信息快速跳過匹配失敗的部分
3. **Boyer-Moore (`O(n/m)` in best case, `O(mn)` in worst case)**
    
    - 使用 **壞字符規則** 和 **好後綴規則** 來大幅減少比較次數
    - 適用於長 `target` 和大 `source`
4. **Z-Algorithm (`O(m + n)`)**
    
    - 使用 **Z 函數** 預處理 `target + "$" + source`
    - 計算 `Z` 陣列來判斷 `target` 是否匹配
5. **二進制哈希（字符串哈希變體）**
    
    - 轉換 `source` 和 `target` 為二進制字符串來進行匹配，減少計算量
    - 在某些特殊情況下效率比 Rabin-Karp 更高

---

## 結論

Rabin-Karp 適用於大規模文本匹配，因為 **使用滾動哈希的方式大幅降低時間複雜度**。這使它比暴力解法更高效，也比 KMP 或 Boyer-Moore 更容易實作。