

|                           |     |
| ------------------------- | --- |
| [[ 左右指針 - 384無重複字符的最長子串]] |     |
|                           |     |

 左右指針 - 384無重複字符的最長子串
 [[Longest Substring Without Repeating Characters]]
```
leetcode question Longest Substring Without Repeating Characters(给定一个字符串，请找出其中无重复字符的最长子字符串)以下是算法, 是否外層迴圈 for i in range(n) 迭代左邊界 i。內層 while 迴圈負責擴展右邊界 j? 為何for loop要用左指針, while loop要用右指針, 如果調換for loop要用右指針, while loop要用左指針是否有差別或者建議怎麼寫?
```

**完全正確。** 在你提供的程式碼中，外層的 `for i in range(n)` 確實是在枚舉**左邊界 `i`**，而內層的 `while` 迴圈則負責不斷向右擴展**右邊界 `j`**，直到遇到重複字元或字串結尾為止。

### 1. 為什麼這段寫法用「左 `for`、右 `while`」？

這種寫法的思考邏輯是：**「固定起點 `i`，測試終點 `j` 最遠能推到哪裡」**。

- **半開區間 `[i, j)`：** 當內層 `while` 迴圈停止時，`j` 停在「第一個重複字元」或 `n` 的位置。因此，當前合法無重複的子字串範圍其實是 `i` 到 `j - 1`，長度剛好是 `(j - 1) - i + 1 = j - i`。
    
- **為什麼 `j` 不用歸零？** 當 `i` 往右移動一格變成 `i + 1` 時，原本 `[i + 1, j - 1]` 這段區間內一定也沒有重複字元，所以 `j` 只需要從原地繼續往右試探即可，這讓整體時間複雜度維持在 $O(N)$。
    

### 2. 如果調換成「右 `for`、左 `while`」有什麼差別？

將兩者調換是**完全可行**的，而且在演算法刷題中，**「外層 `for` 迭代右指標、內層 `while` 收縮左指標」反而是更主流、更推薦的滑動視窗（Sliding Window）標準模板**。

#### 調換後的寫法（推薦模板）：
```python
def length_of_longest_substring(self, s: str) -> int:
    unique_chars = set()
    left = 0
    longest = 0
    
    # 外層 for 負責擴展右邊界 right
    for right in range(len(s)):
        # 若新加入的 s[right] 導致重複，內層 while 負責收縮左邊界 left
        while s[right] in unique_chars:
            unique_chars.remove(s[left])
            left += 1
            
        unique_chars.add(s[right])
        # 此時合法視窗為閉區間 [left, right]，長度為 right - left + 1
        longest = max(longest, right - left + 1)
        
    return longest
```

#### 兩種寫法的核心差異比較

|**比較維度**|**原寫法（左 for、右 while）**|**調換後（右 for、左 while）**|
|---|---|---|
|**思考視角**|固定起點，盡量往後**延伸**|遍歷每個終點，不合法時從左側**淘汰**|
|**視窗區間**|左閉右開 `[i, j)`，長度為 `j - i`|左右皆閉 `[left, right]`，長度為 `right - left + 1`|
|**`while` 條件**|需額外檢查邊界 `j < n` 避免越界|**不需檢查邊界**（因為 `s[right]` 必在集合中，`left` 最多追到 `right` 就會停下）|
|**時間／空間複雜度**|時間 $O(N)$，空間 $O(\vert{}\Sigma\vert{})$|時間 $O(N)$，空間 $O(\vert{}\Sigma\vert{})$（完全相同）|

### 3. 實戰建議

建議優先熟練**調換後的版本（右 `for`、左 `while`）**，原因有兩點：

1. **減少邊界 Bug：** 不需要在 `while` 裡寫 `j < n`，視窗永遠是直覺的閉區間 `[left, right]`，每一步迴圈結束時都能保證視窗是合法的。
    
2. **模板通用性高：** 絕大多數滑動視窗題目（如 _Minimum Size Subarray Sum_、_Longest Repeating Character Replacement_）都遵循「右指標 `for` 迴圈無腦往前推進、狀態不合法時用左指標 `while` 迴圈收縮」的套路，記憶一套模板就能解決所有同類型題目。