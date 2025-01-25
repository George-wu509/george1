32
给定两个字符串 `source` 和 `target`. 求 `source` 中最短的包含 `target` 中每一个字符的子串.

Example
样例 1：
输入：
source = "abc"
target = "ac"
输出：
"abc"
解释：
"abc" 是 source 的包含 target 的每一个字符的最短的子串。

样例 2：
输入：
source = "adobecodebanc"
target = "abc"
输出：
"banc"
解释：
"banc" 是 source 的包含 target 的每一个字符的最短的子串。

样例 3：
输入：
source = "abc"
target = "aa"
输出：
""
解释：
没有子串包含两个 'a'。


```python
from collections import defaultdict
class Solution:
    """
    @param source : A string
    @param target: A string
    @return: A string denote the minimum window, return "" if there is no such a string
    """
    def minWindow(self, source , target):
        # 初始化counter_s和counter_t
        counter_s = defaultdict(int)
        counter_t = defaultdict(int)
        for ch in target:
            counter_t[ch] += 1
        left = 0
        valid = 0
        # 记录最小覆盖子串的起始索引及长度
        start = -1
        minlen = float('inf')
        for right in range(len(source)):
            # 移动右边界, ch 是将移入窗口的字符
            ch = source[right]
            if ch in counter_t:
                counter_s[ch] += 1
                if counter_s[ch] == counter_t[ch]:
                    valid += 1
            
            # 判断左侧窗口是否要收缩
            while valid == len(counter_t):
                # 更新最小覆盖子串
                if right - left < minlen:
                    minlen = right - left
                    start = left
                # left_ch 是将移出窗口的字符
                left_ch = source[left]
                # 左移窗口
                left += 1
                # 进行窗口内数据的一系列更新
                if left_ch in counter_s:
                    counter_s[left_ch] -= 1
                    if counter_s[left_ch] < counter_t[left_ch]:
                        valid -= 1
        # 返回最小覆盖子串
        if start == -1:
            return ""
        return source[start: start + minlen + 1]
```
pass

解說
source = "adobecodebanc"
target = "abc"

Step1
建立兩個dict,  counter_s一個準備儲存source內子字串, 另一個counter_t則是將target轉成dict = {"a":1, "b":1, "c":1}

Step2
雙指針Left=0 and Right=0. Right一步步往右移, 並將字元加入counter_s. 並跟counter_t 比較. 如果Right所在的字元也在counter_t內, 則valid +1. 當valid = 3時代表目前Left到Right的子字串有target所有字元. 
"adobecodebanc"
 L=0    R=5

Step3
接下來開始移動Left指針往右移並跟counter_t比較試圖找到更短的子字串. 如果沒有則換成Right指針往右移, 回到step2