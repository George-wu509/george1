Lintcode 1270
给定一个任意的表示勒索信内容的字符串，和另一个字符串表示杂志的内容，写一个方法判断能否通过剪下杂志中的内容来构造出这封勒索信，若可以，返回 true；否则返回 false。

杂志字符串中的每一个字符仅能在勒索信中使用一次。

**样例 1**
```python
"""
输入 : ransomNote = "aa", magazine = "aab"
输出 : true
解析 : 勒索信的内容可以有杂志内容剪辑而来
```
**样例 2**
```python
"""
输入 : ransomNote = "aaa", magazine = "aab"
输出 : false
解析 : 勒索信的内容无法从杂志内容中剪辑而来
```


```python
class Solution:
    def can_construct(self, ransom_note: str, magazine: str) -> bool:
        if len(ransom_note) > len(magazine):
            return False
        return not collections.Counter(ransom_note) - collections.Counter(magazine)
```
pass