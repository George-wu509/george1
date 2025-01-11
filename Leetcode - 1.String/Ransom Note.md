
**样例 1**
```
输入 : ransomNote = "aa", magazine = "aab"
输出 : true
解析 : 勒索信的内容可以有杂志内容剪辑而来
```
**样例 2**
```
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