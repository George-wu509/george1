Lintcode 648
一个单词的缩写根据以下的形式`<第一个字母><数字><最后一个字母>`，数字表示被缩写部分的长度。下面是一些缩写的例子

```python
"""
a) it                      --> it    (没有缩写)

     1
b) d|o|g                   --> d1g

              1    1  1
     1---5----0----5--8
c) i|nternationalizatio|n  --> i18n

              1
     1---5----0
d) l|ocalizatio|n          --> l10n
```

假设你有一个字典和给你一个单词，判断这个单词的缩写在字典中是否是唯一的。当字典中的**其他单词的缩写**均与它不同的时候， 这个单词的缩写是唯一的.

**样例1**
```python
"""
输入：
[ "deer", "door", "cake", "card" ]
isUnique("dear")
isUnique("cart")
输出：
false
true
解释：
字典中所有单词的缩写为 ["d2r", "d2r", "c2e", "c2d"].
"dear" 的缩写是 "d2r" , 在字典中。
"cart" 的缩写是 "c2t" , 不在字典中。
```
**样例2**
```python
"""
输出：
isUnique("cane")
isUnique("make")
输出：
false
true
解释：
字典中所有单词的缩写为 ["d2r", "d2r", "c2e", "c2d"].
"cane" 的缩写是 "c2e" , 在字典中。
"make" 的缩写是 "m2e" , 不在字典中。
```



```python
class ValidWordAbbr:
    
    def __init__(self, dictionary):
        # do intialization if necessary
        self.map = {}
        for word in dictionary:
            abbr = self.word_to_abbr(word)
            if abbr not in self.map:
                self.map[abbr] = set() 
            self.map[abbr].add(word)

    def word_to_abbr(self, word):
        if len(word) <= 1:
            return word
        return word[0] + str(len(word[1:-1])) + word[-1]
        
    def isUnique(self, word):
        # write your code here
        abbr = self.word_to_abbr(word)
        if abbr not in self.map:
            return True
        for word_in_dict in self.map[abbr]:
            if word != word_in_dict:
                return False
        return True
```
pass