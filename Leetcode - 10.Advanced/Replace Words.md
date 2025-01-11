

**样例1:**
```
输入: 
dict = ["cat", "bat", "rat"]
sentence = "the cattle was rattled by the battery"
输出: 
"the cat was rat by the bat"
```
**样例 2:**
```
输入: 
dict = ["go", "begin", "make","end"]
sentence = "a good beginning makes a good ending"
输出: 
"a go begin make a go end"
```



```python
class Solution:
    def replace_words(self, dict: List[str], sentence: str) -> str:
        words = sentence.split(" ")
        for i in range(len(words)):
            for shortcut in dict:
                if shortcut in words[i]:
                    length = len(shortcut)
                    if length < len(words[i]) and words[i][:length] == shortcut:
                        words[i] = shortcut

        return " ".join(words)
```
pass