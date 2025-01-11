
**样例1**
```
输入: "I speak Goat Latin"
输出: "Imaa peaksmaaa oatGmaaaa atinLmaaaaa"
```
**样例2**
```
输入: "The quick brown fox jumped over the lazy dog"
输出: "heTmaa uickqmaaa rownbmaaaa oxfmaaaaa umpedjmaaaaaa overmaaaaaaa hetmaaaaaaaa azylmaaaaaaaaa ogdmaaaaaaaaaa"
```


```python
    def to_goat_latin(self, s: str) -> str:
        vowels = {"a", "e", "i", "o", "u", "A", "E", "I", "O", "U"}
        n = len(s)
        i, cnt = 0, 1
        words = list()

        while i < n:
            j = i
            while j < n and s[j] != " ":
                j += 1      
            cnt += 1
            if s[i] in vowels:
                words.append(s[i:j] + "m" + "a" * cnt)
            else:
                words.append(s[i+1:j] + s[i] + "m" + "a" * cnt)
            
            i = j + 1
        
        return " ".join(words)
```
pass