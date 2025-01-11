
**样例 1:**
```
输入：S = "abcd", indexes = [0,2], sources = ["a","cd"], targets = ["eee","ffff"]
输出："eeebffff"
解释：
"a" 从 S 中的索引 0 开始，所以它被替换为 "eee"。
"cd" 从 S 中的索引 2 开始，所以它被替换为 "ffff"。
```
**样例 2:**
```
输入：S = "abcd", indexes = [0,2], sources = ["ab","ec"], targets = ["eee","ffff"]
输出："eeecd"
解释：
"ab" 从 S 中的索引 0 开始，所以它被替换为 "eee"。
"ec" 没有从原始的 S 中的索引 2 开始，所以它没有被替换。
```


```python
    def find_replace_string(self, s: str, indexes: List[int], sources: List[str], targets: List[str]) -> str:
        cur_i = 0
        result = []

        zipped = sorted(zip(indexes, sources, targets))

        for i in range(len(zipped)):
            index = zipped[i][0]
            source = zipped[i][1]
            target = zipped[i][2]

            if cur_i < index:
                result.append(s[cur_i:index])
                cur_i = index
            
            if s[index:index + len(source)] == source:
                result.append(target)
                cur_i += len(source)

        if cur_i < len(s):
            result.append(s[cur_i:])

        return ''.join(result)
```
pass