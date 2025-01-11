
**样例 1:**
```
输入:
"123"
6
输出: 
["1*2*3","1+2+3"]
```
**样例 2:**
```
输入:
"232"
8
输出: 
["2*3+2", "2+3*2"]
```
**样例 3:**
```
输入:
"105"
5
输出:
["1*0+5","10-5"]
```
**样例 4:**
```
输入:
"00"
0
输出:
["0+0", "0-0", "0*0"]
```
**样例 5:**
```
输入:
"3456237490",
9191 
输出: 
[]
```


```python
    def add_operators(self, num, target):
        def dfs(idx, tmp, tot, last, res):
            if idx == len(num):
                if tot == target:
                    res.append(tmp)
                return
            for i in range(idx, len(num)):
                x = int(num[idx: i + 1])
                if idx == 0:
                    dfs(i + 1, str(x), x, x, res)
                else:
                    dfs(i + 1, tmp + "+" + str(x), tot + x, x, res)
                    dfs(i + 1, tmp + "-" + str(x), tot - x, -x, res)
                    dfs(i + 1, tmp + "*" + str(x), tot - last + last * x, last * x, res)
                if x == 0:
                    break
        res = []
        dfs(0, "", 0, 0, res)
        return res
```
pass