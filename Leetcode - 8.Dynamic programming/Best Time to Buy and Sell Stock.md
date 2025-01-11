
**样例1**
```plain
输入: [3, 2, 3, 1, 2]
输出: 1
说明：你可以在第三天买入，第四天卖出，利润是 2 - 1 = 1
```
**样例2**
```plain
输入: [1, 2, 3, 4, 5]
输出: 4
说明：你可以在第0天买入，第四天卖出，利润是 5 - 1 = 4
```
**样例3**
```plain
输入: [5, 4, 3, 2, 1]
输出: 0
说明：你可以不进行任何操作然后也得不到任何利润
```


```python
    def max_profit(self, prices: List[int]) -> int:
        total = 0
        low, high = sys.maxsize, 0
        for x in prices:
            if x - low > total:
                total = x - low
            if x < low:
                low = x
        return total
```
pass