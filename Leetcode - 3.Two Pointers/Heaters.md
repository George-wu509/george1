Example
```python
输入：[1,2,3],[2]
输出：1
说明：唯一的一个加热器被放在2的位置，那么只要加热半径为1，就能覆盖到所有房屋了。
```

```python
输入：[1,2,3,4],[1,4]
输出：1
说明：两个加热器分别位于1和4，只需要加热半径为1，就能加热所有房屋了。
```

```python
import bisect
class Solution:
    """
    @param houses: positions of houses
    @param heaters: positions of heaters
    @return: the minimum radius standard of heaters
    """
    def find_radius(self, houses: List[int], heaters: List[int]) -> int:
        # Write your code here
        ans = 0
        heaters.sort()
        for house in houses:
            j = bisect.bisect_right(heaters, house)
            i = j - 1
            rightDistance = heaters[j] - house if j < len(heaters) else float('inf')
            leftDistance = house - heaters[i] if i >= 0 else float('inf')
            curDistance = min(leftDistance, rightDistance)
            ans = max(ans, curDistance)
        return ans
```
pass