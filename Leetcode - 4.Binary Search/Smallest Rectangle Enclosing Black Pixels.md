Lintcode 600
一个由二进制矩阵表示的图，`0` 表示白色像素点，`1` 表示黑色像素点。黑色像素点是联通的，即只有一块黑色区域。像素是水平和竖直连接的，给一个黑色像素点的坐标 `(x, y)` ，返回囊括所有黑色像素点的矩阵的最小面积。

**样例 1:**
```python
"""
输入：["0010","0110","0100"]，x=0，y=2
输出：6
解释：
矩阵左上角坐标是(0, 1), 右下角的坐标是(2, 2)
```
**样例 2:**
```python
"""
输入：["1110","1100","0000","0000"], x = 0, y = 1
输出：6
解释：
矩阵左上角坐标是(0,  0), 右下角坐标是(1, 2)
```


```python
 def min_area(self, image: List[List[str]], x: int, y: int) -> int:
    # 矩阵行数和列数
    m, n = len(image), len(image[0])

    # 二分查找函数
    def binarySearch(image, low, high, minR, maxR, goLower, isRow):
        while low < high:
            mid = (low + high) // 2
            hasBlackPixel = any(image[mid][j] == '1' for j in range(minR, maxR)) \
            if isRow else any(image[i][mid] == '1' for i in range(minR, maxR))
            
            if hasBlackPixel == goLower:
                high = mid
            else:
                low = mid + 1
        return low
    
    # 使用二分查找确定上边界
    top = binarySearch(image, 0, x, 0, n, True, True)
    # 使用二分查找确定下边界
    bottom = binarySearch(image, x + 1, m, 0, n, False, True)
    # 使用二分查找确定左边界
    left = binarySearch(image, 0, y, top, bottom, True, False)
    # 使用二分查找确定右边界
    right = binarySearch(image, y + 1, n, top, bottom, False, False)
    
    # 计算矩形面积
    return (bottom - top) * (right - left)
```
pass