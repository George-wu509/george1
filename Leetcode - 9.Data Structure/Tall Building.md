
```
输入:[5,3,8,3,2,5]
输出:[3,3,5,4,4,4]
解释:
当小Q处于位置0时，他能看到位置0，1，2的3栋高楼。
当小Q位于位置1时，他能看到位置0，1，2的3栋高楼。
当小Q处于位置2时，他可以向前看到位置0，1处的楼，向后看到位置3,5处的楼，加上第3栋楼，共可看到5栋楼。
当小Q处于位置3时，他能看到位置2，3，4，5的4栋高楼。
当小Q处于位置4时，他能看到位置2，3，4，5的4栋高楼。
当小Q处于位置5时，他能看到位置2，3，4，5的4栋高楼。
```


```python
    def tall_building(self, arr):
        n = len(arr);
        left = [0 for i in range(105000)];
        right = [0 for i in range(105000)];
        left_sum = [0 for i in range(105000)];
        right_sum = [0 for i in range(105000)];
        left[0] = -1;
        for i in range(1, n):
            x = i-1;
            while x > -1 and arr[x] <= arr[i]:
                x = left[x];
            left[i] = x;
        right[n-1] = n;
        for i in range(n-2, -1, -1):
            x = i + 1;
            while x < n and arr[x] <= arr[i]:
                x = right[x];
            right[i] = x;
        for i in range(0, n):
            if left[i] == -1:
                left_sum[i] = 1;
            else:
                left_sum[i] = left_sum[left[i]] + 1;
        for i in range(n-1, -1, -1):
            if right[i] == n:
                right_sum[i] = 1;
            else:
                right_sum[i] = right_sum[right[i]] + 1;
        ans = [0 for i in range(n)]
        for i in range(0,n):
            x = 0;
            if i-1 >= 0:
                x += left_sum[i-1];
            if i+1 < n:
                x += right_sum[i+1];
            ans[i] = x+1;
        return ans;
```
pass