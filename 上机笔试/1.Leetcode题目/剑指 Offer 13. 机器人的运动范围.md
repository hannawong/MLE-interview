#### [剑指 Offer 13. 机器人的运动范围](https://leetcode.cn/problems/ji-qi-ren-de-yun-dong-fan-wei-lcof/)

难度中等512

地上有一个m行n列的方格，从坐标 `[0,0]` 到坐标 `[m-1,n-1]` 。一个机器人从坐标 `[0, 0] `的格子开始移动，它每次可以向左、右、上、下移动一格（不能移动到方格外），也不能进入行坐标和列坐标的数位之和大于k的格子。例如，当k为18时，机器人能够进入方格 [35, 37] ，因为3+5+3+7=18。但它不能进入方格 [35, 38]，因为3+5+3+8=19。请问该机器人能够到达多少个格子？

 

**示例 1：**

```
输入：m = 2, n = 3, k = 1
输出：3
```



题解：

```python
class Solution:
    def movingCount(self, m: int, n: int, k: int) -> int:
        dx = [1,-1,0,0]
        dy = [0,0,1,-1]

        def is_valid(xx,yy,k):
            cnt = 0
            while xx :
                digit = xx % 10
                xx = xx // 10
                cnt += digit
            while yy:
                digit = yy % 10
                yy = yy // 10
                cnt += digit
            if cnt <= k:
                return True
            return False

        queue = [(0,0)]
        visited = [[0]*n for _ in range(m)]
        cnt = 1
        while len(queue):
            front = queue[0]
            front_i = front[0]
            front_j = front[1]
            visited[front_i][front_j] = 1
            for kk in range(4):
                xx = front_i + dx[kk]
                yy = front_j + dy[kk]
                if xx >= 0 and xx < m and yy >= 0 and yy < n and not visited[xx][yy] and is_valid(xx,yy,k):
                    visited[xx][yy] = 1
                    queue.append((xx,yy))
                    cnt += 1
            queue = queue[1:]
        return cnt
```



【易错】

- 数位从低到高输出：

```python
while xx :
                digit = xx % 10
                xx = xx // 10
```

注意不是<= 0, xx 最后就应该是0

- 在写循环变量的时候要格外注意是否有重名！！