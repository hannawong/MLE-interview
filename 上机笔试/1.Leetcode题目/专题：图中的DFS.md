# 专题：图中的DFS

模板是：

- 返回条件（如果有）
- 对于所有的邻居，如果符合条件的话就visit他! (必要时回溯)



##### 200. 岛屿数量

```python
给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。
岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。
此外，你可以假设该网格的四条边均被水包围。

示例 1：
输入：grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
输出：1
```



```python
class Solution:
    def numIslands(self, grid) -> int:
        m = len(grid)
        n = len(grid[0])
        visited = [[0]*n for _ in range(m)]

        def dfs(x,y):
            dx = [1,-1,0,0]
            dy = [0,0,1,-1]
            visited[x][y] = 1 ##visited
            for i in range(4): ##再考虑所有邻居
                xx = x+dx[i]; yy = y+dy[i]
                if xx >= 0 and xx < m and yy >= 0 and yy < n and not visited[xx][yy] and grid[xx][yy] == "1":
                    dfs(xx,yy)

        cnt = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == "1" and not visited[i][j]:
                    cnt += 1
                    dfs(i,j)
        return cnt
```

