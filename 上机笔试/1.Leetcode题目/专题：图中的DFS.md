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
        cnt = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1' and visited[i][j] == 0:
                    visited[i][j] == 1
                    self.DFS(grid,visited,i,j)
                    cnt += 1
        return cnt 
        
        
        
    def DFS(self,grid,visited,x,y):
        m = len(grid)
        n = len(grid[0])

        dx = [1,-1,0,0]
        dy = [0,0,1,-1]

        for i in range(4):
            xx = x+dx[i]
            yy = y+dy[i]
            if xx < 0 or xx >= m or yy < 0 or yy >= n or visited[xx][yy] == 1 or grid[xx][yy] == '0': ##邻居不符合条件
                continue
            visited[xx][yy] = 1 ##visit邻居
            self.DFS(grid,visited,xx,yy)
```

