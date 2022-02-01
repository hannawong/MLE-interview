# BFS

经典题：695. 岛屿的最大面积

注意visited的记录

```python
class Solution:
    def maxAreaOfIsland(self, grid) -> int:
        m = len(grid)
        n = len(grid[0])
        visited = [[0]*n for _ in range(m)]
        mmax = 0
        for i in range(m):
            for j in range(n):
                if not visited[i][j] and grid[i][j] == 1:
                    mmax = max(mmax,self.BFS(grid,visited,i,j))
        return mmax
    def BFS(self,grid,visited,begin_x,begin_y):
        m = len(grid)
        n = len(grid[0])
        dx = [1,-1,0,0]
        dy = [0,0,1,-1]
        queue = [(begin_x,begin_y)]
        cnt = 0
        while len(queue):
            front = queue[0]
            cnt += 1
            front_x = front[0]
            front_y = front[1]
            visited[front_x][front_y] = 1 ##在这里需要记录visited一次
            for i in range(4):
                xx = front_x + dx[i]
                yy = front_y + dy[i]
                if xx < 0 or xx >= m or yy < 0 or yy >= n or visited[xx][yy] or not grid[xx][yy]:
                    continue
                queue.append((xx,yy))
                visited[xx][yy] = 1 ##这里还需要记录visited一次
            queue = queue[1:]
        return cnt
```

