# BFS

- 一般看到"最短距离"可以想到用BFS。

- 树的层次遍历

- 遍历整个图的所有节点也用BFS



BFS模板：

```python
class Solution:
    def pacificAtlantic(self, heights):
        m = len(heights)
        n = len(heights[0])
        visited = [[0]*n for _ in range(m)]  ##记录visited
        dx = [1,-1,0,0]  ##几个方向
        dy = [0,0,1,-1]
        queue = [] ##队列
        for i in range(m):
            queue.append((i,0,heights[i][0]))
            visited[i][0] = 1
        for i in range(n):
            queue.append((0,i,heights[0][i]))
            visited[0][i] = 1 ##入队种子，记录visited
        while(len(queue) != 0): ## 终止条件：queue非空
            front = queue[0]
            front_x = front[0]
            front_y = front[1]  
            front_height = front[2] ##取出头
            for i in range(4):
                xx = front_x+dx[i]  ##尝试四个方向
                yy = front_y+dy[i]
                if(xx<0 or xx >=m or yy < 0 or yy >= n or visited[xx][yy] or heights[xx][yy] < front_height): ##不符合条件：出界 or visited
                    continue
                queue.append((xx,yy,heights[xx][yy])) ##符合条件，入队！
                visited[xx][yy] = 1 ##记录走过
            queue = queue[1:] ##pop()
        print(visited)
```

