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





#### 79. 搜索单词

Given an m x n grid of characters board and a string word, return true if word exists in the grid.

The word can be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once.

![img](https://assets.leetcode.com/uploads/2020/11/04/word2.jpg)

```python
Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
Output: true
```

解法：DFS尝试不同的方向

```python
class Solution:
    def exist(self, board, word: str) -> bool:
        m = len(board)
        n = len(board[0])
        dx = [1,-1,0,0]
        dy = [0,0,1,-1]

        def DFS(board,word,x,y,idx,visited):
            if board[x][y]!= word[idx]:
                return False
            if idx == len(word)-1 and board[x][y] == word[idx]: ##终止条件
                return True
            ans = False
            for i in range(4):
                xx = x+dx[i]
                yy = y+dy[i]
                if xx < 0 or xx >= m or yy < 0 or yy >= n or visited[xx][yy]:
                    continue
                visited[xx][yy] = 1
                ans = ans or DFS(board,word,xx,yy,idx+1,visited)
                visited[xx][yy] = 0
            return ans
        ans = False
        for i in range(m):
            for j in range(n):
                visited = [[0]*n for _ in range(m)]
                visited[i][j] = 1
                ans = ans or (DFS(board,word,i,j,0,visited))
        return ans
```

- 易错点1：`visited[x][y]`是需要回溯的，对于那种需要向四方”尝试“DFS的问题，都需要回溯
- 易错点2：每个位置都需要重置visited矩阵，防止上一次的递归结果对这次产生干扰。