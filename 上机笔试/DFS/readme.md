# DFS

#### 1. 连通块数量

##### 1.1 省份数量

```
有 n 个城市，其中一些彼此相连，另一些没有相连。如果城市 a 与城市 b 直接相连，且城市 b 与城市 c 直接相连，那么城市 a 与城市 c 间接相连。

省份 是一组直接或间接相连的城市，组内不含其他没有相连的城市。

给你一个 n x n 的矩阵 isConnected ，其中 isConnected[i][j] = 1 表示第 i 个城市和第 j 个城市直接相连，而 isConnected[i][j] = 0 表示二者不直接相连。

返回矩阵中 省份 的数量。
```

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/number-of-provinces

**示例 1：**

![img](https://assets.leetcode.com/uploads/2020/12/24/graph1.jpg)

```
输入：isConnected = [[1,1,0],[1,1,0],[0,0,1]]
输出：2
```



```python
class Solution {
public:
    int findCircleNum(vector<vector<int>>& isConnected) {
        int m = isConnected.size();
        int n = isConnected[0].size();
        vector<int> visited;
        for(int i = 0;i < n;i++){
            visited.push_back(0);
        }
        int cnt = 0;
        for(int i = 0;i<n;i++){
            if(visited[i] == 0){
                DFS(isConnected,visited,i,n);
                cnt ++;
            }
        }
        return cnt;

    }
    void DFS(vector<vector<int>> & isConnected, vector<int> & visited, int start, int n){
        if(start < 0 || start >= n || visited[start]) return; #不能走的位置
        visited[start] = 1; ##走！
        for(int i = 0;i < n;i++){
            if(i != start && isConnected[start][i]){
                DFS(isConnected,visited,i,n);
            }
        }
    }
};
```



##### 1.2 岛屿数量

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

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/number-of-islands

```python
class Solution {
public:
    int numIslands(vector<vector<char>>& grid) {
        int m = grid.size();
        int n = grid[0].size();
        int** visited = new int*[m];
        for(int i = 0;i<m;i++){
            visited[i] = new int[n];
        }
        for(int i = 0;i<m;i++){
            for (int j = 0;j<n;j++){
                visited[i][j] = 0;
            }
        }
        int cnt = 0;
        for(int i = 0;i<m;i++){
            for(int j = 0;j<n;j++){
                if(grid[i][j] == '1' && !visited[i][j]){
                    DFS(grid,visited,i,j,m,n);
                    cnt ++;
                }
            }
        }
        return cnt;
    }
    void DFS(vector<vector<char>>& grid, int ** & visited, int x, int y, int m,int n){//标visited
        if(x < 0 || x >= m || y<0 || y>=n || visited[x][y] || grid[x][y] == '0') return; //终止条件
        visited[x][y] = 1;
        DFS(grid,visited,x-1,y,m,n);
        DFS(grid,visited,x+1,y,m,n);
        DFS(grid,visited,x,y-1,m,n);
        DFS(grid,visited,x,y+1,m,n);
    }
};
```



#### 2. 拓扑排序 / 环路检测

Leetcode [207. 课程表](https://link.zhihu.com/?target=https%3A//leetcode-cn.com/problems/course-schedule/)

```python
你这个学期必须选修 numCourses 门课程，记为 0 到 numCourses - 1 。
在选修某些课程之前需要一些先修课程。 先修课程按数组 prerequisites 给出，其中 prerequisites[i] = [ai, bi] ，表示如果要学习课程 ai 则 必须 先学习课程  bi 。
例如，先修课程对 [0, 1] 表示：想要学习课程 0 ，你需要先完成课程 1 。
请你判断是否可能完成所有课程的学习？如果可以，返回 true ；否则，返回 false 。

示例 1：
输入：numCourses = 2, prerequisites = [[1,0]]
输出：true
解释：总共有 2 门课程。学习课程 1 之前，你需要完成课程 0 。这是可能的。
```

- 每当有节点被标记为visited，则将其压入栈S
- 一旦发现后向边BACKWARD（指向discovered）, 则报告非DAG并退出

```c++
class Solution {
public:
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        int m = prerequisites.size();
        vector<vector<int>> graph;
        graph.resize(numCourses);
        for(int i = 0;i<m;i++){
            graph[prerequisites[i][1]].push_back(prerequisites[i][0]); //构建邻接矩阵
        }
        int* status = new int [numCourses]; //每个课程的状态，undiscover, discover, visited
        for(int i = 0;i<numCourses;i++){
            status[i] = 0;
        }
        bool ans = true;
        for(int i = 0;i<numCourses;i++){
            if(status[i] == 0){
                ans = ans && DFS(graph,status,i);
            }
        }
        return ans;

    }
    bool DFS(vector<vector<int>> & graph, int * & status, int now){
        status[now] = 1; //discovered
        for(int i = 0;i<graph[now].size();i++){ //遍历neighbor
            int neighbor = graph[now][i];
            if(status[neighbor] == 0){
                DFS(graph,status,neighbor);
            } 
            if(status[neighbor] == 1){ //discovered
                return false;
            }
        }
        status[now] = 2; //visited
        return true;
    }

};
```

