# 图

### **1. 一些定义**

- G = (V,E)

- - Vertex(顶点): n = |V|
  - Edges(边): e = |E|

- undirected graphs(无向图) & directed graphs (有向图)

![img](https://pic2.zhimg.com/80/v2-874a34669aafc418c71842d72c9cba79_1440w.jpg)

- 节点的**度**(degree): 

- - 对于无向图，deg(v) = number of edges incident to v;

  - 对于有向图，

  - - indeg(v) = number of edges entering v;
    - outdeg(v) = number of edges leaving v

  - **[定理**] 在一个图里，所有节点的度之和 = 边数*2。这是因为一条边对整个图度数的贡献是2.

- 路径：

- - **path:**  ![[公式]](https://www.zhihu.com/equation?tex=%5C%7Bx_1%2Cx_2%2C...x_n%5C%7D) such that consecutive vertex are adjacent
  - **simple path:** all the vertex are distinct
  - **simple cycle:** a simple path that ends where it starts

![img](https://pic4.zhimg.com/80/v2-e2e162d5a0c78f6e36dac5fbe583480b_1440w.jpg)

- 连通性

- - **connected(连通):** there is a path between **every pair of vertices**
  - The **connected component(连通分量)** of a node ![[公式]](https://www.zhihu.com/equation?tex=u) is the set of all nodes in the graph reachable by a path from ![[公式]](https://www.zhihu.com/equation?tex=u).
  - A directed graph is **strongly connected（强连通）** if for every pair of vertices ![[公式]](https://www.zhihu.com/equation?tex=u%2C+v) , there is a path from u to v and from v to u.
  - The **strongly connected component(强连通分量)** of a node u in a directed graph is the set of nodes v in the graph such that there is a path from u to v and from v to u.

### 2. 图的表示

### 2.1 邻接矩阵 (adjacency matrix)

![img](https://pic3.zhimg.com/80/v2-4a644a413b6526c61ff9ea2bd79ce5a6_1440w.jpg)

例如：

![img](https://pic2.zhimg.com/80/v2-d28d2ef2a587dfdc7f7c9967561b0a69_1440w.jpg)

【优缺点】

![img](https://pic2.zhimg.com/80/v2-412c30f23d5563b083ff8a5d2002ae15_1440w.jpg)

### 2.2 邻接表 (adjacency list)

为了解决邻接矩阵处理稀疏图时复杂度过高的问题，引入邻接表。

邻接表有n个entries，分别对应n个节点。每个节点引出来一串列表，表示和该节点连接的所有节点。

例如：

![img](https://pic4.zhimg.com/80/v2-a1aedfa578bd3cb5958b0893ea0b7a83_1440w.jpg)


 【优缺点】

![img](https://pic4.zhimg.com/80/v2-50d2d49d161904143f55202b40e8895b_1440w.jpg)

### 3. 宽度优先搜索(BFS)

用一个起始的点s作为“火种”，一层一层向外燃烧，直到把所有和s连通的点都遍历一遍。

![img](https://pic4.zhimg.com/80/v2-471ea3d724f6f5762e7b26f56efbe1bf_1440w.jpg)

用队列来实现BFS，看下面这个图来了解队列的作用：

![img](https://pic1.zhimg.com/80/v2-c0c0df0c3fa873d79617432b043017ec_1440w.jpg)TREE指的是正常的边，而CROSS是指发现下一个节点已经被visit过了

![img](https://pic3.zhimg.com/80/v2-49d87d3a24e6e1acdd79b2dd3cf13e8a_1440w.jpg)

### 3.2 BFS的一些应用

### 3.2.1 连通块个数

```text
【题目】给一个01矩阵，1代表是陆地，0代表海洋， 如果两个1相邻，那么这两个1属于同一个岛。我们只考虑上下左右为相邻。岛屿: 相邻陆地可以组成一个岛屿（相邻:上下左右） 。判断岛屿个数。
例如：
输入
[
[1,1,0,0,0],
[0,1,0,1,1],
[0,0,0,1,1],
[0,0,0,0,0],
[0,0,1,1,1]
]
对应的输出为3。
```

对于每个位置，都做一遍BFS，把它能够到达的点全部遍历一遍，构成一个连通块。这样就求出了所有连通块的个数。下面的代码可以看作BFS最简单的模板，一定要隔一段时间写一遍，不然就忘了....

```cpp
class Solution {
public:
    /**
     * 判断岛屿数量
     * @param grid char字符型vector<vector<>>
     * @return int整型
     */
     struct node{ //节点的结构体
         int x,y;
         node(int xx,int yy){x = xx;y = yy;}
     };
    void bfs(vector<vector<char> >& grid, int ** visited, int start_i,int start_j){
        int dx[] = {0,0,1,-1};
        int dy[] = {1,-1,0,0};
        int m = grid.size();
        int n = grid[0].size();
        queue<node*> Q;
        Q.push(new node(start_i,start_j)); //put a fire seed
        while(!Q.empty()){
            node * front = Q.front(); //get the front node
            int x = front->x;
            int y = front->y;
            for(int k = 0;k<4;k++){ // try 4 directions...
                int xx = x+dx[k];
                int yy = y+dy[k];
                //but you can't land the next node in somewhere!!
                if(xx<0||xx>=m||yy<0||yy>=n||visited[xx][yy]==1||grid[xx][yy]=='0') continue; 
                node* Node = new node(xx,yy);
                grid[xx][yy] = '0';
                Q.push(Node);
            }
            Q.pop();
        }
    }
    int solve(vector<vector<char> >& grid) {
        // write code here
        int cnt = 0;
        int m = grid.size();
        int n = grid[0].size();
        int ** visit = new int*[m];
        for(int i = 0;i<m;i++)
            visit[i] = new int[n];
        for(int i = 0;i<m;i++)
            for(int j = 0;j<n;j++)
                visit[i][j] = 0;
        for(int i = 0;i<m;i++){
            for(int j = 0;j<n;j++){
                if(grid[i][j] == '1'){
                    cnt++;
                    bfs(grid,visit,i,j);
                }
            }
        }
        return cnt;
    }
};
```

### **3.2.2 判断是否是二部图/2-着色问题**

**【定义: Bipartite graphs(二部图)】**vertice##s can be split into two subsets such that there are no edges between vertices in the same subset.

![img](https://pic3.zhimg.com/80/v2-b9618672d73bd04b6ce3dd774829c29a_1440w.jpg)

判断一个图是否是二分图 ![[公式]](https://www.zhihu.com/equation?tex=%5CLeftrightarrow) 图的节点是否可以二-染色（相邻节点不能染成相同颜色）。二分图显然是可以二染色的，就像这样：

![img](https://pic2.zhimg.com/80/v2-2be22852c1b30422652a9b0db021c029_1440w.jpg)

[Leetcode [785. 判断二分图](https://link.zhihu.com/?target=https%3A//leetcode-cn.com/problems/is-graph-bipartite/)]



![img](https://pic4.zhimg.com/80/v2-c13b1c59c890f3f76f41e5f3e50e7df7_1440w.jpg)

```cpp
输入：graph = [[1,2,3],[0,2],[0,1,3],[0,2]] //用邻接表表示
输出：false
解释：不能将节点分割成两个独立的子集，以使每条边都连通一个子集中的一个节点与另一个子集中的一个节点。
```

思路：

- 任选一个节点开始，将其染成红色，并从该节点开始对整个无向图进行遍历；

- 在遍历的过程中，如果我们通过节点 ![[公式]](https://www.zhihu.com/equation?tex=u) 遍历到了节点 ![[公式]](https://www.zhihu.com/equation?tex=v) （即 ![[公式]](https://www.zhihu.com/equation?tex=u) 和 ![[公式]](https://www.zhihu.com/equation?tex=v) 在图中有一条边直接相连），那么会有两种情况：

- - 如果 ![[公式]](https://www.zhihu.com/equation?tex=v) 未被染色，那么我们将其染成与 ![[公式]](https://www.zhihu.com/equation?tex=u) 不同的颜色，并把 ![[公式]](https://www.zhihu.com/equation?tex=v+) 入队。
  - 如果 ![[公式]](https://www.zhihu.com/equation?tex=v) 被染色，并且颜色与 ![[公式]](https://www.zhihu.com/equation?tex=u) 相同，那么说明给定的无向图不是二分图。我们可以直接退出遍历并返回False。

- 返回 True。

```cpp
class Solution {
public:
    bool isBipartite(vector<vector<int>>& graph) {
        int n = graph.size(); //number of nodes
        int* color = new int[n+5];//color of nodes, 2: uncolor, 0: red, 1:blue
        for(int i = 0;i < n;i++) color[i] = 2; //they are uncolored in the beginning
        queue<int> Q; //a queue of nodes
        for(int first_node = 0;first_node<n;first_node++){ //可能有多个连通块！！！
            if(color[first_node]!=2) continue; //has already been colored...
            Q.push(first_node); //push a seed
            color[first_node] = 1; //color it!
            while(!Q.empty()){
            	int front = Q.front();
            	int front_color = color[front];
            	int adj_size = graph[front].size(); //how many nodes are adjacent to front
            	for(int i = 0;i<adj_size;i++){
                	int adj = graph[front][i];
                	int adj_color = color[adj];
                	if(adj_color == 2) { //uncolored
                    	color[adj] = 1-front_color; //then color it;
                    	Q.push(adj); //push it into queue
                	}
                	else if(adj_color == front_color)//can't have same color
                    	return false;
            	}
            	Q.pop();
        	}
        }
        return true;
    }
};
```

注意，这里的坑是可能会有**多个连通块**！！所以必须对每个未染色节点都跑一遍这个bfs算法。

（没想到随手写了个时间击败99.6%用户的解法，纪念一下 

![img](https://pic4.zhimg.com/80/v2-2fa226363a2f036dde1b5b14d8308877_1440w.jpg)

### 4. 深度优先搜索(DFS)

> 悔相道之不察兮，延伫乎吾将反。回朕车以复路兮，及行迷之未远。

### 4.1 定义

**【DFS定义】:** starting from a vertex s, explore the graph as **deeply** as possible, then **backtrack**.

1. Try the first edge out of s, towards some node v.
2. continue from v until you reach a dead end, that is a node whose neighbors have all been **explored**.
3. Backtrack to the first node with an **unexplored** neighbor and repeat 2. 

对树而言，等效于**先序遍历**(先遍历根，再遍历左右子树)。DFS也的确会构造出原图的一棵**支撑树**（DFS tree）。

![img](https://pic2.zhimg.com/80/v2-75c0379aa1ab770b34e1e1a0baaaa8d1_1440w.jpg)一个连通图对应一个DFS tree

![img](https://pic4.zhimg.com/80/v2-9dfc43a2359f316bfb0eab40afaef667_1440w.jpg)多个连通图对应一个DFS森林

### 4.2 示例

### 4.2.1 无向图

在无向图中，有两类边：

- TREE：指向UNDISCOVERD节点。它们构成了一个DFS tree。
- BACKWARD: 指向已经DISCOVER的节点。BACKWARD边不出现在DFS tree中。 

![img](https://pic4.zhimg.com/80/v2-34bee7191acb604b42b57bcfef7c6a63_1440w.jpg)TREE边是正常的边

![img](https://pic2.zhimg.com/80/v2-7b7368658e826bf369e6e22625113da1_1440w.jpg)如果一个节点没有与其相连的UNDISCOVER节点，标记为visited；BACKWARD边指向自己的祖先

![img](https://pic1.zhimg.com/80/v2-69f5284d1691db92777d8b1d882c05cc_1440w.jpg)

![img](https://pic1.zhimg.com/80/v2-d4376fc2ecaa702dbd0aa38ae83c8c90_1440w.jpg)

### 4.2.2 有向图

首先定义一个节点的**start**和**finish**时间。对于节点 ![[公式]](https://www.zhihu.com/equation?tex=u) ， ![[公式]](https://www.zhihu.com/equation?tex=start%28u%29) 就是节点初次被访问、进栈的时间； ![[公式]](https://www.zhihu.com/equation?tex=finish%28u%29) 就是节点u的所有邻居都被访问，u已经完成了历史使命、出栈的时间。

**【定理一：括号引理】**

![img](https://pic2.zhimg.com/80/v2-2162383d7c60bca59d0a1d349976b305_1440w.jpg)粉色：FORWARD，绿色：TREE，黄色：CROSS，蓝色：BACKWARD

**【定理二】**Intervals  ![[公式]](https://www.zhihu.com/equation?tex=%5Bstart%28u%29%2C+f+inish%28u%29%5D)  and ![[公式]](https://www.zhihu.com/equation?tex=%5Bstart%28v%29%2C+f+inish%28v%29%5D) either contain each other (u is an ancestor of v or vice versa)，or they are disjoint.

**【定理三】**If s was the first vertex pushed in the stack and v is the last, the vertices currently in the stack form an **s-v path**.

**【边的分类】**

![img](https://pic4.zhimg.com/80/v2-b6f0e1a74a64fde7f846de50599a7073_1440w.jpg)

1. **FORWARD**:  指向其已经**完全访问过**的**后代**。Edge (u, v) ∈ E is a forward edge if ![[公式]](https://www.zhihu.com/equation?tex=start%28u%29%3C+start%28v%29%3C+f+inish%28v%29%3C+f+inish%28u%29) 
2. **BACKWARD:**  指向其已经**DISCOVER**，但是没有完全访问完的**祖先**。  
3. **CROSS**:  指向已经**完全访问过**的节点。Edge (u, v)∈E is a cross edge if ![[公式]](https://www.zhihu.com/equation?tex=start%28v%29%3C+f+inish%28v%29%3C+start%28u%29%3C+f+inish%28u%29) . u和v没有承袭关系，是属于两棵树/一棵树的两个分支子树。

下面看一个例子：

![img](https://pic4.zhimg.com/80/v2-a33f54bfb7a8c4b5aff1b5b942569ad7_1440w.jpg)

![img](https://pic1.zhimg.com/80/v2-1a16f90d6c8bafa149f95988e52fb418_1440w.jpg)FORWARD：指向visited的节点。有承袭关系。

![img](https://pic1.zhimg.com/80/v2-3f8fdace412e02a8ea3e9d50235ef4e8_1440w.jpg)BACKWARD: 指向已经DISCOVER但是没有visited的节点。有承袭关系。BACKWARD边会形成环路

![img](https://pic3.zhimg.com/80/v2-e668b16bab312fbc1c0be9c07aafca62_1440w.jpg)CROSS指向已经visited的节点，没有承袭关系。所以叫CROSS

![img](https://pic3.zhimg.com/80/v2-72a64d8cc62496e13367a963d4f5e506_1440w.jpg)

![img](https://pic1.zhimg.com/80/v2-bd843a824e934c7c35b5db8f650e2aec_1440w.jpg)

![img](https://pic4.zhimg.com/80/v2-00b70ea702d6342b7059de21406b2c9b_1440w.jpg)

### 【DFS模板】

![img](https://pic3.zhimg.com/80/v2-198a1becfd8b124257671a6f3eb25a22_1440w.jpg)

![img](https://pic4.zhimg.com/80/v2-4d0388c2cdcf150c0dc71562e1b2832f_1440w.jpg)

### 4.2 DFS的一些应用

**4.2.1 环的检测/ 拓扑排序**

对于任何有向无环图(Directed Acyclic Graph, **DAG**), 必定有一种拓扑排序的方法...所以**拓扑排序和检测环路在本质上是一个问题**。

这是因为，任何DAG都必定至少有一个零入度的顶点m。只要把m去掉，剩下的图还是DAG，就又有零入度顶点... 如果m不唯一，那么拓扑排序的方法也不唯一。

**1）零入度算法**

每次都去掉零入度的节点，入栈。

![img](https://pic2.zhimg.com/80/v2-8b8a9b4afd1b85ffe703ae50325ff6b1_1440w.jpg)

**2）零出度算法**

需要使用DFS，借助栈来完成。其间，

- 每当有节点被标记为visited，则将其压入栈S
- 一旦发现后向边BACKWARD, 则报告非DAG并退出

DFS结束后，顺序弹出栈内节点。

下面这道题就是典型的拓扑排序：

Leetcode [207. 课程表](https://link.zhihu.com/?target=https%3A//leetcode-cn.com/problems/course-schedule/)

```text
你这个学期必须选修 numCourses 门课程，记为 0 到 numCourses - 1 。
在选修某些课程之前需要一些先修课程。 先修课程按数组 prerequisites 给出，其中 prerequisites[i] = [ai, bi] ，表示如果要学习课程 ai 则 必须 先学习课程  bi 。
例如，先修课程对 [0, 1] 表示：想要学习课程 0 ，你需要先完成课程 1 。
请你判断是否可能完成所有课程的学习？如果可以，返回 true ；否则，返回 false 。

示例 1：
输入：numCourses = 2, prerequisites = [[1,0]]
输出：true
解释：总共有 2 门课程。学习课程 1 之前，你需要完成课程 0 。这是可能的。
```



```cpp
class Solution {
public:
    stack<int> S;
    bool dfs(int course,vector<vector<int>>& graph,int*&status){
        status[course] = 1; //discovered
        for(int i = 0;i<graph[course].size();i++){
            int neighbor = graph[course][i];
            if(status[neighbor] == 0){ //undiscovered
                return dfs(neighbor,graph,status); //深入到下一个节点dfs
            }
            else if(status[neighbor]==1){ //discovered, this is a backward edge!!
                return false;
            }
            //if status[neighbor] == 2, it is a cross/forward, ignore it!
        }
        status[course] = 2; //visited!!
        S.push(course); //访问完毕，入栈
        return true;
    }
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        //首先，表示成邻接表
        vector<vector<int>> graph;
        graph.resize(numCourses);
        for(int i = 0;i<prerequisites.size();i++){ //every prerequisites
            graph[prerequisites[i][1]].push_back(prerequisites[i][0]);
        }
        //status 用来记录每个节点的访问状态。0: undiscovered, 1: discovered, 2: visited
        int *status = new int [numCourses+5];
        for(int i = 0;i<numCourses;i++) {
            status[i] = 0; 
        }
        for(int i = 0;i<numCourses;i++){
            if(status[i] == 0) { //如果状态是undiscovered，就dfs之。
                if (!dfs(i, graph, status)) //如果不是DAG，返回false
                    return false;
            }
        }
        //while(!S.empty()){  //如果题目要求输出可行的解，将栈内节点倒序输出。
        //    cout<<S.top();
        //    S.pop();
        //}
        return true;
    }
};
```

莫名其妙的击败了98.94%的用户...:)

![img](https://pic3.zhimg.com/80/v2-9489a47bb1ad06777dc4237b7ce3abf6_1440w.jpg)

------



### 最小支撑树

> “还有更荒唐的事呢，他要在普济造一条风雨长廊，把村里每一户人家都连接起来，哈哈，他以为，这样一来，普济人就可免除日晒雨淋之苦了。”

最小生成树的定义：**最小生成树**是一个连通加权无向图中一棵权值最小的生成树。假设给定无向图G一共有n个顶点，那么最小生成树一定会有 **n-1** 条边。

#### Prim算法

prim算法被用来求给定图的最小生成树。

具体算法：

1. 用两个集合A{}，B{}分别表示找到的点集，和未找到的点集；
2. 我们以A中的点为起点a，在B中找一个点为终点b，这两个点构成的边（a，b）的权值是其余边中最小的 （使用堆优化）
3. 重复上述步骤#2，直至B中的点集为空，A中的点集为满。







### Dijkstra算法

- SSSP: Single-Source Shortest Path

- - 给定顶点x，计算x到**其余**各个顶点的最短路径长度
  - Dijkstra, 1959

- APSP: All-Pairs Shortest Path

- - 找出**每对**顶点i和j之间的最短路径及长度
  - Floyd-Warshall, 1962

### 2.1 最短路径树

任一最短路径的前缀，也是一条最短路径：

![img](https://pic1.zhimg.com/80/v2-03717bf324e8f0d67ac4ed2f05fb3800_1440w.jpg)如果pi(v) 是最短路径，那么pi(u)也一定是最短路径，否则矛盾！ 

想象把s“拎起来”，dijkstra算法其实就是在模拟把s“拎起来”的场景：

![img](https://pic3.zhimg.com/80/v2-dac89e6af4c2f84460769b6c57fb2fae_1440w.jpg)

**算法**

一开始，设置每个点的dist[u] = inf，表示从起始节点s到所有点的距离都是inf。设置parent[u] = NULL；

设置工作集S = V，dist[s] = 0

while(!S.empty()){

​    选择工作集S中最小的那个节点u，更新u的所有邻居节点v的dist值：

​        if(dist[v] > dist[u] + w(u,v)){

​               dist[v] = dist[u]+w(u,v) //更新节点dist值

​                parent[u] = v; //认爸爸！

​        }

​    S.delete(u); //访问完毕

}

