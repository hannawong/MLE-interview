用python创建二维数组：

`dp = [[0]*n for _ in range(m)]`

其中n为列数，m为行数

### 1. 字符串

#### 1.1 编辑距离

我们可以写出如下的状态转移方程：

若 A 和 B 的最后一个字母相同：
$$
\begin{aligned} D[i][j] &= \min(D[i][j - 1] + 1, D[i - 1][j]+1, D[i - 1][j - 1])\end{aligned}
$$
若 A 和 B 的最后一个字母不同：
$$
D[i][j] = 1 + \min(D[i][j - 1], D[i - 1][j], D[i - 1][j - 1])
$$
作者：LeetCode-Solution
链接：https://leetcode-cn.com/problems/edit-distance/solution/bian-ji-ju-chi-by-leetcode-solution/



```python
class Solution {
public:
    int minDistance(string word1, string word2) {
        int lena = word1.length();
        int lenb = word2.length();
        int ** dp = new int*[lena+5];
        for(int i = 0;i<lena+5;i++){
            dp[i] = new int [lenb+5];
        }
        for(int i = 0;i<=lena;i++){
            for(int j = 0;j<=lenb;j++){
                dp[i][j] = 0;
            }
        }
        dp[0][0] = 0;  
        for(int i = 1;i<=lena;i++)  //在两个字符串的前面都加上一个pad:'#',之后填充第一行和第一列
            dp[i][0] = i;
        for(int j = 1;j<=lenb;j++){
            dp[0][j] = j;
        }

        for(int i = 1;i<=lena;i++){
            for(int j = 1;j<=lenb;j++){
                if(word1[i-1] == word2[j-1]){ //注意要对应原先的字符位置，所以要-1.
                    dp[i][j] = min(min(dp[i-1][j-1],dp[i-1][j] + 1),dp[i][j-1]+1);
                }
                else{
                    dp[i][j] = min(min(dp[i-1][j]+1,dp[i][j-1]+1),dp[i-1][j-1]+1);
                }
            }
        }
        return dp[lena][lenb];
    }
};
```



#### 1.2 最长公共子序列

```c++
class Solution {
public:
    int longestCommonSubsequence(string text1, string text2) {
        int lena = text1.length();
        int lenb = text2.length();
        int ** dp = new int*[lena+1];
        for(int i = 0;i<=lena;i++)
            dp[i] = new int [lenb+1];
        
        for(int i = 0; i<=lena;i++){
            dp[i][0] = 1;
        }
        for(int i = 0;i<=lenb;i++){
            dp[0][i] = 1;
        }
        for(int i = 1;i<=lena;i++){
            for(int j = 1;j<=lenb;j++){
                if(text1[i-1]==text2[j-1]){
                    dp[i][j] = max(dp[i-1][j-1]+1,max(dp[i-1][j],dp[i][j-1]));
                }
                else{
                    dp[i][j] = max(dp[i-1][j],dp[i][j-1]);
                }
            }
        }
        for(int i = 0;i<=lena;i++){
            for(int j = 0;j<=lenb;j++){
                cout<<dp[i][j]<<" ";
            }
            cout<<endl;
        }
        return dp[lena][lenb]-1;
    }
};
```



#### 1.3 最长公共子串

查找两个字符串a,b中的最长公共子串。若有多个，输出在较短串中最先出现的那个。

注：子串的定义：将一个字符串删去前缀和后缀（也可以不删）形成的字符串。请和“子序列”的概念分开！

```
LCS(i,j) = LCS(i-1,j-1)+1     if str1[i] == str2[j] 
```

选取最大的那个值和位置即可知道子串。

```python
def LCS(str1,str2):
        m = len(str1)
        n = len(str2)
            
        dp = [[0 for i in range(n+1)] for j in range(m+1)]
        maxlen = 0
        end = 0
        for i in range(1,m+1):
            for j in range(1,n+1):
                if str1[i-1] == str2[j-1]:
                    dp[i][j] = 1 + dp[i-1][j-1]
                if dp[i][j] > maxlen:
                    maxlen = dp[i][j]
                    end = j
        return str2[end-maxlen:end]
```



### 2. 0/1背包问题

问题描述：有一个背包可以装物品的总重量为$W$，现有$N$个物品，每个物品中$w[i]$，价值$v[i]$，用背包装物品，能装的最大价值是多少？

**定义状态转移数组dp\[i\]\[j\]，表示前i个物品，背包重量为j的情况下能装的最大价值。**

```python
dp[i][j] = max(dp[i-1][j], dp[i-1][j-w[i]]+v[i])
```

dp\[i-1\]\[j\]表示当前物品不放入背包，dp[i-1\]\[j-w[i]]+v[i]表示当前物品放入背包，**即当前第i个物品要么放入背包，要么不放入背包**。



### 3. 最长上升子序列

每次都找前面比自己小的，最后求dp数组中最大的那个

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)

        dp = [1]*(len(nums)+1)
        for i in range(1,n):
            mmax = 1
            for j in range(0,i):
                if nums[j] < nums[i]:
                    mmax = max(mmax,dp[j]+1)
            dp[i] = mmax
        ans = 0
        for i in dp:
            ans = max(ans,i)
        return ans
```

### 4. 路径个数

##### [63. 不同路径 II](https://leetcode-cn.com/problems/unique-paths-ii/)

一个机器人位于一个 *m x n* 网格的左上角 （起始点在下图中标记为“Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。

现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？

数路径个数这种问题，可以不用BFS/DFS,用一个dp即可解决了。

```python
class Solution {
public:
    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
        int m = obstacleGrid.size();
        int n = obstacleGrid[0].size();
        int ** dp = new int*[m];
        for(int i = 0;i<m;i++){
            dp[i] = new int[n];
        }
        for(int i = 0;i<m;i++){
            for(int j = 0;j<n;j++){
                dp[i][j] = 0;
            }
        }
        bool block = false;
        for(int i = 0;i<m;i++){
            if(obstacleGrid[i][0])
                block = true;
            if(!block)
                dp[i][0] = 1;
        }
        block = false;
        for(int i = 0;i<n;i++){
            if(obstacleGrid[0][i])
                block = true;
            if(!block)
                dp[0][i] = 1;
        }
        for(int i = 1;i<m;i++){
            for(int j = 1;j<n;j++){
                if(obstacleGrid[i][j] == 1){
                    dp[i][j] = 0;
                }
                else{
                    dp[i][j] = dp[i-1][j] + dp[i][j-1];
                }
            }
        }
        return dp[m-1][n-1];
    }
};
```



### 5. 巧妙的递推公式

**HJ61 放苹果** 题目描述：把m个同样的苹果放在n个同样的盘子里，允许有的盘子空着不放，问共有多少种不同的分法？（用K表示）5，1，1和1，5，1 是同一种分法。

数据范围：![img](https://www.nowcoder.com/equation?tex=0%20%5Cle%20m%20%5Cle%2010%20%5C)，![img](https://www.nowcoder.com/equation?tex=1%20%5Cle%20n%20%5Cle%2010%20%5C)。

将问题拆分为：

```
两种情况，一种是有盘子为空，一种是每个盘子上都有苹果。
令(m,n)表示将m个苹果放入n个盘子中的摆放方法总数。
1.假设有一个盘子为空，则(m,n)问题转化为将m个苹果放在n-1个盘子上，即求得(m,n-1)即可
2.假设所有盘子都装有苹果，则每个盘子上至少有一个苹果，即最多剩下m-n个苹果，问题转化为将m-n个苹果放到n个盘子上：即求(m-n，n)
```

【考虑最后一位数组的不同情况】：打家劫舍

翻转字符

```
//考虑数组最后一位的不同情况
class Solution {
public:
    int minFlipsMonoIncr(string s) {
        int n = s.length();
        int *zero_end = new int[n];
        int *one_end = new int [n];
        if(s[0] == '0'){zero_end[0] = 0; one_end[0] = 1;}
        if(s[0] == '1'){zero_end[0] = 1;one_end[0] = 0;}
        for(int i = 1;i<n;i++){
            if(s[i] == '0'){
                zero_end[i] = zero_end[i-1];
                one_end[i] = min(one_end[i-1],zero_end[i-1])+1;
            }
            else{ //s[i] = 1
                one_end[i] = min(one_end[i-1],zero_end[i-1]);
                zero_end[i] = zero_end[i-1]+1;
            }
        }
        return min(one_end[n-1],zero_end[n-1]);
    }
};
```

