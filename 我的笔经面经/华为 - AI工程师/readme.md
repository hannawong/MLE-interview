# 华为 - AI工程师

### 2022.1.5 华为机试：

三道题，分值100/200/300,得100就算过。

第一题：几乎裸的BFS，Leetcode 994, 100%通过

第二题：一个字符串模拟，比较简单，95%通过

第三题：

两个轮船，各有容量A和B。现在有M个货物，每个货物都有一个容量，以及放入船A的利润、放入船B的利润。

输入第一行为A,B,M

之后M行，为每个货物的容量、放入船A的利润、放入船B的利润。

示例：

```
5 6 4
4 3 2
3 1 4
2 2 4
2 3 3
```

输出：11

解释：船A放入货物0或者3，得到利润3；船B放入货物1和2，得到利润8，总共利润11.



思路：首先想到这是个背包，试图用两个dp数组来记录A的背包和B的背包，但是后来发现这样不对，因为dp_A会访问到尚未初始化的dp_B. 想了下用迭代法，但是后来还是觉得不对劲。

于是想到直接用DFS，这个是简单的。DFS(i,A_capacity,B_capacity)表示在货物i之前，在A_capacity和B_capacity的限制下，最大的利润。那么，DFS(i,A_capacity,B_capacity) = max(DFS(i-1,A_capacity-weight[i],B_capacity)+profit_A[i], DFS(i-1,A_capacity, B_capacity-weight[i])+profit_B[i], DFS(i-1,A_capacity,B_capacity))。其中，第一项为第i个物品放入船A的最大利润；第二项为第i个物品放入船B的最大利润；第三项为不放第i个物品的最大利润。

中间会有重复计算，所以用了记忆化搜索。

再后来想到，这不就是个三维dp嘛，直接用dp数组不就完事儿了。但是可惜还是超时，估计是因为三维数组的空间时间复杂度还是太大，毕竟是200\*200\*200维的。不知道有没有二维数组的方法。最后过了30%。



```python
# If you need to import additional packages or classes, please import here.
class Solution():
    dp = None
    def func(self,A_capacity,B_capacity,goods,M): ##goods:[weight,A_profit,B_profit]
        self.dp = [[[0]*(B_capacity+1) for _ in range(A_capacity+1)] for k in range(M+1)]
        for i in range(1,M+1):
            for j in range(A_capacity+1):
                for k in range(B_capacity+1):
                    if j - goods[i-1][0] >= 0:
                        self.dp[i][j][k] = max(self.dp[i][j][k],self.dp[i-1][j-goods[i-1][0]][k]+goods[i-1][1])
                    if k - goods[i - 1][0] >= 0:
                        self.dp[i][j][k] = max(self.dp[i][j][k],self.dp[i-1][j][k-goods[i-1][0]]+goods[i-1][2])
                    self.dp[i][j][k] = max(self.dp[i][j][k],self.dp[i - 1][j][k])
        print(self.dp[M][A_capacity][B_capacity])



if __name__ == "__main__":
    s = Solution()
    #s.func(5,6,[[4,3,2],[3,1,4],[2,2,4],[2,3,3]],4)
    line1 = input().split()
    A_capacity = int(line1[0])
    B_capacity = int(line1[1])
    M = int(line1[-1])
    goods = []
    for i in range(M):
        line = input().split()
        llline = []
        for j in range(len(line)):
            llline.append(int(line[j]))
        goods.append(llline)
    s.func(A_capacity,B_capacity,goods,M)
```



【复盘】：

后来想到的优化点：可以进行状态压缩，变为滚动数组。这样空间复杂度变为O(n^2), 但是时间复杂度还是O(n^3).

```python
class Solution():
    dp = None
    def func(self,A_capacity,B_capacity,goods,M): ##goods:[weight,A_profit,B_profit]
        self.dp = [[[0]*(B_capacity+1) for _ in range(A_capacity+1)] for k in range(2)]
        for i in range(1,M+1):
            for j in range(A_capacity+1):
                for k in range(B_capacity+1):
                    if j - goods[i-1][0] >= 0:
                        self.dp[1][j][k] = max(self.dp[1][j][k],self.dp[0][j-goods[i-1][0]][k]+goods[i-1][1])
                    if k - goods[i - 1][0] >= 0:
                        self.dp[1][j][k] = max(self.dp[1][j][k],self.dp[0][j][k-goods[i-1][0]]+goods[i-1][2])
                    self.dp[1][j][k] = max(self.dp[1][j][k],self.dp[0][j][k])
            for j in range(A_capacity+1):
                for k in range(B_capacity+1):
                    self.dp[0][j][k] = self.dp[1][j][k]
        print(self.dp[0][A_capacity][B_capacity])
```



## 2022.1.10 一面二面

#### 一面：

自我介绍

问了最近实验室的项目

python底层：

- tuple的特性（反正和list也差不多）
- 字典按照value值排序（答出用lambda 表达式即可）
- python的迭代器和生成器有什么区别（生成器用yield返回迭代器）

pytorch底层：

- 把维度为(3,4,5)的tensor变为(12,5)的tensor怎么做？（可以用reshape和view）
- reshape和view有什么区别？（view只支持连续内存空间，这个没有答出来，不过面试官说没关系...）

手撕代码：

leetcode 1020, DFS即可。



#### 二面：

自我介绍

你遇到的最有困难的项目，怎么解决的

问我英语怎么样，平时跟实验室的老外交流怎么样...?? 我说我英语不咋地，面试官说看你托福GRE成绩还挺高，我说那都是应试的...

然后就做了个题 Leetcode 131 分割回文串，先用dp记录每个位置之间是否回文，然后回溯。



结果：

一面二面通过，等待最后一轮主管面。



复盘：

因为时差的原因，早上五点起来面试，脑子糊里糊涂的，写代码的时候比较紧张，没有发挥出平时的水平，不过好在是做出来了，hr说面试官评价也不错。还是以后要多多刷题，以及保持良好心态，发挥出平时的正常水平。

以及华为的面试非常重视底层，甚至都没有涉及到机器学习的知识，这个还是挺奇怪的。不过作为科班出身的学生，底层的基础知识是一定一定要掌握的，因为算法工程师首先是工程师，要戒骄戒躁。





----

2022.1.13

主管面通过啦！