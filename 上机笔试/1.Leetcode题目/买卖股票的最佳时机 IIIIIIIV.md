# 买卖股票的最佳时机 I/II/III/IV

#### 122. 买卖股票的最佳时机 II

给定一个数组 prices ，其中 prices[i] 表示股票第 i 天的价格。

在每一天，你可能会决定购买和/或出售股票。你在任何时候 **最多只能持有一股股票**。你也可以购买它，然后在同一天出售。
返回你能获得的最大利润 。

```python
输入: prices = [7,1,5,3,6,4]
输出: 7
```

**解法：**

因为题中说“最多只能持有一股股票”，所以不妨以每天持有股票的个数来分类。`dp[i][0]`表示第i天，不持有股票的收益；`dp[i][1]`表示第i天，持有股票的收益。想要持有股票，就必须花钱来买它，所以收益是要减去当前股票的价格的。在卖掉的时候，收益要加上当前股票的价格，因为卖出了就可以得到收益。所以，可以得到如下的状态转移函数：

```python
dp[0][i] = max(dp[0][i-1], dp[1][i-1] + prices[i]) ##前一天持有，后一天卖掉
dp[1][i] = max(dp[1][i-1], dp[0][i-1] - prices[i])##前一天没有，后一天买入
```

初始状态：

`dp[0][0] = 0`, `dp[0][1] = -price[0]` 

最后一天一定是不持有股票的收益最大，故返回`dp[0][n-1]`

```python
class Solution:
    def maxProfit(self, prices) -> int:
        n = len(prices)
        dp = [[0]*n for _ in range(2)]
        dp[0][0] = 0
        dp[1][0] = -prices[0]
        for i in range(1,n):
            dp[0][i] = max(dp[0][i-1], dp[1][i-1] + prices[i]) ##前一天持有，后一天卖掉
            dp[1][i] = max(dp[1][i-1], dp[0][i-1] - prices[i])##前一天没有，后一天买入
        print(dp)
        return dp[0][n-1]
```



### 714. 买卖股票的最佳时机含手续费

给定一个整数数组 prices，其中第 i 个元素代表了第 i 天的股票价格 ；整数 fee 代表了交易股票的手续费用。

你可以无限次地完成交易，但是你每笔交易都需要付手续费。如果你已经购买了一个股票，在卖出它之前你就不能再继续购买股票了。

返回获得利润的最大值。

注意：这里的一笔交易指买入持有并卖出股票的整个过程，**每笔交易你只需要为支付一次手续费**。

解法：

还是和上一题一样，有两个状态：持有1个股票、不持有股票。区别就是，在卖出的时候需要花掉手续费。

```python
class Solution:
    def maxProfit(self, prices, fee: int) -> int:
        n = len(prices)
        dp = [[0]*n for _ in range(2)]
        dp[1][0] = -prices[0]
        ans = 0
        for i in range(1,n):
            dp[0][i] = max(dp[0][i-1],dp[1][i-1]-fee+prices[i])
            dp[1][i] = max(dp[1][i-1],dp[0][i-1]-prices[i])
            ans = max(ans,dp[0][i])
        return ans
```



### 309. 最佳买卖股票时机含冷冻期

给定一个整数数组prices，其中第  prices[i] 表示第 i 天的股票价格 。

设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:

**卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。**

注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

解法：

其实，这是一个**有限自动机**。总共无非有三个状态：

- 状态0：持有一股，（必非冷冻期）
- 状态1：持有0股，且不在冷冻期
- 状态2：持有0股，且在冷冻期（现在是卖掉）



```python
class Solution:
    def maxProfit(self, prices) -> int:
        n = len(prices)
        dp = [[0]*n for _ in range(3)]
        dp[0][0] = -prices[0] 
        ans = 0
        for i in range(1,n):
            dp[0][i] = max(dp[1][i-1]-prices[i],dp[0][i-1]) ##原来就有 or 非冷冻期买入
            dp[1][i] = max(dp[1][i-1],dp[2][i-1]) #原来就无 or 冷冻期过去
            dp[2][i] = dp[0][i-1]+prices[i] ##原来持有一股，现在卖掉
            ans = max(max(dp[0][i],dp[1][i]),dp[2][i])
        return ans
```



### 123. 买卖股票的最佳时机 III [hard]

给定一个数组，它的第 `i` 个元素是一支给定的股票在第 `i` 天的价格。

设计一个算法来计算你所能获取的最大利润。你最多可以完成 **两笔** 交易。

**注意：**你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

**解法：**

题中的一个重要限制：“最多可以完成两笔交易”。受到上面题目的启发，想到用有限自动机来表示不同的状态，并画出状态转移图：

![img](https://pic1.zhimg.com/80/v2-3abd42ba6c29475ddd9ef060500b72ce_1440w.png)



一个非常非常容易错的地方是，状态2，4的初始化`dp[2][i],dp[4][i]`应该被初始化为`-price[0]`, 表示现在买入了第0支股票，之前还做过一次买入、卖出。

```python
class Solution:
    def maxProfit(self, prices) -> int:
        n = len(prices)
        dp = [[0]*n for _ in range(5)] ##这里的状态是从0开始标的，和上图不一样
        dp[1][0] = -prices[0]
        dp[3][0] = -prices[0] ##非常容易忽略！
        ans = 0
        for i in range(1,n):
            dp[0][i] = dp[0][i-1]
            dp[1][i] = max(dp[1][i-1],dp[0][i-1]-prices[i])
            dp[2][i] = max(dp[2][i-1],dp[1][i-1]+prices[i])
            dp[3][i] = max(dp[3][i-1],dp[2][i-1]-prices[i])
            dp[4][i] = max(dp[4][i-1],dp[3][i-1]+prices[i])
            ans = max(max(dp[0][i],dp[2][i]),dp[4][i])
        return ans
```



### 188. 买卖股票的最佳时机 IV [hard]

你最多可以完成 **k** 笔交易。

把上面的代码从k = 2改为支持任意k值即可。

