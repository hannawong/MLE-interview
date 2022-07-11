#### [剑指 Offer 14- I. 剪绳子](https://leetcode-cn.com/problems/jian-sheng-zi-lcof/)

难度中等418

给你一根长度为 `n` 的绳子，请把绳子剪成整数长度的 `m` 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 `k[0],k[1]...k[m-1]` 。请问 `k[0]*k[1]*...*k[m-1]` 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。

**示例 1：**

```
输入: 2
输出: 1
解释: 2 = 1 + 1, 1 × 1 = 1
```

题解：

用dp数组记录长度为i的绳子能够切成的最大乘积。

```python
class Solution:
    def cuttingRope(self, n: int) -> int:
        dp = [0]*(n+1)
        dp[1] = 1
        for i in range(2,n+1):
            for j in range(1,i):
                dp[i] = max(dp[i],dp[i-j]*dp[j],dp[i-j]*j,(i-j)*j)##这里易错！！！前面的(i-k)长度可不可以不再切分呢？可以的！
        return dp[-1]
```

