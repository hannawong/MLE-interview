# 专题 - 字符串中的dp

#### 1. 编辑距离问题

可以插入、删除、替换。

那么，dp方程就是：

![img](https://pic1.zhimg.com/80/v2-6fa0a006897663349f9b27ae3551f3db_1440w.jpeg)

```c++
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        lena = len(word1)
        lenb = len(word2)
        dp = [[0]*(lenb+1) for _ in range(lena+1)]
        for i in range(lenb+1):
            dp[0][i] = i
        for j in range(lena+1):
            dp[j][0] = j
        for i in range(1,lena+1):
            for j in range(1,lenb+1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = min(min(dp[i-1][j]+1,dp[i][j-1]+1),dp[i-1][j-1])
                else:
                    dp[i][j] = min(min(dp[i-1][j]+1,dp[i][j-1]+1),dp[i-1][j-1]+1)
        return dp[lena][lenb]
```

【易错点】有三个：

- 当word1[i]!=word2[j]的时候，不要忘记还可能会有`dp[i-1][j-1]+1`, 这对应“**替换**”操作。
- 最后应返回`dp[lena][lenb]`而不是`dp[lena-1][lenb-1]`
- 第0行和第0列都用**相同**的字符“#“来pad。为什么要强调相同的呢，那是因为如果pad的值不同，会导致最后的编辑距离发生变化！因此，一开始dp矩阵的初始化应该是这样的：

![img](https://pic1.zhimg.com/80/v2-7fcbe552cf029ce5ec3ee47ac4e8dcd9_1440w.png)

#### 2. 最长公共子序列

和编辑距离问题不同的是，这个前面需要pad两个**不同**的字符。因为如果pad两个相同的字符会导致公共子序列长度发生变化！

![image.png](https://pic.leetcode-cn.com/1617411822-KhEKGw-image.png)

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        len1 = len(text1)
        len2 = len(text2)
        dp = [[0]*(len2+1) for _ in range(len1+1)]
        for i in range(1,len1+1):
            for j in range(1,len2+1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = max(max(dp[i-1][j],dp[i][j-1]),dp[i-1][j-1]+1)
                else:
                    dp[i][j] = max(dp[i-1][j],dp[i][j-1])
        return dp[len1][len2]
```



注意：注意题中子序列的定义是否可以是连续的，如果必须是连续的:

##### 718. 最长重复子数组

```python
class Solution:
    def findLength(self, nums1, nums2):
        len1 = len(nums1)
        len2 = len(nums2)
        ans = 0
        dp = [[0]*(len2+1) for _ in range(len1+1)]
        for i in range(1,len1+1):
            for j in range(1,len2+1):
                if nums1[i-1] == nums2[j-1]:
                    dp[i][j] = dp[i-1][j-1]+1
                    ans = max(ans,dp[i][j])
                else:
                    dp[i][j] = 0
        return ans
```

![image.png](https://pic.leetcode-cn.com/9b80364c7936ad0fdca0e9405025b2a207a10322e16872a6cb68eb163dee25ee-image.png)

使用滚动数组压缩：

```python
class Solution:
    def findLength(self, nums1, nums2):
        len1 = len(nums1)
        len2 = len(nums2)
        ans = 0
        dp = [[0]*(len2+1) for _ in range(2)]  ##只用两行即可

        for i in range(1,len1+1): ##时间复杂度还是不变
            for j in range(1,len2+1):
                if nums1[i-1] == nums2[j-1]:
                    dp[1][j] = dp[0][j-1]+1
                    ans = max(ans,dp[1][j])
                else:
                    dp[1][j] = 0
            for j in range(1,len2+1):  ##滚动
                dp[0][j] = dp[1][j]
        return ans
```

