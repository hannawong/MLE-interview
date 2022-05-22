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
                    dp[i][j] = max(dp[i-1][j-1]+1,dp[i-1][j],dp[i][j-1])
                else:
                    dp[i][j] = max(dp[i-1][j],dp[i][j-1])
        print(dp)
        ans = ""
        i = len1; j = len2
        while i > 0 and j > 0:
            if dp[i][j] > dp[i-1][j] and dp[i][j] > dp[i][j-1]:
                ans+=text1[i-1]
                i -= 1; j-= 1
            while i >= 1 and dp[i-1][j] == dp[i][j]: ##贪婪的向上移动
                i -= 1
            while j >= 1 and dp[i][j-1] == dp[i][j]:##贪婪的向左移动
                j -= 1
        print(ans)
        return dp[len1][len2]
```

如果要求输出公共子序列，那么就从最右下角的地方开始，如果这个位置比左边、上边都要大的话，就向左上角移动一个单位；之后贪婪的向左、向上移动。



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



#### 139. 单词拆分

给你一个字符串 s 和一个字符串列表 wordDict 作为字典。请你判断是否可以利用字典中出现的单词拼接出 s 。

注意：不要求字典中出现的单词全部都使用，并且字典中的单词可以重复使用。

 1 <= s.length <= 300

```
示例 1：

输入: s = "leetcode", wordDict = ["leet", "code"]
输出: true
解释: 返回 true 因为 "leetcode" 可以由 "leet" 和 "code" 拼接成。
```

题解：

第一个想到的是DFS。但是为什么不能用呢？因为这里s的最大长度可以达到300，用递归的话明显超时。所以，就想到了用dp。dp[i]表示前i位能否由wordDict表示。

那么，dp数组怎么做初始化呢？就是遍历wordDict数组，把前面出现过的词语都标记为True.

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        n = len(s)
        dp = [False]*n
        ###初始化
        for i in range(len(wordDict)):
            word = wordDict[i]
            if s[:len(word)] == word:
                dp[len(word)-1] = True
                
        for i in range(len(s)):
            for word in wordDict:
                if i-len(word)+1>=0 and s[i-len(word)+1:i+1] == word: ##递推条件
                    dp[i] = dp[i] or dp[i-len(word)]
        return dp[-1]
```





#### [97. 交错字符串](https://leetcode-cn.com/problems/interleaving-string/)

难度中等671

给定三个字符串 `s1`、`s2`、`s3`，请你帮忙验证 `s3` 是否是由 `s1` 和 `s2` **交错** 组成的。

两个字符串 `s` 和 `t` **交错** 的定义与过程如下，其中每个字符串都会被分割成若干 **非空** 子字符串：

- `s = s1 + s2 + ... + sn`
- `t = t1 + t2 + ... + tm`
- `|n - m| <= 1`
- **交错** 是 `s1 + t1 + s2 + t2 + s3 + t3 + ...` 或者 `t1 + s1 + t2 + s2 + t3 + s3 + ...`

**注意：**`a + b` 意味着字符串 `a` 和 `b` 连接。

 ![img](https://assets.leetcode.com/uploads/2020/09/02/interleave.jpg)

**示例 1：**

```
输入：s1 = "aabcc", s2 = "dbbca", s3 = "aadbbcbcac"
输出：true
```

解法：

动态规划，`dp[i][j]`用来表示**s1的前i个元素和s2的前j个元素能否组成s3的前i+j个元素**。

为了做好初始化，把整个字符串想象成在前面加一个相同的#号。那么，第一行、第一列就分别表示s2、s1是不是和s3的开头完全匹配。其实，把初始化做好了之后，后面的递推公式就很简单了。

![img](https://pic1.zhimg.com/80/v2-ad34a0af58ef0c01cff9c56f65441995_1440w.png)

```python
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        len1 = len(s1)
        len2 = len(s2)
        len3 = len(s3)
        if len3 != len2 + len1:
            return False
        dp = [[True]*(len2+1) for _ in range(len1+1)]
        ###############初始化 ##################
        for i in range(1,len2+1):
            if s2[i-1] != s3[i-1] or not dp[0][i-1]:
                dp[0][i] = False
        for i in range(1,len1+1):
            if s1[i-1] != s3[i-1] or not dp[i-1][0]:
                dp[i][0] = False
        print(dp)
        ######################################
        for i in range(1,len1+1):
            for j in range(1,len2+1):
                ans = False
                if s1[i-1] == s3[i+j-1]:
                    ans = ans or dp[i-1][j]
                if s2[j-1] == s3[i+j-1]:
                    ans = ans or dp[i][j-1]
                dp[i][j] = ans
        return dp[len1][len2]
```





#### [115. 不同的子序列](https://leetcode.cn/problems/distinct-subsequences/)

难度困难764

给定一个字符串 `s` 和一个字符串 `t` ，计算在 `s` 的子序列中 `t` 出现的个数。

字符串的一个 **子序列** 是指，通过删除一些（也可以不删除）字符且不干扰剩余字符相对位置所组成的新字符串。（例如，`"ACE"` 是 `"ABCDE"` 的一个子序列，而 `"AEC"` 不是）

题目数据保证答案符合 32 位带符号整数范围。

**示例 1：**

```
输入：s = "rabbbit", t = "rabbit"
输出：3
解释：
如下图所示, 有 3 种可以从 s 中得到 "rabbit" 的方案。
```

题解：先在s和t之前padding上“#”，然后构成dp数组。状态转移方程也十分简单，当s[i-1] == t[j-1]时，`dp[i][j] = dp[i-1][j-1]+dp[i-1][j]`. 否则, `dp[i][j] = dp[i-1][j]`. 

```python
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        lens = len(s)
        lent = len(t)
        dp = [[0]*(lent+1) for _ in range(lens+1)]
        for i in range(lens+1):
            dp[i][0] = 1
        for i in range(1,lens+1):
            for j in range(1,lent+1):
                if s[i-1] == t[j-1]:
                    dp[i][j] = dp[i-1][j-1]+dp[i-1][j]
                else:
                    dp[i][j] = dp[i-1][j]
        return dp[lens][lent]
```

 