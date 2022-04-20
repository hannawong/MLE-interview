# 专题：数组中的DFS

### 基础：全排列

```python
class Solution:
    ans = []
    ans_list = []
    def permute(self, nums):
        self.ans = []
        self.ans_list = []
        visited = [0]*len(nums)
        
        def dfs(idx): ##现在在放第idx位
            if idx == len(nums):
                self.ans_list.append(self.ans[:])
                return 
            for i in range(len(nums)):
                if not visited[i]:
                    visited[i] = 1
                    self.ans.append(nums[i])
                    dfs(idx+1)
                    self.ans.pop()
                    visited[i] = 0
        dfs(0)
        return self.ans_list
```



相似题： [60. 排列序列](https://leetcode-cn.com/problems/permutation-sequence/)(hard)

给出集合 `[1,2,3,...,n]`，其所有元素共有 `n!` 种排列。

按大小顺序列出所有排列情况，并一一标记，当 `n = 3` 时, 所有排列如下：

1. `"123"`
2. `"132"`
3. `"213"`
4. `"231"`
5. `"312"`
6. `"321"`

给定 `n` 和 `k`，返回第 `k` 个排列。

**示例 1：**

```
输入：n = 3, k = 3
输出："213"
```

虽然本题有更难的数学解法，但是用暴力回溯生成全排列也可以过！但是要点是需要**剪枝**。即，当我们已经找到第k大的时候，后面的都直接返回。

```python
class Solution:
    ans = []
    cnt = 0 ##记到k
    permute = "" ##记录答案
    def getPermutation(self, n: int, k: int) -> str:

        visited = [0]*(n+1)
        def dfs(idx):
            if self.cnt > k: ##这里剪枝！！！！
                return 
            if idx == n:
                self.cnt += 1
                if self.cnt == k: ##记录答案
                    self.permute = "".join(str(_) for _ in self.ans)
                return
            for i in range(1,n+1):
                if not visited[i]:
                    visited[i] = 1
                    self.ans.append(i)
                    dfs(idx+1)
                    self.ans.pop()
                    visited[i] = 0
        dfs(0)
        return self.permute
```





#### 22. 括号生成

数字 `n` 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 **有效的** 括号组合。

```python
class Solution:
    ans = []
    ans_list = []
    def generateParenthesis(self, n: int):
        self.ans = []
        self.ans_list = []
        def DFS(left_num, right_num):
            if left_num == right_num == n:
                self.ans_list.append("".join(self.ans))
                return
            if left_num < right_num:
                return 
            if left_num > n or right_num > n:
                return 
            ####放左括号
            self.ans.append("(")
            DFS(left_num+1,right_num)
            self.ans.pop()

            ###放右括号
            self.ans.append(")")
            DFS(left_num,right_num+1)
            self.ans.pop()
        DFS(0,0)
        return self.ans_list

```

#### 39. 组合总和

给你一个 无重复元素 的整数数组 candidates 和一个目标整数 target ，找出 candidates 中可以使数字和为目标数 target 的 所有不同组合 ，并以列表形式返回。你可以按 任意顺序 返回这些组合。

candidates 中的 同一个 数字可以 无限制重复被选取 。如果至少一个数字的被选数量不同，则两种组合是不同的。 

```
输入：candidates = [2,3,6,7], target = 7
输出：[[2,2,3],[7]]
```

**解法：** 为了保证“不重复”，例如不能有[2,2,3]和[3,2,2], 需要做到“**不吃回头草**”，不能再加入前面已经遍历过的数字。所以，在每个位置，**要么选它，即now+candidate[idx]；要么不选它，即idx+1.** 

```python
class Solution:
    ans = []
    ans_list = []
    def combinationSum(self, candidates, target: int):
        self.ans = []
        self.ans_list = []
        def DFS(idx,now):
            if now == target:
                self.ans_list.append(self.ans[:])
                return
            if idx >= len(candidates):
                return
            if now > target:
                return 
            ###选
            self.ans.append(candidates[idx])
            DFS(idx,now+candidates[idx])
            self.ans.pop()

            ##不选
            DFS(idx+1,now)
        DFS(0,0)
        return self.ans_list
```

#### [40. 组合总和 II](https://leetcode-cn.com/problems/combination-sum-ii/)

给定一个候选编号的集合 `candidates` 和一个目标数 `target` ，找出 `candidates` 中所有可以使数字和为 `target` 的组合。

`candidates` 中的每个数字在每个组合中只能使用 **一次** 。

**注意：**解集不能包含重复的组合。 

 

**示例 1:**

```
输入: candidates = [10,1,2,7,6,1,5], target = 8,
输出:
[
[1,1,6],
[1,2,5],
[1,7],
[2,6]
]
```



```python
class Solution:
    ans = []
    ans_list = []
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        self.ans = []
        self.ans_list = []
        candidates = sorted(candidates)
        n = len(candidates)
        visited = [0]*n
        
        def DFS(now,idx):
            if now == target:
                self.ans_list.append(self.ans[:])
                return
            if now > target:
                return 
            for i in range(idx,len(candidates)): ###不吃回头草，只能选择idx之后的，而不能选择idx之前的元素
                if visited[i]:
                    continue

                if i>= 1 and candidates[i] == candidates[i-1] and not visited[i-1]:
                    continue
                visited[i] = 1
                self.ans.append(candidates[i])
                DFS(now+candidates[i],i) ##这里是关键，直接idx跳变到了i，说明i之前都不能再选了，“不吃回头草”
                self.ans.pop()
                visited[i] = 0

        DFS(0,0)
        return self.ans_list
```



#### 剑指 Offer 38. 字符串的排列：去重

输入一个字符串，打印出该字符串中字符的所有排列。

你可以以任意顺序返回这个字符串数组，但里面不能有重复元素。

**示例:**

```
输入：s = "abc"
输出：["abc","acb","bac","bca","cab","cba"]
```

解法：为了保证没有重复元素，我们需要把原有的字符串排序，然后保证一件事情，那就是：**一个元素前面的相同元素如果尚未放入，那么这个元素现在就不能放入！**，即 ` if i >= 1 and not visited[i-1] and s[i-1] == s[i]:continue`. 这句话就是整个代码的核心。同时注意要写i >= 1防止出界。

```python
class Solution:
    ans = []
    ans_list = []
    def permutation(self, s: str):
        self.ans = []
        self.ans_list = []
        n = len(s)
        s = sorted(s)
        visited = [0]*n
        def DFS(now):
            if now == n:
                self.ans_list.append("".join(self.ans))
                return
            for i in range(n):
                if visited[i]:
                    continue
                if i >= 1 and not visited[i-1] and s[i-1] == s[i]:
                    continue
                self.ans.append(s[i])
                visited[i] = 1
                DFS(now+1)
                visited[i] = 0
                self.ans.pop()
        DFS(0)
        return self.ans_list
```

- 总结：DFS函数作为一个辅助函数，可以写在主函数体里，这样可以免去很多参数的传入。



### 93. 复原IP地址

有效 IP 地址 正好由四个整数（每个整数位于 0 到 255 之间组成，且不能含有前导 0），整数之间用 '.' 分隔。

例如："0.1.2.201" 和 "192.168.1.1" 是 有效 IP 地址，但是 "0.011.255.245"、"192.168.1.312" 和 "192.168@1.1" 是 无效 IP 地址。
给定一个只包含数字的字符串 s ，用以表示一个 IP 地址，返回所有可能的有效 IP 地址，这些地址可以通过在 s 中插入 '.' 来形成。你 不能 重新排序或删除 s 中的任何数字。你可以按 任何 顺序返回答案。

```python
示例 1：

输入：s = "25525511135"
输出：["255.255.11.135","255.255.111.35"]
```

```
class Solution:
    ans = []
    ans_list = []

    def restoreIpAddresses(self, s: str):
        self.ans = []
        self.ans_list = []
        def is_valid(string): ##判断是否0~255之间
            if len(string) == 0:
                return False
            if string[0] == ' ':
                return False
            if int(string)<0 or int(string) > 255:
                return False
            if len(string) >= 2 and string[0] == "0":
                return False
            return True
        def DFS(s,idx,ip_num):
            if ip_num == 4 and idx == len(s):
                print(self.ans)
                self.ans_list.append(".".join(self.ans[:]))
                return
            if ip_num > 4:
                return
            if idx > len(s):
                return 

            for i in range(1,4):
                if is_valid(s[idx:idx+i]):
                    self.ans.append(s[idx:idx+i])
                    idx += i
                    DFS(s,idx,ip_num+1)
                    idx -= i
                    self.ans.pop()
        DFS(s,0,0)
        return self.ans_list
        

```

【易错点】

- 当len(nums) >= 2 且第一个数为0时，说明有前导零。注意这里的长度是2，而不是1！
- 终止条件就是  `ip_num == 4 and idx == len(s)`而不是`idx >= len(s)`

