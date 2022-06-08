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

虽然本题有更难的数学解法，但是用暴力回溯生成全排列也可以过！但是要点是需要**剪枝**。即，**当我们已经找到第k大的时候，后面的都直接返回。**

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



#### 17 电话号码的字母组合

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        self.ans = []
        self.ans_list = []
        if len(digits) == 0:
            return []
        dic = {'2':"abc",'3':"def",'4':"ghi",'5':"jkl",'6':"mno",'7':"pqrs",'8':"tuv",'9':"wxyz"}
        def dfs(idx): ##到了第idx位
            if idx >= len(digits):
                self.ans_list.append("".join(self.ans))
                return 
            for i in range(len(dic[digits[idx]])): ##随意选一个
                self.ans.append(dic[digits[idx]][i]) 
                dfs(idx+1) ##后面接着选！
                self.ans.pop()
        dfs(0)
        return self.ans_list

```



#### 22. 括号生成

数字 `n` 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 **有效的** 括号组合。

```python
class Solution:
    ans = ""
    ans_list = []
    def generateParenthesis(self, n: int) -> List[str]:
        self.ans = ""
        self.ans_list = []
        def dfs(left,right):
            if left ==right == n:
                print(self.ans)
                self.ans_list.append(self.ans)
                return
            if left > n or right > n:
                return 
            if right > left:
                return 
            self.ans+="("
            dfs(left+1,right)
            self.ans = self.ans[:-1]

            self.ans+=")"
            dfs(left,right+1)
            self.ans = self.ans[:-1]
        dfs(0,0)
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

 禁止重复的关键在于：**if not(idx >= 1 and candidates[idx] == candidates[idx-1] and not visited[idx-1]):**

【千万不要忘记之前sorted】

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
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        self.ans = []
        self.ans_list = []
        candidates = sorted(candidates) ##排序
        visited = [0]*len(candidates)
        def DFS(idx,now,visited):
            if now == target:
                self.ans_list.append(self.ans[:])
                return
            if idx >= len(candidates):
                return
            if now > target:
                return 
            ###选
            if not(idx >= 1 and candidates[idx] == candidates[idx-1] and not visited[idx-1]):  ##禁止重复！！
                
                self.ans.append(candidates[idx])
                visited[idx] = 1
                DFS(idx+1,now+candidates[idx],visited)
                visited[idx] = 0
                self.ans.pop()

            ##不选
            DFS(idx+1,now,visited)
        DFS(0,0,visited)
        return self.ans_list
```





#### [77. 组合](https://leetcode.cn/problems/combinations/)

难度中等992

给定两个整数 `n` 和 `k`，返回范围 `[1, n]` 中所有可能的 `k` 个数的组合。

你可以按 **任何顺序** 返回答案。

 

**示例 1：**

```
输入：n = 4, k = 2
输出：
[
  [2,4],
  [3,4],
  [2,3],
  [1,2],
  [1,3],
  [1,4],
]
```

```python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        self.ans = []
        self.ans_list = []
        def dfs(kk,idx): ##有了几个数，idx到了第几位
            if idx > n and kk == k:
                self.ans_list.append(self.ans[:])
                return 
            if kk > k:
                return
            if idx > n:
                return 
            ###选
            self.ans.append(idx)
            dfs(kk+1,idx+1)
            self.ans.pop()
            ###不选
            dfs(kk,idx+1)
        dfs(0,1)
        return self.ans_list
```

【总结】还是对于每个位置，判断选or不选



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

```python
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        def is_valid(str):
            if len(str) == 0:
                return False
            if int(str) < 0 or int(str) > 255:
                return False
            if len(str) >= 2 and str[0] == "0":
                return False
            return True
        self.ans = []
        self.ans_list = []
        def dfs(s,idx,n):
            if n > 4:
                return 
            if idx == len(s) and n == 4:
                self.ans_list.append(".".join(self.ans[:]))
                return
            for i in range(1,4):
                if is_valid(s[idx:idx+i]):
                    self.ans.append(s[idx:idx+i])
                    dfs(s,idx+i,n+1)
                    self.ans.pop()
        dfs(s,0,0)
        return self.ans_list
        

```

【易错点】

- 当len(nums) >= 2 且第一个数为0时，说明有前导零。注意这里的长度是2，而不是1！
- 终止条件就是  `ip_num == 4 and idx == len(s)`而不是`idx >= len(s)`





#### [140. 单词拆分 II](https://leetcode.cn/problems/word-break-ii/)

难度困难596

给定一个字符串 `s` 和一个字符串字典 `wordDict` ，在字符串 `s` 中增加空格来构建一个句子，使得句子中所有的单词都在词典中。**以任意顺序** 返回所有这些可能的句子。

**注意：**词典中的同一个单词可能在分段中被重复使用多次。

**示例 1：**

```
输入:s = "catsanddog", wordDict = ["cat","cats","and","sand","dog"]
输出:["cats and dog","cat sand dog"]
```

题解：简单的递归

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        self.ans = []
        self.ans_list = []
        def dfs(idx): 
            if idx >= len(s):
                self.ans_list.append(" ".join(self.ans[:]))
                return 
            for word in wordDict:
                if s[idx:idx+len(word)] == word:
                    self.ans.append(word)
                    dfs(idx+len(word))
                    self.ans.pop()
        dfs(0)
        return self.ans_list
```





#### [698. 划分为k个相等的子集](https://leetcode.cn/problems/partition-to-k-equal-sum-subsets/)

难度中等563

给定一个整数数组  `nums` 和一个正整数 `k`，找出是否有可能把这个数组分成 `k` 个非空子集，其总和都相等。

 

**示例 1：**

```
输入： nums = [4, 3, 2, 3, 5, 2, 1], k = 4
输出： True
说明： 有可能将其分成 4 个子集（5），（1,4），（2,3），（2,3）等于总和。
```

题解：我们先找到所有和为target的子数组，然后记下其索引；第二步就是找有没有几个不重合的索引加起来是全集。每一步都使用上面说的方法：判断加/不加

```
class Solution:
    def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
        sums = sum(nums)
        if sums % k != 0:
            return False
        target = sums // k
        
        self.ans = []
        self.ans_list = []
        def find_target(nums,target,now,idx):
            if now == target: ##【易错】这里要放在最前面
                self.ans_list.append(self.ans[:])
                return 
            if now > target:
                return 
            if idx >= len(nums):
                return 
        
            ##选
            self.ans.append(idx)
            find_target(nums,target,now+nums[idx],idx+1)
            self.ans.pop()
            ##不选
            find_target(nums,target,now,idx+1)
        find_target(nums,target,0,0) ##self.ans_list中存放的是所有idx

        self.visited = [0]*len(nums) ###每个都没有覆盖到

        def find_range(idx,visited_cnt): ##试图找到能够覆盖的
            if visited_cnt == len(nums): ##已经找到了覆盖全集的方法。【易错】这个要放在最前面
                return True
            if idx >= len(self.ans_list): ##【易错】这里别犯糊涂写成len(nums)了！
                return False
            ##检查看看能不能放
            ok = True
            ans = False
            for num in self.ans_list[idx]:
                if self.visited[num] == 1:
                    ok = False
                    break
            if ok: ##没有overlap
                tmp_idx = idx
                for i in range(len(self.ans_list[idx])):
                    self.visited[self.ans_list[idx][i]] = 1
                ans = ans or find_range(idx+1,visited_cnt+len(self.ans_list[idx]))

                for i in range(len(self.ans_list[tmp_idx])): ##回溯
                    self.visited[self.ans_list[idx][i]] = 0
            ##不选
            ans = ans or find_range(idx+1,visited_cnt)
            return ans
        return find_range(0,0)
            
```

