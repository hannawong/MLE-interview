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
        self.DFS(nums,0,visited)
        return self.ans_list


    def DFS(self,nums,idx,visited):
        if idx == len(nums):
            self.ans_list.append(self.ans[:])
            return
        for i in range(len(nums)):
            if visited[i]:
                continue
                
            visited[i] = 1
            self.ans.append(nums[i])
            self.DFS(nums,idx+1,visited)
            self.ans.pop() ##回溯
            visited[i] = 0
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
        self.DFS(0,0,n)
        return self.ans_list

    def DFS(self,left_num,right_num,n):
        if right_num > left_num:
            return
        if left_num == n and right_num == n:
            self.ans_list.append(''.join(self.ans))
            return
        if right_num > n or left_num > n:
            return
        ##### 左括号
        self.ans.append("(")
        self.DFS(left_num+1,right_num,n)
        self.ans.pop()
        ##### 右括号
        self.ans.append(")")
        self.DFS(left_num,right_num+1,n)
        self.ans.pop()
```

