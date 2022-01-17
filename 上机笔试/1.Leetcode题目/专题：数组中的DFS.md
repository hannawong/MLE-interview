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

