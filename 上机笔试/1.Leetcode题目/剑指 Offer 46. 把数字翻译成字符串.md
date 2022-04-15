#### [剑指 Offer 46. 把数字翻译成字符串](https://leetcode-cn.com/problems/ba-shu-zi-fan-yi-cheng-zi-fu-chuan-lcof/)

难度中等415

给定一个数字，我们按照如下规则把它翻译为字符串：0 翻译成 “a” ，1 翻译成 “b”，……，11 翻译成 “l”，……，25 翻译成 “z”。一个数字可能有多个翻译。请编程实现一个函数，用来计算一个数字有多少种不同的翻译方法。

 

**示例 1:**

```
输入: 12258
输出: 5
解释: 12258有5种不同的翻译，分别是"bccfi", "bwfi", "bczi", "mcfi"和"mzi"
```



代码：

```python
class Solution:
    def translateNum(self, num: int) -> int:
        num = str(num)
        n = len(num)
        def isvalid(string): ##string 是否是0~25之间
            if int(string) > 25 or int(string) < 0 :
                return False
            if len(string) > 1 and string[0] == "0":
                return False
            return True
        def dfs(idx): ##string num 的idx位之前有多少种可能
            if idx == 0:
                return 1
            if idx == 1:
                if isvalid(num[idx-1:idx+1]):
                    return 2
                else:
                    return 1
            ans = 0
            if idx >= 1 and isvalid(num[idx-1:idx+1]):
                ans += dfs(idx-2)
            ans += dfs(idx-1)
            return ans
        return dfs(n-1)
            
```

