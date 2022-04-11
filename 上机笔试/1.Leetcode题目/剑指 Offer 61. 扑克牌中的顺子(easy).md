#### [剑指 Offer 61. 扑克牌中的顺子](https://leetcode-cn.com/problems/bu-ke-pai-zhong-de-shun-zi-lcof/)(easy)

从**若干副扑克牌**中随机抽 `5` 张牌，判断是不是一个顺子，即这5张牌是不是连续的。2～10为数字本身，A为1，J为11，Q为12，K为13，而大、小王为 0 ，可以看成任意数字。A 不能视为 14。

**示例 2:**

```
输入: [0,0,1,2,5]
输出: True
```

数组的数取值为 [0, 13] .

解法：

我不玩牌...所以还是应该了解一下。”顺子“意思就是5个连续的数字。”0“可以看成任意的数字，可以用来填补空缺。

为了计算出总共的”gap“，我们需要先把**数组排序**。之后统计所有的gap大小，然后去判断我们有没有足够的0来填补这些空缺。

```python
class Solution:
    def isStraight(self, nums: List[int]) -> bool:
        nums = sorted(nums)
        diff_cnt = 0 ##gap
        ptr = 0
        while nums[ptr] == 0:
            ptr += 1 ##统计0的个数
        
        for i in range(ptr+1,len(nums)):
            diff = nums[i] - nums[i-1]
            if diff > 1:
                diff_cnt += diff-1
            if diff == 0:
                return False
        return ptr >= diff_cnt
```

