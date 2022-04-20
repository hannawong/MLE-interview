#### [剑指 Offer 57 - II. 和为s的连续正数序列](https://leetcode-cn.com/problems/he-wei-sde-lian-xu-zheng-shu-xu-lie-lcof/)

难度简单419

输入一个正整数 `target` ，输出所有和为 `target` 的连续正整数序列（至少含有两个数）。

序列内的数字由小到大排列，不同序列按照首个数字从小到大排列。

 

**示例 1：**

```
输入：target = 9
输出：[[2,3,4],[4,5]]
```

**示例 2：**

```
输入：target = 15
输出：[[1,2,3,4,5],[4,5,6],[7,8]]
```



滑动窗口，范围是[left,right)

```python
class Solution:
    def findContinuousSequence(self, target: int):
        left = 1
        right = 1
        cnt = 0
        ans = []
        ans_list = []
        while left <= target // 2+2 and right <= target // 2+2:
            if cnt == target:
                ans_list.append(ans[:])
            cnt += right ##无脑移动右窗口
            ans.append(right)
            right += 1
            while cnt > target: ##直到cnt的值>target
                cnt -= left ##移动左窗口
                left += 1
                ans  = ans[1:]

        return ans_list
```

