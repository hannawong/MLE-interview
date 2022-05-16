#### [剑指 Offer 66. 构建乘积数组](https://leetcode.cn/problems/gou-jian-cheng-ji-shu-zu-lcof/)

难度中等228

给定一个数组 `A[0,1,…,n-1]`，请构建一个数组 `B[0,1,…,n-1]`，其中 `B[i]` 的值是数组 `A` 中除了下标 `i` 以外的元素的积, 即 `B[i]=A[0]×A[1]×…×A[i-1]×A[i+1]×…×A[n-1]`。不能使用除法。

 

**示例:**

```
输入: [1,2,3,4,5]
输出: [120,60,40,30,24]
```



题解：

从左到右记录”前缀积“，从右到左记录”后缀积“

```python
class Solution:
    def constructArr(self, a: List[int]) -> List[int]:
        prefix = [1]
        mul = 1
        for i in range(len(a)):
            mul *= a[i]
            prefix.append(mul)
        print(prefix)
        suffix = [1]
        mul = 1
        for i in range(len(a)-1,-1,-1):
            mul *= a[i]
            suffix.append(mul)
        suffix = suffix[::-1]
        print(suffix)

        ans = []
        for i in range(len(a)):
            ans.append(prefix[i]*suffix[i+1])
        return ans
```

