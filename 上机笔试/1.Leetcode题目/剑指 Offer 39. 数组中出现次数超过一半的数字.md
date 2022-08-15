#### [剑指 Offer 39. 数组中出现次数超过一半的数字](https://leetcode.cn/problems/shu-zu-zhong-chu-xian-ci-shu-chao-guo-yi-ban-de-shu-zi-lcof/)

难度简单303

数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。

 

你可以假设数组是非空的，并且给定的数组总是存在多数元素。



```python
class Solution(object):
    ans = 0  ##记录答案
    def findKthLargest(self, nums, k):
        def quicksort(nums, left, right):
            if left > right:  ##注意这里不是 >= 
                return
            pivot = get_pivot(nums, left, right)
            if pivot == len(nums) - k:  ##已经找到了答案
                self.ans = nums[pivot]
                return
            elif pivot > len(nums) - k:
                quicksort(nums, left, pivot - 1)  ##递归左区间
            else:
                quicksort(nums, pivot + 1, right)  ##递归右区间

        def get_pivot(nums, left, right):
            rand = random.randint(left, right)
            nums[rand], nums[right] = nums[right], nums[rand]
            pivot = nums[right]
            split = left - 1
            for j in range(left, right):
                if nums[j] > pivot:
                    continue
                else:
                    nums[split + 1], nums[j] = nums[j], nums[split + 1]
                    split += 1
            nums[split + 1], nums[right] = nums[right], nums[split + 1]
            return split + 1

        quicksort(nums, 0, len(nums) - 1)
        return self.ans

    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) % 2 == 1:
            k = len(nums) // 2+1
        else:
            k = len(nums) // 2
        return self.findKthLargest(nums, k)

    
```

