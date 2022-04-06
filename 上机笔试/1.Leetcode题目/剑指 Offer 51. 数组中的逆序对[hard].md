# 剑指 Offer 51. 数组中的逆序对

在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组，求出这个数组中的逆序对的总数。

**题解：**

暴力的方法是O(n^2)的，用归并排序可以降为O(nlogn).

求逆序对和归并排序又有什么关系呢？关键就在于「归并」当中**「并」**的过程。例如：

假设我们有两个已排序的序列等待合并，分别是 L = {8,12,16,22,100} 和 R = {9,26,55,64,91}。一开始我们用指针 lPtr = 0 指向 L 的首部，rPtr = 0 指向 R 的头部。记已经合并好的部分为 M。

第一步：

L = [8, 12, 16, 22, 100]   R = [9, 26, 55, 64, 91]  M = []
​       |                                        |
​     lPtr                                    rPtr
我们发现 lPtr 指向的元素小于 rPtr 指向的元素，于是把 lPtr 指向的元素放入答案，并把 lPtr 后移一位。`8` 对逆序对总数的「贡献」为 0。

第二步：

L = [8, 12, 16, 22, 100]   R = [9, 26, 55, 64, 91]  M = [8]
​             |                                  |
​          lPtr                               rPtr

接着我们继续合并，把 `9` 加入了答案，此时 lPtr 指向 `12`，rPtr 指向 `26`。此时lPtr没有移动，逆序对不变。

第三步：

L = [8, 12, 16, 22, 100]   R = [9, 26, 55, 64, 91]  M = [8, 9]
​             |                                        |
​          lPtr                                     rPtr
此时 lPtr 比 rPtr 小，把 lPtr 对应的数加入答案，并考虑它对逆序对总数的贡献为 rPtr 相对 R 首位置的偏移 1（即右边只有一个数比 12 小，所以只有它和 12 构成逆序对），以此类推。

我们发现用这种「算贡献」的思想在合并的过程中计算逆序对的数量的时候，**只在 lPtr 右移的时候计算**，是基于这样的事实：当前 lPtr 指向的数字比 rPtr 小，但是比 R 中 [0 ... rPtr - 1] 的其他数字大，[0 ... rPtr - 1] 的其他数字本应当排在 lPtr 对应数字的左边，但是它排在了右边，所以**这里就贡献了 rPtr 个逆序对**。



```python
class Solution:
    def reversePairs(self, nums) -> int:
        return self.mergesort(nums,0,len(nums)-1)


    def mergesort(self,nums,begin,end): ##左闭右闭
        if begin >= end:  ##单元素区间，必然有序
            return 0
        middle = (begin+end) // 2
        cnt_left = self.mergesort(nums,begin,middle)
        cnt_right = self.mergesort(nums,middle+1,end)
        return cnt_left+cnt_right+self.merge(nums,begin,middle,end)

    def merge(self,nums,begin,middle,end): ##两件事：排序+返回逆序对个数
        ptr1 = begin
        ptr2 = middle+1
        tmp = []
        reverse_cnt = 0
        while(ptr1 <= middle and ptr2 <= end):
            if nums[ptr1] <= nums[ptr2]:
                tmp.append(nums[ptr1])
                reverse_cnt += ptr2-middle-1
                ptr1 += 1
            else:
                tmp.append(nums[ptr2])
                ptr2 += 1
        while(ptr1 <= middle):
            tmp.append(nums[ptr1])
            reverse_cnt += ptr2 - middle - 1
            ptr1 += 1
        while(ptr2 <= end):
            tmp.append(nums[ptr2])
            ptr2 += 1

        for i in range(len(tmp)):
            nums[begin+i] = tmp[i]
        return reverse_cnt
```

【技巧】要先写完mergesort，然后在此结果上修改，使之能够统计逆序对个数。

每次右移左指针的时候，贡献ptr2-middle-1个逆序对

