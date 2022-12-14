[TOC]

# Top-K问题

#### [215. 数组中的第K个最大元素](https://leetcode.cn/problems/kth-largest-element-in-an-array/)

给定整数数组 nums 和整数 k，请返回数组中第 k 个**最大的元素**。

请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。

```python
示例 1:

输入: [3,2,1,5,6,4] 和 k = 2
输出: 5
```

##### 【方法1：小根堆 】

**求topk个最大元素，用小根堆；求topk个最小元素，用大根堆。**

对于前k个元素，直接插入堆即可，这样形成了一个**小根堆**。对于后面的n-k个元素，如果比小根堆的根还小，那么肯定不用管，因为它一定不是第k大的数；反之，如果比小根堆的根大，那么弹出根，插入当前值。最终，小根堆的**根**就是这个第k大的元素了。

![img](https://pic3.zhimg.com/80/v2-b1c67ab85c55c1bc7d189a2e6b323a8c_1440w.png)

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        heap = []
        for i in range(len(nums)):
            if len(heap) < k: ##如果堆尚未满，可以随意插入
                heapq.heappush(heap,nums[i])
            else: ##堆已满
                if nums[i] > heap[0]: ###言外之意：如果现在这个元素比堆顶还小，那么一定不是topk
                    heapq.heappop(heap)
                    heapq.heappush(heap,nums[i])
        return heap[0]
```

时间复杂度 O(nlogn). 因为堆的插入是O(logn), 需插入n次。堆的大小为k，所以空间复杂度为O(k).

##### 【方法2：利用快排】

**快排的原始写法：**

```python
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        
        def partitionsort(nums,low,high): ##排序[low,high]之间的数组
            if low > high: ##单元素区间，必定有序
                return
            partition = find_partition(nums,low,high) ##寻找轴点的位置
            partitionsort(nums,low,partition-1) ##原地排序左边
            partitionsort(nums,partition+1,high) ##原地排序右边
            
        def find_partition(nums,low,high): ##构造轴点
            rand = random.randint(low,high) ##随机选一个轴点
            nums[rand],nums[high] = nums[high],nums[rand] ##和末尾交换，末尾作为轴点
            pivot = nums[high]
            split = low-1
            for j in range(low,high):
                if nums[j] < pivot: 
                    nums[j],nums[split+1] = nums[split+1],nums[j]
                    split+=1
            nums[split+1],nums[high] = nums[high],nums[split+1]
            return split+1
        partitionsort(nums,0,len(nums)-1)
        return nums
```

具体的方法：

1. 首先，选择最后一个元素 -- `pivot = A[right]`作为我们要培养的轴点。
2. 一个指针 `j` 从 `low` 遍历到 `high-1`；另外维护一个split，使得[low,split]的值都<轴点，[split+1,high]的值都>轴点

![img](https://pic1.zhimg.com/80/v2-92472f2ca7ae0c67b3ec74db9fda4027_1440w.png)

在考察 j 时，

- 如果arr[j]>=pivot, 不用特殊处理，直接考虑下一个j+1即可；
- 如果arr[j] < pivot, 则需要把它移到 "<= pivot"那一类中，那么需要交换arr[split+1]和arr[j],并且 split++。

最终，交换arr[split+1]和arr[right], 返回split+1，此点即为轴点

【注】如果不用随机选轴点，最坏情况（**数组已经接近有序**）时，复杂度O(n^2)，因为只有越接近“平均二分”，才能达到O(nlogn)复杂度。但是平均情况还是O(nlogn)。 为了能够达到这个“期望”复杂度，我们需要**随机选择轴点**才行。

【总结】

- 快排是不稳定的排序方法



----



**划分：**  将数组 `a[l⋯r]` 划分成两个子数组 `a[l⋯q−1]`、`a[q+1⋯r]`，使得 `a[l⋯q−1] `中的每个元素小于等于 a[q]，且 a[q]小于等于 `a[q+1⋯r] 中的每个元素`。这个a[q]就称为“**轴点**”。
**递归**： 通过递归调用快速排序，对子数组`a[l⋯q−1]` 和`a[q+1⋯r]` 进行**排序**。

由此可以发现**每次经过划分操作后，我们一定可以确定一个元素的最终位置**！如果位置q是在第 **n-k** 位，那么说明我们已经找到了这个topk元素！

**如果 q 比目标下标（n-k）小，就递归右子区间，否则递归左子区间。**这样就可以把原来递归两个区间变成只递归一个区间，提高了时间效率。

我们只需要在快排的基础上添加一些代码：

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        self.ans = 0 ##用一个全局变量来记录
        def quicksort(nums,left,right):
            if left > right:
                return 
            pivot = partition(nums,left,right)
            if pivot == len(nums)-k: ###找到！
                self.ans = nums[pivot]
                return 
            elif pivot > len(nums)-k: ##轴点偏右了，向左找
                quicksort(nums,left,pivot-1)
            else:
                quicksort(nums,pivot+1,right)
        def partition(nums,left,right):
            split = left-1
            pivot = nums[right]
            for i in range(left,right):
                if nums[i] < pivot:
                    nums[i],nums[split+1] = nums[split+1], nums[i]
                    split += 1
            nums[right], nums[split+1] = nums[split+1], nums[right]
            return split+1
        quicksort(nums,0,len(nums)-1)
        return self.ans
```

`find_partition`函数最后的结果是，arr的左边都<轴点，arr的右边都>轴点，轴点的位置是`split+1`

【易错点】

- 可以看到，这里是用一个变量self.ans来记录答案，只要有一次符合条件，self.ans便会记录下来答案。
- 返回条件是low > high**, 而不是low >= high**, 因为如果是low >= high的话，会导致单元素区间不能记录self.ans. 

【复杂度分析】T(n) = T(n/2) + n, 这不是一个完全的递归树，按照每次排序刚好定位到中间来想，递归树的下一层只有T(n/2)，不考虑另一半。根据master theorem，复杂度为**O(n)**. 当然，最坏情况是O(n^2). 

空间复杂度为递归栈的大小，平均情况为O(logn), 最坏情况为O(n). 不过，快排的这个递归是尾递归，可以比较容易的改写成迭代形式。如果编译器有尾递归优化，空间复杂度为O(1).



[master theorem] ![img](https://pic1.zhimg.com/80/v2-ca7830ee5d665aa8d4437322830bac60_1440w.jpg)









#### [剑指 Offer 40. 最小的k个数](https://leetcode.cn/problems/zui-xiao-de-kge-shu-lcof/)

难度简单431

设计一个算法，找出数组中最小的k个数。以任意顺序返回这k个数均可。

```
示例：
输入： arr = [1,3,5,7,2,4,6,8], k = 4
输出： [1,2,3,4]
```

题解：在寻找轴点并排序的过程中，如果我们发现轴点的位置就是k-1,那么就可以直接返回arr[:k], 因为轴点前面的数一定都<轴点！！

如果发现轴点位置 > k-1,那么继续搜索轴点之前；否则，搜索轴点之后。

由于只需要递归一次，所以时间复杂度是O(n)的。这应该是最优的时间复杂度。

当数字的范围确定时，也可以用直方图计数的方法。

```python
class Solution:
    ans = []
    def getLeastNumbers(self, nums: List[int], k: int) -> List[int]:
        def partitionsort(nums,low,high): ##排序[low,high]之间的数组
            if low > high: ##单元素区间，必定有序
                return
            partition = find_partition(nums,low,high) ##寻找轴点的位置
            if partition == k-1:
                self.ans = nums[:k] ##已经找到对应位置，直接返回
                return 
            if partition > k-1:
                partitionsort(nums,low,partition-1) ##原地排序左边
            if partition < k-1:
                partitionsort(nums,partition+1,high) ##原地排序右边
            
        def find_partition(nums,low,high): ##构造轴点
            rand = random.randint(low,high) ##随机选一个轴点
            nums[rand],nums[high] = nums[high],nums[rand] ##和末尾交换，末尾作为轴点
            pivot = nums[high]
            split = low-1
            for j in range(low,high):
                if nums[j] < pivot: 
                    nums[j],nums[split+1] = nums[split+1],nums[j]
                    split+=1
            nums[split+1],nums[high] = nums[high],nums[split+1]
            return split+1
        partitionsort(nums,0,len(nums)-1)
        return self.ans
```



#### [692. 前K个高频单词](https://leetcode.cn/problems/top-k-frequent-words/)

难度中等456

给定一个单词列表 `words` 和一个整数 `k` ，返回前 `k` 个出现次数最多的单词。

返回的答案应该按单词出现频率由高到低排序。如果不同的单词有相同出现频率， **按字典顺序** 排序。

 

**示例 1：**

```
输入: words = ["i", "love", "leetcode", "i", "love", "coding"], k = 2
输出: ["i", "love"]
解析: "i" 和 "love" 为出现次数最多的两个单词，均为2次。
    注意，按字母顺序 "i" 在 "love" 之前。
```

```Python
class Solution:
    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        dic = {}
        for word in words:
            if word in dic:
                dic[word] += 1
            else:
                dic[word] = 1
        heap = []
        for key in dic:
            heapq.heappush(heap,[-dic[key],key])
        ans = []
        for i in range(k):
            ans.append(heapq.heappop(heap)[1])
        return ans
   
```

【总结】

- 按照频率降序排序；同等频率按照字典序升序排列：`heapq.heappush(heap,[-dic[key],key])`.