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
        for i in range(k):  ##前k个，可以直接push到堆中
            heapq.heappush(heap,nums[i])
        #### python中堆是小根堆
        for i in range(k,len(nums)): ###对于后面的元素
            if nums[i] > heap[0]: ###只有当元素比小根堆的根大时
                heapq.heappop(heap)
                heapq.heappush(heap,nums[i])
        return heap[0]
```

时间复杂度 O(nlogk). 因为堆的插入是O(logk), 需插入n次。堆的大小为k，所以空间复杂度为O(k).

##### 【方法2：利用快排】
**快排的原始写法：**
```python
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        def quicksort(nums,begin,end): ##排序[begin,end]之间的数组
            if begin >= end: ###单元素或无元素区间，必定有序.【易错】千万不要写成begin<=end了！！
                return
            pivot_pos = find_pivot(nums,begin,end) ##重点在于选择轴点，并将数组组织成：
            ###左边的元素都小于轴点，右边的元素都大于轴点
            quicksort(nums,begin,pivot_pos-1) ##排序轴点之前的（原地）
            quicksort(nums,pivot_pos+1,end) ##排序轴点之后的（原地）
        
        def find_pivot(nums,begin,end):
            rand_pos = random.randint(begin,end)  ##随机选一个轴点
            nums[end],nums[rand_pos] = nums[rand_pos], nums[end] ##并和最后一个交换
            pivot_value = nums[end] ##轴点的值
            split = begin
            for i in range(begin,end): ###遍历所有（除了最后一个）的位置
                if nums[i] <= pivot_value: ###【易错】只有当当前位置小于轴点时，才需要和轴点交换，并将轴点+1
                    nums[split],nums[i] = nums[i],nums[split]
                    split += 1
                #### 否则，什么都不做
            ###遍历之后
            nums[end], nums[split] = nums[split], nums[end]
            return split

        quicksort(nums,0,len(nums)-1)
        return nums
```

培养一个轴点：

![img](https://pic1.zhimg.com/80/v2-2b70124e17ea9fea9eda1e9bfb745a58_1440w.jpg)

具体的方法：

1. 首先，选择最后一个元素 -- `pivot = A[begin]`作为我们要培养的轴点。
2. 一个指针 `j` 从 `begin` 遍历到 `end-1`；另外维护一个split，使得**split及**左边的值都<轴点，split右边的值都>轴点

![img](https://pic1.zhimg.com/80/v2-92472f2ca7ae0c67b3ec74db9fda4027_1440w.png)

在考察 j 时，

- 如果arr[j]>=pivot, 不用特殊处理，直接考虑下一个j+1即可；

- 如果arr[j] < pivot, 则需要把它移到 "<= pivot"那一类中，那么需要交换arr[split]和arr[j],并且 split++。

3. 最终，交换arr[split]和arr[right], 返回split，此点即为轴点



【复杂度分析】
由于选取轴点的复杂度为O(n), 则有公式$T(n) = 2T(n/2) + n$。根据master theorem，复杂度为O(nlogn). 当然，最坏情况是O(n^2).空间复杂度为递归栈的大小，平均情况为O(logn), 最坏情况为O(n). 不过，快排的这个递归是尾递归，可以比较容易的改写成迭代形式。如果编译器有尾递归优化，空间复杂度为O(1).
如果不用随机选轴点，最坏情况（**数组已经接近有序**）时，复杂度O(n^2)，因为只有越接近“平均二分”，才能达到O(nlogn)复杂度。但是平均情况还是O(nlogn)。 为了能够达到这个“期望”复杂度，我们需要**随机选择轴点**才行。

[master theorem] ![img](https://pic1.zhimg.com/80/v2-ca7830ee5d665aa8d4437322830bac60_1440w.jpg)

【总结】

- `random.randint(low,high)` 随机选取[low,high]之间的一个元素，左闭右闭。
- 快排是不稳定的排序方法


----

那么如何将快排改造成能解topk问题呢？首先，我们再来分析一下快排，其中有“划分”和“递归”两个过程：

**划分：**  将数组 `a[l⋯r]` 划分成两个子数组 `a[l⋯q−1]`、`a[q+1⋯r]`，使得 `a[l⋯q−1] `中的每个元素小于等于 a[q]，且 a[q]小于等于 `a[q+1⋯r] 中的每个元素`。这个a[q]就称为“**轴点**”。
**递归**： 通过递归调用快速排序，对子数组`a[l⋯q−1]` 和`a[q+1⋯r]` 进行**排序**。

由此可以发现**每次经过划分操作后，我们一定可以确定一个元素（轴点）的最终位置**！如果轴点是在第 **n-k** 位，那么说明我们已经找到了这个topk元素了！

**如果轴点位置比目标下标（n-k）小，就递归右子区间，否则递归左子区间**。 这样就可以把原来递归两个区间变成只递归一个区间！

我们只需要在快排的基础上添加一些代码：

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        self.ans = -1  ###使用这个来记录答案
        def quicksort(nums,begin,end): 
            if begin > end: ###无元素区间.【易错】不是begin>=end
                return
            pivot_pos = find_pivot(nums,begin,end) 
            if pivot_pos == len(nums)-k: ##已经找到！
                self.ans = nums[pivot_pos]
                return 
            if pivot_pos > len(nums)-k: ###递归左边的
                quicksort(nums,begin,pivot_pos-1) ##排序轴点之前的（原地）
            else: ##递归右边的
                quicksort(nums,pivot_pos+1,end) ##排序轴点之后的（原地）
        
        def find_pivot(nums,begin,end):
            rand_pos = random.randint(begin,end) 
            nums[end],nums[rand_pos] = nums[rand_pos], nums[end] 
            pivot_value = nums[end] 
            split = begin
            for i in range(begin,end): 
                if nums[i] <= pivot_value:
                    nums[split],nums[i] = nums[i],nums[split]
                    split += 1
            nums[end], nums[split] = nums[split], nums[end]
            return split

        quicksort(nums,0,len(nums)-1)
        return self.ans
```

`find_partition`函数最后的结果是，arr的左边都<轴点，arr的右边都>轴点，轴点的位置是`split`

【易错点】

- 可以看到，这里是用一个变量self.ans来记录答案，只要有一次符合条件，self.ans便会记录下来答案。
- 返回条件是**low > high, 而不是low >= high**, 因为如果是low >= high的话，会导致单元素区间不能记录self.ans. 

【复杂度分析】T(n) = T(n/2) + O(n), 这不是一个完全的递归树，按照每次排序刚好定位到中间来想，递归树的下一层只有T(n/2)，不考虑另一半。根据master theorem，复杂度为**O(n)**. 当然，最坏情况是O(n^2). 

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
