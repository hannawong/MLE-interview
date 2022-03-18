# Top-K问题

给定整数数组 nums 和整数 k，请返回数组中第 k 个**最大的元素**。

请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。

```python
示例 1:

输入: [3,2,1,5,6,4] 和 k = 2
输出: 5
```



### 方法1：小根堆 

**求topk个最大元素，用小根堆；求topk个最小元素，用大根堆。**

对于前k个元素，直接插入堆即可，这样形成了一个小根堆。对于后面的n-k个元素，如果比小根堆的根还小，那么肯定不用管，因为它一定不是第k大的数；反之，如果比小根堆的根大，那么弹出根，插入当前值。最终，小根堆的**根**就是这个第k大的元素了。

时间复杂度O(nlogn)。

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



### 方法2：利用快排

**划分：**  将数组 a[l⋯r] 划分成两个子数组 a[l⋯q−1]、a[q+1⋯r]，使得 a[l⋯q−1] 中的每个元素小于等于 a[q]，且 a[q]小于等于 a[q+1⋯r] 中的每个元素。这个a[q]就称为“轴点”。
**递归**： 通过递归调用快速排序，对子数组a[l⋯q−1] 和a[q+1⋯r] 进行排序。

由此可以发现每次经过「划分」操作后，我们一定可以确定一个元素的最终位置，即 x 的最终位置为 q，所以只要某次划分的 q 为倒数第 k 个下标的时候，我们就已经找到了答案！

```python
class Solution:
    ans = -1
    def findKthLargest(self, nums: List[int], k: int) -> int:
        self.quick_sort(nums,0,len(nums)-1,k)  
        return self.ans
        
    def quick_sort(self,arr,low,high,k): ##左闭右闭
        if(low > high):
            return 
        mid = self.find_partition(arr, low,high) ##构造轴点，此时左边都<轴点，右边都>轴点
        if mid == len(arr)-k:
            self.ans = arr[mid] ##记录答案
            return 
        self.quick_sort(arr, low,mid - 1,k) ## 排序左边
        self.quick_sort(arr, mid+1,high,k) ##排序右边

    def find_partition(self,arr, low, high):
        pivot_pos = random.randint(low, high)  ##随机选轴点
        arr[pivot_pos],arr[high] = arr[high],arr[pivot_pos] ##和最后一位交换
        pivot = arr[high]
        split = low-1
        for j in range(low,high):
            if (arr[j] < pivot):
                arr[split+1],arr[j] = arr[j],arr[split+1]
                split+=1
        arr[split+1],arr[high] = arr[high],arr[split+1]

        return split + 1
```

`find_partition`函数最后的结果是，arr的左边都<轴点，arr的右边都>轴点，轴点的位置是`split+1`

【注意】randint是左闭右闭，这个和range不一样！

【易错点】可以看到，这里是用一个变量self.ans来记录答案，只要有一次符合条件，self.ans便会记录下来答案，所以可以不用递归的返回值。



【相似题】最小的K个数

设计一个算法，找出数组中最小的k个数。以任意顺序返回这k个数均可。

```
示例：
输入： arr = [1,3,5,7,2,4,6,8], k = 4
输出： [1,2,3,4]
```

最小的K个数，用大根堆解决；相反的，最大的K个数，用小根堆解决。

```python
class Solution:
    def smallestK(self, arr: List[int], k: int) -> List[int]:
        heap = []
        if k == 0:
            return []
        for i in range(len(arr)):
            if len(heap) < k: ##如果堆未满，则直接push
                heapq.heappush(heap,-arr[i]) ##用负数模拟最大堆
            else: ##堆已满
                if arr[i] < -heap[0]: ##如果arr[i]比堆顶还大，那么它一定不是最小的K个数，就不用考虑了。
                    heapq.heappop(heap) ##pop堆顶
                    heapq.heappush(heap,-arr[i]) ##push进新的数
        return [-i for i in heap]
```

