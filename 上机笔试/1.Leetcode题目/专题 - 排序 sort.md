# 专题 - 排序

### 1. 排序算法

#### 1.1 快排

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



【注】如果不用随机选轴点，最坏情况（**数组已经接近有序**）时，复杂度O(n^2)，因为只有越接近“平均二分”，才能达到O(nlogn)复杂度。但是平均情况还是O(nlogn)。 为了能够达到这个“期望”复杂度，我们需要**随机选择轴点**才行。

【总结】

- `random.randint(low,high)` 随机选取[low,high]之间的一个元素，左闭右闭。

- 快排是不稳定的排序方法



#### 1.2 冒泡排序

```python
def bubble_sort(arr,length):
    for i in range(0,length-1): ##迭代length-1次
        for j in range(0,length-i-1): ##每次都把最大的放到最后面
            if(arr[j+1]<arr[j]):
                tmp = arr[j+1]
                arr[j+1] = arr[j]
                arr[j] = tmp
    return arr
```

最坏情况：输入序列完全反序，复杂度O(n^2), 

最好情况：输入序列业已顺序，经过一趟扫描，即确认有序，并随即退出。复杂度O(n). 

冒泡排序是稳定的排序算法，因为两个相邻元素如果相等就不用交换顺序。

#### 1.3 归并排序

归并排序采用分而治之的思想：

```python
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        def mergesort(nums,begin,end):
            if begin >= end: return ###单元素区间，必然有序
            middle = (begin+end) // 2
            mergesort(nums,begin,middle)
            mergesort(nums,middle+1,end)
            merge(nums,begin,middle,end)
            
        def merge(nums,begin,middle,end): #[begin,middle] is sorted, [middle+1,end] is sorted 
            ans = []
            ptr1 = begin; ptr2 = middle+1
            while ptr1 <= middle and ptr2 <= end:
                if nums[ptr1] <= nums[ptr2]: ###【注意】这里是<=, 方可实现稳定排序。同时，这样写也是为了逆序对实现的方便
                    ans.append(nums[ptr1])
                    ptr1 += 1
                else:
                    ans.append(nums[ptr2])
                    ptr2 += 1
            while ptr1 <= middle:
                ans.append(nums[ptr1])
                ptr1 += 1
            while ptr2 <= end:
                ans.append(nums[ptr2])
                ptr2 += 1
            for i in range(begin,end+1):
                nums[i] = ans[i-begin]
        mergesort(nums,0,len(nums)-1)
        return nums
```

复杂度$O(nlogn)$.

归并排序是稳定的排序算法：出现雷同元素时，左侧子向量优先。

#### 1.4 选择排序 - O(n)

回忆冒泡排序...每趟扫描交换都需要O(n)次比较、O(n)次交换，然而其中O(n)次交换完全没有必要。扫描交换的实质效果无非是，通过比较找到当前的最大元素M，并通过交换使之就位。如此看来，在经过O(n)次比较确定M之后，仅需一次交换则足以。

![img](https://pic3.zhimg.com/80/v2-ea8594b03c98efae00b9f4baabadf491_1440w.png)

**稳定性：**如果约定“靠后者优先返回”，则可保证重复元素在列表中的相对次序在排序后不变，即排序算法是稳定的。如下图所示：

![img](https://pic3.zhimg.com/80/v2-2a66f216b0f88f158acf779615c58817_1440w.png)

#### 1.5 插入排序 - O(n)

序列总分为两部分：有序+无序。在S中查找适当的位置以插入**e**：

![img](https://pic3.zhimg.com/80/v2-5856528a10706ec2c259cc47d362e246_1440w.jpeg)

在代码实现的时候注意空位的选择：

```c++
int* selection_sort(int* arr,int len){
    for(int i = 1;i<len;i++){ //从第二个元素开始作为key
        int key = arr[i];
        int j = i-1; //从右向左扫描
        while(key<arr[j]){ //如果一直都比key大
            arr[j+1] = arr[j];
            j--;
        }
        //直到找到<key的值，其后面必是空位
        arr[j+1] = key;
    }
    return arr;
}
```

**性能分析：**

最好情况是序列完全有序的情况，那么每次迭代只需要1次比较、0次交换，累计O(n)时间；

最坏情况是序列完全逆序的情况，那么第k次迭代需要k次比较、1次交换，累计O(n^2)时间。

**稳定性：**

如果key = arr[i]就可以完成插入，说明插入排序是稳定的。

#### 1.6 桶排序





##### [451. 根据字符出现频率排序](https://leetcode.cn/problems/sort-characters-by-frequency/)

难度中等

给定一个字符串 `s` ，根据字符出现的 **频率** 对其进行 **降序排序** 。一个字符出现的 **频率** 是它出现在字符串中的次数。

返回 *已排序的字符串* 。如果有多个答案，返回其中任何一个。

**示例 1:**

```
输入: s = "tree"
输出: "eert"
解释: 'e'出现两次，'r'和't'都只出现一次。
因此'e'必须出现在'r'和't'之前。此外，"eetr"也是一个有效的答案。
```

由于字符串的长度有上界，那么**一个字符出现的最大次数也是有上界的**。这样，我们就可以建立桶，每个桶存储特定频次的字符，然后从后向前遍历桶就行了。

```python
class Solution:
    def frequencySort(self, s: str) -> str:
        maxfreq = 0
        dic = {}
        for i in range(len(s)):
            if s[i] not in dic:
                dic[s[i]] = 1
                maxfreq = max(maxfreq,1)
            else:
                dic[s[i]] += 1
                maxfreq = max(maxfreq,dic[s[i]])
        bucket = [[] for _ in range(maxfreq+1)] ##按照频率构建桶
        for key in dic:
            bucket[dic[key]].append(key)
        ans = ""
        for i in range(len(bucket)-1,-1,-1):
            for j in bucket[i]:
                ans += j * i
        return ans
```



### 2. 逆序对

对于冒泡排序，在序列中交换一对逆序元素，则逆序对总数必然减少。特殊地，交换一对**紧邻**的逆序元素，逆序对总数恰好减少1. 所以，对于bubblesort而言，**交换操作的次数恰好等于输入序列所含逆序对的总数**。

![img](https://pic2.zhimg.com/80/v2-90ed2c9736846ad659b0c193aca37801_1440w.jpeg)

任意给定一个序列，如何统计其中逆序对的总数呢？蛮力算法需要O(n^2)时间，但是借助归并排序，仅需O(nlogn)时间。具体方法见https://github.com/hannawong/MLE-interview/blob/master/%E4%B8%8A%E6%9C%BA%E7%AC%94%E8%AF%95/1.Leetcode%E9%A2%98%E7%9B%AE/%E5%89%91%E6%8C%87%20Offer%2051.%20%E6%95%B0%E7%BB%84%E4%B8%AD%E7%9A%84%E9%80%86%E5%BA%8F%E5%AF%B9%5Bhard%5D.md



### 3. 自定义排序(python)

- `intervals.sort(key=lambda x: x[0])   `: 按第一个元素排序

- 复杂逻辑自定义排序：

  ```python
  import functools
  def cmp(a, b): ##自定义compare函数
      if int(str(a)+str(b)) > int(str(b)+str(a)):
          return  -1
      else:
          return 1
  nums = sorted(nums, key=functools.cmp_to_key(cmp))
  ```

- dictionary 按照value排序：

​       ` dic = sorted(dic.items(), key=lambda x: x[1], reverse=True)`

​      `dictionary.sort(key = lambda x: (-len(x),x)) ##排序`
