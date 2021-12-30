# Python 奇技淫巧

- reverse string   

  ```python
  s = s[::-1]
  ```

- 删除第0个值/最后一个值：

```python
s = s[:-1] //删除最后一个
s = s[1:] // 删除第1个
```

- 排序

  - 正常排序：

    ```python
    nums = sorted(nums,reverse = True)
    ```

  - 自定义排序算法：

  ```python
  import functools
  def compare(A, B): # 名字可以随便取，不一定得是“compare"
  	if str(A)+str(B) > str(B)+str(A):
  			return -1 # 表示此种情况下A排在B左边，其实只要return negative value就行
  	elif str(A)+str(B) > str(B)+str(A):
  			return 1 # 表示此种情况下A排在B右边，其实只要return positive value就行
  	else:
  			return 0 # 表示相等，就按照循环访问的顺序排
  
  nums.sort(key = functools.cmp_to_key(compare))
  ```

  - 自定义排序算法：intervals.sort(key=lambda x: x[0])
  - 字典排序：

  ```python
  sorted(key_value.items(), key = lambda kv:(kv[1], kv[0]))
  ```

  - 手动冒泡排序：

  ```python
  def bubble_sort(arr,length):
      for i in range(0,length-1):
          for j in range(0,length-i-1):
              if(arr[j+1]<arr[j]):
                  tmp = arr[j+1]
                  arr[j+1] = arr[j]
                  arr[j] = tmp
              
      return arr
  ```

  - 手写快排：

  每次选择一个轴点(**pivot**), 使得左边的元素均小于它，右边的元素都大于它。

  ![img](https://pic1.zhimg.com/80/v2-2b70124e17ea9fea9eda1e9bfb745a58_1440w.jpg)

  把前半部分和后半部分分别排序之后，原序列自然有序。因此整体的递归结构如下：

  ```cpp
  void quick_sort(int* arr,int low,int high){//左闭右闭
      if(low >= high) return; //单元素向量必定有序
      int mid = find_partition(arr, low,high); //构造轴点，此时左边都<轴点，右边都>轴点
      quick_sort(arr, low,mid - 1); //排序左边
      quick_sort(arr, mid+1,high); //排序右边
  }
  ```

  现在就剩下find_partition函数了。mergesort的难点在于**合**，而quicksort的难点在于**分**。如何能够实现上述的划分呢？--- 培养一个轴点

  1. 首先，选择最后一个元素--pivot = A[right]作为我们要培养的轴点
  2. 一个指针j从low遍历到high-1；另外维护一个split，使得[low,split]的值都<轴点，[split+1,high]的值都>轴点

  ![img](https://pic1.zhimg.com/80/v2-92d87d44c985b2f9589b7626f58c6ba0_1440w.jpg)

  - 在考察j时，如果arr[j]>=pivot, 不用动；如果arr[j] < pivot, 交换arr[split+1]和arr[j], split++

  \3. 最终，交换arr[split+1]和arr[right], 返回split+1

  ```cpp
  int find_partition(int* arr, int low,int high){
      int pivot = arr[high];
      int split = low-1;
      for(int j = low; j<=high-1;j++){
          if(arr[j]<pivot){
              swap(arr[split+1],arr[j]);
              split++;
          }
      }
      swap(arr[split+1],arr[high]);
      return split+1;
  }
  ```


- 数子串个数: count

  `cnt = s.count(substr,0,len(string))`

- 判断素数

```python
def is_prime4(x):
    if (x == 2) or (x == 3):
        return True
    if (x % 6 != 1) and (x % 6 != 5):
        return False
    for i in range(5, int(x ** 0.5) + 1, 6):
        if (x % i == 0) or (x % (i + 2) == 0):
            return False
    return True
```

- list.remove()只能去掉一个元素
- bin(n)直接把十进制转为2进制; oct转为八进制；hex转16进制





