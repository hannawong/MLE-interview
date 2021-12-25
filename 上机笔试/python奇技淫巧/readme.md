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