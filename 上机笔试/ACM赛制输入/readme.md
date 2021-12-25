# ACM赛制输入

## Python:

1. 一行输入

`inp = input()`

输入为string，需要手动转成int等形式

2. 多行输入，直到EOF为止：

   ```python
   while(True):
       try:
           inp = input().split()
           ...
               
       except:
           break
   ```

   技巧：先当成只有一个输出来写，写完之后运行正确再扩充到多行输入。不然由于try-except, 无法看到报错。