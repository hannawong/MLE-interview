# Linux 命令

### 1. awk

awk 是一种用于处理数据和生成报告的脚本语言。awk 命令编程语言不需要编译，

例如，employment.txt中包含的文本如下：

```text
ajay manager account 45000
sunil clerk account 25000
varun manager sales 50000
amit manager account 47000
tarun peon sales 15000
deepak clerk sales 23000
sunil peon sales 13000
satvik director purchase 80000 
```

1.1 awk default: 直接输出全部

```shell
$ awk '{print}' employee.txt
```

1.2 只输出满足某个pattern的行：

```shell
$ awk '/manager/ {print}' employee.txt 
```

"//"中间的部分是正则表达式

输出：

```
ajay manager account 45000
varun manager sales 50000
amit manager account 47000 
```

1.3 field分割

默认用空格分割，依次存到 \$1,\$2, \$3 \$4,...变量中。\$0 代表一整行。

用-F自己指定分隔符，如`-F,` 表示用逗号分隔。

```shell
$ awk '{print $1,$4}' employee.txt 
```

输出：

```
ajay 45000
sunil 25000
varun 50000
amit 47000
tarun 15000
deepak 23000
sunil 13000
satvik 80000
```

**内建变量：**

- **NR**表示从**awk**开始执行后，按照记录分隔符读取的数据次数，默认的记录分隔符为换行符，因此默认的就是读取的数据行数，**NR**可以理解为Number of Record的缩写。例如：

  ```shell
  $ awk '{print NR,$0}' employee.txt 
  ```

  输出：

  ```
  1 ajay manager account 45000
  2 sunil clerk account 25000
  3 varun manager sales 50000
  4 amit manager account 47000
  5 tarun peon sales 15000
  6 deepak clerk sales 23000
  7 sunil peon sales 13000
  8 satvik director purchase 80000 
  ```

  同时输出了index。

  又如：

  ```shell
  $ awk 'NR==3, NR==6 {print NR,$0}' employee.txt 
  ```

  ```
  3 varun manager sales 50000
  4 amit manager account 47000
  5 tarun peon sales 15000
  6 deepak clerk sales 23000 
  ```

  这个语句可以输出第3行-第6行所有的行！

- NF: 每行的列个数，$NF表示最后一列

  ```shell
  $ awk '{print $1,$NF}' employee.txt 
  ```

  ```
  ajay 45000
  sunil 25000
  varun 50000
  amit 47000
  tarun 15000
  deepak 23000
  sunil 13000
  satvik 80000 
  ```

其他例子：

- ```shell
  $ awk '{ if (length($0) > max) max = length($0) } END { print max }' geeksforgeeks.txt
  ```

  找到最长的行

- ```shell
  $ awk 'length($0) > 10' geeksforgeeks.txt 
  ```

  找到超过10个字符的行

- ```shell
  $ awk '{ if($3 == "B6") print $0;}' geeksforgeeks.txt
  ```

  过滤行

### 2. sed

在不打开文件的前提下实现替换、删除、插入。

##### 2.1 替换

原先的geekfile.txt中文本：

```
unix is great os. unix is opensource. unix is free os.
learn operating system.
unix linux which one you choose.
unix is easy to learn.unix is a multiuser os.Learn unix .unix is a powerful.
```

```shell
$sed 's/unix/linux/' geekfile.txt
```

s代表”substitute“，意为把每行的**第一次出现的**"unix"替换为"linux".

```shell
$sed 's/unix/linux/2' geekfile.txt
```

把每行**第二次**出现的"unix"替换为"linux".

```shell
$sed 's/unix/linux/g' geekfile.txt
```

"g"代表"global",意为把**所有**"unix"都替换为"linux".

```shell
$sed 's/unix/linux/3g' geekfile.txt
```

把第三个~最后一个出现的"unix"都替换为"linux".

#### 2.2 删除

```shell
$ sed '5d' filename.txt
```

删除第五行，d代表"delete".

```shell
$ sed '$d' filename.txt
```

删除最后一行

```shell
$ sed '3,6d' filename.txt
```

删除第3-第6行

```shell
$ sed '12,$d' filename.txt
```

删除第12~最后一行

```shell
$ sed '/abc/d' filename.txt
```

删除某个pattern matching的行



### 3. 其他常见的linux命令

3.1 ps -aux：查看进程

经常和grep配合使用，如`ps -aux | grep "zh-wang"`

3.2 wc -l: 查看文件行数

`ls -lh | grep "^-" | wc -l`可以看整个文件夹内文件的个数

3.3 ls -lh: 看文件权限和大小

![img](https://pic2.zhimg.com/80/v2-2d93112ee1fb49a03ba5f41695701bf1_1440w.png)

3.4 top

top命令是Linux下常用的性能分析工具，能够实时显示系统中各个进程的资源占用状况，类似于Windows的任务管理器。例如`top -u jiayu_xiao`

可显示进程id，user，优先级，cpu占用率，mem占用率，运行时间，运行命令等信息。