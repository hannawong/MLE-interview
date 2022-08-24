# `make` 命令的使用和 `Makefile` 的编写

网上可以找到的比较好的 make 教程：

- [Make 命令教程](https://www.ruanyifeng.com/blog/2015/02/make.html)
- [A Simple Makefile Tutorial](https://www.cs.colby.edu/maxwell/courses/tutorials/maketutor/)

#### 1. 概述

代码变成可执行文件，叫做编译；先编译这个，还是先编译那个（即编译的安排），叫做构建（build）。

`make` 命令的功能就是按照 Makefile 中编写的规则来生成一些文件，这些文件之间会有**依赖的关系**，`make` 会按照依赖关系增量地进行生成，达到**编译一个完整的程序**的目的。

Makefile文件由一系列规则（rules）构成。每条规则的形式如下。

> ```bash
> <target> : <prerequisites> 
> [tab]  <commands>
> ```

每条规则就明确两件事：构建target的prerequisites是什么，以及如何构建(command)。

###### 1.1 target

目标通常是文件名，指明Make命令所要构建的对象，比如上文的 test.o。目标也可以是多个文件名，之间用空格分隔。

除了文件名，目标还可以是某个操作的名字，这称为"伪目标"（phony target）。

> ```bash
> clean:
>       rm *.o
> ```

上面代码的目标是clean，它不是文件名，而是一个操作的名字，属于"伪目标 "，作用是删除对象文件。

> ```bash
> $ make  clean
> ```

但是，如果当前目录中，正好有一个文件叫做clean，那么这个命令不会执行。因为Make发现clean文件已经存在，就认为没有必要重新构建了，就不会执行`rm *.o`这个命令了。

为了避免这种情况，**可以明确声明clean是"伪目标"**，写法如下。

> ```bash
> .PHONY: clean
> clean:
>         rm *.o temp
> ```

声明clean是"伪目标"之后，make就不会去检查是否存在一个叫做clean的文件，而是每次运行都执行对应的命令。

如果Make命令运行时没有指定目标，默认会执行Makefile文件的**第一个目标。**

> ```bash
> $ make
> ```

上面代码执行Makefile文件的**第一个目标**。

###### 1.2 Prerequisite

前置条件通常是**一组**文件名，之间用空格分隔。

> ```bash
> source: file1 file2 file3
>     （command）
> ```

只要有一个前置文件有过更新，"目标"就需要重新构建。



#### 2. 语法

- 正常情况下，make会**打印每条command**，然后再执行，这就叫做回声（echoing）。为了关闭回声，可以在命令的前面加上`@`。由于在构建过程中，需要了解当前在执行哪条命令，所以通常只在注释和echo命令前面加上`@`。

- 通配符（wildcard）: *.o 表示所有后缀名为o的文件。

  > ```bash
  > clean:
  >         rm -f *.o
  > ```

- Make命令允许对文件名，进行类似正则运算的匹配，主要用到的匹配符是%。这样可以针对所有的 .cpp 文件构建对应的 .o 文件。

> ```bash
> %.o: %.c
> ```

使用匹配符%，可以将大量同类型的文件，只用一条规则就完成构建。

- 变量符 `$`:

> ```bash
> txt = Hello World
> test:
>     @echo $(txt)
> ```

上面代码中，变量 txt 等于 Hello World。调用时，变量需要放在 `$( ) `之中。

- 调用Shell变量，需要在美元符号前，再加一个美元符号，这是因为Make命令会对美元符号转义。

> ```bash
> test:
>     @echo $$HOME  ##/data/jiayu_xiao
> ```

- 用 `$^` 代表所有依赖， `$@` 代表目标文件， `$<` 代表第一个依赖，如在 `xxx: aaa bbb ccc` 中，`$^` 代表 `aaa bbb ccc` ，`$@` 代表 `xxx` ，`$<` 代表 `aaa`。



#### 3. 例子

```
CXX ?= g++
LAB_ROOT ?= ../..
BACKEND ?= LINUX
CXXFLAGS ?= --std=c++11 -I $(LAB_ROOT)/HAL/include -DROUTER_BACKEND_$(BACKEND)
LDFLAGS ?= -lpcap
```

这一部分定义了若干个变量，左边是 key ，右边是 value ，`$(LAB_ROOT)` 表示变量 `LAB_ROOT` 的内容。条件赋值 `?=` 表示的是只有在第一次赋值时才生效，可以通过 `make CXX=clang++` 来让 CXX 变量的内容变成 clang++。

```shell
.PHONY: all clean
all: router

clean:
    rm -f *.o router std

%.o: %.cpp 
    $(CXX) $(CXXFLAGS) -c $^ -o $@

hal.o: $(LAB_ROOT)/HAL/src/linux/router_hal.cpp
    $(CXX) $(CXXFLAGS) -c $^ -o $@

router: main.o hal.o protocol.o checksum.o lookup.o forwarding.o
    $(CXX) $^ -o $@ $(LDFLAGS)
```

使用的时候只需要 `make` 就可以了。