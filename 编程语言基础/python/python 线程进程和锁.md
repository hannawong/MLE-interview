# python 线程进程和锁

#### Python 中的多线程“并行“

这几天尝试用 Python3 写一个**多线程**应用，发现了一个之前一直不知道的语言特性：GIL (Global Interpreter Lock) 全局解释器锁。

先写结论：**GIL 的存在让 Python 的多线程应用只能实现并发，而不能实现并行。如果想实现并行，只能通过多进程。**

- 并发：微观上是分时间片的，一个时刻只有一个进程在执行。
- 并行：真正意义上的“并行”，一个时刻有不同的处理机在执行。

在并发中，处理能力并没有得到提升。而并行是真正运用多个处理器同时处理，提升了总体的处理能力。

 python支持multiprocessing 和 multithreading，但是，multithreading其实并不是并行执行，这是因为GIL的缘故。

1、**GIL是什么？**GIL的全称是Global Interpreter Lock(全局解释器锁)，来源是python设计之初的考虑，为了数据安全所做的决定。由于历史原因，Python 并没有锁操作，所以为了防止多个线程之间的读写冲突，Python 使用 GIL 作为全局锁。

![Python 多线程运行](https://i.imgur.com/fdJo9ep.jpg)

2、每个CPU在**同一时间只能执行一个线程**（在单核CPU下的多线程其实都**只是并发，不是并行**，并发和并行从宏观上来讲都是同时处理多路请求的概念。但并发和并行又有区别，并行是指两个或者多个事件在同一时刻发生；而并发是指两个或多个事件在同一**时间间隔**内发生。）



**在Python多线程下，每个线程的执行方式：**

1.获取GIL

2.执行代码直到sleep或者是python虚拟机将其挂起。

3.释放GIL

**可见，某个线程想要执行，必须先拿到GIL，我们可以把GIL看作是“通行证”，并且在一个python进程中，GIL只有一个。拿不到通行证的线程，就不允许进入CPU执行。**

在python2.x里，GIL的释放逻辑是当前线程遇见IO操作或者ticks计数达到100（ticks可以看作是python自身的一个计数器，专门做用于GIL，每次释放后归零，这个计数可以通过 sys.setcheckinterval 来调整），进行释放。

而每次释放GIL锁，线程进行锁竞争、切换线程，会消耗资源。并且由于GIL锁存在，python里一个进程永远只能同时执行一个线程(拿到GIL的线程才能执行)，这就是为什么在多核CPU上，python的多线程效率并不高。



**那么是不是python的多线程就完全没用了呢？**

在这里我们进行分类讨论：

1、CPU密集型代码(各种循环处理、计数等等)，在这种情况下，ticks计数很快就会达到阈值，然后触发GIL的释放与再竞争（多个线程来回切换当然是需要消耗资源的），所以**python下的多线程对CPU密集型代码并不友好。**

2、IO密集型代码(文件处理、网络爬虫等)，多线程能够有效提升效率(单线程下有IO操作会进行IO等待，造成不必要的时间浪费，而开启多线程能在线程A等待时，自动切换到线程B，可以不浪费CPU的资源，从而能提升程序执行效率)。**所以python的多线程对IO密集型代码比较友好。**

而在python3.x中，GIL不使用ticks计数，改为使用计时器（执行时间达到阈值后，当前线程释放GIL），这样对CPU密集型程序更加友好，**但依然没有解决GIL导致的同一时间只能执行一个线程的问题，所以效率依然不尽如人意。**

**多核多线程比单核多线程更差，原因是单核下多线程，每次释放GIL，唤醒的那个线程都能获取到GIL锁，所以能够无缝执行，但多核下，CPU0释放GIL后，其他CPU上的线程都会进行竞争，但GIL可能会马上又被CPU0拿到，导致其他几个CPU上被唤醒后的线程会醒着等待到切换时间后又进入待调度状态，这样会造成线程颠簸(thrashing)，导致效率更低**

回到最开始的问题：经常我们会听到老手说：**“**python下多线程是鸡肋，想要充分利用多核CPU，就用多进程”，原因是什么呢？

原因是：**每个进程有各自独立的GIL**，互不干扰，这样就可以真正意义上的**并行执行**，所以在python中，多进程的执行效率优于多线程(仅仅针对多核CPU而言)。

**所以我们能够得出结论：多核下，想做并行提升效率，比较通用的方法是使用多进程，能够有效提高执行效率**



### multiprocessing

因为进程间并不共享内存，所以无需担心并行中的读写冲突问题。进程之间通过 IPC 进行通信。但 IPC 的 overhead 明显会降低程序的性能。所以多进程适用于通信不频繁的并行程序。

这里，我们可以通过 `Multiprocessing` 来实现一个多进程 Python 程序 (`test3`)。不同的进程会运行在CPU不同的核上，实现真正的「并行」 。



#### [1114. 按序打印](https://leetcode.cn/problems/print-in-order/)

难度简单398

给你一个类：

```
public class Foo {
  public void first() { print("first"); }
  public void second() { print("second"); }
  public void third() { print("third"); }
}
```

三个不同的线程 A、B、C 将会共用一个 `Foo` 实例。

- 线程 A 将会调用 `first()` 方法
- 线程 B 将会调用 `second()` 方法
- 线程 C 将会调用 `third()` 方法

请设计修改程序，以确保 `second()` 方法在 `first()` 方法之后被执行，`third()` 方法在 `second()` 方法之后被执行。

**提示：**

- 尽管输入中的数字似乎暗示了顺序，但是我们并不保证线程在操作系统中的调度顺序。
- 你看到的输入格式主要是为了确保测试的全面性。

 

```python
from threading import Lock

class Foo:
    def __init__(self):
        self.first_lock = Lock()
        self.second_lock = Lock()
        self.first_lock.acquire()
        self.second_lock.acquire()

    def first(self, printFirst: 'Callable[[], None]') -> None:
        
        # printFirst() outputs "first". Do not change or remove this line.
        printFirst()
        self.first_lock.release()


    def second(self, printSecond: 'Callable[[], None]') -> None:
        with self.first_lock:
        # printSecond() outputs "second". Do not change or remove this line.
            printSecond()
            self.second_lock.release()


    def third(self, printThird: 'Callable[[], None]') -> None:
        
        with self.second_lock:
        # printThird() outputs "third". Do not change or remove this line.
            printThird()
```

