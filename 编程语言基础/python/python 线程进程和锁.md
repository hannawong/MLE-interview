# python 线程进程和锁



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

