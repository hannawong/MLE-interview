### 1. 数据类型长度

- char: 1个字节
- bool: 1个字节
- int: 4个字节, `-2**31~2**31-1`; 如果是无符号整型，范围则是`0~2**32-1`
- long long: 8个字节，`-2**63~2**32-1`; 如果是无符号整型，范围则是`0~2**64-1`
- float: 单精度浮点数，4个字节
- double: 双精度浮点数，8个字节
- 指针：都是4个字节，因为内存中地址是32位的（如果是32位计算机的话）
- string: 24个字节



### 2. 结构体内存分配

对结构题存储的三点要求：

1) 结构体变量的首地址能够被其最宽基本类型成员的大小所整除；

2) 结构体每个成员相对于结构体首地址的偏移量都是成员大小的整数倍，如有需要编译器会在成员之间加上填充字节；

3) 结构体的总大小为结构体**最宽基本类型成员大小的整数倍**，如有需要编译器会在最末一个成员之后加上填充字节

不同顺序来定义结构体的成员，最后的占用空间是不一样的！所以，在这里可以进行内存优化。



### 3. 内存管理

![img](https://pic1.zhimg.com/80/v2-f59ca7bfe6ff81384cf3b7f37e9ce117_1440w.jpeg)

1、栈区— 对应`stack`。由编译器自动分配释放 ，存放函数调用的**参数**值、返回值，**局部变量**的值等。

2、堆区 — 对应`heap`。由**new**分配的内存块，他们的释放编译器不去管，由我们的应用程序去控制，一般一个new就要对应一个delete。如果程序员没有释放掉，程序会一直占用内存，导致内存泄漏。在程序结束后，操作系统会自动回收。注意它与数据结构中的堆是两回事，分配方式倒是类似于链表。

3、全局区（静态区）—全局变量和静态变量的存储是放在一块的，**初始化**的全局变量和静态变量在一块区域，即`.data`;  **未初始化**的全局变量和未初始化的静态变量在相邻的另一块区域，即`.bss`。程序结束后由系统释放。

4、常量区 —对应`.rodata`(read only data)。不可修改。程序结束后由系统释放。

5、程序代码区—对应`.text`。存放函数体的二进制代码。



【new/delete和alloc/free的区别】

相同点：

- 都可以用来动态申请内存和释放内存

不同点：  

- new和delete是关键字/运算符，这就意味着它是编译器可控的，比如new一个对象，编译器可以做一些处理，调用对应类的构造函数。而malloc/free是标准库函数，不在编译器控制权限之内，不能把执行构造/析构函数的任务强加给malloc/free。

- new运算符建立的是一个**对象**，你可以访问这个对象的成员函数、不用直接访问其地址空间。new的时候相当于调用了这个对象的**构造函数**。而malloc分配的只是一片**内存**区域，可以直接用**指针访问**，而且还可以在里面移动指针。
- C++中可以用new/delete管理内存，而C中只能用malloc/free来管理动态内存，十分不方便。（因为C++支持**面向对象**，而C不支持）
- new可以**自动计算**需要分配的空间，而malloc需要**手动计算**字节数。

- 综上所述，new建立的是一个对象，而malloc分配的是一块内存。new可以认为是malloc加上构造函数组成，delete可以认为是free加上析构函数组成。new构建的指针是带类型信息的，而malloc返回的都是void* 指针。



### 4. STL

【push_back和emplace_back】

在 `C++11` 之后，`vector` 容器中添加了新的方法：`emplace_back()` ，和 `push_back()` 一样的是都是在容器末尾添加一个新的元素进去，不同的是 `emplace_back()` 在效率上相比较于 `push_back()` 有了一定的提升。

1️⃣在引入**右值引用、转移构造函数**之前，通常使用push_back()向容器中加入一个**右值元素**（如vec.push_back(2)）的时候，首先会调用**构造函数构造这个临时对象**，然后需要**调用拷贝构造函数将这个临时对象放入容器中**。**原来的临时变量释放**。这样造成的问题是**临时变量申请的资源就浪费**。

2️⃣引入了右值引用，转移构造函数后，在push_back()右值时就会调用构造函数和转移构造函数，减少了一次拷贝构造，vector中的对应位置直接指向临时变量。

3️⃣在这上面有进一步优化的空间就是使用emplace_back。

使用emplace_back在容器尾部添加一个元素，这个元素直接在**原地构造**，**不需要触发拷贝构造和移动构造**。而且调用形式更加简洁，可以**直接根据参数初始化**临时对象的成员，而不用显式的创造出来一个对象。

看下面这个例子：

```c++
#include <vector>  
#include <string>  
#include <iostream>  

struct President  
{  
    std::string name;  
    std::string country;  
    int year;  

    President(std::string p_name, std::string p_country, int p_year)  
        : name(std::move(p_name)), country(std::move(p_country)), year(p_year)  
    {  //构造函数
        std::cout << "I am being constructed.\n";  
    }
    President(const President& other)
        : name(std::move(other.name)), country(std::move(other.country)), year(other.year)
    { //拷贝构造函数
        std::cout << "I am being copy constructed.\n";
    }
    President(President&& other)  
        : name(std::move(other.name)), country(std::move(other.country)), year(other.year)  
    {  //移动构造函数
        std::cout << "I am being moved.\n";  
    }  
    President& operator=(const President& other);  
};  

int main()  
{  
    std::vector<President> elections;  
    std::cout << "emplace_back:\n";  
    elections.emplace_back("Nelson Mandela", "South Africa", 1994); //可以直接用参数列表初始化，而不用显式的创建出一个类来。

    std::vector<President> reElections;  
    std::cout << "\npush_back:\n";  
    reElections.push_back(President("Franklin Delano Roosevelt", "the USA", 1936));  //必须显示创建类

}

```

```
emplace_back:
I am being constructed.

push_back:
I am being constructed.
I am being moved.
```

可以看到emplace_back支持直接使用**构造参数列表**来添加元素，可以直接就地初始化，不需要移动、拷贝构造函数。