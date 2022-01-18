# 模板和STL初步

### 1. 函数模板和类模板

##### 1.1 函数模板

有些算法实现与**类型**无关，所以可以将函数的参数类型也定义为一种特殊的“参数”，这样就得到了“函数模板”。

例如：任意类型两个变量相加的“函数模板”

```c++
template <typename T> T sum(T a, T b) { 
	return a + b; 
}
```

这样，不管实际调用时候的参数是什么类型（int, float, class...）只要这个类型定义了加法运算，就可以调用这个函数，而不用对每种类型都定义一遍相同的函数。



##### 1.2 类模板

同样的，在定义类时也可以将一些类型信息抽取出来，用模板参数来替换，从而使类更具通用性。这种类被称为“类模板”。例如：

```c++
template <typename T> class A {
  T data;
public:
  void print() { cout << data << endl; }
}; 

int main() {
  A<int> a;  //调用类模板
  a.print();
}
```

类模板还可以有”模板参数“。其中**类型参数**就是上文说的typename，还可以有非类型参数，如整数：

```c++
template<typename T, unsigned size>
class array {
	T elems[size];
	 ... 
}; 
array<char, 10> array0; 
```



##### 1.3 函数模板特化

有时，有些类型并不合适，则需要对模板进行特殊化处理，这称为“模板特化”。

对函数模板，如果有多个模板参数 ，则特化时必须提供**所有参数**的特例类型，**不能部分特化**。但可以用**重载**来替代部分特化。

例如，函数模板特化：

![img](https://pic2.zhimg.com/80/v2-2c7aa70b702f8e9245d6a506647552b3_1440w.png)

函数模板是不能部分特化的，只能重载：

![img](https://pic3.zhimg.com/80/v2-0b883ecf576574d91911f8f53d78efe1_1440w.png)

##### 1.4 类模板特化

类模板是允许部分特化的，即部分限制模板的通用性，如通用模板为：

`template<typename T1, typename T2> class A { ... }; `

部分特化：第二个类型指定为int

`template<typename T1> class A<T1, int> {...}; `

全部特化：指定所有类型

`template<> class A<int, int\> { ... }; `



【模板与多态】

模板使用泛型标记，使用同一段代码，来关联不同但相似的特定行为，最后可以获得不同的结果。模板也是多态的一种体现。

- 但对模板的处理是在**编译期**进行的，每当编译器发现对模板的一种参数的使用，就生成对应参数的一份代码。这意味着所有模板参数必须在编译期确定，不可以使用变量。这种多态称为**静多态**。
- 基于继承和虚函数的多态在**运行期**处理，称为**动多态**



### 2. 命名空间

为了避免在大规模程序的设计中，以及在程序员使用各种各样的C++库时，标识符的**命名发生冲突**，标准C++引入了关键字namespace（命名空间），可以更好地控制标识符的作用域。所以说，为啥在大规模代码中不写`using namespace std`呢？就是为了**防止污染命名空间**。 

定义命名空间：

```c++
namespace A {
	int x, y;
}
```

使用命名空间：

```c++
A::x = 3;
A::y = 6;
```

使用using声明简化命名空间使用。例如使用整个命名空间：所有成员都直接可用

```c++
using namespace A;
x = 3; y = 6;
```



### 3. STL初步 - 容器与迭代器

STL(standard template library), 标准模板库是基于**模板**编写的。

STL的命名空间是std，一般使用`std::name`来使用STL的函数或对象。也可以使用`using namespace std`来引入STL的命名空间（不推荐在大型工程中使用，容易污染命名空间）

##### 3.1 STL容器

**（1）pair**

最简单的容器，由两个单独数据组成。

```c++
std::pair<int, int> t;
t.first = 4; t.second = 5; //pair支持修改
```

创建：使用函数`make_pair`: `auto t = make_pair(“abc”, 7.8);`可以自动推导成员类型。

支持小于、等于等比较运算符。需先比较first，后比较second。当然，这要求成员类型支持比较(实现比较运算符重载)。

map中大量使用pair这种数据结构。

**（2）tuple**

c++11新增，为pair的扩展，是由若干成员组成的**元组**类型。tuple中的元素要在编译时就确定好，是**不能够修改元素、删除元素**的。不能当成数组来用。

通过`std::get`函数获取数据。

```c++
v0 = std::get<0>(tuple1);
v1 = std::get<1>(tuple2);
```

创建：make_tuple函数。`auto t = make_tuple(“abc”, 7.8, 123, ‘3’);`

创建：tie函数—返回左值引用的元组

```c++
std::tie(x, y, z) = make_tuple(“abc”, 7.8, 123); //等价于 x = "abc"; y = 7.8; z = 123
```

因此常常用于函数多返回值的传递：

```c++
std::tuple<int, double> f(int x){ 
	return make_tuple(x, double(x)/2);
}

int main() {
	int xval; double half_x;
    std::tie(xval, half_x) = f(7);
}
```

**（3）vector**

会自动扩展容量的数组，以循序(Sequential)的方式维护变量集合。 支持下标高速访问。

在末尾添加/删除(高速):  ` x.push_back(1); x.pop_back();`

在中间添加/删除（使用迭代器，低速）:`x.insert(x.begin()+1, 5);`, `x.erase(x.begin()+1);`

**（4）list**

链表容器（底层实现是双向链表），不支持下标等随机访问，但是支持高速的在任意位置插入/删除数据。其访问主要依赖迭代器，操作不会导致迭代器失效（除指向被删除的元素的迭代器外）

**（5）set**

不重复元素组成的**无序**集合。但是其内部还是按大小顺序排列的，比较器由函数对象compare完成。

- 插入：`s.insert(1);`

- 查询：`s.find(1);   //返回迭代器`
- 删除：`s.erase(s.find(1));`   //导致迭代器失效
- 统计：`s.count(1);`   //1的个数，总是0或1

set的内部实现是红黑树这种平衡二叉搜索树，其几乎所有操作复杂度均为O(logn)。这是因为想要实现集合的并、交、补等算法时，需要有序的数据结构。

（6）map

其值类型为pair<Key, T>。map中的元素key互不相同，需要key存在比较器。可以通过下标访问（即使key不是整数）。下标访问时如果元素不存在，则创建对应元素；也可使用insert函数进行插入`s.insert(make_pair(string(“oop”), 1));`

- 查询：find函数，仅需要提供key值，返回迭代器。

- 统计：count函数，仅需要提供key值，返回0或1。

- 删除：erase函数，使用迭代器，导致被删除元素的迭代器失效。

以上部分与set类似。底层实现上也使用红黑树来排解哈希冲突。如果桶中冲突的个数 < 8个，则使用传统的拉链法（链表），如果冲突个数>8个，则把这个链表转化为红黑树。



【总结】

- 序列容器：vector、list
- 关联容器：set、map

序列容器与关联容器的区别：

- 序列容器中的元素有顺序，可以按顺序访问。

- 关联容器中的元素无顺序，可以按数值（大小）访问。

vector中插入删除操作会使操作位置之后全部的迭代器失效，其他容器中只有被删除元素的迭代器失效。



##### 3.2 迭代器

一种检查容器内元素并遍历元素的数据类型。提供一种方法顺序访问一个聚合对象中各个元素, 而又不需暴露该对象的内部表示。使用上类似指针。

例如在vector类中，其实就实现了针对vector的迭代器：

```c++
class vector {
	class iterator {
		...
	}
};
```

`vec.begin()`，返回vector中第一个元素的迭代器；`vec.end()`，返回vector中最后一个元素**之后的位置**的迭代器(左闭右开)。`vec.begin(), vec.end()`都是`vector<int>::iterator `类型的。

- 下一个元素：++iter
- 上一个元素：--iter
- 下n个元素：iter += n
- 上n个元素：iter -= n
- 访问元素值——解引用运算符\*，例如`*iter = 5;`

遍历vector容器的方法：

```
for(vector<int>::iterator it = vec.begin(); it != vec.end(); ++it)  //访问*it,最复杂的写法
for(auto it = vec.begin(); it != vec.end(); ++it) //c++11中用auto来替代迭代器类型
for(auto x : vec) //更简单的写法
```

**迭代器失效：**

调用**insert/erase**后，所**修改位置之后**的所有迭代器失效（原先的内存空间存储的元素被改变）。例如：

```c++
vector<int> vec = {1,2,3,4,5};
auto first = vec.begin();
auto second = vec.begin() + 1;
auto third = vec.begin() + 2;
auto ret = vec.erase(second);
//此时，first指向1，second和third失效
```

![img](https://pic2.zhimg.com/80/v2-3e46f04fb086941d10721acf931e7b8a_1440w.png)

一个绝对安全的准则：在修改过容器后，不使用之前的迭代器。



**调用push_back等修改vector大小的方法时，也可能会使所有迭代器失效。**这是因为vector是会自动扩展容量的数组。除了size，另保存capacity表示最大容量限制。如果size达到了capacity，则另申请一片capacity*2的空间，并整体迁移vector内容。其时间复杂度为均摊O(1)，整体迁移过程使所有迭代器失效。



**在遍历的时候增加元素，也可能会导致迭代器失效。**