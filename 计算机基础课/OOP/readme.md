# OOP

## 1. 封装与接口

【函数重载】：“名一样，意不同”。靠**参数类型**来区分。

```
sum(int a, int b);
sum(double a, double b);
```

【NULL】c++中，NULL被定义为**0**. c++11中引入nullptr，是真正意义上的空**指针**。

【类】：类的成员（数据、函数）访问权限有public, private, protected.

- 被public修饰的成员可以在类外用“.”操作符访问。

- 被private修饰的成员不允许在类外用“.”操作符访问

```c++
class Matrix {
public:
	void fill(char dir);
private:
	int data[6][6];
}; 
```

【this指针】所有成员函数的参数中，隐含着一个指向**当前对象**的指针变量，其名称为this。

【运算符重载】

```c++
class Test {
public:
  int operator() (int a, int b) { //()运算符重载
    cout << "operator() called. " << a << ' ' << b << endl;
    return a + b;
  }
  int& operator[] (const char* name){ // []运算符重载
    	for (int i = 0; i < 7; i++) {
      		if (strcmp(week_name[i], name) == 0) 
				return temp[i];
    	}
  }
  Test operator++ () { //++运算符重载
    ++data;
    return Test(data);
  }
};	

```

【友元】

•在类内进行友元的声明。

•被声明为友元的函数或类，具有对出现友元声明的类的**private及protected**成员的访问权限。即可以访问该类的一切成员。

![1641007664555](C:\Users\zh-wa\AppData\Roaming\Typora\typora-user-images\1641007664555.png)

•友元不传递：朋友的朋友不是你的朋友

•友元不继承：朋友的孩子不是你的朋友

【内联函数】使用内联函数，编译器自动产生等价的表达式。



## 2. 创建与销毁

【构造函数】

- 构造函数没有返回值类型，函数名与类名相同。

- 构造函数可以重载，即可以使用不同的函数参数进行对象初始化

```c++
class Student {
    long ID;
public:
    Student(long id) { ID = id; }
    Student(int year, int order) { 
			ID = year * 10000 + order; 
	 }
    ...
};
```

【析构函数】

- 一个类只有一个析构函数，名称是“~类名”，没有函数返回值，没有函数参数。

- 编译器在对象生命期结束时自动调用类的析构函数，以便释放对象占用的资源，或其他后处理

