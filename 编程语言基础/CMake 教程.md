# CMake 教程

简单来说，CMake就是个用来生成针对**各个平台**的Makefile的工具。

### step1 一个简单的起点

1. 准备计算平方根的代码，这里命名为calculate.cpp：

```c++
// A simple program that computes the square root of a number
#include <cmath>
#include <cstdlib> 
#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " number" << std::endl;
    return 1;
  }

  // convert input to double
  const double inputValue = atof(argv[1]);

  // calculate square root
  const double outputValue = sqrt(inputValue);
  std::cout << "The square root of " << inputValue << " is " << outputValue
            << std::endl;
  return 0;
}
```

2. 创建`CMakeLists.txt`文件，并声明**cmake版本**、**工程名**、**构建目标app的源文件**

```python
cmake_minimum_required(VERSION 3.10) 

project(CalculateSqrt) ##app工程名

add_executable(CalculateSqrt calculate.cpp) ##构建工程的源文件
```

3. 再创建文件夹`build`. 

-----

为构建可执行文件需要执行如下命令:

```text
cd build
Cmake ..
Make
```

可见，现在build文件夹下出现了**MakeFile**，还有Tutorial这个**可执行文件**。

![image-20220823180124874](C:\Users\jiayu\AppData\Roaming\Typora\typora-user-images\image-20220823180124874.png)

使用`./Tutorial 10` 来**执行**这个可执行文件，得到输出`The square root of 10 is 3.16228`

----

现在，我们希望给生成的可执行文件增加一个版本号。

1）`CMakeList.txt`:

```cmake
cmake_minimum_required(VERSION 3.10)

project(Tutorial VERSION 1.0) ###增加版本号

configure_file(TutorialConfig.h.in TutorialConfig.h) ##我们需要配置一个头文件TutorialConfig.h，用来将版本号传入到源代码中去。


set(CMAKE_CXX_STANDARD 11) # specify the C++ standard
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(Tutorial tutorial.cpp)

target_include_directories(Tutorial PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           ) ##${PROJECT_BINARY_DIR}指的是build文件夹的绝对路径。这句话的意思是，要把build文件夹这个路径添加到Tutorial的路径中去。如果不加这句话，会导致编译的时候找不到TutorialConfig.h,因为TutorialConfig.h位于build/文件夹下，而tutorial.cpp位于build的上层文件夹下。
```

2）`TutorialConfig.h.in`:

```c++
#define Tutorial_VERSION_MAJOR @Tutorial_VERSION_MAJOR@
#define Tutorial_VERSION_MINOR @Tutorial_VERSION_MINOR@
```

当CMake配置这个头文件时，”@Tutorial_VERSION_MAJOR@”和”@Tutorial_VERSION_MINOR@”的值将会		被CMakeLists.txt文件中的值替换。(即1和0)

`tutorial.cpp`:

```c++
#include <cmath>
#include <iostream>
#include <string>

#include "TutorialConfig.h" // 增加头文件

int main(int argc, char* argv[])
{
  if (argc < 2) {
    std::cout << argv[0] << " Version " << Tutorial_VERSION_MAJOR << "."
              << Tutorial_VERSION_MINOR << std::endl; //打印版本号
    std::cout << "Usage: " << argv[0] << " number" << std::endl;
    return 1;
  }

  const double inputValue = std::stod(argv[1]); //使用c++11特性

  const double outputValue = sqrt(inputValue);
  std::cout << "The square root of " << inputValue << " is " << outputValue
            << std::endl;
  return 0;
}
```

上面代码最主要的修改就是包含了TutorialConfig.h头文件，并且打印出版本号。同时，我们使用了c++11的特性。

3）make之后，发现增加了`TutorialConfig.h`文件:

![image-20220823190831883](C:\Users\jiayu\AppData\Roaming\Typora\typora-user-images\image-20220823190831883.png)

里面的内容是：

```c++
#define Tutorial_VERSION_MAJOR 1
#define Tutorial_VERSION_MINOR 0
```

4）运行`./Tutorial`来执行可执行文件。得到输出：

```
./Tutorial Version 1.0
Usage: ./Tutorial number
```

这说明传进来的`Tutorial_VERSION_MAJOR` = 1, `Tutorial_VERSION_MINOR` = 0

### step2. 增加一个library

现在我们为我们的工程添加一个库(library)。这个库包含了自己实现的一个用来计算数的平方根函数`mysqrt`。应用程序可以使用这个库来计算平方根，而不是使用编译器提供的标准库`sqrt`。

我们把这个库放到一个叫做”MathFunctions”的子文件夹中。在/MathFunction 文件夹中创建一个文件`mysqrt.cpp`:

```c++
#include <iostream>

double mysqrt(double x) //自己实现的开平方根函数
{
  if (x <= 0) {
    return 0;
  }

  double result = x;

  // do ten iterations
  for (int i = 0; i < 10; ++i) {
    if (result <= 0) {
      result = 0.1;
    }
    double delta = x - (result * result);
    result = result + 0.5 * delta / result;
    std::cout << "Computing sqrt of " << x << " to be " << result << std::endl;
  }
  return result;
}
```

创建文件`MathFunctions.h`:

```c++
double mysqrt(double x);
```

创建CMakeList.txt：

```cmake
add_library(MathFunctions mysqrt.cpp)
```

----

为了使用这个新库，我们需要修改顶层的`CMakeLists.txt`:

```Cmake
cmake_minimum_required(VERSION 3.10)

project(Tutorial VERSION 1.0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(USE_MYMATH "Use tutorial provided math implementation" ON) ##把MathFunctions库做成可选的,并且默认值为ON,这个设置将会被保存到CACHE当中,所以用户每次打开工程时不必重新配置. 值得注意的是，哪怕这里默认值写的是OFF，USE_MYMATH的值也是ON，只有在命令行显式说明 -DUSE_MYMATH=OFF，才能够把它关掉。

configure_file(TutorialConfig.h.in TutorialConfig.h)

if(USE_MYMATH) # add the MathFunctions library
  add_subdirectory(MathFunctions) ##以便使这个库能够被编译到
  list(APPEND EXTRA_LIBS MathFunctions) ##在EXTRA_LIBS列表中增加MathFunctions
  list(APPEND EXTRA_INCLUDES "${PROJECT_SOURCE_DIR}/MathFunctions") ##在EXTRA_INCLUDES列表中增加Mathfunctions那个文件夹路径
endif()

add_executable(Tutorial tutorial.cpp)

target_link_libraries(Tutorial PUBLIC ${EXTRA_LIBS}) ##把新库链接到到可执行程序当中

target_include_directories(Tutorial PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           ${EXTRA_INCLUDES} ##添加新的路径
                           )
```

`tutorial.cpp`:

```c++
#include <cmath>
#include <iostream>
#include <string>

#include "TutorialConfig.h"

#ifdef USE_MYMATH // should we include the MathFunctions header?
#  include "MathFunctions.h"
#endif

int main(int argc, char* argv[])
{
  if (argc < 2) {
    std::cout << argv[0] << " Version " << Tutorial_VERSION_MAJOR << "."
              << Tutorial_VERSION_MINOR << std::endl;
    std::cout << "Usage: " << argv[0] << " number" << std::endl;
    return 1;
  }

  const double inputValue = std::stod(argv[1]);

#ifdef USE_MYMATH    // which square root function should we use?
  const double outputValue = mysqrt(inputValue);
#else
  const double outputValue = sqrt(inputValue);
#endif

  std::cout << "The square root of " << inputValue << " is " << outputValue
            << std::endl;
  return 0;
}
```

`TutorialConfig.h.in`:

```c++
// the configured options and settings for Tutorial
#define Tutorial_VERSION_MAJOR @Tutorial_VERSION_MAJOR@
#define Tutorial_VERSION_MINOR @Tutorial_VERSION_MINOR@
#cmakedefine USE_MYMATH
```

在build文件夹下运行`cmake --build .`, 生成的TutorialConfig.h文件：

（使用命令`cmake ./ -DUSE_MYMATH=OFF`来得到USE_MYMATH = OFF的编译结果）

```c++
// the configured options and settings for Tutorial
#define Tutorial_VERSION_MAJOR 1
#define Tutorial_VERSION_MINOR 0
#define USE_MYMATH
```

运行`./Tutorial 1000`得到输出：

```
Computing sqrt of 1000 to be 500.5
Computing sqrt of 1000 to be 251.249
Computing sqrt of 1000 to be 127.615
Computing sqrt of 1000 to be 67.7253
Computing sqrt of 1000 to be 41.2454
Computing sqrt of 1000 to be 32.7453
Computing sqrt of 1000 to be 31.642
Computing sqrt of 1000 to be 31.6228
Computing sqrt of 1000 to be 31.6228
Computing sqrt of 1000 to be 31.6228
The square root of 1000 is 31.6228
```

 

#### Step3. 安装和测试

install的部分十分简单：对于MathFunctions，我们需要安装库(lib)和头文件(include); 对于Tutorial，我们需要安装可执行文件(bin)和库文件(include)

所以，现在主文件夹下的CMakeLists.txt长这样：

```cmake
cmake_minimum_required(VERSION 3.10)

project(Tutorial VERSION 1.0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(USE_MYMATH "Use tutorial provided math implementation" ON)

configure_file(TutorialConfig.h.in TutorialConfig.h)

if(USE_MYMATH)
  add_subdirectory(MathFunctions)
  list(APPEND EXTRA_LIBS MathFunctions)
  list(APPEND EXTRA_INCLUDES "${PROJECT_SOURCE_DIR}/MathFunctions")
endif()

add_executable(Tutorial tutorial.cpp)

target_link_libraries(Tutorial PUBLIC ${EXTRA_LIBS})

install(TARGETS Tutorial DESTINATION bin) ##把主程序Tutorial安装成可执行二进制文件
install(FILES "${PROJECT_BINARY_DIR}/TutorialConfig.h"
  DESTINATION include  ##把TutorialConfig.h 安装到头文件include中去。
  )

target_include_directories(Tutorial PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           ${EXTRA_INCLUDES}
                           )
```

MathFunction下的CMakeLists.txt长这样：

```cmake
add_library(MathFunctions mysqrt.cpp)

install(TARGETS MathFunctions DESTINATION lib) ##把MathFunctions安装到库(lib)中
install(FILES MathFunctions.h DESTINATION include) ##把MathFunctions.h安装到头文件中
```

之后执行:

```
cmake --install . --prefix "/path/to/installdir"
```

其中，--prefix 指的是**安装的路径**。

现在的安装路径中出现了新的文件夹：

```
- bin/
	- Tutorial* （可执行文件）
- include/
	- MathFunctions.h
	- TutorialConfig.h
- lib/
	- libMathFunctions.a （库文件）
```

现在，我们来**测试**我们写的sqrt函数的正确性。测试的方法是在主文件夹的CMakeLists.txt中增加测试代码：

```cmake
enable_testing() ##开始测试

add_test(NAME Runs COMMAND Tutorial 25) ##测试的名字叫做"Runs", 运行的命令行为`./Tutorial 25`.这是一个最基本的test，只是测试程序有没有崩溃、有没有正常地返回0。
  
add_test(NAME Usage COMMAND Tutorial) ##测试的名字叫做"Usage", 运行的命令为"./Tutorial"。这个时在测试没有传入参数的时候，我们的程序能否正确的打印出"Usage: " << argv[0] << " number".
set_tests_properties(Usage
    PROPERTIES PASS_REGULAR_EXPRESSION "Usage:.*number"
    ) ##正则表达式，看Usage测试的输出是否包含模式"Usage:.*number"
  
function(do_test target arg result) ##一个函数，名称叫做"do_test", 传入参数有三个：target, arg, result. 
    add_test(NAME Comp${arg} COMMAND ${target} ${arg}) ##测试的名字叫做"Comp+${arg}", 运行的命令行是${target} ${arg}（例如./Tutorial 25）
    MESSAGE( STATUS "this var key = ${Comp${arg}}.") ##打印
    set_tests_properties(Comp${arg} ##看看输出的结果是否包含${result}这个模式
      PROPERTIES PASS_REGULAR_EXPRESSION ${result}
      )
endfunction()
  
  # do a bunch of result based tests
do_test(Tutorial 4 "4 is 2")
do_test(Tutorial 9 "9 is 3")
do_test(Tutorial 5 "5 is 2.236")
do_test(Tutorial 7 "7 is 2.645")
do_test(Tutorial 25 "25 is 5")
do_test(Tutorial -25 "-25 is (nan|-nan|0)")
do_test(Tutorial 0.0001 "0.0001 is 0.01")
```

输出结果：

![img](https://pic1.zhimg.com/80/v2-ae296e2cf4cac7156e5e0db03b013eb9_720w.png)

测试全部通过啦！



## Cmake其他语法

- 打印：`MESSAGE( STATUS "this var key = ${arg}.")` 可以打印出`arg`变量的值。`STATUS`表示这句话只是普通输出，而不是ERROR.
- `${PROJECT_SOURCE_DIR}`指的是CMakeLists.txt所在的那个文件夹。