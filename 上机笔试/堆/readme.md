# 堆

```python
###python中堆的接口：
import heapq
heapq.heapify(x) ###就地建堆，x是一个list。O(n)
headq.heappush(x,9) #### insert元素
heapq.heappop(x)  ## del 最小元素
x[0] ##直接访问最小元素
print(heapq.nlargest(3,x)) ## 找到最大的k个元素
print(heapq.nsmallest(3,x))

```

- 注意python没有大根堆的实现，需要转成负数然后调用小根堆接口。
- python中的heapq本身不支持自定义比较函数，可以通过重写对象的__lt__方法的方式来实现自定义比较函数。

```
import heapq
class  P():
    def __init__(self,a,b):
        self.a = a
        self.b = b
    def __lt__(self, other):
        if self.b<other.b:
            return True
        else:
            return False
    def p(self):
        print(self.a,self.b)
 
a = P(3,1)
b = P(2,3)
c = P(10,0)
d = P(3,1)
 
h = []
heapq.heappush(h,a)
heapq.heappush(h,b)
heapq.heappush(h,c)
heapq.heappop(h).p()
```



二叉堆（Binary Heap）没什么神秘，性质比二叉搜索树 BST 还简单。其主要操作就两个，**sink**（下沉）和 **swim**（上浮），用以维护二叉堆的性质。

其主要应用有两个，首先是一种排序方法「**堆排序**」，第二是一种很有用的数据结构「**优先级队列**」。

本文就以实现优先级队列（Priority Queue）为例，描述一下二叉堆怎么运作的。

## 1. 二叉堆概览

首先，二叉堆和二叉树有啥关系呢，为什么人们总是把二叉堆画成一棵二叉树？

因为，二叉堆其实就是一种特殊的二叉树（**完全二叉树**），只不过存储在数组里。

![img](https://pic2.zhimg.com/80/v2-325ba1af6b2f8e67d19a98b7583820cd_1440w.jpg)      

对于一般的二叉树，我们操作节点的指针；而对于完全二叉树，我们可以直接通过**数组索引**访问一个节点的parent, left son, 和right son.

```cpp
// 父节点的索引
int parent(int root) {
    return root / 2;
}
// 左孩子的索引
int left(int root) {
    return root * 2;
}
// 右孩子的索引
int right(int root) {
    return root * 2 + 1;
}
```

画个图立即就能理解了，注意数组的**第一个索引 0 空着不用**：

![img](https://pic3.zhimg.com/80/v2-a864cb044ac85e485b58dbef6969810e_1440w.jpg)

![img](https://pic2.zhimg.com/80/v2-316c6df611d96b04ee0b215a3c3b1a7d_1440w.jpg)  （图：向量和完全二叉树的对应）

可以看到，把 arr[1] 作为整棵树的根的话，每个节点的**父节点**和**左右孩子**的索引都可以通过简单的运算得到，这就是二叉堆设计的一个巧妙之处。为了方便讲解，下面都会画的图都是二叉树结构，相信你能把树和数组对应起来。

二叉堆还分为最大堆和最小堆。最大堆的性质是：**每个节点都大于等于它的两个子节点**。 类似的，最小堆的性质是：**每个节点都小于等于它的子节点**。

两种堆核心思路都是一样的，本文以最大堆为例讲解。

对于一个最大堆，根据其性质，显然堆顶--也就是 arr[1] 一定是所有元素中最大的元素。

## 2. 优先级队列概览

**2.1 优先级队列的概念**

优先级队列这种数据结构有一个很有用的功能，你插入或者删除元素的时候，元素会自动把最大（或者最小）的元素排到**队首**，这底层的原理就是**二叉堆**的操作。

数据结构的功能无非增删改查，优先级队列有两个主要 API，分别是 insert--插入一个元素 和 delMax--删除最大元素（如果底层用最小堆，那么就是 delMin）。

**2.2 为什么要引入优先级队列**

![img](https://pic1.zhimg.com/80/v2-0a40d16919c8160ac6453ac65bc6a444_1440w.jpg)

对于AVL、Splay、Red-Black Tree, 三个接口只需要O(logn),但是BBST远远超出了优先级队列的需求…

**2.3 代码框架**

下面我们实现一个简化的优先级队列，先看下代码框架：

```cpp
class heap {
public:
    vector<int> PQ = {-1}; //用向量来构造堆，索引0不用
    int parent(int root){
        return root/2;
    }
    int left_child(int root){
        return root*2;
    }
    int right_child(int root){
        return root*2+1;
    }
    int GetMax(){
        return PQ[1];
    }
    void insert(int val){//插入元素
    }
    int DelMax(){//删除并返回队列中的最大元素
    }
    void swim(int k){//上浮第 k 个元素，以维护最大堆性质
    }
    void sink(int k){//下沉第 k 个元素，以维护最大堆性质
    }
};
```

### 2.3.1 swim（上浮）

上浮的原因是某些节点会违反堆序性，即有些节点比自己的父亲还要强，此时它会顶替父亲的位置。例如，在这个例子中插入了"42":

![img](https://pic4.zhimg.com/80/v2-90e9d875cd10aa0d8d375e45035bc3f3_1440w.jpg)

![img](https://pic1.zhimg.com/80/v2-30dadf5fb7a3af4e0abd13779f9ee6f4_1440w.jpg)

![img](https://pic1.zhimg.com/80/v2-30dadf5fb7a3af4e0abd13779f9ee6f4_1440w.jpg)

![img](https://pic1.zhimg.com/80/v2-30dadf5fb7a3af4e0abd13779f9ee6f4_1440w.jpg)

![img](https://pic1.zhimg.com/80/v2-51c6907b792886820fcd0556bf21ed94_1440w.jpg)

![img](https://pic4.zhimg.com/80/v2-994cde5d25555c36c1d8ef2802143807_1440w.jpg)

```cpp
    void swim(int k){//上浮第 k 个元素，以维护最大堆性质
        while(k>1){ //尚未到根
            if(PQ[k] > PQ[parent(k)]) {//且比自己父亲强
                swap(PQ[k], PQ[parent(k)]);
                k = parent(k);
            }
            else //满足堆序性
                break;
        }
    }
```

### 2.3.2 insert

先将词条e作为末尾元素接入向量，之后**上浮**之。

![img](https://pic3.zhimg.com/80/v2-355addfc2a147c91b59e4245ac592d9a_1440w.jpg)

![img](https://pic4.zhimg.com/80/v2-cc5a762bb1ca3d01f70dff35f2f25f6b_1440w.jpg)

```cpp
    void insert(int val){//插入元素
        PQ.push_back(val);
        swim(PQ.size()-1);
    }
```

### 2.3.3 sink

下沉比上浮略微复杂一点，因为上浮某个节点 A，只需要A和其父节点比较大小即可；但是下沉某个节点 A，需要 A 和**其两个子节点**比较大小，如果 A 不是最大的就需要调整位置，要把较大的那个子节点和 A 交换。

例如下例，就是下滤"15":

![img](https://pic1.zhimg.com/80/v2-2acc82abc289ff5951ddbb173faa86ac_1440w.jpg)

![img](https://pic1.zhimg.com/80/v2-26d9100f539df81042ae0482a5e700f0_1440w.jpg)

![img](https://pic4.zhimg.com/80/v2-f60e49d3eab55994c28f27fb9b4eb903_1440w.jpg)

```cpp
    void sink(int k){//下沉第 k 个元素，以维护最大堆性质
        while(left_child(k) < PQ.size()){//不是叶子节点
            int max_child = max(PQ[left_child(k)],PQ[right_child(k)]);//找到最大的孩子
            if(PQ[k] > max_child) break; //堪为父亲！
            if(max_child == PQ[left_child(k)]){//最大的是左孩子
                swap(PQ[k],PQ[left_child(k)]);
                k = left_child(k);//下沉
            }
            else {//最大的是右孩子
                swap(PQ[k], PQ[right_child(k)]);
                k = right_child(k);
            }
        }
    }
```

### 2.3.4 Del_Max

删除首个节点（最大的节点），并将最末的元素和根节点对换，然后不断下沉。

![img](https://pic4.zhimg.com/80/v2-1e372181c3cde6b48002925c31bc490b_1440w.jpg)

![img](https://pic1.zhimg.com/80/v2-7f1088d2be8e3c7dd6fa78968b51b7e4_1440w.jpg)

```cpp
    int DelMax(){//删除并返回队列中的最大元素
        int max = PQ[1];
        PQ[1] = PQ[PQ.size()-1];
        PQ.pop_back();
        sink(1);
        return max;
    }
```

### 3. 堆排序

```cpp
vector<int> heap_sort(vector<int> vec){
    //建堆
    heap H;
    for(int i = 0;i<vec.size();i++) { //建堆，O(n)
        H.insert(vec[i]);
    }
    vector<int> ans;
    for(int i = 0;i<vec.size();i++) {
        ans.push_back(H.DelMax());//每次取最大
    }
    return ans;
}
```

### 4. 总结

二叉堆就是一种完全二叉树，所以适合存储在数组中，而且二叉堆拥有一些特殊性质。

二叉堆的操作很简单，主要就是上浮和下沉，来维护堆的性质（堆有序），核心代码也就十行。

优先级队列是基于二叉堆实现的，主要操作是插入和删除。插入是先插到最后，然后上浮到正确位置；删除是调换位置后再删除，然后下沉到正确位置。

基于优先级队列，可以实现时间复杂度O(nlogn)的排序算法。