# 传参

当我们不确定将来要往函数中传入多少个参数，即可使用可变参数，用*args,**kwargs表示。

- \*args称之为Non-keyword Variable Arguments，**无关键字参数**；

- \*kwargs称之为keyword Variable Arguments，**有关键字参数**；

当函数中以列表或者元组的形式传参时，就要使用*args；

当传入字典形式的参数时，就要使用**kwargs。



## 0x01. *args: 无关键词参数

当位置参数与\*args一起使用时，先把参数分配给位置参数再将多余的参数以**元组**形式分配给\*args：

```python
def test(a,b,*args):
    print(a)
    print(b)
    print(args)

test("a","b","c","d","e","f")
```

输出：

```python
a
b
('c', 'd', 'e', 'f')
```



## 0x02. **kwargs: 有关键词参数

当传入函数的参数为**字典格式**时，使用**kwargs。

```python
def test(a,b,*args,**kwargs):
    print(a)
    print(b)
    print(args)
    print(kwargs)

test("a","b","c","d","e","f",name = "hanna",age = 22)
```

输出：

```python
a
b
('c', 'd', 'e', 'f')
{'name': 'hanna', 'age': 22}
```

